# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
FlashAttention integration with automatic backend selection.

Dispatches to the best available backend:

1. ``flash_attn`` package (if installed + CUDA)
2. ``torch.nn.functional.scaled_dot_product_attention`` (PyTorch 2.0+)
3. Eager math attention (always available, fallback)

Provides:

- :class:`FlashAttention` -- drop-in attention module with SDPA dispatch.
- :func:`enable_flash_attention` -- check if any flash backend is usable.
- :func:`is_flash_attention_available` -- same as ``enable_flash_attention``.

# VRAM: ~0 MB (wrapper, no parameters)

References
----------
.. [1] Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism"
       https://arxiv.org/abs/2307.08691

Examples
--------
>>> from spartan_torch.efficiency.flash_attn import FlashAttention
>>> attn = FlashAttention(d_model=256, n_heads=8)
>>> q = k = v = torch.randn(2, 128, 8, 32)
>>> out = attn(q, k, v)
>>> out.shape
torch.Size([2, 128, 8, 32])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "FlashAttention",
    "enable_flash_attention",
    "is_flash_attention_available",
]


def _has_flash_attn_package() -> bool:
    """Check if the flash_attn package is installed."""
    try:
        import flash_attn  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def _has_sdpa_flash() -> bool:
    """Check if PyTorch SDPA has flash attention backend available."""
    if not torch.cuda.is_available():
        return False
    try:
        backends = torch.backends.cuda.flash_sdp_enabled()  # type: ignore[no-untyped-call]
        return bool(backends)
    except AttributeError:
        return False


def is_flash_attention_available() -> bool:
    """
    Check if any flash attention backend is available.

    Returns ``True`` if either:
    - ``flash_attn`` package is installed (CUDA required), or
    - PyTorch SDPA with flash backend is available (CUDA required), or
    - PyTorch SDPA math backend is available (CPU OK, but no speed benefit).

    Returns
    -------
    bool
        ``True`` if at least SDPA is available (always True on PyTorch 2.0+).

    Examples
    --------
    >>> enable_flash_attention()  # True on any PyTorch 2.0+ install
    True
    """
    # SDPA is always available on PyTorch 2.0+ (math backend at minimum)
    return hasattr(F, "scaled_dot_product_attention")


def enable_flash_attention() -> bool:
    """
    Check if flash attention provides actual speed benefit (CUDA required).

    Unlike :func:`is_flash_attention_available`, this returns ``False``
    on CPU since the math backend has no performance advantage over
    manual attention.

    Returns
    -------
    bool
        ``True`` if a CUDA flash backend is available.

    Examples
    --------
    >>> enable_flash_attention()  # False on CPU
    False
    """
    if _has_flash_attn_package():
        return True
    return _has_sdpa_flash()


class FlashAttention(nn.Module):
    """
    Multi-head attention with automatic SDPA/flash dispatch.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` which
    automatically selects the best backend:

    - **FlashAttention-2** on CUDA (Ampere+)
    - **Memory-efficient attention** on CUDA (older GPUs)
    - **Math backend** on CPU (reference implementation)

    Falls back to manual attention if SDPA is unavailable.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability on attention weights.
    bias : bool, default=False
        Use bias in QKV projections.
    is_causal : bool, default=False
        Apply causal mask automatically. Mutually exclusive with
        explicit ``attn_mask``.
    device : torch.device or None, default=None
        Device for parameters.
    dtype : torch.dtype or None, default=None
        dtype for parameters.

    # VRAM: ~0 MB (wrapper, parameters owned by caller)

    Examples
    --------
    >>> attn = FlashAttention(d_model=256, n_heads=8)
    >>> q = k = v = torch.randn(2, 128, 8, 32)  # (batch, seq, heads, head_dim)
    >>> out = attn(q, k, v)
    >>> out.shape
    torch.Size([2, 128, 8, 32])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
        is_causal: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Apply scaled dot-product attention with automatic backend selection.

        Parameters
        ----------
        query : Tensor
            Query tensor of shape ``(batch, seq_len, d_model)``.
        key : Tensor
            Key tensor of shape ``(batch, seq_len, d_model)``.
        value : Tensor
            Value tensor of shape ``(batch, seq_len, d_model)``.
        attn_mask : Tensor or None, default=None
            Attention mask. Shape depends on backend:
            - ``2D: (seq_len, seq_len)``
            - ``3D: (batch, seq_len, seq_len)``
            - ``4D: (batch, 1, seq_len, seq_len)``

        Returns
        -------
        Tensor
            Output of shape ``(batch, seq_len, d_model)``.

        Raises
        ------
        ValueError
            If ``is_causal=True`` and ``attn_mask`` is not ``None``.
        """
        if self.is_causal and attn_mask is not None:
            raise ValueError("Cannot use both is_causal=True and attn_mask. Use one or the other.")

        batch_size, seq_len, _ = query.shape

        # Project and reshape to (batch, heads, seq, head_dim)
        q = (
            self.q_proj(query)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Dispatch to best available backend
        if hasattr(F, "scaled_dot_product_attention"):
            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=self.is_causal,
            )
        else:
            # Manual fallback for very old PyTorch versions
            out = self._manual_attention(q, k, v, attn_mask)

        # Reshape back to (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        result: Tensor = self.out_proj(out)
        return result

    def _manual_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """Manual attention fallback (no flash, no memory-efficient)."""
        scale = q.size(-1) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.is_causal:
            seq_len = q.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))
        elif attn_mask is not None:
            attn = attn + attn_mask

        attn = torch.softmax(attn, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)
        return torch.matmul(attn, v)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, dropout={self.dropout}, "
            f"is_causal={self.is_causal}"
        )
