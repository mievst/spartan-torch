from collections.abc import Callable

import torch
from torch import nn


class QKVNorm(nn.Module):
    """Per-head normalization of the query/key/value vectors.

    Normalizes each of ``q``, ``k``, ``v`` individually after the head split,
    with the norm applied over the per-head axis
    (``(batch, heads, seq_len, head_size)`` -> normalize ``head_size``).
    The three norms are separate modules with independent affine parameters.

    The core of HybridNorm ("HybridNorm: Towards Stable and Efficient
    Transformer Training via Hybrid Normalization", Zhuo et al., 2025,
    arXiv:2503.04598): normalizing the projected QKV decouples the gradient
    flow between the attention weight matrices and, combined with Post-Norm
    in the FFN, stabilizes deep transformer training while matching or
    beating Pre-Norm performance. The same per-head stage pattern covers
    QK-norm (Gemma) via ``v_norm=False`` and any other per-head transform
    placed after the projections.

    Parameters
    ----------
    head_size : int
        Per-head hidden dim. Each norm normalizes the last axis of a
        ``(batch, heads, seq_len, head_size)`` tensor.
    norm_layer : Callable[[int], nn.Module], default=nn.LayerNorm
        Normalization factory called with ``head_size`` (e.g. RMSNorm to
        reproduce the paper exactly).
    v_norm : bool, default=True
        Also normalize the value vectors. ``False`` yields QK-norm (Gemma
        style), which leaves ``v`` untouched.
    """

    def __init__(
        self,
        head_size: int,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        v_norm: bool = True,
    ):
        super().__init__()
        self.head_size = head_size
        self.v_norm = v_norm
        self.norm_q = norm_layer(head_size)
        self.norm_k = norm_layer(head_size)
        self.norm_v = norm_layer(head_size) if v_norm else nn.Identity()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize ``q``, ``k`` and (optionally) ``v``.

        Parameters
        ----------
        q, k, v : torch.Tensor
            ``(batch, heads, seq_len, head_size)`` post-split projections.
            ``k``/``v`` may have ``num_kv_heads != num_heads`` (GQA) — only
            the ``head_size`` axis matters here.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Normalized ``(q, k, v)``, same shapes.
        """
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        return q, k, v
