import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dim and negate the first one.

    ``cat([-x[..., d/2:], x[..., :d/2]], dim=-1)``. This is the conjugate
    multiplier of the HF-style rotation: ``x*cos + rotate_half(x)*sin``.
    """
    d = x.size(-1)
    return torch.cat([-x[..., d // 2 :], x[..., : d // 2]], dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embedding (RoPE) from "RoFormer: Enhanced Transformer
    with Rotary Position Embedding" (Su et al., 2021, arXiv:2104.09864).

    Positions are injected multiplicatively into the attention scores by
    rotating the query/key vectors in per-dimension-pair 2D planes. A token
    at position ``m`` is rotated by angle ``m * theta_i``, so the dot product
    ``q_m @ k_n`` picks up the relative angle ``(m - n) * theta_i`` and the
    attention score depends on relative position only. Unlike additive
    embeddings this needs no lookup table growth for longer sequences and
    plays nicely with KV caching (only the new token gets rotated).

    Uses the HF/LLaMA convention: the last dim is split in halves and the
    ``i``-th pair is ``(i, i + head_size/2)`` with frequency
    ``theta_i = base ** (-2i / head_size)``.

    ``cos``/``sin`` tables are precomputed as non-trainable float32 buffers up
    to ``max_seq_len``. Sequences longer than that recompute the cache on the
    fly and cache the extension, so there is no hard length cap.

    Apply after splitting the Q/K projections into heads: expects
    ``(batch, heads, seq_len, head_size)``. For cross-attention rotate only
    the query via :meth:`rotate`.

    Parameters
    ----------
    head_size : int
        Per-head hidden dim of Q/K. Must be even, because positions rotate
        2D pairs.
    base : float, default=10000.0
        Base of the geometric frequency schedule ``theta_i = base**(-2i/d)``.
    max_seq_len : int, default=4096
        Cache length of precomputed ``cos``/``sin`` tables. Longer inputs
        recompute and extend the cache on the fly.
    """

    def __init__(self, head_size: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        if head_size % 2 != 0:
            raise ValueError(f"head_size ({head_size}) must be even, got odd")

        self.head_size = head_size
        self.base = base
        self.max_seq_len = max_seq_len

        self.register_buffer("cos_cache", torch.zeros(max_seq_len, head_size))
        self.register_buffer("sin_cache", torch.zeros(max_seq_len, head_size))
        self._extend_cache(max_seq_len)

    def _theta(self, length: int, device: torch.device) -> torch.Tensor:
        """Geometric frequency schedule ``theta_i`` for ``head_size//2`` pairs."""
        i = torch.arange(self.head_size // 2, device=device, dtype=torch.float32)
        return torch.pow(self.base, -2.0 * i / self.head_size)

    def _extend_cache(self, length: int) -> None:
        """Recompute cos/sin tables up to ``length`` (float32, then recast)."""
        dtype = self.cos_cache.dtype
        device = self.cos_cache.device
        theta = self._theta(length, device)
        pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        freqs = pos * theta  # (length, head_size/2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.cos_cache = torch.cat([cos, cos], dim=-1).to(dtype)
        self.sin_cache = torch.cat([sin, sin], dim=-1).to(dtype)

    def _tables(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for the given positions, extended beyond ``max_seq_len``.

        Eager path indexes the precomputed cache (extending it on demand).
        Under ``torch.compile`` the tables are computed on the fly instead:
        the cache path needs a host sync (``max().item()``) which breaks the
        captured graph — same approach as HF's on-the-fly rotary.
        """
        if torch.compiler.is_compiling():
            i = torch.arange(self.head_size // 2, device=positions.device, dtype=torch.float32)
            theta = torch.pow(self.base, -2.0 * i / self.head_size)
            freqs = positions.to(torch.float32).unsqueeze(-1) * theta.unsqueeze(0)
            emb = torch.cat([freqs, freqs], dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)
        max_pos = positions.max().item()
        if max_pos >= self.cos_cache.size(0):
            self._extend_cache(max_pos + 1)
        return self.cos_cache[positions].to(dtype), self.sin_cache[positions].to(dtype)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Rotate query/key vectors by their positions.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, heads, seq_len, head_size)``. ``head_size`` must equal
            the module's.
        positions : torch.Tensor | None, default=None
            ``(seq_len,)`` positions for each token (e.g. KV-cache offsets
            during decoding). ``None`` means ``0..seq_len-1``.

        Returns
        -------
        torch.Tensor
            Rotated vectors, same shape as ``x``.
        """
        if x.size(-1) != self.head_size:
            raise ValueError(
                f"last dim ({x.size(-1)}) must equal head_size ({self.head_size})"
            )
        seq_len = x.size(-2)
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        cos, sin = self._tables(positions, x.dtype)
        return x * cos + _rotate_half(x) * sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_positions: torch.Tensor | None = None,
        k_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q`` and ``k`` in place of additive position embeddings.

        Parameters
        ----------
        q : torch.Tensor
            ``(batch, heads, seq_len, head_size)``.
        k : torch.Tensor
            ``(batch, heads, seq_len, head_size)``. May be ``None`` when only
            the query needs rotation (cross-attention).
        q_positions : torch.Tensor | None, default=None
            Per-token positions of ``q``, default ``0..seq_len-1``.
        k_positions : torch.Tensor | None, default=None
            Per-token positions of ``k``, default ``0..seq_len-1``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Rotated ``(q, k)``. When ``k is None``, ``k`` is returned as is.
        """
        q = self.rotate(q, q_positions)
        if k is not None:
            k = self.rotate(k, k_positions)
        return q, k
