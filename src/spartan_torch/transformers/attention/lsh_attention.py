import math

import torch
from torch import nn
import torch.nn.functional as F

from .mha import _split_heads


class ReformerAttention(nn.Module):
    """Locality-sensitive-hashing (LSH) attention — the Reformer core.

    From "Reformer: The Efficient Transformer" (Kitaev et al., 2020,
    arXiv:2001.04451). Keys/values are grouped into buckets by an LSH of
    ``h(x) = argmax([x R; -x R])`` with a fixed random rotation :math:`R`,
    sorted by bucket, and chunked; every query attends only to the keys of its
    own chunk and the preceding one. The score tensor per query is bounded by
    ``2 * bucket_size``, so time/memory drop from :math:`O(n^2)` to
    :math:`O(n \\log n)` (sorting dominates).

    The paper's hashing relies on **shared query/key** projections with
    normalized keys — ``k_j = q_j / ||q_j||`` — so a query always hashes into
    the same bucket as its own key (guaranteed self-attention) and attention
    similarity matches hash locality. ``shared_qk=True`` (default) reproduces
    this; ``shared_qk=False`` uses a separate key projection at the cost of
    hash consistency.

    Multi-round hashing (``n_hashes`` rotations, default 8) lowers the
    probability of two similar vectors being split across buckets. Rounds are
    combined per query with a softmax over each round's log-normalizer
    (rounds where the query found a confident match are weighted higher).

    Important properties:

    * Exact causality: the causal mask compares *original* sequence positions
      (``key_pos <= query_pos``), so no future token ever leaks in — unlike
      the bidirectional-only Linformer.
    * The rotation :math:`R` is a non-trainable buffer fixed at
      initialization. This keeps forward deterministic and, critically, makes
      the hash/sort reproducible when a :class:`ReformerBlock` recomputes its
      forward inside ``torch.utils.checkpoint`` (fresh random rotations per
      call would break the recomputed backward).
    * The hash/sort are index operations — gradients flow only through the
      attention scores within windows.
    * Self-attention only: ``query``, ``key``, ``value`` must share the same
      sequence length (the bucket geometry assumes shared positions). ``mask``
      or ``past_key_value`` raise ``NotImplementedError``; padding must be
      handled by packing.
    * ``n`` is padded internally to a multiple of ``bucket_size`` (padding is
      excluded from attention and from the output).

    Tensors use the ``(batch, seq, embed)`` layout, mirroring
    :class:`MultiHeadAttention`.

    Parameters
    ----------
    in_size : int
        Embedding size of query/key/value inputs.
    head_size : int
        Per-head hidden dim of Q, K, V.
    num_heads : int
        Number of heads.
    out_size : int
        Output embedding size.
    shared_qk : bool, default=True
        Share the query/key projection (the paper's setting): keys are the
        L2-normalized queries. ``False`` uses a separate key projection.
    n_hashes : int, default=8
        Number of independent hash rotations (multi-round LSH).
    bucket_size : int, default=64
        Chunk length the sorted sequence is split into; each query attends to
        its chunk and the preceding chunk (``2 * bucket_size`` candidates).
    n_buckets : int, default=128
        Number of hash buckets (must be even; ``[xR; -xR]`` doubles the
        rotation width). More buckets → finer grouping, larger sort key
        spread.
    attn_p : float, default=0.0
        Dropout probability on attention weights in training mode.
    is_causal : bool, default=True
        Apply the causal mask (key original position ``<=`` query original
        position).
    query_in_size : int | None, default=None
        Accepted for interface parity with the other attention layers; with
        ``shared_qk`` the query input also feeds the keys, so ``None`` is
        required in practice.
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        num_heads: int,
        out_size: int,
        shared_qk: bool = True,
        n_hashes: int = 8,
        bucket_size: int = 64,
        n_buckets: int = 128,
        attn_p: float = 0.0,
        is_causal: bool = True,
        query_in_size: int | None = None,
    ):
        super().__init__()
        if n_buckets % 2 != 0:
            raise ValueError(f"n_buckets must be even, got {n_buckets}")
        if n_hashes < 1 or bucket_size < 1:
            raise ValueError("n_hashes and bucket_size must be positive")

        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.shared_qk = shared_qk
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self.attn_p = attn_p
        self.is_causal = is_causal
        self.query_in_size = in_size if query_in_size is None else query_in_size

        # Fixed random rotation shared across heads (paper: single random
        # matrix; one per hash round). Non-trainable buffer — deterministic
        # and safe under torch.utils.checkpoint recomputation.
        self.register_buffer(
            "R",
            torch.randn(n_hashes, head_size, n_buckets // 2),
        )

        self.query_matrix = nn.Linear(self.query_in_size, head_size * num_heads, bias=False)
        self.key_matrix = nn.Linear(in_size, head_size * num_heads, bias=False)
        self.value_matrix = nn.Linear(in_size, head_size * num_heads, bias=False)
        self.out = nn.Linear(head_size * num_heads, out_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply LSH attention.

        Parameters
        ----------
        query : torch.Tensor
            ``(batch, seq_len, query_in_size)``.
        key : torch.Tensor
            ``(batch, seq_len, in_size)``. Must share ``seq_len`` with
            ``query`` (self-attention only).
        value : torch.Tensor
            ``(batch, seq_len, in_size)``.
        mask : torch.Tensor | None, default=None
            Unsupported — LSH buckets don't admit arbitrary masks. Raises
            ``NotImplementedError``.
        past_key_value : tuple[torch.Tensor, torch.Tensor] | None, default=None
            Unsupported — the LSH window has no position-invariant cache. Raises
            ``NotImplementedError``.

        Returns
        -------
        torch.Tensor
            Attention output ``(batch, seq_len, out_size)``.
        """
        batch = query.size(0)
        n = query.size(1)
        if key.size(1) != n or value.size(1) != n:
            raise ValueError(
                f"ReformerAttention is self-attention; got query_seq_len {n} != "
                f"key/value seq_len {key.size(1)}"
            )
        if mask is not None or past_key_value is not None:
            raise NotImplementedError(
                "ReformerAttention supports no masks or KV cache; use is_causal=True "
                "for causal masking"
            )

        if self.shared_qk:
            q = self.query_matrix(query)
            v = self.value_matrix(value)
            k = q
        else:
            q = self.query_matrix(query)
            k = self.key_matrix(key)
            v = self.value_matrix(value)

        q = _split_heads(q, batch, self.num_heads, self.head_size, n)
        k = _split_heads(k, batch, self.num_heads, self.head_size, n)
        v = _split_heads(v, batch, self.num_heads, self.head_size, n)

        n_pad = int(math.ceil(n / self.bucket_size)) * self.bucket_size
        total = self.n_hashes * n_pad
        chunk_count = total // self.bucket_size

        # --- hash (on the key-side vectors; identical to the query vectors
        # under shared_qk, which is what makes the bucketing consistent) ---
        rot = torch.einsum("bhpd,adg->bhpag", k, self.R)  # (b,h,n,n_hashes,b)
        rot = torch.cat([rot, -rot], dim=-1)  # (b,h,n,n_hashes,n_buckets)
        bucket = rot.argmax(dim=-1)  # (b,h,n,n_hashes)
        bucket = bucket + torch.arange(self.n_hashes, device=rot.device) * self.n_buckets

        # round-major layout: flat index = r * n_pad + pos
        bucket = bucket.permute(0, 1, 3, 2)  # (b,h,n_hashes,n)
        pad_bucket = torch.full(
            (batch, self.num_heads, self.n_hashes, n_pad - n),
            self.n_hashes * self.n_buckets + 1,
            dtype=bucket.dtype,
            device=bucket.device,
        )
        bucket = torch.cat([bucket, pad_bucket], dim=-1).reshape(batch, self.num_heads, -1)
        pos_flat = torch.arange(n_pad, device=query.device).repeat(self.n_hashes)
        pos_flat = pos_flat.unsqueeze(0).unsqueeze(0).expand(batch, self.num_heads, -1)

        sort_key = bucket.long() * n_pad + pos_flat  # (b,h,total)
        sorted_idx = sort_key.argsort(dim=-1, stable=True)  # (b,h,total)

        def gather(x: torch.Tensor) -> torch.Tensor:
            return x.gather(2, sorted_idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))

        pad = (0, 0, 0, n_pad - n)
        q_flat = F.pad(q, pad).repeat(1, 1, self.n_hashes, 1)
        k_flat = F.pad(k, pad).repeat(1, 1, self.n_hashes, 1)
        v_flat = F.pad(v, pad).repeat(1, 1, self.n_hashes, 1)

        bq = gather(q_flat).reshape(batch, self.num_heads, chunk_count, self.bucket_size, self.head_size)
        bk = gather(k_flat).reshape(batch, self.num_heads, chunk_count, self.bucket_size, self.head_size)
        bv = gather(v_flat).reshape(batch, self.num_heads, chunk_count, self.bucket_size, self.head_size)

        spos = pos_flat.gather(2, sorted_idx)  # (b,h,total) original positions
        bq_t = spos.reshape(batch, self.num_heads, chunk_count, self.bucket_size)
        b_valid = (spos < n).reshape(batch, self.num_heads, chunk_count, self.bucket_size)

        def look_one_back(x: torch.Tensor) -> torch.Tensor:
            prev = torch.cat([x[:, :, -1:], x[:, :, :-1]], dim=2)
            return torch.cat([x, prev], dim=3)

        norm = bk.norm(p=2, dim=-1, keepdim=True)  # unit-norm keys (shared-QK)
        bk = torch.where(norm > 0, bk / norm, bk)  # keep zero rows zero: fp16 has
        # no clamp eps, F.normalize would turn padded (all-zero) keys into NaN
        bk_w = look_one_back(bk)
        bv_w = look_one_back(bv)
        bk_t = look_one_back(bq_t)
        bk_valid = look_one_back(b_valid)

        dots = torch.einsum("bhcde,bhcfe->bhcdf", bq, bk_w) / math.sqrt(self.head_size)
        masked = -torch.finfo(dots.dtype).max

        window = b_valid.unsqueeze(-1) & bk_valid.unsqueeze(-2)  # (b,h,C,B,2B)
        dots = dots + (~window).to(dots.dtype) * masked
        if self.is_causal:
            causal = bk_t.unsqueeze(-2) <= bq_t.unsqueeze(-1)
            dots = dots + (~causal).to(dots.dtype) * masked

        weights = F.softmax(dots, dim=-1)
        if self.training and self.attn_p > 0.0:
            weights = F.dropout(weights, self.attn_p, training=True)
        bo = torch.einsum("bhcdf,bhcfe->bhcde", weights, bv_w)  # (b,h,C,B,d)
        slogits = torch.logsumexp(dots, dim=-1)  # (b,h,C,B), pre-dropout

        # --- unsort back to round-major layout ---
        bo = bo.reshape(batch, self.num_heads, -1, self.head_size)
        slogits = slogits.reshape(batch, self.num_heads, -1)
        undo = sorted_idx.argsort(dim=-1)
        o = bo.gather(2, undo.unsqueeze(-1).expand(-1, -1, -1, self.head_size))
        logits = slogits.gather(2, undo)

        o = o.reshape(batch, self.num_heads, self.n_hashes, n_pad, self.head_size)
        logits = logits.reshape(batch, self.num_heads, self.n_hashes, n_pad)
        o = o[..., :n, :]
        logits = logits[..., :n]

        # --- combine rounds: softmax over each round's log-normalizer ---
        probs = F.softmax(logits, dim=2)  # (b,h,n_hashes,n)
        context = torch.einsum("bhrne,bhrn->bhne", o, probs)  # (b,h,n,d)

        return self.out(context.transpose(1, 2).reshape(batch, n, self.num_heads * self.head_size))
