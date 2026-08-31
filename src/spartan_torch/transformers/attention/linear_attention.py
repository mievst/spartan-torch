import torch
from torch import nn
import torch.nn.functional as F

from .mha import _split_heads


class LinearTransformerAttention(nn.Module):
    """Linear (kernelized) attention — the Linear Transformer core.

    From "Transformers are RNNs: Fast Autoregressive Transformers with Linear
    Attention" (Katharopoulos et al., 2020, arXiv:2006.16236). The softmax is
    replaced with a dot product of kernel feature maps
    :math:`\\text{sim}(q, k) = \\phi(q)^\\top \\phi(k)` with the paper's
    :math:`\\phi(x) = \\text{elu}(x) + 1`. Matrix associativity lets the
    product be regrouped:

    .. math::

        V' = \\frac{\\phi(Q)\\,(\\phi(K)^\\top V)}{\\phi(Q)\\,(\\phi(K)^\\top 1)}

    so the :math:`n \\times n` score tensor never materializes and time/memory
    drop from :math:`O(n^2)` to :math:`O(n d^2)` for a fixed per-head width
    ``head_size`` (the denominator is the normalizer over key positions).

    Causal mode (``is_causal=True``) accumulates prefix sums
    :math:`S_i = \\sum_{j \\le i} \\phi(K_j) V_j^\\top` and
    :math:`Z_i = \\sum_{j \\le i} \\phi(K_j)` — the exact training-time view of
    the paper's recurrent (RNN) formulation. This is what makes the mechanism
    decoder-friendly, unlike Linformer.

    Numeric / practical notes:

    * ``elu(x) + 1 > 0`` elementwise, so the normalizer is always positive.
    * The vectorized causal form materializes ``(batch, heads, seq_len,
      head_size, head_size)`` (the paper uses a custom backward to stay at
      :math:`O(d^2)`); keep ``head_size`` small for long causal sequences.
    * The kernel does not support per-position masking: ``mask`` or
      ``past_key_value`` raise ``NotImplementedError``. Padding must be handled
      by packing. Incremental recurrent decoding (constant memory per step)
      is not wired into the ``(key, value)`` cache contract — see
      :class:`MultiHeadAttention`.

    Tensors use the ``(batch, seq, embed)`` layout, mirroring
    :class:`MultiHeadAttention`. Dependency-free and ``torch.compile``-friendly.

    Parameters
    ----------
    in_size : int
        Embedding size of key/value inputs.
    head_size : int
        Per-head hidden dim of Q, K, V.
    num_heads : int
        Number of heads.
    out_size : int
        Output embedding size.
    query_in_size : int | None, default=None
        Embedding size of the query input. ``None`` falls back to
        ``in_size`` (cross-attention support).
    attn_p : float, default=0.0
        Dropout probability on the feature maps :math:`\\phi(Q)` and
        :math:`\\phi(K)` in training mode (per-token feature dropout —
        linear attention never materializes per-query attention weights).
    is_causal : bool, default=False
        Use the causal (prefix-sum) form. Requires ``query_seq_len ==
        ``key_seq_len``.
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        num_heads: int,
        out_size: int,
        query_in_size: int | None = None,
        attn_p: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.query_in_size = in_size if query_in_size is None else query_in_size
        self.attn_p = attn_p
        self.is_causal = is_causal

        self.query_matrix = nn.Linear(self.query_in_size, head_size * num_heads, bias=False)
        self.key_matrix = nn.Linear(self.in_size, head_size * num_heads, bias=False)
        self.value_matrix = nn.Linear(self.in_size, head_size * num_heads, bias=False)
        self.out = nn.Linear(head_size * num_heads, out_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply linear attention.

        Parameters
        ----------
        query : torch.Tensor
            ``(batch, query_seq_len, query_in_size)``.
        key : torch.Tensor
            ``(batch, key_seq_len, in_size)``.
        value : torch.Tensor
            ``(batch, key_seq_len, in_size)``.
        mask : torch.Tensor | None, default=None
            Unsupported — the kernel sums over all key positions, so
            per-position masking cannot be applied. Raises
            ``NotImplementedError``.
        past_key_value : tuple[torch.Tensor, torch.Tensor] | None, default=None
            Unsupported — the recurrent state is the accumulated
            ``(S, Z)`` pair, not a ``(key, value)`` cache. Raises
            ``NotImplementedError``.

        Returns
        -------
        torch.Tensor
            Attention output ``(batch, query_seq_len, out_size)``.
        """
        batch = key.size(0)
        k_len = key.size(1)
        q_len = query.size(1)
        if mask is not None or past_key_value is not None:
            raise NotImplementedError(
                "linear attention supports no per-position masks or KV cache; "
                "use is_causal=True for causal masking"
            )
        if self.is_causal and q_len != k_len:
            raise ValueError(f"causal linear attention requires query_seq_len == key_seq_len, got {q_len} != {k_len}")

        q = _split_heads(self.query_matrix(query), batch, self.num_heads, self.head_size, q_len)
        k = _split_heads(self.key_matrix(key), batch, self.num_heads, self.head_size, k_len)
        v = _split_heads(self.value_matrix(value), batch, self.num_heads, self.head_size, k_len)

        d = self.head_size
        # phi(x) = elu(x) + 1  (elementwise, per the paper)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        if self.training and self.attn_p > 0.0:
            q = F.dropout(q, self.attn_p, training=True)
            k = F.dropout(k, self.attn_p, training=True)

        if self.is_causal:
            # S_i = sum_{j<=i} phi(K_j) V_j^T  ->  (b, h, n, d, d)
            kv = torch.einsum("bhjd,bhje->bhjde", k, v).cumsum(dim=2)
            z = k.cumsum(dim=2)  # (b, h, n, d)
            # out_i = (phi(Q_i) @ S_i) / (phi(Q_i) @ Z_i)
            numerator = torch.einsum("bhld,bhlde->bhle", q, kv)
            denominator = torch.einsum("bhld,bhld->bhl", q, z).unsqueeze(-1)
            context = numerator / denominator
        else:
            # KV = sum_j phi(K_j) V_j^T  ->  (b, h, d, d)
            kv = torch.einsum("bhjd,bhje->bhde", k, v)
            z = k.sum(dim=2)  # (b, h, d)
            numerator = torch.einsum("bhld,bhde->bhle", q, kv)
            denominator = torch.einsum("bhld,bhd->bhl", q, z).unsqueeze(-1)
            context = numerator / denominator

        return self.out(context.transpose(1, 2).reshape(batch, q_len, self.num_heads * self.head_size))
