from collections.abc import Callable

import torch
from torch import nn

from ..attention import MultiHeadAttention


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block.

    Queries come from ``x``, keys/values from ``memory``. Layered as
    ``x + Attn(LN_q(x), LN_mv(memory), LN_mv(memory))`` (pre-LN). A building
    block for encoder-decoder decoders and any retrieval-style architecture
    where one sequence reads another.

    Parameters
    ----------
    query_size : int
        Embedding size of the query input ``x``.
    memory_size : int | None, default=None
        Embedding size of ``memory``. ``None`` falls back to ``query_size``.
    head_size : int
        Per-head hidden dim of Q, K, V.
    num_heads : int
        Number of query heads. Must be a multiple of ``num_kv_heads``.
    out_size : int
        Output embedding size.
    num_kv_heads : int | None, default=None
        Number of shared key/value heads. ``None`` means plain MHA
        (``num_kv_heads == num_heads``).
    norm_layer : Callable[[int], nn.Module], default=nn.LayerNorm
        Normalization factory called with the embedding size.
    attn_p : float, default=0.0
        Dropout probability on attention weights (passed to MHA).
    dropout_p : float, default=0.0
        Dropout probability on the residual branch.
    use_sdpa : bool, default=False
        Route attention through ``F.scaled_dot_product_attention``.
    attn_layer : nn.Module | None, default=None
        Attention sublayer instance. ``None`` builds the default
        :class:`MultiHeadAttention` from ``num_kv_heads``/``use_sdpa`` (which
        are then ignored). Lets any attention variant (e.g. a
        :class:`~spartan_torch.LinformerAttention`) replace the default; the
        instance must accept ``(query, key, value, mask=...)`` and return
        either a tensor or an ``(output, cache)`` tuple. A single instance
        passed to several blocks shares weights.
    """

    def __init__(
        self,
        query_size: int,
        head_size: int,
        num_heads: int,
        out_size: int,
        memory_size: int | None = None,
        num_kv_heads: int | None = None,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        attn_p: float = 0.0,
        dropout_p: float = 0.0,
        use_sdpa: bool = False,
        attn_layer: nn.Module | None = None,
    ):
        super().__init__()
        memory_size = query_size if memory_size is None else memory_size

        self.attn = (
            attn_layer
            if attn_layer is not None
            else MultiHeadAttention(
                memory_size,
                head_size,
                num_heads,
                out_size,
                query_in_size=query_size,
                num_kv_heads=num_kv_heads,
                attn_p=attn_p,
                use_sdpa=use_sdpa,
            )
        )
        self.adapt_residual = nn.Linear(query_size, out_size) if query_size != out_size else nn.Identity()
        self.norm_q = norm_layer(query_size)
        self.norm_mv = norm_layer(memory_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply cross-attention.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, query_seq_len, query_size)``.
        memory : torch.Tensor
            ``(batch, memory_seq_len, memory_size)``. Must be ``<=`` the
            key-side ``max_seq_len`` when an injected
            :class:`~spartan_torch.LinformerAttention` is used.
        memory_mask : torch.Tensor | None, default=None
            Bool tensor (``True`` = masked out) or additive float score bias,
            broadcastable to ``(batch, num_heads, query_seq_len, memory_seq_len)``.
            Typically masks out padding in ``memory``. Unsupported by
            :class:`~spartan_torch.LinformerAttention` (raises
            ``NotImplementedError``) — memory must be packed.

        Returns
        -------
        torch.Tensor
            ``(batch, query_seq_len, out_size)``.
        """
        attn_out = self.attn(self.norm_q(x), self.norm_mv(memory), self.norm_mv(memory), mask=memory_mask)
        h = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        return self.adapt_residual(x) + self.dropout(h)
