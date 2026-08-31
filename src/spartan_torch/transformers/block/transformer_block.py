from collections import OrderedDict
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from ..attention import MultiHeadAttention
from .cross_attention_block import CrossAttentionBlock


class FeedForward(nn.Module):
    """Two-layer feed-forward block with configurable activation.

    ``Linear(hidden_size) -> activation -> Linear(in_size)``. Default FF
    sublayer of :class:`TransformerBlock`; the natural swap point for MoE-style
    expert routing, which just needs the same ``(batch, seq, in_size)`` in/out
    contract.
    """

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        try:
            self.activation = activation(inplace=True)
        except TypeError:
            self.activation = activation()
        self.layers = nn.Sequential(OrderedDict([
            ("lin_1", nn.Linear(in_size, hidden_size)),
            ("act", self.activation),
            ("lin_2", nn.Linear(hidden_size, in_size)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SwiGLUFeedForward(nn.Module):
    """Gated feed-forward block used by the Llama family (LLaMA, Mistral, Qwen).

    ``down_proj(silu(gate_proj(x)) * up_proj(x))``: the ``gate_proj``
    projection gates the ``up_proj`` value projection elementwise through
    SiLU, and ``down_proj`` maps back to the embedding dim. Three projections
    instead of two make the FFN strictly more expressive than the plain
    ``Linear -> activation -> Linear`` :class:`FeedForward` for the same
    ``hidden_size`` (the effective width is doubled).

    This is the natural ``ff_layer`` for :class:`TransformerBlock` in
    Llama-style decoders; ``hidden_size`` is the *intermediate* size from the
    model config (e.g. 5632 in TinyLlama for ``d_model=2048``), which both
    ``gate_proj`` and ``up_proj`` expand into.

    Parameters
    ----------
    in_size : int
        Embedding size of the input/output.
    hidden_size : int
        Intermediate size of the gate and value projections.
    bias : bool, default=False
        Add biases to the projections. Llama-family FFNs run without biases.

    References
    ----------
    "GLU Variants Improve Transformer" (Shazeer, 2020, arXiv:2002.05202).
    """

    def __init__(self, in_size: int, hidden_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(in_size, hidden_size, bias=bias)
        self.up_proj = nn.Linear(in_size, hidden_size, bias=bias)
        self.down_proj = nn.Linear(hidden_size, in_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional grouped-query and cross-attention.

    Layered as ``x + Attn(LN1(x))`` then, when ``with_cross_attn=True``,
    ``h + CrossAttn(LN_q(h), LN_mv(memory), LN_mv(memory))``, then
    ``h + FF(LN2(h))`` (all pre-LN residual sublayers). One class covers the
    whole encoder/decoder spectrum: plain self-attention with ``is_causal=False``
    is an encoder block, ``is_causal=True`` is a decoder self-attention block,
    and adding cross-attention over the encoder memory makes an
    encoder-decoder decoder layer (T5-style).

    The feed-forward sublayer is injectable via ``ff_layer`` so MoE-style
    expert routing or any other FF variant can be dropped in without touching
    the block.

    Parameters
    ----------
    in_size : int
        Embedding size of the input (query, key and value for self-attention).
    head_size : int
        Per-head hidden dim of Q, K, V.
    num_heads : int
        Number of query heads. Must be a multiple of ``num_kv_heads``.
    out_size : int
        Output embedding size.
    ff_hidden_size : int
        Hidden size of the default feed-forward block.
    with_cross_attn : bool, default=False
        Add a pre-norm cross-attention sublayer reading the encoder memory.
    memory_size : int | None, default=None
        Embedding size of the encoder memory (cross-attention only).
        ``None`` falls back to ``out_size``.
    num_kv_heads : int | None, default=None
        Number of shared key/value heads passed to both attention blocks.
        ``None`` means plain MHA (``num_kv_heads == num_heads``).
    is_causal : bool, default=False
        Mask out future tokens in the self-attention (decoder mode).
    activation : type[nn.Module], default=nn.GELU
        Activation class for the default feed-forward block, instantiated with
        ``inplace=True`` when supported (falls back otherwise). Ignored when
        ``ff_layer`` is provided.
    norm_layer : Callable[[int], nn.Module], default=nn.LayerNorm
        Normalization factory called with the embedding size.
    norm_mode : Literal["pre", "hybrid"], default="pre"
        Block-level normalization strategy:

        * ``"pre"`` — the default pre-norm block: ``x + Attn(LN1(x))`` then
          ``h + FF(LN2(h))`` (identity path through the residual).
        * ``"hybrid"`` — HybridNorm ("HybridNorm: Towards Stable and
          Efficient Transformer Training via Hybrid Normalization",
          Zhuo et al., 2025, arXiv:2503.04598): the attention branch skips
          the input norm and instead normalizes the projected QKV per head
          inside the attention module (``qkv_norm=True``), while the FFN
          branch uses Post-Norm with the normalized value as its own residual
          (``LN2(h) + FF(LN2(h))``). HybridNorm* — first block pre-norm,
          rest hybrid — is composed by building the blocks with per-layer
          ``norm_mode`` values in a ``ModuleList``.

          Cross-attention (``with_cross_attn=True``) always stays pre-norm —
          the paper targets self-attention blocks. An injected
          ``attn_layer`` is used as-is and gets no QKV-norm.
    attn_p : float, default=0.0
        Dropout probability on attention weights (passed to both MHA blocks).
    dropout_p : float, default=0.0
        Dropout probability on the residual branches.
    use_sdpa : bool, default=False
        Route attention through ``F.scaled_dot_product_attention``.
    qk_mod : Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None, default=None
        Post-head-split query/key modulation (e.g. a
        ``RotaryPositionalEmbedding``), passed to the self-attention block.
        Cross-attention keeps ``None``. See :class:`MultiHeadAttention`.
    ff_layer : nn.Module | None, default=None
        Feed-forward sublayer instance. ``None`` builds the default
        :class:`FeedForward`. Lets MoE or any custom FF variant replace the
        default. A single instance passed to several blocks shares weights.
    attn_layer : nn.Module | None, default=None
        Self-attention sublayer instance. ``None`` builds the default
        :class:`MultiHeadAttention` from ``num_kv_heads``/``is_causal``/
        ``use_sdpa``/``qk_mod`` (which are then ignored). Lets any
        attention variant (e.g. a
        :class:`~spartan_torch.LinformerAttention`) replace the default; the
        instance must accept ``(query, key, value, mask=..., past_key_value=...)``
        and return either a tensor or an ``(output, cache)`` tuple. A single
        instance passed to several blocks shares weights. Causal/masked
        attention layers only work here if ``mask``/``past_key_value`` stay
        ``None``.
    cross_attn_layer : nn.Module | None, default=None
        Cross-attention sublayer instance (only used when
        ``with_cross_attn=True``). ``None`` builds the default
        :class:`MultiHeadAttention` inside
        :class:`CrossAttentionBlock`. Lets a
        :class:`~spartan_torch.LinformerAttention` read a long encoder memory
        in ``O(query_seq_len * proj_k)``; the instance must accept
        ``(query, key, value, mask=...)`` and return either a tensor or an
        ``(output, cache)`` tuple. A single instance passed to several blocks
        shares weights. Padding ``memory_mask`` is unsupported by
        :class:`~spartan_torch.LinformerAttention` — memory must be packed.
        Requires ``with_cross_attn=True``.
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        num_heads: int,
        out_size: int,
        ff_hidden_size: int,
        with_cross_attn: bool = False,
        memory_size: int | None = None,
        num_kv_heads: int | None = None,
        is_causal: bool = False,
        activation: type[nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        norm_mode: str = "pre",
        attn_p: float = 0.0,
        dropout_p: float = 0.0,
        use_sdpa: bool = False,
        qk_mod: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        ff_layer: nn.Module | None = None,
        attn_layer: nn.Module | None = None,
        cross_attn_layer: nn.Module | None = None,
    ):
        super().__init__()
        if norm_mode not in ("pre", "hybrid"):
            raise ValueError(f"norm_mode must be 'pre' or 'hybrid', got {norm_mode!r}")
        self.norm_mode = norm_mode
        self.with_cross_attn = with_cross_attn
        if cross_attn_layer is not None and not with_cross_attn:
            raise ValueError("cross_attn_layer requires with_cross_attn=True")

        self.attn = (
            attn_layer
            if attn_layer is not None
            else MultiHeadAttention(
                in_size,
                head_size,
                num_heads,
                out_size,
                num_kv_heads=num_kv_heads,
                is_causal=is_causal,
                attn_p=attn_p,
                use_sdpa=use_sdpa,
                qk_mod=qk_mod,
                qkv_norm=norm_mode == "hybrid",
                qkv_norm_layer=norm_layer,
            )
        )
        self.adapt_residual = nn.Linear(in_size, out_size) if in_size != out_size else nn.Identity()
        self.norm1 = norm_layer(in_size) if norm_mode == "pre" else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_p)

        if with_cross_attn:
            self.cross_attn = CrossAttentionBlock(
                out_size,
                head_size,
                num_heads,
                out_size,
                memory_size=memory_size,
                num_kv_heads=num_kv_heads,
                norm_layer=norm_layer,
                attn_p=attn_p,
                dropout_p=dropout_p,
                use_sdpa=use_sdpa,
                attn_layer=cross_attn_layer,
            )

        self.norm2 = norm_layer(out_size)
        self.dropout2 = nn.Dropout(dropout_p)
        self.ff = FeedForward(out_size, ff_hidden_size, activation) if ff_layer is None else ff_layer

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply one transformer block.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, in_size)``. Query, key and value of the
            self-attention are the same tensor.
        memory : torch.Tensor | None, default=None
            ``(batch, memory_seq_len, memory_size)``. Encoder output for the
            cross-attention. Required when ``with_cross_attn=True``, ignored
            otherwise.
        mask : torch.Tensor | None, default=None
            Bool tensor (``True`` = masked out) or additive float score bias,
            broadcastable to ``(batch, num_heads, seq_len, seq_len)``.
            Combined with the causal mask when ``is_causal=True``.
        memory_mask : torch.Tensor | None, default=None
            Bool tensor or additive float score bias for the cross-attention,
            broadcastable to ``(batch, num_heads, seq_len, memory_seq_len)``.
        past_key_value : tuple[torch.Tensor, torch.Tensor] | None, default=None
            Cached key/value states of the self-attention for all tokens before
            ``x`` (KV-cache decoding). Passed through to
            :class:`MultiHeadAttention`.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]
            ``(output, cache)``: the block output ``(batch, seq_len,
            out_size)`` and the current tokens' self-attention key/value states
            to feed the next decode step. ``cache`` is ``None`` when the
            injected ``attn_layer`` does not produce one (e.g.
            :class:`~spartan_torch.LinformerAttention`).
        """
        h = self.norm1(x)
        attn_out = self.attn(h, h, h, mask=mask, past_key_value=past_key_value)
        h, cache = attn_out if isinstance(attn_out, tuple) else (attn_out, None)
        h = self.adapt_residual(x) + self.dropout1(h)

        if self.with_cross_attn:
            if memory is None:
                raise ValueError("memory is required when with_cross_attn=True")
            h = self.cross_attn(h, memory, memory_mask=memory_mask)

        if self.norm_mode == "hybrid":
            h2 = self.norm2(h)
            return h2 + self.dropout2(self.ff(h2)), cache
        return h + self.dropout2(self.ff(self.norm2(h))), cache
