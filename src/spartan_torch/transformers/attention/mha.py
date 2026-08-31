import math
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from ..norm import QKVNorm


def _split_heads(x: torch.Tensor, batch: int, num_heads: int, head_size: int, seq_len: int) -> torch.Tensor:
    """Reshape ``(batch, seq_len, num_heads * head_size)`` to ``(batch, num_heads, seq_len, head_size)``."""
    return x.view(batch, seq_len, num_heads, head_size).transpose(1, 2)


def _fold_mask(
    mask: torch.Tensor | None,
    is_causal: bool,
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
    k_offset: int = 0,
) -> torch.Tensor | None:
    """Merge causal masking into the user mask.

    ``k_offset`` is the number of key/value tokens cached before the current
    ones (KV-cache decoding). The causal diagonal shifts by it so that the
    query at global position ``p + i`` still attends to every key up to ``p + i``
    (during decoding ``q_len == 1`` this yields an empty mask — the new token
    attends to the whole cache).

    Returns a single mask in the layer convention: a bool tensor with ``True``
    marking positions to be masked out, or a float tensor used as an additive
    score bias.
    """
    causal = None
    if is_causal:
        causal = torch.triu(
            torch.ones(q_len, k_len, dtype=torch.bool, device=device),
            diagonal=1 + k_offset,
        )

    if mask is None:
        return causal

    if mask.dtype == torch.bool:
        return mask | causal if causal is not None else mask

    if causal is not None:
        return mask + causal.to(dtype) * torch.finfo(dtype).min
    return mask


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_p: float,
    is_causal: bool,
    mask: torch.Tensor | None,
    use_sdpa: bool,
    training: bool,
    head_size: int,
    k_offset: int = 0,
) -> torch.Tensor:
    """Core scaled dot-product attention.

    ``q``, ``k``, ``v`` have layout ``(batch, num_heads, seq_len, head_size)``.
    ``mask`` is a bool tensor (``True`` = masked out) or an additive float bias,
    broadcastable to the ``(batch, num_heads, q_len, k_len)`` score tensor.
    ``k_offset`` is the number of cached key/value tokens before ``k`` (KV
    cache decoding), used to shift the causal diagonal.

    With ``use_sdpa=True`` the computation is delegated to
    ``F.scaled_dot_product_attention``, which transparently dispatches to the
    fastest available backend (FlashAttention, Memory-Efficient or a fused math
    kernel) depending on device and dtype.
    """
    q_len = q.size(-2)
    k_len = k.size(-2)
    eff_mask = _fold_mask(mask, is_causal, q_len, k_len, q.device, q.dtype, k_offset)

    if use_sdpa:
        dropout_p = attn_p if training else 0.0
        attn_mask = None
        if eff_mask is not None:
            # SDPA uses the inverted convention: True = participate.
            attn_mask = ~eff_mask if eff_mask.dtype == torch.bool else eff_mask
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )

    scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)
    if eff_mask is not None:
        if eff_mask.dtype == torch.bool:
            # -inf in fp16/bf16 turns fully-masked rows into NaN after softmax.
            scores = scores.masked_fill(eff_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + eff_mask

    weights = F.softmax(scores, dim=-1)
    if training and attn_p > 0.0:
        weights = F.dropout(weights, attn_p, training=True)
    return weights @ v


class MultiHeadAttention(nn.Module):
    """Multi-head attention (MHA) with optional grouped-query key/value heads (GQA).

    One class covers the whole MHA/GQA/MQA spectrum. Unlike
    ``torch.nn.MultiheadAttention``:

    * ``head_size`` is a free parameter decoupled from ``num_heads`` — the Q
      projection dim is ``head_size * num_heads``, not necessarily ``in_size``.
    * ``num_kv_heads`` lets key/value heads be shared across groups of query
      heads, shrinking the KV cache and K/V compute during decoding with a
      negligible quality hit. ``num_kv_heads == num_heads`` is plain MHA,
      ``num_kv_heads == 1`` is multi-query attention (MQA).
    * ``query_in_size`` allows the query to come from a different embedding dim
      than key/value (cross-attention).
    * tensors use the ``(batch, seq, embed)`` layout.

    Reference for GQA: "GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints" (Ainslie et al., 2023, arXiv:2305.13245).

    Parameters
    ----------
    in_size : int
        Embedding size of key/value inputs.
    head_size : int
        Per-head hidden dim of Q, K, V.
    num_heads : int
        Number of query heads. Must be a multiple of ``num_kv_heads``.
    out_size : int
        Output embedding size.
    query_in_size : int | None, default=None
        Embedding size of the query input. ``None`` falls back to ``in_size``.
    num_kv_heads : int | None, default=None
        Number of shared key/value heads. ``None`` means plain MHA
        (``num_kv_heads == num_heads``).
    attn_p : float, default=0.0
        Dropout probability applied to attention weights in training mode.
    is_causal : bool, default=False
        Mask out future tokens (query position ``i`` cannot attend to key
        position ``j > i``).
    use_sdpa : bool, default=False
        Route through ``F.scaled_dot_product_attention`` to get fused
        FlashAttention / Memory-Efficient backends on CUDA.
    qk_mod : Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None, default=None
        Post-head-split modulation of the query/key vectors — the natural
        injection point for rotary position embeddings (e.g. a
        ``RotaryPositionalEmbedding``) and other per-head q/k transforms
        (XPos). Applied after head splitting and before GQA repetition:
        ``q, k = qk_mod(q, k, q_positions, k_positions)`` with layout
        ``(batch, heads, seq_len, head_size)`` and ``(seq_len,)`` integer
        position tensors (absolute positions, so KV-cache decoding rotates
        new tokens by their true offset). MHA only calls it; if it is an
        ``nn.Module`` the caller owns it and must register it in its own
        module tree so ``.to(device)`` and ``state_dict`` include it.
    qkv_norm : bool, default=False
        Normalize the projected ``q``/``k``/``v`` per head with a
        :class:`~spartan_torch.QKVNorm` stage (HybridNorm). The norms are
        owned by this module — they live in ``state_dict`` and move with
        ``.to(device)``. Applied right after head splitting, before
        ``qk_mod`` and before GQA repetition, so cached key/value states are
        already normalized (decoding stays consistent with prefill).
    qkv_norm_layer : Callable[[int], nn.Module], default=nn.LayerNorm
        Normalization factory for the QKV-norm stage, called with
        ``head_size``. Ignored unless ``qkv_norm=True``. Pass RMSNorm to
        reproduce the paper exactly.
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        num_heads: int,
        out_size: int,
        query_in_size: int | None = None,
        num_kv_heads: int | None = None,
        attn_p: float = 0.0,
        is_causal: bool = False,
        use_sdpa: bool = False,
        qk_mod: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        qkv_norm: bool = False,
        qkv_norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.out_size = out_size
        self.query_in_size = in_size if query_in_size is None else query_in_size
        self.attn_p = attn_p
        self.is_causal = is_causal
        self.use_sdpa = use_sdpa
        # Plain attribute, not an nn.Module child: qk_mod is a stateless hook
        # and the caller owns its state. Registering it here would duplicate
        # the same module (and its buffers) once per attention block, and
        # shared instances would go stale after in-place cache extension.
        object.__setattr__(self, "qk_mod", qk_mod)
        # Owned by this module, unlike qk_mod: per-head QKV normalization
        # (HybridNorm) has per-block parameters, so it must live in the
        # module tree to be saved/moved with it.
        self.qkv_norm = QKVNorm(head_size, qkv_norm_layer) if qkv_norm else None

        self.query_matrix = nn.Linear(self.query_in_size, head_size * num_heads, bias=False)
        self.key_matrix = nn.Linear(self.in_size, head_size * num_kv_heads, bias=False)
        self.value_matrix = nn.Linear(self.in_size, head_size * num_kv_heads, bias=False)
        self.out = nn.Linear(head_size * num_heads, out_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply (grouped-)multi-head attention.

        Parameters
        ----------
        query : torch.Tensor
            ``(batch, query_seq_len, query_in_size)``. The current tokens —
            the full sequence on the prefill step, only the new token(s) when
            ``past_key_value`` is given.
        key : torch.Tensor
            ``(batch, key_seq_len, in_size)``.
        value : torch.Tensor
            ``(batch, key_seq_len, in_size)``.
        mask : torch.Tensor | None, default=None
            Bool tensor (``True`` = masked out) or additive float score bias,
            broadcastable to ``(batch, num_heads, query_seq_len, key_seq_len)``.
            Combined with the causal mask when ``is_causal=True``.
        past_key_value : tuple[torch.Tensor, torch.Tensor] | None, default=None
            Cached key/value states for all tokens before the current ones,
            each ``(batch, num_heads, past_seq_len, head_size)`` (already
            rotated by :data:`qk_mod` if one is set). When given, the current
            ``k``/``v`` are appended after it for attention and only the current
            tokens' states are returned as the new cache.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            ``(output, cache)``: the attention output
            ``(batch, query_seq_len, out_size)`` and the key/value states of
            the *current* input tokens (after head splitting, ``qk_mod``
            rotation and — for GQA — without shared-head repetition) — the
            ``past_key_value`` to feed the next decode step. The first element
            of the returned cache has ``seq_len == key.size(1)``.
        """
        batch = key.size(0)
        k_len = key.size(1)
        q_len = query.size(1)

        q = _split_heads(self.query_matrix(query), batch, self.num_heads, self.head_size, q_len)
        k = _split_heads(self.key_matrix(key), batch, self.num_kv_heads, self.head_size, k_len)
        v = _split_heads(self.value_matrix(value), batch, self.num_kv_heads, self.head_size, k_len)

        if self.qkv_norm is not None:
            q, k, v = self.qkv_norm(q, k, v)

        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)

        if self.qk_mod is not None:
            q_positions = torch.arange(past_len, past_len + q_len, device=q.device)
            k_positions = torch.arange(past_len, past_len + k_len, device=k.device)
            q, k = self.qk_mod(q, k, q_positions, k_positions)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_full = torch.cat([past_k, k], dim=-2)
            v_full = torch.cat([past_v, v], dim=-2)
        else:
            k_full, v_full = k, v

        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)

        context = _attention(
            q, k_full, v_full, self.attn_p, self.is_causal, mask, self.use_sdpa, self.training, self.head_size, past_len
        )
        out = self.out(context.transpose(1, 2).reshape(batch, q_len, self.num_heads * self.head_size))
        return out, (k, v)
