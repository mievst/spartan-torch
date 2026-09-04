import math
from collections.abc import Callable

import torch
from torch import nn

from .mha import _split_heads
from .performer import (
    PerformerAttention,
    _bidir_context,
    _causal_context,
    _key_mask,
    _softmax_features,
    gaussian_orthogonal_random_matrix,
)

_PROJECTION_PATHS = {
    "q": ("q_proj", "query_matrix", "query"),
    "k": ("k_proj", "key_matrix", "key"),
    "v": ("v_proj", "value_matrix", "value"),
    "o": ("o_proj", "out", "output.dense"),
}


def _resolve(module: nn.Module, path: str) -> nn.Module:
    obj: object = module
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _probe_projection_paths(module: nn.Module) -> dict[str, str]:
    """Find the trained Q/K/V/O projections of ``module`` by name probing.

    Covers the common families: HF decoder-style (``q_proj``/``k_proj``/
    ``v_proj``/``o_proj``), spartan-torch MHA (``query_matrix``/``key_matrix``/
    ``value_matrix``/``out``) and BERT-style (``query``/``key``/``value``/
    ``output.dense``). Raises ``ValueError`` listing which projection is
    missing; fused projections (e.g. GPT-2 ``c_attn``) cannot be found by name
    — pass ``qkv_fn`` instead.
    """
    paths = {}
    for role, candidates in _PROJECTION_PATHS.items():
        for candidate in candidates:
            try:
                _resolve(module, candidate)
            except AttributeError:
                continue
            paths[role] = candidate
            break
        else:
            raise ValueError(
                f"could not locate the '{role}' projection of {type(module).__name__}; "
                f"tried {list(candidates)}. For fused projections pass qkv_fn="
            )
    return paths


def _projection_in_features(module: nn.Module, path: str, role: str) -> int:
    proj = _resolve(module, path)
    weight = getattr(proj, "weight", None)
    if weight is None:
        raise ValueError(
            f"projection '{role}' at '{path}' has no weight tensor; "
            "a PerformerAdapter needs static Q/K/V/O in_features"
        )
    return weight.shape[-1]


def _infer_geometry(module: nn.Module, paths: dict[str, str], head_size: int) -> tuple[int, int]:
    q_proj = _resolve(module, paths["q"])
    q_weight = getattr(q_proj, "weight", None)
    if q_weight is None:
        raise ValueError(f"query projection '{paths['q']}' has no weight tensor")
    q_out = q_weight.shape[0]
    if q_out % head_size != 0:
        raise ValueError(
            f"query projection out_features {q_out} is not a multiple of head_size {head_size}"
        )
    num_heads = q_out // head_size

    o_proj = _resolve(module, paths["o"])
    o_weight = getattr(o_proj, "weight", None)
    if o_weight is None:
        raise ValueError(f"output projection '{paths['o']}' has no weight tensor")
    out_size = o_weight.shape[0]

    k_proj = _resolve(module, paths["k"])
    k_weight = getattr(k_proj, "weight", None)
    if k_weight is None:
        raise ValueError(f"key projection '{paths['k']}' has no weight tensor")
    kv_out = k_weight.shape[0]
    if kv_out != q_out:
        raise ValueError(
            f"key projection out_features {kv_out} != query out_features {q_out}; "
            "grouped-query attention is not supported by a Performer adapter"
        )
    return num_heads, out_size


class PerformerAdapter(nn.Module):
    """Performer (FAVOR+) adapter over a *trained* attention module.

    Repurposes an existing attention (spartan-torch MHA, HF decoder/BERT
    attention, or a custom one) as a linear-complexity Performer **without
    retraining**: the trained Q/K/V/O projection weights are reused as-is and
    only the score computation is replaced by the kernel estimator. The
    kernel is unbiased for softmax, so with enough features
    (``num_features``) the output of the wrapped attention approaches its
    exact softmax output on the same projections — the Performer-adapter
    analogue of LoRA (LoRA adapts *weights* for efficiency; the Performer
    adapter adapts the *algorithm* for throughput at long sequences).

    The projections are resolved by name (see :func:`_probe_projection_paths`)
    or overridden explicitly. The resolved modules stay **owned by the wrapped
    ``attention``** — the adapter holds a reference but never re-registers
    them, so ``state_dict`` and ``parameters()`` keep each projection exactly
    once (the adapter only adds its own non-trainable feature-projection
    buffer). ``freeze=True`` freezes the wrapped attention's parameters so a
    frozen-pretrained + adapter pipeline cannot accidentally train them.

    The interface mirrors :class:`MultiHeadAttention`: ``forward(query, key,
    value, mask, past_key_value)`` with ``(batch, seq, embed)`` layout, but
    returns a plain tensor (no KV cache — kernel attention has none). For
    wrapped modules whose call convention differs (HF decoders), the caller
    provides a facade — see ``experiments/performer``.

    Parameters
    ----------
    attention : nn.Module
        The trained attention module being wrapped. Becomes the adapter's
        single child (keeps the resolved projections in its own tree).
    num_heads : int
        Number of query heads. Must match the wrapped Q projection.
    head_size : int
        Per-head dim of Q/K/V; also the ORF dimension. Must match the wrapped
        projections.
    out_size : int
        Output embedding size. Must match the wrapped O projection.
    num_features : int | None, default=None
        Number of random features ``m``. Defaults to
        ``int(head_size * log(head_size))``.
    q_path, k_path, v_path, o_path : str | None, default=None
        Dotted attribute paths to the trained projection modules. ``None``
        probes the defaults.
    qkv_fn : Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None, default=None
        Fused projection hook for models whose Q/K/V are computed by one
        module (e.g. GPT-2 ``c_attn``): ``q, k, v = qkv_fn(x)`` with ``x``
        the key/value input, each output ``(batch, seq, embed)`` in Q/K/V
        order. When given, overrides the resolved q/k/v paths (self-attention
        fused projections). The ``value`` input argument is ignored.
    qk_mod : Callable | None, default=None
        Post-head-split q/k modulation — the rotary-position injection point,
        same contract as ``MultiHeadAttention.qk_mod``: ``q, k = qk_mod(q, k,
        q_positions, k_positions)`` with ``(batch, heads, seq_len, head_size)``
        and ``(seq_len,)`` integer positions. Applied before the kernel
        estimator. If it is an ``nn.Module``, the caller owns it (store it
        outside the adapter, e.g. on the model).
    freeze : bool, default=True
        Set ``requires_grad_(False)`` on every parameter of the wrapped
        ``attention``. Its projections are read-only feature extractors; the
        adapter adds no trainable parameters of its own.
    is_causal : bool, default=False
        Use the causal (prefix-sum) kernel form. Requires equal query/key
        lengths.
    ortho_scaling : int, default=0
        ORF row-norm scheme, see :class:`PerformerAttention`.

    References
    ----------
    "Rethinking Attention with Performers" (Choromanski et al., 2020,
    arXiv:2009.14794).
    """

    def __init__(
        self,
        attention: nn.Module,
        num_heads: int,
        head_size: int,
        out_size: int,
        num_features: int | None = None,
        q_path: str | None = None,
        k_path: str | None = None,
        v_path: str | None = None,
        o_path: str | None = None,
        qkv_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
        qk_mod: Callable | None = None,
        freeze: bool = True,
        is_causal: bool = False,
        ortho_scaling: int = 0,
    ):
        super().__init__()
        num_features = int(head_size * math.log(head_size)) if num_features is None else num_features
        if num_features < 1:
            raise ValueError(f"num_features must be positive, got {num_features}")

        self.attention = attention
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_size = out_size
        self.num_features = num_features
        self.is_causal = is_causal

        overrides = {"q": q_path, "k": k_path, "v": v_path, "o": o_path}
        paths = {}
        for role, path in overrides.items():
            if path is not None:
                _resolve(attention, path)  # validate early
                paths[role] = path
            else:
                paths[role] = _probe_projection_paths(attention)[role]
        self._paths = paths
        self._qkv_fn = qkv_fn

        for role in ("q", "k", "v", "o"):
            if qkv_fn is not None and role != "o":
                continue
            in_features = _projection_in_features(attention, paths[role], role)
            if in_features < 1:
                raise ValueError(f"projection '{role}' has non-positive in_features {in_features}")

        # Stateless hook, caller owns state: same pattern as MHA.qk_mod.
        object.__setattr__(self, "qk_mod", qk_mod)

        self.register_buffer(
            "R",
            gaussian_orthogonal_random_matrix(num_features, head_size, scaling=ortho_scaling),
        )

        if freeze:
            for p in attention.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_module(
        cls,
        attention: nn.Module,
        head_size: int,
        num_features: int | None = None,
        qk_mod: Callable | None = None,
        freeze: bool = True,
        is_causal: bool = False,
        ortho_scaling: int = 0,
    ) -> "PerformerAdapter":
        """Build an adapter by inferring ``num_heads``/``out_size`` from the
        wrapped Q/O projection shapes (self-attention modules only).

        Raises ``ValueError`` when the module has no identifiable projections
        or when the key projection does not match the query geometry (GQA).
        """
        paths = _probe_projection_paths(attention)
        num_heads, out_size = _infer_geometry(attention, paths, head_size)
        return cls(
            attention,
            num_heads=num_heads,
            head_size=head_size,
            out_size=out_size,
            num_features=num_features,
            q_path=paths["q"],
            k_path=paths["k"],
            v_path=paths["v"],
            o_path=paths["o"],
            qk_mod=qk_mod,
            freeze=freeze,
            is_causal=is_causal,
            ortho_scaling=ortho_scaling,
        )

    def _project(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = key.size(0)
        q_len = query.size(1)
        k_len = key.size(1)

        if self._qkv_fn is not None:
            q, k, v = self._qkv_fn(key)
            if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
                raise ValueError(f"qkv_fn must return (batch, seq, embed) tensors, got shapes "
                                 f"{tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}")
            if q.size(1) != k_len:
                raise ValueError(f"qkv_fn query length {q.size(1)} != key length {k_len}")
            return (
                _split_heads(q, batch, self.num_heads, self.head_size, k_len),
                _split_heads(k, batch, self.num_heads, self.head_size, k_len),
                _split_heads(v, batch, self.num_heads, self.head_size, k_len),
            )

        q = _split_heads(_resolve(self.attention, self._paths["q"])(query), batch, self.num_heads, self.head_size, q_len)
        k = _split_heads(_resolve(self.attention, self._paths["k"])(key), batch, self.num_heads, self.head_size, k_len)
        v = _split_heads(_resolve(self.attention, self._paths["v"])(value), batch, self.num_heads, self.head_size, k_len)
        return q, k, v

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply the Performer adapter over the trained projections.

        Parameters
        ----------
        query : torch.Tensor
            ``(batch, query_seq_len, embed)``.
        key : torch.Tensor
            ``(batch, key_seq_len, embed)``.
        value : torch.Tensor
            ``(batch, key_seq_len, embed)``.
        mask : torch.Tensor | None, default=None
            Key (padding) mask, bool ``(batch, 1, 1, key_seq_len)`` or
            broadcastable. Pair-wise masks raise ``NotImplementedError``.
        past_key_value : tuple | None, default=None
            Unsupported (no incremental cache) — raises ``NotImplementedError``.

        Returns
        -------
        torch.Tensor
            Attention output ``(batch, query_seq_len, out_size)``.
        """
        if past_key_value is not None:
            raise NotImplementedError(
                "PerformerAdapter has no incremental KV cache; pass the full sequence"
            )

        batch = key.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        if self.is_causal and q_len != k_len:
            raise ValueError(f"causal adapter requires query_seq_len == key_seq_len, got {q_len} != {k_len}")

        km = _key_mask(mask, k_len, self.num_heads, q_len) if mask is not None else None

        q, k, v = self._project(query, key, value)

        if self.qk_mod is not None:
            q_positions = torch.arange(q_len, device=q.device)
            k_positions = torch.arange(k_len, device=k.device)
            q, k = self.qk_mod(q, k, q_positions, k_positions)

        qf = _softmax_features(q, self.R, is_query=True)
        kf = _softmax_features(k, self.R, is_query=False)
        if km is not None:
            # Zero the *feature maps* of masked keys (not the raw keys, whose
            # zero input would map to the largest features φ(0) = 1 + eps).
            km_t = km.transpose(-1, -2)
            kf = kf.masked_fill(km_t, 0.0)
            v = v.masked_fill(km_t, 0.0)

        if self.is_causal:
            context = _causal_context(qf, kf, v)
        else:
            context = _bidir_context(qf, kf, v)

        return _resolve(self.attention, self._paths["o"])(
            context.transpose(1, 2).reshape(batch, q_len, self.num_heads * self.head_size)
        )


def performerize_attentions(
    model: nn.Module,
    head_size: int,
    num_features: int | None = None,
    qk_mod: Callable | None = None,
    freeze: bool = True,
    is_causal: bool = False,
    ortho_scaling: int = 0,
) -> list[tuple[nn.Module, PerformerAdapter]]:
    """Replace every attention module in ``model`` with a
    :class:`PerformerAdapter` in place.

    Walks the module tree depth-first and swaps each module that has
    identifiable Q/K/V/O projections (see :func:`_probe_projection_paths`) for
    ``PerformerAdapter.from_module`` built over it. Wrap order is nesting-safe:
    an adapter is inserted for a module only if none of its ancestors was
    already adapted, so a transformer block's self-attention is adapted
    exactly once.

    HF models are recognized by projection names, but their *calling
    convention* (decoder layer → ``self_attn(hidden_states, attention_mask, ...)``)
    differs from the adapter's ``(query, key, value, ...)`` contract, so this
    helper is for models that call their attention the library way. For HF
    models build a facade around each adapter — see ``experiments/performer``.

    Returns
    -------
    list[tuple[nn.Module, PerformerAdapter]]
        ``(original, adapter)`` pairs in DFS order.
    """
    replacements: list[tuple[nn.Module, PerformerAdapter]] = []

    def walk(module: nn.Module) -> None:
        for name, child in list(module._modules.items()):
            if child is None:
                continue
            try:
                paths = _probe_projection_paths(child)
            except ValueError:
                walk(child)
                continue
            num_heads, out_size = _infer_geometry(child, paths, head_size)
            adapter = PerformerAdapter(
                child,
                num_heads=num_heads,
                head_size=head_size,
                out_size=out_size,
                num_features=num_features,
                q_path=paths["q"],
                k_path=paths["k"],
                v_path=paths["v"],
                o_path=paths["o"],
                qk_mod=qk_mod,
                freeze=freeze,
                is_causal=is_causal,
                ortho_scaling=ortho_scaling,
            )
            module._modules[name] = adapter
            replacements.append((child, adapter))
            # Adapter owns the original as its child; don't re-walk it.

    walk(model)
    return replacements
