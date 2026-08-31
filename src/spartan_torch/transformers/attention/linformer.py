import math

import torch
from torch import nn
import torch.nn.functional as F

from .mha import _split_heads


def _init_proj(p: torch.Tensor) -> None:
    """Default ``nn.Linear`` init (``kaiming_uniform_`` with ``a=sqrt(5)``)."""
    nn.init.kaiming_uniform_(p, a=math.sqrt(5))


class LinformerSeqProjection(nn.Module):
    """Sequence-dimension projection (:math:`E`/:math:`F`) for Linformer attention.

    Holds the learned :math:`k \\times n` matrices that contract the
    key/value sequence dimension before the attention product. Injected into
    :class:`LinformerAttention` via ``projection=``. A single instance shared
    across several layers reproduces the paper's cross-layer (``layerwise``)
    parameter sharing — only the projection is tied, Q/K/V linears stay
    per-layer.

    Parameter sharing (paper §4.2) — which matrices exist and their layout:

    ===========  ============================================  ============
    ``sharing``  projection parameters                         # params
    ===========  ============================================  ============
    ``none``     per-head :math:`E_i`, :math:`F_i`             2·H·n·k
    ``headwise`` one :math:`E`, one :math:`F` for all heads    2·n·k
    ``kv``       per-head single matrix (:math:`E_i = F_i`)    H·n·k
    ``layerwise``one matrix everywhere (:math:`E = F`)         n·k
    ===========  ============================================  ============

    The paper reports the most aggressive (``layerwise``) scheme performing
    best — fewest parameters *and* best downstream quality — so it is the
    default.

    Ownership contract (mirrors ``MultiHeadAttention.qk_mod``): an injected
    projection is *not* a child of the attention module. The caller owns it
    and must register it in its own module tree so ``.to(device)`` and
    ``state_dict`` include it. Built-in (``projection=None``) projections are
    registered children of the attention layer.

    Parameters
    ----------
    num_heads : int
        Number of heads. Only ``none``/``kv`` layouts are per-head.
    proj_k : int
        The compressed sequence length :math:`k` (paper notation).
    max_seq_len : int
        Maximum key/value sequence length :math:`n` the matrices are sized
        for. Longer inputs raise ``ValueError``.
    sharing : str, default="layerwise"
        One of ``"none"``, ``"headwise"``, ``"kv"``, ``"layerwise"``
        (default) — see the sharing table above.
    """

    def __init__(
        self,
        num_heads: int,
        proj_k: int,
        max_seq_len: int,
        sharing: str = "layerwise",
    ):
        super().__init__()
        if sharing not in ("none", "headwise", "kv", "layerwise"):
            raise ValueError(f"sharing must be 'none' | 'headwise' | 'kv' | 'layerwise', got {sharing!r}")

        self.num_heads = num_heads
        self.proj_k = proj_k
        self.max_seq_len = max_seq_len
        self.sharing = sharing

        per_head = sharing in ("none", "kv")
        shape = (num_heads, max_seq_len, proj_k) if per_head else (max_seq_len, proj_k)
        self.E = nn.Parameter(torch.empty(*shape))
        if sharing in ("none", "headwise"):
            self.F = nn.Parameter(torch.empty(*shape))
        _init_proj(self.E)
        if sharing in ("none", "headwise"):
            _init_proj(self.F)

    def _contract(self, x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        """Contract the key/value sequence dimension with a projection.

        ``x`` is ``(batch, heads, seq_len, head_size)``, ``proj`` is the
        projection matrix laid out as ``(seq_len, proj_k)`` (shared across
        heads) or ``(num_heads, seq_len, proj_k)`` (per-head). Returns
        ``(batch, heads, proj_k, head_size)``.
        """
        if proj.dim() == 3:
            return torch.einsum("bhld,hlk->bhkd", x, proj)
        return torch.einsum("bhld,lk->bhkd", x, proj)

    def _project(self, x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        if seq_len > self.max_seq_len:
            raise ValueError(f"key sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        return self._contract(x, proj[..., :seq_len, :])

    def project_key(self, x: torch.Tensor) -> torch.Tensor:
        """Contract ``x`` ``(batch, heads, seq_len, head_size)`` with ``E``."""
        return self._project(x, self.E)

    def project_value(self, x: torch.Tensor) -> torch.Tensor:
        """Contract ``x`` with ``F`` (or ``E`` under ``kv``/``layerwise``)."""
        proj = self.E if self.sharing in ("kv", "layerwise") else self.F
        return self._project(x, proj)


class LinformerAttention(nn.Module):
    """Low-rank (linear-complexity) self/cross-attention — the Linformer core.

    From "Linformer: Self-Attention with Linear Complexity" (Wang et al.,
    2020, arXiv:2006.04768). The key/value sequences are contracted from
    ``n`` to ``k`` tokens *before* the attention product:

    .. math::

        \\text{head}_i = \\text{Attention}(Q W_i^Q,\\; E_i\\, K W_i^K,\\; F_i\\, V W_i^V)

    where :math:`E_i, F_i \\in \\mathbb{R}^{k \\times n}` are learned
    sequence-dimension projections. The score tensor becomes
    ``(query_seq_len, k)`` instead of ``(query_seq_len, key_seq_len)``, so
    time and memory drop from :math:`O(n^2)` to :math:`O(nk)` — linear in the
    sequence length for a fixed :math:`k`. Projecting K and V down also cuts
    the K/V cache and, per the paper, matches standard attention quality at
    moderate compression (e.g. ``k=128`` at ``n=512``, ``k=256`` at
    ``n=1024``).

    Since :math:`E_i (K W_i^K) = (E_i K) W_i^K`, the projection and the K/V
    linears could be reordered, but the paper's order (project after the
    linear) is kept. The projection matrices live in a
    :class:`LinformerSeqProjection` — either created internally (a registered
    child) or injected via ``projection=``. Injecting one shared projection
    into several layers reproduces the paper's cross-layer ``layerwise``
    sharing (only :math:`E`/:math:`F` tied; Q/K/V linears stay per-layer).

    Limitations (inherent to the mechanism, not implementation bugs):

    * Bidirectional only. The projection mixes all key positions into ``k``
      slots, so per-position causal/attention masks cannot be applied — the
      paper is an encoder (RoBERTa-style) method. ``mask`` or
      ``past_key_value`` passed to :meth:`forward` raise
      ``NotImplementedError``.
    * No KV-cache decoding — the compressed cache has no per-token
      positional structure.
    * Fixed ``max_seq_len`` budget for the key/value side; shorter inputs
      slice ``E``/``F``.

    Tensors use the ``(batch, seq, embed)`` layout, mirroring
    :class:`MultiHeadAttention`. The module is dependency-free (plain torch)
    and ``torch.compile``-friendly (static ``max_seq_len`` shape).

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
    proj_k : int
        The compressed sequence length :math:`k` (paper notation). Ranks
        ``n`` down to ``k``; ``proj_k >= max_seq_len`` degenerates to plain
        attention with no compression benefit. Ignored when ``projection`` is
        given (the projection carries its own ``proj_k``).
    max_seq_len : int
        Maximum key/value sequence length :math:`n` the internal projection
        is sized for. Ignored when ``projection`` is given.
    sharing : str, default="layerwise"
        Sharing scheme of the *internal* projection — see
        :class:`LinformerSeqProjection`. Ignored when ``projection`` is
        given.
    query_in_size : int | None, default=None
        Embedding size of the query input. ``None`` falls back to
        ``in_size`` (cross-attention support).
    attn_p : float, default=0.0
        Dropout probability applied to attention weights in training mode.
    projection : LinformerSeqProjection | None, default=None
        Sequence-dimension projection to use. ``None`` builds an internal
        one from ``proj_k``/``max_seq_len``/``sharing``. When given, it is
        *not* registered as a child (the caller owns it — same contract as
        ``MultiHeadAttention.qk_mod``), so one instance can be shared across
        several layers for cross-layer ``layerwise`` sharing.
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        num_heads: int,
        out_size: int,
        proj_k: int,
        max_seq_len: int,
        sharing: str = "layerwise",
        query_in_size: int | None = None,
        attn_p: float = 0.0,
        projection: LinformerSeqProjection | None = None,
    ):
        super().__init__()

        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.proj_k = proj_k
        self.max_seq_len = max_seq_len
        self.sharing = sharing
        self.query_in_size = in_size if query_in_size is None else query_in_size
        self.attn_p = attn_p

        if projection is None:
            self.proj = LinformerSeqProjection(num_heads, proj_k, max_seq_len, sharing)
        else:
            # Injected projection: not an nn.Module child. The caller owns it
            # and registers it in its own tree (device, state_dict). Mirrors
            # the MultiHeadAttention.qk_mod ownership contract.
            object.__setattr__(self, "proj", projection)

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
        """Apply Linformer attention.

        Parameters
        ----------
        query : torch.Tensor
            ``(batch, query_seq_len, query_in_size)``.
        key : torch.Tensor
            ``(batch, key_seq_len, in_size)``. ``key_seq_len`` must be
            ``<= max_seq_len``.
        value : torch.Tensor
            ``(batch, key_seq_len, in_size)``.
        mask : torch.Tensor | None, default=None
            Unsupported — projecting key/value over the sequence dimension
            destroys per-position information. Raises ``NotImplementedError``.
        past_key_value : tuple[torch.Tensor, torch.Tensor] | None, default=None
            Unsupported — no KV cache for the compressed keys/values. Raises
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
                "Linformer attention is bidirectional; causal masks and KV cache "
                "are not supported (the sequence-dim projection loses position structure)"
            )

        q = _split_heads(self.query_matrix(query), batch, self.num_heads, self.head_size, q_len)
        k = _split_heads(self.key_matrix(key), batch, self.num_heads, self.head_size, k_len)
        v = _split_heads(self.value_matrix(value), batch, self.num_heads, self.head_size, k_len)

        k_proj = self.proj.project_key(k)
        v_proj = self.proj.project_value(v)

        scores = q @ k_proj.transpose(-2, -1) / math.sqrt(self.head_size)
        weights = F.softmax(scores, dim=-1)
        if self.training and self.attn_p > 0.0:
            weights = F.dropout(weights, self.attn_p, training=True)
        context = weights @ v_proj
        return self.out(context.transpose(1, 2).reshape(batch, q_len, self.num_heads * self.head_size))
