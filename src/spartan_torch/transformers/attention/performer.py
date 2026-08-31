import math

import torch
from torch import nn
import torch.nn.functional as F

from .mha import _split_heads


def gaussian_orthogonal_random_matrix(
    nb_rows: int,
    nb_columns: int,
    scaling: int = 0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Random matrix whose rows are (block-)orthogonal unit vectors.

    Performer's low-variance projection: instead of iid Gaussian rows, rows are
    the rows of orthogonal matrices. With ``scaling=0`` each row's norm is
    sampled from the same chi distribution as an iid ``N(0, I)`` row (keeps the
    Gaussian-kernel estimator unbiased); with ``scaling=1`` every row has the
    exact norm ``sqrt(nb_columns)``.

    ``nb_rows != nb_columns`` is handled by stacking orthogonal square chunks
    (``nb_rows > nb_columns``) or truncating one orthogonal matrix
    (``nb_rows < nb_columns``), so the matrix is still made of orthogonal rows.
    """
    if nb_rows < 1 or nb_columns < 1:
        raise ValueError(f"nb_rows and nb_columns must be positive, got {nb_rows}x{nb_columns}")
    if scaling not in (0, 1):
        raise ValueError(f"scaling must be 0 (sampled norms) or 1 (fixed sqrt(d)), got {scaling}")

    def chunk(cols: int) -> torch.Tensor:
        q, _ = torch.linalg.qr(torch.randn(cols, cols, device=device), mode="reduced")
        return q.t()

    blocks = [chunk(nb_columns) for _ in range(nb_rows // nb_columns)]
    remaining = nb_rows - nb_columns * (nb_rows // nb_columns)
    if remaining > 0:
        blocks.append(chunk(nb_columns)[:remaining])

    rows = torch.cat(blocks, dim=0)  # (nb_rows, nb_columns)

    if scaling == 0:
        multiplier = torch.randn(nb_rows, nb_columns, device=device).norm(dim=1)
    else:
        multiplier = math.sqrt(nb_columns) * torch.ones(nb_rows, device=device)

    matrix = torch.diag(multiplier) @ rows
    if dtype is not None:
        matrix = matrix.to(dtype)
    return matrix


def _softmax_features(
    x: torch.Tensor,
    R: torch.Tensor,
    is_query: bool,
) -> torch.Tensor:
    """Positive random features for the softmax kernel.

    Approximates ``K(x, y) = exp(x·y)`` with
    ``φ(x) = exp(ωᵀx′ − ‖x′‖²/2)``, where ``x′ = x·d^(-1/4)`` scales the dot
    product to the softmax convention (``q′·k′ = q·k/√d``). Everything runs in
    fp32 regardless of the input dtype: the raw features are unbounded above,
    so a fp16 ``exp`` could overflow before the accumulation casts happen —
    computing the features in fp32 keeps them overflow-free and the fp16
    exposure limited to the final cast of the attention output.

    Queries subtract a per-query log-sum max (a row-wise factor that cancels
    in the attention ratio and keeps query features ≤ 1). Keys are left
    unshifted: a per-(batch, head) key shift would couple all positions
    through its magnitude, breaking the causal property that past outputs must
    not depend on future keys. Unlike lucidrains' reference the ``m^-1/2``
    feature scaling is omitted: it cancels in the self-normalized ratio
    anyway and would push the intermediate sums toward fp16 underflow.

    The estimator is unbiased for the softmax kernel, but its variance grows
    with the query/key norms (paper: uniform convergence depends on
    ``‖q‖, ‖k‖``, not on sequence length) — with large post-LayerNorm norms
    more features are needed for a given accuracy.

    ``x`` has layout ``(batch, heads, seq_len, head_size)``, ``R`` is the
    projection ``(num_features, head_size)``. Returns fp32 ``(batch, heads,
    seq_len, num_features)``.
    """
    normalizer = x.shape[-1] ** -0.25

    data_dash = torch.einsum("bhnd,md->bhnm", x.float() * normalizer, R.float())
    diag = ((x * x).sum(-1) * (normalizer**2) / 2.0).float().unsqueeze(-1)

    if is_query:
        data_dash = data_dash - diag - data_dash.amax(dim=-1, keepdim=True).detach()
    else:
        data_dash = data_dash - diag

    return torch.exp(data_dash)


def _bidir_context(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Non-causal FAVOR+: ``φ(Q)(φ(K)ᵀV)`` normalized by ``φ(Q)Σφ(K)``.

    ``q``/``k`` are feature maps ``(batch, heads, seq_len, num_features)``,
    ``v`` is ``(batch, heads, seq_len, head_size)``. The KV product is
    ``(num_features, head_size)`` — independent of ``seq_len``, which is what
    makes Performer linear in the sequence length. The sums run in fp32
    internally (the feature count ``m`` can push a fp16 accumulation past
    overflow) and the result is cast back to ``v``'s dtype.
    """
    out_dtype = v.dtype
    if out_dtype != torch.float32:
        q, k, v = q.float(), k.float(), v.float()

    context = torch.einsum("bhjm,bhje->bhme", k, v)
    k_sum = k.sum(dim=-2)
    num = torch.einsum("bhjm,bhme->bhje", q, context)
    den = torch.einsum("bhjm,bhm->bhj", q, k_sum).unsqueeze(-1)
    return (num / (den + eps)).to(out_dtype)


def _causal_context(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Causal FAVOR+ via chunked running sums.

    The causal prefix sums ``Σ_{j≤i} φ(k_j)`` and ``Σ_{j≤i} φ(k_j)⊗v_j`` are
    materialized for one chunk at a time, so peak memory is
    ``O(chunk_size · num_features · head_size)`` instead of the full
    ``O(seq_len · num_features · head_size)`` (which with
    ``num_features > head_size`` is far too heavy — 8+ GB for
    ``seq_len=16384, m=256``). Math is identical to a global cumsum. The
    running sums run in fp32 (see :func:`_bidir_context`) and the result is
    cast back to ``v``'s dtype.
    """
    out_dtype = v.dtype
    if out_dtype != torch.float32:
        q, k, v = q.float(), k.float(), v.float()
    eps = torch.tensor(eps, dtype=q.dtype, device=q.device)

    outs = []
    acc_k = k[:, :, :1] * 0.0  # (b, h, 1, m)
    acc_kv = torch.zeros_like(k, dtype=v.dtype)[:, :, :1].unsqueeze(-1) * 0.0  # (b, h, 1, m, d)
    for start in range(0, q.size(-2), chunk_size):
        qc = q[:, :, start : start + chunk_size]
        kc = k[:, :, start : start + chunk_size]
        vc = v[:, :, start : start + chunk_size]

        kc_sum = kc.cumsum(dim=-2)
        kvc = torch.einsum("bhjm,bhje->bhjme", kc, vc).cumsum(dim=-3)

        k_full = acc_k + kc_sum
        kv_full = acc_kv + kvc

        num = torch.einsum("bhjm,bhjmd->bhjd", qc, kv_full)
        den = torch.einsum("bhjm,bhjm->bhj", qc, k_full).unsqueeze(-1)
        outs.append(num / (den + eps))

        acc_k = k_full[:, :, -1:]
        acc_kv = kv_full[:, :, -1:]
    return torch.cat(outs, dim=-2).to(out_dtype)


def _key_mask(mask: torch.Tensor, k_len: int, num_heads: int, q_len: int) -> torch.Tensor:
    """Validate a user mask as a per-key (padding) mask and broadcast it.

    Kernel attention cannot apply arbitrary per-pair masks (there is no score
    matrix); only key-level masks make sense — they drop the masked keys'
    features and values from the running sums, which is exactly the padding
    case. Returns a bool ``(batch, 1, 1, k_len)`` tensor.
    """
    if mask.dtype != torch.bool:
        raise NotImplementedError(
            "PerformerAttention supports only boolean key masks (padding); "
            "additive score biases have no kernel equivalent"
        )
    if mask.dim() == 4 and (mask.size(1) != 1 or mask.size(2) != 1):
        raise NotImplementedError(
            "pair-wise masks are not supported by kernel attention; pass a "
            "(batch, 1, 1, key_len) key mask for padding"
        )
    if mask.dim() == 4:
        km = mask
    elif mask.dim() == 3:
        km = mask.unsqueeze(1)
    elif mask.dim() == 2:
        km = mask.unsqueeze(1).unsqueeze(1)
    else:
        raise NotImplementedError(
            f"mask must be (batch, 1, 1, key_len), (batch, 1, key_len) or (batch, key_len), "
            f"got shape {tuple(mask.shape)}"
        )
    if km.size(-1) != k_len:
        raise ValueError(f"key mask length {km.size(-1)} does not match key_seq_len {k_len}")
    return km.to(device=mask.device)


class PerformerAttention(nn.Module):
    """Performer attention (FAVOR+): kernel attention with positive random features.

    From "Rethinking Attention with Performers" (Choromanski et al., 2022,
    arXiv:2009.14794). The softmax kernel ``exp(x·y)`` is estimated by
    positive random features ``φ(x) = exp(ωᵀx − ‖x‖²/2)`` with
    **orthogonal** random vectors (ORF) — lower variance than iid Gaussian
    rows with the same asymptotic cost. Reordering the matrix product as
    ``φ(Q)(φ(K)ᵀV)`` drops attention from :math:`O(n^2)` to
    :math:`O(n \\cdot m \\cdot d)` time and :math:`O(m \\cdot d)` memory
    (``m`` = number of features, independent of ``n``).

    The estimator is unbiased for the softmax kernel, so with enough features
    the output converges to the exact softmax attention of the *same* Q/K/V
    projections — that is the property a Performer *adapter* (see
    :class:`PerformerAdapter`) exploits to convert a trained model. Note that
    the convergence rate depends on the query/key norms (the paper shows the
    number of features needed depends on ``‖q‖, ‖k‖``, not the sequence
    length): large post-LayerNorm norms need more features — hence the paper's
    "minimal fine-tuning recovers accuracy" finding for pretrained-weights
    conversion. The random features are computed in fp32 internally (see
    :func:`_softmax_features`) so the estimator is fp16-friendly.

    Properties:

    * Bidirectional mode is ``O(n·m·d)``; the causal (autoregressive) mode
      uses chunked prefix sums (see :func:`_causal_context`), so memory stays
      bounded regardless of ``seq_len``.
    * The projection ``R`` is a non-trainable buffer, fixed at
      initialization (deterministic, checkpoint-safe). ``feature_redraw_interval``
      optionally redraws it periodically during training — the paper's
      regularization — via a lightweight counter.
    * ``mask`` accepts only **key masks** (padding): ``(batch, 1, 1, key_len)``
      bool. Pair-wise masks and additive biases have no kernel equivalent and
      raise ``NotImplementedError``; ``past_key_value`` (incremental KV cache)
      also raises — there is no per-token cache for a kernel estimator.
    * ``attn_p`` drops the random features (feature-space dropout) in training,
      the kernel analogue of attention-weight dropout.

    Tensors use the ``(batch, seq, embed)`` layout, mirroring
    :class:`MultiHeadAttention`.

    Parameters
    ----------
    in_size : int
        Embedding size of query/key/value inputs.
    head_size : int
        Per-head hidden dim of Q, K, V (and the ORF dimension).
    num_heads : int
        Number of heads.
    out_size : int
        Output embedding size.
    query_in_size : int | None, default=None
        Embedding size of the query input. ``None`` falls back to ``in_size``.
    num_features : int | None, default=None
        Number of random features ``m``. Defaults to
        ``int(head_size * log(head_size))``. Larger ``m`` → tighter
        approximation of softmax at linear cost ``O(n·m·d)``.
    ortho_scaling : int, default=0
        ``0`` samples each ORF row norm from the chi distribution (keeps the
        Gaussian-kernel estimator unbiased), ``1`` fixes every row norm to
        ``sqrt(head_size)``.
    feature_redraw_interval : int | None, default=None
        Redraw ``R`` every this many training steps (paper's regularization).
        ``None`` keeps the projection fixed forever.
    attn_p : float, default=0.0
        Dropout probability on the random features in training mode.
    is_causal : bool, default=False
        Use the causal (prefix-sum) form. Requires ``query_seq_len ==
        ``key_seq_len``.
    eps : float, default=1e-6
        Denominator floor for the normalization.
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        num_heads: int,
        out_size: int,
        query_in_size: int | None = None,
        num_features: int | None = None,
        ortho_scaling: int = 0,
        feature_redraw_interval: int | None = None,
        attn_p: float = 0.0,
        is_causal: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        num_features = int(head_size * math.log(head_size)) if num_features is None else num_features
        if num_features < 1:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if feature_redraw_interval is not None and feature_redraw_interval < 1:
            raise ValueError(f"feature_redraw_interval must be positive or None, got {feature_redraw_interval}")

        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.query_in_size = in_size if query_in_size is None else query_in_size
        self.num_features = num_features
        self.attn_p = attn_p
        self.is_causal = is_causal
        self.feature_redraw_interval = feature_redraw_interval
        self.eps = eps

        self.register_buffer(
            "R",
            gaussian_orthogonal_random_matrix(num_features, head_size, scaling=ortho_scaling),
        )
        if feature_redraw_interval is not None:
            self.register_buffer("_calls_since_redraw", torch.zeros((), dtype=torch.long))

        self.query_matrix = nn.Linear(self.query_in_size, head_size * num_heads, bias=False)
        self.key_matrix = nn.Linear(in_size, head_size * num_heads, bias=False)
        self.value_matrix = nn.Linear(in_size, head_size * num_heads, bias=False)
        self.out = nn.Linear(head_size * num_heads, out_size)

    @torch.no_grad()
    def redraw_projection(self) -> None:
        """Replace ``R`` with a freshly sampled ORF (training-time only)."""
        self.R.copy_(gaussian_orthogonal_random_matrix(self.num_features, self.head_size, dtype=self.R.dtype, device=self.R.device))

    def _maybe_redraw(self) -> None:
        if self.feature_redraw_interval is None or not self.training:
            return
        self._calls_since_redraw += 1
        if self._calls_since_redraw >= self.feature_redraw_interval:
            self.redraw_projection()
            self._calls_since_redraw.zero_()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply Performer attention.

        Parameters
        ----------
        query : torch.Tensor
            ``(batch, query_seq_len, query_in_size)``.
        key : torch.Tensor
            ``(batch, key_seq_len, in_size)``.
        value : torch.Tensor
            ``(batch, key_seq_len, in_size)``.
        mask : torch.Tensor | None, default=None
            Key (padding) mask: bool ``(batch, 1, 1, key_seq_len)`` (or a
            broadcastable variant), ``True`` = padded. Pair-wise masks raise
            ``NotImplementedError``.
        past_key_value : tuple[torch.Tensor, torch.Tensor] | None, default=None
            Unsupported — kernel attention has no per-token KV cache. Raises
            ``NotImplementedError``.

        Returns
        -------
        torch.Tensor
            Attention output ``(batch, query_seq_len, out_size)``.
        """
        if past_key_value is not None:
            raise NotImplementedError(
                "PerformerAttention has no incremental KV cache; pass the full sequence "
                "(kernel attention keeps running feature sums instead of per-token keys)"
            )

        batch = key.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        if self.is_causal and q_len != k_len:
            raise ValueError(f"causal Performer attention requires query_seq_len == key_seq_len, got {q_len} != {k_len}")

        km = _key_mask(mask, k_len, self.num_heads, q_len) if mask is not None else None
        self._maybe_redraw()

        q = _split_heads(self.query_matrix(query), batch, self.num_heads, self.head_size, q_len)
        k = _split_heads(self.key_matrix(key), batch, self.num_heads, self.head_size, k_len)
        v = _split_heads(self.value_matrix(value), batch, self.num_heads, self.head_size, k_len)

        qf = _softmax_features(q, self.R, is_query=True)
        kf = _softmax_features(k, self.R, is_query=False)
        if km is not None:
            # Zero the *feature maps* of masked keys (and their values), not
            # the raw keys: a zero input maps to φ(0) = 1 + eps, i.e. the
            # largest possible features, which would dominate the running sums.
            km_t = km.transpose(-1, -2)
            kf = kf.masked_fill(km_t, 0.0)
            v = v.masked_fill(km_t, 0.0)
        if self.training and self.attn_p > 0.0:
            qf = F.dropout(qf, self.attn_p, training=True)
            kf = F.dropout(kf, self.attn_p, training=True)

        if self.is_causal:
            context = _causal_context(qf, kf, v, self.eps)
        else:
            context = _bidir_context(qf, kf, v, self.eps)

        return self.out(context.transpose(1, 2).reshape(batch, q_len, self.num_heads * self.head_size))
