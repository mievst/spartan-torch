"""Multi-head selective SSM mixer (Mamba-2 SSD block)."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _pad_by_size(x: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Zero-pad ``pad_size`` on the seq dim (dim 1) for 3- or 4-dim tensors."""
    if len(x.shape) == 4:
        pad = (0, 0, 0, 0, 0, pad_size, 0, 0)
    else:
        pad = (0, 0, 0, pad_size, 0, 0)
    return F.pad(x, pad)


def _reshape_into_chunks(x: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """Pad seq to a ``chunk_size`` multiple and split into chunks.

    ``(B, L, ...)`` → ``(B, nchunks, chunk_size, ...)``.
    """
    x = _pad_by_size(x, pad_size)
    if len(x.shape) == 3:
        return x.reshape(x.shape[0], -1, chunk_size, x.shape[2])
    return x.reshape(x.shape[0], -1, chunk_size, x.shape[2], x.shape[3])


def _segment_sum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum over the last dim: ``[..., C]`` → ``[..., C, C]``.

    Entry ``[i, j]`` holds ``sum(x[..., i:j+1])`` for ``j >= i`` and ``-inf``
    above the diagonal (causal intra-chunk mask).
    """
    chunk_size = x.size(-1)
    x = x[..., None].expand(*x.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool), diagonal=0)
    return segsum.masked_fill(~mask, -torch.inf)


class _GatedRMSNorm(nn.Module):
    """RMSNorm with SiLU gating: ``RMS(y * silu(gate)) * weight`` (fp32)."""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = y.dtype
        y = y.to(torch.float32) * F.silu(gate.to(torch.float32))
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * y).to(dtype)


def _ssd_chunked(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    previous_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive SSD (chunked dual form): ``(B, L, H, P)`` in/out.

    ``x``/``B``/``C`` are float32 ``(B, L, H, P)`` / ``(B, L, H, N)`` /
    ``(B, L, H, N)``; ``dt`` is ``(B, L, H)``; ``A``/``D`` are ``(H,)``.
    ``previous_states`` is ``(B, H, P, N)`` (chunked prefill continuation)
    or ``None``. Returns ``(y (B, L, H, P), last_state (B, H, P, N))`` —
    the ``D`` skip is included, gating/norm are the caller's job.
    """
    batch, seqlen, nheads, headdim = x.shape
    nstates = B.shape[-1]
    pad_size = (chunk_size - seqlen % chunk_size) % chunk_size

    d_residual = D[..., None] * _pad_by_size(x, pad_size)
    x = x * dt[..., None]
    a = A.to(x.dtype) * dt

    x, a, B, C = [_reshape_into_chunks(t, pad_size, chunk_size) for t in (x, a, B, C)]
    # A: (B, nchunks, cs, H) -> (B, H, nchunks, cs); x/B/C: (B, nchunks, cs, H, P/N).
    a = a.permute(0, 3, 1, 2)
    a_cumsum = torch.cumsum(a, dim=-1)

    # 1. Intra-chunk (diagonal blocks): causal attention-like term.
    decay = torch.exp(_segment_sum(a))
    g = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
    m = (g[..., None] * decay.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
    y_diag = (m[..., None] * x[:, :, None]).sum(dim=3)

    # 2. Per-chunk states (B terms of the off-diagonal factorization).
    decay_states = torch.exp(a_cumsum[:, :, :, -1:] - a_cumsum)
    b_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (b_decay[..., None, :] * x[..., None]).sum(dim=2)  # (B, nchunks, H, P, N)

    # 3. Inter-chunk recurrence (A terms).
    if previous_states is not None:
        previous = previous_states[:, None].to(dtype=states.dtype, device=states.device)
    else:
        previous = torch.zeros_like(states[:, :1])
    states = torch.cat([previous, states], dim=1)
    decay_chunk = torch.exp(_segment_sum(F.pad(a_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, last_state = new_states[:, :-1], new_states[:, -1]

    # 4. States → outputs (C terms).
    state_decay_out = torch.exp(a_cumsum)
    y_off = (C[..., None, :] * states[:, :, None, ...]).sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None]

    y = y_diag + y_off  # (B, nchunks, cs, H, P)
    y = y.reshape(batch, -1, nheads, headdim)
    y = y + d_residual
    if pad_size > 0:
        y = y[:, :seqlen, :, :]
    _ = nstates
    return y, last_state


def _ssd_step(
    x: torch.Tensor,
    dt_raw: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    dt_limit: tuple[float, float],
    ssm_state: torch.Tensor,
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token SSD step: ``(B, 1, H, P)`` in/out.

    ``x`` is ``(B, 1, E)`` float32, ``B``/``C`` are ``(B, 1, G*N)`` float32,
    ``ssm_state`` is ``(B, H, P, N)``. Returns ``(y (B, 1, E), ssm_state)``.
    """
    batch = x.shape[0]
    nheads = A.shape[0]
    headdim = x.shape[-1] // nheads
    dt = dt_raw[:, 0, :][:, None, ...].transpose(1, 2).expand(batch, dt_raw.shape[-1], headdim)
    dt = F.softplus(dt + dt_bias[..., None].expand_as(dt).to(dt.dtype))
    dt = torch.clamp(dt, dt_limit[0], dt_limit[1])
    a = A[..., None, None].expand(nheads, headdim, ssm_state.shape[-1]).to(torch.float32)
    da = torch.exp(dt[..., None] * a)

    b = B.reshape(batch, n_groups, -1)[..., None, :]
    b = b.expand(batch, n_groups, nheads // n_groups, b.shape[-1]).contiguous().reshape(batch, -1, b.shape[-1])
    db = dt[..., None] * b[..., None, :]
    xh = x.reshape(batch, -1, headdim)
    dbx = db * xh[..., None]

    ssm_state = ssm_state * da + dbx
    c = C.reshape(batch, n_groups, -1)[..., None, :]
    c = c.expand(batch, n_groups, nheads // n_groups, c.shape[-1]).contiguous().reshape(batch, -1, c.shape[-1])
    y = torch.bmm(
        ssm_state.to(c.dtype).reshape(batch * nheads, headdim, -1),
        c.reshape(batch * nheads, -1, 1),
    ).reshape(batch, nheads, headdim)
    d = D[..., None].expand(nheads, headdim)
    y = y + xh * d
    return y.reshape(batch, 1, -1), ssm_state


class Mamba2Mixer(nn.Module):
    """Multi-head selective SSM mixer (Mamba-2 SSD block).

    Unlike :class:`~spartan_torch.MambaMixer` (one big S6 state), the
    expanded dim is split into ``num_heads`` heads of ``head_dim`` sharing a
    scalar per-head decay ``A`` — the Structured State Space Duality form,
    computed here with the chunked naive algorithm (intra-chunk causal
    attention + inter-chunk recurrence) instead of a Python loop:

    ``in_proj`` emits ``[gate | x+B+C | dt]`` (the two leading ``d_mlp``
    slices are always empty — legacy upstream layout, kept for key
    compatibility); ``x``/``B``/``C`` pass a causal depthwise conv, the SSD
    core mixes them per head, and a gated RMSNorm + ``out_proj`` finish the
    block. Tensors use the ``(batch, seq, embed)`` layout.

    The mixer is causal (output ``t`` never sees inputs ``> t``) with an
    ``O(H·P·N)`` recurrent state. ``intermediate_size`` must equal
    ``num_heads * head_dim``.

    Backends: ``use_fast_path=True`` (default) + CUDA + ``mamba-ssm``
    installed → fused ``mamba_chunk_scan_combined``; otherwise pure
    PyTorch (this path is experimental — no coverage in this repo, which
    runs CPU-only). No hard dependencies (lazy import in ``forward``).

    Parameters
    ----------
    d_model : int
        Embedding size of the input/output.
    d_state : int, default=128
        SSM state size per group (``N``).
    d_conv : int, default=4
        Width of the causal depthwise convolution.
    expand : int, default=2
        Block expansion factor (``intermediate = expand * d_model``).
    num_heads : int, default=8
        Number of SSD heads (``H``).
    head_dim : int, default=64
        Per-head dim (``P``); ``expand * d_model`` must equal
        ``num_heads * head_dim``.
    n_groups : int, default=1
        Number of ``B``/``C`` groups shared across heads (GQA-style).
    dt_rank : int | str, default="auto"
        Unused by Mamba-2 (kept for a uniform constructor); ``"auto"``
        resolves to ``ceil(d_model / 16)`` like Mamba-1.
    dt_min : float, default=0.001
    dt_max : float, default=0.1
    dt_floor : float, default=1e-4
        Init range/floor of ``softplus(dt_bias)``.
    dt_limit : tuple[float, float], default=(0.0, float("inf"))
        Runtime clamp of ``dt`` after softplus.
    conv_bias : bool, default=True
        Bias of the depthwise convolution.
    bias : bool, default=False
        Bias of the input/output projections.
    norm_eps : float, default=1e-5
        Epsilon of the gated output RMSNorm.
    chunk_size : int, default=64
        SSD chunk size (reference default 256; smaller is friendlier to
        short CPU sequences and gives identical math).
    use_fast_path : bool, default=True
        Try the fused CUDA kernel when available; pure PyTorch otherwise.

    References
    ----------
    "Transformers are SSMs: Generalized Models and Efficient Algorithms
    Through Structured State Space Duality" (Dao & Gu, 2024,
    arXiv:2405.21060).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 8,
        head_dim: int = 64,
        n_groups: int = 1,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_floor: float = 1e-4,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        conv_bias: bool = True,
        bias: bool = False,
        norm_eps: float = 1e-5,
        chunk_size: int = 64,
        use_fast_path: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        intermediate = int(expand * d_model)
        if intermediate != num_heads * head_dim:
            raise ValueError(
                f"expand * d_model ({intermediate}) must equal num_heads * head_dim ({num_heads * head_dim})"
            )
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.intermediate_size = intermediate
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.dt_limit = tuple(dt_limit)
        self.chunk_size = chunk_size
        self.use_fast_path = use_fast_path

        self.conv_dim = intermediate + 2 * n_groups * d_state
        self.in_proj = nn.Linear(d_model, intermediate + self.conv_dim + num_heads, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        dev = torch.device(device) if device is not None else None
        if dev is not None and dev.type == "meta":
            self.dt_bias = nn.Parameter(torch.empty(num_heads, device="meta", dtype=torch.float32))
            self.A_log = nn.Parameter(torch.empty(num_heads, device="meta", dtype=torch.float32))
            self.D = nn.Parameter(torch.empty(num_heads, device="meta", dtype=torch.float32))
        else:
            self.dt_bias = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
            self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
            self.D = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
            if dev is not None:
                self.dt_bias = nn.Parameter(self.dt_bias.to(dev))
                self.A_log = nn.Parameter(self.A_log.to(dev))
                self.D = nn.Parameter(self.D.to(dev))
            with torch.no_grad():
                self.A_log.copy_(torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32)))
                self.D.copy_(torch.ones(num_heads, dtype=torch.float32))
                dt = torch.exp(
                    torch.rand(num_heads, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                ).clamp(min=dt_floor)
                self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        self.norm = _GatedRMSNorm(intermediate, eps=norm_eps)
        self.out_proj = nn.Linear(intermediate, d_model, bias=bias, **factory_kwargs)

    def init_cache(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate a zero recurrent cache.

        Returns ``(conv_state (B, conv_dim, W), ssm_state (B, H, P, N))``:
        the last ``d_conv`` pre-convolution inputs and the fp32 SSD state.
        """
        dev = self.in_proj.weight.device if device is None else device
        dt = self.in_proj.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(batch_size, self.conv_dim, self.d_conv, device=dev, dtype=dt)
        ssm_state = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.d_state, device=dev, dtype=torch.float32
        )
        return conv_state, ssm_state

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply the SSD mixer.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``. The full sequence on the prefill
            step, only the new token(s) when ``cache`` is given.
        cache : tuple[torch.Tensor, torch.Tensor] | None, default=None
            ``(conv_state, ssm_state)`` from :meth:`init_cache` or a previous
            call — states of all tokens before ``x``.
        mask : torch.Tensor | None, default=None
            Bool tensor, ``True`` = masked out (padding), broadcastable to
            ``(batch, seq_len)``.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            ``(output, cache)``: the mixer output ``(batch, seq_len,
            d_model)`` and the updated ``(conv_state, ssm_state)``.
        """
        batch, seqlen, _ = x.shape
        dtype = x.dtype

        if mask is not None:
            if mask.dtype != torch.bool:  # pragma: no cover
                raise ValueError("mask must be a bool tensor with True = masked out")
            keep = (~mask).to(dtype)
            if keep.dim() == 1:
                keep = keep.unsqueeze(0)
            x = x * keep.unsqueeze(-1).expand(batch, seqlen, x.size(-1))

        proj = self.in_proj(x)
        d_mlp = (proj.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.d_state - self.num_heads) // 2
        _, _, gate, hbc, dt_raw = proj.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )
        raw = hbc.transpose(1, 2)  # (B, conv_dim, L)

        if cache is not None:
            conv_state, ssm_state = cache
            if self.d_conv > 1:
                window = torch.cat([conv_state[:, :, -(self.d_conv - 1) :], raw], dim=-1)
            else:
                window = raw
            hidden_c = F.silu(F.conv1d(window, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim))
            hidden_c = hidden_c[:, :, -seqlen:].transpose(1, 2)
            conv_state = torch.cat([conv_state, raw], dim=-1)[:, :, -self.d_conv :]
            ssm_state = ssm_state.to(torch.float32)
        else:
            # Bitwise the reference prefill line (same op, same summation order).
            hidden_c = F.silu(self.conv1d(raw)[..., :seqlen]).transpose(1, 2)
            if seqlen < self.d_conv:
                conv_state = F.pad(raw, (self.d_conv - seqlen, 0))
            else:
                conv_state = raw[:, :, -self.d_conv :]
            ssm_state = torch.zeros(
                batch, self.num_heads, self.head_dim, self.d_state, device=x.device, dtype=torch.float32
            )

        if mask is not None:
            hidden_c = hidden_c * keep.unsqueeze(-1).expand_as(hidden_c)

        xs, Bs, Cs = torch.split(
            hidden_c,
            [self.intermediate_size, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())

        if self.use_fast_path and x.is_cuda:
            fast = self._cuda_fast_forward(xs, dt_raw, Bs, Cs, gate, A, ssm_state if cache is not None else None)
            if fast is not None:
                scan, ssm_state = fast
                return self.out_proj(scan.to(dtype)), (conv_state, ssm_state)

        if cache is not None and seqlen == 1:
            y, ssm_state = _ssd_step(
                xs.float(), dt_raw, Bs.float(), Cs.float(), A, self.D, self.dt_bias,
                self.dt_limit, ssm_state, self.n_groups,
            )
        else:
            dt = F.softplus(dt_raw.float() + self.dt_bias.float()).clamp(*self.dt_limit)
            xh = xs.reshape(batch, seqlen, -1, self.head_dim).float()
            Bh = Bs.reshape(batch, seqlen, -1, self.d_state).float()
            Ch = Cs.reshape(batch, seqlen, -1, self.d_state).float()
            rep = self.num_heads // self.n_groups
            if rep > 1:
                Bh = Bh.repeat_interleave(rep, dim=2)
                Ch = Ch.repeat_interleave(rep, dim=2)
            y, ssm_state = _ssd_chunked(
                xh, dt, A, Bh, Ch, self.D, self.chunk_size,
                previous_states=ssm_state if cache is not None else None,
            )
        scan = self.norm(y.reshape(batch, seqlen, -1), gate)
        return self.out_proj(scan.to(dtype)), (conv_state, ssm_state)

    def _cuda_fast_forward(
        self,
        xs: torch.Tensor,
        dt_raw: torch.Tensor,
        Bs: torch.Tensor,
        Cs: torch.Tensor,
        gate: torch.Tensor,
        A: torch.Tensor,
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fused CUDA SSD via ``mamba-ssm`` (experimental, needs kernels).

        Returns ``None`` when the kernels are not installed so the caller
        falls back to pure PyTorch. Runtime kernel errors propagate (only
        the missing-dependency case falls back).
        """
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: PLC0415
        except ImportError:
            return None
        batch, seqlen, _ = xs.shape
        dt_limit = {} if self.dt_limit == (0.0, float("inf")) else {"dt_limit": self.dt_limit}
        out = mamba_chunk_scan_combined(
            xs.reshape(batch, seqlen, -1, self.head_dim),
            dt_raw,
            A,
            Bs.reshape(batch, seqlen, self.n_groups, -1),
            Cs.reshape(batch, seqlen, self.n_groups, -1),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=None,
            return_final_states=True,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            initial_states=initial_states,
            **dt_limit,
        )
        scan, ssm_state = out
        scan = self.norm(scan.reshape(batch, seqlen, -1), gate)
        return scan, ssm_state


#: Mixer-level block name (same module; HF calls the norm+residual wrapper Mamba2Block).
Mamba2Block = Mamba2Mixer

__all__ = ["Mamba2Block", "Mamba2Mixer"]
