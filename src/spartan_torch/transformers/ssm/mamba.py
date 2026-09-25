"""Selective state-space (Mamba-1) mixer."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

try:  # Optional parallel scan (torch>=2.9). Absent → sequential loop.
    from torch._higher_order_ops import associative_scan as _associative_scan
except ImportError:  # pragma: no cover
    _associative_scan = None


def _causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Depthwise causal Conv1d + SiLU on a pre-windowed input.

    Used for cached (decode/chunked) steps, where the input window already
    carries the ``kernel_size - 1`` left context, so no padding is needed.
    Takes the last ``seqlen`` outputs — callers pass the window such that
    these align with the current tokens.
    """
    return F.silu(F.conv1d(x, weight, bias, groups=x.size(1)))


def _selective_scan_loop(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential selective scan (Euler discretization, fp32 accumulation).

    ``x``/``dt`` are ``(B, E, L)``, ``A`` is ``(E, N)``, ``B``/``C`` are
    ``(B, L, N)``. ``ssm_state`` is ``(B, E, N)`` fp32. Returns
    ``(scan_output (B, E, L), ssm_state)`` — the ``D`` skip and gate are
    applied by the caller.
    """
    dtype = x.dtype
    A = A.float()
    outs = []
    for t in range(x.size(-1)):
        dt_t = dt[:, :, t].float()  # (B, E)
        dA = torch.exp(A[None] * dt_t[..., None])  # (B, E, N)
        dB = dt_t[..., None] * B[:, t, :][:, None, :].float()  # (B, E, N)
        ssm_state = dA * ssm_state + dB * x[:, :, t].float()[:, :, None]
        outs.append(torch.matmul(ssm_state.to(dtype), C[:, t, :].unsqueeze(-1))[:, :, 0])
    return torch.stack(outs, dim=-1), ssm_state


def _selective_scan_associative(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel selective scan via ``torch.associative_scan``.

    Same math and contract as :func:`_selective_scan_loop`, but the recurrence
    is evaluated with a parallel prefix scan over the time dim instead of a
    Python loop. Materializes ``(B, E, L, N)`` intermediates — faster on
    CUDA, heavier on memory.
    """
    if _associative_scan is None:  # pragma: no cover
        raise RuntimeError("associative_scan is not available in this torch build")
    dtype = x.dtype
    dA = torch.exp(A.float()[None, :, None, :] * dt.float()[:, :, :, None])  # (B, E, L, N)
    dB_u = dt.float()[:, :, :, None] * B.float()[:, None, :, :] * x.float()[:, :, :, None]

    def _combine(left: tuple[torch.Tensor, torch.Tensor], right: tuple[torch.Tensor, torch.Tensor]):
        a_left, b_left = left
        a_right, b_right = right
        return (a_left * a_right, a_right * b_left + b_right)

    combine_mode = "pointwise" if x.device.type in ("cuda", "xpu") else "generic"
    _, all_h = _associative_scan(_combine, (dA, dB_u), dim=2, combine_mode=combine_mode)
    scan = torch.matmul(all_h.permute(0, 2, 1, 3).to(dtype), C.to(dtype).unsqueeze(-1)).squeeze(-1)
    scan = scan.transpose(1, 2)  # (B, L, E) -> (B, E, L)
    return scan, all_h[:, :, -1, :]


class MambaMixer(nn.Module):
    """Selective SSM mixer (Mamba-1 S6 block).

    ``in_proj`` splits the input into a main branch ``x`` and a gate ``z``;
    ``x`` goes through a causal depthwise conv + SiLU, then the selective
    scan with input-dependent ``dt``/``B``/``C`` (Euler discretization):

    .. math::

        h_t = e^{\\Delta_t A} h_{t-1} + \\Delta_t B_t x_t, \\quad
        y_t = C_t h_t + D x_t, \\quad
        out = y \\cdot \\text{SiLU}(z).

    The scan is causal by construction (output ``t`` never sees inputs
    ``> t``) and runs in ``O(L)`` time with an ``O(E·N)`` recurrent state —
    no KV-cache growth. Tensors use the ``(batch, seq, embed)`` layout,
    mirroring :class:`~spartan_torch.MultiHeadAttention`.

    Compute backends (no hard dependencies — ``mamba-ssm`` is imported
    lazily, inside ``forward``):

    * ``use_fast_path=True`` (default) + CUDA + ``mamba-ssm`` installed →
      fused ``selective_scan_fn`` (handles gating and the ``D`` skip
      internally). Otherwise pure PyTorch. The fast path is experimental:
      it has no CPU coverage in this repo.
    * ``use_associative_scan=True`` (default) + a torch shipping
      ``torch._higher_order_ops.associative_scan`` → parallel prefix scan
      instead of the Python loop on cache-free prefills. Numerically
      identical; higher memory. Cached (decode) steps always use the loop:
      the parallel scan can only start from a zero state.

    Honesty boundary: the cache-free prefill matches the HF reference
    bitwise; chunked/decode steps recompute the convolution over a shorter
    window (different summation order), and the recurrence amplifies that
    ``~1e-7`` conv noise to ``~1e-4`` on large models
    (``d_inner=1536``). The reference itself diverges more between its own
    prefill and cache-decode paths (``~9e-4`` on mamba-130m), so this is
    the op's inherent noise floor, not a mapping bug.

    Parameters
    ----------
    d_model : int
        Embedding size of the input/output.
    d_state : int, default=16
        SSM state expansion factor (``N``).
    d_conv : int, default=4
        Width of the causal depthwise convolution.
    expand : int, default=2
        Block expansion factor (``d_inner = expand * d_model``).
    dt_rank : int | str, default="auto"
        Rank of the ``dt`` projection. ``"auto"`` resolves to
        ``ceil(d_model / 16)`` (official default).
    dt_min : float, default=0.001
    dt_max : float, default=0.1
        Range of ``softplus(dt_bias)`` at init.
    dt_init : str, default="random"
        ``"random"`` (uniform) or ``"constant"`` init of ``dt_proj.weight``.
    dt_scale : float, default=1.0
        Scale of the ``dt_proj.weight`` init std (``dt_rank**-0.5``).
    dt_init_floor : float, default=1e-4
        Minimum sampled ``dt`` before the inverse-softplus init.
    conv_bias : bool, default=True
        Bias of the depthwise convolution.
    bias : bool, default=False
        Bias of the input/output projections.
    use_fast_path : bool, default=True
        Try the fused CUDA kernel when available; pure PyTorch otherwise.
    use_associative_scan : bool, default=True
        Parallel prefix scan instead of the sequential loop when available.

    References
    ----------
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    (Gu & Dao, 2024, arXiv:2312.00752).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        use_associative_scan: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.use_fast_path = use_fast_path
        self.use_associative_scan = use_associative_scan

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dev = torch.device(device) if device is not None else None
        if dev is not None and dev.type == "meta":
            self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state, device="meta", dtype=torch.float32))
            self.D = nn.Parameter(torch.empty(self.d_inner, device="meta", dtype=torch.float32))
        else:
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1).contiguous()
            self.A_log = nn.Parameter(A.to(device) if device is not None else A)
            D = torch.ones(self.d_inner, dtype=torch.float32)
            self.D = nn.Parameter(D.to(device) if device is not None else D)
            self._init_dt(dt_init, dt_scale, dt_min, dt_max, dt_init_floor)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def _init_dt(self, dt_init: str, dt_scale: float, dt_min: float, dt_max: float, dt_init_floor: float) -> None:
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:  # pragma: no cover
            raise NotImplementedError(f"dt_init must be 'random' or 'constant', got {dt_init!r}")
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        with torch.no_grad():
            # Inverse-softplus so that softplus(bias) lands in [dt_min, dt_max].
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def init_cache(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate a zero recurrent cache.

        Returns ``(conv_state (B, E, W), ssm_state (B, E, N))``: the last
        ``d_conv`` pre-convolution inputs (right-aligned, zero left-padded)
        and the fp32 SSM state. Feed it as ``cache`` to :meth:`forward` for
        chunked prefill or token-by-token decoding.
        """
        dev = self.in_proj.weight.device if device is None else device
        dt = self.in_proj.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=dev, dtype=dt)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=dev, dtype=torch.float32)
        return conv_state, ssm_state

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply the selective SSM mixer.

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
            ``(batch, seq_len)``. Masked positions contribute zeros to the
            convolution and the scan.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            ``(output, cache)``: the mixer output ``(batch, seq_len,
            d_model)`` and the updated ``(conv_state, ssm_state)`` to feed
            the next decode step.
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

        xz = self.in_proj(x).transpose(1, 2)  # (B, 2E, L)
        raw, gate = xz.chunk(2, dim=1)  # (B, E, L) each

        if cache is not None:
            conv_state, ssm_state = cache
            if self.d_conv > 1:
                window = torch.cat([conv_state[:, :, -(self.d_conv - 1) :], raw], dim=-1)
            else:
                window = raw
            hidden = _causal_depthwise_conv1d(window, self.conv1d.weight, self.conv1d.bias)
            hidden = hidden[:, :, -seqlen:]
            conv_state = torch.cat([conv_state, raw], dim=-1)[:, :, -self.d_conv :]
            ssm_state = ssm_state.to(torch.float32)
        else:
            # Bitwise the reference prefill line (same op, same summation
            # order): conv1d with built-in left padding, then truncate.
            hidden = F.silu(self.conv1d(raw)[..., :seqlen])
            if seqlen < self.d_conv:
                conv_state = F.pad(raw, (self.d_conv - seqlen, 0))
            else:
                conv_state = raw[:, :, -self.d_conv :]
            ssm_state = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)

        if mask is not None:
            hidden = hidden * keep.unsqueeze(1).expand_as(hidden)

        ssm_params = self.x_proj(hidden.transpose(1, 2))  # (B, L, R+2N)
        dt_raw, B, C = torch.split(ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        if self.use_fast_path and x.is_cuda:
            fast = self._cuda_fast_forward(hidden, gate, dt_raw, B, C, A)
            if fast is not None:
                y, ssm_state = fast
                return self.out_proj(y.transpose(1, 2).to(dtype)), (conv_state, ssm_state)

        dt = F.softplus(self.dt_proj(dt_raw)).transpose(1, 2)  # (B, E, L)
        # The parallel scan always starts from a zero state, so cached
        # (non-zero start state) steps must use the sequential loop —
        # same gating as HF (associative only when cache is None).
        if cache is None and self.use_associative_scan and _associative_scan is not None:
            scan, ssm_state = _selective_scan_associative(hidden, dt, A, B, C, ssm_state)
        else:
            scan, ssm_state = _selective_scan_loop(hidden, dt, A, B, C, ssm_state)
        y = scan + hidden * self.D.to(dtype)[None, :, None]
        y = y * F.silu(gate)
        return self.out_proj(y.transpose(1, 2)), (conv_state, ssm_state)

    def _cuda_fast_forward(
        self,
        hidden: torch.Tensor,
        gate: torch.Tensor,
        dt_raw: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        A: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fused CUDA scan via ``mamba-ssm`` (experimental, needs kernels).

        Returns ``None`` when the kernels are not installed so the caller
        falls back to pure PyTorch. Runtime kernel errors propagate (only
        the missing-dependency case falls back).
        """
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: PLC0415
        except ImportError:
            return None
        dt = self.dt_proj(dt_raw).transpose(1, 2)
        out = selective_scan_fn(
            hidden,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=gate,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=True,
        )
        scan, ssm_state = out
        return scan, ssm_state


#: Mixer-level block name from the plan (same module, HF calls the
#: norm+residual wrapper MambaBlock — ours stays a pure mixer).
MambaBlock = MambaMixer

__all__ = ["MambaBlock", "MambaMixer"]
