"""Inference-first selective SSM mixer (Mamba-3 SISO/MIMO block)."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _heavy_tail(x: torch.Tensor) -> torch.Tensor:
    """Heavy-tail activation for the data-dependent state matrix.

    ``1 + x`` for ``x >= 0``, ``1 / (1 - x)`` otherwise — positive,
    continuous and differentiable at zero. Stabilizes WSD training at
    higher learning rates.
    """
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def _apply_rope_adjacent(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """RoPE with adjacent pairs ``(2i, 2i+1)`` (NeoX-style, Mamba-3 MIMO)."""
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def _apply_rope_pairwise(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """RoPE with split-half pairs ``(i, i+S/2)`` (RoFormer-style, Mamba-3 SISO)."""
    half = angles.shape[-1]
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class _RMSNorm(nn.Module):
    """Plain RMSNorm over the last dim (for the B/C projections)."""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * y).to(x.dtype)


class _HeadwiseGatedRMSNorm(nn.Module):
    """Gated RMSNorm per ``(head, head_dim)`` group (Mamba-3 out-proj norm).

    ``norm(y, z)`` with ``y``/``z`` shaped ``(..., H*P)``: gates with
    ``silu(z)``, normalizes each head over ``P``, scales with ``weight``.
    """

    def __init__(self, size: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.head_dim = head_dim
        self.eps = eps

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dtype = y.dtype
        h = y.shape[-1] // self.head_dim
        y = y.float().reshape(*y.shape[:-1], h, self.head_dim)
        y = y * F.silu(z.float().reshape(*z.shape[:-1], h, self.head_dim))
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.reshape(h, self.head_dim) * y).reshape(*y.shape[:-2], -1).to(dtype)


def _trapezoidal_scan_siso(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    decay: torch.Tensor,
    dt: torch.Tensor,
    tr: torch.Tensor,
    D: torch.Tensor,
    ssm_state: torch.Tensor,
    bx_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exponential-trapezoidal selective scan (SISO), fp32 accumulation.

    ``x`` is ``(B, L, H, P)``; ``B``/``C`` are ``(B, L, H, N)`` (already
    rotated); ``decay``/``dt``/``tr`` are ``(B, L, H)`` with
    ``decay = exp(A*dt)``, ``tr = sigmoid(trap)``; ``D`` is ``(H,)``.
    ``ssm_state``/``bx_prev`` are ``(B, H, P, N)``. Returns
    ``(y (B, L, H, P), ssm_state, bx_prev)`` — gating is the caller's job.
    """
    dtype = x.dtype
    outs = []
    for t in range(x.size(1)):
        bx = torch.einsum("bhp,bhn->bhpn", x[:, t].float(), B[:, t].float())
        w = tr[:, t, :, None, None]
        blend = (1.0 - w) * bx + w * 0.5 * (bx + bx_prev)
        ssm_state = decay[:, t, :, None, None] * ssm_state + dt[:, t, :, None, None] * blend
        y_t = torch.einsum("bhn,bhpn->bhp", C[:, t].float(), ssm_state.to(dtype))
        outs.append(y_t + D.to(dtype)[None, :, None] * x[:, t].to(dtype))
        bx_prev = bx
    return torch.stack(outs, dim=1), ssm_state, bx_prev


def _trapezoidal_scan_mimo(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    decay: torch.Tensor,
    dt: torch.Tensor,
    tr: torch.Tensor,
    D: torch.Tensor,
    mimo_x: torch.Tensor,
    ssm_state: torch.Tensor,
    bx_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exponential-trapezoidal selective scan (MIMO), fp32 accumulation.

    ``x`` is ``(B, L, H, P)``; ``B``/``C`` are ``(B, L, R, H, N)``;
    ``decay``/``dt``/``tr`` are ``(B, L, H)``; ``D`` is ``(H,)``;
    ``mimo_x`` is ``(H, R, P)``. The state ``(B, H, P, N)`` is shared
    across ranks (cross-rank interaction); per-rank ``B·x`` terms are
    summed into it. ``bx_prev`` is ``(B, R, H, P, N)``. Returns ``(y (B,
    L, R, H, P) pre-gate/pre-norm, ssm_state, bx_prev)`` — gating, norm
    and the ``mimo_o`` mix are the caller's job.
    """
    dtype = x.dtype
    outs = []
    for t in range(x.size(1)):
        xr = torch.einsum("bhp,hrp->bhrp", x[:, t].float(), mimo_x.float())
        bx = torch.einsum("bhrp,brhn->brhpn", xr, B[:, t].float())
        w = tr[:, t, :][:, None, :, None, None]
        blend = (1.0 - w) * bx + w * 0.5 * (bx + bx_prev)
        ssm_state = (
            decay[:, t, :, None, None] * ssm_state
            + dt[:, t, :, None, None] * blend.sum(dim=1)
        )
        y_r = torch.einsum("brhn,bhpn->brhp", C[:, t].float(), ssm_state.to(dtype))
        skip = D.to(dtype)[None, None, :, None] * xr.to(dtype).transpose(1, 2)
        outs.append((y_r + skip).unsqueeze(1))
        bx_prev = bx
    return torch.cat(outs, dim=1), ssm_state, bx_prev


class Mamba3Mixer(nn.Module):
    """Inference-first selective SSM mixer (Mamba-3 SISO/MIMO block).

    On top of :class:`~spartan_torch.Mamba2Mixer` this adds three changes:

    * exponential-trapezoidal discretization — the input enters through a
      ``trap``-gated blend of the current and previous ``B·x`` terms
      (Euler at ``trap → 0``, full trapezoid at ``trap → 1``), which also
      removes the need for the explicit causal convolution;
    * data-dependent RoPE on ``B``/``C`` (complex-valued state without
      complex arithmetic) for state tracking — adjacent pairs in MIMO
      mode, split-half pairs in SISO mode;
    * MIMO formulation — ``mimo_rank`` streams share one ``(H, N)`` state
      for higher decode arithmetic intensity at the same state size.

    ``in_proj`` emits ``[z | x | B | C | dt | A | trap | angles]``;
    ``B``/``C`` pass RMSNorm, a learnable bias and the rotation; the scan
    output is gated (SISO: ``silu(z)``; MIMO: per-rank ``silu(z_r)`` with
    an optional headwise norm) and mixed back with ``out_proj``. Tensors
    use the ``(batch, seq, embed)`` layout. No convolution anywhere.

    The reference kernels (Triton SISO/MIMO, CuTe decode) are not
    wired — this is the pure-PyTorch path (CPU-friendly, exact math).
    ``use_fast_path`` is accepted for API symmetry and currently always
    takes the pure path.

    Honesty boundary: no runnable reference exists in a CPU-only
    environment (official kernels are Linux/CUDA-only,
    ``transformers`` ships no Mamba-3), so output parity against the
    reference is unverified. Covered instead: strict-loading the real
    ``ib-ssm/mamba3-370M-10BT`` checkpoint (key map proven on real
    weights) plus step-vs-prefill self-agreement on those weights.
    Three details follow the official module structure but their exact
    kernel semantics is assumed from secondary sources: SISO RoPE uses
    split-half pairs (MIMO adjacent — ``rotate_pairwise = not is_mimo``
    in the official code), the ``B``/``C`` bias is added before rotation,
    and the unfused MIMO path (per-rank norm + ``mimo_o`` mix) is taken
    as numerically identical to the fused kernel.

    Parameters
    ----------
    d_model : int
        Embedding size of the input/output.
    d_state : int, default=128
        SSM state size (``N``).
    expand : int, default=2
        Block expansion factor (``d_inner = expand * d_model``).
    head_dim : int, default=64
        Per-head dim (``P``); ``expand * d_model`` must be divisible by it.
    n_groups : int, default=1
        Number of ``B``/``C`` groups broadcast across heads.
    rope_fraction : float, default=0.5
        Fraction of ``d_state`` under rotation (``0.5`` or ``1.0``).
    dt_min : float, default=0.001
    dt_max : float, default=0.1
    dt_floor : float, default=1e-4
        Init range/floor of ``softplus(dt_bias)``.
    a_floor : float, default=1e-4
        Stability floor: ``A <= -a_floor`` elementwise.
    is_outproj_norm : bool, default=False
        Headwise gated RMSNorm on the per-rank outputs (MIMO).
    is_mimo : bool, default=False
        MIMO formulation with ``mimo_rank`` streams (else SISO).
    mimo_rank : int, default=4
        Number of MIMO streams (forced to 1 in SISO mode).
    conv_bias : bool, default=True
        Accepted for constructor symmetry with Mamba-1/2 (no conv here).
    bias : bool, default=False
        Accepted for constructor symmetry (projections are bias-free).
    use_fast_path : bool, default=True
        Reserved for the fused CUDA kernels; pure path is used always.

    References
    ----------
    "Mamba-3: Improved Sequence Modeling using State Space Principles"
    (Lahoti et al., 2026, arXiv:2603.15569).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_floor: float = 1e-4,
        a_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        d_inner = int(expand * d_model)
        if d_inner % head_dim != 0:
            raise ValueError(f"expand * d_model ({d_inner}) must be divisible by head_dim ({head_dim})")
        if rope_fraction not in (0.5, 1.0):
            raise ValueError(f"rope_fraction must be 0.5 or 1.0, got {rope_fraction!r}")
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_inner
        self.num_heads = d_inner // head_dim
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.a_floor = a_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.use_fast_path = use_fast_path

        split = int(d_state * rope_fraction)
        if split % 2 != 0:
            split -= 1
        if split < 2:
            raise ValueError(f"rotated split must hold at least one pair, got {split}")
        self.split_tensor_size = split
        self.num_rope_angles = split // 2

        d_in_proj = (
            2 * d_inner
            + 2 * d_state * n_groups * self.mimo_rank
            + 3 * self.num_heads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

        dev = torch.device(device) if device is not None else None
        if dev is not None and dev.type == "meta":
            extra = {"device": "meta", "dtype": torch.float32}
            self.dt_bias = nn.Parameter(torch.empty(self.num_heads, **extra))
            self.B_bias = nn.Parameter(torch.empty(self.num_heads, self.mimo_rank, d_state, **extra))
            self.C_bias = nn.Parameter(torch.empty(self.num_heads, self.mimo_rank, d_state, **extra))
            self.D = nn.Parameter(torch.empty(self.num_heads, **extra))
        else:
            self.dt_bias = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
            self.B_bias = nn.Parameter(torch.ones(self.num_heads, self.mimo_rank, d_state, dtype=torch.float32))
            self.C_bias = nn.Parameter(torch.ones(self.num_heads, self.mimo_rank, d_state, dtype=torch.float32))
            self.D = nn.Parameter(torch.ones(self.num_heads, dtype=torch.float32))
            if dev is not None:
                self.dt_bias = nn.Parameter(self.dt_bias.to(dev))
                self.B_bias = nn.Parameter(self.B_bias.to(dev))
                self.C_bias = nn.Parameter(self.C_bias.to(dev))
                self.D = nn.Parameter(self.D.to(dev))
            with torch.no_grad():
                dt = torch.exp(
                    torch.rand(self.num_heads, dtype=torch.float32)
                    * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                ).clamp(min=dt_floor)
                self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        self.B_norm = _RMSNorm(d_state)
        self.C_norm = _RMSNorm(d_state)

        if self.is_mimo:
            init = torch.ones(self.num_heads, self.mimo_rank, head_dim)
            self.mimo_x = nn.Parameter((init / self.mimo_rank).to(
                device=device, dtype=torch.float32 if dtype is None else dtype))
            self.mimo_z = nn.Parameter(init.clone().to(
                device=device, dtype=torch.float32 if dtype is None else dtype))
            self.mimo_o = nn.Parameter((init / self.mimo_rank).to(
                device=device, dtype=torch.float32 if dtype is None else dtype))

        if self.is_outproj_norm:
            self.norm = _HeadwiseGatedRMSNorm(d_inner, head_dim)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False, **factory_kwargs)

    def init_cache(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate zero recurrent states.

        Returns ``(angle (B, H, S), ssm_state, bx_prev)`` with the state
        shaped ``(B, H, P, N)`` and ``bx_prev`` shaped ``(B, H, P, N)`` in
        SISO mode / ``(B, R, H, P, N)`` in MIMO mode. Feed as ``cache``
        to :meth:`forward` for chunked prefill or token-by-token decoding.
        """
        dev = self.in_proj.weight.device if device is None else device
        inner = (batch_size, self.num_heads, self.head_dim, self.d_state)
        prev = inner if not self.is_mimo else (batch_size, self.mimo_rank, *inner[1:])
        angle = torch.zeros(batch_size, self.num_heads, self.num_rope_angles, device=dev, dtype=torch.float32)
        return (
            angle,
            torch.zeros(*inner, device=dev, dtype=torch.float32),
            torch.zeros(*prev, device=dev, dtype=torch.float32),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Apply the Mamba-3 mixer.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, d_model)``. The full sequence on the prefill
            step, only the new token(s) when ``cache`` is given.
        cache : tuple | None, default=None
            ``(angle, ssm_state, bx_prev)`` from :meth:`init_cache` or a
            previous call — states of all tokens before ``x``.
        mask : torch.Tensor | None, default=None
            Bool tensor, ``True`` = masked out (padding), broadcastable to
            ``(batch, seq_len)``. Masked positions contribute zeros.

        Returns
        -------
        tuple[torch.Tensor, tuple]
            ``(output, cache)``: the mixer output ``(batch, seq_len,
            d_model)`` and the updated ``(angle, ssm_state, bx_prev)``.
            Cache tensors are detached (no autograd graph leaks across
            generate calls) — chunked training grads are truncated BPTT,
            not full-sequence grads.
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

        z, xs, Bs, Cs, dd_dt, dd_A, trap_raw, ang_raw = torch.split(
            self.in_proj(x),
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.n_groups * self.mimo_rank,
                self.d_state * self.n_groups * self.mimo_rank,
                self.num_heads,
                self.num_heads,
                self.num_heads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        z = z.reshape(batch, seqlen, self.num_heads, self.head_dim)
        xs = xs.reshape(batch, seqlen, self.num_heads, self.head_dim)
        Bs = Bs.reshape(batch, seqlen, self.mimo_rank, self.n_groups, self.d_state)
        Cs = Cs.reshape(batch, seqlen, self.mimo_rank, self.n_groups, self.d_state)

        A = -_heavy_tail(dd_A.float()).clamp(max=-self.a_floor)  # (B, L, H)
        DT = F.softplus(dd_dt.float() + self.dt_bias.float())  # (B, L, H)
        tr = torch.sigmoid(trap_raw.float())  # (B, L, H)
        decay = torch.exp(A * DT)

        # Groups → heads (GQA-style repetition), then bias (both before rotation).
        if self.num_heads % self.n_groups != 0:  # pragma: no cover
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by n_groups ({self.n_groups})")
        rep = self.num_heads // self.n_groups
        B = self.B_norm(Bs).repeat_interleave(rep, dim=3)
        C = self.C_norm(Cs).repeat_interleave(rep, dim=3)
        B = B + self.B_bias.permute(1, 0, 2).reshape(1, 1, self.mimo_rank, self.num_heads, self.d_state).float()
        C = C + self.C_bias.permute(1, 0, 2).reshape(1, 1, self.mimo_rank, self.num_heads, self.d_state).float()

        # Data-dependent RoPE: cumulative angle Φ_t = Φ_prev + Σ dt·φ.
        incr = ang_raw.float().unsqueeze(-2) * DT.unsqueeze(-1)  # (B, L, H, S)
        if cache is not None:
            angle_prev = cache[0]
            phi = angle_prev.unsqueeze(1) + torch.cumsum(incr, dim=1)
        else:
            phi = torch.cumsum(incr, dim=1)
        phi = phi.unsqueeze(2).expand(batch, seqlen, self.mimo_rank, self.num_heads, self.num_rope_angles)
        rope = _apply_rope_adjacent if self.is_mimo else _apply_rope_pairwise
        B = torch.cat([rope(B[..., : self.split_tensor_size], phi), B[..., self.split_tensor_size :]], dim=-1)
        C = torch.cat([rope(C[..., : self.split_tensor_size], phi), C[..., self.split_tensor_size :]], dim=-1)

        if cache is not None:
            ssm_state = cache[1].clone()
            bx_prev = cache[2].clone()
        else:
            ssm_state, bx_prev = None, None

        if self.is_mimo:
            if ssm_state is None:
                ssm_state = torch.zeros(batch, self.num_heads, self.head_dim, self.d_state, device=x.device, dtype=torch.float32)
                bx_prev = torch.zeros(
                    batch, self.mimo_rank, self.num_heads, self.head_dim, self.d_state,
                    device=x.device, dtype=torch.float32,
                )
            y_r, ssm_state, bx_prev = _trapezoidal_scan_mimo(
                xs, B, C, decay, DT, tr, self.D, self.mimo_x, ssm_state, bx_prev
            )  # (B, L, R, H, P)
            z_r = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z.float())
            if self.is_outproj_norm:
                y = self.norm(
                    y_r.reshape(batch, seqlen, self.mimo_rank, -1),
                    z_r.reshape(batch, seqlen, self.mimo_rank, -1),
                ).reshape(batch, seqlen, self.mimo_rank, self.num_heads, self.head_dim)
            else:
                y = y_r * F.silu(z_r)
            y = torch.einsum("blrhp,hrp->blhp", y.float(), self.mimo_o.float())
        else:
            if ssm_state is None:
                ssm_state = torch.zeros(
                    batch, self.num_heads, self.head_dim, self.d_state, device=x.device, dtype=torch.float32
                )
                bx_prev = torch.zeros_like(ssm_state)
            y, ssm_state, bx_prev = _trapezoidal_scan_siso(
                xs, B[:, :, 0], C[:, :, 0], decay, DT, tr, self.D, ssm_state, bx_prev
            )
            y = y * F.silu(z.float())

        y = y.reshape(batch, seqlen, -1)
        return self.out_proj(y.to(dtype)), (phi[:, -1, 0].detach(), ssm_state.detach(), bx_prev.detach())


#: Mixer-level block name (same module).
Mamba3Block = Mamba3Mixer

__all__ = ["Mamba3Block", "Mamba3Mixer"]
