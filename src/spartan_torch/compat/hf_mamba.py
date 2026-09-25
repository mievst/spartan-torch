"""Remap Mamba-1 mixer weights to :class:`~spartan_torch.MambaMixer`.

Block-level remappers (no downloads needed for random-weight parity):

- :func:`remap_hf_mamba_mixer`: HF ``MambaMixer`` (``hidden_size`` /
  ``intermediate_size`` / ``state_size`` / ``conv_kernel`` /
  ``time_step_rank`` config) → ours (``d_model`` / ``d_inner`` / ``d_state``
  / ``d_conv`` / ``dt_rank``).
- :func:`remap_mamba_ssm_mamba`: official ``mamba-ssm`` ``Mamba``
  (``d_model`` / ``d_state`` / ``d_conv`` / ``expand`` / ``dt_rank``) →
  ours.

Both references use identical parameter names (``in_proj``, ``conv1d``,
``x_proj``, ``dt_proj``, ``A_log``, ``D``, ``out_proj``), so the remap is
key filtering (bias keys exist only when the matching ``bias`` /
``conv_bias`` flag is on) plus a coverage report. :func:`hf_mamba_kwargs`
translates an HF ``MambaConfig`` into our constructor kwargs without
importing ``transformers`` (duck-typed).

References
----------
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu &
Dao, 2024, arXiv:2312.00752).
"""

from __future__ import annotations

import torch

from .timm_vit import RemapReport

_REQUIRED_KEYS = (
    "in_proj.weight",
    "conv1d.weight",
    "x_proj.weight",
    "dt_proj.weight",
    "dt_proj.bias",
    "A_log",
    "D",
    "out_proj.weight",
)

_OPTIONAL_KEYS = (
    "in_proj.bias",
    "conv1d.bias",
    "out_proj.bias",
)


def _remap(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], RemapReport]:
    remapped = {k: sd[k] for k in _REQUIRED_KEYS + _OPTIONAL_KEYS if k in sd}
    unmatched = sorted(set(sd) - set(remapped))
    report = RemapReport(source_keys=len(sd), remapped_keys=len(remapped), unmatched_source=unmatched)
    return remapped, report


def remap_hf_mamba_mixer(
    hf_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Remap an HF ``MambaMixer`` state dict to :class:`~spartan_torch.MambaMixer`."""
    return _remap(hf_sd)


def remap_mamba_ssm_mamba(
    ssm_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Remap an official ``mamba-ssm`` ``Mamba`` state dict to :class:`~spartan_torch.MambaMixer`."""
    return _remap(ssm_sd)


def hf_mamba_kwargs(cfg) -> dict:
    """Translate an HF ``MambaConfig`` to :class:`~spartan_torch.MambaMixer` kwargs.

    Duck-typed (attribute access only) so ``transformers`` stays an
    experiments-only dependency. ``time_step_rank="auto"`` passes through —
    both sides resolve it as ``ceil(hidden_size / 16)``.
    """
    return {
        "d_model": cfg.hidden_size,
        "d_state": cfg.state_size,
        "d_conv": cfg.conv_kernel,
        "expand": cfg.expand,
        "dt_rank": cfg.time_step_rank,
        "dt_min": cfg.time_step_min,
        "dt_max": cfg.time_step_max,
        "dt_init": cfg.time_step_init_scheme,
        "dt_scale": cfg.time_step_scale,
        "dt_init_floor": cfg.time_step_floor,
        "conv_bias": cfg.use_conv_bias,
        "bias": cfg.use_bias,
    }
