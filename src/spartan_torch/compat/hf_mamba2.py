"""Remap Mamba-2 mixer weights to :class:`~spartan_torch.Mamba2Mixer`.

Block-level remappers (no downloads needed for random-weight parity):

- :func:`remap_hf_mamba2_mixer`: HF ``Mamba2Mixer`` (``hidden_size`` /
  ``intermediate_size`` / ``state_size`` / ``conv_kernel`` / ``num_heads`` /
  ``head_dim`` / ``n_groups`` / ``chunk_size`` config) → ours (``d_model`` /
  ``d_inner`` / ``d_state`` / ``d_conv`` / ``num_heads`` / ``head_dim`` /
  ``n_groups`` / ``chunk_size``).
- :func:`remap_mamba_ssm_mamba2`: official ``mamba-ssm`` ``Mamba2`` →
  ours.

Both references use identical parameter names (``in_proj``, ``conv1d``,
``dt_bias``, ``A_log``, ``D``, ``norm``, ``out_proj``), so the remap is key
filtering (bias keys exist only when the matching ``bias`` / ``conv_bias``
flag is on) plus a coverage report. :func:`hf_mamba2_kwargs` translates an
HF ``Mamba2Config`` into our constructor kwargs without importing
``transformers`` (duck-typed).

References
----------
"Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality" (Dao & Gu, 2024,
arXiv:2405.21060).
"""

from __future__ import annotations

import torch

from .timm_vit import RemapReport

_REQUIRED_KEYS = (
    "in_proj.weight",
    "conv1d.weight",
    "dt_bias",
    "A_log",
    "D",
    "norm.weight",
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


def remap_hf_mamba2_mixer(
    hf_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Remap an HF ``Mamba2Mixer`` state dict to :class:`~spartan_torch.Mamba2Mixer`."""
    return _remap(hf_sd)


def remap_mamba_ssm_mamba2(
    ssm_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Remap an official ``mamba-ssm`` ``Mamba2`` state dict to :class:`~spartan_torch.Mamba2Mixer`."""
    return _remap(ssm_sd)


def hf_mamba2_kwargs(cfg) -> dict:
    """Translate an HF ``Mamba2Config`` to :class:`~spartan_torch.Mamba2Mixer` kwargs.

    Duck-typed (attribute access only) so ``transformers`` stays an
    experiments-only dependency.
    """
    return {
        "d_model": cfg.hidden_size,
        "d_state": cfg.state_size,
        "d_conv": cfg.conv_kernel,
        "expand": cfg.expand,
        "num_heads": cfg.num_heads,
        "head_dim": cfg.head_dim,
        "n_groups": cfg.n_groups,
        "dt_min": cfg.time_step_min,
        "dt_max": cfg.time_step_max,
        "dt_floor": cfg.time_step_floor,
        "dt_limit": tuple(cfg.time_step_limit),
        "conv_bias": cfg.use_conv_bias,
        "bias": cfg.use_bias,
        "norm_eps": cfg.layer_norm_epsilon,
        "chunk_size": cfg.chunk_size,
    }
