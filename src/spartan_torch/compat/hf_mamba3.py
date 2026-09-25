"""Remap Mamba-3 mixer weights to :class:`~spartan_torch.Mamba3Mixer`.

Block-level remappers:

- :func:`remap_mamba_ssm_mamba3`: official ``mamba-ssm`` ``Mamba3``
  (``in_proj``, ``dt_bias``, ``B_bias``/``C_bias``, ``B_norm``/``C_norm``,
  ``D``, ``mimo_x``/``mimo_z``/``mimo_o`` (MIMO only), ``norm``
  (``is_outproj_norm`` only), ``out_proj``) → ours. Key names are
  identical; the remap is key filtering plus a coverage report.
- :func:`remap_hf_mamba3_mixer`: same layout under an HF-style prefix
  (e.g. community ``mamba3_hf_ready`` checkpoints) → ours.

:func:`official_mamba3_kwargs` translates an official-style config mapping
(``d_model``/``d_state``/``headdim``/``ngroups``/``dt_init_floor``/
``A_floor``/...) into our constructor kwargs; kernel-only keys
(``chunk_size`` and friends) are dropped.

There is no runnable reference in this environment (the official kernels
are Linux/CUDA-only and ``transformers`` ships no Mamba-3): numeric
coverage is strict-loading real checkpoints plus self-agreement
(step-vs-prefill), see ``tests/test_weight_parity.py``.

References
----------
"Mamba-3: Improved Sequence Modeling using State Space Principles"
(Lahoti et al., 2026, arXiv:2603.15569).
"""

from __future__ import annotations

import torch

from .timm_vit import RemapReport

_REQUIRED_KEYS = (
    "in_proj.weight",
    "dt_bias",
    "B_bias",
    "C_bias",
    "B_norm.weight",
    "C_norm.weight",
    "D",
    "out_proj.weight",
)

_OPTIONAL_KEYS = (
    "mimo_x",
    "mimo_z",
    "mimo_o",
    "norm.weight",
)


def _remap(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], RemapReport]:
    remapped = {k: sd[k] for k in _REQUIRED_KEYS + _OPTIONAL_KEYS if k in sd}
    unmatched = sorted(set(sd) - set(remapped))
    report = RemapReport(source_keys=len(sd), remapped_keys=len(remapped), unmatched_source=unmatched)
    return remapped, report


def remap_mamba_ssm_mamba3(
    ssm_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Remap an official ``mamba-ssm`` ``Mamba3`` state dict to :class:`~spartan_torch.Mamba3Mixer`."""
    return _remap(ssm_sd)


def remap_hf_mamba3_mixer(
    hf_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Remap an HF-style Mamba-3 mixer state dict to :class:`~spartan_torch.Mamba3Mixer`."""
    return _remap(hf_sd)


def official_mamba3_kwargs(cfg: dict) -> dict:
    """Translate an official-style Mamba-3 config mapping to our kwargs.

    Accepts the ``ib-ssm``/``state-spaces`` config key spellings
    (``headdim``/``ngroups``/``dt_init_floor``/``A_floor``/``state_size``/
    ``conv_kernel``...). Kernel-only (``chunk_size``) and model-level keys
    are ignored.
    """
    rope = cfg.get("rope_fraction", 0.5)
    n_groups = cfg.get("n_groups", cfg.get("ngroups", 1))
    return {
        "d_model": cfg.get("d_model", cfg.get("hidden_size")),
        "d_state": cfg.get("d_state", cfg.get("state_size", 128)),
        "expand": cfg.get("expand", 2),
        "head_dim": cfg.get("headdim", cfg.get("head_dim", 64)),
        "n_groups": n_groups,
        "rope_fraction": rope,
        "dt_min": cfg.get("dt_min", 0.001),
        "dt_max": cfg.get("dt_max", 0.1),
        "dt_floor": cfg.get("dt_init_floor", cfg.get("dt_floor", 1e-4)),
        "a_floor": cfg.get("A_floor", cfg.get("a_floor", 1e-4)),
        "is_outproj_norm": cfg.get("is_outproj_norm", False),
        "is_mimo": cfg.get("is_mimo", False),
        "mimo_rank": cfg.get("mimo_rank", 4),
    }
