"""Remap torchvision ResNet-18 state_dict to the compat ResNet test assembly.

Source: ``torchvision.models.resnet18(weights=...)`` (BasicBlock layers).
Target: compat assembly in ``tests/test_weight_parity.py`` built from
:class:`~spartan_torch.ResidualBlock`::

    conv1 / bn1            → stem.0 / stem.1 (Sequential stem)
    layer{i}.{b}.conv1/bn1 → stages.{i-1}.{b}.conv1/bn1
    layer{i}.{b}.conv2/bn2 → stages.{i-1}.{b}.conv2/bn2
    layer{i}.{b}.downsample.0/.1
                          → stages.{i-1}.{b}.downsample.0/.1
    fc.*                  → fc.*

Blocks without a downsample in the source simply leave the target
``downsample`` (an ``nn.Identity``) unmapped — :func:`apply_remap` reports it,
and the caller loads with ``strict=False`` for those keys.

References
----------
"Deep Residual Learning for Image Recognition" (He et al., 2015,
arXiv:1512.03385).
"""

from __future__ import annotations

import torch

from .timm_vit import RemapReport


def remap_torchvision_resnet18(
    tv_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    remapped: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []

    for key, val in tv_sd.items():
        if key == "conv1.weight":
            remapped["stem.0.weight"] = val
        elif key.startswith("bn1."):
            remapped[f"stem.1.{key.removeprefix('bn1.')}"] = val
        elif key.startswith("layer"):
            # layer{i}.{b}.{rest} → stages.{i-1}.{b}.{rest}
            parts = key.split(".", 2)
            layer_idx = int(parts[0].removeprefix("layer")) - 1
            remapped[f"stages.{layer_idx}.{parts[1]}.{parts[2]}"] = val
        elif key.startswith("fc."):
            remapped[key] = val
        else:
            unmatched.append(key)

    report = RemapReport(source_keys=len(tv_sd), remapped_keys=len(remapped), unmatched_source=sorted(unmatched))
    return remapped, report


def apply_remap(
    model: torch.nn.Module,
    remapped: dict[str, torch.Tensor],
    report: RemapReport,
    *,
    strict: bool = True,
) -> RemapReport:
    """Load ``remapped`` into ``model`` and fill missing/extra in ``report``."""
    own = set(model.state_dict().keys())
    got = set(remapped.keys())
    report.missing = sorted(own - got)
    report.extra = sorted(got - own)
    model.load_state_dict(remapped, strict=strict)
    return report
