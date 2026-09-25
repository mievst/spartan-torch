"""Remap facebookresearch/dino ``DINOHead`` state_dict keys to :class:`DINOProjectionHead`.

Source: ``DINOHead`` in ``facebookresearch/dino`` (``vision_transformer.py``,
Apache-2.0; ``nn.utils.weight_norm`` API)::

    mlp.{0,2,4}.{weight,bias} → mlp.{0,2,4}.{weight,bias}   (identical layout)
    last_layer.weight_g        → last_layer.parametrizations.weight.original0
    last_layer.weight_v        → last_layer.parametrizations.weight.original1

The old ``weight_norm`` API stores the scale/direction as ``weight_g`` /
``weight_v`` parameters; the ``parametrizations`` API used here stores the
same tensors as ``original0`` / ``original1`` — the forward math is identical
(``g * v / ||v||``), so a direct tensor copy preserves parity.

``DINOProjectionHead`` keeps ``last_linear`` as an alias of ``last_layer``
(backward compatibility), which duplicates the keys in ``state_dict`` — the
remap emits both prefixes pointing at the same tensors so that
``load_state_dict(strict=True)`` passes.

References
----------
"Emerging Properties in Self-Supervised Vision Transformers" (Caron et al.,
2021, arXiv:2104.14294).
"""

from __future__ import annotations

import torch

from .timm_vit import RemapReport


def remap_dino_head(
    ref_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Remap a ``DINOHead`` state_dict to the :class:`DINOProjectionHead` layout."""
    remapped: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []

    for key, val in ref_sd.items():
        if key.startswith("mlp."):
            remapped[key] = val
        elif key == "last_layer.weight_g":
            remapped["last_layer.parametrizations.weight.original0"] = val
            remapped["last_linear.parametrizations.weight.original0"] = val
        elif key == "last_layer.weight_v":
            remapped["last_layer.parametrizations.weight.original1"] = val
            remapped["last_linear.parametrizations.weight.original1"] = val
        else:
            unmatched.append(key)

    report = RemapReport(
        source_keys=len(ref_sd), remapped_keys=len(remapped), unmatched_source=sorted(unmatched)
    )
    return remapped, report
