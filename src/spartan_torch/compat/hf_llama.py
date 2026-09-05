"""Remap torchvision / HF block weights to spartan-torch primitives.

Block-level remappers (no downloads needed for random-weight parity):

- :func:`remap_torchvision_mobilenet_block`: ``torchvision`` ``InvertedResidual``
  (nested ``conv.0.0``/``conv.0.1`` expand pair, ``conv.1.*`` depthwise pair,
  ``conv.2``/``conv.3`` project conv+bn) → ours
  (``conv1/bn1/conv2/bn2/conv3/bn3``).
  Only ``expand_ratio > 1`` blocks map 1-to-1: ours uses ``nn.Identity`` for
  the expand stage at ``expansion <= 1`` while torchvision keeps a real 1x1
  conv there, and stride-2 blocks need ``use_skip=False`` (torchvision has no
  projection shortcut — the residual is dropped, not projected).
- :func:`remap_hf_llama_mlp`: HF ``LlamaMLP`` (``gate_proj``/``up_proj``/
  ``down_proj``, no bias) → :class:`~spartan_torch.SwiGLUFeedForward`
  (keys are identical; the remap is explicitness + a coverage report).

References
----------
"MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al.,
2018, arXiv:1801.04381); "LLaMA: Open and Efficient Foundation Language
Models" (Touvron et al., 2023, arXiv:2302.13971).
"""

from __future__ import annotations

import torch

from .timm_vit import RemapReport

_MOBILENET_KEYMAP = {
    # torchvision nests ConvNormActivation: conv.0/conv.1 are (conv, bn)
    # pairs, conv.2/conv.3 the bare project conv + bn (activations carry no
    # params and contribute no keys).
    "conv.0.0.weight": "conv1.weight",
    "conv.0.1.weight": "bn1.weight",
    "conv.0.1.bias": "bn1.bias",
    "conv.0.1.running_mean": "bn1.running_mean",
    "conv.0.1.running_var": "bn1.running_var",
    "conv.0.1.num_batches_tracked": "bn1.num_batches_tracked",
    "conv.1.0.weight": "conv2.weight",
    "conv.1.1.weight": "bn2.weight",
    "conv.1.1.bias": "bn2.bias",
    "conv.1.1.running_mean": "bn2.running_mean",
    "conv.1.1.running_var": "bn2.running_var",
    "conv.1.1.num_batches_tracked": "bn2.num_batches_tracked",
    "conv.2.weight": "conv3.weight",
    "conv.3.weight": "bn3.weight",
    "conv.3.bias": "bn3.bias",
    "conv.3.running_mean": "bn3.running_mean",
    "conv.3.running_var": "bn3.running_var",
    "conv.3.num_batches_tracked": "bn3.num_batches_tracked",
}


def remap_torchvision_mobilenet_block(
    tv_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    remapped, unmatched = {}, []
    for key, val in tv_sd.items():
        if key in _MOBILENET_KEYMAP:
            remapped[_MOBILENET_KEYMAP[key]] = val
        else:
            unmatched.append(key)
    report = RemapReport(source_keys=len(tv_sd), remapped_keys=len(remapped), unmatched_source=sorted(unmatched))
    return remapped, report


_LLAMA_MLP_KEYS = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


def remap_hf_llama_mlp(
    hf_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    remapped = {k: hf_sd[k] for k in _LLAMA_MLP_KEYS if k in hf_sd}
    unmatched = sorted(set(hf_sd) - set(_LLAMA_MLP_KEYS))
    report = RemapReport(source_keys=len(hf_sd), remapped_keys=len(remapped), unmatched_source=unmatched)
    return remapped, report
