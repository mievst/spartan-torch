"""Pretrained weight remappers: timm/HF/torchvision → spartan-torch primitives.

Each remapper is a pure function ``state_dict → (remapped, report)`` with no
heavy imports at module level (``timm``/``transformers``/``torchvision`` are
imported by the caller — tests and ``scripts/verify_pretrained.py``).

Target key layouts are the compat assemblies defined in
``tests/test_weight_parity.py`` (which mirror ``experiments/vit/...
vision_transformer.py`` for ViT and plain ``ResidualBlock`` stages for
ResNet). Library code in ``src/spartan_torch`` stays free of full-model
assemblies by design (see AGENTS.md).
"""

from .hf_llama import remap_hf_llama_mlp, remap_torchvision_mobilenet_block
from .timm_vit import RemapReport, apply_remap, remap_timm_vit
from .torchvision_resnet import remap_torchvision_resnet18

__all__ = [
    "RemapReport",
    "apply_remap",
    "remap_hf_llama_mlp",
    "remap_timm_vit",
    "remap_torchvision_mobilenet_block",
    "remap_torchvision_resnet18",
]
