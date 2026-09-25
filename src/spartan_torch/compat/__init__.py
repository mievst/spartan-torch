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

from .dino_head import remap_dino_head
from .hf_llama import remap_hf_llama_mlp, remap_torchvision_mobilenet_block
from .hf_mamba import hf_mamba_kwargs, remap_hf_mamba_mixer, remap_mamba_ssm_mamba
from .hf_mamba2 import hf_mamba2_kwargs, remap_hf_mamba2_mixer, remap_mamba_ssm_mamba2
from .hf_mamba3 import official_mamba3_kwargs, remap_hf_mamba3_mixer, remap_mamba_ssm_mamba3
from .timm_vit import RemapReport, apply_remap, remap_timm_vit
from .torchvision_resnet import remap_torchvision_resnet18

__all__ = [
    "RemapReport",
    "apply_remap",
    "hf_mamba_kwargs",
    "hf_mamba2_kwargs",
    "official_mamba3_kwargs",
    "remap_dino_head",
    "remap_hf_llama_mlp",
    "remap_hf_mamba_mixer",
    "remap_hf_mamba2_mixer",
    "remap_hf_mamba3_mixer",
    "remap_mamba_ssm_mamba",
    "remap_mamba_ssm_mamba2",
    "remap_mamba_ssm_mamba3",
    "remap_timm_vit",
    "remap_torchvision_mobilenet_block",
    "remap_torchvision_resnet18",
]
