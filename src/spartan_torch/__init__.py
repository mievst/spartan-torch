# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
spartan-torch: Efficient Deep Learning utilities for low-VRAM training.

This package provides memory-efficient building blocks for deep learning
with a focus on 4GB VRAM budgets and performance.

Key features:
- Gradient Checkpointing (sublinear memory scaling)
- FlashAttention-2 (fused attention kernels with SDPA fallback)
- Quantization (8-bit and 4-bit via bitsandbytes)
- VRAM tracking and profiling utilities

Example:
    >>> from spartan_torch import GradientCheckpointing, FlashAttention
    >>> from spartan_torch.efficiency import quantize_model_int8
    >>>
    >>> attn = FlashAttention(d_model=256, n_heads=8)
    >>> checkpointed = GradientCheckpointing(attn)
"""

from importlib.metadata import version as _version

try:
    __version__ = _version("spartan-torch")
except Exception:
    __version__ = "0.0.0+dev"

from spartan_torch.efficiency import (
    CheckpointPolicy,
    FlashAttention,
    GradientCheckpointing,
    QuantizationConfig,
    checkpoint_sequential,
    enable_flash_attention,
    is_bitsandbytes_available,
    is_flash_attention_available,
    quantize_model_int4,
    quantize_model_int8,
    selective_checkpoint,
)

# VRAM utilities
from spartan_torch.utils import (
    VRAMTracker,
    get_max_vram_allocated,
    get_vram_reserved,
    get_vram_usage,
    reset_vram_stats,
    vram_profile,
)

__all__ = [
    "CheckpointPolicy",
    "FlashAttention",
    # Efficiency
    "GradientCheckpointing",
    "QuantizationConfig",
    "VRAMTracker",
    # Version
    "__version__",
    "checkpoint_sequential",
    "enable_flash_attention",
    "get_max_vram_allocated",
    "get_vram_reserved",
    # VRAM utils
    "get_vram_usage",
    "is_bitsandbytes_available",
    "is_flash_attention_available",
    "quantize_model_int4",
    "quantize_model_int8",
    "reset_vram_stats",
    "selective_checkpoint",
    "vram_profile",
]
