# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
Efficiency utilities for low-VRAM training.

Includes gradient checkpointing, quantization (8-bit, 4-bit), LoRA/QLoRA, FlashAttention.
"""

from spartan_torch.efficiency.checkpointing import (
    CheckpointPolicy,
    GradientCheckpointing,
    checkpoint_sequential,
    selective_checkpoint,
)
from spartan_torch.efficiency.flash_attn import (
    FlashAttention,
    enable_flash_attention,
    is_flash_attention_available,
)
from spartan_torch.efficiency.quantization import (
    QuantizationConfig,
    is_bitsandbytes_available,
    quantize_model_int4,
    quantize_model_int8,
)

__all__ = [
    "CheckpointPolicy",
    "FlashAttention",
    # Checkpointing
    "GradientCheckpointing",
    "QuantizationConfig",
    "checkpoint_sequential",
    # FlashAttention
    "enable_flash_attention",
    "is_bitsandbytes_available",
    "is_flash_attention_available",
    "quantize_model_int4",
    # Quantization
    "quantize_model_int8",
    "selective_checkpoint",
]
