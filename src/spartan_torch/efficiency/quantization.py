# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
Quantization utilities using bitsandbytes.

Provides wrappers for 8-bit and 4-bit quantization of ``nn.Linear`` layers:

- :func:`quantize_model_int8` -- 8-bit quantization via LLM.int8().
- :func:`quantize_model_int4` -- 4-bit quantization (NF4/FP4).
- :class:`QuantizationConfig` -- configuration dataclass.

Requires ``bitsandbytes`` and CUDA.

# VRAM: ~0 MB (wrapper, quantization reduces model VRAM)

References
----------
.. [1] Dettmers et al., "8-bit Optimizers via Block-wise Quantization"
       https://arxiv.org/abs/2110.02861
.. [2] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs"
       https://arxiv.org/abs/2305.14314

Examples
--------
>>> import torch.nn as nn
>>> from spartan_torch.efficiency.quantization import quantize_model_int4
>>> model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
>>> model = quantize_model_int4(model)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = [
    "QuantizationConfig",
    "is_bitsandbytes_available",
    "quantize_model_int4",
    "quantize_model_int8",
]


def is_bitsandbytes_available() -> bool:
    """Check if bitsandbytes is installed."""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def _require_bnb() -> None:
    if not is_bitsandbytes_available():
        raise ImportError(
            "bitsandbytes is required for quantization. Install with: uv add bitsandbytes"
        )


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes quantization requires CUDA. Current device: CPU.")


@dataclass
class QuantizationConfig:
    """
    Configuration for quantization.

    Parameters
    ----------
    bits : int, default=4
        Quantization bits (4 or 8).
    quant_type : str, default="nf4"
        Quantization type for 4-bit: ``"nf4"`` (NormalFloat4) or ``"fp4"``.
    compute_dtype : torch.dtype, default=torch.float16
        Compute dtype for dequantized matmuls.
    compress_statistics : bool, default=True
        Use double quantization (offset + scale are quantized too).
        Only for 4-bit.
    threshold : float, default=0.0
        Outlier threshold for 8-bit. Values above are processed in fp16.
        0.0 disables outlier handling.
    """

    bits: int = 4
    quant_type: str = "nf4"
    compute_dtype: torch.dtype = field(default=torch.float16)
    compress_statistics: bool = True
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.bits == 4 and self.quant_type not in ("nf4", "fp4"):
            raise ValueError(f"quant_type must be 'nf4' or 'fp4', got '{self.quant_type}'")


def _replace_linear_int8(
    module: nn.Module,
    threshold: float,
    has_fp16_weights: bool,
) -> nn.Module:
    """Replace nn.Linear layers with bnb.nn.Linear8bitLt in-place."""
    import bitsandbytes as bnb

    if isinstance(module, nn.Linear):
        int8_linear = bnb.nn.Linear8bitLt(  # type: ignore[attr-defined]
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            has_fp16_weights=has_fp16_weights,
            threshold=threshold,
        )
        int8_linear.weight = bnb.nn.Int8Params(
            module.weight.data,
            has_fp16_weights=has_fp16_weights,  # ty: ignore[unknown-argument]
            requires_grad=has_fp16_weights,
        )
        if module.bias is not None:
            int8_linear.bias = nn.Parameter(module.bias.data)
        return int8_linear

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            int8_linear = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=has_fp16_weights,
                threshold=threshold,
            )
            int8_linear.weight = bnb.nn.Int8Params(
                child.weight.data,
                has_fp16_weights=has_fp16_weights,  # ty: ignore[unknown-argument]
                requires_grad=has_fp16_weights,
            )
            if child.bias is not None:
                int8_linear.bias = nn.Parameter(child.bias.data)
            setattr(module, name, int8_linear)
        else:
            _replace_linear_int8(child, threshold, has_fp16_weights)
    return module


def _replace_linear_int4(
    module: nn.Module,
    compute_dtype: torch.dtype,
    quant_type: str,
    compress_statistics: bool,
) -> nn.Module:
    """Replace nn.Linear layers with bnb.nn.Linear4bit in-place."""
    import bitsandbytes as bnb

    if isinstance(module, nn.Linear):
        int4_linear = bnb.nn.Linear4bit(  # type: ignore[attr-defined, no-untyped-call]
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            compress_statistics=compress_statistics,
        )
        int4_linear.weight = bnb.nn.Params4bit(
            module.weight.data,
            quant_type=quant_type,  # ty: ignore[unknown-argument]
            compress_statistics=compress_statistics,  # ty: ignore[unknown-argument]
            requires_grad=False,
        )
        if module.bias is not None:
            int4_linear.bias = nn.Parameter(module.bias.data)
        return int4_linear

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            int4_linear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                compress_statistics=compress_statistics,
            )
            int4_linear.weight = bnb.nn.Params4bit(
                child.weight.data,
                quant_type=quant_type,  # ty: ignore[unknown-argument]
                compress_statistics=compress_statistics,  # ty: ignore[unknown-argument]
                requires_grad=False,
            )
            if child.bias is not None:
                int4_linear.bias = nn.Parameter(child.bias.data)
            setattr(module, name, int4_linear)
        else:
            _replace_linear_int4(child, compute_dtype, quant_type, compress_statistics)
    return module


def quantize_model_int8(
    model: nn.Module,
    *,
    config: QuantizationConfig | None = None,
) -> nn.Module:
    """
    Quantize all ``nn.Linear`` layers to 8-bit using bitsandbytes LLM.int8().

    Weights are quantized on-the-fly during forward. Outlier features above
    ``threshold`` are processed in fp16 to maintain accuracy.

    Parameters
    ----------
    model : nn.Module
        Model to quantize. Modified in-place.
    config : QuantizationConfig, default=None
        Quantization config. If ``None``, uses defaults (threshold=0.0).

    Returns
    -------
    nn.Module
        The same model with quantized layers.

    Raises
    ------
    ImportError
        If bitsandbytes is not installed.
    RuntimeError
        If CUDA is not available.

    Examples
    --------
    >>> model = nn.Linear(256, 256)
    >>> model = quantize_model_int8(model)
    >>> # Weight is now bnb.nn.Int8Params
    """
    _require_bnb()
    _require_cuda()

    if config is None:
        config = QuantizationConfig(bits=8)
    elif config.bits != 8:
        raise ValueError(f"quantize_model_int8 requires bits=8, got bits={config.bits}")

    return _replace_linear_int8(model, config.threshold, has_fp16_weights=True)


def quantize_model_int4(
    model: nn.Module,
    *,
    config: QuantizationConfig | None = None,
) -> nn.Module:
    """
    Quantize all ``nn.Linear`` layers to 4-bit using bitsandbytes.

    Supports NF4 (NormalFloat4) and FP4 quantization with optional
    double quantization (``compress_statistics``).

    Parameters
    ----------
    model : nn.Module
        Model to quantize. Modified in-place.
    config : QuantizationConfig, default=None
        Quantization config. If ``None``, uses defaults (nf4, fp16 compute).

    Returns
    -------
    nn.Module
        The same model with quantized layers.

    Raises
    ------
    ImportError
        If bitsandbytes is not installed.
    RuntimeError
        If CUDA is not available.

    Examples
    --------
    >>> model = nn.Linear(256, 256)
    >>> model = quantize_model_int4(model)
    >>> # Weight is now bnb.nn.Params4bit
    """
    _require_bnb()
    _require_cuda()

    if config is None:
        config = QuantizationConfig(bits=4)
    elif config.bits != 4:
        raise ValueError(f"quantize_model_int4 requires bits=4, got bits={config.bits}")

    return _replace_linear_int4(
        model,
        config.compute_dtype,
        config.quant_type,
        config.compress_statistics,
    )
