# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""Tests for spartan_torch.efficiency.quantization."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spartan_torch.efficiency.quantization import (
    QuantizationConfig,
    is_bitsandbytes_available,
    quantize_model_int4,
    quantize_model_int8,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for quantization"
)
requires_bnb = pytest.mark.skipif(
    not is_bitsandbytes_available(), reason="bitsandbytes not installed"
)


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""

    def test_default_4bit(self) -> None:
        cfg = QuantizationConfig()
        assert cfg.bits == 4
        assert cfg.quant_type == "nf4"
        assert cfg.compute_dtype == torch.float16
        assert cfg.compress_statistics is True
        assert cfg.threshold == 0.0

    def test_default_8bit(self) -> None:
        cfg = QuantizationConfig(bits=8)
        assert cfg.bits == 8

    def test_invalid_bits(self) -> None:
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            QuantizationConfig(bits=16)

    def test_invalid_quant_type(self) -> None:
        with pytest.raises(ValueError, match="quant_type must be"):
            QuantizationConfig(bits=4, quant_type="int8")

    def test_fp4_valid(self) -> None:
        cfg = QuantizationConfig(quant_type="fp4")
        assert cfg.quant_type == "fp4"


class TestBitsandbytesAvailability:
    """Tests for bitsandbytes detection."""

    def test_is_bitsandbytes_available(self) -> None:
        result = is_bitsandbytes_available()
        assert isinstance(result, bool)


@requires_cuda
@requires_bnb
class TestQuantizeInt8:
    """Tests for 8-bit quantization (requires CUDA)."""

    def test_quantize_linear(self) -> None:
        model = nn.Linear(64, 64).cuda()
        model = quantize_model_int8(model)
        import bitsandbytes as bnb

        assert isinstance(model.weight, bnb.nn.Int8Params)

    def test_quantize_sequential(self) -> None:
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).cuda()
        model = quantize_model_int8(model)
        import bitsandbytes as bnb

        assert isinstance(model[0], bnb.nn.Linear8bitLt)
        assert isinstance(model[2], bnb.nn.Linear8bitLt)

    def test_forward_pass(self) -> None:
        model = nn.Linear(64, 64).cuda()
        model = quantize_model_int8(model)
        x = torch.randn(2, 64, device="cuda")
        y = model(x)
        assert y.shape == (2, 64)

    def test_custom_config(self) -> None:
        cfg = QuantizationConfig(bits=8, threshold=6.0)
        model = nn.Linear(64, 64)
        model = quantize_model_int8(model, config=cfg)
        assert model is not None

    def test_wrong_bits_raises(self) -> None:
        cfg = QuantizationConfig(bits=4)
        model = nn.Linear(64, 64)
        with pytest.raises(ValueError, match="quantize_model_int8 requires bits=8"):
            quantize_model_int8(model, config=cfg)

    def test_preserves_bias(self) -> None:
        model = nn.Linear(64, 64, bias=True)
        model = quantize_model_int8(model)
        assert model.bias is not None
        assert model.bias.shape == (64,)


@requires_cuda
@requires_bnb
class TestQuantizeInt4:
    """Tests for 4-bit quantization (requires CUDA)."""

    def test_quantize_linear(self) -> None:
        model = nn.Linear(64, 64).cuda()
        model = quantize_model_int4(model)
        import bitsandbytes as bnb

        assert isinstance(model, bnb.nn.Linear4bit)

    def test_quantize_sequential(self) -> None:
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).cuda()
        model = quantize_model_int4(model)
        import bitsandbytes as bnb

        assert isinstance(model[0], bnb.nn.Linear4bit)
        assert isinstance(model[2], bnb.nn.Linear4bit)

    def test_forward_pass(self) -> None:
        model = quantize_model_int4(nn.Linear(64, 64)).cuda()
        x = torch.randn(2, 64, device="cuda")
        y = model(x)
        assert y.shape == (2, 64)

    def test_custom_config(self) -> None:
        cfg = QuantizationConfig(bits=4, quant_type="fp4", compress_statistics=False)
        model = nn.Linear(64, 64)
        model = quantize_model_int4(model, config=cfg)
        assert model is not None

    def test_wrong_bits_raises(self) -> None:
        cfg = QuantizationConfig(bits=8)
        model = nn.Linear(64, 64)
        with pytest.raises(ValueError, match="quantize_model_int4 requires bits=4"):
            quantize_model_int4(model, config=cfg)

    def test_preserves_bias(self) -> None:
        model = nn.Linear(64, 64, bias=True)
        model = quantize_model_int4(model)
        assert model.bias is not None
        assert model.bias.shape == (64,)

    def test_no_bias(self) -> None:
        model = nn.Linear(64, 64, bias=False)
        model = quantize_model_int4(model)
        assert model.bias is None
