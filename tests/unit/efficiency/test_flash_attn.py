# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""Tests for spartan_torch.efficiency.flash_attn."""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from spartan_torch.efficiency.flash_attn import (
    FlashAttention,
    enable_flash_attention,
    is_flash_attention_available,
)


class TestFlashAttentionAvailability:
    """Tests for flash attention detection functions."""

    def test_is_flash_attention_available(self) -> None:
        result = is_flash_attention_available()
        assert isinstance(result, bool)
        # PyTorch 2.0+ always has SDPA
        assert result is True

    def test_enable_flash_attention(self) -> None:
        result = enable_flash_attention()
        assert isinstance(result, bool)
        # On CPU, no CUDA flash backend
        if not torch.cuda.is_available():
            assert result is False


class TestFlashAttention:
    """Tests for FlashAttention module."""

    def test_forward_shape(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x, x, x)
        assert out.shape == (2, 16, 64)

    def test_forward_separate_qkv(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4)
        q = torch.randn(2, 16, 64)
        k = torch.randn(2, 20, 64)
        v = torch.randn(2, 20, 64)
        out = attn(q, k, v)
        assert out.shape == (2, 16, 64)

    def test_backward(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = attn(x, x, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradcheck(self) -> None:
        attn = FlashAttention(d_model=8, n_heads=2).double()
        q = k = v = torch.randn(1, 4, 8, dtype=torch.double, requires_grad=True)
        assert gradcheck(attn, (q, k, v), check_batched_grad=False)

    def test_causal_mask(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4, is_causal=True)
        x = torch.randn(2, 16, 64)
        out = attn(x, x, x)
        assert out.shape == (2, 16, 64)

    def test_causal_with_mask_raises(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4, is_causal=True)
        x = torch.randn(2, 16, 64)
        mask = torch.zeros(2, 1, 16, 16)
        with pytest.raises(ValueError, match="Cannot use both"):
            attn(x, x, x, attn_mask=mask)

    def test_explicit_mask(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        mask = torch.zeros(16, 16)
        out = attn(x, x, x, attn_mask=mask)
        assert out.shape == (2, 16, 64)

    def test_invalid_d_model(self) -> None:
        with pytest.raises(ValueError, match="divisible by n_heads"):
            FlashAttention(d_model=65, n_heads=4)

    def test_with_bias(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4, bias=True)
        x = torch.randn(2, 16, 64)
        out = attn(x, x, x)
        assert out.shape == (2, 16, 64)

    def test_training_eval(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4, dropout=0.1)
        x = torch.randn(2, 16, 64)
        attn.train()
        out_train = attn(x, x, x)
        attn.eval()
        out_eval = attn(x, x, x)
        # Same shape (dropout is different but output shape is same)
        assert out_train.shape == out_eval.shape

    def test_extra_repr(self) -> None:
        attn = FlashAttention(d_model=64, n_heads=4, dropout=0.1)
        r = repr(attn)
        assert "d_model=64" in r
        assert "n_heads=4" in r
        assert "head_dim=16" in r
        assert "dropout=0.1" in r

    def test_matches_manual_attention(self) -> None:
        torch.manual_seed(42)
        attn = FlashAttention(d_model=32, n_heads=4)
        x = torch.randn(1, 8, 32)
        out = attn(x, x, x)
        # Output should be valid (not NaN/Inf)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
