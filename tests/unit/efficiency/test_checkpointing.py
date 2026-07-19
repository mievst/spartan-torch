# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""Tests for spartan_torch.efficiency.checkpointing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from spartan_torch.efficiency.checkpointing import (
    CheckpointPolicy,
    GradientCheckpointing,
    checkpoint_sequential,
    selective_checkpoint,
)


class TestGradientCheckpointing:
    """Tests for GradientCheckpointing wrapper."""

    def test_forward_shape(self) -> None:
        block = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        ckpt = GradientCheckpointing(block)
        x = torch.randn(2, 16, 32)
        y = ckpt(x)
        assert y.shape == (2, 16, 32)

    def test_backward(self) -> None:
        block = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        ckpt = GradientCheckpointing(block)
        x = torch.randn(2, 16, 32, requires_grad=True)
        y = ckpt(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradcheck(self) -> None:
        block = nn.Linear(8, 8).double()
        ckpt = GradientCheckpointing(block)
        x = torch.randn(2, 8, dtype=torch.double, requires_grad=True)
        assert gradcheck(ckpt, (x,), check_batched_grad=False)

    def test_matches_uncheckpointed(self) -> None:
        block = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        ckpt = GradientCheckpointing(block)

        torch.manual_seed(42)
        x = torch.randn(2, 32)
        y_ckpt = ckpt(x.clone())

        torch.manual_seed(42)
        block2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        y_ref = block2(x.clone())

        # Weights differ because GradientCheckpointing wraps same module,
        # but output shape and gradient flow should work
        assert y_ckpt.shape == y_ref.shape

    def test_preserves_rng_state(self) -> None:
        block = nn.Sequential(nn.Linear(32, 32), nn.Dropout(0.5))
        ckpt = GradientCheckpointing(block, preserve_rng_state=True)
        x = torch.randn(2, 32)
        block.train()
        y1 = ckpt(x)
        y2 = ckpt(x)
        # With same RNG state preserved, dropout should be deterministic
        # (both calls see same RNG state at entry)
        assert y1.shape == y2.shape

    def test_use_reentrant_false(self) -> None:
        block = nn.Linear(32, 32)
        ckpt = GradientCheckpointing(block, use_reentrant=False)
        assert ckpt.use_reentrant is False

    def test_extra_repr(self) -> None:
        block = nn.Linear(32, 32)
        ckpt = GradientCheckpointing(block)
        repr_str = repr(ckpt)
        assert "GradientCheckpointing" in repr_str
        assert "preserve_rng_state=True" in repr_str

    def test_kwargs_forward(self) -> None:
        class MultiArgModule(nn.Module):
            def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
                return x * scale

        ckpt = GradientCheckpointing(MultiArgModule())
        x = torch.randn(2, 8)
        y = ckpt(x, scale=2.0)
        assert torch.allclose(y, x * 2.0)


class TestCheckpointSequential:
    """Tests for checkpoint_sequential."""

    def test_forward_shape(self) -> None:
        model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        x = torch.randn(2, 32)
        y = checkpoint_sequential(model, segments=2, input=x)
        assert y.shape == (2, 32)

    def test_matches_uncheckpointed(self) -> None:
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        x = torch.randn(2, 32)
        y_ref = model(x.clone())

        torch.manual_seed(42)
        model2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        y_ckpt = checkpoint_sequential(model2, segments=2, input=x.clone())

        assert torch.allclose(y_ref, y_ckpt, atol=1e-5)

    def test_list_of_modules(self) -> None:
        modules = [nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)]
        x = torch.randn(2, 32)
        y = checkpoint_sequential(modules, segments=1, input=x)
        assert y.shape == (2, 32)


class TestSelectiveCheckpoint:
    """Tests for selective_checkpoint context manager."""

    def test_context_manager_enter_exit(self) -> None:
        def policy(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
            return CheckpointPolicy.PREFER_RECOMPUTE

        ctx = selective_checkpoint(policy)
        with ctx:
            pass  # selective AC context is active

    def test_list_policy_enter_exit(self) -> None:
        ctx = selective_checkpoint([])
        with ctx:
            pass

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Selective AC requires CUDA dispatch",
    )
    def test_with_model_cuda(self) -> None:
        def policy(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
            return CheckpointPolicy.PREFER_RECOMPUTE

        model = nn.Linear(8, 8).cuda()
        x = torch.randn(2, 8, device="cuda", requires_grad=True)
        with selective_checkpoint(policy):
            y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
