import numpy as np
import pytest
import torch

from spartan_torch import WarmupScheduler


def make_optimizer(lr=0.1):
    params = [torch.nn.Parameter(torch.randn(2, 2))]
    return torch.optim.SGD(params, lr=lr)


def step(opt, scheduler):
    opt.step()
    scheduler.step()


class TestWarmupScheduler:
    def test_warmup_cosine_curve(self):
        base, min_lr, warmup = 0.1, 1e-9, 10
        opt = make_optimizer(base)
        scheduler = WarmupScheduler(opt, warmup, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100), min_lr=min_lr)

        for _ in range(warmup):
            step(opt, scheduler)
            progress = min(scheduler.last_epoch, warmup) / warmup
            cosine = 0.5 * (1 - np.cos(np.pi * progress))
            expected = min_lr + (base - min_lr) * cosine
            assert opt.param_groups[0]["lr"] == pytest.approx(expected)
            assert scheduler.get_last_lr()[0] == pytest.approx(expected)

        assert scheduler.last_epoch == warmup
        assert opt.param_groups[0]["lr"] == pytest.approx(base)

    def test_handoff_to_inner_scheduler(self):
        base, warmup, t_max = 0.1, 3, 100
        opt = make_optimizer(base)
        inner = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
        scheduler = WarmupScheduler(opt, warmup, inner, min_lr=1e-9)

        for _ in range(warmup):
            step(opt, scheduler)
        # inner scheduler consumed epoch 0 at construction; first step after
        # warmup lands on epoch 1 and cosine decay kicks in
        step(opt, scheduler)
        expected = 0.5 * base * (1 + np.cos(np.pi * (1 / t_max)))
        assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6)
        assert scheduler.get_last_lr()[0] == pytest.approx(expected, rel=1e-6)

        step(opt, scheduler)
        expected = 0.5 * base * (1 + np.cos(np.pi * (2 / t_max)))
        assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6)

    def test_warmup_zero_skips_warmup(self):
        opt = make_optimizer(0.1)
        scheduler = WarmupScheduler(opt, 0, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100))
        step(opt, scheduler)
        # inner scheduler consumed epoch 0 at construction -> first step is epoch 1
        expected = 0.5 * 0.1 * (1 + np.cos(np.pi * (1 / 100)))
        assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-6)

    def test_last_epoch_increments_during_warmup(self):
        opt = make_optimizer(0.1)
        scheduler = WarmupScheduler(opt, 5, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100))
        assert scheduler.last_epoch == 0
        step(opt, scheduler)
        assert scheduler.last_epoch == 1

    def test_starts_from_min_lr(self):
        opt = make_optimizer(0.1)
        scheduler = WarmupScheduler(opt, 10, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100), min_lr=0.05)
        step(opt, scheduler)
        assert scheduler.last_epoch == 1
        progress = 1 / 10
        cosine = 0.5 * (1 - np.cos(np.pi * progress))
        expected = 0.05 + (0.1 - 0.05) * cosine
        assert opt.param_groups[0]["lr"] == pytest.approx(expected)
