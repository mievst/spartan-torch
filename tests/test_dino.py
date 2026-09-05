import copy

import pytest
import torch
from torch import nn

from spartan_torch import (
    Centering,
    DINOProjectionHead,
    DINOLoss,
    MomentumEncoder,
    Sharpening,
)


class TestDINOProjectionHead:
    def test_output_shape(self):
        head = DINOProjectionHead(in_dim=768, hidden_dim=2048, out_dim=65536)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 65536)

    def test_l2_normalized(self):
        head = DINOProjectionHead(in_dim=128, hidden_dim=256, out_dim=512)
        out = head(torch.randn(8, 128))
        assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-5)

    def test_gradient_flow(self):
        head = DINOProjectionHead(in_dim=64, hidden_dim=128, out_dim=256)
        x = torch.randn(3, 64, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        assert head.last_linear.weight.grad is not None

    def test_norm_last_layer_weight_norm(self):
        head = DINOProjectionHead(64, 128, 256, norm_last_layer=True)
        assert hasattr(head.last_linear, "parametrizations")
        bare = DINOProjectionHead(64, 128, 256, norm_last_layer=False)
        assert not hasattr(bare.last_linear, "parametrizations")

    def test_nb_layers_validated(self):
        with pytest.raises(ValueError, match="nb_layers"):
            DINOProjectionHead(64, 128, 256, nb_layers=1)


class TestCentering:
    def test_removes_mean(self):
        center = Centering(dim=32, momentum=0.5)
        x = torch.randn(64, 32) + 3.0
        # EMA center needs several steps to track the batch mean
        out = x
        for _ in range(50):
            out = center(out)
        assert torch.allclose(out.mean(dim=0), torch.zeros(32), atol=1e-3)

    def test_center_updates(self):
        center = Centering(dim=8, momentum=0.5)
        before = center.center.clone()
        center(torch.randn(32, 8))
        assert not torch.equal(center.center, before)

    def test_no_grad_through_center(self):
        center = Centering(dim=8)
        x = torch.randn(16, 8, requires_grad=True)
        out = center(x)
        assert out.requires_grad


class TestSharpening:
    def test_low_temp_peaks_distribution(self):
        sharp = Sharpening(temperature=0.04)
        logits = torch.randn(4, 16)
        out = sharp(logits)
        assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)
        peak = out.max(dim=-1).values.mean()
        assert peak > 0.3

    def test_high_temp_flattens(self):
        flat = Sharpening(temperature=10.0)(torch.randn(4, 16))
        assert flat.max(dim=-1).values.mean() < 0.2


class TestDINOLoss:
    def test_loss_positive_finite(self):
        loss = DINOLoss(out_dim=64, center_momentum=0.9)
        student = [torch.randn(8, 64, requires_grad=True), torch.randn(8, 64)]
        teacher = [torch.randn(8, 64), torch.randn(8, 64)]
        loss.update_center(torch.randn(8, 64))
        l = loss(student, teacher)
        assert l.dim() == 0
        assert torch.isfinite(l)
        l.backward()
        assert student[0].grad is not None

    def test_teacher_grads_stopped(self):
        loss = DINOLoss(out_dim=64)
        student = [torch.randn(8, 64, requires_grad=True), torch.randn(8, 64, requires_grad=True)]
        teacher = [torch.randn(8, 64, requires_grad=True), torch.randn(8, 64, requires_grad=True)]
        l = loss(student, teacher)
        l.backward()
        assert student[0].grad is not None
        assert student[1].grad is not None
        assert teacher[0].grad is None
        assert teacher[1].grad is None

    def test_identical_teacher_student_low_loss(self):
        loss = DINOLoss(out_dim=64, teacher_temp=0.04, student_temp=0.04)
        # one strongly-dominant logit -> near one-hot -> near-zero entropy
        logits = torch.zeros(8, 64)
        logits[:, 0] = 50.0
        l = loss([logits, logits], [logits, logits])
        assert l.item() < 1e-2


class TestMomentumEncoder:
    def _student(self):
        return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))

    def test_teacher_starts_equal(self):
        s = self._student()
        enc = MomentumEncoder.from_student(s, momentum=0.9)
        for tp, sp in zip(enc.teacher.parameters(), s.parameters(), strict=True):
            assert torch.equal(tp, sp)

    def test_update_moves_toward_student(self):
        s = self._student()
        enc = MomentumEncoder.from_student(s, momentum=0.5)
        old_teacher = [p.clone() for p in enc.teacher.parameters()]
        with torch.no_grad():
            for p in s.parameters():
                p.add_(1.0)
        enc.update()
        for tp, old_tp, sp in zip(
            enc.teacher.parameters(), old_teacher, s.parameters(), strict=True
        ):
            expected = 0.5 * old_tp + 0.5 * sp
            assert torch.allclose(tp, expected)

    def test_momentum_override(self):
        s = self._student()
        enc = MomentumEncoder.from_student(s, momentum=0.5)
        old_teacher = [p.clone() for p in enc.teacher.parameters()]
        with torch.no_grad():
            for p in s.parameters():
                p.add_(1.0)
        enc.update(momentum=0.9)
        for tp, old_tp, sp in zip(
            enc.teacher.parameters(), old_teacher, s.parameters(), strict=True
        ):
            expected = 0.9 * old_tp + 0.1 * sp
            assert torch.allclose(tp, expected)

    def test_teacher_frozen_no_grad(self):
        s = self._student()
        enc = MomentumEncoder.from_student(s)
        for p in enc.teacher.parameters():
            assert not p.requires_grad

    def test_momentum_validated(self):
        with pytest.raises(ValueError, match="momentum"):
            MomentumEncoder(self._student(), momentum=1.0)


class TestDINOLossViewAlignment:
    """validation_step must survive mismatched view counts
    (local student views have no teacher target)."""

    def _lightning(self):
        from experiments.vit.dino.dino_model import DINOLightning

        torch.manual_seed(0)
        backbone = dict(
            img_size=32, patch_size=4, in_channels=3, embed_dim=24,
            depth=1, num_heads=2, num_classes=10, ff_hidden_size=96,
        )
        return DINOLightning(
            backbone=backbone, head_hidden_dim=32, head_out_dim=64,
            lr=1e-4, warmup_epochs=1, max_epochs=1,
        )

    def _views(self, n_global=2, n_local=4, b=4, size=32, small=16):
        student = [torch.randn(b, 3, size, size) for _ in range(n_global)]
        student += [torch.randn(b, 3, small, small) for _ in range(n_local)]
        teacher = [torch.randn(b, 3, size, size) for _ in range(n_global)]
        return student, teacher

    def test_matched_views(self):
        pl = self._lightning()
        student, teacher = self._views()
        with torch.no_grad():
            teacher_out = [pl.teacher_encoder(v) for v in teacher]
        acc = pl._proto_accuracy(student, teacher_out)
        assert acc.dim() == 0 and 0.0 <= acc.item() <= 1.0

    def test_extra_local_views_are_skipped(self):
        # 2 global + 4 local student views vs 2 teacher views: must not raise
        pl = self._lightning()
        student, teacher = self._views(n_global=2, n_local=4)
        with torch.no_grad():
            teacher_out = [pl.teacher_encoder(v) for v in teacher]
        acc = pl._proto_accuracy(student, teacher_out)
        assert 0.0 <= acc.item() <= 1.0
