"""Weight parity: compat assemblies from primitives vs timm/torchvision refs.

Compat assemblies live HERE (not in ``src``): full nets are out of concept
per AGENTS.md — the library ships blocks, tests prove the blocks compose
into reference-identical models and load reference weights.

Registry of reproduced architectures:

- ViT-Base/16 (``vit_base_patch16_224``) ← timm — compat ``CompatViT``
- ResNet-18 (BasicBlock × [2,2,2,2]) ← torchvision — compat ``CompatResNet18``

Gates (strict, CPU fp32, eval, fixed seed):

- ``max abs diff < 1e-5``, ``cosine similarity > 0.99999``
- ViT: ``load_state_dict(strict=True)``; ResNet: ``strict=False`` limited to
  ``num_batches_tracked`` buffers (long dtype, excluded from the float
  state_dict comparison by torch)

Markers: ``parity`` (needs ref libs), ``pretrained`` (needs network to
download weights). Random-weight parity runs offline.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from spartan_torch import (
    ClassToken,
    LearnablePositionEmbedding,
    PatchEmbedding,
    ResidualBlock,
    TransformerBlock,
)
from spartan_torch.compat import remap_timm_vit, remap_torchvision_resnet18

pytestmark = pytest.mark.parity

MAX_ABS_DIFF = 1e-5
MIN_COSINE = 0.99999


def parity_metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    max_diff = (a - b).abs().max().item()
    cos = F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()
    return max_diff, cos


def assert_parity(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    max_diff, cos = parity_metrics(a, b)
    assert max_diff < MAX_ABS_DIFF, f"max abs diff {max_diff:.3e} >= {MAX_ABS_DIFF:.0e}"
    assert cos > MIN_COSINE, f"cosine {cos:.8f} <= {MIN_COSINE}"
    return max_diff, cos


class CompatViT(nn.Module):
    """ViT-Base/16 assembly from primitives (test-only, mirrors timm layout).

    LayerNorm ``eps=1e-6`` reproduces timm exactly (``nn.LayerNorm`` defaults
    to ``1e-5``, which alone contributes ~1e-3 output drift over 12 blocks).
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        from functools import partial

        timm_norm = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(3, 768, 16)
        self.cls_token = ClassToken(768)
        self.pos_embed = LearnablePositionEmbedding(197, 768)
        self.encoder = nn.ModuleList([
            TransformerBlock(768, 64, 12, 768, 3072, qkv_bias=True, out_bias=True, norm_layer=timm_norm)
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embed(self.cls_token(self.patch_embed(x)))
        for block in self.encoder:
            x, _ = block(x)
        return self.head(self.norm(x)[:, 0])


class CompatResNet18(nn.Module):
    """ResNet-18 assembly from ResidualBlocks (test-only).

    Key layout mirrors torchvision: ``stem.0``/``stem.1`` (conv/bn),
    ``stages.{layer}.{block}.conv1/bn1/conv2/bn2/downsample.*``, ``fc.*``.
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        stages = []
        in_c = 64
        for li, (d, n) in enumerate(zip([64, 128, 256, 512], [2, 2, 2, 2])):
            blocks = []
            for bi in range(n):
                stride = 2 if (bi == 0 and li > 0) else 1
                blocks.append(ResidualBlock(in_c, d, stride=stride))
                in_c = d
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class TestViTRandomParity:
    def test_strict_load_and_forward(self):
        timm = pytest.importorskip("timm")
        ref = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
        remapped, report = remap_timm_vit(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = CompatViT(num_classes=10)
        own, got = set(ours.state_dict()), set(remapped)
        assert own == got, f"missing={sorted(own - got)} extra={sorted(got - own)}"
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), ref(x))
        print(f"\nViT random parity: max_diff={max_diff:.2e} cos={cos:.8f}")


class TestResNet18RandomParity:
    def test_load_and_forward(self):
        tv = pytest.importorskip("torchvision")
        ref = tv.models.resnet18(weights=None)
        remapped, report = remap_torchvision_resnet18(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = CompatResNet18(num_classes=1000)
        own, got = set(ours.state_dict()), set(remapped)
        # Only long-dtype num_batches_tracked buffers may be missing from the
        # remap comparison; everything float must match 1-to-1.
        missing = own - got
        assert all("num_batches_tracked" in k for k in missing), f"missing={sorted(missing)}"
        assert not (got - own), f"extra={sorted(got - own)}"
        ours.load_state_dict(remapped, strict=False)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), ref(x))
        print(f"\nResNet18 random parity: max_diff={max_diff:.2e} cos={cos:.8f}")


@pytest.mark.pretrained
class TestViTPretrainedParity:
    def test_pretrained_forward_and_accuracy_smoke(self):
        timm = pytest.importorskip("timm")
        try:
            ref = timm.create_model("vit_base_patch16_224", pretrained=True)
        except Exception as e:
            pytest.skip(f"timm weights download failed: {e}")
        remapped, _ = remap_timm_vit(ref.state_dict())
        ours = CompatViT(num_classes=1000)
        ours.load_state_dict(remapped, strict=True)
        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), ref(x))
            # Accuracy smoke on synthetic batch: same weights ⇒ same preds.
            acc = (ours(x).argmax(-1) == ref(x).argmax(-1)).float().mean().item()
        assert acc == 1.0
        print(f"\nViT pretrained parity: max_diff={max_diff:.2e} cos={cos:.8f}")


@pytest.mark.pretrained
class TestResNet18PretrainedParity:
    def test_pretrained_forward_and_accuracy_smoke(self):
        tv = pytest.importorskip("torchvision")
        try:
            ref = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            pytest.skip(f"torchvision weights download failed: {e}")
        remapped, _ = remap_torchvision_resnet18(ref.state_dict())
        ours = CompatResNet18(num_classes=1000)
        ours.load_state_dict(remapped, strict=False)
        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), ref(x))
            acc = (ours(x).argmax(-1) == ref(x).argmax(-1)).float().mean().item()
        assert acc == 1.0
        print(f"\nResNet18 pretrained parity: max_diff={max_diff:.2e} cos={cos:.8f}")
