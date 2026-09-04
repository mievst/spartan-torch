"""Verify state_dict compatibility between our ViT and timm's ViT-Base/16.

Thin demo over :mod:`spartan_torch.compat` (key-mapping lives there, tested
by ``tests/test_weight_parity.py``). Loads a timm ``vit_base_patch16_224``
(random weights), remaps into the compat ViT assembly, strict-loads and
compares one forward pass.

Usage::

    uv run python experiments/vit/timm_compat/check_weights.py
"""

import sys
from functools import partial
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timm

from spartan_torch import (
    ClassToken,
    LearnablePositionEmbedding,
    PatchEmbedding,
    TransformerBlock,
)
from spartan_torch.compat import apply_remap, remap_timm_vit


class CompatViT(nn.Module):
    """Test-only ViT assembly mirroring tests/test_weight_parity.py."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
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


def main() -> None:
    timm_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10).eval()
    remapped, report = remap_timm_vit(timm_model.state_dict())

    our_model = CompatViT(num_classes=10)
    apply_remap(our_model, remapped, report)

    print(f"timm keys:    {report.source_keys}")
    print(f"remapped:     {report.remapped_keys}")
    print(f"missing (our): {report.missing}")
    print(f"extra (ours):   {report.extra}")
    print(f"unmatched (timm): {report.unmatched_source}")

    if report.ok:
        print("\nAll keys match! load_state_dict(strict=True) succeeded.")
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            diff = (our_model(x) - timm_model(x)).abs().max().item()
        print(f"Max output diff (random init): {diff:.2e}")


if __name__ == "__main__":
    main()
