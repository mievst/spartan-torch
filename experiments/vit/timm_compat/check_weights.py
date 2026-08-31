"""Verify state_dict compatibility between our ViT and timm's ViT-Base/16.

Loads a timm ``vit_base_patch16_224`` (random weights) and remaps its
state_dict into our :class:`VisionTransformer`. Prints a summary of
matched / unmatched / missing keys.

Usage::

    uv run python experiments/vit/timm_compat/check_weights.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timm

from vision_transformer import VisionTransformer


def remap_timm_to_ours(timm_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap timm ViT-Base state_dict keys to our VisionTransformer layout.

    Mapping:
        cls_token                  → cls_token.cls_token
        pos_embed                  → pos_embed.pos_embed
        patch_embed.proj.*         → patch_embed.proj.*
        blocks.N.norm1.*           → encoder.N.norm1.*
        blocks.N.attn.qkv.weight   → encoder.N.attn.query_matrix.weight
                                    + encoder.N.attn.key_matrix.weight
                                    + encoder.N.attn.value_matrix.weight
        blocks.N.attn.proj.*       → encoder.N.attn.out.*
        blocks.N.norm2.*           → encoder.N.norm2.*
        blocks.N.mlp.fc1.*         → encoder.N.ff.layers.lin_1.*
        blocks.N.mlp.fc2.*         → encoder.N.ff.layers.lin_2.*
        norm.*                     → norm.*
        head.*                     → head.*
    """
    remapped: dict[str, torch.Tensor] = {}
    qkv_buf: dict[int, torch.Tensor] = {}

    for key, val in timm_sd.items():
        if key == "cls_token":
            remapped["cls_token.cls_token"] = val
        elif key == "pos_embed":
            remapped["pos_embed.pos_embed"] = val
        elif key.startswith("patch_embed."):
            remapped[key] = val
        elif key.startswith("blocks."):
            parts = key.split(".", 2)
            idx = int(parts[1])
            rest = parts[2]
            if rest.startswith("attn.qkv."):
                suffix = rest.replace("attn.qkv.", "")
                qkv_buf.setdefault(idx, None)
                if suffix == "weight":
                    qkv_buf[idx] = val
            elif rest.startswith("attn.proj."):
                suffix = rest.replace("attn.proj.", "")
                remapped[f"encoder.{idx}.attn.out.{suffix}"] = val
            elif rest.startswith("norm1."):
                remapped[f"encoder.{idx}.norm1.{rest.removeprefix('norm1.')}"] = val
            elif rest.startswith("norm2."):
                remapped[f"encoder.{idx}.norm2.{rest.removeprefix('norm2.')}"] = val
            elif rest.startswith("mlp.fc1."):
                suffix = rest.replace("mlp.fc1.", "")
                remapped[f"encoder.{idx}.ff.layers.lin_1.{suffix}"] = val
            elif rest.startswith("mlp.fc2."):
                suffix = rest.replace("mlp.fc2.", "")
                remapped[f"encoder.{idx}.ff.layers.lin_2.{suffix}"] = val
        elif key.startswith("norm."):
            remapped[key] = val
        elif key.startswith("head."):
            remapped[key] = val

    for idx, fused_weight in qkv_buf.items():
        if fused_weight is None:
            continue
        d = fused_weight.shape[0] // 3
        q_w, k_w, v_w = fused_weight[:d], fused_weight[d : 2 * d], fused_weight[2 * d :]
        remapped[f"encoder.{idx}.attn.query_matrix.weight"] = q_w
        remapped[f"encoder.{idx}.attn.key_matrix.weight"] = k_w
        remapped[f"encoder.{idx}.attn.value_matrix.weight"] = v_w

    return remapped


def main() -> None:
    timm_model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=10
    )
    timm_sd = timm_model.state_dict()
    our_sd = remap_timm_to_ours(timm_sd)

    our_model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12,
        num_heads=12, ff_hidden_size=3072, num_classes=10,
    )
    our_keys = set(our_model.state_dict().keys())
    mapped_keys = set(our_sd.keys())

    missing = our_keys - mapped_keys
    extra = mapped_keys - our_keys
    matched = our_keys & mapped_keys

    print(f"timm keys:    {len(timm_sd)}")
    print(f"our keys:     {len(our_keys)}")
    print(f"mapped:       {len(matched)}")
    print(f"missing (our): {len(missing)}")
    print(f"extra (ours):   {len(extra)}")

    if missing:
        print("\nMissing keys (in our model, not in remapped timm):")
        for k in sorted(missing):
            print(f"  {k}")

    if extra:
        print("\nExtra keys (in remapped timm, not in our model):")
        for k in sorted(extra):
            print(f"  {k}")

    if not missing and not extra:
        print("\nAll keys match!")
        our_model.load_state_dict(our_sd, strict=True)
        print("load_state_dict(strict=True) succeeded.")

        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            our_out = our_model(x)
            timm_out = timm_model(x)
        diff = (our_out - timm_out).abs().max().item()
        print(f"Max output diff (random init): {diff:.2e}")


if __name__ == "__main__":
    main()
