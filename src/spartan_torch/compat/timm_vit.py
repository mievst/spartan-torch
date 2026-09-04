"""Remap timm ViT state_dict keys to the spartan-torch compat ViT layout.

Source: ``timm.create_model("vit_base_patch16_224", ...)``.
Target: compat ViT assembled from :class:`PatchEmbedding`,
:class:`ClassToken`, :class:`LearnablePositionEmbedding` and
:class:`TransformerBlock` with ``qkv_bias=True`` (see
``tests/test_weight_parity.py``)::

    cls_token                  → cls_token.cls_token
    pos_embed                  → pos_embed.pos_embed
    patch_embed.proj.*         → patch_embed.proj.*
    blocks.N.norm1.*           → encoder.N.norm1.*
    blocks.N.attn.qkv.weight   → encoder.N.attn.{query,key,value}_matrix.weight
    blocks.N.attn.qkv.bias     → encoder.N.attn.{query,key,value}_matrix.bias
    blocks.N.attn.proj.*       → encoder.N.attn.out.*
    blocks.N.norm2.*           → encoder.N.norm2.*
    blocks.N.mlp.fc1.*         → encoder.N.ff.layers.lin_1.*
    blocks.N.mlp.fc2.*         → encoder.N.ff.layers.lin_2.*
    norm.*                     → norm.*
    head.*                     → head.*

References
----------
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2020, arXiv:2010.11929).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RemapReport:
    """Key-coverage report of a remap operation."""

    source_keys: int = 0
    remapped_keys: int = 0
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    unmatched_source: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unmatched_source


def remap_timm_vit(
    timm_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Remap a timm ViT state_dict to the compat ViT layout.

    The fused ``attn.qkv.{weight,bias}`` tensors are split into thirds
    (query/key/value). Raises :class:`ValueError` if a fused dim is not
    divisible by 3.
    """
    remapped: dict[str, torch.Tensor] = {}
    qkv_w: dict[int, torch.Tensor] = {}
    qkv_b: dict[int, torch.Tensor] = {}
    unmatched: list[str] = []

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
            if rest == "attn.qkv.weight":
                qkv_w[idx] = val
            elif rest == "attn.qkv.bias":
                qkv_b[idx] = val
            elif rest.startswith("attn.proj."):
                remapped[f"encoder.{idx}.attn.out.{rest.removeprefix('attn.proj.')}"] = val
            elif rest.startswith("norm1."):
                remapped[f"encoder.{idx}.norm1.{rest.removeprefix('norm1.')}"] = val
            elif rest.startswith("norm2."):
                remapped[f"encoder.{idx}.norm2.{rest.removeprefix('norm2.')}"] = val
            elif rest.startswith("mlp.fc1."):
                remapped[f"encoder.{idx}.ff.layers.lin_1.{rest.removeprefix('mlp.fc1.')}"] = val
            elif rest.startswith("mlp.fc2."):
                remapped[f"encoder.{idx}.ff.layers.lin_2.{rest.removeprefix('mlp.fc2.')}"] = val
            else:
                unmatched.append(key)
        elif key.startswith("norm.") or key.startswith("head."):
            remapped[key] = val
        else:
            unmatched.append(key)

    for idx, fused in qkv_w.items():
        if fused.shape[0] % 3 != 0:
            raise ValueError(f"blocks.{idx}.attn.qkv.weight dim {fused.shape[0]} not divisible by 3")
        q, k, v = fused.chunk(3, dim=0)
        remapped[f"encoder.{idx}.attn.query_matrix.weight"] = q
        remapped[f"encoder.{idx}.attn.key_matrix.weight"] = k
        remapped[f"encoder.{idx}.attn.value_matrix.weight"] = v
    for idx, fused in qkv_b.items():
        if fused.shape[0] % 3 != 0:
            raise ValueError(f"blocks.{idx}.attn.qkv.bias dim {fused.shape[0]} not divisible by 3")
        q, k, v = fused.chunk(3, dim=0)
        remapped[f"encoder.{idx}.attn.query_matrix.bias"] = q
        remapped[f"encoder.{idx}.attn.key_matrix.bias"] = k
        remapped[f"encoder.{idx}.attn.value_matrix.bias"] = v

    report = RemapReport(source_keys=len(timm_sd), remapped_keys=len(remapped), unmatched_source=sorted(unmatched))
    return remapped, report


def apply_remap(
    model: torch.nn.Module,
    remapped: dict[str, torch.Tensor],
    report: RemapReport,
) -> RemapReport:
    """Strict-load ``remapped`` into ``model`` and fill missing/extra in ``report``."""
    own = set(model.state_dict().keys())
    got = set(remapped.keys())
    report.missing = sorted(own - got)
    report.extra = sorted(got - own)
    model.load_state_dict(remapped, strict=True)
    return report
