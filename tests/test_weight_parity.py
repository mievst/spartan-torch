"""Weight parity: compat assemblies from primitives vs timm/torchvision refs.

Compat assemblies live HERE (not in ``src``): full nets are out of concept
per AGENTS.md — the library ships blocks, tests prove the blocks compose
into reference-identical models and load reference weights.

Registry of reproduced architectures:

- ViT-Base/16 (``vit_base_patch16_224``) ← timm — compat ``CompatViT``
- ResNet-18 (BasicBlock × [2,2,2,2]) ← torchvision — compat ``CompatResNet18``
- LLaMA MLP (``gate/up/down_proj``) ← transformers — ``SwiGLUFeedForward``
  (block-level, identical keys)
- LLaMA rotary (``LlamaRotaryEmbedding`` + ``apply_rotary_pos_emb``) ←
  transformers — ``RotaryPositionalEmbedding`` (math-level, no weights)
- MobileNetV2 ``InvertedResidual`` (``expand_ratio=6``) ← torchvision —
  block-level (``expansion=1`` and stride-2 shortcuts differ by design, see
  ``compat/hf_llama.py``)
- DINO ``DINOHead`` ← facebookresearch/dino (``vision_transformer.py``) —
  ``DINOProjectionHead`` (block-level, random-weight parity; the reference
  implementation is vendored below — upstream has no installable package,
  so there is no ``pretrained`` leg, only the offline ``parity`` one)
- DINO ViT-Small/16 backbone ← facebookresearch/dino torch.hub
  (``dino_deitsmall16_pretrain.pth``) — compat ``CompatViTSmall``
  (``pretrained`` leg, backbone features only: hub weights ship no head)

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
    DINOProjectionHead,
    InvertedResidual,
    LearnablePositionEmbedding,
    Mamba2Mixer,
    Mamba3Mixer,
    MambaMixer,
    PatchEmbedding,
    ResidualBlock,
    RotaryPositionalEmbedding,
    SwiGLUFeedForward,
    TransformerBlock,
)
from spartan_torch.compat import (
    hf_mamba_kwargs,
    hf_mamba2_kwargs,
    official_mamba3_kwargs,
    remap_dino_head,
    remap_hf_llama_mlp,
    remap_hf_mamba_mixer,
    remap_hf_mamba2_mixer,
    remap_hf_mamba3_mixer,
    remap_mamba_ssm_mamba,
    remap_mamba_ssm_mamba2,
    remap_mamba_ssm_mamba3,
    remap_timm_vit,
    remap_torchvision_mobilenet_block,
    remap_torchvision_resnet18,
)

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


class CompatViTSmall(nn.Module):
    """ViT-Small/16 backbone assembly from primitives (test-only).

    Mirrors ``facebookresearch/dino`` ``vit_small`` (``num_classes=0``):
    embed 384, depth 12, heads 6, patch 16, ``qkv_bias=True``,
    ``LayerNorm(eps=1e-6)``. Forward returns post-norm ``[CLS]`` features,
    like the hub model. Key layout is the timm one, so ``remap_timm_vit``
    applies unchanged (it is dimension-agnostic).
    """

    def __init__(self):
        super().__init__()
        from functools import partial

        dino_norm = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(3, 384, 16)
        self.cls_token = ClassToken(384)
        self.pos_embed = LearnablePositionEmbedding(197, 384)
        self.encoder = nn.ModuleList([
            TransformerBlock(384, 64, 6, 384, 1536, qkv_bias=True, out_bias=True, norm_layer=dino_norm)
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(384, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embed(self.cls_token(self.patch_embed(x)))
        for block in self.encoder:
            x, _ = block(x)
        return self.norm(x)[:, 0]


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


class TestSwiGLUHfParity:
    @pytest.mark.parametrize("hidden,inter", [(256, 512), (128, 384)])
    def test_llama_mlp_weights_and_forward(self, hidden, inter):
        hf = pytest.importorskip("transformers")
        cfg = hf.LlamaConfig(hidden_size=hidden, intermediate_size=inter)
        ref = hf.models.llama.modeling_llama.LlamaMLP(cfg)
        remapped, report = remap_hf_llama_mlp(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = SwiGLUFeedForward(hidden, inter, bias=False)
        assert set(ours.state_dict()) == set(remapped)
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 8, hidden)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), ref(x))
        print(f"\nSwiGLU↔LlamaMLP parity ({hidden}/{inter}): max_diff={max_diff:.2e} cos={cos:.8f}")


class TestRoPEHfParity:
    @pytest.mark.parametrize("offset", [0, 17])
    def test_llama_rotary_matches(self, offset):
        hf = pytest.importorskip("transformers")
        B, H, S, D = 2, 4, 12, 64
        cfg = hf.LlamaConfig(hidden_size=H * D, num_attention_heads=H,
                             intermediate_size=128, max_position_embeddings=256)
        ref_rot = hf.models.llama.modeling_llama.LlamaRotaryEmbedding(cfg)
        ours = RotaryPositionalEmbedding(D, max_seq_len=256)

        torch.manual_seed(0)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        pos = torch.arange(offset, offset + S).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            cos, sin = ref_rot(q.transpose(1, 2), pos)
            ref_q, ref_k = hf.models.llama.modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)
            got_q, got_k = ours(q, k, pos[0], pos[0])
            dq, cq = assert_parity(got_q, ref_q)
            dk, ck = assert_parity(got_k, ref_k)
        print(f"\nRoPE↔HF parity (offset={offset}): max_diff={max(dq, dk):.2e}")


class TestMobileNetBlockParity:
    @pytest.mark.parametrize("in_c,out_c,stride", [(24, 32, 2), (32, 64, 2), (64, 96, 1), (160, 320, 1)])
    def test_inverted_residual_block(self, in_c, out_c, stride):
        tv = pytest.importorskip("torchvision")
        from torchvision.models.mobilenetv2 import InvertedResidual as TvBlock

        ref = TvBlock(in_c, out_c, stride, expand_ratio=6)
        remapped, report = remap_torchvision_mobilenet_block(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        # torchvision drops the residual on stride-2 (no projection shortcut).
        ours = InvertedResidual(in_c, out_c, stride=stride, expansion=6,
                                use_skip=(stride == 1 and in_c == out_c))
        assert set(ours.state_dict()) == set(remapped), (
            f"missing={sorted(set(ours.state_dict()) - set(remapped))}")
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(1, in_c, 28, 28)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), ref(x))
        print(f"\nMobileNet block parity ({in_c}->{out_c} s{stride}): max_diff={max_diff:.2e} cos={cos:.8f}")


@pytest.mark.pretrained
class TestMobileNetBlockPretrainedParity:
    def test_pretrained_block_slice(self):
        tv = pytest.importorskip("torchvision")
        try:
            ref = tv.models.mobilenet_v2(weights=tv.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception as e:
            pytest.skip(f"torchvision weights download failed: {e}")
        # features[3]: InvertedResidual(24, 24, s=1, e=6).
        tv_block = ref.features[3]
        remapped, _ = remap_torchvision_mobilenet_block(tv_block.state_dict())
        ours = InvertedResidual(24, 24, stride=1, expansion=6)
        ours.load_state_dict(remapped, strict=True)
        ours.eval()
        tv_block.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 24, 28, 28)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), tv_block(x))
        print(f"\nMobileNet pretrained block parity: max_diff={max_diff:.2e} cos={cos:.8f}")


class RefDINOHead(nn.Module):
    """Vendored reference: ``DINOHead`` from facebookresearch/dino.

    Verbatim copy of ``vision_transformer.py::DINOHead`` (Apache-2.0,
    Caron et al., 2021, arXiv:2104.14294), kept here because upstream ships
    no installable package (``pytest.importorskip`` is impossible). Only used
    as the parity target for :class:`DINOProjectionHead`.
    """

    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class TestDINOHeadParity:
    # RefDINOHead intentionally uses the deprecated nn.utils.weight_norm API:
    # only it produces the upstream weight_g/weight_v key layout we remap from.
    @pytest.mark.filterwarnings("ignore::FutureWarning")
    @pytest.mark.parametrize("norm_last_layer", [True, False])
    def test_dino_head_weights_and_forward(self, norm_last_layer):
        ref = RefDINOHead(64, 256, norm_last_layer=norm_last_layer,
                          nlayers=3, hidden_dim=128, bottleneck_dim=32)
        remapped, report = remap_dino_head(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = DINOProjectionHead(64, 128, 256, bottleneck_dim=32,
                                  norm_last_layer=norm_last_layer)
        assert set(ours.state_dict()) == set(remapped), (
            f"missing={sorted(set(ours.state_dict()) - set(remapped))}")
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(4, 64)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), ref(x))
        print(f"\nDINO head parity (norm_last_layer={norm_last_layer}): "
              f"max_diff={max_diff:.2e} cos={cos:.8f}")


DINO_VITS16_URL = (
    "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
)


@pytest.mark.pretrained
class TestDINOSmallPretrainedParity:
    def test_dino_vits16_backbone(self):
        pytest.importorskip("torchvision")  # required by the dino hubconf
        try:
            ref = torch.hub.load("facebookresearch/dino:main", "dino_vits16", pretrained=False)
        except Exception as e:
            pytest.skip(f"dino repo checkout failed: {e}")
        try:
            sd = torch.hub.load_state_dict_from_url(DINO_VITS16_URL, map_location="cpu")
        except Exception as e:
            pytest.skip(f"dino weights download failed: {e}")
        try:
            ref.load_state_dict(sd, strict=True)
        except Exception as e:
            pytest.skip(f"dino weights layout changed: {e}")

        remapped, report = remap_timm_vit({k: v.cpu() for k, v in sd.items()})
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = CompatViTSmall()
        assert set(ours.state_dict()) == set(remapped), (
            f"missing={sorted(set(ours.state_dict()) - set(remapped))}")
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x), ref(x))
        print(f"\nDINO vits16 backbone parity: max_diff={max_diff:.2e} cos={cos:.8f}")


def _mamba_hf_mixer_config(hf, **overrides):
    """Small HF MambaConfig with the CUDA-only flags pinned off."""
    kw = {
        "hidden_size": 32,
        "state_size": 8,
        "num_hidden_layers": 1,
        "expand": 2,
        "conv_kernel": 4,
        "use_bias": False,
        "use_conv_bias": True,
        "use_mambapy": False,
    }
    kw.update(overrides)
    return hf.MambaConfig(**kw)


class TestMambaHfParity:
    @pytest.mark.parametrize("use_bias,use_conv_bias", [(False, True), (True, True)])
    def test_hf_mixer_weights_and_forward(self, use_bias, use_conv_bias):
        hf = pytest.importorskip("transformers")
        cfg = _mamba_hf_mixer_config(hf, use_bias=use_bias, use_conv_bias=use_conv_bias)
        ref = hf.models.mamba.modeling_mamba.MambaMixer(cfg, layer_idx=0)
        remapped, report = remap_hf_mamba_mixer(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = MambaMixer(**hf_mamba_kwargs(cfg), use_fast_path=False, use_associative_scan=False)
        assert set(ours.state_dict()) == set(remapped), (
            f"missing={sorted(set(ours.state_dict()) - set(remapped))}")
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 12, cfg.hidden_size)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x)[0], ref(x))
        print(f"\nMambaMixer↔HF parity (bias={use_bias}): max_diff={max_diff:.2e} cos={cos:.8f}")

    def test_hf_mixer_step_matches_prefill(self):
        hf = pytest.importorskip("transformers")
        cfg = _mamba_hf_mixer_config(hf)
        ref = hf.models.mamba.modeling_mamba.MambaMixer(cfg, layer_idx=0)
        remapped, _ = remap_hf_mamba_mixer(ref.state_dict())
        ours = MambaMixer(**hf_mamba_kwargs(cfg), use_fast_path=False, use_associative_scan=False)
        ours.load_state_dict(remapped, strict=True)
        ours.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 7, cfg.hidden_size)
        with torch.no_grad():
            full = ours(x)[0]
            cache = ours.init_cache(1)
            outs = []
            for t in range(x.size(1)):
                o, cache = ours(x[:, t : t + 1], cache)
                outs.append(o)
            max_diff, cos = assert_parity(torch.cat(outs, dim=1), full)
        print(f"\nMambaMixer step↔prefill (HF weights): max_diff={max_diff:.2e} cos={cos:.8f}")


class TestMambaSsmRepoParity:
    def test_official_weights_and_forward(self):
        ssm = pytest.importorskip("mamba_ssm")
        ref = ssm.Mamba(d_model=32, d_state=8, d_conv=4, expand=2)
        remapped, report = remap_mamba_ssm_mamba(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = MambaMixer(32, d_state=8, d_conv=4, expand=2, use_fast_path=False,
                          use_associative_scan=False)
        assert set(ours.state_dict()) == set(remapped), (
            f"missing={sorted(set(ours.state_dict()) - set(remapped))}")
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 12, 32)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x)[0], ref(x))
        print(f"\nMambaMixer↔mamba-ssm parity: max_diff={max_diff:.2e} cos={cos:.8f}")


@pytest.mark.pretrained
class TestMamba130mPretrainedParity:
    def test_first_mixer_block(self):
        hf = pytest.importorskip("transformers")
        try:
            model = hf.MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        except Exception as e:
            pytest.skip(f"mamba-130m-hf download failed: {e}")
        cfg = model.config
        ref = model.backbone.layers[0].mixer.eval()
        remapped, report = remap_hf_mamba_mixer(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = MambaMixer(**hf_mamba_kwargs(cfg), use_fast_path=False,
                          use_associative_scan=False).eval()
        ours.load_state_dict(remapped, strict=True)

        torch.manual_seed(0)
        norm = model.backbone.layers[0].norm
        x = norm(torch.randn(2, 16, cfg.hidden_size))
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x)[0], ref(x))
        print(f"\nmamba-130m-hf mixer[0] parity: max_diff={max_diff:.2e} cos={cos:.8f}")


def _mamba2_hf_mixer_config(hf, **overrides):
    """Small HF Mamba2Config (E == H*P enforced). CUDA flags pinned off."""
    kw = {
        "hidden_size": 32,
        "state_size": 8,
        "num_heads": 4,
        "head_dim": 16,
        "n_groups": 2,
        "expand": 2,
        "conv_kernel": 4,
        "chunk_size": 8,
        "use_bias": False,
        "use_conv_bias": True,
    }
    kw.update(overrides)
    return hf.Mamba2Config(**kw)


class TestMamba2HfParity:
    @pytest.mark.parametrize("use_bias,use_conv_bias", [(False, True), (True, True)])
    def test_hf_mixer_weights_and_forward(self, use_bias, use_conv_bias):
        hf = pytest.importorskip("transformers")
        cfg = _mamba2_hf_mixer_config(hf, use_bias=use_bias, use_conv_bias=use_conv_bias)
        ref = hf.models.mamba2.modeling_mamba2.Mamba2Mixer(cfg, layer_idx=0)
        remapped, report = remap_hf_mamba2_mixer(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = Mamba2Mixer(**hf_mamba2_kwargs(cfg), use_fast_path=False)
        assert set(ours.state_dict()) == set(remapped), (
            f"missing={sorted(set(ours.state_dict()) - set(remapped))}")
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 10, cfg.hidden_size)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x)[0], ref(x))
        print(f"\nMamba2Mixer↔HF parity (bias={use_bias}): max_diff={max_diff:.2e} cos={cos:.8f}")

    def test_hf_mixer_step_matches_prefill(self):
        hf = pytest.importorskip("transformers")
        cfg = _mamba2_hf_mixer_config(hf)
        ref = hf.models.mamba2.modeling_mamba2.Mamba2Mixer(cfg, layer_idx=0)
        remapped, _ = remap_hf_mamba2_mixer(ref.state_dict())
        ours = Mamba2Mixer(**hf_mamba2_kwargs(cfg), use_fast_path=False)
        ours.load_state_dict(remapped, strict=True)
        ours.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 7, cfg.hidden_size)
        with torch.no_grad():
            full = ours(x)[0]
            cache = ours.init_cache(1)
            outs = []
            for t in range(x.size(1)):
                o, cache = ours(x[:, t : t + 1], cache)
                outs.append(o)
            max_diff, cos = assert_parity(torch.cat(outs, dim=1), full)
        print(f"\nMamba2Mixer step↔prefill (HF weights): max_diff={max_diff:.2e} cos={cos:.8f}")


class TestMamba2SsmRepoParity:
    def test_official_weights_and_forward(self):
        ssm = pytest.importorskip("mamba_ssm")
        ref = ssm.Mamba2(d_model=32, d_state=8, d_conv=4, expand=2,
                         headdim=16, ngroups=2)
        remapped, report = remap_mamba_ssm_mamba2(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = Mamba2Mixer(32, d_state=8, d_conv=4, expand=2, num_heads=4,
                           head_dim=16, n_groups=2, chunk_size=8, use_fast_path=False)
        assert set(ours.state_dict()) == set(remapped), (
            f"missing={sorted(set(ours.state_dict()) - set(remapped))}")
        ours.load_state_dict(remapped, strict=True)

        ours.eval()
        ref.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 10, 32)
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x)[0], ref(x))
        print(f"\nMamba2Mixer↔mamba-ssm parity: max_diff={max_diff:.2e} cos={cos:.8f}")


@pytest.mark.pretrained
class TestMamba2130mPretrainedParity:
    def test_first_mixer_block(self):
        hf = pytest.importorskip("transformers")
        try:
            model = hf.Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")
        except Exception as e:
            pytest.skip(f"mamba2-130m-hf download failed: {e}")
        cfg = model.config
        ref = model.backbone.layers[0].mixer.eval()
        remapped, report = remap_hf_mamba2_mixer(ref.state_dict())
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"

        ours = Mamba2Mixer(**hf_mamba2_kwargs(cfg), use_fast_path=False).eval()
        ours.load_state_dict(remapped, strict=True)

        torch.manual_seed(0)
        norm = model.backbone.layers[0].norm
        x = norm(torch.randn(2, 8, cfg.hidden_size))
        with torch.no_grad():
            max_diff, cos = assert_parity(ours(x)[0], ref(x))
        print(f"\nmamba2-130m-hf mixer[0] parity: max_diff={max_diff:.2e} cos={cos:.8f}")


@pytest.mark.pretrained
class TestMamba3370mPretrainedWeights:
    # No runnable reference exists in this environment (official kernels are
    # Linux/CUDA-only, transformers ships no Mamba-3): coverage is
    # strict-loading the real ib-ssm checkpoint plus self-agreement
    # (step-vs-prefill) on those weights.
    def test_real_weights_strict_load_and_self_agreement(self):
        pytest.importorskip("safetensors")
        from huggingface_hub import hf_hub_download

        try:
            cfg_path = hf_hub_download("ib-ssm/mamba3-370M-10BT", "config.json")
            st_path = hf_hub_download("ib-ssm/mamba3-370M-10BT", "model.safetensors")
        except Exception as e:
            pytest.skip(f"mamba3-370M download failed: {e}")
        import json

        from safetensors import safe_open

        cfg = json.load(open(cfg_path))
        assert cfg["is_mimo"] is False
        ours = Mamba3Mixer(**official_mamba3_kwargs(cfg)).eval()

        prefix = "backbone.layers.0.mixer."
        with safe_open(st_path, framework="pt") as f:
            sd = {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
        remapped, report = remap_mamba_ssm_mamba3(sd)
        assert report.unmatched_source == [], f"unmatched: {report.unmatched_source}"
        assert set(ours.state_dict()) == set(remapped), (
            f"missing={sorted(set(ours.state_dict()) - set(remapped))}")
        ours.load_state_dict(remapped, strict=True)

        torch.manual_seed(0)
        x = torch.randn(1, 6, ours.d_model)
        with torch.no_grad():
            full = ours(x)[0]
            assert torch.isfinite(full).all()
            cache = ours.init_cache(1)
            outs = []
            for t in range(x.size(1)):
                o, cache = ours(x[:, t : t + 1], cache)
                outs.append(o)
            max_diff, cos = parity_metrics(torch.cat(outs, dim=1), full)
            # Self-agreement (no runnable reference): looser gate than the
            # 1e-5 ref-parity one — RoPE cumsum ordering across steps.
            assert max_diff < 1e-4, f"max abs diff {max_diff:.3e} >= 1e-4"
            assert cos > MIN_COSINE, f"cosine {cos:.8f} <= {MIN_COSINE}"
        print(f"\nmamba3-370M mixer[0] self-agreement: max_diff={max_diff:.2e} cos={cos:.8f}")
