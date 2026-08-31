import pytest
import torch

from spartan_torch import ClassToken, LearnablePositionEmbedding, PatchEmbedding


class TestPatchEmbedding:
    @pytest.mark.parametrize("img_size,patch_size,expected_patches", [
        (32, 4, 64),
        (224, 16, 196),
        (224, 14, 256),
        (64, 8, 64),
    ])
    def test_output_shape_square(self, img_size, patch_size, expected_patches):
        layer = PatchEmbedding(in_channels=3, embed_dim=768, patch_size=patch_size)
        x = torch.randn(2, 3, img_size, img_size)
        out = layer(x)
        assert out.shape == (2, expected_patches, 768)

    @pytest.mark.parametrize("h,w,ph,pw", [
        (48, 64, 8, 8),
        (100, 120, 10, 10),
        (224, 256, 16, 16),
    ])
    def test_non_square_input(self, h, w, ph, pw):
        layer = PatchEmbedding(in_channels=3, embed_dim=512, patch_size=(ph, pw))
        x = torch.randn(1, 3, h, w)
        out = layer(x)
        expected_n = (h // ph) * (w // pw)
        assert out.shape == (1, expected_n, 512)

    def test_grayscale(self):
        layer = PatchEmbedding(in_channels=1, embed_dim=256, patch_size=4)
        x = torch.randn(1, 1, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 256)

    def test_gradient_flow(self):
        layer = PatchEmbedding(in_channels=3, embed_dim=384, patch_size=16)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert layer.proj.weight.grad is not None


class TestClassToken:
    def test_output_shape(self):
        cls = ClassToken(embed_dim=768)
        x = torch.randn(2, 196, 768)
        out = cls(x)
        assert out.shape == (2, 197, 768)

    def test_cls_token_at_position_zero(self):
        cls = ClassToken(embed_dim=128)
        x = torch.randn(4, 10, 128)
        out = cls(x)
        cls_repeated = cls.cls_token.expand(4, -1, -1)
        assert torch.allclose(out[:, 0], cls_repeated)

    def test_patch_tokens_unchanged(self):
        cls = ClassToken(embed_dim=128)
        x = torch.randn(4, 10, 128)
        out = cls(x)
        assert torch.allclose(out[:, 1:], x)

    def test_gradient_flow(self):
        cls = ClassToken(embed_dim=256)
        x = torch.randn(1, 50, 256, requires_grad=True)
        out = cls(x)
        out.sum().backward()
        assert x.grad is not None
        assert cls.cls_token.grad is not None


class TestLearnablePositionEmbedding:
    def test_output_shape(self):
        layer = LearnablePositionEmbedding(max_len=200, embed_dim=768)
        x = torch.randn(2, 197, 768)
        out = layer(x)
        assert out.shape == (2, 197, 768)

    def test_truncation_when_shorter(self):
        layer = LearnablePositionEmbedding(max_len=200, embed_dim=128)
        x = torch.randn(1, 10, 128)
        out = layer(x)
        assert out.shape == (1, 10, 128)

    def test_overflow_raises(self):
        layer = LearnablePositionEmbedding(max_len=10, embed_dim=64)
        x = torch.randn(1, 11, 64)
        with pytest.raises(IndexError, match="exceeds max_len"):
            layer(x)

    def test_positions_are_additive(self):
        layer = LearnablePositionEmbedding(max_len=100, embed_dim=64)
        x = torch.randn(1, 10, 64)
        out = layer(x)
        expected = x + layer.pos_embed[:, :10]
        assert torch.allclose(out, expected)

    def test_gradient_flow(self):
        layer = LearnablePositionEmbedding(max_len=100, embed_dim=256)
        x = torch.randn(1, 10, 256, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert layer.pos_embed.grad is not None


class TestComposition:
    def test_full_pipeline(self):
        patch_embed = PatchEmbedding(3, 768, 16)
        cls_token = ClassToken(768)
        pos_embed = LearnablePositionEmbedding(200, 768)

        x = torch.randn(2, 3, 224, 224)
        tokens = patch_embed(x)
        tokens = cls_token(tokens)
        tokens = pos_embed(tokens)
        assert tokens.shape == (2, 197, 768)
