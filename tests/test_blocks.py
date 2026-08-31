import functools

import pytest
import torch
from torch import nn

from spartan_torch import (
    Bottleneck,
    DepthwiseSeparableConv,
    InvertedResidual,
    ResidualBlock,
)


def reference_residual(block, x):
    identity = x
    out = block.activation(block.bn1(block.conv1(x)))
    out = block.dropout(block.bn2(block.conv2(out)))
    if block.use_skip:
        identity = block.downsample(identity)
        out = out + identity
    return block.activation(out)


def reference_bottleneck(block, x):
    identity = x
    out = block.activation(block.bn1(block.conv1(x)))
    out = block.activation(block.bn2(block.conv2(out)))
    out = block.dropout(block.bn3(block.conv3(out)))
    if block.use_skip:
        identity = block.downsample(identity)
        out = out + identity
    return block.activation(out)


def reference_depthwise_sep(block, x):
    identity = x
    out = block.activation(block.bn1(block.conv1(x)))
    out = block.dropout(block.bn2(block.conv2(out)))
    if block.use_skip:
        out = out + identity
    return block.activation(out)


def reference_inverted_residual(block, x):
    identity = x
    if block.use_expand:
        out = block.activation(block.bn1(block.conv1(x)))
    else:
        out = x
    out = block.activation(block.bn2(block.conv2(out)))
    out = block.dropout(block.bn3(block.conv3(out)))
    if block.use_skip:
        out = out + identity
    return out


class TestResidualBlock:
    @pytest.mark.parametrize("in_c,out_c,stride,size", [
        (16, 16, 1, 32),
        (16, 32, 1, 32),
        (16, 32, 2, 32),
        (3, 64, 2, 32),
    ])
    def test_shapes(self, in_c, out_c, stride, size):
        block = ResidualBlock(in_c, out_c, stride=stride)
        x = torch.randn(2, in_c, size, size)
        out = block(x)
        expected = size // stride
        assert out.shape == (2, out_c, expected, expected)

    def test_matches_reference(self):
        torch.manual_seed(0)
        block = ResidualBlock(16, 16).eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_residual(block, x), atol=1e-6)

    def test_no_skip_is_plain_stack(self):
        torch.manual_seed(0)
        block = ResidualBlock(16, 16, use_skip=False).eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_residual(block, x), atol=1e-6)

    def test_downsample_identity_matches_reference(self):
        torch.manual_seed(0)
        block = ResidualBlock(16, 32, stride=2).eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_residual(block, x), atol=1e-6)

    def test_custom_activation_and_norm(self):
        block = ResidualBlock(16, 16, activation=nn.GELU, norm_layer=functools.partial(nn.GroupNorm, 2))
        assert isinstance(block.activation, nn.GELU)
        assert isinstance(block.bn1, nn.GroupNorm)
        assert block(torch.randn(2, 16, 8, 8)).shape == (2, 16, 8, 8)

    def test_deterministic_in_eval(self):
        block = ResidualBlock(16, 16).eval()
        x = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            assert torch.equal(block(x), block(x))

    def test_identity_skip_when_dimensions_match(self):
        block = ResidualBlock(16, 16)
        assert isinstance(block.downsample, nn.Identity)

    def test_dropout_disabled_is_identity(self):
        assert isinstance(ResidualBlock(16, 16).dropout, nn.Identity)
        assert isinstance(ResidualBlock(16, 16, dropout_p=0.1).dropout, nn.Dropout2d)


class TestBottleneck:
    @pytest.mark.parametrize("in_c,out_c,stride,size,expansion", [
        (64, 64, 1, 32, 4),
        (64, 128, 2, 32, 4),
        (3, 64, 2, 32, 4),
    ])
    def test_shapes(self, in_c, out_c, stride, size, expansion):
        block = Bottleneck(in_c, out_c, stride=stride, expansion=expansion)
        x = torch.randn(2, in_c, size, size)
        out = block(x)
        expected = size // stride
        assert out.shape == (2, out_c * expansion, expected, expected)

    def test_matches_reference(self):
        torch.manual_seed(0)
        block = Bottleneck(64, 64).eval()
        x = torch.randn(2, 64, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_bottleneck(block, x), atol=1e-6)

    def test_downsample_identity_matches_reference(self):
        torch.manual_seed(0)
        block = Bottleneck(64, 128, stride=2).eval()
        x = torch.randn(2, 64, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_bottleneck(block, x), atol=1e-6)

    def test_identity_skip_when_dimensions_match(self):
        assert isinstance(Bottleneck(256, 64).downsample, nn.Identity)

    def test_custom_norm(self):
        block = Bottleneck(64, 64, norm_layer=functools.partial(nn.GroupNorm, 2))
        assert isinstance(block.bn1, nn.GroupNorm)
        assert block(torch.randn(2, 64, 8, 8)).shape == (2, 256, 8, 8)


class TestDepthwiseSeparableConv:
    @pytest.mark.parametrize("in_c,out_c,stride,size", [
        (16, 16, 1, 32),
        (16, 32, 1, 32),
        (16, 32, 2, 32),
        (3, 64, 2, 32),
    ])
    def test_shapes(self, in_c, out_c, stride, size):
        block = DepthwiseSeparableConv(in_c, out_c, stride=stride)
        x = torch.randn(2, in_c, size, size)
        out = block(x)
        expected = size // stride
        assert out.shape == (2, out_c, expected, expected)

    def test_matches_reference(self):
        torch.manual_seed(0)
        block = DepthwiseSeparableConv(16, 16).eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_depthwise_sep(block, x), atol=1e-6)

    def test_depthwise_is_grouped_and_pointwise_is_1x1(self):
        block = DepthwiseSeparableConv(16, 32)
        assert block.conv1.groups == 16
        assert block.conv1.kernel_size == (3, 3)
        assert block.conv1.out_channels == 16
        assert block.conv2.kernel_size == (1, 1)
        assert block.conv2.out_channels == 32

    def test_skip_only_when_stride_one_and_same_channels(self):
        assert DepthwiseSeparableConv(16, 16, use_skip=True).use_skip
        assert not DepthwiseSeparableConv(16, 32, use_skip=True).use_skip
        assert not DepthwiseSeparableConv(16, 16, stride=2, use_skip=True).use_skip

    def test_skip_off_by_default(self):
        assert not DepthwiseSeparableConv(16, 16).use_skip

    def test_no_skip_is_plain_stack(self):
        torch.manual_seed(0)
        block = DepthwiseSeparableConv(16, 32, use_skip=False).eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_depthwise_sep(block, x), atol=1e-6)

    def test_custom_activation_and_norm(self):
        block = DepthwiseSeparableConv(16, 16, activation=nn.GELU, norm_layer=functools.partial(nn.GroupNorm, 2))
        assert isinstance(block.activation, nn.GELU)
        assert isinstance(block.bn1, nn.GroupNorm)
        assert block(torch.randn(2, 16, 8, 8)).shape == (2, 16, 8, 8)

    def test_deterministic_in_eval(self):
        block = DepthwiseSeparableConv(16, 16).eval()
        x = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            assert torch.equal(block(x), block(x))

    def test_dropout_disabled_is_identity(self):
        assert isinstance(DepthwiseSeparableConv(16, 16).dropout, nn.Identity)
        assert isinstance(DepthwiseSeparableConv(16, 16, dropout_p=0.1).dropout, nn.Dropout2d)


class TestInvertedResidual:
    @pytest.mark.parametrize("in_c,out_c,stride,size,expansion", [
        (16, 16, 1, 32, 6),
        (16, 32, 2, 32, 6),
        (24, 32, 2, 32, 6),
        (64, 96, 1, 32, 6),
        (3, 16, 1, 32, 1),
    ])
    def test_shapes(self, in_c, out_c, stride, size, expansion):
        block = InvertedResidual(in_c, out_c, stride=stride, expansion=expansion)
        x = torch.randn(2, in_c, size, size)
        out = block(x)
        expected = size // stride
        assert out.shape == (2, out_c, expected, expected)

    def test_matches_reference(self):
        torch.manual_seed(0)
        block = InvertedResidual(16, 16).eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_inverted_residual(block, x), atol=1e-6)

    def test_linear_bottleneck_no_final_activation(self):
        torch.manual_seed(0)
        block = InvertedResidual(16, 16, expansion=6).eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            out = block.conv3(block.activation(block.bn2(block.conv2(block.activation(block.bn1(block.conv1(x)))))))
            out = block.dropout(block.bn3(out))
            if block.use_skip:
                out = out + x
        assert torch.allclose(block(x), out, atol=1e-6)

    def test_expansion_one_skips_expand_layer(self):
        block = InvertedResidual(16, 16, expansion=1)
        assert not block.use_expand
        assert isinstance(block.conv1, nn.Identity)
        assert isinstance(block.bn1, nn.Identity)
        assert block.conv2.groups == 16

    def test_expansion_scales_hidden_channels(self):
        block = InvertedResidual(16, 24, expansion=6)
        assert block.use_expand
        assert block.conv1.out_channels == 96
        assert block.conv2.in_channels == 96
        assert block.conv2.groups == 96
        assert block.conv3.out_channels == 24

    def test_skip_only_when_stride_one_and_same_channels(self):
        assert InvertedResidual(16, 16).use_skip
        assert not InvertedResidual(16, 32).use_skip
        assert not InvertedResidual(16, 16, stride=2).use_skip

    def test_no_skip_is_plain_stack(self):
        torch.manual_seed(0)
        block = InvertedResidual(16, 32, use_skip=False).eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            assert torch.allclose(block(x), reference_inverted_residual(block, x), atol=1e-6)

    def test_invalid_stride(self):
        with pytest.raises(ValueError):
            InvertedResidual(16, 16, stride=3)

    def test_custom_activation_and_norm(self):
        block = InvertedResidual(16, 16, activation=nn.GELU, norm_layer=functools.partial(nn.GroupNorm, 2))
        assert isinstance(block.activation, nn.GELU)
        assert isinstance(block.bn1, nn.GroupNorm)
        assert block(torch.randn(2, 16, 8, 8)).shape == (2, 16, 8, 8)

    def test_deterministic_in_eval(self):
        block = InvertedResidual(16, 16).eval()
        x = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            assert torch.equal(block(x), block(x))

    def test_dropout_disabled_is_identity(self):
        assert isinstance(InvertedResidual(16, 16).dropout, nn.Identity)
        assert isinstance(InvertedResidual(16, 16, dropout_p=0.1).dropout, nn.Dropout2d)
