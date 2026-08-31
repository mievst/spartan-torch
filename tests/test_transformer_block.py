import pytest
import torch
from torch import nn

from spartan_torch import FeedForward, RotaryPositionalEmbedding, TransformerBlock


def reference_block(block, x):
    h = block.norm1(x)
    h, _ = block.attn(h, h, h)
    h = block.adapt_residual(x) + h
    return h + block.ff(block.norm2(h))


def reference_cross_block(block, x, memory):
    h = block.norm1(x)
    h, _ = block.attn(h, h, h)
    h = block.adapt_residual(x) + h
    h = block.cross_attn(h, memory)
    return h + block.ff(block.norm2(h))


@pytest.fixture
def eval_block():
    return TransformerBlock(64, 16, 4, 64, 256).eval()


@pytest.fixture
def eval_cross_block():
    return TransformerBlock(64, 16, 4, 64, 256, with_cross_attn=True, memory_size=128).eval()


class TestTransformerBlock:
    @pytest.mark.parametrize("in_size,head,num_heads,out_size,ff", [
        (64, 16, 4, 64, 256),
        (32, 8, 8, 32, 128),
    ])
    def test_shapes(self, in_size, head, num_heads, out_size, ff):
        block = TransformerBlock(in_size, head, num_heads, out_size, ff)
        x = torch.randn(2, 10, in_size)
        assert block(x)[0].shape == (2, 10, out_size)

    def test_matches_reference(self, eval_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            assert torch.allclose(eval_block(x)[0], reference_block(eval_block, x), atol=1e-6)

    def test_residual_dim_mismatch_uses_linear(self):
        block = TransformerBlock(48, 16, 4, 64, 256)
        assert isinstance(block.adapt_residual, nn.Linear)
        x = torch.randn(2, 10, 48)
        assert block(x)[0].shape == (2, 10, 64)

    def test_residual_identity_when_dims_match(self):
        block = TransformerBlock(64, 16, 4, 64, 256)
        assert isinstance(block.adapt_residual, nn.Identity)

    def test_gqa_shares_kv_heads(self):
        block = TransformerBlock(64, 16, 4, 64, 256, num_kv_heads=2)
        assert block.attn.num_kv_heads == 2
        assert block(torch.randn(2, 10, 64))[0].shape == (2, 10, 64)

    def test_mqa_single_kv_head(self):
        block = TransformerBlock(64, 16, 4, 64, 256, num_kv_heads=1)
        assert block.attn.num_kv_heads == 1
        assert block(torch.randn(2, 10, 64))[0].shape == (2, 10, 64)

    def test_invalid_num_kv_heads_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            TransformerBlock(64, 16, 4, 64, 256, num_kv_heads=3)

    def test_causal_propagates_to_attention(self):
        block = TransformerBlock(64, 16, 4, 64, 256, is_causal=True)
        assert block.attn.is_causal
        block2 = TransformerBlock(64, 16, 4, 64, 256)
        assert not block2.attn.is_causal

    def test_causal_changes_output(self, eval_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        causal = TransformerBlock(64, 16, 4, 64, 256, is_causal=True).eval()
        with torch.no_grad():
            assert not torch.allclose(eval_block(x)[0], causal(x)[0], atol=1e-6)

    def test_mask_changes_output(self, eval_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(1, 1, 10, 10, dtype=torch.bool)
        mask[:, :, :, 9] = True
        with torch.no_grad():
            assert not torch.allclose(eval_block(x)[0], eval_block(x, mask=mask)[0], atol=1e-6)

    def test_default_activation_is_gelu(self):
        block = TransformerBlock(64, 16, 4, 64, 256)
        assert isinstance(block.ff, FeedForward)
        assert isinstance(block.ff.activation, nn.GELU)

    def test_custom_activation_and_norm(self):
        calls = []
        def norm_factory(size):
            calls.append(size)
            return nn.LayerNorm(size)
        block = TransformerBlock(64, 16, 4, 64, 256, activation=nn.ReLU, norm_layer=norm_factory)
        assert isinstance(block.ff.activation, nn.ReLU)
        assert calls == [64, 64]
        assert block(torch.randn(2, 10, 64))[0].shape == (2, 10, 64)

    def test_deterministic_in_eval(self, eval_block):
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            assert torch.equal(eval_block(x)[0], eval_block(x)[0])

    def test_dropout_disabled_is_identity(self):
        block = TransformerBlock(64, 16, 4, 64, 256)
        assert isinstance(block.dropout1, nn.Dropout)
        assert block.dropout1.p == 0.0
        block2 = TransformerBlock(64, 16, 4, 64, 256, dropout_p=0.1)
        assert block2.dropout1.p == 0.1

    def test_use_sdpa(self):
        block = TransformerBlock(64, 16, 4, 64, 256, use_sdpa=True)
        assert block(torch.randn(2, 10, 64))[0].shape == (2, 10, 64)

    def test_gradient_flows(self):
        block = TransformerBlock(64, 16, 4, 64, 256)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = block(x)[0].sum()
        out.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in block.parameters())

    def test_memory_ignored_without_cross_attn(self):
        block = TransformerBlock(64, 16, 4, 64, 256)
        memory = torch.randn(2, 20, 64)
        assert block(torch.randn(2, 10, 64), memory=memory)[0].shape == (2, 10, 64)

    def test_ff_layer_replaces_ff(self):
        block = TransformerBlock(64, 16, 4, 64, 256, ff_layer=nn.Identity())
        assert isinstance(block.ff, nn.Identity)
        assert block(torch.randn(2, 10, 64))[0].shape == (2, 10, 64)

    def test_ff_layer_skips_default_activation(self):
        block = TransformerBlock(64, 16, 4, 64, 256, activation=nn.ReLU, ff_layer=nn.Identity())
        assert not hasattr(block, "activation")
        assert isinstance(block.ff, nn.Identity)
        assert block(torch.randn(2, 10, 64))[0].shape == (2, 10, 64)

    def test_ff_layer_shared_across_blocks_ties_weights(self):
        shared = nn.Linear(64, 64)
        block1 = TransformerBlock(64, 16, 4, 64, 256, ff_layer=shared)
        block2 = TransformerBlock(64, 16, 4, 64, 256, ff_layer=shared)
        assert block1.ff is block2.ff is shared

    def test_qk_mod_passed_to_self_attention(self):
        rope = RotaryPositionalEmbedding(16)
        block = TransformerBlock(64, 16, 4, 64, 256, qk_mod=rope)
        assert block.attn.qk_mod is rope
        assert block(torch.randn(2, 10, 64))[0].shape == (2, 10, 64)

    def test_qk_mod_changes_output(self):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        plain = TransformerBlock(64, 16, 4, 64, 256, qk_mod=None).eval()
        rope = TransformerBlock(64, 16, 4, 64, 256, qk_mod=RotaryPositionalEmbedding(16)).eval()
        rope.load_state_dict(plain.state_dict())
        with torch.no_grad():
            assert not torch.allclose(plain(x)[0], rope(x)[0], atol=1e-6)


class TestTransformerBlockCrossAttn:
    @pytest.mark.parametrize("in_size,head,num_heads,out_size,ff,mem", [
        (64, 16, 4, 64, 256, 128),
        (32, 8, 8, 32, 128, 64),
    ])
    def test_shapes(self, in_size, head, num_heads, out_size, ff, mem):
        block = TransformerBlock(in_size, head, num_heads, out_size, ff, with_cross_attn=True, memory_size=mem)
        x = torch.randn(2, 10, in_size)
        memory = torch.randn(2, 20, mem)
        assert block(x, memory)[0].shape == (2, 10, out_size)

    def test_memory_size_defaults_to_out_size(self):
        block = TransformerBlock(64, 16, 4, 64, 256, with_cross_attn=True)
        assert block.cross_attn.norm_mv.normalized_shape == (64,)

    def test_matches_reference(self, eval_cross_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        with torch.no_grad():
            assert torch.allclose(
                eval_cross_block(x, memory)[0],
                reference_cross_block(eval_cross_block, x, memory),
                atol=1e-6,
            )

    def test_memory_required_when_cross_attn(self):
        block = TransformerBlock(64, 16, 4, 64, 256, with_cross_attn=True)
        with pytest.raises(ValueError, match="memory is required"):
            block(torch.randn(2, 10, 64))[0]

    def test_causal_changes_output(self, eval_cross_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        non_causal = TransformerBlock(
            64, 16, 4, 64, 256, with_cross_attn=True, memory_size=128, is_causal=False
        ).eval()
        with torch.no_grad():
            assert not torch.allclose(eval_cross_block(x, memory)[0], non_causal(x, memory)[0], atol=1e-6)

    def test_mask_changes_output(self, eval_cross_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        mask = torch.zeros(1, 1, 10, 10, dtype=torch.bool)
        mask[:, :, :, 9] = True
        with torch.no_grad():
            assert not torch.allclose(
                eval_cross_block(x, memory)[0],
                eval_cross_block(x, memory, mask=mask)[0],
                atol=1e-6,
            )

    def test_memory_mask_changes_output(self, eval_cross_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        mem_mask = torch.zeros(1, 1, 10, 20, dtype=torch.bool)
        mem_mask[:, :, :, 19] = True
        with torch.no_grad():
            assert not torch.allclose(
                eval_cross_block(x, memory)[0],
                eval_cross_block(x, memory, memory_mask=mem_mask)[0],
                atol=1e-6,
            )

    def test_residual_linear_when_in_differs_from_out(self):
        block = TransformerBlock(48, 16, 4, 64, 256, with_cross_attn=True, memory_size=128)
        assert isinstance(block.adapt_residual, nn.Linear)
        x = torch.randn(2, 10, 48)
        memory = torch.randn(2, 20, 128)
        assert block(x, memory)[0].shape == (2, 10, 64)

    def test_gqa_shares_kv_heads(self):
        block = TransformerBlock(64, 16, 4, 64, 256, with_cross_attn=True, memory_size=128, num_kv_heads=2)
        assert block.attn.num_kv_heads == 2
        assert block.cross_attn.attn.num_kv_heads == 2

    def test_invalid_num_kv_heads_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            TransformerBlock(64, 16, 4, 64, 256, with_cross_attn=True, num_kv_heads=3)

    def test_cross_attn_layer_requires_with_cross_attn(self):
        with pytest.raises(ValueError, match="with_cross_attn"):
            TransformerBlock(64, 16, 4, 64, 256, cross_attn_layer=nn.Identity())

    def test_custom_activation(self):
        block = TransformerBlock(64, 16, 4, 64, 256, with_cross_attn=True, activation=nn.ReLU)
        assert isinstance(block.ff.activation, nn.ReLU)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 64)
        assert block(x, memory)[0].shape == (2, 10, 64)

    def test_deterministic_in_eval(self, eval_cross_block):
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        with torch.no_grad():
            assert torch.equal(eval_cross_block(x, memory)[0], eval_cross_block(x, memory)[0])

    def test_use_sdpa(self):
        block = TransformerBlock(64, 16, 4, 64, 256, with_cross_attn=True, memory_size=128, use_sdpa=True)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        assert block(x, memory)[0].shape == (2, 10, 64)

    def test_ff_layer_replaces_ff(self):
        block = TransformerBlock(
            64, 16, 4, 64, 256,
            with_cross_attn=True, memory_size=128,
            ff_layer=nn.Identity(),
        )
        assert isinstance(block.ff, nn.Identity)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        assert block(x, memory)[0].shape == (2, 10, 64)

    def test_gradient_flows(self):
        block = TransformerBlock(64, 16, 4, 64, 256, with_cross_attn=True, memory_size=128)
        x = torch.randn(2, 10, 64, requires_grad=True)
        memory = torch.randn(2, 20, 128, requires_grad=True)
        out = block(x, memory)[0].sum()
        out.backward()
        assert x.grad is not None
        assert memory.grad is not None
        assert all(p.grad is not None for p in block.parameters())


class TestFeedForward:
    def test_shapes(self):
        ff = FeedForward(64, 256)
        assert ff(torch.randn(2, 10, 64)).shape == (2, 10, 64)

    def test_default_activation_is_gelu(self):
        assert isinstance(FeedForward(64, 256).activation, nn.GELU)

    def test_custom_activation(self):
        assert isinstance(FeedForward(64, 256, nn.ReLU).activation, nn.ReLU)

    def test_rejects_inplace_on_activation_without_support(self):
        class NoInplace(nn.ReLU):
            def __init__(self, inplace=False):
                super().__init__(inplace)

        ff = FeedForward(64, 256, NoInplace)
        assert isinstance(ff.activation, NoInplace)
        assert ff(torch.randn(2, 10, 64)).shape == (2, 10, 64)
