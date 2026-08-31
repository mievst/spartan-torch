import pytest
import torch
from torch import nn

from spartan_torch import CrossAttentionBlock, LinformerAttention, LinformerSeqProjection


def reference_cross_attn(block, x, memory):
    h, _ = block.attn(block.norm_q(x), block.norm_mv(memory), block.norm_mv(memory))
    return block.adapt_residual(x) + h


@pytest.fixture
def eval_block():
    return CrossAttentionBlock(64, 16, 4, 64, memory_size=128).eval()


class TestCrossAttentionBlock:
    def test_shapes(self):
        block = CrossAttentionBlock(64, 16, 4, 64, memory_size=128)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        assert block(x, memory).shape == (2, 10, 64)

    def test_memory_size_defaults_to_query_size(self):
        block = CrossAttentionBlock(64, 16, 4, 64)
        assert block.norm_mv.normalized_shape == (64,)

    def test_matches_reference(self, eval_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        with torch.no_grad():
            assert torch.allclose(eval_block(x, memory), reference_cross_attn(eval_block, x, memory), atol=1e-6)

    def test_output_embeds_memory(self, eval_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        out = eval_block(x, memory)
        assert out.shape == (2, 10, 64)
        assert torch.isfinite(out).all()

    def test_residual_identity_when_dims_match(self):
        block = CrossAttentionBlock(64, 16, 4, 64, memory_size=128)
        assert isinstance(block.adapt_residual, nn.Identity)

    def test_residual_linear_when_query_differs_from_out(self):
        block = CrossAttentionBlock(48, 16, 4, 64, memory_size=128)
        assert isinstance(block.adapt_residual, nn.Linear)
        x = torch.randn(2, 10, 48)
        memory = torch.randn(2, 20, 128)
        assert block(x, memory).shape == (2, 10, 64)

    def test_gqa_shares_kv_heads(self):
        block = CrossAttentionBlock(64, 16, 4, 64, memory_size=128, num_kv_heads=2)
        assert block.attn.num_kv_heads == 2

    def test_invalid_num_kv_heads_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            CrossAttentionBlock(64, 16, 4, 64, memory_size=128, num_kv_heads=3)

    def test_memory_mask_changes_output(self, eval_block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        mask = torch.zeros(1, 1, 10, 20, dtype=torch.bool)
        mask[:, :, :, 19] = True
        with torch.no_grad():
            assert not torch.allclose(
                eval_block(x, memory),
                eval_block(x, memory, memory_mask=mask),
                atol=1e-6,
            )

    def test_deterministic_in_eval(self, eval_block):
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        with torch.no_grad():
            assert torch.equal(eval_block(x, memory), eval_block(x, memory))

    def test_use_sdpa(self):
        block = CrossAttentionBlock(64, 16, 4, 64, memory_size=128, use_sdpa=True)
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 20, 128)
        assert block(x, memory).shape == (2, 10, 64)

    def test_gradient_flows(self):
        block = CrossAttentionBlock(64, 16, 4, 64, memory_size=128)
        x = torch.randn(2, 10, 64, requires_grad=True)
        memory = torch.randn(2, 20, 128, requires_grad=True)
        out = block(x, memory).sum()
        out.backward()
        assert x.grad is not None
        assert memory.grad is not None
        assert all(p.grad is not None for p in block.parameters())


B, Q, ML, HS, NH, PROJ_K, MAX_SEQ = 2, 10, 20, 16, 4, 3, 20


def project_manual(x, proj):
    """Reference seq-dim contraction matching LinformerSeqProjection._contract."""
    if proj.dim() == 3:
        return torch.einsum("bhld,hlk->bhkd", x, proj)
    return torch.einsum("bhld,lk->bhkd", x, proj)


def make_cross_linformer(**kwargs):
    kw = dict(query_in_size=64, proj_k=PROJ_K, max_seq_len=MAX_SEQ)
    kw.update(kwargs)
    return LinformerAttention(128, HS, NH, 64, **kw)


class TestCrossAttentionWithLinformer:
    def test_attn_layer_replaces_default(self):
        attn = make_cross_linformer()
        block = CrossAttentionBlock(64, HS, NH, 64, memory_size=128, attn_layer=attn)
        assert block.attn is attn

    def test_shapes(self):
        block = CrossAttentionBlock(64, HS, NH, 64, memory_size=128, attn_layer=make_cross_linformer())
        x = torch.randn(B, Q, 64)
        memory = torch.randn(B, ML, 128)
        assert block(x, memory).shape == (B, Q, 64)

    def test_query_in_size_independent_of_memory_size(self):
        block = CrossAttentionBlock(48, HS, NH, 64, memory_size=128, attn_layer=make_cross_linformer(query_in_size=48))
        x = torch.randn(B, Q, 48)
        memory = torch.randn(B, ML, 128)
        assert block(x, memory).shape == (B, Q, 64)

    def test_matches_manual_reference(self):
        attn = make_cross_linformer().eval()
        block = CrossAttentionBlock(64, HS, NH, 64, memory_size=128, attn_layer=attn).eval()
        torch.manual_seed(0)
        x = torch.randn(B, Q, 64)
        memory = torch.randn(B, ML, 128)
        with torch.no_grad():
            h = block.norm_q(x)
            m = block.norm_mv(memory)
            qq = attn.query_matrix(h).view(B, Q, NH, HS).transpose(1, 2)
            kk = attn.key_matrix(m).view(B, ML, NH, HS).transpose(1, 2)
            vv = attn.value_matrix(m).view(B, ML, NH, HS).transpose(1, 2)
            kp = project_manual(kk, attn.proj.E[:ML])
            vp = project_manual(vv, attn.proj.E[:ML])
            w = torch.softmax(qq @ kp.transpose(-2, -1) / HS**0.5, dim=-1)
            attn_out = attn.out((w @ vp).transpose(1, 2).reshape(B, Q, -1))
            ref = block.adapt_residual(x) + attn_out
            assert torch.allclose(block(x, memory), ref, atol=1e-5, rtol=1e-5)

    def test_memory_mask_raises(self):
        block = CrossAttentionBlock(64, HS, NH, 64, memory_size=128, attn_layer=make_cross_linformer())
        x = torch.randn(B, Q, 64)
        memory = torch.randn(B, ML, 128)
        mask = torch.zeros(1, 1, Q, ML, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            block(x, memory, memory_mask=mask)

    def test_shared_projection_ties_weights(self):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")
        block1 = CrossAttentionBlock(64, HS, NH, 64, memory_size=128, attn_layer=make_cross_linformer(projection=proj))
        block2 = CrossAttentionBlock(64, HS, NH, 64, memory_size=128, attn_layer=make_cross_linformer(projection=proj))
        assert block1.attn.proj is block2.attn.proj is proj

    def test_shared_projection_registers_once_in_state_dict(self):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")

        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = proj
                self.b1 = CrossAttentionBlock(64, HS, NH, 64, memory_size=128, attn_layer=make_cross_linformer(projection=proj))
                self.b2 = CrossAttentionBlock(64, HS, NH, 64, memory_size=128, attn_layer=make_cross_linformer(projection=proj))

            def forward(self, x, memory):
                return self.b2(self.b1(x, memory), memory)

        m = Shared()
        assert [k for k in m.state_dict() if "proj" in k] == ["proj.E"]
        x = torch.randn(B, Q, 64)
        memory = torch.randn(B, ML, 128)
        assert m(x, memory).shape == (B, Q, 64)
