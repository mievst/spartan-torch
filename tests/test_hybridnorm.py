import pytest
import torch
from torch import nn

from spartan_torch import MultiHeadAttention, QKVNorm, RMSNorm, TransformerBlock


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def reference_hybrid_block(block, x):
    h, _ = block.attn(x, x, x)
    h = block.adapt_residual(x) + h
    h2 = block.norm2(h)
    return h2 + block.ff(h2)


def reference_post_block(block, x):
    h, _ = block.attn(x, x, x)
    h = block.norm1(block.adapt_residual(x) + h)
    return block.norm2(h + block.ff(h))


class TestPostNorm:
    def test_shapes(self):
        block = TransformerBlock(64, 16, 4, 64, 256, norm_mode="post")
        assert block(torch.randn(2, 10, 64))[0].shape == (2, 10, 64)

    def test_matches_reference(self):
        block = TransformerBlock(64, 16, 4, 64, 256, norm_mode="post").eval()
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            assert close(block(x)[0], reference_post_block(block, x), tol=1e-6)

    def test_differs_from_pre_norm(self):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        pre = TransformerBlock(64, 16, 4, 64, 256, norm_mode="pre").eval()
        post = TransformerBlock(64, 16, 4, 64, 256, norm_mode="post").eval()
        post.load_state_dict(pre.state_dict())
        with torch.no_grad():
            assert not close(pre(x)[0], post(x)[0])

    def test_gradient_flows(self):
        block = TransformerBlock(64, 16, 4, 64, 256, norm_mode="post")
        x = torch.randn(2, 10, 64, requires_grad=True)
        block(x)[0].sum().backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in block.parameters())

    def test_norms_are_real_not_identity(self):
        block = TransformerBlock(64, 16, 4, 64, 256, norm_mode="post")
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)


class TestRMSNorm:
    def test_shape_preserved(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        assert norm(x).shape == x.shape

    def test_normalizes_squares(self):
        norm = RMSNorm(64)
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64) * 3
        with torch.no_grad():
            y = norm(x)
        assert torch.allclose(y.pow(2).mean(-1), torch.ones_like(y.pow(2).mean(-1)), atol=1e-2)

    def test_weight_scales_output(self):
        norm = RMSNorm(64)
        with torch.no_grad():
            norm.weight.copy_(torch.full((64,), 2.0))
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            y = norm(x)
        assert torch.allclose(y.pow(2).mean(-1), torch.ones_like(y.pow(2).mean(-1)) * 4, atol=1e-2)

    def test_bias_optional(self):
        assert RMSNorm(64).bias is None
        assert RMSNorm(64, bias=True).bias is not None
        assert RMSNorm(64, bias=True)(torch.randn(2, 10, 64)).shape == (2, 10, 64)

    def test_tuple_normalized_shape(self):
        norm = RMSNorm((2, 4))
        x = torch.randn(3, 2, 4)
        assert norm(x).shape == (3, 2, 4)
        assert norm.weight.shape == (2, 4)

    def test_matches_functional_rms(self):
        norm = RMSNorm(64).eval()
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            y = norm(x)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
        assert close(y, ref)

    def test_gradient_flows(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None


class TestQKVNorm:
    def test_normalizes_each_of_qkv(self):
        torch.manual_seed(0)
        hs, nh, b, t = 8, 4, 2, 5
        q = torch.randn(b, nh, t, hs) * 4
        k = torch.randn(b, nh, t, hs) * 0.1
        v = torch.randn(b, nh, t, hs) * 9
        qkv = QKVNorm(hs, nn.LayerNorm)
        nq, nk, nv = qkv(q, k, v)
        layer = nn.LayerNorm(hs)
        for n, ref in ((nq, layer(q)), (nk, layer(k)), (nv, layer(v))):
            assert close(n, ref)
        assert nq.shape == q.shape and nk.shape == k.shape and nv.shape == v.shape

    def test_shared_affine_across_heads(self):
        qkv = QKVNorm(8)
        assert qkv.norm_q.weight.shape == (8,)
        assert qkv.norm_q.bias.shape == (8,)

    def test_v_norm_false_leaves_v(self):
        torch.manual_seed(0)
        qkv = QKVNorm(8, v_norm=False)
        assert isinstance(qkv.norm_v, nn.Identity)
        q = torch.randn(2, 4, 5, 8)
        k = torch.randn(2, 4, 5, 8)
        v = torch.randn(2, 4, 5, 8)
        nq, nk, nv = qkv(q, k, v)
        assert nv is v
        assert not close(nq, q) and not close(nk, k)

    def test_rms_norm_layer(self):
        torch.manual_seed(0)
        qkv = QKVNorm(8, RMSNorm)
        q = torch.randn(2, 4, 5, 8)
        nq, _, _ = qkv(q, q, q)
        ref = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5)
        assert close(nq, ref)

    def test_state_dict_round_trip(self):
        qkv = QKVNorm(8)
        keys = set(qkv.state_dict().keys())
        assert keys == {"norm_q.weight", "norm_q.bias", "norm_k.weight", "norm_k.bias",
                        "norm_v.weight", "norm_v.bias"}
        qkv2 = QKVNorm(8)
        qkv2.load_state_dict(qkv.state_dict())


class TestMHAQKVNorm:
    def test_state_dict_contains_qkv_norm(self):
        mha = MultiHeadAttention(16, 8, 4, 16, qkv_norm=True)
        keys = set(mha.state_dict().keys())
        assert {"qkv_norm.norm_q.weight", "qkv_norm.norm_k.weight", "qkv_norm.norm_v.weight"} <= keys
        mha2 = MultiHeadAttention(16, 8, 4, 16, qkv_norm=True)
        mha2.load_state_dict(mha.state_dict())

    def test_no_qkv_norm_by_default(self):
        mha = MultiHeadAttention(16, 8, 4, 16)
        assert mha.qkv_norm is None
        assert not any(k.startswith("qkv_norm.") for k in mha.state_dict())

    def test_applies_and_differs_from_plain(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16)
        plain = MultiHeadAttention(16, 8, 4, 16).eval()
        normed = MultiHeadAttention(16, 8, 4, 16, qkv_norm=True).eval()
        normed.load_state_dict(plain.state_dict(), strict=False)
        with torch.no_grad():
            assert not close(plain(x, x, x)[0], normed(x, x, x)[0])

    def test_shapes_gqa_and_sdpa(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16)
        cases = [({}, 4), ({"num_kv_heads": 2}, 2), ({"use_sdpa": True}, 4)]
        for kwargs, kv_heads in cases:
            mha = MultiHeadAttention(16, 8, 4, 16, qkv_norm=True, **kwargs).eval()
            out, cache = mha(x, x, x)
            assert out.shape == (2, 5, 16)
            assert cache[0].shape == (2, kv_heads, 5, 8)

    def test_manual_equals_sdpa_with_qkv_norm(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16)
        mha = MultiHeadAttention(16, 8, 4, 16, qkv_norm=True).eval()
        mha_sdpa = MultiHeadAttention(16, 8, 4, 16, qkv_norm=True, use_sdpa=True).eval()
        mha_sdpa.load_state_dict(mha.state_dict())
        with torch.no_grad():
            assert close(mha(x, x, x)[0], mha_sdpa(x, x, x)[0])

    def test_kv_cache_decode_matches_prefill(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(16, 8, 4, 16, qkv_norm=True, is_causal=True).eval()
        seq = torch.randn(2, 6, 16)
        with torch.no_grad():
            ref = mha(seq, seq, seq)[0]
            cache = None
            decoded = []
            for i in range(6):
                out, new_kv = mha(seq[:, i:i+1], seq[:, i:i+1], seq[:, i:i+1], past_key_value=cache)
                cache = (new_kv[0], new_kv[1]) if cache is None else (
                    torch.cat([cache[0], new_kv[0]], dim=-2),
                    torch.cat([cache[1], new_kv[1]], dim=-2),
                )
                decoded.append(out)
            assert close(torch.cat(decoded, 1), ref)


class TestHybridBlock:
    @pytest.fixture
    def block(self):
        return TransformerBlock(64, 16, 4, 64, 256, norm_mode="hybrid").eval()

    def test_shapes(self, block):
        x = torch.randn(2, 10, 64)
        assert block(x)[0].shape == (2, 10, 64)

    def test_matches_reference(self, block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            assert close(block(x)[0], reference_hybrid_block(block, x))

    def test_no_input_norm_no_norm1_params(self, block):
        assert isinstance(block.norm1, nn.Identity)
        assert not any(k.startswith("norm1.") for k in block.state_dict())

    def test_attention_has_qkv_norm(self, block):
        assert block.attn.qkv_norm is not None
        assert "attn.qkv_norm.norm_q.weight" in block.state_dict()

    def test_differs_from_pre_norm(self):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        hybrid = TransformerBlock(64, 16, 4, 64, 256, norm_mode="hybrid").eval()
        pre = TransformerBlock(64, 16, 4, 64, 256).eval()
        pre.load_state_dict(hybrid.state_dict(), strict=False)
        with torch.no_grad():
            assert not close(hybrid(x)[0], pre(x)[0])

    def test_hybrid_discards_norm1_grad_flow(self, block):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64, requires_grad=True)
        block(x)[0].sum().backward()
        assert x.grad is not None
        assert block.norm2.weight.grad is not None
        assert block.attn.qkv_norm.norm_q.weight.grad is not None

    def test_invalid_norm_mode(self):
        with pytest.raises(ValueError, match="norm_mode"):
            TransformerBlock(64, 16, 4, 64, 256, norm_mode="sandwich")

    def test_hybrid_cross_attn_shapes(self):
        block = TransformerBlock(64, 16, 4, 64, 256, norm_mode="hybrid",
                                 with_cross_attn=True, memory_size=128).eval()
        x = torch.randn(2, 10, 64)
        memory = torch.randn(2, 15, 128)
        assert block(x, memory=memory)[0].shape == (2, 10, 64)

    def test_hybrid_rmssnorm_reproduces_paper_formula(self):
        block = TransformerBlock(64, 16, 4, 64, 256, norm_mode="hybrid", norm_layer=RMSNorm).eval()
        assert isinstance(block.attn.qkv_norm.norm_q, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            assert close(block(x)[0], reference_hybrid_block(block, x))
