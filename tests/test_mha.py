import pytest
import torch

from spartan_torch import MultiHeadAttention, RotaryPositionalEmbedding

B, Q, K, IN, OUT, HS, NH = 2, 5, 7, 16, 20, 8, 4

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture()
def tensors():
    torch.manual_seed(0)
    q = torch.randn(B, Q, IN)
    kv = torch.randn(B, K, IN)
    return q, kv


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


class TestMultiHeadAttention:
    def test_shapes(self, tensors):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT)
        assert mha(q, kv, kv)[0].shape == (B, Q, OUT)

    def test_manual_equals_sdpa(self, tensors):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT).eval()
        mha_sdpa = MultiHeadAttention(IN, HS, NH, OUT, use_sdpa=True).eval()
        mha_sdpa.load_state_dict(mha.state_dict())
        with torch.no_grad():
            assert close(mha(q, kv, kv)[0], mha_sdpa(q, kv, kv)[0])

    def test_cross_attention_query_in_size(self, tensors):
        _, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, query_in_size=24)
        assert mha(torch.randn(B, Q, 24), kv, kv)[0].shape == (B, Q, OUT)

    def test_causal_blocks_future(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(IN, HS, NH, OUT, is_causal=True).eval()
        kv = torch.randn(B, K, IN)
        with torch.no_grad():
            out = mha(kv, kv, kv)[0]
            kv_future = kv.clone()
            kv_future[:, 1:] = 0.0
            assert close(out[:, 0], mha(kv, kv_future, kv_future)[0][:, 0])
            kv_past = kv.clone()
            kv_past[:, 0] = 0.0
            assert not close(out[:, 1],
                             mha(kv, kv_past, kv_past)[0][:, 1])
    def test_mask_bool_equals_float_equals_is_causal(self, tensors):
        q, kv = tensors
        mask_bool = torch.triu(torch.ones(Q, K, dtype=torch.bool), diagonal=1)
        mask_float = torch.triu(torch.full((Q, K), float("-inf")), diagonal=1)
        mha = MultiHeadAttention(IN, HS, NH, OUT).eval()
        mha_causal = MultiHeadAttention(IN, HS, NH, OUT, is_causal=True)
        mha_causal.load_state_dict(mha.state_dict())
        mha_causal.eval()
        with torch.no_grad():
            o_b = mha(q, kv, kv, mask=mask_bool)[0]
            assert close(o_b, mha(q, kv, kv, mask=mask_float)[0])
            assert close(o_b, mha_causal(q, kv, kv)[0])

    @pytest.mark.parametrize(
        "mask",
        [
            torch.triu(torch.ones(Q, K, dtype=torch.bool), diagonal=1),  # (Q, K)
            torch.triu(torch.ones(1, Q, K, dtype=torch.bool), diagonal=1),  # (1, Q, K)
            torch.triu(torch.ones(B, 1, Q, K, dtype=torch.bool), diagonal=1),  # (B, 1, Q, K)
        ],
    )
    def test_mask_broadcast_shapes(self, tensors, mask):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT).eval()
        with torch.no_grad():
            assert mha(q, kv, kv, mask=mask)[0].shape == (B, Q, OUT)

    def test_fp16_fully_masked_row_no_nan(self, tensors):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT).eval().half()
        full_mask = torch.zeros(1, Q, K, dtype=torch.bool)
        full_mask[:, 0] = True
        with torch.no_grad():
            out = mha(q.half(), kv.half(), kv.half(), mask=full_mask)[0]
            assert not torch.isnan(out).any()

    def test_dropout_train_vs_eval(self, tensors):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, attn_p=0.3)
        mha.train()
        assert not close(mha(q, kv, kv)[0], mha(q, kv, kv)[0], 1e-4)
        mha.eval()
        assert close(mha(q, kv, kv)[0], mha(q, kv, kv)[0], 0.0)

    def test_state_dict_roundtrip(self, tensors):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2, use_sdpa=True)
        state = mha.state_dict()
        clone = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2, use_sdpa=True)
        clone.load_state_dict(state)
        assert clone(q, kv, kv)[0].shape == (B, Q, OUT)


class TestGroupedQueryAttention:
    @pytest.mark.parametrize("num_kv_heads", [1, 2, NH])
    def test_shapes(self, tensors, num_kv_heads):
        q, kv = tensors
        gqa = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=num_kv_heads)
        assert gqa(q, kv, kv)[0].shape == (B, Q, OUT)

    def test_num_kv_heads_equals_num_heads_is_mha(self, tensors):
        q, kv = tensors
        gqa = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=NH)
        mha = MultiHeadAttention(IN, HS, NH, OUT)
        gqa.load_state_dict(mha.state_dict())
        gqa.eval()
        mha.eval()
        with torch.no_grad():
            assert close(gqa(q, kv, kv)[0], mha(q, kv, kv)[0])

    def test_matches_manual_repeat_reference(self, tensors):
        q, kv = tensors
        gqa = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2).eval()
        with torch.no_grad():
            qq = gqa.query_matrix(q).view(B, Q, NH, HS).transpose(1, 2)
            kk = gqa.key_matrix(kv).view(B, K, 2, HS).transpose(1, 2)
            vv = gqa.value_matrix(kv).view(B, K, 2, HS).transpose(1, 2)
            kk = kk.repeat_interleave(NH // 2, dim=1)
            vv = vv.repeat_interleave(NH // 2, dim=1)
            w = torch.softmax(qq @ kk.transpose(-2, -1) / HS**0.5, dim=-1)
            ref = gqa.out((w @ vv).transpose(1, 2).reshape(B, Q, -1))
            assert close(gqa(q, kv, kv)[0], ref)

    def test_manual_equals_sdpa(self, tensors):
        q, kv = tensors
        gqa = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2).eval()
        gqa_sdpa = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2, use_sdpa=True).eval()
        gqa_sdpa.load_state_dict(gqa.state_dict())
        with torch.no_grad():
            assert close(gqa(q, kv, kv)[0], gqa_sdpa(q, kv, kv)[0])

    def test_divisibility_guard(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=3)


class TestGradients:
    @pytest.mark.parametrize("num_kv_heads", [None, 2])
    @pytest.mark.parametrize("use_sdpa", [False, True])
    def test_backward(self, tensors, num_kv_heads, use_sdpa):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=num_kv_heads, use_sdpa=use_sdpa)
        qq = q.clone().requires_grad_(True)
        loss = mha(qq, kv, kv)[0].square().mean()
        loss.backward()
        assert qq.grad is not None
        assert all(p.grad is not None for p in mha.parameters())

    @CUDA
    @pytest.mark.parametrize("num_kv_heads", [None, 2, 1])
    @pytest.mark.parametrize("use_sdpa", [False, True])
    def test_backward_cuda(self, num_kv_heads, use_sdpa):
        mha = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=num_kv_heads, use_sdpa=use_sdpa).cuda()
        q = torch.randn(B, Q, IN, device="cuda", requires_grad=True)
        kv = torch.randn(B, K, IN, device="cuda")
        mha(q, kv, kv)[0].square().mean().backward()
        assert q.grad is not None

    @CUDA
    def test_manual_equals_sdpa_cuda(self):
        mha = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2).cuda().eval()
        mha_sdpa = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2, use_sdpa=True).cuda().eval()
        mha_sdpa.load_state_dict(mha.state_dict())
        q = torch.randn(B, Q, IN, device="cuda")
        kv = torch.randn(B, K, IN, device="cuda")
        with torch.no_grad():
            assert close(mha(q, kv, kv)[0], mha_sdpa(q, kv, kv)[0], 1e-4)


class TestQKMod:
    @pytest.fixture
    def rope(self):
        return RotaryPositionalEmbedding(HS)

    def test_identity_noop_matches_plain(self, tensors):
        q, kv = tensors
        plain = MultiHeadAttention(IN, HS, NH, OUT).eval()
        ident = MultiHeadAttention(IN, HS, NH, OUT, qk_mod=lambda q, k, q_pos, k_pos: (q, k)).eval()
        ident.load_state_dict(plain.state_dict())
        with torch.no_grad():
            assert close(plain(q, kv, kv)[0], ident(q, kv, kv)[0], 0.0)

    def test_matches_manual_reference(self, tensors, rope):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, qk_mod=rope).eval()
        with torch.no_grad():
            out = mha(q, kv, kv)[0]
            qq = mha.query_matrix(q).view(B, Q, NH, HS).transpose(1, 2)
            kk = mha.key_matrix(kv).view(B, K, NH, HS).transpose(1, 2)
            vv = mha.value_matrix(kv).view(B, K, NH, HS).transpose(1, 2)
            qq, kk = rope(qq, kk)
            w = torch.softmax(qq @ kk.transpose(-2, -1) / HS**0.5, dim=-1)
            ref = mha.out((w @ vv).transpose(1, 2).reshape(B, Q, -1))
            assert close(out, ref)

    def test_manual_equals_sdpa(self, tensors, rope):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, qk_mod=rope).eval()
        mha_sdpa = MultiHeadAttention(IN, HS, NH, OUT, qk_mod=rope, use_sdpa=True).eval()
        mha_sdpa.load_state_dict(mha.state_dict())
        with torch.no_grad():
            assert close(mha(q, kv, kv)[0], mha_sdpa(q, kv, kv)[0])

    def test_gqa_rotates_after_repeat(self, tensors, rope):
        q, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2, qk_mod=rope).eval()
        with torch.no_grad():
            out = mha(q, kv, kv)[0]
            qq = mha.query_matrix(q).view(B, Q, NH, HS).transpose(1, 2)
            kk = mha.key_matrix(kv).view(B, K, 2, HS).transpose(1, 2)
            vv = mha.value_matrix(kv).view(B, K, 2, HS).transpose(1, 2)
            kk = kk.repeat_interleave(NH // 2, dim=1)
            vv = vv.repeat_interleave(NH // 2, dim=1)
            qq, kk = rope(qq, kk)
            w = torch.softmax(qq @ kk.transpose(-2, -1) / HS**0.5, dim=-1)
            ref = mha.out((w @ vv).transpose(1, 2).reshape(B, Q, -1))
            assert close(out, ref)

    def test_not_registered_in_state_dict(self, rope):
        mha = MultiHeadAttention(IN, HS, NH, OUT, qk_mod=rope)
        assert "qk_mod" not in mha.state_dict()
        assert mha.qk_mod is rope

    def test_rope_injected_output_differs(self, tensors):
        q, kv = tensors
        plain = MultiHeadAttention(IN, HS, NH, OUT).eval()
        rope_mha = MultiHeadAttention(IN, HS, NH, OUT, qk_mod=RotaryPositionalEmbedding(HS)).eval()
        rope_mha.load_state_dict(plain.state_dict())
        with torch.no_grad():
            assert not close(plain(q, kv, kv)[0], rope_mha(q, kv, kv)[0], 1e-6)


class TestKVCache:
    def test_causal_chunked_equals_prefill(self, tensors):
        _, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, is_causal=True).eval()
        with torch.no_grad():
            full = mha(kv, kv, kv)[0]
            out_first, cache = mha(kv[:, :3], kv[:, :3], kv[:, :3])
            rest = mha(kv[:, 3:], kv[:, 3:], kv[:, 3:], past_key_value=cache)[0]
            assert close(out_first, full[:, :3])
            assert close(rest, full[:, 3:])

    def test_causal_roped_chunked_equals_prefill(self, tensors):
        _, kv = tensors
        rope = RotaryPositionalEmbedding(HS)
        mha = MultiHeadAttention(IN, HS, NH, OUT, is_causal=True, qk_mod=rope).eval()
        with torch.no_grad():
            full = mha(kv, kv, kv)[0]
            out_first, cache = mha(kv[:, :3], kv[:, :3], kv[:, :3])
            rest = mha(kv[:, 3:], kv[:, 3:], kv[:, 3:], past_key_value=cache)[0]
            assert close(out_first, full[:, :3])
            assert close(rest, full[:, 3:])

    def test_incremental_decode_equals_prefill(self, tensors):
        _, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT, is_causal=True).eval()
        with torch.no_grad():
            full = mha(kv, kv, kv)[0]
            cache = None
            decoded = []
            for i in range(K):
                out, new_kv = mha(kv[:, i : i + 1], kv[:, i : i + 1], kv[:, i : i + 1], past_key_value=cache)
                cache = (new_kv[0], new_kv[1]) if cache is None else (
                    torch.cat([cache[0], new_kv[0]], dim=-2),
                    torch.cat([cache[1], new_kv[1]], dim=-2),
                )
                decoded.append(out)
            assert close(torch.cat(decoded, dim=1), full)

    def test_incremental_decode_roped_equals_prefill(self, tensors):
        _, kv = tensors
        rope = RotaryPositionalEmbedding(HS)
        mha = MultiHeadAttention(IN, HS, NH, OUT, is_causal=True, qk_mod=rope).eval()
        with torch.no_grad():
            full = mha(kv, kv, kv)[0]
            cache = None
            decoded = []
            for i in range(K):
                out, new_kv = mha(kv[:, i : i + 1], kv[:, i : i + 1], kv[:, i : i + 1], past_key_value=cache)
                cache = (new_kv[0], new_kv[1]) if cache is None else (
                    torch.cat([cache[0], new_kv[0]], dim=-2),
                    torch.cat([cache[1], new_kv[1]], dim=-2),
                )
                decoded.append(out)
            assert close(torch.cat(decoded, dim=1), full)

    def test_cache_is_current_tokens_only(self, tensors):
        _, kv = tensors
        mha = MultiHeadAttention(IN, HS, NH, OUT).eval()
        with torch.no_grad():
            k_full = mha.key_matrix(kv).view(B, K, NH, HS).transpose(1, 2)
            v_full = mha.value_matrix(kv).view(B, K, NH, HS).transpose(1, 2)
            _, cache = mha(kv, kv, kv)
            assert cache[0].size(-2) == K
            assert close(cache[0], k_full)
            assert close(cache[1], v_full)

    def test_gqa_cache_shares_kv_heads(self, tensors):
        _, kv = tensors
        gqa = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2).eval()
        with torch.no_grad():
            _, cache = gqa(kv, kv, kv)
            assert cache[0].shape == (B, 2, K, HS)
            assert cache[1].shape == (B, 2, K, HS)


class TestCompile:
    def test_torch_compile(self, tensors):
        q, kv = tensors
        mc = torch.compile(MultiHeadAttention(IN, HS, NH, OUT).eval(), backend="eager")
        with torch.no_grad():
            assert mc(q, kv, kv)[0].shape == (B, Q, OUT)
