"""Numerical parity: manual attention path vs F.scaled_dot_product_attention.

``tests/test_mha.py`` already compares ``use_sdpa=False`` against the internal
``use_sdpa=True`` flag (forward only). Here the reference is the external
``torch.nn.functional.scaled_dot_product_attention`` itself, forward AND
backward (input grads + all parameter grads), in fp64 where backends allow.

Convention note: our layer uses ``True = masked out``; ``F.sdpa`` uses
``True = participate`` — the reference inverts bool masks, mirroring what
``_attention`` does on its SDPA branch.

Gates: forward ``max abs diff < 1e-10`` (fp64), backward ``< 1e-8`` (fp64).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from spartan_torch import MultiHeadAttention

B, Q, K, IN, OUT, HS, NH = 2, 5, 7, 16, 20, 8, 4


def _reference_sdpa_forward(mha, q, k, v, mask=None):
    """Manual QKV projections + raw F.sdpa + out projection (the spec)."""
    b, q_len, nq = q.shape[0], q.shape[1], mha.num_heads
    k_len, nkv, hs = k.shape[1], mha.num_kv_heads, mha.head_size
    qq = mha.query_matrix(q).view(b, q_len, nq, hs).transpose(1, 2)
    kk = mha.key_matrix(k).view(b, k_len, nkv, hs).transpose(1, 2)
    vv = mha.value_matrix(v).view(b, k_len, nkv, hs).transpose(1, 2)
    if nkv != nq:
        rep = nq // nkv
        kk = kk.repeat_interleave(rep, dim=1)
        vv = vv.repeat_interleave(rep, dim=1)
    attn_mask = None
    if mask is not None:
        attn_mask = ~mask if mask.dtype == torch.bool else mask
    ctx = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=attn_mask)
    return mha.out(ctx.transpose(1, 2).reshape(b, q_len, -1))


class TestSdpaForwardParity:
    def test_plain_matches_sdpa(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(IN, HS, NH, OUT).double().eval()
        q = torch.randn(B, Q, IN, dtype=torch.float64)
        kv = torch.randn(B, K, IN, dtype=torch.float64)
        with torch.no_grad():
            got = mha(q, kv, kv)[0]
            ref = _reference_sdpa_forward(mha, q, kv, kv)
        assert (got - ref).abs().max().item() < 1e-10

    def test_causal_matches_sdpa(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(IN, HS, NH, OUT, is_causal=True).double().eval()
        kv = torch.randn(B, K, IN, dtype=torch.float64)
        causal_mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
        with torch.no_grad():
            got = mha(kv, kv, kv)[0]
            ref = _reference_sdpa_forward(mha, kv, kv, kv, mask=causal_mask)
        assert (got - ref).abs().max().item() < 1e-10

    def test_bool_and_float_masks_match_sdpa(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(IN, HS, NH, OUT).double().eval()
        q = torch.randn(B, Q, IN, dtype=torch.float64)
        kv = torch.randn(B, K, IN, dtype=torch.float64)
        bool_mask = torch.triu(torch.ones(Q, K, dtype=torch.bool), diagonal=1)
        float_mask = torch.zeros(Q, K, dtype=torch.float64).masked_fill(bool_mask, float("-inf"))
        with torch.no_grad():
            got_b = mha(q, kv, kv, mask=bool_mask)[0]
            got_f = mha(q, kv, kv, mask=float_mask)[0]
            ref = _reference_sdpa_forward(mha, q, kv, kv, mask=bool_mask)
            ref_f = _reference_sdpa_forward(mha, q, kv, kv, mask=float_mask)
        assert (got_b - ref).abs().max().item() < 1e-10
        assert (got_f - ref_f).abs().max().item() < 1e-10
        assert (got_b - got_f).abs().max().item() < 1e-10

    def test_gqa_matches_sdpa(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(IN, HS, NH, OUT, num_kv_heads=2).double().eval()
        q = torch.randn(B, Q, IN, dtype=torch.float64)
        kv = torch.randn(B, K, IN, dtype=torch.float64)
        with torch.no_grad():
            got = mha(q, kv, kv)[0]
            ref = _reference_sdpa_forward(mha, q, kv, kv)
        assert (got - ref).abs().max().item() < 1e-10

    def test_fp32_tolerance_documented(self):
        # fp32 accumulates ~1e-7 matmul ordering noise; gate 1e-5 holds.
        torch.manual_seed(0)
        mha = MultiHeadAttention(IN, HS, NH, OUT).float().eval()
        q = torch.randn(B, Q, IN)
        kv = torch.randn(B, K, IN)
        with torch.no_grad():
            got = mha(q, kv, kv)[0]
            ref = _reference_sdpa_forward(mha, q, kv, kv)
        assert (got - ref).abs().max().item() < 1e-5


class TestSdpaBackwardParity:
    def _grads(self, make_out):
        """Run forward+backward via make_out(mha, q, kv)->tensor; return grads."""
        torch.manual_seed(0)
        mha = MultiHeadAttention(IN, HS, NH, OUT).double()
        q = torch.randn(B, Q, IN, dtype=torch.float64, requires_grad=True)
        kv = torch.randn(B, K, IN, dtype=torch.float64, requires_grad=True)
        make_out(mha, q, kv).square().mean().backward()
        return q.grad.clone(), kv.grad.clone(), [p.grad.clone() for p in mha.parameters()]

    def test_input_and_param_grads_match(self):
        manual = self._grads(lambda m, q, kv: m(q, kv, kv)[0])
        ref = self._grads(lambda m, q, kv: _reference_sdpa_forward(m, q, kv, kv))
        assert (manual[0] - ref[0]).abs().max().item() < 1e-8
        assert (manual[1] - ref[1]).abs().max().item() < 1e-8
        for g, r in zip(manual[2], ref[2]):
            assert (g - r).abs().max().item() < 1e-8

    def test_causal_input_grads_match(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(IN, HS, NH, OUT).double()
        kv = torch.randn(B, K, IN, dtype=torch.float64, requires_grad=True)
        mc = MultiHeadAttention(IN, HS, NH, OUT, is_causal=True).double()
        mc.load_state_dict(mha.state_dict())
        mc(kv, kv, kv)[0].square().mean().backward()
        g_manual = kv.grad.clone()

        torch.manual_seed(0)
        mha2 = MultiHeadAttention(IN, HS, NH, OUT).double()
        kv2 = torch.randn(B, K, IN, dtype=torch.float64, requires_grad=True)
        mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
        _reference_sdpa_forward(mha2, kv2, kv2, kv2, mask=mask).square().mean().backward()
        assert (g_manual - kv2.grad).abs().max().item() < 1e-8

    def test_scale_matches_sdpa_convention(self):
        # Both divide by sqrt(head_size); a scale mismatch would show as a
        # systematic (not noise-level) forward gap on large logits.
        torch.manual_seed(1)
        mha = MultiHeadAttention(IN, HS, NH, OUT).double().eval()
        q = torch.randn(B, Q, IN, dtype=torch.float64) * 3.0
        kv = torch.randn(B, K, IN, dtype=torch.float64) * 3.0
        with torch.no_grad():
            qq = mha.query_matrix(q).view(B, Q, NH, HS).transpose(1, 2) / math.sqrt(HS)
            kk = mha.key_matrix(kv).view(B, K, NH, HS).transpose(1, 2)
            w_manual = torch.softmax(qq @ kk.transpose(-2, -1), dim=-1)
            got = mha(q, kv, kv)[0]
            ref = _reference_sdpa_forward(mha, q, kv, kv)
        assert w_manual.shape == (B, NH, Q, K)
        assert (got - ref).abs().max().item() < 1e-9
