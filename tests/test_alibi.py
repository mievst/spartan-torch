import math

import pytest
import torch

from spartan_torch import ALiBiBias, MultiHeadAttention
from spartan_torch.transformers.positional.alibi import alibi_slopes


def test_slopes_pow2():
    assert alibi_slopes(8) == pytest.approx(
        [2 ** (-8 / 8 * (i + 1)) for i in range(8)], rel=1e-6
    )


def test_slopes_non_pow2_length_and_range():
    slopes = alibi_slopes(12)
    assert len(slopes) == 12
    assert all(0.0 < s < 1.0 for s in slopes)
    # Construction from the paper: slopes(8) + every second slope of slopes(16).
    assert slopes[:8] == pytest.approx(alibi_slopes(8))
    assert slopes[8:] == pytest.approx(alibi_slopes(16)[0::2][:4])


def test_slopes_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        alibi_slopes(0)


def test_bias_shape_and_causal_mask():
    b = ALiBiBias(4)
    out = b(5, 7)
    assert out.shape == (1, 4, 5, 7)
    # Causal: future positions are -inf in every head.
    assert torch.isneginf(out[0, :, 0, 1:]).all()
    assert not torch.isneginf(out[0, :, 0, 0]).any()


def test_bias_linear_in_distance():
    b = ALiBiBias(2)
    out = b(1, 5, causal=False)[0, 0, 0]
    expected = -b.slopes[0].item() * torch.arange(5, dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_symmetric_without_causal():
    b = ALiBiBias(4)
    out = b(6, 6, causal=False)[0, 0]
    assert torch.allclose(out, out.T)


def test_k_offset_keeps_absolute_positions():
    b = ALiBiBias(4)
    full = b(6, 6, causal=False)
    tail = b(2, 2, k_offset=4, causal=False)
    assert torch.allclose(tail, full[:, :, 4:, 4:])


def test_no_max_len_extrapolates():
    b = ALiBiBias(8)
    assert b(4096, 4096).shape == (1, 8, 4096, 4096)


def test_buffers_not_trainable():
    b = ALiBiBias(8)
    assert not any(p.requires_grad for p in b.parameters())
    assert "slopes" not in b.state_dict()


def test_injects_into_mha_as_float_mask():
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, 8, 4, 32).eval()
    alibi = ALiBiBias(4)
    q = torch.randn(2, 6, 32)
    kv = torch.randn(2, 6, 32)
    with torch.no_grad():
        plain = mha(q, kv, kv)[0]
        biased = mha(q, kv, kv, mask=alibi(6, 6))[0]
    assert biased.shape == plain.shape
    assert not torch.allclose(plain, biased)
    assert not torch.isnan(biased).any()


def test_head_ordering_follows_slopes():
    # Head 0 has the largest slope → strongest distance penalty → its
    # attention mass concentrates closer to the diagonal than head -1.
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, 8, 4, 32).eval()
    alibi = ALiBiBias(4)
    kv = torch.randn(1, 16, 32)
    with torch.no_grad():
        q = mha.query_matrix(kv).view(1, 16, 4, 8).transpose(1, 2)
        k = mha.key_matrix(kv).view(1, 16, 4, 8).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(8) + alibi(16, 16)
        w = torch.softmax(scores, dim=-1)[0]
        spread = ((w * torch.arange(16).float()).sum(-1) - torch.arange(16).float()).abs().mean(-1)
    assert spread[0] < spread[-1]
