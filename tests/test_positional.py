import pytest
import torch

from spartan_torch import PositionalEncoding


@pytest.fixture(params=[64, 128])
def emb_size(request):
    return request.param


@pytest.fixture(params=[16, 32])
def seq_len(request):
    return request.param


def test_output_shape(emb_size, seq_len):
    pe = PositionalEncoding(emb_size, max_seq_len=64)
    x = torch.randn(2, seq_len, emb_size)
    out = pe(x)
    assert out.shape == x.shape


def test_adds_pe(emb_size, seq_len):
    pe = PositionalEncoding(emb_size, max_seq_len=64)
    x = torch.zeros(2, seq_len, emb_size)
    out = pe(x)
    expected = pe.pe[:, :seq_len].expand(2, -1, -1)
    assert torch.allclose(out, expected)


def test_pe_shape():
    pe = PositionalEncoding(64, max_seq_len=32)
    assert pe.pe.shape == (1, 32, 64)


def test_sin_cos_pair_identity(emb_size):
    pe = PositionalEncoding(emb_size, max_seq_len=8)
    pairs = pe.pe[:, :, 0::2] ** 2 + pe.pe[:, :, 1::2] ** 2
    assert torch.allclose(pairs, torch.ones_like(pairs), atol=1e-6)


def test_slicing_respects_max_len():
    pe = PositionalEncoding(64, max_seq_len=16)
    x = torch.randn(1, 16, 64)
    assert pe(x).shape == x.shape
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        pe(torch.randn(1, 17, 64))


def test_odd_emb_size_rejected():
    with pytest.raises(ValueError, match="must be even"):
        PositionalEncoding(65)


def test_buffer_not_trainable():
    pe = PositionalEncoding(64)
    assert not any(p.requires_grad for p in pe.parameters())
