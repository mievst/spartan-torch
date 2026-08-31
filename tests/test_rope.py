import pytest
import torch

from spartan_torch import RotaryPositionalEmbedding

B, H, S, D = 2, 4, 8, 16

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


@pytest.fixture
def rope():
    return RotaryPositionalEmbedding(D)


def test_shapes(rope):
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape
    assert kr.shape == k.shape
    assert rope(q, None)[0].shape == q.shape


def test_matches_manual_reference():
    head_size, seq_len = 8, 4
    base = 10000.0
    rope = RotaryPositionalEmbedding(head_size, base=base, max_seq_len=seq_len)
    x = torch.randn(1, 1, seq_len, head_size)

    theta = torch.tensor([base ** (-2 * i / head_size) for i in range(head_size // 2)])
    pos = torch.arange(seq_len, dtype=torch.float32)[:, None]
    freqs = pos * theta
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    rotated = x * cos + torch.cat([-x[..., head_size // 2 :], x[..., : head_size // 2]], dim=-1) * sin

    assert close(rope.rotate(x), rotated)


def test_scores_depend_on_relative_position_only(rope):
    """RoPE core property: shifting both q and k by the same offset preserves scores."""
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    shift = 5
    q_pos = torch.arange(shift, shift + S)
    k_pos = torch.arange(shift, shift + S)

    qr, kr = rope(q, k)
    qr_s, kr_s = rope(q, k, q_pos, k_pos)

    scores = qr @ kr.transpose(-2, -1)
    scores_shifted = qr_s @ kr_s.transpose(-2, -1)
    assert close(scores, scores_shifted)


def test_rotation_preserves_norm(rope):
    x = torch.randn(B, H, S, D)
    assert close(torch.norm(rope.rotate(x), dim=-1), torch.norm(x, dim=-1), 1e-4)


def test_position_zero_identity():
    rope = RotaryPositionalEmbedding(D)
    x = torch.randn(B, H, S, D)
    zeros = torch.zeros(S, dtype=torch.long)
    assert close(rope.rotate(x, zeros), x)


def test_kv_cache_offset_matches_single_step(rope):
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    pos = torch.arange(S) + 3

    qr_full, kr_full = rope(q, k, pos, pos)
    qr_base, kr_base = rope(q, k)
    assert not close(qr_full, qr_base)
    qr_ref, kr_ref = rope(q[:, :, -1:], k[:, :, -1:], pos[-1:], pos[-1:])
    assert close(qr_ref, qr_full[:, :, -1:])
    assert close(kr_ref, kr_full[:, :, -1:])


def test_extends_beyond_max_seq_len():
    rope = RotaryPositionalEmbedding(D, max_seq_len=8)
    x = torch.randn(B, H, 32, D)
    out = rope.rotate(x)
    assert out.shape == x.shape
    assert rope.cos_cache.size(0) == 32
    q = torch.randn(B, H, 16, D)
    assert rope.rotate(q).shape == q.shape


def test_extends_from_arbitrary_positions():
    rope = RotaryPositionalEmbedding(D, max_seq_len=4)
    x = torch.randn(B, H, 2, D)
    pos = torch.tensor([5, 6])
    assert rope.rotate(x, pos).shape == x.shape
    assert rope.cos_cache.size(0) == 7


def test_odd_head_size_rejected():
    with pytest.raises(ValueError, match="must be even"):
        RotaryPositionalEmbedding(15)


def test_wrong_last_dim_rejected(rope):
    x = torch.randn(B, H, S, D + 4)
    with pytest.raises(ValueError, match="head_size"):
        rope.rotate(x)


def test_state_dict_roundtrip():
    rope = RotaryPositionalEmbedding(D)
    state = rope.state_dict()
    clone = RotaryPositionalEmbedding(D)
    clone.load_state_dict(state)
    x = torch.randn(B, H, S, D)
    assert close(rope.rotate(x), clone.rotate(x))


def test_buffers_not_trainable(rope):
    assert not any(p.requires_grad for p in rope.parameters())
    assert all("cache" in n for n in rope.state_dict())


def test_gradients_flow(rope):
    x = torch.randn(B, H, S, D, requires_grad=True)
    rope.rotate(x).square().mean().backward()
    assert x.grad is not None


def test_half_precision():
    rope = RotaryPositionalEmbedding(D).half()
    x = torch.randn(B, H, S, D).half()
    assert rope.rotate(x).dtype == torch.float16
    long_x = torch.randn(B, H, 32, D).half()
    assert rope.rotate(long_x).shape == long_x.shape


@CUDA
def test_cuda_forward():
    rope = RotaryPositionalEmbedding(D).cuda()
    x = torch.randn(B, H, S, D, device="cuda")
    assert rope.rotate(x).device.type == "cuda"
    long_x = torch.randn(B, H, 32, D, device="cuda")
    assert rope.rotate(long_x).shape == long_x.shape


def test_torch_compile():
    rope = torch.compile(RotaryPositionalEmbedding(D).eval(), backend="eager")
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    with torch.no_grad():
        qr, kr = rope(q, k)
    assert qr.shape == q.shape
    assert kr.shape == k.shape
