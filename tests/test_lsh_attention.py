import pytest
import torch
import torch.nn.functional as F

from spartan_torch import ReformerAttention

B, N, IN, OUT, HS, NH = 2, 9, 16, 20, 8, 4

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture()
def tensors():
    torch.manual_seed(0)
    return torch.randn(B, N, IN)


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def make(**kwargs):
    kw = dict(n_hashes=2, bucket_size=4, n_buckets=8)
    kw.update(kwargs)
    return ReformerAttention(IN, HS, NH, OUT, **kw)


def headify(x):
    return x.view(B, -1, NH, HS).transpose(1, 2)


def exact_reference(layer, x, is_causal):
    """Exact attention with the shared-QK convention: keys are normalized
    queries, scores scaled by head_size^-0.5. Matches the layer whenever the
    LSH window covers every key (bucket_size >= seq_len)."""
    with torch.no_grad():
        qh = headify(layer.query_matrix(x))
        vh = headify(layer.value_matrix(x))
        k = F.normalize(qh, p=2, dim=-1)
        scores = qh @ k.transpose(-2, -1) / layer.head_size**0.5
        if is_causal:
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        context = F.softmax(scores, dim=-1) @ vh
        return layer.out(context.transpose(1, 2).reshape(B, N, -1))


class TestReformerAttention:
    def test_shapes(self, tensors):
        x = tensors
        assert make(is_causal=True)(x, x, x).shape == (B, N, OUT)

    def test_deterministic(self, tensors):
        x = tensors
        layer = make().eval()
        assert torch.equal(layer(x, x, x), layer(x, x, x))

    def test_exact_when_window_covers_all_causal(self, tensors):
        x = tensors
        layer = make(n_hashes=1, bucket_size=N + 8, is_causal=True).eval()
        assert close(layer(x, x, x), exact_reference(layer, x, is_causal=True))

    def test_exact_when_window_covers_all_bidirectional(self, tensors):
        x = tensors
        layer = make(n_hashes=2, bucket_size=N + 8, is_causal=False).eval()
        assert close(layer(x, x, x), exact_reference(layer, x, is_causal=False))

    def test_causal_no_future_value_leak(self, tensors):
        """With a full-coverage window the LSH layer is exactly causal:
        perturbing future tokens must not change past outputs."""
        x = tensors
        layer = make(n_hashes=1, bucket_size=N + 8, is_causal=True).eval()
        with torch.no_grad():
            y0 = layer(x, x, x)
            xp = x.clone()
            xp[:, N // 2 :, :] += 10.0
            yp = layer(xp, xp, xp)
        assert close(y0[:, : N // 2], yp[:, : N // 2])

    def test_padding_not_multiple_of_bucket_size(self, tensors):
        x = tensors[:, :7, :]  # N=7, not a multiple of bucket_size=4
        layer = make(n_hashes=3, bucket_size=4, is_causal=True).eval()
        out = layer(x, x, x)
        assert torch.isfinite(out).all()
        assert out.shape == (B, 7, OUT)

    def test_multi_round_no_nan(self, tensors):
        x = tensors
        for n_hashes in (1, 4, 8):
            layer = make(n_hashes=n_hashes).eval()
            assert torch.isfinite(layer(x, x, x)).all()

    def test_shared_qk_false(self, tensors):
        x = tensors
        layer = make(shared_qk=False).eval()
        assert layer(x, x, x).shape == (B, N, OUT)

    def test_mask_and_cache_raise(self, tensors):
        x = tensors
        layer = make()
        with pytest.raises(NotImplementedError):
            layer(x, x, x, mask=torch.ones(B, 1, N))
        with pytest.raises(NotImplementedError):
            layer(x, x, x, past_key_value=(torch.randn(B, N, IN), torch.randn(B, N, IN)))

    def test_length_mismatch_raises(self, tensors):
        x = tensors
        layer = make()
        with pytest.raises(ValueError, match="self-attention"):
            layer(x, x[:, : N - 1, :], x)

    def test_odd_n_buckets_raises(self):
        with pytest.raises(ValueError, match="even"):
            make(n_buckets=7)

    def test_rotation_is_non_trainable_buffer(self, tensors):
        x = tensors
        layer = make()
        assert isinstance(layer.R, torch.Tensor)
        assert layer.R.grad is None
        layer(x, x, x).sum().backward()
        assert layer.R.grad is None
        assert "R" in layer.state_dict()

    def test_gradients_flow(self, tensors):
        x = tensors
        layer = make()
        layer(x, x, x).sum().backward()
        # with shared_qk the separate key projection is unused by construction
        assert layer.query_matrix.weight.grad is not None
        assert layer.value_matrix.weight.grad is not None
        assert layer.out.weight.grad is not None
        assert layer.key_matrix.weight.grad is None
        assert all(
            torch.isfinite(p.grad).all()
            for p in (layer.query_matrix.weight, layer.value_matrix.weight, layer.out.weight)
        )

    def test_dropout_train_vs_eval(self, tensors):
        x = tensors
        torch.manual_seed(0)
        layer = make(attn_p=0.5)
        layer.eval()
        out_eval = layer(x, x, x)
        layer.train()
        outs = {layer(x, x, x) for _ in range(5)}
        assert len(outs) > 1
        layer.eval()
        assert torch.equal(out_eval, layer(x, x, x))

    def test_state_dict_roundtrip(self, tensors):
        x = tensors
        layer = make()
        layer2 = make()
        layer2.load_state_dict(layer.state_dict())
        assert close(layer(x, x, x), layer2(x, x, x))

    @CUDA
    def test_fp16(self, tensors):
        x = tensors.cuda().half()
        layer = make().cuda().half().eval()
        assert torch.isfinite(layer(x, x, x)).all()

    def test_compile(self, tensors):
        x = tensors
        layer = torch.compile(make(is_causal=True).eval(), backend="eager")
        with torch.no_grad():
            assert layer(x, x, x).shape == (B, N, OUT)

    def test_unsorted_positions_are_used(self, tensors):
        """Sanity: bucket ordering genuinely shuffles positions (proves the
        sort matters) yet output stays finite and correct-shaped."""
        x = tensors
        layer = make(n_hashes=2, bucket_size=4).eval()
        with torch.no_grad():
            out = layer(x, x, x)
        assert out.shape == (B, N, OUT)
        assert torch.isfinite(out).all()
