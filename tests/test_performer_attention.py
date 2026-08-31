import math

import pytest
import torch

from spartan_torch import PerformerAttention
from spartan_torch.transformers.attention.performer import (
    _bidir_context,
    _causal_context,
    _softmax_features,
    gaussian_orthogonal_random_matrix,
)

B, N, IN, OUT, HS, NH = 2, 32, 64, 64, 16, 4

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture()
def tensors():
    torch.manual_seed(0)
    # Moderate norms: FAVOR+'s error depends on ||q||, ||k|| (paper), not just m.
    return torch.randn(B, N, IN) * 0.5


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def make(**kwargs):
    kw = dict(num_features=4096)
    kw.update(kwargs)
    return PerformerAttention(IN, HS, NH, OUT, **kw)


def headify(x):
    return x.view(B, -1, NH, HS).transpose(1, 2)


def exact_reference(layer, x, is_causal):
    with torch.no_grad():
        qh = headify(layer.query_matrix(x))
        kh = headify(layer.key_matrix(x))
        vh = headify(layer.value_matrix(x))
        scores = qh @ kh.transpose(-2, -1) / layer.head_size**0.5
        if is_causal:
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        context = torch.softmax(scores, dim=-1) @ vh
        return layer.out(context.transpose(1, 2).reshape(B, N, -1))


class TestPerformerAttention:
    def test_shapes(self, tensors):
        x = tensors
        assert make(is_causal=True)(x, x, x).shape == (B, N, OUT)
        assert make(is_causal=False)(x, x, x).shape == (B, N, OUT)

    def test_deterministic(self, tensors):
        x = tensors
        layer = make().eval()
        assert torch.equal(layer(x, x, x), layer(x, x, x))

    def test_converges_to_exact_bidirectional(self, tensors):
        x = tensors
        layer = make().eval()
        assert close(layer(x, x, x), exact_reference(layer, x, is_causal=False), tol=2e-2)

    def test_converges_to_exact_causal(self, tensors):
        x = tensors
        layer = make(is_causal=True).eval()
        assert close(layer(x, x, x), exact_reference(layer, x, is_causal=True), tol=2e-2)

    def test_accuracy_improves_with_features(self, tensors):
        x = tensors
        base = make()  # fixes the random projections
        with torch.no_grad():
            ref = exact_reference(base.eval(), x, is_causal=False)
            sd = base.state_dict()
            del sd["R"]  # features change with m, projections do not
            low = make(num_features=64).eval()
            low.load_state_dict(sd, strict=False)
            high = make(num_features=8192).eval()
            high.load_state_dict(sd, strict=False)
            err_low = (low(x, x, x) - ref).abs().max().item()
            err_high = (high(x, x, x) - ref).abs().max().item()
        assert err_low > err_high
        assert err_high < 1e-2

    def test_causal_chunked_equals_global_cumsum(self):
        torch.manual_seed(0)
        q = torch.rand(B, NH, 40, 64)  # spans 3 chunks of chunk_size=16
        k = torch.rand(B, NH, 40, 64)
        v = torch.rand(B, NH, 40, HS)
        chunked = _causal_context(q, k, v, chunk_size=16)
        kc = k.cumsum(dim=-2)
        kvc = torch.einsum("bhjm,bhje->bhjme", k, v).cumsum(dim=-3)
        num = torch.einsum("bhjm,bhjmd->bhjd", q, kvc)
        den = torch.einsum("bhjm,bhjm->bhj", q, kc).unsqueeze(-1)
        assert close(chunked, num / (den + 1e-6))

    def test_causal_no_future_value_leak(self, tensors):
        x = tensors
        layer = make(is_causal=True).eval()
        with torch.no_grad():
            y0 = layer(x, x, x)
            xp = x.clone()
            xp[:, N // 2 :, :] += 10.0
            yp = layer(xp, xp, xp)
        assert close(y0[:, : N // 2], yp[:, : N // 2])

    def test_key_mask_equals_truncated_sequence(self, tensors):
        x = tensors
        layer = make().eval()
        mask = torch.zeros(B, 1, 1, N, dtype=torch.bool)
        mask[:, :, :, N // 2 :] = True
        with torch.no_grad():
            masked = layer(x, x, x, mask=mask)
            xp = torch.cat([x[:, : N // 2], torch.zeros_like(x[:, : N // 2])], dim=1)
            truncated = layer(xp, xp, xp, mask=mask)
        assert close(masked[:, : N // 2], truncated[:, : N // 2], tol=1e-4)

    def test_float_mask_raises(self, tensors):
        x = tensors
        with pytest.raises(NotImplementedError):
            make()(x, x, x, mask=torch.zeros(B, 1, 1, N))

    def test_pairwise_mask_raises(self, tensors):
        x = tensors
        with pytest.raises(NotImplementedError):
            make()(x, x, x, mask=torch.zeros(B, 1, N, N, dtype=torch.bool))

    def test_mask_rank_raises(self, tensors):
        x = tensors
        with pytest.raises(NotImplementedError):
            make()(x, x, x, mask=torch.zeros(B, N, dtype=torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0))

    def test_mask_length_mismatch_raises(self, tensors):
        x = tensors
        with pytest.raises(ValueError, match="key mask length"):
            make()(x, x, x, mask=torch.zeros(B, 1, 1, N - 1, dtype=torch.bool))

    def test_past_key_value_raises(self, tensors):
        x = tensors
        with pytest.raises(NotImplementedError):
            make()(x, x, x, past_key_value=(torch.randn(B, N, IN), torch.randn(B, N, IN)))

    def test_causal_length_mismatch_raises(self, tensors):
        x = tensors
        with pytest.raises(ValueError, match="causal"):
            make(is_causal=True)(x, x[:, : N - 1, :], x)

    def test_R_is_non_trainable_buffer(self, tensors):
        x = tensors
        layer = make()
        assert isinstance(layer.R, torch.Tensor)
        layer(x, x, x).sum().backward()
        assert layer.R.grad is None
        assert "R" in layer.state_dict()

    def test_features_positive_and_bounded(self):
        torch.manual_seed(0)
        x = torch.randn(2, 2, 8, 8) * 0.35
        R = gaussian_orthogonal_random_matrix(64, 8)
        qf = _softmax_features(x, R, is_query=True)
        kf = _softmax_features(x, R, is_query=False)
        assert qf.min() > 0.0 and qf.max() <= 1.0  # per-row query shift caps features
        assert kf.min() > 0.0  # keys unshifted (position-independent) but positive
        assert torch.isfinite(qf).all() and torch.isfinite(kf).all()
        assert qf.dtype == torch.float32

    def test_redraw_projection(self, tensors):
        x = tensors
        layer = make(feature_redraw_interval=2)
        layer.train()
        r0 = layer.R.clone()
        layer(x, x, x)
        assert torch.equal(layer.R, r0)
        layer(x, x, x)
        assert not torch.equal(layer.R, r0)
        layer(x, x, x)
        r1 = layer.R.clone()
        layer.eval()
        layer(x, x, x)
        assert torch.equal(layer.R, r1)

    def test_ortho_scaling_1(self, tensors):
        x = tensors
        layer = make(ortho_scaling=1).eval()
        assert close(layer.R.norm(dim=1), torch.full((layer.num_features,), HS**0.5))
        assert layer(x, x, x).shape == (B, N, OUT)

    def test_gradients_flow(self, tensors):
        x = tensors
        layer = make()
        layer(x, x, x).sum().backward()
        for p in (layer.query_matrix.weight, layer.key_matrix.weight, layer.value_matrix.weight, layer.out.weight):
            assert p.grad is not None and torch.isfinite(p.grad).all()

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
        assert torch.equal(layer(x, x, x), layer2(x, x, x))

    @CUDA
    def test_fp16(self, tensors):
        x = tensors.cuda().half()
        layer = make(num_features=16384, is_causal=True).cuda().half().eval()
        assert torch.isfinite(layer(x, x, x)).all()

    def test_compile(self, tensors):
        x = tensors
        layer = torch.compile(make(is_causal=True).eval(), backend="eager")
        with torch.no_grad():
            assert layer(x, x, x).shape == (B, N, OUT)
