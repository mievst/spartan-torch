import pytest
import torch
import torch.nn.functional as F
from torch import nn

from spartan_torch import LinearTransformerAttention

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


def make(**kwargs):
    return LinearTransformerAttention(IN, HS, NH, OUT, **kwargs)


def headify(x):
    return x.view(B, -1, NH, HS).transpose(1, 2)


def feature_maps(layer, q, kv, use_dropout=False):
    with torch.no_grad():
        qf = F.elu(headify(layer.query_matrix(q))) + 1.0
        kf = F.elu(headify(layer.key_matrix(kv))) + 1.0
        vf = headify(layer.value_matrix(kv))
        return qf, kf, vf


def manual(layer, qf, kf, vf, is_causal):
    d = layer.head_size
    if is_causal:
        kv = torch.einsum("bhjd,bhje->bhjde", kf, vf).cumsum(dim=2)
        z = kf.cumsum(dim=2)
        num = torch.einsum("bhld,bhlde->bhle", qf, kv)
        den = torch.einsum("bhld,bhld->bhl", qf, z).unsqueeze(-1)
    else:
        kv = torch.einsum("bhjd,bhje->bhde", kf, vf)
        z = kf.sum(dim=2)
        num = torch.einsum("bhld,bhde->bhle", qf, kv)
        den = torch.einsum("bhld,bhd->bhl", qf, z).unsqueeze(-1)
    context = num / den
    return layer.out(context.transpose(1, 2).reshape(B, -1, layer.num_heads * d))


class TestLinearTransformerAttention:
    def test_shapes(self, tensors):
        q, kv = tensors
        assert make()(q, kv, kv).shape == (B, Q, OUT)

    def test_cross_attention_query_in_size(self, tensors):
        _, kv = tensors
        layer = make(query_in_size=24)
        assert layer(torch.randn(B, Q, 24), kv, kv).shape == (B, Q, OUT)

    def test_matches_manual_reference_bidirectional(self, tensors):
        q, kv = tensors
        layer = make(is_causal=False).eval()
        qf, kf, vf = feature_maps(layer, q, kv)
        assert close(layer(q, kv, kv), manual(layer, qf, kf, vf, is_causal=False))

    def test_matches_manual_reference_causal(self, tensors):
        q, _ = tensors
        layer = make(is_causal=True).eval()
        qf, kf, vf = feature_maps(layer, q, q)
        assert close(layer(q, q, q), manual(layer, qf, kf, vf, is_causal=True))

    def test_causal_prefix_ignores_future(self, tensors):
        q, _ = tensors
        layer = make(is_causal=True).eval()
        with torch.no_grad():
            y0 = layer(q, q, q)
            qp = q.clone()
            qp[:, Q // 2 :, :] += 10.0
            yp = layer(qp, qp, qp)
        assert close(y0[:, : Q // 2], yp[:, : Q // 2])

    def test_causal_is_position_exact(self, tensors):
        """With n = q_len and random data, prefix form must match the
        elementwise sum over j <= i (numerically robust for tiny tensors)."""
        q, _ = tensors
        layer = make(is_causal=True).eval()
        qf, kf, vf = feature_maps(layer, q, q)
        # brute-force: out_i = sum_{j<=i} w_ij v_j / sum_{j<=i} w_ij
        w = qf @ kf.transpose(-2, -1)
        tril = torch.tril(torch.ones(Q, Q))
        w = w * tril
        context = (w @ vf) / w.sum(-1, keepdim=True)
        ref = layer.out(context.transpose(1, 2).reshape(B, Q, -1))
        assert close(layer(q, q, q), ref)

    def test_causal_requires_equal_lengths(self, tensors):
        q, kv = tensors
        with pytest.raises(ValueError, match="query_seq_len"):
            make(is_causal=True)(q, kv, kv)

    def test_mask_and_cache_raise(self, tensors):
        q, kv = tensors
        layer = make()
        with pytest.raises(NotImplementedError):
            layer(q, kv, kv, mask=torch.ones(B, 1, Q))
        with pytest.raises(NotImplementedError):
            layer(q, kv, kv, past_key_value=(torch.randn(B, K, IN), torch.randn(B, K, IN)))

    def test_dropout_train_vs_eval(self, tensors):
        q, kv = tensors
        torch.manual_seed(0)
        layer = make(attn_p=0.5)
        layer.eval()
        out_eval = layer(q, kv, kv)
        layer.train()
        outs = {layer(q, kv, kv) for _ in range(5)}
        assert len(outs) > 1
        layer.eval()
        assert torch.equal(out_eval, layer(q, kv, kv))

    def test_positive_normalizer(self, tensors):
        q, kv = tensors
        layer = make(is_causal=True).eval()
        assert torch.isfinite(layer(q, q, q)).all()

    def test_state_dict_roundtrip(self, tensors):
        q, _ = tensors
        layer = make(is_causal=True)
        layer2 = make(is_causal=True)
        layer2.load_state_dict(layer.state_dict())
        assert close(layer(q, q, q), layer2(q, q, q))

    def test_gradients_flow(self, tensors):
        q, kv = tensors
        layer = make()
        layer(q, kv, kv).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())

    @CUDA
    def test_fp16(self, tensors):
        q, kv = [t.cuda().half() for t in tensors]
        layer = make().cuda().half().eval()
        assert torch.isfinite(layer(q, kv, kv)).all()

    @pytest.mark.parametrize("is_causal", [False, True])
    def test_compile(self, tensors, is_causal):
        q, kv = tensors
        layer = torch.compile(make(is_causal=is_causal).eval(), backend="eager")
        with torch.no_grad():
            if is_causal:
                assert layer(q, q, q).shape == (B, Q, OUT)
            else:
                assert layer(q, kv, kv).shape == (B, Q, OUT)


class TestInjection:
    def test_inject_into_transformer_block(self, tensors):
        from spartan_torch import TransformerBlock

        q, _ = tensors
        block = TransformerBlock(
            in_size=IN,
            head_size=HS,
            num_heads=NH,
            out_size=OUT,
            ff_hidden_size=32,
            attn_layer=make(is_causal=True),
        )
        block.eval()
        assert block(q)[0].shape == (B, Q, OUT)
        # attention weights present exactly once in state_dict
        names = [k for k in block.state_dict() if "query_matrix" in k]
        assert len(names) == 1
