import pytest
import torch
from torch import nn

from spartan_torch import LinformerAttention, LinformerSeqProjection, TransformerBlock

B, Q, K, IN, OUT, HS, NH = 2, 5, 7, 16, 20, 8, 4
PROJ_K, MAX_SEQ = 3, 8

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
    kw = dict(proj_k=PROJ_K, max_seq_len=MAX_SEQ)
    kw.update(kwargs)
    return LinformerAttention(IN, HS, NH, OUT, **kw)


def project_manual(x, proj):
    """Reference seq-dim contraction matching LinformerSeqProjection._contract."""
    if proj.dim() == 3:
        return torch.einsum("bhld,hlk->bhkd", x, proj)
    return torch.einsum("bhld,lk->bhkd", x, proj)


class TestLinformerAttention:
    def test_shapes(self, tensors):
        q, kv = tensors
        assert make()(q, kv, kv).shape == (B, Q, OUT)

    def test_cross_attention_query_in_size(self, tensors):
        _, kv = tensors
        la = make(query_in_size=24)
        assert la(torch.randn(B, Q, 24), kv, kv).shape == (B, Q, OUT)

    @pytest.mark.parametrize("sharing", ["none", "headwise", "kv", "layerwise"])
    def test_matches_manual_reference(self, tensors, sharing):
        q, kv = tensors
        la = make(sharing=sharing).eval()
        with torch.no_grad():
            out = la(q, kv, kv)
            qq = la.query_matrix(q).view(B, Q, NH, HS).transpose(1, 2)
            kk = la.key_matrix(kv).view(B, K, NH, HS).transpose(1, 2)
            vv = la.value_matrix(kv).view(B, K, NH, HS).transpose(1, 2)
            e = la.proj.E
            f = la.proj.E if la.proj.sharing in ("kv", "layerwise") else la.proj.F
            if sharing in ("none", "kv"):
                kp = project_manual(kk, e[:, :K])
                vp = project_manual(vv, f[:, :K])
            else:
                kp = project_manual(kk, e[:K])
                vp = project_manual(vv, f[:K])
            w = torch.softmax(qq @ kp.transpose(-2, -1) / HS**0.5, dim=-1)
            ref = la.out((w @ vp).transpose(1, 2).reshape(B, Q, -1))
            assert close(out, ref)

    def test_scores_have_projected_length(self, tensors):
        q, kv = tensors
        la = make().eval()
        with torch.no_grad():
            qq = la.query_matrix(q).view(B, Q, NH, HS).transpose(1, 2)
            kk = la.key_matrix(kv).view(B, K, NH, HS).transpose(1, 2)
            kp = project_manual(kk, la.proj.E[:K])
            scores = qq @ kp.transpose(-2, -1)
            assert scores.shape == (B, NH, Q, PROJ_K)

    def test_dropout_train_vs_eval(self, tensors):
        q, kv = tensors
        la = make(attn_p=0.3)
        la.train()
        assert not close(la(q, kv, kv), la(q, kv, kv), 1e-4)
        la.eval()
        assert close(la(q, kv, kv), la(q, kv, kv), 0.0)

    def test_state_dict_roundtrip(self, tensors):
        q, kv = tensors
        la = make(sharing="kv")
        clone = make(sharing="kv")
        clone.load_state_dict(la.state_dict())
        assert clone(q, kv, kv).shape == (B, Q, OUT)


class TestSharing:
    def test_none_has_per_head_e_and_f(self):
        la = make(sharing="none")
        assert set(la.state_dict()) >= {"proj.E", "proj.F"}
        assert la.proj.E.shape == (NH, MAX_SEQ, PROJ_K)
        assert la.proj.F.shape == (NH, MAX_SEQ, PROJ_K)

    def test_headwise_has_shared_e_and_f(self):
        la = make(sharing="headwise")
        assert set(la.state_dict()) >= {"proj.E", "proj.F"}
        assert la.proj.E.shape == (MAX_SEQ, PROJ_K)
        assert la.proj.F.shape == (MAX_SEQ, PROJ_K)

    def test_kv_has_single_per_head_matrix(self):
        la = make(sharing="kv")
        assert set(la.state_dict()) == {
            "proj.E", "query_matrix.weight", "key_matrix.weight", "value_matrix.weight", "out.weight", "out.bias"
        }
        assert la.proj.E.shape == (NH, MAX_SEQ, PROJ_K)
        assert not hasattr(la.proj, "F")

    def test_layerwise_has_single_shared_matrix(self):
        la = make(sharing="layerwise")
        assert set(la.state_dict()) == {
            "proj.E", "query_matrix.weight", "key_matrix.weight", "value_matrix.weight", "out.weight", "out.bias"
        }
        assert la.proj.E.shape == (MAX_SEQ, PROJ_K)
        assert not hasattr(la.proj, "F")

    def test_parameter_counts(self):
        per_head = NH * MAX_SEQ * PROJ_K
        shared = MAX_SEQ * PROJ_K
        assert make(sharing="none").proj.E.numel() + make(sharing="none").proj.F.numel() == 2 * per_head
        assert make(sharing="headwise").proj.E.numel() + make(sharing="headwise").proj.F.numel() == 2 * shared
        assert make(sharing="kv").proj.E.numel() == per_head
        assert make(sharing="layerwise").proj.E.numel() == shared

    def test_invalid_sharing_rejected(self):
        with pytest.raises(ValueError, match="sharing"):
            make(sharing="bogus")


class TestSeqLength:
    def test_shorter_key_slices_projection(self, tensors):
        q, _ = tensors
        la = make(max_seq_len=MAX_SEQ).eval()
        short = torch.randn(B, 4, IN)
        assert la(q, short, short).shape == (B, Q, OUT)

    def test_sequence_at_max_len_ok(self):
        la = make()
        x = torch.randn(B, MAX_SEQ, IN)
        assert la(x, x, x).shape == (B, MAX_SEQ, OUT)

    def test_over_max_len_raises(self):
        la = make()
        x = torch.randn(B, MAX_SEQ + 1, IN)
        with pytest.raises(ValueError, match="max_seq_len"):
            la(x, x, x)


class TestUnsupported:
    def test_mask_raises(self, tensors):
        q, kv = tensors
        mask = torch.zeros(1, 1, Q, K, dtype=torch.bool)
        with pytest.raises(NotImplementedError, match="bidirectional"):
            make()(q, kv, kv, mask=mask)

    def test_past_key_value_raises(self, tensors):
        q, kv = tensors
        past = (torch.randn(B, NH, 3, HS), torch.randn(B, NH, 3, HS))
        with pytest.raises(NotImplementedError, match="bidirectional"):
            make()(q, kv, kv, past_key_value=past)


class TestSharedProjection:
    def test_injected_projection_shared_across_layers(self):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")
        attn1 = make(projection=proj)
        attn2 = make(projection=proj)
        assert attn1.proj is attn2.proj is proj
        # шерится ТОЛЬКО проекция, Q/K/V/out линейки у каждого слоя свои
        assert attn1.query_matrix is not attn2.query_matrix
        assert attn1.key_matrix is not attn2.key_matrix

    def test_injected_projection_not_in_attention_state_dict(self):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")
        attn = make(projection=proj)
        assert "proj" not in attn.state_dict()

    def test_internal_projection_is_registered_child(self):
        attn = make()
        assert "proj.E" in attn.state_dict()
        assert isinstance(attn.proj, LinformerSeqProjection)

    def test_model_registers_shared_projection_once(self):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")
        attn1 = make(projection=proj)
        attn2 = make(projection=proj)
        model = nn.Module()
        model.proj = proj  # владелец регистрирует проекцию один раз
        model.attn1 = attn1
        model.attn2 = attn2
        sd = model.state_dict()
        # одна запись "proj.E" на все слои, Q/K/V у каждого слоя свои
        assert "proj.E" in sd
        assert "attn1.proj" not in sd and "attn2.proj" not in sd
        assert "attn1.query_matrix.weight" in sd and "attn2.query_matrix.weight" in sd
        assert torch.equal(sd["proj.E"], proj.E)

    def test_shared_projection_moves_device_together_with_owner(self):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")
        attn = make(projection=proj)
        model = nn.Module()
        model.proj = proj
        model.attn = attn
        model.to(torch.float64)
        assert proj.E.dtype == torch.float64
        assert attn.query_matrix.weight.dtype == torch.float64


class TestGradients:
    @pytest.mark.parametrize("sharing", ["none", "headwise", "kv", "layerwise"])
    def test_backward(self, tensors, sharing):
        q, kv = tensors
        la = make(sharing=sharing)
        qq = q.clone().requires_grad_(True)
        loss = la(qq, kv, kv).square().mean()
        loss.backward()
        assert qq.grad is not None
        assert all(p.grad is not None for p in la.parameters())

    def test_backward_updates_shared_projection(self, tensors):
        q, kv = tensors
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")
        attn1 = make(projection=proj)
        attn2 = make(projection=proj)
        proj.E.grad = None
        attn1(q, kv, kv).square().mean().backward()
        g1 = proj.E.grad.clone()
        proj.E.grad = None
        attn2(q, kv, kv).square().mean().backward()
        g2 = proj.E.grad.clone()
        assert not torch.equal(g1, g2)  # обе ветки копят в один параметр

    @CUDA
    def test_backward_cuda(self):
        la = make().cuda()
        q = torch.randn(B, Q, IN, device="cuda", requires_grad=True)
        kv = torch.randn(B, K, IN, device="cuda")
        la(q, kv, kv).square().mean().backward()
        assert q.grad is not None

    def test_fp16_no_nan(self, tensors):
        q, kv = tensors
        la = make().half().eval()
        with torch.no_grad():
            out = la(q.half(), kv.half(), kv.half())
            assert not torch.isnan(out).any()

    def test_torch_compile(self, tensors):
        q, kv = tensors
        mc = torch.compile(make().eval(), backend="eager")
        with torch.no_grad():
            assert mc(q, kv, kv).shape == (B, Q, OUT)


class TestInjectedIntoBlock:
    def test_attn_layer_replaces_default(self, tensors):
        _, kv = tensors
        attn = make()
        block = TransformerBlock(IN, HS, NH, OUT, 64, attn_layer=attn)
        assert block.attn is attn
        out, cache = block(kv)
        assert out.shape == (B, K, OUT)
        assert cache is None

    def test_shared_instance_ties_weights(self):
        attn = make()
        block1 = TransformerBlock(IN, HS, NH, OUT, 64, attn_layer=attn)
        block2 = TransformerBlock(IN, HS, NH, OUT, 64, attn_layer=attn)
        assert block1.attn is block2.attn is attn

    def test_layerwise_matches_manual_reference(self, tensors):
        _, kv = tensors
        attn = make(sharing="layerwise").eval()
        block = TransformerBlock(IN, HS, NH, OUT, 64, attn_layer=attn).eval()
        with torch.no_grad():
            out = block(kv)[0]
            h = block.norm1(kv)
            qq = attn.query_matrix(h).view(B, K, NH, HS).transpose(1, 2)
            kk = attn.key_matrix(h).view(B, K, NH, HS).transpose(1, 2)
            vv = attn.value_matrix(h).view(B, K, NH, HS).transpose(1, 2)
            kp = project_manual(kk, attn.proj.E[:K])
            vp = project_manual(vv, attn.proj.E[:K])
            w = torch.softmax(qq @ kp.transpose(-2, -1) / HS**0.5, dim=-1)
            attn_out = attn.out((w @ vp).transpose(1, 2).reshape(B, K, -1))
            ref = block.adapt_residual(kv) + attn_out
            ref = ref + block.ff(block.norm2(ref))
            assert close(out, ref)

    def test_blocks_share_projection_only(self):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")
        block1 = TransformerBlock(IN, HS, NH, OUT, 64, attn_layer=make(projection=proj))
        block2 = TransformerBlock(IN, HS, NH, OUT, 64, attn_layer=make(projection=proj))
        assert block1.attn.proj is block2.attn.proj is proj
        assert block1.attn.query_matrix is not block2.attn.query_matrix


class TestInjectedAsCrossAttn:
    def test_cross_attn_layer_replaces_default(self, tensors):
        q, kv = tensors
        attn = make(query_in_size=OUT)
        block = TransformerBlock(IN, HS, NH, OUT, 64, with_cross_attn=True, memory_size=IN, cross_attn_layer=attn)
        assert block.cross_attn.attn is attn
        out, _ = block(q, memory=kv)
        assert out.shape == (B, Q, OUT)

    def test_cross_memory_mask_raises(self, tensors):
        q, kv = tensors
        block = TransformerBlock(IN, HS, NH, OUT, 64, with_cross_attn=True, memory_size=IN, cross_attn_layer=make(query_in_size=OUT))
        mask = torch.zeros(1, 1, Q, K, dtype=torch.bool)
        with pytest.raises(NotImplementedError):
            block(q, memory=kv, memory_mask=mask)

    def test_cross_blocks_share_projection_only(self, tensors):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")
        block1 = TransformerBlock(IN, HS, NH, OUT, 64, with_cross_attn=True, memory_size=IN, cross_attn_layer=make(query_in_size=OUT, projection=proj))
        block2 = TransformerBlock(IN, HS, NH, OUT, 64, with_cross_attn=True, memory_size=IN, cross_attn_layer=make(query_in_size=OUT, projection=proj))
        assert block1.cross_attn.attn.proj is block2.cross_attn.attn.proj is proj

    def test_cross_blocks_register_shared_projection_once(self, tensors):
        proj = LinformerSeqProjection(NH, PROJ_K, MAX_SEQ, sharing="layerwise")

        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = proj
                self.b1 = TransformerBlock(IN, HS, NH, OUT, 64, with_cross_attn=True, memory_size=IN, cross_attn_layer=make(query_in_size=OUT, projection=proj))
                self.b2 = TransformerBlock(IN, HS, NH, OUT, 64, with_cross_attn=True, memory_size=IN, cross_attn_layer=make(query_in_size=OUT, projection=proj))

            def forward(self, x, memory):
                h, _ = self.b1(x, memory=memory)
                h, _ = self.b2(h, memory=memory)
                return h

        m = Shared()
        assert [k for k in m.state_dict() if "proj" in k] == ["proj.E"]
