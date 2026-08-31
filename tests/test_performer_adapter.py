import math

import pytest
import torch
from torch import nn

from spartan_torch import PerformerAdapter, performerize_attentions

B, N, IN, OUT, HS, NH = 2, 32, 64, 64, 16, 4

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def make_attention(family="mha", kv_heads=None):
    att = nn.Module()
    kv_out = NH * HS if kv_heads is None else kv_heads * HS
    if family == "hf":
        att.q_proj = nn.Linear(IN, NH * HS)
        att.k_proj = nn.Linear(IN, kv_out)
        att.v_proj = nn.Linear(IN, kv_out)
        att.o_proj = nn.Linear(NH * HS, OUT)
    elif family == "bert":
        att.query = nn.Linear(IN, NH * HS)
        att.key = nn.Linear(IN, kv_out)
        att.value = nn.Linear(IN, kv_out)
        att.output = nn.Module()
        att.output.dense = nn.Linear(NH * HS, OUT)
    else:
        att.query_matrix = nn.Linear(IN, NH * HS)
        att.key_matrix = nn.Linear(IN, kv_out)
        att.value_matrix = nn.Linear(IN, kv_out)
        att.out = nn.Linear(NH * HS, OUT)
    return att


def resolve(module, path):
    for part in path.split("."):
        module = getattr(module, part)
    return module


def project(att, family, x):
    if family == "hf":
        q, k, v, o = att.q_proj(x), att.k_proj(x), att.v_proj(x), att.o_proj
    elif family == "bert":
        q, k, v, o = att.query(x), att.key(x), att.value(x), att.output.dense
    else:
        q, k, v, o = att.query_matrix(x), att.key_matrix(x), att.value_matrix(x), att.out
    return q, k, v, o


def exact_reference(att, family, x, is_causal):
    with torch.no_grad():
        q, k, v, o = project(att, family, x)
        qh = q.view(B, N, NH, HS).transpose(1, 2)
        kh = k.view(B, N, NH, HS).transpose(1, 2)
        vh = v.view(B, N, NH, HS).transpose(1, 2)
        scores = qh @ kh.transpose(-2, -1) / HS**0.5
        if is_causal:
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        context = torch.softmax(scores, dim=-1) @ vh
        return o(context.transpose(1, 2).reshape(B, N, NH * HS))


@pytest.fixture()
def tensors():
    torch.manual_seed(0)
    return torch.randn(B, N, IN) * 0.5


def make_adapter(att, **kwargs):
    kw = dict(num_features=4096)
    kw.update(kwargs)
    return PerformerAdapter.from_module(att, head_size=HS, **kw)


class TestPerformerAdapter:
    def test_converges_to_exact_spartan_names(self, tensors):
        x = tensors
        att = make_attention()
        adapter = make_adapter(att).eval()
        assert close(adapter(x, x, x), exact_reference(att, "mha", x, False), tol=2e-2)

    def test_converges_to_exact_hf_names(self, tensors):
        x = tensors
        att = make_attention("hf")
        adapter = make_adapter(att).eval()
        assert close(adapter(x, x, x), exact_reference(att, "hf", x, False), tol=2e-2)

    def test_converges_to_exact_bert_names(self, tensors):
        x = tensors
        att = make_attention("bert")
        adapter = make_adapter(att).eval()
        assert close(adapter(x, x, x), exact_reference(att, "bert", x, False), tol=2e-2)

    def test_converges_to_exact_causal(self, tensors):
        x = tensors
        att = make_attention()
        adapter = make_adapter(att, is_causal=True).eval()
        assert close(adapter(x, x, x), exact_reference(att, "mha", x, True), tol=2e-2)

    def test_weights_owned_once(self, tensors):
        x = tensors
        att = make_attention()
        n_orig = sum(p.numel() for p in att.parameters())
        adapter = make_adapter(att)
        n_adapt = sum(p.numel() for p in adapter.parameters())
        sd = adapter.state_dict()
        assert n_adapt == n_orig  # adapter adds no trainable params
        keys = [k for k in sd if not k.startswith("R")]
        # every projection weight appears exactly once, under the attention child
        assert len(keys) == len(list(att.named_parameters())) == 8
        assert all(k.startswith("attention.") for k in keys)

    def test_freeze_defaults_on(self, tensors):
        x = tensors
        att = make_attention()
        make_adapter(att)
        assert all(p.requires_grad is False for p in att.parameters())

    def test_freeze_false_keeps_trainable(self, tensors):
        x = tensors
        att = make_attention()
        make_adapter(att, freeze=False)
        assert all(p.requires_grad for p in att.parameters())

    def test_R_is_buffer_and_adapter_has_no_trainables(self, tensors):
        x = tensors
        adapter = make_adapter(make_attention())
        assert "R" in adapter.state_dict()
        assert not any(p.requires_grad for p in adapter.parameters())

    def test_probe_raises_for_unidentifiable_attention(self):
        with pytest.raises(ValueError, match="projection"):
            make_adapter(nn.Module())

    def test_gqa_raises(self):
        att = make_attention(kv_heads=NH // 2)
        with pytest.raises(ValueError, match="key projection out_features"):
            make_adapter(att)

    def test_qkv_fn_fused_projection(self, tensors):
        x = tensors
        att = make_attention()
        c_attn = nn.Linear(IN, 3 * NH * HS)
        with torch.no_grad():
            c_attn.weight.copy_(torch.cat([att.query_matrix.weight, att.key_matrix.weight, att.value_matrix.weight], dim=0))
            c_attn.bias.copy_(torch.cat([att.query_matrix.bias, att.key_matrix.bias, att.value_matrix.bias]))

        def qkv_fn(x):
            out = c_attn(x)
            return out.split(NH * HS, dim=-1)

        adapter = PerformerAdapter(
            att, num_heads=NH, head_size=HS, out_size=OUT, num_features=4096, qkv_fn=qkv_fn
        ).eval()
        assert close(adapter(x, x, x), exact_reference(att, "mha", x, False), tol=2e-2)

    def test_qk_mod_called(self, tensors):
        x = tensors
        calls = {}

        def qk_mod(q, k, q_positions, k_positions):
            calls["q_pos"] = q_positions
            calls["k_pos"] = k_positions
            calls["shapes"] = (q.shape, k.shape)
            return q, k

        adapter = make_adapter(make_attention(), qk_mod=qk_mod)
        adapter(x, x, x)
        assert calls["shapes"] == ((B, NH, N, HS), (B, NH, N, HS))
        assert calls["q_pos"].tolist() == list(range(N))
        assert calls["k_pos"].tolist() == list(range(N))

    def test_key_mask_equals_truncated(self, tensors):
        x = tensors
        att = make_attention()
        adapter = make_adapter(att).eval()
        mask = torch.zeros(B, 1, 1, N, dtype=torch.bool)
        mask[:, :, :, N // 2 :] = True
        with torch.no_grad():
            masked = adapter(x, x, x, mask=mask)
            xp = torch.cat([x[:, : N // 2], torch.zeros_like(x[:, : N // 2])], dim=1)
            truncated = adapter(xp, xp, xp, mask=mask)
        assert close(masked[:, : N // 2], truncated[:, : N // 2], tol=1e-4)

    def test_pair_mask_and_cache_raise(self, tensors):
        x = tensors
        adapter = make_adapter(make_attention())
        with pytest.raises(NotImplementedError):
            adapter(x, x, x, mask=torch.zeros(B, 1, 1, N, dtype=torch.float32))
        with pytest.raises(NotImplementedError):
            adapter(x, x, x, past_key_value=(torch.randn(B, N, IN), torch.randn(B, N, IN)))

    def test_causal_length_mismatch_raises(self, tensors):
        x = tensors
        adapter = make_adapter(make_attention(), is_causal=True)
        with pytest.raises(ValueError, match="causal"):
            adapter(x, x[:, : N - 1, :], x)

    def test_state_dict_roundtrip(self, tensors):
        x = tensors
        adapter = make_adapter(make_attention())
        adapter2 = make_adapter(make_attention())
        adapter2.load_state_dict(adapter.state_dict())
        assert torch.equal(adapter(x, x, x), adapter2(x, x, x))

    @CUDA
    def test_fp16(self, tensors):
        x = tensors.cuda().half()
        adapter = make_adapter(make_attention(), num_features=16384, is_causal=True).cuda().half().eval()
        assert torch.isfinite(adapter(x, x, x)).all()

    def test_compile(self, tensors):
        x = tensors
        adapter = torch.compile(make_adapter(make_attention(), is_causal=True).eval(), backend="eager")
        with torch.no_grad():
            assert adapter(x, x, x).shape == (B, N, OUT)


class TestPerformerizeAttentions:
    def test_replaces_all_attentions(self, tensors):
        x = tensors
        model = nn.Module()
        model.block1 = nn.Module()
        model.block1.attn = make_attention()
        model.block1.other = nn.Linear(IN, IN)
        model.block2 = make_attention()
        replaced = performerize_attentions(model, head_size=HS)
        assert len(replaced) == 2
        assert isinstance(model.block1.attn, PerformerAdapter)
        assert isinstance(model.block2, PerformerAdapter)
        # adapter wraps the original module as its child
        assert isinstance(model.block1.attn.attention, nn.Module)

    def test_nested_attention_replaced_once(self, tensors):
        x = tensors
        model = nn.Module()
        model.outer = nn.Module()
        model.outer.attn = make_attention()  # direct child
        model.outer.nested = nn.Module()
        model.outer.nested.inner = make_attention()  # deeper, no projections on the way
        replaced = performerize_attentions(model, head_size=HS)
        assert len(replaced) == 2
        # no adapter double-wrapped: each adapter's child is the raw attention
        assert not isinstance(model.outer.attn.attention, PerformerAdapter)
        assert not isinstance(model.outer.nested.inner.attention, PerformerAdapter)

    def test_adapters_match_exact_after_replacement(self, tensors):
        x = tensors
        model = nn.Module()
        model.attn = make_attention()
        performerize_attentions(model, head_size=HS)
        adapter = model.attn
        assert close(adapter(x, x, x), exact_reference(adapter.attention, "mha", x, False), tol=2e-2)

    def test_geometry_mismatch_propagates(self):
        model = nn.Module()
        model.attn = make_attention(kv_heads=NH // 2)  # GQA
        with pytest.raises(ValueError, match="key projection out_features"):
            performerize_attentions(model, head_size=HS)
