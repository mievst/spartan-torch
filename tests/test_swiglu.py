import pytest
import torch
from torch import nn
import torch.nn.functional as F

from spartan_torch import SwiGLUFeedForward, TransformerBlock


def reference_swiglu(ff, x):
    return ff.down_proj(F.silu(ff.gate_proj(x)) * ff.up_proj(x))


@pytest.fixture
def eval_swiglu():
    return SwiGLUFeedForward(64, 256).eval()


class TestSwiGLUFeedForward:
    def test_shapes(self):
        ff = SwiGLUFeedForward(48, 192)
        x = torch.randn(2, 10, 48)
        assert ff(x).shape == (2, 10, 48)

    def test_matches_reference(self, eval_swiglu):
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            assert torch.allclose(eval_swiglu(x), reference_swiglu(eval_swiglu, x), atol=1e-6)

    def test_no_bias_by_default(self):
        ff = SwiGLUFeedForward(48, 192)
        assert all(w.bias is None for w in (ff.gate_proj, ff.up_proj, ff.down_proj))

    def test_optional_bias(self):
        ff = SwiGLUFeedForward(48, 192, bias=True)
        assert all(w.bias is not None for w in (ff.gate_proj, ff.up_proj, ff.down_proj))

    def test_state_dict_keys(self):
        assert set(SwiGLUFeedForward(48, 192).state_dict()) == {
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        }

    def test_backward_produces_gradients(self, eval_swiglu):
        x = torch.randn(2, 10, 64, requires_grad=True)
        y = eval_swiglu(x).sum()
        y.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in eval_swiglu.parameters())

    def test_inside_transformer_block(self):
        block = TransformerBlock(
            64, 16, 4, 64, 256, ff_layer=SwiGLUFeedForward(64, 256)
        )
        x = torch.randn(2, 10, 64)
        out, cache = block(x)
        assert out.shape == (2, 10, 64)
        assert cache is not None
