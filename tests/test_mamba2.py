import pytest
import torch
import torch.nn.functional as F

from spartan_torch import Mamba2Block, Mamba2Mixer

B, L, D, N, W = 2, 10, 32, 8, 4
H, P, G, CS = 4, 16, 2, 8
E = 2 * D  # == H * P


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def make(**kwargs):
    kwargs.setdefault("d_state", N)
    kwargs.setdefault("d_conv", W)
    kwargs.setdefault("num_heads", H)
    kwargs.setdefault("head_dim", P)
    kwargs.setdefault("n_groups", G)
    kwargs.setdefault("chunk_size", CS)
    return Mamba2Mixer(D, **kwargs)


class TestMamba2Shapes:
    def test_prefill_shapes(self):
        torch.manual_seed(0)
        out, (conv_state, ssm_state) = make().eval()(torch.randn(B, L, D))
        assert out.shape == (B, L, D)
        assert conv_state.shape == (B, E + 2 * G * N, W)
        assert ssm_state.shape == (B, H, P, N)

    def test_short_sequence_and_unit_conv(self):
        torch.manual_seed(0)
        m = Mamba2Mixer(D, d_state=N, d_conv=1, num_heads=H, head_dim=P,
                        n_groups=G, chunk_size=CS).eval()
        out, (conv_state, _) = m(torch.randn(B, 3, D))
        assert out.shape == (B, 3, D)
        assert conv_state.shape == (B, E + 2 * G * N, 1)

    def test_bias_variants(self):
        torch.manual_seed(0)
        m = Mamba2Mixer(D, d_state=N, d_conv=W, num_heads=H, head_dim=P,
                        n_groups=G, chunk_size=CS, bias=True, conv_bias=False).eval()
        assert m(torch.randn(B, L, D))[0].shape == (B, L, D)

    def test_block_alias(self):
        assert Mamba2Block is Mamba2Mixer

    def test_head_constraint(self):
        with pytest.raises(ValueError, match="num_heads \\* head_dim"):
            Mamba2Mixer(D, num_heads=3, head_dim=P, n_groups=G)

    def test_meta_device_init(self):
        m = Mamba2Mixer(D, d_state=N, d_conv=W, num_heads=H, head_dim=P,
                        n_groups=G, device="meta")
        assert m.A_log.shape == (H,)


class TestMamba2Causality:
    def test_future_does_not_leak_into_past(self):
        torch.manual_seed(0)
        m = make().eval()
        x = torch.randn(B, L, D)
        with torch.no_grad():
            out = m(x)[0]
            x_future = x.clone()
            x_future[:, 1:] = 0.0
            assert close(out[:, 0], m(x_future)[0][:, 0])
            x_past = x.clone()
            x_past[:, 0] = 0.0
            assert not close(out[:, 1], m(x_past)[0][:, 1])

    def test_masked_suffix_matches_truncated(self):
        torch.manual_seed(0)
        m = make().eval()
        x = torch.randn(B, L, D)
        k = 6
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, k:] = True
        with torch.no_grad():
            out_masked = m(x, mask=mask)[0]
            out_trunc = m(x[:, :k])[0]
            assert close(out_masked[:, :k], out_trunc)


class TestMamba2Cache:
    def test_chunked_prefill_matches(self):
        torch.manual_seed(0)
        m = make().eval()
        x = torch.randn(B, L, D)
        with torch.no_grad():
            ref = m(x)[0]
            cache = m.init_cache(B)
            o1, cache = m(x[:, :4], cache)
            o2, _ = m(x[:, 4:], cache)
            assert close(torch.cat([o1, o2], dim=1), ref)

    def test_step_decoding_matches(self):
        torch.manual_seed(0)
        m = make().eval()
        x = torch.randn(B, L, D)
        with torch.no_grad():
            ref = m(x)[0]
            cache = m.init_cache(B)
            outs = []
            for t in range(L):
                o, cache = m(x[:, t : t + 1], cache)
                outs.append(o)
            assert close(torch.cat(outs, dim=1), ref)


class TestMamba2Backends:
    def test_fast_path_falls_back_without_kernels(self):
        try:
            import mamba_ssm  # noqa: F401
            pytest.skip("mamba-ssm installed — fallback path not exercised")
        except ImportError:
            pass
        torch.manual_seed(0)
        m = make(use_fast_path=True).eval()
        x = torch.randn(B, L, D)
        with torch.no_grad():
            assert m(x)[0].shape == (B, L, D)


class TestMamba2Gradients:
    def test_grad_flows_to_all_params(self):
        torch.manual_seed(0)
        m = make()
        x = torch.randn(B, L, D)
        m(x)[0].square().mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
        watched = {n for n, _ in m.named_parameters()}
        assert {"in_proj.weight", "conv1d.weight", "dt_bias", "A_log", "D",
                "norm.weight", "out_proj.weight"} <= watched


class TestMamba2Init:
    def test_dt_bias_in_range(self):
        torch.manual_seed(0)
        m = make()
        with torch.no_grad():
            dt = F.softplus(m.dt_bias)
        assert (dt >= 0.001).all() and (dt <= 0.1 + 1e-6).all()

    def test_state_matrix_stable(self):
        m = make()
        with torch.no_grad():
            A = -torch.exp(m.A_log.float())
        assert (A < 0).all()

    def test_chunk_size_free(self):
        torch.manual_seed(0)
        a = make(chunk_size=4).eval()
        b = make(chunk_size=16).eval()
        b.load_state_dict(a.state_dict())
        x = torch.randn(B, L, D)
        with torch.no_grad():
            assert close(a(x)[0], b(x)[0])
