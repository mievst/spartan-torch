import math

import pytest
import torch
import torch.nn.functional as F

from spartan_torch import MambaBlock, MambaMixer

B, L, D, N, W = 2, 9, 16, 8, 4
E = 2 * D


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def make(**kwargs):
    kwargs.setdefault("d_state", N)
    kwargs.setdefault("d_conv", W)
    return MambaMixer(D, **kwargs)


class TestMambaShapes:
    def test_prefill_shapes(self):
        torch.manual_seed(0)
        out, (conv_state, ssm_state) = make().eval()(torch.randn(B, L, D))
        assert out.shape == (B, L, D)
        assert conv_state.shape == (B, E, W)
        assert ssm_state.shape == (B, E, N)

    def test_short_sequence_and_unit_conv(self):
        torch.manual_seed(0)
        m = MambaMixer(D, d_state=N, d_conv=1).eval()
        out, (conv_state, _) = m(torch.randn(B, 2, D))
        assert out.shape == (B, 2, D)
        assert conv_state.shape == (B, E, 1)

    def test_bias_variants(self):
        torch.manual_seed(0)
        m = MambaMixer(D, d_state=N, d_conv=W, bias=True, conv_bias=False).eval()
        assert m(torch.randn(B, L, D))[0].shape == (B, L, D)

    def test_block_alias(self):
        assert MambaBlock is MambaMixer

    def test_meta_device_init(self):
        m = MambaMixer(D, d_state=N, d_conv=W, device="meta")
        assert m.A_log.shape == (E, N)


class TestMambaCausality:
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
        k = 5
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, k:] = True
        with torch.no_grad():
            out_masked = m(x, mask=mask)[0]
            out_trunc = m(x[:, :k])[0]
            assert close(out_masked[:, :k], out_trunc)


class TestMambaCache:
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

    def test_single_token_cache_shapes(self):
        torch.manual_seed(0)
        m = make().eval()
        cache = m.init_cache(B)
        with torch.no_grad():
            o, (cs, ss) = m(torch.randn(B, 1, D), cache)
        assert o.shape == (B, 1, D)
        assert cs.shape == (B, E, W)
        assert ss.shape == (B, E, N)


class TestMambaBackends:
    def test_loop_matches_associative(self):
        torch.manual_seed(0)
        m_assoc = make().eval()
        m_loop = make(use_associative_scan=False).eval()
        m_loop.load_state_dict(m_assoc.state_dict())
        x = torch.randn(B, L, D)
        with torch.no_grad():
            assert close(m_assoc(x)[0], m_loop(x)[0])

    def test_loop_matches_associative_fp64(self):
        torch.manual_seed(0)
        m_assoc = make().eval().double()
        m_loop = make(use_associative_scan=False).eval().double()
        m_loop.load_state_dict(m_assoc.state_dict())
        x = torch.randn(B, L, D, dtype=torch.float64)
        with torch.no_grad():
            diff = (m_assoc(x)[0] - m_loop(x)[0]).abs().max().item()
        assert diff < 1e-10

    def test_fast_path_falls_back_without_kernels(self):
        pytest.importorskip("torch")  # always present; documents intent
        try:
            import mamba_ssm  # noqa: F401
            pytest.skip("mamba-ssm installed — fallback path not exercised")
        except ImportError:
            pass
        torch.manual_seed(0)
        m_fast = make(use_fast_path=True).eval()
        m_slow = make(use_fast_path=False, use_associative_scan=False).eval()
        m_slow.load_state_dict(m_fast.state_dict())
        x = torch.randn(B, L, D)
        with torch.no_grad():
            assert close(m_fast(x)[0], m_slow(x)[0])


class TestMambaGradients:
    def test_grad_flows_to_all_params(self):
        torch.manual_seed(0)
        m = make()
        x = torch.randn(B, L, D)
        m(x)[0].square().mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
        watched = {n for n, _ in m.named_parameters()}
        assert {"in_proj.weight", "conv1d.weight", "x_proj.weight", "dt_proj.weight",
                "dt_proj.bias", "A_log", "D", "out_proj.weight"} <= watched

    def test_grad_through_chunked_matches_prefill(self):
        torch.manual_seed(0)
        m1, m2 = make(), make()
        m2.load_state_dict(m1.state_dict())
        x = torch.randn(B, L, D)
        m1(x)[0].square().mean().backward()
        cache = m2.init_cache(B)
        o1, cache = m2(x[:, :4], cache)
        o2, _ = m2(x[:, 4:], cache)
        torch.cat([o1, o2], dim=1).square().mean().backward()
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert n1 == n2
            assert close(p1.grad, p2.grad, tol=1e-4), n1


class TestMambaInit:
    def test_dt_bias_in_range(self):
        torch.manual_seed(0)
        m = make(dt_min=0.001, dt_max=0.1)
        with torch.no_grad():
            dt = F.softplus(m.dt_proj.bias)
        assert (dt >= 0.001).all() and (dt <= 0.1 + 1e-6).all()

    def test_state_matrix_stable(self):
        m = make()
        with torch.no_grad():
            A = -torch.exp(m.A_log.float())
        assert (A < 0).all()

    def test_dt_rank_auto(self):
        assert make().dt_rank == math.ceil(D / 16)
        assert MambaMixer(64).dt_rank == 4
        assert MambaMixer(32, dt_rank=7).dt_rank == 7

    def test_skip_connection_scale(self):
        m = make()
        assert m.D.shape == (E,)
        with torch.no_grad():
            assert close(m.D, torch.ones(E))
