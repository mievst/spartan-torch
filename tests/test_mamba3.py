import pytest
import torch
import torch.nn.functional as F

from spartan_torch import Mamba3Block, Mamba3Mixer

B, L, D, N = 2, 9, 32, 8
H, P, R = 4, 16, 2


def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def make_siso(**kwargs):
    kwargs.setdefault("d_state", N)
    kwargs.setdefault("head_dim", P)
    kwargs.setdefault("n_groups", 1)
    return Mamba3Mixer(D, **kwargs)


def make_mimo(**kwargs):
    kwargs.setdefault("d_state", N)
    kwargs.setdefault("head_dim", P)
    kwargs.setdefault("n_groups", 1)
    kwargs.setdefault("is_mimo", True)
    kwargs.setdefault("mimo_rank", R)
    return Mamba3Mixer(D, **kwargs)


class TestMamba3Shapes:
    def test_siso_prefill_shapes(self):
        torch.manual_seed(0)
        out, (ang, ss, bx) = make_siso().eval()(torch.randn(B, L, D))
        assert out.shape == (B, L, D)
        assert ang.shape == (B, H, N // 4)
        assert ss.shape == (B, H, P, N)
        assert bx.shape == (B, H, P, N)

    def test_mimo_prefill_shapes(self):
        torch.manual_seed(0)
        out, (ang, ss, bx) = make_mimo().eval()(torch.randn(B, L, D))
        assert out.shape == (B, L, D)
        assert ang.shape == (B, H, N // 4)
        assert ss.shape == (B, H, P, N)
        assert bx.shape == (B, R, H, P, N)

    def test_rope_fraction_full(self):
        torch.manual_seed(0)
        m = Mamba3Mixer(D, d_state=N, head_dim=P, rope_fraction=1.0).eval()
        out, (ang, _, _) = m(torch.randn(B, L, D))
        assert out.shape == (B, L, D)
        assert ang.shape == (B, H, N // 2)

    def test_outproj_norm_shapes(self):
        torch.manual_seed(0)
        m = make_mimo(is_outproj_norm=True).eval()
        assert m(torch.randn(B, L, D))[0].shape == (B, L, D)

    def test_block_alias(self):
        assert Mamba3Block is Mamba3Mixer

    def test_bad_configs(self):
        with pytest.raises(ValueError, match="head_dim"):
            Mamba3Mixer(D, head_dim=7, d_state=N)
        with pytest.raises(ValueError, match="rope_fraction"):
            Mamba3Mixer(D, d_state=N, head_dim=P, rope_fraction=0.7)

    def test_meta_device_init(self):
        m = Mamba3Mixer(D, d_state=N, head_dim=P, device="meta")
        assert m.dt_bias.shape == (H,)
        m2 = Mamba3Mixer(D, d_state=N, head_dim=P, is_mimo=True, mimo_rank=R, device="meta")
        assert m2.mimo_x.shape == (H, R, P)


class TestMamba3Causality:
    @pytest.mark.parametrize("factory", [make_siso, make_mimo])
    def test_future_does_not_leak_into_past(self, factory):
        torch.manual_seed(0)
        m = factory().eval()
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
        m = make_siso().eval()
        x = torch.randn(B, L, D)
        k = 5
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, k:] = True
        with torch.no_grad():
            out_masked = m(x, mask=mask)[0]
            out_trunc = m(x[:, :k])[0]
            assert close(out_masked[:, :k], out_trunc)


class TestMamba3Cache:
    @pytest.mark.parametrize("factory", [make_siso, make_mimo])
    def test_step_decoding_matches(self, factory):
        torch.manual_seed(0)
        m = factory().eval()
        x = torch.randn(B, L, D)
        with torch.no_grad():
            ref = m(x)[0]
            cache = m.init_cache(B)
            outs = []
            for t in range(L):
                o, cache = m(x[:, t : t + 1], cache)
                outs.append(o)
            assert close(torch.cat(outs, dim=1), ref)

    @pytest.mark.parametrize("factory", [make_siso, make_mimo])
    def test_chunked_prefill_matches(self, factory):
        torch.manual_seed(0)
        m = factory().eval()
        x = torch.randn(B, L, D)
        with torch.no_grad():
            ref = m(x)[0]
            cache = m.init_cache(B)
            o1, cache = m(x[:, :4], cache)
            o2, _ = m(x[:, 4:], cache)
            assert close(torch.cat([o1, o2], dim=1), ref)


class TestMamba3Gradients:
    @pytest.mark.parametrize("factory", [make_siso, make_mimo])
    def test_grad_flows_to_all_params(self, factory):
        torch.manual_seed(0)
        m = factory()
        x = torch.randn(B, L, D)
        m(x)[0].square().mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
        watched = {n for n, _ in m.named_parameters()}
        assert {"in_proj.weight", "dt_bias", "B_bias", "C_bias", "B_norm.weight",
                "C_norm.weight", "D", "out_proj.weight"} <= watched

    def test_grad_finite_through_steps(self):
        # Cache states are detached (no graph leaks across generate calls),
        # so step-wise grads are truncated BPTT — assert finiteness, not
        # equality with the prefill grads.
        torch.manual_seed(0)
        m = make_siso()
        x = torch.randn(B, L, D)
        cache = m.init_cache(B)
        outs = []
        for t in range(L):
            o, cache = m(x[:, t : t + 1], cache)
            outs.append(o)
        torch.cat(outs, dim=1).square().mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name


class TestMamba3Init:
    def test_dt_bias_in_range(self):
        torch.manual_seed(0)
        m = make_siso()
        with torch.no_grad():
            dt = F.softplus(m.dt_bias)
        assert (dt >= 0.001).all() and (dt <= 0.1 + 1e-6).all()

    def test_heavy_tail_activation(self):
        # A = -heavy_tail(dd_A) <= -a_floor by construction; check the helper.
        from spartan_torch.transformers.ssm.mamba3 import _heavy_tail
        probe = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        assert (_heavy_tail(probe) > 0).all()
        assert torch.isclose(_heavy_tail(torch.tensor(0.0)), torch.tensor(1.0))
        assert torch.isclose(_heavy_tail(torch.tensor(2.0)), torch.tensor(3.0))
        assert torch.isclose(_heavy_tail(torch.tensor(-1.0)), torch.tensor(0.5))

    def test_bc_bias_ones_and_mimo_scale(self):
        m = make_mimo()
        with torch.no_grad():
            assert close(m.B_bias, torch.ones(H, R, N))
            assert close(m.C_bias, torch.ones(H, R, N))
            assert close(m.mimo_x, torch.ones(H, R, P) / R)
            assert close(m.mimo_o, torch.ones(H, R, P) / R)
            assert close(m.mimo_z, torch.ones(H, R, P))

    def test_scan_runs_finite(self):
        torch.manual_seed(0)
        m = make_siso().eval()
        x = torch.randn(1, 4, D)
        with torch.no_grad():
            assert torch.isfinite(m(x)[0]).all()


class TestTrapezoidalBlend:
    def _inputs(self):
        # One head, P = N = 1, L = 2, unit input/B/C, no decay, dt = 1, D = 0.
        x = torch.ones(1, 2, 1, 1)
        B = torch.ones(1, 2, 1, 1)
        C = torch.ones(1, 2, 1, 1)
        decay = torch.ones(1, 2, 1)
        dt = torch.ones(1, 2, 1)
        D = torch.zeros(1)
        ssm = torch.zeros(1, 1, 1, 1)
        bx = torch.zeros(1, 1, 1, 1)
        return x, B, C, decay, dt, D, ssm, bx

    def test_euler_extreme(self):
        # tr = 0: h_t = h_{t-1} + Bx_t → y = [1, 2].
        from spartan_torch.transformers.ssm.mamba3 import _trapezoidal_scan_siso
        x, B, C, decay, dt, D, ssm, bx = self._inputs()
        tr = torch.zeros(1, 2, 1)
        with torch.no_grad():
            y, _, _ = _trapezoidal_scan_siso(x, B, C, decay, dt, tr, D, ssm, bx)
        assert close(y.reshape(-1), torch.tensor([1.0, 2.0]))

    def test_trapezoid_extreme(self):
        # tr = 1: h_t = h_{t-1} + (Bx_t + Bx_{t-1})/2 → y = [0.5, 1.5].
        from spartan_torch.transformers.ssm.mamba3 import _trapezoidal_scan_siso
        x, B, C, decay, dt, D, ssm, bx = self._inputs()
        tr = torch.ones(1, 2, 1)
        with torch.no_grad():
            y, _, _ = _trapezoidal_scan_siso(x, B, C, decay, dt, tr, D, ssm, bx)
        assert close(y.reshape(-1), torch.tensor([0.5, 1.5]))


class TestDataDependentRope:
    def test_zero_angle_is_identity(self):
        from spartan_torch.transformers.ssm.mamba3 import _apply_rope_adjacent, _apply_rope_pairwise
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, 8)
        zero = torch.zeros(2, 3, 4, 4)
        with torch.no_grad():
            assert close(_apply_rope_adjacent(x, zero), x)
            assert close(_apply_rope_pairwise(x, zero), x)

    def test_rotation_preserves_norm(self):
        from spartan_torch.transformers.ssm.mamba3 import _apply_rope_adjacent, _apply_rope_pairwise
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, 8)
        ang = torch.randn(2, 3, 4, 4) * 2.0
        with torch.no_grad():
            for fn in (_apply_rope_adjacent, _apply_rope_pairwise):
                y = fn(x, ang)
                assert close(y.pow(2).sum(-1), x.pow(2).sum(-1))

    def test_pairings_differ(self):
        from spartan_torch.transformers.ssm.mamba3 import _apply_rope_adjacent, _apply_rope_pairwise
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, 8)
        ang = torch.randn(2, 3, 4, 4)
        with torch.no_grad():
            a = _apply_rope_adjacent(x, ang)
            b = _apply_rope_pairwise(x, ang)
            assert not close(a, b)
            # Adjacent rotation only mixes (2i, 2i+1) pairs: an angle that is
            # zero on pair 0 leaves dims 0-1 untouched.
            ang0 = ang.clone()
            ang0[..., 0] = 0.0
            assert close(_apply_rope_adjacent(x, ang0)[..., 0:2], x[..., 0:2])
