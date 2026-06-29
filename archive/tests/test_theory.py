"""Ground-truth FHN theory: fixed points, Jacobian spectrum, derivatives."""
import numpy as np
import fhn_theory as TH
from dynamics import get_external_current

A, B, TAU = TH.A_DEFAULT, TH.B_DEFAULT, TH.TAU_DEFAULT

def fhn_rhs(v, w, u):
    return np.array([v - v ** 3 / 3 - w + u, (v + A - B * w) / TAU])

def test_fixed_points_are_roots():
    for u in [0.0, 0.3, 0.8, 1.2, 1.6]:
        for v in TH.fixed_points(u, A, B):
            w = (v + A) / B
            r = fhn_rhs(v, w, u)
            assert np.allclose(r, 0.0, atol=1e-6), (u, v, r)

def test_jacobian_eig_matches_numerical():
    """Analytic Jacobian eigenvalues equal the eigenvalues of a finite-diff Jacobian."""
    for u in [0.2, 0.5, 1.0, 1.5]:
        v = TH.fixed_points(u, A, B)[0]
        w = (v + A) / B
        h = 1e-6
        J = np.zeros((2, 2))
        for i, dx in enumerate([(h, 0), (0, h)]):
            fp = fhn_rhs(v + dx[0], w + dx[1], u)
            fm = fhn_rhs(v - dx[0], w - dx[1], u)
            J[:, i] = (fp - fm) / (2 * h)
        num = np.sort_complex(np.linalg.eigvals(J))
        ana = np.sort_complex(TH.jacobian_eigs_at(v, A, B, TAU))
        assert np.allclose(num, ana, atol=1e-4), (u, num, ana)

def test_hopf_bifurcation_present():
    """sigma(u) must change sign (a Hopf-type instability is crossed)."""
    sp = TH.spectrum_over_u(np.linspace(0.0, 1.6, 33), A, B, TAU)
    s = sp["sigma"][np.isfinite(sp["sigma"])]
    assert s.min() < 0 < s.max()

def test_limit_cycle_period_in_range():
    """The oscillatory regime has a period of ~37 (matches the literature value)."""
    per, amp = TH.limit_cycle_period(1.0, A, B, TAU, t_max=400.0, dt=0.01)
    assert 30.0 < per < 45.0
    assert amp > 2.0

def test_no_oscillation_when_stable():
    per, _ = TH.limit_cycle_period(0.0, A, B, TAU, t_max=400.0, dt=0.01)
    assert not np.isfinite(per)

def test_external_current_shapes():
    import jax.numpy as jnp
    t = jnp.linspace(0, 80, 100)
    for kind in ["constant", "sine", "chirp", "step", "pulse"]:
        u = jnp.broadcast_to(get_external_current(t, kind, 0.8), t.shape)
        assert u.shape == t.shape and bool(jnp.all(jnp.isfinite(u)))
