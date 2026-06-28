"""Architecture & training-mechanics tests for the shared Koopman model."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import model as M

KEY = jax.random.PRNGKey(0)

def make(backbone, m=4):
    cfg = M.ModelConfig(backbone=backbone, m=m)
    return cfg, M.init_model(cfg, KEY)

@pytest.mark.parametrize("backbone", ["spiral", "bilinear"])
def test_io_shapes(backbone):
    cfg, p = make(backbone)
    x = jax.random.normal(KEY, (11, 2))
    u = jnp.linspace(0.2, 1.8, 11)
    z = M.encode(p, cfg, x)
    assert z.shape == (11, cfg.latent)
    assert M.decode(p, cfg, z).shape == (11, 2)
    assert M.step(p, cfg, z, u, 0.05).shape == (11, cfg.latent)
    assert M.generator(p, cfg, z, u).shape == (11, cfg.latent)

def test_bilinear_reconstruction_is_exact():
    """State-in-latent + fixed linear decoder => exact reconstruction (fix #8)."""
    cfg, p = make("bilinear")
    x = jax.random.normal(KEY, (20, 2)) * 2.0
    x_rec = M.decode(p, cfg, M.encode(p, cfg, x))
    assert float(jnp.max(jnp.abs(x - x_rec))) < 1e-6

def test_bilinear_is_contractive_for_nonneg_u():
    """Re eig(A+uB) <= 0 for all u >= 0 by construction (fix #2)."""
    cfg, p = make("bilinear")
    A, B = M.bilinear_matrices(p)
    for u in [0.0, 0.5, 1.0, 1.8, 3.0]:
        re = np.linalg.eigvals(np.array(A) + u * np.array(B)).real
        assert re.max() < 1e-5, (u, re.max())

def test_spiral_sl_effective_sigma_is_negative_far_outside():
    """Stuart-Landau absorbing ball: at large radius sigma_eff = sigma0 - beta r^2 is strictly negative for every mode, so trajectories cannot escape (the..."""
    cfg, p = make("spiral")
    z = jax.random.normal(KEY, (50, cfg.latent)) * 30.0
    u = jax.random.uniform(KEY, (50,), minval=0.0, maxval=2.0)
    s, o = M.spiral_eigs(p, cfg, z, u)
    assert float(jnp.max(s)) < 0.0

def test_spiral_clamped_sigma_nonpositive():
    """Legacy radial law keeps sigma = -softplus(.) <= 0 (back-compat / ablation)."""
    cfg = M.ModelConfig(backbone="spiral", m=4, radial="clamped")
    p = M.init_model(cfg, KEY)
    z = jax.random.normal(KEY, (50, cfg.latent)) * 3.0
    u = jax.random.uniform(KEY, (50,), minval=0.0, maxval=2.0)
    s, o = M.spiral_eigs(p, cfg, z, u)
    assert float(jnp.max(s)) <= 0.0

@pytest.mark.parametrize("backbone", ["spiral", "bilinear"])
def test_long_rollout_is_bounded(backbone):
    """A long rollout must stay finite and inside an absorbing ball (no blow-up)."""
    cfg, p = make(backbone)
    z0 = M.encode(p, cfg, jax.random.normal(KEY, (8, 2)))
    useq = jnp.broadcast_to(jnp.linspace(0.2, 1.8, 1500), (8, 1500))
    roll = M.rollout(p, cfg, z0, useq, 0.05)
    assert bool(jnp.all(jnp.isfinite(roll)))
    n0 = float(jnp.max(jnp.linalg.norm(z0, axis=-1)))
    nmax = float(jnp.max(jnp.linalg.norm(roll, axis=-1)))
    if backbone == "bilinear":
        assert nmax <= n0 + 1e-3
    else:
        assert nmax <= 50.0 * (n0 + 1.0)

def test_spiral_sl_is_attracting_from_inside_and_outside():
    """The defining limit-cycle property the legacy clamped law lacked: a nearly collapsed orbit GROWS and a huge orbit SHRINKS, both onto a finite nonzer..."""
    cfg, p = make("spiral", m=4)
    dt, N = 0.05, 6000
    u = jnp.broadcast_to(jnp.array([1.0])[:, None], (1, N))

    def final_norm(scale):
        z0 = jax.random.normal(KEY, (1, cfg.latent)) * scale
        roll = M.rollout(p, cfg, z0, u, dt)
        return float(jnp.linalg.norm(roll[0, -1]))

    n_tiny, n_huge = final_norm(1e-3), final_norm(40.0)
    assert jnp.isfinite(jnp.array([n_tiny, n_huge])).all()
    assert n_tiny > 1e-2
    assert n_huge < 40.0
    assert abs(n_tiny - n_huge) < 0.5 * (n_tiny + n_huge)

def test_spiral_rollout_is_differentiable_in_control():
    """Inverse design needs gradients of the rollout w.r.t."""
    cfg, p = make("spiral", m=4)
    def final_v(uval):
        z0 = M.encode(p, cfg, jnp.array([[-1.0, -0.5]]))
        useq = jnp.broadcast_to(jnp.asarray(uval)[None, None], (1, 150))
        roll = M.rollout(p, cfg, z0, useq, 0.05)
        return M.decode(p, cfg, roll[:, -1])[0, 0]
    g = float(jax.grad(final_v)(0.8))
    assert jnp.isfinite(jnp.array(g)) and abs(g) > 1e-6

def test_spiral_radius_conditioning_changes_eigs():
    """Operator conditioning (fix #1): eigenvalues must vary with latent radius."""
    cfg, p = make("spiral")
    r = jnp.array([0.05, 1.0, 3.0])
    u = jnp.array([1.0, 1.0, 1.0])
    s, o = M.spiral_eig_grid(p, cfg, r, u)
    assert float(jnp.std(s[:, 0])) > 1e-4 or float(jnp.std(o[:, 0])) > 1e-4

def test_spiral_eigs_depend_on_control():
    cfg, p = make("spiral")
    z = jax.random.normal(KEY, (1, cfg.latent))
    s0, o0 = M.spiral_eigs(p, cfg, z, jnp.array([0.2]))
    s1, o1 = M.spiral_eigs(p, cfg, z, jnp.array([1.8]))
    assert float(jnp.sum(jnp.abs(s0 - s1)) + jnp.sum(jnp.abs(o0 - o1))) > 1e-4

@pytest.mark.parametrize("backbone", ["spiral", "bilinear"])
def test_discrete_matches_continuous_small_dt(backbone):
    """(z_next - z)/dt -> generator(z,u) as dt -> 0 (operator consistency)."""
    cfg, p = make(backbone)
    z = M.encode(p, cfg, jax.random.normal(KEY, (6, 2)))
    u = jnp.linspace(0.3, 1.5, 6)
    dt = 1e-4
    fd = (M.step(p, cfg, z, u, dt) - z) / dt
    gen = M.generator(p, cfg, z, u)
    assert float(jnp.max(jnp.abs(fd - gen))) < 1e-2

def test_encoder_jvp_matches_finite_difference():
    """The Sobolev term relies on jvp == directional derivative of the encoder."""
    cfg, p = make("spiral")
    x = jax.random.normal(KEY, (2,))
    xd = jax.random.normal(jax.random.PRNGKey(1), (2,)) * 0.1
    jvp = M.encoder_jvp(p, cfg, x, xd)
    h = 1e-3
    fd = (M.encode(p, cfg, x + h * xd) - M.encode(p, cfg, x - h * xd)) / (2 * h)
    assert float(jnp.max(jnp.abs(jvp - fd))) < 5e-3

@pytest.mark.parametrize("loss_kind", ["mse", "huber"])
def test_losses_finite_and_decrease(loss_kind):
    """A few optimisation steps must reduce the total loss (sanity of the loop)."""
    import optax
    from data_gen import generate, fhn_derivatives
    data = generate(n_train=4, n_val=2, t_max=20.0, dt=0.1)
    ys = jnp.array(data["ys_train"]); u = jnp.array(data["u_train"])
    dots = fhn_derivatives(ys, u)
    scales = {"x": jnp.array(data["x_scale"]), "xdot": jnp.array(data["xdot_scale"])}
    cfg = M.ModelConfig(backbone="spiral", m=3)
    p = M.init_model(cfg, KEY)
    opt = optax.adam(3e-3); st = opt.init(p)

    def loss(pp):
        lr, ll, lp = M.compute_losses(pp, cfg, ys, dots, u, float(data["dt"]),
                                      30, 20, scales, loss_kind=loss_kind)
        return lr + ll + lp

    l0 = float(loss(p))
    for _ in range(15):
        g = jax.grad(loss)(p)
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
    l1 = float(loss(p))
    assert np.isfinite(l0) and np.isfinite(l1)
    assert l1 < l0

def test_sobolev_normalisation_scales_present():
    """Derivative normalisation (fix #5): dw/dt scale is markedly smaller than dv/dt."""
    from data_gen import generate
    data = generate(n_train=4, n_val=2, t_max=20.0, dt=0.1)
    xdot_scale = np.array(data["xdot_scale"])
    assert xdot_scale[0] > xdot_scale[1]
