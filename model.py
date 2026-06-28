"""
Shared Koopman model for the FitzHugh-Nagumo system.

This single module replaces the duplicated network code that previously lived in
both ``deep_koopman_hypernetwork.py`` and ``sweep_latent_dimension.py`` (fix #6).
It exposes a *backbone-agnostic* interface so the trainer, the diagnostics and
the tests all speak to the same functions:

    init_model(cfg, key)          -> params
    encode(params, cfg, x)        -> z      (latent, shape (..., 2m))
    decode(params, cfg, z)        -> x_hat  (state, shape (..., 2))
    step(params, cfg, z, u, dt)   -> z_next (one discrete Koopman step)
    generator(params, cfg, z, u)  -> z_dot  (continuous-time latent velocity)

Two backbones are implemented:

* ``"spiral"``  – a radius-conditioned, block-diagonal spiral operator.  Each of
  the ``m`` complex modes evolves as  z_j(t+dt) = e^{sigma_j dt} R(omega_j dt) z_j
  with sigma_j, omega_j produced by a small *hypernetwork* that reads the block
  radius r_j = ||z_j|| AND the control u (fix #1).  Stability is guaranteed by
  sigma = -softplus(.) <= 0 (fix #2).  A linear decoder keeps reconstruction
  well conditioned (the de-risking spirit of fix #8).

* ``"bilinear"`` – a control-affine generator  z_dot = (A + u B) z  (fix #7).  The
  state is embedded directly in the latent, z = [x; psi(x)], with a *fixed*
  linear decoder that reads the first two coordinates, so reconstruction is exact
  by construction (fix #8).  A and B are parameterized so that the symmetric part
  of A + u B is negative semi-definite for every u >= 0, which guarantees bounded
  rollouts (fix #2).

The loss functions (reconstruction, latent linearity, multi-step prediction),
each with an optional Sobolev / derivative term, live here too and are written
once against the abstract interface.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla


# --------------------------------------------------------------------------- #
#  Generic MLP helpers
# --------------------------------------------------------------------------- #
def init_mlp_params(layers, key):
    """Glorot-uniform initialised weights / zero biases for a Swish MLP."""
    keys = jax.random.split(key, len(layers) - 1)
    params = []
    for i in range(len(layers) - 1):
        in_dim, out_dim = layers[i], layers[i + 1]
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        w = jax.random.uniform(keys[i], (in_dim, out_dim), minval=-limit, maxval=limit)
        params.append({"w": w, "b": jnp.zeros((out_dim,))})
    return params


def forward_mlp(x, params):
    """Forward pass of an MLP with Swish hidden activations and a linear head."""
    a = x
    for layer in params[:-1]:
        a = jax.nn.swish(jnp.dot(a, layer["w"]) + layer["b"])
    last = params[-1]
    return jnp.dot(a, last["w"]) + last["b"]


# --------------------------------------------------------------------------- #
#  Model configuration
# --------------------------------------------------------------------------- #
class ModelConfig:
    """Static (non-traced) configuration for a Koopman model."""

    def __init__(self, backbone="spiral", m=4, enc_hidden=(64, 64),
                 hyper_hidden=(32, 32), emb_dim=4):
        assert backbone in ("spiral", "bilinear")
        self.backbone = backbone
        self.m = m                      # number of complex modes -> latent dim 2m
        self.latent = 2 * m
        self.enc_hidden = tuple(enc_hidden)
        self.hyper_hidden = tuple(hyper_hidden)
        self.emb_dim = emb_dim

    def __repr__(self):
        return (f"ModelConfig(backbone={self.backbone}, m={self.m}, "
                f"latent={self.latent})")


# --------------------------------------------------------------------------- #
#  Initialisation
# --------------------------------------------------------------------------- #
def init_model(cfg, key):
    """Initialise all learnable parameters for the requested backbone."""
    if cfg.backbone == "spiral":
        return _init_spiral(cfg, key)
    return _init_bilinear(cfg, key)


def _init_spiral(cfg, key):
    k_enc, k_dec, k_hyp, k_emb = jax.random.split(key, 4)
    enc = init_mlp_params([2, *cfg.enc_hidden, cfg.latent], k_enc)
    # Linear decoder z -> x (fix #8 spirit: linear readout, no compounding).
    dec_w = jax.random.normal(k_dec, (cfg.latent, 2)) * (1.0 / np.sqrt(cfg.latent))
    dec = {"w": dec_w, "b": jnp.zeros((2,))}
    # Hypernetwork: input [r_j, u, emb_j] -> (raw_sigma, omega).
    hyp = init_mlp_params([2 + cfg.emb_dim, *cfg.hyper_hidden, 2], k_hyp)
    # Bias the omega head so modes start as gentle rotators (omega ~ O(0.2..1)).
    b = np.zeros((2,))
    b[0] = 1.0     # raw_sigma -> sigma = -softplus(1.0) ~= -1.3 (mildly damped start)
    b[1] = 0.3     # omega start
    hyp[-1]["b"] = jnp.array(b)
    # Per-block learnable embeddings, spread out so modes specialise to harmonics.
    emb = jax.random.normal(k_emb, (cfg.m, cfg.emb_dim)) * 0.5
    emb = emb + jnp.linspace(-1.0, 1.0, cfg.m)[:, None]   # deterministic spread
    return {"enc": enc, "dec": dec, "hyper": hyp, "emb": emb}


def _init_bilinear(cfg, key):
    k_enc, k_a, k_b, k_ka, k_kb = jax.random.split(key, 5)
    d = cfg.latent
    # Encoder produces only the observable part psi(x); state is prepended.
    enc = init_mlp_params([2, *cfg.enc_hidden, d - 2], k_enc)
    s = 1.0 / np.sqrt(d)
    return {
        "enc": enc,
        # symmetric (dissipative) factors:  sym = -L Lᵀ  (negative semi-definite)
        "La": jax.random.normal(k_a, (d, d)) * 0.1,
        "Lb": jax.random.normal(k_b, (d, d)) * 0.1,
        # skew (conservative / rotational) generators
        "Ka": jax.random.normal(k_ka, (d, d)) * s,
        "Kb": jax.random.normal(k_kb, (d, d)) * s,
    }


# --------------------------------------------------------------------------- #
#  Encoder / decoder
# --------------------------------------------------------------------------- #
def encode(params, cfg, x):
    """Map state x (..., 2) to latent z (..., 2m)."""
    if cfg.backbone == "spiral":
        return forward_mlp(x, params["enc"])
    # bilinear: z = [x ; psi(x)]  (state-in-latent, fix #8)
    psi = forward_mlp(x, params["enc"])
    return jnp.concatenate([x, psi], axis=-1)


def decode(params, cfg, z):
    """Map latent z (..., 2m) back to state x_hat (..., 2)."""
    if cfg.backbone == "spiral":
        return jnp.dot(z, params["dec"]["w"]) + params["dec"]["b"]
    # bilinear: fixed projection onto the first two (= state) coordinates.
    return z[..., :2]


# --------------------------------------------------------------------------- #
#  Spiral backbone: radius-conditioned eigenvalues
# --------------------------------------------------------------------------- #
def spiral_eigs(params, cfg, z, u):
    """Return (sigma, omega), each (..., m), for the current latent z and control u.

    sigma_j = -softplus(.)  <= 0          (stability, fix #2)
    omega_j  = free                       (sign carries rotation direction)
    Both depend on the block radius r_j = ||z_j|| (fix #1) and on u.
    """
    m = cfg.m
    zb = z.reshape(z.shape[:-1] + (m, 2))
    r = jnp.sqrt(jnp.sum(zb ** 2, axis=-1) + 1e-12)          # (..., m)
    u_b = jnp.broadcast_to(u[..., None], r.shape)            # (..., m)
    emb = params["emb"]                                       # (m, emb_dim)
    emb_b = jnp.broadcast_to(emb, r.shape + (cfg.emb_dim,))   # (..., m, emb_dim)
    feat = jnp.concatenate([r[..., None], u_b[..., None], emb_b], axis=-1)
    out = forward_mlp(feat, params["hyper"])                  # (..., m, 2)
    sigma = -jax.nn.softplus(out[..., 0])
    omega = out[..., 1]
    return sigma, omega


def _spiral_step(params, cfg, z, u, dt):
    sigma, omega = spiral_eigs(params, cfg, z, u)
    zb = z.reshape(z.shape[:-1] + (cfg.m, 2))
    scale = jnp.exp(sigma * dt)
    c, s = jnp.cos(omega * dt), jnp.sin(omega * dt)
    z0, z1 = zb[..., 0], zb[..., 1]
    n0 = scale * (c * z0 - s * z1)
    n1 = scale * (s * z0 + c * z1)
    return jnp.stack([n0, n1], axis=-1).reshape(z.shape)


def _spiral_generator(params, cfg, z, u):
    sigma, omega = spiral_eigs(params, cfg, z, u)
    zb = z.reshape(z.shape[:-1] + (cfg.m, 2))
    z0, z1 = zb[..., 0], zb[..., 1]
    d0 = sigma * z0 - omega * z1
    d1 = omega * z0 + sigma * z1
    return jnp.stack([d0, d1], axis=-1).reshape(z.shape)


# --------------------------------------------------------------------------- #
#  Bilinear backbone: control-affine generator  z_dot = (A + u B) z
# --------------------------------------------------------------------------- #
def bilinear_matrices(params):
    """Reconstruct (A, B) from their stable parameterisation.

    A = -La Laᵀ + (Ka - Kaᵀ),  B = -Lb Lbᵀ + (Kb - Kbᵀ).
    sym(A) and sym(B) are negative semi-definite, so for any u >= 0 the symmetric
    part of A + u B is negative semi-definite => Re(eig(A + uB)) <= 0 (fix #2).
    """
    La, Lb, Ka, Kb = params["La"], params["Lb"], params["Ka"], params["Kb"]
    A = -La @ La.T + (Ka - Ka.T)
    B = -Lb @ Lb.T + (Kb - Kb.T)
    return A, B


def bilinear_M(params, u):
    """Generator matrix M(u) = A + u B, shape (..., d, d) (broadcasts over u)."""
    A, B = bilinear_matrices(params)
    u = jnp.asarray(u)
    return A + u[..., None, None] * B


def _bilinear_generator(params, cfg, z, u):
    A, B = bilinear_matrices(params)
    Az = z @ A.T
    Bz = z @ B.T
    return Az + u[..., None] * Bz


def _bilinear_step(params, cfg, z, u, dt):
    A, B = bilinear_matrices(params)
    # M(u) per sample; matrix exponential gives the exact linear flow over dt.
    M = A[None, ...] + u[..., None, None] * B[None, ...]      # (B, d, d)
    expM = jax.vmap(lambda mat: jsla.expm(mat * dt))(M)        # (B, d, d)
    return jnp.einsum("bij,bj->bi", expM, z)


# --------------------------------------------------------------------------- #
#  Backbone dispatch
# --------------------------------------------------------------------------- #
def step(params, cfg, z, u, dt):
    """One discrete Koopman step z_{t+1} = K(u) z_t."""
    if cfg.backbone == "spiral":
        return _spiral_step(params, cfg, z, u, dt)
    return _bilinear_step(params, cfg, z, u, dt)


def generator(params, cfg, z, u):
    """Continuous-time latent velocity z_dot = G(z, u)."""
    if cfg.backbone == "spiral":
        return _spiral_generator(params, cfg, z, u)
    return _bilinear_generator(params, cfg, z, u)


def rollout(params, cfg, z0, u_seq, dt):
    """Roll the latent forward over a control sequence u_seq (..., T).

    Returns predictions for steps 1..T, shape (..., T, 2m).  Batched over a
    leading axis on z0 (B, 2m) and u_seq (B, T).
    """
    def body(z, u_t):
        z_next = step(params, cfg, z, u_t, dt)
        return z_next, z_next

    # scan over time -> need time on axis 0
    u_time = jnp.moveaxis(u_seq, -1, 0)            # (T, B)
    _, zs = jax.lax.scan(body, z0, u_time)         # (T, B, 2m)
    return jnp.moveaxis(zs, 0, 1)                  # (B, T, 2m)


# --------------------------------------------------------------------------- #
#  Spectra for diagnostics (no gradient needed)
# --------------------------------------------------------------------------- #
def spiral_eig_grid(params, cfg, r, u):
    """sigma, omega for scalar/array radius r and control u over all m blocks.

    r, u broadcast to a common batch shape; returns (..., m) arrays.  Used by the
    'eigenvalues over latent radius' diagnostic.
    """
    r = jnp.asarray(r)
    u = jnp.asarray(u)
    r, u = jnp.broadcast_arrays(r, u)
    feat_r = jnp.broadcast_to(r[..., None, None], r.shape + (cfg.m, 1))
    feat_u = jnp.broadcast_to(u[..., None, None], u.shape + (cfg.m, 1))
    emb = jnp.broadcast_to(params["emb"], r.shape + (cfg.m, cfg.emb_dim))
    feat = jnp.concatenate([feat_r, feat_u, emb], axis=-1)
    out = forward_mlp(feat, params["hyper"])
    return -jax.nn.softplus(out[..., 0]), out[..., 1]


def bilinear_spectrum(params, u_grid):
    """Continuous-time eigenvalues of A + uB for each u in u_grid -> (len(u), d)."""
    A, B = bilinear_matrices(params)
    def eigs(u):
        return jnp.linalg.eigvals(A + u * B)
    return jax.vmap(eigs)(jnp.asarray(u_grid))


# --------------------------------------------------------------------------- #
#  Losses (backbone agnostic)
# --------------------------------------------------------------------------- #
def _mse(diff, scale=1.0):
    return jnp.mean((diff / scale) ** 2)


def _huber(diff, delta=1.0, scale=1.0):
    a = jnp.abs(diff / scale)
    quad = jnp.minimum(a, delta)
    lin = a - quad
    return jnp.mean(0.5 * quad ** 2 + delta * lin)


def _resid(diff, scale, kind):
    return _huber(diff, scale=scale) if kind == "huber" else _mse(diff, scale=scale)


def encoder_jvp(params, cfg, x, x_dot):
    """Push state velocity x_dot through the encoder Jacobian -> z_dot."""
    _, z_dot = jax.jvp(lambda xin: encode(params, cfg, xin), (x,), (x_dot,))
    return z_dot


def decoder_jvp(params, cfg, z, z_dot):
    """Push latent velocity z_dot through the decoder Jacobian -> x_dot."""
    _, x_dot = jax.jvp(lambda zin: decode(params, cfg, zin), (z,), (z_dot,))
    return x_dot


def compute_losses(params, cfg, traj, traj_dot, u, dt, n_predict, stride,
                   scales, loss_kind="mse", w_sobolev=0.1, pred_sobolev=False):
    """Reconstruction, latent-linearity and multi-step prediction losses.

    traj      : (B, T, 2)   states
    traj_dot  : (B, T, 2)   exact FHN velocities
    u         : (B, T)      control profile
    scales    : dict with 'x' (2,) and 'xdot' (2,) per-dimension normalisers
                (fix #5: the recovery-variable derivative is ~1/tau smaller, so we
                normalise both state dims and both derivative dims to equal weight)
    """
    B, T, _ = traj.shape
    sx = scales["x"]
    sxd = scales["xdot"]

    # ---- reconstruction (kept on from step 0, fix #3) -----------------------
    x_flat = traj.reshape(-1, 2)
    xd_flat = traj_dot.reshape(-1, 2)
    z_flat = encode(params, cfg, x_flat)
    x_rec = decode(params, cfg, z_flat)
    l_rec = _resid(x_flat - x_rec, sx, loss_kind)
    xrec_dot = jax.vmap(lambda x, xd: decoder_jvp(
        params, cfg, encode(params, cfg, x), encoder_jvp(params, cfg, x, xd)))(x_flat, xd_flat)
    l_rec = l_rec + w_sobolev * _resid(xd_flat - xrec_dot, sxd, loss_kind)

    # ---- latent linearity (one step + derivative consistency) ---------------
    z_seq = z_flat.reshape(B, T, cfg.latent)
    z_curr = z_seq[:, :-1].reshape(-1, cfg.latent)
    z_next = z_seq[:, 1:].reshape(-1, cfg.latent)
    u_curr = u[:, :-1].reshape(-1)
    z_pred = step(params, cfg, z_curr, u_curr, dt)
    l_lin = _resid(z_next - z_pred, 1.0, loss_kind)
    # derivative consistency: encoder-jvp velocity vs latent generator
    z_dot_true = jax.vmap(lambda x, xd: encoder_jvp(params, cfg, x, xd))(x_flat, xd_flat)
    z_dot_pred = generator(params, cfg, z_flat, u.reshape(-1))
    l_lin = l_lin + w_sobolev * _resid(z_dot_true - z_dot_pred, 1.0, loss_kind)

    # ---- multi-step prediction over sliding windows (curriculum on n_predict)-
    n_win = (T - 1 - n_predict) // stride + 1

    def window(i):
        start = i * stride
        z0 = jax.lax.dynamic_slice(z_seq, (0, start, 0), (B, 1, cfg.latent))[:, 0]
        u_w = jax.lax.dynamic_slice(u, (0, start), (B, n_predict))
        x_tgt = jax.lax.dynamic_slice(traj, (0, start + 1, 0), (B, n_predict, 2))
        xd_tgt = jax.lax.dynamic_slice(traj_dot, (0, start + 1, 0), (B, n_predict, 2))
        z_roll = rollout(params, cfg, z0, u_w, dt)                  # (B, n, 2m)
        x_roll = decode(params, cfg, z_roll)
        lp = _resid(x_roll - x_tgt, sx, loss_kind)
        if pred_sobolev:
            # rolled-out velocity through decoder jvp (expensive; off by default).
            # z_roll[:,k] is the latent at time start+1+k, so use the control at
            # that same landing time (not the one-step-earlier u_w).
            u_tgt = jax.lax.dynamic_slice(u, (0, start + 1), (B, n_predict))
            zr = z_roll.reshape(-1, cfg.latent)
            zr_dot = generator(params, cfg, zr, u_tgt.reshape(-1))
            xr_dot = jax.vmap(lambda z, zd: decoder_jvp(params, cfg, z, zd))(zr, zr_dot)
            xr_dot = xr_dot.reshape(B, n_predict, 2)
            lp = lp + w_sobolev * _resid(xr_dot - xd_tgt, sxd, loss_kind)
        return lp

    l_pred = jnp.mean(jax.vmap(window)(jnp.arange(n_win)))
    return l_rec, l_lin, l_pred
