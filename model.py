"""Shared Koopman model for the FitzHugh-Nagumo system."""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

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

class ModelConfig:
    """Static (non-traced) configuration for a Koopman model."""

    def __init__(self, backbone="spiral", m=4, enc_hidden=(64, 64),
                 hyper_hidden=(32, 32), emb_dim=4, radial="stuart_landau",
                 beta_min=1e-2):
        assert backbone in ("spiral", "bilinear")
        assert radial in ("stuart_landau", "clamped")
        self.backbone = backbone
        self.m = m
        self.latent = 2 * m
        self.enc_hidden = tuple(enc_hidden)
        self.hyper_hidden = tuple(hyper_hidden)
        self.emb_dim = emb_dim
        self.radial = radial
        self.beta_min = beta_min

    def __repr__(self):
        return (f"ModelConfig(backbone={self.backbone}, m={self.m}, "
                f"latent={self.latent}, radial={self.radial})")

def init_model(cfg, key):
    """Initialise all learnable parameters for the requested backbone."""
    if cfg.backbone == "spiral":
        return _init_spiral(cfg, key)
    return _init_bilinear(cfg, key)

def _init_spiral(cfg, key):
    k_enc, k_dec, k_hyp, k_emb, k_w = jax.random.split(key, 5)
    enc = init_mlp_params([2, *cfg.enc_hidden, cfg.latent], k_enc)
    dec_w = jax.random.normal(k_dec, (cfg.latent, 2)) * (1.0 / np.sqrt(cfg.latent))
    dec = {"w": dec_w, "b": jnp.zeros((2,))}
    emb = jax.random.normal(k_emb, (cfg.m, cfg.emb_dim)) * 0.5
    emb = emb + jnp.linspace(-1.0, 1.0, cfg.m)[:, None]

    if cfg.radial == "stuart_landau":
        hyp_p = init_mlp_params([1 + cfg.emb_dim, *cfg.hyper_hidden, 2], k_hyp)
        bp = np.zeros((2,))
        bp[0] = 0.1
        bp[1] = -1.0
        hyp_p[-1]["b"] = jnp.array(bp)
        hyp_w = init_mlp_params([2 + cfg.emb_dim, *cfg.hyper_hidden, 1], k_w)
        hyp_w[-1]["b"] = jnp.array([0.3])
        return {"enc": enc, "dec": dec, "hyp_p": hyp_p, "hyp_w": hyp_w, "emb": emb}

    hyp = init_mlp_params([2 + cfg.emb_dim, *cfg.hyper_hidden, 2], k_hyp)
    b = np.zeros((2,))
    b[0] = 1.0
    b[1] = 0.3
    hyp[-1]["b"] = jnp.array(b)
    return {"enc": enc, "dec": dec, "hyper": hyp, "emb": emb}

def _init_bilinear(cfg, key):
    k_enc, k_a, k_b, k_ka, k_kb = jax.random.split(key, 5)
    d = cfg.latent
    enc = init_mlp_params([2, *cfg.enc_hidden, d - 2], k_enc)
    s = 1.0 / np.sqrt(d)
    return {
        "enc": enc,
        "La": jax.random.normal(k_a, (d, d)) * 0.1,
        "Lb": jax.random.normal(k_b, (d, d)) * 0.1,
        "Ka": jax.random.normal(k_ka, (d, d)) * s,
        "Kb": jax.random.normal(k_kb, (d, d)) * s,
    }

def encode(params, cfg, x):
    """Map state x (..., 2) to latent z (..., 2m)."""
    if cfg.backbone == "spiral":
        return forward_mlp(x, params["enc"])
    psi = forward_mlp(x, params["enc"])
    return jnp.concatenate([x, psi], axis=-1)

def decode(params, cfg, z):
    """Map latent z (..., 2m) back to state x_hat (..., 2)."""
    if cfg.backbone == "spiral":
        return jnp.dot(z, params["dec"]["w"]) + params["dec"]["b"]
    return z[..., :2]

def _spiral_blocks(z, cfg):
    """Reshape latent into m 2-D blocks and return (zb, r) with r the block radii."""
    zb = z.reshape(z.shape[:-1] + (cfg.m, 2))
    r = jnp.sqrt(jnp.sum(zb ** 2, axis=-1) + 1e-12)
    return zb, r

def spiral_sl_coeffs(params, cfg, r, u):
    """Stuart-Landau coefficients (sigma0, beta, omega), each (..., m)."""
    u_b = jnp.broadcast_to(u[..., None], r.shape)
    emb = params["emb"]
    emb_b = jnp.broadcast_to(emb, r.shape + (cfg.emb_dim,))
    feat_p = jnp.concatenate([u_b[..., None], emb_b], axis=-1)
    pout = forward_mlp(feat_p, params["hyp_p"])
    sigma0 = pout[..., 0]
    beta = jax.nn.softplus(pout[..., 1]) + cfg.beta_min
    feat_w = jnp.concatenate([r[..., None], u_b[..., None], emb_b], axis=-1)
    omega = forward_mlp(feat_w, params["hyp_w"])[..., 0]
    return sigma0, beta, omega

def spiral_eigs(params, cfg, z, u):
    """Return (sigma_eff, omega), each (..., m), for the current latent z, control u."""
    zb, r = _spiral_blocks(z, cfg)
    if cfg.radial == "stuart_landau":
        sigma0, beta, omega = spiral_sl_coeffs(params, cfg, r, u)
        sigma_eff = sigma0 - beta * r ** 2
        return sigma_eff, omega
    u_b = jnp.broadcast_to(u[..., None], r.shape)
    emb_b = jnp.broadcast_to(params["emb"], r.shape + (cfg.emb_dim,))
    feat = jnp.concatenate([r[..., None], u_b[..., None], emb_b], axis=-1)
    out = forward_mlp(feat, params["hyper"])
    return -jax.nn.softplus(out[..., 0]), out[..., 1]

def _sl_radius_step(sigma0, beta, r0_sq, dt):
    """Exact flow of dot(r^2) = 2 sigma0 r^2 - 2 beta (r^2)^2 over a step dt."""
    tol = 1e-4
    g = jnp.clip(2.0 * sigma0 * dt, -30.0, 30.0)
    em = jnp.exp(-g)
    sig_safe = jnp.where(jnp.abs(sigma0) < tol, tol, sigma0)
    den = sig_safe * em + beta * r0_sq * (1.0 - em)
    R_gen = sig_safe * r0_sq / (den + 1e-30)
    R_lim = r0_sq / (1.0 + 2.0 * beta * r0_sq * dt)
    R_new = jnp.where(jnp.abs(sigma0) < tol, R_lim, R_gen)
    return jnp.clip(R_new, 0.0, 1e12)

def _spiral_step(params, cfg, z, u, dt):
    zb, r = _spiral_blocks(z, cfg)
    z0, z1 = zb[..., 0], zb[..., 1]
    if cfg.radial == "stuart_landau":
        sigma0, beta, omega = spiral_sl_coeffs(params, cfg, r, u)
        R0 = r ** 2
        R_new = _sl_radius_step(sigma0, beta, R0, dt)
        scale = jnp.sqrt(R_new / (R0 + 1e-12))
    else:
        sigma, omega = spiral_eigs(params, cfg, z, u)
        scale = jnp.exp(sigma * dt)
    c, s = jnp.cos(omega * dt), jnp.sin(omega * dt)
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

def bilinear_matrices(params):
    """Reconstruct (A, B) from their stable parameterisation."""
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
    M = A[None, ...] + u[..., None, None] * B[None, ...]
    expM = jax.vmap(lambda mat: jsla.expm(mat * dt))(M)
    return jnp.einsum("bij,bj->bi", expM, z)

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
    """Roll the latent forward over a control sequence u_seq (..., T)."""
    @jax.checkpoint
    def body(z, u_t):
        z_next = step(params, cfg, z, u_t, dt)
        return z_next, z_next

    u_time = jnp.moveaxis(u_seq, -1, 0)
    _, zs = jax.lax.scan(body, z0, u_time)
    return jnp.moveaxis(zs, 0, 1)

def spiral_eig_grid(params, cfg, r, u):
    """sigma, omega for scalar/array radius r and control u over all m blocks."""
    r = jnp.asarray(r)
    u = jnp.asarray(u)
    r, u = jnp.broadcast_arrays(r, u)
    r_b = jnp.broadcast_to(r[..., None], r.shape + (cfg.m,))
    if cfg.radial == "stuart_landau":
        sigma0, beta, omega = spiral_sl_coeffs(params, cfg, r_b, u)
        return sigma0 - beta * r_b ** 2, omega
    u_b = jnp.broadcast_to(u[..., None], u.shape + (cfg.m,))
    emb = jnp.broadcast_to(params["emb"], r.shape + (cfg.m, cfg.emb_dim))
    feat = jnp.concatenate([r_b[..., None], u_b[..., None], emb], axis=-1)
    out = forward_mlp(feat, params["hyper"])
    return -jax.nn.softplus(out[..., 0]), out[..., 1]

def bilinear_spectrum(params, u_grid):
    """Continuous-time eigenvalues of A + uB for each u in u_grid -> (len(u), d)."""
    A, B = bilinear_matrices(params)
    def eigs(u):
        return jnp.linalg.eigvals(A + u * B)
    return jax.vmap(eigs)(jnp.asarray(u_grid))

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
    """Reconstruction, latent-linearity and multi-step prediction losses."""
    B, T, _ = traj.shape
    sx = scales["x"]
    sxd = scales["xdot"]

    x_flat = traj.reshape(-1, 2)
    xd_flat = traj_dot.reshape(-1, 2)
    z_flat = encode(params, cfg, x_flat)
    x_rec = decode(params, cfg, z_flat)
    l_rec = _resid(x_flat - x_rec, sx, loss_kind)
    xrec_dot = jax.vmap(lambda x, xd: decoder_jvp(
        params, cfg, encode(params, cfg, x), encoder_jvp(params, cfg, x, xd)))(x_flat, xd_flat)
    l_rec = l_rec + w_sobolev * _resid(xd_flat - xrec_dot, sxd, loss_kind)

    z_seq = z_flat.reshape(B, T, cfg.latent)
    z_curr = z_seq[:, :-1].reshape(-1, cfg.latent)
    z_next = z_seq[:, 1:].reshape(-1, cfg.latent)
    u_curr = u[:, :-1].reshape(-1)
    z_pred = step(params, cfg, z_curr, u_curr, dt)
    l_lin = _resid(z_next - z_pred, 1.0, loss_kind)
    z_dot_true = jax.vmap(lambda x, xd: encoder_jvp(params, cfg, x, xd))(x_flat, xd_flat)
    z_dot_pred = generator(params, cfg, z_flat, u.reshape(-1))
    l_lin = l_lin + w_sobolev * _resid(z_dot_true - z_dot_pred, 1.0, loss_kind)

    n_win = (T - 1 - n_predict) // stride + 1

    def window(i):
        start = i * stride
        z0 = jax.lax.dynamic_slice(z_seq, (0, start, 0), (B, 1, cfg.latent))[:, 0]
        u_w = jax.lax.dynamic_slice(u, (0, start), (B, n_predict))
        x_tgt = jax.lax.dynamic_slice(traj, (0, start + 1, 0), (B, n_predict, 2))
        xd_tgt = jax.lax.dynamic_slice(traj_dot, (0, start + 1, 0), (B, n_predict, 2))
        z_roll = rollout(params, cfg, z0, u_w, dt)
        x_roll = decode(params, cfg, z_roll)
        lp = _resid(x_roll - x_tgt, sx, loss_kind)
        if pred_sobolev:
            u_tgt = jax.lax.dynamic_slice(u, (0, start + 1), (B, n_predict))
            zr = z_roll.reshape(-1, cfg.latent)
            zr_dot = generator(params, cfg, zr, u_tgt.reshape(-1))
            xr_dot = jax.vmap(lambda z, zd: decoder_jvp(params, cfg, z, zd))(zr, zr_dot)
            xr_dot = xr_dot.reshape(B, n_predict, 2)
            lp = lp + w_sobolev * _resid(xr_dot - xd_tgt, sxd, loss_kind)
        return lp

    l_pred = jnp.mean(jax.lax.map(window, jnp.arange(n_win)))
    return l_rec, l_lin, l_pred
