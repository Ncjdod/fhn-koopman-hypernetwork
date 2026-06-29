"""Phase-Warped Floquet Operator: a non-recursive surrogate G(x0, u(.), t) -> x(t).

The state is written as a steady limit-cycle waveform (a Fourier series in an
accumulated phase Phi(t)) plus a decaying isostable transient:

    Phi(t)   = phi0 + integral_0^t omega(u(tau)) dtau         (prefix-sum over the profile)
    psi_j(t) = rho0_j * exp(integral_0^t kappa_j(u) dtau)     (kappa_j < 0, decays)
    x(t)     = mu + sum_k [A_k cos kPhi + B_k sin kPhi]
                  + sum_j psi_j sum_k [Cc_jk cos kPhi + Cs_jk sin kPhi]

Time enters ONLY through cos/sin(k Phi) (bounded, periodic -> eternal) and the
decaying envelope, so any query time is one O(1) gather+evaluate -- no recursion,
no integration. The cycle coefficients depend on the current context only (every
initial condition relaxes onto the same attractor); phase phi0 and transient
amplitude rho0 carry the x0 dependence.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _init_mlp(sizes, key, scale_last=1.0):
    ks = jax.random.split(key, len(sizes) - 1)
    ps = []
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        lim = np.sqrt(6.0 / (a + b)) * (scale_last if i == len(sizes) - 2 else 1.0)
        ps.append({"w": jax.random.uniform(ks[i], (a, b), minval=-lim, maxval=lim),
                   "b": jnp.zeros((b,))})
    return ps


def _mlp(x, ps):
    a = x
    for layer in ps[:-1]:
        a = jax.nn.swish(a @ layer["w"] + layer["b"])
    return a @ ps[-1]["w"] + ps[-1]["b"]


class PWFOConfig:
    """Static config for the Phase-Warped Floquet Operator."""

    def __init__(self, d=2, K=10, m=1, p=64, enc_hidden=(64, 64),
                 ctx_hidden=(64, 64), head_hidden=(32, 32), omega0=0.05,
                 local_waveform=False):
        self.d, self.K, self.m, self.p = d, K, m, p
        self.enc_hidden = tuple(enc_hidden)
        self.ctx_hidden = tuple(ctx_hidden)
        self.head_hidden = tuple(head_hidden)
        self.omega0 = omega0
        self.local_waveform = local_waveform

    def __repr__(self):
        return f"PWFOConfig(d={self.d}, K={self.K}, m={self.m}, p={self.p})"


def init_pwfo(cfg, key):
    """Initialise all PWFO parameters."""
    ke, kc, kw, kk, kh = jax.random.split(key, 5)
    d, K, m, p = cfg.d, cfg.K, cfg.m, cfg.p
    enc = _init_mlp([d, *cfg.enc_hidden, 2 + m], ke, scale_last=0.5)
    ctx = _init_mlp([6, *cfg.ctx_hidden, p], kc)
    g_omega = _init_mlp([1 + p, *cfg.head_hidden, 1], kw, scale_last=0.1)
    g_kappa = _init_mlp([1 + p, *cfg.head_hidden, m], kk, scale_last=0.1)
    n_head = d + 2 * d * K + 2 * d * m * K
    head_in = p + (1 if cfg.local_waveform else 0)
    head = _init_mlp([head_in, *cfg.ctx_hidden, n_head], kh, scale_last=0.1)
    return {"enc": enc, "ctx": ctx, "g_omega": g_omega, "g_kappa": g_kappa, "head": head}


def _profile_stats(u):
    """Summary features of a current profile u (B,S) -> (B, 2d+4)-ish context input."""
    return jnp.stack([u.mean(1), u.std(1), u.min(1), u.max(1),
                      u[:, 0], u[:, -1]], axis=-1)


def _waveform(params, cfg, feat):
    """Decode context feat (..., p[+1]) into Fourier waveform coeffs (leading-shape ...)."""
    d, K, m = cfg.d, cfg.K, cfg.m
    out = _mlp(feat, params["head"])
    lead = out.shape[:-1]
    i = 0
    mu = out[..., i:i + d]; i += d
    A = out[..., i:i + d * K].reshape(*lead, d, K); i += d * K
    B = out[..., i:i + d * K].reshape(*lead, d, K); i += d * K
    Cc = out[..., i:i + d * m * K].reshape(*lead, d, m, K); i += d * m * K
    Cs = out[..., i:i + d * m * K].reshape(*lead, d, m, K); i += d * m * K
    return mu, A, B, Cc, Cs


def segment_rates(params, cfg, u, c):
    """Per-grid-point frequency omega>0 and isostable decay kappa<0 (B,S),(B,S,m)."""
    S = u.shape[1]
    cb = jnp.broadcast_to(c[:, None, :], (c.shape[0], S, c.shape[1]))
    feat = jnp.concatenate([u[..., None], cb], axis=-1)
    omega = cfg.omega0 + jax.nn.softplus(_mlp(feat, params["g_omega"])[..., 0])
    kappa = -jax.nn.softplus(_mlp(feat, params["g_kappa"]))
    return omega, kappa


def encode_ic(params, cfg, x0):
    """Initial phase phi0 (B,) and transient amplitude rho0 (B,m) from x0."""
    e = _mlp(x0, params["enc"])
    phi0 = jnp.arctan2(e[:, 1], e[:, 0])
    rho0 = e[:, 2:2 + cfg.m]
    return phi0, rho0


def forward(params, cfg, x0, u_grid, t_query, dt):
    """Predict x(t) for query times t_query (B,Q) directly. Returns (B,Q,d)."""
    B, S = u_grid.shape
    c = _mlp(_profile_stats(u_grid), params["ctx"])
    phi0, rho0 = encode_ic(params, cfg, x0)
    omega, kappa = segment_rates(params, cfg, u_grid, c)

    Pw = dt * jnp.cumsum(omega, axis=1)
    Pw_excl = jnp.concatenate([jnp.zeros((B, 1)), Pw[:, :-1]], axis=1)
    Pk = dt * jnp.cumsum(kappa, axis=1)
    Pk_excl = jnp.concatenate([jnp.zeros((B, 1, cfg.m)), Pk[:, :-1]], axis=1)

    s = jnp.clip(jnp.floor(t_query / dt).astype(jnp.int32), 0, S - 1)
    frac = t_query - s.astype(t_query.dtype) * dt

    Phi = (phi0[:, None]
           + jnp.take_along_axis(Pw_excl, s, axis=1)
           + jnp.take_along_axis(omega, s, axis=1) * frac)
    s_m = jnp.broadcast_to(s[:, :, None], s.shape + (cfg.m,))
    L = (jnp.take_along_axis(Pk_excl, s_m, axis=1)
         + jnp.take_along_axis(kappa, s_m, axis=1) * frac[..., None])
    psi = rho0[:, None, :] * jnp.exp(L)

    k = jnp.arange(1, cfg.K + 1)
    cosk = jnp.cos(Phi[..., None] * k)
    sink = jnp.sin(Phi[..., None] * k)

    if cfg.local_waveform:
        u_q = jnp.take_along_axis(u_grid, s, axis=1)
        cb = jnp.broadcast_to(c[:, None, :], (B, s.shape[1], c.shape[1]))
        mu, A, B_, Cc, Cs = _waveform(params, cfg, jnp.concatenate([cb, u_q[..., None]], -1))
        x_cycle = mu + jnp.einsum("bqdk,bqk->bqd", A, cosk) \
                     + jnp.einsum("bqdk,bqk->bqd", B_, sink)
        tr = jnp.einsum("bqdmk,bqk->bqmd", Cc, cosk) + jnp.einsum("bqdmk,bqk->bqmd", Cs, sink)
    else:
        mu, A, B_, Cc, Cs = _waveform(params, cfg, c)
        x_cycle = mu[:, None, :] + jnp.einsum("bdk,bqk->bqd", A, cosk) \
                                 + jnp.einsum("bdk,bqk->bqd", B_, sink)
        tr = jnp.einsum("bdmk,bqk->bqmd", Cc, cosk) + jnp.einsum("bdmk,bqk->bqmd", Cs, sink)
    x_tr = jnp.einsum("bqm,bqmd->bqd", psi, tr)
    return x_cycle + x_tr
