"""
Spectral & dynamical diagnostics for the trained FHN Koopman models.

Produces the figures and tables requested for the article:

  D0  learned spectrum  sigma(u), omega(u)  vs the Jacobian eigenvalues mu_pm(u)
      and the measured limit-cycle frequency                        (fix #9)
  D1  eigenvalue trajectories over the latent radius r              (a)
  D2  Floquet multipliers of the recovered cycle vs ground truth    (b)
  D3  Koopman-mode reconstruction quality (per-mode decomposition)  (c)
  D4  long-horizon rollout error decomposed by spectral band        (d)
  plus the headline validation rollout and the success-rate summary.

Run:  python diagnostics.py --model data/koopman_spiral.pkl
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model as M
import fhn_theory as TH
from dynamics import get_external_current
from simulation import simulate_fhn, simulate_fhn_batch
from data_gen import fhn_derivatives

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
A, B, TAU = TH.A_DEFAULT, TH.B_DEFAULT, TH.TAU_DEFAULT


# --------------------------------------------------------------------------- #
def load_model(path):
    with open(path, "rb") as f:
        out = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, out["params"])
    cfg = M.ModelConfig(backbone=out["cfg"]["backbone"], m=out["cfg"]["m"])
    scales = {k: jnp.asarray(v) for k, v in out["scales"].items()}
    return params, cfg, scales, float(out["dt"]), out


def fixed_point_state(u):
    """Return the FHN fixed point (v*, w*) nearest the cycle for control u."""
    fps = TH.fixed_points(u, A, B)
    if len(fps) == 0:
        return None
    v = fps[np.argsort(np.abs(fps))][0] if len(fps) == 1 else np.sort(fps)[len(fps) // 2]
    return np.array([v, (v + A) / B])


def learned_eigs_state(params, cfg, x, u):
    """Continuous-time (sigma, omega) of the learned operator at state x, control u.

    For spiral: the per-block radius-conditioned values.
    For bilinear: eigenvalues of A + uB (sigma=Re, omega=|Im|), top-m by Re.
    Returns sigma (m,), omega (m,) sorted by descending sigma.
    """
    x = jnp.asarray(x, dtype=jnp.float32)[None, :]
    if cfg.backbone == "spiral":
        z = M.encode(params, cfg, x)
        s, o = M.spiral_eigs(params, cfg, z, jnp.array([float(u)]))
        s, o = np.array(s[0]), np.array(o[0])
    else:
        Amat, Bmat = M.bilinear_matrices(params)
        ev = np.linalg.eigvals(np.array(Amat) + float(u) * np.array(Bmat))
        idx = np.argsort(-ev.real)[:cfg.m]
        s, o = ev.real[idx], np.abs(ev.imag[idx])
    order = np.argsort(-s)
    return s[order], o[order]


# --------------------------------------------------------------------------- #
def diag_spectrum_vs_jacobian(params, cfg, tag, u_grid=None):
    """D0 / fix #9: learned sigma,omega(u) vs Jacobian mu_pm(u) and cycle freq."""
    if u_grid is None:
        u_grid = np.linspace(0.1, 1.7, 17)
    th = TH.spectrum_over_u(u_grid, A, B, TAU)
    cyc = TH.cycle_frequency_over_u(u_grid, a=A, b=B, tau=TAU, t_max=300.0, dt=0.02)

    lead_s, lead_o, all_s, all_o = [], [], [], []
    for u in u_grid:
        xstar = fixed_point_state(u)
        s, o = learned_eigs_state(params, cfg, xstar, u)
        lead_s.append(s[0]); lead_o.append(o[0])
        all_s.append(s); all_o.append(o)
    lead_s, lead_o = np.array(lead_s), np.array(lead_o)
    all_s, all_o = np.array(all_s), np.array(all_o)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for j in range(all_s.shape[1]):
        ax1.plot(u_grid, all_s[:, j], color="0.8", lw=1, zorder=1)
    ax1.plot(u_grid, lead_s, "o-", color="#2ca02c", lw=2, label="learned (dominant mode)")
    ax1.plot(u_grid, th["sigma"], "s--", color="#1f77b4", lw=2, label=r"Jacobian $\mathrm{Re}\,\mu$")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("control  u"); ax1.set_ylabel(r"growth rate  $\sigma$")
    ax1.set_title("Real part: learned vs Jacobian"); ax1.legend(); ax1.grid(alpha=.3)

    for j in range(all_o.shape[1]):
        ax2.plot(u_grid, all_o[:, j], color="0.8", lw=1, zorder=1)
    ax2.plot(u_grid, lead_o, "o-", color="#2ca02c", lw=2, label="learned (dominant mode)")
    ax2.plot(u_grid, th["omega"], "s--", color="#1f77b4", lw=2, label=r"Jacobian $|\mathrm{Im}\,\mu|$")
    ax2.plot(u_grid, cyc, "^:", color="#d62728", lw=2, label=r"limit-cycle $2\pi/T$")
    ax2.set_xlabel("control  u"); ax2.set_ylabel(r"frequency  $\omega$")
    ax2.set_title("Imag part: learned vs Jacobian vs cycle"); ax2.legend(); ax2.grid(alpha=.3)

    fig.suptitle(f"D0  Learned spectrum vs ground truth  [{tag}]", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(PLOTS, f"diag_D0_spectrum_{tag}.png")
    fig.savefig(p, dpi=150); plt.close(fig)

    # correlation in the stable regime (where the linearisation is the attractor)
    mask = th["stable"] & np.isfinite(th["sigma"])
    corr = float(np.corrcoef(lead_s[mask], th["sigma"][mask])[0, 1]) if mask.sum() > 2 else float("nan")
    return {"plot": p, "sigma_corr_stable": corr,
            "u": u_grid.tolist(), "learned_sigma": lead_s.tolist(),
            "learned_omega": lead_o.tolist(), "jac_sigma": th["sigma"].tolist(),
            "jac_omega": th["omega"].tolist(), "cycle_omega": cyc.tolist()}


# --------------------------------------------------------------------------- #
def diag_eigs_over_radius(params, cfg, tag, u_values=(0.3, 1.0, 1.6), r_max=3.0):
    """D1 / fix #1: how the per-block eigenvalues move as the latent radius grows."""
    rs = np.linspace(0.02, r_max, 60)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    info = {}
    for ax, u in zip(axes, u_values):
        if cfg.backbone == "spiral":
            s, o = M.spiral_eig_grid(params, cfg, jnp.asarray(rs), jnp.full_like(jnp.asarray(rs), u))
            s, o = np.array(s), np.array(o)               # (len(rs), m)
            for j in range(cfg.m):
                sc = ax.scatter(o[:, j], s[:, j], c=rs, cmap="viridis", s=12)
            ax.axhline(0, color="r", lw=0.8, ls="--")
            ax.set_title(f"u = {u}")
            ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$\sigma$")
            # cycle radius estimate: smallest r where dominant sigma crosses ~0
            dom = s.max(axis=1)
            cross = np.where(np.diff(np.sign(dom)))[0]
            info[f"u={u}_sigma_range"] = [float(s.min()), float(s.max())]
            info[f"u={u}_omega_at_rmax"] = float(o[-1].max())
            info[f"u={u}_omega_at_rmin"] = float(o[0].max())
        else:
            Amat, Bmat = np.array(M.bilinear_matrices(params)[0]), np.array(M.bilinear_matrices(params)[1])
            ev = np.linalg.eigvals(Amat + u * Bmat)
            ax.scatter(np.abs(ev.imag), ev.real, c="#1f77b4", s=30)
            ax.axhline(0, color="r", lw=0.8, ls="--")
            ax.set_title(f"u = {u} (bilinear: r-independent)")
            ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$\sigma$")
    if cfg.backbone == "spiral":
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="latent radius r", shrink=0.8)
    fig.suptitle(f"D1  Eigenvalue trajectories over latent radius  [{tag}]", fontweight="bold")
    p = os.path.join(PLOTS, f"diag_D1_radius_{tag}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    info["plot"] = p
    return info


# --------------------------------------------------------------------------- #
def _true_monodromy(u, period, dt=0.005):
    """Ground-truth state-space monodromy (2x2) along the FHN cycle at control u."""
    n = int(round(period / dt))
    t = jnp.linspace(0.0, period, n + 1)
    ys = np.array(simulate_fhn(jnp.array([-1.0, -0.5]), jnp.linspace(0, 400, 40001),
                               I_type="constant", I_val=u))
    cyc0 = ys[-(int(period / 0.01)):]            # one period of the settled cycle
    x0 = cyc0[0]
    # integrate variational equation Phi' = J(t) Phi with RK4 along the cycle
    Phi = np.eye(2)
    v, w = float(x0[0]), float(x0[1])
    for _ in range(n):
        def deriv(state, Phi):
            vv, ww = state
            J = np.array([[1 - vv ** 2, -1.0], [1.0 / TAU, -B / TAU]])
            f = np.array([vv - vv ** 3 / 3 - ww + u, (vv + A - B * ww) / TAU])
            return f, J @ Phi
        f1, P1 = deriv((v, w), Phi)
        f2, P2 = deriv((v + dt / 2 * f1[0], w + dt / 2 * f1[1]), Phi + dt / 2 * P1)
        f3, P3 = deriv((v + dt / 2 * f2[0], w + dt / 2 * f2[1]), Phi + dt / 2 * P2)
        f4, P4 = deriv((v + dt * f3[0], w + dt * f3[1]), Phi + dt * P3)
        v += dt / 6 * (f1[0] + 2 * f2[0] + 2 * f3[0] + f4[0])
        w += dt / 6 * (f1[1] + 2 * f2[1] + 2 * f3[1] + f4[1])
        Phi = Phi + dt / 6 * (P1 + 2 * P2 + 2 * P3 + P4)
    return np.linalg.eigvals(Phi), x0


def diag_floquet(params, cfg, dt, tag, u=1.0):
    """D2 / fix (b): Floquet multipliers of the recovered cycle vs ground truth."""
    period, _ = TH.limit_cycle_period(u, A, B, TAU, t_max=400.0, dt=0.01)
    if not np.isfinite(period):
        return {"note": "no limit cycle at this u", "u": u}
    true_mult, x0 = _true_monodromy(u, period, dt=0.005)

    P = int(round(period / dt))
    x0j = jnp.asarray(x0, dtype=jnp.float32)

    def state_return(x):
        z = M.encode(params, cfg, x[None, :])
        useq = jnp.full((1, P), u)
        zr = M.rollout(params, cfg, z, useq, dt)
        return M.decode(params, cfg, zr[:, -1])[0]

    M_state = np.array(jax.jacobian(state_return)(x0j))
    learn_mult = np.linalg.eigvals(M_state)

    # full latent monodromy: d z_final / d z_0  (square, 2m x 2m)
    z0 = M.encode(params, cfg, x0j[None, :])[0]

    def latent_return(z):
        useq = jnp.full((1, P), u)
        return M.rollout(params, cfg, z[None, :], useq, dt)[:, -1][0]
    M_lat = np.array(jax.jacobian(latent_return)(z0))
    lat_mult = np.linalg.eigvals(M_lat)
    lat_top = np.sort(np.abs(lat_mult))[::-1][:4]

    fig, ax = plt.subplots(figsize=(6, 6))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8, label="unit circle")
    ax.scatter(true_mult.real, true_mult.imag, s=130, marker="o",
               facecolors="none", edgecolors="#1f77b4", lw=2, label="true (2x2)")
    ax.scatter(learn_mult.real, learn_mult.imag, s=70, marker="x",
               color="#2ca02c", lw=2, label="learned (state 2x2)")
    ax.scatter(lat_mult.real, lat_mult.imag, s=18, marker=".",
               color="0.6", label="learned (latent)")
    ax.set_aspect("equal"); ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.set_title(f"D2  Floquet multipliers @ u={u}  [{tag}]", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.3)
    p = os.path.join(PLOTS, f"diag_D2_floquet_{tag}.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)

    return {"plot": p, "u": u, "period": float(period),
            "true_multipliers": [complex(c).__repr__() for c in true_mult],
            "true_mult_abs": sorted(np.abs(true_mult).tolist(), reverse=True),
            "learned_state_mult_abs": sorted(np.abs(learn_mult).tolist(), reverse=True),
            "learned_latent_top_abs": lat_top.tolist()}


# --------------------------------------------------------------------------- #
def diag_koopman_modes(params, cfg, dt, tag, u=1.0, n_periods=1.0):
    """D3 / fix (c): per-mode (harmonic) decomposition of the recovered cycle."""
    period, _ = TH.limit_cycle_period(u, A, B, TAU, t_max=400.0, dt=0.01)
    if not np.isfinite(period):
        period = 37.0
    # settle onto the true cycle, encode, roll the learned operator over ~1 period
    ys = np.array(simulate_fhn(jnp.array([-1.0, -0.5]), jnp.linspace(0, 400, 40001),
                               I_type="constant", I_val=u))
    x0 = jnp.asarray(ys[-int(period / 0.01)], dtype=jnp.float32)
    N = int(round(n_periods * period / dt))
    z0 = M.encode(params, cfg, x0[None, :])
    z_roll = M.rollout(params, cfg, z0, jnp.full((1, N), u), dt)[0]   # (N, 2m)
    z_all = np.array(jnp.concatenate([z0, z_roll], axis=0))           # (N+1, 2m)

    # per-mode contribution to the decoded state via the (linear) decoder
    if cfg.backbone == "spiral":
        Dw = np.array(params["dec"]["w"])          # (2m, 2)
        Db = np.array(params["dec"]["b"])
        contrib = np.einsum("tk,kc->tkc", z_all, Dw)   # (T, 2m, 2)
        # group into m blocks (mode j = coords 2j,2j+1)
        block = contrib.reshape(contrib.shape[0], cfg.m, 2, 2).sum(axis=2)  # (T, m, 2)
        recon = block.sum(axis=1) + Db
        # native block order (len m), aligned with `block` (mode j = coords 2j,2j+1)
        sg, og = M.spiral_eigs(params, cfg, z0, jnp.array([float(u)]))
        s, o = np.array(sg[0]), np.array(og[0])
    else:
        Amat = np.array(M.bilinear_matrices(params)[0]) + u * np.array(M.bilinear_matrices(params)[1])
        ev, V = np.linalg.eig(Amat)
        Vinv = np.linalg.inv(V)
        coords = z_all @ Vinv.T                     # modal coordinates (complex)
        # state contribution of each eigen-mode: Re( coords_j * V[:,j] )[:2]
        block = np.zeros((z_all.shape[0], len(ev), 2))
        for j in range(len(ev)):
            mode_state = np.outer(coords[:, j], V[:, j])[:, :2]
            block[:, j, :] = mode_state.real
        recon = block.sum(axis=1)
        # native order (len 2m), aligned with `block` (mode j = ev[j])
        s, o = ev.real, np.abs(ev.imag)

    # energy per mode (on v) and cumulative reconstruction R^2
    energy = np.sqrt((block[..., 0] ** 2).mean(axis=0))      # (m,)
    order = np.argsort(-energy)
    v_ref = block.sum(axis=1)[:, 0]    # bias-free full reconstruction (R^2 reference)
    v_full = recon[:, 0]               # actual decoded output (for plotting)
    cum = np.zeros(len(order))
    acc = np.zeros(recon.shape[0])
    for i, j in enumerate(order):
        acc = acc + block[:, j, 0]
        ss_res = np.sum((v_ref - acc) ** 2)
        ss_tot = np.sum((v_ref - v_ref.mean()) ** 2) + 1e-12
        cum[i] = 1 - ss_res / ss_tot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    t = np.arange(recon.shape[0]) * dt
    ax1.plot(t, v_full, "k", lw=2, label="full reconstruction")
    for i in order[:min(4, len(order))]:
        ax1.plot(t, block[:, i, 0], lw=1, alpha=.8,
                 label=f"mode {i}: ω≈{o[i] if i < len(o) else 0:.2f}, σ≈{s[i] if i < len(s) else 0:.2f}")
    ax1.set_xlabel("time"); ax1.set_ylabel("v contribution")
    ax1.set_title("Per-mode contribution to v(t)"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    ax2.plot(np.arange(1, len(cum) + 1), cum, "o-", color="#2ca02c")
    ax2.set_xlabel("number of modes (by energy)"); ax2.set_ylabel(r"cumulative $R^2$ on v")
    ax2.set_ylim(min(0, cum.min()) - .05, 1.05)
    ax2.set_title("Reconstruction vs #modes"); ax2.grid(alpha=.3)
    fig.suptitle(f"D3  Koopman-mode reconstruction  @ u={u}  [{tag}]", fontweight="bold")
    p = os.path.join(PLOTS, f"diag_D3_modes_{tag}.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)

    table = [{"mode": int(j), "omega": float(o[j] if j < len(o) else np.nan),
              "sigma": float(s[j] if j < len(s) else np.nan),
              "energy_v": float(energy[j])} for j in order]
    return {"plot": p, "u": u, "period": float(period),
            "modes_by_energy": table, "cum_R2": cum.tolist()}


# --------------------------------------------------------------------------- #
def diag_longhorizon_bands(params, cfg, dt, scales, data, tag, traj_idx=0):
    """D4 / fix (d): long-horizon rollout error decomposed by spectral band."""
    ys = jnp.asarray(data["ys_val"][traj_idx])         # (T, 2)
    u = jnp.asarray(data["u_val"][traj_idx])           # (T,)
    T = ys.shape[0]
    z0 = M.encode(params, cfg, ys[0][None, :])
    z_roll = M.rollout(params, cfg, z0, u[None, :T - 1], dt)[0]
    z_all = np.array(jnp.concatenate([z0, z_roll], axis=0))
    x_pred = np.array(M.decode(params, cfg, jnp.asarray(z_all)))
    x_true = np.array(ys)
    t = np.arange(T) * dt
    err = np.linalg.norm((x_pred - x_true) / np.array(scales["x"]), axis=1)

    # spectral-band energy of the residual via FFT of v-error
    ve = x_pred[:, 0] - x_true[:, 0]
    freqs = np.fft.rfftfreq(T, d=dt)
    pe = np.abs(np.fft.rfft(ve)) ** 2
    pt = np.abs(np.fft.rfft(x_true[:, 0])) ** 2
    w = 2 * np.pi * freqs
    bands = [("low (ω<0.15)", w < 0.15),
             ("mid (0.15-0.4)", (w >= 0.15) & (w < 0.4)),
             ("high (ω>=0.4)", w >= 0.4)]
    band_err = {name: float(pe[mask].sum()) for name, mask in bands}
    band_true = {name: float(pt[mask].sum()) for name, mask in bands}
    tot = sum(band_err.values()) + 1e-12

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ax1.plot(t, x_true[:, 0], color="#1f77b4", lw=2, label="true v")
    ax1.plot(t, x_pred[:, 0], "--", color="#2ca02c", lw=2, label="predicted v")
    ax1b = ax1.twinx(); ax1b.plot(t, err, color="#d62728", lw=1, alpha=.6)
    ax1b.set_ylabel("normalised error", color="#d62728")
    ax1.set_xlabel("time"); ax1.set_ylabel("v"); ax1.legend(loc="upper right")
    ax1.set_title("Long-horizon rollout (val/chirp)"); ax1.grid(alpha=.3)

    names = [b[0] for b in bands]
    ax2.bar(names, [band_err[n] / tot for n in names], color="#d62728", alpha=.7, label="residual share")
    ax2.set_ylabel("fraction of residual power")
    ax2.set_title("Error power by spectral band"); ax2.grid(alpha=.3, axis="y")
    for i, n in enumerate(names):
        ax2.text(i, band_err[n] / tot + .01, f"{band_err[n]/tot:.0%}", ha="center", fontsize=9)
    fig.suptitle(f"D4  Long-horizon error by spectral band  [{tag}]", fontweight="bold")
    p = os.path.join(PLOTS, f"diag_D4_bands_{tag}.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)

    nrmse = float(np.sqrt(np.mean(((x_pred[:, 0] - x_true[:, 0]) / float(scales["x"][0])) ** 2)))
    return {"plot": p, "nrmse_v": nrmse,
            "band_residual_fraction": {n: band_err[n] / tot for n in names},
            "band_true_power": band_true}


# --------------------------------------------------------------------------- #
def _rollout_metrics(params, cfg, dt, scales, ys, u, threshold=0.30):
    """Free-running rollout metrics for a batch of trajectories (no plotting)."""
    ys = jnp.asarray(ys); u = jnp.asarray(u)
    Bn, T, _ = ys.shape
    z0 = M.encode(params, cfg, ys[:, 0])
    z_roll = M.rollout(params, cfg, z0, u[:, :T - 1], dt)
    z_all = jnp.concatenate([z0[:, None, :], z_roll], axis=1)
    x_pred = np.array(M.decode(params, cfg, z_all))
    x_true = np.array(ys)
    sx = np.array(scales["x"])
    t = np.arange(T) * dt
    one_p = min(T, int(37.0 / dt))

    inst = np.abs(x_pred[..., 0] - x_true[..., 0]) / sx[0]
    nrmse_full = np.sqrt(np.mean(((x_pred - x_true) / sx) ** 2, axis=(1, 2)))
    nrmse_1p = np.sqrt(np.mean(((x_pred[:, :one_p] - x_true[:, :one_p]) / sx) ** 2, axis=(1, 2)))

    def valid_time(row):
        idx = np.where(row > threshold)[0]
        return t[idx[0]] if len(idx) else t[-1]
    vts = np.array([valid_time(inst[i]) for i in range(Bn)])

    def dom_freq_amp(sig):
        sig = sig - sig.mean()
        ps = np.abs(np.fft.rfft(sig)) ** 2
        fr = np.fft.rfftfreq(len(sig), d=dt)
        k = 1 + int(np.argmax(ps[1:]))                 # skip DC
        return fr[k], (sig.max() - sig.min())
    spec_ok = np.zeros(Bn, dtype=bool)
    for i in range(Bn):
        ft, at = dom_freq_amp(x_true[i, :, 0]); fp, ap = dom_freq_amp(x_pred[i, :, 0])
        spec_ok[i] = (abs(fp - ft) / (ft + 1e-9) < 0.25) and (abs(ap - at) / (at + 1e-9) < 0.35)

    return {"t": t, "inst": inst, "nrmse_full": nrmse_full, "nrmse_1p": nrmse_1p,
            "vts": vts, "spec_ok": spec_ok, "n": Bn,
            "succ_1p": float(np.mean(nrmse_1p < threshold)),
            "succ_full": float(np.mean(nrmse_full < threshold)),
            "spectral": float(spec_ok.mean()),
            "cheb": float(np.max(np.abs(x_pred[..., 0] - x_true[..., 0]), axis=1).mean())}


def _indist_sine_set(dt, n=16, t_max=80.0, seed=777):
    """Fresh in-distribution validation set: sine forcing, new ICs & amplitudes."""
    t = jnp.linspace(0.0, t_max, int(round(t_max / dt)) + 1)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    v0 = jax.random.uniform(k1, (n,), minval=-1.8, maxval=0.8)
    w0 = jax.random.uniform(k2, (n,), minval=-0.8, maxval=0.4)
    I = jax.random.uniform(k3, (n,), minval=0.4, maxval=1.2)
    ys = simulate_fhn_batch(jnp.stack([v0, w0], 1), t, 'sine', I)
    u = jax.vmap(lambda iv: get_external_current(t, 'sine', iv))(I)
    return ys, u


def success_rate(params, cfg, dt, scales, data, tag, threshold=0.30):
    """Free-running rollout success on two held-out sets:
       in-distribution (sine forcing, new ICs/amplitudes) and
       out-of-distribution (the unseen chirp forcing from the dataset).
    Reports horizon-resolved error, valid time, and pointwise + spectral success.
    """
    period = 37.0
    m_ood = _rollout_metrics(params, cfg, dt, scales, data["ys_val"], data["u_val"], threshold)
    ys_id, u_id = _indist_sine_set(dt)
    m_id = _rollout_metrics(params, cfg, dt, scales, ys_id, u_id, threshold)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.hist(m_id["nrmse_1p"], bins=10, color="#2ca02c", alpha=.75, edgecolor="k",
             label=f"in-dist sine (1P)  spec={m_id['spectral']:.0%}")
    ax1.hist(m_ood["nrmse_1p"], bins=10, color="#d62728", alpha=.55, edgecolor="k",
             label=f"OOD chirp (1P)  spec={m_ood['spectral']:.0%}")
    ax1.axvline(threshold, color="k", ls="--")
    ax1.set_xlabel("normalised RMSE over 1 period"); ax1.set_ylabel("# trajectories")
    ax1.set_title("Pointwise error distribution"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    for m_, c, lab in [(m_id, "#2ca02c", "in-dist sine"), (m_ood, "#d62728", "OOD chirp")]:
        ax2.plot(m_["t"], m_["inst"].mean(axis=0), color=c, lw=2, label=lab)
    ax2.axhline(threshold, color="k", ls="--", lw=.8)
    ax2.axvline(period, color="0.5", ls=":", label="1 period")
    ax2.set_xlabel("prediction time"); ax2.set_ylabel("mean normalised error on v")
    ax2.set_title(f"valid time: in-dist {m_id['vts'].mean():.1f}, OOD {m_ood['vts'].mean():.1f}")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    fig.suptitle(f"Free-running rollout success  [{tag}]", fontweight="bold")
    p = os.path.join(PLOTS, f"diag_success_{tag}.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)

    def pack(m_):
        return {"spectral_success": m_["spectral"], "success_1period": m_["succ_1p"],
                "success_2period": m_["succ_full"], "nrmse_1period_mean": float(m_["nrmse_1p"].mean()),
                "nrmse_full_mean": float(m_["nrmse_full"].mean()),
                "valid_time_mean": float(m_["vts"].mean()), "cheb_mean": m_["cheb"], "n": m_["n"]}

    return {"plot": p, "threshold": threshold,
            "in_distribution_sine": pack(m_id), "out_of_distribution_chirp": pack(m_ood),
            # convenience top-level (OOD, the harder test) for backwards-compat
            "spectral_success": m_ood["spectral"], "success_rate": m_ood["succ_full"],
            "success_rate_1period": m_ood["succ_1p"]}


# --------------------------------------------------------------------------- #
def run_all(model_path, data_path=None):
    data_path = data_path or os.path.join(HERE, "data", "fhn_koopman_t80.npz")
    params, cfg, scales, dt, out = load_model(model_path)
    data = np.load(data_path)
    tag = cfg.backbone
    os.makedirs(PLOTS, exist_ok=True)
    print(f"=== diagnostics for {tag} (m={cfg.m}) ===")

    results = {"backbone": tag, "m": cfg.m, "dt": dt}
    results["D0_spectrum"] = diag_spectrum_vs_jacobian(params, cfg, tag)
    print("  D0 spectrum-vs-Jacobian done")
    results["D1_radius"] = diag_eigs_over_radius(params, cfg, tag)
    print("  D1 eigenvalues-over-radius done")
    results["D2_floquet"] = diag_floquet(params, cfg, dt, tag)
    print("  D2 Floquet done")
    results["D3_modes"] = diag_koopman_modes(params, cfg, dt, tag)
    print("  D3 Koopman modes done")
    results["D4_bands"] = diag_longhorizon_bands(params, cfg, dt, scales, data, tag)
    print("  D4 long-horizon bands done")
    results["success"] = success_rate(params, cfg, dt, scales, data, tag)
    print(f"  success rate = {results['success']['success_rate']:.0%}")

    with open(os.path.join(HERE, "data", f"diag_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  saved data/diag_{tag}.json")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(HERE, "data", "koopman_spiral.pkl"))
    p.add_argument("--data", default=None)
    args = p.parse_args()
    run_all(args.model, args.data)


if __name__ == "__main__":
    main()
