"""Generate report figures for the PWFO surrogate (-> plots/pwfo/)."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pwfo_model as P
from pwfo_eval import load
from pwfo_train import firing_const_data

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "plots", "pwfo")
os.makedirs(ART, exist_ok=True)
DT = 0.05


def fig_concept(core):
    params, cfg, dt = core
    t, ys, u = firing_const_data(2, 80.0, dt, seed=3)
    tq = jnp.asarray((np.arange(0, ys.shape[1]) * dt).astype(np.float32))[None].repeat(2, 0)
    c = P._mlp(P._profile_stats(jnp.asarray(u)), params["ctx"])
    om, _ = P.segment_rates(params, cfg, jnp.asarray(u), c)
    Phi = np.array(dt * jnp.cumsum(om, axis=1))
    tt = np.arange(ys.shape[1]) * dt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(tt, Phi[0], "C0", lw=2, label="Φ(t) (cumulative phase)")
    ax[0].set_xlabel("time t"); ax[0].set_ylabel("Φ(t)")
    ax[0].set_title("Phase = prefix-sum of instantaneous ω → linear, evaluable at any t")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].plot(tt, np.cos(Phi[0]), "C3", lw=1.2)
    ax[1].set_xlabel("time t"); ax[1].set_ylabel("cos Φ(t)")
    ax[1].set_title("cos(kΦ) is bounded & periodic ∀ t → oscillation never decays/blows up")
    ax[1].grid(alpha=.3)
    fig.suptitle("Why PWFO evaluates any t in one shot with no recursion", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/fig1_concept.png", dpi=140); plt.close(fig)
    print("fig1_concept")


def fig_core(core):
    params, cfg, dt = core
    t, ys, u = firing_const_data(6, 1500.0, dt, seed=99)
    S = ys.shape[1]
    qidx = np.arange(0, S)
    tq = jnp.asarray((qidx * dt).astype(np.float32))[None].repeat(6, 0)
    xh = np.array(P.forward(params, cfg, jnp.asarray(ys[:, 0]), jnp.asarray(u), tq, dt))
    i = 0
    fig, ax = plt.subplots(2, 1, figsize=(13, 6))
    for axx, (lo, hi, lab) in zip(ax, [(0, 80, "trained window (t<300)"),
                                       (1420, 1500, "EXTRAPOLATION (t≈1450, one-shot)")]):
        m = (t >= lo) & (t < hi)
        axx.plot(t[m], ys[i, m, 0], "k", lw=2.2, label="true FHN")
        axx.plot(t[m], xh[i, m, 0], "C2--", lw=1.6, label="PWFO (1 forward pass)")
        axx.set_ylabel("v"); axx.set_title(lab); axx.legend(fontsize=8); axx.grid(alpha=.3)
    ax[-1].set_xlabel("time")
    fig.suptitle("Constant-current core: sustained oscillation 30+ cycles past training, no recursion",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/fig2_core_farhorizon.png", dpi=140); plt.close(fig)

    ps = int(round(2 * np.pi / 0.155 / dt))
    fig, a = plt.subplots(figsize=(11, 4))
    for arr, c_, lab in [(ys[i, :, 0], "k", "true"), (xh[i, :, 0], "C2", "PWFO")]:
        amps = [np.ptp(arr[k*ps:(k+1)*ps]) for k in range(len(arr)//ps)]
        a.plot(np.arange(len(amps)), amps, c_+"-o", ms=3, label=lab)
    a.axvline(300/ (ps*dt), color="0.6", ls=":", label="train horizon")
    a.set_xlabel("period #"); a.set_ylabel("v amplitude (peak-to-peak)")
    a.set_ylim(bottom=0); a.legend(); a.grid(alpha=.3)
    a.set_title("Amplitude stays flat over 30+ extrapolated cycles (no decay, no divergence)")
    fig.tight_layout(); fig.savefig(f"{ART}/fig3_amplitude_flat.png", dpi=140); plt.close(fig)
    print("fig2_core_farhorizon, fig3_amplitude_flat")


def fig_general(gen):
    if gen is None:
        return
    from operator_data import KINDS
    params, cfg, dt = gen
    d = np.load(os.path.join(HERE, "data", "fhn_operator.npz"))
    ys = np.asarray(d["ys_val"]); u = np.asarray(d["u_val"]); S = ys.shape[1]
    W = int(3 * 37.0 / dt); t0 = (S - W) // 2
    picks = [("ramp", "fig4_general_slow.png", "design regime (slow current) — good"),
             ("chirp", "fig5_general_fast.png", "stress regime (fast current) — adiabatic limit")]
    for kind, fn, sub in picks:
        i = next(j for j in range(ys.shape[0]) if KINDS[j % len(KINDS)] == kind)
        uw = u[i, t0:t0 + W][None]; x0 = ys[i, t0][None]
        tq = jnp.asarray((np.arange(W) * dt).astype(np.float32))[None]
        xh = np.array(P.forward(params, cfg, jnp.asarray(x0), jnp.asarray(uw), tq, dt))[0]
        tt = (t0 + np.arange(W)) * dt
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(tt, u[i, t0:t0 + W], "C4", lw=1.3); ax[0].set_ylabel("I_ext(t)")
        ax[0].set_title(f"Anchored 3-cycle one-shot from a measured state — {sub}")
        ax[0].grid(alpha=.3)
        ax[1].plot(tt, ys[i, t0:t0 + W, 0], "k", lw=2.2, label="true FHN")
        ax[1].plot(tt, xh[:, 0], "C2--", lw=1.5, label="PWFO (one forward pass)")
        ax[1].set_ylabel("v"); ax[1].set_xlabel("time"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
        fig.suptitle("Operator surrogate G(x0, I_ext(.), t): measured state -> next 3 cycles, one shot",
                     fontweight="bold")
        fig.tight_layout(); fig.savefig(f"{ART}/{fn}", dpi=140); plt.close(fig)
        print(fn)


def main():
    core = load(os.path.join(HERE, "data", "pwfo_core_k20.pkl"))
    fig_concept(core)
    fig_core(core)
    gen_path = os.path.join(HERE, "data", "pwfo_general.pkl")
    gen = load(gen_path) if os.path.exists(gen_path) else None
    fig_general(gen)
    print("FIGURES ->", ART)


if __name__ == "__main__":
    main()
