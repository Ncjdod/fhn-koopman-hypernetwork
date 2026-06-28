"""Generate all figures for the educational article (-> plots/article/)."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "plots", "article")
os.makedirs(ART, exist_ok=True)
DT = 0.05
PERIOD_STEPS = int(round(37.0 / DT))

A, B, TAU = 0.7, 0.8, 12.5

def fhn_rhs(v, w, u):
    return v - v ** 3 / 3.0 - w + u, (v + A - B * w) / TAU

def sim_true(u, t_max, x0=(-1.0, -0.5), dt=DT):
    n = int(round(t_max / dt))
    v, w = x0
    out = np.empty((n + 1, 2)); out[0] = (v, w)
    for i in range(n):
        k1 = fhn_rhs(v, w, u)
        k2 = fhn_rhs(v + .5*dt*k1[0], w + .5*dt*k1[1], u)
        k3 = fhn_rhs(v + .5*dt*k2[0], w + .5*dt*k2[1], u)
        k4 = fhn_rhs(v + dt*k3[0], w + dt*k3[1], u)
        v += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        w += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        out[i+1] = (v, w)
    return out

def amp_per_period(v, ps=PERIOD_STEPS):
    n = len(v) // ps
    return np.array([np.ptp(v[k*ps:(k+1)*ps]) for k in range(n)])

def load_models():
    import jax
    import model as M
    import sindyc as S
    with open(os.path.join(HERE, "data", "koopman_spiral.pkl"), "rb") as f:
        sl = pickle.load(f)
    params = jax.tree_util.tree_map(jax.numpy.asarray, sl["params"])
    cfg = M.ModelConfig(backbone=sl["cfg"]["backbone"], m=sl["cfg"]["m"],
                        radial=sl["cfg"].get("radial", "stuart_landau"))
    sl_pack = (params, cfg, sl)
    sd = np.load(os.path.join(HERE, "data", "sindyc_model.npz"), allow_pickle=True)
    sind = S.SINDYcModel(sd["Xi"], degree=int(sd["degree"]),
                         control_cross=bool(sd["control_cross"]), dt=float(sd["dt"]))
    return sl_pack, sind, M

def sl_rollout(M, params, cfg, x0, u_seq):
    import jax.numpy as jnp
    x0 = jnp.asarray(np.atleast_2d(x0), dtype=jnp.float32)
    u_seq = jnp.asarray(np.atleast_2d(u_seq), dtype=jnp.float32)
    z0 = M.encode(params, cfg, x0)
    zr = M.rollout(params, cfg, z0, u_seq, DT)
    x = M.decode(params, cfg, jnp.concatenate([z0[:, None], zr], axis=1))
    return np.array(x)

def fig_phase_portrait():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, u, title in [(axes[0], 0.0, "u = 0.0  (quiescent: spiral to fixed point)"),
                         (axes[1], 0.5, "u = 0.5  (firing: attracting limit cycle)")]:
        vv = np.linspace(-2.5, 2.5, 400)
        ax.plot(vv, vv - vv**3/3 + u, "C0", lw=2, label=r"$\dot v=0$ nullcline")
        ax.plot(vv, (vv + A) / B, "C1", lw=2, label=r"$\dot w=0$ nullcline")
        for x0 in [(-1.5, -0.6), (1.5, 0.4), (0.0, 0.8), (-2.0, 0.0)]:
            tr = sim_true(u, 200, x0)
            ax.plot(tr[:, 0], tr[:, 1], "0.6", lw=.8, alpha=.8)
        tr = sim_true(u, 400, (-1.0, -0.5))
        ax.plot(tr[len(tr)//2:, 0], tr[len(tr)//2:, 1], "C3", lw=2.5,
                label="attractor")
        ax.set_xlabel("v (membrane potential)"); ax.set_ylabel("w (recovery)")
        ax.set_title(title); ax.legend(fontsize=8); ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.2, 1.6); ax.grid(alpha=.3)
    fig.suptitle("FitzHugh-Nagumo phase portrait: a Hopf bifurcation creates the limit cycle",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/fig1_phase_portrait.png", dpi=140); plt.close(fig)
    print("fig1_phase_portrait")

def fig_hopf():
    import fhn_theory as TH
    ug = np.linspace(0.0, 1.8, 60)
    sp = TH.spectrum_over_u(ug)
    amps = []
    for u in ug:
        v = sim_true(u, 8*37.0)[:, 0]
        amps.append(np.ptp(v[len(v)//2:]))
    amps = np.array(amps)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].plot(ug, sp["sigma"], "C0", lw=2)
    ax[0].fill_between(ug, 0, sp["sigma"], where=sp["sigma"] > 0, color="C3", alpha=.25,
                       label="unstable focus (fires)")
    ax[0].set_xlabel("injected current u"); ax[0].set_ylabel(r"Re $\mu(u)$ (focus eigenvalue)")
    ax[0].set_title("Linear stability of the rest state"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(ug, amps, "C3", lw=2)
    ax[1].set_xlabel("injected current u"); ax[1].set_ylabel("limit-cycle amplitude (v peak-to-peak)")
    ax[1].set_title("Cycle is born at the Hopf point, dies at the second"); ax[1].grid(alpha=.3)
    fig.suptitle("The Hopf bifurcation: where the neuron starts and stops firing",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/fig2_hopf.png", dpi=140); plt.close(fig)
    print("fig2_hopf")

def fig_obstruction():
    r = np.linspace(0, 2.2, 300)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for sig in [-0.3, -0.05, 0.0]:
        ax[0].plot(r, sig*r, label=f"σ={sig}")
    ax[0].axhline(0, color="k", lw=.6); ax[0].plot(0, 0, "ko")
    ax[0].set_title("Linear / clamped law  $\\dot r=\\sigma r$, σ≤0\n→ only the dead state r=0 attracts")
    ax[0].set_xlabel("radius r"); ax[0].set_ylabel(r"$\dot r$"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    beta = 0.8
    for s0 in [-0.3, 0.3, 0.8]:
        dr = s0*r - beta*r**3
        ax[1].plot(r, dr, label=f"σ₀={s0}")
        if s0 > 0:
            ax[1].plot(np.sqrt(s0/beta), 0, "C3o", ms=9)
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].set_title("Stuart-Landau law  $\\dot r=\\sigma_0 r-\\beta r^3$\n→ stable cycle at r*=√(σ₀/β) (red dots)")
    ax[1].set_xlabel("radius r"); ax[1].set_ylabel(r"$\dot r$"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle("Why a linear operator cannot hold a limit cycle — and how the cubic term fixes it",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/fig3_obstruction.png", dpi=140); plt.close(fig)
    print("fig3_obstruction")

def fig_sl_training(sl):
    h = sl["history"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, c in [("total", "k"), ("rec", "C0"), ("lin", "C1"), ("pred", "C3")]:
        ax.semilogy(np.asarray(h[k]) + 1e-6, c, lw=1.3, label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title("Stuart-Landau Koopman training (curriculum: horizon grows to a full cycle)")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{ART}/fig4_sl_training.png", dpi=140); plt.close(fig)
    print("fig4_sl_training")

def fig_rollouts(M, params, cfg, sind, currents=(0.5, 0.8)):
    T = 3 * PERIOD_STEPS
    fig, axes = plt.subplots(len(currents), 1, figsize=(12, 3.2*len(currents)), squeeze=False)
    t = np.arange(T + 1) * DT
    for i, u in enumerate(currents):
        true = sim_true(u, T*DT)[:, 0]
        usl = np.full((1, T), u)
        sl = sl_rollout(M, params, cfg, (-1.0, -0.5), usl)[0, :, 0]
        sd = sind.simulate((-1.0, -0.5), np.full(T, u))[0, :, 0]
        ax = axes[i, 0]
        ax.plot(t[:len(true)], true, "k", lw=2.2, label="true FHN")
        ax.plot(t[:len(sd)], sd, "C2--", lw=1.6, label="SINDYc")
        ax.plot(t[:len(sl)], sl, "C3", lw=1.1, alpha=.85, label="SL-Koopman")
        ax.set_ylabel("v"); ax.set_title(f"Free-running rollout, constant u={u} (3 periods)")
        ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.3)
    axes[-1, 0].set_xlabel("time")
    fig.suptitle("Generating dynamics from a cold start: SINDYc tracks; SL keeps the right cycle but drifts in phase",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/fig5_rollouts.png", dpi=140); plt.close(fig)
    print("fig5_rollouts")

def fig_long_horizon(M, params, cfg, sind, u=0.7, periods=25):
    T = periods * PERIOD_STEPS
    true = sim_true(u, T*DT)[:, 0]
    sl = sl_rollout(M, params, cfg, (-1.0, -0.5), np.full((1, T), u))[0, :, 0]
    sd = sind.simulate((-1.0, -0.5), np.full(T, u))[0, :, 0]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(amp_per_period(true), "k-o", ms=3, label="true")
    ax.plot(amp_per_period(sd), "C2-s", ms=3, label="SINDYc")
    ax.plot(amp_per_period(sl), "C3-^", ms=3, label="SL-Koopman")
    ax.axhline(0, color="0.7", lw=.8)
    ax.set_xlabel("period #"); ax.set_ylabel("cycle amplitude (v peak-to-peak)")
    ax.set_title(f"Long-horizon amplitude stability over {periods} periods, u={u}\n"
                 "(the goal: neither decay to 0 nor diverge)")
    ax.legend(); ax.grid(alpha=.3); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(f"{ART}/fig6_long_horizon.png", dpi=140); plt.close(fig)
    print("fig6_long_horizon")

def fig_sindyc_coeffs(sind):
    true_map = {"v": 1.0, "vvv": -1/3, "w": -1.0, "u": 1.0,
                "1": 0.0, "v_w": 0.0}
    names = sind.names
    truth_v = {"v": 1.0, "vvv": -1/3, "w": -1.0, "u": 1.0}
    truth_w = {"1": A/TAU, "v": 1/TAU, "w": -B/TAU}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, k, truth, lab in [(ax[0], 0, truth_v, "v̇"), (ax[1], 1, truth_w, "ẇ")]:
        idx = [i for i in range(len(names)) if abs(sind.Xi[i, k]) > 1e-3 or names[i] in truth]
        xs = np.arange(len(idx)); w = .38
        a.bar(xs - w/2, [sind.Xi[i, k] for i in idx], w, label="discovered", color="C2")
        a.bar(xs + w/2, [truth.get(names[i], 0.0) for i in idx], w, label="true FHN", color="0.5")
        a.set_xticks(xs); a.set_xticklabels([names[i] for i in idx], rotation=45, ha="right")
        a.set_title(f"{lab} coefficients"); a.legend(); a.grid(alpha=.3, axis="y")
        a.axhline(0, color="k", lw=.6)
    fig.suptitle("SINDYc recovers the FitzHugh-Nagumo equations from data (discovered vs true)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{ART}/fig7_sindyc_coeffs.png", dpi=140); plt.close(fig)
    print("fig7_sindyc_coeffs")

def fig_ood_compare(M, params, cfg, sind):
    d = np.load(os.path.join(HERE, "data", "fhn_koopman_t80.npz"))
    yv, uv = d["ys_val"], d["u_val"]
    i = 0
    T = yv.shape[1]
    true = np.asarray(yv[i, :, 0])
    sl = sl_rollout(M, params, cfg, yv[i, 0], uv[i:i+1, :T-1])[0, :, 0]
    sd = sind.simulate(yv[i, 0], uv[i, :T-1])[0, :, 0]
    t = np.arange(T) * DT
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, uv[i], "C4", lw=1.3); ax[0].set_ylabel("u(t) chirp (OOD)")
    ax[0].set_title("Out-of-distribution test: a swept-frequency current never seen in training")
    ax[0].grid(alpha=.3)
    ax[1].plot(t, true, "k", lw=2.2, label="true FHN")
    ax[1].plot(t, sd, "C2--", lw=1.5, label="SINDYc")
    ax[1].plot(t, sl, "C3", lw=1.0, alpha=.8, label="SL-Koopman")
    ax[1].set_ylabel("v"); ax[1].set_xlabel("time"); ax[1].legend(fontsize=9, ncol=3); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{ART}/fig8_ood_compare.png", dpi=140); plt.close(fig)
    print("fig8_ood_compare")

def fig_inverse(sind):
    target = sim_true(0.6, 2.5*37.0)[1:]
    x0 = np.array([-1.0, -0.5])
    xdot_des = sind.f(target[:-1])[:, 0] * 0
    xdot_tgt = np.gradient(target[:, 0], DT)
    u_rec = sind.invert_onestep(target[:-1], xdot_tgt[:-1])
    ach = sind.simulate(x0, u_rec)[0, :, 0]
    t = np.arange(len(target)) * DT
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, u_rec.tolist() + [u_rec[-1]], "C4", lw=1.3)
    ax[0].set_ylabel("recovered I_ext"); ax[0].set_title("Inverse design: current recovered closed-form to realise a target spike train")
    ax[0].grid(alpha=.3)
    ax[1].plot(t, target[:, 0], "k", lw=2.2, label="target v")
    ax[1].plot(np.arange(len(ach))*DT, ach, "C2--", lw=1.5, label="achieved (forward-sim of recovered u)")
    ax[1].set_ylabel("v"); ax[1].set_xlabel("time"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{ART}/fig9_inverse.png", dpi=140); plt.close(fig)
    print("fig9_inverse")

def main():
    fig_phase_portrait()
    fig_hopf()
    fig_obstruction()
    (params, cfg, sl), sind, M = load_models()
    fig_sl_training(sl)
    fig_sindyc_coeffs(sind)
    fig_rollouts(M, params, cfg, sind)
    fig_long_horizon(M, params, cfg, sind)
    fig_ood_compare(M, params, cfg, sind)
    fig_inverse(sind)
    print("ALL FIGURES ->", ART)

if __name__ == "__main__":
    main()
