"""Far-horizon ONE-SHOT evaluation of PWFO: does a single forward pass keep the
oscillation correct (amplitude/waveform flat, phase drift bounded) at t far beyond
the training horizon, with NO recursion?"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

import pwfo_model as P
from pwfo_train import firing_const_data


def load(path):
    with open(path, "rb") as f:
        o = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, o["params"])
    c = o["cfg"]
    cfg = P.PWFOConfig(d=c["d"], K=c["K"], m=c["m"],
                       local_waveform=c.get("local_waveform", False))
    return params, cfg, float(o["dt"])


def amp_per_period(v, ps):
    n = len(v) // ps
    return np.array([np.ptp(v[k * ps:(k + 1) * ps]) for k in range(n)])


def phase_lag(a, b, max_lag):
    """Best alignment lag (samples) between a and b via cross-correlation."""
    a = a - a.mean(); b = b - b.mean()
    best, bl = -1e18, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.dot(a[lag:], b[:len(b) - lag]) if lag < len(a) else 0.0
        else:
            c = np.dot(a[:len(a) + lag], b[-lag:])
        if c > best:
            best, bl = c, lag
    return bl


def anchored_eval(params, cfg, dt, data_path, cycles=3, n=48, seed=5):
    """Operator use-case metric: from x0 at a random anchor t0, predict the next
    `cycles` cycles in one shot; NRMSE vs truth."""
    d = np.load(data_path)
    ys = np.asarray(d["ys_val"]); u = np.asarray(d["u_val"]); S = ys.shape[1]
    W = int(cycles * 37.0 / dt)
    rng = np.random.default_rng(seed)
    sx = ys.reshape(-1, 2).std(0) + 1e-6
    errs = []
    for i in range(min(n, ys.shape[0])):
        t0 = int(rng.integers(0, S - W - 1))
        u_win = u[i, t0:t0 + W][None]
        x0 = ys[i, t0][None]
        qoff = np.arange(0, W)
        tq = jnp.asarray((qoff * dt).astype(np.float32))[None]
        xh = np.asarray(P.forward(params, cfg, jnp.asarray(x0), jnp.asarray(u_win), tq, dt))[0]
        tgt = ys[i, t0:t0 + W]
        errs.append(np.sqrt((((xh - tgt) / sx) ** 2).mean()))
    errs = np.array(errs)
    print(f"  anchored {cycles}-cycle one-shot NRMSE: mean={errs.mean():.3f} "
          f"median={np.median(errs):.3f} (from random measured-state anchors)")
    return errs


def general_eval(params, cfg, dt, data_path, n=24):
    """One-shot eval on time-varying profiles: in-horizon vs far extrapolation."""
    d = np.load(data_path)
    out = {}
    for tag, yk, uk, tk in [("in_horizon", "ys_val", "u_val", "t"),
                            ("far_extrap", "ys_far", "u_far", "t_far")]:
        ys = np.asarray(d[yk])[:n]; u = np.asarray(d[uk])[:n]; t = np.asarray(d[tk])
        S = ys.shape[1]
        qidx = np.arange(0, S, 4)
        tq = jnp.asarray((qidx * dt).astype(np.float32))[None, :].repeat(ys.shape[0], 0)
        xh = np.asarray(P.forward(params, cfg, jnp.asarray(ys[:, 0]), jnp.asarray(u),
                                  tq, dt))
        true = ys[:, qidx]
        sx = ys.reshape(-1, 2).std(0) + 1e-6
        nrmse = np.sqrt((((xh - true) / sx) ** 2).mean())
        finite = bool(np.isfinite(xh).all())
        out[tag] = (nrmse, finite, t[-1])
        print(f"  {tag:11s}: one-shot NRMSE={nrmse:.3f} finite={finite} t_max={t[-1]:.0f}")
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="PWFO one-shot eval")
    p.add_argument("--model", default=os.path.join(here, "data", "pwfo_core_k20.pkl"))
    p.add_argument("--mode", choices=["core", "general"], default="core")
    p.add_argument("--data", default=os.path.join(here, "data", "fhn_operator.npz"))
    p.add_argument("--t-max-far", type=float, default=1500.0)
    p.add_argument("--n", type=int, default=24)
    args = p.parse_args()

    params, cfg, dt = load(args.model)
    if args.mode == "general":
        print(f"=== PWFO general one-shot eval ({cfg}) ===")
        general_eval(params, cfg, dt, args.data, n=args.n)
        anchored_eval(params, cfg, dt, args.data, cycles=3)
        return
    t, ys, u = firing_const_data(args.n, args.t_max_far, dt, seed=99)
    S = ys.shape[1]
    ps = int(round(2 * np.pi / 0.155 / dt))

    qidx = np.arange(0, S, 2)
    tq = jnp.asarray((qidx * dt).astype(np.float32))[None, :].repeat(args.n, 0)
    x0 = jnp.asarray(ys[:, 0])
    ug = jnp.asarray(u)
    f = jax.jit(lambda: P.forward(params, cfg, x0, ug, tq, dt))
    xh = np.asarray(f());
    true = ys[:, qidx]

    early = qidx * dt < 300.0
    far = qidx * dt >= 1000.0
    sx = ys.reshape(-1, 2).std(0) + 1e-6
    nrmse_early = np.sqrt((((xh - true) / sx) ** 2)[:, early].mean())
    nrmse_far = np.sqrt((((xh - true) / sx) ** 2)[:, far].mean())

    psq = max(2, ps // 2)
    amp_t = np.array([np.ptp(true[:, far][:, k * psq:(k + 1) * psq, 0], axis=1).mean()
                      for k in range(far.sum() // psq)])
    amp_p = np.array([np.ptp(xh[:, far][:, k * psq:(k + 1) * psq, 0], axis=1).mean()
                      for k in range(far.sum() // psq)])
    amp_t_early = np.array([np.ptp(true[:, early][:, k * psq:(k + 1) * psq, 0], axis=1).mean()
                           for k in range(early.sum() // psq)])
    amp_p_early = np.array([np.ptp(xh[:, early][:, k * psq:(k + 1) * psq, 0], axis=1).mean()
                           for k in range(early.sum() // psq)])

    far_idx = np.where(far)[0]
    seg = far_idx[:ps * 4]
    lag = phase_lag(xh[0, seg, 0], true[0, seg, 0], max_lag=ps)
    aligned = np.sqrt(((xh[0, seg[:-abs(lag) or None], 0]
                        - true[0, seg[abs(lag):] if lag >= 0 else seg[:lag], 0]) / sx[0]) ** 2).mean() \
        if lag != 0 else nrmse_far

    print(f"=== PWFO far-horizon ONE-SHOT eval ({cfg}) ===")
    print(f"trained horizon t<=300; queried in ONE pass out to t={args.t_max_far}")
    print(f"  amplitude (v ptp): true~{amp_t.mean():.2f}  pred~{amp_p.mean():.2f}  "
          f"(early true~{amp_t_early.mean():.2f} pred~{amp_p_early.mean():.2f})")
    print(f"  amplitude flatness pred far/early = {amp_p.mean()/(amp_p_early.mean()+1e-9):.3f} "
          f"(==1 => no decay/growth over 30+ extra cycles)")
    print(f"  NRMSE pointwise: early(trained)={nrmse_early:.3f}  far(extrapolated)={nrmse_far:.3f}")
    print(f"  far phase lag ~{lag*dt:.2f} time-units; phase-aligned far NRMSE~{aligned:.3f}")

    n_t = 200
    big = jnp.asarray([[1.0]] * args.n) * 0 + jnp.linspace(1, 5, n_t)[None, :]
    j1 = jax.jit(lambda tt: P.forward(params, cfg, x0, ug, tt, dt))
    a = jnp.asarray([[1.0, 2.0]] * args.n); j1(a).block_until_ready()
    b = jnp.asarray([[1e6, 1e6 + 1.0]] * args.n); j1(b).block_until_ready()
    t0 = time.time(); [j1(a).block_until_ready() for _ in range(50)]; ta = (time.time() - t0) / 50
    t0 = time.time(); [j1(b).block_until_ready() for _ in range(50)]; tb = (time.time() - t0) / 50
    print(f"  timing invariance: t~1 -> {ta*1e3:.3f} ms ; t~1e6 -> {tb*1e3:.3f} ms "
          f"(equal => no hidden recursion)")


if __name__ == "__main__":
    main()
