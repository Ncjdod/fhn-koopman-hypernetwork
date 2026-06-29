"""Train the Phase-Warped Floquet Operator surrogate (non-recursive, arbitrary-t)."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import pwfo_model as P
from operator_data import generate as gen_operator


def firing_const_data(n, t_max, dt, seed, u_lo=0.4, u_hi=1.35):
    """Constant firing-current trajectories (the de-risk core)."""
    from operator_data import simulate_batch
    t = np.arange(0.0, t_max + dt * 0.5, dt).astype(np.float32)
    rng = np.random.default_rng(seed)
    u_vals = rng.uniform(u_lo, u_hi, n).astype(np.float32)
    U = np.repeat(u_vals[:, None], len(t), axis=1)
    v0 = rng.uniform(-2.0, 2.0, n); w0 = rng.uniform(-1.0, 1.0, n)
    y0 = np.stack([v0, w0], 1).astype(np.float32)
    ys = np.asarray(simulate_batch(jnp.asarray(y0), jnp.asarray(t), jnp.asarray(U)))
    return t, ys.astype(np.float32), U


def load_freq_table(path):
    d = np.load(path)
    u, om = d["u"], d["omega"]
    fire = np.isfinite(om)
    return (jnp.asarray(u), jnp.asarray(np.where(fire, om, 0.0)),
            float(u[fire].min()), float(u[fire].max()))


def train(ys, u, t_grid, dt, freq_table, cfg, steps=4000, batch=64, q=256,
          lr=2e-3, w_freq=1.0, w_range=0.3, window=4000, seed=0, verbose=True):
    ys = jnp.asarray(ys); u = jnp.asarray(u)
    N, S, d = ys.shape
    W = int(min(window, S - 1))
    t0_max = max(1, S - W)
    sx = jnp.std(ys.reshape(-1, d), axis=0) + 1e-6
    u_tab, om_tab, u_fire_lo, u_fire_hi = freq_table
    win = jnp.arange(W)

    params = P.init_pwfo(cfg, jax.random.PRNGKey(seed))
    sched = optax.cosine_decay_schedule(lr, steps, alpha=1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-5))
    st = opt.init(params)

    def loss_fn(p, idx, t0, qoff):
        off = t0[:, None] + win[None, :]
        u_win = jnp.take_along_axis(u[idx], off, axis=1)
        ys_win = jnp.take_along_axis(ys[idx], off[:, :, None].repeat(d, 2), axis=1)
        x0 = ys_win[:, 0]
        tq = qoff.astype(jnp.float32) * dt
        tgt = jnp.take_along_axis(ys_win, qoff[:, :, None].repeat(d, 2), axis=1)
        xh = P.forward(p, cfg, x0, u_win, tq, dt)
        w_pt = 1.0 + 2.0 * jnp.abs((tgt - tgt.mean(1, keepdims=True)) / sx)
        l_state = jnp.mean(w_pt * ((xh - tgt) / sx) ** 2)
        rng_p = xh.max(1) - xh.min(1)
        rng_t = tgt.max(1) - tgt.min(1)
        l_range = jnp.mean(((rng_p - rng_t) / sx) ** 2)
        c = P._mlp(P._profile_stats(u_win), p["ctx"])
        om_pred, _ = P.segment_rates(p, cfg, u_win, c)
        om_tgt = jnp.interp(u_win, u_tab, om_tab)
        mask = (u_win >= u_fire_lo) & (u_win <= u_fire_hi)
        l_freq = jnp.sum(((om_pred - om_tgt) ** 2) * mask) / (jnp.sum(mask) + 1e-6)
        return l_state + w_freq * l_freq + w_range * l_range, (l_state, l_freq, l_range)

    @jax.jit
    def step(p, st, key):
        ki, kt, kq, kl = jax.random.split(key, 4)
        idx = jax.random.randint(ki, (batch,), 0, N)
        t0 = jax.random.randint(kt, (batch,), 0, t0_max)
        q_unif = jax.random.randint(kq, (batch, q // 2), 0, W)
        q_late = jax.random.randint(kl, (batch, q - q // 2), W // 2, W)
        qoff = jnp.concatenate([q_unif, q_late], axis=1)
        (tot, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, idx, t0, qoff)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, tot, aux

    key = jax.random.PRNGKey(seed + 1)
    t0 = time.time()
    for i in range(steps):
        key, k = jax.random.split(key)
        params, st, tot, aux = step(params, st, k)
        if verbose and (i % max(1, steps // 20) == 0 or i == steps - 1):
            print(f"step {i:5d} | tot {float(tot):.4f} state {float(aux[0]):.4f} "
                  f"freq {float(aux[1]):.5f} range {float(aux[2]):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return params, {"sx": np.array(sx)}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Train PWFO surrogate")
    p.add_argument("--mode", choices=["core", "general"], default="core")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--window", type=int, default=2400)
    p.add_argument("--freq-table", default=os.path.join(here, "data", "fhn_freq_table.npz"))
    p.add_argument("--data", default=os.path.join(here, "data", "fhn_operator.npz"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ftab = load_freq_table(args.freq_table)
    dt = 0.05
    if args.mode == "core":
        t, ys, u = firing_const_data(384, 300.0, dt, seed=7)
        print(f"core data: {ys.shape} constant firing currents")
    else:
        d = np.load(args.data)
        t, ys, u, dt = d["t"], d["ys_train"], d["u_train"], float(d["dt"])
        print(f"general data: {ys.shape}")

    cfg = P.PWFOConfig(d=2, K=args.K, m=args.m,
                       local_waveform=(args.mode == "general"))
    params, extra = train(ys, u, t, dt, ftab, cfg, steps=args.steps, window=args.window)
    out = args.out or os.path.join(here, "data", f"pwfo_{args.mode}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"params": jax.tree_util.tree_map(np.array, params),
                     "cfg": {"d": cfg.d, "K": cfg.K, "m": cfg.m, "p": cfg.p,
                             "local_waveform": cfg.local_waveform},
                     "dt": dt, "mode": args.mode, **extra}, f)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
