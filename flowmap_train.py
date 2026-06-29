"""Train the flow-map stepper with a multi-step (BPTT) curriculum on all currents."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import flowmap_model as F


def curriculum(steps):
    """(K coarse rollout steps, fraction of training) stages; horizon grows."""
    return [(8, 0.15), (32, 0.20), (128, 0.30), (560, 0.35)]


def train(ys, u, dt, cfg, steps=6000, batch=64, lr=2e-3, seed=0, verbose=True):
    yc = jnp.asarray(ys[:, ::cfg.stride])
    uc = jnp.asarray(u[:, ::cfg.stride])
    N, Tc, d = yc.shape
    sx = jnp.std(yc.reshape(-1, d), axis=0) + 1e-6

    params = F.init_flow(cfg, jax.random.PRNGKey(seed))
    sched = optax.cosine_decay_schedule(lr, steps, alpha=1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-5))
    st = opt.init(params)

    def make_step(K):
        K = int(min(K, Tc - 1))

        def loss_fn(p, idx, t0):
            off = t0[:, None] + jnp.arange(K + 1)[None, :]
            u_win = jnp.take_along_axis(uc[idx], off, axis=1)
            y_win = jnp.take_along_axis(yc[idx], off[:, :, None].repeat(d, 2), axis=1)
            x0 = y_win[:, 0]
            xh = F.rollout(p, cfg, x0, u_win)
            w = 1.0 + 2.0 * jnp.abs((y_win - y_win.mean(1, keepdims=True)) / sx)
            return jnp.mean(w * ((xh - y_win) / sx) ** 2)

        @jax.jit
        def stp(p, st, key):
            ki, kt = jax.random.split(key)
            idx = jax.random.randint(ki, (batch,), 0, N)
            t0 = jax.random.randint(kt, (batch,), 0, max(1, Tc - K - 1))
            l, g = jax.value_and_grad(loss_fn)(p, idx, t0)
            upd, st = opt.update(g, st, p)
            return optax.apply_updates(p, upd), st, l
        return stp, K

    stages = curriculum(steps)
    t0 = time.time(); i = 0
    key = jax.random.PRNGKey(seed + 1)
    for K, frac in stages:
        n_ep = int(frac * steps)
        stp, Kc = make_step(K)
        for _ in range(n_ep):
            key, k = jax.random.split(key)
            params, st, l = stp(params, st, k)
            if verbose and (i % max(1, steps // 25) == 0):
                print(f"step {i:5d} K={Kc:4d} | loss {float(l):.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
            i += 1
    return params, {"sx": np.array(sx)}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Train flow-map stepper")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--data", default=os.path.join(here, "data", "fhn_operator.npz"))
    p.add_argument("--out", default=os.path.join(here, "data", "flowmap.pkl"))
    args = p.parse_args()

    d = np.load(args.data)
    ys, u, dt = d["ys_train"], d["u_train"], float(d["dt"])
    cfg = F.FlowConfig(d=2, stride=args.stride)
    print(f"flowmap data {ys.shape}  stride={args.stride} (Δ={args.stride*dt})")
    params, extra = train(ys, u, dt, cfg, steps=args.steps)
    with open(args.out, "wb") as f:
        pickle.dump({"params": jax.tree_util.tree_map(np.array, params),
                     "cfg": {"d": cfg.d, "hidden": cfg.hidden, "stride": cfg.stride},
                     "dt": dt, **extra}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
