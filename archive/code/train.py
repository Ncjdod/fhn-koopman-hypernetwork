"""Curriculum trainer for the FHN Koopman model (shared model.py)."""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import model as M
from data_gen import fhn_derivatives

def to_numpy(tree):
    return jax.tree_util.tree_map(lambda a: np.array(a), tree)

def default_curriculum(epochs, period_steps=740):
    """(n_predict, stride, w_pred, n_epochs) stages; horizon grows to a FULL cycle."""
    e = epochs
    P = period_steps
    s0 = int(0.20 * e)
    s1 = int(0.25 * e)
    s2 = int(0.25 * e)
    s3 = e - s0 - s1 - s2
    return [
        (80,            40,            0.3, s0),
        (max(1, P // 3), max(1, P // 6), 0.6, s1),
        (max(1, 2 * P // 3), max(1, P // 4), 1.0, s2),
        (P + 40,        max(1, P // 3), 1.0, s3),
    ]

def train(data, backbone="spiral", m=6, epochs=1500, lr=1e-3, batch=48,
          loss_kind="mse", w_sobolev=0.1, seed=42, verbose=True,
          radial="stuart_landau"):
    ys_all = jnp.array(data["ys_train"])
    u_all = jnp.array(data["u_train"])
    dt = float(data["dt"])
    dots_all = fhn_derivatives(ys_all, u_all)
    scales = {"x": jnp.array(data["x_scale"]), "xdot": jnp.array(data["xdot_scale"])}
    N = ys_all.shape[0]
    traj_batch = min(batch, N)
    T = ys_all.shape[1]
    period_steps = int(round(37.0 / dt))

    cfg = M.ModelConfig(backbone=backbone, m=m, radial=radial)
    params = M.init_model(cfg, jax.random.PRNGKey(seed))

    sched = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(params)

    stages = default_curriculum(epochs, period_steps)
    history = {"total": [], "rec": [], "lin": [], "pred": [], "stage": []}

    def make_step(n_predict, stride, w_pred):
        n_predict = int(min(n_predict, T - 2))
        stride = int(max(1, min(stride, max(1, T - 1 - n_predict))))

        def loss_fn(p, ys_b, dots_b, u_b):
            l_rec, l_lin, l_pred = M.compute_losses(
                p, cfg, ys_b, dots_b, u_b, dt, n_predict, stride, scales,
                loss_kind=loss_kind, w_sobolev=w_sobolev)
            total = 1.0 * l_rec + 1.0 * l_lin + w_pred * l_pred
            return total, (l_rec, l_lin, l_pred)

        @jax.jit
        def step(p, st, key):
            idx = jax.random.choice(key, N, (traj_batch,), replace=False)
            ys_b, dots_b, u_b = ys_all[idx], dots_all[idx], u_all[idx]
            (tot, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, ys_b, dots_b, u_b)
            upd, st = opt.update(g, st, p)
            p = optax.apply_updates(p, upd)
            return p, st, tot, aux
        return step, n_predict

    epoch = 0
    t0 = time.time()
    key = jax.random.PRNGKey(seed + 1)
    for si, (n_predict, stride, w_pred, n_ep) in enumerate(stages):
        if n_ep <= 0:
            continue
        step, n_predict = make_step(n_predict, stride, w_pred)
        for _ in range(n_ep):
            key, kstep = jax.random.split(key)
            params, opt_state, tot, aux = step(params, opt_state, kstep)
            history["total"].append(float(tot))
            history["rec"].append(float(aux[0]))
            history["lin"].append(float(aux[1]))
            history["pred"].append(float(aux[2]))
            history["stage"].append(si)
            if verbose and (epoch % max(1, epochs // 30) == 0 or epoch == epochs - 1):
                print(f"[{backbone}] ep {epoch:4d} stage{si} np={n_predict:3d} | "
                      f"tot {float(tot):.4f} rec {float(aux[0]):.4f} "
                      f"lin {float(aux[1]):.4f} pred {float(aux[2]):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            epoch += 1

    out = {
        "params": to_numpy(params),
        "cfg": {"backbone": backbone, "m": m, "radial": radial},
        "history": {k: np.array(v) for k, v in history.items()},
        "scales": {"x": np.array(data["x_scale"]), "xdot": np.array(data["xdot_scale"])},
        "dt": dt, "loss_kind": loss_kind,
    }
    return out

def main():
    p = argparse.ArgumentParser(description="Train FHN Koopman model")
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--backbone', choices=['spiral', 'bilinear'], default='spiral')
    p.add_argument('--m', type=int, default=6)
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--radial', choices=['stuart_landau', 'clamped'], default='stuart_landau')
    p.add_argument('--loss-kind', choices=['mse', 'huber'], default='mse')
    p.add_argument('--data', type=str, default=os.path.join(here, 'data', 'fhn_koopman_t80.npz'))
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    data = np.load(args.data)
    out = train(data, backbone=args.backbone, m=args.m, epochs=args.epochs,
                lr=args.lr, batch=args.batch, loss_kind=args.loss_kind,
                radial=args.radial)

    out_path = args.out or os.path.join(here, 'data', f'koopman_{args.backbone}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"Saved trained model -> {out_path}")

if __name__ == '__main__':
    main()
