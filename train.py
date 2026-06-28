"""
Curriculum trainer for the FHN Koopman model (shared model.py).

Implements the training-related fixes:
  * fix #3  reconstruction weight w_rec > 0 from step 0 (no dead decoder phase)
  * fix #4  n_predict grows on a curriculum (short -> long horizon)
  * fix #5  MSE / Huber loss with a separately-weighted, normalised Sobolev term

Saves a pickle with the trained parameters, config, loss history and the
normalisation scales for the diagnostics module to consume.
"""

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


def default_curriculum(epochs):
    """(n_predict, stride, w_pred, n_epochs) stages; horizon grows over training."""
    e = epochs
    return [
        (40,  20,  0.3, int(0.30 * e)),
        (100, 50,  0.7, int(0.30 * e)),
        (200, 100, 1.0, e - int(0.30 * e) - int(0.30 * e)),
    ]


def train(data, backbone="spiral", m=4, epochs=600, lr=3e-4, batch=32,
          loss_kind="mse", w_sobolev=0.1, seed=42, verbose=True):
    ys = jnp.array(data["ys_train"])[:batch]
    u = jnp.array(data["u_train"])[:batch]
    dt = float(data["dt"])
    dots = fhn_derivatives(ys, u)
    scales = {"x": jnp.array(data["x_scale"]), "xdot": jnp.array(data["xdot_scale"])}

    cfg = M.ModelConfig(backbone=backbone, m=m)
    params = M.init_model(cfg, jax.random.PRNGKey(seed))

    sched = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(params)

    stages = default_curriculum(epochs)
    history = {"total": [], "rec": [], "lin": [], "pred": [], "stage": []}

    def make_step(n_predict, stride, w_pred):
        def loss_fn(p):
            l_rec, l_lin, l_pred = M.compute_losses(
                p, cfg, ys, dots, u, dt, n_predict, stride, scales,
                loss_kind=loss_kind, w_sobolev=w_sobolev)
            total = 1.0 * l_rec + 1.0 * l_lin + w_pred * l_pred   # w_rec=1 from step 0
            return total, (l_rec, l_lin, l_pred)

        @jax.jit
        def step(p, st):
            (tot, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
            upd, st = opt.update(g, st, p)
            p = optax.apply_updates(p, upd)
            return p, st, tot, aux
        return step

    epoch = 0
    t0 = time.time()
    for si, (n_predict, stride, w_pred, n_ep) in enumerate(stages):
        if n_ep <= 0:
            continue
        step = make_step(n_predict, stride, w_pred)
        for _ in range(n_ep):
            params, opt_state, tot, aux = step(params, opt_state)
            history["total"].append(float(tot))
            history["rec"].append(float(aux[0]))
            history["lin"].append(float(aux[1]))
            history["pred"].append(float(aux[2]))
            history["stage"].append(si)
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                print(f"[{backbone}] ep {epoch:4d} stage{si} np={n_predict:3d} | "
                      f"tot {float(tot):.4f} rec {float(aux[0]):.4f} "
                      f"lin {float(aux[1]):.4f} pred {float(aux[2]):.4f} "
                      f"({time.time()-t0:.0f}s)")
            epoch += 1

    out = {
        "params": to_numpy(params),
        "cfg": {"backbone": backbone, "m": m},
        "history": {k: np.array(v) for k, v in history.items()},
        "scales": {"x": np.array(data["x_scale"]), "xdot": np.array(data["xdot_scale"])},
        "dt": dt, "loss_kind": loss_kind,
    }
    return out


def main():
    p = argparse.ArgumentParser(description="Train FHN Koopman model")
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--backbone', choices=['spiral', 'bilinear'], default='spiral')
    p.add_argument('--m', type=int, default=4)
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--loss-kind', choices=['mse', 'huber'], default='mse')
    p.add_argument('--data', type=str, default=os.path.join(here, 'data', 'fhn_koopman_t80.npz'))
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    data = np.load(args.data)
    out = train(data, backbone=args.backbone, m=args.m, epochs=args.epochs,
                lr=args.lr, batch=args.batch, loss_kind=args.loss_kind)

    out_path = args.out or os.path.join(here, 'data', f'koopman_{args.backbone}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"Saved trained model -> {out_path}")


if __name__ == '__main__':
    main()
