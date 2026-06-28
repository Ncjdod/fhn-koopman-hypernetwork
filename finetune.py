"""Warm-start a trained Koopman model and push the prediction horizon out to ~full
trajectory length, so the model learns to sustain the limit cycle over many spikes
(the part the short initial curriculum could not reach).

Loads data/koopman_<backbone>.pkl, continues training on the (mixed sine+constant)
dataset with a long-horizon curriculum, and overwrites the pickle.
"""
import os, pickle, time, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax, jax.numpy as jnp, optax
import model as M
from data_gen import fhn_derivatives

HERE = os.path.dirname(os.path.abspath(__file__))


def finetune(path, stages, lr=5e-4, batch=40):
    with open(path, "rb") as f:
        out = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, out["params"])
    cfg = M.ModelConfig(backbone=out["cfg"]["backbone"], m=out["cfg"]["m"])
    dt = float(out["dt"])

    data = np.load(os.path.join(HERE, "data", "fhn_koopman_t80.npz"))
    ys = jnp.array(data["ys_train"])[:batch]
    u = jnp.array(data["u_train"])[:batch]
    dots = fhn_derivatives(ys, u)
    sc = {"x": jnp.array(data["x_scale"]), "xdot": jnp.array(data["xdot_scale"])}
    sx = np.array(data["x_scale"])

    total = sum(s[2] for s in stages)
    sched = optax.cosine_decay_schedule(lr, decay_steps=total, alpha=2e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    st = opt.init(params)

    def full_nrmse(p):
        z0 = M.encode(p, cfg, ys[:, 0])
        zr = M.rollout(p, cfg, z0, u[:, :ys.shape[1] - 1], dt)
        za = jnp.concatenate([z0[:, None], zr], axis=1)
        xp = np.array(M.decode(p, cfg, za))
        return float(np.sqrt(np.mean(((xp[..., 0] - np.array(ys)[..., 0]) / sx[0]) ** 2)))

    print(f"[{cfg.backbone}] start full-horizon train nrmse={full_nrmse(params):.3f}", flush=True)

    def make(npred, stride):
        def loss(p):
            lr_, ll, lp = M.compute_losses(p, cfg, ys, dots, u, dt, npred, stride, sc, "mse", 0.1)
            return 1.0 * lr_ + 1.0 * ll + 1.5 * lp        # emphasise the rollout
        @jax.jit
        def step(p, s):
            v, g = jax.value_and_grad(loss)(p)
            upd, s = opt.update(g, s, p)
            return optax.apply_updates(p, upd), s, v
        return step

    t0 = time.time()
    for npred, stride, n_ep in stages:
        step = make(npred, stride)
        for ep in range(n_ep):
            params, st, v = step(params, st)
        print(f"[{cfg.backbone}] stage np={npred} done loss={float(v):.4f} "
              f"full_nrmse={full_nrmse(params):.3f} ({time.time()-t0:.0f}s)", flush=True)

    out["params"] = jax.tree_util.tree_map(lambda a: np.array(a), params)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[{cfg.backbone}] saved {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="spiral")
    args = p.parse_args()
    path = os.path.join(HERE, "data", f"koopman_{args.backbone}.pkl")
    if args.backbone == "spiral":
        stages = [(200, 100, 150), (600, 300, 150), (1300, 1300, 200)]
        finetune(path, stages, lr=5e-4, batch=40)
    else:
        stages = [(200, 100, 120), (500, 250, 120)]   # expm step is costly
        finetune(path, stages, lr=5e-4, batch=32)


if __name__ == "__main__":
    main()
