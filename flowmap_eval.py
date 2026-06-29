"""Evaluate the flow-map stepper: per-current-type rollout NRMSE + speed."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import pickle
import time
import collections
import numpy as np
import jax
import jax.numpy as jnp

import flowmap_model as F
from operator_data import KINDS


def load(path):
    with open(path, "rb") as f:
        o = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, o["params"])
    c = o["cfg"]
    cfg = F.FlowConfig(d=c["d"], hidden=tuple(c["hidden"]), stride=c["stride"])
    return params, cfg, float(o["dt"])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Flow-map eval")
    p.add_argument("--model", default=os.path.join(here, "data", "flowmap.pkl"))
    p.add_argument("--data", default=os.path.join(here, "data", "fhn_operator.npz"))
    p.add_argument("--cycles", type=int, default=3)
    args = p.parse_args()

    params, cfg, dt = load(args.model)
    d = np.load(args.data)
    ys = np.asarray(d["ys_val"]); u = np.asarray(d["u_val"])
    yc = ys[:, ::cfg.stride]; uc = u[:, ::cfg.stride]
    Tc = yc.shape[1]
    sx = yc.reshape(-1, 2).std(0) + 1e-6
    K = int(args.cycles * 37.0 / (cfg.stride * dt))
    rng = np.random.default_rng(5)
    roll = jax.jit(lambda x0, uw: F.rollout(params, cfg, x0, uw))

    agg = collections.defaultdict(list); allv = []
    for i in range(yc.shape[0]):
        t0 = int(rng.integers(0, Tc - K - 1))
        x0 = jnp.asarray(yc[i, t0][None]); uw = jnp.asarray(uc[i, t0:t0 + K + 1][None])
        xh = np.asarray(roll(x0, uw))[0]
        tgt = yc[i, t0:t0 + K + 1]
        e = np.sqrt((((xh - tgt) / sx) ** 2).mean())
        agg[KINDS[i % len(KINDS)]].append(e); allv.append(e)
    print(f"=== flow-map {cfg} (Δ={cfg.stride*dt}) anchored {args.cycles}-cycle rollout NRMSE ===")
    for k in KINDS:
        if agg[k]:
            print(f"  {k:10s} {np.mean(agg[k]):.3f}")
    print(f"  {'MEAN':10s} {np.mean(allv):.3f}")

    x0 = jnp.asarray(yc[:32, 0]); uw = jnp.asarray(uc[:32, :K + 1])
    roll(x0, uw).block_until_ready()
    t0 = time.time()
    for _ in range(20):
        roll(x0, uw).block_until_ready()
    print(f"  speed: {(time.time()-t0)/20*1e3:.2f} ms for 32 traj x {K} coarse steps "
          f"({args.cycles} cycles)")


if __name__ == "__main__":
    main()
