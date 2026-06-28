"""Inverse design / intervention for the FHN Koopman model."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax

import model as M

def rollout_states(params, cfg, dt, x0, u_seq):
    """Free-running predicted states for control sequence u_seq."""
    z0 = M.encode(params, cfg, x0)
    z_roll = M.rollout(params, cfg, z0, u_seq, dt)
    return M.decode(params, cfg, z_roll)

@functools.partial(jax.jit, static_argnums=(1, 5, 6, 9, 10))
def _inverse_solve(params, cfg, dt, x0, target, horizon, endpoint,
                   u_init, lr, steps, w_smooth, u_lo, u_hi):
    """Adam over the control sequence; jitted core of inverse_current."""

    def project(u):
        return jnp.clip(u, u_lo, u_hi)

    def loss_fn(u):
        uc = project(u)
        x_pred = rollout_states(params, cfg, dt, x0, uc)
        if endpoint:
            err = x_pred[:, -1, :] - target
        else:
            err = x_pred - target
        track = jnp.mean(err ** 2)
        smooth = w_smooth * jnp.mean(jnp.diff(uc, axis=-1) ** 2)
        return track + smooth

    opt = optax.adam(lr)
    st = opt.init(u_init)

    def body(carry, _):
        u, st = carry
        l, g = jax.value_and_grad(loss_fn)(u)
        upd, st = opt.update(g, st, u)
        u = optax.apply_updates(u, upd)
        return (u, st), l

    (u, _), losses = jax.lax.scan(body, (u_init, st), None, length=steps)
    return project(u), losses

def inverse_current(params, cfg, dt, x0, target, horizon=None, steps=400,
                    lr=0.05, u_init=None, u_lo=-2.0, u_hi=3.0, w_smooth=1e-3):
    """Solve for the injected current that drives x0 to ``target``."""
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    B = x0.shape[0]
    endpoint = target.ndim == 2
    if endpoint:
        assert horizon is not None, "give a horizon when matching an endpoint"
        H = int(horizon)
    else:
        H = target.shape[1]
    if u_init is None:
        u_init = jnp.zeros((B, H), dtype=jnp.float32)
    else:
        u_init = jnp.broadcast_to(jnp.asarray(u_init, jnp.float32), (B, H))

    u_opt, losses = _inverse_solve(params, cfg, dt, x0, target, H, endpoint,
                                   u_init, lr, int(steps), w_smooth, u_lo, u_hi)
    x_ach = rollout_states(params, cfg, dt, x0, u_opt)
    return u_opt, losses, x_ach

def load_model(path):
    import pickle
    with open(path, "rb") as f:
        out = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, out["params"])
    cfg = M.ModelConfig(backbone=out["cfg"]["backbone"], m=out["cfg"]["m"],
                        radial=out["cfg"].get("radial", "stuart_landau"))
    return params, cfg, float(out["dt"])

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Inverse-design demo + speed benchmark")
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--model", default=os.path.join(here, "data", "koopman_spiral.pkl"))
    p.add_argument("--horizon", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    params, cfg, dt = load_model(args.model)
    print(f"loaded {cfg}  dt={dt}  device={jax.devices()[0]}")

    from simulation import simulate_fhn_batch
    from dynamics import get_external_current
    import time
    t = jnp.linspace(0.0, args.horizon * dt, args.horizon + 1)
    tgt = simulate_fhn_batch(jnp.tile(jnp.array([[-1.0, -0.5]]), (args.batch, 1)),
                             t, "constant", jnp.ones((args.batch,)))[:, 1:, :]
    x0 = jnp.tile(jnp.array([[-1.0, -0.5]]), (args.batch, 1))

    u_zero = jnp.zeros((args.batch, args.horizon))
    f = jax.jit(lambda u: rollout_states(params, cfg, dt, x0, u))
    f(u_zero).block_until_ready()
    t0 = time.time()
    for _ in range(20):
        f(u_zero).block_until_ready()
    print(f"forward rollout {args.batch}x{args.horizon} steps: "
          f"{(time.time()-t0)/20*1e3:.2f} ms/call")

    t0 = time.time()
    u_opt, losses, x_ach = inverse_current(params, cfg, dt, x0, tgt, steps=400, lr=0.05)
    jax.block_until_ready((u_opt, x_ach))
    dtw = time.time() - t0
    nrmse = float(jnp.sqrt(jnp.mean(((x_ach - tgt)[..., 0]) ** 2)) /
                  (jnp.std(tgt[..., 0]) + 1e-9))
    print(f"inverse design ({args.batch} traj, {args.horizon} steps, 400 iters): "
          f"{dtw*1e3:.0f} ms  final loss {float(losses[-1]):.4e}  v-NRMSE {nrmse:.3f}")
