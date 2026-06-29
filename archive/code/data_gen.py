"""Generate FitzHugh-Nagumo training / validation data for Koopman learning."""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from dynamics import get_external_current
from simulation import simulate_fhn_batch

def fhn_derivatives(ys, u, a=0.7, b=0.8, tau=12.5):
    """Exact continuous-time FHN velocities for a batch (B, T, 2)."""
    v, w = ys[..., 0], ys[..., 1]
    dv = v - v ** 3 / 3.0 - w + u
    dw = (v + a - b * w) / tau
    return jnp.stack([dv, dw], axis=-1)

def _ic(key, n, vlo=-2.0, vhi=1.0, wlo=-1.0, whi=0.5):
    """Diverse initial conditions (v0, w0)."""
    kv, kw = jax.random.split(key)
    v0 = jax.random.uniform(kv, (n,), minval=vlo, maxval=vhi)
    w0 = jax.random.uniform(kw, (n,), minval=wlo, maxval=whi)
    return jnp.stack([v0, w0], axis=1)

def _block(key, n, t_span, I_type, i_lo, i_hi):
    """One training block: n trajectories of a given current type / amplitude band."""
    k_ic, k_I = jax.random.split(key)
    y0 = _ic(k_ic, n)
    I = jax.random.uniform(k_I, (n,), minval=i_lo, maxval=i_hi)
    ys = simulate_fhn_batch(y0, t_span, I_type, I)
    u = jax.vmap(lambda iv: jnp.broadcast_to(
        get_external_current(t_span, I_type, iv), t_span.shape))(I)
    return ys, u

def generate(n_train=64, n_val=24, t_max=100.0, dt=0.05, seed=101,
             i_lo=0.2, i_hi=1.4, n_const=96):
    """Simulate a *diverse* training mix and an out-of-distribution validation set."""
    n_steps = int(round(t_max / dt)) + 1
    t_span = jnp.linspace(0.0, t_max, n_steps)

    key = jax.random.PRNGKey(seed)
    k_sine, k_const, k_step, k_pulse, k_perm, k_val = jax.random.split(key, 6)

    n_step = max(16, n_train // 3)
    n_pulse = max(16, n_train // 3)

    ys_sine, u_sine = _block(k_sine, n_train, t_span, 'sine', i_lo, i_hi)
    ys_const, u_const = _block(k_const, n_const, t_span, 'constant', 0.0, 1.6)
    ys_step, u_step = _block(k_step, n_step, t_span, 'step', 0.4, 1.4)
    ys_pulse, u_pulse = _block(k_pulse, n_pulse, t_span, 'pulse', 0.6, 1.6)

    ys_tr = jnp.concatenate([ys_sine, ys_const, ys_step, ys_pulse], axis=0)
    u_tr = jnp.concatenate([u_sine, u_const, u_step, u_pulse], axis=0)
    perm = jax.random.permutation(k_perm, ys_tr.shape[0])
    ys_tr, u_tr = ys_tr[perm], u_tr[perm]

    ys_va, u_va = _block(k_val, n_val, t_span, 'chirp', i_lo, i_hi)

    dot_tr = fhn_derivatives(ys_tr, u_tr)
    x_scale = jnp.std(ys_tr.reshape(-1, 2), axis=0)
    xdot_scale = jnp.std(dot_tr.reshape(-1, 2), axis=0)

    return {
        "t_span": np.array(t_span),
        "ys_train": np.array(ys_tr), "u_train": np.array(u_tr),
        "ys_val": np.array(ys_va), "u_val": np.array(u_va),
        "x_scale": np.array(x_scale), "xdot_scale": np.array(xdot_scale),
        "dt": float(dt), "t_max": float(t_max),
    }

def main():
    p = argparse.ArgumentParser(description="Generate FHN Koopman dataset")
    p.add_argument('--n-train', type=int, default=64)
    p.add_argument('--n-val', type=int, default=24)
    p.add_argument('--t-max', type=float, default=100.0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--out', type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'data', 'fhn_koopman_t80.npz'))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Simulating {args.n_train} train + {args.n_val} val trajectories "
          f"(t_max={args.t_max}, dt={args.dt})...")
    data = generate(args.n_train, args.n_val, args.t_max, args.dt)
    np.savez(args.out, **data)
    print(f"Saved dataset -> {args.out}")
    print(f"  train {data['ys_train'].shape}  val {data['ys_val'].shape}")
    print(f"  x_scale={data['x_scale']}  xdot_scale={data['xdot_scale']}")

if __name__ == '__main__':
    main()
