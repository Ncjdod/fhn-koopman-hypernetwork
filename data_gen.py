"""
Generate FitzHugh-Nagumo training / validation data for Koopman learning.

Key change from the previous pipeline (fix #4): the horizon is raised to
t_max >= 80 (~ two limit-cycle periods, period ~37) instead of the previous
t_max = 10 which was sub-period and far too easy.  We use dt = 0.05 (T = 1601)
to keep the RK4 cost manageable on CPU while resolving the spike.

Also computes per-dimension normalisation scales for the state and its time
derivative (fix #5: dw/dt is ~1/tau smaller than dv/dt, so the Sobolev loss must
normalise them to comparable magnitude).
"""

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


def generate(n_train=64, n_val=16, t_max=80.0, dt=0.05, seed=101,
             i_lo=0.2, i_hi=1.2, n_const=24):
    """Simulate train (sine + constant currents) and validation (chirp) batches.

    A block of constant-current trajectories in the oscillatory range is added to
    the training set so the model directly observes — and must learn to *sustain* —
    clean limit cycles, not only the amplitude-modulated sine drive.
    """
    n_steps = int(round(t_max / dt)) + 1
    t_span = jnp.linspace(0.0, t_max, n_steps)

    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    # --- training set: sine current, diverse initial conditions --------------
    v0 = jax.random.uniform(k1, (n_train,), minval=-2.0, maxval=1.0)
    w0 = jax.random.uniform(k2, (n_train,), minval=-1.0, maxval=0.5)
    y0 = jnp.stack([v0, w0], axis=1)
    I_train = jax.random.uniform(k3, (n_train,), minval=i_lo, maxval=i_hi)
    ys_sine = simulate_fhn_batch(y0, t_span, 'sine', I_train)
    u_sine = jax.vmap(lambda iv: get_external_current(t_span, 'sine', iv))(I_train)

    # --- training set: constant currents in the oscillatory band -------------
    v0c = jax.random.uniform(k7, (n_const,), minval=-1.8, maxval=0.6)
    w0c = jax.random.uniform(k8, (n_const,), minval=-0.8, maxval=0.4)
    y0c = jnp.stack([v0c, w0c], axis=1)
    I_const = jax.random.uniform(k4, (n_const,), minval=0.7, maxval=1.3)
    ys_const = simulate_fhn_batch(y0c, t_span, 'constant', I_const)
    u_const = jax.vmap(lambda iv: get_external_current(t_span, 'constant', iv)
                       * jnp.ones_like(t_span))(I_const)

    ys_tr = jnp.concatenate([ys_sine, ys_const], axis=0)
    u_tr = jnp.concatenate([u_sine, u_const], axis=0)
    # shuffle so the sine and constant blocks are interleaved (batch slicing fair)
    perm = jax.random.permutation(k6, ys_tr.shape[0])
    ys_tr, u_tr = ys_tr[perm], u_tr[perm]

    # --- validation set: unseen chirp current --------------------------------
    v0v = jax.random.uniform(k4, (n_val,), minval=-1.8, maxval=0.8)
    w0v = jax.random.uniform(k5, (n_val,), minval=-0.8, maxval=0.4)
    y0v = jnp.stack([v0v, w0v], axis=1)
    I_val = jax.random.uniform(k6, (n_val,), minval=i_lo, maxval=i_hi)
    ys_va = simulate_fhn_batch(y0v, t_span, 'chirp', I_val)
    u_va = jax.vmap(lambda iv: get_external_current(t_span, 'chirp', iv))(I_val)

    # --- normalisation scales from training data (fix #5) --------------------
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
    p.add_argument('--n-val', type=int, default=16)
    p.add_argument('--t-max', type=float, default=80.0)
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
