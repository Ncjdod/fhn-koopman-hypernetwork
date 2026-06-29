"""Operator-learning dataset: FHN trajectories under random time-varying currents."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import numpy as np
import jax
import jax.numpy as jnp

A, B, TAU = 0.7, 0.8, 12.5
U_LO, U_HI = -0.5, 2.2


def _f(y, u, a=A, b=B, tau=TAU):
    v, w = y[..., 0], y[..., 1]
    dv = v - v ** 3 / 3.0 - w + u
    dw = (v + a - b * w) / tau
    return jnp.stack([dv, dw], axis=-1)


def simulate_profile(y0, t_grid, u_grid):
    """RK4 integrate FHN for a per-step current profile u_grid; returns (T,2)."""
    dt = t_grid[1] - t_grid[0]
    u_t, u_next = u_grid[:-1], u_grid[1:]
    u_mid = 0.5 * (u_t + u_next)

    def step(y, inp):
        ut, um, un = inp
        k1 = _f(y, ut)
        k2 = _f(y + 0.5 * dt * k1, um)
        k3 = _f(y + 0.5 * dt * k2, um)
        k4 = _f(y + dt * k3, un)
        y2 = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y2, y2

    _, ys = jax.lax.scan(step, y0, (u_t, u_mid, u_next))
    return jnp.concatenate([y0[None, :], ys], axis=0)


simulate_batch = jax.jit(jax.vmap(simulate_profile, in_axes=(0, None, 0)))


def random_profile(rng, t, kind):
    """One random time-varying current u(t) of a given kind, clipped to range."""
    n = len(t)
    if kind == "const":
        u = np.full(n, rng.uniform(U_LO, U_HI))
    elif kind == "step":
        lo, hi = rng.uniform(-0.3, 0.4), rng.uniform(0.5, 2.0)
        t_on = rng.uniform(t[-1] * 0.05, t[-1] * 0.4)
        u = np.where(t >= t_on, hi, lo)
    elif kind == "ramp":
        u = np.linspace(rng.uniform(-0.3, 0.6), rng.uniform(0.6, 2.0), n)
    elif kind == "pulse":
        period = rng.uniform(15.0, 40.0)
        width = rng.uniform(3.0, period * 0.5)
        hi, lo = rng.uniform(0.6, 2.0), rng.uniform(-0.3, 0.3)
        u = np.where(np.mod(t, period) <= width, hi, lo)
    elif kind == "chirp":
        base, amp = rng.uniform(0.5, 1.1), rng.uniform(0.2, 0.6)
        u = base + amp * np.sin(0.05 * t + rng.uniform(0.0005, 0.003) * t ** 2)
    elif kind == "sines":
        base = rng.uniform(0.3, 1.1)
        u = np.full(n, base)
        for _ in range(rng.integers(2, 5)):
            f = rng.uniform(0.01, 0.25)
            u = u + rng.uniform(0.1, 0.5) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    elif kind == "ou":
        theta, mu, sigma = 0.05, rng.uniform(0.3, 1.1), rng.uniform(0.05, 0.2)
        dt = t[1] - t[0]
        u = np.empty(n); u[0] = mu
        noise = rng.standard_normal(n)
        for i in range(1, n):
            u[i] = u[i - 1] + theta * (mu - u[i - 1]) * dt + sigma * np.sqrt(dt) * noise[i]
    else:  # piecewise constant, random hold durations
        u = np.empty(n); i = 0
        while i < n:
            seg = int(rng.uniform(8.0, 60.0) / (t[1] - t[0]))
            u[i:i + seg] = rng.uniform(U_LO, U_HI)
            i += seg
    return np.clip(u, U_LO, U_HI)


KINDS = ["const", "step", "ramp", "pulse", "chirp", "sines", "ou", "piecewise"]


def generate(n, t_max, dt, seed):
    """Batch of (initial state, current profile, full trajectory) on a long grid."""
    t = np.arange(0.0, t_max + dt * 0.5, dt)
    rng = np.random.default_rng(seed)
    U = np.stack([random_profile(rng, t, KINDS[i % len(KINDS)])
                  for i in range(n)], axis=0).astype(np.float32)
    v0 = rng.uniform(-2.0, 2.0, n)
    w0 = rng.uniform(-1.0, 1.0, n)
    y0 = np.stack([v0, w0], axis=1).astype(np.float32)
    ys = np.asarray(simulate_batch(jnp.asarray(y0), jnp.asarray(t), jnp.asarray(U)))
    return {"t": t.astype(np.float32), "ys": ys.astype(np.float32), "u": U,
            "dt": np.float32(dt)}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Generate operator-learning dataset")
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--t-max", type=float, default=300.0)
    p.add_argument("--t-max-far", type=float, default=1500.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--out", default=os.path.join(here, "data", "fhn_operator.npz"))
    args = p.parse_args()

    tr = generate(args.n_train, args.t_max, args.dt, seed=11)
    va = generate(args.n_val, args.t_max, args.dt, seed=22)
    far = generate(max(16, args.n_val // 2), args.t_max_far, args.dt, seed=33)
    np.savez(args.out,
             t=tr["t"], ys_train=tr["ys"], u_train=tr["u"],
             ys_val=va["ys"], u_val=va["u"],
             t_far=far["t"], ys_far=far["ys"], u_far=far["u"], dt=tr["dt"])
    print(f"saved {args.out}")
    print(f"  train {tr['ys'].shape}  val {va['ys'].shape}  far {far['ys'].shape}")
    print(f"  t_max train={args.t_max} far={args.t_max_far} dt={args.dt}")


if __name__ == "__main__":
    main()
