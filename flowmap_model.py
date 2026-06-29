"""Learned flow-map stepper: a fast neural integrator for arbitrary currents.

A Markov residual step  x_{t+Δ} = x_t + g_θ(x_t, u_t, u_{t+Δ})  on a COARSE grid
(Δ = stride·dt). Unlike PWFO this is recurrent (a scan over the horizon), but it
handles arbitrary/fast currents with full waveform+phase, is differentiable in the
current, and is GPU-batched. The coarse step makes it a real speedup over a stiff
solver (the value scales to Hodgkin-Huxley).
"""
import numpy as np
import jax
import jax.numpy as jnp


def _init_mlp(sizes, key, scale_last=0.1):
    ks = jax.random.split(key, len(sizes) - 1)
    ps = []
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        lim = np.sqrt(6.0 / (a + b)) * (scale_last if i == len(sizes) - 2 else 1.0)
        ps.append({"w": jax.random.uniform(ks[i], (a, b), minval=-lim, maxval=lim),
                   "b": jnp.zeros((b,))})
    return ps


def _mlp(x, ps):
    a = x
    for layer in ps[:-1]:
        a = jax.nn.swish(a @ layer["w"] + layer["b"])
    return a @ ps[-1]["w"] + ps[-1]["b"]


class FlowConfig:
    """Static config for the flow-map stepper."""

    def __init__(self, d=2, hidden=(96, 96), stride=4):
        self.d = d
        self.hidden = tuple(hidden)
        self.stride = stride

    def __repr__(self):
        return f"FlowConfig(d={self.d}, stride={self.stride})"


def init_flow(cfg, key):
    """Init the residual stepper MLP [x_t, u_t, u_next] -> Δx."""
    return {"step": _init_mlp([cfg.d + 2, *cfg.hidden, cfg.d], key)}


def step(params, cfg, x, u_t, u_next):
    """One coarse step x -> x_next (residual)."""
    inp = jnp.concatenate([x, u_t[..., None], u_next[..., None]], axis=-1)
    return x + _mlp(inp, params["step"])


def rollout(params, cfg, x0, u_coarse):
    """Roll the stepper over a coarse current sequence u_coarse (B, Tc).

    Returns states at steps 0..Tc (B, Tc+1, d). u_coarse[k] is the current at
    coarse-grid time k.
    """
    u_t = u_coarse[:, :-1]
    u_next = u_coarse[:, 1:]

    @jax.checkpoint
    def body(x, uu):
        x_next = step(params, cfg, x, uu[0], uu[1])
        return x_next, x_next

    seq = (jnp.moveaxis(u_t, 1, 0), jnp.moveaxis(u_next, 1, 0))
    _, xs = jax.lax.scan(body, x0, seq)
    xs = jnp.moveaxis(xs, 0, 1)
    return jnp.concatenate([x0[:, None, :], xs], axis=1)


def coarse_grid(u_full, stride):
    """Subsample a native-dt current profile (B, S) to the coarse grid (B, Tc+1)."""
    return u_full[:, ::stride]
