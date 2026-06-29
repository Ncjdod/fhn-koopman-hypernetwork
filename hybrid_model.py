"""Hybrid surrogate: accurate recurrent flow-map + one-shot PWFO.

Routing per call:
  * finite horizon (any current)      -> flow-map rollout (full waveform+phase, NRMSE ~0.06)
  * very far / unbounded t, slow drive -> PWFO one-shot (instant at any t; cannot step that far)

The flow-map is the accuracy workhorse; PWFO covers the arbitrary-t corner the
stepper cannot reach in finite steps.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import pickle
import numpy as np
import jax
import jax.numpy as jnp

import pwfo_model as P
import flowmap_model as F


def load(pwfo_path, flow_path):
    with open(pwfo_path, "rb") as f:
        op = pickle.load(f)
    pw = jax.tree_util.tree_map(jnp.asarray, op["params"])
    cp = op["cfg"]
    pcfg = P.PWFOConfig(d=cp["d"], K=cp["K"], m=cp["m"],
                        local_waveform=cp.get("local_waveform", False))
    with open(flow_path, "rb") as f:
        of = pickle.load(f)
    fm = jax.tree_util.tree_map(jnp.asarray, of["params"])
    cf = of["cfg"]
    fcfg = F.FlowConfig(d=cf["d"], hidden=tuple(cf["hidden"]), stride=cf["stride"])
    return (pw, pcfg), (fm, fcfg), float(op["dt"])


def current_speed(u_profile, dt):
    """Max |du/dt| over the profile (per trajectory) — the routing signal."""
    return np.max(np.abs(np.diff(np.asarray(u_profile), axis=-1)), axis=-1) / dt


def predict(pwfo, flow, dt, x0, u_profile, t_query, far_cap_cycles=80,
            slow_thresh=0.15):
    """Route to flow-map (finite horizon) or PWFO (very-far + slow). Returns
    (x_pred (B,Q,d), route:str)."""
    (pw, pcfg) = pwfo
    (fm, fcfg) = flow
    x0 = jnp.asarray(x0); u_profile = jnp.asarray(u_profile); t_query = jnp.asarray(t_query)
    maxt = float(jnp.max(t_query))
    far_cap = far_cap_cycles * 37.0
    speed = current_speed(u_profile, dt).max()

    if maxt <= far_cap:
        uc = u_profile[:, ::fcfg.stride]
        Kneed = int(maxt / (fcfg.stride * dt)) + 2
        uc = uc[:, :Kneed]
        xs = F.rollout(fm, fcfg, x0, uc)
        tc = jnp.arange(xs.shape[1]) * (fcfg.stride * dt)
        xp = jnp.stack([
            jnp.stack([jnp.interp(t_query[b], tc, xs[b, :, c]) for c in range(fcfg.d)], -1)
            for b in range(x0.shape[0])], 0)
        return np.asarray(xp), "flowmap"

    if speed <= slow_thresh:
        return np.asarray(P.forward(pw, pcfg, x0, u_profile, t_query, dt)), "pwfo(one-shot far)"

    uc = u_profile[:, ::fcfg.stride]
    xs = F.rollout(fm, fcfg, x0, uc)
    tc = jnp.arange(xs.shape[1]) * (fcfg.stride * dt)
    xp = jnp.stack([
        jnp.stack([jnp.interp(jnp.clip(t_query[b], 0, tc[-1]), tc, xs[b, :, c])
                   for c in range(fcfg.d)], -1) for b in range(x0.shape[0])], 0)
    return np.asarray(xp), "flowmap(capped; fast+far has no exact one-shot)"
