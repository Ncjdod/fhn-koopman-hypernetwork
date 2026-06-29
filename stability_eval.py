"""Long-horizon stability evaluation for the FHN Koopman model."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import json
import numpy as np
import jax.numpy as jnp

from intervention import load_model, rollout_states
from simulation import simulate_fhn_batch

PERIOD = 37.0

def _amp_per_period(v, dt, period_steps):
    """Peak-to-peak amplitude of v in each consecutive period-length window."""
    n_win = len(v) // period_steps
    amps = []
    for k in range(n_win):
        seg = v[k * period_steps:(k + 1) * period_steps]
        amps.append(float(seg.max() - seg.min()))
    return np.array(amps)

def true_cycle_amplitude(u, dt, periods):
    """True FHN peak-to-peak v amplitude on the settled limit cycle at constant u."""
    t_max = periods * PERIOD
    t = jnp.linspace(0.0, t_max, int(round(t_max / dt)) + 1)
    ys = simulate_fhn_batch(jnp.array([[-1.0, -0.5]]), t, "constant", jnp.array([u]))
    v = np.array(ys[0, :, 0])
    tail = v[len(v) // 2:]
    return float(tail.max() - tail.min())

def evaluate(model_path, periods=20, currents=None, out_json=None):
    params, cfg, dt = load_model(model_path)
    period_steps = int(round(PERIOD / dt))
    horizon = periods * period_steps
    if currents is None:
        currents = [0.2, 0.5, 0.7, 1.0, 1.3, 1.5]

    x0 = jnp.tile(jnp.array([[-1.0, -0.5]]), (len(currents), 1))
    u_seq = jnp.tile(jnp.array(currents)[:, None], (1, horizon)).astype(jnp.float32)
    x_pred = np.array(rollout_states(params, cfg, dt, x0, u_seq))
    finite = bool(np.isfinite(x_pred).all())

    report = {"model": model_path, "periods": periods, "dt": dt,
              "finite": finite, "per_current": []}
    print(f"{cfg}  dt={dt}  horizon={horizon} steps ({periods} periods)  "
          f"finite={finite}")
    print(f"{'u':>5} {'amp_first':>9} {'amp_last':>8} {'ratio':>6} "
          f"{'true_amp':>8} {'amp_err':>7}  verdict")
    for i, u in enumerate(currents):
        v = x_pred[i, :, 0]
        amps = _amp_per_period(v, dt, period_steps)
        a_first = float(np.mean(amps[2:5])) if len(amps) >= 5 else float(amps[0])
        a_last = float(np.mean(amps[-3:]))
        ratio = a_last / (a_first + 1e-9)
        t_amp = true_cycle_amplitude(u, dt, periods=8)
        quiescent = t_amp < 0.2
        amp_err = abs(a_last - t_amp) / (t_amp + 1e-9)
        if quiescent:
            ok = a_last < 0.3
            verdict = "quiescent-ok" if ok else "spurious-osc"
        else:
            ok = (0.6 < ratio < 1.6) and amp_err < 0.4
            if ratio < 0.3:
                verdict = "COLLAPSE->0"
            elif ratio > 3.0 or not finite:
                verdict = "DIVERGES"
            elif ok:
                verdict = "stable-ok"
            else:
                verdict = "amp-off"
        print(f"{u:5.2f} {a_first:9.3f} {a_last:8.3f} {ratio:6.2f} "
              f"{t_amp:8.3f} {amp_err:7.2f}  {verdict}")
        report["per_current"].append({
            "u": u, "amp_early": a_first, "amp_last": a_last, "ratio": ratio,
            "true_amp": t_amp, "amp_err": amp_err, "quiescent": quiescent,
            "verdict": verdict, "amps_per_period": amps.tolist()})

    n_fire = sum(1 for r in report["per_current"] if not r["quiescent"])
    n_stable = sum(1 for r in report["per_current"]
                   if r["verdict"] in ("stable-ok",))
    n_good = sum(1 for r in report["per_current"]
                 if r["verdict"] in ("stable-ok", "quiescent-ok"))
    report["firing_stable_rate"] = n_stable / max(1, n_fire)
    report["overall_ok_rate"] = n_good / len(currents)
    print(f"\nfiring currents stable & amplitude-correct: "
          f"{n_stable}/{n_fire}   overall ok: {n_good}/{len(currents)}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved {out_json}")
    return report

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Long-horizon stability evaluation")
    p.add_argument("--model", default=os.path.join(here, "data", "koopman_spiral.pkl"))
    p.add_argument("--periods", type=int, default=20)
    p.add_argument("--out", default=os.path.join(here, "data", "stability_spiral.json"))
    args = p.parse_args()
    evaluate(args.model, periods=args.periods, out_json=args.out)
