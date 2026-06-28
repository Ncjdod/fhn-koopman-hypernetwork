"""Fit + evaluate the SINDYc model (fast CPU route)."""
import os
import argparse
import json
import time
import numpy as np

import sindyc as S

def _nrmse_metrics(model, ys, us, dt, x_scale, threshold=0.30):
    """Free-running rollout success on a batch (mirrors diagnostics._rollout_metrics)."""
    B, T, _ = ys.shape
    x_pred = model.simulate(ys[:, 0], us[:, :T - 1], dt)
    sx = np.asarray(x_scale)
    nrmse_full = np.sqrt(np.mean(((x_pred - ys) / sx) ** 2, axis=(1, 2)))
    one_p = min(T, int(37.0 / dt))
    nrmse_1p = np.sqrt(np.mean(((x_pred[:, :one_p] - ys[:, :one_p]) / sx) ** 2, axis=(1, 2)))

    def dom(sig):
        sig = sig - sig.mean()
        ps = np.abs(np.fft.rfft(sig)) ** 2
        fr = np.fft.rfftfreq(len(sig), d=dt)
        k = 1 + int(np.argmax(ps[1:]))
        return fr[k], (sig.max() - sig.min())
    spec = np.zeros(B, dtype=bool)
    for i in range(B):
        ft, at = dom(ys[i, :, 0]); fp, ap = dom(x_pred[i, :, 0])
        spec[i] = (abs(fp - ft) / (ft + 1e-9) < 0.25) and (abs(ap - at) / (at + 1e-9) < 0.35)
    return {
        "finite": bool(np.isfinite(x_pred).all()),
        "success_2period": float(np.mean(nrmse_full < threshold)),
        "success_1period": float(np.mean(nrmse_1p < threshold)),
        "spectral_success": float(spec.mean()),
        "nrmse_full_mean": float(nrmse_full.mean()),
        "nrmse_1p_mean": float(nrmse_1p.mean()),
        "n": B,
    }

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Fit + evaluate SINDYc")
    p.add_argument("--data", default=os.path.join(here, "data", "fhn_koopman_t80.npz"))
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--exact-deriv", action="store_true",
                   help="use analytic FHN derivatives instead of finite differences")
    p.add_argument("--out", default=os.path.join(here, "data", "sindyc_model.npz"))
    args = p.parse_args()

    d = np.load(args.data)
    ys, us, dt = d["ys_train"], d["u_train"], float(d["dt"])
    x_scale = d["x_scale"]

    t0 = time.time()
    model = S.fit(ys, us, dt, threshold=args.threshold,
                  use_exact_derivatives=args.exact_deriv)
    fit_t = time.time() - t0
    print(f"=== SINDYc fit in {fit_t:.2f}s on CPU "
          f"(deriv={'exact' if args.exact_deriv else 'finite-diff'}) ===")
    for e in model.equations():
        print("  ", e)

    b = model.assert_bounded(u_test=(0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6), periods=200)
    print(f"\nLong-horizon stability (200 periods): v3_coeff={b['v3_coeff']:.4f} "
          f"negative={b['v3_negative']}")
    for u, r in b["rollouts"].items():
        kind = "FIRING " if r["tail_ptp"] > 0.5 else "quiet  "
        print(f"  u={u:4.2f} {kind} finite={r['finite']} "
              f"max|v|={r['max_abs']:.2f} tail_ptp={r['tail_ptp']:.3f}")

    m_ood = _nrmse_metrics(model, d["ys_val"], d["u_val"], dt, x_scale)
    print(f"\nOOD (chirp) free-running rollout over {m_ood['n']} trajectories:")
    print(f"  success(2-period NRMSE<0.30) = {m_ood['success_2period']:.0%}   "
          f"success(1-period) = {m_ood['success_1period']:.0%}   "
          f"spectral = {m_ood['spectral_success']:.0%}")
    print(f"  mean NRMSE: full={m_ood['nrmse_full_mean']:.3f}  "
          f"1-period={m_ood['nrmse_1p_mean']:.3f}  finite={m_ood['finite']}")

    yv, uv = d["ys_val"][0], d["u_val"][0]
    xdot_true_v = S.finite_difference(yv, dt)[:, 0]
    u_rec = model.invert_onestep(yv, xdot_true_v)
    inv_err = float(np.mean(np.abs(u_rec - uv)))
    print(f"\nClosed-form 1-step inversion: mean |u_recovered - u_true| = {inv_err:.4f} "
          f"(over a full OOD chirp trajectory)")

    np.savez(args.out, Xi=model.Xi, degree=model.degree,
             control_cross=model.control_cross, dt=dt,
             names=np.array(model.names))
    print(f"\nsaved {args.out}")

    summary = {"fit_seconds": fit_t, "equations": model.equations(),
               "stability": b, "ood": m_ood, "inversion_mae": inv_err}
    with open(os.path.join(here, "data", "sindyc_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
