"""Precompute the ground-truth limit-cycle frequency omega(u) for PWFO supervision."""
import os
import argparse
import numpy as np
import fhn_theory as TH


def build(u_lo=0.0, u_hi=1.8, n=64, t_max=400.0, dt=0.01):
    """Measured cycle frequency over a u-grid (nan where quiescent)."""
    ug = np.linspace(u_lo, u_hi, n)
    omega = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    for i, u in enumerate(ug):
        per, a = TH.limit_cycle_period(float(u), t_max=t_max, dt=dt)
        amp[i] = a
        if np.isfinite(per) and per > 0:
            omega[i] = 2.0 * np.pi / per
    return ug, omega, amp


def lookup(u, ug, omega):
    """Interpolated omega(u); quiescent (nan) entries treated as a small floor."""
    om = np.where(np.isfinite(omega), omega, 0.0)
    return np.interp(np.asarray(u), ug, om)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Build omega(u) frequency table")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default=os.path.join(here, "data", "fhn_freq_table.npz"))
    args = p.parse_args()
    ug, omega, amp = build(n=args.n)
    np.savez(args.out, u=ug, omega=omega, amp=amp)
    fire = np.isfinite(omega)
    print(f"saved {args.out}")
    print(f"  firing band u in [{ug[fire].min():.2f}, {ug[fire].max():.2f}]  "
          f"omega in [{np.nanmin(omega):.4f}, {np.nanmax(omega):.4f}]  "
          f"amp~{np.nanmax(amp):.2f}")


if __name__ == "__main__":
    main()
