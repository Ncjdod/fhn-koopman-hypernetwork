"""Ground-truth spectral theory for the FitzHugh-Nagumo (FHN) system."""

import numpy as np

A_DEFAULT, B_DEFAULT, TAU_DEFAULT = 0.7, 0.8, 12.5

def fixed_points(u, a=A_DEFAULT, b=B_DEFAULT):
    """Real fixed-point membrane potentials v* for constant control u."""
    roots = np.roots([1.0 / 3.0, 0.0, (1.0 / b - 1.0), (a / b - u)])
    return roots[np.abs(roots.imag) < 1e-7].real

def jacobian_eigs_at(v_star, a=A_DEFAULT, b=B_DEFAULT, tau=TAU_DEFAULT):
    """Eigenvalues (complex) of the FHN Jacobian at membrane potential v_star."""
    T = (1.0 - v_star ** 2) - b / tau
    D = -(b / tau) * (1.0 - v_star ** 2) + 1.0 / tau
    disc = (T / 2.0) ** 2 - D
    sq = np.sqrt(complex(disc))
    return np.array([T / 2.0 + sq, T / 2.0 - sq])

def classify(v_star, a=A_DEFAULT, b=B_DEFAULT, tau=TAU_DEFAULT):
    """Return 'stable' or 'unstable' for the fixed point at v_star."""
    T = (1.0 - v_star ** 2) - b / tau
    D = -(b / tau) * (1.0 - v_star ** 2) + 1.0 / tau
    return "stable" if (D > 0 and T < 0) else "unstable"

def spectrum_over_u(u_grid, a=A_DEFAULT, b=B_DEFAULT, tau=TAU_DEFAULT):
    """Leading Jacobian eigenvalue (largest real part) over a grid of controls."""
    u_grid = np.asarray(u_grid, dtype=float)
    sig = np.full_like(u_grid, np.nan)
    om = np.full_like(u_grid, np.nan)
    vs = np.full_like(u_grid, np.nan)
    stab = np.zeros_like(u_grid, dtype=bool)
    nfp = np.zeros_like(u_grid, dtype=int)
    for i, u in enumerate(u_grid):
        fps = fixed_points(u, a, b)
        nfp[i] = len(fps)
        if len(fps) == 0:
            continue
        v_star = fps[np.argsort(np.abs(fps))][0] if len(fps) == 1 else np.sort(fps)[len(fps) // 2]
        eigs = jacobian_eigs_at(v_star, a, b, tau)
        lead = eigs[np.argmax(eigs.real)]
        sig[i], om[i], vs[i] = lead.real, abs(lead.imag), v_star
        stab[i] = classify(v_star, a, b, tau) == "stable"
    return {"u": u_grid, "sigma": sig, "omega": om, "v_star": vs,
            "stable": stab, "n_fp": nfp}

def limit_cycle_period(u, a=A_DEFAULT, b=B_DEFAULT, tau=TAU_DEFAULT,
                       t_max=400.0, dt=0.01, transient=0.5):
    """Measure the limit-cycle period at constant control u by direct simulation."""
    n = int(t_max / dt) + 1
    v, w = -1.0, -0.5
    vs = np.empty(n)
    for i in range(n):
        vs[i] = v
        def f(vv, ww):
            return (vv - vv ** 3 / 3.0 - ww + u, (vv + a - b * ww) / tau)
        k1 = f(v, w)
        k2 = f(v + dt / 2 * k1[0], w + dt / 2 * k1[1])
        k3 = f(v + dt / 2 * k2[0], w + dt / 2 * k2[1])
        k4 = f(v + dt * k3[0], w + dt * k3[1])
        v = v + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    s = int(n * transient)
    seg = vs[s:]
    amp = seg.max() - seg.min()
    thr = seg.mean()
    up = np.where((seg[:-1] < thr) & (seg[1:] >= thr))[0]
    if len(up) >= 3:
        return float(np.mean(np.diff(up)) * dt), float(amp)
    return float("nan"), float(amp)

def cycle_frequency_over_u(u_grid, **kw):
    """Angular frequency omega = 2*pi/period of the limit cycle over u (nan if none)."""
    out = []
    for u in np.asarray(u_grid, dtype=float):
        per, _ = limit_cycle_period(u, **kw)
        out.append(2.0 * np.pi / per if np.isfinite(per) else np.nan)
    return np.array(out)

if __name__ == "__main__":
    ug = np.linspace(0.0, 1.6, 17)
    sp = spectrum_over_u(ug)
    print(" u    n_fp  v*      sigma     omega   stable")
    for i, u in enumerate(ug):
        print(f"{u:4.2f}  {sp['n_fp'][i]:d}    {sp['v_star'][i]:+.3f}  "
              f"{sp['sigma'][i]:+.4f}  {sp['omega'][i]:.4f}  {sp['stable'][i]}")
