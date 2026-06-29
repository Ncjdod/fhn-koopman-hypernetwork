"""FitzHugh-Nagumo Phase Space Stability and Bifurcation Analyzer."""

import sys
import argparse
import numpy as np

def get_external_current(t, I_type, I_val):
    """Generates external stimulus current dynamically at time t."""
    constant_current = I_val
    step_current = np.where((t >= 10.0) & (t <= 80.0), I_val, 0.0)
    sine_current = I_val * (1.0 + 0.5 * np.sin(0.2 * t))
    pulse_current = np.where(np.mod(t, 20.0) <= 5.0, I_val, 0.0)
    chirp_current = I_val * (1.0 + 0.5 * np.sin(0.05 * t + 0.001 * t**2))

    if I_type == 'step':
        return step_current
    elif I_type == 'sine':
        return sine_current
    elif I_type == 'pulse':
        return pulse_current
    elif I_type == 'chirp':
        return chirp_current
    else:
        return constant_current

def analyze_single_current(I_ext, a, b, tau):
    """Analyzes the phase space state for a single constant current value."""
    roots = np.roots([1.0/3.0, 0.0, (1.0/b - 1.0), (a/b - I_ext)])
    real_roots = roots[np.abs(np.imag(roots)) < 1e-7].real

    num_fixed_points = len(real_roots)
    stable_count = 0
    unstable_count = 0

    states = []

    for v_star in real_roots:
        tr = 1.0 - v_star**2 - b / tau
        det = -b / tau * (1.0 - v_star**2) + 1.0 / tau

        is_stable = (det > 0) and (tr < 0)
        if is_stable:
            stable_count += 1
            if np.abs(v_star) > 1.15:
                states.append("quiescence")
            else:
                states.append("excitable")
        else:
            unstable_count += 1
            states.append("limit_cycle")

    if num_fixed_points == 3:
        if stable_count == 2:
            overall = "bistable"
        else:
            overall = "bifurcated"
    elif num_fixed_points == 1:
        overall = states[0]
    else:
        overall = "unknown"

    return {
        "state": overall,
        "num_fixed_points": num_fixed_points,
        "stable_count": stable_count,
        "unstable_count": unstable_count
    }

def run_analysis(I_type, I_val, required_state=None, t_max=10.0, dt=0.1, a=0.7, b=0.8, tau=12.5):
    """Evaluates the range of states visited by the dynamic current function."""
    n_steps = int(t_max / dt) + 1
    t_span = np.linspace(0.0, t_max, n_steps)
    u_vals = [float(get_external_current(t, I_type, I_val)) for t in t_span]

    unique_states = set()
    num_fixed_points_list = []
    stable_counts = []
    unstable_counts = []

    for u in u_vals:
        res = analyze_single_current(u, a, b, tau)
        unique_states.add(res["state"])
        num_fixed_points_list.append(res["num_fixed_points"])
        stable_counts.append(res["stable_count"])
        unstable_counts.append(res["unstable_count"])

    visited = sorted(list(unique_states))

    print(f"MIN_I_EXT={np.min(u_vals):.4f}")
    print(f"MAX_I_EXT={np.max(u_vals):.4f}")
    print(f"VISITED_STATES={','.join(visited)}")
    print(f"QUIESCENCE={'quiescence' in unique_states}")
    print(f"LIMIT_CYCLE={'limit_cycle' in unique_states}")
    print(f"EXCITABLE={'excitable' in unique_states}")
    print(f"BISTABLE={'bistable' in unique_states}")
    print(f"BIFURCATED={'bifurcated' in unique_states}")

    if required_state is not None:
        if required_state not in unique_states:
            print(f"ERROR: Required state '{required_state}' was not observed in the phase space!")
            sys.exit(1)
        else:
            print(f"SUCCESS: Required state '{required_state}' is active!")

def main():
    """CLI execution entrypoint for phase space analysis."""
    parser = argparse.ArgumentParser(description="FitzHugh-Nagumo Phase Space Analyzer")
    parser.add_argument('--I-val', type=float, default=0.5)
    parser.add_argument('--I-type', type=str, default='constant')
    parser.add_argument('--t-max', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--a', type=float, default=0.7)
    parser.add_argument('--b', type=float, default=0.8)
    parser.add_argument('--tau', type=float, default=12.5)
    parser.add_argument('--require-state', type=str, default=None)
    args = parser.parse_args()

    run_analysis(
        args.I_type, args.I_val, 
        required_state=args.require_state,
        t_max=args.t_max, dt=args.dt,
        a=args.a, b=args.b, tau=args.tau
    )

if __name__ == '__main__':
    main()
