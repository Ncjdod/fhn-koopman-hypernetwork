"""
FitzHugh-Nagumo neural simulation entrypoint.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import csv
import numpy as np
import jax
import jax.numpy as jnp

from dynamics import get_external_current
from simulation import simulate_fhn, simulate_fhn_batch, fit_fhn_parameters
from plotting import plot_results

def main():
    """Main orchestration entrypoint for FHN simulation and parameter fitting."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="FitzHugh-Nagumo Neural Model Simulator and Parameter Estimator")
    
    parser.add_argument('--v0', type=float, default=-1.5, help="Initial membrane potential (default: -1.5)")
    parser.add_argument('--w0', type=float, default=-0.5, help="Initial recovery variable (default: -0.5)")
    parser.add_argument('--a', type=float, default=0.7, help="Parameter a (default: 0.7)")
    parser.add_argument('--b', type=float, default=0.8, help="Parameter b (default: 0.8)")
    parser.add_argument('--tau', type=float, default=12.5, help="Time constant tau (default: 12.5)")
    parser.add_argument('--I', type=float, default=0.5, help="Constant external current amplitude/value I (default: 0.5)")
    parser.add_argument('--I-type', type=str, default='constant', choices=['constant', 'step', 'sine', 'pulse'],
                        help="Type of dynamic external current (constant, step, sine, pulse) (default: constant)")
    
    parser.add_argument('--batch', action='store_true', help="Run in multi-trajectory batch mode using JAX vmap")
    parser.add_argument('--batch-size', type=int, default=5, help="Number of trajectories in the batch (default: 5)")
    
    parser.add_argument('--t-max', type=float, default=100.0, help="Total simulation time (default: 100.0)")
    parser.add_argument('--dt', type=float, default=0.1, help="Sampling time step (default: 0.1)")
    
    parser.add_argument('--output', type=str, default=os.path.join(data_dir, 'fhn_data.csv'), 
                        help="Output CSV filename to save time series")
    parser.add_argument('--no-plot', action='store_true', help="Disable matplotlib plotting")
    parser.add_argument('--save-plot', type=str, default=os.path.join(plots_dir, 'fhn_plot.png'), 
                        help="Save the plot as a PNG image file")
    parser.add_argument('--fit-demo', action='store_true', help="Run parameters fitting demonstration using Optax")
    
    args = parser.parse_args()
    
    n_steps = int(args.t_max / args.dt) + 1
    t_span = jnp.linspace(0.0, args.t_max, n_steps)
    
    ys_single = None
    u_data = None
    y0 = None
    
    if args.batch:
        key = jax.random.PRNGKey(101)
        key1, key2, key3 = jax.random.split(key, 3)
        v0s = jax.random.uniform(key1, (args.batch_size,), minval=-2.0, maxval=1.0)
        w0s = jax.random.uniform(key2, (args.batch_size,), minval=-1.0, maxval=0.5)
        y0_batch = jnp.stack([v0s, w0s], axis=1)
        I_val_batch = jax.random.uniform(key3, (args.batch_size,), minval=0.2, maxval=1.2)
        
        print(f"\n[Batch Mode] Simulating {args.batch_size} FHN trajectories in parallel using JAX vmap...")
        ys = simulate_fhn_batch(
            y0_batch, t_span, args.I_type, I_val_batch,
            a=args.a, b=args.b, tau=args.tau
        )
        
        u_data_batch = jnp.stack([
            jnp.array([get_external_current(t, args.I_type, iv) for t in t_span])
            for iv in I_val_batch
        ], axis=0)
        
        y0 = y0_batch[0]
        u_data = u_data_batch[0]
        ys_single = ys[0]
    else:
        y0 = [args.v0, args.w0]
        u_data = jnp.array([get_external_current(t, args.I_type, args.I) for t in t_span])
        
        print(f"Simulating FitzHugh-Nagumo model...")
        print(f"Parameters: a={args.a}, b={args.b}, tau={args.tau}, Current={args.I_type} (amplitude={args.I})")
        print(f"Time span: [0, {args.t_max}] with dt={args.dt} ({n_steps} points)")
        
        ys_single = simulate_fhn(
            y0, t_span, 
            a=args.a, b=args.b, tau=args.tau, I_type=args.I_type, I_val=args.I
        )
    
    fitted_trajectory = None
    noisy_target = None
    true_a, true_b, true_tau = args.a, args.b, args.tau
    
    if args.fit_demo:
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, ys_single.shape) * 0.08
        noisy_target = ys_single + noise
        
        fitted_params, loss_history, fitted_trajectory = fit_fhn_parameters(
            y0, t_span, noisy_target, args.I_type, args.I if not args.batch else float(I_val_batch[0]), lr=0.03, steps=150
        )
        
        print("\nOptimization Complete!")
        print(f"Target Values: a={true_a:.4f}, b={true_b:.4f}, tau={true_tau:.4f}")
        print(f"Fitted Values: a={fitted_params['a']:.4f}, b={fitted_params['b']:.4f}, tau={fitted_params['tau']:.4f}")
        print(f"Absolute error: a_err={abs(fitted_params['a']-true_a):.4f}, b_err={abs(fitted_params['b']-true_b):.4f}, tau_err={abs(fitted_params['tau']-true_tau):.4f}")
        
    if args.output:
        if args.batch:
            print(f"\nSaving time series batch data to {args.output}...")
            try:
                with open(args.output, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['trajectory_id', 'time', 'v_potential', 'w_recovery', 'I_ext']
                    writer.writerow(header)
                    
                    for m in range(args.batch_size):
                        for i in range(len(t_span)):
                            writer.writerow([
                                m, float(t_span[i]), float(ys[m, i, 0]), float(ys[m, i, 1]), float(u_data_batch[m, i])
                            ])
                print(f"Successfully wrote {args.batch_size * len(t_span)} steps of batch time series data to {args.output}")
            except Exception as e:
                print(f"Error saving CSV: {e}")
        else:
            print(f"\nSaving time series data to {args.output}...")
            try:
                with open(args.output, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['time', 'v_potential', 'w_recovery', 'I_ext']
                    if args.fit_demo:
                        header += ['v_measured', 'w_measured', 'v_fitted', 'w_fitted']
                    writer.writerow(header)
                    
                    for i in range(len(t_span)):
                        row = [float(t_span[i]), float(ys_single[i, 0]), float(ys_single[i, 1]), float(u_data[i])]
                        if args.fit_demo:
                            row += [float(noisy_target[i, 0]), float(noisy_target[i, 1]),
                                    float(fitted_trajectory[i, 0]), float(fitted_trajectory[i, 1])]
                        writer.writerow(row)
                print(f"Successfully wrote {len(t_span)} steps of time series data to {args.output}")
            except Exception as e:
                print(f"Error saving CSV: {e}")
            
    if not args.no_plot or args.save_plot:
        print("\nGenerating matplotlib visualization...")
        plot_results(
            t_span, ys_single, 
            a=args.a, b=args.b, tau=args.tau, I_type=args.I_type, I_val=args.I if not args.batch else float(I_val_batch[0]), 
            u_data=u_data,
            fitted_data=fitted_trajectory, 
            noisy_target=noisy_target,
            save_path=args.save_plot,
            show_plot=not args.no_plot
        )

if __name__ == '__main__':
    main()
