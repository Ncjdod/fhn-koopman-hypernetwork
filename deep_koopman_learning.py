"""
Deep Koopman Learning with Sobolev training for Global Dynamics in the FitzHugh-Nagumo Model.
"""

import os
import sys
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from functools import partial

from dynamics import get_external_current
from simulation import simulate_fhn, simulate_fhn_batch
from phase_space_analyzer import run_analysis

def init_mlp_params(layers, key):
    """Initializes weights and biases for a simple MLP."""
    keys = jax.random.split(key, len(layers) - 1)
    params = []
    for i in range(len(layers) - 1):
        in_dim = layers[i]
        out_dim = layers[i+1]
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        w = jax.random.uniform(keys[i], (in_dim, out_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim,))
        params.append({"w": w, "b": b})
    return params

def forward_mlp(x, params):
    """Computes the forward pass of a simple MLP with Swish activations."""
    activation = x
    for i in range(len(params) - 1):
        layer = params[i]
        activation = jax.nn.swish(jnp.dot(activation, layer["w"]) + layer["b"])
    layer = params[-1]
    return jnp.dot(activation, layer["w"]) + layer["b"]

def compute_fhn_derivatives(ys, u_data, a=0.7, b=0.8, tau=12.5):
    """Computes the exact continuous-time FHN vector field derivatives for a batch of trajectories."""
    v = ys[:, :, 0]
    w = ys[:, :, 1]
    dv = v - (v ** 3) / 3.0 - w + u_data
    dw = (v + a - b * w) / tau
    return jnp.stack([dv, dw], axis=2)

def apply_koopman_operator(z, u, sigma_0, omega_0, sigma_I, omega_I, dt):
    """Applies the block-diagonal parameterized Koopman operator in latent space."""
    sigma = sigma_0 + sigma_I * u[:, None]
    omega = omega_0 + omega_I * u[:, None]
    scale = jnp.exp(sigma * dt)
    cos_w = jnp.cos(omega * dt)
    sin_w = jnp.sin(omega * dt)
    z0 = z[:, :, 0]
    z1 = z[:, :, 1]
    z_next0 = scale * (z0 * cos_w - z1 * sin_w)
    z_next1 = scale * (z0 * sin_w + z1 * cos_w)
    return jnp.stack([z_next0, z_next1], axis=2)

def apply_continuous_koopman(z, u, sigma_0, omega_0, sigma_I, omega_I):
    """Applies the block-diagonal continuous-time parameterized Koopman operator in latent space."""
    sigma = sigma_0 + sigma_I * u[:, None]
    omega = omega_0 + omega_I * u[:, None]
    z0 = z[:, :, 0]
    z1 = z[:, :, 1]
    z_dot0 = sigma * z0 - omega * z1
    z_dot1 = omega * z0 + sigma * z1
    return jnp.stack([z_dot0, z_dot1], axis=2)

def compute_loss_term(diff, power):
    """Computes the scaled L_p norm (where p = power) to preserve scale and gradients."""
    return (jnp.mean(jnp.abs(diff) ** power) + 1e-15) ** (1.0 / power)

def compute_losses(params_dict, trajectories, trajectory_dots, current_profiles, m, n_predict, dt, loss_power=2):
    """Computes the three Deep Koopman learning losses including Sobolev training terms."""
    params_enc = params_dict["enc"]
    params_dec = params_dict["dec"]
    sigma_0 = params_dict["sigma_0"]
    omega_0 = params_dict["omega_0"]
    sigma_I = params_dict["sigma_I"]
    omega_I = params_dict["omega_I"]
    
    batch_size, T, _ = trajectories.shape
    x_flat = trajectories.reshape(-1, 2)
    x_dot_flat = trajectory_dots.reshape(-1, 2)
    
    z_flat = forward_mlp(x_flat, params_enc)
    x_recon_flat = forward_mlp(z_flat, params_dec)
    loss_recon_state = compute_loss_term(x_flat - x_recon_flat, loss_power)
    
    def reconstruct_jvp(x_val, x_dot_val):
        _, z_dot = jax.jvp(lambda x_in: forward_mlp(x_in, params_enc), (x_val,), (x_dot_val,))
        _, x_recon_dot = jax.jvp(lambda z_in: forward_mlp(z_in, params_dec), (forward_mlp(x_val, params_enc),), (z_dot,))
        return x_recon_dot
        
    x_recon_dot_flat = jax.vmap(reconstruct_jvp)(x_flat, x_dot_flat)
    loss_recon_sobolev = compute_loss_term(x_dot_flat - x_recon_dot_flat, loss_power)
    loss_recon = loss_recon_state + loss_recon_sobolev
    
    z_seq = z_flat.reshape(batch_size, T, m, 2)
    z_curr = z_seq[:, :-1]
    z_next_true = z_seq[:, 1:]
    u_curr = current_profiles[:, :-1]
    
    z_curr_flat = z_curr.reshape(-1, m, 2)
    u_curr_flat = u_curr.reshape(-1)
    z_next_pred_flat = apply_koopman_operator(
        z_curr_flat, u_curr_flat, sigma_0, omega_0, sigma_I, omega_I, dt
    )
    z_next_true_flat = z_next_true.reshape(-1, m, 2)
    loss_lin_state = compute_loss_term(z_next_true_flat - z_next_pred_flat, loss_power)
    
    def encoder_jvp(x_val, x_dot_val):
        _, z_dot = jax.jvp(lambda x_in: forward_mlp(x_in, params_enc), (x_val,), (x_dot_val,))
        return z_dot
        
    z_dot_flat = jax.vmap(encoder_jvp)(x_flat, x_dot_flat)
    z_dot_seq = z_dot_flat.reshape(batch_size, T, m, 2)
    
    z_flat_all = z_seq.reshape(-1, m, 2)
    u_flat_all = current_profiles.reshape(-1)
    z_dot_pred_flat = apply_continuous_koopman(z_flat_all, u_flat_all, sigma_0, omega_0, sigma_I, omega_I)
    loss_lin_sobolev = compute_loss_term(z_dot_flat - z_dot_pred_flat.reshape(-1, 2 * m), loss_power)
    loss_lin = loss_lin_state + loss_lin_sobolev
    
    stride = 20
    S = (T - 1 - n_predict) // stride
    
    def predict_forward_recursive(z_init, u_seq):
        def step(z, u):
            z_next = apply_koopman_operator(
                z[jnp.newaxis, :, :], jnp.array([u]), sigma_0, omega_0, sigma_I, omega_I, dt
            )
            z_next = z_next[0]
            return z_next, z_next
        _, z_preds = jax.lax.scan(step, z_init, u_seq)
        return z_preds
        
    def decoder_jvp(z_val, z_dot_val):
        _, x_dot = jax.jvp(lambda z_in: forward_mlp(z_in, params_dec), (z_val,), (z_dot_val,))
        return x_dot
        
    def get_window_loss(idx):
        start = idx * stride
        z_init = jax.lax.dynamic_slice(z_seq, (0, start, 0, 0), (batch_size, 1, m, 2))
        z_init = jnp.squeeze(z_init, axis=1)
        u_seq = jax.lax.dynamic_slice(current_profiles, (0, start), (batch_size, n_predict))
        x_target = jax.lax.dynamic_slice(trajectories, (0, start + 1, 0), (batch_size, n_predict, 2))
        x_dot_target = jax.lax.dynamic_slice(trajectory_dots, (0, start + 1, 0), (batch_size, n_predict, 2))
        
        z_preds = jax.vmap(predict_forward_recursive, in_axes=(0, 0))(z_init, u_seq)
        z_preds_flat = z_preds.reshape(-1, 2 * m)
        x_preds_flat = forward_mlp(z_preds_flat, params_dec)
        x_preds = x_preds_flat.reshape(batch_size, n_predict, 2)
        loss_pred_state = compute_loss_term(x_preds - x_target, loss_power)
        
        z_preds_m2 = z_preds.reshape(-1, m, 2)
        u_seq_flat = u_seq.reshape(-1)
        z_dot_preds_flat = apply_continuous_koopman(z_preds_m2, u_seq_flat, sigma_0, omega_0, sigma_I, omega_I)
        
        z_preds_flat_2m = z_preds.reshape(-1, 2 * m)
        z_dot_preds_flat_2m = z_dot_preds_flat.reshape(-1, 2 * m)
        x_dot_preds_flat = jax.vmap(decoder_jvp)(z_preds_flat_2m, z_dot_preds_flat_2m)
        x_dot_preds = x_dot_preds_flat.reshape(batch_size, n_predict, 2)
        
        loss_pred_sobolev = compute_loss_term(x_dot_preds - x_dot_target, loss_power)
        
        return loss_pred_state + loss_pred_sobolev
        
    window_losses = jax.vmap(get_window_loss)(jnp.arange(S))
    loss_pred = jnp.mean(window_losses)
    
    return loss_recon, loss_lin, loss_pred

def init_deep_koopman_params(m, key):
    """Initializes all learnable parameters for the deep Koopman model."""
    key1, key2, key3, key4 = jax.random.split(key, 4)
    params_enc = init_mlp_params([2, 32, 32, 2 * m], key1)
    params_dec = init_mlp_params([2 * m, 32, 32, 2], key2)
    sigma_0 = jax.random.uniform(key3, (m,), minval=-0.2, maxval=-0.01)
    sigma_I = jax.random.uniform(key4, (m,), minval=-0.05, maxval=0.05)
    omega_0 = jax.random.uniform(key3, (m,), minval=0.1, maxval=0.5)
    omega_I = jax.random.uniform(key4, (m,), minval=-0.05, maxval=0.05)
    return {
        "enc": params_enc,
        "dec": params_dec,
        "sigma_0": sigma_0,
        "omega_0": omega_0,
        "sigma_I": sigma_I,
        "omega_I": omega_I
    }

def main():
    """Simulates FHN data, optimizes the deep Koopman networks using Sobolev training, and validates."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Deep Koopman Learning")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--latent-m', type=int, default=3)
    parser.add_argument('--t-max', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--n-predict', type=int, default=100)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--require-state', type=str, default=None)
    parser.add_argument('--loss-power', type=int, default=6)
    args = parser.parse_args()
    
    print("Performing initial phase space analysis check...")
    run_analysis('sine', 0.5, required_state=args.require_state, t_max=args.t_max, dt=args.dt)
    
    n_steps = int(args.t_max / args.dt) + 1
    t_span = jnp.linspace(0.0, args.t_max, n_steps)
    
    key = jax.random.PRNGKey(101)
    key1, key2, key3 = jax.random.split(key, 3)
    v0s = jax.random.uniform(key1, (args.batch_size,), minval=-2.0, maxval=1.0)
    w0s = jax.random.uniform(key2, (args.batch_size,), minval=-1.0, maxval=0.5)
    y0_batch = jnp.stack([v0s, w0s], axis=1)
    I_val_batch = jax.random.uniform(key3, (args.batch_size,), minval=0.2, maxval=1.2)
    
    print(f"Simulating {args.batch_size} training trajectories...")
    ys = simulate_fhn_batch(y0_batch, t_span, 'sine', I_val_batch)
    u_data_batch = jax.vmap(lambda iv: get_external_current(t_span, 'sine', iv))(I_val_batch)
    
    print("Initializing Deep Koopman network parameters...")
    init_key = jax.random.PRNGKey(42)
    params = init_deep_koopman_params(args.latent_m, init_key)
    
    weights = (1.0, 1.0, 1.0)
    
    def total_loss_fn(params_dict, trajectories, current_profiles):
        trajectory_dots = compute_fhn_derivatives(trajectories, current_profiles)
        l_rec, l_lin, l_pred = compute_losses(
            params_dict, trajectories, trajectory_dots, current_profiles, args.latent_m, args.n_predict, args.dt, args.loss_power
        )
        return weights[0] * l_rec + weights[1] * l_lin + weights[2] * l_pred
        
    lr_schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=args.steps,
        alpha=1e-2
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(p_vars, state, trajectories, current_profiles):
        loss, grads = jax.value_and_grad(total_loss_fn)(p_vars, trajectories, current_profiles)
        updates, state = optimizer.update(grads, state, p_vars)
        p_vars = optax.apply_updates(p_vars, updates)
        return p_vars, state, loss
        
    print(f"Starting Deep Koopman Sobolev training ({args.steps} epochs)...")
    for step in range(args.steps):
        params, opt_state, loss_val = train_step(params, opt_state, ys, u_data_batch)
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            trajectory_dots = compute_fhn_derivatives(ys, u_data_batch)
            l_rec, l_lin, l_pred = compute_losses(
                params, ys, trajectory_dots, u_data_batch, args.latent_m, args.n_predict, args.dt, args.loss_power
            )
            print(f"Epoch {step:03d} | Total Loss: {float(loss_val):.6f} | Recon: {float(l_rec):.6f} | Lin: {float(l_lin):.6f} | Pred: {float(l_pred):.6f}")
            
    print("\nTraining completed successfully! Running validation on unseen dynamic chirp current...")
    y0_val = jnp.array([-1.5, -0.5])
    I_val_chirp = 0.5
    ys_chirp_true = simulate_fhn(y0_val, t_span, I_type='chirp', I_val=I_val_chirp)
    u_chirp_true = get_external_current(t_span, 'chirp', I_val_chirp)
    v_chirp_true = ys_chirp_true[:, 0]
    
    z_chirp_0 = forward_mlp(ys_chirp_true[0], params["enc"])
    z_chirp_0_reshaped = z_chirp_0.reshape(args.latent_m, 2)
    
    def evolve_latent_trajectory(z_init, u_seq):
        def step(z, u):
            z_next = apply_koopman_operator(
                z[jnp.newaxis, :, :], jnp.array([u]), params["sigma_0"], params["omega_0"], params["sigma_I"], params["omega_I"], args.dt
            )
            z_next = z_next[0]
            return z_next, z_next
        _, z_preds = jax.lax.scan(step, z_init, u_seq)
        return z_preds
        
    u_seq_chirp = u_chirp_true[:-1]
    z_preds_latent = evolve_latent_trajectory(z_chirp_0_reshaped, u_seq_chirp)
    z_preds_latent_all = jnp.concatenate([z_chirp_0_reshaped[jnp.newaxis, :, :], z_preds_latent], axis=0)
    z_preds_flat = z_preds_latent_all.reshape(-1, 2 * args.latent_m)
    x_preds_chirp = forward_mlp(z_preds_flat, params["dec"])
    v_chirp_pred_deep = x_preds_chirp[:, 0]
    
    cheb_deep = float(jnp.max(jnp.abs(v_chirp_true - v_chirp_pred_deep)))
    mse_deep = float(jnp.mean((v_chirp_true - v_chirp_pred_deep) ** 2))
    
    print("\nDeep Koopman Sobolev Validation Performance:")
    print(f"  - Membrane Potential MSE: {mse_deep:.6e}")
    print(f"  - Membrane Potential Chebyshev: {cheb_deep:.6e}")
    
    if not args.no_plot:
        if 'seaborn-v0_8-whitegrid' in plt.style.available:
            plt.style.use('seaborn-v0_8-whitegrid')
        else:
            plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        ax1.plot(t_span, u_chirp_true, color='#d62728', linewidth=1.5, label='Dynamic Chirp Input Current')
        ax1.set_title("Unseen Dynamic Stimulus Current Profile (Frequency Sweep)", fontsize=13, fontweight='bold')
        ax1.set_ylabel("Stimulus Magnitude", fontsize=11)
        ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2.plot(t_span, v_chirp_true, label='Ground Truth FHN Simulation', color='#1f77b4', linewidth=2.0)
        ax2.plot(t_span, v_chirp_pred_deep, '--', label=f'Deep Koopman Sobolev (Cheb: {cheb_deep:.4f}, MSE: {mse_deep:.4f})', color='#2ca02c', linewidth=2.0)
        ax2.set_title("Deep Koopman Sobolev Learning Global Trajectory Validation (True vs Predicted)", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Time (dimensionless)", fontsize=11)
        ax2.set_ylabel("Membrane Potential v", fontsize=11)
        ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        save_path = os.path.join(plots_dir, 'fhn_deep_koopman_verification.png')
        plt.savefig(save_path, dpi=300)
        print(f"\nSaved Deep Koopman validation plot to {save_path}")
        plt.close()

if __name__ == '__main__':
    main()
