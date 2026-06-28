import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from dynamics import get_external_current
from simulation import simulate_fhn_batch

def init_mlp_params(layers, key):
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
    activation = x
    for i in range(len(params) - 1):
        layer = params[i]
        activation = jax.nn.swish(jnp.dot(activation, layer["w"]) + layer["b"])
    layer = params[-1]
    return jnp.dot(activation, layer["w"]) + layer["b"]

def compute_fhn_derivatives(ys, u_data, a=0.7, b=0.8, tau=12.5):
    v = ys[:, :, 0]
    w = ys[:, :, 1]
    dv = v - (v ** 3) / 3.0 - w + u_data
    dw = (v + a - b * w) / tau
    return jnp.stack([dv, dw], axis=2)

def apply_koopman_operator(z, u, params_hyper, dt):
    hyper_out = forward_mlp(u[:, None], params_hyper)
    m = z.shape[1]
    sigma = hyper_out[:, :m]
    omega = hyper_out[:, m:]
    scale = jnp.exp(sigma * dt)
    cos_w = jnp.cos(omega * dt)
    sin_w = jnp.sin(omega * dt)
    z0 = z[:, :, 0]
    z1 = z[:, :, 1]
    z_next0 = scale * (z0 * cos_w - z1 * sin_w)
    z_next1 = scale * (z0 * sin_w + z1 * cos_w)
    return jnp.stack([z_next0, z_next1], axis=2)

def apply_continuous_koopman(z, u, params_hyper):
    hyper_out = forward_mlp(u[:, None], params_hyper)
    m = z.shape[1]
    sigma = hyper_out[:, :m]
    omega = hyper_out[:, m:]
    z0 = z[:, :, 0]
    z1 = z[:, :, 1]
    z_dot0 = sigma * z0 - omega * z1
    z_dot1 = omega * z0 + sigma * z1
    return jnp.stack([z_dot0, z_dot1], axis=2)

def compute_loss_term(diff, power):
    return (jnp.mean(jnp.abs(diff) ** power) + 1e-15) ** (1.0 / power)

def compute_losses(params_dict, trajectories, trajectory_dots, current_profiles, m, n_predict, dt, loss_power=2):
    params_enc = params_dict["enc"]
    params_dec = params_dict["dec"]
    params_hyper = params_dict["hyper"]

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
        z_curr_flat, u_curr_flat, params_hyper, dt
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
    z_dot_pred_flat = apply_continuous_koopman(z_flat_all, u_flat_all, params_hyper)
    loss_lin_sobolev = compute_loss_term(z_dot_flat - z_dot_pred_flat.reshape(-1, 2 * m), loss_power)
    loss_lin = loss_lin_state + loss_lin_sobolev

    stride = 20
    S = (T - 1 - n_predict) // stride

    def predict_forward_recursive(z_init, u_seq):
        def step(z, u):
            z_next = apply_koopman_operator(
                z[jnp.newaxis, :, :], jnp.array([u]), params_hyper, dt
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
        z_dot_preds_flat = apply_continuous_koopman(z_preds_m2, u_seq_flat, params_hyper)

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
    key1, key2, key3 = jax.random.split(key, 3)
    params_enc = init_mlp_params([2, 32, 32, 2 * m], key1)
    params_dec = init_mlp_params([2 * m, 32, 32, 2], key2)
    params_hyper = init_mlp_params([1, 16, 16, 2 * m], key3)

    b = np.zeros((2 * m,))
    b[:m] = -0.1
    b[m:] = 0.3
    params_hyper[-1]["b"] = jnp.array(b)

    return {
        "enc": params_enc,
        "dec": params_dec,
        "hyper": params_hyper
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    t_max = 10.0
    dt = 0.01
    batch_size = 8
    n_predict = 100
    loss_power = 6
    steps = 3000
    lr = 3e-4

    n_steps = int(t_max / dt) + 1
    t_span = jnp.linspace(0.0, t_max, n_steps)

    key = jax.random.PRNGKey(101)
    key1, key2, key3 = jax.random.split(key, 3)
    v0s = jax.random.uniform(key1, (batch_size,), minval=-2.0, maxval=1.0)
    w0s = jax.random.uniform(key2, (batch_size,), minval=-1.0, maxval=0.5)
    y0_batch = jnp.stack([v0s, w0s], axis=1)
    I_val_batch = jax.random.uniform(key3, (batch_size,), minval=0.2, maxval=1.2)

    ys = simulate_fhn_batch(y0_batch, t_span, 'sine', I_val_batch)
    u_data_batch = jax.vmap(lambda iv: get_external_current(t_span, 'sine', iv))(I_val_batch)

    m_values = list(range(3, 13))
    l_rec_results = []
    l_lin_results = []

    for m in m_values:
        print(f"Starting sweep training for latent dimension m = {m}...")
        init_key = jax.random.PRNGKey(42)
        params = init_deep_koopman_params(m, init_key)

        def get_loss_weights(step):
            if step < 600:
                return jnp.array([0.0, 2.0, 0.0])
            elif step < 1800:
                alpha = (step - 600) / 1200.0
                return jnp.array([alpha, 2.0 - alpha, alpha])
            else:
                return jnp.array([1.0, 1.0, 1.0])

        def total_loss_fn(params_dict, trajectories, current_profiles, weights):
            trajectory_dots = compute_fhn_derivatives(trajectories, current_profiles)
            l_rec, l_lin, l_pred = compute_losses(
                params_dict, trajectories, trajectory_dots, current_profiles, m, n_predict, dt, loss_power
            )
            return weights[0] * l_rec + weights[1] * l_lin + weights[2] * l_pred

        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=steps,
            alpha=1e-2
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr_schedule)
        )
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(p_vars, state, trajectories, current_profiles, weights):
            loss, grads = jax.value_and_grad(total_loss_fn)(p_vars, trajectories, current_profiles, weights)
            updates, state = optimizer.update(grads, state, p_vars)
            p_vars = optax.apply_updates(p_vars, updates)
            return p_vars, state, loss

        for step in range(steps):
            w_step = get_loss_weights(step)
            params, opt_state, loss_val = train_step(params, opt_state, ys, u_data_batch, w_step)

        trajectory_dots = compute_fhn_derivatives(ys, u_data_batch)
        final_l_rec, final_l_lin, _ = compute_losses(
            params, ys, trajectory_dots, u_data_batch, m, n_predict, dt, loss_power
        )

        l_rec_results.append(float(final_l_rec))
        l_lin_results.append(float(final_l_lin))
        print(f"Dimension m = {m} finished | Final Recon Loss: {float(final_l_rec):.6f} | Final Linearity Loss: {float(final_l_lin):.6f}")

    print("\nSweep complete. Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "royalblue"
    ax1.set_xlabel('Latent Space Dimension m', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reconstruction Loss (L_rec)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(m_values, l_rec_results, 'o-', color=color, linewidth=2, label='Reconstruction Loss (L_rec)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(m_values)

    ax2 = ax1.twinx()
    color = "darkorange"
    ax2.set_ylabel('Latent Linearity Loss (L_lin)', color=color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(m_values, l_lin_results, 's--', color=color, linewidth=2, label='Linearity Loss (L_lin)')
    ax2.tick_params(axis='y', labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.title('Koopman Hypernetwork Latent Space Dimension Sweep\n(Reconstruction vs Linearity Losses)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    save_path = os.path.join(plots_dir, 'fhn_latent_sweep.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved sweep plot to {save_path}")
    plt.close()

if __name__ == '__main__':
    main()
