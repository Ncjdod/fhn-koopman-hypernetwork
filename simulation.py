import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optax
from dynamics import fhn_vector_field

def simulate_fhn(y0, t_span, a=0.7, b=0.8, tau=12.5, I_type='constant', I_val=0.5):
    """Simulates the FHN model using the Diffrax Dopri5 (RK5) fixed-step solver."""
    term = diffrax.ODETerm(fhn_vector_field)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=t_span)
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[-1],
        dt0=t_span[1] - t_span[0],
        y0=jnp.asarray(y0, dtype=jnp.float32),
        args=(a, b, tau, I_type, I_val),
        saveat=saveat
    )
    return sol.ys

def simulate_fhn_batch(y0_batch, t_span, I_type, I_val_batch, a=0.7, b=0.8, tau=12.5):
    """Simulates a batch of FHN trajectories in parallel using jax.vmap."""
    vmapped_solve = jax.vmap(
        lambda y0, I_val: simulate_fhn(y0, t_span, a, b, tau, I_type, I_val),
        in_axes=(0, 0)
    )
    return vmapped_solve(y0_batch, I_val_batch)

def fit_fhn_parameters(y0, t_span, noisy_target, I_type, I_val, lr=0.02, steps=150):
    """Fits the FHN parameters (a, b, tau) to noisy target data using Optax and JAX autodiff."""
    y0 = jnp.asarray(y0, dtype=jnp.float32)
    noisy_target = jnp.asarray(noisy_target, dtype=jnp.float32)
    init_params = jnp.array([0.5, 0.5, 10.0])
    
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_params)
    
    @jax.jit
    def loss_fn(params, y0, t_span, target, I_type, I_val):
        a, b, tau = jnp.abs(params)
        term = diffrax.ODETerm(fhn_vector_field)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=t_span)
        
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[-1],
            dt0=t_span[1] - t_span[0],
            y0=y0,
            args=(a, b, tau, I_type, I_val),
            saveat=saveat
        )
        return jnp.mean((sol.ys - target) ** 2)

    @jax.jit
    def train_step(params, opt_state, y0, t_span, target, I_type, I_val):
        loss, grads = jax.value_and_grad(loss_fn)(params, y0, t_span, target, I_type, I_val)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    params = init_params
    loss_history = []
    
    print("\nStarting Parameter Fitting Demo with Optax (JAX + Diffrax Stack)...")
    print(f"Initial guesses: a={params[0]:.2f}, b={params[1]:.2f}, tau={params[2]:.2f}")
    
    for step in range(steps):
        params, opt_state, loss = train_step(params, opt_state, y0, t_span, noisy_target, I_type, I_val)
        loss_history.append(float(loss))
        
        if step % 15 == 0 or step == steps - 1:
            a_cur, b_cur, tau_cur = np.abs(params)
            print(f"Step {step:03d} | Loss: {float(loss):.6f} | Current guesses: a={a_cur:.4f}, b={b_cur:.4f}, tau={tau_cur:.4f}")
            
    fitted_vals = np.abs(params)
    fitted_params = {"a": float(fitted_vals[0]), "b": float(fitted_vals[1]), "tau": float(fitted_vals[2])}
    
    fitted_trajectory = simulate_fhn(
        y0, t_span, 
        a=fitted_params["a"], 
        b=fitted_params["b"], 
        tau=fitted_params["tau"], 
        I_type=I_type,
        I_val=I_val
    )
    
    return fitted_params, loss_history, fitted_trajectory
