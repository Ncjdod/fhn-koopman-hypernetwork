import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
from dynamics import fhn_vector_field

def simulate_fhn_diffrax(y0, t_span, a=0.7, b=0.8, tau=12.5,
                         I_type='constant', I_val=0.5, rtol=1e-8, atol=1e-10):
    """High-accuracy reference integration with diffrax (adaptive Dopri5)."""
    import diffrax

    y0 = jnp.asarray(y0, dtype=jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32)
    args = (a, b, tau, I_type, I_val)
    term = diffrax.ODETerm(lambda t, y, _: fhn_vector_field(t, y, args))
    sol = diffrax.diffeqsolve(
        term, diffrax.Dopri5(),
        t0=float(t_span[0]), t1=float(t_span[-1]),
        dt0=float(t_span[1] - t_span[0]),
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_span),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=2_000_000,
    )
    return sol.ys

def simulate_fhn(y0, t_span, a=0.7, b=0.8, tau=12.5, I_type='constant', I_val=0.5):
    y0 = jnp.asarray(y0, dtype=jnp.float32)
    dt = t_span[1] - t_span[0]

    def step_fn(y, t):
        args = (a, b, tau, I_type, I_val)
        k1 = fhn_vector_field(t, y, args)
        k2 = fhn_vector_field(t + dt / 2.0, y + dt / 2.0 * k1, args)
        k3 = fhn_vector_field(t + dt / 2.0, y + dt / 2.0 * k2, args)
        k4 = fhn_vector_field(t + dt, y + dt * k3, args)
        y_new = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_new, y

    _, ys = jax.lax.scan(step_fn, y0, t_span[:-1])
    return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)

def simulate_fhn_batch(y0_batch, t_span, I_type, I_val_batch, a=0.7, b=0.8, tau=12.5):
    vmapped_solve = jax.vmap(
        lambda y0, I_val: simulate_fhn(y0, t_span, a, b, tau, I_type, I_val),
        in_axes=(0, 0)
    )
    return vmapped_solve(y0_batch, I_val_batch)

def fit_fhn_parameters(y0, t_span, noisy_target, I_type, I_val, lr=0.02, steps=150):
    y0 = jnp.asarray(y0, dtype=jnp.float32)
    noisy_target = jnp.asarray(noisy_target, dtype=jnp.float32)
    init_params = jnp.array([0.5, 0.5, 10.0])

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_params)

    def loss_fn(params, y0, t_span, target, I_type, I_val):
        a, b, tau = jnp.abs(params)
        ys_pred = simulate_fhn(y0, t_span, a, b, tau, I_type, I_val)
        return jnp.mean((ys_pred - target) ** 2)

    def scan_body(carry, step):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, y0, t_span, noisy_target, I_type, I_val)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        def print_cond(arg):
            step_val, params_val, loss_val = arg
            a_cur = jnp.abs(params_val[0])
            b_cur = jnp.abs(params_val[1])
            tau_cur = jnp.abs(params_val[2])
            jax.debug.print(
                "Step {step:03d} | Loss: {loss:.6f} | Current guesses: a={a:.4f}, b={b:.4f}, tau={tau:.4f}",
                step=step_val,
                loss=loss_val,
                a=a_cur,
                b=b_cur,
                tau=tau_cur
            )

        jax.lax.cond(
            (step % 15 == 0) | (step == steps - 1),
            print_cond,
            lambda arg: None,
            (step, params, loss)
        )

        return (params, opt_state), loss

    print("\nStarting Parameter Fitting Demo with Optax (JAX Stack)...")
    a_init, b_init, tau_init = init_params
    print(f"Initial guesses: a={a_init:.2f}, b={b_init:.2f}, tau={tau_init:.2f}")

    steps_arr = jnp.arange(steps)
    (params_opt, _), loss_history = jax.lax.scan(scan_body, (init_params, opt_state), steps_arr)

    fitted_vals = jnp.abs(params_opt)
    fitted_params = {
        "a": float(fitted_vals[0]),
        "b": float(fitted_vals[1]),
        "tau": float(fitted_vals[2])
    }

    fitted_trajectory = simulate_fhn(
        y0, t_span,
        a=fitted_params["a"],
        b=fitted_params["b"],
        tau=fitted_params["tau"],
        I_type=I_type,
        I_val=I_val
    )

    loss_history_list = [float(x) for x in loss_history]

    return fitted_params, loss_history_list, fitted_trajectory
