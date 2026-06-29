import jax.numpy as jnp

def get_external_current(t, I_type, I_val):
    """Generates external stimulus current dynamically at time t."""
    constant_current = I_val
    step_current = jnp.where((t >= 10.0) & (t <= 80.0), I_val, 0.0)
    sine_current = I_val * (1.0 + 0.5 * jnp.sin(0.2 * t))
    pulse_current = jnp.where(jnp.mod(t, 20.0) <= 5.0, I_val, 0.0)
    chirp_current = I_val * (1.0 + 0.5 * jnp.sin(0.05 * t + 0.001 * t**2))

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

def fhn_vector_field(t, y, args):
    """Computes the vector field for the FitzHugh-Nagumo ordinary differential equations."""
    v, w = y
    a, b, tau, I_type, I_val = args
    I_ext = get_external_current(t, I_type, I_val)
    dv_dt = v - (v**3) / 3.0 - w + I_ext
    dw_dt = (v + a - b * w) / tau
    return jnp.stack([dv_dt, dw_dt])
