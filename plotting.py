import numpy as np
import matplotlib.pyplot as plt

def plot_results(t_span, ys, a, b, tau, I_type, I_val, u_data=None, fitted_data=None, noisy_target=None, save_path=None, show_plot=True):
    """Plots FHN membrane potential and recovery variable time series."""
    v = ys[:, 0]
    w = ys[:, 1]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(12, 6))

    c_v = '#1f77b4'
    c_w = '#ff7f0e'

    plt.plot(t_span, v, label=r'Membrane Potential $v(t)$', color=c_v, linewidth=2.0)
    plt.plot(t_span, w, label=r'Recovery Variable $w(t)$', color=c_w, linewidth=2.0)

    if u_data is not None:
        plt.plot(t_span, u_data, label=r'Stimulus Current $I_{ext}(t)$', color='#d62728', linewidth=1.5, linestyle=':', alpha=0.9)

    if noisy_target is not None:
        plt.scatter(t_span[::5], noisy_target[::5, 0], color='black', alpha=0.3, s=8, label='Noisy Target $v_{meas}$')
    if fitted_data is not None:
        plt.plot(t_span, fitted_data[:, 0], '--', color='#9467bd', linewidth=1.5, label='Fitted $v_{opt}$')

    plt.title(f"FitzHugh-Nagumo Model Dynamics\n(a={a:.2f}, b={b:.2f}, \u03c4={tau:.2f}, Current={I_type} ({I_val:.2f}))", 
              fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Time (dimensionless)", fontsize=12)
    plt.ylabel("State Magnitude", fontsize=12)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.xlim(t_span[0], t_span[-1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization plot to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()
