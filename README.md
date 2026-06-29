# Deep Koopman Operators for the FitzHugh–Nagumo Neuron

A JAX framework that learns a **control-dependent Koopman operator** for the
FitzHugh–Nagumo (FHN) spiking-neuron model: a nonlinear encoder lifts the 2-D
neuron state into a latent space where the dynamics evolve **linearly**, so that
a stimulus-driven neuron can be predicted by a linear operator whose spectrum is
conditioned on the input current.

The full theory (from "what is a neuron model" up to Koopman spectral theory and
Floquet analysis) and all results are in **[`ARTICLE.md`](ARTICLE.md)**. This
README is the quick map.

> **Stuart–Landau Koopman** (`model.py`, `train.py`) — the latent radius obeys the
> Hopf normal form `ṙ = σ₀(u)r − β(u)r³`, so the limit cycle is an *attractor*:
> bounded and non-collapsing **by construction** over any horizon (verified for 25+
> periods, `stability_eval.py`). Inverse design / intervention: `intervention.py`.
> An amortized, non-recursive operator surrogate `G(x₀, I_ext(·), t) → x(t)` is in
> progress (`operator_data.py` builds its dataset).

---

## What this models

$$\dot v = v - \tfrac{1}{3}v^3 - w + u(t), \qquad \dot w = \tfrac{1}{\tau}(v + a - b w)$$

`v` is the membrane potential, `w` the recovery variable, `u(t)` the injected
current (the control). For our parameters (`a=0.7, b=0.8, τ=12.5`) the neuron
undergoes a Hopf bifurcation near `u≈0.33`/`u≈1.42` and fires on a limit cycle of
period ≈ 37 in between.

## Two Koopman backbones (`model.py`)

| backbone | operator | decoder | key property |
|---|---|---|---|
| **`spiral`** (primary) | radius-conditioned block-diagonal spiral, `σ=−softplus≤0` | linear | amplitude-dependent frequency; non-expansive rollouts |
| **`bilinear`** (secondary) | control-affine `ż=(A+uB)z` | fixed, state-in-latent | identifiable, mode-coupling, exact reconstruction |

Both share one encoder/operator/decoder interface and one set of losses
(reconstruction, latent linearity, multi-step prediction) with optional,
per-dimension-normalised **Sobolev** (derivative-matching) terms.

## Repository layout

```
model.py            # shared model: both backbones, operators, losses (the core)
fhn_theory.py       # ground-truth Jacobian spectrum μ±(u) and limit-cycle period
dynamics.py         # FHN vector field and stimulus-current generators
simulation.py       # RK4 batch integrator + diffrax high-accuracy reference solver
data_gen.py         # builds the t_max=80 train/val dataset + normalisation scales
train.py            # curriculum trainer (n_predict 40→100→200), both backbones
run_training.py     # trains both backbones and saves pickles
diagnostics.py      # the 5 diagnostics + success rate (plots -> plots/, json -> data/)
phase_space_analyzer.py  # constant-current stability/bifurcation classifier
fitzhugh_nagumo.py  # standalone FHN simulator CLI + parameter-fitting demo
sweep_latent_dimension.py# latent-dimension sweep utility
tests/              # pytest: theory, model math, stability, jvp, training
ARTICLE.md          # full theoretical write-up + results
```

## Quick start

```bash
pip install -r requirements.txt          # jax, diffrax, equinox, optax, matplotlib, scipy
python data_gen.py                       # -> data/fhn_koopman_t80.npz
python train.py --backbone spiral        # -> data/koopman_spiral.pkl
python diagnostics.py --model data/koopman_spiral.pkl   # -> plots/diag_*.png, data/diag_spiral.json
pytest -q                                # run the test suite
```

To reproduce both trained models used in the article: `python run_training.py`.

> **Compute note.** Everything runs on CPU (JAX falls back automatically). The
> default configurations are sized for CPU; a GPU is not required.

## Diagnostics produced

`diagnostics.py` writes, for each backbone:

* **D0** learned spectrum `σ(u), ω(u)` vs the Jacobian `μ±(u)` and the measured
  cycle frequency — *the sanity check to run first* (fix #9);
* **D1** eigenvalue trajectories over the latent radius (the amplitude-frequency
  coupling, fix #1);
* **D2** Floquet multipliers of the recovered cycle vs the true monodromy;
* **D3** Koopman-mode (harmonic) decomposition of the reconstructed spike;
* **D4** long-horizon rollout error decomposed by spectral band;
* a **success-rate** histogram over the held-out (unseen chirp-current) trajectories.
