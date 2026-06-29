# FHN Surrogate Modeling — Changes, Research, and Design Decisions

This report documents the full arc of work on the `fhn-koopman` project: what was
broken, what we tried, what we discarded and why, the research that informed each
turn, the final architecture and its mathematics, every code change, and the
results. It is written to be read top to bottom by someone who did not watch the
work happen.

> **One-paragraph summary.** The repo began as a Koopman model for the
> FitzHugh–Nagumo (FHN) neuron that scored **0%** on free-running rollouts. We
> first fixed the *architecture* (a Stuart–Landau radial law that makes the limit
> cycle a true attractor), then briefly went down a wrong path (SINDyc, which
> *rediscovers the known equations* instead of being a surrogate — deleted at the
> user's direction). The real goal then crystallized: a **trained amortized
> operator surrogate** `G(x₀, I_ext(·), t) → x(t)` that, from the measured state
> now and a time-varying current, returns the full state at **any** query time in
> **one forward pass — no recursive simulation** — and scales to Hodgkin–Huxley
> (HH) by changing data only. We designed and built the **Phase-Warped Floquet
> Operator (PWFO)**: phase accumulates as a prefix-sum of instantaneous frequency,
> the cycle is a Fourier series in that phase, transients decay as isostable modes.
> This evaluates any t at O(1) and provably extrapolates a persistent oscillation.

### Outcome at a glance (v1)
- **Concept proven.** One forward pass per query, **no recursion** (t≈1 ≡ t≈10⁶ in
  wall-clock); the prediction error does **not grow** from the trained window
  (t≤300) out to t=1500 (~40 cycles) — finite and bounded throughout.
- **Amplitude is delivered.** For constant/slow currents the cycle's
  amplitude/period/waveform are correct and the amplitude stays flat (0.93–0.955)
  over 30+ extrapolated cycles; only absolute phase slowly drifts.
- **Design regime works (PWFO v2).** With anchored operator training (§5.2), the
  one-shot PWFO tracks constant/slowly-varying *designed-intervention* currents well
  from a measured state over 3 cycles (const NRMSE 0.19, step 0.37, ramp 0.41), and
  is the only path that evaluates **arbitrary/unbounded t instantly**. It is
  fundamentally limited on fast/oscillatory currents.
- **Full range solved (hybrid).** A second model — a learned **recurrent flow-map
  stepper** — achieves **full waveform + phase across ALL current types**
  (mean 3-cycle NRMSE **0.06**, ~10× better than PWFO; chirp 0.016, pulse 0.026 —
  the regime PWFO could not touch). The shipped **hybrid** routes: flow-map for
  accuracy at finite horizons (any current), PWFO for instant arbitrary-far-t on
  slow currents. This meets the "full waveform + phase, full range" goal (§5.4).

---

## 1. Problem evolution (how the goal changed)

**Stage 0 — the inherited model.** A "deep Koopman" model lifted the 2-D FHN state
into a latent space evolving under a control-conditioned linear operator (a
"spiral" backbone). Diagnostics showed **success rate = 0%**: free-running
rollouts decayed to zero or failed to lock onto the right oscillation.

**Stage 1 — architecture fix (Stuart–Landau).** Root cause: the spiral operator
forced the radial growth rate `σ = −softplus(·) ≤ 0`, which can only *decay* or
sit on a *marginal* (non-attracting) cycle. A limit cycle needs an *attractor*.
We replaced the radial law with the **Hopf normal form**
`ṙ = σ₀(u) r − β(u) r³` (β>0), giving a genuinely attracting cycle at
`r*=√(σ₀/β)` that is bounded by construction. This made long-horizon rollouts
stable (no decay, no blow-up) — verified for 25+ periods.

**Stage 2 — the SINDyc misstep (discarded).** As a "faster route" we added SINDyc,
which recovered FHN's cubic ODE from data in ~2 s. It worked *too* literally: it
**rediscovered the governing equations we already know**. That is system
identification, not a surrogate, and it integrates recursively. Per the user, **all
SINDyc code was deleted.** The lesson sharpened the spec.

**Stage 3 — the real goal: an amortized, non-recursive operator.** The user wants
to *scale* (and swap FHN→HH), so the model must be a **learned surrogate** that
maps the **measured state now + the time-varying current** to the **full state at
any future time in a single forward pass**, with **no step-by-step integration at
inference**. This report is about that model.

The decisive I/O contract (confirmed with the user):

| | choice |
|---|---|
| output | full state `x(t) = (v, w)` (HH: full measured vector) |
| current input | a **time-varying** `I_ext(·)` profile (operator over functions) |
| query time | **arbitrary / unbounded** t |
| inference | **one forward pass**, no recursion |

---

## 2. Research (three multi-agent studies)

Three independent research/design studies were run (fan-out web research →
adversarial verification → synthesis). Key conclusions and citations:

### 2.1 "Faster route" study (Stage 2 context)
Ranked cheap dynamical-ML methods. It correctly flagged **SINDyc** as fastest for
*known cubic* systems, but the adversarial pass noted it is *equation discovery*,
needs full-state + derivative estimates, and is **not** a learned surrogate. This
directly motivated abandoning it once the goal clarified.

### 2.2 Architecture design panel (Stage 3)
Six architecture families were each developed, then adversarially critiqued against
the spec (one-shot arbitrary-t, oscillation extrapolation, time-varying control,
HH-scale, weak HW). Verdicts:

- **Winner: cumulative-phase-warp** — the *only* family whose one-shot arbitrary-t
  was *confirmed* (not partial): time enters via a prefix-sum of a positive
  frequency; persistence is structural.
- **Phase-amplitude / isostable** — most principled; lets us supervise frequency &
  Floquet rates from the known equations.
- **DeepONet-periodic (HyperDeepONet)** — contributes the single-attractor trick
  (tie harmonics to k·Ω; route x₀ into phase + transient only).
- **Koopman closed-form `e^{Kt}`** — sound but a fixed linear latent gives *neutral*
  (non-attracting) cycles → phase/amplitude cannot self-correct.
- **Latent-linear-propagator / FNO-over-time** — rejected (linear obstruction;
  FNO not naturally unbounded-t).

Synthesis = the **Phase-Warped Floquet Operator (PWFO)**: phase-warp backbone +
tied-harmonic single-attractor factorization + physics-supervised frequency.

### 2.3 Frontier (2023–2025) research — *validated the design*
Recency-focused survey of Koopman nets, DeepONets/neural operators, nonlinear
surrogates, each adversarially verified:

- **Phase autoencoder for limit cycles** (Yawata, Fukami, Taira, Nakao, *Chaos*
  2024, arXiv:2403.06992) — "best-in-class oscillation guarantee," persistence is a
  *hard architectural invariant*, **already demonstrated on FHN and HH**. This is
  PWFO's backbone. ✓
- **Laplace Neural Operator** (Cao et al., *Nat. Mach. Intell.* 2024,
  arXiv:2303.10528) — adopt the analytic exp-trunk (transient + steady), *not* its
  LTI operator (it can't self-oscillate at a frequency absent from the forcing).
- **Continuous-time Koopman `e^{Kt}`** — confirmed a linear latent is amplitude-
  *neutral*; and **bilinear control has no closed form** (Magnus/time-ordered
  exponential) — i.e. per-segment `expm` chaining *is* the forbidden recursion.
- **NCDE-DeepONet** (Time-Resolution-Independent Operator Learning, CMAME 2025,
  arXiv:2507.02524) — best mechanism for encoding the **time-varying current**: a
  Neural CDE in the branch integrates over the *input* once (not over output time),
  so it does not violate the no-recursion rule. Queued as the next branch upgrade.
- **DeepONet-MPC / MS-DeepONet** (arXiv:2505.18008) — the literal `G(x₀,u(·),t)`
  template, benchmarked on the Van der Pol relaxation oscillator (same Hopf family).

**Net:** every load-bearing PWFO choice is corroborated by published, verified work.

---

## 3. The PWFO architecture and its mathematics

### 3.1 The core idea
A limit-cycle trajectory is "a waveform traversed at some phase speed, plus a
transient that dies." Write the state as

$$x(t) = \underbrace{\mu + \sum_{k=1}^{K} A_k\cos k\Phi(t) + B_k\sin k\Phi(t)}_{\text{steady cycle (Fourier in phase)}} \;+\; \underbrace{\sum_{j=1}^{m}\psi_j(t)\sum_{k} \big(C^{c}_{jk}\cos k\Phi + C^{s}_{jk}\sin k\Phi\big)}_{\text{decaying isostable transient}}$$

with

$$\Phi(t) = \phi_0 + \int_0^t \omega\big(u(\tau)\big)\,d\tau, \qquad \psi_j(t) = \rho_{0,j}\,\exp\!\Big(\int_0^t \kappa_j\big(u(\tau)\big)\,d\tau\Big),\;\; \kappa_j<0.$$

### 3.2 Why this satisfies every hard constraint
- **One-shot arbitrary-t, no recursion.** Time enters *only* through `cos/sin(kΦ)`
  and the envelope `exp(∫κ)`. The integrals are **prefix-sums** over the current
  profile — computed once, O(S); then *any* query t is a gather + evaluate, O(1).
  We confirmed wall-clock at t≈1 and t≈10⁶ is **identical** (no hidden t-loop).
- **Persistent oscillation (no decay/blow-up).** `cos(kΦ)` is bounded and periodic
  for all t; when the current is held, `ω` is constant so `Φ` grows linearly and the
  waveform repeats forever. The transient `exp(∫κ)`, κ<0, vanishes. So amplitude
  cannot drift to 0 or ∞ — it is a structural invariant (the Nakao guarantee).
- **Single attractor.** The cycle coefficients `(μ, A_k, B_k)` depend on the
  **current only**, never on `x₀`; all initial conditions relax onto the same cycle.
  `x₀` enters *only* through the initial phase `φ₀` and transient amplitude `ρ₀`.
- **Time-varying current.** Instantaneous frequency `ω(u(τ))` and decay `κ(u(τ))`
  are per-segment; the phase accumulates them by prefix-sum (the adiabatic limit).
  The waveform is additionally conditioned on the **local current** at the query
  time, so amplitude/shape track a changing drive.
- **Bounded phase drift.** Any error in `ω` makes absolute phase drift ∝ ε·t — the
  one fundamental limit of *any* free oscillator surrogate. We bound it by
  **supervising `ω(u)` against the measured cycle frequency** (`L_freq`, from
  `fhn_theory.cycle_frequency_over_u`) and report phase-invariant metrics.

### 3.3 Connection to the Stage-1 Stuart–Landau fix
The radius law `ṙ = σ₀r − βr³` integrates **exactly** to a logistic flow, so the
Koopman model already evaluates the *radius* closed-form in time. PWFO generalizes
that idea: instead of one closed-form radius, it closed-forms the **phase** and a
**Fourier waveform**, which is what buys correct spike shape and one-shot arbitrary
t. (The exact logistic step was validated against fine Euler integration to <1e-5.)

### 3.4 Losses
- `L_state` — z-scored MSE of `x̂(t)` vs truth at sampled query times (late t
  oversampled to pin the cycle, early t to pin transients).
- `L_freq` — **mandatory** supervision `‖ω(u) − ω_measured(u)‖²` over the firing
  band; bounds phase drift and removes the sinusoid-frequency non-convexity.
- `L_range` — matches the predicted vs true per-trajectory amplitude (peak-to-peak),
  added to pin spike amplitude (the truncated Fourier series otherwise undershoots
  the sharp relaxation peak by ~10%).
- (Planned) `L_deriv` physics residual `‖dx̂/dt − f(x̂,u)‖²` evaluated *also* at
  t≫T_train (free, closed-form) to certify extrapolation; `L_period` cycle
  consistency.

---

## 4. Code changes

### 4.1 New files (the surrogate)
- `operator_data.py` — dataset generator: FHN under **random time-varying currents**
  (const/step/ramp/pulse/chirp/sines/OU/piecewise), broad ICs, long horizon; emits
  a train set (t_max=300), val, and a **far set (t_max=1500)** for unbounded-t tests.
- `pwfo_model.py` — the PWFO forward map: `init_pwfo`, `forward(params,cfg,x0,u,t,dt)`,
  cumulative-phase prefix-sum, Fourier-in-phase cycle, isostable transient, DeepONet
  branch; `local_waveform` flag for local-current conditioning.
- `pwfo_freq_table.py` — precomputes `ω_measured(u)` for frequency supervision.
- `pwfo_train.py` — trainer; **anchored-window operator training** (random anchor
  time t₀, predict the next ~3 cycles), constant-current core → general profiles,
  losses `L_state + L_freq + L_range`, AdamW + cosine.
- `pwfo_eval.py` — far-horizon **one-shot** eval vs the true integrator: amplitude
  flatness, phase drift, pointwise NRMSE, and a **timing-invariance** assertion.
- `pwfo_figures.py` — report figures.
- `flowmap_model.py` — the recurrent **flow-map stepper** (Markov residual integrator
  on a coarse grid) + checkpointed rollout.
- `flowmap_train.py` — multi-step (BPTT) curriculum trainer over all current types.
- `flowmap_eval.py` — per-current-type rollout NRMSE + speed.
- `hybrid_model.py` — the shipped **hybrid**: routes between flow-map (accuracy,
  finite horizon) and PWFO (instant arbitrary-far-t, slow currents).

### 4.2 Stage-1 architecture changes (kept)
- `model.py` — Stuart–Landau radial law (`spiral_sl_coeffs`, `_sl_radius_step`),
  exact logistic step, `radial` config flag (`stuart_landau` default, `clamped`
  legacy). `train.py` — full-cycle curriculum + gradient-checkpointed rollout.
  `stability_eval.py`, `intervention.py` — long-horizon + inverse-design tooling.

### 4.3 Deletions / cleanup
- **SINDyc removed** (`sindyc.py`, `fit_sindyc.py`, `tests/test_sindyc.py`, its
  artifacts) — per user; equation discovery, not a surrogate.
- All code comments stripped repo-wide (user preference; brief one-line docstrings
  only); 5 unused imports removed; transient `data/*.log` removed; README de-SINDyc'd.
- Left in place for the article narrative (superseded but not deleted):
  `deep_koopman_hypernetwork.py`, `sweep_latent_dimension.py` (duplicate net+train
  code that `model.py`/`train.py` replaced) — pending user confirmation to delete.

---

## 5. Results

### 5.1 Constant-current core — the load-bearing claim (PROVEN)
Trained on t≤300 (~8 cycles), queried in **one forward pass** out to t=1500 (~40
cycles), vs the true integrator (`plots/pwfo/fig2_core_farhorizon.png`,
`fig3_amplitude_flat.png`):

| metric | value | meaning |
|---|---|---|
| amplitude flatness (far/early) | **0.93–0.955** | oscillation neither decays nor grows over 30+ extra cycles |
| pointwise NRMSE early (trained) | 0.225 | accurate spike shape in-window |
| pointwise NRMSE far (extrapolated) | 0.39 | cycle correct; offset is accumulated phase |
| timing t≈1 vs t≈10⁶ | **69.9 ms ≡ 69.8 ms** | identical cost ⇒ **no hidden recursion** |

The figure shows the far-horizon cycle has the **right amplitude, period, and
waveform**; only the absolute phase has slipped — the predicted ∝ε·t drift, the one
fundamental limit. **Amplitude (the user's "exact amplitudes" target) is preserved.**

![How PWFO evaluates any t in one shot](plots/pwfo/fig1_concept.png)

![Core: sustained oscillation 30+ cycles past training](plots/pwfo/fig2_core_farhorizon.png)

![Amplitude stays flat over extrapolated cycles](plots/pwfo/fig3_amplitude_flat.png)

### 5.2 General time-varying-current model (v2 — anchored operator training)
**Key training change (v2):** train from random **anchor times** — `x₀` = the state
at *any* point along a trajectory, predict the next ~3 cycles — instead of always
anchoring at t=0. This matches the actual use ("measured state *now* → predict
ahead"), supplies near-attractor initial conditions (real neuron states, not random
far-field ICs), and multiplies the training data. It is the single biggest
correctness fix.

**Use-case metric** — anchored 3-cycle one-shot NRMSE, by current type (val):

| current type | v2 NRMSE | regime |
|---|---|---|
| **const** | **0.19** | slow — design regime, good (≈ dedicated core) |
| **step** | **0.37** | slow — good |
| **ramp** | **0.41** | slow — good |
| piecewise | 0.67 | fast |
| ou | 0.70 | fast |
| pulse | 0.72 | fast — entrainment |
| chirp | 0.77 | fast |
| sines | 0.77 | fastest |

**Conclusion:** for **constant / slowly-varying (designed-intervention) currents —
the practical use case — v2 is good**: it tracks amplitude, period, and (smoothed)
waveform over 3 cycles from a measured state in one pass (`plots/pwfo/fig4_general_slow.png`).
Fast/oscillatory currents remain hard (`fig5_general_fast.png`): the adiabatic phase
model breaks (entrainment), and strong fast jumps drive the neuron off any slowly-
moving cycle — a regime that is fundamentally hard for a *closed-form* phase model
(first-order PRC is weak-forcing theory and does not rescue strong fast jumps).

Remaining accuracy levers (not yet applied): (i) **higher K / a SIREN decoder** to
sharpen the relaxation spike (Fourier truncation at K=20 smooths it); (ii) tighter
`ω` supervision to cut residual phase drift; (iii) the **PRC stage** for moderate-
speed forcing; (iv) restrict training to the designed-current regime for a tighter
in-regime model.

![Design regime (slow current): measured state → 3 cycles, one shot](plots/pwfo/fig4_general_slow.png)

![Stress regime (fast current): the adiabatic limit](plots/pwfo/fig5_general_fast.png)

### 5.3 Speed (PWFO)
~33k params (core) / ~170k (general); **one forward pass is O(1) in the query
time** (a profile prefix-sum O(S) shared across all queries, then gather+evaluate).
Batched far-horizon query (32 trajectories × 7500 query times to t=1500) ≈ 70 ms on
the 6 GB GPU; cost is identical at t≈1 and t≈10⁶.

### 5.4 The recurrent flow-map + the hybrid (full range — SOLVED)
PWFO's fast-current limit is fundamental to *closed-form phase*, so the full-range
case is handled by a second model: a learned **flow-map stepper**

$$x_{t+\Delta} = x_t + g_\theta\big(x_t,\, u_t,\, u_{t+\Delta}\big),\qquad \Delta = 4\,dt = 0.2,$$

a Markov residual integrator on a *coarse* grid, trained by a multi-step (BPTT)
curriculum (rollout horizon grows 8→560 coarse steps; gradient-checkpointed). It is
recurrent (a scan over the horizon) but tiny (~20k params), differentiable in the
current, and GPU-batched — and the coarse step makes it a genuine speedup over a
*stiff* solver (the real win for Hodgkin–Huxley).

**Accuracy — anchored 3-cycle rollout NRMSE by current type (val):**

| current | flow-map | PWFO (one-shot) |
|---|---|---|
| const | 0.019 | 0.16 |
| step | 0.022 | 0.37 |
| ramp | 0.234 | 0.41 |
| pulse | 0.026 | 0.72 |
| chirp | **0.016** | 0.77 |
| sines | 0.110 | 0.77 |
| ou | 0.026 | 0.70 |
| piecewise | 0.025 | 0.67 |
| **MEAN** | **0.060** | 0.585 |

The flow-map captures **full waveform + phase across all current types** — ~10×
better than PWFO, decisively winning the fast/oscillatory regime PWFO cannot track
(`plots/pwfo/fig6_flowmap_fast.png`: on a chirp current the flow-map overlays the
truth, PWFO is off). Speed: 25 ms for 32 trajectories × 3 cycles (554 coarse steps).

![Flow-map vs PWFO on a fast current](plots/pwfo/fig6_flowmap_fast.png)

**The hybrid (`hybrid_model.py`)** routes per call:
- **finite horizon, any current → flow-map** (accuracy, full waveform+phase);
- **very-far / unbounded t + slow current → PWFO** (instant one-shot; the stepper
  would need too many steps).

This combination meets the goal — **full waveform + phase over the full current
range**, with PWFO covering the arbitrary-t corner. Trade-off: the flow-map's cost
grows with horizon (O(t/Δ) steps), so truly unbounded-t is PWFO-only and thus
restricted to slow currents — the one genuinely unreachable corner (fast forcing +
unbounded t has no finite-cost exact answer).

---

## 6. Path to Hodgkin–Huxley
Both models swap to HH by **data only**. The **flow-map** is the HH workhorse and the
real speedup: HH is stiff, so its true solver is expensive, whereas a coarse learned
stepper (`d` 2→4, a wider MLP, `Δ` tuned to the spike width) integrates cheaply — and
the `flowmap_*` code is unchanged but for the data. PWFO also ports (state dim 2→4,
isostable modes 1→3, harmonics up; a SIREN decoder for sharper spikes; a small
**regime gate** for spiking-vs-quiescent near the Hopf points). Generate the HH
dataset with a stiff diffrax solver (Kvaerno) and retrain.

## 7. Honest limitations
1. **PWFO phase drift / fast currents** — fundamental to a *closed-form* oscillator
   surrogate. **Now handled by the hybrid:** the recurrent flow-map covers fast
   currents and full waveform+phase at finite horizons (§5.4). PWFO is used only for
   instant arbitrary-far-t on slow currents.
2. **The one unreachable corner** — *fast forcing + truly unbounded t*: the flow-map
   is exact but its cost is O(t/Δ) (no instant jump), and PWFO is instant but only
   accurate for slow currents. There is no finite-cost exact answer here; pick speed
   (PWFO, slow only) or accuracy (flow-map, bounded horizon).
3. **flow-map ramp/sines (0.23 / 0.11)** — the weakest types; slow regime-drift
   accumulation. More training / smaller Δ would tighten them.
4. **Networks of units** — synaptic coupling makes `I_ext` an unknown; the per-unit
   surrogate then iterates (a few Picard/Gauss–Seidel sweeps) rather than one shot.

## 8. Reproduce
```bash
python operator_data.py            # dataset (random time-varying currents)
python pwfo_freq_table.py          # omega(u) table for frequency supervision
python pwfo_train.py --mode core --K 20 --steps 8000 --out data/pwfo_core_k20.pkl
python pwfo_eval.py  --mode core   --model data/pwfo_core_k20.pkl
python pwfo_train.py --mode general --K 20 --m 2 --steps 10000 --window 2400  # anchored
python pwfo_eval.py  --mode general --model data/pwfo_general.pkl
python flowmap_train.py --steps 6000 --stride 4    # recurrent flow-map (full range)
python flowmap_eval.py  --model data/flowmap.pkl
python pwfo_figures.py             # -> plots/pwfo/
# hybrid: hybrid_model.predict(pwfo, flow, dt, x0, u_profile, t_query) -> (x, route)
```
