# FitzHugh-Nagumo Koopman Hypernetwork (Stuart-Landau Model)

This repository contains the code and design documents for the FitzHugh–Nagumo (FHN) Koopman surrogate model based on a Stuart-Landau (Hopf normal-form) latent space.

> **Project Status**: This architecture was unsuccessful for general surrogate modeling of FHN dynamics. It has been replaced by the **Phase-Warped Floquet Operator (PWFO) and Recurrent Flow-Map hybrid model** in the new repository: **[fhn-operator-surrogate](https://github.com/Ncjdod/fhn-operator-surrogate)**.

## The Tried Design: Stuart-Landau Koopman Hypernetwork

The goal was to learn a coordinate mapping (encoder/decoder) in which the non-linear FitzHugh-Nagumo vector field becomes linear and evolved by a parameterized matrix operator.

### Why it was Unsuccessful

1. **Linear Operator Limit Cycle Obstruction**:
   A purely linear latent space $\dot z = K z$ cannot hold an attracting limit cycle. Evolving a constant amplitude requires the growth/decay rate $\sigma$ to be exactly $0$, which is marginal (not attracting). Any tiny error makes the predicted oscillation either decay to zero or blow up to infinity.
2. **The Stuart-Landau Attractor Attempt**:
   We updated the latent space to follow the Stuart-Landau equation:
   $$\dot r_j = \sigma_{0,j}(u)r_j - \beta_j(u)r_j^3$$
   While this successfully resolved the structural stability (boundedness and non-collapse), the model still suffered from:
   * **Phase Drift**: Pointwise predictions over multiple cycles accumulated phase errors, leading to a slide relative to the ground truth.
   * **Amplitude Undershoot**: The linear decoder struggled to capture the sharp relaxation spike of FHN, leading to an amplitude drop (e.g. ~2.5 vs the true ~3.8).

## The Shipped Solution

For a fully working, non-recursive surrogate that resolves these limitations, please see the new **[fhn-operator-surrogate](https://github.com/Ncjdod/fhn-operator-surrogate)** repository, which contains the **Phase-Warped Floquet Operator (PWFO)** and **Recurrent Flow-Map** hybrid model.
