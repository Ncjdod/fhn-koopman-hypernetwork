"""Tests for the SINDYc fast route: equation recovery, structural boundedness, long-horizon non-collapse/non-divergence, and closed-form current inver..."""
import os
import numpy as np
import pytest

import sindyc as S
from data_gen import generate

A, B, TAU = 0.7, 0.8, 12.5

@pytest.fixture(scope="module")
def model():
    data = generate(n_train=24, n_val=6, t_max=80.0, dt=0.05)
    return S.fit(np.array(data["ys_train"]), np.array(data["u_train"]),
                 float(data["dt"]), threshold=0.05), float(data["dt"])

def test_recovers_fhn_equations(model):
    m, _ = model
    names = m.names
    cv = {n: m.Xi[i, 0] for i, n in enumerate(names)}
    cw = {n: m.Xi[i, 1] for i, n in enumerate(names)}
    assert abs(cv["v"] - 1.0) < 0.1
    assert abs(cv["vvv"] - (-1.0 / 3.0)) < 0.1
    assert abs(cv["w"] - (-1.0)) < 0.1
    assert abs(cv["u"] - 1.0) < 0.1
    assert abs(cw["v"] - 1.0 / TAU) < 0.05
    assert abs(cw["1"] - A / TAU) < 0.05
    assert abs(cw["w"] - (-B / TAU)) < 0.05

def test_cubic_is_negative(model):
    """Boundedness is inherited from the negative cubic -v^3/3."""
    m, _ = model
    assert m.cubic_coeff() < 0

def test_long_horizon_bounded_and_noncollapsing(model):
    """100-period free rollout: firing current sustains a nonzero cycle, quiescent current decays -- neither diverges nor spuriously collapses/oscillates."""
    m, dt = model
    fire = m.simulate(np.array([-1.0, -0.5]), np.full(100 * 740, 0.7), dt)[0, :, 0]
    quiet = m.simulate(np.array([-1.0, -0.5]), np.full(100 * 740, 0.0), dt)[0, :, 0]
    assert np.isfinite(fire).all() and np.isfinite(quiet).all()
    tail_fire = fire[len(fire) // 2:]
    assert np.ptp(tail_fire) > 2.0
    assert np.max(np.abs(fire)) < 10.0
    assert np.ptp(quiet[len(quiet) // 2:]) < 0.3

def test_closed_form_inversion(model):
    """Recovered current reproduces a target v-derivative to high accuracy."""
    m, dt = model
    data = generate(n_train=2, n_val=4, t_max=80.0, dt=dt)
    y, u = np.array(data["ys_val"][0]), np.array(data["u_val"][0])
    vdot = S.finite_difference(y, dt)[:, 0]
    u_rec = m.invert_onestep(y, vdot)
    assert float(np.mean(np.abs(u_rec - u))) < 0.05
