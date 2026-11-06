"""Unit tests for run_mcem_noise_separation (no MCEM fits)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mcem_noise_separation as ns  # noqa: E402


def test_rel_zero_and_scale():
    a = np.ones((3, 3))
    assert ns.rel(a, a) == 0.0
    assert abs(ns.rel(2 * a, a) - 1.0) < 1e-9


def test_num_params():
    assert ns.num_params(5) == 5 * 15 - 10
    assert ns.num_params(4) == 4 * 15 - 6


def test_eta_y_mean_over_draws_matches_loop():
    rng = np.random.default_rng(0)
    Zs = rng.normal(size=(6, 2, 3))
    rows, cols = np.where(np.triu(np.ones((6, 6), bool), 1))
    got = ns.eta_y_mean_over_draws(Zs, -1.0, 0.5, rows, cols)
    want = np.mean([-1.0 + 0.5 * np.einsum("ij,ij->i", Zs[rows, :, l], Zs[cols, :, l]) for l in range(3)], axis=0)
    assert np.allclose(got, want)


def test_posterior_mean_eta_x_is_linear_but_eta_y_is_not():
    rng = np.random.default_rng(1)
    Zs = rng.normal(size=(5, 2, 4000))
    F = rng.normal(size=(3, 2))
    zbar = Zs.mean(axis=2)
    assert np.allclose(np.einsum("ikl,dk->idl", Zs, F).mean(axis=2), zbar @ F.T)
    rows, cols = np.array([0]), np.array([0])           # i == j: E[z^T z] != zbar^T zbar
    pm = ns.eta_y_mean_over_draws(Zs, 0.0, 1.0, rows, cols)
    plug = np.einsum("ij,ij->i", zbar[rows], zbar[cols])
    assert pm[0] - plug[0] > 1.5


def test_fit_id_unique():
    ids = {ns.fit_id_of(c, r, k, s) for c in "ABCD" for r in ns.REPS for k in ns.KS for s in ns.STARTS}
    assert len(ids) == 128
