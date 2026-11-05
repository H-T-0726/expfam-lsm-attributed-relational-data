"""Unit tests for run_k_signal_strength (no MCEM fit)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_k_signal_strength as ss  # noqa: E402


def test_population_density_matches_monte_carlo():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((400000, 5)); zp = rng.standard_normal((400000, 5))
    mc = float(np.mean(expit(-1.0 + ss.W_A * np.sum(z * zp, 1))))
    assert abs(ss.population_density(-1.0, ss.W_A) - mc) < 3e-3


def test_population_density_quadrature_converged():
    a = ss.population_density(-1.0, 1.2, m=120)
    b = ss.population_density(-1.0, 1.2, m=200)
    assert abs(a - b) < 1e-7


def test_w0_calibration_equalises_density():
    w0 = ss.w0_c()
    assert abs(ss.population_density(w0, ss.W_C) - ss.population_density(ss.W0_A, ss.W_A)) < 1e-10
    assert w0 < ss.W0_A          # a larger w with symmetric S needs a smaller w0 (density < 1/2)


def test_poisson_coupling_marginal():
    rng = np.random.default_rng(1)
    lam_a = np.full(200000, 1.7); lam_b = np.where(np.arange(200000) % 2 == 0, 3.1, 0.6)
    xa = rng.poisson(lam_a).astype(float)
    xb = ss.couple_poisson(xa, lam_a, lam_b, np.random.default_rng(2))
    for v in (3.1, 0.6):
        sel = lam_b == v
        assert abs(xb[sel].mean() - v) < 0.02
        assert abs(xb[sel].var() - v) < 0.05
    assert np.all(xb[lam_b == 3.1] >= xa[lam_b == 3.1])
    assert np.all(xb[lam_b == 0.6] <= xa[lam_b == 0.6])


def test_poisson_coupling_identity_when_rates_equal():
    rng = np.random.default_rng(3)
    lam = rng.uniform(0.2, 4, size=(30, 15)); xa = rng.poisson(lam).astype(float)
    assert np.array_equal(ss.couple_poisson(xa, lam, lam, np.random.default_rng(4)), xa)


def test_bernoulli_coupling_marginal_and_symmetry():
    n = 700
    rng = np.random.default_rng(5)
    up = np.triu(np.ones((n, n), dtype=bool), 1)
    pa = np.full((n, n), 0.3)
    pc = np.where((np.arange(n)[:, None] + np.arange(n)[None, :]) % 2 == 0, 0.55, 0.1)
    ya = np.zeros((n, n)); ya[up] = rng.random(up.sum()) < 0.3; ya = ya + ya.T
    yc = ss.couple_bernoulli_upper(ya, pa, pc, np.random.default_rng(6))
    assert np.array_equal(yc, yc.T) and np.all(np.diag(yc) == 0)
    for v in (0.55, 0.1):
        sel = up & (pc == v)
        assert abs(yc[sel].mean() - v) < 0.01


def test_prereg_hash_stable_and_cap():
    assert ss.prereg_hash() == ss.prereg_hash()
    planned = ss.expected_new_fits(75) + ss.expected_new_fits(150)
    assert planned == 448
    assert planned + 1 + len(ss.PILOT_FITS) <= ss.FIT_CAP


def test_scale_factors():
    assert math.isclose(ss.C_F ** 2 * ss.ROW_NORM_A, 0.9)
    assert math.isclose(ss.W_C ** 2 / ss.W_A ** 2, 1.8)


def test_select_marks_failed_dataset_incomplete():
    rows = []
    for k in range(1, 8):
        for s in (1, 2):
            rows.append({"condition": "B", "n": 75, "replicate": 1, "k_est": k, "start": s,
                         "fit_status": "FAILED" if (k, s) == (3, 2) else "FIT",
                         "lap_status": "OK", "C_Q": float((k - 4) ** 2), "C_Lap": float((k - 5) ** 2)})
    out = ss.select(rows, ("B",))
    assert {r["status"] for r in out} == {"INCOMPLETE"}


def test_select_argmin_mean_over_starts():
    rows = []
    for k in range(1, 8):
        for s in (1, 2):
            rows.append({"condition": "B", "n": 75, "replicate": 1, "k_est": k, "start": s,
                         "fit_status": "FIT", "lap_status": "OK",
                         "C_Q": float((k - 4) ** 2) + s, "C_Lap": float((k - 5) ** 2)})
    out = {r["criterion"]: r for r in ss.select(rows, ("B",))}
    assert out["C_Q"]["selected_k"] == 4 and out["C_Lap"]["selected_k"] == 5
    assert out["C_Q"]["slack_45"] == pytest.approx(-1.0)
