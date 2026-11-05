"""Unit tests for run_mcem_stability (no MCEM fit)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mcem_stability as ms  # noqa: E402


def test_rq2_data_shared_structure_and_spectrum():
    b = ms.rq2_data(1)
    assert np.array_equal(b[3]["Z"], b[5]["Z"][:, :3]) and np.array_equal(b[3]["F"], b[5]["F"][:, :3])
    for k in (3, 5):
        ev = np.linalg.svd(b[k]["F"], compute_uv=False) ** 2
        assert np.allclose(ev, 2.5) and len(ev) == k
    assert np.array_equal(b[5]["Y"], b[5]["Y"].T)


def test_rq2_data_deterministic():
    a, b = ms.rq2_data(2), ms.rq2_data(2)
    for k in (3, 5):
        assert np.array_equal(a[k]["X"], b[k]["X"]) and np.array_equal(a[k]["Y"], b[k]["Y"])


def test_density_calibration_per_k():
    t = ms.ss.population_density(ms.ss.W0_A, ms.ss.W_A, k=5)
    for k in (3, 5):
        assert abs(ms.ss.population_density(ms.calibrate_w0_k(1.0, k), 1.0, k=k) - t) < 1e-10


def test_budget_within_cap():
    assert ms.rq1_fit_iters() == 8192 and ms.rq2_fit_iters() == 3072
    assert ms.rq1_fit_iters() + ms.rq2_fit_iters() + ms.GATE_FIT_ITERS <= ms.BUDGET_FIT_ITERS


def test_prereg_hash_stable():
    assert ms.prereg_hash() == ms.prereg_hash()
