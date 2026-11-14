"""Tests for AF4 / S1 (no MCEM; uses the AF2 states when present)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af4_laplace_model_comparison as a4  # noqa: E402

HAVE_STATES = (a4.AF2_STATE / "rep01_dense_oracle.npz").exists()


def _case(method="oracle", regime="sparse"):
    ds = a4.af2.generate(0)
    masks = a4.af2.m1.build_masks(0)
    fr = {r["fit_key"]: r for r in a4.read_csv(a4.AF2_FULL / "fit_runs.csv")}
    st = a4.load_state(0, regime, method)
    model = a4.build_model(fr[f"rep01_{regime}_{method}"]["assignment"].split("|"), masks[regime], st)
    return ds, masks, st, model


@pytest.mark.skipif(not HAVE_STATES, reason="AF2 states not available")
def test_independent_phi_equals_model_terms_plus_prior():
    ds, masks, st, model = _case()
    X, Y, Z = np.asarray(ds.X, float), np.asarray(ds.Y, float), np.asarray(st["Zmode"], float)
    parts = a4.log_joint_parts(model, X, Y, Z)
    lx = model.calc_log_likelihood_X(X, Z[:, :, None], model.params["F"])
    ly = model.calc_log_likelihood_Y(Y, Z[:, :, None], model.params["w0"], model.params["w"])
    prior = -0.5 * a4.N * a4.K * math.log(2 * math.pi) - 0.5 * float(np.sum(Z ** 2))
    assert abs(parts["phi"] - (lx + ly + prior)) < 1e-8


@pytest.mark.skipif(not HAVE_STATES, reason="AF2 states not available")
def test_cmp_column_uses_its_own_base_measure():
    from scipy.special import gammaln
    ds, masks, st, model = _case()
    X = np.asarray(ds.X, float)
    spec = model.column_specs[6]
    assert spec.candidate_model == "cmp_nu2p0"
    eta = np.zeros(a4.N)
    lp = spec.log_prob(X[:, 6], eta)
    lz = spec.log_partition(eta)
    assert np.allclose(lp, -2.0 * gammaln(X[:, 6] + 1) - lz)


@pytest.mark.skipif(not HAVE_STATES, reason="AF2 states not available")
def test_saved_mode_is_stationary_and_test_pairs_do_not_enter():
    import laplace_k_criterion as lk
    ds, masks, st, model = _case()
    X, Y, Z = np.asarray(ds.X, float), np.asarray(ds.Y, float), np.asarray(st["Zmode"], float)
    assert np.max(np.abs(lk.gradient(model, X, Y, Z))) <= a4.MODE_GRAD_TOL
    up = np.triu(masks["test"], 1)
    Yf = Y.copy()
    Yf[up] = 1 - Yf[up]
    Yf = np.triu(Yf, 1) + np.triu(Yf, 1).T
    assert a4.phi_true(model, X, Yf, Z) == a4.phi_true(model, X, Y, Z)


@pytest.mark.skipif(not HAVE_STATES, reason="AF2 states not available")
def test_laplace_matches_existing_evaluator_up_to_the_known_constant():
    import laplace_k_criterion as lk
    from scipy.special import gammaln
    ds, masks, st, model = _case()
    X, Y = np.asarray(ds.X, float), np.asarray(ds.Y, float)
    mine = a4.laplace_at(model, X, Y, np.asarray(st["Zmode"], float))
    ref = lk.laplace_log_observed(model, X, Y, np.asarray(st["Zmode"], float), grad_tol=a4.MODE_GRAD_TOL)
    assert mine["status"] == "OK" and ref.ok
    offset = float(np.sum(gammaln(X[:, 3:] + 1)))
    assert abs(mine["laplace"] - (ref.log_observed + offset)) < 1e-7
    assert abs(mine["logdet"] - mine["logdet_chol"]) < 1e-8


def test_protocol_stable_and_zero_mcem():
    p = a4.protocol()
    assert p["mcem_fits"] == 0 and a4.protocol_sha256() == a4.protocol_sha256()
    assert p["budget"]["cases"] == 200
