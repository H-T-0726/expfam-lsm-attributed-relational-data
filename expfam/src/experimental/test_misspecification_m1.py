"""Focused tests for the Issue #134 M1 runner.  No EM is executed."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import run_misspecification_m1 as m1


def test_arms_differ_only_in_the_six_bernoulli_columns():
    diff = [l for l in range(m1.D) if m1.ARMS["correct"][l] != m1.ARMS["wrong"][l]]
    assert diff == list(m1.BERNOULLI_COLUMNS) == [3, 4, 5, 6, 7, 8]
    assert m1.ARMS["correct"] == list(m1.FAMILY_X_TRUE)
    assert [m1.ARMS["correct"][l] for l in diff] == ["bernoulli"] * 6
    assert [m1.ARMS["wrong"][l] for l in diff] == ["poisson"] * 6


def test_plan_is_exactly_80_fits_at_k3():
    plan = m1.planned_fits()
    assert len(plan) == m1.EXPECTED_FITS == 80
    assert len({(p["replicate"], p["regime"], p["arm"]) for p in plan}) == 80
    assert m1.K_FIT == m1.K_TRUE == 3
    assert set(m1.REGIMES) == {"dense", "sparse"}
    assert m1.REGIMES == {"dense": 1.0, "sparse": 0.1}
    assert list(m1.DATA_SEEDS) == list(range(1001001, 1001021))
    # one model seed per replicate, shared by its four fits
    for r in range(20):
        seeds = {p["model_seed"] for p in plan if p["r"] == r}
        assert seeds == {m1.MODEL_SEED_BASE + r}


@pytest.mark.parametrize("r", [0, 7, 19])
def test_mask_contract(r):
    masks = m1.build_masks(r)
    test, pool, dense, sparse = masks["test"], masks["pool"], masks["dense"], masks["sparse"]
    n_all = m1.N * (m1.N - 1) // 2
    assert m1.n_upper(test) == round(n_all * 0.10)
    assert m1.n_upper(test) + m1.n_upper(pool) == n_all
    assert np.array_equal(dense, pool)                     # dense = 100% of the pool
    assert np.all(dense[sparse])                           # sparse subset of dense
    assert not np.any(dense & test) and not np.any(sparse & test)
    assert m1.n_upper(sparse) == round(m1.n_upper(pool) * 0.1)
    for m in (test, dense, sparse):
        assert np.array_equal(m, m.T) and not np.any(np.diag(m))
    again = m1.build_masks(r)                              # deterministic -> identical across arms
    for k in masks:
        assert np.array_equal(masks[k], again[k])


def test_data_draw_is_deterministic_and_bernoulli_columns_are_binary():
    a, b = m1.generate(m1.DATA_SEEDS[0]), m1.generate(m1.DATA_SEEDS[0])
    for k in ("Z", "F", "X", "Y"):
        assert np.array_equal(getattr(a, k), getattr(b, k))
    assert np.all(np.isin(a.X[:, list(m1.BERNOULLI_COLUMNS)], (0.0, 1.0)))


def test_run_fit_passes_fixed_family_and_k3(monkeypatch):
    calls = []

    def fake_run(X, Y, **kw):
        calls.append(kw)
        rng = np.random.default_rng(0)
        return {"Z_est": rng.standard_normal((m1.N, 3)), "w0": -1.0, "w": 1.0,
                "retry_count": 0, "replacement_count": 0, "seed_rescue_count": 0,
                "nan_occurred": False, "numerics_mode": "consistent", "runtime_s": 0.0}

    import em_runner
    monkeypatch.setattr(em_runner, "run_em_experimental", fake_run)
    ds = m1.generate(m1.DATA_SEEDS[0])
    masks = m1.build_masks(0)
    for arm in ("correct", "wrong"):
        fit = {"replicate": "rep01", "r": 0, "data_seed": m1.DATA_SEEDS[0],
               "regime": "sparse", "arm": arm, "model_seed": m1.MODEL_SEED_BASE}
        row = m1.run_fit(fit, ds, masks)
        assert row["status"] == "OK"
    assert [c["family_x_list"] for c in calls] == [m1.ARMS["correct"], m1.ARMS["wrong"]]
    for c in calls:
        assert c["k"] == 3
        assert c["numerics_mode"] == "consistent"
        assert c["failure_policy"] == "fail_fast"
        assert c["allow_support_mismatch"] is False
        assert c["train_mask"] is masks["sparse"]
        assert c["seed"] == m1.MODEL_SEED_BASE


def test_failed_fit_is_recorded_not_retried(monkeypatch):
    n_calls = []

    def boom(X, Y, **kw):
        n_calls.append(1)
        raise FloatingPointError("synthetic")

    import em_runner
    monkeypatch.setattr(em_runner, "run_em_experimental", boom)
    fit = {"replicate": "rep01", "r": 0, "data_seed": m1.DATA_SEEDS[0],
           "regime": "dense", "arm": "wrong", "model_seed": m1.MODEL_SEED_BASE}
    row = m1.run_fit(fit, m1.generate(m1.DATA_SEEDS[0]), m1.build_masks(0))
    assert row["status"] == "FAILED" and "FloatingPointError" in row["error"]
    assert len(n_calls) == 1


def test_no_family_or_k_selection_code_is_imported():
    tree = ast.parse(Path(m1.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    forbidden = {"family_selection", "run_family_selection_pilot", "run_joint_family_k_selection",
                 "laplace_k_criterion", "run_lap_vs_cq_20"}
    assert not (names & forbidden)


def test_paired_analysis_signs():
    rows = []
    for r in range(2):
        for regime in m1.REGIMES:
            for arm, z in (("correct", 0.4), ("wrong", 0.5 if regime == "sparse" else 0.4)):
                rows.append({"replicate": m1.replicate_label(r), "regime": regime, "arm": arm,
                             "status": "OK", "rmse_Z": z, "heldout_mean_log_score": -0.5,
                             "heldout_auc": 0.8, "heldout_ap": 0.6, "w0_hat": -1, "w_hat": 1,
                             "abs_w0_err": 0, "abs_w_err": 0, "gram_corr": 0.8,
                             "n_test_pairs": 278, "K_fit": 3, "retry_count": 0,
                             "replacement_count": 0, "seed_rescue_count": 0, "n_warnings": 0,
                             "error": ""})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        s = m1.analyze(rows, Path(d))
    assert s["regimes"]["sparse"]["delta_RMSE_Z"]["mean"] == pytest.approx(0.1)
    assert s["regimes"]["dense"]["delta_RMSE_Z"]["mean"] == pytest.approx(0.0)
    assert s["interaction_delta"]["mean"] == pytest.approx(0.1)
    assert s["regimes"]["sparse"]["wrong_gt_correct"] == 2
