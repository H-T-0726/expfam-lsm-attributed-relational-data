"""Focused tests for the Issue #141 E1 runner.  No EM is executed."""

from __future__ import annotations

import numpy as np
import pytest

import run_misspecification_m1 as m1
import run_z_point_estimator_e1 as e1


def test_plan_is_m1_twice_with_prespecified_seeds():
    plan = e1.planned_fits()
    assert len(plan) == 160
    a = [p for p in plan if p["part"] == "A"]
    b = [p for p in plan if p["part"] == "B"]
    assert len(a) == len(b) == 80
    m1_plan = m1.planned_fits()
    # Part A is exactly M1's plan, model seeds included (replay)
    assert [(p["replicate"], p["regime"], p["arm"], p["model_seed"]) for p in a] == \
        [(p["replicate"], p["regime"], p["arm"], p["model_seed"]) for p in m1_plan]
    # Part B differs only in the pre-specified model seed 1344000 + r
    for pa, pb in zip(a, b):
        assert (pa["replicate"], pa["regime"], pa["arm"], pa["data_seed"]) == \
            (pb["replicate"], pb["regime"], pb["arm"], pb["data_seed"])
        assert pb["model_seed"] == 1344000 + pa["r"]
    assert len({e1.fit_key(p) for p in plan}) == 160


def test_frozen_m1_settings_are_reused_not_redefined():
    assert (m1.N, m1.D, m1.K_FIT, m1.L, m1.NUM_ITER) == (75, 12, 3, 5, 8)
    assert e1.PARTS["A"] == m1.MODEL_SEED_BASE == 1343000
    assert e1.ESTIMATORS == ("last", "mean5", "mode")
    assert (e1.MODE_GRAD_TOL, e1.MODE_MAX_ITER) == (1e-8, 200)


def test_point_estimates():
    rng = np.random.default_rng(0)
    Zs = rng.normal(size=(6, 3, 5))
    Zm = rng.normal(size=(6, 3))
    est = e1.point_estimates(Zs, Zm)
    assert np.array_equal(est["last"], Zs[:, :, -1])
    assert np.allclose(est["mean5"], Zs.mean(axis=2))
    assert np.array_equal(est["mode"], Zm)


def test_score_estimate_is_rotation_invariant():
    rng = np.random.default_rng(1)
    n = 30
    Z = rng.normal(size=(n, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Y = (rng.random((n, n)) < 0.3).astype(float)
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    test = np.zeros((n, n), dtype=bool)
    test[0, 1:] = True
    test |= test.T
    a = e1.score_estimate(Z, Z, Y, test, -1.0, 1.0)
    b = e1.score_estimate(Z @ Q, Z, Y, test, -1.0, 1.0)
    assert a["rmse_Z"] == pytest.approx(0.0, abs=1e-10)
    assert b["rmse_Z"] == pytest.approx(0.0, abs=1e-10)
    assert a["gram_corr"] == pytest.approx(1.0)
    assert a["heldout_auc"] == pytest.approx(b["heldout_auc"])


def _runs_and_rows(mode_status_wrong="OK"):
    runs, rows = [], []
    for part in ("A", "B"):
        for r in range(20):
            rep = m1.replicate_label(r)
            for regime in ("dense", "sparse"):
                for arm in ("correct", "wrong"):
                    st = mode_status_wrong if (arm == "wrong" and r == 0) else "OK"
                    runs.append({"part": part, "replicate": rep, "regime": regime,
                                 "arm": arm, "status": "OK", "mode_status": st,
                                 "modes_max_abs_diff": 0.0})
                    for est in e1.ESTIMATORS:
                        base = 0.5 if regime == "dense" else 0.9
                        rows.append({"part": part, "replicate": rep, "regime": regime,
                                     "arm": arm, "estimator": est,
                                     "rmse_Z": base + (0.02 if arm == "wrong" else 0.0),
                                     "gram_corr": 0.8, "heldout_mean_log_score": -0.6,
                                     "heldout_auc": 0.7})
    return runs, rows


def test_non_ok_mode_is_excluded_from_complete_pairs_not_imputed():
    runs, rows = _runs_and_rows("NOT_STATIONARY")
    pairs = e1.paired(rows, runs)
    mode_a = [p for p in pairs if p["part"] == "A" and p["estimator"] == "mode"]
    last_a = [p for p in pairs if p["part"] == "A" and p["estimator"] == "last"]
    assert sum(p["complete_dense"] for p in mode_a) == 19
    assert sum(p["complete_dense"] for p in last_a) == 20
    rep01 = next(p for p in mode_a if p["replicate"] == "rep01")
    assert rep01["delta_rmse_Z_dense"] is None and rep01["interaction_rmse_Z"] is None
    s = e1.summarize(runs, rows, pairs)
    assert s["deltas"]["A_mode"]["dense"]["complete_pairs"] == 19
    assert s["mode_status_counts"]["A_dense_wrong"] == {"NOT_STATIONARY": 1, "OK": 19}
    assert s["deltas"]["A_last"]["dense"]["rmse_Z"]["mean"] == pytest.approx(0.02)


def test_outputs_refuse_to_overwrite(tmp_path):
    (tmp_path / "full").mkdir()
    with pytest.raises(SystemExit):
        e1.main(["--execute", "--out-dir", str(tmp_path), "--state-dir",
                 str(tmp_path / "states")])
