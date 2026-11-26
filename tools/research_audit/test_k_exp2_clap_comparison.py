"""Tests for run_k_exp2_clap_comparison.py on FAKE data only (no main-data score)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_k_exp2_clap_comparison as rc  # noqa: E402


def fake_rows(source="main10", series="A", regime="b", k_true=9, rep=0, q_min=5,
              lap_min=9, bad=(), ties=False):
    rows = []
    ds = f"{series}-{regime}-KT{k_true}-r{rep}"
    for k in range(1, 11):
        for t in (0, 1):
            p = k * 15 - k * (k - 1) // 2 + (15 if series in "OB" else 0)
            cq = 1000.0 + 10.0 * (k - q_min) ** 2 + t
            cl = 2000.0 + 7.0 * (k - lap_min) ** 2 - t
            if ties and k in (lap_min, lap_min + 1):
                cl = 2000.0
            st = "NOT_STATIONARY" if (k, t) in bad else "OK"
            rows.append({
                "source": source, "checkpoint": 10 if source == "main10" else 40,
                "fit_id": f"{ds}-k{k:02d}-t{t}", "dataset_id": ds, "series": series,
                "regime": regime, "k_true": k_true, "rep": rep, "k": k, "t": t,
                "C_Q": repr(cq), "num_params": p, "lap_status": st,
                "C_Lap": "" if st != "OK" else repr(cl),
                "lap_fit_mode": repr(cl - 3.0 * k - 2.0 * k - 1.0 * k) if st == "OK" else "",
                "lap_zhat_sq": repr(3.0 * k), "lap_logdet_H": repr(2.0 * k),
                "lap_penalty": repr(1.0 * k), "lap_min_eig_H": "1.0", "lap_grad_inf": "1e-10",
                "lap_iterations": 5, "z_shift_rms": "0.1", "lap_seconds": 0.1,
                "lap_closure_absdiff": "0.0"})
    return rows


def test_pick_tie_smallest():
    assert rc.pick({1: 5.0, 2: 5.0 + 5e-10, 3: 6.0}) == 1
    assert rc.pick({1: 5.0, 2: 4.0}) == 2


def test_selection_both_criteria_and_change():
    res = rc.analyze_rows(fake_rows(q_min=5, lap_min=9))
    s = res["selection"][0]
    assert s["k_Q"] == 5 and s["outcome_Q"] == "under"
    assert s["k_Lap"] == 9 and s["outcome_Lap"] == "exact"
    assert s["changed"] is True and s["direction"] == "Lap>Q"
    sm = res["summary"][0]
    assert (sm["Q_exact"], sm["Q_under"], sm["Lap_exact"], sm["n_changed"]) == (0, 1, 1, 1)


def test_incomplete_never_imputed_denominator_kept():
    rows = fake_rows(rep=0, bad={(9, 1)}) + fake_rows(rep=1)
    res = rc.analyze_rows(rows)
    sel = {s["dataset_id"]: s for s in res["selection"]}
    s0 = sel["A-b-KT9-r0"]
    assert s0["lap_status"] == "INCOMPLETE" and s0["k_Lap"] == ""
    assert s0["k_Lap_init0"] == 9 and s0["k_Lap_init1"] == ""   # per-init: only complete init
    assert s0["k_Q"] == 5                                         # C_Q always reported
    sm = res["summary"][0]
    assert sm["n_datasets"] == 2 and sm["Lap_incomplete"] == 1 and sm["Lap_exact"] == 1
    assert sm["n_complete"] == 1


def test_tie_rule_applied_to_c_lap():
    res = rc.analyze_rows(fake_rows(lap_min=7, ties=True))
    assert res["selection"][0]["k_Lap"] == 7


def test_reference_mismatch_stops():
    rows = fake_rows()
    with pytest.raises(rc.Stop):
        rc.analyze_rows(rows, {("main10", "A-b-KT9-r0"): 6})
    res = rc.analyze_rows(rows, {("main10", "A-b-KT9-r0"): 5})
    assert res["checks"]["C_Q_selection_equals_recorded"] is True


def test_missing_rows_stop():
    rows = [r for r in fake_rows() if not (r["k"] == 3 and r["t"] == 0)]
    with pytest.raises(rc.Stop):
        rc.analyze_rows(rows)


def test_q_terms_split_exact():
    r = {"C_Q": "12345.678", "num_params": 135, "k": 9}
    q = rc.q_terms(r)
    assert q["Q_P_Z"] == pytest.approx(9 * 150 * (1 + math.log(2 * math.pi)))
    assert q["Q_P_Z"] + q["Q_PEN"] + q["Q_D"] == pytest.approx(12345.678, abs=1e-9)


def test_steps_and_focus_terms():
    res = rc.analyze_rows(fake_rows(q_min=5, lap_min=9))
    st = {(s["k_from"]): s for s in res["steps"]}
    assert st[1]["median_d_Q_P_Z"] == pytest.approx(rc.PZ_PER_DIM)
    assert st[1]["median_d_L_zsq"] == pytest.approx(3.0)
    f = res["focus"][0]
    assert f["k_Q"] == 5 and f["d_Q_P_Z_ktrue_minus_kQ"] == pytest.approx(4 * rc.PZ_PER_DIM)
    assert f["d_L_logdet_ktrue_minus_kQ"] == pytest.approx(8.0)


def test_k3_identity_check():
    rows = fake_rows(regime="a", k_true=3) + fake_rows(regime="b", k_true=3)
    assert rc.analyze_rows(rows)["checks"]["K_TRUE3_a_b_identical_C_Lap"] is True
    rows[-1]["C_Lap"] = "1.0"
    assert rc.analyze_rows(rows)["checks"]["K_TRUE3_a_b_identical_C_Lap"] is False


def test_figures_written(tmp_path):
    res = rc.analyze_rows(fake_rows() + fake_rows(series="O", lap_min=9, q_min=9))
    rc.figures(res, tmp_path)
    assert (tmp_path / "fig_exact_by_criterion.png").exists()
    assert (tmp_path / "fig_criterion_curves_focus.png").exists()
    assert (tmp_path / "fig_term_steps_focus.png").exists()


def test_evaluate_state_records_failures_without_retry():
    """Tiny synthetic model (not experiment data): NOT_STATIONARY and ERROR are recorded."""
    from em_runner import build_model
    rng = np.random.default_rng(0)
    n, d, k = 12, 4, 2
    m = build_model(n, d, k, 2, "gaussian", "bernoulli", numerics_mode="consistent")
    m.initialize_params(true_params=None, seed=0)
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((d, k))
    X = Z @ F.T + 0.3 * rng.standard_normal((n, d))
    Y = (rng.random((n, n)) < 0.3).astype(float)
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    m.params.update({"F": F, "sigma": np.eye(d) * 0.1, "w0": -1.0, "w": 0.5, "var_z": 1.0,
                     "Z": Z})
    old = (rc.kd.N, rc.kd.D)
    rc.kd.D = d
    try:
        ns = rc.evaluate_state(m, X, Y, Z, k, max_iter=0)
        assert ns["lap_status"] == "NOT_STATIONARY" and ns["C_Lap"] == ""
        ok = rc.evaluate_state(m, X, Y, Z, k)
        assert ok["lap_status"] == "OK" and float(ok["lap_closure_absdiff"]) < 1e-6
        m.params["F"] = F * np.nan
        er = rc.evaluate_state(m, X, Y, Z, k)
        assert er["lap_status"] != "OK" and er["C_Lap"] == ""
    finally:
        rc.kd.N, rc.kd.D = old
