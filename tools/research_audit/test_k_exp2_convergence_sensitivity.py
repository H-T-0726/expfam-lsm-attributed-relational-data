"""Tests for run_k_exp2_convergence_sensitivity.py (no MCEM except the tiny one)."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cs", HERE / "run_k_exp2_convergence_sensitivity.py")
cs = importlib.util.module_from_spec(spec)
sys.modules["cs"] = cs
spec.loader.exec_module(cs)


def test_jobs_copy_main_manifest_seeds_exactly():
    jobs = cs.jobs_for_cells((("B", "b", 9),))
    assert len(jobs) == 200
    man = {m["fit_id"]: m for m in cs.read_csv(cs.MAIN_BASE / "protocol_v2" / "manifest.csv")}
    for j in jobs:
        m = man[j["fit_id"]]
        assert (j["data_seed"], j["init_seed"], j["mc_seed"]) == (
            int(m["data_seed"]), int(m["init_seed"]), int(m["mc_seed"]))
        assert j["num_iter"] == 40


def test_proposed_cells_are_600_fits_and_disjoint():
    jobs = cs.jobs_for_cells(cs.PROPOSED_CELLS)
    assert len(jobs) == 600 == len({j["fit_id"] for j in jobs})
    assert {(j["series"], j["regime"], j["k_true"]) for j in jobs} == set(cs.PROPOSED_CELLS)


def test_smoke_seeds_differ_from_all_main_seeds():
    main = cs.read_csv(cs.MAIN_BASE / "protocol_v2" / "manifest.csv")
    used = {int(m[c]) for m in main for c in ("data_seed", "init_seed", "mc_seed")}
    sm = cs.jobs_for_cells(cs.PROPOSED_CELLS, seed_offset=cs.SMOKE_SEED_OFFSET)
    assert all(j["fit_id"].startswith("SMOKE-") for j in sm)
    assert not ({j[c] for j in sm for c in ("data_seed", "init_seed", "mc_seed")} & used)


def test_checkpoint_cq_uses_frozen_formula():
    hist = [{"iteration": i, "q_after": -1000.0 - i} for i in range(1, 41)]
    out = cs.checkpoint_cq(hist, 9, "gaussian", "poisson", (10, 20, 40))
    assert [c[0] for c in out] == [10, 20, 40]
    p = 9 * 15 - 9 * 8 // 2 + 15
    for c, q, cq, npar in out:
        assert npar == p
        assert cq == -2.0 * q + p * math.log(150)
        assert q == -1000.0 - c


def test_select_tie_rule_smallest_k():
    assert cs.select({1: 5.0, 2: 5.0 + 5e-10, 3: 7.0}) == 1
    assert cs.select({1: 5.0, 2: 4.0, 3: 4.0}) == 2


def _fake(tmp_path, cells_cq, n_hist=0):
    fits, cps, hist = {}, [], []
    for (ds, kt), by_cp in cells_cq.items():
        for k in range(1, 11):
            for t in (0, 1):
                fid = f"{ds}-k{k:02d}-t{t}"
                fits[fid] = {"fit_id": fid, "dataset_id": ds, "series": "B", "regime": "b",
                             "k_true": str(kt), "rep": "0", "k": str(k), "t": str(t),
                             "status": "OK", "data_equal_main": "True",
                             "final_q_equals_last_q_after": "True", "retry_count": "0",
                             "replacement_count": "0", "nan_count": "0",
                             "mstep_q_decreases": "0"}
                for it in range(1, n_hist + 1):
                    hist.append({"fit_id": fid, "iteration": str(it),
                                 "q_before": "0", "q_after": repr(-1000.0 * k + 50.0 * it
                                                                  - (it * 30.0 if k == 9 else 0)),
                                 "q_diff": "0", "decreased": "False"})
                for cp, fn in by_cp.items():
                    cps.append({"fit_id": fid, "dataset_id": ds, "series": "B", "regime": "b",
                                "k_true": str(kt), "rep": "0", "k": str(k), "t": str(t),
                                "checkpoint": str(cp), "C_Q": repr(fn(k, t)),
                                "repro_equal": "True" if cp == 10 else ""})
    if n_hist:
        return fits, cps, hist
    return fits, cps


def test_analysis_outcomes_gap_and_transition():
    f = {10: lambda k, t: abs(k - 2) * 10.0 + t,
         20: lambda k, t: abs(k - 9) * 10.0 + t,
         40: lambda k, t: abs(k - 9) * 10.0 + t}
    fits, cps = _fake(None, {("B-b-KT9-r0", 9): f})
    rows = {r["checkpoint"]: r for r in cs.analyze_rows(cps, fits)}
    assert rows[10]["selected_k"] == 2 and rows[10]["outcome"] == "under"
    assert rows[20]["selected_k"] == 9 and rows[20]["outcome"] == "exact"
    assert rows[10]["gap_ktrue_vs_best_other"] > 0
    assert rows[40]["gap_ktrue_vs_best_other"] == pytest.approx(-10.0)


def test_analysis_incomplete_when_a_fit_failed():
    f = {c: (lambda k, t: float(k)) for c in (10, 20, 40)}
    fits, cps = _fake(None, {("B-b-KT9-r0", 9): f})
    fits["B-b-KT9-r0-k05-t1"]["status"] = "FAILED"
    rows = cs.analyze_rows(cps, fits)
    assert all(r["outcome"] == "INCOMPLETE" for r in rows)


def _write_run(run, fits, cps, hist=None):
    run.mkdir()
    for name, rows in (("per_fit.csv", list(fits.values())), ("checkpoint_scores.csv", cps),
                       ("convergence_history.csv", hist or [])):
        if not rows:
            continue
        with open(run / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def test_analyze_full_outputs_with_history_and_figures(tmp_path):
    f = {c: (lambda k, t: abs(k - 9) * 3.0) for c in (10, 20, 40)}
    fits, cps, hist = _fake(None, {("B-b-KT9-r0", 9): f, ("B-b-KT9-r1", 9): f}, n_hist=40)
    for r in fits.values():
        if r["dataset_id"].endswith("r1"):
            r["rep"] = "1"
    for c in cps:
        if c["dataset_id"].endswith("r1"):
            c["rep"] = "1"
    run = tmp_path / "run"
    _write_run(run, fits, cps, hist + hist[:5])          # orphan duplicates are identical
    out = tmp_path / "ana"
    st = cs.analyze(run, out)
    assert st["repro_gate_checkpoint10_equals_main"] is True and st["n_ok"] == 40
    for n in ("selected_k_by_dataset.csv", "summary_by_cell_checkpoint.csv",
              "transitions.csv", "cq_trajectory_by_dataset.csv",
              "cq_trajectory_median_by_cell_k.csv", "per_k_improvement.csv",
              "per_k_improvement_median.csv", "fig_exact_by_checkpoint.png",
              "fig_gap_trajectories.png", "fig_cq_trajectories.png",
              "fig_init_dependence.png"):
        assert (out / n).exists(), n
    summ = cs.read_csv(out / "summary_by_cell_checkpoint.csv")
    assert [int(r["exact"]) for r in summ] == [2, 2, 2]
    imp = [r for r in cs.read_csv(out / "per_k_improvement.csv") if r["k"] == "1"][0]
    # C_Q = -2 q + p ln n; q improves by 50 per iteration -> C_Q falls by 100 per iteration
    assert float(imp["rate_last3_at10"]) == pytest.approx(100.0)
    assert float(imp["impr_10_to_40"]) == pytest.approx(3000.0)
    k9 = [r for r in cs.read_csv(out / "per_k_improvement.csv") if r["k"] == "9"][0]
    assert float(k9["rate_last3_at40"]) == pytest.approx(40.0)


def test_analyze_stops_on_differing_duplicate_rows(tmp_path):
    f = {c: (lambda k, t: float(k)) for c in (10, 20, 40)}
    fits, cps = _fake(None, {("B-b-KT9-r0", 9): f})
    bad = dict(cps[0], C_Q="123.0")
    run = tmp_path / "run"
    _write_run(run, fits, cps + [bad])
    with pytest.raises(SystemExit, match="duplicate"):
        cs.analyze(run, tmp_path / "ana", figures=False)


def test_clopper_pearson_known_values():
    assert cs.clopper_pearson(10, 10)[0] == pytest.approx(0.6915, abs=1e-4)
    lo, hi = cs.clopper_pearson(5, 10)
    assert (lo, hi) == (pytest.approx(0.1871, abs=1e-4), pytest.approx(0.8129, abs=1e-4))


def test_analyze_writes_outputs(tmp_path):
    f = {c: (lambda k, t: abs(k - 9) * 3.0) for c in (10, 20, 40)}
    fits, cps = _fake(None, {("B-b-KT9-r0", 9): f})
    run = tmp_path / "run"
    _write_run(run, fits, cps)
    out = tmp_path / "ana"
    cs.analyze(run, out)
    st = json.loads((out / "status.json").read_text())
    assert st["repro_gate_checkpoint10_equals_main"] is True
    assert (out / "transitions.csv").exists()
    with pytest.raises(SystemExit):
        cs.analyze(run, out)          # never overwrite


def test_writer_orders_per_fit_last_and_resume_rule_in_protocol(tmp_path):
    wr = cs.Writers(tmp_path, append=False)
    assert list(wr.files) == ["checkpoint_scores.csv", "convergence_history.csv",
                              "per_fit.csv"]
    wr.close()
    assert "never retried" in cs.RESUME_RULE


def test_execute_refuses_on_hash_mismatch(tmp_path):
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps({"code_sha256": {"x": "y"}}))
    with pytest.raises(SystemExit, match="code hash"):
        cs.execute(p, tmp_path / "out", False)


def test_tiny_fit_prefix_identity():
    """K=1 on smoke data: a 3-iteration run is a bitwise prefix of a 5-iteration run."""
    job = cs.jobs_for_cells((("B", "b", 9),), num_iter=5, seed_offset=cs.SMOKE_SEED_OFFSET,
                            ks={1}, ts={0}, reps={0})[0]
    tbl = cs.bern_table()
    r5 = cs.fit_one(job, tbl, None)
    r3 = cs.fit_one({**job, "num_iter": 3}, tbl, None)
    assert r5["status"] == r3["status"] == "OK"
    assert [(h["q_before"], h["q_after"]) for h in r3["_hist"]] == \
        [(h["q_before"], h["q_after"]) for h in r5["_hist"][:3]]
    assert r3["final_q_equals_last_q_after"] is True
