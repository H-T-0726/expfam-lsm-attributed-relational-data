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


def _fake(tmp_path, cells_cq):
    fits, cps = {}, []
    for (ds, kt), by_cp in cells_cq.items():
        for k in range(1, 11):
            for t in (0, 1):
                fid = f"{ds}-k{k:02d}-t{t}"
                fits[fid] = {"fit_id": fid, "dataset_id": ds, "series": "B", "regime": "b",
                             "k_true": str(kt), "rep": "0", "status": "OK"}
                for cp, fn in by_cp.items():
                    cps.append({"fit_id": fid, "dataset_id": ds, "series": "B", "regime": "b",
                                "k_true": str(kt), "rep": "0", "k": str(k), "t": str(t),
                                "checkpoint": str(cp), "C_Q": repr(fn(k, t)),
                                "repro_equal": "True" if cp == 10 else ""})
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


def test_analyze_writes_outputs(tmp_path):
    f = {c: (lambda k, t: abs(k - 9) * 3.0) for c in (10, 20, 40)}
    fits, cps = _fake(None, {("B-b-KT9-r0", 9): f})
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "per_fit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(fits.values()))))
        w.writeheader()
        w.writerows(fits.values())
    with open(run / "checkpoints.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cps[0]))
        w.writeheader()
        w.writerows(cps)
    out = tmp_path / "ana"
    cs.analyze(run, out)
    st = json.loads((out / "status.json").read_text())
    assert st["repro_gate_checkpoint10_equals_main"] is True
    assert (out / "transitions.csv").exists()
    with pytest.raises(SystemExit):
        cs.analyze(run, out)          # never overwrite


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
