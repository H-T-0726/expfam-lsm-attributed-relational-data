"""Tests for the EM convergence-sensitivity runner and its independent auditor.

Nothing here runs production, and nothing here runs EM in this process.  The
three properties that need real EM -- that the instrumented runner reproduces
``em_runner`` at the baseline budget, that checkpoint 8 of a long run equals a
short run, and that checkpoint evaluation perturbs nothing -- are measured by
``_em_budget_probe.py`` in a CHILD process and asserted in
``test_em_budget_nesting.py``.  That indirection exists because the Phase 8b
suite asserts ``"em_runner" not in sys.modules`` for the whole research_audit
package; importing an EM runner here would break 64 unrelated tests.

The auditor tests work on a synthetic artifact whose checkpoint-8 rows are the
REAL archived rows, so the replication check has something true to verify and
the mutation tests have something true to break.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_em_convergence_sensitivity as AUDIT              # noqa: E402
import run_em_convergence_sensitivity as RUN                  # noqa: E402

ARCHIVE = (ROOT / "expfam" / "results" / "k_selection"
           / "clean_true_k_asymptotics_20260904")
FIXTURE_N = 75
FIXTURE_REPLICATE = 1


# --------------------------------------------------------------- no EM here

def test_no_em_runner_enters_this_process():
    """The Phase 8b tripwire, restated where it is easiest to break."""

    assert "em_runner" not in sys.modules
    assert "em_runner_instrumented" not in sys.modules


def test_the_runner_imports_em_lazily():
    """Module scope must stay EM-free, or importing this file breaks Phase 8b."""

    source = Path(RUN.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.startswith(("import ", "from ")) and "em_runner" in line:
            pytest.fail(f"module-scope EM import: {line!r}")
    assert "from em_runner_instrumented import" in source     # lazy, indented


def test_the_auditor_does_not_import_the_runner():
    """Independence: an auditor that reuses the runner cannot contradict it."""

    source = Path(AUDIT.__file__).read_text(encoding="utf-8")
    for name in ("run_em_convergence_sensitivity", "em_runner",
                 "run_clean_true_k_sweep"):
        assert f"import {name}" not in source, name


# ------------------------------------------------------------ frozen design

def test_the_protocol_forbids_adaptive_stopping():
    body = RUN.frozen_protocol()
    assert body["adaptive_stopping"] is False
    assert body["stopping_reason_expected"] == "fixed_budget"
    assert body["on_nan"] == "fail"


def test_the_ladder_brackets_the_baseline_and_the_primary_budget():
    body = RUN.frozen_protocol()
    assert body["baseline_checkpoint"] == 8
    assert body["primary_checkpoint"] == body["max_iter"] == 40
    ladder = body["checkpoints"]
    assert ladder == sorted(set(ladder))
    assert ladder[0] == body["baseline_checkpoint"]
    assert ladder[-1] == body["max_iter"]


def test_inherited_settings_are_read_from_the_baseline_not_retyped():
    body = RUN.frozen_protocol()
    base = json.loads((ARCHIVE / "protocol.json").read_text(encoding="utf-8"))
    assert body["baseline_protocol_hash"] == base["protocol_hash"]
    for field in ("candidate_k", "starts", "family_x", "family_y", "d", "L",
                  "test_ratio", "tie_tolerance", "data_seed_base",
                  "split_seed_base", "model_seed_base", "numerics_mode",
                  "generator_version"):
        assert body[field] == base["protocol"][field], field


def test_the_gate_cell_is_chosen_by_rule_not_by_inspection():
    assert RUN.GATE_N == min(RUN.N_GRID)
    assert RUN.GATE_REPLICATE == 1


def test_the_protocol_hash_moves_when_the_budget_moves(monkeypatch):
    before = RUN.protocol_hash()
    monkeypatch.setattr(RUN, "MAX_ITER", 41)
    assert RUN.protocol_hash() != before


def test_expected_counts():
    assert RUN.expected_fit_count() == 3 * 8 * 7 * 2 == 336
    assert RUN.expected_checkpoint_rows() == 336 * 6 == 2016


def test_plan_runs_no_em_and_projects_from_the_archive():
    report = RUN.plan()
    assert report["fits"] == 336 and report["checkpoint_rows"] == 2016
    assert report["projected_hours_at_max_iter"] > 0
    assert "em_runner" not in sys.modules


def test_plan_can_price_an_alternative_n_grid():
    three = RUN.plan((75, 100, 150))["projected_hours_at_max_iter"]
    four = RUN.plan((50, 75, 100, 150))["projected_hours_at_max_iter"]
    assert four > three
    assert RUN.plan((50,))["fits"] == 112


# ------------------------------------------------------- production refusals

def test_production_refuses_without_the_flags():
    with pytest.raises(RUN.HarnessStop):
        RUN.run_production(allow_em=False, confirm=True)
    with pytest.raises(RUN.HarnessStop):
        RUN.run_production(allow_em=True, confirm=False)


def test_production_is_fail_closed_on_an_existing_artifact_dir(monkeypatch,
                                                              tmp_path):
    existing = tmp_path / "already_there"
    existing.mkdir()
    monkeypatch.setattr(RUN, "ARTIFACT_DIR", existing)
    with pytest.raises(RUN.HarnessStop) as excinfo:
        RUN.run_production(allow_em=True, confirm=True)
    assert "already exists" in str(excinfo.value)
    assert not any(existing.iterdir())


def test_the_gate_refuses_without_allow_em():
    with pytest.raises(RUN.HarnessStop):
        RUN.run_replication_gate(allow_em=False)


# ------------------------------------------------------------------ fixture

def _archive_rows() -> dict[tuple[int, int], dict[str, str]]:
    with (ARCHIVE / "fit_results.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(int(r["k_est"]), int(r["start"])): r for r in rows
            if int(r["k_true"]) == 5 and int(r["n"]) == FIXTURE_N
            and int(r["replicate"]) == FIXTURE_REPLICATE}


def _build_fixture(run_dir: Path) -> None:
    """A genuinely valid artifact for one cell.

    Checkpoint 8 carries the archived values verbatim, so the replication check
    passes for the right reason.  Later checkpoints are synthesised, with s2
    recomputed from the identity so that the auditor's algebra holds.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    body = dict(RUN.frozen_protocol())
    body["n_grid"] = [FIXTURE_N]
    body["replicates"] = FIXTURE_REPLICATE
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (run_dir / "protocol.json").write_text(json.dumps({
        "protocol": body, "protocol_hash": digest,
        "protocol_version": RUN.PROTOCOL_VERSION,
        "artifact_version": RUN.ARTIFACT_VERSION,
        "experiment_id": RUN.EXPERIMENT_ID,
    }, indent=2, sort_keys=True), encoding="utf-8")

    train, test = AUDIT.build_masks(
        FIXTURE_N,
        AUDIT.split_seed(body["split_seed_base"], 5, FIXTURE_N, FIXTURE_REPLICATE),
        body["test_ratio"])
    train_hash, test_hash = AUDIT.stable_hash(train), AUDIT.stable_hash(test)

    archive = _archive_rows()
    ladder = body["checkpoints"]
    checkpoint_rows: list[dict] = []
    fit_rows: list[dict] = []
    index = 0
    for k_est in body["candidate_k"]:
        for start in body["starts"]:
            index += 1
            reference = archive[(k_est, start)]
            identity = {
                "fit_index": index, "k_true": 5, "n": FIXTURE_N,
                "replicate": FIXTURE_REPLICATE, "k_est": k_est, "start": start,
                "data_seed": int(reference["data_seed"]),
                "split_seed": int(reference["split_seed"]),
                "model_seed": int(reference["model_seed"]),
            }
            npar = int(reference["num_params"])
            for position, iteration in enumerate(ladder):
                if iteration == body["baseline_checkpoint"]:
                    q = float(reference["q_strict"])
                    s1 = float(reference["heldout_mean_log_score"])
                    s3 = float(reference["s3_plugin_conditional"])
                else:
                    q = float(reference["q_strict"]) + 3.0 * position
                    s1 = float(reference["heldout_mean_log_score"]) + 1e-3 * position
                    s3 = float(reference["s3_plugin_conditional"]) - 2.0 * position
                row = dict(identity)
                row.update({
                    "checkpoint_iteration": iteration,
                    "heldout_mean_log_score": s1,
                    "q_strict": q,
                    "s2_q_based": -2.0 * q + npar * math.log(FIXTURE_N),
                    "s3_plugin_conditional": s3,
                    "num_params": npar,
                    "w0_hat": -1.0, "w_hat": 0.5,
                    "cumulative_runtime_s": 1.0 * (position + 1),
                    "nan_iterations_so_far": 0, "attempt_index": 0,
                    "checkpoint_state_digest": hashlib.sha256(
                        f"{index}:{iteration}".encode()).hexdigest(),
                })
                for key in ("gram_rel_change", "f_raw_change", "w_change",
                            "w0_change"):
                    row[key] = 0.01
                    row[f"max_{key}_last4"] = 0.02
                for tau in RUN.TAU_LADDER:
                    row[f"budget_stable_gram_{tau:g}"] = tau > 0.02
                checkpoint_rows.append(row)
            fit_rows.append({
                **identity, "max_iter": body["max_iter"],
                "actual_iterations": body["max_iter"],
                "stopping_reason": "fixed_budget",
                "retry_count": 0, "retry_seed": identity["model_seed"],
                "nan_detected": False, "attempts_used": 1,
                "retry_events": "[]", "runtime_s": 10.0,
                "train_mask_hash": train_hash, "test_mask_hash": test_hash,
            })

    RUN.write_csv(run_dir / "checkpoint_results.csv", checkpoint_rows,
                  RUN.CHECKPOINT_COLUMNS)
    RUN.write_csv(run_dir / "fit_results.csv", fit_rows, RUN.FIT_COLUMNS)
    (run_dir / "replication_vs_baseline.json").write_text(json.dumps({
        "baseline_run": ARCHIVE.name,
        "baseline_protocol_hash": RUN.BASELINE_PROTOCOL_HASH,
        "checkpoint": body["baseline_checkpoint"],
        "compared_rows": len(fit_rows), "mismatches": [], "match": True,
    }, indent=2), encoding="utf-8")
    (run_dir / "runinfo.json").write_text(json.dumps({
        "experiment_id": RUN.EXPERIMENT_ID, "protocol_hash": digest,
        "attempted_fit_count": len(fit_rows),
        "completed_fit_count": len(fit_rows),
        "checkpoint_row_count": len(checkpoint_rows),
        "adaptive_stopping": False,
        "stopping_reasons": ["fixed_budget"],
        "wall_clock_seconds": 10.0 * len(fit_rows) + 5.0,
    }, indent=2), encoding="utf-8")


@pytest.fixture()
def artifact(tmp_path) -> Path:
    run_dir = tmp_path / "em_convergence_sensitivity_20260905"
    _build_fixture(run_dir)
    return run_dir


def _mutate_csv(path: Path, columns: list[str], mutate) -> None:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = mutate(rows)
    RUN.write_csv(path, rows, columns)


# -------------------------------------------------------------- the auditor

def test_the_fixture_passes(artifact):
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "PASS", report["findings"]
    assert report["blocker_count"] == 0 and report["high_count"] == 0
    assert report["recomputed_selections"] == 3 * 6 * 1 * 1


def test_auditor_detects_a_missing_checkpoint(artifact):
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS,
                lambda rows: rows[1:])
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert {f["check"] for f in report["findings"]} & {
        "checkpoint_row_count", "checkpoint_ladder"}


def test_auditor_detects_a_duplicate_checkpoint(artifact):
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS,
                lambda rows: rows[:-1] + [dict(rows[0])])
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "checkpoint_ladder" in {f["check"] for f in report["findings"]}


def test_auditor_detects_an_altered_score(artifact):
    def bump(rows):
        for row in rows:
            if int(row["checkpoint_iteration"]) == 8 and int(row["k_est"]) == 5:
                row["heldout_mean_log_score"] = str(
                    float(row["heldout_mean_log_score"]) + 0.05)
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "replication_vs_baseline" in {f["check"] for f in report["findings"]}


def test_auditor_detects_a_broken_s2_identity(artifact):
    def bump(rows):
        rows[3]["s2_q_based"] = str(float(rows[3]["s2_q_based"]) + 1.0)
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "s2_identity" in {f["check"] for f in report["findings"]}


def test_auditor_detects_retry_metadata_mismatch(artifact):
    def bump(rows):
        rows[0]["retry_count"] = "1"
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    checks = {f["check"] for f in report["findings"]}
    assert "retry_count" in checks and "retry_event_consistency" in checks


def test_auditor_detects_a_hidden_nan(artifact):
    def bump(rows):
        rows[0]["nan_detected"] = "True"
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "nan_detected" in {f["check"] for f in report["findings"]}


def test_auditor_detects_adaptive_stopping(artifact):
    def bump(rows):
        rows[0]["stopping_reason"] = "converged"
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "stopping_reason" in {f["check"] for f in report["findings"]}


def test_auditor_detects_a_short_run(artifact):
    def bump(rows):
        rows[0]["actual_iterations"] = "32"
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "actual_iterations" in {f["check"] for f in report["findings"]}


def test_auditor_detects_a_forged_seed(artifact):
    def bump(rows):
        rows[0]["model_seed"] = str(int(rows[0]["model_seed"]) + 1)
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "seed_formulas" in {f["check"] for f in report["findings"]}


def test_auditor_detects_a_forged_mask_hash(artifact):
    def bump(rows):
        rows[0]["train_mask_hash"] = "0" * 64
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "mask_hashes" in {f["check"] for f in report["findings"]}


def test_auditor_detects_a_tampered_protocol_body(artifact):
    doc = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    doc["protocol"]["max_iter"] = 41
    (artifact / "protocol.json").write_text(json.dumps(doc, indent=2,
                                                       sort_keys=True),
                                            encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "protocol_hash_recomputed" in {f["check"] for f in report["findings"]}


def test_auditor_refuses_a_failed_run(artifact):
    (artifact / "failure.json").write_text("{}", encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "failure_marker" in {f["check"] for f in report["findings"]}


def test_auditor_detects_impossible_runtime_accounting(artifact):
    info = json.loads((artifact / "runinfo.json").read_text(encoding="utf-8"))
    info["wall_clock_seconds"] = 1.0
    (artifact / "runinfo.json").write_text(json.dumps(info, indent=2),
                                           encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "runtime_accounting" in {f["check"] for f in report["findings"]}


def test_auditor_recomputes_selection_from_raw_values(artifact):
    """The selector is restated in the auditor; check it on the fixture."""

    with (artifact / "checkpoint_results.csv").open(encoding="utf-8",
                                                   newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if int(r["checkpoint_iteration"]) == 8]
    means = {}
    for k in RUN.BASE.CANDIDATE_K:
        values = [float(r["heldout_mean_log_score"]) for r in rows
                  if int(r["k_est"]) == k]
        means[k] = float(np.mean(values))
    chosen = AUDIT.select_k(means, 1e-12)
    assert chosen in RUN.BASE.CANDIDATE_K
    assert means[chosen] == max(means.values())


def test_gate_comparison_detects_a_mutated_value():
    """The replication gate must fail on a value that is merely close."""

    archive = _archive_rows()
    reference = archive[(5, 1)]
    good = [{
        "k_true": 5, "n": FIXTURE_N, "replicate": FIXTURE_REPLICATE,
        "k_est": 5, "start": 1, "checkpoint_iteration": 8,
        "data_seed": int(reference["data_seed"]),
        "split_seed": int(reference["split_seed"]),
        "model_seed": int(reference["model_seed"]),
        "num_params": int(reference["num_params"]),
        "heldout_mean_log_score": float(reference["heldout_mean_log_score"]),
        "q_strict": float(reference["q_strict"]),
        "s2_q_based": float(reference["s2_q_based"]),
        "s3_plugin_conditional": float(reference["s3_plugin_conditional"]),
    }]
    assert RUN.compare_to_archive(good)["match"]

    nudged = [dict(good[0])]
    nudged[0]["q_strict"] = math.nextafter(nudged[0]["q_strict"], math.inf)
    result = RUN.compare_to_archive(nudged)
    assert not result["match"]
    assert result["mismatches"][0]["field"] == "q_strict"
