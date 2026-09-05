"""Tests for the EM convergence-sensitivity runner and its independent auditor.

Nothing here runs production and nothing here runs EM in this process.  The
properties that need real EM -- equivalence with ``em_runner`` at the baseline
budget, checkpoint nesting, and invariance of the ACTUAL production callback --
are measured by ``_em_budget_probe.py`` in a CHILD process and asserted in
``test_em_budget_nesting.py``.  That indirection exists because the Phase 8b
suite asserts ``"em_runner" not in sys.modules`` for the whole research_audit
package; importing an EM runner here would break 64 unrelated tests.

The auditor tests work on a synthetic artifact whose baseline-checkpoint rows
are the REAL archived rows, so the replication check has something true to
verify and every mutation test has something true to break.
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
FIXTURE_N = 50            # the gate cell's n, and the smallest scoped n
FIXTURE_REPLICATE = 1


# --------------------------------------------------------------- no EM here

def test_no_em_runner_enters_this_process():
    """The Phase 8b tripwire, restated where it is easiest to break."""

    assert "em_runner" not in sys.modules
    assert "em_runner_instrumented" not in sys.modules


def test_the_runner_imports_em_lazily():
    source = Path(RUN.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.startswith(("import ", "from ")) and "em_runner" in line:
            pytest.fail(f"module-scope EM import: {line!r}")
    assert "from em_runner_instrumented import" in source     # lazy, indented


def test_the_auditor_is_independent_of_the_runner():
    """Directly and indirectly: an auditor that reuses the runner's code, or
    the baseline sweep's, cannot contradict either."""

    source = Path(AUDIT.__file__).read_text(encoding="utf-8")
    for name in ("run_em_convergence_sensitivity", "em_runner",
                 "em_runner_instrumented", "run_clean_true_k_sweep",
                 "run_heldout_k_selection_pilot"):
        assert f"import {name}" not in source, name
    # and it restates the primitives rather than borrowing them
    for symbol in ("def data_seed", "def split_seed", "def model_seed",
                   "def build_masks", "def stable_hash", "def select_k",
                   "def canonical_hash", "def state_digest"):
        assert symbol in source, symbol


# ------------------------------------------------------------ frozen design

def test_scope_is_the_four_point_grid_everywhere():
    body = RUN.protocol_body_from_constants()
    assert RUN.N_GRID == (50, 75, 100, 150)
    assert body["n_grid"] == [50, 75, 100, 150]
    assert RUN.expected_fit_count() == 4 * 8 * 7 * 2 == 448
    assert RUN.expected_checkpoint_rows() == 448 * 6 == 2688
    plan = RUN.plan()
    assert plan["fits"] == 448 and plan["checkpoint_rows"] == 2688
    draft = json.loads(RUN.DRAFT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert draft["expected_long_runs"] == 448
    assert draft["expected_checkpoint_rows"] == 2688
    assert draft["protocol"]["n_grid"] == [50, 75, 100, 150]


def test_the_draft_on_disk_matches_the_runner_constants():
    """Schema drift is a documentation bug that reads as a protocol change."""

    draft = json.loads(RUN.DRAFT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert draft["status"] == "DRAFT"
    assert "protocol_hash" not in draft          # a hash would look like a freeze
    assert draft["protocol"] == RUN.protocol_body_from_constants()
    assert draft["would_freeze_to_hash"] == RUN.canonical_hash(draft["protocol"])
    schema = draft["artifact_schema"]
    assert schema["checkpoint_results.csv"] == RUN.CHECKPOINT_COLUMNS
    assert schema["fit_results.csv"] == RUN.FIT_COLUMNS


def test_the_estimand_is_narrow_and_names_its_forbidden_claims():
    body = RUN.protocol_body_from_constants()
    assert "continuing the same L=5 stochastic MCEM" in body["estimand"]
    for claim in ("pure optimization effect", "the criterion is exonerated",
                  "consistency"):
        assert claim in body["forbidden_claims"], claim
    assert "Monte-Carlo" in body["residual_confound"]


def test_the_protocol_forbids_adaptive_stopping():
    body = RUN.protocol_body_from_constants()
    assert body["adaptive_stopping"] is False
    assert body["stopping_reason_expected"] == "fixed_budget"
    assert body["on_nan"] == "fail"


def test_the_ladder_brackets_the_baseline_and_the_primary_budget():
    body = RUN.protocol_body_from_constants()
    assert body["baseline_checkpoint"] == 8
    assert body["primary_checkpoint"] == body["max_iter"] == 40
    ladder = body["checkpoints"]
    assert ladder == sorted(set(ladder))
    assert ladder[0] == 8 and ladder[-1] == 40


def test_primary_outcomes_are_frozen_with_their_aggregation_order():
    spec = RUN.protocol_body_from_constants()["primary_outcomes"]
    assert "average S1 over the two starts" in spec["P1"]["definition"]
    assert "smallest K" in spec["P1"]["definition"]
    assert "0.5 * sum over start" in spec["P2"]["definition"]
    assert "DESCRIPTIVE ONLY" in spec["P2"]["inference"]
    assert "no p-value" in spec["P2"]["inference"]
    assert "winning margin" in " ".join(spec["secondary"])
    assert "materially" in spec["wording"]        # says NOT to use it


def test_inherited_settings_are_read_from_the_baseline_not_retyped():
    body = RUN.protocol_body_from_constants()
    base = json.loads((ARCHIVE / "protocol.json").read_text(encoding="utf-8"))
    assert body["baseline_protocol_hash"] == base["protocol_hash"]
    for field in ("candidate_k", "starts", "family_x", "family_y", "d", "L",
                  "test_ratio", "tie_tolerance", "data_seed_base",
                  "split_seed_base", "model_seed_base", "numerics_mode",
                  "generator_version"):
        assert body[field] == base["protocol"][field], field


def test_the_gate_cell_is_chosen_by_rule_not_by_inspection():
    assert RUN.GATE_N == min(RUN.N_GRID) == 50
    assert RUN.GATE_REPLICATE == 1
    assert RUN.GATE_EXPECTED_FITS == len(RUN.BASE.CANDIDATE_K) * len(RUN.BASE.STARTS)


def test_the_protocol_hash_moves_when_the_budget_moves(monkeypatch):
    before = RUN.canonical_hash(RUN.protocol_body_from_constants())
    monkeypatch.setattr(RUN, "MAX_ITER", 41)
    assert RUN.canonical_hash(RUN.protocol_body_from_constants()) != before


def test_plan_runs_no_em_and_reports_the_state_archive_size():
    report = RUN.plan()
    assert report["projected_hours_at_max_iter"] > 0
    assert report["archived_state_megabytes_uncompressed"] > 0
    assert "ESTIMATE, not a bound" in report["projection_note"]
    assert "safety stop" in report["projection_note"]
    assert "em_runner" not in sys.modules


def test_plan_can_price_an_alternative_n_grid():
    three = RUN.plan((75, 100, 150))["projected_hours_at_max_iter"]
    four = RUN.plan((50, 75, 100, 150))["projected_hours_at_max_iter"]
    assert four > three
    assert RUN.plan((50,))["fits"] == 112


# ------------------------------------------------- production authorization

@pytest.fixture()
def approved(tmp_path, monkeypatch):
    """A world in which every authorization condition passes.

    Each test then breaks exactly one and asserts the refusal names it.
    """

    body = RUN.protocol_body_from_constants()
    digest = RUN.canonical_hash(body)
    frozen = tmp_path / "protocol.json"
    frozen.write_text(json.dumps({
        "status": "FROZEN", "protocol": body, "protocol_hash": digest,
        "experiment_id": RUN.EXPERIMENT_ID}, indent=2, sort_keys=True),
        encoding="utf-8")
    sha = "a" * 40
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps({
        "experiment_id": RUN.EXPERIMENT_ID, "protocol_hash": digest,
        "approved_run_code_sha": sha}), encoding="utf-8")
    fingerprint = RUN.environment_fingerprint()
    receipt = tmp_path / "gate_receipt.json"
    receipt.write_text(json.dumps({
        "experiment_id": RUN.EXPERIMENT_ID, "protocol_hash": digest,
        "code_sha": sha, "verdict": "PASS",
        "environment_key": RUN.environment_key(fingerprint),
        "environment": fingerprint}), encoding="utf-8")

    monkeypatch.setattr(RUN, "FROZEN_PROTOCOL_PATH", frozen)
    monkeypatch.setattr(RUN, "AUTHORIZATION_PATH", auth)
    monkeypatch.setattr(RUN, "GATE_RECEIPT_PATH", receipt)
    monkeypatch.setattr(RUN, "git_head", lambda: sha)
    monkeypatch.setattr(RUN, "working_tree_clean", lambda: True)
    return {"frozen": frozen, "auth": auth, "receipt": receipt, "sha": sha,
            "digest": digest}


def _failed(monkeypatch=None) -> list[str]:
    return RUN.authorize_production()["failed"]


def test_the_approved_world_authorizes(approved):
    decision = RUN.authorize_production()
    assert decision["production_authorization"] == "YES", decision["failed"]


def test_production_refuses_a_draft_protocol(approved, monkeypatch):
    document = json.loads(approved["frozen"].read_text(encoding="utf-8"))
    document["status"] = "DRAFT"
    approved["frozen"].write_text(json.dumps(document), encoding="utf-8")
    assert "A_frozen_protocol_exists" in _failed()


def test_production_refuses_a_missing_frozen_protocol(approved):
    approved["frozen"].unlink()
    failed = _failed()
    assert "A_frozen_protocol_exists" in failed
    assert "B_protocol_body_hash_matches" in failed


def test_production_refuses_a_tampered_protocol_body(approved):
    document = json.loads(approved["frozen"].read_text(encoding="utf-8"))
    document["protocol"]["max_iter"] = 41
    approved["frozen"].write_text(json.dumps(document), encoding="utf-8")
    assert "B_protocol_body_hash_matches" in _failed()


def test_production_refuses_a_missing_gate_receipt(approved):
    approved["receipt"].unlink()
    failed = _failed()
    assert "E_gate_receipt_exists" in failed
    assert "F_gate_verdict_is_pass" in failed


def test_production_refuses_a_failed_gate_receipt(approved):
    receipt = json.loads(approved["receipt"].read_text(encoding="utf-8"))
    receipt["verdict"] = "FAIL"
    approved["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    assert "F_gate_verdict_is_pass" in _failed()


def test_production_refuses_a_receipt_for_another_protocol(approved):
    receipt = json.loads(approved["receipt"].read_text(encoding="utf-8"))
    receipt["protocol_hash"] = "0" * 64
    approved["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    assert "G_gate_protocol_hash_matches" in _failed()


def test_production_refuses_a_receipt_from_another_commit(approved):
    receipt = json.loads(approved["receipt"].read_text(encoding="utf-8"))
    receipt["code_sha"] = "b" * 40
    approved["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    assert "H_gate_code_sha_matches" in _failed()


def test_production_refuses_an_environment_mismatch(approved):
    receipt = json.loads(approved["receipt"].read_text(encoding="utf-8"))
    receipt["environment_key"] = '{"numpy":"0.0.0"}'
    approved["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    assert "I_gate_environment_matches" in _failed()


def test_production_refuses_a_wrong_head(approved, monkeypatch):
    monkeypatch.setattr(RUN, "git_head", lambda: "c" * 40)
    assert "C_head_is_the_approved_commit" in _failed()


def test_production_refuses_a_dirty_tree(approved, monkeypatch):
    monkeypatch.setattr(RUN, "working_tree_clean", lambda: False)
    assert "D_working_tree_clean" in _failed()


def test_production_refuses_a_baseline_that_is_not_the_frozen_one(approved,
                                                                 monkeypatch):
    monkeypatch.setattr(RUN, "BASELINE_RUN_DIR", ROOT / "does_not_exist")
    assert "J_baseline_protocol_hash_matches" in _failed()


def test_production_refuses_without_the_flags(approved):
    with pytest.raises(RUN.HarnessStop):
        RUN.run_production(allow_em=False, confirm=True)
    with pytest.raises(RUN.HarnessStop):
        RUN.run_production(allow_em=True, confirm=False)


def test_production_refuses_when_authorization_fails(monkeypatch, tmp_path):
    """No CLI flag bypasses the authorization gate."""

    monkeypatch.setattr(RUN, "FROZEN_PROTOCOL_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(RUN, "ARTIFACT_DIR", tmp_path / "artifact")
    with pytest.raises(RUN.HarnessStop) as excinfo:
        RUN.run_production(allow_em=True, confirm=True)
    assert "PRODUCTION AUTHORIZATION = NO" in str(excinfo.value)
    assert not (tmp_path / "artifact").exists()


def test_production_is_fail_closed_on_an_existing_artifact_dir(approved,
                                                              monkeypatch,
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


def test_the_gate_refuses_while_the_protocol_is_a_draft():
    """There is no frozen protocol yet, so the gate cannot run at all."""

    assert not RUN.FROZEN_PROTOCOL_PATH.exists()
    with pytest.raises(RUN.HarnessStop) as excinfo:
        RUN.run_replication_gate(allow_em=True)
    assert "still a DRAFT" in str(excinfo.value)


def test_no_gate_receipt_exists_yet():
    """The gate has not been run; production has nothing to consume."""

    assert not RUN.GATE_RECEIPT_PATH.exists()
    assert not RUN.AUTHORIZATION_PATH.exists()
    assert not RUN.ARTIFACT_DIR.exists()


# ------------------------------------------------------------------ fixture

def _archive_rows() -> dict[tuple[int, int], dict[str, str]]:
    with (ARCHIVE / "fit_results.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(int(r["k_est"]), int(r["start"])): r for r in rows
            if int(r["k_true"]) == 5 and int(r["n"]) == FIXTURE_N
            and int(r["replicate"]) == FIXTURE_REPLICATE}


def _build_fixture(run_dir: Path, frozen_path: Path) -> None:
    """A genuinely valid artifact for one cell.

    The baseline checkpoint carries the archived values verbatim, so the
    replication check passes for the right reason; later checkpoints are
    synthesised with s2 recomputed from the identity and states written so the
    digest and diagnostic checks have real inputs.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    body = dict(RUN.protocol_body_from_constants())
    body["n_grid"] = [FIXTURE_N]
    body["replicates"] = FIXTURE_REPLICATE
    digest = RUN.canonical_hash(body)
    document = {"status": "FROZEN", "protocol": body, "protocol_hash": digest,
                "experiment_id": RUN.EXPERIMENT_ID}
    payload = json.dumps(document, indent=2, sort_keys=True)
    frozen_path.write_text(payload, encoding="utf-8")
    (run_dir / "protocol.json").write_text(payload, encoding="utf-8")

    train, test = AUDIT.build_masks(
        FIXTURE_N,
        AUDIT.split_seed(body["split_seed_base"], 5, FIXTURE_N, FIXTURE_REPLICATE),
        body["test_ratio"])
    train_hash, test_hash = AUDIT.stable_hash(train), AUDIT.stable_hash(test)

    archive = _archive_rows()
    ladder = body["checkpoints"]
    window = body["stability_window"]
    rng = np.random.default_rng(0)
    checkpoint_rows: list[dict] = []
    fit_rows: list[dict] = []
    states: dict[str, np.ndarray] = {}
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
                "train_mask_hash": train_hash, "test_mask_hash": test_hash,
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
                z = rng.normal(size=(FIXTURE_N, k_est))
                f_mat = rng.normal(size=(body["d"], k_est))
                w0_hat, w_hat = -1.0, 0.5
                state_key = f"{index:04d}_it{iteration:03d}"
                states[f"{state_key}_Z"] = z
                states[f"{state_key}_F"] = f_mat
                states[f"{state_key}_wpair"] = np.array([w0_hat, w_hat])
                row = dict(identity)
                row.update({
                    "checkpoint_iteration": iteration,
                    "heldout_mean_log_score": s1,
                    "q_strict": q,
                    "s2_q_based": -2.0 * q + npar * math.log(FIXTURE_N),
                    "s3_plugin_conditional": s3,
                    "num_params": npar,
                    "w0_hat": w0_hat, "w_hat": w_hat,
                    "cumulative_runtime_s": 1.0 * (position + 1),
                    "nan_iterations_so_far": 0, "attempt_index": 0,
                    "checkpoint_state_digest": AUDIT.state_digest(
                        z, f_mat, w0_hat, w_hat),
                    "state_key": state_key,
                })
                for key in RUN.DIAGNOSTIC_KEYS:
                    row[key] = 0.005
                    row[f"max_{key}_last{window}"] = 0.02
                for tau in RUN.TAU_LADDER:
                    row[f"budget_stable_gram_{tau:g}"] = 0.02 < tau
                    row[f"budget_stable_eta_y_{tau:g}"] = 0.02 < tau
                checkpoint_rows.append(row)
            fit_rows.append({
                **identity, "max_iter": body["max_iter"],
                "actual_iterations": body["max_iter"],
                "stopping_reason": "fixed_budget",
                "retry_count": 0, "retry_seed": identity["model_seed"],
                "nan_detected": False, "attempts_used": 1,
                "retry_events": "[]", "runtime_s": 10.0,
            })

    RUN.write_csv(run_dir / "checkpoint_results.csv", checkpoint_rows,
                  RUN.CHECKPOINT_COLUMNS)
    RUN.write_csv(run_dir / "fit_results.csv", fit_rows, RUN.FIT_COLUMNS)
    np.savez_compressed(run_dir / "checkpoint_states.npz", **states)
    (run_dir / "replication_vs_baseline.json").write_text(json.dumps({
        "baseline_run": ARCHIVE.name,
        "baseline_protocol_hash": RUN.BASELINE_PROTOCOL_HASH,
        "checkpoint": body["baseline_checkpoint"],
        "scope": "ALL fits, not only the gate cell",
        "compared_rows": len(fit_rows), "mismatch_count": 0, "match": True,
    }, indent=2), encoding="utf-8")
    (run_dir / "attempt.json").write_text(json.dumps({
        "attempt_id": "fixture-attempt", "status": "SUCCESS",
        "fits": len(fit_rows), "checkpoint_rows": len(checkpoint_rows),
    }, indent=2), encoding="utf-8")
    (run_dir / "runinfo.json").write_text(json.dumps({
        "attempt_id": "fixture-attempt",
        "experiment_id": RUN.EXPERIMENT_ID, "protocol_hash": digest,
        "attempted_fit_count": len(fit_rows),
        "completed_fit_count": len(fit_rows),
        "checkpoint_row_count": len(checkpoint_rows),
        "adaptive_stopping": False,
        "stopping_reasons": ["fixed_budget"],
        "wall_clock_seconds": 10.0 * len(fit_rows) + 5.0,
        "run_code_sha": "d" * 40,
        "working_tree_clean_before": True,
        "gate_receipt_protocol_hash": digest,
        "environment": {"python": "3.13.14", "numpy": np.__version__,
                        "scipy": "absent", "platform": "test",
                        "machine": "test"},
    }, indent=2), encoding="utf-8")


@pytest.fixture()
def artifact(tmp_path, monkeypatch) -> Path:
    run_dir = tmp_path / RUN.EXPERIMENT_ID
    frozen = tmp_path / "frozen_protocol.json"
    _build_fixture(run_dir, frozen)
    monkeypatch.setattr(AUDIT, "FROZEN_PROTOCOL_PATH", frozen)
    return run_dir


def _mutate_csv(path: Path, columns: list[str], mutate) -> None:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    RUN.write_csv(path, mutate(rows), columns)


def _checks(report) -> set[str]:
    return {f["check"] for f in report["findings"]}


# -------------------------------------------------------------- the auditor

def test_the_fixture_passes(artifact):
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "PASS", report["findings"]
    assert report["blocker_count"] == 0 and report["high_count"] == 0
    assert report["recomputed_selections"] == 3 * 6 * 1 * 1
    assert (artifact / "audit_selection_matrix.csv").exists()
    assert report["selection_summary"]


def test_auditor_detects_a_missing_checkpoint(artifact):
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS,
                lambda rows: rows[1:])
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert _checks(report) & {"checkpoint_row_count", "checkpoint_ladder"}


def test_auditor_detects_a_duplicate_checkpoint(artifact):
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS,
                lambda rows: rows[:-1] + [dict(rows[0])])
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "checkpoint_ladder" in _checks(report)


def test_auditor_detects_an_altered_baseline_score(artifact):
    def bump(rows):
        for row in rows:
            if int(row["checkpoint_iteration"]) == 8 and int(row["k_est"]) == 5:
                row["heldout_mean_log_score"] = str(
                    float(row["heldout_mean_log_score"]) + 0.05)
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "replication_vs_baseline" in _checks(report)


def test_auditor_detects_an_altered_later_s1(artifact):
    """H-05: a later checkpoint's S1 must not be unverifiable.

    It cannot be recomputed from the artifact, so the defence is that changing
    it changes the selection the auditor re-derives -- which is exactly what an
    analysis would read.
    """

    def bump(rows):
        for row in rows:
            if int(row["checkpoint_iteration"]) == 40 and int(row["k_est"]) == 7:
                row["heldout_mean_log_score"] = "9.0"
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    selection = {(r["criterion"], r["checkpoint"]): r["selected_k"]
                 for r in _read_selection(artifact)}
    assert selection[("S1", 40)] == 7


def test_auditor_detects_an_altered_later_s3(artifact):
    def bump(rows):
        for row in rows:
            if int(row["checkpoint_iteration"]) == 40 and int(row["k_est"]) == 1:
                row["s3_plugin_conditional"] = "-1e9"
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    AUDIT.audit(artifact)
    selection = {(r["criterion"], r["checkpoint"]): r["selected_k"]
                 for r in _read_selection(artifact)}
    assert selection[("S3", 40)] == 1


def _read_selection(run_dir: Path) -> list[dict]:
    with (run_dir / "audit_selection_matrix.csv").open(
            encoding="utf-8", newline="") as fh:
        return [{**r, "checkpoint": int(r["checkpoint"]),
                 "selected_k": int(r["selected_k"])} for r in csv.DictReader(fh)]


def test_auditor_detects_a_broken_s2_identity(artifact):
    def bump(rows):
        rows[3]["s2_q_based"] = str(float(rows[3]["s2_q_based"]) + 1.0)
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "s2_identity" in _checks(report)


def test_auditor_detects_a_swapped_checkpoint_identity(artifact):
    """H-04: a checkpoint row must belong to the fit it names."""

    def bump(rows):
        rows[0]["k_est"] = str(int(rows[0]["k_est"]) + 1)
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "checkpoint_identity_binding" in _checks(report)


def test_auditor_detects_a_swapped_checkpoint_seed(artifact):
    def bump(rows):
        rows[5]["model_seed"] = str(int(rows[5]["model_seed"]) + 7)
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert "checkpoint_identity_binding" in _checks(report)


def test_auditor_detects_a_nan_diagnostic(artifact):
    def bump(rows):
        rows[2]["eta_y_rel_change"] = "nan"
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "nonfinite_diagnostics" in _checks(report)


def test_auditor_detects_a_false_budget_stable_flag(artifact):
    def bump(rows):
        rows[4]["budget_stable_gram_0.001"] = "True"
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "budget_stable_flags" in _checks(report)


def test_auditor_detects_a_window_maximum_below_its_own_value(artifact):
    def bump(rows):
        rows[6]["max_gram_rel_change_last4"] = "0.0"
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "window_maximum_algebra" in _checks(report)


def test_auditor_detects_a_mutated_state_digest(artifact):
    def bump(rows):
        rows[1]["checkpoint_state_digest"] = "0" * 64
        return rows
    _mutate_csv(artifact / "checkpoint_results.csv", RUN.CHECKPOINT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "checkpoint_state_digest" in _checks(report)


def test_auditor_detects_a_mutated_archived_state(artifact):
    with np.load(artifact / "checkpoint_states.npz") as data:
        states = {k: data[k].copy() for k in data.files}
    key = next(k for k in states if k.endswith("_Z"))
    states[key] = states[key] + 1.0
    np.savez_compressed(artifact / "checkpoint_states.npz", **states)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "checkpoint_state_digest" in _checks(report)


def test_auditor_detects_missing_archived_state(artifact):
    (artifact / "checkpoint_states.npz").unlink()
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "checkpoint_states" in _checks(report)


def test_auditor_detects_retry_metadata_mismatch(artifact):
    def bump(rows):
        rows[0]["retry_count"] = "1"
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert {"retry_count", "retry_event_consistency"} <= _checks(report)


def test_auditor_detects_a_mutated_retry_seed(artifact):
    def bump(rows):
        rows[0]["retry_seed"] = str(int(rows[0]["retry_seed"]) + 1000)
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "retry_seed" in _checks(report)


def test_auditor_detects_a_hidden_nan(artifact):
    def bump(rows):
        rows[0]["nan_detected"] = "True"
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "nan_detected" in _checks(report)


def test_auditor_detects_adaptive_stopping(artifact):
    def bump(rows):
        rows[0]["stopping_reason"] = "converged"
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "stopping_reason" in _checks(report)


def test_auditor_detects_a_short_run(artifact):
    def bump(rows):
        rows[0]["actual_iterations"] = "32"
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "actual_iterations" in _checks(report)


def test_auditor_detects_a_forged_seed(artifact):
    def bump(rows):
        rows[0]["model_seed"] = str(int(rows[0]["model_seed"]) + 1)
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "seed_formulas" in _checks(report)


def test_auditor_detects_a_forged_mask_hash(artifact):
    def bump(rows):
        rows[0]["train_mask_hash"] = "0" * 64
        return rows
    _mutate_csv(artifact / "fit_results.csv", RUN.FIT_COLUMNS, bump)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "mask_hashes" in _checks(report)


def test_auditor_detects_a_tampered_protocol_body(artifact):
    doc = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    doc["protocol"]["max_iter"] = 41
    (artifact / "protocol.json").write_text(json.dumps(doc, indent=2,
                                                       sort_keys=True),
                                            encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "protocol_hash_recomputed" in _checks(report)


def test_auditor_detects_an_artifact_protocol_that_left_the_freeze_commit(
        artifact, monkeypatch, tmp_path):
    other = tmp_path / "other_frozen.json"
    other.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(AUDIT, "FROZEN_PROTOCOL_PATH", other)
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "protocol_bytes_match_freeze_commit" in _checks(report)


def test_auditor_refuses_a_failed_attempt(artifact):
    (artifact / "attempt.json").write_text(json.dumps({
        "attempt_id": "x", "status": "FAILED"}), encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "attempt_status" in _checks(report)


def test_auditor_refuses_an_interrupted_attempt(artifact):
    (artifact / "attempt.json").write_text(json.dumps({
        "attempt_id": "x", "status": "RUNNING"}), encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "attempt_status" in _checks(report)


def test_auditor_refuses_a_failed_run(artifact):
    (artifact / "failure.json").write_text("{}", encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "failure_marker" in _checks(report)


def test_auditor_detects_missing_environment_provenance(artifact):
    info = json.loads((artifact / "runinfo.json").read_text(encoding="utf-8"))
    info.pop("environment")
    (artifact / "runinfo.json").write_text(json.dumps(info), encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert "environment_provenance" in _checks(report)


def test_auditor_detects_impossible_runtime_accounting(artifact):
    info = json.loads((artifact / "runinfo.json").read_text(encoding="utf-8"))
    info["wall_clock_seconds"] = 1.0
    (artifact / "runinfo.json").write_text(json.dumps(info), encoding="utf-8")
    report = AUDIT.audit(artifact)
    assert report["verdict"] == "FAIL"
    assert "runtime_accounting" in _checks(report)


def test_auditor_records_its_own_limitations(artifact):
    report = AUDIT.audit(artifact)
    checks = _checks(report)
    assert "later_checkpoint_s1_s3_limitation" in checks
    assert "estimand_is_narrow" in checks
    assert "diagnostics_are_descriptive" in checks


def test_auditor_reproduces_the_frozen_selector(artifact):
    """P1, recomputed here a second time and compared with the auditor's."""

    AUDIT.audit(artifact)
    rows = _read_selection(artifact)
    with (artifact / "checkpoint_results.csv").open(encoding="utf-8",
                                                    newline="") as fh:
        raw = [r for r in csv.DictReader(fh)
               if int(r["checkpoint_iteration"]) == 8]
    means = {}
    for k in RUN.BASE.CANDIDATE_K:
        values = [float(r["heldout_mean_log_score"]) for r in raw
                  if int(r["k_est"]) == k]
        assert len(values) == 2
        means[k] = float(np.mean(values))
    expected, _ = AUDIT.select_k(means, 1e-12)
    got = next(r for r in rows if r["criterion"] == "S1" and r["checkpoint"] == 8)
    assert got["selected_k"] == expected


# ------------------------------------------------------- gate receipt shape

def test_the_gate_receipt_schema_is_specified_before_the_gate_runs():
    """The gate has not been run; the receipt's contract is fixed anyway."""

    source = Path(RUN.__file__).read_text(encoding="utf-8")
    for field in ("receipt_version", "experiment_id", "protocol_hash",
                  "code_sha", "baseline_protocol_hash", "environment",
                  "environment_key", "gate_cell", "expected_fits",
                  "actual_fits", "compared_fields", "per_fit",
                  "mismatch_count", "verdict", "timestamp_utc",
                  "retry_count_total", "nan_detected_fits"):
        assert f'"{field}"' in source, field


def test_replication_comparison_is_exact():
    """A value that is merely close must fail."""

    archive = RUN.read_archive_fits()
    reference = archive[(5, FIXTURE_N, FIXTURE_REPLICATE, 5, 1)]
    good = [{
        "fit_index": 1, "k_true": 5, "n": FIXTURE_N,
        "replicate": FIXTURE_REPLICATE, "k_est": 5, "start": 1,
        "checkpoint_iteration": 8,
        **{f: (int(reference[f]) if f in RUN.REPLICATION_INT_FIELDS
               else float(reference[f])) for f in RUN.REPLICATION_FIELDS},
    }]
    assert RUN.compare_to_archive(good)["match"]

    nudged = [dict(good[0])]
    nudged[0]["q_strict"] = math.nextafter(nudged[0]["q_strict"], math.inf)
    result = RUN.compare_to_archive(nudged)
    assert not result["match"]
    assert "q_strict" in result["mismatches"][0]["problem"]
