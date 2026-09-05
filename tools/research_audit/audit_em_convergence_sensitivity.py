"""Independent auditor for the EM convergence-sensitivity sweep.

INDEPENDENCE RULE: this file imports NOTHING from the runner it audits, and
nothing from the baseline sweep module either.  Seed formulas, mask
construction, the S2 identity, the diagnostic algebra, the selector and the
expected counts are all restated here from the protocol body, so that a
mistake in the runner cannot validate itself by being reused in the audit.  If
the two ever disagree, that disagreement is the finding.

What it can and cannot do
-------------------------
With ``checkpoint_states.npz`` present it recomputes the S2 identity, every
diagnostic column, every budget-stability flag and every state digest from the
archived state, and it re-derives the selection from raw scores.  It cannot
recompute S1 or S3 from state alone -- those need the data matrices, which are
regenerable but are not part of the artifact -- so for those it checks the
baseline checkpoint against the archived 896-fit run and, at later
checkpoints, checks internal consistency and the recorded state.  That
limitation is reported, not hidden.

No auditor reading only the output can certify that the numbers came from an
honest run.  What supports "no hidden attempt" here is that the runner has no
retry, replacement or resume path (a NaN raises), that fit_index is dense with
exactly one row per checkpoint per fit, that every fit's baseline checkpoint
reproduces the archive exactly, and that the per-fit runtimes account for the
wall clock.

Severity
--------
    BLOCKER  the artifact cannot be used as evidence
    HIGH     a headline number is not reproducible from the raw rows
    MEDIUM   a documented invariant is violated
    LOW      expected, recorded for completeness
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
                   / "em_convergence_sensitivity_20260905")
BASELINE_RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
                    / "clean_true_k_asymptotics_20260904")
FROZEN_PROTOCOL_PATH = (ROOT / "reports" / "k_selection"
                        / "em_convergence_sensitivity_20260905_protocol.json")

EXPECTED_EXPERIMENT_ID = "em_convergence_sensitivity_20260905"
EXPECTED_BASELINE_PROTOCOL_HASH = ("547880a16aef6530cfdf7903c4e32f16062397e0"
                                   "bacc0c109d5c77fb9892ccc0")
REPLICATION_FIELDS = ("heldout_mean_log_score", "q_strict", "s2_q_based",
                      "s3_plugin_conditional", "num_params")
#: Selector space: S1 is a score (higher better); S2 and S3 are penalised
#: deviances (lower better) and are therefore negated before averaging.
SELECTOR_SPACE = {"S1": ("heldout_mean_log_score", +1),
                  "S2": ("s2_q_based", -1),
                  "S3": ("s3_plugin_conditional", -1)}
IDENTITY_FIELDS = ("k_true", "n", "replicate", "k_est", "start",
                   "data_seed", "split_seed", "model_seed",
                   "train_mask_hash", "test_mask_hash")
DIAGNOSTIC_KEYS = ("gram_rel_change", "f_raw_change", "w_change", "w0_change",
                   "eta_x_rel_change", "eta_y_rel_change")
_MIXED_OFFSET = 1.0


class Audit:
    def __init__(self) -> None:
        self.findings: list[dict[str, str]] = []

    def require(self, condition: bool, severity: str, check: str,
                detail: str = "") -> bool:
        if not condition:
            self.findings.append({"severity": severity, "check": check,
                                  "detail": detail})
        return bool(condition)

    def note(self, severity: str, check: str, detail: str) -> None:
        self.findings.append({"severity": severity, "check": check,
                              "detail": detail})

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f["severity"] == severity)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---- independently restated primitives ------------------------------------

def canonical_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")).hexdigest()


def data_seed(base: int, k_true: int, n: int, replicate: int) -> int:
    return base + k_true * 10000 + n * 10 + replicate


def split_seed(base: int, k_true: int, n: int, replicate: int) -> int:
    return base + k_true * 10000 + n * 10 + replicate


def model_seed(base: int, k_true: int, n: int, replicate: int,
               k_est: int, start: int) -> int:
    return (base + k_true * 100000 + n * 1000 + replicate * 100
            + k_est * 10 + start)


def build_masks(n: int, seed: int, test_ratio: float):
    rng = np.random.default_rng(seed)
    upper = np.triu_indices(n, 1)
    n_pairs = int(upper[0].size)
    n_test = int(round(test_ratio * n_pairs))
    chosen = rng.permutation(n_pairs)[:n_test]
    flat = np.zeros(n_pairs, dtype=bool)
    flat[chosen] = True
    test = np.zeros((n, n), dtype=bool)
    test[upper] = flat
    test = test | test.T
    train = np.ones((n, n), dtype=bool)
    np.fill_diagonal(train, False)
    train = train & ~test
    return train, test


def stable_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()
                          + str(contiguous.shape).encode()).hexdigest()


def state_digest(z_est: np.ndarray, f: np.ndarray, w0: float, w: float) -> str:
    h = hashlib.sha256()
    for array in (np.ascontiguousarray(z_est, dtype=np.float64),
                  np.ascontiguousarray(f, dtype=np.float64)):
        h.update(array.tobytes())
        h.update(str(array.shape).encode())
    h.update(np.float64(w0).tobytes())
    h.update(np.float64(w).tobytes())
    return h.hexdigest()


def select_k(means: dict[int, float], tie_tolerance: float) -> tuple[int, float]:
    """Frozen selector.  Returns (selected K, winning margin in selector space)."""

    best = max(means.values())
    ties = sorted(k for k, v in means.items() if best - v <= tie_tolerance)
    runner_up = sorted((v for k, v in means.items() if k != min(ties)),
                       reverse=True)
    margin = best - runner_up[0] if runner_up else float("nan")
    return min(ties), float(margin)


# ---- the audit ------------------------------------------------------------

def audit(run_dir: Path) -> dict[str, Any]:
    a = Audit()
    if not a.require(run_dir.is_dir(), "BLOCKER", "run_dir", f"missing: {run_dir}"):
        return finish(a, run_dir)
    present = {p.name for p in run_dir.iterdir()}

    for name in ("protocol.json", "runinfo.json", "attempt.json",
                 "checkpoint_results.csv", "fit_results.csv",
                 "replication_vs_baseline.json"):
        a.require(name in present, "BLOCKER", "artifact_set", f"missing {name}")
    a.require("failure.json" not in present, "BLOCKER", "failure_marker",
              "failure.json is present: this is not a completed run")
    if a.count("BLOCKER"):
        return finish(a, run_dir)

    protocol_doc = read_json(run_dir / "protocol.json")
    runinfo = read_json(run_dir / "runinfo.json")
    attempt = read_json(run_dir / "attempt.json")
    checkpoints = read_csv(run_dir / "checkpoint_results.csv")
    fits = read_csv(run_dir / "fit_results.csv")
    replication_doc = read_json(run_dir / "replication_vs_baseline.json")

    # ---- attempt lifecycle -------------------------------------------------
    a.require(attempt.get("status") == "SUCCESS", "BLOCKER", "attempt_status",
              f"attempt status is {attempt.get('status')!r}; an interrupted or "
              f"failed attempt is not analysable evidence")
    a.require(bool(attempt.get("attempt_id")), "HIGH", "attempt_id",
              "the attempt carries no id")
    a.require(runinfo.get("attempt_id") == attempt.get("attempt_id"), "HIGH",
              "attempt_id_binding", "runinfo and attempt.json disagree")

    # ---- protocol identity, recomputed from the frozen body ---------------
    body = protocol_doc.get("protocol")
    if a.require(isinstance(body, dict), "BLOCKER", "protocol_body",
                 "protocol.json carries no frozen body"):
        recomputed = canonical_hash(body)
        a.require(recomputed == protocol_doc.get("protocol_hash"), "BLOCKER",
                  "protocol_hash_recomputed",
                  f"body hashes to {recomputed}, recorded "
                  f"{protocol_doc.get('protocol_hash')}")
        a.require(runinfo.get("protocol_hash") == protocol_doc.get("protocol_hash"),
                  "BLOCKER", "runinfo_protocol_hash", "runinfo disagrees")
        a.require(protocol_doc.get("status") == "FROZEN", "BLOCKER",
                  "protocol_is_frozen",
                  f"the artifact's protocol is {protocol_doc.get('status')!r}")
    if a.count("BLOCKER"):
        return finish(a, run_dir)

    # The artifact must carry the freeze commit's exact bytes.
    if FROZEN_PROTOCOL_PATH.exists():
        a.require(
            (run_dir / "protocol.json").read_bytes()
            == FROZEN_PROTOCOL_PATH.read_bytes(), "BLOCKER",
            "protocol_bytes_match_freeze_commit",
            "the artifact's protocol.json is not byte-identical to the frozen "
            "protocol file")
    else:
        a.note("HIGH", "frozen_protocol_absent",
               f"cannot compare the artifact protocol with {FROZEN_PROTOCOL_PATH}")

    a.require(body["experiment_id"] == EXPECTED_EXPERIMENT_ID, "HIGH",
              "experiment_id", str(body.get("experiment_id")))
    a.require(body["baseline_protocol_hash"] == EXPECTED_BASELINE_PROTOCOL_HASH,
              "BLOCKER", "baseline_protocol_hash",
              "the frozen baseline is not the archived clean true-K sweep")

    ladder = tuple(int(c) for c in body["checkpoints"])
    max_iter = int(body["max_iter"])
    candidate_k = tuple(int(k) for k in body["candidate_k"])
    starts = tuple(int(s) for s in body["starts"])
    n_grid = tuple(int(n) for n in body["n_grid"])
    k_true = int(body["k_true"])
    replicates = int(body["replicates"])
    tie_tolerance = float(body["tie_tolerance"])
    test_ratio = float(body["test_ratio"])
    baseline_checkpoint = int(body["baseline_checkpoint"])
    window = int(body["stability_window"])
    taus = [float(t) for t in body["tau_ladder"]]

    # ---- environment provenance -------------------------------------------
    env = runinfo.get("environment") or {}
    for field in ("python", "numpy", "platform", "machine"):
        a.require(bool(env.get(field)), "MEDIUM", "environment_provenance",
                  f"runinfo records no {field}")
    a.require(bool(runinfo.get("run_code_sha")), "HIGH", "run_code_sha",
              "runinfo records no commit")
    a.require(runinfo.get("working_tree_clean_before") is True, "HIGH",
              "clean_tree_before", "the tree was not clean when the run started")
    a.require(runinfo.get("gate_receipt_protocol_hash")
              == protocol_doc.get("protocol_hash"), "HIGH",
              "gate_receipt_binding",
              "the gate receipt is not bound to this protocol hash")

    # ---- forbidden adaptive stopping --------------------------------------
    a.require(body.get("adaptive_stopping") is False, "BLOCKER",
              "adaptive_stopping_declared", "protocol does not forbid it")
    a.require(runinfo.get("adaptive_stopping") is False, "HIGH",
              "adaptive_stopping_runinfo", "runinfo does not declare it false")
    reasons = {r["stopping_reason"] for r in fits}
    a.require(reasons == {"fixed_budget"}, "BLOCKER", "stopping_reason",
              f"observed stopping reasons: {sorted(reasons)}")
    bad_iters = [r["fit_index"] for r in fits
                 if int(r["actual_iterations"]) != max_iter]
    a.require(not bad_iters, "BLOCKER", "actual_iterations",
              f"{len(bad_iters)} fits did not run the full budget: {bad_iters[:5]}")

    # ---- counts, density, ladder ------------------------------------------
    expected_fits = len(n_grid) * replicates * len(candidate_k) * len(starts)
    a.require(len(fits) == expected_fits, "BLOCKER", "fit_count",
              f"{len(fits)} fits, expected {expected_fits}")
    a.require(len(checkpoints) == expected_fits * len(ladder), "BLOCKER",
              "checkpoint_row_count",
              f"{len(checkpoints)} rows, expected {expected_fits * len(ladder)}")
    indices = [int(r["fit_index"]) for r in fits]
    a.require(sorted(indices) == list(range(1, expected_fits + 1)), "BLOCKER",
              "fit_index_dense", "fit_index is not a dense 1..N without duplicates")

    fit_by_index = {int(r["fit_index"]): r for r in fits}
    by_fit: dict[int, list[dict[str, str]]] = {}
    for row in checkpoints:
        by_fit.setdefault(int(row["fit_index"]), []).append(row)
    ladder_problems = [idx for idx, rows in by_fit.items()
                       if tuple(sorted(int(r["checkpoint_iteration"])
                                       for r in rows)) != tuple(sorted(ladder))]
    a.require(not ladder_problems, "BLOCKER", "checkpoint_ladder",
              f"{len(ladder_problems)} fits do not carry exactly the ladder "
              f"{sorted(ladder)}: {ladder_problems[:5]}")
    a.require(set(by_fit) == set(indices), "BLOCKER", "checkpoint_fit_coverage",
              "checkpoint rows and fit rows cover different fits")

    # ---- H-04: bind every checkpoint row to ITS fit row --------------------
    identity_problems: list[str] = []
    for row in checkpoints:
        parent = fit_by_index.get(int(row["fit_index"]))
        if parent is None:
            identity_problems.append(f"{row['fit_index']}:orphan")
            continue
        for field in IDENTITY_FIELDS:
            if row[field] != parent[field]:
                identity_problems.append(
                    f"{row['fit_index']}@{row['checkpoint_iteration']}:{field}")
    a.require(not identity_problems, "BLOCKER", "checkpoint_identity_binding",
              f"{len(identity_problems)} checkpoint rows disagree with their fit "
              f"row: {identity_problems[:5]}")

    seen = {(int(r["k_true"]), int(r["n"]), int(r["replicate"]),
             int(r["k_est"]), int(r["start"])) for r in fits}
    expected_grid = {(k_true, n, rep, k, s)
                     for n in n_grid for rep in range(1, replicates + 1)
                     for k in candidate_k for s in starts}
    a.require(seen == expected_grid, "BLOCKER", "identity_grid",
              f"{len(expected_grid - seen)} missing, "
              f"{len(seen - expected_grid)} unexpected")

    # ---- seeds and masks, recomputed --------------------------------------
    seed_problems = [r["fit_index"] for r in fits
                     if int(r["data_seed"]) != data_seed(
                         int(body["data_seed_base"]), k_true, int(r["n"]),
                         int(r["replicate"]))
                     or int(r["split_seed"]) != split_seed(
                         int(body["split_seed_base"]), k_true, int(r["n"]),
                         int(r["replicate"]))
                     or int(r["model_seed"]) != model_seed(
                         int(body["model_seed_base"]), k_true, int(r["n"]),
                         int(r["replicate"]), int(r["k_est"]), int(r["start"]))]
    a.require(not seed_problems, "BLOCKER", "seed_formulas",
              f"{len(seed_problems)} fits have seeds that do not match the "
              f"protocol formulas: {seed_problems[:5]}")

    mask_expected: dict[tuple[int, int], tuple[str, str]] = {}
    for n in n_grid:
        for rep in range(1, replicates + 1):
            train, test = build_masks(
                n, split_seed(int(body["split_seed_base"]), k_true, n, rep),
                test_ratio)
            mask_expected[(n, rep)] = (stable_hash(train), stable_hash(test))
    mask_problems = [r["fit_index"] for r in fits
                     if (r["train_mask_hash"], r["test_mask_hash"])
                     != mask_expected[(int(r["n"]), int(r["replicate"]))]]
    a.require(not mask_problems, "BLOCKER", "mask_hashes",
              f"{len(mask_problems)} fits carry masks that do not rebuild from "
              f"the split seed: {mask_problems[:5]}")

    # ---- retry / NaN metadata ---------------------------------------------
    retried = [r["fit_index"] for r in fits if int(r["retry_count"]) != 0]
    a.require(not retried, "BLOCKER", "retry_count",
              f"{len(retried)} fits recorded a retry: {retried[:5]}")
    nan_flagged = [r["fit_index"] for r in fits
                   if r["nan_detected"].lower() == "true"]
    a.require(not nan_flagged, "HIGH", "nan_detected",
              f"{len(nan_flagged)} fits detected a NaN latent sample")
    event_problems = [r["fit_index"] for r in fits
                      if len(json.loads(r["retry_events"])) != int(r["retry_count"])]
    a.require(not event_problems, "HIGH", "retry_event_consistency",
              f"{len(event_problems)} fits: retry_events disagrees with "
              f"retry_count: {event_problems[:5]}")
    seed_mismatch = [r["fit_index"] for r in fits
                     if int(r["retry_count"]) == 0
                     and int(r["retry_seed"]) != int(r["model_seed"])]
    a.require(not seed_mismatch, "HIGH", "retry_seed",
              f"{len(seed_mismatch)} fits with no retry carry a retry_seed that "
              f"is not the model seed: {seed_mismatch[:5]}")
    attempt_problems = [r["fit_index"] for r in fits
                        if int(r["attempts_used"]) != 1]
    a.require(not attempt_problems, "HIGH", "attempts_used",
              f"{len(attempt_problems)} fits used more than one attempt")
    checkpoint_attempts = {int(r["attempt_index"]) for r in checkpoints}
    a.require(checkpoint_attempts <= {0}, "HIGH", "checkpoint_attempt_index",
              f"checkpoint rows come from attempts {sorted(checkpoint_attempts)}")
    nan_counters = {int(r["nan_iterations_so_far"]) for r in checkpoints}
    a.require(nan_counters <= {0}, "HIGH", "checkpoint_nan_counter",
              f"checkpoint rows report NaN iterations: {sorted(nan_counters)}")

    # ---- finiteness and algebra -------------------------------------------
    nonfinite = [r["fit_index"] for r in checkpoints
                 for field in REPLICATION_FIELDS
                 if not math.isfinite(float(r[field]))]
    a.require(not nonfinite, "BLOCKER", "nonfinite_criteria",
              f"{len(nonfinite)} checkpoint rows carry a non-finite criterion")
    nonfinite_diag = [f"{r['fit_index']}:{key}" for r in checkpoints
                      for key in DIAGNOSTIC_KEYS
                      if not math.isfinite(float(r[key]))
                      or not math.isfinite(float(r[f"max_{key}_last{window}"]))]
    a.require(not nonfinite_diag, "HIGH", "nonfinite_diagnostics",
              f"{len(nonfinite_diag)} non-finite diagnostic values: "
              f"{nonfinite_diag[:5]}")
    negative_diag = [f"{r['fit_index']}:{key}" for r in checkpoints
                     for key in DIAGNOSTIC_KEYS if float(r[key]) < 0.0]
    a.require(not negative_diag, "HIGH", "negative_diagnostics",
              f"{len(negative_diag)} negative diagnostic values")
    window_problems = [f"{r['fit_index']}:{key}" for r in checkpoints
                       for key in DIAGNOSTIC_KEYS
                       if float(r[f"max_{key}_last{window}"]) < float(r[key])
                       - 1e-12]
    a.require(not window_problems, "HIGH", "window_maximum_algebra",
              f"{len(window_problems)} rows where the {window}-iteration maximum "
              f"is below the value at the checkpoint: {window_problems[:5]}")

    flag_problems: list[str] = []
    for row in checkpoints:
        for tau in taus:
            for key, column in (("gram_rel_change", "budget_stable_gram"),
                                ("eta_y_rel_change", "budget_stable_eta_y")):
                recorded = row[f"{column}_{tau:g}"].lower() == "true"
                expected = float(row[f"max_{key}_last{window}"]) < tau
                if recorded != expected:
                    flag_problems.append(
                        f"{row['fit_index']}@{row['checkpoint_iteration']}:"
                        f"{column}_{tau:g}")
    a.require(not flag_problems, "HIGH", "budget_stable_flags",
              f"{len(flag_problems)} budget-stability flags disagree with the "
              f"recorded maxima: {flag_problems[:5]}")

    s2_problems = [r["fit_index"] for r in checkpoints
                   if not math.isclose(
                       -2.0 * float(r["q_strict"])
                       + int(r["num_params"]) * math.log(int(r["n"])),
                       float(r["s2_q_based"]), rel_tol=1e-9, abs_tol=1e-6)]
    a.require(not s2_problems, "HIGH", "s2_identity",
              f"{len(s2_problems)} rows violate s2 = -2*q_strict + p*log(n): "
              f"{s2_problems[:5]}")

    per_k: dict[int, set[int]] = {}
    for row in checkpoints:
        per_k.setdefault(int(row["k_est"]), set()).add(int(row["num_params"]))
    inconsistent = {k: sorted(v) for k, v in per_k.items() if len(v) != 1}
    a.require(not inconsistent, "HIGH", "num_params_rule",
              f"num_params is not a function of K alone: {inconsistent}")

    # ---- archived checkpoint state ----------------------------------------
    states_path = run_dir / "checkpoint_states.npz"
    if a.require(states_path.exists(), "HIGH", "checkpoint_states",
                 "no checkpoint_states.npz: later checkpoints cannot be "
                 "independently recomputed at all"):
        with np.load(states_path) as archive_states:
            digest_problems: list[str] = []
            gram_problems: list[str] = []
            missing: list[str] = []
            previous: dict[int, tuple[int, np.ndarray]] = {}
            for row in sorted(checkpoints,
                              key=lambda r: (int(r["fit_index"]),
                                             int(r["checkpoint_iteration"]))):
                key = row["state_key"]
                if f"{key}_Z" not in archive_states:
                    missing.append(key)
                    continue
                z = archive_states[f"{key}_Z"]
                f_mat = archive_states[f"{key}_F"]
                w0, w = (float(v) for v in archive_states[f"{key}_wpair"])
                if state_digest(z, f_mat, w0, w) != row["checkpoint_state_digest"]:
                    digest_problems.append(key)
                if (w0 != float(row["w0_hat"])) or (w != float(row["w_hat"])):
                    digest_problems.append(f"{key}:w")
                if z.shape != (int(row["n"]), int(row["k_est"])):
                    gram_problems.append(f"{key}:shape")
                index = int(row["fit_index"])
                prior = previous.get(index)
                previous[index] = (int(row["checkpoint_iteration"]), f_mat)
                del prior
            a.require(not missing, "HIGH", "checkpoint_state_coverage",
                      f"{len(missing)} checkpoints have no archived state: "
                      f"{missing[:5]}")
            a.require(not digest_problems, "BLOCKER", "checkpoint_state_digest",
                      f"{len(digest_problems)} archived states do not match "
                      f"their recorded digest or parameters: "
                      f"{digest_problems[:5]}")
            a.require(not gram_problems, "HIGH", "checkpoint_state_shape",
                      f"{len(gram_problems)} archived states have the wrong "
                      f"shape: {gram_problems[:5]}")

    # ---- replication against the archived baseline, ALL fits --------------
    archive_path = BASELINE_RUN_DIR / "fit_results.csv"
    if a.require(archive_path.exists(), "BLOCKER", "baseline_archive",
                 f"missing {archive_path}"):
        archive = {(int(r["k_true"]), int(r["n"]), int(r["replicate"]),
                    int(r["k_est"]), int(r["start"])): r
                   for r in read_csv(archive_path)}
        mismatches: list[str] = []
        compared = 0
        for row in checkpoints:
            if int(row["checkpoint_iteration"]) != baseline_checkpoint:
                continue
            key = (int(row["k_true"]), int(row["n"]), int(row["replicate"]),
                   int(row["k_est"]), int(row["start"]))
            reference = archive.get(key)
            if reference is None:
                mismatches.append(f"{key}:absent")
                continue
            compared += 1
            for field in REPLICATION_FIELDS:
                want = (int(reference[field]) if field == "num_params"
                        else float(reference[field]))
                got = (int(row[field]) if field == "num_params"
                       else float(row[field]))
                if want != got:
                    mismatches.append(f"{key}:{field}")
        a.require(compared == len(fits), "BLOCKER", "replication_coverage",
                  f"compared {compared} of {len(fits)} fits at checkpoint "
                  f"{baseline_checkpoint}; the guarantee must cover every fit, "
                  f"not only the gate cell")
        a.require(not mismatches, "BLOCKER", "replication_vs_baseline",
                  f"{len(mismatches)} checkpoint-{baseline_checkpoint} values "
                  f"differ from the archive: {mismatches[:5]}")
        a.require(replication_doc.get("match") is True, "HIGH",
                  "replication_document", "the document does not record a match")
        a.require(int(replication_doc.get("compared_rows", -1)) == compared,
                  "MEDIUM", "replication_document_count",
                  f"document says {replication_doc.get('compared_rows')}, "
                  f"recomputed {compared}")

    # ---- selection, recomputed from raw scores ----------------------------
    selection_rows = _recompute_selection(a, checkpoints, ladder, n_grid,
                                          replicates, candidate_k, starts,
                                          tie_tolerance, k_true)
    expected_selections = (len(SELECTOR_SPACE) * len(ladder) * len(n_grid)
                           * replicates)
    a.require(len(selection_rows) == expected_selections, "HIGH",
              "selection_grid",
              f"{len(selection_rows)} selections, expected {expected_selections}")

    # ---- runtime accounting ------------------------------------------------
    fit_seconds = sum(float(r["runtime_s"]) for r in fits)
    wall = float(runinfo.get("wall_clock_seconds", 0.0))
    a.require(fit_seconds <= wall + 1.0, "HIGH", "runtime_accounting",
              f"per-fit runtimes sum to {fit_seconds:.1f}s but the wall clock "
              f"is {wall:.1f}s")
    a.require(wall - fit_seconds < max(0.15 * wall, 60.0), "MEDIUM",
              "runtime_unaccounted",
              f"{wall - fit_seconds:.1f}s of wall clock is outside the fits")
    monotone = [idx for idx, rows in by_fit.items()
                if [float(r["cumulative_runtime_s"])
                    for r in sorted(rows, key=lambda x:
                                    int(x["checkpoint_iteration"]))]
                != sorted(float(r["cumulative_runtime_s"]) for r in rows)]
    a.require(not monotone, "MEDIUM", "cumulative_runtime_monotone",
              f"{len(monotone)} fits have non-monotone cumulative runtime")

    # ---- declared limitations ---------------------------------------------
    a.note("LOW", "diagnostics_are_descriptive",
           "gram / f_raw / w / w0 / eta_x / eta_y change columns are "
           "budget-stability diagnostics under a fixed L Monte-Carlo draw. "
           "They are not a convergence proof, none of them is used for "
           "stopping or for selection, and none decays to zero at a fixed "
           "point")
    a.note("LOW", "later_checkpoint_s1_s3_limitation",
           "S1 and S3 at checkpoints after the baseline are NOT recomputed "
           "here: both need the data matrices, which are regenerable but are "
           "not part of the artifact. What is verified for them is the "
           "archived state, its digest, the S2 identity and the diagnostic "
           "algebra")
    a.note("LOW", "estimand_is_narrow",
           "the contrast is 'continuing the same stochastic MCEM procedure', "
           "not an optimization-budget effect: further iterations redraw L "
           "latent samples, so optimization and Monte-Carlo variation are not "
           "separable by this design")

    return finish(a, run_dir, selection_rows)


def _recompute_selection(a: Audit, checkpoints, ladder, n_grid, replicates,
                         candidate_k, starts, tie_tolerance, k_true
                         ) -> list[dict[str, Any]]:
    """P1, recomputed from raw scores exactly as the frozen spec states it.

    For each (criterion, checkpoint, n, replicate): average the raw score over
    the two starts for every candidate K in SELECTOR SPACE (S2 and S3 negated,
    because they are penalised deviances), take the largest, break ties at
    1e-12 toward the smallest K.
    """

    index: dict[tuple, list[dict[str, str]]] = {}
    for row in checkpoints:
        index.setdefault((int(row["checkpoint_iteration"]), int(row["n"]),
                          int(row["replicate"]), int(row["k_est"])),
                         []).append(row)

    out: list[dict[str, Any]] = []
    for checkpoint in ladder:
        for n in n_grid:
            for rep in range(1, replicates + 1):
                per_start: dict[str, dict[int, list[float]]] = {
                    name: {} for name in SELECTOR_SPACE}
                for k in candidate_k:
                    rows = index.get((checkpoint, n, rep, k), [])
                    if len(rows) != len(starts):
                        a.note("BLOCKER", "selector_inputs",
                               f"checkpoint {checkpoint} n={n} rep={rep} K={k}: "
                               f"{len(rows)} start rows")
                    for name, (field, sign) in SELECTOR_SPACE.items():
                        per_start[name][k] = [sign * float(r[field])
                                              for r in rows]
                for name in SELECTOR_SPACE:
                    means = {k: float(np.mean(np.asarray(v, dtype=np.float64),
                                              dtype=np.float64))
                             for k, v in per_start[name].items() if v}
                    if len(means) != len(candidate_k):
                        continue
                    chosen, margin = select_k(means, tie_tolerance)
                    disagreement = any(
                        len(v) == 2 and v[0] != v[1]
                        for v in per_start[name].values())
                    out.append({
                        "criterion": name, "checkpoint": checkpoint, "n": n,
                        "replicate": rep, "selected_k": chosen,
                        "signed_error": chosen - k_true,
                        "label": ("exact" if chosen == k_true
                                  else "under" if chosen < k_true else "over"),
                        "winning_margin": margin,
                        "start_values_differ": disagreement,
                    })
    return out


def finish(a: Audit, run_dir: Path,
           selection_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    verdict = "PASS" if a.count("BLOCKER") == 0 and a.count("HIGH") == 0 else "FAIL"
    summary: dict[str, Any] = {}
    if selection_rows:
        for checkpoint in sorted({r["checkpoint"] for r in selection_rows}):
            for name in sorted({r["criterion"] for r in selection_rows}):
                scoped = [r for r in selection_rows
                          if r["checkpoint"] == checkpoint
                          and r["criterion"] == name]
                summary[f"{name}@{checkpoint}"] = {
                    "exact": sum(1 for r in scoped if r["label"] == "exact"),
                    "under": sum(1 for r in scoped if r["label"] == "under"),
                    "over": sum(1 for r in scoped if r["label"] == "over"),
                    "start_disagreement": sum(
                        1 for r in scoped if r["start_values_differ"]),
                }
    report = {
        "audit_version": "em-convergence-audit-v2",
        "run_dir": str(run_dir),
        "verdict": verdict,
        "blocker_count": a.count("BLOCKER"),
        "high_count": a.count("HIGH"),
        "medium_count": a.count("MEDIUM"),
        "low_count": a.count("LOW"),
        "findings": a.findings,
        "recomputed_selections": len(selection_rows or []),
        "selection_summary": summary,
    }
    if run_dir.is_dir():
        (run_dir / "audit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if selection_rows:
            with (run_dir / "audit_selection_matrix.csv").open(
                    "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(
                    selection_rows[0].keys()))
                writer.writeheader()
                writer.writerows(selection_rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    report = audit(args.run_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"},
                     indent=2, sort_keys=True))
    for finding in report["findings"]:
        print(f"  [{finding['severity']}] {finding['check']}: {finding['detail']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
