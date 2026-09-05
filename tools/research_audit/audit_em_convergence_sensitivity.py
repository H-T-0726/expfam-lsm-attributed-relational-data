"""Independent auditor for the EM convergence-sensitivity sweep.

INDEPENDENCE RULE: this file imports NOTHING from the runner it audits.  Seed
formulas, mask construction, the S2 identity, the selector and the expected
counts are all restated here from the protocol body, so that a mistake in the
runner cannot validate itself by being reused in the audit.  If the two ever
disagree, that disagreement is the finding.

What it can and cannot do
-------------------------
It re-derives quantities from the artifact and reports disagreements.  It
cannot certify that the numbers came from an honest run -- no auditor reading
only the output can.  What supports "no hidden attempt" here is that the
runner has no retry, replacement or resume path at all (a NaN restart raises),
that fit_index is dense with exactly one row per checkpoint per fit, and that
the per-fit runtimes account for the wall clock.

Severity
--------
    BLOCKER  the artifact cannot be used as evidence
    HIGH     a headline number is not reproducible from the raw rows
    MEDIUM   a documented invariant is violated
    LOW      expected, recorded for completeness

Usage
-----
    python tools/research_audit/audit_em_convergence_sensitivity.py \
        --run-dir expfam/results/k_selection/em_convergence_sensitivity_20260905
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

EXPECTED_EXPERIMENT_ID = "em_convergence_sensitivity_20260905"
EXPECTED_BASELINE_PROTOCOL_HASH = ("547880a16aef6530cfdf7903c4e32f16062397e0"
                                   "bacc0c109d5c77fb9892ccc0")
#: Fields compared against the archive at the baseline checkpoint.  Exact, for
#: the same reason the runner uses exact comparison: csv float repr round-trips.
REPLICATION_FIELDS = ("heldout_mean_log_score", "q_strict", "s2_q_based",
                      "s3_plugin_conditional", "num_params")
SELECTOR_SPACE = {"S1": ("heldout_mean_log_score", +1),
                  "S2": ("s2_q_based", -1),
                  "S3": ("s3_plugin_conditional", -1)}


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


def select_k(means: dict[int, float], tie_tolerance: float) -> int:
    best = max(means.values())
    ties = sorted(k for k, v in means.items() if best - v <= tie_tolerance)
    return min(ties)


# ---- the audit ------------------------------------------------------------

def audit(run_dir: Path) -> dict[str, Any]:
    a = Audit()
    present = {p.name for p in run_dir.iterdir()} if run_dir.is_dir() else set()

    if not a.require(run_dir.is_dir(), "BLOCKER", "run_dir",
                     f"missing: {run_dir}"):
        return finish(a, run_dir)
    for name in ("protocol.json", "runinfo.json", "checkpoint_results.csv",
                 "fit_results.csv", "replication_vs_baseline.json"):
        a.require(name in present, "BLOCKER", "artifact_set", f"missing {name}")
    a.require("failure.json" not in present, "BLOCKER", "failure_marker",
              "failure.json is present: this is not a completed run")
    if a.count("BLOCKER"):
        return finish(a, run_dir)

    protocol_doc = read_json(run_dir / "protocol.json")
    runinfo = read_json(run_dir / "runinfo.json")
    checkpoints = read_csv(run_dir / "checkpoint_results.csv")
    fits = read_csv(run_dir / "fit_results.csv")
    replication_doc = read_json(run_dir / "replication_vs_baseline.json")

    # ---- protocol identity, recomputed from the frozen body ---------------
    body = protocol_doc.get("protocol")
    if a.require(isinstance(body, dict), "BLOCKER", "protocol_body",
                 "protocol.json carries no frozen body"):
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
        recomputed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        a.require(recomputed == protocol_doc.get("protocol_hash"), "BLOCKER",
                  "protocol_hash_recomputed",
                  f"body hashes to {recomputed}, recorded "
                  f"{protocol_doc.get('protocol_hash')}")
        a.require(runinfo.get("protocol_hash") == protocol_doc.get("protocol_hash"),
                  "BLOCKER", "runinfo_protocol_hash", "runinfo disagrees")
    if a.count("BLOCKER"):
        return finish(a, run_dir)

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

    # ---- forbidden adaptive stopping --------------------------------------
    a.require(body.get("adaptive_stopping") is False, "BLOCKER",
              "adaptive_stopping_declared",
              "the protocol does not declare adaptive_stopping false")
    a.require(runinfo.get("adaptive_stopping") is False, "HIGH",
              "adaptive_stopping_runinfo", "runinfo does not declare it false")
    reasons = {r["stopping_reason"] for r in fits}
    a.require(reasons == {"fixed_budget"}, "BLOCKER", "stopping_reason",
              f"observed stopping reasons: {sorted(reasons)}")
    bad_iters = [r["fit_index"] for r in fits
                 if int(r["actual_iterations"]) != max_iter]
    a.require(not bad_iters, "BLOCKER", "actual_iterations",
              f"{len(bad_iters)} fits did not run the full budget: "
              f"{bad_iters[:5]}")

    # ---- counts, density, ladder ------------------------------------------
    expected_fits = len(n_grid) * replicates * len(candidate_k) * len(starts)
    a.require(len(fits) == expected_fits, "BLOCKER", "fit_count",
              f"{len(fits)} fits, expected {expected_fits}")
    a.require(len(checkpoints) == expected_fits * len(ladder), "BLOCKER",
              "checkpoint_row_count",
              f"{len(checkpoints)} rows, expected {expected_fits * len(ladder)}")

    indices = [int(r["fit_index"]) for r in fits]
    a.require(sorted(indices) == list(range(1, expected_fits + 1)), "BLOCKER",
              "fit_index_dense",
              "fit_index is not a dense 1..N without duplicates")

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

    # ---- identity grid -----------------------------------------------------
    seen = {(int(r["k_true"]), int(r["n"]), int(r["replicate"]),
             int(r["k_est"]), int(r["start"])) for r in fits}
    expected_grid = {(k_true, n, rep, k, s)
                     for n in n_grid for rep in range(1, replicates + 1)
                     for k in candidate_k for s in starts}
    a.require(seen == expected_grid, "BLOCKER", "identity_grid",
              f"{len(expected_grid - seen)} missing, {len(seen - expected_grid)} "
              f"unexpected")

    # ---- seeds and masks, recomputed --------------------------------------
    seed_problems: list[str] = []
    for row in fits:
        n, rep = int(row["n"]), int(row["replicate"])
        want_data = data_seed(int(body["data_seed_base"]), k_true, n, rep)
        want_split = split_seed(int(body["split_seed_base"]), k_true, n, rep)
        want_model = model_seed(int(body["model_seed_base"]), k_true, n, rep,
                                int(row["k_est"]), int(row["start"]))
        if (int(row["data_seed"]) != want_data
                or int(row["split_seed"]) != want_split
                or int(row["model_seed"]) != want_model):
            seed_problems.append(row["fit_index"])
    a.require(not seed_problems, "BLOCKER", "seed_formulas",
              f"{len(seed_problems)} fits have seeds that do not match the "
              f"protocol formulas: {seed_problems[:5]}")

    mask_problems: list[str] = []
    for n in n_grid:
        for rep in range(1, replicates + 1):
            train, test = build_masks(
                n, split_seed(int(body["split_seed_base"]), k_true, n, rep),
                test_ratio)
            want = (stable_hash(train), stable_hash(test))
            for row in fits:
                if int(row["n"]) == n and int(row["replicate"]) == rep:
                    if (row["train_mask_hash"], row["test_mask_hash"]) != want:
                        mask_problems.append(row["fit_index"])
    a.require(not mask_problems, "BLOCKER", "mask_hashes",
              f"{len(mask_problems)} fits carry masks that do not rebuild from "
              f"the split seed: {mask_problems[:5]}")

    # ---- finiteness --------------------------------------------------------
    nonfinite = [r["fit_index"] for r in checkpoints
                 for field in REPLICATION_FIELDS
                 if not math.isfinite(float(r[field]))]
    a.require(not nonfinite, "BLOCKER", "nonfinite_criteria",
              f"{len(nonfinite)} checkpoint rows carry a non-finite criterion")

    # ---- S2 identity -------------------------------------------------------
    s2_problems: list[str] = []
    for row in checkpoints:
        n = int(row["n"])
        want = -2.0 * float(row["q_strict"]) + int(row["num_params"]) * math.log(n)
        if not math.isclose(want, float(row["s2_q_based"]),
                            rel_tol=1e-9, abs_tol=1e-6):
            s2_problems.append(row["fit_index"])
    a.require(not s2_problems, "HIGH", "s2_identity",
              f"{len(s2_problems)} rows violate s2 = -2*q_strict + p*log(n): "
              f"{s2_problems[:5]}")

    # ---- num_params depends only on K -------------------------------------
    per_k: dict[int, set[int]] = {}
    for row in checkpoints:
        per_k.setdefault(int(row["k_est"]), set()).add(int(row["num_params"]))
    inconsistent = {k: sorted(v) for k, v in per_k.items() if len(v) != 1}
    a.require(not inconsistent, "HIGH", "num_params_rule",
              f"num_params is not a function of K alone: {inconsistent}")

    # ---- retry / NaN metadata ---------------------------------------------
    retried = [r["fit_index"] for r in fits if int(r["retry_count"]) != 0]
    a.require(not retried, "BLOCKER", "retry_count",
              f"{len(retried)} fits recorded a retry; a restart re-seeds and "
              f"voids pairing with the baseline: {retried[:5]}")
    nan_flagged = [r["fit_index"] for r in fits
                   if r["nan_detected"].lower() == "true"]
    a.require(not nan_flagged, "HIGH", "nan_detected",
              f"{len(nan_flagged)} fits detected a NaN latent sample")
    event_problems = [r["fit_index"] for r in fits
                      if len(json.loads(r["retry_events"])) != int(r["retry_count"])]
    a.require(not event_problems, "HIGH", "retry_event_consistency",
              f"{len(event_problems)} fits have retry_events that disagree with "
              f"retry_count: {event_problems[:5]}")
    attempt_problems = [r["fit_index"] for r in fits
                        if int(r["attempts_used"]) != 1]
    a.require(not attempt_problems, "HIGH", "attempts_used",
              f"{len(attempt_problems)} fits used more than one attempt")
    checkpoint_attempts = {int(r["attempt_index"]) for r in checkpoints}
    a.require(checkpoint_attempts <= {0}, "HIGH", "checkpoint_attempt_index",
              f"checkpoint rows come from attempts {sorted(checkpoint_attempts)}")

    # ---- replication against the archived baseline ------------------------
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
        a.require(compared == len(fits), "HIGH", "replication_coverage",
                  f"compared {compared} of {len(fits)} fits at checkpoint "
                  f"{baseline_checkpoint}")
        a.require(not mismatches, "BLOCKER", "replication_vs_baseline",
                  f"{len(mismatches)} checkpoint-{baseline_checkpoint} values "
                  f"differ from the archive: {mismatches[:5]}")
        a.require(replication_doc.get("match") is True, "HIGH",
                  "replication_document",
                  "replication_vs_baseline.json does not record a match")
        a.require(int(replication_doc.get("compared_rows", -1)) == compared,
                  "MEDIUM", "replication_document_count",
                  f"document says {replication_doc.get('compared_rows')}, "
                  f"recomputed {compared}")

    # ---- selected K, recomputed from the raw rows -------------------------
    selection: dict[tuple[str, int, int, int], int] = {}
    for checkpoint in ladder:
        for n in n_grid:
            for rep in range(1, replicates + 1):
                rows = [r for r in checkpoints
                        if int(r["checkpoint_iteration"]) == checkpoint
                        and int(r["n"]) == n and int(r["replicate"]) == rep]
                for name, (field, sign) in SELECTOR_SPACE.items():
                    means: dict[int, float] = {}
                    for k in candidate_k:
                        values = [sign * float(r[field]) for r in rows
                                  if int(r["k_est"]) == k]
                        if len(values) != len(starts):
                            a.note("BLOCKER", "selector_inputs",
                                   f"{name} checkpoint {checkpoint} n={n} "
                                   f"rep={rep} K={k}: {len(values)} starts")
                            values = values or [float("nan")]
                        means[k] = float(np.mean(np.asarray(values,
                                                            dtype=np.float64),
                                                 dtype=np.float64))
                    selection[(name, checkpoint, n, rep)] = select_k(
                        means, tie_tolerance)
    a.require(len(selection) == len(SELECTOR_SPACE) * len(ladder) * len(n_grid)
              * replicates, "HIGH", "selection_grid",
              f"{len(selection)} selections recomputed")

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
                    for r in sorted(rows, key=lambda x: int(x["checkpoint_iteration"]))]
                != sorted(float(r["cumulative_runtime_s"]) for r in rows)]
    a.require(not monotone, "MEDIUM", "cumulative_runtime_monotone",
              f"{len(monotone)} fits have non-monotone cumulative runtime")

    # ---- expected LOW ------------------------------------------------------
    a.note("LOW", "diagnostics_are_descriptive",
           "gram/f/w/w0 change columns are budget-stability diagnostics under "
           "a fixed L Monte-Carlo draw; they are not a convergence proof and "
           "none of them is used for stopping or for selection")

    return finish(a, run_dir, selection)


def finish(a: Audit, run_dir: Path,
           selection: dict[Any, int] | None = None) -> dict[str, Any]:
    verdict = "PASS" if a.count("BLOCKER") == 0 and a.count("HIGH") == 0 else "FAIL"
    report = {
        "audit_version": "em-convergence-audit-v1",
        "run_dir": str(run_dir),
        "verdict": verdict,
        "blocker_count": a.count("BLOCKER"),
        "high_count": a.count("HIGH"),
        "medium_count": a.count("MEDIUM"),
        "low_count": a.count("LOW"),
        "findings": a.findings,
        "recomputed_selections": len(selection) if selection else 0,
    }
    if run_dir.is_dir():
        (run_dir / "audit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    report = audit(args.run_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
