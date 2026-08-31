"""Independent self-audit of the Phase 7e full held-out K-selection pilot.

The audit never re-fits and never imports the pilot harness's selector.  It
recomputes every reported quantity from the saved artifacts alone, using the
seed convention and selector rule as written in Issue #43, and reports the
difference against the runtime-generated values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    ROOT / "expfam" / "results" / "k_selection" / "heldout_full_pilot_20260824"
)

REPLICATES = (1, 2, 3)
K_CANDIDATES = (1, 2, 3, 4, 5, 6, 7)
STARTS = (1, 2)
TRUE_K = 3
TIE_TOLERANCE = 1e-12
EXPECTED_FITS = 42

WITHIN_REPLICATE_INVARIANT_COLUMNS = (
    "x_hash",
    "training_y_hash",
    "train_mask_hash",
    "test_mask_hash",
    "fit_provenance_hash",
    "target_topology_hash",
    "score_target_hash",
    "preprocessing_hash",
    "score_config_hash",
)

# The frozen final artifact set.  Completeness is checked fail-closed: a
# missing artifact is a BLOCKER and an empty or truncated table can never
# pass by making a comparison loop iterate zero times.
REQUIRED_ARTIFACT_NAMES = (
    "manifest.csv",
    "fit_results.csv",
    "replicate_selection.csv",
    "aggregate_summary.csv",
    "score_by_k.csv",
    "runinfo.json",
    "runinfo.md",
    "stdout.log",
)

EXPECTED_SELECTION_KEYS = frozenset(
    (replicate, k) for replicate in REPLICATES for k in K_CANDIDATES
)
EXPECTED_SELECTION_ROWS = len(EXPECTED_SELECTION_KEYS)
EXPECTED_SCORE_BY_K_ROWS = len(K_CANDIDATES)
EXPECTED_PILOT_KEYS = (
    "n_replicates",
    "true_k",
    "selected_k_counts",
    "true_k_selected_count",
    "descriptive_recovery_rate",
)
EXPECTED_AGGREGATE_ROWS = len(K_CANDIDATES) + len(EXPECTED_PILOT_KEYS)

# Columns of replicate_selection.csv that describe the replicate as a whole and
# must therefore be identical across that replicate's seven candidate-K rows.
REPLICATE_CONSTANT_SELECTION_COLUMNS = (
    "selected_k",
    "best_mean_score",
    "second_best_mean_score",
    "margin",
    "tie_candidates",
)

REQUIRED_RUNINFO_FIELDS = (
    "issue",
    "branch",
    "run_code_sha",
    "base_main_sha",
    "timestamp_utc_start",
    "timestamp_local_start",
    "timestamp_utc_finish",
    "command",
    "python_version",
    "numpy_version",
    "platform",
    "config",
    "candidate_k",
    "starts",
    "replicates",
    "manifest",
    "data_seeds",
    "split_seeds",
    "model_seeds",
    "per_replicate_provenance",
    "git_status_before",
    "git_status_after_scientific_execution",
    "stdout_log",
    "expected_fit_count",
    "actual_fit_count",
    "targets_created",
    "score_rows",
    "failure_state",
    "generated_artifacts",
    "tie_tolerance",
    "tie_rule",
)


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _expected_manifest() -> list[tuple[int, int, int, int, int, int]]:
    """Rebuild the frozen manifest from the written seed convention only."""

    rows: list[tuple[int, int, int, int, int, int]] = []
    for replicate in REPLICATES:
        for k in K_CANDIDATES:
            for start in STARTS:
                rows.append(
                    (
                        replicate,
                        k,
                        start,
                        41000 + replicate,
                        42000 + replicate,
                        43000 + replicate * 1000 + k * 10 + start,
                    )
                )
    return rows


def _select_k(mean_scores: dict[int, float]) -> tuple[int, tuple[int, ...]]:
    best = max(mean_scores.values())
    ties = tuple(
        sorted(k for k, score in mean_scores.items() if best - score <= TIE_TOLERANCE)
    )
    return min(ties), ties


def _sample_std(values: Sequence[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))


def _verdict_for(findings: Sequence[dict[str, str]]) -> str:
    severities = {finding["severity"] for finding in findings}
    if "BLOCKER" in severities or "HIGH" in severities:
        return "FAIL"
    if "MEDIUM" in severities:
        return "PASS_WITH_NOTES"
    return "PASS"


def _incomplete_result(
    run_dir: Path,
    findings: list[dict[str, str]],
    present_files: Sequence[str],
    missing_required: Sequence[str],
) -> dict[str, Any]:
    """Result shape used when the artifact set is too incomplete to recompute.

    Returning early here is what makes the audit fail *closed*: without it a
    missing table would raise, or an empty table would let every comparison
    loop iterate zero times and report PASS.
    """

    return {
        "run_dir": _relative_to_root(run_dir),
        "verdict": _verdict_for(findings),
        "artifacts_present": list(present_files),
        "missing_required_artifacts": list(missing_required),
        "unexpected_artifacts": sorted(
            set(present_files) - set(REQUIRED_ARTIFACT_NAMES)
        ),
        "fit_rows": 0,
        "manifest_rows": 0,
        "duplicate_keys": 0,
        "missing_keys": [],
        "total_retries": 0,
        "total_warnings": 0,
        "total_q_failures": 0,
        "total_nan": 0,
        "independent_means": {},
        "independent_selected_k": {},
        "independent_tie_candidates": {},
        "independent_margin": {},
        "independent_selected_k_counts": {},
        "independent_true_k_selected_count": 0,
        "independent_descriptive_recovery_rate": 0.0,
        "independent_k_stats": {},
        "max_mean_score_difference": 0.0,
        "max_aggregate_difference": 0.0,
        "blocker": sum(1 for f in findings if f["severity"] == "BLOCKER"),
        "high": sum(1 for f in findings if f["severity"] == "HIGH"),
        "medium": sum(1 for f in findings if f["severity"] == "MEDIUM"),
        "low": sum(1 for f in findings if f["severity"] == "LOW"),
        "findings": findings,
    }


def audit(run_dir: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    def record(severity: str, check: str, detail: str) -> None:
        findings.append({"severity": severity, "check": check, "detail": detail})

    # --- required artifact completeness (fail-closed, before any read) ----
    if run_dir.is_dir():
        present_files = sorted(path.name for path in run_dir.iterdir() if path.is_file())
    else:
        present_files = []
        record("BLOCKER", "run_dir_missing", _relative_to_root(run_dir))
    missing_required = [
        name for name in REQUIRED_ARTIFACT_NAMES if name not in present_files
    ]
    for name in missing_required:
        record("BLOCKER", "required_artifact_missing", name)
    unexpected_artifacts = sorted(set(present_files) - set(REQUIRED_ARTIFACT_NAMES))
    if unexpected_artifacts:
        record("HIGH", "unexpected_artifact", f"{unexpected_artifacts}")
    if missing_required:
        return _incomplete_result(run_dir, findings, present_files, missing_required)

    manifest_rows = _read_csv(run_dir / "manifest.csv")
    fit_rows = _read_csv(run_dir / "fit_results.csv")
    selection_rows = _read_csv(run_dir / "replicate_selection.csv")
    aggregate_rows = _read_csv(run_dir / "aggregate_summary.csv")
    score_by_k_rows = _read_csv(run_dir / "score_by_k.csv")
    runinfo = json.loads((run_dir / "runinfo.json").read_text(encoding="utf-8"))

    expected_manifest = _expected_manifest()

    # --- manifest ---------------------------------------------------------
    actual_manifest = [
        (
            int(row["replicate"]),
            int(row["K"]),
            int(row["start"]),
            int(row["data_seed"]),
            int(row["split_seed"]),
            int(row["model_seed"]),
        )
        for row in manifest_rows
    ]
    if len(actual_manifest) != EXPECTED_FITS:
        record("BLOCKER", "manifest_row_count", f"{len(actual_manifest)} != 42")
    if actual_manifest != expected_manifest:
        record("BLOCKER", "manifest_exact_match", "manifest differs from convention")

    # --- fit rows ---------------------------------------------------------
    if len(fit_rows) != EXPECTED_FITS:
        record("BLOCKER", "fit_row_count", f"{len(fit_rows)} != 42")

    fit_keys = [(int(row["replicate"]), int(row["K"]), int(row["start"])) for row in fit_rows]
    if len(fit_keys) != len(set(fit_keys)):
        record("BLOCKER", "fit_duplicate_key", "duplicate (replicate,K,start)")
    expected_keys = {(row[0], row[1], row[2]) for row in expected_manifest}
    if set(fit_keys) != expected_keys:
        missing = sorted(expected_keys - set(fit_keys))
        extra = sorted(set(fit_keys) - expected_keys)
        record("BLOCKER", "fit_key_set", f"missing={missing} extra={extra}")

    for row in fit_rows:
        key = (int(row["replicate"]), int(row["K"]), int(row["start"]))
        expected_seeds = {
            (item[0], item[1], item[2]): (item[3], item[4], item[5])
            for item in expected_manifest
        }
        if key in expected_seeds:
            actual = (int(row["data_seed"]), int(row["split_seed"]), int(row["model_seed"]))
            if actual != expected_seeds[key]:
                record("BLOCKER", "fit_seed", f"{key}: {actual} != {expected_seeds[key]}")
        if row["fit_status"] != "clean":
            record("BLOCKER", "fit_status", f"{key}: {row['fit_status']}")
        if int(row["retry"]) != 0:
            record("BLOCKER", "retry", f"{key}: retry={row['retry']}")
        if int(row["warning_count"]) != 0 or row["warnings"]:
            record("BLOCKER", "warnings", f"{key}: {row['warnings']!r}")
        if row["q_failure"] != "False":
            record("BLOCKER", "q_failure", f"{key}: {row['q_failure']}")
        if row["nan_occurred"] != "False":
            record("BLOCKER", "nan_occurred", f"{key}: {row['nan_occurred']}")
        if row["finite_state"] != "True":
            record("BLOCKER", "finite_state", f"{key}: {row['finite_state']}")
        for column in ("heldout_mean_log_score", "Q_strict"):
            value = float(row[column])
            if not math.isfinite(value):
                record("BLOCKER", f"finite_{column}", f"{key}: {value}")

    # --- within-replicate provenance --------------------------------------
    for replicate in REPLICATES:
        replicate_rows = [row for row in fit_rows if int(row["replicate"]) == replicate]
        if len(replicate_rows) != 14:
            record("BLOCKER", "replicate_row_count", f"r{replicate}: {len(replicate_rows)}")
            continue
        for column in WITHIN_REPLICATE_INVARIANT_COLUMNS:
            distinct = {row[column] for row in replicate_rows}
            if len(distinct) != 1:
                record(
                    "BLOCKER",
                    "within_replicate_invariance",
                    f"r{replicate}.{column}: {len(distinct)} distinct values",
                )
        if len({row["fit_config_hash"] for row in replicate_rows}) != 14:
            record(
                "MEDIUM",
                "fit_config_uniqueness",
                f"r{replicate}: fit_config_hash is not unique per fit",
            )
        for k in K_CANDIDATES:
            starts = sorted(int(row["start"]) for row in replicate_rows if int(row["K"]) == k)
            if starts != [1, 2]:
                record("BLOCKER", "starts_per_k", f"r{replicate} K={k}: {starts}")

    if len({row["score_target_hash"] for row in fit_rows}) != len(REPLICATES):
        record(
            "MEDIUM",
            "score_target_distinctness",
            "score_target_hash is not distinct per replicate",
        )

    # --- independent selector ---------------------------------------------
    independent_means: dict[int, dict[int, float]] = {}
    independent_selection: dict[int, dict[str, Any]] = {}
    for replicate in REPLICATES:
        means: dict[int, float] = {}
        for k in K_CANDIDATES:
            scores = [
                float(row["heldout_mean_log_score"])
                for row in fit_rows
                if int(row["replicate"]) == replicate and int(row["K"]) == k
            ]
            if len(scores) != 2:
                record("BLOCKER", "score_pair", f"r{replicate} K={k}: {len(scores)} scores")
                continue
            means[k] = (scores[0] + scores[1]) / 2
        if len(means) != len(K_CANDIDATES):
            continue
        independent_means[replicate] = means
        selected, ties = _select_k(means)
        ordered = sorted(means.values(), reverse=True)
        independent_selection[replicate] = {
            "selected_k": selected,
            "tie_candidates": ties,
            "best": ordered[0],
            "second_best": ordered[1],
            "margin": ordered[0] - ordered[1],
        }

    # --- replicate_selection structure (fail-closed) ----------------------
    # Without these checks a header-only or truncated file would make the
    # comparison loop below iterate zero times and silently PASS.
    if len(selection_rows) != EXPECTED_SELECTION_ROWS:
        record(
            "BLOCKER",
            "selection_row_count",
            f"{len(selection_rows)} != {EXPECTED_SELECTION_ROWS}",
        )
    selection_keys = [
        (int(row["replicate"]), int(row["K"])) for row in selection_rows
    ]
    duplicate_selection_keys = sorted(
        {key for key in selection_keys if selection_keys.count(key) > 1}
    )
    if duplicate_selection_keys:
        record(
            "BLOCKER",
            "selection_duplicate_key",
            f"duplicate (replicate,K): {duplicate_selection_keys}",
        )
    if set(selection_keys) != EXPECTED_SELECTION_KEYS:
        missing_selection = sorted(EXPECTED_SELECTION_KEYS - set(selection_keys))
        extra_selection = sorted(set(selection_keys) - EXPECTED_SELECTION_KEYS)
        record(
            "BLOCKER",
            "selection_key_set",
            f"missing={missing_selection} extra={extra_selection}",
        )
    for replicate in REPLICATES:
        replicate_selection_rows = [
            row for row in selection_rows if int(row["replicate"]) == replicate
        ]
        if len(replicate_selection_rows) != len(K_CANDIDATES):
            record(
                "BLOCKER",
                "selection_rows_per_replicate",
                f"r{replicate}: {len(replicate_selection_rows)} != {len(K_CANDIDATES)}",
            )
            continue
        for column in REPLICATE_CONSTANT_SELECTION_COLUMNS:
            distinct = {row[column] for row in replicate_selection_rows}
            if len(distinct) != 1:
                record(
                    "BLOCKER",
                    "selection_replicate_consistency",
                    f"r{replicate}.{column}: {sorted(distinct)}",
                )

    # --- compare against replicate_selection.csv --------------------------
    max_mean_diff = 0.0
    for row in selection_rows:
        replicate = int(row["replicate"])
        k = int(row["K"])
        reported = float(row["mean_score"])
        start_mean = (float(row["start1_score"]) + float(row["start2_score"])) / 2
        if reported != start_mean:
            record(
                "BLOCKER",
                "reported_mean_arithmetic",
                f"r{replicate} K={k}: {reported!r} != {start_mean!r}",
            )
        recomputed = independent_means.get(replicate, {}).get(k)
        if recomputed is None:
            continue
        diff = abs(recomputed - reported)
        max_mean_diff = max(max_mean_diff, diff)
        if diff != 0.0:
            record(
                "HIGH",
                "mean_recomputation",
                f"r{replicate} K={k}: diff={diff!r}",
            )
        reported_selected = int(row["selected_k"])
        if reported_selected != independent_selection[replicate]["selected_k"]:
            record(
                "BLOCKER",
                "selected_k",
                f"r{replicate}: reported {reported_selected} vs independent "
                f"{independent_selection[replicate]['selected_k']}",
            )
        reported_ties = tuple(
            int(item) for item in row["tie_candidates"].split("|") if item
        )
        if reported_ties != independent_selection[replicate]["tie_candidates"]:
            record(
                "BLOCKER",
                "tie_candidates",
                f"r{replicate}: {reported_ties} vs "
                f"{independent_selection[replicate]['tie_candidates']}",
            )
        for column, key in (
            ("best_mean_score", "best"),
            ("second_best_mean_score", "second_best"),
            ("margin", "margin"),
        ):
            reported_value = float(row[column])
            expected_value = independent_selection[replicate][key]
            if abs(reported_value - expected_value) > 1e-15:
                record(
                    "HIGH",
                    f"replicate_{column}",
                    f"r{replicate}: {reported_value!r} vs {expected_value!r}",
                )

    # --- aggregate --------------------------------------------------------
    independent_counts: dict[int, int] = {}
    for replicate in REPLICATES:
        selected = independent_selection.get(replicate, {}).get("selected_k")
        if selected is not None:
            independent_counts[selected] = independent_counts.get(selected, 0) + 1
    independent_true_k_count = independent_counts.get(TRUE_K, 0)
    independent_recovery = independent_true_k_count / len(REPLICATES)

    independent_k_stats: dict[int, dict[str, float]] = {}
    for k in K_CANDIDATES:
        values = [independent_means[r][k] for r in REPLICATES if r in independent_means]
        if len(values) != len(REPLICATES):
            record("BLOCKER", "k_aggregate_coverage", f"K={k}: {len(values)} replicates")
            continue
        independent_k_stats[k] = {
            "mean": sum(values) / len(values),
            "std": _sample_std(values),
            "min": min(values),
            "max": max(values),
        }

    # --- aggregate_summary structure (fail-closed) ------------------------
    if len(aggregate_rows) != EXPECTED_AGGREGATE_ROWS:
        record(
            "BLOCKER",
            "aggregate_row_count",
            f"{len(aggregate_rows)} != {EXPECTED_AGGREGATE_ROWS}",
        )
    k_wise_keys = [
        int(row["K"]) for row in aggregate_rows if row["section"] == "k_wise"
    ]
    if len(k_wise_keys) != len(K_CANDIDATES):
        record(
            "BLOCKER",
            "aggregate_k_wise_row_count",
            f"{len(k_wise_keys)} != {len(K_CANDIDATES)}",
        )
    duplicate_k_wise = sorted({k for k in k_wise_keys if k_wise_keys.count(k) > 1})
    if duplicate_k_wise:
        record("BLOCKER", "aggregate_duplicate_k", f"{duplicate_k_wise}")
    if set(k_wise_keys) != set(K_CANDIDATES):
        record(
            "BLOCKER",
            "aggregate_k_wise_key_set",
            f"missing={sorted(set(K_CANDIDATES) - set(k_wise_keys))} "
            f"extra={sorted(set(k_wise_keys) - set(K_CANDIDATES))}",
        )
    pilot_keys = [row["key"] for row in aggregate_rows if row["section"] == "pilot"]
    duplicate_pilot = sorted({key for key in pilot_keys if pilot_keys.count(key) > 1})
    if duplicate_pilot:
        record("BLOCKER", "aggregate_duplicate_pilot_key", f"{duplicate_pilot}")
    if set(pilot_keys) != set(EXPECTED_PILOT_KEYS):
        record(
            "BLOCKER",
            "aggregate_pilot_key_set",
            f"missing={sorted(set(EXPECTED_PILOT_KEYS) - set(pilot_keys))} "
            f"extra={sorted(set(pilot_keys) - set(EXPECTED_PILOT_KEYS))}",
        )
    unexpected_sections = sorted(
        {row["section"] for row in aggregate_rows} - {"k_wise", "pilot"}
    )
    if unexpected_sections:
        record("BLOCKER", "aggregate_unexpected_section", f"{unexpected_sections}")

    max_aggregate_diff = 0.0
    for row in aggregate_rows:
        if row["section"] != "k_wise":
            continue
        k = int(row["K"])
        stats = independent_k_stats.get(k)
        if stats is None:
            continue
        for column, key in (
            ("mean_across_replicates", "mean"),
            ("std_across_replicates", "std"),
            ("min_across_replicates", "min"),
            ("max_across_replicates", "max"),
        ):
            diff = abs(float(row[column]) - stats[key])
            max_aggregate_diff = max(max_aggregate_diff, diff)
            if diff > 1e-12:
                record("HIGH", f"k_aggregate_{key}", f"K={k}: diff={diff!r}")

    pilot = {row["key"]: row["value"] for row in aggregate_rows if row["section"] == "pilot"}
    if int(pilot.get("n_replicates", -1)) != len(REPLICATES):
        record("BLOCKER", "n_replicates", pilot.get("n_replicates", "<missing>"))
    if int(pilot.get("true_k", -1)) != TRUE_K:
        record("BLOCKER", "true_k", pilot.get("true_k", "<missing>"))
    reported_counts = {}
    for item in pilot.get("selected_k_counts", "").split("|"):
        if item:
            k_text, count_text = item.split(":")
            reported_counts[int(k_text)] = int(count_text)
    if reported_counts != independent_counts:
        record(
            "BLOCKER",
            "selected_k_counts",
            f"{reported_counts} vs independent {independent_counts}",
        )
    if int(pilot.get("true_k_selected_count", -1)) != independent_true_k_count:
        record(
            "BLOCKER",
            "true_k_selected_count",
            f"{pilot.get('true_k_selected_count')} vs {independent_true_k_count}",
        )
    # NOTE: a missing key previously produced float("nan"), and every NaN
    # comparison is False, so the check passed silently.  Presence and
    # finiteness are now asserted before the value is compared.
    reported_recovery_text = pilot.get("descriptive_recovery_rate")
    if reported_recovery_text is None:
        record("BLOCKER", "descriptive_recovery_rate", "<missing>")
    else:
        reported_recovery = float(reported_recovery_text)
        if not math.isfinite(reported_recovery):
            record(
                "BLOCKER",
                "descriptive_recovery_rate_finite",
                f"{reported_recovery_text!r}",
            )
        elif abs(reported_recovery - independent_recovery) > 0.0:
            record(
                "BLOCKER",
                "descriptive_recovery_rate",
                f"{reported_recovery_text} vs {independent_recovery}",
            )

    # --- score_by_k structure (fail-closed) -------------------------------
    if len(score_by_k_rows) != EXPECTED_SCORE_BY_K_ROWS:
        record(
            "BLOCKER",
            "score_by_k_row_count",
            f"{len(score_by_k_rows)} != {EXPECTED_SCORE_BY_K_ROWS}",
        )
    score_by_k_keys = [int(row["K"]) for row in score_by_k_rows]
    duplicate_score_k = sorted(
        {k for k in score_by_k_keys if score_by_k_keys.count(k) > 1}
    )
    if duplicate_score_k:
        record("BLOCKER", "score_by_k_duplicate_k", f"{duplicate_score_k}")
    if set(score_by_k_keys) != set(K_CANDIDATES):
        record(
            "BLOCKER",
            "score_by_k_key_set",
            f"missing={sorted(set(K_CANDIDATES) - set(score_by_k_keys))} "
            f"extra={sorted(set(score_by_k_keys) - set(K_CANDIDATES))}",
        )

    for row in score_by_k_rows:
        k = int(row["K"])
        for replicate in REPLICATES:
            reported = float(row[f"replicate_{replicate}_mean"])
            expected = independent_means.get(replicate, {}).get(k)
            if expected is not None and abs(reported - expected) > 0.0:
                record("HIGH", "score_by_k", f"K={k} r{replicate}: {reported!r} vs {expected!r}")

    # --- runinfo ----------------------------------------------------------
    for field in REQUIRED_RUNINFO_FIELDS:
        if field not in runinfo:
            record("BLOCKER", "runinfo_missing_field", field)
    # The saved runinfo declares which artifacts the run produced.  That
    # declaration must agree with the files actually present.  The *method*
    # that captured stdout.log is a separate, unresolved provenance question
    # and is deliberately not inferred here.
    declared_artifacts = runinfo.get("generated_artifacts")
    if not isinstance(declared_artifacts, list):
        record("BLOCKER", "runinfo_generated_artifacts", f"{declared_artifacts!r}")
    else:
        declared_missing = sorted(set(declared_artifacts) - set(present_files))
        undeclared_present = sorted(set(present_files) - set(declared_artifacts))
        if declared_missing:
            record(
                "BLOCKER",
                "runinfo_declared_artifact_absent",
                f"{declared_missing}",
            )
        if undeclared_present:
            record(
                "HIGH",
                "runinfo_undeclared_artifact_present",
                f"{undeclared_present}",
            )
    if len(independent_means) != len(REPLICATES):
        record(
            "BLOCKER",
            "independent_recomputation_coverage",
            f"{sorted(independent_means)} != {list(REPLICATES)}",
        )
    if runinfo.get("expected_fit_count") != EXPECTED_FITS:
        record("BLOCKER", "runinfo_expected_fits", str(runinfo.get("expected_fit_count")))
    if runinfo.get("actual_fit_count") != EXPECTED_FITS:
        record("BLOCKER", "runinfo_actual_fits", str(runinfo.get("actual_fit_count")))
    if runinfo.get("failure_state") != "none":
        record("BLOCKER", "runinfo_failure_state", str(runinfo.get("failure_state")))
    if len(runinfo.get("manifest", [])) != EXPECTED_FITS:
        record("BLOCKER", "runinfo_manifest", str(len(runinfo.get("manifest", []))))
    if runinfo.get("targets_created") != len(REPLICATES):
        record("BLOCKER", "runinfo_targets", str(runinfo.get("targets_created")))
    if runinfo.get("score_rows") != EXPECTED_FITS:
        record("BLOCKER", "runinfo_score_rows", str(runinfo.get("score_rows")))
    runinfo_manifest = [
        (
            item["replicate"],
            item["K"],
            item["start"],
            item["data_seed"],
            item["split_seed"],
            item["model_seed"],
        )
        for item in runinfo.get("manifest", [])
    ]
    if runinfo_manifest != expected_manifest:
        record("BLOCKER", "runinfo_manifest_exact", "runinfo manifest differs from convention")

    return {
        "run_dir": _relative_to_root(run_dir),
        "verdict": _verdict_for(findings),
        "artifacts_present": list(present_files),
        "missing_required_artifacts": list(missing_required),
        "unexpected_artifacts": list(unexpected_artifacts),
        "fit_rows": len(fit_rows),
        "manifest_rows": len(manifest_rows),
        "duplicate_keys": len(fit_keys) - len(set(fit_keys)),
        "missing_keys": sorted(expected_keys - set(fit_keys)),
        "total_retries": sum(int(row["retry"]) for row in fit_rows),
        "total_warnings": sum(int(row["warning_count"]) for row in fit_rows),
        "total_q_failures": sum(1 for row in fit_rows if row["q_failure"] != "False"),
        "total_nan": sum(1 for row in fit_rows if row["nan_occurred"] != "False"),
        "independent_means": {
            str(replicate): {str(k): means[k] for k in K_CANDIDATES}
            for replicate, means in independent_means.items()
        },
        "independent_selected_k": {
            str(replicate): value["selected_k"]
            for replicate, value in independent_selection.items()
        },
        "independent_tie_candidates": {
            str(replicate): list(value["tie_candidates"])
            for replicate, value in independent_selection.items()
        },
        "independent_margin": {
            str(replicate): value["margin"]
            for replicate, value in independent_selection.items()
        },
        "independent_selected_k_counts": {
            str(k): count for k, count in sorted(independent_counts.items())
        },
        "independent_true_k_selected_count": independent_true_k_count,
        "independent_descriptive_recovery_rate": independent_recovery,
        "independent_k_stats": {
            str(k): value for k, value in independent_k_stats.items()
        },
        "max_mean_score_difference": max_mean_diff,
        "max_aggregate_difference": max_aggregate_diff,
        "blocker": sum(1 for f in findings if f["severity"] == "BLOCKER"),
        "high": sum(1 for f in findings if f["severity"] == "HIGH"),
        "medium": sum(1 for f in findings if f["severity"] == "MEDIUM"),
        "low": sum(1 for f in findings if f["severity"] == "LOW"),
        "findings": findings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args(argv)
    result = audit(Path(args.run_dir))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
