"""Independent artifact-only audit of the Phase 8b K_TRUE robustness sweep.

Same philosophy as ``audit_heldout_full_pilot.py``: this module never re-fits,
never imports the harness, and recomputes every reported quantity from the saved
artifacts alone, using the seed convention, gate contract and selector rule as
written in the Phase 8a design and Issue #49.

**Fail-closed.**  A missing artifact, a header-only CSV, a missing row, a
row-count mismatch, a duplicate key, an unexpected key or a failed gate is a
BLOCKER.  There is no "skip if absent" path: the required artifact set is
declared explicitly per audit mode.

**The selector is recomputed independently.**  ``selected_k`` in
``k_true_selection_matrix.csv`` is never trusted; it is re-derived from the
per-(K, start) held-out scores.  Cross-checking the matrix's own derived columns
against each other is not sufficient, because a consistent tampering of
``selected_k`` + ``signed_error`` + ``abs_error`` + ``label`` would pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]

# --- frozen expectations (independent restatement; NOT imported from harness) --
ESTIMANDS = ("A", "B")
ROLE_BY_ESTIMAND = {"A": "primary", "B": "sensitivity"}
NEW_K_TRUE = (1, 2, 4, 5)
ANCHOR_K_TRUE = 3
FULL_K_TRUE_GRID = (1, 2, 3, 4, 5)
K_CANDIDATES = (1, 2, 3, 4, 5, 6, 7)
REPLICATES = (1, 2, 3)
STARTS = (1, 2)
W_REF = 1.5
K_REF = 3
W0_TRUE = -1.0
MASK_DESIGN = "S_C"
RANDOM_DESIGN = "CRN"
TIE_TOLERANCE = 1e-12

DATA_SEED_BASE = 51000
MODEL_SEED_BASE = 530000
ANCHOR_SPLIT_SEED_BASE = 42000

FITS_PER_ESTIMAND = len(NEW_K_TRUE) * len(REPLICATES) * len(K_CANDIDATES) * len(STARTS)  # 168
CELLS_PER_ESTIMAND = len(NEW_K_TRUE) * len(REPLICATES)  # 12
MATRIX_ROWS_PER_ESTIMAND = len(FULL_K_TRUE_GRID) * len(REPLICATES)  # 15

REQUIRED_MASK_PROVENANCE_FIELDS = (
    "split_mask_hash",
    "train_mask_hash",
    "mask_design",
    "mask_group_id",
    "anchor_mask_hash",
    "anchor_train_mask_hash",
    "intentional_seed_reuse",
)

SELECTION_MATRIX_COLUMNS = (
    "estimand", "role", "K_TRUE", "replicate", "selected_k", "signed_error",
    "abs_error", "label", "lineage", "run_code_sha", "artifact_dir",
)
SELECTION_MATRIX_FORBIDDEN = ("best_score", "margin")

SCORE_COLUMN = "heldout_mean_log_score"

# The exact (gate, scope) evidence the harness contract must emit, rebuilt here
# from the frozen scientific grid.  Checking gate NAMES alone is fail-open: a
# harness could emit one M0 row instead of one per (K_TRUE, replicate) cell and
# still look complete.  Nothing below is imported from the harness.
GATE_SCOPE_COMMON_SCOPE = "common"


def _cell_scopes() -> tuple[str, ...]:
    """Estimand-independent mask cells: 4 new K_TRUE x 3 replicates."""

    return tuple(f"K{k}/r{r}" for k in NEW_K_TRUE for r in REPLICATES)


def _estimand_cell_scopes(estimand: str) -> tuple[str, ...]:
    return tuple(f"{estimand}/K{k}/r{r}" for k in NEW_K_TRUE for r in REPLICATES)


def expected_gate_scope_keys(estimand: str) -> set[tuple[str, str]]:
    """Exact per-estimand (gate, scope) set for one artifact directory.

    Independently derived from the frozen grid:
      generator  A : G1p per new K_TRUE, G3p, G4, G4m, G5, G4c
      generator  B : G1 and G2 per new K_TRUE, G3, G4, G4m, G5, G4c
      mask common  : M2, and M0/M3 per (K_TRUE, replicate) cell
      mask per-est : M1, MC2, MC4, and MC1/MC3/MC5 per (estimand, cell)
    """

    if estimand not in ESTIMANDS:
        raise ValueError(f"unknown estimand {estimand!r}")
    keys: set[tuple[str, str]] = set()

    # --- generator gates ---------------------------------------------------
    if estimand == "B":
        for k in NEW_K_TRUE:
            keys.add(("G1", f"B/K{k}"))
            keys.add(("G2", f"B/K{k}"))
        keys.add(("G3", "B"))
    else:
        for k in NEW_K_TRUE:
            keys.add(("G1p", f"A/K{k}"))
        keys.add(("G3p", "A"))
    keys.add(("G4", estimand))
    keys.add(("G4m", estimand))
    keys.add(("G5", estimand))
    keys.add(("G4c", GATE_SCOPE_COMMON_SCOPE))

    # --- mask gates: estimand-independent ---------------------------------
    keys.add(("M2", GATE_SCOPE_COMMON_SCOPE))
    for scope in _cell_scopes():
        keys.add(("M0", scope))
        keys.add(("M3", scope))

    # --- mask gates: per-estimand (S_C) -----------------------------------
    keys.add(("M1", estimand))
    keys.add(("MC2", estimand))
    keys.add(("MC4", estimand))
    for scope in _estimand_cell_scopes(estimand):
        keys.add(("MC1", scope))
        keys.add(("MC3", scope))
        keys.add(("MC5", scope))
    return keys


def expected_config_gate_count(estimand: str) -> int:
    """Fixed expected count, derived from the grid — never from the CSV length."""

    return len(expected_gate_scope_keys(estimand))


# Frozen expectation for the current config (A+B, S_C, 4 new K_TRUE x 3 reps):
#   A = 73 rows, B = 77 rows per artifact directory.
# The 124 reported by ``--config-gate`` is the CLI aggregate over BOTH
# estimands, where the estimand-independent M0/M2/M3 rows are emitted once;
# it is a different object from a single per-estimand artifact file.
EXPECTED_CONFIG_GATE_COUNT = {"A": 73, "B": 77}
EXPECTED_CLI_AGGREGATE_GATE_COUNT = 124

CANONICAL_TRUE_TOKENS = ("True", "true")

LINEAGE_ANCHOR = "phase7e_anchor"
LINEAGE_NEW = "phase8a_new"
PHASE7E_RUN_CODE_SHA = "b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a"
PHASE7E_ARTIFACT_DIR = "expfam/results/k_selection/heldout_full_pilot_20260824"
PHASE7E_DIR = ROOT / "expfam" / "results" / "k_selection" / "heldout_full_pilot_20260824"

# Explicit per-mode required artifact sets.  Nothing is implicitly optional.
AUDIT_MODES = {
    "config": ("runinfo.json", "config_gate.csv", "manifest.csv",
               "mask_provenance.csv", "diagnostics.csv"),
    "selection": ("runinfo.json", "config_gate.csv", "manifest.csv", "mask_provenance.csv",
                  "diagnostics.csv", "fit_results.csv", "k_true_selection_matrix.csv"),
}

# S2_REQUIRED_FOLLOW_UP: direct Phase8b leakage boundary tests before smoke.
# The fit path is hard-stopped in S1, so A01-A03 leakage falsification is still
# carried by the reused Phase 7e boundary; it must be re-established directly in
# Phase 8b before the smoke gate is opened.


@dataclass(frozen=True, slots=True)
class Finding:
    severity: str
    check: str
    detail: str


class Auditor:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def record(self, severity: str, check: str, detail: str) -> None:
        self.findings.append(Finding(severity, check, detail))

    def blocker(self, check: str, detail: str) -> None:
        self.record("BLOCKER", check, detail)

    def require(self, condition: bool, check: str, detail: str) -> bool:
        if not condition:
            self.blocker(check, detail)
        return bool(condition)

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "BLOCKER"]

    @property
    def highs(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "HIGH"]


# ===========================================================================
# frozen expectations recomputed independently
# ===========================================================================


def expected_w_true(estimand: str, k_true: int) -> float:
    if estimand == "A":
        return W_REF
    if estimand == "B":
        return W_REF * math.sqrt(K_REF / k_true)
    raise ValueError(f"unknown estimand {estimand!r}")


def expected_data_seed(k_true: int, replicate: int) -> int:
    return DATA_SEED_BASE + 100 * int(k_true) + int(replicate)


def expected_model_seed(k_true: int, replicate: int, k: int, start: int) -> int:
    return MODEL_SEED_BASE + 10000 * int(k_true) + 1000 * int(replicate) + 10 * int(k) + int(start)


def expected_split_seed(replicate: int) -> int:
    """S_C: the Phase 7e split seed, intentionally reused.  K_TRUE-independent."""

    return ANCHOR_SPLIT_SEED_BASE + int(replicate)


def expected_label(signed_error: int) -> str:
    if signed_error < 0:
        return "under"
    if signed_error > 0:
        return "over"
    return "exact"


def select_k_independently(scores: Mapping[int, Mapping[int, float]]) -> int:
    """2-start mean -> maximum -> numerical tie <= 1e-12 -> smallest K.

    Reimplemented here on purpose: the audit must not import the harness
    selector it is auditing.
    """

    means: dict[int, float] = {}
    for k, by_start in scores.items():
        if set(by_start) != set(STARTS):
            raise ValueError(f"K={k} does not have exactly the two frozen starts")
        means[int(k)] = sum(by_start[s] for s in STARTS) / len(STARTS)
    if set(means) != set(K_CANDIDATES):
        raise ValueError("candidate K set differs from the frozen set")
    best = max(means.values())
    tied = sorted(k for k, value in means.items() if best - value <= TIE_TOLERANCE)
    return min(tied)


# ===========================================================================
# artifact readers (fail-closed)
# ===========================================================================


def _read_csv(path: Path, auditor: Auditor) -> list[dict[str, str]] | None:
    if not path.is_file():
        auditor.blocker("artifact_missing", str(path))
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            auditor.blocker("artifact_headerless", str(path))
            return None
        rows = list(reader)
    if not rows:
        auditor.blocker("artifact_header_only", str(path))
        return None
    return rows


def read_phase7e_anchor(auditor: Auditor,
                        phase7e_dir: Path | None = None) -> dict[int, tuple[str, str]]:
    directory = PHASE7E_DIR if phase7e_dir is None else Path(phase7e_dir)
    rows = _read_csv(directory / "fit_results.csv", auditor)
    anchors: dict[int, tuple[str, str]] = {}
    if rows is None:
        return anchors
    fieldnames = set(rows[0])
    if not auditor.require("test_mask_hash" in fieldnames and "train_mask_hash" in fieldnames,
                           "anchor_schema", "Phase 7e must expose test_mask_hash and train_mask_hash"):
        return anchors
    auditor.require("split_mask_hash" not in fieldnames, "anchor_schema",
                    "Phase 7e must not expose a split_mask_hash column")
    for row in rows:
        replicate = int(row["replicate"])
        pair = (row["test_mask_hash"], row["train_mask_hash"])
        if replicate in anchors and anchors[replicate] != pair:
            auditor.blocker("anchor_inconsistent", f"replicate {replicate}")
        anchors[replicate] = pair
    auditor.require(set(anchors) == set(REPLICATES), "anchor_replicates",
                    f"expected {set(REPLICATES)}, got {sorted(anchors)}")
    return anchors


def recompute_anchor_selection(auditor: Auditor,
                               phase7e_dir: Path | None = None) -> dict[int, int]:
    """Independently re-derive the K_TRUE=3 selected K from Phase 7e scores."""

    directory = PHASE7E_DIR if phase7e_dir is None else Path(phase7e_dir)
    rows = _read_csv(directory / "fit_results.csv", auditor)
    selected: dict[int, int] = {}
    if rows is None:
        return selected
    if not auditor.require(SCORE_COLUMN in rows[0], "anchor_score_column",
                           f"Phase 7e must expose {SCORE_COLUMN}"):
        return selected
    grouped: dict[int, dict[int, dict[int, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["replicate"]), {}) \
               .setdefault(int(row["K"]), {})[int(row["start"])] = float(row[SCORE_COLUMN])
    for replicate, scores in grouped.items():
        try:
            selected[replicate] = select_k_independently(scores)
        except ValueError as exc:
            auditor.blocker("anchor_selector", f"replicate {replicate}: {exc}")
    return selected


# ===========================================================================
# config gate audit (content, not just presence)
# ===========================================================================


def _is_true(value: str) -> bool:
    """Canonical representation only.  An unknown status is a failure."""

    return str(value) in CANONICAL_TRUE_TOKENS


def _is_recognised_status(value: str) -> bool:
    return str(value) in CANONICAL_TRUE_TOKENS + ("False", "false")


def audit_config_gate(rows: Sequence[dict[str, str]], estimand: str, auditor: Auditor,
                      runinfo: Mapping[str, Any] | None) -> None:
    header = set(rows[0])
    if not auditor.require({"gate", "scope", "passed"} <= header, "config_gate_schema",
                           f"config_gate.csv needs gate/scope/passed columns, got {sorted(header)}"):
        return

    expected_keys = expected_gate_scope_keys(estimand)
    expected_count = expected_config_gate_count(estimand)

    actual_keys: list[tuple[str, str]] = []
    for row in rows:
        actual_keys.append((row["gate"], row["scope"]))
        status = row["passed"]
        auditor.require(_is_recognised_status(status), "config_gate_unknown_status",
                        f"gate {row['gate']}/{row['scope']}: unrecognised status {status!r}")
        auditor.require(_is_true(status), "config_gate_failed",
                        f"gate {row['gate']} ({row['scope']}) is not PASS: {row.get('detail', '')}")
        if "failure_reason" in row:
            auditor.require(not str(row["failure_reason"]).strip(), "config_gate_failure_reason",
                            f"gate {row['gate']} carries a failure reason but is marked PASS")

    # --- exact (gate, scope) set equality, not just gate names -------------
    duplicates = sorted({key for key in actual_keys if actual_keys.count(key) > 1})
    auditor.require(not duplicates, "config_gate_duplicate", f"duplicate (gate,scope): {duplicates}")

    actual_set = set(actual_keys)
    missing = sorted(expected_keys - actual_set)
    unexpected = sorted(actual_set - expected_keys)
    auditor.require(not missing, "config_gate_missing_scope",
                    f"{len(missing)} missing (gate,scope) rows, e.g. {missing[:5]}")
    auditor.require(not unexpected, "config_gate_unexpected_scope",
                    f"{len(unexpected)} unexpected (gate,scope) rows, e.g. {unexpected[:5]}")
    auditor.require(actual_set == expected_keys, "config_gate_set_equality",
                    "actual (gate,scope) set differs from the independently built expected set")

    # --- fixed expected count, derived from the grid, never from the CSV ---
    auditor.require(len(rows) == expected_count, "config_gate_row_count",
                    f"{len(rows)} rows != expected {expected_count} for estimand {estimand}")
    auditor.require(expected_count == EXPECTED_CONFIG_GATE_COUNT[estimand],
                    "config_gate_frozen_count",
                    f"grid-derived count {expected_count} != frozen "
                    f"{EXPECTED_CONFIG_GATE_COUNT[estimand]} for estimand {estimand}")

    # --- runinfo cross-checks ---------------------------------------------
    if runinfo is None:
        auditor.blocker("runinfo_unavailable", "config gate audit requires runinfo.json")
        return
    if auditor.require("em_fits_executed" in runinfo, "runinfo_em_field",
                       "runinfo must record em_fits_executed"):
        value = runinfo["em_fits_executed"]
        if auditor.require(isinstance(value, int) and not isinstance(value, bool),
                           "runinfo_em_type", f"em_fits_executed must be an int, got {type(value).__name__}"):
            auditor.require(value == 0, "runinfo_em_fits", f"em_fits_executed != 0: {value}")
    if auditor.require("gate_count" in runinfo, "runinfo_gate_count_field",
                       "runinfo must record gate_count"):
        declared = runinfo["gate_count"]
        if auditor.require(isinstance(declared, int) and not isinstance(declared, bool),
                           "runinfo_gate_count_type",
                           f"gate_count must be an int, got {type(declared).__name__}"):
            auditor.require(declared == expected_count, "config_gate_count",
                            f"declared gate_count {declared} != expected {expected_count}")


def read_runinfo(run_dir: Path, auditor: Auditor, required: bool) -> dict[str, Any] | None:
    path = run_dir / "runinfo.json"
    if not path.is_file():
        if required:
            auditor.blocker("artifact_missing", str(path))
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        auditor.blocker("runinfo_unreadable", f"{path}: {exc}")
        return None


# ===========================================================================
# manifest / mask / diagnostics
# ===========================================================================


def audit_manifest(rows: Sequence[dict[str, str]], estimand: str, auditor: Auditor) -> None:
    auditor.require(len(rows) == FITS_PER_ESTIMAND, "manifest_row_count",
                    f"{len(rows)} != {FITS_PER_ESTIMAND}")
    for field in REQUIRED_MASK_PROVENANCE_FIELDS:
        auditor.require(field in rows[0], "manifest_provenance_field", f"missing {field}")

    keys = [(int(r["K_TRUE"]), int(r["replicate"]), int(r["K"]), int(r["start"])) for r in rows]
    auditor.require(len(keys) == len(set(keys)), "manifest_duplicate_key", "duplicate (K_TRUE,rep,K,start)")
    expected_keys = {(kt, rp, k, s) for kt in NEW_K_TRUE for rp in REPLICATES
                     for k in K_CANDIDATES for s in STARTS}
    auditor.require(set(keys) == expected_keys, "manifest_key_set", "key set differs from frozen set")
    auditor.require(tuple(keys) == tuple(sorted(keys)), "manifest_order", "rows are not in frozen order")
    auditor.require(all(int(r["K_TRUE"]) != ANCHOR_K_TRUE for r in rows), "manifest_anchor_excluded",
                    "K_TRUE=3 must not appear in a new manifest")

    model_seeds = [int(r["model_seed"]) for r in rows]
    auditor.require(len(model_seeds) == len(set(model_seeds)), "manifest_model_seed_unique",
                    "duplicate model seed within estimand")

    for row in rows:
        kt, rp, k, s = int(row["K_TRUE"]), int(row["replicate"]), int(row["K"]), int(row["start"])
        auditor.require(int(row["data_seed"]) == expected_data_seed(kt, rp),
                        "manifest_data_seed", f"K{kt} r{rp}")
        auditor.require(int(row["split_seed"]) == expected_split_seed(rp),
                        "manifest_split_seed", f"K{kt} r{rp}")
        auditor.require(int(row["model_seed"]) == expected_model_seed(kt, rp, k, s),
                        "manifest_model_seed", f"K{kt} r{rp} K{k} s{s}")
        auditor.require(abs(float(row["w_true"]) - expected_w_true(estimand, kt)) <= 1e-12,
                        "manifest_w_true", f"{estimand} K{kt}")
        auditor.require(float(row["w0_true"]) == W0_TRUE, "manifest_w0_true", f"K{kt}")
        auditor.require(row["mask_design"] == MASK_DESIGN, "manifest_mask_design", row["mask_design"])
        auditor.require(row["estimand"] == estimand, "manifest_estimand", row["estimand"])
        auditor.require(row["role"] == ROLE_BY_ESTIMAND[estimand], "manifest_role", row["role"])


def audit_mask_provenance(rows: Sequence[dict[str, str]], estimand: str,
                          anchors: dict[int, tuple[str, str]], auditor: Auditor) -> None:
    auditor.require(len(rows) == CELLS_PER_ESTIMAND, "mask_row_count",
                    f"{len(rows)} != {CELLS_PER_ESTIMAND}")
    keys = [(r["estimand"], int(r["K_TRUE"]), int(r["replicate"])) for r in rows]
    auditor.require(len(keys) == len(set(keys)), "mask_duplicate_key", "duplicate (estimand,K_TRUE,rep)")
    auditor.require(set(keys) == {(estimand, kt, rp) for kt in NEW_K_TRUE for rp in REPLICATES},
                    "mask_key_set", "mask provenance key set differs from the frozen set")
    auditor.require(all(int(r["K_TRUE"]) != ANCHOR_K_TRUE for r in rows), "mask_anchor_row",
                    "Phase 7e anchor rows must not be copied into mask_provenance.csv")
    for field in REQUIRED_MASK_PROVENANCE_FIELDS:
        auditor.require(field in rows[0], "mask_provenance_field", f"missing {field}")

    for row in rows:
        replicate = int(row["replicate"])
        anchor = anchors.get(replicate)
        if anchor is None:
            auditor.blocker("mask_anchor_missing", f"replicate {replicate}")
            continue
        anchor_test, anchor_train = anchor
        auditor.require(row["anchor_mask_hash"] == anchor_test, "mask_anchor_hash",
                        f"r{replicate} anchor_mask_hash differs from the Phase 7e test_mask_hash")
        auditor.require(row["anchor_train_mask_hash"] == anchor_train, "mask_anchor_train_hash",
                        f"r{replicate} anchor_train_mask_hash differs from the Phase 7e train_mask_hash")
        auditor.require(row["split_mask_hash"] == anchor_test, "mask_sc_test_match",
                        f"r{replicate} test mask differs from the anchor")
        auditor.require(row["train_mask_hash"] == anchor_train, "mask_sc_train_match",
                        f"r{replicate} train mask differs from the anchor")
        auditor.require(_is_true(row["intentional_seed_reuse"]), "mask_intentional_reuse",
                        f"r{replicate}")
        auditor.require(_is_true(row["anchor_match"]), "mask_anchor_match", f"r{replicate}")
        auditor.require(int(row["split_seed"]) == expected_split_seed(replicate),
                        "mask_split_seed", f"r{replicate}")


def audit_diagnostics(rows: Sequence[dict[str, str]], auditor: Auditor) -> None:
    """Structure only.  Diagnostic VALUES are never pass/fail."""

    auditor.require(len(rows) == CELLS_PER_ESTIMAND, "diagnostics_row_count",
                    f"{len(rows)} != {CELLS_PER_ESTIMAND}")
    keys = [(int(r["K_TRUE"]), int(r["replicate"])) for r in rows]
    auditor.require(len(keys) == len(set(keys)), "diagnostics_duplicate_key", "duplicate (K_TRUE,rep)")
    auditor.require(set(keys) == {(kt, rp) for kt in NEW_K_TRUE for rp in REPLICATES},
                    "diagnostics_key_set", "diagnostics key set differs from the frozen set")
    auditor.require({int(r["K_TRUE"]) for r in rows} == set(NEW_K_TRUE), "diagnostics_k_true_set",
                    "diagnostics K_TRUE set differs from the frozen new set")
    auditor.require(all(int(r["K_TRUE"]) != ANCHOR_K_TRUE for r in rows), "diagnostics_no_anchor",
                    "K_TRUE=3 diagnostics must not be newly generated")
    forbidden = {"pass", "fail", "passed", "blocking", "gate", "threshold"}
    offending = sorted(set(rows[0]) & forbidden)
    auditor.require(not offending, "diagnostics_record_only",
                    f"diagnostics must not carry pass/fail columns: {offending}")


# ===========================================================================
# selection matrix — independent recomputation
# ===========================================================================


def read_scores(rows: Sequence[dict[str, str]], estimand: str,
                auditor: Auditor) -> dict[tuple[int, int], dict[int, dict[int, float]]]:
    """(K_TRUE, replicate) -> K -> start -> score, from the run's fit_results."""

    if not auditor.require(SCORE_COLUMN in rows[0], "fit_results_score_column",
                           f"fit_results.csv must expose {SCORE_COLUMN}"):
        return {}
    auditor.require(len(rows) == FITS_PER_ESTIMAND, "fit_results_row_count",
                    f"{len(rows)} != {FITS_PER_ESTIMAND}")
    keys = [(int(r["K_TRUE"]), int(r["replicate"]), int(r["K"]), int(r["start"])) for r in rows]
    auditor.require(len(keys) == len(set(keys)), "fit_results_duplicate_key", "duplicate fit key")
    auditor.require(set(keys) == {(kt, rp, k, s) for kt in NEW_K_TRUE for rp in REPLICATES
                                  for k in K_CANDIDATES for s in STARTS},
                    "fit_results_key_set", "fit_results key set differs from the frozen set")
    grouped: dict[tuple[int, int], dict[int, dict[int, float]]] = {}
    for row in rows:
        auditor.require(row["estimand"] == estimand, "fit_results_estimand", row["estimand"])
        cell = (int(row["K_TRUE"]), int(row["replicate"]))
        grouped.setdefault(cell, {}).setdefault(int(row["K"]), {})[int(row["start"])] = \
            float(row[SCORE_COLUMN])
    return grouped


def audit_selection_matrix(rows: Sequence[dict[str, str]], estimand: str,
                           scores: Mapping[tuple[int, int], Mapping[int, Mapping[int, float]]],
                           anchor_selected: Mapping[int, int], auditor: Auditor) -> None:
    header = list(rows[0])
    auditor.require(tuple(header) == SELECTION_MATRIX_COLUMNS, "matrix_columns",
                    f"header differs from the frozen 11-column schema: {header}")
    for forbidden in SELECTION_MATRIX_FORBIDDEN:
        auditor.require(forbidden not in header, "matrix_forbidden_column",
                        f"{forbidden} must not appear in the integrated matrix")

    keys = [(r["estimand"], int(r["K_TRUE"]), int(r["replicate"])) for r in rows]
    expected_keys = {(estimand, kt, rp) for kt in FULL_K_TRUE_GRID for rp in REPLICATES}
    auditor.require(len(keys) == len(set(keys)), "matrix_duplicate_key", f"duplicate key in {keys}")
    missing = sorted(expected_keys - set(keys))
    unexpected = sorted(set(keys) - expected_keys)
    auditor.require(not missing, "matrix_missing_key", f"missing {missing}")
    auditor.require(not unexpected, "matrix_unexpected_key", f"unexpected {unexpected}")
    auditor.require(len(rows) == MATRIX_ROWS_PER_ESTIMAND, "matrix_row_count",
                    f"{len(rows)} != {MATRIX_ROWS_PER_ESTIMAND}")

    for row in rows:
        k_true = int(row["K_TRUE"])
        replicate = int(row["replicate"])
        reported = int(row["selected_k"])

        auditor.require(row["estimand"] in ESTIMANDS, "matrix_unknown_estimand", row["estimand"])
        auditor.require(row["role"] == ROLE_BY_ESTIMAND.get(row["estimand"]), "matrix_role",
                        f"role {row['role']} does not match estimand {row['estimand']}")
        auditor.require(k_true in FULL_K_TRUE_GRID, "matrix_k_true_grid", str(k_true))

        # --- independent recomputation of selected_k -----------------------
        recomputed: int | None = None
        if k_true == ANCHOR_K_TRUE:
            recomputed = anchor_selected.get(replicate)
            if recomputed is None:
                auditor.blocker("matrix_anchor_selector", f"r{replicate}: no anchor recomputation")
        else:
            cell = scores.get((k_true, replicate))
            if cell is None:
                auditor.blocker("matrix_score_source",
                                f"no per-(K,start) scores for K_TRUE={k_true} r{replicate}")
            else:
                try:
                    recomputed = select_k_independently(cell)
                except ValueError as exc:
                    auditor.blocker("matrix_selector", f"K_TRUE={k_true} r{replicate}: {exc}")
        if recomputed is not None:
            auditor.require(reported == recomputed, "matrix_selected_k_recomputed",
                            f"K_TRUE={k_true} r{replicate}: artifact {reported} != recomputed {recomputed}")

        # --- derived columns recomputed from K_TRUE, never cross-checked ----
        signed = reported - k_true
        auditor.require(int(row["signed_error"]) == signed, "matrix_signed_error", str(row))
        auditor.require(int(row["abs_error"]) == abs(signed), "matrix_abs_error", str(row))
        auditor.require(row["label"] == expected_label(signed), "matrix_label", str(row))

        # --- lineage / provenance ------------------------------------------
        auditor.require(bool(row["lineage"]) and bool(row["run_code_sha"]) and bool(row["artifact_dir"]),
                        "matrix_provenance", "lineage/run_code_sha/artifact_dir must be non-empty")
        if k_true == ANCHOR_K_TRUE:
            auditor.require(row["lineage"] == LINEAGE_ANCHOR, "matrix_anchor_lineage",
                            f"K_TRUE=3 row must carry lineage {LINEAGE_ANCHOR}")
            auditor.require(row["run_code_sha"] == PHASE7E_RUN_CODE_SHA, "matrix_anchor_sha",
                            "K_TRUE=3 row must carry the Phase 7e run_code_sha")
            auditor.require(row["artifact_dir"] == PHASE7E_ARTIFACT_DIR, "matrix_anchor_dir",
                            "K_TRUE=3 row must point at the Phase 7e artifact directory")
        else:
            auditor.require(row["lineage"] != LINEAGE_ANCHOR, "matrix_new_lineage",
                            "new K_TRUE row must not claim the Phase 7e anchor lineage")
            auditor.require(row["run_code_sha"] != PHASE7E_RUN_CODE_SHA, "matrix_new_sha",
                            "new K_TRUE row must not carry the Phase 7e run_code_sha")
            auditor.require(row["artifact_dir"] != PHASE7E_ARTIFACT_DIR, "matrix_new_dir",
                            "new K_TRUE row must not point at the Phase 7e artifact directory")


# ===========================================================================
# driver
# ===========================================================================


def audit_run_dir(run_dir: Path, estimand: str, mode: str = "config",
                  phase7e_dir: Path | None = None) -> Auditor:
    auditor = Auditor()
    run_dir = Path(run_dir)
    if estimand not in ESTIMANDS:
        auditor.blocker("unknown_estimand", estimand)
        return auditor
    if mode not in AUDIT_MODES:
        auditor.blocker("unknown_mode", f"{mode!r} not in {sorted(AUDIT_MODES)}")
        return auditor

    required = AUDIT_MODES[mode]
    for name in required:
        if not (run_dir / name).is_file():
            auditor.blocker("required_artifact_missing", f"{mode} mode requires {name}")

    anchors = read_phase7e_anchor(auditor, phase7e_dir)
    runinfo = read_runinfo(run_dir, auditor, required=True)

    gate_rows = _read_csv(run_dir / "config_gate.csv", auditor)
    if gate_rows:
        audit_config_gate(gate_rows, estimand, auditor, runinfo)

    manifest = _read_csv(run_dir / "manifest.csv", auditor)
    if manifest:
        audit_manifest(manifest, estimand, auditor)

    mask_rows = _read_csv(run_dir / "mask_provenance.csv", auditor)
    if mask_rows:
        audit_mask_provenance(mask_rows, estimand, anchors, auditor)

    diag_rows = _read_csv(run_dir / "diagnostics.csv", auditor)
    if diag_rows:
        audit_diagnostics(diag_rows, auditor)

    if mode == "selection":
        fit_rows = _read_csv(run_dir / "fit_results.csv", auditor)
        scores = read_scores(fit_rows, estimand, auditor) if fit_rows else {}
        anchor_selected = recompute_anchor_selection(auditor, phase7e_dir)
        matrix_rows = _read_csv(run_dir / "k_true_selection_matrix.csv", auditor)
        if matrix_rows:
            audit_selection_matrix(matrix_rows, estimand, scores, anchor_selected, auditor)
    return auditor


def _render(auditor: Auditor, run_dir: Path, estimand: str, mode: str) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "estimand": estimand,
        "mode": mode,
        "blocker_count": len(auditor.blockers),
        "high_count": len(auditor.highs),
        "verdict": "PASS" if not auditor.blockers and not auditor.highs else "FAIL",
        "findings": [
            {"severity": f.severity, "check": f.check, "detail": f.detail}
            for f in auditor.findings
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Artifact-only audit of a Phase 8b run directory")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--estimand", choices=ESTIMANDS, required=True)
    parser.add_argument("--mode", choices=sorted(AUDIT_MODES), default="config")
    parser.add_argument("--phase7e-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    auditor = audit_run_dir(args.run_dir, args.estimand, args.mode, args.phase7e_dir)
    report = _render(auditor, args.run_dir, args.estimand, args.mode)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
