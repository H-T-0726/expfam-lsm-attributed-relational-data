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

# S2 (Issue #51) implemented the direct Phase 8b leakage falsification that S1
# left as a follow-up: A01 raw held-out Y injection, A02 ScoreOnlyTarget
# rejection and A03 post-fit mask substitution are now tested against the
# Phase 8b boundary with an injectable counting adapter.  The real EM path is
# still closed; smoke remains gated on a human approval.

LEAKAGE_BOUNDARY_VERSION = "phase8b-leakage-boundary-v1"

# Future smoke/full artifacts carry the leakage-gate evidence.  Audited when the
# artifact is present; S2 itself produces no scientific result artifact.
LEAKAGE_GATE_COLUMNS = (
    "estimand", "role", "K_TRUE", "replicate", "K", "start",
    "pre_fit_test_mask_hash", "pre_fit_train_mask_hash",
    "post_fit_test_mask_hash", "post_fit_train_mask_hash",
    "anchor_mask_hash", "anchor_train_mask_hash",
    "pre_fit_passed", "post_fit_passed", "fit_boundary_status", "boundary_version",
)


def audit_leakage_gate(rows: Sequence[dict[str, str]], estimand: str,
                       anchors: dict[int, tuple[str, str]], auditor: Auditor) -> None:
    """Fail-closed audit of the leakage-gate provenance rows.

    Every fit must show: pre-fit and post-fit hashes equal to each other AND to
    the frozen Phase 7e anchor, on BOTH the test and the train side.
    """

    header = list(rows[0])
    auditor.require(tuple(header) == LEAKAGE_GATE_COLUMNS, "leakage_columns",
                    f"header differs from the frozen schema: {header}")
    auditor.require(len(rows) == FITS_PER_ESTIMAND, "leakage_row_count",
                    f"{len(rows)} != {FITS_PER_ESTIMAND}")
    keys = [(int(r["K_TRUE"]), int(r["replicate"]), int(r["K"]), int(r["start"])) for r in rows]
    auditor.require(len(keys) == len(set(keys)), "leakage_duplicate_key", "duplicate fit key")

    for row in rows:
        replicate = int(row["replicate"])
        anchor = anchors.get(replicate)
        if anchor is None:
            auditor.blocker("leakage_anchor_missing", f"replicate {replicate}")
            continue
        anchor_test, anchor_train = anchor
        auditor.require(row["estimand"] == estimand, "leakage_estimand", row["estimand"])
        auditor.require(row["role"] == ROLE_BY_ESTIMAND[estimand], "leakage_role", row["role"])
        auditor.require(row["boundary_version"] == LEAKAGE_BOUNDARY_VERSION,
                        "leakage_boundary_version", row["boundary_version"])
        for side, anchor_hash in (("test", anchor_test), ("train", anchor_train)):
            pre = row[f"pre_fit_{side}_mask_hash"]
            post = row[f"post_fit_{side}_mask_hash"]
            auditor.require(pre == post, f"leakage_{side}_mask_changed",
                            f"r{replicate}: {side} mask hash changed across the fit")
            auditor.require(pre == anchor_hash, f"leakage_{side}_anchor_mismatch",
                            f"r{replicate}: {side} mask hash differs from the Phase 7e anchor")
        auditor.require(_is_true(row["pre_fit_passed"]), "leakage_pre_fit_failed", str(row))
        auditor.require(_is_true(row["post_fit_passed"]), "leakage_post_fit_failed", str(row))
        auditor.require(row["fit_boundary_status"] == "clean", "leakage_boundary_status",
                        row["fit_boundary_status"])


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


# ===========================================================================
# Phase 8b S2c smoke artifact audit (Issue #55) -- artifact-only, fail closed
# ===========================================================================
#
# Every expectation below is an INDEPENDENT restatement of the frozen protocol.
# Nothing is imported from the harness: not the selector, not the summary
# builder, not the artifact builder.  The two-start means and the selected K
# are recomputed from the six CSV scores alone, so a self-consistent tampering
# of smoke_summary.json cannot pass.

APPROVED_SCIENTIFIC_MAIN_SHA = "68c78e1191889609dead05ea5a9fb11525ce92e2"

# The frozen scientific protocol hash, restated here as an INDEPENDENT
# constant.  It is deliberately not imported from the harness: an audit that
# asks the audited code what the right answer is proves nothing, and a
# self-consistent artifact set that agrees with itself on a fabricated hash
# must still fail.
EXPECTED_SMOKE_PROTOCOL_HASH = \
    "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"

SMOKE_PROTOCOL_ISSUE_NUMBER = 53
SMOKE_EXECUTION_ISSUE_NUMBER = 55
SMOKE_ARTIFACT_VERSION = "phase8b-smoke-artifact-v1"

SMOKE_ESTIMAND = "A"
SMOKE_ROLE = "primary"
SMOKE_K_TRUE = 1
SMOKE_REPLICATE = 1
SMOKE_K_CANDIDATES = (2, 3, 4)
SMOKE_STARTS = (1, 2)
EXPECTED_SMOKE_FITS = 6
EXPECTED_CANARY_FITS = 2
EXPECTED_REAL_EM_BUDGET = 8

SMOKE_DATA_SEED_BASE = 61000
SMOKE_MODEL_SEED_BASE = 630000
SMOKE_SPLIT_SEED = 42001
CANARY_K_EST = 1
CANARY_START = 1
CANARY_MODEL_SEED = 641011
CANARY_STATUS_PASS = "PASS"

# Frozen canary comparison tolerances, restated INDEPENDENTLY.  The values were
# read from the read-only Phase 7e source (``CANARY_ATOL``/``CANARY_RTOL`` in
# run_heldout_k_selection_pilot.py), not guessed and not imported: an audit that
# asks the audited code what tolerance it used cannot detect a relaxed one.
EXPECTED_CANARY_ATOL = 1e-12
EXPECTED_CANARY_RTOL = 1e-10

# Restated independently for the same reason.
EXPECTED_LEAKAGE_BOUNDARY_VERSION = "phase8b-leakage-boundary-v1"

# audit_report.json is this module's OUTPUT, so it is never a required input.
SMOKE_AUDIT_INPUT_FILES = (
    "authorization.json",
    "canary.json",
    "canary_audit.json",
    "runinfo.json",
    "smoke_fit_results.csv",
    "smoke_summary.json",
)

SMOKE_FIT_RESULTS_COLUMNS = (
    "run_code_sha",
    "approved_scientific_main_sha",
    "protocol_hash",
    "estimand",
    "role",
    "K_TRUE",
    "replicate",
    "K",
    "start",
    "data_seed",
    "split_seed",
    "model_seed",
    "pre_fit_test_hash",
    "pre_fit_train_hash",
    "post_fit_test_hash",
    "post_fit_train_hash",
    "anchor_test_hash",
    "anchor_train_hash",
    "boundary_version",
    "fit_status",
    "internal_retry",
    "warning_count",
    "q_failure",
    "nan_occurred",
    "finite_state",
    "heldout_mean_log_score",
    "score_config_hash",
    "canary_provenance",
    "real_canary_fits_executed",
    "real_smoke_fits_executed",
)


def expected_smoke_data_seed(k_true: int = SMOKE_K_TRUE,
                             replicate: int = SMOKE_REPLICATE) -> int:
    return SMOKE_DATA_SEED_BASE + 100 * int(k_true) + int(replicate)


def expected_smoke_model_seed(k: int, start: int, k_true: int = SMOKE_K_TRUE,
                              replicate: int = SMOKE_REPLICATE) -> int:
    return (SMOKE_MODEL_SEED_BASE + 10000 * int(k_true) + 1000 * int(replicate)
            + 10 * int(k) + int(start))


def expected_smoke_manifest_keys() -> tuple[tuple[int, int], ...]:
    return tuple((k, start) for k in SMOKE_K_CANDIDATES for start in SMOKE_STARTS)


ACCEPTED_TRUE_TOKENS = ("True", "true")
ACCEPTED_FALSE_TOKENS = ("False", "false")


def _parse_int_field(row: Mapping[str, str], column: str, auditor: Auditor,
                     check: str, label: str) -> int | None:
    """Strict integer field.  Never raises; a bad value is a BLOCKER.

    ``"2.0"``, ``""``, ``"nan"``, ``"inf"``, ``True`` and any non-numeric text
    are all rejected: the schema stores plain decimal integers.
    """

    if column not in row:
        auditor.blocker(check, f"{label}: column {column} is missing")
        return None
    raw = row[column]
    if not isinstance(raw, str) or raw.strip() == "":
        auditor.blocker(check, f"{label}: {column} is blank")
        return None
    text = raw.strip()
    body = text[1:] if text[:1] in "+-" else text
    if not body.isdigit():
        auditor.blocker(check, f"{label}: {column} is not a plain integer: {raw!r}")
        return None
    try:
        return int(text)
    except ValueError:  # pragma: no cover - isdigit already guarantees this
        auditor.blocker(check, f"{label}: {column} is not an integer: {raw!r}")
        return None


def _parse_finite_float_field(row: Mapping[str, str], column: str, auditor: Auditor,
                              check: str, label: str) -> float | None:
    """Strict finite float field.  NaN, +/-Inf, blank and text are BLOCKERs."""

    if column not in row:
        auditor.blocker(check, f"{label}: column {column} is missing")
        return None
    raw = row[column]
    if not isinstance(raw, str) or raw.strip() == "":
        auditor.blocker(check, f"{label}: {column} is blank")
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        auditor.blocker(check, f"{label}: {column} is not a number: {raw!r}")
        return None
    if not math.isfinite(number):
        auditor.blocker(check, f"{label}: {column} is nonfinite: {raw!r}")
        return None
    return number


def _require_exact_float(payload: Mapping[str, Any], key: str, expected: float,
                         auditor: Auditor, check: str) -> None:
    """A frozen numeric constant must be present, finite and exactly equal.

    Never raises: a missing key, a string, a bool, NaN or +/-Inf is a BLOCKER.
    There is deliberately no tolerance on the tolerance.
    """

    if key not in payload:
        auditor.blocker(check, f"{key} is missing")
        return
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        auditor.blocker(check, f"{key} is not a number: {value!r}")
        return
    number = float(value)
    if not math.isfinite(number):
        auditor.blocker(check, f"{key} is nonfinite: {value!r}")
        return
    auditor.require(number == expected, check, f"{key}: {value!r} != {expected!r}")


def _require_exact_int(payload: Mapping[str, Any], key: str, expected: int,
                       auditor: Auditor, check: str) -> None:
    """A frozen integer field must be present, an exact ``int`` and equal.

    Value equality alone is not enough: ``1.0 == 1`` and ``True == 1`` are both
    true in Python, so a float or a bool could otherwise be substituted for a
    frozen integer and still be audited as PASS.  ``type(value) is int`` is
    required on purpose -- ``isinstance`` would admit ``bool`` -- and nothing is
    ever coerced: ``int("1")`` / ``int(1.0)`` would normalise invalid input away.
    Never raises: a missing key, a float, a bool, a string, ``None`` or any
    other type is a structured BLOCKER.
    """

    if key not in payload:
        auditor.blocker(check, f"{key} is missing")
        return
    value = payload[key]
    if type(value) is not int:
        auditor.blocker(check, f"{key} is not an int: {value!r} "
                               f"({type(value).__name__})")
        return
    auditor.require(value == expected, check, f"{key}: {value!r} != {expected!r}")


def _require_exact_int_list(payload: Mapping[str, Any], key: str,
                            expected: Sequence[int], auditor: Auditor,
                            check: str, *, sort: bool = False) -> None:
    """The list form of :func:`_require_exact_int` (frozen integer vectors).

    Every element must satisfy ``type(item) is int``: ``[2.0, 3, 4]`` and
    ``[True, 3, 4]`` compare equal to ``[2, 3, 4]`` element-wise and would
    otherwise be accepted.  ``sort=True`` compares the two sets in ascending
    order (a reported tie set), never by coercing the entries.
    """

    if key not in payload:
        auditor.blocker(check, f"{key} is missing")
        return
    value = payload[key]
    if not isinstance(value, list):
        auditor.blocker(check, f"{key} is not a list: {value!r}")
        return
    bad = [item for item in value if type(item) is not int]
    if bad:
        auditor.blocker(check, f"{key} holds non-int entries: {bad!r}")
        return
    reported = sorted(value) if sort else list(value)
    wanted = sorted(expected) if sort else list(expected)
    auditor.require(reported == wanted, check, f"{key}: {reported!r} != {wanted!r}")


def _require_nonnegative_int(payload: Mapping[str, Any], key: str,
                            auditor: Auditor, check: str) -> int | None:
    """A count field: present, exactly ``int``, never a bool, never negative.

    Unlike :func:`_require_exact_int` the value is NOT frozen at one number: a
    count that a PASS verdict tolerates (MEDIUM findings do not block) still has
    an integer schema.  Returns the value so the caller can cross-check it.
    """

    if key not in payload:
        auditor.blocker(check, f"{key} is missing")
        return None
    value = payload[key]
    if type(value) is not int:
        auditor.blocker(check, f"{key} is not an int: {value!r} "
                               f"({type(value).__name__})")
        return None
    if value < 0:
        auditor.blocker(check, f"{key} is negative: {value!r}")
        return None
    return value


def _require_exact_int_member(payload: Mapping[str, Any], key: str,
                              allowed: Sequence[int], auditor: Auditor,
                              check: str) -> None:
    """A frozen integer field whose schema also fixes the admissible set."""

    if key not in payload:
        auditor.blocker(check, f"{key} is missing")
        return
    value = payload[key]
    if type(value) is not int:
        auditor.blocker(check, f"{key} is not an int: {value!r} "
                               f"({type(value).__name__})")
        return
    auditor.require(value in tuple(allowed), check,
                    f"{key}: {value!r} not in {list(allowed)!r}")


def _parse_bool_field(row: Mapping[str, str], column: str, auditor: Auditor,
                      check: str, label: str) -> bool | None:
    """Strict boolean field.  Only the canonical tokens are accepted.

    ``"False"`` is a truthy Python string, so a bare truth test would silently
    invert every negative flag; the accepted token set is fixed instead.
    """

    if column not in row:
        auditor.blocker(check, f"{label}: column {column} is missing")
        return None
    raw = row[column]
    if raw in ACCEPTED_TRUE_TOKENS:
        return True
    if raw in ACCEPTED_FALSE_TOKENS:
        return False
    auditor.blocker(check, f"{label}: {column} is not a canonical boolean: {raw!r}")
    return None


def _read_json(path: Path, auditor: Auditor) -> dict[str, Any] | None:
    if not path.is_file():
        auditor.blocker("smoke_artifact_missing", str(path.name))
        return None
    text = path.read_text(encoding="utf-8")
    if text.strip() == "":
        auditor.blocker("smoke_artifact_empty", str(path.name))
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        auditor.blocker("smoke_artifact_malformed", f"{path.name}: {error}")
        return None
    if not isinstance(payload, dict):
        auditor.blocker("smoke_artifact_not_object", str(path.name))
        return None
    return payload


def _as_float(value: str, auditor: Auditor, check: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        auditor.blocker(check, f"not a number: {value!r}")
        return None
    if not math.isfinite(number):
        auditor.blocker(check, f"nonfinite: {value!r}")
        return None
    return number


def _is_false(value: str) -> bool:
    return value in ("False", "false")


def _require_frozen_protocol_hash(payload: Mapping[str, Any], auditor: Auditor,
                                  label: str) -> None:
    """Every artifact must carry the frozen hash, not merely agree with the others."""

    auditor.require(payload.get("protocol_hash") == EXPECTED_SMOKE_PROTOCOL_HASH,
                    "smoke_protocol_hash_frozen",
                    f"{label}: expected {EXPECTED_SMOKE_PROTOCOL_HASH}, "
                    f"got {payload.get('protocol_hash')!r}")


def audit_smoke_authorization(payload: Mapping[str, Any], auditor: Auditor) -> None:
    _require_frozen_protocol_hash(payload, auditor, "authorization.json")
    auditor.require(payload.get("artifact_version") == SMOKE_ARTIFACT_VERSION,
                    "smoke_auth_version", str(payload.get("artifact_version")))
    auditor.require(payload.get("approved_scientific_main_sha") == APPROVED_SCIENTIFIC_MAIN_SHA,
                    "smoke_auth_baseline_sha",
                    f"expected {APPROVED_SCIENTIFIC_MAIN_SHA}, "
                    f"got {payload.get('approved_scientific_main_sha')!r}")
    _require_exact_int(payload, "execution_issue_number", SMOKE_EXECUTION_ISSUE_NUMBER,
                       auditor, "smoke_auth_execution_issue")
    _require_exact_int(payload, "protocol_origin_issue_number", SMOKE_PROTOCOL_ISSUE_NUMBER,
                       auditor, "smoke_auth_protocol_issue")
    auditor.require(payload.get("independent_review_pass") is True,
                    "smoke_auth_independent_review", "INDEPENDENT_REVIEW_PASS is not True")
    auditor.require(payload.get("human_smoke_approval") is True,
                    "smoke_auth_human_approval", "HUMAN_SMOKE_APPROVAL is not True")
    _require_exact_int(payload, "expected_canary_fits", EXPECTED_CANARY_FITS,
                       auditor, "smoke_auth_canary_count")
    _require_exact_int(payload, "expected_smoke_fits", EXPECTED_SMOKE_FITS,
                       auditor, "smoke_auth_smoke_count")
    auditor.require(payload.get("estimand") == SMOKE_ESTIMAND
                    and payload.get("role") == SMOKE_ROLE,
                    "smoke_auth_estimand", str(payload.get("estimand")))
    _require_exact_int(payload, "k_true", SMOKE_K_TRUE, auditor, "smoke_auth_cell")
    _require_exact_int(payload, "replicate", SMOKE_REPLICATE, auditor, "smoke_auth_cell")
    _require_exact_int(payload, "split_seed", SMOKE_SPLIT_SEED,
                       auditor, "smoke_auth_split_seed")
    _require_exact_int(payload, "data_seed", expected_smoke_data_seed(),
                       auditor, "smoke_auth_data_seed")
    _require_exact_int(payload, "canary_model_seed", CANARY_MODEL_SEED,
                       auditor, "smoke_auth_canary_seed")
    expected_models = [expected_smoke_model_seed(k, start)
                       for k, start in expected_smoke_manifest_keys()]
    _require_exact_int_list(payload, "smoke_model_seeds", expected_models,
                            auditor, "smoke_auth_model_seeds")


def audit_smoke_canary(payload: Mapping[str, Any], anchors: Mapping[int, tuple[str, str]],
                       auditor: Auditor, *, expect_execution_mode: str = "real") -> None:
    _require_frozen_protocol_hash(payload, auditor, "canary.json")
    auditor.require(payload.get("status") == CANARY_STATUS_PASS,
                    "smoke_canary_status", str(payload.get("status")))
    auditor.require(payload.get("approved_scientific_main_sha") == APPROVED_SCIENTIFIC_MAIN_SHA,
                    "smoke_canary_baseline_sha",
                    str(payload.get("approved_scientific_main_sha")))
    _require_exact_int(payload, "execution_issue_number", SMOKE_EXECUTION_ISSUE_NUMBER,
                       auditor, "smoke_canary_execution_issue")
    # The canary must carry the protocol-origin issue itself; the authorization
    # carrying it is not evidence about the canary.
    _require_exact_int(payload, "protocol_origin_issue_number", SMOKE_PROTOCOL_ISSUE_NUMBER,
                       auditor, "smoke_canary_protocol_issue")
    auditor.require(payload.get("estimand") == SMOKE_ESTIMAND,
                    "smoke_canary_estimand", str(payload.get("estimand")))
    auditor.require(payload.get("role") == SMOKE_ROLE,
                    "smoke_canary_role", str(payload.get("role")))
    _require_exact_int(payload, "k_true", SMOKE_K_TRUE, auditor, "smoke_canary_k_true")
    _require_exact_int(payload, "replicate", SMOKE_REPLICATE,
                       auditor, "smoke_canary_replicate")
    # Frozen tolerances: never relaxed, and never inspected before being fixed.
    _require_exact_float(payload, "canary_atol", EXPECTED_CANARY_ATOL,
                         auditor, "smoke_canary_atol")
    _require_exact_float(payload, "canary_rtol", EXPECTED_CANARY_RTOL,
                         auditor, "smoke_canary_rtol")
    auditor.require(payload.get("boundary_version") == EXPECTED_LEAKAGE_BOUNDARY_VERSION,
                    "smoke_canary_boundary_version", str(payload.get("boundary_version")))
    auditor.require(payload.get("execution_mode") == expect_execution_mode,
                    "smoke_canary_execution_mode",
                    f"{payload.get('execution_mode')!r} != {expect_execution_mode!r}")
    _require_exact_int(payload, "expected_fit_count", EXPECTED_CANARY_FITS,
                       auditor, "smoke_canary_fit_count")
    _require_exact_int(payload, "actual_fit_count", EXPECTED_CANARY_FITS,
                       auditor, "smoke_canary_fit_count")
    expected_real = EXPECTED_CANARY_FITS if expect_execution_mode == "real" else 0
    _require_exact_int(payload, "real_canary_fits_executed", expected_real,
                       auditor, "smoke_canary_real_fit_count")
    _require_exact_int(payload, "k_est", CANARY_K_EST, auditor, "smoke_canary_cell")
    _require_exact_int(payload, "start", CANARY_START, auditor, "smoke_canary_cell")
    _require_exact_int(payload, "model_seed", CANARY_MODEL_SEED,
                       auditor, "smoke_canary_model_seed")
    _require_exact_int(payload, "data_seed", expected_smoke_data_seed(),
                       auditor, "smoke_canary_data_seed")
    _require_exact_int(payload, "split_seed", SMOKE_SPLIT_SEED,
                       auditor, "smoke_canary_split_seed")
    auditor.require(payload.get("initialization_equal") is True
                    and payload.get("final_outputs_equal") is True,
                    "smoke_canary_invariance", "canary A/B outputs are not invariant")
    _require_exact_int(payload, "internal_retry", 0, auditor, "smoke_canary_retry")
    _require_exact_int(payload, "warning_count", 0, auditor, "smoke_canary_warnings")
    auditor.require(payload.get("q_failure") is False, "smoke_canary_q_failure", "q_failure")
    auditor.require(payload.get("nan_occurred") is False, "smoke_canary_nan", "nan_occurred")
    auditor.require(payload.get("finite_state") is True, "smoke_canary_finite", "finite_state")
    auditor.require(payload.get("fit_payload_a_hash") != payload.get("fit_payload_b_hash"),
                    "smoke_canary_payload_variation",
                    "canary A/B fit payloads are identical")
    anchor = anchors.get(SMOKE_REPLICATE)
    if anchor is not None:
        auditor.require(payload.get("anchor_test_hash") == anchor[0],
                        "smoke_canary_anchor_test", str(payload.get("anchor_test_hash")))
        auditor.require(payload.get("anchor_train_hash") == anchor[1],
                        "smoke_canary_anchor_train", str(payload.get("anchor_train_hash")))


def audit_smoke_fit_rows(rows: Sequence[dict[str, str]],
                         anchors: Mapping[int, tuple[str, str]],
                         auditor: Auditor) -> dict[int, dict[int, float]]:
    """Independently validate the six rows and return the scores by (K, start)."""

    scores: dict[int, dict[int, float]] = {}
    header = tuple(rows[0]) if rows else ()
    if not auditor.require(header == SMOKE_FIT_RESULTS_COLUMNS, "smoke_fit_columns",
                           f"header differs from the frozen schema: {list(header)}"):
        return scores
    if not auditor.require(len(rows) == EXPECTED_SMOKE_FITS, "smoke_fit_row_count",
                           f"{len(rows)} != {EXPECTED_SMOKE_FITS}"):
        return scores

    keys: list[tuple[int, int]] = []
    for index, row in enumerate(rows, start=1):
        k = _parse_int_field(row, "K", auditor, "smoke_fit_key_parse", f"row {index}")
        start = _parse_int_field(row, "start", auditor, "smoke_fit_key_parse", f"row {index}")
        if k is None or start is None:
            continue
        keys.append((k, start))
    auditor.require(len(set(keys)) == len(keys), "smoke_fit_duplicate_key", str(keys))
    auditor.require(tuple(keys) == expected_smoke_manifest_keys(), "smoke_fit_key_order",
                    f"{keys} != {list(expected_smoke_manifest_keys())}")

    anchor = anchors.get(SMOKE_REPLICATE)
    for index, row in enumerate(rows, start=1):
        label = f"row {index}"
        k = _parse_int_field(row, "K", auditor, "smoke_fit_key_parse", label)
        start = _parse_int_field(row, "start", auditor, "smoke_fit_key_parse", label)
        if k is None or start is None:
            continue
        label = f"K={k} start={start}"
        auditor.require(row.get("protocol_hash") == EXPECTED_SMOKE_PROTOCOL_HASH,
                        "smoke_protocol_hash_frozen",
                        f"smoke_fit_results.csv {label}: {row.get('protocol_hash')!r}")
        auditor.require(row["approved_scientific_main_sha"] == APPROVED_SCIENTIFIC_MAIN_SHA,
                        "smoke_fit_baseline_sha", f"{label}: {row['approved_scientific_main_sha']}")
        auditor.require(bool(row["run_code_sha"]), "smoke_fit_run_code_sha", label)
        auditor.require(row["estimand"] == SMOKE_ESTIMAND and row["role"] == SMOKE_ROLE,
                        "smoke_fit_estimand", label)
        k_true = _parse_int_field(row, "K_TRUE", auditor, "smoke_fit_cell", label)
        replicate = _parse_int_field(row, "replicate", auditor, "smoke_fit_cell", label)
        auditor.require(k_true == SMOKE_K_TRUE and replicate == SMOKE_REPLICATE,
                        "smoke_fit_cell", label)
        data_seed = _parse_int_field(row, "data_seed", auditor, "smoke_fit_data_seed", label)
        auditor.require(data_seed == expected_smoke_data_seed(),
                        "smoke_fit_data_seed", f"{label}: {row.get('data_seed')}")
        split_seed = _parse_int_field(row, "split_seed", auditor, "smoke_fit_split_seed", label)
        auditor.require(split_seed == SMOKE_SPLIT_SEED,
                        "smoke_fit_split_seed", f"{label}: {row.get('split_seed')}")
        model_seed = _parse_int_field(row, "model_seed", auditor, "smoke_fit_model_seed", label)
        auditor.require(model_seed == expected_smoke_model_seed(k, start),
                        "smoke_fit_model_seed", f"{label}: {row.get('model_seed')}")
        if anchor is not None:
            for column, expected in (("pre_fit_test_hash", anchor[0]),
                                     ("post_fit_test_hash", anchor[0]),
                                     ("anchor_test_hash", anchor[0]),
                                     ("pre_fit_train_hash", anchor[1]),
                                     ("post_fit_train_hash", anchor[1]),
                                     ("anchor_train_hash", anchor[1])):
                auditor.require(row[column] == expected, "smoke_fit_mask",
                                f"{label}: {column} differs from the Phase 7e anchor")
        auditor.require(row["fit_status"] == "clean", "smoke_fit_status",
                        f"{label}: {row['fit_status']}")
        auditor.require(
            _parse_int_field(row, "internal_retry", auditor, "smoke_fit_retry", label) == 0,
            "smoke_fit_retry", label)
        auditor.require(
            _parse_int_field(row, "warning_count", auditor, "smoke_fit_warnings", label) == 0,
            "smoke_fit_warnings", label)
        auditor.require(
            _parse_bool_field(row, "q_failure", auditor, "smoke_fit_q_failure", label) is False,
            "smoke_fit_q_failure", label)
        auditor.require(
            _parse_bool_field(row, "nan_occurred", auditor, "smoke_fit_nan", label) is False,
            "smoke_fit_nan", label)
        auditor.require(
            _parse_bool_field(row, "finite_state", auditor, "smoke_fit_finite", label) is True,
            "smoke_fit_finite", label)
        auditor.require(row["boundary_version"] == LEAKAGE_BOUNDARY_VERSION,
                        "smoke_fit_boundary_version", label)
        auditor.require(
            _parse_int_field(row, "real_canary_fits_executed", auditor,
                             "smoke_fit_canary_count", label) == EXPECTED_CANARY_FITS,
            "smoke_fit_canary_count", f"{label}: {row.get('real_canary_fits_executed')}")
        auditor.require(
            _parse_int_field(row, "real_smoke_fits_executed", auditor,
                             "smoke_fit_smoke_count", label) == EXPECTED_SMOKE_FITS,
            "smoke_fit_smoke_count", f"{label}: {row.get('real_smoke_fits_executed')}")
        score = _parse_finite_float_field(row, "heldout_mean_log_score", auditor,
                                          "smoke_fit_score", label)
        if score is not None:
            scores.setdefault(k, {})[start] = score
    return scores


def audit_smoke_summary(payload: Mapping[str, Any],
                        scores: Mapping[int, Mapping[int, float]],
                        auditor: Auditor) -> None:
    """Recompute the two-start means and the selector from the CSV scores alone."""

    _require_frozen_protocol_hash(payload, auditor, "smoke_summary.json")
    auditor.require(payload.get("approved_scientific_main_sha") == APPROVED_SCIENTIFIC_MAIN_SHA,
                    "smoke_summary_baseline_sha",
                    str(payload.get("approved_scientific_main_sha")))
    _require_exact_int(payload, "execution_issue_number", SMOKE_EXECUTION_ISSUE_NUMBER,
                       auditor, "smoke_summary_execution_issue")
    _require_exact_int(payload, "protocol_origin_issue_number", SMOKE_PROTOCOL_ISSUE_NUMBER,
                       auditor, "smoke_summary_protocol_issue")
    _require_exact_int(payload, "expected_smoke_fits", EXPECTED_SMOKE_FITS,
                       auditor, "smoke_summary_fit_count")
    _require_exact_int(payload, "actual_smoke_fits", EXPECTED_SMOKE_FITS,
                       auditor, "smoke_summary_fit_count")
    # selected_k is recomputed below, but K recovery is never evaluated: K_TRUE=1
    # is not in the candidate set {2,3,4}, so agreement is impossible by design.
    auditor.require(payload.get("k_recovery_evaluated") is False,
                    "smoke_summary_k_recovery_flag",
                    "k_recovery_evaluated must be False")
    auditor.require(payload.get("selected_k_interpretation") == "record_only",
                    "smoke_summary_selected_k_interpretation",
                    str(payload.get("selected_k_interpretation")))
    _require_exact_int_list(payload, "candidate_k", SMOKE_K_CANDIDATES,
                            auditor, "smoke_summary_candidates")
    # selected_k stays record-only evidence, but its schema type is still an
    # integer drawn from the frozen candidate set.  Checked here as well so a
    # malformed value is a BLOCKER even when the recomputation below cannot run.
    _require_exact_int_member(payload, "selected_k", SMOKE_K_CANDIDATES,
                              auditor, "smoke_summary_selected_k")

    per_k = payload.get("per_k")
    if not auditor.require(isinstance(per_k, Mapping), "smoke_summary_per_k",
                           "per_k is missing or not an object"):
        return
    # per_k keys are strings by schema: compared as strings, never converted.
    if not auditor.require(set(per_k) == {str(k) for k in SMOKE_K_CANDIDATES},
                           "smoke_summary_per_k",
                           f"per_k keys {sorted(map(str, per_k))} are not "
                           f"{sorted(str(k) for k in SMOKE_K_CANDIDATES)}"):
        return
    if not auditor.require(set(scores) == set(SMOKE_K_CANDIDATES), "smoke_summary_score_set",
                           f"scores cover {sorted(scores)}"):
        return

    recomputed: dict[int, float] = {}
    for k in SMOKE_K_CANDIDATES:
        by_start = scores.get(k, {})
        if not auditor.require(set(by_start) == set(SMOKE_STARTS), "smoke_summary_starts",
                               f"K={k} covers starts {sorted(by_start)}"):
            return
        mean = (by_start[1] + by_start[2]) / 2.0
        recomputed[k] = mean
        entry = per_k.get(str(k))
        if not auditor.require(isinstance(entry, Mapping), "smoke_summary_per_k_entry",
                               f"K={k} missing"):
            continue
        for name, expected in (("start_1", by_start[1]), ("start_2", by_start[2]),
                               ("mean", mean)):
            reported = entry.get(name)
            ok = isinstance(reported, (int, float)) and math.isfinite(float(reported)) \
                and abs(float(reported) - expected) <= 1e-12
            auditor.require(ok, "smoke_summary_arithmetic",
                            f"K={k} {name}: reported {reported!r}, recomputed {expected!r}")

    best = max(recomputed.values())
    tied = sorted(k for k, mean in recomputed.items() if best - mean <= TIE_TOLERANCE)
    selected = tied[0]
    _require_exact_int(payload, "selected_k", selected, auditor, "smoke_summary_selected_k")
    _require_exact_int_list(payload, "tie_candidates", tied, auditor,
                            "smoke_summary_tie_candidates", sort=True)


def audit_smoke_summary_provenance(payload: Mapping[str, Any], auditor: Auditor) -> None:
    """The cell provenance recorded in smoke_summary.json, as a schema check.

    Deliberately NOT part of :func:`audit_smoke_summary`: that function must
    contain no reference to the true K whatsoever, so that it cannot even appear
    to compare ``selected_k`` with ``K_TRUE``.  Recording which cell produced the
    summary is provenance; K recovery is still never evaluated.
    """

    _require_exact_int(payload, "k_true", SMOKE_K_TRUE, auditor, "smoke_summary_k_true")


def audit_smoke_runinfo(payload: Mapping[str, Any], auditor: Auditor) -> None:
    _require_frozen_protocol_hash(payload, auditor, "runinfo.json")
    auditor.require(payload.get("approved_scientific_main_sha") == APPROVED_SCIENTIFIC_MAIN_SHA,
                    "smoke_runinfo_baseline_sha",
                    str(payload.get("approved_scientific_main_sha")))
    auditor.require(bool(payload.get("run_code_sha")), "smoke_runinfo_run_code_sha", "missing")
    # The reviewed baseline must be in this commit's history, unconditionally.
    # The previous form ("run SHA differs from the baseline OR ancestry holds")
    # was vacuous for every normal descendant commit, which is exactly the case
    # a real execution runs in.
    auditor.require(payload.get("approved_baseline_is_ancestor") is True,
                    "smoke_runinfo_lineage",
                    "the approved scientific baseline is not recorded as an ancestor "
                    f"of the execution commit: {payload.get('approved_baseline_is_ancestor')!r}")
    _require_exact_int(payload, "execution_issue", SMOKE_EXECUTION_ISSUE_NUMBER,
                       auditor, "smoke_runinfo_execution_issue")
    _require_exact_int(payload, "protocol_origin_issue", SMOKE_PROTOCOL_ISSUE_NUMBER,
                       auditor, "smoke_runinfo_protocol_issue")
    _require_exact_int(payload, "expected_real_em_budget", EXPECTED_REAL_EM_BUDGET,
                       auditor, "smoke_runinfo_budget")
    _require_exact_int(payload, "expected_canary_fits", EXPECTED_CANARY_FITS,
                       auditor, "smoke_runinfo_expected_canary_fits")
    _require_exact_int(payload, "expected_smoke_fits", EXPECTED_SMOKE_FITS,
                       auditor, "smoke_runinfo_expected_smoke_fits")
    _require_exact_int(payload, "actual_canary_fits", EXPECTED_CANARY_FITS,
                       auditor, "smoke_runinfo_canary_count")
    _require_exact_int(payload, "actual_smoke_fits", EXPECTED_SMOKE_FITS,
                       auditor, "smoke_runinfo_smoke_count")
    _require_exact_int(payload, "full_fits_executed", 0,
                       auditor, "smoke_runinfo_full_fits")
    _require_exact_int(payload, "phase7e_rerun_count", 0,
                       auditor, "smoke_runinfo_phase7e_rerun")
    auditor.require(payload.get("working_tree_clean") is True,
                    "smoke_runinfo_working_tree", "the working tree was dirty")
    auditor.require(bool(payload.get("invocation_mode")) and bool(payload.get("requested_command")),
                    "smoke_runinfo_invocation", "invocation provenance is missing")


# ---------------------------------------------------------------------------
# Canary-only independent audit (Issue #55 execution order, step 3)
# ---------------------------------------------------------------------------
#
# The frozen order is: 2 canary fits -> persist evidence -> INDEPENDENTLY AUDIT
# that evidence -> only then 6 smoke fits.  The runner's own
# ``require_canary_pass_evidence`` is producer-side validation and cannot serve
# as that audit, so this module provides the independent one and publishes a
# durable verdict the runner can only read.
#
# It runs on the two files that exist right after the canary; runinfo, the fit
# CSV and the summary do not exist yet and are deliberately not required.

CANARY_AUDIT_VERSION = "phase8b-canary-audit-v1"
CANARY_AUDIT_FILENAME = "canary_audit.json"
CANARY_AUDIT_INPUT_FILES = ("authorization.json", "canary.json")


def audit_canary_run_dir(run_dir: Path, phase7e_dir: Path | None = None, *,
                         expect_execution_mode: str = "real") -> Auditor:
    """Independent, fail-closed audit of the canary evidence alone."""

    auditor = Auditor()
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        auditor.blocker("canary_run_dir_missing", str(run_dir))
        return auditor

    for name in CANARY_AUDIT_INPUT_FILES:
        if not (run_dir / name).is_file():
            auditor.blocker("required_artifact_missing", f"canary mode requires {name}")

    anchors = read_phase7e_anchor(auditor, phase7e_dir)
    authorization = _read_json(run_dir / "authorization.json", auditor)
    canary = _read_json(run_dir / "canary.json", auditor)

    if authorization is not None:
        audit_smoke_authorization(authorization, auditor)
    if canary is not None:
        audit_smoke_canary(canary, anchors, auditor,
                           expect_execution_mode=expect_execution_mode)

    payloads = {name: payload for name, payload in
                (("authorization.json", authorization), ("canary.json", canary))
                if payload is not None}
    protocol_hashes = {p.get("protocol_hash") for p in payloads.values()}
    auditor.require(protocol_hashes == {EXPECTED_SMOKE_PROTOCOL_HASH},
                    "canary_protocol_hash_lineage",
                    f"protocol hashes are not the single frozen value: {protocol_hashes}")
    audit_smoke_run_code_lineage(payloads, None, auditor)
    return auditor


def build_canary_audit_report(auditor: Auditor, run_dir: Path) -> dict[str, Any]:
    """Structured canary verdict.  Safe to build for a malformed artifact set."""

    def field(name: str, key: str, default: Any = None) -> Any:
        path = Path(run_dir) / name
        if not path.is_file():
            return default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return default
        return payload.get(key, default) if isinstance(payload, dict) else default

    blockers = len(auditor.blockers)
    highs = len(auditor.highs)
    mediums = len([f for f in auditor.findings if f.severity == "MEDIUM"])
    audited = sorted(name for name in CANARY_AUDIT_INPUT_FILES
                     if (Path(run_dir) / name).is_file())
    return {
        "audit_version": CANARY_AUDIT_VERSION,
        "status": "PASS" if blockers == 0 and highs == 0 else "FAIL",
        "blocker_count": blockers,
        "high_count": highs,
        "medium_count": mediums,
        "findings": [{"severity": f.severity, "check": f.check, "detail": f.detail}
                     for f in auditor.findings],
        "approved_scientific_main_sha": APPROVED_SCIENTIFIC_MAIN_SHA,
        "run_code_sha": field("canary.json", "run_code_sha"),
        "protocol_hash": EXPECTED_SMOKE_PROTOCOL_HASH,
        "protocol_origin_issue": SMOKE_PROTOCOL_ISSUE_NUMBER,
        "execution_issue": SMOKE_EXECUTION_ISSUE_NUMBER,
        "expected_canary_fits": EXPECTED_CANARY_FITS,
        "actual_canary_fits": field("canary.json", "real_canary_fits_executed"),
        "canary_execution_mode": field("canary.json", "execution_mode"),
        "canary_status": field("canary.json", "status"),
        "audited_files": audited,
        "run_dir": str(run_dir),
    }


def write_canary_audit_report(run_dir: Path, auditor: Auditor) -> Path:
    """Publish the canary verdict exactly once.  An existing verdict is a stop.

    Deliberately a DIFFERENT file from ``audit_report.json``: that name is
    reserved for the final smoke audit and is also never overwritten.
    """

    directory = Path(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"audit run directory does not exist: {directory}")
    path = directory / CANARY_AUDIT_FILENAME
    if path.exists():
        raise FileExistsError(
            f"{CANARY_AUDIT_FILENAME} already exists; a previous canary verdict is "
            f"never overwritten: {path}")
    return _atomic_write_json(path, build_canary_audit_report(auditor, directory))


def audit_canary_verdict_counts(payload: Mapping[str, Any], auditor: Auditor) -> None:
    """The three published counts: integer schema, and true to ``findings``.

    PASS policy is unchanged and is NOT re-decided here: a verdict passes iff it
    records zero BLOCKER and zero HIGH findings, so those two counts stay frozen
    at 0.  ``medium_count`` is deliberately NOT frozen -- a MEDIUM finding may be
    recorded on a PASS verdict -- but it is still an integer >= 0, and all three
    counts must agree with the findings list they summarise.
    """

    _require_exact_int(payload, "blocker_count", 0, auditor, "smoke_canary_audit_counts")
    _require_exact_int(payload, "high_count", 0, auditor, "smoke_canary_audit_counts")
    _require_nonnegative_int(payload, "medium_count", auditor,
                             "smoke_canary_audit_medium_count")

    findings = payload.get("findings")
    if not isinstance(findings, list) or not all(isinstance(f, Mapping) for f in findings):
        auditor.blocker("smoke_canary_audit_findings",
                        f"findings is not a list of objects: {findings!r}")
        return
    for severity, key in (("BLOCKER", "blocker_count"), ("HIGH", "high_count"),
                          ("MEDIUM", "medium_count")):
        reported = payload.get(key)
        if type(reported) is not int:
            continue          # already recorded as a type BLOCKER above
        recomputed = sum(1 for finding in findings if finding.get("severity") == severity)
        auditor.require(reported == recomputed, "smoke_canary_audit_count_consistency",
                        f"{key}={reported!r} but findings hold {recomputed} {severity}")


def audit_canary_audit_report(payload: Mapping[str, Any], auditor: Auditor) -> None:
    """Re-check the canary verdict when the final smoke audit runs."""

    auditor.require(payload.get("audit_version") == CANARY_AUDIT_VERSION,
                    "smoke_canary_audit_version", str(payload.get("audit_version")))
    auditor.require(payload.get("status") == "PASS", "smoke_canary_audit_status",
                    str(payload.get("status")))
    audit_canary_verdict_counts(payload, auditor)
    auditor.require(payload.get("approved_scientific_main_sha") == APPROVED_SCIENTIFIC_MAIN_SHA,
                    "smoke_canary_audit_baseline",
                    str(payload.get("approved_scientific_main_sha")))
    _require_frozen_protocol_hash(payload, auditor, "canary_audit.json")
    _require_exact_int(payload, "execution_issue", SMOKE_EXECUTION_ISSUE_NUMBER,
                       auditor, "smoke_canary_audit_execution_issue")
    _require_exact_int(payload, "protocol_origin_issue", SMOKE_PROTOCOL_ISSUE_NUMBER,
                       auditor, "smoke_canary_audit_protocol_issue")
    _require_exact_int(payload, "expected_canary_fits", EXPECTED_CANARY_FITS,
                       auditor, "smoke_canary_audit_fit_count")
    _require_exact_int(payload, "actual_canary_fits", EXPECTED_CANARY_FITS,
                       auditor, "smoke_canary_audit_fit_count")
    auditor.require(payload.get("canary_execution_mode") == "real",
                    "smoke_canary_audit_execution_mode",
                    str(payload.get("canary_execution_mode")))


def audit_smoke_run_dir(run_dir: Path, phase7e_dir: Path | None = None) -> Auditor:
    """Independent, fail-closed audit of one smoke execution directory."""

    auditor = Auditor()
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        auditor.blocker("smoke_run_dir_missing", str(run_dir))
        return auditor

    for name in SMOKE_AUDIT_INPUT_FILES:
        if not (run_dir / name).is_file():
            auditor.blocker("required_artifact_missing", f"smoke mode requires {name}")

    anchors = read_phase7e_anchor(auditor, phase7e_dir)

    authorization = _read_json(run_dir / "authorization.json", auditor)
    if authorization is not None:
        audit_smoke_authorization(authorization, auditor)

    canary = _read_json(run_dir / "canary.json", auditor)
    if canary is not None:
        audit_smoke_canary(canary, anchors, auditor)

    canary_audit = _read_json(run_dir / CANARY_AUDIT_FILENAME, auditor)
    if canary_audit is not None:
        audit_canary_audit_report(canary_audit, auditor)

    runinfo = _read_json(run_dir / "runinfo.json", auditor)
    if runinfo is not None:
        audit_smoke_runinfo(runinfo, auditor)

    rows = _read_csv(run_dir / "smoke_fit_results.csv", auditor)
    scores = audit_smoke_fit_rows(rows, anchors, auditor) if rows else {}

    summary = _read_json(run_dir / "smoke_summary.json", auditor)
    if summary is not None:
        audit_smoke_summary(summary, scores, auditor)
        audit_smoke_summary_provenance(summary, auditor)

    # Cross-file lineage: exactly one protocol hash and one run-code SHA.
    payloads = {name: payload for name, payload in
                (("authorization.json", authorization), ("canary.json", canary),
                 (CANARY_AUDIT_FILENAME, canary_audit),
                 ("runinfo.json", runinfo), ("smoke_summary.json", summary))
                if payload is not None}
    protocol_hashes = {p.get("protocol_hash") for p in payloads.values()}
    auditor.require(protocol_hashes == {EXPECTED_SMOKE_PROTOCOL_HASH},
                    "smoke_protocol_hash_lineage",
                    f"protocol hashes are not the single frozen value: {protocol_hashes}")
    if rows:
        csv_hashes = {row.get("protocol_hash") for row in rows}
        auditor.require(csv_hashes == {EXPECTED_SMOKE_PROTOCOL_HASH},
                        "smoke_protocol_hash_lineage",
                        f"CSV protocol hashes are not the frozen value: {csv_hashes}")

    audit_smoke_run_code_lineage(payloads, rows, auditor)

    unexpected = sorted(
        p.name for p in run_dir.iterdir()
        if p.name not in SMOKE_AUDIT_INPUT_FILES and p.name != "audit_report.json")
    auditor.require(not unexpected, "smoke_unexpected_artifact", str(unexpected))
    return auditor


def audit_smoke_run_code_lineage(payloads: Mapping[str, Mapping[str, Any]],
                                 rows: Sequence[Mapping[str, str]] | None,
                                 auditor: Auditor) -> None:
    """Every artifact must name exactly ONE run-code SHA.

    The canary and the smoke must have been produced by the same code: a canary
    from a different commit is not evidence about this execution, and a single
    drifted CSV row means the six fits did not come from one run.
    """

    observed: dict[str, str] = {}
    for name, payload in payloads.items():
        sha = payload.get("run_code_sha")
        if not _is_full_commit_sha(sha):
            auditor.blocker("smoke_run_code_sha_format", f"{name}: {sha!r}")
            continue
        observed[name] = sha
    if rows:
        for index, row in enumerate(rows, start=1):
            sha = row.get("run_code_sha")
            if not _is_full_commit_sha(sha):
                auditor.blocker("smoke_run_code_sha_format",
                                f"smoke_fit_results.csv row {index}: {sha!r}")
                continue
            observed[f"smoke_fit_results.csv row {index}"] = sha
    distinct = sorted(set(observed.values()))
    auditor.require(len(distinct) <= 1, "smoke_run_code_sha_lineage",
                    f"artifacts name different run-code SHAs: "
                    f"{ {name: sha for name, sha in observed.items()} }")
    # The run-code SHA is provenance, never approval: it must not be silently
    # substituted for the approved baseline.
    for name, payload in payloads.items():
        auditor.require(payload.get("approved_scientific_main_sha")
                        == APPROVED_SCIENTIFIC_MAIN_SHA,
                        "smoke_baseline_sha_lineage",
                        f"{name}: {payload.get('approved_scientific_main_sha')!r}")


def _is_full_commit_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 40
            and all(character in "0123456789abcdef" for character in value))


# ---------------------------------------------------------------------------
# durable audit evidence
# ---------------------------------------------------------------------------


AUDIT_REPORT_VERSION = "phase8b-smoke-audit-v1"
AUDIT_REPORT_FILENAME = "audit_report.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Temp file -> flush -> fsync -> atomic replace.  No partial file survives."""

    import os
    import tempfile

    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing audit report: {path}")
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write((text + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def build_smoke_audit_report(auditor: Auditor, run_dir: Path) -> dict[str, Any]:
    """Structured audit evidence.  Safe to build even for a malformed artifact set."""

    def field(name: str, key: str, default: Any = None) -> Any:
        path = Path(run_dir) / name
        if not path.is_file():
            return default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return default
        return payload.get(key, default) if isinstance(payload, dict) else default

    blockers = len(auditor.blockers)
    highs = len(auditor.highs)
    mediums = len([f for f in auditor.findings if f.severity == "MEDIUM"])
    audited = sorted(name for name in SMOKE_AUDIT_INPUT_FILES
                     if (Path(run_dir) / name).is_file())
    return {
        "audit_version": AUDIT_REPORT_VERSION,
        "approved_scientific_main_sha": APPROVED_SCIENTIFIC_MAIN_SHA,
        "run_code_sha": field("runinfo.json", "run_code_sha"),
        "protocol_hash": EXPECTED_SMOKE_PROTOCOL_HASH,
        "protocol_origin_issue": SMOKE_PROTOCOL_ISSUE_NUMBER,
        "execution_issue": SMOKE_EXECUTION_ISSUE_NUMBER,
        "status": "PASS" if blockers == 0 and highs == 0 else "FAIL",
        "blocker_count": blockers,
        "high_count": highs,
        "medium_count": mediums,
        "findings": [{"severity": f.severity, "check": f.check, "detail": f.detail}
                     for f in auditor.findings],
        "expected_canary_fits": EXPECTED_CANARY_FITS,
        "actual_canary_fits": field("runinfo.json", "actual_canary_fits"),
        "expected_smoke_fits": EXPECTED_SMOKE_FITS,
        "actual_smoke_fits": field("runinfo.json", "actual_smoke_fits"),
        "selected_k": field("smoke_summary.json", "selected_k"),
        "selected_k_interpretation": field("smoke_summary.json",
                                           "selected_k_interpretation"),
        # The audit recomputes selected_k but never compares it to K_TRUE.
        "k_recovery_evaluated": False,
        "audited_files": audited,
        "run_dir": str(run_dir),
    }


def write_smoke_audit_report(run_dir: Path, auditor: Auditor) -> Path:
    """Publish durable audit evidence exactly once.  An existing report is a stop."""

    directory = Path(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"audit run directory does not exist: {directory}")
    path = directory / AUDIT_REPORT_FILENAME
    if path.exists():
        raise FileExistsError(
            f"{AUDIT_REPORT_FILENAME} already exists; a previous audit verdict is "
            f"never overwritten: {path}")
    return _atomic_write_json(path, build_smoke_audit_report(auditor, directory))


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
    parser.add_argument("--estimand", choices=ESTIMANDS, default=None,
                        help="required for config/selection mode; unused by smoke mode")
    parser.add_argument("--mode", choices=[*sorted(AUDIT_MODES), "canary", "smoke"],
                        default="config")
    parser.add_argument("--phase7e-dir", type=Path, default=None)
    parser.add_argument("--write-report", action="store_true",
                        help="canary mode: publish canary_audit.json; smoke mode: publish "
                             "audit_report.json (neither is ever overwritten)")
    args = parser.parse_args(argv)
    if args.mode == "canary":
        auditor = audit_canary_run_dir(args.run_dir, args.phase7e_dir)
        report = _render(auditor, args.run_dir, SMOKE_ESTIMAND, "canary")
        if args.write_report:
            report["canary_audit"] = str(write_canary_audit_report(args.run_dir, auditor))
    elif args.mode == "smoke":
        auditor = audit_smoke_run_dir(args.run_dir, args.phase7e_dir)
        report = _render(auditor, args.run_dir, SMOKE_ESTIMAND, "smoke")
        if args.write_report:
            report["audit_report"] = str(write_smoke_audit_report(args.run_dir, auditor))
    else:
        if args.estimand is None:
            parser.error(f"--estimand is required for {args.mode} mode")
        auditor = audit_run_dir(args.run_dir, args.estimand, args.mode, args.phase7e_dir)
        report = _render(auditor, args.run_dir, args.estimand, args.mode)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
