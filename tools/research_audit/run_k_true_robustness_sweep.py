"""Phase 8b K_TRUE robustness harness (Issue #49).

Scientific source of truth
--------------------------
- ``reports/k_selection_theory/k_true_robustness_design_20260901.md``
- ``reports/k_selection_theory/k_true_robustness_implementation_plan_20260901.md``

Human Gate (frozen 2026-09-01, GitHub Issue #47)::

    H1 = A+B      H2 = CRN      H3 = H3-a      H4 = S-c

This module deliberately keeps EM behind explicit gates.  Importing it, and the
``--validate-only`` / ``--config-gate`` / ``--record-diagnostics`` command paths,
cannot import or call the EM runner: ``em_runner`` is imported only inside
``AuthorizedEMFitAdapter.fit`` of the Phase 7e module, which those paths never
reach.

The Phase 7e harness (``run_heldout_k_selection_pilot.py``) is imported for its
pure functions and leakage boundary and is **never modified**; reusing the same
implementation of the score, the selector and the fit boundary is the strongest
available guarantee that the protocol is identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPFAM_SRC = ROOT / "expfam" / "src"
EXPERIMENTAL = EXPFAM_SRC / "experimental"
RESEARCH_AUDIT = Path(__file__).resolve().parent
for _path in (str(RESEARCH_AUDIT), str(EXPERIMENTAL), str(EXPFAM_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Phase 7e reuse.  Importing this module does not import em_runner.
from run_heldout_k_selection_pilot import (  # noqa: E402
    CANARY_ATOL,
    CANARY_RTOL,
    AuthorizedEMFitAdapter,
    CanaryFitResult,
    CanaryInvarianceReport,
    CanaryPreflight,
    FitCallBoundary,
    FitPayload,
    FrozenFitConfig,
    FrozenScoreConfig,
    HarnessStop,
    PreparedTrainingData,
    ScoreOnlyTarget,
    SplitDiagnostics,
    SplitPlan,
    StartScore,
    TrainingYValues,
    _expected_test_pairs,
    _make_test_fit_adapter,
    _readonly_copy,
    _reject_forbidden_fit_objects,
    _require,
    _require_clean_smoke_fit,
    _run_two_canary_falsification,
    _store_smoke_fit,
    _TestAuthorizedFitAdapter,
    authorize_canary_preflight,
    build_fit_payload,
    frozen_score_config,
    heldout_raw_eta_pairs,
    make_pair_split,
    make_score_only_target,
    make_training_y_values,
    prepare_training_data,
    score_config_hash,
    score_heldout_bernoulli,
    select_k_from_two_starts,
    stable_array_hash,
    stable_config_hash,
    validate_pair_masks,
)

PHASE = "8b"
ISSUE = 49

# --- HUMAN GATE FROZEN (2026-09-01, GitHub Issue #47) ----------------------
# These are not placeholders.  This is the current executable configuration.
ESTIMANDS = "AB"  # H1
PRIMARY_ESTIMAND = "A"  # H3-a
SENSITIVITY_ESTIMAND = "B"  # H3-a
HIERARCHY = "H3_A"  # H3
RANDOM_DESIGN = "CRN"  # H2: governs data_seed / model_seed ONLY
MASK_DESIGN = "S_C"  # H4: governs split_seed / pair-index mask ONLY

# --- FROZEN EXPERIMENTAL FACTORS -------------------------------------------
FAMILY_X = "poisson"
FAMILY_Y = "bernoulli"
N_NODES = 75
N_FEATURES = 15
L_SAMPLES = 5
NUM_ITER = 8
TEST_RATIO = 0.20
NUMERICS_MODE = "consistent"
VAR_F = 5.0
UNIQ = 0.1
W0_TRUE = -1.0
W_REF = 1.5
K_REF = 3

NEW_K_TRUE = (1, 2, 4, 5)
ANCHOR_K_TRUE = 3
K_CANDIDATES = tuple(range(1, 8))
REPLICATES = (1, 2, 3)
START_LABELS = (1, 2)
TIE_TOLERANCE = np.float64(1e-12)

FITS_PER_ESTIMAND = len(NEW_K_TRUE) * len(REPLICATES) * len(K_CANDIDATES) * len(START_LABELS)
EXPECTED_NEW_FITS = FITS_PER_ESTIMAND * 2  # 336 (A + B)
CELLS_PER_ESTIMAND = len(NEW_K_TRUE) * len(REPLICATES)  # 12

# --- SEED BASES -------------------------------------------------------------
DATA_SEED_BASE = 51000
MODEL_SEED_BASE = 530000
SPLIT_SEED_BASE = 52000  # S_A / S_B only (NOT SELECTED; kept for regression)
ANCHOR_SPLIT_SEED_BASE = 42000  # S_C: intentional Phase 7e split-seed reuse
# Estimand offsets apply to data/model seeds only, and only under INDEPENDENT.
ESTIMAND_SEED_OFFSET = {"A": 0, "B": 5_000_000}

# --- PHASE 7E ANCHOR (READ-ONLY) -------------------------------------------
PHASE7E_DIR = ROOT / "expfam" / "results" / "k_selection" / "heldout_full_pilot_20260824"
PHASE7E_FIT_RESULTS = PHASE7E_DIR / "fit_results.csv"
PHASE7E_RUN_CODE_SHA = "b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a"
PHASE7E_ARTIFACT_DIR = "expfam/results/k_selection/heldout_full_pilot_20260824"
PHASE7E_SPLIT_SEED_BASE = 42000
PHASE7E_DATA_SEED_BASE = 41000
PHASE7E_MODEL_SEED_BASE = 43000

# --- ARTIFACT SCHEMA --------------------------------------------------------
# The canonical required mask-provenance semantic field set.  design §10.5,
# plan §3.4.0 / M1 / §4.1 and static test T12e all use exactly this set.
REQUIRED_MASK_PROVENANCE_FIELDS = (
    "split_mask_hash",
    "train_mask_hash",
    "mask_design",
    "mask_group_id",
    "anchor_mask_hash",
    "anchor_train_mask_hash",
    "intentional_seed_reuse",
)

MANIFEST_COLUMNS = (
    "fit_index",
    "estimand",
    "role",
    "K_TRUE",
    "replicate",
    "K",
    "start",
    "data_seed",
    "split_seed",
    "split_mask_hash",
    "train_mask_hash",
    "mask_design",
    "mask_group_id",
    "anchor_mask_hash",
    "anchor_train_mask_hash",
    "intentional_seed_reuse",
    "model_seed",
    "w0_true",
    "w_true",
)

MASK_PROVENANCE_COLUMNS = (
    "estimand",
    "role",
    "K_TRUE",
    "replicate",
    "split_seed",
    "split_mask_hash",
    "train_mask_hash",
    "mask_design",
    "mask_group_id",
    "anchor_mask_hash",
    "anchor_train_mask_hash",
    "intentional_seed_reuse",
    "anchor_match",
)

DIAGNOSTICS_COLUMNS = (
    "K_TRUE",
    "replicate",
    "estimand",
    "role",
    "w_true",
    "sample_sd_eta_y",
    "y_density",
    "conditional_entropy_bits",
    "oracle_mean_log_score",
    "eta_y_excess_kurtosis",
    "mean_sq_latent_norm",
    "f_frobenius_sq",
    "mean_loading_energy",
)

# The integrated matrix is the ONLY artifact where the Phase 7e anchor and new
# Phase 8b results coexist.  best_score / margin are deliberately absent so the
# schema cannot invite cross-K_TRUE score-level comparison (design §13).
SELECTION_MATRIX_COLUMNS = (
    "estimand",
    "role",
    "K_TRUE",
    "replicate",
    "selected_k",
    "signed_error",
    "abs_error",
    "label",
    "lineage",
    "run_code_sha",
    "artifact_dir",
)
SELECTION_MATRIX_FORBIDDEN_COLUMNS = ("best_score", "margin")

LINEAGE_ANCHOR = "phase7e_anchor"
LINEAGE_NEW = "phase8a_new"


# ===========================================================================
# Step 2 — estimand, role, seeds
# ===========================================================================


def resolve_w_true(estimand: str, k_true: int) -> float:
    """Option A / B are distinguished here and nowhere else."""

    _require(int(k_true) > 0, "k_true must be positive")
    if estimand == "A":
        return float(W_REF)
    if estimand == "B":
        # w_K^2 * K == W_REF^2 * K_REF  (variance matched at the ensemble level)
        return float(W_REF) * math.sqrt(float(K_REF) / float(k_true))
    raise HarnessStop(f"unknown estimand {estimand!r}; estimand is not frozen")


def resolve_role(estimand: str) -> str:
    """H3 is frozen to H3_A: A is primary, B is pre-registered sensitivity."""

    if estimand not in ("A", "B"):
        raise HarnessStop(f"unknown estimand {estimand!r}")
    if ESTIMANDS != "AB":
        return "single"
    if HIERARCHY == "H3_A":
        return {"A": "primary", "B": "sensitivity"}[estimand]
    if HIERARCHY == "H3_B":
        return {"A": "coequal_A", "B": "coequal_B"}[estimand]
    raise HarnessStop("A/B hierarchy is not frozen")


def active_estimands() -> tuple[str, ...]:
    if ESTIMANDS == "AB":
        return ("A", "B")
    if ESTIMANDS in ("A", "B"):
        return (ESTIMANDS,)
    raise HarnessStop("estimand set is not frozen")


def _estimand_offset(estimand: str, random_design: str) -> int:
    """H2 offset.  Zero under CRN; only ever applied to data/model seeds."""

    if random_design == "CRN":
        return 0
    if random_design == "INDEPENDENT":
        return ESTIMAND_SEED_OFFSET[estimand]
    raise HarnessStop(f"unknown random design {random_design!r}")


def expected_data_seed(k_true: int, replicate: int, estimand: str,
                       random_design: str | None = None) -> int:
    """H2-governed."""

    design = RANDOM_DESIGN if random_design is None else random_design
    return DATA_SEED_BASE + 100 * int(k_true) + int(replicate) + _estimand_offset(estimand, design)


def expected_model_seed(k_true: int, replicate: int, k: int, start: int, estimand: str,
                        random_design: str | None = None) -> int:
    """H2-governed."""

    design = RANDOM_DESIGN if random_design is None else random_design
    return (
        MODEL_SEED_BASE
        + 10000 * int(k_true)
        + 1000 * int(replicate)
        + 10 * int(k)
        + int(start)
        + _estimand_offset(estimand, design)
    )


def expected_split_seed(k_true: int, replicate: int) -> int:
    """H4-governed ONLY.

    This function deliberately takes no ``estimand`` argument and never reads
    ``RANDOM_DESIGN``.  H2 governs data/model RNG; H4 alone governs the
    pair-index mask.  Adding an estimand offset here would break the S_C anchor
    alignment for at least one estimand (design §10.7.1, plan §2.3).
    """

    if MASK_DESIGN == "S_A":
        return SPLIT_SEED_BASE + 100 * int(k_true) + int(replicate)
    if MASK_DESIGN == "S_B":
        return SPLIT_SEED_BASE + int(replicate)
    if MASK_DESIGN == "S_C":
        # Intentional reuse of the Phase 7e split seed: a pre-registered
        # common-mask design, not an accidental collision.
        return ANCHOR_SPLIT_SEED_BASE + int(replicate)
    raise HarnessStop("split variant is not frozen")


def mask_group_id(k_true: int, replicate: int) -> str:
    if MASK_DESIGN == "S_A":
        return f"k{int(k_true)}r{int(replicate)}"
    return f"r{int(replicate)}"


def intentional_seed_reuse() -> bool:
    return MASK_DESIGN in ("S_B", "S_C")


# ===========================================================================
# Step 3 — manifest
# ===========================================================================


@dataclass(frozen=True, slots=True)
class ManifestRow:
    fit_index: int
    estimand: str
    role: str
    k_true: int
    replicate: int
    k: int
    start: int
    data_seed: int
    split_seed: int
    split_mask_hash: str
    train_mask_hash: str
    mask_design: str
    mask_group_id: str
    anchor_mask_hash: str
    anchor_train_mask_hash: str
    intentional_seed_reuse: bool
    model_seed: int
    w0_true: float
    w_true: float

    def as_row(self) -> tuple[Any, ...]:
        return (
            self.fit_index, self.estimand, self.role, self.k_true, self.replicate,
            self.k, self.start, self.data_seed, self.split_seed, self.split_mask_hash,
            self.train_mask_hash, self.mask_design, self.mask_group_id,
            self.anchor_mask_hash, self.anchor_train_mask_hash,
            self.intentional_seed_reuse, self.model_seed, self.w0_true, self.w_true,
        )


def build_manifest(estimand: str, masks: Mapping[int, "SplitRecord"] | None = None,
                   anchors: Mapping[int, "AnchorMask"] | None = None) -> list[ManifestRow]:
    """Build the frozen 168-row manifest for one estimand.

    Order is K_TRUE -> replicate -> candidate K -> start.  ``K_TRUE = 3`` is the
    frozen Phase 7e anchor and is never present in a new manifest.
    """

    _require(estimand in ("A", "B"), "unknown estimand")
    role = resolve_role(estimand)
    masks = masks or {}
    anchors = anchors or {}
    rows: list[ManifestRow] = []
    index = 0
    for k_true in NEW_K_TRUE:
        _require(k_true != ANCHOR_K_TRUE, "anchor K_TRUE must not appear in a new manifest")
        for replicate in REPLICATES:
            split = masks.get(replicate)
            anchor = anchors.get(replicate)
            for k in K_CANDIDATES:
                for start in START_LABELS:
                    index += 1
                    rows.append(
                        ManifestRow(
                            fit_index=index,
                            estimand=estimand,
                            role=role,
                            k_true=int(k_true),
                            replicate=int(replicate),
                            k=int(k),
                            start=int(start),
                            data_seed=expected_data_seed(k_true, replicate, estimand),
                            split_seed=expected_split_seed(k_true, replicate),
                            split_mask_hash="" if split is None else split.split_mask_hash,
                            train_mask_hash="" if split is None else split.train_mask_hash,
                            mask_design=MASK_DESIGN,
                            mask_group_id=mask_group_id(k_true, replicate),
                            anchor_mask_hash="" if anchor is None else anchor.test_mask_hash,
                            anchor_train_mask_hash="" if anchor is None else anchor.train_mask_hash,
                            intentional_seed_reuse=intentional_seed_reuse(),
                            model_seed=expected_model_seed(k_true, replicate, k, start, estimand),
                            w0_true=float(W0_TRUE),
                            w_true=resolve_w_true(estimand, k_true),
                        )
                    )
    return rows


def validate_manifest(rows: Sequence[ManifestRow], estimand: str) -> None:
    _require(len(rows) == FITS_PER_ESTIMAND, f"manifest must contain exactly {FITS_PER_ESTIMAND} rows")
    expected_keys = {
        (int(kt), int(r), int(k), int(s))
        for kt in NEW_K_TRUE for r in REPLICATES for k in K_CANDIDATES for s in START_LABELS
    }
    keys = [(row.k_true, row.replicate, row.k, row.start) for row in rows]
    _require(len(keys) == len(set(keys)), "duplicate manifest key")
    _require(set(keys) == expected_keys, "manifest key set differs from the frozen set")
    _require(all(row.k_true != ANCHOR_K_TRUE for row in rows), "anchor K_TRUE present in new manifest")
    _require(tuple(keys) == tuple(sorted(keys)), "manifest order is not K_TRUE/replicate/K/start ascending")
    _require({row.estimand for row in rows} == {estimand}, "manifest estimand is inconsistent")
    _require({row.role for row in rows} == {resolve_role(estimand)}, "manifest role is inconsistent")

    seeds = [row.model_seed for row in rows]
    _require(len(seeds) == len(set(seeds)), "duplicate model seed")
    for row in rows:
        _require(row.data_seed == expected_data_seed(row.k_true, row.replicate, estimand),
                 "data seed violates the frozen convention")
        _require(row.split_seed == expected_split_seed(row.k_true, row.replicate),
                 "split seed violates the frozen convention")
        _require(
            row.model_seed == expected_model_seed(row.k_true, row.replicate, row.k, row.start, estimand),
            "model seed violates the frozen convention",
        )
        _require(row.mask_design == MASK_DESIGN, "mask_design differs from the frozen design")
        _require(abs(row.w_true - resolve_w_true(estimand, row.k_true)) <= 1e-12,
                 "manifest w_true differs from resolve_w_true")
        _require(row.w0_true == W0_TRUE, "manifest w0_true differs from the frozen value")


def check_seed_collisions(manifests: Mapping[str, Sequence[ManifestRow]]) -> dict[str, Any]:
    """Unintended collisions are forbidden for data/model seeds.

    The S_C split-seed reuse is a pre-registered common-mask design and is
    therefore reported separately rather than as a collision.
    """

    phase7e_data = {PHASE7E_DATA_SEED_BASE + r for r in REPLICATES}
    phase7e_split = {PHASE7E_SPLIT_SEED_BASE + r for r in REPLICATES}
    phase7e_model = {
        PHASE7E_MODEL_SEED_BASE + r * 1000 + k * 10 + s
        for r in REPLICATES for k in K_CANDIDATES for s in START_LABELS
    }

    data_seeds: set[int] = set()
    model_seeds: set[int] = set()
    split_seeds: set[int] = set()
    per_estimand_model: dict[str, list[int]] = {}
    per_estimand_data: dict[str, set[int]] = {}
    for estimand, rows in manifests.items():
        per_estimand_model[estimand] = [row.model_seed for row in rows]
        per_estimand_data[estimand] = {row.data_seed for row in rows}
        for row in rows:
            data_seeds.add(row.data_seed)
            model_seeds.add(row.model_seed)
            split_seeds.add(row.split_seed)

    # Uniqueness is required WITHIN an estimand.  Across estimands, H2 = CRN
    # deliberately makes the data/model seeds coincide: that is the definition
    # of a common-random-number design, not a collision.  Under INDEPENDENT the
    # estimand offset separates them instead.
    for estimand, seeds in per_estimand_model.items():
        _require(len(seeds) == len(set(seeds)), f"duplicate model seed within estimand {estimand}")

    shared_model = set.intersection(*(set(v) for v in per_estimand_model.values()))         if len(per_estimand_model) > 1 else set()
    shared_data = set.intersection(*per_estimand_data.values())         if len(per_estimand_data) > 1 else set()
    if len(manifests) > 1:
        if RANDOM_DESIGN == "CRN":
            _require(
                len(shared_model) == len(per_estimand_model[next(iter(per_estimand_model))]),
                "CRN requires the model seeds to correspond across estimands",
            )
            _require(bool(shared_data), "CRN requires the data seeds to correspond across estimands")
        elif RANDOM_DESIGN == "INDEPENDENT":
            _require(not shared_model, "INDEPENDENT requires disjoint model seed blocks")
            _require(not shared_data, "INDEPENDENT requires disjoint data seed blocks")

    _require(not (data_seeds & phase7e_data), "data seed collides with the Phase 7e block")
    _require(not (model_seeds & phase7e_model), "model seed collides with the Phase 7e block")
    _require(not (data_seeds & model_seeds), "data/model seed roles overlap")
    _require(not (data_seeds & split_seeds), "data/split seed roles overlap")
    _require(not (model_seeds & split_seeds), "model/split seed roles overlap")

    reused = sorted(split_seeds & phase7e_split)
    if MASK_DESIGN == "S_C":
        _require(bool(reused), "S_C must reuse the Phase 7e split seeds")
        _require(intentional_seed_reuse(), "S_C split-seed reuse must be recorded as intentional")
    else:
        _require(not reused, "only S_C may reuse the Phase 7e split seeds")
    return {
        "data_seed_distinct": len(data_seeds),
        "model_seed_distinct": len(model_seeds),
        "model_seeds_per_estimand": {e: len(v) for e, v in per_estimand_model.items()},
        "split_seed_distinct": len(split_seeds),
        "random_design": RANDOM_DESIGN,
        "cross_estimand_model_seed_shared": len(shared_model),
        "cross_estimand_sharing_is_intentional": RANDOM_DESIGN == "CRN",
        "phase7e_split_seed_reused": reused,
        "intentional_seed_reuse": intentional_seed_reuse(),
        "unintended_collisions": [],
    }


# ===========================================================================
# Step 4 — Phase 7e anchor reader + canonical mask contract
# ===========================================================================


@dataclass(frozen=True, slots=True)
class AnchorMask:
    """Read-only Phase 7e evidence.  Never regenerated, never rerun."""

    replicate: int
    test_mask_hash: str
    train_mask_hash: str
    source: str


def read_phase7e_anchor_masks(fit_results: Path | None = None) -> dict[int, AnchorMask]:
    """Read the frozen anchor mask hashes from the Phase 7e artifact.

    Phase 7e stores ``test_mask_hash`` and ``train_mask_hash``.  There is no
    ``split_mask_hash`` column; the canonical Phase 8b correspondence is fixed
    in :func:`canonical_hash_contract`.
    """

    path = PHASE7E_FIT_RESULTS if fit_results is None else Path(fit_results)
    _require(path.is_file(), f"Phase 7e anchor artifact is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        _require("test_mask_hash" in fieldnames, "Phase 7e artifact has no test_mask_hash column")
        _require("train_mask_hash" in fieldnames, "Phase 7e artifact has no train_mask_hash column")
        _require("split_mask_hash" not in fieldnames,
                 "unexpected split_mask_hash column in the Phase 7e artifact")
        seen: dict[int, tuple[str, str]] = {}
        for row in reader:
            replicate = int(row["replicate"])
            pair = (row["test_mask_hash"], row["train_mask_hash"])
            if replicate in seen:
                _require(seen[replicate] == pair,
                         f"Phase 7e replicate {replicate} has inconsistent mask hashes")
            else:
                seen[replicate] = pair
    _require(set(seen) == set(REPLICATES), "Phase 7e anchor replicates are incomplete")
    rel = path.relative_to(ROOT).as_posix()
    return {
        replicate: AnchorMask(replicate, test_hash, train_hash, rel)
        for replicate, (test_hash, train_hash) in seen.items()
    }


def canonical_hash_contract() -> dict[str, str]:
    """The frozen canonical mask-hash contract (plan §3.4.0, design §10.4)."""

    return {
        "split_mask_hash": "stable_array_hash(test_mask)",
        "train_mask_hash": "stable_array_hash(train_mask)",
        "anchor_mask_hash": "phase7e_stored_test_mask_hash",
        "anchor_train_mask_hash": "phase7e_stored_train_mask_hash",
        "canonical_object": "test_mask",
    }


def compute_split_mask_hash(test_mask: np.ndarray) -> str:
    """Canonical: the held-out pair-index mask is the test mask."""

    return stable_array_hash(_readonly_copy(test_mask, np.bool_))


def compute_train_mask_hash(train_mask: np.ndarray) -> str:
    return stable_array_hash(_readonly_copy(train_mask, np.bool_))


@dataclass(frozen=True, slots=True)
class SplitRecord:
    replicate: int
    split_seed: int
    train_mask: np.ndarray
    test_mask: np.ndarray
    split_mask_hash: str
    train_mask_hash: str
    diagnostics: SplitDiagnostics


def build_split_record(k_true: int, replicate: int) -> SplitRecord:
    """Generate one pair-index split.  ``make_pair_split`` never sees K_TRUE."""

    split_seed = expected_split_seed(k_true, replicate)
    train_mask, test_mask = make_pair_split(N_NODES, TEST_RATIO, split_seed)
    train_mask = _readonly_copy(train_mask, np.bool_)
    test_mask = _readonly_copy(test_mask, np.bool_)
    diagnostics = validate_pair_masks(train_mask, test_mask, _expected_test_pairs(N_NODES, TEST_RATIO))
    return SplitRecord(
        replicate=int(replicate),
        split_seed=int(split_seed),
        train_mask=train_mask,
        test_mask=test_mask,
        split_mask_hash=compute_split_mask_hash(test_mask),
        train_mask_hash=compute_train_mask_hash(train_mask),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True, slots=True)
class MaskProvenanceRow:
    estimand: str
    role: str
    k_true: int
    replicate: int
    split_seed: int
    split_mask_hash: str
    train_mask_hash: str
    mask_design: str
    mask_group_id: str
    anchor_mask_hash: str
    anchor_train_mask_hash: str
    intentional_seed_reuse: bool
    anchor_match: bool

    def as_row(self) -> tuple[Any, ...]:
        return (
            self.estimand, self.role, self.k_true, self.replicate, self.split_seed,
            self.split_mask_hash, self.train_mask_hash, self.mask_design, self.mask_group_id,
            self.anchor_mask_hash, self.anchor_train_mask_hash,
            self.intentional_seed_reuse, self.anchor_match,
        )


def build_mask_provenance(estimand: str, anchors: Mapping[int, AnchorMask]) -> list[MaskProvenanceRow]:
    """Exactly 12 rows per estimand: 4 new K_TRUE x 3 replicates.

    Phase 7e anchor rows themselves are never copied here; the anchor evidence
    is carried by ``anchor_mask_hash`` / ``anchor_train_mask_hash``.
    """

    role = resolve_role(estimand)
    rows: list[MaskProvenanceRow] = []
    for k_true in NEW_K_TRUE:
        for replicate in REPLICATES:
            split = build_split_record(k_true, replicate)
            anchor = anchors[replicate]
            match = (
                split.split_mask_hash == anchor.test_mask_hash
                and split.train_mask_hash == anchor.train_mask_hash
            )
            rows.append(
                MaskProvenanceRow(
                    estimand=estimand,
                    role=role,
                    k_true=int(k_true),
                    replicate=int(replicate),
                    split_seed=split.split_seed,
                    split_mask_hash=split.split_mask_hash,
                    train_mask_hash=split.train_mask_hash,
                    mask_design=MASK_DESIGN,
                    mask_group_id=mask_group_id(k_true, replicate),
                    anchor_mask_hash=anchor.test_mask_hash,
                    anchor_train_mask_hash=anchor.train_mask_hash,
                    intentional_seed_reuse=intentional_seed_reuse(),
                    anchor_match=bool(match),
                )
            )
    _require(len(rows) == CELLS_PER_ESTIMAND,
             f"mask provenance must contain exactly {CELLS_PER_ESTIMAND} rows per estimand")
    keys = [(r.estimand, r.k_true, r.replicate) for r in rows]
    _require(len(keys) == len(set(keys)), "duplicate mask provenance key")
    _require(all(r.k_true != ANCHOR_K_TRUE for r in rows),
             "anchor K_TRUE row must not appear in mask provenance")
    return rows


# ===========================================================================
# Step 5 — zero-EM configuration gates
# ===========================================================================


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    scope: str
    passed: bool
    detail: str


def _gate(results: list[GateResult], name: str, scope: str, condition: bool, detail: str) -> None:
    results.append(GateResult(name, scope, bool(condition), detail))
    _require(bool(condition), f"config gate {name} failed: {detail}")


def run_generator_gate(estimands: Sequence[str] | None = None) -> list[GateResult]:
    """G1-G5: deterministic / algebraic only.  No sample statistic is a gate."""

    results: list[GateResult] = []
    for estimand in (estimands or active_estimands()):
        if estimand == "B":
            for k_true in NEW_K_TRUE:
                expected = W_REF * math.sqrt(K_REF / k_true)
                _gate(results, "G1", f"B/K{k_true}",
                      resolve_w_true("B", k_true) == expected,
                      "w_K == W_REF*sqrt(K_REF/K_TRUE)")
                _gate(results, "G2", f"B/K{k_true}",
                      abs(resolve_w_true("B", k_true) ** 2 * k_true - W_REF ** 2 * K_REF) <= 1e-9,
                      "w_K^2 * K_TRUE == W_REF^2 * K_REF")
            _gate(results, "G3", "B", resolve_w_true("B", K_REF) == W_REF,
                  "resolve_w_true('B', 3) == 1.5 (anchor compatibility)")
        else:
            for k_true in NEW_K_TRUE:
                _gate(results, "G1p", f"A/K{k_true}", resolve_w_true("A", k_true) == W_REF,
                      "w == 1.5 for all K_TRUE (variance is intentionally K-dependent)")
            _gate(results, "G3p", "A", resolve_w_true("A", K_REF) == W_REF,
                  "resolve_w_true('A', 3) == 1.5 (anchor compatibility)")

    # G4: no unexpected branch in the generator parameter mapping.  Only k (the
    # manipulated factor) and w_true (the estimand rule) may vary across cells.
    frozen = frozen_generator_config()
    varying = {"k", "w_true"}
    for estimand in (estimands or active_estimands()):
        invariant = {
            tuple(sorted((key, value) for key, value in dict(sig).items() if key not in varying))
            for sig in (_generator_call_signature(estimand, k_true) for k_true in NEW_K_TRUE)
        }
        _gate(results, "G4", estimand, len(invariant) == 1,
              "generator arguments other than k and w_true are identical across K_TRUE")
        # The generator call must agree with what the manifest records.
        manifest = build_manifest(estimand)
        mismatched = [
            row for row in manifest
            if dict(_generator_call_signature(estimand, row.k_true))["w_true"] != row.w_true
            or dict(_generator_call_signature(estimand, row.k_true))["w0_true"] != row.w0_true
        ]
        _gate(results, "G4m", estimand, not mismatched,
              "generator call arguments match the manifest values for every cell")
    _gate(results, "G4c", "common",
          frozen["n"] == N_NODES and frozen["d"] == N_FEATURES
          and frozen["family_x"] == FAMILY_X and frozen["family_y"] == FAMILY_Y
          and frozen["numerics_mode"] == NUMERICS_MODE and frozen["w0_true"] == W0_TRUE
          and frozen["var_f"] == VAR_F and frozen["uniq"] == UNIQ,
          "frozen generator constants match the design")

    # G5: the manifest stores the expected w_K.
    for estimand in (estimands or active_estimands()):
        rows = build_manifest(estimand)
        _gate(results, "G5", estimand,
              all(abs(r.w_true - resolve_w_true(estimand, r.k_true)) <= 1e-12 for r in rows)
              and "w_true" in MANIFEST_COLUMNS,
              "manifest carries the expected w_true column")
    return results


def frozen_generator_config() -> dict[str, Any]:
    return {
        "n": N_NODES,
        "d": N_FEATURES,
        "family_x": FAMILY_X,
        "family_y": FAMILY_Y,
        "numerics_mode": NUMERICS_MODE,
        "w0_true": W0_TRUE,
        "var_f": VAR_F,
        "uniq": UNIQ,
        "L": L_SAMPLES,
        "num_iter": NUM_ITER,
        "test_ratio": TEST_RATIO,
    }


def _generator_call_signature(estimand: str, k_true: int) -> tuple[tuple[str, Any], ...]:
    payload = dict(frozen_generator_config())
    payload["k"] = int(k_true)
    payload["w_true"] = resolve_w_true(estimand, k_true)
    return tuple(sorted(payload.items()))


def run_mask_gate(anchors: Mapping[int, AnchorMask] | None = None,
                  estimands: Sequence[str] | None = None,
                  mask_design: str | None = None) -> list[GateResult]:
    """M0-M3 plus the design-specific MA / MB / MC checks.  EM fits = 0."""

    design = MASK_DESIGN if mask_design is None else mask_design
    anchors = read_phase7e_anchor_masks() if anchors is None else anchors
    results: list[GateResult] = []
    estimand_list = list(estimands or active_estimands())

    _gate(results, "M2", "common", design == MASK_DESIGN,
          "mask_design equals the frozen MASK_DESIGN")

    records: dict[tuple[int, int], SplitRecord] = {}
    for k_true in NEW_K_TRUE:
        for replicate in REPLICATES:
            record = build_split_record(k_true, replicate)
            records[(k_true, replicate)] = record
            _gate(results, "M0", f"K{k_true}/r{replicate}",
                  record.diagnostics.test_pairs == _expected_test_pairs(N_NODES, TEST_RATIO),
                  "validate_pair_masks passed with the expected test-pair count")
            _gate(results, "M3", f"K{k_true}/r{replicate}",
                  record.split_mask_hash == stable_array_hash(record.test_mask),
                  "split_mask_hash == stable_array_hash(test_mask)")

    for estimand in estimand_list:
        provenance = build_mask_provenance(estimand, anchors)
        present = set(MASK_PROVENANCE_COLUMNS) & set(REQUIRED_MASK_PROVENANCE_FIELDS)
        _gate(results, "M1", estimand, present == set(REQUIRED_MASK_PROVENANCE_FIELDS),
              "all 7 required mask provenance fields are present")

        if design == "S_A":
            hashes = [r.split_mask_hash for r in provenance]
            _gate(results, "MA1", estimand, len(hashes) == len(set(hashes)),
                  "S_A: split_mask_hash differs per (K_TRUE, replicate)")
            _gate(results, "MA2", estimand, not intentional_seed_reuse(),
                  "S_A: intentional_seed_reuse is False")
        elif design == "S_B":
            for replicate in REPLICATES:
                group = {r.split_mask_hash for r in provenance if r.replicate == replicate}
                _gate(results, "MB1", f"{estimand}/r{replicate}", len(group) == 1,
                      "S_B: new K_TRUE share one mask per replicate")
                _gate(results, "MB2", f"{estimand}/r{replicate}",
                      group.pop() != anchors[replicate].test_mask_hash,
                      "S_B: partial alignment only; the K3 anchor mask differs")
            _gate(results, "MB3", estimand, intentional_seed_reuse(),
                  "S_B: intentional_seed_reuse is True")
        elif design == "S_C":
            for row in provenance:
                anchor = anchors[row.replicate]
                _gate(results, "MC1", f"{estimand}/K{row.k_true}/r{row.replicate}",
                      row.split_mask_hash == anchor.test_mask_hash
                      and row.train_mask_hash == anchor.train_mask_hash,
                      "S_C: BOTH test and train mask hashes match the Phase 7e anchor")
                _gate(results, "MC3", f"{estimand}/K{row.k_true}/r{row.replicate}",
                      bool(row.anchor_mask_hash) and bool(row.anchor_train_mask_hash),
                      "anchor test/train hashes are stored on the row")
                _gate(results, "MC5", f"{estimand}/K{row.k_true}/r{row.replicate}",
                      row.anchor_match is True, "anchor_match is True")
            _gate(results, "MC2", estimand, intentional_seed_reuse(),
                  "S_C: intentional_seed_reuse is True")
            # MC4: the H2 setting must never relax the S_C requirement.
            _gate(results, "MC4", estimand,
                  all(expected_split_seed(kt, r) == ANCHOR_SPLIT_SEED_BASE + r
                      for kt in NEW_K_TRUE for r in REPLICATES),
                  "split seed is independent of RANDOM_DESIGN")
        else:
            raise HarnessStop(f"unknown mask design {design!r}")
    return results


# ===========================================================================
# Step 6 — RECORD ONLY diagnostics
# ===========================================================================
#
# These values are recorded and NEVER used as a pass/fail gate.  Option B keeps
# w_K^2 * K algebraically constant, but the sample sd(eta_Y) observed on a
# finite dataset fluctuates; thresholding a sample statistic would produce
# false failures on a correct generator (plan §3.2, design §12.3).


@dataclass(frozen=True, slots=True)
class DiagnosticRow:
    k_true: int
    replicate: int
    estimand: str
    role: str
    w_true: float
    sample_sd_eta_y: float
    y_density: float
    conditional_entropy_bits: float
    oracle_mean_log_score: float
    eta_y_excess_kurtosis: float
    mean_sq_latent_norm: float
    f_frobenius_sq: float
    mean_loading_energy: float

    def as_row(self) -> tuple[Any, ...]:
        return (
            self.k_true, self.replicate, self.estimand, self.role, self.w_true,
            self.sample_sd_eta_y, self.y_density, self.conditional_entropy_bits,
            self.oracle_mean_log_score, self.eta_y_excess_kurtosis,
            self.mean_sq_latent_norm, self.f_frobenius_sq, self.mean_loading_energy,
        )


def _generate_cell(estimand: str, k_true: int, replicate: int) -> dict[str, Any]:
    """Call the generator only.  This never touches EM."""

    from data_generator_expfam import generate_dual_data  # noqa: PLC0415

    return generate_dual_data(
        n=N_NODES,
        d=N_FEATURES,
        k=int(k_true),
        seed=expected_data_seed(k_true, replicate, estimand),
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
        var_f=VAR_F,
        uniq=UNIQ,
        w0_true=W0_TRUE,
        w_true=resolve_w_true(estimand, k_true),
    )


def build_diagnostics(estimand: str) -> list[DiagnosticRow]:
    """Exactly 12 RECORD ONLY rows per estimand.  No K_TRUE=3 row is created."""

    role = resolve_role(estimand)
    upper = np.triu(np.ones((N_NODES, N_NODES), dtype=bool), 1)
    rows: list[DiagnosticRow] = []
    for k_true in NEW_K_TRUE:
        for replicate in REPLICATES:
            data = _generate_cell(estimand, k_true, replicate)
            Z = np.asarray(data["Z"], dtype=np.float64)
            F = np.asarray(data["F"], dtype=np.float64)
            w_true = resolve_w_true(estimand, k_true)
            eta = W0_TRUE + w_true * (Z @ Z.T)[upper]
            prob = 1.0 / (1.0 + np.exp(-eta))
            clipped = np.clip(prob, 1e-12, 1.0 - 1e-12)
            entropy_bits = float(np.mean(
                -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
            ))
            oracle = float(np.mean(
                clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
            ))
            centred = eta - eta.mean()
            variance = float(np.mean(centred ** 2))
            kurtosis = float(np.mean(centred ** 4) / variance ** 2 - 3.0) if variance > 0 else float("nan")
            rows.append(
                DiagnosticRow(
                    k_true=int(k_true),
                    replicate=int(replicate),
                    estimand=estimand,
                    role=role,
                    w_true=float(w_true),
                    sample_sd_eta_y=float(eta.std()),
                    y_density=float(np.asarray(data["Y"])[upper].mean()),
                    conditional_entropy_bits=entropy_bits,
                    oracle_mean_log_score=oracle,
                    eta_y_excess_kurtosis=kurtosis,
                    mean_sq_latent_norm=float((Z ** 2).sum(axis=1).mean()),
                    f_frobenius_sq=float((F ** 2).sum()),
                    mean_loading_energy=float((F ** 2).sum() / k_true),
                )
            )
    validate_diagnostics(rows)
    return rows


def validate_diagnostics(rows: Sequence[DiagnosticRow]) -> None:
    """Structural contract only.  No diagnostic VALUE is ever a gate."""

    _require(len(rows) == CELLS_PER_ESTIMAND,
             f"diagnostics must contain exactly {CELLS_PER_ESTIMAND} rows per estimand")
    keys = [(r.k_true, r.replicate) for r in rows]
    _require(len(keys) == len(set(keys)), "duplicate diagnostics key")
    _require({r.k_true for r in rows} == set(NEW_K_TRUE), "diagnostics K_TRUE set is wrong")
    _require(all(r.k_true != ANCHOR_K_TRUE for r in rows),
             "K_TRUE=3 diagnostics must not be newly generated; Phase 7e is a frozen anchor")


# ===========================================================================
# Step 7 — artifact schema helpers
# ===========================================================================


def selection_matrix_columns() -> tuple[str, ...]:
    for forbidden in SELECTION_MATRIX_FORBIDDEN_COLUMNS:
        _require(forbidden not in SELECTION_MATRIX_COLUMNS,
                 f"{forbidden} must not appear in the integrated selection matrix")
    return SELECTION_MATRIX_COLUMNS


def build_selection_matrix_anchor_rows(estimand: str,
                                       fit_results: Path | None = None) -> list[tuple[Any, ...]]:
    """Read the K_TRUE=3 anchor selections from Phase 7e.  Never recomputed by EM."""

    path = PHASE7E_DIR / "replicate_selection.csv" if fit_results is None else Path(fit_results)
    _require(path.is_file(), f"Phase 7e selection artifact is missing: {path}")
    role = resolve_role(estimand)
    selected: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            replicate = int(row["replicate"])
            value = int(row["selected_k"])
            if replicate in selected:
                _require(selected[replicate] == value, "inconsistent anchor selected_k")
            else:
                selected[replicate] = value
    _require(set(selected) == set(REPLICATES), "anchor selections are incomplete")
    rows: list[tuple[Any, ...]] = []
    for replicate in REPLICATES:
        k_hat = selected[replicate]
        signed = k_hat - ANCHOR_K_TRUE
        rows.append((
            estimand, role, ANCHOR_K_TRUE, replicate, k_hat, signed, abs(signed),
            selection_label(signed), LINEAGE_ANCHOR, PHASE7E_RUN_CODE_SHA, PHASE7E_ARTIFACT_DIR,
        ))
    return rows


def selection_label(signed_error: int) -> str:
    if signed_error < 0:
        return "under"
    if signed_error > 0:
        return "over"
    return "exact"


def frozen_config() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "issue": ISSUE,
        "estimands": ESTIMANDS,
        "primary_estimand": PRIMARY_ESTIMAND,
        "sensitivity_estimand": SENSITIVITY_ESTIMAND,
        "hierarchy": HIERARCHY,
        "random_design": RANDOM_DESIGN,
        "mask_design": MASK_DESIGN,
        "new_k_true": list(NEW_K_TRUE),
        "anchor_k_true": ANCHOR_K_TRUE,
        "candidate_k": list(K_CANDIDATES),
        "replicates": list(REPLICATES),
        "starts": list(START_LABELS),
        "n": N_NODES,
        "d": N_FEATURES,
        "L": L_SAMPLES,
        "num_iter": NUM_ITER,
        "test_ratio": TEST_RATIO,
        "family_x": FAMILY_X,
        "family_y": FAMILY_Y,
        "numerics_mode": NUMERICS_MODE,
        "w0_true": W0_TRUE,
        "w_ref": W_REF,
        "k_ref": K_REF,
        "tie_tolerance": float(TIE_TOLERANCE),
        "roles": {estimand: resolve_role(estimand) for estimand in active_estimands()},
        "w_true": {
            estimand: {str(k): resolve_w_true(estimand, k) for k in NEW_K_TRUE}
            for estimand in active_estimands()
        },
        "canonical_hash_contract": canonical_hash_contract(),
        "score_config_hash": score_config_hash(frozen_score_config()),
    }


def frozen_config_hash() -> str:
    return stable_config_hash(json.loads(json.dumps(frozen_config(), sort_keys=True)))


# ===========================================================================
# Phase 7e write protection
# ===========================================================================


def require_not_phase7e_path(path: Path | str) -> Path:
    """Phase 7e artifacts are read-only reference evidence."""

    resolved = Path(path).resolve()
    protected = PHASE7E_DIR.resolve()
    if resolved == protected or protected in resolved.parents:
        raise HarnessStop(
            f"refusing to write inside the frozen Phase 7e artifact directory: {resolved}"
        )
    return resolved


EXPECTED_ARTIFACT_NAMES = (
    "manifest.csv",
    "fit_results.csv",
    "replicate_selection.csv",
    "cell_selection.csv",
    "aggregate_summary.csv",
    "config_gate.csv",
    "mask_provenance.csv",
    "diagnostics.csv",
    "k_true_selection_matrix.csv",
    "runinfo.json",
    "runinfo.md",
    "stdout.log",
)


def require_no_existing_artifacts(out_dir: Path) -> None:
    """Refuse to start a run into a directory that already holds artifacts."""

    directory = require_not_phase7e_path(out_dir)
    if not directory.is_dir():
        return
    present = sorted(
        path.name for path in directory.iterdir()
        if path.is_file() and path.name in EXPECTED_ARTIFACT_NAMES
    )
    _require(not present, f"output directory already contains artifacts: {present}")


def require_only_expected_artifacts(out_dir: Path) -> list[str]:
    """Reject any file the artifact contract does not declare."""

    directory = require_not_phase7e_path(out_dir)
    _require(directory.is_dir(), f"output directory does not exist: {directory}")
    present = sorted(path.name for path in directory.iterdir() if path.is_file())
    unexpected = [name for name in present if name not in EXPECTED_ARTIFACT_NAMES]
    _require(not unexpected, f"unexpected artifact in the output directory: {unexpected}")
    return present


def _write_csv(out_dir: Path, name: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    target = require_not_phase7e_path(Path(out_dir) / name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            values = list(row)
            _require(len(values) == len(header), f"{name}: row/header field count mismatch")
            writer.writerow(values)
    return target


def write_manifest_csv(out_dir: Path, rows: Sequence[ManifestRow]) -> Path:
    return _write_csv(out_dir, "manifest.csv", MANIFEST_COLUMNS, [r.as_row() for r in rows])


def write_mask_provenance_csv(out_dir: Path, rows: Sequence[MaskProvenanceRow]) -> Path:
    return _write_csv(out_dir, "mask_provenance.csv", MASK_PROVENANCE_COLUMNS,
                      [r.as_row() for r in rows])


def write_diagnostics_csv(out_dir: Path, rows: Sequence[DiagnosticRow]) -> Path:
    return _write_csv(out_dir, "diagnostics.csv", DIAGNOSTICS_COLUMNS, [r.as_row() for r in rows])


CONFIG_GATE_COLUMNS = ("gate", "scope", "passed", "detail")


def write_config_gate_csv(out_dir: Path, gates: Sequence[GateResult]) -> Path:
    return _write_csv(out_dir, "config_gate.csv", CONFIG_GATE_COLUMNS,
                      [(g.gate, g.scope, g.passed, g.detail) for g in gates])


def write_runinfo_json(out_dir: Path, payload: Mapping[str, Any]) -> Path:
    target = require_not_phase7e_path(Path(out_dir) / "runinfo.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), sort_keys=True, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    return target


def write_zero_em_artifacts(out_dir: Path, estimand: str) -> dict[str, str]:
    """Write every artifact the zero-EM stage can produce, for one estimand.

    This is the production writer path.  It never runs EM and never writes into
    the frozen Phase 7e directory.
    """

    _require(estimand in active_estimands(), "estimand is not in the frozen set")
    target = require_not_phase7e_path(Path(out_dir))
    anchors = read_phase7e_anchor_masks()
    splits = {r: build_split_record(NEW_K_TRUE[0], r) for r in REPLICATES}

    manifest = build_manifest(estimand, masks=splits, anchors=anchors)
    validate_manifest(manifest, estimand)
    gates = run_generator_gate((estimand,)) + run_mask_gate(anchors=anchors, estimands=(estimand,))
    provenance = build_mask_provenance(estimand, anchors)
    diagnostics = build_diagnostics(estimand)

    written = {
        "manifest.csv": str(write_manifest_csv(target, manifest)),
        "mask_provenance.csv": str(write_mask_provenance_csv(target, provenance)),
        "diagnostics.csv": str(write_diagnostics_csv(target, diagnostics)),
        "config_gate.csv": str(write_config_gate_csv(target, gates)),
    }
    written["runinfo.json"] = str(write_runinfo_json(target, {
        "phase": PHASE,
        "issue": ISSUE,
        "estimand": estimand,
        "role": resolve_role(estimand),
        "em_fits_executed": 0,
        "gate_count": len(gates),
        "random_design": RANDOM_DESIGN,
        "mask_design": MASK_DESIGN,
        "hierarchy": HIERARCHY,
        "frozen_config_hash": frozen_config_hash(),
        "canonical_hash_contract": canonical_hash_contract(),
    }))
    require_only_expected_artifacts(target)
    return written


# ===========================================================================
# Step 8 — CLI no-EM paths
# ===========================================================================


def run_validate_only() -> dict[str, Any]:
    """Deterministic static checks.  This path cannot reach EM."""

    manifests = {estimand: build_manifest(estimand) for estimand in active_estimands()}
    for estimand, rows in manifests.items():
        validate_manifest(rows, estimand)
    seed_report = check_seed_collisions(manifests)
    selection_matrix_columns()
    return {
        "mode": "validate-only",
        "em_fits_executed": 0,
        "phase": PHASE,
        "issue": ISSUE,
        "estimands": list(active_estimands()),
        "roles": {estimand: resolve_role(estimand) for estimand in active_estimands()},
        "manifest_rows_per_estimand": {e: len(r) for e, r in manifests.items()},
        "expected_new_fits": EXPECTED_NEW_FITS,
        "manifest_columns": list(MANIFEST_COLUMNS),
        "required_mask_provenance_fields": list(REQUIRED_MASK_PROVENANCE_FIELDS),
        "selection_matrix_columns": list(SELECTION_MATRIX_COLUMNS),
        "seeds": seed_report,
        "frozen_config_hash": frozen_config_hash(),
        "score_config_hash": score_config_hash(frozen_score_config()),
    }


def run_config_gate() -> dict[str, Any]:
    """G1-G5 and M0-M3 / MA / MB / MC1-MC5.  EM fits = 0."""

    anchors = read_phase7e_anchor_masks()
    generator_gates = run_generator_gate()
    mask_gates = run_mask_gate(anchors=anchors)
    gates = generator_gates + mask_gates
    return {
        "mode": "config-gate",
        "em_fits_executed": 0,
        "estimands": list(active_estimands()),
        "mask_design": MASK_DESIGN,
        "random_design": RANDOM_DESIGN,
        "canonical_hash_contract": canonical_hash_contract(),
        "anchor_source": {r: a.source for r, a in sorted(anchors.items())},
        "anchor_mask_hash": {r: a.test_mask_hash for r, a in sorted(anchors.items())},
        "anchor_train_mask_hash": {r: a.train_mask_hash for r, a in sorted(anchors.items())},
        "gate_count": len(gates),
        "gates_passed": sum(1 for g in gates if g.passed),
        "gate_names": sorted({g.gate for g in gates}),
        "all_passed": all(g.passed for g in gates),
        "frozen_config_hash": frozen_config_hash(),
    }


def run_record_diagnostics(out_dir: Path | None = None) -> dict[str, Any]:
    """RECORD ONLY measurements.  No value here can stop a run.  EM fits = 0."""

    payload: dict[str, Any] = {
        "mode": "record-diagnostics",
        "em_fits_executed": 0,
        "record_only": True,
        "blocking": False,
        "note": "diagnostics are RECORD ONLY; no sample statistic is a pass/fail gate",
        "estimands": list(active_estimands()),
        "rows_per_estimand": {},
        "diagnostics": {},
    }
    for estimand in active_estimands():
        rows = build_diagnostics(estimand)
        payload["rows_per_estimand"][estimand] = len(rows)
        payload["diagnostics"][estimand] = [asdict(row) for row in rows]
        if out_dir is not None:
            # Production writer path: emits the full zero-EM artifact set so the
            # independent audit can read exactly what the harness produces.
            payload.setdefault("written", {})[estimand] =                 write_zero_em_artifacts(Path(out_dir) / estimand, estimand)
    payload["k_true_values"] = sorted(NEW_K_TRUE)
    payload["anchor_k_true_rows_generated"] = 0
    return payload




# ===========================================================================
# Phase 8b S2 — direct pre-smoke leakage boundary (Issue #51)
# ===========================================================================
#
# S1 hard-stopped the fit path, so A01-A03 falsification was carried only by
# the reused Phase 7e boundary.  S2 makes the Phase 8b boundary directly
# testable WITHOUT opening the real EM path: an injectable adapter lets the
# adversarial tests prove that a leaking payload is refused *before* any fit
# call, and that a post-fit mask substitution is caught *before* scoring.
#
# This does NOT authorize smoke.  ``--smoke`` remains hard-stopped.


LEAKAGE_BOUNDARY_VERSION = "phase8b-leakage-boundary-v1"
MASKED_CANARY_VALUE = 0          # held-out dyads are filled with this constant


@dataclass(frozen=True, slots=True)
class Phase8bFitRequest:
    """Immutable, train-only fit request.

    There is deliberately **no field that can hold a scoring target**: the
    held-out true Y has no representation on this object at all.
    """

    estimand: str
    role: str
    k_true: int
    replicate: int
    k: int
    start: int
    data_seed: int
    split_seed: int
    model_seed: int
    w_true: float
    w0_true: float
    fit_payload: FitPayload
    fit_config: FrozenFitConfig
    pre_fit_test_mask_hash: str
    pre_fit_train_mask_hash: str
    anchor_mask_hash: str
    anchor_train_mask_hash: str
    frozen_config_hash: str
    score_config_hash: str
    boundary_version: str = LEAKAGE_BOUNDARY_VERSION

    def manifest_key(self) -> tuple[str, int, int, int, int]:
        return (self.estimand, self.k_true, self.replicate, self.k, self.start)


def build_fit_request(
    manifest_row: ManifestRow,
    training_values: TrainingYValues,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    anchor: AnchorMask,
) -> Phase8bFitRequest:
    """Assemble a train-only request.  The held-out Y is never an argument."""

    _require(type(training_values) is TrainingYValues, "fit request requires TrainingYValues")
    payload = build_fit_payload(training_values, train_mask, MASKED_CANARY_VALUE)
    return Phase8bFitRequest(
        estimand=manifest_row.estimand,
        role=manifest_row.role,
        k_true=manifest_row.k_true,
        replicate=manifest_row.replicate,
        k=manifest_row.k,
        start=manifest_row.start,
        data_seed=manifest_row.data_seed,
        split_seed=manifest_row.split_seed,
        model_seed=manifest_row.model_seed,
        w_true=manifest_row.w_true,
        w0_true=manifest_row.w0_true,
        fit_payload=payload,
        fit_config=FrozenFitConfig(
            family_x=FAMILY_X,
            family_y=FAMILY_Y,
            k_est=manifest_row.k,
            L=L_SAMPLES,
            num_iter=NUM_ITER,
            seed=manifest_row.model_seed,
            numerics_mode=NUMERICS_MODE,
        ),
        pre_fit_test_mask_hash=compute_split_mask_hash(test_mask),
        pre_fit_train_mask_hash=compute_train_mask_hash(train_mask),
        anchor_mask_hash=anchor.test_mask_hash,
        anchor_train_mask_hash=anchor.train_mask_hash,
        frozen_config_hash=frozen_config_hash(),
        score_config_hash=score_config_hash(frozen_score_config()),
    )


# --- sealed fake adapter (S2) ---------------------------------------------
#
# The Phase 8b fake adapter is *sealed*: it takes no callback, lambda, partial
# or closure, so no arbitrary code -- and in particular nothing that could
# capture a ScoreOnlyTarget -- can be injected into the fit call.  The only
# adversarial behaviour it can perform is one of a finite, explicit set of
# mask mutations applied to the live mask state during the fit.


class FakeMutationMode(Enum):
    """The complete set of adversarial behaviours a fake fit may perform."""

    NONE = "none"
    TEST_MASK = "test_mask"
    TRAIN_MASK = "train_mask"
    BOTH_MASKS = "both_masks"


@dataclass(slots=True)
class MutableMaskState:
    """The live masks the boundary hashes before AND after the fit.

    The boundary re-reads *this* object after the adapter returns, so a
    mutation performed inside ``fit`` is observable.  There is deliberately no
    caller-supplied post-fit mask argument: the post-fit hash can only come
    from the state the fit itself had access to.
    """

    test_mask: np.ndarray
    train_mask: np.ndarray


def _flip_one_upper_dyad(mask: np.ndarray) -> np.ndarray:
    """Flip a single symmetric off-diagonal entry of a pair mask."""

    flipped = np.array(mask, dtype=bool)
    i, j = 0, 1
    flipped[i, j] = not bool(flipped[i, j])
    flipped[j, i] = flipped[i, j]
    return flipped


SEALED_FAKE_ADAPTER_SLOTS = (
    "calls",
    "mutation_mode",
    "mask_state",
    "mutations_applied",
    "last_payload_hash",
    "last_k_est",
)

# Names that would give an adapter a route to the held-out outcome or to
# arbitrary injected code.  None of them may exist on an authorized adapter.
FORBIDDEN_ADAPTER_ATTRS = (
    "target", "score_target", "held_out_target", "test_values", "Y_full", "Y",
    "on_fit", "callback", "hook", "func", "fn", "closure",
)


class SealedFakeFitAdapter:
    """Test-only fit adapter with no arbitrary-code surface.

    It records how many times the boundary actually invoked a fit, so an
    adversarial test can assert ``calls == 0`` (refused before the fit) or
    ``calls == 1`` (refused after the fit, before any score).  It never
    imports or reaches ``em_runner``.
    """

    __slots__ = SEALED_FAKE_ADAPTER_SLOTS

    def __init__(self, mask_state: MutableMaskState,
                 mutation_mode: FakeMutationMode = FakeMutationMode.NONE) -> None:
        _require(type(mask_state) is MutableMaskState,
                 "sealed fake adapter requires a MutableMaskState")
        _require(type(mutation_mode) is FakeMutationMode,
                 "sealed fake adapter requires a FakeMutationMode")
        self.calls = 0
        self.mutation_mode = mutation_mode
        self.mask_state = mask_state
        self.mutations_applied = 0
        self.last_payload_hash: str | None = None
        self.last_k_est: int | None = None

    def fit(self, request: Phase8bFitRequest) -> dict[str, Any]:
        _require(type(request) is Phase8bFitRequest,
                 "sealed fake adapter requires a Phase8bFitRequest")
        self.calls += 1
        self.last_payload_hash = request.fit_payload.payload_hash
        self.last_k_est = request.k
        self._mutate()
        return {"fake": True, "k_est": request.k, "model_seed": request.model_seed}

    def _mutate(self) -> None:
        """The only adversarial action available: flip one dyad of a mask."""

        mode = self.mutation_mode
        if mode is FakeMutationMode.NONE:
            return
        state = self.mask_state
        if mode in (FakeMutationMode.TEST_MASK, FakeMutationMode.BOTH_MASKS):
            state.test_mask = _flip_one_upper_dyad(state.test_mask)
        if mode in (FakeMutationMode.TRAIN_MASK, FakeMutationMode.BOTH_MASKS):
            state.train_mask = _flip_one_upper_dyad(state.train_mask)
        self.mutations_applied += 1


def _require_adapter_authority(adapter: Any, score_target: ScoreOnlyTarget | None) -> None:
    """A02: only the sealed, callback-free fake adapter may receive a fit.

    Phase 8b S2 authorizes no real adapter at all, so an exact-type policy is
    the tightest boundary available: a subclass, a wrapper, or anything that
    captured a ``ScoreOnlyTarget`` is refused *before* the fit.  When the real
    adapter is eventually opened it needs its own human-gated authority model.
    """

    _require(type(adapter).__name__ != "AuthorizedEMFitAdapter",
             "the real EM adapter is not authorized in Phase 8b S2")
    _require(type(adapter) is SealedFakeFitAdapter,
             f"unauthorized fit adapter type: {type(adapter).__name__}")
    _require(tuple(SealedFakeFitAdapter.__slots__) == SEALED_FAKE_ADAPTER_SLOTS,
             "sealed fake adapter field schema changed")
    for name in FORBIDDEN_ADAPTER_ATTRS:
        _require(not hasattr(adapter, name), f"fit adapter exposes a forbidden field: {name}")
    _require(type(adapter.mutation_mode) is FakeMutationMode,
             "fake adapter mutation mode is unauthorized")
    _require(type(adapter.mask_state) is MutableMaskState,
             "fake adapter mask state is unauthorized")
    _require(type(adapter.calls) is int and adapter.calls >= 0,
             "fake adapter call counter is unauthorized")
    # Defence in depth: nothing held in a sealed slot may be (or alias) the
    # target.  The slot values are passed explicitly because the deep walker
    # cannot see through ``__slots__``.
    _reject_forbidden_fit_objects(
        [getattr(adapter, name, None) for name in SEALED_FAKE_ADAPTER_SLOTS],
        [score_target] if score_target is not None else [],
    )


def _require_adapter_state_binding(adapter: Any, mask_state: Any) -> None:
    """A03: the adapter must hold the very state the boundary monitors.

    Content hashing answers *whether* a mask changed; it cannot answer *whose*
    mask changed.  If the adapter mutates state A while the boundary re-hashes
    a same-content state B, a fit-time mutation is invisible.  So the binding
    is by identity -- ``is``, not ``np.array_equal`` -- and it is required
    before the fit, so a mismatched pair never reaches ``fit`` at all.

    This is a different question from the mask *content* semantics: putting
    freshly copied arrays inside one shared ``MutableMaskState`` stays legal.
    """

    _require(type(mask_state) is MutableMaskState,
             "boundary requires a MutableMaskState")
    adapter_state = getattr(adapter, "mask_state", None)
    _require(adapter_state is not None, "fit adapter exposes no mask state to bind")
    _require(type(adapter_state) is MutableMaskState,
             "fake adapter mask state is unauthorized")
    _require(adapter_state is mask_state,
             "adapter/boundary mask state mismatch: the fit adapter does not hold "
             "the mask state the boundary monitors")


def _require_train_only_payload(request: Phase8bFitRequest, test_mask: np.ndarray) -> None:
    """A01: prove the fit matrix carries no held-out outcome.

    The payload is rebuilt from the train mask with a constant canary at every
    held-out dyad, so any injected true Y at a held-out position changes both
    the payload hash and the held-out slice.
    """

    payload = request.fit_payload
    _require(type(payload) is FitPayload, "fit payload type is unauthorized")
    _require(payload.provenance_version == "masked-fit-matrix-v1",
             "fit payload provenance version changed")
    _require(payload.canary_value == MASKED_CANARY_VALUE, "fit payload canary changed")

    matrix = np.asarray(payload.Y_fit, dtype=np.float64)
    _require(matrix.shape == payload.expected_shape, "fit payload shape changed")
    _require(stable_array_hash(matrix) == payload.payload_hash, "fit payload hash mismatch")

    held_out = np.asarray(test_mask, dtype=bool)
    _require(held_out.shape == matrix.shape, "test mask shape does not match the fit payload")
    held_out_values = matrix[np.triu(held_out, 1)]
    _require(
        bool(np.all(held_out_values == float(MASKED_CANARY_VALUE))),
        "raw held-out Y reached the fit payload",
    )

    train = np.asarray(payload.train_mask, dtype=bool)
    _require(stable_array_hash(train) == payload.train_mask_hash, "fit payload train mask changed")
    _require(not np.any(train & held_out), "fit payload train mask overlaps the held-out mask")


def _require_no_score_target(request: Phase8bFitRequest,
                             score_target: ScoreOnlyTarget | None) -> None:
    """A02: no scoring object, and no alias of its values, may reach the fit."""

    for name in ("score_target", "target", "held_out_target", "test_values", "Y_full"):
        _require(not hasattr(request, name), f"fit request exposes a scoring field: {name}")
    targets = [score_target] if score_target is not None else []
    # Deep traversal: rejects the object itself, its values array and any alias
    # that shares memory with it, anywhere in the request graph.
    _reject_forbidden_fit_objects(request, targets)


def _require_manifest_binding(request: Phase8bFitRequest, manifest_row: ManifestRow) -> None:
    """Every scientific coordinate must match the frozen manifest row."""

    checks = (
        ("estimand", request.estimand, manifest_row.estimand),
        ("role", request.role, manifest_row.role),
        ("K_TRUE", request.k_true, manifest_row.k_true),
        ("replicate", request.replicate, manifest_row.replicate),
        ("K", request.k, manifest_row.k),
        ("start", request.start, manifest_row.start),
        ("data_seed", request.data_seed, manifest_row.data_seed),
        ("split_seed", request.split_seed, manifest_row.split_seed),
        ("model_seed", request.model_seed, manifest_row.model_seed),
        ("w_true", request.w_true, manifest_row.w_true),
        ("w0_true", request.w0_true, manifest_row.w0_true),
    )
    for name, actual, expected in checks:
        _require(actual == expected, f"fit request {name} does not match the manifest: "
                                     f"{actual!r} != {expected!r}")
    _require(type(request.fit_config) is FrozenFitConfig,
             "fit request fit_config type is unauthorized")
    _require(request.role == resolve_role(request.estimand), "fit request role is not the frozen role")
    _require(request.w_true == resolve_w_true(request.estimand, request.k_true),
             "fit request w_true does not match the frozen estimand rule")
    _require(request.fit_config.k_est == manifest_row.k, "fit config k_est does not match candidate K")
    _require(request.fit_config.seed == manifest_row.model_seed, "fit config seed is not the model seed")
    _require(request.frozen_config_hash == frozen_config_hash(), "frozen config hash changed")
    _require(request.score_config_hash == score_config_hash(frozen_score_config()),
             "score config hash changed")
    _require(request.boundary_version == LEAKAGE_BOUNDARY_VERSION, "leakage boundary version changed")


def _require_mask_hashes(request: Phase8bFitRequest, train_mask: np.ndarray,
                         test_mask: np.ndarray, label: str) -> None:
    """Both sides must match the pre-fit values AND the Phase 7e anchor."""

    test_hash = compute_split_mask_hash(test_mask)
    train_hash = compute_train_mask_hash(train_mask)
    _require(test_hash == request.pre_fit_test_mask_hash,
             f"{label}: test mask hash differs from the pre-fit value")
    _require(train_hash == request.pre_fit_train_mask_hash,
             f"{label}: train mask hash differs from the pre-fit value")
    _require(test_hash == request.anchor_mask_hash,
             f"{label}: test mask hash differs from the Phase 7e anchor")
    _require(train_hash == request.anchor_train_mask_hash,
             f"{label}: train mask hash differs from the Phase 7e anchor")


@dataclass(frozen=True, slots=True)
class LeakageGateReport:
    pre_fit_passed: bool
    post_fit_passed: bool
    fit_calls: int
    pre_fit_test_mask_hash: str
    pre_fit_train_mask_hash: str
    post_fit_test_mask_hash: str
    post_fit_train_mask_hash: str
    anchor_mask_hash: str
    anchor_train_mask_hash: str
    boundary_version: str


def _require_post_fit_masks(request: Phase8bFitRequest, mask_state: MutableMaskState) -> None:
    """A03: re-hash the LIVE mask state after the adapter returns.

    This is the guard that catches a mutation performed *inside* the fit: the
    hashes come from the same object the adapter held, not from a value the
    caller supplied afterwards.
    """

    _require(type(mask_state) is MutableMaskState, "boundary requires a MutableMaskState")
    _require_mask_hashes(request, mask_state.train_mask, mask_state.test_mask, "post-fit")


class Phase8bFitBoundary:
    """Fail-closed boundary around a single fit invocation.

    ``run`` refuses a leaking request or an unauthorized adapter BEFORE calling
    the adapter, and refuses a mask mutated during the fit AFTER the adapter
    returns but BEFORE any score is taken.
    """

    __slots__ = ("_adapter",)

    def __init__(self, adapter: Any) -> None:
        # Cheap structural check only.  The *authority* decision belongs to
        # ``check_pre_fit`` so that every adversarial case can be falsified
        # through the real ``run`` path rather than at construction time.
        _require(adapter is not None and hasattr(adapter, "fit"), "adapter must expose fit()")
        self._adapter = adapter

    def check_pre_fit(self, request: Phase8bFitRequest, manifest_row: ManifestRow,
                      mask_state: MutableMaskState,
                      score_target: ScoreOnlyTarget | None) -> None:
        _require(type(request) is Phase8bFitRequest, "boundary requires a Phase8bFitRequest")
        _require(type(manifest_row) is ManifestRow, "boundary requires a ManifestRow")
        _require(type(mask_state) is MutableMaskState, "boundary requires a MutableMaskState")
        # Leakage checks run first: a smuggled held-out outcome -- on the
        # adapter or in the request -- is the most severe condition and must
        # not be masked by an unrelated binding error.
        _require_adapter_authority(self._adapter, score_target)
        _require_adapter_state_binding(self._adapter, mask_state)
        _require_no_score_target(request, score_target)
        _require_train_only_payload(request, mask_state.test_mask)
        _require_mask_hashes(request, mask_state.train_mask, mask_state.test_mask, "pre-fit")
        _require_manifest_binding(request, manifest_row)

    def check_post_fit(self, request: Phase8bFitRequest, mask_state: MutableMaskState,
                       result: Any, score_target: ScoreOnlyTarget | None) -> None:
        _require_post_fit_masks(request, mask_state)
        _require_train_only_payload(request, mask_state.test_mask)
        _require_no_score_target(request, score_target)
        _require_adapter_authority(self._adapter, score_target)
        _require_adapter_state_binding(self._adapter, mask_state)
        if result is not None:
            _reject_forbidden_fit_objects(result, [score_target] if score_target is not None else [])

    def run(self, request: Phase8bFitRequest, manifest_row: ManifestRow,
            mask_state: MutableMaskState,
            score_target: ScoreOnlyTarget | None) -> tuple[Any, LeakageGateReport]:
        """Pre-fit gate -> exactly one adapter call -> post-fit gate.

        The caller cannot supply a post-fit mask: the boundary re-reads the
        same ``mask_state`` the adapter was able to touch.
        """

        self.check_pre_fit(request, manifest_row, mask_state, score_target)
        pre_test = compute_split_mask_hash(mask_state.test_mask)
        pre_train = compute_train_mask_hash(mask_state.train_mask)

        result = self._adapter.fit(request)

        self.check_post_fit(request, mask_state, result, score_target)
        return result, LeakageGateReport(
            pre_fit_passed=True,
            post_fit_passed=True,
            fit_calls=getattr(self._adapter, "calls", 1),
            pre_fit_test_mask_hash=pre_test,
            pre_fit_train_mask_hash=pre_train,
            post_fit_test_mask_hash=compute_split_mask_hash(mask_state.test_mask),
            post_fit_train_mask_hash=compute_train_mask_hash(mask_state.train_mask),
            anchor_mask_hash=request.anchor_mask_hash,
            anchor_train_mask_hash=request.anchor_train_mask_hash,
            boundary_version=LEAKAGE_BOUNDARY_VERSION,
        )


# ===========================================================================
# Smoke authorization gate (S2 adds the gate; it does NOT open smoke)
# ===========================================================================


SMOKE_GATE_NAMES = (
    "ZERO_EM_GATE_PASS",
    "LEAKAGE_GATE_PASS",
    "ANCHOR_MASK_GATE_PASS",
    "INDEPENDENT_REVIEW_PASS",
    "HUMAN_SMOKE_APPROVAL",
)

# Gates a human must grant.  Code must never set these to True on its own.
HUMAN_ONLY_SMOKE_GATES = ("INDEPENDENT_REVIEW_PASS", "HUMAN_SMOKE_APPROVAL")


@dataclass(frozen=True, slots=True)
class SmokeAuthorization:
    zero_em_gate_pass: bool = False
    leakage_gate_pass: bool = False
    anchor_mask_gate_pass: bool = False
    independent_review_pass: bool = False
    human_smoke_approval: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "ZERO_EM_GATE_PASS": self.zero_em_gate_pass,
            "LEAKAGE_GATE_PASS": self.leakage_gate_pass,
            "ANCHOR_MASK_GATE_PASS": self.anchor_mask_gate_pass,
            "INDEPENDENT_REVIEW_PASS": self.independent_review_pass,
            "HUMAN_SMOKE_APPROVAL": self.human_smoke_approval,
        }

    def missing(self) -> list[str]:
        return [name for name, granted in self.as_dict().items() if not granted]

    def authorized(self) -> bool:
        return not self.missing()


def current_smoke_authorization() -> SmokeAuthorization:
    """The machine-checkable gates may be computed; the human gates may not.

    ``INDEPENDENT_REVIEW_PASS`` and ``HUMAN_SMOKE_APPROVAL`` are always False
    here by construction: no code path in this repository may grant them.
    """

    zero_em = run_validate_only()["em_fits_executed"] == 0
    anchors = read_phase7e_anchor_masks()
    mask_gates = run_mask_gate(anchors=anchors)
    leakage = run_leakage_self_check()
    return SmokeAuthorization(
        zero_em_gate_pass=bool(zero_em),
        leakage_gate_pass=bool(leakage["all_passed"]),
        anchor_mask_gate_pass=all(g.passed for g in mask_gates),
        independent_review_pass=False,   # human-only
        human_smoke_approval=False,      # human-only
    )


# ===========================================================================
# Leakage self-check: direct adversarial falsification (Issue #51, HIGH-01)
# ===========================================================================
#
# LEAKAGE_GATE_PASS is derived from THIS battery, not from a positive control.
# Each case declares what the boundary must do -- reject or accept, with how
# many adapter calls and with which refusal reason -- and a case that does not
# behave exactly that way fails the whole gate.


@dataclass(frozen=True, slots=True)
class _AdversarialTargetHoldingAdapter:
    """Self-check fixture: an unauthorized adapter that captured the target.

    It must never receive a fit.  ``fit`` is deliberately functional (it counts
    the call and would hand back the held-out values) so that disabling the
    adapter authority guard makes the A02 case visibly fail instead of
    silently still passing.
    """

    score_target: Any
    calls: list[int]

    def fit(self, request: Phase8bFitRequest) -> dict[str, Any]:
        self.calls.append(1)
        return {"fake": True, "leaked": self.score_target.values}


@dataclass(frozen=True, slots=True)
class _SmuggledRequestWrapper:
    """Self-check fixture: a wrapper that tries to carry the scoring target."""

    inner: Phase8bFitRequest
    stowaway: Any


_LEAKAGE_FIXTURE_CACHE: dict[int, tuple[Any, ...]] = {}


def _leakage_self_check_fixture(index: int = 0):
    """A valid train-only request plus a FRESH mutable mask state.

    The immutable half is memoised; the mask state is rebuilt every call so one
    adversarial case cannot contaminate the next.
    """

    cached = _LEAKAGE_FIXTURE_CACHE.get(index)
    if cached is None:
        estimand = PRIMARY_ESTIMAND
        anchors = read_phase7e_anchor_masks()
        row = build_manifest(estimand)[index]
        split = build_split_record(row.k_true, row.replicate)
        data = _generate_cell(estimand, row.k_true, row.replicate)
        Y = _readonly_copy(data["Y"], np.float64)
        training = make_training_y_values(Y, split.train_mask)
        score_target = make_score_only_target(Y, split.test_mask)
        request = build_fit_request(row, training, split.train_mask, split.test_mask,
                                    anchors[row.replicate])
        cached = (row, split, Y, score_target, request)
        _LEAKAGE_FIXTURE_CACHE[index] = cached
    row, split, Y, score_target, request = cached
    state = MutableMaskState(
        test_mask=np.array(split.test_mask, dtype=bool),
        train_mask=np.array(split.train_mask, dtype=bool),
    )
    return row, split, Y, score_target, request, state


def _guarded_boundary_run(adapter: Any, request: Any, manifest_row: ManifestRow,
                          mask_state: MutableMaskState,
                          score_target: ScoreOnlyTarget | None) -> tuple[bool, str, int]:
    """Run one case through the real boundary; report (rejected, reason, calls)."""

    boundary = Phase8bFitBoundary(adapter)
    try:
        boundary.run(request, manifest_row, mask_state, score_target)
    except HarnessStop as stop:
        return True, str(stop), _adapter_calls(adapter)
    return False, "", _adapter_calls(adapter)


def _adapter_calls(adapter: Any) -> int:
    calls = getattr(adapter, "calls", 0)
    return len(calls) if isinstance(calls, list) else int(calls)


def _case_positive_control() -> tuple[bool, str, int]:
    row, _split, _Y, target, request, state = _leakage_self_check_fixture()
    adapter = SealedFakeFitAdapter(state)
    return _guarded_boundary_run(adapter, request, row, state, target)


def _case_raw_held_out_y() -> tuple[bool, str, int]:
    """A01: the real held-out outcomes are written into the fit matrix."""

    row, split, Y, target, request, state = _leakage_self_check_fixture()
    payload = request.fit_payload
    leaked = np.array(payload.Y_fit, dtype=np.float64)
    rows_i, cols_i = np.where(np.triu(split.test_mask, 1))
    leaked[rows_i, cols_i] = Y[rows_i, cols_i]
    leaked[cols_i, rows_i] = Y[rows_i, cols_i]
    if np.array_equal(leaked, payload.Y_fit):
        return False, "A01 injection was a no-op", 0
    tampered = replace(request, fit_payload=replace(
        payload, Y_fit=leaked, payload_hash=stable_array_hash(leaked)))
    adapter = SealedFakeFitAdapter(state)
    return _guarded_boundary_run(adapter, tampered, row, state, target)


def _case_malicious_adapter() -> tuple[bool, str, int]:
    """A02: the adapter itself holds the ScoreOnlyTarget."""

    row, _split, _Y, target, request, state = _leakage_self_check_fixture()
    adapter = _AdversarialTargetHoldingAdapter(score_target=target, calls=[])
    return _guarded_boundary_run(adapter, request, row, state, target)


def _case_smuggled_request() -> tuple[bool, str, int]:
    """A02: a wrapper request carries the ScoreOnlyTarget alongside the real one."""

    row, _split, _Y, target, request, state = _leakage_self_check_fixture()
    adapter = SealedFakeFitAdapter(state)
    smuggled = _SmuggledRequestWrapper(inner=request, stowaway=target)
    return _guarded_boundary_run(adapter, smuggled, row, state, target)


def _case_mask_mutation(mode: FakeMutationMode) -> tuple[bool, str, int]:
    """A03: the adapter mutates the live mask state during the fit."""

    row, _split, _Y, target, request, state = _leakage_self_check_fixture()
    adapter = SealedFakeFitAdapter(state, mode)
    rejected, reason, calls = _guarded_boundary_run(adapter, request, row, state, target)
    if adapter.mutations_applied != 1:
        return False, "the adapter did not mutate the mask during the fit", calls
    return rejected, reason, calls


def _case_test_mask_mutation() -> tuple[bool, str, int]:
    return _case_mask_mutation(FakeMutationMode.TEST_MASK)


def _case_train_mask_mutation() -> tuple[bool, str, int]:
    return _case_mask_mutation(FakeMutationMode.TRAIN_MASK)


def _case_both_mask_mutation() -> tuple[bool, str, int]:
    return _case_mask_mutation(FakeMutationMode.BOTH_MASKS)


def _case_adapter_state_binding() -> tuple[bool, str, int]:
    """A03: the adapter holds a same-content DECOY state, not the monitored one.

    Without the identity binding this attack is invisible: the adapter mutates
    its own state during the fit while the boundary re-hashes an untouched
    twin and reports post-fit PASS.
    """

    row, _split, _Y, target, request, state = _leakage_self_check_fixture()
    decoy = MutableMaskState(
        test_mask=np.array(state.test_mask, dtype=bool),
        train_mask=np.array(state.train_mask, dtype=bool),
    )
    if decoy is state or not np.array_equal(decoy.test_mask, state.test_mask):
        return False, "the decoy state was not a distinct same-content twin", 0
    # The adapter would mutate its own state; the boundary watches the decoy.
    adapter = SealedFakeFitAdapter(state, FakeMutationMode.TEST_MASK)
    rejected, reason, calls = _guarded_boundary_run(adapter, request, row, decoy, target)
    if adapter.mutations_applied != 0:
        return False, "the adapter mutated although the fit was refused", calls
    return rejected, reason, calls


def _case_anchor_pre_fit_binding() -> tuple[bool, str, int]:
    """The Phase 7e anchor binding must be checked before the fit."""

    row, _split, _Y, target, request, state = _leakage_self_check_fixture()
    tampered = replace(request, anchor_mask_hash="0" * 64)
    adapter = SealedFakeFitAdapter(state)
    return _guarded_boundary_run(adapter, tampered, row, state, target)


# (name, case, expect_rejected, expect_adapter_calls, expected reason fragment)
LEAKAGE_SELF_CHECK_CASES: tuple[tuple[str, Any, bool, int, str], ...] = (
    ("positive_control", _case_positive_control, False, 1, ""),
    ("A01_raw_held_out_y", _case_raw_held_out_y, True, 0,
     "raw held-out Y reached the fit payload"),
    ("A02_malicious_adapter", _case_malicious_adapter, True, 0,
     "unauthorized fit adapter type"),
    ("A02_smuggled_request", _case_smuggled_request, True, 0,
     "boundary requires a Phase8bFitRequest"),
    ("A03_test_mask_mutation", _case_test_mask_mutation, True, 1,
     "post-fit: test mask hash differs"),
    ("A03_train_mask_mutation", _case_train_mask_mutation, True, 1,
     "post-fit: train mask hash differs"),
    ("A03_both_mask_mutation", _case_both_mask_mutation, True, 1,
     "post-fit: test mask hash differs"),
    ("A03_adapter_state_binding", _case_adapter_state_binding, True, 0,
     "adapter/boundary mask state mismatch"),
    ("anchor_pre_fit_binding", _case_anchor_pre_fit_binding, True, 0,
     "differs from the Phase 7e anchor"),
)

LEAKAGE_SELF_CHECK_CASE_NAMES = tuple(case[0] for case in LEAKAGE_SELF_CHECK_CASES)


def run_leakage_self_check() -> dict[str, Any]:
    """Falsify every leakage guard directly.  Real EM fits = 0.

    ``passed`` on a case means *the attack was rejected as specified*, never
    that the attack succeeded.  ``all_passed`` is the AND over every declared
    case; a missing, erroring, wrongly-timed or wrongly-reasoned case makes it
    False.  The fake adapter is called several times across the battery, which
    is why the fake and real counters are reported separately.
    """

    cases: dict[str, dict[str, Any]] = {}
    fake_fit_calls_total = 0
    for name, case_fn, expect_rejected, expect_calls, expect_reason in LEAKAGE_SELF_CHECK_CASES:
        try:
            rejected, reason, calls = case_fn()
        except Exception as exc:  # fail closed: a broken case is a failed gate
            cases[name] = {
                "passed": False,
                "rejected": False,
                "adapter_calls": -1,
                "expected_rejected": expect_rejected,
                "expected_adapter_calls": expect_calls,
                "expected_reason": expect_reason,
                "reason": f"self-check case raised {type(exc).__name__}: {exc}",
            }
            continue
        fake_fit_calls_total += max(int(calls), 0)
        reason_ok = (expect_reason in reason) if expect_rejected else (reason == "")
        cases[name] = {
            "passed": bool(rejected == expect_rejected and calls == expect_calls and reason_ok),
            "rejected": bool(rejected),
            "adapter_calls": int(calls),
            "expected_rejected": expect_rejected,
            "expected_adapter_calls": expect_calls,
            "expected_reason": expect_reason,
            "reason": reason,
        }
    all_passed = (
        set(cases) == set(LEAKAGE_SELF_CHECK_CASE_NAMES)
        and all(case["passed"] for case in cases.values())
    )
    return {
        "mode": "leakage-gate",
        "all_passed": bool(all_passed),
        "cases": cases,
        "case_names": list(LEAKAGE_SELF_CHECK_CASE_NAMES),
        "fake_fit_calls_total": fake_fit_calls_total,
        "real_em_fits_executed": 0,
        "em_fits_executed": 0,
        "boundary_version": LEAKAGE_BOUNDARY_VERSION,
    }


# ===========================================================================
# Phase 8b S2b — human-gated real canary + frozen 6-fit smoke (Issue #53)
# ===========================================================================
#
# The production canary/smoke orchestration is implemented here, and a
# committed production ``SmokeExecutionAuthorization`` now exists (Issue #55):
# it records the explicitly reviewed and human-approved budget of exactly 2 real
# canary fits plus exactly 6 real smoke fits.  Being authorized is still not the
# same as running: every real execution has to clear the zero-EM/preflight gates,
# the run-code and baseline lineage, canary-before-smoke, and the independent
# canary audit -- and a human still has to issue the command.  The full 336-fit
# sweep lies outside this authorization entirely.  The zero-EM command paths
# never touch this section, so ``em_runner`` stays unimported (it is imported
# only inside the Phase 7e ``AuthorizedEMFitAdapter.fit``, which only an
# authorized run can reach), and the authorization-only implementation and
# review work has itself executed zero real fits.
#
# Phase 7e is reused, never modified: its sealed adapter, fit-call boundary,
# training-only data preparation, two-canary falsification, clean-fit gate and
# scorer are the same code the anchor ran under.


# Two DIFFERENT issues, deliberately kept apart (Issue #55 §8):
#   * the protocol was frozen and implemented under Issue #53, and that number
#     is part of the scientific protocol hash;
#   * the real-execution authorization lineage is Issue #55, and that number is
#     execution metadata that must NOT perturb the protocol hash.
SMOKE_PROTOCOL_ISSUE_NUMBER = 53
SMOKE_EXECUTION_ISSUE_NUMBER = 55
SMOKE_PROTOCOL_VERSION = "phase8b-smoke-protocol-v1"
SMOKE_AUTHORIZATION_VERSION = "phase8b-smoke-authorization-v1"

# --- APPROVED SCIENTIFIC BASELINE (frozen, Issue #55 §6) -------------------
# PR #54 merge commit: the reviewed S2b scientific/operational implementation.
# This is a committed literal.  It is never read from the CLI, the environment,
# a config file, the current branch, ``git rev-parse HEAD`` or the
# authorization record itself -- any of those would let the running code
# declare itself approved.
APPROVED_SCIENTIFIC_MAIN_SHA = "68c78e1191889609dead05ea5a9fb11525ce92e2"

# --- FROZEN SMOKE PROTOCOL (Issue #53, merged plan §7) ---------------------
# Operational pipeline smoke on the PRIMARY estimand only.  Running A+B would
# silently change the pre-registered budget from 6 fits to 12.  B is NOT
# dropped: it remains the pre-registered sensitivity estimand for the S3 full
# run (all 168 B rows).
SMOKE_ESTIMAND = "A"
SMOKE_ROLE = "primary"
SMOKE_K_TRUE = 1
SMOKE_REPLICATE = 1
SMOKE_K_CANDIDATES = (2, 3, 4)
SMOKE_STARTS = (1, 2)
EXPECTED_SMOKE_FITS = 6

# Dedicated smoke seed block: disjoint from Phase 7e and from the Phase 8 full
# seed space, so the 8 authorized executions consume no full-run cell.
SMOKE_DATA_SEED_BASE = 61000
SMOKE_MODEL_SEED_BASE = 630000

# --- FROZEN REAL CANARY ----------------------------------------------------
CANARY_K_EST = 1
CANARY_START = 1
EXPECTED_CANARY_FITS = 2

# Future human-authorized real-EM budget.  This branch executes 0.
EXPECTED_REAL_EM_BUDGET = EXPECTED_CANARY_FITS + EXPECTED_SMOKE_FITS  # 8

# --- FROZEN FULL SWEEP (Issue #59, S3) -------------------------------------
# The 336-fit sweep is a SEPARATE human gate from the smoke.  Nothing here
# widens the smoke authorization: the two records have different types, different
# private authority sentinels and different validators.
FULL_EXECUTION_ISSUE_NUMBER = 59
FULL_PROTOCOL_ORIGIN_ISSUE_NUMBER = 49
FULL_PROTOCOL_VERSION = "phase8b-full-protocol-v1"
FULL_AUTHORIZATION_VERSION = "phase8b-full-authorization-v1"
FULL_ARTIFACT_VERSION = "phase8b-full-artifact-v1"

# A + B, K_TRUE {1,2,4,5} x replicate {1,2,3} x K {1..7} x start {1,2}.
EXPECTED_FULL_FITS_PER_ESTIMAND = FITS_PER_ESTIMAND          # 168
EXPECTED_FULL_FITS = EXPECTED_NEW_FITS                       # 336
# The Phase 7e K_TRUE=3 anchor (42 fits) is REUSED, never re-executed.
EXPECTED_FULL_PHASE7E_RERUN_FITS = 0
# The integrated selection view spans K_TRUE {1,2,3,4,5}: the 4 newly executed
# values plus the READ-ONLY Phase 7e anchor.  The anchor contributes 42 unique
# fits that are referenced, never re-executed and never added to the 336.
FULL_K_TRUE_GRID = tuple(sorted(set(NEW_K_TRUE) | {ANCHOR_K_TRUE}))
FULL_SELECTION_MATRIX_ROWS = len(ESTIMANDS) * len(FULL_K_TRUE_GRID) * len(REPLICATES)  # 30
PHASE7E_ANCHOR_FIT_COUNT = 42

# --- FUTURE FULL ARTIFACT SCHEMA (definition only; nothing is written) -----
FULL_ARTIFACT_DIRNAME = "k_true_robustness_full_20260902"
FULL_ARTIFACT_DIR = ROOT / "expfam" / "results" / "k_selection" / FULL_ARTIFACT_DIRNAME
FULL_ARTIFACT_FILES = (
    "authorization.json",
    "manifest.csv",
    "mask_provenance.csv",
    "config_gate.csv",
    "leakage_gate.csv",
    "full_fit_results.csv",
    "selection_matrix.csv",
    "full_summary.json",
    "runinfo.json",
    "audit_report.json",
)
# Written ONLY when a run stops early.  Its presence is by itself proof that the
# directory holds partial evidence and can never be completed or reused.
FULL_FAILURE_FILENAME = "failure.json"
# audit_report.json is the audit OUTPUT and is never a required input.
FULL_AUDIT_INPUT_FILES = tuple(n for n in FULL_ARTIFACT_FILES if n != "audit_report.json")

FULL_LEAKAGE_GATE_COLUMNS = (
    "estimand", "role", "K_TRUE", "replicate", "K", "start",
    "pre_fit_test_mask_hash", "pre_fit_train_mask_hash",
    "post_fit_test_mask_hash", "post_fit_train_mask_hash",
    "anchor_mask_hash", "anchor_train_mask_hash",
    "pre_fit_passed", "post_fit_passed", "fit_boundary_status", "boundary_version",
)

FULL_FIT_RESULTS_COLUMNS = (
    "run_code_sha",
    "approved_scientific_main_sha",
    "protocol_hash",
    "fit_index",
    "estimand",
    "role",
    "K_TRUE",
    "replicate",
    "K",
    "start",
    "data_seed",
    "split_seed",
    "model_seed",
    "mask_design",
    "mask_group_id",
    "anchor_mask_hash",
    "anchor_train_mask_hash",
    "heldout_mean_log_score",
    "internal_retry",
    "warning_count",
    "q_failure",
    "nan_occurred",
    "finite_state",
    "real_full_fits_executed",
)


# --- FUTURE SMOKE ARTIFACT SCHEMA (definition only; nothing is written) ----
SMOKE_ARTIFACT_COLUMNS = (
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


def smoke_data_seed(k_true: int, replicate: int) -> int:
    """Dedicated smoke data seed.  Never reuses the full-run data seed block."""

    return SMOKE_DATA_SEED_BASE + 100 * int(k_true) + int(replicate)


def smoke_model_seed(k_true: int, replicate: int, k: int, start: int) -> int:
    """Dedicated smoke model seed.  Never reuses the full-run model seed block."""

    return (
        SMOKE_MODEL_SEED_BASE
        + 10000 * int(k_true)
        + 1000 * int(replicate)
        + 10 * int(k)
        + int(start)
    )


def smoke_split_seed(k_true: int, replicate: int) -> int:
    """H4-governed ONLY: the smoke split seed is the ordinary S_C split seed.

    No smoke / estimand / data / model offset is applied here.  Applying one
    would break the S_C anchor alignment, which is the whole point of the mask
    design, so this delegates to the single frozen split-seed rule.
    """

    return expected_split_seed(k_true, replicate)


SMOKE_SPLIT_SEED = 42001  # == smoke_split_seed(1, 1); asserted by a static test
CANARY_MODEL_SEED = 641011  # == smoke_model_seed(1, 1, 1, 1)


def phase7e_seed_space() -> dict[str, frozenset[int]]:
    """The Phase 7e data/model seed space, rebuilt from its frozen rules."""

    data = {PHASE7E_DATA_SEED_BASE + replicate for replicate in REPLICATES}
    model = {
        PHASE7E_MODEL_SEED_BASE + 1000 * replicate + 10 * k + start
        for replicate in REPLICATES
        for k in K_CANDIDATES
        for start in START_LABELS
    }
    return {"data": frozenset(data), "model": frozenset(model)}


def phase8_full_seed_space() -> dict[str, frozenset[int]]:
    """The Phase 8b full-run data/model seed space over both estimands."""

    data: set[int] = set()
    model: set[int] = set()
    for estimand in ("A", "B"):
        for k_true in NEW_K_TRUE:
            for replicate in REPLICATES:
                data.add(expected_data_seed(k_true, replicate, estimand))
                for k in K_CANDIDATES:
                    for start in START_LABELS:
                        model.add(expected_model_seed(k_true, replicate, k, start, estimand))
    return {"data": frozenset(data), "model": frozenset(model)}


def smoke_seed_space() -> dict[str, frozenset[int]]:
    """Every data/model seed the 2 canary + 6 smoke executions would consume."""

    data = {smoke_data_seed(SMOKE_K_TRUE, SMOKE_REPLICATE)}
    model = {
        smoke_model_seed(SMOKE_K_TRUE, SMOKE_REPLICATE, k, start)
        for k in SMOKE_K_CANDIDATES
        for start in SMOKE_STARTS
    }
    model.add(CANARY_MODEL_SEED)
    return {"data": frozenset(data), "model": frozenset(model)}


def check_smoke_seed_collisions() -> dict[str, Any]:
    """The smoke block must not intersect Phase 7e or Phase 8 full seeds.

    The split seed is deliberately excluded: under H4=S_C the smoke split seed
    IS the Phase 7e replicate-1 split seed, and that reuse is the design.
    """

    smoke = smoke_seed_space()
    report: dict[str, Any] = {"split_seed_excluded": True}
    for name, other in (("phase7e", phase7e_seed_space()),
                        ("phase8_full", phase8_full_seed_space())):
        for kind in ("data", "model"):
            overlap = sorted(smoke[kind] & other[kind])
            report[f"{name}_{kind}_overlap"] = overlap
            _require(not overlap,
                     f"smoke {kind} seeds collide with {name}: {overlap}")
    report["smoke_data_seeds"] = sorted(smoke["data"])
    report["smoke_model_seeds"] = sorted(smoke["model"])
    return report


# ---------------------------------------------------------------------------
# Frozen smoke protocol hash
# ---------------------------------------------------------------------------


def smoke_protocol_config() -> dict[str, Any]:
    """Everything the authorization is bound to.  Any change changes the hash."""

    return {
        "phase": PHASE,
        # The protocol-origin issue, not the execution issue: binding the
        # execution lineage in here would change the scientific protocol hash
        # for a purely administrative reason.
        "issue": SMOKE_PROTOCOL_ISSUE_NUMBER,
        "protocol_version": SMOKE_PROTOCOL_VERSION,
        "estimand": SMOKE_ESTIMAND,
        "role": SMOKE_ROLE,
        "k_true": SMOKE_K_TRUE,
        "replicate": SMOKE_REPLICATE,
        "k_candidates": list(SMOKE_K_CANDIDATES),
        "starts": list(SMOKE_STARTS),
        "data_seed_base": SMOKE_DATA_SEED_BASE,
        "model_seed_base": SMOKE_MODEL_SEED_BASE,
        "split_seed": SMOKE_SPLIT_SEED,
        "canary_k_est": CANARY_K_EST,
        "canary_start": CANARY_START,
        "canary_model_seed": CANARY_MODEL_SEED,
        "expected_canary_fits": EXPECTED_CANARY_FITS,
        "expected_smoke_fits": EXPECTED_SMOKE_FITS,
        "family_x": FAMILY_X,
        "family_y": FAMILY_Y,
        "n_nodes": N_NODES,
        "n_features": N_FEATURES,
        "L": L_SAMPLES,
        "num_iter": NUM_ITER,
        "test_ratio": TEST_RATIO,
        "numerics_mode": NUMERICS_MODE,
        "var_f": VAR_F,
        "uniq": UNIQ,
        "w0_true": W0_TRUE,
        "w_true": resolve_w_true(SMOKE_ESTIMAND, SMOKE_K_TRUE),
        "mask_design": MASK_DESIGN,
        "hierarchy": HIERARCHY,
        "random_design": RANDOM_DESIGN,
        "score_config_hash": score_config_hash(frozen_score_config()),
        "frozen_config_hash": frozen_config_hash(),
        "boundary_version": LEAKAGE_BOUNDARY_VERSION,
    }


def smoke_protocol_hash() -> str:
    return stable_config_hash(smoke_protocol_config())


# ---------------------------------------------------------------------------
# Smoke manifest: exactly six deterministic rows
# ---------------------------------------------------------------------------


def build_smoke_manifest(anchors: Mapping[int, AnchorMask] | None = None,
                         split: "SplitRecord | None" = None) -> list[ManifestRow]:
    """Build the frozen six smoke rows from the smoke seed rules directly.

    This is a smoke-specific builder on purpose: it never takes a full-run
    manifest and patches seeds afterwards, because that pattern makes a
    seed-block mistake invisible.
    """

    anchors = read_phase7e_anchor_masks() if anchors is None else anchors
    split = build_split_record(SMOKE_K_TRUE, SMOKE_REPLICATE) if split is None else split
    anchor = anchors[SMOKE_REPLICATE]
    _require(split.split_seed == SMOKE_SPLIT_SEED, "smoke split seed changed")

    rows: list[ManifestRow] = []
    for k in SMOKE_K_CANDIDATES:
        for start in SMOKE_STARTS:
            rows.append(ManifestRow(
                fit_index=len(rows) + 1,
                estimand=SMOKE_ESTIMAND,
                role=SMOKE_ROLE,
                k_true=SMOKE_K_TRUE,
                replicate=SMOKE_REPLICATE,
                k=int(k),
                start=int(start),
                data_seed=smoke_data_seed(SMOKE_K_TRUE, SMOKE_REPLICATE),
                split_seed=smoke_split_seed(SMOKE_K_TRUE, SMOKE_REPLICATE),
                split_mask_hash=split.split_mask_hash,
                train_mask_hash=split.train_mask_hash,
                mask_design=MASK_DESIGN,
                mask_group_id=mask_group_id(SMOKE_K_TRUE, SMOKE_REPLICATE),
                anchor_mask_hash=anchor.test_mask_hash,
                anchor_train_mask_hash=anchor.train_mask_hash,
                intentional_seed_reuse=intentional_seed_reuse(),
                model_seed=smoke_model_seed(SMOKE_K_TRUE, SMOKE_REPLICATE, k, start),
                w0_true=W0_TRUE,
                w_true=resolve_w_true(SMOKE_ESTIMAND, SMOKE_K_TRUE),
            ))
    validate_smoke_manifest(rows)
    return rows


def validate_smoke_manifest(rows: Sequence[ManifestRow]) -> None:
    """Exactly six rows in K -> start order, all on the frozen smoke cell."""

    _require(len(rows) == EXPECTED_SMOKE_FITS,
             f"smoke manifest must contain exactly {EXPECTED_SMOKE_FITS} rows")
    expected_keys = tuple((k, start) for k in SMOKE_K_CANDIDATES for start in SMOKE_STARTS)
    _require(tuple((row.k, row.start) for row in rows) == expected_keys,
             "smoke manifest key order changed")
    for index, row in enumerate(rows, start=1):
        _require(row.fit_index == index, "smoke manifest fit_index order changed")
        _require(row.estimand == SMOKE_ESTIMAND, "smoke estimand is not the frozen primary estimand")
        _require(row.role == SMOKE_ROLE, "smoke role changed")
        _require(row.role == resolve_role(row.estimand), "smoke role is not the H3-a role")
        _require(row.k_true == SMOKE_K_TRUE, "smoke K_TRUE changed")
        _require(row.replicate == SMOKE_REPLICATE, "smoke replicate changed")
        _require(row.data_seed == smoke_data_seed(row.k_true, row.replicate),
                 "smoke data seed is not the dedicated smoke seed")
        _require(row.split_seed == SMOKE_SPLIT_SEED, "smoke split seed changed")
        _require(row.model_seed == smoke_model_seed(row.k_true, row.replicate, row.k, row.start),
                 "smoke model seed is not the dedicated smoke seed")
        _require(row.w0_true == W0_TRUE, "smoke w0_true changed")
        _require(row.w_true == resolve_w_true(row.estimand, row.k_true), "smoke w_true changed")
        _require(row.mask_design == MASK_DESIGN, "smoke mask design changed")
        _require(row.intentional_seed_reuse is intentional_seed_reuse(),
                 "smoke intentional seed reuse flag changed")
        # S_C: the smoke masks must be the Phase 7e replicate-1 anchor masks.
        _require(row.split_mask_hash == row.anchor_mask_hash,
                 "smoke test mask does not match the Phase 7e anchor")
        _require(row.train_mask_hash == row.anchor_train_mask_hash,
                 "smoke train mask does not match the Phase 7e anchor")


# ---------------------------------------------------------------------------
# Execution authorization contract
# ---------------------------------------------------------------------------
#
# Two distinct authorities.  The production one is never handed out by any
# function in this repository; the test one is issued by a private factory that
# no CLI or production entry point references.

_SMOKE_EXECUTION_AUTHORITY = object()
_SMOKE_TEST_AUTHORITY = object()

SMOKE_SHA_LENGTH = 40


@dataclass(frozen=True, slots=True)
class SmokeExecutionAuthorization:
    """A committed record binding one reviewed main SHA to one frozen protocol.

    There is deliberately no public constructor path: ``_authority`` must be a
    module-private sentinel, so a CLI flag, an environment variable or a config
    file cannot fabricate one.
    """

    issue_number: int
    approved_main_sha: str
    protocol_hash: str
    estimand: str
    k_true: int
    replicate: int
    smoke_fit_count: int
    canary_fit_count: int
    data_seed_base: int
    model_seed_base: int
    split_seed: int
    independent_review_pass: bool
    human_smoke_approval: bool
    authorization_version: str
    _authority: Any = field(repr=False, compare=False, default=None)

    def is_test_only(self) -> bool:
        return self._authority is _SMOKE_TEST_AUTHORITY


def current_expected_smoke_main_sha() -> str | None:
    """The single trusted source of the reviewed scientific baseline SHA.

    Bound (Issue #55) to ``APPROVED_SCIENTIFIC_MAIN_SHA``: the PR #54 merge
    commit that carries the independently reviewed S2b implementation.

    The value is a committed literal.  It is deliberately NOT derived from the
    CLI, the environment, a config file, the current branch, ``git rev-parse
    HEAD`` or the authorization record -- each of those would let the running
    code declare itself approved.  ``run_code_sha`` is recorded separately as
    provenance and is never an approval source.

    Binding this SHA is not by itself the execution authorization: that is a
    separate committed record (see ``current_smoke_execution_authorization``),
    and this function stays the only trusted source of the baseline it is
    checked against.
    """

    return APPROVED_SCIENTIFIC_MAIN_SHA


# Test-only trusted SHA.  It is never the value production compares against:
# production reads ``current_expected_smoke_main_sha()`` and nothing else.
_TEST_EXPECTED_MAIN_SHA = "a" * SMOKE_SHA_LENGTH


def _require_full_commit_sha(value: Any, label: str) -> None:
    _require(type(value) is str and len(value) == SMOKE_SHA_LENGTH
             and all(character in "0123456789abcdef" for character in value),
             f"{label} is not a full lowercase commit SHA")


def trusted_main_sha_for(test_only: bool) -> str | None:
    """The trusted SHA for one execution mode.

    Production always reads ``current_expected_smoke_main_sha()``; the test-only
    constant is reachable only when the caller has already declared the
    test-only mode, and no production entry point ever does.
    """

    return _TEST_EXPECTED_MAIN_SHA if test_only else current_expected_smoke_main_sha()


def _validate_smoke_execution_authorization(authorization: Any, *,
                                            expected_main_sha: Any,
                                            authority: Any) -> None:
    """Internal validator.  ``expected_main_sha`` comes from a trusted source.

    The SHA identity gate runs before every protocol/count/human check so an
    unrelated field error can never mask a wrong-SHA authorization.
    """

    _require(type(authorization) is SmokeExecutionAuthorization,
             "real execution requires a SmokeExecutionAuthorization")
    _require(authorization._authority is authority,
             "smoke execution authorization provenance is unauthorized")

    # --- trusted reviewed-main SHA identity binding (Issue #53 HIGH-01) ----
    _require(expected_main_sha is not None,
             "no reviewed main SHA has been authorized for real smoke execution")
    _require_full_commit_sha(expected_main_sha, "trusted reviewed main SHA")
    _require_full_commit_sha(authorization.approved_main_sha,
                             "smoke authorization approved_main_sha")
    _require(authorization.approved_main_sha == expected_main_sha,
             "approved main SHA does not match the reviewed execution SHA")

    checks = (
        ("issue_number", authorization.issue_number, SMOKE_EXECUTION_ISSUE_NUMBER),
        ("protocol_hash", authorization.protocol_hash, smoke_protocol_hash()),
        ("estimand", authorization.estimand, SMOKE_ESTIMAND),
        ("k_true", authorization.k_true, SMOKE_K_TRUE),
        ("replicate", authorization.replicate, SMOKE_REPLICATE),
        ("smoke_fit_count", authorization.smoke_fit_count, EXPECTED_SMOKE_FITS),
        ("canary_fit_count", authorization.canary_fit_count, EXPECTED_CANARY_FITS),
        ("data_seed_base", authorization.data_seed_base, SMOKE_DATA_SEED_BASE),
        ("model_seed_base", authorization.model_seed_base, SMOKE_MODEL_SEED_BASE),
        ("split_seed", authorization.split_seed, SMOKE_SPLIT_SEED),
        ("authorization_version", authorization.authorization_version,
         SMOKE_AUTHORIZATION_VERSION),
    )
    # Exact type BEFORE value: Python equality alone would accept an equal
    # float or bool for a frozen integer (``1.0 == 1``, ``True == 1``), and this
    # record is the trust boundary that releases the real 2+6 execution.
    # ``type(...) is type(...)`` on purpose -- ``isinstance`` admits bool -- and
    # nothing is coerced, so an invalid value can never be normalised into a
    # valid one.
    for name, actual, expected in checks:
        _require(type(actual) is type(expected),
                 f"smoke authorization {name} is not a {type(expected).__name__}: "
                 f"{actual!r} ({type(actual).__name__})")
        _require(actual == expected,
                 f"smoke authorization {name} does not match the frozen protocol: "
                 f"{actual!r} != {expected!r}")
    _require(authorization.independent_review_pass is True,
             "smoke authorization is missing INDEPENDENT_REVIEW_PASS")
    _require(authorization.human_smoke_approval is True,
             "smoke authorization is missing HUMAN_SMOKE_APPROVAL")


def validate_smoke_execution_authorization(authorization: Any, *, test_only: bool) -> None:
    """Bind an authorization to the trusted reviewed-main SHA and the protocol.

    The expected SHA is never a parameter of any public entry point: it is
    fetched here from the trusted source, so a caller cannot supply the value
    it is going to be checked against.
    """

    _validate_smoke_execution_authorization(
        authorization,
        expected_main_sha=trusted_main_sha_for(test_only),
        authority=_SMOKE_TEST_AUTHORITY if test_only else _SMOKE_EXECUTION_AUTHORITY,
    )


def current_smoke_execution_authorization() -> SmokeExecutionAuthorization | None:
    """The committed execution authorization recorded for Issue #55.

    Every value below is a reviewed literal.  Nothing here is read from the CLI,
    the environment, a config file, the current branch, ``git rev-parse HEAD``,
    an artifact on disk, or from the frozen constants this record is validated
    against -- a record built out of the values it will be compared with could
    never disagree with them, which is exactly what the independent review of
    this record has to be able to detect.

    Scope, and nothing wider: 2 real canary fits, then 6 real smoke fits only if
    the independent canary audit passes, for a total real EM budget of 8.  It
    does NOT authorize the full 336-fit sweep, replacement fits, seed rescue,
    tolerance relaxation, a Phase 7e rerun, or any automatic continuation.

    Holding this record does not by itself run anything: every real execution
    still has to clear the full preflight (clean working tree, approved-baseline
    ancestry, leakage self-check, Phase 7e anchors) and, for the smoke, the
    independent canary audit -- and a human still has to issue the command.
    """

    return SmokeExecutionAuthorization(
        issue_number=55,
        approved_main_sha="68c78e1191889609dead05ea5a9fb11525ce92e2",
        protocol_hash="1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09",
        estimand="A",
        k_true=1,
        replicate=1,
        smoke_fit_count=6,
        canary_fit_count=2,
        data_seed_base=61000,
        model_seed_base=630000,
        split_seed=42001,
        independent_review_pass=True,
        human_smoke_approval=True,
        authorization_version="phase8b-smoke-authorization-v1",
        _authority=_SMOKE_EXECUTION_AUTHORITY,
    )


def _make_test_smoke_authorization(
    *,
    approved_main_sha: str = _TEST_EXPECTED_MAIN_SHA,
    **overrides: Any,
) -> SmokeExecutionAuthorization:
    """Static-test-only factory.  No CLI or production path references it."""

    fields: dict[str, Any] = {
        "issue_number": SMOKE_EXECUTION_ISSUE_NUMBER,
        "approved_main_sha": approved_main_sha,
        "protocol_hash": smoke_protocol_hash(),
        "estimand": SMOKE_ESTIMAND,
        "k_true": SMOKE_K_TRUE,
        "replicate": SMOKE_REPLICATE,
        "smoke_fit_count": EXPECTED_SMOKE_FITS,
        "canary_fit_count": EXPECTED_CANARY_FITS,
        "data_seed_base": SMOKE_DATA_SEED_BASE,
        "model_seed_base": SMOKE_MODEL_SEED_BASE,
        "split_seed": SMOKE_SPLIT_SEED,
        "independent_review_pass": True,
        "human_smoke_approval": True,
        "authorization_version": SMOKE_AUTHORIZATION_VERSION,
    }
    fields.update(overrides)
    return SmokeExecutionAuthorization(_authority=_SMOKE_TEST_AUTHORITY, **fields)


# ===========================================================================
# Phase 8b S3 -- full 336-fit execution authorization (Issue #59)
# ===========================================================================
#
# Deliberately a SEPARATE gate from the smoke.  A ``SmokeExecutionAuthorization``
# can never authorize ``--full``: the record type differs, the private authority
# sentinel differs, and the validator checks both.  Two independent human gates
# are therefore required before 336 real fits can run, and BOTH are absent here:
#
#   * ``current_expected_full_main_sha()``     -> None (no reviewed baseline yet)
#   * ``current_full_execution_authorization()`` -> None (no committed record yet)
#
# S3-A implements the schema, the validator, the zero-EM preflight and the
# independent audit contract.  It does NOT commit the record and executes 0 fits.

_FULL_EXECUTION_AUTHORITY = object()
_FULL_TEST_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class FullExecutionAuthorization:
    """A committed record binding one reviewed main SHA to the frozen 336-fit sweep.

    There is deliberately no public constructor path: ``_authority`` must be a
    module-private sentinel that no CLI flag, environment variable or config
    file can produce.  It is a DIFFERENT sentinel from the smoke one.
    """

    issue_number: int
    protocol_origin_issue_number: int
    approved_main_sha: str
    protocol_hash: str
    estimands: tuple[str, ...]
    k_true_grid: tuple[int, ...]
    candidate_k: tuple[int, ...]
    starts: tuple[int, ...]
    replicates: tuple[int, ...]
    fits_per_estimand: int
    total_fit_count: int
    data_seed_base: int
    model_seed_base: int
    anchor_split_seed_base: int
    mask_design: str
    random_design: str
    hierarchy: str
    independent_review_pass: bool
    human_full_approval: bool
    authorization_version: str
    _authority: Any = field(repr=False, compare=False, default=None)

    def is_test_only(self) -> bool:
        return self._authority is _FULL_TEST_AUTHORITY


def full_protocol_config() -> dict[str, Any]:
    """Everything a full authorization is bound to.  Any change changes the hash."""

    return {
        "phase": PHASE,
        # The protocol-origin issue, not the execution issue.
        "issue": FULL_PROTOCOL_ORIGIN_ISSUE_NUMBER,
        "protocol_version": FULL_PROTOCOL_VERSION,
        "estimands": list(ESTIMANDS),
        "hierarchy": HIERARCHY,
        "k_true_grid": list(NEW_K_TRUE),
        "anchor_k_true": ANCHOR_K_TRUE,
        "candidate_k": list(K_CANDIDATES),
        "starts": list(START_LABELS),
        "replicates": list(REPLICATES),
        "fits_per_estimand": EXPECTED_FULL_FITS_PER_ESTIMAND,
        "total_fit_count": EXPECTED_FULL_FITS,
        "data_seed_base": DATA_SEED_BASE,
        "model_seed_base": MODEL_SEED_BASE,
        "anchor_split_seed_base": ANCHOR_SPLIT_SEED_BASE,
        "estimand_seed_offset": dict(ESTIMAND_SEED_OFFSET),
        "family_x": FAMILY_X,
        "family_y": FAMILY_Y,
        "n_nodes": N_NODES,
        "n_features": N_FEATURES,
        "L": L_SAMPLES,
        "num_iter": NUM_ITER,
        "test_ratio": TEST_RATIO,
        "numerics_mode": NUMERICS_MODE,
        "var_f": VAR_F,
        "uniq": UNIQ,
        "w0_true": W0_TRUE,
        "w_ref": W_REF,
        "k_ref": K_REF,
        "mask_design": MASK_DESIGN,
        "random_design": RANDOM_DESIGN,
        "score_config_hash": score_config_hash(frozen_score_config()),
        "frozen_config_hash": frozen_config_hash(),
        "boundary_version": LEAKAGE_BOUNDARY_VERSION,
    }


def full_protocol_hash() -> str:
    return stable_config_hash(full_protocol_config())


def current_expected_full_main_sha() -> str | None:
    """The trusted reviewed baseline for a real FULL execution.

    None in this branch: no main SHA has been reviewed and approved for the
    336-fit sweep.  Like the smoke equivalent it is a committed literal source,
    never read from the CLI, the environment, a config file, the current branch
    or ``git rev-parse HEAD``.  It is deliberately NOT
    ``current_expected_smoke_main_sha()``: approving the 8-fit smoke baseline
    must not silently approve a 336-fit sweep.
    """

    return None


def trusted_full_main_sha_for(test_only: bool) -> str | None:
    return _FULL_TEST_EXPECTED_MAIN_SHA if test_only else current_expected_full_main_sha()


# Test-only trusted SHA for the full lineage.  Distinct from the smoke one so a
# test-only smoke lineage can never stand in for a test-only full lineage.
_FULL_TEST_EXPECTED_MAIN_SHA = "c" * SMOKE_SHA_LENGTH


def _validate_full_execution_authorization(authorization: Any, *,
                                           expected_main_sha: Any,
                                           authority: Any) -> None:
    """Internal validator for the 336-fit sweep.  Fails closed on everything.

    Type identity is required before value equality on every frozen field:
    ``1.0 == 1`` and ``True == 1`` are true in Python, and this record is the
    trust boundary that would release 336 real fits.
    """

    _require(type(authorization) is FullExecutionAuthorization,
             "real full execution requires a FullExecutionAuthorization")
    _require(authorization._authority is authority,
             "full execution authorization provenance is unauthorized")

    _require(expected_main_sha is not None,
             "no reviewed main SHA has been authorized for real full execution")
    _require_full_commit_sha(expected_main_sha, "trusted reviewed full main SHA")
    _require_full_commit_sha(authorization.approved_main_sha,
                             "full authorization approved_main_sha")
    _require(authorization.approved_main_sha == expected_main_sha,
             "approved main SHA does not match the reviewed full execution SHA")

    checks = (
        ("issue_number", authorization.issue_number, FULL_EXECUTION_ISSUE_NUMBER),
        ("protocol_origin_issue_number", authorization.protocol_origin_issue_number,
         FULL_PROTOCOL_ORIGIN_ISSUE_NUMBER),
        ("protocol_hash", authorization.protocol_hash, full_protocol_hash()),
        ("estimands", authorization.estimands, tuple(ESTIMANDS)),
        ("k_true_grid", authorization.k_true_grid, tuple(NEW_K_TRUE)),
        ("candidate_k", authorization.candidate_k, tuple(K_CANDIDATES)),
        ("starts", authorization.starts, tuple(START_LABELS)),
        ("replicates", authorization.replicates, tuple(REPLICATES)),
        ("fits_per_estimand", authorization.fits_per_estimand,
         EXPECTED_FULL_FITS_PER_ESTIMAND),
        ("total_fit_count", authorization.total_fit_count, EXPECTED_FULL_FITS),
        ("data_seed_base", authorization.data_seed_base, DATA_SEED_BASE),
        ("model_seed_base", authorization.model_seed_base, MODEL_SEED_BASE),
        ("anchor_split_seed_base", authorization.anchor_split_seed_base,
         ANCHOR_SPLIT_SEED_BASE),
        ("mask_design", authorization.mask_design, MASK_DESIGN),
        ("random_design", authorization.random_design, RANDOM_DESIGN),
        ("hierarchy", authorization.hierarchy, HIERARCHY),
        ("authorization_version", authorization.authorization_version,
         FULL_AUTHORIZATION_VERSION),
    )
    for name, actual, expected in checks:
        _require(type(actual) is type(expected),
                 f"full authorization {name} is not a {type(expected).__name__}: "
                 f"{actual!r} ({type(actual).__name__})")
        _require(actual == expected,
                 f"full authorization {name} does not match the frozen protocol: "
                 f"{actual!r} != {expected!r}")
    # The grid must multiply out to exactly the frozen budget; a record that
    # merely names 336 while carrying a different grid is refused.
    product = (len(authorization.k_true_grid) * len(authorization.replicates)
               * len(authorization.candidate_k) * len(authorization.starts))
    _require(product == authorization.fits_per_estimand,
             f"full authorization grid multiplies to {product}, not "
             f"{authorization.fits_per_estimand}")
    _require(authorization.fits_per_estimand * len(authorization.estimands)
             == authorization.total_fit_count,
             "full authorization per-estimand count does not multiply to the total")
    _require(ANCHOR_K_TRUE not in authorization.k_true_grid,
             "the Phase 7e anchor K_TRUE must never appear in the full grid")
    _require(authorization.independent_review_pass is True,
             "full authorization is missing INDEPENDENT_REVIEW_PASS")
    _require(authorization.human_full_approval is True,
             "full authorization is missing HUMAN_FULL_APPROVAL")


def validate_full_execution_authorization(authorization: Any, *, test_only: bool) -> None:
    """Bind a full authorization to the trusted reviewed-main SHA and the protocol."""

    _validate_full_execution_authorization(
        authorization,
        expected_main_sha=trusted_full_main_sha_for(test_only),
        authority=_FULL_TEST_AUTHORITY if test_only else _FULL_EXECUTION_AUTHORITY,
    )


def current_full_execution_authorization() -> FullExecutionAuthorization | None:
    """Always None in this branch.

    S3-A implements the schema, the validator, the zero-EM preflight and the
    independent audit contract for the 336-fit sweep -- not the record.  The
    approved main SHA cannot exist yet: it is the SHA of main AFTER this branch
    is reviewed and merged, so hard-coding this branch's own SHA here would be a
    self-signed approval.  A later execution Issue commits the record together
    with the two human gates.
    """

    return None


def _make_test_full_authorization(
    *,
    approved_main_sha: str = _FULL_TEST_EXPECTED_MAIN_SHA,
    **overrides: Any,
) -> FullExecutionAuthorization:
    """Static-test-only factory.  No CLI or production path references it."""

    fields: dict[str, Any] = {
        "issue_number": FULL_EXECUTION_ISSUE_NUMBER,
        "protocol_origin_issue_number": FULL_PROTOCOL_ORIGIN_ISSUE_NUMBER,
        "approved_main_sha": approved_main_sha,
        "protocol_hash": full_protocol_hash(),
        "estimands": tuple(ESTIMANDS),
        "k_true_grid": tuple(NEW_K_TRUE),
        "candidate_k": tuple(K_CANDIDATES),
        "starts": tuple(START_LABELS),
        "replicates": tuple(REPLICATES),
        "fits_per_estimand": EXPECTED_FULL_FITS_PER_ESTIMAND,
        "total_fit_count": EXPECTED_FULL_FITS,
        "data_seed_base": DATA_SEED_BASE,
        "model_seed_base": MODEL_SEED_BASE,
        "anchor_split_seed_base": ANCHOR_SPLIT_SEED_BASE,
        "mask_design": MASK_DESIGN,
        "random_design": RANDOM_DESIGN,
        "hierarchy": HIERARCHY,
        "independent_review_pass": True,
        "human_full_approval": True,
        "authorization_version": FULL_AUTHORIZATION_VERSION,
    }
    fields.update(overrides)
    return FullExecutionAuthorization(_authority=_FULL_TEST_AUTHORITY, **fields)


# ---------------------------------------------------------------------------
# Zero-EM full preflight (Issue #59 S3-A)
# ---------------------------------------------------------------------------


def build_full_manifests(masks: Mapping[int, "SplitRecord"] | None = None,
                         anchors: Mapping[int, AnchorMask] | None = None,
                         ) -> dict[str, list[ManifestRow]]:
    """The complete frozen sweep: one 168-row manifest per estimand."""

    return {estimand: build_manifest(estimand, masks=masks, anchors=anchors)
            for estimand in active_estimands()}


def validate_full_manifests(manifests: Mapping[str, Sequence[ManifestRow]]) -> dict[str, Any]:
    """Exactly 336 fits, exactly 168 per estimand, exactly the frozen grid.

    Fails closed: every deviation raises before a single fit could be requested.
    """

    _require(set(manifests) == set(active_estimands()),
             f"full manifests must cover exactly {list(active_estimands())}")
    per_estimand: dict[str, int] = {}
    for estimand, rows in manifests.items():
        validate_manifest(rows, estimand)
        _require(len(rows) == EXPECTED_FULL_FITS_PER_ESTIMAND,
                 f"estimand {estimand} manifest is not {EXPECTED_FULL_FITS_PER_ESTIMAND} rows")
        per_estimand[estimand] = len(rows)
    total = sum(per_estimand.values())
    _require(total == EXPECTED_FULL_FITS,
             f"full sweep must be exactly {EXPECTED_FULL_FITS} fits, got {total}")
    _require(sorted(set(per_estimand.values())) == [EXPECTED_FULL_FITS_PER_ESTIMAND],
             f"A/B split is not {EXPECTED_FULL_FITS_PER_ESTIMAND}/"
             f"{EXPECTED_FULL_FITS_PER_ESTIMAND}: {per_estimand}")
    for estimand, rows in manifests.items():
        _require(all(row.k_true != ANCHOR_K_TRUE for row in rows),
                 "the Phase 7e anchor K_TRUE must never be re-executed")
    return {"fits_per_estimand": per_estimand, "total_fits": total}


def check_full_anchor_agreement(anchors: Mapping[int, AnchorMask] | None = None,
                                ) -> dict[str, Any]:
    """S_C: every full-run split must reproduce the Phase 7e anchor masks.

    Zero EM: the masks are rebuilt from the frozen split rule and compared with
    the read-only Phase 7e anchor on BOTH the test and the train side.
    """

    anchors = read_phase7e_anchor_masks() if anchors is None else anchors
    _require(set(anchors) == set(REPLICATES),
             f"Phase 7e anchors must cover replicates {list(REPLICATES)}")
    checked = 0
    mismatches: list[str] = []
    for k_true in NEW_K_TRUE:
        for replicate in REPLICATES:
            record = build_split_record(k_true, replicate)
            anchor = anchors[replicate]
            checked += 1
            if record.split_mask_hash != anchor.test_mask_hash:
                mismatches.append(f"K{k_true}/r{replicate}: test mask hash")
            if record.train_mask_hash != anchor.train_mask_hash:
                mismatches.append(f"K{k_true}/r{replicate}: train mask hash")
            if record.split_seed != ANCHOR_SPLIT_SEED_BASE + replicate:
                mismatches.append(f"K{k_true}/r{replicate}: split seed")
    _require(not mismatches, f"S_C anchor agreement failed: {mismatches}")
    return {
        "mask_design": MASK_DESIGN,
        "cells_checked": checked,
        "replicates": list(REPLICATES),
        "anchor_dir": str(PHASE7E_DIR),
        "phase7e_rerun_fits": EXPECTED_FULL_PHASE7E_RERUN_FITS,
        "mismatches": mismatches,
    }


def run_full_preflight() -> dict[str, Any]:
    """Every zero-EM gate the 336-fit sweep must clear.  EM fits = 0.

    Fail-closed by construction: each helper raises ``HarnessStop`` instead of
    returning a soft verdict, so a caller cannot proceed past a failed gate.
    """

    anchors = read_phase7e_anchor_masks()
    masks = {r: build_split_record(NEW_K_TRUE[0], r) for r in REPLICATES}
    manifests = build_full_manifests(masks=masks, anchors=anchors)
    manifest_report = validate_full_manifests(manifests)
    anchor_report = check_full_anchor_agreement(anchors)
    seed_report = check_seed_collisions(manifests)

    gates = run_mask_gate(anchors=anchors)
    failed = [f"{g.name}/{g.scope}" for g in gates if not g.passed]
    _require(not failed, f"mask gate failed: {failed}")

    validate_only = run_validate_only()
    config_gate = run_config_gate()
    leakage = run_leakage_self_check()
    _require(validate_only["em_fits_executed"] == 0, "validate-only executed EM")
    _require(config_gate["em_fits_executed"] == 0, "config gate executed EM")
    _require(leakage["all_passed"] is True, "leakage self-check did not pass")
    _require(leakage["real_em_fits_executed"] == 0, "leakage self-check executed real EM")

    return {
        "mode": "full-preflight",
        "em_fits_executed": 0,
        "real_full_fits_executed": 0,
        "issue": FULL_EXECUTION_ISSUE_NUMBER,
        "protocol_origin_issue": FULL_PROTOCOL_ORIGIN_ISSUE_NUMBER,
        "protocol_hash": full_protocol_hash(),
        "protocol_version": FULL_PROTOCOL_VERSION,
        "authorization_version": FULL_AUTHORIZATION_VERSION,
        "expected_full_fits": EXPECTED_FULL_FITS,
        "expected_full_fits_per_estimand": EXPECTED_FULL_FITS_PER_ESTIMAND,
        "manifest": manifest_report,
        "anchor_agreement": anchor_report,
        "seed_collisions": seed_report,
        "mask_gate_total": len(gates),
        "mask_gate_failed": failed,
        "leakage_gate_cases": len(leakage["cases"]),
        "leakage_boundary_version": leakage["boundary_version"],
        "hierarchy": HIERARCHY,
        "mask_design": MASK_DESIGN,
        "random_design": RANDOM_DESIGN,
        "estimands": list(active_estimands()),
        "artifact_directory": str(FULL_ARTIFACT_DIR),
        "artifact_directory_exists": FULL_ARTIFACT_DIR.exists(),
        "artifact_files": list(FULL_ARTIFACT_FILES),
        "full_fit_results_columns": list(FULL_FIT_RESULTS_COLUMNS),
        "trusted_full_main_sha_present": current_expected_full_main_sha() is not None,
        "full_execution_authorization_present":
            current_full_execution_authorization() is not None,
        "phase7e_rerun_fits": EXPECTED_FULL_PHASE7E_RERUN_FITS,
        "phase7e_anchor_dir": str(PHASE7E_DIR),
    }


# ---------------------------------------------------------------------------
# Full 336-fit executor (Issue #59 S3-A)
# ---------------------------------------------------------------------------
#
# PARTIAL FAILURE POLICY (frozen).  If fit N of 336 is not clean:
#
#   1. the run stops immediately at that fit -- nothing after it is attempted;
#   2. NO replacement fit, NO retry, NO alternative seed, NO relaxed tolerance;
#   3. the partial evidence already on disk is PRESERVED, never deleted;
#   4. ``failure.json`` records which fit stopped the run and why;
#   5. no ``full_summary.json``, no ``selection_matrix.csv`` and no completed
#      ``runinfo.json`` are produced, so no audit can return PASS;
#   6. the frozen artifact directory now exists, so the SAME authorization can
#      never start a second run -- a new execution needs a new human gate.

FULL_PARTIAL_FAILURE_POLICY = (
    "stop_immediately",
    "no_replacement_fit",
    "no_retry",
    "no_seed_rescue",
    "no_tolerance_change",
    "preserve_partial_evidence",
    "no_completed_summary",
    "no_audit_pass",
    "rerun_requires_a_new_human_gate",
)


def full_fit_config(row: ManifestRow) -> FrozenFitConfig:
    """Bind ONE frozen full-manifest row to its fit configuration.

    Every value comes from that single row; nothing is read from a neighbouring
    row, a previous fit or a mutable accumulator.
    """

    _require(type(row) is ManifestRow, "full fit config requires a ManifestRow")
    _require(row.estimand in active_estimands(), "full estimand is unexpected")
    _require(row.role == resolve_role(row.estimand), "full role is inconsistent")
    _require(row.k_true in NEW_K_TRUE, "full K_TRUE is outside the frozen grid")
    _require(row.k_true != ANCHOR_K_TRUE, "the Phase 7e anchor K_TRUE must never be re-executed")
    _require(row.replicate in REPLICATES, "full replicate is unexpected")
    _require(row.k in K_CANDIDATES, "full candidate K is unexpected")
    _require(row.start in START_LABELS, "full start is unexpected")
    _require(row.data_seed == expected_data_seed(row.k_true, row.replicate, row.estimand),
             "full data seed changed")
    _require(row.split_seed == expected_split_seed(row.k_true, row.replicate),
             "full split seed changed")
    _require(row.model_seed == expected_model_seed(row.k_true, row.replicate, row.k,
                                                   row.start, row.estimand),
             "full model seed changed")
    _require(row.mask_design == MASK_DESIGN, "full mask design changed")
    return FrozenFitConfig(
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
        k_est=row.k,
        L=L_SAMPLES,
        num_iter=NUM_ITER,
        seed=row.model_seed,
        numerics_mode=NUMERICS_MODE,
    )


@dataclass(frozen=True, slots=True)
class FullPreparedCell:
    """One (estimand, K_TRUE, replicate) cell, prepared train-only, before any fit."""

    estimand: str
    role: str
    k_true: int
    replicate: int
    split: SplitRecord
    anchor: AnchorMask
    preflight: Any
    prepared: Any
    score_Y: np.ndarray
    manifest: tuple[ManifestRow, ...]
    protocol_hash: str


@dataclass(frozen=True, slots=True)
class Phase8bFullRow:
    fit_index: int
    estimand: str
    role: str
    k_true: int
    replicate: int
    k: int
    start: int
    data_seed: int
    split_seed: int
    model_seed: int
    mask_group_id: str
    anchor_mask_hash: str
    anchor_train_mask_hash: str
    heldout_mean_log_score: float
    internal_retry: int
    warning_count: int
    q_failure: bool
    nan_occurred: bool
    finite_state: bool
    pre_fit_test_hash: str
    pre_fit_train_hash: str
    post_fit_test_hash: str
    post_fit_train_hash: str


@dataclass(frozen=True, slots=True)
class Phase8bFullCellResult:
    estimand: str
    role: str
    k_true: int
    replicate: int
    rows: tuple[Phase8bFullRow, ...]
    mean_scores: tuple[tuple[int, float], ...]
    selected_k: int
    tie_candidates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Phase8bFullReport:
    protocol_hash: str
    approved_main_sha: str
    rows: tuple[Phase8bFullRow, ...]
    cells: tuple[Phase8bFullCellResult, ...]
    real_full_fits_executed: int
    test_only: bool


def full_execution_order() -> tuple[tuple[str, int, int], ...]:
    """The frozen deterministic cell order: estimand -> K_TRUE -> replicate."""

    return tuple((estimand, k_true, replicate)
                 for estimand in active_estimands()
                 for k_true in NEW_K_TRUE
                 for replicate in REPLICATES)


def prepare_full_cell(authorization: Any, estimand: str, k_true: int, replicate: int, *,
                      test_only: bool,
                      anchors: Mapping[int, AnchorMask] | None = None) -> FullPreparedCell:
    """Authorization -> anchor masks -> train-only data for exactly one cell."""

    validate_full_execution_authorization(authorization, test_only=test_only)
    _require(full_protocol_hash() == authorization.protocol_hash,
             "full protocol hash changed after authorization")
    anchors = read_phase7e_anchor_masks() if anchors is None else anchors
    anchor = anchors[replicate]
    split = build_split_record(k_true, replicate)
    _require(split.split_mask_hash == anchor.test_mask_hash,
             f"S_C: {estimand}/K{k_true}/r{replicate} test mask differs from the anchor")
    _require(split.train_mask_hash == anchor.train_mask_hash,
             f"S_C: {estimand}/K{k_true}/r{replicate} train mask differs from the anchor")

    rows = tuple(row for row in build_manifest(estimand, masks={replicate: split},
                                               anchors={replicate: anchor})
                 if row.k_true == k_true and row.replicate == replicate)
    _require(len(rows) == len(K_CANDIDATES) * len(START_LABELS),
             f"cell manifest is not {len(K_CANDIDATES) * len(START_LABELS)} rows")

    data = _generate_cell(estimand, k_true, replicate)
    X = _readonly_copy(data["X"], np.float64)
    Y = _readonly_copy(data["Y"], np.float64)
    split_plan = SplitPlan(
        replicate=replicate,
        split_seed=split.split_seed,
        expected_test_pairs=_expected_test_pairs(N_NODES, TEST_RATIO),
        train_mask=split.train_mask,
        test_mask=split.test_mask,
        diagnostics=split.diagnostics,
    )
    preflight = authorize_canary_preflight(split_plan)
    prepared = prepare_training_data(
        X, Y, preflight=preflight,
        train_mask=split.train_mask, test_mask=split.test_mask,
    )
    _require(prepared.test_mask_hash == anchor.test_mask_hash,
             "prepared test mask hash differs from the Phase 7e anchor")
    _require(prepared.train_mask_hash == anchor.train_mask_hash,
             "prepared train mask hash differs from the Phase 7e anchor")
    return FullPreparedCell(
        estimand=estimand,
        role=resolve_role(estimand),
        k_true=k_true,
        replicate=replicate,
        split=split,
        anchor=anchor,
        preflight=preflight,
        prepared=prepared,
        score_Y=Y,
        manifest=rows,
        protocol_hash=full_protocol_hash(),
    )


def _run_full_cell(cell: FullPreparedCell, *, adapter: Any, test_only: bool,
                   first_fit_index: int,
                   on_row: Any = None) -> tuple[Phase8bFullCellResult, int]:
    """Phase A: all 14 fits of one cell.  Phase B: the deferred score phase."""

    frozen_score_hash = score_config_hash(frozen_score_config())
    stored: list[Any] = []
    fit_index = first_fit_index
    for row in cell.manifest:
        config = full_fit_config(row)
        if test_only:
            _require(type(adapter) is _TestAuthorizedFitAdapter,
                     "test full run requires the test adapter")
            boundary = FitCallBoundary._from_preflight_test_only(
                cell.prepared, cell.preflight, config, adapter)
        else:
            _require(type(adapter) is AuthorizedEMFitAdapter,
                     "production full run requires the sealed Phase 7e EM adapter")
            boundary = FitCallBoundary.from_preflight(
                cell.prepared, cell.preflight, config, adapter)
        label = (f"phase8b full fit {fit_index}/{EXPECTED_FULL_FITS} "
                 f"{cell.estimand}/K_TRUE={cell.k_true}/r{cell.replicate}/K={row.k}/"
                 f"start={row.start}")
        result = boundary.call(0)
        # A dirty fit stops the whole sweep here: no retry, no replacement.
        _require_clean_smoke_fit(result, label)
        _require(boundary.test_mask_hash == cell.anchor.test_mask_hash,
                 f"{label}: post-fit test mask differs from the Phase 7e anchor")
        _require(boundary.train_mask_hash == cell.anchor.train_mask_hash,
                 f"{label}: post-fit train mask differs from the Phase 7e anchor")
        stored.append((fit_index, row, _store_smoke_fit(row, result, config, cell.prepared,
                                                        frozen_score_hash),
                       boundary.test_mask_hash, boundary.train_mask_hash))
        fit_index += 1

    # Phase B: the outcome-bearing target exists only after every fit is done.
    target = make_score_only_target(cell.score_Y, cell.prepared.test_mask)
    _require(target.test_mask_hash == cell.prepared.test_mask_hash,
             "score target mask hash mismatch")
    rows: list[Phase8bFullRow] = []
    for index, manifest_row, fit, post_test_hash, post_train_hash in stored:
        eta_pairs = heldout_raw_eta_pairs(fit.Z, fit.w0, fit.w, cell.prepared.test_mask)
        score = score_heldout_bernoulli(target, eta_pairs)
        _require(bool(np.isfinite(score)), f"full held-out score is nonfinite at fit {index}")
        full_row = Phase8bFullRow(
            fit_index=index,
            estimand=cell.estimand,
            role=cell.role,
            k_true=cell.k_true,
            replicate=cell.replicate,
            k=manifest_row.k,
            start=manifest_row.start,
            data_seed=manifest_row.data_seed,
            split_seed=manifest_row.split_seed,
            model_seed=manifest_row.model_seed,
            mask_group_id=manifest_row.mask_group_id,
            anchor_mask_hash=cell.anchor.test_mask_hash,
            anchor_train_mask_hash=cell.anchor.train_mask_hash,
            heldout_mean_log_score=float(score),
            internal_retry=fit.internal_retry,
            warning_count=len(fit.warnings),
            q_failure=fit.q_failure,
            nan_occurred=fit.nan_occurred,
            finite_state=True,
            pre_fit_test_hash=cell.split.split_mask_hash,
            pre_fit_train_hash=cell.split.train_mask_hash,
            post_fit_test_hash=post_test_hash,
            post_fit_train_hash=post_train_hash,
        )
        rows.append(full_row)
        if on_row is not None:
            on_row(full_row)

    start_scores = [StartScore(row.k, row.start, np.float64(row.heldout_mean_log_score))
                    for row in rows]
    selection = select_k_from_two_starts(start_scores, K_CANDIDATES, START_LABELS)
    means: list[tuple[int, float]] = []
    for k in K_CANDIDATES:
        by_start = {row.start: row.heldout_mean_log_score for row in rows if row.k == k}
        _require(set(by_start) == set(START_LABELS), "full aggregation start set changed")
        expected_mean = np.mean(np.asarray([by_start[1], by_start[2]], dtype=np.float64),
                                dtype=np.float64)
        _require(np.float64(selection.mean_scores[k]) == expected_mean,
                 "full aggregation is not the unweighted two-start mean")
        means.append((int(k), float(expected_mean)))

    return Phase8bFullCellResult(
        estimand=cell.estimand,
        role=cell.role,
        k_true=cell.k_true,
        replicate=cell.replicate,
        rows=tuple(rows),
        mean_scores=tuple(means),
        selected_k=int(selection.selected_k),
        tie_candidates=tuple(int(k) for k in selection.tie_candidates),
    ), fit_index


def _run_real_full(authorization: Any, *, adapter: Any, test_only: bool,
                   on_row: Any = None) -> Phase8bFullReport:
    """Exactly 336 clean fits in the frozen order, or an immediate stop."""

    validate_full_execution_authorization(authorization, test_only=test_only)
    anchors = read_phase7e_anchor_masks()
    manifests = build_full_manifests()
    validate_full_manifests(manifests)
    check_seed_collisions(manifests)
    check_full_anchor_agreement(anchors)

    cells: list[Phase8bFullCellResult] = []
    rows: list[Phase8bFullRow] = []
    fit_index = 1
    for estimand, k_true, replicate in full_execution_order():
        cell = prepare_full_cell(authorization, estimand, k_true, replicate,
                                 test_only=test_only, anchors=anchors)
        result, fit_index = _run_full_cell(cell, adapter=adapter, test_only=test_only,
                                           first_fit_index=fit_index, on_row=on_row)
        cells.append(result)
        rows.extend(result.rows)

    executed = fit_index - 1
    _require(executed == EXPECTED_FULL_FITS,
             f"full run executed {executed} fits, not {EXPECTED_FULL_FITS}")
    _require(len(rows) == EXPECTED_FULL_FITS, "full run did not score exactly 336 fits")
    _require(len(cells) == len(full_execution_order()), "full run cell count changed")
    per_estimand = {e: sum(1 for row in rows if row.estimand == e)
                    for e in active_estimands()}
    _require(per_estimand == {e: EXPECTED_FULL_FITS_PER_ESTIMAND for e in active_estimands()},
             f"A/B split is not {EXPECTED_FULL_FITS_PER_ESTIMAND}/"
             f"{EXPECTED_FULL_FITS_PER_ESTIMAND}: {per_estimand}")
    keys = [(row.estimand, row.k_true, row.replicate, row.k, row.start) for row in rows]
    _require(len(set(keys)) == len(keys), "duplicate full fit key")
    _require(tuple(row.fit_index for row in rows) == tuple(range(1, EXPECTED_FULL_FITS + 1)),
             "full fit order is not the frozen deterministic order")
    _require(all(row.k_true != ANCHOR_K_TRUE for row in rows),
             "the Phase 7e anchor K_TRUE was executed")
    return Phase8bFullReport(
        protocol_hash=full_protocol_hash(),
        approved_main_sha=authorization.approved_main_sha,
        rows=tuple(rows),
        cells=tuple(cells),
        real_full_fits_executed=0 if test_only else EXPECTED_FULL_FITS,
        test_only=test_only,
    )


def build_full_selection_matrix(cells: Sequence[Phase8bFullCellResult],
                                run_code_sha: str) -> list[tuple[Any, ...]]:
    """30 logical rows: 24 newly executed cells + 6 Phase 7e anchor references.

    The K_TRUE=3 rows are READ from the Phase 7e artifact.  They reference the
    same 42 unique anchor fits and are never re-executed or re-counted.
    """

    _require_full_commit_sha(run_code_sha, "run code SHA")
    rows: list[tuple[Any, ...]] = []
    for estimand in active_estimands():
        for k_true in NEW_K_TRUE:
            for replicate in REPLICATES:
                matching = [c for c in cells
                            if c.estimand == estimand and c.k_true == k_true
                            and c.replicate == replicate]
                _require(len(matching) == 1,
                         f"missing selection for {estimand}/K{k_true}/r{replicate}")
                cell = matching[0]
                signed = cell.selected_k - k_true
                rows.append((
                    estimand, cell.role, k_true, replicate, cell.selected_k,
                    signed, abs(signed), selection_label(signed),
                    LINEAGE_NEW, run_code_sha, FULL_ARTIFACT_DIRNAME,
                ))
        rows.extend(build_selection_matrix_anchor_rows(estimand))
    ordered = sorted(rows, key=lambda r: (r[0], r[2], r[3]))
    _require(len(ordered) == len(ESTIMANDS) * len(FULL_K_TRUE_GRID) * len(REPLICATES),
             "the integrated selection matrix is not 30 logical rows")
    return ordered


def write_full_failure_json(out_dir: Path, *, fit_index: int, completed_fits: int,
                            reason: str, run_code_sha: str) -> Path:
    """Record WHY the sweep stopped.  Partial evidence is never deleted."""

    return write_json_artifact(Path(out_dir) / FULL_FAILURE_FILENAME, {
        "artifact_version": FULL_ARTIFACT_VERSION,
        "status": "FAILED",
        "phase": PHASE,
        "execution_issue": FULL_EXECUTION_ISSUE_NUMBER,
        "protocol_origin_issue": FULL_PROTOCOL_ORIGIN_ISSUE_NUMBER,
        "protocol_hash": full_protocol_hash(),
        "run_code_sha": run_code_sha,
        "expected_full_fits": EXPECTED_FULL_FITS,
        "completed_full_fits": int(completed_fits),
        "failed_fit_index": int(fit_index),
        "reason": str(reason)[:2000],
        "policy": list(FULL_PARTIAL_FAILURE_POLICY),
        "replacement_fits_executed": 0,
        "retry_count": 0,
    })


def require_new_full_artifact_dir(out_dir: Path | None = None) -> Path:
    """The full execution directory must not exist yet.

    After a partial failure the directory DOES exist, so the same authorization
    can never start a second run: a new execution needs a new human gate.
    """

    directory = FULL_ARTIFACT_DIR if out_dir is None else Path(out_dir)
    require_not_phase7e_path(directory)
    _require(not directory.exists(),
             f"full artifact directory already exists; refusing to overwrite or resume: "
             f"{directory}")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _execute_real_full(authorization: Any, out_dir: Path | None, *,
                       test_adapter: Any, test_only: bool,
                       run_code_sha: str) -> dict[str, Any]:
    """Zero-EM preflight -> reserve dir -> evidence -> adapter -> 336 fits."""

    started_at = _utc_now()
    # --- phase 1: no adapter may exist yet -------------------------------
    validate_full_execution_authorization(authorization, test_only=test_only)
    _require_full_commit_sha(run_code_sha, "run code SHA")
    run_full_preflight()
    directory = require_new_full_artifact_dir(out_dir)
    write_json_artifact(directory / "authorization.json",
                        build_full_authorization_payload(authorization, run_code_sha))
    manifests = build_full_manifests(anchors=read_phase7e_anchor_masks())
    write_csv_artifact(directory / "manifest.csv", MANIFEST_COLUMNS,
                       [row.as_row() for estimand in active_estimands()
                        for row in manifests[estimand]])
    anchors = read_phase7e_anchor_masks()
    write_csv_artifact(directory / "mask_provenance.csv", MASK_PROVENANCE_COLUMNS,
                       [row.as_row() for estimand in active_estimands()
                        for row in build_mask_provenance(estimand, anchors)])
    write_csv_artifact(directory / "config_gate.csv", CONFIG_GATE_COLUMNS,
                       [(g.gate, g.scope, g.passed, g.detail)
                        for g in run_mask_gate(anchors=anchors)])

    # --- phase 2: only now may a fit-capable adapter exist ---------------
    adapter = _resolve_fit_adapter(test_adapter, test_only)
    completed: list[Phase8bFullRow] = []
    try:
        report = _run_real_full(authorization, adapter=adapter, test_only=test_only,
                                on_row=completed.append)
    except BaseException as error:
        # Partial evidence is written and kept.  No summary, no selection
        # matrix, no completed runinfo -> no audit can return PASS.
        write_csv_artifact(directory / "full_fit_results.csv", FULL_FIT_RESULTS_COLUMNS,
                           build_full_artifact_rows(completed, run_code_sha,
                                                    authorization, executed=len(completed)))
        write_csv_artifact(directory / "leakage_gate.csv", FULL_LEAKAGE_GATE_COLUMNS,
                           build_full_leakage_gate_rows(completed))
        write_full_failure_json(directory, fit_index=len(completed) + 1,
                                completed_fits=len(completed), reason=str(error),
                                run_code_sha=run_code_sha)
        raise

    write_csv_artifact(directory / "full_fit_results.csv", FULL_FIT_RESULTS_COLUMNS,
                       build_full_artifact_rows(report.rows, run_code_sha, authorization,
                                                executed=EXPECTED_FULL_FITS))
    write_csv_artifact(directory / "leakage_gate.csv", FULL_LEAKAGE_GATE_COLUMNS,
                       build_full_leakage_gate_rows(report.rows))
    write_csv_artifact(directory / "selection_matrix.csv", SELECTION_MATRIX_COLUMNS,
                       build_full_selection_matrix(report.cells, run_code_sha))
    write_json_artifact(directory / "full_summary.json",
                        build_full_summary_payload(report, run_code_sha))
    completed_at = _utc_now()
    write_json_artifact(directory / "runinfo.json", build_full_runinfo_payload(
        run_code_sha=run_code_sha, out_dir=directory,
        requested_command="--full", invocation_mode="cli",
        started_at=started_at, completed_at=completed_at,
        working_tree_clean=working_tree_is_clean(),
        actual_full_fits=report.real_full_fits_executed,
    ))
    return {
        "mode": "full",
        "artifact_directory": str(directory),
        "em_fits_executed": EXPECTED_FULL_FITS,
        "real_full_fits_executed": report.real_full_fits_executed,
        "canary_fits_executed": 0,
        "smoke_fits_executed": 0,
        "phase7e_rerun_count": 0,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _execute_real_full_test_only(authorization: Any, out_dir: Path, *,
                                 adapter: Any, run_code_sha: str) -> dict[str, Any]:
    """Static-test-only path.  It is the ONLY way to redirect the output dir."""

    _require(type(adapter) is _TestAuthorizedFitAdapter, "test full requires the test adapter")
    _require(out_dir is not None, "the test-only path requires an explicit temp directory")
    return _execute_real_full(authorization, out_dir, test_adapter=adapter, test_only=True,
                              run_code_sha=run_code_sha)


def build_full_authorization_payload(authorization: Any, run_code_sha: str) -> dict[str, Any]:
    _require(type(authorization) is FullExecutionAuthorization,
             "full authorization payload requires a FullExecutionAuthorization")
    _require_full_commit_sha(run_code_sha, "run code SHA")
    return {
        "artifact_version": FULL_ARTIFACT_VERSION,
        "authorization_version": authorization.authorization_version,
        "execution_issue_number": authorization.issue_number,
        "protocol_origin_issue_number": authorization.protocol_origin_issue_number,
        "approved_scientific_main_sha": authorization.approved_main_sha,
        "run_code_sha": run_code_sha,
        "protocol_hash": authorization.protocol_hash,
        "estimands": list(authorization.estimands),
        "k_true_grid": list(authorization.k_true_grid),
        "candidate_k": list(authorization.candidate_k),
        "starts": list(authorization.starts),
        "replicates": list(authorization.replicates),
        "fits_per_estimand": authorization.fits_per_estimand,
        "total_fit_count": authorization.total_fit_count,
        "data_seed_base": authorization.data_seed_base,
        "model_seed_base": authorization.model_seed_base,
        "anchor_split_seed_base": authorization.anchor_split_seed_base,
        "mask_design": authorization.mask_design,
        "random_design": authorization.random_design,
        "hierarchy": authorization.hierarchy,
        "independent_review_pass": authorization.independent_review_pass,
        "human_full_approval": authorization.human_full_approval,
        "partial_failure_policy": list(FULL_PARTIAL_FAILURE_POLICY),
    }


def build_full_artifact_rows(rows: Sequence[Phase8bFullRow], run_code_sha: str,
                             authorization: Any, *, executed: int) -> list[tuple[Any, ...]]:
    _require_full_commit_sha(run_code_sha, "run code SHA")
    return [(
        run_code_sha,
        authorization.approved_main_sha,
        full_protocol_hash(),
        row.fit_index,
        row.estimand,
        row.role,
        row.k_true,
        row.replicate,
        row.k,
        row.start,
        row.data_seed,
        row.split_seed,
        row.model_seed,
        MASK_DESIGN,
        row.mask_group_id,
        row.anchor_mask_hash,
        row.anchor_train_mask_hash,
        repr(row.heldout_mean_log_score),
        row.internal_retry,
        row.warning_count,
        row.q_failure,
        row.nan_occurred,
        row.finite_state,
        int(executed),
    ) for row in rows]


def build_full_leakage_gate_rows(rows: Sequence[Phase8bFullRow]) -> list[tuple[Any, ...]]:
    """Per-fit leakage evidence: pre-fit == post-fit == the Phase 7e anchor."""

    return [(
        row.estimand, row.role, row.k_true, row.replicate, row.k, row.start,
        row.pre_fit_test_hash, row.pre_fit_train_hash,
        row.post_fit_test_hash, row.post_fit_train_hash,
        row.anchor_mask_hash, row.anchor_train_mask_hash,
        True, True, "clean", LEAKAGE_BOUNDARY_VERSION,
    ) for row in rows]


def build_full_summary_payload(report: Phase8bFullReport, run_code_sha: str) -> dict[str, Any]:
    _require(type(report) is Phase8bFullReport, "summary requires a Phase8bFullReport")
    _require_full_commit_sha(run_code_sha, "run code SHA")
    per_cell: dict[str, Any] = {}
    for cell in report.cells:
        key = f"{cell.estimand}/K{cell.k_true}/r{cell.replicate}"
        per_cell[key] = {
            "selected_k": cell.selected_k,
            "tie_candidates": list(cell.tie_candidates),
            "mean_scores": {str(k): value for k, value in cell.mean_scores},
        }
    return {
        "artifact_version": FULL_ARTIFACT_VERSION,
        "execution_issue_number": FULL_EXECUTION_ISSUE_NUMBER,
        "protocol_origin_issue_number": FULL_PROTOCOL_ORIGIN_ISSUE_NUMBER,
        "approved_scientific_main_sha": report.approved_main_sha,
        "run_code_sha": run_code_sha,
        "protocol_hash": report.protocol_hash,
        "score_config_hash": score_config_hash(frozen_score_config()),
        "expected_full_fits": EXPECTED_FULL_FITS,
        "actual_full_fits": len(report.rows),
        "expected_full_fits_per_estimand": EXPECTED_FULL_FITS_PER_ESTIMAND,
        "estimands": list(active_estimands()),
        "candidate_k": list(K_CANDIDATES),
        "k_true_grid": list(NEW_K_TRUE),
        "per_cell": per_cell,
        "selected_k": {f"{c.estimand}/K{c.k_true}/r{c.replicate}": c.selected_k
                       for c in report.cells},
        # The full grid DOES contain K_TRUE, so recovery is evaluated and
        # recorded.  It is a scientific outcome, never an audit gate.
        "k_recovery_evaluated": True,
        "k_recovery_is_not_an_audit_gate": True,
        "anchor_k_true": ANCHOR_K_TRUE,
        "anchor_lineage": LINEAGE_ANCHOR,
        "anchor_artifact_dir": PHASE7E_ARTIFACT_DIR,
        "anchor_run_code_sha": PHASE7E_RUN_CODE_SHA,
        "anchor_unique_fits": PHASE7E_ANCHOR_FIT_COUNT,
    }


def build_full_runinfo_payload(*, run_code_sha: str, out_dir: Path,
                               requested_command: str, invocation_mode: str,
                               started_at: str, completed_at: str,
                               working_tree_clean: bool,
                               actual_full_fits: int) -> dict[str, Any]:
    return {
        "artifact_version": FULL_ARTIFACT_VERSION,
        "phase": PHASE,
        "execution_issue": FULL_EXECUTION_ISSUE_NUMBER,
        "protocol_origin_issue": FULL_PROTOCOL_ORIGIN_ISSUE_NUMBER,
        "run_code_sha": run_code_sha,
        "approved_scientific_main_sha": APPROVED_SCIENTIFIC_MAIN_SHA,
        "approved_baseline_is_ancestor": approved_baseline_is_ancestor(run_code_sha),
        "protocol_hash": full_protocol_hash(),
        "score_config_hash": score_config_hash(frozen_score_config()),
        "frozen_config_hash": frozen_config_hash(),
        "mask_design": MASK_DESIGN,
        "random_design": RANDOM_DESIGN,
        "hierarchy": HIERARCHY,
        "working_tree_clean": bool(working_tree_clean),
        "invocation_mode": invocation_mode,
        "requested_command": requested_command,
        "started_at": started_at,
        "completed_at": completed_at,
        "expected_full_fits": EXPECTED_FULL_FITS,
        "actual_full_fits": int(actual_full_fits),
        "expected_full_fits_per_estimand": EXPECTED_FULL_FITS_PER_ESTIMAND,
        "canary_fits_executed": 0,
        "smoke_fits_executed": 0,
        "replacement_fits_executed": 0,
        "phase7e_rerun_count": 0,
        "anchor_unique_fits": PHASE7E_ANCHOR_FIT_COUNT,
        "partial_failure_policy": list(FULL_PARTIAL_FAILURE_POLICY),
        "artifact_directory": str(out_dir),
        "artifact_files": list(FULL_ARTIFACT_FILES),
    }


def run_real_full(authorization: Any) -> dict[str, Any]:
    """Production full entry point.

    A thin wrapper over the single production workflow: it never validates,
    constructs an adapter or fits on its own.
    """

    return _run_production_full_execution(authorization)


def _run_production_full_execution(authorization: Any) -> dict[str, Any]:
    """THE production workflow for the 336-fit sweep."""

    validate_full_execution_authorization(authorization, test_only=False)
    run_code_sha = current_run_code_sha()
    _require_execution_preconditions(run_code_sha)
    return _execute_real_full(authorization, FULL_ARTIFACT_DIR,
                              test_adapter=None, test_only=False,
                              run_code_sha=run_code_sha)


# ---------------------------------------------------------------------------
# Shared preflight for every real execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmokePreparedCell:
    """The frozen smoke cell, prepared train-only, before any fit."""

    split: SplitRecord
    anchor: AnchorMask
    preflight: CanaryPreflight
    prepared: PreparedTrainingData
    score_Y: np.ndarray
    manifest: tuple[ManifestRow, ...]
    protocol_hash: str


def _require_smoke_anchor_masks(split: SplitRecord, anchor: AnchorMask) -> None:
    """S_C: BOTH the test and the train mask must equal the Phase 7e anchor."""

    _require(split.split_mask_hash == anchor.test_mask_hash,
             "smoke test mask hash differs from the Phase 7e anchor")
    _require(split.train_mask_hash == anchor.train_mask_hash,
             "smoke train mask hash differs from the Phase 7e anchor")


def _run_zero_em_preflight_gates() -> dict[str, Any]:
    """Every zero-EM gate must pass before a single real fit is prepared."""

    validation = run_validate_only()
    _require(validation["em_fits_executed"] == 0, "validate-only reported a nonzero fit count")
    config = run_config_gate()
    _require(bool(config["all_passed"]), "config gate did not pass")
    anchors = read_phase7e_anchor_masks()
    mask_gates = run_mask_gate(anchors=anchors)
    _require(all(gate.passed for gate in mask_gates), "anchor mask gate did not pass")
    leakage = run_leakage_self_check()
    _require(bool(leakage["all_passed"]), "leakage self-check did not pass")
    _require(leakage["real_em_fits_executed"] == 0, "leakage self-check executed a real fit")
    return {
        "config_gate_count": config["gate_count"],
        "leakage_cases": len(leakage["cases"]),
        "anchor_mask_gates": len(mask_gates),
    }


def prepare_smoke_cell(authorization: Any, *, test_only: bool) -> SmokePreparedCell:
    """Authorization -> zero-EM gates -> anchor masks -> train-only data.

    The raw ``Y`` is returned for the deferred score phase only; it never
    enters the fit API, which is owned by the Phase 7e ``FitCallBoundary``.
    """

    validate_smoke_execution_authorization(authorization, test_only=test_only)
    _require(smoke_protocol_hash() == authorization.protocol_hash,
             "smoke protocol hash changed after authorization")
    check_smoke_seed_collisions()
    _run_zero_em_preflight_gates()

    anchors = read_phase7e_anchor_masks()
    anchor = anchors[SMOKE_REPLICATE]
    split = build_split_record(SMOKE_K_TRUE, SMOKE_REPLICATE)
    _require_smoke_anchor_masks(split, anchor)

    manifest = build_smoke_manifest(anchors=anchors, split=split)
    data_seed = smoke_data_seed(SMOKE_K_TRUE, SMOKE_REPLICATE)
    data = _generate_smoke_cell(data_seed)
    X = _readonly_copy(data["X"], np.float64)
    Y = _readonly_copy(data["Y"], np.float64)

    split_plan = SplitPlan(
        replicate=SMOKE_REPLICATE,
        split_seed=split.split_seed,
        expected_test_pairs=_expected_test_pairs(N_NODES, TEST_RATIO),
        train_mask=split.train_mask,
        test_mask=split.test_mask,
        diagnostics=split.diagnostics,
    )
    preflight = authorize_canary_preflight(split_plan)
    prepared = prepare_training_data(
        X, Y,
        preflight=preflight,
        train_mask=split.train_mask,
        test_mask=split.test_mask,
    )
    _require(prepared.test_mask_hash == anchor.test_mask_hash,
             "prepared test mask hash differs from the Phase 7e anchor")
    _require(prepared.train_mask_hash == anchor.train_mask_hash,
             "prepared train mask hash differs from the Phase 7e anchor")
    return SmokePreparedCell(
        split=split,
        anchor=anchor,
        preflight=preflight,
        prepared=prepared,
        score_Y=Y,
        manifest=tuple(manifest),
        protocol_hash=smoke_protocol_hash(),
    )


def _generate_smoke_cell(data_seed: int) -> dict[str, Any]:
    """Generator only.  This never touches EM."""

    from data_generator_expfam import generate_dual_data  # noqa: PLC0415

    _require(data_seed == smoke_data_seed(SMOKE_K_TRUE, SMOKE_REPLICATE),
             "smoke data seed is not the dedicated smoke seed")
    return generate_dual_data(
        n=N_NODES,
        d=N_FEATURES,
        k=SMOKE_K_TRUE,
        seed=int(data_seed),
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
        var_f=VAR_F,
        uniq=UNIQ,
        w0_true=W0_TRUE,
        w_true=resolve_w_true(SMOKE_ESTIMAND, SMOKE_K_TRUE),
    )


# ---------------------------------------------------------------------------
# Real canary: exactly two fits
# ---------------------------------------------------------------------------


def smoke_canary_config() -> FrozenFitConfig:
    """K_est=1, start=1, model seed 641011.  Both canary fits start identically."""

    _require(CANARY_MODEL_SEED == smoke_model_seed(SMOKE_K_TRUE, SMOKE_REPLICATE,
                                                   CANARY_K_EST, CANARY_START),
             "canary model seed is not the dedicated smoke seed")
    return FrozenFitConfig(
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
        k_est=CANARY_K_EST,
        L=L_SAMPLES,
        num_iter=NUM_ITER,
        seed=CANARY_MODEL_SEED,
        numerics_mode=NUMERICS_MODE,
    )


@dataclass(frozen=True, slots=True)
class Phase8bCanaryReport:
    protocol_hash: str
    approved_main_sha: str
    k_est: int
    start: int
    data_seed: int
    split_seed: int
    model_seed: int
    anchor_test_hash: str
    anchor_train_hash: str
    invariance: CanaryInvarianceReport
    real_canary_fits_executed: int
    test_only: bool


def _run_real_canary(authorization: Any, *, adapter: Any, test_only: bool,
                     cell: "SmokePreparedCell | None" = None) -> Phase8bCanaryReport:
    """Reuse the Phase 7e two-canary falsification on the smoke cell.

    ``cell`` lets the caller complete every no-fit preflight *before* any
    directory is reserved or any adapter is constructed; when it is omitted the
    identical preparation runs here.
    """

    cell = prepare_smoke_cell(authorization, test_only=test_only) if cell is None else cell
    config = smoke_canary_config()
    invariance = _run_two_canary_falsification(
        preflight=cell.preflight,
        prepared=cell.prepared,
        score_Y=cell.score_Y,
        config=config,
        adapter=adapter,
        test_only=test_only,
    )
    return Phase8bCanaryReport(
        protocol_hash=cell.protocol_hash,
        approved_main_sha=authorization.approved_main_sha,
        k_est=CANARY_K_EST,
        start=CANARY_START,
        data_seed=smoke_data_seed(SMOKE_K_TRUE, SMOKE_REPLICATE),
        split_seed=SMOKE_SPLIT_SEED,
        model_seed=CANARY_MODEL_SEED,
        anchor_test_hash=cell.anchor.test_mask_hash,
        anchor_train_hash=cell.anchor.train_mask_hash,
        invariance=invariance,
        real_canary_fits_executed=0 if test_only else EXPECTED_CANARY_FITS,
        test_only=test_only,
    )


def run_real_canary(authorization: Any) -> dict[str, Any]:
    """Production canary entry point.

    A thin wrapper over the single production workflow: it never validates,
    constructs an adapter or fits on its own.  Calling it directly is exactly
    as gated as ``--canary``.  Defined near the CLI so it can delegate to
    ``_run_production_execution``.
    """

    return _run_production_execution(authorization, "canary")


def _run_real_canary_test_only(authorization: Any, *, adapter: Any) -> Phase8bCanaryReport:
    """Static-test-only entry point; the production CLI cannot select it."""

    _require(type(adapter) is _TestAuthorizedFitAdapter, "test canary requires the test adapter")
    return _run_real_canary(authorization, adapter=adapter, test_only=True)


# ---------------------------------------------------------------------------
# Real smoke: exactly six fits, then a deferred score phase
# ---------------------------------------------------------------------------


def smoke_fit_config(row: ManifestRow) -> FrozenFitConfig:
    """Bind one frozen smoke manifest row to its fit configuration."""

    _require(type(row) is ManifestRow, "smoke fit config requires a ManifestRow")
    _require(row.estimand == SMOKE_ESTIMAND and row.role == SMOKE_ROLE, "smoke estimand/role changed")
    _require(row.k_true == SMOKE_K_TRUE and row.replicate == SMOKE_REPLICATE, "smoke cell changed")
    _require(row.k in SMOKE_K_CANDIDATES, "smoke candidate K is unexpected")
    _require(row.start in SMOKE_STARTS, "smoke start is unexpected")
    _require(row.data_seed == smoke_data_seed(row.k_true, row.replicate), "smoke data seed changed")
    _require(row.split_seed == SMOKE_SPLIT_SEED, "smoke split seed changed")
    _require(row.model_seed == smoke_model_seed(row.k_true, row.replicate, row.k, row.start),
             "smoke model seed changed")
    return FrozenFitConfig(
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
        k_est=row.k,
        L=L_SAMPLES,
        num_iter=NUM_ITER,
        seed=row.model_seed,
        numerics_mode=NUMERICS_MODE,
    )


@dataclass(frozen=True, slots=True)
class Phase8bSmokeRow:
    k: int
    start: int
    data_seed: int
    split_seed: int
    model_seed: int
    fit_status: str
    heldout_mean_log_score: float
    internal_retry: int
    warning_count: int
    q_failure: bool
    nan_occurred: bool
    finite_state: bool
    pre_fit_test_hash: str
    pre_fit_train_hash: str
    post_fit_test_hash: str
    post_fit_train_hash: str
    score_config_hash: str


@dataclass(frozen=True, slots=True)
class Phase8bSmokeReport:
    protocol_hash: str
    approved_main_sha: str
    rows: tuple[Phase8bSmokeRow, ...]
    mean_scores: tuple[tuple[int, float], ...]
    # RECORDED ONLY.  K_TRUE=1 is not in the candidate set {2,3,4}, so
    # ``selected_k == K_TRUE`` is structurally impossible and this value is
    # never a pass/fail condition, a gate, or scientific evidence.
    selected_k: int
    tie_candidates: tuple[int, ...]
    anchor_test_hash: str
    anchor_train_hash: str
    score_config_hash: str
    real_smoke_fits_executed: int
    test_only: bool


def _run_smoke_fit_phase_8b(cell: SmokePreparedCell, *, adapter: Any,
                            test_only: bool) -> tuple[Any, ...]:
    """Phase A: all six fits complete before any ScoreOnlyTarget exists."""

    manifest = list(cell.manifest)
    validate_smoke_manifest(manifest)
    frozen_score_hash = score_config_hash(frozen_score_config())
    stored: list[Any] = []
    fit_count = 0
    for row in manifest:
        config = smoke_fit_config(row)
        if test_only:
            _require(type(adapter) is _TestAuthorizedFitAdapter, "test smoke requires the test adapter")
            boundary = FitCallBoundary._from_preflight_test_only(
                cell.prepared, cell.preflight, config, adapter)
        else:
            _require(type(adapter) is AuthorizedEMFitAdapter,
                     "production smoke requires the sealed Phase 7e EM adapter")
            boundary = FitCallBoundary.from_preflight(
                cell.prepared, cell.preflight, config, adapter)
        fit_count += 1
        result = boundary.call(0)
        label = f"phase8b smoke K={row.k} start={row.start}"
        _require_clean_smoke_fit(result, label)
        # Post-fit: the masks the boundary owns must still be the anchor masks.
        _require(boundary.test_mask_hash == cell.anchor.test_mask_hash,
                 f"{label}: post-fit test mask hash differs from the Phase 7e anchor")
        _require(boundary.train_mask_hash == cell.anchor.train_mask_hash,
                 f"{label}: post-fit train mask hash differs from the Phase 7e anchor")
        stored.append(_store_smoke_fit(row, result, config, cell.prepared, frozen_score_hash))
    _require(fit_count == EXPECTED_SMOKE_FITS,
             f"smoke did not execute exactly {EXPECTED_SMOKE_FITS} fits")
    _require(len(stored) == EXPECTED_SMOKE_FITS, "smoke did not store exactly six clean fits")
    _require(tuple((row.k, row.start, row.model_seed) for row in stored)
             == tuple((row.k, row.start, row.model_seed) for row in manifest),
             "stored smoke fit order changed")
    return tuple(stored)


def _run_real_smoke(authorization: Any, *, adapter: Any, test_only: bool,
                    cell: "SmokePreparedCell | None" = None) -> Phase8bSmokeReport:
    """Six clean fits, then -- and only then -- the deferred score phase."""

    cell = prepare_smoke_cell(authorization, test_only=test_only) if cell is None else cell
    stored_fits = _run_smoke_fit_phase_8b(cell, adapter=adapter, test_only=test_only)

    # Phase B: the outcome-bearing target is materialized exactly once, after
    # every fit and every blocking gate has passed.
    target = make_score_only_target(cell.score_Y, cell.prepared.test_mask)
    _require(target.test_mask_hash == cell.prepared.test_mask_hash,
             "score target mask hash mismatch")
    frozen_score_hash = score_config_hash(frozen_score_config())

    rows: list[Phase8bSmokeRow] = []
    for stored in stored_fits:
        eta_pairs = heldout_raw_eta_pairs(stored.Z, stored.w0, stored.w, cell.prepared.test_mask)
        score = score_heldout_bernoulli(target, eta_pairs)
        _require(bool(np.isfinite(score)), "smoke held-out score is nonfinite")
        rows.append(Phase8bSmokeRow(
            k=stored.k,
            start=stored.start,
            data_seed=stored.data_seed,
            split_seed=stored.split_seed,
            model_seed=stored.model_seed,
            fit_status="clean",
            heldout_mean_log_score=float(score),
            internal_retry=stored.internal_retry,
            warning_count=len(stored.warnings),
            q_failure=stored.q_failure,
            nan_occurred=stored.nan_occurred,
            finite_state=True,
            pre_fit_test_hash=cell.split.split_mask_hash,
            pre_fit_train_hash=cell.split.train_mask_hash,
            post_fit_test_hash=stored.test_mask_hash,
            post_fit_train_hash=stored.train_mask_hash,
            score_config_hash=stored.score_config_hash,
        ))
    _require(len(rows) == EXPECTED_SMOKE_FITS, "smoke did not score exactly six stored fits")

    start_scores = [StartScore(row.k, row.start, np.float64(row.heldout_mean_log_score))
                    for row in rows]
    selection = select_k_from_two_starts(start_scores, SMOKE_K_CANDIDATES, SMOKE_STARTS)
    means: list[tuple[int, float]] = []
    for k in SMOKE_K_CANDIDATES:
        by_start = {row.start: row.heldout_mean_log_score for row in rows if row.k == k}
        _require(set(by_start) == set(SMOKE_STARTS), "smoke aggregation start set changed")
        expected_mean = np.mean(np.asarray([by_start[1], by_start[2]], dtype=np.float64),
                                dtype=np.float64)
        _require(np.float64(selection.mean_scores[k]) == expected_mean,
                 "smoke aggregation is not the unweighted two-start mean")
        means.append((int(k), float(expected_mean)))

    # selected_k is recorded, never gated on.  K_TRUE=1 is not a candidate.
    return Phase8bSmokeReport(
        protocol_hash=cell.protocol_hash,
        approved_main_sha=authorization.approved_main_sha,
        rows=tuple(rows),
        mean_scores=tuple(means),
        selected_k=int(selection.selected_k),
        tie_candidates=tuple(int(k) for k in selection.tie_candidates),
        anchor_test_hash=cell.anchor.test_mask_hash,
        anchor_train_hash=cell.anchor.train_mask_hash,
        score_config_hash=frozen_score_hash,
        real_smoke_fits_executed=0 if test_only else EXPECTED_SMOKE_FITS,
        test_only=test_only,
    )


def run_real_smoke(authorization: Any) -> dict[str, Any]:
    """Production smoke entry point.

    A thin wrapper over the single production workflow.  There is deliberately
    no ``canary_report``, ``canary_pass``, ``skip_canary_check``, ``out_dir``
    or ``run_code_sha`` parameter: the canary evidence is read from the frozen
    artifact directory and the run-code SHA comes from the internal provenance
    source, so a caller cannot supply either side of a lineage check.
    """

    return _run_production_execution(authorization, "smoke")


def _run_real_smoke_test_only(authorization: Any, *, adapter: Any) -> Phase8bSmokeReport:
    """Static-test-only entry point; the production CLI cannot select it."""

    _require(type(adapter) is _TestAuthorizedFitAdapter, "test smoke requires the test adapter")
    return _run_real_smoke(authorization, adapter=adapter, test_only=True)


def build_smoke_artifact_rows(canary: Phase8bCanaryReport,
                              smoke: Phase8bSmokeReport,
                              run_code_sha: str) -> list[tuple[Any, ...]]:
    """Future artifact rows.  S2b writes nothing; this only fixes the schema."""

    _require(type(canary) is Phase8bCanaryReport, "artifact rows require a canary report")
    _require(type(smoke) is Phase8bSmokeReport, "artifact rows require a smoke report")
    _require(canary.protocol_hash == smoke.protocol_hash, "canary/smoke protocol hash differs")
    canary_provenance = stable_config_hash({
        "k_est": canary.k_est,
        "start": canary.start,
        "model_seed": canary.model_seed,
        "config_hash": canary.invariance.config_hash,
        "payload_a_hash": canary.invariance.fit_payload_a_hash,
        "payload_b_hash": canary.invariance.fit_payload_b_hash,
    })
    rows: list[tuple[Any, ...]] = []
    for row in smoke.rows:
        rows.append((
            run_code_sha, smoke.approved_main_sha, smoke.protocol_hash,
            SMOKE_ESTIMAND, SMOKE_ROLE, SMOKE_K_TRUE, SMOKE_REPLICATE,
            row.k, row.start, row.data_seed, row.split_seed, row.model_seed,
            row.pre_fit_test_hash, row.pre_fit_train_hash,
            row.post_fit_test_hash, row.post_fit_train_hash,
            smoke.anchor_test_hash, smoke.anchor_train_hash,
            LEAKAGE_BOUNDARY_VERSION, row.fit_status,
            row.internal_retry, row.warning_count, row.q_failure, row.nan_occurred,
            row.finite_state, row.heldout_mean_log_score, row.score_config_hash,
            canary_provenance,
            canary.real_canary_fits_executed, smoke.real_smoke_fits_executed,
        ))
    _require(all(len(row) == len(SMOKE_ARTIFACT_COLUMNS) for row in rows),
             "smoke artifact row width does not match the schema")
    return rows


# ===========================================================================
# Phase 8b S2c-A — execution artifacts, canary lineage, CLI wiring (Issue #55)
# ===========================================================================
#
# Everything below is the machinery a future authorized run would use.  None of
# it can run here: ``current_smoke_execution_authorization()`` is still None, so
# the CLI stops before any adapter is constructed.  Binding the reviewed
# baseline SHA is NOT an execution approval.


SMOKE_ARTIFACT_VERSION = "phase8b-smoke-artifact-v1"
SMOKE_ARTIFACT_DIRNAME = "k_true_robustness_smoke_20260901"
SMOKE_ARTIFACT_ROOT = ROOT / "expfam" / "results" / "k_selection"
SMOKE_ARTIFACT_DIR = SMOKE_ARTIFACT_ROOT / SMOKE_ARTIFACT_DIRNAME

# Fixed before any result exists.  ``audit_report.json`` is the audit's OUTPUT,
# so it is not required as an audit input.
SMOKE_ARTIFACT_FILES = (
    "authorization.json",
    "canary.json",
    "canary_audit.json",
    "runinfo.json",
    "smoke_fit_results.csv",
    "smoke_summary.json",
    "audit_report.json",
)
SMOKE_AUDIT_INPUT_FILES = tuple(
    name for name in SMOKE_ARTIFACT_FILES if name != "audit_report.json")

# The independent canary verdict.  It is produced ONLY by
# ``audit_k_true_robustness_sweep``; this module may read and require it, never
# write it.  Letting the runner manufacture its own PASS would make the
# independent audit step decorative.
CANARY_AUDIT_FILENAME = "canary_audit.json"
CANARY_AUDIT_VERSION = "phase8b-canary-audit-v1"
CANARY_AUDIT_REQUIRED_KEYS = (
    "audit_version", "status", "blocker_count", "high_count", "medium_count", "findings",
    "approved_scientific_main_sha", "run_code_sha", "protocol_hash",
    "protocol_origin_issue", "execution_issue", "expected_canary_fits",
    "actual_canary_fits", "canary_execution_mode", "canary_status",
    "audited_files", "authorization_content_sha256", "canary_content_sha256",
)

CANARY_STATUS_PASS = "PASS"
CANARY_STATUS_FAIL = "FAIL"

# Selected-K interpretation, frozen before any smoke result exists.
SELECTED_K_INTERPRETATION = "record_only"


# ---------------------------------------------------------------------------
# git provenance -- NOT an approval source
# ---------------------------------------------------------------------------


def _git_output(arguments: Sequence[str]) -> str:
    import subprocess  # noqa: PLC0415

    completed = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, cwd=str(ROOT), check=False)
    _require(completed.returncode == 0,
             f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def current_run_code_sha() -> str:
    """PROVENANCE ONLY: which commit executed.

    This is deliberately NOT the approval source.  ``current_expected_smoke_
    main_sha()`` alone decides whether a reviewed scientific baseline has been
    approved; recording which commit ran is a separate, purely descriptive
    fact.  Confusing the two would let any commit approve itself.
    """

    sha = _git_output(["rev-parse", "HEAD"])
    _require_full_commit_sha(sha, "run code SHA")
    return sha


def working_tree_is_clean() -> bool:
    return _git_output(["status", "--porcelain"]) == ""


def approved_baseline_is_ancestor(run_code_sha: str | None = None) -> bool:
    """Extra provenance guard: the reviewed baseline is in this commit's history.

    This is a *guard*, never an approval: a commit being a descendant of the
    approved baseline does not make that commit approved.
    """

    import subprocess  # noqa: PLC0415

    head = current_run_code_sha() if run_code_sha is None else run_code_sha
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", APPROVED_SCIENTIFIC_MAIN_SHA, head],
        capture_output=True, text=True, cwd=str(ROOT), check=False)
    return completed.returncode == 0


# ---------------------------------------------------------------------------
# fail-safe artifact I/O
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Write via a temporary file + atomic replace.

    A crashed or truncated write can never be read back as evidence: readers
    only ever see the complete file or no file at all.
    """

    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    directory = path.parent
    _require(directory.is_dir(), f"artifact directory does not exist: {directory}")
    handle, temporary = tempfile.mkstemp(dir=str(directory), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _require_absent_artifact(path: Path) -> None:
    _require(not path.exists(), f"refusing to overwrite an existing artifact: {path}")


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    _require_absent_artifact(path)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False)
    return _atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def write_csv_artifact(path: Path, header: Sequence[str],
                       rows: Sequence[Sequence[Any]]) -> Path:
    _require_absent_artifact(path)
    import io  # noqa: PLC0415

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        _require(len(row) == len(header), "artifact row width does not match the header")
        writer.writerow(row)
    return _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def read_json_artifact(path: Path) -> dict[str, Any]:
    """Strict read: a missing, empty, truncated or non-object file is a stop."""

    path = Path(path)
    _require(path.is_file(), f"required artifact is missing: {path}")
    text = path.read_text(encoding="utf-8")
    _require(text.strip() != "", f"artifact is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise HarnessStop(f"artifact is not valid JSON: {path} ({error})") from error
    _require(isinstance(payload, dict), f"artifact is not a JSON object: {path}")
    return payload


def require_new_smoke_artifact_dir(out_dir: Path | None = None) -> Path:
    """The execution directory must not exist yet.  Nothing is ever overwritten."""

    directory = SMOKE_ARTIFACT_DIR if out_dir is None else Path(out_dir)
    require_not_phase7e_path(directory)
    _require(not directory.exists(),
             f"smoke artifact directory already exists; refusing to overwrite: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def require_existing_smoke_artifact_dir(out_dir: Path | None = None) -> Path:
    directory = SMOKE_ARTIFACT_DIR if out_dir is None else Path(out_dir)
    require_not_phase7e_path(directory)
    _require(directory.is_dir(), f"smoke artifact directory does not exist: {directory}")
    return directory


# ---------------------------------------------------------------------------
# artifact payloads (schema frozen before any result exists)
# ---------------------------------------------------------------------------


def build_authorization_payload(authorization: Any, run_code_sha: str) -> dict[str, Any]:
    _require(type(authorization) is SmokeExecutionAuthorization,
             "authorization payload requires a SmokeExecutionAuthorization")
    _require_full_commit_sha(run_code_sha, "run code SHA")
    return {
        "artifact_version": SMOKE_ARTIFACT_VERSION,
        "execution_issue_number": authorization.issue_number,
        "protocol_origin_issue_number": SMOKE_PROTOCOL_ISSUE_NUMBER,
        "approved_scientific_main_sha": authorization.approved_main_sha,
        "run_code_sha": run_code_sha,
        "protocol_hash": authorization.protocol_hash,
        "authorization_version": authorization.authorization_version,
        "independent_review_pass": authorization.independent_review_pass,
        "human_smoke_approval": authorization.human_smoke_approval,
        "expected_canary_fits": authorization.canary_fit_count,
        "expected_smoke_fits": authorization.smoke_fit_count,
        "expected_real_em_budget": EXPECTED_REAL_EM_BUDGET,
        "estimand": authorization.estimand,
        "role": SMOKE_ROLE,
        "k_true": authorization.k_true,
        "replicate": authorization.replicate,
        "data_seed_base": authorization.data_seed_base,
        "model_seed_base": authorization.model_seed_base,
        "split_seed": authorization.split_seed,
        "data_seed": smoke_data_seed(SMOKE_K_TRUE, SMOKE_REPLICATE),
        "canary_model_seed": CANARY_MODEL_SEED,
        "smoke_model_seeds": [smoke_model_seed(SMOKE_K_TRUE, SMOKE_REPLICATE, k, start)
                              for k in SMOKE_K_CANDIDATES for start in SMOKE_STARTS],
    }


def build_canary_payload(report: Phase8bCanaryReport, run_code_sha: str) -> dict[str, Any]:
    """PASS evidence.  Only a report from a fully clean canary reaches this."""

    _require(type(report) is Phase8bCanaryReport, "canary payload requires a Phase8bCanaryReport")
    _require_full_commit_sha(run_code_sha, "run code SHA")
    _require(report.real_canary_fits_executed
             == (0 if report.test_only else EXPECTED_CANARY_FITS),
             "canary evidence real-fit count is inconsistent with the execution mode")
    invariance = report.invariance
    return {
        "artifact_version": SMOKE_ARTIFACT_VERSION,
        "status": CANARY_STATUS_PASS,
        # A test-only canary can never satisfy a production smoke, and a real
        # canary can never satisfy a test-only smoke: the modes are disjoint.
        "execution_mode": "test_only" if report.test_only else "real",
        "execution_issue_number": SMOKE_EXECUTION_ISSUE_NUMBER,
        "protocol_origin_issue_number": SMOKE_PROTOCOL_ISSUE_NUMBER,
        "approved_scientific_main_sha": report.approved_main_sha,
        "run_code_sha": run_code_sha,
        "protocol_hash": report.protocol_hash,
        "estimand": SMOKE_ESTIMAND,
        "role": SMOKE_ROLE,
        "k_true": SMOKE_K_TRUE,
        "replicate": SMOKE_REPLICATE,
        "k_est": report.k_est,
        "start": report.start,
        "data_seed": report.data_seed,
        "split_seed": report.split_seed,
        "model_seed": report.model_seed,
        "expected_fit_count": EXPECTED_CANARY_FITS,
        # The orchestration structurally performs exactly two boundary calls in
        # both modes; only the REAL count distinguishes them.
        "actual_fit_count": EXPECTED_CANARY_FITS,
        "real_canary_fits_executed": report.real_canary_fits_executed,
        "anchor_test_hash": report.anchor_test_hash,
        "anchor_train_hash": report.anchor_train_hash,
        "fit_config_hash": invariance.config_hash,
        "fit_payload_a_hash": invariance.fit_payload_a_hash,
        "fit_payload_b_hash": invariance.fit_payload_b_hash,
        "initialization_equal": bool(invariance.initialization_equal),
        "final_outputs_equal": bool(invariance.final_outputs_equal),
        "internal_retry": int(invariance.internal_retry),
        "warning_count": 0,
        "q_failure": False,
        "nan_occurred": False,
        "finite_state": True,
        "canary_atol": float(CANARY_ATOL),
        "canary_rtol": float(CANARY_RTOL),
        "boundary_version": LEAKAGE_BOUNDARY_VERSION,
    }


def build_smoke_fit_rows(smoke: Phase8bSmokeReport, canary: Phase8bCanaryReport,
                         run_code_sha: str) -> list[tuple[Any, ...]]:
    return build_smoke_artifact_rows(canary, smoke, run_code_sha)


def build_smoke_summary_payload(smoke: Phase8bSmokeReport,
                                run_code_sha: str) -> dict[str, Any]:
    _require(type(smoke) is Phase8bSmokeReport, "summary requires a Phase8bSmokeReport")
    _require_full_commit_sha(run_code_sha, "run code SHA")
    by_key = {(row.k, row.start): row.heldout_mean_log_score for row in smoke.rows}
    means = dict(smoke.mean_scores)
    per_k: dict[str, Any] = {}
    for k in SMOKE_K_CANDIDATES:
        per_k[str(k)] = {
            "start_1": by_key[(k, 1)],
            "start_2": by_key[(k, 2)],
            "mean": means[k],
        }
    return {
        "artifact_version": SMOKE_ARTIFACT_VERSION,
        "execution_issue_number": SMOKE_EXECUTION_ISSUE_NUMBER,
        "protocol_origin_issue_number": SMOKE_PROTOCOL_ISSUE_NUMBER,
        "approved_scientific_main_sha": smoke.approved_main_sha,
        "run_code_sha": run_code_sha,
        "protocol_hash": smoke.protocol_hash,
        "score_config_hash": smoke.score_config_hash,
        "per_k": per_k,
        "selected_k": smoke.selected_k,
        "tie_candidates": list(smoke.tie_candidates),
        # K_TRUE=1 is not in the candidate set {2,3,4}: agreement with K_TRUE is
        # structurally impossible, so selected_k is operational evidence only.
        "selected_k_interpretation": SELECTED_K_INTERPRETATION,
        "k_recovery_evaluated": False,
        "k_true": SMOKE_K_TRUE,
        "candidate_k": list(SMOKE_K_CANDIDATES),
        "expected_smoke_fits": EXPECTED_SMOKE_FITS,
        "actual_smoke_fits": smoke.real_smoke_fits_executed,
    }


def build_smoke_runinfo_payload(*, run_code_sha: str, out_dir: Path,
                                requested_command: str, invocation_mode: str,
                                started_at: str, completed_at: str,
                                working_tree_clean: bool,
                                actual_canary_fits: int,
                                actual_smoke_fits: int,
                                canary_audit_status: str) -> dict[str, Any]:
    return {
        "artifact_version": SMOKE_ARTIFACT_VERSION,
        "phase": PHASE,
        "execution_issue": SMOKE_EXECUTION_ISSUE_NUMBER,
        "protocol_origin_issue": SMOKE_PROTOCOL_ISSUE_NUMBER,
        # run_code_sha is provenance; approved_scientific_main_sha is approval.
        "run_code_sha": run_code_sha,
        "approved_scientific_main_sha": APPROVED_SCIENTIFIC_MAIN_SHA,
        "approved_baseline_is_ancestor": approved_baseline_is_ancestor(run_code_sha),
        "protocol_hash": smoke_protocol_hash(),
        "score_config_hash": score_config_hash(frozen_score_config()),
        "frozen_config_hash": frozen_config_hash(),
        "working_tree_clean": bool(working_tree_clean),
        "invocation_mode": invocation_mode,
        "requested_command": requested_command,
        "started_at": started_at,
        "completed_at": completed_at,
        "expected_real_em_budget": EXPECTED_REAL_EM_BUDGET,
        "expected_canary_fits": EXPECTED_CANARY_FITS,
        "expected_smoke_fits": EXPECTED_SMOKE_FITS,
        "actual_canary_fits": int(actual_canary_fits),
        "actual_smoke_fits": int(actual_smoke_fits),
        "canary_audit_status": canary_audit_status,
        "full_fits_executed": 0,
        "phase7e_rerun_count": 0,
        "artifact_directory": str(out_dir),
        "artifact_files": list(SMOKE_ARTIFACT_FILES),
    }


# ---------------------------------------------------------------------------
# canary-before-smoke lineage gate
# ---------------------------------------------------------------------------


CANARY_EVIDENCE_REQUIRED_KEYS = (
    "artifact_version", "status", "execution_issue_number",
    "protocol_origin_issue_number", "approved_scientific_main_sha", "run_code_sha",
    "protocol_hash", "estimand", "role", "k_true", "replicate", "k_est", "start",
    "data_seed", "split_seed", "model_seed", "expected_fit_count",
    "actual_fit_count", "anchor_test_hash", "anchor_train_hash",
    "fit_config_hash", "fit_payload_a_hash", "fit_payload_b_hash",
    "initialization_equal", "final_outputs_equal", "internal_retry",
    "warning_count", "q_failure", "nan_occurred", "finite_state",
    "canary_atol", "canary_rtol", "boundary_version",
    "execution_mode", "real_canary_fits_executed",
)


def require_canary_pass_evidence(out_dir: Path, authorization: Any,
                                 anchors: Mapping[int, AnchorMask] | None = None,
                                 *, current_run_code_sha: str,
                                 test_only: bool = False) -> dict[str, Any]:
    """Smoke may not begin without a PASS canary from THIS execution lineage.

    "A canary.json exists" is not enough: the evidence must carry the same
    approved baseline, protocol hash, execution issue, seed block AND the same
    run-code SHA as the smoke run that is about to start.  A format-valid but
    different SHA means the canary was produced by different code, so it is not
    evidence about this execution at all.
    """

    directory = require_existing_smoke_artifact_dir(out_dir)
    payload = read_json_artifact(directory / "canary.json")

    missing = [key for key in CANARY_EVIDENCE_REQUIRED_KEYS if key not in payload]
    _require(not missing, f"canary evidence is incomplete; missing {missing}")

    _require(payload["artifact_version"] == SMOKE_ARTIFACT_VERSION,
             "canary evidence artifact version changed")
    _require(payload["status"] == CANARY_STATUS_PASS,
             f"canary evidence status is not PASS: {payload['status']!r}")
    _require(payload["execution_issue_number"] == SMOKE_EXECUTION_ISSUE_NUMBER,
             "canary evidence belongs to a different execution issue")
    _require(payload["protocol_origin_issue_number"] == SMOKE_PROTOCOL_ISSUE_NUMBER,
             "canary evidence protocol origin issue changed")
    _require(payload["approved_scientific_main_sha"] == authorization.approved_main_sha
             == trusted_main_sha_for(test_only),
             "canary evidence was produced under a different approved baseline")
    # Run-code identity: same code produced the canary and is starting the smoke.
    _require_full_commit_sha(payload["run_code_sha"], "canary evidence run code SHA")
    _require_full_commit_sha(current_run_code_sha, "current smoke run code SHA")
    _require(payload["run_code_sha"] == current_run_code_sha,
             "canary evidence run code SHA does not match the current smoke execution: "
             f"{payload['run_code_sha']} != {current_run_code_sha}")
    _require(payload["protocol_hash"] == authorization.protocol_hash == smoke_protocol_hash(),
             "canary evidence protocol hash does not match this execution")

    _require(payload["estimand"] == SMOKE_ESTIMAND and payload["role"] == SMOKE_ROLE,
             "canary evidence estimand/role changed")
    _require(payload["k_true"] == SMOKE_K_TRUE and payload["replicate"] == SMOKE_REPLICATE,
             "canary evidence cell changed")
    _require(payload["k_est"] == CANARY_K_EST and payload["start"] == CANARY_START,
             "canary evidence K_est/start changed")
    _require(payload["data_seed"] == smoke_data_seed(SMOKE_K_TRUE, SMOKE_REPLICATE),
             "canary evidence data seed changed")
    _require(payload["split_seed"] == SMOKE_SPLIT_SEED, "canary evidence split seed changed")
    _require(payload["model_seed"] == CANARY_MODEL_SEED, "canary evidence model seed changed")

    expected_mode = "test_only" if test_only else "real"
    _require(payload["execution_mode"] == expected_mode,
             f"canary evidence execution mode is {payload['execution_mode']!r}, "
             f"expected {expected_mode!r}")
    _require(payload["expected_fit_count"] == EXPECTED_CANARY_FITS,
             "canary evidence expected fit count changed")
    _require(payload["actual_fit_count"] == EXPECTED_CANARY_FITS,
             "canary evidence did not record exactly two canary fits")
    _require(payload["real_canary_fits_executed"] == (0 if test_only else EXPECTED_CANARY_FITS),
             "canary evidence real-fit count does not match the execution mode")

    anchor = (read_phase7e_anchor_masks() if anchors is None else anchors)[SMOKE_REPLICATE]
    _require(payload["anchor_test_hash"] == anchor.test_mask_hash,
             "canary evidence test anchor hash differs from Phase 7e")
    _require(payload["anchor_train_hash"] == anchor.train_mask_hash,
             "canary evidence train anchor hash differs from Phase 7e")

    _require(payload["initialization_equal"] is True, "canary initialization was not invariant")
    _require(payload["final_outputs_equal"] is True, "canary final outputs were not invariant")
    _require(payload["internal_retry"] == 0, "canary evidence records an internal retry")
    _require(payload["warning_count"] == 0, "canary evidence records warnings")
    _require(payload["q_failure"] is False, "canary evidence records a Q failure")
    _require(payload["nan_occurred"] is False, "canary evidence records a NaN state")
    _require(payload["finite_state"] is True, "canary evidence records a nonfinite state")
    _require(payload["fit_payload_a_hash"] != payload["fit_payload_b_hash"],
             "canary A/B payloads are identical; the canary representation did not vary")

    # Frozen Phase 7e tolerances, never relaxed after seeing output.
    _require(payload["canary_atol"] == float(CANARY_ATOL), "canary atol was changed")
    _require(payload["canary_rtol"] == float(CANARY_RTOL), "canary rtol was changed")
    _require(payload["boundary_version"] == LEAKAGE_BOUNDARY_VERSION,
             "canary evidence boundary version changed")
    return payload


def require_canary_audit_pass(out_dir: Path, authorization: Any,
                              canary_payload: Mapping[str, Any], *,
                              current_run_code_sha: str,
                              test_only: bool = False) -> dict[str, Any]:
    """Smoke may not begin without an INDEPENDENT canary audit that passed.

    ``require_canary_pass_evidence`` is producer-side validation: the same code
    that wrote the evidence deciding the evidence is fine.  The frozen order
    (Issue #55) puts an independent audit between the canary and the smoke, so
    this reads the verdict that ``audit_k_true_robustness_sweep`` published and
    binds it to this execution's baseline, protocol, issues and run-code SHA.

    This module never writes ``canary_audit.json``.
    """

    directory = require_existing_smoke_artifact_dir(out_dir)
    payload = read_json_artifact(directory / CANARY_AUDIT_FILENAME)

    missing = [key for key in CANARY_AUDIT_REQUIRED_KEYS if key not in payload]
    _require(not missing, f"canary audit verdict is incomplete; missing {missing}")

    _require(payload["audit_version"] == CANARY_AUDIT_VERSION,
             "canary audit version changed")
    _require(payload["status"] == "PASS",
             f"the independent canary audit did not pass: {payload['status']!r}")
    _require(payload["blocker_count"] == 0 and payload["high_count"] == 0,
             "the independent canary audit recorded blocking findings: "
             f"{payload['blocker_count']} BLOCKER / {payload['high_count']} HIGH")
    # MEDIUM findings do not block a PASS; the count is still an integer >= 0,
    # and an equal float/bool/string must not be accepted in its place.
    _require(type(payload["medium_count"]) is int and payload["medium_count"] >= 0,
             "the canary audit medium_count is not a non-negative integer: "
             f"{payload['medium_count']!r}")

    _require(payload["approved_scientific_main_sha"] == authorization.approved_main_sha
             == trusted_main_sha_for(test_only),
             "the canary audit was produced under a different approved baseline")
    _require(payload["protocol_hash"] == authorization.protocol_hash == smoke_protocol_hash(),
             "the canary audit protocol hash does not match this execution")
    _require(payload["execution_issue"] == SMOKE_EXECUTION_ISSUE_NUMBER,
             "the canary audit belongs to a different execution issue")
    _require(payload["protocol_origin_issue"] == SMOKE_PROTOCOL_ISSUE_NUMBER,
             "the canary audit protocol origin issue changed")

    _require_full_commit_sha(payload["run_code_sha"], "canary audit run code SHA")
    _require(payload["run_code_sha"] == current_run_code_sha,
             "the canary audit run code SHA does not match the current smoke execution: "
             f"{payload['run_code_sha']} != {current_run_code_sha}")
    _require(payload["run_code_sha"] == canary_payload["run_code_sha"],
             "the canary audit and the canary evidence name different run-code SHAs")

    expected_mode = "test_only" if test_only else "real"
    _require(payload["canary_execution_mode"] == expected_mode,
             f"the canary audit covers a {payload['canary_execution_mode']!r} canary, "
             f"expected {expected_mode!r}")
    _require(payload["canary_status"] == canary_payload["status"] == CANARY_STATUS_PASS,
             "the canary audit did not cover a PASS canary")

    expected_real = 0 if test_only else EXPECTED_CANARY_FITS
    _require(payload["expected_canary_fits"] == EXPECTED_CANARY_FITS,
             "the canary audit expected fit count changed")
    _require(payload["actual_canary_fits"] == expected_real,
             "the canary audit fit count does not match the execution mode")

    audited = list(payload["audited_files"])
    for name in ("authorization.json", "canary.json"):
        _require(name in audited, f"the canary audit did not read {name}")

    # Content binding: the verdict names the exact bytes the independent audit
    # read.  If either source artifact changed afterwards the PASS is stale, and
    # this module can only detect that by rehashing the files itself.
    for name, key in (("authorization.json", "authorization_content_sha256"),
                      ("canary.json", "canary_content_sha256")):
        recorded = payload[key]
        _require(type(recorded) is str and len(recorded) == 64
                 and all(character in "0123456789abcdef" for character in recorded),
                 f"the canary audit {key} is not 64 lowercase hex characters: {recorded!r}")
        source = directory / name
        _require(source.is_file(), f"{name} is missing from the smoke artifact directory")
        current = hashlib.sha256(source.read_bytes()).hexdigest()
        _require(recorded == current,
                 f"{name} changed after the independent canary audit: the verdict binds "
                 f"{recorded}, the file now hashes to {current}")
    return payload


# ---------------------------------------------------------------------------
# production execution wiring (authorized by Issue #55 for 2 canary + 6 smoke)
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    from datetime import datetime, timezone  # noqa: PLC0415

    return datetime.now(timezone.utc).isoformat()


def _require_execution_preconditions(run_code_sha: str) -> None:
    _require_full_commit_sha(run_code_sha, "run code SHA")
    _require(working_tree_is_clean(),
             "the working tree is dirty; refusing to start a real execution")
    _require(approved_baseline_is_ancestor(run_code_sha),
             "the approved scientific baseline is not an ancestor of this commit")


def _resolve_fit_adapter(test_adapter: Any, test_only: bool) -> Any:
    """Construct the fit adapter -- the LAST step before the first fit.

    This is the ONLY place in the module that constructs
    ``AuthorizedEMFitAdapter``.  Every production route therefore has to pass
    through ``_execute_real_canary`` / ``_execute_real_smoke``, which means it
    has already cleared the full preflight and, for smoke, the canary lineage.
    A static test enumerates the construction sites to keep it that way.
    """

    if test_only:
        _require(type(test_adapter) is _TestAuthorizedFitAdapter,
                 "a test-only execution requires the Phase 7e test adapter")
        return test_adapter
    _require(test_adapter is None,
             "a production execution must not be handed a pre-built adapter")
    return AuthorizedEMFitAdapter()


def _execute_real_canary(authorization: Any, out_dir: Path | None, *,
                         test_adapter: Any, test_only: bool,
                         run_code_sha: str) -> dict[str, Any]:
    """Full no-fit preflight -> reserve dir -> evidence -> adapter -> 2 fits.

    Nothing observable happens until every gate that does not need a fit has
    passed: no directory is created, no evidence is written and no adapter is
    constructed while a preflight failure is still possible.
    """

    started_at = _utc_now()
    # --- phase 1: no side effects at all ---------------------------------
    cell = prepare_smoke_cell(authorization, test_only=test_only)
    _require_full_commit_sha(run_code_sha, "run code SHA")
    # --- phase 2: reserve the directory and record the authorization -----
    directory = require_new_smoke_artifact_dir(out_dir)
    write_json_artifact(directory / "authorization.json",
                        build_authorization_payload(authorization, run_code_sha))
    # --- phase 3: only now may a fit-capable adapter exist ---------------
    adapter = _resolve_fit_adapter(test_adapter, test_only)
    report = _run_real_canary(authorization, adapter=adapter, test_only=test_only, cell=cell)
    payload = build_canary_payload(report, run_code_sha)
    write_json_artifact(directory / "canary.json", payload)
    return {
        "mode": "canary",
        "artifact_directory": str(directory),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "canary_status": payload["status"],
        "real_canary_fits_executed": report.real_canary_fits_executed,
        "real_smoke_fits_executed": 0,
        "em_fits_executed": report.real_canary_fits_executed,
    }


def _execute_real_smoke(authorization: Any, out_dir: Path | None, *,
                        test_adapter: Any, test_only: bool,
                        run_code_sha: str) -> dict[str, Any]:
    """Full no-fit preflight -> canary lineage -> adapter -> 6 fits -> artifacts.

    Smoke reuses the canary's directory; it never creates a new one.
    """

    started_at = _utc_now()
    # --- phase 1: no side effects at all ---------------------------------
    cell = prepare_smoke_cell(authorization, test_only=test_only)
    _require_full_commit_sha(run_code_sha, "run code SHA")
    directory = require_existing_smoke_artifact_dir(out_dir)
    canary_payload = require_canary_pass_evidence(
        directory, authorization, current_run_code_sha=run_code_sha, test_only=test_only)
    # The INDEPENDENT canary verdict, produced by the audit module, gates the
    # six fits.  A valid canary.json alone is not enough.
    canary_audit = require_canary_audit_pass(
        directory, authorization, canary_payload,
        current_run_code_sha=run_code_sha, test_only=test_only)
    # --- phase 2: only now may a fit-capable adapter exist ---------------
    adapter = _resolve_fit_adapter(test_adapter, test_only)
    report = _run_real_smoke(authorization, adapter=adapter, test_only=test_only, cell=cell)

    rows = build_smoke_artifact_rows_from_evidence(report, canary_payload, run_code_sha)
    write_csv_artifact(directory / "smoke_fit_results.csv", SMOKE_ARTIFACT_COLUMNS, rows)
    write_json_artifact(directory / "smoke_summary.json",
                        build_smoke_summary_payload(report, run_code_sha))
    completed_at = _utc_now()
    write_json_artifact(directory / "runinfo.json", build_smoke_runinfo_payload(
        run_code_sha=run_code_sha,
        out_dir=directory,
        requested_command="--smoke",
        invocation_mode="cli",
        started_at=started_at,
        completed_at=completed_at,
        working_tree_clean=working_tree_is_clean(),
        actual_canary_fits=canary_payload["real_canary_fits_executed"],
        actual_smoke_fits=report.real_smoke_fits_executed,
        canary_audit_status=canary_audit["status"],
    ))
    return {
        "mode": "smoke",
        "artifact_directory": str(directory),
        "started_at": started_at,
        "completed_at": completed_at,
        "selected_k": report.selected_k,
        "selected_k_interpretation": SELECTED_K_INTERPRETATION,
        "k_recovery_evaluated": False,
        "canary_audit_status": canary_audit["status"],
        "real_canary_fits_executed": canary_payload["real_canary_fits_executed"],
        "real_smoke_fits_executed": report.real_smoke_fits_executed,
        "em_fits_executed": report.real_smoke_fits_executed,
    }


def build_smoke_artifact_rows_from_evidence(smoke: Phase8bSmokeReport,
                                            canary_payload: Mapping[str, Any],
                                            run_code_sha: str) -> list[tuple[Any, ...]]:
    """Six CSV rows, with the canary provenance taken from the PASS evidence."""

    _require(type(smoke) is Phase8bSmokeReport, "artifact rows require a Phase8bSmokeReport")
    _require_full_commit_sha(run_code_sha, "run code SHA")
    canary_provenance = stable_config_hash({
        "k_est": canary_payload["k_est"],
        "start": canary_payload["start"],
        "model_seed": canary_payload["model_seed"],
        "config_hash": canary_payload["fit_config_hash"],
        "payload_a_hash": canary_payload["fit_payload_a_hash"],
        "payload_b_hash": canary_payload["fit_payload_b_hash"],
    })
    rows: list[tuple[Any, ...]] = []
    for row in smoke.rows:
        rows.append((
            run_code_sha, smoke.approved_main_sha, smoke.protocol_hash,
            SMOKE_ESTIMAND, SMOKE_ROLE, SMOKE_K_TRUE, SMOKE_REPLICATE,
            row.k, row.start, row.data_seed, row.split_seed, row.model_seed,
            row.pre_fit_test_hash, row.pre_fit_train_hash,
            row.post_fit_test_hash, row.post_fit_train_hash,
            smoke.anchor_test_hash, smoke.anchor_train_hash,
            LEAKAGE_BOUNDARY_VERSION, row.fit_status,
            row.internal_retry, row.warning_count, row.q_failure, row.nan_occurred,
            row.finite_state, row.heldout_mean_log_score, row.score_config_hash,
            canary_provenance,
            canary_payload["real_canary_fits_executed"], smoke.real_smoke_fits_executed,
        ))
    _require(len(rows) == EXPECTED_SMOKE_FITS, "smoke artifact must have exactly six rows")
    _require(all(len(row) == len(SMOKE_ARTIFACT_COLUMNS) for row in rows),
             "smoke artifact row width does not match the schema")
    return rows


def _run_production_execution(authorization: Any, command: str) -> dict[str, Any]:
    """THE production workflow for a real canary or smoke.

    Both the CLI and the public ``run_real_*`` wrappers delegate here, so there
    is exactly one preflight implementation and no second path that could drift
    away from it.  Nothing here is caller-controlled: the output directory is
    the frozen ``SMOKE_ARTIFACT_DIR`` and the run-code SHA comes from the
    internal provenance source.
    """

    _require(command in ("canary", "smoke"), f"unknown production command {command!r}")
    validate_smoke_execution_authorization(authorization, test_only=False)
    run_code_sha = current_run_code_sha()
    _require_execution_preconditions(run_code_sha)
    if command == "canary":
        return _execute_real_canary(authorization, SMOKE_ARTIFACT_DIR,
                                    test_adapter=None, test_only=False,
                                    run_code_sha=run_code_sha)
    return _execute_real_smoke(authorization, SMOKE_ARTIFACT_DIR,
                               test_adapter=None, test_only=False,
                               run_code_sha=run_code_sha)


def run_real_canary_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Production canary CLI path.  Authorized for exactly 2 real canary fits."""

    authorization = _require_em_authorization(args, "canary")
    return _run_production_execution(authorization, "canary")


def run_real_full_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Production full CLI path.  Authorized for exactly 336 real fits."""

    authorization = _require_em_authorization(args, "full")
    return _run_production_full_execution(authorization)


def run_real_smoke_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Production smoke CLI path.  Authorized for exactly 6 real smoke fits."""

    authorization = _require_em_authorization(args, "smoke")
    return _run_production_execution(authorization, "smoke")


def _execute_real_canary_test_only(authorization: Any, out_dir: Path, *,
                                   adapter: Any, run_code_sha: str) -> dict[str, Any]:
    """Static-test-only path.  It is the ONLY way to redirect the output dir."""

    _require(type(adapter) is _TestAuthorizedFitAdapter, "test canary requires the test adapter")
    _require(out_dir is not None, "the test-only path requires an explicit temp directory")
    return _execute_real_canary(authorization, out_dir, test_adapter=adapter, test_only=True,
                                run_code_sha=run_code_sha)


def _execute_real_smoke_test_only(authorization: Any, out_dir: Path, *,
                                  adapter: Any, run_code_sha: str) -> dict[str, Any]:
    """Static-test-only path.  It is the ONLY way to redirect the output dir."""

    _require(type(adapter) is _TestAuthorizedFitAdapter, "test smoke requires the test adapter")
    _require(out_dir is not None, "the test-only path requires an explicit temp directory")
    return _execute_real_smoke(authorization, out_dir, test_adapter=adapter, test_only=True,
                               run_code_sha=run_code_sha)


def run_smoke_contract() -> dict[str, Any]:
    """Zero-EM: report the frozen execution contract without touching the disk."""

    return {
        "mode": "smoke-contract",
        "em_fits_executed": 0,
        "artifact_version": SMOKE_ARTIFACT_VERSION,
        "artifact_directory": str(SMOKE_ARTIFACT_DIR),
        "artifact_directory_exists": SMOKE_ARTIFACT_DIR.exists(),
        "artifact_files": list(SMOKE_ARTIFACT_FILES),
        "audit_input_files": list(SMOKE_AUDIT_INPUT_FILES),
        "smoke_fit_results_columns": list(SMOKE_ARTIFACT_COLUMNS),
        "canary_evidence_keys": list(CANARY_EVIDENCE_REQUIRED_KEYS),
        "canary_audit_keys": list(CANARY_AUDIT_REQUIRED_KEYS),
        "canary_audit_filename": CANARY_AUDIT_FILENAME,
        "protocol_hash": smoke_protocol_hash(),
        "protocol_origin_issue": SMOKE_PROTOCOL_ISSUE_NUMBER,
        "execution_issue": SMOKE_EXECUTION_ISSUE_NUMBER,
        "approved_scientific_main_sha": APPROVED_SCIENTIFIC_MAIN_SHA,
        "trusted_main_sha_present": current_expected_smoke_main_sha() is not None,
        "execution_authorization_present":
            current_smoke_execution_authorization() is not None,
        "expected_canary_fits": EXPECTED_CANARY_FITS,
        "expected_smoke_fits": EXPECTED_SMOKE_FITS,
        "expected_real_em_budget": EXPECTED_REAL_EM_BUDGET,
        "selected_k_interpretation": SELECTED_K_INTERPRETATION,
        "k_recovery_evaluated": False,
        "real_canary_fits_executed": 0,
        "real_smoke_fits_executed": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 8b K_TRUE robustness harness (Issue #49)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="static checks; EM fits = 0")
    mode.add_argument("--config-gate", action="store_true", help="G/M/MC gates; EM fits = 0")
    mode.add_argument("--record-diagnostics", action="store_true", help="RECORD ONLY; EM fits = 0")
    mode.add_argument("--leakage-gate", action="store_true",
                      help="exercise the leakage boundary with a fake adapter; EM fits = 0")
    mode.add_argument("--smoke-authorization", action="store_true",
                      help="report the smoke authorization gates; EM fits = 0")
    mode.add_argument("--smoke-contract", action="store_true",
                      help="report the frozen execution/artifact contract; EM fits = 0")
    mode.add_argument("--full-preflight", action="store_true",
                      help="every zero-EM gate the 336-fit sweep must clear; EM fits = 0")
    mode.add_argument("--canary", action="store_true", help="leakage falsification (requires EM gates)")
    mode.add_argument("--smoke", action="store_true", help="smoke selection (requires EM gates)")
    mode.add_argument("--full", action="store_true", help="full sweep (requires EM gates)")
    parser.add_argument("--allow-em", action="store_true")
    parser.add_argument("--confirm-k-true-sweep", action="store_true")
    parser.add_argument("--estimand", choices=("A", "B"))
    # --out-dir applies to --record-diagnostics only; a real canary/smoke
    # refuses it (the production artifact directory is frozen).
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="diagnostics output directory; rejected by --canary/--smoke")
    return parser


def _require_em_authorization(args: argparse.Namespace,
                              command: str) -> SmokeExecutionAuthorization:
    """Every EM-bearing command needs a committed execution authorization.

    Returns the validated authorization so the caller can proceed, and fails
    closed otherwise: a missing or invalid record raises before anything else.

    ``--allow-em`` is a speed bump, not proof of anything.  The real gate is
    ``current_smoke_execution_authorization()``, whose committed record is
    validated here against the frozen scientific baseline, the frozen protocol
    hash, the exact fit counts and seeds, the independent review approval and
    the explicit human smoke approval.  No CLI flag, environment variable or
    config file can produce a ``SmokeExecutionAuthorization``, because its
    provenance sentinel is module-private and nothing hands it out.

    The current authorization permits ONLY the frozen execution it describes:
    2 real canary fits and 6 real smoke fits.  Clearing this gate says nothing
    about whether anything has run -- the preflight, lineage, canary-before-
    smoke and independent-audit gates all still apply downstream.

    ``--full`` has its OWN authorization schema (``FullExecutionAuthorization``,
    Issue #59) with its own private authority sentinel, its own reviewed-baseline
    source and its own validator.  A smoke authorization can never be widened
    into a full-run authorization: full is resolved through
    ``current_full_execution_authorization()`` and nothing else, and both of its
    human gates are absent in this branch.
    """

    _require(bool(args.allow_em), f"{command} requires --allow-em")
    if command == "full":
        _require(bool(args.confirm_k_true_sweep), "--full requires --confirm-k-true-sweep")
        _require(args.estimand is not None, "--full requires --estimand")
        _require(args.estimand in active_estimands(),
                 "--estimand is inconsistent with the frozen ESTIMANDS")
        _require(getattr(args, "out_dir", None) is None,
                 "--out-dir is not accepted for a real full run: the production "
                 f"artifact directory is frozen at {FULL_ARTIFACT_DIR}")
        full_authorization = current_full_execution_authorization()
        if full_authorization is None:
            raise HarnessStop(
                "full is not authorized in Phase 8b S3-A: the 336-fit sweep has its "
                "own FullExecutionAuthorization schema, validator and zero-EM "
                "preflight (Issue #59), but no committed record exists and no "
                "reviewed main SHA has been approved for it. A smoke "
                "authorization must never be reused for --full."
            )
        validate_full_execution_authorization(full_authorization, test_only=False)
        run_full_preflight()
        return full_authorization

    _require(command in ("canary", "smoke"), f"unknown EM command {command!r}")
    # HIGH-03: a real execution writes to exactly one frozen directory.  A
    # caller-supplied path could redirect evidence away from the audited
    # location or point at an existing tree, so it is refused outright.
    _require(getattr(args, "out_dir", None) is None,
             f"--out-dir is not accepted for a real {command}: the production "
             f"artifact directory is frozen at {SMOKE_ARTIFACT_DIR}")
    authorization = current_smoke_execution_authorization()
    if authorization is None:
        raise HarnessStop(
            f"{command} is not authorized in Phase 8b S2c: the production canary "
            "and 6-fit smoke path is implemented and the reviewed scientific "
            f"baseline {APPROVED_SCIENTIFIC_MAIN_SHA} is bound, but no committed "
            "SmokeExecutionAuthorization exists (Issue #55). Recording "
            "INDEPENDENT_REVIEW_PASS and HUMAN_SMOKE_APPROVAL against that "
            "reviewed baseline is a separate human gate."
        )
    validate_smoke_execution_authorization(authorization, test_only=False)
    # Authorization first (Issue #53 §26), then the zero-EM gates.  Both are
    # repeated inside ``prepare_smoke_cell``, so a run can never skip them.
    run_validate_only()
    run_config_gate()
    return authorization


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        result: dict[str, Any] = run_validate_only()
    elif args.config_gate:
        result = run_config_gate()
    elif args.record_diagnostics:
        result = run_record_diagnostics(args.out_dir)
    elif args.leakage_gate:
        result = run_leakage_self_check()
        result.setdefault("mode", "leakage-gate")
    elif args.smoke_authorization:
        authorization = current_smoke_authorization()
        result = {
            "mode": "smoke-authorization",
            "em_fits_executed": 0,
            "gates": authorization.as_dict(),
            "missing": authorization.missing(),
            "authorized": authorization.authorized(),
            "human_only_gates": list(HUMAN_ONLY_SMOKE_GATES),
            "execution_authorization_present":
                current_smoke_execution_authorization() is not None,
            "trusted_main_sha_present":
                current_expected_smoke_main_sha() is not None,
            "smoke_protocol_hash": smoke_protocol_hash(),
            "expected_canary_fits": EXPECTED_CANARY_FITS,
            "expected_smoke_fits": EXPECTED_SMOKE_FITS,
            "expected_real_em_budget": EXPECTED_REAL_EM_BUDGET,
            "real_canary_fits_executed": 0,
            "real_smoke_fits_executed": 0,
            "note": "S2b implements the production canary/smoke path; smoke remains "
                    "hard-stopped until a human records INDEPENDENT_REVIEW_PASS and "
                    "HUMAN_SMOKE_APPROVAL against a reviewed main SHA",
        }
    elif args.full_preflight:
        result = run_full_preflight()
    elif args.smoke_contract:
        result = run_smoke_contract()
    elif args.canary:
        # Reachable only with a committed authorization, which does not exist.
        result = run_real_canary_cli(args)
    elif args.smoke:
        result = run_real_smoke_cli(args)
    elif args.full:
        result = run_real_full_cli(args)
    else:  # pragma: no cover - argparse enforces one mode
        raise HarnessStop("no mode selected")
    if not (args.canary or args.smoke or args.full):
        _require(result["em_fits_executed"] == 0, "no-EM path reported a nonzero fit count")
    print(json.dumps(result, sort_keys=True, allow_nan=False, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
