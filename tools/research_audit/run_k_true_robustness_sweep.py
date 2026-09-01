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
import json
import math
import sys
from dataclasses import asdict, dataclass
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
    FrozenFitConfig,
    FrozenScoreConfig,
    HarnessStop,
    SplitDiagnostics,
    _expected_test_pairs,
    _readonly_copy,
    _require,
    frozen_score_config,
    make_pair_split,
    score_config_hash,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 8b K_TRUE robustness harness (Issue #49)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="static checks; EM fits = 0")
    mode.add_argument("--config-gate", action="store_true", help="G/M/MC gates; EM fits = 0")
    mode.add_argument("--record-diagnostics", action="store_true", help="RECORD ONLY; EM fits = 0")
    mode.add_argument("--canary", action="store_true", help="leakage falsification (requires EM gates)")
    mode.add_argument("--smoke", action="store_true", help="smoke selection (requires EM gates)")
    mode.add_argument("--full", action="store_true", help="full sweep (requires EM gates)")
    parser.add_argument("--allow-em", action="store_true")
    parser.add_argument("--confirm-k-true-sweep", action="store_true")
    parser.add_argument("--estimand", choices=("A", "B"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def _require_em_authorization(args: argparse.Namespace, command: str) -> None:
    """Every EM-bearing command needs an explicit multi-gate authorization."""

    _require(bool(args.allow_em), f"{command} requires --allow-em")
    if command == "full":
        _require(bool(args.confirm_k_true_sweep), "--full requires --confirm-k-true-sweep")
        _require(args.estimand is not None, "--full requires --estimand")
        _require(args.estimand in active_estimands(),
                 "--estimand is inconsistent with the frozen ESTIMANDS")
        # Zero-EM gates must have passed before any fit is authorized.
        run_validate_only()
        run_config_gate()
    raise HarnessStop(
        f"{command} is not authorized in Phase 8b S1: implementation and zero-EM "
        "validation only (Issue #49). Full/smoke/canary execution requires a "
        "separate human approval gate."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        result: dict[str, Any] = run_validate_only()
    elif args.config_gate:
        result = run_config_gate()
    elif args.record_diagnostics:
        result = run_record_diagnostics(args.out_dir)
    elif args.canary:
        _require_em_authorization(args, "canary")
        raise AssertionError("unreachable")
    elif args.smoke:
        _require_em_authorization(args, "smoke")
        raise AssertionError("unreachable")
    elif args.full:
        _require_em_authorization(args, "full")
        raise AssertionError("unreachable")
    else:  # pragma: no cover - argparse enforces one mode
        raise HarnessStop("no mode selected")
    _require(result["em_fits_executed"] == 0, "no-EM path reported a nonzero fit count")
    print(json.dumps(result, sort_keys=True, allow_nan=False, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
