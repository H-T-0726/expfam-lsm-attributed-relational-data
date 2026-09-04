"""Clean true-K n-sweep: forward-only production runner.

Draws data from the CANONICAL clean generator (a literal draw from the model
the estimator assumes), fits the objective-consistent lineage on a held-out
dyad split, and records SEVERAL selection criteria from the same fits.

Protocol: ``reports/identifiability/clean_true_k_experiment_protocol_20260904.md``
Theory:   ``reports/identifiability/true_k_identifiability_hardened_20260904.md``
Review:   ``reports/identifiability/true_k_identifiability_review_20260904.md``

PRIMARY QUESTION (this is not a consistency theorem):

    Under a well-specified finite-sample setting produced by the canonical
    clean generator, how does the selected-K pattern of each pre-registered
    K-selection criterion change as n grows?

Hard rules enforced in code:

* the artifact directory must not exist -- no resume, no overwrite;
* any fit failure stops the whole run and writes failure.json (FAIL CLOSED);
* no retry, no replacement fit, no alternative seed, no tolerance change;
* every seed is a pure function of the frozen protocol and the cell index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
for _path in (str(_EXPERIMENTAL), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data_generator_canonical import (            # noqa: E402
    GENERATOR_VERSION,
    f_scale_for_row_norm,
    generate_canonical_data,
    w_for_matched_y_signal,
)
from run_heldout_k_selection_pilot import (       # noqa: E402
    heldout_bernoulli_mean_log_score,
)

# ===========================================================================
# FROZEN PROTOCOL -- nothing below may change after the protocol hash is
# recorded, and nothing here may be chosen or altered after seeing a result.
# ===========================================================================

PROTOCOL_VERSION = "clean-true-k-protocol-v1"
ARTIFACT_VERSION = "clean-true-k-artifact-v1"
EXPERIMENT_ID = "clean_true_k_asymptotics_20260904"

FAMILY_X = "poisson"
FAMILY_Y = "bernoulli"
D_FEATURES = 15
L_SAMPLES = 5
NUM_ITER = 8
NUMERICS_MODE = "consistent"          # objective-consistent lineage; never legacy 0.5
TEST_RATIO = 0.20

K_TRUE_GRID = (1, 3, 5)
N_GRID = (50, 75, 100, 150)
CANDIDATE_K = (1, 2, 3, 4, 5, 6, 7)
STARTS = (1, 2)

# Signal held constant across K_TRUE so that "does the criterion recover K" is
# not confounded with "is the signal louder" (theory audit 9.2 / 7.1).
ROW_NORM_SQ_TARGET = 0.5              # average ||f_l||^2, fixes E[X_l]
W_REF = 1.0
K_REF = 3
W0_TRUE = -1.0

# Runtime tier chosen from the wall-clock benchmark ALONE, before any
# selection result existed.  Replicates per K_TRUE.
TIER = "A"
REPLICATES_BY_K_TRUE = {1: 4, 3: 4, 5: 8}     # K_TRUE=5 is the primary focus

TIE_TOLERANCE = np.float64(1e-12)

DATA_SEED_BASE = 810000
SPLIT_SEED_BASE = 820000
MODEL_SEED_BASE = 830000

POISSON_LAMBDA_MAX = 1.0e6

ARTIFACT_DIRNAME = "clean_true_k_asymptotics_20260904"
ARTIFACT_DIR = ROOT / "expfam" / "results" / "k_selection" / ARTIFACT_DIRNAME

ARTIFACT_FILES = (
    "protocol.json",
    "manifest.csv",
    "generator_provenance.csv",
    "fit_results.csv",
    "selection_matrix.csv",
    "gram_spectrum.csv",
    "summary.json",
    "runinfo.json",
)


class SweepStop(RuntimeError):
    """Fail-closed stop.  Never caught to retry."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SweepStop(message)


# ---------------------------------------------------------------------------
# frozen protocol hash
# ---------------------------------------------------------------------------

def frozen_protocol() -> dict[str, Any]:
    """Everything that defines the experiment, and nothing that identifies a run."""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "family_x": FAMILY_X,
        "family_y": FAMILY_Y,
        "d": D_FEATURES,
        "L": L_SAMPLES,
        "num_iter": NUM_ITER,
        "numerics_mode": NUMERICS_MODE,
        "test_ratio": TEST_RATIO,
        "k_true_grid": list(K_TRUE_GRID),
        "n_grid": list(N_GRID),
        "candidate_k": list(CANDIDATE_K),
        "starts": list(STARTS),
        "row_norm_sq_target": ROW_NORM_SQ_TARGET,
        "w_ref": W_REF,
        "k_ref": K_REF,
        "w0_true": W0_TRUE,
        "tier": TIER,
        "replicates_by_k_true": {str(k): v for k, v in sorted(REPLICATES_BY_K_TRUE.items())},
        "tie_tolerance": float(TIE_TOLERANCE),
        "data_seed_base": DATA_SEED_BASE,
        "split_seed_base": SPLIT_SEED_BASE,
        "model_seed_base": MODEL_SEED_BASE,
        "poisson_lambda_max": POISSON_LAMBDA_MAX,
        "generator_version": GENERATOR_VERSION,
        "criteria": ["S1_heldout_predictive", "S2_q_based", "S3_plugin_conditional"],
        "structural_diagnostic": "S4_poisson_x_gram_spectrum",
    }


def protocol_hash() -> str:
    payload = json.dumps(frozen_protocol(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_fit_count() -> int:
    cells = sum(REPLICATES_BY_K_TRUE[k] for k in K_TRUE_GRID) * len(N_GRID)
    return cells * len(CANDIDATE_K) * len(STARTS)


def expected_cell_count() -> int:
    return sum(REPLICATES_BY_K_TRUE[k] for k in K_TRUE_GRID) * len(N_GRID)


# ---------------------------------------------------------------------------
# deterministic seeds -- pure functions of the cell index
# ---------------------------------------------------------------------------

def data_seed(k_true: int, n: int, replicate: int) -> int:
    return DATA_SEED_BASE + k_true * 10000 + n * 10 + replicate


def split_seed(k_true: int, n: int, replicate: int) -> int:
    return SPLIT_SEED_BASE + k_true * 10000 + n * 10 + replicate


def model_seed(k_true: int, n: int, replicate: int, k_est: int, start: int) -> int:
    return (MODEL_SEED_BASE + k_true * 100000 + n * 1000
            + replicate * 100 + k_est * 10 + start)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def build_masks(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic dyad holdout; identical across candidate K within a cell."""

    rng = np.random.default_rng(seed)
    upper = np.triu_indices(n, 1)
    n_pairs = int(upper[0].size)
    n_test = int(round(TEST_RATIO * n_pairs))
    _require(0 < n_test < n_pairs, f"degenerate split at n={n}")
    chosen = rng.permutation(n_pairs)[:n_test]
    flat = np.zeros(n_pairs, dtype=bool)
    flat[chosen] = True

    test = np.zeros((n, n), dtype=bool)
    test[upper] = flat
    test = test | test.T
    train = np.ones((n, n), dtype=bool)
    np.fill_diagonal(train, False)
    train = train & ~test

    _require(np.array_equal(test, test.T) and not test.diagonal().any(),
             "test mask is not a symmetric hollow matrix")
    _require(not np.any(train & test), "train and test masks overlap")
    return train, test


def stable_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes() + str(contiguous.shape).encode()).hexdigest()


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------

def s3_plugin_conditional(model: Any, X: np.ndarray, Y: np.ndarray,
                          Z_final: np.ndarray, F: np.ndarray,
                          w0: float, w: float, num_params: int, n: int) -> float:
    """A fully specified plug-in CONDITIONAL criterion.

    -2 * [ ln p(X | Z, F) + ln p(Y_train | Z, w0, w) ] + num_params * ln n,
    evaluated at the final latent draw.  The prior term ln p(Z) is NOT included
    and Z is NOT integrated out, which is what distinguishes this from S2.

    IMPORTANT: this is NOT the source paper's Eq.(26)/Eq.(16) criterion.  The
    paper's evaluation procedure -- which Z, and whether an MC average is taken
    -- is not recoverable from the paper text (see
    reports/k_selection_theory/paper_bic_reproduction_alignment_20260904.md),
    so no claim of alignment is made.  S3 is included only as a third,
    self-contained criterion to contrast with S1 and S2.  It must never be
    called a BIC.
    """

    single = Z_final[:, :, np.newaxis]
    ll_x = float(model.calc_log_likelihood_X(X, single, F))
    ll_y = float(model.calc_log_likelihood_Y(Y, single, w0, w))
    return -2.0 * (ll_x + ll_y) + num_params * math.log(n)


def select_k(mean_by_k: dict[int, float]) -> tuple[int, list[int]]:
    """Frozen selector: best two-start mean, tie tolerance 1e-12, smallest K."""

    values = {int(k): np.float64(v) for k, v in mean_by_k.items()}
    _require(set(values) == set(CANDIDATE_K), "selector needs every candidate K")
    best = max(values.values())
    ties = sorted(k for k, v in values.items() if best - v <= TIE_TOLERANCE)
    _require(bool(ties), "selector produced no tie candidate")
    return min(ties), ties


def poisson_x_gram_spectrum(X: np.ndarray) -> dict[str, Any]:
    """S4 structural diagnostic -- NOT a criterion, and it selects no K.

    Sample-moment version of the population identity proved in the theory audit
    (P1): ||f_l||^2 = 2 log E[X_l] and f_l^T f_m = log(E[X_l X_m]/(E[X_l]E[X_m])).
    The estimator is unconstrained, so at finite n the estimated Gram routinely
    leaves the PSD cone and its unthresholded rank is d, not K.  No rank
    threshold is applied here and none may be chosen after seeing results.
    """

    x = np.asarray(X, dtype=np.float64)
    d = x.shape[1]
    mean_x = x.mean(axis=0)
    if np.any(mean_x <= 0.0):
        return {"status": "undefined_zero_column_mean",
                "eigenvalues": None, "min_eigenvalue": None,
                "unthresholded_rank": None, "gap_ratios": None}
    gram = np.empty((d, d))
    for l in range(d):
        gram[l, l] = 2.0 * math.log(mean_x[l])
        for m in range(l + 1, d):
            cross = float(np.mean(x[:, l] * x[:, m]))
            if cross <= 0.0:
                return {"status": "undefined_zero_cross_moment",
                        "eigenvalues": None, "min_eigenvalue": None,
                        "unthresholded_rank": None, "gap_ratios": None}
            gram[l, m] = gram[m, l] = math.log(cross / (mean_x[l] * mean_x[m]))
    eig = np.linalg.eigvalsh(gram)[::-1]
    gaps = [float(eig[i] / eig[i + 1]) if eig[i + 1] != 0.0 else None
            for i in range(d - 1)]
    return {
        "status": "ok",
        "eigenvalues": [float(v) for v in eig],
        "min_eigenvalue": float(np.min(eig)),
        "unthresholded_rank": int(np.linalg.matrix_rank(gram)),
        "gap_ratios": gaps,
    }


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    k_true: int
    n: int
    replicate: int


@dataclass(frozen=True)
class FitRow:
    fit_index: int
    k_true: int
    n: int
    replicate: int
    k_est: int
    start: int
    data_seed: int
    split_seed: int
    model_seed: int
    heldout_mean_log_score: float
    q_strict: float
    s2_q_based: float
    s3_plugin_conditional: float
    num_params: int
    nan_occurred: bool
    nan_count: int
    q_bic_failed: bool
    failure_reason: str
    runtime_s: float


def build_cells() -> list[Cell]:
    cells: list[Cell] = []
    for k_true in K_TRUE_GRID:
        for n in N_GRID:
            for replicate in range(1, REPLICATES_BY_K_TRUE[k_true] + 1):
                cells.append(Cell(k_true, n, replicate))
    _require(len(cells) == expected_cell_count(),
             f"cell count {len(cells)} != {expected_cell_count()}")
    return cells


def build_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for cell in build_cells():
        for k_est in CANDIDATE_K:
            for start in STARTS:
                index += 1
                rows.append({
                    "fit_index": index,
                    "K_TRUE": cell.k_true, "n": cell.n, "replicate": cell.replicate,
                    "K": k_est, "start": start,
                    "data_seed": data_seed(cell.k_true, cell.n, cell.replicate),
                    "split_seed": split_seed(cell.k_true, cell.n, cell.replicate),
                    "model_seed": model_seed(cell.k_true, cell.n, cell.replicate,
                                             k_est, start),
                })
    _require(len(rows) == expected_fit_count(),
             f"manifest {len(rows)} != {expected_fit_count()}")
    seeds = [r["model_seed"] for r in rows]
    _require(len(set(seeds)) == len(seeds), "model seeds are not unique")
    return rows


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT)
    _require(result.returncode == 0, "git rev-parse failed")
    return result.stdout.strip()


def working_tree_clean() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                            text=True, cwd=ROOT)
    return result.returncode == 0 and result.stdout.strip() == ""


def environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})


# ---------------------------------------------------------------------------
# preflight (zero EM)
# ---------------------------------------------------------------------------

def run_preflight() -> dict[str, Any]:
    manifest = build_manifest()
    per_k_true: dict[int, int] = {}
    for row in manifest:
        per_k_true[row["K_TRUE"]] = per_k_true.get(row["K_TRUE"], 0) + 1
    return {
        "mode": "preflight",
        "protocol_hash": protocol_hash(),
        "protocol_version": PROTOCOL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_directory": str(ARTIFACT_DIR),
        "artifact_directory_exists": ARTIFACT_DIR.exists(),
        "tier": TIER,
        "expected_fits": expected_fit_count(),
        "expected_cells": expected_cell_count(),
        "fits_by_k_true": {str(k): v for k, v in sorted(per_k_true.items())},
        "candidate_k": list(CANDIDATE_K),
        "n_grid": list(N_GRID),
        "k_true_grid": list(K_TRUE_GRID),
        "em_fits_executed": 0,
        "generator_version": GENERATOR_VERSION,
        "numerics_mode": NUMERICS_MODE,
        "run_code_sha": git_head(),
        "working_tree_clean": working_tree_clean(),
    }


# ---------------------------------------------------------------------------
# production
# ---------------------------------------------------------------------------

def run_production(*, allow_em: bool, confirm: bool) -> dict[str, Any]:
    _require(allow_em, "production requires --allow-em")
    _require(confirm, "production requires --confirm-clean-true-k-sweep")
    _require(not ARTIFACT_DIR.exists(),
             f"artifact directory already exists; refusing to overwrite or resume: "
             f"{ARTIFACT_DIR}")

    import em_runner                                    # noqa: PLC0415

    started_at = time.time()
    run_code_sha = git_head()
    tree_clean_before = working_tree_clean()
    manifest = build_manifest()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)

    write_json(ARTIFACT_DIR / "protocol.json", {
        "protocol": frozen_protocol(),
        "protocol_hash": protocol_hash(),
        "protocol_version": PROTOCOL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "expected_fits": expected_fit_count(),
        "expected_cells": expected_cell_count(),
        "run_code_sha": run_code_sha,
        "working_tree_clean_before_execution": tree_clean_before,
        "environment": environment(),
        "failure_policy": [
            "stop_immediately", "no_retry", "no_replacement_fit",
            "no_seed_rescue", "no_tolerance_change", "no_resume",
            "preserve_partial_evidence", "rerun_requires_a_new_attempt_id",
        ],
    })
    write_csv(ARTIFACT_DIR / "manifest.csv",
              ["fit_index", "K_TRUE", "n", "replicate", "K", "start",
               "data_seed", "split_seed", "model_seed"], manifest)

    fit_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    gram_rows: list[dict[str, Any]] = []
    cell_scores: dict[tuple[int, int, int], dict[str, dict[int, list[float]]]] = {}
    attempted = 0
    index = 0

    try:
        for cell in build_cells():
            scale = f_scale_for_row_norm(ROW_NORM_SQ_TARGET, d=D_FEATURES, k=cell.k_true)
            w_true = w_for_matched_y_signal(W_REF, k=cell.k_true, k_ref=K_REF)
            dataset = generate_canonical_data(
                n=cell.n, d=D_FEATURES, k=cell.k_true,
                seed=data_seed(cell.k_true, cell.n, cell.replicate),
                family_x=FAMILY_X, family_y=FAMILY_Y,
                f_scale=scale, w0=W0_TRUE, w=w_true,
                poisson_lambda_max=POISSON_LAMBDA_MAX)
            train, test = build_masks(cell.n, split_seed(cell.k_true, cell.n, cell.replicate))
            upper = np.triu(np.ones((cell.n, cell.n), dtype=bool), k=1)
            test_rows, test_cols = np.where(np.triu(test, 1))
            y_test = dataset.Y[test_rows, test_cols]

            provenance_rows.append({
                "K_TRUE": cell.k_true, "n": cell.n, "replicate": cell.replicate,
                "data_seed": data_seed(cell.k_true, cell.n, cell.replicate),
                "split_seed": split_seed(cell.k_true, cell.n, cell.replicate),
                "generator_version": dataset.metadata["generator_version"],
                "F_rank": dataset.metadata["F_rank"],
                "f_scale": dataset.metadata["f_scale"],
                "mean_f_row_norm_sq": float(np.mean(dataset.metadata["f_row_norms_sq"])),
                "w0_true": dataset.metadata["w0"], "w_true": dataset.metadata["w"],
                "link_policy": dataset.metadata["link_policy"],
                "normalization_policy": dataset.metadata["normalization_policy"],
                "x_mean": float(dataset.X.mean()), "x_max": float(dataset.X.max()),
                "y_density": float(dataset.Y[upper].mean()),
                "n_train_pairs": int(np.triu(train, 1).sum()),
                "n_test_pairs": int(test_rows.size),
                "train_mask_hash": stable_hash(train),
                "test_mask_hash": stable_hash(test),
            })

            spectrum = poisson_x_gram_spectrum(dataset.X)
            gram_rows.append({
                "K_TRUE": cell.k_true, "n": cell.n, "replicate": cell.replicate,
                "status": spectrum["status"],
                "unthresholded_rank": spectrum["unthresholded_rank"],
                "min_eigenvalue": spectrum["min_eigenvalue"],
                "eigenvalues": json.dumps(spectrum["eigenvalues"]),
                "gap_ratios": json.dumps(spectrum["gap_ratios"]),
            })

            key = (cell.k_true, cell.n, cell.replicate)
            cell_scores[key] = {name: {k: [] for k in CANDIDATE_K}
                                for name in ("S1", "S2", "S3")}

            for k_est in CANDIDATE_K:
                for start in STARTS:
                    index += 1
                    attempted += 1
                    seed = model_seed(cell.k_true, cell.n, cell.replicate, k_est, start)
                    t0 = time.perf_counter()
                    result = em_runner.run_em_experimental(
                        X=dataset.X, Y=dataset.Y,
                        family_x=FAMILY_X, family_y=FAMILY_Y,
                        k=k_est, L=L_SAMPLES, num_iter=NUM_ITER, seed=seed,
                        train_mask=train, numerics_mode=NUMERICS_MODE,
                        compute_strict_Q=True)
                    elapsed = time.perf_counter() - t0

                    z_est = np.asarray(result["Z_est"], dtype=np.float64)
                    w0_hat = float(result["w0"])
                    w_hat = float(result["w"])
                    eta_matrix = w0_hat + w_hat * (z_est @ z_est.T)
                    eta_test = eta_matrix[test_rows, test_cols]
                    _require(bool(np.all(np.isfinite(eta_test))),
                             f"non-finite held-out eta at fit {index}")
                    score = heldout_bernoulli_mean_log_score(y_test, eta_test)

                    q_strict = float(result["Q_strict"])
                    num_params = int(result["num_params"])
                    s2 = float(result["bic"])
                    s3 = s3_plugin_conditional(
                        result["model"], dataset.X, dataset.Y, z_est,
                        np.asarray(result["F"], dtype=np.float64),
                        w0_hat, w_hat, num_params, cell.n)

                    _require(math.isfinite(score) and math.isfinite(s2)
                             and math.isfinite(s3),
                             f"non-finite criterion value at fit {index}")

                    row = FitRow(
                        fit_index=index, k_true=cell.k_true, n=cell.n,
                        replicate=cell.replicate, k_est=k_est, start=start,
                        data_seed=data_seed(cell.k_true, cell.n, cell.replicate),
                        split_seed=split_seed(cell.k_true, cell.n, cell.replicate),
                        model_seed=seed,
                        heldout_mean_log_score=score,
                        q_strict=q_strict, s2_q_based=s2,
                        s3_plugin_conditional=s3, num_params=num_params,
                        nan_occurred=bool(result["nan_occurred"]),
                        nan_count=int(result["nan_count"]),
                        q_bic_failed=bool(result.get("q_bic_failed", False)),
                        failure_reason=str(result.get("failure_reason") or ""),
                        runtime_s=round(elapsed, 3),
                    )
                    fit_rows.append(asdict(row))
                    # S2 and S3 are penalised deviances: LOWER is better, so the
                    # frozen argmax selector consumes their negatives.
                    cell_scores[key]["S1"][k_est].append(score)
                    cell_scores[key]["S2"][k_est].append(-s2)
                    cell_scores[key]["S3"][k_est].append(-s3)

                    if index % 50 == 0:
                        print(f"  fit {index}/{expected_fit_count()} "
                              f"K_TRUE={cell.k_true} n={cell.n} r={cell.replicate} "
                              f"K={k_est} start={start} {elapsed:.1f}s", flush=True)

    except BaseException as exc:                       # noqa: BLE001 - fail closed
        write_json(ARTIFACT_DIR / "failure.json", {
            "status": "FAILED",
            "protocol_hash": protocol_hash(),
            "experiment_id": EXPERIMENT_ID,
            "attempted_fit_count": attempted,
            "completed_fit_count": len(fit_rows),
            "expected_fits": expected_fit_count(),
            "failed_fit_index": index,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "retry_count": 0,
            "replacement_fits_executed": 0,
            "run_code_sha": run_code_sha,
            "policy": ["stop_immediately", "no_retry", "no_replacement_fit",
                       "no_seed_rescue", "no_resume", "preserve_partial_evidence"],
        })
        if fit_rows:
            write_csv(ARTIFACT_DIR / "fit_results.csv",
                      list(fit_rows[0].keys()), fit_rows)
        raise

    _require(len(fit_rows) == expected_fit_count(),
             f"produced {len(fit_rows)} fits, expected {expected_fit_count()}")

    write_csv(ARTIFACT_DIR / "fit_results.csv", list(fit_rows[0].keys()), fit_rows)
    write_csv(ARTIFACT_DIR / "generator_provenance.csv",
              list(provenance_rows[0].keys()), provenance_rows)
    write_csv(ARTIFACT_DIR / "gram_spectrum.csv",
              list(gram_rows[0].keys()), gram_rows)

    selection_rows: list[dict[str, Any]] = []
    for (k_true, n, replicate), by_criterion in cell_scores.items():
        for name in ("S1", "S2", "S3"):
            means = {}
            for k_est in CANDIDATE_K:
                values = by_criterion[name][k_est]
                _require(len(values) == len(STARTS),
                         f"{name} cell missing a start at K={k_est}")
                means[k_est] = float(np.mean(np.asarray(values, dtype=np.float64),
                                             dtype=np.float64))
            chosen, ties = select_k(means)
            error = chosen - k_true
            selection_rows.append({
                "criterion": name, "K_TRUE": k_true, "n": n, "replicate": replicate,
                "selected_k": chosen,
                "tie_candidates": json.dumps(ties),
                "signed_error": error, "abs_error": abs(error),
                "label": ("exact" if error == 0 else
                          ("over" if error > 0 else "under")),
                "best_mean": means[chosen],
                "mean_scores": json.dumps({str(k): v for k, v in means.items()}),
            })
    write_csv(ARTIFACT_DIR / "selection_matrix.csv",
              list(selection_rows[0].keys()), selection_rows)

    summary = build_summary(selection_rows, fit_rows)
    write_json(ARTIFACT_DIR / "summary.json", summary)

    completed_at = time.time()
    runinfo = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": protocol_hash(),
        "protocol_version": PROTOCOL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "artifact_directory": str(ARTIFACT_DIR),
        "artifact_files": list(ARTIFACT_FILES),
        "tier": TIER,
        "expected_fits": expected_fit_count(),
        "attempted_fit_count": attempted,
        "completed_fit_count": len(fit_rows),
        "expected_cells": expected_cell_count(),
        "retry_count": 0,
        "replacement_fits_executed": 0,
        "seed_rescue_count": 0,
        "tolerance_relaxations": 0,
        "resumed": False,
        "generator_version": GENERATOR_VERSION,
        "numerics_mode": NUMERICS_MODE,
        "family_x": FAMILY_X, "family_y": FAMILY_Y,
        "run_code_sha": run_code_sha,
        "working_tree_clean_before_execution": tree_clean_before,
        "environment": environment(),
        "started_at": started_at, "completed_at": completed_at,
        "wall_clock_seconds": round(completed_at - started_at, 1),
        "nan_fits": int(sum(1 for r in fit_rows if r["nan_occurred"])),
        "q_bic_failed_fits": int(sum(1 for r in fit_rows if r["q_bic_failed"])),
    }
    write_json(ARTIFACT_DIR / "runinfo.json", runinfo)
    return runinfo


def build_summary(selection_rows: list[dict[str, Any]],
                  fit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_criterion: dict[str, Any] = {}
    for name in ("S1", "S2", "S3"):
        rows = [r for r in selection_rows if r["criterion"] == name]
        by_cell = {}
        for k_true in K_TRUE_GRID:
            for n in N_GRID:
                subset = [r for r in rows if r["K_TRUE"] == k_true and r["n"] == n]
                if not subset:
                    continue
                selected = [r["selected_k"] for r in subset]
                by_cell[f"K{k_true}/n{n}"] = {
                    "selected_k": selected,
                    "exact": sum(1 for s in selected if s == k_true),
                    "replicates": len(selected),
                    "mean_selected_k": float(np.mean(selected)),
                    "under": sum(1 for s in selected if s < k_true),
                    "over": sum(1 for s in selected if s > k_true),
                }
        per_criterion[name] = {
            "by_cell": by_cell,
            "total_exact": sum(1 for r in rows if r["label"] == "exact"),
            "total_cells": len(rows),
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_hash": protocol_hash(),
        "tier": TIER,
        "criteria": per_criterion,
        "k_recovery_is_not_an_audit_gate": True,
        "note": ("S1 is the frozen held-out plug-in predictive score; S2 is the "
                 "legacy Q-based criterion (NOT Schwarz BIC); S3 is a plug-in "
                 "conditional criterion defined in this module and is NOT the "
                 "source paper's Eq.(26). S4 is a structural diagnostic that "
                 "selects no K."),
        "total_fits": len(fit_rows),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--allow-em", action="store_true")
    parser.add_argument("--confirm-clean-true-k-sweep", action="store_true")
    args = parser.parse_args(argv)

    if args.preflight:
        print(json.dumps(run_preflight(), sort_keys=True))
        return 0
    if args.full:
        report = run_production(allow_em=args.allow_em,
                                confirm=args.confirm_clean_true_k_sweep)
        print(json.dumps(report, sort_keys=True))
        return 0
    parser.error("choose --preflight or --full")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
