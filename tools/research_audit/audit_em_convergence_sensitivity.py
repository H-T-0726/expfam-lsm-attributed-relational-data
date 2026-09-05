"""Independent auditor for the EM convergence-sensitivity sweep.

INDEPENDENCE, STATED PRECISELY
------------------------------
This file imports nothing from the runner, the sweep, the EM loop or the model
classes.  Seed formulas, mask construction, the Poisson-X and Bernoulli-Y log
likelihoods, ``Q_strict``, the S2 identity, S1, S3, the selector, the tie rule
and the diagnostic algebra are all RESTATED here from the protocol body, so a
mistake in the runner cannot validate itself by being reused in the audit.

It does import ``data_generator_canonical`` to regenerate the synthetic data.
That module is the frozen generator specification with its own test suite, not
the code under audit, but the independence claim is qualified accordingly: a
defect in the generator itself would not be caught here.  Everything else is
independent, and the strongest numerical check -- exact agreement with the
archived 896-fit baseline at the baseline checkpoint, for every fit -- does not
depend on the generator being correct, only on it being the same one.

WHAT IS RECOMPUTED
------------------
From the archived checkpoint state (Z, the full L latent samples, F, sigma, w0,
w, var_z) plus regenerated data and masks, every checkpoint's S1, S3, Q_strict
and S2 are recomputed and compared.  Mutating any of them fails the audit.
Per-iteration diagnostics are recomputed from ``iteration_diagnostics.csv``, so
the runner's own maxima and stability flags are never taken on trust.

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
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
_GENERATOR_DIR = ROOT / "expfam" / "src" / "experimental"
if str(_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_DIR))

from data_generator_canonical import (                       # noqa: E402
    f_scale_for_row_norm,
    generate_canonical_data,
    w_for_matched_y_signal,
)

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
#: deviances (lower better) and are negated before averaging.
SELECTOR_SPACE = {"S1": ("heldout_mean_log_score", +1),
                  "S2": ("s2_q_based", -1),
                  "S3": ("s3_plugin_conditional", -1)}
IDENTITY_FIELDS = ("k_true", "n", "replicate", "k_est", "start",
                   "data_seed", "split_seed", "model_seed",
                   "train_mask_hash", "test_mask_hash")
DIAGNOSTIC_KEYS = ("gram_rel_change", "f_raw_change", "w_change", "w0_change",
                   "eta_x_rel_change", "eta_y_rel_change")
#: Recomputation tolerance.  The runner and the auditor sum the same terms in
#: different orders, so bit-identity is not achievable; this is far tighter
#: than any difference a real tampering could hide behind.
RECOMPUTE_RTOL = 1e-9
RECOMPUTE_ATOL = 1e-6


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


def bernoulli_log_likelihood(x, eta):
    return x * eta - np.logaddexp(0.0, eta)


def poisson_log_likelihood(x, eta):
    """Canonical Poisson log likelihood WITHOUT the -log(x!) constant."""

    return x * eta - np.exp(eta)


def num_params(k: int, d: int) -> int:
    """Poisson-X / Bernoulli-Y: F only, with the rotation constraint removed."""

    return k * d - k * (k - 1) // 2


def q_strict(X, Y, z_samples, f, var_z, w0, w, train_mask) -> float:
    """Restated ``calc_Q_dual_strict_exp`` for the frozen configuration.

    Poisson-X, Bernoulli-Y, objective-consistent numerics.  ``z_samples`` has
    shape (n, k, L) as the runner stores it.
    """

    n, k, L = z_samples.shape
    obs_upper = np.triu(np.asarray(train_mask, dtype=bool), 1)
    mask_f = np.asarray(train_mask, dtype=np.float64)
    total = 0.0
    for sample in range(L):
        z = z_samples[:, :, sample]
        ln_pz = float(-(n * k / 2.0) * np.log(2.0 * np.pi * var_z)
                      - (1.0 / (2.0 * var_z)) * np.sum(z ** 2))
        ln_px = float(np.sum(poisson_log_likelihood(X, z @ f.T)))
        eta_y = w0 + w * (z @ z.T)
        ln_py = 0.5 * float(np.sum(bernoulli_log_likelihood(Y, eta_y) * mask_f))
        total += ln_pz + ln_px + ln_py
    total /= L
    # X-side strict correction; Bernoulli-Y needs none.
    total -= float(np.sum(gammaln(X + 1)))
    del obs_upper
    return total


def s3_plugin_conditional(X, Y, z_final, f, w0, w, params: int, n: int,
                          train_mask) -> float:
    """Restated S3: the plug-in CONDITIONAL criterion at the final draw.

    No prior term, no integration over Z, and -- like the model's likelihood
    hooks -- no Poisson factorial constant.
    """

    mask_f = np.asarray(train_mask, dtype=np.float64)
    ln_x = float(np.sum(poisson_log_likelihood(X, z_final @ f.T)))
    eta_y = w0 + w * (z_final @ z_final.T)
    ln_y = 0.5 * float(np.sum(bernoulli_log_likelihood(Y, eta_y) * mask_f))
    return -2.0 * (ln_x + ln_y) + params * math.log(n)


def heldout_mean_log_score(y_test, eta_test) -> float:
    return float(np.mean(bernoulli_log_likelihood(
        np.asarray(y_test, dtype=np.float64),
        np.asarray(eta_test, dtype=np.float64)), dtype=np.float64))


def select_k(means: dict[int, float], tie_tolerance: float) -> tuple[int, float]:
    """Frozen selector.  Returns (selected K, winning margin in selector space)."""

    best = max(means.values())
    ties = sorted(k for k, v in means.items() if best - v <= tie_tolerance)
    chosen = min(ties)
    others = [v for k, v in means.items() if k != chosen]
    margin = best - max(others) if others else float("nan")
    return chosen, float(margin)


def relative_change(now, prev) -> float:
    return (float(np.linalg.norm(np.asarray(now) - np.asarray(prev)))
            / (1.0 + float(np.linalg.norm(np.asarray(prev)))))


# ---- data regeneration ----------------------------------------------------

class Regenerator:
    """Deterministic data and masks per (n, replicate), cached."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self._cache: dict[tuple[int, int], Any] = {}

    def get(self, n: int, replicate: int):
        key = (n, replicate)
        if key not in self._cache:
            body = self.body
            k_true = int(body["k_true"])
            scale = f_scale_for_row_norm(float(body["row_norm_sq_target"]),
                                         d=int(body["d"]), k=k_true)
            w_true = w_for_matched_y_signal(float(body["w_ref"]), k=k_true,
                                            k_ref=int(body["k_ref"]))
            dataset = generate_canonical_data(
                n=n, d=int(body["d"]), k=k_true,
                seed=data_seed(int(body["data_seed_base"]), k_true, n, replicate),
                family_x=body["family_x"], family_y=body["family_y"],
                f_scale=scale, w0=float(body["w0_true"]), w=w_true,
                poisson_lambda_max=float(body["poisson_lambda_max"]))
            train, test = build_masks(
                n, split_seed(int(body["split_seed_base"]), k_true, n, replicate),
                float(body["test_ratio"]))
            rows, cols = np.where(np.triu(test, 1))
            self._cache[key] = (dataset, train, test, rows, cols,
                                dataset.Y[rows, cols])
        return self._cache[key]


# ---- the audit ------------------------------------------------------------

def audit(run_dir: Path) -> dict[str, Any]:
    a = Audit()
    if not a.require(run_dir.is_dir(), "BLOCKER", "run_dir", f"missing: {run_dir}"):
        return finish(a, run_dir)
    present = {p.name for p in run_dir.iterdir()}

    for name in ("protocol.json", "runinfo.json", "attempt.json",
                 "checkpoint_results.csv", "fit_results.csv",
                 "iteration_diagnostics.csv", "replication_vs_baseline.json"):
        a.require(name in present, "BLOCKER", "artifact_set", f"missing {name}")
    if a.count("BLOCKER"):
        return finish(a, run_dir)

    protocol_doc = read_json(run_dir / "protocol.json")
    runinfo = read_json(run_dir / "runinfo.json")
    attempt = read_json(run_dir / "attempt.json")
    checkpoints = read_csv(run_dir / "checkpoint_results.csv")
    fits = read_csv(run_dir / "fit_results.csv")
    iterations = read_csv(run_dir / "iteration_diagnostics.csv")
    replication_doc = read_json(run_dir / "replication_vs_baseline.json")

    # ---- attempt lifecycle -------------------------------------------------
    a.require(attempt.get("status") == "SUCCESS", "BLOCKER", "attempt_status",
              f"attempt status is {attempt.get('status')!r}; only SUCCESS is "
              f"evidence -- RUNNING may be a killed process, and FAILED or "
              f"ABANDONED can never become SUCCESS")
    a.require(bool(attempt.get("attempt_id")), "HIGH", "attempt_id",
              "the attempt carries no id")
    a.require(runinfo.get("attempt_id") == attempt.get("attempt_id"), "HIGH",
              "attempt_id_binding", "runinfo and attempt.json disagree")

    # ---- protocol identity -------------------------------------------------
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

    if FROZEN_PROTOCOL_PATH.exists():
        a.require((run_dir / "protocol.json").read_bytes()
                  == FROZEN_PROTOCOL_PATH.read_bytes(), "BLOCKER",
                  "protocol_bytes_match_freeze_commit",
                  "the artifact's protocol.json is not byte-identical to the "
                  "frozen protocol file")
    else:
        a.note("HIGH", "frozen_protocol_absent",
               f"cannot compare the artifact protocol with {FROZEN_PROTOCOL_PATH}")

    a.require(body["experiment_id"] == EXPECTED_EXPERIMENT_ID, "HIGH",
              "experiment_id", str(body.get("experiment_id")))
    a.require(body["baseline_protocol_hash"] == EXPECTED_BASELINE_PROTOCOL_HASH,
              "BLOCKER", "baseline_protocol_hash",
              "the frozen baseline is not the archived clean true-K sweep")
    a.require(body.get("runtime_abort_rule") == "none", "MEDIUM",
              "runtime_abort_rule",
              "runtime must not include or exclude an individual fit")

    ladder = tuple(int(c) for c in body["checkpoints"])
    max_iter = int(body["max_iter"])
    candidate_k = tuple(int(k) for k in body["candidate_k"])
    starts = tuple(int(s) for s in body["starts"])
    n_grid = tuple(int(n) for n in body["n_grid"])
    k_true = int(body["k_true"])
    replicates = int(body["replicates"])
    tie_tolerance = float(body["tie_tolerance"])
    baseline_checkpoint = int(body["baseline_checkpoint"])
    primary_checkpoint = int(body["primary_checkpoint"])
    window = int(body["stability_window"])
    taus = [float(t) for t in body["tau_ladder"]]
    d_features = int(body["d"])

    # ---- environment provenance -------------------------------------------
    env = runinfo.get("environment") or {}
    for field in ("python", "numpy", "platform", "machine", "blas"):
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
    a.require(runinfo.get("freeze_tag") == body.get("freeze_tag"), "HIGH",
              "freeze_tag_binding", "runinfo names a different freeze tag")

    # ---- forbidden adaptive stopping --------------------------------------
    a.require(body.get("adaptive_stopping") is False, "BLOCKER",
              "adaptive_stopping_declared", "protocol does not forbid it")
    reasons = {r["stopping_reason"] for r in fits}
    a.require(reasons == {"fixed_budget"}, "BLOCKER", "stopping_reason",
              f"observed stopping reasons: {sorted(reasons)}")
    bad_iters = [r["fit_index"] for r in fits
                 if int(r["actual_iterations"]) != max_iter]
    a.require(not bad_iters, "BLOCKER", "actual_iterations",
              f"{len(bad_iters)} fits did not run the full budget")

    # ---- counts and density ------------------------------------------------
    expected_fits = len(n_grid) * replicates * len(candidate_k) * len(starts)
    a.require(len(fits) == expected_fits, "BLOCKER", "fit_count",
              f"{len(fits)} fits, expected {expected_fits}")
    a.require(len(checkpoints) == expected_fits * len(ladder), "BLOCKER",
              "checkpoint_row_count",
              f"{len(checkpoints)} rows, expected {expected_fits * len(ladder)}")
    a.require(len(iterations) == expected_fits * max_iter, "BLOCKER",
              "iteration_row_count",
              f"{len(iterations)} iteration rows, expected "
              f"{expected_fits * max_iter}")
    indices = [int(r["fit_index"]) for r in fits]
    a.require(sorted(indices) == list(range(1, expected_fits + 1)), "BLOCKER",
              "fit_index_dense", "fit_index is not a dense 1..N")

    fit_by_index = {int(r["fit_index"]): r for r in fits}
    by_fit: dict[int, list[dict[str, str]]] = {}
    for row in checkpoints:
        by_fit.setdefault(int(row["fit_index"]), []).append(row)
    ladder_problems = [idx for idx, rows in by_fit.items()
                       if tuple(sorted(int(r["checkpoint_iteration"])
                                       for r in rows)) != tuple(sorted(ladder))]
    a.require(not ladder_problems, "BLOCKER", "checkpoint_ladder",
              f"{len(ladder_problems)} fits do not carry exactly the ladder")
    a.require(set(by_fit) == set(indices), "BLOCKER", "checkpoint_fit_coverage",
              "checkpoint rows and fit rows cover different fits")

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

    # ---- seeds and masks ---------------------------------------------------
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
              f"{len(seed_problems)} fits have seeds that do not match")

    regenerator = Regenerator(body)
    mask_problems: list[str] = []
    for n in n_grid:
        for rep in range(1, replicates + 1):
            _, train, test, _, _, _ = regenerator.get(n, rep)
            want = (stable_hash(train), stable_hash(test))
            for row in fits:
                if int(row["n"]) == n and int(row["replicate"]) == rep:
                    if (row["train_mask_hash"], row["test_mask_hash"]) != want:
                        mask_problems.append(row["fit_index"])
    a.require(not mask_problems, "BLOCKER", "mask_hashes",
              f"{len(mask_problems)} fits carry masks that do not rebuild")

    # ---- retry / NaN metadata ---------------------------------------------
    retried = [r["fit_index"] for r in fits if int(r["retry_count"]) != 0]
    a.require(not retried, "BLOCKER", "retry_count",
              f"{len(retried)} fits recorded a retry")
    a.require(not [r for r in fits if r["nan_detected"].lower() == "true"],
              "HIGH", "nan_detected", "a fit detected a NaN latent sample")
    a.require(not [r["fit_index"] for r in fits
                   if len(json.loads(r["retry_events"])) != int(r["retry_count"])],
              "HIGH", "retry_event_consistency", "retry_events disagree")
    a.require(not [r["fit_index"] for r in fits
                   if int(r["retry_count"]) == 0
                   and int(r["retry_seed"]) != int(r["model_seed"])],
              "HIGH", "retry_seed", "retry_seed is not the model seed")
    a.require(not [r for r in fits if int(r["attempts_used"]) != 1], "HIGH",
              "attempts_used", "a fit used more than one attempt")

    # ---- iteration diagnostics, and the summaries derived from them -------
    _audit_iteration_diagnostics(a, iterations, checkpoints, fit_by_index,
                                 ladder, max_iter, window, taus)

    # ---- independent recomputation of every checkpoint score --------------
    recomputed_rows = _recompute_checkpoint_scores(
        a, run_dir, checkpoints, body, regenerator, d_features)

    # ---- replication against the archived baseline, ALL fits --------------
    _audit_replication(a, checkpoints, fits, baseline_checkpoint,
                       replication_doc)

    # ---- selection, P1 and P2 ---------------------------------------------
    selection_rows = _recompute_selection(a, checkpoints, ladder, n_grid,
                                          replicates, candidate_k, starts,
                                          tie_tolerance, k_true)
    p2_rows, p2_summary = _compute_p2(a, checkpoints, n_grid, replicates,
                                      candidate_k, starts, baseline_checkpoint,
                                      primary_checkpoint)

    # ---- runtime accounting ------------------------------------------------
    fit_seconds = sum(float(r["runtime_s"]) for r in fits)
    wall = float(runinfo.get("wall_clock_seconds", 0.0))
    a.require(fit_seconds <= wall + 1.0, "HIGH", "runtime_accounting",
              f"per-fit runtimes sum to {fit_seconds:.1f}s but the wall clock "
              f"is {wall:.1f}s")

    # ---- declared limitations ---------------------------------------------
    a.note("LOW", "diagnostics_are_descriptive",
           "the six change columns are budget-stability diagnostics under a "
           "fixed L Monte-Carlo draw. They are not a convergence proof, none "
           "is used for stopping or selection, and none decays to zero")
    a.note("LOW", "generator_is_imported",
           "the auditor regenerates data with the frozen data_generator_"
           "canonical module rather than a second implementation, so a defect "
           "in the generator itself would not be caught here")
    a.note("LOW", "estimand_is_narrow",
           "the contrast is 'continuing the same stochastic MCEM procedure', "
           "not an optimization effect: further iterations redraw L latent "
           "samples, so the two are not separable by this design")

    return finish(a, run_dir, selection_rows, p2_rows, p2_summary,
                  recomputed_rows)


def _audit_iteration_diagnostics(a, iterations, checkpoints, fit_by_index,
                                 ladder, max_iter, window, taus) -> None:
    """Recompute every checkpoint summary from the raw per-iteration rows."""

    by_fit: dict[int, dict[int, dict[str, str]]] = {}
    for row in iterations:
        by_fit.setdefault(int(row["fit_index"]), {})[int(row["iteration"])] = row
    coverage = [idx for idx, rows in by_fit.items()
                if sorted(rows) != list(range(1, max_iter + 1))]
    a.require(not coverage, "BLOCKER", "iteration_coverage",
              f"{len(coverage)} fits lack a dense 1..{max_iter} iteration trace")

    identity_problems = [f"{idx}@{it}"
                         for idx, rows in by_fit.items()
                         for it, row in rows.items()
                         if fit_by_index.get(idx) is None
                         or any(row[f] != fit_by_index[idx][f]
                                for f in ("k_true", "n", "replicate", "k_est",
                                          "start"))]
    a.require(not identity_problems, "BLOCKER", "iteration_identity_binding",
              f"{len(identity_problems)} iteration rows disagree with their fit")

    nonfinite = [f"{r['fit_index']}:{key}" for r in iterations
                 for key in DIAGNOSTIC_KEYS
                 if not math.isfinite(float(r[key])) or float(r[key]) < 0.0]
    a.require(not nonfinite, "HIGH", "iteration_diagnostics_finite",
              f"{len(nonfinite)} non-finite or negative iteration diagnostics")

    value_problems: list[str] = []
    window_problems: list[str] = []
    flag_problems: list[str] = []
    for row in checkpoints:
        idx, iteration = int(row["fit_index"]), int(row["checkpoint_iteration"])
        trace = by_fit.get(idx, {})
        source = trace.get(iteration)
        if source is None:
            value_problems.append(f"{idx}@{iteration}:absent")
            continue
        span = range(max(1, iteration - window + 1), iteration + 1)
        if any(i not in trace for i in span):
            # An incomplete trace is itself the finding; reporting it beats
            # raising, because an auditor that crashes tells the reader
            # nothing about what was wrong.
            value_problems.append(f"{idx}@{iteration}:incomplete_window")
            continue
        for key in DIAGNOSTIC_KEYS:
            if float(row[key]) != float(source[key]):
                value_problems.append(f"{idx}@{iteration}:{key}")
            expected = max(float(trace[i][key]) for i in span)
            if not math.isclose(float(row[f"max_{key}_last{window}"]), expected,
                                rel_tol=1e-12, abs_tol=0.0):
                window_problems.append(f"{idx}@{iteration}:{key}")
        for tau in taus:
            for key, column in (("gram_rel_change", "budget_stable_gram"),
                                ("eta_y_rel_change", "budget_stable_eta_y")):
                expected = max(float(trace[i][key]) for i in span) < tau
                recorded = row[f"{column}_{tau:g}"].lower() == "true"
                if recorded != expected:
                    flag_problems.append(f"{idx}@{iteration}:{column}_{tau:g}")
    a.require(not value_problems, "BLOCKER", "checkpoint_diagnostic_values",
              f"{len(value_problems)} checkpoint diagnostics differ from the "
              f"iteration trace: {value_problems[:5]}")
    a.require(not window_problems, "HIGH", "window_maximum_recomputed",
              f"{len(window_problems)} window maxima are not the maximum of the "
              f"last {window} iteration rows: {window_problems[:5]}")
    a.require(not flag_problems, "HIGH", "budget_stable_flags_recomputed",
              f"{len(flag_problems)} stability flags disagree with the "
              f"recomputed maxima: {flag_problems[:5]}")


def _recompute_checkpoint_scores(a, run_dir, checkpoints, body, regenerator,
                                 d_features) -> int:
    """Recompute S1, S3, Q_strict and S2 for every checkpoint from state."""

    states_path = run_dir / "checkpoint_states.npz"
    if not a.require(states_path.exists(), "BLOCKER", "checkpoint_states",
                     "no checkpoint_states.npz: no later checkpoint can be "
                     "independently recomputed"):
        return 0

    problems: dict[str, list[str]] = {"s1": [], "s3": [], "q": [], "s2": [],
                                      "digest": [], "missing": [], "params": []}
    checked = 0
    with np.load(states_path) as archive_states:
        for row in checkpoints:
            key = row["state_key"]
            if f"{key}_Zsamples" not in archive_states:
                problems["missing"].append(key)
                continue
            n = int(row["n"])
            replicate = int(row["replicate"])
            k_est = int(row["k_est"])
            dataset, train, _test, rows_i, cols_i, y_test = regenerator.get(
                n, replicate)
            z = np.asarray(archive_states[f"{key}_Z"], dtype=np.float64)
            z_samples = np.asarray(archive_states[f"{key}_Zsamples"],
                                   dtype=np.float64)
            f_mat = np.asarray(archive_states[f"{key}_F"], dtype=np.float64)
            w0, w, var_z = (float(v) for v in archive_states[f"{key}_scalars"])
            checked += 1

            if state_digest(z, f_mat, w0, w) != row["checkpoint_state_digest"]:
                problems["digest"].append(key)

            params = num_params(k_est, d_features)
            if params != int(row["num_params"]):
                problems["params"].append(key)

            eta = w0 + w * (z @ z.T)
            got_s1 = heldout_mean_log_score(y_test, eta[rows_i, cols_i])
            if not math.isclose(got_s1, float(row["heldout_mean_log_score"]),
                                rel_tol=RECOMPUTE_RTOL, abs_tol=RECOMPUTE_ATOL):
                problems["s1"].append(key)

            got_q = q_strict(dataset.X, dataset.Y, z_samples, f_mat, var_z,
                             w0, w, train)
            if not math.isclose(got_q, float(row["q_strict"]),
                                rel_tol=RECOMPUTE_RTOL, abs_tol=RECOMPUTE_ATOL):
                problems["q"].append(key)

            got_s2 = -2.0 * got_q + params * math.log(n)
            if not math.isclose(got_s2, float(row["s2_q_based"]),
                                rel_tol=RECOMPUTE_RTOL, abs_tol=RECOMPUTE_ATOL):
                problems["s2"].append(key)

            got_s3 = s3_plugin_conditional(dataset.X, dataset.Y, z, f_mat, w0,
                                           w, params, n, train)
            if not math.isclose(got_s3, float(row["s3_plugin_conditional"]),
                                rel_tol=RECOMPUTE_RTOL, abs_tol=RECOMPUTE_ATOL):
                problems["s3"].append(key)

    a.require(not problems["missing"], "BLOCKER", "checkpoint_state_coverage",
              f"{len(problems['missing'])} checkpoints have no archived state")
    a.require(not problems["digest"], "BLOCKER", "checkpoint_state_digest",
              f"{len(problems['digest'])} archived states do not match their "
              f"digest")
    a.require(not problems["params"], "HIGH", "num_params_recomputed",
              f"{len(problems['params'])} rows disagree with k*d - k(k-1)/2")
    for name, severity in (("s1", "BLOCKER"), ("q", "BLOCKER"),
                           ("s2", "BLOCKER"), ("s3", "BLOCKER")):
        a.require(not problems[name], severity, f"{name}_recomputed",
                  f"{len(problems[name])} checkpoints where the recomputed "
                  f"{name.upper()} differs from the recorded value: "
                  f"{problems[name][:5]}")
    return checked


def _audit_replication(a, checkpoints, fits, baseline_checkpoint,
                       replication_doc) -> None:
    archive_path = BASELINE_RUN_DIR / "fit_results.csv"
    if not a.require(archive_path.exists(), "BLOCKER", "baseline_archive",
                     f"missing {archive_path}"):
        return
    archive: dict[tuple, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in read_csv(archive_path):
        key = (int(row["k_true"]), int(row["n"]), int(row["replicate"]),
               int(row["k_est"]), int(row["start"]))
        if key in archive:
            duplicates.append(str(key))
        archive[key] = row
    a.require(not duplicates, "BLOCKER", "baseline_archive_duplicate_key",
              f"the archive has {len(duplicates)} duplicate keys; the "
              f"comparison target must be unambiguous")

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
              f"compared {compared} of {len(fits)} fits")
    a.require(not mismatches, "BLOCKER", "replication_vs_baseline",
              f"{len(mismatches)} baseline-checkpoint values differ: "
              f"{mismatches[:5]}")
    a.require(replication_doc.get("match") is True, "HIGH",
              "replication_document", "the document does not record a match")


def _recompute_selection(a, checkpoints, ladder, n_grid, replicates,
                         candidate_k, starts, tie_tolerance, k_true):
    """P1, plus start disagreement with its CORRECT definition.

    Start disagreement means the two starts, each running the selector on its
    own scores, choose different K.  It does not mean the raw scores differ --
    they essentially always do, which made the old flag meaningless.
    """

    index: dict[tuple, dict[int, float]] = {}
    for row in checkpoints:
        key = (int(row["checkpoint_iteration"]), int(row["n"]),
               int(row["replicate"]), int(row["k_est"]))
        index.setdefault(key, {})[int(row["start"])] = row

    out: list[dict[str, Any]] = []
    for checkpoint in ladder:
        for n in n_grid:
            for rep in range(1, replicates + 1):
                for name, (field, sign) in SELECTOR_SPACE.items():
                    means: dict[int, float] = {}
                    per_start: dict[int, dict[int, float]] = {
                        s: {} for s in starts}
                    complete = True
                    for k in candidate_k:
                        rows = index.get((checkpoint, n, rep, k), {})
                        if set(rows) != set(starts):
                            a.note("BLOCKER", "selector_inputs",
                                   f"checkpoint {checkpoint} n={n} rep={rep} "
                                   f"K={k}: starts {sorted(rows)}")
                            complete = False
                            continue
                        values = [sign * float(rows[s][field]) for s in starts]
                        means[k] = float(np.mean(np.asarray(values,
                                                            dtype=np.float64),
                                                 dtype=np.float64))
                        for s in starts:
                            per_start[s][k] = sign * float(rows[s][field])
                    if not complete or len(means) != len(candidate_k):
                        continue
                    chosen, margin = select_k(means, tie_tolerance)
                    by_start = {s: select_k(per_start[s], tie_tolerance)[0]
                                for s in starts}
                    out.append({
                        "criterion": name, "checkpoint": checkpoint, "n": n,
                        "replicate": rep, "selected_k": chosen,
                        "signed_error": chosen - k_true,
                        "label": ("exact" if chosen == k_true
                                  else "under" if chosen < k_true else "over"),
                        "winning_margin": margin,
                        "selected_k_start1": by_start[starts[0]],
                        "selected_k_start2": by_start[starts[1]],
                        "start_disagreement": by_start[starts[0]]
                        != by_start[starts[1]],
                    })
    return out


def _compute_p2(a, checkpoints, n_grid, replicates, candidate_k, starts,
                baseline_checkpoint, primary_checkpoint):
    """The frozen P2 aggregation.

    delta(n, r, K) = 0.5 * sum over start of [ S1(...,40) - S1(...,8) ].
    Starts are averaged, never treated as replicates.
    """

    scores: dict[tuple, float] = {}
    for row in checkpoints:
        scores[(int(row["checkpoint_iteration"]), int(row["n"]),
                int(row["replicate"]), int(row["k_est"]), int(row["start"]))] = \
            float(row["heldout_mean_log_score"])

    rows: list[dict[str, Any]] = []
    for n in n_grid:
        for rep in range(1, replicates + 1):
            for k in candidate_k:
                deltas = []
                for s in starts:
                    late = scores.get((primary_checkpoint, n, rep, k, s))
                    early = scores.get((baseline_checkpoint, n, rep, k, s))
                    if late is None or early is None:
                        deltas = []
                        break
                    deltas.append(late - early)
                if len(deltas) != len(starts):
                    a.note("HIGH", "p2_inputs",
                           f"missing S1 for n={n} rep={rep} K={k}")
                    continue
                rows.append({"n": n, "replicate": rep, "k_est": k,
                             "delta": 0.5 * sum(deltas)})

    expected_rows = len(n_grid) * replicates * len(candidate_k)
    a.require(len(rows) == expected_rows, "HIGH", "p2_replicate_rows",
              f"{len(rows)} P2 replicate rows, expected {expected_rows}")

    summary: list[dict[str, Any]] = []
    for n in n_grid:
        for k in candidate_k:
            values = [r["delta"] for r in rows
                      if r["n"] == n and r["k_est"] == k]
            if values:
                summary.append({"n": n, "k_est": k, "replicates": len(values),
                                "mean_delta": float(np.mean(values))})
    a.require(len(summary) == len(n_grid) * len(candidate_k), "HIGH",
              "p2_summary_rows",
              f"{len(summary)} P2 summary rows, expected "
              f"{len(n_grid) * len(candidate_k)}")
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finish(a: Audit, run_dir: Path, selection_rows=None, p2_rows=None,
           p2_summary=None, recomputed=0) -> dict[str, Any]:
    verdict = "PASS" if a.count("BLOCKER") == 0 and a.count("HIGH") == 0 else "FAIL"
    summary: dict[str, Any] = {}
    for row in selection_rows or []:
        bucket = summary.setdefault(f"{row['criterion']}@{row['checkpoint']}",
                                    {"exact": 0, "under": 0, "over": 0,
                                     "start_disagreement": 0})
        bucket[row["label"]] += 1
        bucket["start_disagreement"] += int(bool(row["start_disagreement"]))
    report = {
        "audit_version": "em-convergence-audit-v3",
        "run_dir": str(run_dir),
        "verdict": verdict,
        "blocker_count": a.count("BLOCKER"),
        "high_count": a.count("HIGH"),
        "medium_count": a.count("MEDIUM"),
        "low_count": a.count("LOW"),
        "findings": a.findings,
        "recomputed_checkpoints": recomputed,
        "recomputed_selections": len(selection_rows or []),
        "p2_replicate_rows": len(p2_rows or []),
        "p2_summary_rows": len(p2_summary or []),
        "selection_summary": summary,
    }
    if run_dir.is_dir():
        (run_dir / "audit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        _write_csv(run_dir / "audit_selection_matrix.csv", selection_rows or [])
        _write_csv(run_dir / "audit_p2_replicate.csv", p2_rows or [])
        _write_csv(run_dir / "audit_p2_summary.csv", p2_summary or [])
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
