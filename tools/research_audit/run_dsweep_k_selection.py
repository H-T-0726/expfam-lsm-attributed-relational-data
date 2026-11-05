"""Experiment 2: d-sweep of the K-selection criteria C_Q and C_Lap (lineage E).

Question: when the number of Poisson-X attributes d grows (X-side information
grows, and so does the parameter penalty), how do the C_Q and C_Lap selections
change at n=75, K_GEN=5, and how much of the C_Q / C_Lap difference does the
accounting  C_Q - C_Lap = 2 H_Lap + termA + 2 R3  explain?

Design (approved 2026-11-05; frozen in PREREG below BEFORE any d=30/60 fit):
  * d=15 is the A1 / EM-convergence data and state: the checkpoint-8 states of
    the EM-convergence experiment (dad67b1, external npz) are RE-EVALUATED, not
    refit.  Two d=15 fits are re-run ONLY as a replication gate.
  * d=30 / 60 keep Z, Y, the train/test masks and the original 15 X columns
    bit-identical and append 1 / 3 blocks of 15 Poisson columns.  Each block is
    built exactly like the original loadings (reduced QR, all singular values
    f_scale) from its own seed, so every column is exchangeable with an
    original one and F^T F = (d/15) f_scale^2 I exactly.
  * Fits use the EM-convergence experiment's byte-identical instrumented loop
    (em_runner_instrumented.py, blob 5b3f7baa from dad67b1) with max_iter=8,
    checkpoint 8, on_nan='fail', model seeds from the A1 formula (paired
    across d).  That is the code path that produced the reused d=15 states.
  * Failure handling follows the frozen EM-convergence rule: a NaN restart
    request stops the run (no retry, no reseed).  A Laplace evaluation that is
    not OK makes the dataset INCOMPLETE for C_Lap; nothing is imputed.

Usage:  --gates   data gates 1,3,4,5 + d=15 re-evaluation gate + 2 replication fits
        --full    d=15 re-evaluation + 224 fits at d=30/60   (requires PASSED gates)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
_EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
for _p in (str(_EXPERIMENTAL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_clap_clean_nsweep as a1                          # noqa: E402
import run_clap_checkpoint_reeval as cpr                    # noqa: E402
import k_criterion_panel as panel                           # noqa: E402
import laplace_k_criterion as lk                            # noqa: E402
import em_runner                                            # noqa: E402
from em_runner_instrumented import (RetryDetected,          # noqa: E402
                                    run_em_instrumented, RUNNER_VERSION)
from data_generator_canonical import (build_full_rank_loadings,  # noqa: E402
                                      canonical_poisson_rate, GENERATOR_VERSION)

hist = a1.hist

EXPERIMENT_ID = "dsweep_k_selection_20261105"
OUT_ROOT = ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / EXPERIMENT_ID
INSTRUMENTED = _EXPERIMENTAL / "em_runner_instrumented.py"
INSTRUMENTED_BLOB = "5b3f7baa1d16bdb32686ac615cdc39c56d4018cf"

K_TRUE = 5
N = 75
D0 = hist.D_FEATURES                      # 15
D_LEVELS = (15, 30, 60)
D_NEW = (30, 60)
BLOCKS = {15: 0, 30: 1, 60: 3}
BLOCK_SEED_TAG = 900000                   # block b uses SeedSequence([data_seed, TAG + b])
REPLICATES = tuple(range(1, hist.REPLICATES_BY_K_TRUE[K_TRUE] + 1))
MAX_ITER = 8
CHECKPOINT = 8
REPLICATION_FITS = ((1, 5, 1), (1, 4, 1))           # (replicate, K, start) at d=15
QLAP_SEED_BASE = cpr.QLAP_SEED_BASE
EXPECTED_NEW_FITS = len(D_NEW) * len(REPLICATES) * len(hist.CANDIDATE_K) * len(hist.STARTS)
C2 = 1.0 + math.log(2.0 * math.pi)

PREREG = {
    "experiment_id": EXPERIMENT_ID,
    "frozen_before": "any d=30/60 fit; written by --gates and re-verified by --full",
    "conditions": {"generator": f"canonical-clean ({GENERATOR_VERSION}) + appended blocks",
                   "family_x": hist.FAMILY_X, "family_y": hist.FAMILY_Y, "n": N,
                   "K_GEN": K_TRUE, "d": list(D_LEVELS), "candidate_K": list(hist.CANDIDATE_K),
                   "replicates": list(REPLICATES), "starts": list(hist.STARTS),
                   "L": hist.L_SAMPLES, "em_iterations": MAX_ITER,
                   "numerics_mode": hist.NUMERICS_MODE,
                   "row_norm_sq_target": hist.ROW_NORM_SQ_TARGET,
                   "w0": hist.W0_TRUE, "w_rule": "w_for_matched_y_signal(W_REF=1, K_REF=3)"},
    "target": ("K* = 5 (minimal representation dimension; derivation for canonical "
               "Poisson-X with rank F = K, unconditional law; NOT K0 uniqueness). "
               "Under- and over-selection are both errors relative to K*."),
    "selection": "mean of the criterion over the 2 starts, argmin; C_Lap INCOMPLETE if any fit not OK",
    "primary": ["exact counts per d and criterion", "slack_45 = C(4) - C(5) (mean over starts)",
                "C_Q: fit gain D_K(4)-D_K(5), dP_Z, dP_theta",
                "C_Lap: fit_mode gain, d volume, dP_theta",
                "number of datasets where C_Q and C_Lap select different K"],
    "mechanism": ["F^T F (truth)", "X Fisher at truth and at (theta_hat, Z_hat)",
                  "ln|H| = ln|P_X+prior| + ln|I + P^-1 H_Y| (exact split)",
                  "2 dH_Lap(4->5) = n(1+ln 2pi) - d ln|H|", "termA, R3, residual by K",
                  "gram_rel_change / eta changes at iteration 8"],
    "accounting": "slack_Q = slack_Lap - 2 dH_Lap - d resid,  d resid = resid(5) - resid(4) (exact identity)",
    "hypotheses": {
        "H1": {"statement": "median over replicates of C_Q slack_45 increases with d",
               "support": "m15 < m30 < m60",
               "refute": "m60 <= m15",
               "otherwise": "partial (non-monotone but m60 > m15)"},
        "H2": {"statement": ("2 dH_Lap(4->5) decreases with d, stays positive at d=60 "
                             "(C_Q remains stricter than C_Lap), and accounts for the "
                             "C_Lap - C_Q slack difference"),
               "H2a_refute": "median 2dH_Lap at d=60 >= at d=15",
               "H2b_refute": "median 2dH_Lap at d=60 <= 0",
               "H2c_refute": "at some d, median |d resid| > median |2 dH_Lap|",
               "note": "positivity at d=60 is a prediction from oracle curvature, not a theorem"},
        "H3": {"statement": ("with P_theta held at its d=15 count (diagnostic selector, "
                             "not a criterion) the C_Q slack_45 increases with d"),
               "support": "fixed-penalty median slack m15 < m30 < m60",
               "refute": "fixed-penalty median slack at d=60 <= at d=15"},
        "not_predicted": "exact counts of the real (penalised) criteria; C_Lap direction",
        "scope": ("Gaussian-X thresholds (2.84, 2*pi*e) are NOT used as predictions for "
                  "this Poisson-X / Bernoulli-Y model"),
    },
    "failure_rules": ["data gate failure -> stop before any fit",
                      "replication mismatch beyond NPZ_RTOL/NPZ_ATOL -> stop",
                      "NaN restart request (RetryDetected) -> stop the run (frozen EMC rule)",
                      "Laplace not OK -> C_Lap INCOMPLETE for that dataset and d; no imputation",
                      "paired comparisons across d use only replicates complete at the compared d",
                      "no retry, no reseed, no tolerance change, no resume"],
    "replication_fits": [list(r) for r in REPLICATION_FITS],
    "replication_role": "reproducibility check only; never counted as data",
}


class Stop(RuntimeError):
    pass


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise Stop(msg)


def sha(a: np.ndarray) -> str:
    c = np.ascontiguousarray(a)
    return hashlib.sha256(c.tobytes() + str(c.shape).encode()).hexdigest()


def prereg_hash() -> str:
    return hashlib.sha256(json.dumps(PREREG, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- data
def build_data(rep: int, prov):
    """d=15 dataset (A1 gates inside make_cell) plus the appended blocks."""
    ds, train, test = a1.make_cell(K_TRUE, N, rep, prov[(K_TRUE, N, rep)])
    scale = hist.f_scale_for_row_norm(hist.ROW_NORM_SQ_TARGET, d=D0, k=K_TRUE)
    need(math.isclose(scale, float(ds.metadata["f_scale"]), rel_tol=0, abs_tol=0),
         "f_scale differs from the A1 generator")
    blocks = []
    for b in range(1, max(BLOCKS.values()) + 1):
        rng = np.random.default_rng([hist.data_seed(K_TRUE, N, rep), BLOCK_SEED_TAG + b])
        Fb = build_full_rank_loadings(rng, d=D0, k=K_TRUE, f_scale=scale)
        rate = canonical_poisson_rate(ds.Z @ Fb.T, lambda_max=hist.POISSON_LAMBDA_MAX)
        Xb = rng.poisson(rate).astype(np.float64)
        need(np.all(np.isfinite(Xb)) and np.all(Xb >= 0) and np.all(Xb == np.floor(Xb)),
             f"block {b} support rep {rep}")
        need(np.allclose(Fb.T @ Fb, scale ** 2 * np.eye(K_TRUE), atol=1e-10),
             f"block {b} F^T F rep {rep}")
        blocks.append((Fb, Xb))
    out = {}
    for d in D_LEVELS:
        nb = BLOCKS[d]
        F = np.vstack([ds.F] + [b[0] for b in blocks[:nb]])
        X = np.hstack([ds.X] + [b[1] for b in blocks[:nb]])
        out[d] = {"X": X, "F": F}
    return ds, train, test, out, scale


def gate_union_bound(F: np.ndarray) -> float:
    """Upper bound on P(some eta_il > ln lambda_max) under the UNCONDITIONAL law
    given F:  sum_{i,l} P(N(0, ||f_l||^2) > ln lambda_max)  (Z ~ N(0, I))."""
    t = math.log(hist.POISSON_LAMBDA_MAX)
    s = np.sqrt(np.sum(F ** 2, axis=1))
    return float(N * np.sum(np.exp(norm.logsf(t / s))))


def oracle_curvature(ds, X, F, train) -> dict[str, float]:
    model = em_runner.build_model(N, X.shape[1], K_TRUE, hist.L_SAMPLES,
                                  hist.FAMILY_X, hist.FAMILY_Y, train_mask=train,
                                  numerics_mode=hist.NUMERICS_MODE)
    model.initialize_params(true_params=None, seed=0)
    model.params.update({"Z": ds.Z.copy(), "F": F, "w0": float(ds.metadata["w0"]),
                         "w": float(ds.metadata["w"]), "var_z": 1.0})
    H = lk.negative_hessian(model, X, ds.Y, ds.Z)
    s, ld = np.linalg.slogdet(H)
    cur = x_curvature(F, 1.0, ds.Z)
    return {"oracle_logdetH_per_n": ld / N if s > 0 else float("nan"),
            "oracle_logdet_prior_X_per_n": cur["logdet_prior_X"] / N,
            "oracle_xfisher_median_eig": cur["xfisher_median_eig"],
            "oracle_xfisher_median_mineig": cur["xfisher_median_mineig"]}


def x_curvature(F: np.ndarray, var_z: float, Z: np.ndarray) -> dict[str, float]:
    lam = np.exp(Z @ F.T)                                   # canonical Poisson, no clip
    J = np.einsum("lk,il,lm->ikm", F, lam, F)
    ev = np.linalg.eigvalsh(J)
    P = J + np.eye(F.shape[1])[None] / var_z
    sign, ld = np.linalg.slogdet(P)
    need(bool(np.all(sign > 0)), "prior+X block not PD")
    return {"logdet_prior_X": float(np.sum(ld)),
            "xfisher_median_eig": float(np.median(ev)),
            "xfisher_median_mineig": float(np.median(ev[:, 0]))}


def data_gates(prov, emc) -> tuple[dict, list[dict[str, Any]]]:
    rows, seen_block_seeds, seen_data_seeds = [], set(), set()
    cache = {}
    for rep in REPLICATES:
        ds, train, test, dd, scale = build_data(rep, prov)
        ds2, train2, test2, dd2, _ = build_data(rep, prov)
        need(all(np.array_equal(getattr(ds, f), getattr(ds2, f)) for f in "ZFXY")
             and np.array_equal(train, train2) and np.array_equal(test, test2),
             f"d=15 regeneration not deterministic rep {rep}")
        need(all(np.array_equal(dd[d]["X"], dd2[d]["X"]) for d in D_LEVELS),
             f"appended blocks not deterministic rep {rep}")
        dseed = hist.data_seed(K_TRUE, N, rep)
        need(dseed not in seen_data_seeds, "duplicate data seed")
        seen_data_seeds.add(dseed)
        for b in range(1, 4):
            key = (dseed, BLOCK_SEED_TAG + b)
            need(key not in seen_block_seeds, "duplicate block seed")
            seen_block_seeds.add(key)
        mseeds = set()
        for k in hist.CANDIDATE_K:
            for s in hist.STARTS:
                er = emc[(K_TRUE, N, rep, k, s)]
                ms = hist.model_seed(K_TRUE, N, rep, k, s)
                need(int(er["model_seed"]) == ms and int(er["data_seed"]) == dseed
                     and er["train_mask_hash"] == hist.stable_hash(train)
                     and er["test_mask_hash"] == hist.stable_hash(test),
                     f"EMC identity rep {rep} K {k} s {s}")
                need(ms not in mseeds, "duplicate model seed")
                mseeds.add(ms)
        for d in D_LEVELS:
            X, F = dd[d]["X"], dd[d]["F"]
            need(X.shape == (N, d) and F.shape == (d, K_TRUE), f"shape d={d}")
            need(np.array_equal(X[:, :D0], ds.X) and np.array_equal(F[:D0], ds.F),
                 f"original columns changed d={d} rep {rep}")
            need(np.allclose(F.T @ F, (d / D0) * scale ** 2 * np.eye(K_TRUE), atol=1e-10),
                 f"F^T F d={d}")
            need(np.all(np.isfinite(X)), "non-finite X")
            need(float(np.max(np.exp(ds.Z @ F.T))) <= hist.POISSON_LAMBDA_MAX, "rate gate")
            row = {"d": d, "replicate": rep, "data_seed": dseed,
                   "block_seeds": json.dumps([[dseed, BLOCK_SEED_TAG + b] for b in range(1, BLOCKS[d] + 1)]),
                   "f_scale": scale, "FtF_diag": float(np.mean(np.diag(F.T @ F))),
                   "mean_row_norm_sq": float(np.mean(np.sum(F ** 2, 1))),
                   "max_row_norm_sq": float(np.max(np.sum(F ** 2, 1))),
                   "max_rate_truth": float(np.max(np.exp(ds.Z @ F.T))),
                   "x_mean": float(X.mean()), "x_max": float(X.max()),
                   "gate_fail_prob_union_bound": gate_union_bound(F),
                   "hash_X15": sha(X[:, :D0]), "hash_X": sha(X), "hash_Y": sha(ds.Y),
                   "hash_Z": sha(ds.Z), "hash_train": sha(train), "hash_test": sha(test),
                   **oracle_curvature(ds, X, F, train)}
            rows.append(row)
        for col in ("hash_X15", "hash_Y", "hash_Z", "hash_train", "hash_test"):
            need(len({r[col] for r in rows if r["replicate"] == rep}) == 1,
                 f"{col} differs across d rep {rep}")
        cache[rep] = (ds, train, test, dd)
    return cache, rows


# --------------------------------------------------------------------- evaluation
def evaluate(model, X, Y, Zs, Z, *, d: int, fit_id: int) -> dict[str, Any]:
    k = Zs.shape[1]
    num_params = k * d - k * (k - 1) // 2
    cq = panel.cq_terms(model, X, Y, Zs, num_params=num_params, n=N)
    need(abs(cq["C_Q_recomposed"] - cq["C_Q"]) <= a1.RECOMPOSE_TOL, "C_Q closure")
    lap = panel.laplace_terms(model, X, Y, Z, num_params=num_params, N=N)
    z_hat = lap.pop("Z_hat")
    t = panel.theta_of(model)
    row = {"num_params": num_params, "C_Q": cq["C_Q"], "D_K": cq["D_K"],
           "P_Z": cq["P_Z"], "P_theta": cq["P_theta"], "q_strict": cq["q_strict"],
           "w0_hat": t["w0"], "w_hat": t["w"], "var_z_hat": t["var_z"], **lap}
    if lap["lap_status"] == "OK":
        need(abs(lap["C_Lap_recomposed"] - lap["C_Lap"]) <= a1.RECOMPOSE_TOL, "C_Lap closure")
        row.update(cpr.qlap_diag(model, X, Y, z_hat, QLAP_SEED_BASE + 1000 * fit_id + CHECKPOINT,
                                 cq["q_strict"], lap["lap_log_observed"]))
        cur = x_curvature(t["F"], t["var_z"], z_hat)
        row.update({"logdet_prior_X_hat": cur["logdet_prior_X"],
                    "logdet_Y_increment_hat": lap["lap_logdet_H"] - cur["logdet_prior_X"],
                    "xfisher_median_eig_hat": cur["xfisher_median_eig"],
                    "xfisher_median_mineig_hat": cur["xfisher_median_mineig"],
                    "residual": cq["C_Q"] - lap["C_Lap"]
                    - (N * k * C2 - lap["lap_logdet_H"])})
    return row


DIAG_COLS = ("gram_rel_change", "f_raw_change", "w_change", "w0_change",
             "eta_x_rel_change", "eta_y_rel_change", "max_gram_rel_change_last4",
             "max_eta_x_rel_change_last4", "max_eta_y_rel_change_last4")


def eval_d15(cache, emc, npz, exp1) -> list[dict[str, Any]]:
    """Re-evaluate the reused d=15 states (no MCEM); gate against experiment 1."""
    cp_rows, _ = cpr.git_blob(cpr.EMC_CP_RESULTS)
    cpd = {(int(r["fit_index"]), int(r["checkpoint_iteration"])): r for r in cp_rows}
    rows = []
    for rep in REPLICATES:
        ds, train, _test, dd = cache[rep]
        for k in hist.CANDIDATE_K:
            for s in hist.STARTS:
                fi = int(emc[(K_TRUE, N, rep, k, s)]["fit_index"])
                key, model, Zs, Z = cpr.state(npz, fi, CHECKPOINT, N, k, train)
                r = evaluate(model, dd[15]["X"], ds.Y, Zs, Z, d=15, fit_id=fi)
                e = exp1[(rep, k, s)]
                need(r["C_Q"] == float(e["C_Q"]) and r["lap_status"] == e["lap_status"],
                     f"d=15 C_Q/status differs from experiment 1 {key}")
                if r["lap_status"] == "OK":
                    for col in ("C_Lap", "Q_Lap", "termA", "R3"):
                        need(math.isclose(r[col], float(e[col]), rel_tol=1e-12, abs_tol=1e-9),
                             f"d=15 {col} differs from experiment 1 {key}")
                c = cpd[(fi, CHECKPOINT)]
                rows.append({"d": 15, "replicate": rep, "k_est": k, "start": s,
                             "checkpoint": CHECKPOINT, "source": f"emc_state:{key}",
                             "fit_id": fi, "model_seed": int(emc[(K_TRUE, N, rep, k, s)]["model_seed"]),
                             "fit_status": "REUSED", "n": N,
                             **{col: float(c[col]) for col in DIAG_COLS}, **r})
    return rows


def fit_one(X, Y, train, k, seed):
    holder = {}

    def cb(rec):
        holder["rec"] = rec
    res = run_em_instrumented(X, Y, family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y,
                              k=k, L=hist.L_SAMPLES, max_iter=MAX_ITER,
                              checkpoints=(CHECKPOINT,), seed=seed, train_mask=train,
                              numerics_mode=hist.NUMERICS_MODE, stability_window=4,
                              on_nan="fail", on_checkpoint=cb)
    rec = holder["rec"]
    need(res["retry_count"] == 0 and not res["nan_detected"], "unexpected retry")
    model = rec["model"]
    Zs = np.array(rec["Z_samples"]); Z = np.array(rec["Z_est"])
    model.params.update({"Z": Z.copy(), "var_z": float(rec["var_z"])})
    return res, rec, model, Zs, Z


def state_arrays(rec, Zs, Z):
    return {"Zsamples": Zs, "Z": Z, "F": np.array(rec["F"]), "sigma": np.array(rec["sigma"]),
            "scalars": np.array([rec["w0"], rec["w"], rec["var_z"]], dtype=np.float64)}


def load_exp1() -> tuple[dict, str]:
    path = OUT_ROOT.parent / "clap_checkpoint_reeval_20261105" / "full" / "fit_criteria.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if int(r["n"]) == N and int(r["checkpoint"]) == CHECKPOINT]
    out = {(int(r["replicate"]), int(r["k_est"]), int(r["start"])): r for r in rows}
    need(len(out) == len(REPLICATES) * 14, f"experiment 1 rows {len(out)}")
    return out, a1.sha256_file(path)


def common_inputs():
    need(a1.working_tree_clean(), "working tree must be clean")
    need(a1._git("rev-parse", f"HEAD:expfam/src/experimental/em_runner_instrumented.py")
         == INSTRUMENTED_BLOB, "instrumented runner is not the dad67b1 blob")
    need(a1.sha256_file(a1.EMC_NPZ) == a1.EMC_NPZ_SHA256, "EMC npz sha256")
    prov = a1.load_hist_provenance()
    emc, emc_digest = a1.load_emc_identity()
    exp1, exp1_sha = load_exp1()
    meta = {"experiment_id": EXPERIMENT_ID, "git_head": a1.git_head(),
            "branch": a1._git("rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_clean": True, "environment": a1.environment(),
            "prereg_sha256": prereg_hash(), "instrumented_runner": RUNNER_VERSION,
            "instrumented_blob": INSTRUMENTED_BLOB,
            "inputs": {"emc_npz": str(a1.EMC_NPZ), "emc_npz_sha256": a1.EMC_NPZ_SHA256,
                       "emc_fit_results_sha256": emc_digest,
                       "exp1_fit_criteria_sha256": exp1_sha}}
    return prov, emc, exp1, meta


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """CSV with the union of keys (rows differ: reused vs new, OK vs not OK)."""
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({c: _cell(v) for c, v in r.items()})


def _cell(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, np.integer):
        return int(v)
    return v


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    need(not path.exists(), f"state file exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return a1.sha256_file(path)


# --------------------------------------------------------------------------- stages
def run_gates() -> int:
    out = OUT_ROOT / "gates"
    need(not out.exists(), f"output exists: {out}")
    prov, emc, exp1, meta = common_inputs()
    out.mkdir(parents=True)
    a1.write_json(out / "prereg.json", {**PREREG, "sha256": prereg_hash()})
    gates: dict[str, Any] = {}
    try:
        cache, drows = data_gates(prov, emc)
        write_rows(out / "data_provenance.csv", drows)
        gates.update({"G1_d15_data_mask_seed_emc_identity": "PASS",
                      "G3_original_columns_Z_Y_mask_unchanged": "PASS",
                      "G4_added_columns_match_plan": "PASS",
                      "G5_seeds_unique_rate_gate_finite": "PASS",
                      "max_gate_fail_prob_union_bound": max(r["gate_fail_prob_union_bound"] for r in drows)})
        npz = np.load(a1.EMC_NPZ)
        d15 = eval_d15(cache, emc, npz, exp1)
        gates["G1b_d15_reevaluation_equals_experiment1"] = f"PASS ({len(d15)} states)"
        rep_rows, arrays = [], {}
        for rep, k, s in REPLICATION_FITS:
            ds, train, _t, dd = cache[rep]
            seed = hist.model_seed(K_TRUE, N, rep, k, s)
            res, rec, model, Zs, Z = fit_one(dd[15]["X"], ds.Y, train, k, seed)
            fi = int(emc[(K_TRUE, N, rep, k, s)]["fit_index"])
            key = f"{fi:04d}_it{CHECKPOINT:03d}"
            ref = {"Zsamples": npz[f"{key}_Zsamples"], "F": npz[f"{key}_F"],
                   "sigma": npz[f"{key}_sigma"], "scalars": npz[f"{key}_scalars"]}
            mine = state_arrays(rec, Zs, Z)
            diffs = {name: float(np.max(np.abs(mine[name] - np.asarray(ref[name], float))))
                     for name in ref}
            ok = all(np.allclose(mine[name], np.asarray(ref[name], float),
                                 rtol=a1.NPZ_RTOL, atol=a1.NPZ_ATOL) for name in ref)
            e = exp1[(rep, k, s)]
            ok = ok and math.isclose(rec["bic"], float(e["C_Q"]), rel_tol=a1.NPZ_RTOL,
                                     abs_tol=a1.NPZ_ATOL)
            rep_rows.append({"replicate": rep, "k_est": k, "start": s, "model_seed": seed,
                             "emc_state_key": key, "bic_new": rec["bic"],
                             "C_Q_reference": float(e["C_Q"]), "match": ok,
                             **{f"maxabs_{n_}": v for n_, v in diffs.items()},
                             "runtime_s": res["runtime_s"]})
            for name, arr in mine.items():
                arrays[f"rep{rep}_k{k}_s{s}_{name}"] = arr
        write_rows(out / "replication.csv", rep_rows)
        state_sha = write_npz(STATE_ROOT / "replication_states.npz", arrays)
        need(all(r["match"] for r in rep_rows), "replication fits do not match the archived state")
        gates["G2_replication_fits"] = "PASS (2 fits; reproducibility only, not counted)"
        gates["G6_implementation_frozen"] = f"PASS (clean tree {meta['git_head']}, prereg {prereg_hash()})"
        a1.write_json(out / "gates.json", {**meta, "status": "PASSED", "gates": gates,
                                           "replication_states": str(STATE_ROOT / "replication_states.npz"),
                                           "replication_states_sha256": state_sha})
        print(json.dumps(gates, indent=1))
        return 0
    except Exception as exc:
        a1.write_json(out / "gates.json", {**meta, "status": "FAILED", "gates": gates,
                                           "error_type": type(exc).__name__, "error": str(exc)})
        print(f"GATE FAILED: {type(exc).__name__}: {exc}")
        return 2


def run_full() -> int:
    out = OUT_ROOT / "full"
    need(not out.exists(), f"output exists: {out}")
    g = json.loads((OUT_ROOT / "gates" / "gates.json").read_text(encoding="utf-8"))
    need(g["status"] == "PASSED" and g["prereg_sha256"] == prereg_hash(), "gates not passed / prereg changed")
    prov, emc, exp1, meta = common_inputs()
    out.mkdir(parents=True)
    started = time.time()
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    try:
        cache, drows = data_gates(prov, emc)
        write_rows(out / "data_provenance.csv", drows)
        rows += eval_d15(cache, emc, np.load(a1.EMC_NPZ), exp1)
        print(f"d=15 re-evaluated ({time.time() - started:.0f}s)", flush=True)
        local = 0
        for d in D_NEW:
            for rep in REPLICATES:
                ds, train, _t, dd = cache[rep]
                for k in hist.CANDIDATE_K:
                    for s in hist.STARTS:
                        local += 1
                        fit_id = d * 1000 + local
                        seed = hist.model_seed(K_TRUE, N, rep, k, s)
                        res, rec, model, Zs, Z = fit_one(dd[d]["X"], ds.Y, train, k, seed)
                        need(math.isfinite(rec["bic"]), "non-finite bic")
                        r = evaluate(model, dd[d]["X"], ds.Y, Zs, Z, d=d, fit_id=fit_id)
                        need(math.isclose(r["C_Q"], rec["bic"], rel_tol=1e-9, abs_tol=1e-6),
                             "panel C_Q differs from the runner's BIC")
                        for name, arr in state_arrays(rec, Zs, Z).items():
                            arrays[f"d{d}_rep{rep}_k{k}_s{s}_{name}"] = arr
                        rows.append({"d": d, "replicate": rep, "k_est": k, "start": s,
                                     "checkpoint": CHECKPOINT, "source": "new_fit",
                                     "fit_id": fit_id, "model_seed": seed, "fit_status": "FIT",
                                     "n": N, "em_runtime_s": res["runtime_s"],
                                     **{col: float(rec[col]) for col in DIAG_COLS}, **r})
                print(f"d={d} rep={rep} done ({time.time() - started:.0f}s)", flush=True)
    except (Exception, RetryDetected) as exc:
        write_rows(out / "fit_criteria_partial.csv", rows)
        a1.write_json(out / "failure.json", {**meta, "status": "STOPPED",
                                             "error_type": type(exc).__name__,
                                             "error": str(exc), "completed_rows": len(rows)})
        print(f"STOPPED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    new = sum(r["source"] == "new_fit" for r in rows)
    need(new == EXPECTED_NEW_FITS, f"new fits {new}")
    state_sha = write_npz(STATE_ROOT / "fit_states.npz", arrays)
    sel = []
    for d in D_LEVELS:
        for r in cpr.select([x for x in rows if x["d"] == d]):
            sel.append({"d": d, **r})
    write_rows(out / "fit_criteria.csv", rows)
    write_rows(out / "selection.csv", sel)
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[f"d{r['d']}:{r['lap_status']}"] = statuses.get(f"d{r['d']}:{r['lap_status']}", 0) + 1
    a1.write_json(out / "runinfo.json", {
        **meta, "status": "COMPLETED", "rows": len(rows), "new_fits": new,
        "reused_d15_states": len(rows) - new, "replication_fits_counted": 0,
        "wall_clock_seconds": round(time.time() - started, 1),
        "laplace_status_counts": statuses,
        "fit_states": str(STATE_ROOT / "fit_states.npz"), "fit_states_sha256": state_sha})
    print(f"completed: {statuses}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gates", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    return run_gates() if a.gates else run_full()


if __name__ == "__main__":
    sys.exit(main())
