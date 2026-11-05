"""Controlled signal-strength experiment for K selection (lineage E, prototype).

research/k-selection-causal-analysis-20261106.  Question: under the SAME canonical
model, estimator and families (Poisson-X / Bernoulli-Y, K_GEN=5, d=15), how do the
4->5 selection slack, its X / Y / Z decomposition and the C_Q / C_Lap selections
change when only the loading energy (F) and / or the relational strength (w) change?

Design (frozen in PREREG BEFORE any production fit; see --gates)
  A  baseline   : the #139 A1 dataset itself, bit-identical (Z, F, X, Y, masks); its
                  saved EMC checkpoint-8 states are RE-EVALUATED, not refit.
  B  F-strong   : F_B = c_F F_A (same directions), c_F = sqrt(0.9/0.5), i.e. average
                  ||f_l||^2 0.5 -> 0.9 (the historical per-row value 1-uniq).
                  X_B is an EXACT Poisson coupling of X_A:  X_A + Pois(lam_B - lam_A)
                  where lam_B >= lam_A, Binomial(X_A, lam_B/lam_A) where lam_B < lam_A.
                  Given Z, X_B ~ Pois(lam_B) independently over cells, so (Z, X_B, Y_A)
                  is an exact canonical draw.  Z, Y, masks bit-identical to A.
  C  w-strong   : w_C = c_w w_A, c_w = sqrt(1.8) (same energy ratio as F), w0_C solves
                  E sigma(w0_C + w_C z.z') = E sigma(w0_A + w_A z.z') (population
                  density, Gauss quadrature; NOT the sample density).  Y_C is an EXACT
                  Bernoulli coupling of Y_A on the upper triangle.  Z, F, X, masks
                  bit-identical to A.
  D  both       : X_B with Y_C (shares X with B and Y with C).
  Model seeds: the A1 formula (paired across conditions -> identical initialisation).
  Fits: instrumented runner (dad67b1 blob), L=5, 8 iterations, on_nan='fail'.

Stages
  --gates          data gates, calibration, A re-evaluation gate, 1 replication fit,
                   4 pilot fits (pilot fits are NEVER counted as data)
  --full --n 75    B, C, D at n=75   (336 fits)   [Priority 1]
  --full --n 150   B at n=150        (112 fits)   [Priority 2]
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
from scipy.optimize import brentq
from scipy.special import expit, roots_genlaguerre, roots_hermite
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
from eval_utils import calc_Q_dual_strict_exp               # noqa: E402

hist = a1.hist

EXPERIMENT_ID = "historical_canonical_causal_20261106"
OUT_ROOT = ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID / "signal_strength"
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / EXPERIMENT_ID
INSTRUMENTED_BLOB = "5b3f7baa1d16bdb32686ac615cdc39c56d4018cf"

K_GEN = 5
D = hist.D_FEATURES                                   # 15
ROW_NORM_A = hist.ROW_NORM_SQ_TARGET                  # 0.5
ROW_NORM_B = 0.9
C_F = math.sqrt(ROW_NORM_B / ROW_NORM_A)
C_W = math.sqrt(1.8)
W_A = hist.w_for_matched_y_signal(hist.W_REF, k=K_GEN, k_ref=hist.K_REF)
W0_A = hist.W0_TRUE
CONDITIONS = ("A", "B", "C", "D")
NEW_BY_N = {75: ("B", "C", "D"), 150: ("B",)}
REPLICATES = tuple(range(1, 9))
MAX_ITER = 8
CHECKPOINT = 8
TAG_X = 610000                     # SeedSequence([data_seed, TAG_X]) -> X coupling stream
TAG_Y = 620000                     # SeedSequence([data_seed, TAG_Y]) -> Y coupling stream
QLAP_SEED_BASE = 20261106
REPLICATION_FIT = (75, 1, 5, 1)    # (n, replicate, K, start), condition A
PILOT_FITS = ((75, "B", 1, 5, 1), (75, "C", 1, 5, 1), (75, "D", 1, 5, 1), (150, "B", 1, 5, 1))
FIT_CAP = 480
C2 = 1.0 + math.log(2.0 * math.pi)


def expected_new_fits(n: int) -> int:
    return len(NEW_BY_N[n]) * len(REPLICATES) * len(hist.CANDIDATE_K) * len(hist.STARTS)


PREREG = {
    "experiment_id": EXPERIMENT_ID, "part": "signal_strength",
    "frozen_before": "any production fit; written by --gates and re-verified by --full",
    "conditions": {
        "family_x": hist.FAMILY_X, "family_y": hist.FAMILY_Y, "K_GEN": K_GEN, "d": D,
        "candidate_K": list(hist.CANDIDATE_K), "starts": list(hist.STARTS),
        "replicates": list(REPLICATES), "L": hist.L_SAMPLES, "em_iterations": MAX_ITER,
        "numerics_mode": hist.NUMERICS_MODE, "test_ratio": hist.TEST_RATIO,
        "A": {"row_norm_sq_mean": ROW_NORM_A, "w": W_A, "w0": W0_A, "data": "#139 A1, bit-identical; states reused"},
        "B": {"row_norm_sq_mean": ROW_NORM_B, "c_F": C_F, "w": W_A, "w0": W0_A, "X": "exact Poisson coupling of X_A"},
        "C": {"row_norm_sq_mean": ROW_NORM_A, "w": W_A * C_W, "c_w": C_W,
              "w0": "population-density calibrated (Gauss quadrature)", "Y": "exact Bernoulli coupling of Y_A"},
        "D": {"X": "X_B", "Y": "Y_C"},
        "new_fits": {"n=75": "B, C, D (336)", "n=150": "B (112)"},
        "resource_rule": ("fixed BEFORE results: cap 480 new MCEM fits (448 production + 1 replication "
                          "+ 4 pilot = 453); Priority 1 n=75 B/C/D, Priority 2 n=150 B; Priority 3 "
                          "(K_GEN=3) is NOT refit: the A1 K_TRUE=3 results are used descriptively; "
                          "n=150 C/D are NOT run (would exceed the cap)"),
    },
    "target": "K* = 5 (canonical Poisson-X, rank F = 5; unconditional law). NOT K0 uniqueness.",
    "selection": "mean of the criterion over the 2 starts, argmin, ties -> smallest K; C_Lap INCOMPLETE if any fit of the dataset is not OK; a FAILED fit makes the dataset INCOMPLETE for both criteria",
    "slack": "slack_45 = C(4) - C(5) on start-means (positive favours K=5)",
    "hypotheses": {
        "P1_primary": {"statement": "n=75: F-strong increases the C_Q 4->5 slack (paired B-A)",
                       "support": "median paired difference > 0 AND >= 6/8 replicates positive",
                       "refute": "median paired difference <= 0",
                       "otherwise": "inconclusive"},
        "P1_mechanism": "the B-A change of the C_Q slack is carried by fit_gain_x (prior cost is identical by construction; reported, not tested)",
        "S1_w": {"statement": "n=75: w-strong increases fit_gain_y(4->5) and the C_Q slack (paired C-A)",
                 "support": "both medians > 0 AND >= 6/8 positive for each",
                 "refute": "median paired slack difference <= 0", "otherwise": "inconclusive"},
        "S2_interaction": {"statement": "interaction I = slack_D - slack_B - slack_C + slack_A (C_Q, n=75)",
                           "rule": "descriptive, two-sided: report median, sign counts, and whether D-A exceeds both B-A and C-A; no directional prediction"},
        "S3_criteria": {"statement": "at the same fits, slack_Lap - slack_Q = 2 dH_Lap + d resid (identity); 2 dH_Lap(4->5) > 0 in every fit pair and its median is lower in B, C, D than in A",
                        "refute_positivity": "any 4->5 pair (start mean) with 2 dH_Lap <= 0",
                        "refute_decrease": "median 2 dH_Lap of a strong condition >= that of A"},
        "P2_n150": {"statement": "n=150: paired B-A C_Q slack difference > 0",
                    "support": "median > 0 AND >= 6/8 positive", "refute": "median <= 0"},
        "P3_k3": "descriptive only: per-direction energy in D at K=5 (X 2.7, Y w^2 1.08) vs A1 K_TRUE=3 baseline (X 2.5, Y w^2 1.0); different datasets, not paired",
        "exploratory": ["start agreement", "Poisson max rate", "Bernoulli saturation (mean p(1-p))",
                        "gram_rel_change", "Laplace status", "MC standard error of the slack"],
        "not_claimed": "exact counts are reported but are NOT the success criterion; no causal claim beyond the paired intervention on this design",
    },
    "failure_rules": ["data / calibration / coupling gate failure -> stop before any production fit",
                      "replication mismatch beyond NPZ_RTOL/NPZ_ATOL -> stop",
                      "RetryDetected or exception in a fit -> record FAILED, dataset INCOMPLETE, continue (no retry, no reseed)",
                      "Laplace not OK -> C_Lap INCOMPLETE for that dataset; no imputation",
                      "no tolerance change, no seed rescue, no dataset replacement, no resume after results"],
    "pilot_role": "technical check only; never counted as data; does not change the plan",
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


# ---------------------------------------------------------------- calibration
def population_density(w0: float, w: float, k: int = K_GEN, m: int = 160) -> float:
    """E sigma(w0 + w z.z'), z, z' iid N(0, I_k).

    Given z', z.z' ~ N(0, r) with r = ||z'||^2 ~ chi^2_k.  Outer: generalised
    Gauss-Laguerre in t = r/2 (weight t^(k/2-1) e^-t); inner: Gauss-Hermite.
    """
    t, wt = roots_genlaguerre(m, k / 2.0 - 1.0)
    x, wx = roots_hermite(m)
    g = math.sqrt(2.0) * x                        # N(0,1) nodes
    r = 2.0 * t
    inner = expit(w0 + w * np.sqrt(r)[:, None] * g[None, :]) @ (wx / math.sqrt(math.pi))
    return float(np.sum(wt * inner) / math.gamma(k / 2.0))


def calibrate_w0(w_new: float) -> float:
    target = population_density(W0_A, W_A)
    return float(brentq(lambda b: population_density(b, w_new) - target, -10.0, 5.0, xtol=1e-13))


W_C = W_A * C_W


def w0_c() -> float:
    return calibrate_w0(W_C)


# ---------------------------------------------------------------- couplings
def couple_poisson(X_A: np.ndarray, lam_A: np.ndarray, lam_B: np.ndarray, rng) -> np.ndarray:
    up = lam_B >= lam_A
    add = rng.poisson(np.where(up, lam_B - lam_A, 0.0))
    keep = rng.binomial(X_A.astype(np.int64), np.where(up, 1.0, lam_B / lam_A))
    return np.where(up, X_A + add, keep).astype(np.float64)


def couple_bernoulli_upper(Y_A: np.ndarray, p_A: np.ndarray, p_C: np.ndarray, rng) -> np.ndarray:
    n = Y_A.shape[0]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    ya, pa, pc = Y_A[upper], p_A[upper], p_C[upper]
    u = rng.random(ya.shape)
    up = pc >= pa
    on = np.where(up, (ya == 1) | (u < (pc - pa) / (1.0 - pa)), (ya == 1) & (u < pc / pa))
    Y = np.zeros((n, n))
    Y[upper] = on.astype(np.float64)
    Y = Y + Y.T
    return Y


# ---------------------------------------------------------------- data
def build_all(n: int, rep: int, prov, w0C: float) -> dict[str, Any]:
    ds, train, test = a1.make_cell(K_GEN, n, rep, prov[(K_GEN, n, rep)])
    scale = hist.f_scale_for_row_norm(ROW_NORM_A, d=D, k=K_GEN)
    need(scale == float(ds.metadata["f_scale"]), "f_scale differs from A1")
    need(math.isclose(float(ds.metadata["w"]), W_A, rel_tol=0, abs_tol=0), "w differs from A1")
    dseed = hist.data_seed(K_GEN, n, rep)
    F_A = ds.F
    F_B = C_F * F_A
    lam_A = np.exp(ds.Z @ F_A.T)
    lam_B = np.exp(ds.Z @ F_B.T)
    need(float(lam_B.max()) <= hist.POISSON_LAMBDA_MAX, "rate gate (B)")
    X_B = couple_poisson(ds.X, lam_A, lam_B, np.random.default_rng([dseed, TAG_X]))
    G = ds.Z @ ds.Z.T
    p_A = expit(W0_A + W_A * G)
    p_C = expit(w0C + W_C * G)
    Y_C = couple_bernoulli_upper(ds.Y, p_A, p_C, np.random.default_rng([dseed, TAG_Y]))
    cond = {"A": {"X": ds.X, "Y": ds.Y, "F": F_A, "w": W_A, "w0": W0_A},
            "B": {"X": X_B, "Y": ds.Y, "F": F_B, "w": W_A, "w0": W0_A},
            "C": {"X": ds.X, "Y": Y_C, "F": F_A, "w": W_C, "w0": w0C},
            "D": {"X": X_B, "Y": Y_C, "F": F_B, "w": W_C, "w0": w0C}}
    return {"ds": ds, "train": train, "test": test, "cond": cond, "dseed": dseed}


def union_bound(n: int, F: np.ndarray) -> float:
    t = math.log(hist.POISSON_LAMBDA_MAX)
    s = np.sqrt(np.sum(F ** 2, axis=1))
    return float(n * np.sum(np.exp(norm.logsf(t / s))))


def provenance_rows(n: int, rep: int, bundle) -> list[dict[str, Any]]:
    ds, train = bundle["ds"], bundle["train"]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    tr = np.triu(train, 1)
    rows = []
    for c in CONDITIONS:
        v = bundle["cond"][c]
        X, Y, F = v["X"], v["Y"], v["F"]
        need(np.all(np.isfinite(X)) and np.all(X >= 0) and np.all(X == np.floor(X)), f"X support {c}")
        need(set(np.unique(Y[upper])) <= {0.0, 1.0} and np.array_equal(Y, Y.T)
             and np.all(np.diag(Y) == 0), f"Y support {c}")
        need(int(np.linalg.matrix_rank(F)) == K_GEN, f"rank F {c}")
        p = expit(v["w0"] + v["w"] * (ds.Z @ ds.Z.T))
        lam = np.exp(ds.Z @ F.T)
        sv = np.linalg.svd(F, compute_uv=False)
        rows.append({
            "condition": c, "n": n, "d": D, "k_gen": K_GEN, "replicate": rep,
            "data_seed": bundle["dseed"], "x_coupling_seed": json.dumps([bundle["dseed"], TAG_X]) if c in "BD" else "",
            "y_coupling_seed": json.dumps([bundle["dseed"], TAG_Y]) if c in "CD" else "",
            "f_scale_factor": C_F if c in "BD" else 1.0, "w": v["w"], "w0": v["w0"],
            "mean_row_norm_sq": float(np.mean(np.sum(F ** 2, 1))),
            "max_row_norm_sq": float(np.max(np.sum(F ** 2, 1))),
            "F_singular_values": json.dumps([float(s) for s in sv]),
            "FtF_per_direction": float(np.mean(sv ** 2)), "F_rank": int(np.linalg.matrix_rank(F)),
            "w2_times_K": v["w"] ** 2 * K_GEN, "w2_per_direction": v["w"] ** 2,
            "max_rate_truth": float(lam.max()), "x_mean": float(X.mean()), "x_max": float(X.max()),
            "expected_x_mean_truth": float(lam.mean()),
            "gate_fail_prob_union_bound": union_bound(n, F),
            "y_density_upper": float(Y[upper].mean()), "y_density_train": float(Y[tr].mean()),
            "p_mean_truth_upper": float(p[upper].mean()),
            "population_density": population_density(v["w0"], v["w"]),
            "bernoulli_fisher_mean_p1mp": float(np.mean(p[upper] * (1 - p[upper]))),
            "frac_p_lt_0.05": float(np.mean(p[upper] < 0.05)), "frac_p_gt_0.95": float(np.mean(p[upper] > 0.95)),
            "hash_Z": sha(ds.Z), "hash_X": sha(X), "hash_Y": sha(Y), "hash_F": sha(F),
            "hash_train": sha(train), "hash_test": sha(bundle["test"]),
            "x_identical_to_A": bool(np.array_equal(X, ds.X)), "y_identical_to_A": bool(np.array_equal(Y, ds.Y)),
            "generator": "A1 canonical-clean + exact coupling", "dataset_replaced": False, "seed_rescue": False})
    return rows


# ---------------------------------------------------------------- evaluation
def evaluate(model, X, Y, Zs, Z, *, n: int, fit_id: int) -> dict[str, Any]:
    k = Zs.shape[1]
    num_params = k * D - k * (k - 1) // 2
    cq = panel.cq_terms(model, X, Y, Zs, num_params=num_params, n=n)
    need(abs(cq["C_Q_recomposed"] - cq["C_Q"]) <= a1.RECOMPOSE_TOL, "C_Q closure")
    t = panel.theta_of(model)
    L = Zs.shape[2]
    per = []
    for l in range(L):
        z = Zs[:, :, l:l + 1]
        tot = float(calc_Q_dual_strict_exp(X, Y, z, t["F"], t["sigma"], t["var_z"], t["w0"], t["w"], model))
        qz = panel.log_prior(Zs[:, :, l], t["var_z"])
        qy = float(model.calc_log_likelihood_Y(Y, z, t["w0"], t["w"]))
        per.append((tot - qz - qy, qy, qz, tot))
    per = np.array(per)
    mcse = per.std(axis=0, ddof=1) / math.sqrt(L)
    row = {"num_params": num_params, "C_Q": cq["C_Q"], "q_strict": cq["q_strict"],
           "Q_X": cq["Q_X"], "Q_Y": cq["Q_Y"], "Q_Z": cq["Q_Z"], "D_K": cq["D_K"],
           "P_Z": cq["P_Z"], "P_theta": cq["P_theta"],
           "P_Z_formula": n * k * (math.log(2 * math.pi * t["var_z"]) + 1.0 / 1.0) if t["var_z"] == 1.0 else "",
           "mcse_Q_X": mcse[0], "mcse_Q_Y": mcse[1], "mcse_Q_Z": mcse[2], "mcse_Q_total": mcse[3],
           "w0_hat": t["w0"], "w_hat": t["w"], "var_z_hat": t["var_z"],
           "Fhat_singular_values": json.dumps([float(s) for s in np.linalg.svd(t["F"], compute_uv=False)])}
    lap = panel.laplace_terms(model, X, Y, Z, num_params=num_params, N=n)
    z_hat = lap.pop("Z_hat")
    row.update(lap)
    if lap["lap_status"] == "OK":
        need(abs(lap["C_Lap_recomposed"] - lap["C_Lap"]) <= a1.RECOMPOSE_TOL, "C_Lap closure")
        ly = float(model.calc_log_likelihood_Y(Y, z_hat[:, :, None], t["w0"], t["w"]))
        row["lap_ll_Y_mode"] = ly
        row["lap_ll_X_mode"] = -0.5 * lap["lap_fit_mode"] - ly
        row.update(cpr.qlap_diag(model, X, Y, z_hat, QLAP_SEED_BASE + 1000 * fit_id + CHECKPOINT,
                                 cq["q_strict"], lap["lap_log_observed"]))
        lam = np.exp(z_hat @ t["F"].T)
        J = np.einsum("lk,il,lm->ikm", t["F"], lam, t["F"])
        P = J + np.eye(k)[None] / t["var_z"]
        s, ld = np.linalg.slogdet(P)
        need(bool(np.all(s > 0)), "prior+X block not PD")
        row["logdet_prior_X_hat"] = float(np.sum(ld))
        row["logdet_Y_increment_hat"] = lap["lap_logdet_H"] - float(np.sum(ld))
        row["H_Lap_2"] = n * k * C2 - lap["lap_logdet_H"]
        row["residual"] = cq["C_Q"] - lap["C_Lap"] - row["H_Lap_2"]
        row["identity_error"] = row["residual"] - (row["termA"] + 2 * row["R3"])
    return row


DIAG_COLS = ("gram_rel_change", "f_raw_change", "w_change", "w0_change",
             "eta_x_rel_change", "eta_y_rel_change", "max_gram_rel_change_last4",
             "max_eta_x_rel_change_last4", "max_eta_y_rel_change_last4")


def fit_one(X, Y, train, k, seed):
    holder = {"q": {}}

    def cb(rec):
        holder["q"][int(rec["iteration"])] = (float(rec["q_strict"]), float(rec["bic"]))
        if int(rec["iteration"]) == CHECKPOINT:
            holder["rec"] = rec
    res = run_em_instrumented(X, Y, family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y,
                              k=k, L=hist.L_SAMPLES, max_iter=MAX_ITER,
                              checkpoints=tuple(range(1, MAX_ITER + 1)), seed=seed, train_mask=train,
                              numerics_mode=hist.NUMERICS_MODE, stability_window=4,
                              on_nan="fail", on_checkpoint=cb)
    rec = holder["rec"]
    need(res["retry_count"] == 0 and not res["nan_detected"], "unexpected retry")
    model = rec["model"]
    Zs = np.array(rec["Z_samples"]); Z = np.array(rec["Z_est"])
    model.params.update({"Z": Z.copy(), "var_z": float(rec["var_z"])})
    return res, rec, model, Zs, Z, holder["q"]


def state_arrays(rec, Zs, Z):
    return {"Zsamples": Zs, "Z": Z, "F": np.array(rec["F"]), "sigma": np.array(rec["sigma"]),
            "scalars": np.array([rec["w0"], rec["w"], rec["var_z"]], dtype=np.float64)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    need(not path.exists(), f"refuse to overwrite {path}")
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({c: (repr(float(v)) if isinstance(v, (float, np.floating)) else
                            bool(v) if isinstance(v, np.bool_) else
                            int(v) if isinstance(v, np.integer) else v) for c, v in r.items()})


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    need(not path.exists(), f"state file exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return a1.sha256_file(path)


def common_inputs():
    need(a1.working_tree_clean(), "working tree must be clean")
    need(a1._git("rev-parse", "HEAD:expfam/src/experimental/em_runner_instrumented.py")
         == INSTRUMENTED_BLOB, "instrumented runner is not the dad67b1 blob")
    need(a1.sha256_file(a1.EMC_NPZ) == a1.EMC_NPZ_SHA256, "EMC npz sha256")
    prov = a1.load_hist_provenance()
    emc, emc_digest = a1.load_emc_identity()
    w0C = w0_c()
    meta = {"experiment_id": EXPERIMENT_ID, "part": "signal_strength", "git_head": a1.git_head(),
            "branch": a1._git("rev-parse", "--abbrev-ref", "HEAD"), "working_tree_clean": True,
            "environment": a1.environment(), "prereg_sha256": prereg_hash(),
            "instrumented_runner": RUNNER_VERSION, "instrumented_blob": INSTRUMENTED_BLOB,
            "w0_C": w0C, "w_C": W_C, "c_F": C_F, "c_w": C_W,
            "population_density_A": population_density(W0_A, W_A),
            "population_density_C": population_density(w0C, W_C),
            "inputs": {"emc_npz": str(a1.EMC_NPZ), "emc_npz_sha256": a1.EMC_NPZ_SHA256,
                       "emc_fit_results_sha256": emc_digest}}
    return prov, emc, w0C, meta


def reuse_A(n, rep, bundle, emc, npz, ref) -> list[dict[str, Any]]:
    rows = []
    v = bundle["cond"]["A"]
    for k in hist.CANDIDATE_K:
        for s in hist.STARTS:
            er = emc[(K_GEN, n, rep, k, s)]
            fi = int(er["fit_index"])
            key, model, Zs, Z = cpr.state(npz, fi, CHECKPOINT, n, k, bundle["train"])
            r = evaluate(model, v["X"], v["Y"], Zs, Z, n=n, fit_id=fi)
            e = ref[(n, rep, k, s)]
            need(r["C_Q"] == float(e["C_Q"]) and r["lap_status"] == e["lap_status"],
                 f"A re-evaluation differs from stored C_Q/status {key}")
            if r["lap_status"] == "OK":
                need(math.isclose(r["C_Lap"], float(e["C_Lap"]), rel_tol=1e-12, abs_tol=1e-9),
                     f"A C_Lap differs {key}")
            rows.append({"condition": "A", "n": n, "replicate": rep, "k_est": k, "start": s,
                         "fit_id": fi, "model_seed": int(er["model_seed"]), "source": f"emc_state:{key}",
                         "fit_status": "REUSED", "checkpoint": CHECKPOINT, "exception": "", **r})
    return rows


def load_A_reference() -> dict:
    p = ROOT / "expfam" / "results" / "k_selection" / "clap_checkpoint_reeval_20261105" / "full" / "fit_criteria.csv"
    with open(p, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["checkpoint"]) == CHECKPOINT]
    return {(int(r["n"]), int(r["replicate"]), int(r["k_est"]), int(r["start"])): r for r in rows}


def select(rows, conds) -> list[dict[str, Any]]:
    out = []
    for c in conds:
        for n in sorted({r["n"] for r in rows if r["condition"] == c}):
            for rep in REPLICATES:
                by = {(r["k_est"], r["start"]): r for r in rows
                      if (r["condition"], r["n"], r["replicate"]) == (c, n, rep)}
                if not by:
                    continue
                failed = sorted(f"K{k}s{s}" for (k, s), r in by.items() if r["fit_status"] == "FAILED")
                ks = sorted({k for k, _ in by})
                for crit in ("C_Q", "C_Lap"):
                    base = {"condition": c, "n": n, "replicate": rep, "criterion": crit}
                    if failed or len(by) != len(hist.CANDIDATE_K) * len(hist.STARTS):
                        out.append({**base, "status": "INCOMPLETE", "reason": "FAILED fits: " + ";".join(failed)})
                        continue
                    if crit == "C_Lap":
                        bad = sorted(f"K{k}s{s}:{r['lap_status']}" for (k, s), r in by.items() if r["lap_status"] != "OK")
                        if bad:
                            out.append({**base, "status": "INCOMPLETE", "reason": ";".join(bad)})
                            continue
                    means = {k: float(np.mean([by[(k, s)][crit] for s in hist.STARTS])) for k in ks}
                    chosen, ties, margin, best = a1._select(means, True)
                    per_start = {s: min(ks, key=lambda k: by[(k, s)][crit]) for s in hist.STARTS}
                    out.append({**base, "status": "COMPLETE", "selected_k": chosen,
                                "label": a1.label(chosen, K_GEN), "margin": margin,
                                "slack_45": means[4] - means[5], "slack_56": means[5] - means[6],
                                "selected_start1": per_start[1], "selected_start2": per_start[2],
                                "starts_agree": per_start[1] == per_start[2], "reason": ""})
    return out


# ---------------------------------------------------------------- stages
def run_gates() -> int:
    out = OUT_ROOT / "gates"
    need(not out.exists(), f"output exists: {out}")
    prov, emc, w0C, meta = common_inputs()
    out.mkdir(parents=True)
    a1.write_json(out / "prereg.json", {**PREREG, "sha256": prereg_hash()})
    gates: dict[str, Any] = {}
    fits_used = 0
    try:
        need(abs(meta["population_density_C"] - meta["population_density_A"]) < 1e-10, "w0 calibration")
        gates["G0_w0_calibration"] = f"PASS (w0_C={w0C:.10f}, density {meta['population_density_A']:.8f})"
        prov_rows, bundles = [], {}
        for n in (75, 150):
            for rep in REPLICATES:
                b = build_all(n, rep, prov, w0C)
                b2 = build_all(n, rep, prov, w0C)
                for c in CONDITIONS:
                    need(all(np.array_equal(b["cond"][c][f], b2["cond"][c][f]) for f in ("X", "Y", "F")),
                         f"regeneration not deterministic {c} n{n} r{rep}")
                rows = provenance_rows(n, rep, b)
                byc = {r["condition"]: r for r in rows}
                need(byc["B"]["y_identical_to_A"] and byc["C"]["x_identical_to_A"], "sharing rule")
                need(byc["D"]["hash_X"] == byc["B"]["hash_X"] and byc["D"]["hash_Y"] == byc["C"]["hash_Y"], "D sharing")
                need(len({r["hash_Z"] for r in rows}) == 1 and len({r["hash_train"] for r in rows}) == 1, "Z/mask shared")
                prov_rows += rows
                bundles[(n, rep)] = b
        write_rows(out / "dataset_provenance.csv", prov_rows)
        gates["G1_data_sharing_support_rank_rate"] = (
            f"PASS ({len(prov_rows)} condition-datasets; max union bound "
            f"{max(r['gate_fail_prob_union_bound'] for r in prov_rows):.3g})")
        # coupling marginal check (pooled standardized residual of the coupled block)
        zs = []
        for (n, rep), b in bundles.items():
            ds = b["ds"]
            lamB = np.exp(ds.Z @ b["cond"]["B"]["F"].T)
            zs.append(float((b["cond"]["B"]["X"].sum() - lamB.sum()) / math.sqrt(lamB.sum())))
            G = ds.Z @ ds.Z.T
            up = np.triu(np.ones((n, n), dtype=bool), 1)
            pc = expit(w0C + W_C * G)[up]
            zs.append(float((b["cond"]["C"]["Y"][up].sum() - pc.sum()) / math.sqrt(np.sum(pc * (1 - pc)))))
        gates["G2_coupling_marginal_z"] = {"max_abs_z": max(abs(z) for z in zs), "mean_z": float(np.mean(zs)),
                                           "note": "z of the coupled totals against their canonical means; diagnostic"}
        need(max(abs(z) for z in zs) < 5.0, "coupling marginal z beyond 5")
        npz = np.load(a1.EMC_NPZ)
        ref = load_A_reference()
        a_rows = reuse_A(75, 1, bundles[(75, 1)], emc, npz, ref) + reuse_A(150, 1, bundles[(150, 1)], emc, npz, ref)
        gates["G3_A_reevaluation_equals_stored"] = f"PASS ({len(a_rows)} states, n=75/150 rep 1)"
        # replication fit
        n, rep, k, s = REPLICATION_FIT
        b = bundles[(n, rep)]
        seed = hist.model_seed(K_GEN, n, rep, k, s)
        res, rec, model, Zs, Z, _q = fit_one(b["cond"]["A"]["X"], b["cond"]["A"]["Y"], b["train"], k, seed)
        fits_used += 1
        fi = int(emc[(K_GEN, n, rep, k, s)]["fit_index"])
        key = f"{fi:04d}_it{CHECKPOINT:03d}"
        mine = state_arrays(rec, Zs, Z)
        diffs = {nm: float(np.max(np.abs(mine[nm] - np.asarray(npz[f"{key}_{nm}"], float))))
                 for nm in ("Zsamples", "F", "sigma", "scalars")}
        ok = all(np.allclose(mine[nm], np.asarray(npz[f"{key}_{nm}"], float), rtol=a1.NPZ_RTOL, atol=a1.NPZ_ATOL)
                 for nm in diffs)
        a1.write_json(out / "replication.json", {"fit": list(REPLICATION_FIT), "model_seed": seed, "match": ok,
                                                 "maxabs": diffs, "runtime_s": res["runtime_s"],
                                                 "checkpoints_used": "1..8 (Q only) vs EMC 8..40"})
        need(ok, "replication fit does not match the archived A1 state")
        gates["G4_replication_fit"] = "PASS (1 fit; checkpoints 1..8 leave the path unchanged)"
        # pilot fits
        prow, arrays = [], {}
        for n, c, rep, k, s in PILOT_FITS:
            b = bundles[(n, rep)]
            v = b["cond"][c]
            seed = hist.model_seed(K_GEN, n, rep, k, s)
            t0 = time.time()
            res, rec, model, Zs, Z, qtraj = fit_one(v["X"], v["Y"], b["train"], k, seed)
            fits_used += 1
            r = evaluate(model, v["X"], v["Y"], Zs, Z, n=n, fit_id=9000 + fits_used)
            need(math.isclose(r["C_Q"], rec["bic"], rel_tol=1e-9, abs_tol=1e-6), "panel C_Q != runner BIC")
            # gradient / Hessian finite-difference spot check at the Laplace start point
            g = lk.gradient(model, v["X"], v["Y"], Z)
            H = lk.negative_hessian(model, v["X"], v["Y"], Z)
            rng = np.random.default_rng(1)
            idx = rng.choice(Z.size, 6, replace=False)
            h = 1e-5
            fd_g, fd_h = [], []
            for j in idx:
                e = np.zeros(Z.size); e[j] = h
                E = e.reshape(Z.shape)
                fp = lk.complete_log_density(model, v["X"], v["Y"], Z + E)
                fm = lk.complete_log_density(model, v["X"], v["Y"], Z - E)
                fd_g.append(abs((fp - fm) / (2 * h) - g.reshape(-1)[j]) / (1 + abs(g.reshape(-1)[j])))
                gp = lk.gradient(model, v["X"], v["Y"], Z + E).reshape(-1)
                gm = lk.gradient(model, v["X"], v["Y"], Z - E).reshape(-1)
                col = -(gp - gm) / (2 * h)
                fd_h.append(float(np.max(np.abs(col - H[:, j])) / (1 + np.max(np.abs(H[:, j])))))
            prow.append({"n": n, "condition": c, "replicate": rep, "k_est": k, "start": s, "model_seed": seed,
                         "runtime_s": round(time.time() - t0, 2), "fd_grad_max_rel": max(fd_g),
                         "fd_hess_max_rel": max(fd_h), "q_trajectory": json.dumps(qtraj), **r})
            for nm, arr in state_arrays(rec, Zs, Z).items():
                arrays[f"pilot_n{n}_{c}_r{rep}_k{k}_s{s}_{nm}"] = arr
        write_rows(out / "pilot_fits.csv", prow)
        write_npz(STATE_ROOT / "pilot_states.npz", arrays)
        need(all(r["fd_grad_max_rel"] < 1e-5 and r["fd_hess_max_rel"] < 1e-4 for r in prow), "FD check")
        need(all(math.isfinite(r["C_Q"]) for r in prow), "non-finite C_Q in pilot")
        gates["G5_pilot"] = {"fits": len(prow), "lap_status": [r["lap_status"] for r in prow],
                             "max_identity_error": max(abs(r.get("identity_error", 0.0) or 0.0) for r in prow),
                             "max_fd_grad_rel": max(r["fd_grad_max_rel"] for r in prow),
                             "max_fd_hess_rel": max(r["fd_hess_max_rel"] for r in prow),
                             "runtime_s": [r["runtime_s"] for r in prow],
                             "note": "technical only; pilot criterion values are not inspected for the plan"}
        est = (expected_new_fits(75) * float(np.mean([r["runtime_s"] for r in prow if r["n"] == 75]))
               + expected_new_fits(150) * float(prow[-1]["runtime_s"])) / 3600.0
        gates["G6_resource"] = {"planned_new_fits": expected_new_fits(75) + expected_new_fits(150),
                                "gate_fits": fits_used, "cap": FIT_CAP,
                                "estimated_hours_upper_K5_runtime": round(est, 2)}
        need(expected_new_fits(75) + expected_new_fits(150) + fits_used <= FIT_CAP, "fit cap")
        a1.write_json(out / "gates.json", {**meta, "status": "PASSED", "gates": gates})
        print(json.dumps(gates, indent=1, default=str))
        return 0
    except Exception as exc:
        a1.write_json(out / "gates.json", {**meta, "status": "FAILED", "gates": gates, "fits_used": fits_used,
                                           "error_type": type(exc).__name__, "error": str(exc)})
        print(f"GATE FAILED: {type(exc).__name__}: {exc}")
        return 2


def run_full(n: int) -> int:
    out = OUT_ROOT / f"full_n{n}"
    need(not out.exists(), f"output exists: {out}")
    g = json.loads((OUT_ROOT / "gates" / "gates.json").read_text(encoding="utf-8"))
    need(g["status"] == "PASSED" and g["prereg_sha256"] == prereg_hash(), "gates not passed / prereg changed")
    prov, emc, w0C, meta = common_inputs()
    out.mkdir(parents=True)
    started = time.time()
    rows: list[dict[str, Any]] = []
    iters: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    npz = np.load(a1.EMC_NPZ)
    ref = load_A_reference()
    prov_rows = []
    try:
        bundles = {}
        for rep in REPLICATES:
            bundles[rep] = build_all(n, rep, prov, w0C)
            prov_rows += provenance_rows(n, rep, bundles[rep])
            rows += reuse_A(n, rep, bundles[rep], emc, npz, ref)
        write_rows(out / "dataset_provenance.csv", prov_rows)
        print(f"n={n}: A re-evaluated ({time.time() - started:.0f}s)", flush=True)
        local = 0
        for c in NEW_BY_N[n]:
            for rep in REPLICATES:
                b = bundles[rep]
                v = b["cond"][c]
                for k in hist.CANDIDATE_K:
                    for s in hist.STARTS:
                        local += 1
                        fit_id = n * 10000 + local
                        seed = hist.model_seed(K_GEN, n, rep, k, s)
                        base = {"condition": c, "n": n, "replicate": rep, "k_est": k, "start": s,
                                "fit_id": fit_id, "model_seed": seed, "source": "new_fit", "checkpoint": CHECKPOINT}
                        t0 = time.time()
                        try:
                            res, rec, model, Zs, Z, qtraj = fit_one(v["X"], v["Y"], b["train"], k, seed)
                            t_em = time.time() - t0
                            r = evaluate(model, v["X"], v["Y"], Zs, Z, n=n, fit_id=fit_id)
                            need(math.isclose(r["C_Q"], rec["bic"], rel_tol=1e-9, abs_tol=1e-6), "panel C_Q != BIC")
                            for nm, arr in state_arrays(rec, Zs, Z).items():
                                arrays[f"n{n}_{c}_r{rep}_k{k}_s{s}_{nm}"] = arr
                            rows.append({**base, "fit_status": "FIT", "exception": "",
                                         "em_runtime_s": round(t_em, 3), "eval_runtime_s": round(time.time() - t0 - t_em, 3),
                                         **{col: float(rec[col]) for col in DIAG_COLS}, **r})
                            for it, dg in enumerate(res["iteration_diagnostics"], start=1):
                                iters.append({**{x: base[x] for x in ("condition", "n", "replicate", "k_est", "start", "fit_id")},
                                              "iteration": it, "q_strict": qtraj[it][0], "C_Q": qtraj[it][1],
                                              **{kk: float(vv) for kk, vv in dg.items()}})
                            ledger.append({**base, "status": "FIT", "attempts": 1, "retry": 0,
                                           "runtime_s": round(time.time() - t0, 3), "exception": ""})
                        except (RetryDetected, Stop, FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                            rows.append({**base, "fit_status": "FAILED", "exception": f"{type(exc).__name__}: {exc}"})
                            ledger.append({**base, "status": "FAILED", "attempts": 1, "retry": 0,
                                           "runtime_s": round(time.time() - t0, 3),
                                           "exception": f"{type(exc).__name__}: {exc}"})
                print(f"n={n} {c} rep={rep} done ({time.time() - started:.0f}s)", flush=True)
    except Exception as exc:
        write_rows(out / "fit_criteria_partial.csv", rows)
        write_rows(out / "execution_ledger_partial.csv", ledger)
        a1.write_json(out / "failure.json", {**meta, "status": "STOPPED", "error_type": type(exc).__name__,
                                             "error": str(exc), "completed_rows": len(rows)})
        print(f"STOPPED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    new = sum(r["source"] == "new_fit" for r in rows)
    need(new == expected_new_fits(n), f"new fits {new}")
    state_sha = write_npz(STATE_ROOT / f"fit_states_n{n}.npz", arrays)
    write_rows(out / "fit_criteria.csv", rows)
    write_rows(out / "iteration_diagnostics.csv", iters)
    write_rows(out / "execution_ledger.csv", ledger)
    write_rows(out / "selection.csv", select(rows, ("A",) + NEW_BY_N[n]))
    a1.write_json(out / "runinfo.json", {
        **meta, "status": "COMPLETED", "n": n, "rows": len(rows), "new_fits": new,
        "failed_fits": sum(r["fit_status"] == "FAILED" for r in rows),
        "reused_A_states": sum(r["source"] != "new_fit" for r in rows),
        "wall_clock_seconds": round(time.time() - started, 1),
        "laplace_status_counts": {f"{c}:{s}": sum(1 for r in rows if r["condition"] == c and r.get("lap_status") == s)
                                  for c in CONDITIONS for s in ("OK", "NOT_STATIONARY", "NOT_PD", "FAILED")},
        "fit_states": str(STATE_ROOT / f"fit_states_n{n}.npz"), "fit_states_sha256": state_sha})
    print(f"completed n={n}: new={new}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gates", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--n", type=int, choices=(75, 150))
    a = ap.parse_args(argv)
    if a.gates:
        return run_gates()
    need(a.n is not None, "--full needs --n")
    return run_full(a.n)


if __name__ == "__main__":
    sys.exit(main())
