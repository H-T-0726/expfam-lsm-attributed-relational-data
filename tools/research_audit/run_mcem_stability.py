"""MCEM stability of the signal-strength effect (RQ1) and signal-matched K=3/K=5 (RQ2).

Lineage E (prototype).  Stacked on research/k-selection-causal-analysis-20261106 (PR #155).

RQ1  Same data as the PR #155 signal-strength experiment (n=75, K_GEN=5, d=15, conditions
     A/B/C/D, replicates 1..8), K_fit in {4,5}, starts {1,2}, the SAME model seeds.  Each fit
     is re-run from iteration 1 with the same seed to 64 iterations (no state resume: the
     archived states carry no RNG state).  The iteration-8 state must reproduce the archived
     PR #155 / EMC state (replication gate, every fit).  Full evaluation (C_Q, C_Lap, X/Y/Z
     split, Laplace terms, independent-MC C_Q) at checkpoints 8,16,24,32,48,64; C_Q and the
     step diagnostics at every iteration.
RQ2  New canonical data with per-direction signal matched across K_GEN=3 and K_GEN=5:
     F^T F non-zero eigenvalues 2.5, w^2 = 1, w0 calibrated per K to the A population
     density.  Z_3 = first 3 columns of Z_5, F_3 = first 3 columns of F_5 (shared directions),
     X_3 / Y_3 exact couplings of X_5 / Y_5, shared mask.  K_fit in {K_GEN-1, K_GEN, K_GEN+1},
     32 iterations, checkpoints 8,16,24,32.

Independent MC evaluation: at each full checkpoint a DEEP COPY of the model draws L_EVAL
E-step samples with its own RNG SeedSequence([EVAL_TAG, model_seed, checkpoint]); the
estimation RNG and state are never touched (gate G3 checks bit identity).

Stages:  --gates   |   --rq1 --conds A B   |   --rq1 --conds C D   |   --rq2
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
_EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
for _p in (str(_EXPERIMENTAL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_k_signal_strength as ss                         # noqa: E402
import k_criterion_panel as panel                          # noqa: E402
from em_runner_instrumented import RetryDetected, run_em_instrumented, RUNNER_VERSION  # noqa: E402
from data_generator_canonical import build_full_rank_loadings, canonical_poisson_rate  # noqa: E402

a1, hist, cpr = ss.a1, ss.hist, ss.cpr

EXPERIMENT_ID = "mcem_stability_signal_matched_20261106"
OUT_REL = f"expfam/results/k_selection/{EXPERIMENT_ID}"
OUT_ROOT = ROOT / OUT_REL
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / EXPERIMENT_ID
PREV = ROOT / "expfam" / "results" / "k_selection" / "historical_canonical_causal_20261106"
PREV_STATES = Path(r"D:\tento\kennkyu_artifacts\historical_canonical_causal_20261106\fit_states_n75.npz")
PREV_STATES_SHA = "<filled at gates>"

N = 75
D = 15
REPS = tuple(range(1, 9))
STARTS = (1, 2)
RQ1_K = (4, 5)
RQ1_MAX_ITER = 64
RQ1_CPS = (8, 16, 24, 32, 48, 64)
EMC_CPS = (8, 16, 24, 32)                       # EMC checkpoints also in RQ1_CPS (cross-check of A)
RQ2_KGEN = (3, 5)
RQ2_MAX_ITER = 32
RQ2_CPS = (8, 16, 24, 32)
RQ2_EIG = 2.5
RQ2_W = 1.0
RQ2_DATA_TAG = 7300000
RQ2_SPLIT_BASE = 7400000
RQ2_MODEL_BASE = 7500000
EVAL_TAG = 7600000
L_EVAL = 20
STAB_WINDOW = 8
STAB_TOL = 0.01
BUDGET_FIT_ITERS = 12000
GATE_FIT_ITERS = 64 + 16
WALL_LIMIT_H = 7.0


def rq1_fit_iters() -> int:
    return 4 * len(REPS) * len(RQ1_K) * len(STARTS) * RQ1_MAX_ITER


def rq2_fit_iters() -> int:
    return len(RQ2_KGEN) * len(REPS) * 3 * len(STARTS) * RQ2_MAX_ITER


PREREG = {
    "experiment_id": EXPERIMENT_ID, "lineage": "E (numerics_mode=consistent), prototype; manuscript adoption NOT decided",
    "base": "PR #155 head 72f33cf (research/k-selection-causal-analysis-20261106)",
    "RQ1": {
        "question": "Is the PR #155 signal-strength effect on the C_Q / C_Lap 4->5 slack maintained when the same MCEM fits continue from 8 to 64 iterations?",
        "data": "PR #155 conditions A/B/C/D, n=75, K_GEN=5, d=15, replicates 1..8 (regenerated; hashes checked against PR #155 provenance)",
        "fits": "K_fit in {4,5} x starts {1,2} x 4 conditions x 8 replicates = 128 fits, max_iter 64, L=5, same model seeds as PR #155 (A1 formula)",
        "continuation": "re-run from iteration 1 with the same seed (archived states carry no RNG state, so exact resume is impossible); iteration-8 state must match the archived PR #155 / EMC state within NPZ_RTOL=1e-9, NPZ_ATOL=1e-6 and C_Q must match; a non-matching fit is flagged REPLICATION_FAIL and excluded from continuation claims (kept in all CSVs)",
        "checkpoints_full": list(RQ1_CPS), "checkpoints_C_Q_trajectory": "every iteration 1..64",
        "local_comparison": "S(4->5) = mean_s C(4,s) - mean_s C(5,s) for C in {C_Q, C_Lap}; this is a LOCAL 4-vs-5 comparison, not a K_hat over 1..7",
        "hypotheses": {
            "RQ1-B": "paired slack difference B-A (C_Q) at checkpoint 64 > 0",
            "RQ1-C": "paired slack difference C-A (C_Q) at checkpoint 64 > 0",
            "RQ1-D": "paired slack difference D-A (C_Q) at checkpoint 64 > 0",
            "rule": "MAINTAINED: median > 0 and >= 6/8 positive; NOT_MAINTAINED: median <= 0; otherwise INCONCLUSIVE.  Same rule reported for C_Lap (secondary) and at every checkpoint (descriptive).  Datasets with a REPLICATION_FAIL fit or a non-OK Laplace (C_Lap only) are excluded from that contrast and counted.",
            "sign_change": "descriptive: per condition and criterion, number of datasets whose S(4->5) sign at 64 differs from at 8",
            "criteria_gap": "descriptive: S_Lap - S_Q = 2dH_Lap + d resid at each checkpoint; number of datasets where sign(S_Q) != sign(S_Lap) at 8 and at 64",
        },
        "stability_rules": {
            "fit_STABLE_at_c": f"max over the last {STAB_WINDOW} iterations up to c of gram_rel_change, eta_x_rel_change and eta_y_rel_change all <= {STAB_TOL}; label at 64 is STABLE or NOT_STABLE (never relaxed)",
            "dataset_SIGN_STABLE": "sign of S(4->5) identical at checkpoints 32, 48 and 64",
            "dataset_SLACK_STABLE": "|S(64) - S(48)| <= 2 * se, se = MC standard error of S from the independent evaluation at 64",
            "note": "L=5 stochastic MCEM never converges in the deterministic EM sense; STABLE is a numerical-stability label, not convergence and not consistency",
        },
        "independent_MC": f"L_EVAL={L_EVAL} E-step draws from a deep copy with SeedSequence([{EVAL_TAG}, model_seed, checkpoint]); C_Q_eval and its MC SE; the estimation RNG is untouched (gate G3)",
    },
    "RQ2": {
        "question": "With per-direction signal matched, is K_GEN=5 still harder to select (locally) than K_GEN=3?",
        "data": (f"per replicate: Z5 ~ N(0, I5); F5 = Q diag({RQ2_EIG}^0.5) (reduced QR, all F^T F eigenvalues {RQ2_EIG}); "
                 "Z3 = Z5[:, :3], F3 = F5[:, :3] (eigenvalues 2.5); X5 ~ Pois(exp(Z5 F5^T)), X3 exact coupling of X5; "
                 f"w = {RQ2_W} for both, w0_K calibrated so the population density equals the canonical A density; "
                 "Y5 ~ Bern, Y3 exact coupling of Y5; one train/test mask shared; total signal is NOT matched (K=5 carries 5/3 of K=3)"),
        "seeds": {"data": f"SeedSequence([{RQ2_DATA_TAG}, rep])", "split": f"{RQ2_SPLIT_BASE} + rep",
                  "model": f"{RQ2_MODEL_BASE} + 100*rep + 10*K_fit + start (shared across K_GEN for equal K_fit)"},
        "fits": "K_GEN in {3,5} x K_fit in {K_GEN-1, K_GEN, K_GEN+1} x starts {1,2} x 8 replicates = 96 fits, 32 iterations, L=5",
        "checkpoints_full": list(RQ2_CPS),
        "primary": "paired (by replicate) Delta = S_Q(4->5 | K_GEN=5) - S_Q(2->3 | K_GEN=3) at checkpoint 32",
        "rule": "K_HARDER: median Delta < 0 and >= 6/8 negative; NOT_HARDER: median Delta >= 0; otherwise INCONCLUSIVE.  Also per node fit gains X / Y of the last dimension, the over-fit side S(K_GEN -> K_GEN+1), local correctness (S(K-1->K) > 0 and S(K->K+1) < 0), C_Lap analogues.",
        "caveat": "costs differ slightly: parameter penalty (d-K) ln n is 13 ln 75 for 2->3 and 11 ln 75 for 4->5; Z prior cost identical (n(1+ln 2pi))",
    },
    "budget": {"fit_iterations_cap": BUDGET_FIT_ITERS, "rq1": rq1_fit_iters(), "rq2": rq2_fit_iters(),
               "gates": GATE_FIT_ITERS, "wall_clock_cap_hours": WALL_LIMIT_H,
               "excluded_before_results": "full-candidate K (1..7) supplement for B/C/D at 64 iterations needs 15,360 fit-iterations alone; not run. A has full-K up to 40 iterations in the EMC / checkpoint re-evaluation experiments (prior evidence).",
               "phase_cut_rule": "if the gate timing projects RQ1+RQ2 wall clock > 6 h with 3 parallel processes, RQ2 is not started (RQ1 first); any stage exceeding the 7 h cap stops and is recorded PARTIAL"},
    "failure_rules": ["RetryDetected / exception in a fit -> FAILED row, no retry, no reseed",
                      "Laplace not OK -> C_Lap INCOMPLETE for that dataset/checkpoint, not imputed",
                      "no threshold relaxation, no seed selection, no condition change after results"],
    "required_csvs": ["experiment_protocol.csv", "dataset_provenance.csv", "fit_execution_ledger.csv", "checkpoint_metrics.csv",
                      "checkpoint_criterion_scores.csv", "checkpoint_q_decomposition.csv", "stability_diagnostics.csv",
                      "paired_slack_changes.csv", "local_k4_k5_comparison.csv", "full_k_selection.csv (not produced: not in scope)",
                      "signal_matched_k3_k5.csv", "failure_and_incomplete.csv", "audit_summary.csv"],
}


class Stop(RuntimeError):
    pass


def need(c, m):
    if not c:
        raise Stop(m)


def prereg_hash() -> str:
    return hashlib.sha256(json.dumps(PREREG, sort_keys=True).encode()).hexdigest()


def tree_clean_except_outputs() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", ".", f":(exclude){OUT_REL}"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8").stdout
    return out.strip() == ""


# --------------------------------------------------------------------------- data
def rq1_data(prov, w0C):
    return {rep: ss.build_all(N, rep, prov, w0C) for rep in REPS}


def calibrate_w0_k(w: float, k: int) -> float:
    """w0 such that E sigma(w0 + w z.z'), z in R^k, equals the canonical A population density."""
    from scipy.optimize import brentq
    target = ss.population_density(ss.W0_A, ss.W_A, k=5)
    return float(brentq(lambda b: ss.population_density(b, w, k=k) - target, -15.0, 5.0, xtol=1e-13))


def rq2_data(rep: int) -> dict[str, Any]:
    rng = np.random.default_rng([RQ2_DATA_TAG, rep])
    Z5 = rng.standard_normal((N, 5))
    F5 = build_full_rank_loadings(rng, d=D, k=5, f_scale=math.sqrt(RQ2_EIG))
    lam5 = canonical_poisson_rate(Z5 @ F5.T, lambda_max=hist.POISSON_LAMBDA_MAX)
    X5 = rng.poisson(lam5).astype(np.float64)
    w0 = {k: calibrate_w0_k(RQ2_W, k) for k in RQ2_KGEN}
    up = np.triu(np.ones((N, N), dtype=bool), 1)
    p5 = expit(w0[5] + RQ2_W * (Z5 @ Z5.T))
    Y5 = np.zeros((N, N)); Y5[up] = (rng.random(int(up.sum())) < p5[up]).astype(float); Y5 = Y5 + Y5.T
    Z3, F3 = Z5[:, :3].copy(), F5[:, :3].copy()
    lam3 = np.exp(Z3 @ F3.T)
    X3 = ss.couple_poisson(X5, lam5, lam3, np.random.default_rng([RQ2_DATA_TAG, rep, 3, 1]))
    p3 = expit(w0[3] + RQ2_W * (Z3 @ Z3.T))
    Y3 = ss.couple_bernoulli_upper(Y5, p5, p3, np.random.default_rng([RQ2_DATA_TAG, rep, 3, 2]))
    train, test = hist.build_masks(N, RQ2_SPLIT_BASE + rep)
    return {"train": train, "test": test,
            5: {"Z": Z5, "F": F5, "X": X5, "Y": Y5, "w0": w0[5], "w": RQ2_W, "lam": lam5, "p": p5},
            3: {"Z": Z3, "F": F3, "X": X3, "Y": Y3, "w0": w0[3], "w": RQ2_W, "lam": lam3, "p": p3}}


def sha(a) -> str:
    return ss.sha(a)


def rq2_provenance(rep, b) -> list[dict[str, Any]]:
    up = np.triu(np.ones((N, N), dtype=bool), 1)
    rows = []
    for k in RQ2_KGEN:
        v = b[k]
        sv = np.linalg.svd(v["F"], compute_uv=False)
        need(int(np.linalg.matrix_rank(v["F"])) == k and np.allclose(sv ** 2, RQ2_EIG, atol=1e-9), f"F spectrum K={k}")
        need(float(v["lam"].max()) <= hist.POISSON_LAMBDA_MAX, "rate gate")
        X, Y = v["X"], v["Y"]
        need(np.all(X >= 0) and np.all(X == np.floor(X)) and np.array_equal(Y, Y.T) and np.all(np.diag(Y) == 0), "support")
        rows.append({"rq": "RQ2", "condition": f"K{k}", "n": N, "d": D, "k_gen": k, "replicate": rep,
                     "data_seed": json.dumps([RQ2_DATA_TAG, rep]), "split_seed": RQ2_SPLIT_BASE + rep,
                     "w": v["w"], "w0": v["w0"], "population_density": ss.population_density(v["w0"], v["w"], k=k),
                     "F_eigenvalues": json.dumps([float(s) for s in sv ** 2]), "mean_row_norm_sq": float(np.mean(np.sum(v["F"] ** 2, 1))),
                     "max_rate_truth": float(v["lam"].max()), "x_mean": float(X.mean()), "y_density_upper": float(Y[up].mean()),
                     "bernoulli_fisher_mean_p1mp": float(np.mean(v["p"][up] * (1 - v["p"][up]))),
                     "hash_Z": sha(v["Z"]), "hash_X": sha(X), "hash_Y": sha(Y), "hash_train": sha(b["train"]),
                     "shared": "Z3=Z5[:, :3], F3=F5[:, :3], mask shared; X3/Y3 exact couplings of X5/Y5"})
    return rows


# --------------------------------------------------------------------------- fitting
def independent_eval(model_copy, X, Y, Z_est, seed: int, cp: int, n: int) -> dict[str, float]:
    m = model_copy
    rng = np.random.default_rng([EVAL_TAG, seed, cp])
    Z = np.array(Z_est, dtype=float)
    k = Z.shape[1]
    S = np.zeros((n, k, L_EVAL))
    for l in range(L_EVAL):
        m.params["Z"] = Z.copy()
        Z = m.calc_eta_newton(X, Y, rng=rng, max_iter=10, alpha=0.5)
        S[:, :, l] = Z
    need(np.all(np.isfinite(S)), "non-finite independent samples")
    S = m.scale_Z(S)
    npar = k * D - k * (k - 1) // 2
    cq = panel.cq_terms(m, X, Y, S, num_params=npar, n=n)
    t = panel.theta_of(m)
    per = []
    for l in range(L_EVAL):
        per.append(-2.0 * float(panel.calc_Q_dual_strict_exp(X, Y, S[:, :, l:l + 1], t["F"], t["sigma"], t["var_z"],
                                                             t["w0"], t["w"], m)) + npar * math.log(n))
    return {"C_Q_eval": cq["C_Q"], "Q_X_eval": cq["Q_X"], "Q_Y_eval": cq["Q_Y"], "Q_Z_eval": cq["Q_Z"],
            "C_Q_eval_mcse": float(np.std(per, ddof=1) / math.sqrt(L_EVAL))}


def run_fit(X, Y, train, k, seed, max_iter, cps, *, do_eval=True, n=N):
    snaps, traj = {}, {}

    def cb(rec):
        it = int(rec["iteration"])
        traj[it] = (float(rec["q_strict"]), float(rec["bic"]))
        if it in cps:
            snaps[it] = {"model": copy.deepcopy(rec["model"]), "Zs": np.array(rec["Z_samples"]),
                         "Z": np.array(rec["Z_est"]), "F": np.array(rec["F"]), "sigma": np.array(rec["sigma"]),
                         "w0": float(rec["w0"]), "w": float(rec["w"]), "var_z": float(rec["var_z"]),
                         "bic": float(rec["bic"]),
                         "diag": {kk: float(rec[kk]) for kk in rec if kk.startswith(("max_", "gram_", "f_raw", "w_change", "w0_change", "eta_"))}}
            if do_eval:
                snaps[it]["indep"] = independent_eval(copy.deepcopy(rec["model"]), X, Y, rec["Z_est"], seed, it, n)
    res = run_em_instrumented(X, Y, family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y, k=k, L=hist.L_SAMPLES,
                              max_iter=max_iter, checkpoints=tuple(range(1, max_iter + 1)), seed=seed,
                              train_mask=train, numerics_mode=hist.NUMERICS_MODE,
                              stability_window=STAB_WINDOW, on_nan="fail", on_checkpoint=cb)
    need(res["retry_count"] == 0 and not res["nan_detected"], "unexpected retry")
    return res, snaps, traj


def evaluate_snap(sn, X, Y, n, fit_id) -> dict[str, Any]:
    m = sn["model"]
    m.params.update({"Z": sn["Z"].copy(), "F": sn["F"], "sigma": sn["sigma"], "w0": sn["w0"], "w": sn["w"],
                     "var_z": sn["var_z"]})
    r = ss.evaluate(m, X, Y, sn["Zs"], sn["Z"], n=n, fit_id=fit_id)
    need(math.isclose(r["C_Q"], sn["bic"], rel_tol=1e-9, abs_tol=1e-6), "panel C_Q != runner BIC")
    return r


def state_arrays(sn):
    return {"Zsamples": sn["Zs"], "Z": sn["Z"], "F": sn["F"], "sigma": sn["sigma"],
            "scalars": np.array([sn["w0"], sn["w"], sn["var_z"]], dtype=np.float64)}


def archived_state(cond, rep, k, s, cp, emc, emc_npz, prev_npz):
    if cond == "A":
        fi = int(emc[(5, N, rep, k, s)]["fit_index"])
        key = f"{fi:04d}_it{cp:03d}"
        if f"{key}_Zsamples" not in emc_npz.files:
            return None, key
        return {nm: np.asarray(emc_npz[f"{key}_{nm}"], float) for nm in ("Zsamples", "F", "sigma", "scalars")}, f"emc:{key}"
    if cp != 8:
        return None, ""
    key = f"n{N}_{cond}_r{rep}_k{k}_s{s}"
    return {nm: np.asarray(prev_npz[f"{key}_{nm}"], float) for nm in ("Zsamples", "F", "sigma", "scalars")}, f"pr155:{key}"


def compare_state(mine, ref) -> tuple[bool, float]:
    diffs = [float(np.max(np.abs(mine[nm] - ref[nm]))) for nm in ref]
    ok = all(np.allclose(mine[nm], ref[nm], rtol=a1.NPZ_RTOL, atol=a1.NPZ_ATOL) for nm in ref)
    return ok, max(diffs)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ss.write_rows(path, rows)


def common(meta_extra=None):
    need(tree_clean_except_outputs(), "working tree (outside outputs) must be clean")
    need(a1._git("rev-parse", "HEAD:expfam/src/experimental/em_runner_instrumented.py") == ss.INSTRUMENTED_BLOB,
         "instrumented runner blob")
    need(a1.sha256_file(a1.EMC_NPZ) == a1.EMC_NPZ_SHA256, "EMC npz sha256")
    prev_info = json.loads((PREV / "signal_strength" / "full_n75" / "runinfo.json").read_text(encoding="utf-8"))
    need(a1.sha256_file(PREV_STATES) == prev_info["fit_states_sha256"], "PR #155 n=75 state npz sha256")
    prov = a1.load_hist_provenance()
    emc, _ = a1.load_emc_identity()
    w0C = ss.w0_c()
    meta = {"experiment_id": EXPERIMENT_ID, "git_head": a1.git_head(), "branch": a1._git("rev-parse", "--abbrev-ref", "HEAD"),
            "environment": a1.environment(), "prereg_sha256": prereg_hash(), "instrumented_runner": RUNNER_VERSION,
            "inputs": {"emc_npz": str(a1.EMC_NPZ), "emc_npz_sha256": a1.EMC_NPZ_SHA256,
                       "pr155_states": str(PREV_STATES), "pr155_states_sha256": prev_info["fit_states_sha256"]},
            **(meta_extra or {})}
    return prov, emc, w0C, meta


def prev_provenance() -> dict:
    import pandas as pd
    p = pd.read_csv(PREV / "analysis" / "dataset_provenance.csv")
    return {(int(r.n), r.condition, int(r.replicate)): r for r in p.itertuples()}


def fit_rows(rq, cond, k_gen, rep, k, s, seed, fit_id, res, snaps, traj, X, Y, n, emc, emc_npz, prev_npz, arrays, cps):
    rows, iters = [], []
    rep_status, rep_detail = "NOT_APPLICABLE", ""
    if rq == "RQ1":
        oks, ds = [], []
        for cp in (EMC_CPS if cond == "A" else (8,)):
            ref, key = archived_state(cond, rep, k, s, cp, emc, emc_npz, prev_npz)
            if ref is None:
                continue
            ok, dmax = compare_state(state_arrays(snaps[cp]), ref)
            oks.append(ok); ds.append(f"{key}:{dmax:.2e}")
        rep_status = "MATCH" if all(oks) and oks else "REPLICATION_FAIL"
        rep_detail = ";".join(ds)
    for cp in cps:
        sn = snaps[cp]
        r = evaluate_snap(sn, X, Y, n, fit_id * 100 + cp)
        rows.append({"rq": rq, "experiment_id": EXPERIMENT_ID, "condition": cond, "n": n, "d": D, "k_gen": k_gen,
                     "replicate": rep, "k_fit": k, "start": s, "model_seed": seed, "fit_id": fit_id, "checkpoint": cp,
                     "status": "FIT", "replication_status": rep_status, "replication_detail": rep_detail,
                     **sn["diag"], **sn.get("indep", {}), **r})
        for nm, arr in state_arrays(sn).items():
            arrays[f"{rq}_{cond}_r{rep}_k{k}_s{s}_it{cp}_{nm}"] = arr
    for it, dg in enumerate(res["iteration_diagnostics"], start=1):
        iters.append({"rq": rq, "condition": cond, "k_gen": k_gen, "replicate": rep, "k_fit": k, "start": s,
                      "fit_id": fit_id, "iteration": it, "q_strict": traj[it][0], "C_Q": traj[it][1],
                      **{kk: float(vv) for kk, vv in dg.items()}})
    return rows, iters


# --------------------------------------------------------------------------- stages
def run_gates() -> int:
    out = OUT_ROOT / "gates"
    need(not out.exists(), f"output exists: {out}")
    prov, emc, w0C, meta = common()
    out.mkdir(parents=True)
    a1.write_json(out / "prereg.json", {**PREREG, "sha256": prereg_hash()})
    gates: dict[str, Any] = {}
    try:
        pp = prev_provenance()
        data = rq1_data(prov, w0C)
        bad = 0
        for rep, b in data.items():
            for c in "ABCD":
                r = pp[(N, c, rep)]
                bad += int(sha(b["cond"][c]["X"]) != r.hash_X or sha(b["cond"][c]["Y"]) != r.hash_Y or sha(b["ds"].Z) != r.hash_Z)
        need(bad == 0, f"RQ1 data differ from PR #155 provenance ({bad})")
        gates["G1_rq1_data_equal_pr155"] = "PASS (32 condition-datasets, X/Y/Z hashes)"
        rows = []
        for rep in REPS:
            rows += rq2_provenance(rep, rq2_data(rep))
        need(all(abs(r["population_density"] - ss.population_density(ss.W0_A, ss.W_A)) < 1e-9 for r in rows), "RQ2 density")
        write_rows(out / "rq2_dataset_provenance.csv", rows)
        gates["G2_rq2_data"] = "PASS (rank, eigenvalues 2.5, rate, support, density calibration)"
        emc_npz = np.load(a1.EMC_NPZ); prev_npz = np.load(PREV_STATES)
        b = data[1]; v = b["cond"]["A"]
        seed = hist.model_seed(5, N, 1, 5, 1)
        t0 = time.time()
        res, snaps, traj = run_fit(v["X"], v["Y"], b["train"], 5, seed, RQ1_MAX_ITER, RQ1_CPS)
        t_fit = time.time() - t0
        arrays: dict[str, np.ndarray] = {}
        t1 = time.time()
        frows, _ = fit_rows("RQ1", "A", 5, 1, 5, 1, seed, 1, res, snaps, traj, v["X"], v["Y"], N, emc, emc_npz, prev_npz, arrays, RQ1_CPS)
        t_eval = time.time() - t1
        need(frows[0]["replication_status"] == "MATCH", f"pilot replication: {frows[0]['replication_detail']}")
        gates["G3a_pilot_replicates_emc_8_16_24_32"] = f"PASS ({frows[0]['replication_detail']})"
        res2, snaps2, _ = run_fit(v["X"], v["Y"], b["train"], 5, seed, 16, (16,), do_eval=False)
        same = all(np.array_equal(state_arrays(snaps[16])[nm], state_arrays(snaps2[16])[nm]) for nm in ("Zsamples", "F", "sigma", "scalars"))
        need(same, "independent evaluation changed the estimation path")
        gates["G3b_independent_eval_isolated"] = "PASS (iteration-16 state bit-identical with and without evaluation)"
        need(all(r["lap_status"] == "OK" for r in frows), "pilot Laplace status")
        gates["G3c_pilot_eval"] = {"lap_status": [r["lap_status"] for r in frows],
                                   "max_identity_error": max(abs(r["identity_error"]) for r in frows),
                                   "note": "technical only; values not used to change the plan"}
        per_it = (t_fit + t_eval) / RQ1_MAX_ITER
        proj_rq1 = per_it * rq1_fit_iters() / 3600 / 2          # two parallel RQ1 processes
        proj_rq2 = per_it * rq2_fit_iters() / 3600 * 1.15       # one process, slightly larger K
        gates["G4_timing"] = {"pilot_fit_s": round(t_fit, 1), "pilot_eval_s": round(t_eval, 1),
                              "seconds_per_fit_iteration_incl_eval": round(per_it, 3),
                              "projected_hours_rq1_2proc": round(proj_rq1, 2), "projected_hours_rq2": round(proj_rq2, 2),
                              "rq2_allowed": bool(max(proj_rq1, proj_rq2) <= 6.0),
                              "fit_iterations_planned": rq1_fit_iters() + rq2_fit_iters() + GATE_FIT_ITERS,
                              "cap": BUDGET_FIT_ITERS}
        need(rq1_fit_iters() + rq2_fit_iters() + GATE_FIT_ITERS <= BUDGET_FIT_ITERS, "budget")
        write_rows(out / "pilot_checkpoints.csv", frows)
        a1.write_json(out / "gates.json", {**meta, "status": "PASSED", "gates": gates})
        print(json.dumps(gates, indent=1, default=str))
        return 0
    except Exception as exc:
        a1.write_json(out / "gates.json", {**meta, "status": "FAILED", "gates": gates, "error": f"{type(exc).__name__}: {exc}"})
        print(f"GATE FAILED: {type(exc).__name__}: {exc}")
        return 2


def run_stage(name: str, jobs) -> int:
    out = OUT_ROOT / name
    need(not out.exists(), f"output exists: {out}")
    g = json.loads((OUT_ROOT / "gates" / "gates.json").read_text(encoding="utf-8"))
    need(g["status"] == "PASSED" and g["prereg_sha256"] == prereg_hash(), "gates not passed / prereg changed")
    if name == "rq2":
        need(g["gates"]["G4_timing"]["rq2_allowed"], "RQ2 not allowed by the pre-registered timing rule")
    prov, emc, w0C, meta = common()
    out.mkdir(parents=True)
    started = time.time()
    emc_npz = np.load(a1.EMC_NPZ); prev_npz = np.load(PREV_STATES)
    rows, iters, ledger, arrays = [], [], [], {}
    status = "COMPLETED"
    for job in jobs:
        rq, cond, k_gen, rep, k, s, seed, fit_id, X, Y, train, max_iter, cps = job
        base = {"rq": rq, "condition": cond, "k_gen": k_gen, "replicate": rep, "k_fit": k, "start": s,
                "model_seed": seed, "fit_id": fit_id, "max_iter": max_iter}
        if (time.time() - started) / 3600 > WALL_LIMIT_H - 0.3:
            ledger.append({**base, "status": "NOT_RUN_TIME_CAP", "fit_iterations": 0, "exception": ""})
            status = "PARTIAL"
            continue
        t0 = time.time()
        try:
            res, snaps, traj = run_fit(X, Y, train, k, seed, max_iter, cps)
            fr, it = fit_rows(rq, cond, k_gen, rep, k, s, seed, fit_id, res, snaps, traj, X, Y, N, emc, emc_npz, prev_npz, arrays, cps)
            rows += fr; iters += it
            ledger.append({**base, "status": "FIT", "fit_iterations": max_iter, "replication_status": fr[0]["replication_status"],
                           "runtime_s": round(time.time() - t0, 1), "exception": ""})
        except (RetryDetected, Stop, FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            rows.append({**base, "experiment_id": EXPERIMENT_ID, "n": N, "d": D, "status": "FAILED",
                         "exception": f"{type(exc).__name__}: {exc}"})
            ledger.append({**base, "status": "FAILED", "fit_iterations": max_iter, "runtime_s": round(time.time() - t0, 1),
                           "exception": f"{type(exc).__name__}: {exc}"})
        print(f"[{name}] {rq} {cond} rep{rep} K{k} s{s} done ({time.time() - started:.0f}s)", flush=True)
    sha_ = ss.write_npz(STATE_ROOT / f"{name}_states.npz", arrays)
    write_rows(out / "checkpoint_rows.csv", rows)
    write_rows(out / "iteration_rows.csv", iters)
    write_rows(out / "ledger.csv", ledger)
    a1.write_json(out / "runinfo.json", {**meta, "stage": name, "status": status, "fits": len(ledger),
                                         "fit_status_counts": {s_: sum(l["status"] == s_ for l in ledger) for s_ in {l["status"] for l in ledger}},
                                         "wall_clock_seconds": round(time.time() - started, 1),
                                         "states": str(STATE_ROOT / f"{name}_states.npz"), "states_sha256": sha_})
    print(f"[{name}] {status}", flush=True)
    return 0


def rq1_jobs(conds, prov, w0C):
    data = rq1_data(prov, w0C)
    jobs = []
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    for c in conds:
        for rep in REPS:
            b = data[rep]; v = b["cond"][c]
            for k in RQ1_K:
                for s in STARTS:
                    jobs.append(("RQ1", c, 5, rep, k, s, hist.model_seed(5, N, rep, k, s),
                                 10000 + 1000 * order[c] + 100 * rep + 10 * k + s, v["X"], v["Y"], b["train"],
                                 RQ1_MAX_ITER, RQ1_CPS))
    return jobs


def rq2_jobs():
    jobs = []
    for rep in REPS:
        b = rq2_data(rep)
        for kg in RQ2_KGEN:
            for k in (kg - 1, kg, kg + 1):
                for s in STARTS:
                    jobs.append(("RQ2", f"K{kg}", kg, rep, k, s, RQ2_MODEL_BASE + 100 * rep + 10 * k + s,
                                 20000 + 1000 * kg + 100 * rep + 10 * k + s, b[kg]["X"], b[kg]["Y"], b["train"],
                                 RQ2_MAX_ITER, RQ2_CPS))
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gates", action="store_true")
    g.add_argument("--rq1", action="store_true")
    g.add_argument("--rq2", action="store_true")
    ap.add_argument("--conds", nargs="+", choices=list("ABCD"))
    a = ap.parse_args(argv)
    if a.gates:
        return run_gates()
    if a.rq1:
        need(a.conds, "--rq1 needs --conds")
        prov = a1.load_hist_provenance()
        return run_stage("rq1_" + "".join(a.conds), rq1_jobs(a.conds, prov, ss.w0_c()))
    return run_stage("rq2", rq2_jobs())


if __name__ == "__main__":
    sys.exit(main())
