"""Pre-registered protocol: OLD vs lineage E on the PR #160 E2 FAILED weak-signal data, all pairs (2026-11-07).

Single source of truth for every constant.  Committed BEFORE any new MCEM fit.

Purpose: PR #160 E2 (lineage E, Gaussian-X / Bernoulli-Y, n=75, weak signal, 80%
training pairs) gave C_Q exact 0/4.  PR #161 used a strong-signal n=75 cell
(X SNR 9) where nothing failed and could not answer whether the failure is due
to the estimator.  Here the SAME four weak datasets of PR #160 E2 are reused
(no new data), with all relation pairs observed so that the old estimator
(no mask support, code unchanged) can be run on them.

Lineage E is a prototype; the old estimator (reproduction/src/model.py) is the
0.5 series (KI-001: "the Newton direction cannot be asserted to be correct as a
whole").  Nothing here is manuscript evidence.  C_Q is a Q-based complete-data
/ ICL-type criterion, not Schwarz BIC; C_Lap is a Laplace approximation, not the
exact marginal likelihood.  K_GEN is the generating dimension; K_GEN = K0 is not
assumed.
"""
from __future__ import annotations

import hashlib
import json

EXPERIMENT_ID = "weak_n75_old_vs_expfam_allpairs_20261107"
RESULTS_REL = "expfam/results/k_selection/" + EXPERIMENT_ID
STATE_ROOT = r"D:\tento\kennkyu_artifacts" + "\\" + EXPERIMENT_ID
PR160_REL = "expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106"
PR160_STATES = r"D:\tento\kennkyu_artifacts\fair_model_estimator_criterion_comparison_20261106"

FIT_BUDGET = {"W": 112, "reserve_technical": 0, "total_max": 112}

COMMON = {
    "n": 75, "d": 15, "K_GEN": 5, "K_CAND": list(range(1, 8)), "starts": [1, 2], "replicates": [1, 2, 3, 4],
    "L": 5, "em_iterations": 8, "newton_max_iter": 10, "newton_alpha": 0.5, "adam_max_iter": 50,
    "estimators": {
        "OLD": "reproduction LatentStructuralModel through the PR #160 adapter run_fair_comparison.fit_old (code unchanged; E-step with the 1/2 Y factor of the printed Eq.22/23)",
        "E": "lineage-E run_fair_comparison.fit_E = em_runner_instrumented.run_em_instrumented, numerics consistent, on_nan=fail, train_mask = all pairs",
    },
    "model_seed": "PR #160 E2 rule hist.model_seed(5, 75, rep, k, start) (= the seed of the PR #160 E2 80% fit of the same rep, K, start); SAME seed for OLD and E",
    "shared_initial_state": "both: LatentStructuralModel.initialize_params(true_params=None, seed=model_seed) -> Z, F, sigma; w0 = logit(upper-triangle density of ALL pairs), w = 0.5; E-step rng default_rng(model_seed). Verified per fit BEFORE fitting by hashing the initial Z, F, sigma, w0, w of both estimators; any difference -> STOP. RNG consumption order inside the E-step is implementation-specific and is NOT claimed equal.",
    "failure_policy": "no retry, no seed change, no extra fits; FAILED recorded; dataset INCOMPLETE for K_HAT",
    "threads": 1, "processes": 1, "mem_min_gb": 3.0,
    "selector": "mean over the two starts per candidate K; argmin over K_CAND = 1..7; exact ties -> smallest K; K_HAT only if every candidate K has both starts OK (criteria D/E also need Laplace OK); no best-start / best-iteration selection (analyze_fair_comparison.select, unchanged)",
}

DATA = {
    "source": "run_fair_comparison.e2_data(75, rep, 'weak', 'gaussian'), rep 1..4 = the PR #160 E2 'gaussian | n=75 | weak' datasets (PR #155 condition A: F_A mean row norm^2 0.5, w = sqrt(0.6), w0 = -1; Gaussian-X noise s_l^2 = exp(-||f_l||^2/2), default_rng([data_seed, 630000]); data_seed 860751..860754)",
    "identity_check": "X, Y sha256 must equal x_sha256 / y_sha256 of every PR #160 E2 gaussian_weak_n75 fit JSON of the same rep; otherwise STOP (no substitute data)",
    "train": "all pairs i != j (the PR #155 20% test pairs become training pairs; therefore NO held-out Y score in this condition - the excluded pairs of PR #160 are not reused as a test set)",
    "population_score": "leak-free predictive comparison usable for both conditions: expected Bernoulli log score under the TRUE pair probabilities over all upper pairs (no observed Y used)",
    "truth_scale": "canonical generator (X = Z F^T + noise, no z-scoring): F F^T error and X-signal R^2 are defined",
}

STAGES = {"W": {"estimators": ["OLD", "E"], "mask": "all_pairs", "fits": 4 * 7 * 2 * 2}}

REUSE = {
    "P160_E2_80": "PR #160 E2 gaussian_weak_n75 lineage-E states (56, read-only, 80% training mask, held-out 20%). Re-scored with the same evaluator code; criteria must equal PR #160 criterion_scores.csv (abs diff < 1e-6) or the reuse is reported as failed.",
}

COMPARISONS = {
    "OLD_vs_E_allpairs": "ONE_FACTOR_COMPARISON in the sense of 'estimator implementation' (same data, initial state, seed, budget, evaluator); the estimator factor bundles the 1/2 Y factor with other implementation differences (ridge, sigmoid evaluation, RNG use) - not a 1/2-only comparison",
    "E_80_vs_E_allpairs": "MULTI_FACTOR_COMPARISON: the training mask changes AND the informed w0 init (training density) changes AND the criterion is evaluated on a different Y_train; same X, Y, Z, F, seed, initial Z/F/sigma/w. Not called one-factor. Absolute criterion values are not comparable across conditions (different number of Y terms); only K_HAT and slacks are compared, slacks with this caveat.",
}

CRITERIA = {
    "A_CQ": "-2(Q_X+Q_Y+Q_Z) + p_K ln n (MC mean over the final L samples)  [complete-data type, PRIMARY]",
    "B_CQ_minus_PZ": "-2(Q_X+Q_Y) + p_K ln n  [complete-data type without Q_Z]",
    "C_cond_last": "-2 ln p(X, Y_train | Z_last, theta) + p_K ln n  [conditional-likelihood type]",
    "D_cond_mode": "-2 ln p(X, Y_train | Z_hat, theta) + p_K ln n, Z_hat = joint Q2 mode  [conditional-likelihood type]",
    "E_CLap": "Laplace approximation of -2 ln p(X, Y_train | theta) + p_K ln n (N = n working convention)  [SECONDARY]",
    "PY_type (OLD only, reference)": "per start: -2 max over the 8 iterations of the plug-in ln p(Z,X,Y) + ((k+1)d - k(k-1)/2) ln n; common start-mean selector",
}
PENALTY = "p_K = K d - K(K-1)/2 + d (Gaussian-X)"
PRIMARY = ["K_HAT under C_Q per dataset and estimator", "slack_under = C(4) - C(5) (> 0 correct)", "slack_over = C(5) - C(6) (< 0 correct)",
           "agreement of K_HAT between OLD and E on the same dataset"]
SECONDARY = ["K_HAT under C_Lap and criteria B, C, D", "ZZ^T correlation", "subspace recovery (cannot penalise extra dimensions)",
             "eta_Y relative RMSE", "F F^T relative error", "X-signal R^2", "population Y log score", "w_hat, w0_hat",
             "E-step mode / covariance (same state)", "runtime, failures"]

CASES = {
    "rule": "per dataset under C_Q: OLD label x E label (exact / under / over / INCOMPLETE). Overall case: A if every complete dataset is (OLD exact, E under); B if every complete dataset is (OLD under, E under); C if every complete dataset is (OLD exact, E exact); otherwise D (mixed) with the per-dataset table. The same rule is reported for C_Lap as secondary. Fixed before any fit.",
    "A": "OLD exact, E under: evidence that the estimator implementation affects K selection on this data (not attributable to the 1/2 factor alone)",
    "B": "both under: the old estimator also fails on this weak data with all pairs (not attributable to information shortage alone)",
    "C": "both exact: failure not reproduced with all pairs; the 80% failure differs by mask and w0 init (multi-factor)",
    "D": "mixed: per-dataset slack differences OLD - E, initial states and recovery; 'stable' only if the sign is the same in 4/4 datasets (no significance claim with 4 datasets)",
}

ESTEP = {
    "purpose": "deterministic, no fit: same theta, Z, X, Y; per node i (others fixed), OLD target (1/2 on Y) vs E target",
    "states": "K = 5, start 1, all pairs: lineage-E states reps 1-4 and OLD states reps 1-4 (each implementation's E-step evaluated at both fitted states)",
    "not_claimed": "the single causal effect of the 1/2 factor (a 1/2-only ablation would be a new experiment, only proposed)",
}

STOP = ["the PR #160 weak data cannot be reproduced exactly (hash mismatch)", "initial states of OLD and E differ for the same seed",
        "mathematical mismatch invalidating the comparison", "data leakage", "seed-policy violation", "corrupted CSV",
        "unsafe state handling", "technical defect making the main purpose uninterpretable"]
NOT_STOP = ["K selection failure", "worse accuracy", "solver warnings", "MC variation", "OLD and E disagree", "Laplace failure on a state (recorded, INCOMPLETE)"]


def protocol() -> dict:
    return {"experiment_id": EXPERIMENT_ID, "budget": FIT_BUDGET, "common": COMMON, "data": DATA, "stages": STAGES,
            "reuse": REUSE, "comparisons": COMPARISONS, "criteria": CRITERIA, "penalty": PENALTY, "primary": PRIMARY,
            "secondary": SECONDARY, "cases": CASES, "estep": ESTEP, "stop": STOP, "not_stop": NOT_STOP}


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    assert sum(v["fits"] for v in STAGES.values()) == 112 <= FIT_BUDGET["total_max"]
    print(protocol_hash())
