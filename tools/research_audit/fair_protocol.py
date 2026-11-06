"""Pre-registered protocol for the fair model / estimator / criterion comparison (2026-11-06).

Single source of truth for every experiment constant.  Written and committed
BEFORE any new MCEM fit.  Lineage E prototype; nothing here is manuscript
evidence.  C_Q is not Schwarz BIC; C_Lap is not an exact marginal likelihood.
"""
from __future__ import annotations

import hashlib
import json
import math

EXPERIMENT_ID = "fair_model_estimator_criterion_comparison_20261106"
RESULTS_REL = "expfam/results/k_selection/" + EXPERIMENT_ID
STATE_ROOT = r"D:\tento\kennkyu_artifacts" + "\\" + EXPERIMENT_ID

FIT_BUDGET = {"E1": 320, "E2": 448, "E3": 112, "reserve_technical": 20, "total_max": 900}

COMMON = {
    "L": 5, "em_iterations": 8, "newton_max_iter": 10, "newton_alpha": 0.5,
    "adam_max_iter": 50, "starts": [1, 2], "numerics_mode_E": "consistent",
    "failure_policy": "fail fast: no retry, no seed change, no Z_prev substitution; failed fit -> FAILED, dataset INCOMPLETE for K_HAT",
    "threads": 1, "processes": 1, "mem_min_gb": 3.0,
    "selector": "mean over the two starts per candidate K; argmin; ties within 1e-12 -> smallest K; K_HAT only if every candidate K has both starts OK",
}

# ---------------------------------------------------------------- E1
E1 = {
    "purpose": "same data, same initial state and MC seed: historical Python estimator (reproduction/src/model.py) vs lineage-E consistent estimator; Gaussian-X / Bernoulli-Y, all pairs observed",
    "estimators": {"OLD": "reproduction LatentStructuralModel via adapter (code unchanged): E-step with the 1/2 Y factor, alpha 0.5, 10 Newton steps, ridge 1e-6; Adam w0/w",
                   "E": "expfam/src/experimental em_runner_instrumented.run_em_instrumented, numerics consistent, on_nan=fail, train_mask = all pairs"},
    "generators": {
        "G_old": "reproduction/src/data_generator.set_true_params(n=150, d=15, k=K_GEN, seed=1980+100*K_GEN+1000*(rep-1)) (MATLAB port; Z and X z-scored; F rows norm sqrt(0.9); w=3*randn)",
        "G_canon": "data_generator_canonical.generate_canonical_data(n=150, d=15, k=K_GEN, family_x='gaussian', family_y='bernoulli', f_scale=f_scale_for_row_norm(0.9, d=15, k=K_GEN), sigma_x_var=0.1, w0=-1.0, w=sqrt(6.75/K_GEN), seed=7610000+100*K_GEN+rep)",
    },
    "n": 150, "d": 15, "K_GEN": [5, 9], "replicates": [1, 2], "K_CAND": list(range(1, 11)),
    "model_seed": "7700000 + 1000000*g + 10000*K_GEN + 1000*rep + 10*k + start  (g=0 G_old, g=1 G_canon); SAME seed for OLD and E",
    "shared_initial_state": "both call LatentStructuralModel.initialize_params(true_params=None, seed=model_seed) (identical Z, F, sigma); w0 = logit(upper-triangle edge density), w = 0.5 (E's informed init; the original reproduction used the n x n mean incl. the diagonal - the adapter aligns it, documented); E-step rng = default_rng(model_seed) in both",
    "deviation_from_request": "request: 4 datasets per condition; 2 estimators x 2 generators x 2 K_GEN x 4 x 10 x 2 = 640 > budget 320. Fixed BEFORE any fit: 2 datasets per condition (reps 1, 2), every other factor kept -> 320 fits.",
    "fits": 2 * 2 * 2 * 2 * 10 * 2,
    "historical_reference_not_counted": "#139 paper-type lineage-E fits on G_old reps 1-5 (own init) - HISTORICAL_REFERENCE, not matched-init",
}

# ---------------------------------------------------------------- E2
TAG_GX = 630000          # Gaussian-X noise stream: default_rng([data_seed, TAG_GX])
E2 = {
    "purpose": "fixed estimator (lineage E) and criteria; generator conditions vary: X family (Gaussian vs Poisson), n (75, 150), signal (weak = PR #155 A, strong = PR #155 D)",
    "data": "run_k_signal_strength.build_all(n, rep): same Z, F, Y, train mask (80% pairs, PR #155). Weak: F_A (mean row norm^2 0.5), Y_A (w^2 = 0.6). Strong: F_B = sqrt(1.8) F_A (row norm^2 0.9), Y_C (w^2 = 1.08, w0 recalibrated to the same population density) - F and w change together (MULTI_FACTOR within the signal factor).",
    "gaussian_x": "X_G = Z F^T + eps, eps_il ~ N(0, s_l^2), s_l^2 = exp(-||f_l||^2 / 2) so that E_z[sum_l f_l f_l^T / s_l^2] = E_z[sum_l lambda_il f_l f_l^T] (expected conditional Fisher information matched to Poisson-X column by column; fixed before any fit). Same Z, F, Y, mask as the Poisson cell. Still a condition-group comparison (support, variance function, curvature differ).",
    "n": [75, 150], "replicates": [1, 2, 3, 4], "K_GEN": 5, "K_CAND": list(range(1, 8)),
    "model_seed": "hist.model_seed(5, n, rep, k, start) (PR #155 rule), same for Gaussian and Poisson cells",
    "reuse": {"Poisson weak n=75": "A1 EMC checkpoint-8 states (PR #155 A)", "Poisson weak n=150": "A1 EMC states", "Poisson strong n=75": "PR #155 fit_states_n75 D"},
    "reuse_gate": "refit 3 reused fits (A n=75, D n=75, A n=150; rep 1, K=5, start 1) and require state equality (rtol 1e-9, atol 1e-6) before counting reuse; 3 fits from the technical reserve",
    "new_fits": {"Gaussian (2 n x 2 signal x 4 x 7 x 2)": 224, "Poisson strong n=150 (4 x 7 x 2)": 56},
    "fits": 280,
}

# ---------------------------------------------------------------- E3
TAG_MASK = 640000        # thinning stream: default_rng([data_seed, TAG_MASK])
E3 = {
    "purpose": "same canonical Poisson-X / Bernoulli-Y weak data (PR #155 A, n=75); only the observed relation pairs change",
    "holdout": "the PR #155 test mask (20% of pairs) is excluded from both conditions and used for held-out prediction",
    "conditions": {"100% of training-eligible pairs": "PR #155 A train mask (reuse A1 states)",
                   "80% of training-eligible pairs": "keep a uniformly random 80% of the upper-triangle training pairs (rounded), symmetric, drawn with default_rng([data_seed, TAG_MASK])"},
    "replicates": [1, 2, 3, 4], "K_CAND": list(range(1, 8)), "model_seed": "hist.model_seed(5, 75, rep, k, start)",
    "fits": 56,
}

# ---------------------------------------------------------------- criteria and metrics
CRITERIA = {
    "A_CQ": "-2(Q_X+Q_Y+Q_Z) + p_K ln n (Q2 MC mean over the final L samples)",
    "B_CQ_minus_PZ": "-2(Q_X+Q_Y) + p_K ln n (Q1 MC mean)",
    "C_cond_last": "-2 ln p(X, Y_train | Z_last, theta) + p_K ln n",
    "D_cond_mode": "-2 ln p(X, Y_train | Z_hat, theta) + p_K ln n, Z_hat = joint (Q2) mode",
    "E_CLap": "Laplace approximation of -2 ln p(X, Y_train | theta) + p_K ln n (N = n working convention)",
    "PY_hist (E1 OLD only, reference)": "Python reproduction rule: -2 max over iterations of ln p(Z,X,Y | last Z) + ((k+1)d - k(k-1)/2) ln n",
}
PENALTY = "p_K = K d - K(K-1)/2 (+ d for Gaussian-X); Eq.(26) differs only by the constant d for Poisson-X"
PRIMARY = ["K_HAT over all candidate K", "exact / under / over counts", "slack C(K_GEN-1) - C(K_GEN)", "slack C(K_GEN) - C(K_GEN+1)"]
SECONDARY = ["X reconstruction (in-sample)", "held-out Y log score (E2, E3)", "Gram FF^T recovery (canonical truth)",
             "eta_Y recovery (corr of w z_i'z_j with truth over pairs)", "numerical failures", "runtime"]

LAPLACE_IS = {
    "purpose": "fixed-theta check of the Laplace approximation on EXISTING states (no fit)",
    "states": "canonical A1 n-sweep K_GEN=5, n in {75, 150}, reps 1-8, K in {4, 5, 6}, start 1 (Z_hat saved)",
    "method": "laplace_importance.importance_correction with q = N(Z_hat, H^-1), 8 batches x 512 samples, batch seed 9950000 + 1000*n + 100*rep + 10*K + b",
    "report": "c_K, batch sd, ESS fraction, max normalised weight; correction is NOT treated as the true marginal likelihood when weights degenerate (ESS fraction < 0.01)",
}

STOP = ["mathematical mismatch that invalidates a comparison", "data leakage", "seed-policy violation",
        "corrupted CSV", "implicit clipping in a generator", "conditions cannot be fixed", "unsafe state handling",
        "technical defect that makes the main purpose uninterpretable"]
NOT_STOP = ["K selection failure", "worse accuracy", "solver warnings", "MC variation"]


def protocol() -> dict:
    return {"experiment_id": EXPERIMENT_ID, "budget": FIT_BUDGET, "common": COMMON, "E1": E1, "E2": E2, "E3": E3,
            "criteria": CRITERIA, "penalty": PENALTY, "primary": PRIMARY, "secondary": SECONDARY,
            "laplace_is": LAPLACE_IS, "stop": STOP, "not_stop": NOT_STOP,
            "tags": {"TAG_GX": TAG_GX, "TAG_MASK": TAG_MASK}}


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def e1_seed(g: int, kgen: int, rep: int, k: int, s: int) -> int:
    return 7700000 + 1000000 * g + 10000 * kgen + 1000 * rep + 10 * k + s


def e1_canon_w(kgen: int) -> float:
    return math.sqrt(6.75 / kgen)


if __name__ == "__main__":
    print(json.dumps(protocol(), indent=1)[:2000])
    print(protocol_hash())
    assert E1["fits"] + E2["fits"] + E3["fits"] + 3 <= FIT_BUDGET["total_max"]
