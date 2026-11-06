"""Pre-registered protocol: n=75 old Python estimator vs lineage E (2026-11-06).

Single source of truth for every constant.  Committed BEFORE any new MCEM fit.
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

EXPERIMENT_ID = "n75_old_vs_expfam_controlled_20261106"
RESULTS_REL = "expfam/results/k_selection/" + EXPERIMENT_ID
STATE_ROOT = r"D:\tento\kennkyu_artifacts" + "\\" + EXPERIMENT_ID
PR160_REL = "expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106"
PR160_STATES = r"D:\tento\kennkyu_artifacts\fair_model_estimator_criterion_comparison_20261106"

FIT_BUDGET = {"B": 160, "C": 80, "D": 80, "reserve_technical": 20, "total_max": 340}

COMMON = {
    "n": 75, "d": 15, "K_GEN": 5, "K_CAND": list(range(1, 11)), "starts": [1, 2],
    "L": 5, "em_iterations": 8, "newton_max_iter": 10, "newton_alpha": 0.5, "adam_max_iter": 50,
    "estimators": {
        "OLD": "reproduction LatentStructuralModel through the PR #160 adapter run_fair_comparison.fit_old (code unchanged; E-step with the 1/2 Y factor of the printed Eq.22/23)",
        "E": "lineage-E run_fair_comparison.fit_E = em_runner_instrumented.run_em_instrumented, numerics consistent, on_nan=fail",
    },
    "shared_initial_state": "both: LatentStructuralModel.initialize_params(true_params=None, seed=model_seed) -> Z, F, sigma; w0 = logit(upper-triangle density of the TRAINING pairs), w = 0.5; E-step rng default_rng(model_seed). Verified per fit BEFORE fitting by hashing the initial Z, F, sigma, w0, w of both estimators (dataset_and_initialization_hashes.csv). RNG consumption order inside the E-step is implementation-specific and is NOT claimed equal.",
    "failure_policy": "no retry, no seed change; FAILED recorded; dataset INCOMPLETE for K_HAT",
    "threads": 1, "processes": 1, "mem_min_gb": 3.0,
    "selector": "mean over the two starts per candidate K; argmin over K_CAND = 1..10; exact ties -> smallest K; K_HAT only if every candidate K has both starts OK (criteria D/E also need Laplace OK); no best-start / best-iteration selection",
}

DATA = {
    "G_canon_n75": "nested subsample: the first 75 nodes (rows of X and Z, the 75x75 block of Y) of the PR #160 E1 G_canon n=150 dataset, generate_canonical_data(n=150, d=15, k=5, gaussian/bernoulli, f_scale=f_scale_for_row_norm(0.9,15,5), sigma_x_var=0.1, w0=-1, w=sqrt(6.75/5), seed=7610000+500+rep), rep 1..4. Nodes are iid given F, so this is an exact draw of the canonical model at n=75 with the SAME F, w0, w as the n=150 dataset of the same rep. Decided before any fit to make the n comparison as close to one-factor as possible (reps 1, 2 pair with PR #160 n=150 fits).",
    "G_old_n75": "reproduction data_generator.set_true_params(n=75, d=15, k=5, seed=1980+500+1000*(rep-1)), rep 1, 2 (MATLAB port; Z and X z-scored over the 75 nodes, so it cannot be a subsample of n=150; F and w differ from the n=150 dataset of the same rep because the RNG stream depends on n). Not a one-factor comparison with G_canon or with n=150.",
    "all_pairs": "train mask = all i != j (both B and C; OLD has no mask support and its code is not changed)",
    "mask_80": "Phase D: keep round(0.8 * 2775) = 2220 upper-triangle pairs drawn by default_rng([data_seed, 650000]).choice(without replacement), symmetric. The excluded 555 pairs are used as held-out pairs for the 80% condition ONLY (they are training data in the all-pairs condition).",
    "population_score": "leak-free predictive comparison usable for every condition: expected Bernoulli log score under the TRUE pair probabilities, mean over all upper pairs: sum p_true log p_hat + (1-p_true) log(1-p_hat), p_hat = MC mean over the final L samples (no observed Y used)",
}

def model_seed(g: int, rep: int, k: int, s: int) -> int:
    """g = 0 G_old, 1 G_canon.  Same seed for OLD and E; Phase D reuses the Phase B seed."""
    return 7800000 + 1000000 * g + 1000 * rep + 10 * k + s


STAGES = {
    "B": {"generator": "G_canon_n75", "replicates": [1, 2, 3, 4], "estimators": ["OLD", "E"], "mask": "all_pairs",
          "fits": 4 * 10 * 2 * 2},
    "C": {"generator": "G_old_n75", "replicates": [1, 2], "estimators": ["OLD", "E"], "mask": "all_pairs",
          "fits": 2 * 10 * 2 * 2},
    "D": {"generator": "G_canon_n75", "replicates": [1, 2, 3, 4], "estimators": ["E"], "mask": "mask_80",
          "fits": 4 * 10 * 2, "paired_with": "Phase B lineage-E fits (same data, same model seed); only the training mask (and hence the informed w0 init, which uses the training density) differs"},
}
TAG_MASK = 650000

REUSE = {
    "n150_pairs": "PR #160 E1 states (read-only): G_canon K_GEN=5 reps 1, 2 and G_old K_GEN=5 reps 1, 2, both estimators, K 1..10, starts 1, 2. Re-scored with the same evaluator code as the new states; scores must equal PR #160 criterion_scores.csv (abs diff < 1e-6) or the reuse is reported as failed.",
}

CRITERIA = {
    "A_CQ": "-2(Q_X+Q_Y+Q_Z) + p_K ln n (MC mean over the final L samples)",
    "B_CQ_minus_PZ": "-2(Q_X+Q_Y) + p_K ln n",
    "C_cond_last": "-2 ln p(X, Y_train | Z_last, theta) + p_K ln n",
    "D_cond_mode": "-2 ln p(X, Y_train | Z_hat, theta) + p_K ln n, Z_hat = joint Q2 mode",
    "E_CLap": "Laplace approximation of -2 ln p(X, Y_train | theta) + p_K ln n (N = n working convention)",
    "PY_type (OLD only, reference, NOT pooled with A-E)": "per start: -2 max over the 8 iterations of the plug-in ln p(Z,X,Y) at the last sample + ((k+1)d - k(k-1)/2) ln n; then the common start-mean selector (NOT the original min over 3 inits / 30 iterations)",
}
PENALTY = "p_K = K d - K(K-1)/2 + d (Gaussian-X); Eq.(26) differs by a K-independent constant"
PRIMARY = ["K_HAT (common evaluator, criteria A-E)", "exact / under / over per dataset",
           "slack_under = C(4) - C(5) (> 0 correct direction)", "slack_over = C(5) - C(6) (< 0 correct direction)",
           "agreement of K_HAT between OLD and E on the same dataset"]
SECONDARY = ["ZZ^T correlation over pairs (rotation invariant)", "subspace recovery: sum of squared canonical correlations between Z_hat (MC mean) and Z_true / K_GEN",
             "F F^T relative error (G_canon only; G_old X is z-scored)", "eta_Y relative RMSE vs truth (G_canon only for scale; G_old reported but flagged)",
             "X signal recovery: R^2 of Z_hat F_hat^T against the noiseless Z F^T (G_canon only)", "population Y log score (DATA.population_score)",
             "held-out Y log score on the excluded pairs (Phase D 80% condition only)", "w_hat, w0_hat", "runtime, failures"]
ESTEP = {
    "purpose": "Phase F, deterministic, no fit: same theta, Z, X, Y; per node i (others fixed)",
    "states": "lineage-E states at K = K_GEN = 5, start 1: Phase B reps 1-4 and Phase C reps 1-2 (n=75); PR #160 E1 G_canon / G_old K5 reps 1-2 (n=150)",
    "quantities": "conditional log density of each implementation's own target (OLD: log p(x_i|z_i) + 1/2 sum_j log p(y_ij|z_i,z_j) + log p(z_i); E: same without 1/2), gradient and negative Hessian vs finite differences of that target, conditional mode (Newton alpha 1, 50 steps), Laplace covariance trace ratio OLD/E, X-part and Y-part of the precision trace, MC sample variance of the saved Z samples per estimator (from the fitted states at K=5)",
    "not_claimed": "the single causal effect of the 1/2 factor (other implementation differences exist; a 1/2-only ablation would be a new experiment)",
}

STOP = ["mathematical mismatch invalidating the comparison", "data leakage", "seed-policy violation", "corrupted CSV",
        "implicit generator clipping", "initial states of OLD and E differ for the same seed", "conditions cannot be fixed",
        "unsafe state handling", "technical defect making the main purpose uninterpretable"]
NOT_STOP = ["K selection failure", "worse accuracy", "solver warnings", "MC variation", "OLD and E disagree"]


def protocol() -> dict:
    return {"experiment_id": EXPERIMENT_ID, "budget": FIT_BUDGET, "common": COMMON, "data": DATA,
            "model_seed": "7800000 + 1000000*g + 1000*rep + 10*k + start", "stages": STAGES, "tag_mask": TAG_MASK,
            "reuse": REUSE, "criteria": CRITERIA, "penalty": PENALTY, "primary": PRIMARY, "secondary": SECONDARY,
            "estep": ESTEP, "stop": STOP, "not_stop": NOT_STOP}


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    assert sum(v["fits"] for v in STAGES.values()) == 320 <= FIT_BUDGET["total_max"]
    print(protocol_hash())
