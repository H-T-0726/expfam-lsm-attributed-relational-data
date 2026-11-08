"""Pre-registered protocol: K-selection criteria under conditions close to the paper's Experiment 2 / Fig. 3 (2026-11-09).

Committed BEFORE any fit.  Lineage E prototype; nothing here is manuscript evidence.
This is NOT a reproduction of the paper's Experiment 2: the estimator is lineage E
(consistent numerics, no 1/2 in the Y E-step, sigma_z^2 fixed to 1 + scale_Z, no mu_x),
and several paper conditions are unknown (see CONDITIONS).

Naming: C_Q is a Q-based complete-data / ICL-type criterion, NOT Schwarz BIC; C_Lap is a
Laplace approximation, NOT an exact marginal likelihood; C_ELBO uses q = Laplace Gaussian
(existing variational lower bound), NOT a new BIC.  cond_MC / cond_last / cond_mode are
conditional-likelihood criteria inspired by the paper's Eq.(16)/(26); the paper does not
specify the Z evaluation point, so they are NOT the paper's BIC.
"""
from __future__ import annotations

import hashlib
import json

EXPERIMENT_ID = "paper_fig3_condition_criteria_20261109"
RESULTS_REL = "expfam/results/k_selection/" + EXPERIMENT_ID
STATE_ROOT = r"D:\tento\kennkyu_artifacts" + "\\" + EXPERIMENT_ID

N, D = 150, 15
K_GEN = [1, 3, 5, 7, 9]
K_CAND = list(range(1, 11))
TRIALS = list(range(1, 11))
L = 10
EM_ITERATIONS = 10
FITS = len(K_GEN) * len(K_CAND) * len(TRIALS)   # 500

GENERATOR = {
    "name": "G_paper_text",
    "rule": ("rng = numpy.random.default_rng(data_seed); Z = rng.standard_normal((n, K_GEN)) (no z-score); "
             "F = rng.normal(0, sqrt(10), (d, K_GEN)) (10 read as a VARIANCE; no row normalisation); "
             "w0 = rng.standard_normal(); w = 3 * rng.standard_normal(); "
             "X = Z F' + sqrt(0.1) * rng.standard_normal((n, d)) (mu_x = 0; no z-score); "
             "U = rng.random((n, n)); Y_ij = 1[U_ij < sigmoid(w0 + w z_i'z_j)] for i<j, symmetric, zero diagonal; all pairs observed"),
    "data_seed": "9260000 + 100 * K_GEN  (one dataset per K_GEN, as the paper shows one curve per k*)",
}
TRIAL_DEFINITION = ("paper Sect.4.1: 'We execute 10 trials to implement the algorithm with different initial values' -> "
                    "trial t = an independent initial value and MC stream on the SAME dataset; "
                    "model_seed = 9300000 + 10000*K_GEN + 100*K + t (initialize_params seed and E-step rng)")
ESTIMATOR = ("expfam/src/experimental/em_runner_instrumented.run_em_instrumented(family_x='gaussian', family_y='bernoulli', "
             f"k=K, L={L}, max_iter={EM_ITERATIONS}, train_mask=all pairs, numerics_mode='consistent', on_nan='fail') "
             "(lineage E, informed init, scale_Z; code unchanged). One fit per (K_GEN, K, trial).")

CONDITIONS = [
    # item, paper, this study, class
    ("n, d", "150, 15 (Table I)", "150, 15", "EXACT"),
    ("K_GEN, candidate K", "{1,3,5,7,9}, 1..10 (Table I)", "same", "EXACT"),
    ("L (MC samples), EM iterations", "10, 10 (Table I)", "10, 10", "EXACT"),
    ("trials", "10, 'different initial values' (Sect.4.1)", "10 initial values / MC streams on one dataset per K_GEN", "EXACT (count) / ASSUMED (same data across trials)"),
    ("Z", "z_i ~ N(0, sigma_z^2 I), sigma_z^2 = 1 (Step 1)", "N(0, I), no normalisation", "EXACT"),
    ("F", "each entry ~ N(0, 10) (Step 3)", "N(0, 10) with 10 = variance", "ASSUMED (variance vs sd not stated)"),
    ("Sigma, mu_x", "diag(0.1), 0 (Step 3)", "same", "EXACT"),
    ("w0, w", "'random values' (Step 2)", "w0 ~ N(0,1), w = 3 N(0,1) (MATLAB setTrueParams.m L30-31 / reproduction data_generator)", "ASSUMED (rule from existing code, not from the paper)"),
    ("Y observation", "all pairs (no mask mentioned)", "all pairs", "EXACT (no mask stated)"),
    ("data normalisation", "none stated for synthetic data", "none", "EXACT (none stated)"),
    ("model: mu_x", "estimated (Eq.(11))", "not in lineage E (X has no intercept)", "DIFFERENT"),
    ("model: sigma_z^2", "estimated (Eq.(14))", "fixed 1 + scale_Z after each E-step", "DIFFERENT"),
    ("E-step Y factor", "1/2 in Eq.(22)/(23)", "no extra 1/2 (adopted formula)", "DIFFERENT"),
    ("E-step Newton", "Newton (step not stated)", "damped Newton alpha 0.5, max 10 steps (em_runner)", "IMPLEMENTED (lineage E)"),
    ("w0, w update", "Adam (Eq.(24)/(25))", "lineage E M-step (calc_w0/calc_w)", "IMPLEMENTED (lineage E)"),
    ("initial values", "not stated", "em_runner informed init (initialize_params(seed), F*0.2 etc.)", "IMPLEMENTED (lineage E) / UNKNOWN (paper)"),
    ("BIC Z evaluation point", "not stated", "six criteria evaluated on the same fitted state", "UNKNOWN (paper)"),
    ("BIC trial aggregation (Exp.2)", "not stated (Exp.1,3,4: mean over trials per iteration; best = min)", "per-trial selection (primary) + mean / min over trials (secondary)", "UNKNOWN (paper)"),
    ("penalty", "((K+1)d - K(K-1)/2) ln n (Eq.(26))", "Kd - K(K-1)/2 + d (Gaussian-X) = identical", "EXACT"),
]

CRITERIA = ["CQ", "cond_MC", "cond_last", "cond_mode", "CLap", "CELBO"]
ELBO_M = 2048          # fixed: same budget as criterion_impact_validation_20261108 (n=150 there gave MCSE ~3.5-5)
ELBO_BATCHES = 8
ELBO_SEED = "[20261109, 3, 1 (gaussian), 3 (paper), 150, K_GEN, K, trial] + [batch]"

SELECTION = {
    "primary_per_trial": "for each (K_GEN, trial, criterion): argmin over K=1..10 of the criterion of that trial's fits; ties within 1e-12 -> smallest K; INCOMPLETE if any K is missing/non-finite for that trial (no imputation)",
    "secondary_mean_over_trials": "per K: mean over the 10 trials (all 10 finite required, else INCOMPLETE); argmin over K",
    "secondary_min_over_trials": "per K: min over finite trials (best trial, Python-reproduction style); argmin over K; reported as a reference rule only",
    "mc_uncertain": "C_ELBO per-trial selection flagged MC_UNCERTAIN when runner-up minus best < 2*sqrt(se_best^2+se_second^2)",
}
PRIMARY = ["per (K_GEN, criterion): distribution of the 10 per-trial K_HAT, exact / under / over counts (correct = K_HAT == K_GEN; K_GEN = K0 NOT assumed)",
           "per-trial agreement CQ vs CLap, CLap vs CELBO, CQ vs cond_*",
           "decomposition per (K_GEN, trial) at K_GEN-1->K_GEN and K_GEN->K_GEN+1: slack_Q - slack_Lap = -[d2H + dtermA + d2R3], slack_ELBO - slack_Lap = -d2R3"]
SECONDARY = ["score curves K=1..10", "aggregated selections", "convergence diagnostic (Q_strict per iteration, last-iteration relative change; descriptive, not a gate)",
             "Laplace NOT_STATIONARY / Hessian status by K", "Y edge density and true w0, w per dataset", "comparison with paper Fig.3 digitised curves (PR #158)"]
FAILURE = "no retry, no seed change, no extra iterations; FAILED fit -> trial INCOMPLETE for every criterion"
STOP = ["generator or candidate-K rule mismatch", "seed collision", "evaluator error", "overwrite", "non-finite primary quantities", "unsafe resources"]
NOT_STOP = ["selection failure", "disagreement with the paper", "INCOMPLETE", "saturated Y"]


def protocol() -> dict:
    return {"experiment_id": EXPERIMENT_ID, "n": N, "d": D, "K_GEN": K_GEN, "K_CAND": K_CAND, "trials": TRIALS,
            "L": L, "em_iterations": EM_ITERATIONS, "fits": FITS, "generator": GENERATOR, "trial_definition": TRIAL_DEFINITION,
            "estimator": ESTIMATOR, "conditions": CONDITIONS, "criteria": CRITERIA, "elbo_M": ELBO_M,
            "elbo_batches": ELBO_BATCHES, "elbo_seed": ELBO_SEED, "selection": SELECTION, "primary": PRIMARY,
            "secondary": SECONDARY, "failure": FAILURE, "stop": STOP, "not_stop": NOT_STOP}


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def data_seed(kgen: int) -> int:
    return 9260000 + 100 * kgen


def model_seed(kgen: int, k: int, t: int) -> int:
    return 9300000 + 10000 * kgen + 100 * k + t


if __name__ == "__main__":
    print(protocol_hash())
