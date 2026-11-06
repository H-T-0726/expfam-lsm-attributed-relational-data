"""Curated (documentation) tables for the fair comparison, written from the dictionaries below.

model_equivalence_matrix.csv, math_code_correspondence.csv, implementation_differences.csv,
comparison_fairness_matrix.csv, experiment_protocol.csv.  Every FACT row cites
file:line or a paper equation.  Refuses to overwrite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import fair_protocol as fp  # noqa: E402

OUT = ROOT / fp.RESULTS_REL

MODEL_EQ = [
    ("prior of Z", "N(0, sigma_z^2 I), sigma_z^2 estimated (Sect.2.2, Eq.14)", "N(0, I), var_z fixed 1 + scale_Z", "same at sigma_z^2=1", "model", "FACT"),
    ("X | Z", "N(F z + mu_x, Sigma) (Eq.1)", "N(F z, Sigma) (no intercept)", "same at mu_x = 0", "model", "FACT"),
    ("Y | Z", "Bern(sigmoid(w0 + w z_i'z_j)) (Eq.2)", "same", "same", "model", "FACT"),
    ("pair counting", "{prod_{i!=j}}^{1/2} (Eq.16)", "prod over i<j", "same for symmetric Y (numeric diff 0)", "objective", "DERIVED+OBSERVED"),
    ("diagonal of Y", "y_ii = 1 by convention (Sect.2.2)", "unused", "no effect on likelihood (i != j only)", "model", "FACT"),
    ("Gaussian dispersion", "Sigma diagonal, estimated (Eq.12)", "Sigma diagonal, estimated", "same", "model", "FACT"),
    ("rotation", "not stated; penalty subtracts k(k-1)/2", "(Z,F)->(ZR,FR) invariant", "same (numeric diff 0)", "model", "DERIVED+OBSERVED"),
    ("normalisation of data", "not stated (MATLAB/Python z-score Z and X)", "canonical: none", "generator difference, not a model difference", "generator", "FACT"),
    ("complete-data log density", "ln p(Z)+ln p(X|Z)+ln p(Y|Z) (Eq.18)", "same", "equal at the same (Z, theta): diff 0", "objective", "OBSERVED"),
    ("E-step gradient (Y part)", "(1/2) sum_j (y_ij - s_ij) w z_j (Eq.23)", "w sum_j (y_ij - s_ij) z_j", "paper print has an extra 1/2; adopted = exact derivative of Eq.20 (FD)", "implementation", "DERIVED+OBSERVED"),
    ("E-step precision (Y part)", "(1/2) sum_j s(1-s) w^2 z_j z_j' (Eq.22)", "w^2 sum_j s(1-s) z_j z_j'", "same as above", "implementation", "DERIVED+OBSERVED"),
    ("overall", "-", "-", "MODEL: equivalent (mu_x=0, sigma_z^2=1); OBJECTIVE: equivalent; IMPLEMENTATION: not equivalent (1/2, mask, numerics)", "verdict", "DERIVED+OBSERVED"),
]

CORR = [
    ("initial Z, F, Sigma", "LatentStructuralModel.initialize_params(seed) (reproduction/src/model.py:63-130)", "same method inherited (em_runner.py: model.initialize_params(seed))", "identical for the same seed"),
    ("initial w0, w", "experiment_paper_2.py:186-190: logit(mean(Y)) incl. diagonal, w=0.5", "em_runner.py: logit(upper-triangle density), w=0.5", "adapter uses E's definition (documented)"),
    ("posterior target", "Laplace-conditional per node, Y term with 1/2 (model.py:213-345)", "same scheme, no extra 1/2, mask (model_dual_expfam_masked.py:93-140)", "different target for w != 0"),
    ("Newton gradient / precision", "model.py:276-283 (0.5 w), 337-353 (0.5 w^2), ridge 1e-6", "masked.py:93-140; consistent sigmoid", "FD: OLD rel err 0.33 / 0.093; E 2e-9 / 2e-8"),
    ("Newton step", "alpha 0.5, max 10 (experiment_paper_2.py:215)", "alpha 0.5 (em_runner newton_alpha), max 10", "same settings"),
    ("Laplace centre / covariance", "sample z_i ~ N(eta_i, A_i^-1) after the Newton steps", "same", "same scheme"),
    ("Gibbs order", "nodes i = 1..n sequentially, L draws chained", "same", "same"),
    ("L, iterations", "10, 30 in the original Exp.2 script; 5, 8 here (adapter)", "5, 8", "matched in E1"),
    ("scale_Z", "model.py:468-505 (joint over L)", "inherited", "same"),
    ("RNG", "default_rng(seed) for the E-step", "default_rng(seed) (em_runner)", "same seed; consumption order may differ"),
    ("F update", "model.py:506-547 (closed form)", "model_dual_expfam.py:208 (closed form)", "same formula"),
    ("Sigma update", "model.py:548-588", "model_dual_expfam.py:274", "same formula"),
    ("w0, w update", "Adam lr 0.01, 50 steps (model.py:589-760), unmasked", "Adam lr 0.01, 50 steps (masked.py:142-200), masked, /phi", "same optimiser; mask differs only when not all pairs are used"),
    ("true values used", "var_f scale only (model.py:101-104)", "none", "no leakage"),
]

IMPL_DIFF = [
    ("I1", "Y factor 1/2 in E-step score and curvature", "OLD", "model.py:283, 353", "targets a tempered conditional posterior (Y information halved)"),
    ("I2", "ridge 1e-6 on A_i", "OLD", "model.py:426, 450", "negligible"),
    ("I3", "sigmoid evaluation", "both", "OLD clips logits; E uses the stable branch", "only for extreme eta"),
    ("I4", "observation mask", "E", "masked.py:57-63", "none in E1 (all pairs)"),
    ("I5", "w0 initial value", "adapter", "experiment_paper_2.py:187-188 vs em_runner", "aligned in E1"),
    ("I6", "best iteration / best of 3 inits", "original Python Exp.2", "experiment_paper_2.py:234-245, 289-297", "not used by the adapter; recorded as PY_hist reference"),
    ("I7", "strict Q reads model.params['sigma']", "E evaluator", "eval_utils.calc_Q_dual_strict_exp", "latent hazard; the fair evaluator is independent"),
    ("I8", "MATLAB: gradient lacks w, alpha 0.01, diff not reset", "MATLAB", "calcEtaNewton.m:3-22, 43-49", "not run (no MATLAB); see PR #158"),
]

FAIR_ITEMS = ["observed data", "n, d", "generating model", "K_GEN", "K candidates", "seed", "initial values",
              "estimator", "optimisation budget", "observation mask", "criterion", "parameter penalty"]


def fairness():
    comps = {
        "E1 OLD vs E (same generator, K_GEN, dataset)": (
            ["match"] * 7 + ["differ", "match", "match", "match", "match"],
            "MATCHED_MODEL; MATCHED_CRITERION; ONE_FACTOR_COMPARISON (estimator implementation; the estimator differs in several internal details I1-I5, so 'one factor' = estimator as a whole)"),
        "E1 G_old vs G_canon (same estimator)": (
            ["differ", "match", "differ", "match", "match", "differ", "match", "match", "match", "match", "match", "match"],
            "MULTI_FACTOR_COMPARISON (generator family: F law, Z/X standardisation, w law all differ)"),
        "E1 K_GEN 5 vs 9": (["differ", "match", "match", "differ", "match", "differ", "match", "match", "match", "match", "match", "match"],
                            "MULTI_FACTOR_COMPARISON (different datasets; per-direction signal changes with K under both generators)"),
        "same state, criteria A-E": (["match"] * 10 + ["differ", "match"], "MATCHED_ESTIMATOR; ONE_FACTOR_COMPARISON (criterion)"),
        "E2 weak vs strong": (["differ", "match", "differ", "match", "match", "match", "match", "match", "match", "match", "match", "match"],
                              "MULTI_FACTOR_COMPARISON within signal (F and w together; same Z)"),
        "E2 n=75 vs 150": (["differ", "differ", "match", "match", "match", "differ", "match", "match", "match", "match", "match", "match"],
                           "ONE_FACTOR_COMPARISON in design (n); datasets differ by construction"),
        "E2 Gaussian vs Poisson X": (["differ", "match", "differ", "match", "match", "match", "match", "match", "match", "match", "match", "differ"],
                                     "MULTI_FACTOR_COMPARISON (family; expected Fisher information matched; Gaussian adds d variance parameters)"),
        "E3 train 80% vs 100%": (["match", "match", "match", "match", "match", "match", "match", "match", "match", "differ", "match", "match"],
                                 "ONE_FACTOR_COMPARISON (observation mask; holdout identical)"),
        "historical (paper Fig.3, Python repro, early ExpFam) vs canonical": (["differ"] * 12, "HISTORICAL_REFERENCE / NOT_DIRECTLY_COMPARABLE"),
    }
    rows = []
    for name, (vals, label) in comps.items():
        for item, v in zip(FAIR_ITEMS, vals):
            rows.append(dict(comparison=name, item=item, match=v,
                             must_match=("yes" if item not in ("observed data", "seed") else "depends"),
                             effect=("intended contrast or documented" if v == "differ" else ""), label=label))
    return pd.DataFrame(rows)


def main():
    def w(name, df):
        p = OUT / name
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        df.to_csv(p, index=False, lineterminator="\n")
    OUT.mkdir(parents=True, exist_ok=True)
    w("model_equivalence_matrix.csv", pd.DataFrame(MODEL_EQ, columns=["item", "paper", "expfam_gauss_bern", "verdict", "layer", "label"]))
    w("math_code_correspondence.csv", pd.DataFrame(CORR, columns=["step", "python_reproduction_OLD", "lineage_E", "comparison"]))
    w("implementation_differences.csv", pd.DataFrame(IMPL_DIFF, columns=["id", "difference", "where", "file_lines", "consequence"]))
    w("comparison_fairness_matrix.csv", fairness())
    proto = fp.protocol()
    rows = [dict(section=k, value=json.dumps(v, ensure_ascii=False)) for k, v in proto.items()]
    rows.append(dict(section="protocol_hash", value=fp.protocol_hash()))
    w("experiment_protocol.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
