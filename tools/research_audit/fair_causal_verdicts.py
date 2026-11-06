"""causal_hypothesis_verdicts.csv: candidate -> predicted phenomenon -> falsifying comparison -> result -> verdict.

The result column is computed from the analysis CSVs (no hand-entered numbers); the candidate, prediction,
comparison, fairness label and verdict are fixed text written after the results were seen (INTERPRETATION).  Verdict vocabulary:
SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED / REJECTED / INCONCLUSIVE / UNRESOLVED.
Refuses to overwrite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import fair_protocol as fp  # noqa: E402

RES = ROOT / fp.RESULTS_REL


def main():
    out = RES / "causal_hypothesis_verdicts.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    s = pd.read_csv(RES / "success_failure_conditions.csv")
    ag = pd.read_csv(RES / "estimator_agreement_e1.csv")
    eq = pd.read_csv(RES / "equivalence_numeric_checks.csv")
    es = pd.read_csv(RES / "estep_same_state.csv")
    li = pd.read_csv(RES / "laplace_is_check.csv")
    hk = pd.read_csv(RES / "metrics_at_k_hat.csv")
    pm = pd.read_csv(RES / "predictive_metrics.csv")

    def cnt(exp, cell, crit):
        r = s[(s.experiment == exp) & (s.cell == cell) & (s.criterion == crit)].iloc[0]
        comp = r.datasets - r.incomplete
        return f"{r.exact}/{comp} exact (under {r.under}, over {r.over}, incomplete {r.incomplete})"

    def e2(fam, n, sig, crit):
        return cnt("E2", f"{fam} | n={n} | {sig}", crit)

    rows = []

    def add(cid, cand, pred, comp, label, result, verdict):
        rows.append(dict(id=cid, candidate=cand, predicted_phenomenon=pred, falsifying_comparison=comp,
                         fairness_label=label, result=result, verdict=verdict))

    # H1 model
    mod = eq[eq.layer == "model"]
    eqv = (f"{int(mod['pass'].sum())}/{len(mod)} model-layer identity checks pass, max |diff| {mod.value.abs().max():.3g}; "
           f"all checks: {int(eq['pass'].sum())}/{len(eq)} pass (see equivalence_numeric_checks.csv; OLD E-step FD errors are implementation-layer)")
    add("H1", "paper model and ExpFam Gaussian-X/Bernoulli-Y are different models",
        "complete-data log densities differ at the same (Z, theta)",
        "paper formulae vs ExpFam evaluator on fixed (Z, theta); rotation; pair counting",
        "MATCHED_MODEL (numeric identity check)", eqv,
        "REJECTED (equivalent at mu_x = 0, sigma_z^2 = 1; implementation differs)")

    # H2 estimator
    a = ag.groupby("criterion").same_k_hat.agg(["sum", "count"])
    res2 = ("same K_HAT OLD vs E: " + ", ".join(f"{c} {int(a.loc[c, 'sum'])}/{int(a.loc[c, 'count'])}" for c in a.index)
            + "; C_Q exact: " + ", ".join(
                f"{e} {int(s[(s.experiment == 'E1') & s.cell.str.endswith('| ' + e) & (s.criterion == 'A_CQ')].exact.sum())}/"
                f"{int(s[(s.experiment == 'E1') & s.cell.str.endswith('| ' + e) & (s.criterion == 'A_CQ')].datasets.sum())}"
                for e in ("OLD", "E")) + f"; same-state E-step: median covariance trace ratio OLD/E "
            f"{es.median_cov_trace_ratio_old_over_E.min():.2f}-{es.median_cov_trace_ratio_old_over_E.max():.2f}, "
            f"median relative mode difference {es.median_rel_mode_diff.min():.3f}-{es.median_rel_mode_diff.max():.3f}")
    add("H2", "the old Python estimator (1/2 on the Y score/curvature) is what made the historical K selection succeed",
        "with matched data/init/seed/budget, OLD selects K_GEN with C_Q while E does not",
        "E1 OLD vs E (n=150, Gaussian-X, all pairs, K_GEN 5 and 9, two generators)",
        "ONE_FACTOR_COMPARISON (estimator as a whole; 2 datasets per cell)", res2,
        "NOT_SUPPORTED for C_Q selection in the tested cells only (n=150, Gaussian-X, all pairs; not tested at n=75); the E-step target differs (larger posterior covariance trace)")

    # H3 criterion: Q_Z / Occam term
    e1B = [c + ": " + cnt("E1", c, "B_CQ_minus_PZ") for c in sorted(s[s.experiment == "E1"].cell.unique())]
    res3 = ("B (no Q_Z) in E2/E3: " + "; ".join(f"{c}: {cnt(e, c, 'B_CQ_minus_PZ')}" for e, c in
                                                s[s.experiment.isin(["E2", "E3"])][["experiment", "cell"]].drop_duplicates().values)
            + " | E1: " + "; ".join(e1B))
    add("H3", "the K-proportional Q_Z term (constant nK(1+ln 2pi)/2 under scale_Z) is necessary for C_Q to avoid over-selection",
        "removing Q_Z (B) or using conditional plug-ins (C, D) over-selects",
        "same states, criteria A vs B/C/D", "MATCHED_ESTIMATOR; ONE_FACTOR_COMPARISON (criterion)", res3,
        "SUPPORTED for lineage E (B over-selects in every complete cell); with OLD states B is sometimes exact (estimator x criterion interaction)")

    # H4 sample size
    res4 = "; ".join(f"{f} {sig}: n=75 {e2(f, 75, sig, 'A_CQ')} -> n=150 {e2(f, 150, sig, 'A_CQ')}"
                     for f in ("gaussian", "poisson") for sig in ("weak", "strong"))
    add("H4", "small n (75) is a main cause of C_Q under-selection in the canonical setting",
        "C_Q under-selects at n=75 and reaches K_GEN at n=150 with everything else fixed",
        "E2 n=75 vs n=150 (same generator law, family, signal, mask rule)", "ONE_FACTOR_COMPARISON (n; datasets differ by construction)",
        res4, "SUPPORTED (4 datasets per cell)")

    # H5 signal
    res5 = "; ".join(f"{f} n={n}: weak {e2(f, n, 'weak', 'A_CQ')} -> strong {e2(f, n, 'strong', 'A_CQ')}"
                     for f in ("gaussian", "poisson") for n in (75, 150))
    add("H5", "weak signal (F row norm^2 0.5, w^2 0.6) is a cause of under-selection",
        "strong signal raises the exact share at fixed n",
        "E2 weak vs strong (F and w changed together)", "MULTI_FACTOR_COMPARISON within signal", res5,
        "PARTIALLY_SUPPORTED (direction as predicted; MULTI_FACTOR: F and w changed together, two levels only; no causal ranking against n)")

    # H6 X family
    res6 = "; ".join(f"n={n} {sig}: gaussian {e2('gaussian', n, sig, 'A_CQ')} vs poisson {e2('poisson', n, sig, 'A_CQ')}"
                     for n in (75, 150) for sig in ("weak", "strong"))
    add("H6", "Poisson-X (vs Gaussian-X) is a cause of the failures",
        "Gaussian-X with matched expected Fisher information succeeds where Poisson-X fails",
        "E2 Gaussian vs Poisson X (same Z, F, Y, mask)", "MULTI_FACTOR_COMPARISON (family; Fisher information matched)", res6,
        "NOT_SUPPORTED as a difference within the tested cells (MULTI_FACTOR; at most 1 dataset apart; not a causal claim)")

    # H7 observation mask
    f08 = s[(s.experiment == "E3") & (s.cell == "train fraction 0.8") & (s.criterion == "A_CQ")].iloc[0]
    f10 = s[(s.experiment == "E3") & (s.cell == "train fraction 1.0") & (s.criterion == "A_CQ")].iloc[0]
    res7 = (f"C_Q: 1.0 {cnt('E3', 'train fraction 1.0', 'A_CQ')} (K_HAT {f10.k_hat_values}), 0.8 {cnt('E3', 'train fraction 0.8', 'A_CQ')} "
            f"(K_HAT {f08.k_hat_values}); under-side slack median {f10.slack_under_median:.1f} -> {f08.slack_under_median:.1f}; "
            f"C_Lap: 1.0 {cnt('E3', 'train fraction 1.0', 'E_CLap')}, 0.8 {cnt('E3', 'train fraction 0.8', 'E_CLap')}")
    add("H7", "the 80% relation mask (fewer observed pairs) is a cause of under-selection",
        "fewer training pairs lowers the exact share / under-side slack",
        "E3 train fraction 1.0 vs 0.8 of the PR #155 training pairs (holdout identical)", "ONE_FACTOR_COMPARISON (mask)", res7,
        "INCONCLUSIVE (both 0/4 at n=75 weak: floor effect; median slack moves in the predicted direction but not every dataset)")

    # H8 generator (historical vs canonical)
    res8 = "; ".join(f"{c}: {cnt('E1', c, 'A_CQ')}" for c in sorted(s[s.experiment == "E1"].cell.unique()))
    add("H8", "the old MATLAB-type generator (z-scored Z and X) is what made K=9 selectable",
        "at n=150, C_Q selects K_GEN=9 under G_old but not under G_canon",
        "E1 G_old vs G_canon (same estimator)", "MULTI_FACTOR_COMPARISON (generator family)", res8,
        "NOT_SUPPORTED as a necessary condition at n=150 (both generators exact, 2 datasets each; MULTI_FACTOR; not tested at n=75)")

    # H9 Laplace approximation accuracy
    ok = li[li.lap_status == "OK"]
    deg = int(ok.degenerate.sum())
    res9 = (f"{len(ok)} states, degenerate weights (relative ESS < 0.01) in {deg}; relative ESS median {ok.relative_ess.median():.2g}; "
            f"pooled correction range {ok.c_pooled.min():.1f} to {ok.c_pooled.max():.1f}")
    add("H9", "C_Lap results are driven by the Laplace approximation error",
        "IS correction c_K changes the K ordering between 4, 5, 6",
        "existing A1 states, IS with q = N(Z_hat, H^-1)", "MATCHED_ESTIMATOR (same states)", res9,
        "UNRESOLVED (IS weights degenerate; the correction is not a valid estimate of the marginal likelihood)")

    # H10 prediction vs selection
    m = hk.merge(pm, on=["experiment", "cell", "dataset"], suffixes=("_khat", "_kgen"))
    m = m[m.y_heldout_mean_logscore_khat.notna() & (m.cell != "train fraction 1.0")]   # E3 1.0 = E2 Poisson n=75 weak states
    m["d"] = m.y_heldout_mean_logscore_khat - m.y_heldout_mean_logscore_kgen
    wrong = m[(m.k_hat != 5) & m.criterion.isin(["A_CQ", "E_CLap"])]
    res10 = (f"held-out Y log score at K_HAT minus at K_GEN when K_HAT != K_GEN (A, E; E2/E3): median {wrong.d.median():.4f}, "
             f"max {wrong.d.max():.4f}, worse in {int((wrong.d < 0).sum())}/{len(wrong)} (dataset x criterion, states counted once)")
    add("H10", "a wrong K_HAT predicts held-out relations as well as K_GEN (selection failure is harmless for prediction)",
        "held-out Y log score at K_HAT is not lower than at K_GEN",
        "same dataset, states at K_HAT vs K_GEN", "MATCHED_ESTIMATOR", res10,
        "NOT_SUPPORTED (descriptive: wrong K_HAT is usually slightly worse; differences small, no uncertainty quantified)")

    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(pd.DataFrame(rows)[["id", "verdict"]].to_string())


if __name__ == "__main__":
    main()
