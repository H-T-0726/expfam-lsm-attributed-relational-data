"""fairness_matrix.csv, experiment_protocol.csv, causal_hypotheses.csv for the n=75 study.

Fairness rows and hypothesis texts are fixed documentation; the `result` column of
causal_hypotheses.csv is computed from the analysis CSVs (no hand-entered numbers).  The
verdict texts were written after the results were seen (INTERPRETATION).  Refuses to overwrite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import n75_protocol as npc  # noqa: E402

RES = ROOT / npc.RESULTS_REL
ITEMS = ["model", "generator", "observed data", "n, d", "K_GEN", "K candidates", "initial state", "estimation budget",
         "observation mask", "criterion", "aggregation rule"]

FAIR = {
    "B: OLD vs E (G_canon n=75 all pairs)": (["same"] * 7 + ["same", "same", "same", "same"], "",
        "ONE_FACTOR_COMPARISON (estimator as a whole: 1/2 on Y, ridge, sigmoid evaluation, E-step RNG consumption); MATCHED_CRITERION"),
    "C: OLD vs E (G_old n=75 all pairs)": (["same"] * 11, "", "ONE_FACTOR_COMPARISON (estimator as a whole); MATCHED_CRITERION"),
    "same state, criteria A-E": (["same"] * 9 + ["differ", "same"], "", "MATCHED_ESTIMATOR; ONE_FACTOR_COMPARISON (criterion)"),
    "D vs B: 80% vs all pairs (lineage E)": (["same", "same", "differ", "same", "same", "same", "differ", "same", "differ", "same", "same"],
        "observed data: 20% of Y pairs not used for training; initial w0 uses the training density (Z, F, sigma, w identical)",
        "ONE_FACTOR_COMPARISON (observation mask; the w0 init follows the mask by design)"),
    "G_canon n=75 vs n=150 (reps 1, 2)": (["same", "same", "differ", "differ", "same", "same", "differ", "same", "same", "same", "same"],
        "n=75 data are the first 75 nodes of the n=150 data (same F, w0, w); initial states come from different seeds; criteria penalty uses ln n",
        "ONE_FACTOR_COMPARISON in design (n, nested subsample); seeds of the model differ by construction"),
    "G_old n=75 vs n=150": (["same", "same", "differ", "differ", "same", "same", "differ", "same", "same", "same", "same"],
        "G_old z-scores over the n nodes and draws F, w after Z, so F and w differ between n=75 and n=150 (true w: 0.58 / -8.06 vs 5.03 / -5.93)",
        "MULTI_FACTOR_COMPARISON (n, F, w)"),
    "G_canon vs G_old at n=75": (["same", "differ", "differ", "same", "same", "same", "differ", "same", "same", "same", "same"],
        "z-scoring, F law, w law", "MULTI_FACTOR_COMPARISON"),
    "this study (G_canon n=75) vs PR #160 E2 Gaussian n=75": (["same", "differ", "differ", "same", "same", "differ", "differ", "same", "differ", "same", "same"],
        "X noise variance (0.1 vs exp(-||f||^2/2)), F row norm^2 (0.9 vs 0.5/0.9), w (1.16 vs 0.77/1.04), Z, mask (all vs 80%), K candidates (1..10 vs 1..7)",
        "MULTI_FACTOR_COMPARISON (descriptive only)"),
    "historical (Python 5/5, paper Fig. 3)": (["differ"] * 11, "", "HISTORICAL_REFERENCE / NOT_DIRECTLY_COMPARABLE"),
}


def main():
    def w(name, df):
        p = RES / name
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        df.to_csv(p, index=False, lineterminator="\n")

    rows = []
    for comp, (vals, note, label) in FAIR.items():
        for it, v in zip(ITEMS, vals):
            rows.append(dict(comparison=comp, item=it, status=v, reason_and_effect=(note if v == "differ" else ""), label=label))
    w("fairness_matrix.csv", pd.DataFrame(rows))
    proto = npc.protocol()
    pr = [dict(section=k, value=json.dumps(v, ensure_ascii=False)) for k, v in proto.items()]
    pr.append(dict(section="protocol_hash", value=npc.protocol_hash()))
    w("experiment_protocol.csv", pd.DataFrame(pr))

    s = pd.read_csv(RES / "selection_summary.csv")
    ec = pd.read_csv(RES / "estimator_comparison.csv")
    es = pd.read_csv(RES / "estep_same_state.csv")
    ci = pd.read_csv(RES / "condition_information.csv")
    rec = pd.read_csv(RES / "parameter_recovery.csv")

    def cnt(cond, est, cr):
        r = s[(s.condition == cond) & (s.estimator == est) & (s.criterion == cr)].iloc[0]
        return f"{r.exact}/{r.datasets - r.incomplete} exact (under {r.under}, over {r.over}, INC {r.incomplete}; K_HAT {r.k_hat_by_dataset})"

    c75, o75, m75 = "G_canon | n=75 | all pairs", "G_old | n=75 | all pairs", "G_canon | n=75 | 80% pairs"
    hy = []

    def add(i, cand, pred, comp, label, res, verdict):
        hy.append(dict(id=i, candidate=cand, predicted_phenomenon=pred, falsifying_comparison=comp, fairness_label=label,
                       result=res, verdict=verdict))

    a = ec[ec.condition.isin([c75, o75])].groupby("criterion").same_k_hat.agg(["sum", "count"])
    add("N1", "only the old Python estimator selects K_GEN at n=75 (E under-selects there, OLD does not)",
        "on the same n=75 data, OLD exact and E under",
        "B and C: OLD vs E, same data / init / seed / budget / criteria", "ONE_FACTOR_COMPARISON (estimator)",
        f"C_Q: G_canon OLD {cnt(c75, 'OLD', 'A_CQ')}, E {cnt(c75, 'E', 'A_CQ')}; G_old OLD {cnt(o75, 'OLD', 'A_CQ')}, E {cnt(o75, 'E', 'A_CQ')}; "
        + "same K_HAT: " + ", ".join(f"{c} {int(a.loc[c, 'sum'])}/{int(a.loc[c, 'count'])}" for c in a.index),
        "NOT_SUPPORTED in the tested conditions (both estimators exact on every dataset with C_Q)")
    add("N2", "n=75 by itself makes C_Q under-select",
        "C_Q under-selects in every n=75 condition",
        "this study's n=75 conditions", "within-study observation",
        f"G_canon all pairs E {cnt(c75, 'E', 'A_CQ')}; G_canon 80% pairs {cnt(m75, 'E', 'A_CQ')}; G_old {cnt(o75, 'E', 'A_CQ')}",
        "REJECTED as a general statement: at n=75 with these generators C_Q selects K_GEN; the PR #160 n=75 failures are condition-specific")
    g = ci.groupby("condition")[["x_snr", "x_info", "y_info_median"]].median()
    add("N3", "the PR #160 n=75 failures come from low information per node (X signal-to-noise), not from n alone",
        "conditions where C_Q fails have much less X information per node than conditions where it succeeds",
        "condition_information.csv (true parameters; no fit)", "MULTI_FACTOR_COMPARISON (descriptive only)",
        "; ".join(f"{c}: X SNR {r.x_snr:.2f}, X info {r.x_info:.1f}, Y info {r.y_info_median:.1f}" for c, r in g.iterrows()),
        "HYPOTHESIS (consistent with the data; not tested one factor at a time: X noise, F, w, Z, mask and K candidates differ)")
    add("N4", "fewer observed Y pairs (80%) causes under-selection",
        "80% mask lowers exact share / under-side slack versus all pairs, same data and seeds",
        "D vs B (lineage E)", "ONE_FACTOR_COMPARISON (mask)",
        f"all pairs {cnt(c75, 'E', 'A_CQ')} slack_under {s[(s.condition == c75) & (s.estimator == 'E') & (s.criterion == 'A_CQ')].slack_under_by_dataset.iloc[0]}; "
        f"80% {cnt(m75, 'E', 'A_CQ')} slack_under {s[(s.condition == m75) & (s.estimator == 'E') & (s.criterion == 'A_CQ')].slack_under_by_dataset.iloc[0]}",
        "NOT_SUPPORTED for the selection outcome in this condition (4/4 both); under-side slack is smaller with 80% in 3 of 4 datasets")
    e75 = es[es.n == 75]
    e150 = es[es.n == 150]
    add("N5", "the OLD E-step (1/2 on Y) differs from E more at n=150 than at n=75",
        "covariance trace ratio OLD/E grows with the Y share of the node precision (Y share grows with n and |w|)",
        "same theta / Z / X / Y, per node (no fit)", "MATCHED (same state); not a 1/2-only ablation",
        f"n=75 median ratio {e75.median_cov_trace_ratio.min():.3f}-{e75.median_cov_trace_ratio.max():.3f} (Y share {e75.median_Y_share_of_precision_E.min():.2f}-{e75.median_Y_share_of_precision_E.max():.2f}); "
        f"n=150 {e150.median_cov_trace_ratio.min():.3f}-{e150.median_cov_trace_ratio.max():.3f} (Y share {e150.median_Y_share_of_precision_E.min():.2f}-{e150.median_Y_share_of_precision_E.max():.2f}); "
        f"OLD gradient vs FD of its own (Y-halved) target max rel err {es.max_OLD_grad_rel_err_own.max():.1e}, vs the full target median {es.median_OLD_grad_rel_err_full.min():.2f}-{es.median_OLD_grad_rel_err_full.max():.2f}",
        "SUPPORTED (descriptive, same-state); the effect on K selection was nil in the tested cells")
    r5 = rec[rec["at"] == "K_GEN=5"]
    add("N6", "K selection succeeds only when the latent structure is recovered well",
        "conditions with exact K_HAT have high recovery at K_GEN",
        "parameter_recovery.csv at K = 5", "within-study observation (no failing cell in this study)",
        "subspace recovery at K=5: " + "; ".join(f"{c} {e}: {v:.3f}" for (c, e), v in r5.groupby(["condition", "estimator"]).subspace_recovery.median().items()),
        "INCONCLUSIVE in this study (every complete dataset was exact; PR #160 E2 shows lower recovery where selection failed)")
    w("causal_hypotheses.csv", pd.DataFrame(hy))
    print(pd.DataFrame(hy)[["id", "verdict"]].to_string())


if __name__ == "__main__":
    main()
