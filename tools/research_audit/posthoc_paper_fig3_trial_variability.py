"""POST-HOC, EXPLORATORY diagnostic (written AFTER the selection results were read; not pre-registered).

Question: is per-trial over-selection (K_GEN+1) associated with the K_GEN fit of that trial being worse than the
other trials' K_GEN fits (initial-value / optimisation variability), rather than with the criterion terms?
Outputs trial_variability_posthoc.csv and trial_variability_by_kgen_posthoc.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import paper_fig3_protocol as pp  # noqa: E402

RES = ROOT / pp.RESULTS_REL
COL = {"CQ": "A_CQ", "CLap": "E_CLap", "CELBO": "F_CELBO"}


def main():
    out1, out2 = RES / "trial_variability_posthoc.csv", RES / "trial_variability_by_kgen_posthoc.csv"
    if out1.exists() or out2.exists():
        raise SystemExit("refusing to overwrite")
    s = pd.read_csv(RES / "score_by_state.csv")
    sel = pd.read_csv(RES / "selection_per_trial.csv")
    rows, summ = [], []
    for kg in pp.K_GEN:
        if kg + 1 > max(pp.K_CAND):
            continue
        for c, col in COL.items():
            p = s[s.kgen == kg].pivot_table(index="trial", columns="k", values=col, aggfunc="first")
            a, b = p[kg], p[kg + 1]
            rank = a.rank(method="min")          # 1 = best (lowest) K_GEN fit among trials
            sc = sel[(sel.K_GEN == kg) & (sel.criterion == c)].set_index("trial")
            for t in p.index:
                rows.append({"K_GEN": kg, "criterion": c, "trial": t, "C_at_KGEN": a[t], "C_at_KGEN_plus1": b[t],
                             "rank_of_KGEN_fit_among_trials": rank[t], "selected": sc.loc[t, "outcome"] if t in sc.index else np.nan,
                             "KGEN_fit_minus_best_KGEN_fit": a[t] - a.min()})
            over = sc.outcome == "over"
            summ.append({"K_GEN": kg, "criterion": c, "sd_C_at_KGEN_over_trials": float(a.std(ddof=1)),
                         "range_C_at_KGEN": float(a.max() - a.min()), "sd_C_at_KGEN_plus1": float(b.std(ddof=1)),
                         "min_C_KGEN_minus_min_C_KGEN_plus1": float(a.min() - b.min()),
                         "n_over": int(over.sum()),
                         "median_rank_KGEN_fit_when_over": float(rank[over[over].index].median()) if over.any() else np.nan,
                         "median_rank_KGEN_fit_when_exact": float(rank[sc.index[sc.outcome == 'exact']].median()) if (sc.outcome == "exact").any() else np.nan,
                         "label": "POST-HOC EXPLORATORY"})
    pd.DataFrame(rows).to_csv(out1, index=False, lineterminator="\n")
    pd.DataFrame(summ).to_csv(out2, index=False, lineterminator="\n")
    print(pd.DataFrame(summ).round(1).to_string())


if __name__ == "__main__":
    main()
