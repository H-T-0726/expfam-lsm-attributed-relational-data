"""Aggregate weak_protocol (reads criterion_scores.csv, estep_same_state.csv); writes CSVs and figures.

Selection: analyze_fair_comparison.select (unchanged): mean over the two starts per candidate K, argmin over
K = 1..7, exact ties -> smallest K; INCOMPLETE unless every candidate has both starts OK (D/E need Laplace OK).
Slack: under = C(4) - C(5) (> 0 correct), over = C(5) - C(6) (< 0 correct).
Case rule: weak_protocol.CASES (fixed before any fit).  Refuses to overwrite.  No fit, no sampling.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import weak_protocol as wp  # noqa: E402
from analyze_fair_comparison import select  # noqa: E402  (pure function, PR #160)

RES = ROOT / wp.RESULTS_REL
FIG = RES / "figures"
CRITS = ["A_CQ", "B_CQ_minus_PZ", "C_cond_last", "D_cond_mode", "E_CLap"]
LAB = {"A_CQ": "A C_Q", "B_CQ_minus_PZ": "B C_Q-P_Z", "C_cond_last": "C plug-in last", "D_cond_mode": "D plug-in mode",
       "E_CLap": "E C_Lap", "PY_type": "Python-type (OLD only)"}
CELLS = ["all pairs | OLD", "all pairs | E", "80% (PR #160) | E"]
COL = {"all pairs | OLD": "#c0392b", "all pairs | E": "#2a6fdb", "80% (PR #160) | E": "#7f8c8d"}
STY = {"all pairs | OLD": "-", "all pairs | E": "-", "80% (PR #160) | E": "--"}
written = []


def write(name, df):
    p = RES / name
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    df.to_csv(p, index=False, lineterminator="\n")
    written.append(name)


def savefig(fig, name):
    p = FIG / name
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append("figures/" + name)


def one(series):
    return series.iloc[0] if len(series) else np.nan


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(RES / "criterion_scores.csv")
    sc["cell"] = np.where(sc["mask"] == "mask_80", "80% (PR #160) | E", "all pairs | " + sc.estimator)
    sc["experiment"] = "weak_n75"
    sc["dataset"] = sc.rep.astype(int)
    sc["k_gen"], sc["kmax"] = 5, 7
    sc["lap_status"] = sc.lap_status.fillna("NA")
    hashes = pd.read_csv(RES / "dataset_and_initialization_hashes.csv")

    fit_cols = ["cell", "dataset", "stage", "key", "estimator", "mask", "k", "s", "seed", "status", "runtime_s", "exception",
                "lap_status", "lap_seconds", "src"]
    write("fit_results.csv", sc[fit_cols])
    write("incomplete_and_invalid_runs.csv", sc[(sc.status != "OK") | (sc.lap_status != "OK")][fit_cols])

    full, marg = select(sc, CRITS + ["PY_type"])
    full = full[~((full.criterion == "PY_type") & (full.cell != "all pairs | OLD"))]
    marg = marg[~((marg.criterion == "PY_type") & (marg.cell != "all pairs | OLD"))]
    write("full_k_selection.csv", full)
    write("selection_margins.csv", marg)

    def lab(cell, ds, cr):
        g = full[(full.cell == cell) & (full.dataset == ds) & (full.criterion == cr)]
        return one(g.label), one(g.k_hat)

    def slack(cell, ds, cr, side):
        g = marg[(marg.cell == cell) & (marg.dataset == ds) & (marg.criterion == cr) & (marg.side == side)]
        return float(one(g.slack))

    summ = []
    for (cell, cr), g in full.groupby(["cell", "criterion"]):
        lc = g.label.value_counts()
        m = marg[(marg.cell == cell) & (marg.criterion == cr)]
        mu, mo = m[m.side == "under_side"].sort_values("dataset"), m[m.side == "over_side"].sort_values("dataset")
        summ.append(dict(cell=cell, criterion=cr, datasets=len(g), exact=int(lc.get("exact", 0)), under=int(lc.get("under", 0)),
                         over=int(lc.get("over", 0)), incomplete=int(lc.get("INCOMPLETE", 0)),
                         k_hat_by_dataset=" ".join("NA" if pd.isna(v) else str(int(v)) for v in g.sort_values("dataset").k_hat),
                         slack_under_by_dataset=" ".join(f"{v:.1f}" for v in mu.slack),
                         slack_over_by_dataset=" ".join(f"{v:.1f}" for v in mo.slack),
                         slack_under_median=float(mu.slack.median()) if len(mu) else np.nan,
                         slack_over_median=float(mo.slack.median()) if len(mo) else np.nan))
    write("selection_summary.csv", pd.DataFrame(summ))

    # ---- Phase 2: OLD vs E, same data, all pairs
    ok = sc[sc.status == "OK"]
    ec = []
    for ds in range(1, 5):
        at = ok[(ok.dataset == ds) & (ok.k == 5)]
        for cr in CRITS:
            lo, ko = lab("all pairs | OLD", ds, cr)
            le, ke = lab("all pairs | E", ds, cr)
            ec.append(dict(dataset=ds, criterion=cr, k_hat_OLD=ko, label_OLD=lo, k_hat_E=ke, label_E=le,
                           same_k_hat=bool(ko == ke) if not (pd.isna(ko) or pd.isna(ke)) else np.nan,
                           slack_under_OLD=slack("all pairs | OLD", ds, cr, "under_side"),
                           slack_under_E=slack("all pairs | E", ds, cr, "under_side"),
                           slack_over_OLD=slack("all pairs | OLD", ds, cr, "over_side"),
                           slack_over_E=slack("all pairs | E", ds, cr, "over_side"),
                           w_hat_OLD_k5=float(at[at.cell == "all pairs | OLD"].w_hat.mean()),
                           w_hat_E_k5=float(at[at.cell == "all pairs | E"].w_hat.mean()), w_true=float(at.w_true.iloc[0]),
                           mc_var_OLD_k5=float(at[at.cell == "all pairs | OLD"].mc_sample_var_trace.mean()),
                           mc_var_E_k5=float(at[at.cell == "all pairs | E"].mc_sample_var_trace.mean())))
    ec = pd.DataFrame(ec)
    ec["slack_under_OLD_minus_E"] = ec.slack_under_OLD - ec.slack_under_E
    write("estimator_comparison.csv", ec)

    # ---- Phase 3: E 80% (PR #160) vs E all pairs, same data and seed
    mc = []
    for ds in range(1, 5):
        h = hashes[hashes.rep == ds]
        for cr in CRITS:
            l8, k8 = lab("80% (PR #160) | E", ds, cr)
            la, ka = lab("all pairs | E", ds, cr)
            mc.append(dict(dataset=ds, criterion=cr, k_hat_E_80=k8, label_E_80=l8, k_hat_E_all=ka, label_E_all=la,
                           slack_under_E_80=slack("80% (PR #160) | E", ds, cr, "under_side"),
                           slack_under_E_all=slack("all pairs | E", ds, cr, "under_side"),
                           slack_over_E_80=slack("80% (PR #160) | E", ds, cr, "over_side"),
                           slack_over_E_all=slack("all pairs | E", ds, cr, "over_side"),
                           same_X_Y=bool(h.x_y_equal_pr160.all()), same_seed_rule=True,
                           init_Z_F_sigma_w_equal=bool(h[["init_Z_sha_E_all_eq_E_80", "init_F_sha_E_all_eq_E_80",
                                                          "init_sigma_sha_E_all_eq_E_80", "init_w_sha_E_all_eq_E_80"]].all().all()),
                           init_w0_equal=bool(h.init_w0_sha_E_all_eq_E_80.all()),
                           y_density_train_80=float(h.y_density_pr160_train.iloc[0]), y_density_all=float(h.y_density_all.iloc[0]),
                           n_pairs_80=int(h.n_pairs_pr160_train.iloc[0]), n_pairs_all=int(h.n_pairs_allpairs.iloc[0]),
                           comparison="MULTI_FACTOR (mask + w0 init + evaluated Y_train)"))
    mc = pd.DataFrame(mc)
    mc["slack_under_all_minus_80"] = mc.slack_under_E_all - mc.slack_under_E_80
    write("mask_condition_comparison.csv", mc)

    # ---- case classification (pre-registered rule)
    cases = []
    for cr in ("A_CQ", "E_CLap"):
        per = []
        for ds in range(1, 5):
            lo, _ = lab("all pairs | OLD", ds, cr)
            le, _ = lab("all pairs | E", ds, cr)
            per.append((lo, le))
            cases.append(dict(criterion=cr, dataset=ds, label_OLD=lo, label_E=le, pair=f"OLD {lo} / E {le}"))
        compl = [p for p in per if "INCOMPLETE" not in p]
        if compl and all(p == ("exact", "under") for p in compl):
            c = "A"
        elif compl and all(p == ("under", "under") for p in compl):
            c = "B"
        elif compl and all(p == ("exact", "exact") for p in compl):
            c = "C"
        else:
            c = "D"
        cases.append(dict(criterion=cr, dataset="overall", label_OLD="", label_E="",
                          pair=f"case {c} ({len(compl)} complete datasets)"))
    write("case_classification.csv", pd.DataFrame(cases))

    # ---- recovery / prediction at K_GEN and at K_HAT; metrics by K
    rc = ["zzT_corr", "subspace_recovery", "etaY_rel_rmse", "gram_FFT_rel_err", "x_signal_r2", "x_r2_in_sample", "w_hat",
          "w0_hat", "mc_sample_var_trace"]
    pc = [c for c in ("y_population_logscore", "y_population_logscore_oracle", "y_heldout_logscore") if c in sc]
    rr, pp = [], []
    for (cell, ds), g in ok.groupby(["cell", "dataset"]):
        base = dict(cell=cell, dataset=ds, w_true=float(g.w_true.iloc[0]))
        at5 = g[g.k == 5]
        rr.append(dict(base, at="K_GEN=5", k=5, **{c: float(at5[c].mean()) for c in rc}))
        pp.append(dict(base, at="K_GEN=5", k=5, **{c: float(at5[c].mean()) for c in pc}))
        for cr in ("A_CQ", "E_CLap"):
            kh = one(full[(full.cell == cell) & (full.dataset == ds) & (full.criterion == cr)].k_hat)
            if pd.isna(kh):
                continue
            atk = g[g.k == int(kh)]
            rr.append(dict(base, at=f"K_HAT({cr})", k=int(kh), **{c: float(atk[c].mean()) for c in rc}))
            pp.append(dict(base, at=f"K_HAT({cr})", k=int(kh), **{c: float(atk[c].mean()) for c in pc}))
    rec = pd.DataFrame(rr)
    write("parameter_recovery.csv", rec)
    write("predictive_metrics.csv", pd.DataFrame(pp))
    mbk = ok.groupby(["cell", "k"])[rc + pc[:2]].median().reset_index()
    write("metrics_by_k.csv", mbk)

    # ---- figures
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
    for ds, ax in zip(range(1, 5), axes):
        for cell in CELLS:
            m = ok[(ok.cell == cell) & (ok.dataset == ds)].groupby("k").A_CQ.mean()
            if 5 in m:
                ax.plot(m.index, m - m[5], STY[cell], color=COL[cell], marker="o", ms=4, lw=2, label=cell)
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.axvline(5, color="#bbb", lw=0.8, ls=":")
        ax.set_title(f"dataset {ds}")
        ax.set_xlabel("K")
    axes[0].set_ylabel("C_Q(K) - C_Q(5)  (start mean)")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Weak n=75 data of PR #160 E2: C_Q relative to K=5 (below 0 = preferred over K=5). Lineage E prototype; OLD 0.5 series")
    savefig(fig, "fig1_cq_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, cr in zip(axes, ("A_CQ", "E_CLap")):
        x = np.arange(1, 5)
        for j, cell in enumerate(CELLS):
            v = [slack(cell, ds, cr, "under_side") for ds in x]
            ax.bar(x + (j - 1) * 0.26, v, 0.24, color=COL[cell], label=cell)
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(x)
        ax.set_xlabel("dataset")
        ax.set_title(f"{LAB[cr]}: slack C(4) - C(5)  (> 0 favours K=5)")
    axes[0].legend(fontsize=8, frameon=False)
    savefig(fig, "fig2_under_slack.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, c in zip(axes, ("zzT_corr", "etaY_rel_rmse", "w_hat")):
        r5 = rec[rec["at"] == "K_GEN=5"]
        for j, cell in enumerate(CELLS):
            v = r5[r5.cell == cell].sort_values("dataset")[c]
            ax.bar(np.arange(1, 5) + (j - 1) * 0.26, v, 0.24, color=COL[cell], label=cell)
        if c == "w_hat":
            ax.axhline(float(rec.w_true.iloc[0]), color="#333", ls="--", lw=1)
        ax.set_title(f"{c} at K=5")
        ax.set_xlabel("dataset")
    axes[0].legend(fontsize=8, frameon=False)
    savefig(fig, "fig3_recovery_k5.png")

    es_p = RES / "estep_same_state.csv"
    if es_p.exists():
        es = pd.read_csv(es_p)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for fb, mk in (("E", "o"), ("OLD", "s")):
            e = es[es.state_fitted_by == fb]
            ax.scatter(e.median_Y_share_of_precision_E, e.median_cov_trace_ratio, marker=mk, s=40, label=f"state fitted by {fb}")
        s = np.linspace(0, 0.5, 50)
        ax.plot(s, 1 / (1 - s / 2), color="#999", lw=1, label="1/(1-s/2)")
        ax.set_xlabel("Y share of node precision (E target)")
        ax.set_ylabel("cov trace OLD / E (median node)")
        ax.legend(fontsize=8, frameon=False)
        savefig(fig, "fig4_estep_covariance.png")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    man = dict(experiment_id=wp.EXPERIMENT_ID, git_head=head, protocol_hash=wp.protocol_hash(),
               inputs={p: hashlib.sha256((RES / p).read_bytes()).hexdigest() for p in
                       ("criterion_scores.csv", "dataset_and_initialization_hashes.csv", "estep_same_state.csv") if (RES / p).exists()},
               outputs=written)
    (RES / "analysis_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    main()
