"""Aggregate the n=75 study (reads criterion_scores.csv, estep_nodes.csv); writes CSVs and figures.

Selection rule (pre-registered, n75_protocol.COMMON.selector): mean over the two starts
per candidate K, argmin over K = 1..10, exact ties -> smallest K; INCOMPLETE unless every
candidate has both starts OK (D/E also need Laplace OK).  Datasets are the unit.
Slack: under = C(4) - C(5) (> 0 correct), over = C(5) - C(6) (< 0 correct).
Refuses to overwrite.  No fit, no sampling.
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
import n75_protocol as npc  # noqa: E402
from analyze_fair_comparison import select  # noqa: E402  (pure function, PR #160)

RES = ROOT / npc.RESULTS_REL
FIG = RES / "figures"
CRITS = ["A_CQ", "B_CQ_minus_PZ", "C_cond_last", "D_cond_mode", "E_CLap"]
LAB = {"A_CQ": "A C_Q", "B_CQ_minus_PZ": "B C_Q-P_Z", "C_cond_last": "C plug-in last", "D_cond_mode": "D plug-in mode",
       "E_CLap": "E C_Lap", "PY_type": "Python-type (OLD only)"}
EST_COL = {"OLD": "#c0392b", "E": "#2a6fdb"}


def write(name, df):
    p = RES / name
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    df.to_csv(p, index=False, lineterminator="\n")


def label(r):
    mask = "80% pairs" if r["mask"] == "mask_80" else "all pairs"
    return f"{r['generator']} | n={int(r['n'])} | {mask}"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(RES / "criterion_scores.csv")
    sc["condition"] = sc.apply(label, axis=1)
    sc["experiment"] = sc.stage.map({"B": "B", "C": "C", "D": "D", "P160": "n150 (PR #160)"})
    sc["cell"] = sc.condition + " | " + sc.estimator
    sc["dataset"] = sc.rep.astype(int)
    sc["k_gen"] = 5
    sc["kmax"] = 10
    sc["lap_status"] = sc.lap_status.fillna("NA")

    fit_cols = ["experiment", "cell", "dataset", "stage", "key", "estimator", "k", "s", "seed", "status", "runtime_s",
                "exception", "lap_status", "lap_seconds", "src"]
    write("fit_results.csv", sc[fit_cols])
    write("incomplete_and_invalid_runs.csv", sc[(sc.status != "OK") | (sc.lap_status != "OK")][fit_cols])

    full, marg = select(sc, CRITS + ["PY_type"])
    full = full.merge(sc[["experiment", "cell", "condition", "estimator"]].drop_duplicates(), on=["experiment", "cell"])
    write("full_k_selection.csv", full)
    marg = marg.merge(sc[["experiment", "cell", "condition", "estimator"]].drop_duplicates(), on=["experiment", "cell"])
    write("selection_margins.csv", marg)

    # summary per cell x criterion
    summ = []
    for (cond, est, cr), g in full.groupby(["condition", "estimator", "criterion"]):
        lab = g.label.value_counts()
        m = marg[(marg.condition == cond) & (marg.estimator == est) & (marg.criterion == cr)]
        mu, mo = m[m.side == "under_side"].slack, m[m.side == "over_side"].slack
        summ.append(dict(condition=cond, estimator=est, criterion=cr, datasets=len(g), exact=int(lab.get("exact", 0)),
                         under=int(lab.get("under", 0)), over=int(lab.get("over", 0)), incomplete=int(lab.get("INCOMPLETE", 0)),
                         k_hat_by_dataset=" ".join("NA" if pd.isna(v) else str(int(v)) for v in g.sort_values("dataset").k_hat),
                         slack_under_by_dataset=" ".join(f"{v:.1f}" for v in m[m.side == "under_side"].sort_values("dataset").slack),
                         slack_over_by_dataset=" ".join(f"{v:.1f}" for v in m[m.side == "over_side"].sort_values("dataset").slack),
                         slack_under_median=float(mu.median()) if len(mu) else np.nan,
                         slack_over_median=float(mo.median()) if len(mo) else np.nan))
    summ = pd.DataFrame(summ)
    write("selection_summary.csv", summ)

    # estimator comparison: same dataset, OLD vs E
    ec = []
    for cond in sorted(full[full.estimator == "OLD"].condition.unique()):
        for ds in sorted(full[full.condition == cond].dataset.unique()):
            for cr in CRITS:
                o = full[(full.condition == cond) & (full.dataset == ds) & (full.criterion == cr) & (full.estimator == "OLD")]
                e = full[(full.condition == cond) & (full.dataset == ds) & (full.criterion == cr) & (full.estimator == "E")]
                mo_ = marg[(marg.condition == cond) & (marg.dataset == ds) & (marg.criterion == cr)]
                g = lambda est, side: mo_[(mo_.estimator == est) & (mo_.side == side)].slack  # noqa: E731
                at = sc[(sc.condition == cond) & (sc.dataset == ds) & (sc.k == 5)]
                ec.append(dict(condition=cond, dataset=ds, criterion=cr, k_hat_OLD=o.k_hat.iloc[0], k_hat_E=e.k_hat.iloc[0],
                               same_k_hat=bool(o.k_hat.iloc[0] == e.k_hat.iloc[0]),
                               slack_under_OLD=float(g("OLD", "under_side").iloc[0]) if len(g("OLD", "under_side")) else np.nan,
                               slack_under_E=float(g("E", "under_side").iloc[0]) if len(g("E", "under_side")) else np.nan,
                               slack_over_OLD=float(g("OLD", "over_side").iloc[0]) if len(g("OLD", "over_side")) else np.nan,
                               slack_over_E=float(g("E", "over_side").iloc[0]) if len(g("E", "over_side")) else np.nan,
                               w_hat_OLD_k5=float(at[at.estimator == "OLD"].w_hat.mean()), w_hat_E_k5=float(at[at.estimator == "E"].w_hat.mean()),
                               w_true=float(at.w_true.iloc[0]),
                               mc_var_OLD_k5=float(at[at.estimator == "OLD"].mc_sample_var_trace.mean()),
                               mc_var_E_k5=float(at[at.estimator == "E"].mc_sample_var_trace.mean())))
    write("estimator_comparison.csv", pd.DataFrame(ec))

    # recovery / prediction at K_GEN and at K_HAT (A, E criteria)
    rc = [c for c in ("zzT_corr", "subspace_recovery", "etaY_rel_rmse", "gram_FFT_rel_err", "x_signal_r2", "x_r2_in_sample",
                      "w_hat", "w0_hat", "mc_sample_var_trace") if c in sc]
    pc = [c for c in ("y_population_logscore", "y_population_logscore_oracle", "y_heldout_logscore") if c in sc]
    rows_r, rows_p = [], []
    for (cond, est, ds), g in sc[sc.status == "OK"].groupby(["condition", "estimator", "dataset"]):
        base = dict(condition=cond, estimator=est, dataset=ds, w_true=float(g.w_true.iloc[0]))
        at5 = g[g.k == 5]
        rows_r.append(dict(base, at="K_GEN=5", k=5, **{c: float(at5[c].mean()) for c in rc}))
        rows_p.append(dict(base, at="K_GEN=5", k=5, **{c: float(at5[c].mean()) for c in pc}))
        for cr in ("A_CQ", "E_CLap"):
            f = full[(full.condition == cond) & (full.estimator == est) & (full.dataset == ds) & (full.criterion == cr)]
            if len(f) and not pd.isna(f.k_hat.iloc[0]):
                kh = int(f.k_hat.iloc[0])
                gk = g[g.k == kh]
                rows_r.append(dict(base, at=f"K_HAT({cr})", k=kh, **{c: float(gk[c].mean()) for c in rc}))
                rows_p.append(dict(base, at=f"K_HAT({cr})", k=kh, **{c: float(gk[c].mean()) for c in pc}))
    rec = pd.DataFrame(rows_r)
    pred = pd.DataFrame(rows_p)
    write("parameter_recovery.csv", rec)
    write("predictive_metrics.csv", pred)

    # ---------------------------------------------------------------- figures
    def rel_curve(g, c):
        m = g.groupby("k")[c].mean()
        return m - m.get(5, np.nan)

    # 1. n=75 selection curves OLD vs E (G_canon and G_old), criteria A and E
    conds75 = ["G_canon | n=75 | all pairs", "G_old | n=75 | all pairs"]
    panels = [(c, ds) for c in conds75 for ds in sorted(sc[sc.condition == c].dataset.unique())]
    fig, axs = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 5.2), squeeze=False)
    for j, (cond, ds) in enumerate(panels):
        for i, cr in enumerate(("A_CQ", "E_CLap")):
            ax = axs[i][j]
            for est in ("OLD", "E"):
                g = sc[(sc.condition == cond) & (sc.dataset == ds) & (sc.estimator == est) & (sc.status == "OK")]
                if cr == "E_CLap":
                    g = g[g.lap_status == "OK"]
                rcv = rel_curve(g, cr)
                ax.plot(rcv.index, rcv.values, marker="o", ms=3, color=EST_COL[est], label=est)
            ax.axhline(0, color="0.4", lw=0.8)
            ax.axvline(5, color="0.6", lw=0.8, ls=":")
            ax.set_yscale("symlog", linthresh=10)
            ax.set_title(f"{cond.split(' |')[0]} rep {ds}\n{LAB[cr]}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7)
    fig.suptitle("n=75, all pairs: C(K) - C(5) on the same data, OLD vs E (< 0 at K != 5 means K would beat K_GEN)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_n75_selection_curves_old_vs_E.png", dpi=140)
    plt.close(fig)

    # 2. n=75 vs n=150: K_HAT per dataset, by condition x estimator, criteria A and E
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    order = ["G_canon | n=150 | all pairs", "G_canon | n=75 | all pairs", "G_canon | n=75 | 80% pairs",
             "G_old | n=150 | all pairs", "G_old | n=75 | all pairs"]
    for ax, cr in zip(axs, ("A_CQ", "E_CLap")):
        for x, cond in enumerate(order):
            for est, dx in (("OLD", -0.12), ("E", 0.12)):
                f = full[(full.condition == cond) & (full.estimator == est) & (full.criterion == cr)]
                if not len(f):
                    continue
                kh = f.k_hat.values.astype(float)
                jit = np.linspace(-0.05, 0.05, len(kh))
                ax.scatter(x + dx + jit, kh, color=EST_COL[est], s=28, label=est if x == 0 else None, zorder=3)
                ninc = int(np.isnan(kh).sum())
                if ninc:
                    ax.text(x + dx, 0.6, f"INC {ninc}", fontsize=6, ha="center")
        ax.axhline(5, color="0.4", lw=1, ls="--")
        ax.set_xticks(range(len(order)), [o.replace(" | ", "\n") for o in order], fontsize=7)
        ax.set_title(f"K_HAT per dataset, {LAB[cr]} (dashed = K_GEN 5)", fontsize=9)
        ax.set_ylim(0.3, 10.5)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("K_HAT")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_n75_vs_n150_khat.png", dpi=140)
    plt.close(fig)

    # 3. K_HAT distribution by estimator (n=75 all pairs, both generators), all criteria
    fig, axs = plt.subplots(1, len(CRITS), figsize=(3 * len(CRITS), 3), sharey=True)
    for ax, cr in zip(axs, CRITS):
        for est, dx in (("OLD", -0.2), ("E", 0.2)):
            f = full[full.condition.isin(conds75) & (full.estimator == est) & (full.criterion == cr)]
            cnt = f.k_hat.value_counts().reindex(range(1, 11), fill_value=0)
            ax.bar(np.arange(1, 11) + dx, cnt.values, width=0.38, color=EST_COL[est], label=est)
        ax.axvline(5, color="0.4", lw=1, ls="--")
        ax.set_title(LAB[cr], fontsize=8)
        ax.set_xticks(range(1, 11))
        ax.tick_params(labelsize=6)
    axs[0].set_ylabel("datasets (n=75, all pairs, 6 datasets)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_khat_distribution_by_estimator.png", dpi=140)
    plt.close(fig)

    # 4. C_Q vs C_Lap: under-side slack per dataset
    fig, ax = plt.subplots(figsize=(6, 5))
    mk = {"G_canon | n=75 | all pairs": "o", "G_canon | n=75 | 80% pairs": "^", "G_old | n=75 | all pairs": "s",
          "G_canon | n=150 | all pairs": "D", "G_old | n=150 | all pairs": "P"}
    for cond, m in mk.items():
        for est in ("OLD", "E"):
            a = marg[(marg.condition == cond) & (marg.estimator == est) & (marg.side == "under_side") & (marg.criterion == "A_CQ")]
            b = marg[(marg.condition == cond) & (marg.estimator == est) & (marg.side == "under_side") & (marg.criterion == "E_CLap")]
            j = a.merge(b, on="dataset", suffixes=("_cq", "_lap"))
            if len(j):
                ax.scatter(j.slack_cq, j.slack_lap, marker=m, color=EST_COL[est], s=30, alpha=0.8, label=f"{cond} | {est}")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_yscale("symlog", linthresh=50)
    ax.set_xlabel("C_Q slack C(4) - C(5)  (> 0: K=5 preferred over 4)")
    ax.set_ylabel("C_Lap slack C(4) - C(5)")
    ax.legend(fontsize=5.5, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("C_Q vs C_Lap under-side slack on the same states", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cq_vs_clap_slack.png", dpi=140)
    plt.close(fig)

    # 5. mask effect (lineage E, same data and seeds)
    fig, axs = plt.subplots(2, 4, figsize=(11, 5), squeeze=False)
    for j, ds in enumerate((1, 2, 3, 4)):
        for i, cr in enumerate(("A_CQ", "E_CLap")):
            ax = axs[i][j]
            for cond, col in (("G_canon | n=75 | all pairs", "#2a6fdb"), ("G_canon | n=75 | 80% pairs", "#e8822a")):
                g = sc[(sc.condition == cond) & (sc.dataset == ds) & (sc.estimator == "E") & (sc.status == "OK")]
                if cr == "E_CLap":
                    g = g[g.lap_status == "OK"]
                rcv = rel_curve(g, cr)
                ax.plot(rcv.index, rcv.values, marker="o", ms=3, color=col, label=cond.split("| ")[-1])
            ax.axhline(0, color="0.4", lw=0.8)
            ax.axvline(5, color="0.6", lw=0.8, ls=":")
            ax.set_yscale("symlog", linthresh=10)
            ax.set_title(f"rep {ds}, {LAB[cr]}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7)
    fig.suptitle("Mask effect (lineage E, G_canon n=75, same data and model seeds): C(K) - C(5)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_mask_effect.png", dpi=140)
    plt.close(fig)

    # 6. E-step covariance comparison
    nd = pd.read_csv(RES / "estep_nodes.csv")
    sts = list(dict.fromkeys(nd.state))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].boxplot([nd[nd.state == s].cov_trace_ratio_old_over_E for s in sts], showfliers=False)
    axs[0].set_xticks(range(1, len(sts) + 1), [s.replace("_k5_s1_E", "").replace("e1_", "n150 ") for s in sts], rotation=45, fontsize=6)
    axs[0].set_ylabel("tr(A_OLD^-1) / tr(A_E^-1) per node")
    axs[0].axhline(1, color="0.4", lw=0.8)
    axs[0].set_title("E-step Laplace covariance: OLD / E (same theta, Z, X, Y)", fontsize=9)
    share = nd.prec_trace_Y_full / (nd.prec_trace_Y_full + nd.prec_trace_X + nd.prec_trace_prior)
    axs[1].boxplot([share[nd.state == s] for s in sts], showfliers=False)
    axs[1].set_xticks(range(1, len(sts) + 1), [s.replace("_k5_s1_E", "").replace("e1_", "n150 ") for s in sts], rotation=45, fontsize=6)
    axs[1].set_ylabel("Y share of the precision trace (E target)")
    axs[1].set_title("How much of the node precision comes from Y", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_estep_covariance.png", dpi=140)
    plt.close(fig)

    # 7. selection vs recovery
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    r5 = rec[rec["at"] == "K_GEN=5"]
    for ax, cr in zip(axs, ("A_CQ", "E_CLap")):
        mm = marg[(marg.side == "under_side") & (marg.criterion == cr)]
        j = mm.merge(r5, on=["condition", "estimator", "dataset"])
        fk = full[full.criterion == cr][["condition", "estimator", "dataset", "label"]]
        j = j.merge(fk, on=["condition", "estimator", "dataset"])
        for lab_, col in (("exact", "#4a8a4a"), ("under", "#2a6fdb"), ("over", "#e8822a"), ("INCOMPLETE", "#999999")):
            q = j[j.label == lab_]
            ax.scatter(q.subspace_recovery, q.slack, color=col, s=26, label=lab_)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_yscale("symlog", linthresh=50)
        ax.set_xlabel("subspace recovery at K = 5 (sum sq. canonical corr / 5)")
        ax.set_ylabel(f"{LAB[cr]} slack C(4) - C(5)")
        ax.grid(alpha=0.3)
        ax.set_title(f"Selection vs latent recovery ({LAB[cr]}); each point = dataset x estimator x condition", fontsize=8)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_selection_vs_recovery.png", dpi=140)
    plt.close(fig)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    man = {"git_head": head, "protocol_hash": npc.protocol_hash(),
           "files": {str(p.relative_to(RES)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(RES.rglob("*")) if p.is_file() and p.suffix in (".csv", ".png", ".json")
                     and "fits" not in p.parts and p.name != "analysis_manifest.json"}}
    (RES / "analysis_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(summ[summ.criterion.isin(["A_CQ", "E_CLap", "B_CQ_minus_PZ", "PY_type"])].to_string())


if __name__ == "__main__":
    main()
