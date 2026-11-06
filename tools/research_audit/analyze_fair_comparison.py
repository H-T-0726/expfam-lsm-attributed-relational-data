"""Aggregate the fair comparison (reads criterion_scores.csv); writes CSVs and figures.

Selection rule (pre-registered): mean over the two starts per candidate K,
argmin, ties -> smallest K; K_HAT only if every candidate K has both starts OK
(and, for D/E, Laplace OK).  Datasets, not fits, are the unit.  Refuses to
overwrite.  No fit, no sampling.
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
import fair_protocol as fp  # noqa: E402

RES = ROOT / fp.RESULTS_REL
FIG = RES / "figures"
CRITS = ["A_CQ", "B_CQ_minus_PZ", "C_cond_last", "D_cond_mode", "E_CLap"]
LAB = {"A_CQ": "A C_Q", "B_CQ_minus_PZ": "B C_Q-P_Z", "C_cond_last": "C plug-in last", "D_cond_mode": "D plug-in mode",
       "E_CLap": "E C_Lap", "PY_hist": "Python rule (OLD only)"}
COL = {"A_CQ": "#2a6fdb", "B_CQ_minus_PZ": "#e8822a", "C_cond_last": "#3a9e5b", "D_cond_mode": "#c0392b", "E_CLap": "#7d5ba6",
       "PY_hist": "#555555"}


def need(c, m):
    if not c:
        raise SystemExit("STOP: " + m)


def write(name, df):
    p = RES / name
    need(not p.exists(), f"refusing to overwrite {p}")
    df.to_csv(p, index=False, lineterminator="\n")


def cells(sc):
    """Attach experiment / cell / dataset identifiers; E2 and E3 share the reused A1 n=75 states."""
    rows = []
    for _, r in sc.iterrows():
        if r.kind == "E1":
            g = "G_old" if int(r.g) == 0 else "G_canon"
            rows.append(dict(r, experiment="E1", cell=f"{g} | K_GEN={int(r.kgen)} | {r.estimator}", generator=g,
                             k_gen=int(r.kgen), n=150, dataset=int(r.rep), kmax=10, family_x="gaussian", y_frac=1.0))
        elif r.kind == "E2":
            rows.append(dict(r, experiment="E2", cell=f"{r.xfam} | n={int(r.n)} | {r.signal}", generator="canonical (PR #155)",
                             k_gen=5, dataset=int(r.rep), kmax=7, family_x=r.xfam, y_frac=1.0))
            if r.xfam == "poisson" and r.signal == "weak" and int(r.n) == 75:
                rows.append(dict(r, experiment="E3", cell="train fraction 1.0", generator="canonical (PR #155)", k_gen=5,
                                 dataset=int(r.rep), kmax=7, family_x="poisson", y_frac=1.0))
        else:
            rows.append(dict(r, experiment="E3", cell="train fraction 0.8", generator="canonical (PR #155)", k_gen=5,
                             dataset=int(r.rep), kmax=7, family_x="poisson", y_frac=0.8))
    return pd.DataFrame(rows)


def select(df, crits):
    full, marg = [], []
    keys = ["experiment", "cell", "dataset"]
    for kk, g in df.groupby(keys):
        kgen, kmax = int(g.k_gen.iloc[0]), int(g.kmax.iloc[0])
        for c in crits:
            if c not in g or g[c].isna().all() and c == "PY_hist":
                continue
            ok_rows = g[g.status.astype(str).str.startswith("OK")]
            if c in ("D_cond_mode", "E_CLap"):
                ok_rows = ok_rows[ok_rows.lap_status == "OK"]
            ok_rows = ok_rows[ok_rows[c].notna()]
            cnt = ok_rows.groupby("k").size()
            complete = all(cnt.get(k, 0) == 2 for k in range(1, kmax + 1))
            mean = ok_rows.groupby("k")[c].mean()
            if complete:
                kh = int(min(range(1, kmax + 1), key=lambda k: (mean[k], k)))
                lab = "exact" if kh == kgen else ("under" if kh < kgen else "over")
                srt = mean.sort_values()
                gap = float(srt.iloc[1] - srt.iloc[0])
            else:
                kh, lab, gap = np.nan, "INCOMPLETE", np.nan
            full.append(dict(zip(keys, kk), criterion=c, k_gen=kgen, candidate_k=f"1..{kmax}", k_hat=kh, label=lab,
                             best_minus_second=gap, n_ok_fits=int(len(ok_rows))))
            for a, b, name in ((kgen - 1, kgen, "under_side"), (kgen, kgen + 1, "over_side")):
                if a in mean and b in mean and cnt.get(a, 0) == 2 and cnt.get(b, 0) == 2:
                    marg.append(dict(zip(keys, kk), criterion=c, k_gen=kgen, side=name, k_from=a, k_to=b,
                                     slack=float(mean[a] - mean[b])))
    return pd.DataFrame(full), pd.DataFrame(marg)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(RES / "criterion_scores.csv")
    df = cells(sc)

    # ---- fit results / incomplete -------------------------------------------------
    fit_cols = ["experiment", "cell", "dataset", "stage", "key", "estimator", "k", "s", "seed", "status", "runtime_s",
                "exception", "lap_status", "lap_seconds"]
    write("fit_results.csv", df[[c for c in fit_cols if c in df]])
    inc = df[~df.status.astype(str).str.startswith("OK") | (df.lap_status.astype(str) != "OK")]
    write("incomplete_and_invalid_runs.csv", inc[[c for c in fit_cols if c in inc]])

    crits = CRITS + ["PY_hist"]
    full, marg = select(df, crits)
    write("full_k_selection.csv", full)
    write("selection_margins.csv", marg)

    # ---- condition summary (success / failure map) --------------------------------
    summ = []
    for (e, c, cr), g in full.groupby(["experiment", "cell", "criterion"]):
        lab = g.label.value_counts()
        m = marg[(marg.experiment == e) & (marg.cell == c) & (marg.criterion == cr)]
        mu, mo = m[m.side == "under_side"].slack, m[m.side == "over_side"].slack
        summ.append(dict(experiment=e, cell=c, criterion=cr, datasets=len(g), exact=int(lab.get("exact", 0)),
                         under=int(lab.get("under", 0)), over=int(lab.get("over", 0)), incomplete=int(lab.get("INCOMPLETE", 0)),
                         k_hat_values=" ".join(str(int(v)) for v in sorted(g.k_hat.dropna())),
                         slack_under_median=float(mu.median()) if len(mu) else np.nan,
                         slack_under_min=float(mu.min()) if len(mu) else np.nan,
                         slack_under_max=float(mu.max()) if len(mu) else np.nan,
                         slack_over_median=float(mo.median()) if len(mo) else np.nan,
                         slack_over_min=float(mo.min()) if len(mo) else np.nan,
                         slack_over_max=float(mo.max()) if len(mo) else np.nan))
    summ = pd.DataFrame(summ)
    write("success_failure_conditions.csv", summ)

    # ---- secondary metrics at K = K_GEN (mean over starts) -------------------------
    at = df[(df.k == df.k_gen) & df.status.astype(str).str.startswith("OK")]
    mcols = [c for c in ("x_r2_in_sample", "x_poisson_mean_deviance", "y_heldout_mean_logscore") if c in at]
    rcols = [c for c in ("zzT_corr", "etaY_rel_rmse", "gram_FFT_rel_err") if c in at]
    pred = at.groupby(["experiment", "cell", "dataset"])[mcols].mean().reset_index()
    rec = at.groupby(["experiment", "cell", "dataset"])[rcols].mean().reset_index()
    write("predictive_metrics.csv", pred)
    write("parameter_recovery.csv", rec)
    # predictive / recovery at K_HAT for the main criteria (does the chosen K predict better?)
    hk = []
    for _, r in full[full.label != "INCOMPLETE"].iterrows():
        g = df[(df.experiment == r.experiment) & (df.cell == r.cell) & (df.dataset == r.dataset) & (df.k == r.k_hat)
               & df.status.astype(str).str.startswith("OK")]
        hk.append(dict(experiment=r.experiment, cell=r.cell, dataset=r.dataset, criterion=r.criterion, k_hat=r.k_hat,
                       **{c: float(g[c].mean()) for c in mcols + rcols}))
    write("metrics_at_k_hat.csv", pd.DataFrame(hk))

    # ---- E1 estimator agreement (same data, same initial state, same seed) ---------
    e1 = df[df.experiment == "E1"].copy()
    agree = []
    if len(e1):
        for (gen, kg, rep), g in e1.groupby(["generator", "k_gen", "dataset"]):
            for c in CRITS:
                fo = full[(full.experiment == "E1") & (full.cell == f"{gen} | K_GEN={kg} | OLD") & (full.dataset == rep) & (full.criterion == c)]
                fe = full[(full.experiment == "E1") & (full.cell == f"{gen} | K_GEN={kg} | E") & (full.dataset == rep) & (full.criterion == c)]
                if len(fo) and len(fe):
                    agree.append(dict(generator=gen, k_gen=kg, dataset=rep, criterion=c, k_hat_OLD=fo.k_hat.iloc[0],
                                      k_hat_E=fe.k_hat.iloc[0], same_k_hat=bool(fo.k_hat.iloc[0] == fe.k_hat.iloc[0])))
        pairs = e1.pivot_table(index=["generator", "k_gen", "dataset", "k", "s"], columns="estimator",
                               values=CRITS + ["w_hat", "w0_hat", "zzT_corr", "etaY_rel_rmse"])
        pairs.columns = [f"{a}_{b}" for a, b in pairs.columns]
        pairs = pairs.reset_index()
        for c in CRITS:
            pairs[f"{c}_E_minus_OLD"] = pairs[f"{c}_E"] - pairs[f"{c}_OLD"]
        write("estimator_pairs_e1.csv", pairs)
    write("estimator_agreement_e1.csv", pd.DataFrame(agree))

    # ---- figures -------------------------------------------------------------------
    def bars(ax, sub, title):
        sub = sub.set_index("criterion").reindex([c for c in crits if c in sub.criterion.values])
        bottom = np.zeros(len(sub))
        for lab, colr in (("under", "#8fb3e8"), ("exact", "#4a8a4a"), ("over", "#e39a6b"), ("incomplete", "#cccccc")):
            v = (sub[lab] / sub.datasets).values
            ax.bar(range(len(sub)), v, bottom=bottom, color=colr, edgecolor="white", lw=1.5, label=lab)
            bottom += v
        ax.set_xticks(range(len(sub)), [c.split("_")[0] if c != "PY_hist" else "PY" for c in sub.index])
        ax.set_title(title, fontsize=8)

    for e in ("E1", "E2", "E3"):
        s = summ[summ.experiment == e]
        cl = sorted(s.cell.unique())
        if not cl:
            continue
        fig, axs = plt.subplots(1, len(cl), figsize=(2.6 * len(cl), 3.6), sharey=True, squeeze=False)
        for ax, c in zip(axs[0], cl):
            bars(ax, s[s.cell == c], c.replace(" | ", "\n"))
        axs[0][0].set_ylabel("share of datasets")
        axs[0][0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{e}: under / exact / over K_HAT by criterion (A C_Q, B C_Q-P_Z, C last, D mode, E C_Lap, PY Python rule)", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"fig_{e}_selection_outcome.png", dpi=140)
        fig.savefig(FIG / f"fig_{e}_selection_outcome.pdf")
        plt.close(fig)

    if len(e1):
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        p = pd.read_csv(RES / "estimator_pairs_e1.csv")
        for ax, c in zip(axs, ("A_CQ", "E_CLap", "w_hat")):
            for gen, mk in (("G_old", "o"), ("G_canon", "s")):
                q = p[p.generator == gen]
                ax.scatter(q[f"{c}_OLD"], q[f"{c}_E"], s=10, marker=mk, alpha=0.6, label=gen)
            lo = np.nanmin([p[f"{c}_OLD"].min(), p[f"{c}_E"].min()])
            hi = np.nanmax([p[f"{c}_OLD"].max(), p[f"{c}_E"].max()])
            ax.plot([lo, hi], [lo, hi], color="0.3", lw=1)
            ax.set_xlabel(f"{c} (OLD estimator)")
            ax.set_ylabel(f"{c} (E estimator)")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.suptitle("E1: same data, same initial state and seed; each point = one (dataset, K, start)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "fig_E1_estimator_pairs.png", dpi=140)
        fig.savefig(FIG / "fig_E1_estimator_pairs.pdf")
        plt.close(fig)

    e2 = summ[summ.experiment == "E2"]
    if len(e2):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, fam in zip(axs, ("gaussian", "poisson")):
            for c in ("A_CQ", "B_CQ_minus_PZ", "E_CLap"):
                for sig, ls in (("weak", "-"), ("strong", "--")):
                    q = e2[(e2.criterion == c) & e2.cell.str.startswith(fam) & e2.cell.str.endswith(sig)].copy()
                    q["n"] = q.cell.str.extract(r"n=(\d+)")[0].astype(int)
                    q = q.sort_values("n")
                    ax.plot(q.n, q.exact / (q.datasets - q.incomplete), ls, marker="o", color=COL[c], label=f"{LAB[c]} {sig}")
            ax.set_title(f"E2 {fam}-X: exact K_HAT share vs n", fontsize=10)
            ax.set_xlabel("n")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("share of complete datasets")
        axs[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "fig_E2_success_vs_n.png", dpi=140)
        fig.savefig(FIG / "fig_E2_success_vs_n.pdf")
        plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        m2 = marg[(marg.experiment == "E2") & marg.criterion.isin(["A_CQ", "E_CLap"])]
        for ax, side in zip(axs, ("under_side", "over_side")):
            q = m2[m2.side == side]
            labels = sorted(q.cell.unique())
            for j, c in enumerate(("A_CQ", "E_CLap")):
                for i, cl in enumerate(labels):
                    v = q[(q.cell == cl) & (q.criterion == c)].slack
                    ax.scatter(np.full(len(v), i + (j - 0.5) * 0.25), v, s=14, color=COL[c], label=LAB[c] if i == 0 else None)
            ax.axhline(0, color="0.3", lw=1)
            ax.set_xticks(range(len(labels)), [x.replace(" | ", "\n") for x in labels], fontsize=7)
            ax.set_title(f"E2 slack, {side.replace('_', ' ')} (> 0 favours the larger K)", fontsize=9)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "fig_E2_slack_vs_signal.png", dpi=140)
        fig.savefig(FIG / "fig_E2_slack_vs_signal.pdf")
        plt.close(fig)

    # success map heatmap (exact share), untested cells explicit
    piv = summ[summ.criterion.isin(["A_CQ", "E_CLap", "B_CQ_minus_PZ"])].copy()
    piv["share"] = piv.exact / (piv.datasets - piv.incomplete).replace(0, np.nan)
    piv["row"] = piv.experiment + " | " + piv.cell
    M = piv.pivot_table(index="row", columns="criterion", values="share")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(M) + 1.5))
    im = ax.imshow(M.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(M.values):
        ax.text(j, i, "untested" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(M.shape[1]), [LAB[c] for c in M.columns], fontsize=8)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=7)
    ax.set_title("Share of datasets with K_HAT = K_GEN (tested cells only)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig_success_map.png", dpi=140)
    fig.savefig(FIG / "fig_success_map.pdf")
    plt.close(fig)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    man = {"git_head": head, "protocol_hash": fp.protocol_hash(),
           "files": {str(p.relative_to(RES)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(RES.rglob("*")) if p.is_file() and p.suffix in (".csv", ".png", ".pdf", ".json")
                     and "fits" not in p.parts and p.name != "analysis_manifest.json"}}
    (RES / "analysis_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(summ.to_string())


if __name__ == "__main__":
    main()
