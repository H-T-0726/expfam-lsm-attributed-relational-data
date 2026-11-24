"""Pre-specified analysis for the K-selection Exp2-generalisation run.

Written and committed BEFORE any main-run fit exists; tested on fake data only.
Implements protocol.json["analysis_plan"] exactly:

  unit       independent generated dataset (series x regime x K_TRUE x rep)
  primary    C_Q_bar(K) = mean of C_Q over the 2 inits; argmin over K=1..10;
             ties within 1e-9 -> smallest K
  incomplete any needed fit FAILED/missing -> INCOMPLETE (denominator stays 10)
  secondary  per-init selections and agreement, best-vs-second margin,
             confusion matrices, curves, paired regime comparison,
             non-convergence diagnostic, runtime, failures
  CI         exact Clopper-Pearson 95% for exact / 10

C_Q is the Q-based complete-data / ICL-type criterion, NOT Schwarz BIC.
Agreement is with K_TRUE (the generating K), not with K*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

TIE_TOL = 1e-9
K_CAND = list(range(1, 11))
N_REP = 10


def clopper_pearson(x: int, n: int, alpha: float = 0.05):
    lo = 0.0 if x == 0 else float(beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lo, hi


def argmin_tie_small(values: dict) -> int:
    best = min(values.values())
    return min(k for k, v in values.items() if v <= best + TIE_TOL)


def margin(values: dict) -> float:
    s = sorted(values.values())
    return float(s[1] - s[0]) if len(s) > 1 else float("nan")


def load(run_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    fits = pd.read_csv(run_dir / "per_fit.csv", float_precision="round_trip")
    fits["C_Q"] = pd.to_numeric(fits["C_Q"], errors="coerce")
    dup = fits.fit_id.duplicated().sum()
    if dup:
        raise SystemExit(f"STOP: {dup} duplicated fit_id rows")
    extra = set(fits.fit_id) - set(manifest.fit_id)
    if extra:
        raise SystemExit(f"STOP: fits not in manifest: {sorted(extra)[:5]}")
    return manifest.merge(fits[["fit_id", "status", "failure_reason", "C_Q", "Q_strict",
                                "x_hash", "y_hash", "runtime_s", "q_rel_change_last",
                                "mstep_q_decreases", "w_hat", "w0_hat"]],
                          on="fit_id", how="left")


def select(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (s, r, kt, rep), g in df.groupby(["series", "regime", "k_true", "rep"], sort=True):
        row = dict(series=s, regime=r, k_true=kt, rep=rep, dataset_id=g.dataset_id.iloc[0])
        okm = g.status.eq("OK") & np.isfinite(g.C_Q)
        row["n_fits"] = len(g)
        row["n_missing"] = int(g.status.isna().sum())
        row["n_failed"] = int((g.status.notna() & ~okm).sum())
        # data identity across candidate K and inits (within dataset & regime)
        row["data_identical_across_fits"] = bool(g.x_hash.dropna().nunique() <= 1
                                                 and g.y_hash.dropna().nunique() <= 1)
        ok = g[okm]
        # primary: mean over the two inits, all 20 fits required
        if len(ok) == 2 * len(K_CAND):
            bar = ok.groupby("k").C_Q.mean().to_dict()
            row["primary_k"] = argmin_tie_small(bar)
            row["primary_margin"] = margin(bar)
            row["primary_status"] = "OK"
            for k, v in bar.items():
                row[f"CQbar_K{k}"] = v
        else:
            row["primary_k"] = np.nan
            row["primary_margin"] = np.nan
            row["primary_status"] = "INCOMPLETE"
        for t in (0, 1):
            gt = ok[ok.t == t]
            if len(gt) == len(K_CAND):
                vals = dict(zip(gt.k, gt.C_Q))
                row[f"init{t}_k"] = argmin_tie_small(vals)
                row[f"init{t}_margin"] = margin(vals)
            else:
                row[f"init{t}_k"] = np.nan
                row[f"init{t}_margin"] = np.nan
        row["inits_agree"] = (bool(row["init0_k"] == row["init1_k"])
                              if np.isfinite(row["init0_k"]) and np.isfinite(row["init1_k"]) else np.nan)
        out.append(row)
    sel = pd.DataFrame(out)
    for col in ("primary_k", "init0_k", "init1_k"):
        tag = col.replace("_k", "")
        sel[f"{tag}_outcome"] = np.where(sel[col].isna(), "INCOMPLETE",
                                         np.where(sel[col] == sel.k_true, "exact",
                                                  np.where(sel[col] < sel.k_true, "under", "over")))
    return sel


def summarize(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, r, kt), g in sel.groupby(["series", "regime", "k_true"], sort=True):
        oc = g.primary_outcome.value_counts()
        exact = int(oc.get("exact", 0))
        lo, hi = clopper_pearson(exact, N_REP)
        ks = g.primary_k.dropna().astype(int)
        rows.append(dict(series=s, regime=r, k_true=kt, n_datasets=len(g),
                         exact=exact, under=int(oc.get("under", 0)), over=int(oc.get("over", 0)),
                         incomplete=int(oc.get("INCOMPLETE", 0)),
                         exact_ci95_lo=round(lo, 4), exact_ci95_hi=round(hi, 4),
                         selected_k_distribution=json.dumps({int(k): int(v) for k, v in ks.value_counts().sort_index().items()}),
                         mean_selected_k=float(ks.mean()) if len(ks) else np.nan,
                         init0_exact=int((g.init0_outcome == "exact").sum()),
                         init1_exact=int((g.init1_outcome == "exact").sum()),
                         inits_agree=int((g.inits_agree == True).sum()),  # noqa: E712
                         inits_compared=int(g.inits_agree.notna().sum()),
                         median_primary_margin=float(g.primary_margin.median())))
    return pd.DataFrame(rows)


def confusion(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, r), g in sel.groupby(["series", "regime"], sort=True):
        for kt in sorted(g.k_true.unique()):
            gg = g[g.k_true == kt]
            row = dict(series=s, regime=r, k_true=kt)
            for k in K_CAND:
                row[f"sel_{k}"] = int((gg.primary_k == k).sum())
            row["incomplete"] = int(gg.primary_k.isna().sum())
            rows.append(row)
    return pd.DataFrame(rows)


def paired_regimes(sel: pd.DataFrame) -> pd.DataFrame:
    a = sel[sel.regime == "a"].set_index(["series", "k_true", "rep"])
    b = sel[sel.regime == "b"].set_index(["series", "k_true", "rep"])
    j = a[["primary_k", "primary_outcome"]].join(b[["primary_k", "primary_outcome"]],
                                                   lsuffix="_a", rsuffix="_b", how="outer")
    rows = []
    for (s, kt), g in j.groupby(level=[0, 1]):
        both = g.dropna()
        rows.append(dict(series=s, k_true=kt, pairs=len(both),
                         same_k=int((both.primary_k_a == both.primary_k_b).sum()),
                         a_exact_b_not=int(((both.primary_outcome_a == "exact") & (both.primary_outcome_b != "exact")).sum()),
                         b_exact_a_not=int(((both.primary_outcome_b == "exact") & (both.primary_outcome_a != "exact")).sum()),
                         both_exact=int(((both.primary_outcome_a == "exact") & (both.primary_outcome_b == "exact")).sum())))
    return pd.DataFrame(rows)


def convergence(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["q_rel_change_last"] = pd.to_numeric(d.q_rel_change_last, errors="coerce")
    return (d[d.status == "OK"].groupby(["series", "regime", "k_true", "k"])
            .agg(n=("fit_id", "size"), median_rel_change=("q_rel_change_last", "median"),
                 p90_rel_change=("q_rel_change_last", lambda x: x.quantile(0.9)),
                 fits_with_mstep_q_decrease=("mstep_q_decreases", lambda x: int((x > 0).sum())),
                 median_runtime_s=("runtime_s", "median"))
            .reset_index())


def curves(sel: pd.DataFrame) -> pd.DataFrame:
    cols = [f"CQbar_K{k}" for k in K_CAND]
    ok = sel[sel.primary_status == "OK"].copy()
    rel = ok[cols].sub(ok[cols].min(axis=1), axis=0)
    rel[["series", "regime", "k_true"]] = ok[["series", "regime", "k_true"]]
    return rel.groupby(["series", "regime", "k_true"]).median().reset_index()


def figures(summary, conf, curv, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = ["O", "A", "B", "C"]
    names = {"O": "O: Gauss-X/Bern-Y", "A": "A: Pois-X/Bern-Y", "B": "B: Gauss-X/Pois-Y", "C": "C: Bern-X/Gauss-Y"}
    # 1. confusion heatmaps
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
    for i, r in enumerate(("a", "b")):
        for j, s in enumerate(series):
            ax = axes[i, j]
            c = conf[(conf.series == s) & (conf.regime == r)].sort_values("k_true")
            m = c[[f"sel_{k}" for k in K_CAND]].to_numpy()
            ax.imshow(m, cmap="Blues", vmin=0, vmax=10, aspect="auto")
            for a_ in range(m.shape[0]):
                for b_ in range(m.shape[1]):
                    if m[a_, b_]:
                        ax.text(b_, a_, int(m[a_, b_]), ha="center", va="center",
                                color="white" if m[a_, b_] > 5 else "black", fontsize=8)
            ax.set_xticks(range(10)); ax.set_xticklabels(K_CAND)
            ax.set_yticks(range(len(c))); ax.set_yticklabels(c.k_true)
            ax.set_title(f"{names[s]} / regime {r}", fontsize=9)
            ax.set_xlabel("selected K (primary C_Q)"); ax.set_ylabel("K_TRUE")
    fig.suptitle("Selected K by K_TRUE (10 datasets per row; C_Q, not Schwarz BIC)")
    fig.savefig(out / "fig_confusion.png", dpi=130); plt.close(fig)
    # 2. exact / under / over bars
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), constrained_layout=True, sharey=True)
    for i, r in enumerate(("a", "b")):
        for j, s in enumerate(series):
            ax = axes[i, j]
            c = summary[(summary.series == s) & (summary.regime == r)].sort_values("k_true")
            x = np.arange(len(c))
            bottom = np.zeros(len(c))
            for lab, col in (("exact", "#2a7"), ("under", "#c55"), ("over", "#57c"), ("incomplete", "#999")):
                ax.bar(x, c[lab], bottom=bottom, color=col, label=lab, width=0.7)
                bottom += c[lab].to_numpy()
            ax.set_xticks(x); ax.set_xticklabels(c.k_true); ax.set_title(f"{names[s]} / regime {r}", fontsize=9)
            ax.set_xlabel("K_TRUE")
    axes[0, 0].legend(fontsize=8); axes[0, 0].set_ylabel("datasets"); axes[1, 0].set_ylabel("datasets")
    fig.savefig(out / "fig_exact_under_over.png", dpi=130); plt.close(fig)
    # 3. criterion curves (median of C_Q_bar - min over datasets), Fig.3-style
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for i, r in enumerate(("a", "b")):
        for j, s in enumerate(series):
            ax = axes[i, j]
            c = curv[(curv.series == s) & (curv.regime == r)]
            for _, row in c.iterrows():
                y = [row[f"CQbar_K{k}"] for k in K_CAND]
                ax.plot(K_CAND, y, marker="o", ms=3, lw=1.5, label=f"K_TRUE={int(row.k_true)}")
            ax.set_yscale("symlog", linthresh=10)
            ax.set_title(f"{names[s]} / regime {r}", fontsize=9); ax.set_xlabel("candidate K")
            ax.set_xticks(K_CAND)
    axes[0, 0].set_ylabel("median of C_Q_bar(K) - min_K"); axes[0, 0].legend(fontsize=7)
    fig.savefig(out / "fig_criterion_curves.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(a.manifest)
    df = load(a.run, manifest)
    sel = select(df)
    summ = summarize(sel)
    conf = confusion(sel)
    pair = paired_regimes(sel)
    conv = convergence(df)
    curv = curves(sel)
    sel.to_csv(a.out / "selection_per_dataset.csv", index=False)
    summ.to_csv(a.out / "summary_by_cell.csv", index=False)
    conf.to_csv(a.out / "confusion.csv", index=False)
    pair.to_csv(a.out / "paired_regimes.csv", index=False)
    conv.to_csv(a.out / "convergence_by_cell.csv", index=False)
    curv.to_csv(a.out / "criterion_curves_median.csv", index=False)
    fails = df[df.status.ne("OK")][["fit_id", "dataset_id", "status", "failure_reason"]]
    fails.to_csv(a.out / "failures_and_missing.csv", index=False)
    status = {"n_manifest": int(len(manifest)), "n_recorded": int(df.status.notna().sum()),
              "n_ok": int(df.status.eq("OK").sum()), "n_failed": int((df.status.notna() & df.status.ne("OK")).sum()),
              "n_missing": int(df.status.isna().sum()),
              "datasets_primary_incomplete": int((sel.primary_status != "OK").sum()),
              "all_datasets_data_identical_across_fits": bool(sel.data_identical_across_fits.all()),
              "runtime_total_cpu_h": float(df.runtime_s.sum() / 3600)}
    (a.out / "status.json").write_text(json.dumps(status, indent=2))
    figures(summ, conf, curv, a.out)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
