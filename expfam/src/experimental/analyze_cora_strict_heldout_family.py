"""Pre-specified analysis of the CORA-SH1 main run (protocol v2, 105 fits).

Written before any main-run score was inspected; it implements the frozen
analysis plan (protocol_frozen.json["analysis_plan"]) and nothing else:

  primary   : Delta = log_score(BX) - log_score(GX raw), mc_predictive, K=6,
              paired by (split, seed); descriptive (no p-values).
  secondary : same Delta at K=3; BX-GXc, GXc-GX at K=6; BX-YO, GX-YO at K=6/3;
              AP / AUC / Brier / log_score_pos / log_score_neg; plug-in
              estimators last / mean5; constant train-density baseline.
  diagnostics: fit status, warnings, M-step Q, F Newton convergence, runtime.

Pairs with a failed arm are excluded from that contrast and listed; nothing
is rerun or replaced.

Usage: python analyze_cora_strict_heldout_family.py --run DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARM_COLORS = {"BX": "#2a78d6", "GX": "#eb6834", "YO": "#1baf7a", "GXc": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["log_score", "ap", "auc", "brier", "log_score_pos", "log_score_neg"]
CONTRASTS = [("BX", "GX", 6, "primary"), ("BX", "GX", 3, "sensitivity_K3"),
             ("BX", "GXc", 6, "secondary_GXc"), ("GXc", "GX", 6, "secondary_GXc"),
             ("BX", "YO", 6, "secondary_vs_YO"), ("GX", "YO", 6, "secondary_vs_YO"),
             ("BX", "YO", 3, "secondary_vs_YO"), ("GX", "YO", 3, "secondary_vs_YO")]


FNEWTON_COLS = ["fnewton_msteps", "fnewton_all_converged", "fnewton_max_iters",
                "fnewton_qx_decreases", "fnewton_final_grad_max", "fnewton_min_eig_H",
                "fnewton_eta_absmax", "fnewton_line_search_failures"]


def read_fit_technical(path: Path) -> pd.DataFrame:
    """Read the appended fit_technical.csv.

    Recording-format defect of the v2 runner: rows were appended with their
    own key order but the header came from the first (non-BX) row, so BX rows
    carry 8 extra fields (FNEWTON_COLS, inserted right after ``f_update`` by
    ``run_cell``).  The columns are restored deterministically from that key
    order and checked (arm, f_update, tag, field count); values are untouched.
    The raw file is not modified."""
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    i = header.index("f_update") + 1
    long_header = header[:i] + FNEWTON_COLS + header[i:]
    recs = []
    for r in body:
        if len(r) == len(header):
            rec = dict(zip(header, r))
        elif len(r) == len(long_header):
            rec = dict(zip(long_header, r))
            if not (rec["arm"] == "BX" and rec["f_update"] == "newton"):
                raise ValueError(f"unexpected long row: {r[:3]}")
        else:
            raise ValueError(f"unexpected field count {len(r)}")
        if not rec["tag"].endswith("-" + rec["arm"]):
            raise ValueError(f"tag/arm mismatch: {rec['tag']}")
        recs.append(rec)
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=long_header, lineterminator="\n")
    w.writeheader()
    w.writerows(recs)
    buf.seek(0)
    return pd.read_csv(buf)          # same type inference as a well-formed file


def paired(scores, a, b, K, estimator, metric):
    s = scores[(scores.K == K) & (scores.estimator == estimator)]
    pa = s[s.arm == a].set_index(["split_index", "seed_index"])[metric]
    pb = s[s.arm == b].set_index(["split_index", "seed_index"])[metric]
    d = (pa - pb).dropna().rename("delta").reset_index()
    return d


def summarise(d):
    by_split = d.groupby("split_index")["delta"].mean()
    return {"n_pairs": len(d), "mean": d.delta.mean(), "median": d.delta.median(),
            "min": d.delta.min(), "max": d.delta.max(),
            "n_positive": int((d.delta > 0).sum()),
            "split_mean_min": by_split.min(), "split_mean_max": by_split.max(),
            "n_splits_positive": int((by_split > 0).sum()),
            "n_splits": int(by_split.size)}


def _style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def delta_panel(ax, d, title, ylabel):
    rng = np.random.default_rng(0)
    for s, g in d.groupby("split_index"):
        x = s + rng.uniform(-0.12, 0.12, len(g))
        ax.scatter(x, g.delta, s=36, color=ARM_COLORS["BX"], edgecolor="#fcfcfb",
                   linewidth=1.5, zorder=3)
        ax.hlines(g.delta.mean(), s - 0.28, s + 0.28, color=INK, linewidth=2, zorder=4)
    ax.axhline(0, color=INK2, linewidth=1, linestyle="--", zorder=2)
    ax.set_xticks(sorted(d.split_index.unique()))
    ax.set_xticklabels([f"split {s}" for s in sorted(d.split_index.unique())])
    _style(ax, title, ylabel)
    ax.text(0.99, 0.02, f"mean {d.delta.mean():+.2e} | {int((d.delta > 0).sum())}/{len(d)} > 0",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    run = a.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

    tech = read_fit_technical(run / "fit_technical.csv")
    tech.to_csv(out / "fit_technical_columns_restored.csv", index=False)
    scores = pd.read_csv(run / "heldout_scores.csv")
    ref = pd.read_csv(run / "reference_scores.csv")
    manifest = pd.read_csv(run / "manifest.csv")
    fnl = pd.read_csv(run / "f_newton_log.csv") if (run / "f_newton_log.csv").exists() else pd.DataFrame()
    qh = pd.read_csv(run / "mstep_q_history.csv")
    seed_of = dict(zip(manifest.cell_id, manifest.seed_index))
    scores["seed_index"] = scores.tag.map(seed_of)

    # ── completeness / failures / warnings ───────────────────────────────
    missing = sorted(set(manifest.cell_id) - set(tech.tag))
    fails = tech[tech.fit_status != "OK"]
    score_err = scores[scores.estimator_status != "OK"] if "estimator_status" in scores else scores.iloc[0:0]
    warn_rows = []
    for _, r in tech.iterrows():
        issues = []
        if r.fit_status != "OK":
            issues.append(f"fit_status={r.fit_status}")
        for col, label in (("n_warnings_em", "python_warnings"),
                           ("mstep_q_decreases", "mstep_Q_decreases"),
                           ("sampling_fallbacks", "laplace_sampling_fallbacks"),
                           ("retry_count", "retries"), ("replacement_count", "replacements"),
                           ("seed_rescue_count", "seed_rescues"),
                           ("fnewton_qx_decreases", "F_newton_QX_decreases"),
                           ("fnewton_line_search_failures", "F_newton_line_search_failures")):
            if col in r and pd.notna(r[col]) and r[col] > 0:
                issues.append(f"{label}={int(r[col])}")
        if "fnewton_all_converged" in r and r.arm == "BX" and str(r.fnewton_all_converged) != "True":
            issues.append("F_newton_not_all_converged")
        if issues:
            warn_rows.append({"tag": r.tag, "issues": "; ".join(issues)})
    for _, r in score_err.iterrows():
        warn_rows.append({"tag": r.tag, "issues": f"score_error[{r.estimator}]={r.get('score_error')}"})
    pd.DataFrame(warn_rows, columns=["tag", "issues"]).to_csv(out / "failures_warnings.csv", index=False)

    # ── contrasts ────────────────────────────────────────────────────────
    rows, cells = [], []
    for est in ("mc_predictive", "last", "mean5"):
        for metric in METRICS:
            for x, y, K, role in CONTRASTS:
                d = paired(scores, x, y, K, est, metric)
                if d.empty:
                    continue
                rows.append({"contrast": f"{x}-{y}", "K": K, "role": role,
                             "estimator": est, "metric": metric, **summarise(d)})
                if est == "mc_predictive":
                    cells.append(d.assign(contrast=f"{x}-{y}", K=K, metric=metric))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "contrast_summary.csv", index=False)
    cells = pd.concat(cells)
    cells.to_csv(out / "contrast_cells_mc_predictive.csv", index=False)
    prim = paired(scores, "BX", "GX", 6, "mc_predictive", "log_score")
    prim.to_csv(out / "primary_delta_cells.csv", index=False)
    prim.groupby("split_index")["delta"].agg(["mean", "min", "max", "count"]).to_csv(
        out / "primary_delta_by_split.csv")

    # ── arm-level metric summaries + constant baseline ───────────────────
    mc = scores[scores.estimator == "mc_predictive"]
    arm_sum = mc.groupby(["K", "arm"])[METRICS].agg(["mean", "min", "max"])
    arm_sum.to_csv(out / "arm_metric_summary_mc_predictive.csv")
    base = mc.merge(ref[["split_index", "log_score"]].rename(columns={"log_score": "ref_log_score"}),
                    on="split_index")
    base["minus_constant"] = base.log_score - base.ref_log_score
    base.groupby(["K", "arm"])["minus_constant"].agg(["mean", "min", "max", "count"]).to_csv(
        out / "arm_minus_constant_baseline.csv")

    # ── diagnostics ──────────────────────────────────────────────────────
    ok = tech[tech.fit_status == "OK"]
    diag = ok.groupby(["K", "arm"]).agg(
        n=("tag", "size"), em_seconds_mean=("em_seconds", "mean"),
        em_seconds_max=("em_seconds", "max"), peak_mb_max=("peak_working_set_mb", "max"),
        mstep_q_decreases=("mstep_q_decreases", "sum"),
        sampling_fallbacks=("sampling_fallbacks", "sum"),
        n_warnings=("n_warnings_em", "sum"), Z_mean_norm_mean=("Z_mean_norm", "mean"),
        w0_mean=("w0", "mean"), w_mean=("w", "mean"), sigma_x_min=("sigma_x_min", "min"))
    if "fnewton_max_iters" in ok:
        bx = ok[ok.arm == "BX"].groupby("K").agg(
            fnewton_max_iters=("fnewton_max_iters", "max"),
            fnewton_final_grad_max=("fnewton_final_grad_max", "max"),
            fnewton_min_eig_H=("fnewton_min_eig_H", "min"),
            fnewton_eta_absmax=("fnewton_eta_absmax", "max"),
            fnewton_qx_decreases=("fnewton_qx_decreases", "sum"),
            fnewton_ls_failures=("fnewton_line_search_failures", "sum"))
        bx.to_csv(out / "f_newton_diagnostics_by_K.csv")
    diag.to_csv(out / "diagnostics_by_arm.csv")
    qh["cell"] = qh.tag
    last = qh.sort_values("iteration").groupby("tag").tail(2)
    qtail = last.pivot_table(index="tag", columns="iteration", values="q_diff")
    qtail.to_csv(out / "mstep_q_last_increments.csv")
    status = {"n_manifest": len(manifest), "n_recorded": len(tech),
              "n_ok": int((tech.fit_status == "OK").sum()), "missing": missing,
              "failed": fails[["tag", "fit_status"]].to_dict("records"),
              "n_warning_rows": len(warn_rows),
              "q_last_increment_min": float(qh[qh.iteration == qh.iteration.max()].q_diff.min()),
              "q_last_increment_max": float(qh[qh.iteration == qh.iteration.max()].q_diff.max())}
    (out / "status.json").write_text(json.dumps(status, indent=2, default=str))

    # ── figures ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    delta_panel(ax, prim, "Primary: BX - GX raw, K=6 (mc_predictive)",
                "Delta mean held-out log score")
    fig.text(0.01, 0.005, "dots: (split, seed) pairs; bar: split mean; >0 favours BX. "
             "Descriptive only (one network).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig_primary_delta_K6.png", dpi=160)
    plt.close(fig)

    panels = [("BX", "GX", 3, "Sensitivity: BX - GX raw, K=3"),
              ("BX", "GXc", 6, "Secondary: BX - GXc, K=6"),
              ("BX", "YO", 6, "Secondary: BX - Y-only, K=6"),
              ("GX", "YO", 6, "Secondary: GX raw - Y-only, K=6")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    for axx, (x, y, K, t) in zip(axs.ravel(), panels):
        d = paired(scores, x, y, K, "mc_predictive", "log_score")
        if len(d):
            delta_panel(axx, d, t, "Delta log score")
    fig.tight_layout()
    fig.savefig(out / "fig_secondary_deltas.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    for axx, metric in zip(axs.ravel(), ["log_score", "ap", "auc", "brier"]):
        s6 = mc[mc.K == 6]
        arms = [a_ for a_ in ("BX", "GX", "GXc", "YO") if a_ in set(s6.arm)]
        for i, arm in enumerate(arms):
            v = s6[s6.arm == arm][metric]
            axx.scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v,
                        s=30, color=ARM_COLORS[arm], edgecolor="#fcfcfb", linewidth=1.5,
                        zorder=3)
            axx.hlines(v.mean(), i - 0.28, i + 0.28, color=INK, linewidth=2, zorder=4)
        if metric == "log_score":
            axx.axhline(ref.log_score.mean(), color=INK2, linestyle="--", linewidth=1)
            axx.text(len(arms) - 0.5, ref.log_score.mean(), "constant", fontsize=7,
                     color=INK2, va="bottom", ha="right")
        axx.set_xticks(range(len(arms)))
        axx.set_xticklabels(arms)
        _style(axx, f"K=6 {metric} (mc_predictive)", metric)
    fig.tight_layout()
    fig.savefig(out / "fig_secondary_metrics_K6.png", dpi=160)
    plt.close(fig)

    print(json.dumps(status, indent=1, default=str))
    print(summ[(summ.estimator == "mc_predictive") & (summ.metric == "log_score")].to_string(index=False))


if __name__ == "__main__":
    main()
