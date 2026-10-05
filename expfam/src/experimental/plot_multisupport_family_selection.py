"""Figures for a multi-support family-selection run (main stage only).

Every figure is drawn from the derived CSVs in ``tables/`` (produced by
``summarize_multisupport_family_selection.py``); no figure filters out failed
runs.  ``figures/figure_captions.md`` lists, per figure, the source CSV and
the evaluated / failed counts so that failures are visible next to the plot.

    python plot_multisupport_family_selection.py --run-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette (light), slots in fixed order; sequential blue.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUPS = ("COUNT", "POSITIVE", "BOUNDED")
STAGE = "main"
SG = {"COUNT": "GROUP_COUNT", "POSITIVE": "GROUP_POSITIVE_CONTINUOUS",
      "BOUNDED": "GROUP_BOUNDED_CONTINUOUS", "REAL": "GROUP_REAL"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False})


def rd(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick(rows, **kw):
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]


def one(rows, **kw):
    got = pick(rows, **kw)
    return got[0] if got else None


CAPTIONS: list[str] = []


def caption(name, src, text):
    CAPTIONS.append(f"## {name}\n\n- source: `tables/{src}`\n- {text}\n")


def bars(ax, labels, series, colors, names, ylim=(0, 1.05)):
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    for i, ((vals, err), col, nm) in enumerate(zip(series, colors, names)):
        pos = x - 0.4 + w * (i + 0.5)
        v = np.array([np.nan if a is None else a for a in vals], dtype=float)
        ax.bar(pos, v, w * 0.92, color=col, label=nm, edgecolor=SURF, linewidth=1)
        if err is not None:
            lo = np.array([np.nan if e is None else e[0] for e in err], dtype=float)
            hi = np.array([np.nan if e is None else e[1] for e in err], dtype=float)
            ax.errorbar(pos, v, yerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)], fmt="none", ecolor=INK2,
                        elinewidth=1, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)


def rate(r):
    return None if r is None else float(r["recovery_rate"])


def ci(r):
    return None if r is None else (float(r["wilson95_low"]), float(r["wilson95_high"]))


def cnt(r):
    return "n/a" if r is None else f"{r['n_correct']}/{r['n_evaluated']} (failed {r['n_failed']})"


def fig01(fam, out):
    series, names, notes = [], [], []
    spec = [("A", "SELECTOR_ONLY", "A: selector only (true Z)"),
            ("B", "AUTO_START_FIRST", "B: Auto MCEM, START_FIRST"),
            ("B", "AUTO_START_LAST", "B: Auto MCEM, START_LAST"),
            ("C", "AUTO_START_FIRST", "C: mixed-X Auto, START_FIRST")]
    for exp, meth, nm in spec:
        rows = []
        for g in GROUPS:
            gd = "MIXED" if exp == "C" else g
            rows.append(one(fam, scope="score_decided_only", grouping="pooled_difficulty_n",
                            experiment=exp, stage=STAGE, method=meth, group_design=gd,
                            support_group=SG[g], true_family="ALL"))
        series.append(([rate(r) for r in rows], [ci(r) for r in rows]))
        names.append(nm)
        notes.append(f"{nm}: " + "; ".join(f"{g} {cnt(r)}" for g, r in zip(GROUPS, rows)))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars(ax, GROUPS, series, [C1, C2, C3, C4], names)
    ax.set_ylabel("family recovery rate (score-decided columns)")
    ax.set_title("Family recovery by support group (main stage; 95% Wilson CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "01_family_recovery_by_support_group.png", dpi=160)
    plt.close(fig)
    caption("01_family_recovery_by_support_group.png", "family_recovery_summary.csv",
            "pooled over difficulty and n; failed columns count as incorrect. " +
            " | ".join(notes))


def fig_conf(conf, group, idx, out):
    rows = pick(conf, experiment="A", stage=STAGE, method="SELECTOR_ONLY",
                group_design=group, difficulty="ALL", n="ALL")
    trues = sorted({r["true_family"] for r in rows})
    sels = sorted({r["selected_family"] for r in rows} | set(trues))
    M = np.zeros((len(trues), len(sels)))
    for r in rows:
        M[trues.index(r["true_family"]), sels.index(r["selected_family"])] += int(r["count"])
    P = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * len(sels), 1.4 + 0.75 * len(trues)))
    ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(trues)):
        for j in range(len(sels)):
            ax.text(j, i, f"{P[i, j]:.2f}\n({int(M[i, j])})", ha="center", va="center",
                    fontsize=8, color="white" if P[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(sels)))
    ax.set_xticklabels(sels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(trues)))
    ax.set_yticklabels(trues, fontsize=8)
    ax.set_xlabel("selected family")
    ax.set_ylabel("true family")
    ax.set_title(f"{group}: family confusion, selector only (true Z)\n"
                 "main stage, pooled over difficulty and n; row-normalised (count)",
                 fontsize=9)
    fig.tight_layout()
    name = f"0{idx}_family_confusion_{group.lower()}.png"
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    caption(name, "confusion_matrix_family.csv",
            f"total columns {int(M.sum())}; FAILURE column present if any selection failed.")


def fig_split(fam, key, levels, labels, name, title, out):
    spec = [("A", "SELECTOR_ONLY", "A: selector only"),
            ("B", "AUTO_START_FIRST", "B: Auto MCEM (START_FIRST)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    notes = []
    for ax, (exp, meth, nm) in zip(axes, spec):
        series = []
        for lv in levels:
            rows = []
            for g in GROUPS:
                kw = dict(scope="score_decided_only", experiment=exp, stage=STAGE,
                          method=meth, group_design=g, support_group=SG[g], true_family="ALL")
                kw["grouping"] = "by_difficulty" if key == "difficulty" else "by_n"
                kw[key] = lv
                rows.append(one(fam, **kw))
            series.append(([rate(r) for r in rows], [ci(r) for r in rows]))
            notes.append(f"{nm} {lv}: " + "; ".join(f"{g} {cnt(r)}" for g, r in zip(GROUPS, rows)))
        bars(ax, GROUPS, series, [C1, C2], labels)
        ax.set_title(nm)
    axes[0].set_ylabel("family recovery rate (score-decided columns)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.08), fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    caption(name, "family_recovery_summary.csv", " | ".join(notes))


def fig07(rmse, out):
    order = ("ORACLE", "AUTO_START_FIRST", "SUPPORT_ONLY")
    cols = [C1, C2, C3]
    groups = ("COUNT", "POSITIVE", "BOUNDED", "MIXED")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    notes = []
    for gi, g in enumerate(groups):
        for mi, m in enumerate(order):
            vals = [float(r["rmse_auto"] if m.startswith("AUTO") else
                          r["rmse_oracle"] if m == "ORACLE" else r["rmse_support_only"])
                    for r in rmse if r["stage"] == STAGE and r["group_design"] == g
                    and r["auto_method"] == "AUTO_START_FIRST"
                    and (r["rmse_auto"] if m.startswith("AUTO") else
                         r["rmse_oracle"] if m == "ORACLE" else r["rmse_support_only"]) not in ("", "None")]
            pos = gi * 4 + mi
            if vals:
                bp = ax.boxplot([vals], positions=[pos], widths=0.7, patch_artist=True,
                                showfliers=True)
                for b in bp["boxes"]:
                    b.set(facecolor=cols[mi], alpha=0.85, edgecolor=INK2)
                for k in ("medians",):
                    for l in bp[k]:
                        l.set(color=INK)
            notes.append(f"{g}/{m}: {len(vals)} successful fits")
    ax.set_xticks([gi * 4 + 1 for gi in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel("RMSE(Z) after Procrustes")
    ax.set_title("Latent-position recovery: Oracle vs Auto vs Support-only (main stage)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in
                       zip(cols, ("Oracle (true families)", "Auto (START_FIRST)",
                                  "Support-only (priority-first)"))], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "07_rmse_z_oracle_auto_supportonly.png", dpi=160)
    plt.close(fig)
    caption("07_rmse_z_oracle_auto_supportonly.png", "rmse_paired_gaps.csv",
            "pooled over difficulty and n; failed fits have no RMSE and are counted "
            "in rmse_summary.csv (n_failed). " + "; ".join(notes))


def fig08(sel_path, out):
    rows = [r for r in rd(sel_path) if r["experiment"] == "A" and r["stage"] == STAGE
            and r["decided_by"] == "score" and r["selection_status"] == "SELECTED"]
    models = []
    for g in ("GROUP_COUNT", "GROUP_POSITIVE_CONTINUOUS", "GROUP_BOUNDED_CONTINUOUS"):
        models += sorted({r["true_candidate_model"] for r in rows if r["true_support_group"] == g})
    fig, ax = plt.subplots(figsize=(10, 4.4))
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        for ok, col in ((True, C1), (False, C2)):
            v = [float(r["score_margin"]) for r in rows if r["true_candidate_model"] == m
                 and (r["correct_candidate_model"] == "True") == ok]
            ax.scatter(i + rng.uniform(-0.25, 0.25, len(v)), v, s=9, color=col, alpha=0.6,
                       linewidths=0, label=("correct" if ok else "wrong") if i == 0 else None)
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(2, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score margin (-2 log-lik units, symlog)")
    ax.set_title("Selector-only score margin by true candidate model (main stage)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "08_score_margin_by_true_family.png", dpi=160)
    plt.close(fig)
    caption("08_score_margin_by_true_family.png", "selection_results.csv",
            f"{len(rows)} score-decided selected columns; dashed line = margin 2. "
            "Selection failures have no margin and are listed in failures.csv.")


def fig09(sa, out):
    groups = ("COUNT", "POSITIVE", "BOUNDED", "MIXED")
    full, col, notes = [], [], []
    for g in groups:
        rows = [r for r in sa if r["stage"] == STAGE and r["group_design"] == g]
        both = [r for r in rows if r["both_succeeded"] == "True"]
        full.append(sum(r["full_assignment_agree"] == "True" for r in both) / len(both) if both else None)
        nc = sum(int(r["n_columns"]) for r in both)
        col.append(sum(int(r["n_columns_agree"]) for r in both) / nc if nc else None)
        notes.append(f"{g}: {len(both)}/{len(rows)} datasets with both starts successful")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars(ax, groups, [(full, None), (col, None)], [C1, C3],
         ["identical full assignment", "per-column agreement"])
    ax.set_ylabel("agreement rate (START_FIRST vs START_LAST)")
    ax.set_title("Start sensitivity of the Auto assignment (main stage)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "09_start_agreement.png", dpi=160)
    plt.close(fig)
    caption("09_start_agreement.png", "start_agreement.csv", "; ".join(notes))


def fig10(cand, out):
    rows = pick(cand, scope="all_columns", grouping="by_difficulty", experiment="C",
                stage=STAGE, method="AUTO_START_FIRST", group_design="MIXED")
    models = [m for m in ("poisson", "binomial_n2", "cmp_nu2p0", "cmp_nu0p5", "ztp",
                          "lognormal_s1", "ptn_s1", "logitnormal_s1", "cb",
                          "tn01_s0p25_c0p5", "gaussian")
              if any(r["true_candidate_model"] == m for r in rows)]
    series, notes = [], []
    for diff in ("EASY", "HARD"):
        rr = [one(rows, difficulty=diff, true_candidate_model=m) for m in models]
        series.append(([rate(r) for r in rr], [ci(r) for r in rr]))
        notes.append(f"{diff}: " + "; ".join(f"{m} {cnt(r)}" for m, r in zip(models, rr)))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bars(ax, models, series, [C1, C2], ["EASY", "HARD"])
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("candidate-model recovery (all columns)")
    ax.set_title("Mixed-X Auto (START_FIRST): recovery by true candidate model (main stage)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "10_mixed_x_family_recovery.png", dpi=160)
    plt.close(fig)
    caption("10_mixed_x_family_recovery.png", "candidate_model_recovery_summary.csv",
            "gaussian columns are gate-decided (REAL group). " + " | ".join(notes))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--stage", default="main")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    global STAGE
    STAGE = a.stage
    t, out = a.run_dir / "tables", (a.out or a.run_dir / "figures")
    out.mkdir(exist_ok=True)
    need = ["family_recovery_summary.csv", "candidate_model_recovery_summary.csv",
            "confusion_matrix_family.csv", "rmse_paired_gaps.csv", "start_agreement.csv"]
    if not all((t / n).exists() for n in need):
        print("derived tables missing; run the summariser first")
        return 0
    fam = rd(t / "family_recovery_summary.csv")
    cand = rd(t / "candidate_model_recovery_summary.csv")
    conf = rd(t / "confusion_matrix_family.csv")
    jobs = [lambda: fig01(fam, out)]
    for i, g in enumerate(GROUPS, start=2):
        jobs.append(lambda g=g, i=i: fig_conf(conf, g, i, out))
    jobs += [lambda: fig_split(fam, "difficulty", ("EASY", "HARD"), ("EASY", "HARD"),
                               "05_family_recovery_easy_vs_hard.png",
                               "Family recovery: EASY vs HARD (frozen by calibration)", out),
             lambda: fig_split(fam, "n", ("75", "150"), ("n=75", "n=150"),
                               "06_family_recovery_n75_vs_n150.png",
                               "Family recovery: n=75 vs n=150", out),
             lambda: fig07(rd(t / "rmse_paired_gaps.csv"), out),
             lambda: fig08(t / "selection_results.csv", out),
             lambda: fig09(rd(t / "start_agreement.csv"), out),
             lambda: fig10(cand, out)]
    for job in jobs:
        try:
            job()
        except Exception as exc:                          # data not there yet
            CAPTIONS.append(f"## (figure skipped)\n\n- {type(exc).__name__}: {exc}\n")
    (out / "figure_captions.md").write_text(
        "# Figure captions and counts (auto-generated from tables/)\n\n" + "\n".join(CAPTIONS),
        encoding="utf-8")
    (out / "PLOT_SCRIPT.txt").write_text(
        "generated by expfam/src/experimental/plot_multisupport_family_selection.py\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
