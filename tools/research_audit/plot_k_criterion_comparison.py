"""Issue #128 figures, generated from the committed CSV artifacts only.

Experiment A1 (clean n-sweep): selected K vs n, exact counts vs n,
dataset x criterion disagreement heatmap, criterion curves per cell, and the
K_TRUE-1 -> K_TRUE boundary slack.  Experiment B (paper-type generator):
criterion curves per K_TRUE, one panel row per criterion (scales differ, so
criteria are never overlaid on one axis), and a selection matrix.

Descriptive figures of fixed finite conditions; no rate or probability is
drawn or implied.  Lineage E, MANUSCRIPT_ADOPTION_STATUS NOT_DECIDED_BY_HUMAN.

Usage:
    python plot_k_criterion_comparison.py --a-dir <dir> --b-dir <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap       # noqa: E402

COLOR = {"C_Q": "#2a78d6", "C_Lap": "#eb6834", "S1": "#1baf7a",
         "S3": "#eda100", "C_cond_MC": "#1baf7a", "C_cond_last": "#eda100",
         "C_cond_mode": "#e87ba4"}
MARK = {"C_Q": "o", "C_Lap": "s", "S1": "^", "S3": "D", "C_cond_MC": "^",
        "C_cond_last": "D", "C_cond_mode": "v"}
NAME = {"C_Q": "C_Q (Q-based)", "C_Lap": "C_Lap (Laplace)",
        "S1": "S1 held-out (reference)", "S3": "S3 (diagnostic)",
        "C_cond_MC": "C_cond_MC (diag.)", "C_cond_last": "C_cond_last (diag.)",
        "C_cond_mode": "C_cond_mode (diag.)"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})


def read(p: Path) -> list[dict[str, str]]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# A1
# --------------------------------------------------------------------------

def a_figures(a_dir: Path, out: Path) -> None:
    sel = read(a_dir / "selection.csv")
    curves = read(a_dir / "criterion_curves.csv")
    dec = read(a_dir / "increment_decomposition.csv")
    k_trues = sorted({int(r["k_true"]) for r in sel})
    ns = sorted({int(r["n"]) for r in sel})
    prim = ("C_Q", "C_Lap")

    # A-1 selected K vs n
    fig, axes = plt.subplots(1, len(k_trues), figsize=(4.2 * len(k_trues), 3.6),
                             sharey=True)
    for ax, t in zip(np.atleast_1d(axes), k_trues):
        ax.axhline(t, color=INK, lw=1, ls="--", zorder=1)
        ax.text(ns[-1] + 8, t, "K_TRUE", va="center", fontsize=8, color=INK)
        for j, c in enumerate(prim):
            xs, ys, mx, my = [], [], [], []
            for n in ns:
                ks = [int(r["selected_k"]) for r in sel if r["criterion"] == c
                      and int(r["k_true"]) == t and int(r["n"]) == n
                      and r["status"] == "COMPLETE"]
                off = (j - 0.5) * 6
                rng = np.random.default_rng(n + 7 * t + j)
                xs += [n + off + rng.uniform(-1.5, 1.5) for _ in ks]
                ys += ks
                if ks:
                    mx.append(n + off)
                    my.append(np.mean(ks))
            ax.scatter(xs, ys, s=22, marker=MARK[c], color=COLOR[c],
                       edgecolor="#fcfcfb", linewidth=0.8, zorder=3, alpha=0.9)
            ax.plot(mx, my, color=COLOR[c], lw=2, zorder=2, label=NAME[c])
        ax.set_title(f"K_TRUE = {t}", color=INK)
        ax.set_xticks(ns)
        ax.set_xlabel("n")
        ax.set_yticks(range(1, 8))
    np.atleast_1d(axes)[0].set_ylabel("selected K (points: datasets; line: mean)")
    np.atleast_1d(axes)[0].legend(loc="upper left", frameon=False)
    fig.suptitle("A1: selected K vs n on the same fits (C_Q vs C_Lap)",
                 color=INK)
    save(fig, out, "A1_selected_k_vs_n")

    # A-2 exact counts vs n
    fig, axes = plt.subplots(1, len(k_trues), figsize=(4.2 * len(k_trues), 3.4),
                             sharey=False)
    for ax, t in zip(np.atleast_1d(axes), k_trues):
        width = 0.36
        tot = 0
        for j, c in enumerate(prim):
            ex, tots = [], []
            for n in ns:
                rows = [r for r in sel if r["criterion"] == c
                        and int(r["k_true"]) == t and int(r["n"]) == n]
                ex.append(sum(r["label"] == "exact" for r in rows))
                tots.append(len(rows))
            x = np.arange(len(ns)) + (j - 0.5) * width
            ax.bar(x, ex, width=width - 0.04, color=COLOR[c], label=NAME[c])
            for xi, e, tt in zip(x, ex, tots):
                ax.text(xi, e + 0.08, f"{e}/{tt}", ha="center", fontsize=8,
                        color=INK)
            tot = max(tot, max(tots))
        ax.set_xticks(np.arange(len(ns)), [str(n) for n in ns])
        ax.set_ylim(0, tot + 1)
        ax.set_xlabel("n")
        ax.set_title(f"K_TRUE = {t} ({tot} datasets per n)", color=INK)
    np.atleast_1d(axes)[0].set_ylabel("datasets selecting K = K_TRUE")
    np.atleast_1d(axes)[0].legend(loc="upper left", frameon=False)
    fig.suptitle("A1: number of datasets whose selected K equals K_TRUE "
                 "(counts, not rates)", color=INK)
    save(fig, out, "A1_exact_counts_vs_n")

    # A-3 disagreement heatmap (selected K - K_TRUE)
    crits = ("C_Q", "C_Lap", "S1", "S3")
    keys = sorted({(int(r["k_true"]), int(r["n"]), int(r["replicate"]))
                   for r in sel})
    lookup = {(r["criterion"], int(r["k_true"]), int(r["n"]),
               int(r["replicate"])): r for r in sel}
    M = np.full((len(keys), len(crits)), np.nan)
    A = [["" for _ in crits] for _ in keys]
    for i, key in enumerate(keys):
        for j, c in enumerate(crits):
            r = lookup[(c, *key)]
            if r["status"] == "COMPLETE":
                M[i, j] = int(r["selected_k"]) - key[0]
                A[i][j] = r["selected_k"]
            else:
                A[i][j] = "inc"
    cmap = ListedColormap(["#104281", "#2a78d6", "#86b6ef", "#cde2fb",
                           "#ededea", "#fbd3c2", "#f3a07e", "#eb6834",
                           "#a8401a"])
    norm = BoundaryNorm(np.arange(-4.5, 5.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(5.6, 0.19 * len(keys) + 1.6))
    im = ax.imshow(np.clip(M, -4, 4), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(crits)):
            ax.text(j, i, A[i][j], ha="center", va="center", fontsize=6.5,
                    color=INK)
    ax.set_xticks(range(len(crits)), [NAME[c] for c in crits], rotation=20,
                  ha="right")
    ax.set_yticks(range(len(keys)),
                  [f"K{t} n{n} r{r}" for (t, n, r) in keys], fontsize=6)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, ticks=range(-4, 5))
    cb.set_label("selected K − K_TRUE (cell text: selected K)")
    ax.set_title("A1: selected K by dataset and criterion", color=INK)
    save(fig, out, "A1_disagreement_heatmap")

    # A-4 criterion curves, relative to each dataset's minimum
    for c in prim:
        fig, axes = plt.subplots(len(k_trues), len(ns),
                                 figsize=(3.0 * len(ns), 2.5 * len(k_trues)),
                                 sharex=True, squeeze=False)
        for a_i, t in enumerate(k_trues):
            for b_i, n in enumerate(ns):
                ax = axes[a_i, b_i]
                per = defaultdict(dict)
                for r in curves:
                    if (r["criterion"] == c and int(r["k_true"]) == t
                            and int(r["n"]) == n and r["mean_value"] != ""):
                        per[int(r["replicate"])][int(r["k_est"])] = float(
                            r["mean_value"])
                for rep, cv in sorted(per.items()):
                    ks = sorted(cv)
                    v = np.array([cv[k] for k in ks])
                    if np.all(np.isfinite(v)):
                        ax.plot(ks, v - v.min(), color=COLOR[c], lw=1.2,
                                alpha=0.75, marker=MARK[c], ms=3)
                ax.axvline(t, color=INK, lw=0.9, ls="--")
                ax.set_yscale("symlog", linthresh=10)
                ax.set_title(f"K_TRUE={t}, n={n}", fontsize=8, color=INK)
                if b_i == 0:
                    ax.set_ylabel(f"{c} − min over K")
                if a_i == len(k_trues) - 1:
                    ax.set_xlabel("candidate K")
        fig.suptitle(f"A1: {NAME[c]} curves (mean of 2 starts; one line per "
                     "dataset; dashed = K_TRUE)", color=INK)
        fig.tight_layout()
        save(fig, out, f"A1_curves_{c}")

    # A-5 boundary slack K_TRUE-1 -> K_TRUE and K_TRUE -> K_TRUE+1
    fig, axes = plt.subplots(2, len(k_trues), figsize=(4.2 * len(k_trues), 6.0), squeeze=False)
    for col, t in enumerate(k_trues):
        for row_i, (frm, title) in enumerate(((t - 1, f"K{t-1} → K{t}"),
                                              (t, f"K{t} → K{t+1}"))):
            ax = axes[row_i, col]
            if frm < 1:
                ax.set_visible(False)
                continue
            for j, (c, colname) in enumerate((("C_Q", "CQ_slack"),
                                              ("C_Lap", "Lap_slack"))):
                xs, ys = [], []
                for r in dec:
                    if int(r["k_true"]) == t and int(r["k_from"]) == frm \
                            and r[colname] not in ("", "nan"):
                        xs.append(int(r["n"]) + (j - 0.5) * 6)
                        ys.append(float(r[colname]))
                ax.scatter(xs, ys, s=20, marker=MARK[c], color=COLOR[c],
                           edgecolor="#fcfcfb", linewidth=0.8, label=NAME[c])
            ax.axhline(0, color=INK, lw=0.9)
            ax.set_xticks(ns)
            ax.set_title(f"K_TRUE={t}: {title}", fontsize=9, color=INK)
            ax.set_ylabel("fit gain − complexity increment")
            ax.set_xlabel("n")
    axes[0, -1].legend(loc="best", frameon=False)
    fig.suptitle("A1: margin of the larger K at the K_TRUE boundary "
                 "(> 0: the larger K is preferred); diagnostic only",
                 color=INK)
    fig.tight_layout()
    save(fig, out, "A1_boundary_slack")


# --------------------------------------------------------------------------
# B
# --------------------------------------------------------------------------

B_CRITS = ("C_Q", "C_Lap", "C_cond_MC", "C_cond_last", "C_cond_mode")


def b_figures(b_dir: Path, out: Path) -> None:
    curves = read(b_dir / "criterion_curves.csv")
    sel = read(b_dir / "selection.csv")
    k_trues = sorted({int(r["k_true"]) for r in sel})
    fig, axes = plt.subplots(len(B_CRITS), len(k_trues),
                             figsize=(3.0 * len(k_trues), 2.2 * len(B_CRITS)),
                             sharex=True, squeeze=False)
    for i, c in enumerate(B_CRITS):
        for j, t in enumerate(k_trues):
            ax = axes[i, j]
            per = defaultdict(dict)
            for r in curves:
                if r["criterion"] == c and int(r["k_true"]) == t:
                    per[int(r["replicate"])][int(r["k_est"])] = float(
                        r["mean_value"])
            for rep, cv in sorted(per.items()):
                ks = sorted(cv)
                v = np.array([cv[k] for k in ks])
                ok = np.isfinite(v)
                if ok.any():
                    ax.plot(np.array(ks)[ok], v[ok] - v[ok].min(),
                            color=COLOR[c], lw=1.2, alpha=0.75,
                            marker=MARK[c], ms=3)
            ax.axvline(t, color=INK, lw=0.9, ls="--")
            ax.set_yscale("symlog", linthresh=10)
            if i == 0:
                ax.set_title(f"K_TRUE = {t}", color=INK)
            if j == 0:
                ax.set_ylabel(f"{c}\n− min over K", fontsize=8)
            if i == len(B_CRITS) - 1:
                ax.set_xlabel("candidate K")
                ax.set_xticks(range(1, 11))
    fig.suptitle("B: criterion curves on the same fits, paper-type generator "
                 "(each row its own scale; one line per dataset)", color=INK)
    fig.tight_layout()
    save(fig, out, "B_curves_by_criterion")

    # selection matrix
    keys = sorted({(int(r["k_true"]), int(r["replicate"])) for r in sel})
    lookup = {(r["criterion"], int(r["k_true"]), int(r["replicate"])): r
              for r in sel}
    M = np.full((len(keys), len(B_CRITS)), np.nan)
    A = [["" for _ in B_CRITS] for _ in keys]
    for i, key in enumerate(keys):
        for j, c in enumerate(B_CRITS):
            r = lookup[(c, *key)]
            if r["status"] == "COMPLETE":
                M[i, j] = int(r["selected_k"]) - key[0]
                A[i][j] = r["selected_k"]
            else:
                A[i][j] = "inc"
    cmap = ListedColormap(["#104281", "#2a78d6", "#86b6ef", "#cde2fb",
                           "#ededea", "#fbd3c2", "#f3a07e", "#eb6834",
                           "#a8401a"])
    norm = BoundaryNorm(np.arange(-4.5, 5.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(keys) + 1.6))
    im = ax.imshow(np.clip(M, -4, 4), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(B_CRITS)):
            ax.text(j, i, A[i][j], ha="center", va="center", fontsize=7,
                    color=INK)
    ax.set_xticks(range(len(B_CRITS)), [NAME[c] for c in B_CRITS],
                  rotation=20, ha="right")
    ax.set_yticks(range(len(keys)), [f"K{t} r{r}" for (t, r) in keys],
                  fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, ticks=range(-4, 5))
    cb.set_label("selected K − K_TRUE (clipped at ±4; text: selected K)")
    ax.set_title("B: selected K by dataset and criterion", color=INK)
    save(fig, out, "B_selection_matrix")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", type=Path)
    ap.add_argument("--b-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.a_dir:
        a_figures(a.a_dir, a.out)
    if a.b_dir:
        b_figures(a.b_dir, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
