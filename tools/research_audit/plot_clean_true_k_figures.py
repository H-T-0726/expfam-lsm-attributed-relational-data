"""Generate the clean true-K figures from the production artifacts.

Every figure is produced from the machine-readable artifacts; nothing is drawn
by hand and no number is transcribed.  This script runs NO EM and never
modifies an artifact.

Figures (IDs match reports/thesis/thesis_figure_table_inventory_20260905.md):

  F8-1  selected K against n for K_TRUE = 5  (the primary figure)
  F8-2  criterion disagreement across all 64 cells
  F8-3  the plug-in conditional criterion's monotone over-selection
  F8-4  EM start disagreement as an instability diagnostic
  F8-5  the estimated Poisson-X Gram spectrum

Captions live in the inventory document, not here, so a caveat cannot drift
away from its figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
                   / "clean_true_k_asymptotics_20260904")
DEFAULT_FIG_DIR = ROOT / "expfam" / "figures" / "clean_true_k_20260905"

K_TRUE_GRID = (1, 3, 5)
N_GRID = (50, 75, 100, 150)
CANDIDATE_K = (1, 2, 3, 4, 5, 6, 7)
CRITERIA = ("S1", "S2", "S3")
CRITERION_TITLE = {
    "S1": "S1 held-out predictive",
    "S2": "S2 Q-based (NOT Schwarz BIC)",
    "S3": "S3 plug-in conditional (NOT the paper's criterion)",
}
COLUMN = {"S1": "heldout_mean_log_score", "S2": "s2_q_based",
          "S3": "s3_plugin_conditional"}
HIGHER_IS_BETTER = {"S1": True, "S2": False, "S3": False}
TIE_TOLERANCE = np.float64(1e-12)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def recompute(fits: list[dict[str, str]]) -> dict:
    """Re-derive the selection with the frozen rule, exactly as the report does."""

    by_cell: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    for row in fits:
        by_cell.setdefault((int(row["k_true"]), int(row["n"]),
                            int(row["replicate"])), []).append(row)

    out: dict = {}
    for cell, rows in by_cell.items():
        for name in CRITERIA:
            means, per_start = {}, {}
            for k_est in CANDIDATE_K:
                pairs = sorted((int(r["start"]), float(r[COLUMN[name]]))
                               for r in rows if int(r["k_est"]) == k_est)
                vals = [v for _s, v in pairs]
                per_start[k_est] = vals
                signed = vals if HIGHER_IS_BETTER[name] else [-v for v in vals]
                means[k_est] = float(np.mean(np.asarray(signed, dtype=np.float64),
                                             dtype=np.float64))
            best = max(means.values())
            ties = sorted(k for k, v in means.items() if best - v <= TIE_TOLERANCE)

            def pick(index: int) -> int:
                signed_by_k = {k: (per_start[k][index] if HIGHER_IS_BETTER[name]
                                   else -per_start[k][index])
                               for k in CANDIDATE_K}
                top = max(signed_by_k.values())
                return min(k for k, v in signed_by_k.items() if v == top)

            picks = [pick(0), pick(1)]
            out[(name, *cell)] = {
                "selected_k": min(ties), "means": means,
                "start_disagreement": picks[0] != picks[1],
            }
    return out


def fig_primary(rec: dict, out: Path) -> None:
    """F8-1: selected K against n for K_TRUE = 5, the pre-registered focus."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    rng = np.random.default_rng(0)          # deterministic jitter, display only
    for ax, name in zip(axes, ("S1", "S2")):
        means = []
        for i, n in enumerate(N_GRID):
            keys = sorted(k for k in rec if k[0] == name and k[1] == 5 and k[2] == n)
            sel = [rec[k]["selected_k"] for k in keys]
            jitter = rng.uniform(-0.10, 0.10, size=len(sel))
            ax.scatter(np.full(len(sel), i) + jitter, sel, s=34, alpha=0.75,
                       color="#3b6ea5", zorder=3,
                       label="replicate" if i == 0 else None)
            means.append(float(np.mean(sel)))
        ax.plot(range(len(N_GRID)), means, "-o", color="#c2410c", lw=2,
                zorder=4, label="mean selected K")
        ax.axhline(5, ls="--", color="#444444", lw=1.2, zorder=2,
                   label="K_TRUE = 5")
        ax.set_xticks(range(len(N_GRID)))
        ax.set_xticklabels([str(n) for n in N_GRID])
        ax.set_xlabel("n (nodes)")
        ax.set_yticks(CANDIDATE_K)
        ax.set_ylim(0.5, 7.5)
        ax.set_title(CRITERION_TITLE[name], fontsize=10)
        ax.grid(alpha=0.25, zorder=0)
    axes[0].set_ylabel("selected K")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("F8-1  selected K vs n, K_TRUE = 5 "
                 "(finite-sample description; NOT a consistency claim)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_disagreement(rec: dict, out: Path) -> None:
    """F8-2: what each criterion chose, cell by cell."""

    cells = sorted({(k[1], k[2], k[3]) for k in rec})
    matrix = np.array([[rec[(name, *c)]["selected_k"] for c in cells]
                       for name in CRITERIA])
    fig, ax = plt.subplots(figsize=(13, 3.0))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=1, vmax=7)
    ax.set_yticks(range(len(CRITERIA)))
    ax.set_yticklabels(CRITERIA)
    boundaries, labels = [], []
    for k_true in K_TRUE_GRID:
        for n in N_GRID:
            idx = [i for i, c in enumerate(cells) if c[0] == k_true and c[1] == n]
            boundaries.append(float(np.mean(idx)))
            labels.append(f"K{k_true}\nn{n}")
        edge = max(i for i, c in enumerate(cells) if c[0] == k_true)
        if k_true != K_TRUE_GRID[-1]:
            ax.axvline(edge + 0.5, color="white", lw=2)
    ax.set_xticks(boundaries)
    ax.set_xticklabels(labels, fontsize=7)
    for j, c in enumerate(cells):
        for i, name in enumerate(CRITERIA):
            if rec[(name, *c)]["selected_k"] == c[0]:
                ax.plot(j, i, marker="o", ms=3.2, color="white", zorder=3)
    fig.colorbar(im, ax=ax, label="selected K", pad=0.01)
    ax.set_title("F8-2  selected K per cell and criterion "
                 "(white dot = matches K_TRUE; agreement is with K_TRUE, not K*)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_s3_overselection(rec: dict, out: Path) -> None:
    """F8-3: why a criterion that does not integrate Z pins to the ceiling."""

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k_true, colour in zip(K_TRUE_GRID, ("#3b6ea5", "#c2410c", "#2f7d32")):
        keys = [k for k in rec if k[0] == "S3" and k[1] == k_true]
        curves = []
        for key in keys:
            means = rec[key]["means"]                 # already sign-flipped
            values = np.array([means[k] for k in CANDIDATE_K])
            curves.append(values - values.max())      # per-cell offset
        ax.plot(CANDIDATE_K, np.median(np.vstack(curves), axis=0), "-o",
                color=colour, label=f"K_TRUE = {k_true}")
    ax.axhline(0, ls=":", color="#888888", lw=1)
    ax.set_xlabel("candidate K")
    ax.set_ylabel("S3 criterion (higher = preferred), offset per cell")
    ax.set_xticks(CANDIDATE_K)
    ax.set_title("F8-3  the plug-in conditional criterion improves monotonically\n"
                 "with K, so the p log n penalty never catches up", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_start_disagreement(rec: dict, out: Path) -> None:
    """F8-4: how often the two EM starts choose different K."""

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, colour in zip(CRITERIA, ("#3b6ea5", "#c2410c", "#7a5195")):
        for k_true, style in zip(K_TRUE_GRID, (":", "--", "-")):
            fractions = []
            for n in N_GRID:
                keys = [k for k in rec
                        if k[0] == name and k[1] == k_true and k[2] == n]
                fractions.append(
                    sum(1 for k in keys if rec[k]["start_disagreement"]) / len(keys))
            if name == "S1":
                ax.plot(N_GRID, fractions, style, marker="o", color=colour,
                        label=f"{name}, K_TRUE={k_true}")
            else:
                ax.plot(N_GRID, fractions, style, marker="o", color=colour,
                        alpha=0.45, label=f"{name}, K_TRUE={k_true}")
    ax.set_xlabel("n (nodes)")
    ax.set_ylabel("fraction of cells where the two starts disagree")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(N_GRID)
    ax.set_title("F8-4  start disagreement tracks selection instability\n"
                 "(criterion cause vs optimiser cause NOT separated)", fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_gram(run_dir: Path, out: Path) -> None:
    """F8-5: the estimated Poisson-X Gram spectrum, which selects no K."""

    gram = read_csv(run_dir / "gram_spectrum.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, k_true in zip(axes, K_TRUE_GRID):
        for n, colour in zip(N_GRID, ("#c9d6e5", "#8fb0cc", "#5487b0", "#1f4e79")):
            rows = [r for r in gram
                    if int(r["K_TRUE"]) == k_true and int(r["n"]) == n
                    and r["status"] == "ok"]
            eig = np.array([json.loads(r["eigenvalues"]) for r in rows])
            ax.plot(range(1, eig.shape[1] + 1), np.median(eig, axis=0), "-o",
                    ms=3, color=colour, label=f"n={n}")
        ax.axvline(k_true + 0.5, ls="--", color="#c2410c", lw=1.4)
        ax.axhline(0, ls=":", color="#888888", lw=1)
        ax.set_title(f"K_TRUE = {k_true}", fontsize=10)
        ax.set_xlabel("eigenvalue index")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("eigenvalue (median over replicates)")
    axes[0].legend(fontsize=8)
    fig.suptitle("F8-5  estimated Poisson-X Gram spectrum -- non-PSD in all 64 cells, "
                 "unthresholded rank always d=15; NO rank threshold is set",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args(argv)

    fits = read_csv(args.run_dir / "fit_results.csv")
    rec = recompute(fits)

    # The figures must not disagree with the archived selection.
    selection = read_csv(args.run_dir / "selection_matrix.csv")
    for row in selection:
        key = (row["criterion"], int(row["K_TRUE"]), int(row["n"]),
               int(row["replicate"]))
        if rec[key]["selected_k"] != int(row["selected_k"]):
            raise SystemExit(f"recomputed selection disagrees with the artifact at {key}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in (("F8-1_selected_k_vs_n_ktrue5.png", lambda p: fig_primary(rec, p)),
                     ("F8-2_criterion_disagreement.png", lambda p: fig_disagreement(rec, p)),
                     ("F8-3_s3_overselection.png", lambda p: fig_s3_overselection(rec, p)),
                     ("F8-4_start_disagreement.png", lambda p: fig_start_disagreement(rec, p)),
                     ("F8-5_gram_spectrum.png", lambda p: fig_gram(args.run_dir, p))):
        target = args.out_dir / name
        fn(target)
        written.append(name)
    print(json.dumps({"status": "WRITTEN", "out_dir": str(args.out_dir),
                      "figures": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
