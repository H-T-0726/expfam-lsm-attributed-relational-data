"""Model structure comparison figure (paper model vs ExpFam Gaussian-X/Bernoulli-Y vs OLD/E estimators).

Documentation figure drawn from model_equivalence_matrix.csv / implementation_differences.csv
(no numbers are computed here).  Refuses to overwrite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import fair_protocol as fp  # noqa: E402

RES = ROOT / fp.RESULTS_REL
FIG = RES / "figures"


def main():
    out = FIG / "fig_model_structure_comparison.png"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    FIG.mkdir(parents=True, exist_ok=True)
    me = pd.read_csv(RES / "model_equivalence_matrix.csv")
    rows = [r for _, r in me.iterrows() if r.layer in ("model", "objective", "implementation")]
    colr = {"model": "#dbe8f7", "objective": "#e3f1df", "implementation": "#f9e3d3"}
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.6))
    ax.axis("off")
    cols = ["item", "paper", "expfam_gauss_bern", "verdict"]
    head = ["item", "Mikawa et al. 2024 (paper)", "ExpFam Gaussian-X / Bernoulli-Y", "verdict"]
    xs = [0.0, 0.16, 0.45, 0.72]
    y = 1.0
    for x, h in zip(xs, head):
        ax.text(x, y, h, fontsize=9, weight="bold", va="top", transform=ax.transAxes)
    dy = 0.9 / len(rows)
    for r in rows:
        y -= dy
        ax.add_patch(plt.Rectangle((0, y - dy * 0.15), 1.0, dy * 0.95, transform=ax.transAxes,
                                   color=colr[r.layer], zorder=0))
        for x, c in zip(xs, cols):
            ax.text(x + 0.005, y + dy * 0.55, str(r[c]), fontsize=7, va="center", transform=ax.transAxes, wrap=True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colr.values()]
    ax.legend(handles, [f"layer: {k}" for k in colr], loc="lower right", fontsize=7, frameon=False,
              bbox_to_anchor=(1.0, -0.06), ncol=3)
    ax.set_title("Model structure comparison. MODEL equivalent at mu_x = 0, sigma_z^2 = 1; OBJECTIVE equivalent; "
                 "IMPLEMENTATION differs (OLD E-step halves Y score/curvature)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
