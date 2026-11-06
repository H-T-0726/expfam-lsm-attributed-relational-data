"""Recovery / predictive metrics as a function of the fitted K (mean over starts, median over datasets).

Used for Phase E2 when no criterion A/E picked a wrong K: shows what under-fitting (K < 5) and
over-fitting (K > 5) would cost on the same datasets.  Reads criterion_scores.csv; writes
metrics_by_k.csv and figures/fig8_metrics_by_k.png.  Refuses to overwrite.
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
import n75_protocol as npc  # noqa: E402

RES = ROOT / npc.RESULTS_REL
COLS = ["zzT_corr", "subspace_recovery", "etaY_rel_rmse", "x_signal_r2", "y_population_logscore", "y_heldout_logscore"]


def main():
    out = RES / "metrics_by_k.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    sc = pd.read_csv(RES / "criterion_scores.csv")
    sc = sc[sc.status == "OK"].copy()
    sc["condition"] = sc.generator + " | n=" + sc.n.astype(int).astype(str) + " | " + sc["mask"].map({"all_pairs": "all pairs", "mask_80": "80% pairs"})
    cols = [c for c in COLS if c in sc]
    per_ds = sc.groupby(["condition", "estimator", "rep", "k"])[cols].mean().reset_index()
    agg = per_ds.groupby(["condition", "estimator", "k"])[cols].median().reset_index()
    agg.to_csv(out, index=False, lineterminator="\n")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, c in zip(axs, ("subspace_recovery", "etaY_rel_rmse", "y_population_logscore")):
        for (cond, est), g in agg.groupby(["condition", "estimator"]):
            ax.plot(g.k, g[c], marker="o", ms=3, ls="-" if est == "E" else "--", label=f"{cond} | {est}")
        ax.axvline(5, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("fitted K")
        ax.set_title(c, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=5.5)
    fig.suptitle("Recovery and population Y log score vs fitted K (median over datasets; K_GEN = 5)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "fig8_metrics_by_k.png", dpi=140)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(agg[agg.condition.str.contains("n=75")].round(4).to_string())


if __name__ == "__main__":
    main()
