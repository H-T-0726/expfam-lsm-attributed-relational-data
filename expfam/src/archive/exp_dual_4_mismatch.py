"""
Exp Dual-4: Distribution mismatch analysis & ablation study.

True data: family_x='poisson', family_y='bernoulli'
Setting  : n=150, d=15, k=3,  10 independent trials

Conditions tested:
  3x3 mismatch grid (all model family_x x family_y combos)
  + ablation variants:
    fix_x=True  : no X signal (Z from Y only)
    fix_w=True  : no Y signal (Z from X only)

Outputs:
  - CSV: exp_dual_4_mismatch.csv
  - Figure 4: 3x3 RMSE(Z) heatmap  (model family_x vs family_y)
  - Figure 5: Bar chart (proposed vs mismatch variants vs ablation)
"""

import sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

_SRC = Path(__file__).parent
_RES = _SRC.parent / "results"
_RES.mkdir(exist_ok=True)
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

from data_generator_expfam import generate_dual_data   # noqa
from utils_expfam import run_em_dual                   # noqa

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
TRUE_FX, TRUE_FY = "poisson", "bernoulli"
N, D, K = 150, 15, 3
N_TRIALS  = 10
L, NITER  = 5, 8
BASE_SEED = 6000

FAMILIES  = ["gaussian", "bernoulli", "poisson"]
FAM_LABEL = {"gaussian": "Gaussian", "bernoulli": "Bernoulli", "poisson": "Poisson"}


def one_run(fx_model, fy_model, trial,
            fix_w=False, fix_x=False):
    dseed = BASE_SEED + trial * 100
    mseed = dseed + 50
    data = generate_dual_data(
        n=N, d=D, k=K, seed=dseed,
        family_x=TRUE_FX, family_y=TRUE_FY,
    )
    res = run_em_dual(
        X=data["X"], Y=data["Y"],
        true_params=data,
        family_x=fx_model, family_y=fy_model,
        k=K, L=L, num_iter=NITER, seed=mseed,
        fix_w=fix_w, fix_x=fix_x,
    )
    return {
        "rmse_Z": res["rmse_Z"],
        "rmse_F": res["rmse_F"],
        "rmse_Y": res["rmse_Y"],
        "rmse_X": res["rmse_X"],
        "w0_err": res["w0_err"],
        "w_err":  res["w_err"],
    }


def main():
    t_start = time.perf_counter()
    records = []

    # ── Part A: 3×3 mismatch grid ─────────────────────────────────────
    print("=== Part A: 3x3 Model Mismatch Grid ===")
    grid_conditions = [
        (fx, fy, False, False)
        for fx in FAMILIES
        for fy in FAMILIES
    ]
    for fx_m, fy_m, fw, fx_abl in grid_conditions:
        correct = (fx_m == TRUE_FX and fy_m == TRUE_FY)
        tag = "CORRECT" if correct else "mismatch"
        for trial in range(N_TRIALS):
            print(f"  X={fx_m:9s} Y={fy_m:9s} [{tag}] trial={trial}", end=" ", flush=True)
            rec = one_run(fx_m, fy_m, trial, fix_w=fw, fix_x=fx_abl)
            rec.update(dict(
                model_fx=fx_m, model_fy=fy_m,
                fix_w=fw, fix_x=fx_abl,
                condition=f"X={FAM_LABEL[fx_m]}, Y={FAM_LABEL[fy_m]}",
                trial=trial,
            ))
            records.append(rec)
            print(f"  RMSE(Z)={rec['rmse_Z']:.4f}")

    # ── Part B: Ablation ──────────────────────────────────────────────
    print("\n=== Part B: Ablation Study ===")
    ablation_conditions = [
        # (label, fix_w, fix_x)
        ("Y-only (fix_x=True)",  False, True),   # no X signal
        ("X-only (fix_w=True)",  True,  False),  # no Y signal
    ]
    for label, fw, fx_abl in ablation_conditions:
        for trial in range(N_TRIALS):
            print(f"  [{label}] trial={trial}", end=" ", flush=True)
            rec = one_run(TRUE_FX, TRUE_FY, trial, fix_w=fw, fix_x=fx_abl)
            rec.update(dict(
                model_fx=TRUE_FX, model_fy=TRUE_FY,
                fix_w=fw, fix_x=fx_abl,
                condition=label,
                trial=trial,
            ))
            records.append(rec)
            print(f"  RMSE(Z)={rec['rmse_Z']:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(_RES / "exp_dual_4_mismatch.csv", index=False)
    print(f"\nCSV saved: {_RES / 'exp_dual_4_mismatch.csv'}")

    # ── Aggregate ──────────────────────────────────────────────────────
    agg = df.groupby(["model_fx", "model_fy", "fix_w", "fix_x", "condition"]).agg(
        rmse_Z_mean=("rmse_Z", "mean"),
        rmse_Z_std =("rmse_Z", "std"),
    ).reset_index()

    # 3x3 grid data (no ablation)
    grid_df = agg[(~agg["fix_w"]) & (~agg["fix_x"])].copy()

    # Proposed: correct families, no fix
    proposed_mean = float(grid_df[
        (grid_df.model_fx == TRUE_FX) & (grid_df.model_fy == TRUE_FY)
    ]["rmse_Z_mean"])
    proposed_std = float(grid_df[
        (grid_df.model_fx == TRUE_FX) & (grid_df.model_fy == TRUE_FY)
    ]["rmse_Z_std"])

    # ── Figure 4: 3x3 Heatmap ─────────────────────────────────────────
    heatmap = np.zeros((3, 3))
    for i, fx in enumerate(FAMILIES):
        for j, fy in enumerate(FAMILIES):
            row = grid_df[(grid_df.model_fx == fx) & (grid_df.model_fy == fy)]
            heatmap[i, j] = float(row["rmse_Z_mean"])

    fig4, ax4 = plt.subplots(figsize=(7, 5.5))
    cmap = plt.cm.RdYlGn_r   # red=bad, green=good (inverted)
    im = ax4.imshow(heatmap, cmap=cmap, aspect="auto",
                    vmin=heatmap.min() * 0.95,
                    vmax=heatmap.max() * 1.05)
    plt.colorbar(im, ax=ax4, label="RMSE(Z)")

    ax4.set_xticks(range(3))
    ax4.set_yticks(range(3))
    ax4.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=11)
    ax4.set_yticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=11)
    ax4.set_xlabel("Model family for Y", fontsize=12)
    ax4.set_ylabel("Model family for X", fontsize=12)
    ax4.set_title(
        f"Fig 4: RMSE(Z) Mismatch Heatmap\n"
        f"True data: X={FAM_LABEL[TRUE_FX]}, Y={FAM_LABEL[TRUE_FY]}  "
        f"(n={N}, d={D}, k={K})",
        fontsize=12, fontweight="bold",
    )

    # Annotate cells
    for i in range(3):
        for j in range(3):
            val = heatmap[i, j]
            is_correct = (FAMILIES[i] == TRUE_FX and FAMILIES[j] == TRUE_FY)
            color = "white" if (val > (heatmap.max()*0.6)) else "black"
            text  = f"{val:.3f}"
            if is_correct:
                text += "\n[Proposed]"
            ax4.text(j, i, text, ha="center", va="center",
                     fontsize=10, color=color,
                     fontweight="bold" if is_correct else "normal")
            # Box around correct cell
            if is_correct:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                      linewidth=2.5, edgecolor="#2878b5",
                                      facecolor="none")
                ax4.add_patch(rect)

    plt.tight_layout()
    fig4.savefig(_RES / "fig_dual_4_heatmap.pdf", dpi=300, bbox_inches="tight")
    fig4.savefig(_RES / "fig_dual_4_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"Figure 4 saved: {_RES / 'fig_dual_4_heatmap.png'}")

    # ── Figure 5: Bar Chart (Proposed vs Mismatch vs Ablation) ────────
    # Build comparison bars
    bar_data = []

    # Proposed (correct)
    bar_data.append({
        "label": f"Proposed\n(Pois, Bern)",
        "mean":  proposed_mean,
        "std":   proposed_std,
        "color": "#2878b5",
        "group": "proposed",
    })

    # X-mismatch (fix Y=correct Bernoulli, vary X)
    for fx in FAMILIES:
        if fx == TRUE_FX:
            continue
        row = grid_df[(grid_df.model_fx == fx) & (grid_df.model_fy == TRUE_FY)]
        bar_data.append({
            "label": f"X={FAM_LABEL[fx]}\n(Y=Bern)",
            "mean":  float(row["rmse_Z_mean"]),
            "std":   float(row["rmse_Z_std"]),
            "color": "#e06c75",
            "group": "x_mismatch",
        })

    # Y-mismatch (fix X=correct Poisson, vary Y)
    for fy in FAMILIES:
        if fy == TRUE_FY:
            continue
        row = grid_df[(grid_df.model_fx == TRUE_FX) & (grid_df.model_fy == fy)]
        bar_data.append({
            "label": f"Y={FAM_LABEL[fy]}\n(X=Pois)",
            "mean":  float(row["rmse_Z_mean"]),
            "std":   float(row["rmse_Z_std"]),
            "color": "#f39c12",
            "group": "y_mismatch",
        })

    # Ablation: no X (fix_x=True), no Y (fix_w=True)
    abl_df = agg[(agg["fix_w"] | agg["fix_x"])].copy()
    for label, fw, fxb in ablation_conditions:
        row = abl_df[(abl_df["fix_w"] == fw) & (abl_df["fix_x"] == fxb)]
        if len(row):
            short_label = "No X\n(fix_x)" if fxb else "No Y\n(fix_w)"
            bar_data.append({
                "label": short_label,
                "mean":  float(row["rmse_Z_mean"].iloc[0]),
                "std":   float(row["rmse_Z_std"].iloc[0]),
                "color": "#27ae60",
                "group": "ablation",
            })

    fig5, ax5 = plt.subplots(figsize=(10, 5.5))
    labels  = [b["label"] for b in bar_data]
    means   = [b["mean"]  for b in bar_data]
    stds    = [b["std"]   for b in bar_data]
    colors  = [b["color"] for b in bar_data]
    x_pos   = np.arange(len(bar_data))

    bars = ax5.bar(x_pos, means, yerr=stds, capsize=5,
                   color=colors, alpha=0.85, width=0.6,
                   edgecolor="black", linewidth=0.7,
                   error_kw={"elinewidth": 1.5})

    # Value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + s + 0.005,
                 f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    # Horizontal reference line (proposed)
    ax5.axhline(proposed_mean, color="#2878b5", linestyle="--",
                linewidth=1.5, label=f"Proposed RMSE(Z)={proposed_mean:.3f}")

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(labels, fontsize=10)
    ax5.set_ylabel("RMSE(Z)  (lower is better)", fontsize=12)
    ax5.set_title(
        f"Fig 5: RMSE(Z) Comparison — Proposed vs Mismatch vs Ablation\n"
        f"True data: X=Poisson, Y=Bernoulli  (n={N}, d={D}, k={K}, "
        f"{N_TRIALS} trials, mean+-std)",
        fontsize=11, fontweight="bold",
    )
    ax5.legend(fontsize=10)
    ax5.grid(True, axis="y", alpha=0.4)
    ax5.set_ylim(0, max(means) * 1.25)

    # Group annotations
    group_bounds = {
        "proposed":  (0, 0),
        "x_mismatch": (1, 2),
        "y_mismatch": (3, 4),
        "ablation":   (5, 6),
    }
    group_labels = {
        "proposed":   "Proposed",
        "x_mismatch": "X mismatch",
        "y_mismatch": "Y mismatch",
        "ablation":   "Ablation",
    }
    group_colors = {
        "proposed":   "#2878b5",
        "x_mismatch": "#e06c75",
        "y_mismatch": "#f39c12",
        "ablation":   "#27ae60",
    }

    # Map bar_data groups to actual x positions
    groups_seen = {}
    for xi, b in enumerate(bar_data):
        g = b["group"]
        if g not in groups_seen:
            groups_seen[g] = [xi, xi]
        else:
            groups_seen[g][1] = xi

    for g, (x0, x1) in groups_seen.items():
        y_bracket = ax5.get_ylim()[0] - 0.002
        ax5.annotate(
            group_labels[g],
            xy=((x0 + x1) / 2, -0.02),
            xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=9.5,
            color=group_colors[g], fontweight="bold",
        )

    plt.tight_layout()
    fig5.savefig(_RES / "fig_dual_5_barchart.pdf", dpi=300, bbox_inches="tight")
    fig5.savefig(_RES / "fig_dual_5_barchart.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    print(f"Figure 5 saved: {_RES / 'fig_dual_5_barchart.png'}")

    total_t = time.perf_counter() - t_start
    print(f"\nTotal: {total_t/60:.1f} min")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n=== 3x3 Mismatch Matrix (RMSE(Z) mean) ===")
    print(f"{'':12s}", end="")
    for fy in FAMILIES:
        print(f"  Y={FAM_LABEL[fy]:10s}", end="")
    print()
    for i, fx in enumerate(FAMILIES):
        mark = " *" if fx == TRUE_FX else "  "
        print(f"X={FAM_LABEL[fx]:10s}", end="")
        for j, fy in enumerate(FAMILIES):
            val = heatmap[i, j]
            correct = (fx == TRUE_FX and fy == TRUE_FY)
            tag = " [*]" if correct else "    "
            print(f"  {val:.3f}{tag}", end="")
        print()

    print("\n=== Bar Chart Summary ===")
    for b in bar_data:
        print(f"  {b['label'].replace(chr(10),' '):<25s}: {b['mean']:.4f} +- {b['std']:.4f}")

    print("\n=== Mismatch penalty vs Proposed ===")
    for b in bar_data[1:]:
        ratio = b["mean"] / proposed_mean
        print(f"  {b['label'].replace(chr(10),' '):<25s}: {ratio:.2f}x proposed")


if __name__ == "__main__":
    main()
