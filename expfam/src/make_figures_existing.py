"""
既存 CSV (old model, 0.5 あり) から全条件の図を作成する。

出力先: expfam/figures/distribution_mismatch_fixed/
  heatmap_rmse_scenario_{A/B/C}_old.png
  heatmap_ratio_scenario_{A/B/C}_old.png
  boxplot_scenario_{A/B/C}_old.png
  bar_oracle_conv_worst_old.png
  bar_ratio_per_scenario_old.png
"""

import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_FIG  = _ROOT / "expfam" / "figures" / "distribution_mismatch_fixed"
_RES  = _ROOT / "expfam" / "results"
_FIG.mkdir(parents=True, exist_ok=True)

FAMILIES    = ["gaussian", "bernoulli", "poisson"]
FAM_LABEL   = {"gaussian": "Gaussian", "bernoulli": "Bernoulli", "poisson": "Poisson"}
FAM_SHORT   = {"gaussian": "Gauss",    "bernoulli": "Bern",      "poisson": "Pois"}

SCENARIOS = [
    ("A", "poisson",   "bernoulli"),
    ("B", "gaussian",  "poisson"),
    ("C", "bernoulli", "gaussian"),
]

ORACLE_COLOR = "#2878b5"
CONV_COLOR   = "#e06c75"
WORST_COLOR  = "#c0392b"
NEUTRAL_COLOR= "#95a5a6"

# ── Load and reshape existing data ────────────────────────────────────

all_dfs = []
for scen_tag, true_fx, true_fy in SCENARIOS:
    df = pd.read_csv(_RES / f"exp_scenario_{scen_tag}_exp4_mismatch.csv")
    # Only 3×3 grid (no ablation)
    grid = df[~df["fix_w"] & ~df["fix_x"]].copy()
    grid["scenario"]  = scen_tag
    grid["true_x"]    = true_fx
    grid["true_y"]    = true_fy
    grid["is_oracle"] = (grid["model_fx"] == true_fx) & (grid["model_fy"] == true_fy)
    grid["is_conv"]   = (grid["model_fx"] == "gaussian") & (grid["model_fy"] == "bernoulli")
    all_dfs.append(grid)

df_all = pd.concat(all_dfs, ignore_index=True)

summary_rows = []
for scen_tag, true_fx, true_fy in SCENARIOS:
    sub = df_all[df_all["scenario"] == scen_tag]
    oracle_mean = sub[sub["is_oracle"]]["rmse_Z"].mean()
    for fx in FAMILIES:
        for fy in FAMILIES:
            cond = sub[(sub["model_fx"] == fx) & (sub["model_fy"] == fy)]
            m = cond["rmse_Z"].mean()
            s = cond["rmse_Z"].std()
            summary_rows.append({
                "scenario": scen_tag, "true_x": true_fx, "true_y": true_fy,
                "est_x": fx, "est_y": fy,
                "mean": m, "std": s,
                "ratio": m / oracle_mean,
                "oracle_mean": oracle_mean,
                "is_oracle": (fx == true_fx and fy == true_fy),
                "is_conv":   (fx == "gaussian" and fy == "bernoulli"),
            })
summary = pd.DataFrame(summary_rows)

# Print full tables
for scen_tag, true_fx, true_fy in SCENARIOS:
    sub_s = summary[summary["scenario"] == scen_tag]
    oracle = float(sub_s[sub_s["is_oracle"]]["mean"].iloc[0])
    print(f"\n{'='*68}")
    print(f"Scenario {scen_tag}: True X={FAM_LABEL[true_fx]}, True Y={FAM_LABEL[true_fy]}")
    print(f"Oracle RMSE(Z) = {oracle:.5f}")
    print(f"\n  RMSE(Z) mean±std  (* = Oracle, † = Conv-like)")
    hdr = f"  {'est_X \\ est_Y':<12} " + "".join(f"  {FAM_LABEL[f]:>14}" for f in FAMILIES)
    print(hdr)
    for fx in FAMILIES:
        row_str = f"  {FAM_LABEL[fx]:<12} "
        for fy in FAMILIES:
            r = sub_s[(sub_s["est_x"] == fx) & (sub_s["est_y"] == fy)].iloc[0]
            mark = "*" if r["is_oracle"] else ("†" if r["is_conv"] else " ")
            row_str += f"  {r['mean']:6.4f}±{r['std']:.4f}{mark} "
        print(row_str)
    print(f"\n  悪化倍率 (÷ {oracle:.5f})")
    print(hdr)
    for fx in FAMILIES:
        row_str = f"  {FAM_LABEL[fx]:<12} "
        for fy in FAMILIES:
            r = sub_s[(sub_s["est_x"] == fx) & (sub_s["est_y"] == fy)].iloc[0]
            mark = "*" if r["is_oracle"] else ("†" if r["is_conv"] else " ")
            row_str += f"  {r['ratio']:8.3f}x{mark}         "
        print(row_str)
    worst = sub_s.loc[sub_s["ratio"].idxmax()]
    conv  = sub_s[sub_s["is_conv"]].iloc[0]
    print(f"\n  Worst: X={FAM_LABEL[worst.est_x]}, Y={FAM_LABEL[worst.est_y]} → {worst.ratio:.3f}x")
    print(f"  Conv-like: X=Gaussian, Y=Bernoulli → {conv.ratio:.3f}x  (RMSE={conv['mean']:.4f})")

# ── Figure 1: Heatmaps ────────────────────────────────────────────────
for scen_tag, true_fx, true_fy in SCENARIOS:
    sub_s = summary[summary["scenario"] == scen_tag]
    oracle_i = FAMILIES.index(true_fx)
    oracle_j = FAMILIES.index(true_fy)
    conv_i   = FAMILIES.index("gaussian")
    conv_j   = FAMILIES.index("bernoulli")

    for kind, col, title_sfx, fmt in [
        ("rmse",  "mean",  "RMSE(Z)",   ".4f"),
        ("ratio", "ratio", "Degradation Ratio", ".2f"),
    ]:
        hmap = np.zeros((3, 3))
        for i, fx in enumerate(FAMILIES):
            for j, fy in enumerate(FAMILIES):
                r = sub_s[(sub_s["est_x"] == fx) & (sub_s["est_y"] == fy)].iloc[0]
                hmap[i, j] = r[col]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        vmin = hmap.min() * 0.9 if kind == "rmse" else max(hmap.min() * 0.9, 0)
        im = ax.imshow(hmap, cmap="RdYlGn_r", aspect="auto",
                       vmin=vmin, vmax=hmap.max() * 1.05)
        plt.colorbar(im, ax=ax, label=title_sfx)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=11)
        ax.set_yticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=11)
        ax.set_xlabel("推定 Y 分布族", fontsize=12)
        ax.set_ylabel("推定 X 分布族", fontsize=12)
        ax.set_title(
            f"Scenario {scen_tag}  {title_sfx}  [Old model: 0.5 あり]\n"
            f"True: X={FAM_LABEL[true_fx]}, Y={FAM_LABEL[true_fy]}  "
            f"(n=150, d=15, k=3, 10 trials)",
            fontsize=10, fontweight="bold",
        )
        worst_i, worst_j = np.unravel_index(np.argmax(hmap), hmap.shape)
        for i in range(3):
            for j in range(3):
                val = hmap[i, j]
                txt = format(val, fmt)
                is_o = (i == oracle_i and j == oracle_j)
                is_c = (i == conv_i   and j == conv_j)
                is_w = (i == worst_i  and j == worst_j)
                color = "white" if val > hmap.max() * 0.6 else "black"
                if is_o:   txt += "\n[Oracle]"
                elif is_c: txt += "\n[Conv]"
                elif is_w: txt += "\n[Worst]"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color=color,
                        fontweight="bold" if (is_o or is_w) else "normal")
                if is_o:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               linewidth=2.5, edgecolor=ORACLE_COLOR,
                                               facecolor="none"))
                if is_c:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               linewidth=2.0, edgecolor=CONV_COLOR,
                                               facecolor="none", linestyle="--"))
        plt.tight_layout()
        fname = f"heatmap_{kind}_scenario_{scen_tag}_old"
        for ext in ("pdf", "png"):
            fig.savefig(_FIG / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}.png")

# ── Figure 2: Box plots ───────────────────────────────────────────────
for scen_tag, true_fx, true_fy in SCENARIOS:
    sub_df = df_all[df_all["scenario"] == scen_tag].copy()
    labels_sorted = [
        f"{FAM_SHORT[fx]}-{FAM_SHORT[fy]}"
        for fx in FAMILIES for fy in FAMILIES
    ]
    sub_df["label"] = (
        sub_df["model_fx"].map(FAM_SHORT) + "-" +
        sub_df["model_fy"].map(FAM_SHORT)
    )
    data_for_box = [
        sub_df[sub_df["label"] == lab]["rmse_Z"].values
        for lab in labels_sorted
    ]
    colors_box = []
    for fx in FAMILIES:
        for fy in FAMILIES:
            if fx == true_fx and fy == true_fy:
                colors_box.append(ORACLE_COLOR)
            elif fx == "gaussian" and fy == "bernoulli":
                colors_box.append(CONV_COLOR)
            else:
                colors_box.append(NEUTRAL_COLOR)

    fig, ax = plt.subplots(figsize=(12, 5))
    bps = ax.boxplot(data_for_box, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2),
                     whiskerprops=dict(linewidth=1.2))
    for patch, c in zip(bps["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE(Z)", fontsize=12)
    ax.set_title(
        f"Scenario {scen_tag}  RMSE(Z) distribution  [Old model, 0.5 あり]\n"
        f"True: X={FAM_LABEL[true_fx]}, Y={FAM_LABEL[true_fy]}  "
        f"(Blue=Oracle, Red=Conv-like, 10 trials)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.4)
    oracle_line = float(
        sub_df[sub_df["is_oracle"]]["rmse_Z"].mean()
    )
    ax.axhline(oracle_line, color=ORACLE_COLOR, ls="--", lw=1.5,
               label=f"Oracle mean={oracle_line:.4f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"boxplot_scenario_{scen_tag}_old"
    for ext in ("pdf", "png"):
        fig.savefig(_FIG / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}.png")

# ── Figure 3: Oracle vs Conv vs Worst bar (all scenarios) ────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle("Oracle / Conv-like / Worst RMSE(Z)  [Old model, 0.5 あり]",
             fontsize=12, fontweight="bold")

for ax, (scen_tag, true_fx, true_fy) in zip(axes, SCENARIOS):
    sub_s = summary[summary["scenario"] == scen_tag]
    oracle = sub_s[sub_s["is_oracle"]].iloc[0]
    conv   = sub_s[sub_s["is_conv"]].iloc[0]
    worst  = sub_s.loc[sub_s["ratio"].idxmax()]

    labels = ["Oracle\n正解", "Conv-like\nGauss-Bern", f"Worst\n{FAM_SHORT[worst.est_x]}-{FAM_SHORT[worst.est_y]}"]
    means  = [oracle["mean"], conv["mean"], worst["mean"]]
    stds   = [oracle["std"],  conv["std"],  worst["std"]]
    colors = [ORACLE_COLOR, CONV_COLOR, WORST_COLOR]

    bars = ax.bar(range(3), means, yerr=stds, color=colors, alpha=0.85,
                  capsize=5, error_kw={"elinewidth": 1.5},
                  edgecolor="black", linewidth=0.7)
    for bar, m, s, r in zip(bars, means, stds, [1.0, conv["ratio"], worst["ratio"]]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + max(means) * 0.02,
                f"{m:.4f}\n({r:.2f}x)", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE(Z)", fontsize=11)
    ax.set_title(
        f"Scenario {scen_tag}\n"
        f"X={FAM_LABEL[true_fx][:4]}, Y={FAM_LABEL[true_fy][:4]}",
        fontsize=10, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) * 1.5)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(_FIG / f"bar_oracle_conv_worst_old.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: bar_oracle_conv_worst_old.png")

# ── Figure 4: Degradation ratio bar (all conditions per scenario) ─────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)
fig.suptitle("悪化倍率 (RMSE ÷ Oracle)  [Old model, 0.5 あり]",
             fontsize=12, fontweight="bold")

for ax, (scen_tag, true_fx, true_fy) in zip(axes, SCENARIOS):
    sub_s = summary[summary["scenario"] == scen_tag]
    cond_labels = [f"{FAM_SHORT[fx]}\n{FAM_SHORT[fy]}"
                   for fx in FAMILIES for fy in FAMILIES]
    ratios = []
    stds   = []
    colors_bar = []
    for fx in FAMILIES:
        for fy in FAMILIES:
            r = sub_s[(sub_s["est_x"] == fx) & (sub_s["est_y"] == fy)].iloc[0]
            ratios.append(r["ratio"])
            stds.append(r["std"] / r["oracle_mean"])
            if r["is_oracle"]:
                colors_bar.append(ORACLE_COLOR)
            elif r["is_conv"]:
                colors_bar.append(CONV_COLOR)
            else:
                colors_bar.append(NEUTRAL_COLOR)

    x_pos = np.arange(9)
    bars = ax.bar(x_pos, ratios, color=colors_bar, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.5)
    for bar, r in zip(bars, ratios):
        if r > 1.5:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(ratios) * 0.02,
                    f"{r:.1f}x", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cond_labels, fontsize=7)
    ax.set_ylabel("Degradation Ratio", fontsize=11)
    ax.set_title(
        f"Scenario {scen_tag}\nX={FAM_LABEL[true_fx][:4]}, Y={FAM_LABEL[true_fy][:4]}",
        fontsize=10, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)

# Legend
from matplotlib.patches import Patch
handles = [
    Patch(color=ORACLE_COLOR, label="Oracle (正解)"),
    Patch(color=CONV_COLOR,   label="Conv-like (Gauss-Bern)"),
    Patch(color=NEUTRAL_COLOR,label="Other"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(_FIG / f"bar_degradation_ratio_old.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: bar_degradation_ratio_old.png")

print("\nAll figures saved to:", _FIG)
