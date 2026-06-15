"""
Distribution-mismatch experiment using DualExpFamLSMFixed (Y-side 0.5 removed).

Runs 3 scenarios × 9 distribution combinations × N_TRIALS trials.
Results saved to results/distribution_mismatch_fixed/
Figures saved to figures/distribution_mismatch_fixed/

Usage:
    cd expfam/src
    python run_mismatch_fixed.py
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

_SRC = Path(__file__).parent
_ROOT = _SRC.parent.parent
_RES  = _SRC.parent / "results" / "distribution_mismatch_fixed"
_FIG  = _SRC.parent / "figures" / "distribution_mismatch_fixed"
_RES.mkdir(parents=True, exist_ok=True)
_FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from data_generator_expfam import generate_dual_data      # noqa
from model_dual_expfam_fixed import DualExpFamLSMFixed    # noqa

# ── helpers (copied from utils_expfam to avoid importing the old model) ──

def calc_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def procrustes_rotation(A_est, A_true):
    k_min = min(A_est.shape[1], A_true.shape[1])
    M = A_est[:, :k_min].T @ A_true[:, :k_min]
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt, k_min


def run_em_fixed(X, Y, true_params, family_x, family_y, k,
                 L=5, num_iter=8, seed=42,
                 fix_w=False, fix_x=False):
    """
    MC-EM using DualExpFamLSMFixed. Returns metrics dict.
    Fallback: halve newton_alpha on NaN (up to 2 retries).
    """
    n, d = X.shape
    max_retries = 2

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)
        rng = np.random.default_rng(seed + retry * 1000)

        model = DualExpFamLSMFixed(
            n=n, d=d, k=k, L=L,
            family_x=family_x, family_y=family_y,
        )
        model.initialize_params(true_params=true_params,
                                seed=seed + retry * 1000)

        # ── Informed init: Y side ────────────────────────────────────
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if family_y == "bernoulli":
            density = float(np.clip(np.mean(Y), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"]  = 0.5
        elif family_y == "poisson":
            upper = Y[upper_mask]
            mean_cnt = float(upper[upper > 0].mean()) if np.any(upper > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"]  = 0.1 / (2 ** retry)
        else:  # gaussian
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"]  = 0.5
            model.sigma_y = float(max(upper_vals.std(), 0.01))

        # ── Informed init: X side ────────────────────────────────────
        if family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        Z      = model.params["Z"].copy()
        F      = model.params["F"].copy()
        sigma  = model.params["sigma"].copy()
        w0     = model.params["w0"]
        w      = model.params["w"]
        var_z  = model.params["var_z"]

        if fix_w:
            w = 0.0
            model.params["w"] = 0.0
        if fix_x:
            F = np.zeros((d, k))
            model.params["F"] = np.zeros((d, k))

        Z_prev    = Z.copy()
        nan_count = 0
        Q_history = []

        for iteration in range(1, num_iter + 1):
            # E-step
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(dict(Z=Z.copy(), F=F,
                                         sigma=sigma, w0=w0, w=w))
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            if (np.any(np.isnan(Z_samples)) or
                    np.any(np.isinf(Z_samples))):
                nan_count += 1
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            # M-step
            if not fix_x:
                F     = model.calc_F(X, Z_samples)
                sigma = model.calc_sigma(X, Z_samples, F)
            w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            if not fix_w:
                w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

            # Q (no factorial)
            Q_val = 0.0
            for l in range(L):
                Z_l = Z_samples[:, :, l]
                lnpZ = -(n * k / 2) * np.log(2 * np.pi * var_z) \
                       - (1 / (2 * var_z)) * np.sum(Z_l ** 2)
                lnpX = float(model.calc_log_likelihood_X(
                    X, Z_samples[:, :, l:l+1], F))
                lnpY = float(model.calc_log_likelihood_Y(
                    Y, Z_samples[:, :, l:l+1], w0, w))
                Q_val += lnpZ + lnpX + lnpY
            Q_history.append(Q_val / L)

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

    # ── Metrics ──────────────────────────────────────────────────────
    Z_est = Z_samples[:, :, -1]
    mu_x_est = model._mean_function_x(Z_est @ F.T)
    rmse_X = calc_rmse(X, mu_x_est)

    R, k_min = procrustes_rotation(Z_est, true_params["Z"])
    Z_rot    = Z_est[:, :k_min] @ R
    rmse_Z   = calc_rmse(true_params["Z"][:, :k_min], Z_rot)

    F_aligned = F[:, :k_min] @ R
    rmse_F    = calc_rmse(true_params["F"][:, :k_min], F_aligned)

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    eta_y_est  = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y_est   = model._mean_function(eta_y_est)
    eta_y_true = float(true_params["w0"]) + float(true_params["w"]) * (
        true_params["Z"] @ true_params["Z"].T)
    mu_y_true  = model._mean_function(eta_y_true)
    rmse_Y     = calc_rmse(mu_y_est[upper_mask], mu_y_true[upper_mask])

    return {
        "rmse_Z": rmse_Z, "rmse_F": rmse_F,
        "rmse_X": rmse_X, "rmse_Y": rmse_Y,
        "Q_final": Q_history[-1] if Q_history else float("nan"),
        "nan_occurred": nan_occurred, "nan_count": nan_count,
        "w0_est": float(w0), "w_est": float(w),
    }


# ── Experiment settings ───────────────────────────────────────────────

SCENARIOS = [
    ("A", "poisson",   "bernoulli"),
    ("B", "gaussian",  "poisson"),
    ("C", "bernoulli", "gaussian"),
]
FAMILIES    = ["gaussian", "bernoulli", "poisson"]
FAM_LABEL   = {"gaussian": "Gaussian", "bernoulli": "Bernoulli", "poisson": "Poisson"}
FAM_SHORT   = {"gaussian": "Gauss",    "bernoulli": "Bern",      "poisson": "Pois"}

N, D, K_TRUE = 150, 15, 3
N_TRIALS      = 10
L, NITER      = 5, 8
BASE_SEED     = 6000   # same seed as existing exp4 for reproducibility


def run_all():
    records = []
    total_runs = len(SCENARIOS) * len(FAMILIES) ** 2 * N_TRIALS
    done = 0
    t_start = time.perf_counter()

    for scen_tag, true_fx, true_fy in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Scenario {scen_tag}: X={FAM_LABEL[true_fx]}, Y={FAM_LABEL[true_fy]}")
        print(f"{'='*60}")

        for fx_m in FAMILIES:
            for fy_m in FAMILIES:
                is_oracle = (fx_m == true_fx and fy_m == true_fy)
                is_conv   = (fx_m == "gaussian" and fy_m == "bernoulli")

                for trial in range(N_TRIALS):
                    done += 1
                    data_seed  = BASE_SEED + trial * 100
                    model_seed = data_seed + 50

                    data = generate_dual_data(
                        n=N, d=D, k=K_TRUE,
                        seed=data_seed,
                        family_x=true_fx,
                        family_y=true_fy,
                    )

                    res = run_em_fixed(
                        X=data["X"], Y=data["Y"],
                        true_params=data,
                        family_x=fx_m, family_y=fy_m,
                        k=K_TRUE, L=L, num_iter=NITER,
                        seed=model_seed,
                    )

                    elapsed = (time.perf_counter() - t_start) / 60
                    print(
                        f"  [{done:4d}/{total_runs}] "
                        f"Scen={scen_tag} "
                        f"X={FAM_SHORT[fx_m]:5s} Y={FAM_SHORT[fy_m]:5s} "
                        f"t={trial} "
                        f"RMSE(Z)={res['rmse_Z']:.4f} "
                        f"{'[ORACLE]' if is_oracle else ''}"
                        f"{'[CONV]' if is_conv else ''} "
                        f"  [{elapsed:.1f}min]"
                    )

                    records.append({
                        "scenario":             scen_tag,
                        "true_x_dist":          true_fx,
                        "true_y_dist":          true_fy,
                        "estimated_x_dist":     fx_m,
                        "estimated_y_dist":     fy_m,
                        "trial":                trial,
                        "seed_data":            data_seed,
                        "seed_model":           model_seed,
                        "n":                    N,
                        "d":                    D,
                        "k_true":               K_TRUE,
                        "k_est":                K_TRUE,
                        "L":                    L,
                        "iterations":           NITER,
                        "rmse_z":               res["rmse_Z"],
                        "rmse_f":               res["rmse_F"],
                        "rmse_x":               res["rmse_X"],
                        "rmse_y":               res["rmse_Y"],
                        "q_value":              res["Q_final"],
                        "w0_est":               res["w0_est"],
                        "w_est":                res["w_est"],
                        "nan_occurred":         res["nan_occurred"],
                        "nan_count":            res["nan_count"],
                        "is_oracle_condition":  is_oracle,
                        "is_conventional_like_condition": is_conv,
                    })

    df = pd.DataFrame(records)
    out_csv = _RES / "mismatch_fixed_all_trials.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df)} rows)")
    return df


def make_summary(df):
    """Compute per-condition mean/std and degradation ratio."""
    rows = []
    for scen_tag, true_fx, true_fy in SCENARIOS:
        sub = df[df["scenario"] == scen_tag]
        oracle_mean = float(
            sub[sub["is_oracle_condition"]]["rmse_z"].mean()
        )

        for fx_m in FAMILIES:
            for fy_m in FAMILIES:
                cond = sub[
                    (sub["estimated_x_dist"] == fx_m) &
                    (sub["estimated_y_dist"] == fy_m)
                ]
                m = float(cond["rmse_z"].mean())
                s = float(cond["rmse_z"].std())
                rows.append({
                    "scenario":          scen_tag,
                    "true_x":            true_fx,
                    "true_y":            true_fy,
                    "est_x":             fx_m,
                    "est_y":             fy_m,
                    "rmse_z_mean":       m,
                    "rmse_z_std":        s,
                    "degradation_ratio": m / oracle_mean if oracle_mean > 0 else float("nan"),
                    "oracle_rmse":       oracle_mean,
                    "is_oracle":         (fx_m == true_fx and fy_m == true_fy),
                    "is_conv":           (fx_m == "gaussian" and fy_m == "bernoulli"),
                })
    return pd.DataFrame(rows)


def make_figures(df, summary):
    """Generate heatmaps, bar charts, and box plots."""
    # ── color scheme ──
    ORACLE_COLOR = "#2878b5"
    CONV_COLOR   = "#e06c75"
    WORST_COLOR  = "#c0392b"

    for scen_tag, true_fx, true_fy in SCENARIOS:
        sub_s = summary[summary["scenario"] == scen_tag]

        # ── 1. RMSE heatmap ──────────────────────────────────────────
        heatmap_rmse  = np.zeros((3, 3))
        heatmap_ratio = np.zeros((3, 3))
        oracle_rmse   = float(sub_s[sub_s["is_oracle"]]["rmse_z_mean"].iloc[0])

        for i, fx in enumerate(FAMILIES):
            for j, fy in enumerate(FAMILIES):
                row = sub_s[
                    (sub_s["est_x"] == fx) & (sub_s["est_y"] == fy)
                ]
                heatmap_rmse[i, j]  = float(row["rmse_z_mean"].iloc[0])
                heatmap_ratio[i, j] = float(row["degradation_ratio"].iloc[0])

        for kind, hmap, title_suffix, fmt in [
            ("rmse",  heatmap_rmse,  "RMSE(Z)",      ".4f"),
            ("ratio", heatmap_ratio, "悪化倍率 (ratio)", ".2f"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            im = ax.imshow(hmap, cmap="RdYlGn_r", aspect="auto",
                           vmin=hmap.min() * 0.95,
                           vmax=hmap.max() * 1.05)
            plt.colorbar(im, ax=ax, label=title_suffix)
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=11)
            ax.set_yticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=11)
            ax.set_xlabel("推定 Y 分布族", fontsize=12)
            ax.set_ylabel("推定 X 分布族", fontsize=12)
            ax.set_title(
                f"Scenario {scen_tag}  {title_suffix}  [Fixed model]\n"
                f"True: X={FAM_LABEL[true_fx]}, Y={FAM_LABEL[true_fy]}  "
                f"(n={N}, d={D}, k={K_TRUE}, {N_TRIALS} trials)",
                fontsize=10, fontweight="bold",
            )

            oracle_i = FAMILIES.index(true_fx)
            oracle_j = FAMILIES.index(true_fy)
            conv_i   = FAMILIES.index("gaussian")
            conv_j   = FAMILIES.index("bernoulli")
            worst_i, worst_j = np.unravel_index(np.argmax(hmap), hmap.shape)

            for ii in range(3):
                for jj in range(3):
                    val = hmap[ii, jj]
                    txt = format(val, fmt)
                    is_o = (ii == oracle_i and jj == oracle_j)
                    is_c = (ii == conv_i   and jj == conv_j)
                    is_w = (ii == worst_i  and jj == worst_j)
                    color = "white" if val > hmap.max() * 0.6 else "black"
                    if is_o:
                        txt += "\n[Oracle]"
                    elif is_c:
                        txt += "\n[Conv]"
                    elif is_w:
                        txt += "\n[Worst]"
                    ax.text(jj, ii, txt, ha="center", va="center",
                            fontsize=9, color=color,
                            fontweight="bold" if (is_o or is_w) else "normal")
                    if is_o:
                        rect = plt.Rectangle((jj-0.5, ii-0.5), 1, 1,
                                             linewidth=2.5, edgecolor=ORACLE_COLOR,
                                             facecolor="none")
                        ax.add_patch(rect)
                    if is_c:
                        rect = plt.Rectangle((jj-0.5, ii-0.5), 1, 1,
                                             linewidth=2.0, edgecolor=CONV_COLOR,
                                             facecolor="none", linestyle="--")
                        ax.add_patch(rect)

            plt.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(
                    _FIG / f"heatmap_{kind}_scenario_{scen_tag}.{ext}",
                    dpi=200, bbox_inches="tight",
                )
            plt.close(fig)
            print(f"  Fig saved: heatmap_{kind}_scenario_{scen_tag}.png")

        # ── 2. RMSE box plot ─────────────────────────────────────────
        sub_df = df[df["scenario"] == scen_tag].copy()
        sub_df["label"] = (
            sub_df["estimated_x_dist"].map(FAM_SHORT) + "-" +
            sub_df["estimated_y_dist"].map(FAM_SHORT)
        )
        labels_sorted = [
            f"{FAM_SHORT[fx]}-{FAM_SHORT[fy]}"
            for fx in FAMILIES for fy in FAMILIES
        ]
        data_for_box = [
            sub_df[sub_df["label"] == lab]["rmse_z"].values
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
                    colors_box.append("#95a5a6")

        fig, ax = plt.subplots(figsize=(12, 5))
        bps = ax.boxplot(data_for_box, patch_artist=True,
                         medianprops=dict(color="black", linewidth=2),
                         whiskerprops=dict(linewidth=1.2),
                         capprops=dict(linewidth=1.2))
        for patch, c in zip(bps["boxes"], colors_box):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, 10))
        ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("RMSE(Z)", fontsize=12)
        ax.set_title(
            f"Scenario {scen_tag}  RMSE(Z) distribution  [Fixed model]\n"
            f"True: X={FAM_LABEL[true_fx]}, Y={FAM_LABEL[true_fy]}  "
            f"(Oracle=blue, Conv-like=red, {N_TRIALS} trials)",
            fontsize=10, fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.4)
        oracle_line = float(
            sub_df[sub_df["is_oracle_condition"]]["rmse_z"].mean()
        )
        ax.axhline(oracle_line, color=ORACLE_COLOR, ls="--", lw=1.5,
                   label=f"Oracle mean={oracle_line:.4f}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(
                _FIG / f"boxplot_scenario_{scen_tag}.{ext}",
                dpi=200, bbox_inches="tight",
            )
        plt.close(fig)
        print(f"  Fig saved: boxplot_scenario_{scen_tag}.png")

    # ── 3. Oracle vs Conv vs Worst bar chart (all scenarios) ─────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos  = np.arange(len(SCENARIOS))
    width  = 0.25

    oracle_vals = []
    conv_vals   = []
    worst_vals  = []
    oracle_stds = []
    conv_stds   = []
    worst_stds  = []

    for scen_tag, _, _ in SCENARIOS:
        sub_s = summary[summary["scenario"] == scen_tag]
        oracle_row = sub_s[sub_s["is_oracle"]]
        conv_row   = sub_s[sub_s["is_conv"]]
        worst_row  = sub_s.loc[sub_s["rmse_z_mean"].idxmax()]

        oracle_vals.append(float(oracle_row["rmse_z_mean"].iloc[0]))
        conv_vals.append(  float(conv_row["rmse_z_mean"].iloc[0]))
        worst_vals.append( float(worst_row["rmse_z_mean"]))
        oracle_stds.append(float(oracle_row["rmse_z_std"].iloc[0]))
        conv_stds.append(  float(conv_row["rmse_z_std"].iloc[0]))
        worst_stds.append( float(worst_row["rmse_z_std"]))

    bars_o = ax.bar(x_pos - width, oracle_vals, width, yerr=oracle_stds,
                    color=ORACLE_COLOR, alpha=0.85, label="Oracle (正解)",
                    capsize=4, error_kw={"elinewidth": 1.5})
    bars_c = ax.bar(x_pos,         conv_vals,   width, yerr=conv_stds,
                    color=CONV_COLOR,   alpha=0.85, label="Conv-like (Gauss-Bern)",
                    capsize=4, error_kw={"elinewidth": 1.5})
    bars_w = ax.bar(x_pos + width, worst_vals,  width, yerr=worst_stds,
                    color=WORST_COLOR,  alpha=0.85, label="Worst (最悪)",
                    capsize=4, error_kw={"elinewidth": 1.5})

    for bars, vals in [(bars_o, oracle_vals), (bars_c, conv_vals),
                       (bars_w, worst_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Scenario {s[0]}\n(X={FAM_LABEL[s[1]][:4]}, Y={FAM_LABEL[s[2]][:4]})"
                        for s in SCENARIOS], fontsize=10)
    ax.set_ylabel("RMSE(Z)", fontsize=12)
    ax.set_title("Oracle vs Conv-like vs Worst RMSE(Z)  [Fixed model]",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_FIG / f"bar_oracle_conv_worst.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Fig saved: bar_oracle_conv_worst.png")


def print_tables(summary):
    """Print full 3×3 tables for each scenario."""
    for scen_tag, true_fx, true_fy in SCENARIOS:
        sub = summary[summary["scenario"] == scen_tag]
        oracle_rmse = float(sub[sub["is_oracle"]]["rmse_z_mean"].iloc[0])

        print(f"\n{'='*70}")
        print(f"Scenario {scen_tag}: True X={FAM_LABEL[true_fx]}, True Y={FAM_LABEL[true_fy]}")
        print(f"Oracle RMSE(Z) = {oracle_rmse:.5f}")

        print(f"\n  RMSE(Z) mean [est_X rows × est_Y cols]:")
        header = f"  {'est_X \\ est_Y':<14} " + " ".join(f"{FAM_LABEL[f]:>12}" for f in FAMILIES)
        print(header)
        for fx in FAMILIES:
            row_str = f"  {FAM_LABEL[fx]:<14} "
            for fy in FAMILIES:
                r = sub[(sub["est_x"] == fx) & (sub["est_y"] == fy)]
                m = float(r["rmse_z_mean"].iloc[0])
                s = float(r["rmse_z_std"].iloc[0])
                mark = ""
                if fx == true_fx and fy == true_fy: mark = "*"
                elif fx == "gaussian" and fy == "bernoulli": mark = "†"
                row_str += f"  {m:6.4f}±{s:.4f}{mark:1s} "
            print(row_str)

        print(f"\n  悪化倍率 (÷ {oracle_rmse:.5f}):")
        print(header)
        max_ratio = 0.0
        max_cond  = ""
        conv_ratio = None
        for fx in FAMILIES:
            row_str = f"  {FAM_LABEL[fx]:<14} "
            for fy in FAMILIES:
                r = sub[(sub["est_x"] == fx) & (sub["est_y"] == fy)]
                ratio = float(r["degradation_ratio"].iloc[0])
                mark = ""
                if fx == true_fx and fy == true_fy: mark = "*"
                elif fx == "gaussian" and fy == "bernoulli":
                    mark = "†"
                    conv_ratio = ratio
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_cond  = f"X={FAM_LABEL[fx]}, Y={FAM_LABEL[fy]}"
                row_str += f"  {ratio:8.3f}x{mark:1s}      "
            print(row_str)

        print(f"  Worst: {max_cond} → {max_ratio:.3f}x")
        if conv_ratio is not None:
            print(f"  Conv-like (Gauss-Bern): {conv_ratio:.3f}x")
        print(f"  * = Oracle,  † = Conv-like")


if __name__ == "__main__":
    t0 = time.perf_counter()
    print("Running distribution mismatch experiment (Fixed model)...")
    print(f"  N={N}, D={D}, K_TRUE={K_TRUE}, L={L}, NITER={NITER}, N_TRIALS={N_TRIALS}")
    print(f"  Scenarios: {[s[0] for s in SCENARIOS]}")

    df      = run_all()
    summary = make_summary(df)

    summary_csv = _RES / "mismatch_fixed_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    print_tables(summary)
    make_figures(df, summary)

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min")
    print(f"Results: {_RES}")
    print(f"Figures: {_FIG}")
