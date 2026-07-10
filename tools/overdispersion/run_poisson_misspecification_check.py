"""
人工過分散カウント Y に対する Poisson 誤指定の影響実験（Phase 3）。

生成: Dual-ExpFam 構造（X=Bernoulli, k*=3）で Y を NB2(μ, r_true) から生成。
      r_true ∈ {2, 5, 20, inf}（inf = Poisson、過分散なし）。
      var/mean = 1 + μ/r なので μ≈5 のとき r=2 → 約 3.5、r=5 → 約 2.0、
      r=20 → 約 1.25、inf → 1.0。

推定: strict held-out（train 80% ペアのみで学習、fixed 版系列 masked モデル）
  poisson    : Poisson-Y（誤指定; r_true=inf のときのみ正指定）
  nb_oracle  : NB2-Y、r = r_true（正指定; r_true=inf では省略）
  nb_moment  : NB2-Y、r̂ = poisson フィットの train 残差モーメント推定
               （実務で可能な two-stage 手続き）

評価:
  潜在構造: RMSE(Z)（Procrustes 後）、RMSE(F)、w0_err、w_err
  予測:     held-out mean_ll / RMSE / Pearson / Pearson dispersion
  選択:     BIC(train)

出力:
  expfam/results/overdispersion/poisson_misspecification_summary.csv
  expfam/results/overdispersion/poisson_misspecification_agg.csv
  expfam/results/overdispersion/poisson_misspecification_runinfo.csv
  figures/overdispersion/poisson_misspec_rmse_z.png/pdf
  figures/overdispersion/poisson_misspec_heldout_ll.png/pdf

実行: python tools/overdispersion/run_poisson_misspecification_check.py
"""

import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from utils_expfam import procrustes_rotation, calc_rmse            # noqa
from em_runner import run_em_experimental, predict_mu_y            # noqa
from eval_utils import (                                           # noqa
    make_pair_split, heldout_count_metrics, upper_pairs_of,
    moment_estimate_nb_r)
from data_generator_overdispersed import generate_dual_data_nb_y   # noqa

OUT_DIR = _ROOT / "expfam" / "results" / "overdispersion"
FIG_DIR = _ROOT / "figures" / "overdispersion"

N, D, K_TRUE = 100, 15, 3
W0_TRUE, W_TRUE = 1.5, 0.3
R_TRUE_LIST = [2.0, 5.0, 20.0, float("inf")]
N_TRIALS = 5
TEST_RATIO = 0.2
L, NITER = 5, 8
DATA_SEED_BASE = 51000
SPLIT_SEED_BASE = 52000
MODEL_SEED_BASE = 53000

FIT_COLORS = {   # dataviz palette, fixed order
    "poisson": "#2a78d6",
    "nb_oracle": "#1baf7a",
    "nb_moment": "#eda100",
}
FIT_LABELS = {
    "poisson": "Poisson (misspecified)",
    "nb_oracle": "NB oracle (r = r_true)",
    "nb_moment": "NB moment (r = r̂)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.dpi": 150,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def truth_metrics(res, data):
    """RMSE(Z) (Procrustes), RMSE(F), w0/w err."""
    Z_est, F_est = res["Z_est"], res["F"]
    R, k_min = procrustes_rotation(Z_est, data["Z"])
    rmse_Z = calc_rmse(data["Z"][:, :k_min], Z_est[:, :k_min] @ R)
    rmse_F = calc_rmse(data["F"][:, :k_min], F_est[:, :k_min] @ R)
    return {
        "rmse_Z": rmse_Z, "rmse_F": rmse_F,
        "w0_err": abs(res["w0"] - data["w0"]),
        "w_err": abs(res["w"] - data["w"]),
    }


def one_fit(tag, data, family_y, nb_r, k, seed, train_mask, test_mask):
    res = run_em_experimental(
        data["X"], data["Y"], family_x=data["family_x"], family_y=family_y,
        k=k, L=L, num_iter=NITER, seed=seed, train_mask=train_mask,
        nb_r=nb_r)
    mu = predict_mu_y(res)
    label = "nb" if family_y == "nb" else "poisson"
    te = heldout_count_metrics(data["Y"], mu, test_mask, label, nb_r=nb_r)
    tr = heldout_count_metrics(data["Y"], mu, train_mask, label, nb_r=nb_r)
    row = {"fit_condition": tag, "family_y_fit": family_y,
           "nb_r_fit": (nb_r if nb_r is not None else np.nan),
           "w0_est": res["w0"], "w_est": res["w"],
           "bic_train": res["bic"], "q_strict_train": res["Q_strict"],
           "nan_occurred": res["nan_occurred"], "runtime_s": res["runtime_s"]}
    row.update(truth_metrics(res, data))
    for pre, m in (("test_", te), ("train_", tr)):
        for k2, v in m.items():
            row[pre + k2] = v
    return row, mu


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    fit_i = 0
    for r_true in R_TRUE_LIST:
        r_tag = "inf" if not np.isfinite(r_true) else f"{r_true:g}"
        for trial in range(N_TRIALS):
            data = generate_dual_data_nb_y(
                n=N, d=D, k=K_TRUE,
                seed=DATA_SEED_BASE + trial,        # 同 trial は r 間で同一 Z/F/X 乱数系列
                family_x="bernoulli",
                w0_true=W0_TRUE, w_true=W_TRUE,
                nb_r=(None if not np.isfinite(r_true) else r_true))
            train_mask, test_mask = make_pair_split(
                N, TEST_RATIO, seed=SPLIT_SEED_BASE + trial)
            seed = MODEL_SEED_BASE + trial * 10
            base = {"r_true": (np.inf if not np.isfinite(r_true) else r_true),
                    "r_true_tag": r_tag, "trial": trial,
                    "data_seed": DATA_SEED_BASE + trial,
                    "model_seed": seed,
                    "y_var_mean_ratio_gen": data["y_var_mean_ratio"],
                    "y_mean_gen": data["y_mean"],
                    "y_zero_ratio_gen": data["y_zero_ratio"]}

            # 1) Poisson（誤指定）
            fit_i += 1
            row_p, mu_p = one_fit("poisson", data, "poisson", None,
                                  K_TRUE, seed, train_mask, test_mask)
            rows.append({**base, **row_p})
            print(f"[{fit_i:3d}] r={r_tag:>4s} t={trial} POIS   "
                  f"rmseZ={row_p['rmse_Z']:.3f} te_ll={row_p['test_mean_ll']:.3f} "
                  f"te_disp={row_p['test_pearson_dispersion']:.2f}")

            # r̂（train 残差、two-stage）
            tr_r, tr_c = upper_pairs_of(train_mask)
            r_hat = moment_estimate_nb_r(data["Y"][tr_r, tr_c],
                                         mu_p[tr_r, tr_c])

            # 2) NB oracle（r_true 有限のみ）
            if np.isfinite(r_true):
                fit_i += 1
                row_o, _ = one_fit("nb_oracle", data, "nb", r_true,
                                   K_TRUE, seed, train_mask, test_mask)
                rows.append({**base, **row_o, "r_hat_train": r_hat})
                print(f"[{fit_i:3d}]              NB-orc "
                      f"rmseZ={row_o['rmse_Z']:.3f} "
                      f"te_ll={row_o['test_mean_ll']:.3f}")

            # 3) NB moment
            fit_i += 1
            row_m, _ = one_fit("nb_moment", data, "nb", min(r_hat, 1e6),
                               K_TRUE, seed, train_mask, test_mask)
            rows.append({**base, **row_m, "r_hat_train": r_hat})
            print(f"[{fit_i:3d}]              NB-mom "
                  f"rmseZ={row_m['rmse_Z']:.3f} te_ll={row_m['test_mean_ll']:.3f} "
                  f"r_hat={r_hat:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "poisson_misspecification_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'poisson_misspecification_summary.csv'}")

    # ── Aggregate ────────────────────────────────────────────────────
    mcols = ["rmse_Z", "rmse_F", "w0_err", "w_err",
             "test_mean_ll", "test_rmse", "test_pearson",
             "test_pearson_dispersion", "train_pearson_dispersion",
             "bic_train", "y_var_mean_ratio_gen", "runtime_s"]
    agg_rows = []
    for (r_tag, cond), sub in df.groupby(["r_true_tag", "fit_condition"]):
        row = {"r_true_tag": r_tag, "fit_condition": cond, "n_trials": len(sub),
               "n_nan": int(sub["nan_occurred"].sum())}
        for c in mcols:
            vals = sub[c].dropna().astype(float)
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_std"] = float(vals.std()) if len(vals) else np.nan
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT_DIR / "poisson_misspecification_agg.csv", index=False)

    print("\n=== Aggregated ===")
    show = ["r_true_tag", "fit_condition", "rmse_Z_mean", "test_mean_ll_mean",
            "test_pearson_dispersion_mean", "y_var_mean_ratio_gen_mean"]
    print(agg[show].to_string(index=False))

    # ── Figures ──────────────────────────────────────────────────────
    r_order = ["2", "5", "20", "inf"]
    conds = ["poisson", "nb_oracle", "nb_moment"]

    for metric, fname, ylab, title in [
        ("rmse_Z_mean", "poisson_misspec_rmse_z",
         "RMSE(Z) after Procrustes (lower better)",
         "Latent structure recovery vs overdispersion strength"),
        ("test_mean_ll_mean", "poisson_misspec_heldout_ll",
         "Held-out log-lik / pair (higher better)",
         "Held-out predictive likelihood vs overdispersion strength"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        width = 0.26
        xs = np.arange(len(r_order))
        for ci, cond in enumerate(conds):
            vals, errs = [], []
            for r_tag in r_order:
                sub = agg[(agg["r_true_tag"] == r_tag)
                          & (agg["fit_condition"] == cond)]
                vals.append(sub[metric].values[0] if len(sub) else np.nan)
                std_col = metric[: -len("_mean")] + "_std"   # 末尾のみ置換
                errs.append(sub[std_col].values[0] if len(sub) else np.nan)
            ax.bar(xs + (ci - 1) * width, vals, width=width, yerr=errs,
                   capsize=3, color=FIT_COLORS[cond],
                   label=FIT_LABELS[cond], edgecolor="none")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"r_true={r}\n(var/mean≈{vm})"
                            for r, vm in zip(r_order,
                                             ["3.5", "2.0", "1.25", "1.0"])],
                           fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}\n(n={N}, d={D}, k*={K_TRUE}, X=Bernoulli, "
                     f"strict held-out 20%, {N_TRIALS} trials)")
        ax.legend(frameon=False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{fname}.{ext}", bbox_inches="tight")
        plt.close(fig)
    print(f"Figures: {FIG_DIR}")

    def git_head():
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=_ROOT).stdout.strip()
        except Exception:
            return "unknown"

    pd.DataFrame([{
        "script": "tools/overdispersion/run_poisson_misspecification_check.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/overdispersion-z-ablation",
        "n": N, "d": D, "k_true": K_TRUE,
        "w0_true": W0_TRUE, "w_true": W_TRUE,
        "r_true_list": str(R_TRUE_LIST), "n_trials": N_TRIALS,
        "test_ratio": TEST_RATIO,
        "data_seed_base": DATA_SEED_BASE,
        "split_seed_base": SPLIT_SEED_BASE,
        "model_seed_base": MODEL_SEED_BASE,
        "L": L, "num_iter": NITER,
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]).to_csv(OUT_DIR / "poisson_misspecification_runinfo.csv", index=False)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
