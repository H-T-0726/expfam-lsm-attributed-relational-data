"""
MovieLens strict held-out 実験（Phase 2+3+4 統合）。

従来の masked evaluation（全ペアで学習→評価だけ分割、
run_fixed_real_movielens_heldout_count.py）と異なり、本実験は
**test ペアを学習（E-step / M-step / 尤度）から完全に除外**する
（DualExpFamLSMMasked、リーク無しはユニットテストで検証済み）。

条件（各 split × model seed × k）:
  poisson_strict : Poisson-Y、train ペアのみで学習
  nb_strict      : NB2-Y、train ペアのみで学習。
                   dispersion r̂ は poisson_strict の train ペア Pearson
                   残差からモーメント推定（two-stage、test 情報は不使用）
  poisson_full   : Poisson-Y、全ペアで学習（リーク参照条件 = 従来評価の再現。
                   test 指標の楽観バイアスを定量化するための対照）

主要な問い:
  Q1: strict held-out の test ペアで Pearson 過分散が現れるか
      （in-sample では 0.76〜1.14 と過分散が見えない）
  Q2: NB は Poisson より test 対数尤度・予測を改善するか
  optimism: poisson_full と poisson_strict の test 指標差 = 従来評価の楽観量

出力:
  expfam/results/overdispersion/movielens_strict_heldout_summary.csv
  expfam/results/overdispersion/movielens_strict_heldout_agg.csv
  expfam/results/overdispersion/movielens_strict_heldout_runinfo.csv
  figures/overdispersion/movielens_strict_heldout_comparison.png/pdf

実行: python tools/overdispersion/run_movielens_strict_heldout.py
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

from em_runner import run_em_experimental, predict_mu_y     # noqa
from eval_utils import (                                    # noqa
    make_pair_split, heldout_count_metrics, upper_pairs_of,
    pearson_dispersion, moment_estimate_nb_r)

DATA_DIR = _ROOT / "expfam" / "data" / "movielens_pilot"
OUT_DIR = _ROOT / "expfam" / "results" / "overdispersion"
FIG_DIR = _ROOT / "figures" / "overdispersion"

K_LIST = [3, 5]
SPLIT_TRIALS = [0, 1, 2]
MODEL_TRIALS = [0, 1]
TEST_RATIO = 0.2
L, NITER = 5, 8
HC_THRESHOLD = 80
SPLIT_SEED_BASE = 41000
MODEL_SEED_BASE = 42000

# dataviz palette (fixed categorical order)
COND_COLORS = {
    "poisson_strict": "#2a78d6",   # slot 1 blue
    "nb_strict": "#1baf7a",        # slot 2 aqua
    "poisson_full": "#eda100",     # slot 3 yellow (leakage reference)
}
COND_LABELS = {
    "poisson_strict": "Poisson (strict held-out)",
    "nb_strict": "NB (strict held-out)",
    "poisson_full": "Poisson (full-data training = leakage ref.)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.dpi": 150,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def eval_both(Y, mu, train_mask, test_mask, family_label, nb_r=None):
    tr = heldout_count_metrics(Y, mu, train_mask, family_label, nb_r=nb_r,
                               high_count_threshold=HC_THRESHOLD)
    te = heldout_count_metrics(Y, mu, test_mask, family_label, nb_r=nb_r,
                               high_count_threshold=HC_THRESHOLD)
    row = {}
    for prefix, m in (("train_", tr), ("test_", te)):
        for k2, v in m.items():
            row[prefix + k2] = v
    return row


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "movielens_X_genre.npy").astype(float)
    Y = np.load(DATA_DIR / "movielens_Y_count.npy").astype(float)
    n, d = X.shape

    total = len(K_LIST) * len(SPLIT_TRIALS) * len(MODEL_TRIALS) * 3
    print(f"=== MovieLens strict held-out: {total} fits "
          f"(n={n}, d={d}, test_ratio={TEST_RATIO}) ===")

    rows = []
    fit_i = 0
    for split in SPLIT_TRIALS:
        train_mask, test_mask = make_pair_split(
            n, TEST_RATIO, seed=SPLIT_SEED_BASE + split * 100)
        for k in K_LIST:
            for mt in MODEL_TRIALS:
                seed = MODEL_SEED_BASE + k * 100 + split * 10 + mt
                base = dict(k=k, split_trial=split, model_trial=mt,
                            model_seed=seed,
                            split_seed=SPLIT_SEED_BASE + split * 100)

                # ── 1) Poisson strict ────────────────────────────────
                fit_i += 1
                res_p = run_em_experimental(
                    X, Y, family_x="bernoulli", family_y="poisson",
                    k=k, L=L, num_iter=NITER, seed=seed,
                    train_mask=train_mask)
                mu_p = predict_mu_y(res_p)

                # train ペア残差から r̂（test 情報は不使用）
                tr_r, tr_c = upper_pairs_of(train_mask)
                r_hat = moment_estimate_nb_r(Y[tr_r, tr_c], mu_p[tr_r, tr_c])

                row = {**base, "condition": "poisson_strict",
                       "family_y": "poisson", "nb_r": np.nan,
                       "w0": res_p["w0"], "w": res_p["w"],
                       "bic_train": res_p["bic"],
                       "nan_occurred": res_p["nan_occurred"],
                       "runtime_s": res_p["runtime_s"],
                       "r_hat_train": r_hat}
                row.update(eval_both(Y, mu_p, train_mask, test_mask,
                                     "poisson"))
                rows.append(row)
                print(f"[{fit_i:2d}/{total}] k={k} sp={split} mt={mt} "
                      f"POIS-strict  te_ll={row['test_mean_ll']:.3f} "
                      f"te_disp={row['test_pearson_dispersion']:.3f} "
                      f"te_rmse={row['test_rmse']:.2f} r_hat={r_hat:.1f}")

                # ── 2) NB strict（r̂ 固定） ──────────────────────────
                fit_i += 1
                r_use = min(r_hat, 1e6)
                res_nb = run_em_experimental(
                    X, Y, family_x="bernoulli", family_y="nb",
                    k=k, L=L, num_iter=NITER, seed=seed,
                    train_mask=train_mask, nb_r=r_use)
                mu_nb = predict_mu_y(res_nb)
                row = {**base, "condition": "nb_strict",
                       "family_y": "nb", "nb_r": r_use,
                       "w0": res_nb["w0"], "w": res_nb["w"],
                       "bic_train": res_nb["bic"],
                       "nan_occurred": res_nb["nan_occurred"],
                       "runtime_s": res_nb["runtime_s"],
                       "r_hat_train": r_hat}
                row.update(eval_both(Y, mu_nb, train_mask, test_mask,
                                     "nb", nb_r=r_use))
                rows.append(row)
                print(f"[{fit_i:2d}/{total}]              NB-strict    "
                      f"te_ll={row['test_mean_ll']:.3f} "
                      f"te_disp={row['test_pearson_dispersion']:.3f} "
                      f"te_rmse={row['test_rmse']:.2f}")

                # ── 3) Poisson full（リーク参照） ────────────────────
                fit_i += 1
                res_f = run_em_experimental(
                    X, Y, family_x="bernoulli", family_y="poisson",
                    k=k, L=L, num_iter=NITER, seed=seed,
                    train_mask=None)
                mu_f = predict_mu_y(res_f)
                row = {**base, "condition": "poisson_full",
                       "family_y": "poisson", "nb_r": np.nan,
                       "w0": res_f["w0"], "w": res_f["w"],
                       "bic_train": res_f["bic"],
                       "nan_occurred": res_f["nan_occurred"],
                       "runtime_s": res_f["runtime_s"],
                       "r_hat_train": np.nan}
                row.update(eval_both(Y, mu_f, train_mask, test_mask,
                                     "poisson"))
                rows.append(row)
                print(f"[{fit_i:2d}/{total}]              POIS-full    "
                      f"te_ll={row['test_mean_ll']:.3f} "
                      f"te_disp={row['test_pearson_dispersion']:.3f} "
                      f"te_rmse={row['test_rmse']:.2f}")

    df = pd.DataFrame(rows)
    sum_path = OUT_DIR / "movielens_strict_heldout_summary.csv"
    df.to_csv(sum_path, index=False)
    print(f"\nSaved: {sum_path}")

    # ── Aggregate ────────────────────────────────────────────────────
    metric_cols = [c for c in df.columns
                   if c.startswith(("train_", "test_"))
                   and df[c].dtype != object] + ["r_hat_train", "runtime_s"]
    agg_rows = []
    for (cond, k), sub in df.groupby(["condition", "k"]):
        row = {"condition": cond, "k": k, "n_fits": len(sub),
               "n_nan": int(sub["nan_occurred"].sum())}
        for c in metric_cols:
            vals = sub[c].dropna().astype(float)
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_std"] = float(vals.std()) if len(vals) else np.nan
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg_path = OUT_DIR / "movielens_strict_heldout_agg.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Saved: {agg_path}")

    print("\n=== Aggregated (mean over splits × model seeds) ===")
    show = ["condition", "k", "test_mean_ll_mean", "test_rmse_mean",
            "test_pearson_mean", "test_pearson_dispersion_mean",
            "train_pearson_dispersion_mean", "test_hc_ap_mean"]
    print(agg[show].sort_values(["k", "condition"]).to_string(index=False))

    # ── Figure: grouped bars (k=5) ───────────────────────────────────
    for k_fig in K_LIST:
        a5 = agg[agg["k"] == k_fig].set_index("condition")
        conds = ["poisson_strict", "nb_strict", "poisson_full"]
        specs = [
            ("test_mean_ll_mean", "test_mean_ll_std",
             "Held-out log-lik / pair (higher better)"),
            ("test_rmse_mean", "test_rmse_std", "Held-out RMSE (lower better)"),
            ("test_pearson_mean", "test_pearson_std", "Held-out Pearson corr"),
            ("test_pearson_dispersion_mean", "test_pearson_dispersion_std",
             "Held-out Pearson dispersion (Poisson OK ≈ 1)"),
        ]
        fig, axes = plt.subplots(1, len(specs), figsize=(3.4 * len(specs), 3.8))
        for ax, (mcol, scol, title) in zip(axes, specs):
            xs = np.arange(len(conds))
            vals = [a5.loc[c, mcol] for c in conds]
            errs = [a5.loc[c, scol] for c in conds]
            ax.bar(xs, vals, yerr=errs, capsize=4, width=0.62,
                   color=[COND_COLORS[c] for c in conds], edgecolor="none")
            for x, v in zip(xs, vals):
                ax.annotate(f"{v:.2f}", (x, v), ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=8, color="#0b0b0b")
            if "dispersion" in mcol:
                ax.axhline(1.0, color="#52514e", linewidth=1.0,
                           linestyle="--", alpha=0.7)
            ax.set_xticks(xs)
            ax.set_xticklabels(["Poisson\nstrict", "NB\nstrict",
                                "Poisson\nfull(leak)"], fontsize=8)
            ax.set_title(title, fontsize=8.5)
            ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        fig.suptitle(
            f"MovieLens strict held-out (k={k_fig}, test 20% pairs, "
            f"{len(SPLIT_TRIALS)} splits × {len(MODEL_TRIALS)} seeds)",
            fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"movielens_strict_heldout_comparison_k{k_fig}.{ext}",
                        bbox_inches="tight")
        plt.close(fig)
    print(f"Figures: {FIG_DIR}")

    # ── run info ─────────────────────────────────────────────────────
    def git_head():
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=_ROOT).stdout.strip()
        except Exception:
            return "unknown"

    pd.DataFrame([{
        "script": "tools/overdispersion/run_movielens_strict_heldout.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/overdispersion-z-ablation",
        "n": n, "d": d, "k_list": str(K_LIST),
        "split_trials": str(SPLIT_TRIALS), "model_trials": str(MODEL_TRIALS),
        "test_ratio": TEST_RATIO,
        "split_seed_base": SPLIT_SEED_BASE, "model_seed_base": MODEL_SEED_BASE,
        "L": L, "num_iter": NITER,
        "model_classes": "DualExpFamLSMMasked / DualExpFamLSMNB (fixed-lineage)",
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]).to_csv(OUT_DIR / "movielens_strict_heldout_runinfo.csv", index=False)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
