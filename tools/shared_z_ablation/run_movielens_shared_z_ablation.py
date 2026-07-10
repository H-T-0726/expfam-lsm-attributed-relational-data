"""
MovieLens 共有 Z ablation（strict held-out、Phase 5 後半）。

問い（Q3）:
  1つの Z で X（ジャンル）と Y（共評価カウント）の両方を説明する
  「共有潜在変数仮定」は MovieLens で妥当か。
    - X は Y の held-out 予測に寄与するか（proposed vs y_only）
    - Y は X 再構成・ジャンル構造（NMI/ARI）に寄与するか
      （proposed vs x_only）

条件（各 split × model seed × k）:
  proposed_XY  : X + Y 統合（DualExpFamLSMMasked, strict held-out）
  y_only_fix_x : F=0 固定（X 信号遮断; Z は Y のみから学習）
  x_only_fix_w : w=0 固定（Y 信号遮断; Z は X のみから学習。
                 Y 予測は定数 exp(w0) となる構造的ベースライン）

Y family は Poisson（Phase 1 診断で条件付き過分散が小さいことを確認済み。
NB との比較は run_movielens_strict_heldout.py が担当）。

出力:
  expfam/results/shared_z_ablation/movielens_shared_z_ablation_summary.csv
  expfam/results/shared_z_ablation/movielens_shared_z_ablation_agg.csv
  expfam/results/shared_z_ablation/movielens_shared_z_ablation_runinfo.csv
  figures/shared_z_ablation/movielens_shared_z_ablation.png/pdf

実行: python tools/shared_z_ablation/run_movielens_shared_z_ablation.py
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
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from em_runner import run_em_experimental, predict_mu_y     # noqa
from eval_utils import make_pair_split, heldout_count_metrics  # noqa

DATA_DIR = _ROOT / "expfam" / "data" / "movielens_pilot"
OUT_DIR = _ROOT / "expfam" / "results" / "shared_z_ablation"
FIG_DIR = _ROOT / "figures" / "shared_z_ablation"

K_LIST = [5]
SPLIT_TRIALS = [0, 1, 2]
MODEL_TRIALS = [0, 1]
TEST_RATIO = 0.2
L, NITER = 5, 8
HC_THRESHOLD = 80
SPLIT_SEED_BASE = 41000     # strict_heldout 実験と同一 split を使用（比較可能に）
MODEL_SEED_BASE = 61000
FAMILY_Y = "poisson"

CONDS = ["proposed_XY", "y_only_fix_x", "x_only_fix_w"]
COND_COLORS = {  # dataviz palette, fixed order
    "proposed_XY": "#2a78d6",
    "y_only_fix_x": "#1baf7a",
    "x_only_fix_w": "#eda100",
}
COND_LABELS = {
    "proposed_XY": "Proposed (X+Y)",
    "y_only_fix_x": "Y-only (fix_x: F=0)",
    "x_only_fix_w": "X-only (fix_w: w=0)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.dpi": 150,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def z_cluster_metrics(Z_est, labels, seed=42):
    """フル次元 Z の KMeans による NMI / ARI（クラスタ数 = ラベル種数）。

    注: 既存 pilot（PCA→2D→KMeans）と手続きが異なるため絶対値は
    直接比較しないこと。本実験内の条件間比較にのみ使う。
    """
    n_cl = len(np.unique(labels))
    try:
        km = KMeans(n_clusters=n_cl, random_state=seed,
                    n_init=10).fit_predict(Z_est)
        return (float(normalized_mutual_info_score(labels, km)),
                float(adjusted_rand_score(labels, km)))
    except Exception:
        return float("nan"), float("nan")


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_DIR / "movielens_X_genre.npy").astype(float)
    Y = np.load(DATA_DIR / "movielens_Y_count.npy").astype(float)
    genre_labels = np.load(DATA_DIR / "movielens_primary_genre_labels.npy")
    n, d = X.shape

    total = len(K_LIST) * len(SPLIT_TRIALS) * len(MODEL_TRIALS) * len(CONDS)
    print(f"=== MovieLens shared-Z ablation: {total} fits ===")

    rows = []
    fit_i = 0
    for split in SPLIT_TRIALS:
        train_mask, test_mask = make_pair_split(
            n, TEST_RATIO, seed=SPLIT_SEED_BASE + split * 100)
        for k in K_LIST:
            for mt in MODEL_TRIALS:
                seed = MODEL_SEED_BASE + k * 100 + split * 10 + mt
                for cond in CONDS:
                    fit_i += 1
                    fix_x = (cond == "y_only_fix_x")
                    fix_w = (cond == "x_only_fix_w")
                    res = run_em_experimental(
                        X, Y, family_x="bernoulli", family_y=FAMILY_Y,
                        k=k, L=L, num_iter=NITER, seed=seed,
                        train_mask=train_mask, fix_x=fix_x, fix_w=fix_w)
                    mu = predict_mu_y(res)
                    te = heldout_count_metrics(
                        Y, mu, test_mask, FAMILY_Y,
                        high_count_threshold=HC_THRESHOLD)
                    tr = heldout_count_metrics(
                        Y, mu, train_mask, FAMILY_Y,
                        high_count_threshold=HC_THRESHOLD)

                    # X 再構成
                    mu_x = res["model"]._mean_function_x(
                        res["Z_est"] @ res["F"].T)
                    rmse_x = float(np.sqrt(np.mean((X - mu_x) ** 2)))
                    x_acc = float(np.mean((mu_x > 0.5).astype(int)
                                          == X.astype(int)))
                    nmi, ari = z_cluster_metrics(res["Z_est"], genre_labels)

                    row = {"condition": cond, "k": k,
                           "split_trial": split, "model_trial": mt,
                           "model_seed": seed, "family_y": FAMILY_Y,
                           "w0": res["w0"], "w": res["w"],
                           "bic_train": res["bic"],
                           "rmse_x": rmse_x, "x_binary_acc": x_acc,
                           "nmi_genre": nmi, "ari_genre": ari,
                           "nan_occurred": res["nan_occurred"],
                           "runtime_s": res["runtime_s"]}
                    for pre, m in (("test_", te), ("train_", tr)):
                        for k2, v in m.items():
                            row[pre + k2] = v
                    rows.append(row)
                    print(f"[{fit_i:2d}/{total}] {cond:13s} k={k} sp={split} "
                          f"mt={mt}  te_ll={row['test_mean_ll']:.3f} "
                          f"te_pear={row['test_pearson']:.3f} "
                          f"rmse_x={rmse_x:.3f} NMI={nmi:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "movielens_shared_z_ablation_summary.csv", index=False)

    # ── Aggregate ────────────────────────────────────────────────────
    mcols = [c for c in df.columns
             if c.startswith(("test_", "train_"))
             and df[c].dtype != object] + [
        "rmse_x", "x_binary_acc", "nmi_genre", "ari_genre", "runtime_s"]
    agg_rows = []
    for (cond, k), sub in df.groupby(["condition", "k"]):
        row = {"condition": cond, "k": k, "n_fits": len(sub),
               "n_nan": int(sub["nan_occurred"].sum())}
        for c in mcols:
            vals = sub[c].dropna().astype(float)
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_std"] = float(vals.std()) if len(vals) else np.nan
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT_DIR / "movielens_shared_z_ablation_agg.csv", index=False)

    print("\n=== Aggregated ===")
    show = ["condition", "test_mean_ll_mean", "test_pearson_mean",
            "test_rmse_mean", "rmse_x_mean", "nmi_genre_mean"]
    print(agg[show].to_string(index=False))

    # ── Figure ───────────────────────────────────────────────────────
    a = agg.set_index("condition")
    specs = [
        ("test_mean_ll_mean", "test_mean_ll_std",
         "Held-out Y log-lik / pair"),
        ("test_pearson_mean", "test_pearson_std", "Held-out Y Pearson"),
        ("rmse_x_mean", "rmse_x_std", "X reconstruction RMSE (lower better)"),
        ("nmi_genre_mean", "nmi_genre_std", "NMI vs genre (full-Z KMeans)"),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.4 * len(specs), 3.9))
    for ax, (mcol, scol, title) in zip(axes, specs):
        xs = np.arange(len(CONDS))
        vals = [a.loc[c, mcol] for c in CONDS]
        errs = [a.loc[c, scol] for c in CONDS]
        ax.bar(xs, vals, yerr=errs, capsize=4, width=0.62,
               color=[COND_COLORS[c] for c in CONDS], edgecolor="none")
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}", (x, v), ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=8, color="#0b0b0b")
        ax.set_xticks(xs)
        ax.set_xticklabels(["Proposed\n(X+Y)", "Y-only\n(fix_x)",
                            "X-only\n(fix_w)"], fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.suptitle(
        f"MovieLens shared-Z ablation (k={K_LIST[0]}, Y={FAMILY_Y}, "
        f"strict held-out 20%, {len(SPLIT_TRIALS)} splits × "
        f"{len(MODEL_TRIALS)} seeds)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"movielens_shared_z_ablation.{ext}",
                    bbox_inches="tight")
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
        "script": "tools/shared_z_ablation/run_movielens_shared_z_ablation.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/overdispersion-z-ablation",
        "n": n, "d": d, "k_list": str(K_LIST),
        "split_trials": str(SPLIT_TRIALS), "model_trials": str(MODEL_TRIALS),
        "test_ratio": TEST_RATIO, "family_y": FAMILY_Y,
        "split_seed_base": SPLIT_SEED_BASE,
        "model_seed_base": MODEL_SEED_BASE,
        "L": L, "num_iter": NITER,
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]).to_csv(OUT_DIR / "movielens_shared_z_ablation_runinfo.csv", index=False)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
