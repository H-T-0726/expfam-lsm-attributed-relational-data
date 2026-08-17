"""
Y sparsity stress test — Yの学習観測率を下げたときにXが効くかを見る診断実験。

背景（story diagnostics フェーズ、2026-07-13〜）:
    既存の single_vs_joint 実験（per_column_all vs all_gaussian vs 単独属性）では
    差がごく小さかった（RMSE_Z 0.235 vs 0.234 等）。既存レポートは
    「n=80 の全ペアが観測されており Y 側の情報がすでに濃いため、
     X の追加寄与が相対的に小さい設定である」という仮説を述べているが
    （single_vs_joint_per_column_report_20260711.md, per_column_ablation_report_20260711.md）、
    この仮説自体を検証する実験はまだ行われていない。

    本スクリプトは、Y の学習観測率（train pair の割合）を意図的に下げ、
    Xを使うモデル（per_column_all 等）が y_only より改善するかを確認する。

分割方式:
    1. 固定の test set（全 y_obs_rate 条件で共通、test_ratio=0.2）を
       make_pair_split で作る。評価はこの固定 test set のみで行う。
    2. 残り 80% の「学習可能プール」から、y_obs_rate に応じてさらに
       間引いた train_mask を作る（y_obs_rate=1.0 はプール全部＝
       既存 single_vs_joint と同じ密度）。

今回のスコープ（軽量版、2026-07-13）:
    条件は y_only / single_gaussian / per_column_all / all_gaussian の4つに限定。
    trials は 1〜2（TRIALS 定数で管理）。
    ACTIVE_CONDITIONS / TRIALS を変えるだけでフル条件・フル trial 数に拡張できる
    構造にしている（他は変更不要）。

出力:
    expfam/results/story_diagnostics/y_sparsity_stress_20260713.csv
    expfam/results/story_diagnostics/y_sparsity_stress_20260713_agg.csv
    expfam/results/story_diagnostics/y_sparsity_stress_20260713_runinfo.csv
    figures/story_diagnostics/y_sparsity_rmse_z.png
    figures/story_diagnostics/y_sparsity_test_y_ll.png

実行: python tools/research_audit/run_y_sparsity_stress.py
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
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from utils_expfam import procrustes_rotation, calc_rmse    # noqa: E402
from em_runner import run_em_experimental, predict_mu_y    # noqa: E402
from eval_utils import make_pair_split, heldout_count_metrics, upper_pairs_of  # noqa: E402

OUT_DIR = _ROOT / "expfam" / "results" / "story_diagnostics"
FIG_DIR = _ROOT / "figures" / "story_diagnostics"

# ── 生成設定（run_per_column_single_vs_joint.py の generate() と同一） ──────
N, D, K_TRUE = 80, 9, 2
W0_TRUE, W_TRUE = 1.2, 0.3
L, NITER = 5, 8
TEST_RATIO = 0.2

DATA_SEED_BASE = 94000
MODEL_SEED_BASE = 95000
SPLIT_SEED_BASE = 96000
THIN_SEED_BASE = 97000

FAM_LIST_TRUE = ["gaussian"] * 3 + ["bernoulli"] * 3 + ["poisson"] * 3
TRUE_BLOCKS = {"gauss": np.arange(0, 3), "bern": np.arange(3, 6),
               "pois": np.arange(6, 9)}

# ── 今回のスコープ（軽量版→trials拡張）。フル拡張時はここだけ変える ────────
TRIALS = 10
RUN_TAG = "trials10"   # 出力ファイル名のsuffix。既存 20260713.* を上書きしないため
Y_OBS_RATES = [1.0, 0.5, 0.2, 0.1]
ACTIVE_CONDITIONS = ["y_only", "single_gaussian", "per_column_all", "all_gaussian"]
# フル拡張時: ["y_only", "single_bernoulli", "single_gaussian", "single_poisson",
#              "per_column_all", "all_gaussian", "all_bernoulli"]

CSV_MAIN = f"y_sparsity_stress_20260713_{RUN_TAG}.csv"
CSV_AGG = f"y_sparsity_stress_20260713_{RUN_TAG}_agg.csv"
CSV_RUNINFO = f"y_sparsity_stress_20260713_{RUN_TAG}_runinfo.csv"
FIG_RMSE = f"y_sparsity_rmse_z_{RUN_TAG}.png"
FIG_TESTLL = f"y_sparsity_test_y_ll_{RUN_TAG}.png"


def generate(seed):
    """run_per_column_single_vs_joint.py の generate() と同一（複製）。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, K_TRUE))
    Z = (Z - Z.mean(0)) / Z.std(0)
    F = rng.standard_normal((D, K_TRUE))
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    eta = Z @ F.T
    X = np.zeros((N, D))
    X[:, 0:3] = eta[:, 0:3] + rng.normal(0, 0.3, (N, 3))
    X[:, 3:6] = rng.binomial(1, 1 / (1 + np.exp(-eta[:, 3:6])))
    X[:, 6:9] = rng.poisson(np.exp(np.clip(eta[:, 6:9], -20, 5)))
    eta_y = W0_TRUE + W_TRUE * (Z @ Z.T)
    Y = np.zeros((N, N))
    iu = np.triu_indices(N, 1)
    Y[iu] = rng.poisson(np.exp(np.clip(eta_y[iu], -20, 10)))
    Y = Y + Y.T
    return dict(X=X, Y=Y, Z=Z, F=F)


def build_conditions(X):
    """条件名 → (X_used, used_cols, run_kwargs)。single_vs_joint と同一のロジック。"""
    conds = {}
    for bname, fam in (("gaussian", "gaussian"), ("bernoulli", "bernoulli"),
                       ("poisson", "poisson")):
        key = {"gaussian": "gauss", "bernoulli": "bern", "poisson": "pois"}[bname]
        cols = TRUE_BLOCKS[key]
        conds[f"single_{bname}"] = (X[:, cols], cols,
                                    dict(family_x=fam, family_x_list=None))
    conds["per_column_all"] = (X, np.arange(D),
                               dict(family_x=None, family_x_list=FAM_LIST_TRUE))
    for fam in ("gaussian", "bernoulli", "poisson"):
        conds[f"all_{fam}"] = (X, np.arange(D),
                               dict(family_x=fam, family_x_list=None))
    conds["y_only"] = (X, np.array([], dtype=int),
                       dict(family_x="gaussian", family_x_list=None, fix_x=True))
    return conds


def block_recon_rmse(X_used, used_cols, res):
    """使用列のブロック別 X 再構成 RMSE（single_vs_joint と同一ロジック）。"""
    model = res["model"]
    eta_x = res["Z_est"] @ res["F"].T
    mu_x = model._mean_function_x(eta_x)
    out = {}
    for bname, cols in TRUE_BLOCKS.items():
        pos = [int(np.where(used_cols == c)[0][0]) for c in cols
               if c in used_cols]
        if not pos:
            out[f"x_rmse_{bname}"] = float("nan")
            continue
        resid = X_used[:, pos] - mu_x[:, pos]
        out[f"x_rmse_{bname}"] = float(np.sqrt(np.mean(resid ** 2)))
    return out


def thin_pool_mask(pool_mask, keep_rate, seed):
    """
    pool_mask（学習可能プール、対称 bool）の上三角ペアから keep_rate 割合を
    ランダムに残した train_mask を作る。keep_rate=1.0 なら pool_mask そのもの。
    """
    if keep_rate >= 1.0:
        return pool_mask.copy()
    n = pool_mask.shape[0]
    rows, cols = upper_pairs_of(pool_mask)
    n_pool = len(rows)
    n_keep = max(1, int(round(n_pool * keep_rate)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pool)
    keep_idx = perm[:n_keep]

    train_mask = np.zeros((n, n), dtype=bool)
    train_mask[rows[keep_idx], cols[keep_idx]] = True
    train_mask |= train_mask.T
    np.fill_diagonal(train_mask, False)
    return train_mask


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in range(TRIALS):
        data = generate(DATA_SEED_BASE + trial)
        pool_mask, test_mask = make_pair_split(
            N, TEST_RATIO, seed=SPLIT_SEED_BASE + trial * 100)
        conds_full = build_conditions(data["X"])
        conds = {k: v for k, v in conds_full.items() if k in ACTIVE_CONDITIONS}

        for rate_idx, y_obs_rate in enumerate(Y_OBS_RATES):
            train_mask = thin_pool_mask(
                pool_mask, y_obs_rate,
                seed=THIN_SEED_BASE + trial * 100 + rate_idx)
            n_train_pairs = int(np.triu(train_mask, k=1).sum())

            for cname, (X_used, used_cols, kw) in conds.items():
                res = run_em_experimental(
                    X_used, data["Y"], family_y="poisson",
                    k=K_TRUE, L=L, num_iter=NITER,
                    seed=MODEL_SEED_BASE + trial * 10,
                    train_mask=train_mask, **kw)
                R, k_min = procrustes_rotation(res["Z_est"], data["Z"])
                rmse_Z = calc_rmse(data["Z"][:, :k_min],
                                   res["Z_est"][:, :k_min] @ R)
                mu_y = predict_mu_y(res)
                m_tr = heldout_count_metrics(data["Y"], mu_y, train_mask, "poisson")
                m_te = heldout_count_metrics(data["Y"], mu_y, test_mask, "poisson")
                row = {
                    "condition": cname, "y_obs_rate": y_obs_rate, "trial": trial,
                    "n_cols_used": len(used_cols),
                    "n_train_pairs": n_train_pairs,
                    "rmse_Z": rmse_Z,
                    "w0_err": abs(res["w0"] - W0_TRUE),
                    "w_err": abs(res["w"] - W_TRUE),
                    "train_y_ll": m_tr.get("mean_ll", float("nan")),
                    "test_y_ll": m_te.get("mean_ll", float("nan")),
                    "train_y_rmse": m_tr.get("rmse", float("nan")),
                    "test_y_rmse": m_te.get("rmse", float("nan")),
                    "nan_occurred": res["nan_occurred"],
                    "runtime_s": res["runtime_s"],
                }
                if kw.get("fix_x"):
                    row.update({f"x_rmse_{b}": float("nan") for b in TRUE_BLOCKS})
                else:
                    row.update(block_recon_rmse(X_used, used_cols, res))
                rows.append(row)
                print(f"t={trial} rate={y_obs_rate:.1f} {cname:16s} "
                      f"n_train={n_train_pairs:4d} rmse_Z={rmse_Z:.3f} "
                      f"te_ll={row['test_y_ll']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / CSV_MAIN, index=False)

    agg = df.groupby(["condition", "y_obs_rate"]).agg(
        n_trials=("rmse_Z", "count"),
        n_train_pairs_mean=("n_train_pairs", "mean"),
        rmse_Z_mean=("rmse_Z", "mean"), rmse_Z_std=("rmse_Z", "std"),
        test_y_ll_mean=("test_y_ll", "mean"), test_y_ll_std=("test_y_ll", "std"),
        test_y_rmse_mean=("test_y_rmse", "mean"),
        w0_err_mean=("w0_err", "mean"), w_err_mean=("w_err", "mean"),
        n_nan=("nan_occurred", "sum")).reset_index()

    # per_column_all を基準にした差分（同じ y_obs_rate 内で比較しやすくする）
    percol = agg.loc[agg["condition"] == "per_column_all",
                     ["y_obs_rate", "rmse_Z_mean", "test_y_ll_mean"]].rename(
        columns={"rmse_Z_mean": "rmse_Z_mean_percolumn",
                 "test_y_ll_mean": "test_y_ll_mean_percolumn"})
    agg = agg.merge(percol, on="y_obs_rate", how="left")
    # rmse_Z: 正の値 = per_column_all より悪い（RMSEが大きい）
    agg["rmse_Z_diff_vs_percolumn"] = (agg["rmse_Z_mean"]
                                       - agg["rmse_Z_mean_percolumn"])
    # test_y_ll: 負の値 = per_column_all より悪い（llが低い）
    agg["test_y_ll_diff_vs_percolumn"] = (agg["test_y_ll_mean"]
                                          - agg["test_y_ll_mean_percolumn"])

    agg.to_csv(OUT_DIR / CSV_AGG, index=False)
    print("\n=== Aggregated ===")
    print(agg.to_string(index=False))

    def git_head():
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=_ROOT).stdout.strip()
        except Exception:
            return "unknown"

    runinfo = [{
        "script": "tools/research_audit/run_y_sparsity_stress.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/story-diagnostics",
        "n": N, "d": D, "k_true": K_TRUE,
        "w0_true": W0_TRUE, "w_true": W_TRUE,
        "fam_list_true": str(FAM_LIST_TRUE),
        "trials": TRIALS, "L": L, "num_iter": NITER,
        "test_ratio_fixed": TEST_RATIO,
        "y_obs_rates": str(Y_OBS_RATES),
        "active_conditions": str(ACTIVE_CONDITIONS),
        "data_seed_base": DATA_SEED_BASE,
        "model_seed_base": MODEL_SEED_BASE,
        "split_seed_base": SPLIT_SEED_BASE,
        "thin_seed_base": THIN_SEED_BASE,
        "note": ("test_mask は全 y_obs_rate 条件で固定（trial内で同一）。"
                 "train_mask は学習可能プール(1-test_ratio)から y_obs_rate割合を"
                 "ランダム間引きして作成。y_obs_rate=1.0 はプール全部で"
                 "既存 single_vs_joint と同じ密度。軽量版のためACTIVE_CONDITIONSは"
                 "y_only/single_gaussian/per_column_all/all_gaussianの4条件に限定。"),
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]
    pd.DataFrame(runinfo).to_csv(OUT_DIR / CSV_RUNINFO, index=False)

    make_figures(agg)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


ROLE = {
    "y_only": ("y_only（ベースライン）", "#888888"),
    "single_gaussian": ("single_gaussian（単独属性）", "#4C72B0"),
    "per_column_all": ("per_column_all（本命）", "#C44E52"),
    "all_gaussian": ("all_gaussian（誤指定比較用）", "#55A868"),
    "single_bernoulli": ("single_bernoulli（単独属性）", "#8172B2"),
    "single_poisson": ("single_poisson（単独属性）", "#CCB974"),
    "all_bernoulli": ("all_bernoulli（誤指定比較用）", "#64B5CD"),
}


def _color_label(cname):
    return ROLE.get(cname, (cname, "#333333"))


def make_figures(agg):
    for metric, ylabel, fname, invert_x in (
        ("rmse_Z_mean", "RMSE_Z（低いほど良い）", FIG_RMSE, True),
        ("test_y_ll_mean", "test Y log-likelihood / pair（高いほど良い）",
         FIG_TESTLL, True),
    ):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for cname in ACTIVE_CONDITIONS:
            sub = agg[agg["condition"] == cname].sort_values("y_obs_rate")
            if sub.empty:
                continue
            label, color = _color_label(cname)
            ax.plot(sub["y_obs_rate"], sub[metric], marker="o",
                   label=label, color=color)
        ax.set_xlabel("y_obs_rate（Yの学習観測率）")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Y sparsity stress test（trials={TRIALS}）")
        if invert_x:
            ax.invert_xaxis()
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=150)
        plt.close(fig)
        print(f"saved {FIG_DIR / fname}")


if __name__ == "__main__":
    main()
