"""
人工 mixed-X: 単独属性モデル vs per-column 同時統合 vs 全列共通 family 強制。

ゼミの問い:
    「ジャンルだけ Bernoulli、平均評価だけ Gaussian、評価件数だけ Poisson と
     別々にモデルを回せばよいのでは？」に直接答えるための比較実験。

設定:
    共通の Z_true (n=80, k*=2) から
      Gaussian 3列 / Bernoulli 3列 / Poisson 3列 の混在属性 X と
      Poisson-Y (w0=1.2, w=0.3) を生成。
    Y は strict held-out（pair mask、test_ratio=0.2）で train ペアのみ学習し、
    test ペアの plug-in 対数尤度・RMSE を評価する。

条件（9）:
    single_gaussian / single_bernoulli / single_poisson :
        1 ブロック（3列）のみを X に使用（単独属性 ablation）
    per_column_all :
        9 列すべてを family_x_list 正指定で同時使用（prototype、正指定）
    all_gaussian / all_bernoulli / all_poisson :
        9 列すべてを生値のまま単一 family で強制
        →「比較用の誤指定モデル」（現行の全列共通 family 制約の模擬）。
          自然なモデルではない。
    all_bernoulli_binarized :
        明示的変換による全列 Bernoulli（Gaussian 列: 列中央値超→1、
        Poisson 列: >0→1、Bernoulli 列: そのまま）
        → こちらも誤指定（情報を捨てる変換）の比較用条件。
    y_only :
        fix_x=True（F=0 固定、X 信号遮断）

評価: RMSE_Z（Procrustes 整合後）、strict held-out の train/test Y 対数尤度
      （全定数込み plug-in、/pair）と RMSE、w0/w 誤差、
      使用列のブロック別 X 再構成 RMSE（学習に使った X のスケール）、BIC、実行時間。

出力:
    expfam/results/per_column_family/single_vs_joint_summary.csv
    expfam/results/per_column_family/single_vs_joint_agg.csv
    expfam/results/per_column_family/single_vs_joint_runinfo.csv

実行: python tools/research_audit/run_per_column_single_vs_joint.py
"""

import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from utils_expfam import procrustes_rotation, calc_rmse    # noqa: E402
from em_runner import run_em_experimental, predict_mu_y    # noqa: E402
from eval_utils import make_pair_split, heldout_count_metrics  # noqa: E402

OUT_DIR = _ROOT / "expfam" / "results" / "per_column_family"

N, D, K_TRUE = 80, 9, 2
W0_TRUE, W_TRUE = 1.2, 0.3
N_TRIALS = 3
L, NITER = 5, 8
TEST_RATIO = 0.2
DATA_SEED_BASE = 81000
MODEL_SEED_BASE = 82000
SPLIT_SEED_BASE = 83000

FAM_LIST_TRUE = ["gaussian"] * 3 + ["bernoulli"] * 3 + ["poisson"] * 3
TRUE_BLOCKS = {"gauss": np.arange(0, 3), "bern": np.arange(3, 6),
               "pois": np.arange(6, 9)}

# 各条件: 使用列 / family 指定 / X 変換 / 誤指定モデルか
TRANSFORM_NOTES = {
    "single_gaussian": "Gaussian 3列のみ使用、変換なし（正指定・部分情報）",
    "single_bernoulli": "Bernoulli 3列のみ使用、変換なし（正指定・部分情報）",
    "single_poisson": "Poisson 3列のみ使用、変換なし（正指定・部分情報）",
    "per_column_all": "9列すべて、family_x_list 正指定、変換なし（prototype）",
    "all_gaussian": "誤指定比較用: 9列を生値のまま全列 Gaussian 強制（変換なし）",
    "all_bernoulli": "誤指定比較用: 9列を生値のまま全列 Bernoulli 強制"
                     "（変換なし; 連続値・カウントに Bernoulli 尤度を適用する"
                     " quasi-likelihood 的誤用の模擬）",
    "all_poisson": "誤指定比較用: 9列を生値のまま全列 Poisson 強制"
                   "（変換なし; 負値・0/1 値に Poisson 尤度を適用）",
    "all_bernoulli_binarized": "誤指定比較用: 明示的二値化後に全列 Bernoulli。"
                               "Gaussian 列は列中央値超→1、Poisson 列は >0→1、"
                               "Bernoulli 列はそのまま（情報を捨てる変換）",
    "y_only": "X 不使用（fix_x=True、F=0 固定）",
}


def generate(seed):
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


def binarize_mixed(X):
    """明示的二値化（Gaussian 列: 列中央値超、Poisson 列: >0）。"""
    Xb = X.copy()
    g = TRUE_BLOCKS["gauss"]
    Xb[:, g] = (X[:, g] > np.median(X[:, g], axis=0, keepdims=True)).astype(float)
    p = TRUE_BLOCKS["pois"]
    Xb[:, p] = (X[:, p] > 0).astype(float)
    return Xb


def build_conditions(X):
    """条件名 → (X_used, used_cols, run_kwargs)。"""
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
    conds["all_bernoulli_binarized"] = (binarize_mixed(X), np.arange(D),
                                        dict(family_x="bernoulli",
                                             family_x_list=None))
    conds["y_only"] = (X, np.array([], dtype=int),
                       dict(family_x="gaussian", family_x_list=None, fix_x=True))
    return conds


def block_recon_rmse(X_used, used_cols, res):
    """使用列のブロック別 X 再構成 RMSE（mean スケール、学習時の X に対し）。"""
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


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in range(N_TRIALS):
        data = generate(DATA_SEED_BASE + trial)
        train_mask, test_mask = make_pair_split(
            N, TEST_RATIO, seed=SPLIT_SEED_BASE + trial * 100)
        conds = build_conditions(data["X"])
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
                "condition": cname, "trial": trial,
                "n_cols_used": len(used_cols),
                "rmse_Z": rmse_Z,
                "w0_err": abs(res["w0"] - W0_TRUE),
                "w_err": abs(res["w"] - W_TRUE),
                "train_y_ll": m_tr.get("mean_ll", float("nan")),
                "test_y_ll": m_te.get("mean_ll", float("nan")),
                "train_y_rmse": m_tr.get("rmse", float("nan")),
                "test_y_rmse": m_te.get("rmse", float("nan")),
                "bic": res["bic"],
                "nan_occurred": res["nan_occurred"],
                "runtime_s": res["runtime_s"],
            }
            if kw.get("fix_x"):
                row.update({f"x_rmse_{b}": float("nan") for b in TRUE_BLOCKS})
            else:
                row.update(block_recon_rmse(X_used, used_cols, res))
            rows.append(row)
            print(f"t={trial} {cname:24s} rmse_Z={rmse_Z:.3f} "
                  f"te_ll={row['test_y_ll']:.3f} w_err={row['w_err']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "single_vs_joint_summary.csv", index=False)

    agg = df.groupby("condition").agg(
        rmse_Z_mean=("rmse_Z", "mean"), rmse_Z_std=("rmse_Z", "std"),
        test_y_ll_mean=("test_y_ll", "mean"), test_y_ll_std=("test_y_ll", "std"),
        test_y_rmse_mean=("test_y_rmse", "mean"),
        w0_err_mean=("w0_err", "mean"), w_err_mean=("w_err", "mean"),
        x_rmse_gauss_mean=("x_rmse_gauss", "mean"),
        x_rmse_bern_mean=("x_rmse_bern", "mean"),
        x_rmse_pois_mean=("x_rmse_pois", "mean"),
        n_nan=("nan_occurred", "sum")).reset_index()
    agg.to_csv(OUT_DIR / "single_vs_joint_agg.csv", index=False)
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
        "script": "tools/research_audit/run_per_column_single_vs_joint.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/per-column-validation",
        "n": N, "d": D, "k_true": K_TRUE,
        "w0_true": W0_TRUE, "w_true": W_TRUE,
        "fam_list_true": str(FAM_LIST_TRUE),
        "n_trials": N_TRIALS, "L": L, "num_iter": NITER,
        "test_ratio": TEST_RATIO,
        "data_seed_base": DATA_SEED_BASE,
        "model_seed_base": MODEL_SEED_BASE,
        "split_seed_base": SPLIT_SEED_BASE,
        "condition": cname, "transform_note": note,
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    } for cname, note in TRANSFORM_NOTES.items()]
    pd.DataFrame(runinfo).to_csv(OUT_DIR / "single_vs_joint_runinfo.csv",
                                 index=False)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
