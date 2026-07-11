"""
per-column 属性追加 ablation（人工 mixed-X、strict held-out Y）。

目的:
    属性ブロックを 1 つずつ追加したとき、Z 推定と held-out Y 予測が
    どう変わるかを見る。「属性を増やせば必ず良い」わけではないことの確認
    （最終条件はノイズ属性を追加）。

条件（追加順、すべて per-column 正指定）:
    1. y_only                    : X なし（fix_x=True）
    2. bern_only                 : Bernoulli 3列
    3. bern_gauss                : + Gaussian 3列（計6列）
    4. bern_gauss_pois           : + Poisson 3列（計9列 = full mixed-X）
    5. bern_gauss_pois_noise3    : + Z_true と無関係な Gaussian ノイズ 3列（計12列）

データ生成・評価は run_per_column_single_vs_joint.py と同一
（n=80, k*=2, Poisson-Y, strict held-out test_ratio=0.2）。

出力:
    expfam/results/per_column_family/attribute_ablation_summary.csv
    expfam/results/per_column_family/attribute_ablation_agg.csv
    expfam/results/per_column_family/attribute_ablation_runinfo.csv

実行: python tools/research_audit/run_per_column_attribute_ablation.py
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
N_NOISE = 3
DATA_SEED_BASE = 84000
MODEL_SEED_BASE = 85000
SPLIT_SEED_BASE = 86000
NOISE_SEED_BASE = 87000

GAUSS, BERN, POIS = np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)


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


def build_conditions(X, noise_seed):
    """条件名 → (X_used, family_x_list or None, fix_x, n_cols, step順)。"""
    rng = np.random.default_rng(noise_seed)
    noise = rng.standard_normal((N, N_NOISE))          # Z_true と無関係
    conds = {
        "y_only": (X, None, True, 0),
        "bern_only": (X[:, BERN], ["bernoulli"] * 3, False, 3),
        "bern_gauss": (np.hstack([X[:, BERN], X[:, GAUSS]]),
                       ["bernoulli"] * 3 + ["gaussian"] * 3, False, 6),
        "bern_gauss_pois": (np.hstack([X[:, BERN], X[:, GAUSS], X[:, POIS]]),
                            ["bernoulli"] * 3 + ["gaussian"] * 3
                            + ["poisson"] * 3, False, 9),
        "bern_gauss_pois_noise3": (
            np.hstack([X[:, BERN], X[:, GAUSS], X[:, POIS], noise]),
            ["bernoulli"] * 3 + ["gaussian"] * 3 + ["poisson"] * 3
            + ["gaussian"] * N_NOISE, False, 9 + N_NOISE),
    }
    return conds


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step_order = ["y_only", "bern_only", "bern_gauss", "bern_gauss_pois",
                  "bern_gauss_pois_noise3"]
    rows = []
    for trial in range(N_TRIALS):
        data = generate(DATA_SEED_BASE + trial)
        train_mask, test_mask = make_pair_split(
            N, TEST_RATIO, seed=SPLIT_SEED_BASE + trial * 100)
        conds = build_conditions(data["X"], NOISE_SEED_BASE + trial)
        for step, cname in enumerate(step_order):
            X_used, fam_list, fix_x, n_cols = conds[cname]
            res = run_em_experimental(
                X_used, data["Y"],
                family_x="gaussian" if fam_list is None else None,
                family_y="poisson",
                k=K_TRUE, L=L, num_iter=NITER,
                seed=MODEL_SEED_BASE + trial * 10,
                train_mask=train_mask,
                family_x_list=fam_list, fix_x=fix_x)
            R, k_min = procrustes_rotation(res["Z_est"], data["Z"])
            rmse_Z = calc_rmse(data["Z"][:, :k_min],
                               res["Z_est"][:, :k_min] @ R)
            mu_y = predict_mu_y(res)
            m_te = heldout_count_metrics(data["Y"], mu_y, test_mask, "poisson")
            m_tr = heldout_count_metrics(data["Y"], mu_y, train_mask, "poisson")
            rows.append({
                "condition": cname, "step": step, "trial": trial,
                "n_cols_used": n_cols,
                "rmse_Z": rmse_Z,
                "w0_err": abs(res["w0"] - W0_TRUE),
                "w_err": abs(res["w"] - W_TRUE),
                "train_y_ll": m_tr.get("mean_ll", float("nan")),
                "test_y_ll": m_te.get("mean_ll", float("nan")),
                "test_y_rmse": m_te.get("rmse", float("nan")),
                "nan_occurred": res["nan_occurred"],
                "runtime_s": res["runtime_s"],
            })
            print(f"t={trial} step{step} {cname:24s} rmse_Z={rmse_Z:.3f} "
                  f"te_ll={rows[-1]['test_y_ll']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "attribute_ablation_summary.csv", index=False)

    agg = df.groupby(["step", "condition"]).agg(
        n_cols_used=("n_cols_used", "first"),
        rmse_Z_mean=("rmse_Z", "mean"), rmse_Z_std=("rmse_Z", "std"),
        test_y_ll_mean=("test_y_ll", "mean"), test_y_ll_std=("test_y_ll", "std"),
        test_y_rmse_mean=("test_y_rmse", "mean"),
        w_err_mean=("w_err", "mean"),
        n_nan=("nan_occurred", "sum")).reset_index()
    agg.to_csv(OUT_DIR / "attribute_ablation_agg.csv", index=False)
    print("\n=== Aggregated ===")
    print(agg.to_string(index=False))

    def git_head():
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=_ROOT).stdout.strip()
        except Exception:
            return "unknown"

    pd.DataFrame([{
        "script": "tools/research_audit/run_per_column_attribute_ablation.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/per-column-validation",
        "n": N, "d_full": D, "k_true": K_TRUE,
        "w0_true": W0_TRUE, "w_true": W_TRUE,
        "n_trials": N_TRIALS, "L": L, "num_iter": NITER,
        "test_ratio": TEST_RATIO, "n_noise_cols": N_NOISE,
        "noise_note": "noise 列は Z_true と無関係な N(0,1)。family は gaussian "
                      "正指定（family が正しくても情報がない属性の効果を見る）",
        "data_seed_base": DATA_SEED_BASE, "model_seed_base": MODEL_SEED_BASE,
        "split_seed_base": SPLIT_SEED_BASE, "noise_seed_base": NOISE_SEED_BASE,
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]).to_csv(OUT_DIR / "attribute_ablation_runinfo.csv", index=False)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
