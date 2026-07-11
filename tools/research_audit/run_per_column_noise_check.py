"""
per-column ノイズ属性チェック（人工 mixed-X、strict held-out Y）。

目的:
    「複数属性を入れること自体が良いのではなく、Z・Y に関係する属性を
     適切な family で入れることが重要」を示す。
    ノイズ列（Z_true と無関係）は family 指定が正しくても情報を持たない。

条件（すべて per-column、informative 9列 = gauss3+bern3+pois3 は共通）:
    no_noise       : informative 9列のみ（基準）
    gauss_noise3   : + N(0,1) ノイズ 3列（family=gaussian 正指定）
    bern_noise3    : + Bernoulli(0.5) ノイズ 3列（family=bernoulli 正指定）
    pois_noise3    : + Poisson(2.0) ノイズ 3列（family=poisson 正指定）
    gauss_noise6   : + N(0,1) ノイズ 6列（用量反応）
    gauss_noise12  : + N(0,1) ノイズ 12列（用量反応）

データ生成・評価は run_per_column_single_vs_joint.py と同一
（n=80, k*=2, Poisson-Y, strict held-out test_ratio=0.2）。

出力:
    expfam/results/per_column_family/noise_check_summary.csv
    expfam/results/per_column_family/noise_check_agg.csv
    expfam/results/per_column_family/noise_check_runinfo.csv

実行: python tools/research_audit/run_per_column_noise_check.py
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
DATA_SEED_BASE = 88000
MODEL_SEED_BASE = 89000
SPLIT_SEED_BASE = 90000
NOISE_SEED_BASE = 91000

FAM_INFO = ["gaussian"] * 3 + ["bernoulli"] * 3 + ["poisson"] * 3


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
    """条件名 → (X_used, family_x_list, n_noise, noise_family)。"""
    rng = np.random.default_rng(noise_seed)

    def gauss_noise(m):
        return rng.standard_normal((N, m))

    def bern_noise(m):
        return rng.binomial(1, 0.5, (N, m)).astype(float)

    def pois_noise(m):
        return rng.poisson(2.0, (N, m)).astype(float)

    conds = {
        "no_noise": (X, list(FAM_INFO), 0, "none"),
        "gauss_noise3": (np.hstack([X, gauss_noise(3)]),
                         FAM_INFO + ["gaussian"] * 3, 3, "gaussian"),
        "bern_noise3": (np.hstack([X, bern_noise(3)]),
                        FAM_INFO + ["bernoulli"] * 3, 3, "bernoulli"),
        "pois_noise3": (np.hstack([X, pois_noise(3)]),
                        FAM_INFO + ["poisson"] * 3, 3, "poisson"),
        "gauss_noise6": (np.hstack([X, gauss_noise(6)]),
                         FAM_INFO + ["gaussian"] * 6, 6, "gaussian"),
        "gauss_noise12": (np.hstack([X, gauss_noise(12)]),
                          FAM_INFO + ["gaussian"] * 12, 12, "gaussian"),
    }
    return conds


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in range(N_TRIALS):
        data = generate(DATA_SEED_BASE + trial)
        train_mask, test_mask = make_pair_split(
            N, TEST_RATIO, seed=SPLIT_SEED_BASE + trial * 100)
        conds = build_conditions(data["X"], NOISE_SEED_BASE + trial)
        for cname, (X_used, fam_list, n_noise, noise_fam) in conds.items():
            res = run_em_experimental(
                X_used, data["Y"], family_x=None, family_y="poisson",
                k=K_TRUE, L=L, num_iter=NITER,
                seed=MODEL_SEED_BASE + trial * 10,
                train_mask=train_mask, family_x_list=fam_list)
            R, k_min = procrustes_rotation(res["Z_est"], data["Z"])
            rmse_Z = calc_rmse(data["Z"][:, :k_min],
                               res["Z_est"][:, :k_min] @ R)
            mu_y = predict_mu_y(res)
            m_te = heldout_count_metrics(data["Y"], mu_y, test_mask, "poisson")

            # informative 9列の再構成 RMSE（ノイズ列は含めない）
            eta_x = res["Z_est"] @ res["F"].T
            mu_x = res["model"]._mean_function_x(eta_x)
            x_rmse_info = float(np.sqrt(np.mean(
                (X_used[:, :9] - mu_x[:, :9]) ** 2)))

            rows.append({
                "condition": cname, "trial": trial,
                "n_noise_cols": n_noise, "noise_family": noise_fam,
                "n_cols_total": X_used.shape[1],
                "rmse_Z": rmse_Z,
                "w0_err": abs(res["w0"] - W0_TRUE),
                "w_err": abs(res["w"] - W_TRUE),
                "test_y_ll": m_te.get("mean_ll", float("nan")),
                "test_y_rmse": m_te.get("rmse", float("nan")),
                "x_rmse_informative9": x_rmse_info,
                "nan_occurred": res["nan_occurred"],
                "runtime_s": res["runtime_s"],
            })
            print(f"t={trial} {cname:14s} rmse_Z={rmse_Z:.3f} "
                  f"te_ll={rows[-1]['test_y_ll']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "noise_check_summary.csv", index=False)

    agg = df.groupby("condition").agg(
        n_noise_cols=("n_noise_cols", "first"),
        noise_family=("noise_family", "first"),
        rmse_Z_mean=("rmse_Z", "mean"), rmse_Z_std=("rmse_Z", "std"),
        test_y_ll_mean=("test_y_ll", "mean"), test_y_ll_std=("test_y_ll", "std"),
        w_err_mean=("w_err", "mean"),
        x_rmse_info_mean=("x_rmse_informative9", "mean"),
        n_nan=("nan_occurred", "sum")).reset_index()
    agg = agg.sort_values(["noise_family", "n_noise_cols"])
    agg.to_csv(OUT_DIR / "noise_check_agg.csv", index=False)
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
        "script": "tools/research_audit/run_per_column_noise_check.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/per-column-validation",
        "n": N, "d_informative": D, "k_true": K_TRUE,
        "w0_true": W0_TRUE, "w_true": W_TRUE,
        "n_trials": N_TRIALS, "L": L, "num_iter": NITER,
        "test_ratio": TEST_RATIO,
        "noise_note": "ノイズ列は Z_true と無関係（gauss: N(0,1) / bern: p=0.5 / "
                      "pois: rate=2.0）。family は各ノイズの真の分布を正指定",
        "data_seed_base": DATA_SEED_BASE, "model_seed_base": MODEL_SEED_BASE,
        "split_seed_base": SPLIT_SEED_BASE, "noise_seed_base": NOISE_SEED_BASE,
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]).to_csv(OUT_DIR / "noise_check_runinfo.csv", index=False)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
