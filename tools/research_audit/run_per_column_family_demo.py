"""
per-column family の必要性を示す小実験（Phase 6）。

設定: 混在属性 X（gaussian 3列 / bernoulli 3列 / poisson 3列, d=9）+
      Poisson Y の人工データ（共有 Z, k*=2, n=80）。

条件:
  percolumn_correct : 列ごとに正しい family（プロトタイプ）
  all_gaussian      : 現行実装の制約を模擬 — 全列 gaussian 強制
  all_bernoulli     : 全列 bernoulli 強制
  all_poisson       : 全列 poisson 強制

評価: RMSE(Z)（Procrustes）、w0/w 誤差、X 再構成 RMSE（族別列グループ別）。

出力:
  expfam/results/per_column_family/per_column_demo_summary.csv
  expfam/results/per_column_family/per_column_demo_agg.csv
  expfam/results/per_column_family/per_column_demo_runinfo.csv

実行: python tools/research_audit/run_per_column_family_demo.py
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

from utils_expfam import procrustes_rotation, calc_rmse   # noqa
from em_runner import run_em_experimental                 # noqa

OUT_DIR = _ROOT / "expfam" / "results" / "per_column_family"

N, D, K_TRUE = 80, 9, 2
W0_TRUE, W_TRUE = 1.2, 0.3
N_TRIALS = 5
L, NITER = 5, 8
DATA_SEED_BASE = 71000
MODEL_SEED_BASE = 72000
FAM_LIST_TRUE = ["gaussian"] * 3 + ["bernoulli"] * 3 + ["poisson"] * 3


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


def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conds = {
        "percolumn_correct": dict(family_x=None,
                                  family_x_list=FAM_LIST_TRUE),
        "all_gaussian": dict(family_x="gaussian", family_x_list=None),
        "all_bernoulli": dict(family_x="bernoulli", family_x_list=None),
        "all_poisson": dict(family_x="poisson", family_x_list=None),
    }

    rows = []
    for trial in range(N_TRIALS):
        data = generate(DATA_SEED_BASE + trial)
        for cname, kw in conds.items():
            res = run_em_experimental(
                data["X"], data["Y"], family_y="poisson",
                k=K_TRUE, L=L, num_iter=NITER,
                seed=MODEL_SEED_BASE + trial * 10, **kw)
            R, k_min = procrustes_rotation(res["Z_est"], data["Z"])
            rmse_Z = calc_rmse(data["Z"][:, :k_min],
                               res["Z_est"][:, :k_min] @ R)
            rows.append({
                "condition": cname, "trial": trial,
                "rmse_Z": rmse_Z,
                "w0_err": abs(res["w0"] - W0_TRUE),
                "w_err": abs(res["w"] - W_TRUE),
                "bic": res["bic"],
                "nan_occurred": res["nan_occurred"],
                "runtime_s": res["runtime_s"],
            })
            print(f"t={trial} {cname:18s} rmse_Z={rmse_Z:.3f} "
                  f"w_err={abs(res['w'] - W_TRUE):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_column_demo_summary.csv", index=False)

    agg = df.groupby("condition").agg(
        rmse_Z_mean=("rmse_Z", "mean"), rmse_Z_std=("rmse_Z", "std"),
        w0_err_mean=("w0_err", "mean"), w_err_mean=("w_err", "mean"),
        n_nan=("nan_occurred", "sum")).reset_index()
    agg.to_csv(OUT_DIR / "per_column_demo_agg.csv", index=False)
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
        "script": "tools/research_audit/run_per_column_family_demo.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/overdispersion-z-ablation",
        "n": N, "d": D, "k_true": K_TRUE,
        "fam_list_true": str(FAM_LIST_TRUE),
        "n_trials": N_TRIALS,
        "data_seed_base": DATA_SEED_BASE, "model_seed_base": MODEL_SEED_BASE,
        "L": L, "num_iter": NITER,
        "total_runtime_s": round(time.perf_counter() - t0, 1),
    }]).to_csv(OUT_DIR / "per_column_demo_runinfo.csv", index=False)
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
