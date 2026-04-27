"""
run_exp2_bic_v2.py
------------------
Experiment 2 (BIC dimension identification with varying k*)
論文 Fig.3 相当: k* ∈ {1,3,5,7,9} × k_est=1..10 × N_TRIALS 試行

N_TRIALS=5 に設定（論文の 10 より少ないが BIC の傾向確認には十分）

出力: ../results/exp2_bic_{scenario}.csv
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

_SRC = Path(__file__).parent
_RES = _SRC.parent / "results"
_RES.mkdir(exist_ok=True)
sys.path.insert(0, str(_SRC))

from data_generator_expfam import generate_dual_data
from utils_expfam import run_em_dual, calc_bic_dual

warnings.filterwarnings("ignore")

N, D       = 150, 15
L, NITER   = 5, 8
N_TRIALS   = 5           # 論文は 10、時間節約のため 5
K_TRUE_LIST = [1, 3, 5, 7, 9]
K_EST_LIST  = list(range(1, 11))   # k_est = 1..10

SCENARIOS = [
    ("poisson",   "bernoulli", "A"),
    ("gaussian",  "poisson",   "B"),
    ("bernoulli", "gaussian",  "C"),
]

def run_exp2_bic(true_fx, true_fy, scenario_tag, base_seed=8000):
    records = []
    total = len(K_TRUE_LIST) * len(K_EST_LIST) * N_TRIALS
    done  = 0
    t_start = time.time()
    for k_true in K_TRUE_LIST:
        for k_est in K_EST_LIST:
            for trial in range(N_TRIALS):
                done += 1
                dseed = base_seed + k_true * 1000 + trial * 100
                mseed = dseed + 50
                try:
                    data = generate_dual_data(n=N, d=D, k=k_true, seed=dseed,
                                              family_x=true_fx, family_y=true_fy)
                    res = run_em_dual(
                        X=data["X"], Y=data["Y"], true_params=data,
                        family_x=true_fx, family_y=true_fy,
                        k=k_est, L=L, num_iter=NITER, seed=mseed,
                        compute_strict_Q=True,
                    )
                    bic, npar = calc_bic_dual(
                        Q_strict=res["Q_strict"], k=k_est, n=N, d=D,
                        family_x=true_fx, family_y=true_fy,
                    )
                    rmse_z = res["rmse_Z"]
                except Exception as e:
                    bic, rmse_z = float("nan"), float("nan")
                    print(f"  ERROR k*={k_true} k_est={k_est} t={trial}: {e}")

                records.append({
                    "scenario": scenario_tag, "k_true": k_true, "k_est": k_est,
                    "trial": trial, "rmse_Z": rmse_z, "BIC": bic,
                })

                if done % 25 == 0 or done == total:
                    elapsed = time.time() - t_start
                    eta = elapsed / done * (total - done)
                    print(f"  [{done:4d}/{total}] Scen={scenario_tag}"
                          f" k*={k_true} k_est={k_est:2d}"
                          f"  BIC={bic:.0f}  elapsed={elapsed/60:.1f}m"
                          f"  ETA={eta/60:.1f}m", flush=True)
    return pd.DataFrame(records)

def print_bic_accuracy(df, scenario_tag):
    print(f"\n=== Scenario {scenario_tag}: BIC 次元同定の精度 ===")
    print(f"{'k*':>4}  {'BIC 最小 k_est':>14}  {'判定':>10}")
    print("-" * 36)
    for k_t in K_TRUE_LIST:
        sub = df[df["k_true"] == k_t].groupby("k_est")["BIC"].mean()
        best_k = int(sub.idxmin())
        ok = "PASS" if best_k == k_t else f"FAIL (->k={best_k})"
        print(f"{k_t:>4}  {best_k:>14}  {ok:>10}")

if __name__ == "__main__":
    for true_fx, true_fy, tag in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Scenario {tag}: X={true_fx}, Y={true_fy}")
        out = _RES / f"exp2_bic_{tag}.csv"
        if out.exists():
            print(f"  Already exists: {out}")
            df = pd.read_csv(out)
        else:
            df = run_exp2_bic(true_fx, true_fy, tag)
            df.to_csv(out, index=False)
            print(f"  Saved: {out}")
        print_bic_accuracy(df, tag)
    print("\nAll done.")
