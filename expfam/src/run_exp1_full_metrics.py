"""
run_exp1_full_metrics.py
------------------------
Experiment 1 の全 7 指標版。
論文 Table II 相当: k=1..6 × k*=3 × 全パラメータ (smallest RMSE)

出力: ../results/exp1_full_{scenario}.csv
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
from utils_expfam import run_em_dual, calc_bic_dual, procrustes_rotation, calc_rmse

warnings.filterwarnings("ignore")

N, D, K_TRUE = 150, 15, 3
N_TRIALS     = 10
L, NITER     = 5, 8
K_LIST       = [1, 2, 3, 4, 5, 6]

SCENARIOS = [
    ("poisson",   "bernoulli", "A"),
    ("gaussian",  "poisson",   "B"),
    ("bernoulli", "gaussian",  "C"),
]

def run_exp1_full(true_fx, true_fy, scenario_tag, base_seed=7000):
    records = []
    total = len(K_LIST) * N_TRIALS
    done  = 0
    t_start = time.time()
    for k_est in K_LIST:
        for trial in range(N_TRIALS):
            done += 1
            dseed = base_seed + trial * 100
            mseed = dseed + 50
            data = generate_dual_data(n=N, d=D, k=K_TRUE, seed=dseed,
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
            records.append({
                "scenario": scenario_tag, "k_est": k_est, "trial": trial,
                "rmse_Z":     res["rmse_Z"],
                "rmse_F":     res["rmse_F"],
                "rmse_sigma": res["rmse_sigma"],
                "rmse_Y":     res["rmse_Y"],
                "rmse_X":     res["rmse_X"],
                "w0_err":     res["w0_err"],
                "w_err":      res["w_err"],
                "BIC":        bic,
                "Q_strict":   res["Q_strict"],
            })
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done)
            print(f"  Scen={scenario_tag} k={k_est} t={trial:2d}"
                  f"  Z={res['rmse_Z']:.4f} F={res['rmse_F']:.4f}"
                  f"  Y={res['rmse_Y']:.4f} BIC={bic:.0f}"
                  f"  [{done}/{total}] ETA={eta/60:.1f}min", flush=True)
    return pd.DataFrame(records)

if __name__ == "__main__":
    for true_fx, true_fy, tag in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Scenario {tag}: X={true_fx}, Y={true_fy}")
        out = _RES / f"exp1_full_{tag}.csv"
        if out.exists():
            print(f"  Already exists: {out}, skipping.")
            continue
        df = run_exp1_full(true_fx, true_fy, tag)
        df.to_csv(out, index=False)
        print(f"  Saved: {out}")
        # Print summary (smallest per k)
        g = df.groupby("k_est")[["rmse_Z","rmse_F","rmse_Y","rmse_X","w0_err","w_err","BIC"]].min().round(4)
        print(f"\n  Smallest RMSE per k (Scenario {tag}):")
        print(g.to_string())
    print("\nAll done.")
