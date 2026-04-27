"""
Experiment: RMSE vs sample size n (Poisson model, fixed d=15, k=3).

Tests whether estimation accuracy improves as n increases.
Multiple independent trials per n provide statistical confidence.

Output:
    expfam/results/synthetic_n_trials.csv   -- per-trial raw results
    expfam/results/synthetic_n_summary.csv  -- mean/min/std per n
    expfam/results/synthetic_n_plot.png     -- RMSE(Z), RMSE(X) vs n
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).parent.parent.parent
_SRC  = Path(__file__).parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from utils_expfam import run_em, calc_rmse          # noqa
from data_generator_expfam import generate_poisson_data  # noqa

OUTPUT_DIR = _ROOT / "expfam" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
N_LIST     = [50, 100, 150, 200, 250]
D          = 15
K_TRUE     = 3
L          = 10
NUM_ITER   = 10
N_TRIALS   = 5
BASE_SEED  = 2000   # trial t uses seed BASE_SEED + t * 7 for data gen
                    # and BASE_SEED + t * 7 + 1 for model init

FAMILY     = "poisson"
W0_TRUE    = 0.0
W_TRUE     = 0.5


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Experiment: RMSE vs n  (Poisson, d=15, k=3)")
    print(f"  n_list={N_LIST}, n_trials={N_TRIALS}, "
          f"L={L}, num_iter={NUM_ITER}")
    print("=" * 65)

    trial_records = []
    t_total = time.time()

    for n in N_LIST:
        print(f"\n--- n={n} ---")
        for trial in range(N_TRIALS):
            data_seed  = BASE_SEED + trial * 7
            model_seed = BASE_SEED + trial * 7 + 1

            data = generate_poisson_data(
                n=n, d=D, k=K_TRUE, seed=data_seed,
                w0_true=W0_TRUE, w_true=W_TRUE
            )
            X = data["X"]
            Y = data["Y"]

            t0 = time.time()
            res = run_em(
                X, Y, true_params=data,
                family=FAMILY, k=K_TRUE,
                L=L, num_iter=NUM_ITER,
                seed=model_seed, verbose=False,
            )
            elapsed = time.time() - t0

            rec = {
                "n": n, "k_true": K_TRUE, "trial": trial,
                "data_seed": data_seed,
                "rmse_Z": res["rmse_Z"],
                "rmse_X": res["rmse_X"],
                "Q_final": res["Q_final"],
                "w0_est": res["w0_est"],
                "w_est":  res["w_est"],
                "nan_occurred": res["nan_occurred"],
                "elapsed_s": elapsed,
            }
            trial_records.append(rec)

            print(f"  trial={trial}  RMSE(Z)={res['rmse_Z']:.4f}  "
                  f"RMSE(X)={res['rmse_X']:.4f}  "
                  f"Q={res['Q_final']:.1f}  "
                  f"w0={res['w0_est']:.3f}  w={res['w_est']:.3f}  "
                  f"({elapsed:.1f}s)"
                  + ("  [NaN!]" if res["nan_occurred"] else ""))

    # ── Save trial CSV ────────────────────────────────────────────────
    df_trials = pd.DataFrame(trial_records)
    trial_csv = OUTPUT_DIR / "synthetic_n_trials.csv"
    df_trials.to_csv(trial_csv, index=False)

    # ── Summary (mean / min / std) ────────────────────────────────────
    summary = (
        df_trials.groupby("n")
        .agg(
            rmse_Z_mean=("rmse_Z", "mean"),
            rmse_Z_std= ("rmse_Z", "std"),
            rmse_Z_min= ("rmse_Z", "min"),
            rmse_X_mean=("rmse_X", "mean"),
            rmse_X_std= ("rmse_X", "std"),
            rmse_X_min= ("rmse_X", "min"),
            Q_mean=     ("Q_final","mean"),
            n_trials=   ("trial",  "count"),
        )
        .reset_index()
    )
    summary_csv = OUTPUT_DIR / "synthetic_n_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY: RMSE vs n")
    print("=" * 65)
    print(f"{'n':>6} {'RMSE_Z_mean':>14} {'RMSE_Z_std':>12} "
          f"{'RMSE_Z_min':>12} {'RMSE_X_mean':>14}")
    print("-" * 65)
    for _, row in summary.iterrows():
        print(f"{int(row.n):>6}  {row.rmse_Z_mean:>13.6f}  "
              f"{row.rmse_Z_std:>11.6f}  {row.rmse_Z_min:>11.6f}  "
              f"{row.rmse_X_mean:>13.6f}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ns = summary["n"].values

    for ax, col, title in [
        (axes[0], "rmse_Z", "RMSE(Z) [Procrustes] vs n"),
        (axes[1], "rmse_X", "RMSE(X) vs n"),
    ]:
        mean_col = f"{col}_mean"
        std_col  = f"{col}_std"
        min_col  = f"{col}_min"
        ax.errorbar(ns, summary[mean_col], yerr=summary[std_col],
                    fmt="o-", capsize=5, color="tab:blue",
                    linewidth=2, markersize=7, label="mean ± std")
        ax.plot(ns, summary[min_col], "s--", color="tab:orange",
                linewidth=1.5, markersize=6, label="min")
        ax.set_xlabel("n (sample size)", fontsize=12)
        ax.set_ylabel(col.upper().replace("_", "(").replace("(", "(") + ")",
                      fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ns)

    plt.suptitle(
        f"Poisson LSM — RMSE vs n  (d={D}, k={K_TRUE}, "
        f"n_trials={N_TRIALS}, num_iter={NUM_ITER})",
        fontsize=12
    )
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "synthetic_n_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    total = time.time() - t_total
    print(f"\n  Total time: {total:.1f}s")
    print(f"  Saved: {trial_csv}")
    print(f"  Saved: {summary_csv}")
    print(f"  Saved: {plot_path}")
    print("\n[experiment_synthetic_n DONE]")


if __name__ == "__main__":
    main()
