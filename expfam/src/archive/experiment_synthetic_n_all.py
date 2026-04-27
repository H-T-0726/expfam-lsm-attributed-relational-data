"""
Step 3.8 Experiment 1 -- RMSE vs n, all families, all parameters.

For each family (Bernoulli, Poisson, Gaussian) and each n in N_LIST,
runs N_TRIALS independent EM fits and records:
  RMSE(Z), RMSE(F), RMSE(Sigma), RMSE(Y), RMSE(X), |w0 err|, |w err|

Replicates and extends NOLTA 2024 Table III to all 3 distribution families.

Settings: d=15, k=3, L=10, num_iter=10, n_trials=5

Output:
    expfam/results/synthetic_n_all_trials.csv
    expfam/results/synthetic_n_all_summary.csv
    expfam/results/synthetic_n_all_plot.png
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

from utils_expfam import run_em                                     # noqa
from data_generator_expfam import (                                 # noqa
    generate_bernoulli_data, generate_poisson_data, generate_gaussian_data,
)

OUTPUT_DIR = _ROOT / "expfam" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
N_LIST    = [50, 100, 150, 200, 250, 300]
D         = 15
K_TRUE    = 3
L         = 10
NUM_ITER  = 10
N_TRIALS  = 5
BASE_SEED = 6000

FAMILIES = ["bernoulli", "poisson", "gaussian"]

FAMILY_GEN_PARAMS = {
    "bernoulli": {"w0_true": -1.0, "w_true": 1.5},
    "poisson":   {"w0_true":  0.0, "w_true": 0.5},
    "gaussian":  {"w0_true":  0.0, "w_true": 0.5, "sigma_y_true": 0.1},
}

METRICS = ["rmse_Z", "rmse_F", "rmse_sigma", "rmse_Y", "rmse_X", "w0_err", "w_err"]
METRIC_LABELS = {
    "rmse_Z": "RMSE(Z)", "rmse_F": "RMSE(F)", "rmse_sigma": "RMSE(Sigma)",
    "rmse_Y": "RMSE(Y)", "rmse_X": "RMSE(X)",
    "w0_err": "|w0 err|", "w_err": "|w err|",
}

FAMILY_COLORS = {"bernoulli": "tab:blue", "poisson": "tab:orange", "gaussian": "tab:green"}


def generate_data(family, n, d, k, seed):
    p = FAMILY_GEN_PARAMS[family]
    if family == "bernoulli":
        return generate_bernoulli_data(n=n, d=d, k=k, seed=seed,
                                       w0_true=p["w0_true"], w_true=p["w_true"])
    if family == "poisson":
        return generate_poisson_data(n=n, d=d, k=k, seed=seed,
                                     w0_true=p["w0_true"], w_true=p["w_true"])
    return generate_gaussian_data(n=n, d=d, k=k, seed=seed,
                                  w0_true=p["w0_true"], w_true=p["w_true"],
                                  sigma_y_true=p["sigma_y_true"])


def main():
    print("=" * 65)
    print("  Step 3.8 Exp1 -- RMSE vs n, all families, all params")
    print(f"  n_list={N_LIST}, d={D}, k={K_TRUE}, L={L}, "
          f"num_iter={NUM_ITER}, n_trials={N_TRIALS}")
    print("=" * 65)

    records = []
    t_total = time.time()

    for family in FAMILIES:
        print(f"\n--- Family: {family} ---")
        sy_init = FAMILY_GEN_PARAMS[family].get("sigma_y_true", 1.0)

        for n in N_LIST:
            for trial in range(N_TRIALS):
                data_seed  = BASE_SEED + trial * 19
                model_seed = BASE_SEED + trial * 19 + 1

                data = generate_data(family, n, D, K_TRUE, data_seed)

                t0 = time.time()
                res = run_em(
                    data["X"], data["Y"], true_params=data,
                    family=family, k=K_TRUE,
                    L=L, num_iter=NUM_ITER,
                    seed=model_seed, sigma_y_init=sy_init,
                )
                elapsed = time.time() - t0

                rec = {"family": family, "n": n, "trial": trial}
                for m in METRICS:
                    rec[m] = res[m]
                rec["nan_occurred"] = res["nan_occurred"]
                rec["elapsed_s"]    = elapsed
                records.append(rec)

            vals = {m: [r[m] for r in records
                        if r["family"] == family and r["n"] == n]
                    for m in METRICS}
            print(f"  n={n:3d}  Z={np.mean(vals['rmse_Z']):.4f}  "
                  f"F={np.mean(vals['rmse_F']):.4f}  "
                  f"Sig={np.mean(vals['rmse_sigma']):.4f}  "
                  f"Y={np.mean(vals['rmse_Y']):.4f}  "
                  f"X={np.mean(vals['rmse_X']):.4f}  "
                  f"w0={np.mean(vals['w0_err']):.4f}  "
                  f"w={np.mean(vals['w_err']):.4f}")

    # ----------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------
    df = pd.DataFrame(records)
    trial_csv = OUTPUT_DIR / "synthetic_n_all_trials.csv"
    df.to_csv(trial_csv, index=False)

    agg_dict = {}
    for m in METRICS:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"]  = (m, "std")
        agg_dict[f"{m}_min"]  = (m, "min")
    agg_dict["nan_count"] = ("nan_occurred", "sum")

    summary = df.groupby(["family", "n"]).agg(**agg_dict).reset_index()
    summary_csv = OUTPUT_DIR / "synthetic_n_all_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # ----------------------------------------------------------------
    # Print summary table (Poisson, all params)
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  NOLTA Table III equivalent: Poisson, RMSE vs n (mean over 5 trials)")
    print("=" * 80)
    print(f"{'n':>5}" + "".join(f"  {METRIC_LABELS[m]:>10}" for m in METRICS))
    print("-" * 80)
    pois = summary[summary.family == "poisson"]
    for _, row in pois.iterrows():
        line = f"{int(row.n):>5}"
        for m in METRICS:
            line += f"  {row[f'{m}_mean']:>10.4f}"
        print(line)

    # ----------------------------------------------------------------
    # Plot: 7 subplots (one per metric), 3 lines (one per family)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    n_vals = np.array(N_LIST)

    for ax_i, m in enumerate(METRICS):
        ax = axes_flat[ax_i]
        for fam in FAMILIES:
            fam_df = summary[summary.family == fam]
            means = fam_df[f"{m}_mean"].values
            stds  = fam_df[f"{m}_std"].values
            ax.errorbar(n_vals, means, yerr=stds,
                        fmt="o-", color=FAMILY_COLORS[fam], capsize=4,
                        linewidth=1.8, markersize=5, label=fam)
        ax.set_xlabel("n", fontsize=10)
        ax.set_ylabel(METRIC_LABELS[m], fontsize=10)
        ax.set_title(f"{METRIC_LABELS[m]} vs n", fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_vals)
        ax.set_ylim(bottom=0)

    # Hide unused subplot
    axes_flat[-1].set_visible(False)

    plt.suptitle(
        f"All-Parameter RMSE vs n -- 3 Families\n"
        f"(d={D}, k={K_TRUE}, L={L}, num_iter={NUM_ITER}, n_trials={N_TRIALS})",
        fontsize=12
    )
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "synthetic_n_all_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    total = time.time() - t_total
    print(f"\n  Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"  Saved: {trial_csv}")
    print(f"  Saved: {summary_csv}")
    print(f"  Saved: {plot_path}")
    print("\n[experiment_synthetic_n_all DONE]")


if __name__ == "__main__":
    main()
