"""
run_exp2_bic_kstar.py
--------------------
Reproduces Mikawa et al. 2024, Experiment 2 (Fig. 3):
  - Vary the TRUE latent dimension k* ∈ {1, 3, 5, 7, 9}
  - For each k*, fit the model with k_est ∈ {1, ..., 10}
  - Check whether BIC selects k_est == k*

Usage:
    cd C:/kennkyu/expfam/src
    python run_exp2_bic_kstar.py

Output:
    ../results/exp2_bic_kstar_{scenario}.csv
    ../results/fig_exp2_bic_kstar_{scenario}.pdf
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_SRC = Path(__file__).parent
_RES = _SRC.parent / "results"
_RES.mkdir(exist_ok=True)
sys.path.insert(0, str(_SRC))

from data_generator_expfam import generate_dual_data
from utils_expfam import run_em_dual, calc_bic_dual

warnings.filterwarnings("ignore")

# ── Experiment 2 settings (matching paper Table I Experiment 2) ──
N, D      = 150, 15
L, NITER  = 5, 8          # Our settings (paper uses L=10, NITER=10)
N_TRIALS  = 10
K_TRUE_LIST = [1, 3, 5, 7, 9]  # Vary true k*  (paper: same)
K_EST_LIST  = list(range(1, 11)) # k_est = 1..10 (paper: same)

# Scenarios to run (uncomment desired)
SCENARIOS = [
    ("poisson",   "bernoulli", "A"),  # Scenario A: True X=Poisson, Y=Bernoulli
    ("gaussian",  "poisson",   "B"),  # Scenario B: True X=Gaussian, Y=Poisson
    ("bernoulli", "gaussian",  "C"),  # Scenario C: True X=Bernoulli, Y=Gaussian
]

def run_exp2_bic(true_fx: str, true_fy: str, scenario_tag: str,
                 base_seed: int = 9000) -> pd.DataFrame:
    """
    For each (k_true, k_est) pair:
      - Generate data with true latent dim k_true
      - Fit model with k_est
      - Record BIC, RMSE(Z), Q_strict
    """
    records = []
    total = len(K_TRUE_LIST) * len(K_EST_LIST) * N_TRIALS
    done  = 0

    for k_true in K_TRUE_LIST:
        for k_est in K_EST_LIST:
            for trial in range(N_TRIALS):
                done += 1
                dseed = base_seed + k_true * 1000 + trial * 100
                mseed = dseed + 50
                data = generate_dual_data(n=N, d=D, k=k_true, seed=dseed,
                                          family_x=true_fx, family_y=true_fy)
                try:
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
                        "scenario": scenario_tag,
                        "k_true":   k_true,
                        "k_est":    k_est,
                        "trial":    trial,
                        "rmse_Z":   res["rmse_Z"],
                        "BIC":      bic,
                        "Q_strict": res["Q_strict"],
                        "num_params": npar,
                    })
                except Exception as e:
                    print(f"  ERROR k*={k_true} k_est={k_est} trial={trial}: {e}")
                    records.append({
                        "scenario": scenario_tag,
                        "k_true":   k_true,
                        "k_est":    k_est,
                        "trial":    trial,
                        "rmse_Z":   float("nan"),
                        "BIC":      float("nan"),
                        "Q_strict": float("nan"),
                        "num_params": None,
                    })

                if done % 10 == 0 or done == total:
                    print(f"  [{done:4d}/{total}] Scen={scenario_tag}"
                          f" k*={k_true} k_est={k_est} trial={trial}")

    return pd.DataFrame(records)


def plot_exp2(df: pd.DataFrame, true_fx: str, true_fy: str, scenario_tag: str):
    """Reproduce Fig.3 from Mikawa et al.: BIC vs k_est for each k*."""
    agg = df.groupby(["k_true", "k_est"])["BIC"].mean().reset_index()
    k_trues = sorted(agg["k_true"].unique())

    fig, axes = plt.subplots(1, len(k_trues), figsize=(4 * len(k_trues), 5),
                              sharey=False)
    if len(k_trues) == 1:
        axes = [axes]

    colors = ["#2878b5", "#f39c12", "#27ae60", "#e06c75", "#9b59b6"]
    for ax, k_t, col in zip(axes, k_trues, colors):
        sub = agg[agg["k_true"] == k_t]
        ax.plot(sub["k_est"], sub["BIC"], marker="o", color=col,
                linewidth=2, markersize=6, label=f"k*={k_t}")
        ax.axvline(k_t, color="red", ls="--", lw=1.5, alpha=0.7,
                   label=f"k*={k_t}")
        bic_best = int(sub.loc[sub["BIC"].idxmin(), "k_est"])
        ax.scatter([bic_best], [sub[sub["k_est"] == bic_best]["BIC"].values[0]],
                   color="red", s=120, zorder=5,
                   label=f"BIC min at k={bic_best}")
        ax.set_xlabel("k (estimated)")
        ax.set_ylabel("BIC (mean)")
        ax.set_title(f"k* = {k_t}")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Scenario {scenario_tag}: BIC vs k_est for each k*\n"
        f"(True X={true_fx.capitalize()}, True Y={true_fy.capitalize()}, "
        f"n={N}, d={D})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = _RES / f"fig_exp2_bic_kstar_{scenario_tag}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out}")


def summarize_bic_accuracy(df: pd.DataFrame, scenario_tag: str) -> None:
    """Print table: for each k*, what k_est does BIC select?"""
    print(f"\n=== Scenario {scenario_tag}: BIC Accuracy (BIC selects k_est == k*?) ===")
    print(f"{'k*':>4}  {'BIC best k (mean)':>20}  {'Pass?':>8}")
    print("-" * 40)
    for k_t in sorted(df["k_true"].unique()):
        sub = df[df["k_true"] == k_t].groupby("k_est")["BIC"].mean()
        best_k = int(sub.idxmin())
        passed = "✓ PASS" if best_k == k_t else f"✗ FAIL (→k={best_k})"
        print(f"{k_t:>4}  {best_k:>20}  {passed:>8}")


if __name__ == "__main__":
    for true_fx, true_fy, tag in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Scenario {tag}: X={true_fx}, Y={true_fy}")
        print(f"{'='*60}")

        df = run_exp2_bic(true_fx, true_fy, tag)
        out_csv = _RES / f"exp2_bic_kstar_{tag}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n  CSV saved: {out_csv}")

        plot_exp2(df, true_fx, true_fy, tag)
        summarize_bic_accuracy(df, tag)

    print("\nAll done.")
