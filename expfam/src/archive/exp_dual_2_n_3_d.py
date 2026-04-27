"""
Exp Dual-2 & 3: Asymptotic consistency and scalability.

Exp 2: n in {50,100,150,200,250,300}, d=15, k=3  [n variation]
Exp 3: d in {5,10,15,20,25,30},       n=150, k=3  [d variation]

True data: family_x='poisson', family_y='bernoulli'
10 independent trials each.

Outputs:
  - CSV: exp_dual_2_n.csv, exp_dual_3_d.csv
  - Figure 2: 7-metric RMSE vs n (with error bars)
  - Figure 3: 7-metric RMSE vs d (with error bars)
"""

import sys, time, warnings
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
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

from data_generator_expfam import generate_dual_data   # noqa
from utils_expfam import run_em_dual                   # noqa

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
FAMILY_X, FAMILY_Y = "poisson", "bernoulli"
K_TRUE   = 3
N_TRIALS = 10
L, NITER = 5, 8
BASE_SEED = 5000

# Exp 2: n variation (d fixed)
N_LIST, D_FIXED = [50, 100, 150, 200, 250, 300], 15
# Exp 3: d variation (n fixed)
D_LIST, N_FIXED = [5, 10, 15, 20, 25, 30], 150

METRICS = ["rmse_Z", "rmse_F", "rmse_sigma", "rmse_Y", "rmse_X", "w0_err", "w_err"]
METRIC_LABELS = {
    "rmse_Z":     "RMSE(Z)",
    "rmse_F":     "RMSE(F)",
    "rmse_sigma": "RMSE(Sigma)",
    "rmse_Y":     "RMSE(mu_Y)",
    "rmse_X":     "RMSE(mu_X)",
    "w0_err":     "|w0 error|",
    "w_err":      "|w error|",
}
# Colors for each metric (colorblind-friendly)
METRIC_COLORS = {
    "rmse_Z":     "#2878b5",
    "rmse_F":     "#f39c12",
    "rmse_sigma": "#27ae60",
    "rmse_Y":     "#e06c75",
    "rmse_X":     "#9b59b6",
    "w0_err":     "#1abc9c",
    "w_err":      "#e67e22",
}
METRIC_MARKERS = {
    "rmse_Z": "o", "rmse_F": "s", "rmse_sigma": "^",
    "rmse_Y": "D", "rmse_X": "v", "w0_err": "p", "w_err": "h",
}


def one_run(n, d, trial):
    dseed = BASE_SEED + trial * 1000 + n + d
    mseed = dseed + 500
    data = generate_dual_data(
        n=n, d=d, k=K_TRUE, seed=dseed,
        family_x=FAMILY_X, family_y=FAMILY_Y,
    )
    res = run_em_dual(
        X=data["X"], Y=data["Y"],
        true_params=data,
        family_x=FAMILY_X, family_y=FAMILY_Y,
        k=K_TRUE, L=L, num_iter=NITER, seed=mseed,
    )
    return {m: res[m] for m in METRICS}


def run_experiment(vary_name, vary_list, fixed_n, fixed_d):
    """Run one sweep (n or d variation) and return DataFrame."""
    records = []
    total = len(vary_list) * N_TRIALS
    done  = 0
    for val in vary_list:
        n = val if vary_name == "n" else fixed_n
        d = val if vary_name == "d" else fixed_d
        for trial in range(N_TRIALS):
            done += 1
            print(f"  [{done:3d}/{total}] {vary_name}={val}  trial={trial}", end=" ", flush=True)
            rec = one_run(n, d, trial)
            rec[vary_name] = val
            rec["trial"]   = trial
            records.append(rec)
            print(f"  RMSE(Z)={rec['rmse_Z']:.4f}")
    return pd.DataFrame(records)


def plot_metrics(df, x_col, x_label, title, out_pdf, out_png):
    """7-metric error-bar plot."""
    agg = df.groupby(x_col).agg(
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std":  (m, "std")  for m in METRICS},
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Top: Z, F, Sigma, Y, X (structural metrics)
    ax0 = axes[0]
    for m in ["rmse_Z", "rmse_F", "rmse_sigma", "rmse_Y", "rmse_X"]:
        ax0.errorbar(
            agg[x_col], agg[f"{m}_mean"], yerr=agg[f"{m}_std"],
            marker=METRIC_MARKERS[m], color=METRIC_COLORS[m],
            linewidth=1.8, markersize=6, capsize=3,
            label=METRIC_LABELS[m],
        )
    ax0.set_xlabel(x_label, fontsize=12)
    ax0.set_ylabel("RMSE", fontsize=12)
    ax0.legend(fontsize=10, ncol=2)
    ax0.grid(True, alpha=0.4)
    ax0.set_title("Structural parameters (Z, F, Sigma, Y, X)", fontsize=11)
    ax0.set_xticks(agg[x_col].values)

    # Bottom: w0, w errors
    ax1 = axes[1]
    for m in ["w0_err", "w_err"]:
        ax1.errorbar(
            agg[x_col], agg[f"{m}_mean"], yerr=agg[f"{m}_std"],
            marker=METRIC_MARKERS[m], color=METRIC_COLORS[m],
            linewidth=1.8, markersize=6, capsize=3,
            label=METRIC_LABELS[m],
        )
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel("Absolute Error", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.4)
    ax1.set_title("Relational parameters (w0, w)", fontsize=11)
    ax1.set_xticks(agg[x_col].values)

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_png}")


def main():
    t_start = time.perf_counter()

    print("\n=== Exp Dual-2: n variation ===")
    df2 = run_experiment("n", N_LIST, N_FIXED, D_FIXED)
    df2.to_csv(_RES / "exp_dual_2_n.csv", index=False)
    print(f"CSV saved: {_RES / 'exp_dual_2_n.csv'}")
    plot_metrics(
        df2, x_col="n", x_label="Sample size n",
        title=(f"Exp Dual-2: Asymptotic Consistency vs n\n"
               f"(X={FAMILY_X.capitalize()}, Y={FAMILY_Y.capitalize()}, d={D_FIXED}, k={K_TRUE})"),
        out_pdf=_RES / "fig_dual_2_n.pdf",
        out_png=_RES / "fig_dual_2_n.png",
    )

    print("\n=== Exp Dual-3: d variation ===")
    df3 = run_experiment("d", D_LIST, N_FIXED, D_FIXED)
    df3.to_csv(_RES / "exp_dual_3_d.csv", index=False)
    print(f"CSV saved: {_RES / 'exp_dual_3_d.csv'}")
    plot_metrics(
        df3, x_col="d", x_label="Attribute dimension d",
        title=(f"Exp Dual-3: Scalability vs d\n"
               f"(X={FAMILY_X.capitalize()}, Y={FAMILY_Y.capitalize()}, n={N_FIXED}, k={K_TRUE})"),
        out_pdf=_RES / "fig_dual_3_d.pdf",
        out_png=_RES / "fig_dual_3_d.png",
    )

    total_t = time.perf_counter() - t_start
    print(f"\nTotal: {total_t/60:.1f} min")

    # Print summary for Exp 2
    agg2 = df2.groupby("n")["rmse_Z"].agg(["mean", "std"]).reset_index()
    print("\n--- Exp 2 RMSE(Z) summary ---")
    for _, row in agg2.iterrows():
        print(f"  n={int(row['n'])}: {row['mean']:.4f} +- {row['std']:.4f}")

    agg3 = df3.groupby("d")["rmse_Z"].agg(["mean", "std"]).reset_index()
    print("\n--- Exp 3 RMSE(Z) summary ---")
    for _, row in agg3.iterrows():
        print(f"  d={int(row['d'])}: {row['mean']:.4f} +- {row['std']:.4f}")


if __name__ == "__main__":
    main()
