"""
Exp Dual-1: Latent dimension identification via BIC.

True data: family_x='poisson', family_y='bernoulli'
Setting  : n=150, d=15, k*=3,  k in {1..6},  10 independent trials
Outputs  : CSV + Figure 1 (RMSE(Z) L-shape + BIC V-shape, 2x1 subplots)
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
from utils_expfam import (                             # noqa
    run_em_dual, calc_Q_dual_strict, calc_bic_dual,
    procrustes_rotation, calc_rmse,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
N, D, K_TRUE = 150, 15, 3
FAMILY_X, FAMILY_Y = "poisson", "bernoulli"
K_LIST   = [1, 2, 3, 4, 5, 6]
N_TRIALS = 10
L, NITER = 5, 8
BASE_SEED = 4000
RESULTS_CSV = _RES / "exp_dual_1_k.csv"
FIG_PDF     = _RES / "fig_dual_1_k.pdf"
FIG_PNG     = _RES / "fig_dual_1_k.png"


def one_run(k_est: int, trial: int) -> dict:
    dseed = BASE_SEED + trial * 100
    mseed = BASE_SEED + trial * 100 + 50

    data = generate_dual_data(
        n=N, d=D, k=K_TRUE, seed=dseed,
        family_x=FAMILY_X, family_y=FAMILY_Y,
    )

    t0 = time.perf_counter()
    res = run_em_dual(
        X=data["X"], Y=data["Y"],
        true_params=data,
        family_x=FAMILY_X, family_y=FAMILY_Y,
        k=k_est, L=L, num_iter=NITER,
        seed=mseed,
        compute_strict_Q=True,
    )
    elapsed = time.perf_counter() - t0

    bic, npar = calc_bic_dual(
        Q_strict=res["Q_strict"], k=k_est, n=N, d=D,
        family_x=FAMILY_X, family_y=FAMILY_Y,
    )

    return {
        "k_est":     k_est,
        "trial":     trial,
        "rmse_Z":    res["rmse_Z"],
        "rmse_F":    res["rmse_F"],
        "rmse_sigma":res["rmse_sigma"],
        "rmse_Y":    res["rmse_Y"],
        "rmse_X":    res["rmse_X"],
        "w0_err":    res["w0_err"],
        "w_err":     res["w_err"],
        "Q_final":   res["Q_final"],
        "Q_strict":  res["Q_strict"],
        "BIC":       bic,
        "num_params":npar,
        "elapsed":   elapsed,
        "nan":       int(res["nan_occurred"]),
    }


def main():
    t_start = time.perf_counter()
    records = []
    total = len(K_LIST) * N_TRIALS
    done  = 0

    for k_est in K_LIST:
        for trial in range(N_TRIALS):
            done += 1
            print(f"  [{done:3d}/{total}] k_est={k_est}  trial={trial}", end=" ", flush=True)
            rec = one_run(k_est, trial)
            records.append(rec)
            print(f"  RMSE(Z)={rec['rmse_Z']:.4f}  BIC={rec['BIC']:.1f}")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nCSV saved: {RESULTS_CSV}")

    # ── Aggregate ──────────────────────────────────────────────────────
    agg = df.groupby("k_est").agg(
        rmse_Z_mean=("rmse_Z",  "mean"),
        rmse_Z_std =("rmse_Z",  "std"),
        BIC_mean   =("BIC",     "mean"),
        BIC_std    =("BIC",     "std"),
    ).reset_index()

    # ── Figure 1: RMSE(Z) + BIC ──────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    fig.suptitle(
        f"Exp Dual-1: Dimension Identification\n"
        f"(X={FAMILY_X.capitalize()}, Y={FAMILY_Y.capitalize()}, "
        f"n={N}, d={D}, k*={K_TRUE})",
        fontsize=13, fontweight="bold",
    )

    ks = agg["k_est"].values

    # ── Top: RMSE(Z) ─────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.errorbar(ks, agg["rmse_Z_mean"], yerr=agg["rmse_Z_std"],
                 marker="o", color="#2878b5", linewidth=2.0,
                 markersize=7, capsize=4, elinewidth=1.5,
                 label="RMSE(Z)")
    ax0.axvline(K_TRUE, color="#e06c75", linestyle="--", linewidth=1.5,
                label=f"k* = {K_TRUE}")
    ax0.set_xlabel("Estimated latent dimension k", fontsize=12)
    ax0.set_ylabel("RMSE(Z)", fontsize=12)
    ax0.set_xticks(K_LIST)
    ax0.legend(fontsize=11)
    ax0.grid(True, alpha=0.4)
    ax0.set_title("RMSE(Z) vs k  [L-shape: minimum at k*]", fontsize=11)

    # Annotate minimum
    best_k = agg.loc[agg["rmse_Z_mean"].idxmin(), "k_est"]
    best_v = agg["rmse_Z_mean"].min()
    ax0.annotate(f"k={best_k}", xy=(best_k, best_v),
                 xytext=(best_k + 0.3, best_v + 0.01),
                 fontsize=10, color="#e06c75",
                 arrowprops=dict(arrowstyle="->", color="#e06c75"))

    # ── Bottom: BIC ───────────────────────────────────────────────────
    ax1 = axes[1]
    ax1.errorbar(ks, agg["BIC_mean"], yerr=agg["BIC_std"],
                 marker="s", color="#f39c12", linewidth=2.0,
                 markersize=7, capsize=4, elinewidth=1.5,
                 label="BIC")
    ax1.axvline(K_TRUE, color="#e06c75", linestyle="--", linewidth=1.5,
                label=f"k* = {K_TRUE}")
    ax1.set_xlabel("Estimated latent dimension k", fontsize=12)
    ax1.set_ylabel("BIC", fontsize=12)
    ax1.set_xticks(K_LIST)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.4)
    ax1.set_title("BIC vs k  [V-shape: minimum at k*]", fontsize=11)

    bic_best_k = agg.loc[agg["BIC_mean"].idxmin(), "k_est"]
    bic_best_v = agg["BIC_mean"].min()
    ax1.annotate(f"k={bic_best_k}", xy=(bic_best_k, bic_best_v),
                 xytext=(bic_best_k + 0.3, bic_best_v + abs(bic_best_v)*0.02),
                 fontsize=10, color="#e06c75",
                 arrowprops=dict(arrowstyle="->", color="#e06c75"))

    plt.tight_layout()
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIG_PNG}")

    total_t = time.perf_counter() - t_start
    print(f"\nTotal time: {total_t/60:.1f} min")
    print("\n=== RMSE(Z) mean ===")
    for _, row in agg.iterrows():
        star = " <-- k*" if row.k_est == K_TRUE else ""
        print(f"  k={int(row.k_est)}: {row.rmse_Z_mean:.4f} ± {row.rmse_Z_std:.4f}{star}")
    print("\n=== BIC mean ===")
    for _, row in agg.iterrows():
        star = " <-- BEST" if row.k_est == bic_best_k else ""
        print(f"  k={int(row.k_est)}: {row.BIC_mean:.1f} ± {row.BIC_std:.1f}{star}")


if __name__ == "__main__":
    main()
