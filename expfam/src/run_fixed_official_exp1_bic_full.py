"""
fixed版 Exp1 BIC full 実験スクリプト。

DualExpFamLSMFixed (Y-side 0.5 なし) を使って Exp1 (BIC k選択) を
full条件（10試行）で実行する。修論用の正式結果。

3シナリオ × k=1~6 × 10試行 = 180 fits。

run_fixed_official_exp1_bic_quick.py と同一ロジック。
試行数と出力先のみ変更。既存ファイルは一切変更しない。

実行例:
  cd expfam/src
  python run_fixed_official_exp1_bic_full.py
"""

import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SRC  = Path(__file__).parent
_ROOT = _SRC.parent.parent

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from data_generator_expfam import generate_dual_data       # noqa: E402
from model_dual_expfam_fixed import DualExpFamLSMFixed     # noqa: E402
from utils_expfam import (                                  # noqa: E402
    calc_Q_dual_strict, calc_bic_dual,
    calc_rmse, procrustes_rotation,
)

# ─────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────

OUT_DIR = _ROOT / "expfam" / "results" / "fixed_official" / "exp1"

N, D, K_TRUE = 150, 15, 3
L, NITER     = 5, 8
K_LIST       = [1, 2, 3, 4, 5, 6]
TRIALS       = list(range(10))   # full版: 10試行 (0-9)

# exp_scenario_lib.run_exp1 と同一のseed規則
# run_scenario(seed_offset=...) -> run_exp1(base_seed=4000+seed_offset)
# seed_offset: A=0, B=1000, C=2000
SEED_OFFSETS = {"A": 0, "B": 1000, "C": 2000}
EXP1_BASE    = 4000

SCENARIOS = [
    ("A", "poisson",   "bernoulli"),
    ("B", "gaussian",  "poisson"),
    ("C", "bernoulli", "gaussian"),
]


# ─────────────────────────────────────────────────────────────────────
# Fixed-version EM runner with Q_strict computation
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed_bic(X, Y, true_params, family_x, family_y, k,
                     L=5, num_iter=8, seed=42):
    """
    MC-EM using DualExpFamLSMFixed (Y-side 0.5 removed).
    Computes Q_strict for BIC after EM convergence.
    NaN fallback: halve newton_alpha on NaN (up to 2 retries).
    """
    n, d = X.shape
    max_retries = 2

    for retry in range(max_retries + 1):
        newton_alpha = 0.5 / (2 ** retry)
        rng = np.random.default_rng(seed + retry * 1000)

        model = DualExpFamLSMFixed(
            n=n, d=d, k=k, L=L,
            family_x=family_x, family_y=family_y,
        )
        model.initialize_params(true_params=true_params,
                                seed=seed + retry * 1000)

        # Informed init: Y side
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if family_y == "bernoulli":
            density = float(np.clip(np.mean(Y), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"]  = 0.5
        elif family_y == "poisson":
            upper = Y[upper_mask]
            mean_cnt = float(upper[upper > 0].mean()) if np.any(upper > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"]  = 0.1 / (2 ** retry)
        else:  # gaussian
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"]  = 0.5
            model.sigma_y = float(max(upper_vals.std(), 0.01))

        # Informed init: X side
        if family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        Z     = model.params["Z"].copy()
        F     = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0    = model.params["w0"]
        w     = model.params["w"]
        var_z = model.params["var_z"]

        Z_prev    = Z.copy()
        nan_count = 0

        for iteration in range(1, num_iter + 1):
            # E-step
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(
                    dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w)
                )
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_count += 1
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            # M-step
            F     = model.calc_F(X, Z_samples)
            sigma = model.calc_sigma(X, Z_samples, F)
            w0    = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            w     = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

    # Update model.params to latest values before Q_strict computation
    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})

    # Q_strict via inherited calc_log_likelihood_X/Y (unchanged in Fixed subclass)
    Q_strict = calc_Q_dual_strict(
        X, Y, Z_samples, F, sigma, var_z, w0, w, model
    )

    # Metrics
    Z_est    = Z_samples[:, :, -1]
    mu_x_est = model._mean_function_x(Z_est @ F.T)
    rmse_X   = calc_rmse(X, mu_x_est)

    R, k_min = procrustes_rotation(Z_est, true_params["Z"])
    Z_rot    = Z_est[:, :k_min] @ R
    rmse_Z   = calc_rmse(true_params["Z"][:, :k_min], Z_rot)

    upper_mask_sq = np.triu(np.ones((n, n), dtype=bool), k=1)
    eta_y_est  = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y_est   = model._mean_function(eta_y_est)
    eta_y_true = float(true_params["w0"]) + float(true_params["w"]) * (
        true_params["Z"] @ true_params["Z"].T)
    mu_y_true  = model._mean_function(eta_y_true)
    rmse_Y     = calc_rmse(mu_y_est[upper_mask_sq], mu_y_true[upper_mask_sq])

    return {
        "Q_strict":     Q_strict,
        "rmse_Z":       rmse_Z,
        "rmse_X":       rmse_X,
        "rmse_Y":       rmse_Y,
        "w0_est":       float(w0),
        "w_est":        float(w),
        "nan_occurred": nan_occurred,
        "nan_count":    nan_count,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(SCENARIOS) * len(K_LIST) * len(TRIALS)
    print(f"fixed Exp1 BIC full: {len(SCENARIOS)} scenarios x "
          f"{len(K_LIST)} k x {len(TRIALS)} trials = {total} fits")
    print(f"Output: {OUT_DIR}")
    print()

    t0   = time.perf_counter()
    done = 0
    rows = []

    for scen_tag, fam_x, fam_y in SCENARIOS:
        base_seed = EXP1_BASE + SEED_OFFSETS[scen_tag]

        for k_est in K_LIST:
            for trial in TRIALS:
                done += 1
                seed_data  = base_seed + trial * 100
                seed_model = seed_data + 50

                try:
                    data = generate_dual_data(
                        n=N, d=D, k=K_TRUE, seed=seed_data,
                        family_x=fam_x, family_y=fam_y,
                    )
                    res = run_em_fixed_bic(
                        X=data["X"], Y=data["Y"], true_params=data,
                        family_x=fam_x, family_y=fam_y,
                        k=k_est, L=L, num_iter=NITER, seed=seed_model,
                    )
                    bic, num_params = calc_bic_dual(
                        Q_strict=res["Q_strict"], k=k_est, n=N, d=D,
                        family_x=fam_x, family_y=fam_y,
                    )
                    success   = True
                    err_msg   = ""
                    elapsed   = (time.perf_counter() - t0) / 60
                    print(
                        f"  [{done:3d}/{total}] Scen={scen_tag} k={k_est} t={trial}"
                        f"  rmse_Z={res['rmse_Z']:.4f}"
                        f"  Q_strict={res['Q_strict']:.2f}"
                        f"  BIC={bic:.1f}"
                        f"  [{elapsed:.1f} min]"
                    )

                except Exception as e:
                    traceback.print_exc()
                    bic        = float("nan")
                    num_params = -1
                    res = {
                        "Q_strict": float("nan"),
                        "rmse_Z":   float("nan"),
                        "rmse_X":   float("nan"),
                        "rmse_Y":   float("nan"),
                        "nan_occurred": True,
                        "nan_count":    -1,
                        "w0_est":   float("nan"),
                        "w_est":    float("nan"),
                    }
                    success = False
                    err_msg = str(e)
                    elapsed = (time.perf_counter() - t0) / 60
                    print(
                        f"  [{done:3d}/{total}] Scen={scen_tag} k={k_est} t={trial}"
                        f"  ERROR: {err_msg[:60]}  [{elapsed:.1f} min]"
                    )

                rows.append({
                    "scenario":      scen_tag,
                    "trial":         trial,
                    "seed_data":     seed_data,
                    "seed_model":    seed_model,
                    "k_est":         k_est,
                    "k_true":        K_TRUE,
                    "family_x":      fam_x,
                    "family_y":      fam_y,
                    "n":             N,
                    "d":             D,
                    "q_strict":      res["Q_strict"],
                    "bic":           bic,
                    "num_params":    num_params,
                    "rmse_z":        res["rmse_Z"],
                    "rmse_x":        res["rmse_X"],
                    "rmse_y":        res["rmse_Y"],
                    "success":       success,
                    "error_message": err_msg,
                    "note":          f"fixed_official full k_true={K_TRUE}",
                })

    df = pd.DataFrame(rows)

    # ── summary CSV ────────────────────────────────────────────────────
    out_summary = OUT_DIR / "fixed_exp1_bic_full_summary.csv"
    df.to_csv(out_summary, index=False)
    print(f"\nSaved: {out_summary}  ({len(df)} rows)")

    # ── agg CSV: scenario x k ─────────────────────────────────────────
    agg_rows = []
    for (scen_tag, k_est), sub in df.groupby(
        ["scenario", "k_est"], sort=True
    ):
        fam_x = sub["family_x"].iloc[0]
        fam_y = sub["family_y"].iloc[0]
        agg_rows.append({
            "scenario":      scen_tag,
            "family_x":      fam_x,
            "family_y":      fam_y,
            "k_est":         k_est,
            "n_trials":      len(sub),
            "bic_mean":      sub["bic"].mean(),
            "bic_std":       sub["bic"].std(),
            "q_strict_mean": sub["q_strict"].mean(),
            "rmse_z_mean":   sub["rmse_z"].mean(),
            "rmse_z_std":    sub["rmse_z"].std(),
            "success_rate":  float(sub["success"].mean()),
            "note":          "fixed_official full",
        })
    agg_df = pd.DataFrame(agg_rows)
    out_agg = OUT_DIR / "fixed_exp1_bic_full_agg.csv"
    agg_df.to_csv(out_agg, index=False)
    print(f"Saved: {out_agg}  ({len(agg_df)} rows)")

    # ── bestk CSV: scenario level ──────────────────────────────────────
    bestk_rows = []
    for scen_tag, sub_agg in agg_df.groupby("scenario", sort=True):
        fam_x = sub_agg["family_x"].iloc[0]
        fam_y = sub_agg["family_y"].iloc[0]

        best_idx = sub_agg["bic_mean"].idxmin()
        best_k   = int(sub_agg.loc[best_idx, "k_est"])

        bic_by_k = {
            int(r["k_est"]): r["bic_mean"]
            for _, r in sub_agg.iterrows()
        }

        bestk_rows.append({
            "scenario":           scen_tag,
            "family_x":           fam_x,
            "family_y":           fam_y,
            "best_k_by_bic_mean": best_k,
            "k_true":             K_TRUE,
            "bic_at_k1":          bic_by_k.get(1, float("nan")),
            "bic_at_k2":          bic_by_k.get(2, float("nan")),
            "bic_at_k3":          bic_by_k.get(3, float("nan")),
            "bic_at_k4":          bic_by_k.get(4, float("nan")),
            "bic_at_k5":          bic_by_k.get(5, float("nan")),
            "bic_at_k6":          bic_by_k.get(6, float("nan")),
            "selected_correctly": best_k == K_TRUE,
            "note": (
                f"fixed_official full, BIC mean over {len(TRIALS)} trials"
            ),
        })

    bestk_df = (
        pd.DataFrame(bestk_rows)
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    out_bestk = OUT_DIR / "fixed_exp1_bic_full_bestk.csv"
    bestk_df.to_csv(out_bestk, index=False)
    print(f"Saved: {out_bestk}  ({len(bestk_df)} rows)")

    elapsed_total = (time.perf_counter() - t0) / 60
    print(f"\nDone in {elapsed_total:.1f} min")

    # ── Final summary ─────────────────────────────────────────────────
    print("\n=== BIC best k summary (full 10 trials) ===")
    for _, r in bestk_df.iterrows():
        status = "OK" if r["selected_correctly"] else "NG"
        print(f"  Scen {r['scenario']}"
              f" ({r['family_x'][:4]}-{r['family_y'][:4]})"
              f": best_k={r['best_k_by_bic_mean']}"
              f"  (k_true={r['k_true']}) [{status}]")

    print("\n=== RMSE(Z) at k_true=3 ===")
    for scen_tag in ["A", "B", "C"]:
        row = agg_df[
            (agg_df["scenario"] == scen_tag) & (agg_df["k_est"] == K_TRUE)
        ]
        if len(row):
            r = row.iloc[0]
            print(f"  Scen {scen_tag}: rmse_z_mean={r['rmse_z_mean']:.4f}"
                  f"  (std={r['rmse_z_std']:.4f})")


if __name__ == "__main__":
    main()
