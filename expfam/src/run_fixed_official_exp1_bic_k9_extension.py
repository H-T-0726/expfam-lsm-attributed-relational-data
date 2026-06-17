"""
fixed版 Exp1 BIC K=1〜9 拡張実験スクリプト。

k=7,8,9 のみを追加実行し、既存の k=1〜6 結果と結合して
K=1〜9 の正式 BIC 結果を作成する。

既存の k=1〜6 CSV は再実行せず、読み込んで結合するだけ。
  expfam/results/fixed_official/exp1/fixed_exp1_bic_full_summary.csv

seed規則は run_fixed_official_exp1_bic_full.py と同一:
  base_seed = 4000 + SEED_OFFSETS[scenario]  (A=0, B=1000, C=2000)
  seed_data  = base_seed + trial * 100
  seed_model = seed_data + 50

合計新規実行: 3 × 3 × 10 = 90 fits

出力先: expfam/results/fixed_official/exp1_k9/

実行例:
  cd expfam/src
  python run_fixed_official_exp1_bic_k9_extension.py
"""

import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Settings (must match run_fixed_official_exp1_bic_full.py)
# ─────────────────────────────────────────────────────────────────────

EXP1_DIR = _ROOT / "expfam" / "results" / "fixed_official" / "exp1"
OUT_DIR  = _ROOT / "expfam" / "results" / "fixed_official" / "exp1_k9"
FIG_DIR  = _ROOT / "expfam" / "figures" / "fixed_official"

N, D, K_TRUE  = 150, 15, 3
L, NITER      = 5, 8
K_ADDITIONAL  = [7, 8, 9]    # only these are newly run
K_ALL         = list(range(1, 10))  # 1-9 for combined output

TRIALS        = list(range(10))

SEED_OFFSETS  = {"A": 0, "B": 1000, "C": 2000}
EXP1_BASE     = 4000

SCENARIOS = [
    ("A", "poisson",   "bernoulli"),
    ("B", "gaussian",  "poisson"),
    ("C", "bernoulli", "gaussian"),
]

SCEN_COLORS  = {"A": "#2196F3", "B": "#FF5722", "C": "#4CAF50"}
SCEN_MARKERS = {"A": "o", "B": "s", "C": "^"}
SCEN_LABELS  = {
    "A": "Scenario A (X=Poisson, Y=Bernoulli)",
    "B": "Scenario B (X=Gaussian, Y=Poisson)",
    "C": "Scenario C (X=Bernoulli, Y=Gaussian)",
}


# ─────────────────────────────────────────────────────────────────────
# Fixed-version EM runner with Q_strict (identical to bic_full script)
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed_bic(X, Y, true_params, family_x, family_y, k,
                     L=5, num_iter=8, seed=42):
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
        else:
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"]  = 0.5
            model.sigma_y = float(max(upper_vals.std(), 0.01))

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

        for _ in range(1, num_iter + 1):
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
                Z_new = model.calc_eta_newton(X, Y, rng=rng, max_iter=10,
                                              alpha=newton_alpha)
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_count += 1
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            F     = model.calc_F(X, Z_samples)
            sigma = model.calc_sigma(X, Z_samples, F)
            w0    = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            w     = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    Q_strict = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, var_z, w0, w, model)

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
        "nan_occurred": nan_occurred,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: run k=7,8,9 ──────────────────────────────────────────
    total = len(SCENARIOS) * len(K_ADDITIONAL) * len(TRIALS)
    print(f"fixed Exp1 BIC k=7〜9 extension: "
          f"{len(SCENARIOS)} scenarios x {len(K_ADDITIONAL)} k x "
          f"{len(TRIALS)} trials = {total} fits")
    print(f"Output: {OUT_DIR}")
    print()

    t0   = time.perf_counter()
    done = 0
    rows_new = []

    for scen_tag, fam_x, fam_y in SCENARIOS:
        base_seed = EXP1_BASE + SEED_OFFSETS[scen_tag]
        print(f"Scenario {scen_tag}: X={fam_x}, Y={fam_y}  base_seed={base_seed}")

        for k_est in K_ADDITIONAL:
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
                    success = True
                    err_msg = ""
                    elapsed = (time.perf_counter() - t0) / 60
                    print(
                        f"  [{done:3d}/{total}] Scen={scen_tag} k={k_est} t={trial}"
                        f"  rmse_Z={res['rmse_Z']:.4f}"
                        f"  BIC={bic:.1f}"
                        f"  [{elapsed:.1f} min]"
                    )

                except Exception as e:
                    traceback.print_exc()
                    bic        = float("nan")
                    num_params = -1
                    res = {
                        "Q_strict": float("nan"),
                        "rmse_Z": float("nan"),
                        "rmse_X": float("nan"),
                        "rmse_Y": float("nan"),
                        "nan_occurred": True,
                    }
                    success = False
                    err_msg = str(e)
                    elapsed = (time.perf_counter() - t0) / 60
                    print(
                        f"  [{done:3d}/{total}] Scen={scen_tag} k={k_est} t={trial}"
                        f"  ERROR: {err_msg[:60]}  [{elapsed:.1f} min]"
                    )

                rows_new.append({
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
                    "note":          "fixed_official k7to9 extension",
                })

    df_new = pd.DataFrame(rows_new)
    elapsed_run = (time.perf_counter() - t0) / 60
    print(f"\nk=7〜9 run done in {elapsed_run:.1f} min")

    # Save k=7-9 only
    out_sum79 = OUT_DIR / "fixed_exp1_bic_k7to9_summary.csv"
    df_new.to_csv(out_sum79, index=False)
    print(f"Saved: {out_sum79}  ({len(df_new)} rows)")

    agg79_rows = []
    for (sc, k_est), sub in df_new.groupby(["scenario", "k_est"], sort=True):
        agg79_rows.append({
            "scenario":      sc,
            "family_x":      sub["family_x"].iloc[0],
            "family_y":      sub["family_y"].iloc[0],
            "k_est":         k_est,
            "n_trials":      len(sub),
            "bic_mean":      sub["bic"].mean(),
            "bic_std":       sub["bic"].std(),
            "q_strict_mean": sub["q_strict"].mean(),
            "rmse_z_mean":   sub["rmse_z"].mean(),
            "rmse_z_std":    sub["rmse_z"].std(),
            "success_rate":  float(sub["success"].mean()),
            "note":          "fixed_official k7to9 extension",
        })
    agg79_df = pd.DataFrame(agg79_rows)
    out_agg79 = OUT_DIR / "fixed_exp1_bic_k7to9_agg.csv"
    agg79_df.to_csv(out_agg79, index=False)
    print(f"Saved: {out_agg79}  ({len(agg79_df)} rows)")

    # ── Step 2: merge with existing k=1〜6 ────────────────────────────
    existing_path = EXP1_DIR / "fixed_exp1_bic_full_summary.csv"
    print(f"\nLoading existing k=1〜6: {existing_path}")
    df_old = pd.read_csv(existing_path)
    print(f"  -> {len(df_old)} rows (k={sorted(df_old['k_est'].unique())})")

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.sort_values(["scenario", "k_est", "trial"]).reset_index(drop=True)

    # ── k=1〜9 summary ─────────────────────────────────────────────────
    out_sum_all = OUT_DIR / "fixed_exp1_bic_k1to9_summary.csv"
    df_all.to_csv(out_sum_all, index=False)
    print(f"Saved: {out_sum_all}  ({len(df_all)} rows)")

    # ── k=1〜9 agg ─────────────────────────────────────────────────────
    agg_all_rows = []
    for (sc, k_est), sub in df_all.groupby(["scenario", "k_est"], sort=True):
        agg_all_rows.append({
            "scenario":      sc,
            "family_x":      sub["family_x"].iloc[0],
            "family_y":      sub["family_y"].iloc[0],
            "k_est":         k_est,
            "n_trials":      len(sub),
            "bic_mean":      sub["bic"].mean(),
            "bic_std":       sub["bic"].std(),
            "q_strict_mean": sub["q_strict"].mean(),
            "rmse_z_mean":   sub["rmse_z"].mean(),
            "rmse_z_std":    sub["rmse_z"].std(),
            "success_rate":  float(sub["success"].mean()),
            "note":          "fixed_official k1to9 combined",
        })
    agg_all_df = pd.DataFrame(agg_all_rows)
    out_agg_all = OUT_DIR / "fixed_exp1_bic_k1to9_agg.csv"
    agg_all_df.to_csv(out_agg_all, index=False)
    print(f"Saved: {out_agg_all}  ({len(agg_all_df)} rows)")

    # ── k=1〜9 bestk_mean ─────────────────────────────────────────────
    bestk_mean_rows = []
    for sc, sub_agg in agg_all_df.groupby("scenario", sort=True):
        fam_x = sub_agg["family_x"].iloc[0]
        fam_y = sub_agg["family_y"].iloc[0]
        best_k = int(sub_agg.loc[sub_agg["bic_mean"].idxmin(), "k_est"])
        bic_by_k = {int(r["k_est"]): r["bic_mean"] for _, r in sub_agg.iterrows()}
        row = {
            "scenario":           sc,
            "family_x":           fam_x,
            "family_y":           fam_y,
            "best_k_by_bic_mean": best_k,
            "k_true":             K_TRUE,
            "selected_correctly": best_k == K_TRUE,
        }
        for k in K_ALL:
            row[f"bic_at_k{k}"] = bic_by_k.get(k, float("nan"))
        row["note"] = "fixed_official k1to9 combined, BIC mean over 10 trials"
        bestk_mean_rows.append(row)

    bestk_mean_df = pd.DataFrame(bestk_mean_rows)
    out_bestk_mean = OUT_DIR / "fixed_exp1_bic_k1to9_bestk_mean.csv"
    bestk_mean_df.to_csv(out_bestk_mean, index=False)
    print(f"Saved: {out_bestk_mean}  ({len(bestk_mean_df)} rows)")

    # ── k=1〜9 bestk_by_trial ─────────────────────────────────────────
    trial_rows = []
    for (sc, trial), sub in df_all.groupby(["scenario", "trial"], sort=True):
        if sub["bic"].isna().all():
            best_k = -1
            best_bic = float("nan")
        else:
            idx = sub["bic"].idxmin()
            best_k   = int(sub.loc[idx, "k_est"])
            best_bic = float(sub.loc[idx, "bic"])
        trial_rows.append({
            "scenario":            sc,
            "trial":               trial,
            "best_k_by_trial_bic": best_k,
            "k_true":              K_TRUE,
            "selected_correctly":  best_k == K_TRUE,
            "best_bic":            best_bic,
            "note": "fixed_official k1to9, best k per trial",
        })
    trial_df = pd.DataFrame(trial_rows)
    out_trial = OUT_DIR / "fixed_exp1_bic_k1to9_bestk_by_trial.csv"
    trial_df.to_csv(out_trial, index=False)
    print(f"Saved: {out_trial}  ({len(trial_df)} rows)")

    # ── k=1〜9 bestk_frequency ────────────────────────────────────────
    freq_rows = []
    for sc, sub in trial_df.groupby("scenario", sort=True):
        total_t = len(sub)
        counts  = sub["best_k_by_trial_bic"].value_counts().to_dict()
        for k in K_ALL:
            freq_rows.append({
                "scenario":      sc,
                "k":             k,
                "selected_count": int(counts.get(k, 0)),
                "total_trials":   total_t,
                "selected_rate":  counts.get(k, 0) / total_t,
                "note": "fixed_official k1to9, trial-level best k frequency",
            })
    freq_df = pd.DataFrame(freq_rows)
    out_freq = OUT_DIR / "fixed_exp1_bic_k1to9_bestk_frequency.csv"
    freq_df.to_csv(out_freq, index=False)
    print(f"Saved: {out_freq}  ({len(freq_df)} rows)")

    # ── Figure: BIC k=1〜9 curve ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for sc in ["A", "B", "C"]:
        sub = agg_all_df[agg_all_df["scenario"] == sc].sort_values("k_est")
        ax.errorbar(
            sub["k_est"], sub["bic_mean"],
            yerr=sub["bic_std"],
            label=SCEN_LABELS[sc],
            color=SCEN_COLORS[sc],
            marker=SCEN_MARKERS[sc],
            linewidth=1.8, markersize=6,
            capsize=4, capthick=1.2, elinewidth=1.0,
        )
    ax.axvline(x=K_TRUE, color="black", linestyle="--",
               linewidth=1.0, alpha=0.7, label=f"$k^* = {K_TRUE}$ (true)")
    ax.set_xlabel("Estimated number of latent dimensions $k$")
    ax.set_ylabel("BIC")
    ax.set_title("BIC vs. $k$ (K=1–9)\n"
                 "(fixed implementation, 10 trials, $n=150$, $d=15$, $k^*=3$)")
    ax.set_xticks(K_ALL)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_fig = FIG_DIR / f"fig_fixed_exp1_bic_k1to9.{ext}"
        if out_fig.exists():
            print(f"  SKIP (exists): {out_fig}")
        else:
            fig.savefig(out_fig, dpi=200, bbox_inches="tight")
            print(f"  Saved: {out_fig}")
    plt.close(fig)

    # ── Final summary ─────────────────────────────────────────────────
    elapsed_total = (time.perf_counter() - t0) / 60
    print(f"\nAll done in {elapsed_total:.1f} min")

    print("\n=== BIC best k (mean, k=1〜9) ===")
    for _, r in bestk_mean_df.iterrows():
        status = "OK" if r["selected_correctly"] else "NG"
        print(f"  Scen {r['scenario']}: best_k={r['best_k_by_bic_mean']}"
              f"  (k_true={r['k_true']}) [{status}]"
              f"  k3={r['bic_at_k3']:.1f}  k7={r['bic_at_k7']:.1f}"
              f"  k8={r['bic_at_k8']:.1f}  k9={r['bic_at_k9']:.1f}")

    print("\n=== Trial-level best k frequency (k=3) ===")
    for sc in ["A", "B", "C"]:
        k3_rows = freq_df[(freq_df["scenario"] == sc) & (freq_df["k"] == 3)]
        if len(k3_rows):
            r = k3_rows.iloc[0]
            print(f"  Scen {sc}: k=3 selected {r['selected_count']}/{r['total_trials']}"
                  f" trials ({r['selected_rate']:.0%})")


if __name__ == "__main__":
    main()
