"""
fixed版 Exp3 d-sweep official 実験スクリプト。

DualExpFamLSMFixed (Y-side 0.5 なし) を使って、属性次元 d を変化させたときの
RMSE(Z) の推移を正式に確認する。

Scenario A: true_x=poisson,   true_y=bernoulli
Scenario B: true_x=gaussian,  true_y=poisson
Scenario C: true_x=bernoulli, true_y=gaussian

d_values = [5, 10, 15, 20, 25, 30]  (exp_scenario_lib.D_LIST と同一)
n = 150  (exp_scenario_lib.N と同一)
trials   = 0〜9 (exp_scenario_lib.N_TRIALS と同一)
合計: 3 × 6 × 10 = 180 fits

seedの規則は exp_scenario_lib.run_exp23 と同一:
  dseed = base_seed + trial * 1000 + n + d
  mseed = dseed + 500
  base_seed: A=5000, B=6000, C=7000  (5000 + seed_offset[A/B/C])

既存ファイルは一切変更しない。
出力先: expfam/results/fixed_official/exp3/

実行例:
  cd expfam/src
  python run_fixed_official_exp3_d_sweep.py
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
from utils_expfam import calc_rmse, procrustes_rotation    # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Settings (must match exp_scenario_lib.py)
# ─────────────────────────────────────────────────────────────────────

OUT_DIR   = _ROOT / "expfam" / "results" / "fixed_official" / "exp3"
FIG_DIR   = _ROOT / "expfam" / "figures" / "fixed_official"

N_FIXED  = 150
K_TRUE   = 3
L, NITER = 5, 8
D_LIST   = [5, 10, 15, 20, 25, 30]
TRIALS   = list(range(10))

# base_seed = 5000 + seed_offset (same as exp_scenario_lib run_scenario)
EXP23_BASE   = 5000
SEED_OFFSETS = {"A": 0, "B": 1000, "C": 2000}

SCENARIOS = [
    {"tag": "A", "true_x": "poisson",   "true_y": "bernoulli"},
    {"tag": "B", "true_x": "gaussian",  "true_y": "poisson"},
    {"tag": "C", "true_x": "bernoulli", "true_y": "gaussian"},
]

SCEN_COLORS  = {"A": "#2196F3", "B": "#FF5722", "C": "#4CAF50"}
SCEN_MARKERS = {"A": "o", "B": "s", "C": "^"}
SCEN_LABELS  = {
    "A": "Scenario A (X=Poisson, Y=Bernoulli)",
    "B": "Scenario B (X=Gaussian, Y=Poisson)",
    "C": "Scenario C (X=Bernoulli, Y=Gaussian)",
}


# ─────────────────────────────────────────────────────────────────────
# Fixed-version EM runner for Exp3 d-sweep
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed_d_sweep(X, Y, true_params, family_x, family_y, k,
                         L=5, num_iter=8, seed=42):
    """
    MC-EM using DualExpFamLSMFixed for Exp3 d-sweep.
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
        Q_history = []

        for _ in range(1, num_iter + 1):
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

            # Q monitoring
            Q_val = 0.0
            for l in range(L):
                Z_l  = Z_samples[:, :, l]
                lnpZ = (-(n * k / 2) * np.log(2 * np.pi * var_z)
                        - (1 / (2 * var_z)) * np.sum(Z_l ** 2))
                lnpX = float(model.calc_log_likelihood_X(
                    X, Z_samples[:, :, l:l+1], F))
                lnpY = float(model.calc_log_likelihood_Y(
                    Y, Z_samples[:, :, l:l+1], w0, w))
                Q_val += lnpZ + lnpX + lnpY
            Q_history.append(Q_val / L)

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

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
        "rmse_Z":       rmse_Z,
        "rmse_X":       rmse_X,
        "rmse_Y":       rmse_Y,
        "q_final":      Q_history[-1] if Q_history else float("nan"),
        "w_est":        float(w),
        "nan_occurred": nan_occurred,
        "nan_count":    nan_count,
    }


# ─────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────

def make_fig_d_sweep(agg_df):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for sc in ["A", "B", "C"]:
        sub = agg_df[agg_df["scenario"] == sc].sort_values("d")
        ax.errorbar(
            sub["d"], sub["rmse_z_mean"],
            yerr=sub["rmse_z_std"],
            label=SCEN_LABELS[sc],
            color=SCEN_COLORS[sc],
            marker=SCEN_MARKERS[sc],
            linewidth=1.8, markersize=6,
            capsize=4, capthick=1.2,
            elinewidth=1.0,
        )

    ax.set_xlabel("Attribute dimension $d$")
    ax.set_ylabel("RMSE($\\mathbf{Z}$)")
    ax.set_title("d-sweep: RMSE(Z) vs. Attribute Dimension\n"
                 "(fixed implementation, 10 trials, $n=150$, $k^*=3$)")
    ax.set_xticks(D_LIST)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"fig_fixed_exp3_d_sweep.{ext}"
        if path.exists():
            print(f"  SKIP (exists): {path}")
        else:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_all = len(SCENARIOS) * len(D_LIST) * len(TRIALS)
    print(f"fixed Exp3 d-sweep official: "
          f"{len(SCENARIOS)} scenarios x {len(D_LIST)} d-values x "
          f"{len(TRIALS)} trials = {total_all} fits")
    print(f"D_LIST = {D_LIST}  N_FIXED = {N_FIXED}")
    print(f"Output: {OUT_DIR}")
    print()

    t0   = time.perf_counter()
    done = 0
    rows = []

    for sc in SCENARIOS:
        tag    = sc["tag"]
        true_x = sc["true_x"]
        true_y = sc["true_y"]
        base_seed = EXP23_BASE + SEED_OFFSETS[tag]

        total_sc = len(D_LIST) * len(TRIALS)
        done_sc  = 0
        print(f"{'='*60}")
        print(f"Scenario {tag}: X={true_x}, Y={true_y}  base_seed={base_seed}")
        print(f"{'='*60}")

        for d_val in D_LIST:
            for trial in TRIALS:
                done    += 1
                done_sc += 1
                # Seed formula matching exp_scenario_lib.run_exp23
                # one_run(N, d_val, trial): dseed = base_seed + trial*1000 + n + d
                dseed = base_seed + trial * 1000 + N_FIXED + d_val
                mseed = dseed + 500

                try:
                    data = generate_dual_data(
                        n=N_FIXED, d=d_val, k=K_TRUE, seed=dseed,
                        family_x=true_x, family_y=true_y,
                    )
                    res = run_em_fixed_d_sweep(
                        X=data["X"], Y=data["Y"], true_params=data,
                        family_x=true_x, family_y=true_y,
                        k=K_TRUE, L=L, num_iter=NITER, seed=mseed,
                    )
                    success = True
                    err_msg = ""
                    elapsed = (time.perf_counter() - t0) / 60
                    print(
                        f"  [{done_sc:3d}/{total_sc}] Scen{tag} d={d_val:2d} t={trial}"
                        f"  rmse_Z={res['rmse_Z']:.4f}"
                        f"  q={res['q_final']:.1f}"
                        f"  [{elapsed:.1f} min]"
                    )
                except Exception as e:
                    traceback.print_exc()
                    res = {
                        "rmse_Z": float("nan"), "rmse_X": float("nan"),
                        "rmse_Y": float("nan"), "q_final": float("nan"),
                        "w_est":  float("nan"), "nan_occurred": True,
                        "nan_count": -1,
                    }
                    success = False
                    err_msg = str(e)
                    elapsed = (time.perf_counter() - t0) / 60
                    print(
                        f"  [{done_sc:3d}/{total_sc}] Scen{tag} d={d_val:2d} t={trial}"
                        f"  ERROR: {err_msg[:50]}  [{elapsed:.1f} min]"
                    )

                rows.append({
                    "scenario":      tag,
                    "trial":         trial,
                    "seed_data":     dseed,
                    "seed_model":    mseed,
                    "n":             N_FIXED,
                    "d":             d_val,
                    "k":             K_TRUE,
                    "family_x":      true_x,
                    "family_y":      true_y,
                    "rmse_z":        res["rmse_Z"],
                    "rmse_x":        res["rmse_X"],
                    "rmse_y":        res["rmse_Y"],
                    "q_final":       res["q_final"],
                    "w_est":         res["w_est"],
                    "success":       success,
                    "error_message": err_msg,
                    "note": f"fixed_official exp3 d_sweep scen_{tag.lower()}",
                })

    df = pd.DataFrame(rows)

    # ── summary CSV ────────────────────────────────────────────────────
    out_summary = OUT_DIR / "fixed_exp3_d_sweep_summary.csv"
    if out_summary.exists():
        print(f"\nSKIP (exists): {out_summary}")
    else:
        df.to_csv(out_summary, index=False)
        print(f"\nSaved: {out_summary}  ({len(df)} rows)")

    # ── agg CSV: scenario × d ──────────────────────────────────────────
    agg_rows = []
    for (sc_tag, d_val), sub in df.groupby(["scenario", "d"], sort=True):
        agg_rows.append({
            "scenario":      sc_tag,
            "d":             d_val,
            "n_trials":      len(sub),
            "rmse_z_mean":   sub["rmse_z"].mean(),
            "rmse_z_std":    sub["rmse_z"].std(),
            "rmse_z_median": sub["rmse_z"].median(),
            "rmse_z_min":    sub["rmse_z"].min(),
            "rmse_z_max":    sub["rmse_z"].max(),
            "rmse_x_mean":   sub["rmse_x"].mean(),
            "rmse_y_mean":   sub["rmse_y"].mean(),
            "success_rate":  float(sub["success"].mean()),
            "note": "fixed_official exp3 d_sweep",
        })

    agg_df = pd.DataFrame(agg_rows)
    out_agg = OUT_DIR / "fixed_exp3_d_sweep_agg.csv"
    if out_agg.exists():
        print(f"SKIP (exists): {out_agg}")
    else:
        agg_df.to_csv(out_agg, index=False)
        print(f"Saved: {out_agg}  ({len(agg_df)} rows)")

    # ── trend CSV ─────────────────────────────────────────────────────
    d_min = min(D_LIST)
    d_max = max(D_LIST)
    trend_rows = []
    for sc_tag, grp in agg_df.groupby("scenario", sort=True):
        r_dmin = float(grp.loc[grp["d"] == d_min, "rmse_z_mean"].values[0])
        r_dmax = float(grp.loc[grp["d"] == d_max, "rmse_z_mean"].values[0])
        abs_change = r_dmax - r_dmin
        rel_change = abs_change / r_dmin if r_dmin > 0 else float("nan")
        if abs_change < -1e-6:
            direction = "decreasing"
        elif abs_change > 1e-6:
            direction = "increasing"
        else:
            direction = "flat"
        trend_rows.append({
            "scenario":           sc_tag,
            "d_min":              d_min,
            "d_max":              d_max,
            "rmse_z_at_d_min":    r_dmin,
            "rmse_z_at_d_max":    r_dmax,
            "absolute_change":    abs_change,
            "relative_change_rate": rel_change,
            "trend_direction":    direction,
            "note": (
                f"fixed_official exp3 d_sweep, "
                f"relative = (rmse_d{d_max} - rmse_d{d_min}) / rmse_d{d_min}"
            ),
        })

    trend_df = pd.DataFrame(trend_rows)
    out_trend = OUT_DIR / "fixed_exp3_d_sweep_trend.csv"
    if out_trend.exists():
        print(f"SKIP (exists): {out_trend}")
    else:
        trend_df.to_csv(out_trend, index=False)
        print(f"Saved: {out_trend}  ({len(trend_df)} rows)")

    # ── Figure ─────────────────────────────────────────────────────────
    make_fig_d_sweep(agg_df)

    elapsed_total = (time.perf_counter() - t0) / 60
    print(f"\nDone in {elapsed_total:.1f} min")

    # ── Final summary ──────────────────────────────────────────────────
    print("\n=== d-sweep summary (rmse_z_mean) ===")
    pivot = agg_df.pivot(index="scenario", columns="d", values="rmse_z_mean")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== trend (d5 → d30) ===")
    for _, r in trend_df.iterrows():
        print(f"  Scen {r['scenario']}: "
              f"d={r['d_min']}→{r['d_max']}  "
              f"{r['rmse_z_at_d_min']:.4f}→{r['rmse_z_at_d_max']:.4f}  "
              f"rel_change={r['relative_change_rate']:+.1%}  "
              f"[{r['trend_direction']}]")

    # Success rate check
    total_fits = len(df)
    success_fits = int(df["success"].sum())
    print(f"\n=== Success rate: {success_fits}/{total_fits} fits ===")
    nan_fits = df["rmse_z"].isna().sum()
    if nan_fits > 0:
        print(f"  WARNING: {nan_fits} NaN rmse_z values")
    high_rmse = df[df["rmse_z"] > 1.0]
    if len(high_rmse) > 0:
        print(f"  WARNING: {len(high_rmse)} trials with rmse_z > 1.0")
        for _, row in high_rmse.iterrows():
            print(f"    Scen={row['scenario']} d={row['d']} trial={row['trial']}"
                  f" rmse_z={row['rmse_z']:.4f}")


if __name__ == "__main__":
    main()
