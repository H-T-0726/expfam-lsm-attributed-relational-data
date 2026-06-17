"""
fixed版 Exp4 Scenario A/B official 実験スクリプト。

DualExpFamLSMFixed (Y-side 0.5 なし) を使って Scenario A/B の
分布族誤指定実験 (Exp4) を正式に再実行する。修論用の正式結果。

Scenario A: true_x=poisson,   true_y=bernoulli
Scenario B: true_x=gaussian,  true_y=poisson

条件:
  Grid 3x3 (9条件): est_x × est_y の全組み合わせ
  Ablation (2条件): Y-only (fix_x), X-only (fix_w)
  合計 11条件 × 10試行 = 110 fits/シナリオ

seedの規則は exp_scenario_lib.run_exp4 と同一:
  base_seed = EXP4_BASE_SEED + SEED_OFFSETS[scenario]
  A: base_seed = 6000 + 0    = 6000
  B: base_seed = 6000 + 1000 = 7000
  seed_data  = base_seed + trial * 100
  seed_model = seed_data + 50

既存ファイルは一切変更しない。
出力先: expfam/results/fixed_official/exp4/

実行例:
  cd expfam/src
  python run_fixed_official_exp4_scen_ab.py
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
from utils_expfam import calc_rmse, procrustes_rotation    # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Global settings
# ─────────────────────────────────────────────────────────────────────

OUT_DIR = _ROOT / "expfam" / "results" / "fixed_official" / "exp4"

N, D, K_TRUE = 150, 15, 3
L, NITER     = 5, 8
TRIALS       = list(range(10))   # 10試行 (0-9)

EXP4_BASE    = 6000
SEED_OFFSETS = {"A": 0, "B": 1000}

FAMILIES   = ["gaussian", "bernoulli", "poisson"]
FAM_SHORT  = {"gaussian": "Gauss", "bernoulli": "Bern", "poisson": "Pois"}
HIGH_RMSE_THRESHOLD = 0.7


# ─────────────────────────────────────────────────────────────────────
# Build condition list for one scenario
# ─────────────────────────────────────────────────────────────────────

def make_conditions(scenario_tag, true_x, true_y):
    oracle_key = (true_x, true_y)
    conds = []
    # Grid 3×3
    for est_x in FAMILIES:
        for est_y in FAMILIES:
            if (est_x, est_y) == oracle_key:
                name = f"oracle_{scenario_tag}"
            else:
                name = f"X{FAM_SHORT[est_x]}_Y{FAM_SHORT[est_y]}"
            conds.append({
                "condition_name": name,
                "est_x":   est_x,
                "est_y":   est_y,
                "fix_w":   False,
                "fix_x":   False,
            })
    # Ablation
    conds.append({
        "condition_name": "Y_only",
        "est_x": true_x, "est_y": true_y,
        "fix_w": False, "fix_x": True,
    })
    conds.append({
        "condition_name": "X_only",
        "est_x": true_x, "est_y": true_y,
        "fix_w": True,  "fix_x": False,
    })
    return conds


# ─────────────────────────────────────────────────────────────────────
# Fixed-version EM runner for Exp4
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed_exp4(X, Y, true_params, family_x, family_y, k,
                      L=5, num_iter=8, seed=42,
                      fix_w=False, fix_x=False):
    """
    MC-EM using DualExpFamLSMFixed for Exp4 mismatch.
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

        if fix_w:
            w = 0.0
            model.params["w"] = 0.0
        if fix_x:
            F = np.zeros((d, k))
            model.params["F"] = np.zeros((d, k))

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
            if not fix_x:
                F     = model.calc_F(X, Z_samples)
                sigma = model.calc_sigma(X, Z_samples, F)
            w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            if not fix_w:
                w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
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
# Run one scenario
# ─────────────────────────────────────────────────────────────────────

def run_scenario(scenario_tag, true_x, true_y, t0_global):
    base_seed  = EXP4_BASE + SEED_OFFSETS[scenario_tag]
    conditions = make_conditions(scenario_tag, true_x, true_y)
    oracle_name = f"oracle_{scenario_tag}"

    total = len(conditions) * len(TRIALS)
    print(f"\n{'='*60}")
    print(f"Scenario {scenario_tag}: X={true_x}, Y={true_y}")
    print(f"  {len(conditions)} conditions x {len(TRIALS)} trials = {total} fits")
    print(f"  base_seed={base_seed}")
    print(f"{'='*60}")

    done = 0
    rows = []

    for cond in conditions:
        cname = cond["condition_name"]
        est_x = cond["est_x"]
        est_y = cond["est_y"]
        fix_w = cond["fix_w"]
        fix_x = cond["fix_x"]

        for trial in TRIALS:
            done += 1
            seed_data  = base_seed + trial * 100
            seed_model = seed_data + 50

            try:
                data = generate_dual_data(
                    n=N, d=D, k=K_TRUE, seed=seed_data,
                    family_x=true_x, family_y=true_y,
                )
                res = run_em_fixed_exp4(
                    X=data["X"], Y=data["Y"], true_params=data,
                    family_x=est_x, family_y=est_y,
                    k=K_TRUE, L=L, num_iter=NITER, seed=seed_model,
                    fix_w=fix_w, fix_x=fix_x,
                )
                success = True
                err_msg = ""
                elapsed = (time.perf_counter() - t0_global) / 60
                print(
                    f"  [{done:3d}/{total}] {cname:20s} t={trial}"
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
                elapsed = (time.perf_counter() - t0_global) / 60
                print(
                    f"  [{done:3d}/{total}] {cname:20s} t={trial}"
                    f"  ERROR: {err_msg[:50]}  [{elapsed:.1f} min]"
                )

            rows.append({
                "scenario":       scenario_tag,
                "condition_name": cname,
                "trial":          trial,
                "seed_data":      seed_data,
                "seed_model":     seed_model,
                "true_x":         true_x,
                "true_y":         true_y,
                "est_x":          est_x,
                "est_y":          est_y,
                "fix_w":          fix_w,
                "fix_x":          fix_x,
                "n":              N,
                "d":              D,
                "k":              K_TRUE,
                "rmse_z":         res["rmse_Z"],
                "rmse_x":         res["rmse_X"],
                "rmse_y":         res["rmse_Y"],
                "q_final":        res["q_final"],
                "w_est":          res["w_est"],
                "success":        success,
                "error_message":  err_msg,
                "note": f"fixed_official exp4 scen_{scenario_tag.lower()} 10trials",
            })

    df = pd.DataFrame(rows)
    tag = scenario_tag.lower()

    # ── summary CSV ────────────────────────────────────────────────────
    out_summary = OUT_DIR / f"fixed_exp4_scen_{tag}_summary.csv"
    df.to_csv(out_summary, index=False)
    print(f"\nSaved: {out_summary}  ({len(df)} rows)")

    # ── agg CSV ────────────────────────────────────────────────────────
    agg_rows = []
    group_keys = ["scenario", "condition_name", "true_x", "true_y",
                  "est_x", "est_y", "fix_w", "fix_x"]
    for keys, sub in df.groupby(group_keys, sort=False):
        rec = dict(zip(group_keys, keys))
        rec["n_trials"]        = len(sub)
        rec["rmse_z_mean"]     = sub["rmse_z"].mean()
        rec["rmse_z_std"]      = sub["rmse_z"].std()
        rec["rmse_z_median"]   = sub["rmse_z"].median()
        rec["rmse_z_min"]      = sub["rmse_z"].min()
        rec["rmse_z_max"]      = sub["rmse_z"].max()
        rec["rmse_x_mean"]     = sub["rmse_x"].mean()
        rec["rmse_y_mean"]     = sub["rmse_y"].mean()
        rec["success_rate"]    = float(sub["success"].mean())
        high = (sub["rmse_z"] > HIGH_RMSE_THRESHOLD).sum()
        rec["high_rmse_count"] = int(high)
        rec["high_rmse_rate"]  = float(high / len(sub))
        rec["note"] = (
            f"fixed_official exp4 scen_{tag},"
            f" high_rmse_threshold={HIGH_RMSE_THRESHOLD}"
        )
        agg_rows.append(rec)

    agg_df = pd.DataFrame(agg_rows)
    out_agg = OUT_DIR / f"fixed_exp4_scen_{tag}_agg.csv"
    agg_df.to_csv(out_agg, index=False)
    print(f"Saved: {out_agg}  ({len(agg_df)} rows)")

    # ── ratios CSV ─────────────────────────────────────────────────────
    oracle_row = agg_df[agg_df["condition_name"] == oracle_name]
    if len(oracle_row) == 0:
        print(f"WARNING: {oracle_name} not found — ratios skipped")
        return agg_df

    oracle_mean   = float(oracle_row["rmse_z_mean"].iloc[0])
    oracle_median = float(oracle_row["rmse_z_median"].iloc[0])

    ratio_rows = []
    for _, r in agg_df.iterrows():
        ratio_rows.append({
            "scenario":              scenario_tag,
            "base_condition":        oracle_name,
            "condition_name":        r["condition_name"],
            "est_x":                 r["est_x"],
            "est_y":                 r["est_y"],
            "fix_w":                 r["fix_w"],
            "fix_x":                 r["fix_x"],
            "rmse_z_mean":           r["rmse_z_mean"],
            "oracle_rmse_z_mean":    oracle_mean,
            "ratio_vs_oracle":       (
                r["rmse_z_mean"] / oracle_mean
                if oracle_mean > 0 else float("nan")
            ),
            "rmse_z_median":         r["rmse_z_median"],
            "oracle_rmse_z_median":  oracle_median,
            "median_ratio_vs_oracle": (
                r["rmse_z_median"] / oracle_median
                if oracle_median > 0 else float("nan")
            ),
            "note": f"fixed_official exp4 scen_{tag}, ratio = rmse_z_mean / {oracle_name}_mean",
        })

    ratio_df = pd.DataFrame(ratio_rows)
    out_ratios = OUT_DIR / f"fixed_exp4_scen_{tag}_ratios.csv"
    ratio_df.to_csv(out_ratios, index=False)
    print(f"Saved: {out_ratios}  ({len(ratio_df)} rows)")

    # ── Quick summary print ────────────────────────────────────────────
    print(f"\n=== Scenario {scenario_tag} oracle-based ratios ===")
    print(f"  {oracle_name}: mean={oracle_mean:.5f}  median={oracle_median:.5f}")
    print()
    ratio_sorted = ratio_df.sort_values("ratio_vs_oracle")
    for _, r in ratio_sorted.iterrows():
        print(f"  {r['condition_name']:20s}"
              f"  est_x={r['est_x'][:4]:4s}  est_y={r['est_y'][:4]:4s}"
              f"  ratio={r['ratio_vs_oracle']:6.2f}x"
              f"  mean={r['rmse_z_mean']:.4f}")

    return agg_df


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    print("fixed Exp4 Scenario A/B official")
    print(f"Output: {OUT_DIR}")

    run_scenario("A", true_x="poisson",  true_y="bernoulli", t0_global=t0)
    run_scenario("B", true_x="gaussian", true_y="poisson",   t0_global=t0)

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\nAll done in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
