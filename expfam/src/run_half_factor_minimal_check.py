"""
Phase 1: old (0.5 あり, model_dual_expfam.py) vs fixed (0.5 なし,
model_dual_expfam_fixed.py) の最小比較実験。

KNOWN_ISSUES.md KI-001 (E-step Y側Term3の0.5係数問題) の検証用。
"0.5係数問題 fixed版検証 最小実験計画書" Phase 1 に対応。

このスクリプトは新規ファイルであり、以下は一切変更しない:
  - exp_scenario_lib.py
  - model_dual_expfam.py
  - model_dual_expfam_fixed.py
  - utils_expfam.py
  - run_comparison_quick.py
  - run_mismatch_fixed.py

出力先 (新規ディレクトリのみ):
  expfam/results/half_factor_check/dry_run/   (--mode dry_run)
  expfam/results/half_factor_check/full/      (--mode full)

実行例:
  cd expfam/src
  python run_half_factor_minimal_check.py --mode dry_run
  python run_half_factor_minimal_check.py --mode full
"""

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SRC = Path(__file__).parent
_ROOT = _SRC.parent.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from data_generator_expfam import generate_dual_data  # noqa: E402
from utils_expfam import run_em_dual, calc_rmse, procrustes_rotation  # noqa: E402
from model_dual_expfam_fixed import DualExpFamLSMFixed  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Settings (aligned with exp_scenario_lib.py / exp_run_scenario_{A,B,C}.py)
# ─────────────────────────────────────────────────────────────────────

N, D, K_TRUE = 150, 15, 3
L, NITER = 5, 8
N_TRIALS_FULL = 10

# Same seed_offset as exp_run_scenario_{A,B,C}.py -> run_scenario(seed_offset=...)
# so that, for a given scenario, the data generated here uses the SAME
# data seed series as exp_scenario_{A,B,C}_exp4_mismatch.csv.
SEED_OFFSETS = {"A": 0, "B": 1000, "C": 2000}
EXP4_BASE_SEED = 6000  # see exp_scenario_lib.run_exp4 default base_seed

FAM_LABEL = {"gaussian": "Gaussian", "bernoulli": "Bernoulli", "poisson": "Poisson"}

# ─────────────────────────────────────────────────────────────────────
# Phase 1 conditions
# ─────────────────────────────────────────────────────────────────────

CONDITIONS = [
    dict(
        scenario="A", condition_name="oracle_A",
        true_x="poisson", true_y="bernoulli",
        est_x="poisson", est_y="bernoulli",
        note="Scenario A oracle (correct family)",
    ),
    dict(
        scenario="B", condition_name="oracle_B",
        true_x="gaussian", true_y="poisson",
        est_x="gaussian", est_y="poisson",
        note="Scenario B oracle (correct family)",
    ),
    dict(
        scenario="C", condition_name="oracle_C",
        true_x="bernoulli", true_y="gaussian",
        est_x="bernoulli", est_y="gaussian",
        note="Scenario C oracle (correct family)",
    ),
    dict(
        scenario="C", condition_name="C_23_6x_like",
        true_x="bernoulli", true_y="gaussian",
        est_x="gaussian", est_y="bernoulli",
        note="fig1b gray-bar condition (old model, ~23.6x in exp_scenario_C_exp4_mismatch.csv)",
    ),
    dict(
        scenario="C", condition_name="C_41_5x_like",
        true_x="bernoulli", true_y="gaussian",
        est_x="gaussian", est_y="poisson",
        note="manuscript max-misspec condition (old model, ~41.5x, no bar in fig1b)",
    ),
]


# ─────────────────────────────────────────────────────────────────────
# Fixed-model EM runner with full Q_history
#
# Adapted from run_mismatch_fixed.py's run_em_fixed(). That file is NOT
# modified; this is a separate copy in a new file, with the only addition
# being that the full Q_history (not just the last value) is returned so
# that old_q_start/old_q_end/old_q_delta-style diagnostics are available
# for the fixed model too.
# ─────────────────────────────────────────────────────────────────────

def run_em_fixed_with_history(X, Y, true_params, family_x, family_y, k,
                               L=5, num_iter=8, seed=42,
                               fix_w=False, fix_x=False):
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

        # ── Informed init: Y side ────────────────────────────────────
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if family_y == "bernoulli":
            density = float(np.clip(np.mean(Y), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"] = 0.5
        elif family_y == "poisson":
            upper = Y[upper_mask]
            mean_cnt = float(upper[upper > 0].mean()) if np.any(upper > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"] = 0.1 / (2 ** retry)
        else:  # gaussian
            upper_vals = Y[upper_mask]
            model.params["w0"] = float(upper_vals.mean())
            model.params["w"] = 0.5
            model.sigma_y = float(max(upper_vals.std(), 0.01))

        # ── Informed init: X side ────────────────────────────────────
        if family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        Z = model.params["Z"].copy()
        F = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0 = model.params["w0"]
        w = model.params["w"]
        var_z = model.params["var_z"]

        if fix_w:
            w = 0.0
            model.params["w"] = 0.0
        if fix_x:
            F = np.zeros((d, k))
            model.params["F"] = np.zeros((d, k))

        Z_prev = Z.copy()
        nan_count = 0
        Q_history = []

        for iteration in range(1, num_iter + 1):
            # E-step
            Z_samples = np.zeros((n, k, L))
            for l in range(L):
                model.params.update(dict(Z=Z.copy(), F=F,
                                          sigma=sigma, w0=w0, w=w))
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha
                )
                Z_samples[:, :, l] = Z_new
                Z = Z_new.copy()

            if (np.any(np.isnan(Z_samples)) or
                    np.any(np.isinf(Z_samples))):
                nan_count += 1
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            # M-step
            if not fix_x:
                F = model.calc_F(X, Z_samples)
                sigma = model.calc_sigma(X, Z_samples, F)
            w0 = model.calc_w0(Y, Z_samples, w0, w, max_iter=50)
            if not fix_w:
                w = model.calc_w(Y, Z_samples, w0, w, max_iter=50)
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

            # Q (no factorial) -- same formula as run_mismatch_fixed.py
            Q_val = 0.0
            for l in range(L):
                Z_l = Z_samples[:, :, l]
                lnpZ = -(n * k / 2) * np.log(2 * np.pi * var_z) \
                    - (1 / (2 * var_z)) * np.sum(Z_l ** 2)
                lnpX = float(model.calc_log_likelihood_X(
                    X, Z_samples[:, :, l:l + 1], F))
                lnpY = float(model.calc_log_likelihood_Y(
                    Y, Z_samples[:, :, l:l + 1], w0, w))
                Q_val += lnpZ + lnpX + lnpY
            Q_history.append(Q_val / L)

        nan_occurred = nan_count > 0
        if not nan_occurred:
            break

    # ── Metrics ──────────────────────────────────────────────────────
    Z_est = Z_samples[:, :, -1]
    mu_x_est = model._mean_function_x(Z_est @ F.T)
    rmse_X = calc_rmse(X, mu_x_est)

    R, k_min = procrustes_rotation(Z_est, true_params["Z"])
    Z_rot = Z_est[:, :k_min] @ R
    rmse_Z = calc_rmse(true_params["Z"][:, :k_min], Z_rot)

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    eta_y_est = float(w0) + float(w) * (Z_est @ Z_est.T)
    mu_y_est = model._mean_function(eta_y_est)
    eta_y_true = float(true_params["w0"]) + float(true_params["w"]) * (
        true_params["Z"] @ true_params["Z"].T)
    mu_y_true = model._mean_function(eta_y_true)
    rmse_Y = calc_rmse(mu_y_est[upper_mask], mu_y_true[upper_mask])

    return {
        "rmse_Z": rmse_Z,
        "rmse_X": rmse_X,
        "rmse_Y": rmse_Y,
        "Q_history": Q_history,
        "Q_final": Q_history[-1] if Q_history else float("nan"),
        "w_est": float(w),
        "nan_occurred": nan_occurred,
        "nan_count": nan_count,
    }


# ─────────────────────────────────────────────────────────────────────
# One (condition, trial) -> one row
# ─────────────────────────────────────────────────────────────────────

def run_one(cond: dict, trial: int) -> dict:
    scenario = cond["scenario"]
    base_seed = EXP4_BASE_SEED + SEED_OFFSETS[scenario]
    seed_data = base_seed + trial * 100
    seed_model = seed_data + 50

    row = {
        "scenario": scenario,
        "condition_name": cond["condition_name"],
        "true_x": cond["true_x"], "true_y": cond["true_y"],
        "est_x": cond["est_x"], "est_y": cond["est_y"],
        "trial": trial,
        "seed_data": seed_data, "seed_model": seed_model,
        "note": cond["note"],
    }

    data = generate_dual_data(
        n=N, d=D, k=K_TRUE, seed=seed_data,
        family_x=cond["true_x"], family_y=cond["true_y"],
    )

    # ── old model (model_dual_expfam.py, 0.5 あり) ─────────────────────
    try:
        old = run_em_dual(
            X=data["X"], Y=data["Y"], true_params=data,
            family_x=cond["est_x"], family_y=cond["est_y"],
            k=K_TRUE, L=L, num_iter=NITER, seed=seed_model,
            compute_strict_Q=False,
        )
        old_q_hist = old["Q_history"]
        row.update({
            "old_rmse_z": old["rmse_Z"],
            "old_rmse_y": old["rmse_Y"],
            "old_rmse_x": old["rmse_X"],
            "old_q_final": old["Q_final"],
            "old_q_start": old_q_hist[0] if old_q_hist else float("nan"),
            "old_q_end": old_q_hist[-1] if old_q_hist else float("nan"),
            "old_q_delta": (old_q_hist[-1] - old_q_hist[0]) if len(old_q_hist) >= 2 else float("nan"),
            "old_w_est": old["w_est"],
            "old_success": not old["nan_occurred"],
            "old_error_message": "",
        })
    except Exception as exc:  # pragma: no cover - safety net for dry-run/full
        row.update({
            "old_rmse_z": float("nan"), "old_rmse_y": float("nan"), "old_rmse_x": float("nan"),
            "old_q_final": float("nan"), "old_q_start": float("nan"),
            "old_q_end": float("nan"), "old_q_delta": float("nan"),
            "old_w_est": float("nan"),
            "old_success": False,
            "old_error_message": f"{type(exc).__name__}: {exc}",
        })

    # ── fixed model (model_dual_expfam_fixed.py, 0.5 なし) ─────────────
    try:
        fixed = run_em_fixed_with_history(
            X=data["X"], Y=data["Y"], true_params=data,
            family_x=cond["est_x"], family_y=cond["est_y"],
            k=K_TRUE, L=L, num_iter=NITER, seed=seed_model,
        )
        fix_q_hist = fixed["Q_history"]
        row.update({
            "fixed_rmse_z": fixed["rmse_Z"],
            "fixed_rmse_y": fixed["rmse_Y"],
            "fixed_rmse_x": fixed["rmse_X"],
            "fixed_q_final": fixed["Q_final"],
            "fixed_q_start": fix_q_hist[0] if fix_q_hist else float("nan"),
            "fixed_q_end": fix_q_hist[-1] if fix_q_hist else float("nan"),
            "fixed_q_delta": (fix_q_hist[-1] - fix_q_hist[0]) if len(fix_q_hist) >= 2 else float("nan"),
            "fixed_w_est": fixed["w_est"],
            "fixed_success": not fixed["nan_occurred"],
            "fixed_error_message": "",
        })
    except Exception as exc:  # pragma: no cover
        row.update({
            "fixed_rmse_z": float("nan"), "fixed_rmse_y": float("nan"), "fixed_rmse_x": float("nan"),
            "fixed_q_final": float("nan"), "fixed_q_start": float("nan"),
            "fixed_q_end": float("nan"), "fixed_q_delta": float("nan"),
            "fixed_w_est": float("nan"),
            "fixed_success": False,
            "fixed_error_message": f"{type(exc).__name__}: {exc}",
        })

    # ── diff / ratio ───────────────────────────────────────────────────
    def _safe_diff(a, b):
        if a is None or b is None or np.isnan(a) or np.isnan(b):
            return float("nan")
        return a - b

    def _safe_ratio(a, b):
        if a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0:
            return float("nan")
        return a / b

    row["diff_fixed_old_rmse_z"] = _safe_diff(row["fixed_rmse_z"], row["old_rmse_z"])
    row["ratio_fixed_old_rmse_z"] = _safe_ratio(row["fixed_rmse_z"], row["old_rmse_z"])
    row["diff_fixed_old_rmse_y"] = _safe_diff(row["fixed_rmse_y"], row["old_rmse_y"])
    row["diff_fixed_old_rmse_x"] = _safe_diff(row["fixed_rmse_x"], row["old_rmse_x"])

    return row


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario", "condition_name", "true_x", "true_y", "est_x", "est_y", "note"]
    for keys, sub in df.groupby(group_cols, sort=False):
        rec = dict(zip(group_cols, keys))
        rec["n_trials"] = len(sub)
        rec["old_rmse_z_mean"] = sub["old_rmse_z"].mean()
        rec["old_rmse_z_std"] = sub["old_rmse_z"].std()
        rec["fixed_rmse_z_mean"] = sub["fixed_rmse_z"].mean()
        rec["fixed_rmse_z_std"] = sub["fixed_rmse_z"].std()
        rec["diff_fixed_old_rmse_z_mean"] = (sub["fixed_rmse_z"] - sub["old_rmse_z"]).mean()
        old_mean = sub["old_rmse_z"].mean()
        fixed_mean = sub["fixed_rmse_z"].mean()
        rec["ratio_fixed_old_rmse_z_mean"] = (
            fixed_mean / old_mean if old_mean and not np.isnan(old_mean) and old_mean != 0 else float("nan")
        )
        rec["old_success_rate"] = sub["old_success"].mean()
        rec["fixed_success_rate"] = sub["fixed_success"].mean()
        rows.append(rec)
    # restore note column order at the end
    out = pd.DataFrame(rows)
    cols = [c for c in group_cols if c != "note"] + [
        "n_trials", "old_rmse_z_mean", "old_rmse_z_std",
        "fixed_rmse_z_mean", "fixed_rmse_z_std",
        "diff_fixed_old_rmse_z_mean", "ratio_fixed_old_rmse_z_mean",
        "old_success_rate", "fixed_success_rate", "note",
    ]
    return out[cols]


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["dry_run", "full"], default="dry_run",
        help="dry_run: 1 condition x 1 trial (Scenario C oracle). "
             "full: 5 conditions x N_TRIALS_FULL trials.",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "results" / "half_factor_check" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dry_run":
        conditions = [c for c in CONDITIONS if c["condition_name"] == "oracle_C"]
        n_trials = 1
    else:
        conditions = CONDITIONS
        n_trials = N_TRIALS_FULL

    print(f"Mode: {args.mode}")
    print(f"Conditions: {[c['condition_name'] for c in conditions]}")
    print(f"Trials per condition: {n_trials}")
    print(f"Output dir: {out_dir}")

    t0 = time.perf_counter()
    rows = []
    total = len(conditions) * n_trials
    done = 0
    for cond in conditions:
        for trial in range(n_trials):
            done += 1
            try:
                row = run_one(cond, trial)
            except Exception:
                traceback.print_exc()
                raise
            rows.append(row)
            elapsed = (time.perf_counter() - t0) / 60
            print(
                f"  [{done:3d}/{total}] {cond['condition_name']:14s} t={trial} "
                f"old_rmse_z={row['old_rmse_z']:.4f} "
                f"fixed_rmse_z={row['fixed_rmse_z']:.4f} "
                f"ratio={row['ratio_fixed_old_rmse_z']:.3f} "
                f"[{elapsed:.1f} min]"
            )

    df = pd.DataFrame(rows)

    summary_cols = [
        "scenario", "condition_name", "true_x", "true_y", "est_x", "est_y",
        "trial", "seed_data", "seed_model",
        "old_rmse_z", "fixed_rmse_z", "diff_fixed_old_rmse_z", "ratio_fixed_old_rmse_z",
        "old_rmse_y", "fixed_rmse_y", "diff_fixed_old_rmse_y",
        "old_rmse_x", "fixed_rmse_x", "diff_fixed_old_rmse_x",
        "old_q_final", "fixed_q_final",
        "old_q_start", "old_q_end", "old_q_delta",
        "fixed_q_start", "fixed_q_end", "fixed_q_delta",
        "old_w_est", "fixed_w_est",
        "old_success", "fixed_success",
        "old_error_message", "fixed_error_message",
        "note",
    ]
    df = df[summary_cols]

    out_csv = out_dir / "half_factor_minimal_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df)} rows)")

    agg = make_summary(df)
    out_agg = out_dir / "half_factor_minimal_summary_agg.csv"
    agg.to_csv(out_agg, index=False)
    print(f"Saved: {out_agg}  ({len(agg)} rows)")

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
