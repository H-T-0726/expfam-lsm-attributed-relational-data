"""
Scenario C 追加20試行 (trial 10-29) スクリプト。

Phase 1 full実験 (trial 0-9) で C_23_6x_like / C_41_5x_like の試行間ばらつきが
大きかったため、trial 10-29 の追加20試行を実行して30試行版集計を作る。

既存ファイルは一切変更しない:
  - run_half_factor_minimal_check.py (importのみ)
  - expfam/results/half_factor_check/full/*.csv  (読み取りのみ)
  - その他すべての既存コード・CSV

出力先 (新規ディレクトリのみ):
  expfam/results/half_factor_check/scenario_c_extra/

実行例:
  cd expfam/src
  python run_half_factor_scenario_c_extra.py
"""

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

# 既存スクリプトから関数・定数を再利用 (挙動は変更しない)
from run_half_factor_minimal_check import (  # noqa: E402
    run_one,
    make_summary,
    CONDITIONS,
    N, D, K_TRUE, L, NITER,
    SEED_OFFSETS, EXP4_BASE_SEED,
)

# ─────────────────────────────────────────────────────────────────────
# Scenario C の3条件のみ抽出
# ─────────────────────────────────────────────────────────────────────

SCENARIO_C_CONDS = [
    c for c in CONDITIONS
    if c["condition_name"] in ("oracle_C", "C_23_6x_like", "C_41_5x_like")
]

EXTRA_TRIAL_START = 10
EXTRA_TRIAL_END = 30   # range(10, 30)

# 既存full結果の読み込み元
FULL_CSV = _ROOT / "expfam" / "results" / "half_factor_check" / "full" / "half_factor_minimal_summary.csv"

# 出力先
OUT_DIR = _ROOT / "expfam" / "results" / "half_factor_check" / "scenario_c_extra"


# ─────────────────────────────────────────────────────────────────────
# Extended aggregation: mean/std に加えて median/min/max も計算
# ─────────────────────────────────────────────────────────────────────

def make_summary_extended(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario", "condition_name", "true_x", "true_y", "est_x", "est_y", "note"]
    for keys, sub in df.groupby(group_cols, sort=False):
        rec = dict(zip(group_cols, keys))
        rec["n_trials"] = len(sub)
        rec["old_rmse_z_mean"] = sub["old_rmse_z"].mean()
        rec["old_rmse_z_std"] = sub["old_rmse_z"].std()
        rec["old_rmse_z_median"] = sub["old_rmse_z"].median()
        rec["old_rmse_z_min"] = sub["old_rmse_z"].min()
        rec["old_rmse_z_max"] = sub["old_rmse_z"].max()
        rec["fixed_rmse_z_mean"] = sub["fixed_rmse_z"].mean()
        rec["fixed_rmse_z_std"] = sub["fixed_rmse_z"].std()
        rec["fixed_rmse_z_median"] = sub["fixed_rmse_z"].median()
        rec["fixed_rmse_z_min"] = sub["fixed_rmse_z"].min()
        rec["fixed_rmse_z_max"] = sub["fixed_rmse_z"].max()
        rec["diff_fixed_old_rmse_z_mean"] = (sub["fixed_rmse_z"] - sub["old_rmse_z"]).mean()
        old_mean = sub["old_rmse_z"].mean()
        fixed_mean = sub["fixed_rmse_z"].mean()
        rec["ratio_fixed_old_rmse_z_mean"] = (
            fixed_mean / old_mean if old_mean and not np.isnan(old_mean) and old_mean != 0
            else float("nan")
        )
        rec["old_success_rate"] = sub["old_success"].mean()
        rec["fixed_success_rate"] = sub["fixed_success"].mean()
        rows.append(rec)

    out = pd.DataFrame(rows)
    cols = [c for c in group_cols if c != "note"] + [
        "n_trials",
        "old_rmse_z_mean", "old_rmse_z_std", "old_rmse_z_median",
        "old_rmse_z_min", "old_rmse_z_max",
        "fixed_rmse_z_mean", "fixed_rmse_z_std", "fixed_rmse_z_median",
        "fixed_rmse_z_min", "fixed_rmse_z_max",
        "diff_fixed_old_rmse_z_mean", "ratio_fixed_old_rmse_z_mean",
        "old_success_rate", "fixed_success_rate", "note",
    ]
    return out[cols]


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conditions = SCENARIO_C_CONDS
    trials = list(range(EXTRA_TRIAL_START, EXTRA_TRIAL_END))

    print(f"Scenario C extra trials: trial {EXTRA_TRIAL_START}-{EXTRA_TRIAL_END - 1}")
    print(f"Conditions: {[c['condition_name'] for c in conditions]}")
    print(f"Output dir: {OUT_DIR}")

    # ── 追加20試行を実行 ────────────────────────────────────────────────
    t0 = time.perf_counter()
    rows = []
    total = len(conditions) * len(trials)
    done = 0

    for cond in conditions:
        for trial in trials:
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

    df_extra = pd.DataFrame(rows)[summary_cols]

    # ── 追加20試行 per-trial CSV ────────────────────────────────────────
    out_extra = OUT_DIR / "half_factor_scenario_c_extra_summary.csv"
    df_extra.to_csv(out_extra, index=False)
    print(f"\nSaved: {out_extra}  ({len(df_extra)} rows)")

    # ── 追加20試行 集計 CSV ─────────────────────────────────────────────
    agg_extra = make_summary_extended(df_extra)
    out_extra_agg = OUT_DIR / "half_factor_scenario_c_extra_agg.csv"
    agg_extra.to_csv(out_extra_agg, index=False)
    print(f"Saved: {out_extra_agg}  ({len(agg_extra)} rows)")

    # ── 既存full trial 0-9 から Scenario C 抽出 ─────────────────────────
    print(f"\nReading existing full results: {FULL_CSV}")
    df_full = pd.read_csv(FULL_CSV)
    c_cond_names = [c["condition_name"] for c in SCENARIO_C_CONDS]
    df_full_c = df_full[df_full["condition_name"].isin(c_cond_names)].copy()
    print(f"  Scenario C rows from full: {len(df_full_c)}")

    # ── 30試行結合 ───────────────────────────────────────────────────────
    df_30 = pd.concat([df_full_c, df_extra], ignore_index=True)
    df_30 = df_30.sort_values(["condition_name", "trial"]).reset_index(drop=True)

    out_30 = OUT_DIR / "half_factor_scenario_c_30trial_summary.csv"
    df_30.to_csv(out_30, index=False)
    print(f"Saved: {out_30}  ({len(df_30)} rows)")

    # ── 30試行 集計 CSV ──────────────────────────────────────────────────
    agg_30 = make_summary_extended(df_30)
    out_30_agg = OUT_DIR / "half_factor_scenario_c_30trial_agg.csv"
    agg_30.to_csv(out_30_agg, index=False)
    print(f"Saved: {out_30_agg}  ({len(agg_30)} rows)")

    # ── oracle_C 基準の倍率 ──────────────────────────────────────────────
    def print_ratios(label: str, agg_df: pd.DataFrame):
        row_oc = agg_df[agg_df["condition_name"] == "oracle_C"].iloc[0]
        row_236 = agg_df[agg_df["condition_name"] == "C_23_6x_like"].iloc[0]
        row_415 = agg_df[agg_df["condition_name"] == "C_41_5x_like"].iloc[0]
        print(f"\n  {label}")
        print(f"    old  C_23_6x_like / oracle_C = "
              f"{row_236['old_rmse_z_mean'] / row_oc['old_rmse_z_mean']:.2f}x")
        print(f"    fixed C_23_6x_like / oracle_C = "
              f"{row_236['fixed_rmse_z_mean'] / row_oc['fixed_rmse_z_mean']:.2f}x")
        print(f"    old  C_41_5x_like / oracle_C = "
              f"{row_415['old_rmse_z_mean'] / row_oc['old_rmse_z_mean']:.2f}x")
        print(f"    fixed C_41_5x_like / oracle_C = "
              f"{row_415['fixed_rmse_z_mean'] / row_oc['fixed_rmse_z_mean']:.2f}x")

    print("\n=== oracle_C 基準の悪化倍率 ===")
    print_ratios("追加20試行のみ", agg_extra)
    print_ratios("30試行版 (full10 + extra20)", agg_30)

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
