"""
既存 ablation 結果の棚卸し（Phase 5 前半、read-only 入力）。

共有 Z 仮定の検証に使える既存結果を 1 つの tidy CSV に集約する:
  1. 人工データ（旧 0.5 実装、本文採用）: exp_scenario_{A,B,C}_exp4_mismatch.csv
     の Proposed(oracle) / Y-only(fix_x) / X-only(fix_w) 条件の RMSE(Z)
  2. 人工データ（fixed 版再実行）: fixed_official/exp4/*_agg.csv の同条件
  3. Wine 実データ（fixed 版）: wine_fixed_pilot/wine_ablation_metrics.csv
     （X+Y / X-only / Y-only の AUC / AP / rmse_x）

出力:
  expfam/results/shared_z_ablation/existing_ablation_audit.csv

実行: python tools/shared_z_ablation/audit_existing_ablation_results.py
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
RES = _ROOT / "expfam" / "results"
OUT_DIR = RES / "shared_z_ablation"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # ── 1) 旧 0.5 実装（本文採用）人工データ ─────────────────────────
    for scen in ("A", "B", "C"):
        df = pd.read_csv(RES / f"exp_scenario_{scen}_exp4_mismatch.csv")
        oracle = df[df["correct"] == True]  # noqa: E712
        conds = {
            "proposed_XY": oracle,
            "y_only_fix_x": df[df["fix_x"] == True],   # noqa: E712
            "x_only_fix_w": df[df["fix_w"] == True],   # noqa: E712
        }
        for cname, sub in conds.items():
            if len(sub) == 0:
                continue
            rows.append({
                "source": "old05_manuscript_exp4",
                "implementation": "old_0.5",
                "dataset": f"synthetic_scen_{scen}",
                "condition": cname,
                "n_trials": len(sub),
                "rmse_Z_mean": float(sub["rmse_Z"].mean()),
                "rmse_Z_std": float(sub["rmse_Z"].std()),
                "rmse_X_mean": float(sub["rmse_X"].mean()),
                "auc_y": np.nan, "ap_y": np.nan,
            })

    # ── 2) fixed_official exp4 ───────────────────────────────────────
    for scen in ("a", "b", "c"):
        df = pd.read_csv(RES / "fixed_official" / "exp4"
                         / f"fixed_exp4_scen_{scen}_agg.csv")
        oracle = df[(df["fix_w"] == False) & (df["fix_x"] == False)  # noqa: E712
                    & (df["est_x"] == df["true_x"])
                    & (df["est_y"] == df["true_y"])]
        conds = {
            "proposed_XY": oracle,
            "y_only_fix_x": df[df["fix_x"] == True],   # noqa: E712
            "x_only_fix_w": df[df["fix_w"] == True],   # noqa: E712
        }
        for cname, sub in conds.items():
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            rows.append({
                "source": "fixed_official_exp4",
                "implementation": "fixed",
                "dataset": f"synthetic_scen_{scen.upper()}",
                "condition": cname,
                "n_trials": int(r.get("n_trials", np.nan)),
                "rmse_Z_mean": float(r["rmse_z_mean"]),
                "rmse_Z_std": float(r["rmse_z_std"]),
                "rmse_X_mean": float(r.get("rmse_x_mean", np.nan)),
                "auc_y": np.nan, "ap_y": np.nan,
            })

    # ── 3) Wine 実データ（fixed 版） ─────────────────────────────────
    wine = pd.read_csv(RES / "real_data" / "wine_fixed_pilot"
                       / "wine_ablation_metrics.csv")
    cond_map = {"X+Y": "proposed_XY",
                "Y-only": "y_only_fix_x", "Y_only": "y_only_fix_x",
                "X-only": "x_only_fix_w", "X_only": "x_only_fix_w"}
    for cond_raw, sub in wine.groupby("condition"):
        cname = cond_map.get(cond_raw, cond_raw)
        rows.append({
            "source": "wine_fixed_pilot",
            "implementation": "fixed",
            "dataset": "wine_real",
            "condition": cname,
            "n_trials": len(sub),
            "rmse_Z_mean": np.nan, "rmse_Z_std": np.nan,
            "rmse_X_mean": float(sub["rmse_x"].mean()),
            "auc_y": float(sub["auc_y"].mean()),
            "ap_y": float(sub["ap_y"].mean()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "existing_ablation_audit.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'existing_ablation_audit.csv'}")

    def git_head():
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=_ROOT).stdout.strip()
        except Exception:
            return "unknown"

    pd.DataFrame([{
        "script": "tools/shared_z_ablation/audit_existing_ablation_results.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/overdispersion-z-ablation",
        "inputs": "exp_scenario_*_exp4_mismatch.csv; fixed_official/exp4; "
                  "real_data/wine_fixed_pilot/wine_ablation_metrics.csv",
        "note": "read-only audit; no experiment executed",
    }]).to_csv(OUT_DIR / "existing_ablation_audit_runinfo.csv", index=False)


if __name__ == "__main__":
    main()
