"""
既存 3×3 分布ミスマッチ実験の監査（Phase 7、read-only 入力）。

目的:
  1. 本文採用実験（旧 0.5 実装）exp_scenario_{A,B,C}_exp4_mismatch.csv から
     条件別悪化倍率を再計算し、看板数値（A=3.41×, B=7.35×, C=41.5×、
     図1(b) の 23.6×）の根拠 CSV / 条件を特定する（KI-003）。
  2. fixed 版再実行（fixed_official/exp4）と補助実験
     （distribution_mismatch_fixed）の倍率を同じ定義で並べ、
     旧版と fixed 版の対応表を作る（KI-001/KI-002 の影響整理）。
  3. 修論（過分散・共有Z・per-column）への接続に使える条件を抽出する。

入力（read-only）:
  expfam/results/exp_scenario_{A,B,C}_exp4_mismatch.csv
  expfam/results/fixed_official/exp4/fixed_exp4_scen_{a,b,c}_{agg,ratios}.csv
  expfam/results/distribution_mismatch_fixed/mismatch_fixed_summary.csv
  expfam/results/distribution_mismatch_fixed/comparison_quick.csv

出力（新規のみ）:
  expfam/results/mismatch_audit/mismatch_audit_old05_conditions.csv
  expfam/results/mismatch_audit/mismatch_audit_summary.csv
  expfam/results/mismatch_audit/mismatch_audit_runinfo.csv

実行: python tools/research_audit/audit_mismatch_experiments.py
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
RES = _ROOT / "expfam" / "results"
OUT_DIR = RES / "mismatch_audit"

DOCUMENTED = {
    # (scenario, source): documented max ratio
    ("A", "manuscript"): 3.41,
    ("B", "manuscript"): 7.35,
    ("C", "manuscript"): 41.5,
    ("C", "fig1b_gray_bar"): 23.6,
    ("A", "fixed_official"): 4.34,
    ("B", "fixed_official"): 9.04,
    ("C", "fixed_official"): 40.37,
    ("C", "mismatch_fixed"): 38.97,
}


def audit_old05():
    """本文採用実験（旧 0.5 実装）の条件別倍率を再計算。"""
    frames = []
    for scen in ("A", "B", "C"):
        p = RES / f"exp_scenario_{scen}_exp4_mismatch.csv"
        df = pd.read_csv(p)
        df["scenario_file"] = p.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    rows = []
    for scen, sdf in raw.groupby("scenario"):
        oracle = sdf[sdf["correct"] == True]  # noqa: E712
        oracle_mean = float(oracle["rmse_Z"].mean())
        oracle_median = float(oracle["rmse_Z"].median())
        for (fx, fy, fw, fxx, cond), c in sdf.groupby(
                ["model_fx", "model_fy", "fix_w", "fix_x", "condition"]):
            mean_z = float(c["rmse_Z"].mean())
            rows.append({
                "scenario": scen,
                "model_fx": fx, "model_fy": fy,
                "fix_w": fw, "fix_x": fxx,
                "condition": cond,
                "is_ablation": bool(fw or fxx),
                "is_oracle": bool(c["correct"].iloc[0]),
                "n_trials": len(c),
                "rmse_Z_mean": mean_z,
                "rmse_Z_std": float(c["rmse_Z"].std()),
                "rmse_Z_median": float(c["rmse_Z"].median()),
                "oracle_rmse_Z_mean": oracle_mean,
                "ratio_vs_oracle_mean": mean_z / oracle_mean,
                "ratio_vs_oracle_median":
                    float(c["rmse_Z"].median()) / oracle_median,
            })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1) 旧 0.5 実装（本文採用） ────────────────────────────────────
    old = audit_old05()
    old.to_csv(OUT_DIR / "mismatch_audit_old05_conditions.csv", index=False)

    print("=== Old-0.5 (manuscript basis) exp4: max degradation per scenario ===")
    summary_rows = []
    for scen in ("A", "B", "C"):
        s = old[(old["scenario"] == scen) & (~old["is_oracle"])]
        s_mis = s[~s["is_ablation"]]  # 純粋な family 誤指定のみ
        for label, sub in (("misspec_only", s_mis), ("incl_ablation", s)):
            if len(sub) == 0:
                continue
            imax = sub["ratio_vs_oracle_mean"].idxmax()
            r = sub.loc[imax]
            summary_rows.append({
                "source": "old05_manuscript_exp4",
                "scenario": scen, "condition_set": label,
                "max_ratio": r["ratio_vs_oracle_mean"],
                "max_condition": r["condition"],
                "est_x": r["model_fx"], "est_y": r["model_fy"],
                "fix_w": r["fix_w"], "fix_x": r["fix_x"],
                "oracle_rmse": r["oracle_rmse_Z_mean"],
                "rmse_at_max": r["rmse_Z_mean"],
                "n_trials": r["n_trials"],
            })
            print(f"  {scen} [{label:14s}] max={r['ratio_vs_oracle_mean']:6.2f}x"
                  f"  cond={r['condition']}  fix_w={r['fix_w']} fix_x={r['fix_x']}")

    # 41.5× / 23.6× の特定条件チェック（Scen.C）
    print("\n=== Scen.C specific condition check (KI-003) ===")
    c = old[(old["scenario"] == "C") & (~old["is_ablation"])]
    for fx, fy, tag in (("gaussian", "poisson", "documented 41.5x (text)"),
                        ("gaussian", "bernoulli", "documented 23.6x (fig1b gray bar)")):
        sub = c[(c["model_fx"] == fx) & (c["model_fy"] == fy)]
        if len(sub):
            r = sub.iloc[0]
            summary_rows.append({
                "source": "old05_manuscript_exp4",
                "scenario": "C", "condition_set": f"check:{tag}",
                "max_ratio": r["ratio_vs_oracle_mean"],
                "max_condition": r["condition"],
                "est_x": fx, "est_y": fy,
                "fix_w": False, "fix_x": False,
                "oracle_rmse": r["oracle_rmse_Z_mean"],
                "rmse_at_max": r["rmse_Z_mean"],
                "n_trials": r["n_trials"],
            })
            print(f"  X={fx}, Y={fy}: ratio={r['ratio_vs_oracle_mean']:.2f}x "
                  f"(rmse={r['rmse_Z_mean']:.4f} / oracle={r['oracle_rmse_Z_mean']:.4f})"
                  f"   <- {tag}")

    # ── 2) fixed_official exp4 ───────────────────────────────────────
    print("\n=== fixed_official exp4 ===")
    for scen in ("a", "b", "c"):
        p = RES / "fixed_official" / "exp4" / f"fixed_exp4_scen_{scen}_ratios.csv"
        df = pd.read_csv(p)
        mis = df[(df["fix_w"] == False) & (df["fix_x"] == False)  # noqa: E712
                 & (df["ratio_vs_oracle"] > 1.001)]
        if len(mis) == 0:
            continue
        imax = mis["ratio_vs_oracle"].idxmax()
        r = mis.loc[imax]
        summary_rows.append({
            "source": "fixed_official_exp4",
            "scenario": scen.upper(), "condition_set": "misspec_only",
            "max_ratio": r["ratio_vs_oracle"],
            "max_condition": r["condition_name"],
            "est_x": r["est_x"], "est_y": r["est_y"],
            "fix_w": r["fix_w"], "fix_x": r["fix_x"],
            "oracle_rmse": r["oracle_rmse_z_mean"],
            "rmse_at_max": r["rmse_z_mean"],
            "n_trials": np.nan,
        })
        print(f"  {scen.upper()} max={r['ratio_vs_oracle']:6.2f}x "
              f"cond={r['condition_name']} (est_x={r['est_x']}, est_y={r['est_y']})")

    # ── 3) distribution_mismatch_fixed ───────────────────────────────
    print("\n=== distribution_mismatch_fixed ===")
    p = RES / "distribution_mismatch_fixed" / "mismatch_fixed_summary.csv"
    df = pd.read_csv(p)
    for scen, sdf in df.groupby("scenario"):
        mis = sdf[~sdf["is_oracle"]]
        imax = mis["degradation_ratio"].idxmax()
        r = mis.loc[imax]
        summary_rows.append({
            "source": "distribution_mismatch_fixed",
            "scenario": scen, "condition_set": "misspec_only",
            "max_ratio": r["degradation_ratio"],
            "max_condition": f"X={r['est_x']}, Y={r['est_y']}",
            "est_x": r["est_x"], "est_y": r["est_y"],
            "fix_w": False, "fix_x": False,
            "oracle_rmse": r["oracle_rmse"],
            "rmse_at_max": r["rmse_z_mean"],
            "n_trials": np.nan,
        })
        print(f"  {scen} max={r['degradation_ratio']:6.2f}x "
              f"(est_x={r['est_x']}, est_y={r['est_y']})")

    summary = pd.DataFrame(summary_rows)

    # documented 値との照合列
    def doc_lookup(row):
        key_map = {"old05_manuscript_exp4": "manuscript",
                   "fixed_official_exp4": "fixed_official",
                   "distribution_mismatch_fixed": "mismatch_fixed"}
        return DOCUMENTED.get((row["scenario"], key_map.get(row["source"], "")),
                              np.nan)

    summary["documented_value"] = summary.apply(doc_lookup, axis=1)
    summary["matches_documented"] = np.where(
        summary["documented_value"].notna(),
        (abs(summary["max_ratio"] - summary["documented_value"])
         / summary["documented_value"] < 0.05),
        np.nan)
    summary.to_csv(OUT_DIR / "mismatch_audit_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'mismatch_audit_summary.csv'}")

    def git_head():
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=_ROOT).stdout.strip()
        except Exception:
            return "unknown"

    pd.DataFrame([{
        "script": "tools/research_audit/audit_mismatch_experiments.py",
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "branch": "research/overdispersion-z-ablation",
        "inputs": "exp_scenario_*_exp4_mismatch.csv; fixed_official/exp4; "
                  "distribution_mismatch_fixed",
        "note": "read-only audit; no experiment executed",
    }]).to_csv(OUT_DIR / "mismatch_audit_runinfo.csv", index=False)


if __name__ == "__main__":
    main()
