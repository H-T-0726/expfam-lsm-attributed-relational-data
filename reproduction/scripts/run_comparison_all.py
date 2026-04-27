"""
Run all comparison experiments: Baseline (Mikawa 2024) vs Dual-ExpFam LSM.

Outputs:
  reproduction/results/comparison/comparison_control_exp1.csv   — main comparison
  reproduction/results/comparison/comparison_scen_a_exp1.csv    — auxiliary
  reproduction/results/comparison/not_applicable_log.csv        — Scenario B/C
  reproduction/reports/comparison_summary.md                    — final report

Usage:
  cd C:/kennkyu
  python reproduction/scripts/run_comparison_all.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Path setup
_ROOT = Path(__file__).parent.parent.parent
_REPRO_SRC = _ROOT / "reproduction" / "src"
_EXPFAM_SRC = _ROOT / "expfam" / "src"
sys.path.insert(0, str(_REPRO_SRC))
sys.path.insert(0, str(_EXPFAM_SRC))

from experiment_compare_with_dual import (   # noqa
    run_comparison_exp1_control,
    run_comparison_exp1_scenario_a,
)

_OUT = _ROOT / "reproduction" / "results" / "comparison"
_OUT.mkdir(parents=True, exist_ok=True)
_RPT = _ROOT / "reproduction" / "reports"

# ── Settings ─────────────────────────────────────────────────────────
N_TRIALS  = 5    # per k-value (both baseline and dual see same data)
L         = 5    # MC samples — matched to Dual-ExpFam standard
NUM_ITER  = 8    # EM iterations — matched to Dual-ExpFam standard
BASE_SEED = 4000 # data seeds: 4000, 4100, ..., 4400  (same as exp_scenario_lib.py)
K_LIST    = [1, 2, 3, 4, 5, 6]
N, D, K_TRUE = 150, 15, 3


def _agg(df, group_cols, metrics):
    """Aggregate mean ± std over trials."""
    rows = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for m in metrics:
            vals = grp[m].dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{m}_std"]  = float(vals.std())  if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


METRICS = ["rmse_Z", "rmse_F", "rmse_Y", "rmse_X", "rmse_sigma",
           "w0_err", "w_err", "BIC"]


def main():
    t_start = time.perf_counter()
    print("=" * 65)
    print("Baseline vs Dual-ExpFam Comparison Experiment")
    print(f"Settings: N_TRIALS={N_TRIALS}, L={L}, NUM_ITER={NUM_ITER}, "
          f"k_list={K_LIST}")
    print("=" * 65)

    # ─── Step 1: Main comparison — Control (Gauss-X × Bern-Y) ───────
    print("\n[1/3] Control scenario: True X=Gaussian, True Y=Bernoulli")
    print("      → Main fair comparison\n")
    df_ctrl = run_comparison_exp1_control(
        N_TRIALS=N_TRIALS, L=L, num_iter=NUM_ITER,
        base_seed=BASE_SEED, k_list=K_LIST,
        n=N, d=D, k_true=K_TRUE,
    )
    ctrl_path = _OUT / "comparison_control_exp1.csv"
    df_ctrl.to_csv(ctrl_path, index=False)
    print(f"  Saved: {ctrl_path}")

    # ─── Step 2: Auxiliary — Scenario A (Pois-X × Bern-Y) ──────────
    print("\n[2/3] Scenario A: True X=Poisson, True Y=Bernoulli")
    print("      → Auxiliary (Baseline X is misspecified)\n")
    df_a = run_comparison_exp1_scenario_a(
        N_TRIALS=N_TRIALS, L=L, num_iter=NUM_ITER,
        base_seed=BASE_SEED, k_list=K_LIST,
        n=N, d=D, k_true=K_TRUE,
    )
    a_path = _OUT / "comparison_scen_a_exp1.csv"
    df_a.to_csv(a_path, index=False)
    print(f"  Saved: {a_path}")

    # ─── Step 3: Log not-applicable scenarios ────────────────────────
    print("\n[3/3] Logging not-applicable scenarios (B and C) ...")
    na_records = [
        {"scenario": "B_Gauss_Pois", "model": "baseline",
         "reason": "Y=Poisson is non-binary; Baseline requires Y∈{0,1}",
         "applicable": False},
        {"scenario": "C_Bern_Gauss", "model": "baseline",
         "reason": "Y=Gaussian is continuous; Baseline requires Y∈{0,1}",
         "applicable": False},
    ]
    na_df = pd.DataFrame(na_records)
    na_path = _OUT / "not_applicable_log.csv"
    na_df.to_csv(na_path, index=False)
    print(f"  Saved: {na_path}")

    # ─── Step 4: Build summary tables ────────────────────────────────
    print("\n" + "=" * 65)
    print("Generating summary tables ...")

    # Main comparison table (Control, k_est=k_true=3)
    df_ctrl_k3 = df_ctrl[df_ctrl["k_est"] == K_TRUE]
    main_agg = _agg(df_ctrl_k3, ["model"], METRICS)

    # BIC-based k selection accuracy
    ctrl_agg_bic = _agg(df_ctrl, ["model", "k_est"], ["BIC"])
    bic_best = {}
    for model, grp in ctrl_agg_bic.groupby("model"):
        best_k = int(grp.loc[grp["BIC_mean"].idxmin(), "k_est"])
        bic_best[model] = best_k

    # Auxiliary table (Scenario A, k_est=k_true=3)
    df_a_k3 = df_a[df_a["k_est"] == K_TRUE]
    aux_agg = _agg(df_a_k3, ["model"], METRICS)

    # ─── Step 5: Write comparison_summary.md ─────────────────────────
    elapsed = (time.perf_counter() - t_start) / 60

    _write_summary(
        main_agg, bic_best, aux_agg, na_df,
        ctrl_agg_bic, df_ctrl, df_a,
        elapsed,
    )

    print(f"\nTotal elapsed: {elapsed:.1f} min")
    print("=" * 65)
    print("Comparison experiment completed.")


def _write_summary(main_agg, bic_best, aux_agg, na_df,
                   ctrl_agg_bic, df_ctrl, df_a, elapsed_min):
    """Write final Markdown report."""
    lines = []

    def h(level, text):
        lines.append("#" * level + " " + text)

    def add(*args):
        for s in args:
            lines.append(s)

    h(1, "Comparison Summary: Baseline (Mikawa 2024) vs Dual-ExpFam LSM")
    add(f"\nGenerated: 2026-04-24  |  Elapsed: {elapsed_min:.1f} min",
        f"\nSettings: N_TRIALS={N_TRIALS}, L={L}, NUM_ITER={NUM_ITER}, "
        f"n={N}, d={D}, k*={K_TRUE}\n")

    # ── 1. Audit summary ──────────────────────────────────────────────
    h(2, "1. Audit Summary")
    add(
        "",
        "| Item | Baseline | Dual-ExpFam |",
        "|------|----------|------------|",
        "| X distribution | Gaussian only | gaussian/bernoulli/poisson |",
        "| Y distribution | Bernoulli only | gaussian/bernoulli/poisson |",
        "| L (MC samples) | 10 (original) → **5 (matched)** | 5 |",
        "| EM iterations | 10–30 (original) → **8 (matched)** | 8 |",
        "| Trials | 3 (original) → **5 (matched)** | 10 |",
        "| BIC formula | −2logL + ((k+1)d − k(k−1)/2) ln n | Same (Gauss/Bern) |",
        "",
    )

    # ── 2. Comparability ─────────────────────────────────────────────
    h(2, "2. Comparability Classification")
    add(
        "",
        "| Scenario | Baseline Y-valid? | X-valid? | Type |",
        "|----------|-------------------|---------|------|",
        "| Control: Gauss-X × Bern-Y | ✅ Yes | ✅ Yes | **Main comparison** |",
        "| Scenario A: Pois-X × Bern-Y | ✅ Yes | ❌ Misspecified | Auxiliary |",
        "| Scenario B: Gauss-X × Pois-Y | ❌ Non-binary Y | ✅ Yes | **Not applicable** |",
        "| Scenario C: Bern-X × Gauss-Y | ❌ Continuous Y | ❌ | **Not applicable** |",
        "",
    )

    # ── 3. Main comparison ────────────────────────────────────────────
    h(2, "3. Main Comparison (Control: Gauss-X × Bern-Y, k=k*=3)")
    add(
        "",
        "**Condition**: same data, same L=5, same num_iter=8, same seeds.",
        "Both models are correctly specified for this scenario.",
        "",
        "| Metric | Baseline | Dual-ExpFam (Gauss,Bern) | Δ |",
        "|--------|----------|--------------------------|---|",
    )
    bl_row = main_agg[main_agg["model"] == "baseline"]
    du_row = main_agg[main_agg["model"] == "dual_expfam"]
    for m in ["rmse_Z", "rmse_F", "rmse_Y", "rmse_X", "w0_err", "w_err"]:
        bl_v = float(bl_row[f"{m}_mean"].iloc[0]) if len(bl_row) else float("nan")
        du_v = float(du_row[f"{m}_mean"].iloc[0]) if len(du_row) else float("nan")
        bl_s = float(bl_row[f"{m}_std"].iloc[0]) if len(bl_row) else float("nan")
        du_s = float(du_row[f"{m}_std"].iloc[0]) if len(du_row) else float("nan")
        delta = du_v - bl_v
        sign = "+" if delta > 0 else ""
        label = m.upper().replace("_ERR", " err").replace("_", "(").rstrip(")")
        add(f"| {m} | {bl_v:.4f}±{bl_s:.4f} | {du_v:.4f}±{du_s:.4f} | {sign}{delta:.4f} |")
    add("")

    # BIC k-selection
    add("**BIC-based k* identification (Control):**", "")
    add("| Model | BIC-optimal k | Correct? |",
        "|-------|--------------|---------|")
    for model, best_k in sorted(bic_best.items()):
        correct = "✅" if best_k == K_TRUE else "❌"
        add(f"| {model} | k={best_k} | {correct} |")
    add("")

    # ── 4. Auxiliary comparison ───────────────────────────────────────
    h(2, "4. Auxiliary Comparison (Scenario A: Pois-X × Bern-Y, k=k*=3)")
    add(
        "",
        "**⚠️ Caution**: Baseline X is misspecified here (Gaussian assumption on Poisson data).",
        "This result shows X-mismatch robustness, NOT fundamental model superiority.",
        "",
        "| Metric | Baseline (X misspecified) | Dual-ExpFam (correct) | Δ |",
        "|--------|--------------------------|----------------------|---|",
    )
    bl_a = aux_agg[aux_agg["model"] == "baseline_X_misspecified"]
    du_a = aux_agg[aux_agg["model"] == "dual_expfam"]
    for m in ["rmse_Z", "rmse_F", "rmse_Y", "rmse_X", "w0_err", "w_err"]:
        bl_v = float(bl_a[f"{m}_mean"].iloc[0]) if len(bl_a) else float("nan")
        du_v = float(du_a[f"{m}_mean"].iloc[0]) if len(du_a) else float("nan")
        bl_s = float(bl_a[f"{m}_std"].iloc[0]) if len(bl_a) else float("nan")
        du_s = float(du_a[f"{m}_std"].iloc[0]) if len(du_a) else float("nan")
        delta = du_v - bl_v
        sign = "+" if delta > 0 else ""
        add(f"| {m} | {bl_v:.4f}±{bl_s:.4f} | {du_v:.4f}±{du_s:.4f} | {sign}{delta:.4f} |")
    add("")

    # ── 5. Not applicable ─────────────────────────────────────────────
    h(2, "5. Not Applicable Cases")
    add("", "The following scenarios cannot be run with Baseline:", "")
    for _, row in na_df.iterrows():
        add(f"- **{row['scenario']}**: {row['reason']}")
    add("")

    # ── 6. Conclusion ─────────────────────────────────────────────────
    h(2, "6. Conclusion")
    add(
        "",
        "### What can be claimed from this experiment",
        "",
        "**Control scenario (fair comparison):**",
        "- Both models share the same distributional assumption (Gauss-X × Bern-Y).",
        "- Any performance difference reflects algorithmic/implementation differences,",
        "  NOT distributional advantage.",
        "- DualExpFamLSM(gaussian, bernoulli) should be numerically equivalent to Baseline",
        "  as both implement the same generative model; differences, if any, come from",
        "  minor implementation choices (BIC formula, sigma update, scaling).",
        "",
        "**Scenario A (Auxiliary — X misspecified for Baseline):**",
        "- When true X is Poisson, the Baseline (Gaussian X assumption) is misspecified.",
        "- Dual-ExpFam with correct family_x='poisson' is expected to outperform.",
        "- This advantage is **distributional** (correct vs wrong X model), not algorithmic.",
        "",
        "**Cannot claim from this experiment:**",
        "- Scenarios B and C cannot be evaluated for Baseline due to non-binary Y.",
        "- Dual-ExpFam's advantage in Scenarios B and C cannot be compared against Baseline.",
        "- Any advantage in these scenarios must be attributed to family generality alone.",
        "",
        "### Separation of advantages",
        "",
        "| Advantage type | Evidence | Scenario |",
        "|---------------|---------|---------|",
        "| Algorithmic/estimation strength | Main comparison (Control) | Gauss-X × Bern-Y |",
        "| Distributional (X-mismatch) | Auxiliary comparison | Pois-X × Bern-Y |",
        "| Distributional (Y-mismatch) | Not comparable (Y non-binary) | B, C |",
        "",
    )

    h(2, "7. Unresolved Issues")
    add(
        "",
        "- **Baseline Y-side**: Scenarios B (Pois-Y) and C (Gauss-Y) cannot be run",
        "  with Baseline without Y transformation. A binarisation (y>0 → 1) would",
        "  be possible but is not valid for main results.",
        "- **Trial count mismatch**: Baseline original used 3 trials, Dual-ExpFam used 10.",
        "  This comparison uses 5 (compromise). Results may have higher variance than",
        "  the original paper's 3-trial setup.",
        "- **EM iteration budget**: Original Baseline used 10–30 iterations; here 8 is used",
        "  for fairness. Baseline may not fully converge in 8 iterations.",
        "",
    )

    report_path = _RPT / "comparison_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()
