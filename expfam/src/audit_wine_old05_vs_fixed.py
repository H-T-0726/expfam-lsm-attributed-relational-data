"""
Wine実験: 旧0.5係数あり実装 (model_dual_expfam.py) vs fixed版 (model_dual_expfam_fixed.py)
vs 元論文(reproduction) の関係を、既存CSVの読み込みだけで整理する監査スクリプト。

モデルの再学習・再実行は一切行わない。読み取り専用。

出力先 (新規):
  expfam/results/real_data/wine_old05_audit/wine_file_inventory.csv
  expfam/results/real_data/wine_old05_audit/wine_k6_evidence_table.csv
  expfam/results/real_data/wine_old05_audit/wine_fixed_old05_existing_metrics_comparison.csv

既存のCSV・図・モデル実装は一切変更しない。
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

OUT_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "wine_old05_audit")

FIXED_BIC_K1TO9 = os.path.join(ROOT, "expfam", "results", "real_data", "wine_fixed_pilot", "wine_bic_k1to9.csv")
FIXED_ABLATION = os.path.join(ROOT, "expfam", "results", "real_data", "wine_fixed_pilot", "wine_ablation_metrics.csv")
OLD05_RESULTS = os.path.join(ROOT, "expfam", "results", "wine_dual_results.csv")
PAPER_REPRO_RESULTS = os.path.join(ROOT, "reproduction", "results", "results_real_wine.csv")

# Paper-reported reference values (user-provided, external to this repo; NOT re-derived here).
PAPER_REFERENCE = dict(
    k=6, rmse_x=0.7924, rmse_y=0.1415, w0=-1.1820, w=1.7221,
)


def build_file_inventory():
    rows = [
        dict(file_path="expfam/src/run_fixed_real_wine_pilot.py", category="fixed_wine",
             implementation="fixed_no_0.5", uses_half_factor="no", k_handling="k_sweep_1to9",
             contains_bic="yes", contains_rmse="partial (rmse_x only, no rmse_y)",
             safe_to_use_in_notion="yes",
             caution="ablation phase reuses the swept best_k (=3) as a fixed k; not an independent k=3 hardcode"),
        dict(file_path="expfam/results/real_data/wine_fixed_pilot/wine_bic_k1to9.csv", category="fixed_wine",
             implementation="fixed_no_0.5", uses_half_factor="no", k_handling="k_sweep_1to9",
             contains_bic="yes", contains_rmse="partial (rmse_x only)",
             safe_to_use_in_notion="yes", caution="BIC mean is minimized at k=3 (genuine 9k x 5trial sweep)"),
        dict(file_path="expfam/results/real_data/wine_fixed_pilot/wine_bic_bestk.csv", category="fixed_wine",
             implementation="fixed_no_0.5", uses_half_factor="no", k_handling="k_sweep_1to9 (summary)",
             contains_bic="yes", contains_rmse="no",
             safe_to_use_in_notion="yes", caution="best_k=3 stored here is argmin of wine_bic_k1to9.csv, not independently hardcoded"),
        dict(file_path="expfam/results/real_data/wine_fixed_pilot/wine_ablation_metrics.csv", category="fixed_wine",
             implementation="fixed_no_0.5", uses_half_factor="no", k_handling="k_fixed_3 (=swept best_k)",
             contains_bic="yes", contains_rmse="partial (rmse_x only)",
             safe_to_use_in_notion="yes", caution="ablation (X+Y/X_only/Y_only) computed only at k=best_k=3, not at k=6"),
        dict(file_path="expfam/src/run_wine_dual.py", category="old05_wine",
             implementation="old_0.5", uses_half_factor="yes", k_handling="k_fixed_6",
             contains_bic="yes (single k only, no comparison across k)", contains_rmse="yes (rmse_x, rmse_y)",
             safe_to_use_in_notion="no",
             caution="K=6 is a hardcoded constant (K,L,NITER,N_TRIALS=6,10,20,5); no loop over k, no best_k computation anywhere in this script"),
        dict(file_path="expfam/results/wine_dual_results.csv", category="old05_wine",
             implementation="old_0.5", uses_half_factor="yes", k_handling="k_fixed_6",
             contains_bic="yes (single k only)", contains_rmse="yes (rmse_X, rmse_Y)",
             safe_to_use_in_notion="no",
             caution="No 'k' column at all in this CSV -- confirms no multi-k comparison was ever stored"),
        dict(file_path="expfam/results/wine_F.npy", category="old05_wine",
             implementation="old_0.5", uses_half_factor="yes", k_handling="k_fixed_6",
             contains_bic="no", contains_rmse="no",
             safe_to_use_in_notion="no",
             caution="F matrix from the old 0.5-factor implementation; do not use as a fixed-version result or for F heatmaps in Notion"),
        dict(file_path="expfam/src/model_dual_expfam.py", category="implementation",
             implementation="old_0.5", uses_half_factor="yes", k_handling="n/a (model class, not k-specific)",
             contains_bic="n/a", contains_rmse="n/a",
             safe_to_use_in_notion="no (code, not a result)",
             caution="_calc_gradient L.159 `0.5 * w * (...)`; _calc_precision_matrix L.200 `0.5 * (w**2) * (...)` -- the spurious 0.5 noted in CLAUDE.md"),
        dict(file_path="expfam/src/model_dual_expfam_fixed.py", category="implementation",
             implementation="fixed_no_0.5", uses_half_factor="no", k_handling="n/a (model class, not k-specific)",
             contains_bic="n/a", contains_rmse="n/a",
             safe_to_use_in_notion="yes (code, not a result)",
             caution="L.77 and L.113: same Term3 with the 0.5 factor explicitly removed (`# 0.5 なし (fixed)`)"),
        dict(file_path="reproduction/src/model.py", category="paper_reproduction",
             implementation="old_0.5", uses_half_factor="yes", k_handling="n/a (model class)",
             contains_bic="no (no BIC function in this file)", contains_rmse="n/a",
             safe_to_use_in_notion="no (code, not a result)",
             caution="LatentStructuralModel._calc_gradient L.283 and _calc_precision_matrix L.353 both contain `0.5 * ...` Term3, same issue as model_dual_expfam.py"),
        dict(file_path="reproduction/src/experiment_paper_real.py", category="paper_reproduction",
             implementation="old_0.5 (uses LatentStructuralModel)", uses_half_factor="yes", k_handling="k_fixed_6",
             contains_bic="no (no BIC computed in this script)", contains_rmse="yes (rmse_X, rmse_Y)",
             safe_to_use_in_notion="no",
             caution="L.215 `k = 6  # k=6 (selected by BIC per paper)' -- a citation of the paper's reported value, not an independently computed sweep"),
        dict(file_path="reproduction/src/experiment_paper_2.py", category="paper_reproduction",
             implementation="old_0.5 (uses LatentStructuralModel)", uses_half_factor="yes", k_handling="k_sweep_1to10 (synthetic data only)",
             contains_bic="yes (paper Eq.26 formula, explicitly cited in code)", contains_rmse="unknown (not checked in this audit)",
             safe_to_use_in_notion="no (not applicable to real Wine data)",
             caution="This BIC sweep is run on SYNTHETIC data (n=150,d=15,k_true in {1,3,5,7,9}), never on the real Wine dataset"),
        dict(file_path="reproduction/results/results_real_wine.csv", category="paper_reproduction",
             implementation="old_0.5", uses_half_factor="yes", k_handling="k_fixed_6 (presumed, from experiment_paper_real.py)",
             contains_bic="no", contains_rmse="yes (RMSE_X, RMSE_Y)",
             safe_to_use_in_notion="no",
             caution="No k column; single fit only. Output of experiment_paper_real.py's hardcoded k=6 run"),
        dict(file_path="expfam/src/archive/experiment_real_all.py", category="old_unverified",
             implementation="unknown (single-family Bernoulli LSM, not dual-expfam)", uses_half_factor="unknown (not audited)",
             k_handling="k_fixed_3 (WINE_K=3, chosen as 'latent dim = num wine classes')",
             contains_bic="no", contains_rmse="no",
             safe_to_use_in_notion="no",
             caution="A third, separate legacy Wine script (archived) using a different k=3 hardcode rationale and a different (single-family) model; unrelated to the k=3 vs k=6 BIC question"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "wine_file_inventory.csv"), index=False)
    return out


def build_k6_evidence_table(fixed_bic_df):
    bic_mean_per_k = fixed_bic_df.groupby("k")["bic"].mean()
    fixed_bic_min_k = int(bic_mean_per_k.idxmin())

    rows = [
        dict(source="old05_wine_dual", file_path="expfam/src/run_wine_dual.py",
             k_value=6, k_source="hardcoded", bic_sweep_available="no", bic_min_k="NA",
             evidence=("K,L,NITER,N_TRIALS = 6,10,20,5 (single k, no loop over k); "
                       "no best_k computation in script; wine_dual_results.csv has no 'k' column at all"),
             conclusion="old05_k6_fixed_not_bic_selected"),
        dict(source="fixed_wine_pilot", file_path="expfam/results/real_data/wine_fixed_pilot/wine_bic_k1to9.csv",
             k_value=fixed_bic_min_k, k_source="bic_argmin", bic_sweep_available="yes", bic_min_k=fixed_bic_min_k,
             evidence=(f"best_k = int(bic_df.groupby('k')['bic'].mean().idxmin()) over a genuine k=1..9 x 5trial sweep; "
                       f"BIC_mean(k=3)={bic_mean_per_k.get(3, float('nan')):.1f} vs BIC_mean(k=6)={bic_mean_per_k.get(6, float('nan')):.1f}"),
             conclusion="fixed_bic_min_k3"),
        dict(source="paper_statement", file_path="(paper PDF, not machine-readable in this repo)",
             k_value=6, k_source="paper_statement", bic_sweep_available="no (not available in this repo)", bic_min_k="NA",
             evidence=("User-provided citation of the paper's reported Wine result: "
                       "K=6 based on BIC, L=10, RMSE_X=0.7924, RMSE_Y=0.1415, w0=-1.1820, w=1.7221"),
             conclusion="paper_states_k6_based_on_bic_but_raw_bic_not_available"),
        dict(source="paper_reproduction_real", file_path="reproduction/src/experiment_paper_real.py",
             k_value=6, k_source="hardcoded", bic_sweep_available="no", bic_min_k="NA",
             evidence="L.215: `k = 6  # k=6 (selected by BIC per paper)` -- comment cites the paper; no sweep code in this script",
             conclusion="paper_states_k6_based_on_bic_but_raw_bic_not_available"),
        dict(source="paper_reproduction_synthetic", file_path="reproduction/src/experiment_paper_2.py",
             k_value="NA (estimates k for synthetic k_true in {1,3,5,7,9})", k_source="bic_argmin",
             bic_sweep_available="yes (synthetic data only)", bic_min_k="depends on k_true; not applicable to real Wine",
             evidence="Reproduces paper Eq.(26) BIC formula on synthetic data (n=150, d=15); never applied to the real Wine dataset",
             conclusion="not_applicable_to_real_wine"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "wine_k6_evidence_table.csv"), index=False)
    return out


def build_metrics_comparison(fixed_bic_df, fixed_abl_df, old05_df):
    rows = []

    for k_val, label in [(3, "fixed_current_k3"), (6, "fixed_current_k6")]:
        sub = fixed_bic_df[fixed_bic_df["k"] == k_val]
        rows.append(dict(
            source=label, implementation="fixed_no_0.5", k=k_val,
            BIC_mean=round(sub["bic"].mean(), 2), BIC_std=round(sub["bic"].std(), 2),
            RMSE_X_mean=round(sub["rmse_x"].mean(), 4), RMSE_X_std=round(sub["rmse_x"].std(), 4),
            RMSE_Y_mean="NA", RMSE_Y_std="NA",
            w0_mean=round(sub["w0"].mean(), 4), w0_std=round(sub["w0"].std(), 4),
            w_mean=round(sub["w"].mean(), 4), w_std=round(sub["w"].std(), 4),
            AUC_Y_mean=round(sub["auc_y"].mean(), 4), AP_Y_mean=round(sub["ap_y"].mean(), 4),
            success_rate=round(sub["success"].mean(), 3),
            caution=("fixed version, BIC-optimal k" if k_val == 3 else
                     "fixed version forced to k=6 for reference only; BIC favors k=3 over this. "
                     "RMSE_Y not computed (Bernoulli Y evaluated via AUC/AP instead)"),
        ))

    rows.append(dict(
        source="old05_existing", implementation="old_0.5", k=6,
        BIC_mean=round(old05_df["BIC"].mean(), 2), BIC_std=round(old05_df["BIC"].std(), 2),
        RMSE_X_mean=round(old05_df["rmse_X"].mean(), 4), RMSE_X_std=round(old05_df["rmse_X"].std(), 4),
        RMSE_Y_mean=round(old05_df["rmse_Y"].mean(), 4), RMSE_Y_std=round(old05_df["rmse_Y"].std(), 4),
        w0_mean=round(old05_df["w0"].mean(), 4), w0_std=round(old05_df["w0"].std(), 4),
        w_mean=round(old05_df["w"].mean(), 4), w_std=round(old05_df["w"].std(), 4),
        AUC_Y_mean="NA", AP_Y_mean="NA", success_rate="NA (no success column in source CSV)",
        caution=("Old 0.5-factor implementation, k=6 fixed (no sweep -- see wine_k6_evidence_table.csv). "
                 "Not the same model as fixed_no_0.5; BIC formula (calc_bic_dual) is shared, "
                 "but the underlying Q_strict comes from a different gradient/precision matrix (0.5 factor present)."),
    ))

    rows.append(dict(
        source="paper_reference", implementation="paper_reported (not reproduced in this repo)",
        k=PAPER_REFERENCE["k"],
        BIC_mean="NA", BIC_std="NA",
        RMSE_X_mean=PAPER_REFERENCE["rmse_x"], RMSE_X_std="NA",
        RMSE_Y_mean=PAPER_REFERENCE["rmse_y"], RMSE_Y_std="NA",
        w0_mean=PAPER_REFERENCE["w0"], w0_std="NA",
        w_mean=PAPER_REFERENCE["w"], w_std="NA",
        AUC_Y_mean="NA", AP_Y_mean="NA", success_rate="NA",
        caution=("Reference values as reported by the user from the original paper. "
                 "Not re-derived in this repository; not a strict apples-to-apples comparison. "
                 "Do not use for win/lose comparison against fixed_no_0.5 or old_0.5 results."),
    ))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "wine_fixed_old05_existing_metrics_comparison.csv"), index=False)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fixed_bic_df = pd.read_csv(FIXED_BIC_K1TO9)
    fixed_abl_df = pd.read_csv(FIXED_ABLATION)
    old05_df = pd.read_csv(OLD05_RESULTS)

    inventory = build_file_inventory()
    evidence = build_k6_evidence_table(fixed_bic_df)
    comparison = build_metrics_comparison(fixed_bic_df, fixed_abl_df, old05_df)

    print("File inventory:\n", inventory[["file_path", "category", "k_handling"]].to_string(index=False))
    print("\nK=6 evidence table:\n", evidence[["source", "k_value", "k_source", "conclusion"]].to_string(index=False))
    print("\nMetrics comparison:\n", comparison[["source", "implementation", "k", "BIC_mean", "RMSE_X_mean", "RMSE_Y_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
