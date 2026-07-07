"""
Cora実験 (balanced_degree k-sweep + held-out link prediction) をNotion資料用に
要約するスクリプト。

既存の以下のディレクトリのCSVを読み込むだけで、モデルの再学習・再実行は
一切行わない:
  expfam/results/real_data/cora_balanced_k_sweep/
  expfam/results/real_data/cora_heldout_link_prediction/

出力先 (すべて新規):
  expfam/results/real_data/cora_clean/*.csv
  expfam/figures/real_data/cora_clean/*.png/.pdf

既存のCSV・図・モデル実装は一切変更しない。
"""

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

KSWEEP_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "cora_balanced_k_sweep")
HELDOUT_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "cora_heldout_link_prediction")
HELDOUT_FIG_DIR = os.path.join(ROOT, "expfam", "figures", "real_data", "cora_heldout_link_prediction")

OUT_CSV_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "cora_clean")
OUT_FIG_DIR = os.path.join(ROOT, "expfam", "figures", "real_data", "cora_clean")


def load_inputs():
    paths = dict(
        ksweep_agg=os.path.join(KSWEEP_DIR, "cora_balanced_k_sweep_agg.csv"),
        ksweep_bestk=os.path.join(KSWEEP_DIR, "cora_balanced_k_sweep_bestk.csv"),
        heldout_agg=os.path.join(HELDOUT_DIR, "cora_heldout_agg.csv"),
        heldout_bestk=os.path.join(HELDOUT_DIR, "cora_heldout_bestk.csv"),
        heldout_summary=os.path.join(HELDOUT_DIR, "cora_heldout_summary.csv"),
    )
    data = {}
    missing = []
    for key, path in paths.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            missing.append(path)
    return data, missing


def build_file_inventory():
    rows = [
        dict(file_path="expfam/data/cora/cora.content", category="data", experiment_type="raw_data",
             subset="full Cora (n=2708)", n=2708, d=1433, k_values="n/a", contains_metrics="no",
             contains_figures="no", safe_to_use_in_notion="no (raw data file, not a result)",
             caution="Used as the source for all Cora subsets below; not re-downloaded this session"),
        dict(file_path="expfam/data/cora/cora.cites", category="data", experiment_type="raw_data",
             subset="full Cora (n=2708)", n=2708, d="n/a", k_values="n/a", contains_metrics="no",
             contains_figures="no", safe_to_use_in_notion="no (raw data file, not a result)",
             caution="Citation edge list; symmetrized to undirected Y in all subset scripts"),
        dict(file_path="expfam/src/run_fixed_real_cora_subset_pilot.py", category="bfs_pilot",
             experiment_type="k_metrics_pilot", subset="BFS from highest-degree node", n=300, d=50,
             k_values="unknown (not audited in detail)", contains_metrics="yes", contains_figures="yes",
             safe_to_use_in_notion="no", caution="Oldest pilot; non-balanced BFS subset, only 6/7 classes present; superseded by balanced_degree"),
        dict(file_path="expfam/results/real_data/cora_subset_pilot/", category="bfs_pilot",
             experiment_type="k_metrics_pilot", subset="BFS from highest-degree node", n=300, d=50,
             k_values="unknown", contains_metrics="yes", contains_figures="yes",
             safe_to_use_in_notion="no", caution="Use only as historical context if needed, not as Notion main result"),
        dict(file_path="expfam/src/run_fixed_real_cora_balanced_subset_pilot.py", category="old_or_unclear",
             experiment_type="subset_strategy_comparison", subset="balanced_random vs balanced_degree", n=280, d=50,
             k_values="n/a (subset-selection comparison, not a k-sweep)", contains_metrics="yes", contains_figures="yes",
             safe_to_use_in_notion="no (methodology justification only)",
             caution="Established that balanced_degree (433 edges) is far less degenerate than balanced_random (79 edges, 181 isolated nodes); informs but is not itself the main result"),
        dict(file_path="expfam/results/real_data/cora_balanced_subset_pilot/", category="old_or_unclear",
             experiment_type="subset_strategy_comparison", subset="balanced_random vs balanced_degree", n=280, d=50,
             k_values="n/a", contains_metrics="yes", contains_figures="yes",
             safe_to_use_in_notion="no", caution="Can be cited as the reason balanced_degree was chosen, if needed"),
        dict(file_path="expfam/src/run_fixed_real_cora_balanced_k_sweep.py", category="balanced_degree_k_sweep",
             experiment_type="k_sweep", subset="balanced_degree", n=280, d=50, k_values="1,2,3,4,5,6",
             contains_metrics="yes", contains_figures="yes", safe_to_use_in_notion="yes",
             caution="MAIN RESULT 1. 18/18 fits success (k=1..6 x 3 trials)"),
        dict(file_path="expfam/results/real_data/cora_balanced_k_sweep/", category="balanced_degree_k_sweep",
             experiment_type="k_sweep", subset="balanced_degree", n=280, d=50, k_values="1,2,3,4,5,6",
             contains_metrics="yes", contains_figures="yes", safe_to_use_in_notion="yes",
             caution="best_k differs by criterion: BIC->1, AUC/AP->6, NMI/ARI->3"),
        dict(file_path="expfam/src/run_fixed_real_cora_heldout_link_prediction.py", category="heldout_link_prediction",
             experiment_type="heldout_link_prediction", subset="balanced_degree", n=280, d=50, k_values="3,6",
             contains_metrics="yes", contains_figures="yes", safe_to_use_in_notion="yes",
             caution="MAIN RESULT 2. test_edge_ratio=0.2, split_trials=0,1,2, model_trials=0,1 (n_trials=6 per k)"),
        dict(file_path="expfam/results/real_data/cora_heldout_link_prediction/", category="heldout_link_prediction",
             experiment_type="heldout_link_prediction", subset="balanced_degree", n=280, d=50, k_values="3,6",
             contains_metrics="yes", contains_figures="yes", safe_to_use_in_notion="yes",
             caution="test_AUC/AP both best at k=6; NMI/ARI both best at k=3 (same pattern as in-sample k-sweep)"),
        dict(file_path="expfam/src/run_fixed_real_cora_scaling_heldout.py", category="old_or_unclear",
             experiment_type="scaling_heldout_link_prediction", subset="balanced_degree (n=280/490/700/980)",
             n="280-980", d=50, k_values="3,6", contains_metrics="yes", contains_figures="yes",
             safe_to_use_in_notion="no (not requested for this Cora section; different design from the requested held-out spec)",
             caution="Most recent / largest Cora study (scaling across n), but uses a different evaluation pipeline "
                     "(all-candidate AP/AUC) than cora_heldout_link_prediction; treat as a separate follow-up, not a substitute"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "cora_file_inventory.csv"), index=False)
    return out


def build_k_sweep_summary(data):
    agg = data["ksweep_agg"].copy()
    bestk = data["ksweep_bestk"].iloc[0]

    interp = {}
    for k in agg["k"]:
        tags = []
        if k == int(bestk["best_k_by_BIC"]):
            tags.append("BIC minimum")
        if k == int(bestk["best_k_by_AUC"]):
            tags.append("AUC max (best link reconstruction)")
        if k == int(bestk["best_k_by_AP"]) and "AUC max (best link reconstruction)" not in tags:
            tags.append("AP max (best link reconstruction)")
        if k == int(bestk["best_k_by_NMI"]):
            tags.append("NMI max (best label-structure)")
        if k == int(bestk["best_k_by_ARI"]) and "NMI max (best label-structure)" not in tags:
            tags.append("ARI max (best label-structure)")
        interp[k] = "; ".join(tags) if tags else "not best by any tested criterion"

    out = pd.DataFrame({
        "k": agg["k"],
        "BIC_mean": agg["bic_mean"].round(2),
        "BIC_std": agg["bic_std"].round(2),
        "AUC_mean": agg["auc_mean"].round(4),
        "AUC_std": agg["auc_std"].round(4),
        "AP_mean": agg["ap_mean"].round(4),
        "AP_std": agg["ap_std"].round(4),
        "NMI_mean": agg["nmi_mean"].round(4),
        "NMI_std": agg["nmi_std"].round(4),
        "ARI_mean": agg["ari_mean"].round(4),
        "ARI_std": agg["ari_std"].round(4),
        "silhouette_mean": agg["silhouette_mean"].round(4),
        "silhouette_std": agg["silhouette_std"].round(4),
        "success_rate": agg["success_rate"],
        "runtime_mean": agg["runtime_mean"].round(2),
        "short_interpretation": agg["k"].map(interp),
    })
    out.to_csv(os.path.join(OUT_CSV_DIR, "cora_k_sweep_summary_clean.csv"), index=False)
    return out, bestk


def build_heldout_summary(data):
    agg = data["heldout_agg"].copy()
    raw = data["heldout_summary"]
    bestk = data["heldout_bestk"].iloc[0]

    edge_stats = raw.groupby("k").agg(
        test_positive_edges_mean=("test_edges", "mean"),
        test_negative_edges_mean=("test_neg", "mean"),
    ).reset_index()
    agg = agg.merge(edge_stats, on="k", how="left")

    interp = {}
    for k in agg["k"]:
        tags = []
        if k == int(bestk["best_k_by_test_AUC"]):
            tags.append("best held-out AUC")
        if k == int(bestk["best_k_by_test_AP"]) and "best held-out AUC" not in tags:
            tags.append("best held-out AP")
        if k == int(bestk["best_k_by_NMI"]):
            tags.append("best label-structure (NMI)")
        if k == int(bestk["best_k_by_ARI"]) and "best label-structure (NMI)" not in tags:
            tags.append("best label-structure (ARI)")
        interp[k] = "; ".join(tags) if tags else "not best by any tested criterion"

    out = pd.DataFrame({
        "k": agg["k"],
        "n_trials": agg["n_trials"],
        "train_AUC_mean": agg["train_auc_mean"].round(4),
        "train_AP_mean": agg["train_ap_mean"].round(4),
        "test_AUC_mean": agg["test_auc_mean"].round(4),
        "test_AP_mean": agg["test_ap_mean"].round(4),
        "random_AP_baseline_mean": agg["random_ap_baseline"].round(4),
        "test_positive_edges_mean": agg["test_positive_edges_mean"].round(1),
        "test_negative_edges_mean": agg["test_negative_edges_mean"].round(1),
        "NMI_mean": agg["nmi_mean"].round(4),
        "ARI_mean": agg["ari_mean"].round(4),
        "silhouette_mean": agg["silhouette_mean"].round(4),
        "success_rate": agg["success_rate"],
        "runtime_mean": agg["runtime_mean"].round(2),
        "short_interpretation": agg["k"].map(interp),
    })
    out.to_csv(os.path.join(OUT_CSV_DIR, "cora_heldout_summary_clean.csv"), index=False)
    return out, bestk


def build_bestk_clean(ksweep_summary, ksweep_bestk, heldout_summary, heldout_bestk):
    def v(df, k, col):
        row = df[df["k"] == k]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    rows = [
        dict(criterion="BIC", best_k=int(ksweep_bestk["best_k_by_BIC"]),
             value=round(float(ksweep_bestk["bic_at_best_k"]), 2),
             interpretation="In-sample BIC is minimized at the simplest model (k=1)",
             caution="Based on cora_balanced_k_sweep (in-sample fit); conflicts with the link/structure criteria below"),
        dict(criterion="in_sample_AUC", best_k=int(ksweep_bestk["best_k_by_AUC"]),
             value=round(float(ksweep_bestk["auc_at_best_k"]), 4),
             interpretation="Larger k reconstructs the observed citation links best (in-sample)",
             caution="In-sample reconstruction only, not a held-out generalization measure"),
        dict(criterion="in_sample_AP", best_k=int(ksweep_bestk["best_k_by_AP"]),
             value=round(float(ksweep_bestk["ap_at_best_k"]), 4),
             interpretation="Same direction as in-sample AUC",
             caution="Absolute AP is low (~0.29) due to extreme edge sparsity (y_density~1.1%)"),
        dict(criterion="heldout_AUC", best_k=int(heldout_bestk["best_k_by_test_AUC"]),
             value=round(float(heldout_bestk["test_auc_at_best_k"]), 4),
             interpretation="k=6 generalizes better to held-out citation links than k=3",
             caution="Based on cora_heldout_link_prediction (test_edge_ratio=0.2, 3 splits x 2 model trials); only k=3,6 tested"),
        dict(criterion="heldout_AP", best_k=int(heldout_bestk["best_k_by_test_AP"]),
             value=round(float(heldout_bestk["test_ap_at_best_k"]), 4),
             interpretation="k=6 best among tested k for held-out AP; well above random baseline",
             caution=f"Random AP baseline = {float(heldout_bestk['random_ap_baseline']):.4f}; only k=3 and k=6 were tested, not a full sweep"),
        dict(criterion="NMI", best_k=int(ksweep_bestk["best_k_by_NMI"]),
             value=round(float(ksweep_bestk["nmi_at_best_k"]), 4),
             interpretation="k=3 best recovers the 7-class label structure among tested k (in-sample k-sweep)",
             caution="Absolute NMI (~0.31) is moderate, not a clean recovery of the 7 classes; held-out k-sweep shows the same k=3 pattern (NMI~0.27)"),
        dict(criterion="ARI", best_k=int(ksweep_bestk["best_k_by_ARI"]),
             value=round(float(ksweep_bestk["ari_at_best_k"]), 4),
             interpretation="Same direction as NMI",
             caution="Absolute ARI (~0.19) is low; label structure is only partially recovered, not fully identified"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "cora_bestk_clean.csv"), index=False)
    return out


def build_use_in_notion_plan():
    rows = [
        dict(item="k-sweep metrics figure",
             file_path="expfam/figures/real_data/cora_clean/cora_k_sweep_metrics_clean.png",
             where_to_use="本文：結果1 k-sweep", importance="必須",
             caution="BIC・リンク性能・ラベル構造でbest_kが一致しない"),
        dict(item="k-sweep summary table (cora_k_sweep_summary_clean.csv)",
             file_path="expfam/results/real_data/cora_clean/cora_k_sweep_summary_clean.csv",
             where_to_use="本文：結果1 k-sweepの数値表", importance="必須",
             caution="18/18 fit success。k=1..6, 3trial平均"),
        dict(item="held-out metrics figure",
             file_path="expfam/figures/real_data/cora_clean/cora_heldout_metrics_clean.png",
             where_to_use="本文：結果2 held-out link prediction", importance="必須",
             caution="in-sampleとheld-outの差、random baselineとの比較を明示"),
        dict(item="held-out summary table (cora_heldout_summary_clean.csv)",
             file_path="expfam/results/real_data/cora_clean/cora_heldout_summary_clean.csv",
             where_to_use="本文：結果2 held-outの数値表", importance="必須",
             caution="k=3,6のみ。test_edge_ratio=0.2, 3 split x 2 model trial"),
        dict(item="best-k disagreement table (cora_bestk_clean.csv)",
             file_path="expfam/results/real_data/cora_clean/cora_bestk_clean.csv",
             where_to_use="本文：結果1/結果2のまとめ。「best_kは目的によって変わる」の根拠", importance="必須",
             caution="BIC->k=1, AUC/AP->k=6, NMI/ARI->k=3 で一致しないことを明示する表"),
        dict(item="Z visualization k=3 (clean copy)",
             file_path="expfam/figures/real_data/cora_clean/cora_z_k3_clean.png",
             where_to_use="トグル・補助（潜在空間の直感的な可視化）", importance="推奨（必須ではない）",
             caution="既存図(cora_heldout_z_k3.png)のコピー。再生成はしていない"),
        dict(item="Z visualization k=6 (clean copy)",
             file_path="expfam/figures/real_data/cora_clean/cora_z_k6_clean.png",
             where_to_use="トグル・補助（潜在空間の直感的な可視化）", importance="推奨（必須ではない）",
             caution="既存図(cora_heldout_z_k6.png)のコピー。再生成はしていない"),
        dict(item="BFS pilot (cora_subset_pilot)", file_path="expfam/results/real_data/cora_subset_pilot/",
             where_to_use="使わない", importance="使用不可",
             caution="非balanced・6/7クラスのみのBFS subset。balanced_degreeに上書きされた古い結果"),
        dict(item="balanced subset strategy comparison",
             file_path="expfam/results/real_data/cora_balanced_subset_pilot/",
             where_to_use="使わない（必要ならbalanced_degree選定理由の脚注のみ）", importance="任意",
             caution="balanced_randomがいかに退化しているか(79 edges, 181 isolated)を示す手法選定の根拠資料"),
        dict(item="cora_scaling_heldout (n=280/490/700/980)",
             file_path="expfam/results/real_data/cora_scaling_heldout/",
             where_to_use="今回のCora章では使わない（次のステップ・別章向け）", importance="任意（将来用）",
             caution="評価指標体系が異なる(all-candidate AP/AUC)別の拡張研究。今回依頼された設計とは別物として扱う"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "cora_use_in_notion_plan.csv"), index=False)
    return out


def make_fig_k_sweep_metrics(ksweep_summary, bestk):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ks = ksweep_summary["k"].values

    axes[0].errorbar(ks, ksweep_summary["BIC_mean"], yerr=ksweep_summary["BIC_std"],
                      marker="o", color="#4C72B0", capsize=3)
    best_bic_k = int(bestk["best_k_by_BIC"])
    axes[0].axvline(best_bic_k, color="#C44E52", linestyle="--", linewidth=1.2, label=f"BIC min: k={best_bic_k}")
    axes[0].set_title("BIC vs k")
    axes[0].set_xlabel("k")
    axes[0].legend(fontsize=8)

    axes[1].errorbar(ks, ksweep_summary["AUC_mean"], yerr=ksweep_summary["AUC_std"],
                      marker="o", color="#55A868", capsize=3, label="AUC")
    axes[1].errorbar(ks, ksweep_summary["AP_mean"], yerr=ksweep_summary["AP_std"],
                      marker="s", color="#DD8452", capsize=3, label="AP")
    best_auc_k = int(bestk["best_k_by_AUC"])
    axes[1].axvline(best_auc_k, color="#C44E52", linestyle="--", linewidth=1.2, label=f"AUC/AP max: k={best_auc_k}")
    axes[1].set_title("Link reconstruction (in-sample) vs k")
    axes[1].set_xlabel("k")
    axes[1].legend(fontsize=8)

    axes[2].errorbar(ks, ksweep_summary["NMI_mean"], yerr=ksweep_summary["NMI_std"],
                      marker="o", color="#8172B2", capsize=3, label="NMI")
    axes[2].errorbar(ks, ksweep_summary["ARI_mean"], yerr=ksweep_summary["ARI_std"],
                      marker="s", color="#937860", capsize=3, label="ARI")
    best_nmi_k = int(bestk["best_k_by_NMI"])
    axes[2].axvline(best_nmi_k, color="#C44E52", linestyle="--", linewidth=1.2, label=f"NMI/ARI max: k={best_nmi_k}")
    axes[2].set_title("Label structure vs k")
    axes[2].set_xlabel("k")
    axes[2].legend(fontsize=8)

    fig.suptitle("Cora balanced_degree k-sweep (n=280, d=50)\nBest K depends on the evaluation criterion.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_FIG_DIR, f"cora_k_sweep_metrics_clean.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_fig_heldout_metrics(heldout_summary):
    ks = heldout_summary["k"].tolist()
    metrics = [
        ("train_AUC", "train_AUC_mean"),
        ("test_AUC", "test_AUC_mean"),
        ("train_AP", "train_AP_mean"),
        ("test_AP", "test_AP_mean"),
    ]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ks))
    width = 0.18
    for i, (label, col) in enumerate(metrics):
        offset = (i - 1.5) * width
        ax.bar(x + offset, heldout_summary[col], width, label=label, color=colors[i])

    random_ap = heldout_summary["random_AP_baseline_mean"].iloc[0]
    ax.axhline(random_ap, color="gray", linestyle=":", linewidth=1.5, label=f"random AP baseline={random_ap:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Cora held-out link prediction: train vs test (k=3 vs k=6)")
    ax.legend(fontsize=8, loc="upper left")
    fig.text(
        0.5, -0.02,
        "Note: Held-out evaluation is closer to unknown-link prediction than in-sample reconstruction.",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_FIG_DIR, f"cora_heldout_metrics_clean.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_z_figures():
    copied = []
    skipped = []
    pairs = [
        ("cora_heldout_z_k3", "cora_z_k3_clean"),
        ("cora_heldout_z_k6", "cora_z_k6_clean"),
    ]
    for src_stem, dst_stem in pairs:
        for ext in ("png", "pdf"):
            src = os.path.join(HELDOUT_FIG_DIR, f"{src_stem}.{ext}")
            dst = os.path.join(OUT_FIG_DIR, f"{dst_stem}.{ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied.append(dst)
            else:
                skipped.append(src)
    return copied, skipped


def main():
    os.makedirs(OUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    data, missing = load_inputs()
    print(f"Loaded {len(data)} input files; missing: {missing if missing else 'none'}")

    build_file_inventory()
    ksweep_summary, ksweep_bestk = build_k_sweep_summary(data)
    heldout_summary, heldout_bestk = build_heldout_summary(data)
    build_bestk_clean(ksweep_summary, ksweep_bestk, heldout_summary, heldout_bestk)
    build_use_in_notion_plan()

    make_fig_k_sweep_metrics(ksweep_summary, ksweep_bestk)
    make_fig_heldout_metrics(heldout_summary)
    copied, skipped = copy_z_figures()

    print("\nK-sweep summary:\n", ksweep_summary[["k", "BIC_mean", "AUC_mean", "AP_mean", "NMI_mean", "ARI_mean"]])
    print("\nHeld-out summary:\n", heldout_summary[["k", "train_AUC_mean", "test_AUC_mean", "train_AP_mean", "test_AP_mean"]])
    print(f"\nCopied Z figures: {copied}")
    if skipped:
        print(f"WARNING: missing source Z figures: {skipped}")


if __name__ == "__main__":
    main()
