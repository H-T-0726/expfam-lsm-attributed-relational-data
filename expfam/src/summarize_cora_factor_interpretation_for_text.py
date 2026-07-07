"""
Cora F (factor) 解釈をNotion本文向けに整理し直すスクリプト。

既存の expfam/results/real_data/common_reconstruction_eval/cora_factor_top_words.csv
を読み込むだけで、モデルの再学習・再実行は一切行わない。

重要な制約（必ず報告に明記すること）:
  cora_factor_top_words.csv には、各factorにつき |F値| 上位5件のみが
  保存されている（run_common_realdata_reconstruction_eval.py の
  top_features_per_factor(..., top_n=5) の出力）。元のF行列(50 x k)自体は
  どこにも保存されていない。そのため、本スクリプトは「正方向top5」
  「負方向top5」「絶対値top10」を独立に再現することはできず、
  既存の5件のみから方向別に分類するにとどまる。

出力先 (すべて新規、既存ファイルは一切変更しない):
  expfam/results/real_data/common_reconstruction_eval/cora_k3_factor_top_features_for_text.csv
  expfam/results/real_data/common_reconstruction_eval/cora_k3_factor_summary_for_notion.csv
  expfam/figures/real_data/common_reconstruction_eval/cora_k3_factor_top_features_heatmap_for_text.png/.pdf
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IN_PATH = os.path.join(ROOT, "expfam", "results", "real_data", "common_reconstruction_eval", "cora_factor_top_words.csv")
OUT_CSV_DIR = os.path.join(ROOT, "expfam", "results", "real_data", "common_reconstruction_eval")
OUT_FIG_DIR = os.path.join(ROOT, "expfam", "figures", "real_data", "common_reconstruction_eval")

K_TARGET = 3
NOTE_LIMITED = (
    "feature index only; word name unavailable. Source CSV (cora_factor_top_words.csv) "
    "preserved only the top-5 features by |F| per factor; the full F matrix was not saved, "
    "so this is the complete set available for this direction without re-fitting the model."
)
NOTE_EMPTY = (
    "No feature with this sign was present among the saved top-5-by-|F| entries for this factor "
    "(all 5 saved entries happen to be negative). The true positive-side ranking is unavailable "
    "from existing files without re-fitting the model (out of scope: no retraining this task)."
)


def load_data():
    df = pd.read_csv(IN_PATH)
    return df[df["k"] == K_TARGET].copy()


def build_for_text_table(df_k3):
    rows = []
    for factor in sorted(df_k3["factor"].unique()):
        sub = df_k3[df_k3["factor"] == factor].sort_values("rank")

        # "absolute" direction: the up-to-5 saved rows, re-ranked by |F| (already sorted that way).
        abs_sorted = sub.reindex(sub["F_value"].abs().sort_values(ascending=False).index)
        for rank, (_, r) in enumerate(abs_sorted.iterrows(), start=1):
            idx_match = re.search(r"(\d+)$", str(r["feature_name"]))
            feat_idx = int(idx_match.group(1)) if idx_match else np.nan
            rows.append(dict(
                k=K_TARGET, factor=factor, direction="absolute", rank=rank,
                feature_label=r["feature_name"], feature_index=feat_idx,
                F_value=r["F_value"], abs_F_value=abs(r["F_value"]),
                short_note=NOTE_LIMITED + " (only 5 available, not 10)",
            ))

        pos_sub = sub[sub["F_value"] > 0].sort_values("F_value", ascending=False)
        if len(pos_sub) == 0:
            rows.append(dict(
                k=K_TARGET, factor=factor, direction="positive", rank=1,
                feature_label="N/A", feature_index=np.nan, F_value=np.nan, abs_F_value=np.nan,
                short_note=NOTE_EMPTY,
            ))
        else:
            for rank, (_, r) in enumerate(pos_sub.iterrows(), start=1):
                idx_match = re.search(r"(\d+)$", str(r["feature_name"]))
                feat_idx = int(idx_match.group(1)) if idx_match else np.nan
                rows.append(dict(
                    k=K_TARGET, factor=factor, direction="positive", rank=rank,
                    feature_label=r["feature_name"], feature_index=feat_idx,
                    F_value=r["F_value"], abs_F_value=abs(r["F_value"]),
                    short_note=NOTE_LIMITED,
                ))

        neg_sub = sub[sub["F_value"] < 0].sort_values("F_value", ascending=True)
        for rank, (_, r) in enumerate(neg_sub.iterrows(), start=1):
            idx_match = re.search(r"(\d+)$", str(r["feature_name"]))
            feat_idx = int(idx_match.group(1)) if idx_match else np.nan
            rows.append(dict(
                k=K_TARGET, factor=factor, direction="negative", rank=rank,
                feature_label=r["feature_name"], feature_index=feat_idx,
                F_value=r["F_value"], abs_F_value=abs(r["F_value"]),
                short_note=NOTE_LIMITED,
            ))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "cora_k3_factor_top_features_for_text.csv"), index=False)
    return out


def build_notion_summary(for_text_df):
    rows = []
    for factor in sorted(for_text_df["factor"].unique()):
        sub = for_text_df[for_text_df["factor"] == factor]

        pos = sub[(sub["direction"] == "positive") & (sub["feature_label"] != "N/A")]
        neg = sub[sub["direction"] == "negative"]
        absd = sub[sub["direction"] == "absolute"]

        top_positive = "; ".join(pos["feature_label"].tolist()) if len(pos) else "none available in saved data"
        top_negative = "; ".join(neg["feature_label"].tolist())
        top_absolute = "; ".join(absd["feature_label"].tolist())

        interpretation = (
            f"f{factor}では、一部の単語特徴（{top_absolute}）が強く関係しており、"
            f"この因子が単語出現Xの再構成に使われていることが確認できる。"
            f"ただし、単語名を復元できないため、具体的な研究トピックとしての意味づけは避ける。"
        )
        caution = (
            "単語名はLINQS公開Coraデータに含まれておらず復元不可。"
            "保存データはfactorごとに|F|上位5件のみのため、真の正方向top5は確認できない場合がある。"
            "回転不定性のため因子の意味は確定できない。"
        )

        rows.append(dict(
            factor=factor,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            top_absolute_features=top_absolute,
            interpretation_for_text=interpretation,
            caution=caution,
        ))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_CSV_DIR, "cora_k3_factor_summary_for_notion.csv"), index=False)
    return out


def make_heatmap(for_text_df):
    absd = for_text_df[for_text_df["direction"] == "absolute"].copy()
    features = list(dict.fromkeys(absd["feature_label"].tolist()))  # de-dup, preserve order
    factors = sorted(absd["factor"].unique())

    mat = np.full((len(features), len(factors)), np.nan)
    feat_pos = {f: i for i, f in enumerate(features)}
    for _, r in absd.iterrows():
        mat[feat_pos[r["feature_label"]], factors.index(r["factor"])] = r["F_value"]

    fig, ax = plt.subplots(figsize=(2.2 * len(factors) + 2.5, 0.35 * len(features) + 2.0))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#EDEDED")
    vmax = np.nanmax(np.abs(mat)) if np.any(~np.isnan(mat)) else 1.0
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([f"f{j}" for j in factors])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title(
        "Cora (k=3): top |F| features per factor\n"
        "feature names unavailable (LINQS Cora has no public vocabulary) -- shown as feature index",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="F value")
    fig.text(
        0.5, -0.01,
        "Blank/gray cells = this feature was not in that factor's saved top-5-by-|F| list "
        "(exact value unavailable, NOT necessarily zero). Word names cannot be recovered for this dataset.",
        ha="center", fontsize=7, style="italic", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(OUT_FIG_DIR, f"cora_k3_factor_top_features_heatmap_for_text.{ext}"),
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    return len(features), len(factors), int(np.sum(~np.isnan(mat))), mat.size


def main():
    os.makedirs(OUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    df_k3 = load_data()
    print(f"Loaded {len(df_k3)} rows for k={K_TARGET} from {IN_PATH}")

    for_text = build_for_text_table(df_k3)
    summary = build_notion_summary(for_text)
    n_feat, n_fac, n_filled, n_total = make_heatmap(for_text)

    print(f"\nHeatmap: {n_feat} unique features x {n_fac} factors = {n_total} cells, "
          f"{n_filled} filled ({n_filled/n_total:.0%}), {n_total-n_filled} blank/NaN ({(n_total-n_filled)/n_total:.0%})")
    print("\nfor_text table (head):")
    print(for_text.to_string(index=False))
    print("\nNotion summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
