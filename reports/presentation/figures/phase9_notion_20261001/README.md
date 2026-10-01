# Notion 発表用ページの図（2026-10-01）

`reports/presentation/phase9_progress_notion_presentation_20261001.md` の `[FIGURE n HERE]` に配置する図。
仕様は `reports/presentation/phase9_progress_notion_figure_spec_20261001.md`、キャプションは同じディレクトリの `CAPTIONS.md`。

## 図の一覧

| ファイル | 内容 | 種類 | 本文の位置 |
|---|---|---|---|
| `figure1_latent_model.png` / `.svg` | 共通潜在構造から属性と関係を生成 | 概念図 | §1 |
| `figure2_family_selection_flow.png` / `.svg` | 属性分布族の選択の流れ | 概念図 | §2 |
| `figure3_joint_family_k_pipeline.png` / `.svg` | 分布族 + K の同時選択 | 概念図（右下の実行例だけ CSV から読む） | §4 |
| `figure5_ktrue_exact_counts.png` / `.svg` | 真の潜在次元を変えたときの正解数 | データ図 | §9 |
| `figure6_k34_threshold_balance.png` / `.svg` | K = 3 と K = 4 の境界 | データ図 | §10 |

Notion には PNG を使う（SVG は編集・印刷用）。Figure 4（C_Q と C_Lap の概念比較）は作っていない。

## 生成方法

```
python tools/presentation/generate_phase9_notion_figures.py
```

- 出力先は固定（このディレクトリ）。実行するたびに 10 ファイルをすべて上書き生成する。
- 必要なもの: Python 3.13、matplotlib（日本語フォントは Yu Gothic → Meiryo → Noto Sans JP → … の順に自動選択）。

## 一次 source（Figure 5・6 と Figure 3 の実行例）

| 図 | 読む CSV | 使う列 |
|---|---|---|
| Figure 3 | `expfam/results/joint_family_k_selection/gate75b_20260927/joint_selection.csv` | `replicate`, `start_label`, `k_hat` |
| Figure 5 | `expfam/results/matched_k_true_sensitivity/phase9x_20260928/combined/confusion_Lap.csv`, `confusion_Q.csv` | `K_true`, `K_hat_<K_true>`, `total` |
| Figure 6 | `expfam/results/k34_boundary_decomposition/phase9y_20260929/ktrue4_rows.csv` | `fit_gain_Q_34`, `fit_gain_Lap_34`, `P_Z_increment_34`, `param_increment_Q`, `volume_increment_34`, `param_increment_Lap`, `margin_Q`, `margin_Lap` |

- Figure 5・6 の点の値は CSV から読み、script に数値を書き込んでいない。
- script は監査済みの値（`reports/presentation/phase9_progress_source_audit_20261001.md`）を期待値として assert する: Figure 5 の正解数（C_Lap 10/10/10/8、C_Q 10/9/7/1、各 10）、Figure 6 の行数 10・C_Q の閾値 75(1 + ln 2π) + 9 ln 75 = 251.698…・閾値を上回った数（C_Q 1/10、C_Lap 8/10）、各点と閾値の上下が committed の margin の符号と一致すること、Figure 3 の実行例（2, 3, 3）。一致しなければ図を描かずに止まる。
- 新しい推定・集計・統計量は計算していない（比率・信頼区間・回復確率は描かない）。

## 注意

- **図の数字を手で編集しない。** 変更は script を直して再生成する。
- Figure 1〜3 は概念図で、数値の主張を含まない（Figure 3 の実行例を除く）。
- 結果はすべて固定した人工データ条件での観測で、実装は experimental prototype。
  - Figure 5 は一般的な回復確率ではない。C_Lap が一般に優れることも示さない。
  - Figure 6 は算術的な分解で、過小選択の原因（P_Z が原因など）を示さない。左右の当てはまりの改善は定義が違う。
- 2 つの基準は中立な 2 色（C_Lap 青 `#2a78d6`・丸・実線、C_Q 橙 `#eb6834`・四角・破線）で、色だけでなく marker と線種でも区別する。この 2 色は dataviz の palette validator（light）で全項目 PASS。
