# ChatGPTに渡すべきファイル一覧

作成日：2026-05-31

---

## 優先度 A：必須（まずこれを渡す）

| ファイル | 用途 | 備考 |
|--------|------|------|
| reports/chatgpt_handoff_report.md | 総合報告書（本ファイル） | 全情報の集約 |
| conference_submission_final_draft.md | 完成原稿 | 数式・実験内容・表の確認 |
| figures/fig1a_n_sweep_color.png | 提出用図(a) | n-sweep 結果 |
| figures/fig1b_misspecification_color.png | 提出用図(b) | 誤指定影響 |

---

## 優先度 B：実験数値の確認に必要

| ファイル | 用途 | 備考 |
|--------|------|------|
| expfam/results/exp_scenario_A_exp1_k.csv | Exp1: BIC vs k（Scen.A） | 表1の確認 |
| expfam/results/exp_scenario_A_exp2_n.csv | Exp2: n-sweep（Scen.A） | 図1(a)の数値確認 |
| expfam/results/exp_scenario_A_exp4_mismatch.csv | Exp4: 分布ミスマッチ（Scen.A） | 3.41×の確認 |
| reproduction/results/comparison/comparison_main_table.csv | 先行研究比較 | 表2の確認 |
| expfam/results/distribution_mismatch_fixed/mismatch_fixed_summary.csv | ミスマッチ集約版 | 3シナリオの悪化倍率確認 |

---

## 優先度 C：提案手法の理解に必要（コード）

| ファイル | 用途 | 備考 |
|--------|------|------|
| expfam/src/model_dual_expfam.py | 提案手法本体 | 実装詳細の確認 |
| expfam/src/model_expfam.py | 中間クラス（Y拡張） | Gaussian/Poisson Y の実装 |
| expfam/src/exp_scenario_lib.py | 実験設定定数 | n=150, L=5, NITER=8 等の確認 |

---

## 優先度 D：補足・質疑応答向け

| ファイル | 用途 | 備考 |
|--------|------|------|
| expfam/results/GEMINI_REPORT_MULTI_SCENARIO.md | 3シナリオ全結果報告 | Scen.B/C の詳細数値 |
| figures/figure_color_split_report.md | 図の説明・色の意味 | 図1(b)の見方の確認 |
| CLAUDE.md | 既知の問題・確定事項リスト | 0.5係数バグ、41.5×の整理 |
| reports/claims_and_evidence.md | 主張と根拠の対応表 | 質疑対策 |

---

## 現在の PowerPoint について

- PowerPoint ファイルはリポジトリ外にある（.docx/.pptx は git 管理外）
- 前半スライド（1〜6）の構成はユーザーが設計済み
- 後半（7〜15）は `reports/slide_materials.md` に文章案あり

---

## 注意事項

- 論文 PDF（paper/A_study_on_latent_structural_models_for_binary_rel.pdf）は先行研究論文
- 提案手法の論文 PDF は存在しない（原稿は Markdown）
- `archive/` 内は古いスクリプト・廃版データであり、渡す必要なし
- `.npy` ファイル（expfam/results/wine_F.npy）は numpy バイナリのため渡せない
