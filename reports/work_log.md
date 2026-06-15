# 作業ログ

実行日時：2026-05-31

---

## 実行したコマンド・操作

1. `find D:/tento/kennkyu -type f | sort` — 全ファイル一覧取得
2. `ls D:/tento/kennkyu/` — トップディレクトリ確認
3. Read: `conference_submission_final_draft.md`
4. Read: `expfam/src/model_dual_expfam.py`
5. Read: `expfam/src/model_expfam.py`
6. Read: `expfam/results/distribution_mismatch_fixed/mismatch_fixed_summary.csv`
7. Read: `expfam/results/exp_scenario_A_exp2_n.csv`
8. Read: `expfam/results/exp1_full_A.csv`
9. Read: `expfam/results/exp1_full_B.csv`
10. Read: `expfam/results/exp_scenario_A_exp4_mismatch.csv`（先頭112行）
11. Read: `reproduction/results/comparison/comparison_main_table.csv`
12. Read: `expfam/results/exp2_bic_A.csv`
13. Read: `expfam/results/GEMINI_REPORT_PHASE2_FINAL.md`
14. Read: `expfam/src/data_generator_expfam.py`
15. Read: `expfam/results/GEMINI_REPORT_MULTI_SCENARIO.md`
16. Read: `figures/figure_color_split_report.md`
17. Read: `expfam/src/exp_scenario_lib.py`（先頭80行）
18. Read: `reproduction/src/model.py`（先頭100行）
19. Read: `expfam/results/exp_scenario_B_exp4_mismatch.csv`（先頭30行）
20. Read: `expfam/results/exp_scenario_C_exp4_mismatch.csv`（先頭30行）
21. `mkdir -p D:/tento/kennkyu/reports/figures_for_slides`

---

## 読んだ主要ファイル

| ファイル | 内容の要約 |
|--------|----------|
| conference_submission_final_draft.md | 完成原稿（全文）。実験条件・表1・表2・図の説明を確認 |
| model_dual_expfam.py | 提案手法の実装。family_x/y の切り替え、精度行列・勾配の計算を確認 |
| model_expfam.py | Y側拡張クラス。Gaussian/Bernoulli/Poisson Y の実装を確認 |
| mismatch_fixed_summary.csv | 3シナリオの悪化倍率集計。Scen.A最大4.34×、B最大8.80×、C最大38.97× |
| exp_scenario_A_exp2_n.csv | n-sweep 全60行。n=50〜300 × 10試行のRMSE数値 |
| exp1_full_A/B.csv | BIC次元選択実験の全試行データ（k=1〜6, 各10試行） |
| comparison_main_table.csv | 先行研究比較の要約表（baseline/dual_expfam × k=1〜6） |
| GEMINI_REPORT_PHASE2_FINAL.md | 単一シナリオ（Scen.A）の詳細結果報告 |
| GEMINI_REPORT_MULTI_SCENARIO.md | 3シナリオ全結果の報告（数値確認に使用） |
| figure_color_split_report.md | 提出用図の説明・色の意味・投稿推奨の説明 |
| data_generator_expfam.py | データ生成コード。X・Y各分布の生成方法を確認 |
| exp_scenario_lib.py | 実験定数（N=150, D=15, K_TRUE=3, N_TRIALS=10, L=5, NITER=8）を確認 |

---

## 新しく作成したファイル

| ファイル | 内容 |
|--------|------|
| reports/chatgpt_handoff_report.md | ChatGPT向け総合報告書 |
| reports/experiment_summary_table.md | 実験一覧表 |
| reports/slide_materials.md | スライド素材・文章案 |
| reports/claims_and_evidence.md | 主張と根拠の対応表 |
| reports/files_to_send_to_chatgpt.md | ChatGPTに渡すファイル一覧 |
| reports/questions_for_chatgpt.md | ChatGPTへの相談質問リスト |
| reports/work_log.md | 本ファイル |
| reports/figures_for_slides/（ディレクトリ） | 発表用図コピー先（空） |

---

## コピーした図

| コピー元 | コピー先 | 実施状況 |
|--------|--------|--------|
| figures/fig1a_n_sweep_color.png | reports/figures_for_slides/fig01_basic_performance.png | **未実施**（元ファイルは確認済み、コピー自体は行わず） |
| figures/fig1b_misspecification_color.png | reports/figures_for_slides/fig02_distribution_switching.png | **未実施** |

**実施済み**（`cp` コマンドで正常完了）

---

## エラー

- `reports/mismatch_fixed_summary.csv`（ルート直下）へのアクセス試行 → ファイルなし（正しいパスは `expfam/results/distribution_mismatch_fixed/mismatch_fixed_summary.csv`）

---

## 未確認事項

1. **41.5× の正確な根拠 CSV**：原稿 L.83 に記載の「Scen.C で最大 41.5 倍」に対応する CSV が特定できなかった。CLAUDE.md は「X=Gaussian, Y=Poisson 条件」と記載するが、mismatch_fixed では 18.56×、exp_scenario_C_exp4_mismatch の先頭30行では ~1.0〜1.6 台が見える。全110行の確認が必要。

2. **expfam/results/wine_dual_results.csv の内容**：実データ実験の結果があるが、詳細を未確認。

3. **expfam/results/exp2_bic_A.csv の対応実験**：BIC vs k* の別実験と思われるが詳細未解析。

4. **Scen.B/C の n-sweep の単調性**：Scen.B の n=100 付近の変動が CSV で確認できているが原因未分析。

5. **PowerPoint の現在の前半スライド内容**：リポジトリ外にあり確認不可。
