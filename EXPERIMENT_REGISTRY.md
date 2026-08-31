# EXPERIMENT_REGISTRY.md

実験結果ファイル（CSV・図・スクリプト）の対応表。
数値主張を行う際は、必ずこの表で「状態」と「原稿採用」列を確認すること。

## 状態の分類

- `current_main`：本文（原稿）採用実験
- `current_support`：本文を支える補助実験（直接引用はしないが整合性確認に使用）
- `fixed_support`：fixed版（0.5除去）による補助実験。本文未採用
- `old`：提出前の旧バージョン（提出用図と異なる）
- `archive`：初期実装・古いシナリオ構成の結果。参照のみ
- `ai_generated`：AIによる生成レポート。未検証
- `unverified`：実装は存在するが結果の検証が未完了

---

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| Scen.A Exp1 | k変化（k探索によるBIC選択） | `exp_run_scenario_A.py`, `exp_scenario_lib.py` | `expfam/results/exp_scenario_A_exp1_k.csv` | `expfam/results/fig_scenario_A_exp1_k.*` | current_main | ✓ | k*=3が選択される（claims_and_evidence.md参照） |
| Scen.A Exp2 | n変化（n=50→300のRMSE(Z)変化） | 同上 | `expfam/results/exp_scenario_A_exp2_n.csv` | `expfam/results/fig_scenario_A_exp2_n.*`（旧）／`figures/fig1a_n_sweep_color.*`（提出用、3シナリオ統合） | current_main | ✓ | 49%（48.8%）削減（claims_and_evidence.md L.10） |
| Scen.A Exp3 | d変化（属性次元数の影響） | 同上 | `expfam/results/exp_scenario_A_exp3_d.csv` | `expfam/results/fig_scenario_A_exp3_d.*` | current_support | △ | 原稿本文での直接引用は未確認（claims_and_evidence.md L.17） |
| Scen.A Exp4 | 誤指定実験（X/Y分布族の組み合わせ） | 同上 | `expfam/results/exp_scenario_A_exp4_mismatch.csv` | `expfam/results/fig_scenario_A_exp4_heatmap.*`, `fig_scenario_A_exp5_barchart.*`（旧）／`figures/fig1b_misspecification_color.*`（提出用、3シナリオ統合） | current_main | ✓ | 最大3.41倍（Bern-Bern条件） |
| Scen.B Exp1 | k変化 | `exp_run_scenario_B.py` | `expfam/results/exp_scenario_B_exp1_k.csv` | `expfam/results/fig_scenario_B_exp1_k.*` | current_main | ✓ | k*=3が選択される |
| Scen.B Exp2 | n変化 | 同上 | `expfam/results/exp_scenario_B_exp2_n.csv` | `expfam/results/fig_scenario_B_exp2_n.*`（旧）／`figures/fig1a_n_sweep_color.*`（提出用） | current_main | ✓ | 31%削減 |
| Scen.B Exp3 | d変化 | 同上 | `expfam/results/exp_scenario_B_exp3_d.csv` | `expfam/results/fig_scenario_B_exp3_d.*` | current_support | △ | 原稿本文での直接引用は未確認 |
| Scen.B Exp4 | 誤指定実験 | 同上 | `expfam/results/exp_scenario_B_exp4_mismatch.csv` | `expfam/results/fig_scenario_B_exp4_heatmap.*`, `fig_scenario_B_exp5_barchart.*`（旧）／`figures/fig1b_misspecification_color.*`（提出用） | current_main | ✓ | 最大7.35倍（claims_and_evidence.md L.13、CSVでの条件特定は要確認） |
| Scen.C Exp1 | k変化 | `exp_run_scenario_C.py` | `expfam/results/exp_scenario_C_exp1_k.csv` | `expfam/results/fig_scenario_C_exp1_k.*` | current_main | ✓ | k*=3が選択される |
| Scen.C Exp2 | n変化 | 同上 | `expfam/results/exp_scenario_C_exp2_n.csv` | `expfam/results/fig_scenario_C_exp2_n.*`（旧）／`figures/fig1a_n_sweep_color.*`（提出用） | current_main | ✓ | 62%削減 |
| Scen.C Exp3 | d変化 | 同上 | `expfam/results/exp_scenario_C_exp3_d.csv` | `expfam/results/fig_scenario_C_exp3_d.*` | current_support | △ | Scen.Cはdに対して平坦（CLAUDE.md/claims_and_evidence.md記載、解釈の検証は要） |
| Scen.C Exp4 | 誤指定実験 | 同上 | `expfam/results/exp_scenario_C_exp4_mismatch.csv` | `expfam/results/fig_scenario_C_exp4_heatmap.*`, `fig_scenario_C_exp5_barchart.*`（旧）／`figures/fig1b_misspecification_color.*`（提出用） | current_main | ✓ | 23.6倍（図1b灰色バー）・41.5倍（本文記載、図に対応バーなし）。KI-003参照 |
| exp1_full A/B/C | BIC付き全メトリクス詳細 | `run_exp1_full_metrics.py` | `expfam/results/exp1_full_{A,B,C}.csv` | なし | current_support | △ | Exp1-4の補足。直接引用箇所は未確認 |
| exp2_bic A/B/C | BIC次元選択詳細 | `run_exp2_bic_v2.py` | `expfam/results/exp2_bic_{A,B,C}.csv`, `exp2_bic_log.txt` | なし | current_support | △ | KI-010（BICパラメータ数定義）の検証対象 |
| Control比較 | 先行研究（NOLTA2024再現）との同条件比較 | `reproduction/src/experiment_compare_with_dual.py`, `run_comparison_all.py` | `reproduction/results/comparison/comparison_main_table.csv`, `comparison_aux_table.csv`, `comparison_control_exp1.csv`, `comparison_scen_a_exp1.csv`, `comparison_not_applicable_table.csv` | なし | current_main | ✓ | RMSE(Z)差 < 0.001（claims_and_evidence.md L.11、5試行のみ） |
| fixed版 mismatch grid | fixed版（0.5除去）単独の誤指定3×3×3グリッド | `run_mismatch_fixed.py`, `model_dual_expfam_fixed.py` | `expfam/results/distribution_mismatch_fixed/mismatch_fixed_summary.csv`, `mismatch_fixed_all_trials.csv` | `expfam/figures/distribution_mismatch_fixed/heatmap_*_fixed.*` 等 | fixed_support | ✗ | Scen.C最悪38.97倍（true=bern/gauss, est=poisson/bernoulli）。旧版との比較は含まない |
| fixed版 old-vs-fixed比較 | 旧版とfixed版の対比較（oracle/conv/worst×3シナリオ） | `run_comparison_quick.py`, `model_dual_expfam_fixed.py` | `expfam/results/distribution_mismatch_fixed/comparison_quick.csv` | `expfam/figures/distribution_mismatch_fixed/comparison_old_vs_fixed.*`（生成元未確認、KI-004） | fixed_support | ✗ | ratio_fix_old = 0.27〜1.23倍、条件依存。5試行のみ |
| Wine実験 | Wine実データへの適用 | `run_wine_dual.py` | `expfam/results/wine_dual_results.csv`, `wine_F.npy`、（参考）`reproduction/results/results_real_wine.csv` | なし | unverified | ✗ | 結果の検証・解釈が未完了（KI-006） |
| 旧版図（results内） | 提出前の旧バージョン図（シナリオ別・統合） | `make_figures_existing.py` 等 | — | `expfam/results/fig_scenario_{A,B,C}_*.pdf/png`、`expfam/results/fig1_rmse_vs_n.*`、`expfam/results/fig2_*.*` | old | ✗ | `expfam/README.md`に「提出直前の旧版」と明記。提出用は`figures/`配下 |
| 提出用図 fig1a | n-RMSE（3シナリオ統合、提出用） | （生成元スクリプト未特定。`make_figures_existing.py`等の系譜の可能性） | Scen.A/B/C Exp2 CSV由来 | `figures/fig1a_n_sweep_color.pdf/png` | current_main | ✓ | CLAUDE.mdに提出用図として明記 |
| 提出用図 fig1b | 誤指定（3シナリオ統合、提出用） | 同上 | Scen.A/B/C Exp4 CSV由来 | `figures/fig1b_misspecification_color.pdf/png` | current_main | ✓ | 灰色バー＝23.6倍（Scen.C, Fixed Gauss-X/Bern-Y）。CLAUDE.md「図1(b)と誤指定倍率の扱い」参照 |
| GEMINI_REPORT系 | AI生成の実験レポート（フェーズ別・ステップ別） | — | — | `expfam/results/GEMINI_REPORT_MULTI_SCENARIO.md`, `GEMINI_REPORT_PHASE2_FINAL.md`, `expfam/results/archive/GEMINI_REPORT_*.md`（STEP2〜STEP4_5, EXP1〜EXP5, BIC, ULTIMATE） | ai_generated | ✗ | KI-007。研究者による検証未完了。一次CSVで再確認が必要 |
| archive（初期実装） | 初期シングルシナリオ実装の実験群 | `expfam/src/archive/experiment_*.py` | `expfam/results/archive/*.csv` | `expfam/results/archive/*.png` | archive | ✗ | 現行のシナリオA/B/C構成とは異なる。参照のみ |

---

## fixed版 official再実験（人工データ、2026-06中旬）

0.5係数を除去した `DualExpFamLSMFixed` で、シナリオA/B/CのExp1-4を正式に再実行したもの。本文（原稿）は0.5あり版
（`model_dual_expfam.py`）の結果で書かれているため、以下はいずれも「原稿未採用・KI-001の検証材料」の位置づけ。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| fixed official Exp1 (BIC, k1-9) | fixed版でのBIC k選択（k=1〜9に拡張） | `run_fixed_official_exp1_bic_full.py`, `run_fixed_official_exp1_bic_k9_extension.py`, `run_fixed_official_exp1_bic_quick.py` | `expfam/results/fixed_official/{exp1,exp1_k9,quick}/*.csv` | `expfam/figures/fixed_official/*` | current_support | ✗ | k=9まで拡張しても3シナリオとも過大次元に誤らないことを確認 |
| fixed official Exp2 (n-sweep) | fixed版でのn=50→300 RMSE(Z)推移 | `run_fixed_official_exp2_n_sweep.py` | `expfam/results/fixed_official/exp2/*.csv` | 同上 | current_support | ✗ | A:-40%, B:-17%, C:-62%（`reports/real_data_experiment_plan.md` §2） |
| fixed official Exp3 (d-sweep) | fixed版でのd変化 | `run_fixed_official_exp3_d_sweep.py` | `expfam/results/fixed_official/exp3/*.csv` | 同上 | current_support | ✗ | A:-22.5%、CはflatでBは中央値改善もoutlierあり |
| fixed official Exp4 (mismatch) | fixed版での誤指定3×3grid | `run_fixed_official_exp4_scen_ab.py`, `run_fixed_official_exp4_scen_c.py` | `expfam/results/fixed_official/exp4/*.csv` | 同上 | current_support | ✗ | A最大4.34×, B最大9.04×, C最大40.37×（0.5あり旧版の23.6/41.5倍とは別条件、混同注意） |
| half-factor check | 0.5係数問題の追加検証（dry_run/full/scenario_c_extra） | `run_half_factor_minimal_check.py`, `run_half_factor_scenario_c_extra.py` | `expfam/results/half_factor_check/{dry_run,full,scenario_c_extra}/*.csv` | なし | current_support | ✗ | KI-001関連の補助検証。原稿数値との対応整理は未実施 |

---

## 実データ実験フェーズ（Wine / Cora / MovieLens、2026-06-17〜2026-07-07、fixed版使用）

`reports/real_data_experiment_plan.md`（計画）、`reports/movielens_pilot_design.md`（MovieLens設計）、
`reports/real_data_experiment_summary.md`（総括）を参照。いずれも `DualExpFamLSMFixed` を使用し、
「pilot（試行）→ audit（既存CSV突合、Wineのみ）→ clean/final_clean（本文・スライド用整形）」という系譜を持つ。
整形系スクリプト（`summarize_*.py`, `audit_*.py`）はいずれも**既存CSVの読み込みのみでモデルの再学習は行わない**
（docstringに明記）。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| Wine fixed pilot | Wine実データ、BIC k=1-9、ablation（X+Y/X_only/Y_only） | `run_fixed_real_wine_pilot.py` | `expfam/results/real_data/wine_fixed_pilot/*.csv` | `expfam/figures/real_data/wine_fixed_pilot/*` | current_support | ✗ | KI-006（Wine未評価）はこの実験で実質解消。BIC最小k=3が真のクラス数（3）と一致 |
| Wine old05 audit | 旧0.5版・fixed版・論文再現の3者突合（読取専用） | `audit_wine_old05_vs_fixed.py` | `expfam/results/real_data/wine_old05_audit/*.csv` | なし | current_support | ✗ | 既存CSVの再集計のみ、モデル再実行なし |
| Wine clean | Wine 最終整形（figures/スライド用） | `summarize_wine_for_figures.py` | `expfam/results/real_data/wine_clean/*.csv` | `expfam/figures/real_data/wine_clean/*` | current_support | △ | 発表資料への転記候補 |
| Cora subset pilot (BFS, 不採用) | Cora BFSサブセット | `run_fixed_real_cora_subset_pilot.py` | `expfam/results/real_data/cora_subset_pilot/*.csv` | `expfam/figures/real_data/cora_subset_pilot/*` | archive | ✗ | max-degreeノードからのBFSで1クラスが78%を占め不適切と判断。balanced_degree版に置き換え済み |
| Cora balanced subset / k-sweep / held-out / scaling | balanced_degreeサブセット（採用）での一連の検証（n=280→700） | `run_fixed_real_cora_balanced_subset_pilot.py`, `run_fixed_real_cora_balanced_k_sweep.py`, `run_fixed_real_cora_heldout_link_prediction.py`, `run_fixed_real_cora_scaling_heldout.py` | `expfam/results/real_data/cora_balanced_subset_pilot/`, `cora_balanced_k_sweep/`, `cora_heldout_link_prediction/`, `cora_scaling_heldout/` | 対応図一式 | current_support | △ | held-out test_AP≈2.6〜2.8×random（n=280）。自然ネットワークでの汎化性能を確認。BICは疎密度でk=1を選択する限界あり |
| Cora clean | Cora 最終整形 | `summarize_cora_for_figures.py`, `summarize_cora_factor_interpretation_for_text.py` | `expfam/results/real_data/cora_clean/*.csv` | `expfam/figures/real_data/cora_clean/*` | current_support | △ | 発表資料への転記候補 |
| MovieLens data prep | movie-nodeプロジェクション、genre-stratified subset作成 | `prepare_movielens_data.py` | `expfam/results/real_data/movielens_data_prep/*.csv` | `expfam/figures/real_data/movielens_data_prep/*` | current_support | — | 前処理のみ。出力: `expfam/data/movielens_pilot/*.npy` |
| MovieLens Poisson / heldout count / Bernoulli t80 / colike interpretation | Bernoulli-X / Poisson-Y の新規組み合わせの主実験一式 | `run_fixed_real_movielens_poisson_pilot.py`, `run_fixed_real_movielens_heldout_count.py`, `run_fixed_real_movielens_bernoulli_t80_pilot.py`, `run_fixed_real_movielens_colike_interpretation.py` | `expfam/results/real_data/movielens_{poisson_pilot,heldout_count,bernoulli_t80_pilot,colike_interpretation}/*.csv` | 対応図一式 | current_support | △ | in-sample評価でありstrict held-outではない（pair mask未対応）。Poisson overdispersion（var/mean≈10）あり。t80は補助でCora比較用 |
| MovieLens colike clean / final clean | 本文/Notion用3指標に縮約した版（clean）と、監査用フル指標版（final_clean、cleanを上書きしない） | `summarize_movielens_colike_for_notion.py`, `summarize_movielens_final_for_figures.py` | `expfam/results/real_data/movielens_colike_clean/*.csv`, `movielens_final_clean/*.csv` | `expfam/figures/real_data/movielens_colike_clean/*` | current_support | △ | 両者は役割が異なる（縮約版／フル版）。名前が紛らわしいため参照時は用途を明記すること |
| 3データセット横断再構成比較 | Wine/Cora/MovieLensのF行列再構成の統一評価 | `run_common_realdata_reconstruction_eval.py`, `summarize_common_realdata_reconstruction_eval.py` | `expfam/results/real_data/common_reconstruction_eval/*.csv` | `expfam/figures/real_data/common_reconstruction_eval/*` | current_support | △ | 3データセットの横並び比較。発表・スライド向き |

**原稿採用の凡例：** ✓=既に採用済み／△=採用候補（未確定、本文・スライドへの転記は今後の判断）／✗=未採用（補助・監査用途のみ）。

---

## 過分散・共有Z・per-column family フェーズ（2026-07-08〜、branch: research/overdispersion-z-ablation、fixed系列 experimental 使用）

`reports/research_direction/phase0_current_state_20260708.md`（開始時監査）、
`reports/overdispersion/`（診断・NB設計・pair mask設計）、
`reports/mismatch_audit/mismatch_audit_report_20260708.md`（既存ミスマッチ監査）を参照。
すべて `DualExpFamLSMFixed` を継承する experimental モデル
（`expfam/src/experimental/`: masked / NB / per-column）を使用。旧0.5版は不使用。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| MovieLens 過分散診断 | 周辺 vs 条件付き過分散の分離診断 + plug-in PPC | `tools/overdispersion/diagnose_movielens_overdispersion.py` | `expfam/results/overdispersion/movielens_overdispersion_diagnostics.csv`, `movielens_ppc_summary.csv` | `figures/overdispersion/movielens_y_distribution.*`, `movielens_mean_variance.*` | current_support | ✗ | 周辺 var/mean=9.89 だが条件付き Pearson 過分散は k=3:1.14 / k=5:0.76。KI-012 の再解釈 |
| MovieLens strict held-out | pair mask による strict held-out で Poisson/NB/full(リーク参照) 比較 | `tools/overdispersion/run_movielens_strict_heldout.py` | `expfam/results/overdispersion/movielens_strict_heldout_{summary,agg}.csv` | `figures/overdispersion/movielens_strict_heldout_comparison_k{3,5}.*` | current_support | ✗ | k=5 poisson_strict に発散1fitあり（平均汚染、summary参照）。従来 masked evaluation の楽観を定量化 |
| Poisson誤指定（人工NB-Y） | r_true∈{2,5,20,∞} で Poisson/NB oracle/NB moment 比較（strict held-out + RMSE(Z)） | `tools/overdispersion/run_poisson_misspecification_check.py` | `expfam/results/overdispersion/poisson_misspecification_{summary,agg}.csv` | `figures/overdispersion/poisson_misspec_{rmse_z,heldout_ll}.*` | current_support | ✗ | 生成は gamma-Poisson 混合（正確な NB2） |
| MovieLens 共有Z ablation | Proposed/fix_x(Y-only)/fix_w(X-only) の strict held-out 比較 | `tools/shared_z_ablation/run_movielens_shared_z_ablation.py` | `expfam/results/shared_z_ablation/movielens_shared_z_ablation_{summary,agg}.csv` | `figures/shared_z_ablation/movielens_shared_z_ablation.*` | current_support | ✗ | NMI はフルZ KMeans（既存 pilot の PCA→2D と手続きが異なるため絶対値比較不可） |
| 既存 ablation 棚卸し | 旧/fixed Exp4 + Wine の ablation を tidy CSV に集約（読取専用） | `tools/shared_z_ablation/audit_existing_ablation_results.py` | `expfam/results/shared_z_ablation/existing_ablation_audit.csv` | なし | current_support | ✗ | モデル再実行なし |
| ミスマッチ監査 | 41.5×/23.6×/3.41×/7.35× の根拠CSV・条件特定（KI-003解消）、fixed版対応表 | `tools/research_audit/audit_mismatch_experiments.py` | `expfam/results/mismatch_audit/mismatch_audit_{summary,old05_conditions}.csv` | なし | current_support | ✗ | 読取専用。41.5×=exp_scenario_C_exp4_mismatch.csv の XGauss/YPois 条件（41.45×）と特定 |
| per-column family デモ | 混在X（gauss/bern/pois 各3列）で per-column 正指定 vs 全列共通強制の比較 | `tools/research_audit/run_per_column_family_demo.py`, `expfam/src/experimental/model_dual_expfam_percolumn.py` | `expfam/results/per_column_family/per_column_demo_{summary,agg}.csv` | なし | current_support | ✗ | プロトタイプ。設計書 `reports/research_direction/per_column_family_design_20260708.md` |

---

## per-column family 検証フェーズ（2026-07-11、branch: research/per-column-validation、experimental 使用）

「別々に属性を回せばよいのでは？」（ゼミの問い）に答える検証フェーズ。
per-column は **prototype** の位置づけを維持。総括は
`reports/per_column_family/per_column_final_summary_20260711.md` を参照。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| per-column 数式監査 | 勾配/precision の数値微分照合（Y 3family × mask 有無）、列和構造、全列同一family≡既存モデル、scipy照合、ブロック重み診断 | `tools/research_audit/audit_per_column_math.py` | `expfam/results/per_column_family/per_column_math_audit_summary.csv` | なし | current_support | ✗ | 31/31 PASS。M-step Adam 収束性は未監査 |
| 単独 vs 同時統合 | 人工mixed-X（gauss/bern/pois各3列+Poisson-Y、strict held-out）で single_{g,b,p}/per_column_all/all_{g,b,p}/binarized/y_only の9条件×3seed | `tools/research_audit/run_per_column_single_vs_joint.py` | `expfam/results/per_column_family/single_vs_joint_{summary,agg,runinfo}.csv` | `figures/per_column_family/single_vs_joint_{rmse_z,heldout_ll}.*` | current_support | ✗ | all_* は比較用の誤指定モデル（変換は runinfo 明記）。joint 0.235 vs all_gaussian 0.234 は誤差内、all_bernoulli は1/3 trialで崩壊 |
| 属性追加 ablation | Y-only→+Bern→+Gauss→+Pois→+noise3 の5条件×3seed | `tools/research_audit/run_per_column_attribute_ablation.py` | `expfam/results/per_column_family/attribute_ablation_{summary,agg,runinfo}.csv` | `figures/per_column_family/attribute_ablation_lines.*` | current_support | ✗ | 改善は情報の濃い属性（+Gauss −22%）のみ。+Bern 無効果、+noise 無効果〜微悪化 |
| ノイズ属性チェック | informative9列+ノイズ（gauss3/6/12, bern3, pois3、全て正指定）の6条件×3seed | `tools/research_audit/run_per_column_noise_check.py` | `expfam/results/per_column_family/noise_check_{summary,agg,runinfo}.csv` | `figures/per_column_family/noise_check_lines.*` | current_support | ✗ | 改善なし。悪化は seed 依存（trial1 で +13%）。用量反応は3seedでは確定できず |
| MovieLens mixed-X pilot | genre19(Bern)+平均評価/公開年(z-score Gauss)+評価件数(Pois) の6条件×4fit、strict held-out | `tools/research_audit/run_movielens_mixed_x_percolumn.py` | `expfam/results/per_column_family/movielens_mixed_x_{summary,agg,runinfo}.csv` | `figures/per_column_family/movielens_mixed_x_test_y_ll.*`（`plot_per_column_figures.py` の `fig_movielens_mixed_x()`、2026-07-14 追加） | current_support | ✗ | **ネガティブ結果**: mixed_percolumn −3.815 < genre_only −3.423。X切片なしのため評価件数（平均154）のPoisson列がZ推定を強く支配した可能性がある。評価件数はYと同源（リーク懸念も記録）。切片/スケーリングが実用化の前提と特定 |

---

## story diagnostics フェーズ（2026-07-13、branch: research/story-diagnostics、experimental 使用）

per-column family 検証フェーズで残った2つの未検証仮説
（「人工データで改善幅が小さかったのは Y 側の情報が濃い設定だったからではないか」
「MovieLens で mixed_percolumn が悪化したのはどの属性・どの処理が原因か」）を、
仮説そのものを検証する追加診断実験として実施したフェーズ。
統括は `reports/story_diagnostics/story_diagnostics_summary_20260713.md` を参照。
per-column は引き続き **prototype**。complementary blocks 実験は未着手
（設計メモのみ `story_diagnostics_next_plan_20260713.md` に保持）。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| Y sparsity stress（smoke） | Y の学習観測率 y_obs_rate∈{1.0,0.5,0.2,0.1} × 4条件（y_only/single_gaussian/per_column_all/all_gaussian）、trials 少数の軽量確認 | `tools/research_audit/run_y_sparsity_stress.py` | `expfam/results/story_diagnostics/y_sparsity_stress_20260713_{,agg,runinfo}.csv` | `figures/story_diagnostics/y_sparsity_{rmse_z,test_y_ll}.png` | archive | ✗ | 動作確認用の予備実行。結論の根拠には trials10 版を使うこと |
| Y sparsity stress（trials=10） | 同上を trials=10（160 fits、NaN・発散なし）で再実行 | 同上（`TRIALS` 定数を変更） | `expfam/results/story_diagnostics/y_sparsity_stress_20260713_trials10{,_agg,_runinfo}.csv` | `figures/story_diagnostics/y_sparsity_{rmse_z,test_y_ll}_trials10.png` | current_support | ✗ | **ポジティブ結果**: y_obs_rate=0.1 で RMSE_Z per_column_all 0.343±0.026 < all_gaussian 0.769±0.133 < y_only 1.176±0.086。y_obs_rate=1.0 では差は小さい（0.221〜0.235）。**生成設定は1つのみ**（n=80, d=9, k*=2, gauss3+bern3+pois3, Poisson-Y）であり他設定への一般化は未確認 |
| MovieLens attribute diagnosis（smoke） | genre に属性を1つずつ足す11条件、count 処理3通り（raw Poisson / log-Gaussian / zscore-Gaussian）、1 fit/条件の軽量版 | `tools/research_audit/run_movielens_attribute_diagnosis.py` | `expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_smoke{,_agg,_runinfo}.csv` | `figures/story_diagnostics/movielens_{attribute_test_y_ll,count_treatment_comparison}_smoke.png` | archive | ✗ | 動作確認用の予備実行。結論の根拠には trials4 版を使うこと |
| MovieLens attribute diagnosis（trials=4） | 同上を 4 fits/条件（44 fits、NaN・発散なし）で再実行 | 同上 | `expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_trials4{,_agg,_runinfo}.csv` | `figures/story_diagnostics/movielens_{attribute_test_y_ll,count_treatment_comparison}_trials4.png` | current_support | ✗ | **診断結果**: test_y_ll は genre_only −3.417±0.017 に対し genre+count(raw,Poisson) −3.792±0.057、count を log/z-score で Gaussian 化すると −3.415/−3.416（差 +0.002、std 内）。mixed_percolumn の悪化はほぼ count 列1つの扱いで説明できる。⚠ **リーク注意**: mean_rating・ratings_count は pair split 前に全 u.data から計算され Y と同源。厳密な汎化性能の証拠ではなく**悪化要因の切り分け診断**として扱う。「log count 版が genre_only を上回った」とは言わない（差 +0.003 < std 0.014） |

**このフェーズの結論（`story_diagnostics_summary_20260713.md` §6）:** 属性ごとに分布を変えるだけでは不十分。
ただし Y が疎な人工条件では有効性が見えた。MovieLens の悪化要因は count 属性の扱いと特定。
次フェーズの実装優先項目は **X側切片・スケーリング・属性ブロック重み** の3点。

---

## 理論監査フェーズ（2026-07-18〜19、read-only）

実験ではなくリポジトリ内のコード・CSV・文書に対する読み取り専用の理論監査。
モデル変更・再学習は一切行っていない。

| 成果物 | 内容 | 参照 |
|---|---|---|
| `reports/theory_audit/theory_audit_report_20260718.md` | 本体。同時分布の確定、1/2 不要の導出、旧0.5系列の tempering 解釈、MATLAB `calcGrad`/`calcAi` の不整合、現行 BIC が Schwarz BIC ではなく ICL 型（完全データ型）基準であること、識別可能性が O(k) に一致すること、修論スコープ推奨 | KI-010 |
| `reports/theory_audit/math_code_correspondence_20260718.md` | 数式とコードの対応表 | 同上 |
| `reports/theory_audit/fix_and_experiment_plan_20260718.md` | 修正・実験計画 | `diagnostic_designs_20260719.md` の実験計画2 が参照 |
| `docs/theory_audit/CLAUDE_FABLE_5_THEORY_AUDIT_MASTER_PROMPT.md` | 上記監査の実施に用いたマスタープロンプト（来歴記録）。2026-08-17 に root から移動 | 報告書冒頭 |
| `reports/theory_audit/diagnostic_designs_20260719.md` | 上記監査の再検証と診断設計（すでに main 収録済み） | — |
| `docs/math_notes/half_factor_primary_source_confirmation_20260818.md` | **primary-source confirmation evidence（2026-08-18）。** 先行研究 `paper/A_study_on_latent_structural_models_for_binary_rel.pdf` の印刷式に 1/2 が存在すること（Eq.19/20/22/23、Appendix A-1/A-3/A-5）を研究者本人が直接確認した記録。1/2 の所在を5系統に整理。2026-07-18 の理論監査で `[UNRESOLVED]` としていた点を後から解消したものであり、当時の監査記録の本文は historical record として不変のまま保持する | KI-001 |

---

## クリーンアップ監査（2026-07-07、read-only、dry-run）

削除・移動・リネームを一切行わない dry-run の棚卸し。実行結果ではなく**候補一覧**である点に注意。

| 成果物 | 内容 | 注意 |
|---|---|---|
| `tools/cleanup_audit.py` | ツリーを走査し cleanup 候補を CSV 出力する dry-run スクリプト | 削除・移動・編集は行わない（docstring に明記） |
| `reports/cleanup_audit/cleanup_candidates_20260707.csv` | 753行。KEEP 597 / ARCHIVE_CANDIDATE 131 / DELETE_CANDIDATE 13 / REVIEW_REQUIRED 12 | 自動分類であり**実行判断ではない** |
| `reports/cleanup_audit/cleanup_review_20260707.md` | 上記CSVの人間レビュー用サマリ | DELETE_CANDIDATE 13件は主に `__pycache__` 系 |

---

## パス表記の forward correction（2026-08-21、Phase 5a.1 / issue #19、append-only）

上の各表の **historical row は一切書き換えていない**（rewrite 0 件・delete 0 件）。
Phase 5a のパス参照検査（`tools/validate_registry_paths.py`）で検出された表記の誤り・不整合は、
KI-009 および `docs/math_notes/half_factor_primary_source_confirmation_20260818.md` の行と同じ扱いとする。
すなわち **原文はそのまま historical record として残し、この節に日付入りの forward correction を追記する**。
当時の記録は当時のまま読み、**現行の正しい参照はこの節を正とする**。
実験の数値・結論・状態・原稿採用列はいずれも変更していない（表記のみの訂正である）。

| # | 対象の historical row | 原文（historical record。変更しない） | 現行の正しい参照 | 訂正理由 | validator 分類（原文 → 訂正後） |
|---|---|---|---|---|---|
| A | story diagnostics フェーズの「Y sparsity stress（smoke）」行（状態 `archive` / 原稿採用 ✗） | `expfam/results/story_diagnostics/y_sparsity_stress_20260713_{,agg,runinfo}.csv` | `expfam/results/story_diagnostics/y_sparsity_stress_20260713{,_agg,_runinfo}.csv` | brace 記法の誤り。アンダースコアが brace 群の外側に置かれているため、空 alternative が実在しない `y_sparsity_stress_20260713_.csv` に展開される。同フェーズの trials=10 行はすでに正しい記法で書かれている | KNOWN_NOTATION_DEFECT → PATTERN_RESOLVED |
| C | 「Control比較」行（状態 `current_main` / 原稿採用 ✓） | 実装/スクリプト cell の bare basename `run_comparison_all.py`（同 cell の先行参照は `reproduction/src/experiment_compare_with_dual.py`） | `reproduction/scripts/run_comparison_all.py` | 同 cell の先行参照から reproduction/src/ 配下と読めてしまうが、実ファイルは reproduction/scripts/ 配下にある（同名ファイルはリポジトリ内に1件のみ、2026-08-21 確認）。historical row の bare basename は残し、validator も同名ファイルをリポジトリ全体から検索して推測することはしない（bare basename は UNRESOLVED のまま） | UNRESOLVED → EXISTS_LITERAL |

**A が指す実 artifact は次の3件**（いずれも 2026-08-21 時点で実在）:

- `expfam/results/story_diagnostics/y_sparsity_stress_20260713.csv`
- `expfam/results/story_diagnostics/y_sparsity_stress_20260713_agg.csv`
- `expfam/results/story_diagnostics/y_sparsity_stress_20260713_runinfo.csv`

A の waiver 本体は `tools/validate_registry_paths.py` 内の定数として持たれており、
validator はその定数側の「訂正後の参照」を毎回ワーキングツリーに対して解決する（この .md を読みに行くわけではない）。
両者が食い違わないよう、**定数の原文トークンと訂正後参照がこの文書に実際に書かれていることを self-test で照合している**
（照合対象は文書全体である。原文トークンは historical row にも現れるため、実質的にこの節を守っているのは訂正後参照のほうである）。
waiver は（source document, 原文トークン）の完全一致でのみ効き、訂正後の参照が解決しなくなった時点で失効する。
したがってこの節は原文を隠す免罪符ではなく、**原文の誤りを可視化したまま非ブロッキングにするための記録**である。

なお `.gitignore` により Git 管理外とされている実験成果物への参照は、
研究用ワークステーションには存在し fresh checkout には存在しない。
これらは `LOCAL_ONLY_ARTIFACT` として非ブロッキングに分類され、
**存在確認済み（`EXISTS_LITERAL`）とは別にカウントされる**。
**分類自体**はワークステーションでも fresh checkout でも変わらない。
一方で存在に由来するフィールド（`local_presence`・`matches`・`resolved_via`）とそこから計算される集計値は環境で変わる。

**重要**: この分類は拡張子からの推測では決まらない。
`tools/validate_registry_paths.py` の `LOCAL_ONLY_ARTIFACT_REFERENCES` に
（source document, 原文トークン, 想定される repository-relative path/pattern, 想定される .gitignore 規則）
を明示登録した参照だけが対象であり、登録は KNOWN_NOTATION_DEFECT と同じ完全一致で効く。
`.gitignore` は登録を**裏付ける証拠**であって、登録を**作り出す権限ではない**。
したがって未登録の参照は、拡張子が無視対象であっても通常どおり判定される
（例: expfam/results/ 配下に wine_typoooo.npy のような打ち間違いを書いても TRUE_BROKEN のままである。
1文字違いも、別文書に現れた同一トークンも、登録を継承しない。
なおこの例をバッククォートで囲むと validator が実際に TRUE_BROKEN として検出するため、意図的に平文で書いている）。
登録が根拠とする .gitignore 規則が消えた場合、その登録は黙って残らず失効する。

2026-08-21 時点の登録は次の3件。
issue #19 Finding B が名指しした artifact は `expfam/data/movielens_pilot/*.npy` と
`expfam/results/wine_F.npy` の2つだが、後者は registry 内で
**bare basename 形式と明示パス形式の2通りで書かれている**。
ただし正確には、bare basename 形式は historical row（Wine実験の行）に元からあるもので、
**明示パス形式が現れるのは Phase 5a.1 で追加したこの節の中だけ**（すぐ上のこの段落と、下の登録表）であり、
historical row には現れない。
明示パス形式は issue #19 Finding B が用いている綴りでもあるため、
平文に落とさず登録して validator の検査対象に残す判断をした。
登録は raw token の完全一致で効き、ある綴りから別の綴りへ一般化しないため、
2つの綴りはそれぞれ別エントリとして登録する（したがって artifact 2件・登録3件）。
下表は self-test で機械的に検査している: 各行がちょうど1つの登録トークンを挙げること、
登録の集合と行が挙げるトークンの集合が一致すること、行の重複がないこと、
各行が対応する .gitignore 規則を明記していること。
「想定 path/pattern」列は 同左／同上 という相対表記のため機械比較の対象外であり、人手レビューで担保する。

| source document | 原文トークン | 想定 path/pattern | .gitignore 規則 |
|---|---|---|---|
| `EXPERIMENT_REGISTRY.md` | `expfam/data/movielens_pilot/*.npy` | 同左 | *.npy |
| `EXPERIMENT_REGISTRY.md` | `expfam/results/wine_F.npy`（明示パス形式） | 同左 | *.npy |
| `EXPERIMENT_REGISTRY.md` | 同 cell 先行の `expfam/results/wine_dual_results.csv` から rebase される `wine_F.npy`（bare basename 形式） | 同上 | *.npy |

新しい local-only 参照を registry に書く場合は、上記に登録するまで TRUE_BROKEN として報告される。
これは意図した失敗方向であり、**未登録の参照を黙って非ブロッキングにしない**ための設計である。
詳細な意味と限界は `tools/validate_registry_paths.py` の docstring を参照。

---

## complementary blocks 検証フェーズ（2026-08-21、issue #27、branch: experiment/27-complementary-blocks-consistent、objective-consistent numerics 使用）

Issue #28 / PR #29 の evidence-driven audit（`reports/model_refinement/evidence_driven_model_refinement_audit_20260821.md`）が
Issue #27 を RUN NEXT と判定したことを受けて実施した、事前登録済みの人工データ検証実験。
**objective-consistent lineage（Issue #25 / PR #26）を実験で使用した最初の実行**である。
per-column は引き続き **prototype** であり、本実験は修論の正式提案手法への昇格を意味しない。
実データに関する主張も行わない。総括は
`reports/story_diagnostics/complementary_blocks_consistent_report_20260821.md` を参照。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| complementary blocks（consistent, trials=10） | 属性 block ごとに異なる潜在次元を主に担う complementary-F 人工データ（n=80, d=9, K_TRUE=3, bern→z1 / gauss→z2 / pois→z3）で、`y_obs_rate ∈ {1.0 (dense negative control), 0.1 (sparse primary)}` × 6条件（`y_only` / `single_bernoulli` / `single_gaussian` / `single_poisson` / `per_column_all` / `all_gaussian`）を strict held-out で比較。全 fit `numerics_mode="consistent"`、generator clipping なし、120 fits | `tools/research_audit/run_complementary_blocks_consistent.py`, `expfam/src/experimental/model_dual_expfam_consistent.py` | `expfam/results/story_diagnostics/complementary_blocks_consistent_20260821_{summary,agg,paired,runinfo,generator,blockdiag}.csv` | `figures/story_diagnostics/complementary_blocks_consistent_20260821_{rmse_z,dimwise_rmse,test_y_ll}.png` | current_support | ✗ | **事前登録**: primary domain `y_obs_rate=0.1`、primary endpoint whole-space Procrustes RMSE_Z、primary contrasts は `per_column_all` vs `single_bernoulli`/`single_gaussian`/`single_poisson`/`y_only`、delta = comparator − per_column。**結果**: sparse-Y で4本とも per_column 優位（+0.512 / +0.422 / +0.389 / +0.203、10/10・10/10・10/10・9/10）、dense-Y では同方向だが約1桁縮小（+0.051 / +0.053 / +0.049 / **+0.009**）。**integrity**: 120/120 consistent、internal retry 0、NaN 0、q_bic_failed 0、warning 0、hash 整合 OK。**限定**: ①K_TRUE=3 で既存 sparse-Y 証拠（k\*=2）とは complementary 構造と潜在次元の2点が異なる ②block 間の local-curvature imbalance（gauss/bern 53.9×）は family だけの効果ではなく `sigma_G=0.3` を含む本 generator 設計との組合せ ③`all_gaussian` vs `per_column_all` は same-column misspecification contrast で M-step optimizer 経路が交絡 ④`single_*` vs joint は観測 X 列数も異なる ⑤Poisson X の marginal var > mean は latent heterogeneity であり overdispersion ではない ⑥prototype・実データ主張なし |

---

## matched latent-coverage ablation フェーズ（2026-08-21、issue #31、branch: experiment/31-matched-latent-coverage-ablation、objective-consistent numerics 使用）

Issue #27 / PR #30 の sparse-Y 結果（`per_column_all` が `single_gaussian` より +0.203 RMSE_Z、9/10）が
dimension-coverage 機構と整合的だったものの因果的に isolate されていなかったことを受け、
**latent-coverage / block-rank geometry をより厳密に狙った**事前登録済み ablation。
「alone」「fully isolated」とは主張しない（Bern/Pois の曲率は eta 依存、両者の true block trace は近似一致のみ、
有限標本 latent correlation が残る、joint model 自身の precision 幾何も regime 間で変化する）。
per-column は引き続き **prototype**、実データ主張なし。総括は
`reports/story_diagnostics/matched_latent_coverage_ablation_report_20260821.md` を参照。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| matched latent-coverage ablation（consistent, trials=10） | 同一 F 行を latent-coordinate permutation（`np.roll`, shift=(target−source)%K）した2 regime — `complementary`（bern→z1/gauss→z2/pois→z3）と `full_coverage`（各 family の row1→z1, row2→z2, row3→z3）— を、`y_obs_rate ∈ {1.0 (dense negative control), 0.1 (sparse primary)}` × 4条件（`single_bernoulli`/`single_gaussian`/`single_poisson`/`per_column_all`）＋ regime 非依存の共有 `y_only` で比較。n=80, d=9, K_TRUE=3, sigma_G=0.3, dominant=0.9/minor=0.15。X は **common-random-number inverse-CDF coupling**（Poisson は q=0 を `nextafter(0,1)` のみ置換、eta/mu clipping なし）、Gaussian は共有 `E_gauss` を source-array 同一性と hash で検証。全 fit `numerics_mode="consistent"`、**180 fits** | `tools/research_audit/run_matched_latent_coverage_ablation.py`, `expfam/src/experimental/model_dual_expfam_consistent.py` | `expfam/results/story_diagnostics/matched_latent_coverage_ablation_20260821_{summary,agg,paired,interaction,runinfo,generator,blockdiag}.csv` | `figures/story_diagnostics/matched_latent_coverage_ablation_20260821_{rmse_z,interaction,dimwise_rmse,coverage_spectrum}.png` | current_support | ✗ | **事前登録**: primary domain `y_obs_rate=0.1`、primary endpoint whole-space Procrustes RMSE_Z、primary comparator `single_gaussian`、**唯一の primary estimand** `I_t = delta_G(comp,t) − delta_G(full,t)`。**必須分解成分（co-primary ではない）** `D_G`/`D_J` と恒等式 `I = D_G − D_J`。**結果**: `delta_G` は +0.2280 (10/10) → +0.1141 (10/10)、**I = +0.1139（std 0.0915、median +0.1364、9/10）**、`D_G` = +0.3151 (10/10)、`D_J` = +0.2012 (10/10)、恒等式誤差 max 5.55e-17。dense-Y control は I = +0.0058 (6/10) と約1桁縮小。**secondary は primary に追随しない**: `single_bernoulli` の I = −0.2041（**0/10**）、`single_poisson` の I = −0.0046（6/10）。**integrity**: 180/180 consistent、internal retry 0、NaN 0、q_bic_failed 0、warning 0、hash 整合 OK、generator 15基準 + Gaussian 共有ノイズ provenance 全 PASS、Gaussian true block trace は両 regime で厳密一致（max abs err 2.13e-14、coverage_index 0.0034→0.2233、effective_rank 1.09→2.70）。**限定**: ①「alone / fully isolated」とは書かない ②I は単独解釈禁止・`D_J`=+0.201 と併記必須 ③操作は comparator の block rank も下げる（構成上の事実）④secondary の符号不一致は単一機構説明を支持しない ⑤block trace drift bern −0.240% / gauss +0.000% / pois −0.210%（rescale しない）⑥latent \|corr\| max 0.2943 ⑦K_TRUE=3（既存 sparse-Y 証拠は k\*=2）⑧Issue #27 との bitwise 再現は主張しない（sampling path が異なる／統計モデルは同一）⑨effective_rank は `Pbar_b` 固有値ベース定義に事前固定、3行ブロックで `coverage_index ≤ 1/3`・`effective_rank ≤ 3` ⑩Q ベース基準は条件間ランキングに不使用 ⑪prototype・実データ主張なし |

---

## MovieLens user-disjoint real-data validation フェーズ（2026-08-22、issue #33、branch: experiment/33-movielens-userdisjoint-validation、objective-consistent numerics 使用）

read-only 監査（main = e4be01af）で、既存 MovieLens pilot の 100 映画 subset が
**full-data popularity で選ばれていた**ことを一次確認した（`prepare_movielens_data.py:428` の
`rpm` = 全 u.data、`select_genre_stratified` L.161-167 が `rpm[m]` 降順、選択 100 本すべてが
full-data popularity の第73.6パーセンタイル以上、42/100 が上位10%）。したがって旧 subset を固定した
まま user split を行っても selection leakage が残るため、**movie selection を各 split の train users
のみから再実行**する設計とした。attribute（mean_rating / ratings_count）と z-score・AUC/AP 閾値の
leakage も同時に除去している。**category E（同一 users・同一 214 日窓に由来する依存）と
category F（表現仮説が同一 corpus の先行 diagnostic 由来）は残存し、開示のうえ凍結している。**
per-column は引き続き **prototype**、本文採用不可。総括は
`reports/real_data/movielens_userdisjoint_validation_report_20260822.md` を参照。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| MovieLens user-disjoint validation（consistent, 30 splits） | MovieLens 100K を **simple random user-disjoint split**（`default_rng(130000+s).permutation(943)` → 471 train / 471 test / 1 unused）で 30 回反復。各 split の **train users のみ**から genre-stratified に 100 映画（10 genre × 10 本、rate 閾値 30/943・200/943、fallback 20/943-300/943・10/943-500/943、再チューニングなし）を選択し、**train-only 由来属性**（`mean_rating_train`・`log_count_train` Gaussian、`count_train_raw` Poisson）と external metadata（genre19 Bernoulli、year Gaussian）で 6 条件（`y_only` / `genre_only` / `genre_year` / `genre_logcount_train` / `mixed_train_log` / `mixed_train_raw_poisson`）を比較。学習は `Y_train`（train users の共評価カウント、全 4950 ペア観測）、評価は `Y_test`（test users）上の **Poisson mean log score per pair**。K=3 固定・L=5・num_iter=8・2 model seeds、全 fit `numerics_mode="consistent"`、**360 fits** | `tools/research_audit/run_movielens_userdisjoint_validation.py`, `expfam/src/experimental/model_dual_expfam_consistent.py` | `expfam/results/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_{summary,agg,paired,splitdiag,provenance,runinfo}.csv` | `figures/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_{delta_stability,condition_ll,decomposition}.png` | current_support | ✗ | **事前登録**: 唯一の primary estimand `Delta_s = LL_test(mixed_train_log) − LL_test(genre_only)`、trial unit = user split（split 内 2 seeds を先に平均、**n=30**、60 ではない）、必須分解 `Delta = A + B`、descriptive positive control `P = genre_only − y_only`。**結果**: `Delta` mean **+0.004248**（median +0.006875、**23/30**、std 0.012276、std/\|mean\| **2.89**、empirical 10th-90th percentile [−0.009931, +0.016536]）、`A` +0.002009（22/30）、`B` +0.002239（19/30）、恒等式誤差 **0.000e+00**、positive control `P` +0.012437（25/30、spread ratio 1.07）。secondary: `mixed − y_only` +0.016685（28/30）、`genre_year − genre_only` −0.002222（16/30）、**`mixed_train_raw_poisson − mixed_train_log` −0.100274（1/30 = 29/30 で悪化）**。record-only rank corr(log_event_ratio, Delta) = −0.0234（`rankdata`+`corrcoef`、**p 値なし**）。**integrity**: 360/360 consistent、internal retry 0、NaN 0、q_bic_failed 0、warning 0、非有限 metric 0、重複キー 0、`test_n_pairs`=4950、fit-time ledger **360/360 valid**（`x_input==expected` / `y_input==y_train_hash` / `y_input!=y_test_hash`）、structural 210 行、L2 lint pass ×2、**L3 falsification 7 guards 全発火（checker が空虚でないことを実証）**、L4 独立参照の float 誤差 **4 block すべて 0.000e+00**（tol 1e-12、事後緩和なし）、train/test uid hash 30/30 distinct・重複 0。**限定**: ①primary は **Bernoulli/Gaussian heterogeneous-X** の評価であり、実データ上の Bernoulli/Gaussian/Poisson 同時統合は**検証していない**（条件6は diagnostic 専用）②表現仮説は同一 corpus の先行 leaky diagnostic を見た後に設定（category F、開示のうえ凍結）③**untouched external confirmation ではない** ④30 split は同一 943 users を再利用しており independent replicates ではない。**p 値・信頼区間・検出力・MDE・bootstrap を一切計算していない** ⑤category E 依存は残存（corr(count_full,count_train)=0.943、corr(mr_full,mr_train)=0.976）⑥movie node set は split 間で異なる（連続 split 重なり 70-85）⑦閾値定数 30/200 は 2026-06 full-data pilot 設計由来 ⑧Y は overdispersed（split 半分で var/mean 5.0-5.6、KI-012）。**Poisson LL は score であり正しく指定された尤度ではない** ⑨exposure mismatch は record only・split の drop/redraw/rebalance なし（最大 s=2 で log_event_ratio +0.1962）⑩K=3 は事前固定の設計定数、K 依存性は未確立 ⑪Q ベース基準（Schwarz BIC ではない、KI-010）は K 選択にも条件ランキングにも不使用 ⑫AUC/AP は算出せず、test 由来閾値なし、`mu_y` の rescale なし ⑬prototype・本文採用不可 ⑭事前登録からの逸脱は record-only diagnostic の符号規約（fit 実行前に修正）と console print label（実行後、計算値・CSV・図に影響なし）の 2 点のみ。報告書 §7 に記録 |

---

## K-selection score diagnostic pilot（2026-08-23、issue #37、objective-consistent prototype使用）

同一の42 fits上でQ-based complete-data criterionのlineageを分離し、C1 `bic_impl`、C2 `S_cf`、C3 `S_laplace_post`のcandidate Kに対する挙動を比較したdiagnostic pilot。objective-consistent実装はexperimental/prototypeであり、本文採用可能なモデルへの昇格、Schwarz BICの検証、K-selection consistency、paper Experiment 2 reproductionを意味しない。監査済み総括は `reports/k_selection_theory/k_selection_score_pilot_report_20260823.md` を参照。Issue #37のformal decisionは `C: DESIGN_HELDOUT_K_SELECTION_NEXT`。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| Phase 7b K-selection score pilot | objective-consistent prototypeでC1/C2/C3のK-selection diagnostic（n={75,150}、K_TRUE=3、k=1..7、3 trials、42 fits） | `tools/research_audit/run_k_selection_score_pilot.py` | `expfam/results/k_selection/k_selection_score_pilot_20260823_summary.csv`, `expfam/results/k_selection/k_selection_score_pilot_20260823_agg.csv`, `expfam/results/k_selection/k_selection_score_pilot_20260823_selection.csv`, `expfam/results/k_selection/k_selection_score_pilot_20260823_runinfo.csv` | `figures/k_selection/k_selection_score_pilot_20260823_score_curves.png`, `figures/k_selection/k_selection_score_pilot_20260823_log_det_A.png` | current_support | ✗ | objective-consistent prototype; diagnostic only; Issue #37; decision C。C1/C3はk=3 interiorを6/6、C2はk=7 range boundaryを6/6で選択。C2のstrict decreaseは5/6のみ。correct BIC / corrected BIC / ELBO / marginal-likelihood approximation / consistencyの主張は禁止 |


---

## Phase 7e full held-out K-selection pilot（2026-08-24、issue #43、objective-consistent prototype使用）

Phase 7c (#39) で設計し Phase 7d (#41) で実装・two-canary falsification した leakage-safe held-out K-selection protocol を、候補 K={1,...,7} と 3 dataset replicate へ拡張して exactly once 実行した full pilot。frozen 42-row manifest（replicate {1,2,3} × K {1..7} × start {1,2}）どおりに EM fit を 42 回だけ実行し、全 fit clean（internal retry 0 / warning 0 / Q failure 0 / NaN 0）。EM 開始前に 3 replicate すべての split を PAIR-MASK TOPOLOGY ONLY guard で検証し、score target は各 replicate の 14 clean fit 完了後に 1 回ずつ計 3 回だけ生成した。objective-consistent 実装は experimental/prototype であり、**本文採用不可**。K-selection consistency、BIC・C1/C2/C3 に対する優越、一般的な true-K recovery、実データ妥当性、漸近的性質は主張しない。監査済み総括は `reports/k_selection_theory/heldout_k_selection_full_pilot_report_20260824.md` を参照。Issue #43 の formal decision は `A: REPORT_HELDOUT_PILOT_TO_ADVISOR`。

| 実験ID | 内容 | 実装/スクリプト | 結果CSV | 図 | 状態 | 原稿採用 | 注意 |
|------|----|----------|-------|---|----|------|----|
| Phase 7e full held-out K-selection pilot | transductive dyad holdout（`test_ratio=0.20`）で held-out Bernoulli plug-in mean log score（raw `eta = w0 + w z_i^T z_j`、`y*eta - logaddexp(0,eta)`）による K 選択。family_x=poisson、family_y=bernoulli、K_TRUE=3、n=75、d=15、L=5、num_iter=8、`numerics_mode="consistent"`、candidate K=1..7、2 starts/K、3 dataset replicates、**42 fits** | `tools/research_audit/run_heldout_k_selection_pilot.py`（`--full --allow-em --confirm-full-pilot`）, `tools/research_audit/audit_heldout_full_pilot.py`, `tools/research_audit/build_heldout_full_pilot_report.py`, `expfam/src/experimental/model_dual_expfam_consistent.py` | `expfam/results/k_selection/heldout_full_pilot_20260824/{manifest,fit_results,replicate_selection,aggregate_summary,score_by_k}.csv`, `expfam/results/k_selection/heldout_full_pilot_20260824/runinfo.json` | （なし） | current_support | ✗ | objective-consistent prototype; Issue #43; decision A。RUN_CODE_SHA `b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a`、base main `a11406ca5e93c216bd4faa875fdbe0ca73c406c6`。**結果**: selected K = replicate1:3、replicate2:3、replicate3:5（tie candidates はいずれも単一）、selected-K counts {3:2, 5:1}、K_TRUE selected count 2、**descriptive pilot recovery rate 2/3 = 0.6667**。**integrity**: 42/42 clean、internal retry 0、warning 0、q_failure 0、NaN 0、非有限 0、duplicate/missing key 0、score target ちょうど 3、score rows 42。independent self-audit verdict PASS（BLOCKER 0 / HIGH 0 / MEDIUM 0 / LOW 0）、2-start mean score の runtime との差分 **0.0**、K 別集約の差分 1.73e-18（frozen tolerance 1e-12 未満、丸め順序差、選択結果に影響なし）。**限定**: ①replication unit は独立生成 dataset replicate で **わずか 3 個**、recovery rate は記述値であり信頼区間を伴う推定量ではない ②held-out dyad は node を共有し独立ではない。held-out pair 数（555）は独立標本サイズではない ③score は plug-in であり parameter・Z の不確実性を積分していない（posterior predictive / marginal likelihood / ELBO ではない）④候補 K のモデルは回転不定性を持ち操作上入れ子とは限らない ⑤MCEM 近似（L=5）と有限反復（num_iter=8）が予測ランキングに影響しうる ⑥transductive dyad holdout であり inductive / new-node 一般化（Design B）は現行 API 未サポート ⑦replicate3 の best-second margin は 0.000744 と小さいが、**margin は選択規則に一切使っていない。統計的有意差ではない** ⑧prototype・本文採用不可。**p 値・信頼区間・検出力・bootstrap は一切計算していない** |
