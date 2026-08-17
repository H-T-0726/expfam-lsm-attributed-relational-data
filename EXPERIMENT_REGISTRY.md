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

---

## クリーンアップ監査（2026-07-07、read-only、dry-run）

削除・移動・リネームを一切行わない dry-run の棚卸し。実行結果ではなく**候補一覧**である点に注意。

| 成果物 | 内容 | 注意 |
|---|---|---|
| `tools/cleanup_audit.py` | ツリーを走査し cleanup 候補を CSV 出力する dry-run スクリプト | 削除・移動・編集は行わない（docstring に明記） |
| `reports/cleanup_audit/cleanup_candidates_20260707.csv` | 753行。KEEP 597 / ARCHIVE_CANDIDATE 131 / DELETE_CANDIDATE 13 / REVIEW_REQUIRED 12 | 自動分類であり**実行判断ではない** |
| `reports/cleanup_audit/cleanup_review_20260707.md` | 上記CSVの人間レビュー用サマリ | DELETE_CANDIDATE 13件は主に `__pycache__` 系 |
