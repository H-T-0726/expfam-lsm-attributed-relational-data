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
