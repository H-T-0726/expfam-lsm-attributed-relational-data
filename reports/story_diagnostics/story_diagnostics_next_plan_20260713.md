# story diagnostics フェーズ — 次の実験設計メモ（未実装）

作成日: 2026-07-13
位置づけ: 実験1（Y sparsity stress test、軽量版）の結果（`y_sparsity_stress_report_20260713.md`）を踏まえ、次に着手するか判断するための設計メモ。**本メモの実験2・実験3は未実装**（コード・CSV・図は作成していない）。

実験1の暫定結果: Y観測率を下げると per_column_all / single_gaussian と y_only の差が拡大し、all_gaussian（誤指定）の悪化も顕著になる傾向が軽量予備実験（trials=2、4条件）で見られた。この傾向がフル条件・seed数拡大でも再現するかの確認が実験1の次のステップとして残っている。

---

## 実験2（未実装）：complementary blocks — 属性ブロックがZの別次元を持つ人工データ

**目的**: 現在の人工データ生成（`generate()`）は F を行正規化ランダムベクトルとして生成しており、各属性ブロックが特定の潜在次元に偏っていない（全ブロックが全次元に混ざって依存）。そのため、単独属性だけでもZをある程度推定できてしまい、複数属性統合の意義が見えにくかった可能性がある。属性ブロックごとにZの別々の次元へ主に依存する設計にすることで、統合の意義が出る条件を確認する。

**設計（案）**:
- K_TRUE=3。列lの属する主次元を one-hot ベクトルとし、`F_row = normalize(dominant_weight * e_主次元 + minor_weight * N(0,1)^3)`（dominant_weight≈0.9, minor_weight≈0.15 目安）。
- Bernoulli3列→主にz1、Gaussian3列→主にz2、Poisson3列→主にz3。Yは既存同様 `w0+w*z_i^T z_j`（全次元に依存）。
- 条件: `y_only, single_bernoulli, single_gaussian, single_poisson, per_column_all, all_gaussian, all_bernoulli, all_poisson`（8条件）。
- 指標: 全体RMSE_Z + **次元別RMSE**（`procrustes_rotation` が返す回転行列Rを`Z_est`に適用後、列ごとにRMSE算出。既存関数の追加利用のみで新規アライメントロジック不要）。
- 想定スクリプト名: `tools/research_audit/run_complementary_blocks.py`
- 想定出力: `expfam/results/story_diagnostics/complementary_blocks_20260713.csv`（+`_agg.csv`, `_runinfo.csv`）、`figures/story_diagnostics/complementary_blocks_{rmse_z,dimwise_rmse}.png`、`reports/story_diagnostics/complementary_blocks_report_YYYYMMDD.md`

## 実験3（未実装）：MovieLens属性追加・count処理診断

**目的**: MovieLens pilotで mixed_percolumn が genre_only より悪化した要因を切り分ける。特にratings_countの扱い（生値Poisson vs log変換Gaussian vs z-score Gaussian）が悪化にどう寄与するかを確認する。

**設計（案）**:
- 既存 `run_movielens_mixed_x_percolumn.py` の `build_attributes()` 相当ロジックを複製・拡張（既存スクリプトは変更しない）。追加変換: `log_count_z`（`log1p(ratings_count)` をz-score）、`count_z`（生カウントをz-score）。
- 条件（案、13条件）: `y_only, genre_only, genre+year, genre+avg_rating, genre+rating_count_raw_poisson, genre+log_count_gaussian, genre+zscore_count_gaussian, genre+avg_rating+year, genre+avg_rating+rating_count+year, rating_stats_only, mixed_all_gaussian, mixed_percolumn_raw, mixed_percolumn_log_count`。
- **リーク注意**: `mean_rating`/`ratings_count`は既存同様、train/test split前の全`u.data`から計算する（train-only化は現時点で実装しない。pair-levelのsplitからuser-levelの完全な train-only 統計を作るには大幅な再設計が必要なため）。レポートで強く注意喚起する方針を維持する。
- 想定スクリプト名: `tools/research_audit/run_movielens_attribute_diagnosis.py`
- 想定出力: `expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713.csv`（+`_agg.csv`, `_runinfo.csv`、`leak_caveat`列必須）、`figures/story_diagnostics/movielens_{attribute_test_y_ll,count_treatment_comparison,block_diagnostics}.png`、`reports/story_diagnostics/movielens_attribute_diagnosis_report_YYYYMMDD.md`

## 全体レポート（未実装）

実験1〜3が出揃った段階で `reports/story_diagnostics/story_diagnostics_summary_YYYYMMDD.md` を作成し、以下を含める: 問題意識／実験1結論／実験2結論／実験3結論／言ってよい・言いすぎな主張／次の改善案（X側切片・count属性変換・属性ブロック重み・block-wise diagnostics・train-only metadata）。

## 判断待ちの事項

- 実験1のフル条件（`single_bernoulli`, `single_poisson`, `all_bernoulli`追加）・trial数拡大（3〜5程度）を先に行うか、実験2・3に進むかはユーザ判断待ち。
- `EXPERIMENT_REGISTRY.md` への追記も、どの実験を正式に採用するか決まってから行う。
