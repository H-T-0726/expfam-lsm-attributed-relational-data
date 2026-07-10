# Phase 0: 現状監査（過分散・共有Z・per-column family 研究フェーズ開始時点）

作成日: 2026-07-08
作成ブランチ: `research/overdispersion-z-ablation`（`main` 02311e7 から分岐）

## 0.1 git 状態（作業開始時）

- branch: `main` → 新規ブランチ `research/overdispersion-z-ablation` を作成して移行
- `git status --short -uall`（開始時）:
  ```
  ?? reports/cleanup_audit/cleanup_candidates_20260707.csv
  ?? reports/cleanup_audit/cleanup_review_20260707.md
  ?? tools/cleanup_audit.py
  ```
  （前フェーズの cleanup 監査の未追跡ファイル 3 件のみ。追跡ファイルへの変更なし。今回は触らない。）

## 0.2 実行環境（confirmed）

- Python 3.13.14 / numpy 2.3.5 / pandas 2.3.3 / scipy 1.16.3 / sklearn 1.8.0 / matplotlib 3.10.7
- MovieLens データ実在確認済み: `expfam/data/movielens_pilot/`（`movielens_X_genre.npy` (100,19), `movielens_Y_count.npy` (100,100), `movielens_Y_binary_t10/t20.npy`, `movielens_primary_genre_labels.npy`, `movielens_movie_ids.npy`, `movielens_movies_metadata.csv`）および raw `expfam/data/ml-100k.zip`

## 0.3 実装の現状（confirmed、コード読解に基づく）

| 項目 | 状態 | 根拠 |
|---|---|---|
| モデル継承 | `LatentStructuralModel` → `ExpFamLatentStructuralModel` → `DualExpFamLSM` → `DualExpFamLSMFixed` | `expfam/src/model_*.py` |
| 分布族 | X/Y とも gaussian/bernoulli/poisson の 3 種のみ | `model_dual_expfam.py` VALID_FAMILIES (L.61) |
| family_x | **全列共通のスカラー文字列**（per-column 不可） | `model_dual_expfam.py` L.85, L.91-117 |
| pair mask / 欠測 | **未対応**。`calc_w0`/`calc_w`/`_calc_gradient`/`_calc_precision_matrix` にマスク引数なし | `run_fixed_real_movielens_heldout_count.py` L.5-31 のヘッダに明文化 |
| Negative Binomial | **未対応** | VALID_FAMILIES |
| Categorical | 未対応（KI-005） | 同上 |
| 有向 Y | 未対応（対称・上三角前提） | `utils_expfam.py` の triu 使用箇所多数 |
| 0.5 係数（KI-001） | 旧版 `model_dual_expfam.py` L.159/L.200 に残存。fixed 版 `model_dual_expfam_fixed.py` L.77/L.113 で E-step のみ除去 | 各ファイル読解 |
| fixed 版の残存 0.5 | `calc_w0`/`calc_w` の `grad_sum/(2.0*L*phi)`（`model_expfam.py` L.168/200）と `calc_log_likelihood_Y` の `0.5*np.sum(ln_p)`（L.267）は基底クラスのまま。**これは全 (i,j) 両方向和の対称性補正（=上三角和と等価）であり、E-step の spurious 0.5 とは意味が異なる**（inference; 本フェーズの実装ではこの規約を踏襲する） | `model_expfam.py` 読解 |
| ablation | `run_em_dual(fix_w=, fix_x=)` あり（fix_w: w=0 固定で Y 信号遮断、fix_x: F=0 固定で X 信号遮断） | `utils_expfam.py` L.420-421, L.480-488, L.519-522 |
| BIC | `calc_bic_dual`: num_params = k*d − k(k−1)/2 + [d if Gauss-X] + [1 if Gauss-Y]。w0,w は暗黙扱い（KI-010 未検証） | `utils_expfam.py` L.386-404 |
| Q_strict | Poisson X/Y の gammaln 補正あり。Gaussian X は ln2π 込み | `utils_expfam.py` L.355-379, `model_dual_expfam.py` L.318-323 |

## 0.4 実データ実験の現状（confirmed、既存 CSV に基づく）

- **MovieLens (poisson_pilot / heldout_count)**: n=100 映画, d=19 ジャンル (Bern-X), Y=共評価カウント (Pois-Y)。Y 上三角: mean=45.22, var=447.13, **var/mean=9.888**, zero率=0.000, max=144（本監査で再計算・KI-012 と一致）。評価は **masked evaluation（全ペアで学習→評価のみ分割）** であり strict held-out ではない。
- **Cora (balanced_k_sweep / heldout)**: n=280, 密度 0.011。BIC 最小 k=1 vs AUC/AP 最適 k=6 vs NMI/ARI 最適 k=3 の不一致（KI-011）。
- **Wine (wine_fixed_pilot)**: X+Y / Y-only AUC≈1.0、X-only AUC=0.500（Y がラベル由来のため）。

## 0.5 既知問題の本フェーズへの影響整理

| KI | 内容 | 本フェーズへの影響 |
|---|---|---|
| KI-001 (0.5係数) | 旧版 E-step に spurious 0.5 | **本フェーズの新実験はすべて fixed 版系列（`DualExpFamLSMFixed` とその派生）で統一**。旧版は使わない |
| KI-003 (41.5×根拠) | 根拠 CSV 未特定 | Phase 7 の mismatch 監査で CSV 全走査により確認する |
| KI-010 (BIC検算) | num_params 未検証 | NB 追加時に dispersion パラメータの数え方を明示。BIC は補助指標とし held-out を主指標にする |
| KI-011 (Cora BIC) | 疎ネットワークで BIC 破れ | 本フェーズではモデル選択を BIC に依存させず held-out 比較を主軸にする根拠 |
| KI-012 (過分散・mask) | var/mean≈10、pair mask 未対応 | **本フェーズの主対象**。Phase 1 で診断、Phase 2 で mask 実装、Phase 3-4 で Poisson vs NB |
| KI-002 (新旧混在) | 旧版と fixed 版の数値混在禁止 | 新規結果はすべて新ディレクトリ（overdispersion/ 等）に分離保存 |

## 0.6 出力先（新規作成、既存に上書きしない）

- スクリプト: `tools/overdispersion/`, `tools/shared_z_ablation/`, `tools/research_audit/`, `expfam/src/experimental/`
- 結果: `expfam/results/overdispersion/`, `expfam/results/shared_z_ablation/`, `expfam/results/mismatch_audit/`, `expfam/results/per_column_family/`
- 図: `figures/overdispersion/`, `figures/shared_z_ablation/`, `figures/mismatch_audit/`
- レポート: `reports/research_direction/`, `reports/overdispersion/`, `reports/shared_z_ablation/`, `reports/mismatch_audit/`

## 0.7 実行計画（フェーズ順序）

1. Phase 1: MovieLens 過分散診断（read-only 分析 → 新規 CSV/図）
2. Phase 7: 既存ミスマッチ実験の監査（read-only 分析 → 新規 CSV。安価なので先行実施）
3. Phase 2: pair mask 対応モデル（experimental 新クラス、既存 API 不変）
4. Phase 4: NB-Y モデル（experimental 新クラス、固定 dispersion）
5. Phase 3: Poisson 誤指定実験（人工過分散データ + MovieLens strict held-out）
6. Phase 5: 共有 Z ablation（MovieLens strict held-out での Proposed/fix_x/fix_w + 既存結果整理）
7. Phase 6: per-column family（設計書 + 可能なら最小プロトタイプ）
8. Phase 8: 統合レポート

順序変更の理由: Phase 7（監査）は read-only で他に依存しないため先行。Phase 4（NB）を Phase 3（誤指定実験）の前に置くのは、誤指定実験の比較群（NB oracle）が NB 実装を要するため。
