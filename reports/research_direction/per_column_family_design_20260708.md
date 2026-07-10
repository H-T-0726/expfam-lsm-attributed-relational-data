# per-column family 設計書 + 最小プロトタイプ実装報告

作成日: 2026-07-08
ブランチ: `research/overdispersion-z-ablation`

## 1. 問題（confirmed）

現行実装の `family_x` は**全列共通のスカラー文字列**である
（`expfam/src/model_dual_expfam.py` L.85、`_mean_function_x` L.91-102 が
単一分岐）。したがって「年齢=Gaussian、会員フラグ=Bernoulli、購買回数=Poisson」
のような混在属性の実データを正しく扱えない。MovieLens 案B（user-node、
`reports/movielens_pilot_design.md`）が deferred になった一因もここにある。

## 2. 数理的整理

列 l の分布族を c(l) ∈ {gaussian, bernoulli, poisson} とすると、
X 側対数尤度は列ごとの和に分解される:

```
ln p(x_i | z_i, F) = Σ_l [ T_{c(l)}(x_il) η_il − A_{c(l)}(η_il) ] / φ_{c(l)} + const
η_il = f_l^T z_i
```

列独立なので E-step / M-step の変更は**列単位の重み付けだけ**で済む:

- **E-step 勾配 Term2**: `F^T [ w ⊙ (T(x_i) − A'_{c}(F z_i)) ]`、
  `w_l = 1/σ_l²`（Gaussian 列）、`1`（それ以外）
- **E-step 曲率 Term2**: `F^T diag[c_l] F`、
  `c_l = 1/σ_l²`（Gaussian）、`A''_{c(l)}(η_l)`（Bernoulli/Poisson）
- **M-step F**: 全列 Gaussian → 閉形式（現行どおり）。混在 → 重み付き Adam
  （現行 `_calc_F_adam` は Gaussian 列の 1/σ² 重みを持たないため、
  混在時は σ_l=1 相当になってしまう点に注意 — プロトタイプでは
  `_calc_F_adam_weighted` として補正済み）
- **M-step Σ**: Gaussian 列のみ MLE、他列は 1 固定
- **Q_strict**: Poisson 列のみ `−Σ ln(x_il!)` 補正、Gaussian 列のみ `ln 2π` 項
- **BIC num_params**: Σ の自由度 = **Gaussian 列数**（d ではない）

## 3. 影響範囲（現行コードベース）

| 箇所 | 現行 | 必要な変更 | プロトタイプでの対応 |
|---|---|---|---|
| `model_dual_expfam.py` `_mean_function_x`/`_variance_function_x` | 単一分岐 | 列マスク別適用 | ✓ `model_dual_expfam_percolumn.py` |
| `_calc_gradient`/`_calc_precision_matrix` Term2 | gaussian/other の2分岐 | 列重みベクトル | ✓ 同上 |
| `calc_F` | gaussian→閉形式 / else Adam | 混在→重み付き Adam | ✓ `_calc_F_adam_weighted` |
| `calc_sigma` | gaussian→全列 MLE / else I | Gaussian 列のみ MLE | ✓ |
| `calc_log_likelihood_X` | 単一分岐 | 列グループ和 | ✓ |
| `utils_expfam.calc_Q_dual_strict` | `family_x=='poisson'` で全列階乗補正 | Poisson 列のみ | ✓ `eval_utils.calc_Q_dual_strict_exp`（'mixed' 対応） |
| `utils_expfam.calc_bic_dual` | Σ=d if gaussian | Σ=Gaussian 列数 | ✓ `eval_utils.calc_bic_exp(n_gaussian_x_cols=)` |
| `data_generator_expfam.generate_dual_data` | 単一 family_x | 列別生成 | △ テスト内 `generate_mixed_x_data` のみ（正式生成器は未対応） |
| `run_em_dual` | family_x スカラー | family_x_list 引数 | ✓ `em_runner.run_em_experimental(family_x_list=)` |
| CSV 出力列 | `family_x` 1列 | family_x_list の文字列化 | 実験スクリプト側で対応要 |
| NB-Y との併用 | — | 未実装（NotImplementedError を明示） | ✗ 今後 |
| Categorical 列 | — | 未実装（KI-005 と同根） | ✗ 今後 |

## 4. プロトタイプ実装（confirmed、本フェーズで作成）

- `expfam/src/experimental/model_dual_expfam_percolumn.py`
  （`DualExpFamLSMPerColumn`、fixed 系列 + pair mask 継承）
- テスト `expfam/src/experimental/test_percolumn_model.py` — 全 PASS:
  1. 全列同一 family の per-column == スカラー family モデル（勾配・曲率・尤度が数値一致）
  2. `calc_sigma` が Gaussian 列のみ推定
  3. 混在 X（gauss 3 + bern 3 + pois 3, n=60, k=2）の EM smoke:
     rmse_Z=0.227、w0=1.202（真値 1.2）、w=0.290（真値 0.3）
- デモ実験 `tools/research_audit/run_per_column_family_demo.py`:
  混在データに対し per-column（正指定）vs 全列 gaussian/bernoulli/poisson 強制
  （現行制約の模擬）を比較。結果は
  `expfam/results/per_column_family/per_column_demo_agg.csv`。

## 5. 修論での位置づけ

- **主張できること**: 「実データの混在属性では全列共通 family では不十分であり、
  per-column 化は列単位の重み付けとして既存の MCEM+Laplace 枠に自然に入る。
  最小プロトタイプで混在データの Z 回復を確認した」
- **まだ主張できないこと**: 実データ（MovieLens 案B user-node 等）での有効性、
  列ごとの family を**データから選ぶ**手続き（選択問題は per-column 化で
  組合せ爆発する — d 列 × 3 族。列単位の周辺診断による事前絞り込みが現実的）
- **推奨する進め方**: (1) 本設計書 + プロトタイプを修論「拡張」章の一部に、
  (2) user-node MovieLens（年齢 Gaussian / 性別・職業 Bernoulli 化 /
  評価数 Poisson）への適用を追加実験として計画
