# Negative Binomial (NB2) 対応 設計・実装報告（Phase 4）

作成日: 2026-07-08
ブランチ: `research/overdispersion-z-ablation`
実装: `expfam/src/experimental/model_dual_expfam_nb.py`（`DualExpFamLSMNB`）

## 1. 数式上の整理

NB2（log link, dispersion r 固定）:

```
μ_ij = exp(η_ij),  η_ij = w0 + w z_i^T z_j
p(y|μ,r) = Γ(y+r)/(Γ(r) y!) · (μ/(μ+r))^y (r/(μ+r))^r
E[y] = μ,  Var[y] = μ + μ²/r      （r→∞ で Poisson）
```

| 量 | Poisson | NB2 |
|---|---|---|
| score ∂ln p/∂η | y − μ | (y − μ)·r/(μ+r) |
| 観測情報量 −∂²ln p/∂η² | μ | rμ(y+r)/(μ+r)² |
| **Fisher 情報量**（実装採用） | μ | **μr/(μ+r)** |

- E-step の Newton/Laplace（`_calc_precision_matrix`）には
  **Fisher 情報量（期待情報量）**を使う。常に非負なので精度行列の
  半正定値性が保たれ、観測情報量より安定（Fisher scoring 相当）。
- Fisher 重み μr/(μ+r) は μ→∞ で r に飽和する。Poisson（重み μ が非有界）
  で発生した大カウントペアによる Newton 不安定化を構造的に抑制する
  （MovieLens k=5 で Poisson が 1 fit 発散、NB は安定 — 実測 confirmed）。

### 指数型分布族との関係（重要な整理）

r **固定**の NB2 は自然パラメータ η' = ln(μ/(μ+r)) について 1 パラメータ
指数型分布族である。ただし本実装は既存実装と揃えるため log link
（η = ln μ、**非正準リンク**）を採用する。したがって
「score = T(y) − A'(η)」の正準形は成立せず、score に重み r/(μ+r) が付く。
これは本モデル枠組みが「正準リンクの ExpFam」から
「一般リンク + Fisher scoring」へ自然に拡張できることを示す実例であり、
修論の数理的貢献として記述できる（inference: 論文でのこの位置づけの
新規性は literature check required）。

r を**推定対象**にすると NB は指数型分布族の外に出る（2 パラメータ、
gammaln(y+r) が θ 依存）。本実装の割り切り: r は固定パラメータとし、
**モーメント推定**（Pearson 残差: r̂ = 1/mean(((y−μ̂)²−μ̂)/μ̂²)、
`eval_utils.moment_estimate_nb_r`）を two-stage で使う。
プロファイル尤度による r 推定（M-step に 1 次元最適化を追加）は今後の課題。

## 2. コード上の影響範囲（confirmed）

- 変更した既存ファイル: **なし**（すべて experimental 新規）
- `DualExpFamLSMNB` は `DualExpFamLSMMasked` を継承し、以下のみオーバーライド:
  `_variance_function`（Fisher 重み）、`_y_score_estep`、`_y_residual_mstep`
  （M-step calc_w0/calc_w の残差）、`calc_log_likelihood_Y`（全定数込み NB 尤度）
- 親クラスへは `family_y='poisson'` を渡して exp リンク・初期化を再利用。
  そのため `utils_expfam.calc_Q_dual_strict` / `calc_bic_dual` は
  **NB モデルに使用禁止**（Poisson 階乗補正が誤適用される）。
  代わりに `eval_utils.calc_Q_dual_strict_exp` / `calc_bic_exp`
  （NB の r を +1 パラメータとして数える）を使う。
- pair mask（strict held-out）と自由に併用可能。

## 3. テスト（confirmed、`test_experimental_models.py` 全 PASS）

- r=1e9 の NB が Poisson と score / Fisher 重み / 対数尤度で一致（退化検証）
- 真の μ を与えたモーメント推定が r=5 を r̂=4.95 で回復
- NB-Y 人工データ（n=40, r=5）で EM が NaN なく収束し
  w0=1.214（真値 1.2）、w=0.302（真値 0.3）を回復

## 4. Poisson との比較結果

- **MovieLens（実データ、strict held-out）**: 詳細は
  `movielens_overdispersion_diagnostics_20260708.md` §5。
  要約: te_ll 改善は小さい（k=3 で +0.02/pair）が、Poisson が発散した
  条件で NB は安定。本データは条件付き過分散が小さいため
  「NB が大きく勝つ」データではない。
- **人工過分散データ（r_true ∈ {2,5,20,∞}）**:
  `expfam/results/overdispersion/poisson_misspecification_*.csv`
  （run_poisson_misspecification_check.py）参照。過分散が強いほど
  Poisson の held-out 尤度が劣化し NB が回復するか、Z 推定への影響が
  あるかを定量化する（結果は統合レポート参照）。

## 5. 結論（NB を入れる価値・コスト・意味）

- **実装コスト**: 小（4 メソッドのオーバーライド、約 120 行）。
  既存 API 非破壊で experimental に分離できることを実証。
- **研究上の意味**: (1) 「3×3 の候補で足りるか」への具体的な拡張実例、
  (2) 正準リンク ExpFam → 一般リンク + Fisher scoring への枠組み拡張の実例、
  (3) 過分散の「保険」としての頑健化効果。
- **限界**: r 固定（two-stage）。r のプロファイル推定、zero-inflated 版、
  per-column NB-X は未実装。
