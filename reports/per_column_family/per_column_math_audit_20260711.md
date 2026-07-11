# per-column family 数式監査レポート

作成日: 2026-07-11
ブランチ: `research/per-column-validation`
監査対象: `expfam/src/experimental/model_dual_expfam_percolumn.py`（`DualExpFamLSMPerColumn`、**prototype**）
監査スクリプト: `tools/research_audit/audit_per_column_math.py`
結果 CSV: `expfam/results/per_column_family/per_column_math_audit_summary.csv`

## 結論

**全 31 チェック PASS（FAIL 0）。** per-column family の尤度・勾配・precision
contribution は、実装から独立に書き下した式および数値微分と一致し、
全列同一 family のときは既存のスカラー family モデル（`DualExpFamLSMMasked`）と
数値的に同一（差 0.0）であることを確認した。

## 監査した数理モデル（コードから確定した形）

属性列 l の family を c(l) ∈ {gaussian, bernoulli, poisson} として

```
x_il | z_i ~ ExpFam_{c(l)}( η_il ),   η_il = f_l^T z_i   （切片 μ_l なし）

log p(x_i | z_i) = Σ_l [ x_il η_il − A_{c(l)}(η_il) ] / φ_l + const
    A(η) = log(1+e^η)  (Bernoulli, φ=1)
    A(η) = e^η          (Poisson,   φ=1)
    A(η) = η²/2         (Gaussian,  φ_l = σ_l² を M-step で MLE 推定)
```

- **Gaussian のパラメータ化**: mean link `x ~ N(η, σ_l²)`。Gaussian では
  canonical link = identity なので mean link と canonical link は同一。
  分散は Gaussian 列のみ列ごとの σ_l² を推定し、他 family の列は φ=1 固定
  （`calc_sigma` が Gaussian 列のみ更新することもテスト済み）。
- **切片なし**: η_il = f_l^T z_i のみで、列ごとの切片 μ_l は存在しない
  （先行研究の X モデル eq(2) と同じ規約）。したがって平均が 0 から遠い
  Gaussian 属性は中心化が必要、平均の大きい Poisson 属性（例: 生の評価件数）は
  η を F と Z だけで持ち上げる必要がある。**実データで mixed-X を使う際の
  実務上の注意点**としてレポート全体で引き継ぐ。
- **E-step（Z 推定）への X 側寄与**（Laplace 近似の勾配・precision）:

```
gradient_X(z_i)  = Σ_l w_l ( x_il − A'_{c(l)}(η_il) ) f_l      w_l = 1/σ_l² (Gauss) / 1 (他)
precision_X(z_i) = Σ_l w_l A''_{c(l)}(η_il) f_l f_l^T
```

- **Y 側**: fixed 系列規約（`Σ_{j≠i}`、1/2 なし）+ pair mask
  （train_mask=False のペアは勾配・precision・尤度から除外）。

## チェック項目と結果

| # | チェック | 方法 | 結果 |
|---|---|---|---|
| A | family 基本形 A′, A″ | A′≈FD[A], A″≈FD[A′]（η∈[−3,3]） | 6/6 PASS（≤3e-9） |
| B | `_calc_gradient` | 独立実装の −ln f(z_i\|X,Y) の中心差分と比較。Y∈{poisson, bernoulli, gaussian} × mask{なし, 70%} | 6/6 PASS（≤2.8e-9） |
| C | `_calc_precision_matrix` | `_calc_gradient` の数値ヤコビアンと比較（canonical link なので Hessian は厳密一致するはず） | 6/6 PASS（≤4.2e-10） |
| D | 列和構造 | Term2 を列ごとに素朴計算した Σ_l g_l / Σ_l P_l と比較 | 2/2 PASS（≤9e-16） |
| E | 既存モデル整合 | 全列同一 family の per-column ≡ scalar モデル（勾配・precision・llX）+ `_calc_F_adam_weighted`(重み全1) ≡ 親 `_calc_F_adam` | 10/10 PASS（全て 0.0） |
| F | 尤度 vs scipy | `calc_log_likelihood_X` を scipy.stats と照合（Poisson の −ln(x!) 省略規約を補正） | 1/1 PASS（2.8e-14） |
| G | ブロック重み診断 | ブロック別の列数・llX・E-step 勾配ノルム（情報記録のみ） | 3 INFO |

補足:
- チェック B は「列ごとに正しい family の A′ を使い、それを足し合わせて同じ z_i
  の推定に使う」形（`log p(x_i|z_i) = Σ_l log p_{g_l}(x_il|η_il)`）を監査スクリプト内で
  実装から独立に書き、その数値微分と実装勾配が一致することを確認したもの。
  Bernoulli 列で Bernoulli の A′/A″、Poisson 列で Poisson の A′/A″、Gaussian 列で
  1/σ² 重みが使われていることの直接検証になっている。
- Y 側 3 family × mask あり/なしで通ることから、strict held-out（pair mask）併用時の
  勾配・precision も正しい。

## ブロック重み診断（G、情報）

n=15, d=9（gauss/bern/pois 各 3 列、Gaussian 列 σ²=0.16/0.25/0.36）の監査データで:

| ブロック | 列数 | llX | E-step 勾配ノルム（ノード平均） |
|---|---|---|---|
| Gaussian | 3 | −25.4 | **1.56** |
| Bernoulli | 3 | −30.9 | **0.30** |
| Poisson | 3 | −41.5 | 1.06 |

- 列数が同じでも Z 推定への寄与（勾配の大きさ）はブロック間で数倍異なる。
  Gaussian ブロックは 1/σ² 重み（σ²<1 なら増幅）で大きく、Bernoulli は
  A″≤1/4 のため小さい。
- したがって「Bernoulli 列が多いから支配する」とは限らず、**支配するのは
  列数 × 列あたり情報量（曲率）**である。実データ（例: MovieLens で
  genre 19 列 vs 数値属性 1〜3 列）では列数の不均衡が効きうるため、
  ブロック別の llX・再構成誤差を実験でも記録する。
- block weighting（ブロック別重み付け）は**実装しない**。必要性が観察されたら
  「今後の課題」として記述するに留める（本フェーズの方針）。

## 実装上の数値ガード（監査範囲外だが把握しておくべき点）

- Poisson の η は [−20, 10] に clip、Bernoulli/Poisson の A″ は下限 1e-8、
  σ_l² は下限 1e-6〜1e-8。監査は clip 域に入らない小スケールで実施したため、
  clip 発動時の勾配は厳密な微分と一致しない（発散防止のための意図的な設計）。
- `all_bernoulli` 強制のような誤指定条件では x∉{0,1} に Bernoulli スコア
  x − sigmoid(η) を適用する quasi-likelihood 的な使われ方になる（数式としては
  well-defined だが、正しい確率モデルではない。実験レポートで「比較用の
  誤指定モデル」として扱う）。

## この監査で言えること・言えないこと

- **言える**: per-column prototype の E-step 勾配・precision・X 対数尤度は、
  設計書（`reports/research_direction/per_column_family_design_20260708.md`）の
  数式どおりに実装されており、全列同一 family では既存モデルと厳密に一致する。
- **言えない**: M-step（Adam）の収束性・EM 全体の統計的性質・実データでの有効性は
  本監査の範囲外（実験レポートで別途評価）。per-column family はあくまで
  **prototype** であり、正式手法としての完成度を保証するものではない。
