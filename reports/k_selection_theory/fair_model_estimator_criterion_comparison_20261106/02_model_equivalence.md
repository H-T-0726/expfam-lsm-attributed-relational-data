# 02 モデルの数学的同等性（Phase A、2026-11-06）

一次資料：原論文 PDF（§2.2 Eq.(1)(2)、§3 Eq.(16)〜(23)、§4.1）、`expfam/src/experimental/model_dual_expfam_consistent.py` とその親クラス、`reproduction/src/model.py`。数値確認：`expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106/equivalence_numeric_checks.csv`（`tools/research_audit/fair_model_equivalence.py`、テスト `test_fair_common_evaluator.py` 3/3 PASS）。機械可読の判定は `model_equivalence_matrix.csv`。

## 1. 原論文のモデル（FACT、一次資料）

- z_i ~ N(0, σ_z² I_k)（§2.2）。σ_z² はパラメータで、Eq.(14) で更新される。人工データは σ_z² = 1（§4.1 Step 1）。
- x_i ~ N(F z_i + μ_x, Σ)、Σ = diag(σ_1², …, σ_d²)（Eq.(1)）。**切片 μ_x がある。** 人工データは μ_x = 0（§4.1）。
- y_ij ~ Bern(s_ij)、s_ij = 1/(1 + exp(−w0 − w z_iᵀz_j))（Eq.(2)）。y_ii = 1 と約束する（§2.2）。
- 尤度 Eq.(16)：∏_i p(x_i | …) · {∏_{i≠j} p(y_ij | …)}^{1/2}。
- θ = {F, μ_x, Σ, σ_z², w0, w}（§3）。識別制約（回転など）の記述はない。BIC の罰則 (k+1)d − k(k−1)/2 は回転の自由度を引いている（Eq.(26)）。

## 2. 指数型分布族モデル（Gaussian-X/Bernoulli-Y を指定した場合、FACT）

- z_i ~ N(0, var_z I_K)。var_z = 1 に固定し、E-step のあとに `scale_Z` で標本の平均二乗を 1 にそろえる。
- x_il | z_i ~ N(f_lᵀ z_i, Σ_ll)。**切片なし**（AGENTS.md §1「バイアスなし」）。Gaussian の dispersion は Σ の対角。
- y_ij | z ~ Bern(sigmoid(w0 + w z_iᵀz_j))、i < j。マスクで指定したペアだけを使う（canonical の実験は 80%、E1 は全ペア）。対角は使わない。

## 3. 項目ごとの比較

| 項目 | 原論文 | ExpFam（Gauss/Bern） | 判定 |
|---|---|---|---|
| Z の事前分布 | N(0, σ_z² I)、σ_z² 推定 | N(0, I)、σ_z² = 1 固定＋`scale_Z` | 違う（σ_z² の扱い）。σ_z² = 1 の部分モデルとしては同じ |
| X の条件付き分布 | N(Fz + μ_x, Σ) | N(Fz, Σ) | μ_x = 0 なら同じ。ExpFam には切片がない |
| Y の条件付き分布 | Bern(sigmoid(w0 + w zᵀz)) | 同じ | 同じ |
| F | d×k | d×K | 同じ |
| w, w0 | スカラー | スカラー | 同じ |
| Gaussian の分散 | Σ 対角（推定） | Σ 対角（推定） | 同じ |
| 無向ペアの数え方 | {∏_{i≠j}}^{1/2} | i < j の積 | **対称な Y では同じ値**（数値確認：差 0） |
| Y の対角 | y_ii = 1 と約束（尤度では i≠j のみ） | 使わない | 尤度に影響なし |
| 観測マスク | なし | あり（実験による） | 全ペアなら同じ |
| 回転不変性 | 記述なし（罰則で考慮） | (Z, F) → (ZR, FR) で尤度不変 | 同じ（数値確認：差 0） |
| 正規化 | 本文は記載なし（MATLAB・Python は Z・X を標本 z-score） | canonical 生成器は正規化しない | **生成器の違い**（モデルの違いではない） |
| K の包含関係 | 記述なし | M_K ⊂ M_{K+1}（F の列を 0、Z の 1 列を自由に） | 同じ（Gaussian-Y でない限り。`true_k_identifiability_hardened_20260904.md` P3） |

## 4. 数値による確認（OBSERVED、n=9、d=4、K=2 の固定例）

| 確認 | 値 |
|---|---:|
| Gaussian-X：論文の N(Fz, Σ) の対数密度 − ExpFam 形式（T(x)=x、η=fᵀz、φ=Σ_ll） | 0.0 |
| Bernoulli-Y：論文 Eq.(16)（順序対の和の 1/2） − 無向ペアの和 | 0.0 |
| (ZR, FR) への回転で完全データ対数密度が変わらない | 0.0 |
| 系列 E の strict Q − 共通評価器 | 0.0（Gaussian-X・Poisson-X とも） |
| Python 再現の `calc_log_likelihood` − 共通評価器 | 0.0 |
| 共通評価器の勾配・負の Hessian と有限差分の相対誤差 | 1.7e−9・2.3e−8（Gaussian）、3.9e−9・2.2e−7（Poisson） |

**発見（FACT）**：系列 E の `eval_utils.calc_Q_dual_strict_exp` は、Gaussian-X の項で引数の σ を使わず、`model.params['sigma']` を読む。2 つが違うと −5.9 の差が出た。推定中は両者が同じ値なので既存の結果には影響しないが、評価器としては潜在的な落とし穴である。本研究の評価はすべて、推定器から独立した共通評価器（`fair_common_evaluator.py`）で行う。

## 5. 判定

| 層 | 判定 | 根拠 |
|---|---|---|
| **モデル同等** | **同等（条件付き）**。原論文のモデルで μ_x = 0、σ_z² = 1 としたものと、ExpFam の Gaussian-X/Bernoulli-Y（全ペア観測）は、同じ確率分布族を表す | §3・§4。違いは切片 μ_x と σ_z² の推定で、どちらも人工データの真値（μ_x = 0、σ_z² = 1）で消える（DERIVED） |
| **目的関数同等** | **同等**（同じパラメータ・同じ Z で、完全データ対数密度・条件付き尤度が一致） | §4 の差 0 |
| **実装同等** | **同等ではない** | E-step：Python 再現は Y の score と曲率に 1/2 が付き、有限差分との相対誤差は勾配 0.33、精度行列 0.093（03）。系列 E は有限差分と一致。M-step の更新式は同じ形。マスク・数値処理（安定な sigmoid）も違う |

- 原論文の印刷式 Eq.(22)・(23) の 1/2 は、Eq.(20) の z_i による微分では出てこない（各ペアは i の項と j の項の 2 回現れるので、1/2 と打ち消し合う）。採用式（1/2 なし）は Eq.(20) の z_i による微分と有限差分で一致する（DERIVED）。印刷式との差は、本研究の独立な再導出に基づく意図的な差として扱う（AGENTS.md §2、KI-001、`docs/math_notes/half_factor_primary_source_confirmation_20260818.md`）。印刷式の正誤を判定するものではない。
- **K0（母集団の最小次元）**：Gaussian-X は Σ が未知の対角なので、因子分析と同じ識別条件が必要（`true_k_identifiability_hardened_20260904.md` §6.3。d=15 なら Ledermann の上限は K ≤ 10）。Bernoulli-Y を加えた識別可能性は UNRESOLVED（同 §10）。また、旧生成器（Z・X の標本 z-score）のデータは canonical モデルからの独立な標本ではないので、K0 = K_GEN を主張しない。
