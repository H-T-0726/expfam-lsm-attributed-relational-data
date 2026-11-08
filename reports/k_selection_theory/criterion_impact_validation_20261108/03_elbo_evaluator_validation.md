# 03 C_ELBO 評価器の実装と検証（Phase B、2026-11-08）

## 1. 実装

- **ファイル**：`expfam/src/experimental/elbo_k_criterion.py`（新しく追加）。
- **既存のコードとの関係**：`laplace_k_criterion`・推定器・`scale_Z` は変更していない。Laplace 評価器が定める $\hat Z$（結合モード）と、$H=-\nabla^2\Phi(\hat Z)$（`negative_hessian`）を読むだけである。

### 1.1 評価する量

$$
\mathcal L(q_{\rm Lap})=\mathbb E_{q_{\rm Lap}}\Phi+H(q_{\rm Lap}),\qquad q_{\rm Lap}=\mathcal N(\hat Z,H^{-1})
$$

### 1.2 期待値の計算方法（対象の量は変えていない）

$\Sigma=H^{-1}$ とする（Cholesky 分解で計算し、log 行列式が Laplace の値と一致することを確認している）。

**閉じた形で計算する部分**
- 事前分布：$-\frac{nK}2\log2\pi-\frac12(\|\hat Z\|^2+\mathrm{tr}\Sigma)$
- Gaussian-X：$\sum_{il}[-\frac12\log2\pi s_l-\frac{(x_{il}-f_l^\top\hat z_i)^2+f_l^\top\Sigma_{ii}f_l}{2s_l}]$
- Poisson-X：$\sum_{il}[x_{il}f_l^\top\hat z_i-\exp(f_l^\top\hat z_i+\frac12f_l^\top\Sigma_{ii}f_l)-\log x_{il}!]$（対数正規の積率）
- Y の線形項：$\sum y_{ij}\,\mathbb E[\eta_{ij}]$、$\mathbb E[\eta_{ij}]=w_0+w(\hat z_i^\top\hat z_j+\mathrm{tr}\Sigma_{ij})$

**MC で推定する部分**
- $\mathbb E[\sum_{(i,j)}\log(1+e^{\eta_{ij}})]$ だけ。
- $q_{\rm Lap}$ から**独立に**抽出した $Z=\hat Z+L^{-\top}\varepsilon$（$H=LL^\top$）を使う。MCEM の E-step の標本は使わない。
- MC 標準誤差：$\mathrm{MCSE}(C_{\rm ELBO})=2\,\mathrm{sd}(S)/\sqrt M$（S は 1 抽出あたりの softplus の和）。

### 1.3 seed と状態の扱い

- **seed**：`[20261108, phase, x_family, signal, n, rep, K, start, batch]`。8 バッチ。状態ごとに固有の値になる。
- **Laplace の判定が OK でない状態**：状態をそのまま返し、値は作らない（リッジの追加・補完はしない）。

## 2. テスト（`tools/research_audit/test_elbo_k_criterion.py`、10/10 PASS）

| テスト | 確認したこと |
|---|---|
| `phi_point_matches_complete_log_density`（Gaussian・Poisson） | 本評価器で書いた Φ が、既存の `complete_log_density` と 1e−8 以内で一致する。正規化定数・観測マスク・Bernoulli-Y の対応を確認した |
| `identity_and_entropy`（Gaussian・Poisson） | $H(q)=\frac{nK}2(1+\log2\pi)-\frac12\log|H|$。$C_{\rm ELBO}=C_{\rm Lap}+2R_3$（1e−8）。罰則が既存のものと一致する |
| `gaussian_w0_only_exact` | Gaussian-X で w=0（Φ が Z の 2 次式）なら、ELBO = ℓ_Lap = 厳密な周辺尤度（解析解）で、R3 = 0、MC の分散 = 0 |
| `semi_analytic_matches_brute_force_mc`（Gaussian・Poisson、w=0.8） | 一部を解析的に計算した $\mathbb E_q\Phi$ が、Φ 全体を MC で平均した値（20,000 抽出）と 5 SE 以内で一致する（**同じ対象量**であることの確認） |
| `sampling_covariance_is_H_inverse` | 抽出の共分散が $H^{-1}$ に一致する（正規化されたガウス密度からの正しい抽出） |
| `elbo_is_below_exact_marginal_tiny` | n=3、K=1、Poisson-X/Bernoulli-Y で、3 次元の格子積分による厳密な周辺尤度と比べ、ELBO ≤ ℓ_marg（MC 誤差の範囲で）が成り立つ |
| `not_ok_status_passes_through` | 停留点でない場合、値は NaN で、状態がそのまま返る |

**R3 の符号**：理論上の符号の保証はない。厳密に言えるのは $\ell_{\rm Lap}-R_3=\mathcal L\le\ell_{\rm marg}$ だけである。R3 の符号と大きさは、実測値として 05 で報告する。

**下界と MC 推定値の区別**
- ELBO ≤ ℓ_marg は、厳密な期待値についての定理である。
- MC 推定値は、MC 誤差の分だけそれを上回ることがありうる。

## 3. MC の予算（事前に決めた規則によるパイロット）

**規則**（プロトコル。パイロットの前に commit した）
- 固定した既存の 6 状態で、C_ELBO の MCSE と時間**だけ**を測る。選択は計算しない。
- M ∈ {256, 512, 1024, 2048} のうち、MCSE の中央値が 1.0 以下で、最大時間が 60 秒以下の最小の M を選ぶ。
- 該当する M がなければ 2048 にする。

**結果**（`elbo_mc_budget.json`）

| M | MCSE の中央値 | 最大時間（秒） |
|---|---:|---:|
| 256 | 11.76 | 1.29 |
| 512 | 7.76 | 1.45 |
| 1024 | 5.66 | 1.74 |
| 2048 | 3.99 | 2.41 |

- **条件を満たす M はなかったので、規則どおり M=2048 にした。**
- M=2048 の状態ごとの MCSE は、次のとおりだった。
  - n=75：約 1.6〜1.7
  - n=150：約 4.0〜5.3
- **限界**
  - 時間には余裕があった。M を増やすか、制御変量で分散を下げれば、MCSE を小さくできた。
  - しかし、規則を事後に変えないため、M=2048 のまま使った。
  - その代わり、C_ELBO の選択で、最良と次点の差が 2 SE_diff 未満のものは `MC_UNCERTAIN` とした（04・06）。

## 4. 既存の評価器との照合（Phase C）

- 既存の 448 状態について、A〜E を同じコードで再計算した。
- PR #160 の `criterion_scores.csv` との照合結果は `crosscheck_vs_pr160.csv` にある（04 §1）。
