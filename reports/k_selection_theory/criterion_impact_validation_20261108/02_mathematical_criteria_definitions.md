# 02 基準の数学的定義と恒等式（Phase A、2026-11-08）

導出の詳細は PR #163（`reports/k_selection_theory/bic_theory_candidate_assessment_20261108/` の 03・04・07）にある。ここでは、本研究で使う定義と、そのコード上の対応を固定する。

## 1. 記法

$$
\Phi_K(Z)=\log p(X,Y_{\rm train},Z\mid\hat\theta_K),\qquad
\ell_{{\rm cond},K}(Z)=\log p(X,Y_{\rm train}\mid Z,\hat\theta_K),\qquad
\ell_{{\rm marg},K}=\log\int e^{\Phi_K(Z)}dZ
$$

- $\hat\theta_K$ は、各状態の MCEM の推定値（8 反復、未収束）である。
- Y の項は、学習用ペア（80%）の上三角だけで評価する。
- すべての正規化定数を含める。
  - Gaussian-X：$-\tfrac12\log2\pi s_l$
  - Poisson-X：$-\log x!$
  - 事前分布：$-\tfrac{nK}{2}\log2\pi$

**罰則**：$P_\theta=\big(Kd-K(K-1)/2+d\cdot1\{\text{Gaussian-X}\}\big)\ln n$（`fair_common_evaluator.num_params`）。全基準で同じ値を使う。K の間の差に効くのは $P_K=(Kd-K(K-1)/2)\ln n$ の部分である。

## 2. 6 つの基準

| 記号 | 定義 | コード |
|---|---|---|
| A C_Q | $-2Q_{\rm MC}+P_\theta$、$Q_{\rm MC}=\frac1L\sum_l\Phi(Z_l)$（最終反復の L=5 標本、`scale_Z` 済み） | `fair_common_evaluator.criteria_abc` |
| B C_cond_MC | $-2\frac1L\sum_l\ell_{\rm cond}(Z_l)+P_\theta=C_Q-P_Z$ | 同上 |
| C C_cond_last | $-2\ell_{\rm cond}(Z_{\rm last})+P_\theta$ | 同上 |
| D C_cond_mode | $-2\ell_{\rm cond}(\hat Z)+P_\theta$。$\hat Z$ は Φ の結合モード | `run_criterion_impact.score_state` |
| E C_Lap | $-2[\Phi(\hat Z)+\frac{nK}2\log2\pi-\frac12\log|H|]+P_\theta$ | `laplace_k_criterion` |
| F C_ELBO | $-2[\mathbb E_{q_{\rm Lap}}\Phi+H(q_{\rm Lap})]+P_\theta$、$q_{\rm Lap}=\mathcal N(\hat Z,H^{-1})$、$H(q_{\rm Lap})=\frac{nK}2(1+\log2\pi)-\frac12\log|H|$ | `elbo_k_criterion`（新しく追加） |

- $\hat Z$ と H は、Laplace 評価器と同じものを使う。
  - $Z_{\rm last}$ から出発する。
  - 停留性 max|∇Φ| ≤ 1e−8 と、H の正定値性を確認する。
  - 確認できない状態は INCOMPLETE とする。リッジの追加や補完はしない。

## 3. 恒等式

### 3.1 理想化の式（厳密な事後期待値。実装の C_Q にそのまま成り立つとは扱わない）

$$
C_Q^\ast=C_{\rm marg}+2H(\pi_K),\qquad (C_Q-P_Z)^\ast=C_{\rm marg}-2\mathrm{KL}(\pi_K\|p_Z)
$$

これらは**比較の解釈の枠組み**として使う。実装の値で検証することはできない（厳密な事後分布が分からない）。

### 3.2 実装でそのまま成り立つ式（同じ θ・$\hat Z$・H・正規化）

$$
C_{\rm ELBO}=C_{\rm Lap}+2R_3,\qquad R_3=\Phi(\hat Z)-\frac{nK}2-\mathbb E_{q_{\rm Lap}}\Phi
$$

$$
C_Q=C_{\rm ELBO}+2H(q_{\rm Lap})+\text{termA},\qquad \text{termA}=-2(Q_{\rm MC}-\mathbb E_{q_{\rm Lap}}\Phi)
$$

$$
\Rightarrow\quad C_Q-C_{\rm Lap}=2H(q_{\rm Lap})+\text{termA}+2R_3
$$

- 両辺に同じ MC 推定値 $\mathbb E_{q}\Phi$ が入るので、これらは**代数的に厳密**に成り立つ（数値誤差の範囲で）。各状態で残差を記録した（`score_by_state.csv` の `id1_resid`・`id2_resid`）。
- termA と R3 は**個別に** MC 誤差を持つ。ただし、上の和の中では打ち消し合う。

### 3.3 選択への効き方（K → K+1 の余裕）

余裕 $\text{slack}(K\to K+1)=C(K)-C(K+1)$（正なら K+1 を選ぶ方向）について、

$$
\text{slack}_Q-\text{slack}_{\rm Lap}=-\big[\Delta2H(q_{\rm Lap})+\Delta\text{termA}+\Delta2R_3\big],\qquad
\text{slack}_{\rm ELBO}-\text{slack}_{\rm Lap}=-\Delta2R_3
$$

（Δ は、値(K+1) − 値(K)。）

**注意**
- 個々の K で KL が正でも、選択に効くのは **K の間の増分**である。「ELBO は必ず過小選択する」とは主張しない。
- $2\pi e$ の閾値（PR #163）は、ガウス近似のもとでの $\Delta2H$ の符号の目安にすぎない。選択の成功条件ではない。
- 数値の分解は、因果効果の証明ではない。

## 4. 原論文との対応

- B〜D は、原論文 Eq.(16) の条件付き尤度を、Z の評価点を 3 通りに明示して評価したものである。原論文は評価点を書いていないので、どれも原論文の再現ではない。
- Gaussian-X/Bernoulli-Y のセルだけが、原論文のモデル（μ_x=0、σ_z²=1）に対応する比較である。
- 罰則：原論文の $(K+1)d-K(K-1)/2$ と本研究の $P_\theta$ の差は、K によらない定数で、選択には影響しない（PR #159・#163）。
