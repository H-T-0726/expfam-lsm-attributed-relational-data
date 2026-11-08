# 07 優先候補の導出：C_Lap と C_ELBO(q_Lap)（Phase H、2026-11-08）

**位置づけ**
- どちらも**既存手法**（Laplace による周辺尤度、Laplace ガウスを q とした ELBO）を本モデルに適用したものである。新しい基準として提案するものではない。
- 導出の目的は、各段階でどの近似を使い、どこで正当化が止まるかを明確にすることである。

---

## 0. 共通の出発点

### 0.1 確率モデル（02 §1）

$$
p(X,Y,Z\mid\theta,K)=\prod_i\mathcal N(z_i;0,I_K)\prod_{i,l}p_X(x_{il}\mid f_l^\top z_i,\phi_l)\prod_{(i,j)\in\mathcal O}p_Y(y_{ij}\mid w_0+wz_i^\top z_j,\phi_Y)
$$

### 0.2 目標とする量：モデル K の evidence

$$
\log p(X,Y\mid K)=\log\int\pi(\theta\mid K)\,\exp\{\ell_{\rm marg}(\theta)\}\,d\theta,\qquad \ell_{\rm marg}(\theta)=\log\int e^{\Phi_\theta(Z)}dZ
$$

$-2\log p(X,Y\mid K)$ を最小にする K を選ぶ（目標 G2。01 §5）。

### 0.3 指数型分布族で必要な微分

$\Phi_\theta(Z)=\log p(X,Y,Z\mid\theta)$ の微分は次のとおり。

**勾配**

$$
\nabla_{z_i}\Phi=-z_i+\sum_lf_l\,\frac{x_{il}-A_X'(\eta^X_{il})}{\phi_l}+w\sum_{j:(i,j)\in\mathcal O}\frac{y_{ij}-A_Y'(\eta^Y_{ij})}{\phi_Y}\,z_j
$$

**負のヘッセ行列**：対角ブロックと非対角ブロックは 04 §4.1 のとおりである。
- X の曲率は $A_X''(\eta)/\phi_l$。
- Y の曲率と残差は $A_Y''(\eta)/\phi_Y$、$(y-A_Y'(\eta))/\phi_Y$。

**family が入るのは、A'・A''・φ・log h だけ**である。Gaussian・Bernoulli・Poisson の違いは、すべてこの 4 つで表される（02 §1.2 の表）。

実装：`laplace_k_criterion.gradient`・`negative_hessian`。現在は $\phi_Y=1$（Bernoulli/Poisson-Y）だけに対応している。

---

## 1. 候補 C：C_Lap

### 1.1 仮定

| # | 仮定 |
|---|---|
| (C-a) | θ を固定したとき、$\Phi_\theta$ は唯一の大域的な最大点 $\hat Z$ を持ち、$H=-\nabla^2\Phi_\theta(\hat Z)\succ0$ |
| (C-b) | 事後分布の質量は $\hat Z$ の近くに集中し、3 次以上の項の寄与が無視できる |
| (C-c) | θ の積分について、$\ell_{\rm marg}$ が $\hat\theta$ の近くで 2 次で近似でき、K に依存するブロック（F）の情報量が $\Theta(n)$ で増える（02 §5.1） |
| (C-d) | $\hat\theta$ は $\ell_{\rm marg}$（または $\ell_{\rm Lap}$）の最大点である |

### 1.2 Step 1：Z の積分（04 §3.1）

$$
\ell_{\rm marg}(\theta)\approx\ell_{\rm Lap}(\theta)=\Phi_\theta(\hat Z_\theta)+\frac{nK}2\log2\pi-\frac12\log|H_\theta|
$$

これを、当てはまり・事前分布・体積に分けて書くと、次のようになる。

$$
-2\ell_{\rm Lap}=\underbrace{-2\ell_{\rm cond}(\hat Z)}_{\text{モードでの当てはまり}}+\underbrace{\|\hat Z\|^2+\log|H|}_{\text{体積（Occam 因子）}}
$$

（$-\frac{nK}2\log2\pi$ は、事前分布の正規化項と相殺する。）実装の `lap_fit_mode`・`lap_volume`（`k_criterion_panel.py` L.121-127）と同じ分け方である。

**体積の項の意味**
- 新しい方向の事後精度が $a_i$ なら、その方向は $\sum_i\log a_i$ を加える（Occam 因子）。
- 情報のない方向（$a_i\approx1$、$\hat z\approx0$）では、ほぼ 0 になる。

### 1.3 Step 2：θ の積分（Schwarz 型、02 §5.1）

(C-c) と (C-d) のもとで、

$$
\log p(X,Y\mid K)\approx\ell_{\rm marg}(\hat\theta_K)-\frac12\sum_bp_b\log r_b(n)+O(1)
$$

K に依存するのは F のブロックだけで（$p_F=Kd-K(K-1)/2$、$r_F=n$）、ほかは K によらない定数である。

### 1.4 最終形

$$
\boxed{C_{\rm Lap}(K)=-2\ell_{\rm Lap}(\hat\theta_K)+\Big(Kd-\frac{K(K-1)}2\Big)\ln n\ (+\ \text{K によらない定数})}
$$

既存の実装（`calc_C_Lap`）と同じである。

### 1.5 導出で使った近似

| # | 近似 | 誤差 |
|---|---|---|
| ① | Z の 2 次近似（Step 1） | $e_Z=\ell_{\rm Lap}-\ell_{\rm marg}$。符号は不定 |
| ② | θ の 2 次近似と $O(1)$ の無視（Step 2） | — |
| ③ | $\hat\theta$ を MCEM の推定値で置き換えること（(C-d) が成り立たない） | $e_\theta\ge0$ |
| ④ | 回転の差し引きを正則な点で数えること（02 §4.2） | — |

### 1.6 証明できた性質（DERIVED、数値確認つき）

- Y を含まない Gaussian-X の因子モデル（Φ が Z の 2 次式）では、①は厳密である（T5c）。**X と Y の同時モデルで w≠0 なら、Φ は 2 次式ではなく、①は厳密ではない。**
- $\ell_{\rm Lap}$ は、Z の座標の取り方に依存しない（周辺尤度の近似なので）。$\sigma_z^2$ と (F, w) の冗長性に対して、不変な量を近似している（03 §4.1）。
- 03 §3 との関係：$C_Q^\ast-C_{\rm marg}=2H(\pi)$。$\pi\approx q_{\rm Lap}$ なら、次が成り立つ。
$$
C_Q^\ast-C_{\rm Lap}\approx2H(q_{\rm Lap})=nK(1+\log2\pi)-\log|H|
$$
  - 過去の研究で経験的に見つかった「C_Q − C_Lap ≈ 2H_Lap」（§20）とは、**向きだけが**整合する。
  - この理想化は、termA ≈ 0 と R3 ≈ 0 を別々に意味する。しかし、§20 では、termA（約 −90）と R3（約 +80）がどちらも大きく、両者が打ち消し合って残差が小さくなっていた。
  - したがって、**観測された近い一致は、理論で保証されない打ち消し合いに依存しており、この理想化で説明されたとは言えない**。

### 1.7 証明できていない性質

- ①の誤差の次数と、その K への依存性。とくに、過剰な次元での対称性・多峰性（04 §4.4）。
- ③の大きさと、その K への依存性（04 §5.1）。
- ②で、K > K0 のとき、$p_K/2$ を学習係数に置き換える補正（特異性）。
- 選択の一致性（U6）。
- Gaussian-Y での実装（未対応）。

### 1.8 既存の基準と違う項

$$
C_{\rm Lap}-C_Q=-\big[2H(q_{\rm Lap})+\text{termA}+2R_3\big]
$$

（§3 の恒等式。termA は MCEM 標本と $q_{\rm Lap}$ の差、$R_3$ は §2 の定義。）

### 1.9 導出が止まるところ

- **Step 2 の (C-c) は、X の周辺（または w=0）について、K > K0 で成り立たない（特異）。**
  - その場合、K > K0 の候補に対する $C_{\rm Lap}$ の罰則は、理論的に過大でありうる（$\lambda\le p_K/2$）。
  - この過大な罰則が効くのは K > K0 の候補だけなので、**過大選択を抑える方向**に効く。K̂ を K0 より下げる理由にはならない（K0 のモデルは、generic な真値で正則）。
  - Gaussian-Y で w≠0 なら、過大側の候補はそもそも誤指定で（02 §4.3）、特異性の議論とは前提が違う。
- (C-d) を満たすには、$\ell_{\rm Lap}(\theta)$ を θ について最大化する手順が必要である。それは現在の実装にない（Human Gate）。

---

## 2. 候補 B：C_ELBO(q_Lap)

### 2.1 仮定

| # | 仮定 |
|---|---|
| (B-a) | (C-a) のうち、$H\succ0$ だけ（q を定義するため） |
| (B-b) | $\mathbb E_{q}\Phi$ が有限である |
| (B-c)・(B-d) | (C-c)・(C-d) と同じ（θ の部分） |

**大域的な最大点・単峰性・2 次近似の正確さは要らない。**

### 2.2 Step 1：Z の下界（04 §1・§3.3）

$q_{\rm Lap}=\mathcal N(\hat Z,H^{-1})$ について、

$$
\mathcal L(q_{\rm Lap},\theta)=\mathbb E_{q_{\rm Lap}}\Phi_\theta+\frac{nK}2(1+\log2\pi)-\frac12\log|H|=\ell_{\rm Lap}(\theta)-R_3(\theta)
$$

$$
R_3=\Phi(\hat Z)-\frac{nK}2-\mathbb E_{q_{\rm Lap}}\Phi
$$

また、$\mathcal L(q_{\rm Lap},\theta)\le\ell_{\rm marg}(\theta)$ が、**仮定なしの恒等式（Gibbs の不等式）として成り立つ**。

### 2.3 計算方法

- $\mathbb E_{q_{\rm Lap}}\Phi$ は、$q_{\rm Lap}$ から独立に M 個抽出した $Z^{(m)}=\hat Z+L^{-\top}\varepsilon^{(m)}$（$H=LL^\top$ は Cholesky 分解）の平均で、不偏に推定できる。MC 誤差は $\mathrm{Var}_{q}(\Phi)/M$ である。
- **MCEM の標本は使わない**（04 §2.5）。
- 過去の研究（§20）で計算した「Q(q_Lap)」と同じ部品である。ただし、それを基準として選択に使ったことはない。

### 2.4 最終形

$$
\boxed{C_{\rm ELBO}(K)=-2\mathcal L(q_{\rm Lap},\hat\theta_K)+\Big(Kd-\frac{K(K-1)}2\Big)\ln n=C_{\rm Lap}(K)+2R_3(K)}
$$

### 2.5 証明できた性質（DERIVED）

1. **下界**：$-2\mathcal L\ge-2\ell_{\rm marg}$。したがって、$C_{\rm ELBO}\ge C_{\rm marg}$（θ を固定したとき）。その差は $2\mathrm{KL}(q_{\rm Lap}\|\pi)\ge0$ である。
2. **区間**：$\ell_{\rm Lap}-R_3\le\ell_{\rm marg}$。Laplace の値が下界をどれだけ上回っているかで、Laplace の誤差の「上側」の余地がわかる。ただし、上界は得られない。
3. **Y を含まない Gaussian-X の因子モデルでは $R_3=0$ で、$C_{\rm ELBO}=C_{\rm Lap}=C_{\rm marg}$**（T5d）。X と Y の同時モデル（w≠0）では成り立たない。
4. **C_Q との関係（恒等式）**
$$
C_Q=C_{\rm ELBO}(q_{\rm Lap})+2H(q_{\rm Lap})+\text{termA},\qquad \text{termA}=-2\big(Q_{MC}-\mathbb E_{q_{\rm Lap}}\Phi\big)
$$
   - 定義を代入するだけで成り立つ。
   - 過去の研究の恒等式「C_Q − C_Lap − 2H_Lap = termA + 2R3」（§20、448 状態で 5e−12 以内）と同じものである。

### 2.6 証明できていない性質

- $\mathrm{KL}(q_{\rm Lap}\|\pi_K)$ の K への依存性。KL が K とともに増えると過小選択に偏る（04 §2.3）。
  - 過去の研究では R3 が K とともに増えることが観測されている（§20：K=7 で約 +80。OBSERVED）。$C_{\rm ELBO}-C_{\rm Lap}=2R_3$ なので、**その状態では、C_ELBO は C_Lap より大きい K に厳しくなる**と予想される。
  - これは既存の値からの含意で、選択を計算したものではない。
- 選択の一致性。
- (B-c)・(B-d) に関わる部分（C と同じ）。

### 2.7 既存の基準と違う項

$C_{\rm ELBO}-C_{\rm Lap}=2R_3$。$q_{\rm Lap}$ のもとでの、Φ の 2 次でない部分である。

### 2.8 導出が止まるところ

- 下界であることは厳密だが、**下界が K 間で同じ程度にきつい保証はない**。そのため、選択の正しさは保証されない。
- 平均場の $q_{\rm MF}$ や、より柔軟な q（混合など）で下界をきつくすれば、K への依存性は変わる。しかし、どの q が十分かは理論的に決まらない。

---

## 3. 3 つの量の関係のまとめ（DERIVED、θ を固定）

$$
\underbrace{C_Q}_{\text{現行}}=\underbrace{C_{\rm ELBO}(q_{\rm Lap})}_{\ge C_{\rm marg}}+\underbrace{2H(q_{\rm Lap})}_{nK(1+\log2\pi)-\log|H|}+\text{termA}
$$

$$
C_{\rm ELBO}(q_{\rm Lap})=C_{\rm Lap}+2R_3,\qquad C_{\rm marg}\le C_{\rm ELBO}(q)\ \ (\forall q)
$$

$$
(C_Q-P_Z)^\ast=C_{\rm marg}-2\mathrm{KL}(\pi\|p_Z)
$$

**K を選ぶときに効く差**
- C_Q：$2\Delta H$（事後精度と $2\pi e$ の大小）と、termA の K 依存性。
- C_Lap：$e_Z$（符号は不定）と、$e_\theta$（過小選択の方向）の K 依存性。
- C_ELBO：$2\Delta\mathrm{KL}(q\|\pi)$（過小選択の方向）と、$e_\theta$ の K 依存性。

**C_Lap と C_ELBO を並べたとき**
- 両者の選択が一致すれば、Laplace の 2 次でない部分（R3）が選択に効いていない、という証拠になる。
- 食い違えば、その K の組では近似の誤差が選択を決めうる、という警告になる。
- どちらも、「正しい K」を保証するものではない。
