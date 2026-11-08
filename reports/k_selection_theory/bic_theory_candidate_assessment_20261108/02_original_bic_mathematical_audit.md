# 02 モデルの定義と、原論文 Eq.(26) の BIC 型基準の理論監査（Phase A・B、2026-11-08）

証拠ラベル：FACT（一次資料で確認）／DERIVED（導出）／OBSERVED（過去の一次結果）／INTERPRETATION／HYPOTHESIS。

---

## 1. 研究モデル（Phase A1）

### 1.1 生成モデル（FACT：AGENTS.md §1、RESEARCH_MASTER.md §4）

$$
z_i\overset{\rm iid}{\sim}\mathcal N(0,I_K),\qquad
\eta^Y_{ij}=w_0^Y+w^Y z_i^\top z_j\ (i<j),\qquad
\eta^X_{il}=f_l^\top z_i .
$$

- $w_0^Y, w^Y\in\mathbb R$ はスカラー。X 側にバイアスはない。
- 標準の `DualExpFamLSM` 系では、`family_x` は X の全列で共通の 1 種類である。列ごとに family を変える `family_x_list` は prototype（`model_dual_expfam_percolumn.py`）で、本文には採用できない。
- 標準実装では $\sigma_z^2=1$ に固定している（`var_z`）。原論文は $\sigma_z^2$ を Eq.(14) で推定する。

### 1.2 指数型分布族の共通形（DERIVED、実装の特殊形に合わせる）

一般形は、十分統計量 $T(t)$ を使って次のように書ける。

$$
p(t\mid\eta,\phi)=h(t,\phi)\exp\Big\{\frac{T(t)\eta-A(\eta)}{\phi}\Big\}
$$

本研究の実装は、**正準リンク・$T(t)=t$** の特殊形を使う。

| family | 自然パラメータ η | 対数分配関数 $A(\eta)$ | dispersion φ | 基準測度 $h(t,\phi)$ | 平均 $A'(\eta)$ | 分散関数 $A''(\eta)$ | 条件付き分散 |
|---|---|---|---|---|---|---|---|
| Gaussian | 平均そのもの | $\eta^2/2$ | $\sigma^2$ | $(2\pi\sigma^2)^{-1/2}e^{-t^2/(2\sigma^2)}$ | $\eta$ | $1$ | $\sigma^2$ |
| Bernoulli | ロジット | $\log(1+e^\eta)$ | 1 | 1 | $\sigma(\eta)$ | $\sigma(\eta)(1-\sigma(\eta))$ | 同左 |
| Poisson | 対数率 | $e^\eta$ | 1 | $1/t!$ | $e^\eta$ | $e^\eta$ | 同左 |

**X 側のパラメータ**
- $F=(f_1,\dots,f_d)^\top\in\mathbb R^{d\times K}$
- Gaussian-X のときだけ $\Sigma_X={\rm diag}(\sigma_1^2,\dots,\sigma_d^2)$（`calc_sigma`）
- Bernoulli-X・Poisson-X では φ=1

**Y 側のパラメータ**
- $w_0^Y, w^Y$
- Gaussian-Y のときだけ $\sigma_Y^2$（実装は標準偏差を `self.sigma_y` に持ち、使うときに二乗する）

**対数密度（正規化項を落とさない）**

$$
\log p(t\mid\eta,\phi)=\frac{t\eta-A(\eta)}{\phi}+\log h(t,\phi)
$$

| family | $\log h$ |
|---|---|
| Gaussian | $-\tfrac12\log(2\pi\sigma^2)-t^2/(2\sigma^2)$ |
| Bernoulli | 0 |
| Poisson | $-\log t!$ |

実装：`eval_utils.calc_Q_dual_strict_exp` L.215-228 は、Poisson の $-\sum\log y!$・$-\sum\log x!$ と、Gaussian-Y の $-\tfrac12\log 2\pi$ を補正している。Gaussian-X の定数は、consistent モデルの `calc_log_likelihood_X`（`model_dual_expfam_consistent.py` L.94-96）に入っている。

### 1.3 Y の尤度とペアの数え方（FACT + DERIVED）

- 原論文 Eq.(16) は $\{\prod_{i\ne j}p(y_{ij}\mid\cdot)\}^{1/2}$ である。Y が対称なら、これは $\prod_{i<j}$ と同じで、各ペアを 1 回数える。
- 実装では `0.5 * sum(ln_p * mask_f)`（`model_dual_expfam_consistent.py` L.64）で、同じ量を計算している。
- 尤度レベルの 1/2 は正しい。z_i の勾配・精度に余分な 1/2 が残るかどうかは別の問題で、KI-001 で扱う（本研究の採用式には余分な 1/2 はない）。

---

## 2. 識別可能性（Phase A2）

**既存の結果（FACT：RESEARCH_MASTER.md §17.3・§17.6・§22）**

| 対象 | 証明済み | 未解決 |
|---|---|---|
| 回転 | P4：$O(K)$ の回転・符号反転・座標の置換は、観測分布を変えない。$\mathcal N(0,I)$ の事前分布がスケールを固定する | — |
| Gaussian-X | P5：Σ が既知なら、**X 周辺**の最小次元は $\mathrm{rank}(\mathrm{Cov}(X)-\Sigma)$（rank F = K という generic な条件） | U3：Σ が未知のときの十分条件 |
| Poisson-X | P1：**X 周辺**の最小次元は $\mathrm{rank}(FF^\top)$（クリップなしのリンク、rank F = K） | K0（X と Y の同時分布での最小次元）としての一意性は未確認（§22「証明案は検証中」） |
| Bernoulli-X | P7：一次モーメントは K の情報を持たない | U1：d > 1 での識別可能性 |
| Gaussian-Y | P2：$w\ne0$ なら、単一の dyad の周辺分布から $(w_0,w^2,K,\sigma_Y^2)$ が決まる。P8：w の符号は三角形から決まる。P3：$\{P\in\mathcal M_K:w\ne0\}\cap\mathcal M_{K+1}=\emptyset$（入れ子にならない） | — |
| Bernoulli-Y | — | **U2：識別可能性。U5：入れ子にならないかどうか**（実験で使っている family） |
| Poisson-Y | P6：$E[Y^r]<\infty\iff|w|<1/r$ | U4 |
| 選択の一致性 | — | **U6：どの基準についても、n→∞ での一致性** |

**一般則（既存）**：周辺分布で示した肯定的な識別性は、同時分布に移せる。否定的な結果（識別できないこと）は移せない。

---

## 3. 3 種類の尤度（Phase B1）

### 3.1 定義と関係（DERIVED）

条件付き確率の定義 $p(X,Y,Z\mid\theta)=p(X,Y\mid Z,\theta)\,p(Z)$ から、次の 3 つが得られる。

$$
\ell_{\rm cond}(Z,\theta)=\log p(X,Y\mid Z,\theta)=\sum_{i,l}\log p(x_{il}\mid\eta^X_{il})+\sum_{i<j}\log p(y_{ij}\mid\eta^Y_{ij})
$$

$$
\ell_{\rm comp}(Z,\theta)=\ell_{\rm cond}(Z,\theta)+\log p(Z),\qquad \log p(Z)=-\frac{nK}{2}\log2\pi-\frac12\|Z\|_F^2
$$

$$
\ell_{\rm marg}(\theta)=\log\int_{\mathbb R^{nK}}\exp\{\ell_{\rm comp}(Z,\theta)\}\,dZ
=\log \mathbb E_{p(Z)}\big[p(X,Y\mid Z,\theta)\big]
$$

- **$\ell_{\rm cond}$**：Z を $nK$ 個の固定値として扱う。K を増やすと、自由に動かせる座標が n 個増える。
- **$\ell_{\rm comp}$**：同じく Z の 1 点で評価するが、$\log p(Z)$ を含む。値は Z の座標の取り方に依存する（03 §4）。
- **$\ell_{\rm marg}$**：Z を積分で消した、観測 (X, Y) の尤度。Z の座標の取り方によらない（積分の変数変換で不変）。Schwarz BIC が近似する evidence $\log\int p(X,Y\mid\theta)\pi(\theta)d\theta$ の被積分関数は、この $\ell_{\rm marg}$ である。

**Jensen の不等式からの順序（DERIVED）**
- $\ell_{\rm marg}(\theta)\ \ge\ \mathbb E_{p(Z)}\ell_{\rm cond}(Z,\theta)$
- 一方、$\sup_Z \ell_{\rm cond}(Z,\theta)\ \ge\ \ell_{\rm marg}(\theta)$ も成り立つ。理由：$\ell_{\rm marg}=\log\mathbb E_{p(Z)}[e^{\ell_{\rm cond}}]\le\log\sup_Z e^{\ell_{\rm cond}}$。
- つまり **Z を最大化した $\ell_{\rm cond}$ は、周辺尤度を上から押さえる**。

### 3.2 原論文の各式との対応（FACT：PDF 本文）

| 原論文 | 内容 | 対応 |
|---|---|---|
| Eq.(16) | $L=\prod_i p(x_i\mid F,z_i,\mu_x,\Sigma)\{\prod_{i\ne j}p(y_{ij}\mid z_i,z_j,w_0,w)\}^{1/2}$ | $\ell_{\rm cond}$（$p(Z)$ を含まない。Z を積分していない） |
| Eq.(17) | $Q(\theta,\theta^{\rm old})=\int p(Z\mid X,Y,\theta^{\rm old})\ln p(Z,X,Y\mid\theta)dZ$ | $\ell_{\rm comp}$ の事後期待値 |
| Eq.(18) | $\frac1L\sum_l\{\ln p(Z^{(l)}\mid\theta)+\ln p(X\mid Z^{(l)},\theta)+\ln p(Y\mid Z^{(l)},\theta)\}$ | Eq.(17) の MC 近似 |
| Eq.(26) | $\mathrm{BIC}=-2\ln L+\big((k+1)d-\tfrac{k(k-1)}{2}\big)\ln n$。本文に「the true dimension k affects the likelihood function of L from Eq. (16)」とある | **$L$ は Eq.(16) の $\ell_{\rm cond}$** |
| §4.3 | 「BIC possesses consistency」を、BIC を使う理由として挙げている。参照は Schwarz (1978) | 一致性の証明はない |

### 3.3 Eq.(26) についての 6 つの確認（Phase B1 の問い）

1. **$L_K$ はどの尤度か**：Eq.(16) の条件付き尤度 $\ell_{\rm cond}(Z,\theta)$（FACT）。
2. **Z はどこで固定・平均・積分されているか**：積分はしていない。どの Z（最後の標本、標本平均、モード）で評価したかは、本文に書かれていない（FACT。PR #158 でも UNRESOLVED）。
3. **本文だけで評価方法が一意に決まるか**：決まらない（DERIVED）。$\ell_{\rm cond}$ は Z に強く依存し、たとえば $\sup_Z$ と事後標本では値が大きく違う（PR #159 の C・D）。
4. **Schwarz BIC の導出条件を満たすか**：満たさない（DERIVED）。理由は次の 4 つ。
   - (i) 被積分関数は $\ell_{\rm marg}$ であるべきだが、$\ell_{\rm cond}$ を使っている。
   - (ii) $\ell_{\rm cond}$ を Z について最適化（または事後分布で評価）するなら、Z の $nK$ 個の値は実質的に推定されたパラメータで、ノードの数とともに増える（incidental parameter）。Schwarz の導出は、固定次元のパラメータに対する Laplace 近似であり、この状況を含まない。
   - (iii) 後述（§4.3）の特異性。
   - (iv) 有効標本サイズが定義されていない（§5）。
5. **一致性の主張は、このモデルで証明されているか**：されていない（FACT：論文に証明はない。論文の文献 [20] は Schwarz (1978) である（PDF の文献リストで確認）。Schwarz (1978) が i.i.d. の正則な指数型分布族についての結果であることは書誌的な知識で、本セッションでは本文を読んでいない）。本研究でも、どの基準についても一致性は証明されていない（U6）。
6. **Fig. 3 が支持する範囲**
   - (FACT/OBSERVED、PR #158) 生成次元 1・3・5・7・9 のそれぞれで、1 本の曲線の最小値が真の次元にある。
   - 過大側の余裕は 4〜21 BIC 単位、K\*=9 の過小側の余裕は 57 と小さい。
   - 図から逆算した $-2\ln L$ は 1.6〜193 で、Gaussian の 2,250 成分と 11,175 ペアの対数尤度としては小さすぎる。ln L が正規化・再スケールされた量である可能性がある（HYPOTHESIS、PR #158）。
   - したがって **Fig. 3 が支持するのは「その 5 つのデータセットで、その計算手続き（詳細不明）が真の次元を最小にした」ことだけ**である。一般の一致性も、ほかの条件への一般化も支持しない。

### 3.4 Eq.(26) に問題があると断定できるか（結論）

**断定できること（DERIVED）**
- Eq.(26) は Schwarz BIC の定義（$-2\ell_{\rm marg}(\hat\theta)+p\ln n$）とは別の量である。
- Schwarz BIC としての理論的な正当化はない。

**問題が起きる可能性（DERIVED + 過去の OBSERVED）**
- $\ell_{\rm cond}$ を Z の点推定で評価すると、K を 1 増やすごとに n 個の自由な座標が増える。そのため、罰則 $(d-K)\ln n$ では過大選択を防げない方向に偏る（03 §3、05 の JIC）。
- PR #159〜#162 では、Eq.(16) 型の基準を 3 通りの評価点（MC 平均・最後の標本・モード）で評価した。canonical の d=15・n≥75 と paper-type のすべてで、過大選択が観測された（OBSERVED、系列 E）。
  - ただし、d=60・n=75 では B・C が 8/8 正解した。d=30 では正解と過大選択が混在した（PR #159 04 §1）。
- ただし、それらは**原論文の手続きの再現ではない**（評価点・正規化が不明）。

**機能する可能性（INTERPRETATION）**
- 次の場合には、Eq.(26) 型の値が真の次元の近くで最小になりうる。
  - 真の次元までの当てはまりの改善が非常に大きく、過大側の改善が小さく見える評価手続き（たとえば、何らかの正規化）を使った場合
  - X の信号が非常に強い paper-type の条件
- 原論文の条件は $F\sim\mathcal N(0,10)$・$\Sigma=0.1I$ で、X の信号が極めて強い。

**結論**：「Eq.(26) が間違っている」とは断定しない。断定できるのは次の 2 点である。
- (a) Schwarz BIC としての理論的な保証がない。
- (b) Z の評価点を明示した Eq.(16) 型の基準は、本研究の canonical 条件では過大選択に偏る、という導出上の理由と観測がある。

Fig. 3 で成功した理由は、評価手続きが不明なので UNRESOLVED のままである。

---

## 4. パラメータ数 $p_K=(K+1)d-K(K-1)/2$ の検証（Phase B2）

### 4.1 各項の由来（DERIVED）

| 項目 | 自由度 | K への依存 | 扱い |
|---|---|---|---|
| F | $dK$ | あり | |
| 回転 $O(K)$ | $-K(K-1)/2$ | あり | 周辺モデルでは、$Z\mapsto ZR$（$R\in O(K)$）としても $p(Z)$ は不変で、$\eta^Y$ も不変、$\eta^X$ は $F\mapsto FR$ で吸収される。したがって $(F,w_0,w)$ と $(FR,w_0,w)$ は同じ観測分布を与える。軌道の次元は、F の列が線形独立なときに $K(K-1)/2$ |
| Gaussian-X の Σ | d | なし | |
| μ_x（原論文のモデルのみ） | d | なし | |
| $w_0, w$ | 2 | なし | |
| $\sigma_Y^2$（Gaussian-Y） | 1 | なし | |
| $\sigma_z^2$（原論文） | 1 | なし | **冗長**。$(\sigma_z^2,F,w)\to(\sigma_z^2/c^2,cF,c^2w)$ で観測分布が変わらない（03 §4.1）。数えるべきではない |
| Z | $nK$ | あり | **周辺モデルではパラメータではない**（積分で消える）。Eq.(16) 型で Z を点推定する場合は、実質的なパラメータになるが、Eq.(26) は数えていない |

- 原論文の $(K+1)d-K(K-1)/2$ は、因子分析のパラメータ数（$dK+d-K(K-1)/2$。F と Σ）と同じ形である。
  - +d が Σ の分か μ_x の分かは、本文から決まらない（PR #158 の 01 §2 と同じく UNRESOLVED）。どちらの読み方でも、残りの d 個と w0・w・σ_z² は数えていない。
  - どちらの読み方でも、K の選択には影響しない。
- 現行の実装（`calc_bic_exp` L.245-250）は $Kd-K(K-1)/2+d\cdot1\{\text{Gaussian-X}\}+1\{\text{Gaussian-Y}\}$ を使う。
- 両者の差は K によらない定数なので、**K の選択には影響しない**（PR #159 01 §4 と同じ結論）。

**一般形**

$$
p_K=Kd-\frac{K(K-1)}{2}+c,\qquad p_{K+1}-p_K=d-K
$$

### 4.2 回転を引けば有効な次元になるか（DERIVED + 数値確認）

写像 $(F,\Psi)\mapsto FF^\top+\Psi$（Gaussian-X の周辺共分散）の Jacobian の階数を、d=15 で数値的に計算した（`identity_checks.csv` の T6）。

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| generic な F での階数 | 30 | 44 | 57 | 69 | 80 | 90 | 99 | 107 | 114 | **120** | **120** | **120** |
| $dK+d-K(K-1)/2$ | 30 | 44 | 57 | 69 | 80 | 90 | 99 | 107 | 114 | 120 | **125** | **129** |

- **generic な点では、回転を引いた数が正しい**（K ≤ 10）。
- **K=10 で、X の周辺共分散の自由度 $d(d+1)/2=120$ に達する。K ≥ 11 では、パラメータの数え方がこれを超える。** その領域では、数えた数は X の周辺分布が持つ情報を超える。
  - 原論文の候補は K ≤ 10 で、ちょうど飽和の境界にある（Ledermann の境界 $(d-K)^2\ge d+K$ も K=10 で等号）。
  - ただし、Y も同じ Z を使うので、同時分布での識別可能な次元はこれより大きい可能性がある（UNRESOLVED）。
- **F の最後の列が 0 の点では、階数が落ちる**：K=3 で 57→44、K=5 で 80→69。これは K−1 次元の generic な数と同じである。
  - **X の周辺分布（または w=0 の場合）について、過剰に指定したモデルの真値の集合は特異点である**（Fisher 情報が退化する）。その点では、$p_K/2$ を Watanabe の学習係数 λ に置き換える必要がある（$\lambda\le p_K/2$）。これは sBIC が扱う問題である（05）。
  - **X と Y の同時モデルで $w\ne0$ の場合**：Gaussian-Y では、真値はそもそも $\mathcal M_{K+1}$ に入らない（§4.3、P3）。そのため、「過剰な指定の真値が特異点」という前提自体が成り立たない。Bernoulli-Y では未解決（U5）。
- 罰則の増分 $d-K$ は、K が d に近づくと小さくなる。K の大きい側では、この罰則では過大選択を防げない、という方向の性質である。

### 4.3 Y 側の構造に固有の点：入れ子になっていない（DERIVED、本研究で整理）

- スカラー w をすべての次元が共有し、事前分布 $\mathcal N(0,I)$ が固定されている。そのため、K+1 次元目は「係数を 0 にして切る」ことができない。
  - $\eta^Y_{ij}=w_0+w\sum_{a=1}^{K+1}z_{ia}z_{ja}$ の新しい項 $wz_{i,K+1}z_{j,K+1}$ は、$w\ne0$ である限り、分散 $w^2$ の構造的な揺らぎを全ペアに加える。
- X 側は、F の列を 0 にすれば切れる（入れ子）。
- **Y 側については、$w\ne0$ の真値が $\mathcal M_{K+1}$ に入らない。**
  - Gaussian-Y では、これが P3 として証明されている。
  - Bernoulli-Y では未解決（U5）。
- **意味**
  - Gaussian-Y の場合、過大側の候補 K+1 は「真値を含む特異なモデル」ではなく、「誤指定のモデル」である。
  - その場合、過大選択を防ぐのは、罰則だけではなく当てはまりの悪化（KL の差）でもある。
  - sBIC は「真値を含む入れ子のモデル列」を前提とする（05）。この前提は、Gaussian-Y の Y 部分では成り立たない。

**命題 B-1（条件付き、DERIVED）**

**主張**：仮定の下で、次の不等式が成り立つ。

$$
\mathrm{KL}\big(P_0^{(n)}\,\|\,P_\theta^{(n)}\big)\ \ge\ \lfloor n/2\rfloor\,\delta_{K'}\qquad(\forall\theta\in\Theta_{K'})
$$

**仮定**：真の分布 $P_0$ の単一 dyad の周辺分布と、候補 $\mathcal M_{K'}$ の単一 dyad の周辺分布との KL の下限 $\delta_{K'}=\inf_{\theta\in\Theta_{K'}}\mathrm{KL}(P_{0,\rm dyad}\|P_{\theta,\rm dyad})$ が正である。

**証明**
1. ノードを互いに素な $\lfloor n/2\rfloor$ 組のペアに分ける（完全マッチング）。それらの $y_{ij}$ は、異なる z の組だけに依存する。z は i.i.d. なので、両方のモデルの下で、これらの $y_{ij}$ は互いに独立である。
2. 周辺化（一部の変数への射影）は KL を増やさない（データ処理不等式）。
3. 独立な成分の KL は和になる。

**射程と限界**
- **$P_0$ 自身が LSM の型**（ノードの潜在変数が i.i.d. で、潜在変数が与えられたとき dyad が条件付き独立）であることを仮定している。一般の $P_0$ では、互いに素な dyad の独立性は成り立たない。
- $P_0$ が交換可能でないと dyad ごとに周辺分布が異なるので、$\delta_{K'}$ は dyad ごとの下限として定義し直す必要がある。
- 観測マスクがあると、**観測されたペア**の中で完全マッチングがとれるとは限らない。その場合、$\lfloor n/2\rfloor$ はとれるマッチングの大きさに置き換わる。
- 母集団レベルの分離（期待対数尤度比）を示すだけである。最尤推定値での $\sup_\theta$ の差（標本レベル）や、選択の一致性は示さない。
- $\delta_{K'}>0$ が成り立つかどうか：
  - Gaussian-Y では、P2 のキュムラントの関係から K を合わせられないことが分かっている。ただし、下限が正であること（コンパクト性など）は確認していない。
  - Bernoulli-Y では未解決である。

---

## 5. 標本サイズ n の妥当性（Phase B3）

### 5.1 なぜ log n か：ブロックごとの情報量の増え方（DERIVED、発見的）

Schwarz の議論を θ について行う。周辺対数尤度 $\ell_{\rm marg}(\theta)$ の θ に関する Laplace 近似では、次の項が出る。

$$
-\tfrac12\log|J_n(\hat\theta)|
$$

（$J_n$ は観測情報行列。）パラメータのブロック b の情報量が速さ $r_b(n)$ で増えるとき、ブロック間の相関を無視すると、次のようになる。

$$
-2\log p(X,Y\mid K)\approx-2\ell_{\rm marg}(\hat\theta_K)+\sum_b p_b\log r_b(n)+O(1)
$$

| ブロック | 情報量の速さ | 根拠 |
|---|---|---|
| **F**（K に依存する唯一のブロック） | $\Theta(n)$ | F は $p(X\mid Z)$ にしか現れない。周辺の情報量は、完全データの情報量 $n\,\mathbb E[zz^\top]\otimes\Sigma^{-1}$（上から）と、X の周辺だけの情報量（下から）で挟まれる。どちらも n に比例する |
| $w_0, w$ | 完全データでは $\Theta(n^2)$（ペアの数） | K によらない |
| Σ, σ_Y² | Σ は $\Theta(n)$、σ_Y² はペアの数 | K によらない |

- **K に依存する罰則は $(dK-K(K-1)/2)\log n$ で、n はノード数**である。
- $\log\binom n2$ を使ってもよいのは、K によらない w0・w の部分だけで、選択には影響しない。
- この結論は Daudin, Picard & Robin (2008) の SBM の ICL と同じ考え方である。そこでは、クラスの割合に $\log n$、接続確率に $\log\{n(n-1)/2\}$ を使う。ただし、正確な形は二次資料でしか確認していない（05）。
- したがって、**「Y のペアが多いから $\log(n^2)$」という単純な置き換えは正当化されない**。

**この議論が成り立たない点**
- (0) Y がノードを結合するので、$\ell_{\rm marg}$ は i.i.d. な項の和ではない。Schwarz の議論の前提は、そのままでは満たされない。
- (0′) F の情報量の下限に使った「X の周辺の情報量」は、回転を除いて正則であることが前提である。Bernoulli-X では、これが未解決である（P7・U1）。
- (i) 特異点では $J_n$ が退化し、$p_b/2$ は学習係数に置き換わる。
- (ii) $O(1)$ の項（家族ごとの情報量の定数、事前分布）は K に依存し、有限の n では無視できない。
- (iii) ブロック間の交差項。
- (iv) 潜在変数 Z は周辺化で消えているので、その「オッカム因子」は $\ell_{\rm marg}$ の中に入っている。これは罰則ではない（04 §3）。

### 5.2 X と Y で情報の増え方が違うとき（DERIVED + HYPOTHESIS）

- 本研究の観測：X は $nd$ 個（d 固定なら n に比例）。Y は最大 $n(n-1)/2$ 個のペアで、ペアは z を共有するので、周辺化すると独立ではない。
- K に依存するのは F だけなので、θ 側の罰則の速さは X に支配される（§5.1）。
- 一方、**Y はノードごとの事後精度 $a_i$ を n に比例して大きくする**。精度行列の Y 側の項は $(w^Y)^2\sum_{j\ne i}V_Y(\eta_{ij})z_jz_j^\top$。
- この効果は罰則ではなく、Z の周辺化（04）と、C_Q のエントロピー項（03）に現れる。
- **n が増える設定と d が増える設定は区別が要る**（HYPOTHESIS）。
  - d が増えると、X 由来の事後精度と F の数（罰則 $(d-K)\log n$）が同時に増える。
  - n が増えると、Y 由来の事後精度が増える。
  - 過去の d-sweep（§21）と n×d（§22）でも、2ΔH_Lap は、d が増えると減り（d=15→60）、n が増えても減った（d=15 では非単調）。これは OBSERVED だが、原因は分離されていない。

### 5.3 観測マスク（DERIVED、発見的）

- Y のマスクは、w0・w のブロック（K によらない）と、Z の事後精度（$\sum_{j\in\mathcal O_i}$ が観測されたペアだけになる）に影響する。K に依存する罰則の速さ（F：n）には影響しない。
- X に欠測がある場合、列 l の情報量は観測された行の数 $n_l$ に比例する。そのため、F の行 $f_l$ の罰則は $K\log n_l$ に近くなる（回転の差し引きは列をまたぐので、厳密な分配はない）。
- **当てはまりの項は、観測されたペアだけで評価する**。現行の実装はそうしている（`calc_Q_dual_strict_exp` の `obs_upper`、`laplace_k_criterion._scores` の `mask`）。

### 5.4 分布族ごとの情報行列の違い（DERIVED）

- 情報行列の形は family ごとに違う。たとえば Poisson-X は $F^\top{\rm diag}(e^{\eta})F$、Gaussian-X は $F^\top\Sigma^{-1}F$。
- しかし、Schwarz の近似ではその違いは $O(1)$ の項に入るので、**パラメータの数による罰則は family によらない**。
- 有限の n で family の違いを反映させたいなら、次の 2 通りがある。
  - (a) θ についても完全な Laplace 近似（$-\tfrac12\log|J_n|$）を使う
  - (b) 05 の sBIC・WBIC 型の考え方を使う
- (a) は、θ の情報行列（Z を周辺化したもの）の計算が必要で、現在の実装にはない（06 の候補 C の拡張）。

---

## 6. Phase B のまとめ

| 問い | 答え | ラベル |
|---|---|---|
| Eq.(26) の L | Eq.(16) の条件付き尤度。p(Z) を含まず、Z の評価点は不明 | FACT |
| Schwarz BIC か | 違う。被積分関数が $\ell_{\rm marg}$ ではなく、Z の nK 個の値の扱いが罰則にない | DERIVED |
| 一致性 | 原論文では証明されていない。本研究でも、どの基準についても証明されていない | FACT / UNRESOLVED |
| 罰則 $(K+1)d-K(K-1)/2$ | F の数から回転を引いたものに、K によらない d を加えたもの。generic な点では正しいが、飽和（d=15 で K ≥ 11）と特異点（K > K0）では正しくない | DERIVED + 数値確認 |
| log n の n | K に依存する部分（F）については、ノード数 n の log が、情報量の速さの議論で支持される。ペアの数は K によらない部分にだけ関係する | DERIVED（発見的。正則性を仮定） |
| Y 側の入れ子 | Gaussian-Y では、過大側の K は誤指定のモデルになる（P3）。Bernoulli-Y は未解決 | FACT（P3）／UNRESOLVED（U5） |
