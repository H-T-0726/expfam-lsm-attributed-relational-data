# 13 追加実験 A・B と原因分析・数理的な訂正（2026-11-27）

[← README](README.md)

> **この章の位置づけ**
>
> - 研究結果集の 01〜12 章（2026-11-27 のスナップショット、commit `f8c84d8`）の**後に**行った 2 つの追加実験と、その事後分析をまとめた追補です。
> - 01〜12 章の件数（88 件の台帳、96 項目の数値監査、32 の比較、36 の主張）は**変えていません**。この章の分は、別の追補台帳（[§11](#11-一次資料データ再現)）に分けて記録しました。
> - 実験はすべて系列 **E-scalar**（`DualExpFamLSMConsistent`、prototype）の**探索的な実験**です。fixed 系列（C）の結果ではありません。
> - 結果は `main` に統合されていない作業ブランチにあります。リンクは固定 commit です。
> - 修論本文に採るかどうかは、研究者と指導教員が決めます（Human Gate）。
> - ラベル：**FACT**（一次資料で確認）／**DERIVED**（数式から導出）／**OBSERVED**（実験結果）／**INTERPRETATION**（解釈）／**HYPOTHESIS**（仮説）／**UNRESOLVED**（未解決）。

## 0. 3 行でいうと

1. **A**：Y の観測が少ないと、X の分布を誤指定したときの Z の推定の悪化が大きくなった（この設定で 20/20 データ）。ただし、Y が少ないときは正しく指定しても Z はあまり回復しない。また、Y の予測では対数スコアの逆転が起きた。
2. **B**：Gaussian-X の情報量（期待 Fisher 情報の trace）を Poisson-X と同じまで下げると、Gaussian でも C_Q は K を小さく選んだ（0/10）。C_Lap は多くのデータで $K_{TRUE}$ を選んだ。
3. **原因分析と訂正**：C_Q で K を 1 つ増やすには、X と Y の当てはまりの改善 $2\Delta\ell$ が、1 次元あたり約 456〜496 の固定のコストを超える必要がある。K=8→9 で超えたのは G01 だけだった。

## 1. 用語の確認（初めて読む人向け）

| 区別 | 意味 |
|---|---|
| **Z の回復** と **Y の予測** | Z の回復は、推定した潜在位置 $\hat Z$ が真の $Z$ にどれだけ近いか（Procrustes RMSE_Z、gram 相関）。Y の予測は、学習に使わなかったペアの関係 $y_{ij}$ をどれだけ当てられるか（対数スコア、AUC など）。片方が良くても、もう片方が良いとは限らない |
| **正指定** と **誤指定** | データを作った分布（ここでは Bernoulli）で推定するのが正指定。別の分布（Poisson）で推定するのが誤指定 |
| **分布族の違い** と **Fisher 情報量の違い** | 分布族（Gaussian か Poisson か）が違うと、X が Z について持つ情報量（Fisher 情報）も一緒に変わる。B は、分布族を変えずに情報量だけを下げた条件（GM）を作り、2 つを分けようとした |
| **C_Q** と **C_Lap** | どちらも候補の K ごとに値を計算し、最小の K を選ぶ基準。C_Q は Q 関数（完全データ対数尤度の、E-step のサンプルでの平均）を使う。C_Lap は、Z を積分した尤度をラプラス近似で求める。**C_Q は Schwarz BIC ではなく、C_Lap は厳密な周辺尤度ではない**（§5） |
| **$K_{TRUE}$** と **理論上の真の潜在次元 $K^\ast$** | $K_{TRUE}$ はデータを作るときに使った次元。データを説明できる最小の次元 $K^\ast$ と同じであることは証明されていない（Gaussian-X の $K_{TRUE}=9$ は、識別可能性の既知の十分条件を満たさない。11 章 T6） |
| **fit の成功** と **仮説の支持** | 「600/600 fit が OK」は、計算が数値的に壊れずに終わったという意味。科学的な仮説が支持されたかどうかとは別のこと |
| **plug-in 予測** と **事後予測分布** | plug-in 予測は、1 つの $\hat Z$ を代入して $\sigma(\hat w_0+\hat w\,\hat z_i^\top\hat z_j)$ を計算する。事後予測分布は、Z の不確かさについて予測確率を平均する。この章の予測はすべて plug-in で、事後予測分布は計算していない |

## 2. Experiment A：Y の観測量と X の分布誤指定

### 2.1 研究問い・設計

- **問い**：X の全列が Bernoulli のとき、学習に使う関係 Y が少ないほど、X の分布の誤指定による Z の推定誤差の悪化は大きくなるか。
- **背景**：04 章の M1・MX2 では、Y が少ないほど誤指定の悪影響が大きい傾向が見られた。ただし、どちらも列ごとの family（E-3、prototype）の実験だった。**全列で共通の family を使う構造で、この問いを調べた実験はなかった**（04 章 §9）。

| 項目 | 内容 |
|---|---|
| データ | $n=150$、$d=15$、$K_{TRUE}=3$、X は全列 Bernoulli、Y は Bernoulli。20 の独立データ |
| 比較 | 正指定 Bernoulli-X と誤指定 Poisson-X。同じデータ・同じ mask・同じ seed で、family だけを変えた |
| Y の観測量 | dense（学習ペアの 100%）と sparse（10%、約 1,006 ペア）。test の 10% は共通 |
| 推定 | 2 初期値、L=10、MCEM 10 反復（固定）。protocol は本実験の fit 前に凍結した |
| 主要指標 | $\Delta$ = RMSE_Z(誤指定) − RMSE_Z(正指定)。交互作用 $I=\Delta_{\rm sparse}-\Delta_{\rm dense}$。各データで 2 初期値を平均し、20 データを単位とする |

### 2.2 主要結果（OBSERVED、160/160 fit OK）

| 量 | 平均 | 正のデータ数 | 確認 |
|---|---:|---:|---|
| $\Delta_{\rm dense}$ | +0.012（+0.0116） | 17/20 | TG-A-02 |
| $\Delta_{\rm sparse}$ | +0.223 | 20/20 | TG-A-03 |
| **交互作用 $I$**（primary） | **+0.211**（bootstrap 95% [0.179, 0.242]） | **20/20** | TG-A-04 |

| RMSE_Z の水準 | 正指定 Bernoulli | 誤指定 Poisson |
|---|---:|---:|
| dense | 0.383 | 0.395 |
| sparse | **1.024** | 1.247 |

- **sparse では、正指定でも Z の回復が弱い**：RMSE_Z ≈ 1.02、真の Z との gram 相関 0.240。誤指定は 0.061 で、ほぼ回復していない（TG-A-06）。
  - 元の報告の「0.241」は 0.24046 を丸めた値で、四捨五入すると 0.240 になる。主張は変わらない。
- したがって「誤指定の悪影響が大きい」は、**どちらも回復が弱いなかで、誤指定のほうがさらに悪い**、という意味である。

### 2.3 副次結果：Y の予測（OBSERVED、sparse、Z_last による plug-in 予測）

| 指標 | 正指定 | 誤指定 | 良かった側（20 データ中） | 確認 |
|---|---:|---:|---|---|
| AUC（順位付け、大きいほど良い） | **0.579** | 0.517 | 正指定 20 | TG-A-09 |
| AP（平均適合率） | **0.266** | 0.223 | 正指定 20 | TG-A-10 |
| 対数スコア（大きいほど良い） | −0.573 | **−0.540** | 誤指定 19 | TG-A-11 |
| ECE（小さいほど良い） | 0.100 | **0.070** | 誤指定 20 | TG-A-12 |
| 校正の傾き（1 が理想） | **0.218** | 0.087 | 正指定が 1 に近い 18 | TG-A-14 |
| 予測確率の SD | 0.177 | 0.104 | — | TG-A-15 |
| 定数予測（学習ペアの正例率）の対数スコア | −0.509 | −0.509 | 両モデルとも定数予測より悪い（20/20） | TG-A-13 |

- 4 つの側面を分けて読む。
  - **順位付け**（AUC・AP）：正指定が 20/20 で良い。誤指定の AUC はほぼ 0.5 である。
  - **対数スコア**：誤指定が 19/20 で良い。ただし、Z_last で予測する限り、**両モデルとも定数予測より悪い**。
  - **校正**：指標で結果が分かれる。ECE は誤指定が小さい。校正の傾きは正指定のほうが 1 に近い。
  - **Z の回復**：正指定が良い（§2.2）。
- **INTERPRETATION**：誤指定の予測は確率の広がり（SD）が小さく、定数予測に近い。そのため、外れたときの損が小さく、対数スコアと ECE が良くなった。予測確率の違いがリスクの違いとほとんど対応していない（傾き 0.087）ので、誤指定の確率が「よく校正されている」とは言えない。
- **Z_mean による plug-in 予測**（L=10 サンプルの平均を代入）では、逆転が消えた（TG-A-16）。
  - 正指定が良いデータは 14/20。
  - 定数予測に勝ったデータは、正指定 15/20、誤指定 10/20。
  - ただし Z_mean は縮んでいる（成分の二乗平均 0.60・0.42。Z_last はほぼ 1）。そのため、逆転が消えたことは「予測の広がりが小さくなったこと」とも整合する。**Z_mean 予測は事後予測分布ではない**ので、事後予測分布で逆転が消えるかは分からない。
- dense では、どの指標でも両者の差は小さい（対数スコアの差 +0.0025、正指定が良いデータ 15/20）。

### 2.4 書けること・書けないこと（A）

- 書ける（限定付き）
  - 「この設定で、Y が疎なとき、X の分布誤指定（Bernoulli→Poisson）は Z の回復と予測の順位付けを悪くした」。
  - 「Z_last による plug-in 予測の対数スコアは誤指定のほうが良かった。その予測は確率の広がりが小さく、定数予測に近かった」。
- 必ず併記すること
  - sparse では正指定でも回復が弱い。
  - 両モデルとも定数予測より悪い（Z_last 予測）。
  - Z_mean 予測では逆転が消えた。
- 書けない
  - 「誤指定のほうが予測モデルとして優れている」「誤指定のほうが校正が良い」。
  - 「Y が少ないほど誤指定の悪影響が大きい」という一般法則。MX2 では、誤指定の種類によって向きが逆だった。
  - 「$\hat w$ の縮小が予測改善の原因」。関係は観測しただけで（Spearman −0.48）、介入による検証をしていない。

## 3. Experiment B：Fisher 情報量と K の選択

### 3.1 研究問い・設計

- **問い**：K-exp2 の regime b・$K_{TRUE}=9$ で、Gaussian-X は C_Q で成功し（10/10）、Poisson-X は失敗した（0/10）。この差は、分布族の違いではなく、X が持つ潜在変数の情報量の違いで説明できるか。

| 項目 | 内容 |
|---|---|
| データ | $n=150$、$d=15$、$K_{TRUE}=9$、regime b の信号（$F=sQ$、$s^2=0.5d/K$）、Y は Bernoulli、10 データ |
| 候補 K | 1〜10。各データで 2 初期値、L=10、MCEM 10 反復 |
| 条件 | **G01**：Gaussian-X、分散 0.1／**GM**：Gaussian-X、分散 ≈ 0.76（0.758〜0.773）／**P**：Poisson-X。3 条件で Z・F・Y は同じで、X だけが違う |
| 選択規則 | 2 初期値の基準値を平均し、最小の K を選ぶ。C_Lap は、そのデータの 20 評価のうち 1 つでも非 OK なら INCOMPLETE とし、**分母は 10 のまま** |

**情報量の合わせ方（DERIVED、真の F での期待 Fisher 情報）**
- Gaussian：$I_X = F^\top F/\sigma^2$。
- Poisson：$I_X = F^\top\mathrm{diag}(e^{\lVert f_l\rVert^2/2})F$。$z\sim N(0,I)$ での期待値。
- trace は、G01 が 75、GM と P が約 9.8（9.70〜9.90）（TG-B-08）。GM の分散は、trace が P と一致するように決めた。
- **一致させたのは trace だけ**である。方向ごとの情報は違う（Gaussian は等方、Poisson は条件数 1.11〜1.22）。z ごとの情報の変動、尤度の形、分散を推定するかどうか（Gaussian は推定する）も違う。
  - 条件数の注記：一次 CSV（`information_diagnostics.csv` の本実験の 10 replicate）では 1.11〜1.22 で、`results_report.md` と一致する。`fairness_audit.md` と凍結 protocol B の「1.14〜1.17」は、この 10 replicate の範囲と一致しない。元の文書は書き換えず、ここに記録する。

### 3.2 主要結果（OBSERVED、600/600 fit OK、C_Lap 598 OK・NOT_STATIONARY 2）

| 条件 | C_Q 正解 | C_Q で選んだ K | C_Lap 正解 | C_Lap の内訳 | 確認 |
|---|---:|---|---:|---|---|
| G01 | **10/10** | すべて 9 | 9/10 | INCOMPLETE 1 | TG-B-02、03 |
| GM | **0/10**（過小 10） | 2, 4, 4, 5, 5, 5, 6, 6, 6, 7 | 8/10 | 過小 2（K=8） | TG-B-04、05 |
| P | **0/10**（過小 10） | 4, 5, 5, 5, 5, 5, 5, 6, 7, 7 | 9/10 | INCOMPLETE 1 | TG-B-06、07 |

- 過大選択はどの条件・どの基準でも 0 だった。
- INCOMPLETE の 2 件は、G01 の 1 評価（勾配 3.0e-8）と P の 1 評価（1.9e-7）。どちらも許容値 1e-8 を満たさなかった。許容値の変更や再評価はしていない。
- 2 初期値で選択が食い違ったのは、GM の C_Q で 3/10、P の C_Q で 5/10。それでも、どちらの初期値で選んでも K は 2〜7 で、過小になった。
- GM・P では、選んだ K と K=9 の C_Q の差（190〜595）は、2 初期値間の差より大きかった（TG-B-17）。

### 3.3 推定された分散と情報量（OBSERVED、原因分析 `2f71e61`）

- 推定分散（K=9、10 データの中央値）：G01 0.0949（真値の 0.95 倍）、GM 0.711（0.93 倍）（TG-B-09）。
  - K < 9 では、取りこぼした次元の信号を分散が吸収した値 $\sigma^2+(9-K)\times0.056$ にほぼ一致した。
- 推定後の X 情報の 1 次元あたりの値（K=9）：**G01 9.25、GM 1.37、P 1.17**（TG-B-10）。
  - GM は推定後も P と同程度で、G01 とは約 7 倍の差がある。
  - これは推定モデル上の値で、生成時の真の Fisher 情報とは別のものである。
- Y 側の推定情報（1 次元あたり 5.6〜5.9）は、3 条件でほぼ同じだった。

## 4. C_Q の数式：なぜ情報量が K の選択を左右するか（DERIVED、訂正 `c035df7` を反映）

### 4.1 C_Q の分解

E-step の L 個の MC サンプル $Z_1,\dots,Z_L$ について、X と Y の部分の対数尤度の平均を $\ell(K)$ とおく（strict の定数補正を含む）。

$$
\ell(K)=\frac1L\sum_{l=1}^{L}\bigl[\ln p(X\mid Z_l,\hat\theta)+\ln p(Y\mid Z_l,\hat\theta)\bigr]
$$

- $Q_{\rm strict}=\frac1L\sum_l \ln p(Z_l)+\ell(K)$、$\ln p(Z_l)=-\frac{nK}{2}\ln 2\pi-\frac12\lVert Z_l\rVert_F^2$（$v_z=1$）。
- 実装の `scale_Z` は、L 個のサンプル**全体**で $\frac1L\sum_l\lVert Z_l\rVert_F^2=nK$ にそろえる。$Q_{\rm strict}$ はスケール後のサンプルで計算される（FACT：コード）。$v_z=1$ は全 600 fit で確認した。
- したがって、次が厳密に成り立つ。

$$
C_Q(K) = -2Q_{\rm strict}+p_K\ln n = -2\ell(K) + nK(1+\ln 2\pi) + p_K\ln n
$$

- パラメータ数は $p_K = Kd-\frac{K(K-1)}{2}$（Gaussian-X では $+d$）。$w_0, w$ は数えない（実装の慣行）。
- 全 600 fit で、この式と保存されている `num_params`・C_Q が一致した（TG-B-11）。

### 4.2 K を 1 つ増やす条件（隣接条件）

$\Delta\ell=\ell(K)-\ell(K-1)$、$p_K-p_{K-1}=d-K+1$ より、

$$
C_Q(K)<C_Q(K-1)\iff 2\Delta\ell > n(1+\ln 2\pi)+(d-K+1)\ln n
$$

- 右辺は $n=150$、$d=15$ で $425.68+(16-K)\times5.0106$。K=2 で 495.8、**K=8→9 で 460.756**、K=10 で 455.7。
- **データに依らない固定のコスト**である。$425.68K$ の部分は Z の事前密度から来ていて、罰則 $p_K\ln n$ より大きい。
- 270 組（3 条件 × 10 データ × K=2..10）のすべてで、この同値関係と保存 C_Q の上下が一致した（TG-B-14）。
- **訂正の記録**：原因分析（`2f71e61`）の `diagnostic_report.md` 165 行は、左辺を $-2\Delta(\cdot)$ と書いていて、符号が逆だった。コストの数値（約 461）は正しかった。訂正は `c035df7` の forward correction で行い、元の報告は書き換えていない（冒頭に訂正を指す注記だけを足した）。

### 4.3 K=8→9 で必要な改善と実際の改善（OBSERVED）

保存 C_Q と罰則から、$2\Delta\ell$ は**近似なしに**復元できる。各データで 2 初期値を平均し、10 データの中央値をとった（TG-B-12、TG-B-13）。

| 条件 | $2\Delta\ell$（X+Y、中央値） | 必要量 | 必要量を超えたデータ |
|---|---:|---:|---:|
| G01 | **901.83** | 460.756 | 10/10 |
| GM | **311.63** | 460.756 | 0/10 |
| P | **290.30** | 460.756 | 0/10 |

- GM・P の $2\Delta\ell$ の中央値は、K=5→6 で初めて必要量を下回った。
  - GM は 429.1 で、超えたデータは 4/10。
  - P は 428.6 で、3/10。
- G01 は K ≤ 9 で 747〜902 だったが、K=9→10 では 178 に落ちた。
- **注意**：$\ell$ は最後の E-step の MC サンプルでの平均で、観測データの対数尤度ではない。

### 4.4 隣接条件と大域的な argmin の区別

- 選ばれる K は**全候補の中での最小**（大域的な argmin）で、隣り合う K の比較だけでは決まらない。
- K < 9 について、$C_Q(K)-C_Q(9)=\sum_{j=K+1}^{9}\bigl[c_j-2\Delta\ell_j\bigr]$ と書ける（$c_j$ は上の必要量）。途中で一度改善がコストを下回っても、その後の改善が大きければ、より大きい K が選ばれうる。
- 実際に「隣接条件が最初に破れる手前の K」が argmin と一致したのは 29/30 データだった（TG-B-15）。
- 例外は GM の 1 データ（r009）。3→4 で隣接条件が破れたが、4→5 では満たされ、argmin は 5 だった（C_Q の曲線が単峰でない）。

### 4.5 X と Y の寄与は分けられるか

- 原因分析では、Z_last の 1 サンプルで X・Y・Z・罰則の項に分けた。その結果、「Y の項と事前密度の項は 3 条件でほぼ同じで、違いはほぼ X の項」だった。
- ただし、これは**1 サンプルの近似**である。
  - 1 サンプルでは Z の項が定数にならない。
  - 1 サンプル版で選んだ K は、保存 C_Q の選択と 23/30 でしか一致しなかった。
  - 中央値は足し合わせられない。たとえば P の K=8→9 では、4 項の中央値の和は +132.0、1 サンプル合計の中央値は +165.7、厳密な値は +170.5 だった。
- したがって、X と Y の**合計**の改善（§4.3）は厳密に言えるが、**X 側と Y 側の配分は定性的な記述にとどめる**。厳密に分けるには、L サンプルごとの項の保存が必要。

### 4.6 なぜ情報量が影響したと考えられるか（INTERPRETATION）

- C_Q のコストは、1 次元あたり一定（約 456〜496）である。一方、改善 $2\Delta\ell$ は、新しい次元がどれだけ X と Y をよく説明するかで決まる。これは、その次元についてデータが持つ情報量が大きいほど大きい。
- G01 は X 側の情報が大きく（推定後 1 次元あたり 9.25）、改善が必要量を大きく超えた。GM と P は X 側の情報が小さく（1.37・1.17）、改善は Y 側を合わせても必要量に届かなかった。
- 情報量を P と同程度にした Gaussian（GM）が P と同じように失敗した。このことは、この設定では「分布族そのもの」より「情報量」で説明できる、という見方と整合する。
- 07 章の理想化した関係 $C_Q^\ast\approx C_{\rm marg}+2H(\pi)$（#163、DERIVED）では、事後分布が広い（情報が少ない）ほど、C_Q は大きい K に厳しくなる。今回の結果はこの向きと一致する。ただし、この関係は厳密な事後分布での近似で、今回の MCEM の値で定量的に確かめたものではない。
- **因果の証明ではない**。trace 以外の違い（方向ごとの情報、尤度の形）が影響していないとは言えない。

## 5. C_Lap の扱い

**数式**（DERIVED、実装と照合済み、TG-B-16）

$$
\ln p(X,Y\mid\hat\theta,K)\approx \Phi(\hat Z)+\tfrac{nK}{2}\ln 2\pi-\tfrac12\ln|H|,\qquad
C_{\rm Lap}=-2\ln p(X,Y\mid\hat\theta,K)+d_K\ln n
$$

- 各記号の意味
  - $\Phi(Z)=\ln p(Z)+\ln p(X\mid Z)+\ln p(Y\mid Z)$（strict の定数を含む）
  - $\hat Z$：$\Phi$ の同時モード
  - $H=-\nabla_Z^2\Phi(\hat Z)$：$nK\times nK$
  - $d_K=Kd-K(K-1)/2$
- 展開すると $C_{\rm Lap}=-2\ln p(X,Y\mid\hat Z,\hat\theta)+\lVert\hat Z\rVert^2+\ln|H|+d_K\ln n$ になる。Z の事前密度の $2\pi$ の項は、ラプラス近似の $(2\pi)^{nK/2}$ と打ち消し合う。
  - そのため C_Lap には、C_Q の固定コスト $nK(1+\ln 2\pi)$ に当たる項がない。
  - 代わりに、データに依存する $\lVert\hat Z\rVert^2$ と $\ln|H|$ がコストになる。
- 勾配と $H$ の式は、unique pair の Y の対数尤度を微分した式と一致した。1/2 は付かず、採用式と一致する。

**C_Q との違いと注意**
1. **厳密な周辺尤度ではない**。ラプラス近似の誤差の大きさと向きは測っていない（11 章 T3）。
2. **$\hat\theta$ は MCEM の推定値**で、C_Lap の目的関数について最適化されたものではない（T4）。
3. **単一のモード**での近似で、多峰性は扱わない。$\hat Z$ には `scale_Z` の正規化がかかっていない（C_Q のサンプルとは Z の扱いが違う）。
4. **罰則の数え方が違う**。C_Q の $p_K$ には Gaussian-X の分散 $d$ 個が入るが、C_Lap の $d_K$ には入らない（差は G01・GM で 15、P で 0）。
   - この差は K に依らないので、選ばれる K は変わらない。
   - ただし、**C_Q と C_Lap の絶対値、および family の違う条件間の基準値は比べられない**（FC-TG-10）。
5. **INCOMPLETE は分母から除かない**。除くと、正解率が過大になる（09 章 FC-31）。
6. 今回、C_Lap も GM の 2/10 で過小選択した。#165 の強信号では、50 試行中 20 で過大選択した（07 章）。**C_Lap が正しい基準・常に優れる・過大選択しないとは言えない。**

## 6. 既存研究との関係

### 6.1 分布の誤指定（04 章）

| 実験 | 系列・構造 | Y の量との関係 | この章との関係 |
|---|---|---|---|
| fixed Exp4 | C、全列共通 family、historical generator | 調べていない | 誤指定で RMSE(Z) が悪化（最大 4.34×/9.04×/40.37×）。**数値は比べない**（FC-TG-04、D） |
| M1（#134）・E1（#141） | E-3、列ごとの family（G3/B6/P3）、$n=75$、$d=12$ | B6→P6 で sparse ほど悪化（+0.020 → +0.066） | 向きは A と同じ。比較は記述のみ（FC-TG-03、C） |
| MX2（PR #168） | E-3 | B6→P6 は同じ向き。B6→G6・全 Gaussian では逆 | 誤指定の種類で向きが変わる |
| **Experiment A** | E-scalar、**全列共通 family** | B→P で sparse ほど悪化（+0.012 → +0.223、20/20） | 全列共通の構造で同じ向きが出た。04 章 §9 の空白の一部を埋めた（系列 E-scalar で） |

- 分かったこと：Bernoulli→Poisson の誤指定が、Y が少ないと Z の回復をより大きく悪くするという向きは、列ごとの family（M1・MX2）と全列共通の family（A）の両方で見られた。
- まだ言えないこと：誤指定の種類・sparse 率・信号を変えたときの一般性。fixed 系列（C）での同じ問い。

### 6.2 潜在次元 K の選択（07 章）

| 条件 | 結果 | 実験 |
|---|---|---|
| **正しく選べた** | 方向ごとの信号を固定（K-exp2 regime a、C_Q 200/200）。X の情報が大きい（今回の G01、C_Q 10/10）。fixed Exp1（系列 C、30/30）。Gaussian-Y | 07 章、今回の B |
| **過小選択** | 全体の信号を固定して $K_{TRUE}$ が大きい（K-exp2 A-b-9 の C_Q 0/10 など）。**X の情報を下げた Gaussian**（今回の GM、C_Q 0/10）。$n$ が小さい弱信号 | 07 章、今回の B |
| **過大選択** | 原論文本文どおりの強信号（#165、C_Q 21/50・C_Lap 20/50）。条件付き尤度型の基準（すべての条件） | 07 章 |

- **情報量の役割（今回新しく分かったこと）**：K-exp2 では、Gaussian-X と Poisson-X の差は「family の差」と「情報量の差」が重なっていた。今回、同じ Z・F・Y で X の分散だけを変えると、C_Q の成否が入れ替わった（G01 10/10 → GM 0/10、評価 A の対応比較）。
- **C_Q の数式（今回明確にしたこと）**：07 章 §5 の「K だけで決まる項」$nK(1+\ln 2\pi)$ を使うと、K を増やす条件が $2\Delta\ell$ と固定のコストの比較として厳密に書ける（§4.2）。過小選択は、この比較で改善が届かなかったことにあたる。
- **C_Q と C_Lap の差**：#164・K-exp2 の C_Lap 比較と同じく、C_Q が失敗した条件で C_Lap は多くが $K_{TRUE}$ を選んだ。理由の候補は、C_Lap にはデータに依らない固定コストがないこと（§5）。ただし C_Lap の正しさは示していない。
- **K-exp2 との数値の比較**：今回の P（C_Q 0/10、C_Lap 9/10）と K-exp2 A-b-9（C_Q 0/10、C_Lap 6/10）は、別のデータである。C_Q の差の再現は言えるが、C_Lap の正解数を並べて比べない（FC-TG-09、C）。

## 7. 比較の公平性（09 章の A〜D 基準）

| 比較 | 評価 | 理由 |
|---|---|---|
| A：同じデータでの正指定 vs 誤指定 | **A** | family だけが違う |
| A：dense vs sparse | **A（限定付き）** | 学習 Y の量だけを変えたが、MC・初期化の経路も変わる |
| A：Z_last 予測 vs Z_mean 予測 | **B** | 同じ fit だが Z_mean は縮む。どちらも事後予測ではない |
| A と M1・MX2 | **C** | 系列・列構成・$n$・$d$ が違う |
| A と fixed Exp4 | **D** | 系列・生成器・操作が違う |
| B：G01 vs GM | **A** | Z・F・Y・seed が同じで、X の分散だけが違う |
| B：GM vs P | **B** | trace だけをそろえた。方向ごとの情報・尤度の形・分散推定が違う |
| B：同じ fit での C_Q vs C_Lap | **B** | θ̂ は C_Lap の停留点ではない。INCOMPLETE は C_Lap 側だけ |
| B と K-exp2 本実験 | **C** | データ・乱数の使い方が違う |
| family の違う条件間で基準値の絶対値を比べる | **D** | 正規化・罰則の項数が違う |
| 厳密な $2\Delta\ell$ と 1 サンプルの X・Y 項 | **B** | 合計は厳密、分離は近似 |

詳細は [`data/addendum_20261127_fairness.csv`](data/addendum_20261127_fairness.csv)。

## 8. 限界

1. **系列**：すべて E-scalar（prototype）。fixed 系列の結果ではない。
2. **探索的**：A・B の protocol は fit 前に凍結したが、confirmatory evidence ではない。原因分析と訂正は、結果を見た後の事後分析である。
3. **規模**：A は 1 つの信号設計・1 種類の誤指定・1 つの sparse 率。B は 1 つの $K_{TRUE}$（9）・1 つの信号規則・10 データ。
4. **収束**：MCEM は 10 反復の固定予算。Q の履歴は保存しておらず、収束は判定できない。最後の相対変化の中央値は 2e-4〜1.9e-3、Q の減少回数は 0。
5. **事後予測分布**：MC サンプルを保存していないので計算できない。
6. **X と Y の寄与の配分**：1 サンプル近似で、厳密には分けられない。
7. **理論**：$K_{TRUE}=K^\ast$ は未証明。どの基準の一致性も未証明。C_Lap の近似誤差は未測定。
8. **評価の主体**：作成・分析・点検はすべて同じ AI（Claude）の作成プロセスと別プロセスで行った。**人間による第三者査読ではない。**
9. **実データ**：人工データだけの結果である。実データへの一般化は言えない。

## 9. 確認の種類（成功率としてまとめない）

それぞれ確認した対象が違うので、合算して「成功率」とは言わない。

| 確認 | 結果 | 何を確認したか | 一次資料 |
|---|---|---|---|
| 本実験の独立集計 | 25/25 | runner を import しない別スクリプト。fit 数・ID が凍結 manifest と一致、再実行なし、C_Q = −2Q + p ln n、NPZ の SHA-256、対応する seed・データ、A の $I$・$\Delta$、B の各条件・基準の正解数、過去の結果が不変 | `1ef9096` `independent_tally/` |
| 原因分析スクリプト内の検査 | 11/11 | 再構成したデータの hash、NPZ の SHA-256、held-out 対数スコア・RMSE_Z の再計算、1 サンプル項の和 = strict Q。うち 2 項目は値を記録するだけの検査 | `2f71e61` `consistency_checks.csv` |
| 原因分析の別プロセス確認 | 9/9 | 2 初期値平均・対差・データ数の再集計、8 fit の予測を sklearn で再計算、推定分散・情報量の trace を NPZ から再計算、項の差分。うち 1 項目は値を記録するだけ | `2f71e61` `independent_check.csv` |
| 訂正の別プロセス確認 | 17/17 | 定数 $n(1+\ln2\pi)$、パラメータ数の式、C_Q の恒等式、$v_z=1$、正しい不等号と誤った不等号の一致数（270/270・107/270）、2Δℓ の中央値、集計単位、中央値の非加法性、C_Lap の式、A の校正指標の数え上げ、Z_mean の縮小 | `c035df7` `independent_check_correction.csv` |
| **本結果集の追補の再計算** | VERIFIED 37、MISMATCH 1 | この章の主要数値 38 項目を、固定 commit の一次 CSV から `git show` で読んで再計算。MISMATCH 1 は gram 相関の報告値 0.241（正しい丸めは 0.240）で、主張は変わらない | [`data/addendum_20261127_primary_results.csv`](data/addendum_20261127_primary_results.csv) |

- どれも、作成者と同じ AI のスクリプト・別プロセスによる確認で、人間の第三者査読ではない。
- 01〜12 章の 96 項目の数値監査（VERIFIED 92・RECOMPUTED_ONLY 4）とは別の集計である。分母を合算しない。

## 10. 修論で主張できる範囲（推奨。決定は Human Gate）

| 区分 | 内容 |
|---|---|
| 書ける（限定付き） | A：この設定で、Y が疎なとき、Bernoulli→Poisson の誤指定は Z の回復と順位付けを悪くした（I = +0.211、20/20。AUC・AP 20/20） |
| 書ける（限定付き） | B：X の情報量を下げた Gaussian（GM）は、Poisson と同じく C_Q で過小選択した（0/10）。G01 は 10/10 |
| 書ける（この実装条件での導出） | C_Q の Z の事前密度の項は $nK(1+\ln2\pi)$ になり、K を増やす条件は $2\Delta\ell>n(1+\ln2\pi)+(d-K+1)\ln n$ |
| 書ける（限定付き） | K=8→9 の X+Y の改善は G01 901.83・GM 311.63・P 290.30、必要量 460.756 |
| 必ず併記 | E-scalar（prototype）・探索的。sparse では正指定でも回復が弱い。両モデルとも定数予測に劣る（Z_last 予測）。Z_mean 予測では逆転が消える。trace だけの一致。C_Lap は近似で INCOMPLETE あり |
| 書けない | 「誤指定のほうが予測・校正が良い」／「事後予測分布でも同じ」／「Poisson 分布そのものが原因」「分布族は無関係」／「C_Lap が正しい」／「C_Q は Schwarz BIC」／「収束した」／「X と Y の寄与の正確な配分」／実データ・他の条件への一般化／新規性の断定 |

詳細は [`data/addendum_20261127_claims.csv`](data/addendum_20261127_claims.csv)。

## 11. 一次資料・データ・再現

**一次資料**（すべて未 merge の作業ブランチの固定 commit）

| 内容 | リンク |
|---|---|
| A の凍結 protocol | [`frozen_protocol_A.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/frozen_protocol_A.md) |
| B の凍結 protocol | [`frozen_protocol_B.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/frozen_protocol_B.md) |
| A・B の結果報告 | [`results_report.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/results_report.md)（`1ef9096`） |
| A・B の公平性監査 | [`fairness_audit.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/fairness_audit.md) |
| A の fit ごとの結果 | [`expA/run/per_fit.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/expA/run/per_fit.csv) |
| A の主要結果 | [`expA/analysis/primary_results.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/expA/analysis/primary_results.csv) |
| B の fit ごとの結果 | [`expB/run/per_fit.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/expB/run/per_fit.csv) |
| B の主要結果 | [`expB/analysis/primary_results.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/expB/analysis/primary_results.csv) |
| B の情報量の診断 | [`information_diagnostics.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/expB/tech_check_v2/information_diagnostics.csv) |
| 本実験の独立集計 | [`independent_tally.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/independent_tally/independent_tally.csv) |
| 原因分析の protocol（計算前） | [`analysis_protocol.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/7447ef0/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/analysis_protocol.md)（`7447ef0`） |
| 原因分析の報告（訂正前の原文） | [`diagnostic_report.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/2f71e61/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/diagnostic_report.md)（`2f71e61`） |
| 原因分析の限界 | [`analysis_limitations.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/2f71e61/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/analysis_limitations.md) |
| 予測の分解 | [`prediction_diagnostics.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/2f71e61/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/prediction_diagnostics.csv) |
| 推定パラメータ・情報量 | [`parameter_diagnostics.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/2f71e61/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/parameter_diagnostics.csv) |
| 数理的な最終訂正 | [`forward_correction.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/c035df7/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/forward_correction_20261127/forward_correction.md)（`c035df7`） |
| 厳密な X+Y の改善量 | [`cq_exact_increment_summary.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/c035df7/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/forward_correction_20261127/cq_exact_increment_summary.csv) |
| A の校正指標を分けた表 | [`a_calibration_measures_separated.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/c035df7/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/forward_correction_20261127/a_calibration_measures_separated.csv) |
| C_Lap の式と保存値の照合 | [`clap_formula_check.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/c035df7/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/forward_correction_20261127/clap_formula_check.csv) |
| 訂正の別プロセス確認 | [`independent_check_correction.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/c035df7/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/forward_correction_20261127/independent_check_correction.csv) |
| 図（A の対差と交互作用） | [`figA_delta_and_interaction.png`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/1ef9096/expfam/results/k_selection/two_gaps_20261127/expA/analysis/figures/figA_delta_and_interaction.png) |
| 図（A の予測の分解） | [`figA_prediction_diagnostics.png`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/2f71e61/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/figures/figA_prediction_diagnostics.png) |
| 図（B の分散・情報量・項） | [`figB_variance_information_cq_terms.png`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/2f71e61/expfam/results/k_selection/two_gaps_20261127/diagnostics_20261127/figures/figB_variance_information_cq_terms.png) |

- fig B の項分解のパネルは、1 サンプルの近似による値である（§4.5）。
- NPZ（fit の状態）は git 管理外。SHA-256 は `1ef9096` の `states_sha256_manifest.csv` にある。

**追補データ**（この結果集の中。01〜12 章の 5 つの台帳 CSV とは別）

| ファイル | 内容 |
|---|---|
| [`data/addendum_20261127_primary_results.csv`](data/addendum_20261127_primary_results.csv) | 主要数値 38 項目の再計算（`verify_addendum_20261127.py` が一次 CSV から計算） |
| [`data/addendum_20261127_inventory.csv`](data/addendum_20261127_inventory.csv) | 追加の 4 記録。**新しい実験は TG-A・TG-B の 2 件**。原因分析（TG-DIAG）と訂正（TG-CORR）は同じ fit の分析で、実験として数えない |
| [`data/addendum_20261127_fairness.csv`](data/addendum_20261127_fairness.csv) | 11 の比較の公平性評価 |
| [`data/addendum_20261127_claims.csv`](data/addendum_20261127_claims.csv) | 10 の主張と根拠 |

再現手順（repository root で実行。`1ef9096`・`2f71e61`・`c035df7` が fetch されている必要があります）：

```bash
python docs/research_results/scripts/verify_addendum_20261127.py   # data/addendum_20261127_primary_results.csv
python docs/research_results/scripts/build_addendum_20261127.py    # 追補の台帳 3 種（監査者が書いた記録）
```

- **`EXPERIMENT_REGISTRY.md` は更新していない。** A・B の行は、まだどのブランチの registry にも追記されていない。追記するかどうかと、どのブランチで追記するかは研究者が判断する。
- K-exp2 の registry の行は実験ブランチにだけあり、本結果集のブランチ（main 由来）にはない。この結果集のブランチに持ち込むと、統合時に食い違いが起きうる。
