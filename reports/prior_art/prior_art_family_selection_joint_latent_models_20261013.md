# 先行研究レビュー: 混合観測分布・属性付き関係データの潜在モデル・分布族の選択（2026-10-13）

- 位置づけ: **LITERATURE REVIEW / RESEARCH FRAMING**（Issue #136）。実験・EM・モデル変更なし。
- 比較表: `reports/prior_art/prior_art_comparison_matrix_20261013.csv`（26 本、列は Issue #136 の指定どおり。各行に確認の深さを示す `verification_level` を付けた）
- 目的は novelty を先に決めることではない。既存研究を把握し、修論の差分の置き方と、追加実験の必要性を決めることである。
- 表現の規則: 「世界初」「初めて」「先行研究は存在しない」「novel」とは書かない。**検索した範囲**を必ず添える。

---

## 0. 用語の区別（Issue #136 §3）

本書では次の 5 つを別のものとして扱う。

| 記号 | 意味 | 例 |
|---|---|---|
| **A** | 複数の分布を使える | ExpFam PCA で Gaussian / Bernoulli / Poisson のどれかを選んで使う |
| **B** | 列ごとに異なる分布を指定できる | GLRM の列ごとの loss、mixed-response GLLVM、Fontaine 2025 の属性ごとの GLM |
| **C** | データ型から loss を自動で対応させる | Boolean → logistic loss の mapping |
| **D** | 同じ観測値に複数の候補 family を当て、統計的な基準で選ぶ | Valera & Ghahramani 2017 の列ごとの尤度 mixture の重み |
| **E** | family の不確実性そのものをモデル化する | D をベイズ的に行い、重みの事後分布を持つもの（Valera 2017 は D と E の両方） |

**C は family selection ではない。** たとえば「Boolean の列には logistic loss を使う」は、同じ 0/1 の列に Bernoulli と Poisson を当てて比較することとは異なる。

---

## 1. Search strategy

- **期間・手段**: 2026-10-13 に実施。web 検索（一般検索 30 回。うち extended を 8 回）、一次資料の取得 10 回（出版社ページ・arXiv・PMC・PMLR・proceedings の PDF / HTML）、引用の追跡（各論文の related work の記述からたどった）。
- **データベース**: Google Scholar に相当する一般検索、arXiv、PMLR、NeurIPS proceedings、PMC、Project Euclid、出版社のページ。Web of Science / Scopus は使っていない。
- **主なキーワード**:
  - Stream A: exponential family PCA、deviance matrix factorization、family test、rank
  - Stream B: generalized low rank models、loss per column、heterogeneous data
  - Stream C: generalized linear latent variable model、mixed response、factor number、Laplace / variational
  - Stream D: joint latent space model network node attributes、jointly modeling edges and node attributes、collective matrix factorization、relational data imputation
  - Stream E: automatic discovery of statistical types、likelihood selection latent factor、response distribution selection、family uncertainty
  - Stream F: misspecified response distribution latent variable、categorical treated as continuous、multi-view compensation
- **言語**: 英語の文献のみ。日本語の文献・学位論文は検索していない。

## 2. Paper screening

- 30 回の検索で返ったリンクは重複を含めて約 270 件。そのうちタイトル・要旨・抜粋で内容を確認した記録: **約 80 件**（重複を除いた概数）。
- 比較表に入れた論文: **26 本**（Stream A 3、B 1、C 6、D 10、E 4、F 1、親研究 1）。
- 除外したもの: 画像・時系列・GNN による補完（属性の観測分布を確率モデルとして扱わないもの）、一般の GLM の family 選択の教材、潜在構造を持たないモデル。
- **確認の深さ**（表の `verification_level`）:
  - 本文を読んだもの: Valera 2017（pp.1–4）、Fontaine 2025（PMC 全文）、Lv 2025（§2）、Udell 2016（§1–8）、Klami 2014（txt 全文）、Wang & Carvalho 2023（要旨・§1, 4, 5 の抽出）。
  - それ以外は要旨・出版社のページ・検索の抜粋だけである。要旨だけの論文については、細部（K の選び方など）を「not confirmed」と書いた。

---

## 3. ExpFam PCA / 指数型分布族の低ランクモデル（Stream A）

- **Collins, Dasgupta, Schapire (2001, NIPS)**: PCA の二乗誤差を指数型分布族の対数尤度（Bregman 距離）に置き換え、自然パラメータを低ランクに制約する。行列全体で**1 つの family**であり、family は利用者が決める（A 型）。rank は固定。
  本研究の X 側 `η_il = f_l^T z_i` は、この「自然パラメータの低ランク構造」の系譜にある。
- **Liu, Dobriban, Singer (2018, AoAS) ePCA**: 指数型分布族のノイズのもとでの部分空間の推定。誤った family で fit した場合の解析は確認できなかった。
- **Wang & Carvalho (2023, EJS) Deviance matrix factorization**: GLM の deviance による行列分解。
  - 「**選んだ family の妥当性を一般化 Hosmer–Lemeshow 検定で調べる**」ことと、「**rank を最大 eigengap で決める**」ことを扱う。
  - ただし family は**行列全体で 1 つ**であり、検定は「その family で良いか」の適合度検定である。候補の中から列ごとに選ぶもの（D 型）ではない。
  - family と rank は**逐次的**に扱う。X と Y の結合モデルではない（ネットワークのデータも 1 つの行列として分解する）。

**まとめ**: A 型（複数の分布を使える）は確立している。family の適合度検定と rank の決定を組み合わせた研究もある（DMF）。ただし行列全体の単位である。

## 4. GLRM（Stream B）

**Udell, Horn, Zadeh, Boyd (2016, FnT ML)**（本文 §1–8 を確認）:

- 列ごとに異なる loss（quadratic、Huber、logistic、hinge、Poisson、ordinal、categorical）を共通の低ランク構造で扱う。**B 型（列ごとの指定）は既存**である。
- loss は「データ型を尊重して選ぶ」ものとして例示されている（§5）。§8 には明示的に「**The loss function should usually be chosen by a domain expert**」とある。
  したがって、GLRM の列ごとの loss は C 型（データ型との対応づけ、または分析者の選択）であり、**候補の loss を統計的に比べて選ぶ D 型ではない**。
- rank k と正則化は、held-out の要素を使った交差検証、自由度による AIC/BIC 型の基準、scree などで選ぶ（§8）。
- network の関係 Y を同じ因子から生成する形は確認できなかった。

## 5. GLLVM（Stream C）

- **Moustaki & Knott (2000, Psychometrika)**: 観測変数ごとに異なる指数型分布族を持つ潜在特性モデルと、ML による推定。**本研究の X 側のモデルの族（mixed per-column ExpFam）は古くから存在する。**
- **Niku et al. (2017, JABES; 2019, MEE `gllvm`)**: count・biomass のための GLLVM を Laplace 近似や変分近似で推定する。
  family（Poisson / NB / Tweedie など）は**モデル全体**で指定し、利用者が残差や情報量規準で比べる（gllvm の例では NB と Poisson を情報量規準で比較）。自動の列ごとの選択ではない。
- **Korhonen, Nordhausen, Taskinen (2024, WIREs)**: GLLVM の計算法（Laplace、VA、EM、MCMC）の総説。本研究の C_Lap（Laplace 近似）や MC-EM の位置づけに使える。
- **Cárdenas-Hurtado, Moustaki, Chen, Marra (2025, Psychometrika) GLVM-LSS**: 指数型分布族を超える分布を扱う。測定分布の異なるモデルを**情報量規準で比較できる**と述べている。ただしモデル全体の比較である。
- **family の指定を避ける別の流れ**: Murray, Dunson, Carin, Lucas (2013, JASA) の Gaussian copula 因子モデルは、周辺分布を extended rank likelihood で半パラメトリックに扱い、**family を選ぶ必要自体をなくす**。family selection の競合手法として扱うべき方向である。

**まとめ**: mixed per-column の ExpFam 潜在変数モデルは確立している。factor 数の選択（情報量規準・交差検証など）も標準的である。
family の比較は「分析者がモデル全体を情報量規準で比べる」形では行われている。しかし、**列ごとの候補から自動で選ぶ手続きは、GLLVM の文献の中では今回確認できなかった**。

## 6. ネットワーク + 属性の結合潜在モデル（Stream D）

| 研究 | X（属性） | Y（関係） | Z を共有 | X の family | family の選択 | K の選択 |
|---|---|---|---|---|---|---|
| Hoff, Raftery, Handcock 2002 | 共変量として固定 | Bernoulli（logistic） | — | — | — | — |
| Fosdick & Hoff 2015 | 確率変数（Gaussian） | network | ✓（検定のあと結合モデル） | Gaussian | なし | 未確認 |
| Cai 2017 | 確率変数（Gaussian） | logistic | ✓ | Gaussian | なし | なし |
| Zhang, Xu, Zhu 2022（JLSM） | 変数ごとの GLM | 内積 logistic | ✓ | 指定 | なし | 未確認 |
| Wang, Paul, De Boeck 2023 | 調査項目 | binary | ✓（人と属性の共通空間） | 指定 | なし | 未確認 |
| **Fontaine, Kang, Zhu 2025（JCGS）** | **属性ごとの GLM `b0j + b_j^T z_u`**（Gaussian / logistic） | **logit = α_u + α_v + z_u^T z_v** | **✓** | **分析者が属性ごとに指定** | **なし** | **ELBO** |
| **Lv, Tang, Zhang 2025（arXiv）** | **正準型 ExpFam `γ_j + β_j^T z_i`**（Gaussian, Bernoulli、混合型は応用で） | **logit = α_i + α_i' + z_i^T z_i'** | **✓** | **分析者が指定** | **なし** | **cumulative ordered spike-and-slab 事前分布（真の次元への事後集中を理論的に示す）** |
| Crenshaw, Zhang, Onnela 2026（arXiv） | binary 共変量 | 疎なネットワーク | ✓ | binary | 「covariate selection」は**どの共変量を使うか**の選択で、family の選択ではない | 言及なし |
| **Singh & Gordon 2008（CMF）** | 属性の行列 | エンティティ間の関係の行列 | ✓（エンティティの因子を共有） | **行列ごとに Bregman / ExpFam** | なし | 固定 |
| **Klami, Bouchard, Tripathi 2014（Bayesian CMF）** | 行列ごとに Gaussian / Bernoulli / count | gene-gene などの関係の行列 | ✓（共有因子と固有因子） | **行列ごとに指定**（「link functions can be different for different observation matrices」） | なし | ARD / group sparsity |

**まとめ**:

- 「属性 X と関係 Y の両方を確率変数とし、共有の潜在変数 Z から生成する」モデルは**既存**である（Fosdick & Hoff 2015、Zhang 2022、Fontaine 2025、Lv 2025、CMF）。
- **X を列ごとの指数型分布族、Y を内積型の Bernoulli とし、K を選ぶ**モデルも既存である（Lv 2025、Fontaine 2025）。
- **Y 側も指数型分布族にし、行列ごとに異なる尤度を持たせる**形は、CMF（Singh & Gordon 2008、Klami et al. 2014）にある。ただし CMF の Y はエンティティ間の一般の関係行列で、family は行列ごとに分析者が指定する。
- これらの研究では、**family はいずれも分析者が指定**していた。

## 7. 分布族の自動選択（Stream E、最重要）

**Valera & Ghahramani (2017, ICML)**（本文 pp.1–4 を確認）:

- 低ランクの潜在特徴モデル `p(X | Z, {b_d}) = Π_d p(x^d | Z, b^d)` の上で、**各属性の尤度を候補の型の mixture** `Σ_ℓ w_ℓ^d p_ℓ(x^d | Z, b_ℓ^d)` とする。
  重み w^d に Dirichlet 事前分布を置き、潜在表現と同時に MCMC で推論する。
- 候補: 連続（real / positive real / interval）、離散（categorical / ordinal / count）。連続か離散かは**単純な規則で事前に絞る**。これは本研究の support-aware な候補の制限と同じ発想である。
- count は、ガウスの疑似観測を台へ写す変換で扱い、正準 Poisson の尤度ではない。binary は categorical と ordinal が同等になると述べており、0/1 の列で Bernoulli と Poisson を区別するという問題設定ではない。
- K は固定で、K = 1..10 で結果が頑健であることを示している。**K は選んでいない。**
- network の関係 Y はない。

→ **D 型（同じ列に複数の候補を当てて選ぶ）と E 型（family の不確実性のモデル化）は、X 単独の潜在特徴モデルでは既存である。** 本研究の family selection の最も近い先行研究である。

**Vergari et al. (2019, AAAI) ABDA**: sum-product network の上で、変数ごとの尤度の辞書から「statistical data type and likelihood discovery」を行う。D / E 型は既存だが、低ランクの潜在位置モデルではない。

**型は指定するもの**（family selection ではない）: Valera et al. (2020, JMLR) の GLFM、Nazabal et al. (2020, Pattern Recognition) の HI-VAE は、型を与えて mixed-type の尤度で推定する。B 型である。

**検索で見つからなかったもの**: network の関係 Y と共有 Z を持つモデルの中で、属性の family を候補から選ぶ研究。

## 8. family + K の同時選択

| 研究 | family | K / rank | 同時か |
|---|---|---|---|
| Wang & Carvalho 2023（DMF） | 行列全体で 1 つの family の**適合度検定** | 最大 eigengap | **逐次**（family の妥当性を確認してから rank） |
| Valera & Ghahramani 2017 | 列ごとの mixture の重みで推論 | **固定**（頑健性を確認） | なし |
| Lv 2025、Fontaine 2025 | **指定** | 事前分布 / ELBO で選ぶ | なし |
| Klami 2014 | 行列ごとに指定 | ARD で自動 | なし |
| Valera 2020（GLFM） | 指定 | IBP で自動 | なし |

**今回検索した範囲では、列ごとの family の選択と潜在次元 K の選択を、同じ X–Y 共有潜在モデルの上で一緒に扱う研究は確認できなかった。**
近いのは、X 単独で family を選ぶ（K は固定）Valera 2017 と、X–Y の共有モデルで K を選ぶ（family は固定）Lv 2025 / Fontaine 2025 であり、両者は別々に存在する。

## 9. 誤指定の文献（Stream F）

- **Rhemtulla, Brosseau-Liard, Savalei (2012, Psychological Methods)**: 順序変数を連続として扱う CFA（本研究の B 型の誤指定に近い）。
  カテゴリが 5 未満では、連続として扱うと loading と標準誤差に偏りが出る。カテゴリが多ければ許容できる。**誤指定の影響が条件に依存する**ことの古典的な例である。
- **Valera 2017**: 適切な型を推論すると、「連続は Gaussian、離散は categorical」とするより、データへの当てはまり（尤度）が大きく良くなると報告している。ただし潜在構造の回復の指標ではない。
- **Murray et al. 2013**: family の誤指定を避けるための半パラメトリックな代替である。
- **結合モデルで「片方のデータが他方を補う」**:
  - Fontaine 2025 は、結合モデルの利点は観測された情報が弱いときに最も大きいと報告している（補完の文脈）。
  - Zhang 2022、Crenshaw 2026 も、ネットワークの情報で潜在位置の推定と補完が良くなることを示している。
  - ただしこれらは**誤指定ではなく情報の追加**の話である。
- **見つからなかったもの**: 属性の family を誤指定したときの潜在位置の回復への影響が、関係データの観測量によってどう変わるかを、同じ dataset の対比較で調べた研究。0/1 を Poisson で fit した場合の低ランクの部分空間の回復を扱う研究も、今回の検索では確認できなかった。

---

## 10. 最も近い先行研究（5 本）

**1. Lv, Tang, Zhang (2025, arXiv:2509.18580)**
- 似ている: X は列ごとの正準 ExpFam、Y は内積型の logistic、Z を共有、**潜在次元を選ぶ**（理論つき）。本研究の X / Y の構造と K 選択の両方に最も近い。
- 違う: family は分析者が指定し、選択しない。Y は Bernoulli だけ。X に切片、Y に次数の異質性を持つ。推論は MCMC。
- 本研究が加えようとしていること: 同じ型のモデルで、family の候補の中からの選択と、family の誤指定の影響の特徴づけ。

**2. Fontaine, Kang, Zhu (2025, JCGS)**
- 似ている: 属性ごとの GLM（Gaussian / logistic）、内積型の Bernoulli ネットワーク、Z を共有、K を ELBO で選ぶ、欠測を posterior predictive で補完する。
- 違う: family は指定（count・categorical は未実装）。誤指定は扱っていない。変分推論である。
- 本研究が加えようとしていること: count（Poisson）を含む候補からの選択と、Y の観測量と誤指定の影響の関係。

**3. Valera & Ghahramani (2017, ICML)**
- 似ている: **列ごとに候補の尤度から型を推論する**（D / E 型）。候補を support で事前に絞る。低ランクの潜在構造を共有する。
- 違う: X 単独で network Y がない。K は固定。count は変換による疑似尤度である。Bayesian mixture の重みで、情報量規準による選択ではない。binary の列に Bernoulli と Poisson のどちらを使うかは問題にしていない。
- 本研究が加えようとしていること: family の選択を、共有 Z を通じて関係 Y の情報も使う設定に置くことと、K の選択と一緒に扱うこと。

**4. Klami, Bouchard, Tripathi (2014, ICLR)** / Singh & Gordon (2008, KDD)
- 似ている: 属性の行列と関係の行列が因子を共有し、**行列ごとに異なる ExpFam の尤度**（Y 側も含む）を使う。次元は ARD で自動に決まる（Klami）。
- 違う: family は行列ごとに指定し、選択しない。列ごとではなく行列ごとである。MAP や VB による行列分解である。
- 本研究が加えようとしていること: 列ごとの family と、候補からの選択。

**5. Wang & Carvalho (2023, EJS) DMF**
- 似ている: family の妥当性と rank の決定の**両方**を、指数型分布族の低ランクモデルで扱う。
- 違う: 行列全体で 1 つの family。選択ではなく適合度の検定。逐次的。X–Y の結合モデルではない。
- 本研究が加えようとしていること: 列ごと・X–Y 共有の設定での同時の扱い。

## 11. Already known（既に知られていること）

| 問い | 答え | 根拠 |
|---|---|---|
| Q1. mixed family は既存か | **はい** | Moustaki & Knott 2000、GLRM、GLLVM、CMF、Valera 2020、HI-VAE |
| Q2. per-column family は既存か | **はい**（指定する形で） | GLRM（列ごとの loss）、mixed-response GLLVM、Fontaine 2025、Lv 2025 |
| Q3. 同じ列に複数の候補 family を当てて統計的に選ぶ研究はあるか | **はい（X 単独で）** | Valera & Ghahramani 2017（mixture の重み）、ABDA 2019。行列全体の適合度検定は DMF 2023 |
| Q4. network Y と属性 X を共有 Z でモデル化しながら family を選ぶ研究はあるか | **今回の検索範囲では確認できなかった** | 共有 Z のモデル（Fosdick & Hoff、Zhang 2022、Fontaine 2025、Lv 2025、CMF）はいずれも family を指定していた |
| Q5. family の選択と K の選択を同じ結合モデルで扱う研究はあるか | **今回の検索範囲では確認できなかった** | DMF は単一の行列で逐次的、Valera 2017 は K 固定、Lv / Fontaine は family 固定 |

## 12. Possible gap（今回の検索範囲での差分）

**今回の検索範囲（§1）では、次の 3 つを同時に満たす研究は確認できなかった。**

1. 属性 X（列ごとの指数型分布族）と関係 Y を、共有の潜在変数 Z から生成する確率モデルである。
2. 各属性の列について、support で絞った Gaussian / Bernoulli / Poisson などの候補 family から、データに基づいて family を選ぶ。
3. その family の選択を、潜在次元 K の選択と同じモデルの上で扱う。

また、**属性の family の誤指定が潜在構造の回復に与える影響が、関係 Y の観測量によってどう変わるか**を、同じ dataset の対比較で特徴づけた研究も確認できなかった。

**注意（過大評価を避けるために）**:

- 本研究の 2. と 3. は、現時点では **prototype の機能**である（family 選択は lineage E、K 選択は C_Q / C_Lap の prototype）。一般的な優位性や K の一致性は確立していない。
- Categorical の family は実装していない。
- 差分は「新しいモデルの族」ではなく、**既存の部品（共有 Z の結合モデル、列ごとの family の選択、K の選択）の組み合わせと、その振る舞いの特徴づけ**にある。

## 13. Uncertainty（まだ不確かなこと）

1. **Bayesian の family 選択を network に拡張した研究**が、統計の雑誌の中にある可能性がある。今回は英語の一般検索と主要な proceedings だけを使った。Web of Science / Scopus による系統的な検索と、Valera 2017・Lv 2025・Fontaine 2025 の cited-by を全件確認することは、していない。
2. Zhang 2022、Wang 2023、Fosdick & Hoff 2015 は要旨だけで、K の選び方の細部は確認していない。
3. Lv 2025 は arXiv（v2、2026-06）であり、査読版で内容が変わる可能性がある。混合型の属性の詳細は補遺にあり、確認していない。
4. GLLVM の実務で行われる「family を情報量規準で比べる」作業が、列ごとに自動化されたパッケージの機能として存在するかは確認していない。
5. multi-view / multimodal の深層学習（VAE 系）で、modality の誤指定を他の modality が補う現象を扱った研究は、深くは検索していない。

## 14. M1 の位置づけ

#134 の M1 の結果は次のとおり（系列 E-consistent、固定条件 n = 75、d = 12、K = 3、Bernoulli-Y）。
dense の平均 ΔRMSE_Z は +0.020（15/20）、sparse は +0.066（17/20）、interaction は +0.046（15/20 が正）。

文献との関係:

- **方向は既存の知見と整合する**。
  - 誤指定の影響が条件に依存すること（Rhemtulla 2012）。
  - 適切な尤度を使うと当てはまりが良くなること（Valera 2017）。
  - 結合モデルでは、片方の情報が弱いときにもう片方が効くこと（Fontaine 2025、Zhang 2022）。
- **今回の検索では、同じ形の対比較は見つからなかった**。その形とは、0/1 の属性を Bernoulli と Poisson で fit して潜在位置の回復を比べ、その差が関係 Y の観測量でどう変わるかを見るものである。
- したがって M1 は「新しい理論」ではなく、**既存の知見の方向を、本研究のモデル（X–Y の共有 Z）と support-compatible な誤指定について、固定条件で定量化した characterization** として位置づける。
- **言い方の注意**: M1 の観測量の操作は「観測ペアの mask」であり、edge 密度ではない。sparse では correct でも Z の回復が弱い（Gram 相関 0.34）。

## 15. M2 の判断

**判断: B — 誤指定の種類による違いが重要なので、M2（Bernoulli → Gaussian）を 1 本追加する価値がある。**（実行はしない）

理由:

- 文献で最も一般的な「family を気にしない」実務は、**連続（Gaussian / 二乗誤差）として扱うこと**である。
  - GLRM の quadratic loss の既定。
  - Rhemtulla 2012 の「categorical を continuous として扱う」問題。
  - Valera 2017 の比較対象である「連続は Gaussian」。
  - 審査や先生から最も出やすい質問は「単に Gaussian で扱ったらどうなるのか」である。
- M1 は support-compatible な Poisson 化（A 型）しか見ていない。#130 にある Gaussian 化（B 型）の証拠は、generator と系列が違い、M1 と対になっていない。
- 同じ data・mask・seed で Gaussian の arm を加えれば、「誤指定の種類によって影響が違うか」を 1 つの設計の中で答えられる。
  - Gaussian の影響が Poisson より大きければ、「family を選ぶ意味」の主張を強める。
  - Gaussian でも小さければ、「dense ではどれでも大差ない」という限界を正直に書ける。
  - どちらの結果でも、修論の主張の書き方が変わる。
- 規模は小さい（新しい arm 1 つ × 20 replicate × 2 regime。正しい arm を M1 から再利用できるかどうかは、別の experiment Issue で設計時に決める）。

ただし、§16 の **framing の決定を M2 より先に行う**べきである。M2 は書き直した主張の補強として位置づける。

## 16. 先生への説明（advisor-facing）

1. **調べた範囲**: 指数型分布族の PCA、GLRM、GLLVM、ネットワーク + 属性の結合潜在モデル、分布の自動選択、誤指定の 6 系統で、約 80 件をスクリーニングし、26 本を比較表にまとめました（英語の文献、2026-10-13 時点）。
2. **既にあること**:
   - 属性を列ごとに異なる分布で扱うこと（GLRM、GLLVM）。
   - ネットワークと属性を共通の潜在位置で同時にモデル化すること（Fontaine 2025、Lv 2025 など）。後者は潜在次元の自動選択まで含みます。
   - 列ごとに候補の尤度から分布の型を自動で推論すること（Valera & Ghahramani 2017）。ただしネットワークのない、属性だけのモデルです。
3. **したがって**: 「X と Y の分布を指数型分布族に拡張した」こと自体は、既存研究とかなり重なります。修論の差分としては弱いです。
4. **今回の検索範囲で見つからなかったこと**:
   - 属性とネットワークを共通の Z でモデル化しながら、各属性の分布を候補から選び、さらに潜在次元も選ぶ研究。
   - 属性の分布を間違えたときの潜在構造への影響が、ネットワークの観測量によってどう変わるかを同じデータで比べた研究。
5. **提案**: 修論の主題を「指数型分布族への拡張」から、**「関係データと属性を共有する潜在モデルで、属性の分布の選択がいつ重要になるか（関係データが少ないときに効く）」**へ置き直すことを提案します。そのうえで、「単に正規分布で扱った場合」との比較（M2）を 1 本加えると、主張がはっきりします。

---

## DECISION

## **C: CLOSE_PRIOR_ART_REQUIRES_REFRAMING**

RESEARCH_MASTER の現在の貢献の書き方は、「先行研究の Y = Bernoulli、X = Gaussian の固定を、X 側・Y 側で独立に指定できる指数型分布族へ拡張した」というものである。
この拡張は、Lv 2025 / Fontaine 2025（列ごとの ExpFam の属性 + 共有 Z + K の選択）と CMF（行列ごとの ExpFam の尤度で X と関係を共有）に、かなりの部分が重なる。

最も近い先行研究は特定でき、書き直した後の差分（§12）は検索範囲つきで述べられる。
しかし、それには**主張の置き方を変える必要がある**。主な変更点は次の 2 つである。

- 「拡張」ではなく「共有 Z の下での family の選択と、K の選択との組み合わせ」を主題にする。
- 「Y の観測量による誤指定の影響の変化」を特徴づけとして位置づける。

この判断は合否ではなく、修論のストーリーを良くするためのものである。
主張の書き直し（RESEARCH_MASTER の記述を含む）は Human の判断であり、本 Issue では canonical docs を変更していない。

### Recommended next step

1. （Human）先生と framing を確認する。§16 の 5 点を使う。
2. 確認できたら、M2（Bernoulli → Gaussian、M1 と同じ data・mask・seed）を別の experiment Issue として設計する。
3. （任意）§13-1 の不確実性を下げるため、Valera 2017・Lv 2025・Fontaine 2025 の cited-by を系統的に確認する。

---

## 付録: 一次資料として本文を確認した箇所

| 論文 | 確認した内容 |
|---|---|
| Valera & Ghahramani 2017 | 要旨・§1〜3。列ごとの mixture `Σ_ℓ w_ℓ^d p_ℓ(x^d \| Z, b_ℓ^d)`、Dirichlet 事前分布、MCMC、ガウスの疑似観測による count、K = 1..10 の頑健性 |
| Fontaine, Kang, Zhu 2025 | 内積 logistic のエッジ、属性ごとの GLM、Z の共有、ELBO による K、変分 EM、MCAR / graphical-MAR |
| Lv, Tang, Zhang 2025 | §2 のエッジ・属性のモデル、正準型 ExpFam、「determined by the chosen distribution」 |
| Udell et al. 2016 | §5 の loss の例、§8 の「The loss function should usually be chosen by a domain expert」、rank の交差検証 |
| Klami et al. 2014 | 行列ごとの link、エンティティの因子の共有、ARD、VB |
| Wang & Carvalho 2023 | 要旨（一般化 Hosmer–Lemeshow による family の妥当性の確認、最大 eigengap による rank）、§1, 4, 5 の抽出 |
