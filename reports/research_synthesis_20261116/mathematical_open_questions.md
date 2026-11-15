# 未解決の数理的課題と実装上の制限

- **ラベル**：FACT / DERIVED / HYPOTHESIS / UNRESOLVED。
- **出典**：既存の canonical docs と各 README（commit `f71c04d`）。
- **新しい導出はしていない。** 既存の文書で導かれ、または未解決として記録されている事項を整理し直しただけである。

---

## 1. 本来のモデル（Dual-ExpFam LSM）に関する未解決の課題

### 1.1 潜在次元 K の識別と選択

| ID | 課題 | 現状 | 出典 |
|---|---|---|---|
| K-1 | **Bernoulli-Y での K の識別可能性** | UNRESOLVED。強い結果（P2、P3、P8）は Gaussian-Y に限られる。<br>w0=0 の Bernoulli-Y では、一次モーメントが K の情報を持たない（P7） | [`RESEARCH_MASTER.md` §17.3、§17.6](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md) |
| K-2 | K 選択基準の一致性 | 未証明。C_Q と C_Lap のどちらも、有限標本の記述にとどまる（Z9-U3） | §19 |
| K-3 | C_Q − C_Lap ≈ 2H_Lap が成り立つ理由 | 恒等式 C_Q − C_Lap − 2H_Lap = termA + 2R3 は DERIVED。<br>残差が小さいのは termA と R3 の打ち消し合いによるもので、理論的な保証はない | §20 |
| K-4 | 潜在変数モデルの特異性（O(K) 不変性、rank F < K、w = 0） | BIC の導出の前提（正則性）が成り立たない。RLCT は未知 | §17.4 |
| K-5 | 有効標本数の定義（n、n(n−1)/2、nd） | 未確定。`calc_bic_dual` はノード数 n を使う | §17.4 |
| K-6 | MCEM の未収束（L=5、8 反復）が選択に与える影響 | L ≤ 10、16 反復までは検証した。固定点には達していない | §20、§22 |

**識別可能性の命題の前提・対象・未解決**（[`RESEARCH_MASTER.md` §17.2–§17.6](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)、[`true_k_identifiability_hardened_20260904.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/identifiability/true_k_identifiability_hardened_20260904.md)）。

すべて **population**、つまり観測分布そのものが既知である場合の主張である。有限標本での推定や一致性の主張ではない。

| 命題 | 主張 | 前提 | 適用対象 |
|---|---|---|---|
| P1 | ‖f_l‖² = 2 log E[X_l]、f_l·f_m = log(E[X_l X_m]/(E[X_l]E[X_m]))。FFᵀ が population モーメントから復元でき、X 周辺の最小潜在次元が rank(FFᵀ) | canonical、clip のない exp link、rank F = K（generic） | X 周辺の **K^rank**（K* ではない） |
| P2 | S = z_iᵀz_j の積率母関数は M_S(t) = (1−t²)^{−K/2} なので、S のキュムラントは κ₂(S) = K、κ₄(S) = 6K、κ₆(S) = 120K。<br>Y = w0 + wS + ε から κ₂(Y) = Kw² + σ_y²、κ₄(Y) = 6Kw⁴、κ₆(Y) = 120Kw⁶。<br>したがって w² = κ₆(Y)/(20κ₄(Y))、K = κ₄(Y)/(6w⁴)、σ_y² = κ₂(Y) − Kw² が、**1 つの dyad の周辺分布**から決まる（[hardened §9.2–§9.4](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/identifiability/true_k_identifiability_hardened_20260904.md)） | canonical Gaussian-Y、w ≠ 0 | **母集団の 1 dyad の分布**（Y_ij の分布のキュムラント κ_r(Y)）。1 つの観測値から推定できるという意味ではない。w の符号は決まらない |
| P3 | w ≠ 0 の部分 {P ∈ M_K : w ≠ 0} は M_{K+1} と交わらない | canonical Gaussian-Y、w ≠ 0 | 族の非入れ子性。M_K と M_{K+1} は w = 0 の切片で交わる |
| P4 | O(K) の回転・符号反転・座標の置換は観測分布を変えない。N(0, I) の prior が尺度を固定する | — | Z の不定性 |
| P5 | Σ が既知なら、X 周辺の最小潜在次元 = rank(Cov(X) − Σ) | canonical Gaussian-X、rank F = K（generic） | X 周辺 |
| P6 | E[Y^r] < ∞ ⟺ \|w\| < 1/r | canonical Poisson-Y、clip のない link | モーメントの存在（KI-020） |
| P7 | Bernoulli-X、および w0 = 0 の Bernoulli-Y では、一次モーメントが K の情報を持たない（常に 1/2） | Z が 0 対称 | 否定的な結果。joint に移送できない |
| P8 | E[(Y_ij−w0)(Y_ik−w0)(Y_jk−w0)] = w³K なので、w の符号が三角形から決まる | canonical Gaussian-Y、w ≠ 0、n ≥ 3 | w の符号 |

**注意**
- 周辺分布で示した肯定的な識別性は、joint に移送できる。否定的な非識別性は移送できない（§17.3）。
- K* は M-closed（P0 ∈ ⋃_K M_K）の下でだけ定義される。実データでは一般に存在しない（§17.2）。
- 数値の検証（81 行）のうち、独立に反証可能なのは 41 行で、**P3 と P6 には独立な数値の証拠がない**（§17.1）。
- **未解決**：
  - U2：実験で使っている Bernoulli-Y の一般の識別可能性。
  - U5：Bernoulli-Y / Poisson-Y の非入れ子性。
  - U6：どの基準についても一致性。
  - U7：有限標本の rank 閾値。推定した Gram は PSD ですらない。
  - U9〜U12：その他（§17.6）。

### 1.2 推定アルゴリズム

| ID | 課題 | 現状 |
|---|---|---|
| E-1 | 1/2 係数（KI-001） | 採用式は 1/2 なし。原論文の印刷式には 1/2 がある。<br>本文採用の実験（lineage B）は Y 側だけに 0.5 が残っているので、「Newton 方向が全体として正しいとは断定できない」。<br>fixed 系列（C）以降は 1/2 なしで、採用式と一致する。<br>旧系列と fixed 系列の条件ごとの対応表は存在しない（§15） |
| E-2 | Z の点推定（last / mean5 / joint mode）で結論が変わる | E1：dense の last 推定の誤指定効果は再現しなかった。E2a：解の位置はほぼ初期値で決まる |
| E-3 | Procrustes による評価 | 回転だけの Procrustes。切片がないので、Q は可逆な線形変換で不変（AF3 §2.2、DERIVED） |
| E-4 | legacy 系列の η の clip | B と、それを継承する C・D・F では、η の clip と曲率の下限がある。<br>対象は X 側の平均・曲率（`model_dual_expfam.py` L.102、L.113、L.115）、Y 側（`model_expfam.py` L.57、L.71、L.73）、strict-Q の目的関数（`model_dual_expfam.py` L.325–330 など）。<br>そのため、極端な η では目的関数・score・曲率が一致しない。PC-001・PC-002 として測ったのは F のクラスで、B と C については同じコードからの推論である。<br>lineage E で X 側と Y 側の両方を forward に修正した（[`objective_consistency_fix_20260821.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/per_column_family/objective_consistency_fix_20260821.md)）。過去の結果は変えていない |

### 1.3 観測モデルの設計

| ID | 課題 | 現状 |
|---|---|---|
| M-1 | **X に切片がない**（η = fᵀz） | {0,1} だけの真 Poisson 列は、切片なしではほぼ起きない。<br>負の切片を仮定すると選択と交絡する（AF2 Part A、DERIVED と数値計算）。<br>MovieLens の raw-count Poisson の悪化でも、原因の候補の 1 つ（未解明） |
| M-2 | 過分散・過小分散 | Poisson の分散 = 平均。CMP（PR #129、lineage E）と NB（`_nb.py`、lineage D）はどちらも prototype で、本文の family には含まれない |
| M-3 | Categorical | 未実装（KI-005） |
| M-4 | 列ごとの family | **標準モデルでは `family_x` は全列で共通**である。列ごとに変えられるのは prototype（lineage E・F）だけ |

---

## 2. 分布自動選択（AF）に関する未解決の課題

### 2.1 何を基準にすれば「正しい分布」を選べるのか

既存の文書（[AF3 README §2.3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af3_selector_rescoring_20261113/README.md)、[AF8 `mathematical_derivation.md` §0.3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115/mathematical_derivation.md)）の内容を、期待値をとる分布を明示して書き直した。

(3) の恒等式は、AF8 §0.3 が述べている p_0 = p_true の場合を、一般の p_0 に書き直した標準的な等式である。新しい研究上の命題ではない。

#### (1) 母集団の KL 不等式（DERIVED、標準的な結果）

1 つの X 列について、生成の分布を次のようにおく。

```
z ~ N(0, I_K),    x | z ~ p_true(x | f*ᵀz)
```

真の loading f* は固定値である。このとき、任意の候補 m と、**データに依存しない固定の** f について

```
E_{(z, x) ~ p_true}[ log p_true(x | f*ᵀz) − log p_m(x | fᵀz) ]
    = E_{z ~ N(0,I)}[ KL( p_true(· | f*ᵀz) ‖ p_m(· | fᵀz) ) ]  ≥  0
```

が成り立つ。等号は p_m(·|fᵀz) = p_true(·|f*ᵀz) が z について a.s. 成り立つときに限る。

- 期待値は、**z と x の同時分布**についてとる。
- 左辺の第 2 項を f について最大化した値も、真の候補の値を超えない（**期待値をとった後で**最大化する場合）。

#### (2) Q スコアでこの不等式が使える条件（AF3 §2.3 の条件）

Q スコアは次の式で、z^(l) は近似事後分布 q から取る。

```
Q_j(m) = max_f (1/L) Σ_l Σ_i log p_m(x_ij | fᵀ z_i^(l))
```

- **(1) を適用できる条件**：q が**真のモデル（真の分布の割当てと真の θ）の下での厳密な事後分布** p(Z | X, Y) であること。
  - このとき、データ (X, Y) を生成分布から取り、z を事後分布から取ると、各 i で (z_i, x_ij) の分布は (1) の同時分布と一致する。
  - したがって、データについての期待値をとり、f を固定すれば、(1) の不等式がそのまま成り立つ。
- **(1) が保証しないこと**：
  - **個々の有限標本で真の分布が勝つこと**。標本平均には分散があり、各データで f を最大化すると楽観的な偏りが生じる。対数尤度比の分散が KL に比べて大きければ、誤選択は起こりうる。
  - **q が厳密な事後分布ではない場合**。実際の MCEM では、q は incumbent の分布の割当てと推定した θ̂ の下の近似である。AF3 の (a)〜(c)（独立な誤差、別の分布の事後分布、mode）はこの条件が崩れる場合の**説明仮説（HYPOTHESIS）**で、有限標本の定理ではない。

#### (3) 周辺尤度（Bayes 因子）との違い

p(X | m) = ∫ p(X | f, m) p(f | m) df とおき、データ X が分布 p_0 に従うとする。このとき

```
E_{X ~ p_0}[ log p(X | m_true) − log p(X | m_alt) ] = KL(p_0 ‖ p(·|m_alt)) − KL(p_0 ‖ p(·|m_true))
```

この値が ≥ 0 となる**十分条件**は、p_0 = p(·|m_true)、つまり **f が prior p(f|m_true) から生成されていること**である（必要条件ではない）。

- 本研究の生成器は f を固定値で与える（p_0 = p(X | f*, m_true)）。そのため、この符号は一般には保証されない（AF8 §0.3）。
- AF4〜AF8 の基準は、θ_{−j} を固定し、Z を Laplace 近似している。モデル全体の Bayes 因子でもない。

| ID | 課題 | 現状 |
|---|---|---|
| S-1 | **有限標本で、Z が潜在のときの分布の識別可能性** | UNRESOLVED。<br>Poisson と CMP2 の 1 観測あたりの期待 KL は 0.027〜0.061 で小さく、n=75 では列ごとの対数尤度比は数単位にとどまる（AF2 §2.3）。<br>真の Z では AUC 0.993、推定した Z では 0.93〜0.98（AF7）。この差の理論的な説明はない |
| S-2 | Q スコアの自己確認的な偏り | 3 つの機構 (a) 独立誤差による過分散、(b) incumbent の事後分布、(c) mode による過小分散は **HYPOTHESIS**。<br>同じ 20 データで操作して比べたが、有限標本の定理ではない |
| S-3 | profile 型の基準（f を最適化）の判定境界のずれ | 真の分布によらず、中央値 +1.0 だけ CMP2 側にずれる（AF7、事後の観察）。<br>理論的な導出はない |
| S-4 | 積分した基準の prior（測度）依存 | DERIVED：最適化した値 S* は再パラメータ化で不変だが、積分した値 M は prior に依存する。候補間の差は O(1) で残る。<br>非入れ子の 2 候補では、この差は一般に 0 にならない（AF8 §1）。<br>**prior を何に決めれば「正しい」かは、規範的な問題で未解決** |
| S-5 | prior B の Jacobian | (K/2) log(A''_C(0)/A''_P(0)) = −1.00 nat が、平坦極限でも残る。<br>「prior の効果」と「積分の効果」の配分は座標に依存する（AF8-v3 の監査 W3） |
| S-6 | θ_{−j} を固定した基準の位置づけ | AF4〜AF8 の基準は、anchor θ_{−j} に条件づけたもの。モデル全体の evidence でも BIC でもない。<br>θ の推定誤差が主な原因ではないことは AF7 で観察したが（真の θ を anchor にしても AUC 0.968）、理論はない |
| S-7 | Z の Laplace 近似の誤差 | AF6 の TI で、Δ の中央値 −0.12 と見積もった。<br>AF8-v3 の E と E2 は sparse で最大 3.9 ずれ、16 列で符号が逆になる。**厳密な (Z, f) 周辺尤度にどちらが近いかは判定できない**（IS は E2 と同じ被積分関数を使う） |
| S-8 | 自動選択と Z の推定の相互依存 | AF2：Y が疎になると、X の分布選択も悪化した。AF3：選択は Z を作った分布に引き寄せられた。<br>選択と推定を同時に行ったときに、固定点が正しい割当てに収束する条件は未解決 |
| S-9 | 選択の誤りが Z の推定に与える影響の大きさ | AF2：Support-only に対する上積みは小さい（0.005〜0.006）。<br>**誤選択がどれだけ Z を悪くするかは、AF5〜AF8 では評価していない** |

---

## 3. 実装上の制限（FACT）

| ID | 制限 | 影響 | 出典 |
|---|---|---|---|
| I-1 | AF はすべて lineage E の prototype。AF2 以降は未 merge の PR #129 の generic model を使う | 本文には採用できない（Human Gate） | registry の AF の行 |
| I-2 | CMP の正規化定数の級数評価は、η≈51 で項数の上限（400 万）を超える | AF6 で 3 列 FAILED（すべて真 Poisson、重い裾の列）。AF8 では Bessel 経路で回避した | AF6 §4.4、AF8 §2.3 |
| I-3 | PR #129 では 22 fit が失敗した（main 18、pilot 4、[`failures.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/4fa6159/expfam/results/family_selection/expanded_multisupport_20261006_034826/tables/failures.csv)）。PR 本文によると、main の 18 のうち 16 は CMP の候補を incumbent の loading で認証できなかったもの | 候補の開始点の設計が未決定 | [PR #129](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/pull/129) |
| I-4 | AF8 の初版は鞍点に収束した。v2 は Armijo の判定が φ の丸めで止まった | v3 で証明書型の受理に変えて解消した。ただし証明書は局所的な性質で、Z の大域的な多峰性は確かめていない | AF8-v3 `numerical_audit_v3.md` |
| I-5 | 計算量 | AF8-v3 final は 276,470 CPU 秒（8,640 セル）。1 列・1 候補・1 prior あたり数十秒 | AF8-v3 final `numerical_audit.md` |
| I-6 | 統計の単位とデータ数 | AF1〜AF5 は 20 データ、AF6・AF8 は 40 データ。列と regime は独立ではない。多重比較の補正なし | 各 README |
| I-7 | 正解率の分母 | 失敗と判定不能をどう数えるかで値が変わる。AF6 の FAILED を誤りとして数えると、sparse の平均は 0.225 から 0.200 になる | AF6 §7 |

---

## 4. 研究の範囲を決めるうえで重要な点（INTERPRETATION）

- 本来のモデルの未解決課題（K-1〜K-6、M-1）と、AF の未解決課題（S-1〜S-9）は**別の問題**である。
- AF の課題は、**列ごとの family を許す prototype を前提にしている**。
  - 標準モデル（`family_x` が全列で共通）では、列ごとの自動選択という問題はそもそも生じない。
  - 全列共通の family の選択なら、問題の構造は変わる。この場合は AF1 の support gate と同じ種類の判断になる。ただし、これは検証していない。
- S-1 と S-8（Z が潜在であることによる、分布の識別と推定の相互依存）は、数理的に興味深い問題である。
  - 一方で、修士研究の期間に解決できる見込みを示す証拠は、現在のところない。
  - 現在の証拠は「難しさと限界」の記述にとどまる。
