# 属性分布族の自動選択と潜在次元選択基準の再検討

> 📌 **このページについて**
> 研究進捗の共有用ページです。結果はすべて **固定した人工データ条件** での観測で、実装は **experimental prototype（lineage E）** です。
> 修論本文に採用するかは未決定です（MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN）。
> 記法: ▶ と [TOGGLE START]〜[TOGGLE END] は Notion の toggle block、`> 💡` などは callout block に変換する想定です。

---

## 0. 今回どこまで進めたか

1. 指数型分布族へ拡張したモデルでは、**どの属性列にどの分布を使うかを人が指定**していた。
2. そこで、列ごとに分布族を自動で選ぶ仕組みを、**連続潜在変数の MCEM の上で定式化・実装**した。
3. さらに、**分布の選択と潜在次元 K の選択を 1 本の手順に統合**した。
4. その過程で、K を選ぶのに使っていた**既存の基準 C_Q を再検討**し、Schwarz BIC ではない（Q 型・ICL 型の基準である）ことを整理した。
5. Z を積分した evidence の **Laplace 近似を動機とする新しい基準 C_Lap** を作り、人工データで 2 つの基準を比べた。

> 💡 **ポイント**
> 「分布を扱えるようにする」から一歩進んで、「分布と K をデータから選ぶ」ことに取り組んだ。
> その途中で、K を選ぶ物差しそのものが何を測っているのかを確かめる必要が出てきた、という流れです。

> ✅ **今回確認できたこと（一言版）**
> 分布選択の仕組みは易しい固定条件で意図どおり動いた。2 つの K 基準は同じ推定結果の上でも違う K を選び、既存基準の誤りは主に K を小さく選びすぎること（過小選択）だった。

> ❓ **まだ分からないこと（一言版）**
> どちらの基準が一般に良いか、なぜ過小選択が起きるのか、実データで正しい K を選べるか。

---

## 1. これまでの研究

**この節で分かること**: 研究の土台になっているモデルと、それをどう拡張してきたか。

### 1.1 共通潜在構造から X と Y を生成するモデル

対象（ノード）の間の関係 Y（ネットワーク）と、各対象の属性 X を、**同じ潜在ベクトル z_i** から説明するモデルです（Mikawa et al. 2024 の潜在構造モデル）。
z_i は K 次元の**連続**ベクトルで、クラス番号やクラスタ ID ではありません。

$$
z_i \sim \mathcal N(0,I_K)
$$

$$
\eta_{il}^{X}=f_l^\top z_i
$$

$$
\eta_{ij}^{Y}=w_0+w\,z_i^\top z_j \qquad (i<j)
$$

| 記号 | 意味 |
|---|---|
| $z_i$ | 対象 i の K 次元潜在ベクトル（連続） |
| $X$（$x_{il}$） | 対象 i の属性 l |
| $Y$（$y_{ij}$） | 対象 i と j の関係 |
| $f_l$ | 属性 l の loading（K 次元）。X 側に切片はない |
| $w_0,\ w$ | 関係の全体の起こりやすさと、潜在構造の効き方（どちらもスカラー） |

従来モデルでは X は Gaussian、Y は Bernoulli に固定されていました。推定は MCEM（E-step は Laplace 近似によるサンプリング）です。

[FIGURE 1 HERE]
共通潜在構造 Z から X / Y が生成される図
- Figure purpose: 1 つの潜在ベクトル z_i が属性 x_i と関係 y_ij の両方を生むことを示す。
- Figure specification: z_i, z_j（丸）→ x_i（d 列）と y_ij への矢印。矢印に $f_l^\top z_i$、$w_0+w z_i^\top z_j$。詳細は figure spec 文書の Figure 1。
- Source: `CLAUDE.md` §1、`RESEARCH_MASTER.md` §2・§4。
- Caution: z をクラス・クラスタとして描かない。X の切片を描かない。

### 1.2 指数型分布族への拡張

これまでの研究で、X と Y をそれぞれ Gaussian・Bernoulli・Poisson から選べるように一般化しました（Dual-ExpFam LSM）。
どの分布でも「自然パラメータ η」を通して同じ枠組みで扱えます。

$$
p(x\mid\eta)=h(x)\exp\{T(x)^\top\eta-A(\eta)\}
$$

| 項目 | 従来モデル | 拡張モデル |
|---|---|---|
| X の分布 | Gaussian 固定 | Gaussian / Bernoulli / Poisson |
| Y の分布 | Bernoulli 固定 | Gaussian / Bernoulli / Poisson |
| 推定 | MCEM + Laplace 近似 | 同じ枠組みを一般化 |

> 💡 **ポイント**
> この段階で「複数の分布を扱える」ようになった。ただし、どの分布を使うかは人が指定していた。

▶ **補足: 実装系列について**
[TOGGLE START]

- 標準実装では X の分布は全列共通の 1 種類（`family_x` はスカラー）。列ごとに違う分布を指定できるのは prototype（`family_x_list`）だけ。
- 学会予稿の実験は別の実装系列（Y 側に 1/2 係数がある系列）で行ったもの。今回の結果（1/2 なしの系列から派生した lineage E）とは数値を並べない。
  本研究の採用式は 1/2 なし。予稿の 0.5 は Y 側項にだけ掛かり、Z の事前項・X 側項には掛かっていないため、Newton 方向が全体として正しいとは断定できない（原論文の印刷式には 1/2 がある）。

[TOGGLE END]

---

## 2. 今回の課題

**この節で分かること**: 「分布を扱える」ことと「適切な分布を自動で決められる」ことは別の問題で、今回は後者と K の選択に取り組んだ。

> 💡 **今回の問い**
> 1. 各属性列にどの分布族を使うべきか？
> 2. 潜在次元 K はいくつにすべきか？

| 決めるもの | これまで | 今回 |
|---|---|---|
| 各属性列の分布族 | 人が指定（`family_x` / `family_x_list`） | データから選択 |
| 潜在次元 K | 人が指定 / 既存基準で選択 | データから選択（基準そのものも検討） |

### 2.1 分布族は誰が決めるのか

属性列が 0/1 なのか、回数（カウント）なのか、連続値なのかで、適切な分布は変わります。
列が多いと手で決めるのは負担で、指定を誤るとモデル全体の推定に影響します。

### 2.2 潜在次元 K は誰が決めるのか

K は候補 {1, 2, 3, 4, 5} の中から基準の値が最小のものを選びます。候補の範囲自体は人が決めています。
分布を自動で選ぶなら、K も一緒に選ばないと「データから決める」が半分しか実現しません。

> ⚠️ **注意**
> 今回の目標は「分布と K を全部自動にした」ではありません。下で見るように、分布の自動選択が実際に働く範囲は限られています。

---

## 3. 属性分布族を自動選択する

**この節で分かること**: 列ごとの分布をどういう式で比べ、どういう手順で選ぶか。

### 3.1 基本アイデア

列 l ごとに「どの分布族で説明するか」を表す変数 $c_l$ を置きます。

$$
c_l\in\mathcal M_l
$$

$$
x_{il}\mid z_i,c_l=m
\sim
\mathrm{ExpFam}_m\!\left(\eta_{il}=(f_l^{(m)})^\top z_i\right)
$$

$\mathcal M_l$ は列 l の候補集合（§3.3 の support gate が決める）です。
**loading $f_l^{(m)}$ は分布族ごとに別々に最適化**します（link 関数が違えば最適な loading も違うため）。

> 💡 **ポイント**
> 分布選択の考え方は Structural EM 型の分布族選択（E-step の後で「構造」を選び直す）を参考にしました。
> ただし参考にした考え方は潜在クラスモデルが基本形で、本研究の $z_i$ は連続ベクトルなので、**MCEM の事後サンプルを使う形に再定式化**しています。参考資料の手法をそのまま実装したものではありません。

### 3.2 continuous-Z MCEM への再定式化

E-step で得た同じ事後サンプル $Z^{(1)},\ldots,Z^{(L)}$（L = 5）を固定し、
各候補の分布族について loading を別々に最適化してから、列ごとの score を比べます。

$$
Q_l(m)
=
\frac{1}{L}
\sum_{s=1}^{L}
\sum_{i=1}^{n}
\log
p_m\!\left(
x_{il}
\mid
(f_l^{(m)})^\top
z_i^{(s)}
\right)
$$

$$
\hat c_l
=
\arg\max_{m\in\mathcal M_l}
Q_l(m)
$$

サンプルを固定すると、EM の Q 関数は「Z の事前の部分」「列ごとの X の部分」「Y の部分」に分かれ、
Z の事前の部分と Y の部分は分布族の選び方に依存しません。そのため、列ごとの比較からはこれらが厳密に消えます。

> ✅ **今回確認できたこと**
> 連続潜在変数のままでも、事後サンプルを固定すれば列ごとの分布比較を矛盾なく定義できる。

▶ **候補 family で比較する対数確率**
[TOGGLE START]

比較には各分布の**完全な**対数確率を使います。パラメータに依存しない項でも、分布族によって違う項は比較に必要なので落としません。

Bernoulli:

$$
x\eta-\log(1+e^\eta)
$$

Poisson:

$$
x\eta-e^\eta-\log(x!)
$$

Gaussian:

$$
-\frac{(x-\eta)^2}{2\sigma_l^2}
-\frac12\log\sigma_l^2
-\frac12\log 2\pi
$$

- Poisson の $-\log(x!)$ は 0/1 の列ではたまたま 0 になりますが、式としては残します。
- Gaussian の $-\tfrac12\log 2\pi$ も含めます。$\sigma_l^2$ は残差の平均二乗（その候補での最尤値）です。
- 完全同値のときは Bernoulli → Poisson → Gaussian の順で選びます。

[TOGGLE END]

▶ **詳しい数式を見る（Q 関数の分解）**
[TOGGLE START]

反復 t の E-step のサンプラーの分布を $q_t$ とすると、M-step で最大化する量は

$$
Q(\theta,c\mid q_t)=Q_Z(q_t)+\sum_{l=1}^{d}Q_l(c_l,\psi_l\mid q_t)+Q_Y(w_0,w\mid q_t)
$$

と分かれます。$Q_Z$ と $Q_Y$ は $c$ に依存しないので、列 l の比較では $Q_l$ の差だけが残ります。

- $\psi_{lm}=\{f_l^{(m)}\}$（Bernoulli・Poisson）、$\psi_{lm}=\{f_l^{(m)},\sigma_l^2\}$（Gaussian）。
- これは「固定した $q_t$ の下での座標ごとの改善」であり、EM 全体として観測データの尤度が単調に増えることを保証するものではありません。
- $q_t$ は真の事後分布ではなく、MCEM の Laplace サンプラーの分布です。

[TOGGLE END]

### 3.3 support gate

連続値の密度と離散値の確率は尺度が違い、単位を変えると Gaussian の対数密度だけが動きます。
そのため **尤度の大小で比べてよいのは、同じ種類の値（基底測度）を持つ候補どうしだけ**です。
そこで、観測値の形から候補を機械的に絞る規則（support gate）を先に適用します。尤度は使いません。

| 観測された列 | 候補 | 決め方 |
|---|---|---|
| 0/1 のみ | Bernoulli / Poisson | score |
| 2 以上を含む非負整数 | Poisson | gate |
| その他（非整数・負値を含む） | Gaussian | gate |

> ⚠️ **注意**
> Gaussian / Bernoulli / Poisson の 3 分布を、すべて score で横並び比較しているわけではありません。
> 現在の 3 family では、非自明な score 比較は **0/1 列の Bernoulli vs Poisson** です。

▶ **補足: 扱っていない問題**
[TOGGLE START]

- 「カウント列を log 変換して Gaussian で扱うか」は分布の選択ではなく**表現の選択**で、尤度では決められません。今回は扱っていません。
- X 側に切片がないため、分布族の比較には「列の平均水準がその分布族の既定水準（η = 0 での平均）にどれだけ近いか」も含まれます。今回の人工データはモデルからそのまま生成しているので誤指定は生じませんが、この性質は残ります。
- 同じ測度の上の候補を増やす（例: Poisson と別のカウント分布）には新しい実装が必要です。

[TOGGLE END]

### 3.4 実際の family-selection algorithm

探索中に分布を更新し、最後に分布を固定してもう一度ゼロから推定し直します（hybrid 方式）。
**探索時の fit をそのまま報告せず、選ばれた分布を固定して fresh refit した結果を使います。**

```
観測 X, Y
  ↓
support gate で各列の候補を決める
  ↓
E-step で Z のサンプル（L = 5 本）を得る
  ↓
0/1 の列で Bernoulli と Poisson の loading を別々に最適化し、score を比較
  ↓
score が大きい方に分布を更新（選ばれた候補の loading をそのまま使う）
  ↓
（この探索 MCEM を 8 反復）
  ↓
最終反復の分布の割り当てを固定
  ↓
固定した割り当てで MCEM を新しく実行（fresh refit, 8 反復）
  ↓
最終結果（この refit の値を報告する）
```

[FIGURE 2 HERE]
family-selection flow
- Figure purpose: gate で候補を絞り、0/1 列だけ score で選び、最後に固定して refit する流れを示す。
- Figure specification: 上のフローを縦に図示。gate の 3 分岐、探索ループ、固定、refit。詳細は figure spec の Figure 2。
- Source: `family_selection.support_gate`、`run_hybrid_family_selection`。
- Caution: 3 分布すべてを尤度で比べているように描かない。ループが収束まで回るように描かない。

> ⚠️ **注意**
> 選択の差（margin）は記録していますが、「分布を選んだこと」自体のコスト（離散的な探索のコスト）は罰則に入れていません。

▶ **実装との対応を見る**
[TOGGLE START]

| 内容 | 実装 |
|---|---|
| support gate | `expfam/src/experimental/family_selection.py` `support_gate` |
| 完全対数確率の score | 同 `column_log_likelihood` |
| 候補の最適化 | 同 `optimise_column_loading_bfgs`（解析勾配の BFGS、maxiter 2000、gtol 1e-10） |
| 選択規則と margin | 同 `select_from_records` |
| 探索中の分布更新 | 同 `FamilySelectingPerColumnLSM.select_families`（M-step の `calc_F` 内） |
| hybrid（探索 → 固定 → refit） | 同 `run_hybrid_family_selection`（refit は `failure_policy='fail_fast'`） |
| モデル | `model_dual_expfam_consistent.DualExpFamLSMPerColumnConsistent`（lineage E） |

- 候補の収束判定（勾配の最大値 ≤ 1e-8）は診断用の WARNING で、選択を止める条件ではありません。
- retry・データの差し替え・seed の救済はしません。数値が壊れたら止めて記録します。

[TOGGLE END]

---

## 4. 分布自動選択の実験

**この節で分かること**: K を真値に固定した易しい条件で、分布選択の仕組みが意図どおり動くか。

### 4.1 実験条件

真の分布が分かる人工データで試しました。K の選択はしていません（仕組みの確認を K の選択と切り離すため）。

| 項目 | 条件 |
|---|---|
| n（対象数） | 75 |
| d（属性列数） | 12 |
| K_true | 3 |
| K_fit | 3 |
| X | Gaussian×3 / Bernoulli×6 / Poisson×3 |
| Y | Bernoulli（$w_0=-1,\ w=1$） |
| MC samples L | 5 |
| exploration / refit | 8 / 8 反復 |
| datasets | 3 |
| initial family | start_B（0/1 列を Bernoulli から開始）/ start_P（Poisson から開始） |

▶ **実験条件を見る（詳細）**
[TOGGLE START]

- loading の大きさ: f_scale = √2（平均行エネルギー 0.5）、Gaussian 列の分散 1。
- data seed 951001–951003（search 952001–3、refit 953001–3）。dataset ごとにデータを 1 回だけ生成し、2 つの開始点で共有。
- EM 実行 12/12 成功、retry・差し替え・seed 救済 0、独立監査で BLOCKER 0。
- 候補の最適化の収束 WARNING は 64/576 回（すべて数値精度の限界による停止で、反復上限への到達は 0）。WARNING が選択に与える影響は測っていません。
- artifact 上の `convergence_gate = BLOCKED_FOR_PILOT` は過去の判定ラベルで、この実験では診断用の役割に変更済み（technical validity は VALID）。

[TOGGLE END]

### 4.2 結果

support gate により、真の Gaussian 列（3 列）と真の Poisson 列（3 列、2 以上の値を含む）は gate だけで決まりました。
**score で決めたのは真の Bernoulli 列（0/1 のみ、6 列）だけ**で、これを selector の成績として数えます。

$$
\boxed{36/36}
$$

score で決めた真 Bernoulli 列（3 dataset × 2 開始点 × 6 列）のうち、Bernoulli を選んだ数。

| 指標 | 結果 |
|---|---|
| score 決定列で Bernoulli を選択 | **36/36** |
| Bernoulli → Poisson の誤選択 | **0/36** |
| 開始点による違い | なし（**3/3 dataset** で 2 つの開始点の最終割り当てが一致） |
| 最終 margin（−2 × score の単位、Bernoulli 優位） | 45.69〜65.50（36/36 列） |

| 列の種類 | 決まり方 | 件数（3 dataset × 2 開始点） | selector の成績に含めるか |
|---|---|---|---|
| 真 Gaussian（3 列） | gate | 18/18 が Gaussian | 含めない |
| 真 Poisson（3 列） | gate | 18/18 が Poisson | 含めない |
| 真 Bernoulli（6 列） | **score** | **36/36 が Bernoulli** | **含める** |

> 💡 **この結果が意味すること**
> この固定した人工データ条件では、family-selection machinery が開始点によらず意図した family を選択できることを確認した。
> Poisson から始めても、最初の選択で Bernoulli に入れ替わり、以後は変わらなかった。

> ⚠️ **注意**
> この結果だけから「分布自動選択が一般に成功する」とは言えません。

### 4.3 ここまでで分かったこと

> ✅ **今回確認できたこと**
> 仕組みが最後まで動き、易しい条件で 0/1 列の分布を正しく選べた（feasibility の確認）。

> ❓ **まだ分からないこと**
> - 真の分布が Poisson なのに値が 0/1 しかない列（区別が本当に難しいケース）は、この実験では 0 件で未観測。
> - 信号が弱い・n が小さいなど、判別が難しい条件での挙動。
> - 1 条件・3 dataset の結果であること。

---

## 5. 分布選択と K 選択を統合する

**この節で分かること**: 分布と K を一緒に選ぶ手順と、その結果から何が疑問として出てきたか。

### 5.1 joint pipeline

分布と K は互いに依存します（K が変わると loading も変わる）。そこで、**候補の K ごとに分布選択を最初からやり直します**。

$$
K\in\{1,2,3,4,5\}
$$

各 K について独立に:

```
support gate
  ↓
family selection（探索 MCEM）
  ↓
family freeze（割り当てを固定）
  ↓
fresh refit
  ↓
C_Q(K)
```

最後に

$$
\hat K=\arg\min_K C_Q(K)
$$

を選びます（完全同値なら小さい K）。$C_Q$ は既存の Q 型の K 選択基準で、§6 で中身を説明します。

> ⚠️ **注意**
> family の割り当ては K の間で持ち越しません。warm start もしません（全 30 fit で warm start なしを artifact で確認）。

[FIGURE 3 HERE]
K=1..5 joint selection pipeline
- Figure purpose: K ごとに独立に分布選択と refit を行い、C_Q(K) の最小で K を選ぶことを示す。
- Figure specification: K = 1..5 の 5 本の並列レーン（各レーンに Figure 2 の縮小版）→ C_Q(1..5) → argmin。詳細は figure spec の Figure 3。
- Source: `run_joint_family_k_selection.py`。
- Caution: K 間で矢印（warm start）を引かない。2 つの開始点を独立な dataset として描かない。

### 5.2 実験条件

§4 と同じ条件で、**別の 3 dataset** を使いました。K を 1〜5 で選び、開始点は start_B と start_P の 2 つです。

▶ **実験条件を見る**
[TOGGLE START]

| 項目 | 条件 |
|---|---|
| n / d / K_true | 75 / 12 / 3 |
| X / Y | Gaussian×3 / Bernoulli×6 / Poisson×3、Y Bernoulli（$w_0=-1,\ w=1$） |
| L / exploration / refit | 5 / 8 / 8 |
| K 候補 | 1, 2, 3, 4, 5 |
| datasets | 3（data seed 961001–961003。§4 の 3 dataset とは別） |
| 開始点 | start_B / start_P |
| 実行数 | EM 60/60 成功、retry 0、独立監査 BLOCKER 0 |

[TOGGLE END]

### 5.3 結果

| dataset | K_true | selected K | family result（score 決定列） |
|---|---:|---:|---|
| rep1 | 3 | 2 | 6/6 Bernoulli |
| rep2 | 3 | 3 | 6/6 Bernoulli |
| rep3 | 3 | 3 | 6/6 Bernoulli |

数え方の単位を分けると次のとおりです。

| 単位 | 指標 | 結果 |
|---|---|---|
| dataset | K が正解 | **2/3** |
| pipeline path（dataset × 開始点） | K が正解 | 4/6 |
| pipeline path | 分布と K の両方が正解（joint exact） | 4/6 |
| score 決定列（選ばれた K で） | Bernoulli を選択 | **36/36** |

rep1 の C_Q の値:

| K | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| C_Q(K) | 5382.625 | **5333.020** | 5341.604 | 5517.493 | 5737.582 |

K = 2 と K = 3 の差は 8.584 で、K = 2 が選ばれました。

> ⚠️ **注意**
> - 2 つの開始点は同じデータを使い、最終的に同じ割り当てに着いたため C_Q はビット単位で一致しました。path の 6 件は独立な 6 標本ではありません。
> - §4 の 36/36 と、ここでの 36/36 は**別の dataset** の結果です。合算しません。

▶ **全 dataset の C_Q(K) を見る**
[TOGGLE START]

| dataset | C_Q(1) | C_Q(2) | C_Q(3) | C_Q(4) | C_Q(5) | 選ばれた K | 次点との差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rep1 | 5382.625 | **5333.020** | 5341.604 | 5517.493 | 5737.582 | 2 | 8.584 |
| rep2 | 5303.367 | 5193.320 | **5152.722** | 5349.842 | 5531.293 | 3 | 40.598 |
| rep3 | 5583.194 | 5517.173 | **5441.703** | 5629.297 | 5802.094 | 3 | 75.470 |

（3 dataset × start_B。start_P は同一値。全 30 の (dataset, 開始点, K) で score 決定列は 180/180 が Bernoulli。）

[TOGGLE END]

### 5.4 次に生じた疑問

rep1 で K = 2 になった理由を C_Q の中身で見ると、K を 2 から 3 に上げたときの当てはまりの改善 247.43 に対し、罰則の増え方が 256.02 で、そのうち 212.84 は**データによらない一定の項**でした。

さらに、同じ条件で新しい 20 dataset を用意して C_Q だけで K を選ぶと、K = 3 が 17/20、K = 2 が 3/20 でした。
K を 2 から 3 に上げるときの罰則の増え方は **20/20 の dataset で同じ 256.02** で、当てはまりの改善（162.81〜439.54）がこの一定値を超えるかどうかで K が決まっていました。

> 💡 **ここで生じた問い**
> 「そもそも、現在 K を選ぶために使っている C_Q は何を評価しているのか？」

> ⚠️ **注意**
> 「K = 2 が 1 件出たため基準が間違っている」とは考えていません。
> 検討の動機は、(1) 同条件で K = 2 が繰り返し起きたこと、(2) 判定がデータによらない一定の罰則で決まっている構造、の 2 つです。

▶ **補足: 再現性確認の 20 dataset について**
[TOGGLE START]

- 条件は §5.2 と同じ、新しい 20 dataset（seed 971000+r）、start_B のみ、C_Q のみ。EM 200/200 成功。
- 選ばれた K: K=2 が 3/20、K=3 が 17/20（K=1・4・5 は 0/20）。
- 256.02 = 212.84（Z の事前の項の増分）+ 10 ln 75 = 43.17（パラメータ罰則の増分）。
- この 20 dataset は §8 の比較に使う 20 dataset とは**別の seed の組**です。同じ条件でも C_Q の K=3 は 17/20（ここ）と 14/20（§8）のように変わります。

[TOGGLE END]

---

## 6. 既存の K 選択基準 C_Q を調べる

**この節で分かること**: 既存の Q 型基準 C_Q が何を測っていて、なぜ「1 次元あたり一定の項」が入っているのか。

### 6.1 C_Q の式

$$
C_Q(K)
=
-2Q_{\mathrm{strict}}(K)
+
p_K\log n
$$

$Q_{\mathrm{strict}}$ は EM の Q 関数（E-step のサンプル Z での完全データ対数尤度の平均。全正規化定数込み）、
$p_K$ はパラメータ数です。

$$
p_K = Kd-\frac{K(K-1)}{2}+(\text{Gaussian に選ばれた X 列の数})
$$

（Y が Bernoulli の今回の条件。$w_0,\ w$ は数えない。）

### 6.2 C_Q の分解

$$
C_Q
=
D_K+P_Z+P_\theta
$$

$$
D_K=-2(Q_X+Q_Y)
$$

$$
P_Z=-2Q_Z
$$

$$
P_\theta=p_K\log n
$$

| 項 | 意味 |
|---|---|
| $D_K$ | X, Y への当てはまり（E-step サンプルでの平均） |
| $P_Z$ | 潜在変数 Z の事前密度に由来する項 |
| $P_\theta$ | パラメータ数の罰則 |

分解の値は直接計算した C_Q と 1.8e-12 以内で一致しました（§5 の 30 fit）。

### 6.3 P_Z が意味するもの

推定中に Z のサンプルを「全要素の平均二乗が 1」になるように揃え（scale_Z）、事前分散を 1 に固定している（var_z = 1）現在の約束のもとでは、

$$
P_Z
=
nK(1+\log 2\pi)
$$

が厳密に成り立ちます。今回の n = 75 では、

$$
\Delta P_Z
=
75(1+\log2\pi)
\approx
212.84
$$

> 💡 **重要**
> 潜在次元を 1 つ増やすと、P_Z は今回の条件では約 212.84 増える。この増え方はデータによらない。

> ⚠️ **注意**
> これは「P_Z が過小選択の原因」と意味するものではありません。

▶ **詳しい数式を見る（P_Z の導出）**
[TOGGLE START]

$$
Q_Z=\frac1L\sum_{s=1}^{L}\left[-\frac{nK}{2}\log 2\pi-\frac12\lVert Z^{(s)}\rVert^2\right]
$$

scale_Z により $\frac1L\sum_s\lVert Z^{(s)}\rVert^2=nK$ なので、

$$
P_Z=-2Q_Z=nK\log 2\pi+nK=nK(1+\log 2\pi)
$$

- 係数 $1+\log 2\pi$ は「事前分散を 1 とする」という約束に依存します。Z の尺度を σ に変えても観測データのモデルは同じですが、C_Q は $nK\log\sigma^2$ だけずれます。つまり、この 1 次元あたりの増え方は観測データからは決まりません。
- P_Z は誰かが罰則として意図的に入れた項ではなく、Z の事前密度を含む完全データを評価したことで現れる項です。
- scale_Z を外しても同じ値になるかは確認していません。

[TOGGLE END]

### 6.4 Schwarz BIC ではない

| 呼称 | C_Q に使ってよいか |
|---|---|
| **既存の Q 型基準 / Q-based complete-data / ICL-type criterion** | ✅ |
| Schwarz BIC | ❌ |

Schwarz BIC は原則として、Z を積分した**観測データの周辺尤度**を使います。
C_Q は MCEM の**完全データの Q 関数**（Z を積分せず、Z の事前密度を含む）を使うので、Schwarz BIC ではありません。

> 💡 **ポイント**
> C_Q は構造としては「観測データ型の BIC + 2 × 潜在変数の事後エントロピー」と同じ形をしているので ICL 型と呼べます。
> ただし Z が連続なのでエントロピーは尺度に依存し、ICL の理論的性質はそのまま移りません。

▶ **詳しい説明と実装名を見る**
[TOGGLE START]

- 任意の分布 $q(Z)$ について $\log p(X,Y\mid\theta)=E_q[\log p(Z,X,Y\mid\theta)]+H(q)+\mathrm{KL}(q\,\|\,p(Z\mid X,Y,\theta))$。$Q_{\mathrm{strict}}$ は右辺第 1 項の MC 推定で、観測データの尤度とは $H(q)+\mathrm{KL}$ だけ違い、この差は K に依存します。
- $q$ を独立な Gaussian とみなすと $C_Q=[-2\,\mathrm{ELBO}+p_K\log n]+2H(q)$ となり、$2H(q)$ の定数部分がちょうど $nK(1+\log 2\pi)=P_Z$ です。
- 実装（mixed-family）: Q は `eval_utils.calc_Q_dual_strict_exp`、基準は mixed-family implementation: `calc_bic_exp`。CSV / JSON の列名は歴史的に `bic` のままです（呼称は変更しない）。
- 先行研究の印刷された基準（Eq.26）の当てはまり項も z に条件づけた量で、Schwarz BIC が対象とする周辺尤度ではありません。

[TOGGLE END]

---

## 7. Z を積分した evidence を近似する C_Lap

**この節で分かること**: なぜ別の基準を考えたのか、その式と各項の意味、C_Q との違い、実装が正しいことをどう確かめたか。

### 7.1 なぜ別の基準を考えたか

K を選ぶ目的を「**真の K を回復すること**」に置くなら、Z を積分した観測データの尤度（evidence）が対応する量です。
C_Q は Z を積分していないので、Z を積分する版を作りました（Candidate B）。
evidence は潜在変数の尺度の約束を変えても値が変わらない、という性質もあります。

> 💡 **ポイント**
> 検討した候補は 3 つ（原論文型の条件付き基準、Z を積分した evidence の Laplace 近似、ELBO 型）で、
> 「真の K の回復」という目標に合い、今回の規模で計算できるものとして Laplace 近似（Candidate B）を選びました。

### 7.2 C_Lap の式

$$
C_{\mathrm{Lap}}(K)
=
-2
\left[
\ell_X(\hat Z)
+
\ell_Y(\hat Z)
\right]
+
\lVert\hat Z\rVert^2
+
\log|H(\hat Z)|
+
d_K\log N
$$

- $\hat Z$: Z の**同時最頻値**（joint mode。$nK$ 次元をまとめて最適化）
- $H(\hat Z)$: その点での負の Hessian（**$nK\times nK$**。関係 Y が対象どうしを結ぶので非対角ブロックも含む）
- $d_K=Kd-\frac{K(K-1)}{2}+(\text{Gaussian に選ばれた X 列の数})$（今回の条件では C_Q の $p_K$ と同じ数）
- $N=n=75$: working convention（標本サイズとして理論的に確定したものではない）

### 7.3 各項の意味

| 項 | 意味 |
|---|---|
| $-2[\ell_X(\hat Z)+\ell_Y(\hat Z)]$ | joint mode での data fit |
| $\lVert\hat Z\rVert^2$ | latent prior contribution（Z の事前密度の寄与） |
| $\log\lvert H(\hat Z)\rvert$ | Laplace volume（データが Z をどれだけ絞り込むか。データで決まる） |
| $d_K\log N$ | θ のパラメータ罰則 |

> ⚠️ **注意**
> C_Lap は厳密な周辺尤度ではなく、Laplace 近似を動機とする criterion です。

▶ **詳しい数式を見る（実装での形）**
[TOGGLE START]

実装は次の形で計算しています。

$$
\Phi(Z)=\log p(Z)+\log p(X\mid Z,\hat\theta)+\log p(Y\mid Z,\hat\theta)
$$

$$
\ell_{\mathrm{Lap}}=\Phi(\hat Z)+\frac{nK}{2}\log 2\pi-\frac12\log|H(\hat Z)|,\qquad
C_{\mathrm{Lap}}=-2\,\ell_{\mathrm{Lap}}+d_K\log N
$$

事前分散 1 のとき $\log p(Z)$ の定数 $-\frac{nK}{2}\log 2\pi$ と $+\frac{nK}{2}\log 2\pi$ が打ち消し合うので、§7.2 の式と厳密に同じです。

Hessian の構造（$s,\ c$ は X の残差と曲率、$R,\ C$ は Y の残差と曲率）:

$$
H_{ii}=I+F^\top\mathrm{diag}(c_i)F+w^2\sum_j C_{ij}z_jz_j^\top,\qquad
H_{ij}=w^2C_{ij}z_jz_i^\top-wR_{ij}I\ \ (i\ne j)
$$

- $\hat Z$ は減衰付きの Newton 法で求め、勾配の最大値 ≤ 1e-8 と H の正定値を満たさないときは値を返しません（ridge や jitter は加えない）。
- θ̂ は refit の最終値、$\hat Z$ の初期値は refit の Z の推定値です。

[TOGGLE END]

### 7.4 C_Q との違い

| | C_Q | C_Lap |
|---|---|---|
| Z の扱い | Q 関数の上で残す | Laplace 近似で積分 |
| fit の評価点 | 事後サンプルの MC 平均 | joint mode |
| Z に関する項 | $P_Z=nK(1+\log 2\pi)$（一定） | $\lVert\hat Z\rVert^2+\log\lvert H\rvert$（データで決まる） |
| θ の罰則 | $p_K\log n$ | $d_K\log N$（今回は同じ値） |
| 厳密な周辺尤度か | No | No |

> 💡 **ポイント**
> 2 つの基準の主な違いは、(1) Z に関する項が一定かデータ依存か、(2) 当てはまりをどこで評価するか、の 2 点です。

> どちらが一般に優れているかを、この時点で決めるものではありません。

[FIGURE 4 HERE]
C_Q vs C_Lap comparison
- Figure purpose: 2 つの基準を「当てはまり + Z の項 + θ の罰則」で並べ、違う部分を示す。
- Figure specification: 左右 2 本の概念的な積み上げ棒（数値なし）。詳細は figure spec の Figure 4。
- Source: §6.2、§7.2 の式。
- Caution: 実データの数値を棒の高さに使わない。C_Lap を「正解」の色で示さない。

### 7.5 実装確認

実装した評価関数が式どおりかを、EM を回さない小さなテストで確かめました。

| 確認 | 結果 |
|---|---|
| 勾配と同時 Hessian を有限差分と比較 | 約 1e-9 で一致 |
| Laplace が厳密になる場合（Gaussian の X、w = 0）で厳密な周辺尤度と比較 | 1.4e-14 で一致（K = 1, 2, 3 の 3/3） |
| 潜在変数の尺度を変える | 値も K 間の差も不変 |
| 停留しない・H が正定値でない場合 | 値を返さず状態を報告 |
| 実験での評価可否 | 今回の全比較実験で評価可能（例: §8 では 100/100） |

> ✅ **今回確認できたこと**
> 評価関数は式と一致し、今回の規模（$nK\le 375$）で安定に計算できた。

> ❓ **まだ分からないこと**
> Laplace 近似そのものの誤差（非 Gaussian の事後での精度）と、θ̂ が C_Lap の最大化点でないことの影響。

▶ **Candidate B の実装検証の詳細を見る**
[TOGGLE START]

- 有限差分: 小データ（n = 6、K = 2、X = [Gaussian, Bernoulli, Poisson, Gaussian]）。勾配の最大絶対誤差 5.47e-10（Bernoulli Y）/ 8.55e-10（Poisson Y）、Hessian 5.72e-10 / 1.03e-09。テストの許容値は 1e-6。数値は検証 report の測定値。
- 3 dataset の試験運用: 15/15 で評価可能。K=3 を選んだのは C_Lap 3/3、C_Q 1/3。
- 重点サンプリングによる近似誤差の診断（K = 2, 3）: 相対 ESS 0.4〜12% と重みが退化し、厳密な周辺尤度は得られていません。
- θ̂ の停留性: MCEM 8 反復の θ̂ は C_Lap の停留点ではありません。θ を 1 ステップ動かしても最良 K と次点の順序は 20/20 で保たれましたが、多ステップの確認は結論が出ていません（joint mode の計算が許容値のわずか上で止まる場合がある）。

[TOGGLE END]

---

## 8. C_Q と C_Lap を比較する

**この節で分かること**: 同じ推定結果の上で 2 つの基準を計算すると、どの K を選ぶか。

### 8.1 K_true = 3 baseline

これまでと同じ条件（n = 75、d = 12、X は Gaussian×3 / Bernoulli×6 / Poisson×3、Y は Bernoulli）で、**20 datasets** を新しく用意しました。
分布選択と refit は 1 回だけ行い、**同じ fitted models** の上で C_Q と C_Lap の両方を計算しています。

▶ **実験条件を見る**
[TOGGLE START]

| 項目 | 条件 |
|---|---|
| n / d / K_true | 75 / 12 / 3 |
| X / Y | G×3 / B×6 / P×3、Y Bernoulli（$w_0=-1,\ w=1$）、f_scale = √2 |
| L / exploration / refit | 5 / 8 / 8 |
| K 候補 | 1〜5 |
| 開始点 | start_B のみ |
| datasets | 20（data seed 1001001–1001020） |
| 実行数 | EM 200/200 成功、C_Lap 100/100 評価可能、retry 0 |

[TOGGLE END]

### 8.2 結果

| criterion | K=1 | K=2 | K=3 | K=4 | K=5 | 計 |
|---|---:|---:|---:|---:|---:|---:|
| C_Lap | 0 | 1 | **19** | 0 | 0 | 20 |
| C_Q | 1 | 5 | **14** | 0 | 0 | 20 |

> 💡 **ポイント**
> この固定条件では、真の K = 3 を選んだのは C_Lap: 19/20、C_Q: 14/20 でした。
> C_Q の誤選択は under-selection（K を小さく選ぶ）のみで、over-selection は 0/20 でした。

| 同じ dataset の上での対応 | 件数 |
|---|---:|
| 両方 K = 3 | 14/20 |
| C_Lap だけ K = 3 | 5/20 |
| C_Q だけ K = 3 | 0/20 |
| 両方 K ≠ 3 | 1/20 |

### 8.3 読み方

> ⚠️ **注意**
> - 1 つの固定条件・20 dataset での数です。「C_Lap が一般に良い」とは言えません。
> - 19/20 や 14/20 を「回復確率 95% / 70%」とは読みません。同じ条件・別の seed の組では、C_Q の K = 3 は 17/20 でした（§5.4 の toggle）。

---

## 9. X・Y の signal strength を変える

**この節で分かること**: 関係 Y 側・属性 X 側の信号の強さを変えると、2 つの基準の選択がどう動くか。

§8 は 1 条件だけでした。2 つの基準の違いが信号の強さでどう変わるかを、Y 側と X 側で別々に調べました。**2 つの実験は別のデータ系列**なので、同じ表には並べません。

### 9.1 Y 側

§8 と同じ Z・F・X を使い、Y だけを作り直して $w$ を弱く・強くしました。
$w$ を変えると関係全体の起こりやすさも変わってしまうので、**母集団の平均 edge 確率を 0.3314 に揃える** $w_0$ を較正しました（下の表はこの揃えた版）。

| Y signal | $w$ | $w^2K$ | C_Lap: K=3 | C_Q: K=3 |
|---|---:|---:|---:|---:|
| weak | 1/√2 | 1.5 | 17/20 | 1/20 |
| baseline（§8） | 1 | 3 | 19/20 | 14/20 |
| strong | √2 | 6 | 20/20 | 19/20 |

> 💡 **OBSERVED**
> 検討した条件では、Y signal が強くなるほど K = 3 を選ぶ dataset が増えた。弱い条件では C_Q の多くが K ≤ 2 を選んだ。

> ⚠️ **注意**
> 揃えたのは平均 edge 確率だけで、確率の分布全体・飽和までは揃えていません。純粋な Y signal effect とは言いません。

▶ **実際にどの K が選ばれたか（Y 側）**
[TOGGLE START]

平均 edge 確率を揃えた版（各 20 dataset）:

| 条件 | $w_0$ | C_Lap K=1/2/3/4/5 | C_Q K=1/2/3/4/5 |
|---|---:|---|---|
| weak | −0.8781 | 0/3/17/0/0 | 8/11/1/0/0 |
| baseline | −1 | 0/1/19/0/0 | 1/5/14/0/0 |
| strong | −1.1891 | 0/0/20/0/0 | 0/1/19/0/0 |

揃えない版（$w_0=-1$ 固定、各 20 dataset。実現した edge 密度の median は 0.306 / 0.333 / 0.352 と変化）:

| 条件 | C_Lap K=1/2/3/4/5 | C_Q K=1/2/3/4/5 |
|---|---|---|
| weak | 0/4/16/0/0 | 9/10/1/0/0 |
| baseline | 0/1/19/0/0 | 1/5/14/0/0 |
| strong | 0/0/20/0/0 | 0/1/19/0/0 |

- 揃えた weak の 1 dataset では、C_Lap の最良 K と次点の差が 0.20 と非常に小さかった（追加の診断はしていない）。
- $w_0$ は厳密な密度による 1 次元積分で 1e-12 の基準で較正。Z・F・X は 20/20 で §8 と同一。

[TOGGLE END]

### 9.2 X 側

Z と実現した Y を条件間で共有したまま、属性の loading の大きさ（f_scale）だけを変えました。

| X loading | f_scale | 平均行エネルギー | C_Lap: K=3 | C_Q: K=3 |
|---|---:|---:|---:|---:|
| weak | 1 | 0.25 | 20/20 | 11/20 |
| base | √2 | 0.50 | 20/20 | 15/20 |
| strong | 2 | 1.00 | 18/19 | 18/19 |

注:
- strong は 19 completed、1 incomplete（1 dataset で推定の途中に Poisson の値が数値的に溢れて停止。retry せず不完全として記録）。
- strong では C_Lap が 1 件 K = 4 を選びました（over-selection）。
- base 行は名前も条件も §8 と同じに見えますが、**別のデータ**（Z と Y を共有する生成方法）です。§8 の 19/20・14/20 と比べません。

> 💡 **OBSERVED**
> loading を強くすると、C_Q では K = 3 を選ぶ dataset が 11 → 15 → 18 と増えた。C_Lap はどの条件でもほぼ K = 3 だった。

> ⚠️ **注意**
> f_scale は Gaussian 列の信号対雑音比、Bernoulli 列の飽和、Poisson 列の裾を同時に変えます。Gaussian / Bernoulli / Poisson への影響を分離した純粋な属性情報の効果ではありません。
> また、1 件の数値的な停止から「Poisson 列は一般に悪い」とは言いません。

▶ **実際にどの K が選ばれたか（X 側）**
[TOGGLE START]

| 条件 | 完了 dataset | C_Lap K=1/2/3/4/5 | C_Q K=1/2/3/4/5 |
|---|---:|---|---|
| weak | 20/20 | 0/0/20/0/0 | 0/9/11/0/0 |
| base | 20/20 | 0/0/20/0/0 | 0/5/15/0/0 |
| strong | 19/20 | 0/0/18/1/0 | 0/1/18/0/0 |

- 平均行エネルギー = $\mathrm{f\_scale}^2K/d$。Z・loading の向き・Y は 20/20 で条件間で同一。
- EM は 600 回の計画のうち 593 回実行、592 回成功（strong の 1 dataset が停止したため残りは実行していない）。

[TOGGLE END]

### 9.3 何が見えたか

> ✅ **今回確認できたこと**
> 検討した条件では、Y 側・X 側どちらの信号を強めても、両基準とも K = 3 に寄った。変化は C_Q で大きく、C_Lap は多くの条件でほぼ K = 3 のままだった。

> ❓ **まだ分からないこと**
> 信号のどの側面（平均の強さ、確率の分布の形、分布族ごとの寄与）が効いているのか。

---

## 10. 真の潜在次元 K_true を変える

**この節で分かること**: 真の K を変えると 2 つの基準の選択がどう変わるか。そのために信号の強さをどう揃えたか。

### 10.1 なぜ signal を揃える必要があるか

ここまでの実験はすべて K_true = 3 でした。単に K_true を増やすと、
- loading の総量が増える（X 側の信号が強くなる）
- $w\,z_i^\top z_j$ の分散 $w^2K$ が増える（Y 側の信号が強くなる）
- 関係全体の起こりやすさも変わる

ため、「K_true の違い」と「信号の強さの違い」が混ざってしまいます。

### 10.2 matched design

次の 3 つの平均量を K_true の間で揃えました。

| 揃えた量 | 値 |
|---|---|
| average X loading energy（$\mathrm{f\_scale}^2K/d$） | 0.5 |
| Y natural parameter variance | $w^2K=3$ |
| population mean edge probability | 0.3314 |

| K_true | f_scale | $w$ | $w_0$ |
|---:|---:|---:|---:|
| 1 | √6 | √3 | −0.9306 |
| 2 | √3 | √(3/2) | −0.9780 |
| 3 | √2 | 1 | −1 |
| 4 | √(3/2) | √(3/4) | −1.0128 |

各 K_true について **10 datasets**（rep01..rep10）を使いました。K_true = 3 は新しく推定せず、§8 の 20 dataset のうち **rep01..rep10 の部分集合**を再利用しています。

▶ **実験条件を見る**
[TOGGLE START]

- n = 75、d = 12、X = G×3 / B×6 / P×3、Y Bernoulli、L = 5、8/8 反復、K 候補 1〜5、start_B のみ。
- data seed 1001001–1001010（K_true ごとに別のデータ）。K_true = 1, 2, 4 は新しい EM 300/300 成功、C_Lap 150/150 評価可能。K_true = 3 は新しい EM 0。
- $w_0$ は 1e-12 の基準で較正。

[TOGGLE END]

### 10.3 結果

| K_true | C_Lap exact | C_Q exact |
|---:|---:|---:|
| 1 | 10/10 | 10/10 |
| 2 | 10/10 | 9/10 |
| 3（§8 の rep01..rep10） | 10/10 | 7/10 |
| 4 | **8/10** | **1/10** |

each condition: 10 datasets。K_true = 3 の行は §8 の 20 dataset 版（19/20・14/20）とは別の数です。

▶ **実際にどの K が選ばれたか**
[TOGGLE START]

| K_true | C_Lap K=1/2/3/4/5 | C_Q K=1/2/3/4/5 | 計 |
|---:|---|---|---:|
| 1 | 10/0/0/0/0 | 10/0/0/0/0 | 10 |
| 2 | 0/10/0/0/0 | 1/9/0/0/0 | 10 |
| 3 | 0/0/10/0/0 | 1/2/7/0/0 | 10 |
| 4 | 0/0/2/8/0 | 1/5/3/1/0 | 10 |

- over-selection は両基準とも 0/40。
- K_true = 4 の C_Lap は、最良 K と次点の差が median 13.40、最小 1.13 と小さかった（境界に近い）。

[TOGGLE END]

[FIGURE 5 HERE]
K_true vs exact count
- Figure purpose: K_true が大きいほど exact が減り、C_Q で顕著であることを示す。
- Figure specification: 横軸 K_true（1〜4）、縦軸 exact の**件数**（0〜10）。C_Lap（10, 10, 10, 8）と C_Q（10, 9, 7, 1）。詳細は figure spec の Figure 5。
- Source: `expfam/results/matched_k_true_sensitivity/phase9x_20260928/combined/`。
- Caution: 縦軸を確率・% にしない。誤差棒を描かない。K_true = 5 以上へ外挿しない。

### 10.4 解釈

> 💡 **OBSERVED**
> この固定 matched design では、K_true が大きい条件ほど under-selection が増え、特に C_Q で顕著だった。

> ⚠️ **注意**
> - 各条件 10 datasets・1 つの設計の記述です。general recovery probability ではありません。
> - 揃えたのは 3 つの平均量だけで、1 次元あたりの信号は K_true とともに変わっています。純粋な潜在次元の効果とは言いません。
> - C_Lap も K_true = 4 では 2/10 で K = 3 を選び、境界に近づきました。

---

## 11. K=3 と K=4 の境界では何が起きていたか

**この節で分かること**: K_true = 4 の 10 dataset で、K = 4 が K = 3 に勝つかどうかを、基準の成分の釣り合いとして書き直すとどうなるか。

ここでは新しい推定はせず、§10 で保存済みの値を分解しただけです。

$$
\Delta_{34}=C(4)-C(3)
$$

$$
\Delta_{34}<0\ \Rightarrow\ K=4\ \text{が}\ K=3\ \text{より良い}
$$

K を 3 から 4 に上げると loading の数が 33 から 42 に 9 増えるので、パラメータ罰則の増分は両基準とも $9\log 75\approx 38.86$ です。

### 11.1 C_Q

$$
\Delta_{34}^{Q}
=
-\mathrm{fit\ gain}_{Q}
+
212.84
+
38.86
$$

したがって

$$
\mathrm{fit\ gain}_{Q}
>
251.70
$$

なら K = 4 が選ばれます。$\mathrm{fit\ gain}_{Q}=D_3-D_4$（E-step サンプル平均での当てはまりの改善）。**閾値 251.70 はデータによらず一定**です。

| 指標（K_true = 4、10 datasets） | 値 |
|---|---|
| fit gain_Q の median | 164.57 |
| 閾値 | 251.70（一定） |
| margin（fit gain − 閾値）> 0 | **1/10** |

### 11.2 C_Lap

$$
\Delta_{34}^{\mathrm{Lap}}
=
-\mathrm{fit\ gain}_{\mathrm{Lap}}
+
\Delta
\left(
\lVert\hat Z\rVert^2+\log|H|
\right)
+
38.86
$$

$\mathrm{fit\ gain}_{\mathrm{Lap}}$ は joint mode での当てはまりの改善、$\Delta(\lVert\hat Z\rVert^2+\log|H|)$ は体積の増分です。**罰則側は dataset ごとに変わります。**

| 指標（K_true = 4、10 datasets） | 値 |
|---|---|
| fit gain_Lap の median | 254.75 |
| volume increment の median | 203.88 |
| margin（fit gain − 体積の増分 − 38.86）> 0 | **8/10** |

### 11.3 両者の比較

| | C_Q | C_Lap |
|---|---|---|
| fit gain の評価点 | E-step サンプル平均 | joint mode |
| fit gain の median | 164.57 | 254.75 |
| 罰則側の増分 | 212.84 + 38.86 = 251.70（一定） | 体積の増分（median 203.88）+ 38.86（dataset ごとに変わる） |
| K = 4 が K = 3 に勝った数 | 1/10 | 8/10 |

比較のため K_true = 3（§8 の rep01..rep10）で同じ分解をすると、両基準とも 10/10 で K = 3 が K = 4 に勝っていました。

[FIGURE 6 HERE]
K3→K4 threshold balance
- Figure purpose: C_Q は一定の閾値に fit gain が届かない dataset が多く、C_Lap は dataset ごとに閾値が変わることを示す。
- Figure specification: 2 パネル。横軸 dataset（rep01..rep10）、縦軸は −2 × 対数尤度の単位。点 = fit gain、線・棒 = 罰則側の増分。詳細は figure spec の Figure 6。
- Source: `expfam/results/k34_boundary_decomposition/phase9y_20260929/ktrue4_rows.csv`。
- Caution: 「P_Z が原因」と読める注記を付けない。左右の fit gain を同じ量として 1 つの軸で直接比べない。

▶ **dataset ごとの値を見る（K_true = 4）**
[TOGGLE START]

| dataset | C_Lap の K | C_Q の K | fit gain_Lap | 体積の増分 | margin_Lap | fit gain_Q | margin_Q |
|---|---:|---:|---:|---:|---:|---:|---:|
| rep01 | 4 | 2 | 258.5 | 204.6 | +15.06 | 204.3 | −47.44 |
| rep02 | 4 | 2 | 268.6 | 208.6 | +21.11 | 204.9 | −46.81 |
| rep03 | 3 | 1 | 220.0 | 182.2 | −1.13 | 158.9 | −92.77 |
| rep04 | 4 | 3 | 243.9 | 193.3 | +11.73 | 162.0 | −89.66 |
| rep05 | 4 | 3 | 312.9 | 228.0 | +46.04 | 243.7 | −8.01 |
| rep06 | 3 | 2 | 208.2 | 178.9 | −9.58 | 151.7 | −100.01 |
| rep07 | 4 | 2 | 245.5 | 198.4 | +8.25 | 165.7 | −85.97 |
| rep08 | 4 | 3 | 251.0 | 203.1 | +8.96 | 158.4 | −93.25 |
| rep09 | 4 | 4 | 377.1 | 251.7 | +86.58 | 318.0 | +66.31 |
| rep10 | 4 | 2 | 276.3 | 207.0 | +30.45 | 163.4 | −88.30 |

- margin が正なら K = 4 が K = 3 より良い。margin > 0 は C_Lap 8/10、C_Q 1/10。
- 体積の増分の内訳（median）: Δlog|H| 127.60、Δ‖Ẑ‖² 72.76（median どうしの和は体積の median と一致しない）。
- 再構成の誤差は最大 1.6e-12（許容 1e-10）。
- C_Q が K = 1・2 を選んだ 6 dataset は、K = 3 と 4 の比較だけでは説明されません。

[TOGGLE END]

### 11.4 分かったことと分からないこと

> 💡 **分かったこと**
> K = 3 と 4 の判定は、「K を 1 つ増やすことで得られる fit improvement」と「複雑度側の増分」の釣り合いとして記述できた。
> C_Q では複雑度側が一定（251.70）、C_Lap ではデータで変わる体積が複雑度側に入る。

> ⚠️ **分からないこと**
> なぜ fit gain 自体がその大きさになるのか（E-step サンプル平均と joint mode の違い、1 次元あたりの信号、8 反復の MCEM の誤差など）は分かりません。
>
> したがって、「P_Z が under-selection の原因」とは結論していません。

---

## 12. 今回分かったこと

**この節で分かること**: 今回の成果を、証拠の種類のラベル付きで 7 項目にまとめたもの。

### FACT
連続潜在変数の MCEM の上に、列ごとの family-selection mechanism（support gate + 完全対数確率の score + 探索 → 固定 → refit）を定義・実装した。

### OBSERVED
1 つの固定した人工データ条件で、score で決める真 Bernoulli 列を 36/36 正しく選択した（開始点によらず、3/3 dataset で一致）。

### FACT
各 K で family selection をやり直す joint family + K selection pipeline を実装した（1 条件 3 dataset で、K の正解 2/3、score 決定列 36/36）。

### FACT / DERIVED
既存の Q 型基準 C_Q は Schwarz BIC ではなく、Q-based complete-data / ICL-type criterion である。現在の約束のもとで Z の事前の項 P_Z は 1 次元あたり一定（n = 75 で 212.84）。

### FACT
Z を積分した evidence の Laplace 近似を動機とする C_Lap を定義・実装し、評価関数が式と一致することを確認した。

### OBSERVED
固定した人工データ条件では、C_Q の誤選択は主に under-selection であり、matched K_true design では K_true が大きいほど増えた（exact: C_Q 10/10・9/10・7/10・1/10、C_Lap 10/10・10/10・10/10・8/10）。

### OBSERVED
K = 3 と 4 の境界を、fit gain と complexity increment のバランスとして記述できた（K_true = 4 で K = 4 が勝ったのは C_Q 1/10、C_Lap 8/10）。

> ⚠️ **注意**
> OBSERVED の項目はすべて「n = 75、d = 12、G×3 / B×6 / P×3、Y Bernoulli、MCEM 8 反復、条件ごと 10〜20 dataset」の固定条件での観測です。
> C_Lap にも例外があります（K_true = 4 で 2/10 が過小選択、X 側の strong 条件で 1 件の過大選択）。

▶ **claim ledger（主張の境界の全体）を見る**
[TOGGLE START]

**ALLOWED（そのまま書いてよい）**

| 主張 | 根拠 |
|---|---|
| 既存の C_Q は Schwarz BIC ではなく Q-based complete-data / ICL-type | C_Q の理論整理 |
| scale_Z・var_z = 1 のもとで P_Z = nK(1 + log 2π)、n = 75 で 1 次元あたり 212.84 | 同上、分解の artifact |
| 列ごとの分布選択 prototype を定義・実装した | 設計 report、実装 |
| K ごとに分布選択をやり直す同時選択 pipeline を実装した | 実装 |
| C_Lap を定義・実装し、評価関数が式と一致した | 実装検証 |
| 固定した人工データ条件で、2 基準の選択パターンを観測した | §8〜§10 |
| matched K_true design で、K_true が大きいほど過小選択が増えた（特に C_Q） | §10 |
| K3→K4 の差を既存成分に分解でき、K4 が勝ったのは C_Lap 8/10、C_Q 1/10 | §11 |

**QUALIFIED ONLY（限定語つきでのみ）**

| 主張 | 必須の限定 |
|---|---|
| C_Lap はいくつかの固定条件で C_Q より真の K を選んだ dataset が多かった | 固定条件・条件ごと 10〜20 dataset・MCEM 8 反復・start_B のみ。例外を併記 |
| X / Y の信号を強めると、検討した条件では選択が K = 3 に寄った | Y 側は平均 edge 確率だけ揃えた。X 側は Z・Y 共有の設計で strong は 1 dataset 不完全。純粋な信号の効果ではない |

**NOT ALLOWED** は §13 を参照。

[TOGGLE END]

---

## 13. まだ言えないこと・限界

**この節で分かること**: 今回の結果から言ってはいけないことと、その理由になっている限界。

> ⚠️ **NOT ALLOWED（書いてはいけないこと）**
> - C_Lap は一般に C_Q より優れている
> - K-selection 問題を解決した
> - Candidate B（C_Lap）は一致性をもつ（consistent）
> - 選択数（19/20 など）は一般の recovery probability である
> - P_Z が under-selection の原因である
> - 1 次元あたりの信号の希釈（signal dilution）が原因である
> - family selection が一般に成功する
> - 実データでも正しい K を選択できる
> - Poisson は一般に悪い
> - lineage E（今回の prototype）が自動的に修論本文へ採用可能である
> - C_Q を「Schwarz BIC」、C_Lap を「厳密な周辺尤度」と呼ぶ

| カテゴリ | 今回の限界 |
|---|---|
| Data | 固定した人工データ条件（n = 75、d = 12、G×3 / B×6 / P×3、Y Bernoulli）のみ |
| Sample | 条件ごと 10〜20 datasets（分布選択の実験は 3 datasets） |
| Optimization | MCEM 8 反復、開始点 start_B のみ（K 比較） |
| C_Lap | Laplace 近似の誤差は未評価（重点サンプリングの重みが退化） |
| θ | θ̂ は C_Lap の最大化点ではない |
| Distribution selection | 非自明な比較は 0/1 列の Bernoulli vs Poisson のみ。0/1 に見える真 Poisson 列は未観測。分布を選ぶこと自体のコストは罰則に入っていない |
| Real data | 真の K / family が分からず、評価方法も含めて未評価 |
| Theory | 一致性（consistency）は未解決 |

> ❓ **まだ分からないこと**
> - K_true が大きいときの過小選択が、基準の構造（C_Q の一定の P_Z のような項）によるのか、当てはまりの改善の大きさ（評価点の違い・1 次元あたりの信号・有限回の MCEM）によるのか。
> - 別の信号の揃え方（例: 1 次元あたりのエネルギーを固定）で K_true のパターンが変わるか。
> - n・d・分布族の構成を変えたときの一般化。

▶ **未解決の問い（一覧）を見る**
[TOGGLE START]

| # | 未解決の問い |
|---|---|
| U1 | K3→K4 の fit gain がなぜその大きさになるか |
| U2 | 8 反復の MCEM と、θ̂ が C_Lap の最大化点から離れていることの影響 |
| U3 | 漸近的な振る舞い・一致性 |
| U4 | n / d / K_true / 分布族の構成を変えたときの一般化 |
| U5 | 別の信号の揃え方で K_true のパターンが変わるか |
| U6 | 実データでの K 選択の妥当性（評価方法を含む） |
| U7 | 実装の Laplace 近似と厳密な周辺尤度の差 |
| U8 | joint mode の計算が許容値のわずか上で止まる場合の扱い |

いずれも将来の研究課題で、完了した実験の解釈を妨げるものではありません。

[TOGGLE END]

---

## 14. 現在位置

**この節で分かること**: 何が終わっていて、何が未解決・未決定か。

| 項目 | 状態 |
|---|---|
| 指数型分布族への拡張 | 実装済み |
| per-column family selection | prototype 実装 |
| family-selection feasibility | 確認済み（1 条件・3 datasets） |
| family + K joint pipeline | 実装済み |
| C_Q identity | 整理済み（Q 型・ICL 型、Schwarz BIC ではない） |
| C_Lap | 実装・characterization 済み |
| X / Y signal sensitivity | 実施済み（各条件 19〜20 datasets） |
| K_true sensitivity | 実施済み（各条件 10 datasets） |
| K3→K4 arithmetic | 分解済み（原因は未特定） |
| general consistency | 未解決 |
| real-data K validity | 未解決 |
| manuscript adoption | 未決定 |

> 📌 **MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN**

---

## 15. 今後の判断候補

**この節で分かること**: 次に何をするかの候補。どれもまだ決めていません。

> ⚠️ **注意**
> 以下は候補で、決定事項ではありません。どれを選ぶ場合も、新しい実験は事前に問い・条件・指標を固定してから行います。

| # | 候補 | 何が分かるようになるか |
|---|---|---|
| 1 | larger K_true での under-selection の原因を切り分ける | 基準の構造によるのか、当てはまりの大きさによるのか |
| 2 | family selection の難しい条件（0/1 に見える真 Poisson、弱い信号、小さい n） | 分布選択がどこで失敗し始めるか |
| 3 | 同じ measure 上で family 候補を増やす（例: Poisson と別のカウント分布） | Bernoulli vs Poisson 以外での自動選択 |
| 4 | 実データでの評価方法を設計する | 真値が分からない状況で K・分布を評価する方法 |
| 5 | 修論本文への Candidate B の採用判断 | 本文で何を主張するか |

---

## Appendix

### A. 数式一覧

| # | 名前 | 式 | 意味 |
|---|---|---|---|
| E1 | Z の事前分布 | $z_i\sim\mathcal N(0,I_K)$ | 対象ごとの K 次元連続潜在ベクトル |
| E2 | X の自然パラメータ | $\eta_{il}^X=f_l^\top z_i$ | 属性 l。切片なし |
| E3 | Y の自然パラメータ | $\eta_{ij}^Y=w_0+w\,z_i^\top z_j$ | 関係。$w_0,\ w$ はスカラー |
| E4 | family 変数 | $c_l\in\mathcal M_l,\ x_{il}\mid z_i,c_l=m\sim\mathrm{ExpFam}_m((f_l^{(m)})^\top z_i)$ | 列ごとの分布族。loading は分布族ごとに別 |
| E5 | family score | $Q_l(m)=\frac1L\sum_{s}\sum_i\log p_m(x_{il}\mid (f_l^{(m)})^\top z_i^{(s)})$ | 固定した事後サンプルでの列ごとの完全対数確率 |
| E6 | family 選択 | $\hat c_l=\arg\max_{m\in\mathcal M_l}Q_l(m)$ | 完全同値なら Bernoulli → Poisson → Gaussian |
| E7 | K 選択 | $\hat K=\arg\min_{K\in\{1..5\}}C(K)$ | C は C_Q または C_Lap。完全同値なら小さい K |
| E8 | C_Q | $C_Q(K)=-2Q_{\mathrm{strict}}(K)+p_K\log n$ | 既存の Q 型基準（ICL 型。Schwarz BIC ではない） |
| E9 | C_Q の分解 | $C_Q=D_K+P_Z+P_\theta,\ D_K=-2(Q_X+Q_Y),\ P_Z=-2Q_Z,\ P_\theta=p_K\log n$ | 当てはまり + Z の事前の項 + パラメータ罰則 |
| E10 | P_Z | $P_Z=nK(1+\log 2\pi)$ | scale_Z・var_z = 1 のもと。n = 75 で 1 次元あたり 212.84 |
| E11 | C_Lap | $C_{\mathrm{Lap}}(K)=-2[\ell_X(\hat Z)+\ell_Y(\hat Z)]+\lVert\hat Z\rVert^2+\log\lvert H(\hat Z)\rvert+d_K\log N$ | Z を積分した evidence の Laplace 近似 + θ の罰則。厳密な周辺尤度ではない |
| E12 | Δ34（C_Q） | $\Delta_{34}^Q=-\mathrm{fit\ gain}_Q+212.84+38.86$ | fit gain_Q > 251.70 なら K = 4 |
| E13 | Δ34（C_Lap） | $\Delta_{34}^{\mathrm{Lap}}=-\mathrm{fit\ gain}_{\mathrm{Lap}}+\Delta(\lVert\hat Z\rVert^2+\log\lvert H\rvert)+38.86$ | 罰則側が dataset ごとに変わる |

（$\log\lvert H
vert$ は Hessian の行列式の対数です。）

### B. 実験条件一覧

| 実験 | 目的 | 変えたもの | datasets | K 候補 | 基準 | 主な結果 |
|---|---|---|---:|---|---|---|
| 分布選択 | 仕組みの確認 | 開始点 | 3（× 2 開始点） | 3 に固定 | 列ごとの score | score 決定列 36/36 Bernoulli |
| 分布 + K 同時選択 | 統合 pipeline | K、開始点 | 3（× 2 開始点） | 1〜5 | C_Q | K 正解 2/3 dataset、score 決定列 36/36 |
| 再現性確認 | K = 2 は単発か | dataset | 20 | 1〜5 | C_Q | K=3 17/20、K=2 3/20 |
| baseline 比較 | 2 基準の比較 | 基準 | 20 | 1〜5 | C_Q / C_Lap | K=3: 19/20 vs 14/20 |
| Y 側の信号 | 信号の強さ | $w$（平均 edge 確率を揃える） | 各 20 | 1〜5 | C_Q / C_Lap | K=3: C_Lap 17/20・19/20・20/20、C_Q 1/20・14/20・19/20（weak・baseline・strong） |
| X 側の信号 | 信号の強さ | f_scale | 各 20（strong 19 完了） | 1〜5 | C_Q / C_Lap | K=3: C_Lap 20/20・20/20・18/19、C_Q 11/20・15/20・18/19（weak・base・strong） |
| K_true を変える | 真の次元 | K_true（信号を揃える） | 各 10 | 1〜5 | C_Q / C_Lap | exact: C_Lap 10/10・10/10・10/10・8/10、C_Q 10/10・9/10・7/10・1/10（K_true = 1〜4） |
| K3→K4 の分解 | 境界の記述 | なし（既存値の分解） | 10 | 3 vs 4 | C_Q / C_Lap | K4 が勝った数: C_Lap 8/10、C_Q 1/10 |

共通: n = 75、d = 12、X = Gaussian×3 / Bernoulli×6 / Poisson×3、Y = Bernoulli、L = 5、探索 / refit 8 / 8 反復、retry・差し替え・seed 救済 0。

▶ **seed 一覧と実行数を見る**
[TOGGLE START]

| 実験 | data seed | 開始点 | EM 実行 |
|---|---|---|---|
| 分布選択 | 951001–951003 | start_B / start_P | 12/12 |
| 分布 + K 同時選択 | 961001–961003 | start_B / start_P | 60/60 |
| 再現性確認 | 971001–971020 | start_B | 200/200 |
| C_Lap 試験運用 | 981001–981003 | start_B | — |
| baseline 比較 | 1001001–1001020 | start_B | 200/200 |
| Y 側の信号 | 1001001–1001020（Z・F・X は baseline と同一） | start_B | 各 200（揃えない版 400、揃えた版 400） |
| X 側の信号 | 1001001–1001020（別の生成方法。baseline とは別データ） | start_B | 593 実行 / 592 成功（600 計画） |
| K_true を変える | 1001001–1001010（K_true ごとに別データ。K_true = 3 は baseline の rep01..10） | start_B | 300/300（K_true = 3 は 0） |
| K3→K4 の分解 | なし | — | 0 |

候補の最適化: 解析勾配の BFGS（maxiter 2000、gtol 1e-10）、収束判定（勾配の最大値 ≤ 1e-8）は診断用の WARNING。
retry policy: 数値が壊れたら止めて記録（fail fast）。retry・データの差し替え・seed の救済はしない。

[TOGGLE END]

### C. Candidate B の実装検証

| 確認 | 方法 | 結果 |
|---|---|---|
| 勾配 | 解析勾配と中心差分（h = 1e-5）、n = 6・K = 2 | 最大誤差 5.47e-10（Bernoulli Y）、8.55e-10（Poisson Y） |
| 同時 Hessian | 解析 Hessian と勾配の中心差分（h = 1e-6） | 最大誤差 5.72e-10、1.03e-09。非対角ブロックも一致 |
| 厳密な場合 | Gaussian の X・w = 0 で厳密な周辺尤度と比較 | 差 1.4e-14（K = 1, 2, 3 の 3/3） |
| 尺度不変性 | 潜在変数の尺度を 0.5〜3 倍 | 値と K 間の差が不変（相対 1e-9 以内） |
| 失敗の扱い | 非停留・非正定値 | 値を返さず状態を報告（ridge・jitter なし） |
| 近似誤差の診断 | 重点サンプリング（K = 2, 3） | 相対 ESS 0.4〜12%。厳密な周辺尤度は得られていない |
| θ̂ の停留性 | 保存した θ̂ での勾配と 1 ステップ更新 | θ̂ は停留点ではない。1 ステップで最良と次点の順序は 20/20 保持。多ステップは結論が出ていない |

数値（~1e-9 など）は検証 report の測定値で、テストが固定しているのは許容値（1e-6 など）です。

### D. 用語集

| 用語 | 意味 |
|---|---|
| family | 分布族（Gaussian / Bernoulli / Poisson） |
| support gate | 観測値の形（0/1・整数・連続）から候補 family を機械的に決める規則。尤度を使わない |
| score 決定列 | gate で 1 つに決まらず、score で Bernoulli か Poisson を選んだ列 |
| margin | 選ばれた family と次点の score の差（−2 × score の単位） |
| exploration / refit | 分布を更新しながらの探索 MCEM / 分布を固定してゼロからやり直す推定 |
| start_B / start_P | 0/1 列の分布を Bernoulli / Poisson から開始 |
| C_Q | 既存の Q 型の K 選択基準（Q-based complete-data / ICL-type）。Schwarz BIC ではない |
| C_Lap（Candidate B） | Z を積分した evidence の Laplace 近似を動機とする K 選択基準。厳密な周辺尤度ではない |
| P_Z | C_Q のうち Z の事前密度に由来する項。1 次元あたり n(1 + log 2π) |
| volume（体積） | C_Lap の $\lVert\hat Z\rVert^2+\log\lvert H\rvert$。データが Z をどれだけ絞り込むかを表す |
| joint mode | $nK$ 次元の Z をまとめて最適化した最頻値 |
| exact / under / over | 選んだ K が真値と一致 / 小さい / 大きい |
| dataset と path | dataset = 生成した 1 つのデータ。path = dataset × 開始点（同じデータなので独立ではない） |
| matched design | K_true を変えても平均的な信号の強さが同じになるように条件を揃えた設計 |
| lineage E | 今回の実装系列（experimental prototype）。修論本文への採用は未決定 |

### E. source map

詳細版は `reports/presentation/phase9_progress_notion_source_map_20261001.md`。

| 内容 | report | primary artifact | code |
|---|---|---|---|
| 分布選択の設計（§3） | `reports/distribution_selection/automatic_family_selection_design_20260923.md` | — | `expfam/src/experimental/family_selection.py` |
| 分布選択の実験（§4） | `reports/distribution_selection/phase9c_c2_research_first_pilot_summary_20260925.md` | `expfam/results/family_selection/pilot_c2_20260925/` | `run_family_selection_pilot.py` |
| 分布 + K 同時選択（§5） | `reports/distribution_selection/phase9d_gate75b_joint_family_k_summary_20260927.md` | `expfam/results/joint_family_k_selection/gate75b_20260927/` | `run_joint_family_k_selection.py` |
| 再現性確認（§5.4） | `reports/distribution_selection/phase9e_k_repeatability_summary_20260927.md` | `expfam/results/k_repeatability/phase9e_20260927/` | `run_k_repeatability.py` |
| C_Q の正体（§6） | `reports/k_selection_theory/cq_decomposition_minimal_check_20260923.md`, `cq_theoretical_clarification_20260927.md` | `gate75b_20260927/cq_decomposition.csv` | `eval_utils.calc_Q_dual_strict_exp`, `eval_utils.calc_bic_exp`, `reproduction/src/model.py`（`scale_Z`） |
| C_Lap（§7） | `k_true_criterion_design_comparison_20260928.md`, `laplace_evaluator_zero_em_verification_20260928.md`, `laplace_theta_penalty_theory_20260928.md` | `laplace_pilot/phase9i_20260928/`, `laplace_is_diagnostic/phase9j_20260928/` | `laplace_k_criterion.py`, `run_laplace_pilot.py`, `test_laplace_k_criterion.py` |
| baseline 比較（§8） | `lap_vs_cq_paired_phase9k_20260928.md` | `expfam/results/lap_vs_cq_20/phase9k_20260928/` | `run_lap_vs_cq_20.py` |
| Y 側の信号（§9.1） | `relational_w_sensitivity_phase9p_20260928.md`, `density_controlled_w_sensitivity_phase9t_20260928.md` | `relational_w_sensitivity/phase9p_20260928/`, `density_controlled_w_recalibration/phase9s2_20260928/`, `density_controlled_w_sensitivity/phase9t_20260928/` | `run_w_sensitivity.py`, `density_w_recalibration.py`, `run_density_controlled_w.py` |
| X 側の信号（§9.2） | `attribute_scale_design_phase9u_20260928.md`, `attribute_scale_sensitivity_phase9v_20260928.md` | `attribute_scale_design/phase9u_20260928/`, `attribute_scale_sensitivity/phase9v_20260928/` | `paired_attribute_scale.py`, `run_attribute_scale_sensitivity.py` |
| K_true を変える（§10） | `matched_k_true_design_phase9w_20260928.md`, `matched_k_true_sensitivity_phase9x_20260928.md` | `matched_k_true_design/phase9w_20260928/`, `matched_k_true_sensitivity/phase9x_20260928/` | `matched_k_true_design.py`, `run_matched_k_true.py` |
| K3→K4 の分解（§11） | `k34_boundary_decomposition_phase9y_20260929.md` | `k34_boundary_decomposition/phase9y_20260929/` | `k34_boundary_decomposition.py` |
| 全体の主張の境界（§12〜§14） | `phase9_k_selection_synthesis_20260929.md`, `RESEARCH_MASTER.md` §19, `KNOWN_ISSUES.md` Q・R | — | — |

（report は特記なしで `reports/k_selection_theory/`、artifact は `expfam/results/`、code は `expfam/src/experimental/` 配下。）
