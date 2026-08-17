# 指数型分布族拡張から per-column family へ：混在属性Xを扱うための追加検証

作成日: 2026-07-11
対象: `research/per-column-validation` ブランチで実施した per-column family（`DualExpFamLSMPerColumn`、**prototype**）の検証
参照元: `reports/per_column_family/` 配下4レポート、`expfam/results/per_column_family/` 配下CSV、`tools/research_audit/` 配下実装
位置づけ: 本ページは Notion 転記用のサマリであり、詳細は上記レポート・CSVを一次資料とする。

---

## 1. 一言でいうと

- 前回の発表では、属性データX・関係データYの分布を指数型分布族（exponential family）へ一般化し、GaussianやBernoulli、Poissonなどを同じ枠組み（η = 線形結合、A'(η) = 予測平均、A''(η) = 曲率）で扱えるようにした。
- しかし、その時点でもX側は「属性X全体で1つのfamilyを選ぶ」形のままだった。
- 実データの属性行列Xには、二値・連続値・カウントが同時に混在することが多く、X全体を1つのfamilyで扱うのは不自然な場合がある。
- 今回は、**属性列ごとにfamily g_l を指定できるper-column family（列ごとの分布族）のprototype**を実装し、数式監査と複数の実験で検証した。

---

## 2. 前回までの3段階

| 段階 | X側の扱い | Y側の扱い | できること | まだ足りないこと |
|---|---|---|---|---|
| 元論文・先行研究 | Gaussian固定 | Bernoulli固定 | 属性Xと関係Yを共通の潜在変数（latent variable）Zで説明 | 分布が固定で、他のデータ型に対応できない |
| dual-ExpFam LSM スカラーfamily版（前回発表） | X全体で1つのfamily（例: 全列Gaussian／全列Bernoulli／全列Poisson） | Y全体で1つのfamily | X・Yの分布を指数型分布族の中で切り替え可能 | X内部で列ごとに型が違う（混在属性）場合に対応できない |
| 今回のper-column family版 | X列ごとにfamily g_l を指定 | Yは前回同様の指定方式（変更なし） | 混在属性Xを1つの共通Zに統合 | prototypeであり、切片・スケーリング・ブロック重みなど未解決の課題が残る |

---

## 3. 前回の拡張でまだ足りなかったこと

前回発表までのdual-ExpFam LSM（スカラーfamily版）は、属性データX全体に1つのfamily `g` を仮定していた。

```
x_il | z_i ~ ExpFam_g( η_il ),   η_il = f_l^T z_i
log p(x_i | z_i) = Σ_l log p_g(x_il | η_il)
```

ここで `g` は全列共通のfamily（例: 全列Gaussian、全列Bernoulli、全列Poisson）である。実データで属性の型が混在する場合（0/1のジャンル、連続値の評価、カウントの評価件数など）、いずれか1つのfamilyを無理に全列へ適用する必要があった。

---

## 4. 今回の目的

複数の型が混在する属性（二値・連続値・カウント）を、別々のZではなく**1つの共通Zに統合**して扱えるかを検証すること。あわせて、「属性ごとに別々のモデルを回すだけで十分ではないか」という問いに、数式・実験の両面で答えること。

---

## 5. 数式上、何が変わるか

### 5.1 スカラーfamily版とper-column family版

**前回（スカラーfamily版、X全体で1つのfamily `g`）:**

```
x_il | z_i ~ ExpFam_g( η_il ),   η_il = f_l^T z_i
log p(x_i | z_i) = Σ_l log p_g(x_il | η_il)
```

**今回（per-column family版、列 `l` ごとにfamily `g_l`）:**

```
x_il | z_i ~ ExpFam_{g_l}( η_il ),   η_il = f_l^T z_i　（切片 μ_l なし）
log p(x_i | z_i) = Σ_l log p_{g_l}(x_il | η_il)
```

違いは `p_g` から `p_{g_l}` へ、**分布族が列ごとに変わりうる**ようになった点である。

MovieLens pilot（実データ検証）で実際に使用した属性は以下の通り（`movielens_mixed_x_runinfo.csv` で確認済み）:

| 属性 | 型 | 使用family | 実験での使用状況 |
|---|---|---|---|
| ジャンル有無（19列） | 0/1 | Bernoulli | 実際に使用（genre_only, mixed_percolumn 等） |
| 平均評価 | 連続値 | Gaussian（z-score） | 実際に使用（rating_stats_only, mixed_percolumn） |
| 評価件数 | カウント | Poisson（生値） | 実際に使用（同上）。Yと同じ評価ログ由来のためリーク懸念あり |
| 公開年 | 連続値 | Gaussian（z-score） | 実際に使用（rating_stats_only, mixed_percolumn の transform_note に明記） |

※ 公開年は当初の設計メモでは「Gaussian／Categorical候補」と書かれていたが、今回の実装・実験では実際にGaussian z-scoreとして使用した（Categorical対応は未実装であり、今回の実験対象にも含まれていない）。

### 5.2 Zに対する勾配（gradient）

概念的には、列ごとの対数尤度の勾配を足し合わせるだけである。

```
∇_{z_i} log p(x_i | z_i) = Σ_l ∇_{z_i} log p_{g_l}(x_il | η_il)
```

実装に合う形で書くと、

```
∇_{z_i} log p(x_i | z_i) = Σ_l w_l { x_il − A'_{g_l}(η_il) } f_l
```

- `w_l = 1 / σ_l²`（Gaussian列）
- `w_l = 1`（Bernoulli／Poisson列）

### 5.3 precision contribution（曲率・precisionへの寄与）

Laplace近似（Laplace approximation）で使うprecision（精度行列）への各列の寄与は、次のように列ごとの和として書ける。

```
Λ_X(z_i) = Σ_{l∈G} (1/σ_l²) f_l f_l^T + Σ_{l∈B∪P} A''_{g_l}(η_il) f_l f_l^T
```

（G = Gaussian列の集合、B∪P = Bernoulli／Poisson列の集合）

これは、X側の各属性列がLaplace近似のprecisionへどれくらい寄与するかを表す概念式である。

### 5.4 familyごとの A'(η), A''(η)

| family | 予測平均 A'(η) | 曲率・precisionへの寄与 A''(η) | 解釈 |
|---|---|---|---|
| Bernoulli | σ(η) | σ(η)(1−σ(η)) | 最大1/4なので、1列あたりの曲率は比較的小さい |
| Poisson | exp(η) | exp(η) | 期待カウントが大きいほど曲率も大きくなる |
| Gaussian | η | 1/σ_l² | mean linkであり、canonical formのA''ではなく実装上はσ_l²の逆数が重みとして効く |

**注意（Gaussianの扱い）:** 実装上のGaussian列はmean link（η = 平均そのもの）であり、Bernoulli／Poissonのcanonical formとは前提が異なる。上記表の「Gaussian: A'(η)=η」はmean linkに基づく整理であり、正規分布のcanonical formと単純に同一視しないよう注意する。

**重要なポイント:** familyを列ごとに変えることは、単に分布名を変えるだけではなく、Z推定に対する各属性の重み・曲率も変える。この曲率の違いが、後述のMovieLens pilotでの支配現象（§8-5）の一因になっている。

これらの式・列和構造・全列同一family時の既存モデルとの一致は、数式監査で**31/31 PASS**（`per_column_math_audit_summary.csv`、確認内容は§8-1）として数値的に確認済みである。ただし、この監査はE-stepの勾配・precision・尤度についての検証であり、M-step（Adam）の収束性やEM全体の統計的性質を保証するものではない。

<details>
<summary>導出1：X側対数尤度は何が変わったか</summary>

- スカラーfamily版では、全列が同じfamily `g` に従うため `log p(x_i|z_i) = Σ_l log p_g(x_il|η_il)`。
- per-column family版では、列ごとに `g_l` が異なりうるため `log p(x_i|z_i) = Σ_l log p_{g_l}(x_il|η_il)`。
- 指数型分布族の基本形として、直感的には次のように書ける（Bernoulli／Poissonの直感的説明として。Gaussianはmean linkのため、この形をそのまま当てはめない）。

```
log p_{g_l}(x_il | η_il) = x_il η_il − A_{g_l}(η_il) + c_{g_l}(x_il)
```

- familyが変わると、平均 A'(η) と曲率 A''(η) も列ごとに変わる。
- その結果、Z推定（E-step）への影響も列ごとに変わる。

</details>

<details>
<summary>導出2：Zに対する勾配</summary>

- `η_il = f_l^T z_i` なので `∂η_il / ∂z_i = f_l`。
- 指数型分布族では、概念的に `∂/∂η log p(x|η) = x − A'(η)`。
- したがって、列 `l` 単独の勾配は

```
∇_{z_i} log p_{g_l}(x_il | η_il) = { x_il − A'_{g_l}(η_il) } f_l
```

- 列ごとに足し合わせると

```
∇_{z_i} log p(x_i | z_i) = Σ_l { x_il − A'_{g_l}(η_il) } f_l
```

- Gaussian列では、これに `1/σ_l²` の重みが追加で入る（実装 `model_dual_expfam_percolumn.py` の `_x_weight_vector`）。

</details>

<details>
<summary>導出3：Laplace近似で使うprecision contribution</summary>

- 勾配をさらに `z_i` で微分すると、2階微分に A''(η) が現れる。
- Laplace近似では、負のHessianがprecision（精度行列）に入る。
- したがって、列 `l` の寄与は概念的に `A''_{g_l}(η_il) f_l f_l^T`。
- Gaussian列では `(1/σ_l²) f_l f_l^T`。
- これらが列ごとに足し合わされ、`Λ_X(z_i) = Σ_l w_l A''_{g_l}(η_il) f_l f_l^T` という1本の式でも書ける（w_lは勾配と共通の列重み。Gaussianは mean link のため A''=1 となり、結局 `w_l = 1/σ_l²` がそのまま効く）。本文では実装の分岐（Gaussian／Bernoulli・Poisson）に対応させて分離した式で示した。

</details>

<details>
<summary>導出4：MovieLensでカウント属性が強く効いた理由</summary>

- Poissonでは `A''(η) = exp(η)`。評価件数の期待値が大きいと、曲率も大きくなる。
- Bernoulliでは `A''(η) = σ(η)(1−σ(η)) ≤ 1/4`。
- そのため、曲率ベースではPoisson列がgenre列より強く効きうる。
- ただし、実際のprecision contributionは `A''(η) f_l f_l^T` に依存する（f_lの大きさにも依存する）。
- したがって、これは厳密な証明ではなく、MovieLens pilotで見られた支配現象の診断・説明である（詳細は§8-5）。

</details>

---

## 6. なぜこの拡張が必要か

理由は3つある。

1. **実データの属性は同じ型ではない**（0/1のジャンル、連続値の評価、カウントの評価件数など）。
2. **全列同じfamilyにすると、データの意味が崩れる場合がある**（例: 評価件数にBernoulliを当てはめると確率・カウントの定義域が守られない）。
3. **別々にモデルを回すだけでは、1つの共通Zに統合したことにならない。**

特に3つ目が本質的な論点である。別々に回す方法とper-columnで同時に回す方法の違いを整理する。

| 方法 | 何が分かるか | 限界 |
|---|---|---|
| ジャンルだけで回す | ジャンル単独の効果 | 他の属性と統合されない |
| 平均評価だけで回す | 評価水準単独の効果 | 条件ごとに**別のZ**になる |
| 評価件数だけで回す | 人気度単独の効果 | 条件ごとに**別のZ**になる |
| per-columnで同時に回す | 複数属性を**1つの共通Z**に統合 | family指定・スケール設計が必要（§10参照） |

別々に回す実験は「各属性の単独効果を見るablation」としては有用だが、条件ごとに異なるZが得られるため、「1つの共通潜在空間で属性と関係データを同時に説明する」というLSMの目的自体を果たせない。

---

## 7. 実験内容

| 実験 | 目的 | 確認したこと | fit数 |
|---|---|---|---|
| 数式監査 | 実装が数式通りか確認 | 勾配・precision・尤度・列和構造・既存モデルとの一致 | （31項目PASS、fit数に含めない） |
| single-vs-joint | 別々に回すだけで十分か確認 | 単独属性 vs 同時統合、全列共通family強制との比較 | 27 |
| 属性追加ablation | 属性を足す効果を見る | Y-onlyから属性を1ブロックずつ追加 | 15 |
| ノイズ属性チェック | 属性を増やせば良いかを確認 | 無関係な（family正指定の）属性を追加 | 18 |
| MovieLens pilot | 実データでの挙動を見る | genre／rating stats／mixed-X の比較（strict held-out） | 24 |

**実験fit数は全84 fits**（27+15+18+24）。数式監査31項目はfit数に含まない別枠。全fitでNaN・発散なし。

---

## 8. 実験結果

### 8-1. 数式監査

- **31/31 PASS**（`per_column_math_audit_summary.csv`）。
- 独立実装の対数事後の中心差分と `_calc_gradient` が一致（誤差 最大 ~2.8e-9）。
- `_calc_precision_matrix` が勾配の数値ヤコビアンと一致（誤差 最大 ~4.2e-10）。
- 尤度・勾配・precision contributionの**列和構造**（`gradient_X = Σ_l gradient_l`、`precision_X = Σ_l precision_l`）を確認（誤差 ~1e-16オーダー）。
- **全列同一family時、既存スカラーmodelと数値的に完全一致（差0.0）**。
- ただし、**M-step（Adam）の収束性やEM全体の統計的性質の理論保証ではない**（監査範囲外）。

### 8-2. 人工mixed-X（single-vs-joint、3 trials平均、RMSE_Z / test Y ll[/pair]）

| 条件 | RMSE_Z | test Y ll |
|---|---|---|
| per_column_all（本命） | **0.235 ± 0.016** | **−2.047 ± 0.012** |
| all_gaussian（誤指定比較用） | 0.234 ± 0.018 | −2.048 ± 0.014 |
| single_gaussian（単独属性） | 0.243 ± 0.017 | −2.050 ± 0.017 |
| single_poisson（単独属性） | 0.294 ± 0.013 | −2.067 ± 0.005 |
| single_bernoulli（単独属性） | 0.321 ± 0.018 | −2.078 ± 0.017 |
| y_only（ベースライン） | 0.328 ± 0.022 | −2.079 ± 0.020 |
| all_bernoulli（誤指定比較用、崩壊例あり） | 0.797 ± 0.822 | −36.9 ± 60.3 |

※ `single_gaussian`（0.243）と`single_poisson`（0.294）は別条件であり、値の取り違えに注意（`single_vs_joint_summary.csv`・`agg.csv`双方で再確認済み）。

- **per_column_all は全単独属性条件・all_gaussianと同等以上**（差はごく小さい）。
- **all_gaussianとの差は小さい**（RMSE_Z 0.235 vs 0.234、test ll −2.047 vs −2.048、ほぼ誤差範囲内）。ただし all_gaussian は非Gaussianブロックの X 再構成が1.5〜1.8倍悪化する（本文では割愛、詳細は`single_vs_joint_per_column_report_20260711.md`参照）。
- **all_bernoulli（生値のまま強制）は3trial中1trialで崩壊**（RMSE_Z 1.75、test ll −106）。誤ったfamily強制は破綻しうる。

### 8-3. 属性追加ablation（Y-onlyから1ブロックずつ追加、RMSE_Z）

```
Y-only 0.295 → +Bernoulli 0.296（改善なし）→ +Gaussian 0.231（−22%改善）→ +Poisson 0.229 → +ノイズ3列 0.231（改善なし）
```

- 改善するかどうかは属性の情報量次第で、**単調改善ではない**。
- 意味のあるGaussianブロックの追加が改善のほぼすべてで、情報の薄いBernoulliブロックの追加は効果なし。

### 8-4. ノイズ属性チェック（family正指定でも無関係な属性を追加、RMSE_Z平均）

```
no_noise 0.223 〜 gauss_noise3 0.233 〜 gauss_noise12 0.235
```

- **ノイズ属性は改善せず、平均では横ばい〜微悪化**。
- trial単位ではGaussianノイズ追加で+13%程度の悪化が見られたケースもある（seed依存、詳細はablationレポート参照）。
- **属性を増やせば必ず良いわけではない**ことがここでも確認された。

### 8-5. MovieLens pilot（strict held-out、test Y ll[/pair]）

| 条件 | test Y ll |
|---|---|
| **genre_only** | **−3.423**（最良） |
| y_only | −3.454 |
| mixed_all_gaussian（誤指定比較用） | −3.455 |
| mixed_percolumn | −3.815 |
| rating_stats_only | −3.816 |
| mixed_all_bernoulli（誤指定比較用） | −4.169（最悪） |

- **genre_only が最良、mixed_percolumn は悪化**。
- **mixed_percolumn ≈ rating_stats_only** となっており、評価統計ブロック（平均評価・公開年・評価件数）がgenre 19列の情報を実質的に飲み込んでしまったことを示す。
- **よってMovieLensでper-columnが有効とは言えない。**
- 機構として、X側モデルに切片がないため、平均154の評価件数（Poisson）はη≈5をFとZだけで作る必要があり、そのA''=exp(η)≈150という曲率がgenre列（A''≤0.25）19列合計を上回ったと考えられる（導出4参照）。この支配の説明はA''（曲率）ベースの診断であり、実際のprecisionへの寄与はA''·f_l·f_l^T（f_lの大きさにも依存）である点に注意（A''の大小だけでは厳密なprecision比較にはならない）。
- なお評価件数（ratings_count）はY（共評価カウント）と同じ評価ログ由来であり、情報リークの懸念があるが、それでも悪化した（リークで有利になる以前の問題）。

---

## 9. やったおかげで何が変わるか

- 混在属性を**列ごとのfamily**で扱えるようになった。
- 複数属性を別々のZではなく、**1つの共通Zに統合**できるようになった。
- familyの違いがZ推定に与える影響（曲率の違い）を診断できるようになった。
- 実データでは、family指定だけでなく**切片・スケーリング・ブロック重み**が重要だと分かった（MovieLensの支配現象で実証）。

---

## 10. 限界・注意点

| 限界 | 内容 |
|---|---|
| prototype | 完成手法ではない。正式な手法としての完成度は保証されていない |
| 切片なし | η_il = f_l^T z_i のみで切片μ_lがないため、非中心な属性でZが平均まで説明しようとする可能性がある |
| スケール問題 | カウント属性（例: 評価件数）が強く効きすぎる可能性がある（MovieLensで実証） |
| ブロック重み | 列数や曲率（A''）が大きい属性ブロックがZ推定を支配する可能性がある。block weightingは未実装（今後の課題） |
| family選択 | 現状は列ごとのfamilyを手動指定。データから自動選択する手続きはない |
| M-step未監査 | Adam更新・EM全体の収束性・統計的性質は数式監査の範囲外 |
| MovieLensでの悪化 | pilotでは悪化したため、実データでの有効性は未実証 |

---

## 11. 今後の課題（優先度順）

1. **X側切片の導入**: `η_il = μ_l + f_l^T z_i`。非中心・大スケール属性を扱うための前提条件（MovieLensで必要性が実証された）。
2. カウント属性の前処理比較（log変換 + Gaussian／offset付きPoisson）。
3. ブロック重み・スケーリングの設計（診断のみ実施済み、weighting実装は理論的正当化が先）。
4. seed数を増やした確認（現状3 seedsのpilot規模）。
5. family選択手続きの検討（列ごとのfamilyをデータから選ぶ仕組み。組合せ爆発が課題）。
6. 将来的に複数データセットや複数観測源を扱う場合にも、属性型の混在は生じるため、per-column familyはその基礎部品になり得る。

---

## 12. ゼミで言ってよい主張 / 言いすぎな主張

| 言ってよい主張 | 理由 |
|---|---|
| 前回の指数型分布族拡張ではX全体で1つのfamilyしか選べなかったが、今回は列ごとにfamilyを指定できるprototypeを検証した | §1・§3の整理どおり |
| 人工データでは、単独属性より同時統合が有利な傾向があった | single_vs_jointで全単独条件と同等以上（§8-2） |
| 属性を増やせば必ず良いわけではない | ablation・ノイズチェック双方で非改善例を確認（§8-3, 8-4） |
| MovieLensでは悪化したため、実データ適用には切片・スケーリング等が必要 | mixed_percolumn がgenre_onlyより悪化（§8-5） |
| 数式監査（31/31 PASS）は勾配・precision・尤度の実装正しさを確認したものである | §8-1のとおり |

| 言いすぎな主張 | なぜ危ないか |
|---|---|
| per-columnは完成した正式手法 | prototype。切片なし・ブロック支配・family選択手続きが未解決 |
| MovieLensで有効性を示した | 実際はmixed_percolumnが悪化。genre_onlyが最良だった |
| all-Gaussianより明確に優れている | 人工データでの差は誤差範囲内（RMSE_Z 0.235 vs 0.234） |
| 属性を増やせば性能が上がる | ablation・ノイズチェックで非改善・悪化例が確認されている |
| 数式監査で理論的に完全に正しいと証明した | 監査はE-step（勾配・precision・尤度）の数値一致のみ。M-step・EM全体の統計的性質は監査対象外 |

---

## 13. 最後のまとめ

今回のper-column familyは、マルチドメイン化そのものではなく、単一ドメイン内の混在属性に対応するための拡張である。人工データでは、単独属性を別々に使うより、複数属性を同時に1つのZへ統合する意義が確認できた。一方で、MovieLens pilotでは悪化が見られ、familyを列ごとに指定するだけでは不十分であり、切片・スケーリング・ブロック重みの設計が重要であることが分かった。
