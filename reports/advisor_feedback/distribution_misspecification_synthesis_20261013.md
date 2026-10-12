# 分布の誤指定で何が起きるか — 既存研究の統合と次の実験の設計（2026-10-13）

- 位置づけ: **RESEARCH SYNTHESIS / RESEARCH DESIGN**（Issue #130）。品質監査・合否判定ではない。
- 新しい EM / refit: **0**。新しい推論実験: **0**。
  使ったのは、commit 済みの一次 CSV の読み取りと、同じ dataset どうしの対比較の再計算だけである。
- 補助表: `reports/advisor_feedback/distribution_misspecification_condition_table_20261013.csv`
  （出典・系列・真の family・fit の family・誤指定の種類・Y の情報量・指標・対比較の差、63 行）
- 記号: **FACT** / **DERIVED** / **OBSERVED** / **INTERPRETATION** / **HYPOTHESIS** / **DECISION**
- **系列を混ぜない（KI-002）**: 本書の証拠は 4 つの系列にまたがる。数値は必ず出典の系列の中だけで比べ、系列をまたいで差や倍率を計算しない。

| 略称 | 系列 | 主な出典 |
|---|---|---|
| Exp4-fixed | C: fixed（Y 側の 0.5 なし）、X 全列が 1 つの family | `expfam/results/fixed_official/exp4/fixed_exp4_scen_{a,b,c}_summary.csv` |
| Exp4-old | B: old 0.5（学会予稿の系列） | `expfam/results/exp_scenario_{A,B,C}_exp4_mismatch.csv`（本書では参考のみ） |
| PC-legacy | F: per-column prototype（legacy numerics） | `per_column_family/single_vs_joint_summary.csv`、`story_diagnostics/y_sparsity_stress_20260713_trials10.csv` |
| PC-consistent | E: per-column objective-consistent | `story_diagnostics/complementary_blocks_consistent_20260821_summary.csv`（#27） |
| NB-Y | fixed（Y 側の誤指定） | `overdispersion/poisson_misspecification_summary.csv` |

---

## 1. 先生の質問

> 正しい分布で人工データを生成し、その正しい分布を仮定して精度が良いことだけでは、分布を拡張する意味の説明として弱い。
> 間違った分布を仮定したときに何が起きるのかも見たい。

本書では、これを次の問いに分けて答える。

- RQ1: 正しい family と間違った family で fit したとき、Z の回復はどう変わるか。
- RQ2: Y の予測や関係パラメータ（w0, w）の推定も変わるか。
- RQ3: その影響は、Y の情報が多いときと少ないときで違うか。
- RQ4: 「all-Gaussian が意外に良い」という既存の観察は、一次 artifact で確認できるか。
- RQ5: 既存の結果だけで説明できるか。できないなら、何を 1 つ足すと最も情報が増えるか。

---

## 2. なぜ分布の誤指定を調べるのか

本研究の拡張は「X と Y のそれぞれに適切な指数型分布族を割り当てる」ことである。
正しい family を使ったときに良い結果が出ることだけでは、**その family を選んだことが結果に効いたのか**が分からない。
同じデータを間違った family で fit した結果と比べて初めて、「正しい観測モデルを使う意味」が、どの条件で、どの指標に、どれくらいあるのかを言える。

これは、誤指定すると必ず悪くなることを示すための調査ではない。
**差がない条件があれば、それも「いつ family が重要でないか」という研究結果になる。**

---

## 3. correct / wrong family の定義

- **correct**: 生成に使った family と同じ family で fit する（oracle）。
- **wrong**: 同じ dataset を、生成とは別の family で fit する。
- **どちら側の誤指定か**を必ず区別する。
  - X 側だけ（Y は正しい）: X の分布の誤指定の効果
  - Y 側だけ（X は正しい）: Y の分布の誤指定の効果
  - 両側
- 本書の主な関心は **X 側だけの誤指定**である。先生の問いは、属性の分布を指数型分布族として扱う意味に向いているためである。
- 「列を減らす」比較（`single_*`、`y_only`）は誤指定ではなく、情報量を変える比較である。誤指定の効果として扱わない。

---

## 4. support-compatible と support-invalid の違い

誤指定はすべて同じ意味ではない。本書では次の 3 種類に分ける。

| 種類 | 例 | 意味 | 本書での扱い |
|---|---|---|---|
| **A. support-compatible** | 真の Bernoulli X を Poisson で fit | 観測値 0/1 は Poisson でもあり得る値である。同じデータを、別の「正当な」確率モデルで説明している | **主要な比較** |
| **B. representation / observation-model mismatch** | 真の Bernoulli / Poisson X を Gaussian で fit | 同じ数値を、連続値の観測モデルとして扱う。Gaussian は分散を推定する | 主要な比較（下流の性能への影響として） |
| **C. support-invalid** | 2 以上を含む Poisson の count を Bernoulli で fit、負や連続の Gaussian を Poisson / Bernoulli で fit | その family ではそもそもデータを表現できない。尤度の式は計算できても、確率モデルとしては成り立たない | 通常の「正しい vs 間違い」の比較にはしない。「その family では表現できない」という別の現象として記録する |

**C についての注意（FACT）**: PC-legacy の `all_bernoulli`（count と連続値を Bernoulli として fit）は、3 試行中 1 試行で崩壊した（RMSE_Z 1.75、test Y log-lik −106.5）。
しかもそのとき基準値（criterion）は 9 条件の中で最小だった（#28 監査 §5.4）。
support の外で計算した尤度は、family をまたいで比べられない。

---

## 5. 既存の distribution-mismatch 実験

### 5.1 一覧（どの実験が何に使えるか）

| 実験 | 系列 | データ | 操作 | 同じ dataset の対比較 | Z の指標 | Y の指標 | 本書での使い道 |
|---|---|---|---|---|---|---|---|
| **Exp4-fixed**（fixed official Exp4） | C | n=150, d=15, K=3。X は全列が 1 つの family。Scen A: X=Poisson / Y=Bernoulli、B: X=Gaussian / Y=Poisson、C: X=Bernoulli / Y=Gaussian（σ_y=0.1）。A/B は 10 試行、C は 30 試行 | X family × Y family の 3×3 を fit。K_fit = K_true = 3、family の自動選択なし | **あり**（試行ごとに seed_data が全条件で同じことを確認） | 回転だけの Procrustes RMSE_Z | なし（`rmse_Y` は fit した family の平均関数で計算しているため、Y の family をまたいで比べられない） | **X 側だけ・Y 側だけの誤指定の主要な証拠** |
| Exp4-old | B（0.5 あり） | Exp4-fixed と同じ設計 | 同上 | あり | Procrustes RMSE_Z | なし | 系列による違いの参考のみ（KI-003） |
| single_vs_joint | F（legacy） | n=80, d=9, k*=2、X = G3/B3/P3、Poisson-Y、全ペアで学習、strict held-out、**3 試行** | `per_column_all`（正）vs `all_gaussian` / `all_poisson` / `all_bernoulli` など | あり | Procrustes RMSE_Z | held-out Y log-lik | dense-Y での all-family 強制 |
| y_sparsity_stress（trials10） | F（legacy） | single_vs_joint と同じ generator。Y の学習ペアの観測率を {1.0, 0.5, 0.2, 0.1} に変更、10 試行 | `per_column_all` vs `all_gaussian` など | あり（同じ test mask） | Procrustes RMSE_Z | held-out Y log-lik、w0 / w の誤差 | **Y の情報量との関係の主要な証拠** |
| #27 complementary blocks | E-consistent | n=80, d=9, K=3、属性ブロックごとに別の潜在次元を担う F、Poisson-Y、観測率 {1.0, 0.1}、10 試行 | `per_column_all` vs `all_gaussian` など | あり（hash で確認） | Procrustes RMSE_Z | held-out Y log-lik、w0 の誤差 | 数値的に整合した系列での再確認 |
| NB-Y Poisson 誤指定 | fixed | 過分散の Y（NB、r ∈ {2, 5, 20}）を Poisson で fit、5 試行ずつ | Poisson（誤）vs NB oracle（正） | あり | Procrustes RMSE_Z | held-out Y mean log-lik | Y 側の誤指定で Z と Y の評価が分かれる例 |
| #74 / #75 family selection（Phase 9C/9D/9K…） | E | Phase 9 の mixed-X、Bernoulli-Y | family を**選ぶ** | — | — | — | **使えない**: 選択は常に真の family に到達しており、wrong family を固定した fit が存在しない |

### 5.2 Exp4-fixed の結果（X 側だけの誤指定を中心に、本書で再計算）

対比較の差 = 誤指定の RMSE_Z − oracle の RMSE_Z（同じ試行どうし）。倍率 = 平均の比。

| シナリオ | 真の X / Y | fit の X（種類） | Y は正しい | RMSE_Z 平均（oracle） | 倍率 | 差の中央値 | 悪化した試行 |
|---|---|---|---|---|---|---|---|
| C | Bernoulli / Gaussian（σ_y=0.1） | **Poisson（A: support-compatible）** | ✓ | 0.0231（0.0231） | **1.00** | −0.0000 | **15/30** |
| C | Bernoulli / Gaussian | Gaussian（B: representation） | ✓ | 0.0232（0.0231） | **1.00** | +0.0000 | 16/30 |
| A | Poisson / Bernoulli | Gaussian（B: representation） | ✓ | 0.4855（0.2320） | **2.09** | +0.2937 | **10/10** |
| A | Poisson / Bernoulli | Bernoulli（C: invalid、count ≥ 2） | ✓ | 1.0071（0.2320） | 4.34 | +0.7712 | 10/10 |
| B | Gaussian / Poisson | Bernoulli（C: invalid、実数値） | ✓ | 0.1760（0.1399） | 1.26 | +0.0396 | 10/10 |
| B | Gaussian / Poisson | Poisson（C: invalid、実数値） | ✓ | 0.1953（0.1399） | 1.40 | +0.0525 | 10/10 |

Y 側だけ・両側の誤指定（参考）:

| シナリオ | 誤指定 | 倍率 |
|---|---|---|
| C | Y だけ: Gaussian → Bernoulli / Poisson | 19.6 / 13.4 |
| C | 両側: X Poisson + Y Bernoulli | **40.4**（KI-003 の fixed の最大値） |
| B | Y だけ: Poisson → Bernoulli / Gaussian | 7.9 / 4.5 |
| A | Y だけ: Bernoulli → Gaussian / Poisson | 1.15 / 1.36 |

すべての条件で fit は成功した（success 10/10、30/30）。

**OBSERVED**

- **Exp4 の「誤指定で数倍〜40 倍悪化」という結果は、主に Y 側の誤指定から来ている。**
  X 側だけの誤指定で最も大きいのは、Scen A の 4.34×（support-invalid）と 2.09×（Poisson → Gaussian）である。
- **support-compatible な誤指定（真の Bernoulli X を Poisson で fit）は、既存の証拠では Scen C にしかなく、そこでは差がまったくなかった**（30 試行中 15 で悪化、15 で改善）。
  ただし Scen C の Y は Gaussian で σ_y = 0.1、Y の信号の分散がノイズの約 75 倍あり、ほぼノイズのない関係である（DERIVED: η^Y の SD = 0.5·√3 ≈ 0.87 に対してノイズの SD 0.1）。
  oracle の RMSE_Z が 0.023 と非常に小さいことからも、Z はほぼ Y だけで決まっていると考えられる。
- 同じ「X だけの誤指定」でも、Y が Bernoulli の Scen A では 2〜4 倍の悪化があった。

**系列による違い（参考、数値は混ぜない）**: Exp4-old でも、Scen C の X → Poisson は差なし（0.99×）、Scen A の X → Gaussian は約 2.5×、X → Bernoulli は約 3.4× で、傾向は同じだった。
一方、Scen C の X → Gaussian は Exp4-old で 3.67×、Exp4-fixed で 1.00× と食い違う。この条件の結果は実装の系列に依存しており、本書では解釈に使わない。

---

## 6. all-Gaussian の証拠

`all_gaussian` は「すべての属性列を Gaussian として fit する」条件である。
真の family が Gaussian の列（3/9）は正しく、Bernoulli と Poisson の 6 列は B 型（representation）の誤指定になる。
比較相手の `per_column_all` は、同じ 9 列すべてに正しい family を割り当てる。**違いは family の割当だけ**で、使う列は同じである。

### 6.1 一次 CSV で確認した数値（本書で再計算）

| 出典（系列） | Y の観測率 | RMSE_Z: all_gaussian / per_column_all | 差（試行ごとの平均） | all_gaussian が悪かった試行 | held-out Y log-lik の差 |
|---|---|---|---|---|---|
| single_vs_joint（F、3 試行） | 1.0 | 0.2343 / 0.2347 | **−0.0004** | 2/3 は per_column の勝ち。試行ごとに +0.0014, −0.0066, +0.0039 | −0.0012 |
| y_sparsity trials10（F） | 1.0 | 0.2328 / 0.2214 | +0.0114 | 10/10 | −0.0063（悪化 7/10） |
| y_sparsity trials10（F） | 0.5 | 0.3307 / 0.2660 | +0.0647 | 9/10 | −0.0221（10/10） |
| y_sparsity trials10（F） | 0.2 | 0.7320 / 0.3204 | **+0.4117** | 10/10 | −0.1252（10/10） |
| y_sparsity trials10（F） | 0.1 | 0.7690 / 0.3429 | **+0.4261** | 10/10 | −0.1230（10/10） |
| #27 complementary（E-consistent） | 1.0 | 0.2832 / 0.2609 | +0.0222 | 8/10 | −0.0142（8/10） |
| #27 complementary（E-consistent） | 0.1 | 0.7876 / 0.6696 | **+0.1180** | 8/10 | −0.0199（5/10） |

X の再構成（single_vs_joint、dense）: Bernoulli 列の X RMSE は 0.679 vs 0.450、Poisson 列は 2.555 vs 1.394。
all_gaussian は **X の再構成では明確に悪い**。

### 6.2 結論（RQ4）

- **確認できた（FACT / OBSERVED）**: Y の全ペアを観測している条件では、all-Gaussian は Z の回復でも Y の予測でも正しい family とほぼ同じだった（差 −0.0004〜+0.022）。
  「意外に競争力がある」という #27 / #28 の観察は、一次 CSV で再現した。
- **ただし条件つき**: Y の観測を 1 割に減らすと、all-Gaussian は明確に悪くなった（legacy で +0.43、10/10、consistent で +0.12、8/10）。
  #28 監査 §11.3 のとおり、**all-Gaussian の頑健さは dense-Y の現象であり、一般的な性質ではない。**
- 悪化の大きさは系列と generator によって違う（legacy +0.43、consistent +0.12）。この 2 つは generator（complementary F の有無、k* = 2 と K = 3）も違うので、差を系列の効果とは読まない。

---

## 7. Y の情報量との関係

### 7.1 仮説

> Y の情報が多いと、X の family を多少間違えても Z は Y 側から決まる。Y の情報が少ないと、X の観測モデルの正しさがより重要になる。

### 7.2 既存の証拠

| 証拠 | Y の情報量を変える操作 | X 側の誤指定の効果 | 仮説との関係 |
|---|---|---|---|
| y_sparsity trials10（F） | 学習に使う Y ペアの**観測率**を 1.0 → 0.1 | all_gaussian の悪化 +0.011 → +0.065 → +0.412 → +0.426（単調に増加） | **支持**（同じ dataset・同じ test mask の対比較） |
| #27（E-consistent） | 観測率 1.0 → 0.1 | +0.022 → +0.118 | **支持**（系列を変えても向きは同じ） |
| Exp4-fixed、シナリオ間 | Y の family と強さが違う（C: ほぼノイズなしの Gaussian-Y、A: Bernoulli-Y） | C: 差なし、A: 2〜4 倍 | 向きは整合するが、**真の X の family と誤指定の種類も違うので、Y の情報量の効果としては分離できない** |

**INTERPRETATION**: 仮説は「Y の観測率」という操作については、2 つの系列の同じ dataset の対比較で支持されている。
Y の family や強さ（w）の違いについては、整合的な観察はあるが分離された証拠はない。

### 7.3 混同しないこと

- **観測率（mask）を下げること**と、**edge 密度を下げること（w0 を下げる）**は別物である。
  既存の「sparse-Y」はすべて観測率の操作である。edge 密度や w を変えて X の誤指定の効果を見た実験はない。
- 観測率を下げると、X と Y の情報のバランスだけでなく、w0 / w の推定に使える標本も減る。
  all_gaussian の w0 の誤差は観測率 0.1 で 0.079（per_column は 0.030）に増えており（y_sparsity）、Z の悪化の一部はスカラーのパラメータの誤差を通っている可能性がある（#28 §8.3）。

---

## 8. Z の回復への影響（RQ1）

| 誤指定の種類 | 条件 | Z への影響 |
|---|---|---|
| A: support-compatible（Bernoulli → Poisson） | Exp4-fixed Scen C（ほぼノイズなしの Gaussian-Y） | **差なし**（1.00×、悪化 15/30） |
| A + C の混合（Gaussian → Poisson と Bernoulli → Poisson が同時） | single_vs_joint `all_poisson`（dense、3 試行） | +0.061（3/3 で悪化）。support-invalid な Gaussian 列を含むので、A だけの効果ではない |
| B: representation（→ Gaussian） | Exp4-fixed Scen C（強い Y） | 差なし（1.00×） |
| B: representation（→ Gaussian） | Exp4-fixed Scen A（Bernoulli-Y、X 全列が Poisson） | **2.09×**（+0.29、10/10） |
| B: representation（6/9 列 → Gaussian） | per-column、dense-Y | ほぼ差なし（−0.0004〜+0.022） |
| B: representation（6/9 列 → Gaussian） | per-column、Y 観測率 0.1 | **+0.12〜+0.43**（8/10〜10/10） |
| C: support-invalid | Exp4-fixed Scen A / B、PC-legacy `all_bernoulli` | 1.26〜4.34×。崩壊する試行あり |

**まとめ（OBSERVED）**: X 側の誤指定が Z に与える影響は、**ゼロから数倍まで条件によって大きく違う**。
影響が小さかったのは Y の情報が多い条件、大きかったのは Y の情報が少ない条件か、support-invalid の条件だった。

---

## 9. Y の予測への影響（RQ2）

| 証拠 | Y の指標 | 結果 |
|---|---|---|
| per-column dense（F / E） | held-out Y log-lik | all_gaussian の悪化は −0.001〜−0.014 と小さい |
| per-column 観測率 0.1（F） | held-out Y log-lik | all_gaussian は −0.123（10/10）。Z と同じ向き |
| per-column 観測率 0.1（E-consistent） | held-out Y log-lik | −0.020（5/10）。**Z（+0.118、8/10）ほどはっきりしない** |
| per-column 観測率 0.1（F） | w0 / w の誤差 | all_gaussian 0.079 / 0.055、per_column 0.030 / 0.020 |
| NB-Y を Poisson で fit（Y 側の誤指定） | held-out Y mean log-lik | Poisson（誤）は NB（正）より悪い（15/15）。**一方 RMSE_Z では Poisson の方が良い（15/15）** |
| Exp4 | — | Y の指標は Y の family をまたいで比べられない |

**OBSERVED**: X 側の誤指定では、Y の予測への影響は Z への影響と同じ向きだが、多くの場合 Z より小さかった。
Y 側の誤指定（NB-Y）では、Z の回復と Y の予測が**逆の向き**に動いた。
**「Z の回復」と「Y の予測」は別の目的であり、誤指定の影響も別々に報告する必要がある。**

---

## 10. 既存結果から言えること

1. **分布の誤指定の影響は一様ではない。** X 側だけの誤指定では、Z への影響は「差なし」から「2〜4 倍の悪化」まで条件によって違った。
2. **Y の情報が多いと、X の family の影響は隠れる。** 同じ dataset で Y の観測率を下げると、all-Gaussian と正しい family の差は単調に広がった（2 つの系列で同じ向き）。
3. **all-Gaussian の「意外な強さ」は本物だが、dense-Y の条件に限られる。** Y が少ないと明確に負ける。また X の再構成では常に悪い。
4. **Exp4 の大きな悪化倍率（数倍〜40 倍）は、主に Y 側の誤指定による。** X 側だけの誤指定の最大は 4.34×（support-invalid）である。
5. **support-invalid な誤指定は別の現象である。** 崩壊や比較不能な基準値が起き、「間違えると少し悪くなる」とは質的に違う。
6. **Z の回復と Y の予測は、誤指定に対して別々に動くことがある**（NB-Y では逆向き）。

すべて合成データ・プロトタイプ系列の条件つきの観察であり、一般的な定理ではない。

---

## 11. まだ分からないこと

1. **support-compatible な誤指定（Bernoulli → Poisson）の効果は、Y がほぼノイズなしの条件（Exp4 Scen C）でしか測っていない。**
   そこで差がなかったのは Y が強すぎたためである可能性が高く、現在の研究の中心である Bernoulli-Y・mixed-X の条件（Phase 9 の設定）では一度も測っていない。
   **先生の問いに最も直接答えるはずの比較が、Y が強すぎて結果が出ない条件でしか存在しない。**
2. Y の情報量との関係は**観測率の操作**でしか確認していない。edge 密度や w を変えた場合は未検討。
3. Exp4 のシナリオ間の違いは、Y の family・真の X の family・誤指定の種類が同時に違うので、分離できない。
4. 機構（推定した分散による自動的な重みの低下、曲率 A''/φ の違いなど）は #28 §11.3 の HYPOTHESIS のままで、測定していない。
5. per-column の証拠の試行数は少ない（dense は 3 試行、ほかは 10 試行）。n は 80〜150 だけ。
6. 実データでの誤指定の影響（例: MovieLens の count 列）は合成データと別の問題（切片・スケール・X 側の過分散）と交絡している（#28 §9）。本書では扱わない。

---

## 12. 新しい実験が必要か

- 先生に「間違った分布を当てるとどうなるか」を説明する材料は、**かなりある**（§10）。
  特に「Y が少ないと family の正しさが効く」「all-Gaussian は Y が多いときだけ強い」は、修論のストーリーとして使える。
- しかし、**最も素直な比較（同じ Bernoulli の X を Bernoulli と Poisson で fit する）**の既存の証拠は、Y が強すぎて差が出ない条件だけである。
  この比較は、先生の質問にそのまま対応し、#131 で先生に説明した人工データ（Phase 9 の条件）と同じデータで実行できる。
- どちらの結果でも研究判断が変わる。
  - 差があれば: 「support-compatible な誤指定でも Z の回復は悪くなる」と言える。
  - 差がなければ: 「0/1 の属性では、Bernoulli と Poisson の区別は Z にほとんど効かず、family の重要さは representation（Gaussian 化）や Y の情報量の側にある」と言える。

したがって、**1 つだけ小さな実験を追加する価値がある**。

---

## 13. 最小の実験 1 つ（凍結する設計、実行しない）

### M1 — Bernoulli truth: correct Bernoulli vs wrong Poisson（Phase 9 の条件、dense / sparse の Y 観測率）

| 項目 | 凍結する値 |
|---|---|
| 研究上の問い | 同じ dataset で、真の Bernoulli の 6 列を Bernoulli（正）または Poisson（誤、support-compatible）として fit したとき、Z の回復はどう変わるか。その差は Y の観測率で変わるか |
| 種類 | CHARACTERIZATION（効果の向きと大きさの記述。優劣の検定ではない） |
| データ | Phase 9K と同じ generator と条件: `data_generator_canonical_mixed.py`、n = 75、d = 12、K_true = 3、X = G3 / B6 / P3、Y = Bernoulli、w0 = −1、w = 1、f_scale = √2、sigma_x_var = 1 |
| replicate | 20（data_seed 1001001..1001020、Phase 9K と同じ。Z / F / X / Y はビット単位で 9K と同じ） |
| arm | **correct**: family_x_list = G3 / **B6** / P3。**wrong**: G3 / **P6** / P3（Bernoulli の 6 列だけを Poisson にする。Gaussian と Poisson の列は両 arm で同じ） |
| Y の観測 regime | replicate ごとに固定した test mask（全ペアの 10%、両 arm・両 regime で同じ）。学習に使う残りのペアの観測率 ∈ {**1.0**（dense）, **0.1**（sparse）}。sparse の学習 mask は dense の学習ペアの部分集合とし、両 arm で同じにする |
| K | **K_fit = K_true = 3** に固定。K の選択なし |
| family | **自動選択なし**。arm の family_x_list を固定する |
| 推定 | 系列 E-consistent の per-column モデル（`numerics_mode = "consistent"`）、L = 5、反復 8、初期化の方針と seed は両 arm で同じ |
| 主要な指標 | **Procrustes RMSE_Z**（回転だけ、`procrustes_rmse_z` と同じ定義）。対比較の差 `ΔZ = RMSE_wrong − RMSE_correct` を regime ごとに、中央値・平均・符号の数（wrong が悪い試行数 / 20）で報告する |
| 副次的な指標 | held-out Y の Bernoulli log score（test mask 上）、w0 / w の誤差、Bernoulli 列の X 再構成の誤差、収束・warning・runtime |
| 規模 | 20 replicate × 2 arm × 2 regime = **80 EM**。retry・置き換え・seed の救済なし |
| 結果の採用 | correct が良い・差なし・wrong が一部良い、のどれも研究結果として採用する。結果を見た後に seed・条件・指標を変えない |
| 解釈の制約 | 「Bernoulli → Poisson の support-compatible な誤指定の、この条件での効果」と呼ぶ。Poisson では各列の平均・曲率の形が変わるので、「分布の正しさだけの効果」とは呼ばない。観測率は edge 密度ではない |
| 実装の制約 | 既存の consistent per-column モデルと #27 の strict held-out の流れを再利用し、固定 family で fit するための最小の wrapper と記録だけを追加する。新しい gate や auditor は追加しない（`CLAUDE.md` §6） |
| 扱わないこと | M2（Bernoulli → Gaussian）、M3（Poisson → Gaussian）、Poisson → Bernoulli（support-invalid）、family の選択、K sweep、実データ |

**この設計を選んだ理由**

- 先生の問いを最も素直な形で検証する（同じデータ・同じ K・family だけが違う）。
- support-compatible なので、support-invalid による崩壊と混ざらない。
- #131 で先生に説明した人工データと同じ条件なので、説明の流れがつながる。
- dense / sparse の 2 regime で、§7 の仮説を support-compatible な誤指定についても確認できる。

M2 / M3 は、M1 の結果を見た後に必要なら別に判断する（representation 型の証拠は §6 にすでにある）。

---

## 14. 修論ストーリーへの意味

- 「正しい分布を使えば常に良くなる」とは書けない。既存の証拠は、**dense-Y では all-Gaussian でもほぼ同じ**ことを示している。
- 代わりに、より正確で研究として価値のある主張ができる:
  > **分布の正しさの重要性は、Y の情報量に依存する。** 関係データが十分にあるときは、属性の分布の仮定は Z の推定にほとんど効かない。
  > 関係データが少ないとき、属性を適切な観測モデルで扱うことが Z の推定を大きく改善する（この条件で RMSE_Z が 0.12〜0.43 改善）。
- これは本研究の価値を下げるものではない。実データの多くは関係が少ないか欠測が多いので、「**関係が少ないときこそ属性の分布を正しく扱う意味がある**」という位置づけになる。
- 同時に、support-invalid な扱い（count を 2 値として扱うなど）は崩壊や比較不能な基準値を起こすので、family を考える最低限の理由になる。
- 限界として、Z の回復・Y の予測・X の再構成は誤指定に対して別々に動くことと、証拠がすべて合成データ・プロトタイプ系列であることを明記する。

---

## 15. 先生に 1〜2 分で説明する場合の要点

1. **やったこと**: 同じ人工データを、正しい分布と間違った分布で推定し、潜在構造 Z がどれだけ正しく戻るかを比べた実験が、すでにいくつかあります。
2. **結果 1 — 関係データが多いとき**: 属性をすべて正規分布として扱っても（6/9 列は間違い）、Z の推定はほとんど変わりませんでした（差 0.00〜0.02）。関係 Y がほぼノイズなしの条件では、Bernoulli の属性を Poisson として扱っても差はゼロでした。
3. **結果 2 — 関係データが少ないとき**: 関係の観測を 1 割に減らすと、属性を正規分布として扱った場合は明確に悪くなりました（RMSE_Z で +0.12〜+0.43、ほぼすべての試行）。
4. **つまり**: 分布を正しく選ぶ意味は、関係データが少ないときに大きく、多いときには隠れます。「正しい分布なら常に良い」ではなく、「いつ効くか」が分かってきました。
   なお、2 値の属性を 2 値として扱えないような明らかな誤り（count を 0/1 として扱うなど）は、推定が壊れることがあります。
5. **まだ足りないこと**: 一番素直な比較（0/1 の属性を Bernoulli と Poisson で比べる）は、関係が強すぎて差が出ない条件でしか測っていません。
   いまの研究の人工データで、関係が多い場合と少ない場合の 2 通りで、この比較を 1 回だけ追加したいと考えています（80 回の推定）。

---

## DECISION

## **B: RUN_ONE_BOUNDED_MISSPECIFICATION_EXPERIMENT_NEXT**

既存の証拠で「誤指定の影響は Y の情報量に依存する」という説明はかなりできる。
しかし先生の問いに最も直接対応する support-compatible な比較（Bernoulli → Poisson）は、Y がほぼノイズなしで差が出ようのない条件にしかない。
§13 の M1 だけを凍結した。実行は別の experiment Issue で、Human の承認を得てから行う。

---

## 付録 A. 再計算の方法（推論なし）

- 実行環境: `.venv`（Python 3.13）。worktree の HEAD は `origin/main` = `ee4be23`。
- 各一次 CSV を pandas で読み、同じ試行（`trial`）どうしで `arm − reference` の差を計算した。
  - Exp4-fixed は `fix_w = fix_x = False` の行だけを使い、試行ごとの `seed_data` が全条件で同じことを確認した（paired）。
  - y_sparsity / #27 / single_vs_joint は `per_column_all` を reference にした。
  - NB-Y は `nb_oracle` を reference にした。
- 補助 CSV の `arm_worse_count` は、RMSE や誤差では `arm > ref` の試行数、log-lik では `arm < ref` の試行数である。
- Exp4 の RMSE_Z は、各 runner の中で回転だけの Procrustes を使って計算したもの（`utils_expfam.procrustes_rotation` → `calc_rmse`）。per-column 系の `rmse_Z` も回転だけの Procrustes である（`tools/research_audit/run_y_sparsity_stress.py`、`run_complementary_blocks_consistent.py`、`run_per_column_single_vs_joint.py` の中で `procrustes_rotation` → `calc_rmse` を使っていることを確認した）。
- M1 で再利用する `DualExpFamLSMPerColumnConsistent` が、Bernoulli-Y と Y の mask の両方に対応していることをコードで確認した（`_ObjectiveConsistentYMixin.calc_log_likelihood_Y`）。
- 一時スクリプトは repository に追加していない（本 Issue の成果物は report と table だけ）。上の手順で再現できる。

## 付録 B. 本書で使った主な一次資料

| 対象 | パス |
|---|---|
| Exp4-fixed | `expfam/results/fixed_official/exp4/fixed_exp4_scen_{a,b,c}_summary.csv`、`expfam/src/run_fixed_official_exp4_scen_ab.py`、`..._scen_c.py` |
| Exp4-old | `expfam/results/exp_scenario_{A,B,C}_exp4_mismatch.csv`、`reports/mismatch_audit/mismatch_audit_report_20260708.md` |
| Exp4 の generator | `expfam/src/data_generator_expfam.py::generate_dual_data`（デフォルト: Bernoulli-Y w0 = −1, w = 1.5、Poisson-Y 0, 0.5、Gaussian-Y 0, 0.5, σ_y = 0.1） |
| per-column | `expfam/results/per_column_family/single_vs_joint_summary.csv`、`expfam/results/story_diagnostics/y_sparsity_stress_20260713_trials10.csv` |
| #27 | `expfam/results/story_diagnostics/complementary_blocks_consistent_20260821_summary.csv`、`reports/story_diagnostics/complementary_blocks_consistent_report_20260821.md` |
| #28 | `reports/model_refinement/evidence_driven_model_refinement_audit_20260821.md`（§5, §8, §11） |
| NB-Y | `expfam/results/overdispersion/poisson_misspecification_summary.csv` |
| #74 | `expfam/results/family_selection/pilot_c2_20260925/fit_results.csv`（選択は常に真の family、固定の wrong family の fit なし） |
