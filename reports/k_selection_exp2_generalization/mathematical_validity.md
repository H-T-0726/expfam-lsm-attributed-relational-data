# 生成器と真の K の数学的妥当性

- 作成日：2026-11-25。基点 `origin/main` = `1b5030b`。
- **数値は MCEM を 1 回も使わない生成器診断から得た**（`tools/research_audit/diag_k_exp2_generators.py`）。結果は `expfam/results/k_selection/k_exp2_generalization_20261125/generator_diagnostics/` にある。
  - 4 系列 × K_TRUE {1,3,5,7,9} × 各 5 回の生成。基準の計算も K の選択もしていない。
  - 表の数値は 5 回の平均。
- 系列：O = G/B、A = P/B、B = G/P、C = B/G（X/Y）。
- 生成器：
  - H = historical `generate_dual_data`（既定値）
  - PC = MATLAB 移植 `set_true_params`
  - PT = 元論文の本文どおり（Fig.3 条件の既存研究と同じ規則）
  - CM = canonical-clean-v1 の matched 規則（平均行ノルム² 0.5、w = √(3/K)、w0 = −1。clean n-sweep・d-sweep と同じ）

## 1. 生成モデルは定義した指数型分布族に従うか

| 生成器 | Z | F | X | Y | 判定 |
|---|---|---|---|---|---|
| H | 列ごとに z-score（全 draw で列分散がちょうど 1.000） | 各行を √0.9 のノルムに正規化 | Gaussian-X は生成後に z-score（全 draw で完全一致） | Poisson は η を [−20, 10] に clip | **canonical モデルからの literal な draw ではない**（KI-021）。Gaussian-X・Gaussian-Y では、標本平均・分散が固定された測度 0 の集合に乗るため、どの M_K とも互いに特異になる（未追跡だった理論監査 `fixed_generator_kgen_k0_structural_audit_20260907.md`、`docs/untracked-reports-preservation-20261124` branch。**独立検証はしていない**） |
| PC | z-score | 行正規化 | z-score | w0 ~ N(0,1)、w = 3N(0,1) | H と同じ問題。さらに w の符号と大きさがデータごとに大きく変わる（w²K = 0.14〜300） |
| PT | iid N(0, I) | 各成分 N(0, 10)（分散と解釈） | 正規化なし | w0 ~ N(0,1)、w = 3N(0,1) | **G/B では canonical モデルの literal な draw**（w0・w をデータごとの乱数として扱う点を除く） |
| CM | iid N(0, I) | QR で rank K を保証し、特異値は等しい | 正規化なし | clip なし。危険な率なら fail-fast | **literal な draw**。ただし F の分布（等特異値の直交型）は特定の選択 |

## 2. rank(F) = K_TRUE か

全生成器・全 draw で rank(F) = K_TRUE、rank(Z) = K_TRUE（d = 15 ≥ 9）。

**rank は問題にならない。**

## 3. K_TRUE と K*（最小潜在次元）が一致する条件

一般に K^rank ≤ K* ≤ K_TRUE（`RESEARCH_MASTER.md` §17.2）。したがって **K^rank = K_TRUE が示せれば K* = K_TRUE** である。以下は canonical な（literal な）生成器を前提とする。

| 系列 | 経路 | 条件 | 判定 |
|---|---|---|---|
| A（P/B） | P1：Poisson-X の周辺から FF^T を復元できる | clip なし、rank F = K（generic） | **K* = K_TRUE**（CM のとき） |
| O（G/B）・B（G/P） | P5：Σ が既知なら K^rank = rank(Cov X − Σ) | Σ は未知の対角なので因子分析の識別条件が要る。Ledermann 限界 (d−K)² ≥ d+K は d=15 で K ≤ 10 | K_TRUE ≤ 9 は限界の内側。**K* = K_TRUE は generic に成り立つと期待されるが、本研究では証明していない**。**候補 K=10 はちょうど Ledermann 限界にある** |
| C（B/G） | P2：Gaussian-Y の単一 dyad の周辺から K を決められる | w ≠ 0、Z が iid N(0, I) | **K* = K_TRUE**（CM のとき）。H は Z を z-score するので前提を満たさない |
| Bernoulli-Y・Poisson-Y 単独 | U2・U5 未解決 | — | Y からの識別は未証明 |

**historical 生成器（H・PC）では、Gaussian を含む系列（O・B・C）で M-closed の前提が崩れ、K* が定義できないおそれがある**（上記の未追跡だった監査の DERIVED。未検証）。A（P/B）でも Z の z-score は残る（UNRESOLVED）。

## 4. K_TRUE を変えたとき、Y の密度と信号の強さはどう変わるか

| 生成器・系列 | K=1 | K=3 | K=5 | K=7 | K=9 | 意味 |
|---|---|---|---|---|---|---|
| H・O/A：w²K（Y の η の分散） | 2.25 | 6.75 | 11.25 | 15.75 | 20.25 | **K とともに Y の信号が 9 倍になる** |
| H・O/A：\|η_Y\|>5 の割合（Bernoulli の飽和） | 1.7% | 7.5% | 14.4% | 20.3% | 25.6% | K が大きいほど情報を持たないペアが増える |
| H・O/A：辺密度 | 0.31 | 0.36 | 0.38 | 0.40 | 0.40 | ほぼ一定 |
| H：F の平均行ノルム² | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | X の 1 方向あたりの信号は 0.9/K に**薄まる** |
| PT・O：F の平均行ノルム² | 11 | 29 | 48 | 66 | 83 | X の信号分散 ≈ 10K に対しノイズ分散 0.1。**X はほぼ rank K のノイズなし行列で、X だけで K がほぼ決まる** |
| PT・O：w²K | 0.7〜32 | 0.5〜78 | 4〜23 | 2〜136 | 12〜185 | データごとにばらつきが非常に大きい |
| CM・O/A/C：w²K、行ノルム² | 3.0、0.5 | 同 | 同 | 同 | 同 | 全体の信号をそろえる（1 方向あたりは 1/K に薄まる） |
| CM・O/A：\|η_Y\|>5 の割合 | 1.7% | 2.1% | 1.7% | 1.6% | 1.4% | 飽和はほぼ一定 |

**K_TRUE の違いと信号の強さの違いが、どの生成器でも交絡する。** 何をそろえるか（1 方向あたりの信号か、全体の信号か）は研究問いの定義そのものであり、生成器の実装の問題ではない（`w_for_matched_y_signal` の docstring と Phase 8b の estimand A/B と同じ論点）。

## 5. Poisson の指数関数は overflow しないか

- **Series B（G/P）、H**：w = 0.5、Z は z-score。clip（η > 10）はほぼ発動しないが、裾が非常に重い。
  - Y の最大値は K=1 で 25〜64、K=9 で 704〜2671。
  - 平均は 1.1→3.3。ゼロの割合は約 0.4。
- **canonical Poisson-Y（P6）**：E[Y^r] < ∞ ⇔ |w| < 1/r（Z が iid N(0, I) のとき）。
  - 平均は |w| < 1 で有限、分散は |w| < 1/2 で有限。
  - **historical の w = 0.5 は、分散が発散する境界にちょうどある。**
- **CM（matched）は Series B を生成できない**：w = √(3/K) は K=1, 3 で平均が発散し、K=5, 7, 9 で分散が発散する（K=9 でも w = 0.577 > 0.5）。生成器は fail-fast で停止した（全 25 draw）。
- **Series B を canonical に生成するには、w を新しく決める必要がある。**
  - 全体の信号をそろえる場合：w_K = c/√K で、K=1 でも分散が有限であるには c < 0.5、すなわち w²K < 0.25（**Y の信号が非常に弱い**）。
  - そろえない場合：w を K によらず固定（例 0.45）。w²K は 0.2〜1.8、E[λ] は (1−w²)^{−K/2} e^{w0} で K とともに増える。
  - **これは研究上の比較条件の選択であり、本監査では決めない。**
- Poisson-X（Series A）：CM では行ノルム² 0.5 なので λ の上限は小さく、問題はない。

## 6. Gaussian の分散の定義は正しいか

- 推定器：Gaussian-X の `sigma` は**分散**の対角、Gaussian-Y の `self.sigma_y` は**標準偏差**（使用時に二乗する。`AGENTS.md` §1）。
- canonical-clean-v1：`sigma_x_var` は分散、`sigma_y_sd` は標準偏差で、推定器と一致（G7）。
- historical：Gaussian-X の uniq = 0.1 は分散だが、生成後に z-score するので、データ上の分散は 1 になる。Gaussian-Y の `sigma_y` = 0.1 は標準偏差。
- 元論文：Σ = diag(0.1) は分散（共分散行列）。F の「N(0, 10)」は分散か標準偏差か不明（UNKNOWN）。

## 7. Bernoulli の飽和で情報が失われないか

§4 のとおり。

- H・PC・PT では、K が大きいほど飽和したペアが増える（PT・PC の K=9 では Y のペアの約半数が \|η\| > 5）。
- CM では約 1.5% で一定。
- Bernoulli-X（Series C）は CM で η_X の SD 約 0.7、平均 0.5（P7 により一次モーメントは K の情報を持たない）。

## 8. clip や棄却が確率モデルを変えていないか

- H の Poisson clip [−20, 10] は、発動したペアの分布を変える。今回の既定値では発動率はほぼ 0 だが、**発動しうる設計は literal な draw ではない**。
- Bernoulli の clip（±500）は確率に影響しない。
- CM は clip しない。危険な率なら生成を止める（seed を変えて救済しない）。
- d-sweep では率の上限ゲートが分布を変える確率を上界 1.77×10⁻³³ で評価していた（§21）。今回の CM 条件の Poisson-X（Series A）でも同様のゲート評価が要る。

## 9. 推定側の Ledermann 限界と候補 K = 10

- Gaussian-X（O・B）では、d = 15 で K = 10 がちょうど Ledermann 限界にある。候補 K = 10 は、X 周辺の自由度の上限でもある。
- **候補から K = 10 を除くことは、依頼どおり行わない。** 結果の解釈では、境界での挙動として扱う。
- 罰則の F のパラメータ数 Kd − K(K−1)/2 は K = 10 で 105 であり、正の値である。

## 10. 結論（JUDGMENT）

- **historical 生成器は、K_TRUE を 1〜9 で変える実験の生成器として不適切である。**
  - Z・X の正規化と F の行正規化により、canonical モデルからの literal な draw ではない。
  - Y の信号が K とともに 9 倍になる。
  - Series B は分散発散の境界にある。
- **canonical-clean-v1 は literal な draw だが、K ごとのパラメータ規則を決めなければ使えない。** 特に Series B（Poisson-Y）は、既存の matched 規則では生成できない。
- **元論文の本文どおりの生成器（PT）は G/B では literal な draw である。** ただし X がほぼノイズなしで、w がデータごとに大きく変わるので、「K 選択の難しさ」を測る条件としては特殊である。

以上の選択は研究上の比較条件の定義なので、**研究者の判断が要る（BLOCKER ではなく Human Gate）**。
