# Phase 9U — Y を固定した属性 loading scale の対応設計（2026-09-28）

- 位置づけ: **THEORY / DESIGN / ZERO-EM PAIRED-GENERATOR PREFLIGHT ONLY**（Issue #116）。EM・refit・family 選択・K 選択・Candidate B の評価なし。
- 結果: `expfam/results/attribute_scale_design/phase9u_20260928/`
  （`design.json`, `stream_provenance.csv/json`, `pairing_checks.csv/json`, `attribute_signal_context.csv`, `historical_base_comparison.csv/json`, `future_protocol.json`）
- 実行コード: `5bcdf30`（`expfam/src/experimental/paired_attribute_scale.py`、新しい forward-only の module）。1 回だけ実行。
  historical な canonical generator（`data_generator_canonical.py`, `data_generator_canonical_mixed.py`）と Phase 9K/9T の artifact は変更していない。
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 問い

Y 側の w の感度は Phase 9T までで特徴づけた。次は X 側として、共通の属性 loading scale `f_scale` を変える将来の実験（Phase 9V）のために、
Z と実現した Y を条件間で共有し、X だけが意図した `f_scale` の変化と管理された観測ノイズの結合で変わる対応設計を作れるか。

## 2. f_scale と loading energy（DERIVED、実装で確認）

canonical な構成は `F = Q · f_scale`（`build_full_rank_loadings`: Gaussian 行列の reduced QR、対角の符号を固定、すべての特異値を f_scale に）。
Q は正規直交な列をもつので `Σ_l ‖q_l‖² = ‖Q‖_F² = K`。よって `Σ_l ‖f_l‖² = f_scale² K`、

`average_l ‖f_l‖² = f_scale² · K_true / d`。

K_true = 3、d = 12 で、事前固定の 3 条件は:

| 条件 | f_scale | 平均の ‖f_l‖²（導出） | 実装での値（20/20） | F の特異値 |
|---|---|---|---|---|
| weak | 1 | 0.25 | 0.25 | 1, 1, 1 |
| base | √2 | 0.50 | 0.50 | √2 ×3 |
| strong | 2 | 1.00 | 1.00 | 2, 2, 2 |

loading energy（f_scale²）は baseline に対して half / base / double。ただし各行の ‖f_l‖² は Q によって不均等である（strong で 0.013〜3.22）。

## 3. RNG の問題と設計

historical な mixed generator は 1 本の乱数列を Z → F → X の各列 → Y の順に使う。f_scale を変えると X の分布のパラメータが変わり、
特に Poisson の抽出で消費する乱数の数が変わりうるため、Y の法則が同じでも実現した Y が変わってしまう。これを避けるため、成分ごとに乱数列を分けた。

- **乱数列**: 各 replicate の `data_seed` から `np.random.SeedSequence(data_seed).spawn(15)`。
  child 0 = Z、1 = F の向き、2〜13 = X の列 0〜11、14 = Y。各 child の entropy と spawn_key を `stream_provenance.csv/json` に保存した。
- **Z**: 1 回だけ抽出し、3 条件で共有。
- **F**: 向きの Gaussian 行列を 1 回だけ抽出し、既存の `build_full_rank_loadings`（f_scale = 1）で Q を得て、`F_c = Q · f_scale_c`。
- **X の結合**（各列に専用の乱数列。実行前に固定）:
  - Gaussian: 標準正規のノイズ ε を共有、`X = η + √σ²_x · ε`。
  - Bernoulli: 一様乱数 U を共有、`X = 1[U < σ(η)]`（common-random-number による単調な結合）。
  - **Poisson: 一様乱数 U を共有し、逆累積分布関数 `X = F⁻¹_Pois(U; λ = e^η)`**（`scipy.stats.poisson.ppf`、U = 0 は 0）。
    各要素で `F(x−1) < U ≤ F(x)` を確認する規則を事前に置き、失敗したら方式を変えずに NEEDS_REVISION とする。**20 replicate × 3 条件のすべてで確認に合格した**。
- **Y**: 専用の乱数列の一様乱数 V を共有、`Y = 1[V < σ(−1 + z_iᵀz_j)]`（i < j）。Z・w0・w が同じなので実現した Y は 3 条件で同一になる。
- **周辺の法則**: どの条件も単独では canonical なモデル（z_i ~ N(0, I_3)、η^X = F z、X_l | z ~ 真の family、Y | Z ~ Bernoulli(σ(−1 + z_iᵀz_j))）に従う。
  一様乱数の逆変換は正確な周辺分布を与えるので、結合は条件間の対応だけを変える。Poisson の率は `canonical_poisson_rate`（clipping なし、λ_max = 1e6 の gate）で計算した。

## 4. 推論なしの preflight（20 replicate × 3 条件）

| 確認 | 合格 |
|---|---|
| Z が 3 条件で同一 | 20/20 |
| Q（F の向き）が同一 | 20/20 |
| F が `Q · f_scale` と厳密に一致（weak では F = Q） | 20/20 |
| rank(F) = 3（全条件） | 20/20 |
| **Y が 3 条件で同一** | **20/20** |
| 有限 | 20/20 |
| Poisson の逆関数の確認 | 20/20 |
| Poisson の安全（λ ≤ λ_max） | 20/20 |
| 同じ seed での再生成が一致 | 20/20 |

X は条件間で異なる（意図どおり）。generator の停止、seed の救済、retry はなかった。
**strong の安全**: Poisson の真の率の最大は 349.1（rep ごとの最大の中央値 21.5）、λ_max までの余裕は最小 2865 倍。

## 5. X の信号の文脈（min / median / max、20 replicate）

| 量 | weak | base | strong |
|---|---|---|---|
| η^X の SD（全体） | 0.47 / 0.50 / 0.54 | 0.67 / 0.70 / 0.76 | 0.94 / 1.00 / 1.07 |
| Gaussian: Var(η) / σ² の平均（記述的 SNR） | 0.16 / 0.26 / 0.51 | 0.32 / 0.52 / 1.03 | 0.63 / 1.05 / 2.06 |
| Bernoulli: 確率の SD | 0.089 / 0.112 / 0.130 | 0.122 / 0.150 / 0.170 | 0.163 / 0.195 / 0.215 |
| Bernoulli: 確率の q01 / q99（median） | 0.233 / 0.774 | 0.157 / 0.851 | 0.085 / 0.922 |
| Bernoulli: p < 0.05 / p > 0.95 の割合（median） | 0 / 0 | 0 / 0 | 0.004 / 0.003 |
| Poisson: 真の率の平均 | 1.05 / 1.13 / 1.32 | 1.14 / 1.28 / 1.74 | 1.35 / 1.65 / 3.61 |
| Poisson: 真の率の q99（median） | 3.47 | 5.83 | 12.1 |
| Poisson: 真の率の最大 | 2.8 / 4.6 / 18.7 | 4.3 / 8.8 / 62.8 | 7.8 / 21.5 / 349.1 |
| Poisson: 理論的な母集団平均 `exp(‖f_l‖²/2)` の列平均 | 1.07 / 1.14 / 1.26 | 1.14 / 1.30 / 1.57 | 1.31 / 1.73 / 2.54 |
| Y の edge 密度 | 0.313 / 0.329 / 0.352（3 条件で同一） | 同じ | 同じ |

f_scale は family ごとに違う形で効く。Gaussian では SNR がほぼ f_scale² に比例し、Bernoulli では確率の広がりと端（飽和）が増え、
Poisson では率の分布が右に長くなる（平均より最大と上側分位が大きく変わる）。

## 6. historical な Phase 9K の base との互換性

新しい対応設計の base（f_scale = √2、w0 = −1、w = 1）を、同じ data seed の Phase 9K の生成値と比べた。

| 配列 | bit 単位で一致 |
|---|---|
| Z | 0/20 |
| F | 0/20 |
| X | 0/20 |
| Y | 0/20 |

乱数列を成分ごとに分けたため（historical は 1 本の乱数列から Z を最初に引く）、Z からすでに異なる。これは設計上の意図であり失敗ではないが、
**historical な base は再利用できない**。将来の Phase 9V では base も新しい generator で実行する（**future base rerun: YES**）。

## 7. 将来の Phase 9V protocol

`future_protocol.json`: **FROZEN_NOT_EXECUTED**（`execution_authorized: false`）。

- 条件: weak f_scale = 1、base √2、strong 2。すべて w0 = −1、w = 1、n = 75、d = 12、K_true = 3、X = G3/B6/P3、Y = Bernoulli、σ²_x = 1。
- 推論の設定は Phase 9K/9T と同じ（探索・refit 8/8、K = 1..5、start_B のみ、同じ family 選択・C_Q・Candidate B、同じ 20 の replicate の label と search/refit seed）。
- Phase 9K の protocol から変わるのは `f_scale`（weak・strong）、stage、および **データ生成器**（`paired_attribute_scale.generate`、historical な `build_dataset` ではない）。
- **新しい EM の上限: 600**（3 条件 × 20 dataset × 5 K × 2 EM。base も再実行）。
- 実行前の preflight: 3 条件を推論なしで再生成し、Z・Q・Y の同一性と F の厳密なスケール関係を 20/20 で確認する。
- 主な結果（事前固定）: 技術的な完全性、K_hat_Lap / K_hat_Q、exact/under/over、属性 scale をまたいだ K の対応、delta_23、最良 vs 次点の差、K=2→3 の分解、family 選択の挙動、Candidate B の技術的な診断、family ごとの X の信号の文脈。

## 8. DECISION

## **DECISION: ATTRIBUTE_SCALE_DESIGN_READY**

20 replicate すべてで対応の不変量（Z・Q・Y の同一性、F の厳密なスケール関係、rank 3）が成り立ち、各条件の周辺の法則は canonical なまま、
f_scale = 2 でも Poisson は安全で、Poisson の逆関数の確認にも合格し、将来の protocol を凍結できた。

## 9. 解釈の範囲

READY であっても、これは「純粋な属性の情報の効果」ではない。f_scale は Gaussian の SNR、Bernoulli の飽和、Poisson の率の分布を、family ごとに違う形で変える。
将来の実験は **attribute-loading-scale sensitivity** と呼ぶ。

## 10. CAN SAY / CANNOT SAY

**CAN SAY**
- 属性の loading energy を half / base / double に変えつつ、Z と実現した Y を 3 条件で共有する canonical な合成データの対応設計を、事前に構成できた（20/20 で不変量が成立）。
- f_scale = 2 でも、この 20 seed では Poisson の率は λ_max を大きく下回った（最大 349.1）。
- この設計の base は historical な Phase 9K とは異なるデータ系列であり、Phase 9V では base も実行する必要がある。

**CANNOT SAY**
- 純粋な属性の情報の効果、family によらない信号の効果。
- K 選択に関すること（本 Phase では推論をしていない）。
- 一般的な頑健性・一致性・優越性。

## 11. 最大の UNRESOLVED

f_scale を変えると、family ごとに違う形で X の情報が変わる（Gaussian の SNR、Bernoulli の飽和、Poisson の裾）。
Phase 9V で K 選択が変わったとしても、どの family の変化が効いたのかはこの設計では分けられない。また、行ごとの ‖f_l‖² は Q によって大きく違う（列によって信号が弱い）。
