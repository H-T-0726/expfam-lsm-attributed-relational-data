# AF5 / S2-B 数理・数値の監査

- 作成：2026-11-13
- branch：`experiment/af5-local-laplace-selection`（基点は AF4 `69709e5`）
- 実装系列：lineage E ＋ PR #129 の generic model（prototype、本文採用不可）
- 新しい MCEM fit：0 回

## 1. 目的関数の導出と妥当性（Phase 0、Δ は未計算の時点）

固定するものは、基準モデル a（AF2 の refit 状態）の θ_{−j}^(a)（ほかの列の loading、Gaussian 列の分散、w0、w、var_z）と、ほかの列の分布 c_{−j}^(a) です。

候補 m ∈ {Poisson, CMP(ν=2)} について、次を定義します。

```
Φ_m(Z, f_j) = log N(Z; 0, var_z I) + Σ_{l≠j} log p_{c_l}(x_l | Z f_l) + log p_m(x_j | Z f_j) + log p(Y_train | Z; w0, w)
p_m(x_j | f_j, θ_{−j}) = ∫ exp Φ_m(Z, f_j) dZ                          （Z だけを積分）
S_j(m; a) = max_{f_j} [ Φ_m(Ẑ_m(f_j), f_j) + (nK/2) log 2π − ½ log|H_m(f_j)| ]
```

- S_j は、Z を積分した profile 周辺尤度 max_f log p(X, Y_train | f_j, θ_{−j}, c) の Laplace 近似で、f_j について**補正込みの目的関数を最大化**します（Laplace 近似による type-II 最尤）。
- θ を積分しないので **Bayesian evidence ではなく**、**BIC でも Q でもありません**。
- 両候補とも自由パラメータは f_j（K=3 個）だけなので、罰則は不要です。
- Δ_j = S_j(CMP2) − S_j(Poisson) は、Laplace 近似した profile 尤度比です。非入れ子なので、検定としての較正はありません。

**joint MAP 版との違い**：
- joint MAP 版は f_j を Φ(Ẑ(f), f) の最大化で決め、後から Laplace 項を足します。その勾配は envelope 定理により ∂Φ/∂f_j = Σ_i (x_ij − A'_m(η_ij)) ẑ_i です。
- 主目的関数の勾配には、さらに −½ tr(H⁻¹ dH/df) が加わります。dH/df は dẐ/df = H⁻¹ ∂²Φ/∂Z∂f を通じて Φ の 3 階微分を含むので、両者の最適点は一般に一致しません。
- 主解析は補正込みの目的関数だけを使い、joint MAP 版は副次の記録にとどめます。
- 補正込みの目的関数の勾配は解析的には扱いにくいので、外側は中心差分の勾配（刻み 1e-4）による BFGS で最適化し、内側の mode を厳しく解いて S を滑らかに保ちます（§2）。

**基底測度**：Poisson は h(x) = 1/x!、CMP2 は h(x) = (x!)^(−2) で、対象の列では候補によって違います。そのため、AF4 で正とした独立実装 `log_joint_parts`（`FamilySpec.log_prob` が基底測度を含む）だけを使います。

**旧実装の点検**：旧実装の `complete_log_density` は、count 列（CMP 列も "poisson" ラベル）すべてに −Σ log x! を余分に足します。対象列が Poisson でも CMP2 でも同じラベルなので、このずれは両候補で共通の定数 Σ_{count 列} log x! になります。照合でも誤差 ≤ 5e-13 でした。旧実装は mode を探す直線探索の中でしか使われず、Z によらない定数なので、mode にも Δ にも影響しません。

## 2. Phase 0 の数値監査（`phase0/`）

対象：rep01 × dense/sparse × 3 つの基準モデル × 列 {4（真 Poisson）、7（真 CMP2）} × 2 候補 ＝ 24 構成。真 Poisson 列に CMP2 を当てる構成を含みます。

| 確認 | 結果 |
|---|---|
| 勾配の有限差分（独立実装 Φ） | 相対誤差の最大 4.5e-8 |
| Hessian の有限差分 | 相対誤差の最大 6.9e-11。Y の交差ブロックのノルムは最小 14.7 |
| 正定値性 | 最小固有値は 1.70 |
| 対象列の基底測度（候補ごと） | 24/24 で、Poisson は −log x!、CMP2 は −2 log x! |
| 旧オフセット − Σ_{count 列} log x! | ≤ 4.5e-13（両候補で共通の定数） |
| joint MAP 版の envelope 勾配と有限差分 | 相対誤差 3.7e-8 |
| S の滑らかさ（有限差分の刻み 1e-4 と 1e-3） | 差 ≤ 5.0e-5 |
| 繰り返し計算の再現性 | 24/24 で同じ値 |

### 変更履歴（Δ を一度も計算していない段階）

- **試行 1**（`phase0_attempt1/`）：内側 mode の合格基準を 1e-10 にしていたため、gate は不合格でした。
  - dense の Support-only の 1 構成で、減衰付き Newton が 200 反復回っても勾配 4.4e-9 で止まりました。直線探索が丸め誤差の水準で前進できず、1e-10 は浮動小数点の精度より小さい要求でした。
  - そのため有限差分の評価点で S が −∞ になり、envelope 勾配の照合（0.43）と S の滑らかさの確認（∞）が失敗しました。
- **修正**：減衰付き Newton で 1e-11 を目指した後、純粋な Newton ステップで最大 5 回磨き上げ（勾配が減る場合だけ採用）、合格基準を AF4 で検証済みの 1e-8 にしました。
- **試行 2**：上の表のとおり合格しました。

## 3. 判定不能の基準（protocol で固定）

| 状態 | 条件 |
|---|---|
| FAILED | どちらかの候補で計算が失敗した |
| UNDECIDED_NOT_CONVERGED | どちらかの候補で、最適点での \|S の有限差分勾配\|∞ > 1e-3、または anchor の Zmode から計算し直した S との差 > 1e-6 |
| UNDECIDED_NUMERIC | \|Δ\| ≤ max(1e-2, どちらかの候補の開始点間の S の差) |
| UNDECIDED_LAPLACE | 両候補の IS が読める（相対 ESS ≥ 0.01）うえで、IS 補正で Δ の符号が変わる、または \|Δ\| < 2·SE（補正量の差の batch 標準誤差） |
| DECIDED_LAPLACE_UNASSESSED | IS が読めない。Δ の符号で判定し、印を付ける |
| DECIDED | 上のいずれにも当たらない |

IS は、両候補で同じ乱数 ε を使います（Δ_m = L_m^{−T} ε）。ペア差の batch 間のばらつきは、ESS が小さいと信頼できないので、補正量を高精度な真値としては扱いません。

## 4. Pilot（Phase 1）

pilot（`pilot/`）は rep01–03 × dense/sparse × 3 つの基準モデル × count 6 列 ＝ 108 列で、git `3dca16f`、2331 秒でした。確認したのは実行可能性の指標だけで、**正解率は見ていません**。設定は変えていません。

| 確認 | 結果 |
|---|---|
| 判定状態 | DECIDED 78、DECIDED_LAPLACE_UNASSESSED 28、UNDECIDED_NUMERIC 2。FAILED 0、UNDECIDED_NOT_CONVERGED 0（停止条件は「> 10%」で、該当しない） |
| 外側の最適化 | 最適点での \|S の有限差分勾配\|∞ は中央値 5.0e-7、最大 6.6e-6。反復は最大 14 回 |
| 2 つの開始点の差 | S の差は最大 1.1e-11（同じ最適点に到達） |
| anchor の Zmode から計算し直した S | 差は最大 9.1e-13 |
| 内側の mode | \|∇\|∞ は最大 5.4e-13。Hessian の最小固有値は ≥ 2.47 |
| IS | 読める（両候補とも相対 ESS ≥ 0.01）のは 73%。非有限の重み 0 |
| 計算時間 | 1 列（2 候補 × 2 開始点 ＋ joint MAP 版 ＋ IS）あたり約 22 秒。全 720 列で約 4.3 時間の見込み |
| メモリ | 1 回の評価（225×225 の Hessian を含む）でピーク 2.2 MB |

**判定**：Phase 2 に進む。pilot の 3 データは、本実行で同じ設定のまま計算し直す。
