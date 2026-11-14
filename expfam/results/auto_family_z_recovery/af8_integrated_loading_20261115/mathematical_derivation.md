# AF8 数理の導出：loading f_j を最適化することと積分すること

AF8 は探索的研究である。θ_{−j}（AF6 の Auto anchor）は固定しているので、以下の量はモデル全体の Bayesian evidence ではない。また BIC でもない。

記号:
- n = 75、K = 3。u = (vec Z, f_j) で、dim u = 228。
- 候補 m は Poisson（P）または CMP(ν=2)（C）。
- 対数同時密度は次のとおり。どちらの候補も、同じ X に対して固有の基底測度（P は −log x!、C は −2 log x!）を含む正規化された密度である。

```
Φ_m(Z, f) = log p(X, Y_train, Z | f_j = f, θ_{−j}, m)
```

## 0. AF7 の結論の監査（訂正を含む）

### 0.1 「profile 化」という呼び方の訂正

AF5 の Laplace 版（L）と joint MAP 版（J）は、**どちらも f_j を最適化している**。違うのは最適化の目的関数だけである。

```
S_m(f)  = log ∫ exp Φ_m(Z, f) dZ  の Z についての Laplace 近似
        = Φ_m(Ẑ(f), f) + (nK/2) log 2π − ½ log|H_ZZ(f)|

L:  f_L = argmax_f S_m(f)                      S_L = S_m(f_L)
J:  f_J = argmax_f Φ_m(Ẑ(f), f)  (体積項なし)   S_J = S_m(f_J)
```

- したがって Δ_L − Δ_J は、「Hessian の体積項を含む目的関数で f_j を最適化し直したことによるスコア差」である。gain_m = S_m(f_L) − S_m(f_J) ≥ 0 の候補間の差として表せる。
- AF7 の報告でこれを「profile 化の効果」と呼んだのは不正確だった。L と J はどちらも f についての profile である。
- 「f を積分した基準」と比べるべき相手は、L（profile）である（§1）。

### 0.2 「真の θ と厳密な積分」の訂正（一次成果物で確認）

AF7 のチャット報告に「真の θ と厳密な積分（TI）を使っても、真 Poisson 列の約 18% は誤った」と書いた。これは 2 つの別々の計算を混同していた。一次成果物で確認した実際の組み合わせは次のとおり。

| 計算 | θ_{−j} | Z の積分 | f_j | 成果物 | sparse の真 Poisson 正解率 |
|---|---|---|---|---|---|
| AF6 の TI | **推定値**（AF6 anchor） | 熱力学的積分（数値） | AF5 の f_L で固定 | `af6_.../full/column_selection.csv` の `delta_ti` | 0.780 |
| AF7 の Part 3 | **真の値** | **Laplace** | 真の θ のもとで最適化し直した f_L | `af7_.../p3_true_theta_columns.csv` | 0.824 |

- 「真の θ」と「厳密な積分」を同時に使った計算は行っていない。
- 正しくは次のとおり。
  - 真の θ を使った Laplace では、真 Poisson 列の約 18% を誤った。
  - 推定した θ のまま Z を数値積分しても、Laplace と同じ程度に誤った（約 22%）。
- AF7 の README と registry の行は、2 つを別の行に記録しており、混同していない。訂正が必要なのはチャット報告と PR #170 の本文で、PR #170 には訂正を明記した。

### 0.3 KL ≥ 0 は母集団レベルの性質

次の式は、真のモデルの周辺分布 p_true による**データの期待値**についての性質である。

```
E_{X ~ p_true}[log p_true(X) − log p_alt(X)] = KL(p_true ‖ p_alt) ≥ 0
```

- 個々の有限標本で真の分布が選ばれることは保証しない。log 比の分散が KL に比べて大きければ、誤選択はいくらでも起こりうる。
- さらに、Bayes 因子 ∫p(X|f,m)p(f|m)df をこの形で使えるのは、p_true(X) が真の分布での周辺分布 ∫p(X|f,m_true)p(f|m_true)df と等しい場合だけである。つまり、f が prior から生成されていることが必要である。
- f を固定値として生成したデータ（本研究の生成器）では、E_{X|f}[log BF] ≥ 0 は一般には成り立たない。
- AF7 で「θ が真の値で f を積分すれば Bayes 因子は真の分布を支持する方向にある」と書いたのは、この条件を省いた過大な表現だった。

## 1. f を最適化することと積分すること

Z の積分を Laplace の S_m(f) で置き換えたとき（Z の誤差は AF6 の TI で Δ について中央値 −0.12 と評価済み）:

```
最適化（AF5 の L）:  S*_m = max_f S_m(f)
積分（AF8 の目標）:  M_m  = log ∫ exp(S_m(f)) p(f | m) df
```

T_m(f) = S_m(f) + log p(f|m) とおき、f_D = argmax T（prior 付きの MAP、手法 D）のまわりで f について Laplace 近似すると:

```
M_m ≈ T_m(f_D) + (K/2) log 2π − ½ log|−∇²_f T_m(f_D)|            (手法 E2)
    = S_m(f_D) + log p(f_D|m) + (K/2) log 2π − ½ log|I_m + Λ_m|
```

ここで I_m = −∇²_f S_m(f_D) は f についての曲率、Λ_m は prior の精度である。したがって:

```
M_m − S*_m ≈ [S_m(f_D) − S_m(f_L)]           (≤ 0、prior による縮小で失う当てはまり)
           + log p(f_D|m)                     (prior の密度。正規化定数を含む)
           + (K/2) log 2π − ½ log|I_m + Λ_m|  (f の体積（Occam）項)
```

- **最適化された値 S\* は f のパラメータ化に依存しない**（max は再パラメータ化で不変）。
- **積分された値 M は prior、つまり測度の選び方に依存する。**
- 候補ごとの差 Δ_M − Δ_S\* は O(1) で、prior の選び方で変わる。
- 次元が同じでも、非入れ子の 2 候補では、この O(1) の差は一般に 0 にならない。
- 積分すると、AF7 で見た「境界の位置」が prior によって明示的に定まる。積分すれば必ずずれが正しくなるわけではない。

### 1.1 prior B の平坦極限（Jacobian の効果）

prior B では、標準化した loading g = √a_m f（a_m = A''_m(0)）に N(0, s I) を置く。このとき f には N(0, (s/a_m) I) が誘導される。

```
log p(f|m) = −(K/2) log(2π s) + (K/2) log a_m − a_m ‖f‖²/(2s)
```

s → ∞ でも、候補間の差には次の項が残る。

```
(K/2) log(a_C / a_P) = 1.5 · log 0.5131 = −1.00
```

これは Poisson に有利な差である。つまり、B は「縮小」だけでなく、Jacobian による一定のずれも持つ。prior A（f に共通の N(0, s I)）では、s → ∞ でこの差は 0 になる。

## 2. 同時 Laplace（手法 E）

```
Ψ_m(u) = Φ_m(Z, f) + log p(f|m),    û = argmax_u Ψ_m,    H = −∇²_u Ψ_m(û)  (228×228)
E_m    = Ψ_m(û) + (228/2) log 2π − ½ log|H|
```

### 2.1 完全な負の Hessian のブロック（実装と有限差分で確認、§numerical_audit）

- **Z–Z ブロック（AF5/AF6 と同じ）:**
  - 対角ブロック i: I/var_z + Σ_l A''_l(η_il) f_l f_lᵀ / φ_l + Σ_k C_ik z_k z_kᵀ
  - 非対角ブロック (i, k): C_ik z_k z_iᵀ − w R_ik I
  - ここで C = M w² p(1−p)、R = M(Y − p)、M は train mask を対称化したもの。
- **Z–f ブロック:** r_i = x_ij − A'_m(η_ij) とすると、(z_i, f) のブロックは −∂²Ψ/∂z_ia∂f_b = A''_m(η_ij) f_a z_ib − r_i δ_ab。
- **f–f ブロック:** Σ_i A''_m(η_ij) z_i z_iᵀ + Λ_m（Λ_m = I/v_m は prior の精度）。

### 2.2 Schur 分解と E の性質

```
log|H| = log|H_ZZ| + log|H_ff − H_fZ H_ZZ⁻¹ H_Zf|
E_m = Φ(û) + (nK/2) log 2π − ½ log|H_ZZ(û)|      (= f_E での S_m)
    + log p(f_E) + (K/2) log 2π − ½ log|H_ff·Z|   (f の体積)
```

- **E の f の点 f_E は、Φ + log p を (Z, f) について同時に最大化した点である。** これは「prior 付きの joint MAP」で、AF6 の J と同じ種類の点である。S + log p を最大化する f_D とは異なる。
- E と E2 は同じ積分 M の別々の Laplace 近似である。
  - E2 は、Z の体積項 −½log|H_ZZ(f)| が f に依存することを、最適化点の段階で取り込んでいる。
  - E は、(Z, f) が同時に Gaussian だとみなすので、その依存を f の 2 次までしか取り込まない。
- 両者の差は、AF7 で見た「体積項を含めて f を最適化し直したことの効果」に対応する。
- そこで、IS で f を積分した値（Z は Laplace）を数値的な参照値として、E と E2 の誤差を測る。

### 2.3 確認した事項（実装と検証は numerical_audit.md）

| 項目 | 扱い |
|---|---|
| Z–f 交差ブロック、Y による Z–Z 交差ブロック、prior の Hessian | 実装し、有限差分で確認（相対誤差 < 1e-8） |
| prior の正規化定数、B の Jacobian | log p に含めた。scipy の多変量正規分布と一致することを test で確認 |
| 基底測度 | P は −log x!、C は −2 log x!。同じ X に対する正規化された密度 |
| 回転と尺度 | θ_{−j}（他列の F と Y の w）を固定しているので、Z の回転と尺度は固定される。prior は等方的なので、anchor の Z の座標系に依存しない |
| mode の存在と一意性 | 複数の開始点（anchor の plug-in、0、AF6 の f_L、f_D）で、Ψ の最大値の差を記録 |
| 正定値性 | H の最小固有値を記録し、≤ 0 なら FAILED_NUMERIC |
| 数値積分の誤差 | 主 prior で E2 と IS（t 分布の proposal、400 点）と比べる。Z の Laplace 誤差は AF6 の TI で評価済みで、ここでは扱わない |
| prior スケールへの感度 | 事前に固定した 9 種類の prior（prior_design.md） |
| CMP2 の大きな η | Bessel 経路、u > 1e3 では 1 − I1/I0 を漸近級数で計算（test で確認） |

## 3. 手法 D と E の対比（縮小と積分の切り分け）

- D = max_f T(f)：prior による縮小はあるが、f の体積項はない。
- E は f を積分する（同時 Laplace）。
- E − D には、f の体積項 −½log|H_ff·Z| + (K/2) log 2π と、評価点の違い（f_E と f_D）が入る。
- E2 − D = (K/2) log 2π − ½ log|−∇²T| は、純粋な f の体積項である。
- 「積分による改善」と言えるのは、E（または E2/IS）が D より良い場合に限る。
