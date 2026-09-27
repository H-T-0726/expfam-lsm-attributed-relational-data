# Candidate B（Laplace 観測尤度）評価関数の最小実装と zero-EM 検証（Phase 9H / Issue #88）

- 日付: 2026-09-28
- 位置づけ: **MINIMAL IMPLEMENTATION + ZERO-EM VERIFICATION**。EM 0、pilot 0、K 選択実験 0、C_Q との比較 0。
- baseline: `origin/main = 093172a`（Phase 9G PR #87 merge 済み）
- 実装: `expfam/src/experimental/laplace_k_criterion.py`、検証: `expfam/src/experimental/test_laplace_k_criterion.py`（17 tests）
- ラベル: **FACT** / **DERIVED** / **VERIFIED**（本タスクの zero-EM テストで確認）/ **UNRESOLVED**
- 系列 E（experimental prototype; 本文採用不可）

---

## 1. 実装した量

固定した θ（モデルの現在のパラメータ）での Laplace 近似:

```
laplace_log_observed = Phi(Z_hat) + (nK/2) log(2π) − ½ log|H(Z_hat)|
Phi(Z) = log p(Z) + log p(X|Z,θ) + log p(Y|Z,θ)
C_Lap  = −2 · laplace_log_observed + d_K · log(N)      （calc_C_Lap; 別関数）
```

- **FACT**: `Phi` は既存の `eval_utils.calc_Q_dual_strict_exp` を単一の Z（L=1）で評価したもの。
  事前 `−(nK/2)log(2π var_z) − ‖Z‖²/(2 var_z)`、全正規化定数、Poisson の階乗補正を含む。**新しい尤度は書いていない。**
- **DERIVED**: canonical（var_z = 1）では `(nK/2) log 2π` が事前の定数と相殺し、Phase 9G の式
  `ℓ_X + ℓ_Y − ½‖Ẑ‖² − ½ log|H|` と一致する。実装は相殺前の一般形で評価するので、var_z ≠ 1（尺度検証）でもそのまま使える。
- 積分部分（`laplace_log_observed`）とパラメータ項（`calc_C_Lap`）は別関数である。`calc_C_Lap` は `d_K` と `N` を**呼び出し側に必須で要求**する（N に既定値はない）。
  `loading_parameter_count(K, d) = Kd − K(K−1)/2` は Phase 9G の慣行を返す補助関数で、Gaussian-X 分散や w0/w は呼び出し側が加える。
  N = n は working convention であり、理論的に確定した sample size として埋め込んでいない。

## 2. joint mode の定義

- `Ẑ` は `∇_Z Phi(Ẑ) = 0` を満たす**決定論的な同時最頻値**。E-step の `calc_eta_newton`（最後に事後サンプリングする）は使っていない。
- 解法: 同時 Hessian による減衰付き full Newton。`Phi` での backtracking で単調に上昇させる。途中の反復で H が正定値でなければ勾配方向を使う
  （最適化の手段であって、評価する量は変えない）。
- 採用条件: 最終点で `max|∇Phi| ≤ grad_tol`（既定 1e-8）。満たさなければ `NOT_STATIONARY` を返し、値は NaN。
- 初期値 Z0 は呼び出し側が渡す（例: refit の Z_est）。同じ入力なら同じ出力になる（VERIFIED: 2 回の評価がビット単位で一致）。
- MCEM 本体・scale_Z・現行 C_Q は変更していない。

## 3. Hessian の式（DERIVED; 実装は model の mean/curvature hook を使用）

X 列 l のスコア `s_il = (x_il − μ_il)/φ_l`、曲率 `c_il = A''(η_il)/φ_l`（Gaussian 列は φ_l = σ_l²、Bernoulli/Poisson は φ_l = 1）。
Y は `η_ij = w0 + w z_iᵀ z_j`、`R_ij = m_ij (y_ij − μ_ij)`、`C_ij = m_ij A''(η_ij)`（m はペアマスク、対角 0）。

```
∇_{z_i} Phi = −z_i/var_z + Fᵀ s_i + w Σ_j R_ij z_j
H_ii = I/var_z + Fᵀ diag(c_i) F + w² Σ_j C_ij z_j z_jᵀ
H_ij = w² C_ij z_j z_iᵀ − w R_ij I          (i ≠ j),   H_ji = H_ijᵀ
```

`Σ_{i≠j}` の ½（順序付きペアの二重数え）は、z_i が (i,j) と (j,i) の 2 回現れることで打ち消される。

## 4. 勾配の有限差分（VERIFIED）

固定の小データ（n=6, K=2、X = [Gaussian, Bernoulli, Poisson, Gaussian]、w = 0.9）で、`Phi` の中心差分（h = 1e-5）と解析勾配を nK = 12 成分すべてで比較した。

| Y family | max_abs | max_rel |
|---|---|---|
| Bernoulli | 5.47e-10 | 4.07e-10 |
| Poisson | 8.55e-10 | 8.55e-10 |

許容値 1e-6: 中心差分の打ち切り誤差 O(h²|Phi'''|) と丸め誤差 O(ε|Phi|/h) はどちらも ~1e-9 なので、3 桁の余裕がある（テスト内にコメント）。

## 5. Hessian の有限差分（VERIFIED）

解析勾配の中心差分（h = 1e-6）と解析 H を、全 12×12 成分で比較した。

| Y family | max_abs | max_rel | 非対称性 max\|H−Hᵀ\| | 非対角ブロック max\|H_ij\| (i≠j) |
|---|---|---|---|---|
| Bernoulli | 5.72e-10 | 5.72e-10 | 1.1e-16 | 0.890 |
| Poisson | 1.03e-09 | 1.03e-09 | 4.4e-16 | 5.843 |

- relational の非対角ブロックは実際に非ゼロであり、有限差分と一致した。
- `H_ij = H_jiᵀ` は全ペアで 1e-12 以内。
- Bernoulli の 1 ブロック（i=0, j=3）を §3 の閉じた式と直接照合し、1e-13 以内で一致した。

## 6. Gaussian-X・w = 0 での厳密な周辺尤度（VERIFIED; 最重要）

固定パラメータ（n=8, d=4、Σ_X 対角 [0.5, 1.0, 1.4, 0.8]、w = 0、w0 = −0.4）で、
`x_i ~ N(0, FFᵀ + Σ_X)` の厳密な対数尤度に、Y の定数 η（= w0）での Bernoulli 対数尤度を加えたものと、Candidate B の Laplace 値を比較した。
正規化定数・log det・事前の定数・`(nK/2) log 2π` をすべて含む。

| K | Laplace | 厳密値 | \|差\| | 最終 max\|∇Phi\| |
|---|---|---|---|---|
| 1 | −68.512178975325 | −68.512178975325 | 1.4e-14 | 3.8e-16 |
| 2 | −72.073875104162 | −72.073875104162 | 1.4e-14 | 7.8e-16 |
| 3 | −75.364602196682 | −75.364602196682 | 1.4e-14 | 1.0e-15 |

K ごとに異なる定数（`(nK/2) log 2π`、`log|H|` の次元）を落としていないことが、3 つの K での一致で確認できる。

## 7. 潜在尺度不変性（VERIFIED）

`z' = s z`、事前分散 1 → s²、`F' = F/s`、`w' = w/s²`（観測モデルは同一）。

- **Gaussian-X・w=0**（s = 0.5, 3.0）: Laplace 値が相対 1e-9 以内で一致。
- **mixed X ＋ Bernoulli Y・w = 0.9**（s = 0.5, 2.0; K = 1, 2）: 最大差 6.6e-11。
  最頻値も `Ẑ' = s Ẑ` に対応して移る（1e-7 以内）。
- **K 順序**: `value(K=2) − value(K=1)` はスケール変更の前後で 1e-7 以内で一致（絶対値と K 間の差の両方が不変）。

## 8. 対応・非対応の経路

| 項目 | 対応 |
|---|---|
| モデル | `DualExpFamLSMConsistent`、`DualExpFamLSMPerColumnConsistent` |
| X 列 | Gaussian（分散 = model の sigma 対角）、Bernoulli、Poisson |
| Y | Bernoulli、Poisson（canonical `w0 + w z_iᵀ z_j`）、対称 Y、対称でゼロ対角のペアマスク |
| **非対応（`UnsupportedLaplacePath` を送出）** | Gaussian-Y、NB、legacy（non-consistent）モデル、非対称 Y、非対称マスク |

非対応の経路に暗黙の fallback はない（VERIFIED: Gaussian-Y、非対称 Y、legacy masked model で例外になる）。

## 9. 数値的失敗の扱い

- 最終点が停留条件を満たさない → `NOT_STATIONARY`、値は NaN（VERIFIED）。
- H の `slogdet` の符号が ≤ 0、または最小固有値が ≤ 0 → `HESSIAN_NOT_PD`、値は NaN。**ridge や jitter は加えない**（VERIFIED）。
- `0.5 (H + Hᵀ)` の対称化は浮動小数点の丸めの処理だけで、科学的な量は変えない（解析的には H は厳密に対称）。
- `calc_C_Lap` は status が OK でなければ `C_Lap = NaN` を返し、status を持ち回る。

## 10. 残る理論上の限界（UNRESOLVED）

1. **非 Gaussian 事後での Laplace の精度**: 厳密に確かめたのは事後が Gaussian の場合だけ。Bernoulli/Poisson X と relational Y では Laplace は近似であり、
   n=75、K≤5 で K によって異なる誤差を持つかは未確認（Phase 9G の最大の UNRESOLVED のまま）。
2. **Z の事後の近似的な回転対称性・多峰性**: X が弱いと単一モードでの Laplace は体積を誤りうる。評価関数は H の正定値性を報告するが、多峰性は検出しない。
3. **θ̂ の質**: Laplace＋BIC は θ̂ が `log p(X,Y|θ)` の最大化点であることを前提にする。pilot で使う θ̂ は 8 反復 MCEM の出力（scale_Z の慣行下）で、完全な最尤ではない。
4. **パラメータ項**: `N = n`、`d_K`（F のみ K 依存）は Phase 9G の working convention。特異性・非入れ子性・family 選択のコストは扱っていない（追加していない）。
5. 実際の規模（nK = 375）で、同時最頻値の収束と H の正定値性が成り立つかは、fitted θ̂ がないので本タスクでは確認していない（pilot で status として観測される）。

## 11. Decision

## **READY_FOR_FROZEN_PILOT**

意味するのは、**Candidate B の評価関数が数式と一致し、小規模な事前固定 pilot で使える程度に実装が正しい**ことだけである:
勾配・同時 Hessian（relational の非対角ブロックを含む）は有限差分と ~1e-9 で一致し、
Laplace が厳密になる Gaussian-X・w=0 の場合に厳密な周辺尤度と 1.4e-14 で一致し、潜在尺度の変更に対して値と K 間の差が不変である。
非停留・非正定値の場合は値を出さずに status を返す。

**意味しないこと**: Candidate B が K_true を正しく・一致的に選ぶこと、現行 C_Q より優れること。
§10 の UNRESOLVED（特に非 Gaussian 事後での Laplace 精度と θ̂ の質）は pilot の設計で扱うべき点として残る。
pilot 自体は別の Human Gate（frozen protocol）が必要で、本タスクでは開始していない。
