# 進捗発表資料のための数式 inventory（2026-10-01）

- 位置づけ: source audit（`phase9_progress_source_audit_20261001.md`）の付属文書。**read-only**。
- 各式について: exact formula / meaning / source / code correspondence / slide candidate / 簡略化してよい点・いけない点。
- 記号の約束: n = ノード数、d = 属性列数、K = 潜在次元、L = MC サンプル数（Phase 9 は L = 5）、N = θ 罰則の標本サイズ（N = n = 75 は working convention）。
- slide 番号は `phase9_progress_storyboard_20261001.md` の番号。
- 系列: E1〜E3 は canonical model（全系列共通の生成モデル）。E4 以降は **lineage E（experimental prototype）** の実装に対応する。

---

## E1. 潜在変数の事前分布

```
z_i ~ N(0, I_K),   i = 1, …, n
```

| 項目 | 内容 |
|---|---|
| meaning | 各ノード i の K 次元の連続潜在ベクトル。latent class（離散）ではない |
| source | `CLAUDE.md` §1、`RESEARCH_MASTER.md` §4 |
| code | `reproduction/src/model.py` L.99（`var_z = 1.0`）、`eval_utils.calc_Q_dual_strict_exp` L.207–209（`ln p(Z) = −(nK/2) ln(2π var_z) − ‖Z‖²/(2 var_z)`） |
| slide | 2, 6 |
| 簡略化 OK | 「z_i は K 次元の標準正規」 |
| 簡略化 NG | z_i を「クラス」「クラスタ番号」と呼ばない（Structural EM の latent class と混同させない）。推定時に `scale_Z` で平均二乗を 1 に揃えることを省略して P_Z を説明しない（E10） |

## E2. 属性側の自然パラメータ

```
η_il^X = f_l^T z_i          （X 列 l、切片なし）
x_il | z_i ~ ExpFam_X(η_il^X)
```

| 項目 | 内容 |
|---|---|
| meaning | 列 l の loading `f_l ∈ R^K` と z_i の内積が自然パラメータ。X の尤度は列ごとに因子分解する |
| source | `CLAUDE.md` §1 |
| code | `family_selection._column_eta` L.278–281（`η^(s)_i = f_l^T z_i^(s)`）、`laplace_k_criterion._scores` L.132 |
| slide | 2, 3, 6 |
| 簡略化 OK | F = [f_1, …, f_d]^T として `η^X = Z F^T` |
| 簡略化 NG | 切片 `b_l` を書かない（現行モデルにない。設計 report §5 は切片なしが family 比較を位置適合と交絡させると注意）。従来モデルの X は Gaussian 固定であることと、標準 `DualExpFamLSM` の `family_x` が全列共通であることを混同しない |

## E3. 関係側の自然パラメータ

```
η_ij^Y = w_0 + w z_i^T z_j     （i < j、w_0, w はスカラー）
y_ij | z_i, z_j ~ ExpFam_Y(η_ij^Y)
```

| 項目 | 内容 |
|---|---|
| meaning | ペア (i, j) の関係の強さを内積で表す。w0 が全体の密度、w が潜在構造の効き方 |
| source | `CLAUDE.md` §1 |
| code | `laplace_k_criterion._scores` L.136（`eta_y = w0 + w * (Z @ Z.T)`） |
| slide | 2, 3（Phase 9 の全実験は Y = Bernoulli） |
| 簡略化 OK | 「y_ij ~ Bernoulli(σ(w0 + w z_i^T z_j))」（Phase 9 の条件） |
| 簡略化 NG | w を行列 W_Y と書かない（過去の誤り、RESEARCH_MASTER §4 表）。1/2 係数の話を持ち込むなら CLAUDE.md §2 の 5 系統を区別し、「原論文の印刷式には 1/2 がある」を守る（本資料の Phase 9 系列は fixed 由来で Y 側の extra 1/2 なし） |

## E4. family 変数（列ごとの分布族）

```
c_l ∈ M_l          （M_l は列 l の候補集合。support gate が決める）
x_il | z_i, c_l = m ~ ExpFam_m( η_il = f_l^(m)T z_i ; ψ_lm )
ψ_lm = { f_l^(m) }                 m ∈ {bernoulli, poisson}
ψ_lm = { f_l^(m), σ_l² }           m = gaussian
```

| 項目 | 内容 |
|---|---|
| meaning | 列ごとに「どの分布族で説明するか」を変数として持たせる。loading は family ごとに別物（link が違えば最適な f_l も違う） |
| source | 設計 report `automatic_family_selection_design_20260923.md` §2.1 |
| code | `family_selection.FamilySelectingPerColumnLSM`（`family_x_list` の per-column 割り当て）、`score_column_candidates` L.640–668（候補ごとに loading を別最適化） |
| slide | 5, 6 |
| 簡略化 OK | 「列ごとの family ラベル c_l」 |
| 簡略化 NG | c_l を潜在変数として事後分布を計算しているように書かない（実装は各 M-step での argmax による選択であり、c の事後や事前は置いていない）。f_l を family 共通と書かない |

### 補助式 E4-gate（support gate、FACT: code）

```
全値 ∈ {0,1}                         → M_l = {bernoulli, poisson}   decided_by = score
非負整数かつ max > 1                  → M_l = {poisson}              decided_by = gate
それ以外（非整数 または 負値を含む）   → M_l = {gaussian}             decided_by = gate
```

- source: `family_selection.support_gate` L.204–254。尤度を一切使わない決定的な規則。
- 意味: 基底測度が違う候補（counting measure の PMF と Lebesgue の density）は尤度の大小で比べられない（単位を変えると Gaussian の log-density だけが動く）ので、測度の判定を先に機械的に行う。
- **帰結（DERIVED）**: 現行の 3 family では、score による非自明な比較が起きるのは **0/1 列の Bernoulli vs Poisson だけ**。
- 簡略化 NG: 「Gaussian / Bernoulli / Poisson を全部自動判定した」と書かない。gate で決まった列を selector の正解率に含めない。

## E5. 候補 score（固定した事後サンプルの上での列ごとの完全対数確率）

```
Q_l(m, ψ_lm | q_t) = (1/L) Σ_{s=1}^{L} Σ_{i=1}^{n} log p_m( x_il | f_l^(m)T z_i^(s) ; ψ_lm ),
Z^(s) ~ q_t(Z),  s = 1, …, L     （反復 t の E-step のサンプル。scale_Z 後）
```

各 family の完全な log p（比較では基底測度の項を落とさない）:

```
Bernoulli : x η − log(1 + e^η)
Poisson   : x η − e^η − log(x!)
Gaussian  : −(x − η)²/(2σ_l²) − ½ log σ_l² − ½ log 2π     （σ_l² は残差平均二乗でプロファイル）
```

| 項目 | 内容 |
|---|---|
| meaning | EM の Q 関数のうち列 l の X 項だけを取り出したもの。`Q(θ, c | q_t) = Q_Z(q_t) + Σ_l Q_l(c_l, ψ_l | q_t) + Q_Y(w0, w | q_t)` で、**Q_Z と Q_Y は c に依存しない**ので列ごとの比較から厳密に消える（DERIVED、設計 report §2.2–2.3） |
| source | 設計 report §2.3, §4 |
| code | `family_selection.column_log_likelihood` L.284–335（`score = total / n_samples`、Poisson の `base_measure = −Σ gammaln(x+1) · L`、Gaussian の `−½ log σ²` と `−½ log 2π`）。各候補の ψ̂ は `optimise_column_loading_bfgs`（analytic gradient、maxiter 2000、gtol 1e-10） |
| slide | 6 |
| 簡略化 OK | 「事後サンプルを固定して、列ごとに各 family の対数尤度を最大化して比べる」 |
| 簡略化 NG | **一般論の Structural EM の式（期待十分統計量・BIC スコアなど）に置き換えない。** `−log(x!)` を「定数だから省略」と書かない（0/1 列では偶然 0 だが、family 比較では必要な項）。q_t を真の事後と書かない（MCEM の Laplace サンプラー分布） |

## E6. family 選択規則

```
ĉ_l = argmax_{m ∈ M_l} Q_l(m, ψ̂_lm | q_t)                 （完全同値なら bernoulli > poisson > gaussian の順）
margin_l = [−2 Q_l(次点)] − [−2 Q_l(ĉ_l)] ≥ 0              （記録のみ）
```

- 離散探索の罰則は置かない（HG-3）。`Σ_l log|M_l|` は c について定数なので selector penalty として機能しない（設計 report §6.4.1、DERIVED）。
- 運用（Scheme C, hybrid）: exploration MCEM の各反復の M-step（`calc_F`）で E6 を曖昧列に適用（A-type 更新）→ 最終反復の assignment を固定 → **同じ assignment で新規に MCEM を refit** し、その fit の Q_strict / C_Q を報告値にする。
- code: `select_from_records` L.707–734、`FamilySelectingPerColumnLSM.select_families` L.792–856、`run_hybrid_family_selection` L.1194–1294。
- slide: 6, 7。
- 簡略化 OK: 「score が大きい family を選ぶ（tie は Bernoulli 優先）」。
- 簡略化 NG: 「EM 全体として観測データ尤度を単調に改善する」と書かない（設計 report §2.4 は固定 q_t 下の coordinate ascent までに限定）。exploration の値を報告値と書かない（報告値は refit）。

## E7. K の選択規則

```
K̂ = argmin_{K ∈ {1,…,5}} C(K)          （C = C_Q または C_Lap。完全同値なら小さい K）
```

| 項目 | 内容 |
|---|---|
| meaning | 候補 K ごとに独立に（family 選択から）fit し、基準の最小の K を選ぶ |
| source | 9D report §2、`run_joint_family_k_selection.py` L.26–27 |
| code | `select_k_hat` L.215–221（`min(sorted(cq_by_k), key=…)`、許容幅なし） |
| slide | 9, 10, 13〜15 |
| 簡略化 NG | K の候補グリッド {1..5} が人の指定であることを省略しない。start を選び直していない（「良い方の start を選ぶ」はしない）ことを省略しない |

### 補助式 E7-joint（family + K 同時選択の pipeline）

```
for K in {1,…,5}:                          （K ごとにゼロから。warm start・family 持ち越しなし）
    gate → exploration（E5/E6 による A-type 更新）→ assignment ĉ(K) 固定 → fresh refit → C_Q(K)
K̂ = argmin_K C_Q(K)
```

- 形式的には `(K̂, ĉ) = argmin_{K, c} C_Q(K, c)`（設計 report §9）だが、c は各 K で exploration により決まり、K 方向に profiled されている。この profiling と離散探索のコストは p_K に入っていない（UNRESOLVED）。
- 確認: `gate75b/cq_by_k.csv` の `warm_start_source` は 30/30 行で `none`。

## E8. 現行の Q 型基準 C_Q

```
C_Q(K) = −2 Q_strict(K) + p_K ln n
Q_strict = (1/L) Σ_{s=1}^{L} [ ln p(Z^(s)) + ln p(X | Z^(s), θ̂) + ln p(Y | Z^(s), θ̂) ] + (Poisson 階乗補正)
p_K = K d − K(K−1)/2 + n_gaussian_x_cols + 1{Y = Gaussian}         （w0, w は数えない）
```

| 項目 | 内容 |
|---|---|
| meaning | EM の Q 関数（完全データ対数尤度の近似事後期待値）の MC 推定に BIC 型の罰則を足したもの。**観測データの周辺尤度ではない** |
| source | `cq_theoretical_clarification_20260927.md` §2–3、KI-010 |
| code | Phase 9 では `eval_utils.calc_Q_dual_strict_exp` L.186–229 + `eval_utils.calc_bic_exp` L.232–256（`family_x='mixed'`）。標準系列の同型関数は `utils_expfam.calc_bic_dual`（mixed 非対応）。CSV / JSON の列名は歴史的に `bic` / `BIC` |
| slide | 11 |
| 呼称 | **Q-based complete-data / ICL-type criterion**。「Schwarz BIC」と呼ばない |
| 簡略化 OK | 「C_Q = −2 × (Z を固定した完全データの当てはまり) + パラメータ数 × ln n」 |
| 簡略化 NG | 「BIC」とだけ書かない。関数名 `calc_bic_dual` を Phase 9 の実装として書かない（source audit C-01）。Phase 7e / 8b の held-out 予測スコアと混同しない（KI-019） |

## E9. C_Q の分解

```
C_Q = D_K + P_Z + P_θ
D_K = −2 (Q_X + Q_Y)        当てはまり（MC 平均）
P_Z = −2 Q_Z                潜在変数の事前の項
P_θ = p_K ln n              パラメータ罰則
```

| 項目 | 内容 |
|---|---|
| meaning | C_Q が「当てはまり」「Z の事前密度」「パラメータ数」の 3 つでできていることを見せる |
| source | `cq_decomposition_minimal_check_20260923.md`（#72）、`cq_theoretical_clarification` §2 |
| code | `run_joint_family_k_selection.cq_decomposition` L.244–288（Q_Z, Q_X, Q_Y を分けて再計算し、直接値との差 `abs_diff` を記録。9D で最大 1.8e-12） |
| slide | 11, 16 |
| 簡略化 NG | P_Z を「罰則として意図的に設計された項」と書かない（complete-data を評価した帰結として現れる。`cq_theoretical_clarification` §4） |

## E10. P_Z の閉形式

```
P_Z = nK (1 + ln 2π)                          （scale_Z と var_z = 1 のもと）
n = 75 の 1 次元あたりの増分: 75 (1 + ln 2π) = 212.8407799807…
```

導出（DERIVED）: `Q_Z = (1/L) Σ_s [ −(nK/2) ln 2π − ½ ‖Z^(s)‖² ]`。`scale_Z` が全 (i, k, s) 要素の平均二乗を 1 にするので `(1/L) Σ_s ‖Z^(s)‖² = nK`。よって `−2 Q_Z = nK ln 2π + nK`。

| 項目 | 内容 |
|---|---|
| source | `cq_theoretical_clarification` §2, §4、9D report §Diagnostic |
| code | `reproduction/src/model.py` `scale_Z` L.495–504（`scale = sqrt(k/(avg_sq·k))`）、`var_z = 1.0` |
| primary 数値 | `gate75b/cq_decomposition.csv` の P_Z = 212.84078 × K（K = 1..5 で一定、データ非依存）、`phase9y/ktrue4_summary.json` `P_Z_increment_34` = 212.8408（10/10） |
| slide | 11, 16 |
| 簡略化 NG | 「P_Z が過小選択の原因」と書かない（NOT ALLOWED）。scale_Z を外しても同じになると一般化しない（`cq_theoretical_clarification` §4 の UNRESOLVED）。係数 `1 + ln 2π` は事前分散 1 という慣行に依存する（観測データから識別できない、D-6） |

## E11. Candidate B（C_Lap）

```
C_Lap(K) = −2 [ ℓ_X(Ẑ) + ℓ_Y(Ẑ) ] + ‖Ẑ‖² + ln|H(Ẑ)| + d_K ln N
```

実装上の同値形（FACT: code）:

```
Φ(Z)      = ln p(Z) + ln p(X | Z, θ̂) + ln p(Y | Z, θ̂)          （全正規化定数・Poisson 階乗込み）
Ẑ         : ∇_Z Φ(Ẑ) = 0 を満たす同時最頻値（nK 次元、減衰 full Newton）
H(Ẑ)      = −∇²_Z Φ(Ẑ)                                           （nK × nK、Y による非対角ブロックを含む）
ℓ_Lap     = Φ(Ẑ) + (nK/2) ln 2π − ½ ln|H(Ẑ)|
C_Lap     = −2 ℓ_Lap + d_K ln N
d_K       = K d − K(K−1)/2 + n_gaussian_selected                   （Bernoulli Y の現条件。w0, w は数えない）
N         = 75                                                     （working convention）
```

var_z = 1 で `(nK/2) ln 2π` が事前の定数と相殺するので、上の 2 つは厳密に同値（DERIVED、検証 report §1。本監査で 9K rep01 K=1 の列値 `phi_at_mode`, `logdet_H` から `integration_term` = 5261.2709 を再現）。

| 項 | 意味 |
|---|---|
| `−2[ℓ_X(Ẑ) + ℓ_Y(Ẑ)]` | data fit（joint mode でのデータの当てはまり） |
| `‖Ẑ‖²` | latent prior contribution（Z の事前密度の寄与。定数部分は相殺済み） |
| `ln|H(Ẑ)|` | Laplace volume（事後の広がり＝データが Z をどれだけ決めるか。データ依存の Occam 因子） |
| `d_K ln N` | θ のパラメータ罰則（BIC 型） |

H の構造（DERIVED, 検証 report §3）:

```
H_ii = I/var_z + F^T diag(c_i) F + w² Σ_j C_ij z_j z_j^T
H_ij = w² C_ij z_j z_i^T − w R_ij I        （i ≠ j）
```

| 項目 | 内容 |
|---|---|
| meaning | Z を積分した観測データの evidence `ln p(X, Y | θ, K)` の Laplace 近似 + θ に対する BIC 型の罰則。C_Q の固定の P_Z が、データ依存の体積 `‖Ẑ‖² + ln|H|` に置き換わる |
| source | `k_true_criterion_design_comparison_20260928.md` §3（Phase 9G）、`laplace_evaluator_zero_em_verification_20260928.md`（Phase 9H）、`laplace_theta_penalty_theory_20260928.md`（Phase 9L, N = n の DECISION `RETAIN_N_EQ_n_QUALIFIED`） |
| code | `laplace_k_criterion.laplace_log_observed` L.243–279、`calc_C_Lap` L.296–312、`run_laplace_pilot.d_K` L.81–82 |
| 採用条件 | `max|∇Φ| ≤ 1e-8` かつ H 正定値。満たさなければ値なし（NOT_STATIONARY / HESSIAN_NOT_PD）。ridge / jitter は加えない |
| 実装の確認（report の測定値） | 勾配・同時 Hessian が有限差分と ~1e-9 で一致、Gaussian-X・w = 0 の厳密周辺尤度と 1.4e-14 で一致、潜在尺度変換で値と K 間差が不変（テストの許容値は 1e-6 / 相対 1e-9。source audit C-09） |
| slide | 12, 16 |
| 簡略化 OK | 「Z を積分した尤度の Laplace 近似 + パラメータ罰則」 |
| 簡略化 NG | **厳密な周辺尤度と呼ばない**（Phase 9J の IS 補正は重みが退化し、近似誤差は評価できていない）。θ̂ を最尤推定値と書かない（8 反復 MCEM の出力で、ℓ_Lap の停留点ではない: Phase 9M）。block 対角近似を使ったと書かない（joint の nK × nK）。一致性があると書かない |

## E12. K3 → K4 の差（C_Q）

```
Δ34_Q = C_Q(4) − C_Q(3) = −fit_gain_Q_34 + P_Z_increment_34 + param_increment_34
fit_gain_Q_34      = D_3 − D_4
P_Z_increment_34   = 75 (1 + ln 2π) = 212.8408
param_increment_34 = (p_4 − p_3) ln 75 = 9 ln 75 = 38.8574
K4 が K3 に勝つ ⇔ fit_gain_Q_34 > 251.6982
margin_Q = fit_gain_Q_34 − 251.6982     （正なら K4 が良い）
```

| 項目 | 内容 |
|---|---|
| source | `k34_boundary_decomposition_phase9y_20260929.md` §3–4 |
| code | `expfam/src/experimental/k34_boundary_decomposition.py` |
| primary | `phase9y_20260929/ktrue4_rows.csv`, `ktrue4_summary.json`（再構成残差 ≤ 1.45e-12） |
| slide | 16 |
| 簡略化 NG | K_true = 4 の C_Q の誤選択が全部この不等式で説明されると書かない: C_Q の K̂ は 1/5/3/1（K1..K4）で、**K1・K2 を選んだ 6 dataset は K3 vs K4 だけの比較では説明されない**。閾値 251.70 を「C_Q の欠陥の大きさ」と書かない |

## E13. K3 → K4 の差（C_Lap）

```
Δ34_Lap = C_Lap(4) − C_Lap(3) = −fit_gain_Lap_34 + volume_increment_34 + param_increment_34
D_mode(K)           = −2 Φ(Ẑ) − ‖Ẑ‖² − nK ln 2π              （= −2[ℓ_X(Ẑ) + ℓ_Y(Ẑ)]）
V(K)                = ‖Ẑ‖² + ln|H(Ẑ)|
fit_gain_Lap_34     = D_mode(3) − D_mode(4)
volume_increment_34 = V(4) − V(3)       （= Δ‖Ẑ‖² + Δ ln|H|）
param_increment_34  = 9 ln 75 = 38.8574
margin_Lap = fit_gain_Lap_34 − (volume_increment_34 + 38.8574)   （正なら K4 が良い）
```

| 項目 | 内容 |
|---|---|
| source | 9Y report §3（Phase 9P/9T の `decomposition` と同じ定義） |
| primary | `phase9y_20260929/ktrue4_summary.json`（`volume_increment_34`, `logdet_increment_34`, `zhat_sq_increment_34` を別に記録） |
| slide | 16 |
| 簡略化 OK | 「当てはまりの改善 > 体積の増分 + パラメータ罰則なら K4」 |
| 簡略化 NG | volume を「logdet の差」だけと書かない（source audit C-08）。fit_gain_Q と fit_gain_Lap を同じ量として比べない（前者は MC サンプル平均、後者は joint mode での値。fit gain の違いの理由は UNRESOLVED Z9-U1） |

---

## 付録: 式と slide の対応

| 式 | slide |
|---|---|
| E1〜E3 | 2, 3 |
| E4, E4-gate | 5, 6 |
| E5, E6 | 6, 7 |
| E7, E7-joint | 9, 10, 13〜15 |
| E8〜E10 | 10, 11 |
| E11 | 12 |
| E12, E13 | 16 |
