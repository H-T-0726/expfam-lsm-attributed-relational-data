# Family-selection hard case 監査：no-intercept Poisson と「0/1 に見える真 Poisson 列」（2026-10-02）

- 位置づけ: **THEORY / IMPLEMENTATION AUDIT / ZERO-EXPERIMENT**。
- **新しい EM 0、新しい synthetic dataset 0、新しい seed run 0、コード変更 0。**
  数値は (i) 決定論的な数値積分（Gauss–Hermite 200 点、Gauss–Jacobi 200 点、乱数なし）と
  (ii) commit 済み artifact（`generator_provenance.csv` / `family_by_k.csv` / `summary.json`）の read-only 集計のみ。
  計算スクリプトは付録 A に全文を載せた（repo には追加していない）。
- 系列: family selection は **lineage E（`DualExpFamLSMPerColumnConsistent`、experimental prototype、本文採用不可）**。
- base: `origin/main` = `ee4be23`（2026-10-02 時点）。
- ラベル: **FACT**（コード・artifact・docs で直接確認）／ **DERIVED**（数式から導出）／
  **NUMERICAL**（決定論的数値評価）／ **INTERPRETATION** ／ **UNRESOLVED**。

---

## 0. 結論（先に）

1. **FACT** 現行の全 lineage（DualExpFamLSM・fixed・per-column・objective-consistent・family-selection・
   mixed generator・C_Q・C_Lap）で X 側は `η_il = f_l^T z_i` であり、X 列 intercept は存在しない（§1）。
2. **DERIVED** no-intercept では `η_il ~ N(0, ‖f_l‖²)`、Poisson の母集団平均 rate は
   `E[λ_il] = exp(‖f_l‖²/2) ≥ 1`（等号は `f_l = 0` のみ）。**平均 rate を 1 未満にすることはできない**（§2）。
   Bernoulli の母集団 1 率は `‖f_l‖` によらず **ちょうど 1/2**（§3）。
3. **NUMERICAL** 1 観測が {0,1} に入る周辺確率は `‖f_l‖ = 0` で最大 `2/e = 0.7358` であり、
   n = 75 では **P(列の全観測が {0,1} | f_l) ≤ 1.01×10⁻¹⁰**（任意の f_l について）。
   既存条件（f_scale = √2, K = 3, d = 12）で F について周辺化すると **4.5×10⁻¹²**／列。
   commit 済みの全 family-selection 系 artifact（unique な真 Poisson 列 428 本）での**期待発生数の合計は 2.3×10⁻⁷**（§4）。
   → `ambiguous_true_poisson = 0` は偶然ではなく **STRUCTURALLY RARE**。
4. **DERIVED（本監査の主結果）** {0,1} の値 x について、全ての η で
   `log p_Pois(x|η) = log p_Bern(x|η) + log P_Pois(X ≤ 1 | η)` が成り立ち、右辺第 2 項は常に負。
   したがって、**同じ Z サンプル・同じパラメータ数で尤度を比べる限り、0/1 列では Bernoulli の score が
   Poisson の score を必ず上回る**（データの真の family によらない）（§5）。
   この結果、現行 prototype（3 family + support gate + 完全尤度 score）では
   **family の割り当ては観測 support の決定論的関数**になり、score が結果を変えることはない。
5. **OBSERVED / INTERPRETATION** pilot の 36/36（および Phase 9 の全 score 判定行）は、
   上の理論的順序と実装が既存 C2 artifact（Phase 9 artifact）で一致したことの確認である（§6）。
   データから family を判別した能力の証拠ではなく、**一般的な family 判別精度 100% は支持されない（NOT SUPPORTED）**。
   仮に 0/1 に見える真 Poisson 列が現れても、selector は**必ず Bernoulli を選ぶ**。

---

## 1. canonical X model の確認（FACT）

| 対象 | 確認箇所 | X の自然パラメータ | intercept |
|---|---|---|---|
| canonical docs | root `CLAUDE.md` §1、`RESEARCH_MASTER.md` L80（`x_il ~ ExpFam_X( η_il^X = f_l^T z_i )  バイアスなし`）、L97（旧 eq(2) のバイアスありは「誤（旧）」） | `f_l^T z_i` | なし |
| theory audit | `reports/theory_audit/theory_audit_report_20260718.md` L72, L98 | 同上 | 「全系列共通」でなし |
| `DualExpFamLSM`（0.5 あり） | `expfam/src/model_dual_expfam.py` L140（`eta_x_i = F @ z_i`）、L249 / L312（`eta_x = Z_l @ F.T`） | `F z_i` | なし |
| fixed | `expfam/src/model_dual_expfam_fixed.py` L60 | `F z_i` | なし |
| `DualExpFamLSMPerColumn` | `expfam/src/experimental/model_dual_expfam_percolumn.py` L126, L180, L206, L222 | `F z_i` | なし |
| objective-consistent（lineage E） | `expfam/src/experimental/model_dual_expfam_consistent.py` L90, L150 | `Z F^T` | なし |
| family selection | `expfam/src/experimental/family_selection.py` L42（**HG-5 No X-column intercept**）、L288–291（`_column_eta = einsum(Z_samples, loading)`） | `f_l^T z` | なし |
| mixed generator | `expfam/src/experimental/data_generator_canonical_mixed.py` L187（`eta_x = latent @ loadings.T`） | `Z F^T` | なし |
| generator metadata | 同 L255–260：Poisson 列の `expected_x_mean = exp(‖f_l‖²/2)` | — | — |
| F の構成 | `data_generator_canonical.py` L127–169：Gaussian 行列の reduced QR（符号固定）× `f_scale` | `F = f_scale · Q` | — |
| M-step / candidate optimiser | `family_selection.py` L348–377（勾配 `Σ (x − μ(η)) z`、`η = f_l^T z`） | `f_l^T z` | なし |
| C_Q のパラメータ数 | `expfam/src/utils_expfam.py` L399–402：`kd − k(k−1)/2 + (Gaussian-X なら d) + (Gaussian-Y なら 1)` | — | intercept の項なし（w0, w も数えない） |
| C_Lap の d_K | `laplace_k_criterion.py` L286–291 `loading_parameter_count = kd − k(k−1)/2`、`run_laplace_pilot.py` L82 で `+ n_gaussian_selected` | — | intercept の項なし |

**FACT** Y 側には intercept `w_0^Y` がある（`η^Y = w0 + w z_i^T z_j`）。X 側だけが intercept を持たない非対称な構造である。

**FACT** family-selection design（`reports/distribution_selection/automatic_family_selection_design_20260923.md` §5）は
すでに「η = 0 での既定水準は Gaussian 0 / Bernoulli 0.5 / Poisson 1 に固定される」ことを指摘し、
X intercept の導入を HG-5（Human Gate）として保留している。本監査はこれを**母集団分布と support の確率**まで延長する。

---

## 2. no-intercept Poisson の母集団分布（DERIVED）

前提: `z_i ~ N(0, I_K)`（iid）、`f_l ∈ R^K` は固定、`s_l² := ‖f_l‖²`。

**(2.1)** `η_il = f_l^T z_i` は Gaussian の線形変換なので `η_il ~ N(0, s_l²)`。異なる i について iid。

**(2.2)** `λ_il = exp(η_il)` は log-normal(0, s_l²)：

```
E[λ_il]    = exp(s_l²/2)
Var[λ_il]  = (exp(s_l²) − 1) exp(s_l²)
median     = 1
```

**(2.3)** 列の周辺平均と分散（全分散の公式）:

```
E[x_il]   = E[λ_il] = exp(s_l²/2) ≥ 1                  （等号 ⇔ f_l = 0）
Var[x_il] = E[λ] + Var[λ] = exp(s_l²/2) + (exp(s_l²) − 1) exp(s_l²)
```

**結論（DERIVED）**: no-intercept model では、**Poisson 列の母集団平均 rate は常に 1 以上**であり、
`f_l` をどう選んでも 1 より小さくできない。`f_l` を大きくすると平均も過分散も増えるだけである。
low-rate な count 列（平均 0.1 など）はこのモデルの族に含まれない。

**(2.4) 1 観測の確率**（`h_x(η) = P(X = x | η)`）:

```
P(X=0) = E[exp(−λ)]
P(X=1) = E[λ exp(−λ)]
P(X≤1) = E[h(η)],   h(η) := exp(−e^η)(1 + e^η)
P(X≥2) = 1 − P(X≤1)
```

**(2.5) 「n 観測すべてが {0,1}」の確率 — 条件付きと周辺化を区別する**

- 条件付き（Z と f_l を所与）: `P(全 i で x_il ∈ {0,1} | Z, f_l) = Π_i h(f_l^T z_i)`（乱数 Z に依存する量）。
- Z について周辺化（f_l 所与）: z_i が iid なので `P(全 n 観測 ∈ {0,1} | f_l) = p(s_l)^n`、`p(s) := E_{η~N(0,s²)}[h(η)]`。
- F について周辺化（generator 所与）: `F = f_scale · Q`、Q は Gaussian 行列の QR（符号固定）なので
  Stiefel 多様体上の Haar 分布に従い、行 l の二乗ノルムは `‖q_l‖² ~ Beta(K/2, (d−K)/2)`
  （固定単位ベクトルを一様ランダムな K 次元部分空間に射影した長さの二乗）。よって
  `s_l² = f_scale² · B`、`B ~ Beta(K/2, (d−K)/2)`、`E[s_l²] = f_scale² K/d`。
  `P(全 n 観測 ∈ {0,1}) = E_B[p(f_scale √B)^n]`。
  （既存条件 f_scale = √2, K = 3, d = 12 で `E[s²] = 0.5`。artifact に記録された `f_row_norm_sq` の範囲 0.006–1.48 と整合。）

---

## 3. Bernoulli 側との比較（DERIVED + NUMERICAL）

**(3.1) DERIVED** `σ(−η) = 1 − σ(η)` と `η ~ N(0, s²)` の対称性から
`P(X=1) = E[σ(η)] = 1/2`（**s によらず厳密**）。no-intercept Bernoulli 列の母集団 1 率は常に 0.5 である
（canonical の `RESEARCH_MASTER.md` P7 と同じ事実）。

**FACT（artifact）** commit 済み全 artifact の真 Bernoulli 列の経験平均は 0.307–0.653、
真 Poisson 列の経験平均は 0.747–7.627、真 Poisson 列の最大値の最小は 3（pilot C2 では 4）。

**(3.2) NUMERICAL（表 T1, η ~ N(0, s²), n = 75）**

| s² = ‖f‖² | Pois E[λ] | Pois P0 | Pois P1 | Pois P≥2 | Pois P≤1 | (P≤1)^75 | Bern P1 |
|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 0.3679 | 0.3679 | 0.2642 | 0.7358 | 1.01e-10 | 0.5 |
| 0.1 | 1.051 | 0.3683 | 0.3505 | 0.2813 | 0.7187 | 1.75e-11 | 0.5 |
| 0.25 | 1.133 | 0.3699 | 0.3280 | 0.3022 | 0.6978 | 1.91e-12 | 0.5 |
| **0.5（既存の平均）** | 1.284 | 0.3737 | 0.2986 | 0.3277 | 0.6723 | **1.17e-13** | 0.5 |
| 1.0 | 1.649 | 0.3818 | 0.2589 | 0.3594 | 0.6406 | 3.12e-15 | 0.5 |
| 2.0 | 2.718 | 0.3950 | 0.2134 | 0.3916 | 0.6084 | 6.52e-17 | 0.5 |
| 4.0 | 7.389 | 0.4122 | 0.1682 | 0.4197 | 0.5803 | 1.89e-18 | 0.5 |
| 16.0 | 2981 | 0.4472 | 0.0945 | 0.4584 | 0.5416 | 1.07e-20 | 0.5 |

- Bernoulli: 0 率 = 1 率 = 0.5、P(X≥2) = 0（support 上ありえない）。
- Poisson: P(X≥2) は **最小でも 0.264**（s = 0）。s を大きくすると P0 は増えるが P≥2 も増え、P(X≤1) は単調に下がる。
- **NUMERICAL** `p(s) = P(X≤1)` を s ∈ [0, 6]（刻み 0.001）で評価すると最大は s = 0 の `2/e = 0.735759`、
  s = 6 で 0.528（s → ∞ で 1/2 に近づく：η > 0 の半分では λ → ∞、η < 0 の半分では λ → 0）。
  単調性の一般証明は行っていない（格子上の数値結果）。

**(3.3) support gate が score 比較をする領域**

support gate（`family_selection.support_gate`, L214–264）は観測 support だけで決まる:

| 観測 support | gate の判定 | 真 Gaussian | 真 Bernoulli | 真 Poisson（no-intercept, n = 75） |
|---|---|---|---|---|
| 非整数 or 負を含む | Gaussian（gate） | 確率 1 | 確率 0 | 確率 0 |
| 非負整数・最大 ≥ 2 | Poisson（gate） | 確率 0 | 確率 0 | `1 − p(s)^75 ≥ 1 − 1.01e-10` |
| 全て {0,1} | **score 比較** | 確率 0 | **確率 1** | `p(s)^75 ≤ 1.01e-10` |

**DERIVED** つまり score 比較に入る列は、no-intercept の canonical draw では**事実上すべて真 Bernoulli**である。
「score 比較が起きるかどうか」と「どの真 family が score 比較に入るか」は、selector の能力ではなく
**モデル構造（intercept なし）と n で事前に決まっている**。

---

## 4. 既存 pilot 条件での ambiguous true-Poisson の確率（NUMERICAL）

**(4.1) F について周辺化した 1 列あたりの確率（表 T2）**

| 条件 | f_scale² | K | d | E‖f‖² | E_F[P(全 75 観測 ∈ {0,1})] | 3 列のうち ≥1 列（和の上界） |
|---|---|---|---|---|---|---|
| pilot C2 / 9E / 9K / 9P / 9T / 9V-base | 2.0 | 3 | 12 | 0.50 | **4.49e-12** | ≤ 1.35e-11 |
| 9V weak | 1.0 | 3 | 12 | 0.25 | 1.08e-11 | ≤ 3.24e-11 |
| 9V strong | 4.0 | 3 | 12 | 1.00 | 1.73e-12 | ≤ 5.18e-12 |
| 9X K1 | 6.0 | 1 | 12 | 0.50 | 2.20e-11 | ≤ 6.61e-11 |
| 9X K2 | 3.0 | 2 | 12 | 0.50 | 8.98e-12 | ≤ 2.70e-11 |
| 9X K4 | 1.5 | 4 | 12 | 0.50 | 2.52e-12 | ≤ 7.55e-12 |

**(4.2) commit 済み artifact の各真 Poisson 列について、記録された `f_row_norm_sq` を所与とした確率の和（表 T3）**

| artifact | 真 Poisson 列の数（インスタンス） | 期待 ATP 数 |
|---|---|---|
| family_selection/pilot_c2_20260925 | 9 | 1.65e-11 |
| family_selection/smoke{,_v2,_v3}（n = 40, d = 6, f_scale = 1） | 各 2 | 各 2.32e-07 |
| joint_family_k_selection/gate75b_20260927 | 9 | 1.20e-11 |
| k_repeatability/phase9e_20260927 | 60 | 2.40e-10 |
| lap_vs_cq_20/phase9k_20260928 | 60 | 2.45e-10 |
| relational_w_sensitivity/phase9p（weak / strong） | 各 60 | 各 2.45e-10 |
| density_controlled_w_sensitivity/phase9t（weak / strong） | 各 60 | 各 2.45e-10 |
| attribute_scale_sensitivity/phase9v（weak / base / strong） | 各 60 | 5.49e-10 / 2.40e-10 / 1.19e-10 |
| matched_k_true_sensitivity/phase9x（K1 / K2 / K4） | 各 30 | 3.98e-10 / 2.96e-10 / 5.14e-11 |
| laplace_pilot/phase9i、laplace_is_diagnostic/phase9j | 各 9 | 9.64e-11 / 3.18e-11 |
| **重複を除いた合計**（data_seed・列・‖f‖²・列平均で同一視、428 列） | — | **2.34e-07**（最大の 1 列 1.89e-07、smoke の n = 40） |

（9K / 9P / 9T は同じ X を共有しているため、重複除去後の合計を正とする。）

**(4.3) n についての余地（NUMERICAL）** 最良の場合（f_l = 0、`p = 2/e`）でも
`P(全 n 観測 ∈ {0,1}) ≥ 0.5` となるのは **n ≤ 2**、≥ 0.05 は **n ≤ 9**、≥ 0.01 は **n ≤ 15**。
s² = 0.5 ではそれぞれ n ≤ 1 / 7 / 11。

**判定**: `ambiguous_true_poisson = 0` は**偶然ではない**。no-intercept の canonical model では、
n = 75 で 0/1 だけの真 Poisson 列は 1 列あたり ≤ 10⁻¹⁰ でしか起きず、既存の全 artifact を合わせても
期待発生数は 2.3×10⁻⁷ である。**分類: STRUCTURALLY RARE**（n が 1 桁なら起きうるが、それは既存の K 選択条件と両立しない）。

---

## 5. 主結果：0/1 列では尤度 score は常に Bernoulli を選ぶ（DERIVED）

### 5.1 恒等式

x ∈ {0,1}、任意の実数 η について（Poisson の基底測度 `−log x! = 0`、`family_selection.py` L325–330 は完全尤度でこれを計算している）:

```
log p_Bern(x | η) = x η − log(1 + e^η)
log p_Pois(x | η) = x η − e^η − log x!  =  x η − e^η

log p_Pois(x | η) − log p_Bern(x | η) = log(1 + e^η) − e^η
                                       = log[ e^{−λ}(1 + λ) ]      (λ = e^η)
                                       = log P_Pois(X ≤ 1 | η)
                                       =: −g(η)
```

`g(η) = λ − log(1+λ) > 0`（`log(1+λ) < λ`, λ > 0）。すなわち

> **Poisson の 0/1 データに対する尤度 = Bernoulli の尤度 × 「Poisson のもとで 2 以上が一度も出なかった確率」**。

同値な見方: `P_Pois(X = 1 | X ≤ 1, η) = λ/(1+λ) = σ(η)`。**{0,1} に打ち切った Poisson は、同じ自然パラメータの Bernoulli そのもの**である。
0/1 データが Poisson を支持する情報は「2 以上が出なかった」ことだけであり、それは Poisson にとって常に不利な情報である。

### 5.2 score の順序

`family_selection.select_families`（L792–815）は各 ambiguous 列について、**同じ `Z_samples`** 上で
候補ごとに `f_l` を最適化し（`score_column_candidates`）、ペナルティなし（HG-3）で score 最大の候補を選ぶ（`select_from_records`, L724–733）。
`S_c(f) = (1/L) Σ_s Σ_i log p_c(x_il | f^T z_i^(s))` とおくと、§5.1 から任意の f で

```
S_Pois(f) = S_Bern(f) − (1/L) Σ_s Σ_i g(f^T z_i^(s))  <  S_Bern(f)
```

よって `f̂_P = argmax S_Pois` について

```
max_f S_Bern(f)  ≥  S_Bern(f̂_P)  >  S_Pois(f̂_P)  =  max_f S_Pois(f)

margin_neg2 = −2 max S_Pois − (−2 max S_Bern)  ≥  (2/L) Σ_s Σ_i g(f̂_P^T z_i^(s))  >  0
```

**DERIVED**: 0/1 列では、**真の family・n・‖f‖・K・Z サンプルによらず、Bernoulli が厳密に勝つ**。
CANDIDATE_PRIORITY（tie rule）が効く場面は存在しない（厳密な不等式）。

### 5.3 前提と適用範囲

| 前提・拡張 | 成立 | 理由 |
|---|---|---|
| 両候補が同じ Z サンプルで評価される | FACT | `select_families` が同じ `Z_samples` を渡す |
| 両候補の追加パラメータ数が同じ（0） | FACT | design §6.1、HG-3（ペナルティなし） |
| Bernoulli 候補が少なくとも `S_Bern(f̂_P)` に到達 | 凹関数の BFGS で成立する想定 | 非収束 warning（pilot C2 で最終 8/72、grad_inf ≤ 1.30e-6）がある。margin ≥ 45 に対し score 誤差は極めて小さいと考えられるが**再測定はしていない**（INTERPRETATION） |
| X intercept を入れても成立 | DERIVED | §5.1 は η ごとの恒等式なので `η = b + f^T z` でもそのまま成立 |
| 同じ事前分布での厳密な周辺尤度（Bayes factor） | DERIVED | 被積分関数が各点で小さいので積分も小さい |
| C_Q の X 項（per-column の MC 平均）で family を比べる場合 | DERIVED | 同じ不等式。パラメータ数も同じ |
| C_Lap（Laplace 近似）で family を比べる場合 | **未証明** | mode と Hessian が family で変わる。曲率は `e^η > σ(η)(1−σ(η))` で Poisson の方が常に大きいが、異なる mode での log det の比較は一般には言えない |
| Poisson を勝たせうるもの | DERIVED | family の事前オッズ、Poisson に有利なペナルティ、または 2 以上を含みうる held-out の予測評価など、**尤度以外の要素**だけ |

### 5.4 帰結：family の割り当ては support の関数

**DERIVED** 現行 3 family・support gate・完全尤度 score のもとで、canonical draw に対する最終割り当ては

```
非整数 or 負を含む   → Gaussian
非負整数で最大 ≥ 2  → Poisson
全て {0,1}          → Bernoulli（§5.2 により score は常にこれを選ぶ）
```

という**観測 support の決定論的関数**であり、score は一度も結果を変えない。
起こりうる唯一の誤りは「真 Poisson なのに全観測が {0,1}」→ Bernoulli であり、その確率は §4 の通り。

**INTERPRETATION（量の整合）** §5.2 で導出済みの下界は
`margin_neg2 ≥ (2/L) Σ_s Σ_i g(η_i^(s)) > 0`（η は Poisson 候補の fit 値 `f̂_P^T z_i^(s)`）であり、これだけが厳密な結果である。
参照のため `g(0) = 1 − ln 2 = 0.3069` を使うと `2n g(0)` は n = 75 で 46.0、n = 40 で 24.55 になる。
**これは任意の fit 済み η に対する global lower bound ではなく、η ≈ 0 のときの reference magnitude である**
（下界の値は fit 済み η に依存し、実際に全 artifact の最小 margin 43.6 は 46.0 を下回る）。
「margin は理論上必ず 46 以上」とは読まない。
artifact の margin は n = 75 の全 score 判定行で最小 43.6（全 artifact）、pilot C2 の最終値 45.69–65.50、
n = 40 の smoke で 23.9–24.8 であり、この reference magnitude と**大きさが整合する**。
fit 済みの η を再計算していないので、これは大きさの整合であって検証ではない。

---

## 6. 「selector の能力」と「support gate・構造で決まっている部分」の分離

| 部分 | 何が決めているか | selector の能力の証拠になるか |
|---|---|---|
| Gaussian 列 → Gaussian | gate（support） | ならない（既存 report も除外済み） |
| count（≥2 あり）→ Poisson | gate（support） | ならない（同上） |
| 真 Bernoulli の 0/1 列 → Bernoulli | **§5 の恒等式**（score の形だけで決まる） | **ならない**。真の family によらず必ず Bernoulli になるので、真 Bernoulli で正解したことは判別能力を示さない |
| 真 Poisson の 0/1 列 | §4 で事実上起きない。起きても §5 で必ず Bernoulli | 評価不能（そして評価しても結果は事前に決まっている） |

**INTERPRETATION** 既存 report（`phase9c_c2_research_first_pilot_summary_20260925.md` §5）の
「margin は大きいが、この条件が判別の易しい側にある可能性」という留保は、本監査の結果を踏まえると
**「易しい / 難しい」の問題ではなく、score が構造的に Bernoulli を選ぶ**という形に置き換えるべきである
（過去 report は当時の記録として書き換えない。CLAUDE.md §4）。

**INTERPRETATION（K 選択への含意、既存 artifact に限定）** 本監査で集計した commit 済みの Phase 9 artifact
（9D / 9E / 9I / 9J / 9K / 9P / 9T / 9V / 9X）では、0/1 に見える真 Poisson 列は一度も起きず、
実現した family の割り当ては全行で生成時の family と一致した（score 判定行は全て真 Bernoulli → Bernoulli、gate 判定行は support どおり）。
したがって **報告済みの Phase 9 の K 選択の結果は、それらの artifact では、観測された family 誤割り当てによって交絡していない**。
「oracle family assignment」と呼ぶ場合も、**実現した既存 dataset に限れば**という条件つきである。
割り当て規則（§5.4）は support の決定論的関数だが、真 Poisson の全観測が {0,1} になる事象（§4、1 列あたり ≤ 1.0×10⁻¹⁰）では Bernoulli になるので、
**今後のすべての canonical draw で真の family と一致することの保証ではない**。

---

## 7. まだ言えないこと（UNRESOLVED / 限定）

- §3.2 の `P(X≤1)` の s についての単調性は数値（格子）でのみ確認。一般証明はしていない。
- C_Lap で family を比較した場合の順序（§5.3）は未証明。
- pilot C2 の非収束 candidate（最終 8/72）が score に与えた影響は測っていない（§5.3、既存 report §5 と同じ）。
- 本監査は X 列 intercept の導入や family-selection の目的変更を**提案・決定しない**（HG-5、CLAUDE.md §6 の Human Gate）。
  比較は `x_intercept_research_decision_memo_20261002.md` に分けた。

---

## 付録 A. 数値評価スクリプト（全文、repo 外 scratchpad で実行）

実行環境: `D:\tento\kennkyu\.venv` の Python、numpy 2.3.5、scipy 1.16.3。乱数は使っていない。
引数に worktree の root を与えると、commit 済みの `generator_provenance.csv` を read-only で読む。

```python
"""Deterministic numerical evaluation for the family-selection hard-case audit.

No random numbers, no data generation, no EM.  Gauss-Hermite quadrature over
eta ~ N(mu, s^2) and Gauss-Jacobi (Beta) quadrature over ||f_l||^2.
Reads committed generator_provenance.csv files read-only.
"""
import csv, glob, math, os, sys
import numpy as np
from scipy.special import roots_hermitenorm, roots_jacobi, expit, betaln, gammaln
from scipy.optimize import brentq

GH_X, GH_W = roots_hermitenorm(200)
GH_W = GH_W / GH_W.sum()          # E[g(Z)], Z ~ N(0,1)


def E(g, mu, s):
    return float(np.sum(GH_W * g(mu + s * GH_X)))


def pois_probs(mu, s):
    lam = lambda e: np.exp(e)
    p0 = E(lambda e: np.exp(-lam(e)), mu, s)
    p1 = E(lambda e: lam(e) * np.exp(-lam(e)), mu, s)
    mean = math.exp(mu + s * s / 2)
    return dict(mean=mean, p0=p0, p1=p1, pge2=1 - p0 - p1, ple1=p0 + p1)


def bern_probs(mu, s):
    p1 = E(expit, mu, s)
    return dict(mean=p1, p0=1 - p1, p1=p1, pge2=0.0, ple1=1.0)


def gap(e):
    # g(eta) = -log P_Pois(X<=1 | eta) = e^eta - log(1+e^eta) >= 0
    return np.exp(e) - np.logaddexp(0.0, e)


def fmt(v):
    if v == 0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.3e}"
    return f"{v:.4f}"


out = []
P = out.append
n = 75

P("## T1 no-intercept Poisson vs Bernoulli, eta ~ N(0, s2)")
P("| s2=||f||^2 | Pois E[lam] | Pois P0 | Pois P1 | Pois P>=2 | Pois P<=1 | (P<=1)^75 | log10 | Bern P1 | E[g(eta)] | 2*75*E[g] |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for s2 in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]:
    s = math.sqrt(s2)
    pp = pois_probs(0.0, s); bp = bern_probs(0.0, s)
    q = pp["ple1"] ** n
    eg = E(gap, 0.0, s)
    P(f"| {s2} | {fmt(pp['mean'])} | {fmt(pp['p0'])} | {fmt(pp['p1'])} | {fmt(pp['pge2'])} | {fmt(pp['ple1'])} | {fmt(q)} | {math.log10(q):.2f} | {fmt(bp['p1'])} | {fmt(eg)} | {fmt(2*n*eg)} |")

# max of P(X<=1) over s
grid = np.linspace(0, 6, 6001)
vals = [pois_probs(0.0, s)["ple1"] for s in grid]
i = int(np.argmax(vals))
P(f"\nmax_s P_Pois(X<=1) on s in [0,6]: s={grid[i]:.3f} (s2={grid[i]**2:.3f}), value={vals[i]:.6f}; "
  f"at s=0: {vals[0]:.6f}; at s=6: {vals[-1]:.6f}")
pmax = max(vals)
P(f"upper bound of P(all 75 in {{0,1}} | f) over all f: {pmax**75:.3e} (log10 {75*math.log10(pmax):.2f})")
for tgt in [0.5, 0.05, 0.01]:
    P(f"n such that (max P<=1)^n >= {tgt}: n <= {math.floor(math.log(tgt)/math.log(pmax))};  at s2=0.5: n <= {math.floor(math.log(tgt)/math.log(pois_probs(0,math.sqrt(.5))['ple1']))}")

# marginal over F: ||f||^2 = fs2 * B, B ~ Beta(K/2,(d-K)/2)
P("\n## T2 marginal over generator F (||f||^2 = f_scale^2 * Beta(K/2,(d-K)/2)), n=75")
P("| condition | f_scale^2 | K | d | E||f||^2 | E_F[(P<=1)^75] | log10 | P(>=1 of 3 Pois cols ATP) <= |")
P("|---|---|---|---|---|---|---|---|")


def beta_expect(h, a, b, m=200):
    # Gauss-Jacobi on [0,1]: weight u^(a-1)(1-u)^(b-1)
    x, w = roots_jacobi(m, b - 1, a - 1)       # weight (1-x)^(b-1)(1+x)^(a-1) on [-1,1]
    u = (x + 1) / 2
    w = w / np.sum(w)
    return float(np.sum(w * np.array([h(v) for v in u])))


for name, fs2, K in [("pilot C2 / 9K / 9E / 9P / 9T / 9V-base", 2.0, 3), ("9V weak", 1.0, 3), ("9V strong", 4.0, 3),
                      ("9X K1", 6.0, 1), ("9X K2", 3.0, 2), ("9X K4", 1.5, 4)]:
    d = 12
    q = beta_expect(lambda u: pois_probs(0.0, math.sqrt(fs2 * u))["ple1"] ** n, K / 2, (d - K) / 2)
    P(f"| {name} | {fs2} | {K} | {d} | {fs2*K/d:.3f} | {q:.3e} | {math.log10(q):.2f} | {3*q:.3e} |")

# expected ATP across committed artifacts, using recorded f_row_norm_sq
P("\n## T3 committed artifacts: sum over unique true-Poisson columns of P(all n in {0,1} | recorded ||f_l||^2)")
root = sys.argv[1]
seen = {}
per_dir = []
for p in sorted(glob.glob(os.path.join(root, "expfam/results/**/generator_provenance.csv"), recursive=True)):
    rows = list(csv.DictReader(open(p)))
    if "family_x_true" not in rows[0]:
        continue
    proto = os.path.join(os.path.dirname(p), "protocol.json")
    import json
    nn = json.load(open(proto))["n"]
    s_dir = 0.0; cnt = 0
    for r in rows:
        if r["family_x_true"] != "poisson":
            continue
        key = (r["data_seed"], r["column"], r["f_row_norm_sq"], r["column_mean"])
        prob = pois_probs(0.0, math.sqrt(float(r["f_row_norm_sq"])))["ple1"] ** nn
        s_dir += prob; cnt += 1
        seen[key] = (prob, nn)
    per_dir.append((os.path.relpath(os.path.dirname(p), root).replace("\\", "/"), cnt, s_dir))
for dname, cnt, sd in per_dir:
    P(f"- {dname}: Poisson column instances {cnt}, expected ATP {sd:.3e}")
tot = sum(v[0] for v in seen.values())
mx = max(v[0] for v in seen.values())
P(f"unique (data_seed,column,||f||^2,column_mean) true-Poisson columns: {len(seen)}; expected ATP total {tot:.3e}; max single {mx:.3e}")

# Option B: intercept b
P("\n## T4 with intercept b: Poisson eta ~ N(b, s2), b = log(m) - s2/2 so that E[lam]=m; n=75")
P("| s2 | target mean m | b | P0 | P1 | P>=2 | (P<=1)^75 | E[g] | 2*75*E[g] | Bern b_B matching P(X=1) | Bern P1 |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for s2 in [0.25, 0.5, 1.0]:
    s = math.sqrt(s2)
    for m in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        b = math.log(m) - s2 / 2
        pp = pois_probs(b, s)
        eg = E(gap, b, s)
        target = pp["p1"]
        bB = brentq(lambda c: E(expit, c, s) - target, -40, 40)
        P(f"| {s2} | {m} | {b:.4f} | {fmt(pp['p0'])} | {fmt(pp['p1'])} | {fmt(pp['pge2'])} | {fmt(pp['ple1']**n)} | {fmt(eg)} | {fmt(2*n*eg)} | {bB:.4f} | {fmt(E(expit,bB,s))} |")

# n needed for P(all <=1) >= 0.5 under intercept
P("\n## T5 with intercept: largest n with P(all n in {0,1}) >= 0.5 / 0.9 (marginal over z, f fixed)")
for s2 in [0.25, 0.5, 1.0]:
    s = math.sqrt(s2)
    for m in [0.02, 0.05, 0.1, 0.2]:
        b = math.log(m) - s2 / 2
        pl = pois_probs(b, s)["ple1"]
        P(f"- s2={s2}, m={m}: P(X<=1)={pl:.6f}; n_max(0.5)={math.floor(math.log(.5)/math.log(pl))}, n_max(0.9)={math.floor(math.log(.9)/math.log(pl))}")

print("\n".join(out))
```

（T4 / T5 の結果は `x_intercept_research_decision_memo_20261002.md` と `next_experiment_design_options_20261002.md` で使う。）
