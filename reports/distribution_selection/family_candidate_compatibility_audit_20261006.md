# 候補分布族の数理 compatibility audit（multi-support family selection, 2026-10-06）

- 位置づけ: **THEORY / AUDIT**（実装前の Gate A）。本文書は数式と判定のみを記録し、実験結果を含まない。
- branch: `exp/multisupport-family-selection`（base `origin/main = ee4be23ab9613c5cc454d8cc1af49c1d2d3551bf`）
- 系列: **E（experimental prototype; 本文採用不可）**。生成モデルは変更しない。
- 前提文書: `reports/distribution_selection/automatic_family_selection_design_20260923.md`（Phase 9B 設計）、
  `reports/distribution_selection/phase9c_c2_research_first_pilot_summary_20260925.md`（Phase 9C pilot）。
- 機械可読版: 実行 run の `tables/family_compatibility.csv`（`family_specs.compatibility_rows()` から生成。手編集しない）。

---

## 0. 守るモデル

```
z_i ~ N(0, I_K)
η_il^X = f_l^T z_i                       （intercept なし・link 追加なし）
x_il | z_i, c_l = m ~ ExpFam_m(η_il^X)
log p_m(x | η) = log h_m(x) + η T_m(x) − A_m(η)     （canonical 形, 分散パラメータ φ = 1）
Y: 既存 Bernoulli 関係モデル
```

Gaussian のみ既存実装どおり dispersion 形 `log p = −(x−η)²/(2σ_l²) − ½log σ_l² − ½log 2π`（σ_l² は M-step で推定）を保持する。
新 family はすべて **φ = 1 の canonical 形**で、`η = f^T z` を natural parameter としてそのまま用いる。
family 固有の形状・nuisance（Binomial の試行数、COM-Poisson の ν、各 Normal 系の σ 等）は
**protocol で事前固定する定数**であり、推定しない。

一般化の内容（`PASS_WITH_EXISTING_EXPFAM_GENERALIZATION` の意味）:
既存 X 側コードは `T(x) = x`、`log h(x) ∈ {0, −log x!}` を暗黙に仮定している。
E-step 勾配 `F^T[T(x_i) − A'(Fz_i)]`、曲率 `F^T diag(A''(Fz_i)) F`、M-step 勾配
`Σ_i [T(x_il) − A'(η_il)] z_i` は指数型分布族の一般式であり、`T`・`A'`・`A''` を family ごとに差し替えるだけで
**現行の Laplace E-step / M-step の形を変えずに**扱える。比較 score は `log h` を含む完全な log 確率（密度）を使う。

---

## 1. STRICT 条件（18 項目）

| # | 条件 |
|---|---|
| 1 | scalar observation column |
| 2 | fixed support |
| 3 | regular exponential family として表現可能 |
| 4 | scalar natural parameter |
| 5 | `η = f^T z` を natural parameter として直接使える |
| 6 | natural parameter domain = ℝ（unconstrained linear predictor で常に合法） |
| 7 | `T(x)` を明確に定義できる |
| 8 | `log h(x)` を明確に定義できる |
| 9 | `A(η)` を定義できる |
| 10 | `A'(η)` を定義できる |
| 11 | `A''(η)` を定義できる |
| 12 | 正規化された log 確率 / 密度を計算できる |
| 13 | generator を実装できる |
| 14 | gradient を正しく導出できる |
| 15 | curvature を正しく導出できる |
| 16 | 現行 Laplace E-step と整合する（`A'' > 0` により z_i の条件付き対数事後が strictly concave） |
| 17 | candidate score を full constant 込みで比較できる |
| 18 | support-compatible candidate と同じ base measure 上で比較できる |

---

## 2. 判定結果（PASS）

`Φ` = 標準正規 CDF、`φ` = 標準正規密度、`λ(u) = φ(u)/Φ(u)`（inverse Mills ratio）、`σ(·)` = logistic。

### 2.1 COUNT（counting measure on ℕ = {0,1,2,…}）

| family | candidate_model | support | T(x) | log h(x) | A(η) | A'(η) | A''(η) | 判定 |
|---|---|---|---|---|---|---|---|---|
| poisson | `poisson` | ℕ | x | −log x! | e^η | e^η | e^η | **PASS_STRICT**（既存） |
| binomial | `binomial_n2`（n_trials=2） | {0,…,N} | x | log C(N,x) | N log(1+e^η) | N σ(η) | N σ(η)(1−σ(η)) | **PASS_WITH_EXISTING_EXPFAM_GENERALIZATION** |
| com_poisson | `cmp_nu0p5`, `cmp_nu2p0`（ν 固定） | ℕ | x | −ν log x! | log Σ_{x≥0} e^{ηx}/(x!)^ν | E[x] | Var[x] | **PASS_WITH_EXISTING_EXPFAM_GENERALIZATION** |
| zero_truncated_poisson | `ztp` | {1,2,…} | x | −log x! | e^η + log(1 − e^{−e^η}) | λ/(1−e^{−λ}), λ=e^η | μ(1+λ−μ), μ=A' | **PASS_WITH_EXISTING_EXPFAM_GENERALIZATION** |

理由:

- **Binomial**: N 固定で 1 パラメータの regular EF。domain ℝ。`log C(N,x)` を score から落とせない（Poisson の `−log x!` と異なるため）。
  N=1 が既存 Bernoulli。N=2 は「η=0 での平均 N/2 = 1 が Poisson の e^0 = 1 と一致する」最小の試行数であり、
  intercept のない現行モデルで **位置（平均水準）だけで勝負が決まらない**ように事前に選んだ（設計書 §5 の位置交絡の回避。selector 結果を見ていない）。
  support が {0,…,N} に限られるので、`max x > N` の列では Binomial は**尤度 0（log = −∞）で inadmissible**。これは support 判定として記録し、score 比較から外す。
- **COM-Poisson**: ν > 0 固定なら `(x!)^ν` が超指数的に増えるので級数は全 η ∈ ℝ で収束し、domain ℝ。
  ν = 1 で Poisson に厳密一致（unit test で確認）。ν = 0 は Geometric（domain η<0）になるので**ν > 0 に限る**。
  `A, A', A''` は adaptive に打ち切った級数（log-sum-exp）で計算し、打ち切り誤差の上界（幾何級数 tail bound）を監査する。
  ν = 0.5（過分散側）と ν = 2.0（過小分散側）を **candidate model** として固定し、ν の推定は行わない。
  **COM-Poisson の ν 違いは別 family として数えない**（family = `com_poisson`、candidate_model = `cmp_nu0p5` / `cmp_nu2p0`）。
- **Zero-truncated Poisson**: `p(x) = e^{−λ}λ^x / (x!(1−e^{−λ}))`, x ≥ 1。`A(η) = log(e^{e^η} − 1)` は全 η で有限、domain ℝ。
  truncation 正規化項 `log(1−e^{−λ})` を落とさない。λ → 0 で `−expm1(−λ)/λ` と `1+λ−μ` は桁落ちするので級数で評価する。
  support {1,2,…} なので 0 を含む列では inadmissible。

### 2.2 POSITIVE CONTINUOUS（Lebesgue measure on (0,∞)）

| family | candidate_model | T(x) | log h(x) | A(η) | A'(η) | A''(η) | 判定 |
|---|---|---|---|---|---|---|---|
| lognormal | `lognormal_s1`（σ=1） | log x | −log x − log(σ√2π) − (log x)²/(2σ²) | σ²η²/2 | σ²η | σ² | **PASS_WITH_EXISTING_EXPFAM_GENERALIZATION** |
| positive_truncated_normal | `ptn_s1`（σ=1） | x | −x²/(2σ²) − log(σ√2π) | σ²η²/2 + log Φ(ση) | σ²η + σλ(ση) | σ²[1 − λ(u)(u+λ(u))], u=ση | **PASS_WITH_EXISTING_EXPFAM_GENERALIZATION** |

理由:

- **Log-normal（σ 固定）**: `log x ~ N(σ²η, σ²)`。σ 固定なら natural parameter は scalar で domain ℝ。
  **raw x の密度として評価し、Jacobian `1/x`（`−log x`）を log h に含める**。log 変換後の Gaussian score とは比較しない。
  σ も推定すると natural parameter が 2 次元になり scalar 構造の外に出るので、σ は固定する。
- **Positive truncated Normal（σ 固定）**: `N(μ, σ²)` を (0,∞) に切断、`μ = σ²η`。
  `A(η)` の正規化項 `log Φ(ση)` を落とさない。`log Φ` は `scipy.special.log_ndtr` で安定に計算する。

### 2.3 BOUNDED CONTINUOUS（Lebesgue measure on (0,1)）

| family | candidate_model | T(x) | log h(x) | A(η) | A'(η) | A''(η) | 判定 |
|---|---|---|---|---|---|---|---|
| continuous_bernoulli | `cb` | x | 0 | log((e^η − 1)/η)、A(0)=0 | 1/(1−e^{−η}) − 1/η、A'(0)=1/2 | 1/η² − 1/(4 sinh²(η/2))、A''(0)=1/12 | **PASS_WITH_EXISTING_EXPFAM_GENERALIZATION** |
| logit_normal | `logitnormal_s1`（σ=1） | logit x | −log x − log(1−x) − log(σ√2π) − logit(x)²/(2σ²) | σ²η²/2 | σ²η | σ² | **PASS_WITH_EXISTING_EXPFAM_GENERALIZATION** |
| truncated_normal_01 | `tn01_s0p25_c0p5`（σ=0.25, c=0.5） | x | −(x−c)²/(2σ²) − log(σ√2π) | cη + σ²η²/2 + log[Φ(β) − Φ(α)] | μ + σ[φ(α) − φ(β)]/Z | σ²[1 + (αφ(α) − βφ(β))/Z − ((φ(α)−φ(β))/Z)²] | **PASS_WITH_EXISTING_EXPFAM_GENERALIZATION** |

（TN01: `μ = c + σ²η`, `α = (0−μ)/σ`, `β = (1−μ)/σ`, `Z = Φ(β) − Φ(α)`。）

理由:

- **Continuous Bernoulli**: `p(x) = e^{ηx}/∫_0^1 e^{ηt}dt` on (0,1)（= (0,1) に切断した指数分布で、任意の η ∈ ℝ で正規化可能）。
  `A, A', A''` は η → 0 で見かけ上 0/0 になるので `|η| < 10^{-2}` で級数（A: `η/2 + t²/6 − t⁴/180 + t⁶/2835`, t=η/2）、
  それ以外は `log sinh` の安定形で評価する。η = 0 で Uniform(0,1) に一致（unit test）。
- **Logit-normal（σ 固定）**: `logit x ~ N(σ²η, σ²)`。**Jacobian `1/[x(1−x)]` を含める**。端点 0, 1 は support 外（generator は端点に丸まった値が出たら停止する）。
- **Truncated Normal on (0,1)（σ, c 固定）**: base measure `h(x) ∝ exp(−(x−c)²/(2σ²))` を持つ 1 パラメータ EF。
  c = 0.5 は **support の中点**であり、他の 2 候補の base measure（CB: h = 1、logit-normal: h は x ↔ 1−x で対称）と同じく
  中点対称な基底測度にするための固定ハイパーパラメータである。推定しない・列ごとに変えない（intercept ではない）。
  c = 0 を採ると η = 0 で密度が 0 側に偏り、他候補との違いが「位置」だけで決まる（設計書 §5）ので採らない。
  σ = 0.25 は η = 0 で Uniform（CB の η=0）と視覚的に区別できる広がり（sd ≈ 0.22 vs 0.289）で、±2σ が (0,1) 内に収まる値として、
  **selector 結果を見る前に**固定した。σ = 1 では `h` の (0,1) 上の変動が e^{1/8} ≈ 1.13 倍しかなく CB とほぼ識別不能になる（これは記録のみ。実験しない）。
  `log[Φ(β) − Φ(α)]` は `log_ndtr` 差の安定形（`log(−expm1(·))`）で計算し、CDF 差の桁落ちを避ける。

### 2.4 REAL（Lebesgue measure on ℝ）

| family | candidate_model | 判定 | 理由 |
|---|---|---|---|
| gaussian | `gaussian`（σ_l² 推定） | **PASS_STRICT**（既存・control） | 既存 dispersion 形をそのまま使う。ℝ 上で数理的に自然な競合候補（scalar natural parameter・domain ℝ）が見つからなかった（Student-t / Laplace / logistic は location に関して EF でない）ので**無理に増やさない**。REAL 群は gate だけで決まる control として Experiment C にのみ入れる |

### 2.5 既存 Bernoulli

`bernoulli` は PASS_STRICT（既存）。ただし Binomial(N=1) と同一であり、Phase 9C で Bernoulli vs Poisson は既に調べたので、
**今回の primary candidate set には入れない**（family と candidate model の数え方を混乱させないため）。コードと regression test は維持する。

---

## 3. 判定結果（BLOCKED_OUTSIDE_CURRENT_MODEL）

| family | 理由（どの条件で落ちるか） |
|---|---|
| geometric | 条件 6。`p(x) = (1−p)p^x`、natural parameter `η = log p ∈ (−∞, 0)`。`η = f^T z` は正にもなるので常に合法ではない。合法化には `η = −softplus(f^T z)` 等の link 変更が要り、今回禁止されている |
| negative_binomial（r 固定） | 条件 6。natural parameter `log(μ/(μ+r)) ∈ (−∞,0)`。既存 `model_dual_expfam_nb.py` は Y 側 NB2 + **log link（非 canonical）**であり、X 側へのコピーは禁止されている（canonical にすると domain 制約、log link にすると family-specific link の追加になる） |
| gamma（shape 固定） | 条件 6。natural parameter `−rate ∈ (−∞, 0)` |
| exponential | 条件 6。natural parameter `−λ ∈ (−∞, 0)`。**注:** (0,1) に切断した指数分布は Continuous Bernoulli そのもので、そちらは domain ℝ なので PASS（§2.3） |
| beta | 条件 4・6。natural parameter が `(a−1, b−1)` の 2 次元。片方を固定しても domain が `(−1, ∞)` に制約される |
| weibull（shape k 固定） | 条件 6。`T = x^k`、natural parameter `−λ^{−k} < 0`。shape も推定すると EF でない |
| student_t | 条件 3。location に関して指数型分布族ではない（自由度固定でも） |
| categorical | 条件 1・4。scalar observation の scalar natural parameter ではない（ベクトル natural parameter）。KI-005 |
| inverse_gaussian（追加監査） | 条件 6。2 つの natural parameter がともに負の領域に制約される |

BLOCKED family は**実装しない**。今回の結果から BLOCKED family について何も主張しない。

---

## 4. 比較可能性（Level 0 / Level 1）

**Level 0（support / base measure gate、尤度を使わない決定的判定）:**

| 観測列 | support group | base measure |
|---|---|---|
| 全て非負整数 | `GROUP_COUNT` | counting measure on ℕ |
| 非整数を含み、全て 0 < x < 1 | `GROUP_BOUNDED_CONTINUOUS` | Lebesgue on (0,1) |
| 非整数を含み、全て x > 0、1 以上を含む | `GROUP_POSITIVE_CONTINUOUS` | Lebesgue on (0,∞) |
| それ以外（負値・0 を含む実数） | `GROUP_REAL` | Lebesgue on ℝ |

**Level 1（同一 group 内の score 比較）:** group の候補のうち、列の観測を support に含むもの（admissible）だけを
同一 posterior sample（MCEM）または同一 true-Z surrogate（Experiment A）の上で、
`log h` を含む完全な log 確率 / 密度の L 平均和で比較する。admissible が 1 つだけの列は `decided_by = gate`、
2 つ以上なら `decided_by = score` と記録し、**両者を混同しない**。

**禁止（変更なし）:** 異なる base measure 間（PMF vs density、(0,1) density vs ℝ density の group をまたぐ比較）の raw log-score 比較。
group をまたぐ「データ型の選択」は representation 選択であり、尤度では決められない（設計書 §3.3, U1）。

**注意（記録）:** (0,1) ⊂ (0,∞) なので、(0,1) 内の観測は数学的には POSITIVE 群の密度でも評価できる。
今回の protocol は group を Level 0 で一意に決めるので、そのような group 間比較は行わない。

---

## 5. 自由パラメータ数と penalty

全候補で `f_l ∈ ℝ^K`（K 個）。新 family の nuisance はすべて固定なので **列あたりの追加自由パラメータは 0**。
Gaussian のみ σ_l² で +1 だが、Gaussian は REAL 群の単独候補で比較に入らない。
したがって **score 比較に入る候補どうしの自由パラメータ数は等しく**、raw candidate score の比較は
（固定 posterior sample の下で）同次元の最大尤度比較である。新しい penalty は導入しない（Phase 9B HG-3 (iii) の踏襲）。

---

## 6. この audit から言えないこと

- 「すべての指数型分布族に対応した」とは言えない（BLOCKED 9 family、§3）。
- nuisance（N, ν, σ, c）は固定であり、「nuisance まで自動選択できる」とは言えない。
- 本文書は数理上の互換性判定であり、selector の識別力については何も言わない（それは実験の問い）。
