# canonical clean generator 仕様

**作成日:** 2026-09-04
**baseline main:** `7e335602999977060208ce37ac8cdff8fedfa66e`
**位置づけ:** forward-only。**historical generator (`expfam/src/data_generator_expfam.py`) は変更しない。**
**前提:** `reports/identifiability/true_k_identifiability_hardened_20260904.md`（Phase 1）
および `reports/identifiability/true_k_identifiability_review_20260904.md`（Phase 2）

---

## 0. 目的

> **canonical model family の内部にある既知の `K` から、literal に well-specified な
> synthetic data を生成する。**

Phase 1 §13 で確認したとおり、historical generator は canonical model の literal generator
ではない（`Z` と Gaussian-X の事後 z-score、`F` の行正規化、Poisson の hard clip、
未使用の `sigma_x_true`）。本仕様はそれらを持たない**別モジュール**を定義する。

**非目的:** historical 結果の再現・否定・置換。historical artifact は一切変更しない。

---

## 1. 生成モデル（literal）

```
z_i           ~ N(0, I_K)                     i = 1..n,  iid
x_il | z_i    ~ ExpFam_X( eta^X_il = f_l^T z_i )      l = 1..d,  bias なし
y_ij | z_i,z_j ~ ExpFam_Y( eta^Y_ij = w0 + w z_i^T z_j )   i < j のみ
```

- `F ∈ R^{d×K}`、`w0, w ∈ R` は**スカラー**。
- `x_il` は `z_i` の下で `l` について条件付き独立。
- `y_ij` は `(z_i, z_j)` の下で pair について条件付き独立。
- `Y_ii` は**観測モデルの外**。保存時のみ 0 を置く。

---

## 2. 仕様項目

### 2.1 潜在変数 `Z`

```
Z = rng.standard_normal((n, K))
```

- **生成後の正規化を一切行わない。** column z-score も row 正規化もしない。
- 根拠: Phase 1 §9.1 の `M_S(t) = (1−t²)^{−K/2}` および §7 の Poisson-X モーメント公式は
  いずれも `z ~ N(0, I_K)` を厳密に使う。z-score すると成立しない（Phase 1 G1）。

### 2.2 因子負荷 `F` — full-rank を construction で保証

`rank(F) = K` を **seed rescue なしで** 保証する。手続き:

```
1. G = rng.standard_normal((d, K))
2. Q, R = qr(G, mode="reduced")            # Q: (d,K) 直交列
3. Q = Q * sign(diag(R))                   # 符号を決定論的に固定
4. F = Q @ diag(singular_values)           # singular_values は仕様で固定
5. rank(F) == K を検証。満たさなければ FAIL FAST（再抽選しない）
```

- `singular_values` は既定で全て `f_scale`（等方）。`rank(F) = K` は
  `f_scale > 0` かつ QR が退化しない限り厳密に成立する。
- **失敗時に seed を変えて引き直すことは禁止**（seed rescue）。`HarnessStop` 相当で停止する。
- `d ≥ K` は事前条件。満たさなければ FAIL FAST（Phase 1 §5: `d < K` では X 側から
  `K` は識別できない）。

**`f_scale` の意味:** `Q` の列は正規直交なので `F^T F = f_scale² I_K`。
行ノルムは `||f_l|| = f_scale · ||q_l||` で、`Σ_l ||q_l||² = K` より
平均 `||f_l||² = f_scale² K / d`。

Poisson-X では Phase 1 (7.1) より `E[X_l] = exp(||f_l||²/2)` なので、
`f_scale` は期待カウントを直接支配する。**metadata に実現行ノルムと
`max_l E[X_l]` を必ず記録する。**

### 2.3 Gaussian-X

```
X = Z @ F.T + N(0, Sigma_X)
```

- **生成後 z-score しない**（Phase 1 G3）。
- `Sigma_X` は**対角共分散**。API は**分散ベクトル** `sigma_x_var`（長さ `d` または scalar）
  を取る。**標準偏差ではない。**
- 返り値には `Sigma_X`（`d×d` 対角行列、分散）を入れる。
- 検証: `sigma_x_var > 0` を要求。0 以下は FAIL FAST。

**根拠（semantics 固定）:** 推論側 `DualExpFamLSMConsistent.calc_log_likelihood_X` は
`sigma_diag = np.diag(self.params["sigma"])` を `-0.5 * resid**2 / sigma_diag` に使う
（`[CONFIRMED_IN_REPOSITORY]`）。すなわち `params["sigma"]` の対角は**分散**である。
generator の `Sigma_X` はこれと同じ意味に固定する。

### 2.4 Gaussian-Y

```
Y_ij = w0 + w * (z_i . z_j) + N(0, sigma_y_sd**2)
```

- API は**標準偏差** `sigma_y_sd` を取る。分散は `sigma_y_sd ** 2`。
- 根拠: root `CLAUDE.md` §1 の「実装は標準偏差 `σ_Y` を `self.sigma_y` に保持し、
  使用時に二乗する」に一致させる。
- **X 側は分散、Y 側は標準偏差**という非対称は既存規約に合わせた意図的な選択であり、
  **引数名に semantics を埋め込む**ことで取り違えを防ぐ（`sigma_x_var` / `sigma_y_sd`）。
- 検証: `sigma_y_sd > 0`。

### 2.5 Bernoulli（X 側・Y 側共通）

```
p = sigmoid(eta)        canonical link
```

- **predictor への hard clipping を statistical model として使わない。**
- 数値的に安定な評価のみ許可する。実装は
  `expfam/src/experimental/objective_consistent_numerics.py` の `bernoulli_mean`
  と同じ分岐方式（`eta >= 0` と `eta < 0` で式を切り替える）を使う。
  これは**数学的に恒等**な変形であってモデルを変えない。
- 非有限 `eta` は FAIL FAST。

### 2.6 Poisson（X 側・Y 側共通）

```
lambda = exp(eta)       canonical link
```

- **silent hard clipping 禁止**（Phase 1 G4）。
- `eta > log(finfo(float64).max)` なら **FAIL FAST**（overflow 域）。
- さらに実務的な安全弁として `lambda` の上限 `poisson_lambda_max`（既定 `1e6`）を
  **gate として**持つ。**超えたら clip せず停止する。**
- 実装方針は `objective_consistent_numerics.poisson_mean` と同じ（overflow を
  例外にする、clip しない）。

### 2.7 Poisson-Y のモーメント存在 gate

Phase 1 §11（P6）より canonical Poisson-Y では

```
E[Y^r] < infinity   <=>   |w| < 1/r
```

したがって

| gate | 条件 | 既定 |
|---|---|---|
| `require_finite_mean` | `|w| < 1` | 常に要求 |
| `require_finite_variance` | `|w| < 1/2` | **既定で要求** |

- `family_y == "poisson"` かつ `|w| >= 1/2` の場合、既定で **FAIL FAST**。
- 明示的に `allow_infinite_variance=True` を渡した場合のみ通す。その場合
  metadata に `moment_existence_warning` を記録する。
- 根拠: historical default `w = 0.5` は分散発散の境界そのもの（Phase 1 O1）。
  同じ既定値を無自覚に使わせない。

### 2.8 `Y` の生成と保存

```
1. i < j の upper triangle のみ n(n-1)/2 個をサンプル
2. Y[i,j] = value; Y = Y + Y.T
3. np.fill_diagonal(Y, 0.0)
```

- **サンプルは upper triangle で 1 回だけ**（対称化は保存の都合）。
- `Y_ii` は観測モデル外。0 は「観測なし」の placeholder であって観測値ではない。
  metadata の `diagonal_policy` に明記する。

### 2.9 決定論的 RNG

- `numpy.random.default_rng(seed)` を 1 本だけ使う。
- 消費順序を仕様として固定する: **`Z` → `F` → `X` → `Y`**。
- 同一 `seed` + 同一設定 → **bit-exact に同一出力**。

### 2.10 生成後 validation（すべて FAIL FAST）

| ID | 検証内容 |
|---|---|
| V1 | `Z.shape == (n, K)`、全要素 finite |
| V2 | `F.shape == (d, K)`、全要素 finite、`rank(F) == K` |
| V3 | `X.shape == (n, d)`、全要素 finite |
| V4 | `Y.shape == (n, n)`、全要素 finite、`Y == Y.T`、`diag(Y) == 0` |
| V5 | family support: Bernoulli は `{0,1}`、Poisson は非負整数、Gaussian は実数 |
| V6 | `sigma_x_var > 0`（Gaussian-X 時）、`sigma_y_sd > 0`（Gaussian-Y 時） |
| V7 | `d >= K` |
| V8 | Poisson 使用時、`lambda` が gate 内 |
| V9 | Poisson-Y 使用時、モーメント存在 gate（§2.7） |
| V10 | metadata が §2.11 の全キーを持つ |

**Z の正規化がされていないことの検証:** `Z` の列標本平均が厳密に 0、標本 SD が厳密に 1
に**なっていない**ことを test で確認する（§Phase 5 テスト C）。

### 2.11 metadata（provenance）

生成結果に必ず含める:

```
generator_version          "canonical-clean-v1"
family_x, family_y
n, d, K_true
F_rank                     実測 rank(F)
f_scale
f_row_norms_sq             ||f_l||^2 の配列
Sigma_X                    対角共分散（分散）。Gaussian-X 以外は None
sigma_x_var                入力値
sigma_y_sd                 Gaussian-Y のみ。それ以外は None
w0, w
seed
rng_consumption_order      ("Z", "F", "X", "Y")
link_policy                "canonical_no_clipping_fail_fast"
normalization_policy       "none"
poisson_lambda_max         gate 値
moment_existence           {"mean_finite": bool, "variance_finite": bool}
expected_x_mean            Poisson-X のとき exp(||f_l||^2/2)（理論値）
```

### 2.12 historical generator との関係

- `expfam/src/data_generator_expfam.py` は**読み取り専用**。1 文字も変更しない。
- 新モジュールは既存 production experiment に**接続しない**。
- 名前空間を分けるため `expfam/src/experimental/` 配下に置く
  （既存の forward-only prototype と同じ扱い）。

---

## 3. 本仕様が Phase 1 のどの結果を実装に落とすか

| Phase 1 の結果 | 本仕様での反映 |
|---|---|
| G1（`Z` の z-score） | §2.1 正規化禁止 |
| G2（`F` の行正規化） | §2.2 `F` は自由パラメータ。行ノルムを固定しない |
| G3（Gaussian-X の z-score） | §2.3 正規化禁止 |
| G4（Poisson hard clip） | §2.6 clip 禁止・FAIL FAST |
| G5（`sigma_x_true` 未使用） | §2.3 `sigma_x_var` を実際に使い、metadata に記録 |
| G7（`sigma_y` = SD） | §2.4 `sigma_y_sd` として名前に semantics を埋め込む |
| P1（Poisson-X 識別性）の前提 | §2.2 `rank(F)=K` を construction で保証、§2.6 unclipped link、§2.10 V7 `d≥K` |
| P6 / O1（Poisson-Y モーメント） | §2.7 gate |
| §2.3（`K*` と `K_TRUE` の区別） | §2.2 + V2 + V7 で `K* = K_TRUE` の**必要条件**を強制。十分性は family 依存で `[UNRESOLVED]`（Phase 1 U9） |

**重要な限定:** 本仕様は `K* = K_TRUE` を**保証しない**。保証できるのは
「Phase 1 で識別可能性が証明された条件（P1 の前提）を満たす」ことまでである。
Y 側が Bernoulli の場合、Phase 1 U2 により Y からの識別性は未解決のままである。

---

## 4. 独立レビュー観点（実装前に確認すること）

| # | 観点 | 判定 |
|---|---|---|
| R1 | `Z` に正規化が残っていないか | 実装後 test C で確認 |
| R2 | `F` の rank 保証が seed rescue になっていないか | §2.2 は失敗時に停止する設計 |
| R3 | Poisson に silent clip が残っていないか | test I で確認 |
| R4 | `sigma_x_var` / `sigma_y_sd` の semantics が推論側と一致するか | §2.3 / §2.4 の根拠に一次コード確認あり |
| R5 | Y の対角が観測として扱われていないか | §2.8 + test L |
| R6 | RNG 消費順序が固定されているか | §2.9 + test A |
| R7 | historical generator に触れていないか | git diff で確認 |
