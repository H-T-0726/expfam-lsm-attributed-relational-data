# Gaussian-Y における潜在次元 K の identifiability — 敵対的（反証優先）最終監査

**作成日:** 2026-09-07
**種別:** 理論監査（READ-ONLY・コード変更なし・実験なし・simulation なし・generator 再実行なし）
**branch:** `docs/notion-k-selection-story-20260906`
**姿勢:** falsification-first。定理を成立させるためではなく **壊すために** 監査した。
既存 P2 / P3（`true_k_identifiability_hardened_20260904.md` §9・§12）は **正しいと前提していない**。
すべての分布論をゼロから独立に導出し、そのうえで既存命題と突き合わせた。

---

## Evidence labels

| ラベル | 意味 |
|---|---|
| `[CODE]` | 一次コードを直接読んで確認 |
| `[DERIVED]` | 本監査で独立に導出（証明を本文に置く） |
| `[COUNTEREXAMPLE]` | 実際に構成した反例 |
| `[ASSUMPTION]` | 仮定。導出ではない |
| `[UNRESOLVED]` | 未解決 |

---

## 1. Executive verdict

### **VERDICT: PROVED AS STATED（ただし定理は tight であり、仮定は 1 つも落とせない）**

明示された canonical Gaussian-Y model のもとで、**反例は 1 つも見つからなかった**。
7 系統の攻撃（Attack A–G）はすべて失敗し、さらに **2 本の独立な証明**を得た。

- **証明 1（cumulant）:** `kappa_4(Y) = 6Kw⁴`, `kappa_6(Y) = 120Kw⁶` から `w² = kappa_6/(20 kappa_4)`、
  `K = (200/3) kappa_4³/kappa_6²`。`[DERIVED]`（§6）
- **証明 2（characteristic function、moment 不要・より強い）:**
  `phi_Y(t) = e^{i w0 t}(1 + w²t²)^{−K/2} e^{−σ²t²/2}` の `t → ∞` 漸近から
  `σ²` → `K` → `w²` → `w0` の順に一意に決まる。`[DERIVED]`（§7）

**さらに本監査で新たに得た結果が 3 つある。**

1. **一般公式** `kappa_{2m}(S_K) = (2m−1)! · K`（すべての `m ≥ 1`）。`[DERIVED]`
   既存 P2 の `kappa_2 = K, kappa_4 = 6K, kappa_6 = 120K` はその `m = 1,2,3` である。
2. **アフィン不変な反転公式** `K = (200/3) kappa_4³ / kappa_6²`。
   `Y → aY + b`（`a ≠ 0`）で不変なので、**Y の決定論的な再スケール・平行移動では `K` の識別性は壊れない**。`[DERIVED]`
3. **dyad marginal 族は weak topology で閉じている。** その帰結として
   `P0 ∉ closure(M_{K'})` かつ `inf_{Q ∈ M_{K'}} KL(P0‖Q) > 0`（`K' ≠ K`, `w ≠ 0`）。`[DERIVED]`
   これは既存 U11（`M_K` の閉性未確立）を **この場合に限って解消する**（§10.4）。

**しかし定理は極めて薄い余裕しか持たない。** 決定的な代数的事実は

```
S_K = S_{K'} ⊕ S_{K−K'}        （独立和、K' < K）
```

であり、これは **`K` 次元モデルが「`K'` 次元モデル + 独立ノイズ」に厳密に等しい**ことを意味する。
`M_{K'}` がこれを吸収できない唯一の理由は **ノイズが Gaussian に固定されているから**である。
したがって:

- **ノイズ族を「任意の独立ノイズ」に緩めた瞬間、定理は厳密な反例で崩壊する**（Attack F、§9.6、`[COUNTEREXAMPLE]`）。
- **スカラー `w` を行列 `W` に置き換えても崩壊する**（`W = diag(w, 0)`、§9.5、`[COUNTEREXAMPLE]`）。

**historical fixed Scenario C への適用は禁止**（§15）。fixed generator は `Z` を列 sample z-score するので
`z_i ~ iid N(0, I_K)` が成立せず、本定理の前提が満たされない。
`fixed 30/30` の true-K proof として本定理を引用してはならない。

---

## 2. Exact theorem under audit

### 2.1 監査したモデル `[ASSUMPTION]`

```
K ∈ N_+  （§3.6 で K = 0 の扱いを別途規定）
z_i, z_j  iid ~ N(0, I_K)                       ← latent prior、固定（自由 scale なし）
eps_ij    ~ N(0, sigma_Y²),  eps ⊥ Z            ← ノイズ族は Gaussian に固定
Y_ij = w0 + w · z_i^T z_j + eps_ij      (i < j)
w0 ∈ R,  w ∈ R,  sigma_Y² ≥ 0
```

`M_K^{dyad}` := 上式で `(w0, w, sigma_Y²)` を動かしたときの `Y_ij` の周辺分布の族。
`M_K` := 対応する joint `(X, Y)` の分布族。

### 2.2 監査対象の主張

> **Theorem candidate.** true model で `w ≠ 0` なら、単一 dyad `Y_ij` の population marginal から
> `K` が一意に識別可能である。すなわち `K1 ≠ K2` のとき、`w1 ≠ 0` である限り
> `Law(w01 + w1 z^T z' + eps) = Law(w02 + w2 u^T u' + eps')` は成立しない。
> 従って joint `P(X,Y)` が等しければ Y marginal も等しいので、`K` は joint でも識別可能である。

**注意:** 仮定は非対称である。**`w ≠ 0` は true 側にのみ課され、競合側 `w'` には課されない**
（`w' = 0` も許す）。この非対称性が正しく効くかを §9.3 で個別に検証した。

---

## 3. Canonical implementation / model correspondence `[CODE]`

依頼 §2 の 8 項目を一次コードで確認した。

### 3.1 latent prior は `z_i ~ iid N(0, I_K)` に固定されているか

**YES。** `reproduction/src/model.py:98-99`:

```python
# varZ is fixed to 1.0 for identifiability
var_z = 1.0
```

`params["var_z"] = var_z`（`:128`）として保存され、以後 **どこにも M-step が存在しない**。
repository 全体を検索しても `calc_var_z` は無く、`var_z` は
`utils_expfam.py:176, 478` / `em_runner.py:195` 等で **読み出されるだけ**である。
`reproduction/src/model.py:818` には `assert params['var_z'] == 1.0` という不変条件テストもある。

### 3.2 `tau²` のような自由 scale parameter はあるか

**ない。** §3.1 の通り `var_z` は定数 1.0。
（**もし自由だった場合の影響は §11.C で検討した。結論: `K` は生き残るが `|w|` が識別できなくなる。**
コード側のコメント「fixed to 1.0 for identifiability」はこの事実と整合している。）

### 3.3 Gaussian-Y の平均と分散

**mean = `w0 + w z_i^T z_j`、variance = `sigma_y²`。**
`expfam/src/model_expfam.py:58-59`（`_mean_function` の gaussian 分岐は identity link）、
`:262-264`（log-likelihood が `−0.5(Y−eta)²/sig2 − 0.5 log sig2`）。
`sigma_y` は **標準偏差**として保持され、使用時に二乗される（root `CLAUDE.md` §1 の規約どおり）。

### 3.4 `w` は全 latent direction 共通のスカラーか

**YES。** `model_expfam.py:180-210` の `calc_w` はスカラー `w` に対する Adam 更新であり、
勾配は `np.sum(diff * ZZT)` という 1 個のスカラーである。
**行列 `W`・方向別 `w_r`・anisotropic scale は canonical model に存在しない。**
（**もし許した場合の影響は §11.E。結論: 反例が生じ、識別されるのは `rank(W)` だけになる。**）

### 3.5 `sigma_Y²` の parameter space

| 水準 | 値域 |
|---|---|
| **理論上** | `sigma_Y² ≥ 0`（`0` を含む。§9.4 で `= 0` の場合も定理が成立することを確認した） |
| **実装（M-step）** | `model_expfam.py:234` `sigma_y = sqrt(max(sigma_sq, 1e-6))` → `sigma_Y² ≥ 1e-6` |
| **実装（尤度評価）** | `:263` `sig2 = max(self.sigma_y**2, 1e-8)`、`:146` `_phi()` も同じ floor |

**`1e-6` / `1e-8` は numerical floor であって theory boundary ではない。** 両者を混同しない。
本定理は理論上の `sigma_Y² ≥ 0` 全域で成立する（§9.4）。

### 3.6 候補 `K` の範囲と `K = 0`

runner の候補は `K = 1,…,9`（`run_fixed_official_exp1_bic_full.py:49` の `[1..6]` +
`run_fixed_official_exp1_bic_k9_extension.py:61` の `[7,8,9]`）。`K ≥ 1` のみ。
`K = 0` は実装上構成されない（`self.k = k` に下限チェックは無いが、実行される候補集合に含まれない）。

**理論上の扱い:** 本監査は `K = 0` を `S_0 ≡ 0`（空和）として **候補集合に含めても定理が壊れないこと**を
確認した（Attack F、§9.6.1）。`K = 0` は `w = 0` と観測上同値であり、
`w ≠ 0` の true model とは `kappa_4` で分離される。

### 3.7 X 側 parameter `F` / `family_x` は Y marginal の識別性に影響するか

**しない。** canonical joint model では `X ⊥ Y | Z` であり、`Y | Z` は `(w0, w, sigma_Y²)` にしか依らない。
`Z` の prior は `F` にも `family_x` にも依存しない。したがって `Y_ij` の周辺分布は
`F`・`Sigma_X`・`family_x` を一切含まない。§12 で論理を明示する。

### 3.8 canonical と historical fixed generator の分離

**厳密に分離した。** 本監査は canonical model のみを対象とし、
`expfam/src/data_generator_expfam.py` が実際に生成するデータの分布は **一切根拠に使っていない**。
分離の理由と帰結は §15。

---

## 4. Distribution of `S_K` — ゼロからの独立導出

`S_K := z_i^T z_j = Σ_{r=1}^{K} z_ir z_jr`、`z_i, z_j ~ iid N(0, I_K)` 独立。

### 4.1 単一項の MGF

`U, V ~ N(0,1)` 独立として `E[e^{tUV}]` を求める。`V` で条件づけると `tUV | V ~ N(0, t²V²)` なので

```
E[e^{tUV} | V] = exp( t² V² / 2 )
```

`V² ~ chi²_1` で `chi²_1` の MGF は `(1 − 2s)^{−1/2}`（`s < 1/2`）。`s = t²/2` を代入して

```
E[e^{tUV}] = (1 − t²)^{−1/2},        |t| < 1
```

`[DERIVED]`

### 4.2 `S_K` の MGF

`{z_ir z_jr}_{r=1..K}` は座標が独立なので **iid**。よって

```
M_{S_K}(t) = (1 − t²)^{−K/2},        |t| < 1                              … (4.1)
```

`[DERIVED]`。**依頼 §3 の候補式と一致する。独立に確認した。**

（別解として `z_j` で条件づけると `S | z_j ~ N(0, ‖z_j‖²)`、`‖z_j‖² ~ chi²_K` で
`M_S(t) = E[e^{t²‖z_j‖²/2}] = (1 − t²)^{−K/2}`。同じ結果。）

### 4.3 CGF と cumulant

```
kappa_{S_K}(t) = log M = −(K/2) log(1 − t²) = (K/2) Σ_{m≥1} t^{2m}/m
```

`t^{2m}` の係数は `K/(2m)`。`kappa_r = r! × [t^r 係数]` より

```
kappa_1 = 0                        （奇数次はすべて 0。CGF が t の偶関数）
kappa_2 = 2!  · K/(2·1) = K
kappa_4 = 4!  · K/(2·2) = 24K/4  = 6K
kappa_6 = 6!  · K/(2·3) = 720K/6 = 120K
kappa_8 = 8!  · K/(2·4) = 40320K/8 = 5040K
```

**一般に** `[DERIVED]`:

```
kappa_{2m}(S_K) = (2m)! · K/(2m) = (2m − 1)! · K,        m ≥ 1          … (4.2)
kappa_{2m+1}(S_K) = 0
```

検算: `m=1: 1!K = K` ✓ / `m=2: 3!K = 6K` ✓ / `m=3: 5!K = 120K` ✓ / `m=4: 7!K = 5040K` ✓。

**符号・factorial・係数をすべて再確認した。依頼 §3 の候補
`kappa_2 = K, kappa_4 = 6K, kappa_6 = 120K` は正しい。**

**構造的観察（重要）:** (4.2) は **すべての偶数次 cumulant が `K` に線形**で、
比例係数が `K` に依らない普遍定数 `(2m−1)!` であることを言っている。
すなわち `S_K` の分布は `K` について「1 次元のパラメータ族」として非常に硬い。
これが定理の本質的な理由である。

### 4.4 独立和分解 — **定理の急所** `[DERIVED]`

座標の独立性から、任意の `0 ≤ K' ≤ K` について

```
S_K  =  S_{K'} + S_{K−K'},        両者は独立、それぞれ S_{K'} / S_{K−K'} と同分布   … (4.3)
```

（MGF で確認: `(1−t²)^{−K/2} = (1−t²)^{−K'/2} · (1−t²)^{−(K−K')/2}` ✓）

**含意:** `K` 次元モデルは、**厳密に**「`K'` 次元モデル ＋ `w S_{K−K'}` という独立加法ノイズ」である。
したがって

> **`M_{K'}` が `K` 次元の法則を再現できない理由は「構造が違うから」ではない。
> ノイズ族が Gaussian に固定されているという、ただ 1 点だけである。**

この観察が §9.6 の反例と §11.F の判定を生む。

---

## 5. Gaussian-Y marginal law — 独立導出

```
Y = w0 + w S_K + eps,     eps ~ N(0, sigma²) ⊥ S_K
```

### 5.1 MGF とモーメント存在性

```
M_Y(t) = E[e^{tY}] = e^{w0 t} · M_{S_K}(wt) · e^{sigma² t²/2}
       = e^{w0 t} (1 − w² t²)^{−K/2} e^{sigma² t²/2}
```

有限であるのは `|wt| < 1`、すなわち `|t| < 1/|w|`（`w = 0` なら全実数）。
**原点の近傍で MGF が有限**なので、`Y` の **すべての次数のモーメントが有限**である。
とくに 6 次モーメントは有限（依頼 §7 の要求）。`[DERIVED]`

### 5.2 cumulant

`log M_Y(t) = w0 t + sigma² t²/2 − (K/2) log(1 − w²t²)`
`= w0 t + sigma² t²/2 + (K/2) Σ_{m≥1} w^{2m} t^{2m}/m`

`kappa_r = r! × [t^r 係数]`:

```
kappa_1(Y) = w0
kappa_2(Y) = 2!·( sigma²/2 + K w²/2 ) = K w² + sigma²
kappa_3(Y) = 0
kappa_4(Y) = 4!·( K w⁴ / (2·2) ) = 6 K w⁴
kappa_5(Y) = 0
kappa_6(Y) = 6!·( K w⁶ / (2·3) ) = 120 K w⁶
```

一般に `m ≥ 2` で `kappa_{2m}(Y) = (2m−1)! K w^{2m}`。`[DERIVED]`

**依頼 §4 の候補 `kappa_1 = w0`, `kappa_2 = Kw² + sigma²`, `kappa_4 = 6Kw⁴`, `kappa_6 = 120Kw⁶`
はすべて正しい。独立に確認した。**

**符号の確認:** `w ≠ 0` かつ `K ≥ 1` なら `w⁴ > 0`, `w⁶ > 0` なので

```
kappa_4(Y) = 6Kw⁴ > 0        kappa_6(Y) = 120Kw⁶ > 0
```

**`sigma²` は `kappa_2` にしか現れない。** これが Attack B が失敗する理由である。

---

## 6. Cumulant derivation — 反転公式の反証監査

### 6.1 反転公式

`w ≠ 0` のとき `kappa_4 ≠ 0` なので

```
kappa_6 / kappa_4 = 120Kw⁶ / (6Kw⁴) = 20 w²
⟹   w² = kappa_6 / (20 kappa_4)                                          … (6.1)
⟹   K  = kappa_4 / (6 w⁴)                                                … (6.2)
⟹   sigma² = kappa_2 − K w²                                              … (6.3)
⟹   w0 = kappa_1                                                         … (6.4)
```

`[DERIVED]`。依頼 §6 の候補 (6.1) は正しい。

### 6.2 アフィン不変形（本監査の追加結果）

(6.1) を (6.2) に代入して `w²` を消去する:

```
K = kappa_4 / (6 · (kappa_6/(20 kappa_4))² ) = kappa_4 · 400 kappa_4² / (6 kappa_6²)
  = (400/6) kappa_4³ / kappa_6² = (200/3) kappa_4³ / kappa_6²             … (6.5)
```

検算: `kappa_4³ = 216K³w^{12}`、`kappa_6² = 14400K²w^{12}` なので
`kappa_4³/kappa_6² = (216/14400) K = (3/200) K`、よって `(200/3)·(3/200)K = K` ✓。
**依頼 §6 の候補式は正しい。** `[DERIVED]`

**(6.5) は `Y → aY + b`（`a ≠ 0`）に対して不変である**（`kappa_4 → a⁴kappa_4`, `kappa_6 → a⁶kappa_6`、
`kappa_4³/kappa_6² → a^{12}/a^{12}` で不変）。
すなわち **`K` は Y の location・scale と完全に分離した shape 量である**。§11.H で使う。

### 6.3 反証チェックリスト（依頼 §6）

| 疑い | 判定 |
|---|---|
| factorial 係数ミス | **なし。** (4.2) の一般式 `(2m−1)!K` から `6K`・`120K` を再導出して一致 |
| cumulant / central moment の混同 | **これは実在する罠。** 中心モーメントは `mu_4 = kappa_4 + 3 kappa_2² = 6Kw⁴ + 3(Kw²+sigma²)²` であり **`sigma²` に汚染される**。尖度 `mu_4/mu_2²` だけでは `K` を分離できない。**必ず cumulant を使うこと** |
| `K` が整数であることへの依存 | **依存しない。** (6.1)(6.2) は `K` を実数として決める。`K ∈ (0,∞)` に緩めても証明は不変 |
| `w = 0` での division by zero | **正しく発生する。** `kappa_4 = kappa_6 = 0` で (6.1) が `0/0`。仮定 `w ≠ 0` は省略不可 |
| `sigma²` 未知の影響 | **なし。** `sigma²` は `kappa_4`・`kappa_6` に現れない |
| `w` の sign の影響 | `kappa_4 ∝ w⁴`, `kappa_6 ∝ w⁶` はいずれも偶数冪。**`sign(w)` は失われるが `K` には影響しない** |

---

## 7. Characteristic-function derivation — 独立な第 2 証明

cumulant proof と **論理的に独立**な証明を構成する。こちらはモーメントの存在を一切使わない。

### 7.1 `phi_Y` の導出

`z_j` で条件づけると `S | z_j ~ N(0, ‖z_j‖²)` なので

```
phi_{S_K}(u) = E[ e^{iuS} ] = E_{z_j}[ e^{−u²‖z_j‖²/2} ]
```

`‖z_j‖² ~ chi²_K`、その MGF は `(1−2s)^{−K/2}`。`s = −u²/2` を代入して

```
phi_{S_K}(u) = (1 + u²)^{−K/2},        すべての実 u                        … (7.1)
```

（`|t| < 1` という制限が無いことに注意。MGF と違い CF は全実軸で有効。）

```
phi_Y(t) = e^{i w0 t} · phi_{S_K}(w t) · e^{−sigma² t²/2}
         = e^{i w0 t} (1 + w² t²)^{−K/2} e^{−sigma² t²/2}                  … (7.2)
```

`[DERIVED]`。**依頼 §8 の候補式と一致する。**

### 7.2 CF による証明（moment-free）

> **主張 CF1.** `(w0, K, w², sigma²)` と `(w0', K', w'², sigma'²)` が同じ `phi_Y` を与え、
> かつ `w ≠ 0`, `K > 0` なら、`sigma² = sigma'²`, `K = K'`, `w² = w'²`, `w0 = w0'`。`[DERIVED]`

**証明.** (7.2) の絶対値は `(1+w²t²)^{−K/2} e^{−sigma²t²/2} > 0` で、位相は `e^{i w0 t}`。
両者の絶対値が全実 `t` で一致するので、`−2 log` を取って

```
K log(1 + w² t²) + sigma² t²  =  K' log(1 + w'² t²) + sigma'² t²      （∀ t ∈ R）   … (★)
```

**Step 1（`sigma²`）.** (★) を `t²` で割って `t → ∞`。`log(1+w²t²)/t² → 0` なので

```
sigma² = sigma'²
```

**Step 2（`K`）.** (★) は `K log(1+w²t²) = K' log(1+w'²t²)` に帰着する。
- `w' ≠ 0` の場合: 両辺を `log t` で割って `t → ∞`。`log(1+w²t²)/log t → 2` なので `2K = 2K'`、すなわち `K = K'`。
- `w' = 0` の場合: 右辺 `≡ 0`。左辺は `K ≥ 1`, `w ≠ 0` より `t ≠ 0` で狭義正。**矛盾。**
  ゆえに `w' = 0` は起こりえない。

**Step 3（`w²`）.** `K = K' > 0` を (★) に戻すと `log(1+w²t²) = log(1+w'²t²)` (∀t)、ゆえに `w² = w'²`。

**Step 4（`w0`）.** 絶対値が一致したので位相も一致し、`e^{i w0 t} = e^{i w0' t}` (∀t)、ゆえに `w0 = w0'`。∎

**この証明の利点:**

1. **モーメントの存在を使わない。** Lévy の一意性定理（CF ↔ 法則の 1 対 1）だけを使う。
2. **`w' = 0` の場合を場合分けなしに殺せる**（Step 2 に内蔵）。
3. `sigma² = 0`（degenerate noise）でも `K` が実数でも成立する。
4. `t → ∞` の漸近は `t^{−K}` という **多項式減衰次数**を読むことに対応する。
   すなわち **`K` は CF の裾の減衰指数として観測分布に刻まれている**。

**cumulant proof（§6）と CF proof（§7）は独立である。**
前者はモーメントの有限性を要し、後者は要さない。両者が同じ結論に至った。

---

## 8. 既存 P2 / P3 との突き合わせ

**本監査は既存 P2 / P3 を前提にせず、上記をすべて独立に導出した。** 突き合わせた結果:

| 項目 | 既存 `hardened` §9・§12 | 本監査の独立導出 | 一致 |
|---|---|---|---|
| `M_S(t) = (1−t²)^{−K/2}` | あり | (4.1) | ✓ |
| `kappa_2 = K, kappa_4 = 6K, kappa_6 = 120K` | あり | (4.2) の `m=1,2,3` | ✓ |
| `kappa_4(Y) = 6Kw⁴, kappa_6(Y) = 120Kw⁶` | あり | §5.2 | ✓ |
| `w² = kappa_6/(20 kappa_4)`, `K = kappa_4/(6w⁴)` | あり | (6.1)(6.2) | ✓ |
| P2（単一 dyad から `(w0,w²,K,sigma²)` 一意） | あり | §7.2 CF1 でより強く再証明 | ✓ |
| P3（`M_K ⊄ M_{K+1}`, `w≠0`） | あり（`[ARITHMETIC_CONSISTENCY]` のみ） | §10.3 で CF proof から再導出 | ✓ |

**本監査が既存に追加したもの:**

- 一般公式 `kappa_{2m}(S_K) = (2m−1)!K`（(4.2)）
- 独立和分解 `S_K = S_{K'} ⊕ S_{K−K'}`（(4.3)）— 定理の急所を露出させる
- moment-free の CF proof（§7.2）— 既存 P3 が持たなかった独立証拠を供給する
- アフィン不変な反転公式 (6.5)
- **`M_{K'}^{dyad}` の weak closedness と `inf KL > 0`**（§10.4）— 既存 U11 をこの場合に限り解消
- Attack A–G の網羅的な反例探索（§9）

**既存 P3 の `[ARITHMETIC_CONSISTENCY]`（独立数値証拠なし）という弱さについて:**
本監査の CF proof は乱数も数値も使わない **純解析的**な証明であり、
`kappa` を式に代入し直す循環（既存 §12.2 が自認していた弱点）を持たない。
したがって **P3 の証拠力は本監査によって実質的に補強された**。`[DERIVED]`

---

## 9. Counterexample search — Attack A–G

**すべて canonical model の枠内で、exact equality の反例を本気で探した。**

### 9.1 Attack A — `w` の rescaling

`K' ≠ K` に対し `w'` を選んでモーメントを合わせにいく。

**A-1: 分散を合わせる（`w' = w sqrt(K/K')`）**
`kappa_2' = K'w'² + sigma'² = Kw² + sigma'²` なので `sigma'² = sigma²` で `kappa_2` は一致する。
しかし
```
kappa_4' = 6K'w'⁴ = 6K' · w⁴K²/K'² = 6w⁴K²/K' = kappa_4 · (K/K')
```
`K' ≠ K` なら `kappa_4' ≠ kappa_4`。**FAILS。** `[DERIVED]`

**A-2: `kappa_4` を合わせる（`w' = w(K/K')^{1/4}`）**
`kappa_4' = 6K'w⁴K/K' = 6Kw⁴ = kappa_4` ✓。しかし
```
kappa_6' = 120K'w'⁶ = 120K'w⁶(K/K')^{3/2} = 120 w⁶ K^{3/2} K'^{−1/2}
```
`kappa_6 = 120Kw⁶` と等しくするには `K^{3/2}K'^{−1/2} = K`、すなわち `K' = K`。**FAILS。** `[DERIVED]`

**A-3: `kappa_6` を合わせる**
同様に `kappa_4` がずれる。

> **一般則（`[DERIVED]`）:** `(w0', w'², sigma'²)` は 3 自由度なので
> `kappa_1, kappa_2` と **`kappa_4` か `kappa_6` の一方**は必ず合わせられる。
> しかし **`kappa_4` と `kappa_6` を同時に合わせることは `K' = K` のときに限る。**
> `kappa_6/kappa_4 = 20w²` が `K` に依存しないので `w'² = w²` が強制され、
> そこから `kappa_4` が `K' = K` を強制する。**逃げ道は存在しない。**

### 9.2 Attack B — Gaussian noise とのトレードオフ

`sigma'²` は `kappa_2` にのみ現れる（§5.2）。`kappa_4`・`kappa_6` には一切影響しない。
したがって `sigma'²` をどう動かしても A の結論は変わらない。**FAILS。** `[DERIVED]`

CF proof（§7.2 Step 1）では、`sigma'²` は `t²` の係数として `t → ∞` で分離され、
`K` を決める `log t` の項に干渉できない。**構造的に不可能である。**

### 9.3 Attack C — 競合側に `w' = 0` を許す

`w' = 0` なら `Y' = w0' + eps' ~ N(w0', sigma'²)` で **完全に Gaussian**。
したがって `kappa_4' = 0`。一方 true 側は `kappa_4 = 6Kw⁴ > 0`（`w ≠ 0`, `K ≥ 1`）。**FAILS。** `[DERIVED]`

**この攻撃は定理の仮定の非対称性（`w ≠ 0` は true 側のみ）を突くもので、最も自然な攻撃だった。
しかし `kappa_4 > 0` が Gaussian からの乖離を厳密に定量しており、通らない。**
CF proof の Step 2 が同じことを場合分けなしに処理している。

### 9.4 Attack D — `sigma'² = 0`（degenerate noise）

`sigma'² = 0` は `kappa_2` を減らすだけで `kappa_4`・`kappa_6` を変えない。**FAILS。** `[DERIVED]`

**逆向きも確認した:** true 側で `sigma² = 0` の場合（`Y = w0 + wS_K`）も、
`kappa_4 = 6Kw⁴`, `kappa_6 = 120Kw⁶` は不変なので定理は成立する。
**したがって定理は `sigma_Y² ≥ 0` の閉区間全域で成立し、境界 `sigma_Y² = 0` を除外する必要はない。** `[DERIVED]`

（実装の floor `1e-6`（§3.5）はこの理論的結論に影響しない。）

### 9.5 Attack E — `w` と `−w`（符号非識別性）

(4.1) の CGF は `t` の偶関数なので `S_K` は 0 対称、すなわち `S_K =_d −S_K`。
よって `Law(w0 + wS + eps) = Law(w0 − wS + eps)`。**`sign(w)` は単一 dyad から識別できない。** `[DERIVED]`

**しかし `K` は `w²` を通じてしか `w` に依存しない**（(6.1)(6.2)、(7.2)）ので、
**`sign(w)` の非識別性は `K` の識別性を一切壊さない。** **攻撃として FAILS。** `[DERIVED]`

**注意:** これは「非識別性が見つかった」が「定理を壊さない」典型例である。
定理の主張範囲を `K` に限定していることが本質的に効いている（§11 で分離）。

### 9.6 Attack F — `K' = 0`

`K' = 0` を許した場合、`S_0 = `（空和）` = 0` なので `Y' = w0' + eps' ~ N(w0', sigma'²)`。
これは Attack C（`w' = 0`）と観測上同一である。`kappa_4' = 0 ≠ 6Kw⁴`。**FAILS。** `[DERIVED]`

**逆向き（重要）:** true 側が `K = 0` あるいは `w = 0` の場合、
`Y ~ N(w0, sigma²)` となり **すべての `K'` が同じ法則を与える**。
これは反例ではなく、**定理が明示的に除外している退化ケース**である（§11.A）。

### 9.7 Attack G — parameter limit（極限での一致）

有限パラメータで一致しないことは示した。**極限で厳密一致が生じないか**を検証する。

`(w0'_n, w'_n, sigma'_n) ∈ Theta_{K'}` の列で `Law(Y'_n) → Law(Y)` weakly となるものはあるか。

> **主張 G1.** `M_{K'}^{dyad}` は weak topology で **閉じている**。`[DERIVED]`

**証明.** `phi_n(t) = e^{i a_n t}(1 + b_n t²)^{−K'/2} e^{−c_n t²/2}`、`a_n ∈ R`, `b_n, c_n ≥ 0`。
弱収束すれば CF が各点収束する（Lévy）。
`|phi_n(1)| = (1+b_n)^{−K'/2}e^{−c_n/2}` が収束する。もし `b_n → ∞` または `c_n → ∞` なら
すべての `t ≠ 0` で `|phi_n(t)| → 0` となり、極限関数は `t=0` で `1`、`t≠0` で `0` という不連続関数になる。
これは CF ではない（CF は連続）ので弱収束の極限になりえない。ゆえに `b_n, c_n` は有界。
部分列を取って `b_n → b ≥ 0`, `c_n → c ≥ 0` とすると `|phi_n(t)| → r(t) := (1+bt²)^{−K'/2}e^{−ct²/2} > 0`。
すると各 `t` で `e^{i a_n t} → phi(t)/r(t)` が収束するので `a_n` も収束する（`a ∈ R`）。
極限の CF は `e^{iat}(1+bt²)^{−K'/2}e^{−ct²/2}`、すなわち `M_{K'}^{dyad}` の元。∎

**帰結:** `Law(Y) ∉ M_{K'}^{dyad}`（§7.2）かつ `M_{K'}^{dyad}` が閉なので

```
Law(Y) ∉ closure( M_{K'}^{dyad} )        （K' ≠ K, w ≠ 0）
```

**Attack G は FAILS。** 近似的な一致すら存在しない。`[DERIVED]`

（この帰結は §10.4 で joint へ持ち上げる。）

---

## 10. Does `K` uniquely follow?

### 10.1 十分性の論理（依頼 §7）

必要な論理は **一方向のみ**である:

```
同一分布  ⟹  同一 cumulant  ⟹  K 一致
```

**逆向き（moment sequence が分布を一意に決める＝moment determinacy）は不要である。** `[DERIVED]`
定理は「異なる `K` は異なる分布を与える」を主張しており、
「与えられたモーメント列を持つ分布が唯一である」ことは要求していない。

ただし cumulant proof を使う場合、**両方の分布で 6 次モーメントが有限**であることは必要である。
§5.1 で確認したとおり、両者とも原点近傍で MGF が有限なのですべてのモーメントが存在する。**ギャップなし。** `[DERIVED]`

**CF proof（§7.2）ではモーメント存在すら不要である**ので、こちらが論理的により安全な主証明である。

（参考: 原点近傍で MGF が有限なので `Y` の分布は実際には moment-determinate でもある。
しかし定理はそれを使わない。）

### 10.2 定理の結論（dyad marginal）

> **`w ≠ 0` なる true model に対し、`K' ≠ K` を持つ `M_{K'}^{dyad}` の元は存在しない。**
> したがって `{K' : Law(Y_ij) ∈ M_{K'}^{dyad}} = {K}` は **単元集合**であり、
> `min` は選択規約ではなく一意に決まる。`[DERIVED]`

これは既存 U12（候補集合が連結か）を **この場合に限り解消する**。

### 10.3 P3（非入れ子性）の再導出

`P ∈ M_K^{dyad}` で `w ≠ 0` なら `P ∉ M_{K'}^{dyad}`（∀`K' ≠ K`）。
とくに `K' = K+1` を取れば `M_K^{dyad} ⊄ M_{K+1}^{dyad}`（`w ≠ 0` の部分について）。`[DERIVED]`

**ただし `M_K ∩ M_{K+1} ≠ ∅` である。** `w = 0` の切片では
`Y ~ N(w0, sigma²)` がすべての `K` で実現され、joint でも X 側族が入れ子（`F' = [F, 0]`）なので
同じ joint が両方に属する。**「互いに素」とは書けない。** `[DERIVED]`
（既存 `hardened` §12.3-1 の A-F11 訂正と一致する。本監査でも独立に確認した。）

### 10.4 closure と KL（U11 の部分解決） `[DERIVED]`

> **主張 C2.** canonical Gaussian-Y、`w ≠ 0`、`K' ≠ K` のとき
> ```
> P0 ∉ closure( M_{K'} )      かつ      inf_{Q ∈ M_{K'}} KL(P0 ‖ Q) > 0
> ```

**証明.** (i) joint の弱収束は周辺の弱収束を含意する。`Q_n ∈ M_{K'}`、`Q_n → P0` weakly とすると
`Y_{12}` の周辺も収束する。`Q_n` の `Y_{12}` 周辺は `M_{K'}^{dyad}` に属し、主張 G1 よりこの族は閉。
よって `P0` の `Y_{12}` 周辺が `M_{K'}^{dyad}` に属することになるが、§7.2 CF1 に矛盾。
ゆえに `P0 ∉ closure(M_{K'})`。

(ii) `inf KL = 0` なら `KL(P0‖Q_n) → 0` なる列が取れる。Pinsker より `TV(P0, Q_n) ≤ sqrt(KL/2) → 0`、
よって `Q_n → P0` weakly。(i) に矛盾。∎

**これは既存 U11（`P0 ∉ M_{K+1}` は `P0 ∉ closure(M_{K+1})` を含意しない）を
canonical Gaussian-Y・`w ≠ 0` の場合に限って解消する。**
既存 `hardened` §12.3-2 が「閉性は未確立」としていた箇所に、この場合の答えを与える。
**他の family（Bernoulli-Y / Poisson-Y）については依然として `[UNRESOLVED]`。**

---

## 11. What else is / is not identifiable

**定理が証明するものを厳密に限定する。**

| 量 | 単一 dyad marginal から | 根拠 |
|---|---|---|
| `K` | **識別可能** | (6.5) / §7.2 Step 2 |
| `w0` | **識別可能** | `kappa_1` / §7.2 Step 4 |
| `w²`（= `|w|`） | **識別可能** | (6.1) / §7.2 Step 3 |
| `sigma_Y²` | **識別可能** | (6.3) / §7.2 Step 1 |
| **`sign(w)`** | **識別不能** | §9.5。`S_K` が 0 対称 |
| `F`, `Sigma_X` | **識別されない**（Y には現れない） | §3.7 |
| latent `Z` の回転 `O(K)` | **識別されない**（不変） | `z → Qz` で `z_i^Tz_j` 不変 |

### 11.1 `sign(w)` は `n ≥ 3` から識別される — **別 lemma として分離**

`n ≥ 3` の joint から、三角形 `(1,2,3)` の 3 次同時中心モーメントを取る。
`z_1` について先に期待を取ると `E[z_1 z_1^T] = I_K` より

```
E[ S_12 S_13 | z_2, z_3 ] = z_2^T E[z_1 z_1^T] z_3 = z_2^T z_3 = S_23
⟹ E[ S_12 S_13 S_23 ] = E[ S_23² ] = Var(S_K) = K
```

`Y_ab − w0 = w S_ab + eps_ab` で `eps` は独立平均 0 なので交差項が消え

```
E[ (Y_12 − w0)(Y_13 − w0)(Y_23 − w0) ] = w³ K
```

`K > 0` より符号は `sign(w)`。`[DERIVED]`（既存 P8 と一致。独立に再導出した。）

> **これは `K` 定理の本体には不要である。**
> `K` は `w²` にしか依存しないので（§9.5）、`sign(w)` を決める必要はない。
> **`n ≥ 3` を要求する別 lemma として分離し、定理本体に混ぜない。**

---

## 12. Joint `X, Y` implication

> **主張 J1.** `n ≥ 2` とする。canonical joint model において
> `P_K(X,Y) = P_{K'}(X,Y)` かつ true 側が `w ≠ 0` なら `K = K'`。`[DERIVED]`

**証明.** joint 分布が等しければ、そのすべての周辺分布が等しい。とくに `Y_{12}` の周辺が等しい。
canonical joint model では

```
p(X, Y) = ∫ p(Z) p(X | Z, F, family_x) p(Y | Z, w0, w, sigma_Y²) dZ
```

で `X ⊥ Y | Z` であり、`Z` の prior `N(0, I_K)^{⊗n}` は `F` にも `family_x` にも依存しない。
`(X, {z_l}_{l≠1,2})` をすべて積分すると、残るのは

```
Y_12 = w0 + w z_1^T z_2 + eps_12,   z_1, z_2 ~ iid N(0, I_K)
```

**すなわち §2.1 の dyad model そのもの**であり、`F`・`Sigma_X`・`family_x` を **一切含まない**。
§7.2 CF1 より `K = K'`。∎

### 12.1 何が不要か（依頼 §9）

| 条件 | 必要か |
|---|---|
| `rank(F) = K` | **不要** |
| X の family（Gaussian / Bernoulli / Poisson） | **不要** |
| X 側の identifiability（P1 / P5 / U1 / U3） | **不要** |
| `d ≥ K` | **不要** |
| `n ≥ 3` | **不要**（`n ≥ 2` で足りる。`n ≥ 3` は §11.1 の別 lemma のみ） |

**`K` の識別可能性は Y チャネル単独で完結する。** これは Gaussian-Y の際立った性質であり、
Bernoulli-Y（単一 edge の周辺では情報が乏しく、motif が要る）とは対照的である。

### 12.2 論理的飛躍がないことの確認

依頼が懸念した「単一 dyad marginal → joint」の飛躍は **存在しない**。
使っているのは「等しい分布は等しい周辺分布を持つ」という自明な含意だけであり、
向きは **joint 一致 ⟹ marginal 一致** である（逆向きは使っていない）。
逆向き（marginal 一致 ⟹ joint 一致）は **偽**であり、本証明はそれを必要としない。

---

## 13. Minimal assumptions

証明が実際に使った仮定を、使った箇所とともに列挙する。**1 つも落とせない。**

| # | 仮定 | どこで使うか | 落とすと |
|---|---|---|---|
| A1 | `z_i, z_j ~ iid N(0, I_K)`（`tau = 1` 固定） | (4.1) の `chi²_K` MGF | §14.B/C/D |
| A2 | `z_i ⊥ z_j` | (4.1) の条件づけ | 相関があれば `S` の法則が変わる。`[UNRESOLVED]` |
| A3 | 結合が **スカラー** `w` による `w z_i^T z_j` | (7.1) の指数が `K/2` になること | §14.E（反例あり） |
| A4 | `eps ~ N(0, sigma²)`、**ノイズ族が Gaussian に固定** | (7.2) の `e^{−sigma²t²/2}` が `t²` の項に閉じ込められること | §14.F（**反例あり**） |
| A5 | `eps ⊥ Z` | `phi_Y` の積分解 | 依存があれば分解が壊れる。`[UNRESOLVED]` |
| A6 | `w ≠ 0`（true 側のみ） | §7.2 Step 2 / (6.1) の分母 | §14.A（識別不能） |
| A7 | `K > 0` | 同上 | `K = 0` は `w = 0` と同値 |
| A8 | `sigma_Y² ≥ 0` | — | `= 0` でも成立（§9.4）。**制約は不要** |
| A9 | `n ≥ 2`（joint 版のみ） | §12 | dyad が無ければ Y が観測されない |

**`K` の整数性は不要**（§6.3）。**`d ≥ K`・`rank(F) = K`・X の family は不要**（§12.1）。

---

## 14. Failure modes — 依頼 §11 の 1 項目ずつの判定

| # | 変更 | 判定 | 根拠 |
|---|---|---|---|
| **A** | `w = 0` | **FAILS** | `Y ~ N(w0, sigma²)` となり全 `K` が同じ法則。`kappa_4 = 0` で反転公式が `0/0`。`[COUNTEREXAMPLE]`（自明） |
| **B** | `Z` を列 sample z-score | **NEEDS NEW PROOF / `[UNRESOLVED]`** | A1 が破れ (4.1) が成立しない。`‖z_j‖² ~ chi²_K` でないので (7.1) も無効。さらに `reports/research_scope/fixed_generator_kgen_k0_structural_audit_20260907.md` §12.3 により Scenario C では `P0 ∉ M_K`（∀K）。**本定理は適用不可**（§15） |
| **C** | `z_i ~ N(0, tau² I_K)`、`tau` 未知 | **K は SURVIVES / `|w|` は FAILS** | `S \|z_j ~ N(0, tau²‖z_j‖²)`、`‖z_j‖² = tau² chi²_K` より `M_S(t) = (1 − t²tau⁴)^{−K/2}`。すなわち `S =_d tau² S_K^{std}`。CF は `(1 + w²tau⁴t²)^{−K/2}` で **減衰指数は依然 `K`** → `K` は識別可能。しかし識別されるのは積 `w²tau⁴` だけで **`w` 単独は識別不能**。`[DERIVED]`。コード上 `var_z = 1.0` 固定（§3.1）なので実装では発生しない |
| **D** | `z_i ~ N(0, D_K)`、`D` 未知の不等対角 | **`D ≻ 0` なら SURVIVES / `D ⪰ 0` なら FAILS** | `M_S(t) = Π_r (1 − d_r² t²)^{−1/2}`、CF は `Π_r (1 + w²d_r²t²)^{−1/2}`。`t → ∞` で `~ t^{−(#\{r : d_r > 0\})}` なので **識別されるのは「非零 `d_r` の個数」**。`d_K = 0` を許すと `K=2, D=diag(d,0)` と `K'=1, D'=(d)` が **厳密に同一法則**。`[COUNTEREXAMPLE]` |
| **E** | `w z_i^T z_j → z_i^T W z_j` | **FAILS（識別されるのは `rank(W)` のみ）** | `S_W \| z_j ~ N(0, ‖W z_j‖²)`、`‖Wz_j‖² = Σ_r s_r² chi²_1`（`s_r` は `W` の特異値）。CF は `Π_r (1 + s_r² t²)^{−1/2}`。**反例:** `K=2, W=diag(w,0)` と `K'=1, w'=w` は厳密に同一法則。`[COUNTEREXAMPLE]`。コード上 `w` はスカラー（§3.4）なので実装では発生しない |
| **F** | `eps` を非 Gaussian に緩める | **FAILS（厳密な反例あり）** | §14.1 に構成する。`[COUNTEREXAMPLE]` |
| **G** | `sigma²` が未知 | **SURVIVES** | `sigma²` は `kappa_2` にのみ現れ（§5.2）、CF proof では `t²` の項として `t → ∞` で分離される（§7.2 Step 1）。`[DERIVED]` |
| **H** | Gaussian-Y の後に standardization | **定数によるアフィン標準化なら SURVIVES / 標本統計量による標準化は NEEDS NEW PROOF** | (6.5) の `K = (200/3)kappa_4³/kappa_6²` は `Y → aY+b` で不変（§6.2）なので、真の平均・SD で割る操作では `K` は失われない。しかし **標本平均・標本 SD は dyad 全体の関数**であり、dyad 間に厳密な決定論的制約を入れる（`Σ Y_std = 0` 等）。これは周辺の shape の問題ではなく joint の構造の問題であり、本定理の射程外。`[UNRESOLVED]` |
| **I** | finite sample のみ | **NEEDS NEW PROOF** | 本定理は population の主張である。`n(n−1)/2` 個の dyad は `n` 個の潜在ベクトルを共有し **独立でない**ので、`kappa_4`・`kappa_6` の標本版は iid 平均ではない。推定量の一致性・収束率は本監査の射程外（既存 U6）。`[UNRESOLVED]` |

### 14.1 Attack F の厳密な反例 `[COUNTEREXAMPLE]`

**構成.** (4.3) より `S_2 = S_1 + S_1'`（`S_1, S_1'` 独立同分布）。したがって:

```
true      :  K = 2,  w0 = 0,  w = 1,  sigma² = 0      ⟹  Y  = S_2
competitor:  K' = 1, w0' = 0, w' = 1, eps' ~ Law(S_1) ⟹  Y' = S_1 + eps'
```

`eps'` は `S_1`（2 個の独立標準正規の積、平均 0・分散 1、密度は変形 Bessel 関数 `K_0` を含む）と
同じ分布を持つ **正当な確率分布**であり、`S_1` と独立に取れる。すると

```
Law(Y') = Law(S_1 + S_1') = Law(S_2) = Law(Y)
```

**`K = 2` と `K' = 1` が厳密に同一の法則を与える。** ∎

**一般に任意の `K' < K` について、`eps' ~ Law(w S_{K−K'})` と取れば同じことが起こる。**
すなわち **ノイズ族を指定しなければ `K` は完全に識別不能である。**

> **この反例の意味:** 定理を支えているのは「Gaussian-Y だから `Y` が扱いやすい」ではなく、
> **「ノイズが Gaussian に固定されているため、`w S_{K−K'}` という非 Gaussian な残余を吸収できない」**
> という一点である。`kappa_4(w S_{K−K'}) = 6(K−K')w⁴ > 0` が、その残余が Gaussian でないことの定量である。
> **定理は 1 つの仮定を外しただけで偽になる。**

---

## 15. Historical fixed non-applicability

### 15.1 なぜ適用できないか

`expfam/src/data_generator_expfam.py:283` は生成後に

```python
Z = normalize_zscore(Z, axis=0)
```

を実行し、各列に `Σ_i z_ir = 0` と `Σ_i z_ir² = n` を **厳密に**課す（余次元 `2K`）。
したがって

- `z_i` は **iid ではない**（行間に決定論的依存が入る）。A1 が破れる。
- `‖z_j‖² ~ chi²_K` が成立しない。したがって (4.1) の MGF 導出も (7.1) の CF 導出も **無効**。
- 定理の 2 本の証明はいずれも第一歩で止まる。

これは既存 KI-021 G1 が記録している事実であり、`hardened` §11.3 も
「O1 は canonical model についての主張であり historical データの分布についての主張ではない」と
同じ境界を引いている。

### 15.2 明示的な禁止事項

**以下を書いてはならない:**

- 「Gaussian-Y の identifiability theorem により fixed Scenario C の `K_TRUE = 3` は真の `K` である」
- 「本定理が fixed 30/30 を true-K recovery として正当化する」
- 「Scenario C は Gaussian-Y なので `K` が識別可能な設定である」

**理由は 2 段階ある。**

1. **前提が満たされない**（§15.1）。列 z-score により `z_i ~ iid N(0, I_K)` が成立しない。
2. **さらに強く、`K_0` がそもそも存在しない。**
   `reports/research_scope/fixed_generator_kgen_k0_structural_audit_20260907.md` §12.3 は、
   fixed Scenario C の `p_*^C` が `M_K` のどの元とも一致しないことを（Cramér の分解定理により）導出している。
   `{K : p_* ∈ M_K} = ∅` なので `min` の対象が空である。

**本監査は canonical model についてのみ結論を出した。fixed の数値結果は本定理の証拠として一切使っていない。**

### 15.3 逆に、本定理が言えること

canonical model（列 z-score なし、`z_i ~ iid N(0,I_K)`）で Gaussian-Y・`w ≠ 0` のデータを生成すれば、
**`K_0 = K_GEN` が厳密に成立する**（§10.2 より候補集合が単元集合）。
これは前回監査（`fixed_generator_kgen_k0_structural_audit_20260907.md` §17-1）が挙げた
「generator を canonical に合わせる」という選択肢の **理論的裏付け**になる。
**ただしどの選択肢を採るかは Human Gate であり、本監査は決めない。**

---

## 16. Corrected theorem statement

監査の結果、依頼 §13 の草案に **修正の必要はなかった**が、
使った仮定を省略なく明示した形に書き直す。

> **Theorem（canonical Gaussian-Y、dyad marginal）.** `[DERIVED]`
>
> `K > 0`（実数でよい）とし、`z_i, z_j` は互いに独立に `N(0, I_K)` に従うとする。
> `eps_ij ~ N(0, sigma_Y²)`（`sigma_Y² ≥ 0`）は `(z_i, z_j)` と独立とする。
> `w0, w ∈ R` に対し
>
> ```
> Y_ij = w0 + w · z_i^T z_j + eps_ij
> ```
>
> と置く。**`w ≠ 0` ならば、`Y_ij` の周辺分布は `(w0, K, w², sigma_Y²)` を一意に決定する。**
>
> すなわち、同じ形の任意の表現 `(w0', K', w'², sigma_Y'²)`（`K' > 0`、`w'` は 0 でもよい）が
> 同一の周辺分布を与えるなら `w0' = w0`, `K' = K`, `w'² = w²`, `sigma_Y'² = sigma_Y²` である。
>
> **`sign(w)` は決定されない。**
>
> **Corollary（joint）.** `n ≥ 2` とし、canonical joint model
> `p(X,Y) = ∫ p(Z) p(X|Z,F,family_x) p(Y|Z,w0,w,sigma_Y²) dZ` を考える。
> true 側が `w ≠ 0` なら、`P_K(X,Y) = P_{K'}(X,Y)` は `K = K'` を含意する。
> **この系は `F` の階数・`family_x`・`d ≥ K`・X 側の identifiability を一切要求しない。**
>
> **Corollary（closure）.** さらに `K' ≠ K` のとき `P0 ∉ closure(M_{K'})` かつ
> `inf_{Q ∈ M_{K'}} KL(P0‖Q) > 0` である。
>
> **Lemma（別立て、`n ≥ 3`）.** `E[(Y_12−w0)(Y_13−w0)(Y_23−w0)] = w³K` により `sign(w)` が決まる。
> **これは定理本体には不要であり、混ぜない。**

**証明は §7.2（CF、moment 不要）を主とし、§6（cumulant）を独立な第 2 証明とする。**

---

## 17. What this theorem does NOT prove

**過大解釈を防ぐために明示する。**

1. **「Gaussian-Y なら何でも `K` が分かる」ではない。**
   ノイズ族が Gaussian に固定されていることが本質（§14.1 の反例）。
   結合がスカラー `w` であることも本質（§14.E の反例）。
2. **`sign(w)` を決めない**（§9.5）。`n ≥ 3` の別 lemma が要る。
3. **`F`・`Sigma_X`・X 側パラメータについて何も言わない**（§11）。
4. **`Z` そのものを識別しない。** `O(K)` 回転・符号反転・座標置換は観測分布を変えない。
5. **推定量の consistency を一切含まない**（§14.I、既存 U6）。population の主張である。
6. **有限標本で `kappa_4`・`kappa_6` を推定できるとは言っていない。**
   `n(n−1)/2` 個の dyad は独立でない。標本 cumulant の収束率は未確立。
7. **他の family へ移送できない。** Bernoulli-Y（既存 U2）・Poisson-Y（既存 U4）は依然 `[UNRESOLVED]`。
   本監査は Gaussian-Y のみを扱った。
8. **historical fixed generator のデータについて何も言わない**（§15）。
9. **`M_K` の閉性を一般に示したのではない。** §10.4 は canonical Gaussian-Y・`w ≠ 0` に限る。
   他 family の U11 は未解決のまま。
10. **model selection criterion の妥当性とは無関係である。**
    identifiability は「population で区別できるか」であり、
    `calc_bic_dual`（Q ベース基準、Schwarz BIC ではない。KI-010）が正しく選ぶかは別問題。

---

## 18. Final verdict

### **PROVED AS STATED**

依頼 §1 の Theorem candidate は、そこに明示された canonical Gaussian-Y model のもとで
**そのとおり成立する**。7 系統の攻撃（Attack A–G）はすべて失敗し、反例は 1 つも見つからなかった。
2 本の独立な証明（cumulant / characteristic function）を構成し、
うち CF proof はモーメントの存在すら要さない。

**ただし判定に 3 つの但し書きを付す。**

1. **定理は tight である。** §13 の A1–A7 は **1 つも落とせない**。
   とくに **A4（ノイズが Gaussian）と A3（結合がスカラー）は落とすと厳密な反例が構成できる**
   （§14.1、§14.E）。`S_K = S_{K'} ⊕ S_{K−K'}` という独立和分解がすぐそこにあるため、
   「Gaussian-Y だから安全」という直観は誤りである。**定理を支えているのはノイズ族の固定である。**
2. **既存 P2/P3 と一致したが、それを根拠にしていない。**
   すべて独立に再導出し、そのうえで一致を確認した（§8）。
   既存 P3 が `[ARITHMETIC_CONSISTENCY]`（独立証拠なし）だった弱点は、
   本監査の純解析的 CF proof によって補強される。
3. **historical fixed への適用は禁止**（§15）。列 sample z-score により前提 A1 が破れ、
   さらに前回監査により Scenario C では `K_0` 自体が存在しない。
   **`fixed 30/30` の true-K proof として本定理を引用してはならない。**

**新たに得られた副産物:**

- 一般公式 `kappa_{2m}(S_K) = (2m−1)! K`（§4.3）
- アフィン不変な反転公式 `K = (200/3) kappa_4³/kappa_6²`（§6.2）
- `M_{K'}^{dyad}` の weak closedness と `inf KL > 0`（§9.7・§10.4）→ **既存 U11 をこの場合に限り解消**
- 候補集合が単元集合であること → **既存 U12 をこの場合に限り解消**
- Attack D/E から「識別されるのは結合作用素の非零特異値の個数」という一般化された読み（§14）

---

## 付録: 本監査の手続き

- **READ-ONLY。** コード変更・実験実行・simulation・generator 再実行・Issue/PR 作成はいずれも行っていない。
- 使用した一次コード: `reproduction/src/model.py`（`:98-99, 128, 818`）、
  `expfam/src/model_expfam.py`（`:51-59, 143-147, 180-210, 212-235, 241-269`）、
  `expfam/src/run_fixed_official_exp1_bic_full.py`（`:49`）、
  `expfam/src/run_fixed_official_exp1_bic_k9_extension.py`（`:61`）。
- 参照した既存理論文書: `reports/identifiability/true_k_identifiability_hardened_20260904.md`
  （**突き合わせのみ。前提としていない**）、
  `reports/research_scope/fixed_generator_kgen_k0_structural_audit_20260907.md`（§15 の境界確認）。
- **historical fixed の数値結果を canonical theorem の証拠に使っていない**（依頼 §0 の絶対条件）。
- 既存文書は一切変更していない。本監査は新規ファイル 1 本のみを追加する。
