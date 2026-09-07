# 属性 X の列ごとに分布族が異なるとき、現行の K 選択はどう振る舞うか

**作成日:** 2026-09-07
**種別:** confirmatory experiment（設定を凍結してから実行、結果に応じた変更なし）
**branch:** `experiment/per-column-mixed-x-k-selection-20260907`（isolated worktree、base `bf07887`）
**model:** `experimental/model_dual_expfam_consistent.py` :: `DualExpFamLSMPerColumnConsistent`
（per-column **prototype**、`numerics_mode="consistent"`。root `CLAUDE.md` §3 により**本文採用不可**）
**criterion:** `experimental/eval_utils.py` :: `calc_Q_dual_strict_exp` + `calc_bic_exp`
— **Q-based complete-data criterion**。**Schwarz BIC ではない**（KI-010）。
関数名・CSV 列名の歴史的呼称 `BIC` は変更していないが、本文では一貫して **Q-based criterion** と呼ぶ。

---

## 1. なぜこの実験を行ったか

先生から次の指摘を受けている。

> 「属性 X の列ごとに Bernoulli / Gaussian / Poisson など異なる確率分布を使う場合でも、
> これまでと同じ K の選び方で潜在次元 K を決めてよいのか。理論的な性質も含めて考える必要がある」

この問いに答えるには、まず **「真の K」が何であるかが確定していること**が必要である。
それが確定していなければ、K 選択が「当たった/外れた」を判定する基準自体が存在しない。

先行して次が確定していた。

1. `reports/research_scope/fixed_generator_kgen_k0_structural_audit_20260907.md`
   — 生成時の `K_GEN` と観測分布上の最小潜在次元 `K_0` は別概念であり、
   historical generator では `Z` の列 z-score により `p_* notin M_K`（∀K）となって
   **`K_0` がそもそも存在しない**。
2. `reports/identifiability/gaussian_y_k_identifiability_adversarial_audit_20260907.md`
   — canonical Gaussian-Y（`z_i ~ iid N(0,I_K)`、`w != 0`）では
   **単一 dyad の周辺分布だけで `K` が一意に決まる**（反例探索で棄却されず、独立な 2 証明）。
3. `reports/identifiability/canonical_gaussian_y_k_selection_experiment_20260907.md`
   — その条件で現行 Q-based criterion は `K_0` を 70/70 で選択したが、
   選択を決めているのは明示罰則ではなく **`-2E[log p(Z)]` という決定論的な `O(nK)` の隠れ罰則**だった。

本実験はこの 3 番目の続きであり、**X の family 構成だけを動かしたとき K 選択がどう変わるか**を測る。

**本実験は per-column の identifiability を新たに証明するものではない。** §2 の理由により
`K_0` は Y 側だけで固定されるので、X の family 構成を変えても ground truth は動かない。

---

## 2. Gaussian-Y を固定する理由

`gaussian_y_k_identifiability_adversarial_audit_20260907.md` §12 の系（`n >= 2`）:

canonical joint model では `X ⊥ Y | Z` であり、`Y_ij` の周辺分布は
`F`・`family_x`・`d` を**一切含まず**

```
Y_ij = w0 + w z_i^T z_j + eps_ij,   z_i, z_j ~ iid N(0, I_K),  eps ~ N(0, sigma_y^2)
phi_Y(t) = exp(i w0 t) (1 + w^2 t^2)^{-K/2} exp(-sigma_y^2 t^2 / 2)
```

に帰着する。`w != 0` ならこの周辺分布が `(w0, K, w^2, sigma_y^2)` を一意に決める。

**したがって X の family 構成をどう変えても `K_0` は動かない。**
これが「X の family だけを動かす」実験を成立させる唯一の仕掛けである。
Bernoulli-Y や Poisson-Y を使うと identifiability 自体が未解決（既存 U2 / U4）になり、
`K_0` が不明のまま K 選択を評価することになってしまう。

---

## 3. なぜ `K_0` が既知と言えるか

`w = 0.5 != 0` の本設定では `{K : p_* in M_K} = {K_GEN}` が単元集合であり

```
K_0 = K_GEN = 3        （理論的 ground truth、X の family 構成に依存しない）
```

config gate **C4** がこの条件をコード上で確認している。

**限定:** これは population identifiability の主張であり、
finite-sample recovery・criterion の妥当性・selection consistency のいずれも保証しない。
本実験が測るのはまさにその後半である。

---

## 4. per-column 実装監査

**本実験を回す前に、per-column 実装が single-family 実装と同じ計算をしているかを監査した。**
`tools/research_audit/audit_percolumn_parity.py`、結果は `audit_percolumn_parity.csv`。

同一データ・同一 K・同一 seed・同一初期値で、全 15 列を Bernoulli にして比較した。

- **A.** `DualExpFamLSMFixed(family_x="bernoulli")` + `calc_Q_dual_strict` / `calc_bic_dual`（core 実験の経路）
- **B.** `DualExpFamLSMPerColumnConsistent(family_x_list=["bernoulli"]*15)` + `calc_Q_dual_strict_exp` / `calc_bic_exp`

### 4.1 理論上一致すべき部分 — **すべて bit-identical**

| 項目 | 結果 |
|---|---|
| 初期 `Z` / `F` / `var_z` | `max|diff| = 0.000e+00` |
| `A'_X`（Bernoulli 平均）、`A''_X`（曲率） | `max|diff| = 0.000e+00`（`eta in [-8,8]`） |
| `num_params`（K=1..9） | 完全一致 |
| EM 後の `Z` サンプル | `0.000e+00` |
| 推定 `F` | `0.000e+00` |
| `w0` / `w` / `sigma_y` | `0.000e+00` |
| **selected K** | 両者 `K = 3` |

### 4.2 実装上異なり得る部分 — **既知の定数オフセットのみ**

`calc_Q_dual_strict_exp` は Gaussian-Y に `-0.5 ln(2*pi)` を観測ペアごとに加える一方、
core が使った `calc_Q_dual_strict` はこれを含まない。実測:

```
Q_exp - Q_dual = -10269.1381   （予測値 -0.5*ln(2pi)*11175 = -10269.1381、差 1.8e-12）
C_exp - C_dual = +20538.2762   （K 方向のばらつき 7.3e-12）
```

**この差は `K` に依存しない定数なので argmin は不変**であり、K 選択の比較を壊さない。
ただし **absolute score を canonical core の表・図に混ぜてはならない**。
`run_manifest.json` の `comparability` に明記した。

### 4.3 KI-015（per-column の objective inconsistency）

既知問題を再現確認した。`eta = 30`（Poisson clip 窓 `[-20,10]` の外）で

| 実装 | `A''_X(30)` |
|---|---|
| legacy per-column（`model_dual_expfam_percolumn.py`） | `1.000e-08`（floor、objective-inconsistent） |
| legacy single-family（`model_dual_expfam_fixed.py`） | `1.000e-08`（同じ floor） |
| **consistent per-column（本実験で使用）** | **`9.348e-14`（真値）** |

KI-015 の運用ルール「新しい per-column 実験は `numerics_mode="consistent"` を明示的に選ぶ」に従い、
**本実験は全 config で `DualExpFamLSMPerColumnConsistent` を使用した**（config gate C8）。
したがって **KI-015 は本実験には適用されない**。

さらに consistent 経路の Poisson は **clip せず overflow 時に例外を投げる** fail-fast 設計である。
本実験では例外 0 件（audit A4）、生成時の `max eta = 3.565`、fit 中の `max|eta_x| = 6.588` で
overflow 閾値 `log(finfo.max) = 709.78` から十分遠い。

### 4.4 判定

> **PASS。BLOCKER なし。** per-column 実装は、全列同一 family のとき single-family 実装と
> 同じ目的関数を計算している。唯一の差は K に依存しない定数であり、K 選択比較を壊さない。

---

## 5. 実験条件（実行前に凍結）

| 項目 | 値 |
|---|---|
| `K_0 = K_GEN` | 3 |
| `n` / `d` | 150 / 15 |
| candidate `K` | 1,…,9 |
| trials | 10（各 config） |
| `L` / `num_iter` | 5 / 8 |
| Y | Gaussian、`w0 = 0`、`w = 0.5`、`sigma_y = 0.1` |
| Z | `iid N(0, I_K)`、**generator 側で z-score も標準化も一切しない** |
| F | `N(0, 5)` を行正規化して `||f_l|| = sqrt(1 - uniq)`、`uniq = 0.1`（historical 規約） |
| Gaussian-X 列 | `x ~ N(eta, sigma_x^2)`、`sigma_x^2 = uniq = 0.1`（`||f_l||^2 + sigma_x^2 = 1`） |
| Bernoulli-X 列 | `x ~ Bernoulli(sigmoid(eta))` |
| Poisson-X 列 | `x ~ Poisson(exp(eta))`、**clip なし** |
| estimator init | core runner 逐語（`w0 = mean(Y_upper)`、`w = 0.5`、`sigma_y = max(sd,0.01)`） |
| `F *= 0.2` | **全 config に一律適用**（下記） |
| tie rule | 最小 K（`ties.csv` に別途記録） |

**凍結した意図的差分:** core runner は `F *= 0.2` を離散 X のときだけ適用していた。
本実験では**全 config に一律適用**した。理由は、比較対象である family 構成と初期化が
交絡してはならないためである。この決定は実行前に `run_manifest.json` に記録した。

**paired design（§6 要件）:** `data_seed = 800000 + trial` を**全 config で共有**し、
`Z`・`F`・`Y` は `default_rng(data_seed)` の第 1 ストリームから、
`X` は `default_rng(data_seed + 500000)` の**独立ストリーム**から引く。
これにより **X の family 構成を変えても `Z`・`F`・`Y` は bit-identical** になる。
config gate C1（生成時）と audit A5/A6（実行時、630 行全数）で検証した。

**CONFIG GATE C1–C10: 10/10 PASS**（`config_gate.csv`）。

---

## 6. Homogeneous controls と mixed の構成

`family_config.csv`。列は先頭からブロック割り当て。

| config | kind | Bernoulli | Gaussian | Poisson |
|---|---|---|---|---|
| **H-B** | homogeneous | 15 | 0 | 0 |
| **H-G** | homogeneous | 0 | 15 | 0 |
| **H-P** | homogeneous | 0 | 0 | 15 |
| **M-BG** | mixed | 8 | 7 | 0 |
| **M-BP** | mixed | 8 | 0 | 7 |
| **M-GP** | mixed | 0 | 8 | 7 |
| **M-BGP** | mixed | 5 | 5 | 5 |

homogeneous control を 3 つ置いたのは、**「family そのものの難しさ」と「混在の影響」を分離する**ためである。

---

## 7. 主結果 — mixed-family results

`aggregate_recovery.csv`、図 `recovery_by_family_config.png` / `selected_K_by_family_config.png`

| config | kind | trials | `K_hat = K_0` | recovery rate | mean gap to 2nd best |
|---|---|---|---|---|---|
| H-B | homogeneous | 10 | 10 | **1.00** | 319.98 |
| H-G | homogeneous | 10 | 10 | **1.00** | 314.91 |
| H-P | homogeneous | 10 | 10 | **1.00** | 316.04 |
| M-BG | mixed | 10 | 10 | **1.00** | 321.82 |
| M-BP | mixed | 10 | 10 | **1.00** | 320.94 |
| M-GP | mixed | 10 | 10 | **1.00** | 319.16 |
| M-BGP | mixed | 10 | 10 | **1.00** | 326.63 |
| **合計** | | **70** | **70** | **1.00** | |

- **同点 0 件**（`ties.csv` は空）。tie rule は一度も発動していない。
- 過大選択も過小選択も 0。`selected_k_mean` は全 config で厳密に 3.0。
- 失敗 0 / 非有限 0 / Poisson overflow 0 / internal retry 0。

**パラメータ回収**（`k = K_0` において。Y は paired design で全 config 同一）:

| config | `w_hat`（真 0.5） | `sigma_y_hat`（真 0.1） |
|---|---|---|
| H-B / H-G / H-P | 0.5083 / 0.5085 / 0.5084 | 0.1002 |
| M-BG / M-BP / M-GP / M-BGP | 0.5086 / 0.5083 / 0.5084 / 0.5083 | 0.1002 |

Y 側パラメータは family 構成にほとんど依存しない（`w_hat` の幅は 0.0003）。

---

## 8. K-selection decomposition

`score_decomposition.csv`、図 `score_decomposition_mixed_BGP.png`

criterion は厳密に分解される（630 fits すべてで残差 `<= 1.819e-11`、audit A10）。

```
C_Q(K) = D_K + P_Z(K) + P_theta(K)
  D_K        = -2 E[ log p(X|Z) + log p(Y|Z) + (K-independent strict corrections) ]
  P_Z(K)     = -2 E[ log p(Z) ]
  P_theta(K) = p_hat(K) log n,    p_hat(K) = K d - K(K-1)/2 + n_gaussian_cols + 1
```

`K_0 -> K_0 + 1` での限界収支:

| config | kind | `Δ D_K` | うち X 由来 | うち Y 由来 | `Δ P_Z` | `Δ P_theta` | `Δ C_Q` | implicit/explicit |
|---|---|---|---|---|---|---|---|---|
| H-B | homog | −165.83 | −27.38 | −138.45 | **425.68** | **60.13** | +319.98 | 7.08 |
| H-G | homog | −170.90 | −31.48 | −139.42 | **425.68** | **60.13** | +314.91 | 7.08 |
| H-P | homog | −169.77 | −31.13 | −138.65 | **425.68** | **60.13** | +316.04 | 7.08 |
| M-BG | mixed | −163.98 | −25.78 | −138.21 | **425.68** | **60.13** | +321.82 | 7.08 |
| M-BP | mixed | −164.87 | −23.95 | −140.93 | **425.68** | **60.13** | +320.94 | 7.08 |
| M-GP | mixed | −166.65 | −35.86 | −130.79 | **425.68** | **60.13** | +319.16 | 7.08 |
| M-BGP | mixed | −159.18 | −27.22 | −131.95 | **425.68** | **60.13** | +326.63 | 7.08 |

### 8.1 観察 1 — 2 つの罰則項は X の family 構成に**構造的に**依存しない

- **`Δ P_Z = 425.68` が 7 config で完全に同一。**
  audit A12 が `P_Z(K) = n K (log 2*pi + 1)` を残差 `9.095e-13` で確認している。
  この項は `Z` のスケールを `scale_Z` が固定するため決定論的で、**X を一切含まない**。
- **`Δ P_theta = 60.13` も 7 config で完全に同一。**
  `p_hat` の絶対値は Gaussian 列数だけ増える（H-B: 43 → H-G: 58 at `K=3`）が、
  Gaussian 列数の項は **`K` に依存しない**ので、`K` 方向の差分 `Δ p_hat = 12` は全 config で同じである。

**したがって X の family 構成が動かせるのは `D_K` だけである。** これは実測ではなく構造的事実である。

### 8.2 観察 2 — `D_K` は動くが、決定を覆すには 1 桁足りない

`Δ D_K` の範囲は −159.18（M-BGP）〜 −170.90（H-G）、幅 11.7（約 7%）。
一方で隠れ罰則は 425.68 で固定である。決定的な比は

```
|Δ D_K| / Δ P_Z  =  0.374 〜 0.401     （全 7 config）
```

すなわち **どの family 構成でも、余分な 1 次元が稼ぐ当てはまりは隠れ罰則の 4 割程度にとどまる**。
family 構成による 7% の変動では、この 2.5 倍の安全余裕を埋められない。
**これが「混在させても K 選択が動かなかった」ことの機構的説明である。**

### 8.3 観察 3 — X チャネルの寄与自体が小さい

| config | X 由来の割合 |
|---|---|
| M-BP | 14.5% |
| M-BG | 15.7% |
| H-B | 16.5% |
| M-BGP | 17.1% |
| H-P | 18.3% |
| H-G | 18.4% |
| M-GP | **21.5%** |

`Δ D_K` の **8 割前後が Y 由来**である。`n = 150` では Y が `n(n-1)/2 = 11175` dyad を、
X が `n d = 2250` 要素を供給し、さらに `sigma_y = 0.1` で Y の精度が非常に高い。
**X の family をどう組み替えても、そもそも決定に効く余地が全体の 15〜22% しかない。**

### 8.4 観察 4 — Y 由来の項も config でわずかに動く（`Y` データは同一なのに）

paired design により `Y` は全 config で bit-identical であるにもかかわらず、
`Δ` の Y 部分は −130.79（M-GP）〜 −140.93（M-BP）と動いている。
これは **X が潜在変数の事後分布を通じて Y の当てはまりに間接的に影響する**ためである
（`Z` サンプルが X に依存し、その `Z` で Y 尤度を評価している）。
**観察された事実として記録するが、その大きさの由来は本設計では切り分けていない。**

---

## 9. `p(Z)` counterfactual diagnostic

**診断のみ。criterion の変更提案では一切ない。** `diagnostic_counterfactual_selection.csv`、
図 `prior_counterfactual_by_family_config.png`

`score_without_direct_prior(K) = D_K + P_theta(K)` で選び直すと:

| config | 選択が変わった | 反事実の `K_hat` |
|---|---|---|
| H-B / H-G / H-P | 10/10 / 10/10 / 10/10 | すべて `K = 9` |
| M-BG / M-BP / M-GP / M-BGP | 10/10 ×4 | すべて `K = 9` |
| **合計** | **70 / 70** | **70/70 が `K = 9`（候補上限）** |

**`P_Z` を外すと、どの family 構成でも 1 件も `K_0` を選ばず、全件が候補上限に張り付く。**
mixed でも homogeneous でも完全に同一の挙動であり、
**現行 criterion の `P_Z` 依存度は X の family 構成によって変化しない。**

これは canonical Gaussian-Y core の所見（70/70 変化、67/70 が `K=9`）と同じ向きであり、
本実験ではさらに均一（70/70 が `K=9`）である。

---

## 10. 何が分かったか

**判定は CASE A である。**

1. **明示した有限標本設定において、X の列ごとに分布族を混在させても、
   現行 Q-based criterion は `K_0` を 70/70 trial で選択した。** 同点 0、過大・過小選択 0。
   homogeneous control 3 種でも 30/30 で、**CASE C（homogeneous 段階での破綻）は起きていない**。
2. **混在は margin をわずかに広げた**（homogeneous 平均 316.97 → mixed 平均 322.14、+1.6%）。
   最も混ざった M-BGP が最大 margin（326.63）だった。
   **「混在すると不安定になる」という向きの証拠は得られなかった。**
3. **機構的な理由が特定できた。** criterion の 2 つの罰則項のうち
   - `P_Z` は `n K (log 2*pi + 1)` で **X を含まない**（残差 `9.1e-13`）
   - `P_theta` の `K` 微分は Gaussian 列数に依存しない（`Δ p_hat = 12` で全 config 同一）

   ため、**X の family 構成が動かせるのは当てはまり項 `D_K` だけ**である。
   その `D_K` は隠れ罰則の 0.374〜0.401 倍しかなく、family 構成による変動は 7% にとどまる。
   **2.5 倍の安全余裕を 7% の変動では覆せない。**
4. **X チャネルの寄与は限界収支の 14.5〜21.5% にすぎない**（残りは Y 由来）。
5. **`P_Z` 依存度は family 構成に依存しない**（counterfactual が 7 config すべてで 10/10、全件 `K=9`）。
6. **per-column 実装は single-family 実装と同じ目的関数を計算している**（§4、全列同一 family で bit-identical）。

---

## 11. 何がまだ言えないか

1. **「per-column でも BIC が理論的に正しい」とは言えない。** これは Schwarz BIC ではなく（KI-010）、
   本実験は理論的性質を一切証明していない。
2. **model-selection consistency を示していない。** `n` は 150 の 1 点のみである（既存 U6）。
3. **一般的な true-K recovery ではない。** `K_0 = 3`、`n = 150`、`d = 15`、`w = 0.5`、
   `sigma_y = 0.1`、Gaussian-Y の 1 点、7 つの family 構成のみである。
4. **per-column family の identifiability は証明していない。** `K_0` は Y 側だけで固定しており、
   X の family 構成から `K` が識別できるかは**本実験の対象外**である。
5. **per-column を正式手法として既成事実化しない。** `DualExpFamLSMPerColumnConsistent` は
   prototype であり、root `CLAUDE.md` §3 により**本文採用不可**である。
6. **Y 側寄与が config でわずかに動く理由は切り分けていない**（§8.4）。
7. **absolute score を canonical core の結果と同じ表・図に並べてはならない**
   （criterion 定数オフセット +20538.2762、§4.2）。
8. **列割り当てはブロック配置の 1 通りのみ**である。列の並べ替えやアンバランスな比率
   （例 14:1）は検証していない。
9. **信頼区間・検定・検出力・bootstrap は一切計算していない。** 各 config 10 trial の記述値である。
10. **`d` を動かしていない。** §12 の理由により、これが最も重要な未検証軸である。

---

## 12. 次に何を調べるべきか

§8 の機構から、**検証すべき次の軸が定量的に導かれる**。

現在の決定は次の不等式で決まっている。

```
|Δ D_K^{(X)}| + |Δ D_K^{(Y)}|  <  Δ P_Z + Δ P_theta
      ~28              ~137            425.68     60.13
```

X の family 構成が K 選択を動かすには、`Δ D_K^{(X)}` が罰則と同じオーダーになる必要がある。
現状は約 28 対 486、すなわち **17 倍の開き**がある。したがって次の候補は:

1. **`d` の掃引（最優先）。** X が供給する情報は `n d` に比例するので、
   `d` を 15 → 60, 150 と増やせば `Δ D_K^{(X)}` は原理的に増える。
   `d` が `K_0` 選択に効き始める領域があるかを直接測れる。
   **本実験で `d` を固定したのは、family 構成の効果を単独で見るためである。**
2. **Y チャネルを弱める。** `sigma_y` を上げる（今回 0.1 固定）と `Δ D_K^{(Y)}` が縮み、
   相対的に X の寄与が増える。canonical core でも `sigma_y` は未検証軸として残っている。
3. **極端にアンバランスな混在**（例 Poisson 14 列 : Bernoulli 1 列）や列の並べ替え。
4. **`P_Z` の理論的地位。** 本実験は core の所見
   「選択を決めているのは `O(nK)` の隠れ罰則」が per-column でもそのまま成立することを示した。
   この項が complete-data 基準として正当なのか、`scale_Z` への依存が許容されるのかは
   **未解決のまま**であり、修論で扱うかは Human Gate である。

**先生の問い「per-column でも同じ K の選び方でよいか」に対する現時点の答え:**

> 明示した設定では **挙動は変わらなかった**。ただしそれは「per-column が安全だから」ではなく、
> **現行 criterion では X チャネルが K 選択にほとんど寄与していないから**である。
> `d` を増やす、あるいは Y を弱めると結論が変わりうるので、
> **「per-column でも同じ選び方でよい」と一般化することはできない。**

---

## 付録: artifacts と再現性

**結果 CSV**（`expfam/results/percolumn_mixed_x_k_selection_20260907/`、全 23 本）:
`fit_results.csv`（630 行）, `fit_results_pilot.csv`, `fit_results_full.csv`,
`selection_by_trial.csv`, `ties.csv`（空）, `aggregate_recovery.csv`,
`selected_k_distribution.csv`, `score_by_k.csv`, `score_decomposition.csv`,
`mixed_vs_homogeneous_comparison.csv`, `diagnostic_counterfactual_selection.csv`,
`diagnostics.csv`（+ `_pilot` / `_full`）, `seed_provenance.csv`（+ `_pilot` / `_full`）,
`family_config.csv`, `config_gate.csv`, `audit_results.csv`,
`audit_percolumn_parity.csv`, `parity_by_k.csv`, `run_manifest.json`

**図**（`figures/percolumn_mixed_x_k_selection_20260907/`）:
`recovery_by_family_config.png`, `selected_K_by_family_config.png`,
`score_curve_by_family_config.png`, `score_decomposition_mixed_BGP.png`,
`prior_counterfactual_by_family_config.png`

**integrity:** config gate 10/10 PASS、parity audit 12 PASS / 0 FAIL / 2 INFO、
post-run audit **23/23 PASS**。630 fits / 失敗 0 / 非有限 0 / Poisson overflow 0 /
retry 0 / 同点 0 / seed collision 0 / criterion 代数の独立再構成 `2.910e-11`。

**環境:** Python 3.13.14 / numpy 2.3.5 / scipy 1.16.3 / pandas 2.3.3。
総計算時間 **2.76 h**（630 fits、median 15.69 s/fit）。

**コマンド:**

```
python tools/research_audit/audit_percolumn_parity.py
python tools/research_audit/run_percolumn_mixed_x_k_selection.py --gate --manifest
python tools/research_audit/run_percolumn_mixed_x_k_selection.py --pilot
python tools/research_audit/run_percolumn_mixed_x_k_selection.py --full
python tools/research_audit/run_percolumn_mixed_x_k_selection.py --merge
python tools/research_audit/audit_percolumn_mixed_experiment.py
python tools/research_audit/build_percolumn_mixed_report.py
```
