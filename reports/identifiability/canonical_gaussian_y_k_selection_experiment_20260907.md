# Canonical Gaussian-Y における K-selection 実験 — 理論的に `K_0 = K_GEN` が保証された条件での confirmatory evaluation

**作成日:** 2026-09-07
**種別:** confirmatory experiment（事前登録的に設定を凍結してから実行）
**branch:** `experiment/canonical-gaussian-y-k-recovery-20260907`（isolated git worktree、main・docs branch は無変更）
**lineage:** fixed 系列（`DualExpFamLSMFixed`、Y 側 0.5 なし。root `CLAUDE.md` §3、KI-002）
**criterion:** `calc_bic_dual` — **Q-based complete-data criterion**。**Schwarz BIC ではない**（KI-010）。
関数名・CSV 列名の歴史的呼称 `BIC` は変更していないが、本文では一貫して **Q-based criterion** と呼ぶ。

---

## 1. Executive summary

理論的に `K_0 = K_GEN` が保証された canonical Gaussian-Y 生成条件で、現行 Q-based criterion が
何を選ぶかを**改変・後付け調整なしに**評価した。**450 fits、失敗 0、final integrity audit 18/18 PASS。**

**結果は CASE A（高率回収）である。**

| `K_0` | `n` | trials | `K_hat = K_0` | recovery rate |
|---|---|---|---|---|
| 1 | 150 | 10 | 10 | **1.00** |
| 3 | 50 | 10 | 10 | **1.00** |
| 3 | 150 | 10 | 10 | **1.00** |
| 3 | 300 | 10 | 10 | **1.00** |
| 5 | 150 | 10 | 10 | **1.00** |
| **合計** | | **50** | **50** | **1.00** |

同点は 1 件も発生していない（`ties.csv` は空）。`k = K_0` での推定値は `w_hat = 0.4967`（真値 0.5）、
`sigma_y_hat = 0.1002`（真値 0.1）。

**しかし本実験の価値は「50/50 当たった」ことではない。** 重要なのは **なぜ当たったか**である。

> **中心的所見:** 現行 Q-based criterion が `K_0` を選ぶのは、明示的パラメータ罰則 `p_hat log n` の
> はたらきでは**ない**。選択を決めているのは、`Q_strict` が `E[log p(Z)]` を含むことと
> estimator 側 `scale_Z` の相互作用から生じる **決定論的な `O(nK)` の隠れ罰則**である。
>
> ```
> C_Q(K) = D_K + P_Z(K) + P_theta(K)
> P_Z(K) = -2 E[log p(Z)] = n K (log 2*pi + 1)      <- データを一切含まない
> ```
>
> この恒等式は 450 fits すべてで残差 `4.5e-13` 以下で成立する。
> `P_Z` を外す counterfactual では **50/50 すべてが選択を変え、47/50 が候補上限 `K = 9` を選ぶ**。

副次的に、事前に立てた予測が **外れた**ことも記録する。私は「fit gain は dyad 数 `O(n^2)` に比例するので
`n` を上げると隠れ罰則 `O(n)` を追い越して過大選択に向かう」と予測したが、実測は
fit gain が `n^0.76`、隠れ罰則が厳密に `n^1.0` で、**罰則のほうが速く伸びた**。
margin は `n` とともに `118.6 -> 317.5 -> 634.5` と広がった。予測は棄却された。

**時間制約により optional signal-sensitivity arm（`w in {0.25, 1.0}`）は実行していない**（§14）。

---

## 2. Why this experiment was necessary

先行監査で 2 つのことが確定していた。

1. `reports/research_scope/fixed_generator_kgen_k0_structural_audit_20260907.md`
   — historical fixed generator では `Z` が列 sample z-score されるため、Scenario B・C で
   `p_* notin M_K`（∀K）となり **`K_0` がそもそも存在しない**。
   したがって historical の `30/30` は `K_hat = K_GEN` の一致であって `K_hat = K_0` の一致ではない。
2. `reports/identifiability/gaussian_y_k_identifiability_adversarial_audit_20260907.md`
   — canonical Gaussian-Y（`z_i ~ iid N(0,I_K)`、`w != 0`）では
   **単一 dyad の周辺分布が `K` を一意に決める**（反例探索で棄却されず、独立な 2 証明あり）。

この 2 つを合わせると、**「`K_0 = K_GEN` を理論保証したうえで criterion を評価する」ことが初めて可能になる。**
本実験はその最初の実行である。「K=3 を選ばせること」は目的ではない。

---

## 3. Definition of true K

```
M_K = { p_{theta,K}(X,Y) = int p(Z) p(X|Z) p(Y|Z) dZ : theta in Theta_K },  p(Z): z_i ~ iid N(0, I_K)
K_0 = min { K : p_* in M_K }
```

| 記号 | 意味 | 本実験での値 |
|---|---|---|
| `K_GEN` | generator が `Z` を何列作ったか（手続き上の数） | 1 / 3 / 5 |
| `K_0` | 観測分布の最小潜在次元 | **`= K_GEN`（§5 で保証）** |
| `K_hat` | criterion が実際に選んだ次元 | 実験結果 |

---

## 4. Gaussian-Y identifiability theorem used

`gaussian_y_k_identifiability_adversarial_audit_20260907.md` §16 の定理。

```
S_K = z_i^T z_j,  M_S(t) = (1-t^2)^{-K/2},  kappa_{2m}(S_K) = (2m-1)! K
Y_ij = w0 + w S_K + eps,  eps ~ N(0, sigma_y^2)
  => kappa_4(Y) = 6 K w^4,   kappa_6(Y) = 120 K w^6
  => w^2 = kappa_6 / (20 kappa_4),   K = (200/3) kappa_4^3 / kappa_6^2
phi_Y(t) = exp(i w0 t) (1 + w^2 t^2)^{-K/2} exp(-sigma_y^2 t^2 / 2)
```

`w != 0` なら、`phi_Y` の `t -> infty` 漸近（`sigma^2` は `t^2` の項、`K` は `log t` の項として分離）から
`(w0, K, w^2, sigma_y^2)` が一意に決まる。異なる `K` は同一の population marginal law を作れない。

**適用範囲の限定:** これは population identifiability であり、
finite-sample recovery・criterion の妥当性・selection consistency のいずれも保証しない。
本実験が調べるのはまさにその後半である。

---

## 5. Why `K_0 = K_GEN` is guaranteed here

joint が等しければ単一 dyad の周辺も等しい。canonical joint では `X ⊥ Y | Z` であり、
`Y_ij` の周辺は `F`・`family_x`・`d` を一切含まず §4 の dyad model そのものになる。
したがって `w = 0.5 != 0` の本設定では `{K : p_* in M_K} = {K_GEN}` は単元集合であり、

```
K_0 = K_GEN        （理論的 ground truth）
```

`F` の階数も X 側の識別可能性も**不要**である（同監査 §12.1）。

---

## 6. Why historical fixed is not the same experiment

historical fixed Scenario C と本実験の canonical arm は、**公称パラメータが完全に一致する**。
一次コードで確認した（`data_generator_expfam.py` `_Y_DEFAULTS["gaussian"]` と関数既定値）。

| 項目 | historical fixed Scen.C | canonical arm |
|---|---|---|
| `w0` / `w` / `sigma_y` | 0.0 / 0.5 / 0.1 | 0.0 / 0.5 / 0.1 |
| `var_f` / `uniq` | 5.0 / 0.1 | 5.0 / 0.1 |
| `family_x` / `family_y` | bernoulli / gaussian | bernoulli / gaussian |
| `n` / `d` / `K_GEN` | 150 / 15 / 3 | 150 / 15 / 3 |
| X 後処理 | なし（`clip(±500)` は不活性、KI-021 G6） | なし |
| **Z 後処理** | **列 sample z-score**（`data_generator_expfam.py:283`） | **なし** |

**唯一の実質的差分は 1 行である。** これは意図せず得られた、極めてきれいな対照実験になっている。

その 1 行の帰結は大きい。historical 側では `z_i ~ iid N(0,I_K)` が壊れるため §4 の定理が適用できず、
さらに `p_* notin M_K`（∀K）となり `K_0` が存在しない。
**したがって historical の `30/30` を本実験の結果と同じ意味で読んではならない。** §18 参照。

---

## 7. Exact generator

`expfam/src/experimental/data_generator_canonical_gaussian_y_k.py`（`generator_id = canonical-gaussian-y-k-v1`）

```
rng = np.random.default_rng(seed)
Z   = rng.normal(0.0, 1.0, size=(n, K_GEN))          # NO post-processing of any kind
F   = rng.normal(0.0, sqrt(5.0), size=(d, K_GEN))
F[j,:] <- F[j,:]/||F[j,:]|| * sqrt(1 - 0.1)          # positive diagonal => rank preserved
eta_x = Z @ F.T ;  X = rng.binomial(1, expit(eta_x)) # exact sigmoid, no clipping, no normalisation
eta_y = w0 + w * (Z @ Z.T)
Y[i<j] = eta_y[i<j] + rng.normal(0, sigma_y, n_pairs)
Y = Y + Y.T ; fill_diagonal(Y, 0)                    # no normalisation
```

`w0 = 0.0`, `w = 0.5`, `sigma_y = 0.1`（**すべて結果を見る前に凍結し、一度も変更していない**）。

`F` の行正規化は canonical model からの逸脱ではない。`F` は `Theta_K` の自由パラメータであり、
特定の値を選んでいるだけである（正対角を左から掛けるので `rank(F)` は保存される）。
historical と同じ規約にしたのは X 側の信号スケールを揃えるためである。

**generator tests: 20/20 PASS**（`tools/research_audit/test_canonical_gaussian_y_generator.py`）。
中核は T1: `generator["Z"]` が `np.random.default_rng(seed).normal(0,1,(n,K))` と **bit-identical**
であること。これは分布の検定ではなく **code-path の検定**であり、標準化が 1 行でも入れば必ず落ちる。

**THEORY GATE G1-G10: 10/10 PASS**（`theory_gate.csv`）。seed collision: 110 cells / 110 unique seeds / **0**。

---

## 8. Exact estimator / criterion

**generator 以外は historical fixed official Exp1 K-selection と同一条件にした。**
「generator だけを canonical にしたとき現行 criterion がどう振る舞うか」を isolate するためである。
すべて一次コードから確認し `run_manifest.json` に記録した。

| 項目 | 値 | 出所 |
|---|---|---|
| model | `DualExpFamLSMFixed` | `expfam/src/model_dual_expfam_fixed.py:33` |
| `d` / `L` / `num_iter` | 15 / 5 / 8 | historical Exp1 と同一 |
| candidate `K` | 1,…,9 | 同上 |
| Newton | `max_iter=10`, `alpha = 0.5 / 2^retry`, retry ≤ 2 | `reproduction/src/model.py:360` |
| Adam (`calc_w0`/`calc_w`) | `max_iter=50`, `alpha=0.01`, `beta=(0.9,0.999)` | `model_expfam.py:149,180` |
| init | `w0 = mean(Y_upper)`, `w = 0.5`, `sigma_y = max(sd,0.01)`, `F *= 0.2` | historical runner 逐語 |
| `var_z` | **1.0 固定**（M-step は存在しない） | `reproduction/src/model.py:98-99` |
| criterion | `C_Q(K) = -2 Q_strict + p_hat(K) log n` | `utils_expfam.py:386-404` |
| `p_hat(K)` | `K*d - K(K-1)/2 + 0 + 1` | 同上（Bernoulli-X なので `Sigma_X` 項 0、Gaussian-Y なので +1） |

`Q_strict = E_sample[ log p(Z) + log p(X|Z) + log p(Y|Z) ]`。
本設定は Poisson を含まないので階乗補正は 0 であり `Q_strict = Q_dual`。

**注意（結果解釈に影響する実装事実）:** 初期化器は Gaussian-Y のとき `w` を `0.5` に置く。
本実験の真値も `w = 0.5` である。これは**私が選んだのではなく historical initialiser の定数**であり、
比較可能性のために変更しなかった。ただし「初期値が真値に一致している」ことは
finite-sample の容易さに寄与しうるので、**限定条件として明記する**（§20）。

---

## 9. Pilot integrity gate

pilot = `K_0=3, n=150`, trial 0-2, `K=1..9` の **27 fits**。結果は捨てず full の最初の 3 trial として再利用した。

**gate は integrity のみ。** `selected K == 3`・recovery 率・score curve の形は **gate に使っていない**
（それらは研究結果であるため）。

**PILOT INTEGRITY GATE: 18/18 PASS**（`audit_results_pilot.csv`）。
P2 非有限 0、P3 dataset hash 一致、P5 再構成残差 `1.455e-11 <= 1e-8`、P6 行数一致、P7 `Z` 標準化なし。

---

## 10. Formal experiment design

pilot の実測 runtime **のみ**を根拠に grid を決定した（§0 の禁止事項に従い、選択結果は参照していない）。
`FALLBACK_CELLS` はランナー実装時、すなわち **pilot 実行前**にソース内で凍結されている。

| 測定 | 値 |
|---|---|
| pilot median | 11.44 s/fit (`n=150`) |
| timing probe（seed 1999999、実験 seed 空間外） | `n=50`: 2.88–3.50 s / `n=300`: 22.2–59.6 s |
| PREFERRED FULL (810 fits) の残り見積り | **231 min** |
| CORE FALLBACK (450 fits) の残り見積り | **108 min** |
| 決定時点の実験枠残り | 245 min |

FULL は枠の 94% を消費し安全余裕がなく optional arm も不可能になるため、**CORE FALLBACK を採用**した。
実測は 423 fits / 104.3 min で見積りと整合した（pilot 27 + full 423 = **450 fits**）。

**採用 grid:** `K_0 in {1,3,5} × n=150`（各 10 trial）＋ `K_0=3 × n in {50,300}`（各 10 trial）
= **50 datasets / 450 fits**、candidate `K = 1..9`。

**seed policy（事前確定）:** `data_seed = 900000 + K_GEN*10000 + n*10 + trial`、
`model_seed = data_seed + 50`（`k_est` に依存しない）。同一 dataset に対する候補 K 比較では
`data_seed` は完全に同一で、実行時に dataset hash 一致を assert している。

---

## 11. Primary recovery results

`aggregate_recovery.csv` / `selection_by_trial.csv`

| `K_0` | `n` | trials | `K_hat=K_0` | rate | mean gap to 2nd best |
|---|---|---|---|---|---|
| 1 | 150 | 10 | 10 | 1.00 | 344.3 |
| 3 | 50 | 10 | 10 | 1.00 | 118.6 |
| 3 | 150 | 10 | 10 | 1.00 | 317.5 |
| 3 | 300 | 10 | 10 | 1.00 | 634.5 |
| 5 | 150 | 10 | 10 | 1.00 | 309.3 |

- 全 450 fits 成功、`nan_occurred` 0、internal retry は全件 0。
- **同点 0 件**（`ties.csv` は空）。tie rule（最小 K）は一度も発動していない。
- `k = K_0` での推定: `w_hat = 0.4967`（真 0.5）、`sigma_y_hat = 0.1002`（真 0.1）。

図: `recovery_rate_by_K0.png`, `selected_K_distribution.png`, `score_curve_K0_3_n150.png`

---

## 12. `K_0`-by-`K_hat` confusion

`confusion_matrix.csv`（core arm, `w=0.5`、全 `n` を合算）

| `K_0` \ `K_hat` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **10** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | **30** | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | **10** | 0 | 0 | 0 | 0 |

対角外は 0。over-selection も under-selection も観測されなかった。

---

## 13. Sample-size results

`K_0 = 3` の `n` sweep（各 10 trial）。図: `sample_size_recovery_K0_3.png`

| `n` | recovery | mean `K_hat` | gap to 2nd best |
|---|---|---|---|
| 50 | 10/10 | 3.0 | 118.6 |
| 150 | 10/10 | 3.0 | 317.5 |
| 300 | 10/10 | 3.0 | 634.5 |

**`n` を増やすと margin が単調に広がった。** ただし 3 点のみの記述であり、
consistency も収束も主張しない（§20）。

---

## 14. Signal sensitivity

**未実施。** `[NOT EXECUTED]`

仕様 §15 は optional arm の実行条件として「core 終了後 report 作成前に **60 分以上の余裕**があり、
core integrity が完全 PASS していること」を課している。
full grid は 06:48 JST に完了したが、監視タスクの遅延により**観測したのは 09:02 JST** であり、
5 時間枠の残りは約 48 分で 60 分要件を下回った。
**時間のみを根拠に不実施と判断した。結果は一切参照していない。**

arm はランナー内に完全に仕様化・seed 固定済みで（`--sensitivity`、`w in {0.25, 1.0}`、
`K_0=3`, `n=150`, 10 trials, `K=1..9`, 180 fits, 所要見積り約 34 分）、
**設定を一切変えずにそのまま実行できる。** これが次の 1 手である（§21）。

---

## 15. Q-based score decomposition

criterion は次のように**厳密に**分解される（450 fits すべてで残差 `<= 5.821e-11`、audit P5d）。

```
C_Q(K) = D_K + P_Z(K) + P_theta(K)

D_K       = -2 E[ log p(X|Z) + log p(Y|Z) ]      fit-only
P_Z(K)    = -2 E[ log p(Z) ]                      direct latent-prior score
P_theta(K)= p_hat(K) log n                        explicit parameter penalty
```

**そして `P_Z(K)` は決定論的である。** `[DERIVED]` + `[NUMERIC]`

`_lnpZ` は `-(nK/2) log(2 pi var_z) - (1/(2 var_z)) ||Z_l||^2` であり `var_z = 1`。
estimator の `scale_Z` は `(n,K,L)` サンプル全体で `mean(Z^2) = 1` に正規化するので
`(1/L) sum_l ||Z_l||^2 = nK` が厳密に成立する。したがって

```
P_Z(K) = n K ( log 2*pi + 1 ) = 2.8379 * n * K
```

実測との最大乖離は **`4.547e-13`**。この項は **データを一切含まない**。

`K_0=3, n=150` での平均寄与（`aggregate_score_by_k.csv`）:

| `K` | `D_K` | `P_Z(K)` | `P_theta(K)` | `C_Q(K)` |
|---|---|---|---|---|
| 1 | 2695.5 | 425.7 | 80.2 | 3201.4 |
| 2 | −6712.5 | 851.4 | 150.3 | −5710.8 |
| **3** | −37579.4 | 1277.0 | 215.5 | **−36086.9** |
| 4 | −37748.8 | 1702.7 | 275.6 | −35770.5 |
| 9 | −38440.4 | 3831.1 | 501.1 | −34108.2 |

**`D_K` は `K` について単調減少である。** `K = 3` に極小を持たない。
`C_Q` に極小を作っているのは `P_Z + P_theta` であり、その大部分は `P_Z` である。

図: `score_decomposition_K0_3_n150.png`

---

## 16. What drives selection?

`penalty_vs_fitgain_by_n.csv` — 真値の隣 `K_0 -> K_0 + 1` での限界収支。

| `K_0` | `n` | `Δ D_K` | `Δ P_Z` | `Δ P_theta` | `Δ C_Q` | 停止 | implicit/explicit |
|---|---|---|---|---|---|---|---|
| 1 | 150 | −151.5 | 425.7 | 70.2 | **+344.3** | Yes | **6.1×** |
| 3 | 50 | −70.2 | 141.9 | 46.9 | **+118.6** | Yes | **3.0×** |
| 3 | 150 | −168.3 | 425.7 | 60.1 | **+317.5** | Yes | **7.1×** |
| 3 | 300 | −285.3 | 851.4 | 68.5 | **+634.5** | Yes | **12.4×** |
| 5 | 150 | −166.5 | 425.7 | 50.1 | **+309.3** | Yes | **8.5×** |

**観察 1 — 隠れ罰則が支配的である。**
`P_Z` の 1 次元あたり増分は `P_theta` の **3.0〜12.4 倍**である。
`P_theta` 単独（35〜70）は fit gain（70〜285）を止められない。
**`K_0` で止まっているのはほぼ `P_Z` の力である。**

**観察 2 — 罰則の階数が Schwarz 型と違う。**

```
P_Z     の傾き = n (log 2*pi + 1)  = O(n)         <- 実測 425.68 -> 851.36 でちょうど 2 倍（n^1.00）
P_theta の傾き = (d-K+1) log n     = O(log n)
```

現行 criterion の実効罰則は **`O(nK)`** であって `O(p log n)` ではない。
これは complete-data / ICL 型基準が潜在変数を「データ」として採点することの帰結であり、
`scale_Z` がそのスケールを固定したため**完全に決定論的**になっている。
KI-010 が「Schwarz BIC と呼ばない」としてきた理由に、**定量的な裏付けが付いた**。

**観察 3 — 事前予測は棄却された。** `[PREDICTION REJECTED]`

実験前に私は「Y は `n(n-1)/2` dyad を供給するので fit gain は `O(n^2)`、
隠れ罰則は `O(n)` だから `n` を上げると過大選択に向かう（交差点は `n ≈ 380`）」と予測した。
実測は

```
fit gain :  168.3 -> 285.3   （n 150 -> 300）  = 1.70 倍  ≈ n^0.76
P_Z      :  425.7 -> 851.4                     = 2.00 倍  =  n^1.00
```

**罰則のほうが速く伸びた。** margin は `118.6 -> 317.5 -> 634.5` と広がり、交差点は存在しない。
**予測は棄却する。** fit gain が `n^2` に比例しない理由は本実験の設計では特定できていない
（過剰次元での `w_hat` 収縮、`L=5` の MC 近似、`num_iter=8` の有限反復が候補だが**切り分けていない**）。

---

## 17. Counterfactual direct-prior diagnostic

**診断のみ。新しい criterion の提案ではない。現行 criterion は一切変更していない。**
（`diagnostic_counterfactual_selection.csv`）

`score_without_direct_prior(K) = D_K + P_theta(K)`（`P_Z` を除いた反事実スコア）で選び直すと:

| 指標 | 値 |
|---|---|
| 選択が変わった dataset | **50 / 50（100%）** |
| 反事実の `K_hat` = 9（候補上限） | **47 / 50** |
| 同 = 8 | 2 / 50（すべて `n=50`） |
| 同 = 7 | 1 / 50（`n=50`） |

**`P_Z` を外すと、現行 criterion は `K_0` を 1 件も選ばず、ほぼ全件で候補上限に張り付く。**

これは「`P_Z` を外すべきだ」という主張では**ない**。
言えるのは、**現行 criterion の `K_0` 回収は明示罰則ではなく、
潜在変数を complete data として採点する項に決定的に依存している**という事実だけである。
この項の理論的地位（罰則としての正当性、`n` 依存の階数、`scale_Z` への依存）は
本実験では評価していない。

---

## 18. Comparison with historical fixed (descriptive only)

`historical_vs_canonical_descriptive.csv`

| arm | generator | Z 後処理 | trials | `K_hat=3` |
|---|---|---|---|---|
| historical fixed Scen.C | `data_generator_expfam.generate_dual_data` | **列 sample z-score** | 10 | 10 |
| canonical (本実験) | `data_generator_canonical_gaussian_y_k` | **なし** | 10 | 10 |

### **NOT DIRECTLY COMPARABLE**

同じ「10/10」でも意味が違う。

- historical は **`K_hat = K_GEN`** の一致である。その generator では `K_0` が存在しない
  （`fixed_generator_kgen_k0_structural_audit_20260907.md` §12.3）ので、
  **true-K recovery の証拠にはならない。**
- 本実験は **`K_hat = K_0`** の一致であり、`K_0 = K_GEN` は §5 で理論保証されている。

両者の数値を同じ表・図に並べて「一致した」と読ませてはならない（KI-002 / KI-003 の趣旨）。
上表は **1 行の generator 差分がもたらす解釈の差**を示すためだけに置いている。

---

## 19. What is established

1. canonical Gaussian-Y・`w=0.5`・`n in {50,150,300}`・`K_0 in {1,3,5}` の**明示した有限標本設定**において、
   現行 Q-based criterion は候補 `K=1..9` から `K_0` を **50/50 trial** で選択した。同点 0。
2. 現行 criterion は `C_Q(K) = D_K + P_Z(K) + P_theta(K)` と**厳密に**分解される（残差 `<= 5.8e-11`）。
3. `P_Z(K) = nK(log 2*pi + 1)` は **決定論的**であり（残差 `<= 4.5e-13`）、データ情報を持たない。
   これは `Q_strict` が `E[log p(Z)]` を含むことと `scale_Z` の相互作用の帰結である。
4. `P_Z` の 1 次元あたり傾きは `P_theta` の **3.0〜12.4 倍**であり、実効罰則の階数は
   `O(nK)` であって `O(p log n)` ではない。
5. `P_Z` を除く反事実では **50/50 が選択を変え、47/50 が候補上限 `K=9`** を選ぶ。
6. `n` を 50→300 に増やすと、fit gain は `n^0.76`、`P_Z` は `n^1.00` で伸び、
   **margin は広がった**（118.6 → 317.5 → 634.5）。
7. `k = K_0` において `w_hat = 0.4967`（真 0.5）、`sigma_y_hat = 0.1002`（真 0.1）。
8. integrity: **450 fits / 失敗 0 / 非有限 0 / retry 0 / theory gate 10/10 / final audit 18/18 PASS。**

---

## 20. What is not established

1. **model-selection consistency ではない。** `n` は 3 点のみ。`n -> infinity` の主張は一切していない（既存 U6）。
2. **一般的な true-K recovery ではない。** 検証したのは `K_0 in {1,3,5}`、`w = 0.5`、`sigma_y = 0.1`、
   `d = 15`、Bernoulli-X / Gaussian-Y の 1 点だけである。
3. **Schwarz BIC の妥当性を示すものではない**（KI-010）。むしろ §16 は実効罰則が Schwarz 型と
   異なる階数を持つことを示している。
4. **他 family へ移送できない。** Bernoulli-Y（U2）・Poisson-Y（U4）では `K_0` の識別可能性自体が未解決。
5. **`w` の初期値が真値と一致している**（§8）。この一致が finite-sample の容易さに
   どれだけ寄与したかは**切り分けていない**。signal-sensitivity arm（未実施）が最初の検証になる。
6. **fit gain が `n^2` にならない理由は未同定**（§16 観察 3）。
7. **信頼区間・検定・検出力・bootstrap は一切計算していない。** 各セル 10 trial の記述値である。
8. `P_Z` を罰則として正当化も否定もしていない。§17 は診断であり criterion 提案ではない。
9. **prototype ではないが本文採用可否は未判断。** 昇格は Human Gate。

---

## 21. Questions to bring to professor

1. 真の潜在次元を「観測分布から一意に識別できる `K_0`」として定義する方針でよいか。
   これを採ると、人工データの `K_TRUE` を自動的に真値扱いできなくなる（historical 全実験の読み替えが要る）。
2. Gaussian-Y のように識別可能性を理論保証できる条件を **criterion の評価基準**として使う方向でよいか。
   それとも Bernoulli-Y など識別可能性が未解決の family まで先に広げるべきか。
3. 現行 Q-based criterion の `K_0` 回収が、明示罰則ではなく
   **`-2E[log p(Z)]` という `O(nK)` の隠れ罰則**に依存していることが分かった（§16・§17）。
   この項の理論的地位（complete-data 基準として正しいのか、`scale_Z` への依存は許容されるのか）を
   修論で正面から扱うべきか、それとも「現行基準の記述」に留めるべきか。
4. 次に実行すべきは (a) 未実施の signal-sensitivity arm（`w in {0.25,1.0}`、34 分）か、
   (b) `n` をさらに広げた sweep か、(c) 他 family への拡張か。
5. historical fixed の `30/30` は今後どう記述すべきか。
   本実験の `50/50` と**同じ表に並べない**方針でよいか（§18）。

---

## 22. Reproducibility manifest

`expfam/results/canonical_gaussian_y_k_selection_20260907/run_manifest.json`
（branch / HEAD / package versions / 凍結設定 / seed policy / seed collision check /
runtime decision / sensitivity 不実施理由 / 全 CSV の SHA256 を記録）

**環境:** Python 3.13.14 / numpy 2.3.5 / scipy 1.16.3 / pandas 2.3.3 / matplotlib 3.10.7
Windows 11, Intel64 Family 6 Model 158。総計算時間 **1.82 h**（450 fits）。

**コマンド:**

```
python tools/research_audit/test_canonical_gaussian_y_generator.py
python tools/research_audit/run_canonical_gaussian_y_k_selection.py --gate --manifest
python tools/research_audit/run_canonical_gaussian_y_k_selection.py --pilot
python tools/research_audit/audit_canonical_gaussian_y_k_selection.py --mode pilot
python tools/research_audit/run_canonical_gaussian_y_k_selection.py --full --fallback
python tools/research_audit/run_canonical_gaussian_y_k_selection.py --merge
python tools/research_audit/audit_canonical_gaussian_y_k_selection.py --mode final
python tools/research_audit/build_canonical_gaussian_y_k_report.py
# 未実行（時間要件未達）: --sensitivity
```

**成果物（`expfam/results/canonical_gaussian_y_k_selection_20260907/`）:**
`fit_results.csv`（450 行）, `fit_results_pilot.csv`, `fit_results_full.csv`,
`selection_by_trial.csv`, `ties.csv`(空), `generator_diagnostics.csv`,
`aggregate_recovery.csv`, `confusion_matrix.csv`, `aggregate_score_by_k.csv`,
`penalty_vs_fitgain_by_n.csv`, `diagnostic_counterfactual_selection.csv`,
`historical_vs_canonical_descriptive.csv`, `theory_gate.csv`,
`audit_results_pilot.csv`, `audit_results_final.csv`, `run_manifest.json`

**図（`figures/canonical_gaussian_y_k_selection_20260907/`）:**
`recovery_rate_by_K0.png`, `selected_K_distribution.png`, `score_curve_K0_3_n150.png`,
`score_decomposition_K0_3_n150.png`, `sample_size_recovery_K0_3.png`
（`signal_sensitivity_K0_3.png` は arm 未実施のため無し）
