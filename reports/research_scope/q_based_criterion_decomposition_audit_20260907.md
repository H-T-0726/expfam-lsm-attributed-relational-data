# current Q-based criterion の完全分解監査

**作成日:** 2026-09-07
**改訂:** 2026-09-07（理論解釈の再監査。**実装監査・一次 CSV 監査の結果は変更していない。**
変更箇所は `[CORRECTED-20260907]` / `[ADDED-20260907]` で明示する。改訂一覧は付録 C）
**種別:** READ-ONLY 監査。**コード変更なし・実験実行なし・Issue/PR 作成なし・既存 CSV / figure / historical report の変更なし。**
**新規作成物:** 本ファイル 1 件のみ。
**ブランチ:** `docs/notion-k-selection-story-20260906`（作業開始時 `git status` = `?? reports/research_scope/` のみ）

**本監査における「計算」とは、既にコミット済みの一次 CSV の数値を読み込んで恒等式的に分解したことだけを指す。**
モデルの再フィットは一切行っていない。

---

## 1. Executive summary

### 1.1 確定した対象

現在 K 選択に使われている量は、コード上ではただ 1 本の式である [CODE]。

```
C_Q(K) = −2 · Q_strict(K) + p̂(K) · ln n            utils_expfam.py:403
```

そして `Q_strict` を分解すると、**K 依存項は 5 つしかない**。

```
C_Q(K) = −2·[ lnpZ_avg + lnpX_avg + lnpY_avg + corr ] + p̂(K)·ln n
                ↑           ↑          ↑        ↑            ↑
             (3) Z prior  (1) X fit  (2) Y fit (5) 定数   (4) 明示罰則
```

### 1.2 最重要の 3 点

**(A) Z prior 項はデータにも θ にも依存しない決定論的な値に退化している。**
`scale_Z`（`reproduction/src/model.py:499-503`）が `Z_samples` 配列全体の平均二乗を 1 に正規化し、
`var_z` は恒久的に `1.0`（`model.py:99`）であるため [CODE]:

```
lnpZ_avg = (1/L) Σ_l lnpZ(Z^(l)) = −(nK/2)·(1 + ln 2π)      （L 平均について厳密）
−2 · lnpZ_avg = n·K·(1 + ln 2π) = 2.8378770664 · n · K
```

n = 150 では **K を 1 増やすごとにちょうど +425.6816**。これは fit 情報を一切含まない、
**n に比例する固定次元罰則として機能している** [DERIVED]。

**(B) 明示的 parameter-count 罰則の K 依存部分は `F` 由来の 1 項だけであり、(A) より 1 桁小さい。**

```
Δp̂(K→K+1) · ln n = (d − K) · ln n = 12 · ln 150 = 60.13     （n=150, d=15, K=3→4）
```

すなわち **425.68 : 60.13 ≈ 7.1 : 1** で、Z prior 項が支配的である [DERIVED]。

**(C) fixed 系列 30/30 の K=3 選択は、Z prior 項を外すと成立しない。**
一次 CSV `fixed_exp1_bic_k1to9_summary.csv`（270 行）から本監査が trial 単位で恒等分解した結果 [ARTIFACT + DERIVED]:

| シナリオ | `argmin BIC_impl` の trial 分布 | `argmin S_cf`（Z prior 除去）の trial 分布 |
|---|---|---|
| A (poisX/bernY) | **K=3: 10/10** | K=9: 8, K=8: 2 |
| B (gaussX/poisY) | **K=3: 10/10** | K=9: 9, K=8: 1 |
| C (bernX/gaussY) | **K=3: 10/10** | K=9: 10 |

**30/30 中 30 trial すべてで、Z prior 項を除くと argmin が候補範囲上端（K=8 または 9）に移る。**

### 1.3 「まだ言えないこと」 [CORRECTED-20260907]

- **これをもって「Q-based は間違い」「過剰罰則である」とは言えない。**
  **factorized Gaussian Laplace surrogate** `q̃(Z) = Π_i N(η_i, A_i^{-1})` を置くと、
  その微分エントロピーには `+(nK/2)(1+ln 2π)` という **`lnpZ_avg` と同係数・逆符号の O(nK) 項が代数的に現れる**（§10.2）。
  **これは `q̃` についての代数であって、current algorithm が誘導する法則のエントロピーではない**（§8.3・§10.1）。
  したがって **「426 が entropy と相殺する」とも「426 は純粋な過剰罰則である」とも本監査では確定できない** [UNRESOLVED]。
- **30/30 は K-selection consistency の証拠ではない。** n の 1 点（n=150）のみの有限標本結果である（§13）。

### 1.4 先行監査との関係（重要）

本監査の主要な理論的結論は、**`reports/k_selection_theory/k_selection_theory_audit_20260822.md`（Issue #35）で既に確立されている。**
本監査はそれを引用したのではなく、**コード・一次 CSV・原論文 PDF から独立に再導出し、一致を確認した**。
数値は §5.3・§13.2a の公開値と一致した（例: シナリオ A の `S_cf` = 20513.47, 17692.79, 15182.57, …）。

**本監査の新規増分は 3 点のみである:**

1. `S_cf` の **trial 単位** argmin（先行監査は「trial ごとの `S_cf` は計算していない」と明記。§13.2a）
2. K = 1..9 全域・3 シナリオでの**逐次差分台帳**（先行監査は K=3→4 の 1 ステップのみ）
3. 先行監査 §13.6「`Σ_i ln det A_i` はどこにも記録されていない」の**現状更新**:
   Issue #37 の pilot（2026-08-23）が `sum_log_det_A_post` を記録している（§10.3）。
   ただし **objective-consistent prototype 系列**であり、fixed 系列ではない。

---

## 2. Scope and evidence

### 2.1 対象

- **基本拡張（`DualExpFamLSM` / `DualExpFamLSMFixed`）の Q-based K 選択のみ。**
- 中心は **fixed 系列**（`model_dual_expfam_fixed.py`）。historical（旧 0.5）系列は差分がある箇所でのみ言及する。
- **clean / S1 / K_PRED / per-column / heldout は本監査の中心ではない。** 参照が必要な箇所では系列を明示する。

### 2.2 非対象

- criterion の変更提案
- 新実験
- 「Q-based を捨てるか残すか」の決定

### 2.3 Evidence labels

| ラベル | 意味 |
|---|---|
| `[CODE]` | 実コードの file path + 行番号で確認 |
| `[ARTIFACT]` | 一次 CSV の path + 値で確認 |
| `[PAPER]` | 原論文 PDF から本監査が直接抽出（§12 に抽出手順を記載） |
| `[DERIVED]` | 本監査が自分で行った数学的導出・一次 CSV の恒等分解 |
| `[INFERENCE]` | 一般論からの推測。証明ではない |
| `[UNRESOLVED]` | 証明も反証もできない |

### 2.4 一次資料の読み取り状況

| 資料 | 状況 |
|---|---|
| `expfam/src/utils_expfam.py` 他の実コード | 全文読了 [CODE] |
| `expfam/results/fixed_official/exp1_k9/*.csv` | 270 行を読み込み恒等分解 [ARTIFACT] |
| `paper/A_study_on_latent_structural_models_for_binary_rel.pdf` | 本監査が FlateDecode ストリームを展開し TJ 配列からテキストを抽出。Eq.(16)(17)(18)(26)・Appendix B を直接読了 [PAPER] |
| `paper/2.pdf` | **本監査は一切参照していない。** 書誌情報がリポジトリ内で特定されていないため |
| MATLAB `Mato Lab Program/` | 本監査では未参照（先行監査 §5.5・§7.5 の記録に委ねる） |

---

## 3. Exact current criterion from code

### 3.1 コード行との 1 対 1 対応 [CODE]

```
C_Q(K) = −2 · Q_strict + p̂ · ln(n)                      utils_expfam.py:403

p̂  = K·d − K(K−1)//2                                     utils_expfam.py:399   f_params
    + d · 1{family_x = gaussian}                          utils_expfam.py:400   sigma_x_p
    + 1 · 1{family_y = gaussian}                          utils_expfam.py:401   sigma_y_p

Q_strict = Q_dual + corr                                  utils_expfam.py:372-379

Q_dual = (1/L) Σ_{l=1}^{L} [ lnpZ(Z^(l)) + lnpX(X|Z^(l),F) + lnpY(Y|Z^(l),w0,w) ]
                                                          utils_expfam.py:345-352

lnpZ  = −(nK/2)·ln(2π·var_z) − (1/(2·var_z))·Σ (Z^(l))²   utils_expfam.py:318-321  (_lnpZ)
lnpX  = model.calc_log_likelihood_X(X, Z^(l), F)          utils_expfam.py:349
        → model_dual_expfam.py:291-334
lnpY  = model.calc_log_likelihood_Y(Y, Z^(l), w0, w)      utils_expfam.py:350
        → model_expfam.py:241-269

corr  = −Σ_{i<j} ln(y_ij!)   if family_y = 'poisson'      utils_expfam.py:375-376
        −Σ_{i,l} ln(x_il!)   if family_x = 'poisson'      utils_expfam.py:377-378
```

`var_z` は `initialize_params` で `1.0` に固定され（`reproduction/src/model.py:99`、注釈 `L.98`「varZ is fixed to 1.0 for identifiability」）、
**M-step で更新されない** [CODE]。

### 3.2 1 本の数式としての C_Q(K)

`var_z = 1` を代入し、`Z^(l)` は `scale_Z` 適用後のサンプルとする。

```
C_Q(K) =
   −2 · (1/L) Σ_l [ −(nK/2)·ln 2π − (1/2)‖Z^(l)‖_F² ]                  … (3) Z prior
   −2 · (1/L) Σ_l  ln p(X | Z^(l), F, Σ_X ; family_x)                   … (1) X fit
   −2 · (1/L) Σ_l  ln p(Y | Z^(l), w0, w, σ_Y ; family_y)               … (2) Y fit
   −2 · corr                                                            … (5) その他（K 非依存定数）
   + [ K·d − K(K−1)/2 + d·1{GX} + 1·1{GY} ] · ln n                      … (4) 明示罰則
```

### 3.3 5 分解の内訳

| # | 項 | 数式 | K 依存 |
|---|---|---|:---:|
| **(1)** | X fit | `−2·(1/L) Σ_l ln p(X\|Z^(l),F,Σ_X)` | YES |
| **(2)** | Y fit | `−2·(1/L) Σ_l ln p(Y\|Z^(l),w0,w,σ_Y)` | YES |
| **(3)** | Z prior | `n·K·(1 + ln 2π)`（scale_Z により決定論化） | YES（厳密に線形） |
| **(4)** | 明示罰則 | `[K·d − K(K−1)/2 + d·1{GX} + 1·1{GY}]·ln n` | YES（`K·d − K(K−1)/2` のみ） |
| **(5)** | その他 | Poisson の階乗補正、Gaussian-X の `−(nd/2)ln 2π` | NO |

### 3.4 定数項の扱い — 「削除している」と「実際にはコードに入っている」の区別 [CODE]

| 定数 | family | コードに含まれるか | K 依存 |
|---|---|---|:---:|
| Gaussian-X の `−0.5·ln(2π)`（要素ごと） | Gaussian-X | **含まれる**（`model_dual_expfam.py:323`） | NO |
| Gaussian-Y の `−0.5·ln(2π)`（ペアごと） | Gaussian-Y | **落とされている**（`model_expfam.py:262-264` に `ln 2π` がない）。`calc_Q_dual_strict` も補正しない | NO |
| Poisson-Y の `−Σ_{i<j} ln(y_ij!)` | Poisson-Y | `calc_Q_dual` では落とし、**`calc_Q_dual_strict:375-376` で復元** | NO |
| Poisson-X の `−Σ ln(x_il!)` | Poisson-X | 同上（`:377-378` で復元） | NO |
| Bernoulli の base measure `h(x)=1` | Bernoulli | 欠落なし | — |

**重要:** Gaussian-Y の欠落量は `0.5·ln(2π)·n(n−1)/2`。n = 150 では `0.918938533 × 11175 = 10269.14` [DERIVED]。
**K 依存ではないので argmin は動かないが、シナリオ C の BIC 絶対値は真の `−2 ln p` より 20538.3 だけ小さい。**
シナリオ C の BIC が大きな負値（−36228 @ K=3, [ARTIFACT]）である理由の一部はこれである。
（この非対称性は先行監査 §5.4 が既に指摘している。本監査はコードから独立に再確認した。）

### 3.5 fixed / historical で criterion の**関数形は同一**である [CODE]

`DualExpFamLSMFixed`（`model_dual_expfam_fixed.py`）が上書きするのは
`_calc_gradient`（`:77` の `term3 = w * (...)`）と `_calc_precision_matrix`（`:113` の `term3 = (w**2) * (...)`）の **E-step 2 メソッドだけ**である。

`calc_log_likelihood_X` / `calc_log_likelihood_Y` / `calc_Q_dual_strict` / `calc_bic_dual` は
**両系列で完全に共有されている** [CODE]。

> **したがって KI-001（0.5 係数）は criterion の式そのものには入っていない。**
> 系列差は「E-step が生成する `Z_samples` が違う」という経路でのみ criterion に影響する。

さらに `calc_log_likelihood_Y` は非対角全和に `0.5` を掛ける（`model_expfam.py:266-267`）ので
結果は `Σ_{i<j}` であり、これは**二重計上の補正**であって KI-001 の extra 1/2 とは別物である [DERIVED]。

---

## 4. Q decomposition

### 4.1 コード上の Q は何か [CODE]

| 問い | 答え |
|---|---|
| `E_{q_K(Z)}[ln p_K(Z,X,Y\|θ_K)]` か | **形式的にはその MC 近似だが、`E` と呼ぶには条件が足りない**（§8） |
| 単一 sample か | **違う。L サンプルの単純平均**（`utils_expfam.py:345-352` の `Q += …` → `:352` の `return Q / L`） |
| L samples 平均か | **そう。ただし L 個は iid ではない**（§8.3） |
| iteration 平均か | **違う** |
| best iteration か | **違う** |
| 最終 iteration か | **そう。** EM ループ終了後に 1 回だけ計算される（`run_fixed_official_exp1_bic_full.py:159`） |

**L の値:** fixed official Exp1 では `L = 5`, `num_iter = 8`（`run_fixed_official_exp1_bic_full.py:48`）[CODE]。

### 4.2 実装フロー（fixed official Exp1、`run_fixed_official_exp1_bic_full.py`）[CODE]

```
for iteration in 1..8:                                       L.121
    ── E-step ──
    for l in 1..5:                                           L.124
        model.params.update(Z=Z, F=F, sigma=sigma, w0, w)    L.125-127   ← θ^(t−1)
        Z_new = model.calc_eta_newton(X, Y, rng, alpha)      L.128-130
        Z_samples[:,:,l] = Z_new ;  Z = Z_new                L.131-132   ← 連鎖して次の l へ
    NaN/Inf ガード（Z_prev で置換）                            L.134-137
    Z_samples = model.scale_Z(Z_samples)                     L.139       ← ここで正規化
    Z = Z_samples[:,:,-1]                                    L.141
    ── M-step（scale 済み Z_samples に対して）──
    F     = model.calc_F(X, Z_samples)                       L.144
    sigma = model.calc_sigma(X, Z_samples, F)                L.145
    w0    = model.calc_w0(...) ;  w = model.calc_w(...)      L.146-147
    if family_y == 'gaussian': model.calc_sigma_y(...)       L.148-149
    （※ ここで Q は計算されない。収束監視もしない）

model.params.update({F, sigma, w0, w})                       L.156       ← 同期
Q_strict = calc_Q_dual_strict(X, Y, Z_samples, F, sigma, ...) L.159
bic, num_params = calc_bic_dual(Q_strict, k, n, d, fx, fy)   L.228
```

**確定事項** [CODE]:

1. `Q_strict` は **最終 iteration の scale_Z 適用済み `Z_samples`** で評価される。
2. その `Z_samples` を生成した E-step は **θ^(t−1)** を使っており、`Q_strict` は **θ^(t)** で評価される。**θ が一致していない。**
3. `num_iter = 8` で打ち切るので **EM の停留点ではない**。
4. fixed official Exp1 では **EM ループ内で Q を計算していない**（収束履歴を残していない）。

### 4.3 historical 系列との差 — stale Σ [CODE]

`utils_expfam.calc_Q_dual`（`:324-352`）は引数 `sigma` を**本文で一度も参照していない**。
Gaussian-X の対数尤度は `model.calc_log_likelihood_X` が `self.params["sigma"]`（`model_dual_expfam.py:315`）を読む。

| 系列 / スクリプト | `params["sigma"]` の同期 | Gaussian-X で実害 |
|---|---|---|
| `utils_expfam.run_em_dual`（historical: `exp_scenario_lib.py:82`） | **なし**（E-step 内 `:498` のみ、M-step は `:521`、Q は `:581`） | **あり**（`Σ^(T−1)` と `F^(T)` の混在） |
| `run_fixed_official_exp1_bic_full.py:156` | あり | なし |
| `run_fixed_official_exp1_bic_k9_extension.py:162` | あり | なし |

**本監査で分解した fixed 系列 K=1..9 の結果はこの問題の影響を受けない** [CODE]。
（この点は先行監査 §5.2 が既に確認済み。本監査はコードから独立に再確認した。）

### 4.4 「E」と呼んでよいか [CORRECTED-20260907]

**コードが計算した数値を `E[...]` と書いてはならない** [DERIVED]。
ただし理由は「期待値が存在しないから」ではない。**期待値の対象となる確率法則自体は存在する**（§8.3）。
正しい理由は次の 3 点である:

1. **計算されているのは期待値ではなく、1 回の実現に対する L=5 点平均である。**
   `Q_dual` は `q̄` を「サンプル生成法則が誘導する `Z^(l)` の混合周辺分布」としたとき
   `E_{q̄}[·]` の**単一実現による MC 推定量**であり、期待値そのものではない。
2. **L = 5 個は iid 標本ではない**（burn-in なしの単一連鎖。§8.3）。したがって
   通常の MC 誤差評価も適用できない。
3. **`scale_Z` 後のサンプルで評価されている。** 期待値の対象は Laplace 密度ではなく、
   その決定論的変換による押し出し法則である（§6・§8.3）。

**使ってよい表現:** 「最終 E-step の `L = 5` サンプルにわたる MC 平均」。
**使ってはならない表現:** 「事後期待値 `E_{p(Z|X,Y,θ)}[·]`」。

---

## 5. Z-prior decomposition

### 5.1 コード上の prior [CODE]

`_lnpZ`（`utils_expfam.py:315-321`）:

```python
return float(
    -(n * k / 2.0) * np.log(2.0 * np.pi * var_z)
    - (1.0 / (2.0 * var_z)) * np.sum(Z_l ** 2)
)
```

すなわち `z_i ~ N(0, var_z · I_K)`、`var_z ≡ 1.0`。ユーザー提示の `z_i ~ N(0, I_k)` と一致する [CODE]。

### 5.2 完全展開 [DERIVED]

```
ln p(Z^(l)) = −(nK/2)·ln(2π) − (1/2)·‖Z^(l)‖_F²
```

L 平均をとると（`_lnpZ` は `Σ Z²` についてアフィンなので平均と交換できる）:

```
lnpZ_avg = −(nK/2)·ln(2π) − (1/2)·(1/L) Σ_l ‖Z^(l)‖_F²
```

`scale_Z` 適用後は `(1/L) Σ_l ‖Z^(l)‖_F² = nK`（§6.2）なので:

```
lnpZ_avg = −(nK/2)·ln 2π − (nK/2) = −(nK/2)·(1 + ln 2π) = −1.4189385332 · n K
```

BIC 換算:

```
−2 · lnpZ_avg = n·K·(1 + ln 2π) = 2.8378770664 · n · K
```

### 5.3 exact / 近似の区別 — **最重要の限定** [DERIVED]

| 主張 | 判定 |
|---|---|
| 各サンプル `l` ごとに `‖Z^(l)‖_F² = nK` | **成立しない。** `scale_Z` は配列**全体**に 1 個のスカラーを掛ける（`model.py:499-503`）ので、スライスごとには一定でない |
| **L 平均** `(1/L) Σ_l ‖Z^(l)‖_F² = nK` | **実数演算では厳密に成立** |
| 実数演算での厳密性 | 厳密。`np.mean(Z**2) = 1` が事後条件 |
| float64 での厳密性 | `scale = sqrt(k/(avg_sq*k))` は `1/sqrt(avg_sq)` に対し丸めを 2 回余分に含むため `1 ± O(10⁻¹⁶)`。**「厳密に」は理想化された表現** |
| コード上で成立しない経路 | 2 つのみ: (i) `avg_sq == 0`（Z 全ゼロ、理論上のみ）、(ii) `NaN`。ただし**全ランナーで NaN/Inf ガードが `scale_Z` の前に発火する**（`run_fixed_official_exp1_bic_full.py:134-137` → `:139`）ので (ii) は閉じている [CODE] |

**独立した数値的裏づけ [ARTIFACT]:**
Issue #37 の pilot（`expfam/results/k_selection/k_selection_score_pilot_20260823_runinfo.csv`、gate G1）が
`max lnpZ_abs_err = 2.274e-13`（tol `1e-06`）を 42 fits で記録している。
**ただしこれは objective-consistent prototype 系列であり、fixed 系列ではない。**
`scale_Z` と `_lnpZ` は両系列で共有された関数なので代数は同一だが、
**fixed official Exp1 の CSV には `lnpZ_observed` 列がなく、fixed 系列での数値検証は存在しない** [UNRESOLVED]。

### 5.4 K → K+1 の差 [DERIVED]

```
Δ lnpZ_avg   = −(n/2)·(1 + ln 2π)          = −1.4189385332 n
Δ(−2·lnpZ)   = + n·(1 + ln 2π)             = +2.8378770664 n
```

| n | K あたりの BIC 寄与 | 参考: `(d−K)·ln n`（d=15, K=3） |
|---:|---:|---:|
| **150（合成）** | **425.6816** | 60.13 |
| 178（Wine, d=13） | 505.14 | 51.82 |
| 280（Cora, d=50） | 794.61 | 276.11（K=1→2） |

**ユーザー提示の「n=150 で約 425.68」は正しい。コードから再確認した。**

### 5.5 この項の性質 — direct score と推定過程を分ける [CORRECTED-20260907]

**(a) direct score としての性質 [DERIVED]:**
`Q_strict` の中の `lnpZ_avg` 項は、`scale_Z` により
**同一 `K` ならどの `Z_samples` を得ても L 平均では同じ値になる。**
すなわち **criterion 内の direct な `lnpZ` 項は、`Z` ごとの prior plausibility を区別しない。**
complete-data 基準における `E_q[ln p(Z)]` 項として形式上は正当だが、
実効的には `2.8379·n` という **n に比例する固定次元罰則**として働いている。

**(b) しかし prior 自体はモデルと推定に効いている [CODE]:**
`z_i ~ N(0, var_z·I)` は E-step の

- gradient Term 1: `−(1/var_z)·z_i`（`model_dual_expfam_fixed.py:56`, `model_dual_expfam.py:137`）
- precision Term 1: `(1/var_z)·I`（`model_dual_expfam_fixed.py:101`, `model_dual_expfam.py:181`）

に入っており、**`Z_samples` の生成そのものを規定している**（`A_i ⪰ I` はここから来る。§8.4）。

> **したがって「prior plausibility の情報がゼロ」「prior がモデルに影響しない」という一般化はしない。**
> **正しい表現:「direct な `lnpZ` score は `scale_Z` により `K` だけの決定論的項へ退化している。
> 一方 prior は latent sample の生成過程には依然として影響している。」**

**(c) 設計意図について:**
**この項は「複雑度ペナルティとして設計されたもの」ではない**（コード・docstring・原論文のいずれにも
そう設計したという記述はない）[CODE + PAPER]。

---

## 6. Z-rescaling audit

### 6.1 8 問への回答 [CODE]

| # | 問い | 回答 |
|---|---|---|
| 1 | どの関数で | `LatentStructuralModel.scale_Z`（`reproduction/src/model.py:468-504`）。fixed も historical も override していない |
| 2 | どのタイミングで | **各 EM iteration の E-step 直後**（`run_fixed_official_exp1_bic_full.py:139`） |
| 3 | sampling 前か後か | **後**。L 回の `calc_eta_newton` がすべて終わってから、`Z_samples` 配列全体に適用 |
| 4 | M-step 前か後か | **前**。M-step は scale 済み `Z_samples` に対して行われる（`:144-149`） |
| 5 | Q 計算前か後か | **前**。`Q_strict` は最終 iteration の scale 済み `Z_samples` で計算（`:159`） |
| 6 | criterion 保存前か後か | **前**。CSV の `q_strict` / `bic` はすべて scale 後の値 |
| 7 | F や w も同時変換されるか | **されない。** `scale_Z` は `Z_samples` だけを返す（`:503`）。F・w は**その後の M-step で再推定される** |
| 8 | 同一 joint density の reparameterization か | **違う。objective の値そのものが変わる操作である**（§6.3） |

### 6.2 candidate K ごとの `‖Z‖_F²` 制御 [CODE + DERIVED]

```python
avg_sq = np.mean(Z_samples ** 2)                 # model.py:499  （軸指定なし = 全 n·K·L 要素）
scale  = np.sqrt(k / (avg_sq * k))               # model.py:502  （= 1/sqrt(avg_sq)）
return Z_samples * scale                         # model.py:503
```

事後条件:

```
mean((c·Z)²) = avg_sq / avg_sq = 1
⇔ Σ_{l,i,a} (Z^(l)_{ia})² = n·K·L
⇔ (1/L) Σ_l ‖Z^(l)‖_F² = n·K              ← K に線形。candidate K ごとに自動的にこの値に固定される
```

### 6.3 reparameterization ではない理由 [DERIVED]

モデルには尺度不定性 `(Z, F, w) → (cZ, F/c, w/c²)` があり、**η^X も η^Y も不変**である。
しかし `ln p(Z)` は `c` について不変ではない:

```
ln p(cZ) − ln p(Z) = −(1/2)(c² − 1)·‖Z‖_F²
```

したがって `scale_Z` は**この軌道上の 1 点を選ぶゲージ固定**であり、
**joint density の値を変える操作**である。F・w が同時に `1/c`・`1/c²` 倍されるわけではなく、
**M-step が scale 後の Z を所与として再推定する**という形で追随する [CODE]。

**帰結:** `lnpZ` 項からデータ依存性が完全に抜ける。X fit・Y fit は M-step の再推定により
（Gaussian-X の `calc_F` は閉形式なので）ほぼ回復するが、
`w0`・`w` は Adam の有限反復（`max_iter=50`）で追随するため完全な回復は保証されない [INFERENCE]。

### 6.4 原論文との構造差 [PAPER + CODE]

原論文 Appendix B は `σ_z²` を **M-step で推定する** [PAPER]:

```
Eq.(B-13)  σ_z² = (1/(L n k)) Σ_{l=1}^{L} Σ_{i=1}^{n} ‖z_i^(l)‖²
```

Algorithm 2 の第 11 行も
"compute parameters F, μ_x, Σ and σ_z² based on Eqs. (10)–(12) and (14)" と書いており、
**印刷されたアルゴリズムに `scaleZ` に相当する操作は存在しない** [PAPER]。

**代数的接点 [DERIVED]:** `scale_Z` の下では Eq.(B-13) は厳密に `σ_z² = 1` を返す。
すなわち「`var_z = 1` 固定 + `scale_Z`」と「Eq.(B-13) で `σ_z²` を推定」は**互いの不動点**である。

**しかし Q / BIC への影響は明確に異なる [DERIVED]:**

- `scale_Z` + `var_z = 1`: `lnpZ` は定数 `−(nK/2)(1+ln 2π)` に退化する。
- Eq.(B-13) で真に再推定: plug-in すると `ln p(Z) = −(nK/2)(ln 2π σ̂_z² + 1)` となり、
  **`ln σ̂_z²` の分だけデータ依存性が残る**（ただし主要項はやはり `−(nK/2)(1 + ln 2π)` である）。

**どちらが正しいとは判定しない。EM 軌道の等価性は証明していない** [UNRESOLVED]。

---

## 7. Explicit penalty decomposition

### 7.1 コードからの再構成 [CODE]

```
P(K) = [ K·d − K(K−1)//2  +  d·1{family_x = 'gaussian'}  +  1·1{family_y = 'gaussian'} ] · ln n
```

（`utils_expfam.py:399-403`）

### 7.2 各項の帰属 [CODE + DERIVED]

| 項 | 個数 | 帰属 | 根拠の強さ |
|---|---|---|---|
| `K·d` | `Kd` | **F 由来**（d×K 行列） | [CODE] docstring `"F (rotational constraint)"` |
| `− K(K−1)/2` | | **rotation indeterminacy 補正** | `K(K−1)/2 = dim O(K)`。数値としては一致する [DERIVED]。**ただし「rotation 補正である」と原論文は書いていない**（§7.5） |
| `+ d`（Gaussian-X のみ） | `d` | **Σ_X の対角** | [CODE] docstring `"Sigma: d if family_x='gaussian'"` |
| `+ 1`（Gaussian-Y のみ） | `1` | **σ_Y**（dispersion） | [CODE] docstring `"sigma_y: 1 if family_y='gaussian'"` |
| `w0` | 0 | **数えていない** | [CODE] docstring `"(w0, w are treated as implicit per NOLTA 2024 convention)"` |
| `w` | 0 | **数えていない** | 同上 |
| intercept（μ_x） | 0 | **モデルが持たない**（バイアスなし。root `CLAUDE.md` §1） | [CODE] `calc_F` 注記 `"μ_x is assumed to be 0"` |
| `var_z` | 0 | 数えていない（恒久的に 1.0 なので自由パラメータではない） | [CODE] |
| `Z`（n×K） | 0 | 数えていない（サンプリング対象であって推定対象ではない） | [CODE] |

### 7.3 K 依存 / K 非依存の分離 [DERIVED]

| 部分 | K 依存 | 増分 |
|---|:---:|---|
| `K·d − K(K−1)/2`（F 由来） | **YES** | `Δ = (d − K)`、すなわち `Δ·ln n = (d−K)·ln n` |
| `d·1{GX}` | NO | 0 |
| `1·1{GY}` | NO | 0 |

**K→K+1 の明示罰則増分は `(d − K)·ln n` のみ**（`d − K + 1/2` ではない）[DERIVED]。
n=150, d=15 での実測値: K=1→2 で 70.15, 2→3 で 65.14, 3→4 で 60.13, …, 8→9 で 35.07。

**設計上の非対称性 [DERIVED]:** `n(n−1)/2` 個の Y 観測は fit 項（`lnpY`）には寄与するが、
**K に対する明示罰則には一切寄与しない。** K の罰則は X 側（`F`）からしか入らない。

### 7.4 num_params の一次照合 [ARTIFACT]

`fixed_exp1_bic_k1to9_summary.csv` の `num_params` 列 270 行すべてを
`K·15 − K(K−1)//2 + (15 or 0) + (1 or 0)` と照合し、**全行一致**を確認した（本監査で実行）。

| シナリオ | fx/fy | K=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A | pois/bern | 15 | 29 | 42 | 54 | 65 | 75 | 84 | 92 | 99 |
| B | gauss/pois | 30 | 44 | 57 | 69 | 80 | 90 | 99 | 107 | 114 |
| C | bern/gauss | 16 | 30 | 43 | 55 | 66 | 76 | 85 | 93 | 100 |

さらに `−2·q_strict + num_params·ln 150` から `bic` 列を再構成し、
**270 行全体で最大絶対差 `7.276e-12`** を確認した [ARTIFACT + DERIVED]。

### 7.5 なぜ `ln n` を掛けるのか — 根拠の所在 [PAPER + UNRESOLVED]

**原論文 Eq.(26)** [PAPER]:

```
BIC = −2 ln L + [ (k+1)d − k(k−1)/2 ] ln n
```

原論文本文が述べる理由は次の 2 点だけである [PAPER]:

> "The reasons of using the BIC as the model selection criterion are that
> 1) the objective of this study is to estimate the latent structure underlying the observed data,
> and that 2) BIC possesses consistency in this regard."

- **`ln n` の `n` が何であるべきかの導出は原論文にない。** 参照 [20] を挙げるのみ。
- **`− k(k−1)/2` の意味の説明も原論文にない。** 「rotation indeterminacy 補正である」という帰属は
  リポジトリ側の解釈（`utils_expfam.py:100-102` のコード註釈）である [CODE]。
  `K(K−1)/2 = dim O(K)` と数値が一致すること自体は [DERIVED] だが、
  **著者の意図の証拠にはならない** [UNRESOLVED]。
- **`(k+1)d = kd + d` の `+d` が `μ_x` と `Σ` のどちらを指すかも原論文からは判別できない** [UNRESOLVED]。
  現行 Python は `Σ_X` に割り当てている（`μ_x` を持たないため）。

**一致範囲 [DERIVED]:** `calc_bic_dual` の `p̂` が Eq.(26) と厳密に一致するのは
**Gaussian-X かつ非 Gaussian-Y のときだけ**である（`Kd − K(K−1)/2 + d = (K+1)d − K(K−1)/2`）。
合成 3 シナリオでは **B（gaussX/poisY）のみ一致**し、A（poisX）・C（bernX + gaussY）は一致しない。

> **`ln n` を掛ける根拠は「原論文 Eq.(26) がそうしているから」以上のものが
> コード・原論文・リポジトリ内ドキュメントのいずれにも存在しない。**
> 本モデル固有の Schwarz 展開に基づく導出は [UNRESOLVED]。

---

## 8. q(Z) / Laplace / MC interpretation

### 8.1 `calc_eta_newton` の実際の動作 [CODE]

`reproduction/src/model.py:360-462`:

```python
Z = self.params['Z'].copy()                                   # :402
for i in range(n):                                            # :411
    for iteration in range(max_iter):                         # :413   max_iter=10, alpha=0.5
        grad = self._calc_gradient(X, Y, Z, F, sigma, ...i)   # :417   ← 現在の Z 全体に依存
        A_i  = self._calc_precision_matrix(Z, F, ...i)        # :420
        Z[i,:] = Z[i,:] - alpha * (A_i_inv @ grad)            # :436   ← in-place 更新
    A_i = self._calc_precision_matrix(Z, F, ...i)             # :448   ← 最終位置で再評価
    Z[i,:] = rng.multivariate_normal(Z[i,:], A_i_inv)         # :457   ← 平均 = 現在位置
return Z                                                      # :462
```

### 8.2 `q_K(Z)` の正体 — 候補の判定 [CODE + DERIVED]

| 候補 | 判定 |
|---|---|
| exact posterior | **NO** |
| global Laplace posterior | **NO**（大域の mode/Hessian は一度も計算されない） |
| **node-wise conditional Laplace approximation を用いた sequential (Gauss–Seidel) sampling** | **これが正しい記述** |
| Gibbs-like sequential samples | **YES**（ただし各条件付き分布は Laplace 近似であり厳密条件付きではない） |
| Monte Carlo empirical distribution | 部分的に YES（L=5 点の経験分布として Q に入る） |
| rescale 後の empirical distribution | **YES。Q が実際に評価するのはこれ** |

### 8.3 `q` は何であって、何でないか [CORRECTED-20260907]

**旧版は「`q` は密度として書けない」「`H(q)`・`KL` が定義できない」と書いていた。これは強すぎるので撤回する。**
**確率法則が存在しないことと、閉形式密度が得られないことと、現存 artifact から計算できないことは、
すべて別の判定である。** 以下で 5 段階に分けて判定する。

#### 8.3.1 5 段階の判定

| # | 判定対象 | 判定 | 根拠 |
|---|---|---|---|
| 1 | **確率法則（probability law）が存在するか** | **YES** | アルゴリズムは可測写像（Newton 反復・`np.linalg.inv`/`pinv` の分岐を含む）と非退化 Gaussian 抽出（`:457`、`A_i ⪰ I` より `A_i^{-1}` は正定値）の有限合成である。したがって `Z_samples` 上に well-defined な確率法則を誘導する [DERIVED] |
| 2 | **通常の Lebesgue density を持つか（`scale_Z` 適用前、全 `Z_samples ∈ R^{nKL}`）** | **YES（と考えてよい）** | 1 スイープは各ノードで非退化 Gaussian を逐次に引く Gibbs 型構成であり、条件付き密度の積として `R^{nK}` 上の Lebesgue 密度を持つ。`L` スイープの合成も同様 [DERIVED] |
| 3 | **通常の Lebesgue density を持つか（`scale_Z` 適用後、全 `Z_samples ∈ R^{nKL}`）** | **NO** | `scale_Z` は `Z_samples` 全体を `{W : mean(W²) = 1}`（`R^{nKL}` 内の余次元 1 の球面）へ写す。押し出し測度は**この低次元多様体に台を持ち、`R^{nKL}` 上の Lebesgue 密度を持たない** [DERIVED] |
| 4 | **恒等式で使う `q̄`（単一 `Z ∈ R^{nK}` 上の混合周辺分布）が Lebesgue density を持つか** | **UNRESOLVED（`L ≥ 2` では持つ可能性が高い）** | 制約は `L` 枚**全体**に掛かるので、`L ≥ 2` なら 1 枚のスライスは `nK` 次元方向に自由度を残す。したがって a.c. である可能性が高い [INFERENCE]。**ただし本監査は証明していない。** `L = 1` なら `‖Z‖_F² = nK` に拘束されて退化する（fixed 系列は `L = 5`） |
| 5 | **closed-form density が書けるか** | **NO** | 掃引順序依存の Newton 写像（減衰 `alpha`、`max_iter=10`、`tol` 打ち切り、`inv`/`pinv` 分岐）と `L` 枚全体に掛かる `scale_Z` の結合により、解析形は得られない [DERIVED] |
| 6 | **`H(q̄)` が well-defined か** | **UNRESOLVED** | 判定 4 が成り立ち、かつ積分が収束すれば well-defined。本監査は確認していない |
| 7 | **`H(q̄)`・`KL(q̄‖p_θ(Z\|D))` を現存 artifact から数値評価できるか** | **NO** | scale 前の `Z`・スケール係数 `c`・サンプリング時の `A_i`・Newton 経路のいずれもランナーが保存していない [CODE + ARTIFACT] |

> **要約表現（推奨）:**
> **「current algorithm は `Z_samples` 上の確率法則を誘導するが、その解析的 density と entropy は現在確定していない。」**

#### 8.3.2 `p(Z|X,Y,θ)` の近似と呼んでよいか

**「事後分布 `p(Z|X,Y,θ)` の Laplace 近似」と無限定に呼んではならない** [DERIVED]。理由 5 点:

1. **Gauss–Seidel である。** ノード `i` の更新は、その時点で**既に更新済みのノード**（`j < i`）と
   **まだ更新されていないノード**（`j > i`）が混在した `Z` に依存する（`:417`, `:420`）。
   したがって `q` は **各ノードの Laplace 密度の積ではない**。
2. **mode に到達している保証がない。** Newton は `alpha = 0.5` の減衰付きで `max_iter = 10`、
   `tol = 1e-4` で打ち切られる。`η_i` は「前回位置からの減衰 Newton 数ステップ後の点」であり、
   条件付き mode とは限らない [CODE]。
3. **平均が mode ではなく現在位置である。** `:457` は `rng.multivariate_normal(Z[i,:], A_i_inv)` であり、
   平均は Newton 後の現在位置である。
4. **L = 5 個は iid ではない。** E-step ループは各 `l` で直前のサンプルから Newton を開始し次の `l` に渡す
   （`run_fixed_official_exp1_bic_full.py:131-132`）。**burn-in も thinning もない長さ 5 の単一連鎖**である [CODE]。
5. **`scale_Z` の押し出し。** `Q_strict` が評価するのは Laplace 密度ではなく、
   その**決定論的正規化写像による押し出し**である（§6）。

**使ってよい表現:**
「各 `p(z_i | Z_{−i}, X, Y, θ)` の Laplace 近似を条件付き分布として用いた Gauss–Seidel 逐次サンプリングの、
`scale_Z` 適用後の出力法則」

**使ってはならない表現:** 「事後分布 `p(Z|X,Y,θ)` の（Laplace）近似そのもの」
「`q = Π_i N(η_i, A_i^{-1})`」。

### 8.4 KI-001 との接点 [CODE]

`A_i` の形は全 lineage で

```
A_i = (1/var_z)·I + F^T diag(≥0) F + c·w²·Z^T diag(≥0) Z ,   c ∈ {0.5, 1}
```

fixed は `c = 1`（`model_dual_expfam_fixed.py:113`）、historical は `c = 0.5`（`model_dual_expfam.py:200`）。
**どちらの `c` でも第 2・第 3 項は半正定値なので `var_z = 1` の下で `A_i ⪰ I`、`det A_i ≥ 1`** [DERIVED]。
この事実は §10 で使う。

---

## 9. Q vs observed marginal likelihood identity

### 9.1 一般恒等式の導出 [DERIVED]

`D = (X, Y)` とし、`q(Z)` を `p(Z|D,θ)` に絶対連続な任意の密度とする。

```
ln p(D|θ) = E_q[ ln p(D|θ) ]                                （被積分関数が Z に依存しない）
          = E_q[ ln p(D,Z|θ) − ln p(Z|D,θ) ]                （ベイズの定理）
          = E_q[ ln p(D,Z|θ) ] − E_q[ ln q(Z) ]
            + E_q[ ln q(Z) − ln p(Z|D,θ) ]
```

ここで `Q(q,θ) := E_q[ln p(D,Z|θ)]`、`H(q) := −E_q[ln q]`、`KL := KL(q ‖ p(Z|D,θ))` と置くと:

> **`ln p(D|θ) = Q(q,θ) + H(q) + KL(q ‖ p(Z|D,θ))`**  … (★)

同値な形:

> **`Q = ln p(D|θ) − H(q) − KL(q ‖ p(Z|D,θ))`**

`KL ≥ 0` なので `ln p(D|θ) ≥ Q + H(q)`（ELBO）。

### 9.2 (★) の適用範囲 — 恒等式そのものと、実装固有の評価可能性を分ける [CORRECTED-20260907]

**旧版の誤り（撤回する）:** 旧版は「E-step のサンプルが `θ^(T−1)` 由来で `Q_strict` が `θ^(T)` で評価されるから
(★) を適用できない」と書いていた。**これは誤りである。**

(★) は **任意の `q`** と **任意の `θ`** について成立する。
`q` が `p_{θ_old}(Z|D)` から作られたものであっても、右辺の joint と posterior を `θ_new` で評価することは
恒等式を壊さない。実際、通常の EM の `Q(θ, θ_old) = E_{p(Z|D,θ_old)}[ln p(D,Z|θ)]` はまさにその形である。

> **θ の不一致が意味するのは「恒等式が使えない」ことではなく、
> 「`KL(q ‖ p_{θ^(T)}(Z|D))` がゼロに近いと仮定できない」ことである** [CORRECTED-20260907]。

#### 9.2.1 正しい分類

| 問い | 答え | 分類 |
|---|---|---|
| **(★) という恒等式そのもの** | 任意の `q ≪ p_θ(Z\|D)`、任意の `θ` で成立 | **EXACT** |
| `q` にあたる確率法則は存在するか | 存在する（§8.3.1 判定 1） | **EXACT**（存在は確定） |
| その `q̄` が Lebesgue density を持つか | `L ≥ 2` では持つ可能性が高いが未証明（§8.3.1 判定 4） | **UNRESOLVED** |
| `q̄` の closed-form density | 得られない（§8.3.1 判定 5） | **NOT AVAILABLE** |
| `H(q̄)` が well-defined か | 判定 4 に依存 | **UNRESOLVED** |
| `H(q̄)`・`KL` を現存 artifact から数値評価できるか | scale 前 `Z`・`c`・サンプリング時 `A_i` が未保存 | **NOT COMPUTABLE**（§8.3.1 判定 7） |
| `H(q̄)` に **Gaussian / block-diagonal shortcut** を当てはめてよいか | `q̄` は factorized Gaussian ではない（§8.3.2） | **NOT JUSTIFIED** |
| `θ` の不一致 | (★) は壊れない。`KL` が小さいと仮定できないだけ | **KL ≠ 0（大きさ未知）** |
| `L = 5`、非 iid 連鎖 | `Q_dual` は `E_{q̄}[·]` の単一実現 MC 推定量 | **APPROXIMATE**（誤差評価なし） |
| Newton の有限反復・減衰 | 条件付き mode 非到達 | **APPROXIMATE** |
| `scale_Z` の押し出し | 全 `Z_samples` の関数による非線形正規化。`c` は復元不能 | **NOT COMPUTABLE**（entropy への寄与が定量できない） |

**総合判定 [DERIVED]:**

> **(★) は EXACT であり、current implementation に対しても概念的に適用してよい。**
> すなわち **「`Q_strict` は `ln p(X,Y|θ)` から `H(q̄)` と `KL(q̄‖p_θ(Z|D))` を欠いた量である」
> という記述は正しい。**
>
> **NOT JUSTIFIED なのは恒等式の適用ではなく、
> `H(q̄)` と `KL` の具体的な評価、とくに Gaussian / block-diagonal entropy による shortcut である。**
>
> 現在言えるのは「その 2 項の大きさは現存 artifact からは評価できない」までである。

### 9.3 3 つの量を混同しない [DERIVED + PAPER]

| 記号 | 定義 | `p(Z)` を含むか | `Z` を積分するか |
|---|---|:---:|:---:|
| **(Q1)** | `ln p(X,Y \| Z, θ)` | 含まない | しない |
| **(Q2)** | `ln p(Z, X, Y \| θ)` の事後期待値 ＝ `Q_strict` の対象 | **含む** | しない（MC 事後平均） |
| **(Q3)** | `ln p(X,Y \| θ)` ＝ 観測データ周辺尤度（Schwarz BIC の当てはまり項） | — | **する** |

**`Q_strict` は (Q2) の近似であり、(Q3) ではない。** これが KI-010 の実体である。

### 9.4 Exact identity vs implementation-specific approximation [ADDED-20260907]

**この 5 段階を絶対に混同しないこと。**
段階が下がるほど「一般に正しい命題」から「実装固有の仮定つき計算」へ移る。

| 段階 | 設定 | `H(q)` の扱い | `KL` の扱い | 判定 |
|:--:|---|---|---|---|
| **1** | **一般の変分恒等式** `ln p_θ(D) = E_q[ln p_θ(D,Z)] + H(q) + KL(q ‖ p_θ(Z\|D))`。任意の `q ≪ p_θ(Z\|D)`、任意の `θ` | 定義として存在（可積分性の下で） | 定義として存在、`≥ 0` | **EXACT**（実装に依存しない数学的事実） |
| **2** | `q = ` **exact posterior** `p_θ(Z\|D)` | — | **`KL = 0`**。したがって `ln p_θ(D) = Q + H` | **EXACT**（ただし本モデルで exact posterior は得られない） |
| **3** | `q = ` **global Gaussian Laplace approximation**（`R^{nK}` 上の 1 個の Gaussian、共分散 `Σ_q`） | `H = (1/2) ln{(2πe)^{nK} \|Σ_q\|}` として**計算可能** | `KL ≠ 0`、大きさは未知 | **仮定つき計算可能。current 実装は大域 mode/Hessian を一度も計算しないので該当しない** |
| **4** | `q = ` **product of node-wise Gaussian Laplace distributions** `Π_i N(η_i, A_i^{-1})` | `H = Σ_i[(K/2)ln(2πe) − (1/2)ln det A_i]` として**block-diagonal で計算可能** | `KL ≠ 0`（ノード間相関の無視分を含む）、大きさは未知 | **仮定つき計算可能。§10.2 の surrogate はこの段階の量である** |
| **5** | **current implementation**: Gauss–Seidel 掃引 ＋ 有限減衰 Newton ＋ `L = 5` 依存連鎖 ＋ `scale_Z`（`L` 枚全体に掛かる正規化）の出力法則 | **確率法則は誘導されるが、解析的 density・entropy の評価方法は未確定**（§8.3.1） | 同上 | **UNRESOLVED / NOT COMPUTABLE** |

**混同してはならない典型例:**

- 段階 1 が EXACT であることを根拠に、段階 5 の `H`・`KL` の**値**を語ること。
- 段階 4 の `Ĥ` を「current `q` の entropy」と呼ぶこと（§10.1・§10.2）。
- 段階 5 で closed-form が無いことを根拠に、段階 1 の恒等式そのものを否定すること
  （**旧版はこの誤りを犯していた。§9.2 で撤回済み**）。

---

## 10. Entropy and KL interpretation

### 10.1 Gaussian / block-diagonal entropy shortcut を当てはめてよいか [CORRECTED-20260907]

`q` が `m` 次元 Gaussian なら `H(q) = (1/2) ln{ (2πe)^m |Σ_q| }`、`m = nK`。

**`Σ_q = blockdiag(A_1^{-1}, …, A_n^{-1})` と仮定してよいか → NO** [DERIVED]。
**ここで否定されるのは shortcut の適用可能性であって、entropy という概念の存在ではない**（§8.3.1・§9.4）。

コードから判断できる理由:

1. §8.3.2-(1)：Gauss–Seidel なので出力法則は per-node 密度の積では**ない**。
   これは §9.4 の段階 4 と段階 5 の違いそのものである。
2. `A_i` は掃引中の位置で評価されており、サンプリング時点と最終 `Z_samples` での値も異なる
   （`A_i^post` はサンプリング時の `A_i` ではない。§10.3）。
3. **`scale_Z` は固定スカラー倍ではない。** これが最も見落とされやすい点なので下で分けて扱う。

#### 10.1.1 スケール変換の entropy — 固定 `c` の場合と current の場合 [DERIVED]

**固定 `c`（データに依存しない定数）の場合:** `Z' = cZ` は線形同型なので

```
H(Z') = H(Z) + nK · ln|c|
```

が成立する（Jacobian が `c^{nK}`）。

**current の `scale_Z` はこの形ではない** [CODE + DERIVED]:

```python
avg_sq = np.mean(Z_samples ** 2)      # model.py:499   ← Z_samples 全体（n·K·L 要素）の関数
scale  = 1 / np.sqrt(avg_sq)          # model.py:502
```

- **`c = c(Z_samples)` は確率変数であり、定数ではない。** したがって上の Jacobian 公式は使えない。
- 写像 `W ↦ W / sqrt(mean(W²))` は **`R^{nKL}` から余次元 1 の球面への射影**であり、
  **同型ではない**（半径方向の情報が消える）。押し出しは `R^{nKL}` 上の Lebesgue 密度を持たない（§8.3.1 判定 3）。
- スケール係数 `c` は**ランナーが scale 前の `Z` を返さないため復元できない** [CODE]。

> **したがって `H(q) = (1/2) ln{(2πe)^{nK} |blockdiag(A_i^{-1})|}` を
> current implementation の出力法則に適用してはならないし、
> `+ nK ln c` という固定スケール補正で救うこともできない。**

### 10.2 factorized Gaussian Laplace surrogate における代数的恒等式 [DERIVED / CORRECTED-20260907]

> **本節が扱うのは §9.4 の段階 4 の量である。**
> **ここで計算されるのは `q̃(Z) = Π_i N(η_i, A_i^{-1})` という factorized Gaussian Laplace surrogate の
> 微分エントロピーであって、current implementation が誘導する法則のエントロピーではない。**
> **この一文を落として本節の結論だけを引用してはならない。**

任意の曲率行列族 `{A_i}` に対し **Gaussian entropy surrogate** を

```
Ĥ({A_i}) := Σ_i [ (K/2)·ln(2πe) − (1/2)·ln det A_i ]
          = (nK/2)(1 + ln 2π) − (1/2)·Σ_i ln det A_i
```

と定義する。ここで**第 1 項は `− lnpZ_avg` と完全に一致する**（§5.2）[DERIVED]。

したがって:

```
lnpZ_avg + Ĥ({A_i}) = −(nK/2)(1+ln 2π) + (nK/2)(1+ln 2π) − (1/2)Σ_i ln det A_i
                     = −(1/2) Σ_i ln det A_i
```

> **正確な言い方 [CORRECTED-20260907]:**
> **「factorized Gaussian Laplace surrogate `q̃ = Π_i N(η_i, A_i^{-1})` の微分エントロピーには、
> `lnpZ_avg` と同係数・逆符号の O(nK) 項 `(nK/2)(1+ln 2π)` が代数的に現れる。
> この surrogate を代入する限りにおいて両者は打ち消し合い、残るのは `−(1/2) Σ_i ln det A_i` である。」**
>
> **避けるべき言い方:** 「426 は entropy と相殺する（可能性がある）」。
> この表現は「current `q` の entropy と相殺する」と読まれうるが、その主張は成立していない。

`A_i ⪰ I`（§8.4）より `Σ_i ln det A_i ≥ 0` なので、残差は `≤ 0`。
BIC 換算では `S({A_i}) = S_cf + Σ_i ln det A_i ≥ S_cf`。

**この代数から「相殺する」という結論は出ない** [UNRESOLVED]:

- **`Ĥ` は current implementation の出力法則の entropy ではない。**
  §8.3.2 の 5 点（Gauss–Seidel / mode 非到達 / 平均が現在位置 / 非 iid 連鎖 / `scale_Z` 押し出し）の
  いずれか 1 つでも `q̃` との一致を壊す。§9.4 の段階 4 と段階 5 は別物である。
- `Σ_i ln det A_i` は `K` に依存する（`A_i` は K×K）。**そのオーダーも符号の効き方も未確定。**
- したがって **「426/K は純粋な過剰罰則である」も「entropy と相殺するので問題ない」も、
  どちらも本監査では証明できない。**

**それでも本節が有用な理由 [DERIVED]:**
`lnpZ` 項と Laplace 型 entropy の主要項が**同係数で現れる**という事実自体は、
「426 は自動的に余分である」という直観を**単独では支持しない**ことを示す。
すなわち本節は「426 が過剰である」という結論を**阻止する**役割を持つのであって、
「426 が正当である」ことを支持するのではない。

### 10.3 `Σ_i ln det A_i` の測定状況（先行監査からの現状更新）[ARTIFACT]

先行監査 `k_selection_theory_audit_20260822.md` §13.6 は
「`Σ_i ln det A_i` はどこにも記録されていない」と記録している（2026-08-22 時点で正しい）。

**その翌日の Issue #37 pilot が記録している** [ARTIFACT]:

- `expfam/results/k_selection/k_selection_score_pilot_20260823_summary.csv` の
  `sum_log_det_A_post` 列（42 fits）
- 定義: `A_i^post = _calc_precision_matrix(Z_samples[:,:,l], F, sigma, var_z, w0, w, i)` を
  **最終 `scale_Z` 適用済みサンプルで再評価**したもの。**サンプリング時の `A_i` ではない**
  （runinfo の `a_post_definition`）

**限定条件（絶対に外さないこと）:**

- **系列が違う。** `implementation_lineage = objective-consistent (model_dual_expfam_consistent.py) — PROTOTYPE, NOT manuscript-approved`
- **条件が違う。** シナリオ A のみ、n ∈ {75, 150}、K = 1..7、3 trials（42 fits）。
  本監査が中心に据える fixed 系列（3 シナリオ、K=1..9、10 trials、270 fits）とは別物である。
- **`fixed_official/exp1` および `exp1_k9` の CSV には `log_det` 系の列が存在しない**（本監査で列名を確認）。
  **fixed 系列では `Σ_i ln det A_i` は依然として未測定である** [UNRESOLVED]。

参考（prototype 系列の観測、`k_selection_score_pilot_20260823_selection.csv`）[ARTIFACT]:

| score | 定義 | argmin |
|---|---|---|
| `bic_impl` | 現行 criterion | **K=3 を 6/6** |
| `scf` | `bic_impl − nK(1+ln 2π)`（Z prior 除去） | **K=7（範囲上端）を 6/6** |
| `s_laplace_post` | `scf + Σ_i ln det A_i^post` | **K=3 を 6/6** |

**この 3 行を「Q-based が正しい」「補正 BIC が正しい」と読んではならない。**
`scf` も `s_laplace_post` も**診断量**であり、周辺尤度でも ELBO でも Schwarz BIC でもない（runinfo の `score_c2`/`score_c3` の明記どおり）。

---

## 11. Complete K-dependent term ledger

### 11.1 項ごとの K 依存とオーダー

| # | 項 | criterion に入るか | K 依存 | オーダー（K→K+1 の変化） | 根拠 |
|---|---|---|---|---|---|
| 1 | **X fit** `−2·E[ln p(X\|Z)]` | YES | **YES** | 最大 `O(nd)` 個の項の和。実測は §11.2 | [CODE] `model_dual_expfam.py:291-334` |
| 2 | **Y fit** `−2·E[ln p(Y\|Z)]` | YES | **YES** | 最大 `O(n²)` 個の項の和。実測は §11.2 | [CODE] `model_expfam.py:241-269` |
| 3 | **Gaussian prior normalization** `+nK ln 2π` | YES | **YES** | **厳密に `+n·ln 2π` = 275.68（n=150）** | [CODE] `utils_expfam.py:319` |
| 4 | **Z norm** `+‖Z‖_F²` | YES | **YES（決定論的）** | **厳密に `+n` = 150（n=150）** | [DERIVED] §5.2 |
| 3+4 | **Z prior 合計** | YES | **YES** | **厳密に `+n(1+ln 2π)` = 425.6816** | [DERIVED] |
| 5 | **posterior / MC / Laplace 由来** [CORRECTED-20260907] | **direct term: NO / indirect: YES** | **CONDITIONAL** | **明示的な entropy / log-det curvature 項（`Σ_i ln det A_i` など）は criterion に存在しない**。一方 posterior geometry（`A_i`）は E-step のサンプリングを通じて `Z_samples` を決め、項 1・2 の値に**間接的に影響する** | [CODE] `calc_bic_dual` に該当項なし ＋ `model.py:448-457` |
| 6 | **explicit parameter penalty**（F 由来） | YES | **YES** | **`(d−K)·ln n`**。n=150,d=15 で 70.15→35.07 | [CODE] `utils_expfam.py:399` |
| 6b | explicit penalty（Σ_X, σ_Y 由来） | YES | **NO** | 0 | [CODE] `:400-401` |
| 7 | **rescaling 由来** | 間接 | **CONDITIONAL** | 項 3+4 を決定論化する経路としてのみ。独立項としては存在しない | [DERIVED] §6 |
| 8 | **rotation correction** `−K(K−1)/2` | YES（項 6 の内数） | **YES** | `−K`（`Δ` の内訳）。項 6 に含めて計上済み | [CODE] `:399` |
| 9 | **family-specific terms** |  |  |  |  |
| 9a | Poisson 階乗補正 | YES | **NO** | 0 | [CODE] `:375-378` |
| 9b | Gaussian-X の `−(nd/2)ln 2π` | YES | **NO** | 0 | [CODE] `model_dual_expfam.py:323` |
| 9c | Gaussian-Y の `−0.5 ln 2π`（ペアごと） | **NO（欠落）** | **NO** | 0（欠落量は `0.5 ln 2π · n(n−1)/2` = 10269.14） | [CODE] `model_expfam.py:262-264` |
| 9d | Gaussian-X の `−0.5 Σ ln σ_l²` | YES（項 1 の内数） | **YES**（残差が K で変わる） | 項 1 に含めて計上済み | [CODE] `:322` |
| 9e | Gaussian-Y の `−0.5 ln σ_Y²`（ペアごと） | YES（項 2 の内数） | **YES**（`calc_sigma_y` が K 依存） | 項 2 に含めて計上済み | [CODE] `model_expfam.py:264`, `:212-235` |
| 10 | `w0, w` の parameter count | **NO（数えていない）** | NO | 0（K 非依存の 2 個なので argmin に影響なし。絶対値は `2 ln n` = 10.02 分ずれる） | [CODE] `:398` |
| 11 | `Z`（nK 個）の parameter count | **NO** | — | 0 | [CODE] |
| 12 | `var_z` の parameter count | **NO** | — | 0 | [CODE] |

### 11.2 一次 CSV による実測台帳（fixed 系列、n=150, d=15, K_TRUE=3, 10 trials 平均）

`fixed_exp1_bic_k1to9_summary.csv` から本監査が恒等分解した [ARTIFACT + DERIVED]。
`fit := q_strict − lnpZ_det`、`lnpZ_det = −(nK/2)(1+ln 2π)`。**単位はすべて BIC スケール（−2 倍済み）。**

**シナリオ A（poisson-X / bernoulli-Y）**

| K | `−2·lnpZ` | `−2·fit` | penalty | **BIC** | `Δ(−2 lnpZ)` | `Δ(−2 fit)` | `Δ penalty` | `Δ BIC` |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 425.7 | 20438.3 | 75.2 | 20939.2 | — | — | — | — |
| 2 | 851.4 | 17547.5 | 145.3 | 18544.2 | +425.7 | **−2890.8** | +70.1 | −2395.0 |
| 3 | 1277.0 | 14972.1 | 210.4 | **16459.6** | +425.7 | **−2575.4** | +65.1 | −2084.5 |
| 4 | 1702.7 | 14823.2 | 270.6 | 16796.5 | +425.7 | −149.0 | +60.1 | **+336.8** |
| 5 | 2128.4 | 14682.2 | 325.7 | 17136.3 | +425.7 | −141.0 | +55.1 | +339.8 |
| 6 | 2554.1 | 14543.3 | 375.8 | 17473.1 | +425.7 | −138.9 | +50.1 | +336.9 |
| 7 | 2979.8 | 14428.0 | 420.9 | 17828.7 | +425.7 | −115.3 | +45.1 | +355.5 |
| 8 | 3405.5 | 14329.9 | 461.0 | 18196.4 | +425.7 | −98.1 | +40.1 | +367.7 |
| 9 | 3831.1 | 14257.0 | 496.1 | 18584.2 | +425.7 | −72.9 | +35.1 | +387.8 |

**シナリオ B（gaussian-X / poisson-Y）**

| K | `−2·lnpZ` | `−2·fit` | penalty | **BIC** | `Δ(−2 fit)` | `Δ penalty` | `Δ BIC` |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 425.7 | 43439.7 | 150.3 | 44015.7 | — | — | — |
| 2 | 851.4 | 36617.6 | 220.5 | 37689.4 | **−6822.1** | +70.1 | −6326.3 |
| 3 | 1277.0 | 30661.4 | 285.6 | **32224.0** | **−5956.2** | +65.1 | −5465.4 |
| 4 | 1702.7 | 30480.1 | 345.7 | 32528.6 | −181.2 | +60.1 | **+304.6** |
| 5 | 2128.4 | 30331.1 | 400.9 | 32860.4 | −149.0 | +55.1 | +331.8 |
| 6 | 2554.1 | 30202.3 | 451.0 | 33207.4 | −128.8 | +50.1 | +347.0 |
| 7 | 2979.8 | 30074.3 | 496.1 | 33550.2 | −128.0 | +45.1 | +342.8 |
| 8 | 3405.5 | 29976.1 | 536.1 | 33917.7 | −98.3 | +40.1 | +367.5 |
| 9 | 3831.1 | 29871.8 | 571.2 | 34274.2 | −104.2 | +35.1 | +356.5 |

**シナリオ C（bernoulli-X / gaussian-Y）**

| K | `−2·lnpZ` | `−2·fit` | penalty | **BIC** | `Δ(−2 fit)` | `Δ penalty` | `Δ BIC` |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 425.7 | 5629.1 | 80.2 | 6134.9 | — | — | — |
| 2 | 851.4 | −2969.0 | 150.3 | −1967.3 | **−8598.1** | +70.1 | −8102.3 |
| 3 | 1277.0 | −37720.9 | 215.5 | **−36228.4** | **−34751.9** | +65.1 | −34261.1 |
| 4 | 1702.7 | −37881.0 | 275.6 | −35902.7 | −160.1 | +60.1 | **+325.7** |
| 5 | 2128.4 | −38038.5 | 330.7 | −35579.4 | −157.6 | +55.1 | +323.2 |
| 6 | 2554.1 | −38197.9 | 380.8 | −35263.0 | −159.3 | +50.1 | +316.4 |
| 7 | 2979.8 | −38342.8 | 425.9 | −34937.1 | −144.9 | +45.1 | +325.8 |
| 8 | 3405.5 | −38486.4 | 466.0 | −34614.9 | −143.6 | +40.1 | +322.2 |
| 9 | 3831.1 | −38640.9 | 501.1 | −34308.7 | −154.5 | +35.1 | +306.2 |

### 11.3 台帳から読み取れる構造 [DERIVED]

1. **K < K_GEN 側（過小次元）は fit が支配する。** `Δ(−2 fit)` が −2575 〜 −34752 で、
   Z prior（+425.7）と明示罰則（+65）を桁で圧倒する。**K=3 まで下がるのは fit の力である。**
2. **K > K_GEN 側（過大次元）では fit の改善が 73〜181 に落ち込む。**
   このとき `+425.7`（Z prior）単独で毎ステップ確実に上回る。
3. **明示罰則単独では止まらない。** `Δ penalty` は 35〜60 であり、
   `|Δ(−2 fit)|` = 73〜181 に**すべての K ≥ 3 ステップ・全 3 シナリオで負ける**。
4. **したがって過大次元側で K を止めているのは Z prior 項である。**

---

## 12. Original paper vs current implementation

### 12.1 抽出手順の記載 [PAPER]

`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`（567,126 bytes, 19 pages）の
FlateDecode ストリームを展開し、`TJ` / `Tj` 演算子の literal 文字列を連結して本監査が直接抽出した。
Eq.(16)-(18) は 3 章（p.340-341）、Eq.(26) は 4.3 節（p.343）、Appendix B は p.351-352。
**`paper/2.pdf` は参照していない。**

### 12.2 原論文の記述（抽出そのまま）[PAPER]

```
Eq.(16)  L = ∏_{i=1}^{n} p(x_i | F, z_i, μ_x, Σ) · { ∏_{i≠j} p(y_ij | z_i, z_j, w0, w) }^{1/2}

Eq.(17)  Q(θ, θ_old) = ∫ p(Z | X, Y, θ_old) · ln p(Z, X, Y | θ) dZ

Eq.(18)  Q(θ, θ_old) ≃ (1/L) Σ_l ln p(Z^(l), X, Y | θ)
                     = (1/L) Σ_l { ln p(Z^(l)|θ) + ln p(X|Z^(l),θ) + ln p(Y|Z^(l),θ) }

Eq.(26)  BIC = −2 ln L + [ (k+1)d − k(k−1)/2 ] ln n

本文（4.3 節）  "Note that the true dimension k* affects the likelihood function of L from Eq.(16)."
```

### 12.3 対応・不一致の表

| 観点 | **原論文** | **current code** | 判定 |
|---|---|---|---|
| **`L`（尤度）の定義** | Eq.(16)。`p(x_i\|F,z_i,μ_x,Σ)` と `{∏_{i≠j} p(y_ij\|·)}^{1/2}` の積。**`p(Z)` を含まない** | 対応物は `lnpX_avg + lnpY_avg`（= `Q_strict − lnpZ_avg`） | **構成要素は整合**（Y 側の `Σ_{i<j}` も一致）。ただし `μ_x` が current にない |
| **`Q`（EM の Q）** | Eq.(17)：`∫ p(Z\|X,Y,θ_old) ln p(Z,X,Y\|θ) dZ`。**`p(Z)` を含む** | `Q_strict`（`lnpZ + lnpX + lnpY`）。**`p(Z)` を含む** | **一致** |
| **`Q` の MC 近似** | Eq.(18)：`(1/L) Σ_l ln p(Z^(l),X,Y\|θ)` | `calc_Q_dual`（`utils_expfam.py:345-352`）。同形 | **一致**（ただし `Z^(l)` の生成法に §6.4・§8 の差がある） |
| **`BIC` の当てはまり項** | **Eq.(26) は `ln L`（Eq.16）を使う。**本文が明示 | **`Q_strict`（= Eq.18 の `Q`）を使う** | **不一致（決定的）** |
| **不一致の量** | — | — | `−2·lnpZ_avg = nK(1+ln 2π)` = **425.68·K**（n=150） |
| **`BIC` の罰則項** | `(k+1)d − k(k−1)/2` | `Kd − K(K−1)/2 + d·1{GX} + 1·1{GY}` | **Gaussian-X かつ非 Gaussian-Y のときのみ一致**。合成 B のみ該当 |
| **`σ_z²`** | Eq.(14) / Eq.(B-13) で **M-step 推定** | `var_z = 1.0` 固定 + `scale_Z` | **不一致**（代数的には互いの不動点。§6.4） |
| **`μ_x`** | Eq.(11) で推定 | **持たない**（バイアスなし） | 不一致 |
| **Y 側の 1/2** | Eq.(16)(19)(20)(22)(23) に**あり** | `calc_log_likelihood_Y` の `0.5·Σ_{全非対角}` = `Σ_{i<j}` は Eq.(16) と**一致**。E-step の extra 1/2 は fixed で除去（KI-001） | 尤度レベルは一致。E-step は意図的な差（KI-001） |
| **`ln n` の根拠** | 参照 [20] と「BIC possesses consistency」のみ。導出なし | 原論文 Eq.(26) を踏襲 | **根拠は原論文の慣行のみ** [UNRESOLVED] |

### 12.4 決定的な事実 [PAPER]

> **原論文自身が `L`（Eq.16）と `Q`（Eq.17/18）を別の量として使い分けており、
> BIC には `p(Z)` を含まない `L` を使っている。**
> **current code は `p(Z)` を含む `Q` を使っている。**

### 12.5 それでも「原論文が正しく current が間違い」とは言えない [PAPER + UNRESOLVED]

- **Eq.(16) の `L` も `z_i` に条件づけた量であり、`Z` を積分していない。**
  すなわち原論文 Eq.(26) の当てはまり項は §9.3 の **(Q1)** であって **(Q3)** ではない。
  **「原論文 Eq.(26) は standard Schwarz BIC である」とは言えない。**
- **原論文 Experiment 2 が Eq.(16) の `L` を実際にどう評価したか（どの `Z` を代入したか）は本文から特定できない** [UNRESOLVED]。
- **本研究は原論文 Experiment 2 を再現していない。** 少なくとも次の差がある:
  `σ_z²` の扱い / `L`（10 vs 5）/ 反復（10 vs 8）/ 生成器の正規化 / `μ_x` の有無 /
  family 組合せ（原論文は Gaussian-X + Bernoulli-Y のみ。合成 A/B/C はいずれも別）。

---

## 13. Interpretation of fixed 30/30

### 13.1 一次証拠 [ARTIFACT]

`expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_frequency.csv`:
A / B / C ともに `k=3, selected_count=10, total_trials=10, selected_rate=1.0`。他の K はすべて 0。
条件: n=150, d=15, K_GEN=3, candidate K=1..9, L=5, num_iter=8, 10 trials, 全 270 fits で `success=True`。

### 13.2 「このcriterionでK=3が選ばれた」以上に言えること

**言える [DERIVED、一次 CSV の恒等分解のみ]:**

1. **選択を決めた項は Z prior 項である。**
   `S_cf`（Z prior を除いた診断量）の argmin は **trial 単位 30 trial 中 30 trial で K=8 または K=9（範囲上端）** に移る:
   A = {9:8, 8:2}、B = {9:9, 8:1}、C = {9:10}。
   （**この trial 単位の計算は本監査の新規分である。** 先行監査 §13.2a は trial 平均のみで、
   平均レベルでは 3 シナリオとも argmin = 9 = 範囲上端。値も一致確認済み。）
2. **明示的 parameter-count 罰則だけでは K=3 は保たれない。**
   K ≥ 3 の全ステップ・全 3 シナリオで `Δ penalty`(35〜60) < `|Δ(−2 fit)|`(73〜181)（§11.2）。
3. **K < 3 側は fit が決めている。** `Δ(−2 fit)` が −2575 〜 −34752（§11.2）。

**言えない [UNRESOLVED]:**

- **Schwarz BIC の妥当性**、**一般的な true-K recovery**、**model-selection consistency** のいずれも意味しない
  （KNOWN_ISSUES.md「今すぐ主張してよいこと」の限定と同一）。
- **n 依存性は測定されていない。** n=150 の 1 点のみ。
  （`fixed_official/exp2` の n sweep には BIC 列がなく `k` は k_true 固定。）
- **「Z prior 項が過剰罰則である」とは言えない** [CORRECTED-20260907]**。**
  §10.2 の通り、**factorized Gaussian Laplace surrogate** の entropy には同係数・逆符号の O(nK) 項が
  代数的に現れる。したがって 425.68 が「余分」かどうかは、この監査だけでは決まらない。
  （この surrogate は current algorithm の出力法則の entropy ではない。§9.4 段階 4 と段階 5 の区別。）
- **fit の内訳（X fit と Y fit のどちらが効いたか）は分解できていない。** §13.3。

### 13.3 寄与分解 artifact の存在確認（検索結果）

| 分解したい量 | fixed 系列に artifact があるか |
|---|---|
| `q_strict` → `lnpZ` / fit の 2 分解 | **artifact としては存在しない。** ただし `lnpZ` が決定論的なので **`q_strict` 列と `num_params` 列だけから恒等的に計算できる**（本監査が実施。§11.2） |
| fit → **X fit と Y fit** の 2 分解 | **存在しない。** `fixed_official/exp1`・`exp1_k9` の CSV 列は `q_strict, bic, num_params, rmse_z, rmse_x, rmse_y` のみで、`lnpX` / `lnpY` を分けた列はない [ARTIFACT] |
| `Σ_i ln det A_i`（posterior curvature） | **fixed 系列には存在しない**（列名を全走査して 0 件）。objective-consistent prototype 系列にのみ存在（§10.3） |

> **結論: X fit と Y fit の寄与分解は、現在の fixed 系列 artifact だけでは分解不能である。**
> `lnpX` / `lnpY` を個別に保存した CSV がないため、既存データからは復元できない。
> **本監査は新しい実験を行わない。**

---

## 14. What is established（確立している）

すべて [CODE] / [ARTIFACT] / [PAPER] / [DERIVED] のいずれかで裏づけられる。

| # | 内容 | 根拠 |
|---|---|---|
| E1 | criterion は `−2·Q_strict + p̂·ln n` であり、`Q_strict` は `lnpZ + lnpX + lnpY + 階乗補正` の L 平均である | [CODE] §3.1 |
| E2 | `var_z` は恒久的に 1.0。M-step で更新されない | [CODE] §3.1 |
| E3 | `scale_Z` により `(1/L)Σ_l ‖Z^(l)‖_F² = nK` が L 平均について成立する（実数演算で厳密、float64 で `1±O(10⁻¹⁶)`） | [CODE + DERIVED] §5.3, §6.2 |
| E4 | したがって `−2·lnpZ_avg = nK(1+ln 2π)` は決定論的。n=150 で K あたり 425.6816 | [DERIVED] §5.2, §5.4 |
| E5 | 明示罰則の K 依存部分は `F` 由来のみで、増分は `(d−K)·ln n`。n=150,d=15 で 35〜70 | [CODE + DERIVED] §7.3 |
| E6 | `num_params` 列 270 行と、`−2q+p ln n` からの BIC 再構成が最大誤差 `7.276e-12` で一致 | [ARTIFACT + DERIVED] §7.4 |
| E7 | fixed 系列 K=1..9 の全逐次差分（§11.2 の 3 表） | [ARTIFACT + DERIVED] |
| E8 | K ≥ 3 の全ステップ・全 3 シナリオで `Δ penalty < \|Δ(−2 fit)\| < Δ(−2 lnpZ)` | [DERIVED] §11.3 |
| E9 | `S_cf` の argmin は trial 単位 30/30 で範囲上端（K=8 または 9）に移る | [DERIVED] §13.2 |
| E10 | `q` は Gauss–Seidel 逐次 Laplace サンプリングであり、per-node Laplace の積ではない | [CODE] §8.1-8.3 |
| E10b | **サンプリング手続きは `Z_samples` 上に well-defined な確率法則を誘導する**（可測写像と非退化 Gaussian 抽出の有限合成）[ADDED-20260907] | [DERIVED] §8.3.1 判定 1 |
| E10c | **`scale_Z` 適用後の全 `Z_samples` の法則は `R^{nKL}` 内の余次元 1 の球面に台を持ち、Lebesgue 密度を持たない** [ADDED-20260907] | [DERIVED] §8.3.1 判定 3 |
| E10d | **`scale_Z` のスケール係数は `Z_samples` 全体の関数であり、固定スカラー倍ではない**（`H(cZ)=H(Z)+nK ln\|c\|` は適用できない）[ADDED-20260907] | [CODE + DERIVED] §10.1.1 |
| E10e | **prior `N(0, var_z·I)` は E-step の gradient Term 1・precision Term 1 として `Z_samples` の生成に影響している**（direct score が退化していることと別）[ADDED-20260907] | [CODE] §5.5(b) |
| E11 | L=5 は iid ではなく burn-in なしの単一連鎖 | [CODE] §8.3.2-(4) |
| E12 | `Q_strict` は θ^(T)、`Z_samples` は θ^(T−1) 由来で θ が一致しない | [CODE] §4.2 |
| E13 | `A_i ⪰ I`（両系列、`var_z=1` の下で）。したがって `det A_i ≥ 1` | [CODE + DERIVED] §8.4 |
| E14 | 恒等式 `ln p(D\|θ) = Q + H(q) + KL` は**任意の `q ≪ p_θ(Z\|D)` と任意の `θ` について** 数学的に EXACT。**`q` を作った `θ_old` と評価に使う `θ_new` が異なっても恒等式は壊れない** [CORRECTED-20260907] | [DERIVED] §9.1, §9.2 |
| E14b | θ の不一致が意味するのは「恒等式が使えない」ことではなく「`KL(q ‖ p_{θ^(T)}(Z\|D))` がゼロに近いと仮定できない」ことである [ADDED-20260907] | [DERIVED] §9.2 |
| E15 | 原論文 Eq.(26) の当てはまり項は `ln L`（Eq.16）であり `p(Z)` を含まない。Eq.(18) の `Q` は含む。**両者を原論文自身が使い分けている** | [PAPER] §12.2-12.4 |
| E16 | 原論文は `σ_z²` を Eq.(B-13) で M-step 推定する。`scaleZ` 相当の操作は印刷版にない | [PAPER] §6.4 |
| E17 | Gaussian-Y の `−0.5 ln 2π` が欠落（K 非依存、n=150 で 10269.14） | [CODE + DERIVED] §3.4 |
| E18 | fixed と historical で criterion の関数形は同一。差は E-step 経由のみ | [CODE] §3.5 |
| E19 | historical 系列（`run_em_dual`）は Gaussian-X で stale Σ を持つ。fixed official スクリプトは同期済み | [CODE] §4.3 |

---

## 15. What is not established（確立していない）

| # | 内容 | 分類 |
|---|---|---|
| N1 | `Q_strict` と `ln p(X,Y\|θ)` の差を**数値的に**分解すること [CORRECTED-20260907] | **NOT COMPUTABLE**（恒等式 (★) の適用自体は EXACT。`H(q̄)`・`KL` を現存 artifact から評価できないだけ。§9.2） |
| N1b | `q̄` が Lebesgue density を持つか / `H(q̄)` が well-defined か [ADDED-20260907] | **UNRESOLVED**（§8.3.1 判定 4・6） |
| N2 | `H(q) = (1/2)ln{(2πe)^{nK}\|blockdiag(A_i^{-1})\|}` を current の出力法則に適用すること | **NOT JUSTIFIED**（§9.4 段階 4 ≠ 段階 5。§10.1） |
| N2b | `+ nK ln c` の固定スケール補正で `scale_Z` を扱うこと [ADDED-20260907] | **NOT JUSTIFIED**（`c` は `Z_samples` の関数で定数でない。§10.1.1） |
| N3 | 「Z prior の 425.68/K は過剰罰則である」 | **UNRESOLVED**（§10.2） |
| N4 | 「Z prior 項は（current `q` の）posterior entropy と相殺する」 | **UNRESOLVED**（同上。surrogate 上の代数は §10.2 で成立するが別物） |
| N5 | fixed 系列での `Σ_i ln det A_i` の値・K 依存オーダー | **UNRESOLVED**（未測定。§10.3） |
| N6 | fixed 系列での X fit / Y fit の寄与分離 | **UNRESOLVED**（artifact に列がない。§13.3） |
| N7 | K 選択の n 依存性（fixed 系列） | **UNRESOLVED**（n=150 の 1 点のみ。§13.2） |
| N8 | `ln n` の `n` が本モデルで正しい実効標本数であること | **UNRESOLVED**（§7.5） |
| N9 | `−K(K−1)/2` が原論文で rotation 補正として意図されたこと | **UNRESOLVED**（§7.2, §7.5） |
| N10 | `(k+1)d` の `+d` が `μ_x` と `Σ` のどちらか | **UNRESOLVED**（§7.5） |
| N11 | 原論文 Experiment 2 が Eq.(16) をどう評価したか | **UNRESOLVED**（§12.5） |
| N12 | `scale_Z` 方式と Eq.(B-13) 方式の EM 軌道の等価性 | **UNRESOLVED**（§6.4） |
| N13 | historical 系列の stale Σ が数値に与えた影響量 | **UNRESOLVED**（§4.3） |
| N14 | fixed 系列 270 fits で内部 NaN リセットが起きたか | **UNRESOLVED**（`nan_occurred` が CSV に保存されていない。`success` 列は例外の有無のみ） |
| N15 | `S_cf` / `s_laplace_post` のいずれかが「正しい」基準であること | **NOT JUSTIFIED**（いずれも診断量。§10.3） |

---

## 16. Implications for K-selection theory

**本節は理論的含意の整理であり、criterion の変更提案ではない。**

### 16.1 現行 criterion が実際に持っている「複雑度罰則」の構造 [DERIVED]

```
実効罰則(K) = n·K·(1 + ln 2π)          ← 暗黙。Z prior 由来。O(nK)
            + [K·d − K(K−1)/2]·ln n    ← 明示。F 由来。O(dK·ln n)
```

n=150, d=15 での比 = 425.68 : (d−K)·5.011 ≈ **7.1 : 1**（K=3）。

**n → ∞ での比 [DERIVED]:** `2.8379·n / ((d−K)·ln n) = O(n / ln n) → ∞`。
すなわち **n が増えるほど、暗黙の Z prior 罰則が明示の Schwarz 罰則を相対的にさらに圧倒する。**

### 16.2 何と競合しているか [DERIVED]

| 項 | K→K+1 のオーダー |
|---|---|
| `lnpY` の改善（complete-data、Z を代入） | 最大 `O(n²)` 個の項の和（実オーダー未確定） |
| `lnpX` の改善 | 最大 `O(nd)` 個の項の和 |
| 決定論的 `lnpZ` 項 | **厳密に `2.8379·n`** |
| 明示 Schwarz 罰則 | `(d−K)·ln n` |

**n → ∞ で明示 Schwarz 罰則は他の 3 項に対して漸近的に無視できる。**
競合するのは「当てはまり改善」対「決定論的 `lnpZ` 項」であり、**どちらが勝つかは未解決** [UNRESOLVED]。

### 16.3 incidental parameters との構造的類似 [INFERENCE / CORRECTED-20260907]

潜在変数 `z_i` の総数は `nK` で n に比例して増える。
`Q_strict` は `Z` を周辺化せずサンプルを代入しているため、
**§9 の entropy / KL 欠落と同じ事象**が incidental parameters の形で現れている [DERIVED]。

**旧版の記述「周辺化すればこの問題は生じない」は強すぎるので撤回する** [CORRECTED-20260907]。

正しくは次のとおりである [DERIVED + UNRESOLVED]:

- **周辺化すれば `nK(1+ln 2π)` という決定論的 direct prior 項が同じ形で残るとは限らない。**
  この項は「`Z` を積分せずサンプルを代入する」ことに固有の形をしている。
- **しかし `ln ∫ p(Z,X,Y|θ) dZ` の中には、別の形で K 依存の complexity が現れる。**
  たとえば posterior volume、Hessian determinant、モデル次元、特異性（潜在変数モデルは一般に特異である）など。
- **その形が本モデルで具体的に何になるかは未解決である。**

> **推奨表現:「周辺化すると、current Q-based に現れる決定論的 direct prior 項とは
> 異なる形で K 依存の complexity が現れる。その形が本モデルで何になるかは未解決である。」**
>
> **言ってはならない:「周辺化すれば penalty 問題はなくなる」。**

### 16.4 呼称について（KI-010 の維持）

現行 criterion を **Schwarz BIC と呼んではならない。**
本監査は KNOWN_ISSUES.md KI-010 の位置づけ（**Q-based complete-data criterion / ICL-type**）を維持する。
関数名 `calc_bic_dual`・CSV 列名 `BIC`・過去結果の呼称は**変更しない**（root `CLAUDE.md` §5）。

---

## 17. Unresolved questions

| # | 問い | 解くのに必要なもの |
|---|---|---|
| U1 | fixed 系列で `Σ_i ln det A_i` はどう K 依存するか | fixed 系列での曲率記録（未実施） |
| U2 | fit の X 側 / Y 側の寄与分離 | `lnpX` / `lnpY` を分けて保存する記録（未実施） |
| U3 | current algorithm の出力法則 `q̄` は Lebesgue density を持つか、`H(q̄)` は well-defined か [CORRECTED-20260907] | 確率法則の存在は確定済み（§8.3.1 判定 1）。絶対連続性と可積分性の証明が必要 |
| U3b | `H(q̄)`・`KL(q̄‖p_θ(Z\|D))` の数値評価 [ADDED-20260907] | 現存 artifact では不可（scale 前 `Z`・`c`・サンプリング時 `A_i` が未保存）。新たな記録設計が必要 |
| U3c | 周辺化した `ln p(X,Y\|θ)` において K 依存 complexity がどの形で現れるか [ADDED-20260907] | 本モデル固有の解析（特異性を含む） |
| U4 | K 選択の n 依存性 | n を振った K 選択実験（fixed 系列に存在しない） |
| U5 | `scale_Z` 方式と `σ_z²` 推定方式の EM 等価性 | 証明または反例 |
| U6 | 原論文 Experiment 2 の `L` 評価手続き | 原論文著者への確認、または原論文設定の再現 |
| U7 | `ln n` の `n` の妥当性 | 本モデル固有の実効標本数の解析 |
| U8 | historical 系列の stale Σ の影響量 | 差分測定（未実施） |
| U9 | fixed 系列 270 fits の NaN リセット有無 | 記録がないため事後には確定できない |
| U10 | 30/30 が n=150 以外でも成立するか | n を振った実験 |

---

## 付録 A. 検証コマンド

```bash
# (1) criterion のコード行
sed -n '386,404p' expfam/src/utils_expfam.py        # calc_bic_dual
sed -n '315,352p' expfam/src/utils_expfam.py        # _lnpZ, calc_Q_dual
sed -n '355,379p' expfam/src/utils_expfam.py        # calc_Q_dual_strict
sed -n '468,504p' reproduction/src/model.py         # scale_Z
sed -n '360,462p' reproduction/src/model.py         # calc_eta_newton
sed -n '291,334p' expfam/src/model_dual_expfam.py   # calc_log_likelihood_X
sed -n '241,269p' expfam/src/model_expfam.py        # calc_log_likelihood_Y

# (2) 一次 CSV の恒等分解（本監査が実行した内容）
#     lnpZ_det = -(n*k/2)*(1+log(2*pi));  fit = q_strict - lnpZ_det
#     S_cf     = -2*fit + num_params*log(n)
#     BIC 再構成の最大絶対差 = 7.276e-12（270 行）
#     対象: expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_summary.csv

# (3) 選択結果の一次確認
cat expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_frequency.csv

# (4) prototype 系列の曲率記録（系列が違うことに注意）
head -1 expfam/results/k_selection/k_selection_score_pilot_20260823_summary.csv
cat     expfam/results/k_selection/k_selection_score_pilot_20260823_selection.csv

# (5) 原論文（本監査は FlateDecode ストリームを展開して TJ 配列から抽出した）
#     Eq.(16)(17)(18) : 3 章 p.340-341
#     Eq.(26)         : 4.3 節 p.343
#     Eq.(B-13)       : Appendix B p.352
```

## 付録 B. 先行文書との関係

| 文書 | 関係 |
|---|---|
| `reports/k_selection_theory/k_selection_theory_audit_20260822.md` | **理論的結論の大半は同文書で既に確立済み。** 本監査はコード・一次 CSV・原論文から独立に再導出し一致を確認した。数値（`S_cf` 値、`7.276e-12`、`425.68`）も一致 |
| 同 §13.2a | trial 平均レベルの `S_cf`。**本監査は trial 単位を追加した**（新規） |
| 同 §13.6 | 「`Σ_i ln det A_i` は記録されていない」は 2026-08-22 時点で正しい。**Issue #37 pilot（2026-08-23、prototype 系列）が記録している**（§10.3 で現状更新） |
| `reports/theory_audit/theory_audit_report_20260718.md` | `2.84n` の大きさに既に到達している歴史的記録。ただしそこでは欠落 `H(q)` に帰されており、本監査の `lnpZ_det` とは別の量。**書き換えない** |
| `KNOWN_ISSUES.md` KI-001 / KI-002 / KI-010 | 本監査はいずれの判断も変更しない |
| `EXPERIMENT_REGISTRY.md` | 本監査は追記していない（新実験を行っていないため） |

---

## 付録 C. 2026-09-07 改訂一覧

**改訂対象は本ファイル（同日作成・未コミットの作業中ドラフト）のみである。**
`.claude/rules/historical-records.md` が保護する「日付入りで凍結された過去の記録」の改変ではない。
過去の記録（`reports/theory_audit/*`、`reports/k_selection_theory/*`）は一切変更していない。

**改訂範囲:** 理論解釈のみ。**実装監査（[CODE]）・一次 CSV 監査（[ARTIFACT]）・原論文確認（[PAPER]）の
結果および数値は 1 つも変更していない。**

| # | section | 変更前の主張 | 変更後の主張 | 変更理由 |
|---|---|---|---|---|
| R1 | §9.2, §14 E14, Q1, Q3 | θ の不一致（`Z_samples` は `θ^(T−1)` 由来、`Q_strict` は `θ^(T)`）を理由の 1 つとして、恒等式 `ln p = Q + H + KL` の適用を **NOT JUSTIFIED** とした | 恒等式は**任意の `q`・任意の `θ` で EXACT**。θ の不一致は恒等式を壊さず、**`KL` がゼロに近いと仮定できない**ことを意味するだけ | 通常の EM の `Q(θ,θ_old) = E_{p(Z\|D,θ_old)}[ln p(D,Z\|θ)]` がまさにこの形であり、恒等式の成立に θ の一致は不要 |
| R2 | §8.3, §9.2, §15 N1, Q3 | 「`q` は密度として書けない」「`H(q)`・`KL` が**定義できない**」 | 確率法則は**存在する**（判定 1）。Lebesgue 密度・closed-form・well-definedness・計算可能性を**5 段階に分けて判定**（§8.3.1）。現状は「解析的 density と entropy は未確定」「現存 artifact からは計算不能」 | 逐次 Gaussian 抽出アルゴリズムは出力に確率法則を定める。**閉形式が無いことと分布が存在しないことは別**。「恒等式が成立しない」と「必要量を計算できない」を混同していた |
| R3 | §8.3 | Gauss–Seidel を理由に「閉形式も密度評価も持たない（intractable）」と 1 文で断定 | `scale_Z` 適用**前**は `R^{nKL}` 上で Lebesgue 密度を持つ（判定 2）。適用**後**は余次元 1 の球面に台を持ち Lebesgue 密度を持たない（判定 3）。恒等式で使う単一 `Z` 上の `q̄` は `L≥2` で a.c. の可能性が高いが**未証明**（判定 4） | `scale_Z` の押し出しがどの空間で何を壊すかを区別する必要がある |
| R4 | §10.1 | 「`scale_Z` の大域スケール `c` 倍は entropy を `+nK ln c` 動かすが `c` が復元できない」 | **固定 `c` なら `H(cZ)=H(Z)+nK ln\|c\|`。しかし current の `c = c(Z_samples)` は確率変数であり、写像は同型ですらない**（半径方向が潰れる）。固定スケール補正では扱えない | `c` を「復元できない定数」と書くと、定数として扱えるかのように読める。実際は変換の型が違う |
| R5 | §10.2, §1.3, §13.2, Q4, Q5 | 「surrogate を代入する限りにおいて両者の主要項は厳密に相殺する」「426 が entropy と相殺する可能性がある」 | **「factorized Gaussian Laplace surrogate `q̃ = Π_i N(η_i,A_i^{-1})` の entropy には同係数・逆符号の O(nK) 項が代数的に現れる」**を優先表現とし、**「current `q` の entropy と相殺する」と読める表現を排除**。本節の役割は「426 が過剰である」という結論を**阻止する**ことであって「426 が正当」を支持することではない、と明記 | 段階 4（surrogate）と段階 5（current 実装）の混同を防ぐため |
| R6 | §9.4（新設） | — | **Exact identity vs implementation-specific approximation** の 5 段階表を追加（一般恒等式 / exact posterior / global Laplace / node-wise product / current output law） | 5 段階を混同しないための固定表が必要 |
| R7 | Q2, §11.1 項 5 | 「posterior concentration は含まれていない」と断定 | **direct / indirect を分離**。「posterior entropy・log-det curvature の**明示項**は存在しない」が、「posterior geometry は sampled `Z` を通じて fit 項に**間接的に影響する**」 | `A_i` は `Z_samples` の生成を規定し、その `Z_samples` が `ln p(X\|Z)`・`ln p(Y\|Z)` を決める。「一切影響しない」は誤り |
| R8 | §5.5, Q2 | 「prior plausibility の情報は**ゼロ**」 | **direct score と推定過程を分離**。「direct な `lnpZ` score は `scale_Z` により `K` だけの決定論的項へ退化している。一方 prior は latent sample の生成過程には依然として影響している」 | prior は E-step の gradient Term 1・precision Term 1 に入っている（`A_i ⪰ I` の出所）。「prior が効かない」への一般化を防ぐ |
| R9 | §16.3 | 「`z_i` は本来周辺化されるべき潜在確率変数であり、**周辺化すればこの問題は生じない**」 | 周辺化すると direct prior 項は同じ形では残らないが、**posterior volume / Hessian determinant / モデル次元 / 特異性など別の形で K 依存 complexity が現れる。その形が本モデルで何になるかは未解決** | 「周辺化すれば penalty 問題がなくなる」は強すぎる |
| R10 | §4.4 | 「`E` と呼んではならない。理由: 期待値をとる分布 `q` が閉形式で書けない」 | 期待値の対象となる**確率法則は存在する**。呼んではならない理由は「計算されているのが期待値ではなく 1 実現の L=5 点平均だから」に修正 | R2 と同じ混同の修正 |
| R11 | §14, §15, §17 | — | E10b–E10e / E14b（確立事項）、N1b / N2b（未確立事項）、U3 / U3b / U3c（未解決）を追加・修正 | 上記改訂に対応する台帳の更新 |
| R12 | 末尾（新設） | — | **暫定結論 5 点**を追加 | 修正後の結論を短くまとめるため |

**改訂していないもの（確定事項として保持）:**
`C_Q(K)` の式（§3）／`Q_strict` の 4 成分分解（§3.1）／`scale_Z` による `(1/L)Σ_l‖Z^(l)‖²_F = nK` と
`−2 lnpZ_avg = nK(1+ln 2π)` = 425.6816/K（§5）／`S_cf` の trial 単位 30/30 の argmin 移動（§13.2）／
K>3 側で過大 K を止めている主要な direct 項が Z prior であること（§11.3）／
原論文 Eq.(16) と Eq.(17)(18) の区別と Eq.(26) が Eq.(16) を使うという不一致（§12）／
原論文 Eq.(26) も観測周辺尤度 (Q3) ではないこと（§12.5）。

---

## 最終回答（理論質問 Q1-Q7）

### Q1. current Q-based criterion は、観測分布 `p(X,Y)` の fit を評価する criterion と言えるか？

**言えない。** [DERIVED]

criterion の当てはまり項は `E_q[ln p(X,Y | Z, θ)]`（§9.3 の **(Q1)** の事後平均）であり、
**`Z` を積分した `ln p(X,Y|θ)`（(Q3)）ではない。**

ただし限定を 2 つ付ける [CORRECTED-20260907]:

- **完全に無関係ではない。** 恒等式 `Q = ln p(D|θ) − H(q) − KL` は
  **任意の `q`・任意の `θ` について EXACT** であり、current implementation にも**概念的に適用してよい**（§9.1・§9.2）。
  すなわち「`Q_strict` は `ln p(X,Y|θ)` から `H(q̄)` と `KL(q̄‖p_θ(Z|D))` を欠いた量である」という
  **記述は正しい。**
- **しかしその 2 項の大きさは現存 artifact から数値評価できない**（§8.3.1 判定 7）。
  したがって「観測 fit の近似である、ただしバイアスはこれだけ」という言い方はできない。
  **バイアスの大きさも符号も分からない。**
  （**NOT JUSTIFIED なのは恒等式の適用ではなく、`H`/`KL` の具体的評価と Gaussian shortcut である。**）

### Q2. latent configuration `Z` の posterior concentration / prior plausibility まで含めた criterion と見るべきか？

**direct（criterion に明示的に現れる項）と indirect（sampled `Z` 経由）を分けて答える必要がある** [CORRECTED-20260907]。
旧版は両者を分けずに「prior plausibility の情報はゼロ」「posterior concentration は含まれていない」と
断定していた。これを次の 4 点に置き換える [DERIVED]。

**(a) direct な Z-prior term は存在する。**
`Q_strict` には `lnpZ_avg` が形式上入っている（`utils_expfam.py:348`）。

**(b) ただし `scale_Z` により、その direct term は `K` だけの決定論的値へ退化している。**
`−(nK/2)(1+ln 2π)`。同一 `K` ならどの `Z_samples` を得ても L 平均では同じ値になる。
すなわち **direct な `lnpZ` score は `Z` ごとの prior plausibility を区別しない**（§5.5(a)）。
**「prior が効いていない」という意味ではない**：prior は E-step の gradient Term 1 と precision Term 1 として
`Z_samples` の生成に影響している（§5.5(b)）。

**(c) posterior entropy / log-det curvature を表す明示的な項は criterion に存在しない。**
`Σ_i ln det A_i` に相当する項は `calc_bic_dual` にも `calc_Q_dual_strict` にも現れない（§11.1 の項 5）。

**(d) ただし posterior geometry は sampled `Z` を通じて fit 項に間接的に影響する。**
`A_i`（および Newton 方向）は `Z_samples` の生成を規定し、その `Z_samples` が
`ln p(X|Z)`・`ln p(Y|Z)` の値を決める（`model.py:448-457`）。
**したがって「posterior geometry が criterion に一切影響しない」とは言えない。**

> **整理された答え:**
> **現行 criterion は「direct な Z-prior term を持つが、それは `scale_Z` により `K` のみの決定論的値へ退化しており、
> posterior entropy / log-det curvature の明示項は持たない。ただし posterior geometry は
> sampled `Z` 経由で fit 項へ間接的に影響する」complete-data 型基準である。**

### Q3. Q-based criterion と observed marginal likelihood との差を、entropy / KL の観点からどう正確に説明できるか？

**恒等式レベル（EXACT）:** [DERIVED]

```
Q(q,θ) = ln p(X,Y|θ) − H(q) − KL(q ‖ p(Z|X,Y,θ))
```

すなわち差は **entropy 項と KL 項の 2 つ**である。
**「差は entropy だけ」は誤りである。KL 項が必ず残る。**
**この恒等式は任意の `q`・任意の `θ` について成立するので、current implementation にも適用してよい**
[CORRECTED-20260907]。

**旧版の誤り（撤回する）:** 旧版は「`q` が密度を持たない」「`q` の θ と `Q_strict` の θ が違う」ことを理由に
**恒等式の適用そのものを NOT JUSTIFIED としていた。** どちらも理由にならない:

- 確率法則は存在する（§8.3.1 判定 1）。密度の有無は判定 4 の別問題であり、未解決である。
- θ の不一致は恒等式を壊さない。**`KL` がゼロに近いと仮定できない**ことを意味するだけである（§9.2）。

**current implementation について NOT JUSTIFIED / NOT COMPUTABLE なのは次である** [DERIVED]:

| 対象 | 判定 | 理由 |
|---|---|---|
| 恒等式 (★) の適用 | **EXACT** | 任意の `q`・`θ` で成立 |
| `H(q̄)` の well-definedness | **UNRESOLVED** | `q̄` の絶対連続性が未証明（§8.3.1 判定 4・6） |
| `H(q̄)`・`KL` の数値評価 | **NOT COMPUTABLE** | scale 前 `Z`・`c`・サンプリング時 `A_i` が未保存（判定 7） |
| Gaussian / block-diagonal shortcut の適用 | **NOT JUSTIFIED** | 出力法則は factorized Gaussian ではない（§9.4 段階 4 ≠ 段階 5） |
| 固定スケール補正 `+ nK ln c` | **NOT JUSTIFIED** | `c` は `Z_samples` の関数で定数でない（§10.1.1） |

**言える最大限:**
「`Q_strict` は `ln p(X,Y|θ)` から `H(q̄)` と `KL(q̄ ‖ p_θ(Z|D))` を欠いた量である。
その 2 項の大きさは current implementation・現存 artifact からは評価できない。」

**言ってはならない:** 「entropy の分だけずれる」「その差は `Σ_i ln det A_i` に縮約される」
「恒等式が current implementation に適用できない」。

### Q4. 「K を 1 増やすごとに Z prior 由来で約 426」は、単独で「過剰 penalty」と判断できるか？

**判断できない。** [UNRESOLVED]

**判断できない理由（§10.2）[CORRECTED-20260907]:**
**factorized Gaussian Laplace surrogate** `q̃(Z) = Π_i N(η_i, A_i^{-1})` の微分エントロピー
`Ĥ = (nK/2)(1+ln 2π) − (1/2)Σ_i ln det A_i` には、
`lnpZ_avg` の主要項 `−(nK/2)(1+ln 2π)` と**同係数・逆符号の O(nK) 項が代数的に現れる**。
この surrogate 上では両者は打ち消し合い、残るのは `−(1/2)Σ_i ln det A_i` だけになる。
**すなわち「426 は自動的に余分である」という直観は、この代数だけでは支持されない。**

**逆に「相殺するから問題ない」とも言えない:**
`Ĥ` は **current algorithm が誘導する出力法則の entropy ではない**（§9.4 段階 4 ≠ 段階 5、§10.1）。
Gauss–Seidel 掃引・有限減衰 Newton・非 iid 連鎖・`Z_samples` 全体に掛かる `scale_Z` の
いずれか 1 つでも `q̃` との一致を壊す。

**単独で言えること 2 点 [DERIVED]:**

1. **事実として、この項はデータからの情報を持たない**（`scale_Z` により決定論化）。
2. **事実として、この項が fixed 30/30 の過大次元側を止めている**（§13.2）。
   明示的 parameter-count 罰則だけでは止まらない。

**言ってはならない:** 「426 は過剰罰則である」「426 は正しい罰則である」。

### Q5. posterior entropy に同じ `O(nK)` 項が存在する可能性はあるか？ current implementation について実際にそう言えるか？

**可能性はある。current implementation について「実際にそうだ」とは言えない。** [DERIVED + UNRESOLVED]

**可能性の根拠（§9.4 の段階 4 における代数）[CORRECTED-20260907]:**
`q̃(Z) = Π_i N(η_i, A_i^{-1})` という **factorized Gaussian Laplace surrogate** に対しては

```
H(q̃) = Σ_i [ (K/2)ln(2πe) − (1/2)ln det A_i ] = (nK/2)(1+ln 2π) − (1/2)Σ_i ln det A_i
```

であり、**第 1 項は `−lnpZ_avg` と係数まで完全に一致する**（§10.2）。
さらに `A_i ⪰ I`（§8.4）なので `H(q̃) ≤ −lnpZ_avg`、すなわち `Q + H(q̃) ≤ Q − lnpZ_avg` が従う。

**current implementation について言えない理由:**
**current の出力法則は `q̃` ではない**（§9.4 段階 5）。
§8.3.2 の 5 点（Gauss–Seidel / mode 非到達 / 平均が現在位置 / 非 iid 連鎖 / `scale_Z` 押し出し）の
いずれか 1 つでも積形式 Gaussian という前提を壊す。とくに:

- **`scale_Z` は固定スカラー倍ではない。** `c = c(Z_samples)` は確率変数であり、
  写像 `W ↦ W/√mean(W²)` は同型ですらない（§10.1.1）。したがって
  **`H(cZ) = H(Z) + nK ln|c|` という固定スケール補正で救うことはできない。**
- `c` は**ランナーが scale 前の `Z` を返さないため復元できない** [CODE]。

**すなわち「同じ O(nK) 項が存在する可能性」は段階 4 の代数として確認できるが、
段階 5（current implementation）については well-definedness すら未解決である**（§8.3.1 判定 6）。

**参考（別系列の観測）[ARTIFACT]:**
objective-consistent prototype 系列では `s_laplace_post = S_cf + Σ_i ln det A_i^post` が
`K=3` を 6/6 で選んでいる（§10.3）。
**これは `Ĥ` を代入した代数量の挙動であって、実際の `q` の entropy の測定ではない。
また fixed 系列の結果でもない。**

### Q6. fixed 30/30 は、Q-based criterion の K-selection consistency の証拠なのか、有限標本 simulation evidence なのか？

**有限標本 simulation evidence である。consistency の証拠ではない。** [DERIVED]

理由:

1. **n が 1 点しかない。** n=150 のみ。consistency は `n → ∞` の主張であり、
   n を振った K 選択実験は fixed 系列に存在しない（§13.2, U4）。
2. **K_GEN が 1 点しかない。** K_GEN=3 のみ（fixed 系列）。
3. **条件が限定されている。** d=15、3 family 組合せ、L=5、num_iter=8、10 trials。
4. **選択の主因が理論的に正当化されていない項である。**
   §13.2 の通り 30/30 を決めているのは Z prior 項であり、
   その項の正当性は §10 で未解決である。**成功の機構が正当化されていない以上、
   同じ機構が別の n・K_GEN で同じ結果を出す保証はない。**
5. **選択された K が生成 K と一致したことは、criterion が「正しい」ことを意味しない。**
   偶然に近い条件（fit 改善が 73〜181、Z prior が 425.68）で符号が揃っただけの可能性を排除できない。

**言ってよい形:**
「明示した人工設定（n=150, d=15, K_GEN=3, K=1..9, 10 trials, 3 シナリオ）において、
歴史的に `BIC` と呼ばれてきた Q-based 基準が 30/30 trial で K=3 を選択した。」
（KNOWN_ISSUES.md の既存の限定表現と同一。）

### Q7. この監査だけから、Q-based を捨てて conditional / marginal へ変更すべきと言えるか？

**言えない。** [DERIVED]

理由 4 点:

1. **代替案の優位性を示す証拠がない。**
   本監査は conditional（(Q1)）も marginal（(Q3)）も評価していない。
   `S_cf`（(Q1) 相当の診断量）は fixed 系列で **argmin が範囲上端に張り付き、内点最小を持たない**（§13.2）。
   **すなわち「Z prior を外せば良くなる」という証拠は存在しない。むしろ逆の観測がある。**
2. **原論文側が (Q3) である保証もない。**
   Eq.(26) の当てはまり項は Eq.(16) = (Q1) であり、Schwarz BIC ではない（§12.5）。
   「原論文に合わせる」ことは「正しい基準にする」ことと同じではない。
3. **現行基準の欠陥が定量化されていない。**
   恒等式 (★) の適用自体は EXACT だが、`H(q̄)` も `KL` も現存 artifact から評価できない（§9.2）。
   **どれだけずれているか分からない量を、別の量に替える根拠にはできない。**
4. **これは Human Gate 事項である。** criterion の変更は root `CLAUDE.md` §6 の
   「研究目的・モデル・数式の変更」「評価指標の変更」に該当し、**人間の判断が必要である。**

**本監査が提供できるのは判断材料のみである:**
現行 criterion が数学的・実装的に何を評価しているか（§3-§12）と、
何が未解決か（§15, §17）。**変更の是非は含まない。**

---

## 暫定結論（2026-09-07 修正後）

**本節は結論の要約であり、criterion の採用判断ではない。**

1. **fixed 30/30 の有限標本上の選択機構は、かなり明確になった。**
   一次 CSV の恒等分解により、どの項がどのステップで効いているかを K=1..9・3 シナリオで台帳化できた（§11.2）。
   BIC 再構成の最大絶対差は 270 行で `7.276e-12` であり、分解は数値的に閉じている。

2. **K < 3 側は fit 改善で進み、K > 3 側は direct Z-prior cost `n(1+ln 2π)`（n=150 で 425.68/K）が
   主要因となって止まる。**
   K ≥ 3 の全ステップ・全 3 シナリオで `Δ penalty`(35〜60) < `|Δ(−2 fit)|`(73〜181) < `Δ(−2 lnpZ)`(425.68)。
   Z prior を除いた `S_cf` では argmin が **trial 単位 30/30 で範囲上端**へ移る（§13.2）。

3. **この direct cost が「真の最小潜在次元を選ぶための正しい complexity cost」である理論的根拠は、
   まだない。**
   原論文 Eq.(26) の当てはまり項は `p(Z)` を含まない Eq.(16) の `ln L` であり（§12.4）、
   `ln n` を掛ける根拠も「原論文の慣行」以上のものが確認できていない（§7.5）。
   `scale_Z` は原論文の印刷されたアルゴリズムには存在しない（§6.4）。

4. **ただし 426 が直ちに「過剰」であるとも言えない。**
   factorized Gaussian Laplace surrogate の entropy には同係数・逆符号の O(nK) 項が代数的に現れる（§10.2）。
   これは current algorithm の出力法則の entropy ではないが（§9.4 段階 4 ≠ 段階 5）、
   **「426 は自動的に余分である」という直観を単独では支持しない**ことを示している。
   結論を出すには marginalization / entropy / curvature との関係を整理する必要がある。

5. **次の理論課題は、**
   **(i) current Q-based、(ii) ideal exact-EM Q（exact posterior での `E[ln p(Z,X,Y|θ)]`）、
   (iii) Laplace-marginal approximation（`Z` を積分した近似）**
   **の 3 者で K 依存項がどう変わるかを、数式レベルで比較することである。**
   本監査はその比較のための材料（各項の厳密な形・オーダー・実測値）を揃えたが、比較自体は行っていない。
   これは新実験を要さない解析課題であり、実験設計より先に置ける。
