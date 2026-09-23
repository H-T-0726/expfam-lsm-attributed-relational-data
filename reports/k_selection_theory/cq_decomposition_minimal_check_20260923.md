# C_Q(K) 分解の最小確認 — 現行実装における恒等性チェック

- 日付: 2026-09-23
- Issue: #72（parent #71 / Phase 9A）
- branch: `audit/72-k-criterion-minimal-check`
- baseline: `origin/main = f5a0e8f17a50b7b41371fb2495facd960091978a`
- scope: READ / DERIVE / DOCUMENT ONLY。新規 EM fit 0 件、scientific code 変更 0 行、artifact 再生成 0 件。

**この文書は K 選択理論の再研究ではない。**
指導教員の整理 `C_Q(K) = D_K + P_Z(K) + P_θ(K)` が、現行実装の
`C_current(K) = −2 Q_strict(K) + p_K ln n` と**実装上の恒等式**になるかだけを確認する。

`calc_bic_dual` が Schwarz BIC ではなく Q-based complete-data / ICL-type criterion である
ことは既に canonical に確定しており（root `CLAUDE.md` §5、`KNOWN_ISSUES.md` D 項、
`RESEARCH_MASTER.md` §12.6）、本確認では再証明しない。

---

## 1. current code が実際に計算している式

対象は標準系列 `expfam/src/utils_expfam.py`（`DualExpFamLSM`）。この基準関数は
`model_dual_expfam.py`（0.5 あり）と `model_dual_expfam_fixed.py`（0.5 なし）の
両系列から共通に呼ばれる。

### 1.1 `calc_bic_dual`（L.386–404）

```
p_K  = k*d − k(k−1)/2  +  d*1{family_x = gaussian}  +  1*1{family_y = gaussian}
BIC  = −2 * Q_strict + p_K * ln n
```

`n` は **node 数**（`n, d = X.shape` の `n`）。`w_0^Y`, `w^Y` は NOLTA 2024 慣行として
数えない（KI-010、未検証のまま踏襲）。

### 1.2 `calc_Q_dual_strict`（L.355–378）→ `calc_Q_dual`（L.324–352）

```
Q_dual   = (1/L) Σ_{l=1..L} [ lnpZ(Z^(l)) + lnpX(X | Z^(l), F) + lnpY(Y | Z^(l), w0, w) ]

Q_strict = Q_dual
           − 1{family_y = poisson} * Σ_{i<j} ln(y_ij!)
           − 1{family_x = poisson} * Σ_{i,l} ln(x_il!)
```

構成要素:

| 項 | 実装 | 式 |
|---|---|---|
| `lnpZ` | `_lnpZ`（`utils_expfam.py` L.315） | `−(nk/2) ln(2π var_z) − (1/(2 var_z)) Σ_{i,q} (Z^(l)_{iq})^2` |
| `lnpX` | `DualExpFamLSM.calc_log_likelihood_X`（`model_dual_expfam.py` L.291） | Gaussian: `Σ_{i,l} [−½ r^2/σ_l^2 − ½ ln σ_l^2 − ½ ln 2π]` / Bernoulli: `Σ [x lnS + (1−x) ln(1−S)]` / Poisson: `Σ [x η − e^η]`（`−ln x!` は strict 側で加算） |
| `lnpY` | `ExpFamLatentStructuralModel.calc_log_likelihood_Y`（`model_expfam.py` L.241） | 対角を 0 にした対称行列の `0.5 * Σ`（＝一意ペア `i<j` の和）。Gaussian: `−½(y−η)^2/σ_Y^2 − ½ ln σ_Y^2`（**`−½ ln 2π` を含まない**） |

`var_z = 1.0` に固定（`reproduction/src/model.py` L.99。M-step で更新されない）。

---

## 2. Q_X / Q_Y / Q_Z の対応

`Q_strict` は l に関する平均の線形性により、次の 3 つに**排他的かつ網羅的に**分解できる。

```
Q_Z = (1/L) Σ_l lnpZ(Z^(l))
Q_X = (1/L) Σ_l lnpX(X | Z^(l), F)      − 1{family_x = poisson} Σ_{i,l} ln(x_il!)
Q_Y = (1/L) Σ_l lnpY(Y | Z^(l), w0, w)  − 1{family_y = poisson} Σ_{i<j} ln(y_ij!)

Q_strict = Q_X + Q_Y + Q_Z
```

Poisson の階乗補正は `l` に依存しない定数なので、`(1/L)Σ_l` の内側・外側どちらに置いても同値。
実装は外側（`calc_Q_dual_strict` の `corr`）に置いており、X 側・Y 側の補正を
それぞれ `Q_X` / `Q_Y` に一意に帰属できる形になっている。

### 2.1 `Q_Z` は K に対する決定論的な線形項に退化している

EM ループ内で Q 計算に使う `Z_samples` は毎反復 `model.scale_Z()` を通る
（`utils_expfam.py` L.514）。`scale_Z` は全要素の平均二乗を 1 に正規化する
（`reproduction/src/model.py` L.495–504、`scale = 1/sqrt(mean(Z^2))`）。したがって

```
Σ_l Σ_{i,q} (Z^(l)_{iq})^2 = n*k*L   （厳密）
```

が成立し、`var_z = 1` と合わせて

```
Q_Z    = −(nk/2) ln(2π) − nk/2 = −(nk/2)(1 + ln 2π)
P_Z(K) = −2 Q_Z = n*k*(1 + ln 2π) ≈ 2.8379 * n * k
```

となる。**`P_Z(K)` はデータにも推定値にも依存せず、K に厳密比例する固定罰則である。**
これは `RESEARCH_MASTER.md` §12.6 項 2 / `KNOWN_ISSUES.md` D 項 2 の記述と一致する。

---

## 3. D_K / P_Z / P_theta の対応表

| 指導教員の記法 | 定義 | 現行実装の対応物 | K 依存性 |
|---|---|---|---|
| `D_K` | `−2 { Q_X(K) + Q_Y(K) }` | `−2 ×`（`calc_Q_dual` の `lnpX + lnpY` の L 平均 ＋ Poisson 階乗補正） | データ依存・推定依存 |
| `P_Z(K)` | `−2 Q_Z(K)` | `−2 ×` `_lnpZ` の L 平均 ＝ `n*k*(1 + ln 2π)` | **K に厳密比例の決定論的定数** |
| `P_θ(K)` | `p_K ln n` | `calc_bic_dual` の `num_params * np.log(n)` | `p_K = kd − k(k−1)/2 + d*1{GX} + 1*1{GY}` |
| `C_Q(K)` | `D_K + P_Z + P_θ` | `calc_bic_dual` の戻り値 `bic` | — |

---

## 4. exact equality が成立するか

### **YES（代数的恒等式として成立）**

```
C_current(K) = −2 Q_strict + p_K ln n
             = −2 (Q_X + Q_Y + Q_Z) + p_K ln n
             = [−2(Q_X + Q_Y)] + [−2 Q_Z] + [p_K ln n]
             = D_K + P_Z(K) + P_θ(K)
```

分解は `Q_strict` の定義から直接従い、再定義・近似・追加仮定を一切必要としない。
新しい criterion を導入していないため、過去の `BIC` 列の数値はそのまま `C_Q(K)` として読める。

**限定条件（誠実に記録する）:**

- これは**浮動小数点上の bitwise 一致の主張ではない**。現行コードは `lnpZ + lnpX + lnpY` を
  1 つの累算で合計しているため、3 項を別々に集計して足し直すと加算順序の違いによる
  丸め差が出うる。**代数的には厳密、数値的には丸め誤差の範囲で一致**。
- 分解は `Q_strict` の分解であって、`Q_strict` 自体が観測データ周辺尤度であることは
  依然として主張できない（KI-010 / RM §12.6）。`C_Q` という記法に変えても
  criterion の位置づけは変わらない。

### 4.1 session validation（pure-function 再照合）

§4 の主張は式の読み取りだけに依らず、**実関数を直接呼ぶ算術照合**でも確認した。
EM 反復・推定・データ生成・artifact 書き出しは一切行っていない。

手順（再現するには以下をそのまま行えばよい。**照合スクリプトは commit していない**ので、
下の数値は repository artifact ではなく session validation の記録である）:

1. `DualExpFamLSM` を `n=24, d=6, k=3, L=5` で構築し、`model.scale_Z` に通した乱数 `Z_samples`、
   乱数 `F`、乱数対角 `sigma`、対称化した `Y` を与える。
2. `calc_Q_dual_strict` と `calc_bic_dual` の戻り値を、§2 の定義に従って
   `Q_Z` / `Q_X` / `Q_Y` を別々に集計して再合成した `D_K + P_Z + P_θ` と比較する。
3. `(family_x, family_y) = (gaussian, bernoulli) / (poisson, poisson) /
   (bernoulli, gaussian) / (gaussian, gaussian)` の 4 組で実行する。

結果: 4 組すべてで残差は **`|ΔC| ≤ 5e-13`**（`C_current` の絶対値は `1.2e3`–`2.5e3` 程度）。
§4 の「代数的には厳密、数値的には加算順序による丸め差の範囲で一致」という記述と整合する。

`P_Z` についても同様に `scale_Z` 後の `_lnpZ` を直接評価し、
`P_Z/(nK) = 2.8378770664…= 1 + ln 2π` が `(n,k,L)` を変えても厳密に成り立つことを確認した
（残差 `≤ 3e-14`）。§2.1 の導出と一致する。

---

## 5. Phase 9 で候補 K を自動比較するために接続できるか

**接続できる。** 現行 API のままで、候補 `K = 1,…,K_max` それぞれについて

```
run_em_dual(..., k=K, compute_strict_Q=True) -> Q_strict
calc_bic_dual(Q_strict, K, n, d, family_x, family_y) -> (C_Q(K), p_K)
K_hat = argmin_K C_Q(K)
```

が成立する。`exp_scenario_lib.py` L.86 と `run_common_realdata_reconstruction_eval.py` L.306 が
既にこの呼び出し形になっており、新しい配線を作る必要はない。

接続時に注意すべき点:

1. **Y 側 Gaussian の `−½ ln 2π` が標準系列では落ちている**
   （`model_expfam.py` L.241–264）。ペア数は K に依存しないので `argmin_K` には影響しないが、
   **絶対値は「完全な log density」ではない**。experimental 系列の
   `eval_utils.calc_Q_dual_strict_exp`（L.227–228）はこの定数を加えているため、
   **標準系列と experimental 系列の `BIC` 絶対値は同一条件でも一致しない**
   （KI-002 の系列混在禁止がそのまま効く）。
2. `calc_log_likelihood_X`（Gaussian）の docstring は「`−nd/2 ln(2π)` を省く」と書いているが、
   実コードは要素ごとに `−½ ln 2π` を**加えている**。docstring が実装より古い。
   （本 Issue は READ ONLY のため修正しない。）
3. `calc_Q_dual` の引数 `sigma` は関数内で使われていない。Gaussian-X の `σ_l^2` は
   `model.params["sigma"]` から読まれる（`model_dual_expfam.py` L.315）。
   `model.params["sigma"]` が最後に更新されるのは各反復の **E-step 冒頭**
   （`utils_expfam.py` L.498–499）であるのに対し、`F` は当該反復の M-step 結果が渡される。
   したがって最終 `Q_strict` の Gaussian-X 項は **Σ_X が F より 1 M-step 古い**。
   全候補 K に同じ扱いが適用されるため K 順位への系統的影響は想定しにくいが、
   **Phase 9B で `Q_X` を family 比較スコアとして再利用する場合はこのラグが直接効く**（§6・§7）。
4. `P_Z(K) ≈ 2.8379 n K` は `P_θ(K) = p_K ln n` と**同じ向きの K 罰則**として二重に効いている。
   これは現行 criterion の仕様であって bug ではないが、`C_Q` の 3 項分解で表示すると
   「罰則が 2 本ある」ことが可視化される。Phase 9 で K を自動選択する際に
   この 2 本を勝手に統合・削除しないこと（criterion 変更は Human Gate）。

---

## 6. family 自動選択後に parameter count で注意が必要な点

`calc_bic_dual` の `family_x` は**スカラー文字列 1 個**であり、列ごとの family を表現できない。
Phase 9 で列ごとに family を自動選択すると、`p_K` の X 側 dispersion 項を
`d*1{family_x = gaussian}` では表せなくなる。

repository には既に対応物がある:

- `expfam/src/experimental/eval_utils.py` `calc_bic_exp`（L.232–256）は
  `family_x='mixed'` ＋ `n_gaussian_x_cols` を受け取り、**Gaussian 列数だけ** dispersion を数える。
- 対応する Q も `calc_Q_dual_strict_exp`（L.186–229）が per-column Poisson 階乗補正を持つ。
- per-column prototype 側にも「`utils_expfam.calc_bic_dual` / `calc_Q_dual_strict` は使用不可
  （mixed を知らない）」と明記されている（`model_dual_expfam_percolumn.py` L.22–26）。

したがって **#75 で必要になる調整は「criterion の理論変更」ではなく「per-column 対応版の
parameter count へ配線し直すこと」**である。具体的な注意点:

1. `p_K` の X 側は `Σ_l 1{c_l = gaussian}` に置き換わる。`d*1{GX}` のままでは
   混在列で過大／過小カウントになる。
2. **family 選択自体の自由度（`c_l` の選択コスト）が `p_K` に入っていない。**
   family を data から選んだ後に同じ `p_K` を使うと、選択の自由度が無罰則になる。
   これは #73 / #75 で扱うべき論点であり、**本 Issue では設計しない**。
3. per-column 版を使うと **experimental lineage** になる（prototype・本文採用不可、root `CLAUDE.md` §3）。
   標準系列の過去 `BIC` 値と同じ表・図に並べない。
4. `P_Z(K) = nK(1 + ln 2π)` は family に依存しないため、family 自動選択によって変化しない。
   K 罰則の二重計上（§5-4）は family 選択とは独立した既存事項として残る。

---

## 7. 既知 limitation（再掲のみ・再研究しない）

- `Q_strict` は観測データ周辺尤度ではない。現行基準は Schwarz BIC ではなく
  Q-based complete-data / ICL-type criterion（KI-010、RM §12.6）。呼称制限はそのまま有効。
- model-selection consistency は未証明。K 選択の n 依存性は未測定、本モデルの RLCT は未知。
- effective sample size が未解決: `ln n` の `n` は node 数だが、Y は `n(n−1)/2` ペア、
  X は `n*d` 要素を持つ（KI-010 / RM §12.6 の未解決点）。本確認では何も変更していない。
- Issue #37 は同一 42 fits 上で score 定義により選ばれる K が変わることを示している。
  `C_Q` 分解はこの事実を解消しない。
- Gaussian-Y の `−½ ln 2π` 欠落（§5-1）、Gaussian-X docstring 不一致（§5-2）、
  Σ_X の 1 M-step ラグ（§5-3）は**本 Issue では記録のみ**。修正は別 scope（Human Gate）。
- `w_0^Y`, `w^Y` を `p_K` に数えない慣行は KI-010 のまま未検証。K に依存しない定数なので
  K 順位には影響しない（RM §12.6 項 (ii)）。

---

## 8. Decision

## `PASS_CURRENT_CQ_FOR_PHASE9`

`C_current(K) = D_K + P_Z(K) + P_θ(K)` は現行実装上の代数的恒等式として成立する
（§4、浮動小数点丸めの限定付き）。Phase 9 の候補 K 自動比較は現行 criterion を
そのまま計算する形で接続できる（§5）。criterion の変更は必要ない。

本 Issue の scope に従い、**K 選択理論のこれ以上の調査はここで停止する。**
family 自動選択後の parameter count 配線（§6）は #75 の作業であり、本 Issue では着手しない。

---

## 9. Validation

| 項目 | 結果 |
|---|---|
| 新規 EM fit | 0 |
| simulation | 0（§4.1 は pure-function の算術照合であり、推定もデータ生成も行っていない） |
| scientific code 変更 | 0 行 |
| results / artifacts 変更 | 0 件 |
| 追加ファイル | 本 report 1 件のみ |
| `git diff --check` | clean |

参照した一次証拠:
`expfam/src/utils_expfam.py`（L.315–404, L.478–566）/
`expfam/src/model_dual_expfam.py`（L.274–334）/
`expfam/src/model_expfam.py`（L.241–269）/
`reproduction/src/model.py`（L.99, L.468–504, L.548–）/
`expfam/src/experimental/eval_utils.py`（L.186–256）/
`expfam/src/experimental/model_dual_expfam_percolumn.py`（L.1–27）/
`RESEARCH_MASTER.md` §12.6 / `KNOWN_ISSUES.md` KI-010 ＋ D 項 / root `CLAUDE.md`
