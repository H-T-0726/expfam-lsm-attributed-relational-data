# K_true 回復のための K 選択基準 — 設計比較（Phase 9G / Issue #86）

- 日付: 2026-09-28
- 位置づけ: **THEORY / DESIGN COMPARISON**。EM 0、simulation 0、候補の実装 0、現行 C_Q の変更 0、コード変更 0。
- baseline: `origin/main = 2261b31`
- ラベル: **FACT**（repository のコード・原論文から直接確認）/ **DERIVED**（本報告の導出。仮定を明記）/
  **LIT**（外部文献。原典本文を確認したもののみ最終判断の根拠に使う）/ **INTERPRETATION** / **UNRESOLVED**

## 0. 確認した一次資料

| 資料 | 確認方法 | 最終判断に使うか |
|---|---|---|
| 現行コード（`em_runner.py`, `eval_utils.py`, `reproduction/src/model.py`） | 本文を読んだ | 使う（FACT） |
| 原論文 Mikawa et al. 2024（`paper/…binary_rel.pdf`） | 本文テキスト | 使う（FACT） |
| Phase 9F 報告、Phase 9E summary | 本文 | 使う |
| Drton & Plummer (2017) *JRSS B*, arXiv:1309.0911v3 | **原典本文**（§1, §2） | 使う: 正則 BIC の前提、特異モデル、因子分析 |
| Minka (2000) *Automatic choice of dimensionality for PCA*, NeurIPS | **原典本文**（§2–3） | 使う: 潜在次元選択を観測データ evidence で行う例 |
| Handcock, Raftery & Tantrum (2007) *JRSS A* 170 | **原典本文**（§4, 討論） | 使う: 潜在位置ネットワークモデルで Ẑ に条件づける根拠とその適用範囲 |
| Daudin, Picard & Robin (2008) *Stat. Comput.* 18 | **原典本文**（Prop. 8） | 使う: ノード量とペア量で異なる log 項を使う ICL の例 |
| Celeux (CLADAG 2015) *On the different ways to compute ICL* | 本文 | 補助（ICL の定義・BIC との関係） |
| Schwarz (1978), Kass & Raftery (1995) | PDF は画像スキャンで本文を読めなかった | **使わない**（Drton & Plummer の要約で代替） |
| Biernacki, Celeux & Govaert (2000) | HAL の PDF の文字が化けていて読めなかった | **使わない**（Celeux 2015 の要約のみ参照） |

---

## 1. K 選択の目標（固定）

**生成モデルの母集団レベルの真の潜在次元 K_true / K_0 を回復すること**（Issue #86）。
「明瞭な次元だけ数える」ICL 型の目標ではない。この目標は比較の後で変更しない。

記法: `Φ_K(Z; θ) = ln p(Z) + ℓ_X(Z; θ) + ℓ_Y(Z; θ)`、`ln p(Z) = −(nK/2) ln 2π − ½‖Z‖²`（z_i ~ N(0, I_K)）。
現行（Phase 9F）: `C_Q = −2 E_q[Φ_K] + p_K ln n = D_K + P_Z + P_θ`。

---

## 2. Candidate A — 原論文型の条件付き基準

**FACT（原論文）**: Eq.(16) の `L` は `Π_i p(x_i | F, z_i, …) Π_{i≠j} p(y_ij | z_i, z_j, …)^{1/2}` で、**z に条件づけられ `p(Z)` を含まない**。
Eq.(26) は `BIC = −2 ln L + ((k+1)d − k(k−1)/2) ln n`。論文は BIC 採用の動機として一致性を述べ、Experiment 2 で k = k* が選ばれるかを経験的に確認している（証明はない; Phase 9F）。

一般形: `C_cond(K) = −2 [ℓ_X(Ẑ) + ℓ_Y(Ẑ)] + pen_θ(K)`。

- **現行分解との関係（DERIVED）**: Ẑ を E-step サンプルで平均すると `−2(ℓ_X + ℓ_Y) = D_K`。よって `C_cond = D_K + P_θ`。
  **P_Z は Z に条件づけるので消える。**
- **尺度不変性（DERIVED）**: `(Z, F, w) → (σZ, F/σ, w/σ²)` で `ℓ_X, ℓ_Y` は不変 → 値も K の順序も不変。
- **Ẑ への依存**: 値は Ẑ の選び方（mode / 事後平均 / サンプル）で変わる。事後平均は縮小、サンプルはノイズを含む。どれが正しいかを決める原理はこの基準自体にない。
- **K_true 回復との関係（DERIVED）**: Ẑ を固定するということは、nK 個の潜在座標をパラメータとして当てはめながら、`pen_θ` では数えないことを意味する。
  K を増やせば当てはめる座標が n 個ずつ増えるのに罰則は θ の分だけなので、**Z についての Occam 効果がない**。
  nK は n とともに増える（incidental parameters）ので、正則 BIC の枠組みでも扱えない。
- **LIT（Handcock et al. 2007, §4 本文）**: 潜在位置ネットワークモデルで Ẑ に条件づけた BIC を使うが、その理由の 1 つとして
  「クラスタ数を比べるとき、**条件づける潜在位置の次元はクラスタ数によらず同じ**」を挙げている。
  **K（潜在次元そのもの）を比べる場合はまさにこの前提が崩れる。** また同論文は潜在空間の次元を「ユーザーが指定する」としている。
- **結論**: 尺度不変ではあるが、**K_true 回復を目標とする理論的根拠がない**。原論文と似ていることは正当化にならない。

---

## 3. Candidate B — Z を積分した観測データ尤度（Laplace 近似）

目標: `ln p(X, Y | θ, K) = ln ∫ exp Φ_K(Z; θ) dZ`（nK 次元の積分）。

### 3.1 導出（DERIVED）

Ẑ を Z についての**同時**最頻値（`∇_Z Φ_K = 0`）、`H = −∇²_Z Φ_K(Ẑ)`（nK × nK）とすると

```
ln p(X,Y | θ,K) ≈ Φ_K(Ẑ) + (nK/2) ln 2π − ½ ln|H|
               = ℓ_X(Ẑ) + ℓ_Y(Ẑ) − ½‖Ẑ‖² − ½ ln|H|             （(nK/2) ln 2π は prior の定数と相殺）
```

θ の積分に BIC 型近似を使うと、候補

```
C_Lap(K) = −2[ℓ_X(Ẑ) + ℓ_Y(Ẑ)] + ‖Ẑ‖² + ln|H(Ẑ)| + d_K ln N
```

（`d_K` と N は §7–8）。

**H の構造（DERIVED, canonical Bernoulli-Y, `η_ij = w0 + w z_iᵀ z_j`）**:
- 対角ブロック `H_ii = I + Fᵀ V_X(i) F + w² Σ_{j≠i} V_Y(ij) z_j z_jᵀ` ── **現行の `A_i` と同じ量**（FACT: `_calc_precision_matrix`、正則化 1e-6 I を除く）
- 非対角ブロック `H_ij = w² V_Y(ij) z_j z_iᵀ − w (y_ij − μ_ij) I_K`（i ≠ j）── **Y がノードを結合するので非ゼロ**

### 3.2 性質

- **尺度不変性（DERIVED）**: `Z' = σZ`（事前分散 σ²）とすると `Φ' = Φ − nK ln σ`、`H' = H/σ²` → `−½ ln|H'| = −½ ln|H| + nK ln σ`。
  相殺して**値も K の順序も厳密に不変**（Laplace 近似は線形の変数変換で不変）。
- **現行分解との関係（DERIVED）**: `P_Z = nK(1+ln 2π)` は**データ依存の Laplace 体積項 `‖Ẑ‖² + ln|H|` に置き換わる**（`ln 2π` の定数は厳密に相殺）。
  `D_K` はサンプル平均ではなく同時最頻値 Ẑ で評価する量になる。
- **K が変わるときの nK 次元項**: `ln|H|` は nK 個の固有値の対数和。1 次元加えると n 個の固有値が増え、その大きさはノードあたりの情報量
  （X から d 列分、Y から n−1 ペア分）で決まる。**データが決める Occam 因子**であり、P_Z のような固定値ではない。
- **block vs joint Hessian（DERIVED / UNRESOLVED）**: `ln|H| = Σ_i ln|H_ii| + ln|D^{-1/2} H D^{-1/2}|`（D = blockdiag）。
  第 2 項は Y による結合の寄与で、各非対角ブロックは O(1) でも各行に n−1 個あるため、**無視できる大きさだとは示せない**。
  n = 75, K ≤ 5 なら H は最大 375 × 375 で、**同時 Hessian をそのまま計算できる**。block 近似に頼る理由はない。
- **mixed X・relational Y への適用**: Φ は既存の尤度関数（per-column X, Bernoulli/Poisson/Gaussian）の和なので、そのまま適用できる。
  必要なのは Ẑ の同時最頻値と H の組み立てで、モデルは変わらない。

### 3.3 残る近似（UNRESOLVED）

1. **nK 次元の Laplace 近似の精度**: 積分の次元が n とともに増える。精度はノードあたりの情報量に依存し、
   Y が情報を持つ（w ≠ 0）ときはノードあたり O(n)、X だけならノードあたり O(d) に留まる。K ごとに誤差が違えば順序に効く。
2. **Z の事後の対称性・多峰性**: `ℓ_Y` と事前は `Z → ZR`（直交）で不変で、対称性を破るのは `ℓ_X`（F を通す）だけ。
   X が弱いとき、Z の事後は Y 由来のほぼ平坦な方向を持ちうるので、単一モードでの Laplace は体積を誤る可能性がある（Minka 2000 も PPCA で複数の極大に対処している; LIT）。
3. **θ̂ の質**: Laplace＋BIC は θ̂ が `ln p(X,Y|θ)` の最大化点であることを前提にする。実装の θ̂ は 8 反復の MCEM の出力で、完全な最尤ではない。
4. θ の積分に対する BIC 近似の妥当性（§7–9）。

---

## 4. Candidate C — ELBO に基づく基準

`ELBO(q) = E_q[Φ_K] + H(q) ≤ ln p(X,Y|θ)`（差は `KL(q ‖ p(Z|X,Y,θ))`）。
候補: `C_ELBO(K) = −2 ELBO(q) + d_K ln N`（これを BIC とは呼ばない）。

- **現行 C_Q との違い（DERIVED）**: `C_Q = −2 E_q Φ + P_θ` にエントロピー項 `−2H(q)` を戻したもの。
  独立 Gaussian `q = Π_i N(m_i, A_i^{-1})` なら `H(q) = Σ_i [(K/2)(1+ln 2π) − ½ ln|A_i|]` で、scale_Z の下では
  `C_ELBO = D_K + Σ_i ln|A_i| + P_θ` → **P_Z はエントロピーの定数部分と相殺し、`Σ ln|A_i|` に置き換わる**。
- **尺度不変性（DERIVED）**: q を明示的な分布族（尺度変換で閉じた Gaussian 族）として定義し、Z と一緒に変換するなら、
  `E_q ln p(Z)` と `H(q)` のずれ（∓ nK ln σ）が相殺して**不変**。
- **実際のサンプラーとの関係（FACT/DERIVED）**: 実装の E-step は Z_{\i} に条件づけた逐次 Laplace でサンプルを作り、その後 scale_Z で一括スケールする。
  **そのサンプルの分布は閉じた形の密度を持たず、H(q) を厳密に計算できない**。C を使うには q を明示的に定義し直す必要がある
  （例: 固定点での `Π_i N(m_i, A_i^{-1})`）。その場合、期待値は q から改めて MC で評価する。
- **B との関係（DERIVED）**: q を同時 Gaussian `N(Ẑ, H^{-1})` にとると、2 次まででは `E_q Φ ≈ Φ(Ẑ) − nK/2`、`H(q) = (nK/2)(1+ln 2π) − ½ ln|H|` なので
  `ELBO ≈ Φ(Ẑ) + (nK/2) ln 2π − ½ ln|H|` = **B の Laplace 近似と一致する**。block 対角の q は B の block 近似に対応する。
- **K ごとの近似誤差（DERIVED）**: C は下界であり、ギャップ `KL(q‖posterior)` は K と Y の結合の強さに依存する。
  **下界どうしの比較は、ギャップが小さい K に有利に偏る**。さらに L = 5 本の MC 平均の誤差は `E_q Φ` の分散（nK 次元の和）に比例し、
  K によって異なる。
- **観測尤度の代理としての意味**: 下界としては正しいが、K 間比較の量としては「ギャップが K によらない」という追加の仮定を要する。

---

## 5. 尺度不変性の比較（必須チェック 1）

変換 `z → σz, F → F/σ, w → w/σ², z ~ N(0, σ²I)`（観測モデルは同一）。

| 基準 | 各 K の値 | K の順序 | 根拠 |
|---|---|---|---|
| 現行 C_Q | **不変でない**（`+nK ln σ²`） | **不変でない**（K に比例するずれ） | Phase 9F §4 |
| A: C_cond | 不変 | 不変 | Z に条件づけ、`ℓ_X, ℓ_Y` が不変 |
| B: C_Lap | 不変 | 不変 | Laplace は線形変換で不変（ヤコビアンが `ln|H|` と相殺） |
| C: C_ELBO | 不変（q を明示的に定義した場合） | 不変 | `E_q ln p(Z)` と `H(q)` のずれが相殺 |

`P_θ` は 3 候補とも同じパラメータ数を数えるので、この変換で不変である。

## 6. 現行 `C_Q = D_K + P_Z + P_θ` との関係（必須チェック 2）

| 項 | A | B | C |
|---|---|---|---|
| `D_K` | そのまま（Ẑ で評価） | 同時最頻値 Ẑ で評価した `−2(ℓ_X + ℓ_Y)` | そのまま（q の期待値） |
| `P_Z` | **Z に条件づけるので消える** | **データ依存の Laplace 体積 `‖Ẑ‖² + ln|H|` に置き換わる**（`ln 2π` は相殺） | **エントロピーの定数部分と相殺し、`Σ ln|A_i|`（または `ln|Σ_q^{-1}|`）が残る** |
| `P_θ` | 残る | 残る（N は §8） | 残る（N は §8） |

## 7. パラメータ数（必須チェック 3）

| パラメータ | 数 | K 依存 | 備考 |
|---|---|---|---|
| F（loadings） | `K d − K(K−1)/2` | **あり** | `Z → ZR, F → FR`（R 直交）で X・Y とも不変なので O(K) の次元を引く（KI-010 (i)） |
| Gaussian-X 分散 | Gaussian 列の数 | 割当が固定なら **なし** | 割当が K ごとに変わると数も変わる（family 依存） |
| Y: `w0, w`（Gaussian-Y なら `σ_Y²`） | 2（3） | なし | 全 K で同数 |
| family 依存 | Bernoulli と Poisson は同じ loading 次元 K、分散なし | なし | 両者でパラメータ数の差はない |
| 離散的な family 選択のコスト | — | — | **UNRESOLVED**。本 phase では罰則を追加しない |

**K に依存しないパラメータの扱い（DERIVED）**: 全 K で同じ数・同じ N を使えば K 間の差で相殺し、**K の順序には影響しない**。
絶対値を解釈するときだけ N の選び方が効く。

## 8. 有効サンプルサイズ（必須チェック 4）

LIT（Drton & Plummer §2 本文）: 正則 BIC の `(d/2) ln n` は、**n 個の i.i.d. 観測**のもとで、MLE 近傍で対数尤度が負定値 2 次形式で近似でき、
その Gaussian 積分の逆分散が「n × Fisher 情報」になることから出る。誤差は `O_p(1)`（Laplace なら `O_p(n^{-1/2})`）。
LIT（Daudin et al. 2008 Prop. 8 本文）: SBM の ICL は、ペアの量（接続確率）に `log(n(n−1)/2)`、ノードの量（混合比）に `log n` を使う。
**パラメータ群ごとに N が異なりうる**ことを示す例である。

| 候補 | Z の扱い | K 依存の θ（F）に使う N | K 非依存の θ | 状態 |
|---|---|---|---|---|
| A | 条件づけ（罰則なし） | n（各列の loading は n 個の値から推定） | ペア量 `n(n−1)/2` が自然だが順序に無関係 | Z の nK 座標が罰則の外にあるので、N を決めても基準は完結しない |
| B | Laplace で積分（`ln N` を使わない） | **n**（DERIVED, 経験則的） | 順序に無関係 | 周辺化後はノードが i.i.d. でない（Y が結合）ので、正則 BIC の前提は厳密には成り立たない。**UNRESOLVED** |
| C | 下界の中で積分 | B と同じ | 順序に無関係 | B と同じ |

**DERIVED**: B/C では、sample size が本当に問題になる nK 個の潜在座標を `ln N` ではなく Laplace 体積（データ依存）で扱う。
そのため、残る `ln N` は F に対するものだけになり、F の各列は n 個のノード値から情報を得るので n を使う根拠がある。
ただし「Y を周辺化した後の依存構造の下でも F の情報量が n に比例する」ことは示していない（UNRESOLVED）。
`nd` や `n(n−1)/2` を K 依存の罰則に使う根拠は見つからなかった。

## 9. 特異性（必須チェック 5）

- **X 側（DERIVED）**: K > K_true では、余分な潜在次元に対応する F の列が母集団でゼロになりうる。その方向で Fisher 情報が退化し、**特異**になる。
  LIT（Drton & Plummer §1–2 本文）: 因子分析・reduced-rank regression・潜在クラスは特異モデルで、正則 BIC の前提（2 次近似）を満たさない。
  ただし標準 BIC は「多くの特異な設定で一致的であることが知られている」と同論文は述べている（Keribin 2000 等を引用; 本報告では原典未確認）。
- **Y 側（DERIVED, repository 固有）**: canonical Y は `η_ij = w0 + w Σ_{k=1}^K z_ik z_jk` で、**w はスカラー、事前分散は各次元 1 に固定**である。
  K+1 次元モデルの追加座標 u_i ~ N(0,1) は `w u_i u_j` として η に必ず入り、これを消すパラメータ値は w = 0（全次元を消す）しかない。
  したがって **Y 成分では K モデルは K+1 モデルに入れ子になっていない**。
  K > K_true は「冗長な特異方向」というより「Y の分布の誤指定」に近い。
- **帰結**: 正則 Schwarz BIC の前提は、X 側の特異性と Y 側の非入れ子性の両方で満たされない。
  これが K_true 回復にどう効くか（過大 K が Y の誤指定でどれだけ不利になるか、X 側の特異性で `d_K ln n` がどれだけ過大か）は **UNRESOLVED**。
  このモデルの RLCT の導出は本 phase の範囲外。

---

## 10. 実装の実現可能性

| 候補 | 必要なもの | 推定手続きの変更 | 計算量（n=75, K≤5） |
|---|---|---|---|
| A | Ẑ の定義、`ℓ_X + ℓ_Y` | なし | 小 |
| B | ① 最終 θ̂ の下での Z の**同時**最頻値（既存の per-node Newton をサンプリングなしで収束まで反復）、② 同時 Hessian H の組み立て（対角ブロック = 既存 `A_i`、非対角ブロックは新規）、③ `ln|H|`、④ 既存の尤度関数 | **なし**（refit 後の post-hoc 評価） | H は最大 375×375、log-det は容易 |
| C | q の明示的定義、q からの MC 期待値、H(q) | なし（ただし q の再定義が必要） | 小〜中（MC 誤差の管理が要る） |

## 11. 判定表（点数はつけない。上から順に判定）

| 判定基準（優先順） | A: 条件付き | B: 観測尤度の Laplace | C: ELBO |
|---|---|---|---|
| 1. K_true 回復の目標に合う | **合わない**（Z の Occam 効果なし・incidental） | **合う**（観測データ evidence が対象; LIT Minka 2000 は evidence を真の次元選択に使い、データが十分なら真のモデルを選ぶと述べる） | 部分的（下界。ギャップが K に依存） |
| 2. 潜在尺度に不変 | 不変 | 不変 | 不変（q を明示的に定義した場合） |
| 3. K 間比較として解釈可能 | 不可 | 可（ln p(X,Y|K) の近似。誤差は §3.3） | 条件付き（ギャップが K によらない仮定が要る） |
| 4. 現行モデルで計算可能 | 可 | 可（同時 Hessian も n=75 で可） | 可（q の再定義が要る） |
| 5. 推定手続きからの変更が小さい | 最小 | 小（post-hoc） | 中 |

## 12. PRIMARY CANDIDATE

## **PRIMARY CANDIDATE: B**（Z を積分した観測データ尤度の Laplace 近似 ＋ θ に対する BIC 型項）

理由（判定順）:
1. **目標が K_true 回復と一致する**: 観測データ evidence `p(X,Y|K)` の近似であり、潜在次元選択の標準的な対象（LIT: Minka 2000 の PPCA 次元選択、Drton & Plummer の BIC 背景）。A は Z に Occam 効果がなく、C は K に依存するギャップを持つ下界である。
2. **潜在尺度に厳密に不変**（現行 C_Q の Phase 9F での問題点を構造的に解消する）。
3. **K 間比較の意味が明確**: `P_Z` がデータ依存の Laplace 体積に置き換わり、固定の `nK(1+ln 2π)` を採点しない。
4. **計算可能**: 同時 Hessian も現在の規模で直接計算できる。
5. **推定手続きを変えない**: 既存 refit の θ̂ に対する post-hoc 評価で済む。

Phase 9E で K=3 が増えるかどうかは判断に一切使っていない。

**限定（B が「準備完了」である範囲）**: B は理論的に最も筋の良い候補であり、最小実装に進む根拠はある。
ただし §3.3 の近似（nK 次元 Laplace の精度、Z の事後の対称性、θ̂ が MCEM 出力であること）と §8–9（N、特異性・非入れ子性）は **UNRESOLVED** で、
**B が K_true を一致的に回復することは示していない**。

## 13. CAN SAY / CANNOT SAY

**CAN SAY**
- 現行 C_Q は潜在尺度の慣行に対して不変でなく、候補 A・B・C はいずれも（C は q を明示的に定義すれば）不変である（DERIVED）。
- A（原論文型の Z 条件付き基準）は Z の nK 座標を罰則の外で当てはめるので、K_true 回復の理論的根拠を持たない（DERIVED; LIT Handcock et al. の条件づけの根拠は潜在次元が一定の比較に限られる）。
- B では `P_Z` がデータ依存の Laplace 体積に置き換わり、C では P_Z がエントロピー定数と相殺する（DERIVED）。
- K 依存のパラメータは F（`Kd − K(K−1)/2`）であり、`w0, w` と（割当固定時の）Gaussian 分散は K の順序に影響しない（DERIVED）。
- canonical Y（スカラー w、単位事前分散）では K モデルは K+1 モデルに入れ子になっていない（DERIVED）。

**CANNOT SAY**
- B（または他の候補）が K_true を一致的に回復すること。
- B が Phase 9E の条件で現行 C_Q より良く K_true を選ぶこと（未実装・未評価）。
- F の罰則の N = n が、Y で結合した周辺化モデルの下で厳密に正当化されること。
- 過大 K での特異性・非入れ子性が K 選択に与える影響の大きさ。
- 離散的な family 選択のコストの扱い。
- Schwarz (1978)、Kass & Raftery (1995)、Biernacki et al. (2000) の原典に基づく主張（本文を確認できていない）。

## 14. 次に必要な最小の実装・pilot（Human 判断待ち; 本 phase では着手しない）

1. **zero-EM の関数実装**: 既存の θ̂ を入力として `C_Lap(K)` を計算する純関数（Z の同時最頻値、同時 Hessian、`ln|H|`、既存尤度）。
   同時 Hessian の非対角ブロックは有限差分と照合する。
2. **厳密解がある場合の照合（zero-EM）**: Gaussian-X のみ・w = 0 では事後が厳密に Gaussian で、`ln p(X|θ)` は閉じた形（因子分析の周辺尤度）になる。
   `C_Lap` の Laplace 部分がこれと一致することを、固定した θ で確認する（simulation ではなく代数的照合）。
3. 上の 2 つが通れば、**Phase 9E と同条件の小さな pilot で、現行 C_Q と C_Lap を同じ refit 上で並べる**（新しい EM が必要。frozen protocol と Human Gate が必要）。
   その際、判定は事前に固定し、C_Lap の結果を見て罰則を調整しない。

**最大の UNRESOLVED**: nK 次元（n とともに増える）の Laplace 近似が、Y の結合と Z の事後の近似的な回転対称性のもとで、
**K ごとに異なる誤差を持たずに `ln p(X,Y|θ,K)` を近似できるか**。これが崩れると、B の K 順序は近似誤差に支配されうる。

---

### 参照

repository（FACT）: `expfam/src/experimental/em_runner.py`、`expfam/src/experimental/eval_utils.py`、`reproduction/src/model.py`（`calc_eta_newton`、`_calc_precision_matrix`、`scale_Z`）、
`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`（Eq.(14), (16)–(19), (26), §4.3）、
`reports/k_selection_theory/cq_theoretical_clarification_20260927.md`、`reports/distribution_selection/phase9e_k_repeatability_summary_20260927.md`。

LIT（原典本文を確認）:
- Drton, M., Plummer, M. (2017). A Bayesian information criterion for singular models. *JRSS B* 79(2). arXiv:1309.0911v3（§1, §2, Remark 2.1）。
- Minka, T. P. (2000). Automatic choice of dimensionality for PCA. *NeurIPS 13*（§2, §3, §3.2）。
- Handcock, M. S., Raftery, A. E., Tantrum, J. M. (2007). Model-based clustering for social networks. *JRSS A* 170(2), 301–354（§4, p.307–308; 討論）。
- Daudin, J.-J., Picard, F., Robin, S. (2008). A mixture model for random graphs. *Stat. Comput.* 18, 173–183（Proposition 8）。
- Celeux, G. (2015). On the different ways to compute the integrated completed likelihood criterion. CLADAG 2015（§1–2）。

LIT（本文を確認できず、判断根拠に使っていない）: Schwarz (1978) *Ann. Statist.* 6; Kass & Raftery (1995) *JASA* 90; Biernacki, Celeux & Govaert (2000) *IEEE TPAMI* 22; Keribin (2000); Watanabe (2009)。
