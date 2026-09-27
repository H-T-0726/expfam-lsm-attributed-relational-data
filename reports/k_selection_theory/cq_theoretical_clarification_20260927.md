# 現行 K 選択基準 C_Q(K) の理論的位置づけ（Phase 9F / Issue #84）

- 日付: 2026-09-27
- 位置づけ: **THEORY / AUDIT**。EM 実行 0、simulation 0、criterion 変更 0、コード変更 0。
- baseline: `origin/main = daad101`
- 一次根拠: 現行コード（`expfam/src/experimental/em_runner.py`, `eval_utils.py`, `reproduction/src/model.py`）、
  原論文 PDF（`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`）、#72 報告（検証対象として）、
  Phase 9E 一次 artifact。
- ラベル: **FACT**（コード・論文から直接確認）/ **DERIVED**（本報告での導出。仮定を明記）/
  **LIT**（外部文献由来。repository の事実と区別）/ **OBSERVED**（実験 artifact）/ **INTERPRETATION** / **UNRESOLVED**

---

## 1. Research question

現行 `C_Q(K) = −2 Q_strict(K) + p_K ln n` は、理論的に何を評価しているのか。
特に `P_Z(K) = n K (1 + ln 2π)` と `P_θ(K) = p_K ln n` が同時に K とともに増える構造は、どの model-selection framework に近く、
「二重罰則」「過剰罰則」と呼ぶには何が必要か。

## 2. 現行基準の定義（FACT）

Phase 9D/9E で使った実験系列（`run_em_experimental` → `eval_utils.calc_Q_dual_strict_exp` / `calc_bic_exp`）:

```
Q_strict = (1/L) Σ_l [ ln p(Z^(l)) + ln p(X | Z^(l), F, Σ_X) + ln p(Y | Z^(l), w0, w) ] + (Poisson 階乗補正)
C_Q(K)   = −2 Q_strict + p_K ln n,   p_K = K d − K(K−1)/2 + n_gaussian_x_cols + 1{family_y = gaussian}
```

- `ln p(Z^(l)) = −(nK/2) ln(2π var_z) − (1/(2 var_z)) ||Z^(l)||²`、`var_z = 1` 固定（`reproduction/src/model.py` L.99）。
- `Z^(l)` は E-step の各サンプル: ノードごとに Newton 法で条件付き最頻値 `m_i` を求め、
  `N(m_i, A_i^{-1})`（`A_i` は条件付き負対数事後の Hessian）からサンプルする（`calc_eta_newton`, L.360–462）。
  これは原論文 Eq.(19)–(22) の **Z_{\i} に条件づけた** Laplace 近似である（FACT, 論文 §3.1）。
- `L` 本のサンプルは `scale_Z` で全要素の平均二乗が 1 になるよう一括スケールされる（`em_runner.py`, `model.py` L.468–）。
- `Q_strict` は最終反復の `Z_samples` と M-step 後の θ で評価する。
- 分解（#72, 本報告で再確認）: `C_Q = D_K + P_Z + P_θ`, `D_K = −2(Q_X + Q_Y)`, `P_Z = −2 Q_Z`, `P_θ = p_K ln n`。
  `scale_Z` と `var_z = 1` のもとで `P_Z = nK(1 + ln 2π)` が厳密に成り立つ（#72 §2.1 の導出を検証; 正しい）。

**原論文との関係（FACT, PDF で確認）**: 論文 Eq.(26) は `BIC = −2 ln L + ((k+1)d − k(k−1)/2) ln n`。
ここで `L` は Eq.(16) の尤度 `Π_i p(x_i | F, z_i, …) Π_{i≠j} p(y_ij | z_i, z_j, …)^{1/2}` で、**z に条件づけられ `p(Z)` を含まない**。
論文は Q 関数 Eq.(17)(18) として `E[ln p(Z, X, Y | θ)]` を別に定義している。論文は BIC 採用理由に「BIC は一致性を持つ」を挙げる（§4.3）。
また論文は `σ_z²` を Eq.(14) で推定しているが、本 repository は `var_z = 1` 固定＋`scale_Z` である。
→ **現行 `C_Q` は、論文の印刷式（p(Z) を含まない条件付き尤度）とも、Schwarz BIC（周辺尤度）とも異なる量である。**

## 3. Q1 — `Q_strict` は何の期待値か（DERIVED）

モデル `z_i ~ N(0, I_K)`、`x_il | z_i`、`y_ij | z_i, z_j` のもとで、任意の分布 `q(Z)` に対し恒等式

```
ln p(X, Y | θ) = E_q[ ln p(Z, X, Y | θ) ] + H(q) + KL(q || p(Z | X, Y, θ))          (★)
```

が成り立つ（標準的な変分恒等式; LIT: e.g. Bishop 2006 §9.4）。
`Q_strict` は `E_q[ln p(Z, X, Y | θ)]` の Monte Carlo 推定で、`q` は E-step の Laplace サンプラーの分布である。

- **complete-data log-likelihood の近似事後期待値**（= EM の Q 関数、原論文 Eq.(17)(18)）である。FACT/DERIVED。
- **observed-data likelihood `ln p(X,Y|θ)` ではない**。(★) より両者の差は `H(q) + KL(q||posterior)` で、これは K に依存する。
- **marginal (integrated) likelihood `ln ∫ p(X,Y|θ) p(θ) dθ` でもない**（θ も積分していない）。
- **Schwarz BIC ではない**。Schwarz BIC は `−2 ln p(X,Y|θ̂) + p ln n`（周辺化された観測データ尤度を使う; LIT: Schwarz 1978）。

(★) を `C_Q` に代入すると、`q` を独立 Gaussian `Π_i N(m_i, A_i^{-1})` とみなした場合

```
H(q) = Σ_i [ (K/2)(1 + ln 2π) − ½ ln |A_i| ]
C_Q  = [ −2 ELBO(q) + p_K ln n ] + 2 H(q)
     = [ −2 ELBO(q) + p_K ln n ] + n K (1 + ln 2π) − Σ_i ln |A_i|                    (†)
```

（ELBO = `E_q ln p(Z,X,Y|θ) + H(q)`。**仮定**: q は独立 Gaussian、MC 誤差を無視。実装のサンプラーは Z_{\i} 条件付きの逐次サンプリングなので、
この q は近似的な読み替えである。）

したがって **`P_Z = nK(1+ln 2π)` は、近似事後分布の微分エントロピー `2H(q)` の定数部分とちょうど一致する**。
`scale_Z` のもとでは (†) と `C_Q = D_K + P_Z + P_θ` を比べて `−2 ELBO = D_K + Σ_i ln|A_i|` となる。

### 「ICL-type」と呼ぶことの妥当性

LIT（Biernacki, Celeux & Govaert 2000）: ICL は `ln p(X, ẑ)` を BIC 型に近似したもので、`ICL ≈ BIC + 2 × (分類エントロピー)`（−2 スケール）。
(†) は `C_Q ≈ BIC_ELBO + 2 H(q)` で、**「観測データ型 BIC ＋ 2 × 潜在変数の事後エントロピー」という構造が ICL と同型**である。
この意味で「ICL-type」という呼称は構造的には妥当（DERIVED/INTERPRETATION）。

ただし決定的な違いがある:
- 離散潜在変数（ICL の本来の対象）ではエントロピーは非負で、単位に依存しない。
- 連続潜在変数 Z では `H(q)` は**微分エントロピー**であり、負にもなり、**Z の尺度の取り方（単位）に依存する**（§4）。
- 現行実装は θ を積分しておらず、`ẑ` の MAP でもなくサンプル平均を使う。ICL の「integrated」部分は `p_K ln n` による BIC 型近似に置き換わっている。

→ 「ICL-type」は**構造的アナロジーとしては妥当だが、ICL の理論的性質（離散・スケール不変）はそのまま移らない**。

## 4. Q2 — なぜ `Q_Z` が O(nK) の決定論的項になるか

4 つの要因を分ける。

1. **モデル（z ~ N(0, I_K)）そのもの（FACT/DERIVED）**: nK 個の連続座標の密度を評価すれば `ln p(Z) = −(nK/2) ln 2π − ½||Z||²` で、
   `E||Z||² ≈ nK` なら `−2 ln p(Z) ≈ nK(1+ln 2π)`。**complete-data の密度を評価する限り O(nK) の項は避けられない。**
2. **complete-data / Q-based scoring を使っていること（DERIVED）**: 観測データ尤度（Schwarz）では Z は積分され、`ln p(Z)` という項は現れない。
   その代わり Z を積分する Occam 効果は `Σ_i ln|A_i|` のようにデータ依存で現れる（(†), §5）。
   **P_Z が「罰則」として現れるのは complete-data 構成の帰結**である。
3. **scale_Z（FACT/DERIVED）**: `scale_Z` は平均二乗を厳密に 1 にするので、`Q_Z` が**サンプリング誤差のない決定論的定数**になる。
   ただし値そのものは scale_Z が作ったものではない: 事前分散を固定したモデルの最尤点では、EM の不動点条件から
   `(1/n) Σ_i E[z_i z_iᵀ] = I` が成り立つ（事前共分散を自由にした拡張モデルでも観測尤度が不変なため。標準的議論）ので、
   厳密な事後期待値でも同じ `−(nK/2)(1 + ln 2π)` になる。scale_Z は「揺らぎを消した」のであって、罰則を新たに作ったのではない。
4. **尺度の慣行への依存（DERIVED, 重要）**: 観測データのモデルは `(z ~ N(0, σ²I), F, w) ↔ (z ~ N(0, I), σF, σ²w)` で**同一**である
   （X: `F z` 不変、Y: `w z_iᵀ z_j` 不変）。一方 `D_K` と `p_K` は不変だが `P_Z` は
   `nK(1 + ln 2π + ln σ²)` に変わり、**`C_Q` は `nK ln σ²` だけずれる**。
   すなわち **K 1 次元あたりの `P_Z` の増分 `n(1 + ln 2π)` は、観測データからは識別できない慣行（事前分散 1）で決まっている**。
   Schwarz BIC（観測尤度）はこの変換で不変である。

**まとめ**: `P_Z ∝ nK` は (1) complete-data 構成と (2) 連続潜在変数という組み合わせの必然であり、
係数 `1 + ln 2π` は (3) 事前分散の慣行に依存し、決定論性は `scale_Z` による。

## 5. Q3 — 潜在変数事前項とパラメータ数罰則は両立するか

| framework | 当てはまり項 | Z の扱い | K 罰則の由来 | 現行 C_Q との関係 |
|---|---|---|---|---|
| observed-data marginal likelihood `ln p(X,Y|M)` | θ, Z とも積分 | 積分 | Occam 因子（自動） | 目標とする量ではない |
| Schwarz BIC（LIT: Schwarz 1978） | `ln p(X,Y|θ̂)`（Z 積分済み） | 積分 | `p ln n` のみ | C_Q − BIC ≈ `2H(q)` + (ELBO と観測尤度の差) |
| Laplace 近似（潜在変数も含む） | `ln p(X,Y,Ẑ|θ̂) + (nK/2)ln 2π − ½ ln|H_Z|` | 積分を Laplace 近似 | `½ ln|H_Z|`（データ依存）＋ `p ln n` | (†) の `Σ ln|A_i|` がこれに対応（block 近似） |
| complete-data BIC / classification likelihood | `ln p(X, Ẑ | θ̂)`（Z を推定値で固定） | パラメータ的に固定 | `p ln n`（Z は数えないか別扱い） | C_Q は `ln p(Z)` を含む点で近い |
| ICL（LIT: Biernacki et al. 2000） | `ln p(X, ẑ)`（θ 積分） | MAP 割当 | BIC ＋ 2 × 分類エントロピー | **構造的に最も近い**（C_Q ≈ BIC_ELBO + 2H(q)） |
| 原論文 Eq.(26) | `ln p(X,Y|Z)`（条件付き; p(Z) なし） | 条件づけ | `p ln n` | P_Z を含まない点で C_Q と異なる |

**INTERPRETATION**: 現行 C_Q に「潜在変数事前項」と「パラメータ数罰則」が同時に入るのは、**ICL 型（complete-data ＋ BIC 型 θ 罰則）の構造としては自然**である。
P_Z は第 2 のパラメータ数罰則ではなく、(†) で見たように **事後エントロピー項の定数部分**と読むのが正確である。
「両方 K とともに増える = double counting」という読みは、この構造を取り違えている。

## 6. Q4 — 本当に過剰罰則か

### 何を示せば「redundant / over-penalization」と言えるか

1. **基準の目標を定義する**（例: 観測データ周辺尤度の近似、あるいは K_true の一致推定）。
2. **C_Q の K 増分がその目標の K 増分を系統的に上回る**ことを示す（有限 n で、または n → ∞ で）。
3. 一致性を目標にするなら、`P(K̂ < K_true) → 0` が成り立たない条件を示す。

Phase 9D/9E の頻度（K=2 が 3/20）は 1 条件の観測であり、上のいずれの証明にもならない。

### 部分モデルでの厳密な結果（DERIVED）

**Gaussian-X のみ（w = 0、つまり Y が Z に依存しない; 実装では `fix_w` 経路）** で、事後分布が厳密に Gaussian（Laplace が厳密）、
θ̂ を最尤、`Σ_X` を既知の等方ノイズとする理想化を考える。このとき (★) は q = 厳密事後で等号となり、`A = I + Fᵀ Σ_X^{-1} F`（全ノードで共通）:

```
C_Q(K) = [ −2 ln p(X|θ̂) + p_K ln n ]  +  n K (1 + ln 2π)  −  n ln |I + Fᵀ Σ_X^{-1} F|
         └── Schwarz BIC ──────────┘     └─ 2H(posterior) ────────────────────────────┘
```

白色化した信号固有値を λ とする 1 次元を加えるとき（n 大, 母集団値で評価）:

- 観測尤度の改善: `−2Δ ln p = −n[λ − ln(1+λ)]`
- 事後エントロピー項の増分: `+n(1 + ln 2π) − n ln(1+λ)`
- よって **`ΔC_Q = n(1 + ln 2π − λ) + Δp ln n`**（`ln(1+λ)` は相殺される）

**帰結（この部分モデルでの厳密な主張）**:
- C_Q は `λ > 1 + ln 2π ≈ 2.84` の次元だけを（n → ∞ で）採用する。**`λ < 2.84` の真の次元は n を増やしても採用されない**。
  Schwarz BIC は `λ − ln(1+λ) > Δp ln n / n` なので、任意の `λ > 0` を n → ∞ で採用する。
- Schwarz BIC と比べた C_Q の追加分は次元あたり `n[(1+ln 2π) − ln(1+λ)]`: `λ < e^{1+ln 2π} − 1 ≈ 16.1` なら C_Q の方が厳しく、それ以上なら緩い。
  **「常に過剰罰則」ではなく、信号強度に依存する**。
- 閾値 `1 + ln 2π` は §4 の事前分散慣行に依存する（事前分散 σ² なら閾値は `1 + ln 2π + ln σ²`）。

→ この部分モデルでは、**現行の C_Q 構成は K の一致推定量ではなく、「信号が一定以上に明瞭な潜在次元を数える」ICL 型の基準**として振る舞う。
これは ICL が「分離の悪いクラスタを数えない」ことの連続版と解釈できる（INTERPRETATION）。

### 本モデル全体（UNRESOLVED）

Y が Z に依存する本モデルでは、`A_i` に `w² Σ_{j≠i} V_Y z_j z_jᵀ = O(n)` が入り `Σ_i ln|A_i|` が `O(nK ln n)` で増えうる。
ノード間の結合で事後は独立でなく、`Σ_i ln|A_i|` は block 近似にすぎない。Bernoulli/Poisson では Laplace は厳密でなく、MC（L=5）と `scale_Z` もある。
したがって**上の閾値型の結論が本モデルにそのまま成り立つかは未解決**である。
必要な証拠: (a) 本モデルでの `ln p(X,Y|θ)` と `E_q ln p(Z,X,Y)` の K 増分の漸近展開、または (b) 保存された `A_i` から `Σ ln|A_i|` を計算し
(†) の各項の K 増分を実データ上で比較する診断（新しい EM を要するなら別の Human Gate）。

## 7. Q5 — `ln n` の n

- **FACT**: 現行 `P_θ` の n はノード数（`n, d = X.shape`）。
- **DERIVED**: 現行 `p_K` に入るのは F（と Gaussian 列の分散）だけであり、Z が与えられた条件下で各列の loading `f_l` は **n 個の観測** `x_{1l},…,x_{nl}` から推定される。
  したがって **F の罰則として `ln n` を使うことは、条件付きの情報量の観点からは自然**である。
  Gaussian-X のみの部分モデルでは行が i.i.d. になり（因子分析）、`ln n` は Schwarz BIC そのものの sample size である。
- **Y 側**: `w0, w` は `n(n−1)/2` ペアから推定されるが、K に依存しない定数なので p_K に入れても入れなくても K の順位は変わらない（KI-010 (ii)）。
- **`nd` や `n(n−1)/2` を単純に使えない理由**: Schwarz の導出は独立同分布の観測ごとに θ 全体の情報が O(1) ずつ増えることを前提とする。
  X の要素 `nd` はパラメータごとに寄与が異なり（`f_l` は自分の列の n 個だけ）、Y のペアは Z を周辺化すると独立でない（交換可能配列）。
  パラメータ群ごとに異なる対数項を使う ICL の例がネットワークモデルにあることは知られている（LIT; SBM の ICL, Daudin, Picard & Robin 2008。
  具体的な罰則形は本セッションで原典を再確認していない）。
- **UNRESOLVED**: 本当に sample size の論点が効くのは、F ではなく **Z に関わる nK 個の座標**の扱い（§6 の `Σ ln|A_i|` がノードあたり `d + n − 1` の情報に依存する部分）である。
  これをどの framework で扱うかは未確定で、新しい effective sample size をここで定めることはしない。
- LIT: 過剰次元（K > K_true）ではモデルが特異になり（F の低ランク化）、正則な `p ln n` は漸近的に適切でない可能性がある
  （Watanabe の特異学習理論、Drton & Plummer 2017 の singular BIC。本 repository の RLCT は未知: #72 §7）。

## 8. Q6 — Phase 9E との接続

- **OBSERVED**（Phase 9E, 20 dataset）: K=3 が 17、K=2 が 3。`penalty_increase_23` は全 dataset で 256.02（P_Z 212.84 + P_θ 43.18）で一定。
  `fit_gain_23 = D_2 − D_3` は 162.81–439.54 に分布し、256.02 未満の 3 dataset が K=2 を選んだ。
- **INTERPRETATION**: 「罰則増分が一定で、データ依存の fit 改善がそれを跨ぐかで決まる」という観測は、
  §6 の部分モデルで導いた「次元ごとの固定閾値」構造と**整合する**。
  部分モデルでは `D_K` の減少は次元あたり約 `nλ`（`−2ΔELBO − Δ Σ ln|A_i|`）なので、固定の P_Z 増分 `n(1+ln 2π)` と比較する形になる。
- **HYPOTHESIS（検証していない）**: 部分モデルの式（fit_gain ≈ nλ）を形式的に当てはめると、閾値 256.02 は有限 n で
  `λ ≈ 256.02 / 75 ≈ 3.41`（= 漸近閾値 `1 + ln 2π ≈ 2.84` ＋ `Δp ln n / n ≈ 0.58`）、median の fit_gain 300.7 は `λ ≈ 4.0` に相当する。
  本データは混合 X＋Bernoulli-Y なので、この換算は構造理解のための目安にすぎない。
- Phase 9E は **現行基準の挙動の記述**であり、理論的な無効性の証明には使わない。

## 9. FACT / DERIVED / OBSERVED / INTERPRETATION / UNRESOLVED 一覧

| 区分 | 内容 |
|---|---|
| FACT | C_Q = −2Q_strict + p_K ln n。Q_strict は Laplace サンプラーによる complete-data 対数密度の MC 平均。var_z=1 固定、scale_Z で平均二乗 1。P_Z = nK(1+ln2π) |
| FACT | 原論文 Eq.(26) の `ln L` は z 条件付き（p(Z) なし）。論文は σ_z² を推定。BIC 採用理由に一致性を挙げる |
| DERIVED | C_Q ≈ [−2 ELBO + p_K ln n] + 2H(q)。P_Z は 2H(q) の定数部分（ICL と同型の構造） |
| DERIVED | Z の尺度慣行（事前分散）を変えると、観測モデル・D_K・p_K は不変だが C_Q は nK ln σ² ずれる |
| DERIVED | Gaussian-X のみの部分モデル（厳密事後, 最尤）で ΔC_Q = n(1+ln2π−λ) + Δp ln n。λ < 1+ln2π の次元は n→∞ でも採用されない。Schwarz BIC との差は信号依存で符号が変わる |
| DERIVED | F の罰則に ln n を使うことは、条件付き情報量の観点から自然 |
| OBSERVED | Phase 9E: 罰則増分 256.02 一定、fit_gain_23 が跨いで K=2/3 が分かれた（3/20 が K=2） |
| INTERPRETATION | 「P_Z + P_θ = 二重罰則」は構造の読み違い。P_Z はエントロピー項、P_θ は θ の BIC 罰則で、役割が異なる |
| INTERPRETATION | Phase 9E の観測は部分モデルの固定閾値構造と整合する |
| UNRESOLVED | Y が Z に依存する本モデルで、閾値型の結論と C_Q の漸近挙動が成り立つか |
| UNRESOLVED | Z に関わる nK 座標の effective sample size の扱い |
| UNRESOLVED | 過剰次元での特異性（RLCT）と `p_K ln n` の適切さ |

## 10. Decision

## **C（限定付き）— 具体的な理論的性質が導出でき、criterion 設計を研究課題として始める根拠がある**

根拠:
1. **内部整合性はある**（#72 の恒等式は正しい。実装は定義どおりの量を計算している）。この点は A/B と共通。
2. しかし **本モデルに含まれる部分モデル（Gaussian-X のみ, w = 0）で、C_Q の構成そのもの（実装誤差ではない）が次の性質を持つことが厳密に導ける**:
   - 信号固有値 `λ < 1 + ln 2π` の真の潜在次元を n → ∞ でも選ばない（**K の一致推定量ではない**）
   - 次元ごとの閾値が、観測データから識別できない事前分散の慣行に依存する（**観測モデルを保つ再パラメータ化で不変でない**）
3. これは repository の研究目的（Phase 9D/9E の K_true 回復）や、原論文が BIC 採用理由とした「一致性」と**具体的に食い違う**。
   ICL 型基準としては自然な性質（明瞭な構造だけを数える）だが、「真の K を回復する基準」とは別物である。

**A でない理由**: 原論文が挙げる一致性の根拠は、この構成（complete-data＋P_Z）には移らない。
**B でない理由**: 不明点が残るだけでなく、部分モデルでは理論的な帰結が具体的に導けたため。

**限定**:
- 本モデル全体（Y あり、Bernoulli/Poisson、Laplace 近似、MC）で同じ閾値挙動が成り立つかは **UNRESOLVED**（§6）。
- C は「現行 C_Q を今すぐ置き換える」判断ではない。**本報告は新しい criterion を提案しない**。
  現行 C_Q は、上の性質を明記した上で **ICL 型の経験的基準**として引き続き使える。

## 11. 次に本当に必要な研究判断（Human）

1. **K 選択の目標をどちらに置くか**:
   (a) 生成モデルの K_true を回復する（一致性を目標とする）→ criterion 設計を別 phase で始める根拠がある、
   (b) 「明瞭に決まる潜在次元」を数える ICL 型の目的でよい → 現行 C_Q を性質の明記つきで維持。
2. (a) の場合、最初の理論課題は §6 の UNRESOLVED（Y を含む本モデルでの `E_q ln p(Z,X,Y)` と観測尤度の K 増分）であり、新しい罰則の提案はその後。
3. 論文・原稿での記述: 「BIC」「一致性」の語を C_Q に使わない現行ルール（root CLAUDE.md §5, KI-010）を維持し、
   本報告の「ICL 型構造と部分モデルでの閾値性」を limitation として記すかどうか。

---

### 参照

repository（一次）: `expfam/src/experimental/em_runner.py`（E-step, scale_Z, Q 評価）、`expfam/src/experimental/eval_utils.py`（`calc_Q_dual_strict_exp`, `calc_bic_exp`）、
`reproduction/src/model.py`（`calc_eta_newton` L.360–462, `scale_Z` L.468–, `var_z` L.99）、
`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`（Eq.(14), (16)–(19), (26), §4.3）、
`reports/k_selection_theory/cq_decomposition_minimal_check_20260923.md`、
`reports/distribution_selection/phase9e_k_repeatability_summary_20260927.md`、`RESEARCH_MASTER.md` §12.6、`KNOWN_ISSUES.md` KI-010。

外部文献（LIT; repository の事実とは区別）:
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics* 6(2).
- Biernacki, C., Celeux, G., Govaert, G. (2000). Assessing a mixture model for clustering with the integrated completed likelihood. *IEEE TPAMI* 22(7), 719–725.
  https://ui.adsabs.harvard.edu/abs/2000ITPAM..22..719B/abstract
- Daudin, J.-J., Picard, F., Robin, S. (2008). A mixture model for random graphs. *Statistics and Computing* 18, 173–183.
  https://link.springer.com/article/10.1007/s11222-007-9046-7 （SBM の ICL。罰則の具体形は本セッションで未確認）
- Handcock, M. S., Raftery, A. E., Tantrum, J. M. (2007). Model-based clustering for social networks. *JRSS A* 170(2), 301–354.
  https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-985X.2007.00471.x （潜在空間ネットワークモデルの次元選択の先行例。詳細は未確認）
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §9.4（変分恒等式 (★)）。
- Watanabe, S. (2009). *Algebraic Geometry and Statistical Learning Theory*; Drton, M., Plummer, M. (2017). A Bayesian information criterion for singular models. *JRSS B* 79(2).
