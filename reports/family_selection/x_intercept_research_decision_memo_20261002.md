# X 列 intercept と family-selection claim の研究判断メモ（2026-10-02）

- 位置づけ: **THEORY / DESIGN / DECISION MEMO（決定はしない）**。**新しい EM 0、dataset 0、コード変更 0。**
- 前提資料: `family_selection_hard_case_audit_20261002.md`（以下「監査」）。
- **DECISION: HUMAN DECISION REQUIRED**。X 列 intercept の導入は生成モデル・数式の変更であり、
  root `CLAUDE.md` §6 の Human Gate（family-selection design の HG-5）である。本メモは比較材料のみを示す。
- ラベル: FACT / DERIVED / NUMERICAL / INTERPRETATION / UNRESOLVED。

---

## 0. 判断に直結する 2 つの事実

1. **DERIVED（監査 §2）** 現行 no-intercept model では Poisson 列の母集団平均 rate は `exp(‖f_l‖²/2) ≥ 1`、
   Bernoulli 列の母集団 1 率は厳密に 1/2。n = 75 で「0/1 だけの真 Poisson 列」は 1 列あたり ≤ 1.0×10⁻¹⁰。
2. **DERIVED（監査 §5）** 0/1 列では、任意の η で `log p_Pois = log p_Bern + log P_Pois(X≤1|η)` が成り立つので、
   同じパラメータ数の尤度 score は**常に Bernoulli を選ぶ**。これは **intercept の有無によらない**。

したがって **intercept を入れても「0/1 列で Poisson を正しく選べるか」という問いは生まれない**。
intercept で変わるのは「0/1 だけの真 Poisson 列が起きる頻度」と「Bernoulli と Poisson がどれだけ近いか（margin の大きさ）」であり、
score の向きではない。この点が 3 案の比較の前提になる。

---

## 1. X intercept を入れた場合の理論差分（実装しない）

仮のモデル:

```
z_i  ~ N(0, I_K)
x_il ~ ExpFam_{c_l}( η_il^X = b_l + f_l^T z_i )        b_l ∈ R（列ごと）
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )       （変更なし）
θ' = θ ∪ { b_1, …, b_d }
```

`s_l² := ‖f_l‖²` とすると `η_il ~ N(b_l, s_l²)`。

### A. Poisson で b_l < 0 を使うと低 rate 列を作れるか — YES（DERIVED）

```
E[λ_il] = exp(b_l + s_l²/2)
```

任意の目標平均 m > 0 と任意の s_l² について `b_l = log m − s_l²/2` で実現できる。
**NUMERICAL**（監査付録 T4、n = 75）:

| s² | m | b | P0 | P1 | P≥2 | P(全 75 観測 ∈ {0,1}) |
|---|---|---|---|---|---|---|
| 0.5 | 0.05 | −3.246 | 0.952 | 0.0461 | 0.0019 | 0.868 |
| 0.5 | 0.1 | −2.553 | 0.908 | 0.0855 | 0.0070 | 0.592 |
| 0.5 | 0.2 | −1.859 | 0.828 | 0.148 | 0.0240 | 0.162 |
| 0.5 | 0.3 | −1.454 | 0.759 | 0.194 | 0.0470 | 0.027 |
| 0.5 | 0.5 | −0.943 | 0.644 | 0.253 | 0.103 | 3.0e-4 |
| 0.5 | 1.0 | −0.250 | 0.449 | 0.301 | 0.251 | 4.0e-10 |

### B. Bernoulli の base rate も b_l で調整できるか — YES（DERIVED）

`P(X=1) = E[σ(b_l + s_l Z)]` は b_l について連続・狭義単調増加で、b_l → ±∞ で 1 / 0 に近づくので (0,1) の任意の値を取れる。
（no-intercept では 1/2 に固定。）

### C. Bernoulli vs Poisson の 0/1 hard case を controlled に設計できるか — 発生は YES、判別の評価は NO（DERIVED）

- **発生**: A の表の通り、低 rate（m ≲ 0.1）なら n = 75 でも 0/1 だけの真 Poisson 列が普通に起きる。
- **判別**: 監査 §5 の恒等式は η ごとの式なので `η = b + f^T z` でもそのまま成立し、**0/1 列は必ず Bernoulli が選ばれる**。
  さらに低 rate では `σ(η) ≈ e^η` なので 2 つの family 自体がほぼ同じ分布になる:
  P(X=1) を揃える Bernoulli の切片は Poisson の切片とほぼ一致する（s² = 0.5, m = 0.1 で b_P = −2.553, b_B = −2.570）。
  margin の大きさの参照値 `2n E[g(η)]`（真のパラメータでの母集団期待値。fit 済み η での厳密な下界そのものではない）は
  m = 0.1, s² = 0.5 で **1.06**（no-intercept の既存条件では ≈ 46–80）。
- **帰結（INTERPRETATION）**: intercept で作れる「controlled hard case」は、**score が Poisson を選べるかではなく、
  (i) gate が 2 以上を観測して Poisson を検出できる確率（= 1 − p^n、解析的に計算できる）と
  (ii) Bernoulli で代用したときに下流（Z の回復・K 選択・予測）がどれだけ悪くなるか**を問う設計にしかならない。

### D. 識別性への影響（DERIVED / UNRESOLVED）

- **回転**: `z → Rz`, `F → F R^T` で `b + F z` も `z^T z'` も不変、b は回転しない。**回転の不定性は現行と同じ**であり F ブロックの数え方 `Kd − K(K−1)/2` は変わらない。
- **平行移動**: `z → z + c`, `b → b − F c` は X 側の η を保つが、
  (i) 事前 `N(0, I_K)` の密度を変え、(ii) Y 側で `(z_i+c)^T(z_j+c) = z_i^T z_j + c^T(z_i+z_j) + ‖c‖²` となり不変でない。
  よって**モデルの不変変換ではなく**、b は事前分布と Y によって固定される（現行で X の平均 0 を暗黙に固定していた役割を、b と事前分布が分担する）。
- **列ごとの (b_l, s_l) の識別**: Gaussian は `E[x_l] = b_l` で直接。Poisson は平均 `exp(b + s²/2)` と過分散 `Var/E − 1 = E(e^{s²} − 1)` の 2 つのモーメントで (b, s²) が決まる。
  **Bernoulli は列の周辺分布が 1 率だけなので、列単独では (b, s) は識別されない**（s は他の列・Y との共分散構造を通してのみ決まる）。
  これは現行の Bernoulli でも s が列単独で決まらないのと同じ種類の問題だが、b を足すと単一列の情報で決まる量がさらに 1 つ減る。
- **K の識別への影響**: UNRESOLVED。`RESEARCH_MASTER.md` P7（Bernoulli-X の一次モーメントは Z の 0 対称性により常に 1/2 で K の情報を持たない）は
  b_l を入れると成り立たなくなる（一次モーメントが b_l と s_l に依存する）。U1（Bernoulli-X, d > 1 の識別可能性）は未解決のまま、1 パラメータ/列が加わる。
  b は K に依存しない量なので、K と K+1 の入れ子構造の議論（P3, `w ≠ 0`）自体は変えないと予想されるが、**証明はしていない**。

### E. M-step で追加されるパラメータ（DERIVED）

列 l ごとに `(b_l, f_l) ∈ R^{K+1}` を同時に最適化する。`z̃ = (1, z^T)^T` とおくと canonical link の GLM:

```
Q_l(b_l, f_l) = (1/L) Σ_s Σ_i [ T(x_il) η_il^(s) − A(η_il^(s)) ] / φ_l + const,   η_il^(s) = b_l + f_l^T z_i^(s)
∇ Q_l   = (1/L) Σ_s Σ_i (x_il − A'(η_il^(s))) / φ_l · z̃_i^(s)
∇² Q_l  = −(1/L) Σ_s Σ_i A''(η_il^(s)) / φ_l · z̃_i^(s) z̃_i^(s)T     （負定値 → 凹）
```

- Gaussian: 閉形式 `[b_l; f_l] = (Z̃^T Z̃)^{-1} Z̃^T x_l`（サンプル s を積み上げた最小二乗）、`σ_l² = 残差二乗平均`。
- Bernoulli / Poisson: 現行の per-column BFGS（`family_selection.optimise_column_loading` 相当）の次元が K → K+1 になる。
- w_0^Y, w^Y, σ_Y の M-step は変わらない。

### F. E-step の勾配・Hessian への影響（DERIVED）

確定式（CLAUDE.md §1）の X 項で `F m_i` を `b + F m_i` に置き換えるだけで、形は変わらない:

```
gradient_i = −z_i + F^T V_φ^X [ T_X(x_i) − A_X'(b + F z_i) ]
             + w^Y Σ_{j≠i} [ T_Y(y_ij) − A_Y'(η_ij^Y) ] / φ_Y · z_j
A_i        = I_K + F^T V_X(b + F m_i) F + (w^Y)² Σ_{j≠i} V_Y(η_ij^Y) z_j z_j^T
V_X(·)     = Σ_X^{-1}（Gaussian）,  diag(A_X''(b + F m_i))（Bernoulli / Poisson）
```

**INTERPRETATION** 低 rate の Poisson 列（b ≪ 0）では `A'' = e^{b + f^T m}` が小さく、
**1 列が z_i にもたらす情報（A_i への寄与）が小さくなる**。低 rate 列を作ることは同時に X 側の信号を弱めることでもあり、
K 選択の感度（Phase 9V の「X 信号」）と交絡する。

### G. family score への影響（DERIVED）

- 各候補が `(b_l, f_l)` を最適化する（K+1 次元）。
- 0/1 列: 監査 §5 により**常に Bernoulli**。margin の下界は Poisson 候補の fit 値での `(2/L) Σ_s Σ_i g(b̂_P + f̂_P^T z_i^(s))`（> 0）で、低 rate では 0 に近づく。
- 2 以上を含む列・非整数列: gate が決めるので不変。
- **Gaussian vs 離散の比較は gate が分けたまま**（測度が違う。design §3.3）。intercept は gate の論理に影響しない。

### H. C_Q のパラメータ数への影響（DERIVED）

現行 `calc_bic_dual`（`utils_expfam.py` L399–402）:

```
num_params = [Kd − K(K−1)/2] + (Gaussian-X の列数 or d) + (Gaussian-Y なら 1)        （w0, w は数えない）
```

intercept を入れる場合:

```
num_params' = num_params + d_b,   d_b = intercept を持つ列の数（全列なら d）
C_Q'(K)     = −2 Q_strict' + num_params' · ln n
```

`d_b · ln n` は **K によらない定数**なので、family assignment を固定した K の比較では相殺する。
ただし `Q_strict'` 自体（データ当てはまりの項）は b の推定で変わるので、**C_Q の K 選択結果が変わらない保証はない**。

### I. C_Lap の d_K への影響（DERIVED）

```
d_K' = Kd − K(K−1)/2 + n_gaussian_selected + d_b
C_Lap'(K) = −2[ℓ_X(Ẑ; b) + ℓ_Y(Ẑ)] + ‖Ẑ‖² + ln|H(Ẑ)| + d_K' ln N
```

- `d_b ln N` は K によらず相殺する。
- b は θ の一部であり Z について積分しないので、**H（nK × nK の Z の Hessian）の形は変わらない**。
  ただし H の X ブロックの曲率は `A''(b + f^T ẑ)` で評価されるので値は変わる。
- 9L の `N = n` の議論（F ブロックの情報が O(n)）は b にもそのまま当てはまる（b の情報も O(n)）と予想されるが、**確認していない**。

### J. K 間で相殺するもの／family assignment で数が変わるもの（DERIVED）

| パラメータ | 個数 | K に依存 | family に依存 | K 比較で相殺 | family 比較で相殺 |
|---|---|---|---|---|---|
| F（回転制約込み） | Kd − K(K−1)/2 | する | しない | しない | する |
| b_l（全列に入れる場合） | d | しない | しない | する | する |
| σ_l²（Gaussian-X 列） | Gaussian 列数 | しない | する | family 固定なら する | gate が決めるので実質 する |
| w_0^Y, w^Y | 2 | しない | しない | する（現行は数えていない） | する |
| σ_Y²（Gaussian-Y） | 1 | しない | しない | する | する |

**DERIVED** intercept を**全列に一律に**入れる限り、family assignment によってパラメータ数は変わらない（Bernoulli / Poisson の追加は 0、Gaussian は σ² のみ）。
**family ごとに intercept の有無を変える設計は避けるべき**（数が family で変わり、score 比較に新しいペナルティ差が生まれる）。

---

## 2. 3 案の比較

### Option A：no-intercept model を維持し、family-selection claim を限定する

claim の候補表現（案）: 「現行 3 family と support gate のもとでは、family の割り当ては観測 support によって決まり、
0/1 列では同じパラメータ数の尤度比較が構造的に Bernoulli を選ぶことを示した。既存 C2 artifact では prototype の結果がこの理論的順序と一致した（真 Bernoulli の 0/1 列で全て Bernoulli。一般的な family 判別精度を示すものではない）」。

| 観点 | 評価 |
|---|---|
| research benefit | 監査の恒等式（0/1 では Poisson = Bernoulli × P(X≤1)）と support-determinism を**理論的な整理**として書ける。新規実験なし |
| scientific risk | 低い。ただし「自動 family 選択ができた」とは**書けない**（score が何も決めていないため）。claim は「feasibility」よりさらに弱く、「support で決まることの確認」になる |
| implementation cost | 0 |
| 既存結果との互換性 | 完全（全 Phase 9 結果をそのまま使える。family 交絡がないことはむしろ K 選択の結果を補強する） |
| thesis claim への影響 | family selection は主張の柱にならない。K 選択（Phase 9）と指数型分布族一般化が中心になる |

### Option B：X intercept を正式にモデルへ追加する

必要になるもの:

| 項目 | 内容 |
|---|---|
| theory | 確定式（CLAUDE.md §1）の改訂、§1 D の識別性（特に Bernoulli 列単独で b, s が決まらない点）、P3 等の K 構造の再確認、C_Lap の N = n の議論の b への拡張 |
| implementation | 新しい lineage（例: `model_dual_expfam_intercept.py`）。E-step（η に b）、M-step（K+1 次元の GLM / Gaussian 閉形式）、Q evaluator、Laplace evaluator（Φ と H の X ブロック）、generator（b_l の指定と metadata）、num_params / d_K |
| tests | b = 0 固定で既存 lineage と bit 単位一致する回帰テスト、勾配・Hessian の有限差分テスト、b の閉形式（Gaussian）テスト、generator の moment テスト（`E[λ] = exp(b + s²/2)`） |
| synthetic redesign | 列ごとの b_l の設定方針（平均 rate / base rate を揃えるか）、X 信号（A'' の大きさ）を揃えるか、Phase 9 の「信号を揃えた」設計との対応 |
| parameter penalty | `+d`（全列一律）。K 比較では相殺するが、記録・報告では明示 |
| previous-result compatibility | 既存の全結果は b = 0 の部分モデル。既存 dataset は「b = 0 の新モデルからの draw」として有効だが、**新モデルでの fit 結果は既存系列と混ぜられない**（CLAUDE.md §3、KI-002）。先行研究の定式化（バイアスなし）からの逸脱になる（theory audit §15.2 の確認質問） |

| 観点 | 評価 |
|---|---|
| research benefit | 低 rate count 列・偏った binary 列を扱える（実データの属性に近い）。KI-018（raw-count Poisson 悪化の 4 要因交絡）の 1 要因を切り離す前提になる |
| scientific risk | 中〜高。**family selection の問題は解決しない**（0/1 列は intercept があっても常に Bernoulli）。intercept は X 信号の強さと交絡する（§1 F）。Bernoulli 列の識別がさらに弱くなる |
| implementation cost | 高（新 lineage、評価関数 2 種、generator、テスト、Phase 9 相当の再検証） |
| 既存結果との互換性 | 低（過去の数値とは別系列。比較には同一データでの並行 fit が必要） |
| thesis claim への影響 | モデル拡張としての貢献は増えうるが、修論期間内に Phase 9 相当の K 選択の検証をやり直す必要がある |

### Option C：モデルは変えず、family selection を修論の中心 claim にしない

claim の候補表現（案）: 「指数型分布族への一般化（X / Y の family を指定できる Dual-ExpFam LSM）と、その K 選択基準の整理が中心。
列ごとの自動 family 選択は experimental prototype として構成し、現行の family 集合では support で決まることを示した上で、
低 rate count・intercept・過分散を含む本格的な選択は future work とする」。

| 観点 | 評価 |
|---|---|
| research benefit | 修論の主張を確証のある部分（K 選択の理論整理と Phase 9、指数型分布族一般化）に集中できる。監査の恒等式は「prototype の限界の同定」として価値がある |
| scientific risk | 最も低い |
| implementation cost | 0 |
| 既存結果との互換性 | 完全 |
| thesis claim への影響 | family selection は付録・将来課題の扱い。A と同じ事実を、より控えめな位置づけで書く |

### 比較のまとめ（INTERPRETATION）

| | A 維持＋限定 | B intercept 追加 | C prototype 扱い |
|---|---|---|---|
| 0/1 列で Poisson を選べるようになるか | ならない | **ならない**（恒等式は不変） | — |
| 0/1 の真 Poisson 列が起きるか | 事実上起きない | 低 rate なら普通に起きる | 事実上起きない |
| 追加コスト | 0 | 高 | 0 |
| 既存 Phase 9 との互換 | 完全 | 別系列 | 完全 |
| family selection を主張にできるか | 「support で決まる」まで | 「下流への影響」まで（選択能力ではない） | 主張にしない |

A と C は同じ事実に立ち、違いは**修論での位置づけ**だけである。
B を選ぶ理由は「family selection の改善」ではなく「低 rate / 偏った属性を扱えるモデル拡張」であるべきで、
その場合も family selection の問いは「Bernoulli 代用の下流コスト」に置き換わる。

**DECISION: HUMAN DECISION REQUIRED**（A / B / C の選択、B の場合は新 lineage の scope 承認）。
