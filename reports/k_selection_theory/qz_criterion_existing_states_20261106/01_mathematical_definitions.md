# 01 数式の定義：Q1・Q2・Q3、C_Q、Q_Z、Eq.(26)、C_Lap（2026-11-06）

- 位置づけ：系列 E（prototype、本文採用不可）。新しい MCEM fit はしていない。
- C_Q は Q ベースの complete-data / ICL 型の基準で、**Schwarz BIC ではない**。C_Lap はラプラス近似で、**厳密な周辺尤度ではない**。
- 一次資料：原論文 PDF（`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`）、`expfam/src/experimental/k_criterion_panel.py`、`laplace_k_criterion.py`、`em_runner.py`、`reproduction/src/model.py`（`scale_Z`）、`expfam/src/utils_expfam.py`（`calc_bic_dual`）、`reproduction/src/experiment_paper_2.py`（`calc_BIC`）。

## 1. 3 つの対数尤度（DERIVED）

```
Q1(Z) = ln p(X, Y | Z, θ)                       Z を 1 点に固定した条件付き尤度
Q2(Z) = ln p(Z, X, Y | θ) = ln p(Z) + Q1(Z)      完全データ対数尤度
Q3    = ln p(X, Y | θ) = ln ∫ exp Q2(Z) dZ       観測データの周辺尤度
```

- 原論文 Eq.(16)：L = ∏_i p(x_i | F, z_i, …) {∏_{i≠j} p(y_ij | z_i, z_j, …)}^{1/2}。**Q1**（Z を与えた条件付き尤度。Y は各ペアを 1 回数える）。
- 原論文 Eq.(17)：Q(θ, θ_old) = E_{Z|X,Y,θ_old}[Q2(Z)]。Eq.(18)：その MC 近似 (1/L) Σ_l Q2(Z^(l))。
- 原論文 Eq.(26)：BIC = −2 ln L + ((k+1)d − k(k−1)/2) ln n、ln L は Eq.(16)。**どの Z で評価したかは本文にない**（PR #158 の 01）。
- Schwarz BIC が近似するのは Q3 である。Eq.(26) も C_Q も Q3 ではない。

## 2. 現行の基準と、本研究で比べる 5 つの基準

すべて「小さいほど良い」。p_K ln n は共通の罰則。

| 記号 | 定義 | 評価する量 | Z の扱い |
|---|---|---|---|
| A：C_Q | −2(Q_X + Q_Y + Q_Z) + p_K ln n | Q2 の MC 平均 | E-step の最終反復の L=5 標本の平均 |
| B：C_Q − P_Z | −2(Q_X + Q_Y) + p_K ln n | **Q1 の MC 平均** | 同じ 5 標本 |
| C：C_cond_last | −2 Q1(Z_last) + p_K ln n | Q1 の plug-in | 最後の 1 標本 Z_last = Z_samples[:, :, −1] |
| D：C_cond_mode | −2 Q1(Ẑ) + p_K ln n | Q1 の plug-in | Q2 を Z について最大化した結合モード Ẑ（Laplace の mode） |
| E：C_Lap | −2[Q2(Ẑ) + (nK/2) ln 2π − ½ ln|H(Ẑ)|] + p_K ln n | Q3 のラプラス近似 | Ẑ と、そこでの曲率 H |

- Q_X = (1/L) Σ_l ln p(X | Z_l, θ)、Q_Y = (1/L) Σ_l ln p(Y_train | Z_l, θ)、Q_Z = (1/L) Σ_l ln p(Z_l)。P_Z = −2 Q_Z（`k_criterion_panel.cq_terms`）。
- **DERIVED：B は C_cond_MC と恒等的に等しい。** C_Q = −2(1/L)Σ_l [Q1(Z_l) + ln p(Z_l)] + p ln n なので、C_Q − P_Z = −2 (1/L)Σ_l Q1(Z_l) + p ln n = C_cond_MC。本研究ではこの等式を全状態で数値的に確認する（03）。
- Y は学習用ペア（canonical は全ペアの 80%）だけを使う。fit と C_Lap と同じマスク。
- D の Ẑ は「Q1 を最大にする Z」ではない。Q1 を Z について最大化すると、Z を自由パラメータとして当てはめることになり、K とともに単調に良くなる。D は p(Z) を含む Q2 の最大点で Q1 を評価したもので、事前分布が Z を縮小している。C・D・B は Z の評価点が違うだけで、どれも原論文の BIC と同一ではない（原論文は評価点を書いていない）。

## 3. Q_Z の恒等式（DERIVED、03 で数値確認）

E-step の後、`scale_Z`（`reproduction/src/model.py:468-505`、系列 E は `em_runner.py:272` で呼ぶ）は L 個の標本をまとめて 1 つの定数倍で縮尺し、(1/(nKL)) Σ_l ‖Z_l‖²_F = 1 にする。σ_z² = 1 のとき

```
Q_Z = (1/L) Σ_l [−(nK/2) ln 2π − ½ ‖Z_l‖²_F]
    = −(nK/2) ln 2π − ½ · nK
⇒  P_Z = −2 Q_Z = nK (1 + ln 2π) = 2.8379 · nK
```

- 1 次元増やすごとに P_Z は 2.8379 n 増える（n=50: 141.9、75: 212.8、100: 283.8、150: 425.7）。
- これは θ にも X・Y にも依存しない定数なので、**K から K+1 への余裕 S = C(K) − C(K+1) は、B では A より正確に 2.8379 n だけ大きい**。中央値も同じだけずれる。
- この恒等式は「L 標本の平均」についてだけ成り立つ。最後の 1 標本 Z_last や Ẑ（縮尺しない）の ‖Z‖² は nK と一致しない。

## 4. パラメータ数と罰則（DERIVED）

| 基準 | p_K |
|---|---|
| 原論文 Eq.(26) | (k+1)d − k(k−1)/2 |
| `calc_bic_dual`（初期 ExpFam）、系列 E の C_Q・C_Lap | kd − k(k−1)/2（Gaussian-X は +d、Gaussian-Y は +1） |
| Python 再現 `calc_BIC` | (k+1)d − k(k−1)/2 |

- 差は (k+1)d − kd = d で、**K によらない定数**。したがって Poisson-X の canonical 条件に Eq.(26) の罰則を使っても、K の選択と余裕は変わらない（全基準に d ln n を足すだけ）。Gaussian-X（#139）では、現行の p_K がすでに Eq.(26) と一致する。
- 罰則の増分は (d − K) ln n（K → K+1）。d=15、n=75 で K=4→5 なら 11 · 4.317 = 47.5。

## 5. 何を「原論文型」と呼ぶか

- C と D は「Eq.(16) 型の条件付き plug-in 基準」で、Z の評価点を 2 通りに明示したもの。B は「Eq.(16) 型を MC 平均で評価したもの」。
- **いずれも原論文の BIC の再現ではない。** 原論文の Z の評価点、試行の集計、生成器、ln L の正規化は不明（PR #158）。
