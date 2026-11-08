# 01 原論文 Experiment 2 / Figure 3 の条件と、本研究の条件（2026-11-09）

## 位置づけ

- **ブランチ**：`research/paper-fig3-condition-criteria-20261109`。PR #164 の先端 `433fc07` から分岐した。
- **系列**：E（prototype）。本文採用は未決定。
- **本研究は原論文 Experiment 2 の再現ではない。**
  - 推定器は系列 E である。
  - 原論文の条件には、記載がないものがある（下表の ASSUMED・UNKNOWN）。
- **呼称**：C_Q は Schwarz BIC ではない（ICL 型）。C_Lap は厳密な周辺尤度ではない。C_ELBO は既存の変分下界による基準で、新しい BIC ではない。cond_MC・cond_last・cond_mode は、原論文 Eq.(16)/(26) に着想を得た条件付き尤度型の基準で、原論文の BIC ではない。

## 1. 原論文の記載（FACT。PDF 本文を pypdf で抽出して確認。PR #158 の 01 とも一致）

- **Table I**：Experiment 2 は d=15、n=150、L=10、反復 10、試行 10、候補 k 1〜10、真の次元 1・3・5・7・9。
- **§4.1**
  - Step 1：σ_z²=1、z_i ~ N(0, σ_z² I)。
  - Step 2：「Set the random values for w0 and w」。分布は書かれていない。
  - Step 3：F の各成分 ~ N(0, 10)、Σ = diag(0.1)、μ_x = 0。
  - 試行：「We execute 10 trials to implement the algorithm with different initial values」。
  - 集計：Experiments 1・3・4 については「反復ごとの 10 試行の平均」と「全試行・全反復の最小値」を定義している。**Experiment 2 については集計方法が書かれていない。**
- **§4.3**
  - Eq.(26)：BIC = −2 ln L + ((k+1)d − k(k−1)/2) ln n。L は Eq.(16)（Z を条件とした尤度）。
  - 「Fig. 3 で、各 k\* について k = k\* のとき BIC が最小」。
  - **Z の評価点は書かれていない。**

## 2. 条件の分類（`condition_fidelity.csv`、正本は `tools/research_audit/paper_fig3_protocol.py` の CONDITIONS）

| 項目 | 原論文 | 本研究 | 分類 |
|---|---|---|---|
| n, d | 150, 15 | 同じ | EXACT |
| 真の次元・候補 K | {1,3,5,7,9}・1〜10 | 同じ | EXACT |
| L・EM 反復 | 10・10 | 同じ | EXACT |
| 試行 | 10、「異なる初期値」 | 同じデータに対する 10 個の初期値と MC 系列 | 数は EXACT。**同じデータで行うことは ASSUMED** |
| Z | N(0, I) | N(0, I)、正規化なし | EXACT |
| F | 各成分 N(0, 10) | 10 を**分散**とした N(0, 10)、行の正規化なし | **ASSUMED**（分散か標準偏差か書かれていない） |
| Σ・μ_x | diag(0.1)・0 | 同じ | EXACT |
| w0, w | 「random values」 | w0 ~ N(0,1)、w = 3·N(0,1) | **ASSUMED**（MATLAB の `setTrueParams.m` L30-31 と、Python 移植の `reproduction/src/data_generator.py` の規則。論文の記載ではない） |
| Y の観測 | 全ペア（マスクの記載なし） | 全ペア | EXACT（記載なし） |
| データの正規化 | 人工データについての記載なし | なし | EXACT（記載なし） |
| モデル：μ_x | 推定する（Eq.(11)） | 系列 E にはない | **DIFFERENT** |
| モデル：σ_z² | 推定する（Eq.(14)） | 1 に固定し、E-step のたびに `scale_Z` | **DIFFERENT** |
| E-step の Y 側の係数 | Eq.(22)/(23) に 1/2 | 余分な 1/2 なし（採用式） | **DIFFERENT** |
| Newton の刻み | 記載なし | 減衰 α=0.5、最大 10 ステップ | IMPLEMENTED（系列 E） |
| w0, w の更新 | Adam（Eq.(24)/(25)） | 系列 E の M-step | IMPLEMENTED（系列 E） |
| 初期値 | 記載なし | `em_runner` の informed init | IMPLEMENTED / 原論文は UNKNOWN |
| BIC の Z の評価点 | 記載なし | 同じ fit の状態に 6 基準を適用 | UNKNOWN（原論文） |
| 試行の集計（Exp.2） | 記載なし | 試行ごとの選択（主）＋平均・最小（副） | UNKNOWN（原論文） |
| 罰則 | ((K+1)d − K(K−1)/2) ln n | Kd − K(K−1)/2 + d（Gaussian-X） | **EXACT**（同じ式） |

**既存の再現コードとの違い**：MATLAB/Python の生成器（PR #139 の paper-type）は、原論文の本文と違う。

- F を N(0,5) で引いた後、各行のノルムを √0.9 に固定する。
- Z と X を z-score する。

本研究は**原論文の本文どおり**の生成器（`G_paper_text`）を使う。この生成器で fit を行ったのは、本研究が初めてである（PR #158 では情報量の計算だけに使った）。

## 3. 事前登録

| 項目 | 内容 |
|---|---|
| プロトコル | `tools/research_audit/paper_fig3_protocol.py`（hash `2c61b2594ce3…`） |
| commit | runner と no-fit の preflight は `16e4deb`。fit の前である |
| 解析スクリプト | `d975c5d`。どの採点値も読む前に commit した。合成データでのドライランの後、空のケースへの対処と相対曲線の計算を直した（`d609dfa`）。実際の値は読んでいない |
| データ | K_GEN ごとに 1 データ。data_seed = 9260000 + 100·K_GEN |
| 試行 | model_seed = 9300000 + 10000·K_GEN + 100·K + t。初期化と E-step の乱数に使う |
| fit の数 | 5 × 10 × 10 = **500** |
| 推定器 | `run_em_instrumented`（Gaussian-X、Bernoulli-Y、L=10、10 反復、全ペア、consistent、on_nan=fail）。コードは変更していない |
| C_ELBO の MC | M=2048、8 バッチ（PR #164 と同じ予算） |
| 選択 | 試行ごとの argmin（主）。1e−12 以内の同点は小さい K。欠けている K があれば INCOMPLETE（補完しない）。副として、10 試行の平均（全試行が有限なとき）と最小（参考）でも選ぶ |
| 失敗の扱い | 再試行・seed の変更・反復の追加はしない |

## 4. preflight（fit なし、`preflight.json`）

- seed：505 個がすべて一意で、過去の CSV の seed 値と衝突しなかった。
- **真のパラメータの状態での採点**：5 データとも、Laplace は OK、ELBO も OK、恒等式の残差は 3e−11 以下だった。

**生成したデータ**

| K_GEN | w0 | w | Y の密度 | 確率が 0.01 未満または 0.99 超のペアの割合 |
|---|---:|---:|---:|---:|
| 1 | −1.58 | 2.48 | 0.269 | 0.10 |
| 3 | 0.075 | −5.09 | 0.504 | 0.52 |
| 5 | 0.003 | −1.41 | 0.499 | 0.13 |
| 7 | 0.449 | −4.40 | 0.518 | 0.64 |
| 9 | −2.96 | −4.32 | 0.412 | 0.73 |

**仮定した w の規則の結果**
- w = 3·N(0,1) により、|w| は 1.4〜5.1 になった（K_GEN=1 だけ正）。
- **Y は強く飽和している**（多くのペアで確率が 0 か 1 に近い）。w の符号は、seed による偶然である。
- X の信号は非常に強い。$F^	op F$ の最小固有値は 14〜268 で、雑音の分散は 0.1 である。
- これらは、**論文の記載ではない仮定に依存する**。
