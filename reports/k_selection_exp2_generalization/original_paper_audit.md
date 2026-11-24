# 元論文 Experiment 2 の一次確認

- 作成日：2026-11-25。branch `experiment/k-selection-exp2-generalization-20261125`（基点 `origin/main` = `1b5030b`）。
- 一次資料：`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`（Mikawa et al. 2024, NOLTA 15(2)）を `pdftotext -layout` で読んだ。
- **読めなかったもの：Fig. 3 の画像**（このリポジトリの環境には PDF のページ画像を描画する手段がない）。Fig. 3 の曲線を数値化したデータは、別 branch の既存監査にある（`origin/research/historical-code-lineage-audit-20261106` の `paper_fig3_vector_vertices.csv`。main には未統合で、本監査では検証していない）。
- ラベル：FACT（一次資料で確認）／ASSUMED（一次資料に書かれておらず、仮定が必要）／UNKNOWN（一次資料から決められない）。

## 1. 論文に書かれていること

| 項目 | 内容 | 根拠 | ラベル |
|---|---|---|---|
| 研究問い | BIC で潜在次元を決められるか | §4.1「(2) whether the number of dimension … can be decided using the BIC」、§4.3 | FACT |
| 真の次元 | k* = 1, 3, 5, 7, 9 | §4.3「the true dimension k of the latent vector is set to be 1, 3, 5, 7, 9」 | FACT |
| 候補 | k = 1〜10 | §4.3「we calculate the BIC when k ranges from 1 to 10 for each k*」 | FACT |
| n, d | 150, 15 | §4.1「Experiments 1 and 2 are done … fixing n = 150, d = 15」 | FACT |
| L、反復数、試行数 | 10, 10, 10 | Table I。**ただし pdftotext ではレイアウトが崩れ、各値と行の対応は §4.1〜4.3 の本文と合わせて復元した** | FACT（本文と整合）。表の行対応は読み取り上の復元 |
| 試行の意味 | 「10 trials … with different initial values」 | §4.1 | FACT |
| 試行の間でデータを共有するか | 書かれていない | — | **ASSUMED**（同じデータで初期値だけ変えたと読むのが自然） |
| 試行の集計 | Exp. 1・3・4 は反復ごとの 10 試行平均と、全試行・全反復での最小値。**Exp. 2 の集計方法は書かれていない** | §4.1 | **UNKNOWN** |
| 判定 | 「If the best (smallest) BIC is given when k = k*, then it is suitable …」「From Fig. 3 … the BIC is smallest when k = k* for each k*」 | §4.3 | FACT |
| Fig. 3 | 横軸 k、縦軸 BIC。k* ごとに 1 本の曲線 | §4.3 | FACT（本文）。画像は未確認 |

**Fig. 3 の読み方の制約（FACT からの帰結）**：論文は k* ごとに 1 本の曲線の argmin が k* であることしか述べていない。**「各 k* で 10 試行すべて（10/10）正解した」とは書いていない。** 曲線が 1 試行なのか、10 試行の平均なのか、最良の試行なのかも書いていない。

## 2. 生成モデル（§4.1 Steps 1〜3）

| Step | 本文 | ラベル |
|---|---|---|
| 1 | σ_z² = 1、z_i ~ N(0, σ_z² I) | FACT |
| 2 | 「Set the random values for w0 and w」、s_ij を Eq.(2) で計算し、y_ij ~ Bern(s_ij) | 分布・尺度は **UNKNOWN** |
| 3 | x_i ~ N(F z_i + μ_x, Σ)、F の各成分 ~ N(0, 10)、Σ = diag(0.1, …, 0.1)、μ_x = 0 | FACT。**10 が分散か標準偏差かは UNKNOWN** |
| 正規化 | 人工データの正規化は書かれていない | FACT（記述なし） |

## 3. 尤度と BIC（Eq.(16)・Eq.(26)）

- **Eq.(16)（FACT）**：L = Π_i p(x_i | F, z_i, μ_x, Σ) · Π_{i≠j} p(y_ij | z_i, z_j, w0, w)^{1/2}。**Z を条件とした尤度**で、Z を積分した周辺尤度ではなく、ln p(Z) も含まない。i≠j の 1/2 乗は、無向ペアを 1 回ずつ数えることと同じ。
- **Eq.(26)（FACT）**：BIC = −2 ln L + ((k+1)d − k(k−1)/2) ln n。
  - 罰則のパラメータ数は、F の kd から回転の k(k−1)/2 を引き、Σ の d を足したもの。
  - μ_x（d 個）、w0・w、σ_z² は数えていない（FACT：式に現れない）。
  - log の中は n（ノード数）。
- **Z をどこで評価したか（UNKNOWN）**：MC サンプル、事後平均 μ_i、最終サンプルのどれかが書かれていない。
- **理由づけ（FACT）**：「BIC possesses consistency in this regard」。この主張の根拠はこのモデルについては示されていない（本研究の canonical docs の立場は `RESEARCH_MASTER.md` §17.4：特異性・境界・有効標本数の未定義のため、Schwarz BIC の正当化はこのモデルでは成り立たない）。

## 4. 推定法の違い（元論文と本研究の正式モデル）

| 項目 | 元論文 | 本研究の正式モデル（`AGENTS.md` §1） | ラベル |
|---|---|---|---|
| μ_x | 推定する（Eq.(11)） | なし（X 切片なし） | FACT・**異なる** |
| σ_z² | 推定する（Eq.(14)） | 1 に固定し、各 E-step 後に scale_Z | FACT・**異なる** |
| E-step の Y 側 | 1/2 あり（Eq.(22)/(23)） | extra 1/2 なし（採用式） | FACT・**異なる**（KI-001） |
| X family | Gaussian のみ | G/B/P | — |
| Y family | Bernoulli のみ | G/B/P | — |
| w0・w の更新 | Adam | 系列による（fixed は Newton 型の `calc_w0`/`calc_w`） | 異なる |

## 5. 元論文の実装（MATLAB）が使えるか

- リポジトリの `Mato Lab Program/` に MATLAB コードがある。**Exp. 2 の driver と、Eq.(26) を計算するコードはない**（FACT。`NOLTA_exp_ver3_revise_batch.m` は L を変える実験で、K のループは `kT` を渡すので実際には K を変えていない）。
- `DecideNumFactor.m` は **X だけの因子分析（`factoran`）による BIC** で、罰則 `(i+1)d − i(i−1)/2` は Eq.(26) と同じだが、**どのスクリプトからも呼ばれていない**（FACT。既存監査 `historical_code_lineage_audit_20261106/02_matlab_code_audit.md` と一致。その監査は main に未統合）。
- `setTrueParams.m` は本文と違う：
  - `rng(1980)` で乱数を固定している。
  - Z を列ごとに正規化している（`normalize`）。
  - varF = 5 で、F の各行を √0.9 のノルムに正規化している。
  - X を `normalize` している。
  - w0 = randn、w = 3·randn。
  - （FACT：コードを直接確認した。）
- MATLAB はこの環境にない。

**結論：元論文の実装による Experiment 2 の厳密な再現はできない。** 再現の手順（Z の評価点、試行の集計、w0・w の生成）が UNKNOWN で、Exp. 2 のコードもないため。本研究でできるのは「元論文の条件（n, d, k*, 候補 K, L, 反復, 試行数）に対応させた比較」までである。**「厳密な再現」とは呼ばない。**
