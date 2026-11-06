# 05 K 選択基準の数学的な比較（2026-11-06）

機械可読版：`criterion_definition_matrix.csv`。用語は `reports/k_selection_theory/paper_bic_reproduction_alignment_20260904.md` に合わせる。

- Q1：条件付き（plug-in）尤度 ln p(X, Y | Z, θ)
- Q2：完全データ対数尤度 ln p(Z, X, Y | θ) = ln p(Z) + ln p(X|Z, θ) + ln p(Y|Z, θ)
- Q3：観測データの周辺尤度 ln p(X, Y | θ) = ln ∫ p(Z, X, Y | θ) dZ（Schwarz BIC が近似する対象）

## 1. 各系統のスコア

| スコア | 当てはまり項 | Z の扱い | 罰則 | Fig. 3 相当のスケール |
|---|---|---|---|---|
| 論文 Eq.(26) | Q1（Eq.16） | 記載なし | (k+1)d − k(k−1)/2 | −2 ln L が 1.6〜193（k ≥ k\*） |
| MATLAB の Q | Q2 の MC 平均（L 個） | 事後標本 | なし（BIC を計算しない） | Q/L ≈ −6,500（d=15、k=3） |
| Python 再現 | Q2 の plug-in（最後の Z 1 個） | 30 反復の最大、3 初期値の最小 | (k+1)d − k(k−1)/2 | −2 ln L は 6,000〜20,000 |
| 初期 ExpFam calc_bic_dual | Q2 の MC 平均（Poisson の −ln y! 補正つき） | 最終反復の L 標本 | kd − k(k−1)/2（Gaussian-X は +d、Gaussian-Y は +1） | — |
| 系列 E の C_Q | 同上 | 同上、2 start の平均 | 同上 | — |
| 系列 E の C_Lap | Q3 のラプラス近似 | Z の mode と曲率 | 同上 | — |

（C_Q は Q ベースの complete-data / ICL 型の基準で、Schwarz BIC ではない。C_Lap はラプラス近似で、厳密な周辺尤度ではない。）

## 2. 1 次元増やしたときの差（DERIVED）

C(K) − C(K+1) = −2[ℓ(K) − ℓ(K+1)] − Δ罰則。

- **Q2 型（Python 再現、C_Q、calc_bic_dual）**：Z を `scale_Z` で平均 ‖z_i‖² = K に固定するので、−2 ln p(Z) = nK(1 + ln 2π) + 定数 となり、1 次元ごとに n(1 + ln 2π) = 2.84 n だけ増える（n=150 で 425.7、n=75 で 212.8）。**1 ノードあたり 2.84 の「入場料」**が罰則とは別にかかる。
- **論文 Eq.(26)（Q1）**：p(Z) を含まないので、この入場料はない。Fig. 3 で k > k\* の増分が 4〜37 しかないことは、これと整合する（01 §3。ただし Fig. 3 の ln L が正規化された量である可能性もあり、HYPOTHESIS）。
- **Q3（C_Lap）**：入場料の代わりに、事後分布の体積の項（ln det の差）が入る。
- 罰則の増分：Eq.(26) と calc_bic_dual の (kd − k(k−1)/2) はどちらも 1 次元ごとに (d − K) ln n。n=150、d=15 で、K=4→5 なら 55.1、K=8→9 なら 35.1。

## 3. 何が「同じ」で何が「違う」か

| 比較 | 同じ点 | 違う点 |
|---|---|---|
| 論文 BIC と C_Q | 罰則の形（Gaussian-X） | p(Z) の有無。plug-in か MC 平均か |
| Python 再現と C_Q | p(Z) を含む Q2 | plug-in の最良反復と、最終状態の MC 平均 |
| C_Q と C_Lap | 同じ fit 状態 | Q2 と Q3 の近似 |
| 論文 BIC と Schwarz BIC | 名前 | Q1 か Q3 か |

## 4. 重要度の評価（Q7 への答え）

- **歴史的なデータでは、基準の違いは結果を変えなかった**（OBSERVED）：同じ 5 データで、Python の plug-in（p(Z) あり）、C_Q、C_Lap がすべて k\* を選び、過小側の余裕は 1,500〜3,500（03 §3）。この余裕に比べれば、入場料 426 も罰則の差も小さい。
- **canonical の境界付近では、基準の違いが結果を変える**（OBSERVED、PR #155・#139）：n=75 では C_Q 0/8、C_Lap 3/8。n=100 では C_Q 1/8、C_Lap 8/8。C_Q の入場料（n=75 で 1 次元 213）は、canonical の余裕（C_Q −129、C_Lap −11）と同じ大きさである。
- 論文 Eq.(26) を文字どおり（Q1、p(Z) なし）使ったら canonical で何が選ばれるかは、**まだ計算されていない**。既存の保存状態から plug-in の Q1 を計算できるかは 07 の E1 で確認する。ただし、Q1 は Z を自由パラメータとして当てはめる量なので、K を増やすほど当てはまりが良くなり、罰則 (d−K) ln n だけでは過大選択が起きうる（Fig. 3 で過大側の余裕が 4〜21 しかないことと整合する。HYPOTHESIS）。
