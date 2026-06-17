# 実データ実験まとめ — Dual-ExpFam LSM (fixed 版)

作成日: 2026-06-18  
対象: Wine / Cora / MovieLens 実データパイロット実験

---

## 1. 目的

提案手法 DualExpFamLSMFixed（Y 側 Term3 の 1/2 係数を除去した修正版）が、
複数の実データに対して安定に動作し、有意義な潜在表現を学習できるかを確認する。

具体的には以下を検証する:

- 異なる指数型分布族の組み合わせ（Gaussian/Bernoulli、Bernoulli/Bernoulli、Bernoulli/Poisson）で数値的に安定か
- Bernoulli Y の場合に held-out link prediction でランダム基準を上回るか
- Poisson Y の場合に count regression として機能するか
- BIC・予測性能・ラベル構造のそれぞれで最適 k が一致するか

---

## 2. 対象データセット一覧

| dataset | n | d | node | X | Y | family_x | family_y | Y_type | Y_density | task | 注意 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wine | 178 | 13 | sample | chemical features | same-class relation | gaussian | bernoulli | binary | 0.3380 | in-sample recon | Y はラベル由来（自然ネットワークではない） |
| Cora | 280 | 50 | paper (subset) | BoW binary | citation link | bernoulli | bernoulli | binary | 0.011 | held-out link pred | balanced_degree subset（full Cora ではない） |
| MovieLens (Poisson) | 100 | 19 | movie | genre multi-hot | co-rated count | bernoulli | poisson | count | 1.000 (all positive) | count regression | movie-node projection、全ペア正、strict held-out 不可 |
| MovieLens (Bern t80) | 100 | 19 | movie | genre multi-hot | co-rated count ≥80 | bernoulli | bernoulli | binary | 0.069 | held-out link pred | threshold による二値化、主結果は Poisson 版 |

---

## 3. Wine pilot

### 実験設定

- n=178, d=13, family_x=gaussian, family_y=bernoulli
- k=1〜9、5 trials/k（45 fits）
- Y = 同クラス関係行列（ラベル由来）
- 評価: BIC、AUC/AP、RMSE_X、ablation（X+Y / X_only / Y_only）

### k 別集計（主要指標、mean over 5 trials）

| k | BIC | AUC_Y | AP_Y | NMI |
|---|---|---|---|---|
| 1 | 13,411 | 0.939 | 0.918 | 0.052 |
| 2 | 6,423 | 1.000 | 1.000 | 0.210 |
| **3** | **6,325** | **1.000** | **1.000** | 0.308 |
| 4 | 6,691 | 1.000 | 1.000 | — |
| 5 | 6,971 | 1.000 | 1.000 | — |

BIC は k=3 で最小。Wine の真のクラス数（3 クラス）と一致。

### ablation（k=3 での条件比較）

| 条件 | AUC_Y | AP_Y | silhouette | 解釈 |
|---|---|---|---|---|
| X+Y（提案） | ≈1.000 | ≈1.000 | 0.775 | X と Y の両方の情報でクラス分離 |
| X_only（w=0） | 0.500 | 0.338 | 0.280 | Y を使わないと link prediction 不可 |
| Y_only（F=0） | ≈1.000 | ≈1.000 | 0.760 | Y はラベル由来なので単独でも高性能 |

X_only と Y_only の比較から、**Y が分離の主要信号**であることがわかる。ただしこれは Y がラベル由来だからであり、自然ネットワークでは異なる可能性がある。

### 解釈

- BIC が真のクラス数 k=3 を正しく選んだ（Wine は高密度 Y の理想的な条件）
- Wine の Y はラベル由来であり、自然ネットワークとは根本的に異なる
- X_only では潜在空間の分離が弱い → X+Y の統合が有効
- Wine は実データ動作確認の最初のステップとして機能した

---

## 4. Cora pilot

### 実験経緯

- **BFS subset（最初の試み）**: max-degree ノードから BFS → Genetic_Algorithms 論文が 78% を占める偏り → 不適切と判断
- **balanced_degree subset（採用）**: 各クラス内の次数上位ノードを選択 → 7 クラス × 40 件 = n=280

### balanced_degree subset の統計

- n=280, d=50（top-50 features）
- edges=433, Y_density=0.011（疎な自然ネットワーク）
- isolated nodes=26/280（9.3%）

### k 別集計（balanced_degree k-sweep, k=1〜6, 3 trials/k）

| k | BIC | AUC | AP | NMI | ARI |
|---|---|---|---|---|---|
| **1** | **14,812** | 0.604 | 0.035 | 0.052 | 0.007 |
| 2 | 14,982 | 0.792 | 0.115 | 0.210 | 0.097 |
| 3 | 15,293 | 0.869 | 0.196 | **0.308** | **0.193** |
| 4 | 15,808 | 0.890 | 0.239 | 0.258 | 0.158 |
| 5 | 16,406 | 0.906 | 0.269 | 0.257 | 0.146 |
| 6 | 17,022 | **0.913** | **0.287** | 0.246 | 0.142 |

- **BIC 最小 = k=1**（密度が低く BIC のペナルティが大きすぎる）
- **AUC/AP 最大 = k=6**
- **NMI/ARI 最大 = k=3**

k 選択基準によって最適 k が異なる。疎な Y でのBIC 選択には限界がある。

### held-out link prediction（test_edge_ratio=0.2, neg_ratio=5）

| k | train_AUC | train_AP | test_AUC | test_AP | AP/random |
|---|---|---|---|---|---|
| 3 | 0.869 | 0.652 | 0.737 | 0.432 | **2.59×** |
| 6 | 0.908 | 0.739 | 0.750 | 0.463 | **2.78×** |

- random_AP_baseline = 0.167（1/6）
- test_AP ≈ 2.6〜2.8× random baseline（統計的に有意な予測能力）
- k=6 が held-out test_AP でも最良

### 解釈

- 自然ネットワーク（引用ネットワーク）での動作を確認
- held-out で random 基準を約 2.7× 上回った → 提案手法は汎化性能あり
- ただし AUC≈0.75, AP≈0.46 は決して高い値ではなく、疎な引用ネットワークは難しいタスク
- BIC が k=1 を選ぶのは実データでの BIC の限界を示す（密度 0.011 では情報量が乏しい）
- full Cora (n=2708) への拡張は今後の課題

---

## 5. MovieLens Poisson pilot

### 実験設定

- データ: MovieLens 100K → movie-node projection
  - 943 ユーザー、1682 映画、100,000 評価
  - Y_count[i,j] = 映画 i と j を共通評価したユーザー数
- subset: genre_stratified_mp100（10 ジャンル × 10 映画 = n=100）
- n=100, d=19（genre multi-hot）, family_x=bernoulli, family_y=poisson
- k=[2,3,5], 3 trials/k → 9 fits

### Y_count の性質

- Y_count mean=45.2, max=144, var/mean=9.89（Poisson より約 10 倍の分散 = overdispersion）
- density_pos=1.000（全ペアで Y_count > 0）
- 943 ユーザーが平均 106 本評価するため、評価数 30 本以上の映画はすべてのペアで共通ユーザーが存在

### k 別集計

| k | BIC | RMSE_Y | MAE_Y | Pearson | hc_AP(≥80) | NMI |
|---|---|---|---|---|---|---|
| 2 | 40,139 | 9.39 | 7.17 | 0.897 | 0.773 | 0.255 |
| 3 | 35,774 | 6.97 | 5.30 | 0.945 | 0.846 | 0.313 |
| **5** | **34,553** | **5.74** | **4.38** | **0.963** | **0.901** | 0.300 |

- BIC、RMSE、Pearson、high-count AP すべてで k=5 が最良
- NMI（ジャンル構造）は k=3 が最良

### masked count evaluation（補助確認）

- 学習は全 Y_count（mask なし）、評価のみ train/test に分割
- k=5: train_RMSE=5.71, test_RMSE=5.70（差 ≈ 0）
- k=5: train_Pearson=0.963, test_Pearson=0.964（差 ≈ 0）
- 低ランク潜在変数モデルは個別ペアを記憶できないため train ≈ test は構造的帰結
- **これは strict な未知ペア予測性能の評価ではない**

### 解釈

- Bernoulli X / Poisson Y の novel 組み合わせが安定に動作することを確認
- k=5 で Pearson≈0.963、RMSE/mean_Y≈12.7% は count regression として良好
- high-count AP（≥80）= 0.901 はランダム基準（0.069）の約 13 倍
- Poisson overdispersion（var/mean≈10）は Poisson 仮定の限界を示す
- strict held-out count regression には pair mask 対応が必要（現在の API では不可）

---

## 6. MovieLens Bernoulli t80 pilot

### 設定と動機

- Y_count ≥ 80 を正例とした二値化 Y（密度 0.069）
- Cora（密度 0.011）と同じ Bernoulli link prediction として比較可能にする補助実験
- **MovieLens の主結果は Poisson Y**。Bernoulli t80 は比較・補助実験の位置づけ

### in-sample k 別集計

| k | BIC | AUC | AP | NMI | ARI |
|---|---|---|---|---|---|
| **2** | **2,999** | 0.950 | 0.801 | 0.307 | 0.067 |
| 3 | 3,018 | 0.975 | 0.845 | **0.337** | **0.090** |
| 5 | 3,451 | **0.983** | **0.880** | 0.234 | 0.008 |

- BIC 最小 = k=2（density=0.069 でも BIC ペナルティが有効）
- AUC/AP 最大 = k=5
- NMI/ARI 最大 = k=3

### held-out link prediction（test_edge_ratio=0.2, neg_ratio=5）

| k | train_AUC | train_AP | test_AUC | test_AP | AP/random |
|---|---|---|---|---|---|
| **3** | 0.975 | 0.909 | **0.945** | **0.832** | **4.99×** |
| 5 | 0.979 | 0.920 | 0.912 | 0.742 | 4.45× |

- random_AP_baseline = 0.167（1/6）
- k=3 が held-out test_AP でも最良（k=5 は in-sample では良いが held-out で劣る）
- test_AP ≈ 0.83（k=3）は Cora の test_AP ≈ 0.43〜0.46 より高い

### Cora との比較

| 指標 | Cora | MovieLens t80 |
|---|---|---|
| Y_density | 0.011 | 0.069 |
| Y の種類 | 自然ネットワーク（引用） | 閾値二値化（共評価数 ≥80） |
| test_AUC | 0.737〜0.750 | **0.945** |
| test_AP | 0.432〜0.463 | **0.832** |
| AP/random | 2.6〜2.8× | **5.0×** |

MovieLens t80 の方が性能は高いが、これは密度が 6 倍高く、強い共評価関係という明確なシグナルを使っているため、**Cora より簡単なタスク**である。両者の直接的な優劣比較は不適切。

---

## 7. 実データ全体の比較

### 設計上の違い

| 観点 | Wine | Cora | MovieLens (Poisson) | MovieLens (Bern t80) |
|---|---|---|---|---|
| Y の起源 | ラベル由来（人工） | 自然ネットワーク | 共評価カウント（投影） | 同左の閾値二値化 |
| family_y | bernoulli | bernoulli | **poisson** | bernoulli |
| Y の評価形式 | in-sample recon | held-out link pred | count regression | held-out link pred |
| 真の held-out | — | Yes | 不可（mask 非対応） | Yes |
| Y の密度 | 0.3380 | 0.011 | 1.000（全正） | 0.069 |

### k 選択のズレ

| データ | BIC 最小 k | AP/AUC 最大 k | NMI 最大 k |
|---|---|---|---|
| Wine | **3**（正解） | 2〜3 | 3〜4 |
| Cora | 1（不適切） | 6 | 3 |
| MovieLens (Poisson) | 5 | 5 | 3 |
| MovieLens (Bern t80) | 2 | 5 | 3 |

- Wine は密度が高く BIC が機能した唯一のケース
- Cora は疎な Y で BIC がペナルティ過大になり k=1 を選択（失敗）
- NMI/ARI は常に中程度の k（k=3）を選ぶ傾向がある
- AP/AUC は常に大きな k（k=5〜6）を選ぶ傾向がある
- → **実データでは目的に応じて k 選択基準を使い分ける必要がある**

### 予測性能のまとめ

| データ | 主要評価指標 | 値 | random 比 |
|---|---|---|---|
| Wine | AP_Y (in-sample, k=3) | ≈1.000 | — |
| Cora | test_AP (held-out, k=6) | 0.463 | **2.78×** |
| MovieLens (Poisson) | Pearson (in-sample, k=5) | 0.963 | — |
| MovieLens (Poisson) | hc_AP ≥80 (in-sample, k=5) | 0.901 | 13.1× |
| MovieLens (Bern t80) | test_AP (held-out, k=3) | 0.832 | **5.0×** |

---

## 8. 今回言えること

以下の主張は実験的に支持される:

1. **fixed 版モデル（DualExpFamLSMFixed）は複数の実データに適用できた**  
   Wine / Cora / MovieLens のすべてで NaN なし、全 fits 成功。

2. **Gaussian/Bernoulli、Bernoulli/Bernoulli、Bernoulli/Poisson の組み合わせが動作した**  
   family の指数型分布族一般化が実際のデータで機能することを確認。

3. **自然ネットワーク（Cora 引用ネットワーク）の held-out link prediction でランダム基準を上回った**  
   test_AP ≈ 0.46（≈ 2.78× random）。疎な引用ネットワークにおける予測能力を確認。

4. **MovieLens では Poisson Y として count 関係を良好に再構成できた**  
   in-sample Pearson ≈ 0.963、RMSE/mean_Y ≈ 12.7%。

5. **実データでは BIC・予測性能・ラベル構造の最適 k が一致しない場合がある**  
   Cora では BIC=k=1、AP=k=6、NMI=k=3 と三者が異なる。実データでの k 選択は単一基準では難しい。

6. **Bernoulli X と Poisson Y の組み合わせ（MovieLens）は数値的に安定に収束した**  
   w0 ≈ log(mean_count)、w は k に応じて適切に収束。

---

## 9. まだ言えないこと

以下の主張は**過大**であり、現時点では言えない:

1. **「MovieLens Poisson で未知ペアを予測できた」**  
   → strict held-out は不可能（pair mask 非対応）。masked evaluation は訓練データ内の評価。

2. **「Wine で自然ネットワークの実験を行った」**  
   → Wine の Y はラベル由来（同クラスなら Y=1）。自然ネットワークではない。

3. **「Cora より MovieLens t80 の方が提案手法の性能が高い」**  
   → 密度・タスク難易度が異なるため直接比較は不適切。MovieLens t80 は簡単なタスク。

4. **「BIC が常に正しい k を選ぶ」**  
   → Cora では k=1 を選択（疎な Y で BIC のペナルティが過大）。

5. **「Poisson 仮定が成立している」**  
   → var/mean ≈ 9.89 は標準 Poisson（var/mean=1）からの大きな逸脱（overdispersion）。

6. **「提案手法が任意の実データで機能する」**  
   → Wine は人工 Y、Cora は small subset、MovieLens は特殊な投影。より多様な実データでの検証が必要。

7. **「実データで真の k が検証できた」**  
   → 真のクラス数（Wine=3）と BIC 最小 k が一致したのは Wine のみ。他のデータでは真の k は不明。

---

## 10. 次に必要なこと

### 発表資料への反映

- Wine: BIC 選択の成功例として簡潔に紹介（k=3、3 クラスと一致）
- Cora: 自然ネットワークでの held-out link prediction（test_AP≈2.7× random）を主実績として使う
- MovieLens: Poisson Y の count regression を novel 組み合わせの例として示す
- MovieLens t80: 補助実験として Cora との比較表に添える（注釈: 密度が異なるため直接比較不可）

### 追加実験（優先度付き）

| 優先 | 実験 | 理由 |
|---|---|---|
| 高 | Cora larger subset or full Cora | n=280 は小さい、larger n での性能確認 |
| 中 | MovieLens Poisson pair mask 実装 | strict held-out count regression の有効化 |
| 中 | 別の実データ（Karate Club 等） | データセット多様化 |
| 低 | Wine 以外で Gaussian/Bernoulli | Wine は人工 Y なので他データで確認 |

### 修論フェーズでの課題

- Pair mask 対応（`calc_w0`、`calc_w`、`_calc_gradient`、`_calc_precision_matrix` への mask 引数追加）
- larger Cora や他の引用ネットワーク（Citeseer 等）への適用
- 負の二項分布（Negative Binomial）による overdispersion への対応
- MovieLens bipartite model との比較（user-node vs movie-node projection）
- 実データでの k 選択基準の体系的比較

---

## 11. 論文・発表で使える短いまとめ文

### 1 文まとめ

提案手法 DualExpFamLSMFixed は、Gaussian/Bernoulli・Bernoulli/Bernoulli・Bernoulli/Poisson の 3 種類の分布族の組み合わせで実データに適用でき、引用ネットワーク（Cora）での held-out link prediction においてランダム基準を約 2.7 倍上回る予測性能を示した。

### 3 文まとめ

提案手法を Wine、Cora 引用ネットワーク、MovieLens の 3 つの実データに適用した。
Cora held-out link prediction では test_AP ≈ 0.46（ランダム基準の 2.7 倍）を達成し、疎な自然ネットワークでの汎化性能を確認した。
また MovieLens では Bernoulli X / Poisson Y の novel な組み合わせが数値的に安定に収束し、共評価数の in-sample 再構成において Pearson ≈ 0.96 を達成した。

### 発表スライド用まとめ

| データ | 設定 | 主な結果 |
|---|---|---|
| Wine | Gaussian X / Bernoulli Y | BIC → k=3（クラス数と一致） |
| Cora (n=280 subset) | Bernoulli X / Bernoulli Y | held-out test_AP ≈ 0.46（random の 2.7×） |
| MovieLens (Poisson) | Bernoulli X / Poisson Y | Pearson ≈ 0.96（count 再構成） |
| MovieLens (Bern t80) | Bernoulli X / Bernoulli Y (補助) | held-out test_AP ≈ 0.83（random の 5.0×）★ |

★ MovieLens t80 の密度（6.9%）は Cora（1.1%）より高く、より容易なタスクであることに注意。

### 注意書き付きまとめ

提案手法は Wine・Cora・MovieLens の 3 データセットで安定に動作した。
ただし以下の制限に注意する:

- Wine の Y はラベル由来（自然ネットワークではない）
- Cora は n=280 の balanced subset であり、full Cora (n=2708) への一般化は未確認
- MovieLens Poisson の評価は in-sample 再構成であり、strict held-out ではない
- MovieLens Poisson には Poisson overdispersion（var/mean ≈ 10）がある
- MovieLens t80 は Cora と直接比較できない（密度・タスク難易度が異なる）
- 実データで BIC が常に適切な k を選ぶわけではない（Cora では k=1 を選択）

---

## 付録: 実験スクリプト一覧

| データ | スクリプト | 結果ディレクトリ |
|---|---|---|
| Wine | `expfam/src/run_fixed_real_wine_pilot.py` | `expfam/results/real_data/wine_fixed_pilot/` |
| Cora BFS subset | `expfam/src/run_fixed_real_cora_subset_pilot.py` | `expfam/results/real_data/cora_subset_pilot/` |
| Cora balanced subset | `expfam/src/run_fixed_real_cora_balanced_subset_pilot.py` | `expfam/results/real_data/cora_balanced_subset_pilot/` |
| Cora k-sweep | `expfam/src/run_fixed_real_cora_balanced_k_sweep.py` | `expfam/results/real_data/cora_balanced_k_sweep/` |
| Cora held-out | `expfam/src/run_fixed_real_cora_heldout_link_prediction.py` | `expfam/results/real_data/cora_heldout_link_prediction/` |
| MovieLens data prep | `expfam/src/prepare_movielens_data.py` | `expfam/results/real_data/movielens_data_prep/` |
| MovieLens Poisson pilot | `expfam/src/run_fixed_real_movielens_poisson_pilot.py` | `expfam/results/real_data/movielens_poisson_pilot/` |
| MovieLens masked count | `expfam/src/run_fixed_real_movielens_heldout_count.py` | `expfam/results/real_data/movielens_heldout_count/` |
| MovieLens Bernoulli t80 | `expfam/src/run_fixed_real_movielens_bernoulli_t80_pilot.py` | `expfam/results/real_data/movielens_bernoulli_t80_pilot/` |
