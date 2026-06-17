# MovieLens pilot 設計書

作成日: 2026-06-17
前提: Cora held-out link prediction pilot 完了後の次ステップ

---

## 1. 背景

これまでの実データ実験:

| データ | X | Y | family_x | family_y | 状況 |
|---|---|---|---|---|---|
| Wine | 標準化連続量 (d=13) | クラスラベル由来 (人工) | gaussian | bernoulli | ✓ 完了 |
| Cora | BoW 0/1 (d=50) | 引用関係 (自然ネットワーク) | bernoulli | bernoulli | ✓ 完了 |
| MovieLens | ジャンル multi-hot (d=19) | 共評価関係 (自然関係) | bernoulli | **poisson / bernoulli** | 次ステップ |

Wine と Cora はいずれも Y=Bernoulli だった。
MovieLens は **Y=Poisson（共評価数）** が自然であり、提案手法の指数型分布族一般化の中で最も重要な新規組み合わせを実証できる。

---

## 2. なぜ MovieLens を使うか

**提案手法の主張**:
> Y と X は任意の指数型分布族でよい。

**Wine/Cora での実証**:
- Wine: X=Gaussian, Y=Bernoulli（先行研究と同条件）
- Cora: X=Bernoulli, Y=Bernoulli（両方 Bernoulli）

**MovieLens で追加できる実証**:
- X=Bernoulli（ジャンル multi-hot）, Y=**Poisson**（共評価数）

これにより、Y が離散カウントデータにも適用できることを示す。
またモデルの自然な適用先として解釈しやすい（「共に評価されがちな映画の潜在構造」）。

---

## 3. 現在のモデルと MovieLens の形の違い

### 現在のモデルの仮定

```
z_i  ~ N(0, I_k)              i = 1, ..., n  (ノード数)
y_ij ~ ExpFam_Y(η = w0 + w * z_i^T z_j)    i < j
x_il ~ ExpFam_X(η = f_l^T z_i)             l = 1, ..., d
```

Y は **n × n ノード間関係**（対称）。X は **n × d ノード属性**。

### MovieLens のデータ形式

```
R (943 × 1682) : user × movie 評価行列（1-5 の整数、欠損多数）
```

**二部グラフ**であり、そのまま n × n Y には入れられない。

### 解決策: Projection

同種ノード (movie-movie または user-user) の関係行列 Y に変換する。

---

## 4. 案A: movie-node projection（推奨）

### 概要

```
node = movie   (n ≤ 1682)
X = genre multi-hot vector   (d = 19)
Y_ij = movie i と movie j を共通ユーザーが評価した数
```

### X の設計

`u.item` の genre flags（19 次元 0/1）をそのまま使う。

```
[unknown, Action, Adventure, Animation, Children's, Comedy, Crime,
 Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery,
 Romance, Sci-Fi, Thriller, War, Western]
```

- d = 19（固定、次元削減不要）
- family_x = bernoulli（0/1 multi-hot）
- 1 映画あたりの平均ジャンル数 = 1.72（単一ジャンルが 833 本、複数ジャンルが残り）

### Y の設計候補

| Y_variant | 定義 | family_y | 備考 |
|---|---|---|---|
| Y_count | co-rated count（生カウント） | poisson | 自然・新規・overdispersion 注意 |
| Y_binary | 1 if co-rated_count >= threshold | bernoulli | threshold 設計が必要 |
| Y_jaccard | |A∩B|/|A∪B|（Jaccard類似度） | gaussian | 密度設計は容易だが少し恣意的 |

**推奨**: Y_count（Poisson Y）→ 研究上最も新規性が高い。
**代替**: Y_binary（Bernoulli Y, threshold=20〜50）→ Cora との比較が容易。

### データ統計（実測値）

| 条件 | n | Y density | Y mean | Y var/mean | 備考 |
|---|---|---|---|---|---|
| top-200 movies | 200 | 1.000 | 94.3 | 22.2 | 密すぎる、全ペア連結 |
| random-200 movies | 200 | 0.675 | 6.83 | - | 適度だがジャンル偏り大 |
| random-200, threshold≥5 | 200 | 0.293 | - | - | Y_binary 候補 |
| random-200, threshold≥10 | 200 | 0.180 | - | - | Y_binary 候補 |
| genre-stratified-100 (設計中) | 100 | 0.10〜0.25 想定 | - | - | Pilot 候補 |

**重要な警告**:
- top-N movies は全ペアが共評価（density≈1.0）→ Poisson Y のモデルが退化する
- 人気度のべき乗バイアスが強く、一部映画が Y を支配する
- genre-stratified subset で中程度人気映画を選ぶ必要がある

### メリット / デメリット

| 項目 | 評価 |
|---|---|
| ✓ X が自然（ジャンルは映画の本来の属性） | 高 |
| ✓ Y=Poisson が最も新規な組み合わせ | 高 |
| ✓ 解釈性（「共評価が多い映画は潜在空間で近い」） | 高 |
| ✗ 人気度バイアス（popular 映画が全ペア連結） | 要対処 |
| ✗ ジャンルが sparse（多くは 1-2 ジャンル） | 中程度 |
| ✗ Y_count の overdispersion（var/mean≈22） | 要注意 |

---

## 5. 案B: user-node projection

### 概要

```
node = user   (n = 943)
X = user demographics  (age, gender, occupation)
Y_ij = user i と user j の共同評価映画数 or 評価類似度
```

### X の設計

`u.user` の属性（uid | age | gender | occupation | zip）を使う。

```
X = [age(連続), gender(0/1), occupation_onehot(21次元)] → d ≈ 23
family_x = gaussian (age) or bernoulli (one-hot)
```

※ 混合分布族（age=Gaussian、gender+occupation=Bernoulli）になるため実装が複雑。

### Y の設計

```
Y_ij = user i と user j が共に評価した映画数  → Poisson
Y_ij = 1 if 評価類似度 >= threshold          → Bernoulli
```

### デメリット

| 項目 | 評価 |
|---|---|
| ✗ X が混合型（age=連続, gender/occupation=カテゴリ） | 実装複雑 |
| ✗ zip code は地理情報として扱いにくい | 不使用推奨 |
| ✗ 評価類似度は恣意性が高い | 中程度 |
| ✗ user 数 943 > movie 数サブセット 100〜200 で重い | 要サブセット |
| △ ユーザーの潜在構造（年齢層・性別の映画趣向）は解釈しやすい | 中程度 |

**結論**: 案B は X の混合型問題と実装複雑さから、案A より優先度が低い。

---

## 6. 案C: 二部グラフ拡張は今後の課題

現在のモデルは同種ノードの n × n 関係を仮定している。
user × movie の二部グラフを直接扱うには η_{ij}^Y = u_i^T v_j（u=user latent, v=movie latent）のような拡張が必要であり、
現在の実装の根本的な変更を要する。

**方針**: 二部グラフ拡張は修論フェーズの将来課題とし、今回は案A（movie-movie projection）を採用する。

---

## 7. 各案の比較表

| 項目 | 案A (movie-movie) | 案B (user-user) | 案C (二部) |
|---|---|---|---|
| ノード | movie | user | user + movie |
| X の作り方 | genre multi-hot（19次元） | age/gender/occupation | 不要（または両方） |
| Y の作り方 | 共評価数 or threshold | 共評価数 or 類似度 | 評価値そのもの |
| family_x | bernoulli | gaussian + bernoulli（混合） | — |
| family_y | **poisson** or bernoulli | poisson or bernoulli | Gaussian or Poisson |
| n の想定 | 100〜200 | 100〜300 | 943 × 1682 |
| d の想定 | 19 | 23〜24 | — |
| Y density（想定） | 0.05〜0.30（threshold次第） | 0.15〜0.50 | 6.3% |
| 実装難易度 | 低（X は既存形式と同じ） | 中（X 混合型） | 高（モデル拡張必要） |
| 解釈しやすさ | 高（ジャンル × 共評価） | 中（ユーザー属性） | 高（直接的）だが複雑 |
| 研究の新規性 | 高（Bern/Poisson 新規） | 中（X 混合型は複雑） | 高（将来課題） |
| 主な注意点 | 人気度バイアス、overdispersion | X の混合型処理 | モデル変更必要 |

---

## 8. 最初に採用すべき pilot 設計

**採用: 案A — movie-node projection、genre-stratified subset**

理由:
1. X = genre multi-hot（d=19）は既存実装と全く同じ形式（Bernoulli X）
2. Y = co-rated count（Poisson Y）は提案手法で最も新規な組み合わせ
3. Y = Bernoulli（threshold）は Cora と直接比較できるフォールバック
4. n=100〜200 の小規模 subset で試験可能
5. genre を疑似ラベルとして silhouette/NMI 評価に使える

---

## 9. X の設計

### genre multi-hot（推奨）

```
X = u.item の genre flags（列 6〜24）
X.shape = (n_movies, 19)
family_x = bernoulli
```

**特性**:
- 1682 × 19 の sparse 0/1 行列
- 平均 1.72 ジャンル/映画（多くが 1〜2 ジャンル）
- 全映画がジャンルを持つ（unknown=0 の映画はほとんどなし）
- d=19 は Cora の d=50 より低次元 → 情報量は少ないが pilot には十分

**注意**: Drama(725本), Comedy(505本), Action/Thriller(各251本) に偏りがある。
genre-stratified subset では各ジャンルから均等に選ぶことで偏りを軽減する。

---

## 10. Y の設計

### Y_count（Poisson Y — 第一候補）

```python
# user -> 評価した映画 set を構築
user_movies = {uid: set(mids) for uid, mids in u.data}

# 映画 i と映画 j の共評価数
Y_count[i, j] = len(user_movies が映画 i を含む set ∩ 映画 j を含む set)
```

**期待値の設定**:
- genre-stratified n=100 では co-rated count の平均は 5〜30 程度（映画の人気度次第）
- Poisson モデルの学習には w0 = log(mean_count) で初期化

**問題点**:
- overdispersion: var/mean ≫ 1（top-200 では var/mean=22）
- genre-stratified 中程度人気映画でも var/mean ≈ 5〜15 と予想される
- Poisson model は underdispersion には対応できないが overdispersion は近似的に吸収可能

### Y_binary（Bernoulli Y — フォールバック）

```python
threshold = T  # Y_density が 0.05〜0.15 になるよう調整
Y_binary[i, j] = 1 if Y_count[i, j] >= T else 0
```

**threshold 設計指針** (n=100 genre-stratified 想定):
- 映画の人気度が中程度（評価数 20〜100）→ co-rated count は 5〜30 程度
- threshold=10: density ≈ 0.10〜0.20 (目標)
- threshold=20: density ≈ 0.05〜0.10 (Cora に近い密度)

**推奨**: まず Y_count (Poisson) を試し、学習が不安定なら Y_binary (threshold=10) にフォールバック。

---

## 11. family_x / family_y の候補

| 変数 | family | 理由 | 実装状況 |
|---|---|---|---|
| X (genre) | bernoulli | 0/1 multi-hot、Bernoulli 確率 | ✓ 実装済み |
| Y (co-rated count) | **poisson** | カウントデータ、非負整数 | ✓ 実装済み |
| Y (threshold binary) | bernoulli | 0/1 類似度 | ✓ 実装済み |
| Y (normalized similarity) | gaussian | 連続値 0〜1 | ✓ 実装済み |

**最優先**: `family_x=bernoulli, family_y=poisson`（新規組み合わせ）
**フォールバック**: `family_x=bernoulli, family_y=bernoulli`（Cora と同形式、比較容易）

---

## 12. subset 設計

### Genre-stratified subset（推奨）

Cora の balanced_degree と同様に、各カテゴリから均等にサンプリングする。

#### ジャンル選択

Drama(725), Comedy(505), Action(251), Thriller(251), Romance(247), Adventure(135),
Crime(109), Sci-Fi(101), Horror(92), Mystery(61) を主要 10 ジャンルとする。

```
strategy = genre_stratified_top_rated
per_genre = 10
target_n  = 100   (10 genres × 10 movies)
```

各ジャンルから「評価数が中程度（例: 評価数 30〜200）」の映画を優先選択。
→ 人気バイアスを軽減しつつ Y_count を 0 でなくする。

#### subset 統計の推定

| 条件 | 想定 n | 想定 Y density (Poisson raw) | 想定 Y density (threshold=10) |
|---|---|---|---|
| genre-stratified, 中程度人気 | 100 | 0.30〜0.60 | 0.10〜0.20 |
| genre-stratified, 高人気 | 100 | 0.70〜1.0 | 0.30〜0.60 |

**目標**: Poisson Y の場合、mean(Y) が 10〜50 程度になるよう人気度を調整する。

#### Alternative: popularity-stratified

評価数 30〜200 の映画のみを使い、そこから genre-stratified にサンプリングする。
→ co-rated count の mean が適度な範囲に収まる（5〜30 程度）。

---

## 13. 評価指標

### Y 再構成

| 指標 | Poisson Y | Bernoulli Y |
|---|---|---|
| 二値化 AUC | Y>0 で binary | AUC |
| Average Precision | Y>0 で binary | AP |
| RMSE_Y | sqrt(mean((Y - mu_Y)^2)) | — |
| mean_Y_hat | - | — |

Poisson の場合、AUC/AP のための binary化は Y>0（共評価あり/なし）で行う。

### X 再構成

```
RMSE_X = sqrt(mean((X - sigmoid(Z F^T))^2))  (Bernoulli X)
```

### ラベル構造（genre を疑似ラベルとして使用）

```
silhouette (primary genre で評価)
NMI
ARI
```

### モデル選択

```
BIC (train Y で評価)
```

### Held-out link prediction（Y=Bernoulli の場合のみ直接比較可能）

```
test_AUC
test_AP
random_AP_baseline
```

Poisson Y の held-out 評価:
- Y_test の raw count に対する RMSE
- binary化（Y>0）した AUC/AP

---

## 14. Cora・Wine との比較上の位置づけ

| データセット | 最も示せる特性 | X | Y | n | d |
|---|---|---|---|---|---|
| Wine | X+Y 結合学習の優位性（X=Gaussian） | gaussian | bernoulli (人工) | 178 | 13 |
| Cora | 自然ネットワーク Y での実データ適用 | bernoulli | bernoulli | 280 | 50 |
| **MovieLens** | **Y=Poisson（カウントデータ）への汎用性** | **bernoulli** | **poisson** | **100** | **19** |

**MovieLens が論文で果たす役割**:
提案手法の一般性（任意の指数型分布族）を主張する上で、
Y=Bernoulli だけでなく Y=Poisson への適用例が論拠として必要。
MovieLens はこの最も自然なケース。

---

## 15. 実装ステップ

### Step 1: データ準備スクリプト（`expfam/src/prepare_movielens_data.py`）

```python
# 入力: expfam/data/ml-100k.zip
# 処理:
#   1. u.data (ratings) を読み込み
#   2. u.item (genre flags) を読み込み
#   3. genre-stratified movie subset を選択 (target n=100)
#   4. X = genre multi-hot (n × 19)
#   5. Y_count = co-rated count matrix (n × n)
#   6. Y_binary = (Y_count >= threshold) 各 threshold で
#   7. 統計出力: density, mean, var, overdispersion
# 出力: expfam/data/movielens_pilot/
#   movielens_X.npy, movielens_Y_count.npy
#   movielens_Y_binary_t{threshold}.npy
#   movielens_labels.npy  (primary genre)
#   movielens_movie_ids.npy
#   movielens_data_summary.csv
```

### Step 2: Pilot 実験スクリプト（`expfam/src/run_fixed_real_movielens_pilot.py`）

```python
# Poisson Y パート (primary)
# k = 2, 3, 5 (1 trial each)
# family_x=bernoulli, family_y=poisson

# Bernoulli Y パート (comparison)
# Y_binary (best threshold from data summary)
# k = 2, 3, 5 (1 trial each)

# 評価:
# BIC, binary AUC/AP (Y>0), RMSE_Y, silhouette, NMI, ARI
```

### Step 3: 結果比較

```
Cora (Bern/Bern) vs MovieLens_Poisson (Bern/Poisson) vs MovieLens_Bern (Bern/Bern)
```

---

## 16. 注意点・リスク

### リスク1: Poisson Y の overdispersion

- **問題**: co-rated count は var/mean ≫ 1（測定値: 22 for top-200）
- **対処**: 中程度人気映画を選び var/mean を 5〜10 程度に抑える; RMSE_Y と AUC の両方を報告
- **論文での扱い**: "Poisson Y の近似として有効だが overdispersion は今後の課題" と明記可能

### リスク2: Y が高密度すぎる

- **問題**: 人気映画 top-200 では Y density = 1.0 → 全ペア連結、モデルが η の変動を捉えにくい
- **対処**: 中程度人気（評価数 30〜200）の映画に限定; popularity-stratified が有効

### リスク3: X が低次元・sparse

- **問題**: d=19 はほぼ 1-2 bit しか非ゼロでない movie が多い（単一ジャンル: 833/1682）
- **対処**: genre combination を疑似ラベルとして使うか、複数ジャンルを持つ映画のみをサブセットに含める
- **影響**: X 再構成 RMSE が Y-side の学習と競合する可能性

### リスク4: Poisson Y の Newton 法不安定

- **問題**: Poisson は exp(η) が発散しやすい; η が大きくなると数値不安定
- **対処**: w0 = log(mean_count) で informed initialization; newton_alpha を 0.25 から始める
- **既存**: model_dual_expfam_fixed.py には NaN fallback（alpha 半減）が実装済み

### リスク5: 評価基準の Cora との非対称性

- **問題**: Cora では Y 密度 0.011 なので AP が意味を持つ。Y 密度 0.15 では AP は高くなりやすい。
- **対処**: random_AP_baseline = Y_density を常に報告し、AP / baseline_AP の倍率で比較。

---

## 17. 次に実行すべきスクリプト案

### Phase 1: データ確認（実験前）

```bash
# データ統計確認スクリプト（実行時間 < 1分）
python expfam/src/prepare_movielens_data.py
```

確認事項:
- genre-stratified n=100 での Y_count の mean/var/density
- threshold=10, 20, 30 での Y_binary density
- X の sparsity（per-movie genre count 分布）

### Phase 2: Poisson Y pilot（primary）

```bash
# k=2,3,5、各1試行、合計3 fits
# 目標: Poisson Y モデルが動作するか確認
python expfam/src/run_fixed_real_movielens_pilot.py --family_y poisson
```

### Phase 3: Bernoulli Y pilot（comparison）

```bash
# k=2,3,5、各1試行、threshold 自動選択
# 目標: Cora との直接比較
python expfam/src/run_fixed_real_movielens_pilot.py --family_y bernoulli
```

---

## まとめ

| 項目 | 選択 |
|---|---|
| 採用案 | 案A: movie-node projection |
| X | genre multi-hot (d=19, family_x=bernoulli) |
| Y（第一候補） | co-rated count（family_y=poisson） |
| Y（フォールバック） | threshold binary（family_y=bernoulli, threshold=10〜20） |
| n_subset | 100（genre-stratified、中程度人気映画） |
| k_values | 2, 3, 5 |
| 疑似ラベル | primary genre（silhouette/NMI/ARI の基準） |
| 重要な注意点 | 人気度バイアス対策・overdispersion・Y 密度管理 |
| Cora との差別化 | Y=Poisson（新規）、X が 19 次元（低次元）、疑似ラベルが多値 |
