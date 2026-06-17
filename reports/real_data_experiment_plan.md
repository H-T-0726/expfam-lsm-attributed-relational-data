# 実データ追加実験 計画書

作成日：2026-06-17  
対象実装：`DualExpFamLSMFixed`（Y-side 0.5 なし）

---

## 1. 背景

Dual-ExpFam LSM は、関係データ Y と属性データ X の両方を指数型分布族（Gaussian / Bernoulli / Poisson）でモデル化できる潜在構造モデルである。

先行研究（Mikawa et al., NOLTA 2024）は X=Gaussian, Y=Bernoulli に固定されていた。本研究はこれを一般化し、任意の分布族の組み合わせを扱える。

固定実装 `DualExpFamLSMFixed` により、人工データ上での Exp1〜4 が完了した。次のステップとして、複数の実データセットに適用し、「どのデータで・どの分布族・どの X/Y 情報が効くか」を検証する。

---

## 2. fixed版人工データ実験で確認済みのこと

| 実験 | 結果 |
|------|------|
| Exp1 BIC K=1〜9 | 全シナリオ k=3 が 10/10 試行で選択（k=9 まで拡張しても過大次元に誤らない） |
| Exp2 n-sweep | n 増加で全シナリオ RMSE(Z) 改善（A: −40%, B: −17%, C: −62%） |
| Exp3 d-sweep | d 増加で Scen A は −22.5%、Scen C は flat。Scen B は中央値ベースで改善も outlier あり |
| Exp4 mismatch | 誤指定で Scen A 最大 4.34×、B 最大 9.04×、C 最大 40.37× 悪化（3×3 grid, ablation 除く） |
| 主要図生成 | n-sweep 図・mismatch 図を `expfam/figures/fixed_official/` に出力 |

これらにより、fixed 実装の基本的な信頼性は人工データ上で確認済み。

---

## 3. 実データ実験の目的

1. **実データでの動作確認**：fixed 版が実データ（Y が既知スパースネットワーク、X が実属性）で正常に動作するか
2. **分布族の有効性検証**：どの (family_x, family_y) 組み合わせが BIC や予測精度で優れるか
3. **X/Y の情報量比較**：X+Y / X_only / Y_only で推定精度がどう変わるか（ablation）
4. **比較ベースライン**：先行研究（X=Gaussian, Y=Bernoulli）との性能比較

---

## 4. モデルが要求するデータ形式

### 入力

| 変数 | 形状 | 型 | 説明 |
|------|------|----|------|
| `X` | `n × d` | numpy array | 属性行列。各行がノード、各列が属性次元 |
| `Y` | `n × n` | numpy array | 関係行列（対称）。対角は 0。上三角のみ学習に使用 |
| `family_x` | str | — | `"gaussian"`, `"bernoulli"`, `"poisson"` のいずれか |
| `family_y` | str | — | 同上 |
| `k` | int | — | 潜在次元数（BIC で選択可） |

### family_x の選択基準

| X の性質 | 推奨 family_x |
|----------|--------------|
| 連続値・平均 0 前後（標準化済み） | `gaussian` |
| 0/1 バイナリ（存在有無、タグ等） | `bernoulli` |
| 非負整数（カウント、単語頻度等） | `poisson` |

### family_y の選択基準

| Y の性質 | 推奨 family_y |
|----------|--------------|
| 0/1 隣接行列（友人、引用等） | `bernoulli` |
| 非負整数（取引量、コメント数等） | `poisson` |
| 連続実数（相関行列等） | `gaussian` |

### 実データ特有の問題

- **真の Z が存在しない**：RMSE(Z) が使えない → BIC / held-out link prediction / Y 再構成精度 / X 再構成精度 / ラベル一致率で評価
- **Y が非常にスパース**（引用ネットワーク等では密度 < 0.5%）：Y=Bernoulli 時の初期化で density 補正が重要
- **X が高次元**（d ≫ 15）：d-sweep で確認済みだが、d が大きい場合は収束が遅くなる可能性
- **n が大きい**（> 500）：E-step の O(n²) コストが支配的になる

---

## 5. repo内の実データ候補調査

### 調査結果

| ファイルパス | 形式 | 概要 | すぐ使えるか |
|-------------|------|------|------------|
| `expfam/data/ml-100k.zip` | ZIP | MovieLens 100K（ユーザー×映画評価） | 要展開・前処理 |
| — | scikit-learn | Wine データ（`load_wine()`）| すぐ使える（インポートのみ） |

### Wine データ（scikit-learn）

- **n = 178**, **d = 13**（連続属性 13 種）
- **クラス**: 3 種（class_0: 59, class_1: 71, class_2: 48）
- **既存スクリプト**: `expfam/src/run_wine_dual.py`（X=Gaussian, Y=Bernoulli, k=6 固定、5試行）
- **既存結果**: `expfam/results/wine_dual_results.csv`（rmse_X≈0.574, rmse_Y≈0.490, BIC≈8000）
- **状態**: KI-006 「未評価」— 分布族比較・BIC k 選択・X_only/Y_only 比較が未実施
- **Y の構築**: `Y_{ij} = 1 if label_i == label_j else 0`（同一クラスペアが正例）
- **注意**: Y はクラス情報から構築した人工的な隣接行列であり、自然なネットワーク構造ではない

### MovieLens ml-100k

- **ファイル**: `expfam/data/ml-100k.zip`（展開済みデータは別途必要）
- **archive 結果**: `expfam/results/archive/real_movielens_results.csv`（Poisson AUC=0.947、Bernoulli AUC=0.486、archive 状態・未検証）
- **構造**: n=943 ユーザー, d=最大 18 ユーザー属性, Y=ユーザー×ユーザー共通評価グラフ（要設計）
- **難点**: ユーザー数が多く計算コスト高、Y の定義が自明でない（ユーザー間は直接接続なし）

### repo内の実データ候補まとめ

Wine のみが比較的すぐ使える状態（前処理・スクリプト既存）。MovieLens はアーカイブ状態。その他のデータディレクトリ（`data/`, `datasets/`, `real_data/`）は存在しない。

---

## 6. 外部実データ候補一覧

### 属性付きネットワークデータ（推奨候補）

| # | データセット | ノード | X にするもの | Y にするもの | family_x | family_y | 難易度 | 面白さ | 注意点 |
|---|-------------|--------|------------|------------|----------|----------|--------|--------|--------|
| 1 | **Cora** | 2708 論文 | BoW (0/1 単語特徴量, d=1433) | 引用関係 | bernoulli | bernoulli | 低 | ★★★ | 最標準的ベンチマーク。d が大きいので前処理で削減推奨 |
| 2 | **Citeseer** | 3327 論文 | BoW (0/1, d=3703) | 引用関係 | bernoulli | bernoulli | 低 | ★★★ | Cora と同構造。d がさらに大きい |
| 3 | **Karate Club** (Zachary) | 34 人 | 人手設計属性または one-hot 特徴量 | 友人関係 | gaussian | bernoulli | 低 | ★ | 非常に小規模でデバッグ向き。X が自然に存在しない |
| 4 | **Polblogs** | 1490 ブログ | なし（X なし）→ 構成が難しい | ハイパーリンク | — | bernoulli | 中 | ★★ | X が自然に存在しないため工夫が必要 |
| 5 | **BlogCatalog** | 10312 ユーザー | ユーザーカテゴリ (Bernoulli) | フォロー関係 | bernoulli | bernoulli | 中 | ★★★ | 中規模。ノード分類タスクとの比較が可能 |
| 6 | **Amazon（Co-purchase）** | 数千〜数万商品 | 商品レビューBoW (Poisson) | 共購入リンク | poisson | bernoulli | 高 | ★★★ | X=Poisson の有効性を示せる |
| 7 | **国間貿易ネットワーク** (UN Comtrade) | 約 200 カ国 | GDP, 人口等 (Gaussian) | 貿易額 | gaussian | poisson or gaussian | 高 | ★★★ | Y=Poisson が自然。Gaussian Y の実例として有力 |
| 8 | **Protein-Protein Interaction** | 数百〜数千タンパク質 | Gene Ontology アノテーション (Bernoulli) | 相互作用有無 | bernoulli | bernoulli | 中 | ★★ | 生物学的解釈が可能 |
| 9 | **Twitch Social Networks** | 数千〜数万ユーザー | ゲームカテゴリ (Bernoulli) | 友人関係 | bernoulli | bernoulli | 中 | ★★ | SNAP から取得可能 |
| 10 | **Wikipedia Networks** | 数千記事 | TF-IDF or BoW (Poisson/Bernoulli) | 記事間リンク | poisson | bernoulli | 中 | ★★ | テキスト特徴量が豊富 |

### 評価指標の面から見た推奨理由

- **Cora / Citeseer**: ノード分類ラベルが利用可能 → Zの2D可視化でクラスタリング評価ができる。X=Bernoulli (BoW) の自然なケース。
- **国間貿易**: Y=Poisson (取引量) と Y=Gaussian の両方を試せる。X=Gaussian (経済指標) との組み合わせがシナリオ B に近い。
- **Amazon**: X=Poisson (単語カウント) が自然。シナリオ A (X=Poisson) の実データ版として位置づけられる。

---

## 7. 最初のpilot候補3選

### Pilot 1: Wine（sklearn）— 即実行可能

**なぜ最初に向いているか**
- n=178 で計算コストが低い（1 fit < 1 分）
- 既存スクリプト (`run_wine_dual.py`) が存在するため参考にできる
- クラスラベルが 3 クラスで Z の可視化・評価が容易
- fixed 版への切り替えが最も単純（スクリプト改修のみ）

**X, Y の作り方**
- `X` = `StandardScaler().fit_transform(load_wine().data)` → (178, 13)
- `Y` = `(label_i == label_j).astype(float)` → 対角 0, 同クラス 1（既存実装と同一）

**family_x / family_y の候補**
- メイン: `family_x=gaussian, family_y=bernoulli`（先行研究条件）
- 比較1: `family_x=gaussian, family_y=gaussian`（Y が連続値とみなす場合）
- 比較2: `family_x=bernoulli, family_y=bernoulli`（X を 2 値化した場合）

**k 選択方法**: BIC（k=1〜9）

**評価指標**
- BIC（k 選択）
- rmse_X, rmse_Y（fit の質）
- Z の 2D 可視化（クラス 3 種が分離するか）
- X_only / Y_only / X+Y 比較（ablation）

**可視化方法**
- Z を k=2 まで次元削減（PCA or 直接 2 次元）してクラスラベルで色分け

**実装難易度**: 低（既存スクリプトの改修）

**失敗リスク**
- Y が「同クラス＝リンク」という人工的定義のため、ネットワーク推定としての意味が薄い
- k=6 固定の既存結果は BIC 未実施のため、最適 k が分からない

---

### Pilot 2: Cora — 標準ベンチマーク

**なぜ最初に向いているか**
- グラフ深層学習コミュニティでの標準ベンチマーク（比較が豊富）
- X=BoW (0/1 単語特徴量, Bernoulli)、Y=引用有無 (Bernoulli) が自然
- 7 クラスのラベルが利用可能 → ノード分類的評価が可能
- `torch_geometric` や `dgl` から 1 行で取得可能

**X, Y の作り方**
- `X` = BoW 特徴量 (0/1)、次元削減推奨（例: 上位 50〜100 次元に絞る）
- `Y` = 論文間引用関係（上三角・無向化）、n=2708

**family_x / family_y の候補**
- メイン: `family_x=bernoulli, family_y=bernoulli`（BoW 0/1 + 引用有無）
- 比較1: `family_x=poisson, family_y=bernoulli`（BoW を TF カウントとして扱う場合）
- 比較2: `family_x=gaussian, family_y=bernoulli`（X を標準化した場合）

**k 選択方法**: BIC（k=1〜9）

**評価指標**
- BIC
- Held-out link prediction AUC / AP
- Z の 2D 可視化（7 クラスが分離するか）
- family_x 比較（Bernoulli vs Poisson vs Gaussian）

**可視化方法**
- 7 クラスラベルで Z を色分けした散布図

**実装難易度**: 中（データ読み込みに `torch_geometric` または手動 numpy 変換が必要）

**失敗リスク**
- n=2708 は大きく、E-step が O(n²) で重い（1 fit に 30 分以上かかる可能性）
- Y のスパース性が極端に高い（0.02% 程度）→ Bernoulli 初期化の density が非常に小さい
- d=1433 は大きすぎる → 前処理で削減必須

---

### Pilot 3: 国間貿易ネットワーク（UN Comtrade）

**なぜ最初に向いているか**
- Y=貿易額（Poisson または Gaussian）が自然 → シナリオ B (X=Gaussian, Y=Poisson) の実データ版
- X=GDP、人口、インフレ率等の連続変数 → Gaussian が自然
- n≈200 カ国と小〜中規模で計算コストが低い
- 経済的解釈が明快（潜在変数 Z が経済圏・地域クラスタに対応する可能性）

**X, Y の作り方**
- `X` = World Bank から GDP, 人口等 (n≈200, d≈10〜20) を取得・標準化
- `Y` = 二国間貿易額行列（対称化、ゼロが多い → Poisson 化、または閾値で Bernoulli 化）

**family_x / family_y の候補**
- メイン: `family_x=gaussian, family_y=poisson`（GDP + 貿易量）
- 比較1: `family_x=gaussian, family_y=gaussian`（貿易量を対数変換後 Gaussian）
- 比較2: `family_x=gaussian, family_y=bernoulli`（貿易有無の 0/1 化）

**k 選択方法**: BIC（k=1〜9）

**評価指標**
- BIC
- Y 再構成精度（rmse_Y）
- Z の 2D 可視化（地理的・経済的な地域クラスタが現れるか）
- 地理的近隣との一致率

**可視化方法**
- 世界地図上にノードを配置し、Z の第 1 次元で色分け

**実装難易度**: 高（データ取得・前処理が必要。外部 API または CSV のダウンロード）

**失敗リスク**
- データ取得が煩雑（UN Comtrade のライセンス・API 制限）
- Y に 0 が多く Poisson Y の場合は w の初期化が重要
- 二国間貿易が非対称 → 対称化の方法によって結果が変わる

---

## 8. 実データでの評価指標

| 指標 | 計算方法 | メリット | デメリット |
|------|----------|---------|-----------|
| **BIC** | `-2 * Q_strict + num_params * ln(n)` | モデル選択（k, family）に使える。値が小さいほど良い | 実装の `calc_bic_dual` を要確認（KI-010）。負値も正常（Gaussian Y）|
| **Held-out link prediction** | Y の 10〜20% をマスクして予測、AUC / AP で評価 | 予測タスクとして客観的 | 実装が必要（マスク・予測の分離）|
| **Y 再構成精度 (rmse_Y)** | `||μ_Y(est) - Y_obs||_F` | 観測値との適合度を直接測定 | Y がスパースだと見かけ上は常に低 RMSE になる |
| **X 再構成精度 (rmse_X)** | `||μ_X(est) - X_obs||_F` | X 側の fit の質 | 人工データ実験と同じ指標で比較可能 |
| **既知カテゴリとの一致 (NMI/ARI)** | Z を k-means クラスタリングして既知ラベルと比較 | Z の意味的解釈が可能 | ラベルがあるデータのみ |
| **Z の 2D 可視化** | PCA or tsne で Z を 2D に圧縮してクラスターを目視確認 | 直感的・論文掲載可能 | 定量指標ではない |
| **X+Y vs X_only vs Y_only 比較** | fix_w=True (X_only) / fix_x=True (Y_only) で再実行 | どの情報が有効かを分離できる | 実験数が 3 倍になる |
| **分布族比較（family_x / family_y）** | 同一データで複数の family を試して BIC 比較 | 最適な分布族を発見できる | 実験数が 3〜9 倍になる |

---

## 9. X+Y / X_only / Y_only 比較設計

人工データ Exp4 の ablation（`fix_w=True` で Y 固定、`fix_x=True` で X 固定）を実データに適用する。

```
条件1: X+Y  (通常モード, fix_w=False, fix_x=False)
条件2: X_only (Y 側のパラメータ w, w0 を固定 → fix_w=True)
条件3: Y_only (X 側の F を固定 → fix_x=True)
```

評価指標: BIC, rmse_X, rmse_Y, held-out link prediction AUC

解釈の例:
- `Y_only ≈ X+Y` → X がほとんど Z 推定に寄与していない（Scen C と同様）
- `X_only ≈ X+Y` → Y がほとんど寄与していない（属性データが情報量豊富）
- `X+Y` が両者より明確に良い → X と Y が補完的

---

## 10. 分布族比較設計

同一データに対して複数の (family_x, family_y) 組み合わせを試し、BIC と予測精度で比較する。

**最小セット（3 条件）**
```
family_x=gaussian, family_y=bernoulli  ← 先行研究条件
family_x=bernoulli, family_y=bernoulli ← X が 0/1 の場合
family_x=poisson,   family_y=bernoulli ← X がカウントの場合
```

**拡張セット（データ依存）**
- Y が Poisson 向きのデータ（貿易量等）: `family_y=poisson` を追加
- Y が Gaussian 向きのデータ（相関行列等）: `family_y=gaussian` を追加

BIC が最小の (family_x, family_y) を「データに最適な分布族」として報告する。

---

## 11. held-out link prediction設計

Y の上三角成分から 10〜20% をランダムにマスクし、残り 80〜90% で学習、マスク部分を予測する。

```python
# マスク生成
upper_idx = np.triu_indices(n, k=1)
n_edges = len(upper_idx[0])
mask_rate = 0.2
mask_idx = np.random.choice(n_edges, int(n_edges * mask_rate), replace=False)
Y_train = Y.copy()
Y_train[upper_idx[0][mask_idx], upper_idx[1][mask_idx]] = 0  # 正例をマスク

# 予測: η_Y の sigmoid (Bernoulli) または exp (Poisson) を使う
# AUC / Average Precision を計算
```

**注意**:
- 現在の `run_em_fixed_*` スクリプトには held-out 分割がないため、新規実装が必要
- 負例サンプリング（スパースな Y の場合は正例に対して均等サンプリング）も検討

---

## 12. Z可視化・解釈設計

**Z の次元削減**
- k≤3 の場合: k=2 の成分を直接プロット
- k>3 の場合: PCA (Z への適用) で 2D に圧縮

**プロット構成**
- 色: 既知ラベル / クラスタリング結果
- マーカー: 特徴量グループ（Wine のクラス、Cora のカテゴリ等）
- 注釈: 代表ノードの名前（国名、論文タイトル等）

**定量評価（ラベルがある場合）**
- k-means クラスタリング → Normalized Mutual Information (NMI) と Adjusted Rand Index (ARI) を計算
- family_x / family_y 比較で NMI が変わるかを確認

---

## 13. 複数実データセット比較の設計

**比較の軸**

| 比較軸 | 観察したいこと |
|--------|--------------|
| データセット間 | どのデータで提案手法（X+Y）が X_only / Y_only に比べて有効か |
| family_x 間 | Gaussian / Bernoulli / Poisson のどれが BIC で選ばれるか |
| family_y 間 | 同上 |
| k 間 | BIC の k 選択は各データで安定しているか |

**推奨する比較データ数**: 2〜4 個（少なすぎると一般化できず、多すぎると計算コストが高い）

**推奨組み合わせ（最初のステップ）**

| データ | X の族 | Y の族 | 位置づけ |
|--------|--------|--------|---------|
| Wine (sklearn) | Gaussian | Bernoulli | 先行研究条件の実データ確認 |
| Cora | Bernoulli | Bernoulli | X=Bernoulli (BoW) の有効性確認 |
| 貿易ネットワーク | Gaussian | Poisson | Y=Poisson の実データ確認 |

---

## 14. 従来論文対応実験との関係

### 従来論文の条件

先行研究（Mikawa et al., NOLTA 2024）は `X=Gaussian, Y=Bernoulli` 固定。

### 既存の比較コード

repo 内に以下が存在する：
- `reproduction/src/experiment_compare_with_dual.py`
- `run_comparison_all.py`
- `reproduction/results/comparison/comparison_main_table.csv`（Control 比較、RMSE 差 < 0.001, 5 試行）

### 位置づけ

**実データ実験における従来論文条件の役割**：

1. **ベースライン**: `family_x=gaussian, family_y=bernoulli` の結果が従来論文と等価の条件。実データ分布族比較で必ず含める。
2. **Wine 実験**: 既存 `run_wine_dual.py` は従来論文条件（Gaussian-X, Bernoulli-Y）で実行済み。これを fixed 版に切り替え、BIC k 選択と family 比較を加えれば「従来論文条件 + 提案手法の分布族最適化」の比較になる。
3. **追加確認の優先度**: 実データ実験を先行させ、従来論文条件は「分布族比較の 1 条件」として自然に含める形が効率的。

### 実データ前に実施すべき追加人工データ実験（任意）

- 従来論文の元データ（WineまたはMLで先行研究が使ったデータ）で固定版を動かす実験は、必要に応じて別途計画書を作成する。
- 現時点では「実データ pilot → 分布族比較」の流れを優先する。

---

## 15. 推奨する次の実行順

### Phase 1: Wine pilot（すぐ実行可能）

1. `expfam/src/run_wine_dual.py` を固定版（`DualExpFamLSMFixed`）に切り替えた新スクリプトを作成
2. `family_x=gaussian, family_y=bernoulli` で BIC k=1〜9 を実行（10 試行）
3. X_only / Y_only / X+Y の ablation を実行
4. Z を 2D 可視化（3 クラスの分離を確認）
5. 結果を報告（BIC 最適 k, rmse_X, rmse_Y, クラスタ分離）

### Phase 2: Wine 分布族比較（Phase 1 成功後）

1. `family_x` を bernoulli（X を 2 値化）に変えて再実行
2. BIC で family_x=gaussian vs bernoulli を比較
3. Gaussian-X が BIC で選ばれるか確認（Wine は連続属性なので gaussian が有利なはず）

### Phase 3: Cora pilot（Phase 2 成功後）

1. Cora データを取得・前処理（d を 50〜100 に削減推奨）
2. `family_x=bernoulli, family_y=bernoulli` で BIC k=1〜9 を実行
3. Held-out link prediction（AUC / AP）
4. Z の 7 クラス分離を可視化

### Phase 4: 複数データセット比較（Phase 3 以降）

1. Wine / Cora / 貿易ネットワーク（または MovieLens）で統一評価
2. 各データの BIC 最適 (family_x, family_y) を比較
3. X+Y vs X_only vs Y_only の傾向をデータセット間で比較

---

## 16. 注意点・リスク

### 計算コスト

| データ | n | 推定 fit 時間 |
|--------|---|-------------|
| Wine | 178 | < 1 分 |
| MovieLens (ユーザー数削減版) | ~200 | 1〜3 分 |
| Cora (d 削減) | 2708 | 30〜60 分 |
| 貿易ネットワーク | ~200 | 1〜3 分 |

n が大きいと E-step の O(n²) コストが問題になる。Cora は 10 試行 × k=1〜9 = 90 fits で推定 45〜90 時間になる可能性があるため、まず k を固定（BIC なし）で 5 試行程度から始める。

### Y の定義問題

Wine の Y（同クラス隣接行列）は人工的。自然なネットワーク構造を持つデータ（Cora の引用、貿易ネットワーク）の方が研究として意味がある。

### family の前処理整合性

- Bernoulli-X を使う場合は X を 0/1 に変換する前処理が必要（閾値の選択が結果に影響）
- Poisson-X を使う場合は X が整数（カウント）でなければならない
- Gaussian-X を使う場合は標準化が推奨（`StandardScaler`）

### KI-006（Wine 実験未評価）の解消

Wine pilot（Phase 1）を実行することで KI-006 が解消される。

### 既存スクリプトの注意

既存の `run_wine_dual.py` は旧版 `DualExpFamLSM`（0.5 あり）を使用。新スクリプトでは `DualExpFamLSMFixed` に切り替える。旧版結果（`wine_dual_results.csv`）は参考値として保持し、上書きしない。

---

## 17. ChatGPTにレビューしてほしい点

1. **Wine の Y の構築方法**：同クラス = 1 の隣接行列は研究として適切か。より自然な Y の定義（距離ベース、共起ベース等）はあるか。

2. **Cora の前処理設計**：d=1433 の BoW 特徴量を削減する最良の方法は何か（分散上位、PCA、random projection 等）。削減後に `family_x=bernoulli` を維持できるか。

3. **Held-out link prediction の実装詳細**：スパースネットワーク（Y の密度 < 1%）で正例・負例のサンプリングをどう設計するか（全負例 vs サンプリング、比率は何倍が適切か）。

4. **k 選択の実データへの適用**：人工データでは k*=3 を BIC が正確に選択できたが、実データではどの程度 k が変わるか。大きすぎる k を選ぶリスクはどの程度か。

5. **評価指標の優先順位**：修論の主張（分布族一般化の有効性）を最も説得力高く示すための評価指標は何か（BIC・AUC・NMI のどれを主指標にすべきか）。

6. **計算コスト削減**：Cora（n=2708）で現実的な実験時間内に収める方法はあるか（EM 反復数削減・L 削減・サブサンプリング等の影響）。

---

*作成: Claude Sonnet 4.6（2026-06-17）*  
*参照: `RESEARCH_MASTER.md`, `EXPERIMENT_REGISTRY.md`, `KNOWN_ISSUES.md`, `expfam/README.md`*  
*ステータス: 調査・計画書のみ。実験実行・データ取得・git 操作は未実施。*
