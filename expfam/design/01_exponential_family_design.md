# Exponential Family Latent Structural Model: 設計書

**タイトル案**: *"Generalized Latent Structural Models for Relational Data via Exponential Family Distributions"*

**ターゲット**: KDD 2026 / NeurIPS 2025 (Research Track)

---

## 1. 数理的定式化

### 1.1 ベースモデルの構造（Mikawa et al., 2024 の振り返り）

現行の生成モデルは以下の3層構造を持つ：

```
z_i  ~ N(0, σ²_z I_k)                           [潜在変数の事前分布]
x_i  ~ N(F z_i, Σ)                               [属性データの生成]
y_ij ~ Bernoulli(σ(w₀ + w · z_i^T z_j))          [関係データの生成]
```

**制約**: Y が 0/1 の二値に限定される。カウント・連続・順序データへの適用不可。

---

### 1.2 指数型分布族による一般化

#### 1.2.1 指数型分布族の一般形

指数型分布族の確率密度（質量）関数は次の標準形で表される：

```
p(y; η) = h(y) · exp( η·T(y) − A(η) )
```

| 記号 | 意味 |
|------|------|
| `η`   | 自然パラメータ (natural parameter) |
| `T(y)` | 十分統計量 (sufficient statistic) |
| `A(η)` | 対数分配関数 (log-partition function) |
| `h(y)` | 基底測度 (base measure) |

**重要な性質**:
- `E[T(Y)] = A'(η)`  （平均関数 = 対数分配関数の一階微分）
- `Var[T(Y)] = A''(η)` （分散関数 = 対数分配関数の二階微分、常に ≥ 0）

#### 1.2.2 拡張された生成モデル

Y の生成過程を指数型分布族で置き換える：

```
z_i  ~ N(0, σ²_z I_k)                           [変更なし]
x_i  ~ N(F z_i, Σ)                               [変更なし]
y_ij ~ ExpFam( η_ij )                            [一般化: Bernoulli を包含]
```

ここで**リンク関数**（自然パラメータと潜在変数内積の接続）は：

```
η_ij = g(w₀ + w · z_i^T z_j)
```

`g` はリンク関数（恒等写像 or スケーリング変換）。カノニカルリンクを選択すれば `g(u) = u`。

各分布における対応は以下の通り：

| 分布 | y の値域 | T(y) | A(η) | A'(η) = E[Y\|η] | A''(η) = Var[Y\|η] | リンク |
|------|---------|------|------|-----------------|-------------------|--------|
| **Bernoulli** | {0, 1} | y | log(1+eη) | σ(η) | σ(η)(1−σ(η)) | logit（カノニカル） |
| **Poisson** | {0,1,2,...} | y | eη | eη = λ | eη = λ | log（カノニカル） |
| **Gamma** | (0, ∞) | y | −log(−η) | −1/η | 1/η² | 逆数（カノニカル）or log |
| **Neg. Binomial** | {0,1,...} | y | r·log(1−eη) | r·eη/(1−eη) | r·eη/(1−eη)² | log |

> **ポイント**: Bernoulli は η_ij = w₀ + w·z_i^T z_j で A''(η) = s_ij(1−s_ij) となり、**既存の実装がこの一般形の特殊ケースであることが確認できる**。

---

### 1.3 Q 関数の一般形

EM アルゴリズムの Q 関数は：

```
Q(θ; θ^old) = E_{Z | X, Y, θ^old} [ ln p(X, Y, Z; θ) ]
            = E[ ln p(Z) ] + E[ ln p(X|Z) ] + E[ ln p(Y|Z) ]
```

第一項・第二項は既存と同一。**第三項**のみが変更される：

```
E[ ln p(Y|Z) ] ≈ (1/L) Σ_l Σ_{i<j} [ η_ij^(l) · T(y_ij) − A(η_ij^(l)) + ln h(y_ij) ]
```

ここで `η_ij^(l) = w₀ + w · z_i^(l)^T z_j^(l)`。

> `ln h(y_ij)` は θ に依存しないため M-step では定数として無視できる。

**具体例 — Poisson Q 関数**:

```
E[ ln p(Y|Z) ]_Poisson
  = (1/L) Σ_l Σ_{i<j} [ (w₀ + w·z_i^T z_j) · y_ij − exp(w₀ + w·z_i^T z_j) − ln(y_ij!) ]
```

---

### 1.4 E-step: 一般化されたニュートン法更新式

事後分布 `p(Z|X, Y; θ)` の対数を `z_i` について微分する。

#### 1.4.1 勾配（一階微分）

```
∂ ln f(Z|X, Y; θ) / ∂z_i
  = −(1/σ²_z) z_i
  + F^T Σ^{-1} (x_i − F z_i)
  + (1/2) Σ_{j≠i} [T(y_ij) − A'(η_ij)] · w · z_j
```

**一般化の核心**: `T(y_ij) − A'(η_ij)` = 十分統計量の実現値 − 期待値 = **残差**。

| 分布 | 残差項 |
|------|--------|
| Bernoulli | `y_ij − σ(η_ij)` = `y_ij − s_ij` ← 既存実装と完全一致 |
| Poisson | `y_ij − exp(η_ij)` = `y_ij − λ_ij` |
| Gamma | `y_ij − (−1/η_ij)` = `y_ij − μ_ij` |

#### 1.4.2 精度行列（負のヘッシアン）

```
A_i = (1/σ²_z) I
    + F^T Σ^{-1} F
    + (1/2) Σ_{j≠i} A''(η_ij) · w² · z_j z_j^T
```

**一般化の核心**: `s_ij(1−s_ij)` が `A''(η_ij)` に置き換わるだけ！

| 分布 | `A''(η_ij)` | 正定値保証 |
|------|-------------|-----------|
| Bernoulli | `s_ij(1−s_ij) ∈ (0, 0.25]` | ✅ 常に正 |
| Poisson | `exp(η_ij) = λ_ij > 0` | ✅ 常に正 |
| Gamma | `1/η_ij² > 0` | ✅ 常に正（カノニカル） |

> **重要**: 指数型分布族の分散関数 `A''(η) ≥ 0` は一般に保証されるため、精度行列の正定値性が**分布によらず自動的に保証**される。

---

### 1.5 M-step: w₀, w の更新式

#### 1.5.1 勾配

```
∂Q / ∂w₀ = (1/2L) Σ_l Σ_{i≠j} [T(y_ij) − A'(η_ij^(l))]

∂Q / ∂w  = (1/2L) Σ_l Σ_{i≠j} [T(y_ij) − A'(η_ij^(l))] · z_i^(l)^T z_j^(l)
```

**既存の Adam 最適化器がそのまま再利用可能**。差分は `y_ij − s_ij` を `T(y_ij) − A'(η_ij)` に変えるだけ。

#### 1.5.2 スケーリング変換（Poisson の場合の注意）

Poisson では `λ_ij = exp(η_ij)` が爆発する可能性がある。`z_i^T z_j` の絶対値が大きいとき：

```
η_ij = w₀ + w · z_i^T z_j → +∞ → λ_ij → ∞
```

対策として `w` の初期値を小さく設定 (`w_init ≈ 0.1`)、または `w₀` に上限クリッピングを適用する（後述）。

---

## 2. ラプラス近似の課題解決

### 2.1 精度行列の非特異性の保証

#### 2.1.1 問題の発生源

精度行列 `A_i` の第三項 `Σ_{j≠i} A''(η_ij) w² z_j z_j^T` はランク `min(n-1, k)` の行列の和。`n < k + 1` のとき、あるいは `w ≈ 0` のとき、この項がほぼゼロになり精度行列が ill-conditioned になる。

#### 2.1.2 数学的対策

**対策 1: 対角正則化（既存実装に存在）**

```python
A_i += ε · I_k    # ε = 1e-6
```

これにより最小固有値は `1/σ²_z + ε > 0` に保たれる（第一項が常に正定値なため実質的に問題は少ない）。

**対策 2: Poisson 固有の発散防止**

`A''(η_ij) = exp(η_ij)` が大きすぎる場合の抑制：

```python
# η のクリッピング（発散防止）
eta_ij = np.clip(eta_ij, -20, 20)   # λ in [2e-9, 4.85e8]

# あるいは variance function のキャップ
var_fn = np.minimum(A_double_prime(eta_ij), VAR_CAP)  # VAR_CAP = 1e4
```

**対策 3: Cholesky 分解による安定サンプリング**

精度行列のサンプリングを `linalg.inv` から `Cholesky + 逆行列` に変更：

```python
# Cholesky 分解: A_i = L L^T
L = np.linalg.cholesky(A_i)
# サンプリング: z = η_i + L^{-T} ε, ε ~ N(0, I)
eps = rng.standard_normal(k)
z_sample = eta_i + np.linalg.solve(L.T, eps)
```

これにより `A_i` が正定値であることを明示的に検証でき、LinAlgError の早期検出が可能。

**対策 4: Gamma 分布のパラメータ空間制約**

カノニカルパラメータ `η < 0` の制約を満たすため、リンク関数として**ログリンク**を使用する：

```
μ_ij = exp(w₀ + w · z_i^T z_j)   [log-link, 非カノニカルだが数値安定]
```

これにより `η_ij = −α/μ_ij = −α · exp(−(w₀ + w·z_i^Tz_j))` は常に負となり制約を自動的に満たす。

#### 2.1.3 実装上のロバスト化

```python
def _variance_function(self, eta: np.ndarray, dist: str) -> np.ndarray:
    """指数型分布族の分散関数 A''(η) を安定計算。"""
    if dist == 'bernoulli':
        s = sigmoid(eta)
        return np.clip(s * (1 - s), 1e-8, None)
    elif dist == 'poisson':
        return np.clip(np.exp(np.clip(eta, -30, 20)), 1e-8, 1e6)
    elif dist == 'gamma':
        # log-link 使用時: μ = exp(η), Var ∝ μ² → A''_effective = μ²/α²
        mu = np.exp(np.clip(eta, -30, 20))
        return np.clip(mu ** 2, 1e-8, 1e6)
    elif dist == 'negbinom':
        mu = np.exp(np.clip(eta, -30, 20))
        r = self.dispersion
        return np.clip(mu + mu**2 / r, 1e-8, 1e6)
```

---

## 3. 応用シナリオと分布の提案

### シナリオ A: Poisson — 購買ネットワークのモデリング

**問題設定**:
ECサイトの商品 i と顧客 j の間の「購買回数」`y_ij ∈ {0, 1, 2, 3, ...}` と商品の特徴量ベクトル `x_i ∈ ℝ^d` を同時にモデリングし、商品間の潜在的な協調構造（"一緒に買われやすい理由"）を抽出する。

**モデル**: `y_ij ~ Poisson(exp(w₀ + w · z_i^T z_j))`, `x_i ~ N(F z_i, Σ)`

**学術的新規性**:
- 協調フィルタリング（行列分解）は属性情報を無視する
- コンテンツベース推薦は関係構造を無視する
- **本モデルは両者を統一する初めての確率的生成モデル**
- Poisson ログ線形構造により、カウントの非対称性（`y_ij ≠ y_ji`の一般化）にも対応可能

**データセット候補**: Amazon Review Dataset (McAuley et al., 2023)、UserBehavior（Alibaba）

---

### シナリオ B: Gamma — 金融リスクネットワークの相関強度モデリング

**問題設定**:
金融機関間の「取引量」または「相関係数の絶対値」`y_ij > 0` と各機関の財務指標 `x_i ∈ ℝ^d`（資産規模、レバレッジ比率等）を用いて、システミックリスクの伝播構造を潜在変数で表現する。

**モデル**: `y_ij ~ Gamma(shape=α, rate=α/μ_ij)`, `μ_ij = exp(w₀ + w · z_i^T z_j)`, `x_i ~ N(F z_i, Σ)`

**学術的新規性**:
- 金融ネットワーク分析の多くは二値グラフ（エッジあり/なし）に限定
- 連続的なリスク重みを持つ加重ネットワークの生成モデルが存在しない
- `z_i^T z_j` が**リスク伝播チャネルの強さ**を自然に表現
- `F` の列が**リスクファクター（市場リスク、流動性リスク等）**として解釈可能

**データセット候補**: BIS Bilateral Derivatives Dataset、Bloomberg Financial Network

---

### シナリオ C ⭐: 負の二項分布 — スパース・カウントデータに基づく研究者協調ネットワーク

**問題設定**:
研究者ペア `(i, j)` の「共著論文数」`y_ij ∈ {0, 1, 2, ...}` と研究者の論文 TF-IDF 特徴 `x_i ∈ ℝ^d` を用いて、研究分野の潜在的クラスター構造を発見する。

ネットワークは**極めて疎（大多数のペアが y_ij = 0）かつ過分散（少数ペアが y_ij >> 1）**であり、Poisson では適合しない。

**モデル**:
```
y_ij ~ NegBin(r, μ_ij / (μ_ij + r))     [過分散パラメータ r を追加推定]
μ_ij = exp(w₀ + w · z_i^T z_j)
x_i  ~ N(F z_i, Σ)
```

**学術的新規性**:
- Negative Binomial の `r → ∞` 極限が Poisson に一致し、Bernoulli の二値化も包含する**統一モデル**として位置付けられる
- `r` を学習可能パラメータにすることで、**過分散の程度を自動推定**
- トピックモデル（LDA）との比較軸を作れる（研究者を「研究トピックの混合」として表現）

---

### 3.1 トップカンファレンスに最も適したアプローチの選定

> **選定: シナリオ A — Poisson モデル（購買ネットワーク）を初期 PoC とし、シナリオ C（負の二項分布）で論文を完成させる**

#### 理由

**1. 問題の普遍性（Broader Impact が高い）**

購買データ・ソーシャルメディアのいいね数・遺伝子共発現カウントなど、「カウント関係データ＋属性情報」の組み合わせは産業界・生命科学・社会科学のどこにでも存在する。KDD の審査委員は「実世界のインパクト」を重視するため、Poisson ベースの応用は採択率に直結する。

**2. 数学的クリーンさ（理論的貢献として KDD/NeurIPS に適合）**

Poisson のカノニカルリンク（log-link）は：
- η_ij の値域に制約なし → 数値安定性が高い
- A''(η) = exp(η) > 0 → 精度行列の正定値性が自動保証
- 十分統計量 T(y) = y → M-step の勾配が最もシンプル
- Bernoulli へのソフトな変換（threshold により二値化）が可能

**3. ベースライン比較の豊富さ（査読に有利）**

Poisson 分布を用いたネットワーク埋め込み（PPR、LINE、PoissonMF 等）は既存研究が多く、**明確な比較対象**が確立されている。これは査読者にとって「何が改善されたか」が一目瞭然であり、リジェクトリスクを下げる。

**4. 拡張の自然な流れ（ストーリー構造）**

```
Section 1: Bernoulli (binary) モデルの復元 [再現性の確認]
Section 2: Poisson モデルへの拡張 [本論文の主貢献 1]
Section 3: 負の二項分布による過分散対処 [本論文の主貢献 2]
Section 4: 実世界データでの優位性実証 [KDD に必須]
```

この4段構造は KDD の審査基準（新規性・有用性・実証性）を全て満たす。

---

## 4. 実装ロードマップ（Step 2 以降の計画）

### Phase 1: ExpFam 共通基盤クラスの設計

```
src/
├── model.py                      # 既存（変更最小限）
├── model_expfam.py               # 新規: ExpFamLatentStructuralModel
│   ├── _sufficient_stat(y)       # T(y): 分布依存
│   ├── _log_partition(eta)       # A(η): 分布依存
│   ├── _mean_function(eta)       # A'(η) = E[Y|η]
│   └── _variance_function(eta)   # A''(η) = Var[Y|η]
└── distributions/
    ├── bernoulli.py              # 既存モデルの再実装（テスト用）
    ├── poisson.py                # 新規
    └── negative_binomial.py      # 新規
```

### Phase 2: PoC 実験（Poisson 合成データ）

1. `n=150, d=15, k=3` で Poisson データを生成
2. Bernoulli モデル vs Poisson モデルの RMSE(X), RMSE(Z) 比較
3. カウントデータに対する対数尤度比較（モデル適合度）

### Phase 3: 実データ実験

1. **Amazon Co-Purchase Network** — 商品の共購買カウント + 商品特徴量
2. **DBLP Co-authorship Network** — 共著数 + 著者の研究キーワード
3. **Gene Co-expression Network** — RNA-seq カウント + 遺伝子の機能アノテーション

---

## 5. 数式サマリー（実装リファレンス）

### 共通インターフェース（分布を切り替えるだけで全アルゴリズムが動作）

| 計算ステップ | Bernoulli | Poisson | Gamma (log-link) |
|------------|-----------|---------|-----------------|
| 自然パラメータ | η = w₀ + w·z_i^T z_j | 同左 | 同左 |
| 平均 A'(η) | σ(η) | exp(η) | exp(η) |
| 分散 A''(η) | σ(η)(1-σ(η)) | exp(η) | exp(2η)/α |
| 勾配残差 | y - σ(η) | y - exp(η) | y - exp(η) |
| 精度行列 Hessian 寄与 | A''(η)·w²·z_j z_j^T | 同左 | 同左 |
| M-step 勾配 ∂Q/∂w₀ | Σ(y-σ(η)) | Σ(y-exp(η)) | Σ(y-exp(η)) |

> **結論**: コードの変更量は最小。`_variance_function` と `_mean_function` を抽象化するだけで、既存の E-step・M-step のフレームワークが**そのまま**すべての指数型分布族に対して動作する。これが本拡張の最大の優位性である。

---

*文書バージョン: 1.0 | 作成日: 2026-03-17*
