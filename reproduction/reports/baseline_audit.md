# Baseline Audit: Mikawa 2024 vs Dual-ExpFam LSM
# 作成日: 2026-04-24

---

## 1. Baseline（Mikawa 2024）の仕様

### モデル前提
| 項目 | 仕様 | コード箇所 |
|------|------|-----------|
| X の分布 | Gaussian のみ: x_i ~ N(F z_i, Sigma) | model.py:22–24 |
| Y の分布 | Bernoulli のみ: y_ij ~ Bernoulli(sigmoid(w0 + w z_i^T z_j)) | model.py:24 |
| Z の事前 | N(0, sigma_z^2 I), var_z=1.0 固定 | model.py:99 |
| Sigma | 対角行列のみ（共分散なし） | model.py:112–113 |
| w0, w | スカラー（各 1 つ） | model.py:116–119 |

### E-step
- Newton 法で MAP 推定 (alpha=0.5)
- Laplace 近似で事後分布 N(eta_i, A_i^{-1}) を構築
- A_i = (1/var_z)I + F^T Sigma^{-1} F + (w^2/2) sum_{j≠i} s_ij(1-s_ij) z_j z_j^T
- L サンプル生成（実験 1/3 では L=10）

### M-step
- F: 解析解（Eq.10）
- Sigma: 対角解析解（Eq.12）
- w0, w: Adam（max_iter=50, lr=0.01）

### 入力フォーマット
- X: (n×d) float64, z-score 正規化済（各列ゼロ平均・単位分散）
- Y: (n×n) float64, バイナリ（0.0 or 1.0）, 対称, 対角=0
- true_params: dict（初期化スケール用）

### 乱数 seed
- data_seed=1980（固定）
- init_seed = 42 + trial × 100（試行ごとに異なる）

### 評価指標
- Q 関数（完全対数尤度）
- MAE(Y): |y_ij - hat{y}_ij| 平均
- RMSE(Y): sqrt(mean((Y_true - Y_pred)^2))  ← 観測値 vs 予測確率
- RMSE(X): sqrt(mean((X - Z_est F^T)^2))
- RMSE(F), RMSE(Z): Procrustes 回転後（experiment_paper_1.py）
- |w0 error|, |w error|: スカラー誤差の絶対値
- BIC = -2 logL + ((k+1)d - k(k-1)/2) ln n

### 実験設定
| 実験 | n | d | k* | k range | L | num_iter | n_trials |
|------|---|---|----|---------|----|---------|---------|
| Exp1 | 150 | 15 | 3 | [1..6] | 10 | 10 | 3 |
| Exp2 | 150 | 15 | [1,3,5,7,9] | [1..10] | 10 | 30 | 3 |
| Exp3(n) | [50..300] | 15 | 3 | k=3 | 10 | 30 | 2 |
| Exp3(d) | 150 | [5..30] | 3 | k=3 | 10 | 30 | 2 |

---

## 2. 私の実験（Dual-ExpFam LSM）の仕様

### モデル前提
| 項目 | 仕様 |
|------|------|
| X の分布 | gaussian/bernoulli/poisson（選択可） |
| Y の分布 | gaussian/bernoulli/poisson（選択可） |
| 一般化箇所 | 勾配 Term2 が F^T diag(A_X''(η)) で置き換え、Hessian Term3 が A_Y''(η) で置き換え |

### 実験設定（exp_scenario_lib.py）
| 項目 | 値 |
|------|-----|
| N, D, K_TRUE | 150, 15, 3 |
| N_TRIALS | 10 |
| L | 5 |
| NITER | 8 |
| k range | [1..6] |
| n range | [50..300] |
| d range | [5..30] |

### シナリオ
- Scenario A: True X=Poisson, True Y=Bernoulli
- Scenario B: True X=Gaussian, True Y=Poisson
- Scenario C: True X=Bernoulli, True Y=Gaussian

---

## 3. 比較可能性の分類

| データシナリオ | Baseline 適用可否 | 理由 | 比較種別 |
|--------------|----------------|------|---------|
| Control: Gauss-X × Bern-Y | **適用可能** | Baseline 本来の前提と完全一致 | **主比較** |
| Scenario A: Pois-X × Bern-Y | 一応走るが X は誤モデル | Y=Bernoulli は正しい。X を Gaussian と仮定するのは誤り | **補助比較（X ミススペック）** |
| Scenario B: Gauss-X × Pois-Y | **実行不能（主結果）** | Y=Poisson は非バイナリ（0,1,2,...）。Baseline は Y ∈ {0,1} を必須とする | 実行不能 |
| Scenario C: Bern-X × Gauss-Y | **実行不能（主結果）** | Y=Gaussian は実数値。同上 | 実行不能 |

### 実行不能の詳細

**Scenario B (Pois Y):** Baseline の calc_w0/calc_w は
`Y - sigmoid(logits)` を計算する。Y が {0,1,2,...} のとき数値的には動くが
モデル外の値（Y>1）を Bernoulli ロス関数で処理することになり、
- 対数尤度 ln(S)^Y が Y>1 でも計算されてしまう（負にはならない）
- 推定値の解釈が完全に無効

ゆえに Scenario B/C のデータを Baseline に食わせることを主結果に使うことを禁止する。

---

## 4. 主比較の設計

### Control シナリオ: Gauss-X × Bern-Y

公平比較のための設定の揃え方:
- **データ生成**: `generate_dual_data(family_x='gaussian', family_y='bernoulli')`（Dual-ExpFam 側と同じ関数を使用）
- **試行数**: 5 trials（計算時間とのトレードオフ。seed は両方で同じ）
- **L**: 5（Dual-ExpFam の設定に合わせる）
- **num_iter**: 8（Dual-ExpFam の設定に合わせる）
- **k range**: [1, 2, 3, 4, 5, 6]
- **評価指標**: RMSE(Z), RMSE(F), RMSE(Y), RMSE(X), |w0 err|, |w err|, BIC

RMSE(Z), RMSE(F): Procrustes 回転後
RMSE(Y): mu_est vs mu_true（観測 Y vs 予測ではなく、真の平均 vs 予測平均で比較）

---

## 5. 補助比較の設計

### Scenario A (Pois-X × Bern-Y): ミススペック下ロバストネス

- Dual-ExpFam の場合: Y=Bernoulli（正しい）, X=Poisson（正しい） → 正しいモデル
- Baseline の場合: Y=Bernoulli（正しい）, X=Gaussian（誤り・ミススペック）
- 結論を「Baseline が弱い」ではなく「X のミスモデルがどれだけ推定精度を落とすか」として解釈する

---

## 6. 注意点

1. **BIC の統一**: Baseline の BIC 式 = ((k+1)d - k(k-1)/2) ln n であり、
   Dual-ExpFam の Gauss-X Bern-Y 用 BIC 式 = (kd - k(k-1)/2 + d) ln n と一致することを確認済み。
   ∵ (k+1)d = kd + d

2. **seed の分離**: data_seed と model_seed を分離する（Dual-ExpFam の慣例通り: mseed = dseed + 50）

3. **EM 設定の差**: 元の Baseline 論文 (L=10, num_iter=10-30) vs 本比較 (L=5, num_iter=8)。
   本比較では Dual-ExpFam の設定に揃えることで公平性を確保。
   元論文設定での Baseline 結果は `reproduction/results/` にあるが、
   別の設定での比較のため直接使用しない。

4. **RMSE(Y) の定義差**: Baseline 論文の RMSE(Y) は観測値 Y vs 予測確率。
   本比較では両方とも mu_est vs mu_true（真の期待値との比較）に統一する。
