# 引き継ぎドキュメント — Dual-ExpFam LSM

**作成日:** 2026-03-24
**前セッションの役割:** 実装者・実験実行者
**次セッションの役割:** 厳格な検証者・レビュアー

---

## 1. 研究の全体目的と現在地

### 背景

Mikawa et al., NOLTA 2024 は、関係データ Y が **Bernoulli 分布に固定**された
潜在構造モデル（LSM）を提案した。この固定前提を打破し、Y・X ともに
任意の指数型分布族を許容する **Dual-ExpFam LSM** を構築するのが本研究の目的。

### 達成済み

| フェーズ | 内容 | 状態 |
|---------|------|------|
| 再現実装 | Mikawa et al. の MATLAB コードを Python で再現 | 完了 |
| Dual-ExpFam 設計 | Research Proposal（数式・クラス設計） | 完了 |
| 実装 | `DualExpFamLSM` クラス、E-step/M-step、Q関数、BIC | 完了 |
| テスト | 5つのユニットテスト（全 PASS） | 完了 |
| 実験 A [Poisson×Bernoulli] | Exp1〜4 全実行、Figure生成 | 完了 |
| 実験 B [Gaussian×Poisson]  | Exp1〜4 全実行、Figure生成 | 完了 |
| 実験 C [Bernoulli×Gaussian]| Exp1〜4 全実行、Figure生成 | 完了 |

### 未完了

- [ ] 実験結果の**厳密な検証**（数式・コード整合性、ベースライン比較）← 次のセッションのミッション
- [ ] 論文 `main.tex` の執筆

---

## 2. ディレクトリ構成

```
C:/研究2/
├── expfam/                          ← メインプロジェクト（Dual-ExpFam）
│   ├── CLAUDE.md                   ← 検証者へのルール（必ず読め）
│   ├── handoff.md                  ← このファイル
│   ├── src/
│   │   ├── model_expfam.py         ← ベースクラス（Y-only ExpFam LSM）
│   │   ├── model_dual_expfam.py    ← 本研究の核心：DualExpFamLSM クラス
│   │   ├── utils_expfam.py         ← Q関数・BIC・run_em_dual など共通ユーティリティ
│   │   ├── data_generator_expfam.py← 合成データ生成（generate_dual_data）
│   │   ├── exp_scenario_lib.py     ← 実験ライブラリ（run_exp1〜4, run_scenario）
│   │   ├── exp_run_scenario_A.py   ← Scenario A ランチャー
│   │   ├── exp_run_scenario_B.py   ← Scenario B ランチャー
│   │   ├── exp_run_scenario_C.py   ← Scenario C ランチャー
│   │   ├── test_dual_expfam.py     ← ユニットテスト（全5テスト）
│   │   └── archive/                ← 旧スクリプト（参照用のみ）
│   ├── results/                    ← 実験出力（現行）
│   │   ├── exp_scenario_{A,B,C}_exp{1,2,3,4}_*.csv  ← 数値データ
│   │   ├── fig_scenario_{A,B,C}_exp{1,2,3,4,5}*.pdf/png  ← Figure
│   │   ├── log_scenario_{A,B,C}.txt  ← 実行ログ（全試行の生数値）
│   │   ├── GEMINI_REPORT_MULTI_SCENARIO.md  ← 最終実験報告書
│   │   ├── RESEARCH_PROPOSAL_DUAL_EXPFAM.md ← 設計書
│   │   └── archive/                ← 旧結果（参照用のみ）
│   ├── references/
│   │   └── baseline_metrics.md     ← Mikawa et al. の論文数値（比較基準）
│   └── design/
│       └── 01_exponential_family_design.md  ← 数式設計書
├── reproduction/                   ← Mikawa et al. の再現実装
│   ├── src/
│   │   ├── model.py                ← 元の LSM モデル（Python）
│   │   └── experiment.py           ← 再現実験ランナー
│   └── reports/
│       └── NUMERICAL_COMPARISON_TABLE.md  ← 論文値 vs 再現実装の比較表
└── paper/
    └── A_study_on_latent_structural_models_for_binary_rel.pdf  ← Mikawa 2024 論文
```

---

## 3. 実装の核心：モデルの数式

### 生成モデル

```
z_i ~ N(0, sigma_z^2 * I_k)
x_{ij} ~ ExpFam_X(eta^X_{ij} = f_j^T z_i)         j = 1..d
y_{ij} ~ ExpFam_Y(eta^Y_{ij} = w0 + w * z_i^T z_j) i < j
```

### E-step 勾配（model_dual_expfam.py: _calc_gradient）

```
∇_{z_i} = Term1 + Term2 + Term3

Term1 = -(1/var_z) * z_i                                          [Zの事前分布]
Term2 = (1/phi_X) * F.T @ (T_X(x_i) - A_X'(F z_i))              [X側: NEW]
Term3 = (w/2phi_Y) * Σ_{j≠i} [T_Y(y_ij) - A_Y'(eta^Y_ij)] z_j  [Y側: 継承]
```

### 精度行列（_calc_precision_matrix）

```
Lambda_i = Term1 + Term2 + Term3

Term1 = (1/var_z) * I
Term2 = (1/phi_X) * F.T @ diag(A_X''(F z_i)) @ F
Term3 = (w^2/2phi_Y) * Σ_{j≠i} A_Y''(eta^Y_ij) * z_j z_j.T
```

### M-step

- **F**: Gaussian X → 解析解（最小二乗）; Bernoulli/Poisson X → Adam
- **sigma**: Gaussian X のみ更新（閉形式 MLE）; 非Gaussian は恒等行列固定
- **w0, w**: Adam（継承）
- **sigma_y**: Gaussian Y のみ更新（継承）

---

## 4. 実験結果の保存場所と読み方

### CSV ファイル形式

#### Exp1: `exp_scenario_{tag}_exp1_k.csv`
列: `scenario, k_est, trial, rmse_Z, BIC, Q_strict`

#### Exp2/3: `exp_scenario_{tag}_exp{2,3}_{n,d}.csv`
列: `scenario, n_or_d_value, trial, rmse_Z, rmse_F, rmse_sigma, rmse_Y, rmse_X, w0_err, w_err`

#### Exp4: `exp_scenario_{tag}_exp4_mismatch.csv`
列: `scenario, fit_fx, fit_fy, trial, rmse_Z` (+ `condition` 列: 'grid'/'fix_x'/'fix_w')

### 実行ログの読み方

`log_scenario_A.txt` の末尾に **Mismatch Summary** がある：
```
Proposed (Pois,Bern): 0.2787
X=Gauss (Y=Bern)    : 0.7021 (2.52x)
...
```
この multiplier が 1.0 に近いほど提案手法は優位性を失う。

---

## 5. 次のセッションで重点的に検証すべきポイント

### 優先度 HIGH — コードの正確性

1. **Term2 の係数確認**
   - `model_dual_expfam.py` の `_calc_gradient` を開き、
     Bernoulli/Poisson X の場合に phi_X の係数が正しく掛かっているか
   - 論文の Eq.(23) と 1:1 で対照せよ

2. **Procrustes アライメントの適用確認**
   - `utils_expfam.py` の `run_em_dual` で `procrustes_rotation` が呼ばれているか
   - 全 7 指標のうち RMSE(Z), RMSE(F) にのみアライメントが必要（sigma はスカラー）

3. **BIC の num_params 算出**
   - `calc_bic_dual` の実装を確認
   - 期待値: k*d - k*(k-1)//2 + [d if Gaussian X] + [1 if Gaussian Y]
   - Scenario C で BIC が -35700 になる理由を確認（Gaussian Y の ln(2π) 定数）

4. **fix_x アブレーションの実装**
   - `run_em_dual(fix_x=True)` のとき、F=0 かつ M-step で F が更新されないことを確認

### 優先度 HIGH — 実験結果の妥当性

5. **RMSE(Y) が全シナリオで悪い問題**
   - 再現実装比較表 (`reproduction/reports/NUMERICAL_COMPARISON_TABLE.md`) でも
     RMSE(Y) は論文より悪い傾向 → 実装バグか、計算方法の差異か
   - Dual-ExpFam でも同様なら同じバグが存在する可能性

6. **Scenario A baseline: Proposed RMSE(Z) = 0.279**
   - これは k=3 正解時の値。再現実装（Y-only, Bernoulli）の n=150 での RMSE(Z) = 0.2018 より悪い
   - **なぜ Dual-ExpFam が Y-only より RMSE(Z) で悪いのか？** — X 側のノイズの影響か要確認

7. **Scenario B・C の mismatch の非対称性**
   - Scenario C で X mismatch が軽微（Poisson X が 0.99×）なのに Y mismatch が深刻（15×）
   - これは正しいか、それともコードバグで X 情報が使われていないのか

### 優先度 MEDIUM — 実験設定

8. **L=5, num_iter=8 の妥当性**
   - Q 関数の収束プロットを確認（`exp_scenario_lib.py` に含まれているか）

9. **seed の再現性**
   - 同じ seed で再実行して同じ結果が出るか検証

---

## 6. 重要な既知バグと修正履歴

| バグ | 修正 | ファイル |
|------|------|--------|
| Newton step size alpha=0.01 が小さすぎた | alpha=0.5 に修正 | 再現実装 (reproduction/) |
| calc_log_likelihood_X で -0.5*ln(2π) 欠落 | 追加 | model_dual_expfam.py |
| テストスクリプトの全角文字エンコードエラー | ASCII に置換 | test_dual_expfam.py |

---

## 7. 参考文献

- Mikawa, Kanai, Iida, Mitsumoto, "A study on latent structural models for binary relational data," NOLTA 2024
  → `paper/A_study_on_latent_structural_models_for_binary_rel.pdf`
- 再現実装との数値比較: `reproduction/reports/NUMERICAL_COMPARISON_TABLE.md`
