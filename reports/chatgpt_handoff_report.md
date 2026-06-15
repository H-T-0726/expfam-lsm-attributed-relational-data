# ChatGPT向け研究報告書
## 「属性情報付き関係データを対象とした潜在構造分析手法の指数型分布族への拡張」

作成日：2026-05-31  
調査者：Claude Code（プロジェクト新規調査）

---

## 1. 研究の現状まとめ

- **原稿**：`conference_submission_final_draft.md` が完成済み（日本語、約2ページ予稿形式）
- **実装**：提案手法 `DualExpFamLSM`（Python）が完成・テスト済み
- **実験**：3シナリオ×4実験の人工データ実験が完了し、CSV・図が揃っている
- **先行研究比較**：Control条件での比較実験が完了し RMSE(Z) 差 < 0.001
- **発表資料**：PowerPoint は未確認（リポジトリ外）。前半スライド構成はユーザーが設計済み

---

## 2. プロジェクト構成

```
conference_submission_final_draft.md   完成済み原稿
figures/
  fig1a_n_sweep_color.pdf/png          提出用図(a): n vs RMSE(Z)
  fig1b_misspecification_color.pdf/png 提出用図(b): 誤指定倍率

expfam/
  src/
    model_dual_expfam.py               提案手法本体 DualExpFamLSM
    model_expfam.py                    中間クラス ExpFamLatentStructuralModel
    data_generator_expfam.py           データ生成（Bernoulli/Poisson/Gaussian）
    exp_scenario_lib.py                実験ライブラリ（共通設定）
    exp_run_scenario_A/B/C.py          実験実行スクリプト（各シナリオ）
    utils_expfam.py                    BIC・RMSE・Procrustes 等ユーティリティ
  results/
    exp_scenario_A_exp1_k.csv          シナリオA Exp1: BIC vs k
    exp_scenario_A_exp2_n.csv          シナリオA Exp2: RMSE vs n
    exp_scenario_A_exp3_d.csv          シナリオA Exp3: RMSE vs d
    exp_scenario_A_exp4_mismatch.csv   シナリオA Exp4: 分布ミスマッチ3×3
    exp_scenario_B_exp*.csv            シナリオB 同上
    exp_scenario_C_exp*.csv            シナリオC 同上
    exp1_full_A/B/C.csv                先行研究比較用
    distribution_mismatch_fixed/
      mismatch_fixed_summary.csv       ミスマッチ集約版（別実験）

reproduction/
  src/model.py                         先行研究（Mikawa 2024）Python再現
  results/comparison/comparison_main_table.csv  先行研究比較数値
  
Mato Lab Program/*.m                   先行研究 MATLAB オリジナル

paper/
  A_study_on_latent_structural_models_for_binary_rel.pdf  先行研究論文PDF
```

---

## 3. 実装されている提案手法

### 3-1. モデル構造

| 要素 | 式 | 実装ファイル |
|------|----|----|
| 潜在変数事前分布 | z_i ~ N(0, I_k) | model.py L.100 |
| 属性モデル | x_il ~ ExpFam_X(η_il^X = f_l^T z_i) | model_dual_expfam.py L.91 |
| 関係モデル | y_ij ~ ExpFam_Y(η_ij^Y = w_0^Y + w^Y z_i^T z_j) | model_expfam.py L.51 |

- **ExpFam_X**：gaussian, bernoulli, poisson（実装済み）
- **ExpFam_Y**：gaussian, bernoulli, poisson（実装済み）
- **Categorical**：未実装

### 3-2. 指数型分布族の実装

| 分布 | 実装有無 | X/Y 両方 | A'(η) | A''(η) | コード場所 |
|------|---------|---------|--------|--------|-----------|
| Gaussian | ○ | 両方 | η（恒等写像） | 1/σ² | model_expfam.py L.51-76 |
| Bernoulli | ○ | 両方 | sigmoid(η) | σ(η)(1-σ(η)) | model_expfam.py L.51-76 |
| Poisson | ○ | 両方 | exp(η) | exp(η) | model_expfam.py L.51-76 |
| Categorical | ✗ | - | - | - | 未実装 |

### 3-3. パラメータ推定方法

| ステップ | 方法 | 対象パラメータ | 実装場所 |
|---------|------|----------------|---------|
| E-step | Laplace近似 + Newton法 | 各 z_i の事後モード m_i | model.py (基底クラス) |
| E-step | MCサンプリング | z_i ~ N(m_i, A_i^{-1})、L=5サンプル | utils_expfam.py |
| M-step | 解析解（閉形式） | F（Gaussian X のとき） | model.py 基底クラス |
| M-step | Adam 勾配上昇 | F（非Gaussian X のとき） | model_dual_expfam.py L.219-268 |
| M-step | 解析解 | Σ（Gaussian X のみ） | model_dual_expfam.py L.274-285 |
| M-step | Adam 勾配上昇 | w_0^Y, w^Y（全分布共通） | model_expfam.py L.149-210 |
| M-step | 解析解 | σ_y（Gaussian Y のみ） | model_expfam.py L.212-235 |

**注：EM反復回数 = 8、MCサンプル数 L = 5（exp_scenario_lib.py L.42）**

### 3-4. 精度行列 A_i（論文式・確定）

```
A_i = I_k + F^T V_X(m_i) F + (w^Y)^2 Σ_{j≠i} A_Y''(η_ij^Y) z_j z_j^T
```

- V_X = Σ^{-1}（Gaussian X）
- V_X = diag(A_X''(F m_i))（Bernoulli/Poisson X）

**実装注意（CLAUDE.md より）：**  
`model_dual_expfam.py` L.200 の Term3 に `0.5 * (w**2)` と spurious な 0.5 が残っている。  
論文式は 1/2 なし（正しい）。Python実装は 0.5 付き（修正は今後の課題、現在は触らない）。  
gradient（L.159）にも同様の 0.5 が残っている。

---

## 4. 従来手法との差分

| 項目 | 従来手法（Mikawa 2024） | 提案手法 | コード上の根拠 |
|------|----------------------|---------|--------------|
| 属性データの分布 | Gaussian のみ（固定） | Gaussian / Bernoulli / Poisson を選択可 | DualExpFamLSM.family_x |
| 関係データの分布 | Bernoulli のみ（固定） | Gaussian / Bernoulli / Poisson を選択可 | DualExpFamLSM.family_y（= self.family） |
| 潜在変数 | N(0, σ_z^2 I) | N(0, I_k)、固定 σ_z=1 | model.py（共通） |
| 生成モデル | η^X = F z_i、η^Y = w_0 + w z_i^T z_j | 同形式（分布族のみ変更） | 共通 |
| E-step 精度行列 | F^T Σ^{-1} F（Gaussian X 固定） | F^T V_X(m_i) F（分布族に応じて変化） | model_dual_expfam.py L.183-194 |
| M-step F 更新 | 閉形式解析解 | Gaussian X → 閉形式；非Gaussian → Adam | model_dual_expfam.py L.208-217 |
| M-step Σ 更新 | 閉形式 | Gaussian X → 閉形式；非Gaussian → 適用外（単位行列） | model_dual_expfam.py L.274-285 |
| M-step w_0, w 更新 | Adam（共通） | Adam（共通） | model_expfam.py L.149-210 |
| まだ未実装の部分 | - | Categorical分布、実データ適用、BIC以外の選択基準 | - |

### 専門的説明

従来手法では、E-step精度行列の第2項（属性側）が `F^T Σ^{-1} F`（Gaussian固有）であった。提案手法では、これを `F^T diag(A_X''(F m_i)) F`（分布族の分散関数 A_X'' を使用）に一般化した。M-stepのF更新も同様に、Gaussian X の閉形式解 `F^* = (Σ^{-1} X^T Z)(Z^T Z)^{-1}` を非Gaussian X では Adam によるQ関数最大化に置き換えた。関係データ側は先行研究の Y=Bernoulli 固定を ExpFam_Y に一般化し、Poisson・Gaussian にも対応した。

### 発表用説明

「変えるのは観測分布、残すのは潜在構造」

潜在変数 Z の仮定（N(0,I)）も、潜在変数と観測をつなぐ自然パラメータの形式（η = F z, η = w₀ + w z_iᵀz_j）も、推定方法（MCEM + Laplace近似）も変わらない。変わるのは「どの分布族を使うか」という選択だけ。

---

## 5. 実施済み実験一覧

| 実験名 | 目的 | 比較対象 | 人工/実データ | 評価指標 | 結果CSV |
|--------|------|----------|-------------|---------|--------|
| Exp1: BIC次元選択 | k*=3 が自動同定されるか | k=1〜6 の比較 | 人工 | RMSE(Z), BIC | exp_scenario_{A/B/C}_exp1_k.csv |
| Exp2: n-sweep | n増加で精度改善するか | n=50,100,...,300 | 人工 | RMSE(Z) ほか7指標 | exp_scenario_{A/B/C}_exp2_n.csv |
| Exp3: d-sweep | d増加で精度改善するか | d=5,10,...,30 | 人工 | RMSE(Z) ほか7指標 | exp_scenario_{A/B/C}_exp3_d.csv |
| Exp4: 分布ミスマッチ | 誤った分布族を指定するとどうなるか | 3×3 すべての分布族組み合わせ＋Ablation | 人工 | RMSE(Z) | exp_scenario_{A/B/C}_exp4_mismatch.csv |
| Control比較 | 先行研究と同条件で同等か | Mikawa 2024（X=Gauss, Y=Bern） | 人工 | RMSE(Z/F/Y/X) | reproduction/results/comparison/comparison_main_table.csv |
| ミスマッチ固定版 | （別実験）同条件での3×3検証 | 同上 | 人工 | RMSE(Z), 悪化倍率 | distribution_mismatch_fixed/mismatch_fixed_summary.csv |

**全実験共通条件：n=150（基本値）、d=15、k*=3、10試行、L=5 MCサンプル、8 EM反復**

---

## 6. 各実験の目的・設定・結果・解釈

### Exp1: BIC次元選択

**目的**：提案手法がデータから正しい潜在次元 k*=3 を自動的に選択できるかを確認。

**結果（CSV確認済み）：**

| シナリオ | k=3 RMSE(Z) | BIC最小k | RMSE(Z)最小k |
|--------|-------------|---------|------------|
| A [Pois-Bern] | 0.278±0.012 | k=3 ✓ | k=3 ✓ |
| B [Gauss-Pois] | 0.182 | k=3 ✓ | k=3 ✓ |
| C [Bern-Gauss] | 0.028 | k=3 ✓ | k=3 ✓ |

**表1（原稿記載）との対応：**
- 原稿の表1数値（Scen.A: 0.278、B: 0.182、C: 0.028）と一致（CSV確認済み）

**解釈**：3種類の異なる分布族組み合わせすべてで、BICが真の次元を正しく選択した。

### Exp2: n-sweep

**目的**：サンプル数増加に伴う漸近一致性の確認。

**結果（exp_scenario_A_exp2_n.csv より計算）：**

| n | Scen.A RMSE(Z) | Scen.B | Scen.C |
|---|---------------|--------|--------|
| 50 | 0.406 | 0.190 | 0.053 |
| 100 | 0.319 | 0.162 | 0.035 |
| 150 | 0.279 | 0.148 | 0.029 |
| 200 | 0.247 | 0.139 | 0.025 |
| 250 | 0.225 | 0.135 | 0.022 |
| 300 | 0.208 | 0.131 | 0.020 |
| 削減率 | -49% | -31% | -62% |

**注意点**：Scen.B では n=50→100 の間に一時的な変動が見られる（原稿に記載済み）。

### Exp3: d-sweep

**結果（GEMINI_REPORT_MULTI_SCENARIO.md 参照）：**

| d | Scen.A | Scen.B | Scen.C |
|---|--------|--------|--------|
| 5 | 0.322 | 0.200 | 0.029 |
| 15 | 0.279 | 0.148 | 0.029 |
| 30 | 0.236 | 0.146 | 0.029 |

**注意点**：Scen.C は Y=Gaussian が支配的なため、d 増加の効果がほぼ見られない。これは「Y 側の情報量 >> X 側」という非対称性を示す現象であり、原稿に記載済み。

### Exp4: 分布ミスマッチ

**目的**：誤った分布族を指定した場合の精度劣化を定量的に示す。

**3×3 ヒートマップ結果（CSV確認済み）：**

Scen.A [True X=Poisson, Y=Bernoulli]

| モデルX＼モデルY | Gaussian | Bernoulli | Poisson |
|---------------|---------|----------|---------|
| Gaussian | ~0.714 (2.56×) | ~0.702 (2.52×) | ~0.776 (2.78×) |
| Bernoulli | ~0.790 (2.83×) | ~0.949 (3.41×) | ~0.783 (2.81×) |
| **Poisson** ✓ | ~0.296 (1.06×) | **0.278 (1.00×)** | ~0.373 (1.34×) |

Scen.B [True X=Gaussian, Y=Poisson]

- 正解 (Gaussian, Poisson) RMSE(Z) ≈ 0.178
- 最大悪化: (Gaussian, Bernoulli) ≈ 1.129 → 6.34× ～ 7.35×（trials間のばらつきあり）

**⚠ 原稿記載数値との照合：**
- Scen.A 最大 3.41×：CSV（exp_scenario_A_exp4_mismatch.csv, Bern-Bern条件）で確認可能
- Scen.B 最大 7.35×：mismatch_fixed_summary.csv では 7.80×、exp_scenario_B_exp4 では ~6.34×〜7.35×
- Scen.C 最大 41.5×：mismatch_fixed_summary.csv では 38.97×（Pois-Bern条件）、CLAUDE.md は Gauss-Pois条件と記載
  → **数値の根拠CSVを要確認。原稿 L.83「最大41.5倍」に対応するCSVが明確でない**

### Control比較（先行研究との比較）

**結果（comparison_main_table.csv L.2-13 確認済み）：**

| モデル | RMSE(Z) | RMSE(F) | RMSE(Y) | RMSE(X) |
|-------|---------|---------|---------|---------|
| 先行研究 [1] | 0.179±0.014 | 0.040±0.011 | 0.088±0.008 | 0.307±0.006 |
| 提案手法 | 0.180±0.016 | 0.035±0.006 | 0.088±0.007 | 0.307±0.005 |

- 条件：X=Gaussian, Y=Bernoulli, n=150, 5試行
- 差：RMSE(Z)で 0.001（誤差の範囲内）

---

## 7. 発表で使える図表一覧

| 優先度 | 図表名 | ファイルパス | 何を示す |
|-------|-------|------------|---------|
| A | n-sweep カラー版 | figures/fig1a_n_sweep_color.pdf/png | 3シナリオのRMSE(Z) vs n（n=50基準で正規化） |
| A | 誤指定倍率カラー版 | figures/fig1b_misspecification_color.pdf/png | X-side/Y-side/固定条件の悪化倍率 |
| B | 3×3ヒートマップ（Scen.A） | expfam/results/fig_scenario_A_exp4_heatmap.pdf | 誤指定の全組み合わせRMSE |
| B | BIC vs k（Scen.A） | expfam/results/fig_scenario_A_exp1_k.pdf | BIC次元選択の可視化 |
| C | 7指標 vs n（Scen.A） | expfam/results/fig_scenario_A_exp2_n.pdf | 全評価指標の一覧 |
| C | 各シナリオのヒートマップ | expfam/results/fig_scenario_{B,C}_exp4_heatmap.pdf | 補足: B・Cシナリオの誤指定影響 |

**発表用コピーファイル：**
- `reports/figures_for_slides/fig01_basic_performance.png` = fig1a_n_sweep_color.png
- `reports/figures_for_slides/fig02_distribution_switching.png` = fig1b_misspecification_color.png

---

## 8. 発表で言えること・言えないこと

### 強く言えること（CSV・図で確認済み）

1. **BIC による潜在次元の自動選択が3シナリオすべてで成功**（k=3を正確に選択）
2. **n 増加に伴う推定精度の改善**（3シナリオで49%〜62%の削減を確認）
3. **先行研究と同条件で同等の精度を達成**（RMSE(Z)差 < 0.001）
4. **分布族の誤指定が推定精度を最大3倍以上悪化させる**（Scen.Aで最大3.41×、CSV確認済み）
5. **指数型分布族への拡張がモデルの骨格を変えずに実現可能であること**

### 控えめに言うべきこと（一部条件でのみ確認、または数値に不確実性あり）

1. **41.5倍という数値**：原稿に記載あるが、対応CSVが特定困難。38.97×（mismatch_fixed）は確認済み
2. **Scen.B の n=50→100 で一時的な変動**：原因不明（原稿に記載・説明なし）
3. **Scen.C の d 平坦性**：Y=Gaussian が支配的という解釈は妥当だが、実験回数・条件が限定的

### まだ言えないこと（実験未実施または根拠不足）

1. **実データへの適用**：未実施（wine データの簡易実験 run_wine_dual.py の結果は CSV あるが詳細未確認）
2. **Categorical 分布への対応**：未実装
3. **大規模データへのスケーラビリティ**：n=300 が上限（計算時間の観点からの限界も未調査）
4. **推定アルゴリズムの収束保証**：L=5 サンプル・8反復が十分かどうかの理論的保証なし
5. **実装バグ（0.5 係数）の影響定量化**：現在の実験結果への影響は「推測」にとどまる

---

## 9. 15分発表の後半構成案

前半（6スライド）はユーザー設計済み。後半スライドは以下を提案。

| # | タイトル | 目的 | 目安時間 |
|---|---------|------|---------|
| 7 | 提案手法：全体像 | 「変えるのは観測分布、残すのは潜在構造」 | 1分 |
| 8 | 提案モデルの定式化 | 式(4)(5)の提示 | 1.5分 |
| 9 | 推定方法 | MCEM + Laplace近似の概要 | 1.5分 |
| 10 | 実験設定 | シナリオA/B/C、評価指標の説明 | 1分 |
| 11 | 実験結果(1)：先行研究との比較 | 表2の提示 | 1分 |
| 12 | 実験結果(2)：n-sweep | 図1(a)の提示 | 1分 |
| 13 | 実験結果(3)：分布ミスマッチ | 図1(b)の提示 | 1.5分 |
| 14 | 考察 | 何が分かったか | 1分 |
| 15 | まとめ・今後の課題 | 結論 | 1分 |

**合計：後半 10.5分 + 前半 3〜4分 = 計13.5〜14.5分（質疑含めて15分に収まる）**

---

## 10. スライド文章案

### スライド7：提案手法の全体像

**タイトル**：提案手法：指数型分布族への統一的拡張

**メインメッセージ**：変えるのは観測分布、残すのは潜在構造

**箇条書き**：
- 関係データ Y、属性データ X の分布族をそれぞれ独立に指定可能
- 潜在変数の仮定・自然パラメータの形式・推定方法は従来手法と同一
- 分布族の変更のみで Bernoulli → Poisson → Gaussian に対応

---

### スライド8：提案モデルの定式化

**タイトル**：生成モデル

**式**：

z_i ~ N(0, I_k)

y_ij ~ ExpFam_Y(η_ij^Y),   η_ij^Y = w_0^Y + w^Y z_i^T z_j   (i < j)

x_il ~ ExpFam_X(η_il^X),   η_il^X = f_l^T z_i

θ = {F, w_0^Y, w^Y}（Gaussian X のときのみ Σ を追加推定）

**口頭**：「EpFam_Y に Bernoulli を指定すると先行研究と完全に一致します。分布族の切り替えが式のどこで起きているかを見てください」

---

### スライド9：推定方法

**タイトル**：MCEM + Laplace近似による推定

**箇条書き**：
- E-step：各 z_i の事後分布を Gaussian でラプラス近似 → MC サンプリング
  - 精度行列 A_i：第2項（属性側）が分布族に応じた V_X(m_i) に一般化（先行研究との差異）
- M-step：F（非Gaussian X → Adam）、w_0, w（Adam）、Σ（Gaussian X → 解析解）
- 潜在次元 k の選択：BIC

**口頭**：「先行研究と変わるのは精度行列の第2項と、非Gaussian のときの F 更新のみです」

---

### スライド10：実験設定

**タイトル**：人工データによる評価実験

**内容**：
- n=150, d=15, k*=3（真の潜在次元）、10試行
- 3種類のシナリオ：
  - Scen.A：属性=Poisson、関係=Bernoulli
  - Scen.B：属性=Gaussian、関係=Poisson
  - Scen.C：属性=Bernoulli、関係=Gaussian
- 評価指標：RMSE(Z)（Procrustes回転適用後）を主指標
- 先行研究との比較：X=Gaussian, Y=Bernoulli の Control 条件でも評価

---

### スライド11：実験結果(1) — 先行研究との比較

**タイトル**：先行研究と同条件で同等の精度を達成

**内容（表2）**：

| モデル | RMSE(Z) | RMSE(Y) |
|--------|---------|---------|
| 先行研究[1] | 0.179±0.014 | 0.088±0.008 |
| 提案手法 | 0.180±0.016 | 0.088±0.007 |

**メッセージ**：「先行研究を特殊ケースとして包含しており、性能を損なわない」

---

### スライド12：実験結果(2) — n-sweep

**タイトル**：サンプル数増加に伴い全シナリオで精度改善

**図**：fig1a_n_sweep_color.png（提出用）

**口頭**：「縦軸は n=50 での RMSE(Z) を 1.0 に正規化した相対値です。3シナリオすべてで n が増えるにつれ精度が向上しています」

---

### スライド13：実験結果(3) — 分布ミスマッチ

**タイトル**：誤った分布族指定が推定精度を最大 X 倍悪化させる

**図**：fig1b_misspecification_color.png（提出用）

**口頭**：「横軸が各シナリオ、縦軸が正解条件比の悪化倍率です。水色が X 側のみ誤指定、橙が Y 側のみ誤指定、灰色が従来手法相当（Gauss-X/Bern-Y 固定）です」

**注意**：図に表示される最大値は 23.6×（Scen.C の従来手法相当条件）。原稿本文の 41.5× は図中に対応するバーがない（両方誤指定の最大値）

---

### スライド14：考察

**タイトル**：考察

**箇条書き**：
- 正確な分布族指定が不可欠：誤指定は最大 3〜38 倍の精度劣化を引き起こす
- 分布族を変えても潜在構造の推定は機能する：BIC が全シナリオで k*=3 を正確に選択
- シナリオ C の特殊性（Y=Gaussian 支配）：Y 側の情報量が X 側を圧倒し、X の分布指定の影響が相対的に小さい

**限界**：
- 実装に残る数値的課題（精度行列の 0.5 係数）が結果に与える影響は未検証
- 実データへの適用と Categorical 分布は今後の課題

---

### スライド15：まとめ

**タイトル**：まとめ

**箇条書き**：
- 関係データと属性データの双方を指数型分布族に拡張した潜在構造モデルを提案
- 人工データ実験により、3点を確認：
  1. BIC による潜在次元の正確な自動選択
  2. サンプル数増加に伴う推定精度の一貫した改善
  3. 正確な分布族指定の重要性（誤指定で最大数十倍の精度劣化）
- 先行研究と同条件での同等性も確認（RMSE(Z) 差 < 0.001）
- 今後：実データへの適用、Categorical 分布対応、実装の精緻化

---

## 11. まとめスライド候補

### 強めの結論

1. 「関係データと属性データの双方を指数型分布族に統一的に拡張することで、多様なデータ型に適応可能な潜在構造モデルを実現した」
2. 「分布族の誤指定は推定精度を最大3.41倍以上悪化させることが実験的に示された」
3. 「提案手法は先行研究の特殊ケースを包含し、同条件での同等性も確認された」

### 安全な結論

1. 「3種類の分布族組み合わせすべてで、BICが真の潜在次元を正確に選択することを確認した」
2. 「サンプル数の増加に伴い、全シナリオで推定精度が一貫して改善した」
3. 「先行研究と同一条件での数値比較から、提案手法が従来モデルの性能を維持しつつ適用範囲を拡大できることを示した」

### 控えめな結論

1. 「人工データを用いた実験において、指数型分布族への拡張が有効に機能することを示した」
2. 「提案手法は正確な分布族指定が性能に重要であることを定量的に確認した」
3. 「先行研究との比較実験において、同等の推定精度が得られることが示唆された」

---

## 12. 想定質問と回答

| 想定質問 | 回答案 | 根拠 | 注意点 |
|---------|-------|------|-------|
| 従来手法と何が違うのか | 観測分布を Bernoulli 固定から指数型分布族（Bernoulli/Poisson/Gaussian）の選択制に拡張した | CLAUDE.md、model_dual_expfam.py | 潜在変数の仮定と推定枠組みは同一 |
| なぜ指数型分布族を使うのか | 多様なデータ型（二値・カウント・連続値）を共通の数学的枠組みで統一的に扱えるため | 原稿 Section 2.3、Bishop 2006 | Categorical は現在未実装 |
| どの分布まで扱えるか | 現在の実装では Gaussian, Bernoulli, Poisson の3種類 | model_dual_expfam.py L.61 VALID_FAMILIES | Categorical は未実装 |
| 推定方法は従来と何が違うか | E-step の精度行列第2項が V_X(m_i) に一般化（分布族依存）、非Gaussian X の F 更新が Adam に変更 | model_dual_expfam.py L.167-202, 208-268 | Y 側の更新式は同一 |
| Laplace 近似を使う理由は | z_i の事後分布を Gaussian で近似し解析的に取り扱うことで、効率的なサンプリングが可能になるため | 先行研究 Mikawa 2024 の設計を引き継ぎ | 近似誤差の定量評価は未実施 |
| MCEMを使う理由は | 潜在変数が混合した非線形モデルで完全データ尤度を解析的に扱えないため、MC サンプルで Q 関数を近似する | 先行研究から継承 | L=5 サンプルの十分性は理論的に未保証 |
| 実験では何を確認したのか | BIC次元選択・漸近一致性・分布ミスマッチ影響・先行研究との同等性、の4点 | 原稿 Section 4、各 CSV | 実データ実験は未実施 |
| 従来手法と比べて本当に良いのか | 同条件では同等（RMSE(Z) 差 < 0.001）。「良い」のではなく「同等を保ちつつ適用範囲を拡張した」という主張 | comparison_main_table.csv | Poisson/Gaussian Y/X への適用が主な貢献 |
| どの結果から有効性が言えるか | Scen.A の 3.41× 悪化実験（正しい分布指定 vs 誤指定）と、全シナリオでのBIC次元選択成功 | exp_scenario_A_exp4_mismatch.csv、exp*_exp1_k.csv | シナリオ数（3）が十分かどうかの議論に備える |
| まだできていないことは何か | Categorical分布の実装、実データへの本格適用、精度行列の0.5係数バグ修正 | CLAUDE.md | |
| 今後の課題は何か | 実データ適用、アルゴリズムの精緻化（L・反復数）、分布族自動選択、Categorical対応 | 原稿 Section 5 | |

---

## 13. ChatGPTに相談したいこと

1. **発表全体のストーリー設計**：「指数型分布族への拡張」というテーマで15分の流れをどう構成するか
2. **数式の見せ方**：精度行列 A_i の式はスライドに出すべきか。出すなら何を強調するか
3. **実験結果の見せ方**：3シナリオのうち、本編で扱う実験と補足に回す実験の選び方
4. **41.5倍という数値の扱い**：図中に対応するバーがないが原稿に記載されている。発表でどう触れるか
5. **Scen.C の Y=Gaussian 支配という現象**の説明の仕方（面白い発見だが直感に反する可能性）
6. **「実用上の価値」の主張方法**：人工データのみの実験で、どのような主張が適切か
7. **まとめスライドの表現**：「有効性を確認した」という表現は使えるか

---

## 14. 根拠ファイル一覧

| 内容 | ファイルパス |
|------|------------|
| 完成原稿 | conference_submission_final_draft.md |
| 提案手法 | expfam/src/model_dual_expfam.py |
| 基底クラス（Y拡張） | expfam/src/model_expfam.py |
| 従来手法再現 | reproduction/src/model.py |
| データ生成 | expfam/src/data_generator_expfam.py |
| Exp1 CSV（Scen.A） | expfam/results/exp_scenario_A_exp1_k.csv |
| Exp2 CSV（Scen.A） | expfam/results/exp_scenario_A_exp2_n.csv |
| Exp4 CSV（Scen.A） | expfam/results/exp_scenario_A_exp4_mismatch.csv |
| 先行研究比較 CSV | reproduction/results/comparison/comparison_main_table.csv |
| 図1a (提出用) | figures/fig1a_n_sweep_color.pdf/png |
| 図1b (提出用) | figures/fig1b_misspecification_color.pdf/png |
| 図の説明 | figures/figure_color_split_report.md |
| 実験設定定数 | expfam/src/exp_scenario_lib.py（N=150, D=15, K_TRUE=3, N_TRIALS=10, L=5, NITER=8） |
| 数式メモ | docs/math_notes/half_factor_math_explanation.md |
| CLAUDE.md（既知の問題） | CLAUDE.md |
