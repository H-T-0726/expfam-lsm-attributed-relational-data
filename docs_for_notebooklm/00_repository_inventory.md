# リポジトリ構成整理

**作成日:** 2026-05-18  
**目的:** NotebookLM 投入のための研究理解資料（Phase 0）  
**方針:** 推測禁止・不明点は「未確認」記載・CLAUDE.md（root）の確定事項を優先

---

## 1. 全体構成

```
D:/tento/kennkyu/
├── CLAUDE.md                          ★ 研究全体の確定事項まとめ（最重要）
├── conference_submission_final_draft.md  ★ 学会予稿完成版
│
├── expfam/                            ★ 提案手法 Dual-ExpFam LSM 本体
│   ├── CLAUDE.md                      ⚠ 旧セッション用（内容が root CLAUDE.md と部分的に異なる）
│   ├── handoff.md                     引き継ぎドキュメント（2026-03-24）
│   ├── references/baseline_metrics.md ベースライン数値（先行研究の論文値と再現実装値）
│   ├── design/                        設計メモ（旧Geminiセッション用、参考程度）
│   ├── data/ml-100k.zip               MovieLens実データ（今回は不使用）
│   ├── src/                           ★ ソースコード本体
│   │   ├── model_expfam.py            Y側 ExpFam 拡張基底クラス
│   │   ├── model_dual_expfam.py       ★ 提案手法核心（DualExpFamLSM）
│   │   ├── utils_expfam.py            Q関数・BIC・RMSE・Procrustes・EM実行
│   │   ├── data_generator_expfam.py   人工データ生成
│   │   ├── exp_scenario_lib.py        実験共通設定・実験関数
│   │   ├── exp_run_scenario_A.py      シナリオA実行スクリプト
│   │   ├── exp_run_scenario_B.py      シナリオB実行スクリプト
│   │   ├── exp_run_scenario_C.py      シナリオC実行スクリプト
│   │   ├── run_exp1_full_metrics.py   補足実験（全メトリクス詳細）
│   │   ├── run_exp2_bic_v2.py         BIC詳細実験
│   │   ├── run_wine_dual.py           実データ（Wine）実験
│   │   ├── test_dual_expfam.py        ユニットテスト（5つ全PASS済み）
│   │   └── archive/                   旧実験スクリプト群（後回し）
│   └── results/                       ★ 実験結果
│       ├── exp_scenario_A_exp{1-4}.csv  シナリオA の試行別CSV
│       ├── exp_scenario_B_exp{1-4}.csv  シナリオB の試行別CSV
│       ├── exp_scenario_C_exp{1-4}.csv  シナリオC の試行別CSV
│       ├── exp1_full_{A,B,C}.csv       BIC付き full メトリクス
│       ├── exp2_bic_{A,B,C}.csv        BIC選択実験
│       ├── exp2_bic_log.txt            BIC実験ログ
│       ├── log_scenario_{A,B,C}.txt    シナリオ別実行ログ
│       ├── fig_scenario_{A,B,C}_*.pdf/png  シナリオ別図（全6種×3シナリオ）
│       ├── fig1_rmse_vs_n.*            （ルート figures/ に提出用があるため旧版の可能性）
│       ├── fig2_heatmap_*.*, fig2_mismatch_*.* （同上）
│       ├── GEMINI_REPORT_MULTI_SCENARIO.md  AI生成レポート（未検証）
│       ├── GEMINI_REPORT_PHASE2_FINAL.md    AI生成レポート（未検証）
│       ├── RESEARCH_PROPOSAL_DUAL_EXPFAM.md 旧設計書
│       ├── wine_dual_results.csv, wine_F.npy 実データ実験結果
│       └── archive/                   旧実験結果群（後回し）
│
├── figures/                           ★ 論文提出用図（最終版）
│   ├── fig1a_n_sweep_color.pdf/png    図1（n-RMSE sweep、提出用）
│   ├── fig1b_misspecification_color.pdf/png 図2（分布族誤指定、提出用）
│   └── figure_color_split_report.md  図の作成記録
│
├── reproduction/                      先行研究（Mikawa et al. 2024）Python再現
│   ├── src/model.py                   ★ ベースラインモデル（DualExpFamLSM の親クラスの親）
│   ├── src/data_generator.py          ベースラインデータ生成
│   ├── src/experiment*.py             再現実験スクリプト群
│   ├── results/                       再現実験結果
│   └── reports/                       再現実験レポート
│
├── Mato Lab Program/                  先行研究オリジナルMATLABコード
│   ├── calcEtaNewton.m                ★ E-step（精度行列 calcAi を含む）
│   ├── calcw.m, calcw0.m              w, w0 の M-step
│   └── その他 .m ファイル
│
├── paper/                             先行研究PDF
│   ├── A_study_on_latent_structural_models_for_binary_rel.pdf  ★ 先行研究本体
│   └── 2.pdf                          未確認（別論文の可能性）
│
├── teacher_reply_draft.md             先生への返答（Q1/Q2/Q4）
├── half_factor_teacher_reply.md       ★ 先生への返答（Q3: 1/2係数問題）
├── half_factor_math_explanation.md    1/2不要の数学的証明
├── half_factor_literature_code_check.md  MATLAB vs Python 照合表
├── half_factor_revision_for_manuscript.md  原稿の1/2記述修正案
├── teacher_technical_questions_impl_check.md  実装確認メモ
├── parameter_estimation_*.md          パラメータ推定式の詳細メモ群（複数ファイル）
├── revised_4_2_text.md                4.2節改訂案
├── revised_formula_policy_for_discussion.md  式の方針メモ
│
├── paper_writing_examples/            他学会予稿のPDF参考例（内容未確認）
├── notion_v17_new.py                  Notion連携スクリプト（研究とは無関係）
└── archive/                           旧Notionスクリプト群（無関係）
```

---

## 2. 研究理解に必須のファイル

| 優先度 | ファイル | 理由 |
|--------|---------|------|
| ★★★ | `CLAUDE.md`（root） | 確定事項・過去の誤り・先生対応状況。最重要 |
| ★★★ | `conference_submission_final_draft.md` | 完成版予稿。式・実験条件・数値の一次ソース |
| ★★★ | `expfam/src/model_dual_expfam.py` | 提案手法の核心実装 |
| ★★★ | `expfam/src/model_expfam.py` | Y側 ExpFam 実装（1/2係数問題の所在） |
| ★★ | `expfam/references/baseline_metrics.md` | ベースライン数値・シナリオ別比較表 |
| ★★ | `Mato Lab Program/calcEtaNewton.m` | MATLAB精度行列の実装（1/2なしの根拠） |
| ★★ | `expfam/results/log_scenario_{A,B,C}.txt` | 実験の実行ログ（数値の裏付け） |
| ★★ | `expfam/results/exp_scenario_{A,B,C}_exp2_n.csv` | n-sweep 結果CSV |
| ★★ | `expfam/results/exp_scenario_{A,B,C}_exp4_mismatch.csv` | 誤指定実験 CSV |
| ★ | `expfam/src/utils_expfam.py` | BIC・RMSE計算の実装 |
| ★ | `expfam/src/exp_scenario_lib.py` | 実験条件の定義（N=150, D=15, K*=3, 10試行, L=5, 8iter） |
| ★ | `docs/teacher/half_factor_teacher_reply.md` | Q3回答（1/2問題の結論） |
| ★ | `docs/teacher/teacher_reply_draft.md` | Q1/Q2/Q4回答 |

---

## 3. 既存研究に関するファイル

### 先行研究（Mikawa et al., NOLTA 2024）

| ファイル | 内容 |
|---------|------|
| `paper/A_study_on_latent_structural_models_for_binary_rel.pdf` | 先行研究本体PDF（読み込み不可の可能性あり） |
| `Mato Lab Program/calcEtaNewton.m` | MATLAB E-step 実装（calcAi 関数含む） |
| `Mato Lab Program/calcw.m`, `calcw0.m` | MATLAB M-step 実装 |
| `Mato Lab Program/NOLTA_exp_ver3_revise_batch.m` | MATLAB 実験スクリプト |
| `Mato Lab Program/NOLTA_exp_ver5 (1).pdf` | MATLAB実験のPDF（内容未確認） |

### 先行研究のPython再現実装

| ファイル | 内容 |
|---------|------|
| `reproduction/src/model.py` | Python再現モデル（DualExpFamLSM の祖先クラス） |
| `reproduction/src/data_generator.py` | データ生成（normalize_zscore を expfam が流用） |
| `reproduction/src/experiment_paper_1.py` | 論文実験1再現（k変化） |
| `reproduction/src/experiment_paper_2.py` | 論文実験2再現（BIC） |
| `reproduction/src/experiment_paper_3.py` | 論文実験3再現（n, d変化） |
| `reproduction/src/experiment_compare_with_dual.py` | Control比較実験 |
| `reproduction/results/comparison/comparison_main_table.csv` | 先行研究との比較表CSV |
| `reproduction/reports/FINAL_REPRODUCTION_REPORT.md` | 再現実験の総括レポート |
| `expfam/references/baseline_metrics.md` | 先行研究論文値 + 再現実装値を対照した表 |

---

## 4. 提案手法に関するファイル

### 生成モデル（CLAUDE.md確定）

```
z_i  ~ N(0, I_k)
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )   i < j
x_il ~ ExpFam_X( η_il^X = f_l^T z_i )               バイアスなし
```

- パラメータ: θ = { F, w_0^Y, w^Y }（Gaussian-X のとき対角Σも）
- σ_z^2 = 1 固定（識別性確保、先行研究と同様）

### 精度行列（論文式、CLAUDE.md確定）

```
A_i = I_k + F^T V_X(m_i) F + (w^Y)^2 Σ_{j≠i} A_Y''(η_ij^Y) z_j z_j^T
```

**1/2 は不要（論文 Eq.(6) に 1/2 なし）。MATLABも 1/2 なし。**

### 主要実装ファイル

| ファイル | 役割 | 備考 |
|---------|------|------|
| `expfam/src/model_dual_expfam.py` | 提案手法 DualExpFamLSM | 継承: ExpFamLatentStructuralModel |
| `expfam/src/model_expfam.py` | Y側 ExpFam 基底 | 継承: reproduction/src/model.py の LatentStructuralModel |
| `expfam/src/utils_expfam.py` | Q関数・BIC・RMSE・Procrustes・EM実行 | run_em_dual が中心 |
| `expfam/src/data_generator_expfam.py` | 人工データ生成 | Gaussian X: z-score正規化あり |

---

## 5. 実験コードに関するファイル

### 実験設定（exp_scenario_lib.py L.40-45 から確認）

| 設定項目 | 値 |
|---------|-----|
| n (オブジェクト数) | 150（sweep: 50〜300） |
| d (属性次元数) | 15（sweep: 5〜30） |
| k* (真の潜在次元) | 3 |
| 試行数 | 10 |
| MCサンプル数 L | 5 |
| EMイテレーション数 | 8 |
| k探索範囲 | [1, 2, 3, 4, 5, 6] |

### シナリオ定義

| シナリオ | ExpFam_X | ExpFam_Y | 略称 |
|---------|---------|---------|------|
| A | Poisson | Bernoulli | P-B |
| B | Gaussian | Poisson | G-P |
| C | Bernoulli | Gaussian | B-G |

### 実験スクリプト

| ファイル | 内容 |
|---------|------|
| `exp_scenario_lib.py` | 実験関数の共通実装（exp1〜exp4） |
| `exp_run_scenario_A.py` | シナリオA（P-B）実行 |
| `exp_run_scenario_B.py` | シナリオB（G-P）実行 |
| `exp_run_scenario_C.py` | シナリオC（B-G）実行 |
| `run_exp1_full_metrics.py` | 補足実験（全メトリクス） |
| `run_exp2_bic_v2.py` | BIC詳細実験 |
| `run_wine_dual.py` | Wineデータ実データ実験（補足） |
| `test_dual_expfam.py` | ユニットテスト（5つ全PASS済み） |

### 実験の種類

| 実験 | 内容 |
|-----|------|
| Exp1 (k変化) | k=1〜6 で RMSE(Z) を評価。BIC で k* 選択を確認 |
| Exp2 (n変化) | n=50〜300 で RMSE(Z) の推移を確認 |
| Exp3 (d変化) | d=5〜30 で RMSE(Z) の推移を確認 |
| Exp4 (mismatch) | 誤った family 指定時の RMSE(Z) 悪化を評価 |

---

## 6. 実験結果に関するファイル

### 現行結果（提案手法、参照すべきファイル）

| ファイル | 内容 |
|---------|------|
| `expfam/results/exp_scenario_A_exp1_k.csv` | シナリオA、k変化、試行別raw data |
| `expfam/results/exp_scenario_A_exp2_n.csv` | シナリオA、n変化、試行別raw data |
| `expfam/results/exp_scenario_A_exp3_d.csv` | シナリオA、d変化、試行別raw data |
| `expfam/results/exp_scenario_A_exp4_mismatch.csv` | シナリオA、誤指定実験、試行別raw data |
| （B・Cも同様） | 各シナリオ同形式 |
| `expfam/results/exp1_full_{A,B,C}.csv` | BIC付き全メトリクス（論文表1の元データ） |
| `expfam/results/exp2_bic_{A,B,C}.csv` | BIC選択実験 |
| `expfam/results/log_scenario_A.txt` | シナリオA の実行ログ（収束状況） |
| `expfam/results/log_scenario_B.txt` | シナリオB の実行ログ |
| `expfam/results/log_scenario_C.txt` | シナリオC の実行ログ |

### 論文掲載の主要数値（conference_submission_final_draft.md から）

**表1: RMSE(Z) 平均値（k変化, n=150, d=15, 10試行）**

| k | Scen. A (P-B) | Scen. B (G-P) | Scen. C (B-G) |
|---|-------------|-------------|-------------|
| 1 | 0.953 | 1.063 | 0.998 |
| 2 | 0.766 | 0.651 | 0.576 |
| **3** | **0.278** | **0.182** | **0.028** |
| 4 | 0.505 | 0.436 | 0.299 |
| 5 | 0.707 | 0.538 | 0.388 |
| 6 | 0.692 | 0.603 | 0.418 |
| BIC最小 k | 3 ✓ | 3 ✓ | 3 ✓ |

**n-sweep RMSE(Z) 削減率（n=50→300）**

| シナリオ | n=50 | n=300 | 削減率 |
|---------|------|-------|------|
| A (P-B) | 0.406 | 0.208 | 48.8% |
| B (G-P) | 0.190 | 0.131 | 31.0% |
| C (B-G) | 0.053 | 0.020 | 62.0% |

**誤指定（mismatch）最大悪化倍率**

| シナリオ | 最大悪化倍率 |
|---------|-----------|
| A | 3.41× |
| B | 7.35× |
| C | 41.5× |

**表2: 先行研究との比較（Control条件, n=150, 5試行）**

| モデル | RMSE(Z) |
|-------|--------|
| 先行研究[1] | 0.179 ± 0.014 |
| 提案手法 | 0.180 ± 0.016 |

差は 0.001 未満（実質同等）。

### 論文提出用図

| ファイル | 内容 |
|---------|------|
| `figures/fig1a_n_sweep_color.pdf/png` | 図1: n-RMSE sweep（提出用最終版） |
| `figures/fig1b_misspecification_color.pdf/png` | 図2: 分布族誤指定（提出用最終版） |

---

## 7. 原稿・発表・先生対応に関するファイル

### 原稿

| ファイル | 内容 |
|---------|------|
| `conference_submission_final_draft.md` | 学会予稿完成版（Markdown形式、約120行） |

### 先生対応（CLAUDE.md の先生指摘と対応状況）

| ファイル | 対応内容 |
|---------|---------|
| `docs/teacher/teacher_reply_draft.md` | Q1（指数族はスカラーか）, Q2（Xはper-componentか）, Q4（ΣはΘか）の返答案 |
| `docs/teacher/half_factor_teacher_reply.md` | Q3（精度行列に1/2不要）の返答案（正しい版） |
| `docs/math_notes/half_factor_math_explanation.md` | Q3の数学的証明 |
| `docs/math_notes/half_factor_literature_code_check.md` | MATLAB vs Python の照合表 |
| `docs/math_notes/half_factor_revision_for_manuscript.md` | 原稿の式修正案 |
| `docs/teacher/teacher_technical_questions_impl_check.md` | 実装確認メモ |
| `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` | パラメータ推定式（修正版） |
| `docs/math_notes/parameter_estimation_equation_check.md` | 式の確認メモ |
| `docs/writing/parameter_estimation_revised_text_long.md` | 改訂テキスト（長版） |
| `docs/writing/parameter_estimation_revised_text_short.md` | 改訂テキスト（短版） |
| `docs/math_notes/parameter_estimation_risk_notes.md` | リスク注記 |
| `docs/writing/revised_4_2_text.md` | 4.2節の改訂案 |
| `docs/math_notes/revised_formula_policy_for_discussion.md` | 式方針のメモ |

---

## 8. 後回しにしてよいファイル

| ファイル | 理由 |
|---------|------|
| `expfam/results/wine_dual_results.csv`, `wine_F.npy` | 実データ実験（本論文には含まれない） |
| `expfam/src/run_wine_dual.py` | 同上 |
| `expfam/src/run_exp1_full_metrics.py` | 補足実験スクリプト |
| `expfam/src/run_exp2_bic_v2.py` | 補足実験スクリプト |
| `expfam/src/test_dual_expfam.py` | ユニットテスト（PASS済みで変更なし） |
| `expfam/data/ml-100k.zip` | MovieLens実データ（本研究では使用していない） |
| `expfam/design/01_exponential_family_design.md` | 旧設計メモ（実装済みにつき参考のみ） |
| `expfam/design/GEMINI_PROMPT.md` | Geminiへのプロンプト（研究内容ではない） |
| `expfam/results/GEMINI_REPORT_*.md` | AI生成レポート（研究者による検証未完了） |
| `expfam/results/RESEARCH_PROPOSAL_DUAL_EXPFAM.md` | 旧設計書（実装済みにつき参考のみ） |
| `expfam/results/fig1_rmse_vs_n.*`, `fig2_*.* ` | `figures/` の提出用図と重複の可能性あり（旧版の可能性） |
| `archive/paper_writing_examples/*.pdf` | 他学会予稿の参考例 |
| `paper/2.pdf` | 内容未確認の論文PDF |
| `archive/misc/notion_v17_new.py` | Notion連携スクリプト（研究無関係） |
| `reproduction/src/experiment_paper_1.py` 等 | 再現実験は完了済み、参照程度 |

---

## 9. archive扱いでよさそうなファイル

| ディレクトリ / ファイル | 理由 |
|----------------------|------|
| `archive/` | 旧Notionスクリプト群（研究に無関係） |
| `expfam/src/archive/` | 旧実験スクリプト（exp_dual_*.py, experiment_*.py） |
| `expfam/results/archive/` | 旧実験結果（GEMINI_REPORT_STEP*.md 含む） |

---

## 10. 注意すべき矛盾・未確認事項

### [最重要] 1/2係数問題（精度行列 term3）

| 観点 | 状況 |
|------|------|
| **論文 Eq.(6)** | `(w^Y)^2 Σ_{j≠i} A_Y''(...) z_j z_j^T`  →  **1/2 なし** |
| **MATLAB calcAi**（L.62） | `Ai = ... + FSF + tmp` ただし `tmp = w^2 * Z' * sig * Z - ...` → **1/2 なし** |
| **Python model_expfam.py (L.135)** | `term3 = 0.5 * (w**2) * (Z.T @ diag(var_fn) @ Z)` → **0.5 あり**（spurious） |
| **Python calc_w0 gradient (L.168)** | `grad = -grad_sum / (2.0 * L * phi)` → **0.5 あり** |
| **Python calc_w gradient (L.200)** | `grad = -grad_sum / (2.0 * L * phi)` → **0.5 あり** |
| **model_dual_expfam.py docstring** | Term3 precision を `w^2/2phi_Y` と記述 → **0.5 を正規化項として明示** |
| **CLAUDE.md の結論** | gradient と precision 両方に 0.5 があるため Newton 方向は正しいが、**Laplace近似のサンプリング分散が 2× 膨らむ**。論文の式は正しい。修正は修論フェーズ。 |

⚠ **ポイント**: `model_dual_expfam.py` の docstring は「`w^2/2phi_Y`」と書いており、Pythonコードの0.5を正当化するように見えるが、CLAUDE.md の確定事項では「spurious」と明記。docstring と CLAUDE.md が矛盾している。

### [要注意] expfam/CLAUDE.md は旧セッション用

`expfam/CLAUDE.md` は別セッション（Gemini経由）で作成されたもので「次のClaudeの役割 = 厳格な学術レビュアー」という旧設定が残っている。現在の確定事項は root の `CLAUDE.md` が正しい。

### [未確認] シナリオC の X-mismatch が軽微な理由

- Poisson-X 誤指定（≈0.99×）と No-X ablation（≈1.00×）が提案手法と同等
- `expfam/references/baseline_metrics.md` に記載: 「Y=Gaussianが支配的でX情報が無視されているのか、X側実装バグでX情報が実際には使われていないのかを区別すべき」
- **現時点では未解決**

### [未確認] fig1_rmse_vs_n / fig2_* の位置関係

- `expfam/results/` 配下にも `fig1_rmse_vs_n.*`, `fig2_heatmap_*.*, fig2_mismatch_*.*` がある
- `figures/` 配下には `fig1a_n_sweep_color.*`, `fig1b_misspecification_color.*` がある
- CLAUDE.md は `figures/` 配下を「提出用図」と指定。`expfam/results/` 配下は旧版の可能性大。

### [未確認] paper/2.pdf の内容

タイトル未確認。先行研究の別論文か別資料か不明。

### [未確認] GEMINI_REPORT_*.md の信頼性

Gemini（別AI）が生成したレポートであり、研究者による検証は未完了。NotebookLM への投入は慎重に。

### [未確認] wine_dual_results.csv の位置づけ

実データ（Wine）実験の結果だが、学会予稿には含まれていない。今後の実データ適用タスクの素材か。

### 旧パスの残存（要注意）

`expfam/handoff.md` には `C:/研究2/` というパスが残っている。現在の作業ディレクトリは `D:/tento/kennkyu/`。パスが変わっているため再実行時は注意。

---

## 11. 次に読むべきファイル順

NotebookLM に投入する資料を整備するにあたり、以下の順で内容を精読・整理することを推奨する。

### Phase 1（研究の全体像把握）
1. `CLAUDE.md`（root） — 確定事項・注意点の基準
2. `conference_submission_final_draft.md` — 完成版原稿（モデル・実験・数値の一次ソース）
3. `expfam/references/baseline_metrics.md` — 先行研究との比較・シナリオ別数値一覧

### Phase 2（提案手法の実装理解）
4. `expfam/src/model_dual_expfam.py` — 提案手法核心（クラス設計・E-step・M-step）
5. `expfam/src/model_expfam.py` — Y側拡張基底（1/2問題の所在 L.135, L.168, L.200）
6. `Mato Lab Program/calcEtaNewton.m` — MATLAB calcAi 関数（1/2なしの根拠）
7. `expfam/src/utils_expfam.py` — Q関数・BIC・RMSE・Procrustes

### Phase 3（実験条件と結果の検証）
8. `expfam/src/exp_scenario_lib.py` — 実験設定定数の確認（N, D, K*, 試行数等）
9. `expfam/results/exp1_full_{A,B,C}.csv` — 論文表1の元データ
10. `expfam/results/exp_scenario_{A,B,C}_exp4_mismatch.csv` — 誤指定実験の元データ
11. `expfam/results/log_scenario_{A,B,C}.txt` — 実行ログ（収束状況）

### Phase 4（先生対応・数学的証明）
12. `docs/teacher/half_factor_teacher_reply.md` — Q3回答（1/2問題の最終結論）
13. `docs/math_notes/half_factor_math_explanation.md` — 1/2不要の数学的証明
14. `docs/teacher/teacher_reply_draft.md` — Q1/Q2/Q4回答

### Phase 5（先行研究の理解）
15. `reproduction/src/model.py` — ベースライン実装
16. `reproduction/reports/FINAL_REPRODUCTION_REPORT.md` — 再現実験の総括
