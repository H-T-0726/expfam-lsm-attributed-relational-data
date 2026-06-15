# リポジトリ整理計画

**作成日:** 2026-05-18  
**方針:** ファイル削除・移動・リネームなし。計画と案のみ。

---

## 1. リポジトリの主目的の明確化

**このリポジトリのメイン目的:**  
提案手法 **Dual-ExpFam LSM** の実装・実験を実行・確認・再現するための研究リポジトリ。

```
メイン:   expfam/                    提案手法の実装・実験本体
サブ1:   reproduction/ + Mato Lab Program/ + paper/  先行研究再現・確認
サブ2:   conference_submission_final_draft.md + figures/ + 先生対応メモ  提出・説明資料
サブ3:   docs_for_notebooklm/        NotebookLM 用資料
サブ4:   archive候補ファイル群        後回し・旧版・中間段階
```

---

## 2. 現状のディレクトリ構成とファイル分類

### 2-1. 現在のルートレベル（問題点：.md ファイルが散乱）

```
kennkyu/
├── CLAUDE.md                               ★ 必読（確定事項）
├── conference_submission_final_draft.md    ★ 学会予稿完成版
│
├── expfam/                                 ★ メイン実験フォルダ
├── reproduction/                           先行研究再現
├── Mato Lab Program/                       先行研究MATLABコード
├── paper/                                  先行研究PDF
├── figures/                                ★ 提出用図（最終版）
├── docs_for_notebooklm/                    NotebookLM用資料
├── archive/                                archive済み（旧Notionスクリプト等）
│
├── teacher_reply_draft.md                  ← 先生対応（要整理）
├── half_factor_teacher_reply.md            ← 先生対応（要整理）
├── teacher_technical_questions_impl_check.md  ← 先生対応（要整理）
├── half_factor_math_explanation.md         ← 数式メモ（要整理）
├── half_factor_literature_code_check.md    ← 数式メモ（要整理）
├── half_factor_revision_for_manuscript.md  ← 数式メモ（要整理）
├── parameter_estimation_corrected_formulas.md ← 数式メモ ⚠矛盾資料（要整理）
├── parameter_estimation_equation_check.md ← 数式メモ（要整理）
├── parameter_estimation_risk_notes.md     ← 数式メモ（要整理）
├── parameter_estimation_revised_text_long.md  ← 原稿メモ（要整理）
├── parameter_estimation_revised_text_short.md ← 原稿メモ（要整理）
├── revised_4_2_text.md                    ← 原稿メモ（要整理）
├── revised_formula_policy_for_discussion.md   ← 原稿メモ（要整理）
├── notion_v17_new.py                      ← 研究と無関係（archive候補）
└── paper_writing_examples/                ← 参考例（archive候補）
```

### 2-2. expfam/ の現状

```
expfam/
├── CLAUDE.md           ⚠ 旧Geminiセッション向け（archive候補）
├── handoff.md          ⚠ 旧パス C:/研究2/ 参照（archive候補）
├── design/             ⚠ 旧設計メモ（archive候補）
│   ├── 01_exponential_family_design.md
│   └── GEMINI_PROMPT.md
├── data/               実データ（ml-100k.zip）
├── references/
│   └── baseline_metrics.md   ★ 先行研究比較数値
├── src/                ★ 実装・実験スクリプト（現行）
│   ├── model_dual_expfam.py  ★ 提案手法核心
│   ├── model_expfam.py
│   ├── utils_expfam.py
│   ├── data_generator_expfam.py
│   ├── exp_scenario_lib.py   ★ 実験共通設定
│   ├── exp_run_scenario_A.py ★ 実験実行
│   ├── exp_run_scenario_B.py ★ 実験実行
│   ├── exp_run_scenario_C.py ★ 実験実行
│   ├── run_exp1_full_metrics.py  補足実験
│   ├── run_exp2_bic_v2.py        補足実験
│   ├── run_wine_dual.py          実データ実験
│   ├── test_dual_expfam.py       ユニットテスト
│   └── archive/              旧スクリプト（archive済み）
└── results/            ★ 実験結果
    ├── exp_scenario_{A,B,C}_exp{1-4}.csv  ★ 現行rawデータ
    ├── exp1_full_{A,B,C}.csv             ★ BIC付き全メトリクス
    ├── exp2_bic_{A,B,C}.csv
    ├── exp2_bic_log.txt
    ├── log_scenario_{A,B,C}.txt          ★ 実行ログ
    ├── fig_scenario_{A,B,C}_*.pdf/png    ★ シナリオ別図（古め）
    ├── fig1_rmse_vs_n.*                  旧提出図（figures/に最新版あり）
    ├── fig2_heatmap_*.*, fig2_mismatch_*.* 旧提出図
    ├── GEMINI_REPORT_MULTI_SCENARIO.md   ⚠ AI生成・未検証
    ├── GEMINI_REPORT_PHASE2_FINAL.md     ⚠ AI生成・未検証
    ├── RESEARCH_PROPOSAL_DUAL_EXPFAM.md  旧設計書（archive候補）
    ├── wine_dual_results.csv             実データ実験結果
    ├── wine_F.npy
    └── archive/                          旧結果（archive済み）
```

---

## 3. ファイル分類表（全体）

### カテゴリA: 研究の核心（常にアクセスが必要）

| ファイル | 役割 |
|---------|------|
| `CLAUDE.md`（root） | 確定事項・過去の誤り・方針。全セッションの基準 |
| `conference_submission_final_draft.md` | 学会予稿完成版 |
| `figures/fig1a_n_sweep_color.*` | 提出用図1（最終版 2026-05-07） |
| `figures/fig1b_misspecification_color.*` | 提出用図2（最終版 2026-05-07） |
| `expfam/src/model_dual_expfam.py` | 提案手法核心実装 |
| `expfam/src/model_expfam.py` | Y側ExpFam基底 |
| `expfam/src/utils_expfam.py` | EM実行・評価関数 |
| `expfam/src/exp_scenario_lib.py` | 実験共通設定 |
| `expfam/src/exp_run_scenario_{A,B,C}.py` | 実験実行スクリプト |
| `expfam/results/exp_scenario_*.csv` | 現行実験rawデータ |
| `expfam/results/log_scenario_*.txt` | 実行ログ |
| `expfam/references/baseline_metrics.md` | ベースライン比較数値 |
| `reproduction/src/model.py` | ベースラインモデル（継承元） |

### カテゴリB: 先行研究確認（必要時にアクセス）

| ファイル | 役割 |
|---------|------|
| `paper/A_study_on_latent_structural_models_for_binary_rel.pdf` | NOLTA 2024 本体 |
| `Mato Lab Program/calcEtaNewton.m` | MATLAB精度行列実装（1/2問題確認） |
| `Mato Lab Program/` その他.m | 先行研究MATLAB実装 |
| `reproduction/src/` | 先行研究Python再現 |
| `reproduction/results/` | 再現実験結果 |
| `reproduction/reports/FINAL_REPRODUCTION_REPORT.md` | 再現実験の総括 |

### カテゴリC: 先生対応（提出前・説明時にアクセス）

| ファイル | 優先度 | 役割 |
|---------|:------:|------|
| `teacher_reply_draft.md` | 高 | Q1/Q2/Q4 返答案 |
| `half_factor_teacher_reply.md` | 高 | Q3 返答案（1/2問題） |
| `half_factor_math_explanation.md` | 高 | 1/2不要の数学的証明 |
| `half_factor_literature_code_check.md` | 中 | MATLAB vs Python照合 |
| `teacher_technical_questions_impl_check.md` | 中 | 実装確認メモ |

### カテゴリD: 数式・原稿メモ（参照用・後で整理）

| ファイル | 内容 | 注意点 |
|---------|------|--------|
| `half_factor_revision_for_manuscript.md` | 1/2問題の原稿修正案 | |
| `parameter_estimation_equation_check.md` | 式確認メモ | |
| `parameter_estimation_risk_notes.md` | リスクメモ | |
| `parameter_estimation_revised_text_long.md` | 改訂テキスト（長） | |
| `parameter_estimation_revised_text_short.md` | 改訂テキスト（短） | |
| `revised_4_2_text.md` | 4.2節改訂案 | |
| `revised_formula_policy_for_discussion.md` | 式方針メモ | |
| `parameter_estimation_corrected_formulas.md` | 式集 | **⚠ E-stepの1/2がCLAUDE.mdと矛盾する中間段階資料** |

### カテゴリE: NotebookLM用（docs_for_notebooklm/）

| ファイル | 内容 |
|---------|------|
| `NOTEBOOKLM_RESEARCH_BRIEF.md` | 研究全体の説明資料（検証済み） |
| `00_repository_inventory.md` | リポジトリ構成整理 |
| `01_formula_code_audit.md` | 数式コード監査 |
| `02_experiment_result_verification.md` | 実験結果照合 |
| `03_figure_consistency_check.md` | 図整合性確認 |
| `04_notebooklm_final_validation.md` | 最終検証レポート |
| `06_repository_cleanup_plan.md` | 本ファイル |
| `REPORT_TO_CHATGPT.md` | ChatGPT報告用 |

### カテゴリF: archive候補（今後移動を検討）

| ファイル | 理由 |
|---------|------|
| `notion_v17_new.py` | 研究と無関係（Notion連携スクリプト） |
| `paper_writing_examples/` | 参考例のみ、本研究内容ではない |
| `expfam/CLAUDE.md` | 旧Geminiセッション向け（root CLAUDE.mdが正） |
| `expfam/handoff.md` | 旧パス `C:/研究2/` 参照、現在は機能しない |
| `expfam/design/` | 旧設計メモ（実装済みにつき不要） |
| `expfam/results/GEMINI_REPORT_*.md` | AI生成レポート（研究者検証未完了） |
| `expfam/results/RESEARCH_PROPOSAL_DUAL_EXPFAM.md` | 旧設計書（実装済みにつき不要） |
| `expfam/results/fig1_rmse_vs_n.*` | 提出直前の旧版図（`figures/`に最新版あり） |
| `expfam/results/fig2_heatmap_*.*, fig2_mismatch_*.*` | 同上 |
| `archive/` 配下 | 既にarchive（Notionスクリプト等） |
| `expfam/src/archive/` | 既にarchive（旧実験スクリプト） |
| `expfam/results/archive/` | 既にarchive（旧実験結果） |

---

## 4. 目的別ファイル読み順ガイド

### A. 実験を回したい Claude Code 向け

```
1. README.md                           ← 今後作成予定（クイックスタート）
2. CLAUDE.md                           ← 必読：確定事項・注意点
3. expfam/README.md                    ← 今後作成予定（expfam概要）
4. expfam/src/exp_scenario_lib.py      ← 実験設定（N, D, K*, L, etc.）
5. expfam/src/exp_run_scenario_A.py    ← 実験実行例
6. expfam/src/model_dual_expfam.py     ← 提案手法の実装
7. expfam/results/                     ← 現行実験結果の確認
```

**実行前の注意:**
- Python path: `reproduction/src/` を追加（`model.py` の継承元）
- 作業ディレクトリ: `expfam/src/` から実行
- E-step Term3 に 0.5 が残存（修論フェーズで修正予定）

### B. 研究内容を理解したい Claude Code 向け

```
1. CLAUDE.md                                        ← 確定事項の基準
2. docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md ← 研究の全体像
3. conference_submission_final_draft.md             ← 完成版原稿
4. docs_for_notebooklm/01_formula_code_audit.md    ← 数式とコードの整合
5. docs_for_notebooklm/02_experiment_result_verification.md ← 実験数値の照合
```

### C. 先生への説明・返答を作りたい Claude Code 向け

```
1. CLAUDE.md                            ← 先生指摘と対応状況
2. teacher_reply_draft.md               ← Q1/Q2/Q4 返答案
3. half_factor_teacher_reply.md         ← Q3 返答案（1/2問題）
4. half_factor_math_explanation.md      ← 1/2不要の証明
5. docs_for_notebooklm/01_formula_code_audit.md ← 詳細な証拠整理
```

### D. NotebookLM に投入したい場合

```
1. docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md  ← 主資料
2. docs_for_notebooklm/01_formula_code_audit.md
3. docs_for_notebooklm/02_experiment_result_verification.md
4. docs_for_notebooklm/03_figure_consistency_check.md
5. CLAUDE.md
6. conference_submission_final_draft.md
7. half_factor_math_explanation.md
8. half_factor_literature_code_check.md
9. expfam/references/baseline_metrics.md
```

**投入しない方がよいファイル:**
- `parameter_estimation_corrected_formulas.md`（確定事項と矛盾）
- `expfam/CLAUDE.md`（旧セッション向け）
- `expfam/results/GEMINI_REPORT_*.md`（AI生成・未検証）

---

## 5. 理想的な整理後ディレクトリ構成案

現在の構成を大きく変えずに、以下の方向を提案する。

```
kennkyu/
├── README.md                        ← 【今後作成】リポジトリ入口
├── CLAUDE.md                        ← 確定事項・注意点（既存）
├── conference_submission_final_draft.md  ← 学会予稿完成版（既存）
│
├── expfam/                          ← メイン（提案手法本体）
│   ├── README.md                    ← 【今後作成】expfam概要・実行方法
│   ├── src/                         ← 実装・実験スクリプト（現行）
│   ├── results/                     ← 実験結果（現行）
│   ├── references/                  ← ベースライン数値（現行）
│   └── data/                        ← 実データ（現行）
│
├── reproduction/                    ← 先行研究Python再現（現行）
│   ├── src/
│   ├── results/
│   ├── reports/
│   └── scripts/
│
├── Mato Lab Program/                ← 先行研究MATLABコード（現行）
├── paper/                           ← 先行研究PDF（現行）
├── figures/                         ← 提出用図・最終版（現行）
│
├── docs/                            ← 【今後作成】サブ資料フォルダ
│   ├── teacher/                     ← 先生対応メモ（移動候補）
│   │   ├── teacher_reply_draft.md
│   │   ├── half_factor_teacher_reply.md
│   │   └── teacher_technical_questions_impl_check.md
│   ├── math_notes/                  ← 数式確認メモ（移動候補）
│   │   ├── half_factor_math_explanation.md
│   │   ├── half_factor_literature_code_check.md
│   │   ├── half_factor_revision_for_manuscript.md
│   │   ├── parameter_estimation_equation_check.md
│   │   └── parameter_estimation_risk_notes.md
│   └── writing/                     ← 原稿作成メモ（移動候補）
│       ├── parameter_estimation_revised_text_long.md
│       ├── parameter_estimation_revised_text_short.md
│       ├── revised_4_2_text.md
│       └── revised_formula_policy_for_discussion.md
│
├── docs_for_notebooklm/             ← NotebookLM用資料（現行）
│
└── archive/                         ← archive（既存 + 追加候補）
    ├── notion_scripts/              ← 既存
    ├── misc/                        ← 既存（+ notion_v17_new.py）
    ├── paper_writing_examples/      ← 移動候補
    └── expfam_old/                  ← 移動候補（expfam/CLAUDE.md 等）
```

**注意:** `archive/` 内は git でトラックしない方がよいが、
削除前に内容を確認すること。特に `parameter_estimation_corrected_formulas.md` は
中間段階の矛盾資料だが「旧バージョンの記録」として残す価値もある。

---

## 6. 今後作成すべき README ドラフト案

### 6-1. README.md（ルート）案

```markdown
# Dual-ExpFam LSM — 研究リポジトリ

属性データ X と関係データ Y の両方を指数型分布族に一般化した
潜在構造モデル（Dual-ExpFam LSM）の実装・実験リポジトリ。

ベースライン: Mikawa et al. (NOLTA 2024) — Bernoulli-Y + Gaussian-X 固定

## 最初に読むファイル

| 目的 | ファイル |
|-----|---------|
| 全体像（必読） | `CLAUDE.md` |
| 実験を回す | `expfam/README.md` |
| 研究内容を理解する | `docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md` |
| 学会予稿を読む | `conference_submission_final_draft.md` |
| 先生への返答 | `teacher_reply_draft.md` + `half_factor_teacher_reply.md` |

## ディレクトリ構成

| フォルダ/ファイル | 内容 |
|---------|------|
| `expfam/` | **メイン** — 提案手法の実装・実験・結果 |
| `reproduction/` | 先行研究 (NOLTA 2024) の Python 再現 |
| `Mato Lab Program/` | 先行研究の MATLAB 実装（数式確認用） |
| `paper/` | 先行研究 PDF |
| `figures/` | 提出用図（最終版 2026-05-07） |
| `docs_for_notebooklm/` | NotebookLM 投入用・研究理解用資料 |
| `CLAUDE.md` | 確定事項・過去の誤り・先生対応状況（必読） |
| `conference_submission_final_draft.md` | 学会予稿完成版 |

## クイックスタート（実験実行）

# 環境: Python 3.13, numpy, scipy, matplotlib, pandas
cd expfam/src
python exp_run_scenario_A.py   # Scenario A [Poisson-X, Bernoulli-Y]
python exp_run_scenario_B.py   # Scenario B [Gaussian-X, Poisson-Y]
python exp_run_scenario_C.py   # Scenario C [Bernoulli-X, Gaussian-Y]
# 結果: expfam/results/ に保存

## 重要な注意事項

1. `CLAUDE.md` の「精度行列（確定式）」節を必ず読むこと
2. E-step の Y 側 Term3 に spurious な 0.5 が残存（修論フェーズで修正予定）
3. Python path に `reproduction/src/` が必要（モデルの継承元）
```

### 6-2. expfam/README.md 案

```markdown
# expfam — Dual-ExpFam LSM 本体

## 概要

提案手法 Dual-ExpFam LSM の実装・実験フォルダ。
X 側・Y 側ともに指数型分布族（Gaussian/Bernoulli/Poisson）を任意指定できる
属性情報付き潜在構造モデル。

## 実験シナリオ

| シナリオ | 真の ExpFam_X | 真の ExpFam_Y | 実行スクリプト |
|---------|------------|------------|-------------|
| A [P-B] | Poisson | Bernoulli | `src/exp_run_scenario_A.py` |
| B [G-P] | Gaussian | Poisson | `src/exp_run_scenario_B.py` |
| C [B-G] | Bernoulli | Gaussian | `src/exp_run_scenario_C.py` |

## 実験設定（共通）

n=150, d=15, k*=3, 10試行, L=5 MCサンプル, 8 EMイテレーション

## 主要ファイル（src/）

| ファイル | 役割 |
|---------|------|
| `model_dual_expfam.py` | **提案手法核心**（DualExpFamLSM クラス） |
| `model_expfam.py` | Y側ExpFam拡張基底クラス |
| `utils_expfam.py` | Q関数・BIC・RMSE・Procrustes・EM実行 |
| `data_generator_expfam.py` | 人工データ生成 |
| `exp_scenario_lib.py` | 実験共通設定・実験関数 |
| `exp_run_scenario_{A,B,C}.py` | 実験実行スクリプト |
| `test_dual_expfam.py` | ユニットテスト（5つ全PASS） |

## 実験結果（results/）

| ファイル | 内容 |
|---------|------|
| `exp_scenario_{A,B,C}_exp{1-4}.csv` | 現行実験の試行別rawデータ |
| `exp1_full_{A,B,C}.csv` | BIC付き全メトリクス |
| `log_scenario_{A,B,C}.txt` | 実行ログ |
| `fig_scenario_{A,B,C}_*.pdf/png` | シナリオ別図 |

## 注意事項

1. Python path に `../reproduction/src/` を追加すること（LatentStructuralModel の継承元）
2. E-step 精度行列・勾配の Y 側 Term3 に 0.5 が残存（修論フェーズで修正予定）
3. 提出用最終図は `../figures/` 配下（fig1a/fig1b）を使用する
4. `expfam/CLAUDE.md` は旧Geminiセッション向けのため使用しない（root CLAUDE.md が正）

## クラス継承構造

reproduction/src/model.py（LatentStructuralModel）
└── expfam/src/model_expfam.py（ExpFamLatentStructuralModel）
    └── expfam/src/model_dual_expfam.py（DualExpFamLSM）  ← 提案手法本体
```

---

## 7. archive 候補の優先度と判断基準

### 今すぐ archive してよいもの（依存なし）

| ファイル | 優先度 |
|---------|:------:|
| `notion_v17_new.py` | ★★★ |
| `archive/` 配下（既存） | 整理不要 |
| `expfam/CLAUDE.md` | ★★★ |
| `expfam/handoff.md` | ★★ |
| `expfam/design/` | ★★ |
| `expfam/results/GEMINI_REPORT_*.md` | ★★ |
| `expfam/results/RESEARCH_PROPOSAL_DUAL_EXPFAM.md` | ★★ |
| `paper_writing_examples/` | ★★ |

### 慎重に検討が必要なもの（内容確認後）

| ファイル | 理由 |
|---------|------|
| `parameter_estimation_corrected_formulas.md` | 矛盾資料だが旧バージョンの記録として価値あり |
| `expfam/results/fig1_rmse_vs_n.*` 等の旧版図 | figures/ との新旧比較に使う可能性あり |
| `expfam/results/wine_*` | 実データ実験。修論で使う可能性あり |

### archive しない方がよいもの（今は）

| ファイル | 理由 |
|---------|------|
| `expfam/data/ml-100k.zip` | 将来の実データ実験で必要な可能性 |
| `expfam/src/archive/` 配下 | 旧スクリプトだが参照価値あり |
| `expfam/results/archive/` 配下 | 旧結果だが経緯の記録として価値あり |

---

## 8. 実施すべき作業の優先順位

### Phase A（今すぐできる・影響小）

1. `README.md`（root）を作成 → リポジトリの入口を設ける
2. `expfam/README.md` を作成 → 実験の入口を設ける
3. `notion_v17_new.py` を `archive/` に移動

### Phase B（短期的・中程度の作業）

4. `docs/teacher/` フォルダを作成し、先生対応メモを移動
5. `docs/math_notes/` フォルダを作成し、数式メモを移動
6. `docs/writing/` フォルダを作成し、原稿メモを移動
7. `expfam/CLAUDE.md`、`expfam/handoff.md`、`expfam/design/` を archive に移動

### Phase C（修論フェーズで対応）

8. Python 実装の 1/2 係数修正（`model_dual_expfam.py` L.159, L.200）
9. `expfam/results/GEMINI_REPORT_*.md` の内容を検証してから archive または保存
10. 旧版図 (`expfam/results/fig1_rmse_vs_n.*` 等) の整理

---

## 9. CLAUDE.md に追記すべき内容（提案）

現在の CLAUDE.md はプロジェクト内容に集中しているが、
以下をルートの `README.md` または `CLAUDE.md` の「ファイル構成」節に追記することを提案する：

```markdown
## 読み順（目的別）

### 実験を回す
expfam/README.md → expfam/src/exp_scenario_lib.py → exp_run_scenario_A.py

### 研究を理解する
docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md → conference_submission_final_draft.md

### 先生への返答
teacher_reply_draft.md → half_factor_teacher_reply.md

### NotebookLM投入
docs_for_notebooklm/ 配下の推奨ファイル一覧を参照
```

---

## 10. 現時点での未解決・確認が必要な事項

| 項目 | 内容 |
|-----|------|
| `paper/2.pdf` の内容 | タイトル未確認。先行研究の別論文か不明 |
| `expfam/results/fig1_rmse_vs_n.*` と `figures/fig1a_*` の関係 | 旧版と最終版の確認済み（figures/ が最新） |
| `expfam/data/ml-100k.zip` の今後の使用予定 | 修論での実データ適用で必要か未定 |
| `expfam/results/wine_*` の位置づけ | 補足実験として保持するか archive するか |
