# START_HERE.md

## このリポジトリの目的

先行研究（Mikawa et al., 2024, NOLTA IEICE vol.15 no.2、Bernoulli-Y + Gaussian-X 固定の潜在構造モデル）を、
**指数型分布族（Gaussian / Bernoulli / Poisson）に一般化した潜在構造モデル（Dual-ExpFam LSM）**の研究。
学会予稿としての主要成果は `conference_submission_final_draft.md` にまとまっている。

---

## まず読むファイル（優先順位つき）

1. **START_HERE.md**（このファイル）— 全体の入口
2. **`CLAUDE.md`**（root）— 確定済みの数式・記号・過去に直した誤り・残タスクの正本
3. **`KNOWN_ISSUES.md`** — 既知の問題点・混同しやすい数値の一覧。作業前に必ず確認
4. **`RESEARCH_MASTER.md`** — 研究内容（目的・手法・実装対応表・主張の安全レベル）の正本
5. **`EXPERIMENT_REGISTRY.md`** — 実験結果ファイルの対応表（どのCSVがどの図・どの主張に対応するか）
6. **`conference_submission_final_draft.md`** — 提出予定の原稿本体
7. **`expfam/README.md`** — `expfam/`フォルダ（実装・実験の本体）の使い方
8. **`reports/claims_and_evidence.md`** — 個々の研究主張と根拠CSVの対応表

---

## 高信頼ファイル

- `CLAUDE.md`（root）— 確定式・記号・過去の誤り修正履歴の正本
- `conference_submission_final_draft.md` — 提出原稿
- `figures/fig1a_n_sweep_color.*`, `figures/fig1b_misspecification_color.*` — 提出用の最終図
- `expfam/src/model_dual_expfam.py`, `model_expfam.py`, `utils_expfam.py` — 提案手法の実装本体（コードは直接確認可能な一次情報）
- `expfam/results/exp_scenario_{A,B,C}_exp{1,2,3,4}*.csv` — 本文採用実験の結果CSV
- `reports/claims_and_evidence.md` — 主張と根拠CSVの対応表
- `docs/teacher/teacher_reply_draft.md`, `docs/teacher/half_factor_teacher_reply.md` — 先生への返答案（Q1/Q2/Q3/Q4対応）

## 低信頼・参考扱いファイル

- `expfam/CLAUDE.md` — 旧Geminiセッション向け。確定事項はroot `CLAUDE.md`を優先（KI-008）
- `expfam/results/GEMINI_REPORT_*.md`, `expfam/results/archive/GEMINI_REPORT_*.md` — AI生成・未検証（KI-007）
- `docs_for_notebooklm/*` — NotebookLM向け資料。一部はAI生成の調査結果を含むため、数値は元CSVで再確認すること
- `archive/notion_scripts/*`, `archive/misc/*`, `archive/paper_writing_examples/*` — 研究本体と無関係（KI-009）
- `expfam/results/wine_dual_results.csv`, `wine_F.npy` — 未評価（KI-006）
- `expfam/results/distribution_mismatch_fixed/*` — fixed版（0.5なし）の補助実験。本文には未採用（KI-002）

---

## 絶対に混同してはいけないこと

- **旧版とfixed版**：`model_dual_expfam.py`（0.5あり、本文採用）と`model_dual_expfam_fixed.py`（0.5なし、補助実験のみ）は異なる実装。結果を混在させない（KI-002）。
- **23.6倍 / 41.5倍 / 38.97倍**：いずれもScen.Cの誤指定実験の数値だが、出所が異なる3つの独立した値（KI-003）。
- **原稿式と現行Python実装**：原稿式（1/2なし）が正しいとされているが、現行Python実装の旧版には0.5が残っている（KI-001）。「実装が正しい」と早合点しない。
- **本文採用実験と補助実験**：Exp1-4（`exp_scenario_*`系）は本文採用。`distribution_mismatch_fixed/*`、Wine実験、`results/archive/*`は補助・未採用。

---

## Claude Codeに作業させる前のルール

- 数値主張は必ずCSV・実行ログ等の一次データに紐づける。AI生成レポート（GEMINI_REPORT_*等）を根拠にしない（KI-007）。
- コード修正・実験再実行・CSV再生成・ファイル移動・削除を行う前に、必ず目的とスコープをユーザーに確認する。
- `KNOWN_ISSUES.md`の「まだ主張してはいけないこと」に該当する内容を、報告書や原稿案に書かない。
- 既存の正本ファイル（root `CLAUDE.md`, `conference_submission_final_draft.md`, `README.md`）は、明示的な指示がない限り編集しない。
- 0.5係数問題（KI-001）に触れる際は、「Newton方向が正しいとは断定できない」という限定条件を必ず付記する。
