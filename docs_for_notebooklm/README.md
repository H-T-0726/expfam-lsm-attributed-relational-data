# docs_for_notebooklm/ — 派生資料フォルダ（canonical source ではない）

> ⚠ **このフォルダは研究の一次根拠ではありません。**
> 収録されている資料は、NotebookLM / ChatGPT などの外部 AI ツールへ投入するために作成された
> **派生資料**であり、**AI が生成・要約した内容を含みます。**

---

## 必ず守ること

1. **canonical source ではない。** 研究内容の正本は `RESEARCH_MASTER.md`、
   数値の provenance は `EXPERIMENT_REGISTRY.md`、主張の可否は `KNOWN_ISSUES.md`。
2. **AI 生成・派生資料を含む。** 研究者による検証が完了していない記述が混在している（KI-007）。
3. **数値・式・結論は必ず元へ遡って確認する。**
   - 数値 → 元の結果 CSV（`expfam/results/**`）と runinfo・実行ログ
   - 式 → 実コード（`expfam/src/**`）と `RESEARCH_MASTER.md` §6
   - 結論 → canonical docs（`RESEARCH_MASTER.md` / `KNOWN_ISSUES.md`）
4. **研究主張の一次証拠として単独で使用しない。** 原稿・報告書・スライドの根拠として
   このフォルダのファイルだけを引用してはならない。

---

## 内容の鮮度について

各ファイルは作成時点のリポジトリ状態を反映した**スナップショット**であり、
その後の実験フェーズ・監査結果を反映していない。**本文は更新しない方針**であり、
当時の投入資料の来歴記録として保持している。

したがって、現在の canonical docs と食い違う記述が残っていることがある。
食い違いがある場合は**常に canonical docs を正とする。**

既知の注意点の例:

- **1/2 係数の記述** — このフォルダには 1/2 の所在に関する記述が複数あるが、
  現在の整理は 5 系統（原論文の印刷式＝1/2 あり／old 0.5 Python＝あり／本研究の採用式＝なし／
  fixed Python＝なし／MATLAB `calcAi`＝なし）である。
  現在の正は `RESEARCH_MASTER.md` §6.1 と
  `docs/math_notes/half_factor_primary_source_confirmation_20260818.md`。
- **実験の網羅範囲** — 実データ実験フェーズ（Wine / Cora / MovieLens、2026-06-17〜07-07）以降の
  結果は反映されていないファイルがある。総括は `reports/real_data_experiment_summary.md`。

---

## 収録ファイル

| ファイル | 内容 |
|---|---|
| `NOTEBOOKLM_RESEARCH_BRIEF.md` | NotebookLM 投入用の研究ブリーフ |
| `00_repository_inventory.md` | リポジトリ全体のインベントリ（作成時点） |
| `01_formula_code_audit.md` | 数式とコードの対応監査（作成時点） |
| `02_experiment_result_verification.md` | 実験数値の照合（作成時点） |
| `03_figure_consistency_check.md` | 図の整合性チェック（作成時点） |
| `04_notebooklm_final_validation.md` | 投入前の最終チェック |
| `06_repository_cleanup_plan.md` | 整理計画（作成時点の案） |
| `REPORT_TO_CHATGPT.md` | ChatGPT 向けに作成した報告 |
| `REPORT_HISTORY.md` | 投入・報告の来歴 |

（`05_*` は存在しない。番号は作成順の名残であり欠番がある。）

---

## 関連

- 一次根拠を使うルール — `CLAUDE.md` §4（source priority）
- AI 生成レポートのリスク — `KNOWN_ISSUES.md` KI-007
