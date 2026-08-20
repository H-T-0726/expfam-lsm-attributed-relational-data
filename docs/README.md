# docs/ — 補助資料フォルダ

このフォルダは研究の補助資料を整理したフォルダ。
**研究本体（実装・実験・結果）は `expfam/` にあり、このフォルダは補助的位置づけ。**

---

## フォルダ構成

| フォルダ | 内容 |
|---------|------|
| `teacher/` | 先生への返答案・技術的質問への実装確認メモ |
| `math_notes/` | 数式確認メモ（精度行列の 1/2 問題の証明・一次確認・照合表等） |
| `math_notes/legacy/` | 現在の確定事項と矛盾する旧資料（削除せず保管） |
| `writing/` | 原稿作成メモ（改訂テキスト・節改訂案等） |
| `presentation/` | ゼミ発表・Notion 投稿用の資料一式（8ファイル） |
| `theory_audit/` | 理論監査の実施に用いたマスタープロンプト（来歴記録） |

---

## teacher/ 主要ファイル

| ファイル | 内容 | 優先度 |
|---------|------|:------:|
| `teacher_reply_draft.md` | 先生への返答案（Q1/Q2/Q4）。**作成時点の historical document** | ★★ |
| `half_factor_teacher_reply.md` | 先生への返答案（Q3: 精度行列の 1/2 問題）。**作成時点の historical document** | ★★ |
| `teacher_technical_questions_impl_check.md` | 実装確認メモ | ★ |

> ⚠ 上記の返答案2件は**作成時点の記録**であり、原論文 PDF の直接確認より前に書かれている。
> 「MATLAB 原典にも 1/2 がない」といった記述は MATLAB **実装**についての言及であって、
> **原論文の印刷式には 1/2 がある**（2026-08-18 一次確認）。
> **1/2 の現在の正は `RESEARCH_MASTER.md` §6.1 と
> `math_notes/half_factor_primary_source_confirmation_20260818.md`。**
> 先生対応 Q1–Q4 の現在の整理は `RESEARCH_MASTER.md` §6b を参照。

---

## math_notes/ 主要ファイル

| ファイル | 内容 | 優先度 |
|---------|------|:------:|
| `half_factor_primary_source_confirmation_20260818.md` | **原論文の印刷式に 1/2 があることの一次確認ノート（2026-08-18）＋ 5系統整理。1/2 の現状はこれを正とする** | ★★★ |
| `half_factor_math_explanation.md` | **本研究の採用式で extra 1/2 が消えることの数学的導出**（精度行列・E-step 勾配） | ★★★ |
| `half_factor_literature_code_check.md` | MATLAB vs Python 照合表（**2026-05-08 時点の記録**。PDF 直接確認前のため「断定できない」と書かれている箇所がある。書き換えず historical record として読む） | ★★ |
| `half_factor_revision_for_manuscript.md` | 原稿の 1/2 記述修正案。**作成時点の historical document**（「MATLAB 原典・先生の指摘が全て 1/2 なしを支持」という記述は MATLAB **実装**についてのものであり、**原論文の印刷式には 1/2 がある**。現在の正は `half_factor_primary_source_confirmation_20260818.md` と `RESEARCH_MASTER.md` §6.1） | ★ |
| `parameter_estimation_equation_check.md` | パラメータ推定式の確認メモ | ★ |
| `parameter_estimation_risk_notes.md` | リスク注記 | ★ |
| `revised_formula_policy_for_discussion.md` | 式の方針メモ | ★ |

---

## math_notes/legacy/ の注意

`legacy/parameter_estimation_corrected_formulas.md` は、
E-step の 1/2 係数について **現在の確定事項（`CLAUDE.md` および原稿 Eq.(6)）と矛盾する旧資料**。
削除せず「中間段階の記録」として保管しているが、原稿採用式の根拠としては使わない。

正しい式の根拠: `half_factor_math_explanation.md`（本研究の採用式の導出）と
`half_factor_primary_source_confirmation_20260818.md`（原論文側の一次確認）。

---

## presentation/ ファイル

ゼミ発表・Notion 投稿用にまとめた資料。**canonical docs ではない**（研究内容の正本は
`RESEARCH_MASTER.md`、数値の provenance は `EXPERIMENT_REGISTRY.md`）。

| ファイル | 内容 |
|---------|------|
| `seminar_notion_structure.md` | 発表構成の設計 |
| `seminar_notion_draft_part1.md` 〜 `part4.md` | Notion 投稿用ドラフト（分割版） |
| `seminar_notion_full.md` | 同・統合版 |
| `seminar_notion_full_formula_rich.md` | 同・数式を厚くした版 |
| `seminar_formula_enhancement_plan.md` | 数式提示の強化計画 |

---

## theory_audit/ ファイル

| ファイル | 内容 |
|---------|------|
| `CLAUDE_FABLE_5_THEORY_AUDIT_MASTER_PROMPT.md` | 2026-07-18 の理論監査に用いたマスタープロンプト（来歴記録。2026-08-17 に root から移動） |

監査の**結果**は `reports/theory_audit/` にある（`docs/` 側は入力側の記録）。

---

## writing/ ファイル

| ファイル | 内容 |
|---------|------|
| `parameter_estimation_revised_text_long.md` | 改訂テキスト（長版） |
| `parameter_estimation_revised_text_short.md` | 改訂テキスト（短版） |
| `revised_4_2_text.md` | 4.2 節の改訂案 |
