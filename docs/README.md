# docs/ — 補助資料フォルダ

このフォルダは研究の補助資料を整理したフォルダ。
**研究本体（実装・実験・結果）は `expfam/` にあり、このフォルダは補助的位置づけ。**

---

## フォルダ構成

| フォルダ | 内容 |
|---------|------|
| `teacher/` | 先生への返答案・技術的質問への実装確認メモ |
| `math_notes/` | 数式確認メモ（精度行列の 1/2 問題の証明・照合表等） |
| `math_notes/legacy/` | 現在の確定事項と矛盾する旧資料（削除せず保管） |
| `writing/` | 原稿作成メモ（改訂テキスト・節改訂案等） |

---

## teacher/ 主要ファイル

| ファイル | 内容 | 優先度 |
|---------|------|:------:|
| `teacher_reply_draft.md` | 先生への返答案（Q1/Q2/Q4） | ★★★ |
| `half_factor_teacher_reply.md` | 先生への返答案（Q3: 精度行列の 1/2 問題） | ★★★ |
| `teacher_technical_questions_impl_check.md` | 実装確認メモ | ★ |

---

## math_notes/ 主要ファイル

| ファイル | 内容 | 優先度 |
|---------|------|:------:|
| `half_factor_math_explanation.md` | **1/2 不要の数学的証明**（精度行列・E-step 勾配） | ★★★ |
| `half_factor_literature_code_check.md` | MATLAB vs Python 照合表 | ★★ |
| `half_factor_revision_for_manuscript.md` | 原稿の 1/2 記述修正案 | ★ |
| `parameter_estimation_equation_check.md` | パラメータ推定式の確認メモ | ★ |
| `parameter_estimation_risk_notes.md` | リスク注記 | ★ |
| `revised_formula_policy_for_discussion.md` | 式の方針メモ | ★ |

---

## math_notes/legacy/ の注意

`legacy/parameter_estimation_corrected_formulas.md` は、
E-step の 1/2 係数について **現在の確定事項（`CLAUDE.md` および原稿 Eq.(6)）と矛盾する旧資料**。
削除せず「中間段階の記録」として保管しているが、原稿採用式の根拠としては使わない。

正しい式の根拠: `half_factor_math_explanation.md`（1/2 不要の数学的証明）

---

## writing/ ファイル

| ファイル | 内容 |
|---------|------|
| `parameter_estimation_revised_text_long.md` | 改訂テキスト（長版） |
| `parameter_estimation_revised_text_short.md` | 改訂テキスト（短版） |
| `revised_4_2_text.md` | 4.2 節の改訂案 |
