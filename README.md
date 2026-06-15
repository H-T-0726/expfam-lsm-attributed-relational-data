# Dual-ExpFam LSM 研究リポジトリ

属性データ X と関係データ Y の両方を指数型分布族へ一般化した
潜在構造モデル **Dual-ExpFam LSM** の実装・実験・検証・原稿作成を管理する研究リポジトリ。

ベースライン: Mikawa et al., "A study on latent structural models for binary relational data
with attribute information," NOLTA, IEICE, vol. 15, no. 2, 2024.

**メインは NotebookLM 資料ではなく、`expfam/` にある提案手法の実験コードと実験結果。**

---

## 1. 最初に読むファイル

| 目的 | 最初に読むファイル |
|-----|----------------|
| 実験を回したい | `expfam/README.md` → `expfam/src/exp_scenario_lib.py` |
| 研究全体を理解したい | `CLAUDE.md` → `docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md` |
| 原稿内容を確認したい | `conference_submission_final_draft.md` |
| 先生への返答を作りたい | `docs/teacher/teacher_reply_draft.md` / `docs/teacher/half_factor_teacher_reply.md` |
| NotebookLM に投入したい | `docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md` |
| 実験数値を照合したい | `docs_for_notebooklm/02_experiment_result_verification.md` |

---

## 2. ディレクトリ構成

| フォルダ / ファイル | 内容 | 優先度 |
|------------------|------|:------:|
| **`expfam/`** | **提案手法の実装・実験・結果（メイン）** | ★★★ |
| `reproduction/` | 先行研究 (NOLTA 2024) の Python 再現実装 | ★★ |
| `Mato Lab Program/` | 先行研究の MATLAB 実装（数式・係数確認用） | ★★ |
| `paper/` | 先行研究 PDF（NOLTA 2024） | ★★ |
| `figures/` | 提出用図の最終版（2026-05-07） | ★★ |
| `docs_for_notebooklm/` | NotebookLM 投入用資料・実験照合レポート | ★★ |
| `docs/` | 補助資料（先生対応・数式メモ・原稿メモ） | ★ |
| `archive/` | archive 済みファイル（研究と無関係・旧版） | — |
| **`CLAUDE.md`** | **確定事項・過去の誤り・注意点（必読）** | ★★★ |
| `conference_submission_final_draft.md` | 学会予稿完成版 | ★★★ |

---

## 3. メイン実験フォルダ

`expfam/` が本研究のメイン実験フォルダ。
詳細は **`expfam/README.md`** を参照。

```
expfam/
├── src/          実装・実験スクリプト（提案手法の核心）
├── results/      実験結果（CSV・ログ・図）
└── references/   先行研究との比較数値
```

クラス継承:

```
reproduction/src/model.py          ← 先行研究 Python 再現（ベースクラス）
└── expfam/src/model_expfam.py     ← Y 側 ExpFam 拡張
    └── expfam/src/model_dual_expfam.py  ← 提案手法本体（DualExpFamLSM）
```

---

## 4. 実験を回す場合

```bash
# 作業ディレクトリ
cd expfam/src

# シナリオ A: 真の X=Poisson, Y=Bernoulli
python exp_run_scenario_A.py

# シナリオ B: 真の X=Gaussian, Y=Poisson
python exp_run_scenario_B.py

# シナリオ C: 真の X=Bernoulli, Y=Gaussian
python exp_run_scenario_C.py

# ユニットテスト（5つ全 PASS を確認してから実験）
python test_dual_expfam.py
```

実験結果は `expfam/results/` に保存される。

---

## 5. 重要な注意事項

### 必ず読むこと

- **`CLAUDE.md`** — 確定事項・過去の誤り・精度行列の係数問題・先生対応状況
- **`expfam/README.md`** — 実験の実行方法・主要ファイルの説明・注意事項

### 実装上の注意（コード修正前に確認）

1. **Python path**: `reproduction/src/` を path に追加すること（`LatentStructuralModel` の継承元）
2. **1/2 係数問題**: E-step の Y 側 Term3 に spurious な `0.5` が残存している
   （`model_dual_expfam.py` L.159, L.200）。修論フェーズで修正予定。現時点では触らない。
3. **原稿採用式と Python 実装は異なる**: 精度行列 Term3 は原稿で `(w^Y)^2 Σ_{j≠i}` だが
   Python 実装は `0.5 * (w^Y)^2 Σ_{j≠i}`。詳細は `CLAUDE.md` 参照。

### 数式確認

- 精度行列の正しい式: `CLAUDE.md` の「精度行列（確定式）」節
- 1/2 問題の証明: `docs/math_notes/half_factor_math_explanation.md`
- MATLAB vs Python 照合: `docs/math_notes/half_factor_literature_code_check.md`

---

## 6. 現在の研究状況（2026-05-18 時点）

| 項目 | 状況 |
|-----|------|
| 提案手法の実装 | 完了（`expfam/src/`） |
| 実験 A/B/C の実行 | 完了（`expfam/results/`） |
| 学会予稿 | 完成（`conference_submission_final_draft.md`） |
| 提出用図 | 完成（`figures/`） |
| 先生への返答 | 案完成、未送付（`docs/teacher/teacher_reply_draft.md` 等） |
| Word 文書反映 | 未実施（.docx はリポジトリ外） |
| Python 実装の 1/2 修正 | 修論フェーズで対応予定 |

---

## 7. docs/ フォルダ構成（補助資料）

補助資料は `docs/` に整理済み（root への散乱を解消）。

| フォルダ | 内容 |
|---------|------|
| `docs/teacher/` | 先生対応メモ（返答案・実装確認） |
| `docs/math_notes/` | 数式確認メモ（1/2 問題証明・MATLAB照合等） |
| `docs/math_notes/legacy/` | 中間段階資料（現在の確定事項と矛盾する旧版） |
| `docs/writing/` | 原稿作成メモ（改訂テキスト等） |

### docs/teacher/ 主要ファイル

| ファイル | 内容 |
|---------|------|
| `teacher_reply_draft.md` | Q1/Q2/Q4 先生への返答案 |
| `half_factor_teacher_reply.md` | Q3 返答案（1/2 問題、正しい版） |
| `teacher_technical_questions_impl_check.md` | 実装確認メモ |

### docs/math_notes/ 主要ファイル

| ファイル | 内容 |
|---------|------|
| `half_factor_math_explanation.md` | 1/2 不要の数学的証明 |
| `half_factor_literature_code_check.md` | MATLAB vs Python 照合表 |
| `half_factor_revision_for_manuscript.md` | 原稿の 1/2 記述修正案 |
| `legacy/parameter_estimation_corrected_formulas.md` | ⚠ 中間段階の矛盾資料（原稿採用式の根拠としては使わない） |
