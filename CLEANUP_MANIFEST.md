# CLEANUP_MANIFEST.md

整理候補の一覧。**今回は移動・削除・リネームは一切実施しない。**
本ドキュメントは将来の整理作業のための候補リストであり、各行の「実施可否」は次回以降の判断材料とする。

## 分類

- `keep`：研究の正本・現行成果物。そのまま維持
- `keep_but_warn`：維持するが、混同・誤用のリスクがあるため注意書きが必要
- `support`：補助資料として維持（本文には未採用）
- `archive_candidate`：将来archiveフォルダへ移動する候補
- `ignore_candidate`：研究内容と無関係。.gitignore等で除外する候補
- `delete_candidate_later`：将来的に削除を検討する候補（今は削除しない）

---

| パス | 分類 | 推奨操作 | 理由 | リスク | 実施可否 |
|----|----|------|----|-----|------|
| `CLAUDE.md`（root） | keep | 維持 | 確定式・記号・残タスクの正本 | なし | 実施不要（現状維持） |
| `expfam/CLAUDE.md` | keep_but_warn | 維持＋注意書き追加を検討 | 旧Geminiセッション向け。`expfam/README.md`自身が「正しい確定事項はroot CLAUDE.mdを参照」と明記済み（KI-008） | 古い前提（1/2必要等）を読んで誤った作業をするリスク | 注意書き追加は別途検討。今回は未実施 |
| `expfam/results/GEMINI_REPORT_MULTI_SCENARIO.md`, `GEMINI_REPORT_PHASE2_FINAL.md` | support | 維持（AI生成・未検証と明記） | `expfam/README.md`にも「AI生成レポート（参考のみ、未検証）」と既に記載あり（KI-007） | 数値を一次根拠として誤引用するリスク | 現状維持。EXPERIMENT_REGISTRY.mdで`ai_generated`と明記済み |
| `expfam/results/archive/` | archive_candidate | 現状維持（既にarchive名） | 初期シングルシナリオ実装の結果群。現行シナリオA/B/C構成とは異なる | 古い結果を現行結果と誤認するリスク | 既にarchiveとして分離済み。追加整理は不要 |
| `expfam/src/archive/` | archive_candidate | 現状維持（既にarchive名） | 初期実験スクリプト群（`experiment_poc_*`, `experiment_synthetic_*`等） | 現行の`exp_run_scenario_*.py`と混同するリスク | 既にarchiveとして分離済み |
| `archive/notion_scripts/`（notion_v3〜v16） | archive_candidate | 将来統合・整理候補 | Notion投稿用スクリプトの多数のバージョンが残存。研究本体と無関係（KI-009） | 誤って研究ロジックの一部と誤認するリスクは低いが、容量・視認性の問題 | 統合・削除は将来検討。今回は未実施 |
| `archive/misc/`（notion_report.py, notion_update_v2.py, wait_and_update_notion.py, katex_issues.txt, sec2_v15.py, sec2_v16.py, notion_v17_new.py） | archive_candidate | 将来統合・整理候補 | Notion関連・原稿執筆メモ。研究本体と無関係（KI-009） | 同上 | 将来検討。今回は未実施 |
| `archive/paper_writing_examples/`（403.pdf, 406.pdf, E11.pdf） | ignore_candidate | 現状維持 | 論文執筆の参考PDF。研究データではない | 低 | 維持で問題なし |
| `__pycache__/`（`reproduction/src/__pycache__/`） | ignore_candidate | `.gitignore`登録候補 | Pythonのコンパイル済みキャッシュ（`model.cpython-313.pyc`等）。ソース管理対象外であるべき | 低（既にgit追跡対象でなければ無害） | `.gitignore`への追記は将来検討。今回は未実施 |
| `expfam/figures/distribution_mismatch_fixed/comparison_old_vs_fixed.png/pdf` | keep_but_warn | 維持＋「生成元未確認」の注記が必要 | 生成スクリプトが`expfam/src/*.py`から見つからない（KI-004） | 再現性のない図を本文・報告書で引用してしまうリスク | 注記追加は別途検討。削除・移動は不可（再生成不能のため） |
| `expfam/results/distribution_mismatch_fixed/`（mismatch_fixed_summary.csv, mismatch_fixed_all_trials.csv, comparison_quick.csv, run_log.txt） | support | 維持（fixed版補助実験として明記済み） | `EXPERIMENT_REGISTRY.md`で`fixed_support`と分類済み | 旧版結果と混同するリスク（KI-002, KI-003） | 現状維持 |
| `expfam/figures/distribution_mismatch_fixed/`（56ファイル：heatmap, boxplot, bar_*, comparison_*） | support | 維持 | fixed版補助実験の図一式 | 同上 | 現状維持 |
| Wine関連（`expfam/src/run_wine_dual.py`, `expfam/results/wine_dual_results.csv`, `expfam/results/wine_F.npy`, `reproduction/results/results_real_wine.csv`） | support | 維持（未評価と明記） | KI-006。将来課題として位置づけ | 未評価の結果を本文に流用するリスク | 現状維持。検証完了後に`keep`へ変更検討 |
| `expfam/results/fig_scenario_{A,B,C}_*.pdf/png`, `fig1_rmse_vs_n.*`, `fig2_*.*`（旧版図） | keep_but_warn | 維持＋「旧版」の注記が既に`expfam/README.md`にある | 提出用は`figures/`配下の`fig1a_*`, `fig1b_*`。`expfam/README.md` L.129-130で既に注記済み | 旧版図を提出用と誤認するリスク | 現状維持。`EXPERIMENT_REGISTRY.md`で`old`と明記済み |
| `figures/fig1a_n_sweep_color.*`, `figures/fig1b_misspecification_color.*` | keep | 維持 | 提出用最終図 | なし | 維持 |
| `figures/figure_color_split_report.md` | keep | 維持 | 提出用図の説明資料 | なし | 維持 |
| `docs_for_notebooklm/*` | support | 維持 | NotebookLM向け資料。一部AI生成の調査結果を含む | 数値を一次根拠として誤引用するリスク（KI-007） | 現状維持 |
| `reports/*`（claims_and_evidence.md, chatgpt_handoff_report.md, work_log.md 等） | keep | 維持 | 主張と根拠の対応表、作業ログ等 | なし | 維持 |
| `docs/math_notes/`, `docs/teacher/`, `docs/presentation/`, `docs/writing/` | keep | 維持 | 0.5係数問題の証明・先生への返答案・原稿執筆素材 | なし | 維持 |
| `reproduction/`（先行研究再現実装一式） | keep | 維持 | Control比較の根拠（`comparison/comparison_main_table.csv`等） | なし | 維持 |
| `Mato Lab Program/`（MATLAB原実装） | keep | 維持 | 1/2不要の根拠（calcEtaNewton.m） | なし | 維持 |
| `paper/`（PDF） | keep | 維持 | 先行研究論文PDF（読み込み不可だが原典として保持） | なし | 維持 |
