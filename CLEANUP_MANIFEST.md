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
- `review_required`：削除・archive・リネームいずれの判断も、実施前に人間（またはCodexへの確認タスク経由）の
  内容確認が必須のもの（2026-07-07追加。名称衝突・参照未確認・provenance不明などが該当）

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

### 実データ実験フェーズの追加分類（2026-07-07、`main`マージ後）

2026-06-17〜2026-07-07に追加され、`save-realdata-results-20260707`ブランチ経由で`main`にマージ済みの
Wine/Cora/MovieLens実データ実験一式。マージによりgit履歴の保護下に入ったため、大半を`keep`に分類する。

| パス | 分類 | 推奨操作 | 理由 | リスク | 実施可否 |
|----|----|------|----|-----|------|
| `expfam/src/run_fixed_official_*.py`, `run_half_factor_*.py`, `run_fixed_real_*.py`, `prepare_movielens_data.py`, `audit_wine_old05_vs_fixed.py`, `run_common_realdata_reconstruction_eval.py`, `summarize_*.py`（6ファイル） | keep | 維持 | 実データ実験フェーズの現行スクリプト一式。`main`にマージ済み | なし | 維持 |
| `expfam/results/fixed_official/**`, `expfam/results/half_factor_check/**` | keep | 維持 | fixed版での人工データ正式再実験・0.5係数問題の追加検証（KI-001関連） | なし | 維持 |
| `expfam/results/real_data/**`（`cora_subset_pilot`を除く）, `expfam/figures/real_data/**` | keep | 維持 | Wine/Cora/MovieLens実データ実験の一次結果・図。`EXPERIMENT_REGISTRY.md`に対応行を追加済み | なし | 維持 |
| `expfam/data/cora/`, `expfam/data/movielens_pilot/` | keep | 維持 | 実データ実験の入力データ。再生成にコスト（前処理・ダウンロード）がかかる | なし | 維持 |
| `reports/real_data_experiment_plan.md`, `reports/movielens_pilot_design.md`, `reports/real_data_experiment_summary.md`, `reports/movielens_colike_clean/movielens_colike_notion_summary.md` | keep | 維持 | 実データ実験の設計意図・結論の一次記録 | なし | 維持 |
| `expfam/results/real_data/cora_subset_pilot/`（+対応図） | archive_candidate | 将来archiveへ移動する候補（優先度高） | Cora BFSサブセットが1クラス78%偏りで不適切と判明し、`cora_balanced_subset_pilot/`以降に完全に置き換わっている（`reports/real_data_experiment_summary.md`に明記） | 低（既に不採用と文書化済み） | **archive移動前に本文・スライドからの参照ゼロを`grep`で確認すること（review_required寄り）** |
| `expfam/results/real_data/{wine_fixed_pilot, cora_balanced_subset_pilot, cora_balanced_k_sweep, movielens_poisson_pilot, movielens_bernoulli_t80_pilot, movielens_heldout_count, movielens_colike_interpretation}/` | support | 維持（pilot段階・`_clean`版のソースとして現役） | `_clean`/`_final_clean`版の入力元。監査スクリプト（`audit_wine_old05_vs_fixed.py`等）からも参照される | 中：`_clean`版が全指標を引き継いでいるか未確認のままarchiveすると根拠を失う | 現状維持。archive検討は`_clean`との指標突合後 |
| `expfam/results/real_data/movielens_colike_clean/` と `movielens_final_clean/` | review_required | 統合・注記追加を検討 | 名前が類似し役割が異なる（本文用3指標縮約 vs 監査用フル指標、KI-013） | 中：引用時の取り違えリスク | 統合・リネームは今回実施しない。`EXPERIMENT_REGISTRY.md`に役割差を明記済み |
| `reports/movielens_colike_clean/`（`expfam/results/real_data/movielens_colike_clean/`と同名ディレクトリ） | review_required | 名称衝突の解消方法を検討 | 同名だが前者はNotion用Markdown1本、後者はCSV+図一式で内容が異なる | 低（実害は今のところないが混乱を招く） | リネーム案の作成のみ次回検討 |
| リポジトリ直下 `tatus`（存在する場合） | review_required（削除候補寄り） | 中身確認後に削除を判断 | `git status`/`git log`出力の断片と見られる誤操作の残骸。研究内容とは無関係 | なし（内容確認前提） | 中身確認まで削除しない |

---

## 今後の安全なcleanup手順（2026-07-07追加）

ファイル移動・削除を伴わない「文書更新のみ」のステップを先に、実ファイル操作を伴うものは最後に回す。

### Step 1（最も安全・今回完了）: ドキュメント更新のみ
`EXPERIMENT_REGISTRY.md`, `RESEARCH_MASTER.md`, `START_HERE.md`, `KNOWN_ISSUES.md`, `CLEANUP_MANIFEST.md`（本ファイル）,
`CLAUDE.md` の更新で完了。ファイルの移動・削除・コード変更は一切行っていない。

### Step 2（安全・確認のみ、次にやるべき作業）
1. `cora_subset_pilot/` が原稿・スライド・レポートのどこからも参照されていないことを`grep`で確認する
   （参照ゼロを確認できて初めて`archive_candidate`から実施可能に格上げできる）。
2. `movielens_colike_clean/` と `movielens_final_clean/` の内容差分を突合し、
   「cleanに無くfinal_cleanにだけある指標」が本文・スライドで必要かどうかを確認する。
3. リポジトリ直下に `tatus` のような不審な未追跡ファイルが無いか `git status -uall` で確認する。

### Step 3（ここで初めてファイル操作を検討。今回は実施しない）
4. Step 2で「参照ゼロ」を確認できたものに限り、`archive/` への移動を提案・実施する。
5. 名称衝突（`movielens_colike_clean`等）のリネーム案を実施する（実施前に必ずユーザー承認を得る）。

---

## Codexへ渡す場合の安全な小タスク案（2026-07-07追加）

いずれも「読む・確認する・文書を追記する」レベルで、ファイルの移動・削除を伴わない：

1. `cora_subset_pilot/` への参照が原稿・レポート・スライド資料に一切ないことを`grep`で確認し、結果を報告する。
2. `movielens_colike_clean/` と `movielens_final_clean/` の各CSVの列・行数を比較し、差分表を作成する。
3. `expfam/results/fixed_official/`・`half_factor_check/`の実行結果を`KNOWN_ISSUES.md`のKI-001に追記する
   （0.5係数問題の検証にどう使われたかの一次資料として）。
4. `paper/2.pdf`のタイトル・著者だけを抽出し、`paper/`ディレクトリの役割を明確化する（要約・解釈は不要）。
5. `reports/movielens_colike_clean/`と`expfam/results/real_data/movielens_colike_clean/`の同名衝突について、
   リネーム案を1つ作成する（実施はしない、案のみ）。
6. リポジトリ直下の未追跡ファイル（`tatus`等）の中身を確認し、削除してよいか報告する（削除は実施しない）。
