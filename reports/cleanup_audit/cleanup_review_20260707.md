# Cleanup Review — 2026-07-07

**対象:** `reports/cleanup_audit/cleanup_candidates_20260707.csv`（753行、未コミット）
**目的:** 削除・移動の実行ではなく、人間が判断しやすいレビュー用サマリを作ること。
**実施内容:** 読み取り専用の調査のみ。ファイルの変更・削除・移動・リネームは一切行っていない。

CSV全体の内訳（Python `csv`モジュールで正規パース・確認済み）：

| category | 件数 |
|---|---:|
| KEEP | 597 |
| ARCHIVE_CANDIDATE | 131 |
| DELETE_CANDIDATE | 13 |
| REVIEW_REQUIRED | 12 |

---

## 1. DELETE_CANDIDATE（13件）

| path | reason | risk_level | recommended_action | 本当に低リスクか（コメント） |
|---|---|---|---|---|
| `expfam/src/__pycache__`（ディレクトリ） | Pythonバイトコードキャッシュ、研究データではない | low | 未追跡であることを確認後、人間が削除可 | **はい、真に低リスク。** `.gitignore`で`__pycache__/`が除外設定済み、`git ls-files`でも0件（未追跡）を確認済み。削除してもgit履歴・研究成果には一切影響しない |
| `expfam/src/__pycache__/data_generator_expfam.cpython-313.pyc` | コンパイル済みPythonバイトコード | low | 同上 | 同上。`.py`ソースが残っていれば実行時に自動再生成される |
| `expfam/src/__pycache__/model_dual_expfam.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `expfam/src/__pycache__/model_dual_expfam_fixed.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `expfam/src/__pycache__/model_expfam.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `expfam/src/__pycache__/run_common_realdata_reconstruction_eval.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `expfam/src/__pycache__/run_fixed_real_cora_scaling_heldout.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `expfam/src/__pycache__/run_fixed_real_movielens_colike_interpretation.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `expfam/src/__pycache__/run_half_factor_minimal_check.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `expfam/src/__pycache__/utils_expfam.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `reproduction/src/__pycache__`（ディレクトリ） | Pythonバイトコードキャッシュディレクトリ | low | 同上 | 同上 |
| `reproduction/src/__pycache__/data_generator.cpython-313.pyc` | 同上 | low | 同上 | 同上 |
| `reproduction/src/__pycache__/model.cpython-313.pyc` | 同上（`referenced_by`に`CLEANUP_MANIFEST.md`とあるが、これは文書内で「`__pycache__`は削除候補」と*言及*されているだけで、コードから読み込まれているという意味ではない） | low | 同上 | 同上。`referenced_by`列の意味を取り違えないよう注意（文書上の言及 ≠ 実行時の依存関係） |

**総評:** 13件すべて`__pycache__`配下のバイトコードキャッシュで、内容は完全に均質。`.gitignore`で除外設定済み・未追跡であることをコマンドで再確認したため、**このリストの中では最も自信を持って「低リスク」と言える分類**。ただし「削除・移動はしない」という今回の制約上、実際の削除はユーザー自身の操作に委ねる。

---

## 2. REVIEW_REQUIRED（12件）

12件はすべて**MovieLens co-like実験の名称衝突グループ**（`movielens_colike_clean/` 6件、`movielens_final_clean/` 5件、`reports/movielens_colike_clean/` 1件）で構成されており、他の種類のREVIEW_REQUIREDは存在しない。

| path | reason | 何と役割が衝突しているのか | 残す/統合/リネーム/保留の判断材料 |
|---|---|---|---|
| `expfam/results/real_data/movielens_colike_clean/claims_and_cautions.csv` | 類似名の結果フォルダとの混同リスク（CLEANUP_MANIFEST.md指摘） | `movielens_final_clean/`と字面上「どちらが最終版か」で混同されやすい | **保留＋現状維持を推奨。** 本ファイルは`summarize_movielens_colike_for_notion.py`のみが出力する「本文引用時の注意点」専用ファイルで、`final_clean`側に対応物がない＝衝突ではなく補完関係 |
| `.../movielens_colike_clean/k_interpretation_stability_audit.csv` | 同上 | 同上 | 同上。`colike_clean`固有の安定性監査ファイルで`final_clean`に同名物なし |
| `.../movielens_colike_clean/main_k_interpretation_table.csv` | 同上 | `movielens_final_clean/movielens_main_k_interpretation_table.csv`と内容が重複（後者は列数が多いスーパーセット） | 下記「4」の関係整理を参照。**統合はせず、役割注記を追加する程度が安全** |
| `.../movielens_colike_clean/main_poisson_table.csv` | 同上 | `movielens_final_clean/movielens_main_poisson_table.csv`と数値がほぼ一致（BIC/RMSE_Y/Pearsonは丸め誤差程度の差のみ）。列数は`final_clean`の方が多い（MAE, Spearman, high_colike_AP, NMI, ARI, success_rate, runtime列を追加） | 実際に数値を比較した結果、**同じ実験の「narrow版」と「wide版」であり矛盾はない**。統合するとどちらの用途（本文用の簡潔さ／監査用の網羅性）も失われるため非推奨 |
| `.../movielens_colike_clean/main_recommendation_examples.csv` | 同上 | `final_clean`に対応物なし | 保留・現状維持でよい |
| `.../movielens_colike_clean/supp_lift_baseline_table.csv` | 同上 | `final_clean`に対応物なし | 保留・現状維持でよい |
| `expfam/results/real_data/movielens_final_clean/movielens_file_inventory.csv` | 同上 | `movielens_colike_clean/`との名称類似 | `final_clean`固有のファイル一覧監査。役割衝突なし、保留でよい |
| `.../movielens_final_clean/movielens_main_k_interpretation_table.csv` | 同上 | `colike_clean/main_k_interpretation_table.csv`と重複気味 | 上記と同じ（narrow/wideの関係） |
| `.../movielens_final_clean/movielens_main_poisson_table.csv` | 同上 | `colike_clean/main_poisson_table.csv`と重複気味 | 上記と同じ |
| `.../movielens_final_clean/movielens_supp_lift_baseline_table.csv` | 同上 | `colike_clean/supp_lift_baseline_table.csv`と名称が類似 | 内容比較は未実施（今回は数値突合まで行っていない）。**要追加確認** |
| `.../movielens_final_clean/movielens_use_in_notion_plan.csv` | 同上 | `colike_clean`に直接の対応物なし（`cora_clean/cora_use_in_notion_plan.csv`等、他データセットの同名パターンファイルと横並び） | 保留でよい |
| `reports/movielens_colike_clean/movielens_colike_notion_summary.md` | 同上（ディレクトリ名がexpfam側と衝突） | `expfam/results/real_data/movielens_colike_clean/`と**ディレクトリ名が完全一致**（パスの親が違うだけ） | 下記「4」参照。実際には両方とも同一スクリプト（`summarize_movielens_colike_for_notion.py`）が出力する一体の成果物（CSV+図+Markdown）であり、機能的な衝突ではなく命名慣習上の紛らわしさ |

**総評:** 「衝突」という言葉から連想される「二重管理・矛盾」ではなく、実態は **(a) 同じ実験の narrow版／wide版という意図的な粒度違い** と **(b) `results/`側と`reports/`側で同名ディレクトリが使われている命名慣習の重複** の2種類。中身が矛盾しているわけではないため緊急性は低いが、統合・リネームを行う前に「どちらを本文・スライドで正式引用するか」という研究上の意思決定が必要（ファイル調査だけでは決められない）。

---

## 3. ARCHIVE_CANDIDATE（131件）のうち特に注意が必要なもの

内訳を確認したところ、131件のうち **124件は既に`archive/`配下**（`archive/`本体20件、`expfam/src/archive/`19件、`expfam/results/archive/`85件）で、これらは指示どおり除外する。

**除外後に残る「まだarchive/配下に無いARCHIVE_CANDIDATE」は7件のみで、すべて`cora_subset_pilot`系。** 他に「実験結果やスクリプトと関係しそうなもの」「参照がある可能性があるもの」に該当する対象は見つからなかった。

| path | referenced_by（CSV記載） | 実際にコードを確認した結果 | 注意レベル |
|---|---|---|---|
| `expfam/results/real_data/cora_subset_pilot/cora_subset_data_summary.csv` | `run_fixed_real_cora_balanced_subset_pilot.py; run_fixed_real_cora_subset_pilot.py` | **実際に`run_fixed_real_cora_balanced_subset_pilot.py`のL.665で`pd.read_csv()`により読み込まれている**（BFS版とbalanced版の戦略比較`cora_subset_strategy_comparison.csv`を作るための入力） | **高（CSV上のrisk_level=mediumより実際は高い）。このファイルを動かすとバランス版比較スクリプトの再実行が壊れる** |
| `expfam/results/real_data/cora_subset_pilot/cora_subset_k_metrics.csv` | 同上 | **同上、L.661で読み込まれている（現役の依存関係）** | **高。同上の理由でarchive移動は現時点で不可** |
| `expfam/results/real_data/cora_subset_pilot/cora_subset_z_embeddings.csv` | `run_fixed_real_cora_subset_pilot.py`（自分自身のみ） | 他スクリプトからの読み込みは確認できず | 中〜低。ただし念のため他の`summarize_*`スクリプトからの参照有無を再確認してから判断すべき |
| `expfam/figures/real_data/cora_subset_pilot/cora_subset_k_bic_auc.pdf` | `unreferenced`（CSV上で明示） | 図はコードから再読み込みされる性質のものではないため、この判定は妥当 | 低 |
| `expfam/figures/real_data/cora_subset_pilot/cora_subset_k_bic_auc.png` | `unreferenced` | 同上 | 低 |
| `expfam/figures/real_data/cora_subset_pilot/cora_subset_z_by_label.pdf` | `unreferenced` | 同上 | 低 |
| `expfam/figures/real_data/cora_subset_pilot/cora_subset_z_by_label.png` | `unreferenced` | 同上 | 低 |

**重要な発見：** CSV上は7件とも一律`risk_level=medium`だが、実際にコード（`run_fixed_real_cora_balanced_subset_pilot.py` L.57, 661, 665）を確認したところ、**2件（`cora_subset_data_summary.csv`, `cora_subset_k_metrics.csv`）は現在も別スクリプトの実行時入力として使われている「現役ファイル」**であり、CSVのrisk_level表記より実際のリスクは高い。この2件をarchiveへ移動すると、Cora戦略比較（BFS vs balanced_degree）の再現スクリプトが壊れる。残り5件（`cora_subset_z_embeddings.csv`＋図4点）は他スクリプトからの読み込みが確認できず、比較的安全にarchive候補として扱えそうだが、`run_fixed_real_cora_subset_pilot.py`の実行によって再生成される性質のものなので、緊急でarchiveする理由もない。

**（参考）除外した124件の内訳：**

| 区分 | 件数 |
|---|---:|
| `archive/`（Notion関連スクリプト等、トップレベル） | 20 |
| `expfam/src/archive/`（初期実装スクリプト） | 19 |
| `expfam/results/archive/`（初期実験結果） | 85 |

---

## 4. `movielens_colike_clean` / `movielens_final_clean` / `reports/movielens_colike_clean` の関係（推測）

スクリプトのdocstringとgitコミット履歴、および実際のCSV数値比較から、以下の関係を高い確度で推測できる。

| 項目 | `expfam/results/real_data/movielens_colike_clean/`（+対応figures） | `expfam/results/real_data/movielens_final_clean/` | `reports/movielens_colike_clean/movielens_colike_notion_summary.md` |
|---|---|---|---|
| **ファイル名** | `claims_and_cautions.csv`, `k_interpretation_stability_audit.csv`, `main_k_interpretation_table.csv`, `main_poisson_table.csv`, `main_recommendation_examples.csv`, `supp_lift_baseline_table.csv` | `movielens_file_inventory.csv`, `movielens_main_k_interpretation_table.csv`, `movielens_main_poisson_table.csv`, `movielens_supp_lift_baseline_table.csv`, `movielens_use_in_notion_plan.csv` | `movielens_colike_notion_summary.md`（1ファイルのみ） |
| **更新日（最終コミット）** | 2026-07-07（`d7bb1f9`） | 2026-07-07（`d7bb1f9`） | 2026-07-07（`e3f9243`） — いずれも同一の一括マージ内で追加されており、コミット日時だけでは前後関係を判別できない |
| **置かれている場所** | `expfam/results/real_data/`配下（他データセットの`*_clean/`と横並び） | 同上（`colike_clean`の隣） | `reports/`配下（`expfam/results/`ではなくトップレベルの`reports/`） |
| **生成スクリプト（実際に確認済み）** | `summarize_movielens_colike_for_notion.py` | `summarize_movielens_final_for_figures.py` | `summarize_movielens_colike_for_notion.py`（`movielens_colike_clean`と**同一スクリプト**が出力） |
| **入力（実際に確認済み）** | `movielens_colike_interpretation/`（生の解釈結果、`run_fixed_real_movielens_colike_interpretation.py`が生成） | 同左（`movielens_colike_interpretation/`） | `movielens_colike_clean/`と同時に、同じ入力から生成 |
| **役割** | 本文・Notion掲載用に**3〜4指標へ意図的に縮約**した版（k=2,3,5,8のBIC/RMSE_Y/Pearson等が中心） | 監査・確認用の**フル指標版**（MAE, Spearman, high_colike_AP, NMI, ARI, success_rate, runtimeを追加した上位互換）。`summarize_movielens_final_for_figures.py`のdocstringに「`movielens_colike_clean/`は上書きせず、より完全な指標一式を別途作成する」と明記 | `movielens_colike_clean/`の数値をそのままNotion掲載用の日本語文章に整形したもの（実際に`main_poisson_table.csv`のBIC/RMSE_Y/Pearson値と`notion_summary.md`内の表の数値が一致することを確認済み） |
| **正式版候補** | **本文・スライドで引用するならこちら**（縮約されており可読性が高い、既にNotion文章化もセット） | 数値の裏取り・監査用（引用前にここで数値の整合性を再確認する用途） | Notion投稿の下書きとしての「正式版」候補 |
| **古い版候補** | 該当なし（`movielens_colike_interpretation/`が唯一の「未整形の生データ」で、これが強いて言えば"raw"） | 該当なし | 該当なし |
| **まだ判断できない点** | ①本文・スライドで最終的に`colike_clean`（narrow）と`final_clean`（wide）のどちらの数値セットを正式引用とするかは研究上の意思決定であり、ファイル調査だけでは決められない。②`movielens_final_clean/movielens_supp_lift_baseline_table.csv`と`movielens_colike_clean/supp_lift_baseline_table.csv`の内容差分は今回未突合（数値比較はPoisson主表のみ実施）。③`reports/movielens_colike_clean/`が実際にNotionへ投稿済みかどうかは本リポジトリの情報からは確認不能（外部サービスの状態） | 同上 | 同上 |

**結論（推測の確度）：** 3者は「衝突」ではなく、**同一パイプラインの3つの出力（本文用データ／監査用データ／Notion向け文章）** であり、内容に矛盾はない。ただし`expfam/results/real_data/movielens_colike_clean/`と`reports/movielens_colike_clean/`という**ディレクトリ名の完全一致**は紛らわしく、将来的に一方だけを移動・リネームする作業が発生した場合に混乱を招くリスクは残る。

---

## まとめ

| 分類 | 今回の再検証で追加された知見 |
|---|---|
| DELETE_CANDIDATE (13) | 全件`.gitignore`で除外済み・未追跡であることをコマンドで確認。記載通り低リスクと確認できた |
| REVIEW_REQUIRED (12) | 全件MovieLens co-like関連。「衝突」ではなく「narrow版/wide版の意図的な粒度違い」または「results/とreports/の同名ディレクトリ」であることを実データ突合・docstring確認で特定 |
| ARCHIVE_CANDIDATE (131) | 124件は既存`archive/`配下で対応不要。残り7件（`cora_subset_pilot`系）のうち2件（CSV）は**別スクリプトの現役入力**であり、CSV上の`risk_level=medium`表記より実際のリスクは高いことが判明。図4点は比較的安全 |

**次に行うとよい確認（本レポートでは未実施）：**
- `movielens_final_clean/movielens_supp_lift_baseline_table.csv`と`colike_clean/supp_lift_baseline_table.csv`の数値突合
- `cora_subset_pilot/`の2CSVを本当にarchiveする場合は、`run_fixed_real_cora_balanced_subset_pilot.py`側の入力パスも同時に更新する必要がある（今回は何も変更していない）
