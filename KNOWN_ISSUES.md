# KNOWN_ISSUES.md

## 目的

このリポジトリには、複数のセッション・複数のモデル実装・複数のAI生成レポートが混在している。
本ドキュメントは、研究主張を行う際に**事故（数値の混同・誤った根拠付け・未検証の主張の流用）**を防ぐために、
既知の問題点を一覧化したものである。

事実（コード・CSV・実行ログで確認できること）と解釈（推測・評価）を分けて記載する。
本ドキュメントは新規作成のみであり、既存ファイルの内容は変更していない。

---

## Issue一覧

| ID | 重要度 | 問題 | 現状 | 影響 | 次にやること | 関連ファイル |
|----|------|------|------|------|----------|------------|
| KI-001 | 高 | E-step Y側Term3の0.5係数問題 | 原稿式・root CLAUDE.mdは1/2なしを採用。`model_dual_expfam.py`（L.159, L.200）・`model_expfam.py`（L.109, L.135）には0.5が残存。`model_dual_expfam_fixed.py`（L.77, L.113）は0.5を除去済みだが補助実験のみで使用 | 本文採用実験（Exp1-4）は0.5あり実装で実行済み | 「Newton方向が正しいとは断定できない」という限定条件を主張時に必ず付記する。修論フェーズで0.5除去版での再実験を検討 | `expfam/src/model_dual_expfam.py`, `model_expfam.py`, `model_dual_expfam_fixed.py`, root `CLAUDE.md` |
| KI-002 | 高 | 旧版実装とfixed版実装の実験結果が混在する危険 | `mismatch_fixed_summary.csv`・`comparison_quick.csv`はfixed版（0.5なし）由来。それ以外のExp1-4結果はすべて旧版（0.5あり） | 異なる実装由来の数値を同じ表・図に混在させると誤った比較になる | 数値を引用する際は必ず「旧版」か「fixed版」かを明記する | `expfam/results/distribution_mismatch_fixed/*.csv`, `expfam/results/exp_scenario_*_exp4_mismatch.csv` |
| KI-003 | 高 | 23.6倍 / 41.5倍 / 38.97倍の混同リスク | 23.6倍＝旧版・Scen.C・図1(b)灰色バー（X=Gauss/Y=Bern）。41.5倍＝旧版・Scen.C・本文記載の全条件中最大（X=Gauss/Y=Pois、図に対応バーなし）。38.97倍＝fixed版・Scen.C・mismatch_fixed_summary.csvの別条件（true=bern/gauss, est=poisson/bernoulli） | いずれも近い値だが、モデル・true条件・est条件が異なる独立した数値 | 3つを並べて引用する場合は必ず出所（旧版/fixed版、true/est条件、CSV行）を明記する | `expfam/results/exp_scenario_C_exp4_mismatch.csv`, `expfam/results/distribution_mismatch_fixed/mismatch_fixed_summary.csv`, `figures/fig1b_misspecification_color.*` |
| KI-004 | 中 | `comparison_old_vs_fixed.png/pdf` の生成元不明 | `expfam/figures/distribution_mismatch_fixed/`に存在するが、`expfam/src/*.py`全16ファイルをgrepしても生成スクリプトが見つからない | この図を本文や報告書で引用する場合、再現性が保証できない | 生成元が見つかるまでは「生成元未確認」として扱い、根拠として使わない | `expfam/figures/distribution_mismatch_fixed/comparison_old_vs_fixed.png`, `comparison_old_vs_fixed.pdf` |
| KI-005 | 中 | Categorical未実装 | `model_dual_expfam.py`のVALID_FAMILIESにGaussian/Bernoulli/Poissonのみ。Categoricalは未実装 | 「指数型分布族へ一般化」という主張の範囲はGaussian/Bernoulli/Poissonに限定される | 主張時は対応分布族を明記する。Categorical対応は将来課題として扱う | `expfam/src/model_dual_expfam.py` |
| KI-006 | 中 | Wine実データ実験は未評価 | **部分的に解消（2026-06-18）**。旧0.5版（`run_wine_dual.py`, `wine_dual_results.csv`, `wine_F.npy`）自体は引き続き未評価だが、fixed版（0.5除去）でのWine実データ評価（BIC k選択・ablation・旧版との突合）は`wine_fixed_pilot/`・`wine_old05_audit/`で実施済み | 実データへの適用例として、fixed版に限れば本文・スライドに使える状態（BIC最小k=3が真のクラス数と一致）。旧0.5版のWine結果は引き続き参考値扱い | 学会予稿には未収録のため、修論での記載時は「fixed版のWine実データ検証」であることを明記する | `expfam/src/run_wine_dual.py`, `expfam/results/wine_dual_results.csv`, `expfam/results/wine_F.npy`, `expfam/src/run_fixed_real_wine_pilot.py`, `expfam/results/real_data/wine_fixed_pilot/`, `expfam/results/real_data/wine_old05_audit/` |
| KI-011 | 中 | Cora実データにおけるBICの機能不全 | Cora balanced subset（density=0.011）でBIC最小がk=1を選択する一方、AUC/AP最大はk=6、NMI/ARI最大はk=3と、選択基準ごとに最適kが異なる | 「BICが常に適切なkを選ぶ」と主張できない。実データでのモデル選択は単一指標では不十分 | 主張時は評価指標ごとの最適kを併記する。疎密度データでのBICペナルティの扱いは修論での検討課題 | `expfam/results/real_data/cora_balanced_k_sweep/*.csv`, `reports/real_data_experiment_summary.md` §4, §7 |
| KI-012 | 中 | MovieLens Poisson実験のoverdispersionとstrict held-out未対応 | Y_count（共評価数）はvar/mean≈10でPoisson仮定（var/mean=1）から大きく逸脱。また現在のモデルAPIはpair maskに対応しておらず、strict held-out（未知ペア予測）の評価ができない | 「MovieLensで未知ペアの共評価数を予測できた」という主張はできない。in-sample再構成の結果に限定される | 主張時は「in-sample再構成」であることを明記する。pair mask対応（`calc_w0`/`calc_w`/`_calc_gradient`/`_calc_precision_matrix`への引数追加）は修論フェーズの課題 | `expfam/src/run_fixed_real_movielens_poisson_pilot.py`, `expfam/src/run_fixed_real_movielens_heldout_count.py`, `reports/real_data_experiment_summary.md` §5, §9 |
| KI-013 | 低 | `movielens_colike_clean`と`movielens_final_clean`の名称類似による混同リスク | 両者とも「MovieLens co-like実験の最終整形」という名前だが、前者は本文/Notion用に3指標へ縮約した版、後者は監査用のフル指標版（`summarize_movielens_final_for_figures.py`のdocstringに明記、前者を上書きしない設計） | 引用時にどちらのCSVを参照したか取り違えるリスク | 参照する際は必ずディレクトリ名とファイル名の両方を明記する。将来的にはより区別しやすい命名への変更を検討 | `expfam/results/real_data/movielens_colike_clean/`, `expfam/results/real_data/movielens_final_clean/` |
| KI-007 | 高 | AI生成レポートを根拠にしてしまう危険 | `GEMINI_REPORT_*.md`（`expfam/results/`およびその`archive/`配下）、`docs_for_notebooklm/*`、`reports/*`の一部はAIによる生成・要約であり、研究者による検証が完了していないものを含む | AI生成レポートの数値・結論をそのまま研究主張の根拠にすると、検証されていない情報が伝播する | 数値主張は必ず元のCSV・実行ログに遡って確認する。AI生成レポートは「参考」「未検証」として扱う | `expfam/results/GEMINI_REPORT_*.md`, `expfam/results/archive/GEMINI_REPORT_*.md`, `docs_for_notebooklm/*` |
| KI-008 | 中 | `expfam/CLAUDE.md` は旧セッション由来で低信頼 | `expfam/CLAUDE.md`は旧Geminiセッション向けに書かれたファイル。`expfam/README.md`自身も「正しい確定事項はルートのCLAUDE.mdを参照」と明記している | 古い前提（Σ_{i≠j}に1/2が必要、等）が残っている可能性がある | 確定事項は常にルート`CLAUDE.md`を優先する。`expfam/CLAUDE.md`は参考のみ | `expfam/CLAUDE.md`, `CLAUDE.md`（root） |
| KI-009 | 低 | archive/Notion系ファイルは研究本体ではない | `archive/notion_scripts/`, `archive/misc/`にNotion投稿用スクリプト・katex調査メモ・論文執筆参考PDFが存在する | 研究の数式・実験ロジックとは無関係。誤って参照すると混乱を招く | 研究内容の確認時は参照しない。整理候補としてCLEANUP_MANIFEST.mdに記載 | `archive/notion_scripts/*`, `archive/misc/*`, `archive/paper_writing_examples/*` |
| KI-010 | 低 | BICのパラメータ数定義の確認余地 | `expfam/CLAUDE.md`に記載のnum_params定義（`k*d - k*(k-1)//2 + ...`）の検証は完了していない | BIC値・k選択結果の解釈に影響する可能性がある | `utils_expfam.py`の`calc_bic_dual`実装とBIC定義の手計算照合を別途行う | `expfam/src/utils_expfam.py` |

---

## 今すぐ主張してよいこと

- Dual-ExpFam LSMは、Gaussian / Bernoulli / Poissonの3分布族について、X側・Y側を任意に指定できる実装が存在する（`model_dual_expfam.py`のコード上で確認可能）。
- 3シナリオ（A: Poisson-X/Bernoulli-Y, B: Gaussian-X/Poisson-Y, C: Bernoulli-X/Gaussian-Y）でExp1-4が実行され、結果CSVが存在する。
- 各シナリオでBICによりk*=3が選択される（`exp_scenario_*_exp1_k.csv`で確認可能、`reports/claims_and_evidence.md`にも記載）。
- nの増加に伴いRMSE(Z)が改善する傾向が3シナリオで確認できる（`exp_scenario_*_exp2_n.csv`）。
- fixed版（0.5除去）でシナリオA/B/CのExp1-4が正式に再実行され、結果CSVが存在する（`expfam/results/fixed_official/`）。
- fixed版はWine・Cora・MovieLensの3つの実データセットに適用でき、いずれもNaNなく全fitが成功した（`reports/real_data_experiment_summary.md` §8）。
- Cora（自然な引用ネットワーク）のheld-out link predictionでrandom基準を上回る予測性能を確認した（test_AP≈2.6〜2.8×random）。

## まだ主張してはいけないこと

- 「0.5係数を除去したfixed版の方が常に優れている」という主張（`comparison_quick.csv`のratio_fix_oldは0.27〜1.23倍と条件依存で一貫しない）。
- 「Wine実データでDual-ExpFam LSMが有効である」という主張について：fixed版でのBIC k選択・ablationは実施済みだが、
  Wine の Y はラベル由来の人工的な関係であり「自然ネットワークでの検証」とは言えない（KI-006）。
- 「Categorical分布にも対応している」という主張（未実装）。
- 「41.5倍・23.6倍・38.97倍が同一条件・同一モデルの結果である」という主張（KI-003参照、異なる実験の値）。
- AI生成レポート（GEMINI_REPORT_*等）の結論を一次根拠として引用すること。
- 「MovieLensで未知ペアの共評価数を予測できた」という主張（strict held-outは未実装、in-sample再構成のみ、KI-012）。
- 「Cora（n=280 balanced subset）の結果がfull Cora（n=2708）に一般化する」という主張（未検証）。
- 「実データでBICが常に適切なkを選ぶ」という主張（Coraでは疎密度によりk=1を選択、KI-011）。
- 「実データ実験フェーズの結果が学会予稿の主張に含まれる」という主張（`conference_submission_final_draft.md`には未収録、修論フェーズ向けの追加検証）。

## 修論フェーズで優先的に検証すること

1. KI-001：0.5係数を除去した実装（`model_dual_expfam_fixed.py`相当）でExp1-4を再実行し、本文の数値が変化するか確認する。
   → fixed_official/half_factor_checkで人工データについては実行済み。原稿数値との対応整理が次のステップ。
2. KI-003：23.6倍・41.5倍に対応する条件をfixed版でも計算し、0.5除去の影響を定量化する。
   → fixed版mismatch実験（A:4.34×, B:9.04×, C:40.37×）は実行済み。0.5あり版との対応表作成が未着手。
3. KI-006：Wine実データ実験の結果を検証し、補助実験として使えるか判断する。→ fixed版で完了。旧0.5版との突合も`wine_old05_audit`で実施済み。
4. KI-010：BICのパラメータ数定義を手計算で再検証する。
5. KI-011：疎な実データ（Cora等）でBICのペナルティ項が過大になる問題への対処法を検討する。
6. KI-012：MovieLens Poisson実験のpair mask対応（strict held-out評価の実装）、および負の二項分布によるoverdispersion対応を検討する。
7. Cora実験をfull Cora（n=2708）または他の引用ネットワーク（Citeseer等）に拡張する。
8. MovieLensのuser-node投影・二部グラフ拡張など、他のprojection方式との比較を検討する（`reports/movielens_pilot_design.md` 案B/C）。
