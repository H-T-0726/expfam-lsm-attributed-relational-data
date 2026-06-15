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
| KI-006 | 中 | Wine実データ実験は未評価 | `run_wine_dual.py`, `wine_dual_results.csv`, `wine_F.npy`は存在するが、結果の検証・解釈は実施されていない | 実データへの適用例として本文に使える状態ではない | 補助・将来課題として位置づけ、検証が完了するまで本文には使わない | `expfam/src/run_wine_dual.py`, `expfam/results/wine_dual_results.csv`, `expfam/results/wine_F.npy` |
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

## まだ主張してはいけないこと

- 「0.5係数を除去したfixed版の方が常に優れている」という主張（`comparison_quick.csv`のratio_fix_oldは0.27〜1.23倍と条件依存で一貫しない）。
- 「Wine実データでDual-ExpFam LSMが有効である」という主張（未評価）。
- 「Categorical分布にも対応している」という主張（未実装）。
- 「41.5倍・23.6倍・38.97倍が同一条件・同一モデルの結果である」という主張（KI-003参照、異なる実験の値）。
- AI生成レポート（GEMINI_REPORT_*等）の結論を一次根拠として引用すること。

## 修論フェーズで優先的に検証すること

1. KI-001：0.5係数を除去した実装（`model_dual_expfam_fixed.py`相当）でExp1-4を再実行し、本文の数値が変化するか確認する。
2. KI-003：23.6倍・41.5倍に対応する条件をfixed版でも計算し、0.5除去の影響を定量化する。
3. KI-006：Wine実データ実験の結果を検証し、補助実験として使えるか判断する。
4. KI-010：BICのパラメータ数定義を手計算で再検証する。
