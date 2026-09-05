# KNOWN_ISSUES.md

## 目的

このリポジトリには、複数のセッション・複数のモデル実装・複数のAI生成レポートが混在している。
本ドキュメントは、研究主張を行う際に**事故（数値の混同・誤った根拠付け・未検証の主張の流用）**を防ぐために、
既知の問題点を一覧化したものである。

事実（コード・CSV・実行ログで確認できること）と解釈（推測・評価）を分けて記載する。

本ドキュメントは **継続的に forward update される canonical safety document** である。
historical な issue 行は記録として保持しつつ、**current status は日付入りの forward update 節を正とする**
（最新は「2026-08-31 forward update」）。
current guidance と historical 記述が食い違う場合は forward update 節を優先する。

---

## Issue一覧

| ID | 重要度 | 問題 | 現状 | 影響 | 次にやること | 関連ファイル |
|----|------|------|------|------|----------|------------|
| KI-001 | 高 | E-step Y側Term3の0.5係数問題（**5系統の混同リスクを含む**） | **1/2の所在は5系統で異なる。①Mikawa et al. 2024の印刷された原論文式（Eq.19/20/22/23、Appendix A-1/A-3/A-5）＝1/2あり（2026-08-18に原論文を直接確認）。②old 0.5 Python系列（`model_dual_expfam.py` L.159/L.200、`model_expfam.py` L.109/L.135）＝1/2あり。③本研究の独立再導出・採用式（unique undirected-pair conditional）＝extra 1/2なし。④fixed Python系列（`model_dual_expfam_fixed.py` L.77/L.113）＝1/2なし。⑤MATLAB `calcAi`＝1/2なし。**①と③の差は意図的な設計判断であり、②と④の差は実装系列の差。fixed版は補助実験・実データ実験フェーズで使用 | 本文採用実験（Exp1-4）は0.5あり実装（系統②）で実行済み。**「原論文にも1/2がない」と書くと誤り**（系統①には1/2がある） | 「Newton方向が正しいとは断定できない」という限定条件を主張時に必ず付記する（0.5が掛かるのはY側項のみで、Z事前分布項・X側項には掛かっていないため）。5系統を並べて論じる際は必ず系統名を明記する。主根拠は独立導出であり、MATLABは補助比較として扱う。修論フェーズで0.5除去版での再実験を検討 | `expfam/src/model_dual_expfam.py`, `model_expfam.py`, `model_dual_expfam_fixed.py`, root `CLAUDE.md`, `RESEARCH_MASTER.md` §6.1, **`docs/math_notes/half_factor_primary_source_confirmation_20260818.md`** |
| KI-002 | 高 | 旧版実装とfixed版実装の実験結果が混在する危険 | `mismatch_fixed_summary.csv`・`comparison_quick.csv`はfixed版（0.5なし）由来。それ以外のExp1-4結果はすべて旧版（0.5あり） | 異なる実装由来の数値を同じ表・図に混在させると誤った比較になる | 数値を引用する際は必ず「旧版」か「fixed版」かを明記する | `expfam/results/distribution_mismatch_fixed/*.csv`, `expfam/results/exp_scenario_*_exp4_mismatch.csv` |
| KI-003 | 高 | 23.6倍 / 41.45倍 / 38.97倍の混同リスク | 23.6倍＝旧版・Scen.C・図1(b)灰色バー（est X=Gauss/Y=Bern）。**41.45倍＝旧版・Scen.C・本文記載の全条件中最大。根拠CSVは `expfam/results/exp_scenario_C_exp4_mismatch.csv` の est X=Gaussian, Y=Poisson（両側誤指定）条件と特定完了**（`reports/mismatch_audit/mismatch_audit_report_20260708.md` §1。本文 L.83 の「41.5倍」は同値の丸め、図に対応バーなし）。38.97倍＝fixed版・Scen.C・mismatch_fixed_summary.csvの別条件（true=bern/gauss, est=poisson/bernoulli） | いずれも近い値だが、モデル系列・true条件・est条件が異なる独立した数値 | 3つを並べて引用する場合は必ず出所（旧版/fixed版、true/est条件、CSV行）を明記する。41.45倍は「両側誤指定」であり、図1(b)の23.6倍（X側のみ誤指定の固定条件）とは条件が異なる点も明記する | `expfam/results/exp_scenario_C_exp4_mismatch.csv`, `expfam/results/distribution_mismatch_fixed/mismatch_fixed_summary.csv`, `figures/fig1b_misspecification_color.*`, `reports/mismatch_audit/mismatch_audit_report_20260708.md` |
| KI-004 | 中 | `comparison_old_vs_fixed.png/pdf` の生成元不明 | `expfam/figures/distribution_mismatch_fixed/`に存在するが、`expfam/src/*.py`全16ファイルをgrepしても生成スクリプトが見つからない | この図を本文や報告書で引用する場合、再現性が保証できない | 生成元が見つかるまでは「生成元未確認」として扱い、根拠として使わない | `expfam/figures/distribution_mismatch_fixed/comparison_old_vs_fixed.png`, `comparison_old_vs_fixed.pdf` |
| KI-005 | 中 | Categorical未実装 | `model_dual_expfam.py`のVALID_FAMILIESにGaussian/Bernoulli/Poissonのみ。Categoricalは未実装 | 「指数型分布族へ一般化」という主張の範囲はGaussian/Bernoulli/Poissonに限定される | 主張時は対応分布族を明記する。Categorical対応は将来課題として扱う | `expfam/src/model_dual_expfam.py` |
| KI-006 | 中 | Wine実データ実験は未評価 | **部分的に解消（2026-06-18）**。旧0.5版（`run_wine_dual.py`, `wine_dual_results.csv`, `wine_F.npy`）自体は引き続き未評価だが、fixed版（0.5除去）でのWine実データ評価（BIC k選択・ablation・旧版との突合）は`wine_fixed_pilot/`・`wine_old05_audit/`で実施済み | 実データへの適用例として、fixed版に限れば本文・スライドに使える状態（BIC最小k=3が真のクラス数と一致）。旧0.5版のWine結果は引き続き参考値扱い | 学会予稿には未収録のため、修論での記載時は「fixed版のWine実データ検証」であることを明記する | `expfam/src/run_wine_dual.py`, `expfam/results/wine_dual_results.csv`, `expfam/results/wine_F.npy`, `expfam/src/run_fixed_real_wine_pilot.py`, `expfam/results/real_data/wine_fixed_pilot/`, `expfam/results/real_data/wine_old05_audit/` |
| KI-011 | 中 | Cora実データにおけるBICの機能不全 | Cora balanced subset（density=0.011）でBIC最小がk=1を選択する一方、AUC/AP最大はk=6、NMI/ARI最大はk=3と、選択基準ごとに最適kが異なる | 「BICが常に適切なkを選ぶ」と主張できない。実データでのモデル選択は単一指標では不十分 | 主張時は評価指標ごとの最適kを併記する。疎密度データでのBICペナルティの扱いは修論での検討課題 | `expfam/results/real_data/cora_balanced_k_sweep/*.csv`, `reports/real_data_experiment_summary.md` §4, §7 |
| KI-012 | 中 | MovieLens Poisson実験のoverdispersionとstrict held-out未対応 | Y_count（共評価数）はvar/mean≈10でPoisson仮定（var/mean=1）から大きく逸脱。また現在のモデルAPIはpair maskに対応しておらず、strict held-out（未知ペア予測）の評価ができない | 「MovieLensで未知ペアの共評価数を予測できた」という主張はできない。in-sample再構成の結果に限定される | 主張時は「in-sample再構成」であることを明記する。pair mask対応（`calc_w0`/`calc_w`/`_calc_gradient`/`_calc_precision_matrix`への引数追加）は修論フェーズの課題 | `expfam/src/run_fixed_real_movielens_poisson_pilot.py`, `expfam/src/run_fixed_real_movielens_heldout_count.py`, `reports/real_data_experiment_summary.md` §5, §9 |
| KI-013 | 低 | `movielens_colike_clean`と`movielens_final_clean`の名称類似による混同リスク | 両者とも「MovieLens co-like実験の最終整形」という名前だが、前者は本文/Notion用に3指標へ縮約した版、後者は監査用のフル指標版（`summarize_movielens_final_for_figures.py`のdocstringに明記、前者を上書きしない設計） | 引用時にどちらのCSVを参照したか取り違えるリスク | 参照する際は必ずディレクトリ名とファイル名の両方を明記する。将来的にはより区別しやすい命名への変更を検討 | `expfam/results/real_data/movielens_colike_clean/`, `expfam/results/real_data/movielens_final_clean/` |
| KI-007 | 高 | AI生成レポートを根拠にしてしまう危険 | `GEMINI_REPORT_*.md`（`expfam/results/`およびその`archive/`配下）、`docs_for_notebooklm/*`、`reports/*`の一部はAIによる生成・要約であり、研究者による検証が完了していないものを含む | AI生成レポートの数値・結論をそのまま研究主張の根拠にすると、検証されていない情報が伝播する | 数値主張は必ず元のCSV・実行ログに遡って確認する。AI生成レポートは「参考」「未検証」として扱う | `expfam/results/GEMINI_REPORT_*.md`, `expfam/results/archive/GEMINI_REPORT_*.md`, `docs_for_notebooklm/*` |
| KI-008 | 中 | `expfam/CLAUDE.md` は旧セッション由来で低信頼 | `expfam/CLAUDE.md`は旧Geminiセッション向けに書かれたファイル。`expfam/README.md`自身も「正しい確定事項はルートのCLAUDE.mdを参照」と明記している | 古い前提（Σ_{i≠j}に1/2が必要、等）が残っている可能性がある | 確定事項は常にルート`CLAUDE.md`を優先する。`expfam/CLAUDE.md`は参考のみ | `expfam/CLAUDE.md`, `CLAUDE.md`（root） |
| KI-009 | 低 | archive/Notion系ファイルは研究本体ではない | **現在のGit tree上に実在するのは `archive/notion_scripts/` と `archive/misc/` の2つのみ**（Notion投稿用スクリプト・katex調査メモ等）。**`archive/paper_writing_examples/` はディスク上にもGit追跡上にも存在しない**（2026-08-20確認）。過去の文書（`CLEANUP_MANIFEST.md`・旧`START_HERE.md`等）に同ディレクトリ名の文字列が残っているが、それは当時の記録であり、現在のtreeに存在することを意味しない | 研究の数式・実験ロジックとは無関係。誤って参照すると混乱を招く。存在しないパスを「関連ファイル」として辿ろうとすると時間を失う | 研究内容の確認時は参照しない。整理候補としてCLEANUP_MANIFEST.mdに記載（同ファイルは凍結文書のため書き換えない）。`archive/paper_writing_examples/` を実在するものとして扱わない | `archive/notion_scripts/*`, `archive/misc/*`（**`archive/paper_writing_examples/` は不在**） |
| KI-014 | 中 | historical experiment environment を記録から完全に復元できない | runinfo.csvは`git_head`・`branch`・seedを記録するが、**Pythonバージョンとパッケージ版を記録していない**（既存runinfoのヘッダに該当列が存在しない）。そのため、**過去の実験環境をリポジトリ内の記録だけから完全・一意に復元することはできない**。個々の実験について部分的な手掛かり（`git_head`、当時のコード、依存の暗黙的制約など）が残る可能性までは否定しない。Phase 2（2026-08-20実施）で`reports/environment/baseline_20260818.md`・`.python-version`・`requirements*.txt`を整備したが、**これはfuture reproducible baselineであり、historical environmentの証明ではない** | 過去の数値のbit-exactな再現は保証できない。「過去の実験はこの環境で実行された」とは主張できない | 今後の実験はruninfoに環境情報（`python_version`・`platform`・`requirements_sha256`・主要パッケージ版）を含める（新規実験のみ・既存列順は壊さない）。**過去のCSV/runinfoは変更しない**（事後に推定した環境情報を追記するとhistorical provenanceを損なうため） | `reports/environment/baseline_20260818.md`, `.python-version`, `requirements.txt`, `requirements-dev.txt` |
| KI-010 | 低 | BICのパラメータ数定義の確認余地 | `expfam/CLAUDE.md`に記載のnum_params定義（`k*d - k*(k-1)//2 + ...`）の検証は完了していない | BIC値・k選択結果の解釈に影響する可能性がある | `utils_expfam.py`の`calc_bic_dual`実装とBIC定義の手計算照合を別途行う | `expfam/src/utils_expfam.py` |

**（2026-07-19 更新注記：KI-010 / KI-012 の現在地）**

- KI-012：「現在のモデルAPIはpair maskに対応しておらず」は2026-06時点の記述。
  **experimental系列（`expfam/src/experimental/model_dual_expfam_masked.py`、
  2026-07-10コミット16d456c以降）でpair mask（strict held-out）対応済み**であり、
  per-columnフェーズのMovieLens pilotはstrict held-outで実施されている。
  ただしfixed本体API（`model_dual_expfam_fixed.py`）には未統合のままであり、
  overdispersion（NB対応）はexperimentalのNB版で試行段階。KI-012は
  「fixed本体APIへの統合と正式化」が残課題として継続。
- KI-010：理論監査（2026-07-18、`reports/theory_audit/theory_audit_report_20260718.md`
  §6-7）で次を確認済み：(i) `kd − k(k−1)/2` は観測分布を不変にする直交群 O(k) の
  次元と整合する（導出あり）。(ii) w0・w（2個）はkに依存しない定数のため
  **k選択の順位には影響しない**。(iii) 一方、Q_strictは周辺尤度ではなく
  EMのQ関数のMC近似であるため、現行基準はSchwarz BICではなく
  「Qベース完全データ型基準（ICL-type）」として扱う。この位置づけの問題は
  パラメータ数の定義とは別問題として未解決（`reports/theory_audit/
  diagnostic_designs_20260719.md` §1参照）。

---

## 今すぐ主張してよいこと

**完全な分類は `RESEARCH_MASTER.md` §14「2026-08-31 Current Claim Ledger」を正本とする。**
本リストはその要約であり、食い違う場合は §14 を優先する。

- Dual-ExpFam LSMは、Gaussian / Bernoulli / Poissonの3分布族について、X側・Y側を任意に指定できる実装が存在する（`model_dual_expfam.py`のコード上で確認可能）。**実装レベルの一般化であり性能主張ではない。Categorical は未実装。**
- 3シナリオ（A: Poisson-X/Bernoulli-Y, B: Gaussian-X/Poisson-Y, C: Bernoulli-X/Gaussian-Y）でExp1-4が実行され、結果CSVが存在する。
- **潜在次元の選択（限定付き・QUALIFIED ONLY）:** 明示した人工設定において、
  **歴史的に `BIC` と呼ばれてきた Q ベース基準**が `K = 3` を選択した
  （fixed 系列では候補 `K = 1,…,9` から 30/30 trial、
  `expfam/results/fixed_official/exp1_k9/fixed_exp1_bic_k1to9_bestk_by_trial.csv`）。
  **これは Schwarz BIC の妥当性・一般的な true-K recovery・model-selection consistency のいずれも意味しない**（KI-010 forward update）。
  **「BIC で正しい K を選べる」と読める表現を使わない。**
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
- 「MovieLensで未知ペアの共評価数を予測できた」という主張。
  （**この禁止は 2026-08-31 現在も有効**。ただし括弧内の理由づけは更新されている:
  strict held-out は **experimental 系列（`model_dual_expfam_masked.py`）で実装済み**であり、
  Issue #33 の user-disjoint validation も実施済みである。
  それでもなおこの主張が禁止なのは、Issue #33 report §9 の禁止句リストに含まれること、
  Y が overdispersed で Poisson log-likelihood は score にすぎないこと、
  および lineage が experimental prototype であることによる。詳細は下記 C。
  **「strict held-out は未実装」を current statement として引用しない。**）
- 「Cora（n=280 balanced subset）の結果がfull Cora（n=2708）に一般化する」という主張（未検証）。
- 「実データでBICが常に適切なkを選ぶ」という主張（Coraでは疎密度によりk=1を選択、KI-011）。
- 「実データ実験フェーズの結果が学会予稿の主張に含まれる」という主張（`conference_submission_final_draft.md`には未収録、修論フェーズ向けの追加検証）。
- **「先行研究（Mikawa et al. 2024）の印刷された式にも 1/2 がない」という記述**（原論文 Eq.19/20/22/23・Appendix A-1/A-3/A-5 には 1/2 がある。2026-08-18 に一次確認済み、KI-001）。
- **「過去の実験結果が特定の実行環境で生成された」という主張**（過去の runinfo に環境情報がなく、記録だけから完全・一意には復元できない。`reports/environment/baseline_20260818.md` は今後の baseline であり historical environment の証明ではない、KI-014）。

## 修論フェーズで優先的に検証すること（historical、2026-05〜2026-06 時点の一覧）

> **【2026-08-31】** 以下は当時の一覧であり、**current な TODO ではない**。
> 研究上の TODO は GitHub Issue で管理し、canonical docs には置かない（root `CLAUDE.md` §8）。
> 各項目の 2026-08-31 時点の状態は下の「2026-08-31 forward update」§I を参照。
> 科学的に未解決である事実は `RESEARCH_MASTER.md` §14 の `UNRESOLVED` に記載している。


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

---

## 2026-08-31 forward update（Phase 6〜7e の統合）

**本節自体は append-only である。** ただし本文書全体は current canonical safety document であり、
2026-08-31 の統合では次の 2 種類の変更を行っている。

- **保持したもの:** 「Issue一覧」表の **historical row 本体**（KI-001〜KI-014 の各行と 2026-07-19 更新注記）。
  これらは当時の記録として書き換えていない。
- **forward update したもの:** 文書上部の **current guidance と summary list**。
  具体的には ①冒頭「目的」節の文書自体に関する説明、②「今すぐ主張してよいこと」の
  潜在次元選択に関する記述（無限定の BIC 表現を限定付き表現へ変更）、
  ③「まだ主張してはいけないこと」の MovieLens 行（禁止は維持し、理由を
  strict held-out の現状に合わせて更新）、④「修論フェーズで優先的に検証すること」の
  見出しと説明（historical であることの明示）。

historical state を残す必要がある箇所は `Historical` / `historical state` として明示的に隔離している。
**現在形の主張の正本は `RESEARCH_MASTER.md` §14「2026-08-31 Current Claim Ledger」である。**
本節と上のリストが食い違う場合は、本節と §14 を現在の状態とする。

### A. 既存 KI の現在の状態（historical record は削除しない）

| KI | 2026-08-31 時点の状態 | 根拠 |
|---|---|---|
| KI-001（0.5 係数・5 系統） | **変更なし**。採用式（1/2 なし）を正とし、原論文の印刷式には 1/2 が**ある**。「Newton 方向が全体として正しいとは断定できない」の付記は引き続き必須 | `RESEARCH_MASTER.md` §6.1 |
| KI-002（旧版と fixed の混在） | **範囲が拡大**。現在は **6 lineage**（A 原論文印刷式 / B 旧 0.5 / C fixed / D experimental / **E objective-consistent** / **F per-column prototype**）を区別する必要がある。E・F は Phase 6 以降で新設・多用された | `RESEARCH_MASTER.md` §12.1 |
| KI-003（23.6× / 41.45× / 38.97×） | **変更なし**。加えて fixed 系列の誤指定最悪比は **A 4.3414× / B 9.0405× / C 40.3706×**（ablation 行を除く）であることを一次 CSV で再確認した。`fixed_exp4_scen_c_ratios.csv` の 46.8637× は `fix_w=True` の ablation（`X_only`）であり**誤指定倍率として引用しない** | `RESEARCH_MASTER.md` §15、`expfam/results/fixed_official/exp4/fixed_exp4_scen_{a,b,c}_ratios.csv` |
| KI-005（Categorical 未実装） | **変更なし**。「指数型分布族すべてに対応」とは書かない | — |
| KI-006（Wine） | **変更なし**（部分解消のまま）。Wine の Y はラベル由来 | — |
| KI-010（BIC の呼称・パラメータ数） | **大幅に具体化。詳細は下記 D。** | `RESEARCH_MASTER.md` §12.6 |
| KI-011（Cora で BIC が機能不全） | **変更なし・原因は UNRESOLVED**。`Σ_i ln det A_i` は未測定、試行は 3 のみ。「KI-011 を完全に説明した」とは書かない | Issue #35 PL1 |
| KI-012（MovieLens overdispersion と strict held-out 未対応） | **PARTIALLY RESOLVED。詳細は下記 C。** | `RESEARCH_MASTER.md` §12.5 |
| KI-014（過去実験環境の非復元性） | **変更なし**。加えて Phase 7e について新種の provenance limitation が判明（下記 F） | — |

### B. 新規 KI-015 — per-column prototype の objective inconsistency（tail）と objective-consistent lineage

**事実（Issue #23、一次確認済み）:** `experimental/model_dual_expfam_percolumn.py` は、
Poisson の clip 区間 `[-20,10]` の**外側**と Bernoulli の floor された tail において、
報告している目的関数の勾配・負 Hessian が 0 であるのに、実装の score・precision は非零を返す。
決定論的反例: `eta=11.5, x=3` で実装 score `-22023.465794806718` / precision `22026.465794806718`、
実際の目的関数の有限差分はいずれも `0.0`。**内点（interior）では独立導出・有限差分と一致している**
（勾配 2.26e-10 / 曲率 7.51e-09）。

**対応（Issue #25）:** `experimental/model_dual_expfam_consistent.py`（lineage E）を
**forward 修正**として新設した。**過去の結果は再計算していない。**
`numerics_mode` の既定は `legacy` のままであり、Phase 6 より前の全実験は legacy numerics で実行されている。

**影響:** legacy numerics で実行された per-column 系の過去結果について、
**EM 反復中に clip が発動したかは記録されておらず UNRESOLVED** である。
新しい per-column 実験は `numerics_mode="consistent"` を明示的に選ぶ運用とする。

**まだ主張してはいけないこと:** 「legacy 系列の per-column 結果は clip の影響を受けていない」。

### C. KI-012 の forward update — historical state と current state を分ける

**historical state（2026-06 時点、上の KI-012 行の記述）:**
「現在のモデル API は pair mask に対応しておらず strict held-out 評価ができない」。
**この記述は 2026-06 時点として正しく、削除しない。**

**current state（2026-08-31）:**

1. **pair mask（strict held-out）は experimental 系列で実装済み**（`model_dual_expfam_masked.py`、2026-07-10 以降）。
   ただし **fixed 本体 API（`model_dual_expfam_fixed.py`）には未統合**。
2. **Issue #33 で user-disjoint protocol による実データ検証を実施済み**
   （30 splits、train users のみからの movie selection、train-only 由来属性、360 fits）。
   旧 pilot にあった selection leakage（100 映画 subset が full-data popularity で選ばれていた）を除く設計。
3. **Y の overdispersion は解消していない**（split 半分で var/mean 5.0–5.6、full data で 9.9）。
   Poisson の per-pair log-likelihood は **score として**用いており、正しく指定された尤度ではない。

**それでも「MovieLens で未知ペアの共評価数を予測できた」とは書かない。**
Issue #33 の report §9 が禁止句として明示している
（"we predicted unseen co-rating counts on MovieLens" は FORBIDDEN_CLAIM_PHRASES に含まれる）。
また #28 の **F9**（MovieLens では genre-only X ですら strict held-out Y を確実には改善しない）は未解消である。

### D. KI-010 の forward update — 基準の呼称制限（一次確認により具体化）

**現行基準を「Schwarz BIC」と呼ばない。** 関数名 `calc_bic_dual`・CSV 列名 `BIC`・過去結果の呼称は
provenance のため**変更しない**（この方針は変更なし）。

Issue #35 の理論監査で新たに一次確認されたこと:

1. 現行基準は `BIC_impl = −2·Q_strict + p̂·ln n`。当てはまり項は **EM の Q 関数**であり観測データ周辺尤度ではない。
2. `scale_Z` により潜在変数の事前分布項が定数 `−(nk/2)(1+ln2π)` に退化し、
   **潜在次元 1 あたり `n(1+ln2π) ≈ 2.84n` の固定的な次元罰則**が実効的に働いている。
3. **先行研究の印刷された Eq.(26) の当てはまり項 `ln L`（Eq.16）も `Z` を積分していない**
   （`z_i` に条件づけた量。2026-08-23 に原論文 PDF を一次確認）。
   → **本研究の基準も先行研究の印刷された基準も、Schwarz BIC が対象とする観測データ周辺尤度には対応しない。**

**新たに書いてはいけないこと（KI-010 に追加）:**

- 「先行研究の Eq.(26) は standard Schwarz BIC である」
- 「先行研究の BIC は joint モデルの `Q` を使っている」（一次確認により否定された）
- 診断スコア `S_cf` を「corrected BIC」「modified BIC」「true BIC」と呼ぶこと
- `S_laplace_post` を「ELBO」「variational BIC」と呼ぶこと

**K 選択は現在も UNRESOLVED である。** selection target が未確定（#35 U16）、
K 選択の n 依存性は一度も測定されていない（U2）、本モデルの RLCT は未知（U5）。
Issue #37 は**同一の 42 fits 上で score 定義により選ばれる K が異なる**ことを示した
（C1 `bic_impl` と C3 `S_laplace_post` は k=3 を 6/6、C2 `S_cf` は範囲上端 k=7 を 6/6）。

### E. 新規 KI-016 — sparse/dense Y boundary と latent-coverage の解釈境界

**sparse/dense Y boundary（Issue #27、lineage E+F、人工データ）:**
`y_obs_rate=0.1` では 4 つの primary contrast すべてで複数属性同時利用が優位
（`y_only` +0.5122 10/10、`single_bernoulli` +0.4218 10/10、`single_poisson` +0.3889 10/10、
`single_gaussian` +0.2030 9/10）。`y_obs_rate=1.0` では最良単一 block 比 **+0.0087** まで縮小。

**書いてはいけない:** 「一般に per-column が優れる」「実データでも優れる」
**「dense-Y では無意味／実務的に無意味」**（縮小したという記述までにとどめる）
「正式提案手法として確立した」。

**latent-coverage の解釈境界（Issue #31、lineage E+F、人工データ）:**
primary（Gaussian comparator）は `I = +0.1139`（std 0.0915、9/10）だが、
必須分解の `D_J = +0.2012`（10/10）が大きく、**`I` を単独で解釈してはいけない**。
secondary は追随しない: `single_bernoulli` の `I = −0.2041`（**0/10**）、`single_poisson` は `−0.0046`（6/10）。

**書いてはいけない:** 「latent coverage が改善原因である」「機構を分離した」「alone / fully isolated」。

### F. 新規 KI-017 — Phase 7e の結果境界と stdout provenance limitation

**Phase 7e は実行済みである**（Issue #43 close 済み、PR #44 merge 済み、main `ec6e646c...`）。
「未実行」「planned next experiment」と書かない。

**結果（lineage E、objective-consistent experimental prototype、本文採用不可）:**
selected K = replicate1:**3** / replicate2:**3** / replicate3:**5**、counts `{3:2, 5:1}`、
**descriptive recovery rate 2/3**。42/42 clean。

**書いてはいけない:** 「true K recovery 66.7% という一般性能」「consistency」「BIC より優れる」
「C1/C2/C3 より優れる」「manuscript-level conclusion」「実データ妥当性」「漸近的結果」。
これは **1 synthetic setting × 3 dataset replicate だけの descriptive pilot** である。

**stdout provenance limitation:**
`stdout.log` を生成した outer capture command は
**`NOT RECOVERABLE FROM REPOSITORY EVIDENCE`** である（runner 自身に write 処理がなく、
`runinfo.json` の `command` は inner Python command のみを記録している）。
repository evidence から言えるのは、frozen RUN_CODE_SHA の後に
**42 clean fits からなる 1 successful recorded execution が保存 artifact として存在する**ことまで。

**書いてはいけない:** 「削除された先行実行が存在しないことまで含めて externally proven exactly once」。
この限定は 42 saved fit rows・selected K・arithmetic・seed・hash・leakage isolation を無効化しない。
詳細は `reports/k_selection_theory/heldout_k_selection_full_pilot_provenance_addendum_20260831.md`。

### G. 新規 KI-018 — raw-count Poisson diagnostic の原因は未解明

**事実（Issue #33、lineage E+F、実データ）:** secondary contrast
`mixed_train_raw_poisson − mixed_train_log` は mean **−0.100274**、**29/30 splits で悪化**。
leakage を除いた user-disjoint protocol 上でも、旧 leaky pair-split diagnostic と同じ方向を示した。

**書いてよい:** 「今回の MovieLens 設定・モデル・raw-count 表現では、
raw count を Poisson-X として扱った条件が log-count 表現より一貫して悪化した」。

**書いてはいけない:** 「Poisson は悪い」「Poisson は実データに不適」「Poisson-X は一般に有害」
**「intercept 欠如が原因」「curvature が原因」**。

**原因は UNRESOLVED。** Issue #28 §9.3 の通り、intercept 欠如 / raw scale / Poisson 曲率 /
X 側過分散（`ratings_count` の var/mean = 6.17）の 4 要因が交絡しており、
これらを分離できる設計は #28 の候補 B のみである。B は X 列 intercept と
dispersion-aware count family を伴う設計を要する。**現時点でそのような条件は存在せず、原因は未解明のままである。**

### H. 追加された「まだ主張してはいけないこと」（上のリストへの append）

上の「まだ主張してはいけないこと」に、2026-08-31 時点で次を追加する（既存行は削除しない）。

- 「per-column heterogeneous-X が一般に優れる」「実データでも優れる」「正式提案手法として採用した」（B/E/F、prototype）
- 「MovieLens で提案手法の有効性を確認した」「statistically significant」「robust superiority」
  「30 independent experiments」「causal contribution」（Issue #33 report §9 の禁止句リスト）
- 「Poisson は実データに不適」「Poisson-X は一般に有害」「raw-count 悪化の原因は intercept 欠如／curvature である」（KI-018）
- 「先行研究の Eq.(26) は standard Schwarz BIC である」（KI-010 forward update）
- 「K-selection の consistency を確認した」「BIC で正しい K を選べる」「true K を一般に回復する」（KI-010 forward update）
- 「Phase 7e の 2/3 は true K recovery 66.7% という一般性能である」（KI-017）
- 「Phase 7e は externally proven exactly once である（削除された先行実行の不存在を含めて）」（KI-017）
- 「latent coverage が per-column の改善原因である」（KI-016）
- 「legacy numerics で実行された per-column の過去結果は clip の影響を受けていない」（KI-015）
- 旧 0.5 系列（B）・fixed 系列（C）・consistent 系列（E）の数値を同じ表・図に並べること（KI-002 の拡張）

### I. 修論フェーズで優先的に検証すること（上のリストへの forward note）

上のリスト（項目 1〜8）は 2026-05〜2026-06 時点のものであり、**削除しない**。
2026-08-31 時点の状態は次のとおり。

| 旧項目 | 現在の状態 |
|---|---|
| 1. KI-001 fixed 版での再実行 | **実行済み**（`fixed_official/`）。残るのは条件対応表の作成（§15、RESEARCH_MASTER） |
| 2. KI-003 fixed 版での倍率計算 | **実行済み**（A 4.3414× / B 9.0405× / C 40.3706×）。**条件対応表は未作成**（human decision） |
| 3. KI-006 Wine | **完了** |
| 4. KI-010 パラメータ数の手計算 | **完了**（#35 E1、18 セルで `num_params` 一致を確認）。ただし基準の位置づけ問題は別途 UNRESOLVED |
| 5. KI-011 疎データでのペナルティ | **未解決**。`Σ_i ln det A_i` 未測定 |
| 6. KI-012 pair mask / NB | **部分解消**（上記 C）。fixed 本体 API への統合は未実施 |
| 7. Cora full への拡張 | **未実施** |
| 8. MovieLens 他 projection | **未実施**。ただし #33 で user-disjoint protocol は確立した |

本表は各項目の **現在の状態** を記録したものであり、実施計画ではない
（研究上の TODO は GitHub Issue で管理する。root `CLAUDE.md` §8）。

---

## 2026-09-04 forward update（Phase 8b Attempt 2 の統合）

### J. 新規 KI-019 — 2 つの K 選択基準が併存し、混同されやすい

**重要度: 高**

**事実:** 本リポジトリには**別物の K 選択基準が 2 つ**存在する。

| 基準 | 何を使うか | どこで使われているか |
|---|---|---|
| **legacy Q ベース基準** | `Q_strict`（EM の Q 関数の MC 近似）。観測データの周辺尤度ではない | `expfam/src/utils_expfam.py` の `calc_bic_dual`、fixed 系列の `exp1` 系 K 選択、Wine / Cora 実データの「BIC 最小 k」 |
| **frozen held-out 予測スコア** | held-out Bernoulli raw-eta plug-in mean log score（`y·eta − logaddexp(0,eta)` を held-out upper-triangle test pair 上で平均）、2 start の非加重平均、tie tolerance 1e-12、smallest-K tie rule | **Phase 7e**（`heldout_full_pilot_20260824/`）、**Phase 8b**（`k_true_robustness_full_attempt2_20260904/`） |

**影響:** root `CLAUDE.md` §5 と KI-010 の「モデル選択基準を Schwarz BIC と呼ばない／
Q-based complete-data criterion・ICL-type として扱う」という限定は、**前者（legacy 基準）
にのみ適用される**。これを Phase 7e / 8b の結果に適用すると、**実際には使われていない基準を
使ったと書く**ことになり、事実として誤った記述になる。**2026-09-04 に実際にこの誤記が
生成された**（Phase 8b Attempt 2 の実行直後の口頭サマリで、held-out 予測スコアによる選択を
`Q_strict` ベースの基準として説明した）。逆向きの誤り（legacy の `calc_bic_dual` の結果を
「held-out 予測スコア」と呼ぶ）も同様に誤りである。

**まだ主張してはいけないこと（追加）:**

- 「Phase 7e / Phase 8b の K 選択は `Q_strict` / EM の Q 関数基準 / ICL-type
  complete-data criterion / Schwarz BIC / marginal likelihood で行った」（**事実として誤り**）
- 「`calc_bic_dual` による k 選択は held-out 予測スコアによる選択である」（**逆向きの誤り**）
- 2 つの基準の結果を、基準名を明記せずに同じ表・図・文で並べること

**運用ルール:** K 選択の数値を引用するときは、**必ずどちらの基準か**を明記する
（KI-002 の lineage 明記と同じ運用）。判定方法は次のとおり。

- artifact に `score_config_hash` / `heldout_mean_log_score` / `selection_matrix.csv` が
  あれば **held-out 予測スコア**（Phase 7e / 8b）
- artifact の列名が `BIC` で `calc_bic_dual` 由来であれば **legacy Q ベース基準**（KI-010）

**関連ファイル:** `expfam/src/utils_expfam.py`,
`tools/research_audit/run_heldout_k_selection_pilot.py`（`FrozenScoreConfig` /
`select_k_from_two_starts`）, `tools/research_audit/run_k_true_robustness_sweep.py`,
`RESEARCH_MASTER.md` §12.6 / §12.7 / §12.9,
`reports/k_selection_theory/k_true_robustness_full_report_20260904.md` §2

**KI-010 との関係:** KI-010 は**取り消さない**。KI-010 は legacy 基準についての記述として
引き続き有効であり、KI-019 はその適用範囲を明示するものである。

### K. Phase 8b Attempt 1 の位置づけ（記録）

`expfam/results/k_selection/k_true_robustness_full_20260902/` は
**ABORTED_BY_OPERATOR_INTERRUPT / provenance only / no scientific use** である
（`status = FAILED`、`attempted_fit_count = 3`、`clean_fit_calls = 2`、`scored_rows = 0`、
`retry_count = 0`、`replacement_fits_executed = 0`）。

- この 2 clean fits を科学的主張の根拠にしない。
- Attempt 2 は本結果を一切再利用していない（`partial_results_reused = False`）。
- **artifact を削除・改変しない。**

### L. 追加された「まだ主張してはいけないこと」（2026-09-04）

上のリストに次を追加する（既存行は削除しない）。

- 「Phase 7e / 8b の K 選択は `Q_strict` / ICL-type / Schwarz BIC / marginal likelihood で
  行った」（KI-019）
- 「Phase 8b は K-selection consistency / asymptotic consistency / universal K recovery を
  示した」（各条件 3 replicate の有限標本記述値。RESEARCH_MASTER §12.9）
- 「Phase 8b の `K_TRUE=3` について A と B が独立に 6 セル分の証拠を与える」
  （同一 Phase 7e anchor 42 fits の READ-ONLY 共有参照）
- 「Phase 8b の統合証拠は 420 fits である」（336 新規 + 42 anchor = **378**）
- 「Phase 8b の `K_TRUE=5` での under-selection の原因を特定した」（原因は UNRESOLVED）
- Attempt 1（`k_true_robustness_full_20260902/`）の部分結果を科学的根拠にすること

---

## 2026-09-04 forward update（true-K identifiability 監査の統合）

根拠: `reports/identifiability/true_k_identifiability_hardened_20260904.md`
（独立敵対レビュー済み: `true_k_identifiability_review_20260904.md`）。
数値確認: `tools/research_audit/verify_identifiability_identities.py`。

### M. 新規 KI-020 — canonical Poisson-Y はモーメントが存在しない領域を持ち、historical default `w=0.5` はその境界にある

**重要度: 高**

**事実（`[DERIVED]`、`[CONFIRMED_IN_REPOSITORY]`）:** canonical（clip なし）Poisson-Y

```
lambda_ij = exp(w0 + w z_i^T z_j),   Y_ij | Z ~ Poisson(lambda_ij)
```

について `E[lambda^r] = exp(r w0) (1 − r² w²)^{−K/2}` であり、

```
E[Y^r] < infinity   <=>   |w| < 1/r
```

すなわち **平均が有限なのは `|w| < 1`、分散が有限なのは `|w| < 1/2`**。

`expfam/src/data_generator_expfam.py` の `_Y_DEFAULTS["poisson"]` は `w0=0.0, w=0.5` であり、
これは **`|2w| = 1` ちょうど、分散発散の境界そのもの**である。

**影響:**

- `w = 0.5` の canonical Poisson-Y では **population の分散が存在しない**。
  標本分散は `n` を増やしても収束せず、二乗誤差・相関・CLT 的な議論の前提が成立しない。
- **properness の問題ではない。** 分布は proper でサンプルも a.s. 有限である。
  発散するのは population のモーメントであって実現値ではない。

**誤って書いてはいけないこと（敵対レビューで訂正した誤り）:**

- 「clip があるから historical Poisson-Y のデータが有限である」— **誤り**。
  実現値は clip の有無によらず a.s. 有限。加えて `w0=0, w=0.5` では `P(η > 10)` が
  10⁻⁷ オーダーで **clip はほぼ発動しない**。
- 「historical Poisson-Y のデータは分散無限の分布からのサンプルである」— **これも言えない**。
  historical generator は `Z` を列 z-score しているので `‖z_j‖² ~ χ²_K` が成立せず（KI-021 G1）、
  上の公式自体が historical データの分布には適用できない。**KI-020 は canonical model についての主張である。**

**運用ルール:** canonical Poisson-Y を使う新規実験では **`|w| < 1/2` を generator gate として強制する**。
`expfam/src/experimental/data_generator_canonical.py` は既定でこれを強制し、
`allow_infinite_variance=True` を明示したときだけ通す。

**関連ファイル:** `expfam/src/data_generator_expfam.py`（`_Y_DEFAULTS`）,
`expfam/src/experimental/data_generator_canonical.py`,
`reports/identifiability/true_k_identifiability_hardened_20260904.md` §11

### N. 新規 KI-021 — historical synthetic generator は canonical model の literal generator ではない

**重要度: 高**

**事実（`[CONFIRMED_IN_REPOSITORY]`、`expfam/src/data_generator_expfam.py` を直接読んで確認）:**

| ID | 箇所 | 実装 | canonical model との差 |
|---|---|---|---|
| G1 | L.122-123 / L.282-283 | `Z = normalize_zscore(Z, axis=0)` | **`Z` は iid `N(0,I_K)` ではない**。列標本平均 0・標本 SD 1 に強制されるため行間に依存が入る。`‖z_j‖² ~ χ²_K` も成立しない |
| G2 | L.127-129 / L.288-290 | `F[i,:] = F[i,:]/‖F[i,:]‖ · sqrt(1−σ_ii)` | **`F` は自由パラメータでない**。全行が `‖f_l‖² = 1 − uniq`（既定 0.9）に固定 |
| G3 | L.132-133 / L.297-298 | `X = Z @ F.T + noise; X = normalize_zscore(X, axis=0)` | **Gaussian-X は `N(Fz, Σ)` ではない**。返り値の `F`・`sigma` と最終的な `X` が literal に対応しない |
| G4 | L.80 / L.304 / L.319 | `np.clip(eta, -20, 10)` | Poisson の hard clip。`λ ≤ e^10` に silent に切られる |
| G5 | L.233 | `sigma_x_true: float = 0.1` | **宣言されているが一度も使われない**。Gaussian-X の雑音共分散は `uniq` から作られる |
| G6 | Bernoulli 各所 | `np.clip(eta, -500, 500)` | **数値的に無害**。`sigmoid(±500)` は倍精度で既に飽和しておりモデルを変えない。G4 と同列に扱わない |
| G7 | Gaussian-Y | `rng.normal(0.0, sigma_y_true, ...)` | numpy の第 2 引数は標準偏差。`CLAUDE.md` の規約と整合。**問題なし** |

**影響:**

- **historical の数値結果を無効化するものではない。** それらは「当該 generator が生成したデータ上での観測」として有効である。
- 言えなくなるのは **「canonical model から well-specified に生成したデータで検証した」という強い読み方**だけである。
  G1・G3 により、推定器が仮定するモデルと生成過程が一致していない（mild misspecification）。
- とくに **Poisson-X の識別可能性（`FF^T` のモーメント復元）は G1・G4 の下では前提が崩れる**。
  Phase 7e / Phase 8b はいずれも `family_x=poisson` で `generate_dual_data` を使っているため、
  **当該実験の well-specification はこの識別可能性命題によって保証されない。**
- **G5 は API の不整合**である。`sigma_x_true` を明示指定した過去の呼び出しがあれば、
  その意図と実際の生成条件が食い違っている。網羅確認は未実施（`[UNRESOLVED]`）。

**運用ルール:**

- **historical generator は変更しない。** forward-only の別モジュール
  `expfam/src/experimental/data_generator_canonical.py`（`generator_version = canonical-clean-v1`）を使う。
- historical generator 由来の結果と canonical clean generator 由来の結果を
  **同じ表・図に混在させない**（KI-002 と同じ運用）。
- 生成器を明記して引用する。判定方法: artifact の `generator_version` が
  `canonical-clean-v1` なら clean、記録がなければ historical。

**関連ファイル:** `expfam/src/data_generator_expfam.py`,
`expfam/src/experimental/data_generator_canonical.py`,
`reports/identifiability/canonical_clean_generator_spec_20260904.md`,
`reports/identifiability/true_k_identifiability_hardened_20260904.md` §13

### O. KI-010 / KI-019 の forward update — BIC が使えない理由は「非入れ子」ではない

**KI-010 も KI-019 も取り消さない。** 次を追記する。

Gaussian-Y（`w ≠ 0`）については `M_K ⊄ M_{K+1}`（モデル族が入れ子でない）ことを証明した。
**しかし非入れ子が無効にするのは尤度比検定の χ² 近似と Wilks の定理であって、
Schwarz BIC の導出ではない**（BIC はモデルごとの Laplace 近似であり、
非入れ子モデルの比較に使うのは標準的である）。

**Schwarz BIC がこのモデルで正当化されない実際の理由は次の 3 つである** `[DERIVED]`:

1. **潜在変数モデルが特異である。** `O(K)` 回転不変性により Fisher 情報が退化し、
   `rank(F) < K` や `w = 0` の縮退集合上でも退化する。
   特異モデルでは `(p/2) log n` の罰則が RLCT に置き換わるが、**本モデルの RLCT は未知**
   （`RESEARCH_MASTER.md` §14 U5）。
2. **境界パラメータ**（`Σ_X ⪰ 0`、`σ_y² ≥ 0`、因子分析の `Ψ ⪰ 0`）。
3. **有効標本数が未定義。** ノード数 `n` / dyad 数 `n(n−1)/2` / X 要素数 `nd` の 3 通りがあり、
   `calc_bic_dual` は `log n` に**ノード数**を使う。さらに潜在変数 `Z ∈ R^{n×K}` は
   `n` とともに増える incidental parameter であり、`Q_strict` は `Z` を完全データとして扱う一方
   罰則 `p̂` は `Z` を数えない。

**追加で記録する実装事実** `[CONFIRMED_IN_REPOSITORY]`（`expfam/src/utils_expfam.py`）:
`num_params = kd − k(k−1)/2 + [d if X gaussian] + [1 if Y gaussian]`。
`−k(k−1)/2` は `O(K)` 軌道の次元を既に引いている。
**`w0` と `w` は数えられていない**（NOLTA 2024 の慣行）。
Gaussian-Y では `K` の情報をもっぱら `w` が運ぶことが判明したため、
この扱いは K 選択の文脈では検討に値する。ただし `K` に依存しない定数なので**順位には影響しない**
（`RESEARCH_MASTER.md` §12.6 の既存判断と整合）。**「誤り」とは断定しない。**

### P. 追加された「まだ主張してはいけないこと」（2026-09-04、identifiability 監査）

- 「Poisson-X なら常に K が識別可能」（unclipped link・`rank(F)=K` の generic 条件が要る。KI-021）
- 「Bernoulli では K は識別できない」（反例は `d=1` **かつ X 周辺のみ**。joint の反例ではない）
- 「`w` の符号は識別されない」（単一 dyad 限定。**三角形からは識別される**）
- 「`M_K` と `M_{K+1}` は互いに素」（`w=0` 切片で交わる）
- 「BIC は非入れ子だから使えない」（**理由が誤り**。上記 O）
- 「clip があるから historical Poisson-Y のデータが有限である」（KI-020）
- 「held-out score は `K*` を推定している」**および**「していない」（どちらも未証明）
- 「人工データの `K_TRUE` が真の潜在次元 `K*` である」（`K* ≤ K_TRUE` で等号は自明でない）
- 「実データにおいて `K*` を推定している」（M-closed 仮定が成立しない）
- 「K 選択の一致性を証明した」「n を増やせば必ず `K_TRUE` に収束する」（未解決のまま）
