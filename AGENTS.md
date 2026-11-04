# AGENTS.md — Dual-ExpFam LSM 研究指示（Claude Code / Codex 共通の正本）

Claude Code と Codex に共通する研究指示の唯一の正本。Codex は直接読み、Claude Code は root `CLAUDE.md` の import で読む。
既存文書・コード中の「`CLAUDE.md` §n」は本ファイルの §n を指す（§1 モデル・§2 1/2・§3 実装系列・§4 source priority・§5 主張・§6 Human Gate・§7 安全・§8 参照）。

## 1. 研究の目的と基本モデル

属性付き関係データの潜在構造モデル（Mikawa et al. 2024）を、X 側・Y 側で独立に指数型分布族（Gaussian / Bernoulli / Poisson）へ一般化し、
人工データと実データでその挙動・有効条件・限界を評価する。一般化が有効であることは前提にしない。これに反する式を書かない。

```
z_i  ~ N(0, I_k)
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_i^T z_j )   i < j
x_il ~ ExpFam_X( η_il^X = f_l^T z_i )               バイアスなし
```

- `w_0^Y, w^Y ∈ R` は **スカラー**（行列 W_Y ではない）。
- X の尤度は列 `l` ごとに因子分解する。ただし標準の `DualExpFamLSM` では **`family_x` は全 X 列で共通の 1 種類**。
  列ごとの family（`family_x_list`）は `experimental/model_dual_expfam_percolumn.py` だけにあり、**prototype・本文採用不可**。
- θ = { F, w_0^Y, w^Y } ＋ Gaussian-X のとき Σ_X（対角）／＋ Gaussian-Y のとき σ_Y²（M-step で MLE、`calc_sigma_y`）。
  数理上は `σ_Y²` を dispersion として扱うが、実装は標準偏差を `self.sigma_y` に保持し使用時に二乗する（`self.sigma_y ** 2`）。
- E-step では分散パラメータ φ を落とさない：`V_Y(η) = A_Y''(η) / φ_Y`（φ_Y = 1：Bernoulli/Poisson、σ_Y²：Gaussian）。
  勾配・精度行列 `A_i`・`V_X` の確定式は `RESEARCH_MASTER.md` §6。数式とコードの対応は §7 実装対応表と実コードで確認する。

## 2. 数式の制約：1/2 係数（5 系統を混同しない）

| 系統 | 1/2 |
|---|:---:|
| **Mikawa et al. 2024 の印刷された原論文式**（Eq.19/20/22/23、Appendix A-1/A-3/A-5） | **あり** |
| old 0.5 Python 系列（`model_expfam.py` / `model_dual_expfam.py`） | あり |
| **本研究の独立再導出・採用式**（unique undirected-pair conditional） | **extra 1/2 なし** |
| fixed Python 系列（`model_dual_expfam_fixed.py`） | なし |
| MATLAB `calcAi` | なし |

- **原論文の印刷式には 1/2 がある（2026-08-18 に原論文を直接確認）。「原論文にも 1/2 がない」と書かない。** 採用式との差は独立な再導出に基づく意図的な差であり、
  MATLAB は補助的な比較にとどめる。一次確認 `docs/math_notes/half_factor_primary_source_confirmation_20260818.md`、
  導出 `docs/math_notes/half_factor_math_explanation.md`、詳細 `KNOWN_ISSUES.md` KI-001。
- 既存の数式を独自判断で訂正・変更しない。矛盾を見つけたら根拠（式番号・ファイル・行）とともに報告する。

## 3. 実装系列（結果を混ぜない）

```
reproduction/src/model.py                       LatentStructuralModel（先行研究 Python 再現）
└ expfam/src/model_expfam.py                    0.5 あり
  └ expfam/src/model_dual_expfam.py             0.5 あり ← 学会予稿の本文採用実験
    └ expfam/src/model_dual_expfam_fixed.py     0.5 なし ← 実データ実験フェーズ
      └ expfam/src/experimental/model_dual_expfam_masked.py
        ├ _nb.py    └ _percolumn.py             prototype・本文採用不可
```

- 数値には必ず系列を付け、異なる系列の結果を同じ表・図に混在させない（KI-002）。
- prototype を正式手法・manuscript evidence として扱わない（昇格は §6 の Human Gate）。
- 未マージの PR・作業ブランチ上の結果を、`main` に採用済みの結果として扱わない。

## 4. 一次証拠と再現性

1. 一次データ — 結果 CSV・runinfo・実行ログ・実コード・原論文 PDF（`paper/A_study_on_latent_structural_models_for_binary_rel.pdf`）
2. canonical docs — `RESEARCH_MASTER.md` / `KNOWN_ISSUES.md` / `EXPERIMENT_REGISTRY.md` / 本ファイル（現状はそれぞれ最新の日付入り forward update・Claim Ledger 節を正とする）
3. 日付入りで凍結された `reports/<phase>/` — 当時の記録として読み、書き換えない
4. 参考のみ — `docs_for_notebooklm/*`、`GEMINI_REPORT_*`、`expfam/CLAUDE.md`、`expfam/handoff.md`。AI 生成・派生資料を一次根拠にしない（KI-007）

- 数値主張は必ず 1 に遡る。新しい run は seed・条件・実装系列・git head・環境を記録する（runinfo、KI-014）。
- FACT（一次資料で確認）/ DERIVED（数式的導出）/ OBSERVED（実験結果）/ INTERPRETATION / HYPOTHESIS を区別して書く。

## 5. 実験条件・評価指標・結果の解釈

- 優先順位は scientific validity → evidence integrity / provenance → reproducibility → engineering quality。
- 実験前に、研究問い・主要指標・比較・固定条件と変える条件・seed/replicate・保存先を明確にする。
- **technical validity（run が壊れていないか・protocol を守ったか）と scientific outcome を分ける。**
  悪化・差なし・不安定性も valid run なら有効な研究結果であり、FAIL ではない。良い seed・条件だけを選ばない。
- 結果を見た後に threshold・seed・comparator・metric を動かして同じ run を救わない。exploratory な発見を confirmatory evidence にしない。
  synthetic で成功しただけで real-data の有効性・一般性を主張しない。
- **モデル選択基準を「Schwarz BIC」と呼ばない。** `calc_bic_dual` は観測データの周辺尤度ではなく `Q_strict`（Q 関数の MC 近似）で、
  Q-based complete-data criterion / ICL-type として扱う（`reports/theory_audit/theory_audit_report_20260718.md` §6-7、KI-010）。関数名・CSV 列名 `BIC`・過去の呼称は変更しない。
- 採用式（1/2 なし）を正とする。本文採用実験は 0.5 あり実装で実行されており、0.5 は Y 側項のみで Z 事前分布項・X 側項には掛かっていない。
  そのため 0.5 系列の結果には「Newton 方向が全体として正しいとは断定できない」を必ず付記する（KI-001）。
- Scen.C の「Y=Gaussian が支配」は Exp4 ablation からの推測。誤指定倍率 23.6× / 41.45× / 38.97× は別系列・別条件の値（KI-003）。
- `KNOWN_ISSUES.md`「まだ主張してはいけないこと」を報告・原稿案に書かない。
- 実験設計・gate・severity・数値的な判断の詳細は `docs/research_operating_policy.md`。

## 6. 研究タスクの進め方と Human Gate

**依頼された種類と範囲の作業に直接答え、完了させる。** 別の課題や改善点に気づいても scope を広げず、報告で一言触れる。

- **理論・数式の調査**: 仮定・導出・実装の対応を調べ、重要な問題は根拠付きで報告する。研究テーマを勝手に変えない。
- **実装**: 指定の手法を実装し、関連 test を実行して完了する。無関係なリファクタリング・追加監査を始めない。
- **実験**: 承認済みの範囲では再承認を求めずに実行し、一次結果を再現可能な形で保存する。
- **結果分析**: 観測・成立条件・可能な解釈を、改善・悪化・差なし・不安定性を公平に扱って整理する。
- **研究方針の相談**: 選択肢と長所・短所・根拠を示すが、主題・モデル・主要指標・次の実験は確定しない。

**既定では作らない**（明示的な依頼、または指定された科学的成果物・再現性の確保に必要な場合を除く）：
修論の本文・章立て・タイトル案、指導教員向け資料、研究方針の提案文書、新しい MD レポート、類似の GitHub Issue・PR、
監査 phase、実験ごとの独立監査ツール、科学的根拠のない厳格な数値閾値、同じ問題の無制限な再検証。

作業の終わりにはチャットで **何を検証したか / 主要な数値結果 / 何が分かったか / 何がまだ言えないか / 次に人間が判断すること** を報告する。
この報告のためだけに MD を作らない。WARNING / DIAGNOSTIC だけで進行を止めない。

**Human Gate — 停止して判断を求める**：研究目的・モデル・数式の変更／family・K・主要評価指標・実験条件の変更／
新たな研究実験の開始や結果に基づく次の実験の決定／frozen spec の変更／prototype の正式手法・manuscript evidence への昇格／
重大な問題で結果が解釈不能（BLOCKER）／未承認の破壊的操作。
**Issue close・PR merge・force push・published history rewrite・main への直接変更は agent が行わず、人間自身が実行する。**

**Approved Task**: ユーザーが目的と scope を明示した依頼はそれ自体が承認であり、その範囲の調査・実装・test・承認済み実験・
承認済み script による artifact 生成・provenance 記録・CI 確認と scope 内の修正は個別の再確認なしに進める。working branch での commit / normal push・
Draft PR は、依頼に含まれる場合か承認済みの成果を保存するのに必要な場合に行う。完了したら結果を返し、次の phase を自動で始めない。

## 7. Git と研究成果物の取り扱い

- main を直接編集しない。`git switch -c <type>/<issue#>-<slug>`（`experiment/` `audit/` `maintenance/` `docs/`）で作業する。
- 結果 CSV・図はスクリプト経由でのみ生成し、手で編集しない。過去の CSV / runinfo を書き換えない（事後の推測による追記も禁止）。
- 既存の成果物を上書きしない（再生成は scope に明示された場合のみ）。新しい run は新しい出力先に書く。
- `EXPERIMENT_REGISTRY.md` は追記のみ。既存行とそのパス文字列を書き換えない・削除しない。
- 過去 artifact・frozen protocol・historical verdict は遡及変更しない。
- 実行環境の基準は `.python-version`（3.13.14）と `requirements*.txt`。過去実験の環境を再現するものではない（KI-014）。
- 目的と scope が明示されていないコード修正・実験再実行・ファイル移動/削除は、実行前に確認する。

## 8. 参照先（必要なときだけ読む）

`README.md`（入口・環境構築）／`RESEARCH_MASTER.md`（研究内容の正本・確定式 §6・実装対応 §7・Claim Ledger）／
`KNOWN_ISSUES.md`（事故台帳・主張してよいこと/いけないこと）／`EXPERIMENT_REGISTRY.md`（実験→スクリプト→CSV→主張）／
`docs/research_operating_policy.md`（詳細研究規約）／`docs/math_notes/`（数式ノート）／`conference_submission_final_draft.md`（学会予稿・変更しない）／
`reports/environment/baseline_20260818.md`（実行環境）。TODO は GitHub Issue で管理し、ここにも canonical docs にも書かない。
