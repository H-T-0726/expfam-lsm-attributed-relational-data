# NotebookLM投入前 最終検証レポート

**作成日:** 2026-05-18  
**対象ファイル:** `docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md`  
**方法:** 全節通読 + 10項目チェックリスト照合

---

## 1. 検証結果サマリー

**10項目全て合格。実質的な問題なし。**  
微修正1箇所（表現の統一）のみ実施。数式・数値・研究主張の変更なし。

| # | 確認項目 | 判定 | 根拠 |
|---|---------|:---:|------|
| 1 | 既存研究と本研究が混同されていないか | ✓ | §5（NOLTA 2024）と §6（Dual-ExpFam）を明確に分離 |
| 2 | 4資料（NOLTA PDF・MATLAB・原稿式・Python実装）の違いが正しく分けられているか | ✓ | §5.5 と §14.1 の対照表で整理済み |
| 3 | 1/2係数問題で「Newton方向は正しい」と断定していないか | ✓ | §9.4「posterior 全体の Newton 方向が正しいとは断定できない」と明記 |
| 4 | E-step 1/2、M-step /2L、Q関数 0.5 の3種を混同していないか | ✓ | §7.8（E-step：1/2なし）・§7.10（M-step /2L：正しい）・§9.4（Python実装のM-step, Q関数は正しい）で分離 |
| 5 | 実験結果の数値がPhase 1B CSV照合結果と一致しているか | ✓ | 全数値が照合済み表と一致（§11.1〜11.4） |
| 6 | 図1(b)の23.6倍と全条件最大の41.5倍が混同されていないか | ✓ | §11.6 で両値を表にして明確に分離、§13.4 で「図に41.5×のバーなし」と再確認 |
| 7 | Scenario Cを「支配的」と断定せず「強く寄与した可能性」と書いているか | ✓（修正後） | §11.7「強く寄与した可能性がある」・§14.4（修正済み） |
| 8 | BICを「全試行でk=3」と書かず「10試行平均BICでk=3」と書いているか | ✓ | §11.5・§12 ともに「10試行平均BIC」と明記 |
| 9 | `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` が矛盾資料として扱われているか | ✓ | §14.1 で「中間段階の矛盾資料（採用不可）」と明記、§16 で投入注意ファイルに分類 |
| 10 | 推奨ファイル・投入注意ファイルが明確に分かれているか | ✓ | §15（推奨9ファイル）・§16（注意5ファイル）で一覧化 |

---

## 2. 実施した修正（微修正のみ）

### 修正1（表現の統一）

**場所:** §14.4 Scenario C の X-mismatch

**変更前:** 「Y=Gaussian が Z 推定を支配している可能性」が有力だが断定できない  
**変更後:** 「Y=Gaussian が Z 推定に強く寄与した可能性」が有力だが断定できない

**理由:** §11.7 の表現「Z 推定に強く寄与した可能性がある」と統一し、
「支配」という語の強さを避けるため。数式・数値・主張への影響なし。

---

## 3. 各節のチェック詳細

### §5 既存研究 NOLTA 2024 — 問題なし

- §5.1: 出典（著者・論文名・巻号・ページ）が正確に記載されている
- §5.2: Y=Bernoulli、X=Gaussian に固定と明記。バイアスなしも明記
- §5.3: MCEM + Laplace 近似 + BIC の枠組みが正確
- §5.5: NOLTA PDF Eq.(22)(23) に 1/2 あり・MATLAB に 1/2 なしという内部矛盾が正確に記載
- 本研究の採用方針（再導出 + MATLAB 確認で 1/2 なし）が明記されている

### §7 数式モデル — 問題なし

- §7.8 E-step 勾配 Term3: $w^Y \sum_{j \neq i}$（1/2 なし）✓
- §7.9 精度行列 Term3: $(w^Y)^2 \sum_{j \neq i}$（1/2 なし）✓
- §7.10 M-step /2L: $\frac{1}{2L}\sum_l\sum_{i\neq j}$（正しい）✓
- 各 Term が何の寄与か（Z事前・X尤度・Y尤度）が明記されている
- $\sigma_z^2 = 1$ 固定が明記されている
- $w_0^Y, w^Y$ がスカラーであることが明記されている

### §9 実装との対応 — 問題なし

- クラス継承の3層構造が正確（LatentStructuralModel → ExpFamLatentStructuralModel → DualExpFamLSM）
- DualExpFamLSM が `_calc_gradient` と `_calc_precision_matrix` をオーバーライドと明記
- Python 実装に残る `0.5 *` の行番号が正確（L.159, L.200）
- M-step /2L と Q 関数 Y 側 `0.5 * sum(log p)` が正しいことを明記

### §14 矛盾・注意点 — 問題なし（修正後）

- §14.1 の対照表: 4 資料の比較が正確
- §14.2: Python 実装の行番号が正確、修正方針が明確
- §14.3: 23.6× と 41.5× の由来が明確
- §14.4: 修正後、§11.7 と表現が統一された

---

## 4. 数値照合の最終確認

以下の数値を Phase 1B の CSV 照合結果と再照合：

| 値 | BRIEF記載 | Phase 1B 確認値 | 一致 |
|---|----------|--------------|:---:|
| RMSE(Z) k=3, A | 0.2784 | 0.2784 | ✓ |
| RMSE(Z) k=3, B | 0.1817 | 0.1817 | ✓ |
| RMSE(Z) k=3, C | 0.0284 | 0.0284 | ✓ |
| n=50 RMSE(Z), A | 0.4056 | 0.4056 | ✓ |
| n=300 RMSE(Z), A | 0.2076 | 0.2076 | ✓ |
| 削減率 A | 48.8% | 48.8% | ✓ |
| 削減率 B | 31.0% | 31.0% | ✓ |
| 削減率 C | 62.0% | 62.0% | ✓ |
| mismatch最大 A | 3.41× | 3.406× | ✓（丸め一致） |
| mismatch最大 B | 7.35× | 7.345× | ✓（丸め一致） |
| mismatch最大 C | 41.5× | 41.45× | ✓（丸め一致） |
| Control 差 | 0.0006 | 0.0006 | ✓ |
| BIC 選択 k, 全シナリオ | 3 | 3 | ✓ |
| Scen.B BIC 差 | 180 | 180 | ✓ |
| Y-only RMSE(Z), C | 0.0286 | 0.0286 | ✓ |
| X-only RMSE(Z), C | 1.1024 | 1.1024 | ✓ |
| 図1(b) Scen.C 最大バー | 23.6× | 23.55×（丸め23.6×） | ✓ |

**全数値が Phase 1B の照合結果と一致。**

---

## 5. NotebookLMへの投入判定

### 投入推奨（§15 記載の9ファイル）

| ファイル | 状態 |
|---------|------|
| `CLAUDE.md`（root） | ✓ Phase 1D で更新済み |
| `conference_submission_final_draft.md` | ✓ 完成版原稿 |
| `docs_for_notebooklm/NOTEBOOKLM_RESEARCH_BRIEF.md` | ✓ 本検証で確認済み |
| `docs_for_notebooklm/01_formula_code_audit.md` | ✓ Phase 1A で作成 |
| `docs_for_notebooklm/02_experiment_result_verification.md` | ✓ Phase 1B で作成 |
| `docs_for_notebooklm/03_figure_consistency_check.md` | ✓ Phase 1C で作成 |
| `docs/math_notes/half_factor_math_explanation.md` | ✓ 数学的証明（根拠文書） |
| `docs/math_notes/half_factor_literature_code_check.md` | ✓ MATLAB vs Python 照合表 |
| `expfam/references/baseline_metrics.md` | ✓ 先行研究との比較数値 |

### 投入注意（§16 記載の5ファイル）

| ファイル | 注意事項 |
|---------|---------|
| `docs/math_notes/legacy/parameter_estimation_corrected_formulas.md` | E-step の 1/2 が確定事項と矛盾する中間文書 |
| `expfam/CLAUDE.md` | 旧 Gemini セッション向け（root CLAUDE.md が優先） |
| `expfam/results/GEMINI_REPORT_*.md` | AI生成レポート（未検証部分あり） |
| `expfam/results/archive/` | 旧実験結果（現行結果と混同注意） |
| `archive/paper_writing_examples/*.pdf` | 他学会の参考例（本研究と無関係） |

---

## 6. 残課題の確認

本 BRIEF に記載された未確認事項（§18）：

| 項目 | 優先度 | 対応方針 |
|-----|------|---------|
| E-step 中の \|\|Term2\|\| vs \|\|Term3\|\| の定量比較 | 高 | 修論フェーズで追加実験 |
| 1/2 修正版実装との RMSE 比較実験 | 中 | 修論フェーズ |
| 図 1(b) の Word キャプション現状 | 中 | .docx 確認時に対応 |
| Scen. B での全 10 試行 k=3 選択の安定性 | 低 | 現時点では要注意事項として記載済み |
