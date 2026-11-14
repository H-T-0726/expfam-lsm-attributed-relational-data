# AF8：loading f_j を積分した分布選択基準の数理監査と探索実験（pilot で停止）

- 系列：lineage E＋PR #129 の generic model（prototype、本文には採用不可）。
- 種類：探索的研究。AF6 のデータと保存した Auto anchor を再利用するもので、独立な確認実験ではない。
- 新規の MCEM は 0 回、新しいデータは使っていない。
- **結論：事前登録した pilot の停止規則が作動したため、本実験（40 データ）は行わなかった。** 原因は、最適化の安全装置の欠陥である（`numerical_audit.md`）。
- **f を積分すると Poisson→CMP2 の偏りが改善するかどうかは、まだ答えられていない。** pilot では正解率も選択結果も計算していない。

## 1. 何をしたか

| 段階 | commit | 内容 |
|---|---|---|
| AF7 の監査と訂正 | `0060b16` | `mathematical_derivation.md` §0：用語「profile 化」の訂正、真の θ と TI の混同の訂正、KL ≥ 0 は母集団についての性質であること |
| 数式の導出 | `0060b16` | f を最適化することと積分することの違い（§1）、prior B の平坦極限 −1.00（§1.1）、同時 Laplace と完全な Hessian（§2）、D と E の切り分け（§3） |
| prior の設計 | `0060b16` | `prior_design.md`：生成器の loading は Haar の行で Gaussian ではない。主設定は B（s=2/3）。感度分析に 8 種類 |
| 実装、事前登録、test | `0060b16`、`a37584b`（driver のみ） | `run_af8_integrated_loading.py`、`analyze_af8_integrated_loading.py`、`test_af8_integrated_loading.py`（5 件合格）、`protocol.json`（sha256 `5c8f4b2e…5815`）、`manifest.csv`（8640 セル） |
| pilot | 本 commit | rep01–03 の 648 セル。数値監査だけを行い、正解率は計算していない |
| 停止と原因の診断 | 本 commit | `numerical_audit.md`、`pilot/pilot_numerical_audit.json`、`pilot/pilot_failure_diagnosis.json` |
| 独立監査 | 本 commit | research-auditor：BLOCKER なし、WARNING 6 件（`numerical_audit.md` §6） |

## 2. pilot の結果（数値面）

- 停止規則で作動したのは「開始点による不一致の割合」で、5.6%（閾値 5%）だった。
- 主な原因：
  - joint Newton が鞍点に収束した（負の固有値 3 個、Ψ は最大より 600〜1190 低い）。
  - BFGS の評価中に LinAlgError が出て、Poisson で 22 セルが FAILED になった。いずれも重い裾の 1 列（x の最大 86）。
  - D の zero 開始点では、Z だけの問題が凹でなかった（番兵の値 1e300）。
- 採用した値は、すべて正定値の停留点にある（最小固有値 ≥ 0.93）。
- 近似の比較（主 prior、33 列）:
  - |ΔE − ΔIS| の中央値は 0.19、最大 1.18。|ΔE2 − ΔIS| の中央値は 0.017。
  - ただし IS は E2 と同じ被積分関数を使うので、E と E2 のどちらが厳密な値に近いかは判定できない。
- 計算時間は、1 セルの中央値が主 prior で 16 秒、それ以外で 5 秒。

## 3. 次に人間が判断すること（Human Gate）

`numerical_audit.md` §5 の修正案（1〜6）を実装し、新しい版の protocol を結果を見る前に固定して、pilot をやり直すかどうか。

- 修正案の中身:
  - 修正 Newton（または信頼領域）で鞍点を避ける。
  - 非有限値を −∞ として返す。
  - 不一致の規則を定義し直す。
  - 主となる近似を E にするか E2 にするかを決める。
  - 失敗を Q に戻す規則を見直す（候補に依存した欠測になるため）。
- pilot のデータは本実験の 40 データに含まれることも開示が必要。

## 4. 成果物

| ファイル | 内容 |
|---|---|
| `mathematical_derivation.md`, `prior_design.md`, `protocol.json`, `manifest.csv` | 事前登録（`0060b16`） |
| `numerical_audit.md` | 実装の検証、停止の判定、原因、近似の精度、修正案、独立監査 |
| `pilot/candidate_scores.csv`, `pilot/progress_*.jsonl`, `pilot/runinfo_*.json` | pilot の全 648 セル |
| `pilot/pilot_numerical_audit.json`, `pilot/pilot_numerics_by_column.csv`, `pilot/pilot_failure_diagnosis.json` | 停止規則の計算と原因の診断 |

依頼にあった `selection_results.csv`、`paired_comparison.csv`、`prior_sensitivity.csv`、`statistical_analysis.csv` と図は、**本実験を行っていないので存在しない**（スクリプトは作ってある）。

## 5. 再現（repository root、OMP_NUM_THREADS=2）

```
python -m pytest expfam/src/experimental/test_af8_integrated_loading.py
python expfam/src/experimental/run_af8_integrated_loading.py --pilot --shard i/5      # i=0..4
python expfam/src/experimental/run_af8_integrated_loading.py --collect-pilot
python expfam/src/experimental/analyze_af8_integrated_loading.py --pilot-audit --pilot-diagnose
```

AF6 の fit 状態（`D:\tento\kennkyu_artifacts\af6_laplace_confirmation_20261114\fits\`）が必要である。
