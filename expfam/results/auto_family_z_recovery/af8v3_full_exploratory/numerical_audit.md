# AF8-v3 final：数値監査（40 データ、rep01–40、8,640 セル）

出典：`full/full_numerical_audit.json`、`full/full_pairs.csv`、`full/full_start_certificates.csv`、`full/full_is_draw_log.csv`、
`approximation_diagnostics.csv`。いずれも script（`analyze_af8v3_full_integrated_loading.py --full-audit` / `--analyze`）の出力で、本書はその数値を整理しただけである。
§4 の regime 別の median・分解項の median・|z| の median は、`approximation_diagnostics.csv`（script 出力）の列を集計した DERIVED 値である。
停止規則は正解率を計算する前に評価した（`analyze()` は `all_stop_rules_pass` が真でないと正解率の計算を拒否する）。

## 1. 実行の範囲（FACT）

- code / protocol：commit `e8943d3`（protocol sha256 `e7dfd0ca…`、`json.dumps(protocol(), sort_keys=True)` の hash）。
  実行時の HEAD は `fe98694`（re-pilot の結果を追加しただけで、コードは同じ）。12 shard すべての runinfo が同じ head・protocol を記録している。
- 実行：12 shard を並列、`OMP_NUM_THREADS=1`、合計 276,470 CPU 秒。traceback 0 件。
- 新しい MCEM fit・新しいデータ生成はしていない。80 ケース（40 データ × dense/sparse）の X・Y・mask・anchor state の hash は AF6 と一致（`manifest.csv`）。

## 2. 停止規則（OBSERVED、閾値は re-pilot と同じ。すべて通過）

| 規則 | 閾値 | full（8,640 セル） | re-pilot（648 セル） |
|---|---|---|---|
| S0 完全性 | 期待セルがすべてある | 8,640/8,640（欠損・重複・余分 0） | 648/648 |
| S1 E2 が OK でないセル | ≤ 0.05 | 0 | 0 |
| S2 有効な開始点どうしの不一致 | ≤ 0.05 | 0 | 0 |
| S2b 証明書つき開始点が 2 未満 | ≤ 0.05 | 0 | 0 |
| S3 主 prior の median \|ΔE2−ΔIS\| | ≤ 0.25 | 0.0172（480 対、除外 0） | 0.0124 |
| S3b S3 < median\|ΔE2\|/2 | 真 | 0.0172 < 2.19 | 通過 |
| S4 救済後も失敗した draw を含む IS セル | ≤ 0.05 | 0/960 | 0/72 |
| S4v2（v2 定義）cold start で 1 回でも失敗した draw | ≤ 0.05 | 0/960 | 0/72 |
| S4r 救済した draw を含むセル | ≤ 0.05 | 0/960 | 0/72 |
| S4b IS の最小 ESS | ≥ 40 | 329.4 | 339.2 |
| S5 決定性 | ≤ 1e-8 | 0.0 | 0.0 |
| S6 ΔE2 の FD 刻み幅感度（median） | ≤ 0.05 | 3.1e-7（最大 4.8e-6） | 4.1e-7 |
| S8 Nelder–Mead が T(f_D) を上回った量 | ≤ 1e-3 | 1.8e-12 | 9.1e-13 |
| S9 FD 雑音から見積もった見かけの改善 | ≤ 1e-6 | 1.6e-12 | 7.3e-14 |

S7（FD 微分の検算）は protocol どおり re-pilot だけで行った（4.7e-8）。
NaN を失敗側に写した件数（`nan_counts_mapped_to_failing_side`）はすべて 0。

**その他（FACT）**
- IS の draw：384,000 回中、失敗 0・救済 0。
- S の評価 3,349,643 回中、証明書が得られなかったもの 0。
- 外側の証明書（最悪値）：λ_f²/2 = 8.2e-12、追加 Newton step の改善 8.2e-12、‖∇T‖∞ = 1.0e-5、−∂²T の最小固有値 2.45。
- E（joint Laplace）：λ²/2 ≤ 1.3e-25、Laplace step の変化 ≤ 3.3e-11、最小固有値 0.45。
- 開始点の証明書：25,893 件すべてが CERT_OK、すべて full Newton step（fraction 1.0）。W2 の修正経路（1/2・1/4 step、EXTRA_STEP_INVALID）は一度も使われなかった。
- 開始点が 2 つのセルが 27 ある（rep09 dense 列 3、rep15 dense 列 3、rep15 sparse 列 3 の CMP2 候補 × 9 prior）。
  これらは AF6 の Laplace 解（`af6_fL` 開始点）が存在しない列で、`make_starts` の設計どおりである。2 つの開始点は一致しており S2b も満たす。
- full の rep01–03 と re-pilot の比較（`full/full_vs_repilot_rep01_03_summary.json`、独立監査 W2 を受けて script 化、commit `d5227f0`）：
  648 セルすべて対応、最大 |差| は D 1.4e-12、E2 1.5e-7、E 1.6e-12、IS 7.3e-8、状態の変化 0、Δ の符号の変化 0（re-pilot は `OMP_NUM_THREADS=2`、full は 1）。

## 3. 列の状態（主 prior、480 列、ラベルを使わない）

| 手法 | DECIDED | NUMERIC_UNDECIDED | ONE_FAILED / BOTH_FAILED / MISSING |
|---|---|---|---|
| E2 | 480 | 0 | 0 |
| D（prior MAP） | 478 | 2 | 0 |
| E | 480 | 0 | 0 |
| IS | 480 | 0 | 0 |
| B（AF6 Laplace） | 477 | 0 | 3（AF6 で FAILED、Q で置き換えない） |
| C（joint-MAP Laplace） | 474 | 1 | 5 |

全 prior（4,320 列）では E2 の NUMERIC_UNDECIDED が 6。

## 4. 近似の差（診断、主スコアは E2。`approximation_diagnostics.csv`）

**E2 と IS（480 対）**
- |ΔE2−ΔIS|：median 0.017、最大 0.088。IS の標準誤差で割ると |z| ≤ 3.2（median 0.68）。
- 符号の不一致 0/480、選択の違い 0/480。
- これは「Z の Laplace S(f) を与えたときの外側 f 積分の Gaussian 近似」が正確だという意味であり、Z の Laplace 誤差は検証していない。

**E と E2（480 対）**
- median |ΔE−ΔE2|：dense 0.088、sparse 0.459（全体 0.189）。最大 |ΔE−ΔE2| は 3.88。
- 符号の不一致 16/480、すべて sparse（Poisson が真 9、CMP2 が真 7）。dense では 0。
- 差の分解：f の曲率の項（½(Δlogdet_f − Δlogdet_schur)）は median 0.03、残り（モードと Z の項）が支配的（median −0.19、範囲 −4.3〜4.2）。
- INTERPRETATION：数値誤差（S6・S9 で 1e-6 以下）では説明できない、joint Laplace と nested Laplace の近似の違いである。
  sparse で大きいのは、Y の情報が少なく Z の事後分布が広い（Laplace 近似の前提が弱い）ことと整合するが、どちらが厳密な周辺尤度に近いかはこの実験では判定できない。

**曖昧な列（protocol の事前定義、ラベルを使わない）**：60/480（dense 15、sparse 45）。
定義は |ΔE2| ≤ 1、E と E2 の符号の不一致、IS と E2 の符号の不一致、または NUMERIC_UNDECIDED のいずれか。
これらの列の診断量（IS ± 2se、分解、刻み幅感度）は `approximation_diagnostics.csv` にある。手法・prior・閾値の変更には使っていない。

## 5. 結論（数値面のみ）

- protocol の数値ゲートはすべて通過し、正解率を計算してよい状態だった。
- 「数値的に信頼できる」と言える範囲：E2 の値は数値誤差 ≪ 0.01 で計算できており、IS との差は 0.09 以下。
- 言えないこと：E2（や E）が厳密な (Z, f) 周辺尤度に近いこと。E と E2 は sparse で最大 3.9 ずれ、16 列で符号が逆になる。
