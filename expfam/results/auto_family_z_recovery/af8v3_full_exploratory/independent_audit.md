# AF8-v3 final 40 データ探索実験：独立監査

- 監査者：subagent `research-auditor`（読み取り専用、研究者が明示的に依頼）。
- 対象：commit `fe98694` の code・protocol と、この directory の結果・README・numerical_audit の草稿（結果を commit する前）。
- 制約：監査者はコードを実行できない。中央値や bootstrap の再計算が必要な項目は UNVERIFIED としている。
- 本書は監査報告の要約で、対応は本 session で行った。

## 結論

**BLOCKER は 0 件、WARNING は 4 件、NOTE は 8 件。** 停止規則の gate、Q で置き換えない規則、E2 の式、主要な数値は、CSV とコードで裏づけられた。

## 確認された事項（FACT、監査者による照合）

**protocol の遵守**
- 主手法 E2 と主 prior `B_sg2_0.667` は protocol とコードで一致している。
- 12 shard と解析の runinfo は、すべて同じ head `fe98694`・protocol hash で、tracked file に変更がない。
- 停止規則の閾値は v3 と同じで、それに S4v2 と S0 を追加しただけである（`test_13`）。
- `analyze()` は `all_stop_rules_pass` が真でないと終了する。数値監査と解析は同じ `candidate_scores.csv`（sha256 `886ac8fa…`）を読んでいる。
- B の失敗 3 列と C の失敗 5 列は `FAILED_AF6` として正解に数えていない。Q で置き換えるのは、別の名前をつけた E2orQ だけである。

**数値**
- 監査 JSON の S0–S9 はすべて `false` で、NaN を失敗側に写した件数は 0、完全性は 8640/8640。
- 開始点が 2 つの 27 セルは、AF6 で CMP2 の Laplace が FAILED だった列で、`af6_fL` の開始点がない。主張どおり。

**式の確認**
- S(f) と φ（Z の正規化された prior、X 列ごとの項、Y の上三角の対、余分な ½ なし）を確認した。
- E2 = T + ½K log2π − ½ log|−∂²T|：`fd_hess` が符号を反転している。
- E と IS の式を確認した。
- prior は正規化された N(0, vI) で、B 系では Jacobian を含む。
- 両候補の基底測度は proper である（Poisson は −log x!、CMP2 は −2 log x! − log I0(2√λ)）。

**照合**
- E2 と IS の最大差は 0.0876（|z| 3.20）、符号の不一致は 0。
- E と E2 の符号の不一致は 16 で、すべて sparse。
- `selection_results.csv` から数え直した正解数・誤選択数・失敗数は、`accuracy.csv` と一致した。
  - E2：P 223 / 17、C 216 / 24。
  - B：P 200 / 37、失敗 3、C 228 / 12。
- 分母、対応のある dataset-cluster bootstrap、移行数、データごとの合計も整合している。

## WARNING と対応

| # | 内容 | 対応 |
|---|---|---|
| W1 | README で、E と E2 の sparse での入れ替わりを「9 列」と書いていた。正しくは 16 列（9 は真 Poisson だけの数）。 | README を「16 列（真 Poisson 9・真 CMP2 7、正味 0）」に修正した。 |
| W2 | numerical_audit.md の「full の rep01–03 と re-pilot の差」は script の出力ではなかった。 | `compare_full_vs_repilot()` を追加した（code commit `d5227f0`、結果より前）。出力は `full/full_vs_repilot_rep01_03_summary.json`。差は E2 1.5e-7 などで、Δ の符号の変化は 0。<br>あわせて、numerical_audit.md §4 の median 類は CSV から集計した DERIVED 値だと明記した。 |
| W3 | 「prior（D−B）と積分（E2−D）」への分解は座標に依存する。D は f 上の MAP で、(K/2) log A''_CMP2(0) の Jacobian 差が CMP2 に一律に不利に入る。 | 監査者の手計算 ≈ −1.0 nat を、本 session でコードの値 A''_CMP2(0) = 0.5131 から再計算した（−1.0009 nat）。README §4 に、配分が手法そのものの性質ではないことを追記した。 |
| W4 | 主手法の選択の経緯を開示していなかった。v1 の主スコアは E で、E2 への変更は rep01–03 の pilot の後だった（40 データに含まれ、AF6 でラベルが既知）。prior の設計も、AF6 の誤選択パターンを知った後だった。 | README §1 に経緯と forking-paths の限界を追記した。 |

## NOTE と対応

| # | 内容 | 対応 |
|---|---|---|
| N1 | AUC の bootstrap は 2,000 回で、protocol の文面（10,000 回）と異なる（コードは実行前に commit 済み）。 | README §1 に文面との相違として明記した。 |
| N2 | AUC の bootstrap の分位点（`np.nanquantile`）と、診断用の `diag_median_abs_DeltaE_minus_DeltaE2`（`dropna`）が NaN を飛ばす。 | gate ではなく、採点したすべての手法で `auc_boot_nan` = 0 なので影響はない。記録だけにとどめ、コードは変更していない。 |
| N3 | AUC の列数が手法で異なる（B 477、C 475）。 | README §1 に明記した。 |
| N4 | prior 感度の「選択の変化」は判定不能になった列も数えている。また表に判定不能の列がない。 | README に両方を明記した。 |
| N5 | CI 上限の丸めが 2 か所違っていた（0.163、0.138）。 | 0.162 と 0.137 に修正した。 |
| N6 | `independent_audit.md` がなかった。 | 本書で解消。 |
| N7 | registry の行がない。 | 結果の commit で追記する。 |
| N8 | テスト数・traceback 0・CPU 秒は実行していないので UNVERIFIED。 | 本 session で pytest 42 件の通過を確認した（`d5227f0` の時点）。traceback 0 は shard ログの grep による。CPU 秒はセルごとの wall-clock の合計（OMP=1）。 |

## 主張と証拠

- README の注意書きは揃っている：prototype、exploratory、データの再利用、BIC でも Bayesian evidence でもない、多重比較の補正なし、prior への依存、Z の Laplace 誤差は未検証。
- W1 と N5 を除き、README §3 の数値は CSV と一致した。
- W3・W4 以外に、証拠を超える主張は見つからなかった。
