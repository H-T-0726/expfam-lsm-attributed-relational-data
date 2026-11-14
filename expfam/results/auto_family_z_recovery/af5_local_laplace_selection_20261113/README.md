# AF5 / S2-B：共通パラメータに基づく列単位 Laplace 分布選択（探索的研究）

- 作成：2026-11-14
- branch：`experiment/af5-local-laplace-selection`（基点は AF4 `69709e5`）
- 実装系列：lineage E ＋ PR #129 の generic model（未マージの prototype、本文採用不可）
- 新しい MCEM fit：0 回
- 研究の種類：**EXPLORATORY**
  - AF2 と同じ 20 データを AF3〜AF5 の設計にも使ってきたため、独立な最終検証ではありません。
  - 結論には、新しい seed による確認研究が必要です。

## 1. 問い

AF2 の Q スコア選択は、q を作ったときの分布（Auto は Poisson 開始）に引き寄せられ、自分の選択を確認する方向に誤ります（AF3）。

AF5 の問いは次のとおりです。

> 基準モデル（anchor）のパラメータを共有し、1 列の分布だけを入れ替えて、Z を Laplace 近似で積分した周辺尤度を比べたら、AF2 の誤選択を実際に訂正できるか。

単にスコアの定義を変えただけで、精度が上がったように見えているのではないか、という点も確かめます。

## 2. 方法（数理は `numerical_audit.md` §1）

```
S_j(m; a) = max_{f_j} [ Φ_m(Ẑ_m(f_j), f_j) + (nK/2) log 2π − ½ log|H_m(f_j)| ]
Δ_j       = S_j(CMP2; a) − S_j(Poisson; a)          Δ > 0 なら CMP2 を選ぶ
```

- **固定するもの**：anchor a の θ_{−j}（ほかの列の loading、Gaussian 列の分散、w0、w、var_z）と、ほかの列の分布。
- **最適化するもの**：f_j（K=3）だけを、Laplace 補正込みの目的関数で最大化します。
- H は Y の交差ブロックを含む nK×nK の完全な負 Hessian です。
- **呼び方**：Laplace 近似による type-II 最尤のプロファイル比です。**Bayesian evidence でも BIC でも Q でもありません。**
- **anchor**：
  - auto_first：主解析。AF2 Auto の refit で、正解を使わずに使えます。
  - support_only：副次解析。
  - oracle：診断用で、改善率には含めません。
- **単位**：count 6 列を独立に判定します。逐次更新はしません。
  - 規模：20 データ × dense/sparse × 3 anchor × 6 列 ＝ 720 列。
- **判定状態**：protocol で固定しました（`numerical_audit.md` §3）。
  - 「sign」：判定不能（NUMERIC / NOT_CONVERGED）と失敗を除き、Δ の符号で判定します。UNDECIDED_LAPLACE も符号で数えます。
  - 「confident」：DECIDED のみ。
  - それ以外の列は、AF2 Auto の選択をそのまま使います。
- **統計の単位**：データセット（各 regime 20）。percentile bootstrap（10000 回、seed 20261115）。
- **比較対象**：
  - AF2 Auto の最終選択（primary_metrics）。
  - **同じ anchor 上の Q スコア**（AF3 の Z7 / Z5 / Z3、zero 開始）。スコアの定義を変えた効果だけを取り出すための比較です。

## 3. 手順と provenance

| 段階 | 内容 | git |
|---|---|---|
| Phase 0 | 試行 1 は不合格（内側の許容値が浮動小数点の精度より厳しかった）。試行 1 は `phase0_attempt1/` に残し、Δ を計算する前に修正。試行 2 で合格（`phase0/`） | `3dca16f` |
| protocol | `protocol.json` と `manifest.csv` を Δ の計算前に固定。canonical sha256 は `b0601adb…78aa74` | `3dca16f` |
| Phase 1 | pilot：rep01–03、108 列。実行可能性だけを確認し、正解率は見ず、設定も変えていない（`pilot/`） | 実行 `3dca16f`、commit `7c7e0a8` |
| Phase 2 | 本実行：720 列、14373 秒、warning 0（`full/runinfo.json`） | 実行 `7c7e0a8` |
| 分析 | 事前登録した分析（`full/analysis/`） | 本 commit |
| **事後解析** | 結果を読んだ後に追加し、事前登録していない（`full/analysis/posthoc/`） | 本 commit |

実行環境：Python 3.13.14、numpy 2.3.5、scipy 1.16.3、Windows 11。requirements sha256 は `full/runinfo.json` に記録しています。

## 4. 結果

### 4.1 技術的な妥当性（OBSERVED、`full/analysis/optimization_diagnostics.csv`・`error_diagnostics.csv`）

- FAILED 0、UNDECIDED_NOT_CONVERGED 0／720。UNDECIDED_NUMERIC は 2（sparse、auto と support）。
- 最適点での |S の有限差分勾配|∞ ≤ 1e-5、2 つの開始点の差と anchor の Zmode から計算し直した S との差はともに 0（表示桁）、内側の |∇|∞ ≈ 0、Hessian の最小固有値 ≥ 0.68。
- pilot の 108 列は、本実行で計算し直しても Δ の差 0.0、状態 108/108 で一致しました（事後解析、`posthoc_pilot_vs_full.csv`）。
- **IS が読める割合**：dense 89–96%、sparse 61–71%。
  - IS 補正量の中央値：dense 0.024–0.030、sparse 0.11–0.13。
  - IS で符号が変わった列：各条件 0–2。

### 4.2 正解率（主 anchor = auto_first、sign、count 列）

| regime | Laplace | 同じ anchor の Q（Z7） | AF2 Auto | Laplace 真 CMP2 / 真 Poisson | 参考：joint MAP 版 |
|---|---|---|---|---|---|
| dense | 0.917 [0.87, 0.97] | 0.925 | 0.900 | 0.900 / 0.933 | 0.942 |
| sparse | 0.857 [0.80, 0.91] | 0.783 | 0.767 | 0.864 / 0.850 | 0.824 |

- confident（DECIDED のみ）：dense 0.938（112 列）、sparse 0.859（71 列）。
- 副次の support_only：0.933 / 0.840。診断用の oracle：0.925 / 0.892。

### 4.3 訂正と副作用（判定しない列は AF2 の選択を維持、`full/analysis/paired_comparison.csv`）

| 主 anchor | 参照 | mode | 訂正（参照が誤り → Laplace が正しい） | 副作用（参照が正しい → Laplace が誤り） | データセット単位の正解列数の変化 [95% CI]（改善/同じ/悪化） |
|---|---|---|---|---|---|
| dense | AF2 Auto | sign | 5 / 12 | 3 | +0.10 [−0.20, 0.40]（3/16/1） |
| dense | AF2 Auto | confident | 5 / 12 | 0 | +0.25 [0.05, 0.50]（4/16/0） |
| dense | Q（Z7） | sign | 2 / 9 | 3 | −0.05 [−0.30, 0.15]（1/18/1） |
| sparse | AF2 Auto | sign | 18 / 28 | 8 | +0.50 [0.10, 0.85]（11/6/3） |
| sparse | AF2 Auto | confident | 7 / 28 | 5 | +0.10 [−0.25, 0.45]（5/11/4） |
| sparse | Q（Z7） | sign | 16 / 26 | 8 | +0.40 [0.05, 0.70]（11/6/3） |
| sparse | Q（Z7） | confident | 6 / 26 | 6 | 0.00 [−0.30, 0.30]（5/10/5） |

### 4.4 誤りの方向（事後解析、`posthoc_error_direction.csv`、主 anchor）

| regime | 真の分布 | AF2 Auto の誤り | Q（Z7）の誤り | Laplace の誤り |
|---|---|---|---|---|
| dense | CMP2 | 11/60 | 8 | 6 |
| dense | Poisson | 1/60 | 1 | 4 |
| sparse | CMP2 | 26/59 | 24 | 8 |
| sparse | Poisson | 1/60 | 1 | 9 |

### 4.5 誤った列のマージン（事後解析、`posthoc_margin_wrong_vs_right.csv`、真の分布に向かう符号付きマージンの |·| の中央値）

| regime | Laplace が誤り | Laplace が正しい | AF2 の Q が誤り | AF2 の Q が正しい |
|---|---|---|---|---|
| dense | 0.29 | 5.20 | 2.75 | 8.98 |
| sparse | 0.68 | 3.28 | 4.71 | 8.20 |

- 事後解析：|Δ| > 1 の列だけに絞ると、主 anchor の正解率は dense 0.982（92% の列が残る）、sparse 0.925（78% が残る）でした（`posthoc_abs_delta_threshold.csv`）。
- **この閾値は結果を見た後に決めたので、選択規則としては使えません。**

### 4.6 anchor への依存（`full/analysis/anchor_sensitivity.csv`）

| regime | auto と support の一致 | auto と oracle の一致 |
|---|---|---|
| dense | 118/120 | 119/120 |
| sparse | 115/119 | 111/119 |

- 同じ anchor の Q の正解率は、oracle 0.992/1.000、support 0.892/0.767 と anchor によって大きく変わりました（AF3）。
- Laplace は 0.925–0.933 / 0.840–0.892 で、anchor による変化は小さく収まりました。

### 4.7 Δ の分解（`full/analysis/decomposition.csv`、主 anchor、平均）

- 対象列の Φ の差は、真の分布によらず常に CMP2 に有利です（+8〜+19）。
- ほかの項の差（−3〜−8）と −½ log|H| の差（−3〜−12）が、それを打ち消します。
- 誤った真 CMP2 列は、真の loading のノルムが大きい傾向があります（中央値 2.36 / 2.33。正しい列は 1.87 / 1.80）。

## 5. 問い A〜G への回答

**A. 共通パラメータで 1 列の識別は改善したか**
- OBSERVED：fit ごとに θ̂ が違うことによる AF4 のばらつき（SD 4–12）は、構成上なくなりました。Δ の大きさの中央値は dense で約 5、sparse で約 3 です。
- OBSERVED：誤った列は |Δ| が小さく、ほぼ同点でした。dense 0.29、sparse 0.68 で、正しい列は 5.2 / 3.3 です。AF2 の Q の誤りは同点ではありません（2.75 / 4.71）。
- 解釈：Δ の大きさが判定の信頼度を表しているという意味では改善しました。ただし dense の正解率は、同じ anchor の Q と変わりません（0.917 と 0.925）。

**B. AF2 の誤選択をいくつ訂正したか**（主 anchor、sign）
- dense は 12 列中 5 列、sparse は 28 列中 18 列です。
- 同じ anchor の Q と比べると、dense は 9 列中 2 列、sparse は 26 列中 16 列です。
- **sparse の訂正は、スコアの定義を変えただけの効果では説明できません。** 同じ q を使った Q との paired 比較でも、+0.40 [0.05, 0.70] が残ります。
- **dense では、同じ anchor の Q と比べた改善は確認できませんでした。**

**C. 副作用（正しかった列を誤らせたか）**
- あります。sign では dense 3、sparse 8 です。
- 誤りの方向が変わりました（§4.4）。AF2 の誤りは真 CMP2 → Poisson にほぼ限られていましたが、Laplace の誤りは真の分布によらず同じくらい起きます。
- 判定状態の効き方：
  - dense の副作用 3 列は、いずれも UNDECIDED_LAPLACE でした。confident にすると副作用は 0 になります。
  - sparse の confident でも副作用は 5 です。

**D. anchor に依存するか**
- ほとんど依存しません（§4.6）。oracle の anchor を使っても、正解率は dense 0.925、sparse 0.892 にとどまります。
- 解釈：残る誤りは anchor の誤りや自己確認によるものではなく、Z を知らないときに 1 列だけで分布を見分けることの難しさから来ていると考えられます（HYPOTHESIS）。
  - AF3 で真の Z を与えたときは 60/60 でした。そのため、情報が足りない原因は Z の不確かさにあるとみられます。

**E. 改善は数値誤差を超えるか**
- 最適化の誤差（≤ 1e-5）と判定の数値的不確かさ（1e-2）は、誤った列の |Δ|（0.29 / 0.68）よりも十分小さいので、問題になりません。
- Laplace 近似そのものの誤差は、dense では IS 補正（中央値 0.03）よりも小さく、問題になりません。
- **sparse の改善は、IS で確かめられない列に偏っています。**
  - 主 anchor の DECIDED_LAPLACE_UNASSESSED は 45 列で、正解率は Laplace 0.867、AF2 0.667 です。
  - IS で確かめた DECIDED は 71 列で、0.859 と 0.831 です（事後解析、`posthoc_accuracy_by_status.csv`）。
  - confident では、改善は +0.10 [−0.25, 0.45] に縮みます。
- 結論：sparse の改善は数値誤差を超えますが、Laplace 近似による誤差までは超えると確認できていません。

**F. 局所 Laplace 選択は、新しいアルゴリズムとして研究する価値があるか**
- 条件付きで、あります。根拠は 3 つです。
  - Q で起きていた自己確認による系統的な偏りが消えました。
  - anchor にほとんど依存しないので、正解を使わない Auto の anchor でも使えます。
  - |Δ| が判定の信頼度として働きます。
- 限界：
  - dense の正解率は Q とほぼ同じです。
  - 誤りが真 Poisson 列にも出るようになりました。
  - 1 列あたり約 20 秒かかります。
  - 副次の joint MAP 版も、dense では同じくらいの成績でした（0.942 / 0.824）。補正込みの最適化が必要かどうかは、まだ分かっていません。

**G. 次に何を行うか（提案、決めるのは研究者）**
- 推奨は**新しい seed による確認研究**です。
  - 主 anchor、Δ の符号、判定状態の規則をこのまま固定して、sparse での改善と誤りの方向の変化が再現するかを確かめます。
  - sparse で IS が読めない問題には、提案分布の改善または draw 数の増加を、事前に決めて組み込むのが妥当です。
- 全パラメータ MCEM の再推定は、選択を訂正した後の Z の回復を主張するために必要です。ただし、確認研究の後で行うべきです。
- held-out X は予測についての別の問いなので、優先度は低いと考えます。

## 6. 主張しないこと

- **Z の最終的な回復が改善したとは主張しません。** 訂正した割当てで MCEM を再推定していません。
- 確定したアルゴリズムでも、一般的な有効性でもありません。条件は 1 つ（n=75、K=3、Poisson と CMP2 ν=2）、データは 20 で、探索的研究です。
- oracle の結果は診断用で、改善率には含めていません。
- Bayesian evidence / BIC ではありません。

## 7. 未解決の問題

- sparse の IS が読めない列（29–39%）で、Laplace 近似の誤差がどれほどか。
- 対象列の Φ が常に CMP2 に有利な理由（DERIVED まではしていません）。仮説は、CMP2 の分散が小さいため Ẑ が対象列に合わせて動き、その代償をほかの項が払う、というものです。
- joint MAP 版が dense でやや良かったこと（事前に決めた副次の記録で、検定はしていません）。

## 8. 成果物の対応表

| 依頼された成果物 | 場所 |
|---|---|
| protocol / runinfo | `protocol.json`、`full/runinfo.json`、`pilot/runinfo.json` |
| 数値監査 | `numerical_audit.md`、`phase0/`、`phase0_attempt1/` |
| optimization_diagnostics.csv | `full/analysis/optimization_diagnostics.csv` |
| candidate_scores.csv / column_selection.csv | `full/` |
| paired_comparison.csv / anchor_sensitivity.csv / error_diagnostics.csv | `full/analysis/` |
| 図 | `full/analysis/fig1_delta_by_column.png`、`fig2_corrections_harms.png` |
| 事後解析 | `full/analysis/posthoc/` |

## 9. 再現コマンド（repository root で実行、AF2 の状態は `D:\tento\kennkyu_artifacts\auto_family_z_recovery_af2_20261110\full\fits`）

```
python expfam/src/experimental/run_af5_local_laplace_selection.py --phase0-audit
python expfam/src/experimental/run_af5_local_laplace_selection.py --pilot
python expfam/src/experimental/run_af5_local_laplace_selection.py --execute          # 中断したら --resume
python expfam/src/experimental/run_af5_local_laplace_selection.py --analyze --stage full
python expfam/src/experimental/run_af5_local_laplace_selection.py --posthoc --stage full   # 事後解析
python -m pytest -q expfam/src/experimental/test_af5_local_laplace_selection.py
```
