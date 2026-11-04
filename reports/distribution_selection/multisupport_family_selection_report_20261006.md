# Multi-support family selection — 結果報告（2026-10-06）

- 位置づけ: **CHARACTERIZATION**（controlled synthetic family-selection experiment）。系列 **E（experimental prototype; 本文採用不可）**。
- run: `expfam/results/family_selection/expanded_multisupport_20261006_034826/`
- 数理監査: `family_candidate_compatibility_audit_20261006.md` / 方法・判断記録: `multisupport_family_selection_methods_20261006.md`
- 自動生成の表（本報告の数値の出所）: `seminar_family_selection_summary_20261006.md`、および run の `tables/*.csv`
- audit: `audit_report.json` **PASS（BLOCKER 0, WARNING 4, DIAGNOSTIC 1）**
- K_true = K_fit = 3、Y = Bernoulli（w0 = −1, w = 1）、nuisance 固定。

---

## 1. 研究質問

同一 support・同一 base measure 上に複数の妥当な候補分布があるとき、連続潜在 Z を考慮した candidate score で
真の属性分布族（列ごと）を回収できるか。true Z を与えた selector 単体（A）、support group 別の full MCEM（B）、
全 support を混在させた mixed-X MCEM（C）で測り、ORACLE（真の family 固定）と SUPPORT_ONLY（データ型だけで決める手続き）と比べる。

## 2. 実行したもの

| phase | 内容 | case | 失敗 case | technical gate |
|---|---|---:|---:|---|
| smoke | A/B/C, n=40, rep 1 | 38 | 0 | PASS |
| pilot | A/B/C, n=75, rep 1–3 | 114 | 4 | PASS |
| A_main | selector only, n∈{75,150}, rep 1–20, 3 群 × EASY/HARD | 240 | 0 | PASS |
| B_main_r5 / r10 | support 群別 MCEM, rep 1–10, 4 手法 | 480 | 8 | PASS |
| C_main_r5 / r10 | mixed-X MCEM（d=22）, rep 1–10, 4 手法 | 160 | 10 | PASS |

main の MCEM fit は 640 件、うち **18 件が FAILURE**（2.8%）。retry・置換・seed 変更 0（audit で確認）。
pilot 中に見つけた数値欠陥（truncated normal の分散の桁落ち）は main 開始前に修正した（方法記録 §3、audit WARNING として開示）。
main の全 phase は experiment-path のコードが同一（audit で確認）。

---

## 3. FACT（artifact から直接言えること）

数値は main stage。「score-decided」は admissible 候補が 2 つ以上あり score で決まった列だけ、
「all columns」は失敗 fit の列を不正解として分母に残した全列。

### 3.1 selector only（A, true Z, 各群 2 n × 2 難易度 × 20 rep）

| support group | family recovery | 主な誤り（candidate model, 件数） |
|---|---|---|
| COUNT（5 candidate model, 1200 列） | **0.984**（1181/1200） | poisson→cmp_nu2p0 11、cmp_nu0p5→poisson 6、cmp_nu2p0→poisson 2 |
| POSITIVE（2, 480 列） | **1.000**（480/480） | なし |
| BOUNDED（3, 720 列） | **0.949**（683/720） | tn01→logitnormal 22、logitnormal→tn01 8、cb→tn01 4、tn01→cb 2、cb→logitnormal 1 |

- candidate model 別で最も低いのは `tn01_s0p25_c0p5`（0.900, 216/240）と `poisson`（0.954, 229/240）。
- A の COUNT 列はすべて score-decided（各列に admissible 候補が 3 つ以上あった）。
- n の効果（A, family-level）: COUNT 0.968（n=75）→ 1.000（n=150）、BOUNDED 0.919 → 0.978、POSITIVE は両方 1.000。
- EASY/HARD（A）: COUNT 0.987 / 0.982、POSITIVE 1.000 / 1.000、BOUNDED 0.953 / 0.944。

### 3.2 support 群別 full MCEM（B, 各群 2 n × 2 難易度 × 10 rep）

| support group | Auto START_FIRST score-decided | all columns | Auto START_LAST score-decided | all columns |
|---|---|---|---|---|
| COUNT | 0.970（509/525） | 0.848（509/600, 失敗 75 列 = 5 fit） | 0.972（554/570） | 0.923（554/600, 失敗 30 列 = 2 fit） |
| POSITIVE | 1.000（240/240） | 1.000 | 0.996（233/234） | 0.971（233/240, 失敗 6 列 = 1 fit） |
| BOUNDED | 0.944（340/360） | 0.944 | 0.950（342/360） | 0.950 |

- B の失敗はすべて **COUNT EASY n=150**（7 fit）と **POSITIVE EASY n=150**（1 fit）の AUTO。ORACLE / SUPPORT_ONLY の失敗は 0。
- EASY/HARD（B Auto START_FIRST, score-decided）: COUNT 0.947 / 0.987、POSITIVE 1.000 / 1.000、BOUNDED 0.978 / 0.911。

### 3.3 mixed-X MCEM（C, d=22, 2 n × 2 難易度 × 10 rep）

| support group | Auto START_FIRST score-decided | all columns | Auto START_LAST score-decided | all columns |
|---|---|---|---|---|
| COUNT | 0.970（320/330） | 0.800（失敗 70 列） | 0.976（361/370） | 0.902（失敗 30 列） |
| POSITIVE | 1.000（132/132） | 0.825（失敗 28 列） | 1.000（148/148） | 0.925 |
| BOUNDED | 0.960（190/198） | 0.792（失敗 42 列） | 0.959（213/222） | 0.887 |
| REAL（Gaussian, gate のみ） | – | 0.825（失敗 14 列） | – | 0.925 |

- C の失敗 10 fit はすべて **EASY**（n=75: 7、n=150: 3）の AUTO。HARD は失敗 0。
- 図 10: 失敗を不正解として数えると EASY の candidate-model recovery は 0.57–0.65、HARD は 0.92–1.00。

### 3.4 失敗の内訳（main, 18 fit）

| 原因 | 件数 | 条件 |
|---|---:|---|
| exploration 中、CMP 候補（ν=0.5 が 15、ν=2.0 が 1）の **開始 loading（incumbent）で CMP 級数が評価不能** | 16 | COUNT EASY n=150, MIXED EASY |
| E-step で Poisson の exp(η) overflow（η=749） | 1 | COUNT EASY n=150, START_FIRST |
| PTN 候補の trust-exact 内部（scipy の Cholesky に NaN）; 当該 posterior sample に \|z\|=23.4 | 1 | POSITIVE EASY n=150, START_LAST |

pilot（修正前のコード）では、ほかに PTN 分散の桁落ち 3 件と CMP 開始 loading 1 件があった。

### 3.5 RMSE(Z)（Procrustes 後）: Oracle / Auto / Support-only

同一 dataset で 3 手法とも成功したものに限った paired gap（mean, Auto START_FIRST）:

| design | dataset | Auto − Oracle | Auto − SupportOnly | Auto < SupportOnly |
|---|---|---|---|---|
| COUNT | 35/40 | −0.0016 | −0.0399 | 26/35 |
| POSITIVE | 40/40 | +0.0000 | −0.0183 | 35/40 |
| BOUNDED | 40/40 | +0.0000 | −0.0089 | 28/40 |
| MIXED | 33/40 | −0.0016 | −0.0434 | 24/33 |

- SUPPORT_ONLY は COUNT EASY と MIXED EASY で RMSE が 1.0 を超える fit が各 2 件ある（COUNT EASY n=75 rep9 / n=150 rep6、MIXED EASY n=75 rep5 / n=150 rep7。図 7 の外れ値）。
- Auto の COUNT EASY n=150 の平均 RMSE（0.252）は、失敗しなかった 5/10 fit だけの値である。

### 3.6 start agreement（START_FIRST vs START_LAST）

| design | 両 start 成功 | 最終 assignment 完全一致 | 列単位一致率 |
|---|---|---|---|
| COUNT | 35/40 | 30/35 | 0.989 |
| POSITIVE | 39/40 | 38/39 | 0.996 |
| BOUNDED | 40/40 | 36/40 | 0.983 |
| MIXED | 33/40 | 27/33 | 0.992 |

### 3.7 その他の診断

- candidate optimiser: grad_inf > 1e-8（WARNING 閾値、Phase 9C と同じ）は COUNT・PTN の候補で評価の 27–47%、
  max grad_inf は 5e-4。logit-normal / log-normal は全件 ≤ 1e-12。失敗した候補評価はすべて失敗 fit の中にある。
- E-step の CMP 評価域 guard（Newton step の半減）が働いた fit は全 stage で 26 件（回数は `fit_results.csv` に記録）。
- support gate（Level 0）の群判定は、全列で真の support group と一致した（audit `support_gate_misassigned_columns = 0`）。

---

## 4. INTERPRETATION（FACT からの妥当な解釈）

1. **support だけでは決まらない候補どうしでも、score による分布形の選択は高い率で働いた。**
   true Z の selector では 3 群とも 0.95 以上で、MCEM で Z を推定しても score-decided の recovery は selector とほぼ同じ水準（0.94–1.00）だった。
   このため、**Z を推定することによる劣化は、この条件では精度より数値的頑健性の側に出ている**と読める。
2. **誤りは「数学的に近い候補」の間に集中している。** COUNT は Poisson と COM-Poisson（ν=1 で一致する入れ子の関係）の間、
   BOUNDED は TN(0,1)（σ=0.25）と logit-normal（η=0 付近で分散が近い）の間。calibration で最小 KL が小さかったペアと一致しており、
   selector の失敗というより識別情報の少なさを反映していると考えるのが自然である（ただし因果の証明ではない）。
3. **EASY/HARD は Auto の数値失敗を強く左右した。** 失敗 18 件はすべて EASY（COUNT と POSITIVE では τ=2 の強い信号）で、
   大きなカウント（CMP ν=0.5 の列で数百〜千）が incumbent loading のまま CMP 候補を評価不能な領域に置いたことが主因である。
   これは「EASY ＝分布が離れている」と「EASY ＝ η の幅が大きく数値が荒い」が同じ τ で動く設計上の交絡でもある。
4. **Auto は Oracle とほぼ同じ Z 推定を与え、Support-only より平均的に良かった**（paired gap が全群で負、Auto が良かった dataset は 70–88%）。
   ただし差の多くは Support-only が少数の dataset で大きく崩れることによる（RMSE > 1.0 の外れ値）。
5. 開始点依存は小さい（列単位一致 0.98 以上）が、完全一致しない dataset が群ごとに 1–6 件ある。

## 5. UNRESOLVED（まだ答えられないこと）

- 失敗した 18 fit で、仮に計算が通っていたら何を選んだか（不明。失敗は不正解として数えている）。
- 候補の開始 loading を incumbent ではなく評価可能な点（例: 0）にすれば失敗が消えるか。**protocol の変更になるので未実施**（Human 判断）。
- τ による「分布の近さ」と「潜在信号・数値範囲」の交絡を分けた場合の recovery（今回の設計では分離できない）。
- nuisance（N, ν, σ, c）を推定する場合、K と同時に選ぶ場合、実データの場合。いずれも未実験。
- 候補収束の WARNING（grad_inf 最大 5e-4）が score margin に与える影響の定量評価（margin は `margin_summary.csv` にあるが、誤差評価はしていない）。
- 離散 search cost（penalty）の扱い（Phase 9B U5 のまま）。

## 6. BLOCKED family

geometric、negative binomial、gamma、exponential、weibull（以上は natural parameter の定義域が ℝ でない）、
beta（2 次元・制約付き）、student-t（指数型分布族でない）、categorical（ベクトル natural parameter）、
inverse Gaussian（定義域制約）。理由の詳細は compatibility audit §3 と `tables/family_compatibility.csv`。

## 7. Claim boundary

**言ってよいこと**
- 今回実装した 9 family / 10 candidate model（COUNT 5、POSITIVE 2、BOUNDED 3）＋ Gaussian control について、
  この synthetic 条件（K=3 固定、nuisance 固定、matched generator、n=75/150）での recovery、群間の違い、EASY/HARD と n の違い、
  Auto / Oracle / Support-only の Z 推定の差、start 依存、数値失敗の率と原因。

**言ってはいけないこと**
- 一般に最適な分布を自動選択できる／すべての指数型分布族に対応した／実データでも選べる／人間より優れている／
  K と family を同時に選べる／nuisance まで自動選択できる／BLOCKED family にも対応した。
- 失敗 fit を除いた recovery（score-decided の値）だけを「Auto の性能」として示すこと（必ず all-columns の値と失敗数を併記する）。
- lineage E の結果を本文の提案手法の結果として書くこと。
