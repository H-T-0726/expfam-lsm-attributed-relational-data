# 2 つの追加実験の結果（Experiment A・B、2026-11-27）

- **種類**：探索的（EXPLORATORY）。どちらの protocol も本実験の fit 前に凍結した（`frozen_protocol_A.md`、`frozen_protocol_B.md`）。confirmatory evidence と呼ばない。
- **系列**：lineage E-scalar（`DualExpFamLSMConsistent`、`numerics_mode="consistent"`、`failure_policy="fail_fast"`）。prototype であり、fixed 系列の結果ではない。本文に採るかどうかは Human Gate。
- **基準の呼び方**：C_Q は Q 関数に基づく完全データ型の基準で、Schwarz BIC ではない。C_Lap はラプラス近似で、厳密な周辺尤度ではない。
- **一次資料**
  - fit ごとの結果：`expA/run/per_fit.csv`、`expB/run/per_fit.csv`
  - 解析結果：`exp*/analysis/`
  - 独立集計：`independent_tally/`（runner を import しない別スクリプトで、25/25 項目が一致）
  - 実行記録：`runinfo.csv`
  - NPZ の SHA-256：`states_sha256_manifest.csv`（NPZ は git 管理外）
  - 公平性の評価：`fairness_audit.md`

## 0. 各実験が答えた研究問い

| 実験 | 研究問い | 答え（OBSERVED、この条件のみ） |
|---|---|---|
| A | X の全列が Bernoulli のとき、学習に使う Y が少ないと、X の分布誤指定（Bernoulli→Poisson）による Z の推定誤差の悪化は大きくなるか | **大きくなった**。$I$ の平均は +0.211 で、20/20 データで正だった。ただし sparse では正指定のモデル自身も Z をほとんど回復できていない（§A-4） |
| B | K-exp2 の regime b、$K_{TRUE}=9$ で見られた C_Q の差（Gaussian-X は 10/10、Poisson-X は 0/10）は、潜在変数に関する情報量の差で説明できるか | **C_Q については説明と整合した**。期待 Fisher 情報の trace を Poisson にそろえた Gaussian（GM）は、C_Q で 0/10（10/10 が過小）だった。C_Lap ではどの条件も多くが正解だった |

---

## Experiment A：Y の観測量と、X の分布誤指定の悪影響

### A-1. 実行（OBSERVED）

- 本実験は 160/160 fit が OK だった。FAILED、INCOMPLETE、再実行、seed の置換はない。
- 20 データすべてで $\Delta$ と $I$ を計算できた。
- M-step で Q が減少した回数は全 fit で 0。
- 技術検証として、非学習ペアの Y を反転しても勾配・strict Q・M-step が変わらないことを確認した。pilot は 9/9 OK で、同じ seed での再現も一致した。

### A-2. 主要指標（OBSERVED、20 データ、2 初期値の平均）

指標は Procrustes RMSE_Z（最後の MC サンプル）。$\Delta$ = 誤指定（Poisson）− 正指定（Bernoulli）、$I = \Delta_{\rm sparse} - \Delta_{\rm dense}$。

| 量 | 平均 | 中央値 | SD | 正 / 負 | bootstrap 95% | t 95% |
|---|---:|---:|---:|---:|---|---|
| **$I$（primary）** | **+0.2113** | +0.2145 | 0.0742 | **20 / 0** | [0.179, 0.242] | [0.177, 0.246] |
| $\Delta_{\rm dense}$ | +0.0116 | +0.0088 | 0.0153 | 17 / 3 | [0.006, 0.019] | [0.004, 0.019] |
| $\Delta_{\rm sparse}$ | +0.2229 | +0.2213 | 0.0668 | 20 / 0 | [0.194, 0.251] | [0.192, 0.254] |

- 両側 sign test の p 値は、$I$ が $1.9\times10^{-6}$、$\Delta_{\rm dense}$ が 0.0026。記述用であり、検定の結論には使わない。
- 副次指標（Z の L サンプル平均の RMSE_Z）でも $I$ = +0.220 で、20/20 が正だった。

**各条件の水準（RMSE_Z の平均）**

| | Bernoulli（正） | Poisson（誤） |
|---|---:|---:|
| dense | 0.383 | 0.395 |
| sparse | 1.024 | 1.247 |

図：`expA/analysis/figures/figA_delta_and_interaction.png`

### A-3. 副次指標（OBSERVED）

**gram 相関（fit の平均）**

| | Bernoulli | Poisson |
|---|---:|---:|
| dense | 0.863 | 0.855 |
| sparse | 0.241 | 0.061 |

**test ペアの Bernoulli 対数スコアの差（正 − 誤）**

- dense：+0.0025。正指定が良かったのは 15/20。
- sparse：**−0.0334**。**誤指定のほうが良かったのが 19/20**。

**事後の記述**（protocol で事前に決めた指標ではない）

- $\hat w$ の平均
  - dense：Bernoulli 0.991、Poisson 0.983
  - sparse：Bernoulli 0.759、Poisson 0.428（真値は 1.0）
- 推定 Z の二乗平均はどの条件も ≈ 1。

### A-4. 解釈と、ほかの説明

**INTERPRETATION**

- この設計では、Y の学習ペアを 10%（約 1,006 ペア）に減らすと、X の分布誤指定による Z の悪化が約 19 倍になった（0.012 → 0.223）。
- 向きは、列ごとの family の M1・MX2（B6→P6）で観測された向きと同じだった。

**注意と、ほかの説明**

1. sparse では、正指定のモデルでも RMSE_Z ≈ 1.02、gram 相関 ≈ 0.24 で、Z の回復が弱い。
   - 「誤指定の悪影響が大きい」は、「両方とも回復が弱いなかで、誤指定のほうがさらに悪い」という意味になる。
   - Poisson-sparse の gram 相関は 0.06 で、ほぼ回復していない。
2. sparse での $\Delta$ の大きさの一部は、誤指定の X が Z を Y の情報から引き離した結果とも読める。
   - たとえば $\hat w$ の縮小（事後の記述）。
   - 一方、held-out の Y 予測は sparse で誤指定のほうが良かった。
   - つまり「Z の回復が悪い」ことと「Y の予測が悪い」ことは一致しない。どちらを主要な目的にするかで評価が変わる。
3. 一般性は限られる。今回は 1 つの信号設計（K=3、F = √2.5 Q、w = 1）、1 種類の誤指定（Bernoulli→Poisson）、1 つの sparse 率で、MCEM 10 反復の固定予算である。
4. MX2 では、誤指定の種類によって交互作用の向きが逆になったと報告されている（研究結果集 04 章）。今回の結果は B→P の 1 種類だけについてのもので、「Y が少ないほど誤指定の悪影響が大きい」という一般法則を支持するものではない。

---

## Experiment B：情報量（trace）をそろえた Gaussian-X と Poisson-X

### B-1. 実行（OBSERVED）

- 本実験は 600/600 fit が OK だった。
- C_Lap の評価は 598 OK、NOT_STATIONARY 2 件。
  - `B-G01-r000-k02-t1`：勾配の最大値 3.0e-8
  - `B-P-r007-k03-t1`：勾配の最大値 1.9e-7
  - どちらも grad_tol 1e-8 を満たさなかった。
  - protocol どおり、この 2 データの C_Lap は INCOMPLETE とし、分母は 10 のまま。許容値の変更や再評価はしていない。
- 再実行・seed の置換はない。M-step で Q が減少した回数は 0。
- 実行時間は 6,923 秒（4 worker）。
- 3 条件で Z・F・Y の hash は一致し、X の hash は異なった（独立集計）。

### B-2. 主要指標：選ばれた K（OBSERVED、各 10 データ、$K_{TRUE}=9$）

| 条件 | 基準 | 正解 | 過小 | 過大 | INCOMPLETE | 選ばれた K | 2 初期値で選択が不一致 |
|---|---|---:|---:|---:|---:|---|---:|
| G01（Gaussian、分散 0.1） | C_Q | **10** | 0 | 0 | 0 | すべて 9 | 0 |
| G01 | C_Lap | 9 | 0 | 0 | 1 | 9 ×9 | 0 |
| GM（Gaussian、trace をそろえた分散 ≈ 0.76） | C_Q | **0** | **10** | 0 | 0 | 2, 4, 4, 5, 5, 5, 6, 6, 6, 7 | 3 |
| GM | C_Lap | 8 | 2 | 0 | 0 | 8, 8, 9 ×8 | 0 |
| P（Poisson） | C_Q | **0** | **10** | 0 | 0 | 4, 5, 5, 5, 5, 5, 5, 6, 7, 7 | 5 |
| P | C_Lap | 9 | 0 | 0 | 1 | 9 ×9 | 1 |

**対応比較（同じ replicate、`paired_by_replicate.csv`）**

- C_Q
  - G01 → GM：10/10 の replicate で正解から過小に変わった。
  - GM と P の選択 K：GM < P が 5、GM = P が 3、GM > P が 2。
  - 平均の K − 9：GM −4.0、P −3.6。
- C_Lap
  - GM が過小になった 2 つ（r003、r008、K=8）は、G01 と P ではどちらも 9 だった。

**基準曲線**（`criterion_curves.csv`、図 `figB_criterion_curves.png`）

- C_Q（C(K) − C(9) の中央値）
  - G01：K=1..8 で単調に正。
  - GM と P：どちらも K≈5 付近に約 −370〜−430 の谷があり、似た形だった。
- C_Lap：3 条件とも K=9 が最小（中央値）。

**情報量の診断**（技術検証、真の F から計算、`expB/tech_check_v2/information_diagnostics.csv`）

| 条件 | 期待 Fisher の trace | 固有値 | 条件数 |
|---|---:|---|---:|
| G01 | 75 | すべて 8.33 | 1.00 |
| GM | 9.70〜9.90 | 1.08〜1.10 で等方 | 1.00 |
| P | GM と同じ（相対誤差 < 1e-12） | 0.99〜1.21 | 1.11〜1.22 |

- $\sigma^2_{\rm matched}$ は 0.758〜0.773。
- 期待 Fisher の trace は、10 replicate の全 9 固有値を記録した。

### B-3. 数理的な整理（DERIVED）

- regime b では $F = sQ$、$s^2 = 0.5d/K$ なので $F^\top F = s^2 I$。$K=9$、$d=15$ では $s^2 = 0.833$、$\mathrm{tr}(F^\top F) = 7.5$ になる。
- Gaussian の Fisher は $F^\top F/\sigma^2$。分散 0.1 では trace が 75。
- Poisson の期待 Fisher は $F^\top \mathrm{diag}(e^{\lVert f_l\rVert^2/2})F$ で、trace は約 9.8。
- したがって G01 は、P の約 7.6 倍の情報（trace）を持つ。GM は分散を 0.1 → 約 0.76 にして、この倍率を 1 にした。

### B-4. 解釈と、ほかの説明

**INTERPRETATION**

- K-exp2 で見られた「Gaussian-X は C_Q で成功、Poisson-X は失敗」という差は、この設計では**分布族そのものより、X が持つ潜在変数の情報量（の trace）の差と整合する**。
- 理由：trace を Poisson にそろえると、Gaussian も C_Q で 0/10（すべて過小）になり、C_Q の曲線の形も Poisson と似た。
- C_Lap は、情報量をそろえても多くのデータで $K_{TRUE}$ を選んだ。C_Q の過小選択は、「情報量が少ないと C_Q の K に依存する項が勝つ」という既知の傾向（研究結果集 07 章）と整合する。

**仮説ごとの判定**（protocol の 5 つの結果）

1. trace をそろえると Gaussian も過小選択する：C_Q では**観測された**。C_Lap では観測されなかった（8/10 正解、過小 2）。
2. そろえても Gaussian は成功する：C_Q では観測されなかった。C_Lap では、GM・P とも多くが正解で、情報量の差がなくても成功した。
3. 初期値の不安定性が支配する：支配はしていない。2 初期値の選択の不一致は、GM C_Q で 3/10、P C_Q で 5/10。初期値ごとに選んだ K（各 2 個）もすべて 2〜7 で、どちらの初期値でも過小になる。
4. C_Q と C_Lap で傾向が違う：**観測された**。
5. INCOMPLETE が多い：多くない。C_Lap の 30 セル（3 条件 × 10 データ）のうち 2 つ（G01 r000、P r007）で、C_Q は 0。

**ほかの説明・注意**

1. trace の一致は 1 つの要約にすぎない。方向ごとの情報（Poisson の条件数は約 1.15）、z ごとの情報の変動、尤度の形は違う。「完全に公平な比較」「分布の因果効果」とは言わない。
2. Gaussian-X は各列の分散を推定する。推定された分散（NPZ に保存）は今回集計していない。推定上の情報量が設計上の値と違う可能性がある（未確認）。
3. C_Q の罰則項数は Gaussian で +d 多い。ただし G01 と GM は同じ罰則なので、G01 vs GM の対比はこの違いの影響を受けない（評価 A）。
4. 1 つの $K_{TRUE}$（9）、1 つの信号規則、10 データ、MCEM 10 反復。

---

## 既存の研究との違い（比較の評価は `fairness_audit.md`）

- **A と M1（#134）・MX2（PR #168、未 merge）**：評価 C（記述のみ）。
  - M1 の Δ：dense +0.020 → sparse +0.066。MX2 B6→P6：+0.012 → +0.082。
  - 向きは同じだが、系列（E-3 の列ごとの family と E-scalar）、n、d、列の構成が違う。
  - 今回は全列が同じ family の構造で同じ向きが出た、ということまでが言える。
- **B と K-exp2 の本実験**：評価 C。
  - K-exp2 A-b-9（Poisson-X）：C_Q 0/10、C_Lap 6/10。今回の P：C_Q 0/10、C_Lap 9/10（INCOMPLETE 1）。
  - K-exp2 の Gaussian-X は C_Q 10/10。今回の G01 も 10/10。
  - データと乱数の使い方は違う。C_Q の差は今回も再現したが、C_Lap の正解数は別データなので並べて比較しない。
  - 新しく分かったのは、**同じ Z・F・Y で X の分散だけを変えると、C_Q の成否が入れ替わる**こと。
- fixed 系列の Exp4 などとの数値比較はしない（評価 D）。

## 未解決の点

- A：sparse で、誤指定が Z の回復を悪くしても held-out の Y 予測を良くした理由。$\hat w$ の縮小との関係は事後の観察で、まだ検証していない。
- A：sparse 率、誤指定の種類（B→G など）、K、信号の強さを変えたときの一般性。
- B：推定された Gaussian の分散の値と、推定上の情報量。
- B：C_Lap で GM が過小になった 2 データの原因。
- B：$K_{TRUE}$ を 7 などに変えたときに、同じ対応が保たれるか。
- どちらも MCEM 10 反復の固定予算で、収束を確認していない。
- この報告は同じ AI による作成と点検で、人間の第三者査読ではない。

## 修論での使い方（判断は Human Gate）

- A：「Y が疎なとき、X の分布誤指定の悪影響が大きくなりうる」という例示には使える。条件はすべて書き、正指定でも回復が弱いことと、held-out の結果の逆転を必ず併記する。一般法則としては書けない。
- B：「C_Q が $K_{TRUE}$ を当てられるかは、X が持つ潜在変数の情報量に強く依存し、family の違いはその一部として説明できる」ことの探索的な根拠として使える。
  - G01 vs GM は評価 A の対応比較。GM vs P は評価 B の対応比較で、trace だけをそろえたことを明記する。
- どちらも prototype（E-scalar）の探索的な結果であり、本文の結論にするには人間の判断が必要。
