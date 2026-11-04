# E1：M1 の誤指定効果は Z の点推定にどれだけ依存していたか（Issue #141、2026-11-04）

- 位置づけ: **CHARACTERIZATION（M1 の再評価）**。M1 の結論を保つことも否定することも、成功条件にしていない。
- 系列: E-consistent per-column（`DualExpFamLSMPerColumnConsistent`、`numerics_mode="consistent"`、experimental prototype、本文採用不可）。M1 と同じである。
- 実行コード:
  - `expfam/src/experimental/run_z_point_estimator_e1.py`。設計は commit `8a39ecf` で固定した。`ddfb650` は smoke の照合範囲だけの修正である。
  - full は `ddfb650` で実行した（追跡ファイルの未 commit 変更なし）。
- 結果: `expfam/results/z_point_estimator_e1/e1_20261104/`（`smoke/`、`full/`）。状態ファイルは repository 外に置いた（`STATE_ARCHIVE.md`）。
- 環境: Python 3.13.14 / numpy 2.3.5 / scipy 1.16.3 / scikit-learn 1.8.0、Windows。M1 と同じ Python と numpy である。
- **M1 の artifact と報告書は変更していない。** E1 は別のディレクトリに書き出した。

---

## 1. 問いと設計

**問い**: M1 の ΔRMSE_Z（wrong − correct）は、Z の点推定を `last` から `mean5` / `mode` に替えても保たれるか。model seed を替えたときの変動は、効果に比べて無視できる大きさか。

| 項目 | 内容 |
|---|---|
| データ・mask・モデル設定 | M1 と同一。n=75、d=12、K=3、G3/B6/P3 vs G3/P6/P3、Bernoulli-Y、dense = pool の 100%、sparse = pool の 10%、L=5、8 反復 |
| Part A | M1 の 80 fit の replay（model seed `1343000 + r`）。状態を保存した |
| Part B | model seed `1344000 + r` の 80 fit（#141 で事前に指定）。**model seed は初期値と MC の乱数の両方を決める**ので、A と B の差は「初期値 + MC」による実行間の変動であり、MC 誤差だけではない |
| 点推定 | `last` = 最後の 1 サンプル、`mean5` = 最後の E-step の 5 サンプルの平均、`mode` = θ̂ を固定した `ln p(X, Y_train | Z, θ̂) + ln p(Z)` の joint mode |
| mode の求め方 | #128 と同じ solver。Z0 = last、grad tol 1e-8、最大 200 反復。training mask だけを使う |
| 局所解の診断 | Z0 = mean5 からも mode を求めて比べた |
| 主要な指標 | Part A、`mode` の paired ΔRMSE_Z（dense / sparse）と interaction = Δ_sparse − Δ_dense |
| mode が OK でない fit | 補完しない。complete pair（両 arm とも OK）で集計し、完了数は別に報告する |

`mode` は**最初から正しい推定値として扱っていない**。縮小した推定値であり、θ̂ が正しいことを前提にした条件付きの mode である。

## 2. 技術的な有効性

| 項目 | 結果 |
|---|---|
| fit | 160/160 成功。retry・置き換え・seed の救済・失敗はいずれも 0 |
| Part A の再現 | **REPRODUCED**。M1 の `rmse_Z`・`gram_corr`・held-out の log score・AUC・ŵ0・ŵ の 480 値すべてで abs diff = 0.0。対比較の符号の反転もなし |
| joint mode | 320 回の計算（2 つの開始点 × 160）がすべて OK。max\|∇Φ\| の最大は 9.3e-9、H の最小固有値の最小は 0.099、反復数は 4〜33 |
| 局所解 | 2 つの開始点の mode は、159/160 fit で一致した（差の中央値 1.2e-10）。B rep14 sparse wrong だけ別の局所解に収束した（最大差 1.63、Φ は Z0 = last 側が 0.15 高い、RMSE 0.869 vs 0.874）。主要な解析は事前の規則どおり Z0 = last の mode を使った |
| 実行時間 | 1881.5 秒 |
| smoke | rep01 dense の 2 fit で 12 値が abs diff 0.0。ただし `smoke/reproduction.json` の verdict は、判定コードが smoke の範囲外の 78 fit を欠測として扱ったため NOT_REPRODUCED と書かれている。commit `ddfb650` で修正し、artifact はそのまま残した |

NOT_STATIONARY や HESSIAN_NOT_PD は 0 件だったので、complete pair はすべての条件で 20/20 だった。

## 3. 一次結果

### 3.1 誤指定の効果（paired ΔRMSE_Z = wrong − correct、20 replicate）

| part | 点推定 | dense: 平均 / 中央値 / IQR / 正の数 | sparse: 平均 / 中央値 / IQR / 正の数 | interaction: 平均 / 中央値 / 正の数 |
|---|---|---|---|---|
| A | last（= M1） | +0.020 / +0.025 / [0.011, 0.034] / 15 | +0.066 / +0.060 / [0.028, 0.101] / 17 | +0.046 / +0.041 / 15 |
| A | mean5 | +0.017 / +0.007 / [0.001, 0.025] / 16 | +0.069 / +0.069 / [0.038, 0.105] / 19 | +0.052 / +0.042 / 18 |
| **A** | **mode（主要）** | **+0.015 / +0.006 / [−0.003, 0.011] / 13** | **+0.076 / +0.078 / [0.055, 0.104] / 18** | **+0.061 / +0.064 / 19** |
| B | last | +0.002 / −0.001 / [−0.015, 0.020] / 9 | +0.055 / +0.060 / [0.020, 0.104] / 16 | +0.053 / +0.049 / 17 |
| B | mean5 | +0.004 / +0.000 / [−0.013, 0.009] / 10 | +0.075 / +0.067 / [0.048, 0.115] / 19 | +0.071 / +0.059 / 18 |
| B | mode | +0.009 / +0.004 / [−0.003, 0.007] / 13 | +0.087 / +0.077 / [0.050, 0.127] / 20 | +0.078 / +0.079 / 19 |

### 3.2 実行間の変動（Part A vs Part B）

| 点推定 | 1 fit の RMSE_Z の \|A − B\| の中央値（dense / sparse） | Δ_dense: \|A − B\| の中央値 / 相関 / 符号一致 | Δ_sparse: \|A − B\| の中央値 / 相関 / 符号一致 | interaction の符号一致 |
|---|---|---|---|---|
| last | 0.029 / 0.043 | 0.027 / 0.03 / 8/20 | 0.063 / 0.00 / 13/20 | 14/20 |
| mean5 | 0.017 / 0.041 | 0.022 / 0.50 / 10/20 | 0.043 / 0.52 / 18/20 | 16/20 |
| mode | **0.0014** / 0.044 | **0.0015 / 0.68 / 20/20** | 0.046 / 0.44 / 18/20 | 20/20 |

### 3.3 点推定による膨張（Part A、`RMSE_last − RMSE_mode`、平均）

| | correct | wrong |
|---|---|---|
| dense | 0.135 | 0.140 |
| sparse | 0.205 | 0.195 |

20/20 の fit すべてで `last` の方が大きかった。

### 3.4 副次的な指標（中央値。Part A と Part B でほぼ同じ）

| regime | 点推定 | held-out AUC | held-out mean log score | Gram 相関 |
|---|---|---|---|---|
| dense | last / mean5 / mode | 0.741 / 0.761 / 0.771 | −0.572 / −0.541 / −0.533 | 0.778 / 0.859 / 0.887 |
| sparse | last / mean5 / mode | 0.593 / 0.611 / 0.632 | −0.654 / −0.622 / −0.607 | 0.313 / 0.446 / 0.504 |

- 誤指定による held-out の差（wrong − correct）は、どの点推定でも sparse で負（correct が良い）の側だった。
  - 例: `mode` の Part A では、AUC の差の平均は −0.018（正は 6/20）。
- dense では、どの点推定でも差は ±0.005 以内だった。

## 4. 言えること（OBSERVED、この 1 条件について）

1. **sparse-Y での悪化は、点推定にも seed にも頑健だった。**
   - 6 つの組み合わせ（3 つの点推定 × 2 つの seed）のすべてで、平均 Δ_sparse は +0.055〜+0.087、正の数は 16〜20/20 だった。
   - `mode` では、M1 の報告（+0.066）よりやや大きかった（A +0.076、B +0.087）。
2. **M1 の dense の効果（+0.020、15/20）は、`last` の点推定では seed を替えると再現しなかった。**
   - B の `last` では +0.002（9/20）で、replicate ごとの A と B の相関は 0.03 だった。
   - **M1 の dense の結果の大部分は、実行ごとの変動だった。**
3. **`mode` では、dense の効果は小さいが、実行間でほぼ同じ値になった。**
   - 平均は +0.015 / +0.009、中央値は +0.006 / +0.004、正の数は 13/20。分布は一部の replicate に引っ張られて右に歪んでいる。
   - replicate ごとの A と B の差は中央値 0.0015 で、符号は 20/20 で一致した。
4. **interaction（sparse で効果が大きい）は、`mode` では 19/20 が正だった**（A・B とも）。M1 の `last` の 15/20 より揃っていた。
5. **`last` の点推定は、どの条件でも RMSE_Z を 0.13〜0.21 大きくした。** また、held-out の AUC・log score と Gram 相関も悪かった。
   - 膨張の大きさは arm によってわずかに違った。dense では wrong 側、sparse では correct 側が大きかった。
   - そのため、`last` は dense の Δ を約 0.005 大きく、sparse の Δ を約 0.01 小さく見せていた。
6. **sparse の実行間の変動は、点推定では消えなかった。**
   - 1 fit の RMSE の \|A − B\| は、どの点推定でも約 0.044 だった。
   - これは、sparse では θ̂ 自体が初期値と MC によって変わることを示す。点推定のノイズではない。

## 5. 言えないこと

- `mode` が「正しい」Z の推定値であること。この条件で、真の Z に近く、実行間で安定し、held-out 予測が良かったことだけが言える。
- sparse の実行間の変動が、初期値（局所解）によるのか MC によるのか。現行のコードでは、model seed が両方を決める。
- 一般の誤指定の法則。結果は 1 つの固定条件と 20 replicate × 2 seed の記述であり、検定・信頼区間は計算していない。
- 他の generator・K・family 構成や、K を選ぶ場面での `mode` の振る舞い。高次元の K では H が不定になりうる（#128）。

## 6. 研究上の結論と改善案

### 6.1 M1 の結論への影響

- **sparse-Y では誤指定が Z の回復を悪くする**という M1 の中心の観察は、評価方法と seed に頑健だった。むしろ `mode` では、悪化の大きさと揃い方が強まった。
- **dense-Y の効果については書き直しが必要である。**
  - M1 の「dense でも 15/20 で wrong の方が悪い」は、`last` の点推定と実行間の変動の産物の部分が大きかった。
  - `mode` で見ると、dense の効果は小さく（中央値 +0.004〜+0.006）、正の replicate は 13/20 だった。
  - 「dense では誤指定の影響はほぼ無視できる大きさ（一部の replicate を除く）」と記述するのが妥当である。
- M1 の報告書（#135）は書き換えない。書き直しの案は、本書を参照する形で #135 の review か canonical docs の更新時に Human が判断する。

### 6.2 今後の Z の評価方法（提案、Human Gate）

1. **主要な点推定は `mode` とし、`last` を併記する**（過去の結果との連続性のため）。
   - この条件では、`mode` は dense の評価を実行間でほぼ決定的にした（1 fit の \|A − B\| が 0.029 → 0.0014）。
   - また、すべての fit で solver の状態は OK だった。
2. `mean5` は中間である。`last` よりは良いが、dense の実行間の変動が残った（0.017）。
3. 採用の前に確認すべきこと:
   - (a) mode が NOT_STATIONARY / HESSIAN_NOT_PD になりやすい条件（K > K_true、高い K）での扱いの規則を事前に決める。
   - (b) スケール（縮小）の影響。Procrustes は回転だけなので、ノルム比も併記する。
4. **sparse では seed 1 つの比較は不十分である。**
   - Δ_sparse の \|A − B\| の中央値は 0.046 で、効果の大きさ（約 0.08）の半分程度ある。
   - 今後の sparse 条件の実験は、少なくとも 2 つの model seed で行うか、seed 間の平均を主要な指標にすべきである。

### 6.3 次の優先順位

| 順位 | 候補 | 理由 |
|---|---|---|
| 1 | **M2（Bernoulli → Gaussian）を、E1 の評価方法で行う** | `mode` を主要にして seed を 2 つにすれば、sparse でも符号は 18〜20/20 で安定した。誤指定の種類の比較は、研究の主な問い（#138）に直接答える。M1 の runner と E1 の wrapper を再利用でき、新しい arm だけで済む |
| 2 | **E2（#142）**: 初期値の seed と MC の seed を分け、L を変える | sparse の残りの変動（約 0.044）が、局所解によるのか MC によるのかを決める。ただし推定アルゴリズムの呼び出しの改修（seed の分離）が必要で、Human Gate |
| 3 | E3（#143）: 単純なベースライン | EM なしで安い。いつでも行えるが、研究判断への影響は 1・2 より小さい |

**M2 を先にする理由**
- E1 で、評価方法（`mode` + 2 seed）を変えると sparse の結論が安定することが分かった。そのため、M2 の比較を解釈可能な精度で行える。
- E2 は推定の信頼性の改善だが、seed の分離という推定コードの改修を伴う。

逆に、Human が「sparse の実行間の変動の原因を先に知りたい」と判断するなら、E2 を先にする。

---

## 付録: 出力ファイル

| ファイル | 内容 |
|---|---|
| `full/fit_runs.csv` | fit ごとの status、ŵ0・ŵ、mode の診断（2 つの開始点）、runtime |
| `full/fit_estimators.csv` | fit × 点推定ごとの RMSE_Z、Gram 相関、held-out の log score・AUC、ノルム比 |
| `full/paired_deltas.csv` | part × replicate × 点推定ごとの対比較の差と interaction |
| `full/summary.json` | 本書の表の数値 |
| `full/reproduction.json`、`reproduction_diffs.csv` | Part A と M1 の 480 値の照合 |
| `full/protocol.json`、`preflight.json`、`runinfo.json` | 設計、M1 の preflight（全 PASS）、環境と git |
