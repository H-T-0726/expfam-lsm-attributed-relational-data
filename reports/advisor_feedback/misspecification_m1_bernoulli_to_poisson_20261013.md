# Experiment M1 — Bernoulli で生成した属性を Poisson として誤指定したときの Z 回復（2026-10-13）

- 位置づけ: **CHARACTERIZATION**（Issue #134）。#130 の research synthesis で凍結した最小実験 M1 を、設計どおり 1 回だけ実行した。
  correct family が勝つことは成功条件ではない。
- 系列: **E-consistent per-column**（`DualExpFamLSMPerColumnConsistent`、`numerics_mode="consistent"`、experimental prototype。本文採用不可）。
  Phase 9K の K 選択の pipeline（Candidate B / C_Lap / C_Q）とは別の fit 経路であり、Phase 9K の数値とは混ぜない（KI-002）。
- 実行コード: `31ebcac`（`expfam/src/experimental/run_misspecification_m1.py`）。実行時、追跡対象のファイルに未 commit の変更はなかった（`git_dirty_tracked: false`）。
- 結果: `expfam/results/misspecification_m1/phase_m1_20261013/`
  （`protocol.json`, `preflight.json`, `mask_provenance.csv`, `fit_results.csv`, `paired_rmse_z.csv`, `heldout_y.csv`, `parameter_errors.csv`, `technical_diagnostics.csv`, `summary.json`, `runinfo.json`）
- 新しい EM: 計画 80、試行 80、成功 80。

---

## 1. Research question

同じ人工データで、Bernoulli で生成した 6 列だけを Poisson として fit したとき（support-compatible な誤指定）、共通の潜在構造 Z の回復はどう変わるか。
その変化は、Y の学習に使えるペアが多い場合（dense）と少ない場合（sparse）で違うか。

事前の仮説（#134）は「dense では影響が小さく、sparse では大きいかもしれない」だが、これは成功条件ではない。

---

## 2. Design

| 項目 | 値 |
|---|---|
| データ | Phase 9K と同じ generator（`canonical-clean-mixed-v1`）と条件: n = 75、d = 12、K_true = 3、X = Gaussian×3 / Bernoulli×6 / Poisson×3、Y = Bernoulli、σ²_x = 1、f_scale = √2、w0 = −1、w = 1 |
| replicate | 20（data_seed 1001001..1001020）。各 replicate で full data を 1 回だけ生成し、4 つの fit で共有 |
| arm | **correct** = G3 / B6 / P3、**wrong** = G3 / **P6** / P3（列 3〜8 だけを変更）。生の 0/1 X をそのまま使い、変換しない |
| K | K_fit = K_true = 3 に固定。K の sweep・選択なし |
| family | 各 arm で固定。自動選択なし |
| Y regime | dense（学習候補 pool の 100%）、sparse（pool の 10%） |
| 推定 | `em_runner.run_em_experimental`（`numerics_mode="consistent"`、`failure_policy="fail_fast"`）、L = 5、反復 8、`validate_support=True`、`allow_support_mismatch=False` |
| seed | test mask は `1341000 + r`、sparse の間引きは `1342000 + 10r + regime_index`、モデルは `1343000 + r`（同じ replicate の 4 fit で共通） |
| 規模 | 20 × 2 regime × 2 arm = **80 fit** |
| 主要な指標 | 回転だけの Procrustes RMSE_Z。`delta_RMSE_Z = RMSE_wrong − RMSE_correct`（正 = wrong の方が悪い） |
| 副次的な指標 | held-out Bernoulli の plug-in mean log score（既存の `heldout_bernoulli_mean_log_score`）、held-out の AUC / AP、`|ŵ − 1|`、`|ŵ0 + 1|`、Z Z^T の Gram 相関、status / warning / runtime |
| X の指標 | 計算していない（family をまたいだ生の尤度で Bernoulli と Poisson の優劣を決めないため） |

---

## 3. Why Bernoulli→Poisson

- 0/1 は Poisson の support の中にあるので、**データを表現できない誤り（support-invalid）ではなく、同じデータを別の正当な確率モデルで説明する誤り**である（#130 §4 の A 型）。
  崩壊や比較不能な尤度が起きないので、「分布を間違えると Z にどう効くか」を最も素直に測れる。
- #130 で確認した既存の証拠では、この比較は Y がほぼノイズのない Gaussian-Y の条件（fixed Exp4 Scenario C）にしかなく、差は 0 だった。
  現在の研究の中心である Bernoulli-Y・mixed-X の条件では、一度も測られていなかった。
- 違いは観測モデルだけである。Bernoulli は平均 σ(η)・曲率 σ(η)(1 − σ(η)) ≤ 1/4、Poisson は平均 exp(η)・曲率 exp(η) で、同じ 0/1 に対して η の意味と各列の重みが変わる。
  したがってこの比較は「分布の正しさだけ」の効果ではなく、**平均関数と曲率の形を含めた観測モデル全体の違い**の効果である。

---

## 4. Mask definition

replicate ごとに次の順で作った（`mask_provenance.csv` にすべての replicate の hash とペア数を記録）。

1. 上三角の全 2775 ペアから、固定の **10% = 278 ペア**を test にする（`make_pair_split`）。correct / wrong、dense / sparse のすべてで共通。
2. 残りの **2497 ペア**が学習候補 pool である。
3. **dense** の学習 = pool の 100% = 2497 ペア（全ペアの約 90%）。
4. **sparse** の学習 = pool から無作為に 10% = **250 ペア**（全ペアの約 9%）。

「観測率 1.0 / 0.1」は **test を除いた学習候補 pool に対する割合**であり、全ペアに対する割合ではない。
また、これは観測するペアを減らす操作であり、edge 密度（w0）を下げる操作ではない。full Y の edge 密度は 0.313〜0.341 で、Phase 9K と同じである。

**preflight（EM の前、すべて PASS、`preflight.json`）**

| # | 確認 | 結果 |
|---|---|---|
| 00 | Phase 9K の protocol と条件が一致（n, d, K, family, σ²_x, f_scale, w0, w, seed） | PASS |
| 00b | Phase 9K の `generator_provenance.csv` の列平均と、再生成した X の列平均が 20 × 12 列すべてでビット単位で一致 | PASS |
| 01–04 | Z_true / F_true / 生の X / full Y が両 arm で同一（1 回の決定的な生成を 4 fit で共有） | PASS |
| 05 | family の違いは列 3〜8 の B → P だけ | PASS |
| 06 | K = 3 のみ | PASS |
| 07–10 | test mask が同一、学習 mask が両 arm で同一、sparse ⊂ dense、学習と test が重ならない | PASS（20/20） |
| 11 | fit の数が 80 | PASS |
| 12 | 両 arm とも support に適合（`validate_xy`、不一致の許容なし） | PASS |

focused tests（`expfam/src/experimental/test_misspecification_m1.py`、EM なし）: 10/10 PASS。

---

## 5. Technical completeness

| 項目 | 値 |
|---|---|
| 計画 / 試行 / 成功 | 80 / 80 / **80** |
| retry / 置き換え / seed の救済 | 0 / 0 / 0（`fail_fast` で構造的に 0） |
| 失敗した fit / 非有限 | 0 / 0 |
| warning | 0 |
| K の値 | {3} |
| family の自動選択 | なし |
| 実行時間 | 1 fit あたり中央値 9.5〜11.2 秒、全体 870.7 秒 |

結果を見た後に、条件・seed・指標を変えたり追加したりしていない。

---

## 6. Dense correct vs wrong

| 量 | correct | wrong |
|---|---|---|
| RMSE_Z 中央値（IQR） | 0.476（0.459–0.493） | 0.497（0.474–0.515） |
| RMSE_Z 平均（min–max） | 0.479（0.423–0.568） | 0.499（0.409–0.583） |

**対比較 `delta = wrong − correct`（20 replicate）**: 平均 **+0.020**、中央値 +0.025、IQR 0.011〜0.034（幅 0.023）、範囲 −0.024〜+0.065、**wrong の方が悪かったのは 15/20**（良かったのは 5/20）。

**OBSERVED**: Y が十分にある条件でも、誤指定はほとんどの replicate で Z の回復をわずかに悪くした。
ただし大きさは小さく、correct の RMSE の約 4% であり、correct 自身の replicate 間のばらつき（IQR 0.033）と同じ程度である。

---

## 7. Sparse correct vs wrong

| 量 | correct | wrong |
|---|---|---|
| RMSE_Z 中央値（IQR） | 0.915（0.848–0.968） | 0.983（0.935–1.026） |
| RMSE_Z 平均（min–max） | 0.917（0.747–1.104） | 0.983（0.847–1.114） |

**対比較**: 平均 **+0.066**、中央値 +0.060、IQR 0.028〜0.101（幅 0.074）、範囲 −0.018〜+0.192、**wrong の方が悪かったのは 17/20**（良かったのは 3/20）。

**OBSERVED**: Y の学習ペアが 250 しかないと、誤指定による悪化は dense の約 3 倍（平均 +0.066 vs +0.020）になり、悪化した replicate も増えた（17/20）。

**重要な文脈**: sparse では、correct でも Z の回復はかなり弱い。
- RMSE_Z の中央値は 0.92 である。真の Z と無関係な N(0, I) の Z の RMSE は、n = 75、K = 3 で中央値 1.29（#131 synthesis の参照値、2000 回のシミュレーション）。
- Gram 相関の中央値は 0.34（dense は 0.79）。

つまり sparse は「Z が弱くしか戻らない条件」であり、その中での +0.066 である。相対的には correct の RMSE の約 7% にあたる。

---

## 8. Interaction pattern

`interaction_delta = delta_sparse − delta_dense`（replicate ごと）: 平均 **+0.046**、中央値 +0.041、IQR 0.003〜0.090、範囲 −0.053〜+0.152、**正は 15/20**（負は 5/20）。

**INTERPRETATION**: 多くの replicate で、誤指定の影響は sparse で大きかった。これは #130 の「Y が少ないと X の観測モデルの正しさがより効く」という仮説と**整合する**。
ただし 5/20 では逆向きで、replicate ごとのばらつきは大きい。一般法則や因果的な交互作用とは呼ばない。

---

## 9. Held-out Y

同じ 278 の test ペアで評価した（`heldout_y.csv`）。下の差は正なら correct が良いことを意味する。

| regime | mean log score（correct / wrong の中央値） | 差 correct − wrong（平均、中央値） | AUC（correct / wrong の中央値） | AUC の差（平均） | AP の差（平均） |
|---|---|---|---|---|---|
| dense | −0.574 / −0.565 | +0.004、+0.001 | 0.742 / 0.740 | +0.004 | +0.002 |
| sparse | −0.654 / −0.655 | +0.012、+0.014 | 0.597 / 0.549 | **+0.036** | +0.018 |

**OBSERVED**:
- dense では、Y の予測に差はほぼない（log score、AUC、AP の差はいずれも ≤ 0.004）。
- sparse では、AUC で correct が平均 +0.036 良かった。ただし IQR は −0.016〜+0.076 で、replicate ごとの向きは揃っていない。
- Y の予測の差は、Z の回復の差より不安定だった。
- sparse では両 arm とも AUC が 0.55〜0.60 で、Y の予測自体が弱い。

---

## 10. Parameter recovery

真の値は w0 = −1、w = 1（`parameter_errors.csv`）。

| regime | arm | ŵ0 の中央値 | ŵ の中央値 | `|ŵ0 + 1|` の平均 | `|ŵ − 1|` の平均 |
|---|---|---|---|---|---|
| dense | correct | −0.985 | 0.966 | 0.030 | 0.090 |
| dense | wrong | −1.008 | 0.962 | 0.034 | 0.097 |
| sparse | correct | −0.807 | 0.609 | 0.191 | 0.441 |
| sparse | wrong | −0.974 | 0.552 | **0.137** | 0.459 |

**OBSERVED**:
- dense では、w0 と w の推定に arm の差はほぼない。
- sparse では、両 arm とも ŵ が大きく 0 に縮んだ（中央値 0.55〜0.61）。Z の回復が弱いことと整合する。
- **予想外の観察**: sparse の w0 の誤差は wrong の方が小さかった（対の差の平均 −0.054、中央値 −0.077）。
  correct の ŵ0 は −0.81 と上に偏っている。理由は分からず、本実験では調べていない。そのまま記録する。

---

## 11. Relation to #130 existing evidence

| #130 の既存の証拠 | 系列 | M1 との関係 |
|---|---|---|
| fixed Exp4 Scen C: 真の Bernoulli X → Poisson、Y はほぼノイズなしの Gaussian（σ_y = 0.1）。差は 1.00×（15/30） | C（fixed） | Y が非常に強いと差は 0。M1 の dense（Bernoulli-Y）では、小さいが一貫した悪化（+0.020、15/20）が出た。**Y が完全に Z を決めない条件では、support-compatible な誤指定にも小さなコストがある** |
| per-column の all-Gaussian（B 型: representation）: dense +0.011〜+0.022、観測率 0.1 で +0.12〜+0.43 | F / E-consistent | M1 も「dense で小さく、sparse で大きい」という同じ向きだった。**Poisson 化（A 型）の悪化は、Gaussian 化（B 型）より小さい**（sparse で +0.066 vs +0.12〜+0.43）。ただし generator・列構成・系列が違うので、数値を直接は比べない |
| NB-Y を Poisson で fit すると、Z は良く、Y は悪い | fixed | M1 では Z と Y が同じ向き（sparse で両方 correct が良い）だった。誤指定の向きは、誤指定の種類によって違う |

**まとめ**: M1 は #130 の「X 側の誤指定の影響は Y の情報量に依存する」という整理と整合した。
#130 で残っていた「support-compatible な比較が、Y が強すぎて差が出ない条件にしかない」という空白を、この固定条件について埋めた。

---

## 12. Thesis interpretation

結果は #134 の Case A に近い。

> この固定条件（n = 75、d = 12、K_true = 3、mixed-X、Bernoulli-Y）では、0/1 の属性を Poisson として誤指定すると、Z の回復は dense-Y でわずかに（平均 +0.020、15/20）、sparse-Y でより大きく（平均 +0.066、17/20）悪化した。
> Y を十分に学習できるときは誤指定の影響が小さく、Y が少ないときは正しい観測分布の重要性が増した。

修論のストーリーへの意味:

- 「正しい分布なら常に大きく良くなる」とは言えない。dense ではコストは小さい（RMSE の約 4%）。
- 一方で、**support の中に収まる「穏やかな」誤指定でも、Y の情報が少ないときは Z の回復が一貫して悪くなる**。
  これは「関係データが少ないときこそ属性を適切な指数型分布族で扱う意味がある」という #130 の位置づけを、最も素直な比較で支える。
- 誤指定の種類によって大きさが違い、この条件では Poisson 化（A 型）は Gaussian 化（B 型、#130 の既存の証拠）より穏やかに見える。ただし同じ実験で比べたわけではない。

---

## 13. Limitations

- **1 つの固定条件だけ**である（n = 75、d = 12、K_true = 3、G3/B6/P3、Bernoulli-Y、w0 = −1、w = 1、20 replicate）。一般の法則とは呼ばない。
- **sparse は Z の回復そのものが弱い条件**である（Gram 相関 0.34、RMSE 0.92 vs ランダム 1.29）。この低情報の領域での差であり、「Z をよく推定できる条件で sparse にした」わけではない。
- Y の観測量は **観測ペアの mask** で変えた。edge 密度（w0）や関係の強さ（w）を変えた場合は未検討。
- Bernoulli → Poisson の違いは、平均関数・曲率・列の重みの違いを含む観測モデル全体の違いであり、「分布の正しさだけ」の効果とは分離していない。
- 主な結果は記述的な対比較であり、有意差の検定はしていない。replicate ごとの向きは揃っていない（dense 5/20、sparse 3/20、interaction 5/20 が逆向き）。
- 系列は E-consistent の prototype で、Phase 9K の K 選択の pipeline とは fit の経路が違う。本文の主張には Human の判断が必要。
- sparse で wrong の w0 の誤差が小さかった理由は調べていない。
- X の再構成・X の予測は評価していない（設計どおり）。

---

## 14. Next research action

- 本実験の範囲では、**追加の実験を自動的に始めない**。
- Human の判断の候補（どれも新しい Human Gate が必要）:
  1. この結果で先生に説明する（§12 の Case A 型の解釈と §13 の限界）。**最小のコストの選択肢**であり、#130 / #131 の資料と合わせれば説明は成り立つ。
  2. 誤指定の種類による違いを同じ実験の中で見たい場合は、M2（Bernoulli → Gaussian）を**同じ data・mask・seed で**追加する。
     #130 の既存の証拠では A 型と B 型の比較が系列をまたいでいるので、同じ設計で比べるとこの点を確定できる。
  3. sparse での w0 の偏り（correct の ŵ0 = −0.81）を記述的に調べる。研究の主張には直接は影響しない。
- 私の推奨は 1 である。2 は、修論で「誤指定の種類によって重みが違う」と主張したくなった場合にだけ価値がある。

---

## DECISION

## **A: MISSPECIFICATION_EFFECT_CHARACTERIZED**

80/80 の fit が凍結した設計どおりに完了した（retry・置き換え・失敗・warning は 0）。
preflight の全項目（データの同一性、mask の契約、K = 3、family の差が B6 → P6 だけであること）が PASS だった。
すべての replicate で dense / sparse の対比較と interaction が計算でき、研究質問に答えられる。この判定は correct が勝ったかどうかによらない。
