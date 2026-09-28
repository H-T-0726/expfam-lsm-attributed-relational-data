# Phase 9P — relational 係数 w に対する C_Lap / C_Q の感度（2026-09-28）

- 位置づけ: **CONDITION-SPECIFIC W-SENSITIVITY CHARACTERIZATION**（Issue #104）。頑健性の証明でも、基準の優劣の検定でもない。
- 条件: Phase 9K と同じ（n=75, d=12, K_true=3, X = Gaussian×3 / Bernoulli×6 / Poisson×3, Y = Bernoulli, w0=−1, f_scale=√2, L=5,
  探索・refit とも 8 反復, K=1..5, start_B のみ, N=75, IS 補正なし）で、**w だけ**を 1/√2（weak）と √2（strong）に変えた。
  w=1（baseline）は Phase 9K の記録をそのまま読んだ（**baseline の EM は 0 回追加**）。
- 結果: `expfam/results/relational_w_sensitivity/phase9p_20260928/`
  （`preflight/`＝Z/F/X 同一性と Y の文脈、`weak_w/`・`strong_w/`＝Phase 9K と同じ artifact 一式と `fitted_states.json`、`combined/`＝3 条件の比較）
- 実行コード: `2a99133`（`expfam/src/experimental/run_w_sensitivity.py`）。weak_w と strong_w を各 1 回、順に実行（いずれも git clean、run SUCCESS）。
- 系列 E（experimental prototype; 本文採用不可）
- **数え方**: 同じ 20 replicate の系列（同じ seed・Z・F・X）を 3 つの w で見たものであり、**60 個の独立な dataset ではない**。

---

## 1. 研究上の問い

w だけを弱めた・強めたとき、C_Lap と C_Q の K 選択、K=2 vs 3 の差、最良 vs 次点の差、技術的な挙動、K=2→3 の分解がどう変わるか。

## 2. 事前の確認

- **パイプラインの比較可能性: PASS**。Phase 9K の実行コード `bbf45e4` から現在の main までの差分は、Phase 9M/9N/9O の新しい診断ファイル
  （`theta_stationarity_diagnostic.py`, `multistep_local_pilot.py`, `derivative_failure_classification.py` とそのテスト）・報告書・結果の**追加だけ**で、
  既存ファイル（mixed generator、family selection、joint runner、objective-consistent model、strict Q、`laplace_k_criterion.py`、`run_laplace_pilot.py`、`run_lap_vs_cq_20.py`）の変更は 0 件だった。
- **Z/F/X の同一性: 20/20**。推論なしで 3 条件の dataset を再生成し、Z・F・X が weak/baseline/strong の間でビット単位で一致した（Y は 20/20 で異なる）。
  generator は Z → F → X → Y の順に同じ乱数列から引くため。Y の Bernoulli 乱数の結合が w 間で同一だとは主張しない。
- **baseline の再利用の確認**: Phase 9K の `laplace_by_k.csv` と `family_by_k.csv` から同じ集計関数で計算し直した P1〜P5 は、committed の `paired_summary.json` と完全に一致した。
  K=2→3 の分解も Phase 9K 報告書 P6 の値（fit gain 150.8/347.8/447.1、体積の増分 160.0/231.5/255.3）を再現した。
- `preflight/preflight.json` の `git_dirty: true` は、確認の時点で preflight 自身の出力ディレクトリが未追跡だったためである（コードは `2a99133` で committed）。

## 3. Y の文脈（真の生成値から、記述のみ）

| 条件 | w | w²·K_true | edge 密度（median, 範囲） | 真の η_Y の SD（median） | 真の確率の 5% / 95% 分位（median） |
|---|---|---|---|---|---|
| weak | 0.707 | 1.5 | 0.306（0.287–0.324） | 1.16 | 0.052 / 0.714 |
| baseline | 1 | 3.0 | 0.333（0.313–0.341） | 1.63 | 0.024 / 0.846 |
| strong | 1.414 | 6.0 | 0.352（0.331–0.373） | 2.31 | 0.008 / 0.944 |

w0 = −1 を固定しているため、w を変えると η_Y のばらつきだけでなく、**edge 密度と確率の飽和**も変わる（η_Y の平均はどの条件も ≈ −1.0）。
したがって以下は「w-sensitivity」であり、密度を揃えた純粋な関係信号の効果ではない。

## 4. 結果（条件ごと）

### 技術的な完全性

| 条件 | EM 試行 / 成功 | 不完全な dataset | Candidate B OK / NOT_STATIONARY / HESSIAN_NOT_PD / error | C_Lap 完全な dataset |
|---|---|---|---|---|
| weak | 200 / 200 | 0 | 100 / 0 / 0 / 0 | 20 |
| baseline（Phase 9K） | 200 / 200 | 0 | 100 / 0 / 0 / 0 | 20 |
| strong | 200 / 200 | 0 | 100 / 0 / 0 / 0 | 20 |

新しい EM: 計画 400、試行 400、成功 400。retry・置き換え・seed の救済はなし。

### 選ばれた K

| 条件 | K_hat_Lap（K1/K2/K3/K4/K5） | exact / under / over | K_hat_Q（K1/K2/K3/K4/K5） | exact / under / over |
|---|---|---|---|---|
| weak | 0 / 4 / 16 / 0 / 0 | 16 / 4 / 0 | 9 / 10 / 1 / 0 / 0 | 1 / 19 / 0 |
| baseline | 0 / 1 / 19 / 0 / 0 | 19 / 1 / 0 | 1 / 5 / 14 / 0 / 0 | 14 / 6 / 0 |
| strong | 0 / 0 / 20 / 0 / 0 | 20 / 0 / 0 | 0 / 1 / 19 / 0 / 0 | 19 / 1 / 0 |

どの条件でも過大選択（K ≥ 4）は 0 だった。

### 同じ条件内の C_Q と C_Lap の対応（20 dataset）

| 条件 | 両方 K=3 | C_Lap だけ K=3 | C_Q だけ K=3 | どちらも K≠3 | 同じ K | 異なる K |
|---|---|---|---|---|---|---|
| weak | 1 | 15 | 0 | 4 | 2 | 18 |
| baseline | 14 | 5 | 0 | 1 | 15 | 5 |
| strong | 19 | 1 | 0 | 0 | 19 | 1 |

### 差（min / median / max; 負の数 / 正の数）

| 条件 | delta_Lap_23 = C_Lap(3) − C_Lap(2) | delta_Q_23 = C_Q(3) − C_Q(2) | C_Lap の最良 vs 次点の差 | C_Q の最良 vs 次点の差 |
|---|---|---|---|---|
| weak | −80.2 / −22.0 / 44.8（16 / 4） | −17.3 / 63.3 / 171.3（2 / 18） | 1.54 / 24.0 / 52.9 | 1.33 / 34.3 / 102.1 |
| baseline | −150.3 / −75.9 / 52.3（19 / 1） | −130.2 / −3.7 / 157.1（14 / 6） | 22.0 / 49.2 / 58.4 | 1.96 / 34.8 / 130.2 |
| strong | −254.3 / −176.5 / −34.3（20 / 0） | −257.7 / −147.5 / 24.6（19 / 1） | 34.3 / 56.0 / 71.7 | 24.6 / 136.3 / 183.4 |

C_Lap の最小の 5 つの差: weak は rep10 **1.54**（最良 K=2、次点 K=3）、rep11 4.54、rep07 5.26、rep04 7.02、rep09 10.70。
strong は rep20 34.3、rep02 44.3、rep04 49.8、rep19 52.5、rep09 52.8。

## 5. w をまたいだ対応（replicate ごと）

### 選ばれた K の経路（weak → baseline → strong）

- **C_Lap**: 3 条件で同じ K 16/20（すべて 3→3→3）。変わったのは 4 件: rep04・rep07・rep10 が 2→3→3、rep20 が 2→2→3。
  weak→baseline は same 17 / under→exact 3、baseline→strong は same 19 / under→exact 1。
- **C_Q**: 3 条件で同じ K 1/20（rep15、3→3→3）。経路は 2→3→3 が 7、1→3→3 が 6、2→2→3 が 3、1→2→3・1→1→3・1→2→2 が各 1。
  weak→baseline は same 5 / under→exact 13 / under→under（K の値は変化）2、baseline→strong は same 15 / under→exact 5。
- どちらの基準でも、exact→under、exact→over、over→exact の変化は 0 件だった（3 条件を通して over は 0）。

### 差の変化（baseline からの変化、min / median / max）

| 量 | baseline → weak | baseline → strong |
|---|---|---|
| delta_Lap_23 | −7.6 / **+49.6** / +93.0（符号の変化 3 件） | −139.2 / **−93.9** / −45.1（符号の変化 1 件） |
| delta_Q_23 | +0.2 / **+90.4** / +142.7（符号の変化 12 件） | −178.2 / **−124.2** / −68.5（符号の変化 5 件） |
| C_Lap の最良 vs 次点の差 | −43.9 / −22.9 / +4.1 | −18.0 / +9.9 / +33.2 |
| C_Q の最良 vs 次点の差 | −117.8 / +14.3 / +66.6 | −41.7 / +100.2 / +178.1 |

weak では delta_23 が両基準とも K=2 寄りに動き（20/20 で C_Q、18/20 で C_Lap が正の方向）、strong では両基準とも 20/20 で K=3 寄りに動いた。
移動量の中央値は C_Q の方が大きかった（weak +90.4 vs +49.6、strong −124.2 vs −93.9）。

## 6. K=2 → 3 の分解（min / median / max）

| 量 | weak | baseline | strong |
|---|---|---|---|
| C_Lap: joint mode での fit gain `D_mode(2) − D_mode(3)` | 145.3 / 264.1 / 343.6 | 150.8 / 347.8 / 447.1 | 307.3 / 481.6 / 582.3 |
| C_Lap: Laplace 体積の増分 `Δ(‖Ẑ‖² + log\|H\|)` | 146.9 / 199.3 / 229.2 | 160.0 / 231.5 / 255.3 | 229.8 / 264.1 / 287.3 |
| C_Q: MC 平均での fit gain `D_K(2) − D_K(3)` | 84.7 / 192.8 / 273.3 | 98.9 / 259.7 / 386.3 | 231.4 / 403.5 / 513.7 |
| C_Q: P_Z の増分 | 212.84（一定） | 212.84（一定） | 212.84（一定） |
| 共通: 罰則 P_θ の増分 | 43.17（一定） | 43.17（一定） | 43.17（一定） |
| fit gain の差（C_Lap − C_Q） | 43.6 / 73.2 / 106.8 | 49.1 / 77.3 / 101.9 | 59.1 / 79.9 / 108.2 |

- 記述: どちらの fit gain も w とともに大きくなった。C_Q の Z に関する増分（P_Z）は構成上 212.84 で一定だが、
  C_Lap の Laplace 体積の増分は w とともに増えた（median 199 → 232 → 264）。
- fit gain の差（C_Lap − C_Q）は 3 条件でほぼ同じ範囲だった（median 73–80）。
- C_Q の fit gain が「P_Z の増分 + P_θ の増分」（256.0）を下回った dataset（= delta_Q_23 > 0）は weak 18 / baseline 6 / strong 1、
  C_Lap の fit gain が「体積の増分 + P_θ の増分」を下回った dataset（= delta_Lap_23 > 0）は weak 4 / baseline 1 / strong 0 だった。
  weak では fit gain が両基準とも小さくなったが、C_Lap では体積の増分も小さくなったため、K=3 がより多く残った。
  これは観察された分解の記述であり、一般的な因果の仕組みとしては主張しない。

## 7. 技術的な挙動（Candidate B、条件 × K）

- 300 refit（新しい 200 + baseline 100）すべてで status OK。joint mode の反復は median 4–9（最大 58、strong の K=5 の 1 件）、
  最後の grad_inf は最大 9.8e-9（許容値 1e-8 以下）、H の最小固有値は K とともに小さくなり、最小値は K=5 で 0.10（weak）/ 0.057（baseline）/ 0.12（strong）で、すべて正だった。
- logdet_H は K と w とともに大きくなった（K=3 の median: 441 / 541 / 622）。
- family selection の candidate convergence warning（診断のみ）は K とともに増え（K=1 の median 4–5 から K=5 の median 16–18）、w による大きな違いはなかった。

## 8. family の挙動（二次的な観察）

3 条件 × 5 K × 20 dataset の**すべての refit で、選ばれた family 割当が真の割当（G3/B6/P3）と一致した**。
この条件では、w の変化によって選ばれる family は変わらなかった。family selection の優位性は主張しない。

## 9. DECISION

## **DECISION: RELATIONAL_W_SENSITIVITY_CHARACTERIZED**

両方の新しい条件が事前固定の手順どおり 1 回ずつ完了し（EM 400/400 成功、不完全な dataset 0、Candidate B 200/200 OK）、
Z/F/X の同一性と baseline の再利用を確認したうえで、事前に決めた結果をすべて報告できた。この判定は K=3 の頻度によらない。

## 10. 解釈（INTERPRETATION）

- この条件では、w を弱めると両基準とも K=2 以下を選ぶことが増え、強めると両基準とも K=3 に集まった。
  C_Q の方が w に敏感で（K=3 は 1 → 14 → 19）、C_Lap は weak でも 16/20 で K=3 を選んだ（16 → 19 → 20）。
- 両基準の違いは、主に K=2→3 で Z に関する増分をどう数えるかに対応していた。C_Q の P_Z は w によらず一定だが、C_Lap の Laplace 体積の増分は w とともに変わった。
- ただし w の変化は edge 密度と確率の飽和も変えるため、これを純粋な信号の強さの効果とは言えない。

## 11. CAN SAY / CANNOT SAY

**CAN SAY**
- この n=75, d=12, K_true=3, mixed-X の固定条件で、同じ Z/F/X と seed のまま w を 1/√2, 1, √2 と変えたとき、
  C_Lap の K=3 は 16 → 19 → 20、C_Q の K=3 は 1 → 14 → 19 件で、過大選択は 0 だった。
- 同じ変化で、delta_23 は両基準とも weak で K=2 寄り、strong で K=3 寄りに動き、移動量の中央値は C_Q の方が大きかった。
- K=2→3 で、C_Lap の Laplace 体積の増分は w とともに増え、C_Q の P_Z の増分は一定だった。
- Candidate B は 3 条件のすべての refit で OK で、選ばれた family 割当はすべて真の割当と一致した。

**CANNOT SAY**
- 一般的な頑健性、一致性、C_Lap または C_Q の一般的な優越性、一般的な recovery 確率。
- edge 密度・飽和を揃えた純粋な関係信号の効果。
- 他の n/d/K_true、X の信号、family 構成、実データへの一般化。
- 60 個の独立な dataset としての数え方。
- Phase 9K の結果を書き換えること（baseline は凍結済みの記録のまま）。

## 12. 最大の UNRESOLVED と次の Human 判断

**最大の UNRESOLVED**: weak では C_Lap の最良 vs 次点の差が小さい dataset がある（rep10 は 1.54 で最良 K=2、次に rep11 4.54、rep07 5.26）。
Phase 9M では、別の条件（baseline）で 1 回の局所ステップが C_Lap を最大 2.81 動かし、Phase 9N でも保存済み θ̂ は停留点ではなかった。
したがって weak の小さな差のいくつかは、refit の最適化誤差と同じ程度の大きさでありうる。これらについて局所最適化の診断は、本 Issue の禁止事項に従い行っていない。
また、この感度が w そのものによるのか、edge 密度・飽和の変化によるのかは、この設計では分けられない。

**次の Human 判断**: (a) weak の小さな差の dataset に Phase 9M 型の固定診断を行うか、
(b) w0 を調整して edge 密度を揃えた条件を別に設計するか、(c) ここで w-sensitivity の特徴づけを閉じるか。いずれも新しい Human Gate が必要。
