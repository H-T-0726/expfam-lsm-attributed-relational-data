# AF3 / S0：分布選択の誤りと、選択に使う潜在構造 Z の関係（selector の再スコアリング、MCEM なし）

- 実験 ID：AF3-S0（2026-11-13）
- branch：`experiment/af3-selector-rescoring`（AF2 = PR #167 の上に積んだ。PR #166・#167・#129 はいずれも未マージ）
- 実装系列：**lineage E（E-consistent per-column）＋ PR #129 の generic model（experimental prototype、本文採用不可）**。採用式どおり extra 1/2 なし
- 新しい MCEM fit：**0 回**。AF2 の 20 データを再生成し、AF2 の保存状態（npz）と真の Z を使って、既存の候補スコア（`score_candidate`）と既存の選択規則（`select_column`）を計算し直しただけ
- protocol：`protocol.json`（canonical sha256 `9a2e3dd9ef6c3c84dd5677b124beec681ef91fd7184af08d4aa076539c8bc18d`）。結果を見る前に commit `e55796d` で固定した
- 実行：`full/runinfo.json`（git `e55796d`、tracked-dirty = False、93 秒、警告 0）
- 数値はすべて `full/analysis/` の生成物（`results_tables.md`、`summary.json`、CSV）から取った

## 0. 結論（先に）

1. **[OBSERVED] 真の Z（Z1）では、真 CMP2 列 60/60 を正しく選んだ。** 真 Poisson 列は 58/60 で、残り 2 列は CMP2 に誤選択された（margin −0.49 と −6.70）。
2. **[OBSERVED・統制した操作] 真の Z に独立なノイズを加えると（Z2）、CMP2 → Poisson の誤選択が単調に増えた。**
   - 真 CMP2 列の正解率：s = 0.1 で 0.967、0.2 で 0.823、0.3 で 0.640、0.45 で 0.397、0.6 で 0.253。
   - replicate ごとの傾きは 20/20 で正（平均 1.47、95% CI [1.25, 1.70]）。
   - 誤りの 96.8% は CMP2 → Poisson の方向で、Poisson → CMP2 はほぼ増えなかった。
3. **[OBSERVED] ただし、「Z の誤差の大きさ（RMSE）」だけでは選択の成否は決まらなかった。これは単純な形の仮説に反する。**
   - Oracle の推定 Z サンプル（Z3）は、サンプルあたりの Procrustes RMSE が dense 0.370、sparse 0.558 と大きい。それでも真 CMP2 列を **dense・sparse とも 60/60** で正しく選んだ。
   - 同じ大きさの独立ノイズ（Z2m：dense 0.366、sparse 0.554）では、正解率は 0.533 と 0.310 にとどまった。
4. **[OBSERVED・同じデータでの対比] 決め手は「Z の推定にどの分布を使ったか」だった。**
   - Support-only（CMP2 列を Poisson として推定）の Z サンプル（Z5）では、真 CMP2 列の正解率が dense 0.783、sparse 0.533 に落ちた。
   - Z5 と Z3 の RMSE はほぼ同じである（0.386 と 0.370、0.585 と 0.558）。
   - Z5 − Z3（CMP2 → Poisson の率）は dense +0.217（CI [0.117, 0.333]、10/20 で正・負は 0）、sparse +0.467（CI [0.367, 0.567]、19/20 で正）。
   - Wrong（全列の分布を入れ替えて推定）の Z（Z6）では、両方向に誤った。sparse の真 Poisson 列は正解率 0.117 まで落ちた。**Z の推定に使った分布（incumbent）の方向へ、選択が引き寄せられた。**
5. **[OBSERVED] AF2 の誤選択は、Oracle の Z では全部消え、AF2 自身の refit の Z では再現した。**
   - AF2 Auto で誤選択された真 CMP2 列は 38 列（dense 11、sparse 27）。Z3 ではこの 38 列すべてが正しく選ばれた。
   - Auto の refit Z（Z7）では、33/38 列が誤選択のまま残った。AF2 で正しかった列が Z7 で誤った例は 0 だった。
6. **[OBSERVED] 候補最適化の誤差は原因ではない。**
   - |勾配| の最大は 3.6e-5、ヘッセ行列の最小固有値は ≥ 14.2。スコア誤差の上界 ‖g‖²/(2λ) は最大 4.3e-11 だった。
   - zero 開始と incumbent 開始の signed margin の差は最大 4.3e-13、選択の不一致は 0 だった。
7. **[OBSERVED] 誤選択の多くは僅差ではない。** Z3 と Z5 の誤選択 42 列のうち、|ΔQ| < 2 は 12% だけだった。
8. **[OBSERVED] Oracle の plug-in mode（Z4）では、逆向きの誤りが出た。** 真 CMP2 列は全問正解だったが、真 Poisson 列の正解率は dense 0.767、sparse 0.300 に落ちた（Poisson → CMP2）。

**まとめ（INTERPRETATION）**：
- 「Z が不正確だと分布選択が誤る」は、**独立な誤差については成り立った**。
- しかし MCEM の推定 Z で効いていたのは、誤差の大きさよりも、**その Z が「どの分布を仮定した事後分布」から作られたか**だった。
- Q スコアは、Z を作るのに使った分布を支持しやすい（自己確認的な偏り）。これは AF2 で誤選択が反復後半でも安定して固定されたことと整合する。

## 1. 研究問い

**RQ**：分布選択に使う Z の推定精度が悪くなると、Q スコアで正しい分布を選ぶ率も下がるか。

**仮説**（成立は前提にしない）：真の Z なら正しく選べるが、Z に誤差を加えると CMP2 → Poisson の誤選択が増える。

## 2. 数式

### 2.1 何を計算したか

データは AF2 と同じで、z_i ~ N(0, I_3)、x_ij ~ ExpFam_m(η_ij = f_jᵀ z_i) です。切片はなく、n = 75 です。

各 count 列 j と、与えられた Z サンプル Z^(1..L) について、各候補 m ∈ {Poisson, CMP(ν=2)} の Q スコアを次で計算しました。

```
Q_j(m) = max_f (1/L) Σ_l Σ_i log p_m(x_ij | fᵀ z_i^(l))
```

- log p_m は基底測度を含む完全な対数 pmf です（Poisson：−log x!、CMP2：−2 log x!）。
- loading f は候補ごとに trust-exact 法で最大化しました（AF2 と同じ経路）。
- 選択規則は既存の `select_column` です（同点なら Poisson）。
- signed margin を `ΔQ = 2(Q_j(true) − Q_j(wrong))` と定義し、ΔQ / n も保存しました。

### 2.2 Z の回転・尺度が選択に関係しない理由

切片がないので、任意の可逆行列 A について fᵀ z = (A⁻ᵀ f)ᵀ (A z) が成り立ちます。そのため Q_j(m) は、Z 全体にかける可逆な線形変換（回転・尺度）で変わりません（平行移動では変わります）。

そこで Z の誤差は、回転 Procrustes 後の RMSE と、線形写像で合わせた後の残差の両方で記録しました（`z_error_by_instance.csv`）。

ノイズ s は N(0, I) 上の等方的な誤差の標準偏差で、1 回の抽出では Procrustes RMSE ≈ s になります（`audit/noise_level_interpretation.csv`：s = 0.45 で 0.450、0.6 で 0.588）。

### 2.3 Q による選択が正しくなる条件（DERIVED、母集団レベル）

Z サンプルが**真のモデル（真の分布の割当て・真の θ）の下での厳密な事後分布** p(Z | X, Y) から抽出されているとします。このとき、(x_j, z) の同時分布は生成分布と一致します。したがって任意の候補 m と loading f について

```
E[log p_true(x | z)] − E[log p_m(x | fᵀz)] = E_z KL(p_true(·|z) ‖ p_m(·|z)) ≥ 0
```

が成り立ち、母集団レベルでは真の分布が勝ちます。

この保証は、Z が次のいずれかの場合には成り立ちません。

- **(a) 独立な誤差を含む Z**：x | (z + e) は η の誤差 fᵀe について混合された分布になり、真の条件付き分布より過分散になります。その結果、等分散の Poisson が過小分散の CMP2 より有利になります。偏りの大きさは ‖f‖²·Var(e) とともに増えます。
- **(b) 別の分布 m′ の下での事後分布から作った Z**：z は x_j を m′ の尤度で「説明する」ように引き寄せられるため、(x_j, z) は m′ のモデルらしく見えます。その結果 Q は m′ を支持しやすくなります（自己確認的な偏り）。
- **(c) 事後分布の広がりを持たない点推定（mode）**：z は x_j に過度に適合しているので、条件付き分布は見かけ上過小分散になり、CMP2 が有利になります。

(a)〜(c) は数式から導いた説明仮説です。厳密な有限標本の定理ではありません。今回の実験は (a)〜(c) を同じデータで操作または比較しています。

## 3. 設計（`protocol.json`）

| 項目 | 内容 |
|---|---|
| データ | AF2 の 20 データ（`af2.generate(r)`、seed 1661001+r）。count 列 6 列（真 Poisson 3、真 CMP2 3） |
| 腕（primary） | Z1 真の Z（L=1）／Z2 真の Z ＋ 独立ノイズ（サンプルごとに独立、L=5、s ∈ {0.1, 0.2, 0.3, 0.45, 0.6}、noise seed 5 本）／Z3 Oracle refit の Zsamples／Z5 Support-only refit の Zsamples（Z3・Z5 は dense と sparse） |
| 腕（secondary） | Z4 Oracle の joint mode（plug-in、L=1）／Z6 Wrong refit の Zsamples／Z7 Auto（Poisson 開始）refit の Zsamples／Z2c 共通ノイズ 1 回（L=1）／Z2m 大きさを Oracle に合わせたノイズ（共通 a ＋ サンプルごと b：dense (0.27, 0.25)、sparse (0.41, 0.38)）／Z3・Z5・Z6・Z7 の incumbent 開始 |
| s の範囲 | 提案の {0.1, 0.2, 0.3, 0.45} に 0.6 を追加した。理由は、事前の較正で Oracle サンプルの RMSE が sparse で 0.56 だったため（Z の幾何だけを使い、選択結果は見ていない） |
| 候補・最適化 | {Poisson, CMP(ν=2)}、trust-exact、主解析は zero 開始 |
| 統計単位 | replicate（20）。列と noise seed は replicate の中で平均してから集計した |
| 区間 | replicate 単位の percentile bootstrap（10000 回、seed 20261113） |
| 失敗 | 候補が失敗した列は FAILURE とし、正解に数えず別に報告する（今回は 0 件） |
| 予算 | 79 instance × 20 replicate × 6 列 × 2 候補 = 18,960 回の候補最適化。MCEM は 0 回 |

## 4. 実行前の監査（`audit/`）

| 確認 | 結果 |
|---|---|
| 再生成した X / Y / 真の Z の hash が AF2 の `fit_runs.csv` と一致 | 200/200 |
| npz の sha256 が AF2 の `state_sha256` と一致 | 200/200（実験後にも再確認して 200/200） |
| 形状 | Zsamples 75×3×5、Zmode 75×3、F 9×3（全件） |
| 保存された θ̂（w0, w）が `fit_runs.csv` と一致 | 全件 |
| count 列 6 列は、すべて gate で {Poisson, CMP2} の score 判定 | 全件 |
| 既存 selector の再現：PR #129 実験 A の 2 データを生成器で作り直し、`selector_only` で計算したスコアを、PR #129 が公開した `candidate_scores.csv` と比べた | 102 スコアで差の最大 0.0（bit 一致） |
| AF2 の既存成果物 | `git diff b0a2398` で変更なし |

## 5. 結果（`full/analysis/results_tables.md` から）

### 5.1 primary

| 腕 | regime | 真 CMP2 の正解率 [95% CI] | 真 Poisson の正解率 | CMP2 の ΔQ 中央値 | サンプルあたり RMSE(Z) 中央値 |
|---|---|---|---|---:|---:|
| Z1 真の Z | — | 1.000 [1.000, 1.000] | 0.967 | 13.89 | 0 |
| Z2 s=0.1 | — | 0.967 [0.923, 1.000] | 0.973 | 11.96 | 0.099 |
| Z2 s=0.2 | — | 0.823 [0.733, 0.907] | 0.987 | 8.34 | 0.198 |
| Z2 s=0.3 | — | 0.640 [0.547, 0.733] | 0.983 | 4.59 | 0.298 |
| Z2 s=0.45 | — | 0.397 [0.290, 0.510] | 0.997 | −6.62 | 0.446 |
| Z2 s=0.6 | — | 0.253 [0.167, 0.343] | 0.997 | −17.41 | 0.596 |
| Z3 Oracle サンプル | dense | 1.000 [1.000, 1.000] | 0.983 | 13.71 | 0.370 |
| Z3 Oracle サンプル | sparse | 1.000 [1.000, 1.000] | 1.000 | 13.89 | 0.558 |
| Z5 Support-only サンプル | dense | 0.783 [0.667, 0.883] | 1.000 | 6.56 | 0.386 |
| Z5 Support-only サンプル | sparse | 0.533 [0.433, 0.633] | 1.000 | 1.21 | 0.585 |

- **用量反応（Z2）**：CMP2 → Poisson の率を s に回帰した replicate ごとの傾きは、平均 1.47（CI [1.25, 1.70]）で、20/20 が正。Poisson → CMP2 の傾きは平均 −0.045。
- **Z5 − Z3**（CMP2 → Poisson の率）：dense +0.217（CI [0.117, 0.333]、10 正・10 同値・0 負）、sparse +0.467（CI [0.367, 0.567]、19 正・1 同値・0 負）。

### 5.2 secondary

| 腕 | regime | 真 CMP2 の正解率 | 真 Poisson の正解率 | RMSE(Z) |
|---|---|---|---|---:|
| Z4 Oracle mode（plug-in） | dense | 1.000 | 0.767 | 0.275 |
| Z4 Oracle mode（plug-in） | sparse | 1.000 | 0.300 | 0.411 |
| Z6 Wrong サンプル | dense | 0.733 | 0.717 | 0.390 |
| Z6 Wrong サンプル | sparse | 0.450 | 0.117 | 0.601 |
| Z7 Auto refit サンプル | dense | 0.867 | 0.983 | 0.374 |
| Z7 Auto refit サンプル | sparse | 0.583 | 0.983 | 0.568 |
| Z2c 共通ノイズ s=0.3 / 0.6 | — | 0.653 / 0.313 | 0.987 / 0.987 | 0.297 / 0.602 |
| Z2m 大きさを合わせたノイズ | dense | 0.533 | 0.990 | 0.366 |
| Z2m 大きさを合わせたノイズ | sparse | 0.310 | 0.997 | 0.554 |

- **incumbent 開始**（Z3・Z5・Z6・Z7）：zero 開始と同じ選択だった（差は最大 4.3e-13、事後表）。
- **AF2 との列単位の対応**（`af2_column_linkage.csv`）：

  | AF2 で誤選択された真 CMP2 列 | Z3 で正解 | Z7 で誤りのまま | Z5 で誤りのまま |
  |---|---|---|---|
  | dense（11 列） | 11/11 | 8/11 | 8/11 |
  | sparse（27 列） | 27/27 | 25/27 | 23/27 |

  AF2 で正しかった列のうち、Z5 で新たに誤ったものが dense 5、sparse 5 列あった。
- **真の loading ノルム**（記述）：ノイズ系の腕・Z5・Z6 では、誤選択された真 CMP2 列の ‖f‖ 中央値（2.1〜2.7）が正解列（1.3〜1.9）より大きかった。Z2 の s=0.6 では、‖f‖ と ΔQ の Spearman（列をプールした記述値）が −0.91 だった。
- **誤選択の性質**（事後表）：
  - Z5 の誤選択の |ΔQ| 中央値は dense 5.2、sparse 9.0。
  - Vuong 型 z（記述）の中央値は −0.7 と −1.3。
  - 1 列の中での証拠は「中程度」の強さだが、向きは系統的にそろっている。

図：
- `full/analysis/fig1_cmp_accuracy_vs_z_error.png`（正解率と Z の誤差。Z3 は誤差が大きくても正解率 1 の位置にある）
- `full/analysis/fig2_signed_margin_by_arm.png`
- `full/analysis/fig3_margin_vs_loading_norm.png`

### 5.3 事前に宣言した読み方（G1–G5）

- **G1**（selector の問題）：非該当。Z1 の正解率は 1.0。
- **G2**（Z の誤差が CMP2 → Poisson を生む）：**該当**。ただし独立ノイズというモデルについてのみ。
- **G3**（Oracle の Z でも誤選択する）：非該当。Z3 の誤選択率は 0 と 0。
- **G4**（incumbent に依存した q）：**該当**。Z3 は 0、Z5 − Z3 は 0.217 と 0.467。
- **G5**（僅差が中心）：非該当。|ΔQ| < 2 は 12%。

**事前に書いた G4 の「次の一手」（burn-in / outer cycle）は、Z7 の結果と合わない。**
- 選択した割当てで refit した Z（Z7）で選び直しても、AF2 の誤りの 33/38 が再現した。
- Poisson で burn-in する q は Support-only の q（Z5）にあたり、そこでも誤る。
- したがって「選んだ分布で Z を作り直して選び直す」だけでは、自己確認的な偏りから抜けられない可能性が高い（§7）。

## 6. 研究問いへの答え

- **A. 真の Z でどれだけ選べるか**：真 CMP2 列は 60/60。真 Poisson 列は 58/60（2 列は有限標本での誤り。1 列は僅差、1 列は margin −6.7）。
- **B. ノイズを増やすと Poisson への誤選択は増えるか**：増えた（20/20 replicate で単調増加の傾き）。共通ノイズ（Z2c）でもほぼ同じだった。**統制した操作による結果なので、この種の誤差については因果的に確認できた。**
- **C. Oracle の推定 Z でも誤選択するか**：真 CMP2 列ではしなかった（dense・sparse とも 60/60）。真 Poisson 列の誤りは dense の 1 列だけだった。
- **D. Support-only の推定 Z では悪化するか**：悪化した（dense +0.217、sparse +0.467）。Z3 と Z5 は、同じデータ・同じ seed・同じ推定手順で、CMP2 列の分布の仮定だけが違う。その操作の効果として確認できた。ただし MCEM の確率的な経路の違いを含んでおり、これは分離していない。
- **E. Z の誤差だけで AF2 の誤選択をどこまで説明できるか**：
  - **誤差の大きさだけでは説明できない。** RMSE が同程度の Oracle の Z では誤りが 0 だった。
  - **AF2 の誤りは、誤った分布で作られた Z（Z5・Z7）の上で再現し、正しい分布で作られた Z（Z3）の上で消えた。**
  - AF2 の探索の初期（反復 1、ほぼ初期値の Z）で Poisson が強く支持されたことは、Z2 型の「独立誤差による偏り」で説明できる可能性がある。ただし探索中の Z は保存されておらず、**未検証**である。
  - 「初期の粗い Z で Poisson に寄り（Z2 型）、その後は Poisson で作った q が Poisson を支持し続ける（Z5 型）」という 2 段階の説明は、**未検証の説明仮説**である。
- **F. 別の説明は必要か**：
  - 候補最適化の誤差：否定された（上界 ≤ 4.3e-11、開始値による差なし）。
  - 僅差・MC ノイズ：主因ではない（|ΔQ| < 2 は 12%、ただし MC SE は推定していない）。
  - 反復不足：直接は未検証。ただし Oracle の refit は 8 反復で正しい選択を支える Z を作れており、Support-only も同じ 8 反復で誤る。そのため、反復数より「どの分布で作ったか」が効いている。
  - ν の推定、切片、別の条件：範囲外。

## 7. S1 の候補（実装しない。Human Gate）

| 優先 | 候補 | 根拠 | MCEM |
|---|---|---|---|
| 1 | **q に依存しない基準での割当ての比較**：Oracle / Support-only / Wrong の保存状態に、観測データの周辺尤度の近似（Laplace の joint mode、基底測度を正しく含める）または held-out 予測を当て、正しい割当てを選べるかを見る | 自己確認的な偏りは「同じ q の上での Q 比較」の性質なので、q が違うものどうしを比べられる基準が必要。保存状態だけで最初の判定ができる。AF2 の `mode_phi` は余分な定数を含むので、そのままは使えない | 0 |
| 2 | **候補ごとに Z を再推定して比較**（C 案）：曖昧な列ごとに、その列を各候補にした fit を作り、held-out X（列の一部の観測を隠す）の予測対数尤度で比べる | Z3 と Z5 の対比が示すとおり、どの q も自分の分布を支持するので、in-sample の Q では比べられない | 必要（列 × 候補ぶん） |
| 3 | **予測密度 log E_q[p_m(x_j \| z)] による採点**（E_q[log p] の代わり）。leave-column-out の q と組み合わせる | Jensen の差が分散の大きい候補を有利にする偏り（Z2 型）を避ける理論的な候補。推定の設計が必要 | 小 |
| — | burn-in / outer cycle | Z5・Z7 の結果から、単独では効かない見込み | — |
| — | 安定化付き選択（B） | 僅差は 12% だけで、主因ではない | — |
| — | Soft Selection（D） | incumbent の q で重みを作る限り、同じ偏りを受け継ぐ | — |

## 8. 限界・主張できないこと

- 合成条件は 1 つ（n=75、K=3、τ=2、ν=2 固定、切片なし）で、20 データ。npz 系の腕は MCEM 1 seed 系列の状態を使っている。
- Z2 / Z2m のノイズは等方的・独立なガウスで、推定誤差の構造（縮小、x との依存）を再現していない。「誤差の大きさが同じなら同じ結果」とは言えない、ということ自体が今回の結果である。
- Z3 / Z5 / Z7 は refit の最終反復の Z であり、AF2 の探索中の Z ではない。探索の動きは直接は検証していない。
- Vuong 型 z とサンプルごとの margin は、与えられた Z の下での記述値である。Z の推定に x_j 自身が使われているので独立性の条件を満たさず、検定として読まない。L=5 の連鎖サンプルのばらつきは MC 標準誤差ではない。
- §2.3 の (a)〜(c) は説明仮説である（(a) と (b) は操作・対比で支持されたが、一般的な定理ではない）。
- 「Q 選択は一般に失敗する」「実データでも同じ」「新しい選択法が有効」とは書かない。lineage E と PR #129 の prototype であり、本文には採用できない。

## 9. 成果物

| ファイル | 内容 |
|---|---|
| `protocol.json`、`manifest.csv` | 凍結した protocol と、79 instance × 20 replicate の計画 |
| `audit/` | データ・npz の hash、形状、θ̂、PR #129 での selector 再現、Z 誤差の較正、ノイズ水準の解釈 |
| `full/candidate_scores.csv` | instance × replicate × 列 × 候補：スコア、スコア/n、loading ノルム、grad_inf、収束、反復、ヘッセ行列の最小固有値、スコア誤差の上界、和の検算、所要時間 |
| `full/column_selection.csv` | instance × replicate × 列：選択、正誤、ΔQ、ΔQ/n、観測ごとの差の要約、Vuong 型 z（記述）、サンプルごとの margin、真の ‖f‖、x の要約、Z の誤差 |
| `full/z_error_by_instance.csv` | 腕 × replicate の Z 誤差（Procrustes RMSE、線形残差、サンプル内のばらつき、Z の sha16） |
| `full/runinfo.json` | git、環境、seed、所要時間、警告 |
| `full/analysis/` | 事前登録した集計（腕ごとの率、用量反応、paired 対比、最適化の診断、ノルムとの関係、AF2 との対応、signed margin）、図 3 枚、`results_tables.md`。`posthoc_*` は事後の記述表 |
| `smoke/` | rep01 だけの実行（本実行の前に分析コードの確認に使った） |

コード（`expfam/src/experimental/`）：
- `run_af3_selector_rescoring.py`（runner と audit）
- `analyze_af3_selector_rescoring.py`（集計と図）
- `test_af3_selector_rescoring.py`（6 件）

provenance：
- データ：AF2 の生成器と seed（1661001+r）
- 状態：`D:\tento\kennkyu_artifacts\auto_family_z_recovery_af2_20261110\full\fits\`（sha256 は AF2 の `fit_runs.csv`）
- noise seed：`[1664000, r, arm_code, index, seed]`
- bootstrap seed：20261113

**開示**
- 結果を読んだ後に追加したもの：`posthoc_*`（最適化誤差の規模、開始値による差、誤選択の性質）と、図 2 の縦軸の symlog 化（表示だけ）。事前登録した数値と判定規則は変えていない。
- G4 の「次の一手」が Z7 の結果と合わないことは、結果を見た後の解釈である（§5.3）。

## 10. 再現（repository root で実行）

```bash
python -m pytest -q expfam/src/experimental/test_af3_selector_rescoring.py
python expfam/src/experimental/run_af3_selector_rescoring.py --audit          --out-root <new_dir>
python expfam/src/experimental/run_af3_selector_rescoring.py --write-protocol --out-root <new_dir>
python expfam/src/experimental/run_af3_selector_rescoring.py --smoke          --out-root <new_dir>
python expfam/src/experimental/run_af3_selector_rescoring.py --execute        --out-root <new_dir>
python expfam/src/experimental/run_af3_selector_rescoring.py --analyze --stage full --out-root <new_dir>
```

AF2 の状態ディレクトリ（repository 外）が必要です。
