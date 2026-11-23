# 追加実験の科学的妥当性監査と事前設計（2026-11-23）

- **種類**：THEORY / AUDIT / EXPERIMENT DESIGN。**新しい MCEM・データ生成・数値 pilot・本実験は行っていない。** モデル実装・既存成果物は変更していない。
- **基準 commit**：`c535910`（PR #174 の最新 commit）。PR #168 の成果物は `856d475` で読んだ。
- **ラベル**：FACT（一次資料で確認）／DERIVED（数式から導出）／OBSERVED（既存実験の観測）／INTERPRETATION／HYPOTHESIS／JUDGMENT（研究上の判断。決定ではない）。
- **本書の protocol 案は凍結した正式 protocol ではない。** 研究者と指導教員の承認後に確定する（AGENTS.md §6 Human Gate）。
- 本書で新たに計算した数値は、Cora の生データ（`expfam/data/cora/`）の**記述統計だけ**である（列平均・辺数・ペア数・次数。モデルの fit はしていない）。計算は scratchpad の一時スクリプトで行い、repository には追加していない。

---

## 0. 結論の要約（JUDGMENT）

| 候補 | 判定 | 一言 |
|---|---|---|
| **A. Cora の厳密な held-out 分布比較** | **CONDITIONAL GO（最有力）** | 自然ネットワーク上で「X の family の選択」と「X を使うこと自体」が Y の予測に効くかを、まだ一度も正しく測っていない。既存の Cora held-out は Bernoulli-X だけ・zero-fill・全グラフ次数で選んだ subset で、比較になっていない。ただし**切片がないことによる交絡**を推定対象の定義に組み込む必要がある |
| B. 独立人工データでの誤指定再検証 | **NO-GO（現形のまま）** | Exp4-fixed・M1・E1・PR #168（1,280 fit）で問いの大部分は既に測られている。同じ生成器の新データを足しても、**切片なしによる「表現できる平均の違い」と family の効果の交絡**はそのまま再現されるだけで、新しい研究判断につながらない |
| C. 混在属性を共通 Z で統合する効果 | **NO-GO（独立実験として）** | 混在属性は列ごとの family（prototype F/E）でしか表現できず、正式モデルの証拠にならない。機構の分解は #31 で単独に確立できなかった。**正式モデルで答えられる部分（自然ネットワークで X が Y の予測を助けるか）は A の副次比較に含める** |

---

## 1. 研究全体の現状（FACT、PR #174 総括より要約）

- 目的は「指数型分布族への一般化と、その挙動・有効条件・限界の評価」で、有効性を前提にしない（`RESEARCH_MASTER.md` §1）。
- 正式モデルは `family_x` が全列共通の `DualExpFamLSM`。列ごとの family（F）・objective-consistent（E）・mask（D）は prototype で本文採用不可（§12.1）。
- ALLOWED：実装レベルの一般化、先行研究の再現（lineage B）、n 依存性・誤指定による悪化（lineage C、Exp2/Exp4-fixed）、識別可能性の population 命題。
- QUALIFIED：sparse-Y での補完効果（#27、E+F）、MovieLens の小さな正の向き（#33、E+F）。
- 実データで、**自然な関係 Y に対して strict held-out で family を比べた実験は存在しない**。Cora は唯一の自然ネットワークだが、Bernoulli-X/Bernoulli-Y の 1 設定だけで、zero-fill の評価である（§3）。
- 方向性 3 案（`thesis_direction_comparison.md`）のうち、案 A・B（拡張を中心にする）が推奨されている。本書の追加実験も「拡張の挙動・限界」の証拠を補う位置づけで評価する。

---

## 2. 既存実験との重複監査

### 2.1 候補 A（Cora の held-out 分布比較）

| 既存実験 | 既存の問い | 既存結果（FACT/OBSERVED） | 既存の限界 | 新実験で初めて分かること | 新実験なしで説明できること |
|---|---|---|---|---|---|
| Cora balanced subset / k-sweep（C） | in-sample で K と Z の構造 | Q 基準最小 K=1、AUC/AP 最大 K=6、NMI 最大 K=3（KI-011） | in-sample、Bernoulli-X だけ | — | 「K は目的で変わる」 |
| Cora held-out LP（C、`run_fixed_real_cora_heldout_link_prediction.py`） | 隠したエッジを当てられるか | n=280、K=3/6、test AP 0.432/0.463（random 0.167）、test AUC 0.737/0.750 | **Bernoulli-X だけ**、zero-fill、test 負例は学習に含まれる、全グラフ次数で subset 選択、Y-only なし、sampled negatives | — | 「random より良い順位付けができる」（限定つき） |
| Cora scaling held-out（C） | n=280→980 で同じ評価 | test AP（sampled）0.43〜0.51、all-candidate AP 0.011〜0.025（基準 0.0009〜0.0022） | 同上（`evaluation_mode = zero_filled_edge_hiding_no_pair_mask` と自ら記録） | — | 計算時間の目安（n=700 で 306〜416 s/fit） |
| common reconstruction eval（C） | 3 データ横断の in-sample 再構成 | Cora は Bernoulli-X | in-sample | — | — |
| shared-Z ablation（C／MovieLens は strict） | X+Y / Y-only / X-only | 人工：統合が最良（Scen A/B）。Wine・MovieLens：Y-only と同等以下 | **Cora で Y-only 対照がない**。Wine は Y がラベル由来、MovieLens は投影カウント | **自然ネットワークで X が Y の予測を助けるか** | 「共有 Z の効果は X–Y の構造の重なりに依存する」 |
| MovieLens strict held-out / user-disjoint（E+F） | mixed-X が Y を改善するか | +0.0042（23/30）、split 変動に比べ小さい | 映画ノードの投影、Poisson-Y の過分散、prototype | — | 実データで効果が小さいこと |
| Phase 7e held-out K 選択（E） | 合成データの held-out 基準 | harness・two-canary 漏洩検査・Bernoulli raw-η 対数スコアが実装済み | 合成データ・K 選択が目的 | — | **評価コードの再利用元** |

**重複の判定（JUDGMENT）**：Cora で「同じ X・Y・mask・K・予算で family だけを変える」比較、および Cora の Y-only 対照は**一度も行われていない**。A は既存実験と重複しない。

### 2.2 候補 B（独立人工データでの誤指定検証）

| 既存実験 | 既存の問い | 既存結果 | 既存の限界 | 新実験で初めて分かること | 新実験なしで説明できること |
|---|---|---|---|---|---|
| fixed_official Exp4（C、正式モデル） | X/Y の family 3×3 誤指定 | X だけの誤指定は 1.00×（Scen C）〜4.34×（Scen A、support-invalid）。Y 側は最大 19.6× | Scen 間で Y の family・真の X・誤指定の種類が同時に違う | — | 「誤指定の影響は条件で 0〜数倍」 |
| M1（E、per-column） | B6→P6、Y 観測率 1.0/0.1 | dense +0.020（15/20）、sparse +0.066（17/20） | 1 条件、last 推定 | — | support-compatible 誤指定の向き |
| E1（E） | M1 を mode・別 seed で再評価 | sparse mode +0.076、18/20、dense の last は再現せず | 同じデータ（R-seed） | — | 推定量で結論が揺れること |
| E2a（E） | init と MC の seed 分離 | 分離手段（`init_seed`/`mc_seed`）を実装 | — | — | — |
| **PR #168**（E、未 merge） | 誤指定の種類 × 列数 × Y 観測率 4 水準 | Y=1 で bp6 +0.012、bg6 +0.271、pg3 +0.105、allg +0.322。Y の情報量との交互作用の向きが種類で逆転 | **Gaussian arm は生の整数値を中心化せずに使用**（report §2）。切片なし生成器 | — | 誤指定の種類で効果が大きく違うこと |
| y_sparsity / #27 / single_vs_joint（F/E） | all-Gaussian vs per-column、Y 観測率 | dense ではほぼ同じ、Y 1 割で +0.12〜+0.43 | prototype | — | 「all-Gaussian の強さは dense-Y の現象」 |

**重複の判定**：問い（どの誤指定で・どの Y の情報量で・どれだけ Z が悪化するか）は、PR #168 がほぼ網羅している。未検証なのは「新しいデータでの再現（R-data）」だけで、**その再現は §5 の交絡を解かない**。

### 2.3 候補 C（共通 Z で混在属性を統合する効果）

| 既存実験 | 既存の問い | 既存結果 | 既存の限界 | 新実験で初めて分かること | 新実験なしで説明できること |
|---|---|---|---|---|---|
| per-column single vs joint（F、3 試行） | 単独 block vs 同時統合 | joint が全単独条件より良い（dense、小差） | 3 試行、legacy numerics | — | 「統合は単独より悪くない」 |
| Y sparsity stress（F、10 試行） | Y 観測率と all-Gaussian | 観測率が下がると差が単調に拡大 | prototype | — | Y が少ないほど X の扱いが効く |
| complementary blocks #27（E+F） | 補完的 block で統合の効果 | sparse y_only 比 +0.51、single_gaussian 比 +0.20；dense +0.009 | 機構未確立 | — | 補完設定で統合が効く |
| matched latent coverage #31（E+F） | coverage が原因か | Gaussian comparator I=+0.114（9/10）だが single_bernoulli は I=−0.204（0/10）、D_J=+0.201 | **単一機構として確立せず**（§12.4） | — | 機構は分離できなかった |
| MovieLens mixed X / user-disjoint（E+F） | 実データで mixed-X | 平均 +0.004、変動の範囲内。raw-count Poisson は 29/30 で悪化（原因未解明） | 投影 Y、切片・スケール・過分散が交絡 | — | 実データ効果は小さい |

**重複の判定**：機構の分解（列数・family・coverage・相補性・optimizer）は #27・#31 で試みられ、#31 は「単一機構として確立していない」と結論している。同じ prototype で因子をさらに分けても、per-column の正式採用が未決定である以上、修論の正式モデルの証拠にならない。

---

## 3. 候補 A の監査：既存 Cora held-out は何を評価していたか

### 3.1 コードで確認した事実（FACT）

`expfam/src/run_fixed_real_cora_heldout_link_prediction.py`（scaling 版も同じ方式）：

1. **subset 選択（L.162–171）**：`degree` は `parse_cora()`（L.149–152）で**全グラフの全エッジ**から計算し、各クラスの次数上位 40 件を取る。選ばれたノードの全グラフ平均次数は **12.1**、全ノード平均は **3.9**（本書で生データから計算）。
2. **X の列選択（L.179）**：subset 内の列和上位 50 語。X は目的変数ではないので、これは漏洩ではない。
3. **split（L.232–252）**：subset 内の正例エッジの 20% を test にし、`Y_train` で **0 に置き換える**。モデル（`DualExpFamLSMFixed`）には pair mask がないため、**学習尤度は全ペアを観測として扱う**。
4. **負例（L.255–265、L.543–544）**：train・test の負例は、どちらも `Y_sub == 0` から独立に抽出する。**test 負例のペアは学習尤度に「観測された 0」として入っている**。train 負例と test 負例は重なりうる。
5. **評価（L.335–358）**：test 正例 86 件 vs 負例 430 件（1:5）で AUC・AP。Z は最後の MC サンプル（L.322）。
6. **family**：Bernoulli-X / Bernoulli-Y だけ。Gaussian-X・Y-only は実行していない。

### 3.2 3 つの問題を区別する

| 種類 | 該当箇所 | 何が起きるか | 影響の向き |
|---|---|---|---|
| **欠測の誤処理**（missingness misspecification） | test 正例を `Y_train = 0` にして尤度に入れる | 「未観測」ではなく「辺がないことを観測した」として学習する。モデルは test 正例の確率を**下げる方向**に学習する | test 正例のスコアを下げる（性能を悲観側へ）。Bernoulli の対数スコアでは系統的な偏りになる |
| **漏洩**（label leakage） | (a) test 負例のラベル 0 が学習尤度に入っている。(b) 全グラフ次数（test エッジを含む）で subset を選んでいる | (a) 評価に使う負例を学習で見ている。(b) test エッジを多く持つノードほど選ばれやすい | (a) 負例の確率を下げる（楽観側へ）。(b) test 正例が多く含まれる subset になる |
| **部分集合選択による偏り**（selection / scope） | 次数上位ノードの subset | 次数の高い「核」だけを見ている。密度 0.011 は full Cora（0.00144）の約 8 倍 | 漏洩とは別に、**一般化の範囲**が「Cora の高次数の核」に限られる（KNOWN_ISSUES の NOT ALLOWED と整合） |

**INTERPRETATION**：既存実験が評価していたのは、
> 「高次数ノードの核で、学習時に 0 として隠したペアを、学習時にも 0 として見せたペアより上位に並べられるか」

という **missing-link の順位付け（PU 学習に近い transductive な課題）** である。これは GAE 型の link prediction で一般的な設定でもあり、それ自体は誤りではない。しかし、
- 対数スコア（確率の正しさ）の評価には使えない（欠測の誤処理で偏る）。
- test 負例が学習に入っているので、厳密な held-out ではない。
- 選択に test エッジを使っているので、絶対値（AP 2.6〜2.8×random）には小さくない楽観の余地がある。

向きが逆の 2 つの効果（誤処理は悲観、漏洩は楽観）が混在するため、**既存の数値の偏りの正味の向きは静的には決められない**。

### 3.3 既存実装の数理的な点（候補 A の設計に関わるもの）

| # | 内容 | 根拠 | 分類 |
|---|---|---|---|
| A-1 | lineage C には pair mask がない。mask は D（`model_dual_expfam_masked.py`）と E（`model_dual_expfam_consistent.py`）だけ。`train_mask=None` のとき D は C と数値的に同一（docstring・`test_experimental_models.py`） | FACT（code） | 新実験では **D/E 系の scalar-family クラスを使う必要がある**。数式は正式モデルと同じだが、コード系列は prototype → **Human Gate（§10 G1）** |
| A-2 | 精度行列の Y 項が `Z.T @ np.diag(var_y) @ Z`（fixed L.113、masked L.134、percolumn L.149）。ノード 1 つ・Newton 1 回ごとに n×n の行列を作るので、1 sweep が O(n³) | FACT（code） | n=700 では既存ログで実行できた（306〜416 s/fit）。**full Cora では計算量の BLOCKER**（§3.5）。数式は変えずに `(Z * var_y[:, None]).T @ Z` にできるが、実装変更なので今回はしない |
| A-3 | X 側に**切片がない**（`η^X = f_lᵀ z_i`） | FACT（AGENTS.md §1） | **WARNING**（§3.4、§5） |
| A-4 | Gaussian-X の F は閉形式（`reproduction/src/model.py` `calc_F`）、Bernoulli-X の F は Adam 50 step・lr 0.01（`model_dual_expfam.py` `_calc_F_adam`） | FACT（code） | **WARNING**：反復数が少ないと Bernoulli-X の F が最適化不足になり、family の差と optimizer の差が交絡する |
| A-5 | `scale_Z` は Z 全体の平均二乗を 1 にそろえるだけで、中心化しない | FACT（code） | DIAGNOSTIC：Z の平均のずれ（§3.4 の切片代わり）と相互作用する |
| A-6 | Y 側の 1/2：D/E は Y 項に extra 1/2 なし（採用式と一致）。M-step の `/(2Lφ)` は対称行列の全和を 2 で割る＝上三角和なので整合 | FACT（code）、DERIVED | 問題なし |
| A-7 | η の clip：E（consistent）は Bernoulli/Poisson の clip を除去（PC-001/002 の forward 修正）。Gaussian は不変 | FACT | 新実験は `numerics_mode="consistent"` を推奨 |
| A-8 | E-step のサンプリングに失敗すると mode のまま進む（`reproduction/src/model.py` L.456–460）。runner は数えない | FACT（PR #168 §8 でも指摘） | DIAGNOSTIC：頻度を記録する（`em_runner_instrumented.py` の利用を検討） |
| A-9 | 既存 Cora runner は NaN 時に `newton_alpha` を半減して最大 3 回やり直す（L.274–316） | FACT | 推定量が試行ごとに変わる。新実験は `failure_policy="fail_fast"`（M1・#168 と同じ）にする |
| A-10 | 予測に使う Z は既存では最後の MC サンプル。E1 で、点推定（last / mean5 / mode）により結論が揺れることが分かっている | FACT / OBSERVED | WARNING：主要な点推定を事前に固定する |

### 3.4 切片がないことの Cora への影響（DERIVED）

`z ~ N(0, I_K)` のもとで `f_lᵀ z ~ N(0, ‖f_l‖²)` は 0 について対称なので、

- Bernoulli-X の周辺平均：`E[σ(f_lᵀ z)] = 1/2`（‖f_l‖ によらず厳密）
- Gaussian-X の周辺平均：`E[f_lᵀ z] = 0`
- Poisson-X の周辺平均：`E[exp(f_lᵀ z)] = exp(‖f_l‖²/2) ≥ 1`

Cora の top-50 語（n=280 subset）の列平均は **最小 0.064・中央値 0.084・最大 0.521**（本書で計算。n=700 でも中央値 0.080）。

- **Bernoulli-X**：平均 0.08 を表すには `f_lᵀ z_i ≈ logit(0.08) ≈ −2.4` を多くのノードで実現する必要がある。z の事前分布は平均 0 なので、Z の行の平均を共通方向 u にずらして「切片の代わり」にするしかない。これは (i) 事前分布の項 `n‖m‖²/2` で罰せられ、(ii) **潜在次元の一部を消費**し、(iii) Y 側で `w (z_i+m)ᵀ(z_j+m) = w (z_iᵀz_j + mᵀz_i + mᵀz_j + ‖m‖²)` となって、**ノードごとの次数効果と w0 のずれ**を同時に作る。
- **Gaussian-X（生の 0/1）**：平均のずれは 0.08 で、標準偏差 √(0.08·0.92) ≈ 0.27 に比べて小さい。ずれの二乗は推定分散 σ_l² に吸収される。
- **分散による重み**：Gaussian-X の Z への寄与は `Fᵀ Σ⁻¹ F` で、σ_l² ≈ 0.08 なら 1 要素あたりの精度は約 12.5。Bernoulli-X の曲率 `μ(1−μ)` は最大 0.25（μ=0.08 なら 0.074）。**family を変えると、X と Y の相対的な重みも同時に変わる**。

**帰結（JUDGMENT）**：Cora で Bernoulli-X と Gaussian-X を比べた差は、「分布の形の正しさ」だけでなく、**(a) 表現できる平均の違い、(b) X と Y の相対的な重み、(c) F の optimizer の違い**を含む。新実験の推定対象は、
> 「切片のない正式モデルを、Bernoulli-X として使った場合と Gaussian-X として使った場合の held-out Y 予測の差」

と定義し、「family の正しさの効果」とは呼ばない。(a) を取り除くには X 側の切片（モデル変更、Human Gate）が必要で、今回は提案しない。

### 3.5 n=280 / 700 / full Cora の実行可能性（静的な見積もり。ベンチマークなし）

| 項目 | n=280 | n=700 | full（n=2708） |
|---|---:|---:|---:|
| ペア数（上三角） | 39,060 | 244,650 | 3,665,278 |
| 辺数（全グラフ次数で選んだ既存 subset） | 433 | 1,470 | 5,278 |
| 密度 | 0.0111 | 0.0060 | 0.00144 |
| 20% dyad holdout の test ペア数 | 7,812 | 48,930 | 733,056 |
| 期待される test 正例数（20%） | 約 87 | **約 294** | 約 1,056 |
| training ペア数（80%） | 31,248 | 195,720 | 2,932,222 |
| n×n float64 1 枚 | 0.6 MB | 3.9 MB | 58.7 MB |
| joint mode の Hessian（nK×nK、K=6） | 13 MB | 141 MB | **2.1 GB** |
| 既存ログの実行時間（lineage C、L=5、NITER=8） | 40 s/fit | 306 s（K=3）、416 s（K=6） | 未実行。n=980 の 740〜836 s から n² 〜 n³ で外挿すると **約 1.6〜4.9 h/fit**（推定） |

- 出典：ペア数・辺数・密度は本書で生データから計算。実行時間は `expfam/results/real_data/cora_scaling_heldout/cora_scaling_agg.csv` と `cora_scaling_stress_summary.csv`（lineage C）。**lineage E（consistent・mask）の実行時間は測っていない**。外挿は A-2 の O(n³) 項によるもので、別マシン・別負荷では変わる。
- 辺数は「全グラフ次数で選んだ」既存 subset の値である。新設計の「学習エッジだけの次数で選ぶ」subset では少し変わる。
- **n=280**：test 正例が約 87 件で、自然比率の対数スコアは負例に支配され、正例側の変動が大きい。主設計には小さすぎる。
- **full Cora**：選択の偏りがない点で最も望ましいが、A-2 の計算量と joint mode の Hessian（K=6 で 2.1 GB）のため、**実装の変更（数式は不変の性能改善）と等価性 test が先に必要**。これは今回の禁止事項（実装修正）に当たる。
- **n=700**：既存の実行実績があり、test 正例が約 300 件、joint mode も実行可能。→ **主設計に推奨**。

### 3.6 再利用できる部分と新たに必要な実装

| 区分 | 内容 |
|---|---|
| 再利用 | `experimental/em_runner.py::run_em_experimental`（`train_mask`、`numerics_mode="consistent"`、`fix_x=True` で Y-only、`init_seed`/`mc_seed` 分離、`failure_policy="fail_fast"`）、`DualExpFamLSMConsistent`（scalar family + mask）、`laplace_k_criterion.find_joint_mode`（E1・#168 の mode 推定）、`tools/research_audit/run_heldout_k_selection_pilot.py` の `validate_pair_masks`・`heldout_raw_eta_pairs`・`heldout_bernoulli_mean_log_score`・two-canary 漏洩検査の考え方、`parse_cora()` |
| 新規 | (1) Cora 用の strict held-out runner（dyad split → 学習エッジだけの次数で subset 選択 → mask の制限 → arm × K × seed の manifest → runinfo）。(2) test ペア全体での AP・AUC・Brier・calibration の集計。(3) 既存出力を上書きしない新しい出力先 |
| モデル変更 | **なし**（n=700 の場合）。full Cora にする場合だけ A-2 の性能改善が必要（別途承認） |

**数理的な比較可能性（DERIVED）**：同じ X・Y・train mask・K・L・NITER・Newton 設定・seed なら、3 つの arm は同じ Y 尤度・同じ事前分布を持ち、X の尤度項だけが違う（Y-only は X 項が 0）。held-out Y の対数スコアは同じ Bernoulli の確率で評価するので、**X の family をまたいで比べられる**（X の尤度は family をまたいで比べない）。

---

## 4. 候補 A の評価設計

### 4.1 主比較と副次比較

| arm | X の扱い | 位置づけ |
|---|---|---|
| **BX** | Bernoulli-X / Bernoulli-Y、生の 0/1 | 主比較 |
| **GX-raw** | Gaussian-X / Bernoulli-Y、**生の 0/1 を連続値として**扱う。Σ_X（対角）は M-step の MLE | 主比較 |
| **YO** | Y-only（`fix_x=True`、F=0）。X 項が 0 なので family によらない | 副次比較 |
| GX-ctr（任意） | Gaussian-X、**列ごとに中心化した X**（X は目的変数ではないので全ノードで中心化してよい） | **運用上の比較**。GX-raw と別の表にし、結果を混ぜない。採否は G5 |
| REF | 学習ペアの密度だけの定数予測（fit なし） | スケールの参照 |

- GX-ctr は「切片の役割を前処理に持たせた Gaussian」で、データが変わるので family の比較ではない。BX には同じ処理ができない。

### 4.2 指標

- **主指標**：test ペア全体（自然な比率、負例のサンプリングなし）での held-out Bernoulli の平均対数スコア `mean(y·η − log(1+e^η))`。η は raw（clip なし）。
- **主要な推定量**：`Δ_BG = LS(BX) − LS(GX-raw)`（正なら Bernoulli-X が良い）。
- **副次指標**：`LS(BX) − LS(YO)`、`LS(GX-raw) − LS(YO)`；test ペア全体での AP・AUC；Brier score；calibration-in-the-large（予測平均 vs test の正例率）と 10 分位の reliability；正例・負例に分けた対数スコア；split 間・seed 間の変動；fit の失敗率；学習時間。
- 既存の sampled-negative（1:5）AP は、過去との連続性のための参考値としてだけ出す。

### 4.3 統計単位と不確実性

- **統計単位は「Cora という 1 つのネットワーク」**。同じネットワークから作った split は独立なデータセットではない。p 値・「有意」・split を n とした信頼区間は使わない。
- 報告：split ごとの paired Δ（seed 平均）、5 split の値すべて・中央値・範囲・符号の数；split 内の seed 間 SD；split 内で test ペアをノード単位で再抽出する bootstrap 区間（**test 集合の標本変動だけ**を表す記述的な量。再学習の変動は含まない）。
- 解釈の規則（事前に固定）：|Δ| が seed 間の変動・split 間の範囲より小さければ「推定の変動の範囲内で差は見えない」と書く。数値の閾値で合否は決めない。

### 4.4 守ること

- train にも test にも入らないペアを学習中に暗黙の 0 にしない（subset 外のペアはそもそも尤度に入らない。subset 内は train mask の外を学習に使わない）。
- subset の選択には**学習エッジだけの次数**を使う（§7 手順 2）。
- K・NITER・前処理・点推定・arm の追加は**実行前に固定**し、test の結果で変えない。
- 失敗した fit は置き換えない。両 arm が有効なペアで paired に比べ、失敗率は arm ごとに別に報告する。

---

## 5. 候補 B の監査：独立人工データでの誤指定検証

### 5.1 切片がないことによる交絡（DERIVED）

§3.4 の式により、PR #168 の生成器（`data_generator_canonical_mixed.py`、`η^X = Z Fᵀ`、L.187、切片なし）では：

- 真の Bernoulli 列の周辺平均は **1/2**。
- それを Poisson で fit すると、モデルが Z の事前分布のもとで表す周辺平均は `exp(‖f‖²/2) ≥ 1`。0.5 に合わせるには Z の平均をずらす必要がある。
- Gaussian で fit すると（PR #168 は生の整数値を中心化せずに使用、report §2）、モデルの周辺平均は 0。

したがって **bp6（+0.012〜+0.082）と bg6（+0.146〜+0.271）の差には、「分布の形の違い」と「表現できる平均の違い（＝切片がないこと）」が混ざっている**。Exp4-fixed の X 側誤指定（Scen A の Poisson→Gaussian 2.09× など）も同じ生成器の性質（`data_generator_expfam.py`、切片なし）を持つ。

### 5.2 各論点への回答

| 論点 | 判断 |
|---|---|
| 切片なしで解釈可能な比較が成り立つ条件 | (i) 推定対象を「切片なしの正式モデルを family X で使った場合」と定義し、family の正しさとは呼ばない場合。(ii) 誤指定された family でも周辺平均を表現できる組合せ（例：真の Gaussian を Gaussian で fit する以外にはほぼない）。どちらも「family の効果」の純粋な分離にはならない |
| 既存の誤指定実験から既に主張できること | 「切片なしの本モデルでは、X 側の誤指定の影響は種類と Y の情報量で 0 から数倍まで変わり、交互作用の向きも種類で違う」（Exp4-fixed、M1/E1/#168、限定つき） |
| 同じ生成器の独立新データの付加価値 | R-seed → R-data への格上げ。ただし同じ交絡をそのまま再現するだけで、**解釈は変わらない**。研究判断（どの主張を書くか）に新しい情報をほとんど与えない |
| 全列共通 family の正式モデルに限定する意義 | lineage の問題（prototype F）を外せる。ただし正式モデルで混在 X は作れないので、Exp4 型（全列が 1 つの family）に戻ることになり、Exp4-fixed と重複する |
| Y の観測率を操作するときの問題 | 正式実装（C）に mask がないため、D/E（prototype）が必要。D/E の scalar クラスは数式上は正式モデルと同じだが、コード系列としての採用は Human Gate |
| 1/2・clip・最適化の制約 | 1/2 は D/E で採用式と一致。clip は E で解消。Gaussian（閉形式）と Bernoulli/Poisson（Adam）の F の optimizer の違いは残り、誤指定の効果と交絡する（#27 でも `all_gaussian` を primary から外した理由） |

**判定**：交絡を解けない以上、単純にデータ数を増やす実験は推奨しない（NO-GO）。もし将来、X 側に切片を入れたモデル（モデル変更、Human Gate）を検討するなら、そのときに「切片あり／なし × family」の 2×2 を設計すると交絡を分離できる。これは今回の範囲外で、決定もしない。

---

## 6. 候補 C の監査：混在属性を共通 Z で統合する効果

### 6.1 まだ不明なこと

joint モデルの改善（#27 の sparse で y_only 比 +0.51）が、次のどれに由来するかは**分離されていない**：

1. 属性列が増えたこと（情報量）
2. family を適切に指定したこと（#168 の bg6・allg の比較が部分的に関係）
3. 潜在次元の coverage（#31：Gaussian comparator では整合、secondary では符号不一致、D_J も大きい → 未確立）
4. X と Y の相補性（#27 の sparse/dense の差）
5. optimizer や dispersion の差（Gaussian は閉形式＋推定分散、他は Adam・φ=1）

5 は family の割当てと切り離せない（family を変えると optimizer と重みが自動的に変わる、§3.4）。2 と 5 の分離には、同じ optimizer・同じ重みで family だけを変える実装が必要で、正式モデルの範囲を超える。

### 6.2 方向性 3 案との整合

| 案 | C の扱い |
|---|---|
| 案 A（拡張中心・分布既知） | 正式モデルは全列共通 family なので、C は prototype の補足証拠（QUALIFIED）にとどまる。新実験は不要 |
| 案 B（案 A ＋ 自動選択の限界） | 同上。混在属性は AF の前提（列ごとの family）とつながるが、AF は prototype で、C の新実験は AF の問いにも答えない |
| 案 C（相互依存を主題化） | C の機構分解は案 C 寄りの研究で、案 C 自体の実現可能性が低い |

**判定**：独立実験としては NO-GO。per-column の正式採用（Human Gate）が決まらない限り、追加の因子分解は修論の主張を強めない。**正式モデルで答えられる部分（自然ネットワークで X を使うことが Y の予測を助けるか）は、A の YO 比較に含める。**

---

## 7. 数値的な正当性（候補 A で使う実装）

| 項目 | 内容（lineage E：`DualExpFamLSMConsistent`、scalar family、mask あり） | 分類 |
|---|---|---|
| 理論上の目的関数 | `Σ_i ln N(z_i; 0, I) + Σ_{i,l} ln p_X(x_il | f_lᵀz_i) + Σ_{(i,j)∈O, i<j} ln p_Y(y_ij | w0 + w z_iᵀz_j)`。O は train mask の上三角 | — |
| 勾配 | Y 項 `w Σ_{j:(i,j)∈O} [y_ij − A'(η_ij)]/φ_Y z_j`（extra 1/2 なし）、X 項 `Fᵀ (x_i − A_X'(F z_i))/φ_X` | FACT（masked L.93–116） |
| 精度行列 | `I + Fᵀ V_X F + w² Σ_{j∈O_i} V_Y(η_ij) z_j z_jᵀ` | FACT（masked L.118–136） |
| 1/2 係数 | E-step・M-step とも採用式と整合（A-6） | 問題なし |
| η の clip | consistent mode で Bernoulli の clip なし。非有限値は例外 | 問題なし |
| Gaussian の dispersion | X：Σ_X 対角 MLE（下限 1e-8）。Y は Bernoulli なので σ_Y² なし | DIAGNOSTIC（生の 0/1 では σ_l² ≈ p(1−p) + p²程度で下限に達しない見込み） |
| Y pair mask | `_mask_f` で E-step・M-step・Q をすべて制限。対角は False | FACT。two-canary 漏洩検査（test の y を 0/1 に差し替えて fit が同一か）を runner の test に入れる |
| MCEM の停止 | 固定反復（NITER）。収束判定なし。§20 で L=5・40 反復でも固定点に達していないことが分かっている | **WARNING**：推定量は「固定予算の MCEM の出力」と定義し、収束後の MLE とは呼ばない。Bernoulli-X の F（Adam）の最適化不足を、学習データだけの診断（M-step 前後の Q、F の勾配ノルム）で記録する |
| 失敗した fit | `fail_fast`（retry・置換・seed 変更なし）。失敗率は結果として報告 | — |
| Z の推定量 | 主要：θ̂ 固定の joint mode（`find_joint_mode`、Z0 = 最後のサンプル）。副次：last、mean5 | WARNING（A-10）。2 つの開始点で mode の一致を記録（E1 の方式） |
| 初期値・乱数 | `init_seed` と `mc_seed` を分け、split・K・seed 番号から決定的に作る。**同じ (split, K, seed) では全 arm で同じ seed** | — |
| E-step のサンプリング fallback | 失敗時に mode を保持（A-8） | DIAGNOSTIC：頻度を記録 |
| 計算量 | O(n³) の `np.diag`（A-2） | n=700 では問題なし。full Cora では BLOCKER |

**BLOCKER（既存コードをそのまま使う場合）**：`run_fixed_real_cora_heldout_link_prediction.py` を family だけ変えて再実行すること。zero-fill・test 負例の学習混入・全グラフ次数による選択があり、§4 の対数スコア比較の妥当性を満たさない。**新しい runner が必要**。

---

## 8. 候補の比較（JUDGMENT）

| 観点 | A | B | C |
|---|---|---|---|
| 1. 研究目的との一致 | 高（実データでの挙動・限界） | 中（既に答えがある） | 中（正式モデル外） |
| 2. 既存実験との差 | **大**（Cora の family 比較・Y-only・strict held-out は未実施） | 小（#168 と重複） | 小（#27/#31 と重複） |
| 3. 明確な仮説 | あり（§9 H1・H2） | 交絡のため曖昧 | 機構が多すぎて単一仮説にならない |
| 4. 正負どちらでも有益か | **はい**（差なし＝「実データでは X の family も X 自体も Y の予測に効かない」も限界の証拠） | 再現しても解釈は不変 | 機構が分かれなければ不明のまま |
| 5. 正式モデルの証拠か | 数式は正式モデル（scalar family）。コード系列は E → G1 | 正式モデルなら Exp4 と重複 | ならない（per-column） |
| 6. 実データの証拠か | **はい**（唯一の自然ネットワーク） | いいえ | いいえ |
| 7. 既存コードでの実現性 | 中（新 runner が必要、モデル変更なし） | 高 | 高 |
| 8. 計算コスト | n=700 で約 120 fit、1 fit 5〜7 分程度（lineage C の実績からの推定） | 中 | 中 |
| 9. 未解決課題の解決 | KNOWN_ISSUES の Cora 評価の限界（zero-fill・選択）を解消。Cora の shared-Z の空白を埋める | 交絡は残る | #31 の未確立は残る |
| 10. 修論での使い道 | 実データ章の中心結果（改善・差なし・悪化のどれでも） | 付録の再現 | 付録 |
| **判定** | **CONDITIONAL GO** | **NO-GO** | **NO-GO** |

**実行の優先順位**：1. A（条件つき）のみ。B・C は実行しない。

**実行しない方がよい実験**：
- 既存 Cora runner の family 差し替えによる再実行（§7 BLOCKER）。
- PR #168 と同じ生成器での独立データ再実行（交絡が残る）。
- per-column の因子分解の追加（正式モデルの証拠にならない）。
- full Cora での実行（性能改善の実装と承認が先）。
- test の結果を見て K・前処理・点推定を選ぶ解析。

---

## 9. 最有力候補の protocol 案（未凍結）

| 項目 | 案 |
|---|---|
| 名称 | CORA-SH1：Cora strict held-out の X family 比較 |
| 種類 | CHARACTERIZATION（効果の向きと大きさの記述。優劣の検定ではない） |
| 研究上の問い | 自然な引用ネットワークで、切片のない正式モデルの X を Bernoulli として使うか Gaussian として使うかで、held-out Y の確率予測は変わるか。X を使うこと自体は Y-only より予測を良くするか |
| 仮説 | H1（主）：`Δ_BG ≠ 0`（向きは事前に仮定しない。§3.4 の通り、Bernoulli は分布の形が正しいが平均を表現しにくく、Gaussian はその逆）。H2（副）：`LS(BX) − LS(YO)` と `LS(GX-raw) − LS(YO)` の向き |
| データ | `expfam/data/cora/`（n=2708、d=1433、無向化した 5,278 辺）。X は 0/1 の単語、Y は引用の有無 |
| 手順 1：split | full Cora の全ペア（上三角）を、split ごとに一様に 80% train / 20% test へ分ける（dyad holdout。正例・負例を区別しない）。split seed は事前に固定。S = 5 |
| 手順 2：subset | 各 split で、**train エッジだけ**から各ノードの次数を計算し、各クラスで次数上位 100 ノード（同順位は node id で決める）→ n = 700。train/test mask を subset 内のペアに制限する |
| 手順 3：X | subset 内の train に依存しない規則で列を選ぶ：**full Cora の列和上位 50 語**（Y を使わない。既存の n=280 固定の語彙とは異なることを記録） |
| arm | BX、GX-raw、YO（主・副）。GX-ctr は G5 で採否を決める |
| K | {3, 6} を事前に固定し、両方報告。**主要な K は実行前に研究者が決める**（JUDGMENT：K=6 を主要にする案。理由は、§3.4 の切片の代わりに 1 次元を使う可能性があるので、容量の小さい K=3 では Bernoulli-X に構造的な不利が出やすいため。結果を見て変えない） |
| 推定 | `DualExpFamLSMConsistent`、`numerics_mode="consistent"`、L=5、NITER=8（既存の Cora・M1・#168 と同じ。予算を増やす場合は実行前に決める）、Newton max_iter=10・alpha=0.5、informed init（w0 = logit(train 密度)、w=0.5、非 Gaussian-X は F×0.2）、`failure_policy="fail_fast"` |
| seed | split × K ごとに 3 個の model seed（`init_seed`/`mc_seed` を分離）。全 arm で共有 |
| 規模 | 5 split × 2 K × 3 seed × 3 arm = **90 fit**（GX-ctr を入れると 120） |
| Z の点推定 | 主要：θ̂ 固定の joint mode（Z0 = last）。副次：last、mean5。2 開始点の mode の差を記録 |
| 主指標 | test ペア全体での held-out Bernoulli 平均対数スコア（raw η） |
| 主推定量 | `Δ_BG`（split ごとに seed 平均、K ごと） |
| 副次 | YO との差、AP・AUC（test ペア全体）、Brier、calibration、正例・負例別の対数スコア、seed 間・split 間の変動、失敗率、mode の診断、M-step Q の診断、runtime |
| 統計解析 | §4.3。5 split の値をすべて示す。p 値・有意性・split を独立とみなした区間は使わない |
| 漏洩の検査 | 実行前に two-canary 検査（test の y を 0 と 1 に差し替えた 2 つの payload で、fit の出力が bit 単位で同一）と、mask の検査（train ∩ test = ∅、対称、対角 False）を runner の test として通す。検査コードは既存 harness から最小限だけ移す（新しい監査ツールは作らない） |
| 記録 | runinfo（seed、mask の hash、subset の node id hash、語彙の hash、git head、Python・主要パッケージ版、KI-014） |
| 出力先 | 新しいディレクトリ（例：`expfam/results/real_data/cora_strict_heldout_family_<date>/`）。既存の Cora 出力を上書きしない |
| 結果の採用 | BX が良い・差なし・GX が良い・YO と同等、のどれも研究結果として採用する。結果を見た後で K・seed・前処理・指標を変えない |
| 解釈の制約 | 「切片のない本モデルの Bernoulli-X と Gaussian-X の比較」と呼び、「分布の正しさの効果」とは呼ばない。範囲は「Cora の学習グラフ上の高次数の核（n=700）」で、full Cora・他のネットワークへの一般化は主張しない |
| 扱わないこと | K 選択、family の自動選択、Poisson-X、切片の追加、per-column、full Cora |

**計算量の見積もり**：lineage C の n=700 実績（306〜416 s/fit）を当てはめると、90 fit で約 8〜10 時間（直列）。joint mode と lineage E のオーバーヘッドは未測定。**実行前に smoke（数 fit）で時間を確かめる必要があるが、それも新しい MCEM なので承認後に行う。**

---

## 10. 指導教員・研究者に承認を求める事項（Human Gate）

| # | 事項 | 理由 |
|---|---|---|
| G1 | 実データの held-out 実験に lineage E の scalar-family クラス（`DualExpFamLSMConsistent`）を使い、その結果を修論の証拠に使ってよいか（付録扱いか本文か） | 数式は正式モデルと同じだが、コード系列は prototype（§12.1） |
| G2 | 推定対象を「切片のない正式モデルでの family の比較」と定義すること | 「分布の正しさの効果」とは呼べない（§3.4） |
| G3 | 主要な K（3 か 6 か）と NITER | 実行前に固定する必要がある |
| G4 | subset を n=700（学習エッジの次数で選択）にすること。full Cora を後で行うか | full Cora は性能改善の実装（A-2）と別の承認が必要 |
| G5 | 運用上の比較 GX-ctr を含めるか | 含めると 30 fit 増え、結果の書き分けが必要 |
| G6 | 新しい runner の実装と、それに伴う smoke・本実験の実行 | 新しい研究実験の開始（AGENTS.md §6） |
| G7 | B・C を行わないこと | 本書の NO-GO 判定は提案であり決定ではない |

---

## 付録 A. 本書で読んだ主な一次資料

| 対象 | パス |
|---|---|
| Cora held-out runner | `expfam/src/run_fixed_real_cora_heldout_link_prediction.py`（L.149–152、162–171、232–265、272–328、335–358、537–571） |
| Cora scaling | `expfam/src/run_fixed_real_cora_scaling_heldout.py`、`expfam/results/real_data/cora_scaling_heldout/*.csv`、`cora_scaling_heldout_run.log` |
| Cora 結果 | `expfam/results/real_data/cora_heldout_link_prediction/cora_heldout_{agg,data_summary}.csv`、`cora_clean/*.csv` |
| モデル | `expfam/src/model_dual_expfam_fixed.py`、`model_dual_expfam.py`、`model_expfam.py`、`reproduction/src/model.py`（L.360–460）、`expfam/src/experimental/model_dual_expfam_masked.py`、`model_dual_expfam_consistent.py`、`em_runner.py`（L.90–330） |
| held-out 評価 | `tools/research_audit/run_heldout_k_selection_pilot.py`（L.127–240） |
| 誤指定 | `reports/advisor_feedback/distribution_misspecification_synthesis_20261013.md`、`expfam/results/misspecification_m1/phase_m1_20261013/summary.json`、`reports/research_improvement/e1_z_point_estimator_m1_20261104.md`、PR #168 `856d475:reports/misspecification_extended/results_20261110.md`、`expfam/src/experimental/data_generator_canonical_mixed.py`（L.187） |
| 統合・ablation | `reports/shared_z_ablation/shared_z_ablation_report_20260708.md`、`reports/per_column_family/single_vs_joint_per_column_report_20260711.md`、`RESEARCH_MASTER.md` §8b・§12・§20 |
| 総括 | `reports/research_synthesis_20261116/*.md`（PR #174） |
| 正本 | `AGENTS.md`、`RESEARCH_MASTER.md`、`KNOWN_ISSUES.md`（KI-011、KI-012）、`EXPERIMENT_REGISTRY.md`（L.61–80） |

## 付録 B. 本書で計算した記述統計（モデル fit なし）

| 量 | 値 |
|---|---|
| full Cora | n=2708、d=1433、辺 5,278、ペア 3,665,278、密度 0.00144、X 全体の密度 0.0127 |
| 既存 n=280 subset | 辺 433、密度 0.01109、top-50 語の列平均 最小 0.064 / 中央値 0.084 / 最大 0.521、選択ノードの全グラフ平均次数 12.14（全体 3.90） |
| 既存 n=700 subset | 辺 1,470、密度 0.00601、top-50 語の列平均 中央値 0.080、選択ノードの全グラフ平均次数 7.94 |
| full Cora の top-50 語 | 列平均 最小 0.058 / 中央値 0.073 / 最大 0.400 |
