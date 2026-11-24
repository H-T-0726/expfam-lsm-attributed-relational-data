# 事前登録 protocol（**案。未凍結**）

- 作成日：2026-11-25。branch `experiment/k-selection-exp2-generalization-20261125`。
- **状態：DRAFT / NOT FROZEN。** §0 の判断が決まるまで凍結しない。本実験は開始していない。
- 凍結するときは、この文書の内容を機械可読の `protocol.json` にし、その SHA-256 と実行コードの commit を `execution_manifest.md` に記録する。そのあとで実行する。

## 0. 本実験の前に研究者が決めること（Human Gate）

| # | 判断 | 選択肢 | 推奨（JUDGMENT） | 理由 |
|---|---|---|---|---|
| **D1** | 実装系列 | (a) lineage E-scalar（`DualExpFamLSMConsistent`、canonical numerics、fail_fast）<br>(b) fixed（C、legacy numerics、本文の主系列候補） | (a)。ただし結果は「lineage E、本文採用は未決定」と明記する | 既存の K 選択研究（Phase 7e〜9、clean n-sweep、Fig.3 条件、d-sweep）がすべて E で、比較できる。C_Lap が E でしか動かない。fixed の η clip は飽和の多い条件で目的関数とずれる。**ただし E-scalar を本文の根拠にするかは、修論採用資料の Human Gate（判断事項 2）と同じ判断になる** |
| **D2** | Series A/B/C の生成器 | (a) historical（fixed official Exp1 と比較できる）<br>(b) canonical-clean-v1 | (b) | historical は literal な draw ではなく、K_TRUE とともに Y の信号が 9 倍になる（`mathematical_validity.md` §1・§4・§10） |
| **D3** | K_TRUE をまたいだ信号の規則 | (a) 1 方向あたりの信号を固定（w と f_scale を K によらず固定。元論文の Step 2・3 と同じ考え方。全体の信号は K とともに増える）<br>(b) 全体の信号を固定（w²K と平均行ノルム² を固定。clean n-sweep・d-sweep の規則。1 方向あたりは 1/K に薄まる）<br>(c) 両方（計算量 2 倍） | (a) を主、(b) を感度分析（計算量が許すなら (c)） | (a) は「生成に使った各次元が同じ強さで存在するとき、基準がその数を選べるか」を問う。(b) は「同じ情報量を多くの次元に分けたとき」の検出限界を問い、K=9 で 1 方向あたりの信号が K=1 の 1/9 になる。**どちらを主にするかで研究問いの意味が変わる** |
| **D4** | Series B（Poisson-Y）の w・w0 | 規則 (a) なら w を全 K で固定し、\|w\| < 1/2（分散が有限）。例：w = 0.4、w0 = 0<br>規則 (b) なら w_K = c/√K で c < 0.5（w²K < 0.25）| 研究者が値を決める | canonical Poisson-Y は \|w\| ≥ 1/2 で分散が発散する（P6）。既存の matched 規則（c = √3）は全 K で生成できない（`mathematical_validity.md` §5） |
| **D5** | Series O の扱い | (a) canonical-clean-v1（G/B）だけ<br>(b) (a) に加えて、元論文の本文どおりの生成器（PT）を「元論文との対応条件」として追加 | (b) | PT は X がほぼノイズなしで、w がデータごとに大きく変わる特殊な条件。元論文との対応はこちらで見る。既存の Fig.3 条件研究（1 データ × 10 初期値、main 未統合）を、独立なデータの反復で補う |
| **D6** | 計算予算 | §8 の表 | §8 | — |

**BLOCKER（数学的に実行できない、または解釈できない）はない。** D2〜D4 を決めないと Series B の生成ができず、各系列の K_TRUE 間比較の意味が定まらない。そのため本実験は開始しない。

## 1. 研究問い（primary）

正式モデル（全 X 列で共通の family、X 切片なし、スカラー w0・w、採用式）で、生成時の潜在次元 K_TRUE ∈ {1,3,5,7,9} を変えたとき、候補 K ∈ {1,…,10} から Q ベースの基準 C_Q が選ぶ K は、K_TRUE とどれだけ一致し、どちらに外れるか。family の組（O・A・B・C）でどう違うか。

- 評価するのは **K_TRUE との一致**である。**K* との一致とは言わない**。K* = K_TRUE が成り立つ条件は `mathematical_validity.md` §3 にあり、それを満たす系列でも本研究では証明していない部分がある。
- C_Q は Schwarz BIC ではない（ICL 型）。

## 2. 固定する条件

| 項目 | 値 |
|---|---|
| 系列（X/Y） | O = Gaussian/Bernoulli、A = Poisson/Bernoulli、B = Gaussian/Poisson、C = Bernoulli/Gaussian |
| 対象外 | 列ごとの family、自動分布選択、COM-Poisson、NB |
| K_TRUE | {1, 3, 5, 7, 9}（**削らない**） |
| 候補 K | {1, …, 10}（**削らない**） |
| n、d | 150、15 |
| L（MC サンプル数）、EM 反復数 | 10、10（元論文の Table I と同じ） |
| 独立な生成データの数 R | 系列 × K_TRUE ごとに 10 |
| 初期値の数 T | データごとに 2（init_seed と mc_seed を分ける） |
| 観測 | 全ペア（held-out なし） |
| 初期化 | 観測データだけから作る（w0 = データの logit 密度など、既存の informed init）。**真値を fit に渡さない** |

## 3. seed の規則

- data_seed = 2026112500·10 + 10⁵·series_index + 10³·K_TRUE + r（r = 0..9）。系列 × K_TRUE × r で一意。候補 K によらない（**同じデータを全候補 K で使う**）。
- init_seed = data_seed·10 + t、mc_seed = data_seed·10 + 5 + t（t = 0, 1）。候補 K によらない。
- 実行前に、全 seed の重複がないことを検査する。

## 4. 基準

| 役割 | 基準 | 適用範囲 |
|---|---|---|
| **primary** | C_Q = −2 Q_strict + p̂ ln n（`calc_bic_dual`。p̂ = Kd − K(K−1)/2 + d·[G-X] + [G-Y]） | 全系列 |
| secondary | C_Lap（`laplace_k_criterion.py`。同じ fit 上） | O・A・B（**C は Gaussian-Y なので対象外**）。D1 で (a) を選んだ場合だけ |
| 参考 | 元論文 Eq.(26) に着想を得た条件付き plug-in（最終サンプル） | 全系列。**Eq.(26) の実装とは呼ばない** |
| 含めない | held-out 予測スコア | 別の fit（80% 学習）が要り、目的（予測）が違う。必要なら別の事前登録にする |

## 5. 選択規則

- **primary**：データ r ごとに、各候補 K について 2 つの初期値のうち Q_strict が大きい（= C_Q が小さい）fit を採用し、C_Q が最小の K を選ぶ。
- **同点**：差が 10⁻⁹ 以内なら小さい K。
- **secondary**：初期値ごとの選択（各 t で独立に argmin）。2 つの初期値の選択の一致率を報告する。
- **参考**：元論文に寄せた規則。10 データの平均曲線の argmin、と最良データの曲線の argmin。**主な結論には使わない。**

## 6. 失敗と数値異常

- fail_fast とする。retry・seed の変更・反復の追加・置き換え fit・resume による救済はしない。
- 失敗した fit（例外、非有限の Q_strict や θ）は `failures.csv` に記録する。
- primary：ある候補 K で 2 つの初期値が両方失敗したら、そのデータは INCOMPLETE（補完しない）。1 つだけ失敗したら、残る 1 つを使い、そのことを記録する。
- 生成器の fail-fast 停止（GeneratorStop）は STOP 条件とし、実行を止めて報告する（seed を変えない）。
- C_Lap の NOT_STATIONARY などは C_Lap の選択だけを INCOMPLETE にする。

## 7. 集計と出力

- 集計単位はデータ（系列 × K_TRUE × r）。初期値の反復を独立なデータとして数えない。
- 主要な出力：
  - K_TRUE ごとの選択 K の分布
  - 混同行列（K_TRUE × 選択 K）
  - 正解・過小・過大の数
  - family 別、基準別の比較
  - K ごとの基準の曲線
  - 初期値の影響（2 初期値の一致率、Q_strict の差）
  - 収束の診断（最後の反復の Q_strict の相対変化。記述のみでゲートにしない）
  - 失敗の一覧
  - 元論文 Fig.3 に対応する図（各 K_TRUE の平均曲線）
  - K_TRUE が大きい場合の挙動
  - 基準の間で結論が異なる条件
- 保存先：`expfam/results/k_selection/k_exp2_generalization_20261125/main/`（per-fit CSV、per-dataset 選択、集計 CSV、failures、seed manifest、runinfo、code SHA、protocol SHA-256、図）。

## 8. 期待 fit 数と計算予算

1 fit あたりの時間は `timing_smoke/timing_smoke.json`（lineage E と fixed、K=1 と 10、L=10、10 反復）による。並列は 8 worker を仮定する。

| 構成 | fit 数 | 備考 |
|---|---|---|
| O・A・B・C × 5 K_TRUE × 10 候補 × R=10 × T=2 | 4,000 | primary |
| D5 (b)：PT 生成器の Series O | 1,000 | 元論文との対応条件 |
| D3 (c)：信号規則の感度分析（(b) 規則で同じ構成） | +4,000 | 任意 |

所要時間の見積もりは `execution_manifest.md` §3 に記録する。

## 9. 結果を見た後に変えないこと

family、K_TRUE、候補 K、seed、反復数、L、基準、選択規則、失敗処理、R、T。
