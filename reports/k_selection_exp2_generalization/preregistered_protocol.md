# 事前登録 protocol（研究者の決定 D1〜D6 を反映、2026-11-25）

- 機械可読な正本は `expfam/results/k_selection/k_exp2_generalization_20261125/protocol/protocol.json` である。その SHA-256 と、凍結時の commit は `execution_manifest.md` に記録する。この文書は同じ内容を人が読める形にしたもの。
- 2026-11-25 の最初の版（DRAFT、未凍結）は commit `0450981` にある。本版は研究者の決定で選択肢を確定したもので、§0 の推奨と異なる点はそのまま決定に従った。

## 0. 決定（研究者、2026-11-25）

| # | 決定 |
|---|---|
| D1 | **lineage E の scalar 版**（`DualExpFamLSMConsistent`、`numerics_mode='consistent'`）。修論の正式な実装への昇格ではない。結果は追加検証として管理する |
| D2 | **canonical-clean-v1**（`data_generator_canonical.generate_canonical_data`。正規化なし、clip なし、全 X 列で同じ family） |
| D3 | **(c) 2 種類とも行う。** regime a（方向ごとの信号を固定）が PRIMARY、regime b（全体の信号を固定）が事前登録した感度分析 |
| D4 | Series B（Poisson-Y）：regime a は w = 0.12、regime b は w_K = 0.12√(3/K)。w0(K) = (K/2) log(1−w²) で母集団平均を 1 に固定する |
| D5 | 元論文型の生成器の追加 fit は行わない。Series O を分布の組合せについての対応条件とする。**元論文 Exp.2 の厳密な再現とは呼ばない** |
| D6 | 4 系列 × 5 K_TRUE × 10 候補 K × 10 データ × 2 初期値 × 2 regime = **8,000 fit**。最大 8 並列 |

## 1. 研究問い

指数型分布族に一般化した正式モデル（全 X 列で共通の family）で、潜在次元 K_TRUE が増え、信号の強さが変わったとき、C_Q による K の選択の成功・過小選択・過大選択はどう変わるか。

- 評価するのは**生成時の K（K_TRUE）との一致**である。K* との一致とは言わない（§9）。
- C_Q は Q ベースの完全データ型（ICL 型）の基準であり、**Schwarz BIC ではない**。

## 2. 生成（canonical-clean-v1）

| 項目 | Series O・A・C | Series B |
|---|---|---|
| X / Y | O: Gaussian/Bernoulli、A: Poisson/Bernoulli、C: Bernoulli/Gaussian | Gaussian/Poisson |
| w（regime a） | 1.0 | 0.12 |
| w（regime b） | √(3/K) | 0.12√(3/K) |
| w0 | Bernoulli-Y：母集団の平均辺確率 0.20 になるよう較正（§3）。Gaussian-Y：0 | (K/2) log(1−w²) |
| F | F = s·Q。Q は d×K の列直交行列（ガウス行列の QR、diag(R) の符号で向きを固定）。regime a：s = √(d/6)（平均行ノルム² = 0.5K/3）。regime b：s = √(0.5d/K)（平均行ノルム² = 0.5） | 同じ |
| ノイズ | Gaussian-X：条件付き分散 0.1。Gaussian-Y：条件付き分散 1.0（SD 1.0） | Gaussian-X：分散 0.1 |
| Z | iid N(0, I_K)、正規化なし | 同じ |

- **regime 間の共有**：data seed は regime に依存しないので、同じ（系列、K_TRUE、反復）では Z と Q が regime a・b で同一になる（preflight で 400 データ全部を確認）。X・Y の残りの乱数は parameter によって消費量が変わりうるので、共有は保証しない。**regime a・b を独立なデータとして数えない。**
- **信号の分散をそろえても、観測分布や情報量がそろうわけではない。**

### Poisson-Y（Series B）の数式の確認

- E[exp(w z_i·z_j)] = E_{z_j}[exp(w²‖z_j‖²/2)] = (1−w²)^{−K/2}（|w| < 1）。
- したがって、E[λ] = 1 ⇔ w0 = (K/2) log(1−w²)。
- 数値の確認：解析式と、χ²_K についての数値積分（独立な経路）の両方で、全 10 条件で 1 との差が 10⁻⁹ 未満（test）。
- 最大の |w| は regime b の K=1 で 0.208 なので、**全条件で |w| < 1/4**。canonical Poisson-Y では E[Y^r] < ∞ ⇔ r|w| < 1（P6）なので、4 次までのモーメントが有限である（test）。
- 生成後の clip と再正規化はない（canonical-clean-v1 は clip せず、危険な率では停止する）。

## 3. Bernoulli-Y の w0 の較正（生成データを見ずに決定論的に決める）

- 目標：E[σ(w0 + w S_K)] = 0.20。S_K = z_i·z_j。
- 主の方法：S_K の厳密な密度（`matched_k_true_design.density`）を QUADPACK で積分（epsabs 10⁻¹⁴、epsrel 10⁻¹²）し、brentq（xtol 10⁻¹⁴）で解く。
- 照合 1：独立な固定幅台形則（s = e^t、h = 10⁻³）で同じ根を求める。**|Δw0| ≤ 10⁻⁸ を要求。**
- 照合 2：S | z_j ~ N(0, R)、R ~ χ²_K と分解し、Gauss–Hermite（200 点）と R の適応積分で計算する（独立な経路）。**|p − 0.20| ≤ 10⁻⁸ を要求。**
- 主の方法の残差：|p − 0.20| ≤ 10⁻¹⁰。
- **どれか 1 つでも満たさなければ STOP**（再調整しない）。
- 結果（全 10 条件で合格）：

| regime | K=1 | K=3 | K=5 | K=7 | K=9 |
|---|---|---|---|---|---|
| a（w=1） | −1.59665 | −1.96727 | −2.28763 | −2.57219 | −2.83015 |
| b（w=√(3/K)） | −1.85525 | −1.96727 | −1.99901 | −2.01409 | −2.02292 |

## 4. 推定

- `em_runner.run_em_experimental`：`numerics_mode='consistent'`、`failure_policy='fail_fast'`、L = 10、10 反復、`compute_strict_Q=True`、`mstep_q_diagnostic=True`（各反復の M-step 前後の Q を記録する。推定経路は変えない）。
- fit に渡すのは観測された X・Y だけで、真値は渡さない（test で確認）。初期化は観測データからの informed init（`initialize_params(true_params=None)`）。
- seed：
  - data_seed = 20261125000 + 100000·系列番号 + 1000·K_TRUE + 反復番号。regime・候補 K・初期値によらない。
  - init_seed = 10·data_seed + t、mc_seed = 10·data_seed + 5 + t（t = 0, 1）。候補 K と regime によらない。
- BLAS のスレッド数は 1、worker は 8。

## 5. 基準と選択規則

- **PRIMARY**：C_Q(K) = −2 Q_strict + p̂ ln n。p̂ = Kd − K(K−1)/2 + d·[G-X] + [G-Y]。
  - データごと・候補 K ごとに、2 つの初期値の C_Q の**非加重平均** C̄_Q(K) を計算し、それが最小の K を選ぶ。
  - 差が 10⁻⁹ 以内の同点は、小さい K を選ぶ。
- **SECONDARY**：
  - 初期値ごとの選択と、2 初期値の一致率
  - C̄_Q の 1 位と 2 位の差（margin）
  - 混同行列
  - 正解・過小・過大の数
  - 基準の曲線
  - regime a・b の対応比較
- **C_Lap はこの実行に含めない。** 保存した状態（θ̂ と Z）を使い、MCEM なしで評価する別の事前登録解析とする。Gaussian-Y の Series C は C_Lap が未対応なので対象外。

## 6. 失敗、中断、再開

- fail_fast とする。retry、seed の変更、置き換えの fit、反復の追加はしない。
- 次の fit は FAILED として 1 回だけ記録し、再実行しない：例外、非有限の Q_strict・C_Q・θ̂、Q/BIC の計算失敗、retry または置換が 0 でない fit。
- primary の選択に必要な 20 fit のどれかが FAILED か欠けていれば、そのデータは INCOMPLETE とする。補完しない。**分母は 10 のまま**で、INCOMPLETE の数を併記する。
- 初期値ごとの選択では、その初期値の 10 fit がそろっている場合だけ選択する。
- STOP 条件：
  - 生成器か設計規則の停止（GeneratorStop、DesignStop）
  - コードの hash の不一致
  - execute 時の working tree が clean でない
  - 出力先が既にあるのに `--resume` がない
  - 凍結した較正表との不一致
- **再開の規則**：記録された行のない fit_id だけを、凍結した seed で実行する。各 fit は seed の決定論的な関数なので、新しい乱数試行は増えない。FAILED は再実行しない。再開のたびに `run_events.jsonl` に記録する。

## 7. 集計と出力

- 統計の単位は独立に生成したデータ（系列 × regime × K_TRUE × 反復）。初期値と regime を独立な反復として数えない。
- セルごとの正解数 / 10 に、Clopper–Pearson の正確な 95% 信頼区間を付ける。
- 出力：
  - `selection_per_dataset.csv`
  - `summary_by_cell.csv`
  - `confusion.csv`
  - `paired_regimes.csv`
  - `convergence_by_cell.csv`（最後の 2 反復の M-step 後の Q の相対変化。MC 誤差を含む記述統計で、ゲートにしない）
  - `criterion_curves_median.csv`
  - `failures_and_missing.csv`
  - `status.json`
  - 図 3 枚（混同行列、正解・過小・過大、基準の曲線＝Fig.3 に対応する図）
- 解析スクリプト `tools/research_audit/analyze_k_exp2_generalization.py` は、本実験の前に commit し、偽データだけで test した。

## 8. 計算予算

- 技術 smoke（main と別の seed）の実行時間から見積もる（`execution_manifest.md`）。
- 8 worker での wall 時間の見積もりが **45 時間**（36 時間の 25% 増し）を超える場合、またはメモリが不足する場合は、本実験を開始せずに報告する。

## 9. K_TRUE と K*

| 系列 | K* = K_TRUE が言えるか | 根拠 |
|---|---|---|
| A（Poisson-X） | 言える（今回の構成で条件を満たす） | P1：clip なしの exp link、rank F = K は構成で保証される |
| C（Gaussian-Y） | 言える（今回の構成で条件を満たす） | P2：w ≠ 0、Z iid N(0, I)。Gaussian-Y の敵対的監査（`docs/untracked-reports-preservation-20261124` branch）でも反例なし。独立検証は既存の §17 まで |
| O・B（Gaussian-X） | **未証明** | Σ は未知の対角なので、因子分析の識別が必要。Anderson–Rubin の十分条件 d ≥ 2K+1 は d=15 で K ≤ 7 まで。**K_TRUE = 9 はこの十分条件を満たさない**（Ledermann 限界 K ≤ 10 の内側ではある）。Bernoulli-Y・Poisson-Y からの識別は未解決（U2・U5） |

## 10. 結果を見る前に決めた主張の限界（結果報告に必ず書く）

- データは 10 個なので、成功率の推定精度に限界がある（10/10 でも 95% CI の下限は 0.69）。
- K=9 では候補の上限が 10 なので、過大選択は 10 にしかなりえない。
- Gaussian-X では、K が大きいと識別条件に注意が必要（§9）。
- MCEM の有限反復（10 回）の影響と基準の性質は、完全には分けられない。
- regime a・b は信号の設計の違いを調べるもので、情報量をそろえるものではない。
- 系列の間の差を、family だけの因果効果と断定しない（F・w・w0 の規則も系列で違う）。
- 元論文 Fig.3 の厳密な再現ではない。
- C_Q は Schwarz BIC ではない。
- lineage E の正式採用は Human Gate を通っていない。

## 11. 結果を見た後に変えないこと

family、K_TRUE、候補 K、seed、反復数、L、w・w0・F の規則、ノイズ、基準、選択規則、同点規則、失敗処理、R、T。
