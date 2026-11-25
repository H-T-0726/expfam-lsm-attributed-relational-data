# K 選択実験（Exp.2 対応条件の一般化）：最終 claim ledger

- 作成日：2026-11-25。対象は本実験の結果（commit `b8f2e3a`）と、独立監査（`independent_audit.md`）。
- ラベル：FACT（一次資料で確認）／DERIVED（数式・コードからの導出）／OBSERVED（実験結果）／INTERPRETATION（解釈）／UNRESOLVED（未解決）。
- **共通の限定**（以下のすべての主張に付ける）：
  - lineage E の scalar 版（prototype。Human Gate 未通過）
  - 基準は C_Q（Q ベースの完全データ型、ICL 型。Schwarz BIC ではない）
  - n = 150、d = 15、L = 10、MCEM 10 反復、canonical-clean-v1
  - 一致の対象は生成時の K（K_TRUE）。K* ではない
  - 各セル 10 データ

## 1. 命題ごとの監査

### A：方向ごとの信号を固定した設計では、検証した 4 系列と K_TRUE ∈ {1,3,5,7,9} のすべてで、生成時の K と一致した

| 判定 | **支持される（OBSERVED）** |
|---|---|
| 根拠 | `analysis/summary_by_cell.csv` の regime a の 20 セル、`independent_audit/independent_cell_counts.csv`。両方で 200/200 |
| 選択規則 | 2 初期値の C_Q の非加重平均の argmin（`protocol.json` の analysis_plan.primary_selection）。同点規則は一度も使われていない |
| 精度 | 各セル 10/10、Clopper–Pearson 95% CI [0.69, 1.00] |
| 付ける限定 | ①「すべて」はセルごとに 10 データでの観測であり、成功率 1 を示したのではない。②O・B（Gaussian-X）では K* = K_TRUE が未証明（`preregistered_protocol.md` §9）なので「生成時の K と一致」と書く。③regime a では K とともに全体の信号が増える（F の平均行ノルム² = 0.5K/3、O・A・C の w²K = K） |

### B：全体の信号を固定した設計では、A-b-7・A-b-9・B-b-9 の 3 セルで過小選択が生じた

| 判定 | **支持される（OBSERVED）** |
|---|---|
| 根拠 | `summary_by_cell.csv`：A-b-7 は 1/10 正解・9 過小、A-b-9 は 0/10・10 過小、B-b-9 は 5/10・5 過小。独立集計でも同じ |
| 補足 | regime b の他の 17 セルは 10/10。過大選択は 0。誤りはこの 3 セルの 24 データだけで、すべて過小選択 |
| 付ける限定 | ①CI：A-b-7 [0.003, 0.445]、A-b-9 [0, 0.309]、B-b-9 [0.187, 0.813]。②regime b は事前登録した感度分析であり、PRIMARY ではない |

### C：信号の配分が K 選択に関係することは示されたが、family 固有の効果や推定不足とは分離できていない

| 部分 | 判定 |
|---|---|
| 「信号の設計（regime）を変えると選択が変わった」 | **OBSERVED**。同じ Z・Q を共有した対応比較（`paired_regimes.csv`）で、a だけ正解が 24 データ、b だけ正解が 0 データ。regime は設計で操作した要因なので、この生成器のもとでは regime の切り替えが選択を変えたと言える |
| 「信号の**配分**が原因」 | **INTERPRETATION（限定付き）**。regime a・b は、1 方向あたりの信号と全体の信号、X 側と Y 側、Bernoulli-Y の w0 の較正値を同時に変えている。どの成分が効いたかは分けられない |
| family 固有の効果との分離 | **UNRESOLVED**。系列の間で、ノイズの大きさ（Gaussian-X は分散 0.1、Poisson は分散 ≈ 平均）や、w・w0 の規則（B の Y は w = 0.12√(3/K)）も違う |
| 推定不足（MCEM 10 反復）との分離 | **UNRESOLVED**。B-b-9 では、K = 9 の fit が最後の 3 反復でまだ 1 反復あたり約 100 改善しており、K < 9 より約 67 速い（`results_report.md` §6、結果を見た後の探索的な記述）。A-b ではこの差が約 1 で小さい |
| 文言の修正案 | 「信号の配分が K 選択に**関係する**ことが示された」は、「**信号の設計（regime）の切り替えが、K_TRUE ≥ 7 の一部の条件で選択を変えた**」とする方が根拠に忠実である |

### D：C_Q に含まれる潜在変数の事前密度の項が強い次元罰則として働くが、それが個々の過小選択の原因であるとはまだ断定できない

| 部分 | 判定 |
|---|---|
| 項の値 | **DERIVED（コードで確認）**。`calc_Q_dual_strict_exp`（`eval_utils.py` L.208–209）の ln p(Z) は、−(nK/2) ln(2π var_z) − Σ‖z‖²/(2 var_z) の L 個のサンプルについての平均である。`scale_Z`（`reproduction/src/model.py` L.468–505）は L 個のサンプル全体で平方平均を 1 にそろえ、var_z = 1 である。したがって、この平均はデータによらず −nK(1 + ln 2π)/2 になる。C_Q では **+nK(1 + ln 2π) = 425.68·K**（n = 150）である |
| 「強い罰則」 | **DERIVED**。K → K+1 の増分は 425.68 で、パラメータの罰則の増分 (d − K) ln n ≤ 70.2（K ≥ 1）の 6 倍以上である。また n に比例して増えるので、log n に比例する Schwarz 型の罰則より強い |
| 過大選択 0 との関係 | **INTERPRETATION**。K_TRUE を超える K での C̄_Q の増分の中央値は、1 段あたり約 270〜430 であり（`criterion_curves_median.csv`）、この項と整合する |
| 過小選択の原因か | **UNRESOLVED**。この項は小さい K を有利にするので、過小選択に寄与しうる。しかし、同じ項がある O-b-9・C-b-9 は 10/10 だった。またこの項は K だけで決まり、データの違いを説明しない。個々の過小選択の原因とは断定できない |
| 付ける限定 | この項は Q ベースの基準の構造であり、観測データの周辺尤度（Schwarz BIC が近似する量）には現れない（KI-010）。「罰則」と呼ぶのは、K だけで決まり、K とともに増えるという性質を指す比喩である |

### E：今回の結果は lineage E の prototype であり、正式な修論本文への採用は Human Gate を通っていない

| 判定 | **FACT** |
|---|---|
| 根拠 | `protocol.json` の lineage：「E scalar (DualExpFamLSMConsistent, numerics_mode=consistent); not adopted for manuscript」。`AGENTS.md` §3（prototype を manuscript evidence にしない）と §6（昇格は Human Gate）。`implementation_audit.md` §5（MANUSCRIPT_ADOPTION_STATUS: NOT_DECIDED_BY_HUMAN）。`EXPERIMENT_REGISTRY.md` の追記行で「原稿採用 ✗」 |
| 補足 | 未マージの作業ブランチ上の結果でもある（`AGENTS.md` §3） |

## 2. 修論で使える主張（Human Gate を通った場合。共通の限定を必ず付ける）

| # | 主張 | 種類 | 根拠 |
|---|---|---|---|
| T1 | 方向ごとの信号を固定した設計では、4 つの family の組合せと K_TRUE = 1〜9 のすべてのセルで、10 データすべてについて C_Q が生成時の K を選んだ | OBSERVED | 命題 A |
| T2 | 全体の信号を固定した設計では、Poisson-X / Bernoulli-Y の K_TRUE = 7・9 と Gaussian-X / Poisson-Y の K_TRUE = 9 で過小選択が生じ、他の 17 セルは 10/10 だった | OBSERVED | 命題 B |
| T3 | 過大選択は 400 データ（異なる条件は 360）で 0 件だった | OBSERVED | `independent_audit.md` |
| T4 | 信号の設計の切り替えは、同じ潜在変数を共有した対応比較で K_TRUE ≥ 7 の一部の条件の選択を変えた（a だけ正解 24、逆は 0） | OBSERVED | 命題 C |
| T5 | C_Q の ln p(Z) 項は 1 次元あたり n(1 + ln 2π) の、K だけで決まる増分を持つ | DERIVED | 命題 D |
| T6 | 誤りはすべて過小選択であり、K_TRUE が大きく、1 方向あたりの信号が弱い条件に限られた | OBSERVED | 命題 B・C |

## 3. まだ研究上の限界として扱う問題（本文には限界・今後の課題として書く）

| # | 問題 | 状態 | 次に必要なもの |
|---|---|---|---|
| U1 | B-b-9 の過小選択が MCEM の 10 反復での打ち切りによるか | UNRESOLVED | 収束感度実験（`convergence_sensitivity_design.md`。設計・実装・技術検証まで済み、本実験は未実行） |
| U2 | A-b の過小選択の原因（基準の性質か、Poisson-X の情報量か、局所解か） | UNRESOLVED | 収束感度実験の A-b-9 対照で、打ち切りの影響を否定できるかが分かる。それ以上は別の設計が要る |
| U3 | family 固有の効果 | UNRESOLVED（今回の設計では分けられない） | 系列の間で信号・ノイズの規則をそろえた設計 |
| U4 | Gaussian-X 系列で K* = K_TRUE | 未証明 | 因子分析の識別条件の検討（K_TRUE = 9 は Anderson–Rubin の十分条件を満たさない） |
| U5 | C_Q の一致性 | 未証明（KI-010） | 理論 |
| U6 | 元論文 Fig.3 との数値比較 | 不可（手順が特定できない） | — |
| U7 | 1 つの n・d・生成規則での結果であること | 限界 | n・d を変えた実験 |
| U8 | lineage E の本文採用 | Human Gate | 研究者・指導教員の判断 |
| U9 | C_Lap による選択 | 未評価 | 保存した NPZ を使う別の事前登録解析（Series C は対象外） |

## 4. 書いてはいけないこと

- 「元論文 Exp.2・Fig.3 を再現した」「Schwarz BIC で選んだ」「C_Q は一致性を持つ」。
- 「一般化モデルなら K を正しく選べる」（条件を付けない一般化）。
- 「K* を選んだ」（O・B）。
- 「系列の間の差は family の効果である」。
- 「過小選択は ln p(Z) 項が原因である」「過小選択は MCEM の未収束が原因である」（どちらも未検証）。
- 「regime b で 176/200」を、K_TRUE = 3 の重複を断らずに独立な証拠として数えること。
