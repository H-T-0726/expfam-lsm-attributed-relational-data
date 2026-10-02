# 研究全体の残る evidence gap（family selection と K 選択）（2026-10-02）

- 位置づけ: **SYNTHESIS / AUDIT / ZERO-EXPERIMENT**。新しい EM 0、dataset 0、コード変更 0。
- 入力: `reports/family_selection/` の 4 本（本日付）、`reports/k_selection_theory/phase9_k_selection_synthesis_20260929.md`（以下「9Z」）、
  `RESEARCH_MASTER.md`、`KNOWN_ISSUES.md`。
- 系列: family selection と Phase 9 の K 選択はすべて lineage E（prototype、本文採用は未決定）。
- 本書は**判断材料**であり、どの実験を行うかは決めない（CLAUDE.md §6：結果に基づく次の実験の決定は Human Gate）。

---

## 1. K 選択（Phase 9）の gap map

### 1.1 完了している要素（9Z に基づく FACT）

| 要素 | phase | 主な結果（9Z の該当節） | 現行 claim に対する役割 |
|---|---|---|---|
| baseline | 9K（20 dataset） | C_Lap K3 19/20、C_Q 14/20（§6） | 固定条件での特徴づけの中心 |
| Y sensitivity | 9P / 9S / 9T | w を弱めると K ≤ 2 が増え（特に C_Q）、強めると K=3 に集まる。平均 edge 確率を揃えても残る（§7） | 「Y 信号に依存する」の根拠 |
| X sensitivity | 9U / 9V | loading energy を上げると C_Q の K3 が 11→15→18。C_Lap はほぼ K3、strong で過大 1、不完全 1（§8） | 「X 信号に依存する」の根拠 |
| K_true sensitivity | 9W / 9X | 過小選択が K_true とともに増える（C_Q 10/9/7/1、C_Lap 10/10/10/8）（§9） | 「大きい K_true で過小選択」の根拠 |
| K3→K4 decomposition | 9Y | 1e-10 以内で再構成、余裕が正：C_Lap 8/10、C_Q 1/10（§10） | 境界の算術的記述 |
| Laplace approximation diagnostic | 9H / 9J | 評価関数は有限差分と ~1e-9 で一致。IS の重みは強く退化し、厳密な周辺尤度は得られていない（§5） | 実装の正しさは十分、近似誤差は UNRESOLVED |
| θ stationarity | 9M / 9N / 9O / 9Q / 9R | θ̂ は ℓ_Lap の停留点でない。1 ステップでは順序保持 20/20、多ステップは inconclusive（§5） | 局所的な順序の安定性まで |

**本監査からの追加（FACT、既存 artifact に限定）**: 本監査で集計した commit 済みの Phase 9 artifact では、0/1 に見える真 Poisson 列は一度も起きず、実現した family の割り当ては生成時の family と一致した。
したがって **報告済みの Phase 9 の K 選択の結果は、それらの artifact では、観測された family 誤割り当てによって交絡していない**
（「oracle family assignment」と呼ぶ場合も、実現した既存 dataset に限る）。
割り当て規則は observed support の決定論的関数だが、真 Poisson の全観測が {0,1} になる非常に稀な事象（1 列あたり ≤ 1.0×10⁻¹⁰、監査 §4）では
Bernoulli に割り当てられるので、**今後のすべての canonical draw で真の family と一致することの保証ではない**。

### 1.2 追加候補の判定

「CURRENT CLAIM」= 9Z §14 の候補表現（固定した人工データ条件 n = 75, d = 12, G3/B6/P3, Y = Bernoulli での有限標本の特徴づけ）。

| 追加候補 | CURRENT CLAIM に必要 | claim を強めるのに必要 | どの claim を守る／強めるためか |
|---|---|---|---|
| n sensitivity | **NO** | **YES** | 「C_Lap と C_Q の違いは n = 75 に固有ではない」「n とともに過小選択が減るか」（Z9-U3 の経験的側面）。Phase 9 以前の clean true-K n-sweep は別基準・別条件で、C_Lap には使えない |
| d sensitivity | NO | YES | F のパラメータ数 Kd − K(K−1)/2 と X の情報量が d とともに変わる。「P_θ と fit gain の釣り合い（9Y）」が d に依存するかを確かめないと、境界の記述は d = 12 に限られる（Z9-U4） |
| L sensitivity | NO | YES（C_Q 側） | C_Q は L = 5 の MC 平均。「C_Q の過小選択は MC ノイズではない」を言うには必要。C_Lap は joint mode なので直接の依存は θ̂ 経由のみ。9E と 9K の seed による変動（17/20 vs 14/20）の一部が L 由来かも分かっていない |
| EM iteration sensitivity | NO | **YES（優先度高）** | θ̂ は ℓ_Lap の停留点でない（9M）、多ステップは inconclusive（9N / 9R）。「C_Lap の選択は 8 反復の打ち切りの産物ではない」を守るには最も直接的（Z9-U2） |
| family composition | NO | YES | 現行 claim は G3/B6/P3 に限定。**本監査の NUMERICAL**: no-intercept・‖f‖² = 0.5 で列あたりの曲率 E[A''] は Poisson 1.284、Bernoulli 0.225、Gaussian（σ² = 1）1。X の情報は構成に強く依存するので、「混合型属性一般」に広げるなら必要。family は support で決まるので、構成を変えても family 選択の交絡は入らない |
| Y = Poisson | NO | YES | 「指数型分布族への一般化のもとでの K 選択基準」と言うなら Y 側 family を変える必要。canonical Poisson-Y は分散の有限性に \|w\| < 1/2 が必要（generator の moment 条件）で、Y 信号の設計が Bernoulli と別になる |
| model misspecification | NO | YES（実データの前提） | 実データの K 選択を主張する前に、生成モデルと fit モデルがずれたとき（過分散、intercept の欠如、family の誤指定）の C_Lap / C_Q の振る舞いを知る必要。KI-018 と直結 |
| real data | NO | YES（ただし設計が先） | 真の K がない状況での評価方法が未定（Z9-U6）。設計なしに実行しても claim にならない |

**K-selection evidence: SUFFICIENT FOR CURRENT CLAIM**（固定条件の有限標本の特徴づけとして）。
claim を強める場合の優先順位（INTERPRETATION）: EM iteration sensitivity → family composition / d → n → L → Y = Poisson → misspecification → real data。

---

## 2. 研究全体の分類

### ALREADY SUFFICIENT（現行の限定つき claim に対して）

| 項目 | 根拠 |
|---|---|
| 現行 C_Q は Schwarz BIC ではなく Q-based complete-data / ICL-type | KI-010、9Z §2 |
| C_Lap の定義・実装・評価関数の正しさ | 9Z §3, §5（9H） |
| Phase 9 の固定条件での K 選択の特徴づけ（baseline、Y / X / K_true 感度、K3→K4 の算術） | 9Z §6–§10、§15 |
| canonical clean generator（Z 非正規化、Poisson link 非打ち切り、seed rescue なし） | `reports/identifiability/canonical_clean_generator_spec_20260904.md` |
| **family selection の限定 claim**（現行 3 family・support gate・尤度 score では割り当ては support で決まり、0/1 列は構造的に Bernoulli） | **本日付の監査 §5（DERIVED）**、coverage map |
| 1/2 係数の系統の区別 | CLAUDE.md §2、KI-001 |

### NEEDS MORE EVIDENCE FOR STRONGER CLAIM

| 強い claim | 足りないもの |
|---|---|
| C_Lap は C_Q より true K を選びやすい（条件を広げて） | n / d / composition / Y family の感度（§1.2） |
| C_Lap の選択は有限反復の MCEM に頑健 | EM iteration sensitivity（Z9-U2） |
| 大きい K_true での過小選択の原因 | 9Y の fit gain の分解（Z9-U1）。現状は算術的記述まで |
| **列ごとの自動 family 選択が「判別」できる** | 現行 score では原理的に不可能（監査 §5）。score の変更（D4）か、intercept 下での下流コスト評価（D3）が前提 |
| per-column heterogeneous-X の有用性 | 既存 KNOWN_ISSUES の禁止句のとおり。prototype のまま |

### UNRESOLVED THEORY

| 問い | 出典 |
|---|---|
| 実装の ℓ_Lap と厳密な周辺尤度の差 | Z9-U7（9J で IS が退化） |
| K 選択の一致性・漸近的な振る舞い | Z9-U3 |
| w の率、θ の罰則の N = n の正当化の残り | 9L |
| Bernoulli-X（d > 1）の識別可能性 | RESEARCH_MASTER U1 |
| Bernoulli-Y の一般的な識別可能性（実験で使っている family）、Poisson-Y の識別可能性 | RESEARCH_MASTER U2、U4 |
| C_Lap（Laplace 近似）で family を比べたときの順序（尤度・厳密周辺尤度では Bernoulli 優位が証明済み） | 監査 §5.3 |
| X intercept を入れたときの識別性（Bernoulli 列単独で b, s が決まらない、P7 が崩れる）と K 構造（P3）への影響 | `x_intercept_research_decision_memo_20261002.md` §1 D |
| P_Pois(X ≤ 1) の ‖f‖ についての単調性（数値では確認） | 監査 §3.2（軽微） |
| 0.5 係数系列の Newton 方向の正しさ | KI-001（断定できないことを付記） |

### REAL-DATA DESIGN NEEDED

| 問い | 設計で決める必要があること |
|---|---|
| 真の K が分からない実データでの K 選択の妥当性 | 評価の基準（held-out、安定性、外部ラベル等）を事前に固定（Z9-U6） |
| raw-count Poisson-X の悪化の原因（KI-018） | intercept / raw scale / Poisson 曲率 / 過分散の 4 要因を分離できる条件。X intercept と dispersion-aware count family（#28 候補 B）が前提 |
| 実データ属性の family 指定 | 現行 gate は 0/1 属性をすべて Bernoulli、2 以上を含む count を Poisson にする。**低 rate count 属性や偏った binary 属性は no-intercept model の族に入らない**（Poisson 平均 ≥ 1、Bernoulli 率 = 1/2）。実データで family を論じるなら intercept の判断（HG-5）が先 |
| 実データでの support の外挿 | 学習データで 0/1 だった属性が新しい対象で 2 以上をとると、Bernoulli は確率 0 を与える。予測評価の設計で扱いを決める必要 |

---

## 3. 最も価値の高い次の研究判断（INTERPRETATION）

**family selection を修論でどう位置づけるか（Option A / B / C）**。理由:

1. 監査 §5 により、現行 prototype の family selection は support で決まり、score は何も判断していないことが確定した。
   これ以上「正解率」を測る実験は、どの Option でも情報を生まない。
2. B（intercept 追加）は family selection の問題を解かない（0/1 列は intercept があっても常に Bernoulli）。
   B を選ぶ理由はモデル拡張（低 rate / 偏った属性、KI-018 の分離）であり、その場合 Phase 9 相当の検証のやり直しが必要。
3. A / C は追加コスト 0 で、「既存 Phase 9 artifact では観測された family 誤割り当てによる交絡がない」という整理をそのまま使える（今後の draw への保証ではない）。

**DECISION: HUMAN DECISION REQUIRED**（A / B / C）。**recommended next experiment: NONE YET**。

---

## NOTION_UPDATE_RECOMMENDATIONS

Notion は今回変更していない。以下は提案のみ。

| 何を追加 | どのページ | なぜ |
|---|---|---|
| 「分布の自動選択について」の短い節：現行 3 family と support gate では、family の割り当ては観測 support で決まり、0/1 の列は尤度比較で構造的に Bernoulli になる（`log p_Pois = log p_Bern + log P_Pois(X≤1)`）。pilot の 36/36 は data から family を判別した能力ではなく、この理論的順序と実装が既存 C2 artifact で一致したことの確認（一般的な family 判別精度 100% は支持されない） | 研究状況ページ（原稿 `reports/notion/research_status_notion_20260906.md`、branch `docs/notion-k-selection-story-20260906`、未 merge）の「8. ここまでで答えられたこと」または「9. まだ切り分けられていないこと」 | 36/36 が「自動選択が機能した」と読まれるのを防ぐ。先生との議論で「難しいケースは試したか」に答えられる |
| 既存 Phase 9 artifact では 0/1 に見える真 Poisson 列は起きず、実現した family の割り当ては生成時の family と一致したので、報告済みの K 選択の結果は観測された family 誤割り当てによって交絡していない（今後の draw への保証ではない）、という 1 文 | 同ページの K 選択の節（Phase 9 の結果を説明する箇所） | K 選択の結果の解釈を補強する（肯定的な追加） |
| 「次に先生に相談したいこと」に Option A / B / C（family selection の位置づけと X intercept） | 同ページ「10. 先生に相談したいこと」 | 次の研究判断は Human（指導教員を含む）の判断事項であるため |
| （任意）現行モデルでは Poisson 列の平均 rate が 1 以上、Bernoulli 列の 1 率が 1/2 に固定されるという限定 | ゼミ用 Notion 原稿（`docs/presentation/seminar_notion_full.md` など、「データ型に応じて分布を選択できる」の箇所の注記） | 「データ型に応じて選択できる」は人手指定の意味で正しいが、低 rate count・偏った binary 属性はモデルの族に入らないことを併記すると誤解を防げる |
| per-column family の Notion 要約の「自動選択する手続きはない」（2026-07-11 時点の記述）の後継情報 | `reports/per_column_family/notion_per_column_family_summary_20260711.md` に対応する Notion ページ | その後 prototype（#74）ができ、本監査で support-determinism が判明した。過去ページは書き換えず、新しい注記として追加する |
