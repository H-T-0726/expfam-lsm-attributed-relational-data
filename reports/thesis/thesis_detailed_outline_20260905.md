# 修士論文 詳細アウトライン

**作成日:** 2026-09-05
**目的:** 節・小節まで下ろし、各節に **目的 / 要点 / 数式 / コード / レポート / artifact /
図表 / 書ける主張 / 限定 / 未解決** を紐づける。**これを見れば本文を書き始められる**状態にする。

**原則:** 節ごとに「その節が支える主張が claim ledger にあるか」を確認済み。
ledger にない主張を書く節は作らない。

参照: `RESEARCH_MASTER.md` §14/§16/§17、`KNOWN_ISSUES.md`、
`thesis_storyline_20260905.md`（弱点 W1–W7）、`thesis_figure_table_inventory_20260905.md`。

---

## 第 1 章 序論

### 1.1 背景 — 属性つき関係データ
- **目的:** ノードが属性を持つネットワークデータの例と、共通潜在構造を推定する動機。
- **要点:** 属性 `X` と関係 `Y` は別々に見ると情報が足りないが、共通の `z_i` を仮定すると補完しうる。
- **限定:** 補完が有効なのは特定 regime のみ（第 7 章）。ここでは動機として述べるにとどめる。

### 1.2 問題設定と本研究の問い
- **要点:** 観測分布を指数型分布族へ拡張したとき、**潜在次元 `K` をどう選ぶか**。
- **本研究の問い（3 つ）:** ①拡張モデルは生成モデルとして成立するか ②「真の `K`」とは何か
  ③その `K` は観測分布から決まるか、有限データで選べるか。

### 1.3 本研究の貢献
- 命題 P1・P2・P3・P6・P8（第 4・5 章）、clean generator、事前登録した n-sweep。
- **書き方の注意:** 「性能が向上した」ではなく「**何が言えて何が言えないかを確定した**」型。
  （`thesis_storyline_20260905.md` §6）

### 1.4 論文の構成

---

## 第 2 章 先行研究と再現

### 2.1 潜在構造モデル
- **数式:** `z_i ~ N(0,I_K)`、`y_ij ~ Bernoulli(σ(w0 + w z_i^T z_j))`。
- **出所:** `paper/A_study_on_latent_structural_models_for_binary_rel.pdf`（`[PRIMARY_SOURCE]`）

### 2.2 Python 再現
- **コード:** `reproduction/src/model.py`
- **artifact:** `comparison_main_table.csv`
- **書ける主張:** 同条件で RMSE(Z) 差 < 0.001（**ALLOWED**）
- **限定:** **5 試行のみ**

### 2.3 元論文の K 選択基準
- **要点:** Eq.(26) の当てはまり項 Eq.(16) の `ln L` は `p(Z)` を含まず `Z` を積分しない（Q1 型）。
- **出所:** `reports/k_selection_theory/paper_bic_reproduction_alignment_20260904.md`
- **限定:** **評価手続き（どの `Z`、MC 平均か）は本文から特定不能** `[UNRESOLVED]`

---

## 第 3 章 提案モデル — 指数型分布族への拡張

### 3.1 生成モデル
- **数式:** `z_i ~ N(0,I_K)`、`x_il ~ ExpFam_X(f_l^T z_i)`、`y_ij ~ ExpFam_Y(w0 + w z_i^T z_j)`（`i<j`）
- **図:** **F3-1**（未作成、概念図）
- **限定:** `w0, w` は**スカラー**。X は per-component だが `family_x` は全列共通。
  **Categorical 未実装**（KI-005）

### 3.2 family 依存の推定
- **数式:** E-step の `V_Y(η) = A_Y''(η)/φ_Y`、`A_i = I_k + F^T V_X F + w² Σ V_Y z_j z_j^T`
- **表:** **T3-1**（未作成）
- **限定:** **0.5 係数（KI-001）。**「Newton 方向が全体として正しいとは断定できない」を必ず付記
- **出所:** root `CLAUDE.md` §1・§2、`docs/math_notes/half_factor_math_explanation.md`

### 3.3 実装 lineage
- **要点:** 6 系統が存在し、**数値を引用するときは必ず系列を明記**（KI-002）。
- **限定:** experimental / per-column / objective-consistent は **prototype・本文採用不可**

---

## 第 4 章 生成モデルの整合性

**この章は今回の新規貢献。**

### 4.1 properness
- **書ける主張:** finite `n,d,K`・finite parameters・Gaussian 分散 > 0 で proper（**ALLOWED**）
- **出所:** `true_k_identifiability_hardened_20260904.md` §3

### 4.2 properness と finite moment の分離
- **命題 P6:** `E[Y^r] < ∞ ⟺ |w| < 1/r`（平均 `|w|<1`、分散 `|w|<1/2`）
- **証明:** `E[λ^r] = e^{rw0}(1−r²w²)^{−K/2}` ＋ Stirling 数と Lyapunov 不等式
- **表:** **T4-1**（未作成）
- **書ける主張:** historical 既定値 `w=0.5` が**分散発散の境界**（**ALLOWED**、KI-020）
- **限定:** **canonical model の主張。** historical データは clip 済みかつ `Z` が z-score 済みで
  `χ²_K` が成立しないため、historical データの分布についての主張ではない。
  **「clip があるからデータが有限」は誤り**（実現値は clip なしでも a.s. 有限）

### 4.3 historical generator との乖離
- **表:** **T4-2**（既存、理論監査 §13）
- **書ける主張:** G1–G5 の具体的差分（**ALLOWED**、KI-021）
- **限定:** **過去結果を無効化しない。** 失われるのは well-specified という読み方のみ

### 4.4 canonical clean generator
- **コード:** `expfam/src/experimental/data_generator_canonical.py`（`canonical-clean-v1`）
- **テスト:** `test_data_generator_canonical.py`（46 件）
- **要点:** 正規化なし・clip なし・`rank(F)=K` を構成保証（**seed rescue 禁止**）・
  分散パラメータの意味を引数名に埋め込む・Poisson-Y は `|w|<1/2` を強制
- **限定:** **historical generator は 1 文字も変更していない**

---

## 第 5 章 真の潜在次元と識別可能性

**この章は今回の新規貢献。**

### 5.1 「真の K」の定義
- **数式:** `M_K = {∫p(Z)p(X|Z)p(Y|Z)dZ : θ∈Θ_K}`、`K* = min{K : P0 ∈ M_K}`
- **表:** **T5-2**（既存）— `K_TRUE` / `K*` / `K^rank` / `K°` の区別
- **書ける主張:** `K^rank ≤ K* ≤ K_TRUE`、**どの等号も自明でない**（**ALLOWED**）
- **限定:** **M-closed 仮定が必要。実データでは `K*` が存在しない**

### 5.2 properness と identifiability の分離
- **要点:** parameter（`O(K)` を法として）/ order / functional / local·global /
  generic·everywhere / estimability の 6 区分
- **限定:** **本モデルでは `O(K)` を法とせずに parameter identifiability は決して成立しない**

### 5.3 不変性と退化
- **数式:** `z→Qz`, `F→FQ^T`（`Q ∈ O(K)`）で観測分布不変。prior `N(0,I)` がスケールを固定
- **要点:** 退化ケース `F=0`、`w=0`、`rank(F)<K`、`d<K`
- **限定:** 「K が存在する」と「K を識別できる」は別

### 5.4 family 別の識別可能性
- **表:** **T5-1**（既存）
- **命題 P1（Poisson-X）:** `‖f_l‖²=2log E[X_l]`、`f_l·f_m=log(E[X_lX_m]/(E[X_l]E[X_m]))`
  → **X 周辺の最小次元 = `rank(FF^T)`**
- **命題 P5（Gaussian-X, Σ 既知）:** `rank(Cov(X)−Σ)`
- **反例:** Bernoulli-X `d=1`（**X 周辺のみ**）、Gaussian-X `d=2,K=1`（counting）
- **限定:** すべて **population**・**generic**。**否定的な非識別性は joint へ移送できない**

### 5.5 Gaussian-Y の識別可能性
- **命題 P2:** `M_S(t)=(1−t²)^{−K/2}` → `κ_2=K, κ_4=6K, κ_6=120K` →
  **単一 dyad から `(w0,w²,K,σ_y²)`**
- **命題 P8:** 三角形の 3 次同時中心モーメント `= w³K` → **`w` の符号が決まる**
- **図:** **F5-1**（要作成）
- **限定:** **Gaussian-Y 限定。`w ≠ 0` が必要**

### 5.6 モデル族の入れ子性
- **命題 P3:** Gaussian-Y `w≠0` で `{P ∈ M_K : w≠0} ∩ M_{K+1} = ∅`
- **限定:** **`M_K` と `M_{K+1}` 自体は `w=0` 切片で交わる。「互いに素」と書かない**

### 5.7 Bernoulli-Y — 未解決
- **要点:** `w0=0` では edge density が無情報。K の情報は共有ノード motif に入り、
  `‖z_i‖² ~ χ²_K` を通じて現れるが、sigmoid の Gaussian 平均が閉形式を持たない
- **限定:** **`[UNRESOLVED]`（U2）。しかも実験で使っている family である**
- **W1（storyline の弱点）をここで明示する**

---

## 第 6 章 分布誤指定の影響

### 6.1 実験設定
- **コード:** `expfam/src/exp_run_scenario_{A,B,C}.py`、`run_fixed_official_exp4_*`
### 6.2 結果
- **図:** **F6-1**（既存 artifact、図の有無は要確認）
- **書ける主張:** 誤指定最悪 A 4.34× / B 9.04× / C 40.37×（**ALLOWED**）
- **限定:** **fixed 系列（lineage C）。** ablation 行（`fix_w=True`）は誤指定倍率ではない。
  **旧 0.5 系列の 23.6× / 41.45× と同じ表・図に並べない**（KI-003）

---

## 第 7 章 属性と関係の情報補完

### 7.1 動機 — sparse Y
### 7.2 実験設定
- **コード:** `run_complementary_blocks_consistent.py`
### 7.3 結果
- **図:** **F7-1**（既存）
- **書ける主張:** **QUALIFIED ONLY**
- **限定:** **prototype（本文採用不可）・人工データのみ・sparse-Y regime・
  complementary な属性構造に限る。dense-Y では +0.0087 まで縮小することを必ず併記。
  `single_vs_joint` では符号が逆（−0.0004）**
- **NOT ALLOWED:** 「per-column が一般に優れる」「実データでも有効」

---

## 第 8 章 K 選択の有限標本挙動

**この章は今回の新規貢献。**

### 8.1 問い
- **要点:** **consistency theorem ではない。** 「有限の `n` を動かしたとき selected-K パターンが
  どう変わるか」の記述的観測

### 8.2 実験設計と事前登録
- **表:** **T8-1**（既存）
- **artifact:** `clean_true_k_asymptotics_20260904/`、protocol hash `547880a1…`
- **要点:** 896 fits、`K_TRUE ∈ {1,3,5}`（反復 4/4/8）× `n ∈ {50,75,100,150}` ×
  candidate K 1..7 × start {1,2}。**結果を見る前に protocol を凍結**
- **限定:** 監査 PASS は**手続きの健全性**であって科学的主張の正しさではない

### 8.3 生成データの設計不変量
- **表:** **T8-2**（既存）
- **要点:** 全 64 セルで `rank(F)=K_TRUE`、正規化なし、clip なし、
  平均 `‖f_l‖²=0.500000`、`w²K=3.000000`
- **限定:** **この信号整合自体が設計判断**

### 8.4 主要結果 — `K_TRUE = 5`
- **図:** **F8-1**（既存）/ **表:** **T8-3**（既存）
- **書ける主張（QUALIFIED ONLY）:**
  > テストした有限の `n` の範囲（50–150）では、`n` の増加にともなって平均 selected K が
  > 真値へ近づき、under-selection が減少する傾向が観測された。
- **必須の併記:** ①**真値一致数は単調でない**（S1 は n=75 で 0/8）②誤りは一貫して過小選択
  ③反復は 4 または 8 のみ ④**一致は `K_TRUE` とのものであって `K*` とのものではない**
- **NOT ALLOWED:** 「`n` を増やすと `K=5` に収束した」「一致性を示した」

### 8.5 control と criterion 比較
- **図:** **F8-2**・**F8-3**（既存）
- **要点:** S1 39/64、S2 37/64、S3 3/64。一致は S1 vs S2 44/64、三者一致 0/64
- **`K_TRUE=1` の 4/4 の扱い:** **成功例に使えない。** 候補下端＋最小 K tie rule による
  **下限効果と交絡**（理論監査 §17.4）
- **S3 の扱い:** `Z` を積分しない Q1 型は候補上限に張り付く。
  **ただし S3 は本研究の定義であり原論文 Eq.(26) ではない。
  S3 の失敗を原論文 BIC の失敗と読まない**

### 8.6 安定性診断
- **図:** **F8-4**（既存）
- **要点:** 初期値不一致は S1/`K_TRUE=5` で `n=50` 8/8 → `n=150` 1/8
- **限定:** **criterion 由来か最適化由来かは分離できていない**

### 8.7 構造診断 — Poisson-X Gram spectrum
- **図:** **F8-5**（既存）
- **要点:** 全 64 セルで非 PSD、閾値なし階数は常に `d=15`
- **限定:** **rank 閾値を設定していない。この診断は selected K を作らない**（U7）

---

## 第 9 章 K 選択基準の理論的位置づけ

### 9.1 3 つの尤度
- **表:** Q1 / Q2 / Q3 の区別（`k_selection_theory_map_20260905.md` §1）
- **書ける主張:** 現行 `calc_bic_dual` は `Q_strict`（Q2 の MC 平均）を使い **Q3 ではない**。
  **したがって Schwarz BIC ではない**（**ALLOWED**、KI-010・KI-019）

### 9.2 BIC が正当化されない理由
- **要点:** ①モデルの**特異性**（`O(K)` 不変性で Fisher 情報が退化。RLCT 未知）
  ②**境界パラメータ** ③**有効標本数が未定義**（ノード数 / dyad 数 / X 要素数。
  実装は `log n` にノード数。`Z` は `nK` 個の incidental parameter）
- **限定:** **「非入れ子だから使えない」は誤り。** 非入れ子が壊すのは LRT の χ² 近似と Wilks

### 9.3 held-out 予測スコアは何を選ぶか
- **限定:** **`[UNRESOLVED]`（U10）。** plug-in raw-eta score は proper scoring rule ではない
- **確実に言えること:** **選択（selection）≠ 識別（identification）**

---

## 第 10 章 実データへの適用

### 10.1 実データでは `K*` が存在しない
- **要点:** M-closed が成立しないので最小次元という概念自体が使えない
- **書き方:** selected K を「潜在構造の次元」と呼ばず「**予測性能を最大化する候補次元**」と書く

### 10.2 Cora
- **表:** **T9-1**（既存）
- **限定:** 基準ごとに最適 k が割れる（KI-011）。**原因は UNRESOLVED**。
  subset（n=280）から full Cora への一般化は未検証

### 10.3 MovieLens
- **図:** **F9-1**（既存）
- **限定:** **平均方向は正だが split 間変動に比べて小さい。** 有意性検定・CI・検出力は未計算。
  30 splits は独立 replicate ではない。overdispersion var/mean ≈ 10（KI-012）
- **NOT ALLOWED:** 「MovieLens で提案手法の有効性を確認した」

### 10.4 実応用での手続きと警告
- **出所:** `real_application_interpretation_20260905.md` §4
- **要点:** 実際にできるのは held-out 予測・複数初期値/基準の一致確認・margin の確認
- **警告 W-a〜W-e:** 小 `n` は過小選択 / 小さい K は下限効果かも /
  `Z` を積分しない基準は使わない / 初期値 1 つは不安定 / Poisson-Y は `|w|<1/2`
- **限定:** いずれも**今回の合成設定での観測**。実データでの成立は保証しない
- **未解決:** transductive のみ（inductive 未評価）、観測ゼロと欠測を区別していない

---

## 第 11 章 限界と今後

### 11.1 未解決事項
- **表:** **T10-1**（既存、`k_selection_theory_map_20260905.md` §9）
- U1–U12、RLCT、有効標本数の定義

### 11.2 ストーリー上つながっていない箇所
- **表:** **T10-2**（既存、`thesis_storyline_20260905.md` §2）
- **W1 を最重要として明示:** **理論の強い結果は Gaussian-Y、実験は Bernoulli-Y。
  理論が実験を正当化しているわけではない**

### 11.3 今後の実験
- ①`K_TRUE=1` の下限効果の切り分け（**新しい事前登録が必要**）
- ②start 数を増やして不安定性の原因を分離
- ③X の寄与を分離した測定
- ④Bernoulli-Y の識別可能性（理論作業）

---

## 付録

| ID | 内容 | 出所 |
|---|---|---|
| A | 命題 P1・P2・P3・P5・P6・P8 の完全証明 | `true_k_identifiability_hardened_20260904.md` |
| B | 数値検証（81 rows、**独立 41 / 構成上 40**） | `verify_identifiability_identities.py` |
| C | clean generator 仕様 | `canonical_clean_generator_spec_20260904.md` |
| D | 凍結プロトコル | `clean_true_k_experiment_protocol_20260904.md` |
| E | 独立監査の設計と 15 種の変異テスト | `audit_clean_true_k_sweep.py`, `test_clean_true_k_sweep.py` |
| F | 敵対レビュー記録 | `true_k_identifiability_review_20260904.md` |
| G | 実装 lineage 一覧（KI-002） | root `CLAUDE.md` §3 |

---

## 執筆時のチェックリスト

各節を書き終えるたびに確認する。

1. **その節の主張は claim ledger の ALLOWED か QUALIFIED ONLY にあるか。**
2. **QUALIFIED ONLY なら、必須の限定語をすべて書いたか。**
3. **数値を引用したなら、実装 lineage を明記したか**（KI-002）。
4. **`K_TRUE` との一致を `K*` の回復と書いていないか。**
5. **「Schwarz BIC」と書いていないか**（KI-010・KI-019）。
6. **有限標本の観測を一致性として書いていないか。**
7. **`K_TRUE=1` を成功例として使っていないか。**
8. **S3 の結果を原論文の基準の話にしていないか。**
9. **prototype lineage を本文採用可能なものとして扱っていないか。**
