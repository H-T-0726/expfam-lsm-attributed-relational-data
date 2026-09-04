# 修論 figure / table インベントリ

**作成日:** 2026-09-05
**目的:** 修論に必要な図表を洗い出し、**どれが既存でどれが未作成か**、
**それぞれが支えられる主張と必要な限定**を紐づける。

**原則:** 「図が綺麗だから載せる」ではなく、**その図が支える主張が claim ledger にあるか**で選ぶ。
ledger にない主張を支える図は載せない。

`既存` = 生成済み artifact がある / `要再生成` = script はあるが図がない / `未作成` = script から作る必要

---

## 章 3 — 提案モデル

### F3-1 モデル概要図

| 項目 | 内容 |
|---|---|
| 提案タイトル | Dual-ExpFam LSM の生成構造 |
| 目的 | `z_i → (x_i, y_ij)` の依存関係と、X 側・Y 側で family が独立に選べることを一目で示す |
| 出所 | **未作成**（概念図。データ不要） |
| caption 案 | 潜在ベクトル `z_i ~ N(0, I_K)` から、属性 `x_il ~ ExpFam_X(f_l^T z_i)` と関係 `y_ij ~ ExpFam_Y(w0 + w z_i^T z_j)` が条件付き独立に生成される。`w0, w` はスカラー |
| 支える主張 | 実装レベルの一般化（**ALLOWED**） |
| 限定 | 性能主張ではない。**Categorical は未実装**（KI-005） |
| 章節 | 3.1 |

### T3-1 family 別の `A'(η)` / `A''(η)` 対応表

| 項目 | 内容 |
|---|---|
| 目的 | family ごとに E-step / M-step の何が変わるかを示す |
| 出所 | **未作成**（root `CLAUDE.md` §1 と実装から手作業で表化） |
| 支える主張 | family-aware な推定が必要であること |
| 限定 | **0.5 係数（KI-001）。**「Newton 方向が全体として正しいとは断定できない」を付記 |
| 章節 | 3.2 |

---

## 章 4 — 生成モデルの整合性

### T4-1 Poisson-Y のモーメント存在条件

| 項目 | 内容 |
|---|---|
| 目的 | `E[Y^r] < ∞ ⟺ \|w\| < 1/r` と、既定値 `w=0.5` が分散発散の境界であることを示す |
| 出所 | **未作成**（`verify_identifiability_identities.py` の `poisson_y_moment_existence` 出力を表化） |
| caption 案 | canonical Poisson-Y のモーメント有限性。平均は `\|w\|<1`、分散は `\|w\|<1/2` を要する。historical 既定値 `w=0.5` は分散発散の境界そのもの |
| 支える主張 | 命題 P6・KI-020（**ALLOWED**、条件明記） |
| 限定 | **canonical model の主張。** historical データは clip 済みかつ `Z` が z-score 済みで `χ²_K` が成立しないため、この表は historical データの分布についての主張ではない |
| 章節 | 4.2 |

### T4-2 historical generator と canonical model の差分（G1–G7）

| 項目 | 内容 |
|---|---|
| 目的 | 5 つの具体的な乖離を、行番号つきで示す |
| 出所 | **既存**（`true_k_identifiability_hardened_20260904.md` §13 の表をそのまま） |
| 支える主張 | KI-021（**ALLOWED**） |
| 限定 | **「過去結果が無効」ではない。** 失われるのは well-specified という読み方のみ |
| 章節 | 4.3 |

---

## 章 5 — 真の潜在次元と識別可能性

### T5-1 family 別の識別可能性まとめ

| 項目 | 内容 |
|---|---|
| 目的 | どの family で何が証明され、何が未解決かを一覧化 |
| 出所 | **既存**（`k_selection_theory_map_20260905.md` §4 の表） |
| 支える主張 | 命題 P1・P2・P3・P5・P6・P8、反例 C1–C4 |
| 限定 | すべて **population** の主張。**Bernoulli-Y は UNRESOLVED** |
| 章節 | 5.4 |

### T5-2 `K_TRUE` / `K*` / `K^rank` / `K°` の区別

| 項目 | 内容 |
|---|---|
| 目的 | 4 つの「K」が別物であることを固定する |
| 出所 | **既存**（`k_selection_theory_map_20260905.md` §2） |
| 支える主張 | `K^rank ≤ K* ≤ K_TRUE`、どの等号も自明でない |
| 限定 | **実データでは `K*` が存在しない**（M-closed 不成立） |
| 章節 | 5.1 |

### F5-1 `S = z_i^T z_j` のキュムラントと `K` の関係

| 項目 | 内容 |
|---|---|
| 目的 | `κ_2 = K`, `κ_4 = 6K`, `κ_6 = 120K` を数値確認とともに示す |
| 出所 | **要再生成**（`verify_identifiability_identities.py` の `S_kappa_*` 出力から作図） |
| caption 案 | `M_S(t) = (1−t²)^{−K/2}` から導かれるキュムラント。K=1,3,5 で解析値と 4×10⁶ 標本の推定値が一致（相対誤差 ≤ 0.030） |
| 支える主張 | 命題 P2 の基礎（**ALLOWED**） |
| 限定 | Gaussian-Y 限定。**実験で使う Bernoulli-Y には適用できない** |
| 章節 | 5.5 |

---

## 章 6 — 分布誤指定の影響

### F6-1 誤指定倍率（fixed 系列）

| 項目 | 内容 |
|---|---|
| 目的 | 分布族の誤指定が RMSE(Z) を悪化させることを示す |
| 出所 | **既存**（`fixed_official/exp4/fixed_exp4_scen_{a,b,c}_ratios.csv`）。図の有無は要確認 |
| caption 案 | fixed 系列における分布族誤指定の RMSE(Z) 悪化倍率。最悪 A 4.34× / B 9.04× / C 40.37×（いずれも `fix_w=False`, `fix_x=False`） |
| 支える主張 | ledger の ALLOWED 行 |
| 限定 | **fixed 系列（lineage C）。** ablation 行（`fix_w=True`）は誤指定倍率ではない。**旧 0.5 系列の 23.6× / 41.45× と同じ図に並べない**（KI-003） |
| 章節 | 6.2 |

---

## 章 7 — 属性と関係の補完

### F7-1 sparse-Y での per-column 利得

| 項目 | 内容 |
|---|---|
| 出所 | **既存**（`complementary_blocks_consistent_20260821_paired.csv`） |
| caption 案 | sparse-Y complementary 設定における per-column joint モデルの RMSE_Z 改善（+0.5122 / +0.4218 / +0.3889 / +0.2030） |
| 支える主張 | **QUALIFIED ONLY** |
| 限定 | **experimental prototype（本文採用不可）・人工データのみ・sparse-Y regime・complementary な属性構造に限る。dense-Y では +0.0087 まで縮小することを必ず併記** |
| 章節 | 7.3 |

---

## 章 8 — K 選択の有限標本挙動（今回の新規）

### T8-1 実験設計と実行 integrity

| 項目 | 内容 |
|---|---|
| 目的 | 事前登録した格子と、実際に実行された内容が一致することを示す |
| 出所 | **既存**（`clean_true_k_results_20260905.md` §1・§2、`protocol.json`、`runinfo.json`） |
| caption 案 | clean true-K n-sweep の設計と実行記録。protocol hash `547880a1…`、896 fits（expected = actual = unique）、retry / replacement / seed rescue / tolerance 緩和 / resume すべて 0、独立監査 PASS |
| 支える主張 | 実行の健全性（**ALLOWED**） |
| 限定 | 監査 PASS は手続きの健全性であって科学的主張の正しさではない |
| 章節 | 8.2 |

### T8-2 生成データの設計不変量

| 項目 | 内容 |
|---|---|
| 目的 | 信号強度が `K_TRUE` で交絡していないことを示す |
| 出所 | **既存**（`generator_provenance.csv`、report §2） |
| caption 案 | 全 64 セルで `rank(F) = K_TRUE`、正規化なし、clip なし、平均 `‖f_l‖² = 0.500000`、`w²K = 3.000000`、Y density 0.318–0.340 |
| 支える主張 | 交絡を排した設計（**ALLOWED**） |
| 限定 | **この整合自体が設計判断。** `f_scale` 固定の設計では別結果になりうる |
| 章節 | 8.3 |

### F8-1 `K_TRUE = 5` の n 依存（主要図）

| 項目 | 内容 |
|---|---|
| 目的 | **本研究の主結果。** `n` を増やしたときの selected K の変化 |
| 出所 | **未作成**（`selection_matrix.csv` から作図） |
| 図案 | 横軸 `n`（50/75/100/150）、縦軸 selected K。replicate ごとの点＋平均線。真値 `K=5` に水平線。S1 と S2 を別系列 |
| caption 案 | `K_TRUE = 5` における selected K。held-out 基準の真値一致は 2/8, 0/8, 4/8, 8/8、平均 selected K は 2.62, 3.00, 4.50, 5.00。**平均は単調に増加したが一致数は単調ではない**（n=75 で 0/8） |
| 支える主張 | **QUALIFIED ONLY** |
| 限定 | **有限 4 点の記述。一致性ではない。** 反復は 8 のみ。誤りは一貫して過小選択。**一致は `K_TRUE` とのものであり `K*` とのものではない** |
| 章節 | 8.4 |

### T8-3 全条件の真値一致数（S1 / S2 / S3）

| 項目 | 内容 |
|---|---|
| 出所 | **既存**（report §3 の表） |
| caption 案 | 基準別・`K_TRUE` 別・`n` 別の真値一致数。S1 合計 39/64、S2 37/64、S3 3/64 |
| 支える主張 | **QUALIFIED ONLY** |
| 限定 | **`K_TRUE=1` の 4/4 は下限効果と交絡しており成功例に使えない**（候補下端＋最小 K tie rule） |
| 章節 | 8.4 |

### F8-2 基準間の不一致

| 項目 | 内容 |
|---|---|
| 目的 | 3 基準が同じ fit から違う K を選ぶことを示す |
| 出所 | **未作成**（report §5 の表から作図） |
| 図案 | 64 セル × 3 基準のヒートマップ（selected K を色で） |
| caption 案 | 同一の fit 証拠から得た 3 基準の selected K。S1 と S2 は 64 中 44 セルで一致、S1 と S3 は 2 セル、S2 と S3 は 0 セル、三者一致は 0 セル |
| 支える主張 | **criterion-dependent K selection**（**QUALIFIED ONLY**） |
| 限定 | **どれが「正しい」かは決められない**（`K*` を確定できていないため） |
| 章節 | 8.5 |

### F8-3 S3 の過大選択

| 項目 | 内容 |
|---|---|
| 目的 | `Z` を積分しない基準が候補上限に張り付くことを示す |
| 出所 | **未作成**（`fit_results.csv` の `s3_plugin_conditional` から作図） |
| 図案 | 横軸 candidate K、縦軸 S3 の値（cell 平均）。単調減少（＝より良く見える）であることを示す |
| caption 案 | plug-in conditional 基準は候補 K の増加とともに単調に改善して見え、`p log n` の罰則が追いつかない。64 セル中 3 セルしか真値に一致せず、ほぼ全セルで候補上限 `K=7` を選んだ |
| 支える主張 | Q1 型基準への警告（**QUALIFIED ONLY**） |
| 限定 | **S3 は本研究が定義した基準であり原論文 Eq.(26) ではない。** 原論文の評価手続きは特定不能。**S3 の失敗を原論文の基準の失敗と読んではならない** |
| 章節 | 8.5 |

### F8-4 初期値不一致（安定性診断）

| 項目 | 内容 |
|---|---|
| 出所 | **未作成**（`fit_results.csv` から per-start argmax を計算） |
| 図案 | 横軸 `n`、縦軸「2 つの初期値が別の K を選んだセルの割合」 |
| caption 案 | S1 / `K_TRUE=5` において初期値が異なる K を選んだセル: `n=50` で 8/8、`n=150` で 1/8 |
| 支える主張 | 不安定性の記述（**QUALIFIED ONLY**） |
| 限定 | **criterion 由来か最適化由来かは分離できていない** |
| 章節 | 8.6 |

### F8-5 Poisson-X Gram spectrum（構造診断）

| 項目 | 内容 |
|---|---|
| 出所 | **未作成**（`gram_spectrum.csv` から作図） |
| 図案 | 横軸 固有値番号 1..15、縦軸 固有値（cell 中央値）。`K_TRUE` ごとに系列を分け、`K_TRUE` の位置に縦線 |
| caption 案 | 標本モーメントから推定した Poisson-X Gram 行列の固有値。**全 64 セルで最小固有値が負**（中央値 −1.80 〜 −0.52）であり、閾値なしの階数は常に `d = 15` |
| 支える主張 | U7 の具体化（**ALLOWED**、事実の記述） |
| 限定 | **rank 閾値を設定していない。** 結果を見てから閾値を決めることを protocol が禁じている。**この図は selected K を作らない** |
| 章節 | 8.7 |

---

## 章 9 — 実データ

### F9-1 MovieLens user-disjoint validation

| 項目 | 内容 |
|---|---|
| 出所 | **既存**（`RESEARCH_MASTER.md` §12.5 の artifact） |
| 支える主張 | **QUALIFIED ONLY** |
| 限定 | **平均方向は正だが split 間変動に比べて小さい。** 有意性検定・CI・検出力は未計算。30 splits は独立 replicate ではない |
| 章節 | 9.3 |

### T9-1 Cora の基準別最適 k

| 項目 | 内容 |
|---|---|
| 出所 | **既存**（`cora_balanced_k_sweep/*.csv`） |
| caption 案 | Cora balanced subset における評価指標別の最適 k。BIC 最小は k=1、AUC/AP 最大は k=6、NMI/ARI 最大は k=3 |
| 支える主張 | KI-011（単一指標では不十分） |
| 限定 | **原因は UNRESOLVED。** subset（n=280）が full Cora に一般化するかも未検証 |
| 章節 | 9.2 |

---

## 章 10 — 限界

### T10-1 未解決事項一覧

| 項目 | 内容 |
|---|---|
| 出所 | **既存**（`k_selection_theory_map_20260905.md` §9） |
| 支える主張 | — |
| 章節 | 10.1 |

### T10-2 storyline の弱点（W1–W7）

| 項目 | 内容 |
|---|---|
| 目的 | 繋げたくなるが証拠が繋がらない箇所を明示 |
| 出所 | **既存**（`thesis_storyline_20260905.md` §2） |
| 支える主張 | — |
| 限定 | **とくに W1（理論は Gaussian-Y、実験は Bernoulli-Y）を隠さない** |
| 章節 | 10.2 |

---

## 作成が必要なもの（優先順）

| 優先 | ID | 内容 | 元データ |
|---|---|---|---|
| **1** | **F8-1** | `K_TRUE=5` の n 依存（**主要図**） | `selection_matrix.csv` |
| **2** | **F8-2** | 基準間の不一致ヒートマップ | `selection_matrix.csv` |
| 3 | F8-3 | S3 の過大選択 | `fit_results.csv` |
| 4 | F8-5 | Gram spectrum | `gram_spectrum.csv` |
| 5 | F8-4 | 初期値不一致 | `fit_results.csv` |
| 6 | F3-1 | モデル概要図（概念図） | — |
| 7 | T3-1 / T4-1 | family 別の表 | 実装・理論監査 |
| 8 | F5-1 | キュムラント確認図 | identity checker |

**F6-1・F7-1・F9-1・T9-1 は既存 artifact があるので、
図の有無と生成 script の所在を `EXPERIMENT_REGISTRY.md` で確認してから判断する。**

**注意:** 図は必ず **script 経由で生成**する（root `CLAUDE.md` §7）。手で編集しない。
