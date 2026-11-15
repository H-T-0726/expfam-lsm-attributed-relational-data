# 指導教員との打ち合わせ資料（2026-11-16、監査後の改訂版）

短時間で研究の状況を説明するための資料である。

- 詳細な数値と根拠は、同じフォルダの `evidence_matrix.md`、`af1_af8_timeline.md`、`mathematical_open_questions.md`、`thesis_direction_comparison.md` にある。
- リンクはすべて commit 固定の permalink である。
- **方向性（案 A・B・C）は未決定である。** 案 B は私の研究上の推奨で、承認ではない。

---

## 1. 研究目的

属性付き関係データの潜在構造モデル（Mikawa et al., 2024）を指数型分布族へ一般化し、**その挙動・有効な条件・限界を評価する**。一般化が有効であることは前提にしない（[`RESEARCH_MASTER.md` §1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)）。

## 2. 先行研究との差

| | 先行研究 | 本研究 |
|---|---|---|
| Y（関係）の分布 | Bernoulli に固定 | Gaussian / Bernoulli / Poisson から選ぶ |
| X（属性）の分布 | Gaussian に固定 | Gaussian / Bernoulli / Poisson から選ぶ（**全列で共通の 1 種類**） |
| E-step | Laplace 近似 | 分散関数 A''(η)/φ を用いて一般化 |
| K の選択 | 情報量規準 | Q ベースの基準（**Schwarz BIC ではない**）と、Laplace 型の C_Lap（prototype） |
| 理論 | — | 識別可能性の命題（population、条件つき） |

## 3. 提案モデルの数式

```
z_i  ~ N(0, I_K)
y_ij ~ ExpFam_Y( η_ij^Y = w_0 + w z_iᵀ z_j )     i < j（w_0、w はスカラー）
x_il ~ ExpFam_X( η_il^X = f_lᵀ z_i )             （切片なし）

A_i = I_K + Fᵀ V_X(m_i) F + w² Σ_{j≠i} V_Y(η_ij^Y) z_j z_jᵀ,     V(η) = A''(η)/φ
```

- φ は、Bernoulli と Poisson で 1、Gaussian で σ²。
- 採用式では、Σ_{j≠i} に extra 1/2 を付けない。

## 4. 正式モデルと prototype の違い

| | 正式モデル | prototype（本文採用不可。昇格は先生と相談） |
|---|---|---|
| 実装 | `DualExpFamLSM`（lineage B）、`DualExpFamLSMFixed`（lineage C） | lineage D・E・F、PR #129 の generic model（COM-Poisson） |
| X の分布 | 全列で共通の 1 種類 | 列ごとに変えられる。自動選択も含む |
| 使った実験 | 学会予稿（B）、fixed_official と実データ（C） | M1、E1、PR #168、per-column、K 選択（Phase 9 以降）、AF1〜AF8 |

**分布の自動選択は、列ごとの family を前提にしている。そのため、すべて prototype の結果である。**

## 5. 得られた主な成果

**本文に書ける結果（ALLOWED。系列を明記する）**

| 結果 | 系列 | 限定 |
|---|---|---|
| X と Y を独立に指定できる実装 | B | 実装レベルの一般化で、性能の主張ではない。Y 側に 0.5 が残る（§6） |
| 先行研究と同じ条件での一致（RMSE(Z) の差 < 0.001、5 試行） | B | 差 < 0.001 は k_est = 3・4 のときで、ほかの K では最大 0.024。比べた両方が 0.5 を含む系統。実装の同等性の確認で、採用式の検証ではない |
| n を増やすと RMSE(Z) が下がる（n=50→300 で −41〜−59%） | C | 明示した生成設定の中だけ |
| 分布の誤指定で RMSE(Z) が悪化する（最悪 4.3〜40.4 倍） | C | 最大の 9.0 倍と 40.4 倍は X と Y の両側を誤指定したセル。Y 側だけでは最大 7.9 倍と 19.6 倍、X 側だけでは最大 4.3 倍 |

**理論（population、条件つき。[§17.3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)）**
- canonical Poisson-X では、FFᵀ がモーメントから決まる（K^rank）。
- canonical Gaussian-Y で w ≠ 0 なら、**1 つの dyad の母集団の周辺分布**から K が決まる。1 つの観測値から推定できるという意味ではない。
- 実験で使っている **Bernoulli-Y での K の識別は未解決**。どの基準も一致性は未証明。

**先生の質問「間違った分布を仮定すると何が起きるか」への応答**（lineage E。本文に使うかは未決定）
- 真の Bernoulli 属性を Poisson で fit すると、Z の推定誤差は dense で +0.020（15/20）、sparse で +0.066（17/20）増えた。
  - sparse の悪化は、別の seed 系列でも再現した（[M1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/advisor_feedback/misspecification_m1_bernoulli_to_poisson_20261013.md)、[E1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/research_improvement/e1_z_point_estimator_m1_20261104.md)）。
- 誤指定の種類によって、Y の情報量との関係の向きが変わる（[PR #168](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/pull/168)）。
  - Bernoulli→Poisson は、Y が少ないほど悪化が大きい。
  - Bernoulli→Gaussian は、Y が多いほうが大きい。

**属性と関係の補完**（prototype、人工データ）
- sparse-Y で属性が補完的な設定では、複数の属性を同時に使うと Z の推定が改善した。dense-Y では改善は小さい。

## 6. 実装と理論の既知の不一致

1. **1/2 係数（KI-001）**
   - 原論文の印刷式には 1/2 がある。本研究の採用式は、独立な導出に基づいて 1/2 を付けない。
   - 学会予稿の実装（B）は、Y 側の項だけに 0.5 が残っている。そのため「Newton 方向が全体として正しいとは断定できない」と付記する。
   - fixed 系列（C）以降は 1/2 なしで、採用式と一致する。
   - B と C の条件ごとの対応表はない（[§15](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)）。
2. **η の clip**
   - 旧系列（B・C・D・F）では、X 側・Y 側・目的関数に η の clip がある。そのため、極端な η では目的関数・score・曲率が一致しない。
   - lineage E で修正した。過去の結果は変えていない。
3. **K の基準の名前**：コードと CSV の `BIC` は、Q ベースの完全データ型（ICL 型）の基準で、Schwarz BIC ではない（KI-010）。
4. **列ごとの family**：標準モデルにはない。prototype だけにある。

## 7. 分布の自動選択（AF1〜AF8）の結果と限界

| 段階 | 結果 |
|---|---|
| 選択は正解したか | 0/1 の列（Bernoulli か Poisson か）：解析的にほぼ決まり、全列正解した。<br>Poisson か CMP(ν=2) か：Q による自動選択は、dense 0.90、sparse 0.77 で正解した |
| Z の推定は改善したか | 自動選択は、count 列をすべて Poisson とする既定より、Z の推定誤差を小さく改善した（dense 0.005、sparse 0.006。joint mode では CI は 0 を含まない。ほかの Z 推定量では sparse の CI は 0 を含む。1 条件）。<br>完全な誤指定よりは大きく改善した。Oracle には届かなかった |
| Q を上回る基準を作れたか | Laplace 型の基準は、作ったデータでは改善した。新しい 40 データの**事前登録確認では未達成**（+0.225 [−0.075, 0.525]）。<br>loading を積分した基準は、数値的には安定したが、改善は判定境界の移動で prior に依存した（**NO-GO**） |
| 何が分かったか | Q による選択は、Z を推定したときに仮定した分布の側に引き寄せられる（同じ 20 データの統制比較）。<br>近い分布どうしでは、誤りの向きに偏りがある。Z が潜在であることがそれに関わるという説明は、解釈・仮説の段階 |

**限界**
- 生成条件は 1 つ（n=75、K=3）。prototype の結果である。
- 改善した基準の Z への効果は評価していない。
- 機構の説明は仮説である。

## 8. 方向性 3 案の比較（未決定）

| 案 | 内容 | 長所 | 主な懸念 |
|---|---|---|---|
| A | 拡張を中心に据え、分布は既知とする | 目的と一致し、ALLOWED の結果で書ける | 「分布が未知のとき」に答えていない |
| **B**（推奨） | 拡張を中心とし、自動選択の可能性と限界を追加分析にする | 目的の「限界の評価」と一致する。先生の質問に 2 段階で答えられる。新しい実験は不要 | prototype の結果をどこまで本文に入れるか。否定的な結果の書き方 |
| C | Z の推定と分布選択の相互依存を中心課題にする | 新しさはありうる | 理論も確認もこれから。主題が変わる。期間内の実現性が低い |

## 9. 先生に判断してほしいこと

1. **方向性**：案 A・B・C のどれが修士研究として適切か。
2. **prototype の扱い**：lineage E の結果（M1、E1、#168、per-column、AF、K の選択基準 C_Lap）を本文に使ってよいか。付録や予備的検討にとどめるべきか。
3. **1/2 係数**：学会予稿の数値（lineage B）と fixed 系列（C）の数値を、論文でどう使い分けるか。
4. **自動選択の結果の位置づけ**：「Q による選択は多くの列で正解し、Z を小さく改善したが、それを上回る基準は確認できなかった」を、1 章として書くか、限界の節に圧縮するか。
5. **COM-Poisson**：未 merge の PR #129 の generic model を論文で扱ってよいか。
6. **残りの期間**：新しい実験はせずに執筆に移るか。追加で検証すべき点があるか（例：Bernoulli-Y の K、誤指定の別条件）。
