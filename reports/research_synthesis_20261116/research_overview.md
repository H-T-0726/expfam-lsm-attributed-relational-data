# 研究全体の概要と目的（2026-11-16 時点の総括）

- **位置づけ**：AF1〜AF8-v3（分布の自動選択の探索的研究）を終えた時点で、修士研究全体を整理した資料。指導教員と方向性を相談するための土台として作った。
- **作成の条件**：新しい実験・MCEM・データ生成はしていない。既存の一次資料（CSV、runinfo、README、protocol、コード、canonical docs）を読み直しただけで、既存の成果物は書き換えていない。
- **ラベル**：**FACT**（一次資料で確認）、**DERIVED**（数式から導いたもの）、**OBSERVED**（実験の観測）、**INTERPRETATION**（解釈）、**HYPOTHESIS**（未検証の仮説）、**JUDGMENT**（研究上の判断。結論ではなく提案）。
- **論文の結論はここでは決めない。** 本文に採用するか（Human Gate）は研究者と指導教員が決める。

リンクは、すべて commit 固定の GitHub permalink である。

| 対象 | commit | 備考 |
|---|---|---|
| `main` と AF1〜AF8-v3 | [`f71c04d`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d) | AF 系の branch は main に merge されていない。この commit が main の子孫で、AF をすべて含む |
| PR #168 | [`856d475`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/856d475) | |
| PR #129 | [`4fa6159`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/4fa6159) | |

[R]: https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d

---

## 1. 修士研究の本来の目的（FACT）

出典：[`RESEARCH_MASTER.md` §1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)

> Gaussian / Bernoulli / Poisson を対象として、属性情報付き関係データの潜在構造モデル（Latent Structural Model）を指数型分布族へ一般化する。
> 先行研究が Y=Bernoulli・X=Gaussian に固定していた分布仮定を、X 側・Y 側で独立に指定できる形へ拡張し、
> 人工データおよび実データを用いて、その挙動・有効となる条件・限界を評価する。
> 一般化が有効であることは前提としない。

- 2026-05 の学会予稿の段階では「一般化が有効であることを示す」と書いていた。
- 2026-08 以降の証拠で、有効性は条件に依存することが分かった。そのため、現在の目的文は「挙動・条件・限界の評価」になっている（§1 の Historical note）。

## 2. 先行研究と提案モデル（FACT）

**先行研究**：Mikawa et al. (2024), NOLTA IEICE 15(2)。
- Y（関係）は Bernoulli、X（属性）は Gaussian に固定している。
- 推定は MCEM と Laplace 近似で行い、情報量規準で潜在次元 K を選ぶ。

**提案モデル（Dual-ExpFam LSM）**：[`AGENTS.md` §1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/AGENTS.md)、[`RESEARCH_MASTER.md` §3–§6](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)

```
z_i  ~ N(0, I_K)
y_ij ~ ExpFam_Y( η_ij^Y = w_0^Y + w^Y z_iᵀ z_j )     i < j
x_il ~ ExpFam_X( η_il^X = f_lᵀ z_i )                 （バイアスなし）
```

- w_0^Y と w^Y はスカラーである。
- θ = {F, w_0^Y, w^Y} に、Gaussian-X のときは Σ_X（対角）、Gaussian-Y のときは σ_Y² を加える。
- 標準の `DualExpFamLSM` では、**`family_x` は全 X 列で共通の 1 種類**である。列ごとに family を変えられるのは prototype（lineage F・E）だけで、本文には採用できない。

**E-step の精度行列と勾配（採用式。Σ_{j≠i} に extra 1/2 はない）**：

```
V_Y(η) = A_Y''(η) / φ_Y          φ_Y = 1（Bernoulli / Poisson）、σ_Y²（Gaussian）
A_i    = I_K + Fᵀ V_X(m_i) F + (w^Y)² Σ_{j≠i} V_Y(η_ij^Y) z_j z_jᵀ
∇_{z_i}（Y 側の項） = w^Y Σ_{j≠i} [T_Y(y_ij) − A_Y'(η_ij^Y)] / φ_Y · z_j
```

- V_X は、Gaussian-X なら Σ_X⁻¹、Bernoulli / Poisson-X なら diag(A_X''(F m_i)) である。
- **1/2 係数（KI-001）**：
  - 原論文の印刷式には 1/2 が**ある**（2026-08-18 に一次確認）。本研究の採用式は、独立な導出に基づいて 1/2 を付けない。
  - 学会予稿の実験は 0.5 が残る実装（lineage B）で実行している。したがって「Newton 方向が全体として正しいとは断定できない」を必ず付記する。
    - 0.5 が掛かるのは Y 側の項だけで、Z の事前分布の項と X 側の項には掛かっていない（[KI-001](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/KNOWN_ISSUES.md)、[§6.1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)）。
  - **系列ごとの 1/2**：

    | 系列 | 1/2 |
    |---|---|
    | 原論文の印刷式（A） | あり |
    | old 0.5 Python（B）と `reproduction/src/model.py` | あり |
    | 採用式 | なし |
    | fixed Python（C）とその派生（D・E・F） | なし |
    | MATLAB `calcAi` | なし（補助比較） |

**実装と理論の既知の不一致（FACT、要約）**：
1. **1/2 係数**：lineage B には Y 側の 0.5 が残っている（上記）。
2. **η の clip**：legacy 系列（B と、それを継承する C・D・F）では、η の clip と曲率の下限がある。対象は X 側の平均・曲率（例：`model_dual_expfam.py` L.102、L.113、L.115）だけではなく、Y 側（`model_expfam.py` L.57、L.71、L.73）と strict-Q の目的関数（`model_dual_expfam.py` L.325–330 など）にも及ぶ。そのため、極端な η では目的関数・score・曲率が一致しない。PC-001・PC-002 として測ったのは per-column（F）のクラスで、B と C については同じコードからの推論である。lineage E で X 側と Y 側の両方を forward に修正し、過去の結果は変えていない（[`objective_consistency_fix_20260821.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/per_column_family/objective_consistency_fix_20260821.md)）。
3. **列ごとの family**：標準モデルでは `family_x` は全列で共通である。列ごとの family は prototype（E・F）だけ。
4. **基準の名前**：`calc_bic_dual` と CSV の列名 `BIC` は、Schwarz BIC ではない（下記）。

**K の選択基準の呼び方**：
- `calc_bic_dual` は −2·Q_strict + p ln n である。これは Q-based complete-data / ICL 型の基準で、**Schwarz BIC ではない**（KI-010）。
- 関数名と CSV の列名 `BIC` は、provenance を保つために変えていない。

## 3. 実装の系列（FACT、混ぜない）

出典：[`RESEARCH_MASTER.md` §12.1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)

| 記号 | 系列 | 実体 | 位置づけ |
|---|---|---|---|
| A | 原論文の印刷式 | Mikawa et al. 2024 | 1/2 あり |
| B | old 0.5 Python | `model_dual_expfam.py` | 学会予稿の本文採用実験 |
| C | fixed Python | `model_dual_expfam_fixed.py` | 実データ実験フェーズ、fixed_official |
| D | experimental | `experimental/model_dual_expfam_masked.py` / `_nb.py` | 本文採用不可 |
| E | objective-consistent | `experimental/model_dual_expfam_consistent.py` | Phase 6 以降、M1、E1、AF。本文採用不可 |
| F | per-column prototype | `experimental/model_dual_expfam_percolumn.py` | 列ごとの family。本文採用不可 |

AF1〜AF8 はすべて lineage E の prototype である。AF2 以降は、未 merge の PR #129 の generic model（COM-Poisson を含む）も使っている。

## 4. 現時点で確認できていること（要約、詳細は `evidence_matrix.md`）

### 4.1 本来の研究（指数型分布族への拡張）

**ALLOWED**（[`RESEARCH_MASTER.md` §14](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)）：
- **実装（lineage B）**：X 側と Y 側を、Gaussian / Bernoulli / Poisson から独立に指定できる実装がある。
  - 実装レベルの一般化で、性能の主張ではない。
  - Y 側には 0.5 が残っているので、KI-001 の限定を付ける。
- **先行研究の再現（lineage B）**：先行研究と同じ条件で、再現実装と一致した（RMSE(Z) の差 < 0.001、5 試行）。
  - 差が < 0.001 なのは k_est = 3（0.00054）と 4 だけで、ほかの K では最大 0.024（k_est = 6、`comparison_main_table.csv`）。
  - `model_dual_expfam.py` と `reproduction/src/model.py` は、どちらも 0.5 を含む同じ系統である。
  - したがってこの一致は「実装が先行研究の再現実装と同じ挙動をする」ことの確認で、**採用式（1/2 なし）の検証ではない**。
- **n への依存（fixed 系列 C、1/2 なし）**：人工データで n を増やすと RMSE(Z) が下がった（n=50 → 300 で A −49.3%、B −41.2%、C −58.6%）。
  - 旧 0.5 系列の値（49%、31%、62%）と同じ表に置かない（KI-002）。
- **誤指定（fixed 系列 C）**：分布を誤指定すると RMSE(Z) が悪化した。最悪の倍率は A 4.34×、B 9.04×、C 40.37×。
  - **倍率の内訳（OBSERVED、[誤指定の統合 §5.2](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/reports/advisor_feedback/distribution_misspecification_synthesis_20261013.md)）**：
    - 最大の B 9.04× と C 40.37× は、X と Y の両側を誤指定したセルの値。
    - Y 側だけの誤指定では最大 7.9×（B）と 19.6×（C）。
    - X 側だけでは最大 4.34×（A、support-invalid）。
    - C の 40.37× は Y 側だけの値の約 2 倍なので、X と Y の組み合わせの影響もある。
  - 旧 0.5 系列の 41.45× などと条件が違うので、差を「0.5 除去の効果」と読まない（KI-003、§15）。

**理論**（[§17.3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/RESEARCH_MASTER.md)、すべて population、つまり観測分布そのものについての主張。前提の詳細は `mathematical_open_questions.md` §1.1）：
- **P1**：canonical で clip のない Poisson-X では、FFᵀ を population モーメントから復元できる。
  - 対象は X 周辺の K^rank で、K* ではない。前提は rank F = K（generic）。
- **P2**：canonical Gaussian-Y で w ≠ 0 なら、**1 つの dyad の周辺分布**（母集団の分布）から K、w²、σ_y² が決まる。
  - **1 つの観測値から K を推定できるという意味ではない。**
  - 有限個の、互いに独立でない dyad から推定するときの性質は、別の問題である。
- **P3**：canonical Gaussian-Y で、w ≠ 0 の部分は M_{K+1} と交わらない。
- **P6**：Poisson-Y のモーメントの存在条件。
- **未解決**：実験で使っている **Bernoulli-Y の K の識別は未解決**（U2）。どの基準についても、一致性は証明していない（U6）。

**QUALIFIED ONLY（限定を付けて書く）**：
- 属性と関係の補完：sparse-Y で属性 block が補完的な人工設定では、per-column の同時利用が RMSE_Z を改善した。prototype の結果で、dense-Y では改善が小さい。
- 実データ（MovieLens）：平均の向きは正だが、split 間の変動に比べて小さい。

**誤指定と Y の情報量**（lineage E、本文採用は未決定）：
- M1：真の Bernoulli 属性を Poisson として fit すると、Z の推定誤差が dense で +0.020（15/20）、sparse で +0.066（17/20）増えた。
- E1：別の seed 系列でも、joint mode の Z で同じ向きが再現した。
- PR #168：誤指定の種類によって、Y の情報量との交互作用の向きが変わった。

### 4.2 分布の自動選択（AF1〜AF8-v3、すべて prototype）

次の 4 つを区別する（詳細は `evidence_matrix.md` §B0）：
1. 選択が正解したか。
2. 合理的な比較手法より改善したか。
3. 新しいデータで改善が再現したか。
4. 事前登録した成功条件を達成したか。

- **{0,1} 列の Bernoulli vs Poisson（AF1）**
  - 選択は解析的にほぼ決まる（DERIVED）。
  - 全列が正解し、Wrong による Z の悪化を回避した（事前登録した主対比どおり）。
- **Poisson vs CMP(ν=2)**：データに依存する統計的な判断になる。
  - **正解率**：Q による自動選択の正解率は、AF2 で dense 0.900、sparse 0.767 だった。新しい 40 データ（AF6、同じ anchor の Q）では 0.946 と 0.813 で、近い水準だった。
  - **Z の推定**：AF2 では、Auto の RMSE(Z) が Support-only（count 列をすべて Poisson）より小さかった（dense 0.0051 [0.0023, 0.0079]、sparse 0.0059 [0.0013, 0.0109]）。
    - CI は 0 を含まないが、差は小さい。
    - **これは主指標（joint mode）だけの結果である。** 副指標の last と mean5 では、sparse の CI は 0 を含む（last 0.0054 [−0.0048, 0.0162]、mean5 0.0063 [−0.0016, 0.0149]）。
    - sparse で 2 列以上を誤選択したデータでは、Support-only より良くならなかった（AF2 README の事後表）。
    - 事前登録の副対比で、成功の閾値は protocol で定めていない。新しいデータでの Z の再評価はしていない。
    - PR #129 の別の設計でも、同じ向きだった。
  - Oracle には届かなかった（sparse で 0.0116 劣る）。
  - Q スコアは、Z を作るのに使った分布の側に引き寄せられた（AF3）。
  - Laplace 型の基準では、誤りの向きが CMP2→Poisson から Poisson→CMP2 に入れ替わった（AF5〜AF6）。
  - Laplace 型の基準は、手法を作ったデータでは Q より良かった（AF5、sparse +0.40）。しかし新しいデータで事前登録した確認では、主要評価項目を達成しなかった（AF6：+0.225 [−0.075, 0.525]）。
  - loading を積分した基準（AF8-v3）は数値的には安定して評価できた。しかし改善は判定境界の移動で、prior の尺度に強く依存した（NO-GO）。
- **Z の推定が良くなるかは、AF5〜AF8 では評価していない。** Z の評価は AF1・AF2 と PR #129 にある。

## 5. 資料の構成

| ファイル | 内容 |
|---|---|
| `research_overview.md` | 本書。目的・モデル・系列・確認済み事項の概要 |
| `af1_af8_timeline.md` | AF1〜AF8-v3 の経緯。各段階の問い・数式・条件・結果・否定された仮説・次へ進んだ理由 |
| `evidence_matrix.md` | 主張と一次資料の対応表。「再現済み」「独立に確認されていない」「事後解析」を区別する |
| `mathematical_open_questions.md` | 未解決の数理的な課題と、実装上の制限 |
| `thesis_direction_comparison.md` | 修士研究の方向性 3 案の比較 |
| `advisor_meeting_brief.md` | 指導教員との打ち合わせ用の資料 |
