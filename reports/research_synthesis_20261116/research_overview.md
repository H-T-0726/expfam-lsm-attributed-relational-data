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
- X 側と Y 側を、Gaussian / Bernoulli / Poisson から独立に指定できる実装がある。これは実装レベルの一般化で、性能の主張ではない。
- 先行研究と同じ条件で、再現実装と一致した（RMSE(Z) の差 < 0.001、5 試行）。
- fixed 系列の人工データで、n を増やすと RMSE(Z) が下がった（n=50 → 300 で A −49.3%、B −41.2%、C −58.6%）。
- fixed 系列で、分布を誤指定すると RMSE(Z) が悪化した。最悪の倍率は A 4.34×、B 9.04×、C 40.37×。**大きな倍率は主に Y 側の誤指定による。**

**理論**（§17、population レベル）：
- Poisson-X は FFᵀ をモーメントから復元できる（P1）。
- Gaussian-Y では 1 つの dyad から K が決まる（P2）。
- Gaussian-Y の族は入れ子ではない（P3）。
- Poisson-Y のモーメントの存在条件（P6）。
- いずれも条件つきである。**Bernoulli-Y の K の識別は未解決。**

**QUALIFIED ONLY（限定を付けて書く）**：
- 属性と関係の補完：sparse-Y で属性 block が補完的な人工設定では、per-column の同時利用が RMSE_Z を改善した。prototype の結果で、dense-Y では改善が小さい。
- 実データ（MovieLens）：平均の向きは正だが、split 間の変動に比べて小さい。

**誤指定と Y の情報量**（lineage E、本文採用は未決定）：
- M1：真の Bernoulli 属性を Poisson として fit すると、Z の推定誤差が dense で +0.020（15/20）、sparse で +0.066（17/20）増えた。
- E1：別の seed 系列でも、joint mode の Z で同じ向きが再現した。
- PR #168：誤指定の種類によって、Y の情報量との交互作用の向きが変わった。

### 4.2 分布の自動選択（AF1〜AF8-v3、すべて prototype）

- {0,1} 列で Bernoulli と Poisson を比べる場合、選択は解析的にほぼ決まる（DERIVED）。そのため、自動選択で誤指定の損失を回避できた（AF1）。
- Poisson と CMP(ν=2) の選択は、データに依存する統計的な判断である。
  - Q スコアは、Z を作るのに使った分布の側に引き寄せられた（AF3）。
  - Laplace 型の基準では、誤りの向きが CMP2→Poisson から Poisson→CMP2 に入れ替わった（AF5〜AF6）。
  - 事前登録した確認実験の主要評価項目は未達成だった（AF6）。
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
