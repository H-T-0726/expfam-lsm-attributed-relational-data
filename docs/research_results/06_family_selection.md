# 06 分布の自動選択（AF1–AF8、PR #129）

[← README](README.md)

> **注意**：本章はすべて拡張モデル（系列 E-3、列ごとの family）と、COM-Poisson を含む generic model（PR #129、未 merge）の prototype です。AF1–AF8 はいずれも **`main` に未統合**（commit `f71c04d`。これは `docs/thesis-adoption-20261124` の先端 `8a66894` の祖先で、`git diff f71c04d 8a66894 -- expfam/results/auto_family_z_recovery` に差分はない（2026-11-27 確認）。したがって AF の結果ファイルはどちらの commit でも同じ）。本文採用は未決定です。

## 1. この実験で知りたかったこと

- 属性の分布をデータから自動で選べるか。
- 自動選択は、誤指定による $Z$ の推定の悪化を防げるか。
- どの選択基準が使えるか（Q スコア、Laplace、loading を積分した基準）。

## 2. 実験前に分かっていたこと

- M1（[04 章](04_distribution_misspecification.md)）で、Bernoulli→Poisson の誤指定が sparse-Y で $Z$ の推定を悪化させることが分かっていた。
- 先行研究のレビュー（#136）で、混合 family や family 選択の関連研究があることが分かっていた。

## 3. 実験条件

| 段階 | 内容 | データ | 種類 |
|---|---|---|---|
| AF1 | {0,1} 列で Bernoulli vs Poisson | M1 のデータ（20 replicate × dense/sparse） | 事前登録・confirmatory |
| AF2（Part A を含む） | Poisson vs CMP2（COM-Poisson、ν=2） | 20 データ、1 条件 | 事前登録 |
| AF3 | Q スコアの偏りの統制比較 | AF2 と同じ 20 データ | 探索 |
| AF4 | 割当て全体の Laplace | 同上 | 探索 |
| AF5 | 列ごとの Laplace 選択 | 同上（**手法を作ったのと同じデータ**） | 探索 |
| AF6 | AF5 の確認 | **新しい 40 データ** | 事前登録・confirmatory |
| AF7 | Poisson→CMP2 誤選択の原因分解 | AF6 のデータ | 事後解析 |
| AF8 / AF8-v2 | loading を積分した基準の pilot | — | 停止規則で停止（本実験なし） |
| AF8-v3 | 数値を認証した積分型基準（E2） | AF6 のデータを再利用 | 探索 |
| PR #129 | count / positive / bounded の候補から選ぶ | A/B/C の 3 条件 | 1 回の実行（22 fit 失敗） |

## 4. 比較した方法

- **Oracle**：真の family を指定。
- **Auto**：データから選んだ family。
- **Wrong**：誤った family を指定。
- **Support-only**：値の範囲だけで決める（count 列はすべて Poisson）。

## 5. 数式と評価指標

- {0,1} の値 $x$ について、同じ $\eta$ での対数尤度の差は
  $\log p_{\rm Bern}(x\mid\eta)-\log p_{\rm Pois}(x\mid\eta)=e^\eta-\log(1+e^\eta)>0$。
  したがって {0,1} 列では Bernoulli が常に勝つ（DERIVED、[AF1 README §0](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af1_20261109/README.md)）。
- 積分型の基準は、loading の事前分布（測度）に依存し、非入れ子の 2 候補では平坦な極限でも $O(1)$ の差が残る（DERIVED、AF8 `mathematical_derivation.md` §1）。
- 評価：列ごとの選択の正解（率）、RMSE(Z) の対差、balanced 正解率。

## 6. 結果の表

| 段階 | 主な結果 | 確認 |
|---|---|---|
| AF1 | Auto の列選択の誤りは 0（160 fit）。Wrong の悪化（sparse 0.0816）を回避 | VERIFIED（E-01：誤り 0、160 fit） |
| AF2 | Q による正解：dense 108/120（0.900）、sparse 92/120（0.767）。誤りはほぼ CMP2→Poisson。Auto − Support-only は小さい改善（主指標の joint mode だけ CI が 0 を含まない） | 報告書の値（本監査では未再計算） |
| AF3 | Q スコアは「$Z$ を作った分布」に引き寄せられる（Oracle の $Z$ で 60/60、Support-only の $Z$ で 0.533） | 報告書の値 |
| AF5 | 列ごとの Laplace は sparse で Q を +0.40 [0.05, 0.70] 上回った（同じデータで設計・評価） | 報告書の値 |
| **AF6** | **sparse の A−B は +0.225 [−0.075, 0.525]。事前登録した主要評価項目は未達成**（success False） | VERIFIED（E-02） |
| AF8-v3 | balanced 0.915、E2 − AF6 Laplace は +0.023 [−0.004, +0.050]。prior の尺度で 0.68〜0.915 と変わる | 報告書の値 |
| PR #129 | 真の $Z$ で COUNT の正解 0.984 | 報告書の値 |

## 7. 結果の図

本章の数値は、AF1・AF6 以外は本監査で再計算していないため、図は作っていない（「数値を検証できた実験だけを可視化する」方針）。

## 8. その結果から分かること

- DERIVED + OBSERVED：{0,1} 列の Bernoulli vs Poisson は、解析的にほぼ決まり、自動選択は誤指定の悪化を避けた。
- OBSERVED：近い分布（Poisson vs CMP2）でも 8 割前後は選べたが、誤りは一方向（CMP2→Poisson）に偏った。
- **OBSERVED（否定的結果）：Laplace 型の選択が Q より良いという探索段階の結果は、新しいデータでの事前登録した確認（AF6）で達成されなかった。**
- DERIVED：積分型の基準は prior に依存する。AF8-v3 は NO-GO と判断された。

## 9. 分からないこと

- 自動選択によって $Z$ の推定が一般に改善するか（AF2 の Support-only に対する改善は 1 条件・小さな差で、新しいデータでは再評価していない）。
- AF6 の未達成は「効果がない」ことの証明ではない（検出力は約 0.30）。
- Bernoulli と Poisson を区別できるのは {0,1} 列だからで、一般の count 列では限界がある。
- 実データでの自動選択は未検証。

## 10. 公平性・科学的妥当性

| 比較 | 評価 | 理由 |
|---|---|---|
| AF1 の Auto / Oracle / Wrong | **A** | 同一データ。ただし結果は解析的にほぼ決まり、Auto と Oracle は refit seed を共有（[FC-09](data/comparison_fairness_matrix.csv)） |
| AF5 の Laplace vs Q | **C** | 手法を設計したのと同じデータで評価（FC-10） |
| AF6 の Laplace vs Q | **A** | 新しいデータ・事前登録（FC-11） |
| AF8-v3 vs AF6 | **C** | データ再利用・forking paths（FC-12） |
| support が違う分布の尤度比較 | **B** | 正規化定数・support の違いを前提として明示する必要がある（FC-13） |

## 11. 関連する他の実験

- 誤指定の影響（M1・E1）：[04 章](04_distribution_misspecification.md)
- Phase 9C・9D（固定 $K$ と同時選択）、PR #129：台帳 E-9C、E-9D、E-129
- 総括：[`evidence_matrix.md`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/8a66894/reports/research_synthesis_20261116/evidence_matrix.md)（未 merge）

## 12. 一次データとリンク（すべて未 merge）

- [AF1](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af1_20261109)
- [AF2](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af2_20261110)
- [AF6 `statistical_analysis.csv`](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/blob/f71c04d/expfam/results/auto_family_z_recovery/af6_laplace_confirmation_20261114/statistical_analysis.csv)
- [AF8-v3](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/f71c04d/expfam/results/auto_family_z_recovery/af8v3_full_exploratory)
- [PR #129 の結果](https://github.com/H-T-0726/expfam-lsm-attributed-relational-data/tree/4fa6159/expfam/results/family_selection/expanded_multisupport_20261006_034826)
