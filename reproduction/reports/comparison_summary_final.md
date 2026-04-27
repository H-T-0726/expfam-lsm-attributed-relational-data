# Baseline vs Dual-ExpFam LSM: Comparison Summary (Final)
# 作成日: 2026-04-24

実験設定: N_TRIALS=5, L=5, NUM_ITER=8, n=150, d=15, k*=3, k_range=[1..6]
データ生成: generate_dual_data (expfam/src/data_generator_expfam.py)
Baseline: reproduction/src/model.py (Mikawa 2024, 改変なし)
Dual-ExpFam: expfam/src/model_dual_expfam.py

---

## 1. ファイル確認結果

| ファイル | 状態 | 行数 |
|---------|------|------|
| reproduction/reports/baseline_audit.md | 存在・完全 | 145 |
| reproduction/src/experiment_compare_with_dual.py | 存在・完全 | 495 |
| reproduction/scripts/run_comparison_all.py | 存在・完全 | 303 |
| reproduction/results/comparison/comparison_control_exp1.csv | 存在・完全 | 61 (60 data rows) |
| reproduction/results/comparison/comparison_scen_a_exp1.csv | 存在・完全 | 61 (60 data rows) |
| reproduction/results/comparison/not_applicable_log.csv | 存在・完全 | 3 (2 data rows) |
| reproduction/reports/comparison_summary.md | 存在・完全 | 144 |

全ファイル存在確認済み。2 つのバックグラウンドシェルは両方 exit code 0 で正常終了。
出力ファイルは最終版（16:47-16:50 に書き込み完了）。

---

## 2. 比較可能性の分類

| シナリオ | Y がバイナリか | X が Gaussian か | Baseline 適用 | 比較種別 |
|---------|--------------|-----------------|--------------|---------|
| Control: Gauss-X x Bern-Y | Yes | Yes | 完全適用可 | **主比較** |
| Scenario A: Pois-X x Bern-Y | Yes | No (Poisson) | Y のみ適用可、X はミススペック | 補助比較 |
| Scenario B: Gauss-X x Pois-Y | No (Poisson Y) | Yes | **実行不能** | 比較不能 |
| Scenario C: Bern-X x Gauss-Y | No (Gaussian Y) | No | **実行不能** | 比較不能 |

---

## 表1: 主比較 — Control (Gauss-X x Bern-Y)

### 1a. k=k*=3 での指標比較（5 trials, mean +/- std）

| Metric | Baseline | Dual-ExpFam (Gauss, Bern) | Delta |
|--------|----------|---------------------------|-------|
| RMSE(Z) | 0.1793 +/- 0.0144 | 0.1799 +/- 0.0155 | +0.00054 |
| RMSE(F) | 0.0397 +/- 0.0114 | 0.0351 +/- 0.0062 | -0.00460 |
| RMSE(Y) | 0.0881 +/- 0.0082 | 0.0877 +/- 0.0075 | -0.00035 |
| RMSE(X) | 0.3069 +/- 0.0056 | 0.3074 +/- 0.0049 | +0.00056 |
| |w0 err| | 0.0599 +/- 0.0162 | 0.0567 +/- 0.0151 | -0.00316 |
| |w err| | 0.0828 +/- 0.0397 | 0.0794 +/- 0.0385 | -0.00342 |

注: 5 試行すべてで両モデルが同一データ・同一シードを使用。
   試行 1-4 では RMSE(Z) が完全一致（差 = 0.000）、試行 0 のみ差 0.003。
   全 Delta は |0.005| 以下であり、両モデルは実質的に同等。

### 1b. BIC による k 同定（k_range=[1..6]）

| Model | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | BIC 最小 k | 正解 |
|-------|-----|-----|-----|-----|-----|-----|-----------|-----|
| Baseline | 18117 | 15054 | **11915** | 12384 | 12818 | 13291 | k=3 | Yes |
| Dual-ExpFam | 18141 | 15105 | **11949** | 12397 | 12848 | 13295 | k=3 | Yes |

両モデルとも k*=3 を正しく同定。BIC カーブの形状はほぼ一致。

### 1c. per-trial RMSE(Z) at k=3

| Trial | Baseline | Dual-ExpFam | Diff |
|-------|----------|-------------|------|
| 0 | 0.2037 | 0.2063 | +0.0027 |
| 1 | 0.1791 | 0.1791 | 0.0000 |
| 2 | 0.1739 | 0.1739 | 0.0000 |
| 3 | 0.1661 | 0.1661 | 0.0000 |
| 4 | 0.1741 | 0.1741 | 0.0000 |

---

## 表2: 補助比較 — Scenario A (Pois-X x Bern-Y)

**注意**: Baseline の X モデルはこのシナリオでミススペックされている
（Poisson データに Gaussian 仮定を適用）。
この結果は「Dual-ExpFam が一般に優れている」ことを示すものではなく、
「X の分布仮定が誤っている場合にどの程度推定が劣化するか」を示す。

### 2a. k=k*=3 での指標比較（5 trials, mean +/- std）

| Metric | Baseline (X misspecified) | Dual-ExpFam (correct) | Delta | DE better by |
|--------|--------------------------|----------------------|-------|--------------|
| RMSE(Z) | 0.6818 +/- 0.0710 | 0.2798 +/- 0.0158 | -0.4020 | 59.0% |
| RMSE(F) | 1.2904 +/- 0.2025 | 0.0695 +/- 0.0066 | -1.2209 | 94.6% |
| RMSE(Y) | 0.2066 +/- 0.0052 | 0.1250 +/- 0.0081 | -0.0816 | 39.5% |
| RMSE(X) | 1.6967 +/- 0.1118 | 1.2100 +/- 0.0208 | -0.4867 | 28.7% |
| |w0 err| | 0.1125 +/- 0.0626 | 0.0419 +/- 0.0087 | -0.0705 | 62.7% |
| |w err| | 0.4498 +/- 0.0816 | 0.1481 +/- 0.0292 | -0.3017 | 67.1% |

### 2b. BIC による k 同定（Scenario A）

| Model | BIC 最小 k | 正解 (k*=3) |
|-------|-----------|------------|
| Baseline (X misspecified) | k=4 | No |
| Dual-ExpFam (Pois-X, Bern-Y) | k=3 | Yes |

---

## 表3: 比較不能条件

| Scenario | True family_y | 理由 |
|----------|--------------|------|
| B: Gauss-X x Pois-Y | Poisson | Y が非バイナリ (0,1,2,...を取る)。Baseline は Y in {0,1} を内部で仮定しており、Poisson Y をそのまま入力するとモデルとして無効。 |
| C: Bern-X x Gauss-Y | Gaussian | Y が連続実数値。同上の理由で Baseline は適用不能。 |

---

## 研究レポート用文章

### A. 結果要約（短い版）

Baseline（Mikawa et al., 2024）と Dual-ExpFam LSM を比較した。
両モデルが同一の分布仮定を共有する制御条件（X=Gaussian, Y=Bernoulli）では、
RMSE(Z) の差は平均 0.0005 以下であり、BIC による潜在次元 k の同定精度も同一で、
両モデルは実質的に同等の推定性能を示した。
一方、真の X 分布が Poisson である条件では、Gaussian X を仮定する Baseline の RMSE(Z)
は Dual-ExpFam の約 2.4 倍となり、k の同定も失敗した。
Y が非バイナリとなる Scenarios B・C については Baseline が適用不能なため、
直接比較は行っていない。

### B. 考察（やや丁寧な版）

制御条件（X=Gaussian, Y=Bernoulli）での比較実験において、Dual-ExpFam LSM は
Baseline と実質的に同等の推定精度を達成した。
これは、Dual-ExpFam が Baseline の設定（Gaussian X, Bernoulli Y）を特殊ケースとして
包含しており、その設定での推定性能を損なっていないことを示している。
per-trial の比較では 5 試行中 4 試行で RMSE(Z) が完全に一致し、
残る 1 試行でも差は 0.003 程度であった。

Scenario A（真の X=Poisson, Y=Bernoulli）では、Baseline の Gaussian X 仮定が
ミススペックとなり、RMSE(Z) が Dual-ExpFam の約 2.4 倍、RMSE(F) が約 18 倍に悪化した。
また BIC による k 同定において Baseline は k=4（誤り）を選択したのに対し、
Dual-ExpFam は k=3（正解）を選択した。
ただし、この差の主因はアルゴリズム自体の強さではなく、
正しい分布族（Poisson X）を指定できることによる「モデル適合性の優位」である。

Scenarios B（Y=Poisson）および C（Y=Gaussian）については、
Baseline が Y in {0,1} を必須とするモデル仮定を持つため、
これらのシナリオに直接適用することができず、比較は実施していない。

### C. 論文・報告書向け慎重な結論

本実験から得られる結論は以下の通りである。
Dual-ExpFam LSM は Baseline（Mikawa et al., 2024）が想定する条件（X=Gaussian, Y=Bernoulli）
において同等の推定精度を示す。これにより、提案手法が Baseline の性能を損なうことなく
その機能を包含していることが確認された。
Baseline の仮定から外れる条件、特に属性データ X の分布仮定が誤っている場合
（Scenario A: 真の X=Poisson に対し Gaussian と仮定）には、
提案手法が潜在変数・因子行列・モデルパラメータのすべての指標で明確に優れた結果を示し、
BIC による次元選択の精度も向上した。
この優位性は「より広い分布族に対して適切にモデル化できること」に起因するものであり、
同一条件下でのアルゴリズム的優位を示すものではない。
Y が非バイナリとなる条件（Scenarios B, C）については Baseline との比較が原理的に
不可能であるため、それらの条件での Dual-ExpFam の性能は別途評価が必要である。

---

## 再実行コマンド

```bash
# 作業ディレクトリ: C:/kennkyu
python reproduction/scripts/run_comparison_all.py
# 所要時間: 約 38 分 (N_TRIALS=5, L=5, NUM_ITER=8)
```

---

## 保存ファイル一覧

| ファイル | 内容 |
|---------|------|
| reproduction/reports/baseline_audit.md | Baseline 仕様・比較可能性監査メモ |
| reproduction/src/experiment_compare_with_dual.py | Baseline ラッパー (model.py 改変なし) |
| reproduction/scripts/run_comparison_all.py | 全比較実験実行スクリプト |
| reproduction/results/comparison/comparison_control_exp1.csv | 主比較 raw データ (60 rows) |
| reproduction/results/comparison/comparison_scen_a_exp1.csv | 補助比較 raw データ (60 rows) |
| reproduction/results/comparison/not_applicable_log.csv | 実行不能ケース記録 |
| reproduction/results/comparison/comparison_main_table.csv | 主比較集計テーブル (CSV) |
| reproduction/results/comparison/comparison_aux_table.csv | 補助比較集計テーブル (CSV) |
| reproduction/results/comparison/comparison_not_applicable_table.csv | 比較不能条件テーブル (CSV) |
| reproduction/reports/comparison_summary.md | 自動生成サマリー |
| reproduction/reports/comparison_summary_final.md | 本ファイル (最終版) |
