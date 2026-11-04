# Multi-support family selection — seminar summary tables (20261006)

> **自動生成（FACT のみ）。** 数値はすべて run の CSV から生成した。解釈は別の報告書に分離している。
> 系列 **E（experimental prototype; 本文採用不可）**。controlled synthetic experiment（nuisance 固定・K=3 固定）。

- run: `expfam/results/family_selection/expanded_multisupport_20261006_034826/`
- init git SHA: `37a17ac5b13146ffc7b59f9b272549731a06d119`; phase SHAs: smoke=`8ba1b677`, pilot=`8ba1b677`, smoke=`39d04c9b`, pilot=`39d04c9b`, A_main=`39d04c9b`, B_main_r5=`39d04c9b`, C_main_r5=`11e1355c`, B_main_r10=`11e1355c`, C_main_r10=`11e1355c`
- protocol sha256: `405fc22f6743f86b785df7be44bb2a428101903652d3f5ba37cc21aa72ac376b`
- audit verdict: **PASS** (BLOCKER 0, WARNING 4)
- completed cases by phase (failed): A_main 240 (0), B_main_r10 240 (1), B_main_r5 240 (7), C_main_r10 80 (5), C_main_r5 80 (5), pilot 114 (4), smoke 38 (0)
- generated (UTC): 2026-10-05T21:08:59.580542+00:00

## 表1 実装・検証した family（数理 audit と test 結果）

| family | support group | 判定 | T(x) | 固定 hyperparameter | 実装 | test |
|---|---|---|---|---|---|---|
| gaussian | GROUP_REAL | PASS_STRICT | x | – | True | PASS(12 tests) |
| bernoulli | GROUP_COUNT | PASS_STRICT | x | – | True | PASS(15 tests) |
| poisson | GROUP_COUNT | PASS_STRICT | x | – | True | PASS(18 tests) |
| binomial | GROUP_COUNT | PASS_WITH_EXISTING_EXPFAM_GENERALIZATION | x | n_trials=2 | True | PASS(5 tests) |
| com_poisson | GROUP_COUNT | PASS_WITH_EXISTING_EXPFAM_GENERALIZATION | x | nu in {0.5, 2.0} (separate candidate models) | True | PASS(10 tests) |
| zero_truncated_poisson | GROUP_COUNT | PASS_WITH_EXISTING_EXPFAM_GENERALIZATION | x | – | True | PASS(6 tests) |
| lognormal | GROUP_POSITIVE_CONTINUOUS | PASS_WITH_EXISTING_EXPFAM_GENERALIZATION | log x | sigma=1 | True | PASS(5 tests) |
| positive_truncated_normal | GROUP_POSITIVE_CONTINUOUS | PASS_WITH_EXISTING_EXPFAM_GENERALIZATION | x | sigma=1 | True | PASS(5 tests) |
| continuous_bernoulli | GROUP_BOUNDED_CONTINUOUS | PASS_WITH_EXISTING_EXPFAM_GENERALIZATION | x | – | True | PASS(6 tests) |
| logit_normal | GROUP_BOUNDED_CONTINUOUS | PASS_WITH_EXISTING_EXPFAM_GENERALIZATION | logit x | sigma=1 | True | PASS(5 tests) |
| truncated_normal_01 | GROUP_BOUNDED_CONTINUOUS | PASS_WITH_EXISTING_EXPFAM_GENERALIZATION | x | sigma=0.25;center=0.5 | True | PASS(5 tests) |
| geometric | GROUP_COUNT | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |
| negative_binomial | GROUP_COUNT | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |
| gamma | GROUP_POSITIVE_CONTINUOUS | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |
| exponential | GROUP_POSITIVE_CONTINUOUS | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |
| beta | GROUP_BOUNDED_CONTINUOUS | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |
| weibull | GROUP_POSITIVE_CONTINUOUS | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |
| student_t | GROUP_REAL | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |
| categorical | NONE | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |
| inverse_gaussian | GROUP_POSITIVE_CONTINUOUS | BLOCKED_OUTSIDE_CURRENT_MODEL | – | – | False | NOT_APPLICABLE |

candidate model（priority 順。SUPPORT_ONLY は各群の先頭 admissible、START_FIRST/LAST は先頭/末尾）:

- GROUP_COUNT: `poisson` → `binomial_n2` → `cmp_nu2p0` → `cmp_nu0p5` → `ztp`
- GROUP_POSITIVE_CONTINUOUS: `lognormal_s1` → `ptn_s1`
- GROUP_BOUNDED_CONTINUOUS: `logitnormal_s1` → `cb` → `tn01_s0p25_c0p5`
- GROUP_REAL: `gaussian`

## 表2 support group 別 family recovery（main stage, EASY/HARD・n 込み）

score-decided = admissible 候補が 2 つ以上あり score で決まった列のみ。all = gate で決まった列と失敗も含む全列（失敗は不正解として分母に残す）。値は rate (正解/評価数)。

| support group | scope | A: selector only (true Z) | B: Auto START_FIRST | B: Auto START_LAST | C: mixed Auto START_FIRST | C: mixed Auto START_LAST |
|---|---|---|---|---|---|---|
| COUNT | score_decided_only | 0.984 (1181/1200) | 0.970 (509/525) | 0.972 (554/570) | 0.970 (320/330) | 0.976 (361/370) |
| COUNT | all_columns | 0.984 (1181/1200) | 0.848 (509/600, failed 75) | 0.923 (554/600, failed 30) | 0.800 (320/400, failed 70) | 0.902 (361/400, failed 30) |
| POSITIVE | score_decided_only | 1.000 (480/480) | 1.000 (240/240) | 0.996 (233/234) | 1.000 (132/132) | 1.000 (148/148) |
| POSITIVE | all_columns | 1.000 (480/480) | 1.000 (240/240) | 0.971 (233/240, failed 6) | 0.825 (132/160, failed 28) | 0.925 (148/160, failed 12) |
| BOUNDED | score_decided_only | 0.949 (683/720) | 0.944 (340/360) | 0.950 (342/360) | 0.960 (190/198) | 0.959 (213/222) |
| BOUNDED | all_columns | 0.949 (683/720) | 0.944 (340/360) | 0.950 (342/360) | 0.792 (190/240, failed 42) | 0.887 (213/240, failed 18) |
| REAL | score_decided_only | – | – | – | – | – |
| REAL | all_columns | – | – | – | 0.825 (66/80, failed 14) | 0.925 (74/80, failed 6) |

### 表2b true candidate model 別 recovery（A: selector only, main, score-decided, candidate-model level）

| true candidate model | candidate-model recovery |
|---|---|
| cb (GROUP_BOUNDED_CONTINUOUS) | 0.979 (235/240) |
| logitnormal_s1 (GROUP_BOUNDED_CONTINUOUS) | 0.967 (232/240) |
| tn01_s0p25_c0p5 (GROUP_BOUNDED_CONTINUOUS) | 0.900 (216/240) |
| binomial_n2 (GROUP_COUNT) | 1.000 (240/240) |
| cmp_nu0p5 (GROUP_COUNT) | 0.975 (234/240) |
| cmp_nu2p0 (GROUP_COUNT) | 0.992 (238/240) |
| poisson (GROUP_COUNT) | 0.954 (229/240) |
| ztp (GROUP_COUNT) | 1.000 (240/240) |
| lognormal_s1 (GROUP_POSITIVE_CONTINUOUS) | 1.000 (240/240) |
| ptn_s1 (GROUP_POSITIVE_CONTINUOUS) | 1.000 (240/240) |

| true family | family-level recovery (A, score-decided) |
|---|---|
| continuous_bernoulli (GROUP_BOUNDED_CONTINUOUS) | 0.979 (235/240) |
| logit_normal (GROUP_BOUNDED_CONTINUOUS) | 0.967 (232/240) |
| truncated_normal_01 (GROUP_BOUNDED_CONTINUOUS) | 0.900 (216/240) |
| binomial (GROUP_COUNT) | 1.000 (240/240) |
| com_poisson (GROUP_COUNT) | 0.983 (472/480) |
| poisson (GROUP_COUNT) | 0.954 (229/240) |
| zero_truncated_poisson (GROUP_COUNT) | 1.000 (240/240) |
| lognormal (GROUP_POSITIVE_CONTINUOUS) | 1.000 (240/240) |
| positive_truncated_normal (GROUP_POSITIVE_CONTINUOUS) | 1.000 (240/240) |

## 表3 RMSE(Z)（Procrustes 後, main stage）: Oracle / Auto / Support-only

| design | difficulty | n | Oracle mean±sd (fits, failed) | Auto START_FIRST | Auto START_LAST | Support-only |
|---|---|---|---|---|---|---|
| COUNT | EASY | 75 | 0.300±0.019 (10, 0) | 0.305±0.025 (10, 0) | 0.310±0.020 (10, 0) | 0.424±0.262 (10, 0) |
| COUNT | EASY | 150 | 0.286±0.061 (10, 0) | 0.252±0.009 (10, 5) | 0.263±0.024 (10, 2) | 0.424±0.324 (10, 0) |
| COUNT | HARD | 75 | 0.457±0.036 (10, 0) | 0.449±0.042 (10, 0) | 0.449±0.042 (10, 0) | 0.465±0.031 (10, 0) |
| COUNT | HARD | 150 | 0.324±0.014 (10, 0) | 0.324±0.014 (10, 0) | 0.324±0.014 (10, 0) | 0.321±0.021 (10, 0) |
| POSITIVE | EASY | 75 | 0.365±0.012 (10, 0) | 0.365±0.012 (10, 0) | 0.365±0.012 (10, 0) | 0.397±0.017 (10, 0) |
| POSITIVE | EASY | 150 | 0.305±0.010 (10, 0) | 0.305±0.010 (10, 0) | 0.306±0.008 (10, 1) | 0.320±0.012 (10, 0) |
| POSITIVE | HARD | 75 | 0.466±0.039 (10, 0) | 0.466±0.039 (10, 0) | 0.466±0.039 (10, 0) | 0.485±0.023 (10, 0) |
| POSITIVE | HARD | 150 | 0.329±0.013 (10, 0) | 0.329±0.013 (10, 0) | 0.329±0.013 (10, 0) | 0.336±0.014 (10, 0) |
| BOUNDED | EASY | 75 | 0.496±0.055 (10, 0) | 0.491±0.046 (10, 0) | 0.491±0.046 (10, 0) | 0.513±0.059 (10, 0) |
| BOUNDED | EASY | 150 | 0.327±0.017 (10, 0) | 0.326±0.018 (10, 0) | 0.326±0.018 (10, 0) | 0.329±0.015 (10, 0) |
| BOUNDED | HARD | 75 | 0.394±0.028 (10, 0) | 0.398±0.030 (10, 0) | 0.399±0.023 (10, 0) | 0.409±0.034 (10, 0) |
| BOUNDED | HARD | 150 | 0.300±0.014 (10, 0) | 0.302±0.015 (10, 0) | 0.301±0.013 (10, 0) | 0.301±0.012 (10, 0) |
| MIXED | EASY | 75 | 0.347±0.069 (10, 0) | 0.308±0.021 (10, 5) | 0.339±0.071 (10, 2) | 0.421±0.209 (10, 0) |
| MIXED | EASY | 150 | 0.281±0.053 (10, 0) | 0.285±0.043 (10, 2) | 0.286±0.053 (10, 1) | 0.392±0.279 (10, 0) |
| MIXED | HARD | 75 | 0.382±0.022 (10, 0) | 0.383±0.016 (10, 0) | 0.382±0.014 (10, 0) | 0.406±0.020 (10, 0) |
| MIXED | HARD | 150 | 0.300±0.008 (10, 0) | 0.300±0.008 (10, 0) | 0.300±0.008 (10, 0) | 0.304±0.017 (10, 0) |

paired gap（同一 dataset 内の差、3 手法とも成功した dataset のみ・件数を併記）:

| design | Auto | datasets | mean(Auto−Oracle) | mean(Auto−SupportOnly) | Auto < SupportOnly の dataset 数 |
|---|---|---|---|---|---|
| COUNT | AUTO_START_FIRST | 35/40 | -0.0016 | -0.0399 | 26/35 |
| COUNT | AUTO_START_LAST | 38/40 | -0.0002 | -0.0713 | 29/38 |
| POSITIVE | AUTO_START_FIRST | 40/40 | +0.0000 | -0.0183 | 35/40 |
| POSITIVE | AUTO_START_LAST | 39/40 | +0.0002 | -0.0180 | 34/39 |
| BOUNDED | AUTO_START_FIRST | 40/40 | +0.0000 | -0.0089 | 28/40 |
| BOUNDED | AUTO_START_LAST | 40/40 | +0.0001 | -0.0088 | 29/40 |
| MIXED | AUTO_START_FIRST | 33/40 | -0.0016 | -0.0434 | 24/33 |
| MIXED | AUTO_START_LAST | 37/40 | -0.0003 | -0.0527 | 26/37 |

## 表4 EASY / HARD（difficulty は selector と独立に distribution distance で事前 freeze）

| support group | τ EASY | τ HARD | D(τ_EASY) | D(τ_HARD) | A recovery EASY | A recovery HARD | B Auto(FIRST) EASY | B Auto(FIRST) HARD |
|---|---|---|---|---|---|---|---|---|
| COUNT | 2.0 | 0.25 | 0.113 | 0.070 | 0.987 (592/600) | 0.982 (589/600) | 0.947 (213/225) | 0.987 (296/300) |
| POSITIVE | 2.0 | 0.25 | 63.443 | 1.710 | 1.000 (240/240) | 1.000 (240/240) | 1.000 (120/120) | 1.000 (120/120) |
| BOUNDED | 0.25 | 2.0 | 0.086 | 0.069 | 0.953 (343/360) | 0.944 (340/360) | 0.978 (176/180) | 0.911 (164/180) |

D(τ) = 各 true candidate について最も近い同群 competitor への期待 pointwise-projection KL（nats/観測）の平均（下界）。

## 補助表 n=75 vs n=150（score-decided, family-level）

| support group | A n=75 | A n=150 | B Auto(FIRST) n=75 | B Auto(FIRST) n=150 |
|---|---|---|---|---|
| COUNT | 0.968 (581/600) | 1.000 (600/600) | 0.953 (286/300) | 0.991 (223/225) |
| POSITIVE | 1.000 (240/240) | 1.000 (240/240) | 1.000 (120/120) | 1.000 (120/120) |
| BOUNDED | 0.919 (331/360) | 0.978 (352/360) | 0.917 (165/180) | 0.972 (175/180) |

## 補助表 start agreement（START_FIRST vs START_LAST, main）

| design | datasets | both succeeded | identical final assignment | per-column agreement |
|---|---|---|---|---|
| COUNT | 40 | 35 | 30/35 | 0.989 |
| POSITIVE | 40 | 39 | 38/39 | 0.996 |
| BOUNDED | 40 | 40 | 36/40 | 0.983 |
| MIXED | 40 | 33 | 27/33 | 0.992 |

## 補助表 candidate optimiser diagnostics（main; grad_inf > 1e-8 は WARNING）

| experiment | candidate model | evaluations | failed | grad_inf>1e-8 | median grad_inf | max grad_inf |
|---|---|---|---|---|---|---|
| A | binomial_n2 | 248 | 0 | 67 | 5.23e-10 | 6.70e-07 |
| A | cb | 720 | 0 | 123 | 3.38e-11 | 2.04e-07 |
| A | cmp_nu0p5 | 1200 | 0 | 510 | 2.24e-09 | 6.81e-05 |
| A | cmp_nu2p0 | 1200 | 0 | 409 | 6.99e-10 | 1.12e-04 |
| A | logitnormal_s1 | 720 | 0 | 0 | 1.95e-14 | 2.12e-13 |
| A | lognormal_s1 | 480 | 0 | 0 | 1.51e-14 | 1.72e-13 |
| A | poisson | 1200 | 0 | 466 | 1.44e-09 | 1.27e-04 |
| A | ptn_s1 | 480 | 0 | 185 | 2.70e-09 | 3.05e-05 |
| A | tn01_s0p25_c0p5 | 720 | 0 | 118 | 1.42e-11 | 2.24e-07 |
| A | ztp | 240 | 0 | 113 | 5.65e-09 | 2.73e-05 |
| B | binomial_n2 | 1832 | 0 | 441 | 3.60e-11 | 1.74e-06 |
| B | cb | 5760 | 0 | 792 | 1.22e-11 | 2.87e-07 |
| B | cmp_nu0p5 | 8760 | 0 | 3332 | 1.27e-09 | 2.15e-04 |
| B | cmp_nu2p0 | 8760 | 0 | 3037 | 6.02e-10 | 3.14e-04 |
| B | logitnormal_s1 | 5760 | 0 | 0 | 1.55e-14 | 8.67e-13 |
| B | lognormal_s1 | 3792 | 0 | 0 | 9.30e-15 | 1.37e-12 |
| B | poisson | 8760 | 0 | 3435 | 1.63e-09 | 1.12e-04 |
| B | ptn_s1 | 3792 | 0 | 1270 | 2.59e-10 | 1.21e-04 |
| B | tn01_s0p25_c0p5 | 5760 | 0 | 869 | 1.23e-11 | 2.99e-07 |
| B | ztp | 1752 | 0 | 539 | 3.16e-10 | 1.66e-04 |
| C | binomial_n2 | 1136 | 0 | 322 | 7.64e-11 | 2.12e-06 |
| C | cb | 3360 | 0 | 427 | 1.47e-11 | 2.32e-07 |
| C | cmp_nu0p5 | 5600 | 0 | 2251 | 1.86e-09 | 4.98e-04 |
| C | cmp_nu2p0 | 5600 | 0 | 2112 | 1.41e-09 | 1.29e-04 |
| C | logitnormal_s1 | 3360 | 0 | 0 | 1.67e-14 | 5.15e-13 |
| C | lognormal_s1 | 2240 | 0 | 0 | 9.27e-15 | 3.63e-13 |
| C | poisson | 5600 | 0 | 2311 | 1.76e-09 | 2.05e-04 |
| C | ptn_s1 | 2240 | 0 | 704 | 2.06e-10 | 2.54e-05 |
| C | tn01_s0p25_c0p5 | 3360 | 0 | 517 | 8.43e-12 | 3.25e-07 |
| C | ztp | 1120 | 0 | 367 | 5.40e-10 | 5.28e-05 |

