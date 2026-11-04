# RESUME — expanded_multisupport_20261006_034826

- updated (UTC): 2026-10-05T21:08:34.390924+00:00
- note: all scheduled phases finished

## Phase status

| phase | gate | expected | completed | failed cases | unexpected | fit failure fraction | technical gate |
|---|---|---:|---:|---:|---:|---:|---|
| smoke | Gate D (selector + MCEM smoke) | 38 | 38 | 0 | 0 | 0.000 | PASS |
| pilot | Gate D' (small pilot) | 114 | 114 | 4 | 0 | 0.042 | PASS |
| A_main | Gate E (selector-only main) | 240 | 240 | 0 | 0 | 0.000 | PASS |
| B_main_r5 | Gate F (support-group MCEM, rep 1-5) | 240 | 240 | 7 | 0 | 0.029 | PASS |
| C_main_r5 | Gate G (mixed-X MCEM, rep 1-5) | 80 | 80 | 5 | 0 | 0.062 | PASS |
| B_main_r10 | Gate F (support-group MCEM, rep 6-10) | 240 | 240 | 1 | 0 | 0.004 | PASS |
| C_main_r10 | Gate G (mixed-X MCEM, rep 6-10) | 80 | 80 | 5 | 0 | 0.062 | PASS |

## Existing tables

- tables/candidate_model_recovery_summary.csv
- tables/candidate_registry.csv
- tables/candidate_scores.csv
- tables/confusion_matrix_candidate.csv
- tables/confusion_matrix_family.csv
- tables/convergence_summary.csv
- tables/dataset_manifest.csv
- tables/difficulty_calibration.csv
- tables/failures.csv
- tables/family_compatibility.csv
- tables/family_recovery_summary.csv
- tables/fit_results.csv
- tables/margin_summary.csv
- tables/optimization_diagnostics.csv
- tables/rmse_paired_gaps.csv
- tables/rmse_summary.csv
- tables/selection_results.csv
- tables/selection_trace.csv
- tables/start_agreement.csv

## Existing figures

- figures/01_family_recovery_by_support_group.png
- figures/02_family_confusion_count.png
- figures/03_family_confusion_positive.png
- figures/04_family_confusion_bounded.png
- figures/05_family_recovery_easy_vs_hard.png
- figures/06_family_recovery_n75_vs_n150.png
- figures/07_rmse_z_oracle_auto_supportonly.png
- figures/08_score_margin_by_true_family.png
- figures/09_start_agreement.png
- figures/10_mixed_x_family_recovery.png

## How to resume

Completed cases are skipped automatically; failed cases are completed outcomes and are NOT rerun.

```
cd expfam/src/experimental
python drive_multisupport_family_selection.py --run-dir D:/tento/kennkyu/expfam/results/family_selection/expanded_multisupport_20261006_034826 --workers 10
```

A single phase can be resumed with:

```
python run_multisupport_family_selection.py run --run-dir D:/tento/kennkyu/expfam/results/family_selection/expanded_multisupport_20261006_034826 --phase <phase>
```

Blocked families (mathematical audit): geometric, negative_binomial, gamma, exponential, beta, weibull, student_t, categorical, inverse_gaussian (see tables/family_compatibility.csv).
