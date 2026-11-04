# Figure captions and counts (auto-generated from tables/)

## 01_family_recovery_by_support_group.png

- source: `tables/family_recovery_summary.csv`
- pooled over difficulty and n; failed columns count as incorrect. A: selector only (true Z): COUNT 1181/1200 (failed 0); POSITIVE 480/480 (failed 0); BOUNDED 683/720 (failed 0) | B: Auto MCEM, START_FIRST: COUNT 509/525 (failed 0); POSITIVE 240/240 (failed 0); BOUNDED 340/360 (failed 0) | B: Auto MCEM, START_LAST: COUNT 554/570 (failed 0); POSITIVE 233/234 (failed 0); BOUNDED 342/360 (failed 0) | C: mixed-X Auto, START_FIRST: COUNT 320/330 (failed 0); POSITIVE 132/132 (failed 0); BOUNDED 190/198 (failed 0)

## 02_family_confusion_count.png

- source: `tables/confusion_matrix_family.csv`
- total columns 1200; FAILURE column present if any selection failed.

## 03_family_confusion_positive.png

- source: `tables/confusion_matrix_family.csv`
- total columns 480; FAILURE column present if any selection failed.

## 04_family_confusion_bounded.png

- source: `tables/confusion_matrix_family.csv`
- total columns 720; FAILURE column present if any selection failed.

## 05_family_recovery_easy_vs_hard.png

- source: `tables/family_recovery_summary.csv`
- A: selector only EASY: COUNT 592/600 (failed 0); POSITIVE 240/240 (failed 0); BOUNDED 343/360 (failed 0) | A: selector only HARD: COUNT 589/600 (failed 0); POSITIVE 240/240 (failed 0); BOUNDED 340/360 (failed 0) | B: Auto MCEM (START_FIRST) EASY: COUNT 213/225 (failed 0); POSITIVE 120/120 (failed 0); BOUNDED 176/180 (failed 0) | B: Auto MCEM (START_FIRST) HARD: COUNT 296/300 (failed 0); POSITIVE 120/120 (failed 0); BOUNDED 164/180 (failed 0)

## 06_family_recovery_n75_vs_n150.png

- source: `tables/family_recovery_summary.csv`
- A: selector only 75: COUNT 581/600 (failed 0); POSITIVE 240/240 (failed 0); BOUNDED 331/360 (failed 0) | A: selector only 150: COUNT 600/600 (failed 0); POSITIVE 240/240 (failed 0); BOUNDED 352/360 (failed 0) | B: Auto MCEM (START_FIRST) 75: COUNT 286/300 (failed 0); POSITIVE 120/120 (failed 0); BOUNDED 165/180 (failed 0) | B: Auto MCEM (START_FIRST) 150: COUNT 223/225 (failed 0); POSITIVE 120/120 (failed 0); BOUNDED 175/180 (failed 0)

## 07_rmse_z_oracle_auto_supportonly.png

- source: `tables/rmse_paired_gaps.csv`
- pooled over difficulty and n; failed fits have no RMSE and are counted in rmse_summary.csv (n_failed). COUNT/ORACLE: 40 successful fits; COUNT/AUTO_START_FIRST: 35 successful fits; COUNT/SUPPORT_ONLY: 40 successful fits; POSITIVE/ORACLE: 40 successful fits; POSITIVE/AUTO_START_FIRST: 40 successful fits; POSITIVE/SUPPORT_ONLY: 40 successful fits; BOUNDED/ORACLE: 40 successful fits; BOUNDED/AUTO_START_FIRST: 40 successful fits; BOUNDED/SUPPORT_ONLY: 40 successful fits; MIXED/ORACLE: 40 successful fits; MIXED/AUTO_START_FIRST: 33 successful fits; MIXED/SUPPORT_ONLY: 40 successful fits

## 08_score_margin_by_true_family.png

- source: `tables/selection_results.csv`
- 2400 score-decided selected columns; dashed line = margin 2. Selection failures have no margin and are listed in failures.csv.

## 09_start_agreement.png

- source: `tables/start_agreement.csv`
- COUNT: 35/40 datasets with both starts successful; POSITIVE: 39/40 datasets with both starts successful; BOUNDED: 40/40 datasets with both starts successful; MIXED: 33/40 datasets with both starts successful

## 10_mixed_x_family_recovery.png

- source: `tables/candidate_model_recovery_summary.csv`
- gaussian columns are gate-decided (REAL group). EASY: poisson 23/40 (failed 14); binomial_n2 26/40 (failed 14); cmp_nu2p0 23/40 (failed 14); cmp_nu0p5 25/40 (failed 14); ztp 25/40 (failed 14); lognormal_s1 26/40 (failed 14); ptn_s1 26/40 (failed 14); logitnormal_s1 26/40 (failed 14); cb 26/40 (failed 14); tn01_s0p25_c0p5 24/40 (failed 14); gaussian 26/40 (failed 14) | HARD: poisson 39/40 (failed 0); binomial_n2 40/40 (failed 0); cmp_nu2p0 39/40 (failed 0); cmp_nu0p5 40/40 (failed 0); ztp 40/40 (failed 0); lognormal_s1 40/40 (failed 0); ptn_s1 40/40 (failed 0); logitnormal_s1 39/40 (failed 0); cb 38/40 (failed 0); tn01_s0p25_c0p5 37/40 (failed 0); gaussian 40/40 (failed 0)
