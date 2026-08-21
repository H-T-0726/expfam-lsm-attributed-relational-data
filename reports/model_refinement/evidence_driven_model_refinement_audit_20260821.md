# Evidence-Driven Model Refinement Audit

## 1. Scope

This is a **read-only research audit**. Its purpose is to decide, from evidence that
already exists in this repository, what the next research step should be. It is not a
model-refinement phase, and it does not implement, validate, or promote anything.

The question it answers is not "which change would make the model look better", but:

```
OBSERVED FAILURE
  -> MECHANISM HYPOTHESIS
    -> ALTERNATIVE EXPLANATIONS
      -> DISCRIMINATING EXPERIMENT
        -> ONLY THEN MODIFICATION
```

Explicitly out of scope and **not done**: new synthetic data generation, any EM run,
MovieLens re-training, re-running historical experiments, hyper-parameter tuning, seed
changes, model modification, and the Issue #27 experiment.

What was done: read the actual code and the canonical documents, and **recompute
aggregates, paired differences, win counts, criterion values and row counts directly
from the primary CSVs**. Numbers written in existing reports were treated as claims to
be checked, not as evidence.

Every substantive statement below carries one of the labels
`VERIFIED` / `OBSERVED` / `DERIVED` / `SUPPORTED` / `PARTIALLY_SUPPORTED` /
`CONFOUNDED` / `UNTESTED` / `CONTRADICTED` / `UNRESOLVED`.
Inference is never written as `OBSERVED`.

### Conclusion discipline

The audit was planned before the recomputation, and the plan recorded several prior
hypotheses. Those hypotheses were held as hypotheses. The conclusion sections (§17-§22)
are derived from the recomputation in §5-§16. Candidates A-E in §16 were scored on a
uniform rubric fixed before scoring.

**Revision history.** This document was revised on 2026-08-21 after an independent review
of PR #29. Five corrections were applied and their downstream consequences were re-derived
rather than patched:

| # | correction | where | downstream effect |
|---|---|---|---|
| 1 | The §8.2 "misspecification cost vs integration value" reading was **not a causal decomposition**; the three conditions differ in more than one factor at a time | §8.2, §6 F2, §7, §11.3, §16, §18, §21, §22, §23 | the "8.4x / 4.1x" causal claim is withdrawn; the three comparisons are renamed as descriptive contrasts A/B/C; **PATH 4 and PATH 2 are stronger than the earlier draft concluded** |
| 2 | Candidate E was scored against the superseded July memo instead of the **actual current body of GitHub Issue #27** | §16.2, §16.3, §18 | all four earlier criticisms of E are withdrawn as factually wrong; E rescored 22 -> 28; **the Issue #27 verdict changes from REDESIGN to RUN NEXT** |
| 3 | X-side and Y-side overdispersion were conflated under one candidate and one verdict | §9.3, §14, §15, §19, §20, §23 | candidate J splits into J-Y (`NOT_SUPPORTED`) and J-X (`UNTESTED`, `CONFOUNDED`); the blanket "NB/dispersion NOT_JUSTIFIED" is withdrawn |
| 4 | The `1/sigma_hat^2 ~ 2.0e-4` figure was cited as `VERIFIED` in §11.2 although `sigma_hat` was never persisted | §11.2 | relabelled `DERIVED` / `PARTIALLY_SUPPORTED`, consistent with §9.2 |
| 5 | Candidate A cannot isolate the baseline, because under Poisson `alpha` moves the mean, the variance and `A''` together | §16.2, §16.3, §17, §22, §23 | A rescored 30 -> 27 with an added dispersion factor; **the F3 branch is identified as structurally blocked** |

Where a conclusion changed, the earlier conclusion is stated and marked withdrawn rather
than silently replaced.

---

## 2. Commit / evidence hierarchy

- Actual `origin/main` at audit time: `e132bedc8d91bb5253418b78a48ffb8ef453e7b0`
- Audit branch: `audit/28-evidence-driven-model-refinement`, created directly from that commit
- Working tree at branch creation: clean
- New model fits run: **NO**
- Model / results / registry / canonical documents changed: **NO**
- Tracked files added by this audit: this report only

### Evidence hierarchy applied

1. Primary CSV / runinfo / actual code
2. The script whose primary execution conditions can be confirmed
3. Verified audit output
4. Summary report
5. AI-generated historical prose

Level 4 and 5 material was used only to locate evidence, never as the source of a
number. Where a level-4 number disagrees with the primary artifact of the same
experiment, the disagreement is reported in §13 and the primary artifact wins.

### Recomputation scripts

All recomputation was done with short read-only scripts kept **outside the repository**
(session scratchpad). They only read CSV / `.npy` inputs and print to stdout. No
repository file was written except this report.

---

## 3. Experiment inventory

Row counts, condition counts, trial counts, seed sets and NaN flags below were read
from the primary CSVs, not from any report. Where a file carries a `nan_occurred`,
`n_nan` or `success` column, the value was `False` / `0` / `True` for **every** row.
The old-0.5 scenario CSVs (row 1) carry **no such column at all**, so nothing is
claimed about NaN there; their row counts are 60 for each Exp1/Exp2/Exp3 file and 110
for each Exp4 file, per scenario. `numerics` is `legacy` everywhere: the
objective-consistent lineage did not exist when any of these ran.

| # | Experiment | Lineage / class | Data | Conditions | n | k | Trials / fits | Primary metrics | Paired? | Numerics | Role | Known caveats | Primary artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Scen A/B/C Exp1-4 (old 0.5) | `DualExpFamLSM` (0.5 present) | synthetic | 3 scenarios x Exp1-4 | 150 | 3 | 10/cond | RMSE(Z), criterion | by seed | legacy | claim-bearing (paper) | KI-001/002/003 | `expfam/results/exp_scenario_*_exp*.csv` |
| 2 | fixed_official Exp1 (k sweep) | `DualExpFamLSMFixed` | synthetic | 3 scen x k=1..6 | 150 | 1-6 | 180 rows = 3x10x6 | Q_strict, criterion | by seed | legacy | support | not in the paper | `fixed_official/exp1/fixed_exp1_bic_full_summary.csv` |
| 3 | fixed_official Exp2 (n sweep) | Fixed | synthetic | 3 scen x n=50..300 | 50-300 | 3 | 180 rows | RMSE(Z) | by seed | legacy | support | registry note disagrees, §13 | `fixed_official/exp2/fixed_exp2_n_sweep_summary.csv` |
| 4 | fixed_official Exp3 (d sweep) | Fixed | synthetic | 3 scen x d=5..30 | 150 | 3 | 180 rows | RMSE(Z) | by seed | legacy | support | Scen B mean vs median differ | `fixed_official/exp3/fixed_exp3_d_sweep_summary.csv` |
| 5 | fixed_official Exp4 (mismatch) | Fixed | synthetic | 11 cond x 3 scen | 150 | 3 | A 110, B 110, C 330 | RMSE(Z) ratio vs oracle | by seed | legacy | support | different conditions from the old-0.5 ratios | `fixed_official/exp4/fixed_exp4_scen_{a,b,c}_summary.csv` |
| 6 | fixed mismatch grid | Fixed | synthetic | 3x3x3 | 150 | 3 | 270 rows | RMSE(Z) | by seed | legacy | support | KI-002 | `distribution_mismatch_fixed/mismatch_fixed_all_trials.csv` |
| 7 | Wine fixed pilot | Fixed | real (Y label-derived) | k=1..9 + 3 ablations | 178 | 1-9 | 45 + 15 | criterion, AUC/AP, silhouette | by seed | legacy | support | Y is artificial (KI-006) | `real_data/wine_fixed_pilot/*.csv` |
| 8 | Cora balanced k sweep | Fixed | real citation net | k=1..6 | 280 | 1-6 | 18 rows (3/k) | Q_strict, criterion, AUC/AP/NMI/ARI | by seed | legacy | support | density 0.011, KI-011 | `real_data/cora_balanced_k_sweep/*.csv` |
| 9 | Cora held-out link prediction | Fixed | real | k=3,6 | 280 | 3,6 | 6/k | test AUC/AP | by seed | legacy | support | subset, not full Cora | `real_data/cora_heldout_link_prediction/*.csv` |
| 10 | MovieLens held-out count | Fixed | real (in-sample split) | k=3,5 | 100 | 3,5 | 6/k | RMSE, Pearson, hc_AP | by seed | legacy | support | pair-mask not used here (KI-012) | `real_data/movielens_heldout_count/*.csv` |
| 11 | MovieLens overdispersion diagnostics + PPC | experimental (masked) | real | marginal + conditional k=3,5 | 100 | 3,5 | 2 conditional fits, 300 PPC reps | var/mean, Pearson dispersion | no | legacy | diagnostic | plug-in PPC | `overdispersion/movielens_{overdispersion_diagnostics,ppc_summary}.csv` |
| 12 | MovieLens strict held-out (pair mask) | experimental masked / NB | real | 3 cond x k=3,5 | 100 | 3,5 | 36 rows = 6/cell | test ll, RMSE, hc_AUC/AP | yes (split x model) | legacy | diagnostic | one divergent poisson_strict fit at k=5 | `overdispersion/movielens_strict_heldout_summary.csv` |
| 13 | Poisson misspecification (synthetic NB-Y) | experimental NB | synthetic | r in {2,5,20,inf} x {poisson, nb_oracle, nb_moment} | - | - | 55 rows, 5 trials/cell | RMSE(Z), test ll | yes | legacy | diagnostic | `nb_oracle` absent for r=inf | `overdispersion/poisson_misspecification_summary.csv` |
| 14 | MovieLens shared-Z ablation | experimental masked | real | proposed / X-only / Y-only | 100 | 5 | 18 rows = 6/cond | strict test ll, NMI | yes (3 split x 2 model) | legacy | diagnostic | NMI procedure differs from pilot | `shared_z_ablation/movielens_shared_z_ablation_summary.csv` |
| 15 | per-column math audit | experimental per-column | fixtures | 7 check families | small | small | 34 rows = 31 PASS + 3 INFO | max abs diff vs finite difference | n/a | legacy | verification | interior only (see §12) | `per_column_family/per_column_math_audit_summary.csv` |
| 16 | per-column demo | experimental per-column | synthetic mixed X | 4 cond | - | - | 20 rows, 5 trials | RMSE(Z), criterion | yes | legacy | diagnostic | prototype | `per_column_family/per_column_demo_summary.csv` |
| 17 | single vs joint | experimental per-column | synthetic mixed X, strict held-out | 9 cond | 80 | 2 | 27 rows = 9x3 | RMSE(Z), test_y_ll, per-block X RMSE, criterion | yes | legacy | diagnostic | 3 seeds only | `per_column_family/single_vs_joint_summary.csv` |
| 18 | attribute ablation | experimental per-column | synthetic | 5 steps | 80 | 2 | 15 rows = 5x3 | RMSE(Z), test_y_ll | yes | legacy | diagnostic | 3 seeds only | `per_column_family/attribute_ablation_summary.csv` |
| 19 | noise attribute check | experimental per-column | synthetic | 6 cond (0/3/6/12 noise cols) | 80 | 2 | 18 rows = 6x3 | RMSE(Z), test_y_ll | yes | legacy | diagnostic | 3 seeds only | `per_column_family/noise_check_summary.csv` |
| 20 | MovieLens mixed-X pilot | experimental per-column | real, strict held-out | 6 cond | 100 | 3 | 24 rows = 6 x (2 split x 2 model) | test_y_ll, RMSE, Spearman | yes | legacy | diagnostic | leakage caveat in runinfo | `per_column_family/movielens_mixed_x_summary.csv` |
| 21 | Y sparsity stress (trials 10) | experimental per-column | synthetic, strict held-out | 4 cond x 4 rates | 80 | 2 | 160 rows = 4x4x10 | RMSE(Z), test_y_ll | yes (trial-matched) | legacy | diagnostic | **one generative configuration only** | `story_diagnostics/y_sparsity_stress_20260713_trials10.csv` |
| 22 | MovieLens attribute diagnosis (trials 4) | experimental per-column | real, strict held-out | 11 cond | 100 | 3 | 44 rows = 11 x (2 split x 2 model) | test_y_ll, hc_AUC/AP, per-column X RMSE, w0, w | yes (fit-matched) | legacy | diagnostic | leakage; **2 independent splits** | `story_diagnostics/movielens_attribute_diagnosis_20260713_trials4.csv` |
| 23 | Y sparsity smoke / attribute diagnosis smoke | experimental per-column | as above | as above | - | - | 32 / 11 rows | as above | partial | legacy | archive | superseded by 21 / 22 | `story_diagnostics/y_sparsity_stress_20260713.csv`, `story_diagnostics/movielens_attribute_diagnosis_20260713_smoke.csv` |

### Effective sample sizes (VERIFIED by row counts and key columns)

This is the first thing the audit fixes, because most later disagreements are about how
much a number is worth, not what the number is.

| Experiment | Nominal | What is actually independent |
|---|---|---|
| MovieLens attribute diagnosis | "trials 4" | **2 data splits x 2 model seeds**. Split-level spread reaches 0.098 in test_y_ll, model-seed spread within a split is far smaller. Effective independent replication = **2**. |
| MovieLens mixed-X pilot | "4 fits" | same 2x2 structure. Effective = **2**. |
| MovieLens strict held-out / shared-Z | 6 fits/cell | 3 split trials x 2 model seeds. Effective = **3**. |
| noise check / attribute ablation / single vs joint | 3 seeds | 3 fully independent trials (data, model and split seeds all vary). Effective = **3**. |
| Y sparsity stress trials10 | 10 trials | 10 independent trials, trial-matched across conditions and rates - but **one generative configuration** (n=80, d=9, k*=2, gauss3+bern3+pois3, Poisson-Y, w0=1.2, w=0.3). |
| Cora k sweep | 3 trials/k | 3. |
| fixed_official Exp1-4 | 10 (30 for Scen C Exp4) | 10 (30). |

---

## 4. Current model lineage

```
reproduction/src/model.py            LatentStructuralModel                      (prior work reproduction)
+- expfam/src/model_expfam.py        ExpFamLatentStructuralModel                (Y-side ExpFam, 0.5 present)
   +- model_dual_expfam.py           DualExpFamLSM                              (0.5 present; paper experiments)
      +- model_dual_expfam_fixed.py  DualExpFamLSMFixed                         (no extra 1/2; real-data phase)
         +- experimental/model_dual_expfam_masked.py  DualExpFamLSMMasked       (pair mask / strict held-out)
            +- experimental/model_dual_expfam_nb.py   DualExpFamLSMNB
            +- experimental/model_dual_expfam_percolumn.py DualExpFamLSMPerColumn   (legacy per-column prototype)
            +- experimental/model_dual_expfam_consistent.py
                 DualExpFamLSMConsistent / DualExpFamLSMPerColumnConsistent     (objective-consistent, forward-only)
```

Facts that must not be blurred (all `VERIFIED` against code and the two 2026-08-21 reports):

- The **thesis / main method is still a single scalar `family_x` shared by all X columns.**
  `DualExpFamLSMPerColumn` is an experimental prototype and is not promoted here.
- `numerics_mode` defaults to `"legacy"` in `em_runner.run_em_experimental`. **Every
  experiment in §3 ran on legacy numerics**; the consistent classes did not exist yet.
- The 1/2 question (KI-001) is a separate axis from everything in this audit and is not
  re-adjudicated. The printed Mikawa et al. equations contain 1/2; the old Python lineage
  contains 0.5; the adopted derivation and the fixed/masked/per-column lineage do not.
- `scale_Z` is applied **unconditionally** in `expfam/src/experimental/em_runner.py` line 226
  (all MC samples rescaled to mean square 1). There is no switch. This is a global
  confounder for every scale-related mechanism in §14. `VERIFIED` from code.

---

## 5. Success map

Failures are only interpretable against what already works. All figures below were
recomputed from the primary CSVs.

### 5.1 Correctly specified synthetic recovery (fixed lineage)

`VERIFIED`

- **Latent dimension selection.** In `fixed_exp1_bic_full_summary.csv`, taking the
  argmin of the Q-based criterion per trial over k in 1..6: scenario A **10/10** trials
  select k=3, B **10/10**, C **10/10** (k_true = 3). The bestk file agrees.
- **Consistency in n.** RMSE(Z) at n=50 -> n=300: A 0.3417 -> 0.1733 (**-49.3%**),
  B 0.1902 -> 0.1118 (**-41.2%**), C 0.0398 -> 0.0165 (**-58.6%**). Recomputed values
  match `fixed_exp2_n_sweep_improvement.csv` exactly (49.3 / 41.2 / 58.6).
- **Behaviour in d.** A improves monotonically 0.2639 -> 0.2046 (**-22.5%**);
  C is flat (0.0232 -> 0.0231); B improves in the median (0.1701 -> 0.1202) but not in
  the mean (0.1695 -> 0.1665), i.e. outlier-driven.
- **Misspecification hurts.** Worst-vs-oracle RMSE(Z) ratio: A **4.34x** (XBern_YBern),
  B **9.04x** (XPois_YBern), C **40.37x** (XPois_YBern). These reproduce the registry
  values exactly.

### 5.2 Dense Y: adding X changes little

`VERIFIED`. From `single_vs_joint_summary.csv` (dense Y, strict held-out, 3 trials,
trial-matched):

| comparison | mean RMSE(Z) difference | per-trial | per_column wins |
|---|---:|---|---:|
| `all_gaussian` - `per_column_all` | **-0.0004** | +0.0014, -0.0066, +0.0039 | 2/3 |
| `single_gaussian` - `per_column_all` | +0.0081 | +0.0095, +0.0084, +0.0065 | 3/3 |
| `y_only` - `per_column_all` | +0.0935 | +0.0556, +0.1237, +0.1013 | 3/3 |

So at dense Y, **per-column correct specification is not better than forcing every column
to Gaussian** - the sign is in fact marginally the other way. The same picture appears at
`y_obs_rate = 1.0` in the sparsity experiment (§8).

### 5.3 Sparse Y: X integration does help

`SUPPORTED` (10 trial-matched trials, one generative configuration). See §8 for the
decomposition. `per_column_all` is best at every rate; at `y_obs_rate = 0.1` it beats
`y_only` by 0.833 RMSE(Z) (10/10) and `all_gaussian` by 0.426 (10/10).

### 5.4 per-column joint vs single-family

`VERIFIED`, with the size of the effect stated honestly: `per_column_all` beats every
single-block model in 3/3 trials, but the margins are +0.0081 (single_gaussian),
+0.0594 (single_poisson), +0.0861 (single_bernoulli). It beats the raw-value forced
misspecifications `all_poisson` (+0.0610, 3/3) and `all_bernoulli` (+0.5628, 3/3),
where `all_bernoulli` collapsed in exactly **1 of 3** trials (RMSE(Z) 1.747,
test_y_ll -106.5; the other two trials were 0.328 and 0.318).

Also `VERIFIED` and important for model comparison: the criterion of the collapsed
`all_bernoulli` condition is the **smallest** of all nine conditions
(mean 7216.4, versus `per_column_all` 11647.8 and `y_only` 14718.1). A criterion computed
under an invalid likelihood is not comparable across families.

### 5.5 Gaussian misspecification is benign for Z and Y

`VERIFIED` across three independent settings. This is §11.

### 5.6 Real data

`VERIFIED`

- **Wine**: criterion argmin at **k=3**, matching the 3 true classes (mean criterion at
  k=1..9: 13410.8, 6423.2, **6324.8**, 6691.1, 6961.1, 7501.9, 8103.3, 8752.1, 9435.9).
  Ablation: `X+Y` AUC 1.000, `Y_only` AUC 0.9997, `X_only` AUC **0.500**. Wine's apparent
  success is carried entirely by Y, and Wine's Y is label-derived, so it is not evidence
  about attributes.
- **Cora** held-out link prediction: test_AP 0.4319 (k=3) and 0.4631 (k=6) against a
  random baseline of 0.1667, i.e. **2.59x and 2.78x**. Reproduces the 2.6-2.8x claim.
- **MovieLens** Bernoulli-X / Poisson-Y converges with no NaN in every fit recorded, and
  strict held-out (pair-masked) prediction is stable at k=3 (test ll -3.459, sd 0.009).

---

## 6. Failure map

| id | phenomenon | primary evidence | trials/fits | effect | paired evidence | reproducibility | confounders | candidate mechanisms | already ruled out | unresolved | class | scientific importance | next discriminating experiment | modification justified now |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F1** | At dense Y, per-column / single / all-Gaussian differ very little | `single_vs_joint_summary.csv`; `y_sparsity_..._trials10.csv` at rate 1.0 | 3 trials; 10 trials | all_gaussian - per_column = **-0.0004** (3 trials) and **+0.0114** (10 trials, 10/10) | yes, trial-matched | reproduced in two independent experiments | generator makes every block individually informative; n=80 is small | Y already carries enough information; F rows are dense so blocks are not complementary | that it is a NaN/divergence artefact (0 NaN) | whether a complementary-block generator changes it | **III** (design) | HIGH - it bounds the whole per-column claim | E (redesigned, §16) | NO |
| **F2** | As Y gets sparse, the per-column advantage grows | `y_sparsity_..._trials10.csv` | 10 paired trials per contrast per rate | contrast A (same 9 columns, `all_gaussian - per_column`): +0.011 / +0.065 / +0.412 / +0.426 across rates 1.0/0.5/0.2/0.1; contrast B (`single_gaussian - per_column`): +0.010 / +0.013 / +0.044 / +0.083. **A and B are separate contrasts, not orthogonal components (§8.2)** | yes | monotone in both contrasts; 10/10 wins at rates 0.2 and 0.1 | single generative configuration (n=80, d=9, k*=2); scalar-parameter error also grows | X compensates when Y information falls | seed dependence (10/10) | generalization to other n, d, k*, family mixes and to a complementary-block F | **II/III** | HIGH - the only positive result of the per-column line | E (Issue #27 as currently written, §16/§18) | NO |
| **F3** | MovieLens: adding raw `ratings_count` as a Poisson column degrades held-out Y prediction | `movielens_attribute_diagnosis_..._trials4.csv`; `movielens_mixed_x_summary.csv` | 4 fits (2 splits) in each of two experiments | test_y_ll -0.374 vs genre_only, **0/4 better**, per-fit -0.314..-0.441; w0 3.417 -> 3.156, w 0.272 -> 0.330, hc_AUC 0.970 -> 0.949 | yes, fit-matched | reproduced in two separate experiments, 4/4 fits each | leakage (count and Y share `u.data`); only 2 splits; legacy numerics | no X intercept; raw count scale; Poisson fixed dispersion with A''=exp(eta); **X-side count overdispersion (var/mean 6.17)**; precision-block dominance; count informativeness | Poisson X clipping (§12); **Y-side** overdispersion (§9, J-Y; Y is identical across all conditions); NaN/divergence (0) | actual `A''/phi f f^T` share; whether an intercept fixes it; whether raw count under Gaussian **with genre still Bernoulli** also degrades; whether a dispersion-aware count family for X fixes it | **II** (with a CLASS III leakage caveat) | HIGH - the only real-data failure of the per-column line | C (measure), then B extended with an X-dispersion arm (§16) | NO |
| **F4** | Transforming the count removes the degradation | same CSV | 4 fits | log-Gaussian **+0.0021**, z-score-Gaussian **+0.0016** vs genre_only; the two differ by +0.0005 (sd 0.0049) | yes | 2/4 fits better each, i.e. indistinguishable from genre_only | same as F3 | removal of the large baseline; switch to an estimated-dispersion family | that log specifically matters (z-score works identically) | which of centering / scaling / estimated dispersion is the operative part | **II** | HIGH - it is the causal-separation lever we already have | B (§16) | NO |
| **F5** | Adding noise attributes does not help and sometimes hurts | `noise_check_summary.csv` | 3 trials x 5 noise conditions | mean deltas: gauss3 **+0.0098**, gauss6 **-0.0005**, gauss12 **+0.0120**, bern3 +0.0004, pois3 +0.0010 | yes, trial-matched | **not reproducible as a dose response** - §10 | 3 seeds; single configuration | Gaussian noise gets `1/sigma_hat^2` weight; local optima | a monotone dose response (0/3 trials monotone) | whether a systematic effect exists at larger trial counts | **III**, possibly II later | MEDIUM | D (§16) | NO |
| **F6** | all-Gaussian is unexpectedly strong for Z and Y | `single_vs_joint`; `movielens_attribute_diagnosis`; `poisson_misspecification` | 3 + 4 + 15 paired | §11 | yes in all three | three independent settings, consistent direction | different data, metrics and families | quasi-likelihood robustness; estimated dispersion auto-downweighting; Y dominance; metric divergence (Z vs density) | that it is a fluke of one experiment | which of the candidate mechanisms dominates | **II/III** | HIGH - the strongest counterargument to the per-column claim | A and C give partial answers | NO |
| **F7** | Poisson / Bernoulli objective-score-curvature inconsistency in the legacy lineage | `per_column_math_code_audit_20260821.md` PC-001/PC-002 plus the code | deterministic counterexamples | at eta=11.5, x=3: implemented score -22023.47, precision 22026.47; the actual objective's finite-difference score and negative Hessian are both 0 | n/a | exactly reproducible | none - it is a code fact | hard clip `[-20,10]`; probability floor 1e-10; curvature floor 1e-8 | **resolved in the consistent lineage** (Issue #25 / PR #26) | whether it ever activated during historical EM runs | **I** | HIGH for future work, **LOW as an explanation of F3** (§12) | none for the defect itself; activation logging needed for the history | already fixed forward; not re-opened here |
| **F8** | On Cora, the k criterion disagrees with AUC/AP/NMI | `cora_balanced_k_sweep_summary.csv` | 3 trials x 6 k | criterion argmin k=1; AP/AUC argmax k=6; NMI/ARI argmax k=3 | by seed | consistent across 3 trials | density 0.011; n=280 subset | penalty too large in sparse data; **Q_strict itself is non-monotone in k**; the criterion is Q-based, not Schwarz | that the parameter count is wrong (`p = kd - k(k-1)/2` reproduces exactly) | why Q degrades for k>=4 (optimization vs MC vs Laplace) | **III** plus optimization | MEDIUM-HIGH | a Q-vs-k optimization diagnostic (§17 A4) | NO |
| **F9** (new) | On MovieLens, even genre-only X does not reliably help strict held-out Y | `movielens_shared_z_ablation_summary.csv` | 6 fits (3 splits x 2 seeds), k=5 | proposed_XY - y_only_fix_x = **-0.039** test ll, X helps **3/6** | yes, fit-matched | sign disagrees with the attribute-diagnosis experiment (+0.034, 4/4 at k=3) | different k, script, evaluation | X contributes little at this n and density; k differs | none | whether attribute integration helps MovieLens **at all** | **II/III** | HIGH - it weakens the premise of the whole MovieLens line | a matched-protocol re-run (§17, below the top five) | NO |

F9 is added by this audit. It was not in the Issue #28 list, and it changes how F3 should
be read: **F3 is a failure of one attribute, but F9 says the attributes we have on
MovieLens may not carry usable information about Y in the first place.**

---

## 7. Evidence strength

### Top 5 strongest pieces of existing evidence

1. **Correct-specification recovery on synthetic data** (`VERIFIED`). 180-row k sweep,
   180-row n sweep, 180-row d sweep, 550 rows of Exp4, all with 10-30 independent trials,
   0 NaN, and exact reproduction of the reported ratios (4.34 / 9.04 / 40.37).
2. **The MovieLens raw-count degradation** (`VERIFIED` as an observation). -0.374 test ll
   with 0/4 fits better, an effect roughly 20x the fit-to-fit sd, reproduced in two
   separate experiments with different scripts and seeds, and accompanied by a coherent
   secondary signature (w0 down, w up, hc_AUC down). Qualifier: those two experiments
   share the **same n=100 movie subset**, so they replicate the fitting procedure and not
   the sample (§9.4).
3. **The count column accounts for essentially the whole 22-column degradation**
   (`VERIFIED`). `mixed_percolumn_raw` - `genre_count_raw_poisson` = +0.0037 (sd 0.0073)
   and `mixed_percolumn_raw` - `rating_stats_only` = -0.0088 (sd 0.0060).
4. **The sparse-Y interaction** (`SUPPORTED`). Monotone in rate, 10/10 wins at the two
   sparsest rates, trial-matched, 0 NaN - but one generative configuration. The cleanest
   single number is contrast A in §8.2: on the **identical 9 columns**, per-column beats
   forced-Gaussian by 0.426 RMSE(Z) at `y_obs_rate = 0.1` (10/10) and by only 0.011 at
   dense Y (10/10). This is the per-column mechanism's own contrast and it is
   regime-dependent.
5. **Poisson beats a correctly specified NB on Z recovery while losing on likelihood**
   (`VERIFIED`). 5/5 paired trials at every r_true in {2, 5, 20}, in both directions. This
   is the cleanest existing evidence that family choice acts on Z through weighting rather
   than through distributional correctness.

### Prior hypotheses carried into this audit, and what happened to them

Recorded here so that the conclusions can be checked against what was believed before the
recomputation.

| Prior hypothesis (pre-recomputation) | Status after recomputation |
|---|---|
| "no intercept alone" and "raw scale alone" are each insufficient to break the model | `PARTIALLY_SUPPORTED` but **CONFOUNDED**: `mixed_all_gaussian` keeps the raw count and has no intercept and loses only 0.021, but it also forces genre to Gaussian, so it isolates neither factor (§9) |
| Poisson implied precision dominates | `PARTIALLY_SUPPORTED` on the data side (81.2x, §9.2); `UNRESOLVED` on the model side (F never saved) |
| The sparse-Y result is mostly about family correctness, not about integrating more blocks | `SUPPORTED` and quantified (§8.2). This changed the thesis recommendation |
| Cora's k problem is not only a penalty problem | `VERIFIED` - Q_strict peaks at k=2 (§13.6) |
| NB / overdispersion is not needed | **Split.** `SUPPORTED` for the **Y side** (§9, candidate J-Y). **Withdrawn for the X side**: `ratings_count` is 6.17x over-dispersed and X-side count dispersion is `UNTESTED` (candidate J-X). The blanket phrasing was wrong |

---

## 8. Sparse-Y evidence

Source: `expfam/results/story_diagnostics/y_sparsity_stress_20260713_trials10.csv`
(160 rows = 4 conditions x 4 rates x 10 trials, 0 NaN). All differences are computed
**within trial** and then averaged; win counts are over the 10 trials.

### 8.1 Levels and paired differences (RMSE(Z))

| y_obs_rate | train pairs | per_column_all | single_gaussian | all_gaussian | y_only |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 2528 | 0.2214 | 0.2318 | 0.2328 | 0.3078 |
| 0.5 | 1264 | 0.2660 | 0.2788 | 0.3307 | 0.4603 |
| 0.2 | 506 | 0.3204 | 0.3643 | 0.7320 | 0.8570 |
| 0.1 | 253 | 0.3429 | 0.4261 | 0.7690 | 1.1758 |

| y_obs_rate | `single_gaussian - per_column` | `all_gaussian - per_column` | `all_gaussian - single_gaussian` | `y_only - per_column` |
|---:|---:|---:|---:|---:|
| 1.0 | +0.0104 (8/10) | +0.0114 (10/10) | +0.0010 (3/10) | +0.0864 (10/10) |
| 0.5 | +0.0129 (9/10) | +0.0647 (9/10) | +0.0518 (7/10) | +0.1943 (10/10) |
| 0.2 | +0.0439 (10/10) | +0.4117 (10/10) | +0.3677 (10/10) | +0.5367 (10/10) |
| 0.1 | +0.0832 (10/10) | +0.4261 (10/10) | +0.3429 (10/10) | +0.8329 (10/10) |

Held-out Y log-likelihood gives the same ordering: `per_column_all` beats `all_gaussian`
by +0.0063 (7/10) at rate 1.0 and by +0.1230 (10/10) at rate 0.1.

### 8.2 Three named contrasts (descriptive, not a causal decomposition)

**Correction note (2026-08-21, PR #29 review).** An earlier version of this section read
`all_gaussian - single_gaussian` as a "pure misspecification cost" and
`single_gaussian - per_column_all` as an "integration value", and reported their ratio
(8.4x, 4.1x) as a causal decomposition. **That reading is withdrawn.** The three
conditions do not differ in one factor at a time:

| condition | columns used | family assignment |
|---|---|---|
| `single_gaussian` | 3 (the Gaussian block only) | correct for those 3 |
| `all_gaussian` | all 9 | 3 correct, 6 wrong |
| `per_column_all` | all 9 | all 9 correct |

`all_gaussian - single_gaussian` therefore mixes **"6 columns were added"** with
**"those 6 were assigned the wrong family"**. It is not a pure misspecification cost.

The three comparisons are reported below as **named descriptive contrasts**. They satisfy
the algebraic identity `A = B + C`, but that identity is bookkeeping, not orthogonality:
B and C are not independent causal components and must not be presented as a variance
decomposition of the sparse-Y benefit.

| contrast | definition | what varies | rate 1.0 | 0.5 | 0.2 | 0.1 |
|---|---|---|---:|---:|---:|---:|
| **A** same-column family-assignment contrast | `all_gaussian - per_column_all` | family assignment only, 9 columns in both arms | +0.0114 (10/10) | +0.0647 (9/10) | +0.4117 (10/10) | +0.4261 (10/10) |
| **B** additional-block contrast under correct specification | `single_gaussian - per_column_all` | 3 correctly specified columns vs 9 correctly specified columns | +0.0104 (8/10) | +0.0129 (9/10) | +0.0439 (10/10) | +0.0832 (10/10) |
| **C** misspecified-addition contrast | `all_gaussian - single_gaussian` | 6 columns added **and** assigned the wrong family | +0.0010 (3/10) | +0.0518 (7/10) | +0.3677 (10/10) | +0.3429 (10/10) |

What can be said, descriptively:

- `SUPPORTED`: the sparse-Y interaction is real and monotone in every contrast. All three
  grow as Y is thinned.
- `OBSERVED`: **contrast A is the cleanest single comparison available**, because both arms
  use the identical 9 columns and differ only in the family assigned to each. At
  `y_obs_rate = 0.1` per-column beats forced-Gaussian by **0.426 RMSE(Z), 10/10 trials**;
  at dense Y the same contrast is only **0.011**. Contrast A is the direct empirical
  statement of what the per-column mechanism does, and it is regime-dependent.
- `OBSERVED`: contrast B, the empirical benefit of correctly integrating six further
  blocks, is much smaller in absolute terms (+0.083 at rate 0.1, +0.010 at dense Y) though
  still 10/10 at the two sparsest rates.
- `NOT CLAIMED`: that the sparse-Y benefit "is mostly family correctness rather than
  integration". A and B answer different questions on different column sets; their sizes
  cannot be compared as competing causal shares.

The earlier report `reports/story_diagnostics/story_diagnostics_summary_20260713.md`
reports the per_column-vs-all_gaussian and per_column-vs-y_only gaps only, i.e. contrast A
and the y_only comparison. Nothing in it is wrong. What this audit adds is contrast B, and
the explicit warning that B and C are not orthogonal components.

### 8.3 Limits

- `UNTESTED`: any other generative configuration. n=80, d=9, k*=2, 3 Gaussian +
  3 Bernoulli + 3 Poisson columns, Poisson Y, w0=1.2, w=0.3 - one setting.
- `UNTESTED`: `single_bernoulli`, `single_poisson`, `all_bernoulli`, `all_poisson` under
  reduced Y (the trials-10 run kept only 4 conditions).
- `CONFOUNDED`: thinning Y reduces both the X-versus-Y information balance and the
  effective sample for the scalar parameters. w0 error rises from 0.012 to 0.030 for
  `per_column_all` and to 0.079 for `all_gaussian` between rates 1.0 and 0.1, so part of
  the RMSE(Z) gap may run through scalar-parameter error rather than through the X
  information channel.
- The test mask is fixed within a trial across rates (per runinfo). That is what makes the
  rate comparison paired, and is a design strength.

---

## 9. MovieLens count failure

Sources: `expfam/results/story_diagnostics/movielens_attribute_diagnosis_20260713_trials4.csv`
(44 rows = 11 conditions x 2 splits x 2 model seeds, 0 NaN),
`expfam/results/per_column_family/movielens_mixed_x_summary.csv` (24 rows),
and the pilot inputs under `expfam/data/movielens_pilot/`.

### 9.1 Reconstructed condition table (recomputed, fit-matched)

| condition | mean test_y_ll | sd | paired delta vs genre_only | fits better | w0 | w | hc_AUC | x_rmse count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| genre_only | -3.4172 | 0.0172 | 0 | - | 3.4169 | 0.2718 | 0.9702 | - |
| genre_year | -3.4192 | 0.0084 | -0.0020 | 2/4 | 3.4167 | 0.2718 | 0.9705 | - |
| genre_avg_rating | -3.4195 | 0.0090 | -0.0022 | 1/4 | 3.4172 | 0.2716 | 0.9715 | - |
| **genre + count (raw, Poisson)** | **-3.7917** | 0.0573 | **-0.3745** | **0/4** | **3.1558** | **0.3304** | **0.9494** | 27.47 |
| genre + count (log, Gaussian) | -3.4151 | 0.0123 | +0.0021 | 2/4 | 3.4182 | 0.2715 | 0.9707 | 0.833 |
| genre + count (z-score, Gaussian) | -3.4156 | 0.0162 | +0.0016 | 2/4 | 3.4177 | 0.2716 | 0.9711 | 0.818 |
| rating_stats_only (3 cols, raw count Poisson) | -3.7791 | 0.0560 | -0.3619 | 0/4 | 3.1545 | 0.3311 | 0.9497 | 28.59 |
| mixed_percolumn (raw count, 22 cols) | -3.7879 | 0.0525 | -0.3707 | 0/4 | 3.1567 | 0.3305 | 0.9492 | 27.65 |
| mixed_percolumn (log count, 22 cols) | -3.4145 | 0.0142 | +0.0027 | 3/4 | 3.4177 | 0.2717 | 0.9712 | 0.833 |
| mixed_all_gaussian (22 cols raw, all Gaussian) | -3.4377 | 0.0180 | -0.0205 | 0/4 | 3.4399 | 0.2670 | 0.9710 | 71.06 |
| y_only | -3.4516 | 0.0268 | -0.0344 | 0/4 | 3.4382 | 0.2679 | 0.9689 | - |

Cross-checks (`VERIFIED`, fit-matched):

- `mixed_percolumn_raw` - `genre_count_raw_poisson` = **+0.0037** (sd 0.0073)
- `mixed_percolumn_raw` - `rating_stats_only` = **-0.0088** (sd 0.0060)
- `genre_log_count_gaussian` - `genre_zscore_count_gaussian` = **+0.0005** (sd 0.0049)

The entire 22-column degradation is therefore attributable to the single raw-count Poisson
column, and **log and z-score are interchangeable as remedies**.

The mixed-X pilot reproduces the same structure independently: `mixed_percolumn` -0.3917
and `rating_stats_only` -0.3930 against genre_only, with the same w0/w signature
(3.412 -> 3.151 and 0.273 -> 0.331).

### 9.2 Data-side facts (recomputed from `expfam/data/movielens_pilot/`)

Measured quantities - `VERIFIED`:

- `ratings_count` over the n=100 subset: mean **154.38**, sd 30.86, min 73, median 162,
  max 200 (capped). Matches `attribute_stats` in the runinfo exactly.
- var/mean of `ratings_count` = 6.17, so the count column is itself over-dispersed
  relative to Poisson.
- Genre base rates: 19 columns, mean 0.131, max 0.36, so `A'' = p(1-p) <= 0.2304` per
  column and **1.9015 summed over all 19 columns** at the marginal rates.
- Four genre columns are identically zero over this subset (`p = 0`), a separate
  degenerate-support issue for a no-intercept Bernoulli column (§12.4).
- Poisson has `phi = 1` fixed; Gaussian columns get `phi = sigma_hat_l^2` updated by MLE
  each M-step (`model_dual_expfam_percolumn.py`, `calc_sigma` / `_x_weight_vector`).

Quantities computed from those - `DERIVED`, not measured:

- Reference points for reading `x_rmse_count_raw` (`VERIFIED` from the data): a
  constant-mean predictor gives RMSE **30.71**, a zero predictor gives **157.40**.
  The observed value under Poisson is **27.47**, i.e. *better than the constant-mean
  predictor*, so the Poisson arm genuinely tracks the column. The observed value under
  `mixed_all_gaussian` is **71.06**, i.e. between the two references: the Gaussian arm
  tracks the column only partially and leaves most of the baseline in the residual, which
  the estimated `sigma_hat^2` then absorbs.
- If the Poisson count column is fitted near its marginal mean - which the 27.47 figure
  supports - its curvature weight is `A'' = mu ~ 154.4`, i.e. **about 81.2x the summed
  curvature weight of all 19 genre columns** - before `f_l f_l^T` is taken into account.
- Under Gaussian the same column gets `A''/phi = 1/sigma_hat^2`. Treating 71.06 as a proxy
  for `sigma_hat` gives a weight of order `2.0e-4`, roughly `7.8e5` times smaller than the
  Poisson weight. On that reading **Gaussian does not fit the count column so much as
  discard it.** `sigma_hat` was not persisted, so this is an order-of-magnitude argument,
  not a measurement.

### 9.3 Candidate causes

| # | candidate | status | basis |
|---|---|---|---|
| A | X intercept / baseline absent | **UNTESTED** | No condition with an intercept has ever been run in this repository. The transform results are consistent with an intercept explanation but do not test it: `mixed_all_gaussian` keeps a raw mean of 154 with no intercept and loses only 0.021, which shows "no intercept" is not sufficient on its own - but that contrast simultaneously changes the genre family, so it does not isolate the intercept either. |
| B | raw count scale | **CONFOUNDED** with A and C | Every condition that removes the scale problem also removes the baseline (log, z-score) and also changes the family (to Gaussian). No condition keeps genre Bernoulli while putting the raw count under Gaussian. |
| C | Poisson `A'' = exp(eta)` gives high curvature | **PARTIALLY_SUPPORTED** | Data-side weight ratio 81.2x against all genre columns (`VERIFIED`); the Gaussian arm with the same raw column and the same absence of an intercept loses 18x less (`VERIFIED`); Poisson has no dispersion parameter to absorb the mismatch, Gaussian does (`VERIFIED` from code). The step from "large curvature weight" to "dominates the Z update" is `DERIVED`, not measured. |
| D | actual precision dominance `A''/phi f_l f_l^T` | **UNRESOLVED** | `F` was not saved for any of these runs (`*.npy` is git-ignored and the scripts do not persist it). `||f_l||` cannot be measured, so the actual per-block precision share cannot be computed from existing artifacts. **No value is imputed.** |
| E | block / column imbalance (1 count column vs 19 genre columns) | **CONFOUNDED** with C and D, and weakened | `rating_stats_only` uses only 3 columns and degrades by the same amount (-0.362), so column *count* imbalance is not the operative variable; the identity of the column is. |
| F | `ratings_count` information content | **PARTIALLY_SUPPORTED as a negative** | When the same variable is entered in a form the model can absorb (log or z-score Gaussian) the paired delta is +0.0021 / +0.0016, i.e. no measurable gain. The count carries essentially no usable information about held-out Y beyond genre, in either representation. F9 points the same way for genre itself. |
| G | metadata leakage / shared source | **CONFOUNDED, and it cuts the other way** | `mean_rating` and `ratings_count` are computed from the full `u.data` before the pair split, and Y comes from the same log (runinfo `leak_caveat`). Leakage would bias these conditions to look *better*, yet they look worse. Leakage therefore cannot explain the degradation, but it does invalidate the conditions as generalization evidence. |
| H | optimizer instability | **NOT_SUPPORTED** | 0 NaN in 44/44 fits; sd within the raw-count conditions is 0.052-0.057, larger than elsewhere but an order of magnitude below the effect; the w0/w shift is systematic (4/4 in the same direction), not erratic. |
| I | legacy numerical clipping | **NOT_SUPPORTED at convergence, UNRESOLVED during EM** | The clip is `[-20, 10]`, i.e. `mu` up to 22026. The count column needs `ln(154.4) = 5.04` on average and at most `ln(200) = 5.30`. The observed `x_rmse_count_raw = 27.47` is consistent with a fitted mean near the data mean, i.e. an interior eta. Activation during EM iterations was never recorded (neither script requests `compute_clip_diagnostic`), so it cannot be excluded outright. §12.3. |
| J-Y | **Y-side** Poisson misspecification / overdispersion of the co-rating count | **NOT_SUPPORTED as an explanation of F3** | The marginal var/mean of Y is 9.89, but the plug-in posterior predictive check reproduces it (rep_mean 9.79, p = 0.15), and the conditional Pearson dispersion is 1.13 at k=3 and 0.76 at k=5. On strict held-out at k=3 NB improves test ll by only +0.020 (6/6). On synthetic NB-Y data NB is *worse* for Z at every r (§11). **Decisively: Y is byte-identical across all 11 conditions of this experiment, so no property of Y can produce a between-condition difference.** |
| J-X | **X-side** overdispersion of `ratings_count` relative to the Poisson assumed for it | **UNTESTED, and CONFOUNDED with C** | `VERIFIED`: `ratings_count` has var/mean = **6.17** over the n=100 subset, so a Poisson likelihood assumes an sd of `sqrt(154.4) = 12.4` where the data show 30.9. `DERIVED`: since the X-side weight is `A''/phi`, a dispersion-matched count family would carry `mu/phi = 154.4/6.17 = 25.0` instead of 154.4 - i.e. **X-side overdispersion accounts for a factor of about 6.2 of the 81.2x weight ratio in §9.2, leaving about 13.2x** attributable to baseline/curvature. **No experiment in this repository has ever compared Poisson-X against a dispersion-aware count family for X.** Critically, every remedy that worked in F4 (log-Gaussian, z-score-Gaussian) *also* fixes the X dispersion mismatch, because Gaussian estimates `sigma^2` from the data. So F4 is equally consistent with a J-X explanation as with an intercept or a scale explanation. |

### 9.4 What must not be concluded

- **Do not conclude from "log-Gaussian fixed it" that the missing intercept is the cause.**
  z-score-Gaussian fixes it equally well (+0.0005 apart), and z-scoring does not stabilize
  the count variance - it only recentres and rescales. Both remedies change three things
  at once: baseline, scale, and family/dispersion.
- **Do not call the log-count condition an improvement.** +0.0021 against a fit-to-fit sd
  of 0.012-0.017, with 2/4 fits better, is "indistinguishable from genre_only".
- **Do not treat any of this as generalization evidence.** The leakage caveat in the
  runinfo is binding.
- Effective independent replication is **2 data splits**, not 4.
- **The two experiments are not independent replications of the data.** The mixed-X pilot
  and the attribute diagnosis use different seed bases (92000/93000 versus 102000/103000)
  and different scripts, but the **same n=100 movie subset** built by
  `prepare_movielens_data.py` and stored in `expfam/data/movielens_pilot/`. They replicate
  the fitting procedure, not the sample. A different subset of MovieLens has never been
  tried. `VERIFIED` from the runinfo and the shared `.npy` inputs.

---

## 10. Noise attribute evidence

Source: `expfam/results/per_column_family/noise_check_summary.csv` (18 rows = 6 conditions
x 3 trials, 0 NaN). Nine informative columns are always present; noise columns are drawn
independently of Z and their family is always specified correctly.

| condition | noise cols | family | paired delta RMSE(Z) vs no_noise, per trial | mean | worse |
|---|---:|---|---|---:|---:|
| gauss_noise3 | 3 | gaussian | -0.0004, **+0.0303**, -0.0006 | +0.0098 | 1/3 |
| gauss_noise6 | 6 | gaussian | -0.0013, -0.0003, -0.0001 | -0.0005 | 0/3 |
| gauss_noise12 | 12 | gaussian | +0.0070, **+0.0300**, -0.0009 | +0.0120 | 2/3 |
| bern_noise3 | 3 | bernoulli | +0.0002, +0.0009, +0.0002 | +0.0004 | 3/3 |
| pois_noise3 | 3 | poisson | +0.0014, +0.0015, -0.0001 | +0.0010 | 2/3 |

Per-trial dose response for Gaussian noise (0 -> 3 -> 6 -> 12 columns):

```
trial 0:  0.2077  0.2073  0.2064  0.2147     monotone increasing?  NO
trial 1:  0.2273  0.2576  0.2270  0.2573     monotone increasing?  NO
trial 2:  0.2352  0.2347  0.2351  0.2344     monotone increasing?  NO
```

Findings:

- `CONTRADICTED`: a monotone dose response. **0 of 3 trials** show one. Trial 1 is bimodal
  (bad at 3 and at 12 columns, fine at 6), which is the signature of an occasional poor
  local optimum, not of a per-column noise penalty.
- `OBSERVED`: the entire mean effect is one trial. Excluding trial 1, the largest mean
  effect of any noise condition is +0.0035.
- `VERIFIED`: Bernoulli and Poisson noise are harmless at this scale (max +0.0015).
- `UNTESTED`: whether a systematic effect exists at larger trial counts. With 3 trials and
  a between-trial spread of about 0.03, this design cannot detect an effect smaller than
  roughly 0.03 in RMSE(Z).

**There is no evidence here for block weighting, column weighting, regularization, or
attribute selection.** The existing summary report's phrasing ("worsening is seed
dependent", "dose response cannot be established with 3 seeds") is accurate; this audit
adds that the non-monotonicity holds in every individual trial, which is a stronger
negative than "cannot be established".

### 10.1 The other half: informative attributes

Source: `expfam/results/per_column_family/attribute_ablation_summary.csv` (15 rows =
5 steps x 3 trials, 0 NaN). Paired against `y_only` within trial:

| condition | columns | mean delta RMSE(Z) vs y_only | relative | improved |
|---|---:|---:|---:|---:|
| bern_only | 3 | +0.0008 | +0.3% | 1/3 |
| bern_gauss | 6 | **-0.0638** | **-21.6%** | 3/3 |
| bern_gauss_pois | 9 | -0.0663 | -22.5% | 3/3 |
| bern_gauss_pois_noise3 | 12 | -0.0644 | -21.8% | 3/3 |

Step-by-step: `bern_only -> bern_gauss` is -0.0647 (3/3), `bern_gauss ->
bern_gauss_pois` is -0.0025 (2/3), `bern_gauss_pois -> +noise3` is +0.0020 (1/3).

`SUPPORTED`: essentially the whole gain comes from **one** informative block (the Gaussian
one). Adding the Bernoulli block does nothing (+0.3%), adding the Poisson block adds
almost nothing (-0.0025), and adding noise costs almost nothing (+0.0020). Taken together
with §10, the picture is that **attribute value in this generator is concentrated in one
block, and the model is close to indifferent to everything else** - which is the same
message as F1 and further limits how much "joint integration of heterogeneous blocks" can
be claimed. 3 trials only.

---

## 11. All-Gaussian robustness

This is the most important counterargument to the per-column line and is treated as such.

### 11.1 The observation, in three independent settings

`VERIFIED`

1. **Synthetic mixed X, dense Y** (`single_vs_joint`, 3 trials, trial-matched):
   `all_gaussian - per_column_all` = **-0.0004** RMSE(Z) and **+0.0012** test_y_ll.
   Forcing all nine columns to Gaussian is, within this evidence, as good as specifying
   every column correctly - for Z and for Y. It is *not* as good for X reconstruction:
   `x_rmse_bern` 0.679 vs 0.450 and `x_rmse_pois` 2.555 vs 1.394.
2. **MovieLens** (`movielens_attribute_diagnosis`, 4 fits): `mixed_all_gaussian` loses
   only 0.021 test ll against genre_only while carrying the same raw count column that
   costs 0.374 under Poisson.
3. **Synthetic NB-Y** (`poisson_misspecification`, 5 paired trials per r):

| r_true | poisson RMSE(Z) | nb_oracle RMSE(Z) | paired delta | poisson test ll | nb_oracle test ll | paired delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.3801 | 0.4411 | **-0.0610 (poisson better 5/5)** | -3.3772 | -2.6173 | -0.7599 (nb better 5/5) |
| 5 | 0.2981 | 0.3367 | **-0.0386 (5/5)** | -2.7132 | -2.4797 | -0.2335 (5/5) |
| 20 | 0.2521 | 0.2679 | **-0.0158 (5/5)** | -2.3594 | -2.3232 | -0.0362 (5/5) |
| inf | 0.2362 | (n/a) | - | -2.2298 | - | - |

A **correctly specified** NB is beaten on latent-space recovery by a **misspecified**
Poisson in 15 of 15 paired comparisons, while losing on held-out likelihood in 15 of 15.

### 11.2 Candidate explanations

| explanation | status | basis |
|---|---|---|
| Gaussian quasi-likelihood robustness | `PARTIALLY_SUPPORTED` | Consistent with 1 and 2, but does not explain 3, where the robust-looking model is Poisson, not Gaussian |
| Estimated dispersion auto-downweights badly scaled or badly specified blocks | `PARTIALLY_SUPPORTED` | The mechanism is `VERIFIED` from code (Gaussian X columns get `phi = sigma_hat_l^2` updated by MLE; Bernoulli and Poisson have `phi = 1` fixed). Its **magnitude** on MovieLens is only `DERIVED`: `sigma_hat` was never persisted, so the `1/sigma_hat^2 ~ 2.0e-4` figure in §9.2 rests on using `x_rmse_count_raw = 71.06` as a proxy. Treat it as an order-of-magnitude argument, not a measurement. Also consistent with Gaussian being safe for Z yet bad for X reconstruction (that part is `VERIFIED`: `x_rmse_bern` 0.679 vs 0.450, `x_rmse_pois` 2.555 vs 1.394) |
| Y is strong enough that X hardly matters | `SUPPORTED at dense Y, CONTRADICTED at sparse Y` | At rate 1.0 all X variants lie within 0.011; at rate 0.1 the gap is 0.426 (§8). Regime-specific |
| X reconstruction and Z recovery are different objectives | `SUPPORTED` | `single_vs_joint` shows Gaussian forcing costs 1.5-1.8x in per-block X RMSE while costing nothing in RMSE(Z); the NB result shows the same split in the opposite direction (better density, worse Z) |
| Finite sample | `UNTESTED` | n=80 and n=100 in the relevant experiments; no n sweep exists for these comparisons |
| Family mismatch acts weakly along the Z directions | `UNRESOLVED` | This is the geometric restatement of the previous two and has not been measured |

### 11.3 The unifying hypothesis, stated as a hypothesis

`PARTIALLY_SUPPORTED`, `DERIVED`, **not** established: within this model class the
exponential family of a block acts on Z mainly through the **implied precision weight**
`A''(eta)/phi` it attaches to that block, and only secondarily through distributional
correctness. Under that reading:

- a family whose dispersion is **estimated** (Gaussian) self-attenuates when the block
  fits badly, which makes it safe for Z and bad for reconstruction;
- a family whose dispersion is **fixed at 1** (Bernoulli, Poisson) cannot self-attenuate,
  so a high-mean Poisson column injects a very large precision block;
- a family with a **larger** assumed variance (NB) down-weights its own residuals and
  therefore recovers Z less sharply even when it is correct.

All three bullets are consistent with all three settings in §11.1. What is missing to
promote this from hypothesis to finding is a **measurement of the actual per-block
precision contribution** - which is candidate C in §16 and is currently `UNRESOLVED`
because F was never saved.

**Consequence for the research claim, stated by regime.** The relevant comparison is
contrast A of §8.2 - `all_gaussian` versus `per_column_all` on the identical 9 columns.

- **Dense Y**: the claim "specifying the correct family per column improves Z estimation"
  is **not supported**. It is `CONTRADICTED in sign` in `single_vs_joint` (-0.0004, 3
  trials) and worth only +0.0114 in the 10-trial run.
- **Sparse Y**: the same claim **is supported** in this generator: contrast A reaches
  **+0.4261 RMSE(Z) at `y_obs_rate = 0.1`, 10/10 trials**.

So all-Gaussian robustness is a **dense-Y phenomenon**, not a general one. Any statement
about per-column family assignment must carry the Y-information regime with it, and any
statement about all-Gaussian robustness must do the same. Both directions of this were
partly obscured in the earlier draft of §8.2 and are corrected here.

---

## 12. Numerical defects vs structural failures

Three things are routinely conflated in this repository and are separated here.

### 12.1 The legacy numerical defect (CLASS I)

`VERIFIED` from `reports/per_column_family/per_column_math_code_audit_20260821.md` and
from the code:

- **PC-001 (HIGH)**: in `expfam/src/experimental/model_dual_expfam_percolumn.py` the
  Poisson X path clips `eta` to `[-20, 10]` in the mean, the curvature and the likelihood,
  but the score and precision use the post-clip residual without the clip derivative. At
  `eta=11.5, x=3` the actual objective's finite-difference score and negative Hessian are
  both `0`, while the implementation returns `-22023.4658` and `22026.4658`.
- **PC-002 (MEDIUM)**: the Bernoulli X likelihood floors probabilities to
  `[1e-10, 1-1e-10]` and the precision applies an independent `1e-8` floor, so the same
  objective/derivative mismatch appears in the extreme logit tails.

### 12.2 Lineage status of PC-001 / PC-002

This must be stated precisely. A blanket "the per-column prototype has an unresolved HIGH
finding" is **wrong**.

| lineage | PC-001 / PC-002 status |
|---|---|
| `DualExpFamLSMPerColumn` (legacy per-column) | **Confirmed present.** The clip `[-20,10]` and the floors are still in the file at `e132bed` |
| `DualExpFamLSMConsistent` / `DualExpFamLSMPerColumnConsistent` (`objective_consistent_numerics.py`, `model_dual_expfam_consistent.py`) | **Resolved for Bernoulli and Poisson.** Both files were read in this audit: there is no `clip`, no probability floor and no independent curvature floor on those two paths; the sigmoid is sign-partitioned, `poisson_mean` guards on `log(np.finfo(dtype).max)`, and mean / curvature / log-likelihood are all built from the same helpers. `VERIFIED` here. The accompanying test results (9/9 PASS, finite-difference score error 1.75e-06 at eta=11.5) are quoted from `reports/per_column_family/objective_consistency_fix_20260821.md` and **were not re-run in this audit** |
| Every experiment listed in §3 | **Ran on legacy numerics.** `numerics_mode` did not exist yet |
| The consistent lineage | **Never used in a claim-bearing or diagnostic experiment.** Deterministic smokes only |

So: the defect is real, it is fixed forward, the fix is unvalidated in any experiment, and
all existing evidence predates it. These are four separate statements and none of them
implies another.

### 12.3 Did the defect cause F3?

`NOT_SUPPORTED at convergence`, `UNRESOLVED during EM`.

- The upper clip at `eta = 10` corresponds to `mu = 22026`. The MovieLens count column
  needs `ln(154.38) = 5.04` on average and at most `ln(200) = 5.30`. `VERIFIED`.
- The fitted `x_rmse_count_raw = 27.47` is **below** the RMSE of a constant-mean predictor
  (30.71) and far below that of a zero predictor (157.40), so the Poisson arm is fitting
  the column at roughly its own scale, which puts `eta` near `ln(154)` and inside the clip
  interval. `DERIVED` from `VERIFIED` inputs.
- Y is Poisson with `w0 ~ 3.42`, `w ~ 0.27` and a mean co-rating count of 45.2, also
  interior. `VERIFIED`.
- Clip activation during EM iterations was never recorded: neither
  `tools/research_audit/run_movielens_attribute_diagnosis.py` nor
  `tools/research_audit/run_movielens_mixed_x_percolumn.py` requests
  `compute_clip_diagnostic`, and the diagnostic itself is documented as post-hoc on the
  final estimate only. `VERIFIED`. Transient activation therefore cannot be excluded -
  `UNRESOLVED`.

**Conclusion: F3 is a CLASS II phenomenon, not a manifestation of the CLASS I defect.**
Replacing the numerics would not be expected to remove F3.

### 12.4 A separate CLASS I exposure that has never been examined

`OBSERVED` (data), `UNTESTED` (activation): four of the 19 MovieLens genre columns are
identically zero over the n=100 subset (`p = 0`). Under a no-intercept Bernoulli column
the likelihood is maximized by driving `eta -> -infinity`, i.e. by inflating `||f_l||`
without bound, which is exactly the regime where PC-002's floors bite. This is present in
**every** genre-containing condition equally, so it cannot explain the differential count
failure - but it is an independent, untested reason why a no-intercept X model may be
poorly posed on this dataset, and it should not be folded into the count argument.

---

## 13. Evaluation limitations (CLASS III)

These must not be addressed by changing the model.

1. **Leakage in the MovieLens attribute experiments.** `mean_rating` and `ratings_count`
   are computed from the full `u.data` before the pair split; Y is built from the same
   log. Recorded in the runinfo `leak_caveat` and in the design memo. These runs are valid
   as *cause isolation*, invalid as *generalization evidence*. `VERIFIED`.
2. **Effective replication.** "4 trials" on MovieLens means 2 splits x 2 model seeds. The
   split-level spread for `genre_count_raw_poisson` is 0.098 while the model-seed spread
   within a split is about 0.02: the design is **split-limited**. `VERIFIED`.
3. **Dense Y hides X contributions.** F1 and §8 together show that any experiment run at
   `y_obs_rate = 1.0` in this generator has an effect ceiling of about 0.011 RMSE(Z). Any
   future comparison of X-side variants at dense Y will be uninformative by construction.
4. **Criterion naming.** The criterion is `-2 Q_strict + p ln n` with `Q_strict` an MC
   approximation of the EM Q function, not the observed-data marginal likelihood. It must
   be called a **Q-based complete-data / ICL-type criterion**, never Schwarz BIC. The
   names `calc_bic_dual`, `calc_bic_exp` and the CSV column `BIC` stay unchanged (KI-010).
5. **The criterion is not comparable across families.** `VERIFIED` in §5.4: the collapsed
   `all_bernoulli` condition attains the smallest criterion of all nine conditions.
6. **Cora k selection is not only a penalty problem.** Recomputed from
   `cora_balanced_k_sweep_summary.csv` with n=280 and `ln n = 5.634790`. `BIC` reproduces
   `-2 Q + p ln n` to `0.00e+00` and `p = kd - k(k-1)/2` exactly for every k:

   | k | p | Q_strict (mean) | -2 dQ vs k=1 | penalty increase vs k=1 | AUC | AP | NMI |
   |---:|---:|---:|---:|---:|---:|---:|---:|
   | 1 | 50 | -7265.10 | 0.00 | 0.00 | 0.604 | 0.035 | 0.052 |
   | 2 | 99 | **-7212.06** | -106.08 | 276.10 | 0.792 | 0.115 | 0.210 |
   | 3 | 147 | -7232.18 | -65.84 | 546.57 | 0.869 | 0.195 | **0.308** |
   | 4 | 194 | -7357.16 | +184.11 | 811.41 | 0.890 | 0.239 | 0.258 |
   | 5 | 240 | -7526.97 | +523.74 | 1070.61 | 0.906 | 0.269 | 0.257 |
   | 6 | 285 | -7708.04 | +885.88 | 1324.18 | **0.913** | **0.287** | 0.245 |

   `Q_strict` **peaks at k=2 and then falls**, although the models are nested. So **even a
   zero penalty would not select k=6**; it would select k=2. The penalty is large, but the
   fitted objective is also failing to improve. This is `VERIFIED`, and it means KI-011's
   framing ("the penalty is too large at low density") is incomplete. The additional
   mechanism - optimization, MC noise, or the sequential Laplace approximation - is
   `UNRESOLVED`.

   **F8 is not a per-column phenomenon and must not be merged with one.** The Cora runs
   use a single scalar `family_x = "bernoulli"` with `DualExpFamLSMFixed`; no per-column
   model, no mixed families and no count column is involved. Any statement that couples
   "the criterion misbehaves" to "heterogeneous X families" is unsupported.
7. **A registry annotation disagrees with its own primary artifact.** `VERIFIED`:
   `expfam/results/fixed_official/exp2/fixed_exp2_n_sweep_improvement.csv` records relative
   improvements of **49.3% / 41.2% / 58.6%** for scenarios A / B / C, and recomputation
   from `fixed_exp2_n_sweep_summary.csv` reproduces exactly those values. The
   `EXPERIMENT_REGISTRY.md` note for that row states "A:-40%, B:-17%, C:-62%", citing
   `reports/real_data_experiment_plan.md` §2, which indeed carries those figures. Every
   other number in that same plan table (Exp1 10/10, Exp3 -22.5%, Exp4 4.34x / 9.04x /
   40.37x) reproduces exactly, so the discrepancy is confined to the Exp2 row. The
   possibility that the plan quoted medians rather than means was checked and rejected:
   medians give -47.9% / -42.1% / -59.0%, which is not 40 / 17 / 62 either. **Nothing was
   edited by this audit**; a dated, append-only forward correction is proposed as a
   separate ranked action in §17.
8. **MovieLens attribute integration is inconsistent between protocols.** F9: the shared-Z
   ablation at k=5 finds `proposed_XY` worse than `y_only_fix_x` by 0.039 test ll (X helps
   3/6), while the attribute diagnosis at k=3 finds `genre_only` better than `y_only` by
   0.034 (4/4). Different k, script and evaluation. `CONFOUNDED`.

---

## 14. Causal-separation matrix

| mechanism | evidence for | evidence against | confounded with | isolated by an existing experiment? | smallest experiment that would isolate it | possible modification |
|---|---|---|---|---|---|---|
| **intercept / baseline absent** | 4 zero-variance genre columns force `||f||` growth (§12.4); a mean-154 column is unreachable from `f^T z` with `E[z] = 0` (`DERIVED`) | `mixed_all_gaussian` has no intercept, keeps the raw mean, and loses only 0.021 | scale/centering, Poisson curvature, family dispersion | **NO** | A (baseline stress at fixed latent signal), then B (intercept x representation factorial) | X column offset `eta = mu_l + f_l^T z_i` |
| **scale / centering** | log and z-score are equally effective (+0.0005 apart) and both change centering and scale | z-score does not stabilize count variance yet works, so variance stabilization is not the operative part | intercept, family dispersion | **NO** | B | count preprocessing convention |
| **Poisson curvature `A'' = exp(eta)`** | data-side weight 154 against 1.90 for all genre columns = 81.2x; the Gaussian arm with the same column loses 18x less | the step to "dominates Z" is not measured | F norm, block imbalance, scale | partially (the Poisson and Gaussian arms differ by 18x) | C | Poisson offset/exposure; block weighting |
| **F norm / actual precision contribution** | none measured | none | everything above | **NO - F was never saved** | C | (measurement first) |
| **block imbalance (1 vs 19 columns)** | none | `rating_stats_only` uses 3 columns and degrades identically (-0.362) | column identity | **YES, and it argues against** | already isolated | block weighting (not motivated) |
| **attribute informativeness** | count in an absorbable representation gives +0.002, i.e. nothing; genre gives +0.034 at k=3 but -0.039 at k=5 (F9) | at sparse Y in synthetic data X clearly helps (§8) | leakage, Y density, k | partially | matched-protocol MovieLens re-run | attribute selection (not motivated) |
| **Y density** | §8, monotone across 4 rates, 10 paired trials | one generative configuration; scalar-parameter error also grows | generator design (dense F rows) | **YES for this generator** | E = Issue #27 as currently written, or another second configuration | none |
| **numerical clipping / floors** | PC-001 / PC-002 deterministic counterexamples | MovieLens etas are interior (`VERIFIED`); resolved forward in the consistent lineage | EM-transient activation never logged | **YES at convergence, NO during EM** | activation logging during EM | already fixed forward |
| **optimization / local optima** | noise trial 1 bimodal; `all_bernoulli` collapse in 1/3; `poisson_strict` k=5 max test RMSE 48.19 against a mean of 14.58; Cora Q non-monotone in k | 0 NaN everywhere; most runs stable | MC sampling, `scale_Z`, Adam schedule | **NO** | multi-restart Q comparison at fixed data and fixed k | optimization / convergence change |
| **`scale_Z`** | applied unconditionally (`em_runner.py` line 226); forces mean square 1 on all MC samples, which interacts with any block that wants a large `||z||` | no measured failure attributed to it | every scale mechanism above | **NO** | the non-destructive `apply_scale_z` ablation already designed in `reports/theory_audit/diagnostic_designs_20260719.md` §3 | make it switchable, default unchanged |
| **leakage** | runinfo `leak_caveat`; count and Y share `u.data` | would bias the affected conditions upward, yet they are worse | attribute informativeness | partially | train-only attribute construction | **none - this is CLASS III** |
| **Y-side overdispersion** | marginal var/mean 9.89 | PPC p = 0.15 reproduces it; conditional dispersion 1.13 / 0.76; NB beats Poisson by only 0.020 at k=3; NB is worse for Z at every r; and Y is identical across the F3 conditions | the k=5 Poisson divergence inflates NB's apparent gain | **YES** | already isolated | Y-side NB / dispersion (not motivated for Z recovery) |
| **X-side count overdispersion** | `ratings_count` var/mean = **6.17** (`VERIFIED`); a Poisson X likelihood therefore over-weights that column by about 6.2x relative to a dispersion-matched count family (`DERIVED`) | none - it has never been tested | Poisson curvature, baseline/scale, intercept; **and every F4 remedy fixes it too** | **NO** | an X-side arm that keeps the column raw and Bernoulli genre intact but fits it with a dispersion-aware count family (quasi-Poisson / NB-X), against raw Poisson-X | X-side dispersion parameter or quasi-likelihood weight for count columns |
| **k-selection criterion** | Cora argmin k=1 versus AP argmax k=6 | Wine and all three synthetic scenarios select correctly (10/10 x3) | Q non-monotonicity, sparsity, subset choice | partially | Q-vs-k optimization diagnostic | criterion change - **premature** |

---

## 15. Candidate model modifications

`JUSTIFIED_NOW` requires all five of: (1) a repeatedly observed concrete failure;
(2) the mechanism directly supported, or the main alternatives excluded; (3) a clear
mathematical role; (4) a before/after validation designable in advance; (5) not motivated
by "the original paper had it".

| # | modification | observed failure motivating it | mechanism addressed | evidence for | evidence against | alternative explanation | discriminating experiment already done? | status |
|---|---|---|---|---|---|---|---|---|
| 1 | **X column intercept** `eta = mu_l + f_l^T z_i` | F3; also the 4 degenerate genre columns | intercept / baseline | transform remedies work; a mean-154 column is unreachable from `E[z] = 0` | `mixed_all_gaussian` has no intercept and barely degrades | scale, Poisson dispersion, informativeness | **NO** | **NEEDS_DIAGNOSTIC_FIRST** (fails conditions 1 and 2; note also that a bias term in X was previously *removed* as an error - RESEARCH_MASTER §4 eq(2) - so re-adding it must be justified by evidence, never by the paper) |
| 2 | **count preprocessing convention** (declare that high-baseline counts are centred/scaled or given an offset) | F3 / F4 | scale / centering | log and z-score both restore parity, 4 fits, two experiments | rests on 2 independent splits; it is a protocol, not a model change | family dispersion | partially (F4) | **OPTIONAL_ENGINEERING** - acceptable as an explicitly declared and logged preprocessing convention, never as a silent default, and it must not be described as an improvement (+0.002 < sd) |
| 3 | **Poisson offset / exposure** `eta = log(o_i) + f_l^T z_i` | F3 | intercept / baseline for counts specifically | the principled version of 1 for count data | same gaps as 1 | same as 1 | **NO** | **NEEDS_DIAGNOSTIC_FIRST** |
| 4 | **attribute block weighting** | F3, F5 | block imbalance | none that survives §10 and §14 | `rating_stats_only` (3 columns) degrades as much as 22 columns; no dose response in 0/3 trials; no likelihood justification | Poisson dispersion explains F3 without weighting | yes, and it argues against | **NOT_JUSTIFIED** |
| 5 | **column weighting** | F5 | column imbalance | none | as above | as above | yes | **NOT_JUSTIFIED** |
| 6 | **F regularization** | inferred `||F||` growth | F norm | `DERIVED` only; `||F||` was never recorded in any run | no measured failure attributable to it | `scale_Z` already constrains Z, not F | **NO** | **FUTURE_WORK** (record `||F||` first) |
| 7 | **attribute selection** | F5 | informativeness | none | noise columns cause no systematic degradation | - | yes | **NOT_JUSTIFIED** |
| 8a | **Y-side dispersion / NB** | MovieLens Y var/mean 9.89 | Y-side overdispersion | +0.020 test ll at k=3 (6/6) | PPC p = 0.15; conditional dispersion 1.13 / 0.76; NB worse for Z in 15/15 paired synthetic comparisons; the k=5 gain comes from a Poisson divergence | latent structure already absorbs the marginal overdispersion | yes | **NOT_JUSTIFIED for Z recovery**; **FUTURE_WORK** for density prediction only |
| 8b | **X-side count dispersion** (dispersion parameter or quasi-likelihood weight on count X columns) | F3 | X-side overdispersion / over-weighting of a count column | `ratings_count` var/mean = 6.17 (`VERIFIED`), implying about a 6.2x over-weighting relative to a dispersion-matched count family (`DERIVED`); it is a live competing explanation for F3 that no condition separates from intercept, scale or curvature | none, because it has never been tried; the 6.2x factor leaves about 13.2x of the §9.2 weight ratio unexplained, so it is unlikely to be the whole story | intercept, scale, Poisson curvature - all confounded with it | **NO** | **NEEDS_DIAGNOSTIC_FIRST** - and note this is a *different* modification from 8a, on a different side of the model; do not report them under one verdict |
| 9 | **optimization / convergence change** | Cora Q non-monotone in k; `all_bernoulli` collapse 1/3; `poisson_strict` k=5 divergence | optimization | three independent `VERIFIED` signals | no diagnosis of which stage fails; a blind change risks breaking historical comparability | MC sampling, `scale_Z`, Adam schedule, Laplace approximation | **NO** | **NEEDS_DIAGNOSTIC_FIRST** - and this is the second-best-evidenced problem area in the repository after F3 |
| 10 | **`scale_Z` made switchable** (default unchanged) | none directly | scale confounding | it is an unconditional transformation of the posterior samples that deviates from the MCEM target (`diagnostic_designs_20260719.md` §3) and confounds every scale mechanism in §14 | no measured failure | - | **NO** | **OPTIONAL_ENGINEERING** - a default-preserving flag is a no-op for existing results and unlocks an ablation; the ablation itself is NEEDS_DIAGNOSTIC_FIRST |
| 11 | **k-selection criterion change** | F8 | criterion | criterion/metric disagreement on Cora | Q itself is non-monotone, so a criterion change alone would not fix it; the criterion selects correctly in 4 of the 5 settings tested | optimization failure at high k | **NO** | **NEEDS_DIAGNOSTIC_FIRST**, and it is CLASS III - do not fix an evaluation problem by changing the generative model |

---

## 16. Candidate diagnostic experiments A-E

### 16.1 The rubric (fixed before scoring)

Each candidate is scored 1-5 on seven criteria, independently of the others:

- **IG** information gain
- **DIR** directness to an actually observed failure
- **CSP** causal separation power - how many confounded mechanisms in §14 it separates
- **UNC** reduction of the specific open items listed below
- **IND** independence from model modification (5 = needs none, 1 = needs the very
  modification under evaluation)
- **AMB** freedom from ambiguous-result risk (5 = both outcomes are interpretable)
- **THE** thesis relevance

**Gating rule, declared before scoring:** under the principle this phase operates on
("ONLY THEN MODIFICATION"), a candidate that requires implementing the modification it is
meant to evaluate cannot be the *first* experiment, regardless of its total. It is
reported at its true score and marked gated.

Open items the candidates are scored against:

- U1 actual per-block precision `A''/phi f_l f_l^T` (F never saved)
- U2 whether an X intercept fixes raw-count Poisson
- U3 whether raw count under Gaussian **with genre still Bernoulli** degrades
- U4 whether baseline magnitude alone breaks a no-intercept model
- U5 whether clipping / floors ever activated during EM historically
- U6 whether the consistent lineage changes any conclusion
- U7 why `Q_strict` is non-monotone in k on Cora
- U8 whether attribute integration helps MovieLens at all (F9)
- U9 whether the sparse-Y interaction generalizes beyond one generative configuration

### 16.2 The candidates

**Source discipline (added 2026-08-21, PR #29 review).** Candidate E is scored against the
**actual current body of GitHub Issue #27**, retrieved with `gh issue view 27`, not against
the July 2026 design memo. An earlier version of this audit scored the memo by mistake;
that scoring is withdrawn and replaced below. The relevant contents of the actual issue are
summarized in §18.

**A - Poisson baseline stress (revised).** Synthetic, two generation factors, fitted with
the current no-intercept model and correct per-column families:

- baseline `alpha` in `mu_il = alpha * exp(f_l^T z_i)`, with `F` and `Z` generation held
  fixed across arms;
- generative dispersion of the count column: equidispersed Poisson versus over-dispersed
  (matching the observed `ratings_count` var/mean of 6.17), still **fitted** as Poisson.

Question it can answer: *under the current no-intercept model, does raising the Poisson
count baseline - and separately, does generative over-dispersion of that column - produce
systematic degradation, and how does it scale?*

**Question it cannot answer** (added after the PR #29 review): under a Poisson likelihood
`alpha` moves the mean, the variance and `A''` **simultaneously**, because
`A'' = mu = alpha * exp(f^T z)`. A therefore **cannot separate** "missing intercept" from
"Poisson curvature" from "mean-variance coupling"; those three are one knob in a
no-intercept Poisson model. The second factor does separate generative over-dispersion from
the baseline, since dispersion can be varied at fixed mean. Targets U4 and part of J-X.

**B - Intercept x representation x X-dispersion factorial (revised).** Arms: raw Poisson
without intercept (exists); raw Poisson **with intercept** (requires the intercept);
transformed Gaussian without intercept (exists); raw count with genre kept Bernoulli and
the count fitted by a **dispersion-aware count family** (requires that family); optionally
transformed Gaussian with intercept. Question: *is it the intercept, the
representation/scale, or the X-side dispersion?* Targets U2, U3 and J-X, and it is the
**only** candidate that can establish intercept causality.

**C - Curvature / precision block diagnostic.** Instrumentation, not a new model: persist
`F` and the converged `eta`, then compute per-block `A''(eta)/phi * f_l f_l^T` and its
trace / norm share of the total X precision, on the existing MovieLens condition set and on
synthetic mixed-X. Question: *does one count column actually dominate the Z update?*
Targets U1, and U5 if clip activation is logged during EM at the same time. It yields
**mediation / correlational** evidence: a large precision share constrains the mechanism
but does not by itself establish that the share is what causes the degradation.

**D - Noise dose-response.** Increase the number of noise columns with enough trials to
detect a systematic degradation. Question: *is block weighting or regularization actually
needed?* Targets the F5 null.

**E - Complementary blocks = Issue #27 as currently written.** Synthetic generator in which
the Bernoulli, Gaussian and Poisson blocks load mainly on latent dimensions 1, 2 and 3
respectively; `n=80`, `d=9`, `K_TRUE=3`, Poisson Y, `L=5`, `num_iter=8`, **10 trials**,
fixed 20% test set shared across all conditions in a trial, and **two Y-information levels
from the same trainable pool**: `y_obs_rate = 1.0` (dense negative control) and `0.1`
(sparse primary). Conditions: `y_only`, `single_bernoulli`, `single_gaussian`,
`single_poisson`, `per_column_all`, plus `all_gaussian` as a secondary misspecification
control. Every fit must use `numerics_mode="consistent"` and the mode must be verified in
the result. The generator is forbidden to clip Poisson `eta`, and observed min/max `eta`
must be recorded. Primary endpoint is **pre-specified**: paired RMSE(Z) after Procrustes at
`y_obs_rate = 0.1`, with primary contrasts `per_column_all` versus each single family and
versus `y_only`. Question: *when attribute blocks carry different pieces of the latent
structure, is there a measurable reason to estimate them jointly under one shared Z?*
Targets U9 and U6, and its pre-specified primary endpoint is exactly contrast B of §8.2.

### 16.3 Scores

Scores below are the **re-scoring** required by the PR #29 review. Every change is traced
to a specific finding rather than to a desired ordering.

| candidate | IG | DIR | CSP | UNC | IND | AMB | THE | total | gated? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **B** intercept x representation x X-dispersion | 5 | 5 | 5 | 5 | **1** | 3 | 4 | **28** | **YES** - requires the X intercept and a dispersion-aware X family |
| **E** complementary blocks (Issue #27, current body) | 4 | 4 | 3 | 4 | 5 | 4 | 4 | **28** | no |
| **A** Poisson baseline + dispersion stress | 4 | 5 | 3 | 3 | 5 | 3 | 4 | **27** | no |
| **C** curvature / precision diagnostic | 3 | 5 | 3 | 4 | 5 | 4 | 3 | **27** | no |
| **D** noise dose-response | 2 | 2 | 2 | 3 | 5 | 2 | 3 | **19** | no |

Changes from the earlier scoring, with their causes:

- **A: CSP 4 -> 3, UNC 4 -> 3, AMB 4 -> 3 (total 30 -> 27).** Cause: the PR #29 review
  point that `alpha` moves mean, variance and `A''` together under Poisson. A separates the
  baseline/curvature bundle from leakage, informativeness and block count - all of which
  were already excluded - but it does not separate anything *inside* the bundle. AMB falls
  because a **rising** curve is now ambiguous between three mechanisms; only a **flat**
  curve is unambiguous. CSP recovers from 2 to 3 because the added generative-dispersion
  factor does separate X-side over-dispersion from the baseline at fixed mean. DIR stays 5:
  it remains the closest synthetic analogue of the F3 configuration.
- **B: unchanged at 28, still gated.** The review reinforces its role: it is now the only
  candidate that can establish intercept causality, and it must additionally carry an
  X-dispersion arm. IND stays 1.
- **C: unchanged at 27.** The review's point that A2 is mediation evidence was already the
  stated basis for CSP 3.
- **E: 22 -> 28.** Causes, all from reading the actual issue body: it fixes
  `numerics_mode="consistent"` and forbids generator clipping, so it becomes the first
  empirical use of the consistent lineage (U6) - IG 3 -> 4, UNC 3 -> 4. It carries a
  **dense-Y negative control** and a **pre-specified primary endpoint** that is exactly
  contrast B of §8.2, plus dimension-wise RMSE under one shared Procrustes rotation as a
  mechanism diagnostic - DIR 3 -> 4, CSP 2 -> 3. It excludes the raw invalid-support arms,
  forbids post-hoc tuning of `w0`, `w` and the dominant weight, forbids BIC-based ranking,
  and requires a null to be reported as a null - AMB 3 -> 4. It supplies the second
  generative configuration that PATH 4 needs - THE 3 -> 4.
- **D: unchanged at 19.**

Two honesty checks on the re-scoring:

1. **Is this score-fudging to flip the ordering?** Every raise for E is traceable to a
   clause in the actual issue body that the earlier scoring simply had not read, and every
   cut to A is traceable to review FINDING 5. The counterfactual is explicit: had Issue #27
   still contained only the July memo, E would have kept roughly its old score and A would
   still lead.
2. **Is E still "designed to succeed"?** Partly, and that is why AMB is 4 and not 5. The
   generator is built so that the blocks are complementary. What lowers the risk is that
   the issue pre-registers the endpoint, requires the dense-Y control, and explicitly
   requires a null to be reported. The external-validity limitation remains and the issue
   itself lists it as a required LIMITATION.

### 16.4 Ranking

```
1. E   (28)                          <- highest ungated; run this one if only one runs
2. A   (27) / C (27)  - tie          <- both address the F3 branch, neither can close it
4. D   (19)
--  B  (28 raw, gated by IND = 1)    <- becomes the right experiment once C or A motivates it
```

**Tie-break, declared here:** A is listed before C because A produces new controlled data
*and* can carry C's instrumentation inside its own runs, whereas C only instruments
configurations that must be re-fitted anyway. This is a tie-break, not a score difference.

**Consequence of the re-scoring for the F3 branch.** With A's CSP reduced and B gated, the
F3 branch is now **structurally blocked**: no ungated candidate can separate intercept from
curvature from X-dispersion, and the candidate that can (B) requires exactly the
modifications whose justification is in question. This deadlock is itself a finding, and it
is a substantive reason why the next experiment is not on the F3 branch.

---

## 17. Ranked next actions

At most five. Each is a recommendation only; nothing here was executed. This list was
**re-ordered** after the PR #29 review; see §16.3 for the score changes that drove it.

### P0 - A1. Run Issue #27 as currently written (DIAGNOSTIC EXPERIMENT)

- **Question.** When attribute blocks carry different pieces of the latent structure, is
  there a measurable reason to estimate them jointly under one shared Z rather than fitting
  one attribute family at a time?
- **Current evidence.** F1 (dense-Y differences of about 0.011); §8.2 contrast B (+0.010 at
  dense Y, +0.083 at `y_obs_rate = 0.1`, 10/10) - the smallest of the three sparse-Y
  contrasts and the one E pre-registers as its primary endpoint; F2 (contrast A grows to
  +0.426 at rate 0.1, 10/10) established on **one** generative configuration.
- **Missing evidence.** U9 (a second generative configuration, here one deliberately built
  with complementary blocks) and U6 (**no experiment has ever used the consistent
  lineage**). Also missing: any evidence at all on `single_bernoulli` and `single_poisson`
  under reduced Y - the trials-10 run kept only four conditions.
- **Minimal design.** The design is already written in the issue and needs no change:
  10 trials x 2 Y-observation rates x 6 conditions = 120 fits, shared data and test split
  within a trial, `numerics_mode="consistent"` verified per fit, no clipping in the
  generator with min/max `eta` recorded, pre-specified paired RMSE(Z) endpoint at
  `y_obs_rate = 0.1`, dense-Y negative control, dimension-wise RMSE under one shared
  Procrustes rotation, and an explicit prohibition on ranking conditions by the criterion.
- **Decision enabled.** Whether PATH 4 survives a second, adversarially-controlled
  configuration; and whether the consistent lineage behaves at all in practice.
- **Risk.** The generator is constructed so complementary integration *should* help, so a
  positive result has limited external validity. The issue mitigates this with the dense-Y
  control and by requiring nulls to be reported. One residual design note this audit adds:
  E uses `K_TRUE = 3` while the existing sparse-Y evidence used `k* = 2`, so E differs from
  it in **two** respects (F structure and k) and a difference in outcome cannot be
  attributed to complementarity alone. Record this as a limitation; it does not warrant
  changing the design.
- **Before the F3 branch:** YES - the F3 branch is currently blocked (§16.4).

### P1 - A2. Per-block precision instrumentation (ALGORITHM VALIDATION)

- **Question.** What share of the X precision `sum_l A''(eta_il)/phi_l f_l f_l^T` does each
  attribute block actually contribute at convergence?
- **Current evidence.** The data-side ratio only (81.2x, of which about 6.2x is now
  attributed to X-side over-dispersion and about 13.2x to baseline/curvature - both
  `DERIVED`), with `||f_l||` unknown.
- **Missing evidence.** U1. `F` is not persisted by any of the relevant scripts, and
  `*.npy` is git-ignored.
- **Minimal design.** A diagnostic function plus persistence of `F` and the converged
  `eta`; no model change. Apply it to a re-fit of the existing MovieLens condition set, and
  to A6's runs at no extra cost.
- **Decision enabled.** Whether "precision-block dominance" graduates from
  `PARTIALLY_SUPPORTED` to a finding, or is falsified.
- **Risk.** A dominance measurement is **mediation / correlational** evidence: it
  constrains the mechanism but does not establish that the share is what causes the
  degradation, and it cannot distinguish intercept from curvature from X-dispersion. The
  report that uses it must say so. It also re-fits the same n=100 subset, so it adds fits
  without adding samples.
- **Before the F3 branch closes:** it is a prerequisite, not a conclusion.

### P1 - A3. Forward correction of the Exp2 registry annotation (CLAIM RESTRICTION)

- **Question.** Which numbers may be quoted for the fixed-lineage n sweep?
- **Current evidence.** Primary artifact: 49.3% / 41.2% / 58.6% (`VERIFIED`, two
  independent recomputations). Registry note and `reports/real_data_experiment_plan.md`
  §2: 40% / 17% / 62%. The median reading was checked and rejected (-47.9 / -42.1 / -59.0%).
- **Missing evidence.** None - this is settled.
- **Minimal design.** An append-only, dated forward-correction row in
  `EXPERIMENT_REGISTRY.md`, in the same style as its 2026-08-21 Phase 5a.1 section.
  **Do not edit the historical row and do not edit the dated plan document.**
- **Decision enabled.** Prevents an incorrect figure entering the thesis.
- **Risk.** None, provided the historical text is left intact.
- **Independent of everything else:** YES - it costs nothing and blocks nothing.

### P2 - A6. Poisson baseline + X-dispersion stress (DIAGNOSTIC EXPERIMENT)

- **Question.** Under the current no-intercept model, does raising the Poisson count
  baseline - and separately, does generative over-dispersion of that column at fixed
  mean - produce systematic degradation, and how does it scale?
- **Current evidence.** F3; the 81.2x data-side weight ratio; the Gaussian arm with the
  same raw column loses 18x less.
- **Missing evidence.** U4, and the generative-dispersion half of J-X.
- **Minimal design.** Synthetic mixed-X; hold `F` and `Z` generation fixed; cross a
  multiplicative count baseline over about four levels spanning `mu` from about 1 to about
  200 with two generative dispersion settings (equidispersed, and var/mean about 6.2);
  fit with correct per-column families and the current model; `numerics_mode="consistent"`;
  log clip/floor activation during EM. Report RMSE(Z), held-out Y ll, `w0`, `w`, per-block
  X RMSE, `||F||` and the per-block precision share.
- **Decision enabled.** A **flat** damage surface falsifies the whole baseline/curvature
  bundle at once. A **rising** surface does not identify which of intercept, curvature or
  mean-variance coupling is responsible - it only establishes that the bundle matters, and
  it hands the question to B.
- **Risk.** Precisely that asymmetry: only the negative outcome is unambiguous. Under
  Poisson, `A'' = mu = alpha * exp(f^T z)`, so `alpha` moves the mean, the variance and the
  curvature together.
- **Known confound to design around.** `scale_Z` forces the MC samples to mean square 1
  unconditionally (§4), so as the baseline rises the model can only respond by growing
  `||F||`, not `||Z||`. The response is therefore *mediated* by `scale_Z`. Record `||F||`
  per arm, and either run the `apply_scale_z` on/off ablation alongside or state explicitly
  that the measured surface is conditional on `scale_Z` being on.
- **Before #27:** NO - it now ranks below it.

### P2 - A4. Q-versus-k optimization diagnostic (ALGORITHM VALIDATION)

- **Question.** Why does `Q_strict` fall for k >= 4 on Cora when the models are nested?
- **Current evidence.** `VERIFIED` non-monotonicity (peak at k=2); `all_bernoulli` collapse
  in 1/3 trials; `poisson_strict` divergence at k=5 (max test RMSE 48.19 against a mean of
  14.58).
- **Missing evidence.** U7 - whether the cause is initialization / local optima, MC sample
  dependence, `scale_Z`, or the sequential Laplace approximation.
- **Minimal design.** Multiple restarts at fixed data and fixed k, comparing final
  `Q_strict`; the L sweep (5 / 10 / 20) designed in `diagnostic_designs_20260719.md` §5;
  the non-destructive `apply_scale_z` on/off ablation from §3 of the same memo.
- **Decision enabled.** Whether F8 is a criterion problem, an optimization problem, or
  both - and therefore whether a criterion change is even the right lever.
- **Risk.** Scope creep into a full MCEM study. Keep it to n <= 280 and one fixed dataset.

### Below the top five (recorded, not ranked)

- **Matched-protocol MovieLens attribute check.** F9 shows +0.034 (4/4) at k=3 in one
  protocol and -0.039 (3/6) at k=5 in another. One script, one evaluation, both k, genre-only
  X versus `y_only`, train-only attributes if feasible. Needed before **any** MovieLens
  attribute-integration claim, but it does not gate anything above.
- **X-side count dispersion probe.** This is an **arm of B**, not a separate experiment:
  raw count, genre kept Bernoulli, count fitted by a dispersion-aware family. It requires an
  X-side dispersion capability that does not exist today, so it inherits B's gate.

---

## 18. Issue #27 decision

**Decision: RUN NEXT.**

**Correction note (2026-08-21, PR #29 review).** An earlier version of this section
returned **REDESIGN**, based on the July 2026 design memo
`reports/story_diagnostics/story_diagnostics_next_plan_20260713.md`. That was an error of
sourcing: the memo is not the issue. The actual current body of GitHub Issue #27 was
retrieved with `gh issue view 27` and is materially different. **The earlier REDESIGN
verdict and all four of its stated defects are withdrawn.** What follows is decided from
the actual issue text.

### 18.1 The four earlier criticisms, checked against the actual issue

| # | earlier criticism | what the actual Issue #27 says | verdict |
|---|---|---|---|
| 1 | "No Y-density axis" | Two Y-information conditions from the same trainable pool, `y_obs_rate = 1.0` as an explicit **dense-Y negative control** and `0.1` as the **sparse-Y primary condition**, with the same fixed 20% test set in both | **WITHDRAWN - factually wrong** |
| 2 | "No pre-registered decomposition" | A **pre-specified primary endpoint** (paired RMSE(Z) after Procrustes at `y_obs_rate = 0.1`), pre-specified primary contrasts (`per_column_all` versus each of `single_bernoulli`, `single_gaussian`, `single_poisson`, `y_only`), an explicit sign convention, `all_gaussian` retained as a secondary misspecification control, and dimension-wise RMSE under **one shared** Procrustes rotation. It also forbids redefining the endpoint after seeing results | **WITHDRAWN** - the two contrasts §8.2 needed (A and B) are both pre-registered |
| 3 | "Raw forced-misspecification arms" | "Do **not** include raw `all_bernoulli` or raw `all_poisson` on the mixed continuous X as primary conditions... do not rerun them merely to make a dramatic comparison" | **WITHDRAWN - factually wrong** |
| 4 | "Numerics and trial count unspecified" | `numerics_mode="consistent"` mandated and verified per fit; **no clipping permitted in the generator**, with min/max `eta` recorded for Bernoulli X, Poisson X and Poisson Y; **10 trials**; 120-fit integrity checks enumerated | **WITHDRAWN - factually wrong** |

Nothing of substance survives from the earlier list. One **new, minor** observation
replaces it, and it is not a defect requiring redesign: Issue #27 uses `K_TRUE = 3` whereas
the existing sparse-Y evidence used `k* = 2`, so E differs from the trials-10 configuration
in **two** respects (complementary F structure *and* k). A difference in outcome therefore
cannot be attributed to complementarity alone. This should be recorded as a limitation in
E's own report.

The actual issue additionally contains protections the earlier assessment gave it no credit
for: post-hoc tuning of `w0`, `w` and the dominant weight is forbidden; a null or negative
result is declared acceptable and must be reported as such; "per-column is always better"
is forbidden; and the criterion is explicitly barred from being used to rank conditions -
which is independently correct, since §5.4 of this audit shows a collapsed misspecified
condition attaining the smallest criterion value of all.

### 18.2 Why RUN NEXT, from the re-scoring

- E is the **highest-scoring ungated candidate** (28; §16.3), and it reached that score
  through changes each traceable to a clause of the actual issue body.
- Its **pre-specified primary endpoint is exactly contrast B of §8.2** - the smallest and
  therefore least-established of the three sparse-Y contrasts, and the one on which the
  per-column claim is weakest. It is testing our own weak point, not a strong point.
- It supplies the **second generative configuration** that PATH 4 requires (U9), and it
  covers `single_bernoulli` and `single_poisson` under reduced Y, which no existing run does.
- It would be the **first empirical use of the objective-consistent lineage** (U6). §12.2
  records that the lineage exists, is tested at the unit level, and has never been used in
  an experiment. That gap does not close by itself.
- The competing branch is blocked. After review FINDING 5, no ungated candidate can
  separate intercept from curvature from X-dispersion on the F3 branch (§16.4), and the
  candidate that can (B) needs exactly the capabilities whose justification is in question.
  Waiting for the F3 branch would mean running an experiment whose positive outcome is
  known in advance to be ambiguous.

### 18.3 Why not DEFER, REDESIGN or REPLACE

- **DEFER** would require something above it that is un-blocked. There is not: A6 and C
  both sit on the blocked F3 branch, and A3 is a documentation action that blocks nothing.
- **REDESIGN** was the earlier verdict and is withdrawn; none of its four grounds survives
  contact with the actual issue text. The one residual observation (K_TRUE = 3 vs k* = 2)
  is a limitation to record, not a design fault to fix.
- **REPLACE** would require a better experiment answering the same question. None of A, B,
  C or D answers E's question at all; they address the F3 branch.

### 18.4 Guard against the two forbidden reasons

Neither forbidden reason was used. **Not** "run it because the issue already exists": E was
re-scored on the same seven criteria as everything else, and it was the actual issue text -
not its existence - that changed the score. **Not** "defer it because of positive-story
risk": that concern is real, it is priced into AMB = 4 rather than 5, and it is handled by
the issue's own dense-Y control and null-reporting requirement rather than by refusing to
run the experiment.

**Instruction implied.** Keep Issue #27 OPEN and run it as written, under its own stated
absolute-stop conditions. Add the `K_TRUE = 3` versus `k* = 2` note to its report's
LIMITATION section. Do not weaken the pre-specified endpoint after seeing results.

---

## 19. JUSTIFIED_NOW

**No model modification reaches JUSTIFIED_NOW.**

Applying the five conditions honestly:

| modification | (1) repeated concrete failure | (2) mechanism supported or alternatives excluded | (3) clear mathematical role | (4) before/after validation designable | (5) not "the paper had it" | verdict |
|---|---|---|---|---|---|---|
| X intercept | partly - 4 fits / 2 splits, one n=100 subset | **NO** - U2, U3, U4 and J-X all open, and after review FINDING 5 no ungated experiment can separate them | yes | yes | must be argued explicitly, since the X bias was previously removed as an error | fails (1), (2) |
| Poisson offset | as above | **NO** | yes | yes | yes | fails (1), (2) |
| block / column weighting | **NO** - §10 is a null | **NO** | weak - no likelihood justification | yes | yes | fails (1), (2), (3) |
| F regularization | **NO** | **NO** - `||F||` never recorded | yes | yes | yes | fails (1), (2) |
| attribute selection | **NO** | **NO** | yes | yes | yes | fails (1), (2) |
| Y-side NB / dispersion | **NO** - §9 candidate J-Y | **CONTRADICTED** | yes | yes | yes | fails (1), (2) |
| X-side count dispersion | partly - F3, 4 fits / 2 splits | **NO** - `UNTESTED`, and confounded with intercept, scale and curvature | yes | yes | yes | fails (1), (2) |
| optimization change | yes - three independent signals | **NO** - no diagnosis of which stage | not yet specified | yes | yes | fails (2), (3) |
| criterion change | yes - F8 | **NO** - Q itself is non-monotone | yes | yes | yes | fails (2) |
| `scale_Z` switch | no failure attributed | n/a | yes | yes | yes | not a fix; see below |

Two things **are** justified now, and neither is a model modification. They are listed so
that "nothing is justified" is not misread as "nothing should be done":

- **A3** (§17): the forward correction of the Exp2 registry annotation. Purely a claim
  restriction, backed by `VERIFIED` recomputation.
- **A policy, not a change:** every new per-column experiment should select
  `numerics_mode="consistent"` explicitly and log clip/floor activation, because the
  legacy defect is real (§12.1) and the consistent lineage already exists and is tested.
  This changes no code and no historical result.

Making `scale_Z` switchable with the current behaviour as the default is
`OPTIONAL_ENGINEERING`: it is a no-op for every existing result and it unlocks an ablation
that several mechanisms in §14 are confounded by. It is not a fix and must not be presented
as one.

**A structural note added after the PR #29 review.** The F3 branch cannot reach
JUSTIFIED_NOW by accumulating more un-gated evidence, because no un-gated experiment
separates the intercept from the Poisson curvature from the X-side dispersion (§16.4,
§23). Closing it requires accepting a *provisional* implementation of the X intercept and
of a dispersion-aware X count family purely in order to run B. That is a human scientific
decision about how much provisional implementation is acceptable, not a gap this audit can
close with more analysis. It is recorded here rather than disguised as a NEEDS_DIAGNOSTIC
that some future read-only step could satisfy.

---

## 20. NOT_JUSTIFIED_YET

Ordered by how much evidence would be needed to move them.

1. **X intercept / Poisson offset** - `NEEDS_DIAGNOSTIC_FIRST`, with the structural caveat
   in §19: no un-gated experiment can separate it from the Poisson curvature or from the
   X-side dispersion, so A6 and C can motivate B but cannot substitute for it. Note for
   the write-up: a per-column bias in X was **removed** as an error (RESEARCH_MASTER §4,
   eq(2): `N(w_0l + z_i^T w_l, sigma_l^2)` -> `N(f_l^T z_i, sigma_l^2)`). Re-introducing it
   must rest on the diagnostic evidence, and the report must say so, or it will read as
   reverting to the prior work's formulation.
2. **Block weighting / column weighting** - `NOT_JUSTIFIED`. The motivating observation
   (F5) is a null in every individual trial, and F3 does not require it: `rating_stats_only`
   uses 3 columns and degrades as much as the 22-column condition, so column-count
   imbalance is not the operative variable. Introducing weights would also further damage the likelihood interpretation
   of the criterion, which §5.4 shows is already fragile across families.
3. **F regularization** - `FUTURE_WORK`. Record `||F||` first; nothing currently measures
   it.
4. **Attribute selection** - `NOT_JUSTIFIED`. No systematic degradation from uninformative
   columns has been demonstrated.
5. **Y-side NB / dispersion** - `NOT_JUSTIFIED for Z recovery`; `FUTURE_WORK` for density
   prediction only, where the honest statement is +0.020 test ll at k=3. This verdict
   rests entirely on Y-side evidence and **says nothing about the X side**.
5b. **X-side count dispersion** - `NEEDS_DIAGNOSTIC_FIRST`. `ratings_count` is 6.17x
   over-dispersed relative to the Poisson assumed for it, no experiment has ever fitted a
   dispersion-aware count family to an X column, and every remedy that worked in F4 also
   removes the dispersion mismatch. It is a live competing explanation for F3 on the same
   footing as the intercept, and it must be an arm of whatever experiment settles F3.
6. **Optimization / convergence change** - `NEEDS_DIAGNOSTIC_FIRST` (A4). The evidence
   that *something* is wrong is good; the evidence about *what* is absent.
7. **k-selection criterion change** - `NEEDS_DIAGNOSTIC_FIRST`, and CLASS III. Do not
   attempt to fix an evaluation problem by changing the generative model.
8. **Promotion of the per-column prototype to the thesis method** - `NOT_JUSTIFIED`
   **today**, and explicitly re-openable after E. §5.2, §8.2 contrast A and §11.3 together
   mean the strongest defensible statement is conditional on the Y-information regime and
   rests on one generative configuration. Issue #27 supplies the second configuration; a
   positive result there would make a *conditional* method claim arguable, and the issue
   itself states that even then the prototype is not automatically promoted.

---

## 21. Candidate thesis paths

| path | evidence strength | novelty | unresolved issues | implementation required | experiment required | oral-defense strength | risk | feasibility |
|---|---|---|---|---|---|---|---|---|
| **1. X/Y exponential-family generalization** | **Strongest.** 180 + 180 + 180 + 550 rows of synthetic evidence with 10-30 trials, 10/10 k selection in three scenarios, misspecification ratios 4.34 / 9.04 / 40.37 reproduced exactly, Wine k=3, Cora 2.6-2.8x random | moderate - generalizing a fixed-family LSM | KI-001 hedge; criterion naming (KI-010); Cora Q non-monotonicity | none | none | **high** - every number reproduces from primary CSVs | low | already done |
| **2. + per-column heterogeneous X** | **Regime-dependent, and stronger than the earlier draft allowed.** Contrast A (identical 9 columns, family assignment only): -0.0004 at dense Y in `single_vs_joint` (3 trials) and +0.0114 in the 10-trial run, but **+0.4261 at `y_obs_rate = 0.1`, 10/10** (§8.2). MovieLens is negative and F9 questions attribute value there at all | high | PC-001 in the legacy lineage (fixed forward, never validated in an experiment); prototype status; one generative configuration | consistent-lineage validation | E (Issue #27) | **moderate** - the hostile question "is it better than forcing Gaussian?" answers "no at dense Y, clearly yes at sparse Y in one generator" | medium-high | possible after E |
| **3. + evidence-driven count/intercept refinement** | **Insufficient.** 4 fits / 2 splits on a single n=100 subset; mechanism `UNRESOLVED`; leakage caveat binding; and after review FINDING 5 the branch is structurally blocked (§16.4) | high if it works | U1-U4 and J-X all open | X intercept **and** an X-side dispersion capability | A6, C, then B | low today | high | not feasible this cycle |
| **4. Conditions under which attribute integration is effective (sparse Y)** | **Moderate, and better founded than the earlier draft stated.** 10 trial-matched trials, monotone across 4 rates, 10/10 at the sparsest rates. The earlier draft discounted it by netting contrast A against contrast B; that decomposition is withdrawn (§8.2). Remaining limitation: **one generative configuration** | moderate | U9 - needs a second configuration | none | **E = Issue #27, as currently written** | moderate-to-high | medium | feasible with exactly one more experiment, which is already designed |
| **5. Family generalization plus explicit diagnostic limitations** | **Strong**, because it is built from what is already `VERIFIED`, including the negative results | low as novelty, high as scholarship | none - the limitations are the content | none | none | **high** - it pre-empts hostile questions by stating the limit first | low | already done |

### Recommended path

**Path 1 as the backbone, framed with Path 5, with Path 4 as the primary extension and
Path 2 conditional on it.**

- Path 1 supplies the claims that survive recomputation without qualification.
- Path 5 supplies the honest boundary: at **dense** Y, forcing every column to Gaussian is
  as good for Z and Y and worse only for X reconstruction (§11); the criterion is Q-based
  and not comparable across families (§5.4, §13); on Cora the fitted Q is non-monotone in k
  (§13.6); MovieLens attributes did not help (§9, F9); and the F3 mechanism is
  `UNRESOLVED`.
- Path 4 is the extension the evidence actually points at, **conditional on E reproducing
  the sparse-Y interaction in a second generative configuration**. Its statement must carry
  the regime and the contrast: "on identical columns, per-column family assignment is worth
  about 0.43 RMSE(Z) when only 10% of training pairs are observed and about 0.01 when all
  are (one generator, 10 paired trials)".

**Correction relative to the earlier draft.** That draft demoted Path 4 on the strength of
an §8.2 "decomposition" that has since been withdrawn as non-causal (PR #29 review,
FINDING 1). With contrast A read as what it is - the per-column mechanism's own
same-column comparison - Path 4 and Path 2 are **stronger** than the earlier draft
concluded, though still regime-restricted and still resting on one generative
configuration. The correction, not a preference, produced this change.

Confidence: **moderate-to-high** for Path 1 plus Path 5 (every supporting number reproduced
from primary artifacts); **moderate** for Path 4 pending E; **low** for Path 3 today.

---

## 22. Final evidence-based recommendation

**Run exactly one experiment next: Issue #27 as currently written (candidate E, action A1
in §17), unchanged in design, with the `K_TRUE = 3` versus `k* = 2` note added to its
report's limitations.**

Why this one:

1. It is the **highest-scoring ungated candidate** (28; §16.3), scored on the same rubric
   as every other candidate, against the **actual issue text** rather than the superseded
   July memo.
2. Its pre-specified primary endpoint is **contrast B of §8.2** - the weakest and least
   established of the three sparse-Y contrasts. It tests the per-column claim where that
   claim is thinnest, not where it is strongest.
3. It closes the two most consequential gaps that no other candidate touches: a **second
   generative configuration** for the only positive result the per-column line has (U9),
   and the **first empirical use of the objective-consistent lineage**, which currently
   exists, passes unit tests, and has never been run in an experiment (U6, §12.2).
4. The competing branch is blocked. After review FINDING 5, `alpha` in A6 moves the mean,
   the variance and `A''` together, so a positive A6 result cannot separate intercept from
   curvature from mean-variance coupling; C yields mediation evidence only; and B, the one
   candidate that can separate them, requires the very intercept and X-dispersion
   capabilities whose justification is in question. Running an F3-branch experiment now
   means running one whose positive outcome is known in advance to be ambiguous.
5. It requires **no model modification** and no new dataset.

Do **not** implement the X intercept, a Poisson offset, an X-side dispersion parameter,
block weighting, column weighting, F regularization, attribute selection, Y-side NB, or a
criterion change. None of them reaches JUSTIFIED_NOW (§19), and the F3 branch cannot be
closed by any experiment that avoids them.

Alongside it, two zero-risk items: the A3 registry forward correction, and the policy that
every new per-column run selects `numerics_mode="consistent"` and logs clip/floor
activation - a policy Issue #27 already enforces for itself.

---

## 23. Decision tree

Two branches. They are independent, and only one of them is currently traversable.

```
BRANCH 1 - the per-column claim itself   [traversable now]
|
+-- What does the sparse-Y evidence actually say?   [3 named contrasts, 8.2]
|     A  same-column family assignment  : +0.011 (dense)  -> +0.426 (rate 0.1), 10/10
|     B  add 6 correctly specified cols : +0.010 (dense)  -> +0.083 (rate 0.1), 10/10
|     C  add 6 cols as the wrong family : +0.001 (dense)  -> +0.343 (rate 0.1), 10/10
|     A = B + C algebraically, but B and C are NOT orthogonal causal components.
|     Do not report "the benefit is mostly family correctness".
|
+-- Does it hold outside this one generator?
      -> UNKNOWN  [U9]   n=80, d=9, k*=2, dense random F rows, one family mix
      |
      +-- Run Issue #27 as written (candidate E, 28, highest ungated)
            pre-specified endpoint = contrast B at y_obs_rate = 0.1
            dense-Y arm = negative control
            numerics_mode = "consistent"   -> also closes U6
            |
            +-- E reproduces the interaction
            |     -> PATH 4 becomes claimable, regime-restricted, two configurations
            |     -> PATH 2 becomes arguable as a conditional method claim
            |
            +-- E does not reproduce it
                  -> the sparse-Y result is specific to the dense-random-F generator
                  -> PATH 4 is withdrawn; PATH 1 + PATH 5 stand alone
                  -> this is an acceptable, reportable outcome


BRANCH 2 - the MovieLens raw-count failure F3   [currently BLOCKED]
|
+-- Ruled out already
|     legacy Poisson clip     : eta = ln(154) = 5.04 vs boundary 10   [NOT_SUPPORTED at
|                               convergence; during-EM activation UNRESOLVED, never logged]
|     Y-side overdispersion   : PPC p = 0.15; conditional dispersion 1.13 / 0.76;
|                               and Y is identical across all 11 conditions   [NOT_SUPPORTED]
|     block-count imbalance   : rating_stats_only uses 3 columns, degrades by the same
|                               -0.362   [it is the column's identity, not the count]
|     leakage                 : would flatter these conditions; they are worse
|                               [cannot explain it, but voids them as generalization evidence]
|     optimizer instability   : 0 NaN in 44/44, systematic 4/4 w0/w shift   [NOT_SUPPORTED]
|
+-- Still confounded with one another, and NOT separated by any existing condition
|     missing X intercept
|     raw baseline / scale
|     Poisson curvature A'' = exp(eta)
|     X-side count overdispersion (var/mean 6.17; about 6.2x of the 81.2x weight ratio)
|     actual precision share A''/phi f f^T   [UNRESOLVED - F was never saved]
|
|     Note: every remedy that worked in F4 (log-Gaussian, z-score-Gaussian) removes the
|     baseline, the scale AND the dispersion mismatch at once. F4 discriminates nothing.
|
+-- Which experiment separates them?
      |
      +-- A6 baseline + dispersion stress   [no model change]
      |     under Poisson, alpha moves mean, variance and A'' together
      |     -> FLAT surface  : falsifies the whole bundle at once   [decisive]
      |     -> RISING surface: bundle matters, but intercept vs curvature vs
      |                        mean-variance coupling remain unseparated   [ambiguous]
      |
      +-- C precision instrumentation   [no model change]
      |     -> measures the share, closes U1
      |     -> mediation / correlational only; does not separate the three either
      |
      +-- B intercept x representation x X-dispersion factorial   [GATED]
            the ONLY design that separates them
            requires: X column intercept AND a dispersion-aware X count family
            i.e. it requires the modifications whose justification is in question
            |
            +-- => BRANCH 2 cannot be closed without first accepting a provisional
                   implementation. That is a human scientific decision, not an
                   evidence gap this audit can fill.

(parallel, independent of both branches)
  Cora k selection F8 : Q_strict peaks at k=2 and falls; even a zero penalty would not
                        select k=6   [VERIFIED]  -> A4 optimization diagnostic BEFORE any
                        criterion change   [CLASS III; scalar family_x, not per-column]
  MovieLens F9        : +0.034 (4/4, k=3) vs -0.039 (3/6, k=5)  -> matched-protocol re-run
                        before ANY MovieLens attribute claim
```

---

## Evidence label index

- `VERIFIED` - recomputed in this audit from a primary CSV, runinfo, data file or the
  actual code.
- `OBSERVED` - present in a primary artifact and read directly, without recomputation.
- `DERIVED` - follows from verified facts by explicit reasoning; not measured.
- `SUPPORTED` / `PARTIALLY_SUPPORTED` - evidence points this way, within the stated limits.
- `CONFOUNDED` - the available comparison changes more than one factor at a time.
- `UNTESTED` - no experiment in this repository addresses it.
- `CONTRADICTED` - a primary artifact points the other way.
- `UNRESOLVED` - the required measurement does not exist in any stored artifact.

## Research integrity

- New model fits run: **NO**
- Model code changed: **NO**
- Results, figures, CSV or runinfo changed: **NO**
- Registry or canonical research documents changed: **NO**
- Historical records edited: **NO** (the Exp2 discrepancy is reported forward, §13.7)
- Issue #27 run: **NO**
- Prototype promoted to thesis method: **NO**
