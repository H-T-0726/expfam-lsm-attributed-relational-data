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
hypotheses. Those hypotheses were held as hypotheses. The conclusion sections
(§17-§22) are derived from the recomputation in §5-§16, and in at least one case the
recomputation changed the answer: the decomposition in §8 moved the recommended thesis
backbone away from the per-column story and toward the family-generalization story.
Candidates A-E in §16 were scored on a uniform rubric that was fixed before scoring.

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
| **F2** | As Y gets sparse, the per-column advantage grows | `y_sparsity_..._trials10.csv` | 10 paired trials per contrast per rate | all_gaussian - per_column: +0.011 / +0.065 / +0.412 / +0.426 across rates 1.0/0.5/0.2/0.1 | yes | monotone; 10/10 wins at rates 0.2 and 0.1 | single generative configuration; scalar-parameter error also grows | X compensates when Y information falls | seed dependence (10/10) | generalization to other n, d, k*, family mixes | **II/III** | HIGH - the only positive result of the per-column line | E (redesigned) or a second generative configuration | NO |
| **F3** | MovieLens: adding raw `ratings_count` as a Poisson column degrades held-out Y prediction | `movielens_attribute_diagnosis_..._trials4.csv`; `movielens_mixed_x_summary.csv` | 4 fits (2 splits) in each of two experiments | test_y_ll -0.374 vs genre_only, **0/4 better**, per-fit -0.314..-0.441; w0 3.417 -> 3.156, w 0.272 -> 0.330, hc_AUC 0.970 -> 0.949 | yes, fit-matched | reproduced in two separate experiments, 4/4 fits each | leakage (count and Y share `u.data`); only 2 splits; legacy numerics | no X intercept; raw count scale; Poisson fixed dispersion with A''=exp(eta); precision-block dominance; count informativeness | Poisson X clipping (§12); Y overdispersion (§9); NaN/divergence (0) | actual `A''/phi f f^T` share; whether an intercept fixes it; whether raw count under Gaussian **with genre still Bernoulli** also degrades | **II** (with a CLASS III leakage caveat) | HIGH - the only real-data failure of the per-column line | A, then C, then B (§16) | NO |
| **F4** | Transforming the count removes the degradation | same CSV | 4 fits | log-Gaussian **+0.0021**, z-score-Gaussian **+0.0016** vs genre_only; the two differ by +0.0005 (sd 0.0049) | yes | 2/4 fits better each, i.e. indistinguishable from genre_only | same as F3 | removal of the large baseline; switch to an estimated-dispersion family | that log specifically matters (z-score works identically) | which of centering / scaling / estimated dispersion is the operative part | **II** | HIGH - it is the causal-separation lever we already have | B (§16) | NO |
| **F5** | Adding noise attributes does not help and sometimes hurts | `noise_check_summary.csv` | 3 trials x 5 noise conditions | mean deltas: gauss3 **+0.0098**, gauss6 **-0.0005**, gauss12 **+0.0120**, bern3 +0.0004, pois3 +0.0010 | yes, trial-matched | **not reproducible as a dose response** - §10 | 3 seeds; single configuration | Gaussian noise gets `1/sigma_hat^2` weight; local optima | a monotone dose response (0/3 trials monotone) | whether a systematic effect exists at larger trial counts | **III**, possibly II later | MEDIUM | D (§16) | NO |
| **F6** | all-Gaussian is unexpectedly strong for Z and Y | `single_vs_joint`; `movielens_attribute_diagnosis`; `poisson_misspecification` | 3 + 4 + 15 paired | §11 | yes in all three | three independent settings, consistent direction | different data, metrics and families | quasi-likelihood robustness; estimated dispersion auto-downweighting; Y dominance; metric divergence (Z vs density) | that it is a fluke of one experiment | which of the candidate mechanisms dominates | **II/III** | HIGH - the strongest counterargument to the per-column claim | A and C give partial answers | NO |
| **F7** | Poisson / Bernoulli objective-score-curvature inconsistency in the legacy lineage | `per_column_math_code_audit_20260821.md` PC-001/PC-002 plus the code | deterministic counterexamples | at eta=11.5, x=3: implemented score -22023.47, precision 22026.47; the actual objective's finite-difference score and negative Hessian are both 0 | n/a | exactly reproducible | none - it is a code fact | hard clip `[-20,10]`; probability floor 1e-10; curvature floor 1e-8 | **resolved in the consistent lineage** (Issue #25 / PR #26) | whether it ever activated during historical EM runs | **I** | HIGH for future work, **LOW as an explanation of F3** (§12) | none for the defect itself; activation logging needed for the history | already fixed forward; not re-opened here |
| **F8** | On Cora, the k criterion disagrees with AUC/AP/NMI | `cora_balanced_k_sweep_summary.csv` | 3 trials x 6 k | criterion argmin k=1; AP/AUC argmax k=6; NMI/ARI argmax k=3 | by seed | consistent across 3 trials | density 0.011; n=280 subset | penalty too large in sparse data; **Q_strict itself is non-monotone in k**; the criterion is Q-based, not Schwarz | that the parameter count is wrong (`p = kd - k(k-1)/2` reproduces exactly) | why Q degrades for k>=4 (optimization vs MC vs Laplace) | **III** plus optimization | MEDIUM-HIGH | a Q-vs-k optimization diagnostic (§17 A4) | NO |
| **F9** (new) | On MovieLens, even genre-only X does not reliably help strict held-out Y | `movielens_shared_z_ablation_summary.csv` | 6 fits (3 splits x 2 seeds), k=5 | proposed_XY - y_only_fix_x = **-0.039** test ll, X helps **3/6** | yes, fit-matched | sign disagrees with the attribute-diagnosis experiment (+0.034, 4/4 at k=3) | different k, script, evaluation | X contributes little at this n and density; k differs | none | whether attribute integration helps MovieLens **at all** | **II/III** | HIGH - it weakens the premise of the whole MovieLens line | a matched-protocol re-run (§17 A5) | NO |

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
   sparsest rates, trial-matched, 0 NaN - but one generative configuration.
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
| NB / overdispersion is not needed | `SUPPORTED` (§9, candidate J) |

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

### 8.2 Decomposition (this audit's main addition)

`single_gaussian` uses only the 3 Gaussian columns and specifies them correctly.
`all_gaussian` uses all 9 columns and specifies 6 of them wrongly.
`per_column_all` uses all 9 and specifies all of them correctly. Therefore

- `all_gaussian - single_gaussian` isolates the **cost of X-family misspecification**;
- `single_gaussian - per_column_all` isolates the **value of adding 6 more correctly
  specified columns**.

At `y_obs_rate = 0.2` these are **+0.3677** and **+0.0439**: the misspecification cost is
**8.4x** the joint-integration value. At rate 0.1 they are 0.3429 and 0.0832, a factor of
**4.1x**.

`SUPPORTED`: the sparse-Y interaction is real and monotone.
`SUPPORTED`: most of it is "avoid specifying the wrong family", not "integrate more
attribute blocks".

This distinction is not made in
`reports/story_diagnostics/story_diagnostics_summary_20260713.md`, which reports the
per_column-vs-all_gaussian and per_column-vs-y_only gaps only. Nothing in that report is
wrong; the decomposition simply was not performed.

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
| J | Poisson distributional misspecification / overdispersion of Y | **NOT_SUPPORTED** | The marginal var/mean of Y is 9.89, but the plug-in posterior predictive check reproduces it (rep_mean 9.79, p = 0.15), and the conditional Pearson dispersion is 1.13 at k=3 and 0.76 at k=5. On strict held-out at k=3 NB improves test ll by only +0.020 (6/6). On synthetic NB-Y data NB is *worse* for Z at every r (§11). In any case Y is identical across all conditions in this experiment. |

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
| Estimated dispersion auto-downweights badly scaled or badly specified blocks | `PARTIALLY_SUPPORTED` | Directly supported by the MovieLens numbers: `1/sigma_hat^2 ~ 2.0e-4` for the raw count under Gaussian versus `A'' ~ 154` under Poisson (`VERIFIED`, §9.2). Also explains why Gaussian is safe for Z yet bad for X reconstruction |
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

**Consequence for the research claim:** the sentence "specifying the correct family per
column improves Z estimation" is not supported at dense Y and is `CONTRADICTED in sign` by
`single_vs_joint` (-0.0004). What is supported is the narrower, conditional statement in
§8.

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
| `DualExpFamLSMConsistent` / `DualExpFamLSMPerColumnConsistent` (`objective_consistent_numerics.py`, `model_dual_expfam_consistent.py`) | **Resolved.** Clip removed, canonical objective/score/curvature, dtype-derived overflow guard, 9/9 new tests PASS, finite-difference score error 1.75e-06 at eta=11.5 (Issue #25 / PR #26) |
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
| **Y density** | §8, monotone across 4 rates, 10 paired trials | one generative configuration; scalar-parameter error also grows | generator design (dense F rows) | **YES for this generator** | E (redesigned), or a second configuration | none |
| **numerical clipping / floors** | PC-001 / PC-002 deterministic counterexamples | MovieLens etas are interior (`VERIFIED`); resolved forward in the consistent lineage | EM-transient activation never logged | **YES at convergence, NO during EM** | activation logging during EM | already fixed forward |
| **optimization / local optima** | noise trial 1 bimodal; `all_bernoulli` collapse in 1/3; `poisson_strict` k=5 max test RMSE 48.19 against a mean of 14.58; Cora Q non-monotone in k | 0 NaN everywhere; most runs stable | MC sampling, `scale_Z`, Adam schedule | **NO** | multi-restart Q comparison at fixed data and fixed k | optimization / convergence change |
| **`scale_Z`** | applied unconditionally (`em_runner.py` line 226); forces mean square 1 on all MC samples, which interacts with any block that wants a large `||z||` | no measured failure attributed to it | every scale mechanism above | **NO** | the non-destructive `apply_scale_z` ablation already designed in `reports/theory_audit/diagnostic_designs_20260719.md` §3 | make it switchable, default unchanged |
| **leakage** | runinfo `leak_caveat`; count and Y share `u.data` | would bias the affected conditions upward, yet they are worse | attribute informativeness | partially | train-only attribute construction | **none - this is CLASS III** |
| **overdispersion** | marginal var/mean 9.89 | PPC p = 0.15 reproduces it; conditional dispersion 1.13 / 0.76; NB beats Poisson by only 0.020 at k=3; NB is worse for Z at every r | the k=5 Poisson divergence inflates NB's apparent gain | **YES** | already isolated | NB / dispersion (not motivated for Z) |
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
| 8 | **dispersion / NB** | MovieLens var/mean 9.89 | overdispersion | +0.020 test ll at k=3 (6/6) | PPC p = 0.15; conditional dispersion 1.13 / 0.76; NB worse for Z in 15/15 paired synthetic comparisons; the k=5 gain comes from a Poisson divergence | latent structure already absorbs the marginal overdispersion | yes | **NOT_JUSTIFIED for Z recovery**; **FUTURE_WORK** for density prediction only |
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

**A - Poisson baseline stress.** Synthetic. Hold the latent signal fixed and vary only the
count baseline (e.g. `x_il ~ Poisson(alpha * exp(f_l^T z_i))` with `f` fixed across arms),
correct Poisson family, current no-intercept model. Question: *does a large count baseline
alone break a no-intercept model, and how does the damage scale with the baseline?*
Targets U4, and U1 if instrumented. Needs no model change - only a generator arm. RMSE(Z)
is measurable, unlike on MovieLens, and there is no leakage.

**B - Intercept x representation factorial.** Arms: raw Poisson without intercept (exists),
raw Poisson with intercept (**requires the intercept**), transformed Gaussian without
intercept (exists), optionally transformed Gaussian with intercept. Question: *is it the
intercept, the representation/scale, or both?* Targets U2 and U3 directly.

**C - Curvature / precision block diagnostic.** Instrumentation, not a new model: persist
`F` and the converged `eta`, then compute per-block `A''(eta)/phi * f_l f_l^T` and its
trace / norm share of the total X precision, on the existing MovieLens condition set and
on synthetic mixed-X. Question: *does one count column actually dominate the Z update?*
Targets U1, and U5 if clip activation is logged during EM at the same time.

**D - Noise dose-response.** Increase the number of noise columns with enough trials to
detect a systematic degradation. Question: *is block weighting or regularization actually
needed?* Targets the F5 null.

**E - Complementary blocks (Issue #27).** Synthetic generator where each attribute block
loads mainly on a different latent dimension; per-dimension RMSE reported. Question:
*when blocks carry different information about Z, does joint integration matter?* Targets
U9 and the weak component identified in §8.2.

### 16.3 Scores

| candidate | IG | DIR | CSP | UNC | IND | AMB | THE | total | gated? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A** Poisson baseline stress | 4 | 5 | 4 | 4 | 5 | 4 | 4 | **30** | no |
| **B** intercept x representation | 5 | 5 | 5 | 5 | **1** | 3 | 4 | **28** | **YES** - requires the X intercept |
| **C** curvature / precision diagnostic | 3 | 5 | 3 | 4 | 5 | 4 | 3 | **27** | no |
| **E** complementary blocks (#27) | 3 | 3 | 2 | 3 | 5 | 3 | 3 | **22** | no |
| **D** noise dose-response | 2 | 2 | 2 | 3 | 5 | 2 | 3 | **19** | no |

Score justifications, briefly:

- **A**: DIR 5 because the observed failure is literally a high-baseline count column under
  Poisson. CSP 4 because it separates baseline magnitude from family choice, leakage and
  informativeness, but cannot separate "no intercept" from "large baseline" - in a
  no-intercept model those are the same knob, which is precisely why its answer is a
  precondition for B. AMB 4 because a flat damage curve falsifies the baseline hypothesis
  just as informatively as a rising curve confirms it.
- **B**: the highest raw score, and gated. CSP 5 and UNC 5 are real; IND 1 is decisive
  under the declared rule.
- **C**: IG 3 and CSP 3 because measuring a dominance share does not by itself establish
  causation - a large precision share could be a correlate of a badly posed column rather
  than the operative cause. It is nonetheless the only way to close U1.
- **E**: DIR 3, not 1. §8.2 shows the joint-integration component of the sparse-Y result is
  small (+0.044 against a misspecification cost of +0.368), and F1 is an observed
  phenomenon whose stated explanation is exactly the generator property E manipulates. But
  CSP 2, because it separates none of the confounded mechanisms in §14, and AMB 3, because
  a generator built so that integration must help produces a result that is hard to defend.
- **D**: DIR 2 because F5 is currently a **null**, not a failure; AMB 2 because a larger
  null is still a null, and a small positive would be hard to distinguish from the
  local-optimum behaviour already visible in trial 1.

### 16.4 Ranking

```
1. A   (30)                          <- run this one if only one runs
2. C   (27)                          <- pure instrumentation; closes U1
3. B   (28 raw, gated by IND = 1)    <- becomes the right experiment once A has answered U4
4. E   (22)                          <- see the Issue #27 decision in 18
5. D   (19)
```

C is instrumentation rather than an experiment, so recording the per-block precision inside
A's runs costs almost nothing. That is an implementation note, not a merged
recommendation: the single primary recommendation remains A.

---

## 17. Ranked next actions

At most five. Each is a recommendation only; nothing here was executed.

### P0 - A1. Poisson baseline stress (DIAGNOSTIC EXPERIMENT)

- **Question.** Does a large count baseline alone degrade a no-intercept
  exponential-family X model, and does the degradation scale with the baseline?
- **Current evidence.** F3 (-0.374 test ll, 0/4 fits, two experiments); the data-side
  curvature ratio of 81.2x; the Gaussian arm with the same raw column loses 18x less.
- **Missing evidence.** U4 entirely, U1 partially. No condition anywhere in this
  repository varies the baseline while holding the latent signal fixed.
- **Minimal design.** Synthetic mixed-X; hold the `F` and `Z` generation fixed; vary only a
  multiplicative count baseline over roughly four levels spanning `mu` from about 1 to
  about 200; correct per-column families; current model. Report RMSE(Z), held-out Y ll,
  `w0`, `w`, per-block X RMSE, `||F||` and the per-block precision share. Use enough
  trials that the trial-matched paired difference is readable (the sparse-Y experiment
  needed 10). Run with `numerics_mode="consistent"` and log clip/floor activation during
  EM.
- **Decision enabled.** Whether the intercept / offset branch (B, then modifications 1 and
  3) is worth opening at all.
- **Risk.** If the damage curve is flat, the intercept branch is falsified - a useful
  outcome, not a failed experiment.
- **Known confound to design around.** `scale_Z` forces the MC samples to mean square 1
  unconditionally (§4), so as the baseline rises the model can only respond by growing
  `||F||`, not `||Z||`. A1's response is therefore *mediated* by `scale_Z`. Record `||F||`
  per arm, and either run the `apply_scale_z` on/off ablation alongside or state
  explicitly that the measured curve is conditional on `scale_Z` being on.
- **Before #27:** YES.

### P1 - A2. Per-block precision instrumentation (ALGORITHM VALIDATION)

- **Question.** What share of the X precision `sum_l A''(eta_il)/phi_l f_l f_l^T` does each
  attribute block actually contribute at convergence?
- **Current evidence.** The data-side ratio only (81.2x), with `||f_l||` unknown.
- **Missing evidence.** U1. `F` is not persisted by any of the relevant scripts, and
  `*.npy` is git-ignored.
- **Minimal design.** A diagnostic function plus persistence of `F` and the converged
  `eta`; no model change. Apply it to the existing MovieLens condition set and to A1's runs.
- **Decision enabled.** Whether "precision-block dominance" graduates from
  `PARTIALLY_SUPPORTED` to a finding, or is falsified.
- **Risk.** A dominance measurement is correlational; it constrains but does not prove the
  mechanism. The report that uses it must say so.
- **Before #27:** YES.

### P1 - A3. Forward correction of the Exp2 registry annotation (CLAIM RESTRICTION)

- **Question.** Which numbers may be quoted for the fixed-lineage n sweep?
- **Current evidence.** Primary artifact: 49.3% / 41.2% / 58.6% (`VERIFIED`, two
  independent recomputations). Registry note and `reports/real_data_experiment_plan.md`
  §2: 40% / 17% / 62%.
- **Missing evidence.** None - this is settled.
- **Minimal design.** An append-only, dated forward-correction row in
  `EXPERIMENT_REGISTRY.md`, in the same style as its 2026-08-21 Phase 5a.1 section.
  **Do not edit the historical row and do not edit the dated plan document.**
- **Decision enabled.** Prevents an incorrect figure entering the thesis.
- **Risk.** None, provided the historical text is left intact.
- **Before #27:** YES - it costs nothing and is independent of everything else.

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
- **Before #27:** not required, but it outranks #27 on evidence.

### P2 - A5. Matched-protocol MovieLens attribute check (DIAGNOSTIC EXPERIMENT)

- **Question.** Does attribute integration help MovieLens **at all** (F9)?
- **Current evidence.** Contradictory: +0.034 (4/4) at k=3 in one protocol, -0.039 (3/6) at
  k=5 in another.
- **Missing evidence.** U8. The two runs differ in k, script, evaluation and split count.
- **Minimal design.** One script, one evaluation, both k values, genre-only X versus
  y_only, train-only attribute construction if feasible (otherwise repeat the leakage
  caveat verbatim).
- **Decision enabled.** Whether MovieLens can carry any attribute-integration claim.
- **Risk.** A null removes MovieLens as a positive dataset for the thesis - which would
  itself be an honest and reportable result.
- **Before #27:** not required.

---

## 18. Issue #27 decision

**Decision: REDESIGN.**

This decision was made after the §16.3 scoring. It is neither "defer because of
positive-story risk" nor "run because the issue already exists"; both of those reasons
were excluded in advance.

**What E would answer, and how much that is worth.** §8.2 is the strongest reason to take
E seriously: the joint-integration component of the sparse-Y result is only +0.044 RMSE(Z)
against a misspecification cost of +0.368, and F1 shows the same near-null at dense Y. The
stated explanation for both - that the current generator makes every block individually
sufficient because `F` rows are dense random vectors - is exactly what E manipulates. E
therefore tests a real, quantified weakness in our own positive claim, and it targets U9.
That is why it scores DIR 3 rather than DIR 1.

**Why it is not next.** It scores 22 against A's 30. It separates none of the confounded
mechanisms in §14 (CSP 2), and it does not touch the only failure observed on real data.

**Why the recorded design cannot be run as written.** The design in
`reports/story_diagnostics/story_diagnostics_next_plan_20260713.md` (experiment 2)
predates the evidence in this audit and has four concrete defects:

1. **No Y-density axis.** It leaves the Y observation rate at the dense setting. §8 and F1
   show that at dense Y the entire achievable effect in this generator is about 0.011
   RMSE(Z). Run as written, E would very likely produce a null that says nothing about
   complementary blocks.
2. **No pre-registered decomposition.** The quantity actually in question is
   `(single-block correct) - (joint correct)`, the integration value, separated from
   `(all-forced-Gaussian) - (single-block correct)`, the misspecification cost. The memo
   reports overall and per-dimension RMSE only, which would leave exactly the ambiguity
   §8.2 had to resolve after the fact.
3. **Raw-value forced-misspecification arms.** The memo includes `all_bernoulli` and
   `all_poisson` on raw values. `single_vs_joint` shows `all_bernoulli` collapsing in 1 of
   3 trials (RMSE(Z) 1.747, test ll -106.5) and contaminating the mean. Either drop those
   arms, pre-register a robust summary, or report them separately.
4. **Numerics and trial count predate current knowledge.** The memo predates
   `numerics_mode` (Issue #25 / PR #26) and fixes no trial count. E should run on the
   consistent lineage with clip/floor activation logged, and needs a trial count in the
   range that made §8 readable (10 trial-matched trials), not the 2-3 of the earlier
   pilots.

**Implied instruction.** Keep Issue #27 OPEN. Do not run it as specified. Revise the design
along points 1-4, and schedule it after A1 - not because A1 is logically a prerequisite for
E, but because A1 outranks it on the uniform rubric and because E's redesign should be able
to use whatever A1 and A2 reveal about precision weighting.

---

## 19. JUSTIFIED_NOW

**No model modification reaches JUSTIFIED_NOW.**

Applying the five conditions honestly:

| modification | (1) repeated concrete failure | (2) mechanism supported or alternatives excluded | (3) clear mathematical role | (4) before/after validation designable | (5) not "the paper had it" | verdict |
|---|---|---|---|---|---|---|
| X intercept | partly - 4 fits / 2 splits | **NO** - U2, U3, U4 all open | yes | yes | must be argued explicitly, since the X bias was previously removed as an error | fails (1), (2) |
| Poisson offset | as above | **NO** | yes | yes | yes | fails (1), (2) |
| block / column weighting | **NO** - §10 is a null | **NO** | weak - no likelihood justification | yes | yes | fails (1), (2), (3) |
| F regularization | **NO** | **NO** - `||F||` never recorded | yes | yes | yes | fails (1), (2) |
| attribute selection | **NO** | **NO** | yes | yes | yes | fails (1), (2) |
| NB / dispersion | **NO** - §9 candidate J | **CONTRADICTED** | yes | yes | yes | fails (1), (2) |
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

---

## 20. NOT_JUSTIFIED_YET

Ordered by how much evidence would be needed to move them.

1. **X intercept / Poisson offset** - `NEEDS_DIAGNOSTIC_FIRST`. Needs A1, then B. Note for
   the write-up: a per-column bias in X was **removed** as an error (RESEARCH_MASTER §4,
   eq(2): `N(w_0l + z_i^T w_l, sigma_l^2)` -> `N(f_l^T z_i, sigma_l^2)`). Re-introducing it
   must rest on the diagnostic evidence, and the report must say so, or it will read as
   reverting to the prior work's formulation.
2. **Block weighting / column weighting** - `NOT_JUSTIFIED`. The motivating observation
   (F5) is a null in every individual trial, and the real-data failure (F3) is explained
   without it. Introducing weights would also further damage the likelihood interpretation
   of the criterion, which §5.4 shows is already fragile across families.
3. **F regularization** - `FUTURE_WORK`. Record `||F||` first; nothing currently measures
   it.
4. **Attribute selection** - `NOT_JUSTIFIED`. No systematic degradation from uninformative
   columns has been demonstrated.
5. **NB / dispersion** - `NOT_JUSTIFIED for Z recovery`; `FUTURE_WORK` for density
   prediction only, where the honest statement is +0.020 test ll at k=3.
6. **Optimization / convergence change** - `NEEDS_DIAGNOSTIC_FIRST` (A4). The evidence
   that *something* is wrong is good; the evidence about *what* is absent.
7. **k-selection criterion change** - `NEEDS_DIAGNOSTIC_FIRST`, and CLASS III. Do not
   attempt to fix an evaluation problem by changing the generative model.
8. **Promotion of the per-column prototype to the thesis method** - `NOT_JUSTIFIED`.
   §5.2, §8.2 and §11 together mean the strongest defensible statement is conditional, not
   general.

---

## 21. Candidate thesis paths

| path | evidence strength | novelty | unresolved issues | implementation required | experiment required | oral-defense strength | risk | feasibility |
|---|---|---|---|---|---|---|---|---|
| **1. X/Y exponential-family generalization** | **Strongest.** 180 + 180 + 180 + 550 rows of synthetic evidence with 10-30 trials, 10/10 k selection in three scenarios, misspecification ratios 4.34 / 9.04 / 40.37 reproduced exactly, Wine k=3, Cora 2.6-2.8x random | moderate - generalizing a fixed-family LSM | KI-001 hedge; criterion naming (KI-010); Cora Q non-monotonicity | none | none | **high** - every number reproduces from primary CSVs | low | already done |
| **2. + per-column heterogeneous X** | **Weak.** per_column - all_gaussian = -0.0004 at dense Y (3 trials); MovieLens negative; F9 questions attribute value on MovieLens at all | high | PC-001 in the legacy lineage (fixed forward, unvalidated); prototype status | consistent-lineage validation | several | **low** - the obvious question "is it better than forcing Gaussian?" currently answers "no, at dense Y" | high | possible, not defensible today |
| **3. + evidence-driven count/intercept refinement** | **Insufficient.** 4 fits / 2 splits; mechanism `UNRESOLVED`; leakage caveat binding | high if it works | U1-U4 all open | X intercept | A1, B, plus validation | low today | high | not feasible this cycle |
| **4. Conditions under which attribute integration is effective (sparse Y)** | **Moderate.** 10 trial-matched trials, monotone across 4 rates, 10/10 at the sparsest rates - but **one generative configuration**, and §8.2 shows the effect is mostly misspecification avoidance rather than integration | moderate | U9; needs a second configuration | none | E (redesigned) or another configuration | moderate | medium | feasible with one more experiment |
| **5. Family generalization plus explicit diagnostic limitations** | **Strong**, because it is built from what is already `VERIFIED`, including the negative results | low as novelty, high as scholarship | none - the limitations are the content | none | none | **high** - it pre-empts hostile questions by stating the limit first | low | already done |

### Recommended path

**Path 1 as the backbone, framed with Path 5, and Path 4 as a conditional extension.**

- Path 1 supplies the claims that survive recomputation without qualification.
- Path 5 supplies the honest boundary: forcing every column to Gaussian is nearly as good
  for Z and Y and worse only for X reconstruction (§11); the criterion is Q-based and not
  comparable across families (§5.4, §13); on Cora the fitted Q is non-monotone in k (§13.6);
  MovieLens attributes did not help (§9, F9).
- Path 4 is added **only if** a second generative configuration reproduces the sparse-Y
  interaction, and it must be stated with the §8.2 decomposition attached, i.e. "when Y is
  sparse, specifying the right family per column matters much more than adding blocks".

**This differs from the hypothesis carried into the audit**, which put Path 4 first. The
§8.2 decomposition demoted it: the component unique to per-column joint integration is
+0.044 RMSE(Z) against a +0.368 misspecification cost, and at dense Y it is negative. The
recomputation, not the plan, produced this ordering.

Confidence: **moderate-to-high** for Path 1 plus Path 5 (every supporting number reproduced
from primary artifacts); **low** for Path 2 or Path 3 as a thesis backbone today.

---

## 22. Final evidence-based recommendation

**Run exactly one experiment next: A1, the Poisson baseline stress test (§17), with the A2
per-block precision instrumentation recorded inside the same runs.**

Why this one:

1. It is the only top-ranked candidate that attacks the **only failure observed on real
   data** (F3) while requiring **no model modification**, which is what the operating
   principle of this phase demands.
2. It removes the confound that currently makes F3 uninterpretable. Every remedy that works
   (log, z-score) changes baseline, scale and family dispersion simultaneously, and the one
   condition that comes closest to isolating "no intercept with raw scale"
   (`mixed_all_gaussian`) also changes the genre family. A1 varies exactly one thing.
3. Both outcomes are decisive. A rising damage curve opens the intercept / offset branch
   and makes B the right follow-up. A flat curve falsifies the baseline hypothesis and
   moves attention to the dispersion asymmetry in §11.3, which is currently the
   best-supported unifying hypothesis in this audit.
4. It is synthetic, so RMSE(Z) is measurable and the MovieLens leakage caveat does not
   apply.
5. F3 rests on a single n=100 MovieLens subset (§9.4). Chasing the mechanism on that same
   subset - which is what B would do - adds fits without adding samples. A1 moves the
   question to a setting where the sample can be replicated at will and the truth is known.

Do **not** implement the X intercept, block weighting, column weighting, F regularization,
attribute selection, NB, or a criterion change before A1 reports. None of them reaches
JUSTIFIED_NOW (§19).

Alongside it, two zero-risk items: the A3 registry forward correction, and the policy that
every new per-column run selects `numerics_mode="consistent"` and logs clip/floor
activation.

---

## 23. Decision tree

```
MovieLens raw-count failure (F3)   [VERIFIED: -0.374 test ll, 0/4 fits, 2 experiments]
|
+-- Is it the legacy numerical defect (PC-001)?
|     eta needed = ln(154) = 5.04, clip boundary = 10 (mu = 22026);
|     x_rmse is consistent with an interior fit.
|     -> NO at convergence                      [NOT_SUPPORTED]
|     -> during-EM activation was never logged  [UNRESOLVED]
|        => log clip/floor activation in every future run (costless),
|           but do not expect it to explain F3.
|
+-- Is it Y-side overdispersion?
|     PPC p = 0.15 reproduces var/mean 9.89; conditional dispersion 1.13 / 0.76;
|     NB gains +0.020 at k=3; NB is worse for Z in 15/15 synthetic paired comparisons.
|     -> NO   [NOT_SUPPORTED]   => NB / dispersion stays NOT_JUSTIFIED.
|
+-- Is it block-count imbalance (1 count column vs 19 genre columns)?
|     rating_stats_only uses 3 columns and degrades by the same -0.362.
|     -> NO   [it is the column's identity, not the count]
|        => block weighting stays NOT_JUSTIFIED.
|
+-- Is it leakage?
|     Leakage would flatter these conditions; they are worse.
|     -> cannot explain the degradation   [CONFOUNDED, and it cuts the other way]
|        => but it does void these runs as generalization evidence   [CLASS III].
|
+-- Has the actual precision dominance A''/phi f f^T been measured?
|     F was never saved (*.npy is git-ignored).
|     -> NO   [UNRESOLVED]
|        => A2 instrumentation. The data-side ratio is 81.2x, but ||f_l|| is unknown
|           and is NOT imputed.
|
+-- Is it the baseline magnitude, the representation/scale, or the family's fixed
    dispersion?
      Known: log and z-score are interchangeable (+0.0005 apart), so variance
             stabilization is not the operative part.
      Known: the same raw column under Gaussian (estimated sigma^2) costs 18x less,
             but that arm also changes the genre family, so it isolates nothing.
      Unknown: U2, U3, U4.
      |
      +-- STEP 1: A1 baseline stress    [no model change]
      |     |
      |     +-- damage rises with the baseline
      |     |     => the baseline / intercept branch is live
      |     |        => STEP 2: B, intercept x representation factorial
      |     |             (this is the point at which implementing the X intercept
      |     |              becomes justified, and not before)
      |     |             |
      |     |             +-- intercept fixes raw Poisson -> X intercept / Poisson offset
      |     |             |                                  becomes JUSTIFIED
      |     |             +-- it does not                 -> representation convention,
      |     |                                                or the dispersion branch below
      |     |
      |     +-- damage is flat in the baseline
      |           => the intercept hypothesis is FALSIFIED
      |              => dispersion branch: the operative variable is that Bernoulli and
      |                 Poisson have phi fixed at 1 while Gaussian estimates it (11.3)
      |                 => the question becomes how a per-block dispersion or weighting
      |                    should be *derived*, not bolted on - a modelling question,
      |                    not an engineering one.
      |
      +-- (parallel, independent of the above)
            Cora k selection (F8): Q_strict peaks at k=2 and falls
              -> even a zero penalty would not select k=6
              -> this is NOT primarily a penalty problem   [VERIFIED]
              -> A4 optimization diagnostic before any criterion change   [CLASS III]

            MovieLens attribute value (F9): +0.034 (4/4, k=3) vs -0.039 (3/6, k=5)
              -> A5 matched-protocol re-run before any MovieLens attribute claim
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
