# Complementary Mixed-X Blocks — Objective-Consistent Validation (Issue #27)

## Scope

Pre-registered synthetic validation of the question left open by Issue #28 / PR #29:

> If Gaussian / Bernoulli / Poisson attribute blocks carry different pieces of the latent
> structure, is there a measurable reason to estimate them jointly with one shared Z
> instead of fitting each attribute family separately?

This is a **claim-bearing synthetic experiment**, but `DualExpFamLSMPerColumn(Consistent)`
remains an **experimental prototype**. A positive result here does not promote it to the
thesis/main method, and this report makes no real-data claim.

- Base commit: `46fbc544b553f296e81de51635b0cda5161fbc15`
- Branch: `experiment/27-complementary-blocks-consistent`
- Script: `tools/research_audit/run_complementary_blocks_consistent.py`
- Fits: **120** (10 trials x 2 Y-observation rates x 6 conditions)
- Numerics: `numerics_mode="consistent"` in **120/120** fits — the **first empirical use**
  of the objective-consistent lineage merged in PR #26.
- Model code, runner, existing tests and historical results: **unchanged**.

Every design constant, the primary domain, the primary endpoint, the primary contrasts and
the delta sign convention were fixed **before** any fit was run and were not changed
afterwards.

---

## 1. Design (pre-registered, unchanged after seeing results)

```
N=80, D=9, K_TRUE=3, L=5, num_iter=8, trials=10, test_ratio=0.2
y_obs_rate in {1.0 (dense negative control), 0.1 (sparse primary)}
w0_true=1.2, w_true=0.3, sigma_G=0.3, dominant_weight=0.9, minor_weight=0.15
Z_true = column-wise standardized N(0,I) draw
generator clipping: NONE (canonical exp / stable sigmoid)
blocks: bernoulli cols 0-2 -> z1, gaussian cols 3-5 -> z2, poisson cols 6-8 -> z3
conditions: y_only / single_bernoulli / single_gaussian / single_poisson
            / per_column_all / all_gaussian
seeds: data 110000+t, model 111000+10t, split 112000+100t, thin 113000+100t+rate_idx
```

Within a trial, `Z_true / F_true / X / Y / test_mask` are shared by all six conditions;
within `(trial, rate)` the `train_mask` is shared as well. A 20% test pair set is fixed once
per trial and is identical for both rates.

**Primary endpoint** (fixed in advance): whole-space Procrustes `RMSE_Z` at
`y_obs_rate = 0.1`. **Primary contrasts**: `per_column_all` versus `single_bernoulli`,
`single_gaussian`, `single_poisson`, `y_only`.
**delta = comparator_rmse − per_column_rmse**, so positive favours per-column.

---

## 2. VERIFIED

Facts checked directly from code, hashes or the produced artifacts.

- **120/120 fits completed**, exactly 10 rows per `(rate, condition)` cell, no duplicate
  `(trial, rate, condition)` key.
- **`numerics_mode == "consistent"` in 120/120**, asserted from each returned result.
  `clip_diag.status == "not_applicable"` in 120/120, i.e. no hard Poisson clipping path.
- **`internal_retry_detected` = 0 / 120.** `run_em_experimental` retries internally with a
  different seed and a halved Newton alpha and resets `nan_count` per retry, so
  `res["nan_occurred"]` can be `False` even after a NaN reset occurred. Each fit was
  therefore run with `verbose=True` inside `contextlib.redirect_stdout`, and the runner's
  literal `"[NaN iter="` reset message was searched for. It never appeared.
- **`nan_occurred` = 0 / 120**, `nan_count` = 0, **`q_bic_failed` = 0 / 120**,
  `failure_reason` null in every row, **0 warnings** captured by
  `warnings.catch_warnings(record=True)`.
- **Pairing integrity**: dataset hash (X, Y, Z_true, F_true) identical across all six
  conditions within each trial; test-mask hash identical within each trial; train-mask hash
  identical within each `(trial, rate)`.
- **Support validation** passed for all six conditions
  (`validate_support=True, allow_support_mismatch=False`).
- **All scientific metrics finite.** The only non-finite cells are block-diagnostic columns
  for blocks a condition does not observe (e.g. `prec_t_gauss_*` under
  `single_bernoulli`), which is by construction.
- **Precision-trace identity holds numerically**: `sum_j t_ilj == t_il` to
  `max |err| = 4.26e-14` over all 120 fits.
- **Generator complementarity** (10 trials x 9 columns): mean |dominant loading| 0.973
  (bern) / 0.975 (gauss) / 0.981 (pois), minimum 0.858; mean off-dimension norm 0.196 /
  0.198 / 0.180; mean dominant share of squared norm 0.948 / 0.951 / 0.963;
  `||f_l|| = 1.0000` for every column.
- **No generator clipping was needed**: observed eta ranges were
  `[-3.05, 3.07]` (bern), `[-3.31, 3.28]` (gauss), `[-3.02, 3.14]` (pois), far inside the
  float64 representable domain (`log(finfo.max) = 709.78`).

---

## 3. OBSERVED — primary result (`y_obs_rate = 0.1`)

Trial-matched paired differences in `RMSE_Z`, 10 paired trials.
Positive delta favours `per_column_all`.

| contrast | delta_rmse mean | sd | median | trials favouring per-column | delta test LL mean | trials favouring per-column (LL) |
|---|---:|---:|---:|---:|---:|---:|
| vs `y_only` | **+0.5122** | 0.1438 | +0.5688 | **10/10** | +0.6599 | 10/10 |
| vs `single_bernoulli` | **+0.4218** | 0.1433 | +0.4208 | **10/10** | +0.4678 | 10/10 |
| vs `single_poisson` | **+0.3889** | 0.1459 | +0.4303 | **10/10** | +0.4237 | 10/10 |
| vs `single_gaussian` | **+0.2030** | 0.1223 | +0.2138 | **9/10** | +0.1771 | 10/10 |

Levels (mean over 10 trials):

| condition | RMSE_Z | sd | dim1 | dim2 | dim3 | test Y LL |
|---|---:|---:|---:|---:|---:|---:|
| `per_column_all` | **0.6696** | 0.1316 | 0.9206 | 0.2914 | 0.6187 | **−2.3975** |
| `all_gaussian` | 0.7876 | 0.0600 | 1.0597 | 0.2924 | 0.7901 | −2.4174 |
| `single_gaussian` | 0.8726 | 0.0696 | 1.0365 | 0.2820 | 1.0354 | −2.5746 |
| `single_poisson` | 1.0585 | 0.0632 | 1.1976 | 1.2065 | 0.6773 | −2.8211 |
| `single_bernoulli` | 1.0914 | 0.0279 | 0.9622 | 1.1222 | 1.1682 | −2.8653 |
| `y_only` | 1.1819 | 0.0456 | 1.1728 | 1.1638 | 1.1989 | −3.0574 |

All four pre-specified primary contrasts favour `per_column_all`, and all four favour it on
held-out Y log-likelihood in 10/10 trials.

## 4. OBSERVED — dense-Y negative control (`y_obs_rate = 1.0`)

| contrast | delta_rmse mean | sd | trials favouring per-column | delta test LL mean |
|---|---:|---:|---:|---:|
| vs `single_bernoulli` | +0.0530 | 0.0132 | 10/10 | +0.0381 |
| vs `y_only` | +0.0510 | 0.0157 | 10/10 | +0.0413 |
| vs `single_poisson` | +0.0490 | 0.0084 | 10/10 | +0.0358 |
| vs `single_gaussian` | **+0.0087** | 0.0166 | 9/10 | +0.0145 |

Levels: `per_column_all` 0.2609, `single_gaussian` 0.2696, `all_gaussian` 0.2832,
`single_poisson` 0.3099, `y_only` 0.3119, `single_bernoulli` 0.3139.

The direction is the same as at sparse Y, but **the magnitudes shrink by roughly an order
of magnitude** (0.512 → 0.051 against `y_only`; 0.203 → 0.009 against `single_gaussian`).

## 5. OBSERVED — secondary control (`all_gaussian`)

This is the **same-column misspecification contrast (family specification and M-step
optimizer path confounded)**, not a pure family-assignment contrast: `all_gaussian` uses the
analytical closed-form F update while the mixed per-column model uses weighted Adam.

| rate | delta_rmse mean | trials favouring per-column | delta test LL mean | trials favouring per-column (LL) |
|---:|---:|---:|---:|---:|
| 0.1 | +0.1180 | 8/10 | +0.0199 | **5/10** |
| 1.0 | +0.0222 | 8/10 | +0.0142 | 8/10 |

At sparse Y, per-column beats forced-Gaussian on `RMSE_Z` (8/10, +0.118) but the held-out
likelihood difference is **a coin flip (5/10)**.

## 6. OBSERVED — dimension-wise mechanism diagnostic

Dimension-wise RMSE uses the **same** whole-space Procrustes rotation `R`
(no per-dimension oracle alignment). Paired deltas at `y_obs_rate = 0.1`:

| comparator | dim1 (bern) | | dim2 (gauss) | | dim3 (pois) | |
|---|---:|---:|---:|---:|---:|---:|
| `y_only` | +0.2523 | 8/10 | +0.8723 | 10/10 | +0.5802 | 10/10 |
| `single_bernoulli` | +0.0416 | 6/10 | +0.8308 | 10/10 | +0.5495 | 10/10 |
| `single_gaussian` | +0.1160 | 8/10 | **−0.0095** | **5/10** | +0.4168 | 10/10 |
| `single_poisson` | +0.2770 | 8/10 | +0.9151 | 10/10 | **+0.0587** | **7/10** |
| `all_gaussian` | +0.1391 | 8/10 | **+0.0010** | **5/10** | +0.1715 | 10/10 |

`per_column_all` never has an advantage on the dimension a comparator already covers with
the correct block (dim2 vs `single_gaussian`: 5/10; dim3 vs `single_poisson`: 7/10). Its
advantage comes from the dimensions the comparator does not cover.

## 7. OBSERVED — precision-trace diagnostics

Plug-in mean trace contribution to the X-side local precision,
`t_il = c_il ||f_l||^2` with `c = 1/diag(sigma)` (Gaussian, variance not squared),
`p(1-p)` (Bernoulli), `exp(eta)` (Poisson).

**True generator** (10 trials, `||f_l|| = 1` for every column):

| block | true t (block sum) | share | to dim1 | to dim2 | to dim3 |
|---|---:|---:|---:|---:|---:|
| bernoulli | 0.619 | 0.016 | 0.587 | 0.019 | 0.013 |
| gaussian | 33.333 | **0.858** | **0.959** | 31.683 | 0.691 |
| poisson | 4.902 | 0.126 | 0.096 | 0.085 | 4.721 |

gauss/bern = **53.9x**, gauss/pois = **6.8x**.

**Fitted `per_column_all`** block shares: gaussian 0.904 / poisson 0.081 / bernoulli 0.015
at rate 0.1 (0.871 / 0.111 / 0.018 at rate 1.0), i.e. the fitted model reproduces the same
dominance ordering.

**Finite-sample latent correlation** (recorded, no whitening applied): per-trial
`|corr(z_i, z_j)|` reached 0.293 (max over 10 trials x 3 pairs).

---

## 8. INTERPRETATION

**When is joint per-column integration better than fitting one attribute family at a time,
and when is the difference small?**

- **It is better when Y is sparse.** At `y_obs_rate = 0.1` all four pre-specified primary
  contrasts favour `per_column_all` (10/10, 10/10, 10/10, 9/10 on `RMSE_Z`; 10/10 on
  held-out LL). The margin against the best single block is +0.203 `RMSE_Z`.
- **The difference is small when Y is dense.** The same contrasts shrink to
  +0.053 / +0.051 / +0.049 / **+0.009**. Against the best single block the advantage at
  dense Y is under 0.01 `RMSE_Z` with 9/10 wins — small enough that it should not be
  presented as a practical gain.
- **The mechanism is dimension coverage, not uniform superiority.** The dimension-wise
  diagnostic shows `per_column_all` has **no** advantage on the latent dimension a
  comparator already covers correctly (dim2 vs `single_gaussian`, 5/10, −0.0095; dim3 vs
  `single_poisson`, 7/10, +0.059). Its whole-space advantage is assembled from the
  dimensions each single-family model cannot see. This is the mechanism the complementary
  generator was built to expose, and it behaved as designed.
- **The blocks are not equally informative, and that shapes the result.** The Gaussian block
  carries 0.858 of the true X-side precision trace and 53.9x the Bernoulli block. Consistent
  with that, `single_gaussian` is by far the strongest single-family model, and dim2 is the
  best-recovered dimension in every condition that observes the Gaussian block (0.28–0.29).
  **Do not read the joint advantage as evidence that all three families contribute equally.**
- **A concrete caveat the diagnostics revealed.** In the true generator the Gaussian block
  contributes **more** precision to dim1 (0.959) than the Bernoulli block's own on-dimension
  contribution (0.587), because minor-weight leakage is multiplied by the Gaussian block's
  much larger curvature. So dim1 is not cleanly "the Bernoulli dimension" in precision terms
  even though it is in loading terms — and dim1 is indeed the worst-recovered dimension
  everywhere (0.92 at best). The complementarity is clean for dim2 and dim3 and only
  partial for dim1.
- **The forced-Gaussian control does not settle the family question.** At sparse Y,
  per-column beats `all_gaussian` on `RMSE_Z` in 8/10 trials but on held-out likelihood in
  only 5/10, and the two arms also differ in their M-step optimizer. This contrast is
  reported, but no family-assignment conclusion is drawn from it.

**Relation to prior evidence.** The Issue #28 audit recorded the sparse-Y interaction on a
single generative configuration (n=80, d=9, k\*=2, dense random F). It is reproduced here in
a second, deliberately complementary configuration, with the objective-consistent numerics.
It is **not** reproduced as a claim about the size of the effect at dense Y, where it is
small in both configurations.

---

## 9. LIMITATION

1. **Deliberately constructed complementary blocks.** The generator was built so that each
   family block loads mainly on a different latent dimension. External validity is limited
   and no real-data claim follows.
2. **One synthetic family configuration**, `n = 80`, `K_TRUE = 3` fixed, **10 trials only**.
   Effect sizes are reported; no p-values or significance claims are made, and none were
   pre-specified.
3. **`K_TRUE = 3` here versus `k* = 2` in the existing sparse-Y evidence.** This experiment
   differs from that evidence in **two** respects — complementary F structure *and* latent
   dimension — so a difference in outcome **must not** be attributed to complementarity
   alone.
4. **Family / dispersion / link-induced local-curvature imbalance under this pre-registered
   generator.** The per-observation `A''(eta)/phi` is 11.11 (Gaussian) / 1.64 (Poisson) /
   0.21 (Bernoulli). This is **not** a pure family effect: the Gaussian value depends on the
   pre-registered `sigma_G = 0.3` (`A''/phi = 1/sigma^2`) and the Bernoulli and Poisson
   values depend on the eta distribution. It was measured and retained, not removed.
5. **`all_gaussian` vs `per_column_all` is a same-column misspecification contrast with the
   family specification and the M-step optimizer path confounded** (analytical closed form
   vs weighted Adam). It is not a pure family-assignment effect.
6. **`single_*` vs `per_column_all` also differs in the number of observed X columns** (3 vs
   9), not only in family integration. That is part of the research question but it is not a
   pure family-assignment contrast.
7. **The Poisson X marginal variance exceeds its mean** (mean 1.62, var 5.50) purely because
   of latent heterogeneity: for a conditionally Poisson generator
   `Var(X) = E[mu(Z)] + Var(mu(Z))`. This is **not** evidence of Poisson overdispersion.
8. **The latent dimensions are not exactly orthogonal in finite samples** (max
   `|corr| = 0.293` over the 10 trials). No whitening was applied and no seed was changed.
   Recovery of a dimension a block does not own can be partly explained by this.
9. **The precision-trace quantities are plug-in trace contributions at the final estimate**,
   not posterior information, and they were never used for weighting.
10. **The sparse-Y condition intentionally increases the potential value of X.** It is a
    stress condition, not a typical operating point.
11. **The per-column model remains an experimental prototype**; this result does not promote
    it to the thesis/main method, and the repository's Q-based criterion was deliberately
    not used to rank conditions (families and observed X dimensions differ, so criterion
    values are not comparable across conditions).

---

## 10. Artifacts

```
tools/research_audit/run_complementary_blocks_consistent.py
expfam/results/story_diagnostics/complementary_blocks_consistent_20260821_summary.csv
                                  ..._agg.csv
                                  ..._paired.csv
                                  ..._runinfo.csv
                                  ..._generator.csv
                                  ..._blockdiag.csv
figures/story_diagnostics/complementary_blocks_consistent_20260821_rmse_z.png
                          ..._dimwise_rmse.png
                          ..._test_y_ll.png
```

Smoke (1 trial x 2 rates x 6 conditions = 12 fits) was run first to a location outside the
repository and is deliberately **not** a tracked artifact. It passed every integrity
criterion and no scientific claim is based on it.

## Research integrity

- New model fits run: **YES — 120, all pre-registered**
- Model / runner / existing tests changed: **NO**
- Historical results overwritten: **NO**
- Seeds changed, dropped or retried after a failure: **NO** (no failure occurred)
- Design changed after seeing results: **NO**
- Real-data claim: **NO**
- Prototype promoted to thesis/main method: **NO**
