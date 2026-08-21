# Matched Latent-Coverage Ablation (Issue #31)

## Scope

GitHub Issue #31 is the sole authoritative specification for this experiment. Every design
constant, the primary domain, the primary endpoint, the primary comparator, the single
primary estimand, the mandatory decomposition, the seed policy and the interpretation rules
were fixed **before** any fit was run and were not changed afterwards.

> **Research question.** Does the joint per-column advantage become substantially smaller
> when each individual family already covers all latent dimensions, while family
> composition, column count, loading magnitudes, Gaussian sigma, K, Y information and the
> fitting procedure are held fixed?

This experiment **more tightly targets latent-coverage / block-rank geometry while holding
the major previously identified factors fixed**. It does **not** isolate that factor alone.

- Base commit: `c55638f8e9124b65a0d27c67f4c8c9ba088937ca`
- Branch: `experiment/31-matched-latent-coverage-ablation`
- Script: `tools/research_audit/run_matched_latent_coverage_ablation.py`
- Fits: **180** (2 regimes x 2 rates x 4 X conditions x 10 trials, plus 2 rates x shared
  `y_only` x 10 trials), all `numerics_mode="consistent"`
- Model code, runner, existing tests, historical results: **unchanged**

`DualExpFamLSMPerColumnConsistent` remains an **experimental prototype**. Nothing here
promotes it, and no real-data claim follows.

---

## 1. VERIFIED

Facts checked directly from code, hashes, asserts or the produced artifacts.

### 1.1 Design integrity

- **180/180 fits completed.** Exactly 10 rows per `(regime, rate, condition)` cell (16
  cells) and exactly 10 `y_only` rows per rate. No duplicate
  `(trial, rate, regime, condition)` key.
- **`numerics_mode == "consistent"` in 180/180**, asserted from each returned result.
  `clip_diag.status == "not_applicable"` in 180/180, i.e. no hard Poisson clipping path.
- **`internal_retry_detected` = 0 / 180.** `run_em_experimental` retries internally with a
  different seed and a halved Newton alpha and resets `nan_count` per retry, so
  `res["nan_occurred"]` can be `False` after a NaN reset. Every fit was therefore run with
  `verbose=True` inside `contextlib.redirect_stdout` and the runner's literal
  `"[NaN iter="` reset message was searched for. It never appeared.
- **`nan_occurred` = 0, `nan_count` = 0, `q_bic_failed` = 0, `failure_reason` null in every
  row, 0 warnings** captured by `warnings.catch_warnings(record=True)`.
- **Support validation** passed in every condition
  (`validate_support=True, allow_support_mismatch=False`).
- **Pairing.** Within each trial the `Z_true`, `Y` and test-mask hashes are identical across
  both regimes and all conditions; within each `(trial, rate)` the train-mask hash is
  identical; the model seed is identical across both regimes.
- **All scientific metrics finite.**
- **Precision-trace identity** `sum_j t_ilj == t_il`: max abs err `4.26e-14`.
- **Decomposition identity** `I_t == D_G,t - D_J,t`: max abs err **`5.55e-17`**.

### 1.2 Generator-only pre-fit gate — all 10 trials

Every Issue #31 acceptance criterion passed before any fit was started.

| # | criterion | result |
|---|---|---|
| 1 | row-norm equality between regimes (`rtol=0, atol=1e-15`) | PASS |
| 2 | absolute component multiset equality (exact) | PASS |
| 3 | intended source and target dominant dimension per row | PASS |
| 4 | block precision eigenvalues recorded | PASS |
| 5 | block trace recorded | PASS |
| 6 | block `coverage_index` recorded | PASS |
| 7 | block `effective_rank` recorded (fixed `Pbar_b` definition) | PASS |
| 8 | Gaussian `trace_comp == trace_full` strict tolerance | PASS (max abs err `2.13e-14`) |
| 9 | Bernoulli / Poisson trace drift recorded, no rescaling | PASS |
| 10 | shared `Z_true` / `Y` / `eta_y`; Y symmetric; zero diagonal; only `i<j` sampled | PASS |
| 11 | finite eta and mu everywhere; no clipping | PASS |
| 12 | latent cross-dimension correlations recorded | PASS |
| 13 | family-specific RNG spawn provenance recorded | PASS |
| 14 | inverse-CDF / common-U coupling monotonicity — Bernoulli and Poisson | PASS / PASS |
| 15 | `X_pois` finite, `>= 0`, integer-valued | PASS |
| provenance | `gaussian_common_noise_source_equal`, `gaussian_common_noise_hash_equal` | PASS / PASS |

### 1.3 The manipulation did what it was designed to do

TRUE generator block precision `Pbar_b`, mean over 10 trials. For a 3-row block
`coverage_index <= 1/3` and `effective_rank <= 3`, so ~0.22 and ~2.7 mean "close to
isotropic / near full rank", not "poor".

| block | regime | eig1 | eig2 | eig3 | trace | coverage_index | effective_rank |
|---|---|---:|---:|---:|---:|---:|---:|
| bernoulli | complementary | 0.5962 | 0.0208 | 0.0034 | 0.6204 | **0.00554** | **1.084** |
| bernoulli | full_coverage | 0.2743 | 0.2013 | 0.1463 | 0.6219 | **0.23527** | **2.798** |
| gaussian | complementary | 31.9292 | 1.2904 | 0.1138 | **33.3333** | **0.00341** | **1.090** |
| gaussian | full_coverage | 15.4174 | 10.4741 | 7.4419 | **33.3333** | **0.22326** | **2.704** |
| poisson | complementary | 4.7012 | 0.1896 | 0.0096 | 4.9004 | **0.00199** | **1.086** |
| poisson | full_coverage | 2.2747 | 1.6138 | 1.0222 | 4.9107 | **0.20910** | **2.701** |

- **Gaussian total local precision is held exactly fixed** (trace 33.3333 in both regimes,
  max abs err `2.13e-14`) while its orientation changes from near rank-1 to near-isotropic.
- Block trace drift, complementary vs full coverage: bernoulli **-0.240%**, gaussian
  **+0.000%**, poisson **-0.210%**. Recorded, never rescaled.
- Latent cross-dimension correlation: mean `|corr|` 0.1056, **max 0.2943** over 10 trials.
  No whitening was applied and no seed was changed.

### 1.4 Sampling provenance

- **Poisson sampling:** common-random-number inverse-CDF coupling, with q=0 replaced only by
  `nextafter(0,1)` for floating-point endpoint safety; **no eta/mu clipping**.
- **Gaussian sampling:** `X_gauss = eta_gauss + SIGMA_G * E_gauss` with the **same
  parameter-free pre-drawn `E_gauss`** shared across both regimes, verified by exact
  source-array identity and hash equality — never by subtracting eta back out of a
  floating-point X.
- **Bernoulli sampling:** `X_bern = (U_bern < sigmoid(eta_bern))` with the same pre-drawn
  `U_bern`.
- Spawn policy: `np.random.SeedSequence(120000 + trial).spawn(6)`, fixed order
  `Z | F | X_bern | X_gauss | X_pois | Y`.
- **Bitwise reproduction of Issue #27 is NOT claimed** — the sampling implementation path
  differs. The marginal generator distribution is the same statistical model.

### 1.5 Validator correction during the pre-fit gate (transparency record)

During the pre-fit generator gate, an **implementation-only validator** incorrectly required
bitwise equality of Gaussian residuals reconstructed as `X - eta` across regimes.
**No model fit had been run.** Human review determined that this predicate was **not part of
the pre-registered Issue #31 specification** and is invalid under floating-point rounding:
`fl(fl(eta + noise) - eta)` need not equal `noise`, and the observed discrepancy was
`2.220446e-16` = 1 ULP, below `eps * max|eta| = 5.57e-16`.

It was replaced by a **direct exact identity/hash check of the shared pre-drawn Gaussian
noise array `E_gauss`**. The complete generator-only gate was then rerun from the beginning
for all 10 trials. **No seed, parameter, generator distribution, endpoint, model fit, or
scientific result was changed.** This was **not** a scientific failure.

---

## 2. OBSERVED — primary result (`y_obs_rate = 0.1`, comparator `single_gaussian`)

Exactly one primary estimand. All quantities are trial-matched over 10 paired trials.
`delta_G(r,t) = RMSE_Z(single_gaussian) - RMSE_Z(per_column_all)`, positive = joint better.
`I_t = delta_G(comp,t) - delta_G(full,t)`.

| quantity | mean | std | median | sign count |
|---|---:|---:|---:|---:|
| `delta_G_complementary` | +0.2280 | 0.0956 | +0.2351 | 10/10 > 0 |
| `delta_G_fullcoverage` | +0.1141 | 0.0496 | +0.1060 | 10/10 > 0 |
| **`I` (PRIMARY)** | **+0.1139** | **0.0915** | **+0.1364** | **9/10 > 0** |
| `D_G` (comparator shift) | +0.3151 | 0.0756 | +0.3111 | 10/10 > 0 |
| `D_J` (joint shift) | +0.2012 | 0.0745 | +0.1814 | 10/10 > 0 |

Identity `I == D_G - D_J` verified, max abs error **`5.55e-17`**.

Levels (RMSE_Z, mean over 10 trials):

| regime | condition | RMSE_Z | sd | dim1 | dim2 | dim3 | test Y LL |
|---|---|---:|---:|---:|---:|---:|---:|
| complementary | per_column_all | 0.6219 | 0.0811 | 0.8232 | 0.2767 | 0.6336 | -2.3768 |
| complementary | single_gaussian | 0.8499 | 0.0975 | 1.0132 | 0.3180 | 1.0071 | -2.5614 |
| complementary | single_poisson | 1.0491 | 0.0682 | 1.1519 | 1.1931 | 0.7229 | -2.9129 |
| complementary | single_bernoulli | 1.1125 | 0.0735 | 0.9059 | 1.2341 | 1.1584 | -2.9165 |
| full_coverage | per_column_all | 0.4208 | 0.0277 | 0.4327 | 0.4027 | 0.4219 | -2.2158 |
| full_coverage | single_gaussian | 0.5349 | 0.0560 | 0.5458 | 0.5108 | 0.5380 | -2.3049 |
| full_coverage | single_poisson | 0.8525 | 0.0635 | 0.8319 | 0.8776 | 0.8359 | -2.6810 |
| full_coverage | single_bernoulli | 1.1153 | 0.0646 | 1.1194 | 1.1311 | 1.0813 | -2.9260 |
| shared | y_only | 1.1739 | 0.0502 | 1.1603 | 1.1872 | 1.1679 | -3.0160 |

**`I` must not be read alone.** The decomposition shows that under full coverage the
comparator improved by `D_G = +0.315` **and the joint model also improved substantially, by
`D_J = +0.201`**. The interaction is the residual of two large same-signed shifts, not a
case of "the comparator caught up while the joint model stood still".

## 3. OBSERVED — dense-Y negative control (`y_obs_rate = 1.0`)

| quantity | mean | std | median | sign count |
|---|---:|---:|---:|---:|
| `delta_G_complementary` | +0.0055 | 0.0169 | +0.0062 | 7/10 |
| `delta_G_fullcoverage` | -0.0004 | 0.0089 | +0.0020 | 6/10 |
| `I` | +0.0058 | 0.0207 | +0.0055 | 6/10 |
| `D_G` | +0.0241 | 0.0179 | +0.0262 | 9/10 |
| `D_J` | +0.0183 | 0.0101 | +0.0150 | 10/10 |

At dense Y every quantity collapses by roughly an order of magnitude and the sign counts
become close to a coin flip. The negative control behaves as a negative control.

## 4. OBSERVED — pre-specified secondary interactions (`y_obs_rate = 0.1`)

| comparator | `delta_G_comp` | `delta_G_full` | `I` | sign | `D_G` | `D_J` |
|---|---:|---:|---:|---:|---:|---:|
| `single_gaussian` (primary) | +0.2280 (10/10) | +0.1141 (10/10) | **+0.1139** | 9/10 | +0.3151 | +0.2012 |
| `single_bernoulli` | +0.4905 (10/10) | +0.6946 (10/10) | **-0.2041** | **0/10** | -0.0029 (5/10) | +0.2012 |
| `single_poisson` | +0.4271 (10/10) | +0.4318 (10/10) | **-0.0046** | 6/10 | +0.1965 | +0.2012 |

The two secondary interactions do **not** follow the primary. For `single_bernoulli` the
interaction is **negative in 10/10 trials**: the Bernoulli arm barely moved between regimes
(`D_G = -0.003`, 5/10) while the joint model improved by `+0.201`, so the joint advantage
over Bernoulli **grew** under full coverage. For `single_poisson` the two shifts almost
exactly cancel (`+0.197` vs `+0.201`), giving an interaction indistinguishable from zero.

At dense Y both secondary interactions are also negative
(`single_bernoulli` -0.0252, 0/10; `single_poisson` -0.0161, 2/10).

## 5. OBSERVED — held-out Y

Paired held-out Y mean log-likelihood per pair (positive = joint better):

| rate | regime | vs `single_gaussian` | vs `single_bernoulli` | vs `single_poisson` | vs `y_only` |
|---|---|---:|---:|---:|---:|
| 0.1 | complementary | +0.1846 (10/10) | +0.5397 (10/10) | +0.5361 (10/10) | +0.6392 (10/10) |
| 0.1 | full_coverage | +0.0891 (9/10) | +0.7102 (10/10) | +0.4652 (10/10) | +0.8001 (10/10) |
| 1.0 | complementary | +0.0035 (4/10) | +0.0172 (8/10) | +0.0164 (7/10) | +0.0208 (8/10) |
| 1.0 | full_coverage | **-0.0082 (3/10)** | +0.0344 (9/10) | +0.0145 (7/10) | +0.0308 (9/10) |

The held-out likelihood reproduces the RMSE_Z ordering, including the shrinkage of the
joint-vs-Gaussian gap under full coverage, and shows the joint model **losing** to
`single_gaussian` at dense Y under full coverage (3/10).

## 6. OBSERVED — dimension-wise recovery and fitted precision

Dimension-wise RMSE uses the **same** whole-space Procrustes rotation (no per-dimension
oracle alignment). In the complementary regime the recovery is strongly anisotropic —
`single_gaussian` recovers dim2 well (0.318) and dim1/dim3 poorly (1.013 / 1.007) — while in
full coverage every condition becomes nearly isotropic (`single_gaussian` 0.546 / 0.511 /
0.538). `per_column_all` goes from (0.823, 0.277, 0.634) to (0.433, 0.403, 0.422).

Fitted X-side block trace share for `per_column_all` at `y_obs_rate = 0.1`:

| regime | bernoulli | gaussian | poisson | gaussian `effective_rank` |
|---|---:|---:|---:|---:|
| complementary | 0.0211 | **0.8976** | 0.0813 | 1.081 |
| full_coverage | 0.0297 | **0.7791** | 0.1912 | 2.509 |

The Gaussian block dominates the fitted X-side precision in both regimes, and its fitted
effective rank tracks the generator manipulation (1.08 -> 2.51).

## 7. INTERPRETATION

**Outcome classification: A, with an important qualification from the mandatory
decomposition and a contradiction in the secondary arms.**

- **Primary.** The joint advantage over `single_gaussian` at sparse Y is roughly halved when
  the Gaussian block already spans all three latent dimensions: `delta_G` falls from +0.228
  to +0.114, giving `I = +0.114` (9/10 trials, median +0.136). Following the pre-registered
  decision rule, the strongest allowable statement is:

  > The joint advantage was larger under the complementary / lower-rank family-block
  > geometry, and the pre-specified decomposition shows how much of that interaction came
  > from the single-Gaussian arm versus the joint arm.

  Concretely: of the interaction, the comparator shift contributes `D_G = +0.315` and the
  joint shift contributes `D_J = +0.201`. **Both arms improved substantially under full
  coverage.** The result is *consistent with* a latent-coverage / block-rank geometry effect
  in this controlled synthetic setting.

- **The secondary arms do not support a simple coverage story.** The `single_bernoulli`
  interaction is **negative in 10/10 trials** and the `single_poisson` interaction is
  **approximately null**. A pure "coverage explains the joint advantage" account would
  predict the same sign for all three comparators. It does not hold here: what changed
  between regimes was not only each comparator's coverage but also the joint model's own
  precision geometry, which improved by the same `+0.201` against every comparator. Whether
  a comparator's interaction comes out positive, null or negative is therefore governed by
  how much *that* comparator itself gained from full coverage — `+0.315` (Gaussian),
  `+0.197` (Poisson), `-0.003` (Bernoulli).

- **The dense-Y negative control behaves correctly**: all interactions collapse to within
  about 0.02 RMSE_Z with near-even sign counts, and the joint model even loses slightly to
  `single_gaussian` on held-out likelihood under full coverage.

- **The manipulation was clean at the generator level.** Row norms and absolute component
  multisets were preserved exactly; the Gaussian block's total true precision trace was held
  fixed to `2.13e-14`; only the orientation changed (coverage_index 0.0034 -> 0.2233,
  effective rank 1.09 -> 2.70).

**This is not a causal proof.** The words "fully isolated", "isolated alone" and "dimension
assignment alone caused the effect" are not applicable, and no universal per-column
superiority is claimed.

---

## 8. LIMITATION

1. **Not isolated alone.** Bernoulli and Poisson curvature remains eta-dependent; their true
   block traces are only approximately matched (drift -0.240% / -0.210%); finite-sample
   latent correlation remains (max `|corr|` 0.2943); and the joint model's own precision
   geometry changes across regimes (fitted Gaussian effective rank 1.08 -> 2.51).
2. **The interaction does not hold the joint model fixed.** `I = D_G - D_J` is an exact
   identity and `D_J = +0.201` is large. `I` must never be reported alone.
3. **The manipulation also changes the comparator's block rank**, not only its "coverage":
   generator effective rank goes 1.09 -> 2.70 per block. That is a construction fact.
4. **The secondary interactions contradict a single-mechanism reading** (`single_bernoulli`
   negative 10/10, `single_poisson` null). This is reported as-is and is not reinterpreted
   as success.
5. **Deliberately constructed geometry.** Both regimes are synthetic constructions; external
   validity is limited and no real-data claim follows.
6. **One synthetic configuration**, `n = 80`, `K_TRUE = 3`, **10 trials only**. Effect sizes
   only; **no p-values** and none were pre-specified.
7. **`K_TRUE = 3` here versus `k* = 2`** in the pre-Issue-#27 sparse-Y evidence.
8. **No bitwise reproduction of Issue #27** is claimed (different sampling implementation
   path; same statistical model).
9. **The precision quantities are plug-in / generator orientation diagnostics**, not
   posterior information, and were never used for weighting. For a 3-row block
   `coverage_index <= 1/3` and `effective_rank <= 3`.
10. The repository's Q-based criterion was **not** used to rank conditions; the `bic` field
    is recorded as diagnostic only.
11. **The per-column model remains an experimental prototype.**

## 9. Decision gate

The primary interaction is positive (9/10) but the mandatory decomposition and the two
secondary arms show that latent coverage is **not** a single sufficient explanation of the
Issue #27 result. Under the pre-registered gate this is not a clean "coverage interaction
clear" outcome, and it is certainly not a null that would justify withdrawing the story
outright.

The pre-registered instruction in both cases points the same way: **do not redesign F to
force success and do not run further synthetic mechanism tuning.** The residual information
imbalance between blocks (Gaussian carries ~0.86 of the true X-side precision trace) remains
a candidate obstacle to interpretation; per the gate this is **mentioned as a possible future
information-balance experiment only and is not created here.** The next scientific direction
is leakage-free real-data validation.

## 10. Artifacts

```
tools/research_audit/run_matched_latent_coverage_ablation.py
expfam/results/story_diagnostics/matched_latent_coverage_ablation_20260821_summary.csv
                                  ..._agg.csv
                                  ..._paired.csv
                                  ..._interaction.csv
                                  ..._runinfo.csv
                                  ..._generator.csv
                                  ..._blockdiag.csv
figures/story_diagnostics/matched_latent_coverage_ablation_20260821_rmse_z.png
                          ..._interaction.png
                          ..._dimwise_rmse.png
                          ..._coverage_spectrum.png
```

Smoke (18 fits) was run first to a location outside the repository and is deliberately
**not** a tracked artifact. It passed every integrity criterion and no scientific claim is
based on it.

## Research integrity

- New model fits run: **180, all pre-registered**
- Seeds changed, dropped, retried or rescued: **NO** (no failure occurred)
- Design, endpoint, estimand or tolerance changed after seeing results: **NO**
- Model / runner / existing tests changed: **NO**
- Historical results overwritten: **NO**
- Issue #27 artifacts touched: **NO**
- Real-data claim: **NO**
- Prototype promoted to thesis/main method: **NO**
