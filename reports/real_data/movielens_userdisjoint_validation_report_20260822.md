# User-disjoint real-data validation on MovieLens 100K with train-only movie selection and train-only derived attributes

**Issue:** #33 (pre-registered; the live Issue body is the authoritative frozen specification)
**Branch:** `experiment/33-movielens-userdisjoint-validation`
**Design base:** `e4be01afd1e911ee0d6bed491166258a07af1f0f`
**Run:** 2026-08-22, 360 fits, 65.7 min
**Lineage:** `expfam/src/experimental/em_runner.py` + `model_dual_expfam_consistent.py`
(`DualExpFamLSMPerColumnConsistent`, `numerics_mode="consistent"`; per-column is a **prototype**,
extra-1/2-なし fixed 系列の派生). Model and runner code were **not** modified.

---

## 1. Scope and what this is not

This is a **user-disjoint real-data validation with train-only movie selection and train-only
derived attributes**.

The primary real-data contrast evaluates a heterogeneous-X Bernoulli/Gaussian configuration.
It does not validate simultaneous Bernoulli/Gaussian/Poisson X integration on real data.
This experiment prevents further test-user leakage inside the new protocol, but
it is not an untouched external confirmation.

---

## 2. Why the previous MovieLens evidence was not usable

Established by the read-only audit on `main = e4be01af`, from source and by recomputation
against the tracked `expfam/data/ml-100k.zip`:

| | Leakage class | Old pilot | Evidence |
|---|---|---|---|
| A | model-training | NOT PRESENT | `make_pair_split` + `train_mask` give a strict pair-level held-out |
| B | attribute | **PRESENT** | `mean_rating` / `ratings_count` computed before any split from the full log (`run_movielens_attribute_diagnosis.py:103-119`); `corr(ratings_count, Y_rowsum) = 0.628` |
| C | selection | **PRESENT** | `prepare_movielens_data.py:428` computes `rpm` over ALL of `u.data`; `select_genre_stratified` (L.134-176) ranks by `rpm[m]` descending inside a 30–200 window (L.161-167). All 100 pilot movies sit at or above the **73.6th percentile** of full-data popularity (median 89.3rd, 42/100 in the top decile); selected mean `ratings_count` 154.4 vs 59.5 over all 1682 movies |
| D | preprocessing | **PRESENT** | z-scores use full-data moments; `high_count_threshold = np.median(y_upper)` sets the AUC/AP positive class from the evaluation-side Y itself |
| E | dependence w/o leakage | PRESENT | same users, same 214-day window |
| F | model-selection | POTENTIAL | the log-count representation was introduced after seeing an earlier outcome |

**Consequence, and the reason the old 100-movie subset was not reused:** a user split repairs
A/B/D but **not** C. Keeping those 100 IDs fixed would leave test-user rating events inside the
subset-selection step.

### What this experiment removes, and what it does not

| Class | Status here | Mechanism |
|---|---|---|
| A model-training | NOT PRESENT | `Y_test` is a separate matrix; only `Y_train` is ever passed to `run_em_experimental` (enforced per fit, §5) |
| B attribute | NOT PRESENT | every train-derived attribute takes a `tag="train"` `EventView` as its only rating-event input |
| C selection | NOT PRESENT | movie selection is train-only, re-run per split |
| D preprocessing | NOT PRESENT | z-scores use that split's train-derived values; **no AUC/AP is computed**, so no test-derived threshold exists; `mu_y` is never rescaled |
| E dependence | **PRESENT, deliberately** | not removable; bounds the claim (§8) |
| F model-selection | **PRESENT, disclosed and frozen** | see limitation 2 |
| G split-design | NOT PRESENT | the split is a plain permutation of user IDs; no rating-derived quantity enters it |

---

## 3. Protocol as executed

**Split.** `SPLIT_SEED_BASE = 130000`; per split `s`, `default_rng(130000 + s).permutation(943 uids)`
→ 471 train / 471 test / 1 unused. Disjointness and union asserted on all 30 splits.
No activity blocking; no rating-derived quantity enters the split.

**Movie selection (train-only, per split).** `rate_train[m] = |train events on m| / 471`,
genre-stratified 10 target genres × 10 movies, fixed historical rate thresholds 30/943 and
200/943 with the 20/943–300/943 and 10/943–500/943 fallbacks, **not retuned**. Those constants
come from the 2026-06 full-data pilot design (`prepare_movielens_data.py:58-59`); their
provenance is disclosed here. All 30 splits yielded exactly 100 movies with 10 per genre.

**Attributes.** `genre19` Bernoulli (u.item flags, values asserted in {0,1}); `year` Gaussian
(title `(YYYY)`, z-scored over the selected nodes); `mean_rating_train` Gaussian (train users,
z-scored); `log_count_train` Gaussian (`z(log1p(train count))`); `count_train_raw` Poisson
(diagnostic condition only).

**Y.** `Y_train` / `Y_test` = co-rating counts over the train / test users on the same selected
nodes; `train_mask = None` (all 4950 pairs observed). No `Y_test` rescaling, no test-derived
threshold, no AUC/AP.

**Model.** `K = 3` fixed (design constant, not selected from data), `L = 5`, `num_iter = 8`,
`family_y = "poisson"`, `numerics_mode = "consistent"` on 360/360 fits.
`MODEL_SEED_BASE = 131000`, seed `= 131000 + s*10 + m`.

**Conditions (exactly 6).** `y_only` (0 cols used; genre19 passed with `fix_x=True`, F fixed at 0),
`genre_only` (19), `genre_year` (20), `genre_logcount_train` (20), `mixed_train_log` (22),
`mixed_train_raw_poisson` (22, diagnostic only).

### Exposure diagnostics — record only, nothing dropped

The gate reproduced the values pre-registered in Issue #33 **exactly**:

```
log_event_ratio     mean +0.0036   sd 0.0581   |max| 0.1962
Y_test zero pairs   0 – 7 of 4950      Y_train zero pairs 0 – 6 of 4950
per-movie test events, minimum over all splits: 22   (no movie had zero)
most asymmetric split s=2: 45076 train / 54848 test events, log_event_ratio +0.1962,
                           Y_train_mean 21.70 vs Y_test_mean 28.78     — KEPT
```

No split was dropped, redrawn or rebalanced. `unused_uid` differs in all 30 splits; the 30
`train_uid_hash` and 30 `test_uid_hash` values are all distinct with zero overlap.

---

## 4. Results

**Trial unit = user split.** The two model seeds inside a split are averaged first; `n = 30`,
never 60. Percentiles are `np.percentile(v, [10, 90], method="linear")`, pre-registered.

### Primary

> Over 30 repeated user-disjoint splits, Delta had mean +0.004248, median +0.006875,
> range [-0.029884, +0.020767], and was positive in 23/30 splits.
> Across-split spread (std) was 0.012276; |mean Delta| was 0.004248;
> std/|mean Delta| was 2.890071.
> The empirical 10th-90th percentile range across the 30 repeated splits was
> [-0.009931, +0.016536].
> The decomposition components were A: mean +0.002009, 22/30 positive;
> B: mean +0.002239, 19/30 positive; the identity Delta = A + B held to 0.000e+00.
> The descriptive positive control (genre_only - y_only) had mean +0.012437,
> 25/30 positive.
> No significance test, confidence interval, or power analysis was performed; the 30
> splits reuse the same 943 users and are not independent replicates.

`Delta_s = LL_test(mixed_train_log, s) − LL_test(genre_only, s)`, Poisson mean log score per
pair over all 4950 pairs of `Y_test`.

### Mandatory decomposition — Delta is not interpreted without A and B

| Component | mean | median | min | max | positive | std | std/\|mean\| | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Delta` (mixed − genre_only) | +0.004248 | +0.006875 | −0.029884 | +0.020767 | 23/30 | 0.012276 | 2.890071 | −0.009931 | +0.016536 |
| `A` (mixed − genre_logcount) | +0.002009 | +0.002783 | −0.020701 | +0.020560 | 22/30 | 0.010122 | 5.037997 | −0.012454 | +0.013967 |
| `B` (genre_logcount − genre_only) | +0.002239 | +0.001965 | −0.016676 | +0.024600 | 19/30 | 0.009346 | 4.174988 | −0.006835 | +0.015429 |
| `P` positive control (genre_only − y_only) | +0.012437 | +0.011100 | −0.018080 | +0.051359 | 25/30 | 0.013343 | 1.072826 | −0.002055 | +0.027185 |

Identity `Delta = A + B` held to `0.000e+00` across all 30 splits.

`A` and `B` are an **exact telescoping decomposition of conditional model contrasts**, not a
partition of Delta into isolated attribute effects:

```
A = LL_test(mixed_train_log)      - LL_test(genre_logcount_train)
B = LL_test(genre_logcount_train) - LL_test(genre_only)
Delta = A + B
```

The mean primary contrast decomposes into two conditional contrasts of similar average
magnitude: A = +0.002009 and B = +0.002239. Thus the observed mean Delta is not numerically
concentrated in B alone under this ordering of model comparisons. **This decomposition does not
isolate causal or unique attribute contributions, and interactions among attributes / model
parameters remain possible.** Each component has across-split spread 4–5× its own mean
(`std/|mean|` = 5.04 for A, 4.17 for B).

### Positive control

The descriptive positive control is directionally positive in 25/30 splits (mean +0.012437),
showing that this pipeline can produce an attribute-associated predictive contrast under this
protocol. It is about **2.9×** the primary Delta and is the only contrast whose spread is close
to its mean (ratio 1.07). This is a descriptive sanity check only; no significance test was
performed on it, and it is not evidence of a causal or "real" effect.

### Secondary (pre-registered, never promoted)

| Contrast | metric | mean | median | positive | std | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|
| `mixed_train_log − y_only` | test_y_ll | +0.016685 | +0.018746 | 28/30 | 0.012747 | +0.001633 | +0.029343 |
| `genre_year − genre_only` | test_y_ll | −0.002222 | +0.000572 | 16/30 | 0.012215 | −0.018209 | +0.007915 |
| `mixed_train_raw_poisson − mixed_train_log` | test_y_ll | **−0.100274** | −0.102664 | **1/30** | 0.051452 | −0.152302 | −0.049727 |
| `mixed_train_log − genre_only` | test_y_rmse | −0.012143 | −0.018482 | 11/30 | 0.044468 | −0.057064 | +0.020481 |
| `mixed_train_log − genre_only` | test_spearman | +0.000578 | +0.000950 | 20/30 | 0.002508 | −0.001364 | +0.003345 |
| `mixed_train_log − genre_only` | train_y_ll | −0.000872 | −0.000986 | 5/30 | 0.001325 | −0.002618 | +0.000887 |

The raw-count Poisson diagnostic is the largest and most consistent effect in the whole
experiment: negative in **29 of 30** splits, roughly 24× the primary in magnitude. The direction
matches the earlier leaky pair-split diagnostic (−0.374 there), so it reproduces under the new
protocol at a smaller magnitude. It remains **diagnostic only** and is not the primary model.

`test_y_rmse` moves in the same direction as the primary (lower is better: mean −0.0121) while
being positive in only 11/30 splits; `train_y_ll` is slightly negative in 25/30, i.e. the mixed
model is not simply fitting `Y_train` harder.

### Condition means (test Poisson log score per pair, n = 30 splits)

| condition | test_y_ll mean | median | std | test_y_rmse | test_spearman |
|---|---:|---:|---:|---:|---:|
| `y_only` | −3.388823 | −3.296100 | 0.296966 | 6.593853 | 0.837781 |
| `genre_only` | −3.376385 | −3.288063 | 0.289667 | 6.565946 | 0.839928 |
| `genre_year` | −3.378608 | −3.286058 | 0.291901 | 6.576827 | 0.839554 |
| `genre_logcount_train` | −3.374147 | −3.283014 | 0.289870 | 6.558095 | 0.840210 |
| `mixed_train_log` | −3.372138 | −3.284399 | 0.291423 | 6.553802 | 0.840506 |
| `mixed_train_raw_poisson` | −3.472411 | −3.401990 | 0.283251 | 6.865947 | 0.816044 |

The between-split std of any single condition (≈0.29) is more than an order of magnitude larger
than every contrast between conditions. This is why the paired within-split difference is the
estimand and the per-condition level is not.

### Record-only diagnostic

Rank correlation between `log_event_ratio` and `Delta_s` across the 30 splits = **−0.0234**,
computed as `rankdata` + `np.corrcoef` (no `spearmanr`, no p-value, no significance
interpretation). Never used to drop, redraw or rebalance a split.

---

## 5. Integrity

All Issue #33 gates were enforced as hard STOPs inside the run. None fired.

| Gate | Result |
|---|---|
| fits | **360/360** exactly; 0 duplicates of `(split, model_seed, condition)`; no seed drop, no rescue retry, no post-result transformation change |
| internal retry (`"[NaN iter="` in redirected stdout, `verbose=True`) | **0** |
| `nan_occurred` / `nan_count` | **False / 0** |
| `q_bic_failed` | **False** (value unused; Q-based criterion not used for K or ranking) |
| warnings (`catch_warnings(record=True)`) | **0** |
| `validate_xy` support validation | passed on every fit |
| `numerics_mode == "consistent"` | **360/360** |
| finite metrics | all finite |
| `test_n_pairs` | 4950 on every fit |
| decomposition identity | `max|Delta − (A+B)| = 0.000e+00` |
| exposure diagnostics | all 13 fields recorded for all 30 splits |
| environment provenance | python 3.13.14, Windows-11-10.0.26200-SP0, numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, `requirements_sha256 = 1b773ed9…` |

### No-test-contribution checker — four layers, non-tautological

* **L1** structural guard with **use-time revalidation**: `EventView` is frozen but the DataFrame
  is mutable, so `_require_train()` recomputes `normalized_event_hash(ev.df)` on **every** use
  and compares it to the construction hash, re-checks `actual_uids ⊆ / == allowed_uids` and
  `len == 471`, and the ledger records the **recomputed** hash, never the stored string.
  `ItemMetadataView` asserts the exact allowed column set (mid, title, 19 genre flags), absence
  of any rating-derived column, unique mid, genre values in {0,1}, and hash consistency.
* **L2** resolved-annotation signature lint via `typing.get_type_hints` (not
  `parameter.annotation`): `select_movies` **pass**, `build_train_attributes` **pass** — exactly
  one `EventView`, exactly one `ItemMetadataView`, no raw DataFrame parameter, no banned
  event parameter, and no event data reachable through a closure cell or module global.
* **L3** falsification negative controls: **7 guards fired, all required.** Feeding the corpus
  view and the test view to both functions raised `ProvenanceError`; a one-value mutation of a
  deep copy raised "mutated after construction"; constructing an `ItemMetadataView` carrying a
  rating-derived column raised. **The checker is therefore not vacuous.**
* **L4** independent reference cross-check with the pre-registered EXACT/FLOAT split.
  EXACT quantities (movie_ids and order, UID sets, genre flags, parsed integer year, per-movie
  train count, per-movie train rating sum, `count_train_raw`, source-event and item-metadata
  hashes) matched exactly. FLOAT quantities compared with
  `np.allclose(rtol=0.0, atol=1e-12, equal_nan=False)` after asserting the integer sufficient
  statistics first:

  | block | n | `reference_float_max_abs_error` |
  |---|---:|---:|
  | `mean_rating_train_raw` | 210 | **0.000e+00** |
  | `mean_rating_train_z` | 210 | **0.000e+00** |
  | `year_z` | 210 | **0.000e+00** |
  | `log_count_train_z` | 210 | **0.000e+00** |

  The independent `np.bincount`/`np.unique` reference reproduced every float attribute
  bit-identically, comfortably inside the 1e-12 tolerance. **The tolerance was fixed before the
  run and was not relaxed.**

### Fit-time input ledger — non-vacuous

`provenance.csv` carries a `stage` column with evidence from both stages: **210 `structural`
rows** (all `tag = "train"`, `n_uids = 471`, recomputed hash == stored hash) and **360
`fit_time` rows**. Every `fit_time` row satisfied, immediately before `run_em_experimental`:
`x_input_hash == expected_x_provenance_hash`, `y_input_hash == y_train_hash`, and
`y_input_hash != y_test_hash`. `ledger_valid` is true on **360/360**, and
`len(fit_time rows) == n_fits == len(summary rows) == 360`, so no fit exists without a
validated ledger row. `Y_test`'s hash never appears as a fit input.

---

## 6. Reading of the result

The primary contrast is **positive in direction but small relative to its own across-split
spread**: mean +0.004248 with std 0.012276, i.e. `std/|mean| = 2.89`, positive in 23/30 splits,
and an empirical 10th–90th percentile range that spans zero. The mandatory decomposition gives
two conditional contrasts of similar average magnitude (A = +0.002009, B = +0.002239), so the
observed mean Delta is not numerically concentrated in B alone under this ordering of model
comparisons; neither component is large relative to its own spread. This decomposition does not
isolate causal or unique attribute contributions, and interactions among attributes / model
parameters remain possible.

The descriptive positive control is directionally positive in 25/30 splits (mean +0.012437,
spread ratio 1.07), showing that this pipeline can produce an attribute-associated predictive
contrast under this protocol. No significance test was performed on it.

No significance test, confidence interval, or power analysis was performed; the 30 splits reuse
the same 943 users and are not independent replicates. The numbers above are a repeated
user-disjoint split stability evaluation, not a population inference, and no qualitative label
is attached to the spread.

The clearest signal in the experiment is the diagnostic one: raw-count Poisson X degrades the
test score in 29 of 30 splits (mean −0.100274), reproducing the direction seen in the earlier
leaky pair-split diagnostic under the new protocol.

---

## 7. Deviations from the pre-registered specification

Two, both confined to a record-only diagnostic, both before any tracked result existed, and
neither touching a seed, condition, family assignment, `K`, split, estimand, tolerance or
phrase list:

1. **`log_event_ratio` sign convention.** The script as first written computed
   `log(n_train_events / n_test_events)`. Issue #33 pre-registers split `s=2`
   (45076 train / 54848 test) as **+0.1962**, i.e. `log(n_test / n_train)`. The script was
   corrected to the Issue's convention **before any fit was run**, and the gate then reproduced
   the pre-registered mean/sd/|max| and the split-2 values exactly. Only the sign of a
   record-only diagnostic was affected; the underlying event counts were identical either way.
2. **One console print label.** After the full run, a hard-coded console string that still read
   "train/test convention" was corrected to `log(n_test_events / n_train_events)`. This is a
   print label only. It changed **no computed value, no CSV, no figure**; the
   `runinfo.rank_corr_convention` field already recorded the correct convention at run time.

An earlier interrupted attempt (2026-08-22 02:20–02:30) wrote the script and then stopped at a
Claude session limit. It ran **zero fits** and produced no result artifact, so nothing was
re-run, rescued or duplicated.

---

## 8. Limitations

1. The primary real-data contrast evaluates a heterogeneous-X Bernoulli/Gaussian configuration:
   genre columns are Bernoulli and year / mean_rating_train / log_count_train are Gaussian.
   It does not validate simultaneous Bernoulli/Gaussian/Poisson X integration on real data.
   The raw-count Poisson condition `mixed_train_raw_poisson` is DIAGNOSTIC ONLY and is NOT the
   primary model.
2. The representation hypothesis was selected after earlier diagnostic analysis on the same
   MovieLens 100K corpus (raw Poisson count → degradation; log-count Gaussian → degradation
   removed).
3. This experiment prevents further test-user leakage inside the new protocol, but
   it is not an untouched external confirmation.
4. No transformation, family, K, or condition changed after Issue #33 was created.
5. The 30 repeated splits reuse the same 943 users and the same raw events; they are not
   independent replicates and no inferential statistic is computed from them.
6. Category E dependence remains and is not meant to be removed: train and test users rated the
   same movies over the same 214-day window. `corr(count_full, count_train) = 0.943`;
   `corr(mr_full, mr_train) = 0.976`. The claim is generalisation to a DISJOINT USER GROUP
   WITHIN the same collection period, not to an independent population.
7. The movie node set differs between splits (consecutive overlap 70–85). "Same movie nodes"
   holds within a split only.
8. The 30/200 selection constants originate from the 2026-06 full-data pilot design.
9. Y is overdispersed (var/mean 5.0–5.6 on the split halves, 9.9 on full data, KI-012).
   The Poisson per-pair log-likelihood is used as a score, not as evidence of correct specification.
10. Exposure mismatch between the train and test user groups is recorded, not controlled; the
    most asymmetric split (s=2) has `log_event_ratio` +0.1962.
11. An activity-blocked matched-pair user split would reduce that mismatch (measured: sd of
    `log_event_ratio` 0.0581 → 0.0024) and is left as a future sensitivity candidate; it is NOT
    run here because it would use test users' total rating activity in the design.
12. `K = 3` is a fixed design constant; K dependence is not established.
13. The Q-based criterion is not Schwarz BIC (KI-010) and is used for neither K selection nor
    condition ranking.
14. The per-column model remains a PROTOTYPE (CLAUDE.md sections 1 and 3). No manuscript
    adoption; no claim about real-data performance of the method beyond this protocol.

---

## 9. Allowed and prohibited claim language

Allowed: "user-disjoint real-data validation with train-only movie selection and train-only
derived attributes"; "evaluation-user-disjoint validation with train-only construction";
"a Bernoulli/Gaussian heterogeneous-X configuration was evaluated under the user-disjoint
protocol"; "artificial-data evidence separately covers the Gaussian/Bernoulli/Poisson
complementary-block configuration" (Issue #27 / #31 line — reported separately, never in the
same claim, table, or figure as this real-data result).

<!-- PROHIBITED-CLAIM-LIST:BEGIN -->

```
FORBIDDEN_CLAIM_PHRASES = [
    "first genuinely leakage-free real-data validation",
    "fully independent confirmatory MovieLens test",
    "untouched external confirmation",
    "leakage-free validation",
    "all three X families are validated on MovieLens",
    "Bernoulli/Gaussian/Poisson joint integration improves real-data performance",
    "the three-family per-column method was validated on real data",
    "the Poisson model fits",
    "we predicted unseen co-rating counts on MovieLens",
]
```

Additionally prohibited but enforced by human review rather than the deterministic validator:
"significant" / "significantly" / "p-value" as a claim; any confidence interval presented as an
inferential interval; "n = 30 independent trials"; "independent statistical replicates"; power
or MDE used as an inferential claim; any bootstrap or resampling inference; the qualitative
spread labels "larger" / "comparable" / "smaller"; "confirms" / "proves" / "detects" /
"establishes"; "Schwarz BIC"; quoting any number without naming its implementation lineage
(KI-002).

<!-- PROHIBITED-CLAIM-LIST:END -->

---

## 10. Artifacts

```
tools/research_audit/run_movielens_userdisjoint_validation.py
expfam/results/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_{summary,agg,paired,splitdiag,provenance,runinfo}.csv
figures/real_data/movielens_userdisjoint/movielens_userdisjoint_20260822_{delta_stability,condition_ll,decomposition}.png
```

Provenance hashes: `data_zip_sha256 = 50d2a982…`, `corpus_sha256 = 2ae70cb9…`,
`item_metadata_sha256 = cfc54a87…`. Everything is built from the tracked
`expfam/data/ml-100k.zip`; the untracked `expfam/data/movielens_pilot/*.npy` are never read.
No existing script, CSV, figure or `.npy` was modified.
