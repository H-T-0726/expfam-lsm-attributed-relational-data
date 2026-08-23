# Phase 7c Held-out K-selection leakage matrix

## Scope and severity

Primary protocol: Design A, transductive Y-dyad holdout, with fixed `test_ratio=0.20`.
`Y_test_target` is score-only data and must not enter fitting, initialization,
preprocessing, feature creation, selection, tuning, retries, or Z inference. Production
held-out plug-in log scoring uses raw eta and the family formulas frozen before any fit; it
does not use `predict_mu_y` as a generic score source. All hashes are full SHA-256 over
shape, dtype, and contiguous bytes unless explicitly labeled as source-file hashes.

Severity: BLOCKER invalidates the pilot result; HIGH prevents the pilot from starting; MEDIUM requires documented resolution before smoke; LOW is an interpretation/auditability limitation.

## Machine-checkable leakage ledger

| Object | Allowed training input | Prohibited input | Enforcement point | Existing audit evidence | Planned assertion/hash | Severity if violated |
|---|---|---|---|---|---|---|
| Split manifest | `n`, fixed `test_ratio=0.20`, split seed | X values, any Y values, fit results, K | future pure preflight; all replicate splits generated before fitting | deterministic RNG split; `test_split_masks_partition` | expected/actual test-pair count; complete manifest hash; one failure causes PILOT GLOBAL STOP BEFORE ANY FIT | BLOCKER |
| Y train mask | mask complement of test over off-diagonal dyads | K-specific mask; result-dependent redraw | `DualExpFamLSMMasked.set_train_mask` | shape, symmetry, diagonal checks | symmetric; diagonal false; hash identical across K/starts | BLOCKER |
| Y test mask | fixed upper-pair target set | train overlap; K-specific filtering | future preflight + `eval_utils.py::upper_pairs_of` | `test_split_masks_partition` | `train & test == false`; union equals off-diagonal; target-index hash | BLOCKER |
| Graph structure guard | **PAIR-MASK TOPOLOGY ONLY**: train/test masks, node/pair indices, unweighted train-mask graph | Y values, `Y==1`, positive counts, prevalence, weighted degree, outcome summaries, fit quality | future pure split validator applied to every planned replicate before any fit | pair-mask report notes isolated-node risk | connected unweighted train-mask graph; min train-mask degree >=2; min test-mask degree >=1; no redraw/ratio change/pair repair/drop; any failure = global pre-fit stop | BLOCKER |
| Fit-time Y payload | Y_train values at train cells; finite family-support-valid test-independent canary at masked cells | actual `Y_test_target` object/array; NaN canary | typed score-only target boundary + fit-call spy + `train_mask` | masked perturbation unit test | `fit_y_hash == expected_canary_payload_hash`; `fit_y_hash != raw_test_target_payload_hash`; object-identity/call-graph assertion; two-canary invariance | BLOCKER |
| E-step Y score | train-mask Y only | test Y in gradient | `model_dual_expfam_masked.py::_calc_gradient` | `_mask_f` multiplication; perturbation test | canary-invariant Z/fit hash under fixed seed | BLOCKER |
| E-step curvature | train-mask dyads only | test dyads in precision | `model_dual_expfam_masked.py::_calc_precision_matrix` | `_mask_f` multiplication; static code audit | targeted precision comparison and complete-run canary invariance | BLOCKER |
| Relational M-step | train Y only | test Y in `w0`, `w`, `sigma_y` | `calc_w0`, `calc_w`, `calc_sigma_y` | perturbation tests for `w0/w`; code audit for Gaussian sigma | full fitted-parameter canary invariance, including Gaussian sigma | BLOCKER |
| Initialization Y statistics | upper-triangular `model.train_mask` values | test density/mean/std/counts | `em_runner.py::run_em_experimental` informed init | `obs_upper`/`y_obs` static inspection | before normal smoke, pure helper or monkeypatch/call spy captures Z/F/`w0`/`w`/applicable `sigma_y` init; A/B equal within pre-fixed tolerance | BLOCKER |
| Complete-run canary | identical X, train mask, training Y, seeds, K, preprocessing; only finite support-valid masked payload differs | any target-dependent change | Phase 3A, the first authorized EM action | no complete-run test yet | compare Z, F, `w0`, `w`, applicable `sigma_y`, `Q_strict`, and train-objective diagnostics within pre-fixed tolerance; any change blocks normal smoke | BLOCKER |
| Z inference source | all allowed X + train-mask Y | target Y or test-derived summaries | masked E-step through `run_em_experimental` | current call chain inspected | fit ledger states `Z_source=X_allowed+Y_train`; target hash absent from fit inputs | BLOCKER |
| X raw | exogenous X for all fixed nodes | X constructed from test Y | future harness input boundary | Design A conditioning set explicitly permits node X | raw X hash identical across K/starts | BLOCKER |
| X transformations | function of allowed X only; if relationally derived, train Y only | test Y, test exposure/counts | future preprocessing stage | user-disjoint runner uses typed train views and provenance | transformer config/hash and transformed-X hash shared across K | BLOCKER |
| Centering/scaling | all allowed transductive X, or train-Y-derived objects only where relevant | test Y statistics; K-specific fit | preprocessing before K loop | user-disjoint `build_train_attributes` precedent | location/scale arrays hashed; one object reused across K | BLOCKER |
| Node selection | pre-fixed synthetic nodes or exogenous rule; otherwise train Y only | test Y degree/popularity/outcomes | pre-split design or train-only constructor | user-disjoint `select_movies(EventView(tag=train),...)` | node-ID hash recorded and identical across K | BLOCKER |
| Feature selection | pre-registered features or train-only rule | test-score screening; test-Y-derived or K-specific choice | preprocessing before K loop | user-disjoint signature lint precedent | feature-name/order hash identical across K | BLOCKER |
| Derived X | exogenous metadata or train-Y-only construction | any test-Y target/statistic | train-only builder | user-disjoint `build_train_attributes` and independent recomputation | source hashes + output hash; negative-control lint | BLOCKER |
| Threshold | fixed before fitting or computed from train Y only | test label prevalence/score | preflight configuration | user-disjoint threshold provenance precedent | threshold source and numeric value in manifest | BLOCKER |
| Family transform | pre-specified by chosen family using allowed X/train Y only | test-driven family or transform choice | preflight configuration | `validate_xy` family support checks train-mask Y only | family/config hash shared across K | BLOCKER |
| Exposure/count summaries | train Y or exogenous exposure only | test Y counts/exposure | train-only preprocessing | user-disjoint train-event construction precedent | source tag/hash and recomputation check | BLOCKER |
| Candidate K / key set | exactly `{1,2,3,4,5,6,7}` and exactly two fixed starts per K/replicate | adaptive extension/truncation; missing/extra key | immutable manifest before any fit | Phase 7b range precedent | `expected (replicate,K,start) key set == actual key set`, in addition to uniqueness checks | BLOCKER |
| Data seed | fixed per dataset replicate, independent of K | K-indexed data generation | manifest built before K loop | Phase 7b k-independent data seed precedent | one dataset hash per replicate across K | BLOCKER |
| Split seed | fixed per dataset replicate, independent of K | seed redraw after graph/data/result inspection | manifest built before K loop | deterministic `make_pair_split` | one mask hash per replicate across K; no redraw counter | BLOCKER |
| Model seed / start failure | exactly two pre-fixed starts, same seed labels across K | failed-fit seed rescue; extra/best start; use of one surviving start; replicate drop | manifest + fit loop | Phase 7b explicit model seed manifest precedent | no substitutions; any required start failure = PILOT GLOBAL STOP | BLOCKER |
| Preprocessing hashes | fixed objects from allowed sources | test-derived or K-specific objects | fit-time ledger | MovieLens user-disjoint `x_input_hash == expected_x_provenance_hash` | full SHA-256 ledger; unique=1 across K within replicate | BLOCKER |
| Score target | actual Y only at fixed test-mask upper pairs | train pairs; filtered targets; tuning on test score | score-only stage after fit | `upper_pairs_of`; held-out metrics pattern | target index/value hash fixed across K; count exact | BLOCKER |
| Bernoulli score | raw `eta`; `y*eta-logaddexp(0,eta)` | probability clipping/rounding, thresholding, test prevalence, K-specific epsilon | pure pair scorer | `bernoulli_log_likelihood(y,eta)` | exact reference arrays and fixed test-index hash | BLOCKER |
| Poisson score | raw `eta`; `y*eta-exp(eta)-gammaln(y+1)` | `mu` floor, omission of constant in reported absolute score, K-specific clipping | pure pair scorer | objective helper supplies no-constant eta form; existing `poisson_ll_pairs` floor is not production reference | exact reference arrays; full constant retained; fixed test-index hash | BLOCKER |
| Gaussian score | raw `mu=eta`; full density with fit result's train-estimated `sigma_y` | `predict_mu_y` clipping; test-estimated variance | pure pair scorer | `gaussian_ll_pairs`; masked `calc_sigma_y` | raw-eta reference comparison; fit `sigma_y` identity; no target input to variance | BLOCKER |
| Score-family/config freeze | exactly one primary Y family and immutable score configuration before any fit | family switching after smoke/results; family mixing; cross-family joint score | immutable pre-fit manifest | no current general K selector | family/config hash identical across all K/starts | BLOCKER |
| Score tuning | none | choosing epsilon, clipping, threshold, family, tie tolerance after scores | immutable design constants | no current general K selector | config hash written before fit | BLOCKER |
| Retry/failure handling | all required fits pass under the pre-fixed runner policy | seed rescue, fit/start/replicate deletion, K-specific retry | future hard gates | Phase 7b internal-retry detection precedent | any internal retry/NaN/Q failure or missing required fit = PILOT GLOBAL STOP before selector | BLOCKER |
| Selector | arithmetic mean of exactly two start scores, maximize, in-memory float64 `score_best-score_K <= 1e-12`, smaller K | best-start selection, ensemble prediction, one-start fallback, CSV-rounded tie, statistical-equivalence interpretation | post-fit pure aggregation | none yet | independently recompute from complete per-fit rows and exact key set | HIGH |

## Test-Y flow prohibition

```text
Y_test_target ──X──> E-step
              ──X──> M-step
              ──X──> initialization
              ──X──> preprocessing / node or feature construction
              ──X──> threshold / family / K / seed / retry choice
              ──✓──> final score on fixed test_mask only
```

The actual test target wrapper/array must remain outside the fit-call object graph. Masked
cells in the fit payload use a finite, family-support-valid, test-independent canary and
remain unobserved because `train_mask=False`; NaN is prohibited because `NaN * 0` is not
safe. Hash inequality alone is insufficient. A typed boundary or call spy must establish
that the target object was never passed to fitting, while hashes establish that the actual
fit Y equals the expected canary payload. The implementation review must prove
initialization and complete-run canary invariance rather than trusting zero-filled storage
conventions.

## K-comparability invariants

Before execution, the manifest contains the exact expected Cartesian key set
`(replicate, K in {1..7}, start in {1,2})`. After execution,
`expected key set == actual key set` is mandatory; uniqueness alone is insufficient.
Within dataset replicate `r`, all candidate K and both model starts must have identical
hashes for node IDs, raw X, transformed X, preprocessing, train mask, test mask, train Y
observed values, test target indices/values, score configuration, family configuration, and
split seed. Only K, dimension-shaped initialization, the pre-fixed model-start seed, fitted
values, runtime, and diagnostics may differ.

## Frozen protocol summary

- Primary design: Design A, transductive dyad holdout.
- Estimand: held-out plug-in Y mean log score conditional on observed X for all fixed nodes
  and training Y context.
- Test ratio: `0.20`, with no guard-driven modification.
- K: `{1,2,3,4,5,6,7}`.
- Starts: exactly two fixed algorithmic starts; arithmetic mean of their scores, never best
  start or ensemble prediction.
- Tie: in-memory float64 `score_best - score_K <= 1e-12`, then smallest K; numerical
  protection only.
- Replication unit: independently generated dataset replicate; dyads are only the
  within-replicate risk average and starts are not replicates.
- Failure: any split-guard failure before fitting, or any required fit/start failure after
  fitting begins, causes PILOT GLOBAL STOP; no redraw, repair, rescue, or drop.
- Canary order: initialization/static gates, independent static code review, complete-run
  two-canary falsification, normal smoke, independent review, pilot.
- Design B: NOT CURRENTLY SUPPORTED.
- Proposed Issue #39 decision: `A: RUN_HELDOUT_K_SELECTION_PILOT`, meaning only after
  implementation, static review, canary falsification, smoke, and independent review pass.

## Independent-audit disposition after document fix

- BLOCKER: 0 unresolved for recommended Design A.
- HIGH: 0 unresolved for recommended Design A.
- MEDIUM: 0 unresolved at the document-design level. The four audit findings—family score
  source, split/global failure policy, runner-level canary evidence, and pre-smoke review
  order—are now explicit mandatory implementation gates.
- LOW: 3 retained limitations: present code/tests do not yet provide the future canary
  evidence; target non-flow needs a typed/call-spy assertion in addition to hashes; and the
  held-out score integrates neither latent nor parameter uncertainty.

Any test Y influence on initialization, Z/E-step, precision, M-step, dispersion, or
preprocessing is BLOCKING. Outcome-aware redraw, K-specific split/target filtering, failed
start seed rescue, or silent failed-replicate drop is likewise BLOCKING.

Design B is **NOT CURRENTLY SUPPORTED**. Its missing frozen-parameter new-node inference
API is a blocker for Design B, not an unresolved path in the rejected Design B protocol.
