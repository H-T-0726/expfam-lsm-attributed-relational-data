# Phase 7c Held-out K-selection implementation plan

## Scope

This is a future-work plan only. It does not implement a runner, execute EM, generate
data/results, or authorize a pilot. The recommended Design A can be implemented without
changing any model class or `em_runner.py`. The pilot primary Y family and score
configuration must be frozen exactly once **BEFORE ANY FIT** in the implementation issue;
they are not selected after smoke or results. Cross-family joint scoring is prohibited.

## Phase 1 — split / preflight manifest

### Files likely reused

- `expfam/src/experimental/eval_utils.py::make_pair_split`, `upper_pairs_of`
- `expfam/src/experimental/diagnostics.py::validate_xy`
- SHA-256 array-manifest patterns in `tools/research_audit/run_k_selection_score_pilot.py`
- dataset/test/train-mask hash gates in `tools/research_audit/run_complementary_blocks_consistent.py`

### Files likely new

- Future runner: `tools/research_audit/run_heldout_k_selection_pilot.py`
- Prefer runner-local pure split validation initially; no model edit.

### Required tests

- Manifest contains candidate K exactly 1..7, two fixed starts, K-independent data/split
  seeds, and the exact expected `(replicate,K,start)` Cartesian key set.
- `make_pair_split(n,0.2,seed)` is deterministic.
- Train/test masks are symmetric, diagonal false, disjoint, and cover all off-diagonal dyads.
- Expected and actual upper-triangular test-pair counts are recorded and equal under the
  pre-fixed rounding rule for `test_ratio=0.20`.
- Generate splits for **all planned dataset replicates before any fit**.
- Apply a **PAIR-MASK TOPOLOGY ONLY** validator using only masks, node/pair indices, and
  unweighted train-mask topology: connected train graph, min train-mask degree >=2, min
  test-mask degree >=1.
- The guard must not inspect Y values, `Y==1`, positive counts, prevalence, weighted
  degrees, outcome summaries, or fit quality.
- One guard failure causes **PILOT GLOBAL STOP BEFORE ANY FIT**. No failed-replicate drop,
  replacement replicate, seed redraw, ratio change, K-specific redraw, or pair repair.

### Stop gate

STOP before any fit if the working branch/commit, runner hash, exact manifest key set, split
hashes, graph guards for any replicate, output paths, or no-overwrite checks fail.

## Phase 2 — pure helpers and static leakage assertions

### Files likely reused

- `expfam/src/experimental/model_dual_expfam_masked.py`
- `expfam/src/experimental/model_dual_expfam_consistent.py`
- `expfam/src/experimental/em_runner.py::run_em_experimental`
- `expfam/src/experimental/test_experimental_models.py::test_masked_ignores_heldout_pairs`
- provenance/fit-ledger patterns from `tools/research_audit/run_movielens_userdisjoint_validation.py`

### Files likely new or extended

- Pure helper tests, preferably `expfam/src/experimental/test_heldout_k_selection_helpers.py` or a narrowly scoped extension to `test_experimental_models.py`.
- A fit-input ledger inside the future runner.
- Pure raw-eta family score helpers and a pure pair-mask topology validator.

### Required tests

- Static assertions that `DualExpFamLSMConsistent` retains the supplied train mask.
- Freeze exactly one pilot Y family and immutable score config before any fit.
- Bernoulli reference: raw `eta`, `y*eta-logaddexp(0,eta)` via
  `bernoulli_log_likelihood`; no probability clipping/rounding/threshold/K-specific epsilon.
- Poisson reference: raw `eta`, `y*eta-exp(eta)-gammaln(y+1)`; retain the constant and do
  not use the `mu >= 1e-10` floor in `poisson_ll_pairs` as the production reference.
- Gaussian reference: raw `mu=eta`, full density, fit result's train-estimated `sigma_y`;
  do not use `predict_mu_y` clipping and never re-estimate variance from test Y.
- Use a finite family-support-valid `Y_canary_A/B`; NaN canaries are prohibited because
  `NaN * 0` is unsafe.
- Capture initialization either by (A) isolating a pure initialization helper or (B) a
  monkeypatch/call spy. Under A/B compare Z init, F init, `w0` init, `w` init, and
  applicable `sigma_y` init within a pre-fixed tolerance.
- Keep `Y_test_target` in a typed score-only wrapper outside fitting scope. At the fit call,
  assert object non-identity/non-flow and
  `fit_y_hash == expected_canary_payload_hash != raw_test_target_payload_hash`; hash
  inequality alone is not accepted as proof.
- X/preprocessing/node/feature, train/test mask, target, split-seed, family, and score-config
  hashes are identical across all K/starts within a replicate.
- Validate `expected (replicate,K,start) key set == actual/prepared key set`; `nunique==1`
  checks alone are insufficient.
- Negative-control/falsification checks demonstrate guards actually fire.

### Stop gate

Any target-to-fit dependency, initialization canary sensitivity, K-specific preprocessing,
mask mismatch, score reference mismatch, incomplete key set, or non-firing negative control
is BLOCKER/HIGH and stops implementation.

## Phase 2.5 — independent STATIC CODE REVIEW

Review the future runner, pure score/guard helpers, tests, immutable manifest, fit/target
boundary, failure policy, and output/no-overwrite paths without executing EM. Verify that
the model classes and `em_runner.py` are unchanged and that no test-derived object can enter
the fit call.

### Stop gate

Any BLOCKER/HIGH finding stops work. Phase 3A is prohibited until this static review passes.

## Phase 3A — complete-run two-canary falsification

This is the **first authorized EM action**. It runs only after Phase 2.5 passes and before
the normal smoke. Use the same X, training Y values, `train_mask`, preprocessing, K, and
seeds, changing only finite family-support-valid values at `train_mask=False` between
`Y_canary_A` and `Y_canary_B`.

### Required comparison

- captured initialization: Z, F, `w0`, `w`, and applicable `sigma_y`;
- complete-run outputs: Z estimate, F, `w0`, `w`, applicable `sigma_y`, `Q_strict`, and
  train-objective diagnostics;
- equality/invariance under one pre-fixed numerical tolerance, with identical keys/hashes
  for every allowed input.

### Stop gate

Any A/B difference is a **BLOCKING LEAKAGE FAILURE**. Do not run normal smoke. Masked
payloads must never be NaN.

## Phase 3B — normal small smoke harness

### Files likely reused

- `run_em_experimental` from `expfam/src/experimental/em_runner.py`; reconstruct raw
  `eta = w0 + w Z Z^T` for production scoring rather than using `predict_mu_y` generically
- consistent model lineage from `model_dual_expfam_consistent.py`
- raw-eta pure family score helpers reference-tested in Phase 2
- stdout retry detection and no-overwrite patterns from the Phase 7b runner

### Files likely new or extended

- The future pilot runner only.
- If Bernoulli Y is selected, a pure Bernoulli pair-score adapter may be added to `eval_utils.py`, with unit tests. This is evaluation support, not a model change.

### Required tests

- Reference formula checks for the selected family's raw-eta per-dyad score.
- Bernoulli uses `y*eta-logaddexp(0,eta)` without probability clipping or thresholding.
- Poisson uses `y*eta-exp(eta)-gammaln(y+1)`; omission of `-log(y!)` would be a
  K-invariant offset on the same target, but production absolute reporting retains it and
  does not use the existing mu-floor helper as its reference.
- Gaussian uses unclipped `mu=eta`, train-estimated `sigma_y`, and full normalization.
- Per-fit score equals the mean over exactly the fixed upper-triangular test indices.
- Independent selector recomputation uses the arithmetic mean of exactly two start scores,
  argmax sign, and the in-memory float64 condition
  `score_best - score_K <= 1e-12`, then smaller K. It is roundoff protection, not
  statistical indistinguishability, and is never applied to CSV-rounded values.
- Best-start selection, ensemble prediction, and treating starts as replicates are
  prohibited.
- Smoke manifest uses only a predeclared subset and separate output paths.

### Stop gate

Do not run normal smoke until Phase 3A passes. Once authorized, any internal retry, NaN,
non-finite score, Q failure, seed substitution, missing fit/start, or ledger mismatch causes
**PILOT GLOBAL STOP** before selection. Never select with one surviving start, drop a
replicate, or add a replacement seed.

## Phase 4 — independent code/result review

### Files reviewed

- The three Phase 7c design documents.
- Future runner and pure helper/test diffs.
- Exact manifest, output paths, score implementation, and leakage ledger.

### Required review

- Recompute mask and preprocessing hashes independently.
- Trace test target dataflow through initialization, E-step, M-step, preprocessing, prediction, and scoring.
- Verify the call-boundary evidence, initialization capture, and Phase 3A complete-run
  canary invariance rather than accepting hash inequality alone.
- Verify transductive language and reject inductive claims.
- Verify dyads are not used as independent replication units.
- Verify no cross-family joint score.
- Verify no model/runner modifications slipped into scope.

### Stop gate

Any BLOCKER/HIGH finding means **design/implementation not ready** and prohibits the pilot
until resolved and re-reviewed.

## Phase 5 — pilot run

### Preconditions

- Phases 1–4 passed.
- The exactly-one primary Y family and score configuration were frozen before the first fit
  and have not changed; dataset-replicate count, complete seed/key manifest, `L`,
  `num_iter`, smoke subset, and output paths are likewise frozen.
- Candidate K remains 1..7 and exactly two model starts are shared across K.
- User explicitly authorizes smoke and then pilot execution in separate steps.

### Outputs later, not in Phase 7c

- Per-fit primary scores and integrity/provenance fields.
- Per-replicate `K_hat_pred` and selection frequency.
- Descriptive comparison with Phase 7b C1/C2/C3 on the same fits only, keeping quantities separate.

### Stop gate

No seed rescue, candidate deletion, target filtering, post-hoc preprocessing, overwrite,
failed-replicate drop, or replacement replicate. Any failed required start/fit, missing key,
or integrity gate causes **PILOT GLOBAL STOP**; selection with one surviving start is
prohibited.

## Existing code reuse summary

| Need | Existing code | Reuse status |
|---|---|---|
| Pair split | `eval_utils.py::make_pair_split` | reuse; add pure graph guards outside model |
| Masked E/M | `model_dual_expfam_masked.py` | reuse unchanged |
| Consistent numerics | `model_dual_expfam_consistent.py` | reuse unchanged |
| EM orchestration/init | `em_runner.py::run_em_experimental` | reuse unchanged |
| Raw Y natural parameter | fit result `Z_est,w0,w` | reconstruct `eta=w0+wZZ^T`; do not use `predict_mu_y` as generic production-score source |
| Bernoulli scoring | `objective_consistent_numerics.py::bernoulli_log_likelihood` | pure raw-eta pair adapter; no probability clipping/threshold |
| Poisson scoring | `objective_consistent_numerics.py::poisson_log_likelihood` + `scipy.special.gammaln` | pure raw-eta full-score adapter; existing mu-floor helper is not production reference |
| Gaussian scoring | `eval_utils.py::gaussian_ll_pairs` formula + fit `sigma_y_est` | use raw unclipped `eta`; never estimate variance from test Y |
| Leakage tests | `test_experimental_models.py` | extend coverage later; do not run fits in Phase 7c |
| Hash/provenance gates | Phase 7b, complementary-blocks, user-disjoint runners | copy patterns, not results |

## Model-change determination

- Recommended Design A: **model change required = no**. `model_dual_expfam_masked.py`, `model_dual_expfam_consistent.py`, and `em_runner.py` remain unchanged.
- Design B: **NOT CURRENTLY SUPPORTED**. A model/API extension is required for frozen train
  parameters and new-node conditional Z inference; therefore it is excluded from Phase 7c.
  MovieLens user-disjoint validation is provenance precedent, not evidence of model-node
  new-node inference.

## Design-only self-review after independent-audit fixes

| Severity | Finding | Disposition |
|---|---|---|
| BLOCKER | None unresolved for Design A | 0 |
| HIGH | None unresolved for Design A | 0 |
| MEDIUM | Previous M1 family score API/source | resolved in the documents by raw-eta family references; implementation remains a mandatory pre-fit gate |
| MEDIUM | Previous M2 split/failure policy | resolved: topology-only, all splits preflighted, one failure = global stop |
| MEDIUM | Previous M3 runner-level canary | resolved: initialization capture plus Phase 3A complete-run A/B gate before smoke |
| MEDIUM | Previous M4 pre-smoke review order | resolved: Phase 2.5 static review, then Phase 3A, then Phase 3B |
| LOW | Current code/tests do not yet provide future canary evidence | expected implementation evidence; no fit authorized by this document |
| LOW | Hashes alone do not prove target non-flow | typed boundary/call spy is now mandatory |
| LOW | Score is plug-in, not uncertainty-integrated | disclosed; never call posterior predictive, marginal likelihood, or ELBO |
| Design-B blocker | no clean new-node Z inference/frozen-parameter API | Design B NOT CURRENTLY SUPPORTED and excluded from Phase 7c |

## Proposed Issue #39 decision

Exactly one proposal: **A: RUN_HELDOUT_K_SELECTION_PILOT**, meaning **after
implementation, static review, canary falsification, smoke, and independent review pass**.
No smoke or pilot is run in this design phase.

Falsifier: any canary A/B effect on fit outputs; need for outcome-aware guards; inability to
preserve identical split/target/preprocessing across K; family score disagreement with its
reference formula; need to drop a failed start/replicate; test-derived
feature/preprocessing entering fitting; need to change the model objective for Design A; or
any BLOCKER/HIGH in independent static review.
