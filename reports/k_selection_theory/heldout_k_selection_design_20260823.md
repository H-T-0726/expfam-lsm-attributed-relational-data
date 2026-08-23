# Phase 7c Held-out K-selection design

## 1. Scope

This document freezes a design-only proposal for Issue #39. No EM fit, K sweep, synthetic-data fit, CSV generation, or figure generation is part of Phase 7c.

The proposed pilot asks a **transductive** question: for a fixed node set, how well does each candidate K predict Y dyads that were excluded from fitting, conditional on the same allowed X and the remaining Y dyads? It does not answer new-node induction, X prediction, joint X/Y prediction, or consistency for recovering `K_TRUE`.

The intended lineage is the objective-consistent experimental/prototype implementation in `expfam/src/experimental/model_dual_expfam_consistent.py`. It is not manuscript-approved.

## 2. Operational estimand

For independent dataset replicate `r`, candidate K, and pre-fixed model start `m`, let `T_r` be the upper-triangular held-out Y dyads and let `theta_hat_{rKm}, Z_hat_{rKm}` be fitted using only `X_allowed,r` and `Y_train,r` under `train_mask_r`.

The primary per-fit quantity is the **held-out plug-in Y mean log score**
(`held-out plug-in log score` for short):

```text
R_rKm = (1 / |T_r|) sum_(i,j in T_r)
         log p_family(Y_ij | Z_hat_rKm, theta_hat_rKm)
```

Here `Z_hat_rKm` and `theta_hat_rKm` are plug-in quantities fitted from
`X_allowed,r + Y_train,r` only. The conditional information set is the same for every K.
The score integrates neither parameter uncertainty nor Z uncertainty. It is not a posterior
predictive score, integrated predictive distribution, marginal likelihood, ELBO, BIC, or
C1/C2/C3.

The two pre-fixed model starts are averaged within `(r,K)` before K selection. Dyads are averaged to define prediction risk; they are not treated as independent replication units.

## 3. Design A

### Transductive dyad holdout

- Nodes: identical in train and test.
- Split: `expfam/src/experimental/eval_utils.py::make_pair_split(n, test_ratio=0.20, seed=split_seed_r)` partitions all off-diagonal dyads into symmetric `train_mask_r` and `test_mask_r`. The ratio is fixed before any fit; expected and actual upper-triangular test-pair counts are recorded and asserted.
- Split guard: **PAIR-MASK TOPOLOGY ONLY**. It may inspect only `train_mask`, `test_mask`, node/pair indices, and the unweighted train-mask graph topology. It asserts symmetric masks, diagonal false, no overlap, complete off-diagonal union, the fixed expected pair count up to the pre-specified rounding rule, connected train graph, minimum train-mask degree at least 2, and minimum test-mask degree at least 1. It must not inspect Y values, `Y == 1`, positive counts, Y prevalence, weighted degree, outcome summaries, or model fit quality.
- Global split gate: all planned dataset-replicate splits are generated and guarded before any fit. One failure causes **PILOT GLOBAL STOP BEFORE ANY FIT**. There is no failed-replicate drop, replacement replicate, seed redraw, ratio change, K-specific redraw, or pair addition/deletion repair.
- X/estimand: all fixed nodes' observed exogenous/raw X is allowed because the primary
  estimand is **Y prediction conditional on observed X for all fixed nodes and training Y
  context**. Any X feature derived from relational events, exposure, counts, thresholds, or
  Y must use `Y_train` only. This is not X holdout, new-node prediction, joint X/Y
  prediction, or inductive generalization.
- Z inference: all node Z values are inferred jointly from `X_allowed + Y_train` only.
- Prediction/scoring: production scoring reconstructs raw `eta_ij = w0 + w z_i^T z_j` from the fit result and uses the family-specific formulas in Section 7. `predict_mu_y` is not the common production score source. Scoring uses only upper-triangular `test_mask` pairs.
- K comparability: the node set, X, preprocessing objects, train/test masks, target Y values, and scoring function are identical across K.

### Current enforcement evidence

`expfam/src/experimental/model_dual_expfam_masked.py::DualExpFamLSMMasked` applies `train_mask` in:

- `set_train_mask`: shape/symmetry validation and diagonal exclusion;
- `_calc_gradient`: excludes test Y from the E-step score;
- `_calc_precision_matrix`: excludes test dyads from E-step curvature;
- `calc_w0` and `calc_w`: excludes test Y from relational M-step updates;
- `calc_sigma_y`: uses observed upper-triangular train pairs only;
- `calc_log_likelihood_Y`: uses masked pairs only.

`expfam/src/experimental/em_runner.py::run_em_experimental` constructs the masked/consistent model through `build_model`. Its informed Y initialization defines `obs_upper = triu(model.train_mask,1)` and computes `y_obs = Y[obs_upper]`; therefore Bernoulli density, Poisson positive-count mean, and Gaussian mean/standard deviation initialization exclude test dyads. `calc_F` and `calc_sigma` use X, which is allowed by this design.

`expfam/src/experimental/model_dual_expfam_consistent.py::DualExpFamLSMConsistent` inherits the masked E/M mechanics and overrides only objective-consistent family numerics. `expfam/src/experimental/test_experimental_models.py::test_masked_ignores_heldout_pairs` already verifies invariance of gradient, `calc_w0`, `calc_w`, and Y likelihood under held-out-Y perturbation. The implementation phase must extend this evidence to runner initialization and complete-fit canary invariance before any pilot.

Test targets must not be passed as observational inputs. A future harness keeps
`Y_test_target` in a score-only wrapper outside the fitting API scope and passes a finite,
family-support-valid, test-independent canary payload at masked cells together with
`train_mask`; the mask, not the payload value, defines observation. Masked payloads must
not be NaN because `NaN * 0` is not numerically safe. A call spy or typed boundary plus
object identity/content hashes must verify that the target object never enters the fit graph.
Two different canary payloads must give invariant initialization and complete-run fit outputs
under the pre-fixed tolerance before normal smoke. This is not zero-imputation as an
observed zero.

### Mandatory two-canary gate

Before normal smoke, use identical X, training Y values, `train_mask`, K, preprocessing,
and seeds, changing only finite family-support-valid values at `train_mask=False` between
`Y_canary_A` and `Y_canary_B`. Initialization capture must compare Z init, F init, `w0`
init, `w` init, and Gaussian `sigma_y` init. Because the runner currently returns no
initialization snapshot, the future implementation must either (A) isolate a pure
initialization helper or (B) capture initialization with a monkeypatch/call spy, without
changing the model's scientific objective. The complete-run comparison must include Z
estimate, F, `w0`, `w`, applicable `sigma_y`, `Q_strict`, and train-objective diagnostics,
using a pre-fixed numerical tolerance. Any difference is a **BLOCKING LEAKAGE FAILURE**;
normal smoke must not start.

## 4. Design B

### Node-disjoint / inductive holdout

The scientifically clean version would:

1. learn `F,w0,w` and dispersion from train nodes only;
2. freeze those parameters;
3. infer each new test-node `z_i` from allowed test-node X and, only if pre-specified, non-target Y context;
4. score target test dyads that were never used to infer either endpoint Z.

The current API does not implement this protocol:

- `reproduction/src/model.py::LatentStructuralModel.__init__` fixes `n`, and `initialize_params` creates a joint `(n,k)` Z matrix.
- `expfam/src/experimental/em_runner.py::run_em_experimental` jointly initializes and updates Z, F, dispersion, `w0`, and `w`; it has no freeze-trained-parameters/new-node-transform mode.
- No `infer_new_node`, `transform`, or equivalent new-node Z API exists under `expfam/src/experimental/`.
- Including test nodes with their X in a joint fit would allow test X to update F and would be partially transductive, not the train-parameters-frozen inductive estimand above.
- Using test Y context would require a target/context mask and a frozen-parameter conditional inference API; neither exists.

`tools/research_audit/run_movielens_userdisjoint_validation.py` is valuable leakage-ledger precedent, but it is not new-model-node inference: users are split to construct separate relation matrices while the model nodes remain the selected movies. It therefore does not make Design B available.

Conclusion: **Design B = NOT CURRENTLY SUPPORTED** and must not enter the Phase 7c
primary pilot. A future Design B requires explicit model/runner support and separate review.

## 5. A vs B comparison

| Criterion | Design A: transductive dyads | Design B: node-disjoint |
|---|---|---|
| Scientific question | Predict missing Y among known nodes | Predict relations involving new nodes |
| Z for scored endpoints | Learned from allowed X + other train Y | Must be newly inferred after train-parameter fit |
| Current mask enforcement | Yes | No |
| Test target excluded from E/M/init | Enforceable and partly tested | No supported two-stage API |
| Same split across K | Directly enforceable by hashes | Conceptually possible, not currently executable cleanly |
| Model change needed | No | Yes |
| Primary recommendation | Yes | No |

## 6. Recommended primary design

**Design A — transductive dyad holdout** is recommended.

The reason is not convenience. Its estimand is explicit; current `train_mask` mechanics reach E-step, relational M-step, Gaussian-Y dispersion, strict Q diagnostics, and Y initialization; the held-out target can be kept outside Z inference; and one split/preprocessing manifest can be shared identically across K. These properties are directly grounded in the functions listed in Section 3.

## 7. Primary predictive target

Exactly one primary target is proposed: **family-correct held-out plug-in Y mean log
score per held-out upper-triangular dyad**, maximized within one pre-specified Y family.
The pilot primary Y family and the complete score configuration are frozen **BEFORE ANY
FIT**, not after smoke and not after viewing results. Phase 7c defines the score for every
supported family; the implementation issue must select exactly one primary family before
its first fit. Scores from different families are not pooled or compared as a joint
cross-family criterion, and the family is never switched in response to results.

Production scoring uses raw
`eta_ij = w0 + w z_i^T z_j`; it does not use `predict_mu_y` as a family-generic score
source.

| Y family | Production per-dyad held-out plug-in log score | Planned stable source | Constant / clipping policy |
|---|---|---|---|
| Bernoulli | `y*eta - log(1 + exp(eta))`, evaluated as `y*eta - logaddexp(0,eta)` | `objective_consistent_numerics.py::bernoulli_log_likelihood(y, eta)` on fixed test pairs | No probability clipping, rounding, thresholding, test-prevalence threshold, or K-specific epsilon |
| Poisson | `y*eta - exp(eta) - gammaln(y+1)` | raw-eta scorer using `objective_consistent_numerics.py::poisson_log_likelihood(y, eta)` plus `-gammaln(y+1)` | `-log(y!)` is K-invariant on the same target but is retained for the absolute production log score. `eval_utils.py::poisson_ll_pairs` has a `mu >= 1e-10` floor and is not the production-selector reference implementation |
| Gaussian | `-0.5 * [(y-eta)^2/sigma_y^2 + log(2*pi*sigma_y^2)]`, with `mu=eta` | raw `eta` plus the fit result's train-mask-estimated `sigma_y` | Do not use `predict_mu_y`'s `[0,1e5]` clipping; do not re-estimate variance from test Y; retain the full density normalization |

The future implementation needs pure pair-score adapters or runner-local stable scorers;
this is evaluation support, not a model-objective change. `eval_utils.py::heldout_count_metrics`
must not be reused unchanged for Bernoulli because its dispatcher omits that family and its
additional dispersion metrics are count-oriented.

## 8. Z inference rule

```text
Z inference input = X_allowed for all fixed nodes + Y_train under train_mask
Z inference prohibited input = Y_test_target and every statistic derived from it
```

For a held-out target `Y_ij`, neither `Y_ij` nor any test-outcome aggregate, threshold,
feature, exposure summary, preprocessing choice, or tuning decision may affect `z_i`,
`z_j`, F, `w0`, `w`, dispersion, initialization, retry/seed choice, or fit inclusion. All
test dyads are excluded jointly by one common mask; this is not pairwise leave-one-out that
uses the other test dyads as context. Test Y enters only after fitting, in the final score
calculation.

## 9. K selector and tie rule

- Candidate range: `K_CANDIDATES = {1,2,3,4,5,6,7}`.
- Sign: larger held-out mean log score is better.
- Manifest: the exact expected set of every `(dataset replicate, K, start)` key is frozen before execution. Post-run validation requires `expected key set == actual key set`, not only per-column `nunique == 1` checks.
- Starts: exactly two pre-fixed model seeds per `(dataset replicate, K)`; the same two integer seed labels are reused across all K. Their two held-out plug-in log scores are averaged before selection. This is not best-start selection, ensemble prediction, or statistical replication.
- Selector for dataset replicate `r`:

```text
Sbar_r(K) = (R_rK,start1 + R_rK,start2) / 2
K_hat_pred,r = smallest K among {K: Sbar_r(K) >= max_K' Sbar_r(K') - 1e-12}
```

- Tie tolerance: on the in-memory float64 mean log scores, K is a tie candidate when
  `score_best - score_K <= 1e-12`; choose the smallest candidate K. The rule is exact /
  roundoff protection only, is not applied to CSV-rounded values, and does not mean
  “statistically indistinguishable.”
- Data seed and split seed are independent of K and recorded in a pre-fit manifest.
- Within each replicate, raw/transformed X hash, preprocessing hash, train/test-mask hashes,
  target hash, score-config hash, and split seed must be identical across every K/start.
- No failed-fit seed rescue, candidate drop, replicate drop, or K-specific filtering. If
  either required start fails for any K, the remaining start is not used alone and the
  result is **PILOT GLOBAL STOP**.

## 10. Replication/statistical unit

For the synthetic pilot, the **independently generated dataset replicate** is the formal
replication unit. One pre-fixed dyad split is used per dataset replicate; the two algorithmic
model starts are averaged inside it. Per-replicate `K_hat_pred,r` and score contrasts may be
summarized across dataset replicates.

Held-out dyads define the within-replicate risk average but are dependent and are not
independent sample units. Their count is not a statistical independent sample size.
Repeated splits of one fixed graph may be reported only as split-stability diagnostics and
must not be labeled independent replicates.

## 11. Predictive-optimal K vs K_TRUE

`K_hat_pred` optimizes finite-sample plug-in prediction under a specific mask, family, fitting algorithm, and candidate range. It may differ from generative `K_TRUE` under finite samples, regularization/approximation, weakly identified dimensions, or model misspecification. Agreement with `K_TRUE` in one scenario does not establish consistency; disagreement does not by itself prove model failure.

## 12. Theory limitations

- The scored distribution is plug-in, not integrated over parameter or latent-variable uncertainty.
- Dyads sharing nodes are dependent.
- Random dyad masking addresses a transductive MCAR-style target, not MNAR observation processes.
- Candidate-K models have rotation/non-identifiability issues and need not be nested in the operational algorithm.
- Approximate MCEM and finite iterations can affect predictive ranking.
- Properness of the family log score concerns the predicted Y distribution, not recovery of literal latent dimension.

## 13. Claim boundary

Allowed: leakage-audited transductive held-out Y comparison; predictive K under the frozen protocol; descriptive comparison with Phase 7b C1/C2/C3 after a future run.

Prohibited: inductive/new-node generalization; K-selection consistency; `K_hat_pred = K_TRUE` as a theorem; independent-dyad inference; correct BIC/corrected BIC/ELBO/marginal-likelihood claims; cross-family joint score; paper Experiment 2 reproduction; manuscript approval of the prototype.

## 14. Independent-audit disposition and mandatory implementation gates

Independent design audit verdict: **DESIGN_APPROVED_AFTER_FIX**. The four MEDIUM findings
were (1) family score API/source, (2) split/failure policy, (3) runner-level canary evidence,
and (4) pre-smoke review ordering. This revision resolves their design-document ambiguity.
They remain mandatory implementation gates, not optional follow-up work:

1. implement/reference-test raw-eta family scorers without changing the model objective;
2. implement the topology-only split validator and global pre-fit stop;
3. pass initialization capture and complete-run two-canary invariance before normal smoke;
4. pass independent static code review before the first authorized EM action.

There is no remaining Design-A BLOCKER/HIGH/MEDIUM at the document-design level. The
known LOW limitations are that current tests do not yet supply the required future evidence,
the target/fit boundary must be machine-enforced rather than inferred from hashes alone, and
the score is plug-in rather than uncertainty-integrated. Design B has a design-specific
blocker—no frozen-parameter new-node Z inference API—and remains non-primary and
**NOT CURRENTLY SUPPORTED**.

## 15. Proposed decision gate

**A: RUN_HELDOUT_K_SELECTION_PILOT**, meaning only: **after implementation, static
review, canary falsification, smoke, and independent review pass**. This decision is not
authorization to run an EM smoke or pilot during Phase 7c.

Falsifier: withdraw decision A if canary A/B changes any fit-side output; if an
outcome-aware guard is needed; if the same split, target, and preprocessing cannot be held
fixed across K; if a family score differs from its reference formula; if failed starts or
replicates must be dropped; if test-derived features/preprocessing reach the fit; if Design A
requires a model-objective change; or if independent static review finds a BLOCKER/HIGH.

Design acceptance checklist:

- [x] primary target exactly one
- [x] transductive / inductive not conflated
- [x] Z inference source fixed
- [x] test Y target excluded from Z inference by design
- [x] initialization mask path confirmed in current code
- [x] preprocessing policy fixed
- [x] same split across K fixed
- [x] data seed independent of K
- [x] score sign fixed
- [x] tie rule fixed
- [x] replication unit fixed
- [x] `K_hat_pred` distinguished from `K_TRUE`
- [x] current Design A API enforcement confirmed statically
- [x] no experiment executed
