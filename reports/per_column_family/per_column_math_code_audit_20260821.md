# Per-column Math / Code Audit

## Scope

This is an audit-only review of the experimental heterogeneous-X implementation at
commit `17609f891c048face3e736ec108171dbfe9787c9` (latest `main` when the audit
branch was created on 2026-08-21). The primary question is whether
`DualExpFamLSMPerColumn` is mathematically and programmatically consistent as an
**experimental** shared-Z model with Gaussian, Bernoulli, and Poisson X columns.

The audit used the actual implementation and canonical root documents as primary
evidence. Historical reports already present in `reports/per_column_family/` were
not used as primary evidence. No implementation, test, result, provenance, or CI
file was changed, and no research experiment was run.

**Overall mathematical verdict: PARTIALLY.** The implementation agrees with the
independently derived heterogeneous-family likelihood, score, curvature, F update,
and Gaussian variance update while every natural parameter remains in the
non-clipped interior. Uniform-family equivalence also holds independently for all
three families. It is not globally objective-consistent, however: outside the
Poisson clipping interval, and in the numerically floored Bernoulli tails, the
reported X objective is locally flat while the implemented score and precision
remain nonzero. The Poisson inconsistency is a HIGH finding and must be resolved or
explicitly bounded before a validation experiment is treated as evidence for the
prototype.

## Commit / lineage

- Base commit: `17609f891c048face3e736ec108171dbfe9787c9`
- Audit branch: `audit/23-per-column-math-code-audit`
- Audited class lineage:
  `DualExpFamLSMPerColumn` → `DualExpFamLSMMasked` →
  `DualExpFamLSMFixed` → `DualExpFamLSM` →
  `ExpFamLatentStructuralModel` → `LatentStructuralModel`.
- `model_dual_expfam_percolumn.py` overrides the X link/curvature helpers,
  E-step gradient and precision, F update, sigma update, and X log-likelihood.
- The Y side and pair mask come from the masked/fixed lineage.

This lineage must not be conflated with KI-001. The original printed Mikawa et al.
equations contain 1/2; the old Python lineage also contains 0.5; the adopted
independent derivation uses no extra 1/2; the fixed Python lineage removes it; and
the masked/per-column lineage inherits that fixed behavior. This audit does not
re-adjudicate the 1/2 question. The canonical warning that the old-0.5 Python
Newton direction cannot be asserted correct as a whole still applies to that old
lineage because only its Y term is halved; it is not a new conclusion about the
per-column X implementation.

## Model audited

For node `i` and X column `l`, with row loading vector `f_l`, the intended model is

```text
eta_il = f_l^T z_i
x_il | z_i ~ ExpFam_{c(l)}(eta_il, phi_l)
c(l) in {Gaussian, Bernoulli, Poisson}
```

Conditional on `z_i` and F, X columns factor independently. Gaussian columns have
column-specific dispersion `v_l`; Bernoulli and Poisson columns have dispersion 1.
The same Z is shared by every X column and by Y.

## Independent derivation

Write the X contribution for one node as

```text
q_X,i(z_i, F) = sum_l log p_c(l)(x_il | eta_il = f_l^T z_i).
```

For the exponential-dispersion form,

```text
log p(x | eta, phi) = [x eta - A(eta)] / phi + b(x, phi).
```

Therefore, before any numerical clipping,

```text
mu_l       = A'_l(eta_il)
score_z    = sum_l f_l (x_il - mu_l) / phi_l
precision_X = -d^2 q_X,i / dz_i dz_i^T
            = sum_l [A''_l(eta_il) / phi_l] f_l f_l^T
grad_F[l]  = sum_i (x_il - mu_il) z_i^T / phi_l.
```

In matrix form, with `r_i = x_i - mu_i`, column weights
`a_l = 1/v_l` for Gaussian and `a_l = 1` otherwise, and curvature weights
`c_l = A''_l(eta_il) a_l`,

```text
score_z     = F^T (a .* r_i)
precision_X = F^T diag(c) F.
```

`_calc_gradient` returns the gradient of the **negative** log conditional, so its
X component is the negative of `score_z`. `_calc_precision_matrix` returns the
positive precision contribution above.

## Family derivative table

| Family | Natural parameter | `A'(eta)` / mean | `A''(eta)` | dispersion | z score contribution | z precision contribution | F-row gradient | X log-likelihood |
|---|---|---|---|---|---|---|---|---|
| Gaussian | `eta=f_l^T z_i`, also the mean | `eta` | `1` | `v_l = sigma_l^2` | `f_l(x_il-eta)/v_l` | `f_l f_l^T/v_l` | `sum_i (x_il-eta_il)z_i^T/v_l` | `-0.5[(x_il-eta_il)^2/v_l + log(2 pi v_l)]` |
| Bernoulli | logit `eta` | `s(eta)` | `s(eta)(1-s(eta))` | `1` | `f_l(x_il-s)` | `s(1-s) f_l f_l^T` | `sum_i(x_il-s_il)z_i^T` | `x eta-log(1+exp eta)` |
| Poisson | log-rate `eta` | `exp(eta)` | `exp(eta)` | `1` | `f_l(x_il-exp eta)` | `exp(eta) f_l f_l^T` | `sum_i(x_il-exp eta_il)z_i^T` | `x eta-exp eta-log(x!)` |

The implementation-level Poisson X likelihood omits `-log(x!)`; the experimental
strict-Q path adds it only for the Poisson columns. The omission is constant in F
and Z and does not affect interior derivatives.

## Code-equation correspondence

| Item | Expected math / behavior | Actual code | Verdict |
|---|---|---|---|
| `family_x_list` | length d, each entry one supported family | Validated at `model_dual_expfam_percolumn.py:51-57`; copied at line 63 | VERIFIED |
| `_col_idx` | disjoint, exhaustive column partition | Constructed by exact family equality at lines 64-68 | VERIFIED |
| `_mean_function_x` | identity / sigmoid / exponential by column | Lines 80-90 dispatch correctly; Poisson uses clipped eta | VERIFIED in interior; CONTRADICTED globally by clipping |
| `_variance_function_x` | `1`, `s(1-s)`, `exp(eta)` | Lines 92-105 dispatch correctly but floor/clip tails | VERIFIED in interior; CONTRADICTED globally in tails |
| `_x_weight_vector` | `1/v_l` for Gaussian, 1 otherwise | Lines 107-114 use `1/diag(sigma)` for Gaussian columns | VERIFIED; `diag(sigma)` stores variance despite local name `sd` |
| `_calc_gradient` | negative prior + X score + inherited Y score, returned with negative sign | Lines 120-135 implement that decomposition and masked fixed-lineage Y | VERIFIED in interior; clipped-tail findings apply |
| `_calc_precision_matrix` | prior precision + X negative Hessian + inherited Y precision | Lines 137-151 implement weighted `F^T diag(c) F` and masked Y | VERIFIED in interior; clipped-tail findings apply |
| `calc_F` | Gaussian closed form; otherwise family-wise score update | Lines 157-160 use inherited closed form only when all Gaussian, otherwise weighted Adam | VERIFIED |
| `_calc_F_adam_weighted` | sum of `(x-mu)z^T/phi` by row | Lines 162-194 use the same per-column weight vector | VERIFIED in interior; Poisson/Bernoulli tail findings apply |
| `calc_sigma` | MLE variance only for Gaussian columns; non-Gaussian sentinel 1 | Lines 196-209 average squared residuals across nodes and MC samples | VERIFIED |
| `calc_log_likelihood_X` | sum column-family log-likelihoods | Lines 215-237 implement Gaussian full constant, Bernoulli, and Poisson without factorial | VERIFIED in interior; tail findings apply |
| inherited Y gradient | fixed, masked, no extra 1/2 | `model_dual_expfam_masked.py:93-116` | VERIFIED lineage; not a new KI-001 adjudication |
| inherited Y precision | fixed, masked, no extra 1/2 | `model_dual_expfam_masked.py:118-136` | VERIFIED lineage; not a new KI-001 adjudication |
| train/pair mask | symmetric mask, diagonal excluded, applied to every Y path | Mask validation at masked lines 56-68; E-step at 113/133; M-step at 157/187; likelihood at 248-249 | VERIFIED |
| mixed Q path | model likelihood plus Poisson-column factorial constants | `eval_utils.py:186-229` calls the model and corrects only `columns_of("poisson")` | VERIFIED |
| mixed information-criterion count | `f_params=kd-k(k-1)/2`, plus one dispersion per Gaussian X column, plus applicable Gaussian-Y dispersion | `eval_utils.py:232-256`; `em_runner.py:251-256` passes the Gaussian-column count | VERIFIED; inherited convention intentionally does not add `w0,w`, and remains a Q-based complete-data/ICL-type criterion, not Schwarz BIC (KI-010) |
| standard scalar Q/BIC exclusion | mixed path must not call scalar-only utilities | `em_runner.py:27,247-256` uses `calc_Q_dual_strict_exp` / `calc_bic_exp` | VERIFIED |
| scalar `family_x` fallback | list selects per-column model; absent list selects scalar masked model | `em_runner.py:31-50,123-125`; explicit list wins and scalar `family_x` is ignored | VERIFIED; no accidental mixed-to-scalar fallback found |

All-Gaussian per-column construction sets `family_x="gaussian"`. Uniform all-
Bernoulli or all-Poisson construction retains `family_x="mixed"`, but column
dispatch, strict-Q corrections, and parameter counts use `_col_idx`, so the marker
does not change the audited likelihood or update behavior.

## Gaussian dispersion semantics

**Stored quantity.** `LatentStructuralModel.initialize_params` initializes
`sigma=I`. Its parent `calc_sigma` computes

```text
diag((1/(L n)) sum_s (X-Z_s F^T)^T (X-Z_s F^T)),
```

at `reproduction/src/model.py:548-587`. Thus each diagonal element is the
Gaussian **variance** `v_l=sigma_l^2`, not a standard deviation. The per-column
mixed update independently computes the same residual mean square at
`model_dual_expfam_percolumn.py:196-209`.

- Gradient weighting: `1/diag(sigma) = 1/v_l = 1/sigma_l^2` — VERIFIED.
- Curvature weighting: the same `1/v_l`, not `1/v_l^2` — VERIFIED.
- Likelihood weighting: squared residual divided by `v_l`, with `-0.5 log(v_l)` — VERIFIED.
- Sigma update: stores residual variance; non-Gaussian diagonal entries remain 1 — VERIFIED.

The local variable name `sd` at per-column lines 112 and 224 is misleading, but
the operations use it as a variance and are mathematically correct. This differs
from Gaussian Y, where `self.sigma_y` stores a standard deviation and callers use
`self.sigma_y ** 2`; the two conventions must not be merged.

Independent mixed-sigma check: the Gaussian-column result was
`0.0899374939020963`, exactly equal to an independently calculated residual mean
square; the other two diagonal entries were `1.0`.

## Poisson clipping

The implementation defines `eta_c=clip(eta,-20,10)` in the mean, curvature, and
X log-likelihood paths. Inside `(-20,10)`, `d eta_c/d eta=1`, and likelihood,
score, and curvature agree with the canonical Poisson objective. Outside the
interval, `d eta_c/d eta=0`. The implemented likelihood

```text
x eta_c - exp(eta_c)
```

is therefore locally constant as a function of the original eta, so its exact
score and negative Hessian are both zero. The implemented gradient instead uses
`x-exp(eta_c)`, and the precision uses `exp(eta_c)`, without multiplying by the
clip derivative. The F Adam path uses the same nonzero residual.

At deterministic `eta=11.5`, `x=3`, `F=1`, finite differences of the actual
implemented likelihood gave score `0.0` and negative Hessian `0.0`; the
implemented paths returned score `-22023.465794806718` and precision
`22026.465794806718`. This is an exact objective/derivative contradiction, not a
rounding inference. See Finding PC-001.

## Uniform-family equivalence

An independent deterministic fixture used `n=5,d=4,k=2,L=2`, identical initial F,
a non-unit Gaussian variance diagonal `[0.25,0.7,1.6,2.3]`, and the same Y/mask.
It compared the per-column model to `DualExpFamLSMMasked` for every uniform family.
The reproducible construction used
`Z=[[0.2,-0.1],[0.5,0.3],[-0.4,0.7],[0.8,-0.6],[-0.3,-0.2]]`,
`F=[[0.6,-0.2],[-0.1,0.5],[0.3,0.4],[-0.7,0.1]]`, two identical MC
copies of Z, a full off-diagonal pair mask, and identical model parameters.
Gaussian X was `Z@F.T` plus the explicit row-wise noise
`[[.1,-.2,.05,0],[-.1,.1,0,.2],[.2,0,-.1,.1],[0,.05,.1,-.2],[-.05,-.1,.2,.05]]`;
Bernoulli X was `(Z@F.T>=0)`; Poisson X was
`[[1,0,2,1],[2,1,1,0],[0,2,1,1],[3,1,2,0],[1,0,1,2]]`.

| Uniform X family | gradient max abs | precision max abs | X-LL abs | F-update max abs | sigma max abs | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Gaussian | 0 | 0 | `7.11e-15` | 0 | 0 | VERIFIED |
| Bernoulli | 0 | 0 | 0 | 0 | 0 | VERIFIED |
| Poisson | 0 | 0 | 0 | 0 | 0 | VERIFIED |

Uniform equivalence does not prove global mathematical correctness: the scalar
and per-column Poisson/Bernoulli paths share the same tail safeguards, so they can
agree with each other while both disagree with the derivative of their reported
clipped/floored objectives.

## Finite-difference checks

The mixed fixture had one Gaussian, one Bernoulli, and one Poisson column,
`n=1,d=3,k=2`, `z=[0.35,-0.2]`,
`F=[[0.7,0],[0,1.25],[0.25,0.375]]`, `x=[0.4,1,2]`, Gaussian variance
`0.65`, and interior natural parameters `[0.245,-0.25,0.0125]`. Its reference
objective was written independently from the family definitions above, rather
than by copying model helper outputs.
Central differences used `h=1e-6` for the gradient and `h=2e-4` for the Hessian.

- X objective: implementation versus independent formula absolute difference `0.0`.
- z gradient: maximum absolute analytical/finite-difference difference `2.26e-10`.
- z curvature: maximum absolute analytical/finite-difference difference `7.51e-09`.
- Mixed Gaussian/Bernoulli/Poisson fixture: PASS in the non-clipped interior.
- Poisson upper-tail fixture (`eta=11.5`): objective derivative relation FAIL,
  with the exact values in the Poisson clipping section.
- Bernoulli floored-tail fixture (`eta=30,x=0`): actual likelihood finite-difference
  score `0.0`, implemented score approximately `-1.0`; actual negative Hessian
  `0.0`, implemented precision `1e-8`. See Finding PC-002.

## Existing tests

Commands run unchanged under repository Python 3.13.14:

```text
python expfam/src/experimental/test_percolumn_model.py
python expfam/src/experimental/test_experimental_models.py
python expfam/src/experimental/test_diagnostics_validation.py
```

Results:

- `test_percolumn_model.py`: all three tests PASS; mixed smoke RMSE(Z)=0.227,
  `w0=1.202`, `w=0.290`, reported criterion `8109.6`.
- `test_experimental_models.py`: all masked/fixed, mask, NB-limit, and small EM
  tests PASS.
- `test_diagnostics_validation.py`: all diagnostics/validation tests PASS.

Coverage assessment: the existing per-column uniform test compares gradient,
precision, and X likelihood, but uses identity X sigma for those comparisons and
does not directly compare F updates or uniform-family sigma behavior. It also
does not assert objective/derivative consistency outside clipping/floor regions.
The independent checks above fill those audit gaps without changing tests.

## Findings

### PC-001 — HIGH — Poisson clipped X objective is not the objective differentiated by E/M updates

1. **Severity:** HIGH.
2. **Title:** Poisson clipped X objective is not the objective differentiated by E/M updates.
3. **Exact file + lines:** `expfam/src/experimental/model_dual_expfam_percolumn.py:88-89,102-104,126-128,143-145,180-182,234-236`.
4. **Mathematical expectation:** if the implemented objective is
   `x*clip(eta)-exp(clip(eta))`, its score and negative Hessian with respect to
   the original eta are zero strictly outside the clip interval. Alternatively,
   an unclipped/smooth objective must have score and curvature equal to its own
   derivatives.
5. **Actual implementation:** likelihood clips eta, but E-step score, precision,
   and F Adam use the post-clip residual/curvature without the derivative of the
   clip.
6. **Reproducible trigger/check:** one Poisson column, `F=1,z=11.5,x=3`.
   Actual objective finite differences: score `0`, negative Hessian `0`.
   Implemented score `-22023.465794806718`, precision `22026.465794806718`.
7. **Consequence:** when clipping activates, Newton and Adam updates do not
   optimize the X objective reported in Q; convergence/Q interpretation and
   derivative-based inference are not mathematically coherent.
8. **Existing tests catch it:** NO. Diagnostics expose clip activation and even
   document the mismatch, but do not make the model path objective-consistent.
9. **Minimal follow-up direction:** in a separately approved implementation
   issue, select one explicit objective and make likelihood, score, curvature,
   and F update its derivatives; add upper/lower boundary finite-difference
   tests and monitor activation during, not only after, EM.

### PC-002 — MEDIUM — Bernoulli probability/curvature floors break exact tail derivatives

1. **Severity:** MEDIUM.
2. **Title:** Bernoulli probability and curvature floors break exact tail derivatives.
3. **Exact file + lines:** `expfam/src/experimental/model_dual_expfam_percolumn.py:86-87,99-101,126-128,180-182,229-233`.
4. **Mathematical expectation:** a stable Bernoulli objective such as
   `x*eta-logaddexp(0,eta)` has score `x-sigmoid(eta)` and curvature
   `sigmoid(eta)(1-sigmoid(eta))`; if the probability itself is hard-floored,
   the derivative of that floored objective is zero in the flat region.
5. **Actual implementation:** X likelihood floors probabilities to
   `[1e-10,1-1e-10]`; score uses the unfloored stable sigmoid; precision applies
   an independent `1e-8` minimum.
6. **Reproducible trigger/check:** one Bernoulli column, `F=1,z=30,x=0`.
   Actual floored-likelihood finite-difference score/negative Hessian were zero;
   implemented score was about `-1` and precision `1e-8`.
7. **Consequence:** in extreme logits, Q, score, and precision describe different
   numerical objectives. The threshold is more extreme than Poisson's upper clip,
   so the immediate risk is lower but the exact model claim is still false there.
8. **Existing tests catch it:** NO.
9. **Minimal follow-up direction:** use one numerically stable Bernoulli
   log-likelihood and its exact derivatives, or explicitly define/test a coherent
   surrogate objective and curvature policy.

### Notes

- **NOTE:** Gaussian X dispersion is implemented correctly, but local variable
  names `sd` obscure that the stored value is variance.
- **NOTE:** Uniform-family equivalence is exact for the audited paths, including
  non-unit Gaussian variances and F/sigma updates.
- **NOTE:** Mixed parameter counting correctly adds one parameter per Gaussian X
  column. The criterion remains the repository's Q-based complete-data/ICL-type
  criterion and must not be called Schwarz BIC.

Finding counts: **BLOCKER 0 / HIGH 1 / MEDIUM 1 / LOW 0 / NOTE 3**.

## VERIFIED

- Column-family validation and indexing are deterministic and exhaustive.
- Gaussian variance storage and all `1/v_l` weightings are consistent.
- Interior Gaussian, Bernoulli, and Poisson X likelihood/gradient/curvature agree
  with an independent derivation and finite differences.
- F and sigma updates have the expected family-wise form in the interior.
- Uniform-family equivalence holds for gradient, precision, X likelihood, F
  update, and sigma behavior for all three families.
- Train masks cover the inherited Y E-step, M-step, and likelihood paths.
- Mixed strict-Q and mixed parameter-count paths are selected by `em_runner`;
  scalar-only Q/BIC utilities are not used for mixed X.

## DERIVED

- The X score is `F^T(a .* (x-mu))`, with Gaussian `a_l=1/v_l`.
- The X precision is `F^T diag(A''/phi) F`.
- The per-row F score is the MC-averaged weighted residual outer product.
- A hard-clipped/floored objective has zero derivative in its flat exterior;
  nonzero residual/curvature updates there cannot be its exact derivatives.

## UNRESOLVED

- This audit did not measure clip activation in historical runs or infer it from
  archived results. Historical result provenance was intentionally untouched.
- No implementation remedy was selected; that requires separate human-approved
  scope and follow-up validation.
- Statistical usefulness of heterogeneous families is not established by this
  code/math audit.

## CONTRADICTED

- Global consistency of the Poisson clipped likelihood, gradient, and precision
  is contradicted outside `[-20,10]`.
- Global consistency of the Bernoulli floored likelihood and implemented
  score/curvature is contradicted in the extreme floored tails.
- A smoke-test PASS alone is not proof of objective consistency; current tests
  pass while the deterministic tail counterexamples remain.

## Suitability for next validation experiment

The prototype is **not yet suitable for a claim-bearing validation experiment**.
The non-clipped interior is coherent and the main mixed-family plumbing is
verified, so a subsequent experiment is technically plausible. Before treating
such an experiment as evidence, human review should decide how PC-001 is resolved
and require boundary finite-difference tests plus during-EM clip diagnostics.
PC-002 should be resolved or explicitly bounded in the same follow-up design.
No new experiment was started by this audit.

## Thesis-status statement

The standard thesis/main model continues to use one scalar `family_x` shared by
all X columns. `DualExpFamLSMPerColumn` remains an experimental prototype; the
findings and positive interior checks here do not establish empirical benefit or
formal-method status.

**本監査は per-column prototype を修論の正式提案手法へ昇格させるものではない。**
