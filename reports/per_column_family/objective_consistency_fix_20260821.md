# Objective Consistency Fix

## Scope

Issue #25 adds an explicitly selected, forward-only numerical lineage for future
experimental work. Its purpose is to make the Bernoulli and Poisson objective,
score, and curvature describe one canonical mathematical model. It does not
rewrite the legacy implementation or reinterpret historical results.

- Repository base: `ded896752cb3333bdc1aabbb63fdd78dda50411d`
- Branch: `experiment/25-objective-consistent-numerics`
- Claim-bearing validation experiment: **NOT RUN**
- Historical results: **NOT CHANGED**
- Prototype promoted to thesis method: **NO**

## Base / head

The branch was created directly from the actual `origin/main` at
`ded896752cb3333bdc1aabbb63fdd78dda50411d`. The final implementation head is the
commit containing this report and is recorded in the Draft PR metadata; this
avoids embedding a self-referential commit hash in its own commit contents.

The two post-PR-#24 cleanup commits were also checked: both
`git diff --stat d8ddc29f81b96b824fa6010fcbcdf3a24695468c..origin/main` and the
full diff returned no tree changes. History was not rewritten.

## Historical lineages preserved

These historical/core files have zero diff from the base:

- `expfam/src/model_expfam.py`
- `expfam/src/model_dual_expfam.py`
- `expfam/src/model_dual_expfam_fixed.py`
- `reproduction/**`
- `expfam/src/experimental/model_dual_expfam_masked.py`
- `expfam/src/experimental/model_dual_expfam_percolumn.py`
- `expfam/src/experimental/model_dual_expfam_nb.py`

Consequently, old 0.5, fixed, masked, legacy per-column, and NB behavior remain
identifiable at their original class names. Existing scripts that omit
`numerics_mode` still select the same legacy classes and numerics.

## New forward lineage

New code is isolated in:

- `objective_consistent_numerics.py`: stable canonical family helpers;
- `model_dual_expfam_consistent.py`: consistent masked/scalar and per-column
  classes;
- `em_runner.py`: explicit `numerics_mode="consistent"` selection.

No surrogate or capped statistical model was introduced.

## Inheritance

```text
legacy fixed/masked lineage (unchanged)
├─ DualExpFamLSMMasked
│  └─ DualExpFamLSMConsistent                 [new scalar forward class]
└─ DualExpFamLSMPerColumn
   └─ DualExpFamLSMPerColumnConsistent        [new per-column forward class]
```

Technically, a private Y-numerics mixin precedes each legacy parent in the MRO.
It overrides only the Bernoulli/Poisson Y mean, curvature, and likelihood. Each
new class overrides the corresponding X hooks. Fixed no-extra-1/2 mathematics,
pair masks, Gaussian dispersion handling, EM updates, and strict-Q conventions
remain inherited.

## Bernoulli objective

The consistent lineage evaluates

```text
log p(x|eta) = x*eta - logaddexp(0, eta)
score         = x - sigmoid(eta)
curvature     = sigmoid(eta)*(1-sigmoid(eta))
```

The sigmoid uses a sign-partitioned stable evaluation, so the unsafe exponential
branch is not evaluated. There is no probability floor in the objective and no
independent curvature floor.

## Poisson objective

The consistent lineage evaluates the canonical log-link model

```text
mu             = exp(eta)
log p(x|eta)   = x*eta - exp(eta)       # -log(x!) added by strict-Q
score          = x - exp(eta)
curvature      = exp(eta)
```

There is no `clip(eta,-20,10)` in the new helpers/classes. In particular,
`eta=11.5` is evaluated directly in float64.

## Overflow policy

Poisson eta must be finite. The overflow boundary is derived at runtime as
`log(np.finfo(dtype).max)`. Eta above that boundary, non-finite eta, a failed
`exp`, or a non-finite likelihood result raises a clear `FloatingPointError`.
Underflow at a finite negative eta is allowed to follow the runtime floating
dtype; it is not replaced by an arbitrary lower scientific cutoff.

The runner re-raises a consistent-mode `FloatingPointError`, including one from
strict-Q, instead of returning a misleading NaN Q/BIC result. Legacy Q failure
handling is unchanged.

## Objective-score-curvature correspondence

| Path | Objective | Score/mean hook | Curvature hook | Result |
|---|---|---|---|---|
| scalar X Bernoulli | stable canonical | same sigmoid | exact `s(1-s)` | PASS |
| scalar X Poisson | canonical, no clip | same `exp(eta)` | same `exp(eta)` | PASS |
| Y Bernoulli | masked stable canonical | same sigmoid | exact `s(1-s)` | PASS |
| Y Poisson | masked canonical, no clip | same `exp(eta)` | same `exp(eta)` | PASS |
| per-column Bernoulli | stable canonical by selected columns | same helper | same helper | PASS |
| per-column Poisson | canonical by selected columns | same helper | same helper | PASS |
| Gaussian X/Y | inherited unchanged | inherited unchanged | inherited unchanged | exact regression PASS |

## Issue #23 before / after

### PC-001 Poisson, `eta=11.5`, `x=3`

- Legacy control: finite-difference objective score `0`; implemented score
  `-22023.465794806718`; implemented precision `22026.465794806718`. The Issue
  #23 contradiction remains reproducible in the protected legacy class.
- Consistent: canonical score `-98712.7710107605`, curvature
  `98715.7710107605`. Central-difference absolute errors were `1.75e-06` for
  score and `8.23e-03` for curvature (`1.77e-11` and `8.34e-08` relative,
  respectively).
- Status: **RESOLVED in explicit consistent mode; legacy preserved.**

### PC-002 Bernoulli, `eta=30`, `x=0`

- Legacy control: floored-objective finite-difference score `0`, implemented
  score approximately `-1`, implemented precision `1e-8`.
- Consistent: score `-0.9999999999999065`; finite-difference absolute error
  `2.24e-12`. Mathematical/runtime curvature is approximately `9.35e-14`, with
  no artificial floor.
- Extreme-tail second differences are cancellation dominated, so no false
  finite-difference curvature precision claim is made.
- Status: **RESOLVED in explicit consistent mode; legacy preserved.**

## Finite differences

The reference objectives were written directly from the equations above rather
than copied from model mean/likelihood helpers.

| Fixture | gradient step | curvature step | score abs. error | curvature abs. error |
|---|---:|---:|---:|---:|
| Bernoulli `eta=.4,x=1` | `1e-6` | `1e-4` | `2.74e-11` | `9.99e-09` |
| Bernoulli `eta=30,x=0` | `1e-4` | N/A | `2.24e-12` | cancellation-limited; not claimed |
| Bernoulli `eta=-30,x=1` | `1e-4` | N/A | `2.24e-12` | cancellation-limited; not claimed |
| Poisson `eta=.7,x=3` | `1e-6` | `1e-4` | `1.93e-10` | `7.62e-08` |
| Poisson `eta=11.5,x=3` | `1e-5` | `1e-3` | `1.75e-06` | `8.23e-03` |
| Poisson `eta=-25,x=0` | `1e-3` | `1e-1` | `2.31e-18` | `1.16e-14` |
| mixed Gaussian/Bernoulli/Poisson z fixture | `1e-6` | `2e-4` | `2.26e-10` max | `7.51e-09` max |

## Runner selection

- omitted/default `numerics_mode`: `"legacy"`;
- scalar + `"consistent"`: `DualExpFamLSMConsistent`;
- `family_x_list` + `"consistent"`:
  `DualExpFamLSMPerColumnConsistent`;
- invalid mode: explicit `ValueError`;
- NB + `"consistent"`: explicit `NotImplementedError`;
- result dict records the selected `numerics_mode`.

On the deterministic smoke fixture, omitting the option and passing
`numerics_mode="legacy"` produced exactly equal Z/F/sigma, scalar parameters,
strict Q, criterion value, and failure counters.

Small deterministic scalar Poisson/Poisson and mixed per-column/Poisson EM
smokes both completed with finite strict Q, no NaN, and the expected new class.
No large or claim-bearing experiment was run.

For diagnostics, legacy mode retains `poisson_clip_diagnostics`. Consistent mode
returns `status="not_applicable"` with an explanation and `x_side/y_side=None`;
it does not report a fabricated zero clip rate. For consistent Poisson only,
`predict_mu_y` returns the guarded canonical mean without the legacy display
clipping. Gaussian prediction behavior, including its existing display clip,
remains exactly legacy; Bernoulli already lies inside the display bounds.

## NB isolation

`DualExpFamLSMNB`, its inheritance, NB2 likelihood, score, Fisher curvature, and
clipping are unchanged. Default/legacy NB construction still selects exactly
`DualExpFamLSMNB`. The new mode does not imply that canonical Poisson changes fix
NB; `numerics_mode="consistent"` with `family_y="nb"` is rejected explicitly.

Both existing NB suites passed unchanged.

## Existing tests

All were executed without modifying their assertions:

```text
python expfam/src/experimental/test_percolumn_model.py              PASS
python expfam/src/experimental/test_experimental_models.py          PASS
python expfam/src/experimental/test_diagnostics_validation.py       PASS
python expfam/src/experimental/test_nb_math_audit.py                 PASS
python expfam/src/test_dual_expfam.py                                PASS (9/9 families)
```

## New tests

```text
python -W error expfam/src/experimental/test_objective_consistent_numerics.py
```

Result: **9/9 PASS**, including legacy counterexamples, consistent corrections,
Bernoulli tails, Poisson interior/old-boundary/negative-tail/overflow guards,
scalar X/Y paths, mixed per-column finite differences, runner selection,
deterministic EM smokes, strict-Q, NB isolation, and Gaussian regression.
The Gaussian regression covers likelihood, gradient, precision, F update, X
variance update, Gaussian-Y dispersion update, and clipped plug-in prediction.
The overflow-boundary checks cover both float32 and float64: the value immediately
below each dtype-derived boundary remains finite, while the next representable
value above it fails explicitly.

## Strict-Q consistency

`eval_utils.calc_Q_dual_strict_exp` calls the new model's consistent X/Y
likelihoods. It adds only parameter-independent family constants:

- `-sum(log(x!))` for scalar or selected per-column Poisson X;
- `-sum(log(y!))` over observed upper-triangle Poisson Y pairs.

A deterministic manual reconstruction matched strict-Q to `1e-12`. Thus the Q
objective and the new score/curvature describe the same objective up to those
constants. Historical `BIC`/`calc_bic_exp` names and the repository's Q-based
complete-data/ICL-type convention are unchanged; this report does not call it
Schwarz BIC.

## Lightweight CI

```text
python -m compileall -q -x 'archive[\\/]' tools expfam/src reproduction/src
python tools/validate_registry_paths.py --self-test
python tools/validate_registry_paths.py
```

- Syntax: PASS, exit 0.
- Validator self-test: 240/240 PASS, FAIL 0, SKIP 0, exit 0.
- Registry: TRUE_BROKEN 0, exit 0.

## Known limitations

- This is an experimental forward lineage, not a retroactive repair of prior
  estimates or their provenance.
- Canonical Poisson deliberately fails when float dtype cannot represent the
  requested mean/objective. It does not supply a capped/smooth surrogate.
- At extreme Bernoulli logits the true curvature is below practical
  second-difference resolution and may round to zero in the runtime dtype; no
  independent curvature floor is introduced.
- Numerical consistency does not establish empirical usefulness of
  heterogeneous X families.
- NB objective consistency remains separate, explicitly out of scope.

## Research integrity

- Historical implementation changed: **NO**
- Historical results changed or regenerated: **NO**
- CSV/runinfo/NPY/NPZ/figures/PDF changed: **NO**
- Registry or canonical research docs changed: **NO**
- Claim-bearing validation experiment run: **NO**
- Provenance rewritten: **NO**

## Thesis status

The standard thesis/main model remains the scalar-family historical lineage.
The consistent scalar/per-column classes are explicit future experimental
options only. The per-column prototype is not promoted to the thesis/main method.

**Prototype promoted to thesis method: NO.**
