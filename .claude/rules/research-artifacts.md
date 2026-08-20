---
name: research-artifacts
description: How to handle generated experiment artifacts (results, runinfo, figures, arrays) in this repository.
paths:
  - "expfam/results/**"
  - "expfam/figures/**"
  - "figures/**"
  - "reproduction/results/**"
  - "**/*.csv"
  - "**/*runinfo*"
  - "**/*.npy"
  - "**/*.npz"
---

# Working with generated research artifacts

These files are **outputs**, not sources. They are evidence for claims in the
manuscript, so their value comes entirely from being traceable to the run that
produced them.

## Never hand-edit

Do not fix a value, a header, a column order, or a typo in a result file by
editing it. A hand-edited artifact is indistinguishable from a fabricated one
once the session ends. If a number looks wrong, the defect is in the script or
in the interpretation — fix that and regenerate.

The same applies to `runinfo` files. Do not backfill fields that were not
recorded at run time (environment, seed, package versions). Reconstructed
provenance is a guess wearing the costume of a measurement — that is exactly
the failure `KNOWN_ISSUES.md` KI-014 exists to prevent.

## Regenerate through the owning script

Every artifact *should* have an owning script. Locate it in
`EXPERIMENT_REGISTRY.md` when possible, before touching anything. Some
artifacts in this repository do not have one — `KNOWN_ISSUES.md` KI-004 records
a figure whose generation script could not be found. When no owning script can
be located, treat the artifact as non-reproducible / provenance unresolved and
say so, rather than inventing a regeneration path.

Run scripts from the repository root; output locations are baked into the
scripts and differ between phases.

## Before quoting a number from one of these files

Confirm all of the following, and state them alongside the number:

- **Implementation lineage** — which model file produced it. The `0.5`-bearing
  series, the `fixed` series, and `experimental/` prototypes are different
  models and their numbers do not belong in the same table or figure
  (`CLAUDE.md` §3, KI-002).
- **Scenario and families** — scenario A/B/C, and the X-side and Y-side
  distribution families, for both the true and the estimated model.
- **Seed and trial count** — several comparison CSVs rest on 5 trials.
- **Metric** — RMSE(Z), AUC/AP, NMI/ARI, and the Q-based criterion recorded in
  the `BIC` column are not interchangeable.
- **Registry row** — the `状態` and `原稿採用` columns say whether the number is
  quotable in the manuscript at all.

## When adding new artifacts

Append a row to `EXPERIMENT_REGISTRY.md`; never rewrite or delete an existing
row, including its path strings. Paths are part of the provenance record even
when the layout looks inconsistent.

Existing artifacts from earlier runs are not overwritten by a new run. Write to
a new output directory instead.
