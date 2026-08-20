---
name: prepare-experiment
description: Use before starting or re-running any experiment in this repository - walks the pre-flight checklist (issue, branch, lineage, families, dataset, split, seed, output path, runinfo, metrics, registry plan) and stops without executing anything.
---

# Prepare an experiment

Pre-flight checklist. **This skill does not run experiments.** It ends with a
filled-in plan for the researcher to approve. Running the experiment is a
separate, explicitly requested step.

## Checklist

Work through every item. An item that cannot be answered is a blocker, not a
detail to settle later — an experiment launched with an unanswered item produces
a result nobody can cite.

### Scope and workspace

1. **GitHub Issue** — which issue does this run belong to? Ongoing work is
   tracked there, not in the canonical documents.
2. **Branch** — a working branch exists and is checked out
   (`git switch -c experiment/<issue#>-<slug>`), the tree is clean, and the
   branch is not `main`.

### Model

3. **Implementation lineage** — exactly which model file. The `0.5`-bearing
   series, `model_dual_expfam_fixed.py`, and `experimental/` prototypes are
   different models; prototypes cannot support manuscript claims
   (`CLAUDE.md` §3). Record the choice; it must accompany every number the run
   produces.
4. **X family** — and whether the standard single-family model is enough. A
   per-column family list only exists in the experimental prototype.
5. **Y family** — plus the dispersion parameters this implies: `Σ_X` for
   Gaussian-X, `σ_Y²` for Gaussian-Y.
6. **k, n, d** and any sweep ranges.

### Data

7. **Dataset** — synthetic scenario (A/B/C) or real data, and its exact source
   and preprocessing.
8. **Train/test split** — how pairs and nodes are split, if at all.
9. **Leakage risk** — for held-out link prediction, confirm the split is strict
   (unseen pairs) rather than in-sample reconstruction. The standard API does
   not support pair masking; only the experimental masked model does, and
   claiming strict held-out without it is prohibited (KI-012).

### Execution

10. **Seed** — recorded and reproducible. Also the trial count; several existing
    comparison results rest on 5 trials, which is a real limitation.
11. **Output directory** — a *new* path. Never overwrite an existing result
    directory. Confirm the path against `EXPERIMENT_REGISTRY.md`.
12. **Metrics** — which are computed and which will actually be quoted. If the
    Q-based selection criterion is used, keep the existing `BIC` naming in code
    and CSV columns, and do not call it a Schwarz BIC in prose (KI-010).

### Provenance

13. **runinfo** — will be written, matching the established column order
    (`script,datetime,git_head,branch,inputs,note`). Do not reorder or rename
    existing columns.
14. **Environment provenance** — per KI-014, new runs should additionally record
    `python_version`, `platform`, requirements/lock provenance, and the git head,
    so that this run *is* reproducible even though past runs are not.

    Two hard constraints: past `runinfo` files are never edited to add these
    fields retroactively, and this skill does not change the runinfo format on
    its own. If the recording script needs to be extended, propose it as a task
    and stop.

15. **Registry plan** — the row to be *appended* to `EXPERIMENT_REGISTRY.md`
    after the run, including the intended `状態` and `原稿採用` values. Existing
    rows and their path strings are never rewritten or deleted.
16. **Historical result protection** — confirm the run writes nothing into an
    existing results directory, and touches no existing CSV, runinfo, or figure.

## Output

A filled checklist, an explicit list of unanswered items, and the exact command
that *would* be run. Then stop and wait for approval.
