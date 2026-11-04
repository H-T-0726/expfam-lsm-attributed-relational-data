---
name: prepare-experiment
description: Pre-flight review of an experiment plan (issue, branch, lineage, families, data/split, seed, output path, metrics, provenance, registry row). Run only when the researcher explicitly asks for a plan check with /prepare-experiment.
disable-model-invocation: true
---

# Prepare an experiment (plan check)

Invoked explicitly by the researcher to review an experiment plan before it
runs. It reviews the plan; it does not run the experiment. It is not a
precondition for approved work: an experiment already approved under
`CLAUDE.md` §6 (Approved Task) proceeds without this skill.

Severity follows `docs/research_operating_policy.md` (BLOCKER / WARNING /
DIAGNOSTIC). Only an item that would make the result uninterpretable or
unciteable is a BLOCKER; everything else is recorded and does not stop the plan.

## Checklist

**Scope and workspace**

1. GitHub Issue the run belongs to.
2. Working branch (not `main`). Note any uncommitted or untracked files so they
   are not mixed into the run's commit; their presence alone is not a blocker.

**Model** (`CLAUDE.md` §1, §3)

3. Implementation lineage — exact model file. Prototypes under `experimental/`
   cannot support manuscript claims.
4. X family (single shared family in the standard model; per-column only in the
   prototype) and Y family, with the dispersion parameters they imply
   (`Σ_X` for Gaussian-X, `σ_Y²` for Gaussian-Y).
5. k, n, d and any sweep ranges.

**Data**

6. Dataset — synthetic scenario or real data, exact source and preprocessing.
7. Train/test split and leakage risk. Strict held-out link prediction needs pair
   masking, which only the experimental masked model provides (KI-012).

**Execution**

8. Seeds and number of trials/replicates.
9. Output directory — a new path; nothing existing is overwritten.
10. Metrics — computed vs. quoted. The Q-based criterion keeps its `BIC` name in
    code/CSV and is not called a Schwarz BIC in prose (KI-010).

**Provenance**

11. runinfo — written in the established column order
    (`script,datetime,git_head,branch,inputs,note`); for new runs, also
    environment provenance per KI-014. Past runinfo is never backfilled.
12. Registry row to append to `EXPERIMENT_REGISTRY.md` after the run, with the
    intended `状態` / `原稿採用`. Existing rows are never rewritten.

## Output

The filled checklist, the open items with their severity, and the command that
would run the experiment. Running it is the researcher's call; if they already
approved the run, say which items (if any) are BLOCKERs and otherwise proceed as
they direct.
