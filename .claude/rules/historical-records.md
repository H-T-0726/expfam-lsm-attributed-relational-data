---
name: historical-records
description: How to read and amend dated, frozen records (phase reports, math notes, handoffs) without rewriting history.
paths:
  - "reports/**"
  - "docs/math_notes/**"
  - "docs/teacher/**"
  - "CLEANUP_MANIFEST.md"
  - "START_HERE.md"
  - "expfam/handoff.md"
---

# Reading and amending historical records

These documents are **dated records of what was known at the time**. They are
not descriptions of the current state, and several of them are known to
disagree with it.

## Read them as testimony, not as specification

When one of these files contradicts a canonical document, the canonical
document wins for the *present* question — but the historical file is still
correct as a record of its own moment. Both facts hold simultaneously. Report
the disagreement; do not silently pick one.

The source-of-truth order is `AGENTS.md` §4.

## Do not update the body text

Do not edit a dated record so that it agrees with what is known now. In
particular:

- Do not change a conclusion that was later revised.
- Do not change hedged wording ("not yet confirmed", "generation script not
  found") into confident wording because the question has since been settled.
  That converts an honest record of uncertainty into a false claim that the
  uncertainty never existed.
- Do not correct numbers, paths, or file names that were accurate when written
  and have since changed.
- Do not delete a record because its conclusion was superseded.

## How to supersede instead

When you notice a historical record disagrees with the current state, report
the disagreement in your answer. Do not create a new report just to record it.

When the researcher asks for a correction, it happens **forward**, in a newer
artifact — a row or dated note in the relevant canonical document, or a new
dated record under `reports/<phase>/` — never by editing the old record. The
dated forward-update notes in `KNOWN_ISSUES.md` follow this pattern.

If a historical file must carry a pointer, the only acceptable change is an
*added*, clearly dated forward reference that leaves the original text intact.
Ask before adding even that.

## Frequent traps in this repository

- A path named in an old document may no longer exist. Its presence in the
  record is not evidence that it exists now (KI-009).
- `expfam/CLAUDE.md` is an old-session file and is excluded from loading on
  purpose; the root `AGENTS.md` (imported by the root `CLAUDE.md`) is
  authoritative (KI-008).
- AI-generated reports (`GEMINI_REPORT_*`, `docs_for_notebooklm/*`) are
  derivative and unverified. They are never a primary source for a number
  (KI-007).
