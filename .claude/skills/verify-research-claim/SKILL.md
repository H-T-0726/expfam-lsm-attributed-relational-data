---
name: verify-research-claim
description: Use when a research claim, number, or ratio from this repository needs to be traced to primary evidence before it goes into a manuscript, slide, report, or answer - resolves claim to canonical docs, then the experiment registry, then the primary CSV/runinfo/code/paper, and reports claimability and evidence status as two separate verdicts.
---

# Verify a research claim

Read-oriented workflow. This skill **never** modifies code, results, figures, or
documents. If verification reveals that something should change, report it and
stop; the change is a separate, human-approved task.

## When to use

A claim is being made or repeated: a number ("41.45×"), a comparison ("fixed is
better"), a property ("BIC selects k=3"), or an attribution ("the original paper
has no 1/2"). Also use it when asked to check something someone else wrote.

## Workflow

Work through the chain in order. Do not skip a level because a later level looks
convenient.

### 1. State the claim precisely

Write the claim as one sentence. Then note what would have to be true for it to
hold: which scenario, which distribution families on the X side and the Y side
(true *and* estimated), which dataset, which metric, which k, how many trials.
A claim that cannot be pinned to these is already `UNRESOLVED` — say so instead
of guessing which experiment was meant.

### 2. Canonical documents — determine **claimability**

Check `KNOWN_ISSUES.md`, specifically the two lists at the bottom:
"今すぐ主張してよいこと" and "まだ主張してはいけないこと". Then `RESEARCH_MASTER.md`
and `CLAUDE.md` for the accepted formulation; `CLAUDE.md` §5 lists claims that
must carry an explicit qualifier, and a claim that drops its qualifier is not
the same claim.

This step decides **claimability only**:

| claimability | meaning |
|---|---|
| `ALLOWED` | The claim may be stated as-is. |
| `QUALIFIED ONLY` | May be stated only with a required qualifier (e.g. the implementation series, "in-sample reconstruction", the Newton-direction caveat). Name the qualifier. |
| `NOT ALLOWED` | Listed under "まだ主張してはいけないこと", or otherwise ruled out by canonical policy. |

**Claimability is not evidence.** `NOT ALLOWED` is a policy verdict about what
may be asserted; it says nothing about what the data show. Do **not** convert it
into `CONTRADICTED`, and do not stop here — continue through steps 3 and 4 and
determine the evidence status independently.

The two often diverge, and the divergence is the useful output. A number can be
`VERIFIED` in the CSV yet `NOT ALLOWED` because the conditions do not support
the sentence being built around it; a claim can be `NOT ALLOWED` yet
`UNRESOLVED` simply because nobody has run the experiment that would settle it.

### 3. EXPERIMENT_REGISTRY.md

Locate the row. Record the `状態` and `原稿採用` values — a number from an `old`,
`ai_generated`, `unverified`, or `fixed_support` row is not manuscript-quotable
even when it is numerically correct. Follow the row to the exact CSV path and
the owning script.

### 4. Primary evidence

Open the primary source and confirm the number is actually there:

- result CSV — find the row and column, not just the file
- `runinfo` — script, datetime, git head, branch, inputs
- the model/experiment source code — for claims about what the method does
- `paper/A_study_on_latent_structural_models_for_binary_rel.pdf` — for any claim
  about the baseline paper's printed equations

Report the file, and the row/column or equation number, that carries the value.

### 5. Label the **evidence status**

Decided from the primary evidence chain alone, independently of step 2:

| label | meaning |
|---|---|
| `VERIFIED` | Found in primary evidence, conditions match the claim exactly. |
| `DERIVED` | Follows from primary evidence by stated reasoning or arithmetic. Show the derivation. |
| `HISTORICAL` | Supported only by a dated record; true as a record of its time, not confirmed for the present state. |
| `UNRESOLVED` | No primary source located, or the claim is too underspecified to test. This is the correct label when the evidence is simply absent — including when the claim is `NOT ALLOWED`. Not a failure to report; report it. |
| `CONTRADICTED` | **Primary evidence positively disagrees** — the CSV, the code, or the paper says something incompatible. Reserved for this case only. A canonical policy entry saying a claim must not be made is not, by itself, contradicting evidence. |

## Non-negotiables

- **Every number carries its implementation lineage.** State whether it came
  from the `0.5`-bearing series, the `fixed` series, or an `experimental/`
  prototype (`CLAUDE.md` §3, KI-002). A number without a lineage is not
  reportable. Never place numbers from different series in one table or figure.
- **Never confirm a number from a derived document alone.** `GEMINI_REPORT_*`,
  `docs_for_notebooklm/*`, old handoffs, and previous session summaries are
  unverified derivatives (KI-007). They may point you at a source; they cannot
  be the source.
- **Similar numbers are not the same number.** 23.6× / 41.45× / 38.97× come
  from different series and different conditions (KI-003). Verify each
  separately and print the conditions next to each.
- **Do not reconcile a contradiction on your own authority.** If two sources
  disagree, report both with their provenance and let the researcher decide.
- **Do not rewrite historical wording** to match what verification found. Record
  the finding separately.
- **Preserve terminology.** The model-selection criterion is a Q-based
  complete-data / ICL-type criterion, not a Schwarz BIC, even though the
  function, the CSV column, and past results are all named `BIC` — those names
  stay unchanged (`CLAUDE.md` §5, KI-010).
- **The 1/2 coefficient has five distinct lineages** (`CLAUDE.md` §2). Never
  collapse them. In particular the baseline paper's printed equations *do*
  carry the 1/2.

## Output

Report the two axes separately for every claim. Collapsing them hides exactly
the cases that matter.

- **claim** — restated in one sentence
- **claimability** — `ALLOWED` / `QUALIFIED ONLY` / `NOT ALLOWED`, with the
  required qualifier or the canonical entry that rules it out
- **evidence status** — `VERIFIED` / `DERIVED` / `HISTORICAL` / `UNRESOLVED` /
  `CONTRADICTED`
- **primary source** — path plus row/column or equation number
- **implementation lineage** — which model series produced the number
- **conditions** — scenario, families (true and estimated), dataset, metric,
  k, seed, trial count
