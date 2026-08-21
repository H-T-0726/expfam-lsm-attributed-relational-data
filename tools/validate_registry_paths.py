"""Read-only validator for path references in research provenance documents.

Phase 5a of the repository migration plan (issue #15).

The validator reads Markdown provenance documents -- by default
``EXPERIMENT_REGISTRY.md`` -- extracts backtick-delimited tokens, decides which
of them are path references, and checks whether those references can be
resolved against the current working tree.

The validator itself performs no repository write:

* it never opens a repository file for writing, and never creates, moves or
  removes one;
* no ``git`` command and no other subprocess is executed;
* no network access;
* the filesystem walk is confined to the repository root and does not follow
  symlinks.

One byproduct is outside the module's control: importing or byte-compiling it
makes CPython write ``tools/__pycache__/validate_registry_paths.*.pyc``, which
``.gitignore`` excludes.  Likewise ``--self-test`` writes, and in two suites
also deletes, its fixtures inside ``tempfile.TemporaryDirectory()``, which
lands outside the repository unless ``TMPDIR``/``TEMP`` has been pointed inside
it.  Those two suites additionally refuse to run against any directory that is
not one of their own fixture roots (see ``_refuse_outside_fixture``).

Two deliberate semantics are worth stating up front:

* Existence is decided from the working tree, not from ``git ls-files``.  The
  single exception is a reference listed in
  :data:`LOCAL_ONLY_ARTIFACT_REFERENCES`, whose *classification* is deliberately
  not taken from the tree at all: it is LOCAL_ONLY_ARTIFACT in either
  environment.  What the tree still decides for such a reference is the
  presence-derived detail -- ``local_presence``, ``matches``, ``match_count``,
  ``resolved_via`` and the presence clause of ``reason`` -- so those fields, and
  the summary counters computed from them, do differ between a workstation and
  CI.  Every other reference is decided from the tree outright, so a fresh
  checkout may see fewer files than this run does.
* Matching is case sensitive on every platform, including Windows.  A reference
  whose case does not match the tree is reported, because it would fail on a
  case-sensitive filesystem.

``--self-test`` runs two suites: fixture checks against a throwaway tree (see
the note on ``TMPDIR`` above) and adversarial checks against the real working
tree, whose anchors are derived from whatever currently resolves rather than
from hard-coded file names.

Two classifications exist so that a documented, non-blocking finding is never
disguised as a verified one (issue #19, Phase 5a.1):

* ``LOCAL_ONLY_ARTIFACT`` -- the reference is one of the individually listed
  entries in :data:`LOCAL_ONLY_ARTIFACT_REFERENCES`, matched on the exact pair
  (source document, raw token), and the forms actually being resolved equal the
  path or pattern that entry records.  Such an artifact legitimately lives on a
  research workstation and is legitimately absent from a fresh CI checkout, so
  its *classification* is stable across both, while the presence-derived fields
  listed above differ.  It is non-blocking, listed individually and counted on
  its own line -- never folded into EXISTS_LITERAL or PATTERN_RESOLVED.

  Registration is the authorization, and it is deliberately the only one.
  ``.gitignore`` is corroborating evidence: an entry names the rule expected to
  exclude its path, that rule is re-read from the repository on every run, and
  the entry lapses if the rule is gone, if the file gains a negation this
  reader cannot evaluate, or if the rule would not in fact exclude the recorded
  path.  What ``.gitignore`` can never do is create an entry.  An unregistered
  reference is classified by the ordinary rules however its suffix reads, so a
  missing ``expfam/results/wine_typoooo.npy`` is TRUE_BROKEN exactly like a
  missing ``.csv`` would be, and neither a near miss of a registered reference
  nor a registered token appearing in another document inherits the entry.

* ``KNOWN_NOTATION_DEFECT`` -- the reference is one of the individually listed
  entries in :data:`KNOWN_NOTATION_DEFECTS`, matched on the exact pair
  (source document, raw token) so that neither a near miss nor the same token
  in another document is covered.  The waiver only applies while the recorded
  forward correction still resolves against the tree, so it can never outlive
  the artifacts it points at.  The historical text stays as written; the defect
  stays visible and separately counted; it simply does not block CI.

Six limitations are worth knowing before wiring this into CI:

* A reference whose first segment names an existing top-level directory is
  treated as repository-relative, so it is reported broken rather than read as
  an abbreviation.  This repository has both a top-level ``figures/`` and an
  ``expfam/figures/``, so a future row abbreviating the latter to
  ``figures/...`` would be reported as broken even though the artifact exists.
  That is the deliberate trade for being able to detect a renamed or removed
  top-level directory at all.

* Directory names taken from ``.gitignore`` are excluded from the index and are
  listed on every run.  The rule is general, so a future bare ``name/`` line in
  ``.gitignore`` would remove that whole subtree from the index and turn
  references into it from blocking into undecidable.  Watch the ``skipped``
  and ``indexed`` lines for a sudden change.
* Inline mathematics that contains braces, and two-part names such as Git
  branches, land in UNRESOLVED.  ``--fail-on-unresolved`` is therefore only
  usable on documents free of both.
* LOCAL_ONLY_ARTIFACT is a maintained list, which is the point: an artifact
  that stops being local-only, or a reference that is edited, has to be
  reviewed rather than silently re-derived.  The cost is that a genuinely new
  local-only reference is reported TRUE_BROKEN until someone adds it here.
  That is the intended direction to fail in -- the reference stays visible and
  blocking rather than being quietly excused.
* A registration is scoped to its document, not to the row it was written
  for.  Once ``expfam/results/wine_F.npy`` is registered for
  ``EXPERIMENT_REGISTRY.md``, any future row of that document writing the same
  token is non-blocking too -- including after the artifact is renamed or
  deleted.  An entry whose ``expected`` differs from its ``raw`` is narrower
  than that: it also requires the rebasing to land on the recorded path, so a
  future row writing the bare ``wine_F.npy`` beside a different base is not
  covered and stays UNRESOLVED.  Narrowing the key with a line number was considered and rejected:
  the document is append-only, so line numbers move, and the same reasoning
  that keeps :data:`KNOWN_NOTATION_DEFECTS` line-independent applies here.  The
  breadth is therefore deliberate, and it is bounded by the document: a
  registered token in another document inherits nothing.
* An entry states that the artifact is not in Git; it cannot state that the
  artifact was never force-added.  On 2026-08-21 ``git ls-files`` was run with
  all nine of this repository's ``*.ext`` rules as its pathspec -- ``*.pyc``,
  ``*.pyo``, ``*.pyd``, ``*.log``, ``*.tmp``, ``*.npy``, ``*.npz``, ``*.pkl``,
  ``*.pickle`` -- and returned nothing, so nothing of those forms is tracked.

Usage::

    python tools/validate_registry_paths.py
    python tools/validate_registry_paths.py --json
    python tools/validate_registry_paths.py --verbose
    python tools/validate_registry_paths.py --source KNOWN_ISSUES.md
    python tools/validate_registry_paths.py --self-test

Exit codes::

    0  run completed, no blocking finding
    1  blocking finding (TRUE_BROKEN, or UNRESOLVED with --fail-on-unresolved)
    2  invalid invocation or internal error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

SCHEMA_VERSION = 3

DEFAULT_SOURCES = ("EXPERIMENT_REGISTRY.md",)

EXIT_OK = 0
EXIT_BLOCKING = 1
EXIT_USAGE = 2

#: Always skipped while indexing the working tree.  Further directory names are
#: added at runtime from the repository's own ``.gitignore`` so that the index
#: does not depend on machine-local scratch directories such as ``.venv``.
SKIPPED_TREE_DIRS = frozenset({".git"})

#: Suffixes that make a final path segment look like a file name.  It covers
#: the extensions that occur in this repository's provenance documents plus
#: common data and archive formats, so that a plausible future reference is not
#: silently dismissed.  The set is closed on purpose: an unknown suffix costs a
#: NON_PATH classification, which is visible under ``--verbose``.
KNOWN_EXTENSIONS = frozenset(
    {
        "bat", "bib", "cfg", "cmd", "csv", "dat", "docx", "eps", "gz", "h5",
        "html", "in", "ini", "ipynb", "jpeg", "jpg", "json", "lock", "log",
        "m", "mat", "md", "npy", "npz", "parquet", "pdf", "pickle", "pkl",
        "png", "ps1", "py", "rst", "sh", "svg", "tar", "tex", "toml", "tsv",
        "txt", "xlsx", "yaml", "yml", "zip",
    }
)

#: Characters that mark a token as source code, mathematics or prose rather
#: than a path.
CODE_ONLY_CHARACTERS = frozenset("()[]<>|=$;&!\"'%^+")

#: The " (2)" copy suffix that Windows and macOS append to duplicated files.
#: The MATLAB directory of this repository really does contain names such as
#: ``EffcalcEtaNewton (1).m``, so this one shape is exempt from the parenthesis
#: rule above.  An expression such as ``k(k-1)/2`` does not match it and stays
#: NON_PATH.
COPY_SUFFIX_RE = re.compile(r" \(\d+\)")

GLOB_CHARACTERS = frozenset("*?")

MAX_BRACE_VARIANTS = 256
MAX_BRACE_DEPTH = 8
MAX_REPRESENTATIVE_MATCHES = 5
MAX_BASENAME_EVIDENCE = 5

#: Marks a self-test check whose anchor was unavailable, so it did not run.
SKIP_MARKER = "SKIP"


@dataclass(frozen=True)
class HistoricalException:
    """An intentionally preserved reference to a path absent from the tree."""

    reason: str
    evidence: str


#: Exact, individually justified exceptions.  Matching is exact equality on the
#: normalised path (no prefix, glob or directory-wide suppression); a near miss
#: such as ``archive/paper_writing_example`` is deliberately not covered.
INTENTIONAL_HISTORICAL_PATHS: dict[str, HistoricalException] = {
    "archive/paper_writing_examples": HistoricalException(
        reason=(
            "Reference PDFs of other conferences' papers. Intentionally kept as a "
            "historical reference in frozen documentation even though the directory "
            "is present neither on disk nor in the Git index."
        ),
        evidence=(
            "KNOWN_ISSUES.md KI-009 (absence re-confirmed 2026-08-20); "
            ".gitignore rule '/archive/paper_writing_examples/**'"
        ),
    ),
}


@dataclass(frozen=True)
class NotationDefect:
    """A known notation defect preserved verbatim in a historical record.

    Matching is exact equality on the pair (``source``, ``raw``): the document
    the token was written in and the token exactly as it was written there.  A
    line number is deliberately not part of the key, so appending a forward
    correction above or below the historical row cannot detach the waiver from
    it -- and, equally deliberately, the same defective token quoted inside that
    forward correction is covered by the same entry.

    ``correction`` is the reference that supersedes the defective one.  It is
    resolved against the working tree on every run, and the waiver only applies
    while it resolves: the entry cannot outlive the artifacts it points at, and
    it can never assert an existence the tree does not support.
    """

    source: str
    raw: str
    correction: str
    reason: str
    evidence: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.raw)


#: Exact, individually justified notation defects.  Each entry names one token
#: in one document; near misses (``runinfos`` for ``runinfo``), the same token
#: in another document, and any other reference to the same artifacts are all
#: outside the waiver and keep their ordinary classification.
KNOWN_NOTATION_DEFECTS: dict[tuple[str, str], NotationDefect] = {
    defect.key: defect
    for defect in (
        NotationDefect(
            source="EXPERIMENT_REGISTRY.md",
            raw=(
                "expfam/results/story_diagnostics/"
                "y_sparsity_stress_20260713_{,agg,runinfo}.csv"
            ),
            correction=(
                "expfam/results/story_diagnostics/"
                "y_sparsity_stress_20260713{,_agg,_runinfo}.csv"
            ),
            reason=(
                "Known brace-notation defect in a historical registry row: the "
                "underscore sits before the brace group instead of inside each "
                "alternative, so the empty alternative expands to the "
                "non-existent 'y_sparsity_stress_20260713_.csv'. The three "
                "artifacts the row documents do exist and are named by the "
                "recorded correction, which this run resolved. The historical "
                "text is preserved as written and is not normalised."
            ),
            # ASCII only: this string is rendered to a console whose encoding
            # varies, and to JSON written with ensure_ascii, so the Japanese
            # section heading is described rather than quoted.
            evidence=(
                "EXPERIMENT_REGISTRY.md, appended forward-correction section "
                "dated 2026-08-21 (Phase 5a.1 / issue #19, append-only); the "
                "sibling row of the same phase already uses the correct "
                "'..._trials10{,_agg,_runinfo}.csv' notation"
            ),
        ),
    )
}


@dataclass(frozen=True)
class LocalOnlyArtifactReference:
    """An explicitly registered reference to a Git-excluded research artifact.

    Registration is the authorization.  ``.gitignore`` is evidence that
    supports an entry and can withdraw it, but it never creates one: an
    unregistered reference is classified by the ordinary rules however its
    suffix reads, so a mistyped artifact name cannot become non-blocking by
    accident.  Matching is exact equality on the pair (``source``, ``raw``),
    the same discipline as :data:`KNOWN_NOTATION_DEFECTS`.

    ``expected`` is the repository-relative path or pattern the reference
    denotes once any deterministic contextual rebasing has been applied.  The
    reference is only honoured when the forms actually under consideration
    equal it, so a registration for a bare basename cannot be inherited by a
    near miss, and a registration cannot silently start covering some other
    path if the document around it changes.

    ``ignore_rule`` is the ``.gitignore`` line that is expected to exclude
    every path ``expected`` can denote.  It is re-checked against the
    repository on every run, and the entry lapses if that line is gone.
    """

    source: str
    raw: str
    expected: str
    ignore_rule: str
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.raw)


#: Exact, individually justified local-only references.  Adding an entry is a
#: reviewed decision to treat one named reference in one named document as an
#: artifact Git deliberately does not carry.  Nothing is covered implicitly:
#: neither another token with the same suffix, nor the same token in another
#: document, nor a near miss of a registered one.
LOCAL_ONLY_ARTIFACT_REFERENCES: dict[
    tuple[str, str], LocalOnlyArtifactReference
] = {
    reference.key: reference
    for reference in (
        LocalOnlyArtifactReference(
            source="EXPERIMENT_REGISTRY.md",
            raw="expfam/data/movielens_pilot/*.npy",
            expected="expfam/data/movielens_pilot/*.npy",
            ignore_rule="*.npy",
            reason=(
                "Output of the MovieLens data-preparation step. The arrays are "
                "regenerated by 'prepare_movielens_data.py' and are excluded "
                "from Git by '*.npy', so they exist on a workstation that ran "
                "the preparation and never in a fresh checkout. Registered for "
                "issue #19 Finding B; the row that writes it is a real "
                "provenance record, not a broken reference."
            ),
        ),
        LocalOnlyArtifactReference(
            source="EXPERIMENT_REGISTRY.md",
            raw="expfam/results/wine_F.npy",
            expected="expfam/results/wine_F.npy",
            ignore_rule="*.npy",
            reason=(
                "The same artifact as the bare 'wine_F.npy' entry below, "
                "written as an explicit repository-relative path. Issue #19 "
                "Finding B names the artifact in exactly this form. Stated "
                "plainly, though: every place this exact token is currently "
                "written in EXPERIMENT_REGISTRY.md is inside the Phase 5a.1 "
                "section itself -- both its prose and its registration table -- "
                "and never a historical row, so this entry was needed because of "
                "that section; a fresh-checkout run reported the token "
                "TRUE_BROKEN without it. Registered rather than reworded so the "
                "explicit spelling, which is the natural way to write the "
                "reference and the one the issue uses, is covered for future "
                "rows too. Separate from the bare-basename entry because a "
                "registration is keyed on the exact raw token and never "
                "generalises from one spelling of a path to another."
            ),
        ),
        LocalOnlyArtifactReference(
            source="EXPERIMENT_REGISTRY.md",
            raw="wine_F.npy",
            expected="expfam/results/wine_F.npy",
            ignore_rule="*.npy",
            reason=(
                "Latent-position matrix of the old-0.5 Wine run (KI-006), "
                "written in the registry as a sibling of "
                "'expfam/results/wine_dual_results.csv' and therefore denoting "
                "'expfam/results/wine_F.npy'. Excluded from Git by '*.npy'. "
                "The expected path is recorded here rather than inferred, so "
                "the registration cannot follow the token if the surrounding "
                "cell changes. Registered for issue #19 Finding B."
            ),
        ),
    )
}


class Classification(str, Enum):
    """Classification assigned by the validator to a single candidate."""

    EXISTS_LITERAL = "EXISTS_LITERAL"
    PATTERN_RESOLVED = "PATTERN_RESOLVED"
    TRUE_BROKEN = "TRUE_BROKEN"
    INTENTIONAL_HISTORICAL = "INTENTIONAL_HISTORICAL"
    #: Provably excluded from Git; may legitimately be absent in a fresh
    #: checkout.  Never an assertion that the artifact exists.
    LOCAL_ONLY_ARTIFACT = "LOCAL_ONLY_ARTIFACT"
    #: Listed in KNOWN_NOTATION_DEFECTS: a defect preserved on purpose, still
    #: reported, superseded by a forward correction that does resolve.
    KNOWN_NOTATION_DEFECT = "KNOWN_NOTATION_DEFECT"
    NON_PATH = "NON_PATH"
    UNRESOLVED = "UNRESOLVED"


#: Stable report ordering, independent of dict or filesystem iteration order.
CLASSIFICATION_ORDER = (
    Classification.EXISTS_LITERAL,
    Classification.PATTERN_RESOLVED,
    Classification.LOCAL_ONLY_ARTIFACT,
    Classification.KNOWN_NOTATION_DEFECT,
    Classification.INTENTIONAL_HISTORICAL,
    Classification.NON_PATH,
    Classification.UNRESOLVED,
    Classification.TRUE_BROKEN,
)

#: Classifications a human reviewer has to look at one by one.
PROBLEM_CLASSIFICATIONS = (
    Classification.TRUE_BROKEN,
    Classification.UNRESOLVED,
    Classification.KNOWN_NOTATION_DEFECT,
    Classification.LOCAL_ONLY_ARTIFACT,
    Classification.INTENTIONAL_HISTORICAL,
)

#: Non-blocking by construction: each names a documented, individually visible
#: situation rather than an unverified reference.  The exit status acts on
#: TRUE_BROKEN (and on UNRESOLVED under --fail-on-unresolved) only.
NON_BLOCKING_BY_POLICY = (
    Classification.KNOWN_NOTATION_DEFECT,
    Classification.LOCAL_ONLY_ARTIFACT,
    Classification.INTENTIONAL_HISTORICAL,
)


# --------------------------------------------------------------------------
# candidate model
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    """One backtick-delimited token together with everything decided about it."""

    source: str
    line: int
    column: int
    cell: int | None
    raw: str
    normalized: str = ""
    path_like: bool = False
    classification: Classification = Classification.NON_PATH
    reason: str = ""
    #: How the candidate was resolved: "" when it was not, "root" when it
    #: resolved as written, "inferred-context" when a base directory had to be
    #: taken from the surrounding table cell.  A structured field rather than a
    #: substring of ``reason``, so that rewording a message cannot silently
    #: change the reported counts.
    resolved_via: str = ""
    #: True when the verdict depends on a file type that ``.gitignore``
    #: excludes, so a fresh checkout may classify this candidate differently.
    #: Structured rather than inferred from the evidence text.
    gitignore_sensitive: bool = False
    #: For LOCAL_ONLY_ARTIFACT only: "present" or "absent" in the tree this run
    #: walked.  The classification stays the same in both cases.  This field is
    #: one of several that legitimately differ between a workstation and CI --
    #: ``matches``, ``match_count``, ``resolved_via`` and the presence clause of
    #: ``reason`` move with it, as do the summary counters computed from them.
    local_presence: str = ""
    #: The reference that supersedes or explains the raw token: the
    #: forward-corrected path for KNOWN_NOTATION_DEFECT, and the registered
    #: expected path or pattern for LOCAL_ONLY_ARTIFACT.
    correction: str = ""
    variants: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)
    match_count: int = 0
    evidence: list[str] = field(default_factory=list)

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line}:c{self.column}"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "line": self.line,
            "column": self.column,
            "cell": self.cell,
            "raw": self.raw,
            "normalized": self.normalized,
            "path_like": self.path_like,
            "classification": self.classification.value,
            "resolved_via": self.resolved_via,
            "gitignore_sensitive": self.gitignore_sensitive,
            "local_presence": self.local_presence,
            "correction": self.correction,
            "reason": self.reason,
            "variants": list(self.variants),
            "match_count": self.match_count,
            "representative_matches": self.matches[:MAX_REPRESENTATIVE_MATCHES],
            "evidence": list(self.evidence),
        }


# --------------------------------------------------------------------------
# markdown extraction
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s{0,3}(?:```+|~~~+)")
_INLINE_CODE_RE = re.compile(r"(?P<ticks>`+)(?P<body>.+?)(?P=ticks)")


def _split_table_cells(line: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of the cells of a Markdown table row.

    Cell boundaries are unescaped ``|`` characters that sit outside inline code
    spans, so a pipe inside backticks does not split a cell.
    """
    spans: list[tuple[int, int]] = []
    protected = [False] * len(line)
    for match in _INLINE_CODE_RE.finditer(line):
        for index in range(match.start(), match.end()):
            protected[index] = True

    start = 0
    for index, char in enumerate(line):
        if char != "|" or protected[index]:
            continue
        if index > 0 and line[index - 1] == "\\":
            continue
        spans.append((start, index))
        start = index + 1
    spans.append((start, len(line)))
    return spans


def _cell_index_for(spans: Sequence[tuple[int, int]], offset: int) -> int | None:
    for index, (start, end) in enumerate(spans):
        if start <= offset < end:
            return index
    return None


def extract_candidates(source_name: str, text: str) -> list[Candidate]:
    """Extract inline-code tokens from Markdown text.

    Fenced code blocks are skipped entirely: they hold example code, not
    provenance references, and indexing them would flood the report.
    """
    candidates: list[Candidate] = []
    in_fence = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        is_table_row = line.lstrip().startswith("|")
        spans = _split_table_cells(line) if is_table_row else []

        for match in _INLINE_CODE_RE.finditer(line):
            body = match.group("body")
            if not body.strip():
                continue
            column = match.start("body") + 1
            cell = _cell_index_for(spans, match.start()) if is_table_row else None
            candidates.append(
                Candidate(
                    source=source_name,
                    line=line_number,
                    column=column,
                    cell=cell,
                    raw=body,
                )
            )
    return candidates


# --------------------------------------------------------------------------
# normalisation and path-likeness
# --------------------------------------------------------------------------


def normalize_token(raw: str) -> str:
    """Normalise a raw token to a POSIX-style repository-relative form.

    Backslashes become ``/`` so that Windows-style references are handled;
    duplicate separators collapse; a leading ``./`` is dropped.  A trailing
    ``/`` is preserved because it carries the "this is a directory" intent.
    """
    token = raw.strip()
    token = token.replace("\\", "/")
    while "//" in token:
        token = token.replace("//", "/")
    while token.startswith("./"):
        token = token[2:]
    return token


def _extension_of(segment: str) -> str | None:
    _, dot, suffix = segment.rpartition(".")
    if not dot or not suffix:
        return None
    return suffix.lower()


def _has_known_extension(segment: str) -> bool:
    suffix = _extension_of(segment)
    return suffix is not None and suffix in KNOWN_EXTENSIONS


def _has_extension_signal(segment: str) -> bool:
    """True when a segment ends in something shaped like a file extension.

    A wildcarded suffix counts as well, because the registry writes figure pairs
    as ``name.*``.  Without this the reference would be dismissed as a plain
    identifier and a broken figure reference could never be detected.
    """
    suffix = _extension_of(segment)
    if suffix is None:
        return False
    if suffix in KNOWN_EXTENSIONS:
        return True
    return any(char in GLOB_CHARACTERS for char in suffix)


def _strip_braces(token: str) -> str:
    """Remove brace groups so structural tests ignore their contents."""
    previous = None
    current = token
    while previous != current:
        previous = current
        current = re.sub(r"\{[^{}]*\}", "", current)
    return current


def is_path_like(normalized: str) -> tuple[bool, str]:
    """Decide whether a normalised token is a path reference.

    Returns ``(path_like, reason)``.  The reason is reported for both answers so
    that a human reviewer can see why a token was or was not followed up.
    """
    if not normalized:
        return False, "empty token"

    without_copy_suffix = COPY_SUFFIX_RE.sub("", normalized)
    if any(char in CODE_ONLY_CHARACTERS for char in without_copy_suffix):
        return False, "contains characters used by code, mathematics or prose"

    body = normalized.rstrip("/")
    if not body:
        return False, "separator-only token"

    outside_braces = _strip_braces(normalized)
    segments = [segment for segment in body.split("/") if segment]

    signals: list[str] = []
    if "/" in outside_braces.rstrip("/"):
        signals.append("path separator")
    if normalized.endswith("/"):
        signals.append("trailing separator")
    if segments and _has_extension_signal(segments[-1]):
        signals.append("file extension")

    if not signals:
        if "{" in normalized or "}" in normalized:
            # A bare brace group names alternatives without a parent directory.
            # It is kept as a candidate so that find_ambiguity() can report why
            # it cannot be located, rather than dismissing it as non-path.
            return True, "brace group without separator or file extension"
        return False, "no path separator, no trailing separator, no file extension"

    return True, "path signals: " + ", ".join(signals)


def find_ambiguity(normalized: str) -> str | None:
    """Return a reason string when a token cannot be interpreted unambiguously.

    Ambiguous notations are never reported as broken: the validator cannot tell
    what they were meant to denote, so they are surfaced for human review.
    """
    outside_braces = _strip_braces(normalized)

    if "{" in outside_braces or "}" in outside_braces:
        return "unbalanced brace group"

    if "," in outside_braces:
        return (
            "comma outside a brace group: looks like several references in one "
            "token; the validator does not guess a separator"
        )

    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return "absolute or drive-qualified path: outside the repository root"

    body = normalized.rstrip("/")
    segments = [segment for segment in body.split("/") if segment]

    if any(segment in ("..", ".") for segment in segments):
        return "contains '..' or '.': may leave the repository root"

    if "{" in normalized:
        stripped_segments = [
            _strip_braces(segment) or segment for segment in segments
        ]
        has_separator = "/" in outside_braces.rstrip("/")
        has_extension = bool(stripped_segments) and _has_extension_signal(
            stripped_segments[-1]
        )
        if not has_separator and not has_extension:
            return (
                "brace group without a parent directory or file extension: names "
                "alternatives whose location cannot be derived"
            )

    if interpret_extension_alternation(normalized) is not None:
        return None

    for segment in segments[:-1]:
        if _has_known_extension(_strip_braces(segment) or segment):
            return (
                f"non-final segment {segment!r} carries a file extension, and "
                "the token is not the 'name.pdf/png' alternation shape: a file "
                "cannot also be a directory"
            )

    return None


def interpret_extension_alternation(normalized: str) -> list[str] | None:
    """Rewrite ``dir/name.pdf/png`` into ``dir/name.pdf`` and ``dir/name.png``.

    The registry writes a figure pair that way.  The shape is recognised
    structurally, not by listing particular extensions: the final segment must
    be a bare known extension carrying no dot of its own, the segment before it
    must end in a known extension, and no earlier segment may carry one.  Both
    rewritten forms have to resolve, so the interpretation can confirm a pair
    but never hide a missing half.
    """
    body = normalized.rstrip("/")
    if normalized.endswith("/"):
        return None
    segments = [segment for segment in body.split("/") if segment]
    if len(segments) < 2:
        return None

    last = segments[-1]
    if "." in last or last.lower() not in KNOWN_EXTENSIONS:
        return None

    parent = segments[-2]
    parent_extension = _extension_of(_strip_braces(parent) or parent)
    if parent_extension is None or parent_extension not in KNOWN_EXTENSIONS:
        return None
    if parent_extension == last.lower():
        return None

    for segment in segments[:-2]:
        if _has_known_extension(_strip_braces(segment) or segment):
            return None

    stem = parent[: parent.rfind(".")]
    prefix = "/".join(segments[:-2])
    head = f"{prefix}/" if prefix else ""
    return [f"{head}{stem}.{parent_extension}", f"{head}{stem}.{last}"]


def segment_count(normalized: str) -> int:
    return len([segment for segment in normalized.rstrip("/").split("/") if segment])


def names_a_location(normalized: str) -> bool:
    """True when the token states both a directory and what to find in it.

    Only such a token may be declared broken once every interpretation has
    failed: it named a location and that location is not there.  Two shapes are
    deliberately excluded:

    * a single-segment token (``run_wine_dual.py``, ``cora_clean/``) never says
      where it lives;
    * a multi-segment token whose last segment is neither a directory marker,
      nor a file name, nor a pattern -- ``1/sigma_Y2`` is a fraction, not a
      path, and prose of that shape must not be reported as a missing file.
    """
    if segment_count(normalized) < 2:
        return False
    if normalized.endswith("/"):
        return True
    last = normalized.rstrip("/").split("/")[-1]
    if any(char in GLOB_CHARACTERS for char in last):
        return True
    # Any dotted suffix counts here, not only a known extension: a reference to
    # `results/output.qqq` still names a file, and refusing to report it merely
    # because ".qqq" is unfamiliar would be a silent gap.
    return _extension_of(_strip_braces(last) or last) is not None


# --------------------------------------------------------------------------
# brace expansion
# --------------------------------------------------------------------------


class ExpansionError(Exception):
    """Raised when a brace group cannot be expanded within the configured bounds."""


def expand_braces(token: str, depth: int = 0) -> list[str]:
    """Expand ``{a,b}`` groups into concrete alternatives.

    Empty alternatives (``file_{,_agg}.csv``) and nesting are supported.  No
    shell is involved; expansion is bounded by :data:`MAX_BRACE_VARIANTS` and
    :data:`MAX_BRACE_DEPTH`.
    """
    if depth > MAX_BRACE_DEPTH:
        raise ExpansionError(f"brace nesting deeper than {MAX_BRACE_DEPTH}")

    start = token.find("{")
    if start < 0:
        return [token]

    level = 0
    end = -1
    for index in range(start, len(token)):
        if token[index] == "{":
            level += 1
        elif token[index] == "}":
            level -= 1
            if level == 0:
                end = index
                break
    if end < 0:
        raise ExpansionError("unbalanced brace group")

    prefix = token[:start]
    suffix = token[end + 1 :]
    alternatives = _split_alternatives(token[start + 1 : end])

    results: list[str] = []
    for alternative in alternatives:
        for expanded in expand_braces(prefix + alternative + suffix, depth + 1):
            results.append(expanded)
            if len(results) > MAX_BRACE_VARIANTS:
                raise ExpansionError(
                    f"more than {MAX_BRACE_VARIANTS} brace expansions"
                )
    return results


def _split_alternatives(body: str) -> list[str]:
    """Split brace-group alternatives on top-level commas.

    Each alternative is stripped of surrounding whitespace, because prose
    writes ``{a, b, c}`` where a shell would write ``{a,b,c}``.  Whitespace
    inside an alternative is preserved, so ``Mato Lab Program`` survives.
    """
    alternatives: list[str] = []
    level = 0
    current: list[str] = []
    for char in body:
        if char == "{":
            level += 1
        elif char == "}":
            level -= 1
        if char == "," and level == 0:
            alternatives.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    alternatives.append("".join(current).strip())
    return alternatives


# --------------------------------------------------------------------------
# working tree index
# --------------------------------------------------------------------------


class TreeIndex:
    """Sorted, symlink-free index of the files and directories under ``root``."""

    def __init__(self, root: Path, skipped: frozenset[str] = SKIPPED_TREE_DIRS) -> None:
        self.root = root
        self.skipped = skipped
        files: list[str] = []
        directories: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames if name not in skipped)
            relative_base = Path(dirpath).relative_to(root).as_posix()
            prefix = "" if relative_base == "." else relative_base + "/"
            for name in dirnames:
                directories.append(prefix + name)
            for name in sorted(filenames):
                files.append(prefix + name)
        self.files = sorted(files)
        self.directories = sorted(directories)
        self.all_paths = sorted(self.files + self.directories)
        self._file_set = frozenset(self.files)
        self._directory_set = frozenset(self.directories)
        self.root_entries = frozenset(
            path for path in self.all_paths if "/" not in path
        )
        self._by_basename: dict[str, list[str]] = {}
        for path in self.all_paths:
            self._by_basename.setdefault(path.rsplit("/", 1)[-1], []).append(path)

    def exists(self, path: str, directory_only: bool = False) -> bool:
        if directory_only:
            return path in self._directory_set
        return path in self._file_set or path in self._directory_set

    def paths_with_basename(self, basename: str) -> list[str]:
        return self._by_basename.get(basename, [])

    def match(self, pattern: str) -> list[str]:
        """Return every indexed path matching ``pattern``.

        ``*`` and ``?`` never cross a separator; ``**`` matches any number of
        segments.  A trailing ``/`` restricts the match to directories.
        """
        directory_only = pattern.endswith("/")
        body = pattern.rstrip("/")
        if not body:
            return []
        if not any(char in GLOB_CHARACTERS for char in body):
            return [body] if self.exists(body, directory_only) else []

        regex = re.compile(_glob_to_regex(body))
        haystack = self.directories if directory_only else self.all_paths
        return [path for path in haystack if regex.match(path)]


def _glob_to_regex(pattern: str) -> str:
    """Translate a glob pattern into an anchored regular expression."""
    out: list[str] = ["^"]
    index = 0
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "*":
            if pattern.startswith("**", index):
                index += 2
                if pattern.startswith("/", index):
                    # 'a/**/b' also matches 'a/b'
                    out.append("(?:.*/)?")
                    index += 1
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
            index += 1
            continue
        if char == "?":
            out.append("[^/]")
            index += 1
            continue
        out.append(re.escape(char))
        index += 1
    out.append("$")
    return "".join(out)


# --------------------------------------------------------------------------
# gitignore evidence (partial, evidence only)
# --------------------------------------------------------------------------


def read_gitignore_rules(root: Path) -> tuple[list[str], frozenset[str], bool]:
    """Read the two ``.gitignore`` forms this validator understands.

    Returns ``(extension_patterns, bare_directory_names, has_negation)``:

    * ``*.ext`` lines.  They annotate results, and they are the only rule form
      strong enough to support a LOCAL_ONLY_ARTIFACT verdict: Git applies such
      a line to a file of that suffix at any depth, which this module can
      decide for a given reference without consulting Git.
    * ``name/`` lines that carry no path separator of their own, such as
      ``.venv/`` or ``__pycache__/``.  Those directories are excluded from the
      working-tree index so that the index -- and therefore the verdict -- does
      not depend on machine-local scratch directories.
    * whether the file contains any ``!`` negation.  A negation can re-include
      a file that an earlier rule excluded, and this reader does not implement
      Git's ordering and anchoring semantics for that.  When one is present the
      whole LOCAL_ONLY_ARTIFACT determination is switched off rather than
      guessed at: an unimplemented rule form must not soften a verdict.

    Every other ``.gitignore`` form -- anchored directory prefixes such as
    ``expfam/results/raw/``, bare file names, ``.env.*`` -- is deliberately not
    interpreted here.  A reference those rules cover keeps whatever
    classification the tree gives it; this is a partial reader, not a
    Git-compatible one, and it only ever declines to act.
    """
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return [], frozenset(), False
    extensions: list[str] = []
    directories: set[str] = set()
    has_negation = False
    for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("!"):
            has_negation = True
            continue
        if re.fullmatch(r"\*\.[A-Za-z0-9_]+", stripped):
            extensions.append(stripped)
            continue
        if re.fullmatch(r"[A-Za-z0-9_.-]+/", stripped):
            directories.add(stripped[:-1])
    return sorted(set(extensions)), frozenset(directories), has_negation


# --------------------------------------------------------------------------
# validator
# --------------------------------------------------------------------------


@dataclass
class Report:
    root: str
    sources: list[str]
    candidates: list[Candidate]
    skipped_directories: list[str] = field(default_factory=list)
    indexed_paths: int = 0
    #: True when UNRESOLVED also makes the run fail, which changes what the
    #: coverage sentence and the JSON summary are allowed to claim.
    unresolved_is_blocking: bool = False

    @property
    def summary(self) -> dict[str, int]:
        counts = {item.value: 0 for item in CLASSIFICATION_ORDER}
        for candidate in self.candidates:
            counts[candidate.classification.value] += 1
        return counts

    @property
    def inferred_context_count(self) -> int:
        """Entries located only against a base inferred from context.

        They are resolutions of an interpretation, not of the token as written,
        so they are reported separately instead of being folded into a single
        "resolved" number.
        """
        return len(self.inferred_context_candidates)

    @property
    def inferred_context_candidates(self) -> list[Candidate]:
        return [
            candidate
            for candidate in self.candidates
            if candidate.resolved_via == "inferred-context"
        ]

    @property
    def unverified_counts(self) -> dict[str, int]:
        """Candidates that were neither confirmed to exist nor reported broken.

        Whatever the exit-status policy, these are the ones a human still has
        to look at: UNRESOLVED could not be located, NON_PATH was judged not to
        be a path at all, INTENTIONAL_HISTORICAL is a documented absence,
        KNOWN_NOTATION_DEFECT is a documented defect superseded by a forward
        correction, and LOCAL_ONLY_ARTIFACT is outside Git and therefore not
        evidence of existence even when it is present here.
        """
        summary = self.summary
        return {
            item.value: summary[item.value]
            for item in (
                Classification.UNRESOLVED,
                Classification.NON_PATH,
                Classification.INTENTIONAL_HISTORICAL,
                Classification.KNOWN_NOTATION_DEFECT,
                Classification.LOCAL_ONLY_ARTIFACT,
            )
        }

    @property
    def unchecked_count(self) -> int:
        """How many of those the exit status does not act on, under this policy."""
        counts = self.unverified_counts
        if self.unresolved_is_blocking:
            counts = dict(counts)
            counts.pop(Classification.UNRESOLVED.value, None)
        return sum(counts.values())

    @property
    def gitignore_sensitive_candidates(self) -> list[Candidate]:
        """Candidates whose verdict could change in a fresh checkout."""
        return [c for c in self.candidates if c.gitignore_sensitive]

    def by_classification(self, classification: Classification) -> list[Candidate]:
        return [c for c in self.candidates if c.classification is classification]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            # Absolute and therefore machine-specific: a byte-for-byte golden
            # comparison across checkouts has to exclude this field.
            "root": self.root,
            "sources": list(self.sources),
            "skipped_directories": list(self.skipped_directories),
            "indexed_paths": self.indexed_paths,
            "unresolved_is_blocking": self.unresolved_is_blocking,
            "summary": {
                "candidates": len(self.candidates),
                **self.summary,
                "local_only_present": sum(
                    1
                    for c in self.candidates
                    if c.local_presence == "present"
                ),
                "local_only_absent": sum(
                    1 for c in self.candidates if c.local_presence == "absent"
                ),
                "non_blocking_by_policy": sum(
                    self.summary[item.value] for item in NON_BLOCKING_BY_POLICY
                ),
                "resolved_via_inferred_context": self.inferred_context_count,
                "not_covered_by_exit_status": self.unchecked_count,
                "verdict_depends_on_gitignored_artifacts": len(
                    self.gitignore_sensitive_candidates
                ),
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class Validator:
    """Classify the path references of a set of Markdown provenance documents."""

    def __init__(
        self,
        root: Path,
        local_only_references: dict[
            tuple[str, str], LocalOnlyArtifactReference
        ] | None = None,
    ) -> None:
        self.root = root
        #: Injectable so the fixture suites can register their own references
        #: instead of mutating module state.  Production always uses the
        #: module-level mapping.
        self.local_only_references = (
            LOCAL_ONLY_ARTIFACT_REFERENCES
            if local_only_references is None
            else local_only_references
        )
        (
            self.ignored_extension_patterns,
            ignored_directories,
            gitignore_has_negation,
        ) = read_gitignore_rules(root)
        #: False disables every LOCAL_ONLY_ARTIFACT verdict.  See
        #: read_gitignore_rules(): a negation makes this reader's view of
        #: .gitignore incomplete, and an incomplete view may not excuse a
        #: reference from the ordinary rules.
        self.local_only_supported = not gitignore_has_negation
        self.skipped_directories: frozenset[str] = SKIPPED_TREE_DIRS | ignored_directories
        self.index = TreeIndex(root, self.skipped_directories)

    # -- resolution helpers -------------------------------------------------

    def _resolve_variants(self, variants: Sequence[str]) -> tuple[list[str], list[str]]:
        """Return ``(matches, unmatched_variants)`` for already expanded variants."""
        matches: list[str] = []
        unmatched: list[str] = []
        for variant in variants:
            found = self.index.match(variant)
            if found:
                matches.extend(found)
            else:
                unmatched.append(variant)
        return sorted(set(matches)), unmatched

    def _sibling_base(
        self, candidate: Candidate, earlier: Sequence[Candidate]
    ) -> str | None:
        """Directory prefix implied by an earlier reference in the same table cell.

        The registry writes sibling references as a suffix of the same depth as
        the preceding full path (``expfam/results/a.csv``, then ``b.npy``).  The
        base is therefore the preceding path minus as many trailing segments as
        the current token has.
        """
        if candidate.cell is None:
            return None
        depth = segment_count(candidate.normalized)
        for previous in reversed(earlier):
            if (
                previous.source != candidate.source
                or previous.line != candidate.line
                or previous.cell != candidate.cell
            ):
                continue
            if previous.classification not in (
                Classification.EXISTS_LITERAL,
                Classification.PATTERN_RESOLVED,
            ):
                continue
            # Use the form the previous token actually resolved as, so that a
            # chain of sibling references keeps a repository-relative base.
            resolved_as = previous.variants[0] if previous.variants else previous.normalized
            segments = [s for s in resolved_as.rstrip("/").split("/") if s]
            if len(segments) <= depth:
                continue
            prefix = segments[: len(segments) - depth]
            if any(char in part for part in prefix for char in "*?{}"):
                # A patterned prefix cannot serve as a concrete base directory.
                continue
            return "/".join(prefix)
        return None

    def _suffix_matches(self, variant: str) -> list[str] | None:
        """Tree paths that end with ``variant`` as a whole-segment suffix.

        A multi-segment tail is a much stronger signal than a shared basename:
        when the tree contains it under some other parent, the reference was
        most likely written relative to a base the document does not state.
        That is reported as evidence and keeps the candidate UNRESOLVED; it is
        never treated as a resolution.

        Two guards keep this from becoming a false-negative hole:

        * a token whose first segment names an existing repository-root entry
          is a root-relative path, so it is broken rather than abbreviated and
          is never downgraded -- ``figures/real_data/wine_clean/`` stays
          TRUE_BROKEN even though ``expfam/figures/real_data/wine_clean``
          exists;
        * patterns are excluded, so a broken wildcard still ends up
          TRUE_BROKEN.

        The number of tail matches deliberately does not enter the decision.
        Letting it decide would make a more ambiguous reference blocking and a
        less ambiguous one not, on an accident of the tree.
        """
        if any(char in variant for char in "*?{}"):
            return None
        body = variant.rstrip("/")
        segments = [s for s in body.split("/") if s]
        if len(segments) < 2:
            return None
        if segments[0] in self.index.root_entries:
            return None
        needle = "/" + body
        found = [path for path in self.index.all_paths if path.endswith(needle)]
        return found or None

    def _basename_evidence(self, candidate: Candidate) -> list[str]:
        """Non-authoritative hint listing tree paths with the same basename."""
        body = candidate.normalized.rstrip("/")
        basename = body.rsplit("/", 1)[-1]
        if not basename or any(char in basename for char in "*?{}"):
            return []
        found = self.index.paths_with_basename(basename)
        if not found:
            return []
        shown = ", ".join(found[:MAX_BASENAME_EVIDENCE])
        more = "" if len(found) <= MAX_BASENAME_EVIDENCE else ", ..."
        return [
            f"informational only (not used for classification): {len(found)} "
            f"tree path(s) share the basename {basename!r}: {shown}{more}"
        ]

    def _ignored_extension_evidence(self, paths: Sequence[str]) -> list[str]:
        """Warn when a candidate depends on a file type that .gitignore excludes.

        Emitted for resolved matches AND for unresolved variants, because the
        fresh-checkout caveat matters most exactly when nothing was found.
        """
        hits = sorted(
            {
                pattern
                for pattern in self.ignored_extension_patterns
                for path in paths
                if path.endswith(pattern[1:])
            }
        )
        if not hits:
            return []
        return [
            "this reference depends on a file type excluded by .gitignore "
            f"({', '.join(hits)}); such artifacts may be present locally and "
            "absent from a fresh checkout, so the verdict can differ in CI"
        ]

    # -- local-only (gitignored) evidence -----------------------------------

    def _form_excluded_by(self, form: str, rule: str) -> bool:
        """True when ``rule`` excludes every path ``form`` can denote.

        Used to validate a registered entry, never to discover one.  ``form``
        is one brace-expanded variant and may still contain ``*``, ``?`` or
        ``**``.  The test is structural, so it answers the same on a
        workstation and in a fresh checkout:

        * a directory reference (trailing ``/``) is refused -- a directory
          holds whatever it holds, and nothing licenses a claim about its
          contents;
        * the reference must name a location, not a bare basename, so that the
          recorded path is reviewable;
        * the final segment must end in the rule's suffix *literally*, with no
          wildcard inside the suffix itself, so that every path the glob can
          match carries that suffix.  ``*.npy`` qualifies, ``*.np?`` does not;
        * ``**`` may cross separators, which changes only which directory a
          match lands in, never its suffix.
        """
        if form.endswith("/") or segment_count(form) < 2:
            return False
        suffix = rule[1:]
        last = form.rsplit("/", 1)[-1]
        if len(last) < len(suffix) or not last.endswith(suffix):
            return False
        return not any(char in GLOB_CHARACTERS for char in last[-len(suffix) :])

    def _expected_variants(
        self, reference: LocalOnlyArtifactReference
    ) -> list[str] | None:
        """The registered expected forms, or ``None`` if the entry is not usable.

        This is where a registration is checked rather than trusted.  It fails
        closed on every count: an unparsable expected form, a ``.gitignore``
        that no longer carries the recorded rule, a rule this reader cannot
        evaluate exactly, or an expected form that rule would not actually
        exclude.  A registration that stops being supported by the repository
        therefore stops applying, instead of going on silently excusing a
        reference.
        """
        if not self.local_only_supported:
            return None
        if reference.ignore_rule not in self.ignored_extension_patterns:
            return None
        try:
            variants = sorted(set(expand_braces(normalize_token(reference.expected))))
        except ExpansionError:
            return None
        if not variants:
            return None
        if not all(
            self._form_excluded_by(variant, reference.ignore_rule)
            for variant in variants
        ):
            return None
        return variants

    def _local_only_reference(
        self, candidate: Candidate
    ) -> tuple[LocalOnlyArtifactReference, list[str]] | None:
        """The registered entry for this candidate, with its expected forms.

        Returns ``None`` when the candidate is not registered or the entry is
        no longer authorized.  Whether the *forms under consideration* equal
        the expected ones is decided by the caller, because that depends on
        which interpretation of the token is being tried.
        """
        reference = self.local_only_references.get((candidate.source, candidate.raw))
        if reference is None:
            return None
        variants = self._expected_variants(reference)
        if variants is None:
            return None
        return reference, variants

    def _mark_local_only(
        self,
        candidate: Candidate,
        reference: LocalOnlyArtifactReference,
        matches: Sequence[str],
        unmatched: Sequence[str],
        resolved_via: str,
    ) -> None:
        candidate.classification = Classification.LOCAL_ONLY_ARTIFACT
        candidate.local_presence = "present" if matches else "absent"
        candidate.resolved_via = resolved_via if matches else ""
        candidate.correction = reference.expected
        how = (
            "as written"
            if resolved_via == "root"
            else "after rebasing onto the base implied by the preceding "
            "reference in the same table cell"
        )
        candidate.reason = (
            "explicitly registered as a local-only research artifact: the "
            f"reference matches the registered entry {how}, and its recorded "
            f"path {reference.expected!r} is excluded from Git by the "
            f"{reference.ignore_rule} rule this run read from .gitignore. Such "
            "an artifact is present on a workstation that produced it and "
            "legitimately absent from a fresh checkout; in this working tree "
            f"it is {candidate.local_presence}. Not a statement that the "
            "artifact exists."
        )
        candidate.evidence.append(f"registration reason: {reference.reason}")
        self._record_matches(candidate, matches, unmatched)

    def _correction_resolves(self, correction: str) -> bool:
        """True when every variant of a recorded forward correction is present.

        A waiver is only as good as the reference that replaces it, so this is
        re-checked on every run instead of being trusted from the table.
        """
        try:
            variants = expand_braces(normalize_token(correction))
        except ExpansionError:
            return False
        _, unmatched = self._resolve_variants(variants)
        return not unmatched

    # -- classification -----------------------------------------------------

    def classify(self, candidate: Candidate, earlier: Sequence[Candidate]) -> None:
        candidate.normalized = normalize_token(candidate.raw)
        path_like, reason = is_path_like(candidate.normalized)
        candidate.path_like = path_like
        if not path_like:
            candidate.classification = Classification.NON_PATH
            candidate.reason = reason
            return

        ambiguity = find_ambiguity(candidate.normalized)
        if ambiguity is not None:
            candidate.classification = Classification.UNRESOLVED
            candidate.reason = ambiguity
            candidate.evidence.extend(self._basename_evidence(candidate))
            return

        alternation = interpret_extension_alternation(candidate.normalized)
        base_forms = alternation if alternation is not None else [candidate.normalized]
        if alternation is not None:
            candidate.evidence.append(
                "read as extension-alternation notation; both "
                f"{alternation[0]!r} and {alternation[1]!r} must resolve"
            )

        try:
            expanded: list[str] = []
            for form in base_forms:
                expanded.extend(expand_braces(form))
            candidate.variants = sorted(set(expanded))
        except ExpansionError as error:
            candidate.classification = Classification.UNRESOLVED
            candidate.reason = f"brace expansion refused: {error}"
            return

        has_pattern = alternation is not None or any(
            char in GLOB_CHARACTERS or char == "{" for char in candidate.normalized
        )

        matches, unmatched = self._resolve_variants(candidate.variants)

        # Looked up before the tree is consulted for a verdict, so that a
        # registered reference gets the same classification whether or not
        # this particular machine happens to hold the artifact.  Registration
        # is the authorization; the expected forms must match what is actually
        # being resolved, or the ordinary rules apply.
        registered = self._local_only_reference(candidate)
        if registered is not None and candidate.variants == registered[1]:
            self._mark_local_only(
                candidate, registered[0], matches, unmatched, "root"
            )
            return

        if not unmatched:
            candidate.classification = (
                Classification.PATTERN_RESOLVED
                if has_pattern
                else Classification.EXISTS_LITERAL
            )
            candidate.resolved_via = "root"
            candidate.reason = (
                "resolved relative to the repository root"
                if alternation is None
                else "resolved relative to the repository root after reading the "
                "token as extension-alternation notation"
            )
            self._record_matches(candidate, matches)
            return

        defect = KNOWN_NOTATION_DEFECTS.get((candidate.source, candidate.raw))
        if defect is not None and self._correction_resolves(defect.correction):
            candidate.classification = Classification.KNOWN_NOTATION_DEFECT
            candidate.correction = defect.correction
            candidate.reason = defect.reason
            candidate.evidence.append(defect.evidence)
            candidate.evidence.append(
                f"superseded by {defect.correction!r}, which this run resolved "
                "against the working tree; the waiver lapses if it stops "
                "resolving"
            )
            self._record_matches(candidate, matches, unmatched)
            return

        locatable = names_a_location(base_forms[0])
        sibling_tried = False

        if not matches:
            base = self._sibling_base(candidate, earlier)
            sibling_tried = base is not None
            if base is not None:
                rebased = [f"{base}/{variant}" for variant in candidate.variants]
                rebased_matches, rebased_unmatched = self._resolve_variants(rebased)
                # A registration may record the rebased form, so that a bare
                # basename the document writes as a sibling is stable across
                # environments.  Condition: the rebasing must land exactly on
                # the registered expected form -- the entry is never allowed
                # to follow the token onto some other path.
                if (
                    registered is not None
                    and sorted(set(rebased)) == registered[1]
                ):
                    candidate.variants = rebased
                    self._mark_local_only(
                        candidate,
                        registered[0],
                        rebased_matches,
                        rebased_unmatched,
                        "inferred-context",
                    )
                    return
                if not rebased_unmatched:
                    # Always PATTERN_RESOLVED, even for a token that carries no
                    # wildcard: what resolved is an interpretation of the token,
                    # not the token as written, and that distinction is what the
                    # reviewer needs to see.
                    candidate.classification = Classification.PATTERN_RESOLVED
                    candidate.resolved_via = "inferred-context"
                    candidate.reason = (
                        "not resolvable from the repository root, but resolved "
                        f"against the base {base!r} implied by the preceding "
                        "reference in the same table cell"
                    )
                    candidate.variants = rebased
                    self._record_matches(candidate, rebased_matches)
                    return
                candidate.evidence.append(
                    f"base {base!r} implied by the same table cell did not resolve "
                    f"{', '.join(repr(v) for v in rebased_unmatched)}"
                )

            historical = INTENTIONAL_HISTORICAL_PATHS.get(
                candidate.normalized.rstrip("/")
            )
            if historical is not None:
                candidate.classification = Classification.INTENTIONAL_HISTORICAL
                candidate.reason = historical.reason
                candidate.evidence.append(historical.evidence)
                return

        if locatable:
            excluded = sorted(
                {
                    segment
                    for variant in unmatched
                    for segment in _strip_braces(variant).split("/")
                    if segment in self.skipped_directories
                }
            )
            if excluded:
                candidate.classification = Classification.UNRESOLVED
                candidate.reason = (
                    "the path lies inside a directory that is excluded from the "
                    f"working-tree index ({', '.join(excluded)}), so its presence "
                    "cannot be decided"
                )
                self._record_matches(candidate, matches, unmatched)
                return

            relocated = {
                variant: found
                for variant in unmatched
                if (found := self._suffix_matches(variant)) is not None
            }
            if len(relocated) == len(unmatched):
                candidate.classification = Classification.UNRESOLVED
                candidate.reason = (
                    "the path as written does not exist, its first segment is "
                    "not a repository-root entry, and the tree does contain the "
                    "same multi-segment tail elsewhere, so the reference is most "
                    "likely written relative to a base the document never states"
                )
                for variant, found in sorted(relocated.items()):
                    shown = ", ".join(repr(path) for path in found[:MAX_BASENAME_EVIDENCE])
                    more = "" if len(found) <= MAX_BASENAME_EVIDENCE else ", ..."
                    candidate.evidence.append(
                        f"{variant!r} occurs in the tree as {shown}{more}"
                    )
                self._record_matches(candidate, matches, unmatched)
                return

            # Say only what was actually attempted.  The same-cell base is
            # tried solely when nothing at all matched, so a partially matching
            # brace group must not claim that interpretation was exhausted.
            if sibling_tried:
                attempted = (
                    "neither relative to the repository root nor against the "
                    "base implied by the same table cell"
                )
            elif matches:
                attempted = (
                    "relative to the repository root, where the remaining "
                    "variant(s) of the same token do match, so no other "
                    "interpretation was attempted"
                )
            elif candidate.cell is None:
                attempted = (
                    "relative to the repository root, and the token is not in a "
                    "table cell that could imply another base"
                )
            else:
                attempted = (
                    "relative to the repository root, and the same table cell "
                    "offered no base to try instead"
                )
            candidate.classification = Classification.TRUE_BROKEN
            candidate.reason = (
                "the token names a location, but "
                f"{len(unmatched)} of {len(candidate.variants)} variant(s) match "
                f"nothing {attempted}: "
                f"{', '.join(repr(v) for v in unmatched)}"
            )
            self._record_matches(candidate, matches, unmatched)
            return

        candidate.classification = Classification.UNRESOLVED
        candidate.reason = (
            "the token does not name a location: it is a single segment, so the "
            "document never says where it lives"
            if segment_count(base_forms[0]) < 2
            else "the token does not name a location: its last segment is neither "
            "a directory marker, nor a file name, nor a pattern, so it cannot be "
            "read as a path reliably"
        )
        self._record_matches(candidate, matches, unmatched)
        candidate.evidence.extend(self._basename_evidence(candidate))

    def _record_matches(
        self,
        candidate: Candidate,
        matches: Sequence[str],
        unmatched: Sequence[str] = (),
    ) -> None:
        candidate.matches = list(matches)
        candidate.match_count = len(matches)
        notes = self._ignored_extension_evidence([*matches, *unmatched])
        candidate.gitignore_sensitive = bool(notes)
        candidate.evidence.extend(notes)

    # -- driver -------------------------------------------------------------

    def run(self, sources: Sequence[Path]) -> Report:
        candidates: list[Candidate] = []
        source_names: list[str] = []
        for source in sources:
            relative = source.relative_to(self.root).as_posix()
            source_names.append(relative)
            text = source.read_text(encoding="utf-8")
            for candidate in extract_candidates(relative, text):
                self.classify(candidate, candidates)
                candidates.append(candidate)
        candidates.sort(key=lambda c: (c.source, c.line, c.column, c.raw))
        return Report(
            root=self.root.as_posix(),
            sources=source_names,
            candidates=candidates,
            skipped_directories=sorted(self.skipped_directories),
            indexed_paths=len(self.index.all_paths),
        )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def render_text(report: Report, verbose: bool) -> str:
    lines: list[str] = []
    lines.append("registry path validator (read-only)")
    lines.append(f"root    : {report.root}")
    lines.append(f"sources : {', '.join(report.sources)}")
    if report.skipped_directories:
        lines.append(f"skipped : {', '.join(report.skipped_directories)}")
    lines.append(f"indexed : {report.indexed_paths} paths")
    lines.append("")
    lines.append(f"candidates : {len(report.candidates)}")
    summary = report.summary
    width = max(len(item.value) for item in CLASSIFICATION_ORDER)
    for item in CLASSIFICATION_ORDER:
        lines.append(f"  {item.value.ljust(width)} : {summary[item.value]}")
    lines.append("")
    lines.append(
        f"{report.inferred_context_count} candidate(s) were located only against a "
        "base directory inferred from the surrounding table cell, rather than "
        "resolving as written."
    )
    local_only = report.by_classification(Classification.LOCAL_ONLY_ARTIFACT)
    present = sum(1 for c in local_only if c.local_presence == "present")
    lines.append(
        f"{len(local_only)} LOCAL_ONLY_ARTIFACT candidate(s): explicitly "
        f"registered references to Git-excluded artifacts, {present} present in "
        f"this tree and {len(local_only) - present} absent. The classification "
        "is the same in either case, so it does not flip between a workstation "
        "and CI; the presence-derived fields (local_presence, matches, "
        "resolved_via) do differ. It is never a claim that the artifact exists. "
        "Only a registered reference can land here: an unregistered missing "
        "path is TRUE_BROKEN whatever its extension."
    )
    defects = report.by_classification(Classification.KNOWN_NOTATION_DEFECT)
    lines.append(
        f"{len(defects)} KNOWN_NOTATION_DEFECT candidate(s): historical notation "
        "preserved as written, superseded by a forward correction that resolved "
        "on this run."
    )
    unverified = report.unverified_counts
    breakdown = ", ".join(f"{name} {count}" for name, count in unverified.items())
    lines.append(
        f"{sum(unverified.values())} candidate(s) were neither confirmed to exist "
        f"nor reported broken ({breakdown})."
    )
    policy = (
        "TRUE_BROKEN and UNRESOLVED both fail this run (--fail-on-unresolved)"
        if report.unresolved_is_blocking
        else "TRUE_BROKEN alone fails this run"
    )
    lines.append(
        "non-blocking by policy: "
        + ", ".join(item.value for item in NON_BLOCKING_BY_POLICY)
        + "; each is listed individually below and counted on its own line."
    )
    lines.append(
        f"policy: {policy}; {report.unchecked_count} of those candidate(s) are "
        "outside the exit status and need human review. A zero exit status never "
        "means every reference was verified."
    )
    sensitive = report.gitignore_sensitive_candidates
    lines.append(
        f"{len(sensitive)} verdict(s) depend on artifacts excluded by .gitignore "
        "and may differ in a fresh checkout"
        + (
            ": " + ", ".join(f"{c.source}:{c.line} `{c.raw}`" for c in sensitive)
            if sensitive
            else ""
        )
        + "."
    )
    lines.append("")

    for classification in PROBLEM_CLASSIFICATIONS:
        group = report.by_classification(classification)
        lines.append(f"{classification.value} ({len(group)})")
        if not group:
            lines.append("  (none)")
        for candidate in group:
            lines.extend(_render_candidate(candidate))
        lines.append("")

    inferred = report.inferred_context_candidates
    lines.append(f"located via inferred context ({len(inferred)})")
    if not inferred:
        lines.append("  (none)")
    for candidate in inferred:
        lines.extend(_render_candidate(candidate))
    lines.append("")

    if verbose:
        lines.append(f"all candidates ({len(report.candidates)})")
        for candidate in report.candidates:
            lines.extend(_render_candidate(candidate))
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_candidate(candidate: Candidate) -> list[str]:
    lines = [f"  {candidate.location}  `{candidate.raw}`"]
    lines.append(f"      classification : {candidate.classification.value}")
    if candidate.local_presence:
        lines.append(f"      local presence : {candidate.local_presence}")
    if candidate.correction:
        lines.append(f"      correction     : {candidate.correction}")
    lines.append(f"      normalized     : {candidate.normalized}")
    lines.append(f"      reason         : {candidate.reason}")
    if candidate.variants and candidate.variants != [candidate.normalized]:
        lines.append(f"      variants       : {', '.join(candidate.variants)}")
    lines.append(f"      matches        : {candidate.match_count}")
    for match in candidate.matches[:MAX_REPRESENTATIVE_MATCHES]:
        lines.append(f"        - {match}")
    if candidate.match_count > MAX_REPRESENTATIVE_MATCHES:
        lines.append(
            f"        ... {candidate.match_count - MAX_REPRESENTATIVE_MATCHES} more"
        )
    for item in candidate.evidence:
        lines.append(f"      evidence       : {item}")
    return lines


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfTestCase:
    name: str
    token: str
    expected: Classification


def _build_fixture_tree(root: Path) -> None:
    """Create the throwaway tree used by ``--self-test`` (outside the repository)."""
    _refuse_outside_fixture(root)
    files = [
        "expfam/results/exp_scenario_A_exp1_k.csv",
        "expfam/results/exp_scenario_B_exp1_k.csv",
        "expfam/results/fig_scenario_A_exp1_k.pdf",
        "expfam/results/fig_scenario_A_exp1_k.png",
        "expfam/results/exp1_full_A.csv",
        "expfam/results/exp1_full_B.csv",
        "expfam/results/exp1_full_C.csv",
        "expfam/results/wine_F.npy",
        "expfam/results/wine_dual_results.csv",
        "expfam/results/real_data/wine_clean/summary.csv",
        "expfam/results/real_data/cora_clean/summary.csv",
        # Mirrors the real repository, which has both a top-level figures/ tree
        # and an expfam/figures/ tree.  Needed so that the root-entry guard on
        # the unstated-base downgrade is actually exercised.
        "expfam/figures/real_data/wine_clean/plot.png",
        "figures/fig1a_n_sweep_color.pdf",
        "figures/fig1a_n_sweep_color.png",
        "archive/notion_scripts/post.py",
        "Mato Lab Program/calcEtaNewton.m",
        "Mato Lab Program/EffcalcEtaNewton (1).m",
        "reports/plan.md",
        "expfam/data/movielens_pilot/movielens_movies_metadata.csv",
        "expfam/data/movielens_pilot/movielens_X_genre.npy",
        "expfam/results/story_diagnostics/y_sparsity_stress_20260713.csv",
        "expfam/results/story_diagnostics/y_sparsity_stress_20260713_agg.csv",
        "expfam/results/story_diagnostics/y_sparsity_stress_20260713_runinfo.csv",
        "reproduction/results/raw/raw_output.csv",
        # Present, genuinely ignored, and deliberately unregistered: each is a
        # control proving that an ignored suffix alone authorizes nothing.
        "expfam/results/run_trace.log",
        "expfam/results/scratch.tmp",
        "expfam/results/state.pkl",
        "expfam/results/state.pickle",
        "expfam/results/bundle.npz",
        ".venv/lib/exp_scenario_A_exp1_k.csv",
    ]
    for relative in files:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        chr(10).join(
            [
                "*.npy", "*.npz", "*.pkl", "*.pickle", "*.log", "*.tmp",
                ".venv/", "__pycache__/", "reproduction/results/raw/",
            ]
        )
        + chr(10),
        encoding="utf-8",
    )


#: Registrations used by the fixture suites.  They exist so the fixture can
#: exercise the registered path without mutating module state; every token the
#: fixture checks that is NOT listed here must fall back to ordinary semantics,
#: which is what most of the local-only checks below are actually testing.
FIXTURE_LOCAL_ONLY_REFERENCES: dict[
    tuple[str, str], LocalOnlyArtifactReference
] = {
    reference.key: reference
    for reference in (
        LocalOnlyArtifactReference(
            source="<self-test>",
            raw="expfam/results/wine_F.npy",
            expected="expfam/results/wine_F.npy",
            ignore_rule="*.npy",
            reason="fixture: explicit path form",
        ),
        LocalOnlyArtifactReference(
            source="<self-test>",
            raw="expfam/results/*.npy",
            expected="expfam/results/*.npy",
            ignore_rule="*.npy",
            reason="fixture: wildcard form",
        ),
        LocalOnlyArtifactReference(
            source="<self-test>",
            raw="expfam/data/movielens_pilot/*.npy",
            expected="expfam/data/movielens_pilot/*.npy",
            ignore_rule="*.npy",
            reason="fixture: wildcard form in a second directory",
        ),
        LocalOnlyArtifactReference(
            source="<self-test>",
            raw="wine_F.npy",
            expected="expfam/results/wine_F.npy",
            ignore_rule="*.npy",
            reason="fixture: bare basename resolved through a sibling base",
        ),
    )
}


SELF_TEST_CASES: tuple[SelfTestCase, ...] = (
    SelfTestCase(
        "existing literal file",
        "expfam/results/exp_scenario_A_exp1_k.csv",
        Classification.EXISTS_LITERAL,
    ),
    SelfTestCase(
        "existing directory",
        "expfam/results/real_data/wine_clean/",
        Classification.EXISTS_LITERAL,
    ),
    SelfTestCase("existing top-level directory", "figures/", Classification.EXISTS_LITERAL),
    SelfTestCase(
        "missing literal path, root anchored",
        "expfam/results/exp_scenario_Z_exp1_k.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "typo in a middle directory",
        "expfam/reslts/exp_scenario_A_exp1_k.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "wildcard single match",
        "expfam/results/wine_dual_*.csv",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "wildcard multiple matches",
        "expfam/results/fig_scenario_A_exp1_k.*",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "wildcard zero match",
        "expfam/results/fig_scenario_Q_*.png",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "typo in a wildcard prefix",
        "expfam/results/figg_scenario_A_exp1_k.*",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "recursive wildcard",
        "expfam/results/real_data/**",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "recursive wildcard, missing parent",
        "expfam/nowhere/**",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "brace, all alternatives resolve",
        "expfam/results/exp1_full_{A,B,C}.csv",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "brace, one alternative missing",
        "expfam/results/exp1_full_{A,B,D}.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "brace with empty alternative, all resolve",
        "expfam/results/exp1_full_{A,B}{,}.csv",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "brace with empty alternative, one missing",
        "expfam/results/exp_scenario_A_exp1_k{,_missing}.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "brace combined with wildcard",
        "expfam/results/real_data/{wine,cora}_clean/*.csv",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "nested brace",
        "expfam/results/exp1_full_{A,{B,C}}.csv",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "bare brace group without parent",
        "{dry_run,full,scenario_c_extra}",
        Classification.UNRESOLVED,
    ),
    SelfTestCase(
        "windows separators, existing",
        "expfam\\results\\exp_scenario_A_exp1_k.csv",
        Classification.EXISTS_LITERAL,
    ),
    SelfTestCase(
        "windows separators, missing",
        "expfam\\results\\nope.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "path containing spaces, existing",
        "Mato Lab Program/calcEtaNewton.m",
        Classification.EXISTS_LITERAL,
    ),
    SelfTestCase(
        "path containing spaces, missing",
        "Mato Lab Program/calcNothing.m",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase("code identifier", "DualExpFamLSM", Classification.NON_PATH),
    SelfTestCase("function identifier", "calc_bic_dual", Classification.NON_PATH),
    SelfTestCase("function call notation", "fig_movielens_mixed_x()", Classification.NON_PATH),
    SelfTestCase("line reference", "L.159", Classification.NON_PATH),
    SelfTestCase("symbol", "Q_strict", Classification.NON_PATH),
    SelfTestCase("status label", "current_main", Classification.NON_PATH),
    SelfTestCase("directory-looking bare word", "__pycache__", Classification.NON_PATH),
    SelfTestCase(
        "intentional historical exception",
        "archive/paper_writing_examples/",
        Classification.INTENTIONAL_HISTORICAL,
    ),
    SelfTestCase(
        "historical near miss, singular",
        "archive/paper_writing_example/",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "historical near miss, suffixed",
        "archive/paper_writing_examples_typo/",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "historical near miss, glob below the exception",
        "archive/paper_writing_examples/*.pdf",
        Classification.TRUE_BROKEN,
    ),
    # Both were EXISTS_LITERAL / PATTERN_RESOLVED before issue #19.  The
    # reference is unchanged and still must not be reported broken; what
    # changed is that presence in this tree is no longer reported as verified
    # existence, because Git does not carry the artifact.
    SelfTestCase(
        "ignored but existing artifact",
        "expfam/results/wine_F.npy",
        Classification.LOCAL_ONLY_ARTIFACT,
    ),
    SelfTestCase(
        "ignored artifact wildcard",
        "expfam/results/*.npy",
        Classification.LOCAL_ONLY_ARTIFACT,
    ),
    # -- issue #19 Finding B: local-only artifacts ------------------------
    SelfTestCase(
        # Unregistered.  An ignored suffix plus an existing parent directory
        # is NOT authorization; this is the case that must never be excused.
        "B6 unregistered ignored artifact, absent, is blocking",
        "expfam/results/absent_artifact.npy",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "B6 unregistered ignored artifact, mistyped registered name",
        "expfam/results/wine_typoooo.npy",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "B6 unregistered ignored artifact, plausible sibling name",
        "expfam/results/wine_G.npy",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "B6 near miss of a registered wildcard is not covered",
        "expfam/results/*.npz",
        Classification.PATTERN_RESOLVED,
    ),
    # One control per remaining ignored suffix: present -> ordinary existence,
    # absent -> ordinary blocking.  None of them may become local-only.
    SelfTestCase("suffix control: .log present", "expfam/results/run_trace.log",
                 Classification.EXISTS_LITERAL),
    SelfTestCase("suffix control: .log absent", "expfam/results/absent.log",
                 Classification.TRUE_BROKEN),
    SelfTestCase("suffix control: .tmp present", "expfam/results/scratch.tmp",
                 Classification.EXISTS_LITERAL),
    SelfTestCase("suffix control: .tmp absent", "expfam/results/absent.tmp",
                 Classification.TRUE_BROKEN),
    SelfTestCase("suffix control: .pkl present", "expfam/results/state.pkl",
                 Classification.EXISTS_LITERAL),
    SelfTestCase("suffix control: .pkl absent", "expfam/results/absent.pkl",
                 Classification.TRUE_BROKEN),
    SelfTestCase("suffix control: .pickle present", "expfam/results/state.pickle",
                 Classification.EXISTS_LITERAL),
    SelfTestCase("suffix control: .pickle absent", "expfam/results/absent.pickle",
                 Classification.TRUE_BROKEN),
    SelfTestCase("suffix control: .npz present", "expfam/results/bundle.npz",
                 Classification.EXISTS_LITERAL),
    SelfTestCase("suffix control: .npz absent", "expfam/results/absent.npz",
                 Classification.TRUE_BROKEN),
    SelfTestCase(
        "B3 missing non-ignored path is still blocking",
        "expfam/results/absent_artifact.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "B4 known extension that .gitignore does not cover",
        "expfam/results/absent_artifact.mat",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "B5 anchored .gitignore rule is not a supported local-only form",
        "reproduction/results/raw/absent.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "B6 wildcard over an ignored suffix",
        "expfam/data/movielens_pilot/*.npy",
        Classification.LOCAL_ONLY_ARTIFACT,
    ),
    SelfTestCase(
        "B6 wildcard over a non-ignored suffix in the same directory",
        "expfam/data/movielens_pilot/*.csv",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "B6 recursive wildcard is not local-only",
        "expfam/data/movielens_pilot/**",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "B6 directory holding ignored files is not local-only",
        "expfam/data/movielens_pilot/",
        Classification.EXISTS_LITERAL,
    ),
    SelfTestCase(
        "adversarial: typo in the directory of an ignored artifact",
        "expfam/reslts/wine_F.npy",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "adversarial: ignored artifact in a directory that does not exist",
        "expfam/nowhere/wine_F.npy",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        # Not registered, so the local-only lookup returns nothing before any
        # suffix or head analysis happens, and the ordinary verdict stands.
        # (Under the pre-review design this shape needed a dedicated guard;
        # registration subsumes it.)
        "adversarial: fully globbed head is not local-only, resolves normally",
        "**/wine_F.npy",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "adversarial: fully globbed head is not local-only, blocking when missing",
        "**/absent_thing.npy",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        # Plausible-looking and genuinely ignored, but not registered, so it
        # gets the ordinary verdict like anything else.
        "unregistered recursive ignored wildcard is not local-only",
        "expfam/**/*.npy",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        # Not registered, so it is never local-only.  The wildcarded suffix is
        # also the shape _form_excluded_by refuses when it validates a
        # registration's expected form, which is the only place that check runs.
        "adversarial: wildcarded suffix is not local-only, resolves normally",
        "expfam/results/wine_F.np?",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "adversarial: wildcarded suffix is not local-only, blocking when missing",
        "expfam/results/absent_thing.np?",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "adversarial: ignored suffix in a non-final segment",
        "expfam/results/wine_F.npy/nested.csv",
        Classification.UNRESOLVED,
    ),
    SelfTestCase(
        "adversarial: brace mixing an ignored and a missing tracked variant",
        "expfam/results/{wine_F.npy,absent_thing.csv}",
        Classification.TRUE_BROKEN,
    ),
    # -- issue #19 Finding A: the corrected notation must resolve ---------
    SelfTestCase(
        "A4 forward-corrected brace notation",
        "expfam/results/story_diagnostics/"
        "y_sparsity_stress_20260713{,_agg,_runinfo}.csv",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "A2 near miss of the corrected notation",
        "expfam/results/story_diagnostics/"
        "y_sparsity_stress_20260713{,_agg,_runinfos}.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase("bare basename without context", "run_wine_dual.py", Classification.UNRESOLVED),
    SelfTestCase("bare directory without context", "cora_clean/", Classification.UNRESOLVED),
    SelfTestCase(
        "wildcarded extension is a path signal",
        "expfam/results/fig_scenario_A_exp1_k.*",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "wildcarded extension, bare basename without context",
        "fig_scenario_A_exp1_k.*",
        Classification.UNRESOLVED,
    ),
    SelfTestCase(
        "wildcarded extension, root anchored and missing",
        "expfam/results/fig_scenario_Z_exp1_k.*",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "extension alternation, both halves present",
        "figures/fig1a_n_sweep_color.pdf/png",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "extension alternation, one half missing",
        "figures/fig1a_n_sweep_color.pdf/svg",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "extension alternation combined with brace and wildcard",
        "expfam/results/fig_scenario_{A}_*.pdf/png",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "non-final extension that is not alternation",
        "expfam/results/exp1_full_A.csv/nested/thing.csv",
        Classification.UNRESOLVED,
    ),
    SelfTestCase(
        "missing top-level directory is still blocking",
        "nosuchtop/results/exp1.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "missing top-level directory, wildcard form",
        "nosuchtop/results/*.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "real path containing parentheses",
        "Mato Lab Program/EffcalcEtaNewton (1).m",
        Classification.EXISTS_LITERAL,
    ),
    SelfTestCase(
        "missing path containing parentheses",
        "Mato Lab Program/NotThere (2).m",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "mathematical expression containing a slash",
        "kd - k(k-1)/2",
        Classification.NON_PATH,
    ),
    SelfTestCase(
        "mathematical expression, unicode minus",
        "kd − k(k−1)/2",
        Classification.NON_PATH,
    ),
    SelfTestCase(
        "fraction shaped like a two-segment path",
        "1/sigma_Y2",
        Classification.UNRESOLVED,
    ),
    SelfTestCase(
        "reference written relative to an unstated base",
        "real_data/wine_clean/summary.csv",
        Classification.UNRESOLVED,
    ),
    SelfTestCase(
        "same tail, but the first segment is a repository-root entry",
        "figures/real_data/wine_clean/plot.png",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "same tail, root-entry first segment, directory form",
        "figures/real_data/wine_clean/",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "unstated base, but the tail does not exist anywhere",
        "real_data/wine_clean/absent.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "unstated base with a wildcard is still blocking",
        "real_data/wine_clean/*.csv",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "brace alternatives written with spaces",
        "expfam/results/exp1_full_{A, B, C}.csv",
        Classification.PATTERN_RESOLVED,
    ),
    SelfTestCase(
        "reference into an index-excluded directory",
        "expfam/src/__pycache__/",
        Classification.UNRESOLVED,
    ),
    SelfTestCase(
        "several references in one token",
        "expfam/results/a.csv, expfam/results/b.csv",
        Classification.UNRESOLVED,
    ),
    SelfTestCase("parent directory traversal", "../outside/secret.csv", Classification.UNRESOLVED),
    SelfTestCase("absolute posix path", "/etc/passwd", Classification.UNRESOLVED),
    SelfTestCase(
        "absolute windows path",
        "C:\\Windows\\system32\\drivers",
        Classification.UNRESOLVED,
    ),
    SelfTestCase(
        "unknown extension, root anchored, missing",
        "expfam/results/output.qqq",
        Classification.TRUE_BROKEN,
    ),
    SelfTestCase(
        "unbalanced brace",
        "expfam/results/exp1_full_{A,B.csv",
        Classification.UNRESOLVED,
    ),
)


def validator_report_from_text(validator: Validator, markdown: str) -> Report:
    """Classify the tokens of an in-memory Markdown snippet."""
    accepted: list[Candidate] = []
    for candidate in extract_candidates("<self-test>", markdown):
        validator.classify(candidate, accepted)
        accepted.append(candidate)
    return Report(
        root=validator.root.as_posix(), sources=["<self-test>"], candidates=accepted
    )


def _token_checks(validator: Validator) -> list[tuple[str, str, str, bool]]:
    results: list[tuple[str, str, str, bool]] = []
    for case in SELF_TEST_CASES:
        report = validator_report_from_text(validator, f"| x | `{case.token}` |\n")
        actual = report.candidates[0].classification
        results.append(
            (case.name, case.expected.value, actual.value, actual is case.expected)
        )
    return results


def _structural_checks(validator: Validator) -> list[tuple[str, str, str, bool]]:
    results: list[tuple[str, str, str, bool]] = []

    def record(name: str, expected: object, actual: object) -> None:
        results.append((name, str(expected), str(actual), expected == actual))

    report = validator_report_from_text(
        validator, "| row | `expfam/results/wine_dual_results.csv`, `wine_F.npy` | n |\n"
    )
    # The sibling base is still inferred and the artifact is still located --
    # that is what this check protects.  Since issue #19 the label states that
    # the artifact is outside Git rather than verified to exist.
    record(
        "sibling continuation locates the artifact",
        "LOCAL_ONLY_ARTIFACT/inferred-context/expfam/results/wine_F.npy",
        "{}/{}/{}".format(
            report.candidates[1].classification.value,
            report.candidates[1].resolved_via,
            report.candidates[1].matches[0] if report.candidates[1].matches else "-",
        ),
    )

    report = validator_report_from_text(
        validator,
        "| r | `expfam/results/fig_scenario_A_exp1_k.pdf`, "
        "`fig_scenario_A_exp1_k.*` | n |\n",
    )
    record(
        "wildcarded sibling continuation resolves",
        "PATTERN_RESOLVED/2",
        "{}/{}".format(
            report.candidates[1].classification.value, report.candidates[1].match_count
        ),
    )

    report = validator_report_from_text(
        validator,
        "| r | `expfam/results/fig_scenario_A_exp1_k.pdf`, "
        "`fig_scenario_Z_missing.*` | n |\n",
    )
    record(
        "broken wildcarded sibling is not silently resolved",
        "UNRESOLVED",
        report.candidates[1].classification.value,
    )

    report = validator_report_from_text(
        validator, "| `expfam/results/wine_dual_results.csv` | `wine_F.npy` |\n"
    )
    record(
        "sibling continuation does not cross cells",
        "UNRESOLVED",
        report.candidates[1].classification.value,
    )

    report = validator_report_from_text(
        validator, "| row | `expfam/results/wine_dual_results.csv`, `wine_G.npy` | n |\n"
    )
    # The Phase 5a assertion, restored unchanged: `wine_G.npy` is not
    # registered, so no local-only entry covers it and it keeps the ordinary
    # verdict.  Presence and match count are asserted too, which the baseline
    # check did not do.
    record(
        "broken sibling is not silently resolved",
        "UNRESOLVED//0",
        "{}/{}/{}".format(
            report.candidates[1].classification.value,
            report.candidates[1].local_presence,
            report.candidates[1].match_count,
        ),
    )

    report = validator_report_from_text(
        validator, "```\n`expfam/results/nope.csv`\n```\n\n`figures/`\n"
    )
    record("fenced code block skipped", 1, len(report.candidates))

    report = validator_report_from_text(validator, "intro\n\n`figures/`\n")
    record("line number preserved", 3, report.candidates[0].line)

    report = validator_report_from_text(
        validator, "`expfam/results/nope.csv`\n`expfam/results/nope.csv`\n"
    )
    record("duplicate references kept separately", 2, len(report.candidates))
    record(
        "duplicate references classified alike",
        "TRUE_BROKEN|TRUE_BROKEN",
        "|".join(c.classification.value for c in report.candidates),
    )

    report = validator_report_from_text(
        validator, "| a | `expfam/results/wine_dual_results.csv`, `wine_F.npy` | b |\n"
    )
    record(
        "cell index stable across inline code",
        (2, 2),
        (report.candidates[0].cell, report.candidates[1].cell),
    )

    report = validator_report_from_text(
        validator, "| a `figures/` | b `expfam/results/wine_F.npy` |\n"
    )
    record(
        "cell index distinguishes neighbouring cells",
        (1, 2),
        (report.candidates[0].cell, report.candidates[1].cell),
    )

    report = validator_report_from_text(validator, "`expfam/results/wine_F.npy`\n")
    record(
        "gitignored artifact is not reported as verified existence",
        "LOCAL_ONLY_ARTIFACT/present",
        "{}/{}".format(
            report.candidates[0].classification.value,
            report.candidates[0].local_presence,
        ),
    )
    record(
        "gitignore annotation is attached as evidence",
        True,
        any(".gitignore" in item for item in report.candidates[0].evidence),
    )

    try:
        expand_braces("expfam/results/" + "{a,b}" * 9 + ".csv")
        bounded = False
    except ExpansionError:
        bounded = True
    record("brace expansion bounded", True, bounded)

    first = validator_report_from_text(validator, "`expfam/results/*.csv`\n")
    second = validator_report_from_text(validator, "`expfam/results/*.csv`\n")
    record(
        "match ordering deterministic",
        first.candidates[0].matches,
        second.candidates[0].matches,
    )
    record(
        "match ordering sorted",
        sorted(first.candidates[0].matches),
        first.candidates[0].matches,
    )
    record(
        "single-star glob does not cross a separator",
        [],
        validator.index.match("expfam/*.csv"),
    )
    record(
        "tree index confined to the root",
        True,
        all(
            not path.startswith("..") and not path.startswith("/")
            for path in validator.index.all_paths
        ),
    )

    record(
        "gitignored scratch directory excluded from the index",
        True,
        not any(path.startswith(".venv") for path in validator.index.all_paths),
    )
    record(
        "path-specific gitignore rule does not skip a directory",
        True,
        "reproduction" not in validator.skipped_directories,
    )
    record(
        "gitignore caveat is attached to a failing reference too",
        True,
        any(
            ".gitignore" in item
            for item in validator_report_from_text(
                validator, "`expfam/results/absent_artifact.npy`\n"
            ).candidates[0].evidence
        ),
    )

    report = validator_report_from_text(
        validator,
        "| r | `expfam/results/real_data/wine_clean/summary.csv`, "
        "`cora_clean/summary.csv` | n |\n",
    )
    record(
        "sibling chain uses the resolved form of the previous token",
        "PATTERN_RESOLVED",
        report.candidates[1].classification.value,
    )

    report = validator_report_from_text(
        validator,
        "| r | `expfam/results/exp1_full_{A,B,C}.csv`, `nope.csv` | n |\n",
    )
    record(
        "patterned prefix is not used as a base directory",
        "UNRESOLVED",
        report.candidates[1].classification.value,
    )

    first = validator_report_from_text(
        validator, "| a | `expfam/results/exp_scenario_A_exp1_k.csv` | b |\n"
    )
    probe = Candidate(source="other.md", line=1, column=1, cell=1, raw="wine_F.npy")
    validator.classify(probe, first.candidates)
    record(
        "sibling continuation does not cross documents",
        "UNRESOLVED",
        probe.classification.value,
    )

    sibling = validator_report_from_text(
        validator,
        "| r | `expfam/results/wine_dual_results.csv`, `wine_F.npy` | n |\n",
    )
    record("inferred-context resolutions are counted separately", 1,
           sibling.inferred_context_count)
    record(
        "gitignore-sensitive verdicts are counted structurally",
        1,
        len(sibling.gitignore_sensitive_candidates),
    )

    # Reason strings are part of the reviewed output, so they are asserted.
    partial = validator_report_from_text(
        validator, "| a | `expfam/results/exp1_full_{A,B,D}.csv` | b |\n"
    )
    record(
        "a partially matching token does not claim the cell base was tried",
        True,
        "no other interpretation was attempted" in partial.candidates[0].reason,
    )
    lonely = validator_report_from_text(
        validator, "| a | `expfam/results/absent_thing.csv` | b |\n"
    )
    record(
        "a wholly unmatched token says the cell offered no base",
        True,
        "offered no base to try instead" in lonely.candidates[0].reason,
    )
    outside = validator_report_from_text(validator, "`expfam/results/absent.csv`\n")
    record(
        "a token outside any table cell says so",
        True,
        "not in a table cell" in outside.candidates[0].reason,
    )
    tried = validator_report_from_text(
        validator,
        "| r | `expfam/results/real_data/wine_clean/summary.csv`, "
        "`nope_dir/absent.csv` | n |\n",
    )
    record(
        "a token whose cell base was tried and failed says so",
        "TRUE_BROKEN/True",
        "{}/{}".format(
            tried.candidates[1].classification.value,
            "nor against the base implied by the same table cell"
            in tried.candidates[1].reason,
        ),
    )
    record(
        "a single-segment token is described as such",
        True,
        "it is a single segment" in
        validator_report_from_text(validator, "`run_wine_dual.py`\n").candidates[0].reason,
    )
    record(
        "a non-locatable multi-segment token is described as such",
        True,
        "its last segment is neither" in
        validator_report_from_text(validator, "`1/sigma_Y2`\n").candidates[0].reason,
    )

    blocking = validator_report_from_text(validator, "`expfam/results/nope.csv`\n")
    blocking.unresolved_is_blocking = True
    record("unchecked count drops UNRESOLVED when it blocks", 0, blocking.unchecked_count)
    relaxed = validator_report_from_text(validator, "`some_basename.csv`\n")
    record("unchecked count includes UNRESOLVED by default", 1, relaxed.unchecked_count)

    return results


def _refuse_outside_fixture(root: Path) -> None:
    """Abort unless ``root`` is a throwaway fixture tree.

    The two suites below are the only code in this module that removes or
    rewrites a file.  They are called with a ``tempfile.TemporaryDirectory()``
    and nothing else, and this guard makes that a checked precondition rather
    than a convention: a fixture root always carries the prefix this module
    gave it, and a repository root -- which has a ``.git`` entry -- never does.
    """
    resolved = root.resolve()
    if (resolved / ".git").exists():
        raise RuntimeError(f"refusing to modify a repository checkout: {resolved}")
    if not resolved.name.startswith("registry_path_validator_"):
        raise RuntimeError(f"refusing to modify a non-fixture directory: {resolved}")


def _fresh_checkout_checks(root: Path) -> list[tuple[str, str, str, bool]]:
    """The fixture tree with every gitignored artifact removed.

    This is the CI side of the local-only policy: the same references, on a
    tree that never received the untracked research outputs.  What must not
    change is the classification; what may change is the presence.
    """
    results: list[tuple[str, str, str, bool]] = []

    def record(name: str, expected: object, actual: object) -> None:
        results.append((name, str(expected), str(actual), expected == actual))

    _refuse_outside_fixture(root)
    for artifact in sorted(root.rglob("*.npy")):
        artifact.unlink()
    validator = Validator(root, FIXTURE_LOCAL_ONLY_REFERENCES)

    record(
        "fresh checkout: ignored artifact keeps its classification",
        "LOCAL_ONLY_ARTIFACT",
        _classify_one(validator, "expfam/results/wine_F.npy"),
    )
    probe = Candidate(
        source="<self-test>", line=1, column=1, cell=None,
        raw="expfam/results/wine_F.npy",
    )
    validator.classify(probe, [])
    record("fresh checkout: presence flips to absent", "absent", probe.local_presence)
    record("fresh checkout: nothing is claimed to match", 0, probe.match_count)
    record(
        "fresh checkout: ignored wildcard keeps its classification",
        "LOCAL_ONLY_ARTIFACT",
        _classify_one(validator, "expfam/data/movielens_pilot/*.npy"),
    )
    record(
        "fresh checkout: non-ignored sibling of an ignored file still resolves",
        "PATTERN_RESOLVED",
        _classify_one(validator, "expfam/data/movielens_pilot/*.csv"),
    )
    record(
        "fresh checkout: a missing non-ignored path is still blocking",
        "TRUE_BROKEN",
        _classify_one(validator, "expfam/results/absent_artifact.csv"),
    )
    record(
        "fresh checkout: a mistyped directory is still blocking",
        "TRUE_BROKEN",
        _classify_one(validator, "expfam/reslts/wine_F.npy"),
    )
    return results


def _unsupported_gitignore_checks(root: Path) -> list[tuple[str, str, str, bool]]:
    """A ``.gitignore`` this reader cannot evaluate exactly disables the policy.

    A ``!`` negation can re-include a file an earlier rule excluded.  Rather
    than approximate Git's ordering rules, the local-only verdict is withdrawn
    entirely and the references fall back to the ordinary ones.
    """
    results: list[tuple[str, str, str, bool]] = []

    def record(name: str, expected: object, actual: object) -> None:
        results.append((name, str(expected), str(actual), expected == actual))

    _refuse_outside_fixture(root)
    (root / ".gitignore").write_text(
        "*.npy" + chr(10) + "!expfam/results/wine_F.npy" + chr(10) + ".venv/" + chr(10),
        encoding="utf-8",
    )
    validator = Validator(root, FIXTURE_LOCAL_ONLY_REFERENCES)
    record("negated .gitignore: local-only policy is off", False,
           validator.local_only_supported)
    record(
        "negated .gitignore: present artifact falls back to the tree verdict",
        "EXISTS_LITERAL",
        _classify_one(validator, "expfam/results/wine_F.npy"),
    )
    record(
        "negated .gitignore: absent artifact is blocking again",
        "TRUE_BROKEN",
        _classify_one(validator, "expfam/results/absent_artifact.npy"),
    )

    # The registered rule is simply gone.  The registration is still in the
    # mapping, so this is the case where an entry must lapse rather than go on
    # excusing a reference the repository no longer says anything about.
    (root / ".gitignore").write_text(
        chr(10).join(["*.npz", ".venv/", "__pycache__/"]) + chr(10),
        encoding="utf-8",
    )
    lapsed = Validator(root, FIXTURE_LOCAL_ONLY_REFERENCES)
    record(
        "withdrawn .gitignore rule: local-only policy is still supported",
        True,
        lapsed.local_only_supported,
    )
    record(
        "withdrawn .gitignore rule: the registration no longer authorizes",
        None,
        lapsed._local_only_reference(
            Candidate(source="<self-test>", line=1, column=1, cell=None,
                      raw="expfam/results/wine_F.npy")
        ),
    )
    record(
        "withdrawn .gitignore rule: present artifact falls back to the tree",
        "EXISTS_LITERAL",
        _classify_one(lapsed, "expfam/results/wine_F.npy"),
    )
    record(
        "withdrawn .gitignore rule: absent artifact is blocking again",
        "TRUE_BROKEN",
        _classify_one(lapsed, "expfam/results/absent_artifact.npy"),
    )
    return results


def _mutate_segment(path: str, index: int) -> str:
    """Return ``path`` with one segment corrupted so that it cannot exist."""
    trailing = "/" if path.endswith("/") else ""
    segments = [s for s in path.rstrip("/").split("/") if s]
    target = segments[index]
    stem, dot, suffix = target.rpartition(".")
    segments[index] = f"{stem}_ZQX{dot}{suffix}" if dot else f"{target}_ZQX"
    return "/".join(segments) + trailing


def _real_tree_checks(root: Path) -> list[tuple[str, str, str, bool]]:
    """Adversarial checks against the actual working tree.

    The anchors are taken from whatever currently resolves in the default
    source, so the suite adapts to the tree instead of hard-coding file names.
    It is reproducible from this file together with the default source document
    and the working tree it describes -- not from this file in isolation.

    A missing anchor is recorded as SKIP and counted separately, so the suite
    quietly shrinking is visible in the summary rather than hidden behind a
    green total.
    """
    results: list[tuple[str, str, str, bool]] = []

    def record(name: str, expected: object, actual: object) -> None:
        results.append((name, str(expected), str(actual), expected == actual))

    def skip(name: str, why: str) -> None:
        results.append((name, SKIP_MARKER, f"{SKIP_MARKER} ({why})", True))

    try:
        sources = resolve_sources(root, None)
    except (ValueError, OSError) as error:
        # Not a skip: run from inside the repository, the default source must
        # be readable, and losing it would silently delete this whole suite.
        record("real tree: default source is readable", "readable", str(error))
        return results

    validator = Validator(root)
    report = validator.run(sources)

    record(
        "real tree: index stays inside the repository",
        True,
        all(
            not path.startswith(("..", "/")) and ":" not in path.split("/")[0]
            for path in validator.index.all_paths
        ),
    )
    record(
        "real tree: rerun is identical",
        report.to_dict(),
        Validator(root).run(sources).to_dict(),
    )

    literals = [
        c
        for c in report.candidates
        if c.classification is Classification.EXISTS_LITERAL
        and segment_count(c.normalized) >= 3
    ]
    if not literals:
        skip("real tree: literal mutation controls", "no multi-segment literal")
    else:
        anchor = literals[0].normalized
        record(
            "real tree: literal anchor resolves",
            "EXISTS_LITERAL",
            _classify_one(validator, anchor),
        )
        record(
            "real tree: corrupted final segment is blocking",
            "TRUE_BROKEN",
            _classify_one(validator, _mutate_segment(anchor, -1)),
        )
        record(
            "real tree: corrupted middle segment is blocking",
            "TRUE_BROKEN",
            _classify_one(validator, _mutate_segment(anchor, 1)),
        )
        record(
            "real tree: corrupted top-level segment is blocking",
            "TRUE_BROKEN",
            _classify_one(validator, _mutate_segment(anchor, 0)),
        )
        record(
            "real tree: windows separators resolve the same",
            "EXISTS_LITERAL",
            _classify_one(validator, anchor.replace("/", "\\")),
        )

    patterns = [
        c
        for c in report.candidates
        if c.classification is Classification.PATTERN_RESOLVED
        and c.resolved_via == "root"
        and "*" in c.normalized
        and segment_count(c.normalized) >= 3
    ]
    if not patterns:
        skip("real tree: wildcard mutation controls", "no rooted wildcard")
    else:
        anchor = patterns[0].normalized
        record(
            "real tree: wildcard anchor resolves",
            "PATTERN_RESOLVED",
            _classify_one(validator, anchor),
        )
        record(
            "real tree: corrupted wildcard directory is blocking",
            "TRUE_BROKEN",
            _classify_one(validator, _mutate_segment(anchor, 1)),
        )

    exception = next(iter(INTENTIONAL_HISTORICAL_PATHS))
    record(
        "real tree: documented historical exception",
        "INTENTIONAL_HISTORICAL",
        _classify_one(validator, exception + "/"),
    )
    record(
        "real tree: historical near miss is blocking",
        "TRUE_BROKEN",
        _classify_one(validator, exception + "_typo/"),
    )
    record(
        "real tree: glob below the historical exception is blocking",
        "TRUE_BROKEN",
        _classify_one(validator, exception + "/*.pdf"),
    )
    # -- issue #19 Finding A: the notation-defect waiver -------------------
    # Snapshot: the lapse check below temporarily replaces an entry, and
    # mutating the mapping being iterated would otherwise depend on CPython
    # rehashing details rather than on anything the language guarantees.
    for defect in list(KNOWN_NOTATION_DEFECTS.values()):
        record(
            f"real tree: notation defect {defect.raw!r} is waived in its own document",
            "KNOWN_NOTATION_DEFECT",
            _classify_one(validator, defect.raw, source=defect.source),
        )
        record(
            "real tree: the recorded forward correction resolves",
            "PATTERN_RESOLVED",
            _classify_one(validator, defect.correction, source=defect.source),
        )
        record(
            "real tree: the waived token is not covered in another document",
            "TRUE_BROKEN",
            _classify_one(validator, defect.raw, source="KNOWN_ISSUES.md"),
        )
        record(
            "real tree: an unwaived document name is not covered either",
            "TRUE_BROKEN",
            _classify_one(validator, defect.raw, source=defect.source + ".bak"),
        )
        for near_miss in (
            defect.raw.replace("runinfo}", "runinfos}"),
            defect.raw.replace("{,agg", "{,aggs"),
            defect.raw.replace("20260713", "20260714"),
            defect.raw.replace("story_diagnostics", "story_diagnostic"),
            # Same artifacts, same normalised path, different raw text: the
            # waiver is keyed on what the document actually says, and anything
            # else keeps the ordinary verdict rather than being matched loosely.
            "./" + defect.raw,
        ):
            if near_miss == defect.raw:
                continue
            record(
                f"real tree: near miss {near_miss!r} is not waived",
                "TRUE_BROKEN",
                _classify_one(validator, near_miss, source=defect.source),
            )
        # End to end, not just the predicate: a waiver whose correction no
        # longer resolves must give the defective token back to the ordinary
        # rules, which report it broken.
        lapsed = NotationDefect(
            source=defect.source,
            raw=defect.raw,
            correction=_mutate_segment(defect.correction.split("{")[0] + "x.csv", -1),
            reason=defect.reason,
            evidence=defect.evidence,
        )
        record(
            "real tree: a lapsed correction does not resolve",
            False,
            validator._correction_resolves(lapsed.correction),
        )
        original = dict(KNOWN_NOTATION_DEFECTS)
        KNOWN_NOTATION_DEFECTS[lapsed.key] = lapsed
        try:
            record(
                "real tree: a lapsed waiver hands the token back as broken",
                "TRUE_BROKEN",
                _classify_one(validator, defect.raw, source=defect.source),
            )
        finally:
            KNOWN_NOTATION_DEFECTS.clear()
            KNOWN_NOTATION_DEFECTS.update(original)
        record(
            "real tree: the waiver is restored after the lapse check",
            "KNOWN_NOTATION_DEFECT",
            _classify_one(validator, defect.raw, source=defect.source),
        )

        # F5: the waiver constant lives in this file, while the forward
        # correction lives in the document.  Nothing would otherwise keep the
        # two in step, so the agreement is asserted rather than assumed: both
        # the defective token and its correction must appear in the source
        # document as inline code.
        # Compare on the repository-relative posix path, which is what
        # Validator.run puts in Candidate.source and therefore what the waiver
        # key is matched against.  Matching on the basename would silently
        # skip -- and a skip counts as a pass -- for a waiver recorded in a
        # document below the repository root.
        document = next(
            (
                s
                for s in sources
                if s.relative_to(root).as_posix() == defect.source
            ),
            None,
        )
        if document is None:
            # Same reasoning as the local-only guard below: a skip counts as a
            # pass, so a waiver registered against a document the default run
            # never reads would silently retire its own anchoring check.
            record(
                f"real tree: {defect.source} is in the validated source set",
                True,
                False,
            )
        else:
            tokens = {
                c.raw for c in extract_candidates(defect.source,
                                                  document.read_text(encoding="utf-8"))
            }
            # Scope note: membership is tested over the whole document, not
            # over the forward-correction section alone.  For ``raw`` that is
            # weak on purpose -- the historical row contains it too, so this
            # half only proves the waiver still has something to waive.  The
            # ``correction`` half is the load-bearing one: that string occurs
            # nowhere else in the document, so it fails if the section goes.
            record(
                "real tree: the waived token appears in its source document",
                True,
                defect.raw in tokens,
            )
            record(
                "real tree: the recorded correction appears there too",
                True,
                defect.correction in tokens,
            )

    # -- issue #19 Finding C: run_comparison_all.py provenance -------------
    record(
        "real tree: the forward-corrected script path resolves",
        "EXISTS_LITERAL",
        _classify_one(validator, "reproduction/scripts/run_comparison_all.py"),
    )
    record(
        "real tree: the historical bare basename is not guessed",
        "UNRESOLVED",
        _classify_one(validator, "run_comparison_all.py"),
    )
    record(
        "real tree: the path the historical cell implies is still blocking",
        "TRUE_BROKEN",
        _classify_one(validator, "reproduction/src/run_comparison_all.py"),
    )
    record(
        "real tree: an arbitrary unique basename is not guessed either",
        "UNRESOLVED",
        _classify_one(validator, "data_generator.py"),
    )
    # The isolated check above does not reach the sibling-rebase branch.  The
    # real registry cell puts a resolved path before the bare basename, so the
    # validator does try 'reproduction/src/run_comparison_all.py' -- and must
    # still decline to reach for the file that exists under another parent.
    in_context = validator_report_from_text(
        validator,
        "| Control | `reproduction/src/experiment_compare_with_dual.py`, "
        "`run_comparison_all.py` | n |",
    )
    record(
        "real tree: the bare basename stays unresolved in its real cell context",
        "EXISTS_LITERAL/UNRESOLVED/0",
        "{}/{}/{}".format(
            in_context.candidates[0].classification.value,
            in_context.candidates[1].classification.value,
            in_context.candidates[1].match_count,
        ),
    )
    record(
        "real tree: the failed sibling base is reported as evidence",
        True,
        any(
            "reproduction/src" in item
            for item in in_context.candidates[1].evidence
        ),
    )

    # -- issue #19 Finding B: only registered references are local-only ----
    for reference in list(LOCAL_ONLY_ARTIFACT_REFERENCES.values()):
        # Asserted against the real document rather than a synthetic cell, so
        # that the whole chain -- extraction, any contextual rebasing, and the
        # registration -- is what is being checked.
        occurrences = [
            c
            for c in report.candidates
            if c.source == reference.source and c.raw == reference.raw
        ]
        record(
            f"real tree: registered {reference.raw!r} occurs in its document",
            True,
            bool(occurrences),
        )
        record(
            f"real tree: every occurrence of {reference.raw!r} is local-only",
            True,
            all(
                c.classification is Classification.LOCAL_ONLY_ARTIFACT
                and c.correction == reference.expected
                for c in occurrences
            ),
        )
        if normalize_token(reference.raw) != normalize_token(reference.expected):
            # The entry records a rebased form, so the token on its own -- with
            # no cell to rebase against -- must NOT be excused.  Registration
            # is necessary but never sufficient.
            record(
                f"real tree: {reference.raw!r} without its cell context is not local-only",
                True,
                _classify_one(validator, reference.raw, source=reference.source)
                != Classification.LOCAL_ONLY_ARTIFACT.value,
            )
        record(
            f"real tree: registered {reference.raw!r} is not covered elsewhere",
            True,
            _classify_one(validator, reference.raw, source="KNOWN_ISSUES.md")
            != Classification.LOCAL_ONLY_ARTIFACT.value,
        )
        # One-character near misses of the registered raw token.
        body, dot, suffix = reference.raw.rpartition(".")
        for near_miss in (body + "x" + dot + suffix, body[:-1] + dot + suffix,
                          reference.raw.replace("/", "/x", 1)):
            if near_miss == reference.raw:
                continue
            record(
                f"real tree: near miss {near_miss!r} does not inherit the entry",
                True,
                _classify_one(validator, near_miss, source=reference.source)
                != Classification.LOCAL_ONLY_ARTIFACT.value,
            )
        # The expected path must be the one the entry records, and the entry
        # must not cover the token when it denotes something else.
        record(
            f"real tree: expected form of {reference.raw!r} is excluded by its rule",
            True,
            validator._expected_variants(reference) is not None,
        )
        # The two guards that police the CONTENT of a registration.  They are
        # the reviewed-decision surface the whole design leans on, so each gets
        # a negative case built from this very entry.
        record(
            "real tree: a registration whose rule does not exclude its path lapses",
            None,
            validator._expected_variants(
                LocalOnlyArtifactReference(
                    source=reference.source,
                    raw=reference.raw,
                    expected="expfam/results/wine_dual_results.csv",
                    ignore_rule=reference.ignore_rule,
                    reason="probe: expected form is not covered by the rule",
                )
            ),
        )
        record(
            "real tree: a registration whose expected form is a bare name lapses",
            None,
            validator._expected_variants(
                LocalOnlyArtifactReference(
                    source=reference.source,
                    raw=reference.raw,
                    expected="wine_F.npy",
                    ignore_rule=reference.ignore_rule,
                    reason="probe: expected form names no location",
                )
            ),
        )
        record(
            "real tree: a registration with an unexpandable expected form lapses",
            None,
            validator._expected_variants(
                LocalOnlyArtifactReference(
                    source=reference.source,
                    raw=reference.raw,
                    expected="expfam/results/" + "{a,b}" * 9 + ".npy",
                    ignore_rule=reference.ignore_rule,
                    reason="probe: expected form exceeds the expansion bound",
                )
            ),
        )
        record(
            "real tree: a registration naming an absent .gitignore rule lapses",
            None,
            validator._expected_variants(
                LocalOnlyArtifactReference(
                    source=reference.source,
                    raw=reference.raw,
                    expected=reference.expected,
                    ignore_rule="*.no_such_rule",
                    reason="probe: rule is not in .gitignore",
                )
            ),
        )

        # A registration must correspond to a token that really is written in
        # the document it names; otherwise it is a stale entry sitting in the
        # mapping with nothing to justify it.
        document = next(
            (
                src
                for src in sources
                if src.relative_to(root).as_posix() == reference.source
            ),
            None,
        )
        if document is None:
            # Deliberately a failure, not a skip.  A skip counts as a pass, so
            # registering against a document the default run never reads would
            # silently retire every guard below for that entry.  Adding such a
            # registration has to be a decision someone makes on purpose.
            record(
                f"real tree: {reference.source} is in the validated source set",
                True,
                False,
            )
        else:
            tokens = {
                c.raw
                for c in extract_candidates(
                    reference.source, document.read_text(encoding="utf-8")
                )
            }
            record(
                f"real tree: registered {reference.raw!r} is a token of its document",
                True,
                reference.raw in tokens,
            )
            # Drift in both directions, which is what actually went wrong
            # once: the document's registration table must name exactly the
            # registered tokens, one per row.  Counting rows alone would pass
            # while a row's contents drifted, and asking only whether the token
            # appears somewhere in the document would be satisfied by the
            # historical row it was registered for.  So this is a bijection:
            # every registration is named by a row, and every row names one.
            row_prefix = "| `" + reference.source + "` |"
            rows = [
                line
                for line in document.read_text(encoding="utf-8").splitlines()
                if line.startswith(row_prefix)
            ]
            registered_here = {
                other.raw
                for other in LOCAL_ONLY_ARTIFACT_REFERENCES.values()
                if other.source == reference.source
            }
            # A multiset, not a set: two rows naming the same registration
            # would satisfy set equality while leaving the table wrong.
            named: list[str] = []
            rows_naming_exactly_one = 0
            rule_stated = 0
            for line in rows:
                hits = registered_here & {
                    c.raw for c in extract_candidates(reference.source, line)
                }
                if len(hits) != 1:
                    continue
                rows_naming_exactly_one += 1
                token = next(iter(hits))
                named.append(token)
                # The rule column is plain text, so it is matched as a
                # substring rather than as an inline-code token; writing it as
                # code would make it a path candidate in its own right.  It is
                # matched against the LAST cell, not the whole row: a wildcard
                # registration such as 'dir/*.npy' contains the rule inside its
                # own token, so a whole-row test would pass for that row even
                # with the rule column deleted.
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if cells and LOCAL_ONLY_ARTIFACT_REFERENCES[
                    (reference.source, token)
                ].ignore_rule == cells[-1]:
                    rule_stated += 1
            record(
                f"real tree: every registration for {reference.source} is in its table",
                sorted(registered_here),
                sorted(named),
            )
            record(
                f"real tree: no {reference.source} table row is duplicated",
                len(registered_here),
                len(named),
            )
            record(
                f"real tree: every {reference.source} table row names one registration",
                len(rows),
                rows_naming_exactly_one,
            )
            record(
                f"real tree: every {reference.source} row states its .gitignore rule",
                len(rows),
                rule_stated,
            )

    # -- issue #19 Finding B: unregistered ignored references stay ordinary --
    for token in (
        "expfam/results/wine_typoooo.npy",
        "expfam/results/wine_G.npy",
        "expfam/results/absent_artifact.npy",
        "expfam/data/movielens_pilot/absent_thing.npy",
        "expfam/results/absent.log",
        "expfam/results/absent.tmp",
        "expfam/results/absent.pkl",
        "expfam/results/absent.pickle",
        "expfam/results/absent.npz",
    ):
        record(
            f"real tree: unregistered ignored reference {token!r} is blocking",
            "TRUE_BROKEN",
            _classify_one(validator, token, source="EXPERIMENT_REGISTRY.md"),
        )

    # -- issue #19 Finding B: local-only artifacts in the real tree --------
    record(
        "real tree: a registry-referenced ignored artifact is local-only",
        "LOCAL_ONLY_ARTIFACT",
        _classify_one(
            validator,
            "expfam/data/movielens_pilot/*.npy",
            source="EXPERIMENT_REGISTRY.md",
        ),
    )
    # Assert the property, not a label: what this token classifies as in
    # another document depends on whether the artifacts happen to be on this
    # machine.  What must hold in both environments is that the registration
    # does not reach across documents.
    record(
        "real tree: the same token is not covered in another document",
        True,
        _classify_one(
            validator,
            "expfam/data/movielens_pilot/*.npy",
            source="KNOWN_ISSUES.md",
        )
        != Classification.LOCAL_ONLY_ARTIFACT.value,
    )
    record(
        "real tree: an ignored artifact under a mistyped directory is blocking",
        "TRUE_BROKEN",
        _classify_one(validator, "expfam/data/movielens_pilo/*.npy"),
    )
    record(
        "real tree: a non-ignored missing file beside it is blocking",
        "TRUE_BROKEN",
        _classify_one(validator, "expfam/data/movielens_pilot/absent_thing.csv"),
    )
    record(
        "real tree: the non-blocking policy classes are none of the blocking ones",
        True,
        Classification.TRUE_BROKEN not in NON_BLOCKING_BY_POLICY
        and Classification.UNRESOLVED not in NON_BLOCKING_BY_POLICY,
    )

    record(
        "real tree: parent traversal never touches the filesystem",
        "UNRESOLVED",
        _classify_one(validator, "../outside/secret.csv"),
    )
    record(
        "real tree: absolute path is refused",
        "UNRESOLVED",
        _classify_one(validator, "/etc/passwd"),
    )
    return results


def _classify_one(
    validator: Validator, token: str, source: str | None = None
) -> str:
    """Classification of ``token``, optionally as if written in ``source``.

    ``source`` matters because a KNOWN_NOTATION_DEFECT waiver is keyed on the
    document the token was written in, so the checks that probe the waiver have
    to be able to name a real document -- and to name a different one.
    """
    if source is None:
        report = validator_report_from_text(validator, f"| x | `{token}` |\n")
        return report.candidates[0].classification.value
    candidate = Candidate(source=source, line=1, column=1, cell=1, raw=token)
    validator.classify(candidate, [])
    return candidate.classification.value


def run_self_test(stream) -> int:
    with tempfile.TemporaryDirectory(prefix="registry_path_validator_") as temporary:
        root = Path(temporary).resolve()
        _build_fixture_tree(root)
        validator = Validator(root, FIXTURE_LOCAL_ONLY_REFERENCES)
        results = _token_checks(validator) + _structural_checks(validator)

    # Each of the following needs a differently shaped fixture tree, so each
    # gets its own throwaway directory rather than mutating a shared one.
    with tempfile.TemporaryDirectory(prefix="registry_path_validator_fresh_") as temporary:
        root = Path(temporary).resolve()
        _build_fixture_tree(root)
        results += _fresh_checkout_checks(root)

    with tempfile.TemporaryDirectory(prefix="registry_path_validator_neg_") as temporary:
        root = Path(temporary).resolve()
        _build_fixture_tree(root)
        results += _unsupported_gitignore_checks(root)

    results += _real_tree_checks(find_repository_root(Path(__file__).resolve().parent))

    width = max(len(name) for name, _, _, _ in results)
    failures = sum(1 for *_, ok in results if not ok)
    skipped = sum(1 for _, expected, _, _ in results if expected == SKIP_MARKER)
    for name, expected, actual, ok in results:
        status = "SKIP" if expected == SKIP_MARKER else "PASS" if ok else "FAIL"
        stream.write(
            f"{status}  {name.ljust(width)}  expected={_shorten(expected)}  "
            f"actual={_shorten(actual)}\n"
        )
    stream.write(
        f"\nself-test: {len(results) - failures - skipped}/{len(results)} PASS, "
        f"{failures} FAIL, {skipped} SKIP\n"
    )
    if skipped:
        stream.write(
            "note: a SKIP means an anchor was unavailable and those checks did "
            "not run; the totals above are not comparable with a run that had "
            "none.\n"
        )
    return EXIT_OK if failures == 0 else EXIT_BLOCKING


def _shorten(value: str, limit: int = 160) -> str:
    return value if len(value) <= limit else value[:limit] + "..."


# --------------------------------------------------------------------------
# command line interface
# --------------------------------------------------------------------------


def find_repository_root(start: Path) -> Path:
    for directory in [start, *start.parents]:
        if (directory / ".git").exists():
            return directory
    return start.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_registry_paths.py",
        description=(
            "Read-only check of the backtick-delimited path references in this "
            "repository's provenance documents. Nothing inside the repository is "
            "written, moved or removed."
        ),
        epilog=(
            "exit codes: 0 = no blocking finding, 1 = blocking finding, "
            "2 = invalid invocation or internal error"
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        metavar="PATH",
        help=(
            "repository-relative Markdown document to check; may be repeated and "
            f"replaces the default set ({', '.join(DEFAULT_SOURCES)})"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write the full machine-readable result to stdout instead of a report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also list every candidate, including the NON_PATH ones",
    )
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="treat UNRESOLVED as blocking as well (default: only TRUE_BROKEN)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "run the built-in tests: fixture checks in a temporary directory "
            "plus adversarial checks against this working tree"
        ),
    )
    return parser


def resolve_sources(root: Path, requested: Iterable[str] | None) -> list[Path]:
    names = list(requested) if requested else list(DEFAULT_SOURCES)
    resolved: list[Path] = []
    for name in names:
        candidate = (root / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"source outside the repository root: {name}")
        if not candidate.is_file():
            raise ValueError(f"source document not found: {name}")
        resolved.append(candidate)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Provenance documents contain non-ASCII prose, while the console encoding
    # varies between machines and CI runners.  Never abort a read-only report
    # because a character cannot be encoded.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    if args.self_test:
        return run_self_test(sys.stdout)

    root = find_repository_root(Path(__file__).resolve().parent)

    try:
        sources = resolve_sources(root, args.source)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    try:
        report = Validator(root).run(sources)
    except (OSError, UnicodeDecodeError, RecursionError) as error:
        sys.stderr.write(f"internal error: {error}\n")
        return EXIT_USAGE

    report.unresolved_is_blocking = args.fail_on_unresolved

    if args.json:
        # ASCII-only so that the bytes are identical whatever the console
        # encoding of the machine or CI runner happens to be.
        json.dump(report.to_dict(), sys.stdout, indent=2, ensure_ascii=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(report, args.verbose))

    summary = report.summary
    blocking = summary[Classification.TRUE_BROKEN.value]
    if args.fail_on_unresolved:
        blocking += summary[Classification.UNRESOLVED.value]
    return EXIT_BLOCKING if blocking else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
