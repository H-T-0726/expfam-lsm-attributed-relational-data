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
``.gitignore`` excludes.  Likewise ``--self-test`` writes its fixtures into
``tempfile.TemporaryDirectory()``, which lands outside the repository unless
``TMPDIR``/``TEMP`` has been pointed inside it.

Two deliberate semantics are worth stating up front:

* Existence is decided from the working tree, not from ``git ls-files``, so an
  artifact that ``.gitignore`` excludes but that is present locally counts as
  existing.  A fresh checkout may therefore see fewer files than this run does.
* Matching is case sensitive on every platform, including Windows.  A reference
  whose case does not match the tree is reported, because it would fail on a
  case-sensitive filesystem.

``--self-test`` runs two suites: fixture checks against a throwaway tree (see
the note on ``TMPDIR`` above) and adversarial checks against the real working
tree, whose anchors are derived from whatever currently resolves rather than
from hard-coded file names.

Three limitations are worth knowing before wiring this into CI:

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

SCHEMA_VERSION = 2

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


class Classification(str, Enum):
    """Classification assigned by the validator to a single candidate."""

    EXISTS_LITERAL = "EXISTS_LITERAL"
    PATTERN_RESOLVED = "PATTERN_RESOLVED"
    TRUE_BROKEN = "TRUE_BROKEN"
    INTENTIONAL_HISTORICAL = "INTENTIONAL_HISTORICAL"
    NON_PATH = "NON_PATH"
    UNRESOLVED = "UNRESOLVED"


#: Stable report ordering, independent of dict or filesystem iteration order.
CLASSIFICATION_ORDER = (
    Classification.EXISTS_LITERAL,
    Classification.PATTERN_RESOLVED,
    Classification.INTENTIONAL_HISTORICAL,
    Classification.NON_PATH,
    Classification.UNRESOLVED,
    Classification.TRUE_BROKEN,
)

#: Classifications a human reviewer has to look at one by one.
PROBLEM_CLASSIFICATIONS = (
    Classification.TRUE_BROKEN,
    Classification.UNRESOLVED,
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


def read_gitignore_rules(root: Path) -> tuple[list[str], frozenset[str]]:
    """Read the two ``.gitignore`` forms this validator understands.

    Returns ``(extension_patterns, bare_directory_names)``:

    * ``*.ext`` lines, used only to annotate results.  A file may exist in the
      local working tree while being absent from a fresh checkout; the value
      never influences a classification.
    * ``name/`` lines that carry no path separator of their own, such as
      ``.venv/`` or ``__pycache__/``.  Those directories are excluded from the
      working-tree index so that the index -- and therefore the verdict -- does
      not depend on machine-local scratch directories.

    Every other ``.gitignore`` form is deliberately ignored; this is a partial
    reader, not a Git-compatible one.
    """
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return [], frozenset()
    extensions: list[str] = []
    directories: set[str] = set()
    for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        if re.fullmatch(r"\*\.[A-Za-z0-9_]+", stripped):
            extensions.append(stripped)
            continue
        if re.fullmatch(r"[A-Za-z0-9_.-]+/", stripped):
            directories.add(stripped[:-1])
    return sorted(set(extensions)), frozenset(directories)


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
        """PATTERN_RESOLVED entries that needed a base inferred from context.

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
        be a path at all, and INTENTIONAL_HISTORICAL is a documented absence.
        """
        summary = self.summary
        return {
            item.value: summary[item.value]
            for item in (
                Classification.UNRESOLVED,
                Classification.NON_PATH,
                Classification.INTENTIONAL_HISTORICAL,
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

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ignored_extension_patterns, ignored_directories = read_gitignore_rules(root)
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

        locatable = names_a_location(base_forms[0])
        sibling_tried = False

        if not matches:
            base = self._sibling_base(candidate, earlier)
            sibling_tried = base is not None
            if base is not None:
                rebased = [f"{base}/{variant}" for variant in candidate.variants]
                rebased_matches, rebased_unmatched = self._resolve_variants(rebased)
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
        f"of the {summary[Classification.PATTERN_RESOLVED.value]} PATTERN_RESOLVED, "
        f"{report.inferred_context_count} needed a base directory inferred from the "
        "surrounding table cell rather than resolving as written."
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
    lines.append(f"PATTERN_RESOLVED via inferred context ({len(inferred)})")
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
        ".venv/lib/exp_scenario_A_exp1_k.csv",
    ]
    for relative in files:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "*.npy\n*.npz\n.venv/\n__pycache__/\nreproduction/results/raw/\n",
        encoding="utf-8",
    )


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
    SelfTestCase(
        "ignored but existing artifact",
        "expfam/results/wine_F.npy",
        Classification.EXISTS_LITERAL,
    ),
    SelfTestCase(
        "ignored artifact wildcard",
        "expfam/results/*.npy",
        Classification.PATTERN_RESOLVED,
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
    record(
        "sibling continuation resolves",
        "PATTERN_RESOLVED/expfam/results/wine_F.npy",
        "{}/{}".format(
            report.candidates[1].classification.value,
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
    record(
        "broken sibling is not silently resolved",
        "UNRESOLVED",
        report.candidates[1].classification.value,
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
        "gitignored artifact still counts as existing",
        "EXISTS_LITERAL",
        report.candidates[0].classification.value,
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


def _classify_one(validator: Validator, token: str) -> str:
    report = validator_report_from_text(validator, f"| x | `{token}` |\n")
    return report.candidates[0].classification.value


def run_self_test(stream) -> int:
    with tempfile.TemporaryDirectory(prefix="registry_path_validator_") as temporary:
        root = Path(temporary).resolve()
        _build_fixture_tree(root)
        validator = Validator(root)
        results = _token_checks(validator) + _structural_checks(validator)

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
