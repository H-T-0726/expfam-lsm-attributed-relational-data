"""Documentation consistency tests for the clean true-K study.

A number that appears in a report and disagrees with the artifact is a silent
provenance break: the reader has no way to tell which one is wrong.  These
tests recompute the headline numbers from
``expfam/results/k_selection/clean_true_k_asymptotics_20260904/`` and require
that every document quoting them agrees.

They also enforce the repository's claim rules on the new documents: the
Q-based criterion must never be called a Schwarz BIC, and every document that
reports a recovery count must say that agreement is with K_TRUE rather than
with K*.

No EM, no artifact modified, nothing regenerated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (ROOT / "expfam" / "results" / "k_selection"
           / "clean_true_k_asymptotics_20260904")
REPORTS = ROOT / "reports"

RESULTS = REPORTS / "identifiability" / "clean_true_k_results_20260905.md"
TEACHER = REPORTS / "identifiability" / "teacher_discussion_summary_20260905.md"
QUESTIONS = REPORTS / "identifiability" / "teacher_expected_questions_20260905.md"
HANDOFF = REPORTS / "identifiability" / "AUTONOMOUS_RESEARCH_HANDOFF_20260905.md"
THEORY_MAP = REPORTS / "identifiability" / "k_selection_theory_map_20260905.md"
STORYLINE = REPORTS / "thesis" / "thesis_storyline_20260905.md"
OUTLINE = REPORTS / "thesis" / "thesis_detailed_outline_20260905.md"
INVENTORY = REPORTS / "thesis" / "thesis_figure_table_inventory_20260905.md"
REAL_APP = REPORTS / "thesis" / "real_application_interpretation_20260905.md"
RESEARCH_MASTER = ROOT / "RESEARCH_MASTER.md"
REGISTRY = ROOT / "EXPERIMENT_REGISTRY.md"
KNOWN_ISSUES = ROOT / "KNOWN_ISSUES.md"

# Documents that quote the experiment's numbers and therefore carry its caveats.
RESULT_QUOTING_DOCS = (RESULTS, TEACHER, QUESTIONS, HANDOFF, RESEARCH_MASTER, REGISTRY)

CANDIDATE_K = (1, 2, 3, 4, 5, 6, 7)
N_GRID = (50, 75, 100, 150)
K_TRUE_GRID = (1, 3, 5)
CRITERIA = ("S1", "S2", "S3")
COLUMN = {"S1": "heldout_mean_log_score", "S2": "s2_q_based",
          "S3": "s3_plugin_conditional"}
HIGHER_IS_BETTER = {"S1": True, "S2": False, "S3": False}
TIE_TOLERANCE = np.float64(1e-12)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def selection() -> dict:
    """Selected K per (criterion, K_TRUE, n, replicate), from the RAW values."""

    fits = _read_csv(RUN_DIR / "fit_results.csv")
    by_cell: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    for row in fits:
        by_cell.setdefault((int(row["k_true"]), int(row["n"]),
                            int(row["replicate"])), []).append(row)
    out = {}
    for cell, rows in by_cell.items():
        for name in CRITERIA:
            means = {}
            for k_est in CANDIDATE_K:
                vals = [float(r[COLUMN[name]]) for r in rows
                        if int(r["k_est"]) == k_est]
                assert len(vals) == 2, (name, cell, k_est)
                signed = vals if HIGHER_IS_BETTER[name] else [-v for v in vals]
                means[k_est] = float(np.mean(np.asarray(signed, dtype=np.float64),
                                             dtype=np.float64))
            best = max(means.values())
            ties = sorted(k for k, v in means.items() if best - v <= TIE_TOLERANCE)
            out[(name, *cell)] = min(ties)
    return out


def _exact(selection: dict, name: str, k_true: int, n: int) -> int:
    keys = [k for k in selection if k[0] == name and k[1] == k_true and k[2] == n]
    return sum(1 for k in keys if selection[k] == k_true)


def _mean(selection: dict, name: str, k_true: int, n: int) -> float:
    keys = [k for k in selection if k[0] == name and k[1] == k_true and k[2] == n]
    return float(np.mean([selection[k] for k in keys]))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- structure

def test_artifact_set_is_complete_and_has_no_failure_marker():
    for name in ("protocol.json", "manifest.csv", "generator_provenance.csv",
                 "fit_results.csv", "selection_matrix.csv", "gram_spectrum.csv",
                 "summary.json", "runinfo.json", "audit_report.json"):
        assert (RUN_DIR / name).exists(), name
    assert not (RUN_DIR / "failure.json").exists()


def test_fit_count_is_896_everywhere_it_is_quoted():
    fits = _read_csv(RUN_DIR / "fit_results.csv")
    assert len(fits) == 896
    runinfo = json.loads((RUN_DIR / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["attempted_fit_count"] == 896
    assert runinfo["completed_fit_count"] == 896
    for path in RESULT_QUOTING_DOCS:
        assert "896" in _text(path), path.name


def test_integrity_counters_are_zero_and_documented():
    runinfo = json.loads((RUN_DIR / "runinfo.json").read_text(encoding="utf-8"))
    for key in ("retry_count", "replacement_fits_executed", "seed_rescue_count",
                "tolerance_relaxations"):
        assert runinfo[key] == 0, key
    assert runinfo["resumed"] is False


def test_protocol_hash_agrees_across_artifact_and_documents():
    protocol = json.loads((RUN_DIR / "protocol.json").read_text(encoding="utf-8"))
    digest = protocol["protocol_hash"]
    assert len(digest) == 64
    runinfo = json.loads((RUN_DIR / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["protocol_hash"] == digest
    prefix = digest[:8]
    for path in (RESULTS, HANDOFF, RESEARCH_MASTER, REGISTRY):
        assert prefix in _text(path), path.name


def test_audit_verdict_is_pass_and_reported_as_such():
    report = json.loads((RUN_DIR / "audit_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS"
    assert report["blocker_count"] == 0
    assert report["high_count"] == 0
    assert report["medium_count"] == 0
    for path in (RESULTS, HANDOFF, REGISTRY):
        assert "PASS" in _text(path), path.name


# ------------------------------------------------------- quoted result numbers

@pytest.mark.parametrize("name,k_true,expected", [
    ("S1", 5, [2, 0, 4, 8]),
    ("S2", 5, [0, 0, 1, 7]),
    ("S1", 3, [1, 1, 3, 4]),
    ("S2", 3, [2, 3, 4, 4]),
    ("S1", 1, [4, 4, 4, 4]),
    ("S2", 1, [4, 4, 4, 4]),
])
def test_quoted_exact_counts_match_the_artifact(selection, name, k_true, expected):
    """The counts every document repeats must come from the raw values."""

    observed = [_exact(selection, name, k_true, n) for n in N_GRID]
    assert observed == expected, (name, k_true, observed)


@pytest.mark.parametrize("name,expected", [
    ("S1", [2.62, 3.00, 4.50, 5.00]),
    ("S2", [1.75, 3.25, 3.62, 4.88]),
])
def test_quoted_mean_selected_k_matches_the_artifact(selection, name, expected):
    observed = [round(_mean(selection, name, 5, n), 2) for n in N_GRID]
    assert observed == expected, (name, observed)


@pytest.mark.parametrize("name,expected", [("S1", 39), ("S2", 37), ("S3", 3)])
def test_quoted_totals_match_the_artifact(selection, name, expected):
    keys = [k for k in selection if k[0] == name]
    assert sum(1 for k in keys if selection[k] == k[1]) == expected


def test_quoted_criterion_agreement_matches_the_artifact(selection):
    cells = sorted({(k[1], k[2], k[3]) for k in selection})
    agree = {}
    for a, b in (("S1", "S2"), ("S1", "S3"), ("S2", "S3")):
        agree[f"{a}-{b}"] = sum(1 for c in cells
                                if selection[(a, *c)] == selection[(b, *c)])
    assert agree == {"S1-S2": 44, "S1-S3": 2, "S2-S3": 0}, agree
    assert sum(1 for c in cells
               if len({selection[(n, *c)] for n in CRITERIA}) == 1) == 0

    # The report must state the SAME pairwise counts it was generated from.
    import re

    line = next(l for l in _text(RESULTS).splitlines()
                if "対ごとの一致セル数" in l)
    for pair, value in (("S1 vs S2", 44), ("S1 vs S3", 2), ("S2 vs S3", 0)):
        match = re.search(re.escape(pair) + r"\s+(\d+)", line)
        assert match is not None, (pair, line)
        assert int(match.group(1)) == value, (pair, match.group(1), value)


def test_the_exact_count_is_not_monotone_and_the_documents_say_so(selection):
    """The n=75 dip is the reason no convergence claim is allowed."""

    counts = [_exact(selection, "S1", 5, n) for n in N_GRID]
    assert counts[1] < counts[0], "the dip at n=75 is the fact being documented"
    for path in (RESULTS, TEACHER, HANDOFF):
        assert "単調で" in _text(path), path.name


def test_gram_is_non_psd_in_every_cell_as_documented():
    gram = _read_csv(RUN_DIR / "gram_spectrum.csv")
    assert len(gram) == 64
    negative = [r for r in gram if float(r["min_eigenvalue"]) < 0]
    assert len(negative) == 64
    assert all(int(r["unthresholded_rank"]) == 15 for r in gram)
    for path in (RESULTS, HANDOFF):
        assert "PSD" in _text(path), path.name


def test_generator_invariants_hold_and_are_documented():
    prov = _read_csv(RUN_DIR / "generator_provenance.csv")
    assert len(prov) == 64
    for row in prov:
        assert int(row["F_rank"]) == int(row["K_TRUE"])
        assert row["normalization_policy"] == "none"
        assert row["link_policy"] == "canonical_no_clipping_fail_fast"
        assert row["generator_version"] == "canonical-clean-v1"
        assert abs(float(row["mean_f_row_norm_sq"]) - 0.5) < 1e-9
        assert abs(float(row["w_true"]) ** 2 * int(row["K_TRUE"]) - 3.0) < 1e-9


# ------------------------------------------------------------- claim rules

PROHIBITION_HEADERS = (
    "NOT ALLOWED", "書いてはいけない", "書けない", "禁止",
    "チェックリスト", "答えられない",
)
DENIAL_MARKERS = (
    "ではない", "ではありません", "呼ばない", "NOT", "not ", "誤り",
    "使えない", "正当化されない", "近似するのは", "と書いていないか",
)


def _prohibition_lines(text: str) -> set[int]:
    """Line numbers inside a section that LISTS forbidden claims.

    A forbidden-claim list must be able to quote the forbidden claim verbatim;
    otherwise the rule could not be stated.  Everywhere else the phrase must
    carry an explicit denial.
    """

    inside, out = False, set()
    for index, line in enumerate(text.splitlines()):
        if line.lstrip().startswith("#"):
            inside = any(header in line for header in PROHIBITION_HEADERS)
        elif any(header in line for header in PROHIBITION_HEADERS):
            inside = True
        elif inside and line.strip() == "" :
            pass
        if inside:
            out.add(index)
    return out


@pytest.mark.parametrize("path", [
    RESULTS, TEACHER, QUESTIONS, HANDOFF, THEORY_MAP, STORYLINE, OUTLINE,
    INVENTORY, REAL_APP,
], ids=lambda p: p.name)
def test_documents_never_call_the_q_based_criterion_a_schwarz_bic(path):
    """KI-010 / KI-019.

    The phrase may appear only in a denial, or inside a section that lists
    claims which must NOT be made -- such a list has to quote the claim.
    """

    text = _text(path)
    exempt = _prohibition_lines(text)
    for index, line in enumerate(text.splitlines()):
        if "Schwarz BIC" not in line or index in exempt:
            continue
        assert any(marker in line for marker in DENIAL_MARKERS), (path.name, line)


@pytest.mark.parametrize("path", [RESULTS, TEACHER, HANDOFF, REGISTRY],
                         ids=lambda p: p.name)
def test_documents_state_that_agreement_is_with_k_true_not_k_star(path):
    """Every document quoting a recovery count must carry this distinction."""

    text = _text(path)
    assert "K_TRUE" in text and "K*" in text, path.name
    assert any(marker in text for marker in
               ("`K*` との一致ではない", "`K*` との一致ではありません",
                "`K*` とのものではない", "K* との一致ではない",
                "K* との一致ではありません")), path.name


@pytest.mark.parametrize("path", [RESULTS, TEACHER, QUESTIONS, HANDOFF],
                         ids=lambda p: p.name)
def test_documents_flag_the_k_true_1_floor_effect(path):
    """K_TRUE=1 scoring 4/4 must never be presented as a success."""

    assert "下限効果" in _text(path), path.name


@pytest.mark.parametrize("path", [RESULTS, TEACHER, QUESTIONS, HANDOFF],
                         ids=lambda p: p.name)
def test_documents_separate_s3_from_the_source_paper(path):
    text = _text(path)
    assert "S3" in text or "plug-in" in text, path.name
    assert any(marker in text for marker in
               ("原論文 Eq.(26) ではない", "原論文の基準ではない",
                "原論文 Eq.(26)ではない", "元論文の基準そのものではありません",
                "元論文の Eq.(26) そのものではありません",
                "元論文の基準の失敗と読んではいけません",
                "原論文の基準の失敗と読んではならない")), path.name


@pytest.mark.parametrize("path", [RESULTS, TEACHER, HANDOFF, STORYLINE, OUTLINE],
                         ids=lambda p: p.name)
def test_documents_record_that_consistency_is_not_proven(path):
    text = _text(path)
    assert any(marker in text for marker in
               ("一致性", "consistency")), path.name
    assert any(marker in text for marker in
               ("未解決", "証明していない", "ではない", "NOT")), path.name


def test_bernoulli_y_gap_is_stated_where_it_matters():
    """The theory's strong results are Gaussian-Y; the experiment is Bernoulli-Y."""

    for path in (RESULTS, TEACHER, HANDOFF, STORYLINE, THEORY_MAP):
        text = _text(path)
        assert "Bernoulli-Y" in text, path.name
        assert any(marker in text for marker in ("U2", "未解決", "UNRESOLVED")), path.name


def test_known_issues_registers_the_new_findings():
    text = _text(KNOWN_ISSUES)
    assert "KI-020" in text and "KI-021" in text
    assert "canonical-clean-v1" in text


# Numbers that only this experiment produces.  A document containing one of
# these is quoting the clean true-K sweep and must name its lineage.
EXPERIMENT_FINGERPRINTS = (
    "39/64", "37/64", "3/64", "44/64", "2.62", "4.88", "896",
    "2/8 | **0/8**",
)
# NOTE: the generator name "canonical-clean-v1" is deliberately NOT a
# fingerprint.  The pre-registration spec names the generator but predates
# and quotes no result, so requiring a lineage label there would be noise.

# Every markdown document this session added or touched, discovered from disk
# rather than hand-listed.  An earlier version of this test iterated over a
# hand-picked tuple that happened to exclude the two documents which violated
# the rule -- a test written to pass.  Scope is now derived, so a new document
# cannot be omitted by forgetting to add it.
def _session_documents() -> list[Path]:
    roots = (REPORTS / "identifiability", REPORTS / "thesis")
    found = [p for root in roots if root.is_dir()
             for p in sorted(root.glob("*.md"))]
    return found + [RESEARCH_MASTER, REGISTRY]


def _quotes_the_experiment(path: Path) -> bool:
    text = _text(path)
    return any(mark in text for mark in EXPERIMENT_FINGERPRINTS)


def test_the_lineage_check_covers_every_document_that_quotes_the_experiment():
    """B2: the scope of the lineage rule must be derived, never hand-picked.

    This test exists because the previous version of the lineage test iterated
    over four hand-listed documents, and the documents it omitted were exactly
    the ones violating the rule.
    """

    quoting = [p for p in _session_documents() if _quotes_the_experiment(p)]
    # The 2026-09-04 teacher draft predates the results and must not be forced
    # to carry them; everything else that quotes the numbers is in scope.
    quoting = [p for p in quoting
               if p.name != "teacher_discussion_summary_20260904.md"]
    assert len(quoting) >= 8, [p.name for p in quoting]
    for path in (RESULTS, HANDOFF, REGISTRY, RESEARCH_MASTER, TEACHER,
                 QUESTIONS, STORYLINE, OUTLINE, INVENTORY, REAL_APP):
        if _quotes_the_experiment(path):
            assert path in quoting, path.name


def test_lineage_is_named_wherever_the_experiment_is_quoted():
    """CLAUDE.md section 3 / KI-002: numbers must carry their lineage.

    Scope is DISCOVERED (every session markdown that quotes an experiment
    fingerprint), not hand-listed.
    """

    offenders = []
    for path in _session_documents():
        if path.name == "teacher_discussion_summary_20260904.md":
            continue                       # pre-results draft, superseded
        if not _quotes_the_experiment(path):
            continue
        text = _text(path)
        if not any(marker in text for marker in
                   ("本文採用不可", "lineage E", "experimental prototype")):
            offenders.append(path.name)
    assert not offenders, offenders
