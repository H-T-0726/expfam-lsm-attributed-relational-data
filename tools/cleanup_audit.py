"""Dry-run cleanup audit for repository files.

This script does not delete, move, rename, or edit repository artifacts.
It scans the current tree and writes a CSV report of cleanup candidates to:

    reports/cleanup_audit/cleanup_candidates_20260707.csv
"""

from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "reports" / "cleanup_audit" / "cleanup_candidates_20260707.csv"

CSV_COLUMNS = [
    "path",
    "category",
    "reason",
    "risk_level",
    "referenced_by",
    "recommended_action",
]

CATEGORY_PRIORITY = {
    "KEEP": 1,
    "ARCHIVE_CANDIDATE": 2,
    "REVIEW_REQUIRED": 3,
    "DELETE_CANDIDATE": 4,
}

REFERENCE_EXTENSIONS = {
    ".md",
    ".py",
    ".m",
    ".txt",
    ".rst",
    ".tex",
    ".yml",
    ".yaml",
    ".json",
}

UNREFERENCED_EXTENSIONS = {
    ".csv",
    ".png",
    ".pdf",
    ".txt",
    ".md",
    ".npy",
    ".zip",
}

IGNORED_DIRS = {
    ".git",
    ".claude",
    ".codex",
    ".agents",
}

KNOWN_ROOT_FILES = {
    ".gitignore",
    "CLAUDE.md",
    "CLEANUP_MANIFEST.md",
    "EXPERIMENT_REGISTRY.md",
    "KNOWN_ISSUES.md",
    "README.md",
    "RESEARCH_MASTER.md",
    "START_HERE.md",
    "conference_submission_final_draft.md",
}

KEEP_GLOBS = [
    "CLAUDE.md",
    "CLEANUP_MANIFEST.md",
    "EXPERIMENT_REGISTRY.md",
    "KNOWN_ISSUES.md",
    "README.md",
    "RESEARCH_MASTER.md",
    "START_HERE.md",
    "figures/fig1a_n_sweep_color.*",
    "figures/fig1b_misspecification_color.*",
    "figures/figure_color_split_report.md",
    "Mato Lab Program/**",
    "paper/**",
    "reproduction/**",
    "docs/**",
    "docs_for_notebooklm/**",
    "reports/*.md",
    "reports/movielens_colike_clean/**",
    "expfam/data/cora/**",
    "expfam/data/movielens_pilot/**",
    "expfam/results/exp_scenario_*.csv",
    "expfam/results/exp1_full_*.csv",
    "expfam/results/exp2_bic_*.csv",
    "expfam/results/exp2_bic_log.txt",
    "expfam/results/fig_scenario_*",
    "expfam/results/fig1_*",
    "expfam/results/fig2_*",
    "expfam/results/GEMINI_REPORT_*.md",
    "expfam/results/RESEARCH_PROPOSAL_DUAL_EXPFAM.md",
    "expfam/results/wine_dual_results.csv",
    "expfam/results/distribution_mismatch_fixed/**",
    "expfam/results/fixed_official/**",
    "expfam/results/half_factor_check/**",
    "expfam/figures/fixed_official/**",
    "expfam/figures/distribution_mismatch_fixed/**",
    "expfam/results/real_data/real_data_experiment_summary.csv",
    "expfam/results/real_data/wine_fixed_pilot/**",
    "expfam/results/real_data/wine_old05_audit/**",
    "expfam/results/real_data/wine_clean/**",
    "expfam/results/real_data/cora_balanced_subset_pilot/**",
    "expfam/results/real_data/cora_balanced_k_sweep/**",
    "expfam/results/real_data/cora_heldout_link_prediction/**",
    "expfam/results/real_data/cora_scaling_heldout/**",
    "expfam/results/real_data/cora_clean/**",
    "expfam/results/real_data/movielens_data_prep/**",
    "expfam/results/real_data/movielens_poisson_pilot/**",
    "expfam/results/real_data/movielens_heldout_count/**",
    "expfam/results/real_data/movielens_bernoulli_t80_pilot/**",
    "expfam/results/real_data/movielens_colike_interpretation/**",
    "expfam/results/real_data/movielens_colike_clean/**",
    "expfam/results/real_data/movielens_final_clean/**",
    "expfam/results/real_data/common_reconstruction_eval/**",
    "expfam/figures/real_data/wine_fixed_pilot/**",
    "expfam/figures/real_data/wine_clean/**",
    "expfam/figures/real_data/cora_balanced_subset_pilot/**",
    "expfam/figures/real_data/cora_balanced_k_sweep/**",
    "expfam/figures/real_data/cora_heldout_link_prediction/**",
    "expfam/figures/real_data/cora_scaling_heldout/**",
    "expfam/figures/real_data/cora_clean/**",
    "expfam/figures/real_data/movielens_data_prep/**",
    "expfam/figures/real_data/movielens_poisson_pilot/**",
    "expfam/figures/real_data/movielens_heldout_count/**",
    "expfam/figures/real_data/movielens_bernoulli_t80_pilot/**",
    "expfam/figures/real_data/movielens_colike_interpretation/**",
    "expfam/figures/real_data/movielens_colike_clean/**",
    "expfam/figures/real_data/common_reconstruction_eval/**",
]

ARCHIVE_GLOBS = [
    "archive/notion_scripts/**",
    "archive/misc/**",
    "expfam/src/archive/**",
    "expfam/results/archive/**",
    "expfam/results/real_data/cora_subset_pilot/**",
    "expfam/figures/real_data/cora_subset_pilot/**",
]

REVIEW_GLOBS = [
    "expfam/results/real_data/movielens_colike_clean/**",
    "expfam/results/real_data/movielens_final_clean/**",
    "reports/movielens_colike_clean/**",
]


@dataclass(frozen=True)
class Row:
    path: str
    category: str
    reason: str
    risk_level: str
    referenced_by: str
    recommended_action: str


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def iter_paths() -> list[Path]:
    paths: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        relative_parts = path.relative_to(REPO_ROOT).parts
        if any(part in IGNORED_DIRS for part in relative_parts):
            continue
        if path == OUTPUT_PATH:
            continue
        paths.append(path)
    return paths


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def build_reference_index(files: list[Path]) -> dict[str, str]:
    reference_files = [
        path
        for path in files
        if path.is_file()
        and path.suffix.lower() in REFERENCE_EXTENSIONS
        and rel(path) != "tools/cleanup_audit.py"
    ]
    corpus = {rel(path): read_text(path) for path in reference_files}
    index: dict[str, str] = {}

    for target in files:
        if not target.is_file():
            continue
        target_rel = rel(target)
        target_slash = target_rel
        target_backslash = target_rel.replace("/", "\\")
        basename = target.name
        hits: list[str] = []
        for ref_path, text in corpus.items():
            if ref_path == target_rel:
                continue
            if target_slash in text or target_backslash in text or basename in text:
                hits.append(ref_path)
        index[target_rel] = ";".join(hits[:12]) if hits else ""

    return index


def mentioned_in_manifest(path: str, manifest_text: str) -> bool:
    basename = Path(path).name
    return path in manifest_text or path.replace("/", "\\") in manifest_text or basename in manifest_text


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def is_suspicious_root_file(path: Path) -> bool:
    if path.parent != REPO_ROOT or not path.is_file():
        return False
    if path.name in KNOWN_ROOT_FILES:
        return False
    if path.suffix.lower() in {".md", ".txt", ".pdf"}:
        return False
    return True


def add_row(rows: dict[str, Row], row: Row) -> None:
    existing = rows.get(row.path)
    if existing is None or CATEGORY_PRIORITY[row.category] > CATEGORY_PRIORITY[existing.category]:
        rows[row.path] = row


def audit() -> list[Row]:
    all_paths = iter_paths()
    files = [path for path in all_paths if path.is_file()]
    dirs = [path for path in all_paths if path.is_dir()]
    refs = build_reference_index(files)

    manifest_text = read_text(REPO_ROOT / "CLEANUP_MANIFEST.md")
    registry_text = read_text(REPO_ROOT / "EXPERIMENT_REGISTRY.md")
    keep_source_text = manifest_text + "\n" + registry_text

    rows: dict[str, Row] = {}

    for path in dirs:
        path_rel = rel(path)
        if path.name == "__pycache__":
            add_row(
                rows,
                Row(
                    path_rel,
                    "DELETE_CANDIDATE",
                    "Python bytecode cache directory; generated artifact, not research data.",
                    "low",
                    "N/A",
                    "Human can delete after confirming it is untracked; no script action taken.",
                ),
            )

    for path in files:
        path_rel = rel(path)
        reference_hits = refs.get(path_rel, "")

        if path.suffix == ".pyc":
            add_row(
                rows,
                Row(
                    path_rel,
                    "DELETE_CANDIDATE",
                    "Compiled Python bytecode; generated artifact.",
                    "low",
                    reference_hits or "N/A",
                    "Human can delete after confirming it is untracked; no script action taken.",
                ),
            )

        if is_suspicious_root_file(path):
            add_row(
                rows,
                Row(
                    path_rel,
                    "REVIEW_REQUIRED",
                    "Unexpected file at repository root; possible accidental command output.",
                    "low",
                    reference_hits or "unreferenced",
                    "Inspect contents before deciding whether deletion is safe.",
                ),
            )

        if (
            path.suffix.lower() in UNREFERENCED_EXTENSIONS
            and not reference_hits
            and not matches_any(path_rel, KEEP_GLOBS)
            and not matches_any(path_rel, ARCHIVE_GLOBS)
            and not mentioned_in_manifest(path_rel, keep_source_text)
            and path_rel != "reports/cleanup_audit/cleanup_candidates_20260707.csv"
        ):
            add_row(
                rows,
                Row(
                    path_rel,
                    "REVIEW_REQUIRED",
                    "No reference found from markdown or source/script files by path or basename search.",
                    "medium",
                    "unreferenced",
                    "Review provenance manually before any delete/archive action.",
                ),
            )

    for path in files + dirs:
        path_rel = rel(path)
        reference_hits = refs.get(path_rel, "")

        if matches_any(path_rel, ARCHIVE_GLOBS):
            add_row(
                rows,
                Row(
                    path_rel,
                    "ARCHIVE_CANDIDATE",
                    "Listed in cleanup manifest as old/archive/pilot material or already under archive.",
                    "medium" if "cora_subset_pilot" in path_rel else "low",
                    reference_hits or ("CLEANUP_MANIFEST.md" if mentioned_in_manifest(path_rel, manifest_text) else "unreferenced"),
                    "Keep in place until human confirms references and approves archive policy.",
                ),
            )

        if matches_any(path_rel, REVIEW_GLOBS):
            add_row(
                rows,
                Row(
                    path_rel,
                    "REVIEW_REQUIRED",
                    "Cleanup manifest notes similar or colliding result/report folder names.",
                    "medium",
                    reference_hits or "CLEANUP_MANIFEST.md",
                    "Compare roles and references; do not rename, merge, move, or delete without approval.",
                ),
            )

        if matches_any(path_rel, KEEP_GLOBS) or mentioned_in_manifest(path_rel, keep_source_text):
            add_row(
                rows,
                Row(
                    path_rel,
                    "KEEP",
                    "Covered by EXPERIMENT_REGISTRY.md or CLEANUP_MANIFEST.md as keep/support/current material.",
                    "low",
                    reference_hits or ("EXPERIMENT_REGISTRY.md/CLEANUP_MANIFEST.md" if mentioned_in_manifest(path_rel, keep_source_text) else "manifest pattern"),
                    "Keep; not a cleanup target in this dry-run.",
                ),
            )

    return sorted(rows.values(), key=lambda row: (row.category, row.path))


def write_csv(rows: list[Row]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "category": row.category,
                    "reason": row.reason,
                    "risk_level": row.risk_level,
                    "referenced_by": row.referenced_by,
                    "recommended_action": row.recommended_action,
                }
            )


def main() -> None:
    rows = audit()
    write_csv(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.category] = counts.get(row.category, 0) + 1

    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}")
    print(f"Total rows: {len(rows)}")
    for category in ["DELETE_CANDIDATE", "ARCHIVE_CANDIDATE", "REVIEW_REQUIRED", "KEEP"]:
        print(f"{category}: {counts.get(category, 0)}")


if __name__ == "__main__":
    main()
