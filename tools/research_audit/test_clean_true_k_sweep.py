"""Tests for the clean true-K sweep runner and its independent auditor.

These tests execute NO EM and never touch
``expfam/results/k_selection/clean_true_k_asymptotics_20260904``.  Every
artifact used here is a synthetic fixture built in ``tmp_path``.

The point of the tampering tests is that an auditor which cannot fail is not an
auditor: each one takes a valid fixture, introduces exactly one defect that a
dishonest or broken run would produce, and requires the auditor to reject it.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_EXPERIMENTAL = _HERE.parents[1] / "expfam" / "src" / "experimental"
for _path in (str(_HERE), str(_EXPERIMENTAL)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import audit_clean_true_k_sweep as A          # noqa: E402
import run_clean_true_k_sweep as R            # noqa: E402


# ---------------------------------------------------------------------------
# runner-side structural tests (no EM)
# ---------------------------------------------------------------------------

def test_protocol_hash_is_stable_and_matches_the_auditor():
    """The auditor restates the hash independently; they must agree."""

    assert R.protocol_hash() == A.EXPECTED_PROTOCOL_HASH


def test_auditor_does_not_import_the_runner():
    """An auditor that imported the runner would certify a mutated runner."""

    import ast

    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "run_clean_true_k_sweep" not in imported, imported
    assert not any("data_generator" in name for name in imported), imported


def test_budget_is_exactly_896_fits_over_64_cells():
    assert R.expected_fit_count() == 896
    assert R.expected_cell_count() == 64
    assert A.EXPECTED_FITS == 896
    assert A.EXPECTED_CELLS == 64


def test_manifest_is_complete_and_seeds_are_unique():
    manifest = R.build_manifest()
    assert len(manifest) == 896
    assert [r["fit_index"] for r in manifest] == list(range(1, 897))
    seeds = [r["model_seed"] for r in manifest]
    assert len(set(seeds)) == len(seeds)


def test_every_cell_carries_the_full_candidate_grid():
    manifest = R.build_manifest()
    by_cell: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for row in manifest:
        by_cell.setdefault(
            (row["K_TRUE"], row["n"], row["replicate"]), set()
        ).add((row["K"], row["start"]))
    wanted = {(k, s) for k in R.CANDIDATE_K for s in R.STARTS}
    assert len(by_cell) == 64
    for cell, keys in by_cell.items():
        assert keys == wanted, cell


def test_k_true_5_is_the_primary_focus():
    """The pre-registered tier gives K_TRUE=5 twice the replicates."""

    assert R.REPLICATES_BY_K_TRUE == {1: 4, 3: 4, 5: 8}
    assert R.TIER == "A"


def test_legacy_lineage_is_never_used():
    """The word may appear in prose; the VALUE must never be passed."""

    import ast

    assert R.NUMERICS_MODE == "consistent"
    source = Path(R.__file__).read_text(encoding="utf-8")
    for forbidden in ('numerics_mode="legacy"', "numerics_mode='legacy'"):
        assert forbidden not in source, forbidden

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "numerics_mode":
                    assert isinstance(keyword.value, ast.Name), ast.dump(keyword.value)
                    assert keyword.value.id == "NUMERICS_MODE", keyword.value.id


def test_masks_partition_the_upper_triangle_and_are_reproducible():
    for n in R.N_GRID:
        train, test = R.build_masks(n, 12345)
        again, _ = R.build_masks(n, 12345)
        assert np.array_equal(train, again)
        pairs = n * (n - 1) // 2
        assert int(np.triu(train, 1).sum()) + int(np.triu(test, 1).sum()) == pairs
        assert int(np.triu(test, 1).sum()) == int(round(0.20 * pairs))
        assert not np.any(train & test)
        assert np.array_equal(test, test.T)
        assert not test.diagonal().any()
        assert not train.diagonal().any()


def test_different_split_seeds_give_different_masks():
    first, _ = R.build_masks(75, 1)
    second, _ = R.build_masks(75, 2)
    assert not np.array_equal(first, second)


def test_selector_is_the_frozen_phase7e_rule():
    means = {1: -0.7, 2: -0.5, 3: -0.5, 4: -0.9, 5: -1.0, 6: -1.1, 7: -1.2}
    chosen, ties = R.select_k(means)
    assert chosen == 2 and ties == [2, 3], "ties must resolve to the smallest K"
    means[3] = -0.5 - 1e-9
    chosen, ties = R.select_k(means)
    assert chosen == 2 and ties == [2], "1e-12 is roundoff protection, not equivalence"


def test_seed_rules_agree_between_runner_and_auditor():
    for k in R.K_TRUE_GRID:
        for n in R.N_GRID:
            for rep in (1, 2):
                assert R.data_seed(k, n, rep) == A.data_seed(k, n, rep)
                assert R.split_seed(k, n, rep) == A.split_seed(k, n, rep)
                for k_est in R.CANDIDATE_K:
                    for start in R.STARTS:
                        assert (R.model_seed(k, n, rep, k_est, start)
                                == A.model_seed(k, n, rep, k_est, start))


def test_gram_spectrum_selects_no_k():
    rng = np.random.default_rng(0)
    x = rng.poisson(2.0, size=(500, 6)).astype(np.float64)
    spectrum = R.poisson_x_gram_spectrum(x)
    assert spectrum["status"] == "ok"
    assert "selected_k" not in spectrum
    assert len(spectrum["eigenvalues"]) == 6


def test_gram_spectrum_reports_undefined_rather_than_guessing():
    x = np.zeros((10, 3))
    assert R.poisson_x_gram_spectrum(x)["status"] == "undefined_zero_column_mean"


def test_production_refuses_without_the_explicit_flags():
    with pytest.raises(R.SweepStop, match="--allow-em"):
        R.run_production(allow_em=False, confirm=True)
    with pytest.raises(R.SweepStop, match="confirm"):
        R.run_production(allow_em=True, confirm=False)


def test_runner_has_no_resume_or_retry_path():
    source = Path(R.__file__).read_text(encoding="utf-8")
    for forbidden in ("--resume", "resume_from", "allow_resume", "skip_completed",
                      "exist_ok=True"):
        assert forbidden not in source, forbidden
    assert "refusing to overwrite or resume" in source


# ---------------------------------------------------------------------------
# fixture builder
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_valid_fixture(root: Path) -> Path:
    """A structurally valid artifact set with fabricated (not fitted) numbers."""

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    manifest = R.build_manifest()

    R.write_json(root / "protocol.json", {
        "protocol": R.frozen_protocol(), "protocol_hash": R.protocol_hash(),
        "protocol_version": R.PROTOCOL_VERSION,
        "artifact_version": R.ARTIFACT_VERSION,
        "experiment_id": R.EXPERIMENT_ID,
        "expected_fits": R.expected_fit_count(),
        "expected_cells": R.expected_cell_count(),
        "run_code_sha": "0" * 40,
        "working_tree_clean_before_execution": True,
        "environment": R.environment(),
        "failure_policy": ["stop_immediately"],
    })
    R.write_csv(root / "manifest.csv",
                ["fit_index", "K_TRUE", "n", "replicate", "K", "start",
                 "data_seed", "split_seed", "model_seed"], manifest)

    fits, cell_scores = [], {}
    for row in manifest:
        k_true, n, rep = row["K_TRUE"], row["n"], row["replicate"]
        k_est, start = row["K"], row["start"]
        s1 = -0.5 - 0.02 * abs(k_est - k_true) + rng.normal(0, 0.001)
        s2 = 1000.0 + 30.0 * k_est + rng.normal(0, 1)
        s3 = 900.0 + 25.0 * k_est + rng.normal(0, 1)
        fits.append({
            "fit_index": row["fit_index"], "k_true": k_true, "n": n,
            "replicate": rep, "k_est": k_est, "start": start,
            "data_seed": row["data_seed"], "split_seed": row["split_seed"],
            "model_seed": row["model_seed"],
            "heldout_mean_log_score": s1, "q_strict": -500.0,
            "s2_q_based": s2, "s3_plugin_conditional": s3,
            "num_params": 15 * k_est, "nan_occurred": False, "nan_count": 0,
            "q_bic_failed": False, "failure_reason": "", "runtime_s": 1.0,
        })
        bucket = cell_scores.setdefault((k_true, n, rep),
                                        {"S1": {}, "S2": {}, "S3": {}})
        for name, value in (("S1", s1), ("S2", -s2), ("S3", -s3)):
            bucket[name].setdefault(k_est, []).append(value)
    R.write_csv(root / "fit_results.csv", list(fits[0].keys()), fits)

    provenance, gram = [], []
    for cell in R.build_cells():
        pairs = cell.n * (cell.n - 1) // 2
        n_test = int(round(0.20 * pairs))
        w_true = R.w_for_matched_y_signal(R.W_REF, k=cell.k_true, k_ref=R.K_REF)
        provenance.append({
            "K_TRUE": cell.k_true, "n": cell.n, "replicate": cell.replicate,
            "data_seed": R.data_seed(cell.k_true, cell.n, cell.replicate),
            "split_seed": R.split_seed(cell.k_true, cell.n, cell.replicate),
            "generator_version": R.GENERATOR_VERSION, "F_rank": cell.k_true,
            "f_scale": 1.0, "mean_f_row_norm_sq": 0.5,
            "w0_true": R.W0_TRUE, "w_true": w_true,
            "link_policy": "canonical_no_clipping_fail_fast",
            "normalization_policy": "none",
            "x_mean": 1.3, "x_max": 30.0, "y_density": 0.33,
            "n_train_pairs": pairs - n_test, "n_test_pairs": n_test,
            "train_mask_hash": f"tr{cell.k_true}-{cell.n}-{cell.replicate}",
            "test_mask_hash": f"te{cell.k_true}-{cell.n}-{cell.replicate}",
        })
        gram.append({
            "K_TRUE": cell.k_true, "n": cell.n, "replicate": cell.replicate,
            "status": "ok", "unthresholded_rank": 15, "min_eigenvalue": -0.5,
            "eigenvalues": json.dumps([1.0] * 15),
            "gap_ratios": json.dumps([1.0] * 14),
        })
    R.write_csv(root / "generator_provenance.csv",
                list(provenance[0].keys()), provenance)
    R.write_csv(root / "gram_spectrum.csv", list(gram[0].keys()), gram)

    selection = []
    for (k_true, n, rep), by_criterion in cell_scores.items():
        for name in ("S1", "S2", "S3"):
            means = {k: float(np.mean(v)) for k, v in by_criterion[name].items()}
            chosen, ties = R.select_k(means)
            error = chosen - k_true
            selection.append({
                "criterion": name, "K_TRUE": k_true, "n": n, "replicate": rep,
                "selected_k": chosen, "tie_candidates": json.dumps(ties),
                "signed_error": error, "abs_error": abs(error),
                "label": ("exact" if error == 0 else
                          ("over" if error > 0 else "under")),
                "best_mean": means[chosen],
                "mean_scores": json.dumps({str(k): v for k, v in means.items()}),
            })
    R.write_csv(root / "selection_matrix.csv",
                list(selection[0].keys()), selection)
    R.write_json(root / "summary.json", R.build_summary(selection, fits))
    R.write_json(root / "runinfo.json", {
        "experiment_id": R.EXPERIMENT_ID, "protocol_hash": R.protocol_hash(),
        "attempted_fit_count": 896, "completed_fit_count": 896,
        "retry_count": 0, "replacement_fits_executed": 0,
        "seed_rescue_count": 0, "tolerance_relaxations": 0, "resumed": False,
    })
    return root


@pytest.fixture(scope="module")
def valid_fixture(tmp_path_factory):
    return build_valid_fixture(tmp_path_factory.mktemp("valid") / "run")


def _tampered(valid: Path, tmp_path: Path, mutate) -> Path:
    target = tmp_path / "run"
    shutil.copytree(valid, target)
    mutate(target)
    return target


# ---------------------------------------------------------------------------
# the auditor must PASS a valid set and FAIL every tampered one
# ---------------------------------------------------------------------------

def test_auditor_passes_a_valid_artifact_set(valid_fixture):
    report = A.audit(valid_fixture)
    assert report["verdict"] == "PASS", report["findings"]
    assert report["blocker_count"] == 0 and report["high_count"] == 0
    assert report["stats"]["fits"] == 896
    assert report["stats"]["cells"] == 64
    assert report["stats"]["selection_rows"] == 192


def _mutate_selection(target: Path) -> None:
    rows = _read_csv(target / "selection_matrix.csv")
    rows[0]["selected_k"] = str(int(rows[0]["selected_k"]) + 1)
    _write_csv(target / "selection_matrix.csv", rows)


def _mutate_drop_fit(target: Path) -> None:
    rows = _read_csv(target / "fit_results.csv")
    _write_csv(target / "fit_results.csv", rows[:-1])


def _mutate_seed(target: Path) -> None:
    rows = _read_csv(target / "fit_results.csv")
    rows[10]["model_seed"] = str(int(rows[10]["model_seed"]) + 7)
    _write_csv(target / "fit_results.csv", rows)


def _mutate_legacy(target: Path) -> None:
    payload = json.loads((target / "protocol.json").read_text(encoding="utf-8"))
    payload["protocol"]["numerics_mode"] = "legacy"
    (target / "protocol.json").write_text(json.dumps(payload), encoding="utf-8")


def _mutate_normalisation(target: Path) -> None:
    rows = _read_csv(target / "generator_provenance.csv")
    rows[3]["normalization_policy"] = "zscore"
    _write_csv(target / "generator_provenance.csv", rows)


def _mutate_clipping(target: Path) -> None:
    rows = _read_csv(target / "generator_provenance.csv")
    rows[2]["link_policy"] = "clipped"
    _write_csv(target / "generator_provenance.csv", rows)


def _mutate_rank(target: Path) -> None:
    rows = _read_csv(target / "generator_provenance.csv")
    rows[0]["F_rank"] = "0"
    _write_csv(target / "generator_provenance.csv", rows)


def _mutate_retry(target: Path) -> None:
    payload = json.loads((target / "runinfo.json").read_text(encoding="utf-8"))
    payload["retry_count"] = 3
    (target / "runinfo.json").write_text(json.dumps(payload), encoding="utf-8")


def _mutate_resume(target: Path) -> None:
    payload = json.loads((target / "runinfo.json").read_text(encoding="utf-8"))
    payload["resumed"] = True
    (target / "runinfo.json").write_text(json.dumps(payload), encoding="utf-8")


def _mutate_failure_marker(target: Path) -> None:
    (target / "failure.json").write_text('{"status": "FAILED"}', encoding="utf-8")


def _mutate_signal(target: Path) -> None:
    rows = _read_csv(target / "generator_provenance.csv")
    rows[0]["mean_f_row_norm_sq"] = "2.0"
    _write_csv(target / "generator_provenance.csv", rows)


def _mutate_summary(target: Path) -> None:
    payload = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    payload["criteria"]["S1"]["total_exact"] = 999
    (target / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _mutate_protocol_hash(target: Path) -> None:
    payload = json.loads((target / "protocol.json").read_text(encoding="utf-8"))
    payload["protocol_hash"] = "f" * 64
    (target / "protocol.json").write_text(json.dumps(payload), encoding="utf-8")


def _mutate_nonfinite(target: Path) -> None:
    rows = _read_csv(target / "fit_results.csv")
    rows[5]["heldout_mean_log_score"] = "nan"
    _write_csv(target / "fit_results.csv", rows)


def _mutate_missing_file(target: Path) -> None:
    (target / "gram_spectrum.csv").unlink()


ATTACKS = {
    "tampered_selected_k": (_mutate_selection, "selection_recomputation"),
    "dropped_one_fit": (_mutate_drop_fit, "fit_row_count"),
    "seed_rescue": (_mutate_seed, "seed_rule"),
    "legacy_lineage": (_mutate_legacy, "numerics_mode"),
    "generator_normalisation": (_mutate_normalisation, "generator_normalization"),
    "generator_clipping": (_mutate_clipping, "generator_link"),
    "degenerate_F": (_mutate_rank, "generator_rank"),
    "retries_declared": (_mutate_retry, "retry_count"),
    "resumed_run": (_mutate_resume, "resumed"),
    "failure_marker_present": (_mutate_failure_marker, "failure_marker"),
    "broken_signal_matching": (_mutate_signal, "signal_matching_x"),
    "inflated_summary": (_mutate_summary, "summary_exact_count"),
    "wrong_protocol_hash": (_mutate_protocol_hash, "protocol_hash"),
    "nonfinite_criterion": (_mutate_nonfinite, "criterion_finite"),
    "missing_required_file": (_mutate_missing_file, "required_file"),
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_auditor_rejects_a_tampered_artifact_set(name, valid_fixture, tmp_path):
    """An auditor that cannot fail is not an auditor."""

    mutate, expected_check = ATTACKS[name]
    target = _tampered(valid_fixture, tmp_path, mutate)
    report = A.audit(target)
    assert report["verdict"] == "FAIL", (name, report["findings"])
    checks = {f["check"] for f in report["findings"]
              if f["severity"] in ("BLOCKER", "HIGH")}
    assert expected_check in checks, (name, sorted(checks))


def test_auditor_never_writes_into_the_run_directory(valid_fixture, tmp_path):
    target = _tampered(valid_fixture, tmp_path, lambda _p: None)
    before = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
    A.audit(target)
    after = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
    assert before == after
