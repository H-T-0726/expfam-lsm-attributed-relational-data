"""Static and adversarial tests for the Phase 8b K_TRUE robustness harness.

No test in this module runs EM.  Fit-bearing paths are exercised only through
their authorization gates, which must refuse.  Artifact-level negative tests
build a synthetic run directory in a temp dir and break exactly one thing.

Test ids follow the implementation-plan contract (T01-T32, A01-A25).
"""

from __future__ import annotations

import csv
import dataclasses
import importlib
import json
import math
import pathlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT / "expfam" / "src"), str(ROOT / "expfam" / "src" / "experimental")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_k_true_robustness_sweep as H  # noqa: E402
import audit_k_true_robustness_sweep as A  # noqa: E402
from run_heldout_k_selection_pilot import HarnessStop  # noqa: E402


# ===========================================================================
# helpers
# ===========================================================================


@pytest.fixture(scope="module")
def anchors():
    return H.read_phase7e_anchor_masks()


def _write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def build_valid_run_dir(tmp_path: Path, estimand: str = "B") -> Path:
    """Contract-valid config-mode directory, written by the PRODUCTION writer."""

    run_dir = tmp_path / f"run_{estimand}"
    H.write_zero_em_artifacts(run_dir, estimand)
    return run_dir


# --- synthetic score artifacts for selection-mode audit --------------------
# Schema matches the future smoke/full fit_results contract: no new scientific
# schema is invented here.
FIT_RESULT_COLUMNS = (
    "estimand", "role", "K_TRUE", "replicate", "K", "start", SCORE_COLUMN := "heldout_mean_log_score",
)


def _synthetic_scores(estimand: str, winner: dict[tuple[int, int], int] | None = None):
    """Deterministic per-(K,start) scores whose argmax is a known K."""

    winner = winner or {}
    rows = []
    truth: dict[tuple[int, int], int] = {}
    role = H.resolve_role(estimand)
    for k_true in H.NEW_K_TRUE:
        for replicate in H.REPLICATES:
            best_k = winner.get((k_true, replicate), k_true)
            truth[(k_true, replicate)] = best_k
            for k in H.K_CANDIDATES:
                for start in H.START_LABELS:
                    score = -0.5 - 0.01 * abs(k - best_k) + 0.0001 * start
                    rows.append((estimand, role, k_true, replicate, k, start, score))
    return rows, truth


def build_selection_run_dir(tmp_path: Path, estimand: str = "B",
                            winner: dict[tuple[int, int], int] | None = None) -> Path:
    """Config-mode artifacts plus fit_results.csv and the integrated matrix."""

    run_dir = build_valid_run_dir(tmp_path, estimand)
    rows, truth = _synthetic_scores(estimand, winner)
    _write_csv(run_dir / "fit_results.csv", FIT_RESULT_COLUMNS, rows)
    _write_csv(run_dir / "k_true_selection_matrix.csv", H.SELECTION_MATRIX_COLUMNS,
               _matrix_rows(estimand, truth))
    return run_dir


def _matrix_rows(estimand: str, truth: dict[tuple[int, int], int]) -> list[tuple]:
    role = H.resolve_role(estimand)
    rows = list(H.build_selection_matrix_anchor_rows(estimand))
    for k_true in H.NEW_K_TRUE:
        for replicate in H.REPLICATES:
            selected = truth[(k_true, replicate)]
            signed = selected - k_true
            rows.append((
                estimand, role, k_true, replicate, selected, signed, abs(signed),
                H.selection_label(signed), H.LINEAGE_NEW, "deadbeef",
                f"expfam/results/k_selection/k_true_robustness_{estimand}_test",
            ))
    return rows


def _rows_of(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rewrite(path: Path, rows: list[dict[str, str]], header=None) -> None:
    header = header or list(rows[0])
    _write_csv(path, header, [[r[c] for c in header] for r in rows])


# ===========================================================================
# T01-T04 — estimand / w_true contract
# ===========================================================================


def test_T01_anchor_w_invariant_across_estimands():
    assert H.resolve_w_true("A", 3) == H.resolve_w_true("B", 3) == 1.5


def test_T02_option_b_algebraic_variance_invariant():
    values = {H.resolve_w_true("B", k) ** 2 * k for k in (1, 2, 3, 4, 5, 6, 7)}
    assert max(values) - min(values) <= 1e-9
    assert abs(values.pop() - H.W_REF ** 2 * H.K_REF) <= 1e-9


def test_T03_option_b_exact_formula():
    for k in H.NEW_K_TRUE:
        assert H.resolve_w_true("B", k) == H.W_REF * math.sqrt(H.K_REF / k)


def test_T04_option_a_fixed():
    assert {H.resolve_w_true("A", k) for k in H.NEW_K_TRUE} == {1.5}


def test_T04b_unknown_estimand_stops():
    with pytest.raises(HarnessStop):
        H.resolve_w_true("C", 3)


# ===========================================================================
# T05-T11 — manifest and seeds
# ===========================================================================


def test_T05_manifest_row_count():
    for estimand in ("A", "B"):
        assert len(H.build_manifest(estimand)) == 168
    assert H.EXPECTED_NEW_FITS == 336


def test_T06_manifest_key_set_and_order():
    rows = H.build_manifest("A")
    keys = [(r.k_true, r.replicate, r.k, r.start) for r in rows]
    assert keys == sorted(keys)
    assert set(keys) == {
        (kt, rp, k, s) for kt in H.NEW_K_TRUE for rp in H.REPLICATES
        for k in H.K_CANDIDATES for s in H.START_LABELS
    }


def test_T07_manifest_excludes_anchor_k_true():
    for estimand in ("A", "B"):
        assert all(r.k_true != H.ANCHOR_K_TRUE for r in H.build_manifest(estimand))


def test_T08_model_seeds_distinct_within_estimand():
    for estimand in ("A", "B"):
        seeds = [r.model_seed for r in H.build_manifest(estimand)]
        assert len(seeds) == len(set(seeds)) == 168


def test_T09_no_unintended_collision_with_phase7e():
    manifests = {e: H.build_manifest(e) for e in ("A", "B")}
    report = H.check_seed_collisions(manifests)
    assert report["unintended_collisions"] == []
    phase7e_data = {41000 + r for r in H.REPLICATES}
    phase7e_model = {43000 + r * 1000 + k * 10 + s
                     for r in H.REPLICATES for k in H.K_CANDIDATES for s in H.START_LABELS}
    for rows in manifests.values():
        assert not ({r.data_seed for r in rows} & phase7e_data)
        assert not ({r.model_seed for r in rows} & phase7e_model)


def test_T09b_split_seed_reuse_is_intentional():
    manifests = {e: H.build_manifest(e) for e in ("A", "B")}
    report = H.check_seed_collisions(manifests)
    assert report["phase7e_split_seed_reused"] == [42001, 42002, 42003]
    assert report["intentional_seed_reuse"] is True


def test_T10_role_separation_between_data_and_model_seeds():
    rows = H.build_manifest("A")
    assert not ({r.data_seed for r in rows} & {r.model_seed for r in rows})
    assert not ({r.data_seed for r in rows} & {r.split_seed for r in rows})
    assert not ({r.model_seed for r in rows} & {r.split_seed for r in rows})


def test_T11_seeds_are_deterministic():
    a = [(r.data_seed, r.split_seed, r.model_seed) for r in H.build_manifest("B")]
    b = [(r.data_seed, r.split_seed, r.model_seed) for r in H.build_manifest("B")]
    assert a == b


def test_T11b_crn_shares_data_and_model_seeds_across_estimands():
    """H2 = CRN: A and B correspond on data/model RNG by design."""

    a = H.build_manifest("A")
    b = H.build_manifest("B")
    assert [r.model_seed for r in a] == [r.model_seed for r in b]
    assert [r.data_seed for r in a] == [r.data_seed for r in b]


# ===========================================================================
# T12a-T12q — mask design, H2/H4 separation, schema
# ===========================================================================


def test_T12a_S_A_masks_differ_per_cell(monkeypatch):
    monkeypatch.setattr(H, "MASK_DESIGN", "S_A")
    hashes = {H.build_split_record(kt, r).split_mask_hash
              for kt in H.NEW_K_TRUE for r in H.REPLICATES}
    assert len(hashes) == len(H.NEW_K_TRUE) * len(H.REPLICATES)


def test_T12b_S_B_shares_only_among_new_k_true(monkeypatch, anchors):
    monkeypatch.setattr(H, "MASK_DESIGN", "S_B")
    for replicate in H.REPLICATES:
        group = {H.build_split_record(kt, replicate).split_mask_hash for kt in H.NEW_K_TRUE}
        assert len(group) == 1
        assert group.pop() != anchors[replicate].test_mask_hash


def test_T12c_S_C_matches_phase7e_anchor(anchors):
    for k_true in H.NEW_K_TRUE:
        for replicate in H.REPLICATES:
            record = H.build_split_record(k_true, replicate)
            assert record.split_mask_hash == anchors[replicate].test_mask_hash
            assert record.train_mask_hash == anchors[replicate].train_mask_hash


def test_T12d_mask_mismatch_raises_harness_stop(anchors):
    broken = dict(anchors)
    broken[1] = H.AnchorMask(1, "deadbeef", anchors[1].train_mask_hash, "fake")
    with pytest.raises(HarnessStop):
        H.run_mask_gate(anchors=broken, estimands=("B",))


def test_T12e_manifest_has_all_seven_provenance_fields():
    assert set(H.REQUIRED_MASK_PROVENANCE_FIELDS) <= set(H.MANIFEST_COLUMNS)
    assert set(H.REQUIRED_MASK_PROVENANCE_FIELDS) <= set(H.MASK_PROVENANCE_COLUMNS)
    assert len(H.REQUIRED_MASK_PROVENANCE_FIELDS) == 7


def test_T12f_expected_split_seed_takes_no_estimand():
    """H4 alone governs the split seed: no estimand argument, no H2 lookup."""

    import ast
    import inspect
    import textwrap

    params = list(inspect.signature(H.expected_split_seed).parameters)
    assert params == ["k_true", "replicate"]

    # Inspect the executable body only; the docstring legitimately explains
    # why RANDOM_DESIGN must NOT be consulted here.
    tree = ast.parse(textwrap.dedent(inspect.getsource(H.expected_split_seed)))
    func = tree.body[0]
    body = func.body[1:] if ast.get_docstring(func) else func.body
    names = {node.id for stmt in body for node in ast.walk(stmt) if isinstance(node, ast.Name)}
    names |= {node.attr for stmt in body for node in ast.walk(stmt)
              if isinstance(node, ast.Attribute)}
    assert "RANDOM_DESIGN" not in names
    assert "estimand" not in names
    assert "MASK_DESIGN" in names


def test_T12g_split_seed_invariant_to_random_design(monkeypatch):
    before = {(kt, r): H.expected_split_seed(kt, r) for kt in H.NEW_K_TRUE for r in H.REPLICATES}
    monkeypatch.setattr(H, "RANDOM_DESIGN", "INDEPENDENT")
    after = {(kt, r): H.expected_split_seed(kt, r) for kt in H.NEW_K_TRUE for r in H.REPLICATES}
    assert before == after


def test_T12h_independent_times_S_C_is_legal(monkeypatch, anchors):
    monkeypatch.setattr(H, "RANDOM_DESIGN", "INDEPENDENT")
    gates = H.run_mask_gate(anchors=anchors, estimands=("A", "B"))
    assert all(g.passed for g in gates)
    assert any(g.gate == "MC1" for g in gates)


def test_T12i_independent_separates_data_and_model_seeds(monkeypatch):
    monkeypatch.setattr(H, "RANDOM_DESIGN", "INDEPENDENT")
    assert H.expected_data_seed(1, 1, "A") != H.expected_data_seed(1, 1, "B")
    assert H.expected_model_seed(1, 1, 1, 1, "A") != H.expected_model_seed(1, 1, 1, 1, "B")


def test_T12j_random_design_and_mask_design_are_separate_fields():
    config = H.frozen_config()
    assert config["random_design"] == "CRN"
    assert config["mask_design"] == "S_C"
    assert "random_design" in config and "mask_design" in config


def test_T12k_split_mask_hash_is_hash_of_test_mask():
    record = H.build_split_record(1, 1)
    from run_heldout_k_selection_pilot import stable_array_hash

    assert record.split_mask_hash == stable_array_hash(record.test_mask)
    assert record.split_mask_hash != stable_array_hash(record.train_mask)
    assert H.canonical_hash_contract()["canonical_object"] == "test_mask"


def test_T12l_anchor_hashes_come_from_phase7e_columns(anchors):
    rows = _rows_of(H.PHASE7E_FIT_RESULTS)
    for replicate in H.REPLICATES:
        source = next(r for r in rows if int(r["replicate"]) == replicate)
        assert anchors[replicate].test_mask_hash == source["test_mask_hash"]
        assert anchors[replicate].train_mask_hash == source["train_mask_hash"]


def test_T12m_S_C_requires_both_test_and_train(anchors):
    """A test-only match must not pass."""

    tampered = dict(anchors)
    tampered[2] = H.AnchorMask(2, anchors[2].test_mask_hash, "not-the-train-hash", "fake")
    with pytest.raises(HarnessStop):
        H.run_mask_gate(anchors=tampered, estimands=("A",))


def test_T12n_mask_provenance_exactly_twelve(anchors):
    for estimand in ("A", "B"):
        rows = H.build_mask_provenance(estimand, anchors)
        assert len(rows) == 12
        keys = [(r.estimand, r.k_true, r.replicate) for r in rows]
        assert len(keys) == len(set(keys))


def test_T12o_diagnostics_exactly_twelve_and_no_anchor():
    rows = H.build_diagnostics("A")
    assert len(rows) == 12
    assert {r.k_true for r in rows} == set(H.NEW_K_TRUE)
    assert all(r.k_true != H.ANCHOR_K_TRUE for r in rows)


def test_T12p_mask_provenance_excludes_anchor_rows(anchors):
    rows = H.build_mask_provenance("B", anchors)
    assert all(r.k_true != H.ANCHOR_K_TRUE for r in rows)
    # anchor evidence is carried by reference, not by copying anchor rows
    assert all(r.anchor_mask_hash == anchors[r.replicate].test_mask_hash for r in rows)


def test_T12q_selection_matrix_has_eleven_columns(tmp_path):
    assert len(H.selection_matrix_columns()) == 11
    run_dir = build_selection_run_dir(tmp_path, "B")
    with (run_dir / "k_true_selection_matrix.csv").open(encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    header = len(lines[0].split(","))
    assert header == 11
    assert all(len(line.split(",")) == header for line in lines[1:])


# ===========================================================================
# T13-T21 — protocol identity, generator identities, gates
# ===========================================================================


def test_T13_make_pair_split_ignores_k_true():
    a = H.make_pair_split(H.N_NODES, H.TEST_RATIO, 42001)
    b = H.make_pair_split(H.N_NODES, H.TEST_RATIO, 42001)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    import inspect

    assert "k" not in inspect.signature(H.make_pair_split).parameters


def test_T14_score_config_hash_matches_phase7e():
    stored = _rows_of(H.PHASE7E_FIT_RESULTS)[0]["score_config_hash"]
    from run_heldout_k_selection_pilot import frozen_score_config, score_config_hash

    assert score_config_hash(frozen_score_config()) == stored


def test_T15_selector_is_imported_from_phase7e():
    import run_heldout_k_selection_pilot as P

    assert H.select_k_from_two_starts is P.select_k_from_two_starts


def test_T16_mean_squared_latent_norm_equals_k_true():
    for row in H.build_diagnostics("A"):
        assert abs(row.mean_sq_latent_norm - row.k_true) <= 1e-12


def test_T17_frobenius_norm_is_k_invariant():
    expected = H.N_FEATURES * (1.0 - H.UNIQ)
    for row in H.build_diagnostics("B"):
        assert abs(row.f_frobenius_sq - expected) <= 1e-9


def test_T18_k_true_one_boundary_structure():
    data = H._generate_cell("A", 1, 1)
    F = np.asarray(data["F"])
    assert F.shape == (H.N_FEATURES, 1)
    assert np.linalg.matrix_rank(F) == 1
    norms = np.linalg.norm(F, axis=1)
    assert np.allclose(norms, math.sqrt(1.0 - H.UNIQ))


def test_T19_score_invariant_under_sign_flip():
    data = H._generate_cell("A", 1, 1)
    Z = np.asarray(data["Z"])
    assert np.allclose(Z @ Z.T, (-Z) @ (-Z).T)


def test_T20_config_gate_fails_closed(anchors):
    broken = {r: H.AnchorMask(r, "x", "y", "fake") for r in H.REPLICATES}
    with pytest.raises(HarnessStop):
        H.run_mask_gate(anchors=broken)


def test_T21_diagnostics_never_block():
    result = H.run_record_diagnostics()
    assert result["record_only"] is True
    assert result["blocking"] is False
    assert result["em_fits_executed"] == 0
    # No diagnostic value appears in any gate name.
    gate_source = __import__("inspect").getsource(H.run_mask_gate) + \
        __import__("inspect").getsource(H.run_generator_gate)
    for token in ("sample_sd", "y_density", "conditional_entropy", "oracle_mean_log_score"):
        assert token not in gate_source


# ===========================================================================
# T22-T26 — CLI gates and no-EM boundary
# ===========================================================================


def test_T22_full_requires_allow_em():
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--confirm-k-true-sweep", "--estimand", "A"])
    assert "allow-em" in str(excinfo.value)


def test_T23_full_requires_confirm_flag():
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--estimand", "A"])
    assert "confirm-k-true-sweep" in str(excinfo.value)


def test_T23b_full_is_refused_even_with_every_flag(monkeypatch):
    """S3-D withdrew the stale S3-C approval, so --full is closed again."""

    reached = _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"])
    assert "not authorized" in str(excinfo.value)
    assert reached == [], "--full must not reach the production full workflow"


def test_T23c_smoke_and_canary_reach_only_the_guarded_workflow(monkeypatch):
    """Issue #55 committed the authorization, so the CLI clears that gate now.

    No test may continue into the real workflow, so it is replaced by a stop.
    """

    reached = _block_production_execution(monkeypatch)
    for command in ("--smoke", "--canary"):
        with pytest.raises(HarnessStop):
            H.main([command, "--allow-em"])
    assert [name for name, _auth in reached] == ["smoke", "canary"]


def test_T24_estimand_must_match_frozen_set(monkeypatch):
    reached = _block_full_production_execution(monkeypatch)
    monkeypatch.setattr(H, "ESTIMANDS", "A")
    with pytest.raises(HarnessStop):
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "B"])
    assert reached == [], "a per-estimand scope never reaches the full workflow"


def test_T24b_unfrozen_hierarchy_stops(monkeypatch):
    monkeypatch.setattr(H, "HIERARCHY", "<H3_A|H3_B>")
    with pytest.raises(HarnessStop):
        H.resolve_role("A")


def test_T24c_roles_are_frozen_to_h3a(monkeypatch):
    assert H.resolve_role("A") == "primary"
    assert H.resolve_role("B") == "sensitivity"
    monkeypatch.setattr(H, "HIERARCHY", "H3_B")
    assert H.resolve_role("A") == "coequal_A"
    assert H.resolve_role("B") == "coequal_B"


def test_T24d_manifest_role_matches_resolve_role():
    for estimand in ("A", "B"):
        rows = H.build_manifest(estimand)
        assert {r.role for r in rows} == {H.resolve_role(estimand)}


def test_T25_no_em_paths_report_zero_fits():
    for result in (H.run_validate_only(), H.run_config_gate(), H.run_record_diagnostics()):
        assert result["em_fits_executed"] == 0


def test_T26_no_em_import_in_subprocess():
    """A fresh interpreter must not load em_runner on the no-EM paths."""

    code = (
        "import sys;"
        f"sys.path.insert(0, r'{HERE}');"
        "import run_k_true_robustness_sweep as H;"
        "H.run_validate_only(); H.run_config_gate();"
        "print('em_runner' in sys.modules, 'model_dual_expfam_consistent' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"


# ===========================================================================
# T27-T32 — artifact guards, anchor reuse, Phase 7e protection
# ===========================================================================


def test_T27_existing_artifact_dir_is_refused(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "manifest.csv").write_text("x", encoding="utf-8")
    with pytest.raises(HarnessStop):
        H.require_no_existing_artifacts(target)


def test_T28_unexpected_artifact_is_refused(tmp_path):
    target = tmp_path / "out2"
    target.mkdir()
    (target / "surprise.csv").write_text("x", encoding="utf-8")
    with pytest.raises(HarnessStop):
        H.require_only_expected_artifacts(target)


def test_T29_anchor_rows_match_phase7e_selection():
    stored = {int(r["replicate"]): int(r["selected_k"])
              for r in _rows_of(H.PHASE7E_DIR / "replicate_selection.csv")}
    rows = H.build_selection_matrix_anchor_rows("B")
    assert len(rows) == 3
    for row in rows:
        replicate, selected = row[3], row[4]
        assert selected == stored[replicate]
        assert row[2] == H.ANCHOR_K_TRUE


def test_T30_anchor_rows_carry_provenance():
    for row in H.build_selection_matrix_anchor_rows("A"):
        assert row[8] == H.LINEAGE_ANCHOR
        assert row[9] == H.PHASE7E_RUN_CODE_SHA
        assert row[10] == H.PHASE7E_ARTIFACT_DIR


def test_T31_phase7e_directory_is_write_protected():
    with pytest.raises(HarnessStop):
        H.require_not_phase7e_path(H.PHASE7E_DIR / "fit_results.csv")
    with pytest.raises(HarnessStop):
        H.require_not_phase7e_path(H.PHASE7E_DIR)
    # a normal target is allowed
    assert H.require_not_phase7e_path(ROOT / "expfam" / "results" / "k_selection" / "other")


def test_T32_selection_matrix_excludes_score_columns():
    for forbidden in ("best_score", "margin"):
        assert forbidden not in H.SELECTION_MATRIX_COLUMNS
    assert forbidden not in A.SELECTION_MATRIX_COLUMNS


# ===========================================================================
# Adversarial tests (A01-A25)
# ===========================================================================


def test_A04_manifest_w_true_tampering_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "manifest.csv"
    rows = _rows_of(path)
    rows[0]["w_true"] = "9.99"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "manifest_w_true" for f in auditor.blockers)


def test_A05_model_seed_tampering_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "manifest.csv"
    rows = _rows_of(path)
    rows[5]["model_seed"] = "1"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "manifest_model_seed" for f in auditor.blockers)


def test_A06_anchor_k_true_in_new_manifest_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "manifest.csv"
    rows = _rows_of(path)
    rows[0]["K_TRUE"] = "3"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert auditor.blockers


def test_A07_role_tampering_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "manifest.csv"
    rows = _rows_of(path)
    for row in rows:
        row["role"] = "primary"          # B promoted to primary post hoc
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "manifest_role" for f in auditor.blockers)


def test_A08_missing_manifest_row_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "manifest.csv"
    rows = _rows_of(path)
    _rewrite(path, rows[:-1])
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "manifest_row_count" for f in auditor.blockers)


def test_A09_header_only_artifact_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    _write_csv(run_dir / "mask_provenance.csv", H.MASK_PROVENANCE_COLUMNS, [])
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "artifact_header_only" for f in auditor.blockers)


def test_A10_missing_required_artifact_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    (run_dir / "diagnostics.csv").unlink()
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "required_artifact_missing" for f in auditor.blockers)


def test_A11_selection_matrix_tampering_is_rejected(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    rows[0]["selected_k"] = "7"          # inconsistent with signed_error/label
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    assert any(f.check.startswith("matrix_") for f in auditor.blockers)


def test_A12_duplicate_key_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "mask_provenance.csv"
    rows = _rows_of(path)
    rows[1] = dict(rows[0])
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "mask_duplicate_key" for f in auditor.blockers)


def test_A13_diagnostics_cannot_carry_pass_fail(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "diagnostics.csv"
    rows = _rows_of(path)
    header = list(rows[0]) + ["passed"]
    for row in rows:
        row["passed"] = "True"
    _rewrite(path, rows, header)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "diagnostics_record_only" for f in auditor.blockers)


def test_A14_single_replicate_mask_mismatch_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "mask_provenance.csv"
    rows = _rows_of(path)
    rows[0]["split_mask_hash"] = "tampered"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "mask_sc_test_match" for f in auditor.blockers)


def test_A20_split_seed_offset_would_break_the_anchor(monkeypatch, anchors):
    """H2 must never offset the split seed; if it did, S_C would fail."""

    monkeypatch.setattr(H, "ANCHOR_SPLIT_SEED_BASE", 42500)
    with pytest.raises(HarnessStop):
        H.run_mask_gate(anchors=anchors, estimands=("A",))


def test_A21_independent_times_S_C_is_not_rejected(monkeypatch, anchors):
    monkeypatch.setattr(H, "RANDOM_DESIGN", "INDEPENDENT")
    gates = H.run_mask_gate(anchors=anchors, estimands=("A", "B"))
    assert all(g.passed for g in gates)


def test_A22_split_hash_from_train_mask_is_wrong():
    record = H.build_split_record(2, 2)
    from run_heldout_k_selection_pilot import stable_array_hash

    assert record.split_mask_hash != stable_array_hash(record.train_mask)


def test_A23_test_only_match_does_not_pass(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "A")
    path = run_dir / "mask_provenance.csv"
    rows = _rows_of(path)
    for row in rows:
        row["train_mask_hash"] = "not-the-anchor-train-hash"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "A", "config")
    assert any(f.check == "mask_sc_train_match" for f in auditor.blockers)


def test_A24_new_k3_diagnostics_row_is_rejected(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "diagnostics.csv"
    rows = _rows_of(path)
    extra = dict(rows[0])
    extra["K_TRUE"] = "3"
    _rewrite(path, rows + [extra])
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check in ("diagnostics_row_count", "diagnostics_no_anchor",
                           "diagnostics_k_true_set") for f in auditor.blockers)


def test_A25_valid_run_dir_passes_audit(tmp_path):
    """Control: the audit is not vacuously failing."""

    for estimand in ("A", "B"):
        run_dir = build_valid_run_dir(tmp_path, estimand)
        auditor = A.audit_run_dir(run_dir, estimand, "config")
        assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
        assert not auditor.highs


def test_A26_audit_does_not_import_the_harness_selector():
    import inspect

    source = inspect.getsource(A)
    assert "run_k_true_robustness_sweep" not in source
    assert "select_k_from_two_starts" not in source


# ===========================================================================
# HIGH-01 — independent audit must be fail-closed
# ===========================================================================


def test_H01_control_selection_mode_passes(tmp_path):
    """Control: a correct selection-mode directory audits clean."""

    for estimand in ("A", "B"):
        run_dir = build_selection_run_dir(tmp_path, estimand)
        auditor = A.audit_run_dir(run_dir, estimand, "selection")
        assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]


def test_H01_consistent_tampering_is_still_detected(tmp_path):
    """The core of HIGH-01.

    Change selected_k AND signed_error AND abs_error AND label so the matrix is
    internally consistent.  Cross-checking derived columns would pass; the audit
    must still FAIL because it re-derives selected_k from the per-(K,start)
    scores in fit_results.csv.
    """

    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    target = next(r for r in rows if int(r["K_TRUE"]) == 4 and int(r["replicate"]) == 2)
    bogus = 7
    signed = bogus - int(target["K_TRUE"])
    target["selected_k"] = str(bogus)
    target["signed_error"] = str(signed)
    target["abs_error"] = str(abs(signed))
    target["label"] = "over" if signed > 0 else ("under" if signed < 0 else "exact")
    _rewrite(path, rows)

    auditor = A.audit_run_dir(run_dir, "B", "selection")
    checks = {f.check for f in auditor.blockers}
    assert "matrix_selected_k_recomputed" in checks, checks
    # the internally-consistent derived columns did NOT trip anything
    assert "matrix_signed_error" not in checks
    assert "matrix_label" not in checks


def test_H01_anchor_selected_k_is_recomputed_from_phase7e(tmp_path):
    """K_TRUE=3 rows are re-derived from the Phase 7e per-(K,start) scores."""

    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    target = next(r for r in rows if int(r["K_TRUE"]) == 3 and int(r["replicate"]) == 3)
    assert target["selected_k"] == "5"      # Phase 7e recorded value
    target["selected_k"] = "3"
    target["signed_error"] = "0"
    target["abs_error"] = "0"
    target["label"] = "exact"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    assert any(f.check == "matrix_selected_k_recomputed" for f in auditor.blockers)


def test_H01_missing_selection_matrix_fails(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    (run_dir / "k_true_selection_matrix.csv").unlink()
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    assert any(f.check == "required_artifact_missing" for f in auditor.blockers)


def test_H01_missing_fit_results_fails(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    (run_dir / "fit_results.csv").unlink()
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    assert any(f.check == "required_artifact_missing" for f in auditor.blockers)


def test_H01_config_gate_failed_row_fails(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = _rows_of(path)
    rows[0]["passed"] = "False"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "config_gate_failed" for f in auditor.blockers)


def test_H01_config_gate_missing_required_gate_fails(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = [r for r in _rows_of(path) if r["gate"] != "MC1"]
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "config_gate_missing_scope" for f in auditor.blockers)


def test_H01_config_gate_duplicate_row_fails(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = _rows_of(path)
    _rewrite(path, rows + [dict(rows[0])])
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "config_gate_duplicate" for f in auditor.blockers)


def test_H01_runinfo_nonzero_em_fits_fails(tmp_path):
    import json as _json

    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "runinfo.json"
    payload = _json.loads(path.read_text(encoding="utf-8"))
    payload["em_fits_executed"] = 42
    path.write_text(_json.dumps(payload), encoding="utf-8")
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "runinfo_em_fits" for f in auditor.blockers)


def test_H01_matrix_duplicate_key_fails(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    _rewrite(path, rows + [dict(rows[0])])
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    assert any(f.check == "matrix_duplicate_key" for f in auditor.blockers)


def test_H01_matrix_missing_key_fails(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = [r for r in _rows_of(path) if not (int(r["K_TRUE"]) == 5 and int(r["replicate"]) == 1)]
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    assert any(f.check == "matrix_missing_key" for f in auditor.blockers)


def test_H01_matrix_unexpected_k_true_fails(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    rows[-1]["K_TRUE"] = "9"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    checks = {f.check for f in auditor.blockers}
    assert "matrix_unexpected_key" in checks or "matrix_k_true_grid" in checks


def test_H01_matrix_wrong_estimand_fails(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    rows[0]["estimand"] = "A"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    assert any(f.check in ("matrix_unexpected_key", "matrix_missing_key", "matrix_role")
               for f in auditor.blockers)


def test_H01_matrix_wrong_role_fails(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    for row in rows:
        row["role"] = "primary"           # B silently promoted
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    assert any(f.check == "matrix_role" for f in auditor.blockers)


def test_H01_anchor_row_with_new_run_sha_fails(tmp_path):
    """A K3 anchor row must not be re-labelled with a Phase 8b run SHA."""

    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    target = next(r for r in rows if int(r["K_TRUE"]) == 3)
    target["run_code_sha"] = "phase8b-new-sha"
    target["lineage"] = A.LINEAGE_NEW
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    checks = {f.check for f in auditor.blockers}
    assert "matrix_anchor_lineage" in checks and "matrix_anchor_sha" in checks


def test_H01_new_row_claiming_anchor_lineage_fails(tmp_path):
    run_dir = build_selection_run_dir(tmp_path, "B")
    path = run_dir / "k_true_selection_matrix.csv"
    rows = _rows_of(path)
    target = next(r for r in rows if int(r["K_TRUE"]) == 1)
    target["lineage"] = A.LINEAGE_ANCHOR
    target["run_code_sha"] = A.PHASE7E_RUN_CODE_SHA
    target["artifact_dir"] = A.PHASE7E_ARTIFACT_DIR
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "selection")
    checks = {f.check for f in auditor.blockers}
    assert {"matrix_new_lineage", "matrix_new_sha", "matrix_new_dir"} & checks


def test_H01_unknown_audit_mode_fails(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    auditor = A.audit_run_dir(run_dir, "B", "no-such-mode")
    assert any(f.check == "unknown_mode" for f in auditor.blockers)


def test_H01_selector_reimplemented_not_imported():
    import inspect

    source = inspect.getsource(A)
    assert "run_k_true_robustness_sweep" not in source
    assert "from run_heldout_k_selection_pilot" not in source
    assert "def select_k_independently" in source


def test_H01_independent_selector_matches_tie_rule():
    scores = {k: {1: -1.0, 2: -1.0} for k in A.K_CANDIDATES}
    scores[5] = {1: -0.5, 2: -0.5}
    scores[2] = {1: -0.5, 2: -0.5}          # exact tie with K=5
    assert A.select_k_independently(scores) == 2   # smallest K among ties


# ===========================================================================
# MEDIUM-01 — production diagnostics writer path
# ===========================================================================


def test_M01_production_writer_emits_contract_diagnostics(tmp_path):
    """Exercise run_record_diagnostics(out_dir) -> _write_csv, not a test writer."""

    result = H.run_record_diagnostics(tmp_path)
    assert result["em_fits_executed"] == 0
    assert result["record_only"] is True and result["blocking"] is False

    for estimand in ("A", "B"):
        path = tmp_path / estimand / "diagnostics.csv"
        assert path.is_file(), f"production writer did not emit {path}"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            rows = list(reader)

        assert header == H.DIAGNOSTICS_COLUMNS
        assert len(rows) == 12
        assert {int(r["K_TRUE"]) for r in rows} == set(H.NEW_K_TRUE)
        assert {int(r["replicate"]) for r in rows} == set(H.REPLICATES)
        assert sum(1 for r in rows if int(r["K_TRUE"]) == H.ANCHOR_K_TRUE) == 0
        keys = [(int(r["K_TRUE"]), int(r["replicate"])) for r in rows]
        assert len(keys) == len(set(keys)) == 12
        assert {r["role"] for r in rows} == {H.resolve_role(estimand)}
        assert {r["estimand"] for r in rows} == {estimand}
        # RECORD ONLY: no pass/fail or blocking semantics may leak into the file
        assert not ({"pass", "fail", "passed", "blocking", "gate", "threshold"} & set(header))
        for column in ("sample_sd_eta_y", "y_density", "conditional_entropy_bits",
                       "oracle_mean_log_score", "mean_sq_latent_norm", "f_frobenius_sq",
                       "mean_loading_energy"):
            assert column in header


def test_M01_production_writer_output_passes_production_audit(tmp_path):
    """The writer and the independent reader must actually connect."""

    H.run_record_diagnostics(tmp_path)
    for estimand in ("A", "B"):
        auditor = A.audit_run_dir(tmp_path / estimand, estimand, "config")
        assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
        assert not auditor.highs


def test_M01_writer_refuses_phase7e_directory():
    with pytest.raises(HarnessStop):
        H.write_zero_em_artifacts(H.PHASE7E_DIR, "A")


def test_M01_writer_path_runs_no_em(tmp_path):
    """Fresh interpreter: the writer path must not import any EM module."""

    code = (
        "import sys, tempfile;"
        "sys.path.insert(0, r'" + str(HERE) + "');"
        "import run_k_true_robustness_sweep as H;"
        "d = tempfile.mkdtemp();"
        "r = H.run_record_diagnostics(d);"
        "print(r['em_fits_executed'], 'em_runner' in sys.modules,"
        " 'model_dual_expfam_consistent' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0 False False"


# ===========================================================================
# HIGH (residual) — config gate scope-level fail-open
# ===========================================================================


def test_G_positive_control_config_audit_passes(tmp_path):
    """Production config artifacts audit clean, with the exact frozen counts."""

    for estimand, expected in (("A", 73), ("B", 77)):
        run_dir = build_valid_run_dir(tmp_path, estimand)
        rows = _rows_of(run_dir / "config_gate.csv")
        assert len(rows) == expected
        assert A.expected_config_gate_count(estimand) == expected
        assert A.EXPECTED_CONFIG_GATE_COUNT[estimand] == expected
        auditor = A.audit_run_dir(run_dir, estimand, "config")
        assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]


def test_G_expected_set_matches_actual_production_keys(tmp_path):
    """The independently built expected set equals what the harness emits."""

    for estimand in ("A", "B"):
        run_dir = build_valid_run_dir(tmp_path, estimand)
        actual = {(r["gate"], r["scope"]) for r in _rows_of(run_dir / "config_gate.csv")}
        assert actual == A.expected_gate_scope_keys(estimand)


def test_G_cli_aggregate_is_124_and_distinct_from_per_estimand():
    """124 is the CLI aggregate over both estimands, not a per-artifact count."""

    result = H.run_config_gate()
    assert result["gate_count"] == A.EXPECTED_CLI_AGGREGATE_GATE_COUNT == 124
    assert result["gates_passed"] == 124
    assert sum(A.EXPECTED_CONFIG_GATE_COUNT.values()) != 124  # 73 + 77 = 150


# --- runinfo requirement ---------------------------------------------------


def test_G_missing_runinfo_fails(tmp_path):
    for mode in ("config", "selection"):
        run_dir = (build_valid_run_dir(tmp_path / mode, "B") if mode == "config"
                   else build_selection_run_dir(tmp_path / mode, "B"))
        (run_dir / "runinfo.json").unlink()
        auditor = A.audit_run_dir(run_dir, "B", mode)
        checks = {f.check for f in auditor.blockers}
        assert "required_artifact_missing" in checks, (mode, checks)


def test_G_missing_runinfo_em_field_fails(tmp_path):
    import json as _json

    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "runinfo.json"
    payload = _json.loads(path.read_text(encoding="utf-8"))
    payload.pop("em_fits_executed")
    path.write_text(_json.dumps(payload), encoding="utf-8")
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "runinfo_em_field" for f in auditor.blockers)


def test_G_missing_runinfo_gate_count_field_fails(tmp_path):
    import json as _json

    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "runinfo.json"
    payload = _json.loads(path.read_text(encoding="utf-8"))
    payload.pop("gate_count")
    path.write_text(_json.dumps(payload), encoding="utf-8")
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "runinfo_gate_count_field" for f in auditor.blockers)


def test_G_runinfo_invalid_field_type_fails(tmp_path):
    import json as _json

    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "runinfo.json"
    payload = _json.loads(path.read_text(encoding="utf-8"))
    payload["em_fits_executed"] = "0"
    payload["gate_count"] = "77"
    path.write_text(_json.dumps(payload), encoding="utf-8")
    auditor = A.audit_run_dir(run_dir, "B", "config")
    checks = {f.check for f in auditor.blockers}
    assert "runinfo_em_type" in checks and "runinfo_gate_count_type" in checks


def test_G_wrong_declared_gate_count_fails(tmp_path):
    """config_gate.csv stays correct; only the declared count is wrong."""

    import json as _json

    for wrong in (76, 78):
        run_dir = build_valid_run_dir(tmp_path / str(wrong), "B")
        path = run_dir / "runinfo.json"
        payload = _json.loads(path.read_text(encoding="utf-8"))
        payload["gate_count"] = wrong
        path.write_text(_json.dumps(payload), encoding="utf-8")
        auditor = A.audit_run_dir(run_dir, "B", "config")
        assert any(f.check == "config_gate_count" for f in auditor.blockers), wrong


# --- scope-level completeness ---------------------------------------------


def test_G_single_cell_scope_row_deletion_fails(tmp_path):
    """Delete ONE MC1 scope row; other MC1 rows remain, so the NAME survives."""

    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = _rows_of(path)
    victim = next(r for r in rows if r["gate"] == "MC1" and r["scope"] == "B/K4/r2")
    rows.remove(victim)
    _rewrite(path, rows)

    assert any(r["gate"] == "MC1" for r in _rows_of(path))   # name still present
    auditor = A.audit_run_dir(run_dir, "B", "config")
    checks = {f.check for f in auditor.blockers}
    assert "config_gate_missing_scope" in checks, checks
    assert "config_gate_row_count" in checks


def test_G_one_row_per_gate_collapse_fails(tmp_path):
    """Mass collapse: keep one PASS row per gate NAME.  Name-only audit would pass."""

    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = _rows_of(path)
    collapsed: dict[str, dict] = {}
    for row in rows:
        collapsed.setdefault(row["gate"], row)
    kept = list(collapsed.values())
    _rewrite(path, kept)

    # every gate NAME survives, and every remaining row is PASS
    assert {r["gate"] for r in kept} == {r["gate"] for r in rows}
    assert all(r["passed"] == "True" for r in kept)
    assert len(kept) < len(rows)

    auditor = A.audit_run_dir(run_dir, "B", "config")
    checks = {f.check for f in auditor.blockers}
    assert "config_gate_missing_scope" in checks, checks
    assert "config_gate_row_count" in checks
    assert "config_gate_set_equality" in checks


def test_G_unexpected_scope_fails(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = _rows_of(path)
    bogus = dict(rows[0])
    bogus["scope"] = "B/K9/r9"          # unregistered cell
    _rewrite(path, rows + [bogus])
    auditor = A.audit_run_dir(run_dir, "B", "config")
    checks = {f.check for f in auditor.blockers}
    assert "config_gate_unexpected_scope" in checks, checks


def test_G_malformed_scope_fails(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = _rows_of(path)
    victim = next(r for r in rows if r["gate"] == "M0")
    victim["scope"] = "not-a-cell"
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    checks = {f.check for f in auditor.blockers}
    assert "config_gate_unexpected_scope" in checks and "config_gate_missing_scope" in checks


def test_G_duplicate_gate_scope_fails(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = _rows_of(path)
    _rewrite(path, rows + [dict(rows[3])])
    auditor = A.audit_run_dir(run_dir, "B", "config")
    assert any(f.check == "config_gate_duplicate" for f in auditor.blockers)


def test_G_unknown_status_token_fails(tmp_path):
    run_dir = build_valid_run_dir(tmp_path, "B")
    path = run_dir / "config_gate.csv"
    rows = _rows_of(path)
    rows[0]["passed"] = "yes"           # not the canonical representation
    _rewrite(path, rows)
    auditor = A.audit_run_dir(run_dir, "B", "config")
    checks = {f.check for f in auditor.blockers}
    assert "config_gate_unknown_status" in checks and "config_gate_failed" in checks


def test_G_expected_set_built_without_importing_harness():
    import inspect

    source = inspect.getsource(A.expected_gate_scope_keys)
    assert "run_k_true_robustness_sweep" not in source
    assert "H." not in source
    module_source = inspect.getsource(A)
    assert "import run_k_true_robustness_sweep" not in module_source


# ===========================================================================
# Phase 8b S2 — direct pre-smoke leakage falsification (Issue #51)
# ===========================================================================
#
# Every test here uses the SEALED fake adapter: it accepts no callback, so no
# test can inject arbitrary code -- or a captured ScoreOnlyTarget -- into the
# fit call.  No test reaches em_runner or any model module: real EM calls = 0.


import inspect as _inspect  # noqa: E402

import numpy as _np  # noqa: E402
from run_heldout_k_selection_pilot import (  # noqa: E402
    make_score_only_target,
    make_training_y_values,
)


def _leakage_fixture(estimand: str = "A", index: int = 0):
    """A valid train-only request plus a FRESH mutable mask state."""

    anchors = H.read_phase7e_anchor_masks()
    row = H.build_manifest(estimand)[index]
    split = H.build_split_record(row.k_true, row.replicate)
    data = H._generate_cell(estimand, row.k_true, row.replicate)
    Y = _np.array(data["Y"], dtype=_np.float64)
    training = make_training_y_values(Y, split.train_mask)
    score_target = make_score_only_target(Y, split.test_mask)
    request = H.build_fit_request(row, training, split.train_mask, split.test_mask,
                                  anchors[row.replicate])
    state = H.MutableMaskState(test_mask=_np.array(split.test_mask, dtype=bool),
                               train_mask=_np.array(split.train_mask, dtype=bool))
    return row, split, Y, score_target, request, state


def _executable_body(func) -> str:
    """The function's source with its docstring removed."""

    import ast as _ast
    import textwrap as _textwrap

    tree = _ast.parse(_textwrap.dedent(_inspect.getsource(func)))
    body = tree.body[0].body
    if body and isinstance(body[0], _ast.Expr) and isinstance(body[0].value, _ast.Constant):
        body = body[1:]
    return "\n".join(_ast.unparse(node) for node in body)


def _sealed(state, mode=None):
    if mode is None:
        return H.SealedFakeFitAdapter(state)
    return H.SealedFakeFitAdapter(state, mode)


# --- positive control ------------------------------------------------------


def test_valid_sealed_fake_adapter_one_call():
    """Without this control, the negative tests could be failing for other reasons."""

    row, _split, _Y, target, request, state = _leakage_fixture()
    adapter = _sealed(state)
    result, report = H.Phase8bFitBoundary(adapter).run(request, row, state, target)

    assert adapter.calls == 1
    assert adapter.mutations_applied == 0
    assert report.pre_fit_passed and report.post_fit_passed
    assert report.pre_fit_test_mask_hash == report.post_fit_test_mask_hash == request.anchor_mask_hash
    assert report.pre_fit_train_mask_hash == report.post_fit_train_mask_hash \
        == request.anchor_train_mask_hash
    assert result["k_est"] == row.k


def test_S2_positive_both_estimands():
    for estimand in ("A", "B"):
        row, _split, _Y, target, request, state = _leakage_fixture(estimand)
        adapter = _sealed(state)
        H.Phase8bFitBoundary(adapter).run(request, row, state, target)
        assert adapter.calls == 1
        assert request.role == H.resolve_role(estimand)
        assert request.w_true == H.resolve_w_true(estimand, row.k_true)


def test_same_content_mask_clone_passes():
    """Content hashes, not object identity: a clone of the masks still passes."""

    row, split, _Y, target, request, _state = _leakage_fixture()
    clone = H.MutableMaskState(test_mask=_np.array(split.test_mask, dtype=bool),
                               train_mask=_np.array(split.train_mask, dtype=bool))
    assert clone.test_mask is not split.test_mask and clone.train_mask is not split.train_mask
    adapter = _sealed(clone)
    # one shared MutableMaskState -- copied ARRAYS are fine, a copied STATE is not
    assert adapter.mask_state is clone
    _, report = H.Phase8bFitBoundary(adapter).run(request, row, clone, target)
    assert adapter.calls == 1 and report.post_fit_passed
    assert report.pre_fit_test_mask_hash == report.post_fit_test_mask_hash


# --- A01: raw held-out Y injection -----------------------------------------


def test_A01_raw_heldout_y_injection_boundary_calls_zero():
    """Inject the true held-out Y into the fit matrix at held-out positions."""

    row, split, Y, target, request, state = _leakage_fixture()
    payload = request.fit_payload

    leaked = _np.array(payload.Y_fit, dtype=_np.float64)
    rows_i, cols_i = _np.where(_np.triu(split.test_mask, 1))
    leaked[rows_i, cols_i] = Y[rows_i, cols_i]        # real held-out outcomes
    leaked[cols_i, rows_i] = Y[rows_i, cols_i]
    assert not _np.array_equal(leaked, payload.Y_fit), "injection must change the matrix"

    tampered_payload = H.FitPayload(
        Y_fit=leaked,
        train_mask=payload.train_mask,
        payload_hash=H.stable_array_hash(leaked),      # attacker also fixes the hash
        train_mask_hash=payload.train_mask_hash,
        expected_shape=payload.expected_shape,
        expected_dtype=payload.expected_dtype,
        provenance_version=payload.provenance_version,
        canary_value=payload.canary_value,
        _authority=payload._authority,
    )
    tampered = dataclasses.replace(request, fit_payload=tampered_payload)

    adapter = _sealed(state)
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)

    assert "held-out Y reached the fit payload" in str(excinfo.value)
    assert adapter.calls == 0, "the fit must be refused BEFORE the adapter is called"


def test_A01_partial_injection_single_dyad_rejected():
    """Even one leaked dyad must be refused."""

    row, split, Y, target, request, state = _leakage_fixture()
    payload = request.fit_payload
    leaked = _np.array(payload.Y_fit, dtype=_np.float64)
    rows_i, cols_i = _np.where(_np.triu(split.test_mask, 1))
    idx = next(i for i in range(len(rows_i))
               if Y[rows_i[i], cols_i[i]] != float(H.MASKED_CANARY_VALUE))
    r, c = rows_i[idx], cols_i[idx]
    leaked[r, c] = Y[r, c]
    leaked[c, r] = Y[r, c]

    tampered = dataclasses.replace(request, fit_payload=dataclasses.replace(
        payload, Y_fit=leaked, payload_hash=H.stable_array_hash(leaked)))

    adapter = _sealed(state)
    with pytest.raises(HarnessStop):
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert adapter.calls == 0


def test_A01_payload_hash_mismatch_rejected_before_fit():
    """A silent matrix edit that does not update the hash is also refused."""

    row, split, _Y, target, request, state = _leakage_fixture()
    payload = request.fit_payload
    leaked = _np.array(payload.Y_fit, dtype=_np.float64)
    rows_i, cols_i = _np.where(_np.triu(split.test_mask, 1))
    leaked[rows_i[0], cols_i[0]] = 1.0
    leaked[cols_i[0], rows_i[0]] = 1.0
    tampered = dataclasses.replace(request, fit_payload=dataclasses.replace(payload, Y_fit=leaked))

    adapter = _sealed(state)
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert "payload hash mismatch" in str(excinfo.value)
    assert adapter.calls == 0


def test_A01_builder_refuses_full_Y_as_training_values():
    """make_training_y_values(Y, test_mask) cannot be passed off as training data."""

    row, split, Y, _target, _request, _state = _leakage_fixture()
    wrong = make_training_y_values(Y, split.test_mask)   # held-out dyads, typed as training
    anchors = H.read_phase7e_anchor_masks()
    with pytest.raises(HarnessStop):
        H.build_fit_request(row, wrong, split.train_mask, split.test_mask, anchors[row.replicate])


# --- A02: ScoreOnlyTarget rejection through the production boundary ---------


class _TargetHoldingAdapter:
    """A malicious adapter that captured the real Phase 7e ScoreOnlyTarget."""

    def __init__(self, target):
        self.target = target
        self.calls = 0

    def fit(self, request):
        self.calls += 1
        return {"fake": True, "leaked": self.target.values}


def test_A02_score_only_target_adapter_rejected_calls_zero():
    """An adapter holding the scoring target never receives a fit."""

    row, _split, _Y, target, request, state = _leakage_fixture()
    adapter = _TargetHoldingAdapter(target)

    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(request, row, state, target)

    assert "unauthorized fit adapter type" in str(excinfo.value)
    assert adapter.calls == 0, "the malicious adapter must never be invoked"


def test_A02_score_only_target_request_rejected_calls_zero():
    """A wrapper request smuggling the scoring target never reaches a fit."""

    row, _split, _Y, target, request, state = _leakage_fixture()

    @dataclasses.dataclass(frozen=True)
    class SmuggledRequest:
        inner: H.Phase8bFitRequest
        stowaway: object

    smuggled = SmuggledRequest(request, target)
    adapter = _sealed(state)

    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(smuggled, row, state, target)

    assert "Phase8bFitRequest" in str(excinfo.value)
    assert adapter.calls == 0
    # the same object graph is also refused by the deep leakage walk
    with pytest.raises(HarnessStop) as deep:
        H._require_no_score_target(smuggled, target)
    assert "ScoreOnlyTarget reached fit boundary" in str(deep.value)


def test_A02_target_values_alias_rejected_before_fit():
    """A raw alias of target.values riding along the request is refused."""

    row, _split, _Y, target, request, state = _leakage_fixture()

    class AliasCarrier:
        def __init__(self, values):
            self.smuggled_values = values

    tampered = dataclasses.replace(request, fit_config=AliasCarrier(target.values))
    adapter = _sealed(state)
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert "target.values" in str(excinfo.value)
    assert adapter.calls == 0


def test_A02_score_target_memory_view_rejected():
    """A numpy view that shares memory with target.values is still caught."""

    row, _split, _Y, target, request, state = _leakage_fixture()
    view = target.values[:]
    assert _np.shares_memory(view, target.values)

    class ViewCarrier:
        def __init__(self, values):
            self.v = values

    tampered = dataclasses.replace(request, fit_config=ViewCarrier(view))
    adapter = _sealed(state)
    with pytest.raises(HarnessStop):
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert adapter.calls == 0


def test_A02_request_has_no_scoring_field():
    """Structural guarantee: the request type cannot hold a scoring target."""

    _row, _split, _Y, _target, request, _state = _leakage_fixture()
    field_names = {f.name for f in dataclasses.fields(request)}
    for forbidden in ("score_target", "target", "held_out_target", "test_values",
                      "Y_full", "Y", "test_mask"):
        assert forbidden not in field_names, forbidden
    for forbidden in ("score_target", "target", "held_out_target", "test_values", "Y_full"):
        assert not hasattr(request, forbidden)


def test_A02_sealed_adapter_schema_has_no_target_or_callback_field():
    """§6: the sealed adapter's field set is checked explicitly, not by DFS."""

    assert H.SealedFakeFitAdapter.__slots__ == H.SEALED_FAKE_ADAPTER_SLOTS
    assert set(H.SEALED_FAKE_ADAPTER_SLOTS) == {
        "calls", "mutation_mode", "mask_state", "mutations_applied",
        "last_payload_hash", "last_k_est",
    }
    _row, _split, _Y, _target, _request, state = _leakage_fixture()
    adapter = _sealed(state)
    for forbidden in H.FORBIDDEN_ADAPTER_ATTRS:
        assert not hasattr(adapter, forbidden), forbidden
    assert not hasattr(adapter, "__dict__"), "sealed adapter must not accept new attributes"
    with pytest.raises(AttributeError):
        adapter.score_target = object()


# --- adapter policy: no arbitrary code surface -----------------------------


def test_arbitrary_callback_adapter_rejected():
    """An adapter carrying an arbitrary fit hook is refused before the fit."""

    row, _split, _Y, target, request, state = _leakage_fixture()

    class CallbackAdapter:
        def __init__(self, on_fit):
            self.on_fit = on_fit
            self.calls = 0

        def fit(self, req):
            self.calls += 1
            self.on_fit(req)
            return {"fake": True}

    captured = []
    adapter = CallbackAdapter(lambda req: captured.append(target.values))
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(request, row, state, target)
    assert "unauthorized fit adapter type" in str(excinfo.value)
    assert adapter.calls == 0 and captured == []


def test_adapter_subclass_rejected():
    """Exact-type policy: even a subclass of the sealed adapter is refused."""

    row, _split, _Y, target, request, state = _leakage_fixture()

    class SubclassAdapter(H.SealedFakeFitAdapter):
        __slots__ = ()

    adapter = SubclassAdapter(state)
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(request, row, state, target)
    assert "unauthorized fit adapter type" in str(excinfo.value)
    assert adapter.calls == 0


def test_sealed_adapter_source_has_no_arbitrary_hook():
    """§4: no callback/closure/partial injection point survives in the adapter."""

    source = _inspect.getsource(H.SealedFakeFitAdapter)
    for token in ("on_fit", "callback", "lambda", "partial", "closure", "__call__"):
        assert token not in source, token
    signature = _inspect.signature(H.SealedFakeFitAdapter.__init__)
    assert list(signature.parameters) == ["self", "mask_state", "mutation_mode"]
    assert not hasattr(H, "CountingFakeFitAdapter"), "the callback adapter must be gone"


def test_S2_real_em_adapter_is_refused_by_the_boundary():
    row, _split, _Y, target, request, state = _leakage_fixture()

    class AuthorizedEMFitAdapter:      # name-matched stand-in
        calls = 0

        def fit(self, request):        # pragma: no cover - must never run
            raise AssertionError("real EM adapter must not be reachable here")

    adapter = AuthorizedEMFitAdapter()
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(request, row, state, target)
    assert "real EM adapter is not authorized" in str(excinfo.value)
    assert adapter.calls == 0


# --- A03: mask mutation performed INSIDE the adapter ------------------------


def _mutation_case(mode):
    row, _split, _Y, target, request, state = _leakage_fixture()
    adapter = _sealed(state, mode)
    boundary = H.Phase8bFitBoundary(adapter)
    with pytest.raises(HarnessStop) as excinfo:
        boundary.run(request, row, state, target)
    assert adapter.calls == 1, "the mutation must be caught AFTER the fit, before scoring"
    assert adapter.mutations_applied == 1, "the adapter must actually have mutated the mask"
    return str(excinfo.value), state, request


def test_A03_test_mask_mutated_inside_adapter():
    message, state, request = _mutation_case(H.FakeMutationMode.TEST_MASK)
    assert "post-fit" in message and "test mask" in message
    assert H.compute_split_mask_hash(state.test_mask) != request.pre_fit_test_mask_hash
    assert H.compute_train_mask_hash(state.train_mask) == request.pre_fit_train_mask_hash


def test_A03_train_mask_mutated_inside_adapter():
    message, state, request = _mutation_case(H.FakeMutationMode.TRAIN_MASK)
    assert "post-fit" in message and "train mask" in message
    assert H.compute_train_mask_hash(state.train_mask) != request.pre_fit_train_mask_hash
    assert H.compute_split_mask_hash(state.test_mask) == request.pre_fit_test_mask_hash


def test_A03_both_masks_mutated_inside_adapter():
    message, state, request = _mutation_case(H.FakeMutationMode.BOTH_MASKS)
    assert "post-fit" in message
    assert H.compute_split_mask_hash(state.test_mask) != request.pre_fit_test_mask_hash
    assert H.compute_train_mask_hash(state.train_mask) != request.pre_fit_train_mask_hash


def test_A03_score_not_reached_after_postfit_failure():
    """The caller's score / artifact / selection steps never run."""

    row, _split, _Y, target, request, state = _leakage_fixture()
    counters = {"score": 0, "artifact": 0, "selection": 0}
    adapter = _sealed(state, H.FakeMutationMode.TEST_MASK)
    boundary = H.Phase8bFitBoundary(adapter)

    with pytest.raises(HarnessStop):
        result, _report = boundary.run(request, row, state, target)
        counters["score"] += 1          # unreachable: run() raised
        counters["artifact"] += 1
        counters["selection"] += 1

    assert adapter.calls == 1
    assert counters == {"score": 0, "artifact": 0, "selection": 0}


def test_A03_no_caller_supplied_post_fit_mask_argument():
    """§8: the post-fit hash cannot come from a caller-supplied value."""

    signature = _inspect.signature(H.Phase8bFitBoundary.run)
    assert list(signature.parameters) == ["self", "request", "manifest_row",
                                          "mask_state", "score_target"]
    for method in (H.Phase8bFitBoundary.run, H.Phase8bFitBoundary.check_post_fit):
        assert "post_fit_masks" not in _inspect.signature(method).parameters


def test_A03_boundary_rereads_the_same_mask_state_object():
    """The boundary must hash the live state, not a copy taken pre-fit."""

    source = _inspect.getsource(H.Phase8bFitBoundary.run)
    assert "self.check_post_fit(request, mask_state, result, score_target)" in source
    assert "_require_post_fit_masks" in _inspect.getsource(H.Phase8bFitBoundary.check_post_fit)


# --- A03 identity binding: the fit must touch the monitored state ----------


def test_adapter_mask_state_must_be_boundary_mask_state():
    """A same-content DECOY state must be refused BEFORE the fit.

    Content hashing says whether a mask changed; it cannot say whose mask
    changed.  Without an identity binding the adapter could mutate state A
    while the boundary re-hashes an untouched twin B and reports PASS.
    """

    row, split, _Y, target, request, state_a = _leakage_fixture()
    state_b = H.MutableMaskState(test_mask=_np.array(state_a.test_mask, dtype=bool),
                                 train_mask=_np.array(state_a.train_mask, dtype=bool))
    # same content, different objects -- on every level
    assert state_b is not state_a
    assert state_b.test_mask is not state_a.test_mask
    assert _np.array_equal(state_b.test_mask, state_a.test_mask)
    assert _np.array_equal(state_b.train_mask, state_a.train_mask)
    assert H.compute_split_mask_hash(state_b.test_mask) == \
        H.compute_split_mask_hash(state_a.test_mask)

    before_a = (_np.array(state_a.test_mask), _np.array(state_a.train_mask))
    before_b = (_np.array(state_b.test_mask), _np.array(state_b.train_mask))

    adapter = H.SealedFakeFitAdapter(state_a)          # adapter holds A
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(request, row, state_b, target)   # boundary watches B

    assert "adapter/boundary mask state mismatch" in str(excinfo.value)
    assert adapter.calls == 0, "a mismatched pair must never reach the fit"
    assert adapter.mutations_applied == 0
    assert _np.array_equal(state_a.test_mask, before_a[0])
    assert _np.array_equal(state_a.train_mask, before_a[1])
    assert _np.array_equal(state_b.test_mask, before_b[0])
    assert _np.array_equal(state_b.train_mask, before_b[1])


def test_decoy_state_attack_is_refused_before_any_mutation():
    """The stronger form: the adapter is armed to mutate, and still never runs.

    The point is not "fit, then detect"; it is "a mismatched pair is not
    allowed to fit at all", so no mutation is ever performed.
    """

    row, _split, _Y, target, request, state_a = _leakage_fixture()
    decoy = H.MutableMaskState(test_mask=_np.array(state_a.test_mask, dtype=bool),
                               train_mask=_np.array(state_a.train_mask, dtype=bool))
    adapter = H.SealedFakeFitAdapter(state_a, H.FakeMutationMode.TEST_MASK)

    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(request, row, decoy, target)

    assert "adapter/boundary mask state mismatch" in str(excinfo.value)
    assert adapter.calls == 0
    assert adapter.mutations_applied == 0
    # neither state moved, so the boundary never had to rely on a hash compare
    assert H.compute_split_mask_hash(state_a.test_mask) == request.pre_fit_test_mask_hash
    assert H.compute_split_mask_hash(decoy.test_mask) == request.pre_fit_test_mask_hash


def test_binding_is_identity_not_content():
    """`np.array_equal` on the masks is explicitly not sufficient."""

    row, _split, _Y, target, request, state_a = _leakage_fixture()
    twin = H.MutableMaskState(test_mask=state_a.test_mask, train_mask=state_a.train_mask)
    # even sharing the very same arrays is not enough: the STATE must be shared
    assert twin.test_mask is state_a.test_mask and twin.train_mask is state_a.train_mask
    assert twin is not state_a

    adapter = H.SealedFakeFitAdapter(state_a)
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(request, row, twin, target)
    assert "adapter/boundary mask state mismatch" in str(excinfo.value)
    assert adapter.calls == 0

    executable = _executable_body(H._require_adapter_state_binding)
    assert "adapter_state is mask_state" in executable
    assert "array_equal" not in executable, "the binding must not fall back to content"


@pytest.mark.parametrize("broken", ["adapter_state_type", "boundary_state_type",
                                    "adapter_state_missing"])
def test_state_binding_fails_closed_on_malformed_inputs(broken):
    """§4: a malformed state raises HarnessStop, never AttributeError/TypeError."""

    _row, _split, _Y, _target, _request, state = _leakage_fixture()

    class LooseAdapter:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    if broken == "adapter_state_type":
        adapter, boundary_state = LooseAdapter(mask_state=object()), state
    elif broken == "boundary_state_type":
        adapter, boundary_state = H.SealedFakeFitAdapter(state), object()
    else:
        adapter, boundary_state = LooseAdapter(), state

    with pytest.raises(HarnessStop):
        H._require_adapter_state_binding(adapter, boundary_state)


def test_state_binding_is_checked_before_the_fit():
    """The guard runs in check_pre_fit, ahead of the adapter call."""

    pre_source = _inspect.getsource(H.Phase8bFitBoundary.check_pre_fit)
    run_source = _inspect.getsource(H.Phase8bFitBoundary.run)
    assert "_require_adapter_state_binding(self._adapter, mask_state)" in pre_source
    assert run_source.index("self.check_pre_fit(") < run_source.index("self._adapter.fit(")
    # and re-verified after the fit as defence in depth
    assert "_require_adapter_state_binding" in _inspect.getsource(
        H.Phase8bFitBoundary.check_post_fit)


def test_self_check_fails_if_state_binding_guard_disabled(monkeypatch):
    """§15: the binding case alone can fail the machine leakage gate."""

    monkeypatch.setattr(H, "_require_adapter_state_binding", lambda *a, **k: None)
    result = H.run_leakage_self_check()
    assert result["all_passed"] is False
    assert result["cases"]["A03_adapter_state_binding"]["passed"] is False
    # with the guard gone the decoy attack actually goes through: the fit runs
    # and the boundary's post-fit check on the untouched twin does NOT reject
    assert result["cases"]["A03_adapter_state_binding"]["rejected"] is False
    assert H.current_smoke_authorization().leakage_gate_pass is False


# --- manifest binding negatives --------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("k", 6),
    ("start", 2),
    ("model_seed", 999999),
    ("data_seed", 999999),
    ("split_seed", 999999),
    ("k_true", 5),
    ("estimand", "B"),
    ("role", "sensitivity"),
    ("w_true", 9.9),
    ("w0_true", 0.5),
])
def test_S2_manifest_binding_mismatch_rejected_before_fit(field, value):
    row, _split, _Y, target, request, state = _leakage_fixture()
    tampered = dataclasses.replace(request, **{field: value})
    adapter = _sealed(state)
    with pytest.raises(HarnessStop):
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert adapter.calls == 0, f"{field} mismatch must be caught before the fit"


@pytest.mark.parametrize("field", ["frozen_config_hash", "score_config_hash"])
def test_S2_config_hash_mismatch_rejected(field):
    row, _split, _Y, target, request, state = _leakage_fixture()
    tampered = dataclasses.replace(request, **{field: "tampered"})
    adapter = _sealed(state)
    with pytest.raises(HarnessStop):
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert adapter.calls == 0


# --- pre-fit hash negatives ------------------------------------------------


@pytest.mark.parametrize("field", [
    "pre_fit_test_mask_hash",
    "pre_fit_train_mask_hash",
    "anchor_mask_hash",
    "anchor_train_mask_hash",
])
def test_S2_pre_fit_hash_mismatch_rejected_before_fit(field):
    row, _split, _Y, target, request, state = _leakage_fixture()
    tampered = dataclasses.replace(request, **{field: "0" * 64})
    adapter = _sealed(state)
    with pytest.raises(HarnessStop):
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert adapter.calls == 0


def test_S2_test_only_anchor_match_is_not_enough():
    """Matching the anchor on the test side alone must not pass."""

    row, split, _Y, target, request, _state = _leakage_fixture()
    other = H.build_split_record(H.NEW_K_TRUE[0], 2 if split.replicate != 2 else 3)
    state = H.MutableMaskState(test_mask=_np.array(split.test_mask, dtype=bool),
                               train_mask=_np.array(other.train_mask, dtype=bool))
    adapter = _sealed(state)
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(request, row, state, target)
    assert "train mask" in str(excinfo.value)
    assert adapter.calls == 0


# --- fail-closed input validation ------------------------------------------


def test_S2_malformed_fit_config_fails_closed_with_harness_stop():
    """A malformed sub-object must raise HarnessStop, not AttributeError."""

    row, _split, _Y, target, request, state = _leakage_fixture()

    class NotAFitConfig:
        pass

    tampered = dataclasses.replace(request, fit_config=NotAFitConfig())
    adapter = _sealed(state)
    with pytest.raises(HarnessStop):
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert adapter.calls == 0


@pytest.mark.parametrize("bad", ["manifest_row", "mask_state"])
def test_S2_malformed_boundary_inputs_fail_closed(bad):
    row, _split, _Y, target, request, state = _leakage_fixture()
    adapter = _sealed(state)
    args = {"manifest_row": row, "mask_state": state}
    args[bad] = object()
    with pytest.raises(HarnessStop):
        H.Phase8bFitBoundary(adapter).run(request, args["manifest_row"],
                                          args["mask_state"], target)
    assert adapter.calls == 0


def test_S2_leakage_check_precedes_binding_check():
    """A request that is BOTH leaking and mis-bound reports the leakage."""

    row, split, Y, target, request, state = _leakage_fixture()
    payload = request.fit_payload
    leaked = _np.array(payload.Y_fit, dtype=_np.float64)
    r, c = _np.where(_np.triu(split.test_mask, 1))
    leaked[r, c] = Y[r, c]
    leaked[c, r] = Y[r, c]
    tampered = dataclasses.replace(
        request,
        fit_payload=dataclasses.replace(payload, Y_fit=leaked,
                                        payload_hash=H.stable_array_hash(leaked)),
        model_seed=123456,           # also mis-bound
    )
    adapter = _sealed(state)
    with pytest.raises(HarnessStop) as excinfo:
        H.Phase8bFitBoundary(adapter).run(tampered, row, state, target)
    assert "held-out Y reached the fit payload" in str(excinfo.value)
    assert adapter.calls == 0


# --- HIGH-01: the self-check is derived from the attacks -------------------


def test_leakage_self_check_runs_all_cases():
    result = H.run_leakage_self_check()

    assert set(result["cases"]) == set(H.LEAKAGE_SELF_CHECK_CASE_NAMES)
    assert set(H.LEAKAGE_SELF_CHECK_CASE_NAMES) == {
        "positive_control",
        "A01_raw_held_out_y",
        "A02_malicious_adapter",
        "A02_smuggled_request",
        "A03_test_mask_mutation",
        "A03_train_mask_mutation",
        "A03_both_mask_mutation",
        "A03_adapter_state_binding",
        "anchor_pre_fit_binding",
    }
    assert result["all_passed"] is True
    assert all(case["passed"] for case in result["cases"].values())

    cases = result["cases"]
    assert cases["positive_control"]["rejected"] is False
    assert cases["positive_control"]["adapter_calls"] == 1
    for name in ("A01_raw_held_out_y", "A02_malicious_adapter",
                 "A02_smuggled_request", "A03_adapter_state_binding",
                 "anchor_pre_fit_binding"):
        assert cases[name]["rejected"] is True, name
        assert cases[name]["adapter_calls"] == 0, name
    for name in ("A03_test_mask_mutation", "A03_train_mask_mutation",
                 "A03_both_mask_mutation"):
        assert cases[name]["rejected"] is True, name
        assert cases[name]["adapter_calls"] == 1, name
        assert "post-fit" in cases[name]["reason"], name

    # §19: fake adapter calls are reported separately from real EM fits
    assert result["fake_fit_calls_total"] == 4
    assert result["real_em_fits_executed"] == 0
    assert result["em_fits_executed"] == 0


def test_leakage_self_check_is_not_hard_coded():
    """§16: all_passed must be an AND over the cases, never a literal."""

    import ast as _ast
    import textwrap as _textwrap

    tree = _ast.parse(_textwrap.dedent(_inspect.getsource(H.run_leakage_self_check)))
    assigned = [node.value for node in _ast.walk(tree)
                if isinstance(node, _ast.Assign)
                and any(isinstance(t, _ast.Name) and t.id == "all_passed" for t in node.targets)]
    assert len(assigned) == 1
    value = assigned[0]
    assert not isinstance(value, _ast.Constant), "all_passed is hard-coded"
    assert isinstance(value, _ast.BoolOp) and isinstance(value.op, _ast.And)
    # one conjunct enumerates the declared cases, the other ANDs their verdicts
    rendered = [_ast.unparse(v) for v in value.values]
    assert any("LEAKAGE_SELF_CHECK_CASE_NAMES" in r for r in rendered), rendered
    assert any(r.startswith("all(") and "passed" in r for r in rendered), rendered


def test_self_check_fails_if_A01_guard_disabled(monkeypatch):
    monkeypatch.setattr(H, "_require_train_only_payload", lambda *a, **k: None)
    result = H.run_leakage_self_check()
    assert result["all_passed"] is False
    assert result["cases"]["A01_raw_held_out_y"]["passed"] is False


def test_self_check_fails_if_A02_guard_disabled(monkeypatch):
    monkeypatch.setattr(H, "_require_adapter_authority", lambda *a, **k: None)
    result = H.run_leakage_self_check()
    assert result["all_passed"] is False
    assert result["cases"]["A02_malicious_adapter"]["passed"] is False


def test_self_check_fails_if_A03_guard_disabled(monkeypatch):
    monkeypatch.setattr(H, "_require_post_fit_masks", lambda *a, **k: None)
    result = H.run_leakage_self_check()
    assert result["all_passed"] is False
    for name in ("A03_test_mask_mutation", "A03_train_mask_mutation",
                 "A03_both_mask_mutation"):
        assert result["cases"][name]["passed"] is False, name


def test_self_check_fails_if_a_case_raises(monkeypatch):
    """A broken case fails closed instead of quietly disappearing."""

    def boom():
        raise RuntimeError("broken case")

    cases = tuple(("A01_raw_held_out_y", boom, True, 0, "raw held-out Y") if c[0] ==
                  "A01_raw_held_out_y" else c for c in H.LEAKAGE_SELF_CHECK_CASES)
    monkeypatch.setattr(H, "LEAKAGE_SELF_CHECK_CASES", cases)
    result = H.run_leakage_self_check()
    assert result["all_passed"] is False
    assert "RuntimeError" in result["cases"]["A01_raw_held_out_y"]["reason"]


def test_leakage_gate_pass_is_derived_from_self_check(monkeypatch):
    """§18: LEAKAGE_GATE_PASS is the self-check verdict, not a constant."""

    assert H.current_smoke_authorization().leakage_gate_pass is True

    monkeypatch.setattr(H, "run_leakage_self_check",
                        lambda: {"all_passed": False, "cases": {}, "em_fits_executed": 0})
    assert H.current_smoke_authorization().leakage_gate_pass is False

    source = _inspect.getsource(H.current_smoke_authorization)
    assert "run_leakage_self_check()" in source
    assert 'leakage_gate_pass=bool(leakage["all_passed"])' in source


def test_leakage_self_check_runs_no_em_in_a_fresh_process():
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(HERE) + "');"
        "import run_k_true_robustness_sweep as H;"
        "r = H.run_leakage_self_check();"
        "print(r['all_passed'], r['fake_fit_calls_total'], r['real_em_fits_executed'],"
        " 'em_runner' in sys.modules, 'model_dual_expfam_consistent' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True 4 0 False False"


# --- smoke authorization ---------------------------------------------------


def test_S2_full_stays_hard_stopped_and_smoke_only_reaches_the_guard(monkeypatch):
    full_reached = _block_full_production_execution(monkeypatch)
    reached = _block_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"])
    assert "not authorized" in str(excinfo.value)
    assert reached == [], "--full must never reach the SMOKE production workflow"
    assert full_reached == [], "--full must not reach its own workflow either"

    for command in (["--smoke", "--allow-em"], ["--canary", "--allow-em"]):
        with pytest.raises(HarnessStop):
            H.main(command)
    assert [name for name, _auth in reached] == ["smoke", "canary"]


def test_S2_authorization_human_gates_are_never_granted_by_code():
    authorization = H.current_smoke_authorization()
    assert authorization.independent_review_pass is False
    assert authorization.human_smoke_approval is False
    assert not authorization.authorized()
    missing = authorization.missing()
    assert "INDEPENDENT_REVIEW_PASS" in missing and "HUMAN_SMOKE_APPROVAL" in missing
    assert set(H.HUMAN_ONLY_SMOKE_GATES) == {"INDEPENDENT_REVIEW_PASS", "HUMAN_SMOKE_APPROVAL"}


def test_S2_machine_gates_pass():
    authorization = H.current_smoke_authorization()
    assert authorization.zero_em_gate_pass is True
    assert authorization.leakage_gate_pass is True
    assert authorization.anchor_mask_gate_pass is True
    assert set(H.SMOKE_GATE_NAMES) == {
        "ZERO_EM_GATE_PASS", "LEAKAGE_GATE_PASS", "ANCHOR_MASK_GATE_PASS",
        "INDEPENDENT_REVIEW_PASS", "HUMAN_SMOKE_APPROVAL",
    }


# --- audit: leakage-gate provenance schema ---------------------------------


def _leakage_gate_rows(estimand: str, anchors, mutate=None) -> list[tuple]:
    role = H.resolve_role(estimand)
    rows = []
    for row in H.build_manifest(estimand):
        a = anchors[row.replicate]
        rows.append([
            estimand, role, row.k_true, row.replicate, row.k, row.start,
            a.test_mask_hash, a.train_mask_hash, a.test_mask_hash, a.train_mask_hash,
            a.test_mask_hash, a.train_mask_hash, "True", "True", "clean",
            A.LEAKAGE_BOUNDARY_VERSION,
        ])
    if mutate:
        mutate(rows)
    return rows


def test_S2_audit_leakage_gate_positive_control(anchors):
    auditor = A.Auditor()
    rows = [dict(zip(A.LEAKAGE_GATE_COLUMNS, r)) for r in _leakage_gate_rows("B", anchors)]
    anchor_map = {r: (a.test_mask_hash, a.train_mask_hash) for r, a in anchors.items()}
    A.audit_leakage_gate(rows, "B", anchor_map, auditor)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]


@pytest.mark.parametrize("column,check", [
    ("post_fit_test_mask_hash", "leakage_test_mask_changed"),
    ("post_fit_train_mask_hash", "leakage_train_mask_changed"),
    ("pre_fit_passed", "leakage_pre_fit_failed"),
    ("post_fit_passed", "leakage_post_fit_failed"),
    ("fit_boundary_status", "leakage_boundary_status"),
    ("boundary_version", "leakage_boundary_version"),
])
def test_S2_audit_leakage_gate_negatives(anchors, column, check):
    auditor = A.Auditor()
    rows = [dict(zip(A.LEAKAGE_GATE_COLUMNS, r)) for r in _leakage_gate_rows("B", anchors)]
    rows[0][column] = "tampered"
    anchor_map = {r: (a.test_mask_hash, a.train_mask_hash) for r, a in anchors.items()}
    A.audit_leakage_gate(rows, "B", anchor_map, auditor)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))


def test_S2_audit_leakage_gate_row_count_and_duplicates(anchors):
    anchor_map = {r: (a.test_mask_hash, a.train_mask_hash) for r, a in anchors.items()}
    rows = [dict(zip(A.LEAKAGE_GATE_COLUMNS, r)) for r in _leakage_gate_rows("B", anchors)]

    auditor = A.Auditor()
    A.audit_leakage_gate(rows[:-1], "B", anchor_map, auditor)
    assert any(f.check == "leakage_row_count" for f in auditor.blockers)

    auditor2 = A.Auditor()
    A.audit_leakage_gate(rows[:-1] + [dict(rows[0])], "B", anchor_map, auditor2)
    assert any(f.check == "leakage_duplicate_key" for f in auditor2.blockers)


# ===========================================================================
# Phase 8b S2b — human-gated real canary + frozen 6-fit smoke (Issue #53)
# ===========================================================================
#
# Every test here drives the PRODUCTION orchestration with a Phase 7e
# test-only fit adapter.  Real EM executions remain 0 and ``em_runner`` is
# never imported.


from run_heldout_k_selection_pilot import (  # noqa: E402
    CanaryFitResult,
    InitializationSnapshot,
    _TestAuthorizedFitAdapter,
)


class _FakeFitRecorder:
    """Counts fits and records the exact arguments the boundary supplied."""

    def __init__(self, forced_scores=None):
        self.calls = 0
        self.seeds = []
        self.k_values = []
        self.saw_score_target = False
        self._forced = forced_scores or {}

    def __call__(self, **kwargs):
        self.calls += 1
        self.seeds.append(kwargs["seed"])
        self.k_values.append(kwargs["k"])
        assert set(kwargs) >= {"X", "Y", "train_mask", "k", "seed"}
        assert "score_target" not in kwargs and "test_mask" not in kwargs
        k = kwargs["k"]
        n = kwargs["Y"].shape[0]
        rng = _np.random.default_rng(kwargs["seed"])
        Z = rng.normal(size=(n, k))
        F = rng.normal(size=(kwargs["X"].shape[1], k))
        # A forced Z scale lets a test steer which K wins without touching gates.
        scale = self._forced.get(k)
        if scale is not None:
            Z = Z * 0.0 + scale
        init = InitializationSnapshot(Z=Z.copy(), F=F.copy(), w0=-1.0, w=1.5, sigma_y=None)
        return CanaryFitResult(
            initialization=init, Z=Z, F=F, w0=-1.0, w=1.5, sigma_y=None,
            Q_strict=-123.0, train_objective_diagnostics={"ok": 1.0},
            internal_retry=0, q_failure=False, warnings=(), nan_occurred=False,
        )


def _test_authorization(**overrides):
    return H._make_test_smoke_authorization(**overrides)


def _test_adapter(recorder):
    return H._make_test_fit_adapter(recorder, score_targets=())


# --- frozen smoke protocol -------------------------------------------------


def test_S2b_smoke_manifest_is_exactly_six_rows():
    manifest = H.build_smoke_manifest()
    assert len(manifest) == H.EXPECTED_SMOKE_FITS == 6
    assert [(row.k, row.start) for row in manifest] == [
        (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)]
    assert [row.fit_index for row in manifest] == [1, 2, 3, 4, 5, 6]


def test_S2b_smoke_estimand_is_primary_A_only():
    manifest = H.build_smoke_manifest()
    assert H.SMOKE_ESTIMAND == "A" and H.SMOKE_ROLE == "primary"
    assert H.SMOKE_ROLE == H.resolve_role(H.SMOKE_ESTIMAND)
    assert {row.estimand for row in manifest} == {"A"}
    assert {row.role for row in manifest} == {"primary"}


def test_S2b_B_sensitivity_is_not_dropped_from_the_full_run():
    """Smoke is A-only for budget reasons; B stays a full-run estimand."""

    assert H.active_estimands() == ("A", "B")
    assert len(H.build_manifest("B")) == H.FITS_PER_ESTIMAND == 168
    assert H.resolve_role("B") == "sensitivity"
    assert H.EXPECTED_NEW_FITS == 336


def test_S2b_smoke_cell_is_k_true_one_replicate_one():
    manifest = H.build_smoke_manifest()
    assert H.SMOKE_K_TRUE == 1 and H.SMOKE_REPLICATE == 1
    assert {row.k_true for row in manifest} == {1}
    assert {row.replicate for row in manifest} == {1}


def test_S2b_frozen_smoke_seeds_are_exact():
    assert H.SMOKE_DATA_SEED_BASE == 61000
    assert H.SMOKE_MODEL_SEED_BASE == 630000
    assert H.smoke_data_seed(1, 1) == 61101
    assert H.smoke_model_seed(1, 1, 2, 1) == 641021
    assert H.smoke_model_seed(1, 1, 2, 2) == 641022
    assert H.smoke_model_seed(1, 1, 3, 1) == 641031
    assert H.smoke_model_seed(1, 1, 3, 2) == 641032
    assert H.smoke_model_seed(1, 1, 4, 1) == 641041
    assert H.smoke_model_seed(1, 1, 4, 2) == 641042
    assert H.CANARY_MODEL_SEED == H.smoke_model_seed(1, 1, 1, 1) == 641011

    manifest = H.build_smoke_manifest()
    assert {row.data_seed for row in manifest} == {61101}
    assert [row.model_seed for row in manifest] == [
        641021, 641022, 641031, 641032, 641041, 641042]


def test_S2b_smoke_seeds_are_disjoint_from_phase7e_and_full():
    report = H.check_smoke_seed_collisions()
    for key in ("phase7e_data_overlap", "phase7e_model_overlap",
                "phase8_full_data_overlap", "phase8_full_model_overlap"):
        assert report[key] == [], (key, report[key])
    smoke = H.smoke_seed_space()
    assert smoke["data"] == frozenset({61101})
    assert len(smoke["model"]) == 7          # 6 smoke + 1 canary
    assert report["split_seed_excluded"] is True


def test_S2b_split_seed_is_S_C_and_carries_no_smoke_offset():
    assert H.SMOKE_SPLIT_SEED == 42001
    assert H.smoke_split_seed(1, 1) == H.expected_split_seed(1, 1) == 42001
    assert {row.split_seed for row in H.build_smoke_manifest()} == {42001}

    executable = _executable_body(H.smoke_split_seed)
    for forbidden in ("SMOKE_DATA_SEED_BASE", "SMOKE_MODEL_SEED_BASE",
                      "estimand", "ESTIMAND_SEED_OFFSET", "61000", "630000"):
        assert forbidden not in executable, forbidden
    assert "expected_split_seed" in executable


def test_S2b_smoke_masks_equal_the_phase7e_replicate_one_anchor(anchors):
    split = H.build_split_record(H.SMOKE_K_TRUE, H.SMOKE_REPLICATE)
    anchor = anchors[1]
    assert split.split_mask_hash == anchor.test_mask_hash
    assert split.train_mask_hash == anchor.train_mask_hash
    for row in H.build_smoke_manifest():
        assert row.split_mask_hash == row.anchor_mask_hash == anchor.test_mask_hash
        assert row.train_mask_hash == row.anchor_train_mask_hash == anchor.train_mask_hash


def test_S2b_smoke_mask_gate_requires_both_sides(anchors):
    split = H.build_split_record(H.SMOKE_K_TRUE, H.SMOKE_REPLICATE)
    other = H.build_split_record(H.NEW_K_TRUE[0], 2)
    good = anchors[1]

    H._require_smoke_anchor_masks(split, good)          # positive control

    test_only_match = H.AnchorMask(replicate=1, test_mask_hash=good.test_mask_hash,
                                   train_mask_hash=other.train_mask_hash,
                                   source=good.source)
    with pytest.raises(HarnessStop) as excinfo:
        H._require_smoke_anchor_masks(split, test_only_match)
    assert "train mask" in str(excinfo.value)

    train_only_match = H.AnchorMask(replicate=1, test_mask_hash=other.split_mask_hash,
                                    train_mask_hash=good.train_mask_hash,
                                    source=good.source)
    with pytest.raises(HarnessStop) as excinfo2:
        H._require_smoke_anchor_masks(split, train_only_match)
    assert "test mask" in str(excinfo2.value)


def test_S2b_canary_protocol_is_exact():
    config = H.smoke_canary_config()
    assert H.CANARY_K_EST == 1 and H.CANARY_START == 1
    assert config.k_est == 1
    assert config.seed == 641011
    assert config.family_x == H.FAMILY_X and config.family_y == H.FAMILY_Y
    assert config.L == H.L_SAMPLES and config.num_iter == H.NUM_ITER
    assert config.numerics_mode == H.NUMERICS_MODE


def test_S2b_future_real_execution_budget_is_two_plus_six():
    assert H.EXPECTED_CANARY_FITS == 2
    assert H.EXPECTED_SMOKE_FITS == 6
    assert H.EXPECTED_REAL_EM_BUDGET == 8


def test_S2b_protocol_hash_is_stable_and_binds_the_protocol(monkeypatch):
    baseline = H.smoke_protocol_hash()
    assert baseline == H.smoke_protocol_hash()
    config = H.smoke_protocol_config()
    for key in ("estimand", "k_true", "replicate", "k_candidates", "starts",
                "data_seed_base", "model_seed_base", "split_seed",
                "expected_smoke_fits", "expected_canary_fits", "canary_model_seed"):
        assert key in config, key
    monkeypatch.setattr(H, "SMOKE_K_CANDIDATES", (2, 3, 5))
    assert H.smoke_protocol_hash() != baseline


# --- authorization contract ------------------------------------------------


def test_S2b_production_authorization_is_committed():
    """Issue #55 recorded the human approval; the record is a committed literal."""

    authorization = H.current_smoke_execution_authorization()
    assert authorization is not None and authorization.is_test_only() is False
    H.validate_smoke_execution_authorization(authorization, test_only=False)
    assert authorization.issue_number == 55


def test_S2b_no_cli_or_env_can_assert_human_approval(monkeypatch):
    options = {action.option_strings[0] for action in H._build_parser()._actions
               if action.option_strings}
    for forbidden in ("--human-approved", "--reviewed", "--independent-review-pass",
                      "--human-smoke-approval", "--approve"):
        assert forbidden not in options, forbidden

    for name in ("PHASE8B_HUMAN_SMOKE_APPROVAL", "HUMAN_SMOKE_APPROVAL",
                 "INDEPENDENT_REVIEW_PASS", "PHASE8B_SMOKE_AUTHORIZED"):
        monkeypatch.setenv(name, "1")
    # the committed record is unchanged by any of them
    authorization = H.current_smoke_execution_authorization()
    assert authorization.human_smoke_approval is True
    assert authorization.approved_main_sha == H.APPROVED_SCIENTIFIC_MAIN_SHA
    assert H.current_smoke_authorization().authorized() is False

    executable = _executable_body(H.current_smoke_execution_authorization)
    for forbidden in ("environ", "getenv", "argv", "args", "input(", "open("):
        assert forbidden not in executable, forbidden
    assert H.current_smoke_execution_authorization().human_smoke_approval is True


def test_S2b_authorization_requires_the_production_authority():
    test_auth = _test_authorization()
    # a test record is fine for the test path ...
    H.validate_smoke_execution_authorization(test_auth, test_only=True)
    # ... but must never satisfy the production path
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_smoke_execution_authorization(test_auth, test_only=False)
    assert "provenance is unauthorized" in str(excinfo.value)

    forged = dataclasses.replace(test_auth, _authority=object())
    for test_only in (True, False):
        with pytest.raises(HarnessStop):
            H.validate_smoke_execution_authorization(forged, test_only=test_only)


@pytest.mark.parametrize("field,value", [
    ("issue_number", 999),
    ("protocol_hash", "tampered"),
    ("estimand", "B"),
    ("k_true", 3),
    ("replicate", 2),
    ("smoke_fit_count", 12),
    ("canary_fit_count", 1),
    ("data_seed_base", 51000),
    ("model_seed_base", 530000),
    ("split_seed", 42002),
    ("authorization_version", "v0"),
    ("independent_review_pass", False),
    ("human_smoke_approval", False),
    ("approved_main_sha", "not-a-sha"),
    ("approved_main_sha", "A" * 40),
])
def test_S2b_authorization_field_mismatch_fails_closed(field, value):
    authorization = _test_authorization(**{field: value})
    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(authorization, test_only=True)


def test_S2b_authorization_rejects_non_records():
    for bad in (None, object(), {"human_smoke_approval": True}):
        with pytest.raises(HarnessStop):
            H.validate_smoke_execution_authorization(bad, test_only=True)


def test_S2b_test_authority_is_unreachable_from_production():
    """The CLI and the production entry points never name the test factories."""

    production = (H.main, H._require_em_authorization, H._build_parser,
                  H.run_real_canary, H.run_real_smoke,
                  H.current_smoke_execution_authorization)
    forbidden = ("_make_test_smoke_authorization", "_make_test_fit_adapter",
                 "_run_real_canary_test_only", "_run_real_smoke_test_only",
                 "_SMOKE_TEST_AUTHORITY", "_TestAuthorizedFitAdapter")
    for function in production:
        source = _inspect.getsource(function)
        for name in forbidden:
            assert name not in source, (function.__name__, name)


# --- production entry points are implemented but disabled ------------------


def test_S2b_production_entrypoints_refuse_a_test_only_authorization(monkeypatch):
    """A test-only record is still refused; the committed one only reaches the guard."""

    # no guard yet: the workflow validates the authorization as its FIRST step,
    # so a test-only record is refused before any preflight, directory or adapter
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    for entrypoint in (H.run_real_canary, H.run_real_smoke):
        with pytest.raises(HarnessStop) as excinfo:
            entrypoint(_test_authorization())
        assert "provenance is unauthorized" in str(excinfo.value)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    _assert_no_new_production_artifacts()

    reached = _block_production_execution(monkeypatch)
    for entrypoint in (H.run_real_canary, H.run_real_smoke):
        with pytest.raises(HarnessStop):
            entrypoint(H.current_smoke_execution_authorization())
    assert [name for name, _auth in reached] == ["canary", "smoke"]


def test_S2b_cli_canary_and_smoke_pass_the_committed_authorization_through(monkeypatch):
    reached = _block_production_execution(monkeypatch)
    for command in (["--canary", "--allow-em"], ["--smoke", "--allow-em"]):
        with pytest.raises(HarnessStop):
            H.main(command)
    assert [name for name, _auth in reached] == ["canary", "smoke"]
    for _name, authorization in reached:
        assert type(authorization) is H.SmokeExecutionAuthorization
        assert authorization.is_test_only() is False


def test_S2b_full_remains_blocked_by_its_own_gate(monkeypatch):
    """Issue #59 gave --full its OWN schema; a smoke record still cannot reach it."""

    reached = _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"])
    message = str(excinfo.value)
    assert "not authorized" in message
    assert "never be reused for --full" in message
    assert reached == []
    executable = _executable_body(H._require_em_authorization)
    full_branch = executable.split("if command == 'full':")[1].split('_require(command in')[0]
    assert "current_smoke_execution_authorization" not in full_branch
    assert "current_full_execution_authorization" in full_branch


def test_S2b_real_adapter_is_never_reached_by_the_test_suite(monkeypatch):
    class Tripwire:
        def fit(self, invocation):          # pragma: no cover - must never run
            raise AssertionError("the real EM adapter was reached from a test")

    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", Tripwire)
    _block_production_execution(monkeypatch)
    _block_full_production_execution(monkeypatch)
    for command in (["--canary", "--allow-em"], ["--smoke", "--allow-em"],
                    ["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"]):
        with pytest.raises(HarnessStop):
            H.main(command)
    for entrypoint in (H.run_real_canary, H.run_real_smoke):
        with pytest.raises(HarnessStop):
            entrypoint(_test_authorization())
    assert "em_runner" not in sys.modules


# --- fake orchestration: canary --------------------------------------------


def test_S2b_canary_runs_exactly_two_fake_fits_from_one_state():
    recorder = _FakeFitRecorder()
    report = H._run_real_canary_test_only(_test_authorization(),
                                          adapter=_test_adapter(recorder))
    assert recorder.calls == 2
    assert recorder.seeds == [641011, 641011], "both canary fits must share the model seed"
    assert recorder.k_values == [1, 1]
    assert report.invariance.initialization_equal is True
    assert report.invariance.final_outputs_equal is True
    assert report.invariance.internal_retry == 0
    assert report.invariance.fit_payload_a_hash != report.invariance.fit_payload_b_hash
    assert report.model_seed == 641011 and report.data_seed == 61101
    assert report.split_seed == 42001 and report.k_est == 1 and report.start == 1
    assert report.real_canary_fits_executed == 0 and report.test_only is True
    assert "em_runner" not in sys.modules


def test_S2b_canary_refuses_a_non_test_adapter():
    _row, _split, _Y, _target, _request, state = _leakage_fixture()
    for adapter in (H.SealedFakeFitAdapter(state), object()):
        with pytest.raises(HarnessStop):
            H._run_real_canary_test_only(_test_authorization(), adapter=adapter)


# --- fake orchestration: smoke ---------------------------------------------


def test_S2b_smoke_runs_exactly_six_fake_fits_in_order():
    recorder = _FakeFitRecorder()
    report = H._run_real_smoke_test_only(_test_authorization(),
                                         adapter=_test_adapter(recorder))
    assert recorder.calls == 6
    assert recorder.seeds == [641021, 641022, 641031, 641032, 641041, 641042]
    assert recorder.k_values == [2, 2, 3, 3, 4, 4]
    assert len(report.rows) == 6
    assert [(row.k, row.start) for row in report.rows] == [
        (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)]
    assert all(_np.isfinite(row.heldout_mean_log_score) for row in report.rows)
    assert all(row.fit_status == "clean" and row.internal_retry == 0
               and row.warning_count == 0 and row.q_failure is False
               and row.nan_occurred is False and row.finite_state is True
               for row in report.rows)
    assert [k for k, _ in report.mean_scores] == [2, 3, 4]
    assert report.real_smoke_fits_executed == 0 and report.test_only is True
    assert "em_runner" not in sys.modules


def test_S2b_smoke_two_start_means_are_unweighted():
    recorder = _FakeFitRecorder()
    report = H._run_real_smoke_test_only(_test_authorization(),
                                         adapter=_test_adapter(recorder))
    by_key = {(row.k, row.start): row.heldout_mean_log_score for row in report.rows}
    for k, mean in report.mean_scores:
        assert mean == pytest.approx((by_key[(k, 1)] + by_key[(k, 2)]) / 2.0, rel=0, abs=1e-15)


def test_S2b_score_target_is_created_only_after_all_six_fits(monkeypatch):
    """Phase A must complete before the outcome-bearing target exists."""

    events = []
    real_target = H.make_score_only_target

    def watched(Y, test_mask):
        events.append(("target", len(events)))
        return real_target(Y, test_mask)

    recorder = _FakeFitRecorder()

    class CountingRecorder(_FakeFitRecorder):
        def __call__(self, **kwargs):
            events.append(("fit", len(events)))
            return _FakeFitRecorder.__call__(self, **kwargs)

    counting = CountingRecorder()
    monkeypatch.setattr(H, "make_score_only_target", watched)
    H._run_real_smoke_test_only(_test_authorization(), adapter=_test_adapter(counting))

    kinds = [kind for kind, _ in events]
    assert kinds.count("fit") == 6
    assert kinds.count("target") == 1
    assert kinds.index("target") == 6, kinds
    del recorder


def test_S2b_smoke_refuses_a_non_test_adapter():
    _row, _split, _Y, _target, _request, state = _leakage_fixture()
    for adapter in (H.SealedFakeFitAdapter(state), object()):
        with pytest.raises(HarnessStop):
            H._run_real_smoke_test_only(_test_authorization(), adapter=adapter)


# --- selected_k is never a gate --------------------------------------------


@pytest.mark.parametrize("winner", [2, 3, 4])
def test_S2b_selected_k_outcome_does_not_change_smoke_success(winner):
    """Whichever K wins, the smoke completes identically.  K_TRUE=1 is not a candidate."""

    forced = {k: (0.9 if k == winner else 0.1) for k in H.SMOKE_K_CANDIDATES}
    recorder = _FakeFitRecorder(forced_scores=forced)
    report = H._run_real_smoke_test_only(_test_authorization(),
                                         adapter=_test_adapter(recorder))
    assert recorder.calls == 6
    assert len(report.rows) == 6
    assert report.selected_k in H.SMOKE_K_CANDIDATES
    assert H.SMOKE_K_TRUE not in H.SMOKE_K_CANDIDATES, \
        "selected_k == K_TRUE is structurally impossible in smoke"


def test_S2b_selected_k_is_not_referenced_by_any_gate():
    """Static check: no blocking condition reads selected_k."""

    for function in (H._run_smoke_fit_phase_8b, H.prepare_smoke_cell,
                     H.validate_smoke_execution_authorization,
                     H.validate_smoke_manifest, H._require_em_authorization,
                     H.current_smoke_authorization):
        assert "selected_k" not in _inspect.getsource(function), function.__name__
    scoring = _inspect.getsource(H._run_real_smoke)
    gating = [line for line in scoring.splitlines()
              if "selected_k" in line and ("_require(" in line or "assert " in line)]
    assert gating == [], gating


# --- failure tests: fail closed --------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("estimand", "B"),
    ("role", "sensitivity"),
    ("k_true", 2),
    ("replicate", 2),
    ("k", 5),
    ("start", 3),
    ("data_seed", 51101),
    ("model_seed", 541021),
    ("split_seed", 52001),
    ("w_true", 9.9),
    ("w0_true", 0.5),
    ("mask_design", "S_A"),
])
def test_S2b_tampered_smoke_manifest_row_is_rejected(field, value):
    manifest = H.build_smoke_manifest()
    manifest[0] = dataclasses.replace(manifest[0], **{field: value})
    with pytest.raises(HarnessStop):
        H.validate_smoke_manifest(manifest)


def test_S2b_wrong_smoke_manifest_length_is_rejected():
    manifest = H.build_smoke_manifest()
    with pytest.raises(HarnessStop):
        H.validate_smoke_manifest(manifest[:-1])
    with pytest.raises(HarnessStop):
        H.validate_smoke_manifest(manifest + [manifest[-1]])


def test_S2b_reordered_smoke_manifest_is_rejected():
    manifest = H.build_smoke_manifest()
    swapped = [manifest[1], manifest[0]] + manifest[2:]
    with pytest.raises(HarnessStop):
        H.validate_smoke_manifest(swapped)


def test_S2b_anchor_mask_mismatch_stops_before_any_fit(monkeypatch):
    other = H.build_split_record(H.NEW_K_TRUE[0], 2)
    monkeypatch.setattr(H, "build_split_record", lambda k_true, replicate: other)
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._run_real_smoke_test_only(_test_authorization(), adapter=_test_adapter(recorder))
    assert "Phase 7e anchor" in str(excinfo.value)
    assert recorder.calls == 0


def test_S2b_protocol_hash_drift_stops_before_any_fit(monkeypatch):
    authorization = _test_authorization()
    monkeypatch.setattr(H, "SMOKE_K_CANDIDATES", (2, 3, 5))
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._run_real_smoke_test_only(authorization, adapter=_test_adapter(recorder))
    assert "protocol_hash" in str(excinfo.value) or "protocol hash" in str(excinfo.value)
    assert recorder.calls == 0


def test_S2b_seed_collision_stops_before_any_fit(monkeypatch):
    monkeypatch.setattr(H, "SMOKE_DATA_SEED_BASE", H.DATA_SEED_BASE)
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop):
        H._run_real_smoke_test_only(_test_authorization(), adapter=_test_adapter(recorder))
    assert recorder.calls == 0


def test_S2b_failing_zero_em_gate_stops_before_any_fit(monkeypatch):
    monkeypatch.setattr(H, "run_leakage_self_check",
                        lambda: {"all_passed": False, "cases": {},
                                 "em_fits_executed": 0, "real_em_fits_executed": 0})
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._run_real_smoke_test_only(_test_authorization(), adapter=_test_adapter(recorder))
    assert "leakage self-check" in str(excinfo.value)
    assert recorder.calls == 0


def test_S2b_smoke_fit_config_rejects_a_foreign_row():
    full_row = H.build_manifest("A")[0]
    with pytest.raises(HarnessStop):
        H.smoke_fit_config(full_row)


# --- future artifact schema (definition only) ------------------------------


def test_S2b_artifact_schema_is_defined_and_nothing_is_written(tmp_path):
    required = {
        "run_code_sha", "approved_scientific_main_sha", "protocol_hash", "estimand", "role",
        "K_TRUE", "replicate", "K", "start", "data_seed", "split_seed", "model_seed",
        "pre_fit_test_hash", "pre_fit_train_hash", "post_fit_test_hash",
        "post_fit_train_hash", "anchor_test_hash", "anchor_train_hash",
        "boundary_version", "fit_status", "internal_retry", "warning_count",
        "q_failure", "nan_occurred", "finite_state", "heldout_mean_log_score",
        "score_config_hash", "canary_provenance", "real_canary_fits_executed",
        "real_smoke_fits_executed",
    }
    assert set(H.SMOKE_ARTIFACT_COLUMNS) == required
    assert len(H.SMOKE_ARTIFACT_COLUMNS) == len(required)

    canary = H._run_real_canary_test_only(_test_authorization(),
                                          adapter=_test_adapter(_FakeFitRecorder()))
    smoke = H._run_real_smoke_test_only(_test_authorization(),
                                        adapter=_test_adapter(_FakeFitRecorder()))
    rows = H.build_smoke_artifact_rows(canary, smoke, "0" * 40)
    assert len(rows) == 6
    assert all(len(row) == len(H.SMOKE_ARTIFACT_COLUMNS) for row in rows)
    by_name = dict(zip(H.SMOKE_ARTIFACT_COLUMNS, rows[0]))
    assert by_name["real_canary_fits_executed"] == 0
    assert by_name["real_smoke_fits_executed"] == 0
    assert list(tmp_path.iterdir()) == [], "S2b must not write any artifact"


def test_S2b_no_smoke_artifact_is_created():
    """S2b defines the schema only: the results tree must be untouched.

    (Pre-existing Phase 7d ``k_selection_score_pilot_smoke_*`` files are a
    different experiment; the check is that S2b adds or changes nothing.)
    """

    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "expfam/results"],
        capture_output=True, text=True, cwd=ROOT)
    assert status.returncode == 0, status.stderr
    assert status.stdout.strip() == "", status.stdout

    assert not hasattr(H, "write_smoke_artifacts")
    assert not hasattr(H, "run_smoke_cli")
    for name in dir(H):
        assert not (name.startswith("write_") and "smoke" in name), name


# --- real EM exclusion -----------------------------------------------------


def test_S2b_fake_orchestration_imports_no_em_in_a_fresh_process():
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(HERE) + "');"
        "import numpy as np;"
        "import run_k_true_robustness_sweep as H;"
        "from run_heldout_k_selection_pilot import CanaryFitResult, InitializationSnapshot;"
        "calls = [];"
        "\ndef fake(**kw):\n"
        "    calls.append(kw['seed']);\n"
        "    Z = np.random.default_rng(kw['seed']).normal(size=(kw['Y'].shape[0], kw['k']));\n"
        "    init = InitializationSnapshot(Z=Z.copy(), F=Z[:1].copy(), w0=-1.0, w=1.5, sigma_y=None);\n"
        "    return CanaryFitResult(initialization=init, Z=Z, F=Z[:1].copy(), w0=-1.0, w=1.5,\n"
        "        sigma_y=None, Q_strict=-1.0, train_objective_diagnostics={},\n"
        "        internal_retry=0, q_failure=False, warnings=(), nan_occurred=False)\n"
        "a = H._make_test_smoke_authorization();"
        "H._run_real_canary_test_only(a, adapter=H._make_test_fit_adapter(fake, score_targets=()));"
        "H._run_real_smoke_test_only(a, adapter=H._make_test_fit_adapter(fake, score_targets=()));"
        "print(len(calls), 'em_runner' in sys.modules,"
        " 'model_dual_expfam_consistent' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "8 False False"


def test_S2b_zero_em_modes_report_the_execution_authorization_state(capsys):
    assert H.main(["--smoke-authorization"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_authorization_present"] is True
    # the machine-checkable gate record is a different thing and still False
    assert payload["authorized"] is False
    assert payload["expected_canary_fits"] == 2
    assert payload["expected_smoke_fits"] == 6
    assert payload["expected_real_em_budget"] == 8
    assert payload["real_canary_fits_executed"] == 0
    assert payload["real_smoke_fits_executed"] == 0
    assert payload["smoke_protocol_hash"] == H.smoke_protocol_hash()
    assert payload["em_fits_executed"] == 0


# ---------------------------------------------------------------------------
# HIGH-01: the approved main SHA must bind to a TRUSTED reviewed SHA
# ---------------------------------------------------------------------------
#
# Format validation alone let any well-formed SHA through to the real adapter.
# The gate is now identity against a single trusted source that no caller,
# environment variable, CLI flag or repository state can influence.


class _AdapterTripwire:
    """Counts constructions AND fits of the stand-in real EM adapter."""

    constructions = 0
    fits = 0

    def __init__(self):
        type(self).constructions += 1

    def fit(self, invocation):          # pragma: no cover - must never run
        type(self).fits += 1
        raise AssertionError("the real EM adapter was reached")

    @classmethod
    def reset(cls):
        cls.constructions = 0
        cls.fits = 0


def _production_authorization(**overrides):
    """A record carrying the PRODUCTION authority sentinel.

    Building one here is the exact path the reviewer found; the test exists to
    prove it still cannot reach a real adapter.  No production code issues one.
    """

    fields = {
        "issue_number": H.SMOKE_EXECUTION_ISSUE_NUMBER,
        "approved_main_sha": "1" * H.SMOKE_SHA_LENGTH,
        "protocol_hash": H.smoke_protocol_hash(),
        "estimand": H.SMOKE_ESTIMAND,
        "k_true": H.SMOKE_K_TRUE,
        "replicate": H.SMOKE_REPLICATE,
        "smoke_fit_count": H.EXPECTED_SMOKE_FITS,
        "canary_fit_count": H.EXPECTED_CANARY_FITS,
        "data_seed_base": H.SMOKE_DATA_SEED_BASE,
        "model_seed_base": H.SMOKE_MODEL_SEED_BASE,
        "split_seed": H.SMOKE_SPLIT_SEED,
        "independent_review_pass": True,
        "human_smoke_approval": True,
        "authorization_version": H.SMOKE_AUTHORIZATION_VERSION,
    }
    fields.update(overrides)
    return H.SmokeExecutionAuthorization(_authority=H._SMOKE_EXECUTION_AUTHORITY, **fields)


def test_S2c_trusted_main_sha_is_the_approved_baseline():
    """Issue #55 §6: the reviewed PR #54 merge commit, exactly."""

    assert H.APPROVED_SCIENTIFIC_MAIN_SHA == "68c78e1191889609dead05ea5a9fb11525ce92e2"
    assert H.current_expected_smoke_main_sha() == H.APPROVED_SCIENTIFIC_MAIN_SHA
    H._require_full_commit_sha(H.APPROVED_SCIENTIFIC_MAIN_SHA, "baseline")
    # the baseline binding and the execution authorization stay separate values:
    # the record is checked AGAINST this source, never derived from it
    assert H.current_smoke_execution_authorization().approved_main_sha == \
        H.current_expected_smoke_main_sha()
    assert H.current_smoke_authorization().authorized() is False


def test_HIGH01_trusted_sha_source_is_a_literal(monkeypatch):
    """§21: the trusted source cannot be steered by env, CLI, git or the record."""

    executable = _executable_body(H.current_expected_smoke_main_sha)
    assert executable.strip() == "return APPROVED_SCIENTIFIC_MAIN_SHA"
    for forbidden in ("getenv", "environ", "argv", "subprocess", "git", "run(",
                      "rev-parse", "approved_main_sha", "authorization", "Path", "open("):
        assert forbidden not in executable, forbidden

    for name in ("APPROVED_MAIN_SHA", "SMOKE_APPROVED_SHA", "HUMAN_SMOKE_APPROVAL",
                 "INDEPENDENT_REVIEW_PASS", "PHASE8B_APPROVED_MAIN_SHA"):
        monkeypatch.setenv(name, "b" * 40)
    assert H.current_expected_smoke_main_sha() == H.APPROVED_SCIENTIFIC_MAIN_SHA
    assert H.current_smoke_execution_authorization().approved_main_sha == \
        H.APPROVED_SCIENTIFIC_MAIN_SHA

    options = {option for action in H._build_parser()._actions
               for option in action.option_strings}
    for forbidden in ("--approved-main-sha", "--main-sha", "--sha", "--approved",
                      "--human-approved", "--reviewed"):
        assert forbidden not in options, forbidden


def test_HIGH01_expected_sha_is_not_a_public_entrypoint_parameter():
    """§7/§12: callers cannot supply the value they are checked against."""

    for entrypoint in (H.run_real_canary, H.run_real_smoke,
                       H.validate_smoke_execution_authorization,
                       H.prepare_smoke_cell, H.main):
        parameters = _inspect.signature(entrypoint).parameters
        assert "expected_main_sha" not in parameters, entrypoint.__name__
        assert "approved_main_sha" not in parameters, entrypoint.__name__
    # the internal validator is the only place that takes it, and it is private
    assert "expected_main_sha" in _inspect.signature(
        H._validate_smoke_execution_authorization).parameters


def test_HIGH01_identity_gate_is_not_a_self_comparison():
    """§22: the record must be compared to the trusted value, not to itself."""

    executable = _executable_body(H._validate_smoke_execution_authorization)
    assert "authorization.approved_main_sha == expected_main_sha" in executable
    assert "authorization.approved_main_sha == authorization.approved_main_sha" not in executable

    production = _executable_body(H.validate_smoke_execution_authorization)
    assert "trusted_main_sha_for(test_only)" in production
    assert "authorization.approved_main_sha" not in production

    # and the single trusted resolver reads the production source for real runs
    resolver = _executable_body(H.trusted_main_sha_for)
    assert "current_expected_smoke_main_sha()" in resolver
    assert "authorization" not in resolver
    assert H.trusted_main_sha_for(False) == H.current_expected_smoke_main_sha()
    assert H.trusted_main_sha_for(True) == H._TEST_EXPECTED_MAIN_SHA


def test_HIGH01_production_wrapper_never_selects_the_test_path():
    """Production callers always pass test_only=False literally."""

    for caller in (H.run_real_canary, H.run_real_smoke, H.run_real_canary_cli,
                   H.run_real_smoke_cli):
        source = _inspect.getsource(caller)
        assert "test_only=True" not in source
        assert "_TEST_EXPECTED_MAIN_SHA" not in source
    # they all delegate to the one production workflow, which is test_only=False
    workflow = _inspect.getsource(H._run_production_execution)
    assert "test_only=False" in workflow
    assert "test_only=True" not in workflow
    for name in ("main", "_require_em_authorization", "_build_parser",
                 "current_expected_smoke_main_sha"):
        assert "_TEST_EXPECTED_MAIN_SHA" not in _inspect.getsource(getattr(H, name)), name


# --- §13: trusted SHA absent ----------------------------------------------


@pytest.mark.parametrize("entrypoint", ["run_real_canary", "run_real_smoke"])
def test_HIGH01_absent_trusted_sha_blocks_a_valid_looking_authorization(monkeypatch,
                                                                       entrypoint):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    monkeypatch.setattr(H, "current_expected_smoke_main_sha", lambda: None)

    authorization = _production_authorization(approved_main_sha="1" * 40)
    with pytest.raises(HarnessStop) as excinfo:
        getattr(H, entrypoint)(authorization)

    assert "no reviewed main SHA has been authorized" in str(excinfo.value)
    assert _AdapterTripwire.constructions == 0
    assert _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules


# --- §14: trusted SHA present but different (the key regression) -----------


@pytest.mark.parametrize("entrypoint", ["run_real_canary", "run_real_smoke"])
def test_HIGH01_wrong_but_well_formed_sha_is_rejected(monkeypatch, entrypoint):
    """Both SHAs are format-valid; only identity separates them."""

    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    monkeypatch.setattr(H, "current_expected_smoke_main_sha", lambda: "a" * 40)

    authorization = _production_authorization(approved_main_sha="b" * 40)
    H._require_full_commit_sha(authorization.approved_main_sha, "record")   # format is fine
    H._require_full_commit_sha("a" * 40, "trusted")

    with pytest.raises(HarnessStop) as excinfo:
        getattr(H, entrypoint)(authorization)

    assert "approved main SHA does not match the reviewed execution SHA" in str(excinfo.value)
    assert _AdapterTripwire.constructions == 0
    assert _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules


def test_HIGH01_sha_identity_fails_before_other_field_errors(monkeypatch):
    """§18: a wrong SHA is not masked by an unrelated protocol error."""

    monkeypatch.setattr(H, "current_expected_smoke_main_sha", lambda: "a" * 40)
    authorization = _production_authorization(approved_main_sha="b" * 40,
                                              issue_number=999,
                                              protocol_hash="tampered",
                                              human_smoke_approval=False)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_smoke_execution_authorization(authorization, test_only=False)
    assert "approved main SHA does not match the reviewed execution SHA" in str(excinfo.value)


@pytest.mark.parametrize("bad_trusted", ["", "a" * 39, "a" * 41, "A" * 40,
                                         "g" * 40, 12345, b"a" * 40])
def test_HIGH01_malformed_trusted_sha_fails_closed(monkeypatch, bad_trusted):
    """§6: the trusted side is format-checked too; a bad one never authorizes."""

    monkeypatch.setattr(H, "current_expected_smoke_main_sha", lambda: bad_trusted)
    authorization = _production_authorization(approved_main_sha=str(bad_trusted)
                                              if isinstance(bad_trusted, str) else "a" * 40)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_smoke_execution_authorization(authorization, test_only=False)
    assert "trusted reviewed main SHA" in str(excinfo.value)


@pytest.mark.parametrize("bad_record", ["", "a" * 39, "A" * 40, "z" * 40, None, 1])
def test_HIGH01_malformed_record_sha_fails_closed(monkeypatch, bad_record):
    monkeypatch.setattr(H, "current_expected_smoke_main_sha", lambda: "a" * 40)
    authorization = _production_authorization(approved_main_sha=bad_record)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_smoke_execution_authorization(authorization, test_only=False)
    assert "approved_main_sha is not a full lowercase commit SHA" in str(excinfo.value)


# --- §15: test-only positive; §16: no production positive ------------------


def test_HIGH01_test_only_trusted_sha_positive_validation():
    """The test path validates against its own trusted SHA and passes."""

    assert H._TEST_EXPECTED_MAIN_SHA == "a" * 40
    authorization = H._make_test_smoke_authorization()
    assert authorization.approved_main_sha == H._TEST_EXPECTED_MAIN_SHA
    H.validate_smoke_execution_authorization(authorization, test_only=True)

    # ... and the same record is still refused on the production path
    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(authorization, test_only=False)


@pytest.mark.parametrize("wrong", ["b" * 40, "0" * 40, "f" * 40])
def test_HIGH01_test_path_also_requires_sha_identity(wrong):
    authorization = H._make_test_smoke_authorization(approved_main_sha=wrong)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_smoke_execution_authorization(authorization, test_only=True)
    assert "approved main SHA does not match the reviewed execution SHA" in str(excinfo.value)


def test_S2c_only_the_approved_baseline_sha_validates():
    """The bound baseline is accepted; every other well-formed SHA is refused."""

    # positive: validation alone, never orchestration -- no real fit is attempted
    H.validate_smoke_execution_authorization(
        _production_authorization(approved_main_sha=H.APPROVED_SCIENTIFIC_MAIN_SHA),
        test_only=False)

    for sha in ("0" * 40, "1" * 40, "a" * 40,
                "6e3641bdb8470415601e60f21f54ede20af0926e",
                "e72eaf864b617a6e84837b57e0b80ae1eee52320"):
        with pytest.raises(HarnessStop) as excinfo:
            H.validate_smoke_execution_authorization(
                _production_authorization(approved_main_sha=sha), test_only=False)
        assert "approved main SHA does not match" in str(excinfo.value)


def test_S2c_machine_gate_never_grants_the_human_gates():
    """§46: the machine-checkable gate record never asserts the human gates.

    It stays a different object from the committed execution authorization.
    """

    authorization = H.current_smoke_authorization()
    assert authorization.independent_review_pass is False
    assert authorization.human_smoke_approval is False
    assert authorization.authorized() is False


def test_HIGH01_fake_orchestration_still_completes_end_to_end():
    """§25: the identity gate does not break the test-only orchestration."""

    canary_recorder = _FakeFitRecorder()
    canary = H._run_real_canary_test_only(H._make_test_smoke_authorization(),
                                          adapter=_test_adapter(canary_recorder))
    smoke_recorder = _FakeFitRecorder()
    smoke = H._run_real_smoke_test_only(H._make_test_smoke_authorization(),
                                        adapter=_test_adapter(smoke_recorder))
    assert canary_recorder.calls == 2 and smoke_recorder.calls == 6
    assert canary.approved_main_sha == smoke.approved_main_sha == H._TEST_EXPECTED_MAIN_SHA
    assert canary.real_canary_fits_executed == 0 and smoke.real_smoke_fits_executed == 0
    assert len(smoke.rows) == 6
    assert "em_runner" not in sys.modules


def test_HIGH01_authorization_report_exposes_the_trusted_sha_state(capsys):
    """§19: report presence only -- never a placeholder or dummy SHA."""

    assert H.main(["--smoke-authorization"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trusted_main_sha_present"] is True
    assert payload["execution_authorization_present"] is True
    assert payload["authorized"] is False
    rendered = json.dumps(payload)
    for leak in ("0" * 40, "1" * 40, "a" * 40, "approved_main_sha"):
        assert leak not in rendered, leak
    assert H.APPROVED_SCIENTIFIC_MAIN_SHA not in rendered


# ===========================================================================
# Phase 8b S2c-A — baseline binding, execution artifacts, audit (Issue #55)
# ===========================================================================
#
# The production orchestration is driven end-to-end with a Phase 7e test-only
# fit adapter writing into tmp_path.  Real EM executions remain 0 and no
# artifact is ever written under expfam/results.


import hashlib as _hashlib  # noqa: E402
import shutil as _shutil  # noqa: E402


APPROVED_BASELINE = "68c78e1191889609dead05ea5a9fb11525ce92e2"


def _stamp_canary_audit_digests(directory):
    """Re-bind a fixture's verdict to the fixture's own canary artifacts.

    Test-only, and for the same reason the baseline is re-stamped: the auditor
    audits a copy stamped with the real frozen baseline, so the bytes it hashed
    are not the bytes of the test-only fixture.  Attacks mutate the artifacts
    AFTER this call, so the stale-PASS binding remains under test.
    """

    path = directory / A.CANARY_AUDIT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, key in A.CANARY_AUDIT_CONTENT_KEYS:
        payload[key] = _hashlib.sha256((directory / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload


def _write_test_canary_audit(directory):
    """Publish canary_audit.json with the INDEPENDENT auditor, test-only lineage.

    The auditor hard-codes the real frozen baseline (that is the point of it
    being independent), so the fixture audits a copy stamped with that baseline
    and then re-stamps the verdict with the test-only trusted SHA.  The verdict
    itself is produced by ``audit_k_true_robustness_sweep``, never by the
    runner.
    """

    staging = directory.parent / (directory.name + "_auditstage")
    _shutil.copytree(directory, staging)
    for name in ("authorization.json", "canary.json"):
        _patch_json(staging / name, approved_scientific_main_sha=APPROVED_BASELINE)
    auditor = A.audit_canary_run_dir(staging, expect_execution_mode="test_only")
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    A.write_canary_audit_report(staging, auditor)
    payload = json.loads((staging / A.CANARY_AUDIT_FILENAME).read_text(encoding="utf-8"))
    payload["approved_scientific_main_sha"] = H._TEST_EXPECTED_MAIN_SHA
    (directory / A.CANARY_AUDIT_FILENAME).write_text(
        json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    _shutil.rmtree(staging)
    return _stamp_canary_audit_digests(directory)


def _fake_run(out_dir, *, authorization=None, run_code_sha="0" * 40, audit_canary=True):
    """Drive the production canary+smoke wiring with fake fits into tmp_path."""

    authorization = authorization or _test_authorization()
    canary_recorder = _FakeFitRecorder()
    canary = H._execute_real_canary_test_only(
        authorization, out_dir, adapter=_test_adapter(canary_recorder),
        run_code_sha=run_code_sha)
    if audit_canary:
        _write_test_canary_audit(out_dir)
    smoke_recorder = _FakeFitRecorder()
    smoke = H._execute_real_smoke_test_only(
        authorization, out_dir, adapter=_test_adapter(smoke_recorder),
        run_code_sha=run_code_sha)
    return canary, smoke, canary_recorder, smoke_recorder


def _promote_to_real_fixture(source, destination):
    """Artifact-level fixture: the same files stamped as a real execution.

    The audit is an artifact-only auditor, so it is exercised against a
    synthesized artifact set exactly like the config/selection audits are.
    """

    _shutil.copytree(source, destination)
    patches = {
        "authorization.json": {"approved_scientific_main_sha": APPROVED_BASELINE},
        "canary.json": {"approved_scientific_main_sha": APPROVED_BASELINE,
                        "execution_mode": "real", "real_canary_fits_executed": 2},
        "canary_audit.json": {"approved_scientific_main_sha": APPROVED_BASELINE,
                              "canary_execution_mode": "real",
                              "actual_canary_fits": 2},
        "runinfo.json": {"approved_scientific_main_sha": APPROVED_BASELINE,
                         "actual_canary_fits": 2, "actual_smoke_fits": 6,
                         "working_tree_clean": True,
                         "approved_baseline_is_ancestor": True},
        "smoke_summary.json": {"approved_scientific_main_sha": APPROVED_BASELINE,
                               "actual_smoke_fits": 6},
    }
    for name, patch in patches.items():
        path = destination / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(patch)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    # the canary artifacts were just re-stamped; re-bind the verdict to them
    _stamp_canary_audit_digests(destination)

    path = destination / "smoke_fit_results.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    canary_i = header.index("real_canary_fits_executed")
    smoke_i = header.index("real_smoke_fits_executed")
    baseline_i = header.index("approved_scientific_main_sha")
    fixed = [lines[0]]
    for line in lines[1:]:
        cells = line.split(",")
        cells[canary_i], cells[smoke_i] = "2", "6"
        cells[baseline_i] = APPROVED_BASELINE
        fixed.append(",".join(cells))
    path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    return destination


def _real_fixture(tmp_path):
    fake_dir = tmp_path / "fake"
    _fake_run(fake_dir)
    return _promote_to_real_fixture(fake_dir, tmp_path / "real")


def _patch_json(path, **updates):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return payload


def _csv_rows(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].split(","), [line.split(",") for line in lines[1:]]


def _write_csv_rows(path, header, rows):
    path.write_text("\n".join([",".join(header), *(",".join(r) for r in rows)]) + "\n",
                    encoding="utf-8")


# --- approved scientific baseline ------------------------------------------


def test_S2c_baseline_is_the_pr54_merge_commit():
    assert H.APPROVED_SCIENTIFIC_MAIN_SHA == APPROVED_BASELINE
    assert A.APPROVED_SCIENTIFIC_MAIN_SHA == APPROVED_BASELINE
    assert H.current_expected_smoke_main_sha() == APPROVED_BASELINE


def test_S2c_run_code_sha_is_provenance_not_approval():
    """§7: git may supply provenance, never approval."""

    trusted = _executable_body(H.current_expected_smoke_main_sha)
    for forbidden in ("subprocess", "rev-parse", "_git_output", "current_run_code_sha"):
        assert forbidden not in trusted, forbidden

    provenance = _executable_body(H.current_run_code_sha)
    assert "_git_output" in provenance
    assert "APPROVED_SCIENTIFIC_MAIN_SHA" not in provenance

    # the two are separate fields everywhere they are recorded
    assert "run_code_sha" in H.SMOKE_ARTIFACT_COLUMNS
    assert "approved_scientific_main_sha" in H.SMOKE_ARTIFACT_COLUMNS


def test_S2c_ancestry_check_is_a_guard_not_an_approval():
    """§27: descending from the baseline does not make a commit approved."""

    assert H.approved_baseline_is_ancestor() is True
    source = _inspect.getsource(H.approved_baseline_is_ancestor)
    assert "never an approval" in source
    # and it is not consulted by the authorization validator
    assert "approved_baseline_is_ancestor" not in _inspect.getsource(
        H._validate_smoke_execution_authorization)


# --- issue lineage ---------------------------------------------------------


def test_S2c_protocol_and_execution_issues_are_separate():
    assert H.SMOKE_PROTOCOL_ISSUE_NUMBER == 53
    assert H.SMOKE_EXECUTION_ISSUE_NUMBER == 55
    assert A.SMOKE_PROTOCOL_ISSUE_NUMBER == 53
    assert A.SMOKE_EXECUTION_ISSUE_NUMBER == 55
    # the protocol hash carries the PROTOCOL issue only
    assert H.smoke_protocol_config()["issue"] == 53
    assert not hasattr(H, "SMOKE_ISSUE_NUMBER"), "the ambiguous name must be gone"
    # the authorization carries the EXECUTION issue only
    assert H._make_test_smoke_authorization().issue_number == 55
    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(
            H._make_test_smoke_authorization(issue_number=53), test_only=True)


def test_S2c_protocol_hash_is_unchanged_by_the_execution_lineage():
    """§45: execution metadata must not perturb the scientific protocol hash."""

    assert H.smoke_protocol_hash() == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    config = H.smoke_protocol_config()
    assert 55 not in config.values()
    assert "execution_issue" not in config
    assert "approved_scientific_main_sha" not in config
    assert "run_code_sha" not in config


# --- current authorization state -------------------------------------------


def test_S2c_execution_authorization_is_present_and_valid():
    authorization = H.current_smoke_execution_authorization()
    assert authorization is not None and authorization.is_test_only() is False
    H.validate_smoke_execution_authorization(authorization, test_only=False)
    gates = H.current_smoke_authorization()
    assert gates.independent_review_pass is False
    assert gates.human_smoke_approval is False
    assert gates.authorized() is False


@pytest.mark.parametrize("command", ["canary", "smoke"])
def test_S2c_cli_gate_hands_the_committed_authorization_to_the_workflow(command,
                                                                       monkeypatch):
    """Issue #55 authorized 2+6, so the CLI clears the gate; tests never run it."""

    reached = _block_production_execution(monkeypatch)
    with pytest.raises(HarnessStop):
        H.main([f"--{command}", "--allow-em"])
    assert [name for name, _auth in reached] == [command]
    authorization = reached[0][1]
    assert type(authorization) is H.SmokeExecutionAuthorization
    assert authorization.approved_main_sha == APPROVED_BASELINE
    assert authorization.is_test_only() is False


def test_S2c_full_remains_isolated(monkeypatch):
    reached = _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"])
    assert "never be reused for --full" in str(excinfo.value)
    assert reached == [], "--full never reaches a production workflow"
    executable = _executable_body(H._require_em_authorization)
    full_branch = executable.split("if command == 'full':")[1].split("_require(command in")[0]
    assert "current_smoke_execution_authorization" not in full_branch


def test_S2c_real_adapter_is_unreachable_through_the_cli(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    _block_production_execution(monkeypatch)
    _block_full_production_execution(monkeypatch)
    for command in (["--canary", "--allow-em"], ["--smoke", "--allow-em"],
                    ["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"]):
        with pytest.raises(HarnessStop):
            H.main(command)
    assert _AdapterTripwire.constructions == 0
    assert _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules


def test_S2c_production_cli_helpers_never_touch_the_test_path():
    forbidden = ("_execute_real_canary_test_only", "_execute_real_smoke_test_only",
                 "_make_test_smoke_authorization", "_make_test_fit_adapter",
                 "_TEST_EXPECTED_MAIN_SHA")
    for function in (H.run_real_canary_cli, H.run_real_smoke_cli, H.main,
                     H._require_em_authorization, H._build_parser):
        source = _inspect.getsource(function)
        for name in forbidden:
            assert name not in source, (function.__name__, name)
    assert "test_only=False" in _inspect.getsource(H._run_production_execution)
    for name in forbidden:
        assert name not in _inspect.getsource(H._run_production_execution), name


# --- artifact contract -----------------------------------------------------


def test_S2c_artifact_contract_is_frozen():
    assert H.SMOKE_ARTIFACT_DIRNAME == "k_true_robustness_smoke_20260901"
    assert H.SMOKE_ARTIFACT_DIR == (
        ROOT / "expfam" / "results" / "k_selection" / H.SMOKE_ARTIFACT_DIRNAME)
    assert set(H.SMOKE_ARTIFACT_FILES) == {
        "authorization.json", "canary.json", "canary_audit.json", "runinfo.json",
        "smoke_fit_results.csv", "smoke_summary.json", "audit_report.json"}
    # the independent canary verdict is a distinct, reserved filename
    assert H.CANARY_AUDIT_FILENAME == A.CANARY_AUDIT_FILENAME == "canary_audit.json"
    assert set(A.CANARY_AUDIT_INPUT_FILES) == {"authorization.json", "canary.json"}
    # audit_report.json is the audit OUTPUT, never a required input
    assert "audit_report.json" not in H.SMOKE_AUDIT_INPUT_FILES
    assert set(H.SMOKE_AUDIT_INPUT_FILES) == set(A.SMOKE_AUDIT_INPUT_FILES)
    assert tuple(H.SMOKE_ARTIFACT_COLUMNS) == A.SMOKE_FIT_RESULTS_COLUMNS


def test_S2c_no_new_artifact_is_created_by_this_branch():
    """The archived S2c evidence is tracked; nothing new or modified may appear."""

    _assert_no_new_production_artifacts()


def test_S2c_artifact_directory_must_be_new(tmp_path):
    directory = tmp_path / "run"
    H.require_new_smoke_artifact_dir(directory)
    assert directory.is_dir()
    with pytest.raises(HarnessStop) as excinfo:
        H.require_new_smoke_artifact_dir(directory)
    assert "already exists" in str(excinfo.value)


def test_S2c_existing_files_are_never_overwritten(tmp_path):
    path = tmp_path / "canary.json"
    H.write_json_artifact(path, {"a": 1})
    with pytest.raises(HarnessStop) as excinfo:
        H.write_json_artifact(path, {"a": 2})
    assert "refusing to overwrite" in str(excinfo.value)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_S2c_artifact_directory_cannot_be_a_phase7e_path():
    with pytest.raises(HarnessStop):
        H.require_new_smoke_artifact_dir(H.PHASE7E_DIR)
    with pytest.raises(HarnessStop):
        H.require_existing_smoke_artifact_dir(H.PHASE7E_DIR)


def test_S2c_writes_are_atomic(tmp_path, monkeypatch):
    """A failed write leaves neither a partial file nor a temp file behind."""

    path = tmp_path / "runinfo.json"
    real_replace = __import__("os").replace

    def exploding_replace(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr("os.replace", exploding_replace)
    with pytest.raises(OSError):
        H.write_json_artifact(path, {"a": 1})
    monkeypatch.setattr("os.replace", real_replace)

    assert not path.exists(), "no partial artifact may survive"
    assert list(tmp_path.iterdir()) == [], "no temporary file may be left behind"


@pytest.mark.parametrize("content", ["", "   ", "{", '{"a": 1', "[1,2]", "null"])
def test_S2c_partial_or_malformed_json_is_refused(tmp_path, content):
    path = tmp_path / "canary.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(HarnessStop):
        H.read_json_artifact(path)


def test_S2c_smoke_contract_mode_is_zero_em(capsys):
    assert H.main(["--smoke-contract"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["em_fits_executed"] == 0
    # the archived evidence directory is tracked in the repository, so the
    # contract must report the truth rather than a hard-coded False
    assert payload["artifact_directory_exists"] == H.SMOKE_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()
    assert payload["approved_scientific_main_sha"] == APPROVED_BASELINE
    assert payload["trusted_main_sha_present"] is True
    assert payload["execution_authorization_present"] is True
    assert payload["protocol_origin_issue"] == 53 and payload["execution_issue"] == 55
    assert payload["expected_canary_fits"] == 2 and payload["expected_smoke_fits"] == 6
    assert payload["expected_real_em_budget"] == 8
    assert payload["selected_k_interpretation"] == "record_only"
    assert payload["k_recovery_evaluated"] is False
    assert payload["real_canary_fits_executed"] == 0
    assert payload["real_smoke_fits_executed"] == 0


# --- fake end-to-end execution wiring --------------------------------------


def test_S2c_fake_end_to_end_writes_the_frozen_artifact_set(tmp_path):
    out = tmp_path / "run"
    canary, smoke, canary_recorder, smoke_recorder = _fake_run(out)

    assert canary_recorder.calls == 2
    assert canary_recorder.seeds == [641011, 641011]
    assert smoke_recorder.calls == 6
    assert smoke_recorder.seeds == [641021, 641022, 641031, 641032, 641041, 641042]

    assert sorted(p.name for p in out.iterdir()) == [
        "authorization.json", "canary.json", "canary_audit.json", "runinfo.json",
        "smoke_fit_results.csv", "smoke_summary.json"]
    assert not any(p.name.startswith(".tmp-") for p in out.iterdir())

    assert canary["canary_status"] == "PASS"
    assert canary["real_canary_fits_executed"] == 0
    assert smoke["real_smoke_fits_executed"] == 0
    assert smoke["selected_k_interpretation"] == "record_only"
    assert smoke["k_recovery_evaluated"] is False
    assert "em_runner" not in sys.modules


def test_S2c_smoke_artifacts_carry_the_frozen_protocol(tmp_path):
    out = tmp_path / "run"
    _fake_run(out)
    header, rows = _csv_rows(out / "smoke_fit_results.csv")
    assert tuple(header) == H.SMOKE_ARTIFACT_COLUMNS
    assert len(rows) == 6
    by_name = [dict(zip(header, row)) for row in rows]
    assert [(int(r["K"]), int(r["start"])) for r in by_name] == [
        (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)]
    assert [int(r["model_seed"]) for r in by_name] == [
        641021, 641022, 641031, 641032, 641041, 641042]
    assert {int(r["data_seed"]) for r in by_name} == {61101}
    assert {int(r["split_seed"]) for r in by_name} == {42001}

    summary = json.loads((out / "smoke_summary.json").read_text(encoding="utf-8"))
    assert summary["k_recovery_evaluated"] is False
    assert summary["selected_k_interpretation"] == "record_only"
    assert summary["candidate_k"] == [2, 3, 4] and summary["k_true"] == 1

    runinfo = json.loads((out / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["full_fits_executed"] == 0
    assert runinfo["phase7e_rerun_count"] == 0
    assert runinfo["execution_issue"] == 55
    assert runinfo["protocol_origin_issue"] == 53
    assert runinfo["invocation_mode"] and runinfo["requested_command"]
    assert runinfo["approved_scientific_main_sha"] == APPROVED_BASELINE
    assert runinfo["run_code_sha"] != runinfo["approved_scientific_main_sha"]


# --- canary-before-smoke lineage gate --------------------------------------


def test_S2c_smoke_requires_canary_evidence(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_smoke_test_only(_test_authorization(), out,
                                        adapter=_test_adapter(recorder),
                                        run_code_sha="0" * 40)
    assert "canary.json" in str(excinfo.value)
    assert recorder.calls == 0, "smoke must not fit without canary evidence"


@pytest.mark.parametrize("updates,fragment", [
    ({"status": "FAIL"}, "status is not PASS"),
    ({"execution_mode": "test_only_but_claimed"}, "execution mode"),
    ({"approved_scientific_main_sha": "b" * 40}, "approved baseline"),
    ({"protocol_hash": "tampered"}, "protocol hash"),
    ({"execution_issue_number": 53}, "execution issue"),
    ({"protocol_origin_issue_number": 55}, "protocol origin issue"),
    ({"data_seed": 51101}, "data seed"),
    ({"split_seed": 42002}, "split seed"),
    ({"model_seed": 641021}, "model seed"),
    ({"k_est": 3}, "K_est"),
    ({"actual_fit_count": 1}, "exactly two canary fits"),
    ({"expected_fit_count": 6}, "expected fit count"),
    ({"real_canary_fits_executed": 2}, "real-fit count"),
    ({"anchor_test_hash": "0" * 64}, "test anchor"),
    ({"anchor_train_hash": "0" * 64}, "train anchor"),
    ({"initialization_equal": False}, "initialization"),
    ({"final_outputs_equal": False}, "final outputs"),
    ({"internal_retry": 1}, "internal retry"),
    ({"warning_count": 1}, "warnings"),
    ({"q_failure": True}, "Q failure"),
    ({"nan_occurred": True}, "NaN"),
    ({"finite_state": False}, "nonfinite"),
    ({"canary_atol": 1e-6}, "atol"),
    ({"canary_rtol": 1e-3}, "rtol"),
    ({"boundary_version": "v0"}, "boundary version"),
])
def test_S2c_tampered_canary_evidence_blocks_smoke(tmp_path, updates, fragment):
    out = tmp_path / "run"
    canary_recorder = _FakeFitRecorder()
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(canary_recorder),
                                     run_code_sha="0" * 40)
    assert canary_recorder.calls == 2
    _patch_json(out / "canary.json", **updates)

    smoke_recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_smoke_test_only(_test_authorization(), out,
                                        adapter=_test_adapter(smoke_recorder),
                                        run_code_sha="0" * 40)
    assert fragment in str(excinfo.value), str(excinfo.value)
    assert smoke_recorder.calls == 0, "smoke must not fit on bad canary evidence"


@pytest.mark.parametrize("content", ["", "{", '{"status": "PASS"'])
def test_S2c_partial_canary_evidence_blocks_smoke(tmp_path, content):
    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    (out / "canary.json").write_text(content, encoding="utf-8")
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop):
        H._execute_real_smoke_test_only(_test_authorization(), out,
                                        adapter=_test_adapter(recorder),
                                        run_code_sha="0" * 40)
    assert recorder.calls == 0


def test_S2c_canary_evidence_missing_a_key_blocks_smoke(tmp_path):
    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    payload = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    payload.pop("initialization_equal")
    (out / "canary.json").write_text(json.dumps(payload), encoding="utf-8")
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_smoke_test_only(_test_authorization(), out,
                                        adapter=_test_adapter(recorder),
                                        run_code_sha="0" * 40)
    assert "incomplete" in str(excinfo.value)
    assert recorder.calls == 0


def test_S2c_canary_from_another_lineage_blocks_smoke(tmp_path):
    """Canary evidence written for a different run directory is not accepted."""

    other = tmp_path / "other"
    H._execute_real_canary_test_only(_test_authorization(), other,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    _patch_json(other / "canary.json", protocol_hash="a different protocol")
    target = tmp_path / "run"
    target.mkdir()
    _shutil.copy(other / "canary.json", target / "canary.json")
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop):
        H._execute_real_smoke_test_only(_test_authorization(), target,
                                        adapter=_test_adapter(recorder),
                                        run_code_sha="0" * 40)
    assert recorder.calls == 0


def test_S2c_a_test_only_canary_cannot_authorize_a_production_smoke(tmp_path):
    """The real/test execution modes are disjoint in both directions."""

    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    payload = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "test_only"
    assert payload["real_canary_fits_executed"] == 0
    production = _production_authorization(approved_main_sha=APPROVED_BASELINE)
    with pytest.raises(HarnessStop) as excinfo:
        H.require_canary_pass_evidence(out, production,
                                       current_run_code_sha="0" * 40, test_only=False)
    assert "execution mode" in str(excinfo.value) or "approved baseline" in str(excinfo.value)


# --- independent smoke audit -----------------------------------------------


def test_S2c_audit_positive_control(tmp_path):
    auditor = A.audit_smoke_run_dir(_real_fixture(tmp_path))
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs


def test_S2c_audit_does_not_import_the_harness():
    source = _inspect.getsource(A)
    assert "import run_k_true_robustness_sweep" not in source
    for function in (A.audit_smoke_run_dir, A.audit_smoke_summary, A.audit_smoke_fit_rows):
        body = _inspect.getsource(function)
        assert "H." not in body
        assert "run_k_true_robustness_sweep" not in body


def test_S2c_audit_recomputes_seeds_and_keys_independently():
    assert A.expected_smoke_data_seed() == 61101
    assert A.expected_smoke_manifest_keys() == ((2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2))
    assert [A.expected_smoke_model_seed(k, s)
            for k, s in A.expected_smoke_manifest_keys()] == [
        641021, 641022, 641031, 641032, 641041, 641042]
    assert A.CANARY_MODEL_SEED == 641011 and A.SMOKE_SPLIT_SEED == 42001


@pytest.mark.parametrize("name", list(A.SMOKE_AUDIT_INPUT_FILES))
def test_S2c_audit_requires_every_input_file(tmp_path, name):
    directory = _real_fixture(tmp_path)
    (directory / name).unlink()
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check in ("required_artifact_missing", "smoke_artifact_missing")
               for f in auditor.blockers)


def test_S2c_audit_rejects_an_unexpected_artifact(tmp_path):
    directory = _real_fixture(tmp_path)
    (directory / "extra.csv").write_text("x\n", encoding="utf-8")
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_unexpected_artifact" for f in auditor.blockers)


@pytest.mark.parametrize("name,updates,check", [
    ("authorization.json", {"approved_scientific_main_sha": "b" * 40}, "smoke_auth_baseline_sha"),
    ("authorization.json", {"execution_issue_number": 53}, "smoke_auth_execution_issue"),
    ("authorization.json", {"protocol_origin_issue_number": 55}, "smoke_auth_protocol_issue"),
    ("authorization.json", {"independent_review_pass": False}, "smoke_auth_independent_review"),
    ("authorization.json", {"human_smoke_approval": False}, "smoke_auth_human_approval"),
    ("authorization.json", {"expected_smoke_fits": 12}, "smoke_auth_smoke_count"),
    ("authorization.json", {"split_seed": 42002}, "smoke_auth_split_seed"),
    ("authorization.json", {"smoke_model_seeds": [1, 2, 3, 4, 5, 6]}, "smoke_auth_model_seeds"),
    ("canary.json", {"status": "FAIL"}, "smoke_canary_status"),
    ("canary.json", {"execution_mode": "test_only"}, "smoke_canary_execution_mode"),
    ("canary.json", {"real_canary_fits_executed": 1}, "smoke_canary_real_fit_count"),
    ("canary.json", {"internal_retry": 1}, "smoke_canary_retry"),
    ("canary.json", {"warning_count": 2}, "smoke_canary_warnings"),
    ("canary.json", {"q_failure": True}, "smoke_canary_q_failure"),
    ("canary.json", {"nan_occurred": True}, "smoke_canary_nan"),
    ("canary.json", {"anchor_test_hash": "0" * 64}, "smoke_canary_anchor_test"),
    ("canary.json", {"model_seed": 641021}, "smoke_canary_model_seed"),
    ("runinfo.json", {"actual_canary_fits": 3}, "smoke_runinfo_canary_count"),
    ("runinfo.json", {"actual_smoke_fits": 5}, "smoke_runinfo_smoke_count"),
    ("runinfo.json", {"full_fits_executed": 336}, "smoke_runinfo_full_fits"),
    ("runinfo.json", {"phase7e_rerun_count": 1}, "smoke_runinfo_phase7e_rerun"),
    ("runinfo.json", {"working_tree_clean": False}, "smoke_runinfo_working_tree"),
    ("runinfo.json", {"expected_real_em_budget": 344}, "smoke_runinfo_budget"),
    ("smoke_summary.json", {"k_recovery_evaluated": True}, "smoke_summary_k_recovery_flag"),
    ("smoke_summary.json", {"selected_k_interpretation": "evidence"},
     "smoke_summary_selected_k_interpretation"),
    ("smoke_summary.json", {"actual_smoke_fits": 5}, "smoke_summary_fit_count"),
    ("smoke_summary.json", {"candidate_k": [1, 2, 3]}, "smoke_summary_candidates"),
])
def test_S2c_audit_json_negatives(tmp_path, name, updates, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / name, **updates)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))


def test_S2c_audit_detects_a_tampered_selected_k(tmp_path):
    directory = _real_fixture(tmp_path)
    summary = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    other = [k for k in (2, 3, 4) if k != summary["selected_k"]][0]
    _patch_json(directory / "smoke_summary.json", selected_k=other)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_summary_selected_k" for f in auditor.blockers)


def test_S2c_audit_detects_a_tampered_mean(tmp_path):
    directory = _real_fixture(tmp_path)
    summary = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    summary["per_k"]["3"]["mean"] = summary["per_k"]["3"]["mean"] + 0.5
    (directory / "smoke_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_summary_arithmetic" for f in auditor.blockers)


def test_S2c_audit_detects_consistent_score_tampering(tmp_path):
    """Changing a CSV score without changing the summary is caught."""

    directory = _real_fixture(tmp_path)
    header, rows = _csv_rows(directory / "smoke_fit_results.csv")
    index = header.index("heldout_mean_log_score")
    rows[0][index] = str(float(rows[0][index]) + 1.0)
    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows)
    auditor = A.audit_smoke_run_dir(directory)
    checks = {f.check for f in auditor.blockers}
    assert "smoke_summary_arithmetic" in checks


@pytest.mark.parametrize("column,value,check", [
    ("model_seed", "999999", "smoke_fit_model_seed"),
    ("split_seed", "42002", "smoke_fit_split_seed"),
    ("data_seed", "51101", "smoke_fit_data_seed"),
    ("approved_scientific_main_sha", "b" * 40, "smoke_fit_baseline_sha"),
    ("fit_status", "retried", "smoke_fit_status"),
    ("internal_retry", "1", "smoke_fit_retry"),
    ("warning_count", "2", "smoke_fit_warnings"),
    ("q_failure", "True", "smoke_fit_q_failure"),
    ("nan_occurred", "True", "smoke_fit_nan"),
    ("finite_state", "False", "smoke_fit_finite"),
    ("heldout_mean_log_score", "nan", "smoke_fit_score"),
    ("pre_fit_test_hash", "0" * 64, "smoke_fit_mask"),
    ("post_fit_train_hash", "0" * 64, "smoke_fit_mask"),
    ("real_canary_fits_executed", "0", "smoke_fit_canary_count"),
    ("real_smoke_fits_executed", "5", "smoke_fit_smoke_count"),
    ("boundary_version", "v0", "smoke_fit_boundary_version"),
    ("K_TRUE", "3", "smoke_fit_cell"),
    ("estimand", "B", "smoke_fit_estimand"),
])
def test_S2c_audit_csv_negatives(tmp_path, column, value, check):
    directory = _real_fixture(tmp_path)
    header, rows = _csv_rows(directory / "smoke_fit_results.csv")
    rows[0][header.index(column)] = value
    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))


def test_S2c_audit_row_count_and_order_negatives(tmp_path):
    directory = _real_fixture(tmp_path)
    header, rows = _csv_rows(directory / "smoke_fit_results.csv")

    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows[:-1])
    assert any(f.check == "smoke_fit_row_count"
               for f in A.audit_smoke_run_dir(directory).blockers)

    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows + [list(rows[0])])
    assert any(f.check == "smoke_fit_row_count"
               for f in A.audit_smoke_run_dir(directory).blockers)

    _write_csv_rows(directory / "smoke_fit_results.csv", header,
                    [rows[1], rows[0], *rows[2:]])
    assert any(f.check == "smoke_fit_key_order"
               for f in A.audit_smoke_run_dir(directory).blockers)

    duplicated = [list(rows[0]), list(rows[0]), *rows[2:]]
    _write_csv_rows(directory / "smoke_fit_results.csv", header, duplicated)
    checks = {f.check for f in A.audit_smoke_run_dir(directory).blockers}
    assert "smoke_fit_duplicate_key" in checks


def test_S2c_audit_detects_cross_file_lineage_drift(tmp_path):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "smoke_summary.json", protocol_hash="drifted")
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_protocol_hash_lineage" for f in auditor.blockers)


@pytest.mark.parametrize("winner", [2, 3, 4])
def test_S2c_audit_passes_for_any_selected_k(tmp_path, winner):
    """§44: the audit recomputes selected_k but never evaluates K recovery."""

    forced = {k: (0.9 if k == winner else 0.1) for k in H.SMOKE_K_CANDIDATES}
    out = tmp_path / "fake"
    authorization = _test_authorization()
    H._execute_real_canary_test_only(authorization, out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    _write_test_canary_audit(out)
    H._execute_real_smoke_test_only(
        authorization, out,
        adapter=_test_adapter(_FakeFitRecorder(forced_scores=forced)),
        run_code_sha="0" * 40)
    directory = _promote_to_real_fixture(out, tmp_path / "real")
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    summary = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_k"] in (2, 3, 4)
    assert summary["k_recovery_evaluated"] is False


def test_S2c_audit_never_compares_selected_k_to_k_true():
    body = _inspect.getsource(A.audit_smoke_summary)
    assert "SMOKE_K_TRUE" not in body
    assert "recovery" in body  # only as the explicit non-evaluation flag


def test_S2c_audit_cli_smoke_mode(tmp_path, capsys):
    directory = _real_fixture(tmp_path)
    assert A.main(["--run-dir", str(directory), "--mode", "smoke"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "PASS" and report["mode"] == "smoke"

    _patch_json(directory / "canary.json", status="FAIL")
    assert A.main(["--run-dir", str(directory), "--mode", "smoke"]) == 1


# --- real EM exclusion -----------------------------------------------------


def test_S2c_zero_em_paths_import_no_em_in_a_fresh_process():
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(HERE) + "');"
        "import run_k_true_robustness_sweep as H;"
        "import audit_k_true_robustness_sweep as A;"
        "H.run_validate_only(); H.run_config_gate(); H.run_leakage_self_check();"
        "H.current_smoke_authorization(); H.run_smoke_contract();"
        "H.current_expected_smoke_main_sha(); H.current_smoke_execution_authorization();"
        "print('em_runner' in sys.modules,"
        " 'model_dual_expfam_consistent' in sys.modules,"
        " H.current_smoke_execution_authorization() is not None)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False True"


# ===========================================================================
# S2c-A review findings HIGH-01..03 / MEDIUM-01..03 (Issue #55 rereview)
# ===========================================================================


# --- HIGH-01: canary/smoke run-code identity -------------------------------


def test_HIGH01_S2c_same_run_sha_positive(tmp_path):
    """Identical run-code SHA: the six smoke fits proceed."""

    out = tmp_path / "run"
    authorization = _test_authorization()
    canary_recorder = _FakeFitRecorder()
    H._execute_real_canary_test_only(authorization, out,
                                     adapter=_test_adapter(canary_recorder),
                                     run_code_sha="0" * 40)
    assert canary_recorder.calls == 2
    _write_test_canary_audit(out)

    smoke_recorder = _FakeFitRecorder()
    H._execute_real_smoke_test_only(authorization, out,
                                    adapter=_test_adapter(smoke_recorder),
                                    run_code_sha="0" * 40)
    assert smoke_recorder.calls == 6


def test_HIGH01_S2c_wrong_valid_run_sha_blocks_smoke(tmp_path, monkeypatch):
    """Both SHAs are format-valid; only identity separates them."""

    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    seen_targets = []
    real_target = H.make_score_only_target
    monkeypatch.setattr(H, "make_score_only_target",
                        lambda Y, mask: (seen_targets.append(1), real_target(Y, mask))[1])

    out = tmp_path / "run"
    authorization = _test_authorization()
    H._execute_real_canary_test_only(authorization, out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    evidence = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    assert evidence["run_code_sha"] == "0" * 40

    smoke_recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_smoke_test_only(authorization, out,
                                        adapter=_test_adapter(smoke_recorder),
                                        run_code_sha="1" * 40)

    assert "run code SHA does not match the current smoke execution" in str(excinfo.value)
    assert smoke_recorder.calls == 0
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert seen_targets == [], "no ScoreOnlyTarget may be built"
    assert not (out / "smoke_fit_results.csv").exists()


@pytest.mark.parametrize("bad", ["", "0" * 39, "0" * 41, "A" * 40, "z" * 40, None, 7])
def test_HIGH01_S2c_malformed_run_sha_blocks_smoke(tmp_path, bad):
    out = tmp_path / "run"
    authorization = _test_authorization()
    H._execute_real_canary_test_only(authorization, out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    _patch_json(out / "canary.json", run_code_sha=bad)
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop):
        H._execute_real_smoke_test_only(authorization, out,
                                        adapter=_test_adapter(recorder),
                                        run_code_sha="0" * 40)
    assert recorder.calls == 0


def test_HIGH01_S2c_run_sha_binding_is_a_required_keyword():
    parameters = _inspect.signature(H.require_canary_pass_evidence).parameters
    assert parameters["current_run_code_sha"].kind is _inspect.Parameter.KEYWORD_ONLY
    assert parameters["current_run_code_sha"].default is _inspect.Parameter.empty
    # ast.unparse normalises quoting, so compare against the normalised form
    body = _executable_body(H.require_canary_pass_evidence)
    assert "payload['run_code_sha'] == current_run_code_sha" in body
    assert "_require_full_commit_sha(current_run_code_sha" in body


# --- HIGH-02: frozen protocol hash + cross-file run SHA --------------------


def test_HIGH02_S2c_audit_freezes_the_protocol_hash_independently():
    assert A.EXPECTED_SMOKE_PROTOCOL_HASH == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert A.EXPECTED_SMOKE_PROTOCOL_HASH == H.smoke_protocol_hash()
    source = _inspect.getsource(A)
    assert "import run_k_true_robustness_sweep" not in source
    assert "smoke_protocol_hash()" not in source


def test_HIGH02_S2c_self_consistent_fake_protocol_hash_is_rejected(tmp_path):
    """All five artifacts agree on a fabricated hash; the audit still blocks."""

    directory = _real_fixture(tmp_path)
    fake = "f" * 64
    for name in ("authorization.json", "canary.json", "runinfo.json", "smoke_summary.json"):
        _patch_json(directory / name, protocol_hash=fake)
    header, rows = _csv_rows(directory / "smoke_fit_results.csv")
    index = header.index("protocol_hash")
    for row in rows:
        row[index] = fake
    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows)

    auditor = A.audit_smoke_run_dir(directory)
    checks = {f.check for f in auditor.blockers}
    assert "smoke_protocol_hash_frozen" in checks, checks
    assert len(auditor.blockers) > 0


@pytest.mark.parametrize("name", ["authorization.json", "canary.json",
                                  "runinfo.json", "smoke_summary.json"])
def test_HIGH02_S2c_per_file_protocol_hash_is_checked(tmp_path, name):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / name, protocol_hash="f" * 64)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_protocol_hash_frozen" for f in auditor.blockers)


def test_HIGH02_S2c_canary_run_sha_drift_is_rejected(tmp_path):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary.json", run_code_sha="1" * 40)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_run_code_sha_lineage" for f in auditor.blockers)


def test_HIGH02_S2c_single_csv_row_run_sha_drift_is_rejected(tmp_path):
    directory = _real_fixture(tmp_path)
    header, rows = _csv_rows(directory / "smoke_fit_results.csv")
    rows[3][header.index("run_code_sha")] = "1" * 40
    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_run_code_sha_lineage" for f in auditor.blockers)


@pytest.mark.parametrize("name", ["runinfo.json", "smoke_summary.json",
                                  "authorization.json"])
def test_HIGH02_S2c_malformed_run_sha_is_rejected(tmp_path, name):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / name, run_code_sha="not-a-sha")
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_run_code_sha_format" for f in auditor.blockers)


def test_HIGH02_S2c_summary_carries_the_full_lineage(tmp_path):
    directory = _real_fixture(tmp_path)
    summary = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    for key in ("run_code_sha", "approved_scientific_main_sha", "protocol_hash",
                "execution_issue_number", "protocol_origin_issue_number"):
        assert key in summary, key


# --- HIGH-03: the production output directory is frozen --------------------


@pytest.mark.parametrize("command", ["canary", "smoke"])
def test_HIGH03_S2c_production_rejects_an_out_dir(tmp_path, command, monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    target = tmp_path / "attacker"
    with pytest.raises(HarnessStop) as excinfo:
        H.main([f"--{command}", "--allow-em", "--out-dir", str(target)])
    assert "--out-dir is not accepted" in str(excinfo.value)
    assert not target.exists()
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0


def test_HIGH03_S2c_production_cli_uses_only_the_frozen_directory():
    for function in (H.run_real_canary_cli, H.run_real_smoke_cli,
                     H.run_real_canary, H.run_real_smoke):
        assert "args.out_dir" not in _inspect.getsource(function)
    # the single production workflow hard-codes the frozen directory
    workflow = _executable_body(H._run_production_execution)
    assert "SMOKE_ARTIFACT_DIR" in workflow
    assert "out_dir" not in workflow.replace("SMOKE_ARTIFACT_DIR", "")
    gate = _executable_body(H._require_em_authorization)
    assert "out_dir" in gate and "SMOKE_ARTIFACT_DIR" in gate


def test_HIGH03_S2c_only_the_test_path_can_redirect_the_directory():
    for function in (H.run_real_canary_cli, H.run_real_smoke_cli, H.main):
        source = _inspect.getsource(function)
        for name in ("_execute_real_canary_test_only", "_execute_real_smoke_test_only"):
            assert name not in source, (function.__name__, name)
    for function in (H._execute_real_canary_test_only, H._execute_real_smoke_test_only):
        assert "out_dir" in _inspect.signature(function).parameters


def test_HIGH03_S2c_out_dir_still_works_for_diagnostics(tmp_path):
    result = H.run_record_diagnostics(tmp_path / "diag")
    assert result["em_fits_executed"] == 0
    assert (tmp_path / "diag").is_dir()


# --- MEDIUM-01: durable audit report ---------------------------------------


def test_MEDIUM01_S2c_audit_report_is_published(tmp_path):
    directory = _real_fixture(tmp_path)
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers
    path = A.write_smoke_audit_report(directory, auditor)

    assert path.name == "audit_report.json"
    assert len(list(directory.glob("audit_report.json"))) == 1
    report = json.loads(path.read_text(encoding="utf-8"))
    for key in ("audit_version", "approved_scientific_main_sha", "run_code_sha",
                "protocol_hash", "protocol_origin_issue", "execution_issue", "status",
                "blocker_count", "high_count", "medium_count", "findings",
                "expected_canary_fits", "actual_canary_fits", "expected_smoke_fits",
                "actual_smoke_fits", "selected_k", "selected_k_interpretation",
                "k_recovery_evaluated", "audited_files"):
        assert key in report, key
    assert report["status"] == "PASS"
    assert report["blocker_count"] == 0 and report["high_count"] == 0
    assert report["approved_scientific_main_sha"] == APPROVED_BASELINE
    assert report["protocol_hash"] == A.EXPECTED_SMOKE_PROTOCOL_HASH
    assert report["expected_canary_fits"] == 2 and report["actual_canary_fits"] == 2
    assert report["expected_smoke_fits"] == 6 and report["actual_smoke_fits"] == 6
    assert report["k_recovery_evaluated"] is False
    assert report["selected_k_interpretation"] == "record_only"
    assert sorted(report["audited_files"]) == sorted(A.SMOKE_AUDIT_INPUT_FILES)


def test_MEDIUM01_S2c_audit_report_records_failure(tmp_path):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary.json", status="FAIL")
    auditor = A.audit_smoke_run_dir(directory)
    A.write_smoke_audit_report(directory, auditor)
    report = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["blocker_count"] > 0
    assert any(f["check"] == "smoke_canary_status" for f in report["findings"])


def test_MEDIUM01_S2c_audit_report_is_never_overwritten(tmp_path):
    directory = _real_fixture(tmp_path)
    A.write_smoke_audit_report(directory, A.audit_smoke_run_dir(directory))
    first = (directory / "audit_report.json").read_text(encoding="utf-8")

    _patch_json(directory / "canary.json", status="FAIL")
    with pytest.raises(FileExistsError):
        A.write_smoke_audit_report(directory, A.audit_smoke_run_dir(directory))
    assert (directory / "audit_report.json").read_text(encoding="utf-8") == first


def test_MEDIUM01_S2c_audit_report_write_is_atomic(tmp_path, monkeypatch):
    directory = _real_fixture(tmp_path)
    auditor = A.audit_smoke_run_dir(directory)
    real_replace = __import__("os").replace
    monkeypatch.setattr("os.replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        A.write_smoke_audit_report(directory, auditor)
    monkeypatch.setattr("os.replace", real_replace)
    assert not (directory / "audit_report.json").exists()
    assert not any(p.name.startswith(".tmp-") for p in directory.iterdir())


def test_MEDIUM01_S2c_audit_report_survives_a_malformed_artifact_set(tmp_path):
    directory = _real_fixture(tmp_path)
    (directory / "runinfo.json").write_text("{ broken", encoding="utf-8")
    auditor = A.audit_smoke_run_dir(directory)
    A.write_smoke_audit_report(directory, auditor)
    report = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["run_code_sha"] is None


def test_MEDIUM01_S2c_audit_cli_publishes_the_report(tmp_path, capsys):
    directory = _real_fixture(tmp_path)
    assert A.main(["--run-dir", str(directory), "--mode", "smoke", "--write-report"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "PASS"
    assert (directory / "audit_report.json").is_file()
    assert payload["audit_report"].endswith("audit_report.json")


def test_MEDIUM01_S2c_audit_report_is_not_an_audit_input(tmp_path):
    directory = _real_fixture(tmp_path)
    A.write_smoke_audit_report(directory, A.audit_smoke_run_dir(directory))
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert "audit_report.json" not in A.SMOKE_AUDIT_INPUT_FILES


# --- MEDIUM-02: preflight before any side effect ---------------------------


def test_MEDIUM02_S2c_preflight_failure_leaves_no_residue(tmp_path, monkeypatch):
    """A failing zero-EM gate stops before dir, evidence and adapter."""

    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    monkeypatch.setattr(H, "run_leakage_self_check",
                        lambda: {"all_passed": False, "cases": {},
                                 "em_fits_executed": 0, "real_em_fits_executed": 0})
    out = tmp_path / "run"
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_canary_test_only(_test_authorization(), out,
                                         adapter=_test_adapter(recorder),
                                         run_code_sha="0" * 40)
    assert "leakage self-check" in str(excinfo.value)
    assert not out.exists(), "no artifact directory may be reserved"
    assert recorder.calls == 0
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0


def test_MEDIUM02_S2c_anchor_failure_leaves_no_residue(tmp_path, monkeypatch):
    other = H.build_split_record(H.NEW_K_TRUE[0], 2)
    monkeypatch.setattr(H, "build_split_record", lambda k_true, replicate: other)
    out = tmp_path / "run"
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_canary_test_only(_test_authorization(), out,
                                         adapter=_test_adapter(recorder),
                                         run_code_sha="0" * 40)
    assert "Phase 7e anchor" in str(excinfo.value)
    assert not out.exists()
    assert recorder.calls == 0


def test_MEDIUM02_S2c_existing_directory_stops_before_the_adapter(tmp_path, monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    out = tmp_path / "run"
    out.mkdir()
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_canary_test_only(_test_authorization(), out,
                                         adapter=_test_adapter(recorder),
                                         run_code_sha="0" * 40)
    assert "already exists" in str(excinfo.value)
    assert recorder.calls == 0
    assert _AdapterTripwire.constructions == 0
    assert list(out.iterdir()) == []


def test_MEDIUM02_S2c_preflight_precedes_every_side_effect():
    """Static order: prepare -> reserve dir -> evidence -> adapter -> fit."""

    body = _executable_body(H._execute_real_canary)
    order = [body.index(token) for token in (
        "prepare_smoke_cell(", "require_new_smoke_artifact_dir(",
        "write_json_artifact(", "_resolve_fit_adapter(", "_run_real_canary(")]
    assert order == sorted(order), body

    smoke_body = _executable_body(H._execute_real_smoke)
    smoke_order = [smoke_body.index(token) for token in (
        "prepare_smoke_cell(", "require_canary_pass_evidence(",
        "_resolve_fit_adapter(", "_run_real_smoke(")]
    assert smoke_order == sorted(smoke_order), smoke_body
    # smoke never reserves a new directory
    assert "require_new_smoke_artifact_dir" not in smoke_body


def test_MEDIUM02_S2c_production_adapter_is_built_only_after_preflight():
    body = _executable_body(H._resolve_fit_adapter)
    assert "AuthorizedEMFitAdapter()" in body
    for function in (H._execute_real_canary, H._execute_real_smoke):
        source = _executable_body(function)
        assert "AuthorizedEMFitAdapter()" not in source
    with pytest.raises(HarnessStop):
        H._resolve_fit_adapter(object(), True)
    with pytest.raises(HarnessStop):
        H._resolve_fit_adapter("a pre-built adapter", False)


# --- MEDIUM-03: strict audit parsers ---------------------------------------


@pytest.mark.parametrize("column,value", [
    ("K", "not-an-int"),
    ("K", ""),
    ("K", "nan"),
    ("K", "2.0"),
    ("start", "1.5"),
    ("data_seed", ""),
    ("split_seed", "abc"),
    ("model_seed", "inf"),
    ("model_seed", "-inf"),
    ("internal_retry", "?"),
    ("warning_count", "nan"),
    ("K_TRUE", "one"),
    ("replicate", " "),
    ("real_canary_fits_executed", "two"),
    ("real_smoke_fits_executed", "6.0"),
    ("heldout_mean_log_score", "nan"),
    ("heldout_mean_log_score", "inf"),
    ("heldout_mean_log_score", "-inf"),
    ("heldout_mean_log_score", ""),
    ("heldout_mean_log_score", "not-a-score"),
    ("q_failure", "0"),
    ("nan_occurred", "yes"),
    ("finite_state", "1"),
])
def test_MEDIUM03_S2c_malformed_csv_field_is_a_blocker_not_a_crash(tmp_path, column, value):
    directory = _real_fixture(tmp_path)
    header, rows = _csv_rows(directory / "smoke_fit_results.csv")
    rows[0][header.index(column)] = value
    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows)

    auditor = A.audit_smoke_run_dir(directory)          # must not raise
    assert len(auditor.blockers) > 0, (column, value)
    # and a structured report can still be published
    A.write_smoke_audit_report(directory, auditor)
    report = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"


def test_MEDIUM03_S2c_strict_parsers_never_raise():
    auditor = A.Auditor()
    for row, column in (({}, "K"), ({"K": ""}, "K"), ({"K": "x"}, "K"),
                        ({"K": "2.0"}, "K"), ({"K": "nan"}, "K"), ({"K": None}, "K")):
        assert A._parse_int_field(row, column, auditor, "c", "l") is None
    assert A._parse_int_field({"K": "2"}, "K", auditor, "c", "l") == 2
    assert A._parse_int_field({"K": "-3"}, "K", auditor, "c", "l") == -3

    for row in ({}, {"s": ""}, {"s": "x"}, {"s": "nan"}, {"s": "inf"}, {"s": "-inf"}):
        assert A._parse_finite_float_field(row, "s", auditor, "c", "l") is None
    assert A._parse_finite_float_field({"s": "-0.5"}, "s", auditor, "c", "l") == -0.5

    assert A._parse_bool_field({"b": "True"}, "b", auditor, "c", "l") is True
    assert A._parse_bool_field({"b": "False"}, "b", auditor, "c", "l") is False
    for row in ({}, {"b": ""}, {"b": "1"}, {"b": "yes"}, {"b": "TRUE"}):
        assert A._parse_bool_field(row, "b", auditor, "c", "l") is None
    assert auditor.blockers, "every rejection must be recorded"


def test_MEDIUM03_S2c_false_token_is_not_treated_as_truthy():
    auditor = A.Auditor()
    assert A._parse_bool_field({"q_failure": "False"}, "q_failure", auditor, "c", "l") is False
    assert bool("False") is True, "the naive truth test would be wrong"
    assert not auditor.blockers


def test_MEDIUM03_S2c_audit_cli_exit_codes(tmp_path, capsys):
    directory = _real_fixture(tmp_path)
    assert A.main(["--run-dir", str(directory), "--mode", "smoke"]) == 0
    capsys.readouterr()
    header, rows = _csv_rows(directory / "smoke_fit_results.csv")
    rows[0][header.index("K")] = "not-an-int"
    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows)
    assert A.main(["--run-dir", str(directory), "--mode", "smoke"]) == 1


# --- full positive end-to-end with the audit report ------------------------


def test_S2c_full_e2e_with_audit_report(tmp_path):
    out = tmp_path / "fake"
    authorization = _test_authorization()
    canary_recorder = _FakeFitRecorder()
    H._execute_real_canary_test_only(authorization, out,
                                     adapter=_test_adapter(canary_recorder),
                                     run_code_sha="0" * 40)
    _write_test_canary_audit(out)
    smoke_recorder = _FakeFitRecorder()
    H._execute_real_smoke_test_only(authorization, out,
                                    adapter=_test_adapter(smoke_recorder),
                                    run_code_sha="0" * 40)
    assert canary_recorder.calls == 2 and smoke_recorder.calls == 6

    directory = _promote_to_real_fixture(out, tmp_path / "real")
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    A.write_smoke_audit_report(directory, auditor)

    report = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert sorted(p.name for p in directory.iterdir()) == [
        "audit_report.json", "authorization.json", "canary.json", "canary_audit.json",
        "runinfo.json", "smoke_fit_results.csv", "smoke_summary.json"]
    assert "em_runner" not in sys.modules


def test_S2c_frozen_science_after_the_fixes():
    assert H.smoke_protocol_hash() == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert H.APPROVED_SCIENTIFIC_MAIN_SHA == APPROVED_BASELINE
    assert (H.SMOKE_PROTOCOL_ISSUE_NUMBER, H.SMOKE_EXECUTION_ISSUE_NUMBER) == (53, 55)
    assert (H.SMOKE_ESTIMAND, H.SMOKE_ROLE) == ("A", "primary")
    assert (H.SMOKE_K_TRUE, H.SMOKE_REPLICATE) == (1, 1)
    assert H.SMOKE_K_CANDIDATES == (2, 3, 4) and H.SMOKE_STARTS == (1, 2)
    assert (H.EXPECTED_CANARY_FITS, H.EXPECTED_SMOKE_FITS) == (2, 6)
    assert H.smoke_data_seed(1, 1) == 61101 and H.CANARY_MODEL_SEED == 641011
    assert [H.smoke_model_seed(1, 1, k, s) for k in (2, 3, 4) for s in (1, 2)] == [
        641021, 641022, 641031, 641032, 641041, 641042]
    assert H.SMOKE_SPLIT_SEED == 42001
    assert len(H.build_manifest("A")) == len(H.build_manifest("B")) == 168
    assert H.EXPECTED_NEW_FITS == 336
    assert H.current_smoke_execution_authorization() is not None
    assert H.current_smoke_authorization().authorized() is False


# ===========================================================================
# HIGH: public production entry points must not bypass the S2c orchestration
# ===========================================================================
#
# ``run_real_canary`` / ``run_real_smoke`` used to validate the authorization
# and then construct the real adapter themselves, skipping the preflight and
# -- for smoke -- the canary lineage entirely.  They are now thin wrappers over
# the single production workflow.


def _stage_production_probe(monkeypatch, tmp_path, *, run_code_sha="1" * 40):
    """Point the frozen production directory at tmp_path and arm a tripwire.

    ``SMOKE_ARTIFACT_DIR`` is a module constant, not a caller parameter: this
    redirect exists only so the production workflow can be exercised without
    writing under expfam/results.
    """

    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    monkeypatch.setattr(H, "SMOKE_ARTIFACT_DIR", tmp_path / "frozen_smoke")
    monkeypatch.setattr(H, "working_tree_is_clean", lambda: True)
    monkeypatch.setattr(H, "current_run_code_sha", lambda: run_code_sha)
    monkeypatch.setattr(H, "approved_baseline_is_ancestor", lambda sha=None: True)
    return H.SMOKE_ARTIFACT_DIR


def _valid_production_authorization():
    return _production_authorization(approved_main_sha=H.APPROVED_SCIENTIFIC_MAIN_SHA)


# --- A: direct run_real_canary with a failing preflight ---------------------


def test_HIGHPUB_direct_canary_stops_at_the_preflight(tmp_path, monkeypatch):
    directory = _stage_production_probe(monkeypatch, tmp_path)
    monkeypatch.setattr(H, "run_leakage_self_check",
                        lambda: {"all_passed": False, "cases": {},
                                 "em_fits_executed": 0, "real_em_fits_executed": 0})

    with pytest.raises(HarnessStop) as excinfo:
        H.run_real_canary(_valid_production_authorization())

    assert "leakage self-check" in str(excinfo.value)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert not directory.exists(), "no artifact directory may be reserved"
    assert not (directory / "authorization.json").exists()
    assert "em_runner" not in sys.modules


def test_HIGHPUB_direct_canary_stops_on_an_anchor_failure(tmp_path, monkeypatch):
    directory = _stage_production_probe(monkeypatch, tmp_path)
    other = H.build_split_record(H.NEW_K_TRUE[0], 2)
    monkeypatch.setattr(H, "build_split_record", lambda k_true, replicate: other)

    with pytest.raises(HarnessStop) as excinfo:
        H.run_real_canary(_valid_production_authorization())

    assert "Phase 7e anchor" in str(excinfo.value)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert not directory.exists()


# --- B: direct run_real_smoke with no canary evidence -----------------------


def test_HIGHPUB_direct_smoke_requires_canary_evidence(tmp_path, monkeypatch):
    directory = _stage_production_probe(monkeypatch, tmp_path)

    with pytest.raises(HarnessStop) as excinfo:
        H.run_real_smoke(_valid_production_authorization())

    message = str(excinfo.value)
    assert "smoke artifact directory does not exist" in message or "canary.json" in message
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert not (directory / "smoke_fit_results.csv").exists()
    assert "em_runner" not in sys.modules


def test_HIGHPUB_direct_smoke_with_empty_directory_requires_canary(tmp_path, monkeypatch):
    directory = _stage_production_probe(monkeypatch, tmp_path)
    directory.mkdir(parents=True)

    with pytest.raises(HarnessStop) as excinfo:
        H.run_real_smoke(_valid_production_authorization())

    assert "canary.json" in str(excinfo.value)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert list(directory.iterdir()) == []


# --- C: direct run_real_smoke with a canary from a different run ------------


def _stage_real_looking_canary(directory, *, run_code_sha):
    """Write a PASS canary that claims a real execution under ``run_code_sha``."""

    staging = directory.parent / "staging"
    H._execute_real_canary_test_only(_test_authorization(), staging,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha=run_code_sha)
    payload = json.loads((staging / "canary.json").read_text(encoding="utf-8"))
    payload.update({
        "execution_mode": "real",
        "real_canary_fits_executed": 2,
        "approved_scientific_main_sha": H.APPROVED_SCIENTIFIC_MAIN_SHA,
        "run_code_sha": run_code_sha,
    })
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "canary.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_HIGHPUB_direct_smoke_rejects_a_canary_from_another_run(tmp_path, monkeypatch):
    directory = _stage_production_probe(monkeypatch, tmp_path, run_code_sha="1" * 40)
    evidence = _stage_real_looking_canary(directory, run_code_sha="0" * 40)
    assert evidence["run_code_sha"] == "0" * 40

    with pytest.raises(HarnessStop) as excinfo:
        H.run_real_smoke(_valid_production_authorization())

    assert "run code SHA does not match the current smoke execution" in str(excinfo.value)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert not (directory / "smoke_fit_results.csv").exists()


def test_HIGHPUB_direct_smoke_rejects_a_failed_canary(tmp_path, monkeypatch):
    directory = _stage_production_probe(monkeypatch, tmp_path, run_code_sha="1" * 40)
    _stage_real_looking_canary(directory, run_code_sha="1" * 40)
    _patch_json(directory / "canary.json", status="FAIL")

    with pytest.raises(HarnessStop) as excinfo:
        H.run_real_smoke(_valid_production_authorization())

    assert "status is not PASS" in str(excinfo.value)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0


def test_HIGHPUB_direct_smoke_rejects_a_test_only_canary(tmp_path, monkeypatch):
    """A canary produced by the fake path can never authorize a real smoke."""

    directory = _stage_production_probe(monkeypatch, tmp_path, run_code_sha="1" * 40)
    staging = tmp_path / "staging"
    H._execute_real_canary_test_only(_test_authorization(), staging,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="1" * 40)
    directory.mkdir(parents=True, exist_ok=True)
    _shutil.copy(staging / "canary.json", directory / "canary.json")

    with pytest.raises(HarnessStop):
        H.run_real_smoke(_valid_production_authorization())
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0


# --- D: one adapter construction site, one production workflow --------------


def test_HIGHPUB_adapter_is_constructed_in_exactly_one_place():
    import ast as _ast

    source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    tree = _ast.parse(source)
    sites = []
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
                and node.func.id == "AuthorizedEMFitAdapter"):
            continue
        owner = None
        for candidate in _ast.walk(tree):
            if isinstance(candidate, _ast.FunctionDef) and \
                    candidate.lineno <= node.lineno <= (candidate.end_lineno or node.lineno):
                if owner is None or candidate.lineno > owner.lineno:
                    owner = candidate
        sites.append(owner.name if owner else "<module>")
    assert sites == ["_resolve_fit_adapter"], sites


def test_HIGHPUB_public_entrypoints_are_thin_wrappers():
    for function in (H.run_real_canary, H.run_real_smoke):
        body = _executable_body(function)
        assert "_run_production_execution" in body
        assert "AuthorizedEMFitAdapter" not in body
        assert "validate_smoke_execution_authorization" not in body
        assert "_run_real_canary(" not in body and "_run_real_smoke(" not in body
        # single argument: no caller-supplied lineage, directory or SHA
        assert list(_inspect.signature(function).parameters) == ["authorization"]


def test_HIGHPUB_cli_and_public_share_one_workflow():
    for function in (H.run_real_canary_cli, H.run_real_smoke_cli):
        body = _executable_body(function)
        assert "_run_production_execution" in body
        assert "AuthorizedEMFitAdapter" not in body
        assert "_execute_real_canary(" not in body and "_execute_real_smoke(" not in body

    workflow = _executable_body(H._run_production_execution)
    for token in ("validate_smoke_execution_authorization", "current_run_code_sha()",
                  "_require_execution_preconditions", "SMOKE_ARTIFACT_DIR",
                  "test_only=False", "test_adapter=None"):
        assert token in workflow, token
    assert "args" not in workflow and "out_dir=" not in workflow


def test_HIGHPUB_no_caller_controlled_bypass_parameters():
    forbidden = ("canary_pass", "canary_report", "skip_canary_check", "out_dir",
                 "run_code_sha", "current_run_code_sha", "expected_main_sha",
                 "approved_main_sha", "adapter", "test_only")
    for function in (H.run_real_canary, H.run_real_smoke,
                     H.run_real_canary_cli, H.run_real_smoke_cli):
        parameters = set(_inspect.signature(function).parameters)
        assert not (parameters & set(forbidden)), (function.__name__, parameters)
    workflow_parameters = set(_inspect.signature(H._run_production_execution).parameters)
    assert workflow_parameters == {"authorization", "command"}


def test_HIGHPUB_production_callgraph_never_reaches_the_test_path():
    forbidden = ("_execute_real_canary_test_only", "_execute_real_smoke_test_only",
                 "_run_real_canary_test_only", "_run_real_smoke_test_only",
                 "_make_test_smoke_authorization", "_make_test_fit_adapter",
                 "_TEST_EXPECTED_MAIN_SHA", "_SMOKE_TEST_AUTHORITY")
    for function in (H.run_real_canary, H.run_real_smoke, H._run_production_execution,
                     H.run_real_canary_cli, H.run_real_smoke_cli, H.main):
        source = _inspect.getsource(function)
        for name in forbidden:
            assert name not in source, (function.__name__, name)


# --- E: the valid test-only lineage still completes --------------------------


def test_HIGHPUB_valid_test_only_lineage_still_runs_two_then_six(tmp_path):
    out = tmp_path / "run"
    authorization = _test_authorization()
    canary_recorder = _FakeFitRecorder()
    H._execute_real_canary_test_only(authorization, out,
                                     adapter=_test_adapter(canary_recorder),
                                     run_code_sha="0" * 40)
    _write_test_canary_audit(out)
    smoke_recorder = _FakeFitRecorder()
    H._execute_real_smoke_test_only(authorization, out,
                                    adapter=_test_adapter(smoke_recorder),
                                    run_code_sha="0" * 40)
    assert canary_recorder.calls == 2 and smoke_recorder.calls == 6

    directory = _promote_to_real_fixture(out, tmp_path / "real")
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers and not auditor.highs
    A.write_smoke_audit_report(directory, auditor)
    report = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert "em_runner" not in sys.modules


def test_HIGHPUB_current_state_keeps_production_out_of_the_test_suite(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    _block_production_execution(monkeypatch)
    _block_full_production_execution(monkeypatch)
    assert H.current_smoke_execution_authorization() is not None
    for command in (["--canary", "--allow-em"], ["--smoke", "--allow-em"],
                    ["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"]):
        with pytest.raises(HarnessStop):
            H.main(command)
    for entrypoint in (H.run_real_canary, H.run_real_smoke):
        with pytest.raises(HarnessStop):
            entrypoint(H.current_smoke_execution_authorization())
        with pytest.raises(HarnessStop):
            entrypoint(_test_authorization())
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    _assert_no_new_production_artifacts()


# ===========================================================================
# PR #56 review: HIGH-01 canary audit gate / MEDIUM-01 ancestry condition
# ===========================================================================


# --- HIGH-01: the independent canary audit gates the smoke -----------------


def test_CANARYAUDIT_module_is_independent_of_the_runner():
    source = _inspect.getsource(A)
    assert "import run_k_true_robustness_sweep" not in source
    for function in (A.audit_canary_run_dir, A.build_canary_audit_report,
                     A.write_canary_audit_report):
        body = _inspect.getsource(function)
        assert "run_k_true_robustness_sweep" not in body
        assert "select_k_from_two_starts" not in body


def test_CANARYAUDIT_runs_on_canary_evidence_alone(tmp_path):
    """Only authorization.json + canary.json exist right after the canary."""

    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    assert sorted(p.name for p in out.iterdir()) == ["authorization.json", "canary.json"]
    for absent in ("runinfo.json", "smoke_fit_results.csv", "smoke_summary.json"):
        assert not (out / absent).exists()

    _patch_json(out / "authorization.json", approved_scientific_main_sha=APPROVED_BASELINE)
    _patch_json(out / "canary.json", approved_scientific_main_sha=APPROVED_BASELINE)
    auditor = A.audit_canary_run_dir(out, expect_execution_mode="test_only")
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    assert set(A.CANARY_AUDIT_INPUT_FILES) == {"authorization.json", "canary.json"}
    for absent in ("runinfo.json", "smoke_fit_results.csv", "smoke_summary.json"):
        assert absent not in A.CANARY_AUDIT_INPUT_FILES


def test_CANARYAUDIT_report_content_and_atomicity(tmp_path, monkeypatch):
    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    payload = _write_test_canary_audit(out)

    for key in ("audit_version", "status", "blocker_count", "high_count", "findings",
                "approved_scientific_main_sha", "run_code_sha", "protocol_hash",
                "protocol_origin_issue", "execution_issue", "expected_canary_fits",
                "actual_canary_fits", "audited_files"):
        assert key in payload, key
    assert payload["status"] == "PASS"
    assert payload["blocker_count"] == 0 and payload["high_count"] == 0
    assert payload["protocol_hash"] == A.EXPECTED_SMOKE_PROTOCOL_HASH
    assert payload["protocol_origin_issue"] == 53 and payload["execution_issue"] == 55
    assert payload["expected_canary_fits"] == 2
    assert sorted(payload["audited_files"]) == ["authorization.json", "canary.json"]
    assert (out / "canary_audit.json").is_file()
    assert not (out / "audit_report.json").exists(), "the final report name is reserved"


def test_CANARYAUDIT_report_is_never_overwritten(tmp_path):
    out = tmp_path / "run"
    _fake_run(out)
    first = (out / "canary_audit.json").read_text(encoding="utf-8")
    auditor = A.audit_canary_run_dir(out, expect_execution_mode="test_only")
    with pytest.raises(FileExistsError):
        A.write_canary_audit_report(out, auditor)
    assert (out / "canary_audit.json").read_text(encoding="utf-8") == first


def test_CANARYAUDIT_report_write_is_atomic(tmp_path, monkeypatch):
    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    auditor = A.audit_canary_run_dir(out, expect_execution_mode="test_only")
    real_replace = __import__("os").replace
    monkeypatch.setattr("os.replace", lambda src, dst: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        A.write_canary_audit_report(out, auditor)
    monkeypatch.setattr("os.replace", real_replace)
    assert not (out / "canary_audit.json").exists()
    assert not any(p.name.startswith(".tmp-") for p in out.iterdir())


def test_CANARYAUDIT_failed_canary_yields_a_FAIL_verdict(tmp_path):
    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    _patch_json(out / "canary.json", status="FAIL",
                approved_scientific_main_sha=APPROVED_BASELINE)
    _patch_json(out / "authorization.json", approved_scientific_main_sha=APPROVED_BASELINE)
    auditor = A.audit_canary_run_dir(out, expect_execution_mode="test_only")
    A.write_canary_audit_report(out, auditor)
    payload = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL" and payload["blocker_count"] > 0


def test_CANARYAUDIT_cli_mode(tmp_path, capsys):
    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    # the CLI audits a real-baseline canary; stamp the copy accordingly
    for name in ("authorization.json", "canary.json"):
        _patch_json(out / name, approved_scientific_main_sha=APPROVED_BASELINE)
    _patch_json(out / "canary.json", execution_mode="real", real_canary_fits_executed=2)

    assert A.main(["--run-dir", str(out), "--mode", "canary", "--write-report"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "PASS" and report["mode"] == "canary"
    assert report["canary_audit"].endswith("canary_audit.json")
    assert (out / "canary_audit.json").is_file()
    assert not (out / "audit_report.json").exists()


def test_CANARYAUDIT_cli_exit_code_on_failure(tmp_path, capsys):
    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    assert A.main(["--run-dir", str(out), "--mode", "canary"]) == 1


# --- HIGH-01 negatives: smoke must not start ------------------------------


def _canary_only(tmp_path, *, run_code_sha="0" * 40):
    out = tmp_path / "run"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha=run_code_sha)
    return out


def _attempt_smoke(out, monkeypatch, *, run_code_sha="0" * 40):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    recorder = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_smoke_test_only(_test_authorization(), out,
                                        adapter=_test_adapter(recorder),
                                        run_code_sha=run_code_sha)
    assert recorder.calls == 0, "no smoke fit may run"
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert not (out / "smoke_fit_results.csv").exists()
    return str(excinfo.value)


def test_CANARYAUDIT_A_missing_verdict_blocks_smoke(tmp_path, monkeypatch):
    out = _canary_only(tmp_path)
    assert not (out / "canary_audit.json").exists()
    message = _attempt_smoke(out, monkeypatch)
    assert "canary_audit.json" in message


@pytest.mark.parametrize("updates,fragment", [
    ({"status": "FAIL"}, "independent canary audit did not pass"),
    ({"blocker_count": 1}, "blocking findings"),
    ({"high_count": 2}, "blocking findings"),
    ({"run_code_sha": "1" * 40}, "run code SHA does not match"),
    ({"run_code_sha": "not-a-sha"}, "run code SHA"),
    ({"protocol_hash": "f" * 64}, "protocol hash"),
    ({"approved_scientific_main_sha": "b" * 40}, "approved baseline"),
    ({"execution_issue": 53}, "different execution issue"),
    ({"protocol_origin_issue": 55}, "protocol origin issue"),
    ({"expected_canary_fits": 6}, "expected fit count"),
    ({"actual_canary_fits": 1}, "fit count does not match"),
    ({"canary_execution_mode": "real"}, "canary audit covers"),
    ({"canary_status": "FAIL"}, "did not cover a PASS canary"),
    ({"audit_version": "v0"}, "audit version changed"),
    ({"audited_files": ["canary.json"]}, "did not read authorization.json"),
])
def test_CANARYAUDIT_tampered_verdict_blocks_smoke(tmp_path, monkeypatch, updates, fragment):
    out = _canary_only(tmp_path)
    _write_test_canary_audit(out)
    _patch_json(out / "canary_audit.json", **updates)
    message = _attempt_smoke(out, monkeypatch)
    assert fragment in message, message


@pytest.mark.parametrize("content", ["", "   ", "{", '{"status": "PASS"', "[1,2]", "null"])
def test_CANARYAUDIT_malformed_verdict_blocks_smoke(tmp_path, monkeypatch, content):
    out = _canary_only(tmp_path)
    _write_test_canary_audit(out)
    (out / "canary_audit.json").write_text(content, encoding="utf-8")
    _attempt_smoke(out, monkeypatch)


def test_CANARYAUDIT_incomplete_verdict_blocks_smoke(tmp_path, monkeypatch):
    out = _canary_only(tmp_path)
    _write_test_canary_audit(out)
    payload = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    payload.pop("canary_execution_mode")
    (out / "canary_audit.json").write_text(json.dumps(payload), encoding="utf-8")
    message = _attempt_smoke(out, monkeypatch)
    assert "incomplete" in message


def test_CANARYAUDIT_test_only_verdict_cannot_authorize_production(tmp_path):
    """A test-only canary verdict is not evidence for a real smoke."""

    out = _canary_only(tmp_path)
    payload = _write_test_canary_audit(out)
    assert payload["canary_execution_mode"] == "test_only"

    canary_payload = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    production = _production_authorization(approved_main_sha=APPROVED_BASELINE)
    with pytest.raises(HarnessStop):
        H.require_canary_audit_pass(out, production, canary_payload,
                                    current_run_code_sha="0" * 40, test_only=False)


def test_CANARYAUDIT_J_valid_verdict_allows_exactly_six(tmp_path):
    out = tmp_path / "run"
    _canary, _smoke, canary_recorder, smoke_recorder = _fake_run(out)
    assert canary_recorder.calls == 2 and smoke_recorder.calls == 6
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS"
    runinfo = json.loads((out / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["canary_audit_status"] == "PASS"
    assert "em_runner" not in sys.modules


def test_CANARYAUDIT_gate_runs_before_the_adapter():
    body = _executable_body(H._execute_real_smoke)
    order = [body.index(token) for token in (
        "prepare_smoke_cell(", "require_existing_smoke_artifact_dir(",
        "require_canary_pass_evidence(", "require_canary_audit_pass(",
        "_resolve_fit_adapter(", "_run_real_smoke(")]
    assert order == sorted(order), body


def test_CANARYAUDIT_runner_never_writes_the_verdict():
    """§9 trust boundary: only the audit module publishes canary_audit.json."""

    runner = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    assert "write_canary_audit_report" not in runner
    assert "build_canary_audit_report" not in runner
    # the runner references the filename only to READ and REQUIRE it
    body = _executable_body(H.require_canary_audit_pass)
    assert "read_json_artifact" in body
    assert "write_json_artifact" not in body
    for function in (H._execute_real_canary, H._execute_real_smoke):
        source = _executable_body(function)
        assert "CANARY_AUDIT_FILENAME" not in source or "write" not in source.split(
            "CANARY_AUDIT_FILENAME")[0][-40:]
    assert hasattr(A, "write_canary_audit_report")


def test_CANARYAUDIT_final_audit_requires_the_verdict(tmp_path):
    directory = _real_fixture(tmp_path)
    assert (directory / "canary_audit.json").is_file()
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]

    (directory / "canary_audit.json").unlink()
    auditor2 = A.audit_smoke_run_dir(directory)
    assert any(f.check in ("required_artifact_missing", "smoke_artifact_missing")
               for f in auditor2.blockers)


@pytest.mark.parametrize("updates,check", [
    ({"status": "FAIL"}, "smoke_canary_audit_status"),
    ({"blocker_count": 1}, "smoke_canary_audit_counts"),
    ({"approved_scientific_main_sha": "b" * 40}, "smoke_canary_audit_baseline"),
    ({"protocol_hash": "f" * 64}, "smoke_protocol_hash_frozen"),
    ({"execution_issue": 53}, "smoke_canary_audit_execution_issue"),
    ({"actual_canary_fits": 1}, "smoke_canary_audit_fit_count"),
    ({"canary_execution_mode": "test_only"}, "smoke_canary_audit_execution_mode"),
    ({"audit_version": "v0"}, "smoke_canary_audit_version"),
])
def test_CANARYAUDIT_final_audit_negatives(tmp_path, updates, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json", **updates)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))


# --- MEDIUM-01: the ancestry flag must be exactly True ---------------------


def test_ANCESTRY_flag_must_be_true(tmp_path):
    directory = _real_fixture(tmp_path)
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers

    _patch_json(directory / "runinfo.json", approved_baseline_is_ancestor=False)
    auditor2 = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_runinfo_lineage" for f in auditor2.blockers)


def test_ANCESTRY_descendant_commit_no_longer_slips_through(tmp_path):
    """The old condition was vacuous for every normal descendant commit."""

    directory = _real_fixture(tmp_path)
    _patch_json(directory / "runinfo.json",
                run_code_sha="f" * 40, approved_baseline_is_ancestor=False)
    header, rows = _csv_rows(directory / "smoke_fit_results.csv")
    index = header.index("run_code_sha")
    for row in rows:
        row[index] = "f" * 40
    _write_csv_rows(directory / "smoke_fit_results.csv", header, rows)
    for name in ("authorization.json", "canary.json", "canary_audit.json",
                 "smoke_summary.json"):
        _patch_json(directory / name, run_code_sha="f" * 40)

    auditor = A.audit_smoke_run_dir(directory)
    checks = {f.check for f in auditor.blockers}
    # the run-code lineage itself is consistent ...
    assert "smoke_run_code_sha_lineage" not in checks, checks
    # ... but the ancestry requirement now blocks
    assert "smoke_runinfo_lineage" in checks, checks


@pytest.mark.parametrize("value", [None, "True", 1, "yes", 0, False])
def test_ANCESTRY_non_true_values_are_rejected(tmp_path, value):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "runinfo.json", approved_baseline_is_ancestor=value)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_runinfo_lineage" for f in auditor.blockers), value


def test_ANCESTRY_condition_is_unconditional():
    body = _executable_body(A.audit_smoke_runinfo)
    assert "approved_baseline_is_ancestor" in body
    # the vacuous disjunction must be gone
    assert "!= APPROVED_SCIENTIFIC_MAIN_SHA" not in body
    assert " or payload.get('approved_baseline_is_ancestor')" not in body


# --- everything else must still hold ---------------------------------------


def test_FINAL_previous_protections_are_intact():
    assert H.smoke_protocol_hash() == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert A.EXPECTED_SMOKE_PROTOCOL_HASH == H.smoke_protocol_hash()
    assert H.APPROVED_SCIENTIFIC_MAIN_SHA == APPROVED_BASELINE
    assert H.current_smoke_execution_authorization() is not None
    assert H.current_smoke_authorization().authorized() is False
    assert (H.EXPECTED_CANARY_FITS, H.EXPECTED_SMOKE_FITS,
            H.EXPECTED_REAL_EM_BUDGET) == (2, 6, 8)
    assert H.smoke_data_seed(1, 1) == 61101 and H.SMOKE_SPLIT_SEED == 42001
    assert H.CANARY_MODEL_SEED == 641011
    assert [H.smoke_model_seed(1, 1, k, s) for k in (2, 3, 4) for s in (1, 2)] == [
        641021, 641022, 641031, 641032, 641041, 641042]
    assert len(H.build_manifest("A")) == len(H.build_manifest("B")) == 168
    assert H.EXPECTED_NEW_FITS == 336


# ===========================================================================
# PR #56 rereview MEDIUM: the canary-only auditor must check every frozen field
# ===========================================================================
#
# A canary_audit.json marked PASS has to mean "the frozen canary protocol was
# independently verified in full".  Before this fix, estimand, role, K_TRUE,
# replicate, the protocol-origin issue, both frozen tolerances and the boundary
# version could all be tampered with and the canary-only audit still passed.


def _canary_audit_fixture(tmp_path, **canary_updates):
    """A canary-only artifact set stamped for a real execution."""

    out = tmp_path / "canary_only"
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    _patch_json(out / "authorization.json",
                approved_scientific_main_sha=APPROVED_BASELINE)
    _patch_json(out / "canary.json",
                approved_scientific_main_sha=APPROVED_BASELINE,
                execution_mode="real", real_canary_fits_executed=2)
    if canary_updates:
        _patch_json(out / "canary.json", **canary_updates)
    return out


def _write_raw_canary(path, **updates):
    """Write canary.json allowing NaN/Infinity literals for malformed fixtures."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")


# --- independent frozen constants ------------------------------------------


def test_CANARYAUDIT_frozen_constants_match_phase7e_source():
    """The tolerances are restated independently, not imported."""

    from run_heldout_k_selection_pilot import CANARY_ATOL, CANARY_RTOL

    assert A.EXPECTED_CANARY_ATOL == float(CANARY_ATOL) == 1e-12
    assert A.EXPECTED_CANARY_RTOL == float(CANARY_RTOL) == 1e-10
    assert A.EXPECTED_LEAKAGE_BOUNDARY_VERSION == "phase8b-leakage-boundary-v1"
    assert A.SMOKE_ESTIMAND == "A" and A.SMOKE_ROLE == "primary"
    assert A.SMOKE_K_TRUE == 1 and A.SMOKE_REPLICATE == 1
    assert A.SMOKE_PROTOCOL_ISSUE_NUMBER == 53 and A.SMOKE_EXECUTION_ISSUE_NUMBER == 55

    source = _inspect.getsource(A)
    assert "import run_k_true_robustness_sweep" not in source
    assert "from run_heldout_k_selection_pilot import" not in source
    body = _inspect.getsource(A.audit_smoke_canary)
    assert "H." not in body and "run_k_true_robustness_sweep" not in body


def test_CANARYAUDIT_positive_control_still_passes(tmp_path):
    out = _canary_audit_fixture(tmp_path)
    auditor = A.audit_canary_run_dir(out)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    A.write_canary_audit_report(out, auditor)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS"


# --- A-H: one frozen field mutated at a time -------------------------------


@pytest.mark.parametrize("updates,check", [
    ({"estimand": "B"}, "smoke_canary_estimand"),
    ({"role": "sensitivity"}, "smoke_canary_role"),
    ({"k_true": 2}, "smoke_canary_k_true"),
    ({"replicate": 2}, "smoke_canary_replicate"),
    ({"protocol_origin_issue_number": 55}, "smoke_canary_protocol_issue"),
    ({"canary_atol": 1e-6}, "smoke_canary_atol"),
    ({"canary_rtol": 1e-3}, "smoke_canary_rtol"),
    ({"boundary_version": "wrong-version"}, "smoke_canary_boundary_version"),
])
def test_CANARYAUDIT_frozen_field_mutation_blocks(tmp_path, updates, check):
    out = _canary_audit_fixture(tmp_path, **updates)
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))

    # a FAIL verdict may be published, a PASS one never
    A.write_canary_audit_report(out, auditor)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "FAIL"
    assert verdict["blocker_count"] > 0


@pytest.mark.parametrize("updates,check", [
    ({"estimand": None}, "smoke_canary_estimand"),
    ({"role": None}, "smoke_canary_role"),
    ({"k_true": "1"}, "smoke_canary_k_true"),
    ({"k_true": True}, "smoke_canary_k_true"),
    ({"replicate": 1.5}, "smoke_canary_replicate"),
    ({"protocol_origin_issue_number": 54}, "smoke_canary_protocol_issue"),
    ({"protocol_origin_issue_number": None}, "smoke_canary_protocol_issue"),
    ({"boundary_version": None}, "smoke_canary_boundary_version"),
])
def test_CANARYAUDIT_frozen_field_type_mutation_blocks(tmp_path, updates, check):
    out = _canary_audit_fixture(tmp_path, **updates)
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))


@pytest.mark.parametrize("field", ["estimand", "role", "k_true", "replicate",
                                   "protocol_origin_issue_number", "canary_atol",
                                   "canary_rtol", "boundary_version"])
def test_CANARYAUDIT_missing_frozen_field_blocks(tmp_path, field):
    out = _canary_audit_fixture(tmp_path)
    payload = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    payload.pop(field)
    (out / "canary.json").write_text(json.dumps(payload), encoding="utf-8")
    auditor = A.audit_canary_run_dir(out)
    assert auditor.blockers, field


# --- malformed tolerances ---------------------------------------------------


@pytest.mark.parametrize("key,value", [
    ("canary_atol", None),
    ("canary_atol", "1e-6"),
    ("canary_atol", "1e-12"),
    ("canary_atol", float("nan")),
    ("canary_atol", float("inf")),
    ("canary_atol", float("-inf")),
    ("canary_atol", True),
    ("canary_atol", [1e-12]),
    ("canary_rtol", None),
    ("canary_rtol", "1e-3"),
    ("canary_rtol", "1e-10"),
    ("canary_rtol", float("nan")),
    ("canary_rtol", float("inf")),
    ("canary_rtol", float("-inf")),
    ("canary_rtol", True),
    ("canary_rtol", {"v": 1e-10}),
])
def test_CANARYAUDIT_malformed_tolerance_fails_closed(tmp_path, key, value):
    out = _canary_audit_fixture(tmp_path)
    _write_raw_canary(out / "canary.json", **{key: value})

    auditor = A.audit_canary_run_dir(out)          # must not raise
    check = "smoke_canary_atol" if key == "canary_atol" else "smoke_canary_rtol"
    assert any(f.check == check for f in auditor.blockers), \
        (key, value, sorted({f.check for f in auditor.blockers}))

    A.write_canary_audit_report(out, auditor)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "FAIL"


def test_CANARYAUDIT_exact_float_helper_never_raises():
    auditor = A.Auditor()
    for payload in ({}, {"t": None}, {"t": "1e-12"}, {"t": True},
                    {"t": float("nan")}, {"t": float("inf")}, {"t": [1]},
                    {"t": 1e-6}):
        A._require_exact_float(payload, "t", 1e-12, auditor, "c")
    assert len(auditor.blockers) == 8
    clean = A.Auditor()
    A._require_exact_float({"t": 1e-12}, "t", 1e-12, clean, "c")
    assert not clean.blockers


def test_CANARYAUDIT_relaxed_tolerance_cannot_be_published_as_pass(tmp_path):
    """§12: no mutated canary may ever yield status=PASS."""

    for updates in ({"canary_atol": 1e-3}, {"canary_rtol": 1.0},
                    {"estimand": "B"}, {"k_true": 4},
                    {"boundary_version": "v0"},
                    {"protocol_origin_issue_number": 55}):
        directory = _canary_audit_fixture(tmp_path / f"case{abs(hash(str(updates)))}",
                                          **updates)
        auditor = A.audit_canary_run_dir(directory)
        A.write_canary_audit_report(directory, auditor)
        verdict = json.loads((directory / "canary_audit.json").read_text(encoding="utf-8"))
        assert verdict["status"] != "PASS", updates


# --- the CLI refuses to stamp PASS on a mutated canary ---------------------


def test_CANARYAUDIT_cli_reports_fail_for_a_mutated_frozen_field(tmp_path, capsys):
    out = _canary_audit_fixture(tmp_path, canary_atol=1e-6)
    assert A.main(["--run-dir", str(out), "--mode", "canary", "--write-report"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "FAIL"
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "FAIL"


# --- the final smoke audit inherits the same checks ------------------------


@pytest.mark.parametrize("updates,check", [
    ({"estimand": "B"}, "smoke_canary_estimand"),
    ({"role": "sensitivity"}, "smoke_canary_role"),
    ({"k_true": 3}, "smoke_canary_k_true"),
    ({"replicate": 3}, "smoke_canary_replicate"),
    ({"protocol_origin_issue_number": 55}, "smoke_canary_protocol_issue"),
    ({"canary_atol": 1e-6}, "smoke_canary_atol"),
    ({"canary_rtol": 1e-3}, "smoke_canary_rtol"),
    ({"boundary_version": "v0"}, "smoke_canary_boundary_version"),
])
def test_CANARYAUDIT_final_smoke_audit_checks_the_same_fields(tmp_path, updates, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary.json", **updates)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))


# --- positive smoke control still runs -------------------------------------


def test_CANARYAUDIT_positive_lineage_still_runs_six(tmp_path):
    out = tmp_path / "run"
    _canary, _smoke, canary_recorder, smoke_recorder = _fake_run(out)
    assert canary_recorder.calls == 2 and smoke_recorder.calls == 6
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS"

    directory = _promote_to_real_fixture(out, tmp_path / "real")
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    A.write_smoke_audit_report(directory, auditor)
    assert json.loads(
        (directory / "audit_report.json").read_text(encoding="utf-8"))["status"] == "PASS"
    assert "em_runner" not in sys.modules


# ===========================================================================
# PR #56 rereview MEDIUM: frozen integer fields need a strict integer type
# ===========================================================================
#
# ``1.0 == 1``, ``53.0 == 53`` and ``True == 1`` are all true in Python, so a
# frozen integer compared by value alone could be replaced by an equal float or
# bool and the independent canary audit would still publish PASS.  Every frozen
# integer in the canary evidence (canary.json, authorization.json and the
# published canary_audit.json verdict) now requires ``type(value) is int``.


def test_STRICTINT_helper_requires_an_exact_int_and_never_raises():
    rejected = (
        {},                       # missing key
        {"v": 1.0},               # equal float
        {"v": True},              # bool is not an int here
        {"v": False},
        {"v": "1"},               # no string coercion
        {"v": None},
        {"v": []},
        {"v": {}},
        {"v": float("nan")},
        {"v": float("inf")},
        {"v": float("-inf")},
        {"v": 2},                 # right type, wrong value
    )
    for payload in rejected:
        auditor = A.Auditor()
        A._require_exact_int(payload, "v", 1, auditor, "c")   # must not raise
        assert len(auditor.blockers) == 1, payload
        assert auditor.blockers[0].check == "c"

    clean = A.Auditor()
    A._require_exact_int({"v": 1}, "v", 1, clean, "c")
    A._require_exact_int({"v": 53}, "v", 53, clean, "c")
    A._require_exact_int({"v": 0}, "v", 0, clean, "c")
    assert not clean.findings


def test_STRICTINT_helper_does_not_use_isinstance_or_coercion():
    body = _inspect.getsource(A._require_exact_int)
    assert "type(value) is not int" in body
    assert "isinstance(" not in body
    assert "int(value)" not in body and "float(value)" not in body


def test_STRICTINT_list_helper_requires_exact_ints():
    for payload in ({}, {"v": None}, {"v": [1.0, 2]}, {"v": [True, 2]},
                    {"v": ["1", 2]}, {"v": (1, 2)}, {"v": [1]}, {"v": [2, 1]}):
        auditor = A.Auditor()
        A._require_exact_int_list(payload, "v", [1, 2], auditor, "c")
        assert len(auditor.blockers) == 1, payload
    clean = A.Auditor()
    A._require_exact_int_list({"v": [1, 2]}, "v", [1, 2], clean, "c")
    assert not clean.findings


# --- the three fields named in the review ----------------------------------


@pytest.mark.parametrize("updates,check", [
    ({"k_true": 1.0}, "smoke_canary_k_true"),
    ({"replicate": 1.0}, "smoke_canary_replicate"),
    ({"protocol_origin_issue_number": 53.0}, "smoke_canary_protocol_issue"),
])
def test_STRICTINT_equal_float_cannot_replace_a_frozen_int(tmp_path, updates, check):
    out = _canary_audit_fixture(tmp_path, **updates)
    auditor = A.audit_canary_run_dir(out)                 # must not raise
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))

    A.write_canary_audit_report(out, auditor)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "FAIL"
    assert verdict["blocker_count"] > 0


@pytest.mark.parametrize("value", [1.0, True, False, "1", None, [], {}, 2])
def test_STRICTINT_canary_k_true_rejects_every_non_int(tmp_path, value):
    out = _canary_audit_fixture(tmp_path, k_true=value)
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == "smoke_canary_k_true" for f in auditor.blockers), value


@pytest.mark.parametrize("value", [1.0, True, False, "1", None, 2])
def test_STRICTINT_canary_replicate_rejects_every_non_int(tmp_path, value):
    out = _canary_audit_fixture(tmp_path, replicate=value)
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == "smoke_canary_replicate" for f in auditor.blockers), value


@pytest.mark.parametrize("value", [53.0, True, "53", None, 55, 0])
def test_STRICTINT_canary_protocol_issue_rejects_every_non_int(tmp_path, value):
    out = _canary_audit_fixture(tmp_path, protocol_origin_issue_number=value)
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == "smoke_canary_protocol_issue" for f in auditor.blockers), value


@pytest.mark.parametrize("key", ["k_true", "replicate", "protocol_origin_issue_number"])
def test_STRICTINT_canary_nonfinite_int_field_fails_closed(tmp_path, key):
    for name, value in (("nan", float("nan")), ("inf", float("inf")),
                        ("ninf", float("-inf"))):
        out = _canary_audit_fixture(tmp_path / f"{key}_{name}")
        _write_raw_canary(out / "canary.json", **{key: value})
        auditor = A.audit_canary_run_dir(out)             # must not raise
        assert auditor.blockers, (key, value)
        A.write_canary_audit_report(out, auditor)
        verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
        assert verdict["status"] == "FAIL"


# --- every other frozen integer in canary.json -----------------------------


@pytest.mark.parametrize("key,check", [
    ("execution_issue_number", "smoke_canary_execution_issue"),
    ("expected_fit_count", "smoke_canary_fit_count"),
    ("actual_fit_count", "smoke_canary_fit_count"),
    ("real_canary_fits_executed", "smoke_canary_real_fit_count"),
    ("k_est", "smoke_canary_cell"),
    ("start", "smoke_canary_cell"),
    ("model_seed", "smoke_canary_model_seed"),
    ("data_seed", "smoke_canary_data_seed"),
    ("split_seed", "smoke_canary_split_seed"),
    ("internal_retry", "smoke_canary_retry"),
    ("warning_count", "smoke_canary_warnings"),
])
def test_STRICTINT_every_canary_integer_field_rejects_an_equal_float(tmp_path, key, check):
    baseline = _canary_audit_fixture(tmp_path / "baseline")
    frozen = json.loads((baseline / "canary.json").read_text(encoding="utf-8"))
    assert type(frozen[key]) is int, (key, frozen[key])

    out = _canary_audit_fixture(tmp_path / "float", **{key: float(frozen[key])})
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == check for f in auditor.blockers), \
        (key, sorted({f.check for f in auditor.blockers}))

    if frozen[key] in (0, 1):
        out_bool = _canary_audit_fixture(tmp_path / "bool", **{key: bool(frozen[key])})
        auditor_bool = A.audit_canary_run_dir(out_bool)
        assert any(f.check == check for f in auditor_bool.blockers), key


# --- authorization.json frozen integers ------------------------------------


@pytest.mark.parametrize("key,check", [
    ("execution_issue_number", "smoke_auth_execution_issue"),
    ("protocol_origin_issue_number", "smoke_auth_protocol_issue"),
    ("expected_canary_fits", "smoke_auth_canary_count"),
    ("expected_smoke_fits", "smoke_auth_smoke_count"),
    ("k_true", "smoke_auth_cell"),
    ("replicate", "smoke_auth_cell"),
    ("split_seed", "smoke_auth_split_seed"),
    ("data_seed", "smoke_auth_data_seed"),
    ("canary_model_seed", "smoke_auth_canary_seed"),
])
def test_STRICTINT_every_authorization_integer_field_rejects_an_equal_float(
        tmp_path, key, check):
    out = _canary_audit_fixture(tmp_path)
    frozen = json.loads((out / "authorization.json").read_text(encoding="utf-8"))
    assert type(frozen[key]) is int, (key, frozen[key])

    _patch_json(out / "authorization.json", **{key: float(frozen[key])})
    auditor = A.audit_canary_run_dir(out)                 # must not raise
    assert any(f.check == check for f in auditor.blockers), \
        (key, sorted({f.check for f in auditor.blockers}))


@pytest.mark.parametrize("value", [1.0, True, False, "1", None])
def test_STRICTINT_authorization_k_true_rejects_every_non_int(tmp_path, value):
    out = _canary_audit_fixture(tmp_path)
    _patch_json(out / "authorization.json", k_true=value)
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == "smoke_auth_cell" for f in auditor.blockers), value


def test_STRICTINT_authorization_model_seed_vector_rejects_equal_floats(tmp_path):
    out = _canary_audit_fixture(tmp_path)
    frozen = json.loads((out / "authorization.json").read_text(encoding="utf-8"))
    assert all(type(seed) is int for seed in frozen["smoke_model_seeds"])
    _patch_json(out / "authorization.json",
                smoke_model_seeds=[float(seed) for seed in frozen["smoke_model_seeds"]])
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == "smoke_auth_model_seeds" for f in auditor.blockers)


# --- the published canary verdict, re-checked by the final smoke audit -----


@pytest.mark.parametrize("updates,check", [
    ({"blocker_count": 0.0}, "smoke_canary_audit_counts"),
    ({"high_count": 0.0}, "smoke_canary_audit_counts"),
    ({"blocker_count": False}, "smoke_canary_audit_counts"),
    ({"execution_issue": 55.0}, "smoke_canary_audit_execution_issue"),
    ({"protocol_origin_issue": 53.0}, "smoke_canary_audit_protocol_issue"),
    ({"expected_canary_fits": 2.0}, "smoke_canary_audit_fit_count"),
    ({"actual_canary_fits": 2.0}, "smoke_canary_audit_fit_count"),
])
def test_STRICTINT_canary_verdict_integer_fields_are_strict(tmp_path, updates, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json", **updates)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))


# --- final smoke audit inherits the strict canary typing -------------------


@pytest.mark.parametrize("updates,check", [
    ({"k_true": 1.0}, "smoke_canary_k_true"),
    ({"replicate": 1.0}, "smoke_canary_replicate"),
    ({"protocol_origin_issue_number": 53.0}, "smoke_canary_protocol_issue"),
    ({"k_true": True}, "smoke_canary_k_true"),
    ({"replicate": "1"}, "smoke_canary_replicate"),
    ({"protocol_origin_issue_number": None}, "smoke_canary_protocol_issue"),
])
def test_STRICTINT_final_smoke_audit_rejects_the_same_types(tmp_path, updates, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary.json", **updates)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))


# --- positive control: the canonical integers still pass -------------------


def test_STRICTINT_canonical_integers_still_pass(tmp_path):
    out = _canary_audit_fixture(tmp_path, k_true=1, replicate=1,
                                protocol_origin_issue_number=53)
    canary = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    assert (type(canary["k_true"]), type(canary["replicate"]),
            type(canary["protocol_origin_issue_number"])) == (int, int, int)

    auditor = A.audit_canary_run_dir(out)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    A.write_canary_audit_report(out, auditor)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS" and verdict["blocker_count"] == 0


def test_STRICTINT_no_regression_for_the_full_smoke_fixture(tmp_path):
    directory = _real_fixture(tmp_path)
    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs


# ===========================================================================
# PR #56 rereview: the same strict-integer sweep over the FINAL smoke JSON
# ===========================================================================
#
# Same root cause as the canary sweep above (``55 == 55.0``, ``0 == False``):
# runinfo.json and smoke_summary.json carry frozen/evidence integers that were
# still compared by value alone.  Booleans keep their ``is True`` / ``is False``
# contract and float fields keep their finite/exact-float contract; only integer
# schema fields are tightened here.


def _final_audit(directory):
    """Audit a final smoke fixture and publish the report; return both."""

    auditor = A.audit_smoke_run_dir(directory)
    A.write_smoke_audit_report(directory, auditor)
    report = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    return auditor, report


# --- runinfo.json: equal floats --------------------------------------------


@pytest.mark.parametrize("key,check", [
    ("execution_issue", "smoke_runinfo_execution_issue"),
    ("protocol_origin_issue", "smoke_runinfo_protocol_issue"),
    ("expected_real_em_budget", "smoke_runinfo_budget"),
    ("expected_canary_fits", "smoke_runinfo_expected_canary_fits"),
    ("expected_smoke_fits", "smoke_runinfo_expected_smoke_fits"),
    ("actual_canary_fits", "smoke_runinfo_canary_count"),
    ("actual_smoke_fits", "smoke_runinfo_smoke_count"),
    ("full_fits_executed", "smoke_runinfo_full_fits"),
    ("phase7e_rerun_count", "smoke_runinfo_phase7e_rerun"),
])
def test_STRICTINTFINAL_runinfo_rejects_an_equal_float(tmp_path, key, check):
    directory = _real_fixture(tmp_path)
    frozen = json.loads((directory / "runinfo.json").read_text(encoding="utf-8"))
    assert type(frozen[key]) is int, (key, frozen[key])

    _patch_json(directory / "runinfo.json", **{key: float(frozen[key])})
    auditor, report = _final_audit(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (key, sorted({f.check for f in auditor.blockers}))
    assert report["status"] == "FAIL" and report["blocker_count"] > 0


@pytest.mark.parametrize("key,value,check", [
    ("execution_issue", True, "smoke_runinfo_execution_issue"),
    ("actual_canary_fits", True, "smoke_runinfo_canary_count"),
    ("expected_canary_fits", True, "smoke_runinfo_expected_canary_fits"),
    ("full_fits_executed", False, "smoke_runinfo_full_fits"),
    ("phase7e_rerun_count", False, "smoke_runinfo_phase7e_rerun"),
    ("actual_smoke_fits", "6", "smoke_runinfo_smoke_count"),
    ("expected_real_em_budget", None, "smoke_runinfo_budget"),
    ("protocol_origin_issue", "53", "smoke_runinfo_protocol_issue"),
])
def test_STRICTINTFINAL_runinfo_rejects_bools_strings_and_none(tmp_path, key, value, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "runinfo.json", **{key: value})
    auditor = A.audit_smoke_run_dir(directory)          # must not raise
    assert any(f.check == check for f in auditor.blockers), \
        (key, value, sorted({f.check for f in auditor.blockers}))


@pytest.mark.parametrize("key,check", [
    ("execution_issue", "smoke_runinfo_execution_issue"),
    ("expected_canary_fits", "smoke_runinfo_expected_canary_fits"),
    ("full_fits_executed", "smoke_runinfo_full_fits"),
])
def test_STRICTINTFINAL_runinfo_missing_integer_field_blocks(tmp_path, key, check):
    directory = _real_fixture(tmp_path)
    payload = json.loads((directory / "runinfo.json").read_text(encoding="utf-8"))
    payload.pop(key)
    (directory / "runinfo.json").write_text(json.dumps(payload), encoding="utf-8")
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), key


# --- runinfo.json: booleans are NOT integers -------------------------------


@pytest.mark.parametrize("key,value,check", [
    ("working_tree_clean", 1, "smoke_runinfo_working_tree"),
    ("working_tree_clean", 1.0, "smoke_runinfo_working_tree"),
    ("approved_baseline_is_ancestor", 1, "smoke_runinfo_lineage"),
    ("approved_baseline_is_ancestor", "True", "smoke_runinfo_lineage"),
])
def test_STRICTINTFINAL_runinfo_booleans_stay_literal(tmp_path, key, value, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "runinfo.json", **{key: value})
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), (key, value)


# --- smoke_summary.json: equal floats --------------------------------------


@pytest.mark.parametrize("key,check", [
    ("execution_issue_number", "smoke_summary_execution_issue"),
    ("protocol_origin_issue_number", "smoke_summary_protocol_issue"),
    ("k_true", "smoke_summary_k_true"),
    ("expected_smoke_fits", "smoke_summary_fit_count"),
    ("actual_smoke_fits", "smoke_summary_fit_count"),
    ("selected_k", "smoke_summary_selected_k"),
])
def test_STRICTINTFINAL_summary_rejects_an_equal_float(tmp_path, key, check):
    directory = _real_fixture(tmp_path)
    frozen = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    assert type(frozen[key]) is int, (key, frozen[key])

    _patch_json(directory / "smoke_summary.json", **{key: float(frozen[key])})
    auditor, report = _final_audit(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (key, sorted({f.check for f in auditor.blockers}))
    assert report["status"] == "FAIL" and report["blocker_count"] > 0


@pytest.mark.parametrize("key,value,check", [
    ("k_true", True, "smoke_summary_k_true"),
    ("k_true", "1", "smoke_summary_k_true"),
    ("k_true", None, "smoke_summary_k_true"),
    ("execution_issue_number", True, "smoke_summary_execution_issue"),
    ("protocol_origin_issue_number", None, "smoke_summary_protocol_issue"),
    ("actual_smoke_fits", "6", "smoke_summary_fit_count"),
])
def test_STRICTINTFINAL_summary_rejects_bools_strings_and_none(tmp_path, key, value, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "smoke_summary.json", **{key: value})
    auditor = A.audit_smoke_run_dir(directory)          # must not raise
    assert any(f.check == check for f in auditor.blockers), \
        (key, value, sorted({f.check for f in auditor.blockers}))


# --- selected_k: integer schema, frozen candidate set, recomputed value ----


@pytest.mark.parametrize("value", [True, "3", None, 1.0, 3.0, []])
def test_STRICTINTFINAL_selected_k_rejects_every_non_int(tmp_path, value):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "smoke_summary.json", selected_k=value)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_summary_selected_k" for f in auditor.blockers), value


@pytest.mark.parametrize("value", [1, 5, 0, -3])
def test_STRICTINTFINAL_selected_k_must_be_in_the_frozen_candidate_set(tmp_path, value):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "smoke_summary.json", selected_k=value)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_summary_selected_k" for f in auditor.blockers), value


def test_STRICTINTFINAL_selected_k_float_of_the_real_value_blocks(tmp_path):
    """§24: the actual selected K, written as a float, must not pass."""

    directory = _real_fixture(tmp_path)
    selected = json.loads(
        (directory / "smoke_summary.json").read_text(encoding="utf-8"))["selected_k"]
    assert type(selected) is int and selected in (2, 3, 4)

    _patch_json(directory / "smoke_summary.json", selected_k=float(selected))
    auditor, report = _final_audit(directory)
    assert any(f.check == "smoke_summary_selected_k" for f in auditor.blockers)
    assert report["status"] == "FAIL"


def test_STRICTINTFINAL_selected_k_interpretation_is_untouched(tmp_path):
    directory = _real_fixture(tmp_path)
    summary = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_k_interpretation"] == "record_only"
    assert summary["k_recovery_evaluated"] is False
    body = _inspect.getsource(A.audit_smoke_summary)
    assert "SMOKE_K_TRUE" not in body          # K recovery is never evaluated
    assert "recovery" in body


@pytest.mark.parametrize("value", [1, 0, "False", None])
def test_STRICTINTFINAL_k_recovery_flag_stays_a_literal_bool(tmp_path, value):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "smoke_summary.json", k_recovery_evaluated=value)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_summary_k_recovery_flag" for f in auditor.blockers), value


# --- candidate_k / tie_candidates: element-wise strict integers -------------


@pytest.mark.parametrize("value", [
    [2.0, 3, 4],
    [2, 3.0, 4],
    [2, 3, 4.0],
    [True, 3, 4],
    ["2", 3, 4],
    [2, 3],
    [2, 3, 4, 5],
    None,
])
def test_STRICTINTFINAL_candidate_k_rejects_non_int_elements(tmp_path, value):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "smoke_summary.json", candidate_k=value)
    auditor, report = _final_audit(directory)
    assert any(f.check == "smoke_summary_candidates" for f in auditor.blockers), value
    assert report["status"] == "FAIL"


def test_STRICTINTFINAL_tie_candidates_reject_equal_floats_and_bools(tmp_path):
    directory = _real_fixture(tmp_path)
    tied = json.loads(
        (directory / "smoke_summary.json").read_text(encoding="utf-8"))["tie_candidates"]
    assert tied and all(type(t) is int for t in tied)

    for value in ([float(t) for t in tied], [True], ["2"], [bool(tied[0])]):
        _patch_json(directory / "smoke_summary.json", tie_candidates=value)
        auditor = A.audit_smoke_run_dir(directory)      # must not raise
        assert any(f.check == "smoke_summary_tie_candidates" for f in auditor.blockers), value


def test_STRICTINTFINAL_tie_candidates_accept_the_recomputed_set(tmp_path):
    directory = _real_fixture(tmp_path)
    tied = json.loads(
        (directory / "smoke_summary.json").read_text(encoding="utf-8"))["tie_candidates"]
    _patch_json(directory / "smoke_summary.json", tie_candidates=list(reversed(tied)))
    auditor = A.audit_smoke_run_dir(directory)
    assert not any(f.check == "smoke_summary_tie_candidates" for f in auditor.blockers)


# --- per_k keys stay strings ----------------------------------------------


def test_STRICTINTFINAL_per_k_keys_are_string_typed_and_exact(tmp_path):
    directory = _real_fixture(tmp_path)
    summary = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    assert set(summary["per_k"]) == {"2", "3", "4"}
    assert all(type(key) is str for key in summary["per_k"])

    body = _inspect.getsource(A.audit_smoke_summary)
    assert "int(k)" not in body                 # no key coercion was introduced

    for mutated in ({"2": summary["per_k"]["2"], "3": summary["per_k"]["3"]},
                    dict(summary["per_k"], **{"5": summary["per_k"]["4"]})):
        _patch_json(directory / "smoke_summary.json", per_k=mutated)
        auditor = A.audit_smoke_run_dir(directory)
        assert any(f.check == "smoke_summary_per_k" for f in auditor.blockers), mutated


# --- float schema fields are NOT integerised -------------------------------


def test_STRICTINTFINAL_float_fields_keep_their_float_contract(tmp_path):
    directory = _real_fixture(tmp_path)
    summary = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    assert all(isinstance(entry["mean"], float) for entry in summary["per_k"].values())
    canary = json.loads((directory / "canary.json").read_text(encoding="utf-8"))
    assert isinstance(canary["canary_atol"], float) and canary["canary_atol"] == 1e-12
    assert isinstance(canary["canary_rtol"], float) and canary["canary_rtol"] == 1e-10

    auditor = A.audit_smoke_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]


# --- the helpers themselves -------------------------------------------------


def test_STRICTINTFINAL_member_helper_is_strict_and_never_raises():
    for payload in ({}, {"v": 3.0}, {"v": True}, {"v": "3"}, {"v": None},
                    {"v": []}, {"v": 1}, {"v": 5}):
        auditor = A.Auditor()
        A._require_exact_int_member(payload, "v", (2, 3, 4), auditor, "c")
        assert len(auditor.blockers) == 1, payload
    clean = A.Auditor()
    for value in (2, 3, 4):
        A._require_exact_int_member({"v": value}, "v", (2, 3, 4), clean, "c")
    assert not clean.findings
    body = _inspect.getsource(A._require_exact_int_member)
    assert "type(value) is not int" in body and "isinstance(" not in body


def test_STRICTINTFINAL_sorted_list_helper_is_element_strict():
    auditor = A.Auditor()
    A._require_exact_int_list({"v": [3, 2]}, "v", [2, 3], auditor, "c", sort=True)
    assert not auditor.findings
    for value in ([2.0, 3], [True, 3], ["2", 3], [2], (2, 3)):
        bad = A.Auditor()
        A._require_exact_int_list({"v": value}, "v", [2, 3], bad, "c", sort=True)
        assert len(bad.blockers) == 1, value


# --- positive control -------------------------------------------------------


def test_STRICTINTFINAL_canonical_final_fixture_still_passes(tmp_path):
    directory = _real_fixture(tmp_path)
    runinfo = json.loads((directory / "runinfo.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "smoke_summary.json").read_text(encoding="utf-8"))
    for key, expected in (("execution_issue", 55), ("protocol_origin_issue", 53),
                          ("expected_real_em_budget", 8), ("expected_canary_fits", 2),
                          ("expected_smoke_fits", 6), ("actual_canary_fits", 2),
                          ("actual_smoke_fits", 6), ("full_fits_executed", 0),
                          ("phase7e_rerun_count", 0)):
        assert type(runinfo[key]) is int and runinfo[key] == expected, key
    for key, expected in (("execution_issue_number", 55),
                          ("protocol_origin_issue_number", 53),
                          ("k_true", 1), ("expected_smoke_fits", 6),
                          ("actual_smoke_fits", 6)):
        assert type(summary[key]) is int and summary[key] == expected, key
    assert summary["candidate_k"] == [2, 3, 4]
    assert all(type(k) is int for k in summary["candidate_k"])
    assert runinfo["approved_baseline_is_ancestor"] is True
    assert runinfo["working_tree_clean"] is True

    auditor, report = _final_audit(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    assert report["status"] == "PASS" and report["blocker_count"] == 0
    assert summary["selected_k_interpretation"] == "record_only"
    assert summary["k_recovery_evaluated"] is False


# --- canary regression ------------------------------------------------------


@pytest.mark.parametrize("updates,check", [
    ({"k_true": 1.0}, "smoke_canary_k_true"),
    ({"replicate": 1.0}, "smoke_canary_replicate"),
    ({"protocol_origin_issue_number": 53.0}, "smoke_canary_protocol_issue"),
])
def test_STRICTINTFINAL_canary_sweep_has_not_regressed(tmp_path, updates, check):
    out = _canary_audit_fixture(tmp_path, **updates)
    auditor = A.audit_canary_run_dir(out)
    assert any(f.check == check for f in auditor.blockers), \
        (check, sorted({f.check for f in auditor.blockers}))
    A.write_canary_audit_report(out, auditor)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "FAIL"


def test_STRICTINTFINAL_canary_positive_control_still_passes(tmp_path):
    out = _canary_audit_fixture(tmp_path)
    auditor = A.audit_canary_run_dir(out)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs


# --- frozen protocol and authorization are untouched ------------------------


def test_STRICTINTFINAL_protocol_and_authorization_are_unchanged():
    assert A.EXPECTED_SMOKE_PROTOCOL_HASH == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert (A.SMOKE_ESTIMAND, A.SMOKE_ROLE) == ("A", "primary")
    assert A.SMOKE_K_TRUE == 1 and A.SMOKE_REPLICATE == 1
    assert A.SMOKE_K_CANDIDATES == (2, 3, 4) and A.SMOKE_STARTS == (1, 2)
    assert (A.EXPECTED_CANARY_FITS, A.EXPECTED_SMOKE_FITS,
            A.EXPECTED_REAL_EM_BUDGET) == (2, 6, 8)
    assert H.EXPECTED_NEW_FITS == 336
    assert H.current_smoke_execution_authorization() is not None
    authorization = H.current_smoke_authorization()
    assert authorization.independent_review_pass is False
    assert authorization.human_smoke_approval is False
    assert authorization.authorized() is False


# ===========================================================================
# PR #56 rereview: canary_audit.json.medium_count strict-integer boundary
# ===========================================================================
#
# medium_count was the one published count with no contract at all: missing,
# 0.0, False and "0" were all accepted.  It is NOT frozen at zero -- the PASS
# policy is BLOCKER=0 and HIGH=0, and a MEDIUM finding does not block -- so the
# contract is "present, exactly int, >= 0, and equal to the MEDIUM findings".


def _medium_finding(check="synthetic", detail="schema control"):
    return {"severity": "MEDIUM", "check": check, "detail": detail}


# --- §9: type attacks on the published verdict ------------------------------


@pytest.mark.parametrize("value", [0.0, False, True, "0", None, -1, [], {}, 1.0])
def test_MEDIUMCOUNT_final_audit_rejects_a_non_int(tmp_path, value):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json", medium_count=value)
    auditor, report = _final_audit(directory)          # must not raise
    assert any(f.check == "smoke_canary_audit_medium_count" for f in auditor.blockers), \
        (value, sorted({f.check for f in auditor.blockers}))
    assert report["status"] == "FAIL" and report["blocker_count"] > 0


def test_MEDIUMCOUNT_final_audit_rejects_a_missing_count(tmp_path):
    directory = _real_fixture(tmp_path)
    payload = json.loads((directory / "canary_audit.json").read_text(encoding="utf-8"))
    payload.pop("medium_count")
    (directory / "canary_audit.json").write_text(json.dumps(payload), encoding="utf-8")
    auditor, report = _final_audit(directory)
    assert any(f.check == "smoke_canary_audit_medium_count" for f in auditor.blockers)
    assert report["status"] == "FAIL"


def test_MEDIUMCOUNT_helper_is_strict_and_never_raises():
    for payload in ({}, {"v": 0.0}, {"v": False}, {"v": True}, {"v": "0"},
                    {"v": None}, {"v": -1}, {"v": []}, {"v": float("nan")}):
        auditor = A.Auditor()
        assert A._require_nonnegative_int(payload, "v", auditor, "c") is None, payload
        assert len(auditor.blockers) == 1, payload
    clean = A.Auditor()
    for value in (0, 1, 7):
        assert A._require_nonnegative_int({"v": value}, "v", clean, "c") == value
    assert not clean.findings
    body = _inspect.getsource(A._require_nonnegative_int)
    assert "type(value) is not int" in body and "isinstance(" not in body


# --- §10 / §6: the counts must agree with the findings they summarise -------


def test_MEDIUMCOUNT_count_must_match_zero_medium_findings(tmp_path):
    directory = _real_fixture(tmp_path)
    verdict = json.loads((directory / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["findings"] == [] and verdict["medium_count"] == 0

    _patch_json(directory / "canary_audit.json", medium_count=1)
    auditor, report = _final_audit(directory)
    assert any(f.check == "smoke_canary_audit_count_consistency" for f in auditor.blockers)
    assert report["status"] == "FAIL"


def test_MEDIUMCOUNT_count_must_match_one_medium_finding(tmp_path):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json",
                findings=[_medium_finding()], medium_count=0)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_canary_audit_count_consistency" for f in auditor.blockers)


@pytest.mark.parametrize("severity,key", [("BLOCKER", "blocker_count"),
                                          ("HIGH", "high_count")])
def test_MEDIUMCOUNT_blocking_counts_must_match_the_findings(tmp_path, severity, key):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json",
                findings=[{"severity": severity, "check": "x", "detail": "y"}])
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_canary_audit_count_consistency" for f in auditor.blockers), key


@pytest.mark.parametrize("findings", [None, "[]", {}, [1, 2], ["MEDIUM"]])
def test_MEDIUMCOUNT_findings_must_be_a_list_of_objects(tmp_path, findings):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json", findings=findings)
    auditor = A.audit_smoke_run_dir(directory)          # must not raise
    assert any(f.check == "smoke_canary_audit_findings" for f in auditor.blockers), findings


# --- §12 / §5: a PASS verdict may carry MEDIUM findings ---------------------


def test_MEDIUMCOUNT_a_medium_finding_does_not_break_the_pass_policy(tmp_path):
    """§5: PASS iff BLOCKER=0 and HIGH=0.  MEDIUM alone changes nothing."""

    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json",
                findings=[_medium_finding()], medium_count=1)
    auditor, report = _final_audit(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    assert report["status"] == "PASS"


def test_MEDIUMCOUNT_pass_policy_is_still_decided_by_blocker_and_high(tmp_path):
    """The published verdict itself: MEDIUM does not turn a PASS into a FAIL."""

    auditor = A.Auditor()
    auditor.record("MEDIUM", "synthetic", "a non-blocking observation")
    report = A.build_canary_audit_report(auditor, tmp_path)
    assert report["status"] == "PASS"
    assert report["blocker_count"] == 0 and report["high_count"] == 0
    assert report["medium_count"] == 1 and type(report["medium_count"]) is int

    blocking = A.Auditor()
    blocking.blocker("synthetic", "a blocking finding")
    assert A.build_canary_audit_report(blocking, tmp_path)["status"] == "FAIL"


def test_MEDIUMCOUNT_published_verdict_is_self_consistent(tmp_path):
    """What the auditor publishes must satisfy the contract it enforces."""

    auditor = A.Auditor()
    auditor.record("MEDIUM", "synthetic", "a non-blocking observation")
    report = A.build_canary_audit_report(auditor, tmp_path)
    checker = A.Auditor()
    A.audit_canary_verdict_counts(report, checker)
    assert not checker.findings, [f"{f.check}: {f.detail}" for f in checker.findings]


# --- §11: the canonical verdict ---------------------------------------------


def test_MEDIUMCOUNT_positive_control(tmp_path):
    directory = _real_fixture(tmp_path)
    verdict = json.loads((directory / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["blocker_count"] == 0 and verdict["high_count"] == 0
    assert verdict["medium_count"] == 0 and verdict["findings"] == []
    assert type(verdict["medium_count"]) is int
    assert verdict["status"] == "PASS"

    auditor, report = _final_audit(directory)
    assert not auditor.blockers and not auditor.highs
    assert report["status"] == "PASS"


def test_MEDIUMCOUNT_canary_only_verdict_carries_the_count(tmp_path):
    out = _canary_audit_fixture(tmp_path)
    auditor = A.audit_canary_run_dir(out)
    A.write_canary_audit_report(out, auditor)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS"
    assert type(verdict["medium_count"]) is int and verdict["medium_count"] == 0


# --- §7 / §13: the runner's smoke gate --------------------------------------


def test_MEDIUMCOUNT_runner_required_keys_include_the_count():
    assert "medium_count" in H.CANARY_AUDIT_REQUIRED_KEYS
    for key in ("blocker_count", "high_count", "findings", "status"):
        assert key in H.CANARY_AUDIT_REQUIRED_KEYS, key
    contract = H.run_smoke_contract()
    assert "medium_count" in contract["canary_audit_keys"]
    assert contract["em_fits_executed"] == 0


def test_MEDIUMCOUNT_smoke_gate_rejects_a_missing_count(tmp_path, monkeypatch):
    out = _canary_only(tmp_path)
    _write_test_canary_audit(out)
    payload = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    payload.pop("medium_count")
    (out / "canary_audit.json").write_text(json.dumps(payload), encoding="utf-8")
    message = _attempt_smoke(out, monkeypatch)
    assert "incomplete" in message and "medium_count" in message


@pytest.mark.parametrize("value", [0.0, False, True, "0", None, -1])
def test_MEDIUMCOUNT_smoke_gate_rejects_a_non_int(tmp_path, monkeypatch, value):
    out = _canary_only(tmp_path)
    _write_test_canary_audit(out)
    _patch_json(out / "canary_audit.json", medium_count=value)
    message = _attempt_smoke(out, monkeypatch)
    assert "medium_count" in message, message


def test_MEDIUMCOUNT_smoke_gate_still_accepts_a_medium_finding(tmp_path, monkeypatch):
    """Schema hardening only: a PASS verdict with a MEDIUM still authorises."""

    out = _canary_only(tmp_path)
    _write_test_canary_audit(out)
    _patch_json(out / "canary_audit.json", findings=[_medium_finding()], medium_count=1)
    canary_payload = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    verdict = H.require_canary_audit_pass(out, _test_authorization(), canary_payload,
                                          current_run_code_sha="0" * 40, test_only=True)
    assert verdict["medium_count"] == 1 and verdict["status"] == "PASS"


def test_MEDIUMCOUNT_runner_still_only_reads_the_verdict():
    runner = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    assert "write_canary_audit_report" not in runner
    assert "build_canary_audit_report" not in runner
    body = _executable_body(H.require_canary_audit_pass)
    assert "read_json_artifact" in body and "write_json_artifact" not in body


# --- §15: previously fixed fields have not regressed ------------------------


@pytest.mark.parametrize("name,updates,check", [
    ("canary.json", {"k_true": 1.0}, "smoke_canary_k_true"),
    ("canary.json", {"replicate": 1.0}, "smoke_canary_replicate"),
    ("canary.json", {"protocol_origin_issue_number": 53.0}, "smoke_canary_protocol_issue"),
    ("runinfo.json", {"execution_issue": 55.0}, "smoke_runinfo_execution_issue"),
    ("runinfo.json", {"expected_real_em_budget": 8.0}, "smoke_runinfo_budget"),
    ("runinfo.json", {"actual_smoke_fits": 6.0}, "smoke_runinfo_smoke_count"),
    ("runinfo.json", {"full_fits_executed": 0.0}, "smoke_runinfo_full_fits"),
    ("smoke_summary.json", {"k_true": 1.0}, "smoke_summary_k_true"),
    ("smoke_summary.json", {"candidate_k": [2.0, 3, 4]}, "smoke_summary_candidates"),
    ("canary_audit.json", {"blocker_count": 0.0}, "smoke_canary_audit_counts"),
    ("canary_audit.json", {"actual_canary_fits": 2.0}, "smoke_canary_audit_fit_count"),
])
def test_MEDIUMCOUNT_strict_int_sweep_has_not_regressed(tmp_path, name, updates, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / name, **updates)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (name, sorted({f.check for f in auditor.blockers}))


def test_MEDIUMCOUNT_selected_k_float_still_blocks(tmp_path):
    directory = _real_fixture(tmp_path)
    selected = json.loads(
        (directory / "smoke_summary.json").read_text(encoding="utf-8"))["selected_k"]
    _patch_json(directory / "smoke_summary.json", selected_k=float(selected))
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_summary_selected_k" for f in auditor.blockers)


# --- §16 / §17: frozen science and the authorization ------------------------


def test_MEDIUMCOUNT_frozen_science_and_authorization_are_unchanged():
    assert A.APPROVED_SCIENTIFIC_MAIN_SHA == "68c78e1191889609dead05ea5a9fb11525ce92e2"
    assert A.EXPECTED_SMOKE_PROTOCOL_HASH == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert H.smoke_protocol_hash() == A.EXPECTED_SMOKE_PROTOCOL_HASH
    assert (A.SMOKE_PROTOCOL_ISSUE_NUMBER, A.SMOKE_EXECUTION_ISSUE_NUMBER) == (53, 55)
    assert (A.SMOKE_ESTIMAND, A.SMOKE_ROLE) == ("A", "primary")
    assert A.SMOKE_K_TRUE == 1 and A.SMOKE_REPLICATE == 1
    assert A.SMOKE_K_CANDIDATES == (2, 3, 4) and A.SMOKE_STARTS == (1, 2)
    assert (A.EXPECTED_CANARY_FITS, A.EXPECTED_SMOKE_FITS,
            A.EXPECTED_REAL_EM_BUDGET) == (2, 6, 8)
    assert H.EXPECTED_NEW_FITS == 336

    assert H.current_smoke_execution_authorization() is not None
    authorization = H.current_smoke_authorization()
    assert authorization.independent_review_pass is False
    assert authorization.human_smoke_approval is False
    assert authorization.authorized() is False


# ===========================================================================
# PR #56 remote review HIGH: bind the canary verdict to the audited content
# ===========================================================================
#
# A canary_audit.json PASS used to name the baseline, the protocol hash, the
# issues and the run-code SHA, but nothing about the bytes it audited.  Anyone
# could mutate authorization.json or canary.json after the audit and the runner
# would still accept the stale PASS.  The verdict now carries the SHA-256 of the
# exact bytes the auditor read, and both the runner gate and the final smoke
# audit rehash the files and compare.


def _sha256_file(path):
    return _hashlib.sha256(path.read_bytes()).hexdigest()


def _smoke_after(out, monkeypatch, *, run_code_sha="0" * 40):
    """Attempt the production-equivalent smoke; return (proceeded, message)."""

    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    recorder = _FakeFitRecorder()
    try:
        H._execute_real_smoke_test_only(_test_authorization(), out,
                                        adapter=_test_adapter(recorder),
                                        run_code_sha=run_code_sha)
        proceeded, message = True, ""
    except HarnessStop as error:
        proceeded, message = False, str(error)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    return proceeded, message, recorder.calls


def _audited_canary_dir(tmp_path, name="run"):
    out = tmp_path / name
    H._execute_real_canary_test_only(_test_authorization(), out,
                                     adapter=_test_adapter(_FakeFitRecorder()),
                                     run_code_sha="0" * 40)
    _write_test_canary_audit(out)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS"
    return out


# --- the exact-byte digest itself -------------------------------------------


def test_CONTENTBIND_digest_is_sha256_of_the_exact_file_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{"a": 1}\r\n')
    auditor = A.Auditor()
    payload, digest = A._read_json_and_digest(path, auditor)
    assert payload == {"a": 1}
    assert digest == _hashlib.sha256(b'{"a": 1}\r\n').hexdigest()
    assert digest == A._sha256_hex(path.read_bytes())
    assert len(digest) == 64 and digest == digest.lower()
    assert not auditor.findings

    # the bytes are what is hashed, not a canonical reserialization
    other = tmp_path / "other.json"
    other.write_bytes(b'{"a":1}')
    _payload, other_digest = A._read_json_and_digest(other, A.Auditor())
    assert other_digest != digest


def test_CONTENTBIND_reader_reads_the_file_exactly_once():
    body = _inspect.getsource(A._read_json_and_digest)
    assert body.count("read_bytes()") == 1
    assert "read_text(" not in body
    # the digest must not be recomputed when the report is built
    builder = _inspect.getsource(A.build_canary_audit_report)
    assert "_sha256_hex(" not in builder and "hashlib" not in builder
    assert "read_bytes" not in builder
    assert "auditor.audited_digests" in builder


def test_CONTENTBIND_audit_module_hashes_independently():
    source = _inspect.getsource(A)
    assert "import run_k_true_robustness_sweep" not in source
    assert "import hashlib" in source
    for function in (A._sha256_hex, A._read_json_and_digest,
                     A.audit_canary_verdict_content_binding):
        body = _inspect.getsource(function)
        assert "run_k_true_robustness_sweep" not in body


def test_CONTENTBIND_digest_survives_a_malformed_payload(tmp_path):
    path = tmp_path / "broken.json"
    path.write_bytes(b"{not json")
    auditor = A.Auditor()
    payload, digest = A._read_json_and_digest(path, auditor)
    assert payload is None
    assert digest == _hashlib.sha256(b"{not json").hexdigest()
    assert auditor.blockers


# --- the published verdict ---------------------------------------------------


def test_CONTENTBIND_canary_verdict_records_both_digests(tmp_path):
    out = _canary_audit_fixture(tmp_path)
    auditor = A.audit_canary_run_dir(out)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert auditor.audited_digests == {
        "authorization.json": _sha256_file(out / "authorization.json"),
        "canary.json": _sha256_file(out / "canary.json"),
    }

    A.write_canary_audit_report(out, auditor)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS"
    assert verdict["authorization_content_sha256"] == _sha256_file(out / "authorization.json")
    assert verdict["canary_content_sha256"] == _sha256_file(out / "canary.json")
    for key in ("authorization_content_sha256", "canary_content_sha256"):
        value = verdict[key]
        assert type(value) is str and len(value) == 64
        assert set(value) <= set("0123456789abcdef")


def test_CONTENTBIND_cli_published_verdict_is_bound(tmp_path, capsys):
    out = _canary_audit_fixture(tmp_path)
    assert A.main(["--run-dir", str(out), "--mode", "canary", "--write-report"]) == 0
    capsys.readouterr()
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["authorization_content_sha256"] == _sha256_file(out / "authorization.json")
    assert verdict["canary_content_sha256"] == _sha256_file(out / "canary.json")


def test_CONTENTBIND_hex_helper_is_strict():
    for value in ("abc", "0" * 63, "0" * 65, "A" * 64, "0" * 63 + "g",
                  None, 1, True, ["0" * 64], "0" * 64 + " "):
        auditor = A.Auditor()
        assert A._require_sha256_hex({"d": value}, "d", auditor, "c") is None, value
        assert len(auditor.blockers) == 1, value
    missing = A.Auditor()
    assert A._require_sha256_hex({}, "d", missing, "c") is None
    assert len(missing.blockers) == 1
    clean = A.Auditor()
    assert A._require_sha256_hex({"d": "a1" * 32}, "d", clean, "c") == "a1" * 32
    assert not clean.findings
    body = _inspect.getsource(A._require_sha256_hex)
    assert ".lower()" not in body and ".upper()" not in body      # nothing is normalised


# --- final smoke audit: attacks A / B / C ------------------------------------


@pytest.mark.parametrize("name,updates", [
    ("canary.json", {"k_true": 1.0}),
    ("canary.json", {"protocol_origin_issue_number": 53.0}),
    ("authorization.json", {"k_true": 1.0}),
    ("authorization.json", {"harmless_extra_field": "x"}),
])
def test_CONTENTBIND_final_audit_detects_a_post_audit_mutation(tmp_path, name, updates):
    directory = _real_fixture(tmp_path)
    before = _sha256_file(directory / name)
    _patch_json(directory / name, **updates)
    assert _sha256_file(directory / name) != before, "the mutation must change the bytes"

    auditor, report = _final_audit(directory)
    assert any(f.check == "smoke_canary_audit_content_binding" for f in auditor.blockers), \
        (name, sorted({f.check for f in auditor.blockers}))
    assert report["status"] == "FAIL" and report["blocker_count"] > 0


def test_CONTENTBIND_final_audit_detects_a_whitespace_only_mutation(tmp_path):
    """Exact-byte contract: reformatting the same JSON is still a change."""

    directory = _real_fixture(tmp_path)
    path = directory / "authorization.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, sort_keys=True, indent=4), encoding="utf-8")

    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == "smoke_canary_audit_content_binding" for f in auditor.blockers)


@pytest.mark.parametrize("key", ["authorization_content_sha256", "canary_content_sha256"])
def test_CONTENTBIND_final_audit_rejects_a_tampered_digest(tmp_path, key):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json", **{key: "0" * 64})
    auditor, report = _final_audit(directory)
    assert any(f.check == "smoke_canary_audit_content_binding" for f in auditor.blockers)
    assert report["status"] == "FAIL"


@pytest.mark.parametrize("value", ["abc", "0" * 63, "A" * 64, "0" * 63 + "z", None, 1, True])
def test_CONTENTBIND_final_audit_rejects_a_malformed_digest(tmp_path, value):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / "canary_audit.json", canary_content_sha256=value)
    auditor = A.audit_smoke_run_dir(directory)          # must not raise
    assert any(f.check == "smoke_canary_audit_content_digest" for f in auditor.blockers), value


@pytest.mark.parametrize("key", ["authorization_content_sha256", "canary_content_sha256"])
def test_CONTENTBIND_final_audit_rejects_a_missing_digest(tmp_path, key):
    directory = _real_fixture(tmp_path)
    payload = json.loads((directory / "canary_audit.json").read_text(encoding="utf-8"))
    payload.pop(key)
    (directory / "canary_audit.json").write_text(json.dumps(payload), encoding="utf-8")
    auditor, report = _final_audit(directory)
    assert any(f.check == "smoke_canary_audit_content_digest" for f in auditor.blockers)
    assert report["status"] == "FAIL"


# --- the runner gate: attacks A / B / C / D / E ------------------------------


def test_CONTENTBIND_runner_required_keys_include_both_digests():
    for key in ("authorization_content_sha256", "canary_content_sha256"):
        assert key in H.CANARY_AUDIT_REQUIRED_KEYS, key
        assert key in H.run_smoke_contract()["canary_audit_keys"], key
    assert H.run_smoke_contract()["em_fits_executed"] == 0


def test_CONTENTBIND_runner_verifies_before_the_adapter():
    body = _executable_body(H._execute_real_smoke)
    order = [body.index(token) for token in (
        "prepare_smoke_cell(", "require_existing_smoke_artifact_dir(",
        "require_canary_pass_evidence(", "require_canary_audit_pass(",
        "_resolve_fit_adapter(", "_run_real_smoke(")]
    assert order == sorted(order), body
    gate = _executable_body(H.require_canary_audit_pass)
    assert "hashlib.sha256(" in gate and "read_bytes()" in gate
    # the runner verifies, it never produces a verdict
    assert "write_json_artifact" not in gate
    runner = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    assert "build_canary_audit_report" not in runner
    assert "write_canary_audit_report" not in runner


@pytest.mark.parametrize("name,updates", [
    ("canary.json", {"k_true": 1.0}),
    ("canary.json", {"protocol_origin_issue_number": 53.0}),
    ("authorization.json", {"k_true": 1.0}),
    ("authorization.json", {"harmless_extra_field": "x"}),
])
def test_CONTENTBIND_smoke_stops_on_a_post_audit_mutation(tmp_path, monkeypatch,
                                                          name, updates):
    out = _audited_canary_dir(tmp_path)
    _patch_json(out / name, **updates)
    proceeded, message, fits = _smoke_after(out, monkeypatch)
    assert proceeded is False and fits == 0
    assert name in message and "changed after the independent canary audit" in message
    assert not (out / "smoke_fit_results.csv").exists()


@pytest.mark.parametrize("key,value", [
    ("canary_content_sha256", "0" * 64),
    ("authorization_content_sha256", "0" * 64),
    ("canary_content_sha256", "abc"),
    ("canary_content_sha256", "A" * 64),
    ("canary_content_sha256", "0" * 63 + "z"),
    ("canary_content_sha256", None),
    ("canary_content_sha256", 1),
    ("authorization_content_sha256", True),
])
def test_CONTENTBIND_smoke_stops_on_a_tampered_or_malformed_digest(tmp_path, monkeypatch,
                                                                   key, value):
    out = _audited_canary_dir(tmp_path)
    _patch_json(out / "canary_audit.json", **{key: value})
    proceeded, message, fits = _smoke_after(out, monkeypatch)
    assert proceeded is False and fits == 0
    # either the format check names the key, or the rehash reports the mismatch
    assert key in message or "changed after the independent canary audit" in message, message


@pytest.mark.parametrize("key", ["authorization_content_sha256", "canary_content_sha256"])
def test_CONTENTBIND_smoke_stops_on_a_missing_digest(tmp_path, monkeypatch, key):
    out = _audited_canary_dir(tmp_path)
    payload = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    payload.pop(key)
    (out / "canary_audit.json").write_text(json.dumps(payload), encoding="utf-8")
    proceeded, message, fits = _smoke_after(out, monkeypatch)
    assert proceeded is False and fits == 0
    assert "incomplete" in message and key in message


# --- positive control --------------------------------------------------------


def test_CONTENTBIND_unchanged_artifacts_still_run_six_fake_fits(tmp_path, monkeypatch):
    out = _audited_canary_dir(tmp_path)
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["authorization_content_sha256"] == _sha256_file(out / "authorization.json")
    assert verdict["canary_content_sha256"] == _sha256_file(out / "canary.json")

    proceeded, _message, fits = _smoke_after(out, monkeypatch)
    assert proceeded is True and fits == 6
    assert "em_runner" not in sys.modules


def test_CONTENTBIND_final_audit_positive_control(tmp_path):
    directory = _real_fixture(tmp_path)
    verdict = json.loads((directory / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["authorization_content_sha256"] == \
        _sha256_file(directory / "authorization.json")
    assert verdict["canary_content_sha256"] == _sha256_file(directory / "canary.json")
    auditor, report = _final_audit(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    assert report["status"] == "PASS"


# --- the scientific protocol is untouched ------------------------------------


def test_CONTENTBIND_protocol_hash_is_unchanged():
    config = H.smoke_protocol_config()
    for key in config:
        assert "content_sha256" not in key, key
    assert "sha256" not in _inspect.getsource(H.smoke_protocol_config)
    assert H.smoke_protocol_hash() == A.EXPECTED_SMOKE_PROTOCOL_HASH == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert A.APPROVED_SCIENTIFIC_MAIN_SHA == "68c78e1191889609dead05ea5a9fb11525ce92e2"
    assert (A.SMOKE_PROTOCOL_ISSUE_NUMBER, A.SMOKE_EXECUTION_ISSUE_NUMBER) == (53, 55)
    assert (A.SMOKE_ESTIMAND, A.SMOKE_ROLE) == ("A", "primary")
    assert A.SMOKE_K_TRUE == 1 and A.SMOKE_REPLICATE == 1
    assert A.SMOKE_K_CANDIDATES == (2, 3, 4) and A.SMOKE_STARTS == (1, 2)
    assert (A.EXPECTED_CANARY_FITS, A.EXPECTED_SMOKE_FITS,
            A.EXPECTED_REAL_EM_BUDGET) == (2, 6, 8)
    assert H.EXPECTED_NEW_FITS == 336

    assert H.current_smoke_execution_authorization() is not None
    authorization = H.current_smoke_authorization()
    assert authorization.independent_review_pass is False
    assert authorization.human_smoke_approval is False
    assert authorization.authorized() is False


# --- previously fixed protections have not regressed -------------------------


@pytest.mark.parametrize("name,updates,check", [
    ("canary_audit.json", {"medium_count": 0.0}, "smoke_canary_audit_medium_count"),
    ("canary_audit.json", {"blocker_count": 0.0}, "smoke_canary_audit_counts"),
    ("runinfo.json", {"execution_issue": 55.0}, "smoke_runinfo_execution_issue"),
    ("runinfo.json", {"approved_baseline_is_ancestor": 1}, "smoke_runinfo_lineage"),
    ("smoke_summary.json", {"k_true": 1.0}, "smoke_summary_k_true"),
    ("smoke_summary.json", {"candidate_k": [2.0, 3, 4]}, "smoke_summary_candidates"),
])
def test_CONTENTBIND_previous_protections_hold(tmp_path, name, updates, check):
    directory = _real_fixture(tmp_path)
    _patch_json(directory / name, **updates)
    auditor = A.audit_smoke_run_dir(directory)
    assert any(f.check == check for f in auditor.blockers), \
        (name, sorted({f.check for f in auditor.blockers}))


# ===========================================================================
# PR #56 content-binding rereview MEDIUM 1: one audit-time snapshot
# ===========================================================================
#
# The verdict used to take its digests from the audit but its metadata from a
# second read at report-build time, so a file changed in between produced a
# report that mixed new metadata with an old digest.  Every source-derived field
# now comes from the payload the auditor parsed, out of the same bytes it hashed.


import ast as _ast          # noqa: E402
import textwrap as _textwrap  # noqa: E402


def _audit_snapshot_dir(tmp_path, name="snapshot"):
    """A production-lineage canary artifact pair, audited but not yet reported."""

    out = _canary_audit_fixture(tmp_path / name)
    auditor = A.audit_canary_run_dir(out)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    return out, auditor


def test_SNAPSHOTBIND_auditor_stores_payload_and_digest_from_one_read(tmp_path):
    out, auditor = _audit_snapshot_dir(tmp_path)
    assert set(auditor.audited_payloads) == {"authorization.json", "canary.json"}
    assert set(auditor.audited_digests) == {"authorization.json", "canary.json"}
    for name in ("authorization.json", "canary.json"):
        raw = (out / name).read_bytes()
        assert auditor.audited_digests[name] == _hashlib.sha256(raw).hexdigest()
        assert auditor.audited_payloads[name] == json.loads(raw.decode("utf-8"))


def test_SNAPSHOTBIND_snapshot_is_an_independent_copy(tmp_path):
    out, auditor = _audit_snapshot_dir(tmp_path)
    stored = auditor.audited_payloads["canary.json"]
    fresh, _digest = A._read_json_and_digest(out / "canary.json", A.Auditor())
    assert stored == fresh and stored is not fresh
    assert "copy.deepcopy(" in _inspect.getsource(A.audit_canary_run_dir)

    first = A.build_canary_audit_report(auditor, out)
    second = A.build_canary_audit_report(auditor, out)
    assert first == second, "the verdict must not depend on when it is built"


@pytest.mark.parametrize("key,mutation,report_key", [
    ("status", "FAIL", "canary_status"),
    ("execution_mode", "test_only", "canary_execution_mode"),
    ("real_canary_fits_executed", 0, "actual_canary_fits"),
    ("run_code_sha", "1" * 40, "run_code_sha"),
])
def test_SNAPSHOTBIND_report_keeps_the_audited_values(tmp_path, key, mutation, report_key):
    """§7: mutate the source AFTER the audit, BEFORE the report is built."""

    out, auditor = _audit_snapshot_dir(tmp_path)
    audited_value = json.loads((out / "canary.json").read_text(encoding="utf-8"))[key]
    audited_digest = auditor.audited_digests["canary.json"]
    assert audited_value != mutation

    _patch_json(out / "canary.json", **{key: mutation})
    current_digest = _hashlib.sha256((out / "canary.json").read_bytes()).hexdigest()
    assert current_digest != audited_digest

    report = A.build_canary_audit_report(auditor, out)
    assert report[report_key] == audited_value, "the report followed the current file"
    assert report[report_key] != mutation
    assert report["canary_content_sha256"] == audited_digest
    assert report["canary_content_sha256"] != current_digest


def test_SNAPSHOTBIND_authorization_variant(tmp_path):
    """§8: the authorization snapshot and digest are equally audit-time."""

    out, auditor = _audit_snapshot_dir(tmp_path)
    audited_digest = auditor.audited_digests["authorization.json"]
    audited_payload = dict(auditor.audited_payloads["authorization.json"])

    _patch_json(out / "authorization.json", harmless_extra_field="x")
    current_digest = _hashlib.sha256((out / "authorization.json").read_bytes()).hexdigest()
    assert current_digest != audited_digest

    report = A.build_canary_audit_report(auditor, out)
    assert report["authorization_content_sha256"] == audited_digest
    assert report["authorization_content_sha256"] != current_digest
    assert auditor.audited_payloads["authorization.json"] == audited_payload
    assert "harmless_extra_field" not in auditor.audited_payloads["authorization.json"]


def test_SNAPSHOTBIND_audited_files_reflect_what_was_read(tmp_path):
    out, auditor = _audit_snapshot_dir(tmp_path)
    (out / "canary.json").unlink()               # gone after the audit
    report = A.build_canary_audit_report(auditor, out)
    assert sorted(report["audited_files"]) == ["authorization.json", "canary.json"]


def test_SNAPSHOTBIND_builder_never_rereads_the_source(tmp_path):
    """§10: static proof, alongside the behavioural tests above."""

    tree = _ast.parse(_textwrap.dedent(_inspect.getsource(A.build_canary_audit_report)))
    forbidden_attributes = {"read_text", "read_bytes", "load", "loads", "is_file",
                            "open", "iterdir", "stat"}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Attribute):
            assert node.attr not in forbidden_attributes, node.attr
        if isinstance(node, _ast.Name):
            assert node.id not in {"open", "json"}, node.id
    assert "auditor.audited_payloads" in _inspect.getsource(A.build_canary_audit_report)


def test_SNAPSHOTBIND_stale_snapshot_report_is_rejected_by_the_runner(tmp_path):
    """§9: internally consistent snapshot, and still no stale PASS for the smoke."""

    out, auditor = _audit_snapshot_dir(tmp_path)
    _patch_json(out / "canary.json", warning_count=1)      # changed after the audit
    A.write_canary_audit_report(out, auditor)              # snapshot verdict, PASS
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS"
    assert verdict["canary_content_sha256"] == auditor.audited_digests["canary.json"]

    canary_payload = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    authorization = _production_authorization(approved_main_sha=APPROVED_BASELINE)
    with pytest.raises(HarnessStop) as excinfo:
        H.require_canary_audit_pass(out, authorization, canary_payload,
                                    current_run_code_sha="0" * 40, test_only=False)
    assert "changed after the independent canary audit" in str(excinfo.value)


# ===========================================================================
# PR #56 content-binding rereview MEDIUM 2: no-restamp audit -> runner
# ===========================================================================
#
# Every other integration fixture re-stamps the verdict for the test-only
# lineage.  These two tests use production-lineage artifacts end to end: the
# real auditor writes the verdict, nothing touches it, and the runner gate
# consumes exactly those bytes.  No adapter is built and no EM runs.


def _no_restamp_guard(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("the direct integration test must not restamp digests")

    monkeypatch.setattr(sys.modules[__name__], "_stamp_canary_audit_digests", _forbidden)


def _direct_audited_dir(tmp_path, monkeypatch, name="direct"):
    """audit module -> persisted verdict, with no copy/restamp of any kind."""

    _no_restamp_guard(monkeypatch)
    out = _canary_audit_fixture(tmp_path / name)
    auditor = A.audit_canary_run_dir(out)          # the real independent auditor
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    A.write_canary_audit_report(out, auditor)      # the audit module persists it
    return out


def test_DIRECTBIND_untouched_verdict_passes_the_runner_gate(tmp_path, monkeypatch):
    out = _direct_audited_dir(tmp_path, monkeypatch)
    verdict_bytes = (out / "canary_audit.json").read_bytes()
    verdict = json.loads(verdict_bytes)
    authorization_bytes = (out / "authorization.json").read_bytes()
    canary_bytes = (out / "canary.json").read_bytes()

    # §17: the persisted digests are the digests of these exact files
    assert verdict["status"] == "PASS"
    assert verdict["authorization_content_sha256"] == \
        _hashlib.sha256(authorization_bytes).hexdigest()
    assert verdict["canary_content_sha256"] == _hashlib.sha256(canary_bytes).hexdigest()

    canary_payload = json.loads(canary_bytes.decode("utf-8"))
    authorization = _production_authorization(approved_main_sha=APPROVED_BASELINE)
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    accepted = H.require_canary_audit_pass(out, authorization, canary_payload,
                                           current_run_code_sha="0" * 40, test_only=False)
    assert accepted["status"] == "PASS"

    # §16: nothing was re-stamped, rewritten or rehashed along the way
    assert (out / "canary_audit.json").read_bytes() == verdict_bytes
    assert (out / "authorization.json").read_bytes() == authorization_bytes
    assert (out / "canary.json").read_bytes() == canary_bytes
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    assert not (out / "smoke_fit_results.csv").exists()


def test_DIRECTBIND_uses_the_production_frozen_constants(tmp_path, monkeypatch):
    """§14: the artifacts match production; no constant is patched to fit."""

    out = _direct_audited_dir(tmp_path, monkeypatch)
    authorization = json.loads((out / "authorization.json").read_text(encoding="utf-8"))
    canary = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    verdict = json.loads((out / "canary_audit.json").read_text(encoding="utf-8"))

    assert A.APPROVED_SCIENTIFIC_MAIN_SHA == H.APPROVED_SCIENTIFIC_MAIN_SHA \
        == H.current_expected_smoke_main_sha() == APPROVED_BASELINE
    for payload in (authorization, canary, verdict):
        assert payload["approved_scientific_main_sha"] == APPROVED_BASELINE
        assert payload["protocol_hash"] == H.smoke_protocol_hash() \
            == A.EXPECTED_SMOKE_PROTOCOL_HASH
    assert verdict["execution_issue"] == 55 and verdict["protocol_origin_issue"] == 53
    assert verdict["canary_execution_mode"] == "real"
    assert verdict["actual_canary_fits"] == A.EXPECTED_CANARY_FITS == 2
    # the production authorization gate is untouched by this test
    assert H.current_smoke_execution_authorization() is not None


@pytest.mark.parametrize("name,updates", [
    ("canary.json", {"k_true": 1.0}),
    ("canary.json", {"protocol_origin_issue_number": 53.0}),
    ("authorization.json", {"k_true": 1.0}),
    ("authorization.json", {"harmless_extra_field": "x"}),
])
def test_DIRECTBIND_mutation_after_the_untouched_verdict_stops_the_runner(
        tmp_path, monkeypatch, name, updates):
    """§19: the persistent regression for the original HIGH, with no restamping."""

    out = _direct_audited_dir(tmp_path, monkeypatch)
    verdict_bytes = (out / "canary_audit.json").read_bytes()
    _patch_json(out / name, **updates)

    canary_payload = json.loads((out / "canary.json").read_text(encoding="utf-8"))
    authorization = _production_authorization(approved_main_sha=APPROVED_BASELINE)
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    with pytest.raises(HarnessStop) as excinfo:
        H.require_canary_audit_pass(out, authorization, canary_payload,
                                    current_run_code_sha="0" * 40, test_only=False)
    message = str(excinfo.value)
    assert name in message and "changed after the independent canary audit" in message
    assert (out / "canary_audit.json").read_bytes() == verdict_bytes
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0


# ===========================================================================
# Issue #55 authorization-only stage: the committed execution authorization
# ===========================================================================
#
# The human approval was recorded in Issue #55, so
# ``current_smoke_execution_authorization()`` now returns a committed record
# instead of None.  Its scope is exactly 2 real canary fits and, only after the
# independent canary audit passes, 6 real smoke fits -- 8 real EM fits in total.
# This stage implements and reviews that record; it executes NOTHING.
#
# Because the authorization exists, a test that drives the production CLI would
# otherwise continue past the authorization gate into the real workflow.  Every
# such test installs ``_block_production_execution`` first, so the single
# production workflow is replaced by a stop: the wiring is still asserted, no
# adapter is built, no artifact directory is reserved and no EM runs.


AUTHORIZED_SMOKE_FIELDS = {
    "issue_number": 55,
    "approved_main_sha": "68c78e1191889609dead05ea5a9fb11525ce92e2",
    "protocol_hash": "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09",
    "estimand": "A",
    "k_true": 1,
    "replicate": 1,
    "smoke_fit_count": 6,
    "canary_fit_count": 2,
    "data_seed_base": 61000,
    "model_seed_base": 630000,
    "split_seed": 42001,
    "independent_review_pass": True,
    "human_smoke_approval": True,
    "authorization_version": "phase8b-smoke-authorization-v1",
}


ARCHIVED_SMOKE_ARTIFACTS = frozenset({
    "authorization.json", "canary.json", "canary_audit.json", "runinfo.json",
    "smoke_fit_results.csv", "smoke_summary.json", "audit_report.json",
})


def _assert_no_new_production_artifacts():
    """The archived S2c evidence may exist; nothing new or modified may appear.

    PR #58 committed the 7 frozen artifacts, so asserting that the production
    directory does not exist would now assert the opposite of the intended
    invariant.  What must stay true is that no test creates, modifies or adds a
    production artifact.
    """

    if H.SMOKE_ARTIFACT_DIR.exists():
        assert {p.name for p in H.SMOKE_ARTIFACT_DIR.iterdir()} == ARCHIVED_SMOKE_ARTIFACTS
    status = subprocess.run(["git", "status", "--porcelain", "--", "expfam/results"],
                            capture_output=True, text=True, cwd=ROOT)
    assert status.returncode == 0 and status.stdout.strip() == "", status.stdout


class _RealAdapterForbidden:
    """Constructing the real EM adapter inside the test suite is a hard error."""

    def __init__(self, *args, **kwargs):          # pragma: no cover - must not run
        raise AssertionError(
            "SAFETY NET: the real AuthorizedEMFitAdapter was constructed by the "
            "test suite; the Phase 8b suite executes zero real EM")

    def fit(self, invocation):                    # pragma: no cover - must not run
        raise AssertionError("SAFETY NET: real EM fit was invoked by the test suite")


@pytest.fixture(autouse=True)
def _forbid_the_real_em_adapter(monkeypatch):
    """Last-resort zero-EM net for every test in this module.

    A test that needs to count adapter constructions still installs its own
    tripwire; this only guarantees that nothing can reach real EM by accident
    now that a production full authorization record exists.
    """

    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _RealAdapterForbidden)


def _block_full_production_execution(monkeypatch):
    """ZERO-EM guard for the production FULL workflow.

    The committed authorization is real, so ``--full --allow-em`` legitimately
    clears the authorization gate.  Replacing the single production full
    workflow with a stop keeps the suite at zero real fits while the CLI wiring
    is still asserted.  Returns the list of authorizations it intercepted.
    """

    reached = []

    def _blocked(authorization):
        reached.append(authorization)
        raise HarnessStop(
            "test guard: the real 336-fit full execution is never run by the test suite")

    monkeypatch.setattr(H, "_run_production_full_execution", _blocked)
    return reached


def _block_production_execution(monkeypatch):
    """ZERO-EM guard: a test may reach the production workflow, never run it.

    The authorization is committed, so ``--canary`` / ``--smoke`` legitimately
    clear the authorization gate.  Replacing the single production workflow with
    a stop keeps the whole suite at zero real fits while the CLI wiring is still
    asserted.  Returns the list of (command, authorization) it intercepted.
    """

    reached = []

    def _blocked(authorization, command):
        reached.append((command, authorization))
        raise HarnessStop(
            f"test guard: the real {command} execution is never run by the test suite")

    monkeypatch.setattr(H, "_run_production_execution", _blocked)
    return reached


# --- the committed record ---------------------------------------------------


def test_AUTHORIZATIONONLY_record_is_present_and_exact():
    authorization = H.current_smoke_execution_authorization()
    assert authorization is not None
    assert type(authorization) is H.SmokeExecutionAuthorization
    assert authorization.is_test_only() is False
    for name, expected in AUTHORIZED_SMOKE_FIELDS.items():
        actual = getattr(authorization, name)
        assert actual == expected, name
        assert type(actual) is type(expected), (name, type(actual))
    assert authorization._authority is H._SMOKE_EXECUTION_AUTHORITY
    assert authorization._authority is not H._SMOKE_TEST_AUTHORITY


def test_AUTHORIZATIONONLY_record_validates_against_the_frozen_protocol():
    authorization = H.current_smoke_execution_authorization()
    H.validate_smoke_execution_authorization(authorization, test_only=False)   # no EM
    # the record agrees with the independently frozen implementation constants
    assert authorization.approved_main_sha == H.current_expected_smoke_main_sha()
    assert authorization.protocol_hash == H.smoke_protocol_hash()
    assert authorization.issue_number == H.SMOKE_EXECUTION_ISSUE_NUMBER == 55
    assert authorization.canary_fit_count == H.EXPECTED_CANARY_FITS == 2
    assert authorization.smoke_fit_count == H.EXPECTED_SMOKE_FITS == 6
    assert (authorization.canary_fit_count + authorization.smoke_fit_count
            == H.EXPECTED_REAL_EM_BUDGET == 8)
    assert authorization.authorization_version == H.SMOKE_AUTHORIZATION_VERSION
    assert "em_runner" not in sys.modules


def test_AUTHORIZATIONONLY_record_is_not_valid_for_the_test_only_lineage():
    """The production record must not double as a test-only authorization."""

    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(
            H.current_smoke_execution_authorization(), test_only=True)
    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(_test_authorization(), test_only=False)


# --- literality: committed values, not self-generated ones ------------------


def test_AUTHORIZATIONONLY_record_is_committed_literals():
    """§15: nothing about the record may be derived at runtime."""

    body = _executable_body(H.current_smoke_execution_authorization)
    for forbidden in ("os.", "environ", "getenv", "argv", "sys.", "subprocess",
                      "_git_output", "git", "rev-parse", "Path", "open(",
                      "read_text", "read_bytes", "json", "input(", "config"):
        assert forbidden not in body, forbidden
    # not signed with the very constants it is validated against
    for forbidden in ("smoke_protocol_hash()", "current_expected_smoke_main_sha()",
                      "APPROVED_SCIENTIFIC_MAIN_SHA", "SMOKE_EXECUTION_ISSUE_NUMBER",
                      "SMOKE_K_TRUE", "SMOKE_REPLICATE", "EXPECTED_SMOKE_FITS",
                      "EXPECTED_CANARY_FITS", "SMOKE_DATA_SEED_BASE",
                      "SMOKE_MODEL_SEED_BASE", "SMOKE_SPLIT_SEED",
                      "SMOKE_AUTHORIZATION_VERSION"):
        assert forbidden not in body, forbidden
    assert "_SMOKE_EXECUTION_AUTHORITY" in body
    assert "_SMOKE_TEST_AUTHORITY" not in body
    for literal in ("68c78e1191889609dead05ea5a9fb11525ce92e2",
                    "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09",
                    "phase8b-smoke-authorization-v1"):
        assert literal in body, literal


def test_AUTHORIZATIONONLY_no_cli_or_env_can_fabricate_or_change_it(monkeypatch):
    for name in ("PHASE8B_HUMAN_SMOKE_APPROVAL", "HUMAN_SMOKE_APPROVAL",
                 "INDEPENDENT_REVIEW_PASS", "PHASE8B_SMOKE_AUTHORIZED",
                 "APPROVED_MAIN_SHA", "PHASE8B_SMOKE_FIT_COUNT"):
        monkeypatch.setenv(name, "b" * 40)
    monkeypatch.setattr(sys, "argv", ["run", "--smoke", "--allow-em", "--approve"])
    authorization = H.current_smoke_execution_authorization()
    for name, expected in AUTHORIZED_SMOKE_FIELDS.items():
        assert getattr(authorization, name) == expected, name

    options = {option for action in H._build_parser()._actions
               for option in action.option_strings}
    for forbidden in ("--human-approved", "--reviewed", "--approve", "--approved",
                      "--independent-review-pass", "--human-smoke-approval",
                      "--approved-main-sha", "--smoke-fit-count", "--budget"):
        assert forbidden not in options, forbidden


def test_AUTHORIZATIONONLY_no_public_factory_for_the_production_authority():
    public = [name for name in dir(H) if not name.startswith("_")]
    for name in public:
        assert "SMOKE_EXECUTION_AUTHORITY" not in name, name
    # the only construction sites of the production sentinel
    source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    uses = [line.strip() for line in source.splitlines()
            if "_SMOKE_EXECUTION_AUTHORITY" in line]
    assert len(uses) == 3, uses          # definition, validator selection, the record
    assert "_make_test_smoke_authorization" not in \
        _inspect.getsource(H.current_smoke_execution_authorization)


# --- falsification: every frozen field is load-bearing ----------------------


@pytest.mark.parametrize("field,value", [
    ("approved_main_sha", "0" * 40),
    ("approved_main_sha", "4e89a10cacc855975cd76f891605e3758e6d2835"),
    ("approved_main_sha", "not-a-sha"),
    ("protocol_hash", "f" * 64),
    ("issue_number", 53),
    ("issue_number", 56),
    ("estimand", "B"),
    ("k_true", 3),
    ("replicate", 2),
    ("smoke_fit_count", 12),
    ("smoke_fit_count", 336),
    ("canary_fit_count", 1),
    ("canary_fit_count", 4),
    ("data_seed_base", 51000),
    ("model_seed_base", 530000),
    ("split_seed", 42002),
    ("independent_review_pass", False),
    ("human_smoke_approval", False),
    ("authorization_version", "phase8b-smoke-authorization-v2"),
])
def test_AUTHORIZATIONONLY_mutated_record_is_rejected(field, value):
    mutated = dataclasses.replace(H.current_smoke_execution_authorization(),
                                  **{field: value})
    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(mutated, test_only=False)
    assert "em_runner" not in sys.modules


@pytest.mark.parametrize("authority", ["test", "none", "other"])
def test_AUTHORIZATIONONLY_wrong_authority_is_rejected(authority):
    sentinel = {"test": H._SMOKE_TEST_AUTHORITY, "none": None,
                "other": object()}[authority]
    mutated = dataclasses.replace(H.current_smoke_execution_authorization(),
                                  _authority=sentinel)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_smoke_execution_authorization(mutated, test_only=False)
    assert "provenance" in str(excinfo.value) or "authorization" in str(excinfo.value)


def test_AUTHORIZATIONONLY_a_plain_object_is_rejected():
    for candidate in (None, object(), {"issue_number": 55},
                      dataclasses.asdict(H.current_smoke_execution_authorization())):
        with pytest.raises(HarnessStop):
            H.validate_smoke_execution_authorization(candidate, test_only=False)


# --- full remains impossible ------------------------------------------------


def test_AUTHORIZATIONONLY_full_is_still_unauthorized(monkeypatch):
    full_reached = _block_full_production_execution(monkeypatch)
    reached = _block_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"])
    message = str(excinfo.value)
    assert "not authorized" in message and "never be reused for --full" in message
    assert reached == [], "--full must not reach the SMOKE production workflow"
    assert full_reached == [], "--full must not reach the full production workflow"

    executable = _executable_body(H._require_em_authorization)
    full_branch = executable.split("if command == 'full':")[1].split("_require(command in")[0]
    assert "current_smoke_execution_authorization" not in full_branch
    # Issue #59: --full resolves through its OWN gate.  S3-B bound the reviewed
    # baseline; S3-D withdrew the stale S3-C record, so it is absent again.
    assert H.current_full_execution_authorization() is None
    assert H.current_expected_full_main_sha() == "02ef35add45036975162b6a267f6428c3b380459"
    assert H.current_smoke_execution_authorization().smoke_fit_count == 6
    assert H.EXPECTED_NEW_FITS == 336, "the full budget is a different, unauthorized number"


def test_AUTHORIZATIONONLY_budget_is_exactly_two_plus_six():
    authorization = H.current_smoke_execution_authorization()
    assert authorization.canary_fit_count == 2
    assert authorization.smoke_fit_count == 6
    assert H.EXPECTED_REAL_EM_BUDGET == 8
    contract = H.run_smoke_contract()
    assert contract["expected_canary_fits"] == 2
    assert contract["expected_smoke_fits"] == 6
    assert contract["expected_real_em_budget"] == 8
    assert contract["em_fits_executed"] == 0
    assert contract["real_canary_fits_executed"] == 0
    assert contract["real_smoke_fits_executed"] == 0
    assert contract["execution_authorization_present"] is True
    assert contract["artifact_directory_exists"] == H.SMOKE_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()


# --- this stage executes nothing --------------------------------------------


def test_AUTHORIZATIONONLY_stage_executes_zero_fits(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    reached = _block_production_execution(monkeypatch)
    for command in (["--canary", "--allow-em"], ["--smoke", "--allow-em"]):
        with pytest.raises(HarnessStop):
            H.main(command)
    assert [name for name, _auth in reached] == ["canary", "smoke"]
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    _assert_no_new_production_artifacts()
    assert "em_runner" not in sys.modules


def test_AUTHORIZATIONONLY_production_directory_is_never_written_by_a_test():
    assert H.SMOKE_ARTIFACT_DIR.name == "k_true_robustness_smoke_20260901"
    assert H.run_smoke_contract()["artifact_directory_exists"] == \
        H.SMOKE_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()


def test_AUTHORIZATIONONLY_zero_em_modes_still_import_no_em(tmp_path):
    code = (
        "import sys;"
        "sys.path.insert(0, r'" + str(HERE) + "');"
        "import run_k_true_robustness_sweep as H;"
        "a = H.current_smoke_execution_authorization();"
        "H.validate_smoke_execution_authorization(a, test_only=False);"
        "H.run_validate_only(); H.run_smoke_contract();"
        "print(a is not None, a.is_test_only(), 'em_runner' in sys.modules,"
        " 'model_dual_expfam_fixed' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True False False False"
    _assert_no_new_production_artifacts()


# --- the legacy machine-checkable gate stays a separate concept -------------


def test_AUTHORIZATIONONLY_machine_gate_record_is_unchanged():
    """§9: ``current_smoke_authorization`` is the computed-gate report, not the
    execution authorization, and its human-only gates stay False by construction."""

    gates = H.current_smoke_authorization()
    assert gates.independent_review_pass is False
    assert gates.human_smoke_approval is False
    assert gates.authorized() is False
    assert type(gates) is not H.SmokeExecutionAuthorization
    source = _inspect.getsource(H.current_smoke_authorization)
    assert "current_smoke_execution_authorization" not in source


# ===========================================================================
# Issue #55 authorization-only MEDIUM-01: strict typing on the frozen fields
# ===========================================================================
#
# The validator compared the frozen authorization fields by value alone, so
# Python equality (``1.0 == 1``, ``True == 1``) accepted an equal float or bool
# in place of a frozen integer -- on the record that releases the real 2+6
# execution.  Exact type identity is now required before the value comparison,
# with no coercion (``isinstance`` would admit bool; ``int(...)`` would
# normalise an invalid value into a valid one).


AUTHORIZATION_INTEGER_FIELDS = {
    "issue_number": 55,
    "k_true": 1,
    "replicate": 1,
    "smoke_fit_count": 6,
    "canary_fit_count": 2,
    "data_seed_base": 61000,
    "model_seed_base": 630000,
    "split_seed": 42001,
}


def _authorized_record():
    return H.current_smoke_execution_authorization()


def _rejects(**changes):
    mutated = dataclasses.replace(_authorized_record(), **changes)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_smoke_execution_authorization(mutated, test_only=False)
    return str(excinfo.value)


# --- the rule itself --------------------------------------------------------


def test_AUTHORIZATIONTYPE_validator_requires_exact_type_then_value():
    body = _executable_body(H._validate_smoke_execution_authorization)
    assert "type(actual) is type(expected)" in body
    assert "isinstance(actual" not in body
    for forbidden in ("int(actual)", "float(actual)", "str(actual)", "bool(actual)"):
        assert forbidden not in body, forbidden
    # the value comparison is still there, after the type gate
    assert "actual == expected" in body
    assert body.index("type(actual) is type(expected)") < body.index("actual == expected")


# --- positive control -------------------------------------------------------


def test_AUTHORIZATIONTYPE_production_record_still_validates():
    authorization = _authorized_record()
    assert authorization is not None and authorization.is_test_only() is False
    H.validate_smoke_execution_authorization(authorization, test_only=False)   # no EM
    for name, expected in AUTHORIZATION_INTEGER_FIELDS.items():
        value = getattr(authorization, name)
        assert type(value) is int and value == expected, name
    assert "em_runner" not in sys.modules


# --- §9: an equal float for every frozen integer field ----------------------


@pytest.mark.parametrize("field,expected", sorted(AUTHORIZATION_INTEGER_FIELDS.items()))
def test_AUTHORIZATIONTYPE_equal_float_is_rejected(field, expected):
    substitute = float(expected)
    assert substitute == expected, "the attack must be value-equal"
    message = _rejects(**{field: substitute})
    assert field in message and "is not a int" in message
    _assert_no_new_production_artifacts()


# --- §10: equal bools ------------------------------------------------------


@pytest.mark.parametrize("field", ["k_true", "replicate"])
def test_AUTHORIZATIONTYPE_equal_bool_is_rejected(field):
    assert True == 1 and getattr(_authorized_record(), field) == 1
    message = _rejects(**{field: True})
    assert field in message and "bool" in message


@pytest.mark.parametrize("field", sorted(AUTHORIZATION_INTEGER_FIELDS))
def test_AUTHORIZATIONTYPE_string_and_none_are_rejected(field, ):
    expected = AUTHORIZATION_INTEGER_FIELDS[field]
    for substitute in (str(expected), None, [expected], {expected}):
        message = _rejects(**{field: substitute})
        assert field in message


def test_AUTHORIZATIONTYPE_string_fields_keep_their_type_and_value():
    for field, substitute in (("protocol_hash", 0), ("estimand", 0),
                              ("authorization_version", 1),
                              ("protocol_hash", None), ("estimand", None),
                              ("authorization_version", None)):
        message = _rejects(**{field: substitute})
        assert field in message
    # the frozen values themselves are unchanged
    authorization = _authorized_record()
    assert authorization.protocol_hash == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert authorization.estimand == "A"
    assert authorization.authorization_version == "phase8b-smoke-authorization-v1"


def test_AUTHORIZATIONTYPE_approved_main_sha_check_is_not_weakened():
    for substitute in ("68C78E1191889609DEAD05EA5A9FB11525CE92E2", "68c78e11", 0, None,
                       "4e89a10cacc855975cd76f891605e3758e6d2835"):
        message = _rejects(approved_main_sha=substitute)
        assert "SHA" in message or "approved main SHA" in message
    assert _authorized_record().approved_main_sha == \
        "68c78e1191889609dead05ea5a9fb11525ce92e2"


# --- §10: the human gates keep their literal-bool contract -----------------


def test_AUTHORIZATIONTYPE_human_gates_stay_literal_bools():
    body = _executable_body(H._validate_smoke_execution_authorization)
    assert "authorization.independent_review_pass is True" in body
    assert "authorization.human_smoke_approval is True" in body
    for field in ("independent_review_pass", "human_smoke_approval"):
        for substitute in (1, 1.0, "True", None, False):
            _rejects(**{field: substitute})
    # and the genuine bools still pass
    authorization = _authorized_record()
    assert authorization.independent_review_pass is True
    assert authorization.human_smoke_approval is True
    H.validate_smoke_execution_authorization(authorization, test_only=False)


# --- §11: nothing that used to be rejected has become acceptable -----------


@pytest.mark.parametrize("field,value", [
    ("approved_main_sha", "0" * 40),
    ("protocol_hash", "f" * 64),
    ("issue_number", 53),
    ("estimand", "B"),
    ("k_true", 3),
    ("replicate", 2),
    ("smoke_fit_count", 12),
    ("smoke_fit_count", 336),
    ("canary_fit_count", 1),
    ("data_seed_base", 51000),
    ("model_seed_base", 530000),
    ("split_seed", 42002),
    ("independent_review_pass", False),
    ("human_smoke_approval", False),
    ("authorization_version", "phase8b-smoke-authorization-v2"),
])
def test_AUTHORIZATIONTYPE_value_falsification_still_rejects(field, value):
    _rejects(**{field: value})


def test_AUTHORIZATIONTYPE_authority_and_shape_falsification_still_rejects():
    for sentinel in (H._SMOKE_TEST_AUTHORITY, None, object()):
        _rejects(_authority=sentinel)
    for candidate in (None, object(), {"issue_number": 55},
                      dataclasses.asdict(_authorized_record())):
        with pytest.raises(HarnessStop):
            H.validate_smoke_execution_authorization(candidate, test_only=False)


# --- the fix changes nothing else ------------------------------------------


def test_AUTHORIZATIONTYPE_frozen_expectations_are_untouched():
    assert H.SMOKE_EXECUTION_ISSUE_NUMBER == 55
    assert H.APPROVED_SCIENTIFIC_MAIN_SHA == "68c78e1191889609dead05ea5a9fb11525ce92e2"
    assert H.smoke_protocol_hash() == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert (H.SMOKE_ESTIMAND, H.SMOKE_K_TRUE, H.SMOKE_REPLICATE) == ("A", 1, 1)
    assert (H.EXPECTED_SMOKE_FITS, H.EXPECTED_CANARY_FITS) == (6, 2)
    assert (H.SMOKE_DATA_SEED_BASE, H.SMOKE_MODEL_SEED_BASE, H.SMOKE_SPLIT_SEED) == \
        (61000, 630000, 42001)
    assert H.SMOKE_AUTHORIZATION_VERSION == "phase8b-smoke-authorization-v1"
    assert H.EXPECTED_NEW_FITS == 336
    # the record is still committed literals, not self-generated
    body = _executable_body(H.current_smoke_execution_authorization)
    for forbidden in ("environ", "getenv", "argv", "subprocess", "rev-parse",
                      "smoke_protocol_hash()", "current_expected_smoke_main_sha()",
                      "APPROVED_SCIENTIFIC_MAIN_SHA"):
        assert forbidden not in body, forbidden


def test_AUTHORIZATIONTYPE_stage_still_executes_nothing(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    full_reached = _block_full_production_execution(monkeypatch)
    reached = _block_production_execution(monkeypatch)
    for command in (["--canary", "--allow-em"], ["--smoke", "--allow-em"]):
        with pytest.raises(HarnessStop):
            H.main(command)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"])
    assert "never be reused for --full" in str(excinfo.value)
    assert full_reached == []
    assert [name for name, _auth in reached] == ["canary", "smoke"]
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    _assert_no_new_production_artifacts()
    assert "em_runner" not in sys.modules

    contract = H.run_smoke_contract()
    assert contract["execution_authorization_present"] is True
    assert contract["em_fits_executed"] == 0
    assert contract["real_canary_fits_executed"] == 0
    assert contract["real_smoke_fits_executed"] == 0


# ===========================================================================
# Issue #59 Phase 8b S3-A: the full 336-fit execution gate
# ===========================================================================
#
# The 336-fit sweep is a SECOND human gate, independent of the smoke.  S3-A
# implements the schema, the validator, the zero-EM preflight and the
# independent audit contract; it commits no record and executes no fit.
#
# The load-bearing property is negative: a SmokeExecutionAuthorization -- the
# one record that actually exists and is valid today -- must never authorize
# --full, by type, by sentinel and by validator.


FULL_INTEGER_FIELDS = {
    "issue_number": 59,
    "protocol_origin_issue_number": 49,
    "fits_per_estimand": 168,
    "total_fit_count": 336,
    "data_seed_base": 51000,
    "model_seed_base": 530000,
    "anchor_split_seed_base": 42000,
}


def _full_authorization(**overrides):
    return H._make_test_full_authorization(**overrides)


def _full_rejects(**changes):
    mutated = dataclasses.replace(_full_authorization(), **changes)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_full_execution_authorization(mutated, test_only=True)
    return str(excinfo.value)


# --- the gate is absent, and separate from the smoke gate -------------------


def test_FULLGATE_the_full_authorization_record_is_absent():
    """S3-D withdrew the stale S3-C record; the reviewed baseline stays bound."""

    assert H.current_full_execution_authorization() is None
    assert H.current_expected_full_main_sha() == "02ef35add45036975162b6a267f6428c3b380459"
    assert H.trusted_full_main_sha_for(test_only=False) == "02ef35add45036975162b6a267f6428c3b380459"
    # the smoke gate is present; that must not leak into the full gate
    assert H.current_smoke_execution_authorization() is not None
    assert H.current_expected_smoke_main_sha() is not None
    assert H.current_expected_full_main_sha() != H.current_expected_smoke_main_sha()


def test_FULLGATE_smoke_authorization_can_never_authorize_full():
    """The headline requirement of Issue #59."""

    smoke = H.current_smoke_execution_authorization()
    assert type(smoke) is H.SmokeExecutionAuthorization
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_full_execution_authorization(smoke, test_only=False)
    assert "FullExecutionAuthorization" in str(excinfo.value)
    with pytest.raises(HarnessStop):
        H.validate_full_execution_authorization(smoke, test_only=True)
    # and the reverse: a full record cannot stand in for a smoke authorization
    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(_full_authorization(), test_only=True)
    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(_full_authorization(), test_only=False)


def test_FULLGATE_sentinels_and_baselines_are_distinct():
    assert H._FULL_EXECUTION_AUTHORITY is not H._SMOKE_EXECUTION_AUTHORITY
    assert H._FULL_TEST_AUTHORITY is not H._SMOKE_TEST_AUTHORITY
    assert H._FULL_EXECUTION_AUTHORITY is not H._FULL_TEST_AUTHORITY
    assert H._FULL_TEST_EXPECTED_MAIN_SHA != H._TEST_EXPECTED_MAIN_SHA
    # no public factory hands out the production sentinel
    for name in dir(H):
        if not name.startswith("_"):
            assert "FULL_EXECUTION_AUTHORITY" not in name.upper() or name.startswith("_")
    source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    uses = [line.strip() for line in source.splitlines()
            if "_FULL_EXECUTION_AUTHORITY" in line]
    # definition + validator selection only: no committed production record
    assert len(uses) == 2, uses


def test_FULLGATE_cli_full_is_refused_and_never_reaches_a_fit(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    full_reached = _block_full_production_execution(monkeypatch)
    reached = _block_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"])
    message = str(excinfo.value)
    assert "not authorized" in message
    assert "never be reused for --full" in message
    assert "FullExecutionAuthorization" in message
    assert reached == [], "--full never touches the smoke workflow"
    assert full_reached == [], "--full never reaches its own production workflow"
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    _assert_no_new_production_artifacts()


def test_FULLGATE_full_refuses_an_out_dir(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB",
                "--out-dir", "attacker"])
    assert "--out-dir is not accepted" in str(excinfo.value)
    assert not pathlib.Path("attacker").exists()
    assert _AdapterTripwire.constructions == 0


def test_FULLGATE_no_cli_or_env_can_fabricate_a_full_authorization(monkeypatch):
    for name in ("PHASE8B_HUMAN_FULL_APPROVAL", "HUMAN_FULL_APPROVAL",
                 "INDEPENDENT_REVIEW_PASS", "PHASE8B_FULL_AUTHORIZED"):
        monkeypatch.setenv(name, "1")
    options = {option for action in H._build_parser()._actions
               for option in action.option_strings}
    for forbidden in ("--human-full-approved", "--full-approved", "--authorize-full",
                      "--approve-full", "--full-fit-count"):
        assert forbidden not in options, forbidden
    assert H.current_full_execution_authorization() is None
    body = _executable_body(H.current_full_execution_authorization)
    assert body.strip() == "return None"


# --- validator: strict types, exact grid, literal human gates ---------------


def test_FULLGATE_valid_test_only_record_passes():
    H.validate_full_execution_authorization(_full_authorization(), test_only=True)
    with pytest.raises(HarnessStop):
        H.validate_full_execution_authorization(_full_authorization(), test_only=False)


@pytest.mark.parametrize("field,expected", sorted(FULL_INTEGER_FIELDS.items()))
def test_FULLGATE_equal_float_and_bool_are_rejected(field, expected):
    message = _full_rejects(**{field: float(expected)})
    assert field in message and "is not a int" in message
    if expected in (0, 1):
        assert field in _full_rejects(**{field: bool(expected)})


@pytest.mark.parametrize("field,value", [
    ("issue_number", 55),
    ("protocol_origin_issue_number", 53),
    ("protocol_hash", "f" * 64),
    ("estimands", ("A",)),
    ("estimands", ("A", "B", "C")),
    ("k_true_grid", (1, 2, 3, 4, 5)),
    ("k_true_grid", (1, 2, 4)),
    ("candidate_k", (1, 2, 3, 4, 5, 6)),
    ("starts", (1,)),
    ("replicates", (1, 2)),
    ("fits_per_estimand", 84),
    ("total_fit_count", 168),
    ("total_fit_count", 378),
    ("data_seed_base", 61000),
    ("model_seed_base", 630000),
    ("anchor_split_seed_base", 52000),
    ("mask_design", "S_A"),
    ("mask_design", "S_B"),
    ("random_design", "INDEPENDENT"),
    ("hierarchy", "H3_B"),
    ("independent_review_pass", False),
    ("human_full_approval", False),
    ("authorization_version", "phase8b-full-authorization-v2"),
    ("approved_main_sha", "0" * 40),
    ("approved_main_sha", "not-a-sha"),
])
def test_FULLGATE_frozen_field_mutation_is_rejected(field, value):
    _full_rejects(**{field: value})


def test_FULLGATE_anchor_k_true_can_never_enter_the_grid():
    message = _full_rejects(k_true_grid=(1, 2, 3, 4, 5), fits_per_estimand=210,
                            total_fit_count=420)
    assert "anchor" in message or "grid" in message or "k_true_grid" in message
    # even a grid that multiplies correctly must not contain the anchor
    assert H.ANCHOR_K_TRUE == 3 and 3 not in H.NEW_K_TRUE


def test_FULLGATE_grid_must_multiply_to_the_declared_budget():
    message = _full_rejects(fits_per_estimand=336)
    assert "multiplies" in message or "fits_per_estimand" in message
    message = _full_rejects(candidate_k=(1, 2, 3, 4, 5, 6, 7, 8))
    assert "candidate_k" in message


@pytest.mark.parametrize("authority", ["smoke_execution", "smoke_test", "none", "other"])
def test_FULLGATE_wrong_authority_is_rejected(authority):
    sentinel = {"smoke_execution": H._SMOKE_EXECUTION_AUTHORITY,
                "smoke_test": H._SMOKE_TEST_AUTHORITY,
                "none": None, "other": object()}[authority]
    message = _full_rejects(_authority=sentinel)
    assert "provenance" in message


def test_FULLGATE_human_gates_are_literal_bools():
    body = _inspect.getsource(H._validate_full_execution_authorization)
    assert "authorization.independent_review_pass is True" in body
    assert "authorization.human_full_approval is True" in body
    assert "== True" not in body
    for field in ("independent_review_pass", "human_full_approval"):
        for value in (1, 1.0, "True", None, False):
            _full_rejects(**{field: value})


def test_FULLGATE_validator_checks_type_before_value():
    body = _inspect.getsource(H._validate_full_execution_authorization)
    assert body.index("type(actual) is type(expected)") < body.index("actual == expected")
    assert "isinstance(actual" not in body
    for forbidden in ("int(actual)", "float(actual)", "str(actual)", "bool(actual)"):
        assert forbidden not in body


# --- zero-EM preflight: 336, 168/168, anchors, seeds ------------------------


def test_FULLGATE_preflight_is_zero_em_and_exact():
    report = H.run_full_preflight()
    assert report["em_fits_executed"] == 0
    assert report["real_full_fits_executed"] == 0
    assert report["expected_full_fits"] == 336
    assert report["expected_full_fits_per_estimand"] == 168
    assert report["manifest"]["total_fits"] == 336
    assert report["manifest"]["fits_per_estimand"] == {"A": 168, "B": 168}
    assert report["anchor_agreement"]["mismatches"] == []
    assert report["anchor_agreement"]["cells_checked"] == 12
    assert report["anchor_agreement"]["phase7e_rerun_fits"] == 0
    assert report["seed_collisions"]["unintended_collisions"] == []
    assert report["mask_gate_failed"] == []
    assert report["hierarchy"] == "H3_A" and report["mask_design"] == "S_C"
    assert report["random_design"] == "CRN" and report["estimands"] == ["A", "B"]
    assert report["full_execution_authorization_present"] is False
    assert report["trusted_full_main_sha_present"] is True
    assert report["phase7e_rerun_fits"] == 0
    assert "em_runner" not in sys.modules
    _assert_no_new_production_artifacts()


def test_FULLGATE_preflight_cli_mode_is_zero_em(capsys):
    assert H.main(["--full-preflight"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "full-preflight"
    assert payload["em_fits_executed"] == 0
    assert payload["manifest"]["total_fits"] == 336
    assert payload["artifact_directory_exists"] is False
    assert "em_runner" not in sys.modules


def test_FULLGATE_manifest_is_exactly_336_over_two_estimands():
    manifests = H.build_full_manifests()
    assert sorted(manifests) == ["A", "B"]
    report = H.validate_full_manifests(manifests)
    assert report == {"fits_per_estimand": {"A": 168, "B": 168}, "total_fits": 336,
                      "global_fit_index_range": [1, 336],
                      "fit_index_range_by_estimand": {"A": [1, 168], "B": [169, 336]}}
    keys = {(e, r.k_true, r.replicate, r.k, r.start)
            for e, rows in manifests.items() for r in rows}
    assert len(keys) == 336
    assert {k[1] for k in keys} == {1, 2, 4, 5}
    assert 3 not in {k[1] for k in keys}, "the Phase 7e anchor must not be re-executed"
    assert {k[3] for k in keys} == set(range(1, 8))
    assert {k[4] for k in keys} == {1, 2}


@pytest.mark.parametrize("mutation", ["drop_row", "drop_estimand", "duplicate_row",
                                      "anchor_row", "extra_row"])
def test_FULLGATE_manifest_validation_is_fail_closed(mutation):
    manifests = {e: list(rows) for e, rows in H.build_full_manifests().items()}
    if mutation == "drop_row":
        manifests["A"] = manifests["A"][:-1]
    elif mutation == "drop_estimand":
        manifests.pop("B")
    elif mutation == "duplicate_row":
        manifests["A"] = manifests["A"][:-1] + [manifests["A"][0]]
    elif mutation == "anchor_row":
        manifests["A"] = manifests["A"][:-1] + [
            dataclasses.replace(manifests["A"][-1], k_true=H.ANCHOR_K_TRUE)]
    elif mutation == "extra_row":
        manifests["A"] = manifests["A"] + [manifests["A"][0]]
    with pytest.raises(HarnessStop):
        H.validate_full_manifests(manifests)


def test_FULLGATE_anchor_agreement_is_checked_without_em():
    report = H.check_full_anchor_agreement()
    assert report["mismatches"] == []
    assert report["cells_checked"] == len(H.NEW_K_TRUE) * len(H.REPLICATES) == 12
    assert report["mask_design"] == "S_C"
    assert report["phase7e_rerun_fits"] == 0
    assert "em_runner" not in sys.modules
    body = _inspect.getsource(H.check_full_anchor_agreement)
    for forbidden in ("AuthorizedEMFitAdapter", "_run_real_", "adapter"):
        assert forbidden not in body


def test_FULLGATE_anchor_agreement_fails_closed_on_a_wrong_anchor():
    anchors = H.read_phase7e_anchor_masks()
    tampered = dict(anchors)
    tampered[1] = dataclasses.replace(anchors[1], test_mask_hash="0" * 64)
    with pytest.raises(HarnessStop) as excinfo:
        H.check_full_anchor_agreement(tampered)
    assert "anchor agreement failed" in str(excinfo.value)
    with pytest.raises(HarnessStop):
        H.check_full_anchor_agreement({1: anchors[1]})


def test_FULLGATE_seed_collision_check_covers_the_full_sweep():
    report = H.check_seed_collisions(H.build_full_manifests())
    assert report["unintended_collisions"] == []
    assert report["model_seeds_per_estimand"] == {"A": 168, "B": 168}
    assert report["model_seed_distinct"] == 168        # CRN: shared across estimands
    assert report["cross_estimand_sharing_is_intentional"] is True
    assert report["phase7e_split_seed_reused"] == [42001, 42002, 42003]
    assert report["intentional_seed_reuse"] is True
    # the full seed space must not touch the smoke block
    smoke = H.smoke_seed_space()
    full = H.phase8_full_seed_space()
    for role in ("data", "model"):
        assert not (smoke[role] & full[role]), role


# --- independent full audit -------------------------------------------------


def test_FULLGATE_audit_restates_the_full_contract_independently():
    assert A.EXPECTED_FULL_PROTOCOL_HASH == H.full_protocol_hash()
    assert A.EXPECTED_FULL_FITS == 336 and A.EXPECTED_FULL_FITS_PER_ESTIMAND == 168
    assert len(A.expected_full_keys()) == 336
    assert A.FULL_FIT_RESULTS_COLUMNS == H.FULL_FIT_RESULTS_COLUMNS
    assert set(A.FULL_AUDIT_INPUT_FILES) == set(H.FULL_AUDIT_INPUT_FILES)
    source = _inspect.getsource(A)
    assert "import run_k_true_robustness_sweep" not in source


def test_FULLGATE_audit_of_a_missing_directory_is_fail_closed(tmp_path):
    auditor = A.audit_full_run_dir(tmp_path / "nope")
    assert any(f.check == "full_run_dir_missing" for f in auditor.blockers)
    report = A.build_full_audit_report(auditor, tmp_path / "nope")
    assert report["status"] == "FAIL" and report["blocker_count"] > 0


def test_FULLGATE_audit_of_an_empty_directory_requires_every_artifact(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    auditor = A.audit_full_run_dir(directory)
    missing = [f.detail for f in auditor.blockers if f.check == "required_artifact_missing"]
    assert len(missing) == len(A.FULL_AUDIT_INPUT_FILES)
    report = A.build_full_audit_report(auditor, directory)
    assert report["status"] == "FAIL"
    assert report["audit_version"] == "phase8b-full-audit-v1"
    assert report["expected_full_fits"] == 336


def test_FULLGATE_audit_report_is_never_overwritten(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    auditor = A.audit_full_run_dir(directory)
    A.write_full_audit_report(directory, auditor)
    assert (directory / "audit_report.json").is_file()
    with pytest.raises(FileExistsError):
        A.write_full_audit_report(directory, auditor)


def test_FULLGATE_audit_rejects_a_wrong_authorization_payload(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    payload = {
        "artifact_version": "phase8b-full-artifact-v1",
        "authorization_version": "phase8b-full-authorization-v1",
        "protocol_hash": A.EXPECTED_FULL_PROTOCOL_HASH,
        "execution_issue_number": 55,             # smoke issue: wrong
        "protocol_origin_issue_number": 49,
        "fits_per_estimand": 168.0,               # equal float: wrong type
        "total_fit_count": 336,
        "data_seed_base": 51000, "model_seed_base": 530000,
        "anchor_split_seed_base": 42000,
        "k_true_grid": [1, 2, 3, 4, 5],           # contains the anchor
        "candidate_k": [1, 2, 3, 4, 5, 6, 7],
        "starts": [1, 2], "replicates": [1, 2, 3],
        "estimands": ["A"],                       # not A+B
        "mask_design": "S_C", "random_design": "CRN",
        "independent_review_pass": True, "human_full_approval": True,
        # role 1 wrong, and role 2 == role 3
        "scientific_baseline_sha": "a" * 40,
        "reviewed_full_execution_main_sha": "a" * 40,
        "run_code_sha": "a" * 40,
    }
    auditor = A.Auditor()
    A.audit_full_authorization(payload, auditor)
    checks = {f.check for f in auditor.blockers}
    for expected in ("full_auth_execution_issue", "full_auth_fits_per_estimand",
                     "full_auth_k_true_grid", "full_auth_anchor_excluded",
                     "full_auth_estimands", "full_auth_baseline_not_run_sha",
                     "full_auth_scientific_baseline"):
        assert expected in checks, (expected, sorted(checks))


def test_FULLGATE_audit_fit_row_contract(tmp_path):
    auditor = A.Auditor()
    A.audit_full_fit_rows([], auditor)
    assert any(f.check == "full_fit_columns" for f in auditor.blockers)

    header = {c: "" for c in A.FULL_FIT_RESULTS_COLUMNS}
    auditor = A.Auditor()
    A.audit_full_fit_rows([dict(header)], auditor)
    assert any(f.check == "full_fit_row_count" for f in auditor.blockers)


def test_FULLGATE_audit_key_set_is_the_frozen_grid():
    keys = A.expected_full_keys()
    assert len(keys) == 336
    assert {k[0] for k in keys} == {"A", "B"}
    assert {k[1] for k in keys} == {1, 2, 4, 5}
    assert 3 not in {k[1] for k in keys}
    per_estimand = {e: sum(1 for k in keys if k[0] == e) for e in ("A", "B")}
    assert per_estimand == {"A": 168, "B": 168}


def test_FULLGATE_audit_cli_exposes_full_mode(tmp_path, capsys):
    directory = tmp_path / "run"
    directory.mkdir()
    assert A.main(["--run-dir", str(directory), "--mode", "full"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "full" and report["verdict"] == "FAIL"


# --- nothing in this stage runs EM ------------------------------------------


def test_FULLGATE_stage_executes_zero_real_em(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    H.run_full_preflight()
    H.build_full_manifests()
    H.check_full_anchor_agreement()
    with pytest.raises(HarnessStop):
        H.validate_full_execution_authorization(
            H.current_full_execution_authorization(), test_only=False)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    assert not H.FULL_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()


def test_FULLGATE_full_artifact_directory_is_frozen_and_absent():
    assert H.FULL_ARTIFACT_DIRNAME == "k_true_robustness_full_20260902"
    assert H.FULL_ARTIFACT_DIR == (
        ROOT / "expfam" / "results" / "k_selection" / H.FULL_ARTIFACT_DIRNAME)
    assert not H.FULL_ARTIFACT_DIR.exists()
    assert H.FULL_ARTIFACT_DIR != H.SMOKE_ARTIFACT_DIR
    assert "audit_report.json" not in H.FULL_AUDIT_INPUT_FILES
    assert set(H.FULL_ARTIFACT_FILES) - set(H.FULL_AUDIT_INPUT_FILES) == {"audit_report.json"}


def test_FULLGATE_frozen_design_is_unchanged():
    assert H.ESTIMANDS == "AB"                    # H1
    assert H.RANDOM_DESIGN == "CRN"               # H2
    assert H.HIERARCHY == "H3_A"                  # H3
    assert H.MASK_DESIGN == "S_C"                 # H4
    assert H.NEW_K_TRUE == (1, 2, 4, 5) and H.ANCHOR_K_TRUE == 3
    assert H.K_CANDIDATES == (1, 2, 3, 4, 5, 6, 7)
    assert H.REPLICATES == (1, 2, 3) and H.START_LABELS == (1, 2)
    assert H.FITS_PER_ESTIMAND == 168 and H.EXPECTED_NEW_FITS == 336
    # the smoke protocol is untouched by the full gate
    assert H.smoke_protocol_hash() == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"
    assert H.APPROVED_SCIENTIFIC_MAIN_SHA == "68c78e1191889609dead05ea5a9fb11525ce92e2"
    assert H.full_protocol_hash() != H.smoke_protocol_hash()


# ===========================================================================
# Issue #59 Phase 8b S3-A: the production full executor (336 fits)
# ===========================================================================
#
# Exercised with the fake adapter only.  Real EM stays at 0: the production
# authorization and the production reviewed baseline are both absent, so
# ``--full --allow-em`` cannot run, and every test below uses the test-only
# lineage with a temp directory.


class _DirtyAtFitRecorder(_FakeFitRecorder):
    """A fake adapter that returns ONE dirty fit at a chosen call index."""

    def __init__(self, fail_at, mode="retry"):
        super().__init__()
        self.fail_at = fail_at
        self.mode = mode

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        if self.calls != self.fail_at:
            return result
        if self.mode == "raise":
            raise RuntimeError(f"injected adapter failure at fit {self.calls}")
        overrides = {"retry": {"internal_retry": 1},
                     "warning": {"warnings": ("injected",)},
                     "q_failure": {"q_failure": True},
                     "nan": {"nan_occurred": True}}[self.mode]
        return dataclasses.replace(result, **overrides)


def _full_test_authorization(**overrides):
    return H._make_test_full_authorization(**overrides)


def _run_full_fake(out_dir, recorder=None, run_code_sha="0" * 40):
    recorder = recorder or _FakeFitRecorder()
    H._execute_real_full_test_only(_full_test_authorization(), out_dir,
                                   adapter=_test_adapter(recorder),
                                   run_code_sha=run_code_sha)
    return recorder


def _promote_full_fixture(source, destination,
                          reviewed_sha="02ef35add45036975162b6a267f6428c3b380459"):
    """The same artifacts stamped as a real, complete execution.

    The test-only lineage deliberately fabricates role 2 and role 3 so a
    test-only record can never satisfy the production validator.  A REAL
    artifact set, however, is produced against the committed reviewed
    full-execution baseline, and the independent auditor holds that literal
    (Issue #59 S3-E).  So the promotion stamps role 2 to the reviewed
    baseline exactly as it stamps the git ancestry conclusions below: the
    fixture is made to look like the real execution it stands in for.
    Production trust boundaries are untouched -- nothing here changes which
    SHA the runner trusts, and the test-only authorization still carries
    "c" * 40.
    """

    _shutil.copytree(source, destination)
    path = destination / "runinfo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({"actual_full_fits": 336, "working_tree_clean": True,
                    "working_tree_clean_before_execution": True,
                    "approved_baseline_is_ancestor": True,
                    # the test lineage uses fabricated SHAs, so the ancestry
                    # conclusions a real run computes with git are stamped here
                    "scientific_baseline_is_ancestor_of_reviewed_full": True,
                    "reviewed_full_baseline_is_ancestor_of_run_code": True})
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    for name in ("authorization.json", "runinfo.json", "full_summary.json"):
        target = destination / name
        if not target.is_file():
            continue
        stamped = json.loads(target.read_text(encoding="utf-8"))
        if "reviewed_full_execution_main_sha" in stamped:
            stamped["reviewed_full_execution_main_sha"] = reviewed_sha
            target.write_text(json.dumps(stamped, sort_keys=True, indent=2),
                              encoding="utf-8")
    csv_path = destination / "full_fit_results.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    index = header.index("real_full_fits_executed")
    reviewed_index = header.index("reviewed_full_execution_main_sha")
    fixed = [lines[0]]
    for line in lines[1:]:
        cells = line.split(",")
        cells[index] = "336"
        cells[reviewed_index] = reviewed_sha
        fixed.append(",".join(cells))
    csv_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    return destination


# --- exactly 336, in the frozen order ---------------------------------------


def test_FULLEXEC_executes_exactly_336_fake_fits(tmp_path):
    recorder = _run_full_fake(tmp_path / "run")
    assert recorder.calls == 336, "the sweep must call the adapter exactly 336 times"
    assert len(recorder.seeds) == 336 and len(set(recorder.seeds)) == 168  # CRN across A/B
    assert "em_runner" not in sys.modules


def test_FULLEXEC_order_is_deterministic_and_frozen(tmp_path):
    recorder = _run_full_fake(tmp_path / "run")
    expected_seeds = [
        H.expected_model_seed(k_true, replicate, k, start, estimand)
        for estimand in ("A", "B")
        for k_true in (1, 2, 4, 5)
        for replicate in (1, 2, 3)
        for k in range(1, 8)
        for start in (1, 2)
    ]
    assert recorder.seeds == expected_seeds
    assert recorder.k_values == [k for _ in range(24) for k in range(1, 8) for _ in (1, 2)]
    # a second identical run reproduces the same order exactly
    again = _run_full_fake(tmp_path / "run2")
    assert again.seeds == recorder.seeds


def test_FULLEXEC_execution_order_helper_matches_the_grid():
    order = H.full_execution_order()
    assert len(order) == 24
    assert order[0] == ("A", 1, 1) and order[-1] == ("B", 5, 3)
    assert [c[0] for c in order] == ["A"] * 12 + ["B"] * 12
    assert {c[1] for c in order} == {1, 2, 4, 5}
    assert 3 not in {c[1] for c in order}


def test_FULLEXEC_never_makes_a_337th_call(tmp_path):
    recorder = _run_full_fake(tmp_path / "run")
    assert recorder.calls == 336
    # the run is over; nothing may call the adapter again
    before = recorder.calls
    with pytest.raises(HarnessStop):
        H._execute_real_full_test_only(_full_test_authorization(), tmp_path / "run",
                                       adapter=_test_adapter(recorder),
                                       run_code_sha="0" * 40)
    assert recorder.calls == before, "a re-run must not add a single fit"


def test_FULLEXEC_k_true_3_is_never_fitted(tmp_path):
    recorder = _run_full_fake(tmp_path / "run")
    rows = list(csv.DictReader((tmp_path / "run" / "full_fit_results.csv").open(encoding="utf-8")))
    assert len(rows) == 336
    assert {int(r["K_TRUE"]) for r in rows} == {1, 2, 4, 5}
    assert all(int(r["K_TRUE"]) != 3 for r in rows)
    # the anchor seed block was never touched by any fit
    anchor_seeds = {530000 + 1000 * 3 + 10 * k + s for k in range(1, 8) for s in (1, 2)}
    assert not (set(recorder.seeds) & anchor_seeds)


# --- partial failure policy --------------------------------------------------


@pytest.mark.parametrize("mode", ["retry", "warning", "q_failure", "nan", "raise"])
def test_FULLEXEC_middle_failure_stops_immediately(tmp_path, mode):
    recorder = _DirtyAtFitRecorder(fail_at=100, mode=mode)
    out = tmp_path / "run"
    with pytest.raises((HarnessStop, RuntimeError)):
        H._execute_real_full_test_only(_full_test_authorization(), out,
                                       adapter=_test_adapter(recorder),
                                       run_code_sha="0" * 40)
    # the failing fit is the last one attempted: no replacement, no retry
    assert recorder.calls == 100, f"{recorder.calls} fits ran; the sweep must stop at 100"
    assert (out / "failure.json").is_file()
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert failure["expected_full_fits"] == 336
    assert failure["replacement_fits_executed"] == 0 and failure["retry_count"] == 0
    assert "no_replacement_fit" in failure["policy"]
    assert "rerun_requires_a_new_human_gate" in failure["policy"]


def test_FULLEXEC_partial_run_produces_no_summary_and_no_audit_pass(tmp_path):
    out = tmp_path / "run"
    recorder = _DirtyAtFitRecorder(fail_at=17, mode="retry")
    with pytest.raises(HarnessStop):
        H._execute_real_full_test_only(_full_test_authorization(), out,
                                       adapter=_test_adapter(recorder),
                                       run_code_sha="0" * 40)
    # partial evidence is preserved ...
    assert (out / "full_fit_results.csv").is_file()
    partial = list(csv.DictReader((out / "full_fit_results.csv").open(encoding="utf-8")))
    assert len(partial) == 14, "the completed cell's rows are kept"
    # ... and the completed artifacts do NOT exist
    for absent in ("full_summary.json", "selection_matrix.csv", "runinfo.json"):
        assert not (out / absent).exists(), absent
    auditor = A.audit_full_run_dir(out)
    report = A.build_full_audit_report(auditor, out)
    assert report["status"] == "FAIL" and report["blocker_count"] > 0
    assert any(f.check == "full_partial_execution" for f in auditor.blockers)


def test_FULLEXEC_same_authorization_cannot_rerun_after_a_failure(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(HarnessStop):
        H._execute_real_full_test_only(_full_test_authorization(), out,
                                       adapter=_test_adapter(_DirtyAtFitRecorder(5)),
                                       run_code_sha="0" * 40)
    second = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_full_test_only(_full_test_authorization(), out,
                                       adapter=_test_adapter(second),
                                       run_code_sha="0" * 40)
    assert "already exists" in str(excinfo.value)
    assert second.calls == 0, "a re-run must not execute a single replacement fit"
    assert (out / "failure.json").is_file(), "partial evidence must not be deleted"


def test_FULLEXEC_failure_policy_is_declared():
    assert H.FULL_PARTIAL_FAILURE_POLICY == (
        "stop_immediately", "no_replacement_fit", "no_retry", "no_seed_rescue",
        "no_tolerance_change", "preserve_partial_evidence", "no_completed_summary",
        "no_audit_pass", "rerun_requires_a_new_human_gate")
    body = _executable_body(H._run_full_cell)
    for forbidden in ("except", "retry", "replacement", "fallback"):
        assert forbidden not in body.lower().replace("internal_retry", ""), forbidden


# --- authorization boundary at the executor ---------------------------------


def test_FULLEXEC_smoke_authorization_cannot_execute_full(tmp_path):
    recorder = _FakeFitRecorder()
    for authorization in (H.current_smoke_execution_authorization(), _test_authorization()):
        with pytest.raises(HarnessStop) as excinfo:
            H._execute_real_full_test_only(authorization, tmp_path / "run",
                                           adapter=_test_adapter(recorder),
                                           run_code_sha="0" * 40)
        assert "FullExecutionAuthorization" in str(excinfo.value)
    assert recorder.calls == 0
    assert not (tmp_path / "run").exists()


def test_FULLEXEC_absent_authorization_constructs_no_adapter(monkeypatch, tmp_path):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    monkeypatch.setattr(H, "FULL_ARTIFACT_DIR", tmp_path / "frozen_full")
    _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop):
        H.run_real_full(H.current_full_execution_authorization())
    with pytest.raises(HarnessStop):
        H._run_production_full_execution(_full_test_authorization())
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert not (tmp_path / "frozen_full").exists()
    assert "em_runner" not in sys.modules


def test_FULLEXEC_production_path_never_takes_a_prebuilt_adapter():
    for function in (H.run_real_full, H.run_real_full_cli, H._run_production_full_execution):
        parameters = set(_inspect.signature(function).parameters)
        assert not (parameters & {"adapter", "test_adapter", "out_dir", "test_only",
                                  "run_code_sha"}), function.__name__
    body = _executable_body(H._run_production_full_execution)
    assert "FULL_ARTIFACT_DIR" in body and "test_only=False" in body
    for forbidden in ("_execute_real_full_test_only", "_make_test_full_authorization",
                      "_FULL_TEST_AUTHORITY", "_TestAuthorizedFitAdapter"):
        assert forbidden not in _inspect.getsource(H._run_production_full_execution)


# --- the integrated selection matrix ----------------------------------------


def test_FULLEXEC_selection_matrix_is_30_logical_rows(tmp_path):
    _run_full_fake(tmp_path / "run")
    rows = list(csv.DictReader((tmp_path / "run" / "selection_matrix.csv").open(encoding="utf-8")))
    assert len(rows) == 30
    assert tuple(rows[0]) == A.SELECTION_MATRIX_COLUMNS
    keys = {(r["estimand"], int(r["K_TRUE"]), int(r["replicate"])) for r in rows}
    assert keys == {(e, kt, rp) for e in ("A", "B") for kt in (1, 2, 3, 4, 5)
                    for rp in (1, 2, 3)}
    lineage = {}
    for row in rows:
        lineage[row["lineage"]] = lineage.get(row["lineage"], 0) + 1
    assert lineage == {"phase8a_new": 24, "phase7e_anchor": 6}


def test_FULLEXEC_anchor_rows_reference_phase7e_and_are_not_re_executed(tmp_path):
    recorder = _run_full_fake(tmp_path / "run")
    rows = list(csv.DictReader((tmp_path / "run" / "selection_matrix.csv").open(encoding="utf-8")))
    anchor_rows = [r for r in rows if int(r["K_TRUE"]) == 3]
    assert len(anchor_rows) == 6
    for row in anchor_rows:
        assert row["lineage"] == "phase7e_anchor"
        assert row["run_code_sha"] == H.PHASE7E_RUN_CODE_SHA
        assert row["artifact_dir"] == H.PHASE7E_ARTIFACT_DIR
    # the anchor selections come from the Phase 7e artifact, not from this run
    anchor_source = list(csv.DictReader(
        (H.PHASE7E_DIR / "replicate_selection.csv").open(encoding="utf-8")))
    expected = {int(r["replicate"]): int(r["selected_k"]) for r in anchor_source}
    for row in anchor_rows:
        assert int(row["selected_k"]) == expected[int(row["replicate"])]
    # and the 336 executed fits never include an anchor fit
    assert recorder.calls == 336
    assert H.PHASE7E_ANCHOR_FIT_COUNT == 42
    anchor_fits = list(csv.DictReader((H.PHASE7E_DIR / "fit_results.csv").open(encoding="utf-8")))
    unique = {(int(r["replicate"]), int(r["K"]), int(r["start"])) for r in anchor_fits}
    assert len(unique) == 42, "the Phase 7e anchor still holds exactly 42 unique fits"
    summary = json.loads((tmp_path / "run" / "full_summary.json").read_text(encoding="utf-8"))
    assert summary["anchor_unique_fits"] == 42
    assert summary["actual_full_fits"] == 336, "anchor fits are never added to the 336"


def test_FULLEXEC_selection_matrix_has_the_required_columns(tmp_path):
    _run_full_fake(tmp_path / "run")
    rows = list(csv.DictReader((tmp_path / "run" / "selection_matrix.csv").open(encoding="utf-8")))
    for column in ("estimand", "role", "K_TRUE", "replicate", "selected_k", "signed_error",
                   "abs_error", "lineage", "run_code_sha", "artifact_dir"):
        assert column in rows[0], column
    for row in rows:
        signed = int(row["selected_k"]) - int(row["K_TRUE"])
        assert int(row["signed_error"]) == signed
        assert int(row["abs_error"]) == abs(signed)
        assert row["role"] == ("primary" if row["estimand"] == "A" else "sensitivity")


# --- the audit of a complete run --------------------------------------------


def test_FULLEXEC_complete_run_passes_the_independent_audit(tmp_path):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    auditor = A.audit_full_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs
    A.write_full_audit_report(directory, auditor)
    report = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["expected_full_fits"] == 336


def test_FULLEXEC_recovery_mismatch_does_not_fail_the_operational_audit(tmp_path):
    """§4: audit PASS never depends on selected_k == K_TRUE."""

    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    rows = list(csv.DictReader((directory / "selection_matrix.csv").open(encoding="utf-8")))
    new_rows = [r for r in rows if r["lineage"] == "phase8a_new"]
    mismatched = [r for r in new_rows if int(r["selected_k"]) != int(r["K_TRUE"])]
    assert mismatched, "the fake run should not recover K_TRUE everywhere"
    auditor = A.audit_full_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    # the audit never inspects recovery as a gate
    body = _inspect.getsource(A.audit_selection_matrix)
    assert "== k_true" not in body.replace(" ", "").lower() or "signed" in body


def test_FULLEXEC_audit_detects_a_tampered_selection(tmp_path):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    path = directory / "selection_matrix.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    k_index, sel_index = header.index("K_TRUE"), header.index("selected_k")
    fixed = [lines[0]]
    for line in lines[1:]:
        cells = line.split(",")
        if cells[k_index] != "3" and cells[sel_index] == cells[k_index]:
            cells[sel_index] = str(int(cells[sel_index]) + 1)
        fixed.append(",".join(cells))
    path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check in ("matrix_selected_k_recomputed", "matrix_signed_error")
               for f in auditor.blockers)


@pytest.mark.parametrize("field,value", [
    ("lineage", "phase7e_anchor"),
    ("run_code_sha", "b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a"),
    ("artifact_dir", "expfam/results/k_selection/heldout_full_pilot_20260824"),
])
def test_FULLEXEC_audit_rejects_a_new_row_claiming_the_anchor(tmp_path, field, value):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    path = directory / "selection_matrix.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    k_index, index = header.index("K_TRUE"), header.index(field)
    fixed = [lines[0]]
    for line in lines[1:]:
        cells = line.split(",")
        if cells[k_index] == "1":
            cells[index] = value
        fixed.append(",".join(cells))
    path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check.startswith("matrix_new_") for f in auditor.blockers)


def test_FULLEXEC_audit_detects_a_missing_fit(tmp_path):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    path = directory / "full_fit_results.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check == "full_fit_row_count" for f in auditor.blockers)


def test_FULLEXEC_zero_real_em_in_every_path(tmp_path, monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    recorder = _run_full_fake(tmp_path / "run")
    assert recorder.calls == 336
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    summary = json.loads((tmp_path / "run" / "full_summary.json").read_text(encoding="utf-8"))
    assert summary["actual_full_fits"] == 336
    runinfo = json.loads((tmp_path / "run" / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["actual_full_fits"] == 0, "test-only lineage records 0 REAL fits"
    assert runinfo["phase7e_rerun_count"] == 0
    assert runinfo["canary_fits_executed"] == 0 and runinfo["smoke_fits_executed"] == 0
    assert runinfo["replacement_fits_executed"] == 0
    _assert_no_new_production_artifacts()


# ===========================================================================
# PR #60 remote review: BLOCKER-01 / HIGH-02 / HIGH-03 / MEDIUM-04
# ===========================================================================


# --- BLOCKER-01: the working-tree state is frozen BEFORE the run writes ----


def test_FINDINGS_working_tree_state_is_frozen_before_the_artifact_dir(tmp_path, monkeypatch):
    """A correct run creates untracked output; that must not make it unauditable.

    Production timing: clean BEFORE the directory exists -> the directory is
    created -> git status is now dirty -> runinfo still records the frozen
    pre-execution True -> the independent audit can PASS.
    """

    out = tmp_path / "run"
    observations = []

    def _clean_then_dirty():
        # True only while the artifact directory does not exist yet
        clean = not out.exists()
        observations.append(clean)
        return clean

    monkeypatch.setattr(H, "working_tree_is_clean", _clean_then_dirty)
    _run_full_fake(out)

    assert observations == [True], "git status must be read exactly once, before the write"
    assert out.exists(), "the run created its own untracked artifact directory"
    assert H.working_tree_is_clean() is False, "the tree is now 'dirty' by that definition"

    runinfo = json.loads((out / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["working_tree_clean"] is True
    assert runinfo["working_tree_clean_before_execution"] is True

    directory = _promote_full_fixture(out, tmp_path / "real")
    auditor = A.audit_full_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]


def test_FINDINGS_runinfo_never_rereads_git_status():
    body = _executable_body(H.build_full_runinfo_payload)
    assert "working_tree_is_clean()" not in body
    assert "working_tree_clean_before_execution" in body
    executor = _executable_body(H._execute_real_full)
    # exactly one evaluation, and it happens before the directory is reserved
    assert executor.count("working_tree_is_clean()") == 1
    assert executor.index("working_tree_is_clean()") < \
        executor.index("require_new_full_artifact_dir(")


def test_FINDINGS_dirty_tree_still_blocks_a_production_run(tmp_path, monkeypatch):
    """The production path keeps the clean-tree requirement; test-only records it.

    A production attempt cannot even reach that check today -- the authorization
    boundary refuses first -- so the requirement is pinned in the source and the
    earlier refusal is asserted here.
    """

    monkeypatch.setattr(H, "working_tree_is_clean", lambda: False)
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_full(_full_test_authorization(), tmp_path / "run",
                             test_adapter=None, test_only=False,
                             run_code_sha="0" * 40)
    assert "provenance is unauthorized" in str(excinfo.value)
    assert not (tmp_path / "run").exists()

    body = _executable_body(H._execute_real_full)
    assert "if not test_only:" in body
    assert "the working tree is dirty before the full execution" in body
    assert "approved_baseline_is_ancestor_of(" in body
    # the requirement sits before the directory is reserved
    assert body.index("the working tree is dirty before the full execution") <         body.index("require_new_full_artifact_dir(")


def test_FINDINGS_audit_requires_the_frozen_pre_execution_state(tmp_path):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    _patch_json(directory / "runinfo.json", working_tree_clean_before_execution=False)
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check == "full_runinfo_working_tree_frozen" for f in auditor.blockers)


# --- HIGH-02: the failing fit index comes from the fit-call boundary -------


@pytest.mark.parametrize("fail_at", [1, 14, 15, 100, 336])
def test_FINDINGS_failure_index_is_exact(tmp_path, fail_at):
    out = tmp_path / "run"
    recorder = _DirtyAtFitRecorder(fail_at=fail_at, mode="retry")
    with pytest.raises(HarnessStop):
        H._execute_real_full_test_only(_full_test_authorization(), out,
                                       adapter=_test_adapter(recorder),
                                       run_code_sha="0" * 40)
    assert recorder.calls == fail_at, "the sweep must stop at the failing fit"
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert failure["failed_fit_index"] == fail_at
    assert failure["attempted_fit_count"] == fail_at
    assert failure["clean_fit_calls"] == fail_at - 1
    assert failure["failure_phase"] == "fit"
    assert failure["replacement_fits_executed"] == 0 and failure["retry_count"] == 0
    # the scored rows are whole cells only: scoring is deferred to the cell end
    assert failure["scored_rows"] == ((fail_at - 1) // 14) * 14


def test_FINDINGS_failure_index_is_exact_for_an_adapter_exception(tmp_path):
    out = tmp_path / "run"
    recorder = _DirtyAtFitRecorder(fail_at=100, mode="raise")
    with pytest.raises(RuntimeError):
        H._execute_real_full_test_only(_full_test_authorization(), out,
                                       adapter=_test_adapter(recorder),
                                       run_code_sha="0" * 40)
    assert recorder.calls == 100
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert failure["attempted_fit_count"] == 100 and failure["failed_fit_index"] == 100
    assert failure["clean_fit_calls"] == 99 and failure["failure_phase"] == "fit"


def test_FINDINGS_a_score_failure_is_not_reported_as_a_fit_failure(tmp_path, monkeypatch):
    """Deferred scoring can fail after 14 clean fits; that is not a fit failure."""

    calls = {"n": 0}
    real_score = H.score_heldout_bernoulli

    def _failing_score(target, eta_pairs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected score failure")
        return real_score(target, eta_pairs)

    monkeypatch.setattr(H, "score_heldout_bernoulli", _failing_score)
    out = tmp_path / "run"
    recorder = _FakeFitRecorder()
    with pytest.raises(RuntimeError):
        H._execute_real_full_test_only(_full_test_authorization(), out,
                                       adapter=_test_adapter(recorder),
                                       run_code_sha="0" * 40)
    assert recorder.calls == 14, "a whole cell was fitted before scoring began"
    failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_phase"] == "score"
    assert failure["attempted_fit_count"] == 14
    assert failure["clean_fit_calls"] == 14, "the 14 clean fits are not lost"
    assert failure["scored_rows"] == 2


def test_FINDINGS_progress_counters_are_owned_by_the_fit_boundary():
    body = _executable_body(H._run_full_cell)
    assert "progress.begin_fit()" in body
    assert body.index("progress.begin_fit()") < body.index("boundary.call(0)")
    assert "progress.fit_completed_clean()" in body
    assert "progress.begin_scoring()" in body and "progress.row_scored()" in body
    failure_body = _executable_body(H.write_full_failure_json)
    for forbidden in ("len(completed)", "completed_fits", "scored_rows + 1"):
        assert forbidden not in failure_body, forbidden


def test_FINDINGS_complete_run_records_the_counters(tmp_path):
    _run_full_fake(tmp_path / "run")
    runinfo = json.loads((tmp_path / "run" / "runinfo.json").read_text(encoding="utf-8"))
    assert runinfo["attempted_fit_count"] == 336
    assert runinfo["clean_fit_calls"] == 336
    assert runinfo["scored_rows"] == 336
    assert not (tmp_path / "run" / "failure.json").exists()


@pytest.mark.parametrize("field", ["attempted_fit_count", "clean_fit_calls", "scored_rows"])
def test_FINDINGS_audit_requires_the_counters(tmp_path, field):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    _patch_json(directory / "runinfo.json", **{field: 335})
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check.startswith("full_runinfo_") for f in auditor.blockers)


# --- HIGH-03: one approved baseline across every full artifact -------------


def test_FINDINGS_runinfo_baseline_comes_from_the_authorization(tmp_path):
    _run_full_fake(tmp_path / "run")
    authorization = json.loads(
        (tmp_path / "run" / "authorization.json").read_text(encoding="utf-8"))
    runinfo = json.loads((tmp_path / "run" / "runinfo.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run" / "full_summary.json").read_text(encoding="utf-8"))
    baseline = authorization["reviewed_full_execution_main_sha"]
    assert baseline == H._FULL_TEST_EXPECTED_MAIN_SHA
    assert runinfo["reviewed_full_execution_main_sha"] == baseline
    assert summary["reviewed_full_execution_main_sha"] == baseline
    rows = list(csv.DictReader(
        (tmp_path / "run" / "full_fit_results.csv").open(encoding="utf-8")))
    assert {r["reviewed_full_execution_main_sha"] for r in rows} == {baseline}
    assert baseline != runinfo["run_code_sha"]
    # HIGH-05: role 1 is the frozen constant, role 2 comes from the caller
    assert runinfo["scientific_baseline_sha"] == H.APPROVED_SCIENTIFIC_MAIN_SHA
    assert authorization["scientific_baseline_sha"] == H.APPROVED_SCIENTIFIC_MAIN_SHA
    assert summary["scientific_baseline_sha"] == H.APPROVED_SCIENTIFIC_MAIN_SHA
    assert {r["scientific_baseline_sha"] for r in rows} == {H.APPROVED_SCIENTIFIC_MAIN_SHA}
    assert baseline != H.APPROVED_SCIENTIFIC_MAIN_SHA, "the two roles are distinct values"
    body = _executable_body(H.build_full_runinfo_payload)
    assert "'scientific_baseline_sha': APPROVED_SCIENTIFIC_MAIN_SHA" in body
    assert "'reviewed_full_execution_main_sha': approved_main_sha" in body
    assert "approved_baseline_is_ancestor_of(" in body


@pytest.mark.parametrize("target", ["runinfo.json", "full_summary.json", "csv_row"])
def test_FINDINGS_audit_rejects_a_split_baseline(tmp_path, target):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    if target == "csv_row":
        _patch_full_csv_cell(directory, "reviewed_full_execution_main_sha", 5, "d" * 40)
    else:
        _patch_json(directory / target, reviewed_full_execution_main_sha="d" * 40)
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check == "full_baseline_sha_lineage" for f in auditor.blockers), \
        sorted({f.check for f in auditor.blockers})


def _patch_full_csv_cell(directory, column, line_number, value):
    """Overwrite one cell of full_fit_results.csv (1-based data line)."""

    path = directory / "full_fit_results.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    index = header.index(column)
    cells = lines[line_number].split(",")
    cells[index] = value
    lines[line_number] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_FINDINGS_audit_rejects_a_baseline_equal_to_the_run_sha(tmp_path):
    # role 3 is set to the reviewed baseline the promotion stamps into role 2,
    # so the two roles collapse and the audit must reject the artifact set
    _run_full_fake(tmp_path / "run", run_code_sha="02ef35add45036975162b6a267f6428c3b380459")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check in ("full_baseline_not_run_sha", "full_auth_baseline_not_run_sha")
               for f in auditor.blockers)


def test_FINDINGS_full_ancestry_helper_is_explicit():
    body = _executable_body(H.approved_baseline_is_ancestor_of)
    assert "APPROVED_SCIENTIFIC_MAIN_SHA" not in body
    assert "approved_main_sha" in body
    parameters = list(_inspect.signature(H.approved_baseline_is_ancestor_of).parameters)
    assert parameters == ["approved_main_sha", "run_code_sha"]
    # the smoke helper is untouched
    smoke_body = _executable_body(H.approved_baseline_is_ancestor)
    assert "APPROVED_SCIENTIFIC_MAIN_SHA" in smoke_body


# --- MEDIUM-04: --full always means the complete A+B sweep -----------------


@pytest.mark.parametrize("estimand", ["A", "B"])
def test_FINDINGS_full_refuses_a_per_estimand_scope(estimand, monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", estimand])
    message = str(excinfo.value)
    assert "always executes the complete AB sweep" in message
    assert "336" in message and "168" in message
    assert _AdapterTripwire.constructions == 0


@pytest.mark.parametrize("argv", [
    ["--full", "--allow-em", "--confirm-k-true-sweep"],
    ["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"],
])
def test_FINDINGS_full_scope_ab_reaches_the_authorization_gate(argv, monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    reached = _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(argv)
    assert "not authorized" in str(excinfo.value)
    assert reached == [], "the AB scope stops at the authorization gate"
    assert _AdapterTripwire.constructions == 0


def test_FINDINGS_full_scope_constant_matches_the_frozen_estimands():
    assert H.FULL_ESTIMAND_SCOPE == "AB" == H.ESTIMANDS
    assert "".join(H.active_estimands()) == H.FULL_ESTIMAND_SCOPE
    options = {option for action in H._build_parser()._actions
               for option in action.option_strings}
    assert "--estimand" in options
    choices = next(a.choices for a in H._build_parser()._actions
                   if a.option_strings and a.option_strings[0] == "--estimand")
    assert set(choices) == {"A", "B", "AB"}
    # per-estimand modes still work for the zero-EM audits
    assert H.main(["--validate-only"]) == 0


def test_FINDINGS_full_executes_both_estimands_when_scoped_ab(tmp_path):
    recorder = _run_full_fake(tmp_path / "run")
    rows = list(csv.DictReader(
        (tmp_path / "run" / "full_fit_results.csv").open(encoding="utf-8")))
    per_estimand = {}
    for row in rows:
        per_estimand[row["estimand"]] = per_estimand.get(row["estimand"], 0) + 1
    assert per_estimand == {"A": 168, "B": 168}
    assert recorder.calls == 336


# ===========================================================================
# PR #60 re-review: HIGH-05 (SHA roles) / MEDIUM-06 (fit order)
# ===========================================================================


def _full_fixture(tmp_path):
    _run_full_fake(tmp_path / "run")
    return _promote_full_fixture(tmp_path / "run", tmp_path / "real")


# --- HIGH-05: three separated roles ----------------------------------------


def test_ROLES_three_fields_exist_in_every_artifact(tmp_path):
    directory = _full_fixture(tmp_path)
    authorization = json.loads((directory / "authorization.json").read_text(encoding="utf-8"))
    runinfo = json.loads((directory / "runinfo.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "full_summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((directory / "full_fit_results.csv").open(encoding="utf-8")))

    for payload in (authorization, runinfo, summary):
        for field in H.FULL_SHA_ROLE_FIELDS:
            assert field in payload, field
    for field in H.FULL_SHA_ROLE_FIELDS:
        assert field in rows[0], field

    scientific = H.APPROVED_SCIENTIFIC_MAIN_SHA
    reviewed = authorization["reviewed_full_execution_main_sha"]
    run_code = authorization["run_code_sha"]
    assert scientific == "68c78e1191889609dead05ea5a9fb11525ce92e2"
    assert len({scientific, reviewed, run_code}) == 3, "the three roles are distinct values"

    for payload in (authorization, runinfo, summary):
        assert payload["scientific_baseline_sha"] == scientific
        assert payload["reviewed_full_execution_main_sha"] == reviewed
        assert payload["run_code_sha"] == run_code
    assert {r["scientific_baseline_sha"] for r in rows} == {scientific}
    assert {r["reviewed_full_execution_main_sha"] for r in rows} == {reviewed}
    assert {r["run_code_sha"] for r in rows} == {run_code}
    # the ambiguous legacy name is gone from the full artifacts
    for payload in (authorization, runinfo, summary):
        assert "approved_scientific_main_sha" not in payload
    assert "approved_scientific_main_sha" not in rows[0]


def test_ROLES_audit_holds_the_scientific_literal_independently():
    assert A.EXPECTED_SCIENTIFIC_BASELINE_SHA == "68c78e1191889609dead05ea5a9fb11525ce92e2"
    assert A.EXPECTED_SCIENTIFIC_BASELINE_SHA == H.APPROVED_SCIENTIFIC_MAIN_SHA
    assert A.FULL_SHA_ROLE_FIELDS == H.FULL_SHA_ROLE_FIELDS
    source = _inspect.getsource(A)
    assert "import run_k_true_robustness_sweep" not in source


def test_ROLES_reviewed_full_sha_is_bound_but_unauthorized():
    """S3-B bound role 2; S3-D reclosed the gate, so role 2 stays unconsumed."""

    assert H.current_expected_full_main_sha() == "02ef35add45036975162b6a267f6428c3b380459"
    assert H.trusted_full_main_sha_for(test_only=False) == "02ef35add45036975162b6a267f6428c3b380459"
    assert H.current_full_execution_authorization() is None
    body = _executable_body(H.current_expected_full_main_sha)
    assert body.strip() == "return REVIEWED_FULL_EXECUTION_MAIN_SHA"


def test_ROLES_production_lineage_chain_is_required():
    body = _executable_body(H._execute_real_full)
    assert "APPROVED_SCIENTIFIC_MAIN_SHA, authorization.approved_main_sha" in \
        body.replace("\n", " ").replace("  ", " ") or \
        "approved_baseline_is_ancestor_of(APPROVED_SCIENTIFIC_MAIN_SHA" in body
    assert "approved_baseline_is_ancestor_of(authorization.approved_main_sha" in body
    assert "must not be the scientific" in body


@pytest.mark.parametrize("target", ["authorization.json", "runinfo.json",
                                    "full_summary.json", "csv_row"])
def test_ROLES_tamper_scientific_baseline_fails(tmp_path, target):
    directory = _full_fixture(tmp_path)
    if target == "csv_row":
        _patch_full_csv_cell(directory, "scientific_baseline_sha", 7, "e" * 40)
    else:
        _patch_json(directory / target, scientific_baseline_sha="e" * 40)
    auditor = A.audit_full_run_dir(directory)
    checks = {f.check for f in auditor.blockers}
    assert "full_scientific_baseline_sha" in checks or \
        "full_auth_scientific_baseline" in checks, sorted(checks)


@pytest.mark.parametrize("target", ["runinfo.json", "full_summary.json", "csv_row"])
def test_ROLES_tamper_reviewed_full_sha_fails(tmp_path, target):
    directory = _full_fixture(tmp_path)
    if target == "csv_row":
        _patch_full_csv_cell(directory, "reviewed_full_execution_main_sha", 9, "f" * 40)
    else:
        _patch_json(directory / target, reviewed_full_execution_main_sha="f" * 40)
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check == "full_baseline_sha_lineage" for f in auditor.blockers)


@pytest.mark.parametrize("target", ["runinfo.json", "full_summary.json", "csv_row"])
def test_ROLES_tamper_run_code_sha_fails(tmp_path, target):
    directory = _full_fixture(tmp_path)
    if target == "csv_row":
        _patch_full_csv_cell(directory, "run_code_sha", 3, "1" * 40)
    else:
        _patch_json(directory / target, run_code_sha="1" * 40)
    auditor = A.audit_full_run_dir(directory)
    checks = {f.check for f in auditor.blockers}
    assert "full_baseline_sha_lineage" in checks or "full_run_code_sha_lineage" in checks, \
        sorted(checks)


def test_ROLES_reviewed_sha_in_the_scientific_field_fails(tmp_path):
    """Putting role 2's SHA into role 1's field must never pass."""

    directory = _full_fixture(tmp_path)
    reviewed = json.loads(
        (directory / "authorization.json").read_text(encoding="utf-8")
    )["reviewed_full_execution_main_sha"]
    for name in ("authorization.json", "runinfo.json", "full_summary.json"):
        _patch_json(directory / name, scientific_baseline_sha=reviewed)
    auditor = A.audit_full_run_dir(directory)
    checks = {f.check for f in auditor.blockers}
    assert "full_scientific_baseline_sha" in checks or \
        "full_auth_scientific_baseline" in checks, sorted(checks)


@pytest.mark.parametrize("field", ["scientific_baseline_is_ancestor_of_reviewed_full",
                                   "reviewed_full_baseline_is_ancestor_of_run_code"])
def test_ROLES_missing_ancestry_conclusion_fails(tmp_path, field):
    directory = _full_fixture(tmp_path)
    _patch_json(directory / "runinfo.json", **{field: False})
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check in ("full_scientific_baseline_ancestry",
                           "full_reviewed_baseline_ancestry")
               for f in auditor.blockers)


def test_ROLES_clean_fixture_still_passes(tmp_path):
    directory = _full_fixture(tmp_path)
    auditor = A.audit_full_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]
    assert not auditor.highs


# --- MEDIUM-06: fit_index and the deterministic order ----------------------


def test_ORDER_auditor_rebuilds_the_order_independently():
    ordered = A.expected_ordered_full_keys()
    assert len(ordered) == 336
    assert ordered[0] == ("A", 1, 1, 1, 1)
    assert ordered[13] == ("A", 1, 1, 7, 2)
    assert ordered[14] == ("A", 1, 2, 1, 1)
    assert ordered[167] == ("A", 5, 3, 7, 2)
    assert ordered[168] == ("B", 1, 1, 1, 1)
    assert ordered[-1] == ("B", 5, 3, 7, 2)
    assert set(ordered) == A.expected_full_keys()
    assert len(set(ordered)) == len(ordered)
    body = _inspect.getsource(A.expected_ordered_full_keys)
    assert "run_k_true_robustness_sweep" not in body and "H." not in body
    # and it agrees with what the harness actually executed
    assert [(e, kt, rp) for e, kt, rp in H.full_execution_order()] == \
        list(dict.fromkeys((k[0], k[1], k[2]) for k in ordered))


def test_ORDER_clean_run_matches_the_frozen_order(tmp_path):
    directory = _full_fixture(tmp_path)
    rows = list(csv.DictReader((directory / "full_fit_results.csv").open(encoding="utf-8")))
    observed = [(r["estimand"], int(r["K_TRUE"]), int(r["replicate"]),
                 int(r["K"]), int(r["start"])) for r in rows]
    assert tuple(observed) == A.expected_ordered_full_keys()
    assert [int(r["fit_index"]) for r in rows] == list(range(1, 337))
    auditor = A.audit_full_run_dir(directory)
    assert not auditor.blockers, [f"{f.check}: {f.detail}" for f in auditor.blockers]


def test_ORDER_swapped_fit_index_fails(tmp_path):
    """1. fit_index of two rows swapped, rows left in place."""

    directory = _full_fixture(tmp_path)
    _patch_full_csv_cell(directory, "fit_index", 10, "20")
    _patch_full_csv_cell(directory, "fit_index", 20, "10")
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check == "full_fit_index_position" for f in auditor.blockers)


def test_ORDER_swapped_rows_with_fixed_index_fails(tmp_path):
    """2. two whole rows swapped while fit_index stays 1..336 in place."""

    directory = _full_fixture(tmp_path)
    path = directory / "full_fit_results.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    index = header.index("fit_index")
    first, second = lines[30].split(","), lines[60].split(",")
    first[index], second[index] = second[index], first[index]   # keep 1..336 in place
    lines[30], lines[60] = ",".join(second), ",".join(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    auditor = A.audit_full_run_dir(directory)
    assert any(f.check == "full_fit_execution_order" for f in auditor.blockers), \
        sorted({f.check for f in auditor.blockers})


def test_ORDER_duplicate_fit_index_fails(tmp_path):
    """3. a duplicated fit_index."""

    directory = _full_fixture(tmp_path)
    _patch_full_csv_cell(directory, "fit_index", 40, "39")
    auditor = A.audit_full_run_dir(directory)
    checks = {f.check for f in auditor.blockers}
    assert "full_fit_index_sequence" in checks or "full_fit_index_position" in checks, \
        sorted(checks)


def test_ORDER_out_of_range_fit_index_fails(tmp_path):
    """4. fit_index 337."""

    directory = _full_fixture(tmp_path)
    _patch_full_csv_cell(directory, "fit_index", 336, "337")
    auditor = A.audit_full_run_dir(directory)
    checks = {f.check for f in auditor.blockers}
    assert "full_fit_index_sequence" in checks or "full_fit_index_position" in checks, \
        sorted(checks)


def test_ORDER_four_independent_checks_exist():
    body = _inspect.getsource(A.audit_full_fit_rows)
    for check in ("full_fit_row_count", "full_fit_key_set", "full_fit_execution_order",
                  "full_fit_index_sequence", "full_fit_index_position"):
        assert check in body, check


def test_ORDER_zero_em_maintained(tmp_path, monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    directory = _full_fixture(tmp_path)
    A.audit_full_run_dir(directory)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    assert H.current_full_execution_authorization() is None
    assert H.current_expected_full_main_sha() == "02ef35add45036975162b6a267f6428c3b380459"
    assert not H.FULL_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()


# ===========================================================================
# Issue #59 Phase 8b S3-E: REVISED reviewed full-execution baseline binding ONLY
# ===========================================================================
#
# The reviewed PR #63 merge SHA is now the trusted role-2 value, replacing the
# pre-fix S3-A merge SHA that S3-B had bound.  Nothing else changed: no
# production FullExecutionAuthorization record exists, neither human gate is
# granted, and --full still stops before any adapter.  Rebinding reviewed code
# provenance is NOT an approval and NOT a scientific-protocol change.

REVIEWED_FULL_MAIN_SHA = "02ef35add45036975162b6a267f6428c3b380459"
# The superseded pre-fix baseline, kept as evidence.  The historical S3-C human
# approval (Issue #59 comment 5511177444) was granted against THIS SHA and
# executed 0 real full EM; it is never transferred to the revised baseline.
HISTORICAL_REVIEWED_FULL_MAIN_SHA = "8b6b43c9f5f5750d19409bb9afd6cf4d87d0ea1f"
SCIENTIFIC_BASELINE_SHA = "68c78e1191889609dead05ea5a9fb11525ce92e2"


def test_BASELINE_reviewed_full_sha_is_the_exact_literal():
    assert H.REVIEWED_FULL_EXECUTION_MAIN_SHA == REVIEWED_FULL_MAIN_SHA
    assert H.current_expected_full_main_sha() == REVIEWED_FULL_MAIN_SHA
    assert H.trusted_full_main_sha_for(test_only=False) == REVIEWED_FULL_MAIN_SHA
    H._require_full_commit_sha(H.current_expected_full_main_sha(), "reviewed full SHA")


def test_BASELINE_roles_stay_separate():
    scientific = H.APPROVED_SCIENTIFIC_MAIN_SHA
    reviewed = H.current_expected_full_main_sha()
    assert scientific == SCIENTIFIC_BASELINE_SHA
    assert reviewed == REVIEWED_FULL_MAIN_SHA
    assert reviewed != scientific, "role 1 and role 2 are different values"
    # role 1 is unchanged by this binding
    assert H.current_expected_smoke_main_sha() == scientific
    assert A.EXPECTED_SCIENTIFIC_BASELINE_SHA == scientific
    # role 3 is not fixed by this PR, and role 2 is bound but unauthorized
    assert H.current_full_execution_authorization() is None
    assert H.current_expected_full_main_sha() == reviewed


def test_BASELINE_authorization_record_is_absent_again():
    assert H.current_full_execution_authorization() is None
    body = _executable_body(H.current_full_execution_authorization)
    assert body.strip() == "return None"
    assert "FullExecutionAuthorization(" not in body
    for forbidden in ("human_full_approval", "independent_review_pass"):
        assert forbidden not in body, forbidden
    # no production record is constructed anywhere outside the test factory
    source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    constructions = [line.strip() for line in source.splitlines()
                     if "FullExecutionAuthorization(" in line
                     and "type(" not in line and "is FullExecutionAuthorization" not in line]
    assert len(constructions) == 1, constructions      # the test-only factory only
    assert "_FULL_TEST_AUTHORITY" in source.split("FullExecutionAuthorization(")[-1][:400]


@pytest.mark.parametrize("argv", [
    ["--full", "--allow-em", "--confirm-k-true-sweep"],
    ["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"],
])
def test_BASELINE_full_still_hard_stops(argv, monkeypatch, tmp_path):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    monkeypatch.setattr(H, "FULL_ARTIFACT_DIR", tmp_path / "frozen_full")
    reached = _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(argv)
    message = str(excinfo.value)
    assert "not authorized" in message
    assert "NO committed FullExecutionAuthorization record exists" in message
    assert "HUMAN_FULL_APPROVAL" in message
    assert "never be reused for --full" in message
    assert reached == []
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert not (tmp_path / "frozen_full").exists()
    assert "em_runner" not in sys.modules


def test_BASELINE_refusal_branch_is_fail_closed_dead_code():
    """The record exists, so the refusal is unreachable -- but it must remain."""

    body = _executable_body(H._require_em_authorization)
    full_branch = body.split("if command == 'full':")[1]
    assert "no reviewed main SHA has been approved" not in full_branch
    assert "REVIEWED_FULL_EXECUTION_MAIN_SHA" in full_branch
    assert "if full_authorization is None:" in full_branch
    assert "raise HarnessStop" in full_branch


def test_BASELINE_smoke_isolation_is_unchanged():
    smoke = H.current_smoke_execution_authorization()
    assert smoke is not None
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_full_execution_authorization(smoke, test_only=False)
    assert "FullExecutionAuthorization" in str(excinfo.value)
    with pytest.raises(HarnessStop):
        H.validate_smoke_execution_authorization(H._make_test_full_authorization(),
                                                 test_only=False)
    assert H.smoke_protocol_hash() == \
        "1f6fae965cffcfc362836554a171152f2e60e67a801eb5ec09b034976315ec09"


def test_BASELINE_a_forged_record_still_fails_the_provenance_gate():
    """Only the committed record carries the production authority sentinel."""

    forged = H._make_test_full_authorization(
        approved_main_sha=REVIEWED_FULL_MAIN_SHA)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_full_execution_authorization(forged, test_only=False)
    assert "provenance is unauthorized" in str(excinfo.value)
    assert H.current_full_execution_authorization() is None
    assert forged.is_test_only() is True


def test_BASELINE_source_is_a_committed_literal():
    """§5: the trusted source is never derived at runtime."""

    tree = _ast.parse(_textwrap.dedent(
        _inspect.getsource(H.current_expected_full_main_sha)))
    calls = [getattr(n.func, "id", getattr(n.func, "attr", "?"))
             for n in _ast.walk(tree) if isinstance(n, _ast.Call)]
    assert calls == [], f"the trusted source must not call anything: {calls}"
    names = {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
    # "str" is the return annotation; the only value referenced is the literal
    assert names - {"str"} == {"REVIEWED_FULL_EXECUTION_MAIN_SHA"}
    body = _executable_body(H.current_expected_full_main_sha)
    for forbidden in ("os.", "environ", "getenv", "argv", "sys.", "subprocess", "git",
                      "rev-parse", "current_run_code_sha", "Path", "open(", "json",
                      "config", "branch", "HEAD"):
        assert forbidden not in body, forbidden
    # the literal itself is a module-level committed constant
    module_source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    assert f'REVIEWED_FULL_EXECUTION_MAIN_SHA = "{REVIEWED_FULL_MAIN_SHA}"' in module_source


def test_BASELINE_env_and_cli_cannot_change_it(monkeypatch):
    for name in ("REVIEWED_FULL_EXECUTION_MAIN_SHA", "PHASE8B_FULL_MAIN_SHA",
                 "APPROVED_MAIN_SHA", "PHASE8B_FULL_AUTHORIZED"):
        monkeypatch.setenv(name, "9" * 40)
    monkeypatch.setattr(sys, "argv", ["run", "--full", "--allow-em", "--approve-full"])
    assert H.current_expected_full_main_sha() == REVIEWED_FULL_MAIN_SHA
    assert H.current_full_execution_authorization() is None
    options = {option for action in H._build_parser()._actions
               for option in action.option_strings}
    for forbidden in ("--full-main-sha", "--reviewed-full-sha", "--approve-full",
                      "--human-full-approved"):
        assert forbidden not in options, forbidden


def test_BASELINE_zero_em_state_is_unchanged(tmp_path, monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    report = H.run_full_preflight()
    assert report["em_fits_executed"] == 0
    assert report["real_full_fits_executed"] == 0
    assert report["trusted_full_main_sha_present"] is True
    assert report["full_execution_authorization_present"] is False
    assert report["manifest"]["total_fits"] == 336
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    assert not H.FULL_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()


# ===========================================================================
# Issue #59 Phase 8b S3-D: the S3-C authorization is STALE and withdrawn
# ===========================================================================
#
# S3-C committed a real FullExecutionAuthorization for exactly 336 fits, granted
# by the human in Issue #59 comment 5511177444.  It was NEVER consumed: real
# full EM executed under it = 0 and the production artifact directory was never
# created.  S3-D then changed the execution path (the persisted full manifest
# now carries global fit_index 1..336), so that approval no longer describes the
# code it would run.  The record is therefore withdrawn here, the historical
# provenance is preserved rather than deleted, and NO new approval is
# fabricated.  Nothing in this group executes EM.

HISTORICAL_S3C_AUTHORIZED_FIELDS = {
    "issue_number": 59,
    "protocol_origin_issue_number": 49,
    "approved_main_sha": "8b6b43c9f5f5750d19409bb9afd6cf4d87d0ea1f",
    "estimands": ("A", "B"),
    "k_true_grid": (1, 2, 4, 5),
    "fits_per_estimand": 168,
    "total_fit_count": 336,
}


# --- the production gate is closed again ------------------------------------


def test_AUTHZ_production_authorization_is_closed():
    """§1/§10: the revised execution code must not be executable."""

    assert H.current_full_execution_authorization() is None
    body = _executable_body(H.current_full_execution_authorization)
    assert body.strip() == "return None"
    assert "FullExecutionAuthorization(" not in body
    for forbidden in ("human_full_approval", "independent_review_pass",
                      "_FULL_EXECUTION_AUTHORITY"):
        assert forbidden not in body, forbidden


def test_AUTHZ_no_new_human_approval_is_fabricated():
    """Exactly one construction site remains, and it is the test-only factory."""

    source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    constructions = [line.strip() for line in source.splitlines()
                     if "FullExecutionAuthorization(" in line
                     and "type(" not in line and "is FullExecutionAuthorization" not in line]
    assert len(constructions) == 1, constructions
    assert "_FULL_TEST_AUTHORITY" in source.split("FullExecutionAuthorization(")[-1][:400]
    factory = H._make_test_full_authorization()
    assert factory._authority is H._FULL_TEST_AUTHORITY
    assert factory.is_test_only() is True
    # a test-only record can never satisfy the production validator
    with pytest.raises(HarnessStop):
        H.validate_full_execution_authorization(factory, test_only=False)


def test_AUTHZ_the_stale_approval_is_explained_not_deleted():
    doc = H.current_full_execution_authorization.__doc__
    for fragment in ("S3-C", "STALE", "0", "fit_index", "FRESH", "merge"):
        assert fragment in doc, fragment
    assert "None" in doc


def test_AUTHZ_historical_provenance_is_preserved():
    """§1: the Issue/comment provenance is kept as historical evidence."""

    assert H.FULL_HUMAN_AUTHORIZATION_ISSUE_NUMBER == 59
    assert H.FULL_HUMAN_AUTHORIZATION_ISSUE_COMMENT_ID == 5511177444
    assert type(H.FULL_HUMAN_AUTHORIZATION_ISSUE_COMMENT_ID) is int
    scope = H.FULL_HUMAN_AUTHORIZATION_SCOPE
    for fragment in ("336", "A=168", "B=168", "rerun=0", "retry=0", "replacement=0"):
        assert fragment in scope, fragment
    exclusions = H.FULL_HUMAN_AUTHORIZATION_EXCLUSIONS
    for excluded in ("rerun_after_success", "rerun_after_partial_failure",
                     "replacement_fit", "retry", "alternate_seed", "relaxed_tolerance",
                     "phase7e_anchor_rerun", "canary_rerun", "smoke_rerun",
                     "any_337th_fit"):
        assert excluded in exclusions, excluded
    # the historical scope still describes the same frozen sweep
    assert H.EXPECTED_FULL_FITS == HISTORICAL_S3C_AUTHORIZED_FIELDS["total_fit_count"]
    assert H.EXPECTED_FULL_FITS_PER_ESTIMAND == \
        HISTORICAL_S3C_AUTHORIZED_FIELDS["fits_per_estimand"]


def test_AUTHZ_the_stale_approval_is_not_transferred_to_the_new_baseline():
    """S3-E rebinds role 2; the S3-C approval stays attached to the OLD SHA."""

    historical = HISTORICAL_S3C_AUTHORIZED_FIELDS["approved_main_sha"]
    assert historical == HISTORICAL_REVIEWED_FULL_MAIN_SHA
    assert H.HISTORICAL_S3C_REVIEWED_FULL_EXECUTION_MAIN_SHA == historical
    # the current role 2 is the revised baseline, NOT the one that was approved
    assert H.REVIEWED_FULL_EXECUTION_MAIN_SHA == REVIEWED_FULL_MAIN_SHA
    assert H.REVIEWED_FULL_EXECUTION_MAIN_SHA != historical
    assert H.current_expected_full_main_sha() == H.REVIEWED_FULL_EXECUTION_MAIN_SHA
    assert H.APPROVED_SCIENTIFIC_MAIN_SHA == "68c78e1191889609dead05ea5a9fb11525ce92e2"
    assert H.current_expected_full_main_sha() != H.APPROVED_SCIENTIFIC_MAIN_SHA
    # rebinding did not reopen the gate
    assert H.current_full_execution_authorization() is None


def test_AUTHZ_cli_full_stops_before_any_adapter(monkeypatch):
    """§10: --full must stop before AuthorizedEMFitAdapter is constructed."""

    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    reached = _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"])
    message = str(excinfo.value)
    assert "not authorized" in message
    assert "NO committed FullExecutionAuthorization record exists" in message
    assert "STALE" in message and "5511177444" in message
    assert reached == [], "the production full workflow is never reached"
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    assert not H.FULL_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()


def test_AUTHZ_env_and_cli_cannot_reopen_the_gate(monkeypatch):
    for name in ("HUMAN_FULL_APPROVAL", "PHASE8B_HUMAN_FULL_APPROVAL",
                 "INDEPENDENT_REVIEW_PASS", "PHASE8B_FULL_AUTHORIZED",
                 "PHASE8B_FULL_FIT_COUNT", "REVIEWED_FULL_EXECUTION_MAIN_SHA"):
        monkeypatch.setenv(name, "9" * 40)
    monkeypatch.setattr(sys, "argv", ["run", "--full", "--allow-em", "--approve-full",
                                      "--full-fit-count", "672"])
    assert H.current_full_execution_authorization() is None
    options = {option for action in H._build_parser()._actions
               for option in action.option_strings}
    for forbidden in ("--approve-full", "--human-full-approved", "--full-fit-count",
                      "--authorize-full", "--full-main-sha"):
        assert forbidden not in options, forbidden


def test_AUTHZ_no_network_in_the_authorization_path():
    source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    for forbidden in ("import requests", "import urllib", "http://", "https://api.",
                      "gh api", "GITHUB_TOKEN"):
        assert forbidden not in source, forbidden


def test_AUTHZ_grid_multiplies_to_exactly_336():
    """The frozen plan is unchanged by withdrawing the record."""

    manifests = H.build_full_manifests()
    report = H.validate_full_manifests(manifests)
    assert report["fits_per_estimand"] == {"A": 168, "B": 168}
    assert report["total_fits"] == 336
    assert (len(H.NEW_K_TRUE) * len(H.REPLICATES) * len(H.K_CANDIDATES)
            * len(H.START_LABELS)) == 168
    assert 168 * len(H.active_estimands()) == 336 == H.EXPECTED_FULL_FITS


def test_AUTHZ_anchor_k_true_is_excluded_and_never_rerun():
    assert 3 not in H.NEW_K_TRUE
    assert H.ANCHOR_K_TRUE == 3
    assert H.EXPECTED_FULL_PHASE7E_RERUN_FITS == 0
    assert H.PHASE7E_ANCHOR_FIT_COUNT == 42
    report = H.run_full_preflight()
    assert report["phase7e_rerun_fits"] == 0
    assert report["anchor_agreement"]["phase7e_rerun_fits"] == 0
    assert report["em_fits_executed"] == 0


def test_AUTHZ_smoke_isolation_is_unchanged():
    smoke = H.current_smoke_execution_authorization()
    full = H._make_test_full_authorization()
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_full_execution_authorization(smoke, test_only=False)
    assert "FullExecutionAuthorization" in str(excinfo.value)
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_smoke_execution_authorization(full, test_only=False)
    assert "SmokeExecutionAuthorization" in str(excinfo.value)
    assert H.smoke_protocol_hash() != H.full_protocol_hash()


def test_AUTHZ_one_time_semantics_are_documented_and_enforced():
    """Whenever the gate reopens, it still permits ONE attempt and no resume."""

    body = _executable_body(H.require_new_full_artifact_dir)
    assert "not directory.exists()" in body
    assert "refusing to overwrite or resume" in body
    source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    for forbidden in ("--resume", "resume_from", "allow_resume", "skip_completed"):
        assert forbidden not in source, forbidden
    assert H.FULL_PARTIAL_FAILURE_POLICY[-1] == "rerun_requires_a_new_human_gate"


def test_AUTHZ_existing_directory_still_refuses(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(HarnessStop) as excinfo:
        H.require_new_full_artifact_dir(out)
    assert "already exists" in str(excinfo.value)


def test_AUTHZ_a_second_attempt_executes_no_replacement_fit(tmp_path):
    """One clean attempt only: the frozen directory blocks any second run."""

    recorder = _run_full_fake(tmp_path / "run")
    assert recorder.calls == 336
    second = _FakeFitRecorder()
    with pytest.raises(HarnessStop) as excinfo:
        H._execute_real_full_test_only(H._make_test_full_authorization(),
                                       tmp_path / "run",
                                       adapter=_test_adapter(second),
                                       run_code_sha="0" * 40)
    assert "already exists" in str(excinfo.value)
    assert second.calls == 0, "a rerun must not execute a single fit"


# --- this PR executes nothing ------------------------------------------------


def test_AUTHZ_this_pr_executes_zero_real_em(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    report = H.run_full_preflight()
    assert report["em_fits_executed"] == 0
    assert report["real_full_fits_executed"] == 0
    assert report["full_execution_authorization_present"] is False
    assert report["trusted_full_main_sha_present"] is True
    assert report["artifact_directory_exists"] is False
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    assert not H.FULL_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()


def test_AUTHZ_the_suite_never_invokes_the_production_full_command():
    """No test drives `--full --allow-em` into the real workflow.

    Every such test installs ``_block_full_production_execution`` first, and an
    autouse fixture forbids constructing the real EM adapter at all.  Both nets
    stay in place even though the authorization gate is closed again.
    """

    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = _ast.parse(source)
    lines = source.splitlines()
    offenders = []
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name.startswith("test_"):
            body = "\n".join(lines[node.lineno - 1: node.end_lineno])
            drives_full = ('"--full"' in body and '"--allow-em"' in body
                           and '"--confirm-k-true-sweep"' in body) \
                or "H.run_real_full(" in body or "H._run_production_full_execution(" in body
            if drives_full and "_block_full_production_execution(" not in body \
                    and "--out-dir" not in body:
                offenders.append(node.name)
    assert offenders == [], offenders
    assert "_forbid_the_real_em_adapter" in source


# ===========================================================================
# Issue #59 Phase 8b S3-D: the persisted full manifest carries GLOBAL indices
# ===========================================================================
#
# FINAL PRE-EXECUTION REVIEW, HIGH.  `build_manifest` is a per-estimand builder
# whose fit_index is local (1..168).  Concatenating A and B at CSV-write time
# persisted `manifest.csv` fit_index 1..168, 1..168 while the execution counter
# and `full_fit_results.csv` carried the global 1..336, so the two artifacts
# disagreed about the identity of every B fit.  Fixed by assigning the global
# index in `build_full_manifests`, enforcing it in the zero-EM preflight
# validator and re-deriving it independently in the artifact auditor.


def _global_manifests():
    return H.build_full_manifests()


def _pre_fix_manifests():
    """Exactly what the PRE-FIX builder produced: two local 1..168 blocks."""

    return {estimand: H.build_manifest(estimand) for estimand in ("A", "B")}


def _reindex(manifests, indices):
    """Rebuild the flattened manifest with an arbitrary fit_index sequence."""

    flattened = [row for estimand in ("A", "B") for row in manifests[estimand]]
    assert len(flattened) == len(indices)
    rebuilt = [dataclasses.replace(row, fit_index=index)
               for row, index in zip(flattened, indices)]
    return {"A": rebuilt[:168], "B": rebuilt[168:]}


# --- the builders ------------------------------------------------------------


def test_MANIFESTINDEX_per_estimand_builder_keeps_local_indices():
    """§4: `build_manifest` semantics are deliberately unchanged."""

    for estimand in ("A", "B"):
        rows = H.build_manifest(estimand)
        assert len(rows) == 168
        assert [row.fit_index for row in rows] == list(range(1, 169))


def test_MANIFESTINDEX_full_manifests_are_globally_indexed():
    manifests = _global_manifests()
    assert [row.fit_index for row in manifests["A"]] == list(range(1, 169))
    assert [row.fit_index for row in manifests["B"]] == list(range(169, 337))
    flattened = H.flatten_full_manifests(manifests)
    assert [row.fit_index for row in flattened] == list(range(1, 337))
    assert [row.estimand for row in flattened] == ["A"] * 168 + ["B"] * 168
    assert all(row.k_true != H.ANCHOR_K_TRUE for row in flattened)
    keys = [(row.estimand, row.k_true, row.replicate, row.k, row.start)
            for row in flattened]
    assert len(set(keys)) == 336
    assert keys == [(estimand, k_true, replicate, k, start)
                    for estimand in ("A", "B")
                    for k_true in (1, 2, 4, 5)
                    for replicate in (1, 2, 3)
                    for k in (1, 2, 3, 4, 5, 6, 7)
                    for start in (1, 2)]


def test_MANIFESTINDEX_global_index_is_independent_of_the_mask_arguments():
    anchors = H.read_phase7e_anchor_masks()
    masks = {r: H.build_split_record(1, r) for r in H.REPLICATES}
    manifests = H.build_full_manifests(masks=masks, anchors=anchors)
    assert [row.fit_index for row in H.flatten_full_manifests(manifests)] \
        == list(range(1, 337))


def test_MANIFESTINDEX_builder_asserts_the_frozen_estimand_order(monkeypatch):
    assert H.FULL_ESTIMAND_ORDER == ("A", "B")
    monkeypatch.setattr(H, "ESTIMANDS", "A")
    with pytest.raises(HarnessStop) as excinfo:
        H.build_full_manifests()
    assert "frozen full execution order" in str(excinfo.value)


# --- the zero-EM preflight validator ----------------------------------------


def test_MANIFESTINDEX_validator_rejects_the_reset_at_the_ab_boundary():
    """The exact defect the final pre-execution review found."""

    manifests = _pre_fix_manifests()
    assert [row.fit_index for row in manifests["A"]] == list(range(1, 169))
    assert [row.fit_index for row in manifests["B"]] == list(range(1, 169))
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_full_manifests(manifests)
    assert "fit_index" in str(excinfo.value)


@pytest.mark.parametrize("indices, label", [
    (list(range(2, 338)), "off_by_one"),
    (list(range(1, 168)) + list(range(169, 338)), "gap_at_168"),
    (list(range(1, 169)) + list(range(1, 169)), "reset_at_b"),
])
def test_MANIFESTINDEX_validator_rejects_malformed_sequences(indices, label):
    with pytest.raises(HarnessStop):
        H.validate_full_manifests(_reindex(_global_manifests(), indices))


def test_MANIFESTINDEX_validator_rejects_a_duplicate_with_one_missing():
    indices = list(range(1, 337))
    indices[4] = 4                       # row 5 duplicates row 4; 5 disappears
    with pytest.raises(HarnessStop) as excinfo:
        H.validate_full_manifests(_reindex(_global_manifests(), indices))
    assert "duplicate" in str(excinfo.value) or "fit_index" in str(excinfo.value)


def test_MANIFESTINDEX_validator_rejects_a_correct_set_in_shuffled_rows():
    """§7: the set alone is not enough -- the row position must agree."""

    indices = list(range(1, 337))
    indices[0], indices[-1] = indices[-1], indices[0]
    assert sorted(indices) == list(range(1, 337))
    with pytest.raises(HarnessStop):
        H.validate_full_manifests(_reindex(_global_manifests(), indices))


def test_MANIFESTINDEX_validator_accepts_the_fixed_manifest():
    report = H.validate_full_manifests(_global_manifests())
    assert report["global_fit_index_range"] == [1, 336]
    assert report["fit_index_range_by_estimand"] == {"A": [1, 168], "B": [169, 336]}
    assert report["fits_per_estimand"] == {"A": 168, "B": 168}
    assert report["total_fits"] == 336


def test_MANIFESTINDEX_preflight_reports_the_global_range_with_zero_em(monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    report = H.run_full_preflight()
    assert report["em_fits_executed"] == 0 and report["real_full_fits_executed"] == 0
    assert report["manifest"]["global_fit_index_range"] == [1, 336]
    assert report["manifest"]["fit_index_range_by_estimand"] == {"A": [1, 168],
                                                                "B": [169, 336]}
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0


# --- the persisted artifact --------------------------------------------------


def _manifest_rows(directory):
    with (directory / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_MANIFESTINDEX_persisted_manifest_csv_is_global(tmp_path):
    _run_full_fake(tmp_path / "run")
    rows = _manifest_rows(tmp_path / "run")
    assert len(rows) == 336
    assert [row["fit_index"] for row in rows] == [str(i) for i in range(1, 337)]
    assert [row["estimand"] for row in rows] == ["A"] * 168 + ["B"] * 168
    assert [row["fit_index"] for row in rows if row["estimand"] == "A"] \
        == [str(i) for i in range(1, 169)]
    assert [row["fit_index"] for row in rows if row["estimand"] == "B"] \
        == [str(i) for i in range(169, 337)]
    assert all(row["K_TRUE"] != "3" for row in rows)


def test_MANIFESTINDEX_persisted_manifest_matches_full_fit_results(tmp_path):
    """§8: planned fit N and executed fit N are the same fit."""

    _run_full_fake(tmp_path / "run")
    manifest = _manifest_rows(tmp_path / "run")
    with (tmp_path / "run" / "full_fit_results.csv").open(encoding="utf-8",
                                                          newline="") as handle:
        results = list(csv.DictReader(handle))
    assert len(manifest) == len(results) == 336
    identity = ("fit_index", "estimand", "role", "K_TRUE", "replicate", "K", "start",
                "data_seed", "split_seed", "model_seed", "mask_group_id")
    for position, (planned, executed) in enumerate(zip(manifest, results), start=1):
        for column in identity:
            assert planned[column] == executed[column], (position, column)


def test_MANIFESTINDEX_executor_binds_the_manifest_index_to_the_execution_index(tmp_path):
    """The cell manifest carries the global index the counter hands out."""

    cell = H.prepare_full_cell(H._make_test_full_authorization(), "B", 5, 3,
                              test_only=True)
    assert [row.fit_index for row in cell.manifest] == list(range(323, 337))
    body = _executable_body(H._run_full_cell)
    assert "row.fit_index == fit_index" in body


# --- the INDEPENDENT artifact auditor ---------------------------------------


def _patch_manifest_index_column(directory, values):
    path = directory / "manifest.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    column = header.index("fit_index")
    assert len(lines) - 1 == len(values)
    for offset, value in enumerate(values, start=1):
        cells = lines[offset].split(",")
        cells[column] = str(value)
        lines[offset] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _full_manifest_checks(auditor):
    return sorted({finding.check for finding in auditor.blockers
                   if finding.check.startswith("full_manifest_global")
                   or finding.check == "full_manifest_result_alignment"})


def test_MANIFESTINDEX_auditor_accepts_the_fixed_manifest(tmp_path):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    auditor = A.audit_full_run_dir(directory)
    assert _full_manifest_checks(auditor) == []


def test_MANIFESTINDEX_auditor_rejects_the_reset_at_the_ab_boundary(tmp_path):
    """The pre-fix artifact must fail the INDEPENDENT audit."""

    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    _patch_manifest_index_column(directory, list(range(1, 169)) * 2)
    auditor = A.audit_full_run_dir(directory)
    checks = _full_manifest_checks(auditor)
    assert "full_manifest_global_index_sequence" in checks, checks
    assert "full_manifest_global_index_position" in checks, checks


@pytest.mark.parametrize("values, label", [
    (list(range(2, 338)), "off_by_one"),
    (list(range(1, 168)) + list(range(169, 338)), "gap_at_168"),
])
def test_MANIFESTINDEX_auditor_rejects_malformed_sequences(tmp_path, values, label):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    _patch_manifest_index_column(directory, values)
    auditor = A.audit_full_run_dir(directory)
    assert "full_manifest_global_index_sequence" in _full_manifest_checks(auditor)


def test_MANIFESTINDEX_auditor_rejects_a_shuffled_global_index(tmp_path):
    """A correct SET in the wrong rows is still a BLOCKER."""

    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    values = list(range(1, 337))
    values[0], values[-1] = values[-1], values[0]
    assert sorted(values) == list(range(1, 337))
    _patch_manifest_index_column(directory, values)
    auditor = A.audit_full_run_dir(directory)
    assert "full_manifest_global_index_position" in _full_manifest_checks(auditor)


def test_MANIFESTINDEX_auditor_rejects_a_duplicate_with_one_missing(tmp_path):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    values = list(range(1, 337))
    values[4] = 4
    _patch_manifest_index_column(directory, values)
    auditor = A.audit_full_run_dir(directory)
    checks = _full_manifest_checks(auditor)
    assert "full_manifest_global_index_duplicate" in checks, checks


def test_MANIFESTINDEX_auditor_rejects_a_non_integer_index(tmp_path):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    values = [str(i) for i in range(1, 337)]
    values[0] = "1.0"
    _patch_manifest_index_column(directory, values)
    auditor = A.audit_full_run_dir(directory)
    assert "full_manifest_global_index_parse" in _full_manifest_checks(auditor)


def test_MANIFESTINDEX_auditor_detects_a_manifest_result_disagreement(tmp_path):
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    path = directory / "manifest.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    cells = lines[1].split(",")
    cells[header.index("K")] = "7"
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    auditor = A.audit_full_run_dir(directory)
    assert "full_manifest_result_alignment" in _full_manifest_checks(auditor)


def test_MANIFESTINDEX_auditor_rebuilds_the_expected_order_independently():
    """§7: the auditor never imports the harness."""

    source = pathlib.Path(A.__file__).read_text(encoding="utf-8")
    assert "import run_k_true_robustness_sweep" not in source
    assert "from run_k_true_robustness_sweep" not in source
    order = A.expected_ordered_full_keys()
    assert len(order) == 336 and len(set(order)) == 336
    assert order[0] == ("A", 1, 1, 1, 1)
    assert order[167][0] == "A" and order[168][0] == "B"
    assert order[-1] == ("B", 5, 3, 7, 2)
    body = _executable_body(A.audit_full_manifest_global_index)
    assert "expected_ordered_full_keys()" in body


def test_MANIFESTINDEX_zero_em_is_maintained(tmp_path, monkeypatch):
    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    _run_full_fake(tmp_path / "run")
    directory = _promote_full_fixture(tmp_path / "run", tmp_path / "real")
    A.audit_full_run_dir(directory)
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    assert H.current_full_execution_authorization() is None
    assert not H.FULL_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()


# ===========================================================================
# Issue #59 Phase 8b S3-E: the REVISED reviewed baseline is bound, gate closed
# ===========================================================================
#
# PR #63 fixed the full manifest (global fit_index 1..336) and was independently
# reviewed and human-merged.  S3-E binds that exact merge SHA as the CURRENT
# role-2 reviewed full-execution baseline.  It grants nothing: no production
# FullExecutionAuthorization is created, the historical S3-C approval is not
# transferred, no real EM runs, and the frozen protocol hash is untouched.

REVISED_REVIEWED_FULL_MAIN_SHA = "02ef35add45036975162b6a267f6428c3b380459"
HISTORICAL_S3C_HUMAN_APPROVAL_COMMENT_ID = 5511177444


def test_S3E_current_role2_is_the_revised_reviewed_baseline():
    """4/10: one committed literal, and every accessor returns exactly it."""

    assert H.REVIEWED_FULL_EXECUTION_MAIN_SHA == REVISED_REVIEWED_FULL_MAIN_SHA
    assert H.current_expected_full_main_sha() == REVISED_REVIEWED_FULL_MAIN_SHA
    assert H.trusted_full_main_sha_for(test_only=False) == REVISED_REVIEWED_FULL_MAIN_SHA
    H._require_full_commit_sha(H.current_expected_full_main_sha(), "reviewed full SHA")
    # the test-only lineage is untouched and can never stand in for production
    assert H.trusted_full_main_sha_for(test_only=True) == H._FULL_TEST_EXPECTED_MAIN_SHA
    assert H.trusted_full_main_sha_for(test_only=True) != REVISED_REVIEWED_FULL_MAIN_SHA


def test_S3E_historical_role2_is_preserved_and_distinct():
    """3: history is kept explicitly, never silently rewritten."""

    assert (H.HISTORICAL_S3C_REVIEWED_FULL_EXECUTION_MAIN_SHA
            == HISTORICAL_REVIEWED_FULL_MAIN_SHA)
    assert (H.HISTORICAL_S3C_REVIEWED_FULL_EXECUTION_MAIN_SHA
            != H.REVIEWED_FULL_EXECUTION_MAIN_SHA)
    assert (H.current_expected_full_main_sha()
            != H.HISTORICAL_S3C_REVIEWED_FULL_EXECUTION_MAIN_SHA)
    # the three roles stay separate: role 1 is unchanged by this rebinding
    assert H.APPROVED_SCIENTIFIC_MAIN_SHA == SCIENTIFIC_BASELINE_SHA
    assert H.REVIEWED_FULL_EXECUTION_MAIN_SHA != H.APPROVED_SCIENTIFIC_MAIN_SHA
    assert (H.HISTORICAL_S3C_REVIEWED_FULL_EXECUTION_MAIN_SHA
            != H.APPROVED_SCIENTIFIC_MAIN_SHA)


def test_S3E_the_revised_literal_is_committed_not_derived():
    """4: role 2 is never read from HEAD, git, the env, the CLI or a config."""

    tree = _ast.parse(_textwrap.dedent(
        _inspect.getsource(H.current_expected_full_main_sha)))
    calls = [getattr(n.func, "id", getattr(n.func, "attr", "?"))
             for n in _ast.walk(tree) if isinstance(n, _ast.Call)]
    assert calls == [], f"the trusted source must not call anything: {calls}"
    names = {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
    assert names - {"str"} == {"REVIEWED_FULL_EXECUTION_MAIN_SHA"}
    body = _executable_body(H.current_expected_full_main_sha)
    for forbidden in ("os.", "environ", "getenv", "argv", "sys.", "subprocess", "git",
                      "rev-parse", "current_run_code_sha", "Path", "open(", "json",
                      "config", "branch", "HEAD"):
        assert forbidden not in body, forbidden
    module_source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    assert (f'REVIEWED_FULL_EXECUTION_MAIN_SHA = "{REVISED_REVIEWED_FULL_MAIN_SHA}"'
            in module_source)


def test_S3E_env_and_cli_cannot_rebind_role2(monkeypatch):
    for name in ("REVIEWED_FULL_EXECUTION_MAIN_SHA", "PHASE8B_FULL_MAIN_SHA",
                 "APPROVED_MAIN_SHA", "PHASE8B_FULL_AUTHORIZED"):
        monkeypatch.setenv(name, "9" * 40)
    monkeypatch.setattr(sys, "argv", ["run", "--full", "--allow-em", "--approve-full"])
    assert H.current_expected_full_main_sha() == REVISED_REVIEWED_FULL_MAIN_SHA
    assert H.current_full_execution_authorization() is None


def test_S3E_production_authorization_is_still_absent():
    """5: binding a reviewed baseline is not an approval."""

    assert H.current_full_execution_authorization() is None
    body = _executable_body(H.current_full_execution_authorization)
    assert body.strip() == "return None"
    for forbidden in ("FullExecutionAuthorization(", "human_full_approval",
                      "independent_review_pass", "_FULL_EXECUTION_AUTHORITY"):
        assert forbidden not in body, forbidden


def test_S3E_the_historical_approval_is_never_reused():
    """5: comment 5511177444 must not map to an ACTIVE production record."""

    assert (H.FULL_HUMAN_AUTHORIZATION_ISSUE_COMMENT_ID
            == HISTORICAL_S3C_HUMAN_APPROVAL_COMMENT_ID)
    module_source = pathlib.Path(H.__file__).read_text(encoding="utf-8")
    # exactly one construction site remains, and it is the test-only factory
    constructions = [line.strip() for line in module_source.splitlines()
                     if "FullExecutionAuthorization(" in line
                     and "type(" not in line
                     and "is FullExecutionAuthorization" not in line]
    assert len(constructions) == 1, constructions
    assert ("_FULL_TEST_AUTHORITY"
            in module_source.split("FullExecutionAuthorization(")[-1][:400])
    # the production authority sentinel is never attached to a committed record
    authority_uses = [line.strip() for line in module_source.splitlines()
                      if "_FULL_EXECUTION_AUTHORITY" in line
                      and not line.strip().startswith("#")]
    assert authority_uses, "the sentinel must still exist"
    for line in authority_uses:
        assert ("_FULL_TEST_AUTHORITY if test_only else" in line
                or line.startswith("_FULL_EXECUTION_AUTHORITY = object()")), line
    doc = H.current_full_execution_authorization.__doc__
    assert "STALE" in doc and "None" in doc


@pytest.mark.parametrize("argv", [
    ["--full", "--allow-em", "--confirm-k-true-sweep"],
    ["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "AB"],
])
def test_S3E_full_still_stops_before_any_adapter(argv, monkeypatch, tmp_path):
    """5/13: the CLI must stop before _run_production_full_execution."""

    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    monkeypatch.setattr(H, "FULL_ARTIFACT_DIR", tmp_path / "frozen_full")
    reached = _block_full_production_execution(monkeypatch)
    with pytest.raises(HarnessStop) as excinfo:
        H.main(argv)
    message = str(excinfo.value)
    assert "not authorized" in message
    assert "NO committed FullExecutionAuthorization record exists" in message
    assert "STALE" in message
    assert str(HISTORICAL_S3C_HUMAN_APPROVAL_COMMENT_ID) in message
    assert HISTORICAL_REVIEWED_FULL_MAIN_SHA in message
    assert reached == [], "the production full workflow is never reached"
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert not (tmp_path / "frozen_full").exists()
    assert "em_runner" not in sys.modules


def test_S3E_auditor_expects_the_revised_baseline_independently():
    """6: the auditor holds its own literal and never imports the runner."""

    assert (A.EXPECTED_REVIEWED_FULL_EXECUTION_BASELINE_SHA
            == REVISED_REVIEWED_FULL_MAIN_SHA)
    assert (A.HISTORICAL_S3C_REVIEWED_FULL_EXECUTION_BASELINE_SHA
            == HISTORICAL_REVIEWED_FULL_MAIN_SHA)
    assert (A.EXPECTED_REVIEWED_FULL_EXECUTION_BASELINE_SHA
            != A.HISTORICAL_S3C_REVIEWED_FULL_EXECUTION_BASELINE_SHA)
    assert (A.EXPECTED_REVIEWED_FULL_EXECUTION_BASELINE_SHA
            != A.EXPECTED_SCIENTIFIC_BASELINE_SHA)
    source = pathlib.Path(A.__file__).read_text(encoding="utf-8")
    assert "import run_k_true_robustness_sweep" not in source
    assert f'"{REVISED_REVIEWED_FULL_MAIN_SHA}"' in source


def test_S3E_protocol_hash_is_unchanged_by_the_rebinding():
    """7: rebinding reviewed code provenance is not a protocol change."""

    assert (H.full_protocol_hash()
            == "2d19c5fe6edadd0823925ed7dd051cb27837bccf51d5102e0bcee53271654eb9")
    assert A.EXPECTED_FULL_PROTOCOL_HASH == H.full_protocol_hash()
    # the reviewed baseline is provenance, never an input to the frozen protocol
    flat = json.dumps(H.full_protocol_config(), sort_keys=True, default=str)
    assert REVISED_REVIEWED_FULL_MAIN_SHA not in flat
    assert HISTORICAL_REVIEWED_FULL_MAIN_SHA not in flat


def test_S3E_full_manifest_global_indices_are_intact():
    """8: the PR #63 HIGH fix is untouched by this rebinding."""

    manifests = H.build_full_manifests()
    flattened = H.flatten_full_manifests(manifests)
    assert len(flattened) == 336
    assert [row.fit_index for row in flattened] == list(range(1, 337))
    assert len({row.fit_index for row in flattened}) == 336
    assert [row.fit_index for row in manifests["A"]] == list(range(1, 169))
    assert [row.fit_index for row in manifests["B"]] == list(range(169, 337))
    assert all(row.k_true != H.ANCHOR_K_TRUE for row in flattened)
    # the executor still binds the persisted index to the executed index
    cell_body = _executable_body(H._run_full_cell)
    assert "row.fit_index == fit_index" in cell_body


def test_S3E_zero_em_state_is_unchanged(tmp_path, monkeypatch):
    """12: the rebinding executes nothing."""

    _AdapterTripwire.reset()
    monkeypatch.setattr(H, "AuthorizedEMFitAdapter", _AdapterTripwire)
    report = H.run_full_preflight()
    assert report["em_fits_executed"] == 0
    assert report["real_full_fits_executed"] == 0
    assert report["expected_full_fits"] == 336
    assert report["trusted_full_main_sha_present"] is True
    assert report["full_execution_authorization_present"] is False
    assert report["phase7e_rerun_fits"] == 0
    assert report["artifact_directory_exists"] is False
    assert _AdapterTripwire.constructions == 0 and _AdapterTripwire.fits == 0
    assert "em_runner" not in sys.modules
    assert not H.FULL_ARTIFACT_DIR.exists()
    _assert_no_new_production_artifacts()
