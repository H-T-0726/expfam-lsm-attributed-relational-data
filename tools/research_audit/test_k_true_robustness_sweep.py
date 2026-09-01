"""Static and adversarial tests for the Phase 8b K_TRUE robustness harness.

No test in this module runs EM.  Fit-bearing paths are exercised only through
their authorization gates, which must refuse.  Artifact-level negative tests
build a synthetic run directory in a temp dir and break exactly one thing.

Test ids follow the implementation-plan contract (T01-T32, A01-A25).
"""

from __future__ import annotations

import csv
import importlib
import math
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


def test_T23b_full_is_refused_even_with_every_flag():
    with pytest.raises(HarnessStop) as excinfo:
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "A"])
    assert "not authorized" in str(excinfo.value)


def test_T23c_smoke_and_canary_are_refused():
    for command in ("--smoke", "--canary"):
        with pytest.raises(HarnessStop):
            H.main([command, "--allow-em"])


def test_T24_estimand_must_match_frozen_set(monkeypatch):
    monkeypatch.setattr(H, "ESTIMANDS", "A")
    with pytest.raises(HarnessStop):
        H.main(["--full", "--allow-em", "--confirm-k-true-sweep", "--estimand", "B"])


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
