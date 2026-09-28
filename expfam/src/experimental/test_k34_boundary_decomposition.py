"""Focused zero-EM tests for the Phase 9Y read-only decomposition (Issue #124)."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import k34_boundary_decomposition as kd                           # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401

REPO = kd.REPO_ROOT


def _rows(kt):
    combined = kd._read(REPO, kd.SOURCES["combined_rows"])
    k_hat = {r["replicate"]: (int(r["K_hat_Lap"]), int(r["K_hat_Q"]))
             for r in combined if r["K_true"] == str(kt)}
    lap = [r for r in kd._read(REPO, kd.SOURCES[f"ktrue{kt}_laplace"])
           if r["replicate"] in kd.REPLICATES]
    cq = [r for r in kd._read(REPO, kd.SOURCES[f"ktrue{kt}_cq"])
          if r["replicate"] in kd.REPLICATES]
    return kd.decompose(lap, cq, k_hat), lap


def test_1_no_inference_imports():
    tree = ast.parse(Path(kd.__file__).read_text("utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names} | {n.module for n in ast.walk(tree)
                                     if isinstance(n, ast.ImportFrom)}
    assert imported.isdisjoint({
        "em_runner", "family_selection", "laplace_k_criterion",
        "model_dual_expfam_consistent", "run_joint_family_k_selection",
        "run_family_selection_pilot", "theta_stationarity_diagnostic",
        "multistep_local_pilot"})


def test_2_exact_replicates():
    rows, _ = _rows(4)
    assert [r["replicate"] for r in rows] == [f"rep{i:02d}" for i in range(1, 11)]


def test_3_sign_convention():
    rows, lap = _rows(4)
    c = {(r["replicate"], int(r["k"])): float(r["C_Lap"]) for r in lap}
    for r in rows:
        assert r["delta34_Lap"] == c[(r["replicate"], 4)] - c[(r["replicate"], 3)]
        assert (r["group_Lap"] == "Lap_K4_over_K3") == (r["delta34_Lap"] < 0)
        assert (r["margin_Lap"] > 0) == (r["delta34_Lap"] < 0)
        assert (r["margin_Q"] > 0) == (r["delta34_Q"] < 0)


def test_4_5_6_reconstruction_within_tolerance():
    for kt in (3, 4):
        rows, _ = _rows(kt)
        assert all(r["available"] for r in rows)
        assert max(r["residual_Lap"] for r in rows) <= kd.RESIDUAL_TOL
        assert max(r["residual_Q"] for r in rows) <= kd.RESIDUAL_TOL
        for r in rows:
            assert abs(r["volume_increment_34"] - r["logdet_increment_34"]
                       - r["zhat_sq_increment_34"]) <= 1e-9


def test_7_parameter_increment_from_actual_d_K():
    import laplace_k_criterion as lk
    import run_laplace_pilot as lp
    for k in range(1, 6):
        for g in (0, 3):
            assert kd.d_K(k, 12, g) == lp.d_K(k, 12, g) == \
                lk.loading_parameter_count(k, 12) + g
    rows, _ = _rows(4)
    for r in rows:
        assert abs(r["param_increment_Lap"] - r["param_increment_Lap_from_dK"]) <= 1e-10
        assert abs(r["param_increment_Q"] - r["param_increment_Q_from_num_params"]) <= 1e-10


def _hashes():
    return {n: hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
            for n, rel in kd.SOURCES.items()}


def test_8_9_sources_read_only_and_unchanged():
    before = _hashes()
    _rows(3)
    _rows(4)
    assert _hashes() == before
    assert set(kd.verify_sources()) == set(kd.SOURCES)
