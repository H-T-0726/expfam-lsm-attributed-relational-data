"""Tests for AF3 / S0 selector re-scoring (no MCEM, no result file needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af3_selector_rescoring as a3  # noqa: E402


def test_noise_is_deterministic_and_seed_dependent():
    ds = a3.af2.generate(0)
    Z = np.asarray(ds.Z, float)
    inst = [i for i in a3.arm_instances() if i["arm"] == "Z2"]
    a = a3.latent_samples(inst[0], 0, Z)[0]
    b = a3.latent_samples(inst[0], 0, Z)[0]
    c = a3.latent_samples(inst[1], 0, Z)[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (a3.N, a3.K, a3.L_SAMPLES)


def test_true_arm_is_true_z():
    ds = a3.af2.generate(0)
    Zs, init = a3.latent_samples(a3.arm_instances()[0], 0, np.asarray(ds.Z, float))
    assert init is None and np.array_equal(Zs[:, :, 0], ds.Z)


def test_score_invariant_to_rotation_and_scale_of_z():
    """Q_j is invariant to an invertible linear map of Z (no intercept)."""
    from family_selection_multisupport import score_candidate
    reg, _ = a3.registry_and_priority()
    ds = a3.af2.generate(0)
    Zs = np.asarray(ds.Z, float)[:, :, None]
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))
    Zr = np.einsum("nkl,kj->njl", Zs, 1.7 * Q)
    for c in a3.CANDS:
        s0 = score_candidate(ds.X[:, 6], Zs, reg[c], np.zeros(3), 6, route=a3.ROUTE).score
        s1 = score_candidate(ds.X[:, 6], Zr, reg[c], np.zeros(3), 6, route=a3.ROUTE).score
        assert abs(s0 - s1) < 1e-7


def test_per_observation_contributions_sum_to_score_and_margin_sign():
    reg, prio = a3.registry_and_priority()
    ds = a3.af2.generate(0)
    inst = a3.arm_instances()[0]
    cand, cols, _ = a3.score_instance(inst, 0, ds, reg, prio)
    assert len(cols) == len(a3.COUNT_COLS) and len(cand) == 2 * len(a3.COUNT_COLS)
    for r in cand:
        if r["status"] == "OK":
            assert r["per_obs_sum_check_abs"] < 1e-8
    for r in cols:
        if r["status"] == "SELECTED":
            assert (r["signed_margin"] > 0) == r["correct"] or r["signed_margin"] == 0
            assert abs(r["signed_margin"] - 2 * a3.N * r["per_obs_diff_mean"]) < 1e-7


def test_incumbent_start_uses_saved_loading():
    if not (a3.AF2_STATE / "rep01_dense_oracle.npz").exists():
        pytest.skip("AF2 states not available")
    ds = a3.af2.generate(0)
    inst = [i for i in a3.arm_instances() if i["arm"] == "Z3" and i["start"] == "incumbent"][0]
    Zs, F = a3.latent_samples(inst, 0, np.asarray(ds.Z, float))
    st = a3.load_npz(0, inst["regime"], "oracle")
    assert np.array_equal(F, st["F"]) and np.array_equal(Zs, st["Zsamples"])


def test_protocol_is_stable_and_budget_matches_manifest():
    p = a3.protocol()
    assert a3.protocol_sha256() == a3.protocol_sha256()
    assert p["mcem_fits"] == 0
    assert p["budget"]["candidate_optimisations"] == len(a3.arm_instances()) * 20 * 12
    assert len({i["instance"] for i in a3.arm_instances()}) == len(a3.arm_instances())
