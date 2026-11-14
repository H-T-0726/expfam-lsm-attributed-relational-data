"""Tests for AF5 / S2-B (no MCEM; uses AF2 states when present)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af5_local_laplace_selection as a5  # noqa: E402

HAVE = (a5.af4.AF2_STATE / "rep01_dense_oracle.npz").exists()


def _prob(cand, anchor="support_only", j=7, regime="dense"):
    fr = {r["fit_key"]: r for r in a5.af4.read_csv(a5.af4.AF2_FULL / "fit_runs.csv")}
    ds, masks = a5.af2.generate(0), a5.af2.m1.build_masks(0)
    return a5.LocalProblem(ds, masks, regime, anchor, 0, j, cand, fr)


@pytest.mark.skipif(not HAVE, reason="AF2 states not available")
def test_only_target_column_and_family_change():
    p, c = _prob("poisson"), _prob("cmp_nu2p0")
    assert [s.candidate_model for k, s in enumerate(p.model.column_specs) if k != 7] == \
           [s.candidate_model for k, s in enumerate(c.model.column_specs) if k != 7]
    assert c.model.column_specs[7].candidate_model == "cmp_nu2p0"
    assert p.model.column_specs[7].candidate_model == "poisson"
    for key in ("sigma", "w0", "w", "var_z"):
        assert np.array_equal(np.asarray(p.model.params[key]), np.asarray(c.model.params[key]))
    p.set_f(np.ones(3))
    c.set_f(2 * np.ones(3))
    Fp, Fc = p.model.params["F"], c.model.params["F"]
    assert np.array_equal(np.delete(Fp, 7, 0), np.delete(Fc, 7, 0))


@pytest.mark.skipif(not HAVE, reason="AF2 states not available")
def test_inner_mode_reaches_tolerance_and_is_deterministic():
    p = _prob("cmp_nu2p0")
    f = a5.plugin_start(p)
    a = p.laplace(f, Z0=p.Z_anchor)
    b = p.laplace(f, Z0=p.Z_anchor)
    assert a["ok"] and a["inner_grad"] <= a5.INNER_GRAD_TOL and a["S"] == b["S"]


@pytest.mark.skipif(not HAVE, reason="AF2 states not available")
def test_envelope_gradient_matches_fd():
    p = _prob("poisson", anchor="auto_first", j=4)
    f = a5.plugin_start(p)
    v, g = p.joint_phi_and_grad(f)
    fd = np.array([(p.joint_phi_and_grad(f + e)[0] - p.joint_phi_and_grad(f - e)[0]) / 2e-5
                   for e in np.eye(3) * 1e-5])
    assert np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(fd))) < 1e-4


def test_protocol_exploratory_and_zero_mcem():
    p = a5.protocol()
    assert p["mcem_fits"] == 0 and p["study_type"].startswith("EXPLORATORY")
    assert len(p) == len(set(p))          # no duplicated top-level keys (AF4 lesson)
