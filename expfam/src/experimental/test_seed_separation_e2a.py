"""Focused tests for Issue #142 E2a: seed separation contract and plan."""

from __future__ import annotations

import numpy as np
import pytest

import run_seed_separation_e2a as e2a
import run_z_point_estimator_e1 as e1
from em_runner import run_em_experimental


def _tiny(seed=0):
    rng = np.random.default_rng(seed)
    n, d = 20, 4
    X = rng.normal(size=(n, d))
    Y = np.triu((rng.random((n, n)) < 0.3).astype(float), 1)
    return X, Y + Y.T


def _fit(**kw):
    X, Y = _tiny()
    return run_em_experimental(X, Y, family_x="gaussian", family_y="bernoulli", k=2,
                               L=2, num_iter=2, compute_strict_Q=False,
                               numerics_mode="consistent", failure_policy="fail_fast", **kw)


def _same(a, b):
    return all(np.array_equal(np.asarray(a[q]), np.asarray(b[q]))
               for q in ("Z_samples", "Z_est", "F", "sigma", "w0", "w"))


def test_default_seed_equals_explicit_init_and_mc_seed():
    old = _fit(seed=7)
    new = _fit(seed=7, init_seed=7, mc_seed=7)
    assert _same(old, new)
    assert new["retry_count"] == new["replacement_count"] == new["seed_rescue_count"] == 0


def test_each_seed_changes_the_result():
    base = _fit(seed=7)
    assert not _same(base, _fit(seed=7, mc_seed=8))      # MC only
    assert not _same(base, _fit(seed=7, init_seed=8))    # init only


def test_plan_is_80_new_sparse_fits_with_shared_seeds_across_arms():
    plan = e2a.planned_fits()
    assert len(plan) == 80 and {p["regime"] for p in plan} == {"sparse"}
    assert {p["cell"] for p in plan} == {"AB", "BA"}
    for p in plan:
        a, b = 1343000 + p["r"], 1344000 + p["r"]
        assert (p["init_seed"], p["mc_seed"]) == {"AB": (a, b), "BA": (b, a)}[p["cell"]]
    # reused cells are exactly E1 Part A / Part B (seed = init = mc)
    assert e2a.SEED_BASE == {"A": e1.PARTS["A"], "B": e1.PARTS["B"]}
    for cell, part in e2a.REUSED.items():
        assert e2a.seeds_of(cell, 5) == (e1.PARTS[part] + 5,) * 2
    keys = {(p["cell"], p["replicate"]): set() for p in plan}
    for p in plan:
        keys[(p["cell"], p["replicate"])].add((p["arm"], p["init_seed"], p["mc_seed"]))
    assert all(len(v) == 2 and len({s[1:] for s in v}) == 1 for v in keys.values())


def test_contrasts():
    c = e2a.contrasts({"AA": 0.10, "AB": 0.12, "BA": 0.05, "BB": 0.09})
    assert c["mc_at_initA"] == pytest.approx(0.02)
    assert c["mc_at_initB"] == pytest.approx(0.04)
    assert c["init_at_mcA"] == pytest.approx(-0.05)
    assert c["init_at_mcB"] == pytest.approx(-0.03)
    assert c["interaction"] == pytest.approx(0.01)
