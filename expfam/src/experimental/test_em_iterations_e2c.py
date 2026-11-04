"""Focused tests for Issue #142 E2c: checkpoint recording contract and plan."""

from __future__ import annotations

import numpy as np

import run_em_iterations_e2c as e2c
from em_runner import run_em_experimental


def _fit(num_iter, **kw):
    rng = np.random.default_rng(0)
    n, d = 20, 4
    X = rng.normal(size=(n, d))
    Y = np.triu((rng.random((n, n)) < 0.3).astype(float), 1)
    return run_em_experimental(X, Y + Y.T, family_x="gaussian", family_y="bernoulli", k=2,
                               L=2, num_iter=num_iter, seed=3, compute_strict_Q=False,
                               numerics_mode="consistent", failure_policy="fail_fast", **kw)


def _same(a, b):
    return all(np.array_equal(np.asarray(a[q]), np.asarray(b[q]))
               for q in ("Z_samples", "F", "sigma", "w0", "w"))


def test_checkpoints_do_not_change_the_fit_and_are_opt_in():
    plain = _fit(4)
    rec = _fit(4, checkpoint_iters={2, 4})
    assert _same(plain, rec)
    assert "checkpoints" not in plain and "theta_history" not in plain
    assert [h["iteration"] for h in rec["theta_history"]] == [1, 2, 3, 4]
    assert sorted(rec["checkpoints"]) == [2, 4]


def test_checkpoint_equals_shorter_run_and_final_state():
    rec = _fit(4, checkpoint_iters={2, 4})
    short = _fit(2)
    ck2, ck4 = rec["checkpoints"][2], rec["checkpoints"][4]
    assert _same(short, ck2)
    assert _same(rec, ck4)


def test_plan_is_aa_and_ba_with_shared_mc_seed():
    plan = e2c.planned_fits()
    assert len(plan) == 80 and {p["regime"] for p in plan} == {"sparse"}
    for p in plan:
        assert p["mc_seed"] == 1343000 + p["r"]
        assert p["init_seed"] == {"AA": 1343000, "BA": 1344000}[p["cell"]] + p["r"]
    assert e2c.CHECKPOINTS == (8, 16, 24, 32, 40) and e2c.NUM_ITER == 40
