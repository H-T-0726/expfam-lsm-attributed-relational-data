"""Zero-EM focused tests for the Phase 9U paired generator (Issue #116)."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import paired_attribute_scale as pa                               # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401

SEED = 1001001


def test_1_z_shared():
    s = pa.generate_all(SEED)
    assert all(np.array_equal(s[c].Z, s["weak"].Z) for c in pa.CONDITIONS)


def test_2_orientation_shared_and_exact_scale():
    s = pa.generate_all(SEED)
    q = s["weak"].Q
    assert np.allclose(q.T @ q, np.eye(pa.K_TRUE), atol=1e-12)
    for c, f in pa.F_SCALES.items():
        assert np.array_equal(s[c].Q, q)
        assert np.array_equal(s[c].F, q * f)
        assert math.isclose(np.mean(np.sum(s[c].F ** 2, axis=1)),
                            f * f * pa.K_TRUE / pa.D, rel_tol=1e-12)
    assert [f * f * pa.K_TRUE / pa.D for f in pa.F_SCALES.values()] == \
        [0.25, pytest_approx(0.5), 1.0]


def pytest_approx(v):
    import pytest
    return pytest.approx(v, abs=1e-15)


def test_3_y_shared():
    s = pa.generate_all(SEED)
    assert all(np.array_equal(s[c].Y, s["weak"].Y) for c in pa.CONDITIONS)
    assert not np.array_equal(s["weak"].X, s["strong"].X)


def test_4_column_streams_isolated():
    d1 = pa._draws(SEED)
    d2 = pa._draws(SEED)
    for l in range(pa.D):
        assert np.array_equal(d1["column_base"][l], d2["column_base"][l])
    # each column / Y has its own child: drawing column 0 differently never
    # touches another component
    ch = pa.streams(SEED)
    assert len({tuple(c.spawn_key) for c in ch}) == pa.N_CHILDREN
    other = np.random.default_rng(ch[2 + 3]).random(pa.N)   # Bernoulli col 3
    assert np.array_equal(other, d1["column_base"][3])
    y = np.random.default_rng(ch[2 + pa.D]).random(pa.N * (pa.N - 1) // 2)
    assert np.array_equal(y, d1["V"])


def test_5_supports_and_poisson_inversion():
    for c, ds in pa.generate_all(SEED).items():
        for l, fam in enumerate(pa.FAMILY_X):
            assert pa._support_ok(ds.X[:, l], fam)
        assert ds.poisson_inversion_ok
    u = np.array([0.0, 0.3, 0.999999])
    x, ok = pa.poisson_inverse_cdf(u, np.array([2.0, 2.0, 2.0]))
    assert ok and x[0] == 0.0 and x[1] == 1.0


def test_6_deterministic_repeat():
    a, b = pa.generate_all(SEED), pa.generate_all(SEED)
    for c in pa.CONDITIONS:
        assert np.array_equal(a[c].X, b[c].X)
        assert np.array_equal(a[c].Y, b[c].Y)


def test_7_no_inference_imports():
    tree = ast.parse(Path(pa.__file__).read_text("utf-8"))
    top = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)} | {
        a.name for n in tree.body if isinstance(n, ast.Import)
        for a in n.names}
    assert top.isdisjoint({"family_selection", "em_runner",
                           "model_dual_expfam_consistent",
                           "laplace_k_criterion", "run_joint_family_k_selection"})


def test_8_future_base_reuse_decision():
    same = [{"Z_identical": True, "F_identical": True, "X_identical": True,
             "Y_identical": True}] * 20
    diff = same[:19] + [dict(same[0], Y_identical=False)]
    assert pa.base_reuse_decision(same) == {
        "historical_base_reusable": True,
        "future_base_rerun_required": False, "future_new_em_cap": 400}
    assert pa.base_reuse_decision(diff) == {
        "historical_base_reusable": False,
        "future_base_rerun_required": True, "future_new_em_cap": 600}
