"""Regression tests for the AF8-v3 final fixes (audit W2: outer certificate; W3: NaN aggregation) and the analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8                    # noqa: E402
import run_af8v3_full_integrated_loading as vf             # noqa: E402
import analyze_af8v3_full_integrated_loading as an         # noqa: E402


class FakeL:
    """T(f) = -1/2 (f - m)^T A (f - m) (+ value); `invalid(f)` marks points where S is undefined."""

    def __init__(self, A, m, invalid=lambda f: False, value=0.0, nan_at=lambda f: False):
        self.A, self.m, self.invalid, self.value, self.nan_at = np.asarray(A, float), np.asarray(m, float), invalid, value, nan_at
        self.J = type("J", (), {"pv": 1.0})()

    def T(self, f):
        f = np.asarray(f, float)
        if self.invalid(f):
            return False, float("nan")
        if self.nan_at(f):
            return True, float("nan")
        d = f - self.m
        return True, self.value - 0.5 * float(d @ self.A @ d)


A3 = np.diag([8.0, 10.0, 12.0])


def test_01_quadratic_certified():
    r = vf.certify_outer(FakeL(A3, [0.3, -0.2, 0.1]), np.zeros(3))
    assert r["status"] == "CERT_OK" and vf.start_valid(r)
    assert np.max(np.abs(r["f"] - [0.3, -0.2, 0.1])) < 1e-5 and r["extra_step_fraction"] == 1.0


def test_02_extra_step_invalid_everywhere_is_not_certified():
    """T undefined at f + a p for every checked fraction: v3 returned CERT_OK with gain NaN; now EXTRA_STEP_INVALID."""
    m = np.array([0.3, -0.2, 0.1])
    d = np.array([2e-5, 0.0, 0.0])
    f0 = m + d
    bad = [m, m + 0.5 * d, m + 0.75 * d]          # f0 + a p for a = 1, 1/2, 1/4 (p = -d on the quadratic)

    def invalid(f):
        return min(np.max(np.abs(f - b)) for b in bad) < 1e-8
    r = vf.certify_outer(FakeL(A3, m, invalid), f0)
    assert r["status"] == "EXTRA_STEP_INVALID" and r["extra_step_invalid"] == 3 and not vf.start_valid(r)


def test_03_extra_step_nan_value_not_certified():
    m = np.array([0.0, 0.0, 0.0])
    f0 = np.array([1e-5, 0, 0])
    p_target = m
    L = FakeL(A3, m, nan_at=lambda f: np.max(np.abs(f - p_target)) < 1e-7)
    r = vf.certify_outer(L, f0)
    # the full step lands at m (NaN value): treated as not finite -> half step is used; never CERT_OK with NaN gain
    assert r["status"] != "CERT_OK" or (np.isfinite(r["newton_gain_f"]) and r["extra_step_fraction"] < 1.0)


def test_04_v3_bug_reproduced_and_fixed():
    """direct comparison with the v3 certify_outer on the same fake objective."""
    import run_af8v3_integrated_loading as v3
    m = np.array([0.2, 0.1, -0.1])
    f0 = m + np.array([2e-5, 0, 0])

    def invalid(f):
        return np.max(np.abs(f - m)) < 1e-7          # the exact maximiser (target of the Newton step) is invalid
    old = v3.certify_outer(FakeL(A3, m, invalid), f0)
    new = vf.certify_outer(FakeL(A3, m, invalid), f0)
    assert old["status"] == "CERT_OK" and np.isnan(old["newton_gain_f"])      # the audited defect
    assert new["status"] == "CERT_OK" and np.isfinite(new["newton_gain_f"]) and new["extra_step_fraction"] == 0.5
    assert vf.start_valid(new) and not vf.start_valid({**old, "hess_f": old["hess_f"]})


def test_05_non_pd_and_nonfinite_rejected():
    r = vf.certify_outer(FakeL(np.diag([8.0, -1.0, 12.0]), [0, 0, 0]), np.zeros(3))
    assert r["status"] == "T_STATIONARY_NOT_MAX" and not vf.start_valid(r)
    r = vf.certify_outer(FakeL(A3, [0, 0, 0], nan_at=lambda f: True), np.zeros(3))
    assert r["status"] == "START_INVALID"
    r = vf.certify_outer(FakeL(A3, [0, 0, 0], invalid=lambda f: True), np.zeros(3))
    assert r["status"] == "START_INVALID"


def test_06_fd_nonfinite_rejected():
    calls = {"n": 0}

    def nan_at(f):
        return np.max(np.abs(f)) > 0 and np.isclose(np.max(np.abs(f)), 1e-4)      # NaN at the gradient stencil
    r = vf.certify_outer(FakeL(A3, [0, 0, 0], nan_at=nan_at), np.zeros(3))
    assert r["status"] == "FD_NONFINITE" and not vf.start_valid(r)


def test_07_worst_maps_nan_to_failing_side():
    log = {}
    assert an.worst([1.0, np.nan, 2.0], "max", "a", log) == np.inf and log["a"] == 1
    assert an.worst([1.0, np.nan], "min", "b", log) == -np.inf
    assert an.worst([], "max", "c", log) == np.inf
    assert an.worst([1.0, 3.0, 2.0], "median", "d", log) == 2.0 and log["d"] == 0


def test_08_completeness_detects_missing_and_duplicates():
    exp = an.expected_cells((0,))
    full = exp.copy()
    assert an.completeness(full, (0,))["complete"]
    assert an.completeness(full.iloc[1:], (0,))["missing"] == 1
    d = an.completeness(pd.concat([full, full.iloc[:1]]), (0,))
    assert d["duplicates"] == 1 and not d["complete"]


def test_09_column_states():
    assert an.state(True, True, False, 0.5, 0.01) == "DECIDED"
    assert an.state(True, True, False, 0.005, 0.01) == "NUMERIC_UNDECIDED"
    assert an.state(True, False, False, np.nan, 0.01) == "ONE_FAILED"
    assert an.state(False, False, False, np.nan, 0.01) == "BOTH_FAILED"
    assert an.state(True, True, True, 0.5, 0.01) == "MISSING"


def test_10_denominators_and_misselection():
    s = pd.DataFrame({"replicate": ["rep01"] * 6, "true_model": [a8.P] * 3 + [a8.C] * 3,
                      "state_X": ["DECIDED", "DECIDED", "ONE_FAILED", "DECIDED", "NUMERIC_UNDECIDED", "MISSING"],
                      "sel_X": [a8.P, a8.C, "", a8.C, "", ""]})
    k = an.counts(s, "X")
    r = an.rates(k)
    assert r["acc_poisson"] == 1 / 3 and r["acc_cmp2"] == 1 / 3          # failures/undecided/missing never correct
    assert r["mis_P_to_C"] == 1 / 3 and r["mis_C_to_P"] == 0
    assert r["acc_poisson_evaluable"] == 0.5 and r["acc_cmp2_evaluable"] == 1.0
    assert k["failed_P"] == 1 and k["undecided_C"] == 1 and k["missing_C"] == 1
    assert abs(r["failed_rate"] + r["undecided_rate"] + r["missing_rate"] + r["decided_rate"] - 1) < 1e-12


def test_11_auc_rank():
    auc, n = an._auc([3, 2, 1, np.nan], [True, True, False, False])
    assert auc == 1.0 and n == 3


def test_12_previous_artifacts_unchanged():
    root = a8.ROOT
    out = subprocess.run(["git", "diff", "--stat", "c1f37a9", "--",
                          "expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115",
                          "expfam/results/auto_family_z_recovery/af8v2_numerical_repair_20261115",
                          "expfam/results/auto_family_z_recovery/af8v3_numerical_stability_20261115",
                          "expfam/src/experimental/run_af8_integrated_loading.py",
                          "expfam/src/experimental/run_af8v2_integrated_loading.py",
                          "expfam/src/experimental/run_af8v3_integrated_loading.py"],
                         cwd=root, capture_output=True, text=True, check=True).stdout
    assert out.strip() == ""


def test_13_stop_rules_not_relaxed():
    import run_af8v3_integrated_loading as v3
    for k, v in v3.STOP_RULES_V3.items():
        assert vf.STOP_RULES[k] == v
    assert vf.STOP_RULES["S4v2_IS_cells_with_any_cold_start_failure_share_max"] == 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
