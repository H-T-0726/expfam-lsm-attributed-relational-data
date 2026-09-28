"""Zero-EM focused tests for the Phase 9O classifier (Issue #102)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import derivative_failure_classification as dfc                   # noqa: E402
import laplace_k_criterion as lk                                  # noqa: E402
import multistep_local_pilot as mp                                # noqa: E402
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401
from test_theta_stationarity_diagnostic import _mixed_state       # noqa: E402


def _result(status, value, **kw):
    base = dict(phi_at_mode=-10.0, logdet_H=3.0, logdet_sign=1.0,
                min_eigenvalue_H=0.5, grad_inf=1e-9, iterations=2, n=8, k=2,
                Z_hat=np.zeros((8, 2)), notes=[])
    base.update(kw)
    return lk.LaplaceResult(status=status, log_observed=value, **base)


def _patched(monkeypatch, outcome):
    def fake(*args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(dfc.lk, "laplace_log_observed", fake)


def test_1_ok_result_keeps_metadata(monkeypatch):
    state = _mixed_state()
    _patched(monkeypatch, _result("OK", -5.0, iterations=3))
    rec = dfc.evaluate_structured(state, np.zeros(state.n_coords))
    assert rec["status"] == "OK" and rec["log_observed"] == -5.0
    assert rec["iterations"] == 3 and rec["logdet_sign"] == 1.0
    assert dfc.mechanism(rec) == dfc.MECH_OK


def test_2_not_stationary_is_not_collapsed(monkeypatch):
    state = _mixed_state()
    _patched(monkeypatch, _result("NOT_STATIONARY", float("nan"),
                                  grad_inf=3e-6, iterations=200,
                                  notes=["max|grad Phi|=3e-06 > 1e-08"]))
    rec = dfc.evaluate_structured(state, np.zeros(state.n_coords))
    assert rec["status"] == "NOT_STATIONARY" and rec["grad_inf"] == 3e-6
    assert rec["iterations"] == 200 and "max|grad Phi|" in rec["notes"]
    assert math.isfinite(rec["phi_at_mode"]) and not rec["log_observed_finite"]
    assert dfc.mechanism(rec) == dfc.MECH_NOT_STATIONARY


def test_3_hessian_not_pd_keeps_eigenvalue_and_sign(monkeypatch):
    state = _mixed_state()
    _patched(monkeypatch, _result("HESSIAN_NOT_PD", float("nan"),
                                  min_eigenvalue_H=-0.02, logdet_sign=-1.0))
    rec = dfc.evaluate_structured(state, np.zeros(state.n_coords))
    assert rec["min_eigenvalue_H"] == -0.02 and rec["logdet_sign"] == -1.0
    assert dfc.mechanism(rec) == dfc.MECH_HESSIAN


def test_4_exception_class_and_message(monkeypatch):
    state = _mixed_state()
    _patched(monkeypatch, np.linalg.LinAlgError("Singular matrix"))
    rec = dfc.evaluate_structured(state, np.zeros(state.n_coords))
    assert rec["status"] == "EXCEPTION"
    assert rec["exception_class"] == "LinAlgError"
    assert rec["exception_message"] == "Singular matrix"
    assert dfc.mechanism(rec) == dfc.MECH_EXCEPTION


def test_5_labels_follow_phase9n_chart_order():
    state = _mixed_state()
    labels = dfc.coordinate_labels(state)
    nh, ng = state.n_horizontal, len(state.gauss)
    assert len(labels) == state.n_coords
    g = np.arange(state.n_coords, dtype=float)
    blocks = mp.RecenteredState(state, state.base()).blocks(g)
    assert [b for b, _ in labels[:nh]] == ["F_horizontal"] * nh
    assert [b for b, _ in labels[nh:nh + ng]] == ["log_sigma"] * ng
    assert labels[nh + ng] == ("w0", "w0") and labels[-1] == ("w", "w")
    assert blocks["abs_grad_w0"] == nh + ng and blocks["abs_grad_w"] == nh + ng + 1
    assert [l for _, l in labels[nh:nh + ng]] == [
        f"log_sigma[column={l}]" for l in state.gauss]


def _rows(fails):
    rows = []
    for i in range(4):
        for side in "+-":
            m = dfc.MECH_NOT_STATIONARY if (i, side) in fails else dfc.MECH_OK
            rows.append({"coordinate_index": i, "label": f"c{i}",
                         "block": "F_horizontal", "side": side,
                         "status": "", "mechanism": m, "grad_inf": 0.0,
                         "min_eigenvalue_H": 0.0, "logdet_sign": 1.0,
                         "logdet_H": 0.0, "iterations": 0, "notes": "",
                         "exception_class": "", "exception_message": ""})
    return rows


def test_6_one_sided_failure_counts():
    s = dfc.summarize_state(_rows({(2, "+")}))
    assert (s["coordinates"], s["failing_coordinates"], s["failing_sides"]) \
        == (4, 1, 1)
    assert s["pattern"] == "one coordinate, one side"
    assert dfc.decide([s]) == "FAILURE_MECHANISM_CLASSIFIED"


def test_7_two_sided_failure_counts():
    s = dfc.summarize_state(_rows({(1, "+"), (1, "-")}))
    assert (s["failing_coordinates"], s["failing_sides"]) == (1, 2)
    assert s["pattern"] == "one coordinate, both sides"
    none = dfc.summarize_state(_rows(set()))
    many = dfc.summarize_state(_rows({(0, "+"), (3, "-")}))
    assert dfc.decide([s, none]) == "FAILURE_REPRODUCTION_MISMATCH"
    assert dfc.decide([s, many]) == "FAILURE_REPRODUCTION_MISMATCH"


def test_8_no_optimization_or_accept_step(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("optimization step called")
    monkeypatch.setattr(mp, "ascend", forbidden)
    monkeypatch.setattr(mp.RecenteredState, "accept", forbidden)
    monkeypatch.setattr(mp.td, "one_step", forbidden)
    state = _mixed_state()
    F0, var0 = state.F0.copy(), state.var0.copy()
    w0, w, Z0 = state.w0_0, state.w_0, state.Z_hat0.copy()
    rows = dfc.classify_state(state)
    assert len(rows) == 2 * state.n_coords
    assert np.array_equal(state.F0, F0) and np.array_equal(state.var0, var0)
    assert (state.w0_0, state.w_0) == (w0, w)
    assert np.array_equal(state.Z_hat0, Z0)
    assert np.array_equal(state.model.params["F"], F0)
