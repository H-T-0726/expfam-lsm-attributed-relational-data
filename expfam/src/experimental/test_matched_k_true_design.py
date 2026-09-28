"""Zero-EM focused tests for the Phase 9W design (Issue #120)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                  # noqa: E402
import matched_k_true_design as mk                                # noqa: E402
import run_family_selection_pilot as pilot                        # noqa: E402
import run_lap_vs_cq_20 as pc                                     # noqa: E402
from model_dual_expfam_consistent import (                        # noqa: E402
    DualExpFamLSMPerColumnConsistent)
from test_joint_family_k_selection import _forbid_em              # noqa: E402,F401

COARSE = mk.PRIMARY_SETTINGS[0]


def test_1_x_energy():
    for k in mk.K_TRUES:
        assert abs(mk.f_scale(k) ** 2 * k / mk.D - 0.5) <= 1e-15
        assert math.isclose(mk.f_scale(k), math.sqrt(6 / k), rel_tol=1e-15)


def test_2_y_variance():
    for k in mk.K_TRUES:
        assert abs(mk.w_of(k) ** 2 * k - 3.0) <= 1e-15
        assert math.isclose(mk.w_of(k), math.sqrt(3 / k), rel_tol=1e-15)


def test_3_k3_is_phase9k():
    assert mk.f_scale(3) == pc.PROTOCOL.f_scale == math.sqrt(2.0)
    assert mk.w_of(3) == pc.PROTOCOL.w == 1.0
    assert mk.W0_REF == pc.PROTOCOL.w0 == -1.0


def test_4_density_normalization_and_second_moment():
    c = mk.density_checks()
    assert c["pass"]
    for k in mk.K_TRUES:
        assert c[f"K{k}"]["second_moment_error"] <= mk.DENSITY_TOL


def test_5_calibration_deterministic_and_k3_recovery():
    a = mk.calibrate(mk.Adaptive(COARSE[1], COARSE[2]))
    assert a == mk.calibrate(mk.Adaptive(COARSE[1], COARSE[2]))
    assert abs(a["K3_recovery_error"]) <= 1e-12


def test_6_crosscheck_deterministic():
    h = mk.H_STEPS[0]
    assert mk.calibrate(mk.ExpTrapezoid(h)) == mk.calibrate(mk.ExpTrapezoid(h))


def test_7_k1_boundary_static():
    audit = mk.k1_boundary_audit()
    assert audit["loading_parameter_count"][1] == 12
    assert not audit["blocker"]
    proto = mk.protocol_for(1, -1.0)
    ds = pilot.build_dataset(proto, proto.replicates[0])
    assert ds.F.shape == (12, 1) and np.linalg.matrix_rank(ds.F) == 1
    model = DualExpFamLSMPerColumnConsistent(
        n=ds.X.shape[0], d=12, k=1, L=1,
        family_x_list=list(proto.family_x_list), family_y="bernoulli")
    model.initialize_params(true_params=None, seed=0)
    res = lk.laplace_log_observed(model, ds.X, ds.Y,
                                  np.zeros((ds.X.shape[0], 1)))
    assert res.status in (lk.STATUS_OK, lk.STATUS_NOT_STATIONARY,
                          lk.STATUS_HESSIAN_NOT_PD)
    assert res.Z_hat.shape == (ds.X.shape[0], 1)


def test_8_k3_anchor_compatibility_deterministic():
    a = mk.k3_anchor_compatibility(-1.0)
    assert a == mk.k3_anchor_compatibility(-1.0)
    assert a["protocol_fields_differing_from_phase9k"] == ["stage"]


def test_9_10_11_protocol_scope():
    assert mk.K_TRUES == (1, 2, 3, 4)
    assert mk.CANDIDATE_K == (1, 2, 3, 4, 5)
    for k in mk.K_TRUES:
        p = mk.protocol_for(k, -1.0)
        assert p.k_candidates == mk.CANDIDATE_K
        assert [r.label for r in p.replicates] == [f"rep{r:02d}"
                                                   for r in range(1, 11)]
        assert p.starts == (("start_B", "bernoulli"),)


def test_12_em_cap_follows_anchor_decision():
    for reusable, cap in ((True, 300), (False, 400)):
        new_k = [k for k in mk.K_TRUES if not (k == 3 and reusable)]
        assert len(new_k) * mk.FUTURE_REPLICATES * 5 * 2 == cap


ARTIFACT = (Path(__file__).resolve().parents[2]
            / "results/matched_k_true_design/phase9w_20260928")


def test_13_sanitize_future_spec_fail_closed():
    import pytest
    spec = mk.sanitize_future_spec(mk.protocol_for(1, -0.9).as_json())
    auth = spec["execution_authorization"]
    assert auth["authorized"] is False and auth["status"] == "NOT_AUTHORIZED_YET"
    assert "75" not in auth["gate"]
    assert spec["starts"] == mk.START_B_ONLY
    assert spec["start_policy"].startswith("start_B only")
    bad = dict(mk.protocol_for(1, -0.9).as_json(),
               starts=mk.START_B_ONLY + [{"label": "start_A",
                                          "ambiguous_start": "poisson"}])
    with pytest.raises(SystemExit):
        mk.sanitize_future_spec(bad)


def test_14_future_protocol_artifact():
    import json
    fp = json.loads((ARTIFACT / "future_protocol.json").read_text("utf-8"))
    design = json.loads((ARTIFACT / "design.json").read_text("utf-8"))
    assert fp["status"] == "FROZEN_NOT_EXECUTED"
    assert fp["execution_authorized"] is False
    assert fp["future_new_em_cap"] == 300 and fp["K3_anchor_reusable"] is True
    for name, c in fp["conditions"].items():
        auth = c["execution_authorization"]
        assert auth["authorized"] is False
        assert auth["status"] == "NOT_AUTHORIZED_YET"
        assert "Issue #75" not in str(auth.get("authorized_in", ""))
        assert "75-B" not in str(auth.get("gate", ""))
        assert c["starts"] == mk.START_B_ONLY
        assert c["start_policy"].startswith("start_B only")
        k = int(name[1:])
        assert c["w0"] == design["w0"][str(k)]
        assert c["f_scale"] == mk.f_scale(k) and c["w"] == mk.w_of(k)
    assert fp["p_target"] == design["p_target"]
