"""Zero-EM checks for Issue #128 (k_criterion_panel + experiment A1 runner).

No fitting, no sampling: EM entry points fail the test if reached.  The
Phase 9H tests only exercised the per-column model class; the scalar-family
``DualExpFamLSMConsistent`` path used by experiments A1 and B is checked here
(gradient / Hessian against central differences, equality with the
per-column class at a uniform family, decomposition closure).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_EXP = ROOT / "expfam" / "src" / "experimental"
for _p in (str(_EXP), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import k_criterion_panel as panel                                # noqa: E402
import laplace_k_criterion as lk                                 # noqa: E402
import run_clap_clean_nsweep as a1                               # noqa: E402
import run_clean_true_k_sweep as hist                            # noqa: E402
from model_dual_expfam_consistent import (                       # noqa: E402
    DualExpFamLSMConsistent,
    DualExpFamLSMPerColumnConsistent,
)


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    import em_runner

    def _no_em(*_a, **_k):
        raise AssertionError("EM was reached in a zero-EM test")

    monkeypatch.setattr(em_runner, "run_em_experimental", _no_em)
    monkeypatch.setattr(DualExpFamLSMConsistent, "calc_eta_newton", _no_em)
    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton",
                        _no_em)


def _case(family_x, n=7, d=4, k=2, seed=3, masked=True):
    rng = np.random.default_rng(seed)
    F = rng.normal(scale=0.5, size=(d, k))
    Z = rng.normal(size=(n, k))
    if family_x == "poisson":
        X = rng.poisson(1.5, size=(n, d)).astype(float)
    else:
        X = rng.normal(size=(n, d))
    up = np.triu((rng.random((n, n)) < 0.4).astype(float), 1)
    Y = up + up.T
    mask = None
    if masked:
        m = np.triu(rng.random((n, n)) < 0.8, 1)
        mask = m | m.T
    sig = np.diag(rng.uniform(0.5, 1.5, size=d))
    theta = {"F": F, "sigma": sig, "w0": -0.4, "w": 0.8, "var_z": 1.0}
    scalar = DualExpFamLSMConsistent(n=n, d=d, k=k, L=1, family_x=family_x,
                                     family_y="bernoulli", train_mask=mask)
    percol = DualExpFamLSMPerColumnConsistent(
        n=n, d=d, k=k, L=1, family_x_list=[family_x] * d,
        family_y="bernoulli", train_mask=mask)
    for m_ in (scalar, percol):
        m_.initialize_params(true_params=None, seed=0)
        m_.params.update({key: (np.array(v) if isinstance(v, np.ndarray)
                                else v) for key, v in theta.items()})
    return scalar, percol, X, Y, Z


def _fd_grad(model, X, Y, Z, h=1e-5):
    g = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for a in range(Z.shape[1]):
            p, m = Z.copy(), Z.copy()
            p[i, a] += h
            m[i, a] -= h
            g[i, a] = (lk.complete_log_density(model, X, Y, p)
                       - lk.complete_log_density(model, X, Y, m)) / (2 * h)
    return g


def _fd_hess(model, X, Y, Z, h=1e-6):
    n, k = Z.shape
    H = np.zeros((n * k, n * k))
    for c in range(n * k):
        p, m = Z.ravel().copy(), Z.ravel().copy()
        p[c] += h
        m[c] -= h
        H[:, c] = -(lk.gradient(model, X, Y, p.reshape(n, k)).ravel()
                    - lk.gradient(model, X, Y, m.reshape(n, k)).ravel()) / (2 * h)
    return H


@pytest.mark.parametrize("family_x", ["poisson", "gaussian"])
def test_scalar_family_gradient_and_hessian_match_differences(family_x):
    model, _, X, Y, Z = _case(family_x)
    ga = lk.gradient(model, X, Y, Z)
    gn = _fd_grad(model, X, Y, Z)
    assert np.max(np.abs(ga - gn)) < 1e-6
    Ha = lk.negative_hessian(model, X, Y, Z)
    Hn = _fd_hess(model, X, Y, Z)
    assert np.max(np.abs(Ha - Hn)) < 1e-6
    assert np.allclose(Ha, Ha.T, atol=1e-12)


@pytest.mark.parametrize("family_x", ["poisson", "gaussian"])
def test_scalar_family_equals_uniform_per_column(family_x):
    s, p, X, Y, Z = _case(family_x)
    assert math.isclose(lk.complete_log_density(s, X, Y, Z),
                        lk.complete_log_density(p, X, Y, Z),
                        rel_tol=1e-12, abs_tol=1e-9)
    assert np.allclose(lk.gradient(s, X, Y, Z), lk.gradient(p, X, Y, Z),
                       atol=1e-10)
    assert np.allclose(lk.negative_hessian(s, X, Y, Z),
                       lk.negative_hessian(p, X, Y, Z), atol=1e-10)
    rs = lk.laplace_log_observed(s, X, Y, Z)
    rp = lk.laplace_log_observed(p, X, Y, Z)
    assert rs.status == rp.status == "OK"
    assert math.isclose(rs.log_observed, rp.log_observed, rel_tol=1e-10)


@pytest.mark.parametrize("family_x", ["poisson", "gaussian"])
def test_laplace_terms_close_and_are_deterministic(family_x):
    model, _, X, Y, Z = _case(family_x)
    a = panel.laplace_terms(model, X, Y, Z, num_params=7, N=7)
    b = panel.laplace_terms(model, X, Y, Z, num_params=7, N=7)
    assert a["lap_status"] == "OK"
    assert abs(a["C_Lap_recomposed"] - a["C_Lap"]) < 1e-8
    assert a["C_Lap"] == b["C_Lap"] and np.array_equal(a["Z_hat"], b["Z_hat"])
    # C_Lap = -2 l_Lap + d ln N with l_Lap = Phi + nK/2 ln2pi - 1/2 logdet
    n, k = Z.shape
    lap = (a["lap_phi_at_mode"] + 0.5 * n * k * math.log(2 * math.pi)
           - 0.5 * a["lap_logdet_H"])
    assert math.isclose(a["C_Lap"], -2 * lap + 7 * math.log(7), rel_tol=1e-12)


@pytest.mark.parametrize("family_x", ["poisson", "gaussian"])
def test_cq_and_cond_closure(family_x):
    model, _, X, Y, Z = _case(family_x)
    rng = np.random.default_rng(9)
    Zs = np.stack([Z + 0.1 * rng.normal(size=Z.shape) for _ in range(3)], 2)
    cq = panel.cq_terms(model, X, Y, Zs, num_params=7, n=7)
    assert abs(cq["C_Q_recomposed"] - cq["C_Q"]) < 1e-8
    lap = panel.laplace_terms(model, X, Y, Zs[:, :, -1], num_params=7, N=7)
    cond = panel.cond_terms(model, X, Y, Z_samples=Zs, Z_last=Zs[:, :, -1],
                            Z_mode=lap["Z_hat"], mode_ok=True,
                            num_params=7, n=7)
    assert math.isclose(cond["C_cond_MC"], cq["D_K"] + cq["P_theta"],
                        rel_tol=1e-12)
    assert math.isclose(cond["C_cond_mode"],
                        lap["lap_fit_mode"] + lap["lap_penalty"],
                        rel_tol=1e-12)
    # conditional at the last sample is what S3 evaluates up to constants
    assert math.isfinite(cond["C_cond_last"])


def test_cond_mode_is_nan_when_mode_not_ok():
    model, _, X, Y, Z = _case("poisson")
    c = panel.cond_terms(model, X, Y, Z_samples=Z[:, :, None], Z_last=Z,
                         Z_mode=Z, mode_ok=False, num_params=3, n=7)
    assert math.isnan(c["C_cond_mode"])


def test_non_bernoulli_y_is_rejected():
    model, _, X, Y, Z = _case("poisson")
    model.family = "poisson"
    with pytest.raises(ValueError):
        panel.cq_terms(model, X, Y, Z[:, :, None], num_params=3, n=7)


def test_selector_tie_rule_margin_and_direction():
    k, ties, margin, best = a1._select({1: 5.0, 2: 3.0, 3: 3.0, 4: 9.0}, True)
    assert (k, ties, margin, best) == (2, [2, 3], 0.0, 3.0)
    k, _, margin, _ = a1._select({1: -0.6, 2: -0.5, 3: -0.7}, False)
    assert k == 2 and math.isclose(margin, 0.1)
    assert a1.label(3, 3) == "exact" and a1.label(2, 3) == "under" \
        and a1.label(4, 3) == "over"


def _rows(lap_bad=None):
    rows = []
    for k in hist.CANDIDATE_K:
        for s in hist.STARTS:
            v = (k - 3) ** 2
            rows.append({"k_true": 3, "n": 50, "replicate": 1, "k_est": k,
                         "start": s, "C_Q": v + 1.0, "C_Lap": v + 2.0,
                         "S1": -v - 0.5, "S3": -k * 1.0,
                         "lap_status": ("NOT_STATIONARY"
                                        if (k, s) == lap_bad else "OK"),
                         "D_K": -k * 10.0, "P_Z": 5.0 * k, "P_theta": 2.0 * k,
                         "lap_fit_mode": -k * 9.0, "lap_volume": 4.0 * k,
                         "lap_penalty": 2.0 * k})
    return rows


def test_analyse_selects_and_reports_incomplete_without_imputation():
    out = a1.analyse(_rows())
    sel = {r["criterion"]: r for r in out["selection"]}
    assert sel["C_Q"]["selected_k"] == 3 and sel["C_Lap"]["selected_k"] == 3
    assert sel["S1"]["selected_k"] == 3 and sel["S3"]["selected_k"] == 7
    assert len(out["curves"]) == 4 * 7
    assert len(out["decomposition"]) == 6
    bad = a1.analyse(_rows(lap_bad=(5, 2)))
    sel = {r["criterion"]: r for r in bad["selection"]}
    assert sel["C_Lap"]["status"] == "INCOMPLETE"
    assert sel["C_Lap"]["selected_k"] == ""
    assert sel["C_Q"]["status"] == "COMPLETE"
    dec = {(r["k_from"]): r for r in bad["decomposition"]}
    assert dec[4]["Lap_complete"] is False and math.isnan(dec[4]["Lap_delta"])
    assert dec[2]["Lap_complete"] is True


def test_analyse_rejects_missing_candidate_k():
    rows = [r for r in _rows() if r["k_est"] != 7]
    with pytest.raises(a1.A1Stop):
        a1.analyse(rows)


def test_seed_rules_are_the_historical_functions():
    assert hist.model_seed(1, 50, 1, 1, 1) == 980111
    assert hist.data_seed(1, 50, 1) == 820501
    assert hist.split_seed(1, 50, 1) == 830501
    assert len(a1.all_cells()) == 64


def test_protocol_hash_is_deterministic_and_records_adoption_status():
    assert a1.protocol_hash() == a1.protocol_hash()
    p = a1.frozen_protocol()
    assert p["manuscript_adoption_status"] == "NOT_DECIDED_BY_HUMAN"
    assert p["historical_protocol_hash"] == a1.HIST_PROTOCOL_HASH
    assert json.dumps(p, sort_keys=True)


def test_historical_artifacts_present_and_protocol_hash_matches():
    for name in a1.HIST_FILES:
        assert (a1.HIST_DIR / name).is_file()
    body = json.loads((a1.HIST_DIR / "protocol.json").read_text("utf-8"))
    assert body["protocol_hash"] == a1.HIST_PROTOCOL_HASH


def test_output_locations_are_new():
    assert "clap_clean_nsweep_20261006" in str(a1.OUT_ROOT)
    assert a1.OUT_ROOT != a1.HIST_DIR
    assert "kennkyu_artifacts" in str(a1.STATE_ROOT)
