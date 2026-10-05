"""Contract tests for the multi-support selector, generator and calibration."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import family_selection_multisupport as fsm                               # noqa: E402
from data_generator_multisupport import generate_multisupport_data      # noqa: E402
from family_selection_multisupport import (                              # noqa: E402
    ROUTE_PHASE9C, ROUTE_TRUST_EXACT, gate_columns, score_candidate,
    select_column, selector_only, support_group_of,
)
from family_specs import (                                               # noqa: E402
    GROUP_BOUNDED, GROUP_COUNT, GROUP_POSITIVE, GROUP_REAL, PRIMARY_PRIORITY,
    BernoulliSpec, COMPoissonSpec, PoissonSpec, build_primary_registry,
)

REG = build_primary_registry()


def _Zs(n, k=3, L=3, seed=0):
    return np.random.default_rng(seed).standard_normal((n, k, L))


@pytest.mark.parametrize("spec", [BernoulliSpec(), PoissonSpec()],
                         ids=lambda s: s.candidate_model)
def test_trust_exact_route_matches_phase9c_route(spec):
    Zs = _Zs(80)
    x = spec.sample(Zs[:, :, 0] @ np.array([0.5, -0.3, 0.2]), np.random.default_rng(1))
    a = score_candidate(x, Zs, spec, np.zeros(3), 0, route=ROUTE_PHASE9C)
    b = score_candidate(x, Zs, spec, np.zeros(3), 0, route=ROUTE_TRUST_EXACT)
    assert a.status == b.status == "OK"
    assert abs(a.score - b.score) < 1e-8 * abs(a.score)
    np.testing.assert_allclose(a.loading, b.loading, atol=1e-6)


def test_cmp_nu_one_scores_like_poisson_through_the_generic_route():
    Zs = _Zs(60)
    x = PoissonSpec().sample(Zs[:, :, 0] @ np.array([0.4, 0.4, -0.2]),
                             np.random.default_rng(2))
    a = score_candidate(x, Zs, PoissonSpec(), np.zeros(3), 0, route=ROUTE_TRUST_EXACT)
    b = score_candidate(x, Zs, COMPoissonSpec(1.0), np.zeros(3), 0, route=ROUTE_TRUST_EXACT)
    assert abs(a.score - b.score) < 1e-8 * abs(a.score)


def test_level0_support_groups():
    rng = np.random.default_rng(3)
    eta = rng.normal(0, 0.7, 200)
    for cm, spec in REG.items():
        x = spec.sample(eta, rng)
        assert support_group_of(x) == spec.support_group, cm


def test_admissibility_by_support():
    X = np.array([[0, 1, 3], [1, 2, 1], [2, 1, 2]], dtype=float)
    gates = gate_columns(X, REG, PRIMARY_PRIORITY)
    assert gates[0].admissible == ("poisson", "binomial_n2", "cmp_nu2p0", "cmp_nu0p5")
    assert gates[1].admissible == ("poisson", "binomial_n2", "cmp_nu2p0", "cmp_nu0p5", "ztp")
    assert gates[2].admissible == ("poisson", "cmp_nu2p0", "cmp_nu0p5", "ztp")
    assert all(g.decided_by == "score" for g in gates)


def test_failed_candidate_never_hands_the_win_to_another(monkeypatch):
    Zs = _Zs(50)
    x = PoissonSpec().sample(Zs[:, :, 0] @ np.array([0.3, 0.1, 0.0]), np.random.default_rng(4))
    good = score_candidate(x, Zs, PoissonSpec(), np.zeros(3), 0, route=ROUTE_TRUST_EXACT)
    bad = fsm.CandidateRecordMS(column=0, family_name="com_poisson",
                                candidate_model="cmp_nu0p5", status="FAILED",
                                failure_reason="FloatingPointError: injected")
    sel = select_column([good, bad], PRIMARY_PRIORITY[GROUP_COUNT])
    assert sel.status == "FAILURE" and sel.selected == ""


def test_exact_tie_uses_frozen_priority():
    a = fsm.CandidateRecordMS(0, "com_poisson", "cmp_nu0p5", "OK", score=-10.0)
    b = fsm.CandidateRecordMS(0, "poisson", "poisson", "OK", score=-10.0)
    sel = select_column([a, b], PRIMARY_PRIORITY[GROUP_COUNT])
    assert sel.selected == "poisson" and sel.margin_neg2 == 0.0


def test_generator_reproducible_and_information_matched():
    specs = [REG[c] for c in ("poisson", "cb", "lognormal_s1", "gaussian")]
    a = generate_multisupport_data(n=60, k=3, seed=77, column_specs=specs, tau=1.0)
    b = generate_multisupport_data(n=60, k=3, seed=77, column_specs=specs, tau=1.0)
    for key in ("X", "Y", "Z", "F"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    info = np.array(a.metadata["info_at_zero"])
    # average of ||f_l||^2 A''(0) over the columns equals tau by construction
    assert abs(np.mean(np.sum(a.F ** 2, axis=1) * info) - 1.0) < 1e-12


@pytest.mark.parametrize("group,tau", [(GROUP_COUNT, 2.0), (GROUP_POSITIVE, 2.0),
                                       (GROUP_BOUNDED, 0.25)])
def test_selector_is_consistent_at_large_n(group, tau):
    """With the true Z and a large n every true candidate must be recovered.

    The smallest calibrated per-observation divergence in these settings is
    ~0.03, so n = 3000 gives an expected log-likelihood ratio of ~90 nats.
    This checks that the scores are correct, not how the selector does at the
    experiment's n.
    """
    cms = list(PRIMARY_PRIORITY[group]) * 2          # d >= K for every group
    specs = [REG[c] for c in cms]
    ds = generate_multisupport_data(n=3000, k=3, seed=5150, column_specs=specs, tau=tau)
    gates, sels = selector_only(ds.X, ds.Z, REG, PRIMARY_PRIORITY, route=ROUTE_TRUST_EXACT)
    chosen = [s.selected for s in sels]
    assert chosen == cms, (chosen, [s.margin_neg2 for s in sels])


def test_estep_guard_is_inert_when_never_triggered_and_counts_when_it_is():
    from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent
    from model_dual_expfam_generic import DualExpFamLSMGeneric
    rng = np.random.default_rng(9)
    n, k = 25, 3
    specs = [REG["cmp_nu0p5"], REG["poisson"], REG["cb"], REG["lognormal_s1"]]
    Zt = rng.standard_normal((n, k))
    F = rng.normal(0, 0.3, (4, k))
    X = np.column_stack([s.sample(Zt @ F[j], rng) for j, s in enumerate(specs)])
    Y = (rng.random((n, n)) < 0.3).astype(float)
    Y = np.triu(Y, 1); Y = Y + Y.T
    m = DualExpFamLSMGeneric(n=n, d=4, k=k, L=1, column_specs=specs)
    m.initialize_params(seed=3)
    m.params["F"] = F.copy()
    m.params["w0"], m.params["w"] = -1.0, 0.5
    a = m.calc_eta_newton(X, Y, rng=np.random.default_rng(1), alpha=0.5)
    b = DualExpFamLSMPerColumnConsistent.calc_eta_newton(
        m, X, Y, rng=np.random.default_rng(1), alpha=0.5)
    np.testing.assert_array_equal(a, b)
    assert m.estep_backtracks == 0
    # a huge count under COM-Poisson nu=0.5 starting at eta ~ 0: the plain
    # Newton step overshoots out of the certifiable range; the guard halves it
    specs2 = [REG["cmp_nu0p5"], REG["cb"]]
    X2 = np.column_stack([np.full(n, 2.0), np.full(n, 0.5)])
    X2[0, 0] = 2000.0
    m2 = DualExpFamLSMGeneric(n=n, d=2, k=k, L=1, column_specs=specs2)
    m2.initialize_params(seed=3)
    m2.params["Z"] = np.zeros((n, k))
    m2.params["F"] = np.array([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    m2.params["w0"], m2.params["w"] = -1.0, 0.1
    m2.calc_eta_newton(X2, Y, rng=np.random.default_rng(1), alpha=0.5)
    assert m2.estep_backtracks > 0
