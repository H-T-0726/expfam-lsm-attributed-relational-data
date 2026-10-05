"""Deterministic tests for the generic family specifications (Gate C).

Every quantity is checked against something computed independently of the
formula under test: normalisation and moments by numerical summation /
quadrature, derivatives by central finite differences, and limits against the
family they must approach.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).parent))

from family_specs import (  # noqa: E402
    BernoulliSpec, BinomialSpec, COMPoissonSpec, ContinuousBernoulliSpec,
    GaussianSpec, LogitNormalSpec, LogNormalSpec, PoissonSpec,
    PositiveTruncatedNormalSpec, SupportViolation, TruncatedNormal01Spec,
    ZeroTruncatedPoissonSpec, build_primary_registry, compatibility_rows,
    log_ndtr_diff, PRIMARY_PRIORITY,
)
from objective_consistent_numerics import (  # noqa: E402
    bernoulli_log_likelihood, poisson_log_likelihood,
)

DISCRETE = [PoissonSpec(), BinomialSpec(2), COMPoissonSpec(0.5),
            COMPoissonSpec(2.0), ZeroTruncatedPoissonSpec(), BernoulliSpec()]
CONTINUOUS = [LogNormalSpec(1.0), PositiveTruncatedNormalSpec(1.0),
              ContinuousBernoulliSpec(), LogitNormalSpec(1.0),
              TruncatedNormal01Spec(0.25, 0.5)]
ALL_CANONICAL = DISCRETE + CONTINUOUS
ETA_GRID = np.array([-6.0, -2.5, -1.0, -0.3, -0.004, 0.0, 0.004, 0.3, 1.0, 2.5, 4.0])


def _eta_grid(spec):
    # CMP nu=0.5 has mean ~exp(2 eta); keep the grid where the series is
    # cheap but still covers the generator's range with a margin.
    if spec.candidate_model == "cmp_nu0p5":
        return ETA_GRID[ETA_GRID <= 2.5]
    if spec.candidate_model == "tn01_s0p25_c0p5":
        return np.concatenate([ETA_GRID, [-25.0, 25.0]])
    return ETA_GRID


def _discrete_support(spec, eta):
    hi = 400 if spec.candidate_model != "cmp_nu0p5" else 3000
    x = np.arange(0, hi + 1, dtype=float)
    return x[spec.support_mask(x)]


def _continuous_integral(spec, eta, g):
    lo, hi = (0.0, 1.0) if spec.support_group.endswith("BOUNDED_CONTINUOUS") else (0.0, np.inf)

    def integrand(x):
        xa = np.array([x])
        return float(np.exp(spec.log_prob(xa, np.array([eta]))[0]) * g(xa)[0])

    if hi == np.inf:
        val = integrate.quad(integrand, lo, 1.0, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
        val += integrate.quad(integrand, 1.0, np.inf, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
        return val
    # split to help the endpoints of logit-normal / truncated normal
    pts = [1e-12, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-12]
    return sum(integrate.quad(integrand, a, b, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
               for a, b in zip(pts[:-1], pts[1:]))


@pytest.mark.parametrize("spec", DISCRETE, ids=lambda s: s.candidate_model)
def test_discrete_normalisation_and_moments(spec):
    for eta in _eta_grid(spec):
        x = _discrete_support(spec, eta)
        lp = spec.log_prob(x, np.full_like(x, eta))
        p = np.exp(lp)
        T = spec.sufficient_statistic(x)
        assert abs(p.sum() - 1.0) < 1e-10, (spec.candidate_model, eta, p.sum())
        mean = float(np.sum(p * T))
        var = float(np.sum(p * (T - mean) ** 2))
        assert abs(mean - spec.mean_statistic(np.array([eta]))[0]) < 1e-9 * max(1, mean)
        assert abs(var - spec.variance_statistic(np.array([eta]))[0]) < 1e-8 * max(1, var)


@pytest.mark.parametrize("spec", CONTINUOUS, ids=lambda s: s.candidate_model)
def test_continuous_normalisation_and_moments(spec):
    for eta in _eta_grid(spec):
        one = _continuous_integral(spec, eta, lambda x: np.ones_like(x))
        assert abs(one - 1.0) < 1e-7, (spec.candidate_model, eta, one)
        mean = _continuous_integral(spec, eta, spec.sufficient_statistic)
        A1 = spec.mean_statistic(np.array([eta]))[0]
        assert abs(mean - A1) < 1e-6 * max(1.0, abs(A1)), (spec.candidate_model, eta)
        var = _continuous_integral(spec, eta,
                                   lambda x: (spec.sufficient_statistic(x) - A1) ** 2)
        A2 = spec.variance_statistic(np.array([eta]))[0]
        assert abs(var - A2) < 1e-6 * max(1.0, A2), (spec.candidate_model, eta, var, A2)


@pytest.mark.parametrize("spec", ALL_CANONICAL, ids=lambda s: s.candidate_model)
def test_derivatives_match_central_differences(spec):
    h = 1e-5
    for eta in _eta_grid(spec):
        e = np.array([eta])
        A_p, A_m = spec.log_partition(e + h)[0], spec.log_partition(e - h)[0]
        fd1 = (A_p - A_m) / (2 * h)
        assert abs(fd1 - spec.mean_statistic(e)[0]) < 1e-6 * max(1.0, abs(fd1)), \
            (spec.candidate_model, eta)
        M_p, M_m = spec.mean_statistic(e + h)[0], spec.mean_statistic(e - h)[0]
        fd2 = (M_p - M_m) / (2 * h)
        assert abs(fd2 - spec.variance_statistic(e)[0]) < 1e-5 * max(1.0, abs(fd2)), \
            (spec.candidate_model, eta, fd2, spec.variance_statistic(e)[0])
        assert spec.variance_statistic(e)[0] > 0.0


@pytest.mark.parametrize("spec", ALL_CANONICAL, ids=lambda s: s.candidate_model)
def test_column_gradient_and_curvature_match_finite_differences(spec):
    """d/df sum_i log p(x_i | f^T z_i) and its Hessian, against differences."""
    rng = np.random.default_rng(11)
    n, k = 30, 3
    Z = rng.standard_normal((n, k))
    f_true = np.array([0.3, -0.2, 0.25])
    x = spec.sample(Z @ f_true, np.random.default_rng(5))
    f = np.array([0.1, 0.05, -0.1])

    def ll(fv):
        return float(np.sum(spec.log_prob(x, Z @ fv)))

    grad = Z.T @ (spec.sufficient_statistic(x) - spec.mean_statistic(Z @ f))
    hess = -(Z.T * spec.variance_statistic(Z @ f)) @ Z
    h = 1e-5
    for j in range(k):
        ej = np.zeros(k); ej[j] = h
        fd = (ll(f + ej) - ll(f - ej)) / (2 * h)
        assert abs(fd - grad[j]) < 1e-5 * max(1.0, abs(fd))
        g_p = Z.T @ (spec.sufficient_statistic(x) - spec.mean_statistic(Z @ (f + ej)))
        g_m = Z.T @ (spec.sufficient_statistic(x) - spec.mean_statistic(Z @ (f - ej)))
        fd_col = (g_p - g_m) / (2 * h)
        assert np.max(np.abs(fd_col - hess[:, j])) < 1e-5 * max(1.0, np.max(np.abs(fd_col)))


def test_cmp_nu_one_equals_poisson():
    cmp1, pois = COMPoissonSpec(1.0), PoissonSpec()
    eta = np.linspace(-5, 4, 31)
    np.testing.assert_allclose(cmp1.log_partition(eta), pois.log_partition(eta), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cmp1.mean_statistic(eta), pois.mean_statistic(eta), rtol=1e-11)
    np.testing.assert_allclose(cmp1.variance_statistic(eta), pois.variance_statistic(eta), rtol=1e-10)
    x = np.arange(0, 30, dtype=float)
    for e in (-1.0, 0.0, 2.0):
        np.testing.assert_allclose(cmp1.log_prob(x, np.full_like(x, e)),
                                   pois.log_prob(x, np.full_like(x, e)), atol=1e-11)


def test_cmp_series_tail_bound_recorded_and_cap_raises():
    spec = COMPoissonSpec(0.5)
    diag = spec.series_diagnostics(np.array([-1.0, 0.0, 2.0]))
    assert diag["max_log_tail"] < -40.0 and diag["M"] > 0
    with pytest.raises(FloatingPointError):
        spec.log_partition(np.array([40.0]))      # needs > 2e6 terms


def test_cb_eta_zero_is_uniform_and_continuous_across_series_threshold():
    cb = ContinuousBernoulliSpec()
    x = np.linspace(0.01, 0.99, 7)
    np.testing.assert_allclose(cb.log_prob(x, np.zeros_like(x)), 0.0, atol=1e-15)
    assert cb.log_partition(np.array([0.0]))[0] == 0.0
    assert cb.mean_statistic(np.array([0.0]))[0] == 0.5
    assert abs(cb.variance_statistic(np.array([0.0]))[0] - 1.0 / 12.0) < 1e-15
    for thr in (1e-2, 0.1):
        lo, hi = np.array([thr * (1 - 1e-9)]), np.array([thr * (1 + 1e-9)])
        pairs = ((cb.log_partition, cb.mean_statistic),
                 (cb.mean_statistic, cb.variance_statistic),
                 (cb.variance_statistic, None))
        for fn, deriv in pairs:
            # the two sides differ by 2e-9 * thr in eta; remove the genuine
            # change (first order) and require the branch jump to be tiny
            slope = deriv(np.array([thr]))[0] if deriv else -thr / 120.0
            jump = fn(hi)[0] - fn(lo)[0] - slope * (hi[0] - lo[0])
            assert abs(jump) < 1e-13


def test_tn01_wide_sigma_approaches_continuous_bernoulli():
    cb = ContinuousBernoulliSpec()
    tn = TruncatedNormal01Spec(1e4, 0.5)
    x = np.linspace(0.02, 0.98, 9)
    for e in (-3.0, 0.0, 2.0):
        eta = np.full_like(x, e)
        np.testing.assert_allclose(tn.log_prob(x, eta), cb.log_prob(x, eta), atol=1e-6)


def test_ptn_far_from_truncation_approaches_normal():
    ptn = PositiveTruncatedNormalSpec(1.0)
    eta = 12.0                     # mu = 12 sigma: truncation mass ~ 1e-33
    x = np.linspace(9, 15, 7)
    normal = -0.5 * (x - eta) ** 2 - 0.5 * math.log(2 * math.pi)
    np.testing.assert_allclose(ptn.log_prob(x, np.full_like(x, eta)), normal, atol=1e-12)


def test_existing_families_match_existing_numerics():
    rng = np.random.default_rng(3)
    eta = rng.normal(0, 1.5, 200)
    xb = rng.integers(0, 2, 200).astype(float)
    xp = rng.poisson(2.0, 200).astype(float)
    np.testing.assert_array_equal(BernoulliSpec().log_prob(xb, eta),
                                  bernoulli_log_likelihood(xb, eta))
    np.testing.assert_allclose(PoissonSpec().log_prob(xp, eta),
                               poisson_log_likelihood(xp, eta) - gammaln(xp + 1),
                               rtol=0, atol=1e-12)
    g = GaussianSpec()
    np.testing.assert_allclose(g.log_prob(xp, eta, sigma_sq=2.0),
                               -0.5 * (xp - eta) ** 2 / 2.0 - 0.5 * math.log(2.0)
                               - 0.5 * math.log(2 * math.pi), atol=1e-13)


@pytest.mark.parametrize("spec", ALL_CANONICAL, ids=lambda s: s.candidate_model)
def test_support_violation_and_nonfinite_fail_fast(spec):
    bad = {"GROUP_COUNT": np.array([0.5]),
           "GROUP_POSITIVE_CONTINUOUS": np.array([-1.0]),
           "GROUP_BOUNDED_CONTINUOUS": np.array([1.0])}[spec.support_group]
    with pytest.raises(SupportViolation):
        spec.log_prob(bad, np.array([0.0]))
    good = spec.sample(np.zeros(3), np.random.default_rng(0))
    with pytest.raises(FloatingPointError):
        spec.log_prob(good, np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(FloatingPointError):
        spec.log_partition(np.array([np.inf]))


def test_extra_support_cases():
    with pytest.raises(SupportViolation):
        ZeroTruncatedPoissonSpec().log_prob(np.array([0.0]), np.array([0.0]))
    with pytest.raises(SupportViolation):
        BinomialSpec(2).log_prob(np.array([3.0]), np.array([0.0]))
    with pytest.raises(SupportViolation):
        LogitNormalSpec(1.0).log_prob(np.array([0.0]), np.array([0.0]))


@pytest.mark.parametrize("spec", ALL_CANONICAL, ids=lambda s: s.candidate_model)
def test_sampling_reproducible_in_support_and_mean_consistent(spec):
    eta = np.full(20000, 0.4)
    a = spec.sample(eta, np.random.default_rng(123))
    b = spec.sample(eta, np.random.default_rng(123))
    np.testing.assert_array_equal(a, b)
    assert spec.in_support(a)
    T = spec.sufficient_statistic(a)
    mean, var = spec.mean_statistic(eta[:1])[0], spec.variance_statistic(eta[:1])[0]
    assert abs(T.mean() - mean) < 5 * math.sqrt(var / eta.size)


def test_ztp_small_lambda_series_is_accurate():
    ztp = ZeroTruncatedPoissonSpec()
    # eta = -40: lam ~ 4e-18. mean -> 1, var -> lam/2, A -> eta.
    e = np.array([-40.0])
    lam = math.exp(-40.0)
    assert abs(ztp.mean_statistic(e)[0] - (1 + lam / 2)) < 1e-15
    assert abs(ztp.variance_statistic(e)[0] / (lam / 2) - 1.0) < 1e-12
    assert abs(ztp.log_partition(e)[0] - (-40.0 + lam / 2)) < 1e-12


def test_log_ndtr_diff_stable_regions():
    from scipy.stats import norm
    for a, b in [(-3.0, 1.0), (2.0, 4.0), (-12.0, -10.0), (10.0, 12.0)]:
        ref = math.log(norm.cdf(b) - norm.cdf(a)) if (b < 8 and a > -8) else None
        val = log_ndtr_diff(np.array([a]), np.array([b]))[0]
        assert np.isfinite(val)
        if ref is not None and a < 0 < b:
            assert abs(val - ref) < 1e-12
    # symmetric tails agree
    assert abs(log_ndtr_diff(np.array([10.0]), np.array([12.0]))[0]
               - log_ndtr_diff(np.array([-12.0]), np.array([-10.0]))[0]) < 1e-12


def test_registry_priority_and_compatibility_rows():
    reg = build_primary_registry()
    for group, order in PRIMARY_PRIORITY.items():
        for cm in order:
            assert reg[cm].support_group == group
    rows = compatibility_rows()
    names = [r["family_name"] for r in rows]
    assert len(names) == len(set(names))
    assert {r["compatibility_status"] for r in rows} == {
        "PASS_STRICT", "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
        "BLOCKED_OUTSIDE_CURRENT_MODEL"}


def _quad_moments(logp, lo, hi, points=None):
    from scipy import integrate as it
    xs = np.linspace(lo, min(hi, lo + 50), 20001)
    m = np.max(logp(xs))
    f = lambda k: it.quad(lambda x: x ** k * np.exp(logp(x) - m), lo, hi, points=points,
                          limit=2000, epsabs=0, epsrel=1e-13)[0]
    m0 = f(0)
    mu = f(1) / m0
    var = it.quad(lambda x: (x - mu) ** 2 * np.exp(logp(x) - m), lo, hi, points=points,
                  limit=2000, epsabs=0, epsrel=1e-13)[0] / m0
    return mu, var


@pytest.mark.parametrize("eta", [-1e4, -1000.0, -300.0, -100.0, 100.0, 300.0, 1000.0, 1e4])
def test_tn01_far_tail_moments(eta):
    tn = TruncatedNormal01Spec(0.25, 0.5)
    sc = 1 / abs(eta)
    pts = [p for p in (sc * 1e-3, sc * 1e-1, sc, 10 * sc, 0.5, 1 - 10 * sc, 1 - sc,
                       1 - sc * 1e-1) if 0 < p < 1]
    mu, var = _quad_moments(lambda x: eta * x - (x - 0.5) ** 2 / 0.125, 0.0, 1.0, pts)
    assert abs(tn.mean_statistic(np.array([eta]))[0] / mu - 1) < 1e-9
    assert abs(tn.variance_statistic(np.array([eta]))[0] / var - 1) < 1e-6


@pytest.mark.parametrize("u", [-1e5, -1e4, -1e3, -100.0, -20.0001, -19.9999])
def test_ptn_far_tail_moments(u):
    from scipy import integrate as it
    ptn = PositiveTruncatedNormalSpec(1.0)
    t = -u
    f = lambda k: it.quad(lambda w: (w / t) ** k * np.exp(-w - (w / t) ** 2 / 2) / t, 0,
                          np.inf, epsabs=0, epsrel=1e-13, limit=500)[0]
    m0 = f(0)
    mean = f(1) / m0
    var = it.quad(lambda w: ((w / t) - mean) ** 2 * np.exp(-w - (w / t) ** 2 / 2) / t, 0,
                  np.inf, epsabs=0, epsrel=1e-13, limit=500)[0] / m0
    assert abs(ptn.mean_statistic(np.array([u]))[0] / mean - 1) < 1e-9
    assert abs(ptn.variance_statistic(np.array([u]))[0] / var - 1) < 1e-6


def test_ztp_variance_at_large_lambda_and_binomial_curvature_positive():
    ztp = ZeroTruncatedPoissonSpec()
    for e in (40.0, 100.0, 700.0):        # var = mu(1 + lam - mu) -> lam
        assert abs(ztp.variance_statistic(np.array([e]))[0] / math.exp(e) - 1) < 1e-12
    v = BinomialSpec(2).variance_statistic(np.array([-60.0, 40.0, 60.0]))
    assert np.all(v > 0)
    assert abs(v[1] / (2 * math.exp(-40.0)) - 1) < 1e-12
