"""Deterministic tests for the forward objective-consistent lineage.

Run with:
    python expfam/src/experimental/test_objective_consistent_numerics.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

_HERE = Path(__file__).parent
_SRC = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

import em_runner  # noqa: E402
from em_runner import build_model, predict_mu_y, run_em_experimental  # noqa: E402
from eval_utils import calc_Q_dual_strict_exp  # noqa: E402
from model_dual_expfam_consistent import (  # noqa: E402
    DualExpFamLSMConsistent,
    DualExpFamLSMPerColumnConsistent,
)
from model_dual_expfam_masked import DualExpFamLSMMasked  # noqa: E402
from model_dual_expfam_nb import DualExpFamLSMNB  # noqa: E402
from model_dual_expfam_percolumn import DualExpFamLSMPerColumn  # noqa: E402
from objective_consistent_numerics import (  # noqa: E402
    bernoulli_curvature,
    bernoulli_log_likelihood,
    bernoulli_mean,
    poisson_curvature,
    poisson_log_likelihood,
    poisson_mean,
)


def _central_gradient(fun, eta, h):
    return (fun(eta + h) - fun(eta - h)) / (2.0 * h)


def _negative_hessian(fun, eta, h):
    return -(fun(eta + h) - 2.0 * fun(eta) + fun(eta - h)) / h ** 2


def test_bernoulli_objective_score_curvature():
    h_grad, h_hess = 1e-6, 1e-4
    eta, x = 0.4, 1.0
    objective = lambda e: float(bernoulli_log_likelihood(x, e))
    score = x - float(bernoulli_mean(eta))
    curvature = float(bernoulli_curvature(eta))
    assert np.isclose(_central_gradient(objective, eta, h_grad), score,
                      rtol=1e-8, atol=1e-10)
    assert np.isclose(_negative_hessian(objective, eta, h_hess), curvature,
                      rtol=1e-6, atol=1e-8)

    # Issue #23 tail: stable objective remains sloped; no probability floor.
    eta, x = 30.0, 0.0
    objective = lambda e: float(bernoulli_log_likelihood(x, e))
    score = x - float(bernoulli_mean(eta))
    assert np.isclose(_central_gradient(objective, eta, 1e-4), score,
                      rtol=1e-9, atol=1e-10)
    assert 0.0 <= float(bernoulli_curvature(eta)) < 1e-12

    # Negative tail uses the same canonical objective.
    eta, x = -30.0, 1.0
    objective = lambda e: float(bernoulli_log_likelihood(x, e))
    score = x - float(bernoulli_mean(eta))
    assert np.isclose(_central_gradient(objective, eta, 1e-4), score,
                      rtol=1e-9, atol=1e-10)
    print("PASS: test_bernoulli_objective_score_curvature")


def test_poisson_objective_score_curvature():
    fixtures = (
        (0.7, 3.0, 1e-6, 1e-4, 2e-6),
        (11.5, 3.0, 1e-5, 1e-3, 2e-6),
        (-25.0, 0.0, 1e-3, 1e-1, 2e-3),
    )
    for eta, x, h_grad, h_hess, rtol in fixtures:
        objective = lambda e: float(poisson_log_likelihood(x, e))
        score = x - float(poisson_mean(eta))
        curvature = float(poisson_curvature(eta))
        assert np.isclose(_central_gradient(objective, eta, h_grad), score,
                          rtol=rtol, atol=1e-10), eta
        assert np.isclose(_negative_hessian(objective, eta, h_hess), curvature,
                          rtol=rtol, atol=1e-12), eta
    print("PASS: test_poisson_objective_score_curvature")


def test_issue23_legacy_counterexamples_and_consistent_fix():
    legacy_p = DualExpFamLSMPerColumn(
        n=1, d=1, k=1, L=1, family_x_list=["poisson"],
        family_y="gaussian")
    consistent_p = DualExpFamLSMPerColumnConsistent(
        n=1, d=1, k=1, L=1, family_x_list=["poisson"],
        family_y="gaussian")
    legacy_b = DualExpFamLSMPerColumn(
        n=1, d=1, k=1, L=1, family_x_list=["bernoulli"],
        family_y="gaussian")
    consistent_b = DualExpFamLSMPerColumnConsistent(
        n=1, d=1, k=1, L=1, family_x_list=["bernoulli"],
        family_y="gaussian")
    for model in (legacy_p, consistent_p, legacy_b, consistent_b):
        model.initialize_params(seed=1)
    F = np.ones((1, 1))
    h = 1e-4

    def x_ll(model, x, eta):
        z = np.array([[[eta]]])
        return model.calc_log_likelihood_X(np.array([[x]]), z, F)

    legacy_p_fd = _central_gradient(lambda e: x_ll(legacy_p, 3.0, e), 11.5, h)
    legacy_p_score = 3.0 - float(legacy_p._mean_function_x(np.array([11.5]))[0])
    assert legacy_p_fd == 0.0 and not np.isclose(legacy_p_fd, legacy_p_score)
    consistent_p_fd = _central_gradient(
        lambda e: x_ll(consistent_p, 3.0, e), 11.5, h)
    consistent_p_score = 3.0 - float(
        consistent_p._mean_function_x(np.array([11.5]))[0])
    assert np.isclose(consistent_p_fd, consistent_p_score, rtol=1e-7)

    legacy_b_fd = _central_gradient(lambda e: x_ll(legacy_b, 0.0, e), 30.0, h)
    legacy_b_score = -float(legacy_b._mean_function_x(np.array([30.0]))[0])
    assert legacy_b_fd == 0.0 and not np.isclose(legacy_b_fd, legacy_b_score)
    consistent_b_fd = _central_gradient(
        lambda e: x_ll(consistent_b, 0.0, e), 30.0, h)
    consistent_b_score = -float(
        consistent_b._mean_function_x(np.array([30.0]))[0])
    assert np.isclose(consistent_b_fd, consistent_b_score, rtol=1e-9)
    print("PASS: test_issue23_legacy_counterexamples_and_consistent_fix")


def test_scalar_x_and_y_paths():
    for family, x, eta in (("bernoulli", 1.0, 0.6),
                           ("poisson", 2.0, 1.2)):
        model = DualExpFamLSMConsistent(
            n=2, d=1, k=1, L=1, family_x=family, family_y=family)
        model.initialize_params(seed=1)
        F = np.ones((1, 1))
        Z_x = np.array([[[eta]], [[eta]]])
        X = np.full((2, 1), x)
        ll_x = model.calc_log_likelihood_X(X, Z_x, F)
        reference_x = 2.0 * float(
            bernoulli_log_likelihood(x, eta) if family == "bernoulli"
            else poisson_log_likelihood(x, eta))
        assert np.isclose(ll_x, reference_x, atol=1e-12)
        expected_mean = (bernoulli_mean(eta) if family == "bernoulli"
                         else poisson_mean(eta))
        expected_curvature = (bernoulli_curvature(eta)
                              if family == "bernoulli"
                              else poisson_curvature(eta))
        assert np.isclose(model._mean_function_x(np.array([eta]))[0],
                          expected_mean)
        assert np.isclose(model._variance_function_x(np.array([eta]))[0],
                          expected_curvature)

        Y = np.array([[0.0, x], [x, 0.0]])
        Z_y = np.zeros((2, 1, 1))
        ll_y = model.calc_log_likelihood_Y(Y, Z_y, eta, 0.0)
        reference_y = float(
            bernoulli_log_likelihood(x, eta) if family == "bernoulli"
            else poisson_log_likelihood(x, eta))
        assert np.isclose(ll_y, reference_y, atol=1e-12)
        assert np.isclose(model._mean_function(np.array([eta]))[0],
                          expected_mean)
        assert np.isclose(model._variance_function(np.array([eta]))[0],
                          expected_curvature)
    print("PASS: test_scalar_x_and_y_paths")


def test_mixed_percolumn_finite_difference():
    model = DualExpFamLSMPerColumnConsistent(
        n=1, d=3, k=2, L=1,
        family_x_list=["gaussian", "bernoulli", "poisson"],
        family_y="gaussian")
    model.initialize_params(seed=1)
    variance = 0.65
    model.params["sigma"] = np.diag([variance, 1.0, 1.0])
    z = np.array([0.35, -0.2])
    F = np.array([[0.7, 0.0], [0.0, 1.25], [0.25, 0.375]])
    X = np.array([[0.4, 1.0, 2.0]])

    def objective(q):
        eta = F @ q
        return float(
            -0.5 * np.log(2.0 * np.pi * variance)
            -0.5 * (X[0, 0] - eta[0]) ** 2 / variance
            +X[0, 1] * eta[1] - np.logaddexp(0.0, eta[1])
            +X[0, 2] * eta[2] - np.exp(eta[2]))

    actual = model.calc_log_likelihood_X(X, z.reshape(1, 2, 1), F)
    assert np.isclose(actual, objective(z), atol=1e-14)
    eta = F @ z
    mu = np.array([eta[0], 1.0 / (1.0 + np.exp(-eta[1])), np.exp(eta[2])])
    score = F.T @ (np.array([1.0 / variance, 1.0, 1.0]) * (X[0] - mu))
    curvature = F.T @ np.diag([
        1.0 / variance,
        mu[1] * (1.0 - mu[1]),
        mu[2],
    ]) @ F
    h_grad, h_hess = 1e-6, 2e-4
    numeric_score = np.empty(2)
    numeric_curvature = np.empty((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h_grad
        numeric_score[j] = (objective(z + e) - objective(z - e)) / (2 * h_grad)
    for j in range(2):
        for k in range(2):
            ej = np.zeros(2); ej[j] = h_hess
            ek = np.zeros(2); ek[k] = h_hess
            numeric_curvature[j, k] = -(
                objective(z + ej + ek) - objective(z + ej - ek)
                - objective(z - ej + ek) + objective(z - ej - ek)
            ) / (4 * h_hess ** 2)
    assert np.max(np.abs(score - numeric_score)) < 5e-10
    assert np.max(np.abs(curvature - numeric_curvature)) < 2e-8
    actual_gradient = model._calc_gradient(
        X, np.zeros((1, 1)), z.reshape(1, 2), F,
        model.params["sigma"], 1.0, 0.0, 0.0, 0)
    actual_precision = model._calc_precision_matrix(
        z.reshape(1, 2), F, model.params["sigma"], 1.0, 0.0, 0.0, 0)
    assert np.allclose(actual_gradient, z - score, atol=1e-12)
    assert np.allclose(actual_precision, np.eye(2) + curvature, atol=1e-12)
    print("PASS: test_mixed_percolumn_finite_difference")


def test_poisson_overflow_and_nonfinite_fail_fast():
    for dtype in (np.float32, np.float64):
        log_max = dtype(np.log(np.finfo(dtype).max))
        safe = np.nextafter(log_max, dtype(-np.inf))
        unsafe = np.nextafter(log_max, dtype(np.inf))
        assert np.all(np.isfinite(poisson_mean(np.array([safe], dtype=dtype))))
        try:
            poisson_mean(np.array([unsafe], dtype=dtype))
            raise AssertionError(f"dtype={dtype} eta={unsafe!r} should fail")
        except FloatingPointError:
            pass

    for eta in (np.inf, -np.inf, np.nan):
        try:
            poisson_mean(np.array([eta], dtype=np.float64))
            raise AssertionError(f"eta={eta!r} should fail")
        except FloatingPointError:
            pass

    model = DualExpFamLSMConsistent(
        n=1, d=1, k=1, L=1, family_x="poisson", family_y="gaussian")
    model.initialize_params(seed=1)
    try:
        model.calc_log_likelihood_X(
            np.array([[0.0]]),
            np.array([[[np.log(np.finfo(np.float64).max) + 1.0]]]),
            np.ones((1, 1)))
        raise AssertionError("model objective should propagate overflow guard")
    except FloatingPointError:
        pass
    print("PASS: test_poisson_overflow_and_nonfinite_fail_fast")


def test_runner_selection_nb_isolation_and_diagnostic():
    legacy = build_model(4, 2, 1, 1, "bernoulli", "poisson")
    consistent = build_model(
        4, 2, 1, 1, "bernoulli", "poisson", numerics_mode="consistent")
    percolumn = build_model(
        4, 2, 1, 1, None, "poisson",
        family_x_list=["gaussian", "poisson"], numerics_mode="consistent")
    nb = build_model(4, 2, 1, 1, "bernoulli", "nb", nb_r=5.0)
    assert type(legacy) is DualExpFamLSMMasked
    assert type(consistent) is DualExpFamLSMConsistent
    assert type(percolumn) is DualExpFamLSMPerColumnConsistent
    assert type(nb) is DualExpFamLSMNB
    try:
        build_model(4, 2, 1, 1, "bernoulli", "nb", nb_r=5.0,
                    numerics_mode="consistent")
        raise AssertionError("consistent + NB must be rejected")
    except NotImplementedError:
        pass

    X, Y = _small_poisson_data(n=8, d=3, seed=13)
    default = run_em_experimental(
        X, Y, family_x="poisson", family_y="poisson", k=1, L=1,
        num_iter=1, seed=4)
    explicit_legacy = run_em_experimental(
        X, Y, family_x="poisson", family_y="poisson", k=1, L=1,
        num_iter=1, seed=4, numerics_mode="legacy")
    for key in ("Z_est", "Z_samples", "F", "sigma"):
        assert np.array_equal(default[key], explicit_legacy[key]), key
    for key in ("w0", "w", "var_z", "sigma_y_est", "Q_strict", "bic",
                "num_params", "nan_occurred", "nan_count"):
        assert default[key] == explicit_legacy[key], key
    res = run_em_experimental(
        X, Y, family_x="poisson", family_y="poisson", k=1, L=1,
        num_iter=1, seed=4, numerics_mode="consistent",
        compute_clip_diagnostic=True)
    assert res["numerics_mode"] == "consistent"
    assert res["clip_diag"]["status"] == "not_applicable"
    assert res["clip_diag"]["x_side"] is None
    # Post-hoc prediction also avoids the legacy display clip in consistent mode.
    fake = dict(res)
    fake["Z_est"] = np.zeros_like(res["Z_est"])
    fake["w0"] = 12.0
    fake["w"] = 0.0
    assert np.all(predict_mu_y(fake) > 1e5)
    legacy_fake = dict(fake, numerics_mode="legacy", model=legacy)
    assert np.all(predict_mu_y(legacy_fake, clip_max=1e3) == 1e3)

    # Gaussian prediction semantics remain exactly legacy (including display clip).
    gaussian_legacy = build_model(2, 1, 1, 1, "gaussian", "gaussian")
    gaussian_consistent = build_model(
        2, 1, 1, 1, "gaussian", "gaussian", numerics_mode="consistent")
    gaussian_result = {
        "Z_est": np.zeros((2, 1)), "w0": -3.0, "w": 0.0,
        "model": gaussian_consistent, "numerics_mode": "consistent",
    }
    assert np.array_equal(predict_mu_y(gaussian_result), np.zeros((2, 2)))
    gaussian_result.update(w0=2e5, model=gaussian_legacy,
                           numerics_mode="legacy")
    legacy_prediction = predict_mu_y(gaussian_result)
    gaussian_result.update(model=gaussian_consistent,
                           numerics_mode="consistent")
    assert np.array_equal(predict_mu_y(gaussian_result), legacy_prediction)
    assert np.all(legacy_prediction == 1e5)
    print("PASS: test_runner_selection_nb_isolation_and_diagnostic")


def _small_poisson_data(n=10, d=3, seed=7):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0.0, 0.4, size=(n, 1))
    F = rng.normal(0.0, 0.3, size=(d, 1))
    X = rng.poisson(np.exp(Z @ F.T)).astype(float)
    eta_y = 0.3 + 0.15 * (Z @ Z.T)
    Y = np.zeros((n, n))
    upper = np.triu_indices(n, 1)
    Y[upper] = rng.poisson(np.exp(eta_y[upper]))
    Y += Y.T
    return X, Y


def test_runner_smokes_and_q_overflow_propagation():
    X, Y = _small_poisson_data()
    scalar = run_em_experimental(
        X, Y, family_x="poisson", family_y="poisson", k=1, L=2,
        num_iter=2, seed=5, numerics_mode="consistent")
    assert type(scalar["model"]) is DualExpFamLSMConsistent
    assert np.isfinite(scalar["Q_strict"]) and not scalar["nan_occurred"]

    X_mixed = X.copy()
    X_mixed[:, 0] = np.linspace(-0.5, 0.5, len(X))
    X_mixed[:, 1] = (X_mixed[:, 1] > 0).astype(float)
    mixed = run_em_experimental(
        X_mixed, Y, family_x=None, family_y="poisson", k=1, L=2,
        num_iter=2, seed=5,
        family_x_list=["gaussian", "bernoulli", "poisson"],
        numerics_mode="consistent")
    assert type(mixed["model"]) is DualExpFamLSMPerColumnConsistent
    assert np.isfinite(mixed["Q_strict"]) and not mixed["nan_occurred"]

    original = em_runner.calc_Q_dual_strict_exp
    def _overflow(*args, **kwargs):
        raise FloatingPointError("forced overflow guard")
    em_runner.calc_Q_dual_strict_exp = _overflow
    try:
        try:
            run_em_experimental(
                X, Y, family_x="poisson", family_y="poisson", k=1, L=1,
                num_iter=1, seed=5, numerics_mode="consistent")
            raise AssertionError("consistent Q overflow must propagate")
        except FloatingPointError:
            pass
    finally:
        em_runner.calc_Q_dual_strict_exp = original
    print("PASS: test_runner_smokes_and_q_overflow_propagation")


def test_strict_q_constants_and_gaussian_regression():
    model = DualExpFamLSMConsistent(
        n=3, d=1, k=1, L=1, family_x="poisson", family_y="poisson")
    model.initialize_params(seed=1)
    X = np.array([[0.0], [1.0], [2.0]])
    Y = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    Zs = np.array([[[0.1]], [[-0.2]], [[0.3]]])
    F = np.array([[0.4]])
    sigma = np.eye(1)
    q = calc_Q_dual_strict_exp(X, Y, Zs, F, sigma, 1.0, 0.2, 0.1, model)
    lnpz = -1.5 * np.log(2.0 * np.pi) - 0.5 * np.sum(Zs[:, :, 0] ** 2)
    upper = np.triu_indices(3, 1)
    manual = (lnpz + model.calc_log_likelihood_X(X, Zs, F)
              + model.calc_log_likelihood_Y(Y, Zs, 0.2, 0.1)
              - np.sum(gammaln(X + 1)) - np.sum(gammaln(Y[upper] + 1)))
    assert np.isclose(q, manual, atol=1e-12)

    legacy = DualExpFamLSMMasked(
        n=4, d=2, k=1, L=1, family_x="gaussian", family_y="gaussian")
    consistent = DualExpFamLSMConsistent(
        n=4, d=2, k=1, L=1, family_x="gaussian", family_y="gaussian")
    for item in (legacy, consistent):
        item.initialize_params(seed=2)
        item.sigma_y = 0.7
        item.params["sigma"] = np.diag([0.4, 1.3])
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(4, 1)); Zs = Z[:, :, None]
    F = rng.normal(size=(2, 1)); X = rng.normal(size=(4, 2))
    Y = rng.normal(size=(4, 4)); Y = np.triu(Y, 1); Y += Y.T
    assert legacy.calc_log_likelihood_X(X, Zs, F) == \
        consistent.calc_log_likelihood_X(X, Zs, F)
    assert legacy.calc_log_likelihood_Y(Y, Zs, 0.2, 0.1) == \
        consistent.calc_log_likelihood_Y(Y, Zs, 0.2, 0.1)
    for i in range(4):
        assert np.array_equal(
            legacy._calc_gradient(X, Y, Z, F, legacy.params["sigma"],
                                  1.0, 0.2, 0.1, i),
            consistent._calc_gradient(X, Y, Z, F, consistent.params["sigma"],
                                      1.0, 0.2, 0.1, i))
        assert np.array_equal(
            legacy._calc_precision_matrix(
                Z, F, legacy.params["sigma"], 1.0, 0.2, 0.1, i),
            consistent._calc_precision_matrix(
                Z, F, consistent.params["sigma"], 1.0, 0.2, 0.1, i))
    assert np.array_equal(legacy.calc_F(X, Zs), consistent.calc_F(X, Zs))
    F_new = legacy.calc_F(X, Zs)
    assert np.array_equal(legacy.calc_sigma(X, Zs, F_new),
                          consistent.calc_sigma(X, Zs, F_new))
    assert legacy.calc_sigma_y(Y, Zs, 0.2, 0.1) == \
        consistent.calc_sigma_y(Y, Zs, 0.2, 0.1)
    print("PASS: test_strict_q_constants_and_gaussian_regression")


if __name__ == "__main__":
    tests = (
        test_bernoulli_objective_score_curvature,
        test_poisson_objective_score_curvature,
        test_issue23_legacy_counterexamples_and_consistent_fix,
        test_scalar_x_and_y_paths,
        test_mixed_percolumn_finite_difference,
        test_poisson_overflow_and_nonfinite_fail_fast,
        test_runner_selection_nb_isolation_and_diagnostic,
        test_runner_smokes_and_q_overflow_propagation,
        test_strict_q_constants_and_gaussian_regression,
    )
    for test in tests:
        test()
    print(f"\nALL OBJECTIVE-CONSISTENCY TESTS PASSED ({len(tests)}/{len(tests)})")
