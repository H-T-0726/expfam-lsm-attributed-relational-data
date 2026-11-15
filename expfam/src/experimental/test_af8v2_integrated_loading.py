"""Numerical regression tests for AF8-v2 (no MCEM)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8        # noqa: E402
import run_af8v2_integrated_loading as v2      # noqa: E402

ROOT = HERE.parent.parent.parent
AF6_STATE = Path(r"D:\tento\kennkyu_artifacts\af6_laplace_confirmation_20261114\fits")


# toy: Psi(x, y) = -(x^2 - 1)^2 - y^2 ; (0, 0) is a stationary point with indefinite negH (saddle of the maximisation)
def toy_vgh(z):
    x, y = z
    v = -(x * x - 1) ** 2 - y * y
    g = np.array([-4 * x * (x * x - 1), -2 * y])
    H = np.array([[12 * x * x - 4, 0.0], [0.0, 2.0]])
    return v, g, H


def test_01_saddle_is_not_accepted_as_maximum():
    r = v2.newton_max(toy_vgh, np.array([0.0, 0.0]), 1e-10, 100, 1.0)
    assert r.status in ("OK", "OK_FLOOR") and abs(abs(r.x[0]) - 1) < 1e-8 and r.min_eig > 0
    assert r.n_negcurv_escape >= 1


def test_02_indefinite_hessian_gives_ascent_steps():
    vals = []

    def rec(z):
        out = toy_vgh(z)
        vals.append(out[0])
        return out
    r = v2.newton_max(rec, np.array([0.1, 0.5]), 1e-10, 100, 1.0)   # negH_xx < 0 at the start
    assert r.status in ("OK", "OK_FLOOR") and r.n_modified >= 1
    assert r.value > toy_vgh(np.array([0.1, 0.5]))[0]


def test_03_failed_line_search_does_not_move():
    x0 = np.array([0.3, 0.2])

    def only_start(z):
        if np.array_equal(z, x0):
            return toy_vgh(z)
        raise v2.EvalFail("invalid everywhere else")
    r = v2.newton_max(only_start, x0, 1e-10, 50, 1.0)
    assert r.status == "LINESEARCH_FAIL" and np.array_equal(r.x, x0) and np.isfinite(r.value)


def _case(r, g):
    case = a8.load_case(r, g)
    case["regime"] = g
    return case


needs_state = pytest.mark.skipif(not AF6_STATE.exists(), reason="AF6 fit states not available")


@needs_state
def test_04_heavy_tailed_poisson_column_no_exception_and_valid_maximum():
    case = _case(0, "sparse")                      # rep01 sparse column 3 (x_max 86): AF8 failures
    J, _ = a8.joint_for(case, 3, a8.P, a8.MAIN_PRIOR)
    Za = np.asarray(case["st"]["Zmode"], float)
    L = v2.Local(J, Za)
    out = v2.method_D_E2(L, {"plugin_anchorZ": a8.fixed_z_start(a8.P, case["X"][:, 3], Za), "zero": np.zeros(3)})
    assert out["E2_status"] == "OK" and out["D_n_valid_starts"] == 2 and out["D_multistart"] == "AGREE"
    assert np.isfinite(out["E2"])


@needs_state
def test_05_T_is_deterministic_in_evaluation_order():
    case = _case(1, "sparse")
    Za = np.asarray(case["st"]["Zmode"], float)
    fs = [np.array([0.4, -0.2, 0.1]), np.zeros(3), np.array([-0.8, 0.5, 0.3])]
    vals = []
    for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        L = v2.Local(a8.joint_for(case, 8, a8.P, a8.MAIN_PRIOR)[0], Za)
        res = {}
        for i in order:
            res[i] = L.T(fs[i])
        vals.append([res[i] for i in range(3)])
    assert vals[0] == vals[1] == vals[2]


def _toy_joint(j=3, fam="cmp_nu2p0", seed=3):
    rng = np.random.default_rng(seed)
    n, k = 12, 3
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((5, k)) * 0.6
    fams = ["gaussian", "poisson", "poisson", "cmp_nu2p0", "cmp_nu2p0"]
    eta = Z @ F.T
    X = np.column_stack([eta[:, 0] + rng.standard_normal(n), rng.poisson(np.exp(eta[:, 1])), rng.poisson(np.exp(eta[:, 2])),
                         rng.poisson(np.exp(eta[:, 3] / 2)), rng.poisson(np.exp(eta[:, 4] / 2))])
    Y = (rng.random((n, n)) < 0.4).astype(float)
    Y = np.triu(Y, 1) + np.triu(Y, 1).T
    train = rng.random((n, n)) < 0.7
    fams[j] = fam
    return a8.JointU(X, Y, train, fams, F, np.ones(5), -1.0, 0.9, 1.0, j, 0.5), Z, F


def test_06_fd_gradient_and_full_hessian():
    J, Z, F = _toy_joint()
    n, k = Z.shape
    Z0, f0 = Z + 0.1, F[3] + 0.2
    u0 = np.concatenate([Z0.ravel(), f0])

    def split(u):
        return u[:n * k].reshape(n, k), u[n * k:]
    g = np.concatenate([a.ravel() for a in J.grad_u(Z0, f0)])
    H = J.negH_u(Z0, f0)
    h = 1e-6
    for q in range(len(u0)):
        e = np.zeros(len(u0))
        e[q] = h
        assert abs((J.psi(*split(u0 + e)) - J.psi(*split(u0 - e))) / (2 * h) - g[q]) < 1e-6 * max(1, abs(g[q]))
        gp = np.concatenate([a.ravel() for a in J.grad_u(*split(u0 + e))])
        gm = np.concatenate([a.ravel() for a in J.grad_u(*split(u0 - e))])
        assert np.max(np.abs(-(gp - gm) / (2 * h) - H[:, q])) < 1e-6 * max(1, np.max(np.abs(H[:, q])))


def test_07_schur_complement_equals_curvature_of_profiled_joint():
    """envelope theorem: -d2/df2 max_Z Psi(Z, f) = H_ff - H_fZ H_ZZ^-1 H_Zf at the joint mode."""
    J, Z, F = _toy_joint()
    n, k = Z.shape

    def vgh_u(u):
        Zu, f = u[:-k].reshape(n, k), u[-k:]
        gZ, gf = J.grad_u(Zu, f)
        return J.psi(Zu, f), np.concatenate([gZ.ravel(), gf]), J.negH_u(Zu, f)
    r = v2.newton_max(vgh_u, np.concatenate([Z.ravel(), F[3]]), 1e-10, 200, 2.0)
    assert r.status in ("OK", "OK_FLOOR")
    Zh, fh = r.x[:-k].reshape(n, k), r.x[-k:]
    H = J.negH_u(Zh, fh)
    nk = n * k
    schur = H[nk:, nk:] - H[nk:, :nk] @ np.linalg.solve(H[:nk, :nk], H[:nk, nk:])

    def prof(f):
        def vgh(zv):
            J.F[J.j] = f
            Zz = zv.reshape(n, k)
            return J.phi(Zz), J.gradZ(Zz).ravel(), J.negHZZ(Zz)
        rr = v2.newton_max(vgh, Zh.ravel(), 1e-11, 200, 2.0)
        return rr.value + a8.log_prior(f, J.pv)
    h = 1e-4
    Hp = np.zeros((k, k))
    p0 = prof(fh)
    for a in range(k):
        for b in range(k):
            ea, eb = np.eye(k)[a] * h, np.eye(k)[b] * h
            Hp[a, b] = -(prof(fh + ea + eb) - prof(fh + ea - eb) - prof(fh - ea + eb) + prof(fh - ea - eb)) / (4 * h * h)
    assert abs(prof(fh) - p0) < 1e-12
    assert np.max(np.abs(Hp - schur)) < 1e-4 * max(1, np.max(np.abs(schur)))


def test_08_prior_normalisation_and_jacobian():
    from scipy.stats import multivariate_normal
    for cand in a8.CANDS:
        v = a8.prior_var("B_sg2_0.667", cand)
        a = a8.A2_ZERO[cand]
        sd = math.sqrt(v)
        grid = np.linspace(-7 * sd, 7 * sd, 41)
        d = grid[1] - grid[0]
        tot = sum(math.exp(a8.log_prior(np.array([x, y, z]), v)) for x in grid for y in grid for z in grid) * d ** 3
        assert abs(tot - 1) < 1e-6
        f = np.array([0.3, -1.2, 0.7])
        via_g = multivariate_normal(np.zeros(3), (2 / 3) * np.eye(3)).logpdf(math.sqrt(a) * f) + 1.5 * math.log(a)
        assert abs(a8.log_prior(f, v) - via_g) < 1e-12


def test_09_cmp2_bessel_stable():
    eta = np.array([-5.0, 0.0, 6.0, 18.0])
    k = np.arange(200000)
    brute = np.array([float(np.logaddexp.reduce(k * e - 2 * gammaln(k + 1))) for e in eta])
    assert np.max(np.abs(a8.cmp2_A(eta) - brute) / np.maximum(1, np.abs(brute))) < 1e-12
    big = np.array([40.0, 51.8, 80.0])
    assert np.all(np.isfinite(a8.cmp2_A(big)) & np.isfinite(a8.cmp2_dA(big)) & (a8.cmp2_d2A(big) > 0))
    u = 2 * np.exp(big / 2)
    assert np.max(np.abs(a8.cmp2_d2A(big) / (u / 4) - 1)) < 1e-3


class _FakeL:
    """T(f) = log N(f; 0, I) + c ; invalid (numerical failure) where f[0] > thr."""
    def __init__(self, thr=np.inf, c=2.0):
        self.thr, self.c = thr, c

    def T(self, f):
        if f[0] > self.thr:
            return False, float("nan")
        return True, float(-0.5 * f @ f - 1.5 * math.log(2 * math.pi) + self.c)


def test_10_is_denominator_and_failed_draws():
    r = v2.method_IS(_FakeL(), np.zeros(3), np.eye(3), seed=1, draws=2000)
    assert r["IS_draws"] == 2000 and r["IS_failed"] == 0 and r["IS_status"] == "OK"
    assert abs(r["IS"] - 2.0) < 4 * r["IS_mc_se"] + 1e-3            # exact log integral = c
    rf = v2.method_IS(_FakeL(thr=1.0), np.zeros(3), np.eye(3), seed=1, draws=2000)
    assert rf["IS_failed"] > 0 and rf["IS_finite"] + rf["IS_failed"] == 2000
    assert rf["IS_status"] == "PARTIAL_FAIL" and not np.isfinite(rf["IS"])       # failures are not zero weights
    assert rf["IS_lower_bound_if_failed"] < r["IS"]                              # denominator kept at 2000


def test_11_candidate_dependent_failure_is_not_a_selection():
    ok = {"status": "OK", "E2": -100.0}
    bad = {"status": "FAILED"}
    assert v2.selection_state(ok, bad)[0] == "UNDECIDED_ONE_FAILED" and v2.selection_state(ok, bad)[1] is None
    assert v2.selection_state(bad, bad)[0] == "UNDECIDED_BOTH_FAILED"
    assert v2.selection_state(ok, {"status": "OK", "E2": -100.005})[0] == "UNCERTAIN_NUMERIC"
    st, sel, d = v2.selection_state(ok, {"status": "OK", "E2": -98.0})
    assert st == "DECIDED" and sel == a8.C and d == 2.0
    assert v2.selection_state(ok, {"status": "FLAGGED", "E2": -98.0})[0] == "UNDECIDED_ONE_FAILED"


def test_12_af8_artifacts_unchanged():
    paths = ["expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115",
             "expfam/src/experimental/run_af8_integrated_loading.py",
             "expfam/src/experimental/analyze_af8_integrated_loading.py",
             "expfam/src/experimental/test_af8_integrated_loading.py"]
    r = subprocess.run(["git", "diff", "--quiet", v2.AF8_HEAD, "--", *paths], cwd=ROOT)
    assert r.returncode == 0
    r2 = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    assert r2.stdout.strip() == ""
