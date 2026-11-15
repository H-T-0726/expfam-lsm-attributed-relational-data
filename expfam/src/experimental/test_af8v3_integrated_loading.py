"""AF8-v3 regression tests (numerical only; no labels, no accuracy)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_af8_integrated_loading as a8      # noqa: E402
import run_af8v2_integrated_loading as v2    # noqa: E402
import run_af8v3_integrated_loading as v3    # noqa: E402

FIX = v3.OUT / "regression" / "known_failed_draws_v2.csv"


def quad_vgh(A, b, q=0.0):
    """phi = -1/2 x'Ax + b'x, value rounded to a grid q (emulates floating-point resolution); exact gradient."""
    def vgh(x):
        v = -0.5 * x @ A @ x + b @ x
        if q:
            v = np.round(v / q) * q
        return v, b - A @ x, A.copy()
    return vgh


def test_01_resolution_stall_is_resolved_by_polish_and_certified():
    A = np.array([[3.0, 0.4], [0.4, 1.5]])
    b = np.array([1.0, -2.0])
    xs = np.linalg.solve(A, b)
    vgh = quad_vgh(A, b, q=1e-8)                 # phi resolution 1e-8: Armijo cannot see gains < 1e-8
    x0 = xs + np.array([2e-5, -1e-5])            # decrement/2 ~ 1e-10 < resolution
    r2 = v2.newton_max(vgh, x0, 1e-12, 50, 2.0, accept_gtol=1e-12)
    assert r2.status != "OK"                     # v2 logic cannot reach gtol (stall)
    r3 = v3.newton_max3(vgh, x0, 1e-12, 50, 2.0)
    assert r3.status == "OK" and r3.lam2_half <= v3.CERT_DEC
    assert np.max(np.abs(r3.x - xs)) < 1e-12


def test_02_certificate_rejects_large_decrement():
    A = np.eye(2)
    b = np.zeros(2)
    vgh = quad_vgh(A, b)
    x = np.array([3e-6, 0.0])                    # lambda^2/2 = 4.5e-12 > CERT_DEC
    v, g, H = vgh(x)
    ok, info = v3.certify(vgh, x, v, g, H)
    assert not ok and info["lam2_half"] > v3.CERT_DEC


def test_03_certificate_rejects_large_laplace_change():
    """small decrement but log|negH| varies fast: the extra-Newton-step Laplace check must fail."""
    def vgh(x):
        v = -0.5 * x[0] ** 2
        return v, np.array([-x[0]]), np.array([[np.exp(1e6 * x[0])]])   # artificial curvature field
    x = np.array([1e-7])
    v, g, H = vgh(x)
    ok, info = v3.certify(vgh, x, v, g, H)
    assert info["lam2_half"] <= v3.CERT_DEC
    assert not ok and info["lap_step_change"] > v3.CERT_LAP


def test_04_saddle_never_certified():
    def vgh(x):
        v = -(x[0] ** 2 - 1) ** 2 + x[1] ** 2 * 0.0 - x[1] ** 2
        g = np.array([-4 * x[0] * (x[0] ** 2 - 1), -2 * x[1]])
        H = np.array([[12 * x[0] ** 2 - 4, 0.0], [0.0, 2.0]])
        return v, g, H
    r = v3.newton_max3(vgh, np.array([0.0, 0.3]), 1e-10, 100, 1.0)
    if r.status == "OK":
        _, _, H = vgh(r.x)
        assert np.linalg.eigvalsh(H)[0] > 0 and abs(abs(r.x[0]) - 1) < 1e-8
    else:
        assert r.status in ("SADDLE", "NOT_CERTIFIED", "LINESEARCH_FAIL", "NOT_CONVERGED")


def test_05_polish_never_enters_non_pd_region():
    """from a point where the pure Newton step jumps to a non-concave region, polish must refuse the step."""
    def vgh(x):                                   # phi = -x^4/4 + x^2/2 * 0 ... concave only for |x| small? use cos
        v = np.cos(x[0])
        return v, np.array([-np.sin(x[0])]), np.array([[np.cos(x[0])]])
    x = np.array([1.4])                           # negH = cos(1.4) > 0, Newton step -> x - tan(1.4) ~ -4.4 (cos < 0)
    v, g, H = vgh(x)
    xp, *_ , n = v3.polish(vgh, x, v, g, H)
    assert n == 0 and xp[0] == 1.4


class _FakeJ:
    pv = 1.0


class _FakeL:
    """S fails (cold) at draws whose first coordinate > thr; rescue returns `rescue_phi_shift`-shifted mode."""
    def __init__(self, thr, rescue_ok, phi_shift):
        self.J, self.thr, self.rescue_ok, self.shift = _FakeJ(), thr, rescue_ok, phi_shift

    def S(self, f):
        if f[0] > self.thr:
            return v3.SVal3(False, float("nan"), None, "S_NOT_CERTIFIED", phi_last=-1.0, grad_inf=1e-6)
        return v3.SVal3(True, -0.5 * float(f @ f), None, "OK", phi_last=-1.0)

    def solve_from(self, f, Z0):
        if not self.rescue_ok:
            return v3.SVal3(False, float("nan"), None, "S_NOT_CERTIFIED", phi_last=-2.0)
        return v3.SVal3(True, -0.5 * float(f @ f), None, "OK", phi_last=-1.0 + self.shift)


def test_06_IS_failed_draw_invalidates_estimate_and_stays_in_denominator():
    L = _FakeL(thr=1.5, rescue_ok=False, phi_shift=0.0)
    r = v3.method_IS(L, np.zeros(3), np.eye(3), None, seed=1, draws=400)
    assert r["IS_failed"] > 0 and np.isnan(r["IS"]) and r["IS_status"] == "PARTIAL_FAIL"
    assert r["IS_finite"] + r["IS_failed"] + r["IS_rescued"] == 400
    log = json.loads(r["IS_draw_log"])
    assert len(log) == r["IS_failed"] and all(e["outcome"] == "FAILED" for e in log)


def test_07_IS_rescue_only_for_same_mode():
    same = v3.method_IS(_FakeL(1.5, True, 1e-9), np.zeros(3), np.eye(3), None, seed=1, draws=400)
    assert same["IS_failed"] == 0 and same["IS_rescued"] > 0 and np.isfinite(same["IS"])
    other = v3.method_IS(_FakeL(1.5, True, 0.5), np.zeros(3), np.eye(3), None, seed=1, draws=400)
    assert other["IS_rescued"] == 0 and other["IS_failed"] == same["IS_rescued"] and np.isnan(other["IS"])


class _QuadL:
    """T(f) = -1/2 (f-m)'A(f-m); valid everywhere."""
    def __init__(self, A, m):
        self.A, self.m = A, m

    def T(self, f):
        d = np.asarray(f) - self.m
        return True, -0.5 * float(d @ self.A @ d)


def test_08_outer_certificate_on_quadratic_and_indefinite():
    m = np.array([0.3, -1.0, 2.0])
    A = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 4.0]])
    r = v3.certify_outer(_QuadL(A, m), m + 0.2)
    assert r["status"] == "CERT_OK" and np.max(np.abs(r["f"] - m)) < 1e-6 and r["lam2_half_f"] <= v3.OUTER_CERT_DEC
    r2 = v3.certify_outer(_QuadL(np.diag([1.0, -1.0, 1.0]), m), m)
    assert r2["status"] == "T_STATIONARY_NOT_MAX"


@pytest.mark.skipif(not FIX.exists(), reason="fixtures not generated")
def test_09_known_v2_failed_draws_are_certified_same_mode():
    fx = pd.read_csv(FIX)
    assert len(fx) == 9
    cache = {}
    for _, r in fx.iterrows():
        key = (int(r.replicate[3:]) - 1, r.regime)
        if key not in cache:
            cs = a8.load_case(*key)
            cs["regime"] = r.regime
            cache[key] = cs
        case = cache[key]
        J, _ = a8.joint_for(case, int(r.column), r.candidate, a8.MAIN_PRIOR)
        L = v3.Local3(J, np.asarray(case["st"]["Zmode"], float))
        f = np.array([float(a) for a in r.f.split("|")])
        s = L.S(f)
        assert s.ok, (r.to_dict(), s)
        assert s.lam2_half <= v3.CERT_DEC and s.lap_step_change <= v3.CERT_LAP
        assert abs(s.phi_last - r.v2_phi) <= 1e-9          # same mode as the v2 stall point


@pytest.mark.skipif(not FIX.exists(), reason="fixtures not generated")
def test_10_determinism_fresh_objects():
    r = pd.read_csv(FIX).iloc[0]
    case = a8.load_case(int(r.replicate[3:]) - 1, r.regime)
    case["regime"] = r.regime
    f = np.array([float(a) for a in r.f.split("|")])
    vals = []
    for pre in (None, f + 0.3, np.zeros(3)):
        L = v3.Local3(a8.joint_for(case, int(r.column), r.candidate, a8.MAIN_PRIOR)[0], np.asarray(case["st"]["Zmode"], float))
        if pre is not None:
            L.T(pre)
        vals.append(L.T(f)[1])
    assert max(vals) - min(vals) == 0.0


def test_11_outer_known_cells_certified():
    for rep, g, j, pr in (("rep01", "sparse", 4, "A_sf2_0.667"), ("rep02", "sparse", 5, a8.MAIN_PRIOR)):
        case = a8.load_case(int(rep[3:]) - 1, g)
        case["regime"] = g
        J, _ = a8.joint_for(case, j, a8.C, pr)
        Za, starts = v3.make_starts(case, j, a8.C)
        de = v3.method_D_E2(v3.Local3(J, Za), starts)
        assert de["E2_status"] == "OK", de.get("D_start_status")
        assert de["D_lam2_half_f"] <= v3.OUTER_CERT_DEC and de["D_min_eig_f"] > 0


def test_12_parent_artifacts_unchanged():
    paths = ["expfam/results/auto_family_z_recovery/af8_integrated_loading_20261115",
             "expfam/results/auto_family_z_recovery/af8v2_numerical_repair_20261115",
             "expfam/src/experimental/run_af8_integrated_loading.py", "expfam/src/experimental/analyze_af8_integrated_loading.py",
             "expfam/src/experimental/run_af8v2_integrated_loading.py", "expfam/src/experimental/analyze_af8v2_integrated_loading.py",
             "expfam/src/experimental/test_af8v2_integrated_loading.py"]
    r = subprocess.run(["git", "diff", "--quiet", v3.V2_HEAD, "--", *paths], cwd=a8.ROOT)
    assert r.returncode == 0
