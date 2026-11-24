"""K-selection Exp2-generalisation: frozen design rules (no inference here).

Decisions D1-D6 of the researcher (2026-11-25) are encoded as pure functions:

  series   O = gaussian X / bernoulli Y      A = poisson X / bernoulli Y
           B = gaussian X / poisson Y        C = bernoulli X / gaussian Y
  regime   a  per-direction signal fixed (PRIMARY)
           b  total signal fixed (PREREGISTERED SENSITIVITY)

  relational weight  O/A/C: a -> 1.0,  b -> sqrt(3/K)
                     B    : a -> 0.12, b -> 0.12 sqrt(3/K)
  loadings           F = f_scale * Q, Q in R^{d x K} with orthonormal columns
                     (canonical-clean-v1 ``build_full_rank_loadings``)
                     a: f_scale = sqrt(d/6)      -> mean ||f_l||^2 = 0.5 K / 3
                     b: f_scale = sqrt(0.5 d/K)  -> mean ||f_l||^2 = 0.5
  intercept w0       B (Poisson-Y): w0 = (K/2) log(1 - w^2)  => E[lambda] = 1
                     C (Gaussian-Y): w0 = 0
                     O/A (Bernoulli-Y): population mean edge probability 0.20,
                     solved deterministically by quadrature + root finding
                     (three independent quadratures must agree; else STOP)
  noise              Gaussian-X variance 0.1, Gaussian-Y variance 1.0 (sd 1.0)

Data are drawn by ``data_generator_canonical.generate_canonical_data`` (literal
draws, no normalisation, no clipping).  The data seed does not depend on the
regime, the candidate K or the initial value, so Z and Q are shared between
regimes (paired design) and every candidate K / initial value of a dataset sees
the same (X, Y) within a regime.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, gammaln

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_generator_canonical import (  # noqa: E402
    GENERATOR_VERSION, generate_canonical_data, poisson_y_moment_existence)
import matched_k_true_design as mkd  # noqa: E402  (density of S_K, quadratures)

DESIGN_VERSION = "k-exp2-generalization-design-v1"

N, D = 150, 15
K_TRUES = (1, 3, 5, 7, 9)
K_CANDIDATES = tuple(range(1, 11))
N_REPLICATES = 10
N_INITS = 2
L_MC = 10
NUM_ITER = 10
SERIES = {"O": ("gaussian", "bernoulli"), "A": ("poisson", "bernoulli"),
          "B": ("gaussian", "poisson"), "C": ("bernoulli", "gaussian")}
SERIES_INDEX = {"O": 0, "A": 1, "B": 2, "C": 3}
REGIMES = ("a", "b")

SIGMA_X_VAR = 0.1
SIGMA_Y_SD = 1.0
GAUSS_Y_W0 = 0.0
BERN_P_TARGET = 0.20
POISSON_Y_MEAN = 1.0
W_REL = {"O": 1.0, "A": 1.0, "C": 1.0, "B": 0.12}
K_REF = 3

# calibration tolerances (frozen)
CAL_ROOT_XTOL = 1e-14
CAL_TRAPEZOID_H = 1e-3
CAL_TOL_W0_AGREE = 1e-8        # |w0(adaptive) - w0(trapezoid)|
CAL_TOL_P_RESID = 1e-10        # |pbar_adaptive(w0) - 0.20|
CAL_TOL_P_INDEP = 1e-8         # |pbar_chi2_hermite(w0) - 0.20|
HERMITE_NODES = 200

SEED_BASE = 20_261_125_000


class DesignStop(RuntimeError):
    """A frozen design rule cannot be satisfied; the run must STOP."""


# --------------------------------------------------------------------------
# parameter rules
# --------------------------------------------------------------------------

def w_of(series: str, regime: str, k: int) -> float:
    base = W_REL[series]
    if regime == "a":
        return float(base)
    if regime == "b":
        return float(base * math.sqrt(K_REF / k))
    raise ValueError(regime)


def f_scale_of(regime: str, k: int) -> float:
    if regime == "a":
        return math.sqrt(D / 6.0)
    if regime == "b":
        return math.sqrt(0.5 * D / k)
    raise ValueError(regime)


def rho_x_of(regime: str, k: int) -> float:
    """Target mean squared row norm of F."""
    return 0.5 * k / 3.0 if regime == "a" else 0.5


def poisson_y_w0(w: float, k: int) -> float:
    """w0 with E[exp(w0 + w z_i.z_j)] = 1 for z_i, z_j iid N(0, I_K).

    E[exp(w S)] = E_{z_j}[exp(w^2 ||z_j||^2 / 2)] = (1 - w^2)^{-K/2}  (|w|<1),
    so exp(w0) (1 - w^2)^{-K/2} = 1  <=>  w0 = (K/2) log(1 - w^2).
    """
    if not abs(w) < 1.0:
        raise DesignStop(f"|w| < 1 required for a finite Poisson-Y mean: w={w}")
    return 0.5 * k * math.log1p(-w * w)


def poisson_y_mean_by_quadrature(w0: float, w: float, k: int) -> float:
    """Independent check: E_R[exp(w0 + w^2 R / 2)], R ~ chi^2_K, by quadrature."""
    logc = -(k / 2) * math.log(2.0) - gammaln(k / 2)
    f = lambda r: math.exp(w0 + 0.5 * w * w * r + logc + (k / 2 - 1) * math.log(r) - r / 2) \
        if r > 0 else 0.0  # noqa: E731
    total = 0.0
    for lo, hi in ((0.0, 1.0), (1.0, math.inf)):
        val, _ = quad(f, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=400)
        total += val
    return total


# --- Bernoulli-Y mean edge probability ------------------------------------

def pbar_adaptive(b: float, w: float, k: int) -> float:
    """E[sigmoid(b + w S_K)] via the exact density of S_K (adaptive quadrature).

    ``mkd._pair`` integrates sigmoid(b+ws)+sigmoid(b-ws) over s >= 0 against
    the symmetric density, which equals E[sigmoid(b + w S)].
    """
    return float(mkd.Adaptive(epsabs=1e-14, epsrel=1e-12).p_bar(b, w, k))


def pbar_trapezoid(b: float, w: float, k: int) -> float:
    return float(mkd.ExpTrapezoid(h=CAL_TRAPEZOID_H).p_bar(b, w, k))


_GH = np.polynomial.hermite_e.hermegauss(HERMITE_NODES)


def pbar_chi2_hermite(b: float, w: float, k: int) -> float:
    """Independent route: S | z_j ~ N(0, R), R = ||z_j||^2 ~ chi^2_K.

    E[sigmoid(b + w S)] = E_R E_U sigmoid(b + w sqrt(R) U), U ~ N(0,1):
    Gauss-Hermite (probabilists') in U, adaptive quadrature in R.
    """
    x, wt = _GH
    wt = wt / math.sqrt(2 * math.pi)
    logc = -(k / 2) * math.log(2.0) - gammaln(k / 2)

    def inner(r):
        if r <= 0:
            return 0.0
        dens = math.exp(logc + (k / 2 - 1) * math.log(r) - r / 2)
        return dens * float(np.dot(wt, expit(b + w * math.sqrt(r) * x)))

    total = 0.0
    for lo, hi in ((0.0, 1.0), (1.0, 60.0), (60.0, math.inf)):
        val, _ = quad(inner, lo, hi, epsabs=1e-13, epsrel=1e-11, limit=400)
        total += val
    return total


def calibrate_bernoulli_w0(w: float, k: int, target: float = BERN_P_TARGET) -> dict:
    """Deterministic w0 with population mean edge probability ``target``.

    Root by brentq on the adaptive quadrature; cross-checked by an independent
    fixed-step quadrature and by the chi^2 x Gauss-Hermite route.  Any check
    outside its frozen tolerance raises DesignStop (no retuning).
    """
    fa = lambda b: pbar_adaptive(b, w, k) - target  # noqa: E731
    ft = lambda b: pbar_trapezoid(b, w, k) - target  # noqa: E731
    lo, hi = -1.0, 1.0
    while fa(lo) > 0:
        lo *= 2.0
        if lo < -1e3:
            raise DesignStop("no lower bracket")
    while fa(hi) < 0:
        hi *= 2.0
        if hi > 1e3:
            raise DesignStop("no upper bracket")
    w0_a = brentq(fa, lo, hi, xtol=CAL_ROOT_XTOL, rtol=4 * np.finfo(float).eps, maxiter=500)
    w0_t = brentq(ft, lo, hi, xtol=CAL_ROOT_XTOL, rtol=4 * np.finfo(float).eps, maxiter=500)
    p_a = pbar_adaptive(w0_a, w, k)
    p_i = pbar_chi2_hermite(w0_a, w, k)
    out = {"w": w, "k": k, "target": target, "w0": w0_a, "w0_trapezoid": w0_t,
           "abs_w0_diff": abs(w0_a - w0_t), "p_adaptive": p_a,
           "p_chi2_hermite": p_i}
    if not (abs(w0_a - w0_t) <= CAL_TOL_W0_AGREE and abs(p_a - target) <= CAL_TOL_P_RESID
            and abs(p_i - target) <= CAL_TOL_P_INDEP):
        raise DesignStop(f"Bernoulli w0 calibration checks failed: {out}")
    return out


def bernoulli_w0_table() -> dict:
    """{(series, regime, k): calibration dict} for series O and A (same w rule)."""
    table = {}
    for regime in REGIMES:
        for k in K_TRUES:
            cal = calibrate_bernoulli_w0(w_of("O", regime, k), k)
            table[(regime, k)] = cal
    return table


def w0_of(series: str, regime: str, k: int, bern_table: dict | None = None) -> float:
    fy = SERIES[series][1]
    w = w_of(series, regime, k)
    if fy == "poisson":
        return poisson_y_w0(w, k)
    if fy == "gaussian":
        return GAUSS_Y_W0
    if bern_table is None:
        raise DesignStop("Bernoulli-Y w0 needs the frozen calibration table")
    return float(bern_table[(regime, k)]["w0"])


# --------------------------------------------------------------------------
# seeds and manifest
# --------------------------------------------------------------------------

def data_seed(series: str, k_true: int, rep: int) -> int:
    """Regime-, candidate-K- and init-independent (paired, same data)."""
    return SEED_BASE + 100_000 * SERIES_INDEX[series] + 1_000 * k_true + rep


def init_seed(dseed: int, t: int) -> int:
    return dseed * 10 + t


def mc_seed(dseed: int, t: int) -> int:
    return dseed * 10 + 5 + t


def fit_id(series, regime, k_true, rep, k, t) -> str:
    return f"{series}-{regime}-KT{k_true}-r{rep}-k{k:02d}-t{t}"


def dataset_id(series, regime, k_true, rep) -> str:
    return f"{series}-{regime}-KT{k_true}-r{rep}"


def build_manifest() -> list[dict]:
    rows = []
    for s in SERIES:
        for regime in REGIMES:
            for kt in K_TRUES:
                for rep in range(N_REPLICATES):
                    ds = data_seed(s, kt, rep)
                    for k in K_CANDIDATES:
                        for t in range(N_INITS):
                            rows.append(dict(
                                fit_id=fit_id(s, regime, kt, rep, k, t),
                                dataset_id=dataset_id(s, regime, kt, rep),
                                series=s, regime=regime, k_true=kt, rep=rep, k=k, t=t,
                                family_x=SERIES[s][0], family_y=SERIES[s][1],
                                data_seed=ds, init_seed=init_seed(ds, t),
                                mc_seed=mc_seed(ds, t)))
    return rows


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    series: str
    regime: str
    k_true: int
    rep: int
    w: float
    w0: float
    f_scale: float
    seed: int


def dataset_spec(series, regime, k_true, rep, bern_table=None) -> DatasetSpec:
    return DatasetSpec(series, regime, k_true, rep, w_of(series, regime, k_true),
                       w0_of(series, regime, k_true, bern_table),
                       f_scale_of(regime, k_true), data_seed(series, k_true, rep))


def generate(spec: DatasetSpec):
    fx, fy = SERIES[spec.series]
    if fy == "poisson":
        me = poisson_y_moment_existence(spec.w)
        if not abs(spec.w) < 0.25:
            raise DesignStop(f"|w| < 1/4 required (finite 4th moment): w={spec.w}")
        if not me["variance_finite"]:
            raise DesignStop("Poisson-Y variance not finite")
    return generate_canonical_data(
        n=N, d=D, k=spec.k_true, seed=spec.seed, family_x=fx, family_y=fy,
        f_scale=spec.f_scale, sigma_x_var=SIGMA_X_VAR, w0=spec.w0, w=spec.w,
        sigma_y_sd=SIGMA_Y_SD)


def array_hash(a: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


