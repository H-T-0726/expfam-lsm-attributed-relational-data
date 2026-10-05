"""Generic scalar exponential-family specifications for X columns.

Forward-only (multi-support family selection, 2026-10-06).  Nothing here
changes an existing model, generator or selector; the existing Gaussian /
Bernoulli / Poisson code paths keep their own numerics, and a regression test
pins that the specs below reproduce them.

Every non-Gaussian family is written in canonical form with dispersion 1::

    log p_m(x | eta) = log h_m(x) + eta * T_m(x) - A_m(eta),   eta = f_l^T z_i

with ``eta`` used directly as the natural parameter (no intercept, no extra
link).  Family-specific shape / nuisance constants (Binomial trials, the
COM-Poisson nu, the sigma of the Normal-derived families, the centre of the
(0,1) truncated Normal) are FIXED by protocol and never estimated.

The Gaussian keeps the existing dispersion form with a profiled column
variance: it is the REAL-support control and does not compete with anything.

Numerics policy
---------------
No clipping, no flooring and no substitution.  A non-finite input or output
raises ``FloatingPointError``; an observation outside a family's support raises
``SupportViolation`` from ``log_prob``.  The COM-Poisson series is truncated
adaptively with an explicit tail bound and raises if the bound cannot be met
within a hard cap.

The mathematical audit behind every formula is
``reports/distribution_selection/family_candidate_compatibility_audit_20261006.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import expit, gammaln, log_ndtr, ndtr
from scipy.stats import truncnorm

from objective_consistent_numerics import (
    bernoulli_curvature,
    bernoulli_log_likelihood,
    bernoulli_mean,
    poisson_curvature,
    poisson_mean,
)

FAMILY_SPECS_VERSION = "family-specs-v1"

GROUP_COUNT = "GROUP_COUNT"
GROUP_POSITIVE = "GROUP_POSITIVE_CONTINUOUS"
GROUP_BOUNDED = "GROUP_BOUNDED_CONTINUOUS"
GROUP_REAL = "GROUP_REAL"
SUPPORT_GROUPS = (GROUP_COUNT, GROUP_POSITIVE, GROUP_BOUNDED, GROUP_REAL)

BASE_MEASURE = {
    GROUP_COUNT: "counting measure on {0,1,2,...}",
    GROUP_POSITIVE: "Lebesgue measure on (0, inf)",
    GROUP_BOUNDED: "Lebesgue measure on (0, 1)",
    GROUP_REAL: "Lebesgue measure on R",
}

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# COM-Poisson adaptive truncation.  The series is cut at M once the log of the
# geometric tail bound, relative to log Z, is below this value for every eta in
# the batch; exp(-40) ~ 4e-18 is below double precision relative resolution.
CMP_LOG_TAIL_TOL = -40.0

# Below these magnitudes the closed forms cancel catastrophically and a series
# is used instead.  Each threshold is far inside the series' accurate range
# (first omitted term ~1e-20 or smaller) and far outside the closed form's
# cancellation range (relative error ~1e-13 or smaller at the threshold).
CB_SERIES_ABS_ETA = 1e-2
ZTP_SERIES_LAMBDA = 1e-2


class SupportViolation(ValueError):
    """An observation lies outside a family's support (probability zero)."""


def _finite(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"{name} contains a non-finite value")
    return arr


def _checked(value: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} produced a non-finite value")
    return value


def log_ndtr_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``log(Phi(b) - Phi(a))`` for ``a < b`` without CDF-difference cancellation."""

    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64),
                               np.asarray(b, dtype=np.float64))
    if np.any(~(a < b)):
        raise FloatingPointError("log_ndtr_diff requires a < b elementwise")
    out = np.empty_like(a)
    left = b <= 0.0                # both in the lower tail
    right = a >= 0.0               # both in the upper tail: use symmetry
    middle = ~(left | right)
    if np.any(left):
        lb, la = log_ndtr(b[left]), log_ndtr(a[left])
        out[left] = lb + np.log(-np.expm1(la - lb))
    if np.any(right):
        la, lb = log_ndtr(-a[right]), log_ndtr(-b[right])
        out[right] = la + np.log(-np.expm1(lb - la))
    if np.any(middle):
        out[middle] = np.log1p(-(ndtr(a[middle]) + ndtr(-b[middle])))
    return _checked(out, "log_ndtr_diff")


def _log_phi(u: np.ndarray) -> np.ndarray:
    return -0.5 * u * u - LOG_SQRT_2PI


# ==========================================================================
# base class
# ==========================================================================

@dataclass(frozen=True)
class FamilySpec:
    """One candidate model: a family plus its fixed hyperparameters."""

    family_name: str
    candidate_model: str
    support_group: str
    fixed_hyperparameters: dict = field(default_factory=dict)
    # Free parameters per column besides the K loadings.  0 for every
    # canonical family here (nuisances are fixed); 1 for the Gaussian
    # (profiled column variance).
    extra_free_parameters: int = 0
    # Gaussian only: the existing dispersion form with a profiled variance.
    profiled_variance: bool = False
    T_definition: str = "x"
    eta_domain: str = "R"
    # True when A, A', A'' exist for every eta but can only be CERTIFIED on a
    # bounded range here (COM-Poisson: the summation window grows with eta).
    evaluation_range_limited: bool = False

    # ---- support ---------------------------------------------------------
    def support_mask(self, x: np.ndarray) -> np.ndarray:            # pragma: no cover
        raise NotImplementedError

    def in_support(self, x: np.ndarray) -> bool:
        x = _finite(x, f"{self.candidate_model} observations")
        return bool(np.all(self.support_mask(x)))

    def _require_support(self, x: np.ndarray) -> np.ndarray:
        x = _finite(x, f"{self.candidate_model} observations")
        if not np.all(self.support_mask(x)):
            raise SupportViolation(
                f"{self.candidate_model}: observation outside the support")
        return x

    # ---- exponential-family pieces --------------------------------------
    def sufficient_statistic(self, x: np.ndarray) -> np.ndarray:
        return self._require_support(x).copy()

    def log_base_measure(self, x: np.ndarray) -> np.ndarray:        # pragma: no cover
        raise NotImplementedError

    def log_partition(self, eta: np.ndarray) -> np.ndarray:         # pragma: no cover
        raise NotImplementedError

    def mean_statistic(self, eta: np.ndarray) -> np.ndarray:        # pragma: no cover
        raise NotImplementedError

    def variance_statistic(self, eta: np.ndarray) -> np.ndarray:    # pragma: no cover
        raise NotImplementedError

    def log_prob(self, x: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Complete normalised log probability / density (base measure included)."""

        x = self._require_support(x)
        eta = _finite(eta, f"{self.candidate_model} eta")
        value = (self.log_base_measure(x) + eta * self.sufficient_statistic(x)
                 - self.log_partition(eta))
        return _checked(value, f"{self.candidate_model} log_prob")

    def sample(self, eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def info_at_zero(self) -> float:
        """Fisher information of eta at eta = 0, i.e. A''(0) (dispersion 1)."""
        return float(self.variance_statistic(np.zeros(1))[0])

    @property
    def parameter_count_note(self) -> str:
        return (f"K loadings + {self.extra_free_parameters} extra free "
                f"parameter(s); fixed: {self.fixed_hyperparameters or 'none'}")

    def registry_row(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "candidate_model": self.candidate_model,
            "support_group": self.support_group,
            "hyperparameters": _hp_string(self.fixed_hyperparameters),
            "T_definition": self.T_definition,
            "eta_domain": self.eta_domain,
            "extra_free_parameters": self.extra_free_parameters,
            "implementation_version": FAMILY_SPECS_VERSION,
        }


def _hp_string(hp: dict) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted(hp.items())) if hp else ""


# ==========================================================================
# existing families
# ==========================================================================

class GaussianSpec(FamilySpec):
    """Existing Gaussian-X control: dispersion form, profiled column variance.

    ``log_prob`` needs the variance; ``mean_statistic`` is eta and
    ``variance_statistic`` is 1, matching the existing per-column model where
    the 1/sigma^2 weight is applied separately.
    """

    def __init__(self, sigma_x_var_generator: float = 1.0) -> None:
        super().__init__(family_name="gaussian", candidate_model="gaussian",
                         support_group=GROUP_REAL,
                         fixed_hyperparameters={},
                         extra_free_parameters=1, profiled_variance=True,
                         T_definition="x (dispersion form, sigma^2 estimated)")
        object.__setattr__(self, "sigma_x_var_generator",
                           float(sigma_x_var_generator))

    def support_mask(self, x):
        return np.isfinite(x)

    def log_base_measure(self, x):
        raise NotImplementedError("Gaussian uses the dispersion form; call "
                                  "log_prob(x, eta, sigma_sq=...)")

    def log_partition(self, eta):
        eta = _finite(eta, "gaussian eta")
        return 0.5 * eta * eta

    def mean_statistic(self, eta):
        return _finite(eta, "gaussian eta").copy()

    def variance_statistic(self, eta):
        return np.ones_like(_finite(eta, "gaussian eta"))

    def log_prob(self, x, eta, sigma_sq: float | None = None):
        if sigma_sq is None:
            raise ValueError("Gaussian log_prob needs sigma_sq")
        x = _finite(x, "gaussian observations")
        eta = _finite(eta, "gaussian eta")
        resid = x - eta
        value = (-0.5 * resid ** 2 / sigma_sq - 0.5 * math.log(sigma_sq)
                 - LOG_SQRT_2PI)
        return _checked(value, "gaussian log_prob")

    def info_at_zero(self) -> float:
        return 1.0 / self.sigma_x_var_generator

    def sample(self, eta, rng):
        eta = _finite(eta, "gaussian eta")
        noise = rng.standard_normal(eta.shape) * math.sqrt(self.sigma_x_var_generator)
        return eta + noise


class BernoulliSpec(FamilySpec):
    def __init__(self) -> None:
        super().__init__(family_name="bernoulli", candidate_model="bernoulli",
                         support_group=GROUP_COUNT)

    def support_mask(self, x):
        return (x == 0.0) | (x == 1.0)

    def log_base_measure(self, x):
        return np.zeros_like(x)

    def log_partition(self, eta):
        return _checked(np.logaddexp(0.0, _finite(eta, "bernoulli eta")),
                        "bernoulli A")

    def mean_statistic(self, eta):
        return bernoulli_mean(eta)

    def variance_statistic(self, eta):
        return bernoulli_curvature(eta)

    def log_prob(self, x, eta):
        x = self._require_support(x)
        return bernoulli_log_likelihood(x, eta)

    def sample(self, eta, rng):
        return rng.binomial(1, bernoulli_mean(eta)).astype(np.float64)


class PoissonSpec(FamilySpec):
    def __init__(self) -> None:
        super().__init__(family_name="poisson", candidate_model="poisson",
                         support_group=GROUP_COUNT)

    def support_mask(self, x):
        return (x >= 0.0) & (x == np.floor(x))

    def log_base_measure(self, x):
        return -gammaln(x + 1.0)

    def log_partition(self, eta):
        return poisson_mean(eta)

    def mean_statistic(self, eta):
        return poisson_mean(eta)

    def variance_statistic(self, eta):
        return poisson_curvature(eta)

    def sample(self, eta, rng):
        return rng.poisson(poisson_mean(eta)).astype(np.float64)


# ==========================================================================
# COUNT candidates
# ==========================================================================

class BinomialSpec(FamilySpec):
    def __init__(self, n_trials: int) -> None:
        n_trials = int(n_trials)
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        super().__init__(family_name="binomial",
                         candidate_model=f"binomial_n{n_trials}",
                         support_group=GROUP_COUNT,
                         fixed_hyperparameters={"n_trials": n_trials})
        object.__setattr__(self, "n_trials", n_trials)

    def support_mask(self, x):
        return (x >= 0.0) & (x <= self.n_trials) & (x == np.floor(x))

    def log_base_measure(self, x):
        N = float(self.n_trials)
        return gammaln(N + 1.0) - gammaln(x + 1.0) - gammaln(N - x + 1.0)

    def log_partition(self, eta):
        return _checked(self.n_trials * np.logaddexp(0.0, _finite(eta, "binomial eta")),
                        "binomial A")

    def mean_statistic(self, eta):
        return self.n_trials * bernoulli_mean(eta)

    def variance_statistic(self, eta):
        return self.n_trials * bernoulli_curvature(eta)

    def sample(self, eta, rng):
        return rng.binomial(self.n_trials, bernoulli_mean(eta)).astype(np.float64)


CMP_BLOCK_ELEMENTS = 20_000_000
CMP_MAX_WINDOW = 4_000_000


class NotEvaluable(FloatingPointError):
    """The quantity exists mathematically but cannot be certified here.

    Raised by the COM-Poisson series when the summation window needed to meet
    the tail bound exceeds ``CMP_MAX_WINDOW``.  It is a FloatingPointError, so
    every fail-fast path treats it as a failure; only the candidate trust
    region may treat a NotEvaluable TRIAL point as a rejected step.
    """


_CMP_MEMO: dict = {}
_CMP_MEMO_ORDER: list = []


def _cmp_window(eta: float, nu: float, width_sd: float) -> tuple[int, int]:
    """Summation window around the mode of the log-concave term sequence."""
    log_mode = eta / nu
    if log_mode < math.log(40.0):
        lo = 0
        mode = math.exp(log_mode) if log_mode < 700 else math.inf
        hi = int(max(2.0 * mode + 30.0, 30.0) * (width_sd / 12.0))
        return lo, max(hi, 30)
    if log_mode > 40.0:
        # mode > e^40: the window would exceed any cap by many orders of
        # magnitude (and math.exp would overflow further out)
        raise NotEvaluable(f"COM-Poisson series (nu={nu}) at eta={eta}: mode "
                           f"exp({log_mode:.1f}) is beyond the certifiable range")
    mode = math.exp(log_mode)
    sd = math.sqrt(mode / nu) + 1.0
    lo = int(max(0.0, mode - width_sd * sd))
    hi = int(mode + width_sd * sd) + 1
    return lo, hi


def _cmp_series(eta: np.ndarray, nu: float) -> dict[str, Any]:
    """Certified log-sum-exp evaluation of the COM-Poisson series.

    The term sequence t_x = eta x - nu log x! is log-concave in x, so after a
    window [lo, hi] around the mode both tails are bounded by geometric
    series: right of hi with ratio exp(eta - nu log(hi+1)) < 1 and left of lo
    with ratio lo^nu exp(-eta) < 1.  The window is widened until both bounds
    are below exp(CMP_LOG_TAIL_TOL) relative to Z.  Cost is O(sqrt(mode))
    per eta instead of O(mode).  Distinct eta values are evaluated once.
    """

    eta = _finite(eta, "cmp eta")
    flat = eta.ravel()
    if flat.size == 0:
        empty = np.empty_like(eta)
        return {"logZ": empty, "mean": empty, "var": empty, "M": 0,
                "max_log_tail": -np.inf}
    memo_key = (nu, flat.tobytes())
    hit = _CMP_MEMO.get(memo_key)
    if hit is not None:
        return {"logZ": hit["logZ"].reshape(eta.shape).copy(),
                "mean": hit["mean"].reshape(eta.shape).copy(),
                "var": hit["var"].reshape(eta.shape).copy(),
                "M": hit["M"], "max_log_tail": hit["max_log_tail"]}
    uniq, inverse = np.unique(flat, return_inverse=True)
    logZ = np.empty_like(uniq)
    mean = np.empty_like(uniq)
    var = np.empty_like(uniq)
    W_max, tail_max = 0, -np.inf
    # Small modes (window starts at 0) are evaluated together, vectorised;
    # large modes one at a time with their own window.
    low = uniq / nu < math.log(40.0)
    if np.any(low):
        li = np.nonzero(low)[0]
        blk = _cmp_series_low(uniq[li], nu)
        logZ[li], mean[li], var[li] = blk["logZ"], blk["mean"], blk["var"]
        W_max, tail_max = blk["M"], blk["max_log_tail"]
    for j in np.nonzero(~low)[0]:
        block = _cmp_series_one(float(uniq[j]), nu)
        logZ[j], mean[j], var[j] = block["logZ"], block["mean"], block["var"]
        W_max = max(W_max, block["M"])
        tail_max = max(tail_max, block["max_log_tail"])
    out = {"logZ": _checked(logZ[inverse], "cmp logZ"),
           "mean": _checked(mean[inverse], "cmp mean"),
           "var": _checked(var[inverse], "cmp var"),
           "M": int(W_max), "max_log_tail": float(tail_max)}
    _CMP_MEMO[memo_key] = out
    _CMP_MEMO_ORDER.append(memo_key)
    if len(_CMP_MEMO_ORDER) > 8:
        _CMP_MEMO.pop(_CMP_MEMO_ORDER.pop(0), None)
    return {"logZ": out["logZ"].reshape(eta.shape).copy(),
            "mean": out["mean"].reshape(eta.shape).copy(),
            "var": out["var"].reshape(eta.shape).copy(),
            "M": out["M"], "max_log_tail": out["max_log_tail"]}


_LOGFACT_CACHE: dict = {}


def _cmp_series_low(etas: np.ndarray, nu: float) -> dict[str, Any]:
    """Vectorised series for etas whose mode is < 40 (window [0, M])."""
    M = _cmp_window(float(np.max(etas)), nu, 12.0)[1]
    while True:
        key = (nu, M)
        lf = _LOGFACT_CACHE.get(key)
        if lf is None:
            lf = nu * gammaln(np.arange(M + 1, dtype=np.float64) + 1.0)
            if len(_LOGFACT_CACHE) > 64:
                _LOGFACT_CACHE.clear()
            _LOGFACT_CACHE[key] = lf
        x = np.arange(M + 1, dtype=np.float64)
        log_terms = etas[:, None] * x[None, :] - lf[None, :]
        peak = np.max(log_terms, axis=1, keepdims=True)
        weights = np.exp(log_terms - peak)
        total = np.sum(weights, axis=1)
        logZ = peak[:, 0] + np.log(total)
        log_r = etas - nu * math.log(M + 1.0)
        if np.all(log_r < 0.0):
            rel = (log_terms[:, -1] + log_r - np.log(-np.expm1(log_r))) - logZ
            if np.all(rel < CMP_LOG_TAIL_TOL):
                break
        M *= 2
        if M + 1 > CMP_MAX_WINDOW:
            raise NotEvaluable(f"COM-Poisson low-mode block exceeded the cap (nu={nu})")
    probs = weights / total[:, None]
    mean = probs @ x
    var = np.sum(probs * (x[None, :] - mean[:, None]) ** 2, axis=1)
    return {"logZ": logZ, "mean": mean, "var": var, "M": int(M + 1),
            "max_log_tail": float(np.max(rel))}


def _cmp_series_one(eta: float, nu: float, keep_probs: bool = False) -> dict[str, Any]:
    """One eta: windowed sum with certified two-sided geometric tail bounds."""
    width_sd = 12.0
    while True:
        lo, hi = _cmp_window(eta, nu, width_sd)
        if hi - lo + 1 > CMP_MAX_WINDOW:
            raise NotEvaluable(
                f"COM-Poisson series (nu={nu}) needs a window of {hi - lo + 1} "
                f"terms at eta={eta} (cap {CMP_MAX_WINDOW})")
        x = np.arange(lo, hi + 1, dtype=np.float64)
        log_terms = eta * x - nu * gammaln(x + 1.0)
        peak = float(np.max(log_terms))
        weights = np.exp(log_terms - peak)
        total = float(np.sum(weights))
        logZ = peak + math.log(total)
        # geometric bounds need only ratio < 1; the factor r/(1-r) is included
        log_r_hi = eta - nu * math.log(hi + 1.0)
        ok = log_r_hi < 0.0
        tails = []
        if ok:
            tails.append(log_terms[-1] + log_r_hi - math.log(-math.expm1(log_r_hi)))
        if lo > 0:
            log_r_lo = nu * math.log(lo) - eta
            if log_r_lo < 0.0:
                tails.append(log_terms[0] + log_r_lo - math.log(-math.expm1(log_r_lo)))
            else:
                ok = False
        rel = (max(tails) - logZ) if tails else -np.inf
        if ok and rel < CMP_LOG_TAIL_TOL:
            break
        width_sd *= 2.0
    probs = weights / total
    m = float(probs @ x)
    v = float(probs @ (x - m) ** 2)
    out = {"logZ": logZ, "mean": m, "var": v, "M": int(hi - lo + 1),
           "max_log_tail": float(rel), "lo": lo}
    if keep_probs:
        out["probs"], out["x"] = probs, x
    return out


def _cmp_series_block(flat: np.ndarray, nu: float) -> dict[str, Any]:
    """Single-eta helper kept for the calibration (returns the pmf window)."""
    if np.asarray(flat).size != 1:
        raise ValueError("_cmp_series_block takes exactly one eta")
    one = _cmp_series_one(float(np.asarray(flat).ravel()[0]), nu, keep_probs=True)
    return {"x": one["x"], "probs": one["probs"][None, :], "M": one["M"],
            "logZ": np.array([one["logZ"]]), "lo": one["lo"]}


class COMPoissonSpec(FamilySpec):
    def __init__(self, nu: float) -> None:
        nu = float(nu)
        if not nu > 0.0:
            raise ValueError("COM-Poisson nu must be > 0 (nu = 0 is the "
                             "geometric, whose natural domain is eta < 0)")
        tag = f"{nu:.1f}".replace(".", "p")
        super().__init__(family_name="com_poisson",
                         candidate_model=f"cmp_nu{tag}",
                         support_group=GROUP_COUNT,
                         fixed_hyperparameters={"nu": nu},
                         evaluation_range_limited=True)
        object.__setattr__(self, "nu", nu)

    def support_mask(self, x):
        return (x >= 0.0) & (x == np.floor(x))

    def log_base_measure(self, x):
        return -self.nu * gammaln(x + 1.0)

    def log_partition(self, eta):
        return _cmp_series(eta, self.nu)["logZ"]

    def mean_statistic(self, eta):
        return _cmp_series(eta, self.nu)["mean"]

    def variance_statistic(self, eta):
        return _cmp_series(eta, self.nu)["var"]

    def series_diagnostics(self, eta) -> dict[str, Any]:
        out = _cmp_series(eta, self.nu)
        return {"M": out["M"], "max_log_tail": out["max_log_tail"]}

    def sample(self, eta, rng):
        """Inverse-CDF draw; exactly one uniform per observation."""
        eta = _finite(eta, "cmp eta")
        flat = eta.ravel()
        u = rng.random(flat.size)
        out = np.empty_like(flat)
        for i in range(flat.size):
            block = _cmp_series_one(float(flat[i]), self.nu, keep_probs=True)
            cdf = np.cumsum(block["probs"])
            j = int(np.searchsorted(cdf, u[i], side="right"))
            if j >= block["probs"].size:
                raise FloatingPointError("COM-Poisson inverse-CDF ran past the "
                                         "summation window")
            out[i] = float(block["x"][j])
        return out.reshape(eta.shape)


def _ztp_g(lam: np.ndarray) -> np.ndarray:
    """(1 - e^{-lam}) / lam, stable at small lam."""
    out = np.empty_like(lam)
    small = lam < ZTP_SERIES_LAMBDA
    ls = lam[small]
    out[small] = 1.0 - ls / 2.0 + ls ** 2 / 6.0 - ls ** 3 / 24.0 + ls ** 4 / 120.0
    lb = lam[~small]
    out[~small] = -np.expm1(-lb) / lb
    return out


def _ztp_one_plus_lam_minus_mean(lam: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """1 + lam - mu for the ZTP mean mu = lam / (1 - e^{-lam})."""
    out = np.empty_like(lam)
    small = lam < ZTP_SERIES_LAMBDA
    ls = lam[small]
    # mu = 1 + lam/2 + lam^2/12 - lam^4/720 + lam^6/30240 - ...
    out[small] = ls / 2.0 - ls ** 2 / 12.0 + ls ** 4 / 720.0 - ls ** 6 / 30240.0
    out[~small] = 1.0 + lam[~small] - mean[~small]
    return out


class ZeroTruncatedPoissonSpec(FamilySpec):
    def __init__(self) -> None:
        super().__init__(family_name="zero_truncated_poisson",
                         candidate_model="ztp", support_group=GROUP_COUNT)

    def support_mask(self, x):
        return (x >= 1.0) & (x == np.floor(x))

    def log_base_measure(self, x):
        return -gammaln(x + 1.0)

    def log_partition(self, eta):
        eta = _finite(eta, "ztp eta")
        lam = poisson_mean(eta)
        # A = log(e^lam - 1) = lam + eta + log(g(lam)), g = (1 - e^-lam)/lam
        return _checked(lam + eta + np.log(_ztp_g(lam)), "ztp A")

    def mean_statistic(self, eta):
        lam = poisson_mean(_finite(eta, "ztp eta"))
        return _checked(1.0 / _ztp_g(lam), "ztp mean")

    def variance_statistic(self, eta):
        lam = poisson_mean(_finite(eta, "ztp eta"))
        mean = 1.0 / _ztp_g(lam)
        return _checked(mean * _ztp_one_plus_lam_minus_mean(lam, mean), "ztp var")

    def sample(self, eta, rng):
        from scipy.stats import poisson
        lam = poisson_mean(_finite(eta, "ztp eta"))
        u = rng.random(lam.shape)
        q = (1.0 - u) * (-np.expm1(-lam))
        x = poisson.isf(q, lam)
        if not np.all(np.isfinite(x)) or np.any(x < 1):
            raise FloatingPointError("ZTP inverse-CDF sampling left the support")
        return np.asarray(x, dtype=np.float64)


# ==========================================================================
# POSITIVE candidates
# ==========================================================================

class LogNormalSpec(FamilySpec):
    """log x ~ N(sigma^2 eta, sigma^2); raw-x density with the 1/x Jacobian."""

    def __init__(self, sigma: float) -> None:
        sigma = float(sigma)
        tag = f"{sigma:g}".replace(".", "p")
        super().__init__(family_name="lognormal",
                         candidate_model=f"lognormal_s{tag}",
                         support_group=GROUP_POSITIVE,
                         fixed_hyperparameters={"sigma": sigma},
                         T_definition="log x")
        object.__setattr__(self, "sigma", sigma)

    def support_mask(self, x):
        return x > 0.0

    def sufficient_statistic(self, x):
        return np.log(self._require_support(x))

    def log_base_measure(self, x):
        lx = np.log(x)
        s2 = self.sigma ** 2
        return -lx - math.log(self.sigma) - LOG_SQRT_2PI - lx * lx / (2.0 * s2)

    def log_partition(self, eta):
        eta = _finite(eta, "lognormal eta")
        return 0.5 * self.sigma ** 2 * eta * eta

    def mean_statistic(self, eta):
        return self.sigma ** 2 * _finite(eta, "lognormal eta")

    def variance_statistic(self, eta):
        return np.full_like(_finite(eta, "lognormal eta"), self.sigma ** 2)

    def sample(self, eta, rng):
        eta = _finite(eta, "lognormal eta")
        e = rng.standard_normal(eta.shape)
        x = np.exp(self.sigma ** 2 * eta + self.sigma * e)
        if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
            raise FloatingPointError("lognormal sample left (0, inf)")
        return x


class PositiveTruncatedNormalSpec(FamilySpec):
    """N(sigma^2 eta, sigma^2) truncated to (0, inf)."""

    def __init__(self, sigma: float) -> None:
        sigma = float(sigma)
        tag = f"{sigma:g}".replace(".", "p")
        super().__init__(family_name="positive_truncated_normal",
                         candidate_model=f"ptn_s{tag}",
                         support_group=GROUP_POSITIVE,
                         fixed_hyperparameters={"sigma": sigma})
        object.__setattr__(self, "sigma", sigma)

    def support_mask(self, x):
        return x > 0.0

    def log_base_measure(self, x):
        return -x * x / (2.0 * self.sigma ** 2) - math.log(self.sigma) - LOG_SQRT_2PI

    def log_partition(self, eta):
        eta = _finite(eta, "ptn eta")
        u = self.sigma * eta
        return _checked(0.5 * u * u + log_ndtr(u), "ptn A")

    def _mills(self, u):
        return np.exp(_log_phi(u) - log_ndtr(u))

    def mean_statistic(self, eta):
        eta = _finite(eta, "ptn eta")
        u = self.sigma * eta
        return _checked(self.sigma * (u + self._mills(u)), "ptn mean")

    def variance_statistic(self, eta):
        eta = _finite(eta, "ptn eta")
        u = self.sigma * eta
        lam = self._mills(u)
        value = self.sigma ** 2 * (1.0 - lam * (u + lam))
        _checked(value, "ptn var")
        if np.any(value <= 0.0):
            raise FloatingPointError("ptn variance lost positivity "
                                     "(cancellation in 1 - lam(u+lam))")
        return value

    def sample(self, eta, rng):
        eta = _finite(eta, "ptn eta")
        mu = self.sigma ** 2 * eta
        u = rng.random(eta.shape)
        x = truncnorm.ppf(u, (0.0 - mu) / self.sigma, np.inf,
                          loc=mu, scale=self.sigma)
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
            raise FloatingPointError("ptn sample left (0, inf)")
        return x


# ==========================================================================
# BOUNDED candidates
# ==========================================================================

class ContinuousBernoulliSpec(FamilySpec):
    """p(x) = exp(eta x) / int_0^1 exp(eta t) dt on (0, 1)."""

    def __init__(self) -> None:
        super().__init__(family_name="continuous_bernoulli",
                         candidate_model="cb", support_group=GROUP_BOUNDED)

    def support_mask(self, x):
        return (x > 0.0) & (x < 1.0)

    def log_base_measure(self, x):
        return np.zeros_like(x)

    def log_partition(self, eta):
        eta = _finite(eta, "cb eta")
        out = np.empty_like(eta)
        small = np.abs(eta) < CB_SERIES_ABS_ETA
        t = eta[small] / 2.0
        out[small] = eta[small] / 2.0 + t ** 2 / 6.0 - t ** 4 / 180.0 + t ** 6 / 2835.0
        e = eta[~small]
        at = np.abs(e) / 2.0
        log_sinh_over_t = at + np.log(-np.expm1(-2.0 * at)) - math.log(2.0) - np.log(at)
        out[~small] = e / 2.0 + log_sinh_over_t
        return _checked(out, "cb A")

    def mean_statistic(self, eta):
        eta = _finite(eta, "cb eta")
        out = np.empty_like(eta)
        small = np.abs(eta) < CB_SERIES_ABS_ETA
        es = eta[small]
        out[small] = 0.5 + es / 12.0 - es ** 3 / 720.0 + es ** 5 / 30240.0
        e = eta[~small]
        out[~small] = 0.5 + 0.5 / np.tanh(e / 2.0) - 1.0 / e
        return _checked(out, "cb mean")

    def variance_statistic(self, eta):
        eta = _finite(eta, "cb eta")
        out = np.empty_like(eta)
        small = np.abs(eta) < 0.1
        es = eta[small]
        out[small] = (1.0 / 12.0 - es ** 2 / 240.0 + es ** 4 / 6048.0
                      - es ** 6 / 172800.0)
        e = eta[~small]
        out[~small] = 1.0 / e ** 2 - 0.25 / np.sinh(e / 2.0) ** 2
        return _checked(out, "cb var")

    def sample(self, eta, rng):
        eta = _finite(eta, "cb eta")
        u = rng.random(eta.shape)
        x = np.empty_like(eta)
        tiny = np.abs(eta) < 1e-12
        x[tiny] = u[tiny]
        e = eta[~tiny]
        x[~tiny] = np.log1p(u[~tiny] * np.expm1(e)) / e
        if not np.all(np.isfinite(x)) or np.any((x <= 0.0) | (x >= 1.0)):
            raise FloatingPointError("cb sample left (0, 1)")
        return x


class LogitNormalSpec(FamilySpec):
    """logit x ~ N(sigma^2 eta, sigma^2); raw-x density with the Jacobian."""

    def __init__(self, sigma: float) -> None:
        sigma = float(sigma)
        tag = f"{sigma:g}".replace(".", "p")
        super().__init__(family_name="logit_normal",
                         candidate_model=f"logitnormal_s{tag}",
                         support_group=GROUP_BOUNDED,
                         fixed_hyperparameters={"sigma": sigma},
                         T_definition="logit x")
        object.__setattr__(self, "sigma", sigma)

    def support_mask(self, x):
        return (x > 0.0) & (x < 1.0)

    def sufficient_statistic(self, x):
        x = self._require_support(x)
        return np.log(x) - np.log1p(-x)

    def log_base_measure(self, x):
        lg = np.log(x) - np.log1p(-x)
        return (-np.log(x) - np.log1p(-x) - math.log(self.sigma) - LOG_SQRT_2PI
                - lg * lg / (2.0 * self.sigma ** 2))

    def log_partition(self, eta):
        eta = _finite(eta, "logitnormal eta")
        return 0.5 * self.sigma ** 2 * eta * eta

    def mean_statistic(self, eta):
        return self.sigma ** 2 * _finite(eta, "logitnormal eta")

    def variance_statistic(self, eta):
        return np.full_like(_finite(eta, "logitnormal eta"), self.sigma ** 2)

    def sample(self, eta, rng):
        eta = _finite(eta, "logitnormal eta")
        e = rng.standard_normal(eta.shape)
        x = expit(self.sigma ** 2 * eta + self.sigma * e)
        if not np.all(np.isfinite(x)) or np.any((x <= 0.0) | (x >= 1.0)):
            raise FloatingPointError("logit-normal sample rounded onto an "
                                     "endpoint of (0, 1)")
        return x


class TruncatedNormal01Spec(FamilySpec):
    """N(c + sigma^2 eta, sigma^2) truncated to (0, 1); base measure centred at c."""

    def __init__(self, sigma: float, center: float) -> None:
        sigma, center = float(sigma), float(center)
        st = f"{sigma:g}".replace(".", "p")
        ct = f"{center:g}".replace(".", "p")
        super().__init__(family_name="truncated_normal_01",
                         candidate_model=f"tn01_s{st}_c{ct}",
                         support_group=GROUP_BOUNDED,
                         fixed_hyperparameters={"sigma": sigma, "center": center})
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "center", center)

    def support_mask(self, x):
        return (x > 0.0) & (x < 1.0)

    def log_base_measure(self, x):
        d = x - self.center
        return -d * d / (2.0 * self.sigma ** 2) - math.log(self.sigma) - LOG_SQRT_2PI

    def _ab(self, eta):
        mu = self.center + self.sigma ** 2 * eta
        return mu, (0.0 - mu) / self.sigma, (1.0 - mu) / self.sigma

    def log_partition(self, eta):
        eta = _finite(eta, "tn01 eta")
        _, a, b = self._ab(eta)
        value = (self.center * eta + 0.5 * self.sigma ** 2 * eta * eta
                 + log_ndtr_diff(a, b))
        return _checked(value, "tn01 A")

    def _ratios(self, eta):
        mu, a, b = self._ab(eta)
        logZ = log_ndtr_diff(a, b)
        ra = np.exp(_log_phi(a) - logZ)
        rb = np.exp(_log_phi(b) - logZ)
        return mu, a, b, ra, rb

    def mean_statistic(self, eta):
        eta = _finite(eta, "tn01 eta")
        mu, _, _, ra, rb = self._ratios(eta)
        return _checked(mu + self.sigma * (ra - rb), "tn01 mean")

    def variance_statistic(self, eta):
        eta = _finite(eta, "tn01 eta")
        _, a, b, ra, rb = self._ratios(eta)
        value = self.sigma ** 2 * (1.0 + a * ra - b * rb - (ra - rb) ** 2)
        _checked(value, "tn01 var")
        if np.any(value <= 0.0):
            raise FloatingPointError("tn01 variance lost positivity")
        return value

    def sample(self, eta, rng):
        eta = _finite(eta, "tn01 eta")
        mu, a, b = self._ab(eta)
        u = rng.random(eta.shape)
        x = np.asarray(truncnorm.ppf(u, a, b, loc=mu, scale=self.sigma),
                       dtype=np.float64)
        if not np.all(np.isfinite(x)) or np.any((x <= 0.0) | (x >= 1.0)):
            raise FloatingPointError("tn01 sample left (0, 1)")
        return x


# ==========================================================================
# registry
# ==========================================================================

def build_primary_registry() -> dict[str, FamilySpec]:
    """Every candidate model of the primary experiment, by candidate_model id.

    Fixed hyperparameters are frozen here and recorded in protocol.json; see
    the compatibility audit for the a-priori rationale of each value.
    """

    specs = [
        PoissonSpec(),
        BinomialSpec(2),
        COMPoissonSpec(2.0),
        COMPoissonSpec(0.5),
        ZeroTruncatedPoissonSpec(),
        LogNormalSpec(1.0),
        PositiveTruncatedNormalSpec(1.0),
        LogitNormalSpec(1.0),
        ContinuousBernoulliSpec(),
        TruncatedNormal01Spec(0.25, 0.5),
        GaussianSpec(1.0),
    ]
    return {s.candidate_model: s for s in specs}


# Frozen candidate priority per support group (first = SUPPORT_ONLY choice and
# START_FIRST; last = START_LAST; also the exact-tie rule).  Rationale: the
# conventional default for the data type first, the truncated family last.
PRIMARY_PRIORITY: dict[str, tuple[str, ...]] = {
    GROUP_COUNT: ("poisson", "binomial_n2", "cmp_nu2p0", "cmp_nu0p5", "ztp"),
    GROUP_POSITIVE: ("lognormal_s1", "ptn_s1"),
    GROUP_BOUNDED: ("logitnormal_s1", "cb", "tn01_s0p25_c0p5"),
    GROUP_REAL: ("gaussian",),
}

PRIORITY_RATIONALE = (
    "Frozen before any selector run. Within each support group the "
    "conventional default for the data type comes first (Poisson for counts, "
    "log-normal for positive measurements, logit-normal for proportions), "
    "then the remaining candidates, with the truncated family last. "
    "SUPPORT_ONLY uses the first support-admissible candidate; START_FIRST / "
    "START_LAST use the first / last admissible candidate; an exact score tie "
    "is broken by this order.")


def compatibility_rows(test_status: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Machine-readable form of the compatibility audit (one row per family)."""

    test_status = test_status or {}
    passed = [
        ("gaussian", GROUP_REAL, "PASS_STRICT",
         "existing control; dispersion form with estimated sigma^2; no competing "
         "scalar-natural-parameter family on R", "x", "R", 1, "", True),
        ("bernoulli", GROUP_COUNT, "PASS_STRICT",
         "existing; equals Binomial(1); kept for regression, not in the primary "
         "candidate set", "x", "R", 0, "", True),
        ("poisson", GROUP_COUNT, "PASS_STRICT", "existing", "x", "R", 0, "", True),
        ("binomial", GROUP_COUNT, "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
         "regular 1-parameter EF for fixed N; needs log h = log C(N,x) and support "
         "{0..N}", "x", "R", 0, "n_trials=2", True),
        ("com_poisson", GROUP_COUNT, "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
         "nu>0 fixed: series converges for all eta; A by adaptive truncated "
         "series with tail bound; nu=1 is Poisson", "x", "R", 0,
         "nu in {0.5, 2.0} (separate candidate models)", True),
        ("zero_truncated_poisson", GROUP_COUNT,
         "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
         "A = log(exp(e^eta) - 1) finite for all eta; support {1,2,...}",
         "x", "R", 0, "", True),
        ("lognormal", GROUP_POSITIVE, "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
         "sigma fixed: scalar natural parameter; T = log x; raw-x density with "
         "the 1/x Jacobian", "log x", "R", 0, "sigma=1", True),
        ("positive_truncated_normal", GROUP_POSITIVE,
         "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
         "sigma fixed; A includes log Phi(sigma eta)", "x", "R", 0, "sigma=1", True),
        ("continuous_bernoulli", GROUP_BOUNDED,
         "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
         "exp(eta x) on (0,1) normalisable for all eta; series near eta=0",
         "x", "R", 0, "", True),
        ("logit_normal", GROUP_BOUNDED, "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
         "sigma fixed; T = logit x; Jacobian 1/[x(1-x)]", "logit x", "R", 0,
         "sigma=1", True),
        ("truncated_normal_01", GROUP_BOUNDED,
         "PASS_WITH_EXISTING_EXPFAM_GENERALIZATION",
         "sigma and centre fixed; A includes log[Phi(b)-Phi(a)] (stable form)",
         "x", "R", 0, "sigma=0.25;center=0.5", True),
    ]
    blocked = [
        ("geometric", GROUP_COUNT, "natural parameter log p in (-inf,0); "
         "eta = f^T z is not always legal without a new link"),
        ("negative_binomial", GROUP_COUNT, "fixed r: natural parameter "
         "log(mu/(mu+r)) in (-inf,0); existing NB is Y-side NB2 with a "
         "non-canonical log link and may not be copied to X"),
        ("gamma", GROUP_POSITIVE, "fixed shape: natural parameter -rate < 0"),
        ("exponential", GROUP_POSITIVE, "natural parameter -lambda < 0 "
         "(the (0,1)-truncated exponential is the Continuous Bernoulli, PASS)"),
        ("beta", GROUP_BOUNDED, "two natural parameters; fixing one leaves "
         "domain (-1, inf)"),
        ("weibull", GROUP_POSITIVE, "fixed shape: natural parameter "
         "-lambda^-k < 0; free shape is not an EF"),
        ("student_t", GROUP_REAL, "not an exponential family in location"),
        ("categorical", "NONE", "vector natural parameter / non-scalar "
         "observation (KI-005)"),
        ("inverse_gaussian", GROUP_POSITIVE, "both natural parameters "
         "restricted to negative half-lines"),
    ]
    rows = []
    for (name, group, status, reason, T, dom, k_extra, hp, impl) in passed:
        rows.append({"family_name": name, "support_group": group,
                     "compatibility_status": status, "reason": reason,
                     "T_definition": T, "eta_domain": dom,
                     "free_parameter_count": f"K+{k_extra}",
                     "fixed_hyperparameters": hp, "implemented": impl,
                     "test_status": test_status.get(name, "NOT_RECORDED")})
    for (name, group, reason) in blocked:
        rows.append({"family_name": name, "support_group": group,
                     "compatibility_status": "BLOCKED_OUTSIDE_CURRENT_MODEL",
                     "reason": reason, "T_definition": "",
                     "eta_domain": "restricted" if name != "student_t" else "n/a",
                     "free_parameter_count": "", "fixed_hyperparameters": "",
                     "implemented": False, "test_status": "NOT_APPLICABLE"})
    return rows


__all__ = [
    "FAMILY_SPECS_VERSION", "SUPPORT_GROUPS", "GROUP_COUNT", "GROUP_POSITIVE",
    "GROUP_BOUNDED", "GROUP_REAL", "BASE_MEASURE", "SupportViolation",
    "NotEvaluable", "FamilySpec", "GaussianSpec", "BernoulliSpec", "PoissonSpec",
    "BinomialSpec", "COMPoissonSpec", "ZeroTruncatedPoissonSpec",
    "LogNormalSpec", "PositiveTruncatedNormalSpec", "ContinuousBernoulliSpec",
    "LogitNormalSpec", "TruncatedNormal01Spec", "build_primary_registry",
    "PRIMARY_PRIORITY", "PRIORITY_RATIONALE", "compatibility_rows",
    "log_ndtr_diff",
]
