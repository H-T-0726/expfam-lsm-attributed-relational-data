"""Canonical, objective-consistent Bernoulli and Poisson numerics.

This module is forward-only.  Legacy model classes deliberately retain their
historical clipping/flooring behavior.  Every helper here evaluates one
canonical objective together with its exact score/curvature ingredients.
"""

import numpy as np


def _finite_float_array(value, *, name):
    """Return a floating array and reject non-finite inputs explicitly."""
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"{name} must contain only finite values")
    return arr


def _require_finite(value, *, name):
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} produced a non-finite value")
    return value


def bernoulli_mean(eta):
    """Stable canonical sigmoid without evaluating the unsafe branch."""
    eta = _finite_float_array(eta, name="Bernoulli eta")
    out = np.empty_like(eta)
    positive = eta >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    out[~positive] = exp_eta / (1.0 + exp_eta)
    return out


def bernoulli_curvature(eta):
    """Canonical A''(eta) = sigmoid(eta) * (1 - sigmoid(eta))."""
    mean = bernoulli_mean(eta)
    return mean * (1.0 - mean)


def bernoulli_log_likelihood(x, eta):
    """Canonical Bernoulli log likelihood, evaluated with logaddexp."""
    eta = _finite_float_array(eta, name="Bernoulli eta")
    x = _finite_float_array(x, name="Bernoulli observations")
    try:
        with np.errstate(over="raise", invalid="raise"):
            value = x * eta - np.logaddexp(0.0, eta)
    except FloatingPointError as exc:
        raise FloatingPointError(
            "Bernoulli log likelihood overflowed or became invalid"
        ) from exc
    return _require_finite(value, name="Bernoulli log likelihood")


def poisson_mean(eta):
    """Canonical exp(eta), with dtype-derived overflow and finiteness guards."""
    eta = _finite_float_array(eta, name="Poisson eta")
    log_max = np.log(np.finfo(eta.dtype).max)
    if np.any(eta > log_max):
        largest = float(np.max(eta))
        raise FloatingPointError(
            "Poisson exp(eta) would overflow for "
            f"dtype {eta.dtype}: max eta={largest}, log(finfo.max)={log_max}"
        )
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            mean = np.exp(eta)
    except FloatingPointError as exc:
        raise FloatingPointError(
            f"Poisson exp(eta) failed for dtype {eta.dtype}"
        ) from exc
    return _require_finite(mean, name="Poisson mean")


def poisson_curvature(eta):
    """Canonical A''(eta) = exp(eta)."""
    return poisson_mean(eta)


def poisson_log_likelihood(x, eta):
    """Canonical Poisson log likelihood without the constant -log(x!)."""
    eta = _finite_float_array(eta, name="Poisson eta")
    x = _finite_float_array(x, name="Poisson observations")
    mean = poisson_mean(eta)
    try:
        with np.errstate(over="raise", invalid="raise"):
            value = x * eta - mean
    except FloatingPointError as exc:
        raise FloatingPointError(
            "Poisson log likelihood overflowed or became invalid"
        ) from exc
    return _require_finite(value, name="Poisson log likelihood")
