"""Forward-only objective-consistent experimental model lineage.

The legacy fixed/masked/per-column classes remain unchanged for historical
reproducibility.  These classes replace only Bernoulli/Poisson numerical
evaluation; Gaussian behavior and the fixed/masked mathematical lineage are
inherited unchanged.
"""

import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from model_dual_expfam_masked import DualExpFamLSMMasked  # noqa: E402
from model_dual_expfam_percolumn import DualExpFamLSMPerColumn  # noqa: E402
from objective_consistent_numerics import (  # noqa: E402
    bernoulli_curvature,
    bernoulli_log_likelihood,
    bernoulli_mean,
    poisson_curvature,
    poisson_log_likelihood,
    poisson_mean,
)


class _ObjectiveConsistentYMixin:
    """Canonical Bernoulli/Poisson Y hooks shared by both new classes."""

    numerics_mode = "consistent"

    def _mean_function(self, eta):
        if self.family == "bernoulli":
            return bernoulli_mean(eta)
        if self.family == "poisson":
            return poisson_mean(eta)
        return np.asarray(eta).copy()

    def _variance_function(self, eta):
        if self.family == "bernoulli":
            return bernoulli_curvature(eta)
        if self.family == "poisson":
            return poisson_curvature(eta)
        return np.full_like(
            eta, 1.0 / max(self.sigma_y ** 2, 1e-8), dtype=float
        )

    def calc_log_likelihood_Y(self, Y, Z_samples, w0, w):
        """Masked Y objective matching the score and curvature hooks above."""
        _, _, L = Z_samples.shape
        ll = 0.0
        for sample in range(L):
            Z_l = Z_samples[:, :, sample]
            eta = w0 + w * (Z_l @ Z_l.T)
            if self.family == "bernoulli":
                ln_p = bernoulli_log_likelihood(Y, eta)
            elif self.family == "poisson":
                ln_p = poisson_log_likelihood(Y, eta)
            else:
                sig2 = max(self.sigma_y ** 2, 1e-8)
                ln_p = (-0.5 * (Y - eta) ** 2 / sig2
                        - 0.5 * np.log(sig2))
            ll += 0.5 * float(np.sum(ln_p * self._mask_f))
        return ll / L


class DualExpFamLSMConsistent(_ObjectiveConsistentYMixin,
                               DualExpFamLSMMasked):
    """Scalar-family masked model with objective-consistent numerics."""

    def _mean_function_x(self, eta_x):
        if self.family_x == "gaussian":
            return np.asarray(eta_x).copy()
        if self.family_x == "bernoulli":
            return bernoulli_mean(eta_x)
        return poisson_mean(eta_x)

    def _variance_function_x(self, eta_x):
        if self.family_x == "bernoulli":
            return bernoulli_curvature(eta_x)
        if self.family_x == "poisson":
            return poisson_curvature(eta_x)
        return np.ones_like(eta_x, dtype=float)

    def calc_log_likelihood_X(self, X, Z_samples, F):
        _, _, L = Z_samples.shape
        ll = 0.0
        for sample in range(L):
            eta_x = Z_samples[:, :, sample] @ F.T
            if self.family_x == "gaussian":
                sigma_diag = np.maximum(np.diag(self.params["sigma"]), 1e-8)
                resid = X - eta_x
                ln_p = (-0.5 * resid ** 2 / sigma_diag
                        - 0.5 * np.log(sigma_diag)
                        - 0.5 * np.log(2.0 * np.pi))
            elif self.family_x == "bernoulli":
                ln_p = bernoulli_log_likelihood(X, eta_x)
            else:
                ln_p = poisson_log_likelihood(X, eta_x)
            ll += float(np.sum(ln_p))
        return ll / L

    def __repr__(self):
        return (f"DualExpFamLSMConsistent(n={self.n}, d={self.d}, k={self.k}, "
                f"family_x='{self.family_x}', family_y='{self.family}', "
                f"n_train_pairs={self.n_train_pairs()})")


class DualExpFamLSMPerColumnConsistent(_ObjectiveConsistentYMixin,
                                        DualExpFamLSMPerColumn):
    """Per-column prototype with objective-consistent future numerics."""

    def _mean_function_x(self, eta_x):
        eta_x = np.asarray(eta_x)
        out = np.empty_like(eta_x, dtype=float)
        g = self._col_idx["gaussian"]
        b = self._col_idx["bernoulli"]
        p = self._col_idx["poisson"]
        if len(g):
            out[..., g] = eta_x[..., g]
        if len(b):
            out[..., b] = bernoulli_mean(eta_x[..., b])
        if len(p):
            out[..., p] = poisson_mean(eta_x[..., p])
        return out

    def _variance_function_x(self, eta_x):
        eta_x = np.asarray(eta_x)
        out = np.empty_like(eta_x, dtype=float)
        g = self._col_idx["gaussian"]
        b = self._col_idx["bernoulli"]
        p = self._col_idx["poisson"]
        if len(g):
            out[..., g] = 1.0
        if len(b):
            out[..., b] = bernoulli_curvature(eta_x[..., b])
        if len(p):
            out[..., p] = poisson_curvature(eta_x[..., p])
        return out

    def calc_log_likelihood_X(self, X, Z_samples, F):
        _, _, L = Z_samples.shape
        sigma = self.params["sigma"]
        g = self._col_idx["gaussian"]
        b = self._col_idx["bernoulli"]
        p = self._col_idx["poisson"]
        ll = 0.0
        for sample in range(L):
            eta_x = Z_samples[:, :, sample] @ F.T
            if len(g):
                variance = np.maximum(np.diag(sigma)[g], 1e-8)
                resid = X[:, g] - eta_x[:, g]
                ll += float(np.sum(-0.5 * resid ** 2 / variance
                                   - 0.5 * np.log(variance)
                                   - 0.5 * np.log(2.0 * np.pi)))
            if len(b):
                ll += float(np.sum(
                    bernoulli_log_likelihood(X[:, b], eta_x[:, b])
                ))
            if len(p):
                ll += float(np.sum(
                    poisson_log_likelihood(X[:, p], eta_x[:, p])
                ))
        return ll / L

    def __repr__(self):
        counts = {f: len(idx) for f, idx in self._col_idx.items() if len(idx)}
        return (f"DualExpFamLSMPerColumnConsistent(n={self.n}, d={self.d}, "
                f"k={self.k}, family_x_cols={counts}, "
                f"family_y='{self.family}')")
