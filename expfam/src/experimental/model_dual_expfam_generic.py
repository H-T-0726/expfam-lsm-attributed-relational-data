"""Generic per-column exponential-family X model (forward-only, lineage E).

``DualExpFamLSMPerColumnConsistent`` hard-codes T(x) = x and the three
families {gaussian, bernoulli, poisson}.  This subclass replaces exactly the
places where that assumption lives -- the X residual T(x) - A'(eta) in the
E-step gradient and the M-step loading gradient, the curvature A''(eta) in the
E-step precision, and the complete X log-likelihood -- with the per-column
``FamilySpec`` quantities.  Nothing else changes:

* the Y side (objective-consistent Bernoulli/Poisson/Gaussian hooks),
* the Newton/Laplace sampler, ``scale_Z``, the M-step order,
* the Gaussian column handling (dispersion form, profiled sigma^2, 1/sigma^2
  weights),
* the 50-step weighted Adam used for the loading M-step.

For specs restricted to {gaussian, bernoulli, poisson} the E-step and M-step
numbers are bit-for-bit those of the parent class (pinned by
``test_generic_regression.py``).  ``calc_log_likelihood_X`` here is the
COMPLETE log-likelihood (base measure included, e.g. Poisson -log x!), so the
strict Q of this module adds no X-side correction afterwards; it agrees with
``eval_utils.calc_Q_dual_strict_exp`` to rounding on G/B/P.

The generative model is unchanged: z ~ N(0, I_K), eta^X_il = f_l^T z_i with
no intercept, x_il ~ ExpFam_{c_l}(eta) in canonical form.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from family_specs import FamilySpec, GaussianSpec, NotEvaluable
from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent
from em_runner import EMFailFast
from eval_utils import calc_bic_exp

GENERIC_MODEL_VERSION = "generic-percolumn-v1"

_LEGACY = ("gaussian", "bernoulli", "poisson")


def _legacy_label(spec: FamilySpec) -> str:
    """Placeholder label for the parent's bookkeeping.

    The parent only uses its labels to find Gaussian columns (sigma, weights)
    once every family-dependent method below is overridden, so every
    non-Gaussian column is labelled with a non-Gaussian legacy name.
    """
    if spec.candidate_model in _LEGACY:
        return spec.candidate_model
    return "poisson"


class DualExpFamLSMGeneric(DualExpFamLSMPerColumnConsistent):
    """Per-column model whose columns are arbitrary scalar ``FamilySpec``s."""

    def __init__(self, n, d, k, L=10, *, column_specs: Sequence[FamilySpec],
                 family_y="bernoulli", sigma_y=1.0, train_mask=None):
        column_specs = list(column_specs)
        if len(column_specs) != d:
            raise ValueError(f"column_specs length {len(column_specs)} != d={d}")
        super().__init__(n=n, d=d, k=k, L=L,
                         family_x_list=[_legacy_label(s) for s in column_specs],
                         family_y=family_y, sigma_y=sigma_y,
                         train_mask=train_mask)
        self._install_specs(column_specs)

    # ------------------------------------------------------------------
    def _install_specs(self, column_specs: Sequence[FamilySpec]) -> None:
        self.column_specs = list(column_specs)
        legacy = [_legacy_label(s) for s in self.column_specs]
        self.family_x_list = legacy
        self._col_idx = {
            fam: np.array([j for j, f in enumerate(legacy) if f == fam], dtype=int)
            for fam in _LEGACY
        }
        self._all_gaussian = (len(self._col_idx["gaussian"]) == self.d)
        self.family_x = "gaussian" if self._all_gaussian else "mixed"
        groups: dict[str, list[int]] = {}
        for j, spec in enumerate(self.column_specs):
            groups.setdefault(spec.candidate_model, []).append(j)
        self._spec_groups = [(self.column_specs[idx[0]], np.array(idx, dtype=int))
                             for idx in groups.values()]
        self._suff_src = None
        self._suff = None
        self._limited_groups = [(spec, idx) for spec, idx in self._spec_groups
                                if spec.evaluation_range_limited]
        if not hasattr(self, "estep_backtracks"):
            self.estep_backtracks = 0

    def reassign_specs(self, column_specs: Sequence[FamilySpec]) -> None:
        column_specs = list(column_specs)
        if len(column_specs) != self.d:
            raise ValueError("column_specs length != d")
        self._install_specs(column_specs)

    def candidate_models(self) -> list[str]:
        return [s.candidate_model for s in self.column_specs]

    # ------------------------------------------------------------------
    # sufficient statistics T(X), cached per X object
    # ------------------------------------------------------------------
    def sufficient_statistics(self, X: np.ndarray) -> np.ndarray:
        if self._suff_src is X and self._suff is not None:
            return self._suff
        X = np.asarray(X)
        TX = np.empty(X.shape, dtype=np.float64)
        for spec, idx in self._spec_groups:
            if isinstance(spec, GaussianSpec):
                TX[:, idx] = X[:, idx]
            else:
                TX[:, idx] = spec.sufficient_statistic(X[:, idx])
        self._suff_src = X
        self._suff = TX
        return TX

    # ------------------------------------------------------------------
    # X link pieces (A' and A'') per column
    # ------------------------------------------------------------------
    def _mean_function_x(self, eta_x):
        eta_x = np.asarray(eta_x)
        out = np.empty_like(eta_x, dtype=float)
        for spec, idx in self._spec_groups:
            if isinstance(spec, GaussianSpec):
                out[..., idx] = eta_x[..., idx]
            else:
                out[..., idx] = spec.mean_statistic(eta_x[..., idx])
        return out

    def _variance_function_x(self, eta_x):
        eta_x = np.asarray(eta_x)
        out = np.empty_like(eta_x, dtype=float)
        for spec, idx in self._spec_groups:
            if isinstance(spec, GaussianSpec):
                out[..., idx] = 1.0
            else:
                out[..., idx] = spec.variance_statistic(eta_x[..., idx])
        return out

    # ------------------------------------------------------------------
    # E-step (parent code with x replaced by T(x))
    # ------------------------------------------------------------------
    def _calc_gradient(self, X, Y, Z, F, sigma, var_z, w0, w, i):
        z_i = Z[i, :]
        t_i = self.sufficient_statistics(X)[i, :]

        term1 = -(1.0 / var_z) * z_i

        eta_x_i = F @ z_i
        residual_x = t_i - self._mean_function_x(eta_x_i)
        term2 = F.T @ (residual_x * self._x_weight_vector(sigma))

        eta_y = w0 + w * (Z @ z_i)
        mu_y = self._mean_function(eta_y)
        residual_y = self._y_score_estep(Y[i, :], mu_y) * self._mask_f[i, :]
        term3 = w * (Z.T @ residual_y)

        return -(term1 + term2 + term3)

    # _calc_precision_matrix is inherited unchanged: it already reads A''
    # through _variance_function_x and the 1/sigma^2 weights.

    # ------------------------------------------------------------------
    # Laplace-Newton E-step with a domain guard for range-limited specs
    # ------------------------------------------------------------------
    def _x_evaluable(self, eta_x_i) -> bool:
        for spec, idx in self._limited_groups:
            try:
                spec.mean_statistic(eta_x_i[idx])
            except NotEvaluable:
                return False
        return True

    def calc_eta_newton(self, X, Y, rng=None, max_iter=10, alpha=0.01, tol=1e-4):
        """The existing Newton/Laplace sampler, plus one guard.

        Without a range-limited spec (every G/B/P configuration) this is the
        inherited method itself, so nothing changes.  With one (COM-Poisson),
        a Newton trial point at which that column's series cannot be
        certified is not taken: the SAME Newton step is halved until the
        point is evaluable (at most 60 halvings, else NotEvaluable).  The
        target -- the mode of the same log posterior and the Laplace
        covariance there -- is unchanged; every halving is counted in
        ``self.estep_backtracks`` and reported.
        """
        if not self._limited_groups:
            return super().calc_eta_newton(X, Y, rng=rng, max_iter=max_iter,
                                           alpha=alpha, tol=tol)
        if rng is None:
            rng = np.random.default_rng()
        Z = self.params['Z'].copy()
        F = self.params['F']
        sigma = self.params['sigma']
        var_z = self.params['var_z']
        w0 = self.params['w0']
        w = self.params['w']
        n, k = Z.shape
        for i in range(n):
            for iteration in range(max_iter):
                z_prev = Z[i, :].copy()
                grad = self._calc_gradient(X, Y, Z, F, sigma, var_z, w0, w, i)
                A_i = self._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)
                A_i = (A_i + A_i.T) / 2.0
                A_i += 1e-6 * np.eye(k)
                try:
                    A_i_inv = np.linalg.inv(A_i)
                except np.linalg.LinAlgError:
                    A_i_inv = np.linalg.pinv(A_i)
                step = alpha * (A_i_inv @ grad)
                proposal = Z[i, :] - step
                halvings = 0
                while not self._x_evaluable(F @ proposal):
                    halvings += 1
                    if halvings > 60:
                        raise NotEvaluable(f"E-step: no evaluable Newton step for "
                                           f"row {i} after 60 halvings")
                    step = 0.5 * step
                    proposal = Z[i, :] - step
                self.estep_backtracks += halvings
                Z[i, :] = proposal
                diff = np.mean(np.abs(Z[i, :] - z_prev))
                if diff < tol:
                    break
                if np.linalg.norm(grad) < tol:
                    break
            A_i = self._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)
            A_i = (A_i + A_i.T) / 2.0
            A_i += 1e-6 * np.eye(k)
            try:
                A_i_inv = np.linalg.inv(A_i)
                A_i_inv = (A_i_inv + A_i_inv.T) / 2.0
                Z[i, :] = rng.multivariate_normal(Z[i, :], A_i_inv)
            except (np.linalg.LinAlgError, ValueError):
                pass
        return Z

    # ------------------------------------------------------------------
    # M-step loading update (parent Adam with x replaced by T(x))
    # ------------------------------------------------------------------
    def _calc_F_adam_weighted(self, X, Z_samples, max_iter=50, lr=0.01,
                              beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6):
        n, k, L = Z_samples.shape
        TX = self.sufficient_statistics(X)
        F = self.params["F"].copy()
        sigma = self.params["sigma"]
        wvec = self._x_weight_vector(sigma)
        m = np.zeros_like(F)
        v = np.zeros_like(F)

        for t in range(1, max_iter + 1):
            grad = np.zeros_like(F)
            for l in range(L):
                Z_l = Z_samples[:, :, l]
                eta_x = Z_l @ F.T
                residual = (TX - self._mean_function_x(eta_x)) * wvec[None, :]
                grad += residual.T @ Z_l
            grad /= L

            g = -grad
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g ** 2
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            F_new = F - lr * m_hat / (np.sqrt(v_hat) + eps)
            if np.max(np.abs(F_new - F)) < tol:
                return F_new
            F = F_new
        return F

    # ------------------------------------------------------------------
    # complete X log-likelihood (base measure included)
    # ------------------------------------------------------------------
    def calc_log_likelihood_X(self, X, Z_samples, F):
        _, _, L = Z_samples.shape
        sigma = self.params["sigma"]
        ll = 0.0
        for sample in range(L):
            eta_x = Z_samples[:, :, sample] @ F.T
            for spec, idx in self._spec_groups:
                if isinstance(spec, GaussianSpec):
                    variance = np.maximum(np.diag(sigma)[idx], 1e-8)
                    resid = X[:, idx] - eta_x[:, idx]
                    ll += float(np.sum(-0.5 * resid ** 2 / variance
                                       - 0.5 * np.log(variance)
                                       - 0.5 * np.log(2.0 * np.pi)))
                else:
                    ll += float(np.sum(spec.log_prob(X[:, idx], eta_x[:, idx])))
        return ll / L

    def __repr__(self):                                    # pragma: no cover
        return (f"DualExpFamLSMGeneric(n={self.n}, d={self.d}, k={self.k}, "
                f"specs={self.candidate_models()}, family_y='{self.family}')")


# --------------------------------------------------------------------------
# strict Q (complete) and the generic fail-fast MCEM runner
# --------------------------------------------------------------------------

def calc_Q_strict_generic(X, Y, Z_samples, F, sigma, var_z, w0, w, model) -> float:
    """Complete-data Q with every normalising constant.

    Same assembly as ``eval_utils.calc_Q_dual_strict_exp`` except that the X
    part is already complete (``calc_log_likelihood_X`` above includes every
    base measure), so no X-side correction is added.
    """
    from scipy.special import gammaln

    n, k, L = Z_samples.shape
    label = getattr(model, "family_y_label", model.family)
    obs_upper = np.triu(model.train_mask, k=1)
    Q = 0.0
    for l in range(L):
        Z_l = Z_samples[:, :, l]
        lnpZ = float(-(n * k / 2.0) * np.log(2.0 * np.pi * var_z)
                     - (1.0 / (2.0 * var_z)) * np.sum(Z_l ** 2))
        lnpX = model.calc_log_likelihood_X(X, Z_samples[:, :, l:l + 1], F)
        lnpY = model.calc_log_likelihood_Y(Y, Z_samples[:, :, l:l + 1], w0, w)
        Q += lnpZ + lnpX + lnpY
    Q /= L
    corr = 0.0
    if label == "poisson":
        corr -= float(np.sum(gammaln(Y[obs_upper] + 1)))
    elif label == "gaussian":
        corr -= 0.5 * np.log(2.0 * np.pi) * float(obs_upper.sum())
    value = Q + corr
    if not np.isfinite(value):
        raise FloatingPointError("strict Q is not finite")
    return float(value)


def informed_init(model: DualExpFamLSMGeneric, Y: np.ndarray, family_y: str) -> None:
    """The em_runner informed initialisation, generalised to any spec.

    Y side is copied from ``em_runner.run_em_experimental``.  X side: every
    non-Gaussian row is shrunk by 0.2, which for Bernoulli/Poisson is exactly
    the existing rule and for the new canonical families keeps eta in the same
    numerically safe neighbourhood of 0.
    """
    obs_upper = np.triu(model.train_mask, k=1)
    y_obs = Y[obs_upper]
    if family_y == "bernoulli":
        observed_density = float(y_obs.mean())
        if not (0.0 < observed_density < 1.0):
            raise EMFailFast(f"observed Y density {observed_density}: no "
                             f"relational signal; refusing a clipped w0")
        density = float(np.clip(observed_density, 1e-6, 1 - 1e-6))
        model.params["w0"] = np.log(density / (1 - density))
        model.params["w"] = 0.5
    elif family_y == "poisson":
        mean_cnt = float(y_obs[y_obs > 0].mean()) if np.any(y_obs > 0) else 1.0
        model.params["w0"] = np.log(mean_cnt + 1e-10)
        model.params["w"] = 0.1
    else:
        model.params["w0"] = float(y_obs.mean())
        model.params["w"] = 0.5
        model.sigma_y = float(max(y_obs.std(), 0.01))
    for j, spec in enumerate(model.column_specs):
        if not isinstance(spec, GaussianSpec):
            model.params["F"][j, :] *= 0.2


def run_em_generic(X, Y, *, column_specs: Sequence[FamilySpec], family_y: str,
                   k: int, L: int = 5, num_iter: int = 8, seed: int = 42,
                   newton_alpha: float = 0.5, compute_strict_Q: bool = True,
                   model_factory=None, verbose: bool = False) -> dict[str, Any]:
    """Fail-fast MCEM with fixed per-column specs.

    The loop is ``em_runner.run_em_experimental`` with
    ``failure_policy='fail_fast'`` and ``numerics_mode='consistent'``,
    statement for statement (same RNG stream, same informed init, same
    M-step order).  There is no retry, no Z replacement and no seed change;
    a non-finite E-step raises ``EMFailFast``.
    """
    n, d = X.shape
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    factory = model_factory or DualExpFamLSMGeneric
    model = factory(n=n, d=d, k=k, L=L, column_specs=list(column_specs),
                    family_y=family_y)
    model.initialize_params(true_params=None, seed=seed)
    informed_init(model, Y, family_y)

    Z = model.params["Z"].copy()
    F = model.params["F"].copy()
    sigma = model.params["sigma"].copy()
    w0 = float(model.params["w0"])
    w = float(model.params["w"])
    var_z = float(model.params["var_z"])

    for iteration in range(1, num_iter + 1):
        Z_samples = np.zeros((n, k, L))
        for l in range(L):
            model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma, w0=w0, w=w))
            Z_new = model.calc_eta_newton(X, Y, rng=rng, max_iter=10,
                                          alpha=newton_alpha)
            Z_samples[:, :, l] = Z_new
            Z = Z_new.copy()
        if np.any(~np.isfinite(Z_samples)):
            raise EMFailFast(f"non-finite E-step at iteration {iteration} "
                             f"(seed={seed}); no repair, no retry")
        Z_samples = model.scale_Z(Z_samples)
        Z = Z_samples[:, :, -1].copy()

        F = model.calc_F(X, Z_samples)
        sigma = model.calc_sigma(X, Z_samples, F)
        w0 = float(model.calc_w0(Y, Z_samples, w0, w, max_iter=50))
        w = float(model.calc_w(Y, Z_samples, w0, w, max_iter=50))
        if family_y == "gaussian":
            model.calc_sigma_y(Y, Z_samples, w0, w)
        if verbose:                                        # pragma: no cover
            print(f"    iter={iteration:2d} w0={w0:.4f} w={w:.4f}")

    model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
    Q_strict = float("nan")
    bic = float("nan")
    npar = 0
    if compute_strict_Q:
        Q_strict = calc_Q_strict_generic(X, Y, Z_samples, F, sigma, var_z,
                                         w0, w, model)
        n_gauss = sum(isinstance(s, GaussianSpec) for s in model.column_specs)
        bic, npar = calc_bic_exp(Q_strict, k, n, d,
                                 "gaussian" if n_gauss == d else "mixed",
                                 model.family, n_gaussian_x_cols=n_gauss)
    return {
        "Z_est": Z_samples[:, :, -1].copy(),
        "Z_samples": Z_samples,
        "F": F, "sigma": sigma, "w0": w0, "w": w, "var_z": var_z,
        "model": model,
        "Q_strict": Q_strict, "bic": bic, "num_params": int(npar),
        "failure_policy": "fail_fast",
        "retry_count": 0, "replacement_count": 0, "seed_rescue_count": 0,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "estep_backtracks": int(getattr(model, "estep_backtracks", 0)),
        "numerics_mode": "consistent",
        "model_version": GENERIC_MODEL_VERSION,
    }


__all__ = ["GENERIC_MODEL_VERSION", "DualExpFamLSMGeneric",
           "calc_Q_strict_generic", "informed_init", "run_em_generic"]
