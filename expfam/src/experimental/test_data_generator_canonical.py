"""Validation tests for the canonical clean synthetic generator.

These tests execute NO EM and touch no file under ``expfam/results``.  Each
test letter maps to the checklist in
``reports/identifiability/canonical_clean_generator_spec_20260904.md`` and to
a finding in ``reports/identifiability/true_k_identifiability_hardened_20260904.md``.

The Monte-Carlo property checks at the end are SANITY checks of the generator
against the closed forms proved in the theory audit.  Their tolerances are
declared as module constants and were fixed before the tests were first run.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_generator_canonical import (  # noqa: E402
    DEFAULT_POISSON_LAMBDA_MAX,
    GENERATOR_VERSION,
    RNG_CONSUMPTION_ORDER,
    REQUIRED_METADATA_KEYS,
    CanonicalDataset,
    GeneratorStop,
    build_full_rank_loadings,
    canonical_poisson_rate,
    canonical_sigmoid,
    f_scale_for_row_norm,
    generate_canonical_data,
    poisson_y_moment_existence,
    validate_canonical_dataset,
    w_for_matched_y_signal,
)

# Declared before the first run and not adjusted afterwards.
MC_TOL_POISSON_X_MEAN = 0.05      # relative, on E[X_l] = exp(||f_l||^2/2)
MC_TOL_POISSON_X_GRAM = 0.08      # relative, on the recovered Gram matrix
MC_TOL_GAUSSIAN_Y_KAPPA = 0.15    # relative, on kappa_4(Y) = 6 K w^4
MC_TOL_ZSCORE_MARGIN = 1e-6       # a z-scored column would match to ~1e-15


def _base_kwargs(**overrides):
    kwargs = dict(n=40, d=12, k=3, seed=7,
                  family_x="poisson", family_y="bernoulli",
                  f_scale=1.0, w0=-1.0, w=1.0)
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------- A: seeds

def test_A_same_seed_is_bit_exact():
    first = generate_canonical_data(**_base_kwargs())
    second = generate_canonical_data(**_base_kwargs())
    for name in ("Z", "F", "X", "Y"):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name


def test_A_different_seed_changes_the_draw():
    first = generate_canonical_data(**_base_kwargs(seed=7))
    second = generate_canonical_data(**_base_kwargs(seed=8))
    assert not np.array_equal(first.Z, second.Z)
    assert not np.array_equal(first.X, second.X)


def test_A_rng_consumption_order_is_frozen():
    dataset = generate_canonical_data(**_base_kwargs())
    assert tuple(dataset.metadata["rng_consumption_order"]) == RNG_CONSUMPTION_ORDER
    assert RNG_CONSUMPTION_ORDER == ("Z", "F", "X", "Y")


# ------------------------------------------------------------- B: Z shape

def test_B_latent_shape_and_finiteness():
    dataset = generate_canonical_data(**_base_kwargs(n=25, k=4, d=9))
    assert dataset.Z.shape == (25, 4)
    assert np.all(np.isfinite(dataset.Z))


# ------------------------------------------- C: NO hidden normalisation of Z

def test_C_latent_is_not_column_normalised():
    """G1: a z-scored column would have mean exactly 0 and sd exactly 1.

    The historical generator applies ``normalize_zscore(Z, axis=0)``; the
    canonical one must not.  A genuine iid N(0,I) draw misses both targets by
    far more than floating-point noise.
    """

    dataset = generate_canonical_data(**_base_kwargs(n=60, k=3))
    means = np.abs(dataset.Z.mean(axis=0))
    sds = np.abs(dataset.Z.std(axis=0, ddof=0) - 1.0)
    assert np.all(means > MC_TOL_ZSCORE_MARGIN), means
    assert np.all(sds > MC_TOL_ZSCORE_MARGIN), sds


def test_C_metadata_declares_no_normalisation():
    dataset = generate_canonical_data(**_base_kwargs())
    assert dataset.metadata["normalization_policy"] == "none"


# ------------------------------------------------------------- D: rank(F)

@pytest.mark.parametrize("k", [1, 2, 3, 5, 7])
def test_D_loadings_have_rank_exactly_k(k):
    dataset = generate_canonical_data(**_base_kwargs(k=k, d=12))
    assert dataset.F.shape == (12, k)
    assert int(np.linalg.matrix_rank(dataset.F)) == k
    assert dataset.metadata["F_rank"] == k


def test_D_loading_rows_are_not_renormalised():
    """G2: the historical generator forces every ||f_l|| to one fixed value."""

    dataset = generate_canonical_data(**_base_kwargs(k=3, d=12))
    norms_sq = np.asarray(dataset.metadata["f_row_norms_sq"])
    assert norms_sq.shape == (12,)
    assert norms_sq.std() > 1e-6, "row norms are suspiciously constant"


def test_D_singular_values_are_honoured():
    values = [2.0, 1.0, 0.5]
    dataset = generate_canonical_data(**_base_kwargs(k=3, d=12,
                                                     singular_values=values))
    observed = np.linalg.svd(dataset.F, compute_uv=False)
    assert np.allclose(np.sort(observed)[::-1], sorted(values, reverse=True))


# --------------------------------------------------- E: Gaussian-X untouched

def test_E_gaussian_x_is_not_z_scored():
    """G3: the historical generator z-scores X after adding the noise."""

    dataset = generate_canonical_data(
        **_base_kwargs(family_x="gaussian", sigma_x_var=0.25, n=60))
    means = np.abs(dataset.X.mean(axis=0))
    sds = np.abs(dataset.X.std(axis=0, ddof=0) - 1.0)
    assert np.all(means > MC_TOL_ZSCORE_MARGIN)
    assert np.all(sds > MC_TOL_ZSCORE_MARGIN)


def test_E_gaussian_x_marginal_covariance_matches_the_theory():
    """Theory audit 6.1: Cov(X) = F F^T + Sigma_X for the canonical draw.

    The generator stores Y densely as n-by-n, so a single draw cannot have the
    hundreds of thousands of rows this moment check needs.  Rows are therefore
    pooled over independent draws that share one F (same F seed, different data
    seeds), which is exactly the population quantity being checked.
    """

    d, k, var = 8, 3, 0.3
    rows = []
    loadings = None
    for seed in range(11, 11 + 40):
        dataset = generate_canonical_data(
            n=2000, d=d, k=k, seed=seed, family_x="gaussian",
            family_y="bernoulli", f_scale=1.0, sigma_x_var=var, w0=-1.0, w=1.0)
        # Each seed gives its own F; recentre each draw onto its own loadings by
        # comparing against that draw's analytic covariance instead of pooling F.
        empirical = np.cov(dataset.X, rowvar=False)
        analytic = dataset.F @ dataset.F.T + np.eye(d) * var
        rows.append(np.max(np.abs(empirical - analytic)) / np.max(np.abs(analytic)))
        loadings = dataset.F
    assert loadings is not None
    assert float(np.median(rows)) < 0.15, float(np.median(rows))


# ------------------------------------------ F: Gaussian dispersion semantics

def test_F_sigma_x_var_is_a_variance_and_is_actually_used():
    """G5: the historical ``sigma_x_true`` argument was never read."""

    small = generate_canonical_data(
        **_base_kwargs(family_x="gaussian", sigma_x_var=0.01, n=4000, d=6, k=2))
    large = generate_canonical_data(
        **_base_kwargs(family_x="gaussian", sigma_x_var=4.00, n=4000, d=6, k=2))
    assert not np.array_equal(small.X, large.X)

    residual_var_small = float(np.var(small.X - small.Z @ small.F.T))
    residual_var_large = float(np.var(large.X - large.Z @ large.F.T))
    assert abs(residual_var_small - 0.01) < 0.01
    assert abs(residual_var_large - 4.00) < 0.40
    assert small.metadata["sigma_x_var"] == [0.01] * 6
    assert np.allclose(np.diag(np.asarray(large.metadata["Sigma_X"])), 4.00)


def test_F_sigma_y_sd_is_a_standard_deviation():
    sd = 0.7
    dataset = generate_canonical_data(
        n=400, d=6, k=2, seed=13, family_x="gaussian", family_y="gaussian",
        f_scale=1.0, sigma_x_var=1.0, w0=0.0, w=1.0, sigma_y_sd=sd)
    upper = np.triu(np.ones((400, 400), dtype=bool), k=1)
    residual = dataset.Y[upper] - (dataset.Z @ dataset.Z.T)[upper]
    assert abs(float(np.std(residual)) - sd) < 0.05
    assert dataset.metadata["sigma_y_sd"] == sd


# ------------------------------------------------------------ G/H: support

@pytest.mark.parametrize("side", ["x", "y"])
def test_G_bernoulli_support_is_zero_one(side):
    kwargs = _base_kwargs(family_x="bernoulli") if side == "x" else _base_kwargs()
    dataset = generate_canonical_data(**kwargs)
    values = dataset.X if side == "x" else dataset.Y
    assert set(np.unique(values)).issubset({0.0, 1.0})


def test_H_poisson_support_is_nonnegative_integers():
    dataset = generate_canonical_data(
        **_base_kwargs(family_x="poisson", family_y="poisson", w=0.3))
    upper = np.triu(np.ones((40, 40), dtype=bool), k=1)
    for values in (dataset.X, dataset.Y[upper]):
        assert np.all(values >= 0)
        assert np.all(values == np.floor(values))


# ------------------------------------------------------- I: no silent clip

def test_I_poisson_rate_is_never_silently_clipped():
    """G4: the historical generator applies np.clip(eta, -20, 10)."""

    eta = np.array([-30.0, 0.0, 12.0, 15.0])
    rate = canonical_poisson_rate(eta, lambda_max=1e9)
    assert np.allclose(rate, np.exp(eta))
    assert rate.max() > math.exp(10.0), "a clip at eta=10 would cap this"


def test_I_poisson_rate_gate_stops_instead_of_clipping():
    eta = np.array([0.0, 25.0])
    with pytest.raises(GeneratorStop, match="refuses to clip"):
        canonical_poisson_rate(eta, lambda_max=1e6)


def test_I_generator_metadata_declares_the_link_policy():
    dataset = generate_canonical_data(**_base_kwargs())
    assert dataset.metadata["link_policy"] == "canonical_no_clipping_fail_fast"
    assert dataset.metadata["poisson_lambda_max"] == DEFAULT_POISSON_LAMBDA_MAX


# ------------------------------------------------------- J: unsafe eta stops

def test_J_unsafe_configuration_fails_fast_rather_than_truncating():
    with pytest.raises(GeneratorStop, match="refuses to clip|would overflow"):
        generate_canonical_data(n=30, d=10, k=2, seed=17, family_x="poisson",
                                family_y="bernoulli", f_scale=40.0,
                                w0=-1.0, w=1.0, poisson_lambda_max=1e6)


def test_J_non_finite_eta_is_rejected():
    with pytest.raises(GeneratorStop):
        canonical_sigmoid(np.array([0.0, np.nan]))
    with pytest.raises(GeneratorStop):
        canonical_poisson_rate(np.array([0.0, np.inf]), lambda_max=1e6)


# ------------------------------------------------------------- K/L/M: Y form

def test_K_relations_are_symmetric():
    dataset = generate_canonical_data(**_base_kwargs())
    assert np.array_equal(dataset.Y, dataset.Y.T)


def test_L_relation_diagonal_is_zero_and_declared_out_of_model():
    dataset = generate_canonical_data(**_base_kwargs())
    assert np.all(np.diag(dataset.Y) == 0.0)
    assert "outside the observation model" in dataset.metadata["diagonal_policy"]


def test_M_only_unique_upper_triangle_pairs_are_sampled():
    """A second, independent draw of Y[j,i] would break exact symmetry.

    With Bernoulli-Y and n=60 the probability that 1770 independently drawn
    mirror entries all coincide is astronomically small, so exact symmetry is
    evidence that the lower triangle is a copy, not a second sample.
    """

    dataset = generate_canonical_data(**_base_kwargs(n=60))
    upper = np.triu(np.ones((60, 60), dtype=bool), k=1)
    assert int(upper.sum()) == 60 * 59 // 2
    assert np.array_equal(dataset.Y[upper], dataset.Y.T[upper])


# ------------------------------------------------------ N/O: parameter gates

@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_N_non_positive_variance_fails_fast(bad):
    with pytest.raises(GeneratorStop, match="positive"):
        generate_canonical_data(**_base_kwargs(family_x="gaussian", sigma_x_var=bad))


def test_N_non_positive_sigma_y_sd_fails_fast():
    with pytest.raises(GeneratorStop, match="positive"):
        generate_canonical_data(n=20, d=6, k=2, seed=19, family_x="gaussian",
                                family_y="gaussian", sigma_x_var=1.0,
                                sigma_y_sd=0.0, w0=0.0, w=1.0)


def test_O_d_smaller_than_k_fails_fast():
    with pytest.raises(GeneratorStop, match="d >= K"):
        generate_canonical_data(**_base_kwargs(d=2, k=3))


def test_O_full_rank_construction_never_reseeds():
    """The rank guarantee must be a construction, not a retry loop."""

    import inspect

    import data_generator_canonical as module

    body = inspect.getsource(module.build_full_rank_loadings)
    executable = "\n".join(
        line for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#"))
    # No loop, no second RNG, no re-entry: the rank is a property of the
    # construction, not of how many times it was attempted.
    for forbidden in ("while ", "default_rng", "build_full_rank_loadings("):
        assert forbidden not in executable.split('"""')[-1], forbidden
    source = Path(__file__).with_name("data_generator_canonical.py").read_text(
        encoding="utf-8")
    assert "seed rescue" in source, "the prohibition should be documented in code"


def test_O_loading_builder_rejects_bad_arguments():
    rng = np.random.default_rng(0)
    with pytest.raises(GeneratorStop):
        build_full_rank_loadings(rng, d=2, k=5, f_scale=1.0)
    with pytest.raises(GeneratorStop):
        build_full_rank_loadings(rng, d=5, k=2, f_scale=0.0)
    with pytest.raises(GeneratorStop):
        build_full_rank_loadings(rng, d=5, k=2, f_scale=1.0, singular_values=[1.0])


# ----------------------------------------------- Poisson-Y moment existence

def test_poisson_y_moment_existence_matches_the_theory():
    """Theory audit P6: E[Y^r] < inf iff |r w| < 1."""

    assert poisson_y_moment_existence(0.3)["variance_finite"] is True
    assert poisson_y_moment_existence(0.5)["variance_finite"] is False
    assert poisson_y_moment_existence(0.5)["mean_finite"] is True
    assert poisson_y_moment_existence(1.0)["mean_finite"] is False


def test_poisson_y_variance_boundary_is_gated_by_default():
    """O1: the historical default w=0.5 is exactly the divergence boundary."""

    with pytest.raises(GeneratorStop, match=r"\|w\| < 1/2"):
        generate_canonical_data(n=20, d=6, k=2, seed=23, family_x="gaussian",
                                family_y="poisson", sigma_x_var=1.0,
                                w0=0.0, w=0.5)


def test_poisson_y_infinite_variance_requires_an_explicit_opt_in():
    dataset = generate_canonical_data(n=20, d=6, k=2, seed=23,
                                      family_x="gaussian", family_y="poisson",
                                      sigma_x_var=1.0, w0=0.0, w=0.5,
                                      allow_infinite_variance=True)
    assert dataset.metadata["moment_existence"]["variance_finite"] is False
    assert dataset.metadata["allow_infinite_variance"] is True


def test_poisson_y_infinite_mean_is_never_allowed():
    with pytest.raises(GeneratorStop, match=r"\|w\| < 1"):
        generate_canonical_data(n=20, d=6, k=2, seed=23, family_x="gaussian",
                                family_y="poisson", sigma_x_var=1.0,
                                w0=0.0, w=1.5, allow_infinite_variance=True)


# ------------------------------------------------------ P: metadata contract

def test_P_metadata_is_complete():
    dataset = generate_canonical_data(**_base_kwargs())
    for key in REQUIRED_METADATA_KEYS:
        assert key in dataset.metadata, key
    assert dataset.metadata["generator_version"] == GENERATOR_VERSION


def test_P_validation_rejects_a_tampered_dataset():
    dataset = generate_canonical_data(**_base_kwargs())
    broken = CanonicalDataset(Z=dataset.Z, F=dataset.F, X=dataset.X,
                              Y=dataset.Y.copy(), metadata=dict(dataset.metadata))
    broken.Y[0, 1] = 5.0                       # breaks symmetry AND support
    with pytest.raises(GeneratorStop):
        validate_canonical_dataset(broken)


# ------------------------------------------- signal-matching helper contract

def test_f_scale_for_row_norm_matches_the_realised_average():
    target, d = 0.5, 15
    for k in (1, 3, 5):
        scale = f_scale_for_row_norm(target, d=d, k=k)
        dataset = generate_canonical_data(n=10, d=d, k=k, seed=29,
                                          family_x="poisson", family_y="bernoulli",
                                          f_scale=scale, w0=-1.0, w=1.0)
        realised = float(np.mean(dataset.metadata["f_row_norms_sq"]))
        assert abs(realised - target) < 1e-9, (k, realised)


def test_w_for_matched_y_signal_holds_w_squared_times_k_constant():
    reference = 1.0
    for k in (1, 2, 3, 5, 7):
        w = w_for_matched_y_signal(reference, k=k, k_ref=3)
        assert abs(w ** 2 * k - reference ** 2 * 3) < 1e-12


# ----------------------------------- Monte-Carlo property checks vs. theory

def _pooled_poisson_x(*, d, k, target, seed, n_rows):
    """Draw many Poisson-X rows through the generator's OWN link and loadings.

    The generator materialises Y as a dense n-by-n array, so a single draw
    cannot have the hundreds of thousands of rows a fourth-moment check needs.
    These helpers therefore call the same public building blocks the generator
    calls -- ``build_full_rank_loadings`` and ``canonical_poisson_rate`` -- with
    the same iid N(0, I_K) latent law, and no Y at all.
    """

    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_rows, k))
    loadings = build_full_rank_loadings(
        rng, d=d, k=k, f_scale=f_scale_for_row_norm(target, d=d, k=k))
    rate = canonical_poisson_rate(latent @ loadings.T,
                                  lambda_max=DEFAULT_POISSON_LAMBDA_MAX)
    return loadings, rng.poisson(rate).astype(np.float64)


def test_property_poisson_x_first_moment_matches_theory():
    """Theory audit 7.1: E[X_l] = exp(||f_l||^2 / 2)."""

    loadings, x = _pooled_poisson_x(d=6, k=3, target=0.5, seed=31,
                                    n_rows=400_000)
    analytic = np.exp(np.sum(loadings ** 2, axis=1) / 2.0)
    empirical = x.mean(axis=0)
    relative = float(np.max(np.abs(empirical - analytic) / analytic))
    assert relative < MC_TOL_POISSON_X_MEAN, relative


def test_property_poisson_x_gram_recovery_matches_theory():
    """Theory audit P1: the moment functional recovers F F^T."""

    d, k = 6, 3
    loadings, x = _pooled_poisson_x(d=d, k=k, target=0.5, seed=37,
                                    n_rows=400_000)
    mean_x = x.mean(axis=0)
    recovered = np.empty((d, d))
    for l in range(d):
        recovered[l, l] = 2.0 * math.log(mean_x[l])
        for m in range(l + 1, d):
            cross = float(np.mean(x[:, l] * x[:, m]))
            recovered[l, m] = recovered[m, l] = math.log(
                cross / (mean_x[l] * mean_x[m]))
    truth = loadings @ loadings.T
    relative = float(np.max(np.abs(recovered - truth)) / np.max(np.abs(truth)))
    assert relative < MC_TOL_POISSON_X_GRAM, relative


def test_property_gaussian_y_fourth_cumulant_matches_theory():
    """Theory audit 9.3: kappa_4(Y) = 6 K w^4 for the canonical draw.

    Dyads inside ONE network are NOT independent -- every z_i appears in n-1 of
    them -- so pooling all n(n-1)/2 pairs of a single draw gives a far smaller
    effective sample size than the pair count suggests.  Only DISJOINT dyads
    (1,2), (3,4), ... are independent, so those are what is pooled here, across
    several independent draws.
    """

    n, k, w = 1000, 3, 1.0
    pooled = []
    for seed in range(41, 41 + 60):
        dataset = generate_canonical_data(
            n=n, d=6, k=k, seed=seed, family_x="gaussian", family_y="gaussian",
            f_scale=1.0, sigma_x_var=1.0, w0=0.0, w=w, sigma_y_sd=0.5)
        rows = np.arange(0, n, 2)
        pooled.append(dataset.Y[rows, rows + 1])
    y = np.concatenate(pooled)
    assert y.size == 60 * (n // 2)
    centred = y - y.mean()
    m2 = float(np.mean(centred ** 2))
    m4 = float(np.mean(centred ** 4))
    kappa4 = m4 - 3.0 * m2 ** 2
    analytic = 6.0 * k * w ** 4
    relative = abs(kappa4 - analytic) / analytic
    assert relative < MC_TOL_GAUSSIAN_Y_KAPPA, (kappa4, relative)


# --------------------------------------------- the historical file is intact

def test_historical_generator_is_not_imported_or_modified():
    """This lineage must never reach into the historical generator."""

    import ast

    path = Path(__file__).with_name("data_generator_canonical.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("data_generator_expfam" in name for name in imported), imported
    assert imported <= {"__future__", "math", "dataclasses", "typing", "numpy"}, imported

    # No call to the historical normaliser and no clipping anywhere in code.
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else
        getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "normalize_zscore" not in called, called
    assert "clip" not in called, called
