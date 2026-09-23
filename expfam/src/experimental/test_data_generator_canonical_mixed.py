"""Zero-fit tests for the mixed-X canonical generator (Issue #74, Scope B).

These tests execute NO EM and touch no file under ``expfam/results``.  A
module-scoped autouse fixture makes an accidental MCEM call raise, so "no EM
was run" is enforced rather than asserted in prose.
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
    GENERATOR_VERSION,
    GeneratorStop,
    f_scale_for_row_norm,
    generate_canonical_data,
)
from data_generator_canonical_mixed import (  # noqa: E402
    MIXED_GENERATOR_VERSION,
    MIXED_RNG_CONSUMPTION_ORDER,
    REQUIRED_MIXED_METADATA_KEYS,
    generate_canonical_mixed_data,
    validate_mixed_canonical_dataset,
)

BASE = dict(n=30, d=6, k=3, family_y="bernoulli",
            f_scale=f_scale_for_row_norm(0.5, d=6, k=3),
            sigma_x_var=1.0, w0=-1.0, w=1.0)
MIXED_LIST = ["gaussian", "gaussian", "bernoulli", "bernoulli", "poisson", "poisson"]


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    """Fail loudly if anything in this module tries to run an EM fit."""

    import em_runner
    from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent

    def _no_em(*_args, **_kwargs):
        raise AssertionError("an EM fit was attempted inside a zero-fit test")

    monkeypatch.setattr(em_runner, "run_em_experimental", _no_em)
    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton", _no_em)


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_shapes_finiteness_and_symmetry():
    data = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    n, d, k = BASE["n"], BASE["d"], BASE["k"]
    assert data.Z.shape == (n, k)
    assert data.F.shape == (d, k)
    assert data.X.shape == (n, d)
    assert data.Y.shape == (n, n)
    for array in (data.Z, data.F, data.X, data.Y):
        assert np.all(np.isfinite(array))
    assert np.array_equal(data.Y, data.Y.T)
    assert np.all(np.diag(data.Y) == 0.0)
    assert int(np.linalg.matrix_rank(data.F)) == k


def test_per_column_support_matches_its_family():
    data = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    for column, family in enumerate(MIXED_LIST):
        values = data.X[:, column]
        if family == "bernoulli":
            assert set(np.unique(values)).issubset({0.0, 1.0})
        elif family == "poisson":
            assert np.all(values >= 0.0)
            assert np.all(values == np.floor(values))
        else:
            # A Gaussian column is almost surely not integer-valued.
            assert not np.all(values == np.floor(values))


def test_f_and_z_are_shared_across_columns():
    """One Z and one F drive every column; only the observation law differs."""

    data = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    eta = data.Z @ data.F.T
    assert eta.shape == data.X.shape
    # For a Gaussian column the residual around its own eta has the declared
    # variance; that only holds if the column really used the shared eta.
    residual = data.X[:, 0] - eta[:, 0]
    assert abs(float(np.var(residual)) - 1.0) < 0.6


# --------------------------------------------------------------------------
# reproducibility and RNG consumption order
# --------------------------------------------------------------------------

def test_same_seed_and_config_is_bitwise_reproducible():
    first = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    second = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    for name in ("Z", "F", "X", "Y"):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name


def test_a_different_seed_gives_different_data():
    first = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    second = generate_canonical_mixed_data(seed=941002, family_x_list=MIXED_LIST, **BASE)
    assert not np.array_equal(first.X, second.X)


def test_z_and_f_match_the_scalar_generator_but_x_does_not():
    """Z and F are drawn first, in the same order, so they must agree.

    X does not, and must not be expected to: the mixed draw consumes the RNG
    one column at a time (``MIXED_RNG_CONSUMPTION_ORDER``) instead of drawing
    the whole matrix at once.  Pinning this down stops a later reader from
    assuming the two generators are interchangeable.
    """

    uniform = ["gaussian"] * BASE["d"]
    mixed = generate_canonical_mixed_data(seed=941001, family_x_list=uniform, **BASE)
    scalar = generate_canonical_data(seed=941001, family_x="gaussian", **BASE)
    assert np.array_equal(mixed.Z, scalar.Z)
    assert np.array_equal(mixed.F, scalar.F)
    assert not np.array_equal(mixed.X, scalar.X)
    assert mixed.metadata["generator_version"] == MIXED_GENERATOR_VERSION
    assert scalar.metadata["generator_version"] == GENERATOR_VERSION


def test_z_is_not_normalised_after_drawing():
    data = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    expected = np.random.default_rng(941001).standard_normal(
        (BASE["n"], BASE["k"]))
    assert np.array_equal(data.Z, expected)
    assert abs(float(np.mean(data.Z ** 2)) - 1.0) > 1e-12
    assert data.metadata["normalization_policy"] == "none"


# --------------------------------------------------------------------------
# metadata / provenance
# --------------------------------------------------------------------------

def test_metadata_is_complete_and_records_the_family_list():
    data = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    meta = data.metadata
    missing = [key for key in REQUIRED_MIXED_METADATA_KEYS if key not in meta]
    assert missing == []
    assert meta["family_x_list"] == MIXED_LIST
    assert meta["family_x_counts"] == {"gaussian": 2, "bernoulli": 2, "poisson": 2}
    assert tuple(meta["rng_consumption_order"]) == MIXED_RNG_CONSUMPTION_ORDER
    assert meta["link_policy"] == "canonical_no_clipping_fail_fast"
    assert meta["seed"] == 941001
    assert meta["F_rank"] == BASE["k"]


def test_dispersion_metadata_is_per_column_and_honest():
    variances = [0.25, 4.0, 1.0, 1.0, 1.0, 1.0]
    data = generate_canonical_mixed_data(
        seed=941001, family_x_list=MIXED_LIST,
        **{**BASE, "sigma_x_var": variances})
    recorded = data.metadata["sigma_x_var"]
    assert recorded[0] == 0.25 and recorded[1] == 4.0
    # Non-Gaussian columns consumed no variance, so none is claimed.
    assert recorded[2:] == [None, None, None, None]
    # The declared variance is actually used (G5).
    eta = data.Z @ data.F.T
    assert np.var(data.X[:, 0] - eta[:, 0]) < np.var(data.X[:, 1] - eta[:, 1])


def test_expected_x_mean_is_only_defined_for_poisson_columns():
    data = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    expected = data.metadata["expected_x_mean"]
    norms = data.metadata["f_row_norms_sq"]
    for column, family in enumerate(MIXED_LIST):
        if family == "poisson":
            assert expected[column] == pytest.approx(math.exp(norms[column] / 2.0))
        else:
            assert expected[column] is None


# --------------------------------------------------------------------------
# fail-fast: no clipping, no silent repair, no seed rescue
# --------------------------------------------------------------------------

def test_unsafe_poisson_rate_stops_instead_of_clipping():
    with pytest.raises(GeneratorStop, match="refuses to clip"):
        generate_canonical_mixed_data(
            seed=941001, family_x_list=["poisson"] * BASE["d"],
            **{**BASE, "f_scale": 40.0, "poisson_lambda_max": 1.0e3})


def test_malformed_configurations_are_rejected():
    with pytest.raises(GeneratorStop, match="family_x_list length"):
        generate_canonical_mixed_data(
            seed=1, family_x_list=["gaussian"] * 3, **BASE)
    with pytest.raises(GeneratorStop, match="not one of"):
        generate_canonical_mixed_data(
            seed=1, family_x_list=["gaussian"] * 5 + ["categorical"], **BASE)
    with pytest.raises(GeneratorStop, match="d >= K"):
        generate_canonical_mixed_data(
            seed=1, family_x_list=["gaussian"] * 2,
            **{**BASE, "d": 2, "k": 3})
    with pytest.raises(GeneratorStop, match="strictly positive"):
        generate_canonical_mixed_data(
            seed=1, family_x_list=MIXED_LIST,
            **{**BASE, "sigma_x_var": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]})


def test_validator_catches_a_corrupted_dataset():
    data = generate_canonical_mixed_data(seed=941001, family_x_list=MIXED_LIST, **BASE)
    data.X[0, 2] = 0.5           # column 2 is declared Bernoulli
    with pytest.raises(GeneratorStop, match="Bernoulli but has a value"):
        validate_mixed_canonical_dataset(data)
