"""Gate B: the generic per-column path reproduces the existing G/B/P path.

If any test here fails, the multi-support experiments must not run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from data_generator_canonical_mixed import generate_canonical_mixed_data  # noqa: E402
from em_runner import run_em_experimental                                # noqa: E402
from eval_utils import calc_Q_dual_strict_exp                            # noqa: E402
from family_selection import run_hybrid_family_selection                 # noqa: E402
from family_selection_multisupport import run_generic_hybrid            # noqa: E402
from family_specs import (                                               # noqa: E402
    GROUP_BOUNDED, GROUP_COUNT, GROUP_POSITIVE, GROUP_REAL,
    BernoulliSpec, GaussianSpec, PoissonSpec,
)
from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent  # noqa: E402
from model_dual_expfam_generic import (                                  # noqa: E402
    DualExpFamLSMGeneric, run_em_generic,
)

SPEC = {"gaussian": GaussianSpec(), "bernoulli": BernoulliSpec(),
        "poisson": PoissonSpec()}
FAMILIES = ["gaussian"] * 3 + ["bernoulli"] * 4 + ["poisson"] * 3
PHASE9C_REGISTRY = dict(SPEC)
PHASE9C_PRIORITY = {GROUP_COUNT: ("bernoulli", "poisson"),
                    GROUP_POSITIVE: ("gaussian",), GROUP_BOUNDED: ("gaussian",),
                    GROUP_REAL: ("gaussian",)}


@pytest.fixture(scope="module")
def data():
    ds = generate_canonical_mixed_data(n=40, d=len(FAMILIES), k=3, seed=4242,
                                       family_x_list=FAMILIES, f_scale=1.2)
    return ds


def _pair(ds, k=3, L=3):
    n, d = ds.X.shape
    old = DualExpFamLSMPerColumnConsistent(n=n, d=d, k=k, L=L,
                                           family_x_list=FAMILIES)
    new = DualExpFamLSMGeneric(n=n, d=d, k=k, L=L,
                               column_specs=[SPEC[f] for f in FAMILIES])
    old.initialize_params(seed=7)
    new.initialize_params(seed=7)
    return old, new


def test_estep_gradient_and_precision_bitwise(data):
    old, new = _pair(data)
    p = old.params
    sigma = np.diag(np.linspace(0.5, 2.0, data.X.shape[1]))
    for i in range(data.X.shape[0]):
        g_old = old._calc_gradient(data.X, data.Y, p["Z"], p["F"], sigma, 1.0, -1.0, 0.7, i)
        g_new = new._calc_gradient(data.X, data.Y, p["Z"], p["F"], sigma, 1.0, -1.0, 0.7, i)
        np.testing.assert_array_equal(g_old, g_new)
        A_old = old._calc_precision_matrix(p["Z"], p["F"], sigma, 1.0, -1.0, 0.7, i)
        A_new = new._calc_precision_matrix(p["Z"], p["F"], sigma, 1.0, -1.0, 0.7, i)
        np.testing.assert_array_equal(A_old, A_new)


def test_mstep_loading_update_bitwise(data):
    old, new = _pair(data)
    Zs = np.random.default_rng(1).standard_normal((data.X.shape[0], 3, 3))
    np.testing.assert_array_equal(old.calc_F(data.X, Zs), new.calc_F(data.X, Zs))
    np.testing.assert_array_equal(old.calc_sigma(data.X, Zs, old.params["F"]),
                                  new.calc_sigma(data.X, Zs, new.params["F"]))


def test_complete_x_loglik_equals_existing_plus_base_measure(data):
    from scipy.special import gammaln
    old, new = _pair(data)
    Zs = np.random.default_rng(2).standard_normal((data.X.shape[0], 3, 2))
    pois = [j for j, f in enumerate(FAMILIES) if f == "poisson"]
    expected = (old.calc_log_likelihood_X(data.X, Zs, old.params["F"])
                - float(np.sum(gammaln(data.X[:, pois] + 1))))
    got = new.calc_log_likelihood_X(data.X, Zs, new.params["F"])
    assert abs(got - expected) < 1e-9 * abs(expected)


def test_full_mcem_bitwise_and_strict_q(data):
    ref = run_em_experimental(data.X, data.Y, family_x="mixed", family_y="bernoulli",
                              k=3, L=3, num_iter=4, seed=99, family_x_list=FAMILIES,
                              compute_strict_Q=True, numerics_mode="consistent",
                              failure_policy="fail_fast")
    gen = run_em_generic(data.X, data.Y, column_specs=[SPEC[f] for f in FAMILIES],
                         family_y="bernoulli", k=3, L=3, num_iter=4, seed=99)
    for key in ("Z_est", "F", "sigma"):
        np.testing.assert_array_equal(ref[key], gen[key])
    assert ref["w0"] == gen["w0"] and ref["w"] == gen["w"]
    assert abs(ref["Q_strict"] - gen["Q_strict"]) < 1e-9 * abs(ref["Q_strict"])
    assert abs(ref["bic"] - gen["bic"]) < 1e-9 * abs(ref["bic"])
    assert ref["num_params"] == gen["num_params"]
    # strict Q from the existing helper on the generic fit's state agrees too
    q_old = calc_Q_dual_strict_exp(data.X, data.Y, gen["Z_samples"], gen["F"],
                                   gen["sigma"], 1.0, gen["w0"], gen["w"], ref["model"])
    assert abs(q_old - gen["Q_strict"]) < 1e-9 * abs(q_old)


@pytest.mark.parametrize("start,named", [("START_FIRST", "bernoulli"),
                                         ("START_LAST", "poisson")])
def test_phase9c_hybrid_reproduced(data, start, named):
    old = run_hybrid_family_selection(data.X, data.Y, k=3, ambiguous_start=named,
                                      L=3, exploration_num_iter=3, refit_num_iter=3,
                                      search_seed=11, refit_seed=12)
    new = run_generic_hybrid(data.X, data.Y, k=3, registry=PHASE9C_REGISTRY,
                             priority=PHASE9C_PRIORITY, start=start, L=3,
                             exploration_num_iter=3, refit_num_iter=3,
                             search_seed=11, refit_seed=12)
    assert old["initial_assignment"] == new["start_models"]
    assert old["selected_assignment"] == new["exploration"]["selected_models"]
    old_trace = old["selection_trace"]
    new_trace = new["exploration"]["selection_trace"]
    assert len(old_trace) == len(new_trace)
    for a, b in zip(old_trace, new_trace):
        assert a["iteration"] == b["iteration"] and a["column"] == b["column"]
        assert a["selected_family"] == b["selected_candidate"]
        assert a["margin_neg2"] == b["margin_neg2"]
        assert a["selected_loading_sha256"] == b["selected_loading_sha256"]
    for key in ("Z_est", "F", "sigma"):
        np.testing.assert_array_equal(old["refit"][key], new["refit"][key])
    assert abs(old["refit"]["Q_strict"] - new["refit"]["Q_strict"]) \
        < 1e-9 * abs(old["refit"]["Q_strict"])
