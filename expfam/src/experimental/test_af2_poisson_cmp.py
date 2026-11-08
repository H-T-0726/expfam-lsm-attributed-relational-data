"""Tests for AF2 (Poisson vs COM-Poisson nu=2) and the opt-in train_mask
pass-through added to the PR #129 generic selector / EM.

Research contracts protected:
  * train_mask=None and an explicit all-pairs mask give the same result
    (pass-through does not change PR #129 behaviour);
  * Y outside the train mask cannot influence exploration, candidate scores,
    the selected models or the refit; both stages receive the same mask;
  * the E1 joint-mode evaluator is valid for the generic (CMP) model: its
    gradient and negative Hessian match finite differences of Phi;
  * same seeds reproduce; an Auto refit on the Oracle models equals Oracle;
  * the analytic orderings used in the AF2 README (Part A) hold numerically;
  * design contracts: plan, seeds, swap / support-only assignments.
EM runs use reduced iteration counts; the contracts do not depend on them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent.parent / "tools" / "research_audit"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import family_selection_multisupport as fsm                             # noqa: E402
import laplace_k_criterion as lk                                        # noqa: E402
import run_af2_poisson_cmp as af2                                       # noqa: E402
import run_misspecification_m1 as m1                                    # noqa: E402
from model_dual_expfam_generic import run_em_generic                    # noqa: E402

ITERS = 2


@pytest.fixture(scope="module")
def rep1():
    return af2.generate(0), m1.build_masks(0)


def _hybrid(ds, mask, Y=None, start="START_FIRST", seed=21, refit_seed=22):
    reg, prio = af2.registry_and_priority()
    return fsm.run_generic_hybrid(
        ds.X, ds.Y if Y is None else Y, k=3, registry=reg, priority=prio, start=start,
        L=5, exploration_num_iter=ITERS, refit_num_iter=ITERS, search_seed=seed,
        refit_seed=refit_seed, route=af2.ROUTE, train_mask=mask)


def _same(a, b):
    ea, eb = a["exploration"], b["exploration"]
    assert ea["selected_models"] == eb["selected_models"]
    for ta, tb in zip(ea["selection_trace"], eb["selection_trace"], strict=True):
        for key in ("candidate_scores_json", "selected_candidate", "margin_neg2"):
            assert ta.get(key) == tb.get(key)
    for key in ("Z_samples", "F", "sigma"):
        np.testing.assert_array_equal(a["refit"][key], b["refit"][key])
    assert a["refit"]["w0"] == b["refit"]["w0"] and a["refit"]["w"] == b["refit"]["w"]


def test_full_mask_equals_unmasked(rep1):
    ds, _ = rep1
    full = np.ones((af2.N, af2.N), dtype=bool)
    np.fill_diagonal(full, False)
    _same(_hybrid(ds, None), _hybrid(ds, full))


def test_masked_out_y_cannot_change_anything(rep1):
    ds, masks = rep1
    train = masks["sparse"]
    Y2 = ds.Y.copy()
    hidden = ~train
    np.fill_diagonal(hidden, False)
    Y2[hidden] = 1.0 - Y2[hidden]
    _same(_hybrid(ds, train), _hybrid(ds, train, Y=Y2))
    c, d = _hybrid(ds, None), _hybrid(ds, None, Y=Y2)
    assert not np.array_equal(c["refit"]["Z_samples"], d["refit"]["Z_samples"])


def test_both_stages_receive_the_mask(rep1, monkeypatch):
    ds, masks = rep1
    seen = {}
    orig_cls = fsm.FamilySelectingGenericLSM

    class Spy(orig_cls):
        def __init__(self, **kw):
            seen["exploration"] = np.asarray(kw.get("train_mask"), bool).copy()
            super().__init__(**kw)

    orig_em = fsm.run_em_generic

    def spy_em(*a, **kw):
        seen["refit"] = np.asarray(kw.get("train_mask"), bool).copy()
        return orig_em(*a, **kw)

    monkeypatch.setattr(fsm, "FamilySelectingGenericLSM", Spy)
    monkeypatch.setattr(fsm, "run_em_generic", spy_em)
    res = _hybrid(ds, masks["sparse"])
    np.testing.assert_array_equal(seen["exploration"], masks["sparse"])
    np.testing.assert_array_equal(seen["refit"], masks["sparse"])
    np.testing.assert_array_equal(res["refit"]["model"].train_mask, masks["sparse"])


def test_reproducible_and_auto_equals_oracle_on_same_models(rep1):
    ds, masks = rep1
    a, b = _hybrid(ds, masks["sparse"], refit_seed=555), _hybrid(ds, masks["sparse"], refit_seed=555)
    _same(a, b)
    reg, _ = af2.registry_and_priority()
    fixed = run_em_generic(ds.X, ds.Y, column_specs=[reg[c] for c in a["exploration"]["selected_models"]],
                           family_y="bernoulli", k=3, L=5, num_iter=ITERS, seed=555,
                           newton_alpha=0.5, train_mask=masks["sparse"])
    for key in ("Z_samples", "F", "sigma"):
        np.testing.assert_array_equal(a["refit"][key], fixed[key])


def test_joint_mode_evaluator_is_valid_for_the_generic_cmp_model(rep1):
    ds, masks = rep1
    reg, _ = af2.registry_and_priority()
    res = run_em_generic(ds.X, ds.Y, column_specs=[reg[c] for c in af2.TRUE_MODELS],
                         family_y="bernoulli", k=3, L=5, num_iter=ITERS, seed=7,
                         newton_alpha=0.5, train_mask=masks["sparse"])
    model = res["model"]
    lk._check_supported(model, ds.X, ds.Y)
    Z = res["Z_samples"][:, :, -1].copy()
    g = lk.gradient(model, ds.X, ds.Y, Z)
    H = lk.negative_hessian(model, ds.X, ds.Y, Z)
    rng = np.random.default_rng(0)
    h = 1e-5
    for _ in range(12):
        i, k = int(rng.integers(0, af2.N)), int(rng.integers(0, 3))
        E = np.zeros_like(Z)
        E[i, k] = h
        fd = (lk.complete_log_density(model, ds.X, ds.Y, Z + E)
              - lk.complete_log_density(model, ds.X, ds.Y, Z - E)) / (2 * h)
        assert fd == pytest.approx(g[i, k], rel=1e-5, abs=1e-5)
        gd = (lk.gradient(model, ds.X, ds.Y, Z + E) - lk.gradient(model, ds.X, ds.Y, Z - E)) / (2 * h)
        np.testing.assert_allclose(-gd.ravel(), H[:, i * 3 + k], rtol=1e-4, atol=1e-4)


def test_part_a_orderings_on_01_data():
    reg, _ = af2.registry_and_priority()
    from family_specs import BernoulliSpec
    eta = np.linspace(-8, 4, 241)
    pois, cmp2 = reg["poisson"], reg["cmp_nu2p0"]
    bern = BernoulliSpec()
    for x in (0.0, 1.0):
        xv = np.full_like(eta, x)
        lp, lc, lb = pois.log_prob(xv, eta), cmp2.log_prob(xv, eta), bern.log_prob(xv, eta)
        assert np.all(lb - lp > 0) and np.all(lc - lp > 0)
        np.testing.assert_allclose(lb - lp, np.exp(eta) - np.log1p(np.exp(eta)), rtol=1e-9, atol=1e-12)
    # on counts >= 2 the CMP2-vs-Poisson ordering changes sign (no pointwise dominance)
    d2 = cmp2.log_prob(np.full_like(eta, 3.0), eta) - pois.log_prob(np.full_like(eta, 3.0), eta)
    assert np.any(d2 > 0) and np.any(d2 < 0)


def test_design_contracts(rep1):
    plan = af2.planned_fits()
    assert len(plan) == af2.PLANNED_FITS == 200
    assert len({af2.fit_key(p) for p in plan}) == 200
    for p in plan:
        assert p["model_seed"] == af2.MODEL_SEED_BASE + p["r"]
        assert p["data_seed"] == af2.DATA_SEEDS[p["r"]]
    assert [af2.SWAP[m] for m in af2.TRUE_MODELS][3:] == ["cmp_nu2p0"] * 3 + ["poisson"] * 3
    ds, masks = rep1
    reg, prio = af2.registry_and_priority()
    gates = fsm.gate_columns(ds.X, reg, prio)
    assert [g.admissible[0] for g in gates][3:] == ["poisson"] * 6
    assert [g.decided_by for g in gates] == ["gate"] * 3 + ["score"] * 6
    assert not np.any(np.triu(masks["sparse"] & masks["test"], 1))
    assert af2.protocol()["planned_fits"] == 200
