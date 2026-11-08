"""Tests for AF1 (Oracle / Auto / Wrong on M1 data) and the opt-in
``train_mask`` pass-through of ``family_selection``.

What is protected (research contracts, not incidental details):
  * mask=None and an explicit all-pairs mask give the same selector result
    (the pass-through does not change the historical selector);
  * Y values outside the train mask cannot influence exploration, candidate
    scores, the selected families or the refit (no test-Y leakage);
  * exploration and refit see the same train mask;
  * M1 train/test pairs are disjoint and all methods share data and masks;
  * same settings + seeds reproduce bit-for-bit;
  * an Auto refit that selected the Oracle families equals the Oracle fit.

EM is run with reduced iteration counts on the real M1 replicate 1 so that
the tests stay fast; the contracts do not depend on the iteration count.
"""

from __future__ import annotations

import numpy as np
import pytest

import family_selection as fs
import run_auto_family_z_recovery as af
import run_misspecification_m1 as m1
from em_runner import run_em_experimental

ITERS = 2


@pytest.fixture(scope="module")
def rep1():
    return m1.generate(m1.DATA_SEEDS[0]), m1.build_masks(0)


def _hybrid(ds, mask, Y=None, start="poisson", seed=11, refit_seed=12):
    return fs.run_hybrid_family_selection(
        ds.X, ds.Y if Y is None else Y, k=3, ambiguous_start=start, L=5,
        exploration_num_iter=ITERS, refit_num_iter=ITERS,
        search_seed=seed, refit_seed=refit_seed, train_mask=mask)


def _same(a, b):
    assert a["selected_assignment"] == b["selected_assignment"]
    assert a["margins"] == b["margins"]
    for ta, tb in zip(a["selection_trace"], b["selection_trace"], strict=True):
        for key in ("score_bernoulli", "score_poisson", "selected_family", "margin_neg2"):
            assert ta[key] == tb[key]
    for key in ("Z_samples", "F", "sigma"):
        np.testing.assert_array_equal(a["refit"][key], b["refit"][key])
    assert a["refit"]["w0"] == b["refit"]["w0"] and a["refit"]["w"] == b["refit"]["w"]


# ── mask pass-through ─────────────────────────────────────────────────────

def test_full_mask_equals_historical_unmasked_selector(rep1):
    ds, _ = rep1
    full = np.ones((m1.N, m1.N), dtype=bool)
    np.fill_diagonal(full, False)
    _same(_hybrid(ds, None), _hybrid(ds, full))


def test_masked_out_y_cannot_change_anything(rep1):
    ds, masks = rep1
    train = masks["sparse"]
    Y2 = ds.Y.copy()
    hidden = ~train
    np.fill_diagonal(hidden, False)
    Y2[hidden] = 1.0 - Y2[hidden]              # flip every unseen pair (incl. test)
    assert np.array_equal(Y2, Y2.T)
    a, b = _hybrid(ds, train), _hybrid(ds, train, Y=Y2)
    _same(a, b)                                # candidate scores included
    # and the mask really matters: without it the flipped Y changes the result
    c, d = _hybrid(ds, None), _hybrid(ds, None, Y=Y2)
    assert not np.array_equal(c["refit"]["Z_samples"], d["refit"]["Z_samples"])


def test_exploration_and_refit_receive_the_same_mask(rep1, monkeypatch):
    ds, masks = rep1
    seen = {}
    orig_cls = fs.FamilySelectingPerColumnLSM

    class Spy(orig_cls):
        def __init__(self, **kw):
            seen["exploration"] = np.asarray(kw.get("train_mask"), bool).copy()
            super().__init__(**kw)

    orig_em = fs.run_em_experimental

    def spy_em(*a, **kw):
        seen["refit"] = np.asarray(kw.get("train_mask"), bool).copy()
        return orig_em(*a, **kw)

    monkeypatch.setattr(fs, "FamilySelectingPerColumnLSM", Spy)
    monkeypatch.setattr(fs, "run_em_experimental", spy_em)
    res = _hybrid(ds, masks["sparse"])
    np.testing.assert_array_equal(seen["exploration"], masks["sparse"])
    np.testing.assert_array_equal(seen["refit"], masks["sparse"])
    np.testing.assert_array_equal(res["refit"]["model"].train_mask, masks["sparse"])


def test_selector_is_reproducible(rep1):
    ds, masks = rep1
    _same(_hybrid(ds, masks["sparse"]), _hybrid(ds, masks["sparse"]))


def test_auto_refit_on_oracle_families_equals_oracle_fit(rep1):
    ds, masks = rep1
    train = masks["sparse"]
    hyb = _hybrid(ds, train, refit_seed=777)
    oracle = run_em_experimental(
        ds.X, ds.Y, family_x=None, family_y="bernoulli", k=3, L=5, num_iter=ITERS,
        seed=777, train_mask=train, family_x_list=hyb["selected_assignment"],
        compute_strict_Q=False, validate_support=True, allow_support_mismatch=False,
        numerics_mode="consistent", failure_policy="fail_fast")
    for key in ("Z_samples", "F", "sigma"):
        np.testing.assert_array_equal(hyb["refit"][key], oracle[key])
    assert hyb["refit"]["w0"] == oracle["w0"] and hyb["refit"]["w"] == oracle["w"]


# ── candidate score: independent check of the complete log-likelihoods ────

def test_candidate_score_is_the_complete_loglik():
    from scipy.stats import bernoulli, poisson
    rng = np.random.default_rng(0)
    Zs = rng.normal(size=(30, 3, 5))
    f = np.array([0.4, -0.3, 0.2])
    x = rng.integers(0, 2, size=30).astype(float)
    eta = np.einsum("nks,k->ns", Zs, f)
    exp_b = np.mean([bernoulli.logpmf(x, 1 / (1 + np.exp(-eta[:, s]))).sum() for s in range(5)])
    exp_p = np.mean([poisson.logpmf(x, np.exp(eta[:, s])).sum() for s in range(5)])
    got_b = fs.column_log_likelihood(x, Zs, f, "bernoulli")
    got_p = fs.column_log_likelihood(x, Zs, f, "poisson")
    got_b = got_b[0] if isinstance(got_b, tuple) else got_b
    got_p = got_p[0] if isinstance(got_p, tuple) else got_p
    assert got_b == pytest.approx(exp_b, rel=1e-10)
    assert got_p == pytest.approx(exp_p, rel=1e-10)


# ── AF1 design contracts (no EM) ──────────────────────────────────────────

def test_plan_shape_and_seeds():
    plan = af.planned_fits()
    assert len(plan) == af.PLANNED_FITS == 320
    assert len({af.fit_key(p) for p in plan}) == 320
    for p in plan:
        assert p["model_seed"] == af.SERIES[p["series"]] + p["r"]
        assert p["data_seed"] == m1.DATA_SEEDS[p["r"]]
        if p["method"].startswith("auto"):
            assert p["search_seed"] == af.SEARCH_SEED_BASE[p["series"]] + p["r"]
    # series seeds are E1's
    assert af.SERIES == {"A": 1343000, "B": 1344000}
    assert af.FIXED_FAMILIES["oracle"] == list(m1.FAMILY_X_TRUE)
    assert af.FIXED_FAMILIES["wrong"][3:9] == ["poisson"] * 6


def test_masks_disjoint_and_shared_for_all_replicates():
    for r in range(len(m1.DATA_SEEDS)):
        masks = m1.build_masks(r)
        for regime in ("dense", "sparse"):
            assert not np.any(np.triu(masks[regime] & masks["test"], 1))
            assert np.array_equal(masks[regime], masks[regime].T)
        assert np.all(masks["dense"][masks["sparse"]])
        # rebuilding gives the same masks (every method of a cell uses build_masks(r))
        again = m1.build_masks(r)
        for k in masks:
            assert np.array_equal(masks[k], again[k])


def test_protocol_is_json_stable():
    assert af.protocol_sha256() == af.protocol_sha256()
    assert af.protocol()["planned_fits"] == 320
