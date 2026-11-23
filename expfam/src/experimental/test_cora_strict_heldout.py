"""Technical tests for the CORA-SH1 runner (``cora_strict_heldout.py``).

Run:  .venv/Scripts/python.exe -m pytest expfam/src/experimental/test_cora_strict_heldout.py -q

Scope (limited to what the scientific comparison needs):
  masks / selection / preprocessing use no test information; masked pairs
  contribute exactly nothing to the Y log-likelihood, E-step gradient and
  precision; objective and score agree (finite differences) with no extra 1/2;
  the fast precision equals the original; arms differ only in the X model;
  Y-only carries no X information; no hidden retry; failures are recorded;
  the fit is invariant to the values written at test pairs (two canaries).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
_ROOT = _SRC.parent.parent
for _p in (_HERE, _SRC, _ROOT / "reproduction" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cora_strict_heldout as csh                                  # noqa: E402
import em_runner                                                   # noqa: E402
import laplace_k_criterion as lk                                   # noqa: E402
from model_dual_expfam import DualExpFamLSM                        # noqa: E402

SMOKE = csh.CONFIG["smoke_split_index"]


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cora():
    return csh.load_cora()


@pytest.fixture(scope="module")
def smoke_split(cora):
    return csh.build_split(cora, SMOKE)


def _synthetic_split(seed=0, n=36, d=8, k=2, test_fraction=0.2):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    eta = -1.5 + 1.2 * Z @ Z.T
    Y = (rng.random((n, n)) < 1 / (1 + np.exp(-eta))).astype(float)
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    X = (rng.random((n, d)) < 0.3).astype(float)
    a, b = np.triu_indices(n, 1)
    t = rng.random(len(a)) < test_fraction
    test = np.zeros((n, n), bool)
    test[a[t], b[t]] = True
    test |= test.T
    train = ~test
    np.fill_diagonal(train, False)
    csh.validate_masks(train, test)
    summary = {"train_mask_hash": csh.array_hash(train),
               "test_mask_hash": csh.array_hash(test)}
    return csh.SplitData(-1, seed, np.arange(n), np.arange(d), X,
                         X - X.mean(0, keepdims=True), Y,
                         np.where(test, 0.0, Y), train, test, summary)


def _model(cls, n, d, k, family_x, mask, seed=1):
    m = cls(n=n, d=d, k=k, L=2, family_x=family_x, family_y="bernoulli",
            train_mask=mask)
    m.initialize_params(seed=seed)
    return m


def _theta(seed, n, d, k, family_x):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    F = rng.standard_normal((d, k)) * 0.5
    sigma = np.diag(rng.uniform(0.3, 1.5, d)) if family_x == "gaussian" else np.eye(d)
    return Z, F, sigma, 1.0, -1.0, 0.7


# ── data, selection, masks ─────────────────────────────────────────────────

def test_parse_matches_historical(cora):
    import run_fixed_real_cora_heldout_link_prediction as hist
    node_ids, X_full, label_idx, label_set, edges, degree = hist.parse_cora()
    assert np.array_equal(cora.node_ids, node_ids)
    assert np.array_equal(cora.X_full, X_full.astype(np.float64))
    assert np.array_equal(cora.labels, label_idx)
    assert tuple(label_set) == cora.label_set
    assert set(map(tuple, cora.edges.tolist())) == edges
    assert cora.edges.shape == (5278, 2)


def test_pair_index_is_a_bijection():
    n = 9
    a, b = np.triu_indices(n, 1)
    idx = csh.pair_index(a, b, n)
    assert np.array_equal(np.sort(idx), np.arange(n * (n - 1) // 2))


def test_dyad_split_exact_and_deterministic():
    s1 = csh.make_dyad_split(100, 5, 0.2)
    s2 = csh.make_dyad_split(100, 5, 0.2)
    assert np.array_equal(s1, s2)
    assert s1.sum() == round(0.2 * 4950)


@pytest.mark.parametrize("selection", csh.SELECTIONS)
def test_nothing_the_fit_sees_depends_on_test_labels(cora, selection):
    """Remove every test-pair edge from the full graph: the node set, the
    vocabulary, X, the masks and Y_fit must be identical (only Y_true differs)."""
    ref = csh.build_split(cora, SMOKE, selection=selection)
    n_full = len(cora.node_ids)
    is_test = csh.make_dyad_split(n_full, ref.split_seed, csh.CONFIG["test_fraction"])
    pidx = csh.pair_index(cora.edges[:, 0], cora.edges[:, 1], n_full)
    pruned = replace(cora, edges=cora.edges[~is_test[pidx]])
    alt = csh.build_split(pruned, SMOKE, selection=selection)
    for field in ("selected", "vocab", "X_raw", "X_centered", "train_mask",
                  "test_mask", "Y_fit"):
        assert np.array_equal(getattr(alt, field), getattr(ref, field)), field
    assert ref.Y_true[ref.test_mask].sum() > 0
    assert alt.Y_true[alt.test_mask].sum() == 0


def test_primary_selection_is_label_free(cora, smoke_split):
    """Primary 'full' uses every node; 'random_per_class' ignores Y entirely
    (same nodes when all edges are deleted)."""
    assert csh.CONFIG["selection"] == "full"
    assert np.array_equal(smoke_split.selected, np.arange(len(cora.node_ids)))
    a = csh.build_split(cora, SMOKE, selection="random_per_class")
    b = csh.build_split(replace(cora, edges=cora.edges[:0]), SMOKE,
                        selection="random_per_class")
    assert np.array_equal(a.selected, b.selected) and len(a.selected) == 700
    assert np.all(np.bincount(cora.labels[a.selected]) == 100)


def test_train_degree_selection_uses_train_edges_only(cora):
    sp = csh.build_split(cora, SMOKE, selection="train_degree")
    n_full = len(cora.node_ids)
    is_test = csh.make_dyad_split(n_full, sp.split_seed, 0.2)
    deg_tr = csh.train_degree(cora.edges, is_test, n_full)
    assert np.array_equal(csh.select_nodes(cora.labels, deg_tr, 100), sp.selected)
    assert len(np.unique(sp.selected)) == 700


def test_masks_partition_subset_pairs(smoke_split):
    s = smoke_split
    csh.validate_masks(s.train_mask, s.test_mask)
    assert np.all(s.Y_fit[s.test_mask] == 0)
    assert np.array_equal(s.Y_fit[s.train_mask], s.Y_true[s.train_mask])
    assert np.array_equal(s.Y_true, s.Y_true.T) and not np.any(np.diag(s.Y_true))
    frac = s.summary["n_test_pairs"] / s.summary["n_pairs"]
    assert 0.19 < frac < 0.21


def test_validate_masks_rejects_bad_masks():
    tr = np.ones((4, 4), bool)
    np.fill_diagonal(tr, False)
    te = np.zeros((4, 4), bool)
    csh.validate_masks(tr, te)
    bad = te.copy()
    bad[0, 1] = bad[1, 0] = True
    with pytest.raises(ValueError):
        csh.validate_masks(tr, bad)                 # overlap
    tr2 = tr.copy()
    tr2[0, 1] = False
    with pytest.raises(ValueError):
        csh.validate_masks(tr2, te)                 # asymmetric / not covering


# ── masked pairs contribute nothing ────────────────────────────────────────

@pytest.mark.parametrize("family_x", ["bernoulli", "gaussian"])
def test_masked_pairs_contribute_zero(family_x):
    sp = _synthetic_split(seed=3)
    n, d = sp.X_raw.shape
    k = 2
    m = _model(csh.CoraConsistentFast, n, d, k, family_x, sp.train_mask)
    Z, F, sigma, var_z, w0, w = _theta(4, n, d, k, family_x)
    Y0 = sp.Y_fit
    Y1 = np.where(sp.test_mask, 1.0, sp.Y_fit)
    ll0 = m.calc_log_likelihood_Y(Y0, Z[:, :, None], w0, w)
    ll1 = m.calc_log_likelihood_Y(Y1, Z[:, :, None], w0, w)
    assert ll0 == ll1
    for i in range(n):
        g0 = m._calc_gradient(sp.X_raw, Y0, Z, F, sigma, var_z, w0, w, i)
        g1 = m._calc_gradient(sp.X_raw, Y1, Z, F, sigma, var_z, w0, w, i)
        assert np.array_equal(g0, g1)
    # precision: Y term equals the explicit sum over TRAIN partners only
    for i in range(n):
        P = m._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)
        P_noY = m._calc_precision_matrix(Z, F, sigma, var_z, w0, 0.0, i)
        eta = w0 + w * Z @ Z[i]
        v = 1 / (1 + np.exp(-eta))
        v = v * (1 - v)
        js = np.where(sp.train_mask[i])[0]
        manual = w ** 2 * sum(v[j] * np.outer(Z[j], Z[j]) for j in js)
        assert np.allclose(P - P_noY, manual, rtol=1e-12, atol=1e-12)


# ── objective / score / curvature consistency, no extra 1/2 ────────────────

def _independent_objective(X, Y, mask, Z, F, sigma, w0, w, family_x):
    """Written from the model definition, independent of the classes:
    prior + sum_{i,l} log p(x_il) + sum_{i<j, (i,j) in O} log p(y_ij)."""
    val = -0.5 * np.sum(Z ** 2)
    ex = Z @ F.T
    if family_x == "bernoulli":
        val += np.sum(X * ex - np.logaddexp(0, ex))
    else:
        s2 = np.diag(sigma)[None, :]
        val += np.sum(-0.5 * (X - ex) ** 2 / s2)
    n = len(Z)
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j]:
                e = w0 + w * Z[i] @ Z[j]
                val += Y[i, j] * e - np.logaddexp(0, e)
    return val


@pytest.mark.parametrize("family_x", ["bernoulli", "gaussian"])
def test_estep_gradient_matches_objective_no_extra_half(family_x):
    sp = _synthetic_split(seed=5, n=14, d=5)
    n, d = sp.X_raw.shape
    k = 2
    m = _model(csh.CoraConsistentFast, n, d, k, family_x, sp.train_mask)
    Z, F, sigma, var_z, w0, w = _theta(6, n, d, k, family_x)
    h = 1e-6
    for i in (0, 5, n - 1):
        g = -m._calc_gradient(sp.X_raw, sp.Y_fit, Z, F, sigma, var_z, w0, w, i)
        fd = np.zeros(k)
        for a in range(k):
            Zp, Zm = Z.copy(), Z.copy()
            Zp[i, a] += h
            Zm[i, a] -= h
            fd[a] = (_independent_objective(sp.X_raw, sp.Y_fit, sp.train_mask, Zp,
                                            F, sigma, w0, w, family_x)
                     - _independent_objective(sp.X_raw, sp.Y_fit, sp.train_mask, Zm,
                                              F, sigma, w0, w, family_x)) / (2 * h)
        assert np.allclose(g, fd, rtol=1e-6, atol=1e-6), (g, fd)
        # curvature: E-step precision == -d(gradient)/dz_i (finite difference)
        P = m._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)
        H = np.zeros((k, k))
        for a in range(k):
            Zp, Zm = Z.copy(), Z.copy()
            Zp[i, a] += h
            Zm[i, a] -= h
            H[:, a] = (m._calc_gradient(sp.X_raw, sp.Y_fit, Zp, F, sigma, var_z, w0, w, i)
                       - m._calc_gradient(sp.X_raw, sp.Y_fit, Zm, F, sigma, var_z, w0, w, i)
                       ) / (2 * h)
        assert np.allclose(P, H, rtol=1e-5, atol=1e-6), (P, H)


def test_negative_control_old_half_factor_is_detected():
    """The legacy 0.5 class must FAIL the same check (the test can see 1/2)."""
    sp = _synthetic_split(seed=5, n=14, d=5)
    n, d = sp.X_raw.shape
    k = 2
    full = ~np.eye(n, dtype=bool)
    old = DualExpFamLSM(n=n, d=d, k=k, L=2, family_x="bernoulli", family_y="bernoulli")
    Z, F, sigma, var_z, w0, w = _theta(6, n, d, k, "bernoulli")
    i, h = 3, 1e-6
    g = -old._calc_gradient(sp.X_raw, sp.Y_true, Z, F, sigma, var_z, w0, w, i)
    fd = np.zeros(k)
    for a in range(k):
        Zp, Zm = Z.copy(), Z.copy()
        Zp[i, a] += h
        Zm[i, a] -= h
        fd[a] = (_independent_objective(sp.X_raw, sp.Y_true, full, Zp, F, sigma, w0, w,
                                        "bernoulli")
                 - _independent_objective(sp.X_raw, sp.Y_true, full, Zm, F, sigma, w0, w,
                                          "bernoulli")) / (2 * h)
    assert not np.allclose(g, fd, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("family_x", ["bernoulli", "gaussian"])
def test_joint_mode_objective_gradient_hessian_consistent(family_x):
    sp = _synthetic_split(seed=7, n=12, d=4)
    n, d = sp.X_raw.shape
    k = 2
    m = _model(csh.CoraConsistentFast, n, d, k, family_x, sp.train_mask)
    Z, F, sigma, var_z, w0, w = _theta(8, n, d, k, family_x)
    m.params.update(F=F, sigma=sigma, w0=w0, w=w, var_z=var_z)
    phi = lambda ZZ: lk.complete_log_density(m, sp.X_raw, sp.Y_fit, ZZ)  # noqa: E731
    # objective equals the independent objective up to Z-free constants
    c1 = phi(Z) - _independent_objective(sp.X_raw, sp.Y_fit, sp.train_mask, Z, F,
                                         sigma, w0, w, family_x)
    Z2 = Z + 0.1
    c2 = phi(Z2) - _independent_objective(sp.X_raw, sp.Y_fit, sp.train_mask, Z2, F,
                                          sigma, w0, w, family_x)
    assert abs(c1 - c2) < 1e-9
    g = lk.gradient(m, sp.X_raw, sp.Y_fit, Z)
    h = 1e-6
    for i, a in ((0, 0), (4, 1), (n - 1, 0)):
        Zp, Zm = Z.copy(), Z.copy()
        Zp[i, a] += h
        Zm[i, a] -= h
        assert abs((phi(Zp) - phi(Zm)) / (2 * h) - g[i, a]) < 1e-5
    H = lk.negative_hessian(m, sp.X_raw, sp.Y_fit, Z)
    for i in range(n):
        block = H[i * k:(i + 1) * k, i * k:(i + 1) * k]
        P = m._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)
        assert np.allclose(block, P, rtol=1e-12, atol=1e-12)


# ── fast precision == original ─────────────────────────────────────────────

@pytest.mark.parametrize("family_x", ["bernoulli", "gaussian"])
def test_fast_precision_equals_original(family_x):
    sp = _synthetic_split(seed=9, n=50, d=6, k=3)
    n, d = sp.X_raw.shape
    k = 4
    fast = _model(csh.CoraConsistentFast, n, d, k, family_x, sp.train_mask)
    orig = _model(csh.CoraConsistentCounting, n, d, k, family_x, sp.train_mask)
    Z, F, sigma, var_z, w0, w = _theta(10, n, d, k, family_x)
    for i in range(n):
        a = fast._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)
        b = orig._calc_precision_matrix(Z, F, sigma, var_z, w0, w, i)
        assert np.allclose(a, b, rtol=1e-13, atol=1e-13)


def test_fast_fit_equals_original_fit():
    sp = _synthetic_split(seed=11, n=40, d=6)
    out = {}
    for kind in ("fast", "original"):
        tech, _, st = csh.run_cell(sp, "BX", 3, 101, 202, model_kind=kind,
                                   num_iter=4, compute_mode=False)
        assert tech["fit_status"] == "OK"
        out[kind] = st
    assert np.allclose(out["fast"]["Z_samples"], out["original"]["Z_samples"],
                       rtol=0, atol=1e-9)
    assert abs(float(out["fast"]["w"]) - float(out["original"]["w"])) < 1e-10


# ── arms, Y-only, retries, failures, canaries ──────────────────────────────

def test_arms_differ_only_in_x_model(monkeypatch):
    sp = _synthetic_split(seed=12)
    calls = {}

    def fake(X, Y, family_x, family_y, **kw):
        calls[(family_x, kw["fix_x"], float(X.mean()))] = dict(kw, family_y=family_y,
                                                               Y=Y, X=X)
        raise RuntimeError("stop")

    monkeypatch.setattr(em_runner, "run_em_experimental", fake)
    for arm in ("BX", "GX", "YO", "GXc"):
        csh.run_cell(sp, arm, 3, 1, 2)
    kws = list(calls.values())
    assert len(kws) == 4
    ref = kws[0]
    for kw in kws[1:]:
        for key in ("k", "L", "num_iter", "seed", "init_seed", "mc_seed",
                    "numerics_mode", "failure_policy", "family_y",
                    "compute_strict_Q", "mstep_q_diagnostic"):
            assert kw[key] == ref[key], key
        assert kw["train_mask"] is ref["train_mask"]
        assert kw["Y"] is ref["Y"]
    fams = {(a["family_x"], a["fix_x"], a["x_variant"]) for a in csh.ARMS.values()}
    assert fams == {("bernoulli", False, "raw"), ("gaussian", False, "raw"),
                    ("bernoulli", True, "raw"), ("gaussian", False, "centered")}


def test_y_only_has_no_x_contribution():
    sp = _synthetic_split(seed=13)
    tech, _, st = csh.run_cell(sp, "YO", 2, 5, 6, num_iter=3, compute_mode=False)
    assert tech["fit_status"] == "OK"
    assert np.all(st["F"] == 0.0)
    n, d = sp.X_raw.shape
    m = _model(csh.CoraConsistentFast, n, d, 2, "bernoulli", sp.train_mask)
    Z, _, sigma, var_z, w0, w = _theta(14, n, d, 2, "bernoulli")
    F0 = np.zeros((d, 2))
    Xr = np.random.default_rng(0).integers(0, 2, (n, d)).astype(float)
    for i in range(n):
        assert np.array_equal(
            m._calc_gradient(sp.X_raw, sp.Y_fit, Z, F0, sigma, var_z, w0, w, i),
            m._calc_gradient(Xr, sp.Y_fit, Z, F0, sigma, var_z, w0, w, i))
    m.params.update(F=F0, sigma=sigma, w0=w0, w=w, var_z=var_z)
    assert np.array_equal(lk.gradient(m, sp.X_raw, sp.Y_fit, Z),
                          lk.gradient(m, Xr, sp.Y_fit, Z))


def test_no_hidden_retry_and_counters():
    sp = _synthetic_split(seed=15)
    tech, scores, _ = csh.run_cell(sp, "GX", 2, 7, 8, num_iter=3)
    assert tech["fit_status"] == "OK"
    assert tech["failure_policy"] == "fail_fast"
    assert tech["retry_count"] == tech["replacement_count"] == tech["seed_rescue_count"] == 0
    assert tech["estep_sweeps"] == 3 * csh.CONFIG["L"]
    assert tech["sampling_draws"] + tech["sampling_fallbacks"] == 3 * csh.CONFIG["L"] * 36
    assert {r["estimator"] for r in scores} == {"last", "mean5", "mc_predictive"}
    tech2, scores2, _ = csh.run_cell(sp, "GX", 2, 7, 8, num_iter=3, compute_mode=True)
    assert {r["estimator"] for r in scores2} == {"last", "mean5", "mode", "mc_predictive"}
    assert tech2["mode_status"] in ("OK", "NOT_STATIONARY", "HESSIAN_NOT_PD")


def test_mc_predictive_matches_definition():
    sp = _synthetic_split(seed=18)
    n = sp.X_raw.shape[0]
    rng = np.random.default_rng(1)
    Zs = rng.standard_normal((n, 2, 5))
    w0, w = -1.2, 0.8
    got = csh.heldout_scores_mc_predictive(Zs, w0, w, sp)
    rows, cols, y = csh.heldout_pair_arrays(sp)
    p = np.mean([1 / (1 + np.exp(-(w0 + w * np.sum(Zs[rows, :, l] * Zs[cols, :, l], 1))))
                 for l in range(5)], axis=0)
    naive = np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert abs(got["log_score"] - naive) < 1e-12
    one = csh.heldout_scores_mc_predictive(Zs[:, :, :1], w0, w, sp)
    plug = csh.heldout_scores(Zs[:, :, 0], w0, w, sp)
    assert abs(one["log_score"] - plug["log_score"]) < 1e-12


def test_nonfinite_fit_is_recorded_not_rescued():
    sp = _synthetic_split(seed=16)
    bad = replace(sp, X_raw=np.full_like(sp.X_raw, np.nan))
    tech, scores, st = csh.run_cell(bad, "GX", 2, 9, 10, num_iter=2)
    assert tech["fit_status"].startswith("FAILED:")
    assert scores == [] and st == {}


def test_two_canary_fit_invariance():
    sp = _synthetic_split(seed=17)
    Y1 = np.where(sp.test_mask, 1.0, sp.Y_fit)
    runs = []
    for Y in (sp.Y_fit, Y1):
        tech, scores, st = csh.run_cell(sp, "BX", 2, 11, 12, num_iter=3,
                                        Y_fit_override=Y)
        runs.append((tech, scores, st))
    (t0, s0, st0), (t1, s1, st1) = runs
    assert t0["y_fit_hash"] != t1["y_fit_hash"]
    for key in st0:
        assert np.array_equal(st0[key], st1[key]), key
    for a, b in zip(s0, s1):
        assert a["log_score"] == b["log_score"]


def test_manifest_shape_and_seed_sharing():
    cells = csh.build_manifest()
    assert len(cells) == 90
    assert len({c["cell_id"] for c in cells}) == 90
    groups = {}
    for c in cells:
        groups.setdefault((c["split_index"], c["K"], c["seed_index"]), set()).add(
            (c["init_seed"], c["mc_seed"]))
    assert all(len(v) == 1 for v in groups.values())
    assert len({next(iter(v)) for v in groups.values()}) == 30
    assert all(c["split_index"] != SMOKE for c in cells)
