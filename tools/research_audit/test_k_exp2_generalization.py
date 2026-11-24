"""Unit tests for the K-selection Exp2-generalisation design, runner and analysis.

No main-run data are used.  EM is exercised only on tiny problems or through a
monkeypatched estimator.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in ("expfam/src", "expfam/src/experimental", "reproduction/src", "tools/research_audit"):
    sys.path.insert(0, str(ROOT / _p))

import k_exp2_design as kd  # noqa: E402
import run_k_exp2_generalization as runner  # noqa: E402
import analyze_k_exp2_generalization as an  # noqa: E402

TABLE = kd.bernoulli_w0_table()


# ---------------------------------------------------------------- parameters

@pytest.mark.parametrize("regime", kd.REGIMES)
@pytest.mark.parametrize("k", kd.K_TRUES)
def test_poisson_y_w_below_quarter_and_moments(regime, k):
    w = kd.w_of("B", regime, k)
    assert abs(w) < 0.25                      # 4|w| < 1  => E[Y^4] finite (P6)
    assert 4 * abs(w) < 1


@pytest.mark.parametrize("regime", kd.REGIMES)
@pytest.mark.parametrize("k", kd.K_TRUES)
def test_poisson_y_population_mean_one(regime, k):
    w = kd.w_of("B", regime, k)
    w0 = kd.poisson_y_w0(w, k)
    analytic = math.exp(w0) * (1 - w * w) ** (-k / 2)
    assert abs(analytic - 1.0) < 1e-12
    assert abs(kd.poisson_y_mean_by_quadrature(w0, w, k) - 1.0) < 1e-9


def test_w_rules():
    for k in kd.K_TRUES:
        for s in "OAC":
            assert kd.w_of(s, "a", k) == 1.0
            assert math.isclose(kd.w_of(s, "b", k), math.sqrt(3 / k))
        assert kd.w_of("B", "a", k) == 0.12
        assert math.isclose(kd.w_of("B", "b", k), 0.12 * math.sqrt(3 / k))
        # regime b keeps w^2 K constant
        assert math.isclose(kd.w_of("O", "b", k) ** 2 * k, 3.0)


@pytest.mark.parametrize("regime", kd.REGIMES)
@pytest.mark.parametrize("k", kd.K_TRUES)
def test_bernoulli_calibration_three_routes(regime, k):
    cal = TABLE[(regime, k)]
    assert abs(cal["p_adaptive"] - 0.20) <= kd.CAL_TOL_P_RESID
    assert abs(cal["p_chi2_hermite"] - 0.20) <= kd.CAL_TOL_P_INDEP
    assert cal["abs_w0_diff"] <= kd.CAL_TOL_W0_AGREE


def test_bernoulli_mean_monte_carlo_sanity():
    # independent stochastic route at K=9, regime a (strongest case)
    rng = np.random.default_rng(0)
    w0 = TABLE[("a", 9)]["w0"]
    z1 = rng.standard_normal((400_000, 9)); z2 = rng.standard_normal((400_000, 9))
    p = 1 / (1 + np.exp(-(w0 + np.sum(z1 * z2, 1))))
    assert abs(p.mean() - 0.20) < 5 * p.std() / math.sqrt(len(p))


@pytest.mark.parametrize("regime", kd.REGIMES)
@pytest.mark.parametrize("k", (1, 9))
def test_loadings_rank_and_row_norm(regime, k):
    ds = kd.generate(kd.dataset_spec("O", regime, k, 0, TABLE))
    assert np.linalg.matrix_rank(ds.F) == k
    assert abs(np.mean(np.sum(ds.F ** 2, 1)) - kd.rho_x_of(regime, k)) < 1e-12
    q = ds.F / kd.f_scale_of(regime, k)
    assert np.allclose(q.T @ q, np.eye(k), atol=1e-12)


def test_regimes_share_z_and_q():
    for s in kd.SERIES:
        a = kd.generate(kd.dataset_spec(s, "a", 9, 3, TABLE))
        b = kd.generate(kd.dataset_spec(s, "b", 9, 3, TABLE))
        assert np.array_equal(a.Z, b.Z)
        assert np.allclose(a.F / kd.f_scale_of("a", 9), b.F / kd.f_scale_of("b", 9), atol=1e-12)


def test_generation_is_deterministic_and_k_independent():
    s1 = kd.dataset_spec("A", "a", 5, 2, TABLE)
    d1, d2 = kd.generate(s1), kd.generate(s1)
    assert np.array_equal(d1.X, d2.X) and np.array_equal(d1.Y, d2.Y)
    # the dataset seed does not depend on candidate K or init
    rows = [r for r in kd.build_manifest() if r["dataset_id"] == "A-a-KT5-r2"]
    assert len(rows) == 20 and len({r["data_seed"] for r in rows}) == 1


@pytest.mark.parametrize("series", list(kd.SERIES))
def test_support_and_moments(series):
    fx, fy = kd.SERIES[series]
    ds = kd.generate(kd.dataset_spec(series, "a", 9, 0, TABLE))
    iu = np.triu_indices(kd.N, 1)
    if fy == "bernoulli":
        assert set(np.unique(ds.Y[iu])) <= {0.0, 1.0}
    if fy == "poisson":
        assert (ds.Y >= 0).all() and np.array_equal(ds.Y, np.round(ds.Y))
    if fx == "bernoulli":
        assert set(np.unique(ds.X)) <= {0.0, 1.0}
    if fx == "poisson":
        assert (ds.X >= 0).all() and np.array_equal(ds.X, np.round(ds.X))
    assert np.allclose(ds.Y, ds.Y.T) and np.all(np.diag(ds.Y) == 0)


def test_poisson_x_canonical_exp_link_mean():
    # E[X_l] = exp(||f_l||^2 / 2) under canonical Poisson-X; large-n check
    # same construction as generate_canonical_data (QR loadings, exp link, no clip);
    # X only, because the n x n Y would not fit in memory at this n
    rng = np.random.default_rng(11)
    from data_generator_canonical import build_full_rank_loadings
    F = build_full_rank_loadings(rng, d=15, k=9, f_scale=kd.f_scale_of("a", 9))
    Z = rng.standard_normal((200_000, 9))
    X = rng.poisson(np.exp(Z @ F.T))
    expect = np.exp(np.sum(F ** 2, 1) / 2)
    se = np.sqrt(X.var(0) / len(X))
    assert np.all(np.abs(X.mean(0) - expect) < 6 * se)


def test_gaussian_noise_variances():
    ds = kd.generate(kd.dataset_spec("C", "a", 1, 0, TABLE))
    assert ds.metadata["sigma_y_sd"] == 1.0
    ds2 = kd.generate(kd.dataset_spec("O", "a", 1, 0, TABLE))
    assert ds2.metadata["sigma_x_var"] == [0.1] * kd.D


# ---------------------------------------------------------------- manifest/seeds

def test_manifest_size_and_unique_seeds():
    rows = kd.build_manifest()
    assert len(rows) == 8000
    assert len({r["fit_id"] for r in rows}) == 8000
    per_ds = {}
    for r in rows:
        per_ds.setdefault((r["series"], r["k_true"], r["rep"]), set()).add(r["data_seed"])
    assert all(len(v) == 1 for v in per_ds.values())
    assert len({next(iter(v)) for v in per_ds.values()}) == len(per_ds)
    inits = {(r["series"], r["k_true"], r["rep"], r["t"]): (r["init_seed"], r["mc_seed"]) for r in rows}
    flat = [s for p in inits.values() for s in p]
    assert len(set(flat)) == len(flat)
    assert not set(flat) & {next(iter(v)) for v in per_ds.values()}


# ---------------------------------------------------------------- runner

def _fake_result(k, q=-1000.0):
    return {"Q_strict": q, "bic": -2 * q + (k * 15 - k * (k - 1) // 2 + 15) * math.log(150),
            "num_params": k * 15 - k * (k - 1) // 2 + 15, "q_bic_failed": False,
            "F": np.zeros((15, k)), "sigma": np.ones(15), "w0": 0.1, "w": 0.2, "var_z": 1.0,
            "sigma_y_est": 1.0, "retry_count": 0, "replacement_count": 0,
            "seed_rescue_count": 0, "nan_count": 0, "failure_reason": None,
            "Z_est": np.zeros((150, k)), "Z_samples": np.zeros((150, k, 2)),
            "mstep_q_history": [{"iteration": 1, "q_before": -2.0, "q_after": -1.5,
                                 "q_diff": 0.5, "decreased": False},
                                {"iteration": 2, "q_before": -1.5, "q_after": -1.4,
                                 "q_diff": 0.1, "decreased": False}]}


def test_runner_passes_only_observed_data(monkeypatch, tmp_path):
    import em_runner
    seen = {}

    def fake(X, Y, fx, fy, k, **kw):
        seen["args"] = (X, Y); seen["kw"] = kw
        return _fake_result(k)
    monkeypatch.setattr(em_runner, "run_em_experimental", fake)
    job = [r for r in kd.build_manifest() if r["fit_id"] == "O-a-KT9-r0-k10-t1"][0]
    row = runner.fit_one(job, TABLE, str(tmp_path))
    truth = kd.generate(kd.dataset_spec("O", "a", 9, 0, TABLE))
    assert row["status"] == "OK"
    assert np.array_equal(seen["args"][0], truth.X) and np.array_equal(seen["args"][1], truth.Y)
    for v in seen["kw"].values():          # no array-valued truth leaks via kwargs
        assert not isinstance(v, np.ndarray)
    assert seen["kw"]["failure_policy"] == "fail_fast"
    assert seen["kw"]["numerics_mode"] == "consistent"
    assert seen["kw"]["init_seed"] == job["init_seed"] and seen["kw"]["mc_seed"] == job["mc_seed"]
    assert float(row["C_Q_recomputed_absdiff"]) < 1e-9
    assert (tmp_path / row["state_file"]).exists()


def test_runner_records_failure_without_retry(monkeypatch, tmp_path):
    import em_runner
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise em_runner.EMFailFast("non-finite E-step")
    monkeypatch.setattr(em_runner, "run_em_experimental", boom)
    job = kd.build_manifest()[0]
    row = runner.fit_one(job, TABLE, str(tmp_path))
    assert row["status"] == "FAILED" and "EMFailFast" in row["failure_reason"]
    assert calls["n"] == 1


def test_runner_nonfinite_q_is_failed(monkeypatch, tmp_path):
    import em_runner
    monkeypatch.setattr(em_runner, "run_em_experimental",
                        lambda X, Y, fx, fy, k, **kw: _fake_result(k, q=float("nan")))
    row = runner.fit_one(kd.build_manifest()[0], TABLE, None)
    assert row["status"] == "FAILED"


def test_writer_roundtrip_and_resume_set(tmp_path, monkeypatch):
    import em_runner
    monkeypatch.setattr(em_runner, "run_em_experimental",
                        lambda X, Y, fx, fy, k, **kw: _fake_result(k, q=-1234.56789012345))
    W = runner.Writers(tmp_path)
    job = kd.build_manifest()[5]
    row = runner.fit_one(job, TABLE, None)
    cq = row["C_Q"]
    W.write(row)
    back = pd.read_csv(tmp_path / "per_fit.csv", float_precision="round_trip")
    assert float(back.C_Q.iloc[0]) == float(cq)
    assert runner.recorded_fit_ids(tmp_path) == {job["fit_id"]}
    assert len(pd.read_csv(tmp_path / "mstep_q_history.csv")) == 2


def test_tiny_real_em_boundary_k1_k10():
    """Real estimator on a tiny problem: candidate K = 1 and 10 run and are finite."""
    from em_runner import run_em_experimental
    from data_generator_canonical import generate_canonical_data
    ds = generate_canonical_data(n=30, d=15, k=9, seed=3, family_x="gaussian",
                                 family_y="bernoulli", f_scale=kd.f_scale_of("a", 9),
                                 sigma_x_var=0.1, w0=-2.8, w=1.0)
    for k in (1, 10):
        r = run_em_experimental(ds.X, ds.Y, "gaussian", "bernoulli", k, L=2, num_iter=2,
                                numerics_mode="consistent", failure_policy="fail_fast",
                                init_seed=1, mc_seed=2)
        assert np.isfinite(r["Q_strict"]) and np.isfinite(r["bic"])
        assert r["retry_count"] == 0 and r["replacement_count"] == 0


# ---------------------------------------------------------------- analysis

def _fake_run(tmp_path, kt=3, fail=None, tie=False):
    rows = []
    man = [r for r in kd.build_manifest() if r["dataset_id"] == f"O-a-KT{kt}-r0"]
    for r in man:
        cq = (r["k"] - kt) ** 2 + 0.1 * r["t"]
        if tie and r["k"] in (kt, kt + 1):
            cq = 0.0
        rows.append(dict(fit_id=r["fit_id"], status="OK", failure_reason="", C_Q=cq, Q_strict=0,
                         x_hash="x", y_hash="y", runtime_s=1.0, q_rel_change_last=0.01,
                         mstep_q_decreases=0, w_hat=1, w0_hat=0))
    if fail:
        for row in rows:
            if row["fit_id"] == fail:
                row.update(status="FAILED", C_Q=float("nan"))
    pd.DataFrame(rows).to_csv(tmp_path / "per_fit.csv", index=False)
    return pd.DataFrame(man)


def test_analysis_primary_mean_and_outcome(tmp_path):
    man = _fake_run(tmp_path, kt=3)
    sel = an.select(an.load(tmp_path, man))
    assert sel.primary_k.iloc[0] == 3 and sel.primary_outcome.iloc[0] == "exact"
    assert sel.inits_agree.iloc[0] == True  # noqa: E712


def test_analysis_tie_smallest_k(tmp_path):
    man = _fake_run(tmp_path, kt=3, tie=True)
    sel = an.select(an.load(tmp_path, man))
    assert sel.primary_k.iloc[0] == 3


def test_analysis_failed_fit_makes_incomplete(tmp_path):
    man = _fake_run(tmp_path, kt=3, fail="O-a-KT3-r0-k07-t1")
    sel = an.select(an.load(tmp_path, man))
    assert sel.primary_status.iloc[0] == "INCOMPLETE"
    assert sel.init0_k.iloc[0] == 3 and np.isnan(sel.init1_k.iloc[0])


def test_clopper_pearson_known_values():
    lo, hi = an.clopper_pearson(10, 10)
    assert abs(lo - 0.6915) < 1e-4 and hi == 1.0
    lo, hi = an.clopper_pearson(0, 10)
    assert lo == 0.0 and abs(hi - 0.3085) < 1e-4
    lo, hi = an.clopper_pearson(5, 10)
    assert abs(lo - 0.1871) < 1e-4 and abs(hi - 0.8129) < 1e-4
