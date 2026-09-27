"""Zero-EM contract tests for the Issue #75 joint family + K pipeline.

The merged #74 driver is replaced by a stub; the stub's criterion goes through
the REAL ``calc_Q_dual_strict_exp`` / ``calc_bic_exp`` with a fake likelihood,
so p_K, C_Q and the #72 decomposition are exercised on the production path.
Any attempt to reach real EM fails the test.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import audit_joint_family_k_selection as auditor               # noqa: E402
import family_selection as fs                                  # noqa: E402
import run_family_selection_pilot as pilot                     # noqa: E402
import run_joint_family_k_selection as joint                   # noqa: E402
from eval_utils import calc_bic_exp, calc_Q_dual_strict_exp    # noqa: E402


@pytest.fixture(autouse=True)
def _forbid_em(monkeypatch):
    import em_runner
    from model_dual_expfam_consistent import DualExpFamLSMPerColumnConsistent

    def _no_em(*_args, **_kwargs):
        raise AssertionError("real EM was reached inside a zero-EM test")

    monkeypatch.setattr(em_runner, "run_em_experimental", _no_em)
    monkeypatch.setattr(fs, "run_em_experimental", _no_em)
    monkeypatch.setattr(fs, "run_family_exploration", _no_em)
    monkeypatch.setattr(DualExpFamLSMPerColumnConsistent, "calc_eta_newton",
                        _no_em)
    monkeypatch.setattr(pilot, "_git_dirty", lambda: False)
    monkeypatch.setitem(joint.EXECUTION_AUTHORIZATION, "authorized", True)


# --------------------------------------------------------------------------
# a stub for run_hybrid_family_selection
# --------------------------------------------------------------------------

# Fake X log-likelihood by K, chosen so that C_Q is smallest at K = 3.
FIT_BY_K = {1: -1000.0, 2: -600.0, 3: -300.0, 4: -290.0, 5: -285.0}


class FakeModel:
    family_x = "mixed"
    family = "bernoulli"
    family_y_label = "bernoulli"
    train_mask = None

    def __init__(self, assignment, lnpx):
        self.assignment, self.lnpx = assignment, lnpx

    def columns_of(self, family):
        return np.array([c for c, f in enumerate(self.assignment)
                         if f == family], dtype=int)

    def calc_log_likelihood_X(self, X, Z, F):
        return self.lnpx

    def calc_log_likelihood_Y(self, Y, Z, w0, w):
        return -500.0


class Stub:
    """Records every call. ``poisson_at`` = {(start, k): [cols]} forces those
    ambiguous columns to Poisson; ``fit`` overrides FIT_BY_K per (rep, start)."""

    def __init__(self, poisson_at=None, fit=None, fail_at=None):
        self.calls = []
        self.poisson_at = poisson_at or {}
        self.fit = fit or {}
        self.fail_at = fail_at

    def __call__(self, X, Y, *, k, ambiguous_start, family_y, L,
                 exploration_num_iter, refit_num_iter, search_seed,
                 refit_seed, verbose=False, execution_hook=None):
        start = "start_B" if ambiguous_start == "bernoulli" else "start_P"
        rep = f"rep{search_seed - 962000}"
        self.calls.append({"rep": rep, "start": start, "k": k,
                           "X_id": id(X), "X": X.copy(),
                           "search_seed": search_seed,
                           "refit_seed": refit_seed})
        X[:] = -99.0            # a driver that scribbles on its input
        gates = fs.support_gate(self.calls[-1]["X"])
        ambiguous = [g.column for g in gates if g.is_ambiguous]
        initial = fs.initial_assignment(gates, ambiguous_start)
        selected = list(initial)
        for c in ambiguous:
            selected[c] = ("poisson" if c in self.poisson_at.get((start, k), [])
                           else "bernoulli")
        for kind, seed in (("exploration", search_seed), ("refit", refit_seed)):
            execution_hook(kind, "STARTED", {"seed": seed})
            if self.fail_at == (rep, start, k, kind):
                raise fs.SelectorStop("stub failure")
            execution_hook(kind, "SUCCESS", {"seed": seed})

        trace, cand = [], []
        for c in ambiguous:
            for it in range(1, exploration_num_iter + 1):
                ok = not (k == 2 and it == 1)          # some finite warnings
                row = {"iteration": it, "column": c,
                       "selected_family": selected[c],
                       "previous_family": initial[c], "changed": False,
                       "score_bernoulli": -10.0, "score_poisson": -12.0,
                       "margin_neg2": 4.0, "all_candidates_converged": ok,
                       "candidate_optimizer": "bfgs",
                       "selected_candidate_family": selected[c],
                       "selected_candidate_score": -10.0,
                       "selected_loading": "0.1|0.2",
                       "selected_loading_sha256": f"h{c}{k}",
                       "installed_loading_sha256": f"h{c}{k}",
                       "selected_loading_installed": True,
                       "loading_installation_policy":
                           "selected_candidate_loading"}
                for fam in ("bernoulli", "poisson"):
                    bad = fam == "poisson" and not ok
                    row.update({f"{fam}_optimizer": "bfgs",
                                f"{fam}_n_iter": 7,
                                f"{fam}_converged": not bad,
                                f"{fam}_grad_inf": 3e-8 if bad else 1e-10,
                                f"{fam}_scipy_success": not bad,
                                f"{fam}_scipy_status": 2 if bad else 0})
                trace.append(row)
            for fam in ("bernoulli", "poisson"):
                cand.append({"column": c, "candidate_family": fam,
                             "score": -10.0, "neg2_score": 20.0,
                             "optimizer": "bfgs", "optimiser_iterations": 7,
                             "optimiser_converged": True,
                             "optimiser_gradient_inf": 1e-10,
                             "loading_norm": 0.5})

        n, d = X.shape
        lnpx = self.fit.get((rep, start), FIT_BY_K)[k]
        model = FakeModel(selected, lnpx)
        Z = np.full((n, k, L), 0.3)
        Q = calc_Q_dual_strict_exp(self.calls[-1]["X"], Y, Z, None, None, 1.0,
                                   -1.0, 1.0, model)
        n_gauss = sum(f == "gaussian" for f in selected)
        bic, npar = calc_bic_exp(Q, k, n, d, "mixed", "bernoulli",
                                 n_gaussian_x_cols=n_gauss)
        refit = {"Q_strict": Q, "bic": bic, "num_params": npar,
                 "q_bic_failed": False, "nan_occurred": False,
                 "runtime_s": 0.01, "w0": -1.0, "w": 1.0, "F": None,
                 "var_z": 1.0, "Z_samples": Z, "model": model,
                 "failure_policy": "fail_fast", "retry_count": 0,
                 "replacement_count": 0, "seed_rescue_count": 0}
        return {"gates": [g.as_row() for g in gates],
                "ambiguous_columns": ambiguous,
                "initial_assignment": initial, "selected_assignment": selected,
                "selection_trace": trace, "candidate_rows": cand,
                "margins": {c: 4.0 for c in ambiguous}, "refit": refit,
                "n_gaussian_x_cols": n_gauss,
                "search_seed": search_seed, "refit_seed": refit_seed,
                "integrity": {"failure_policy": "fail_fast", "retry_count": 0,
                              "replacement_count": 0,
                              "seed_rescue_count": 0}}


@pytest.fixture
def draws(monkeypatch):
    """Count real generator calls (cheap, no EM)."""

    calls = []
    real = pilot.generate_canonical_mixed_data

    def counting(**kwargs):
        calls.append(kwargs["seed"])
        return real(**kwargs)

    monkeypatch.setattr(pilot, "generate_canonical_mixed_data", counting)
    return calls


def _run(tmp_path, stub=None, **kwargs):
    stub = stub or Stub()
    out = tmp_path / "run"
    summary = joint.execute(out, driver=stub, **kwargs)
    return out, summary, stub


def _rows(out, name):
    with (out / name).open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def _rewrite(out, name, index, **changes):
    rows = _rows(out, name)
    rows[index].update(changes)
    with (out / name).open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# A. frozen protocol literal
# --------------------------------------------------------------------------

def test_A_the_frozen_issue_75_protocol():
    p = joint.PROTOCOL
    assert (p.n, p.d, p.k_true, p.L) == (75, 12, 3, 5)
    assert list(p.family_x_list) == ["gaussian"] * 3 + ["bernoulli"] * 6 + \
        ["poisson"] * 3
    assert p.family_y == "bernoulli"
    assert (p.sigma_x_var, p.w0, p.w) == (1.0, -1.0, 1.0)
    assert math.isclose(p.f_scale, math.sqrt(2.0), rel_tol=1e-12)
    assert list(p.k_candidates) == [1, 2, 3, 4, 5]
    assert (p.exploration_num_iter, p.refit_num_iter) == (8, 8)
    assert [(r.label, r.data_seed, r.search_seed, r.refit_seed)
            for r in p.replicates] == [
        ("rep1", 961001, 962001, 963001), ("rep2", 961002, 962002, 963002),
        ("rep3", 961003, 962003, 963003)]
    assert joint.STARTS == (("start_B", "bernoulli"), ("start_P", "poisson"))
    assert p.expected_em_executions == 60
    payload = p.as_json()
    assert payload["candidate_optimizer_settings"]["maxiter"] == 2000
    assert payload["candidate_optimizer_settings"]["gtol"] == 1e-10
    assert payload["candidate_convergence_rule"][
        "convergence_grad_inf_tol"] == 1e-8
    a = auditor.Audit()
    auditor.check_protocol(payload, a)
    assert a.findings == []


def test_A_execution_is_not_authorised_yet(tmp_path, monkeypatch):
    monkeypatch.setitem(joint.EXECUTION_AUTHORIZATION, "authorized", False)
    with pytest.raises(joint.RunnerStop, match="not authorised"):
        joint.execute(tmp_path / "x", driver=Stub())
    assert not (tmp_path / "x").exists()


def test_A_the_default_driver_is_the_merged_74_driver(tmp_path):
    # With no stub the runner reaches the #74 driver, which the guard stops.
    with pytest.raises(AssertionError, match="real EM"):
        joint.execute(tmp_path / "x")
    assert (tmp_path / "x" / "failure.json").is_file()


# --------------------------------------------------------------------------
# B. one data draw per replicate
# --------------------------------------------------------------------------

def test_B_one_generator_call_per_replicate(tmp_path, draws):
    _run(tmp_path)
    assert draws == [961001, 961002, 961003]


# --------------------------------------------------------------------------
# C. no cross-K warm start / state reuse
# --------------------------------------------------------------------------

def test_C_the_driver_has_no_warm_start_input():
    params = set(inspect.signature(fs.run_hybrid_family_selection).parameters)
    assert params == {"X", "Y", "k", "ambiguous_start", "family_y", "L",
                      "exploration_num_iter", "refit_num_iter",
                      "search_seed", "refit_seed", "verbose",
                      "execution_hook", "candidate_optimizer"}


def test_C_every_call_gets_fresh_unmodified_data(tmp_path):
    _, _, stub = _run(tmp_path)
    by_rep = {}
    for call in stub.calls:
        by_rep.setdefault(call["rep"], []).append(call)
    for calls in by_rep.values():
        first = calls[0]["X"]
        # the stub overwrote its input each time; later calls still see data
        for c in calls:
            np.testing.assert_array_equal(c["X"], first)
            assert not np.all(c["X"] == -99.0)


def test_C_auditor_flags_carried_over_initial_assignment(tmp_path):
    out, _, _ = _run(tmp_path)
    rows = _rows(out, "cq_by_k.csv")
    index = next(i for i, r in enumerate(rows)
                 if r["start_label"] == "start_P" and r["k"] == "2")
    carried = rows[index]["initial_assignment"].replace("poisson", "bernoulli")
    _rewrite(out, "cq_by_k.csv", index, initial_assignment=carried)
    report = auditor.audit(out)
    assert any(f["check"] == "cross_k" for f in report["findings"])
    assert report["technical_validity"] == "INVALID"


def test_C_auditor_flags_a_declared_warm_start(tmp_path):
    out, _, _ = _run(tmp_path)
    _rewrite(out, "cq_by_k.csv", 3, warm_start_source="k=2")
    report = auditor.audit(out)
    assert any(f["check"] == "cross_k" for f in report["findings"])


# --------------------------------------------------------------------------
# D. per-K family rerun
# --------------------------------------------------------------------------

def test_D_family_selection_runs_once_per_replicate_start_k(tmp_path):
    stub = Stub(poisson_at={("start_B", 1): [3]})
    out, _, stub = _run(tmp_path, stub)
    keys = [(c["rep"], c["start"], c["k"]) for c in stub.calls]
    assert len(keys) == 30 and len(set(keys)) == 30
    fam = _rows(out, "family_by_k.csv")
    col3 = {(r["replicate"], r["k"]): r["selected_family"] for r in fam
            if r["start_label"] == "start_B" and r["column"] == "3"}
    assert col3[("rep1", "1")] == "poisson"
    # K=2 made its own choice rather than inheriting K=1's
    assert col3[("rep1", "2")] == "bernoulli"


# --------------------------------------------------------------------------
# E. no true-label leakage
# --------------------------------------------------------------------------

def test_E_the_driver_never_receives_truth(tmp_path, monkeypatch):
    seen = []

    def spy(X, Y, **kwargs):
        seen.append(set(kwargs))
        return Stub()(X, Y, **kwargs)

    joint.execute(tmp_path / "run", driver=spy)
    assert all(keys == {"k", "ambiguous_start", "family_y", "L",
                        "exploration_num_iter", "refit_num_iter",
                        "search_seed", "refit_seed", "verbose",
                        "execution_hook"} for keys in seen)


def test_E_k_hat_does_not_depend_on_the_true_k(tmp_path):
    out, _, _ = _run(tmp_path)
    cq, fam = _rows(out, "cq_by_k.csv"), _rows(out, "family_by_k.csv")
    base = joint.path_results(joint.PROTOCOL, cq, fam)
    other = joint.path_results(
        dataclasses.replace(joint.PROTOCOL, k_true=5), cq, fam)
    assert [p["k_hat"] for p in base] == [p["k_hat"] for p in other]
    assert {p["k_category"] for p in other} == {"under"}
    assert list(inspect.signature(joint.select_k_hat).parameters) == \
        ["cq_by_k"]


# --------------------------------------------------------------------------
# F. mixed parameter count from the selected assignment
# --------------------------------------------------------------------------

def test_F_gaussian_count_comes_from_the_selected_assignment(tmp_path,
                                                              monkeypatch):
    real = pilot.generate_canonical_mixed_data

    def binary_column_0(**kwargs):
        data = real(**kwargs)
        data.X[:, 0] = (data.X[:, 0] > 0).astype(float)   # now gate-ambiguous
        return data

    monkeypatch.setattr(pilot, "generate_canonical_mixed_data",
                        binary_column_0)
    out, _, _ = _run(tmp_path)
    for row in _rows(out, "cq_by_k.csv"):
        k = int(row["k"])
        assert row["n_gaussian_x_cols"] == "2"
        assert int(row["num_params"]) == k * 12 - k * (k - 1) // 2 + 2
    assert auditor.audit(out)["technical_validity"] == "VALID"


def test_F_auditor_rejects_a_hard_coded_count(tmp_path):
    out, _, _ = _run(tmp_path)
    rows = _rows(out, "cq_by_k.csv")
    _rewrite(out, "cq_by_k.csv", 0,
             num_params=str(int(rows[0]["num_params"]) + 1))
    report = auditor.audit(out)
    assert any(f["check"] == "criterion" for f in report["findings"])


# --------------------------------------------------------------------------
# G. argmin with the exact tie rule
# --------------------------------------------------------------------------

def test_G_argmin_and_exact_tie():
    assert joint.select_k_hat({1: 5.0, 2: 3.0, 3: 4.0}) == 2
    assert joint.select_k_hat({3: 1.0, 2: 1.0, 4: 1.0}) == 2
    tiny = np.nextafter(1.0, 0.0)
    assert joint.select_k_hat({2: 1.0, 3: tiny}) == 3      # no tolerance band
    assert joint.cq_ordering({1: 2.0, 2: 1.0, 3: 1.0}) == [2, 3, 1]
    assert joint.runner_up_gap({1: 5.0, 2: 3.0, 3: 4.0}) == 1.0
    with pytest.raises(joint.RunnerStop):
        joint.select_k_hat({1: float("nan"), 2: 1.0})
    assert auditor.select_k_hat({3: 1.0, 2: 1.0}) == 2


def test_G_k_hat_uses_final_refit_cq(tmp_path):
    out, summary, _ = _run(tmp_path)
    assert set(summary["P1_k_recovery"]["k_hat_by_path"].values()) == {3}
    for row in _rows(out, "joint_selection.csv"):
        curve = {k: float(row[f"C_Q_k{k}"]) for k in range(1, 6)}
        assert int(row["k_hat"]) == min(curve, key=curve.get)


def test_G_auditor_rejects_a_wrong_k_hat(tmp_path):
    out, _, _ = _run(tmp_path)
    _rewrite(out, "joint_selection.csv", 0, k_hat="4")
    report = auditor.audit(out)
    assert any(f["check"] == "joint_selection" for f in report["findings"])


# --------------------------------------------------------------------------
# H. never choose the better start
# --------------------------------------------------------------------------

def test_H_both_starts_are_kept_with_their_own_k_hat(tmp_path):
    fit_p = {**FIT_BY_K, 4: -150.0}           # start_P rep2 prefers K=4
    stub = Stub(fit={("rep2", "start_P"): fit_p})
    out, summary, _ = _run(tmp_path, stub)
    paths = _rows(out, "joint_selection.csv")
    assert len(paths) == 6
    k_hat = {(p["replicate"], p["start_label"]): p["k_hat"] for p in paths}
    assert k_hat[("rep2", "start_B")] == "3"
    assert k_hat[("rep2", "start_P")] == "4"
    assert summary["P4_start_stability"]["rep2"]["k_hat_agree"] is False
    assert summary["P1_k_recovery"]["over"] == 1
    assert auditor.audit(out)["technical_validity"] == "VALID"


# --------------------------------------------------------------------------
# I. joint recovery aggregation
# --------------------------------------------------------------------------

def test_I_family_accuracy_counts_score_decided_columns_only(tmp_path):
    stub = Stub(poisson_at={("start_B", 3): [4]})
    out, summary, _ = _run(tmp_path, stub)
    paths = {(p["replicate"], p["start_label"]): p
             for p in _rows(out, "joint_selection.csv")}
    b = paths[("rep1", "start_B")]
    assert b["score_decided_columns"] == "3|4|5|6|7|8"
    assert b["bernoulli_to_poisson_at_k_hat"] == "1"   # 9-11 are not counted
    assert b["joint_exact"] == "False"
    assert paths[("rep1", "start_P")]["joint_exact"] == "True"
    assert summary["P2_family_recovery_at_k_hat"][
        "bernoulli_to_poisson_total"] == 3
    assert summary["P3_joint_exact_paths"] == 3
    assert auditor.audit(out)["technical_validity"] == "VALID"


def test_I_auditor_rejects_a_summary_that_disagrees(tmp_path):
    out, _, _ = _run(tmp_path)
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    summary["P3_joint_exact_paths"] = 0
    (out / "summary.json").write_text(json.dumps(summary), "utf-8")
    report = auditor.audit(out)
    assert any(f["check"] == "summary" for f in report["findings"])


# --------------------------------------------------------------------------
# J. 60-execution ledger
# --------------------------------------------------------------------------

def test_J_a_complete_run_is_exactly_60_executions(tmp_path):
    out, _, _ = _run(tmp_path)
    ledger = _rows(out, "execution_ledger.csv")
    assert len(ledger) == 60
    assert {r["status"] for r in ledger} == {"SUCCESS"}
    assert {r["k"] for r in ledger} == {"1", "2", "3", "4", "5"}
    runinfo = json.loads((out / "runinfo.json").read_text("utf-8"))
    assert runinfo["em_executions"] == 60
    report = auditor.audit(out)
    assert report["technical_validity"] == "VALID", report["findings"]
    assert report["candidate_convergence_diagnostic"]["candidate_warnings"] > 0
    assert report["warning_count"] >= 1


def test_J_auditor_rejects_a_hidden_retry(tmp_path):
    out, _, _ = _run(tmp_path)
    rows = _rows(out, "execution_ledger.csv")
    rows.append(dict(rows[-1], sequence="61"))
    with (out / "execution_ledger.csv").open("w", encoding="utf-8",
                                             newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = auditor.audit(out)
    assert report["technical_validity"] == "INVALID"


# --------------------------------------------------------------------------
# K. failure preservation
# --------------------------------------------------------------------------

def test_K_a_mid_run_failure_keeps_partial_evidence(tmp_path):
    stub = Stub(fail_at=("rep2", "start_P", 4, "refit"))
    with pytest.raises(fs.SelectorStop):
        joint.execute(tmp_path / "run", driver=stub)
    out = tmp_path / "run"
    failure = json.loads((out / "failure.json").read_text("utf-8"))
    ledger = _rows(out, "execution_ledger.csv")
    assert failure["attempted_em_executions"] == len(ledger) == 38
    assert ledger[-1]["status"] == "FAILED"
    assert (failure["replicate"], failure["start_label"], failure["k"],
            failure["execution_kind"]) == ("rep2", "start_P", 4, "refit")
    assert "cq_by_k.csv" in failure["artifacts_written"]
    report = auditor.audit(out)
    assert report["run_status"] == "FAILED"
    assert report["technical_validity"] == "INVALID"
    assert report["blocker_count"] == 0            # the evidence is intact


def test_K_a_non_finite_criterion_stops_the_run(tmp_path):
    stub = Stub(fit={("rep1", "start_B"): {**FIT_BY_K, 2: float("nan")}})
    with pytest.raises(joint.RunnerStop, match="not finite"):
        joint.execute(tmp_path / "run", driver=stub)
    assert (tmp_path / "run" / "failure.json").is_file()


# --------------------------------------------------------------------------
# L. auditor independence
# --------------------------------------------------------------------------

def test_L_the_auditor_does_not_import_the_runner():
    source = (_HERE / "audit_joint_family_k_selection.py").read_text("utf-8")
    code = source.split('"""', 2)[2]
    assert "run_joint_family_k_selection" not in code
    assert "run_family_selection_pilot" not in code
    assert auditor.EXPECTED["expected_em_executions"] == 60


def test_L_runner_drift_is_caught_by_the_auditor(tmp_path):
    drifted = dataclasses.replace(
        joint.PROTOCOL, k_candidates=(1, 2, 3, 4),
        replicates=joint.PROTOCOL.replicates[:2])
    out, _, _ = _run(tmp_path, protocol=drifted)
    report = auditor.audit(out)
    assert report["technical_validity"] == "INVALID"
    messages = " ".join(f["message"] for f in report["findings"])
    assert "k_candidates" in messages and "replicates" in messages


# --------------------------------------------------------------------------
# C_Q decomposition (diagnostic)
# --------------------------------------------------------------------------

def test_decomposition_recomposes_the_direct_criterion(tmp_path):
    out, _, _ = _run(tmp_path)
    for row in _rows(out, "cq_decomposition.csv"):
        k = int(row["k"])
        assert math.isclose(float(row["C_Q_recomposed"]),
                            float(row["C_Q_direct"]), rel_tol=1e-12)
        assert math.isclose(float(row["P_theta"]),
                            int(row["num_params"]) * math.log(75),
                            rel_tol=1e-12)
        # P_Z depends only on the Z samples: here Z = 0.3 everywhere
        expected_pz = 75 * k * math.log(2 * math.pi) + 75 * k * 0.09
        assert math.isclose(float(row["P_Z"]), expected_pz, rel_tol=1e-12)


def test_selected_loading_mismatch_is_a_blocker(tmp_path):
    out, _, _ = _run(tmp_path)
    _rewrite(out, "selection_trace.csv", 0, installed_loading_sha256="zzz")
    assert auditor.audit(out)["technical_validity"] == "INVALID"
