"""Phase 9D (Issue #75): joint family + K selection, one frozen synthetic run.

Research question (Issue #75): given only observed X, Y and a human-specified
K grid, can the merged #74 family-selection machinery and the existing
mixed-family criterion ``C_Q(K)`` be connected into one auditable pipeline
that selects the score-decided column families and K?

For each replicate ONE dataset is drawn. For each (replicate, start, K) the
merged #74 driver ``run_hybrid_family_selection`` runs from scratch: support
gate -> Bernoulli/Poisson candidate scoring on the ambiguous 0/1 columns ->
exploration -> frozen assignment -> fresh fixed-family refit. ``C_Q(K)`` is
the refit's existing criterion value (``calc_bic_exp`` via
``run_em_experimental``); this module adds no criterion and no penalty.

    C_Q(K) = -2 Q_strict(K) + p_K ln n
    p_K    = K d - K(K-1)/2 + n_gaussian_x_cols + 1{family_y = gaussian}

It is the Q-based complete-data / ICL-type criterion, NOT Schwarz BIC
(KI-010). The CSV/JSON field ``bic`` keeps the historical name.

Independence across K is structural: the driver has no warm-start input, and
each call receives only X, Y, K, the start family, and the frozen seeds.
The true K and the true families are never passed to it; they appear only in
the evaluation rows.

Each (replicate, start) path returns its own K_hat = argmin_K C_Q(K) (exact
floating tie -> smaller K). The better start is never chosen.

Lineage E (experimental prototype; not adoptable for the manuscript).

Usage, only after a human authorises Gate 75-B in Issue #75::

    python run_joint_family_k_selection.py --out <fresh directory>

``audit_report.json`` is written by ``audit_joint_family_k_selection.py``,
which keeps its own transcription of the frozen protocol.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import gammaln

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import family_selection as fs                                      # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
from data_generator_canonical import f_scale_for_row_norm          # noqa: E402
from data_generator_canonical_mixed import MIXED_GENERATOR_VERSION  # noqa: E402

RUNNER_VERSION = "joint-family-k-runner-v1"
STAGE = "joint"
STARTS = pilot.STARTS                     # (("start_B", "bernoulli"), ("start_P", "poisson"))

# Gate 75-B is not authorised yet (Issue #75: "EM NOT YET AUTHORIZED").
# execute() refuses to run until a human flips this and says where.
EXECUTION_AUTHORIZATION: dict[str, Any] = {
    "gate": "75-B",
    "authorized": True,
    "authorized_by": "Human",
    "authorized_in": "Issue #75 Human Gate comment 5855763104",
}


@dataclass(frozen=True)
class Replicate:
    label: str
    data_seed: int
    search_seed: int
    refit_seed: int


@dataclass(frozen=True)
class JointProtocol:
    """Issue #75 frozen protocol. Attribute names match the #74 Protocol so
    the #74 data builders can be reused unchanged."""

    stage: str
    n: int
    d: int
    k_true: int
    family_x_list: tuple[str, ...]
    family_y: str
    sigma_x_var: float
    w0: float
    w: float
    f_scale: float
    L: int
    exploration_num_iter: int
    refit_num_iter: int
    k_candidates: tuple[int, ...]
    replicates: tuple[Replicate, ...]

    @property
    def expected_em_executions(self) -> int:
        """exploration + refit per (replicate, start, K)."""

        return len(self.replicates) * len(STARTS) * len(self.k_candidates) * 2

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["family_x_list"] = list(self.family_x_list)
        payload["k_candidates"] = list(self.k_candidates)
        payload["replicates"] = [asdict(r) for r in self.replicates]
        payload["starts"] = [{"label": label, "ambiguous_start": family}
                             for label, family in STARTS]
        payload["expected_em_executions"] = self.expected_em_executions
        payload["candidate_optimizer"] = fs.PHASE9C_CANDIDATE_OPTIMIZER
        payload["candidate_optimizer_settings"] = {
            "method": fs.BFGS_METHOD, "jac": "analytic_production_gradient",
            "maxiter": fs.BFGS_MAXITER, "gtol": fs.BFGS_GTOL,
            "finite_difference_jacobian": False, "fallback_solvers": []}
        payload["candidate_convergence_rule"] = {
            "rule": "finite and final analytic gradient infinity norm <= tol",
            "convergence_grad_inf_tol": fs.CANDIDATE_GRAD_INF_TOL,
            "role": "diagnostic (WARNING), research-first policy",
        }
        payload["ambiguous_loading_installation"] = fs.LOADING_INSTALLATION_POLICY
        payload["refit_failure_policy"] = "fail_fast"
        payload["k_criterion"] = {
            "name": "C_Q",
            "formula": "-2 Q_strict + p_K ln n",
            "p_K": "K*d - K*(K-1)/2 + n_gaussian_x_cols(selected) "
                   "+ 1{family_y=gaussian}",
            "source": "run_em_experimental -> eval_utils.calc_bic_exp",
            "interpretation": "Q-based complete-data / ICL-type; not "
                              "Schwarz BIC",
            "tie_rule": "exact floating equality -> smaller K",
            "family_search_penalty": "none",
        }
        payload["cross_k_state"] = "none (each K fit from scratch)"
        payload["start_policy"] = "both starts retained; never choose the " \
                                  "better start"
        payload["data_draws"] = "one per replicate, shared by all K and starts"
        payload["execution_authorization"] = dict(EXECUTION_AUTHORIZATION)
        return payload


PROTOCOL = JointProtocol(
    stage=STAGE,
    n=75,
    d=12,
    k_true=3,
    family_x_list=("gaussian",) * 3 + ("bernoulli",) * 6 + ("poisson",) * 3,
    family_y="bernoulli",
    sigma_x_var=1.0,
    w0=-1.0,
    w=1.0,
    f_scale=f_scale_for_row_norm(0.5, d=12, k=3),            # = sqrt(2)
    L=5,
    exploration_num_iter=8,
    refit_num_iter=8,
    k_candidates=(1, 2, 3, 4, 5),
    replicates=(
        Replicate("rep1", 961001, 962001, 963001),
        Replicate("rep2", 961002, 962002, 963002),
        Replicate("rep3", 961003, 962003, 963003),
    ),
)


class RunnerStop(RuntimeError):
    """A technical blocker: stop, preserve evidence, never repair."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerStop(message)


# --------------------------------------------------------------------------
# ledger: the #74 ledger with K recorded on every attempt
# --------------------------------------------------------------------------

class JointExecutionLedger(pilot.ExecutionLedger):
    FIELDS = ("sequence", "stage", "replicate", "start_label", "k",
              "execution_kind", "seed", "status", "started_utc",
              "finished_utc", "detail")

    def start(self, replicate: str, start_label: str, kind: str,
              seed: int, k: int | str = "") -> dict[str, Any]:
        entry = {
            "sequence": len(self.entries) + 1, "stage": self.stage,
            "replicate": replicate, "start_label": start_label, "k": k,
            "execution_kind": kind, "seed": int(seed), "status": "STARTED",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "finished_utc": "", "detail": "",
        }
        self.entries.append(entry)
        self._flush()
        return entry


# --------------------------------------------------------------------------
# K selection on the stored criterion values
# --------------------------------------------------------------------------

def select_k_hat(cq_by_k: dict[int, float]) -> int:
    """argmin_K C_Q(K); exact floating tie -> smaller K. No tolerance band."""

    _require(bool(cq_by_k), "no C_Q values to select from")
    for k, value in cq_by_k.items():
        _require(math.isfinite(value), f"C_Q(K={k}) is not finite: {value!r}")
    return min(sorted(cq_by_k), key=lambda k: cq_by_k[k])


def cq_ordering(cq_by_k: dict[int, float]) -> list[int]:
    """K values from best to worst, exact ties broken by smaller K."""

    return sorted(cq_by_k, key=lambda k: (cq_by_k[k], k))


def runner_up_gap(cq_by_k: dict[int, float]) -> float:
    values = sorted(cq_by_k.values())
    return float(values[1] - values[0]) if len(values) > 1 else float("nan")


def k_category(k_hat: int, k_true: int) -> str:
    return "exact" if k_hat == k_true else ("under" if k_hat < k_true
                                            else "over")


# --------------------------------------------------------------------------
# #72 decomposition (diagnostic only; never used to select)
# --------------------------------------------------------------------------

def cq_decomposition(refit: dict[str, Any], X: np.ndarray, Y: np.ndarray,
                     n: int) -> dict[str, float]:
    """D_K + P_Z + P_theta, evaluated from the refit's own state.

    Q_Z, Q_X and Q_Y are the three terms of ``calc_Q_dual_strict_exp``
    (#72 report, section 2), recomputed separately. The selection value is
    the refit's direct ``bic`` field; this only explains it.
    """

    model = refit["model"]
    Z_samples = refit["Z_samples"]
    F, var_z = refit["F"], refit["var_z"]
    w0, w = refit["w0"], refit["w"]
    n_obs, k, L = Z_samples.shape
    q_z = q_x = q_y = 0.0
    for l in range(L):
        Z_l = Z_samples[:, :, l]
        q_z += float(-(n_obs * k / 2.0) * np.log(2.0 * np.pi * var_z)
                     - (1.0 / (2.0 * var_z)) * np.sum(Z_l ** 2))
        q_x += float(model.calc_log_likelihood_X(
            X, Z_samples[:, :, l:l + 1], F))
        q_y += float(model.calc_log_likelihood_Y(
            Y, Z_samples[:, :, l:l + 1], w0, w))
    q_z, q_x, q_y = q_z / L, q_x / L, q_y / L

    poisson_cols = model.columns_of("poisson")
    if len(poisson_cols):
        q_x -= float(np.sum(gammaln(X[:, poisson_cols] + 1)))
    label = getattr(model, "family_y_label", model.family)
    mask = getattr(model, "train_mask", None)
    obs_upper = (np.triu(np.ones((n_obs, n_obs), dtype=bool), k=1)
                 if mask is None else np.triu(mask, k=1))
    if label == "poisson":
        q_y -= float(np.sum(gammaln(Y[obs_upper] + 1)))
    elif label == "gaussian":
        q_y -= 0.5 * np.log(2.0 * np.pi) * float(obs_upper.sum())

    p_theta = float(refit["num_params"]) * float(np.log(n))
    d_k, p_z = -2.0 * (q_x + q_y), -2.0 * q_z
    recomposed = d_k + p_z + p_theta
    direct = float(refit["bic"])
    return {"Q_X": q_x, "Q_Y": q_y, "Q_Z": q_z, "D_K": d_k, "P_Z": p_z,
            "P_theta": p_theta, "C_Q_direct": direct,
            "C_Q_recomposed": recomposed,
            "abs_diff": abs(recomposed - direct)}


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------

def _tag(rows: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return [{**row, "k": k} for row in rows]


def cq_row(protocol: JointProtocol, replicate: Replicate, start_label: str,
           k: int, result: dict[str, Any]) -> dict[str, Any]:
    refit = result["refit"]
    integrity = result["integrity"]
    diagnostic = result["candidate_convergence_diagnostic"]
    return {
        "replicate": replicate.label, "start_label": start_label, "k": k,
        "data_seed": replicate.data_seed,
        "search_seed": result["search_seed"],
        "refit_seed": result["refit_seed"],
        "n": protocol.n, "d": protocol.d, "L": protocol.L,
        "initial_assignment": "|".join(result["initial_assignment"]),
        "selected_assignment": "|".join(result["selected_assignment"]),
        "n_gaussian_x_cols": result["n_gaussian_x_cols"],
        "Q_strict": refit.get("Q_strict"),
        "num_params": refit.get("num_params"),
        "C_Q": refit.get("bic"),
        "q_bic_failed": refit.get("q_bic_failed"),
        "nan_occurred": refit.get("nan_occurred"),
        "refit_runtime_s": refit.get("runtime_s"),
        "failure_policy": integrity["failure_policy"],
        "retry_count": integrity["retry_count"],
        "replacement_count": integrity["replacement_count"],
        "seed_rescue_count": integrity["seed_rescue_count"],
        "warm_start_source": "none",
        "candidate_convergence_status": diagnostic["status"],
        "candidate_warnings": diagnostic["candidate_warnings"],
        "candidate_evaluations": diagnostic["candidate_evaluations"],
    }


def family_rows(protocol: JointProtocol, replicate: Replicate,
                start_label: str, k: int, result: dict[str, Any]
                ) -> list[dict[str, Any]]:
    ambiguous = set(result["ambiguous_columns"])
    rows = []
    for column, selected in enumerate(result["selected_assignment"]):
        rows.append({
            "replicate": replicate.label, "start_label": start_label, "k": k,
            "column": column,
            "decided_by": "score" if column in ambiguous else "gate",
            "initial_family": result["initial_assignment"][column],
            "selected_family": selected,
            "final_margin_neg2": result["margins"].get(column, ""),
            "family_x_true": protocol.family_x_list[column],
        })
    return rows


# --------------------------------------------------------------------------
# aggregation: P1-P4 from the per-(replicate, start, K) rows alone
# --------------------------------------------------------------------------

def path_results(protocol: JointProtocol, cq_rows: Sequence[dict[str, Any]],
                 fam_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (replicate, start) pipeline path. True labels enter only
    here, as evaluation targets, after K_hat has been chosen."""

    score_cols = [c for c, f in enumerate(protocol.family_x_list)
                  if f == "bernoulli"]                    # columns 3-8
    out = []
    for replicate in protocol.replicates:
        for start_label, _ in STARTS:
            mine = [r for r in cq_rows if r["replicate"] == replicate.label
                    and r["start_label"] == start_label]
            cq = {int(r["k"]): float(r["C_Q"]) for r in mine}
            k_hat = select_k_hat(cq)
            at_k = {int(r["column"]): r for r in fam_rows
                    if r["replicate"] == replicate.label
                    and r["start_label"] == start_label
                    and int(r["k"]) == k_hat}
            selected = [at_k[c]["selected_family"] for c in score_cols]
            b_to_p = sum(1 for f in selected if f == "poisson")
            all_six = all(f == "bernoulli" for f in selected)
            assignment = next(r["selected_assignment"] for r in mine
                              if int(r["k"]) == k_hat)
            out.append({
                "replicate": replicate.label, "start_label": start_label,
                "k_hat": k_hat,
                "k_hat_equals_k_true": k_hat == protocol.k_true,
                "k_category": k_category(k_hat, protocol.k_true),
                "runner_up_gap": runner_up_gap(cq),
                "cq_ordering": "|".join(str(k) for k in cq_ordering(cq)),
                **{f"C_Q_k{k}": cq[k] for k in sorted(cq)},
                "score_decided_columns": "|".join(map(str, score_cols)),
                "selected_families_at_k_hat": "|".join(selected),
                "bernoulli_to_poisson_at_k_hat": b_to_p,
                "all_six_bernoulli_at_k_hat": all_six,
                "margins_at_k_hat": "|".join(
                    str(at_k[c]["final_margin_neg2"]) for c in score_cols),
                "selected_assignment_at_k_hat": assignment,
                "joint_exact": bool(k_hat == protocol.k_true and all_six),
            })
    return out


def start_stability(paths: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """P4, per replicate. Starts share data: they are not independent."""

    out = {}
    for label in dict.fromkeys(p["replicate"] for p in paths):
        pair = {p["start_label"]: p for p in paths if p["replicate"] == label}
        b, p = pair["start_B"], pair["start_P"]
        out[label] = {
            "k_hat_agree": b["k_hat"] == p["k_hat"],
            "assignment_at_k_hat_agree": (b["selected_assignment_at_k_hat"]
                                          == p["selected_assignment_at_k_hat"]),
            "cq_ordering_agree": b["cq_ordering"] == p["cq_ordering"],
        }
    return out


def build_summary(protocol: JointProtocol, cq_rows, fam_rows, paths,
                  warnings_by_k) -> dict[str, Any]:
    return {
        "stage": protocol.stage,
        "runner_version": RUNNER_VERSION,
        "paths": len(paths),
        "P1_k_recovery": {
            "k_hat_by_path": {f"{p['replicate']}/{p['start_label']}":
                              p["k_hat"] for p in paths},
            "exact": sum(p["k_category"] == "exact" for p in paths),
            "under": sum(p["k_category"] == "under" for p in paths),
            "over": sum(p["k_category"] == "over" for p in paths),
        },
        "P2_family_recovery_at_k_hat": {
            "bernoulli_to_poisson_total": sum(
                p["bernoulli_to_poisson_at_k_hat"] for p in paths),
            "all_six_exact_paths": sum(p["all_six_bernoulli_at_k_hat"]
                                       for p in paths),
        },
        "P3_joint_exact_paths": sum(p["joint_exact"] for p in paths),
        "P4_start_stability": start_stability(paths),
        "candidate_convergence_warnings_by_k": warnings_by_k,
        "integrity": {
            key: sum(int(r[key]) for r in cq_rows)
            for key in ("retry_count", "replacement_count",
                        "seed_rescue_count")},
        "claim_boundary": {
            "unit": "pipeline path (replicate, start); starts share data "
                    "and are not independent replicates; no confidence "
                    "interval from six paths",
            "family_accuracy_columns": "score-decided true-Bernoulli "
                                       "columns 3-8 only",
            "criterion": "C_Q: Q-based complete-data / ICL-type, not "
                         "Schwarz BIC; discrete family search not "
                         "separately penalised (known limitation)",
            "do_not_claim": [
                "K-selection consistency",
                "general family-selection validity",
                "general joint-selection reliability",
                "real-data effectiveness",
                "superiority over human choices",
                "robustness beyond this frozen condition and K grid",
            ],
            "lineage": "E (experimental prototype; not adoptable for the "
                       "manuscript)",
        },
    }


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def build_runinfo(protocol: JointProtocol, *, started: str, git_sha: str,
                  git_dirty: bool, em_executions: int, status: str,
                  finished: str | None = None) -> dict[str, Any]:
    import platform

    return {
        "runner_version": RUNNER_VERSION,
        "selector_version": fs.SELECTOR_VERSION,
        "generator_version": MIXED_GENERATOR_VERSION,
        "stage": protocol.stage,
        "git_sha": git_sha, "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "started_utc": started, "finished_utc": finished,
        "run_status": status,
        "em_executions": int(em_executions),
        "em_executions_semantics": "real EM executions ATTEMPTED",
        "expected_em_executions": protocol.expected_em_executions,
        "numerics_mode": "consistent",
        "failure_policy": "fail_fast",
        "lineage": "E (experimental prototype; not adoptable for the "
                   "manuscript)",
    }


ARTIFACTS = ("protocol.json", "runinfo.json", "execution_ledger.csv",
             "generator_provenance.csv", "support_gate.csv",
             "family_scores.csv", "selection_trace.csv", "family_by_k.csv",
             "cq_by_k.csv", "cq_decomposition.csv", "joint_selection.csv",
             "summary.json")
FAILURE_ARTIFACT = "failure.json"


def execute(out_dir: Path, *, protocol: JointProtocol = PROTOCOL,
            driver: Callable[..., dict[str, Any]] | None = None,
            verbose: bool = False) -> dict[str, Any]:
    """Run the frozen protocol exactly once and write its artifacts.

    ``driver`` defaults to the merged #74 ``run_hybrid_family_selection``;
    tests pass a stub. The dataset is drawn once per replicate; every
    (start, K) call receives fresh copies of X and Y and nothing else from
    any other call.
    """

    _require(bool(EXECUTION_AUTHORIZATION["authorized"]),
             "Gate 75-B is not authorised: Issue #75 says EM is not yet "
             "authorised. A human must record the authorisation first.")
    _require(not out_dir.exists(),
             f"{out_dir} already exists; a recorded run is never overwritten")
    run = driver if driver is not None else fs.run_hybrid_family_selection

    started = datetime.now(timezone.utc).isoformat()
    code_sha, code_dirty = pilot._git_sha(), pilot._git_dirty()
    out_dir.mkdir(parents=True, exist_ok=False)
    pilot._write_json(out_dir / "protocol.json", protocol.as_json())
    pilot._write_json(out_dir / "runinfo.json", build_runinfo(
        protocol, started=started, git_sha=code_sha, git_dirty=code_dirty,
        em_executions=0, status="RUNNING"))
    ledger = JointExecutionLedger(out_dir / "execution_ledger.csv",
                                  protocol.stage)

    tables: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("generator_provenance", "support_gate",
                              "family_scores", "selection_trace",
                              "family_by_k", "cq_by_k", "cq_decomposition")}
    warnings_by_k: dict[str, int] = {}
    context = {"replicate": "", "start_label": "", "k": "",
               "execution_kind": ""}
    try:
        for replicate in protocol.replicates:
            context["replicate"] = replicate.label
            dataset = pilot.build_dataset(protocol, replicate)   # one draw
            tables["generator_provenance"].extend(
                pilot.generator_provenance_rows(protocol, replicate, dataset))
            for start_label, ambiguous_start in STARTS:
                for k in protocol.k_candidates:
                    context.update(start_label=start_label, k=k)

                    def hook(kind, status, info, _r=replicate.label,
                             _s=start_label, _k=k):
                        context["execution_kind"] = kind
                        if status == "STARTED":
                            hook.entry = ledger.start(_r, _s, kind,
                                                      info["seed"], _k)
                        else:
                            ledger.finish(hook.entry, status)

                    X, Y = dataset.X.copy(), dataset.Y.copy()
                    result = run(
                        X, Y, k=k, ambiguous_start=ambiguous_start,
                        family_y=protocol.family_y, L=protocol.L,
                        exploration_num_iter=protocol.exploration_num_iter,
                        refit_num_iter=protocol.refit_num_iter,
                        search_seed=replicate.search_seed,
                        refit_seed=replicate.refit_seed,
                        verbose=verbose, execution_hook=hook)
                    refit = result["refit"]
                    _require(not refit.get("q_bic_failed")
                             and math.isfinite(float(refit.get("bic"))),
                             f"{replicate.label}/{start_label}/K={k}: final "
                             f"C_Q is not finite")
                    result["replicate"] = replicate.label
                    result["start_label"] = start_label
                    tagged = [{"replicate": replicate.label,
                               "start_label": start_label, **row}
                              for row in result["selection_trace"]]
                    result["candidate_convergence_diagnostic"] = \
                        fs.candidate_convergence_diagnostic(
                            tagged, result["candidate_rows"])
                    warnings_by_k[f"{replicate.label}/{start_label}/K={k}"] = \
                        result["candidate_convergence_diagnostic"][
                            "candidate_warnings"]
                    tables["support_gate"].extend(_tag(
                        pilot.support_gate_rows(protocol, result), k))
                    tables["family_scores"].extend(_tag(
                        pilot.family_score_rows(protocol, result), k))
                    tables["selection_trace"].extend(_tag(
                        pilot.selection_trace_rows(protocol, result), k))
                    tables["family_by_k"].extend(family_rows(
                        protocol, replicate, start_label, k, result))
                    tables["cq_by_k"].append(cq_row(
                        protocol, replicate, start_label, k, result))
                    tables["cq_decomposition"].append({
                        "replicate": replicate.label,
                        "start_label": start_label, "k": k,
                        "num_params": refit["num_params"],
                        **cq_decomposition(refit, dataset.X, dataset.Y,
                                           protocol.n)})
    except BaseException as exc:                # noqa: BLE001 - evidence first
        ledger.fail_open_entries(f"{type(exc).__name__}: {exc}")
        written = _write_tables(out_dir, tables)
        pilot._write_json(out_dir / "runinfo.json", build_runinfo(
            protocol, started=started, git_sha=code_sha,
            git_dirty=code_dirty, em_executions=ledger.attempted,
            status="FAILED", finished=datetime.now(timezone.utc).isoformat()))
        pilot._write_json(out_dir / FAILURE_ARTIFACT, {
            "run_status": "FAILED", "stage": protocol.stage,
            "exception_type": type(exc).__name__, "message": str(exc),
            "attempted_em_executions": ledger.attempted,
            "expected_em_executions": protocol.expected_em_executions,
            **context,
            "retry_count": 0, "replacement_count": 0, "seed_rescue_count": 0,
            "git_sha": code_sha, "git_dirty": code_dirty,
            "artifacts_written": written,
            "note": "Stopped. Do not rerun, reseed or change conditions; "
                    "hand this to a human.",
        })
        raise

    _require(ledger.attempted == protocol.expected_em_executions,
             f"attempted {ledger.attempted} EM executions, expected "
             f"{protocol.expected_em_executions}")
    paths = path_results(protocol, tables["cq_by_k"], tables["family_by_k"])
    tables["joint_selection"] = paths
    summary = build_summary(protocol, tables["cq_by_k"],
                            tables["family_by_k"], paths, warnings_by_k)
    _write_tables(out_dir, tables)
    pilot._write_json(out_dir / "summary.json", summary)
    pilot._write_json(out_dir / "runinfo.json", build_runinfo(
        protocol, started=started, git_sha=code_sha, git_dirty=code_dirty,
        em_executions=ledger.attempted, status="SUCCESS",
        finished=datetime.now(timezone.utc).isoformat()))
    return summary


def _write_tables(out_dir: Path, tables: dict[str, list]) -> list[str]:
    written = []
    for name, rows in tables.items():
        if rows:
            pilot._write_csv(out_dir / f"{name}.csv", rows)
            written.append(f"{name}.csv")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen Issue #75 joint family + K protocol once.")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh output directory (must not exist)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    summary = execute(args.out, verbose=args.verbose)
    print(f"K_hat by path: {summary['P1_k_recovery']['k_hat_by_path']}")
    print(f"joint exact paths: {summary['P3_joint_exact_paths']}/"
          f"{summary['paths']}; artifacts in {args.out}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
