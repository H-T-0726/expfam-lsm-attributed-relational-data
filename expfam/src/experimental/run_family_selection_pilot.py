"""Frozen runner for the Phase 9C family-selection smoke and pilot (Issue #74).

The C1 smoke and C2 pilot conditions are written out literally below, exactly
as Issue #74 froze them, and they are the ONLY source of run configuration.
The command line takes a stage name and an output directory and nothing else:
there is no flag for n, d, K, the family pattern, L, num_iter, the seeds or
the starts.  That is deliberate.  A runner that accepts those as arguments
lets a disappointing result be answered by a different argument, and the whole
point of freezing a protocol is that the answer arrives before anyone knows
whether they like it.

Nothing here reruns, retries, reseeds or repairs.  The refit runs under
``failure_policy='fail_fast'`` (see ``em_runner.EMFailFast``), so a non-finite
E-step ends the run instead of being repaired into a clean-looking one.

What this measures, and what it does not
----------------------------------------
Under the three implemented families and the strict support gate, the
candidate score only decides anything for columns whose observations all lie
in {0, 1}: Bernoulli against Poisson.  Gaussian and multi-valued-count columns
are settled by the gate, with no score involved.  So this is a FEASIBILITY
PILOT for the selection machinery, not evidence about automatic distribution
selection in general, and an accuracy number that mixes gate-decided columns
into the denominator would be meaningless.  ``support_gate.csv`` carries
``decided_by`` for exactly this reason.

Usage (only after a human has authorised the stage in Issue #74)::

    python run_family_selection_pilot.py --stage smoke --out <fresh directory>
    python run_family_selection_pilot.py --stage pilot --out <fresh directory>

The output directory must not already exist: an existing run is never
overwritten.  ``audit_report.json`` is NOT written here -- it is produced by
``audit_family_selection_pilot.py``, which reads the artifacts and does not
import this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_generator_canonical import f_scale_for_row_norm          # noqa: E402
from data_generator_canonical_mixed import (                       # noqa: E402
    MIXED_GENERATOR_VERSION,
    generate_canonical_mixed_data,
)
from family_selection import (                                     # noqa: E402
    BFGS_GTOL,
    BFGS_MAXITER,
    BFGS_METHOD,
    CANDIDATE_GRAD_INF_TOL,
    LOADING_INSTALLATION_POLICY,
    OPTIMIZER_BFGS,
    PHASE9C_CANDIDATE_OPTIMIZER,
    SELECTOR_VERSION,
    pilot_convergence_gate,
    run_hybrid_family_selection,
)

RUNNER_VERSION = "family-selection-pilot-runner-v2"

# Issue #74 froze num_iter for the fixed-family REFIT only.  It says nothing
# about how long the exploration should run, so that value is a scientific
# protocol choice this implementation must not promote on its own.  It is
# written down here with its approval state attached, recorded into
# protocol.json, and checked before any EM runs: a human has to set
# ``approved`` to True (and say where they approved it) before a stage can
# execute.  Silently shipping 8 because 8 seemed reasonable is exactly the
# kind of implementation-made protocol decision the review asked us not to
# make.
EXPLORATION_NUM_ITER_APPROVAL: dict[str, Any] = {
    "parameter": "exploration_num_iter",
    "value": 8,
    "approved": True,
    "rationale": "match the already-frozen fixed-family refit budget "
                 "(num_iter=8) and avoid introducing a new tuning axis in "
                 "this feasibility pilot",
    "approved_by": "Human",
    "approved_in": "Issue #74 Human Gate approval comment",
    "approval_url": "https://github.com/H-T-0726/"
                    "expfam-lsm-attributed-relational-data/issues/74"
                    "#issuecomment-5795174457",
    "approval_date": "2026-09-23",
    "scope": "Issue #74 Phase 9C feasibility pilot only; does not generalise "
             "to other experiments or to any manuscript claim",
    "note": "Issue #74 froze refit num_iter only. This value was approved "
            "separately and explicitly; no stage may execute without it.",
}

# Both starts are run on the same data with the same seeds; the pair is what
# makes the incumbent-loading path dependence visible.
STARTS: tuple[tuple[str, str], ...] = (("start_B", "bernoulli"),
                                       ("start_P", "poisson"))


@dataclass(frozen=True)
class Replicate:
    """One dataset and the three seeds that produce its two runs."""

    label: str
    data_seed: int
    search_seed: int
    refit_seed: int


@dataclass(frozen=True)
class Protocol:
    """A frozen stage configuration. Every field comes from Issue #74."""

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
    refit_num_iter: int
    exploration_num_iter: int
    replicates: tuple[Replicate, ...]

    @property
    def k_fit(self) -> int:
        """K is held at its true value; #74 runs no K search."""

        return self.k_true

    @property
    def expected_em_executions(self) -> int:
        """Two EM runs per (replicate, start): one exploration, one refit."""

        return len(self.replicates) * len(STARTS) * 2

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["family_x_list"] = list(self.family_x_list)
        payload["replicates"] = [asdict(r) for r in self.replicates]
        payload["k_fit"] = self.k_fit
        payload["starts"] = [{"label": label, "ambiguous_start": family}
                             for label, family in STARTS]
        payload["expected_em_executions"] = self.expected_em_executions
        payload["human_approvals"] = [dict(EXPLORATION_NUM_ITER_APPROVAL)]
        # Which optimiser produced the candidate scores is part of what the
        # run means, so it is recorded in the protocol rather than left to be
        # inferred from the rows.
        payload["candidate_optimizer"] = PHASE9C_CANDIDATE_OPTIMIZER
        payload["candidate_optimizer_settings"] = (
            {"method": BFGS_METHOD, "jac": "analytic_production_gradient",
             "maxiter": BFGS_MAXITER, "gtol": BFGS_GTOL,
             "finite_difference_jacobian": False, "fallback_solvers": []}
            if PHASE9C_CANDIDATE_OPTIMIZER == OPTIMIZER_BFGS else {})
        payload["candidate_convergence_rule"] = {
            "rule": "finite and final analytic gradient infinity norm <= tol",
            "convergence_grad_inf_tol": CANDIDATE_GRAD_INF_TOL,
            "scipy_success_is_criterion": False,
        }
        # What the exploration M-step installs for an ambiguous column
        # (Gate 74-B5). Recorded so a run says which semantics produced it.
        payload["ambiguous_loading_installation"] = LOADING_INSTALLATION_POLICY
        return payload


# --------------------------------------------------------------------------
# The frozen protocols.  Issue #74, sections C1 and C2.
# --------------------------------------------------------------------------

SMOKE = Protocol(
    stage="smoke",
    n=40,
    d=6,
    k_true=3,
    family_x_list=("gaussian", "gaussian", "bernoulli",
                   "bernoulli", "poisson", "poisson"),
    family_y="bernoulli",
    sigma_x_var=1.0,
    w0=-1.0,
    w=1.0,
    f_scale=f_scale_for_row_norm(0.5, d=6, k=3),          # = 1.0
    L=5,
    refit_num_iter=8,
    # Issue #74 fixes num_iter for the fixed-family refit only.  The
    # exploration length is not stated there, so it is frozen HERE at the same
    # value rather than left to the call site, and it is recorded in
    # protocol.json. It is not a tuning knob.
    exploration_num_iter=8,
    replicates=(Replicate(label="rep1", data_seed=941001,
                          search_seed=942001, refit_seed=943001),),
)

PILOT = Protocol(
    stage="pilot",
    n=75,
    d=12,
    k_true=3,
    family_x_list=("gaussian", "gaussian", "gaussian",
                   "bernoulli", "bernoulli", "bernoulli",
                   "bernoulli", "bernoulli", "bernoulli",
                   "poisson", "poisson", "poisson"),
    family_y="bernoulli",
    sigma_x_var=1.0,
    w0=-1.0,
    w=1.0,
    f_scale=f_scale_for_row_norm(0.5, d=12, k=3),         # = sqrt(2)
    L=5,
    refit_num_iter=8,
    exploration_num_iter=8,
    replicates=(
        Replicate(label="rep1", data_seed=951001,
                  search_seed=952001, refit_seed=953001),
        Replicate(label="rep2", data_seed=951002,
                  search_seed=952002, refit_seed=953002),
        Replicate(label="rep3", data_seed=951003,
                  search_seed=952003, refit_seed=953003),
    ),
)

PROTOCOLS: dict[str, Protocol] = {"smoke": SMOKE, "pilot": PILOT}


class RunnerStop(RuntimeError):
    """Fail-fast stop for the runner."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerStop(message)


class ExecutionLedger:
    """Record of every real EM execution this run ATTEMPTED.

    An attempt is written to disk BEFORE the execution starts, so a run that
    dies inside an EM fit still leaves evidence of what it had begun.  The
    attempted count -- not the number of calls that returned -- is what the
    execution cap and the stop rule are about: EM work that failed still
    happened.

    The file is rewritten atomically on every transition rather than appended
    to, so a reader never sees a half-written row.
    """

    FIELDS = ("sequence", "stage", "replicate", "start_label",
              "execution_kind", "seed", "status", "started_utc",
              "finished_utc", "detail")

    def __init__(self, path: Path, stage: str) -> None:
        self.path = path
        self.stage = stage
        self.entries: list[dict[str, Any]] = []
        self._flush()

    @property
    def attempted(self) -> int:
        """Real EM executions started, whether or not they returned."""

        return len(self.entries)

    def start(self, replicate: str, start_label: str, kind: str,
              seed: int) -> dict[str, Any]:
        entry = {
            "sequence": len(self.entries) + 1,
            "stage": self.stage,
            "replicate": replicate,
            "start_label": start_label,
            "execution_kind": kind,
            "seed": int(seed),
            "status": "STARTED",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "finished_utc": "",
            "detail": "",
        }
        self.entries.append(entry)
        self._flush()
        return entry

    def finish(self, entry: dict[str, Any], status: str,
               detail: str = "") -> None:
        entry["status"] = status
        entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
        entry["detail"] = detail
        self._flush()

    def fail_open_entries(self, detail: str) -> None:
        for entry in self.entries:
            if entry["status"] == "STARTED":
                self.finish(entry, "FAILED", detail)

    def open_entry(self) -> dict[str, Any] | None:
        for entry in reversed(self.entries):
            if entry["status"] in ("STARTED", "FAILED"):
                return entry
        return None

    def _flush(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.FIELDS))
            writer.writeheader()
            for entry in self.entries:
                writer.writerow({key: entry.get(key, "") for key in self.FIELDS})
        temporary.replace(self.path)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_HERE,
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):        # pragma: no cover
        return "unavailable"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=_HERE,
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):        # pragma: no cover
        return True
    return bool(out.strip())


def build_runinfo(protocol: Protocol, *, started: str,
                  finished: str | None = None,
                  em_executions: int = 0,
                  status: str = "RUNNING",
                  git_sha: str | None = None,
                  git_dirty: bool | None = None) -> dict[str, Any]:
    """Provenance for the run.

    ``em_executions`` counts real EM executions ATTEMPTED, so a run that died
    inside a fit still reports the work it started.  Written once before the
    first execution and rewritten at the end.

    ``git_sha`` and ``git_dirty`` are captured by the caller BEFORE the run
    creates its own output directory, because "was the working tree clean"
    is a question about the code that ran, and a run writing its own artifacts
    into the repository would otherwise report itself as a modification.
    """

    return {
        "runner_version": RUNNER_VERSION,
        "selector_version": SELECTOR_VERSION,
        "generator_version": MIXED_GENERATOR_VERSION,
        "stage": protocol.stage,
        "git_sha": _git_sha() if git_sha is None else git_sha,
        "git_dirty": _git_dirty() if git_dirty is None else git_dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "started_utc": started,
        "finished_utc": finished,
        "run_status": status,
        "em_executions": int(em_executions),
        "em_executions_semantics": "real EM executions ATTEMPTED, counted "
                                   "before each call, not on return",
        "expected_em_executions": protocol.expected_em_executions,
        "numerics_mode": "consistent",
        "failure_policy": "fail_fast",
        "candidate_optimizer": PHASE9C_CANDIDATE_OPTIMIZER,
        "ambiguous_loading_installation": LOADING_INSTALLATION_POLICY,
        "lineage": "E (experimental prototype; not adoptable for the manuscript)",
    }


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_dataset(protocol: Protocol, replicate: Replicate):
    """Draw one replicate's dataset. No inference happens here."""

    return generate_canonical_mixed_data(
        n=protocol.n, d=protocol.d, k=protocol.k_true,
        seed=replicate.data_seed,
        family_x_list=list(protocol.family_x_list),
        family_y=protocol.family_y,
        f_scale=protocol.f_scale,
        sigma_x_var=protocol.sigma_x_var,
        w0=protocol.w0, w=protocol.w,
    )


def generator_provenance_rows(protocol: Protocol, replicate: Replicate,
                              dataset) -> list[dict[str, Any]]:
    """One row per column: what was generated, and what the column looks like.

    ``ambiguous_true_poisson`` marks a truly-Poisson column whose draw happens
    to contain only 0 and 1.  The support gate cannot tell such a column from a
    Bernoulli one, so it is a separate category rather than a mis-selection --
    and it is recorded, never redrawn.  Redrawing it would be seed rescue.
    """

    meta = dataset.metadata
    rows = []
    for column in range(protocol.d):
        values = dataset.X[:, column]
        true_family = protocol.family_x_list[column]
        binary_only = bool(np.all((values == 0.0) | (values == 1.0)))
        rows.append({
            "replicate": replicate.label,
            "data_seed": replicate.data_seed,
            "column": column,
            "family_x_true": true_family,
            "generator_version": meta["generator_version"],
            "f_row_norm_sq": meta["f_row_norms_sq"][column],
            "sigma_x_var": ("" if meta["sigma_x_var"][column] is None
                            else meta["sigma_x_var"][column]),
            "column_mean": float(np.mean(values)),
            "column_min": float(np.min(values)),
            "column_max": float(np.max(values)),
            "binary_valued": binary_only,
            "ambiguous_true_poisson": bool(true_family == "poisson"
                                           and binary_only),
        })
    return rows


# --------------------------------------------------------------------------
# secondary diagnostic: Z recovery
# --------------------------------------------------------------------------

def procrustes_rmse_z(Z_est: np.ndarray, Z_true: np.ndarray) -> float:
    """RMSE between the estimated and true Z after the optimal rotation.

    The model identifies Z only up to an orthogonal transform, so a raw RMSE
    would measure the arbitrary basis as much as the recovery.  Computed here
    rather than imported from ``utils_expfam`` so that the runner's import
    graph stays inside the experimental lineage.

    This is a SECONDARY diagnostic.  #74 is about whether the family-selection
    machinery runs and what it picks; Z recovery is context, not the estimand.
    """

    Z_est = np.asarray(Z_est, dtype=np.float64)
    Z_true = np.asarray(Z_true, dtype=np.float64)
    width = min(Z_est.shape[1], Z_true.shape[1])
    rotation_u, _, rotation_vt = np.linalg.svd(
        Z_est[:, :width].T @ Z_true[:, :width])
    aligned = Z_est[:, :width] @ (rotation_u @ rotation_vt)
    return float(np.sqrt(np.mean((aligned - Z_true[:, :width]) ** 2)))


# --------------------------------------------------------------------------
# one (replicate, start) run
# --------------------------------------------------------------------------

def run_one(protocol: Protocol, replicate: Replicate, dataset,
            start_label: str, ambiguous_start: str,
            verbose: bool = False,
            execution_hook: Any = None) -> dict[str, Any]:
    """Run the hybrid selection once. Two EM executions happen inside."""

    result = run_hybrid_family_selection(
        dataset.X, dataset.Y, k=protocol.k_fit,
        ambiguous_start=ambiguous_start,
        family_y=protocol.family_y,
        L=protocol.L,
        exploration_num_iter=protocol.exploration_num_iter,
        refit_num_iter=protocol.refit_num_iter,
        search_seed=replicate.search_seed,
        refit_seed=replicate.refit_seed,
        verbose=verbose,
        execution_hook=execution_hook,
    )
    result["replicate"] = replicate.label
    result["start_label"] = start_label
    # Secondary diagnostic. run_em_experimental does not compute an RMSE (it
    # is not given the truth), so it is computed here against the generator's
    # Z rather than left as an empty column that looks measured.
    estimated_z = result["refit"].get("Z_est")
    result["rmse_Z"] = ("" if estimated_z is None
                        else procrustes_rmse_z(estimated_z, dataset.Z))
    return result


# --------------------------------------------------------------------------
# artifact rows
# --------------------------------------------------------------------------

def support_gate_rows(protocol: Protocol, result: dict[str, Any]
                      ) -> list[dict[str, Any]]:
    rows = []
    for gate in result["gates"]:
        column = int(gate["column"])
        rows.append({
            "replicate": result["replicate"],
            "start_label": result["start_label"],
            **gate,
            "family_x_true": protocol.family_x_list[column],
        })
    return rows


def family_score_rows(protocol: Protocol, result: dict[str, Any]
                      ) -> list[dict[str, Any]]:
    rows = []
    for row in result["candidate_rows"]:
        column = int(row["column"])
        rows.append({
            "replicate": result["replicate"],
            "start_label": result["start_label"],
            **row,
            "family_x_true": protocol.family_x_list[column],
            "selected_family": result["selected_assignment"][column],
        })
    return rows


def selection_trace_rows(protocol: Protocol, result: dict[str, Any]
                         ) -> list[dict[str, Any]]:
    rows = []
    for row in result["selection_trace"]:
        column = int(row["column"])
        rows.append({
            "replicate": result["replicate"],
            "start_label": result["start_label"],
            **row,
            "family_x_true": protocol.family_x_list[column],
        })
    return rows


def fit_result_row(protocol: Protocol, result: dict[str, Any]
                   ) -> dict[str, Any]:
    refit = result["refit"]
    integrity = result["integrity"]
    ambiguous = result["ambiguous_columns"]
    selected = result["selected_assignment"]
    # The accuracy figure is computed over SCORE-decided columns only.
    correct = sum(1 for column in ambiguous
                  if selected[column] == protocol.family_x_list[column])
    return {
        "replicate": result["replicate"],
        "start_label": result["start_label"],
        "ambiguous_start": result["ambiguous_start"],
        "n": protocol.n, "d": protocol.d, "k_fit": protocol.k_fit,
        "k_true": protocol.k_true, "L": protocol.L,
        "exploration_num_iter": protocol.exploration_num_iter,
        "refit_num_iter": protocol.refit_num_iter,
        "search_seed": result["search_seed"],
        "refit_seed": result["refit_seed"],
        "family_x_true": "|".join(protocol.family_x_list),
        "initial_assignment": "|".join(result["initial_assignment"]),
        "selected_assignment": "|".join(selected),
        "n_ambiguous_columns": len(ambiguous),
        "n_ambiguous_correct": correct,
        "n_gaussian_x_cols": result["n_gaussian_x_cols"],
        "Q_strict": refit.get("Q_strict"),
        "bic": refit.get("bic"),
        "num_params": refit.get("num_params"),
        "rmse_Z": result.get("rmse_Z", ""),
        "w0_est": refit.get("w0", ""),
        "w_est": refit.get("w", ""),
        "runtime_s": refit.get("runtime_s"),
        "failure_policy": integrity["failure_policy"],
        "retry_count": integrity["retry_count"],
        "replacement_count": integrity["replacement_count"],
        "seed_rescue_count": integrity["seed_rescue_count"],
        "nan_occurred": refit.get("nan_occurred"),
        "q_bic_failed": refit.get("q_bic_failed"),
        "convergence_gate": result["convergence_gate"]["status"],
    }


def build_summary(protocol: Protocol, results: Sequence[dict[str, Any]],
                  provenance: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate, with the claim boundary attached to the numbers themselves."""

    traces = [row for result in results for row in result["selection_trace"]]
    candidate_rows = [row for result in results
                      for row in result["candidate_rows"]]
    gate = pilot_convergence_gate(traces, candidate_rows)

    per_run = []
    for result in results:
        ambiguous = result["ambiguous_columns"]
        selected = result["selected_assignment"]
        per_run.append({
            "replicate": result["replicate"],
            "start_label": result["start_label"],
            "selected_assignment": selected,
            "ambiguous_columns": ambiguous,
            "ambiguous_selected": [selected[c] for c in ambiguous],
            "ambiguous_true": [protocol.family_x_list[c] for c in ambiguous],
            "margins": {str(k): v for k, v in result["margins"].items()},
        })

    # start_B vs start_P agreement, per replicate, on the ambiguous columns.
    agreement = {}
    by_replicate: dict[str, dict[str, list[str]]] = {}
    for result in results:
        by_replicate.setdefault(result["replicate"], {})[
            result["start_label"]] = result["selected_assignment"]
    for label, starts in by_replicate.items():
        if len(starts) == 2:
            first, second = (starts[name] for name, _ in STARTS)
            agreement[label] = bool(first == second)

    true_bernoulli_to_poisson = 0
    ambiguous_true_poisson = 0
    for result in results:
        for column in result["ambiguous_columns"]:
            true_family = protocol.family_x_list[column]
            chosen = result["selected_assignment"][column]
            if true_family == "bernoulli" and chosen == "poisson":
                true_bernoulli_to_poisson += 1
            if true_family == "poisson":
                ambiguous_true_poisson += 1

    return {
        "stage": protocol.stage,
        "runner_version": RUNNER_VERSION,
        "convergence_gate": gate,
        "per_run": per_run,
        "start_agreement_by_replicate": agreement,
        "true_bernoulli_selected_as_poisson": true_bernoulli_to_poisson,
        "ambiguous_true_poisson_decisions": ambiguous_true_poisson,
        "ambiguous_true_poisson_columns": [
            {"replicate": row["replicate"], "column": row["column"]}
            for row in provenance if row["ambiguous_true_poisson"]
        ],
        "integrity": {
            "retry_count": sum(r["integrity"]["retry_count"] for r in results),
            "replacement_count": sum(r["integrity"]["replacement_count"]
                                     for r in results),
            "seed_rescue_count": sum(r["integrity"]["seed_rescue_count"]
                                     for r in results),
        },
        "claim_boundary": {
            "score_decides_only": "columns whose observations are all in {0,1} "
                                  "(Bernoulli vs Poisson)",
            "gate_decides": "Gaussian columns and counts above 1",
            "do_not_report": [
                "an accuracy figure that includes gate-decided columns",
                "evidence for automatic distribution selection in general",
                "that automatic selection beats a human-specified family",
                "any claim about real data",
            ],
            "lineage": "E (experimental prototype; not adoptable for the "
                       "manuscript)",
        },
    }


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write an empty artifact: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


ARTIFACT_NAMES = (
    "protocol.json",
    "runinfo.json",
    "execution_ledger.csv",
    "generator_provenance.csv",
    "support_gate.csv",
    "family_scores.csv",
    "selection_trace.csv",
    "fit_results.csv",
    "summary.json",
)

# Written only when a run dies part-way through.
FAILURE_ARTIFACT = "failure.json"


def write_artifacts(out_dir: Path, *,
                    runinfo: dict[str, Any],
                    provenance: Sequence[dict[str, Any]],
                    gates: Sequence[dict[str, Any]],
                    scores: Sequence[dict[str, Any]],
                    traces: Sequence[dict[str, Any]],
                    fits: Sequence[dict[str, Any]],
                    summary: dict[str, Any] | None) -> list[str]:
    """Write the result tables into an ALREADY RESERVED run directory.

    protocol.json, the initial runinfo.json and the ledger are written before
    the first EM execution, not here.  Empty tables are skipped rather than
    written as headers alone, and the names actually written are returned so
    that a partial run can say which evidence it managed to preserve.
    """

    written: list[str] = []
    _write_json(out_dir / "runinfo.json", runinfo)
    written.append("runinfo.json")
    for name, rows in (("generator_provenance.csv", provenance),
                       ("support_gate.csv", gates),
                       ("family_scores.csv", scores),
                       ("selection_trace.csv", traces),
                       ("fit_results.csv", fits)):
        if rows:
            _write_csv(out_dir / name, rows)
            written.append(name)
    if summary is not None:
        _write_json(out_dir / "summary.json", summary)
        written.append("summary.json")
    return written


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def execute(stage: str, out_dir: Path, verbose: bool = False) -> dict[str, Any]:
    """Run a frozen stage exactly once and write its artifacts.

    The run directory, the frozen protocol and the initial runinfo are all
    committed to disk BEFORE the first EM execution, and every execution is
    recorded in the ledger before it starts.  If anything raises, the partial
    evidence and a failure.json stay behind and the exception propagates: the
    stage is never rerun, reseeded or repaired.
    """

    _require(stage in PROTOCOLS,
             f"unknown stage {stage!r}; choose from {sorted(PROTOCOLS)}")
    protocol = PROTOCOLS[stage]
    _require(not out_dir.exists(),
             f"{out_dir} already exists; a recorded run is never overwritten")
    _require(bool(EXPLORATION_NUM_ITER_APPROVAL["approved"]),
             "exploration_num_iter is not frozen by Issue #74 and has not "
             "been approved by a human yet. Issue #74 froze the refit "
             "num_iter only. Set EXPLORATION_NUM_ITER_APPROVAL['approved'] "
             "to True, with approved_by and approved_in filled in, once a "
             "human has approved the value; until then no stage executes.")

    started = datetime.now(timezone.utc).isoformat()
    # Captured before the run writes anything: the run's own artifacts are not
    # a modification of the code under test.
    code_sha = _git_sha()
    code_dirty = _git_dirty()

    # --- reserve the run directory and commit the protocol before any EM ---
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "protocol.json", protocol.as_json())
    _write_json(out_dir / "runinfo.json",
                build_runinfo(protocol, started=started, em_executions=0,
                              status="RUNNING", git_sha=code_sha,
                              git_dirty=code_dirty))
    ledger = ExecutionLedger(out_dir / "execution_ledger.csv", protocol.stage)

    provenance: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    context: dict[str, Any] = {"replicate": "", "start_label": "",
                               "execution_kind": ""}

    try:
        for replicate in protocol.replicates:
            context["replicate"] = replicate.label
            dataset = build_dataset(protocol, replicate)
            provenance.extend(
                generator_provenance_rows(protocol, replicate, dataset))
            for start_label, ambiguous_start in STARTS:
                context["start_label"] = start_label

                def hook(kind: str, status: str, info: dict[str, Any],
                         _replicate=replicate, _start=start_label) -> None:
                    context["execution_kind"] = kind
                    if status == "STARTED":
                        hook.entry = ledger.start(          # type: ignore[attr-defined]
                            _replicate.label, _start, kind, info["seed"])
                    else:
                        ledger.finish(hook.entry, status)   # type: ignore[attr-defined]

                result = run_one(protocol, replicate, dataset, start_label,
                                 ambiguous_start, verbose=verbose,
                                 execution_hook=hook)
                results.append(result)
                gates.extend(support_gate_rows(protocol, result))
                scores.extend(family_score_rows(protocol, result))
                traces.extend(selection_trace_rows(protocol, result))
                fits.append(fit_result_row(protocol, result))
    except BaseException as exc:                # noqa: BLE001 - evidence first
        ledger.fail_open_entries(f"{type(exc).__name__}: {exc}")
        runinfo = build_runinfo(
            protocol, started=started,
            finished=datetime.now(timezone.utc).isoformat(),
            em_executions=ledger.attempted, status="FAILED",
            git_sha=code_sha, git_dirty=code_dirty)
        written = write_artifacts(
            out_dir, runinfo=runinfo, provenance=provenance, gates=gates,
            scores=scores, traces=traces, fits=fits, summary=None)
        _write_json(out_dir / FAILURE_ARTIFACT, {
            "run_status": "FAILED",
            "stage": protocol.stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "attempted_em_executions": ledger.attempted,
            "expected_em_executions": protocol.expected_em_executions,
            "replicate": context["replicate"],
            "start_label": context["start_label"],
            "execution_kind": context["execution_kind"],
            # Structurally zero: nothing here retries, replaces Z or reseeds.
            "retry_count": 0,
            "replacement_count": 0,
            "seed_rescue_count": 0,
            "git_sha": code_sha,
            "git_dirty": code_dirty,
            "artifacts_written": written,
            "completed_runs": len(results),
            "note": "The stage stopped here. Do not rerun, reseed or widen "
                    "any budget in response; hand this to a human.",
        })
        raise

    _require(ledger.attempted == protocol.expected_em_executions,
             f"attempted {ledger.attempted} EM executions, expected "
             f"{protocol.expected_em_executions}")

    summary = build_summary(protocol, results, provenance)
    runinfo = build_runinfo(
        protocol, started=started,
        finished=datetime.now(timezone.utc).isoformat(),
        em_executions=ledger.attempted, status="SUCCESS",
        git_sha=code_sha, git_dirty=code_dirty)
    write_artifacts(out_dir, runinfo=runinfo, provenance=provenance,
                    gates=gates, scores=scores, traces=traces, fits=fits,
                    summary=summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a frozen Issue #74 family-selection stage exactly once.")
    parser.add_argument("--stage", required=True, choices=sorted(PROTOCOLS),
                        help="which frozen protocol to run")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh output directory (must not exist)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    summary = execute(args.stage, args.out, verbose=args.verbose)
    gate = summary["convergence_gate"]["status"]
    print(f"stage={summary['stage']} convergence_gate={gate}")
    print(f"artifacts written to {args.out}")
    if gate != "READY_FOR_PILOT":
        print("BLOCKED_FOR_PILOT: at least one ambiguous column had a "
              "candidate that did not converge. Do not raise the optimiser "
              "budget, change the seed or rerun; hand this to a human.")
        return 2
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
