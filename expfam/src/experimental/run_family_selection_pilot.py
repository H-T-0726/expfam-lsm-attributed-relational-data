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
    SELECTOR_VERSION,
    pilot_convergence_gate,
    run_hybrid_family_selection,
)

RUNNER_VERSION = "family-selection-pilot-runner-v1"

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


def build_runinfo(protocol: Protocol, *, started: str, finished: str,
                  em_executions: int) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "selector_version": SELECTOR_VERSION,
        "generator_version": MIXED_GENERATOR_VERSION,
        "stage": protocol.stage,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "started_utc": started,
        "finished_utc": finished,
        "em_executions": int(em_executions),
        "expected_em_executions": protocol.expected_em_executions,
        "numerics_mode": "consistent",
        "failure_policy": "fail_fast",
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
# one (replicate, start) run
# --------------------------------------------------------------------------

def run_one(protocol: Protocol, replicate: Replicate, dataset,
            start_label: str, ambiguous_start: str,
            verbose: bool = False) -> dict[str, Any]:
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
    )
    result["replicate"] = replicate.label
    result["start_label"] = start_label
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
        "rmse_Z": refit.get("rmse_Z", ""),
        "w0_est": refit.get("w0", refit.get("w0_est", "")),
        "w_est": refit.get("w", refit.get("w_est", "")),
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
    "generator_provenance.csv",
    "support_gate.csv",
    "family_scores.csv",
    "selection_trace.csv",
    "fit_results.csv",
    "summary.json",
)


def write_artifacts(out_dir: Path, protocol: Protocol, *,
                    runinfo: dict[str, Any],
                    provenance: Sequence[dict[str, Any]],
                    gates: Sequence[dict[str, Any]],
                    scores: Sequence[dict[str, Any]],
                    traces: Sequence[dict[str, Any]],
                    fits: Sequence[dict[str, Any]],
                    summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "protocol.json", protocol.as_json())
    _write_json(out_dir / "runinfo.json", runinfo)
    _write_csv(out_dir / "generator_provenance.csv", provenance)
    _write_csv(out_dir / "support_gate.csv", gates)
    _write_csv(out_dir / "family_scores.csv", scores)
    _write_csv(out_dir / "selection_trace.csv", traces)
    _write_csv(out_dir / "fit_results.csv", fits)
    _write_json(out_dir / "summary.json", summary)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def execute(stage: str, out_dir: Path, verbose: bool = False) -> dict[str, Any]:
    """Run a frozen stage exactly once and write its artifacts."""

    _require(stage in PROTOCOLS,
             f"unknown stage {stage!r}; choose from {sorted(PROTOCOLS)}")
    protocol = PROTOCOLS[stage]
    _require(not out_dir.exists(),
             f"{out_dir} already exists; a recorded run is never overwritten")

    started = datetime.now(timezone.utc).isoformat()
    provenance: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    em_executions = 0

    for replicate in protocol.replicates:
        dataset = build_dataset(protocol, replicate)
        provenance.extend(generator_provenance_rows(protocol, replicate, dataset))
        for start_label, ambiguous_start in STARTS:
            result = run_one(protocol, replicate, dataset,
                             start_label, ambiguous_start, verbose=verbose)
            em_executions += 2            # one exploration, one refit
            results.append(result)
            gates.extend(support_gate_rows(protocol, result))
            scores.extend(family_score_rows(protocol, result))
            traces.extend(selection_trace_rows(protocol, result))
            fits.append(fit_result_row(protocol, result))

    _require(em_executions == protocol.expected_em_executions,
             f"ran {em_executions} EM executions, expected "
             f"{protocol.expected_em_executions}")

    summary = build_summary(protocol, results, provenance)
    runinfo = build_runinfo(
        protocol, started=started,
        finished=datetime.now(timezone.utc).isoformat(),
        em_executions=em_executions)
    write_artifacts(out_dir, protocol, runinfo=runinfo, provenance=provenance,
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
