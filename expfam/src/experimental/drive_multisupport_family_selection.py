"""Sequential driver for the multi-support family-selection run.

Runs the frozen phases in order, and after every phase:

* computes the technical gate of that phase (UNEXPECTED_EXCEPTION count and
  the numerical FAILURE fraction of its fits) -- accuracy plays no role;
* writes ``audit/status_snapshot_<phase>.json`` and refreshes ``RESUME.md``;
* runs the summariser / figure / report scripts if they exist (their failure
  is logged and never stops the experiment);
* stops on a technical gate failure (any UNEXPECTED_EXCEPTION).

The r10 phases run only under the protocol's extension rule.  Nothing here
reruns a failed case, changes a seed, or reads a recovery number.

    python drive_multisupport_family_selection.py --run-dir DIR [--workers 10]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from run_multisupport_family_selection import (  # noqa: E402
    PHASES, atomic_write_json, atomic_write_text, cmd_run, enumerate_cases,
    read_csv,
)

ORDER = ["smoke", "pilot", "A_main", "B_main_r5", "C_main_r5", "B_main_r10",
         "C_main_r10"]
GATE_NAME = {"smoke": "Gate D (selector + MCEM smoke)",
             "pilot": "Gate D' (small pilot)",
             "A_main": "Gate E (selector-only main)",
             "B_main_r5": "Gate F (support-group MCEM, rep 1-5)",
             "C_main_r5": "Gate G (mixed-X MCEM, rep 1-5)",
             "B_main_r10": "Gate F (support-group MCEM, rep 6-10)",
             "C_main_r10": "Gate G (mixed-X MCEM, rep 6-10)"}
POST_SCRIPTS = ["summarize_multisupport_family_selection.py",
                "plot_multisupport_family_selection.py",
                "report_multisupport_family_selection.py"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def phase_status(run_dir: Path, phase: str) -> dict:
    expected = {c["case_id"]: c for c in enumerate_cases(phase)}
    done = {r["case_id"]: r for r in read_csv(run_dir / "completed_cases.csv")
            if r["case_id"] in expected}
    fails = [r for r in read_csv(run_dir / "tables" / "failures.csv")
             if r["case_id"] in expected]
    unexpected = sorted({r["case_id"] for r in fails
                         if r["error_class"] == "UNEXPECTED_EXCEPTION"})
    case_fail = sorted({cid for cid, r in done.items() if r["status"] != "SUCCESS"})
    fits = [c for c in expected.values() if c["method"] != "SELECTOR_ONLY"]
    fit_fail = [c for c in fits if c["case_id"] in case_fail]
    frac = (len(fit_fail) / len(fits)) if fits else 0.0
    return {"phase": phase, "gate": GATE_NAME[phase], "checked_utc": _now(),
            "expected_cases": len(expected), "completed_cases": len(done),
            "missing_cases": sorted(set(expected) - set(done)),
            "failed_cases": case_fail, "unexpected_exception_cases": unexpected,
            "fit_cases": len(fits), "fit_failures": len(fit_fail),
            "fit_failure_fraction": frac,
            "column_selection_failures_in_successful_cases": sum(
                1 for r in fails if r["error_class"] == "OUTCOME_COLUMN_SELECTION"),
            "technical_gate": ("PASS" if not unexpected and len(done) == len(expected)
                               else "FAIL")}


def write_resume(run_dir: Path, statuses: dict, note: str) -> None:
    lines = [f"# RESUME — {run_dir.name}", "",
             f"- updated (UTC): {_now()}",
             f"- note: {note}", "",
             "## Phase status", "",
             "| phase | gate | expected | completed | failed cases | unexpected | "
             "fit failure fraction | technical gate |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for ph in ORDER:
        st = statuses.get(ph)
        if st is None:
            lines.append(f"| {ph} | {GATE_NAME[ph]} | {len(enumerate_cases(ph))} | "
                         f"0 | – | – | – | NOT RUN |")
            continue
        lines.append(f"| {ph} | {st['gate']} | {st['expected_cases']} | "
                     f"{st['completed_cases']} | {len(st['failed_cases'])} | "
                     f"{len(st['unexpected_exception_cases'])} | "
                     f"{st['fit_failure_fraction']:.3f} | {st['technical_gate']} |")
    tables = sorted(p.name for p in (run_dir / "tables").glob("*.csv"))
    figs = sorted(p.name for p in (run_dir / "figures").glob("*.png"))
    lines += ["", "## Existing tables", "", *(f"- tables/{t}" for t in tables),
              "", "## Existing figures", "", *(f"- figures/{f}" for f in figs) or ["- (none)"],
              "", "## How to resume", "",
              "Completed cases are skipped automatically; failed cases are "
              "completed outcomes and are NOT rerun.", "",
              "```",
              f"cd expfam/src/experimental",
              f"python drive_multisupport_family_selection.py --run-dir {run_dir.as_posix()} --workers 10",
              "```", "",
              "A single phase can be resumed with:", "",
              "```",
              f"python run_multisupport_family_selection.py run --run-dir {run_dir.as_posix()} --phase <phase>",
              "```", "",
              "Blocked families (mathematical audit): geometric, negative_binomial, "
              "gamma, exponential, beta, weibull, student_t, categorical, "
              "inverse_gaussian (see tables/family_compatibility.csv)."]
    atomic_write_text(run_dir / "RESUME.md", "\n".join(lines) + "\n")


def post_process(run_dir: Path, log) -> None:
    for script in POST_SCRIPTS:
        path = _HERE / script
        if not path.exists():
            continue
        proc = subprocess.run([sys.executable, str(path), "--run-dir", str(run_dir)],
                              capture_output=True, text=True)
        log.write(f"{_now()} post {script} rc={proc.returncode}\n"
                  f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}\n")
        log.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--phases", nargs="*", default=ORDER)
    a = ap.parse_args(argv)
    run_dir = a.run_dir
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    log = open(run_dir / "logs" / "driver.log", "a", encoding="utf-8")
    statuses = {}
    for ph in ORDER:
        if ph in [r["phase"] for r in read_csv(run_dir / "completed_cases.csv")]:
            statuses[ph] = phase_status(run_dir, ph)
    for ph in [p for p in ORDER if p in a.phases]:
        if ph.endswith("_r10"):
            base = ph.replace("_r10", "_r5")
            st = statuses.get(base) or phase_status(run_dir, base)
            ok = (st["technical_gate"] == "PASS" and st["fit_failure_fraction"] <= 0.10)
            log.write(f"{_now()} extension rule for {ph}: base={base} "
                      f"gate={st['technical_gate']} frac={st['fit_failure_fraction']:.3f} "
                      f"-> {'RUN' if ok else 'SKIP'}\n")
            log.flush()
            if not ok:
                statuses[ph] = {**phase_status(run_dir, ph), "technical_gate": "SKIPPED_BY_RULE"}
                atomic_write_json(run_dir / "audit" / f"status_snapshot_{ph}.json", statuses[ph])
                continue
        log.write(f"{_now()} start {ph}\n")
        log.flush()
        write_resume(run_dir, statuses, f"running {ph}")
        cmd_run(run_dir, ph, a.workers)
        st = phase_status(run_dir, ph)
        statuses[ph] = st
        atomic_write_json(run_dir / "audit" / f"status_snapshot_{ph}.json", st)
        log.write(f"{_now()} end {ph} gate={st['technical_gate']}\n")
        log.flush()
        post_process(run_dir, log)
        write_resume(run_dir, statuses, f"finished {ph} ({st['technical_gate']})")
        if st["technical_gate"] != "PASS":
            write_resume(run_dir, statuses,
                         f"STOPPED after {ph}: technical gate FAIL "
                         f"(unexpected exceptions or missing cases)")
            log.write(f"{_now()} STOP\n")
            return 1
    write_resume(run_dir, statuses, "all scheduled phases finished")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
