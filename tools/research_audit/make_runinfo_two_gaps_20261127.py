"""Write runinfo.csv and states_sha256_manifest.csv for the two-gaps experiments (2026-11-27).

Reads only per_fit.csv, run_events.jsonl and protocol.json of expA/expB, plus the environment.
Usage: python tools/research_audit/make_runinfo_two_gaps_20261127.py
"""
import csv
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy
import scipy

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "expfam/results/k_selection/two_gaps_20261127"


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    info, man = [], []
    for exp in ("A", "B"):
        d = BASE / f"exp{exp}"
        proto = json.loads((d / "protocol/protocol.json").read_text(encoding="utf-8"))
        ev = [json.loads(x) for x in (d / "run/run_events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        start = next(e for e in ev if e["event"] == "start")
        fin = [e for e in ev if e["event"] == "finished"]
        rows = rd(d / "run/per_fit.csv")
        main_rows = [r for r in rows if r["phase"] == "main"]
        st = Counter(r["status"] for r in main_rows)
        lap = Counter(r.get("lap_status", "") for r in main_rows if "lap_status" in r)
        pilot = rd(d / "pilot/per_fit.csv") if (d / "pilot/per_fit.csv").exists() else []
        info.append({
            "experiment": exp, "lineage": "E-scalar (DualExpFamLSMConsistent, consistent, fail_fast)",
            "kind": "EXPLORATORY", "protocol_frozen_at": proto["frozen_at"],
            "protocol_git_head": proto["git_head"], "run_git_head": start["git_head"],
            "run_start": start["time"], "run_finished": fin[-1]["time"] if fin else "",
            "wall_s": fin[-1]["wall_s"] if fin else "", "workers": start["workers"],
            "deadline": start["deadline"], "n_jobs_frozen": proto["n_jobs"],
            "main_fits_rows": len(main_rows), "main_ok": st.get("OK", 0),
            "main_failed": len(main_rows) - st.get("OK", 0),
            "lap_status_counts": json.dumps(dict(lap)) if lap else "",
            "pilot_fits": len(pilot), "pilot_ok": sum(r["status"] == "OK" for r in pilot),
            "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "blas_threads": 1,
            "runinfo_written_at_git_head": git("rev-parse", "HEAD"),
        })
        for r in main_rows:
            if r["status"] == "OK":
                man.append({"experiment": exp, "fit_id": r["fit_id"],
                            "path": f"exp{exp}/run/states/{r['state_file']}", "sha256": r["state_sha256"]})
    with open(BASE / "runinfo.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(info[0]))
        w.writeheader()
        w.writerows(info)
    with open(BASE / "states_sha256_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "fit_id", "path", "sha256"])
        w.writeheader()
        w.writerows(man)
    print(f"runinfo rows {len(info)}, manifest rows {len(man)}")


if __name__ == "__main__":
    main()
