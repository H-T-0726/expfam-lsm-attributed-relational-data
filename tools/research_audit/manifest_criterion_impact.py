"""Evidence and lineage manifest for the criterion impact validation (sha256 of inputs and outputs)."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import criterion_impact_protocol as cp  # noqa: E402

RES = ROOT / cp.RESULTS_REL


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = RES / "evidence_and_lineage_manifest.csv"
    if out.exists():
        raise SystemExit("refusing to overwrite manifest")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rows = []
    code = ["tools/research_audit/criterion_impact_protocol.py", "tools/research_audit/run_criterion_impact.py",
            "tools/research_audit/analyze_criterion_impact.py", "tools/research_audit/test_elbo_k_criterion.py",
            "expfam/src/experimental/elbo_k_criterion.py", "expfam/src/experimental/laplace_k_criterion.py",
            "tools/research_audit/fair_common_evaluator.py", "tools/research_audit/run_fair_comparison.py",
            "tools/research_audit/evaluate_fair_comparison.py"]
    for c in code:
        rows.append({"role": "code", "path": c, "sha256": sha(ROOT / c), "lineage": "E (prototype)",
                     "phase": "both", "note": "unchanged existing code" if "fair" in c or "laplace" in c else "new"})
    rows.append({"role": "input (existing states)", "path": "expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106/criterion_scores.csv",
                 "sha256": sha(ROOT / "expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106/criterion_scores.csv"),
                 "lineage": "E (prototype)", "phase": "C (EXPLORATORY)",
                 "note": "PR #160 E2 448 states (npz under D:/tento/kennkyu_artifacts, not in git); cross-checked 2240/2240"})
    for p in sorted(RES.rglob("*")):
        if p.is_file() and p.name != out.name and "confirm_fits" not in p.parts:
            ph = "E (CONFIRMATORY)" if "confirm" in p.name else ("C (EXPLORATORY)" if "existing" in p.name or "pr160" in p.name else "both")
            rows.append({"role": "output", "path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p),
                         "lineage": "E (prototype)", "phase": ph, "note": ""})
    nfits = len(list((RES / "confirm_fits").glob("*.json")))
    rows.append({"role": "output (fit records)", "path": cp.RESULTS_REL + "/confirm_fits/*.json", "sha256": "",
                 "lineage": "E (prototype)", "phase": "E (CONFIRMATORY)",
                 "note": f"{nfits} JSON rows with npz sha256; npz states under {cp.STATE_ROOT} (not in git)"})
    df = pd.DataFrame(rows)
    df["git_head_at_manifest"] = head
    df["protocol_hash"] = cp.protocol_hash()
    df.to_csv(out, index=False, lineterminator="\n")
    print(len(df), "rows")


if __name__ == "__main__":
    main()
