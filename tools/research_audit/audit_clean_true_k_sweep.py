"""Artifact-only independent auditor for the clean true-K n-sweep.

This module does NOT import the runner.  Every structural constant it checks is
restated here as an independent literal, so an artifact set produced by a
mutated runner cannot certify itself.  It runs no EM, writes nothing except its
own report, and never modifies an input artifact.

Findings are classified BLOCKER / HIGH / MEDIUM / LOW.  A BLOCKER or HIGH means
the results must NOT be promoted to a canonical conclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# --- independent restatement of the frozen protocol -----------------------
EXPECTED_PROTOCOL_HASH = "547880a16aef6530cfdf7903c4e32f16062397e0bacc0c109d5c77fb9892ccc0"
EXPECTED_EXPERIMENT_ID = "clean_true_k_asymptotics_20260904"
EXPECTED_GENERATOR_VERSION = "canonical-clean-v1"
EXPECTED_NUMERICS_MODE = "consistent"
EXPECTED_FAMILY_X = "poisson"
EXPECTED_FAMILY_Y = "bernoulli"
EXPECTED_TIER = "A"
EXPECTED_D = 15
EXPECTED_K_TRUE_GRID = (1, 3, 5)
EXPECTED_N_GRID = (50, 75, 100, 150)
EXPECTED_CANDIDATE_K = (1, 2, 3, 4, 5, 6, 7)
EXPECTED_STARTS = (1, 2)
EXPECTED_REPLICATES = {1: 4, 3: 4, 5: 8}
EXPECTED_CELLS = sum(EXPECTED_REPLICATES[k] for k in EXPECTED_K_TRUE_GRID) * len(EXPECTED_N_GRID)
EXPECTED_FITS = EXPECTED_CELLS * len(EXPECTED_CANDIDATE_K) * len(EXPECTED_STARTS)
EXPECTED_CRITERIA = ("S1", "S2", "S3")
TIE_TOLERANCE = np.float64(1e-12)

REQUIRED_FILES = (
    "protocol.json", "manifest.csv", "generator_provenance.csv",
    "fit_results.csv", "selection_matrix.csv", "gram_spectrum.csv",
    "summary.json", "runinfo.json",
)

# Independently restated seed rules.
def data_seed(k_true: int, n: int, replicate: int) -> int:
    return 810000 + k_true * 10000 + n * 10 + replicate


def split_seed(k_true: int, n: int, replicate: int) -> int:
    return 820000 + k_true * 10000 + n * 10 + replicate


def model_seed(k_true: int, n: int, replicate: int, k_est: int, start: int) -> int:
    return 830000 + k_true * 100000 + n * 1000 + replicate * 100 + k_est * 10 + start


@dataclass
class Auditor:
    findings: list[dict[str, str]] = field(default_factory=list)

    def add(self, severity: str, check: str, detail: str) -> None:
        self.findings.append({"severity": severity, "check": check, "detail": detail})

    def require(self, condition: bool, severity: str, check: str, detail: str) -> bool:
        if not condition:
            self.add(severity, check, detail)
        return bool(condition)

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f["severity"] == severity)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(run_dir: Path) -> dict[str, Any]:
    a = Auditor()

    # ---- files -----------------------------------------------------------
    present = {p.name for p in run_dir.iterdir() if p.is_file()}
    for name in REQUIRED_FILES:
        a.require(name in present, "BLOCKER", "required_file", f"missing {name}")
    if a.count("BLOCKER"):
        return finish(a, run_dir, {})
    a.require("failure.json" not in present, "BLOCKER", "failure_marker",
              "failure.json is present: this is not a completed run")

    protocol = read_json(run_dir / "protocol.json")
    runinfo = read_json(run_dir / "runinfo.json")
    summary = read_json(run_dir / "summary.json")
    manifest = read_csv(run_dir / "manifest.csv")
    fits = read_csv(run_dir / "fit_results.csv")
    selection = read_csv(run_dir / "selection_matrix.csv")
    provenance = read_csv(run_dir / "generator_provenance.csv")
    gram = read_csv(run_dir / "gram_spectrum.csv")

    # ---- protocol identity ----------------------------------------------
    a.require(protocol.get("protocol_hash") == EXPECTED_PROTOCOL_HASH, "BLOCKER",
              "protocol_hash", f"got {protocol.get('protocol_hash')}")
    a.require(runinfo.get("protocol_hash") == EXPECTED_PROTOCOL_HASH, "BLOCKER",
              "runinfo_protocol_hash", f"got {runinfo.get('protocol_hash')}")
    a.require(summary.get("protocol_hash") == EXPECTED_PROTOCOL_HASH, "HIGH",
              "summary_protocol_hash", f"got {summary.get('protocol_hash')}")
    a.require(protocol.get("experiment_id") == EXPECTED_EXPERIMENT_ID, "HIGH",
              "experiment_id", f"got {protocol.get('experiment_id')}")

    frozen = protocol.get("protocol", {})
    a.require(frozen.get("generator_version") == EXPECTED_GENERATOR_VERSION, "BLOCKER",
              "generator_version", f"got {frozen.get('generator_version')}")
    a.require(frozen.get("numerics_mode") == EXPECTED_NUMERICS_MODE, "BLOCKER",
              "numerics_mode",
              f"got {frozen.get('numerics_mode')}; the legacy 0.5 lineage is forbidden")
    a.require(frozen.get("family_x") == EXPECTED_FAMILY_X, "HIGH", "family_x",
              f"got {frozen.get('family_x')}")
    a.require(frozen.get("family_y") == EXPECTED_FAMILY_Y, "HIGH", "family_y",
              f"got {frozen.get('family_y')}")
    a.require(frozen.get("d") == EXPECTED_D, "HIGH", "d", f"got {frozen.get('d')}")
    a.require(frozen.get("tier") == EXPECTED_TIER, "HIGH", "tier",
              f"got {frozen.get('tier')}")
    a.require(tuple(frozen.get("k_true_grid", ())) == EXPECTED_K_TRUE_GRID, "HIGH",
              "k_true_grid", f"got {frozen.get('k_true_grid')}")
    a.require(tuple(frozen.get("n_grid", ())) == EXPECTED_N_GRID, "HIGH",
              "n_grid", f"got {frozen.get('n_grid')}")
    a.require(tuple(frozen.get("candidate_k", ())) == EXPECTED_CANDIDATE_K, "HIGH",
              "candidate_k", f"got {frozen.get('candidate_k')}")
    a.require(tuple(frozen.get("starts", ())) == EXPECTED_STARTS, "HIGH",
              "starts", f"got {frozen.get('starts')}")

    # ---- counts ----------------------------------------------------------
    a.require(len(manifest) == EXPECTED_FITS, "BLOCKER", "manifest_row_count",
              f"{len(manifest)} != {EXPECTED_FITS}")
    a.require(len(fits) == EXPECTED_FITS, "BLOCKER", "fit_row_count",
              f"{len(fits)} != {EXPECTED_FITS}")
    a.require(len(provenance) == EXPECTED_CELLS, "HIGH", "provenance_row_count",
              f"{len(provenance)} != {EXPECTED_CELLS}")
    a.require(len(gram) == EXPECTED_CELLS, "HIGH", "gram_row_count",
              f"{len(gram)} != {EXPECTED_CELLS}")
    a.require(len(selection) == EXPECTED_CELLS * len(EXPECTED_CRITERIA), "BLOCKER",
              "selection_row_count",
              f"{len(selection)} != {EXPECTED_CELLS * len(EXPECTED_CRITERIA)}")
    a.require(int(runinfo.get("attempted_fit_count", -1)) == EXPECTED_FITS, "HIGH",
              "attempted_fit_count", f"got {runinfo.get('attempted_fit_count')}")
    a.require(int(runinfo.get("completed_fit_count", -1)) == EXPECTED_FITS, "HIGH",
              "completed_fit_count", f"got {runinfo.get('completed_fit_count')}")

    # ---- fit index integrity --------------------------------------------
    indices = [int(r["fit_index"]) for r in fits]
    a.require(sorted(indices) == list(range(1, EXPECTED_FITS + 1)), "BLOCKER",
              "fit_index_sequence",
              f"duplicates={len(indices) - len(set(indices))}, "
              f"missing={sorted(set(range(1, EXPECTED_FITS + 1)) - set(indices))[:8]}")

    # ---- cell / grid coverage -------------------------------------------
    expected_cells = {(k, n, r)
                      for k in EXPECTED_K_TRUE_GRID
                      for n in EXPECTED_N_GRID
                      for r in range(1, EXPECTED_REPLICATES[k] + 1)}
    fit_cells = {(int(r["k_true"]), int(r["n"]), int(r["replicate"])) for r in fits}
    a.require(fit_cells == expected_cells, "BLOCKER", "cell_coverage",
              f"missing={sorted(expected_cells - fit_cells)[:6]}, "
              f"unexpected={sorted(fit_cells - expected_cells)[:6]}")

    for cell in sorted(expected_cells):
        rows = [r for r in fits
                if (int(r["k_true"]), int(r["n"]), int(r["replicate"])) == cell]
        keys = {(int(r["k_est"]), int(r["start"])) for r in rows}
        wanted = {(k, s) for k in EXPECTED_CANDIDATE_K for s in EXPECTED_STARTS}
        if not a.require(keys == wanted, "BLOCKER", "cell_grid",
                         f"cell {cell} has {len(keys)} of {len(wanted)} (K,start)"):
            break

    # ---- seeds -----------------------------------------------------------
    seed_bad = []
    for r in fits:
        k, n, rep = int(r["k_true"]), int(r["n"]), int(r["replicate"])
        if (int(r["data_seed"]) != data_seed(k, n, rep)
                or int(r["split_seed"]) != split_seed(k, n, rep)
                or int(r["model_seed"]) != model_seed(k, n, rep, int(r["k_est"]),
                                                      int(r["start"]))):
            seed_bad.append(int(r["fit_index"]))
    a.require(not seed_bad, "BLOCKER", "seed_rule",
              f"{len(seed_bad)} fits violate the independently restated seed rule: "
              f"{seed_bad[:8]}")
    model_seeds = [int(r["model_seed"]) for r in fits]
    a.require(len(set(model_seeds)) == len(model_seeds), "HIGH", "model_seed_unique",
              f"{len(model_seeds) - len(set(model_seeds))} duplicates")

    # manifest and fit_results must agree on every seed
    man_by_index = {int(r["fit_index"]): r for r in manifest}
    mismatched = [int(r["fit_index"]) for r in fits
                  if int(man_by_index[int(r["fit_index"])]["model_seed"])
                  != int(r["model_seed"])]
    a.require(not mismatched, "BLOCKER", "manifest_fit_agreement",
              f"{len(mismatched)} model seeds differ from the manifest")

    # ---- masks -----------------------------------------------------------
    for row in provenance:
        n = int(row["n"])
        n_pairs = n * (n - 1) // 2
        n_test = int(row["n_test_pairs"])
        n_train = int(row["n_train_pairs"])
        if not a.require(n_train + n_test == n_pairs, "BLOCKER", "mask_partition",
                         f"n={n} cell: train {n_train} + test {n_test} != {n_pairs}"):
            break
        if not a.require(n_test == int(round(0.20 * n_pairs)), "HIGH", "test_ratio",
                         f"n={n}: test {n_test} != {int(round(0.20 * n_pairs))}"):
            break
    hashes = {(r["K_TRUE"], r["n"], r["replicate"]): r["test_mask_hash"]
              for r in provenance}
    a.require(len(set(hashes.values())) == len(hashes), "MEDIUM", "mask_distinct",
              "two cells share a test mask hash")

    # ---- generator provenance -------------------------------------------
    for row in provenance:
        if not a.require(int(row["F_rank"]) == int(row["K_TRUE"]), "BLOCKER",
                         "generator_rank",
                         f"F_rank {row['F_rank']} != K_TRUE {row['K_TRUE']}"):
            break
        if not a.require(row["normalization_policy"] == "none", "BLOCKER",
                         "generator_normalization",
                         f"got {row['normalization_policy']}"):
            break
        if not a.require(row["link_policy"] == "canonical_no_clipping_fail_fast",
                         "BLOCKER", "generator_link", f"got {row['link_policy']}"):
            break
        if not a.require(row["generator_version"] == EXPECTED_GENERATOR_VERSION,
                         "BLOCKER", "generator_version_row",
                         f"got {row['generator_version']}"):
            break
        if not a.require(abs(float(row["mean_f_row_norm_sq"]) - 0.5) < 1e-9, "HIGH",
                         "signal_matching_x",
                         f"mean ||f_l||^2 = {row['mean_f_row_norm_sq']} != 0.5"):
            break
        k = int(row["K_TRUE"])
        if not a.require(abs(float(row["w_true"]) ** 2 * k - 3.0) < 1e-9, "HIGH",
                         "signal_matching_y",
                         f"w^2 K = {float(row['w_true']) ** 2 * k} != 3"):
            break

    # ---- numerical health -----------------------------------------------
    nan_fits = [r["fit_index"] for r in fits if r["nan_occurred"] != "False"]
    a.add("LOW", "nan_fits", f"{len(nan_fits)} fits reported nan_occurred=True")
    qbic_failed = [r["fit_index"] for r in fits if r["q_bic_failed"] != "False"]
    a.require(not qbic_failed, "HIGH", "q_bic_failed",
              f"{len(qbic_failed)} fits failed Q/BIC computation: {qbic_failed[:8]}")
    nonfinite = [r["fit_index"] for r in fits
                 if not all(math.isfinite(float(r[c])) for c in
                            ("heldout_mean_log_score", "q_strict", "s2_q_based",
                             "s3_plugin_conditional"))]
    a.require(not nonfinite, "BLOCKER", "criterion_finite",
              f"{len(nonfinite)} fits carry a non-finite criterion: {nonfinite[:8]}")

    # ---- no retries / replacements --------------------------------------
    for key in ("retry_count", "replacement_fits_executed", "seed_rescue_count",
                "tolerance_relaxations"):
        a.require(int(runinfo.get(key, -1)) == 0, "HIGH", key,
                  f"got {runinfo.get(key)}")
    a.require(runinfo.get("resumed") is False, "BLOCKER", "resumed",
              f"got {runinfo.get('resumed')}")

    # ---- INDEPENDENT recomputation of the selection ----------------------
    recomputed: dict[tuple[str, int, int, int], int] = {}
    by_cell: dict[tuple[int, int, int], list[dict[str, str]]] = {}
    for r in fits:
        by_cell.setdefault(
            (int(r["k_true"]), int(r["n"]), int(r["replicate"])), []).append(r)

    for cell, rows in by_cell.items():
        for name, column, higher_is_better in (
                ("S1", "heldout_mean_log_score", True),
                ("S2", "s2_q_based", False),
                ("S3", "s3_plugin_conditional", False)):
            means: dict[int, np.float64] = {}
            for k_est in EXPECTED_CANDIDATE_K:
                vals = [float(r[column]) for r in rows if int(r["k_est"]) == k_est]
                if len(vals) != len(EXPECTED_STARTS):
                    a.add("BLOCKER", "recompute_starts",
                          f"{name} {cell} K={k_est} has {len(vals)} starts")
                    means = {}
                    break
                signed = np.asarray(vals if higher_is_better else [-v for v in vals],
                                    dtype=np.float64)
                means[k_est] = np.mean(signed, dtype=np.float64)
            if not means:
                continue
            best = max(means.values())
            ties = sorted(k for k, v in means.items() if best - v <= TIE_TOLERANCE)
            recomputed[(name, *cell)] = min(ties)

    disagreements = []
    for row in selection:
        key = (row["criterion"], int(row["K_TRUE"]), int(row["n"]),
               int(row["replicate"]))
        if key not in recomputed:
            a.add("BLOCKER", "selection_key", f"unrecomputable row {key}")
            continue
        if recomputed[key] != int(row["selected_k"]):
            disagreements.append((key, recomputed[key], int(row["selected_k"])))
    a.require(not disagreements, "BLOCKER", "selection_recomputation",
              f"{len(disagreements)} rows disagree with the independent "
              f"recomputation: {disagreements[:5]}")

    a.require({r["criterion"] for r in selection} == set(EXPECTED_CRITERIA), "HIGH",
              "criteria_present",
              f"got {sorted({r['criterion'] for r in selection})}")

    # selection labels must match their own signed errors
    label_bad = [(row["criterion"], row["K_TRUE"], row["n"], row["replicate"])
                 for row in selection
                 if row["label"] != ("exact" if int(row["signed_error"]) == 0 else
                                     ("over" if int(row["signed_error"]) > 0
                                      else "under"))
                 or int(row["signed_error"]) != int(row["selected_k"]) - int(row["K_TRUE"])]
    a.require(not label_bad, "HIGH", "selection_labels",
              f"{len(label_bad)} rows have an inconsistent label/error")

    # ---- summary consistency --------------------------------------------
    for name in EXPECTED_CRITERIA:
        rows = [r for r in selection if r["criterion"] == name]
        exact = sum(1 for r in rows if r["label"] == "exact")
        reported = summary.get("criteria", {}).get(name, {}).get("total_exact")
        a.require(reported == exact, "HIGH", "summary_exact_count",
                  f"{name}: summary says {reported}, selection_matrix says {exact}")

    # ---- S4 diagnostic: recorded, never used to select --------------------
    a.require(not any("selected_k" in k for k in (gram[0].keys() if gram else ())),
              "BLOCKER", "gram_selects_no_k",
              "the Gram diagnostic must not carry a selected K")
    psd_violations = sum(1 for r in gram
                         if r["min_eigenvalue"] not in ("", "None")
                         and float(r["min_eigenvalue"]) < 0)
    a.add("LOW", "gram_psd_violations",
          f"{psd_violations} of {len(gram)} cells have a negative minimum "
          f"eigenvalue in the estimated Gram (expected; see U7)")

    return finish(a, run_dir, {
        "fits": len(fits), "cells": len(by_cell), "selection_rows": len(selection),
        "nan_fits": len(nan_fits),
    })


def finish(a: Auditor, run_dir: Path, stats: dict[str, Any]) -> dict[str, Any]:
    verdict = "PASS" if a.count("BLOCKER") == 0 and a.count("HIGH") == 0 else "FAIL"
    return {
        "audit_version": "clean-true-k-audit-v1",
        "run_dir": str(run_dir),
        "verdict": verdict,
        "blocker_count": a.count("BLOCKER"),
        "high_count": a.count("HIGH"),
        "medium_count": a.count("MEDIUM"),
        "low_count": a.count("LOW"),
        "findings": a.findings,
        "stats": stats,
        "expected_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "expected_fits": EXPECTED_FITS,
        "expected_cells": EXPECTED_CELLS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)

    report = audit(args.run_dir)
    if args.write_report:
        target = args.run_dir / "audit_report.json"
        if target.exists():
            report["note"] = "audit_report.json already exists and was NOT overwritten"
        else:
            target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
