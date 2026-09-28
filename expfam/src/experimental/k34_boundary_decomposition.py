"""Phase 9Y (Issue #124): read-only K3 -> K4 boundary decomposition.

Post-hoc exploratory characterization of the committed Phase 9X K_true = 4
fits (and, as a predeclared context comparator, the Phase 9K K_true = 3
rep01..rep10 fits). Reads committed CSV/JSON only: no EM, refit, family
search, Candidate-B evaluation or theta optimization is imported or run.

Sign convention: Delta34 = C(K=4) - C(K=3); negative => K4 preferred.

C_Lap (existing Phase 9P/9T convention, run_w_sensitivity.decomposition,
generalized from K2->3 to K3->4):
    -2 ell_Lap = D_mode + V,  D_mode = -2 phi - ||Z_hat||^2 - nK ln(2 pi),
                              V = ||Z_hat||^2 + log|H|
    C_Lap = -2 ell_Lap + d_K ln N
    Delta34_Lap = -fit_gain_Lap_34 + volume_increment_34 + param_increment_34
    fit_gain_Lap_34 = D_mode(3) - D_mode(4), volume_increment_34 = V(4) - V(3)
C_Q (run_joint_family_k_selection.cq_decomposition):
    C_Q = D_K + P_Z + P_theta,  D_K = -2 (Q_X + Q_Y), P_Z = -2 Q_Z,
    P_theta = num_params ln n
    Delta34_Q = -fit_gain_Q_34 + P_Z_increment_34 + param_increment_34
    fit_gain_Q_34 = D_K(3) - D_K(4)
Threshold margins (positive => K4 beats K3):
    margin_Lap = fit_gain_Lap_34 - (volume_increment_34 + param_increment)
    margin_Q   = fit_gain_Q_34 - (P_Z_increment_34 + param_increment)
Reconstruction: |reconstructed - direct| <= 1e-10 for every replicate.
Lineage E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import csv                                                          # noqa: E402

REPO_ROOT = _HERE.parents[2]
PHASE9X_MERGE = "1cb316bdde7b05b9c17c73eafede0d2ce367e536"
X_ROOT = "expfam/results/matched_k_true_sensitivity/phase9x_20260928"
K9_ROOT = "expfam/results/lap_vs_cq_20/phase9k_20260928"
SOURCES = {
    "ktrue4_laplace": f"{X_ROOT}/K4/laplace_by_k.csv",
    "ktrue4_cq": f"{X_ROOT}/K4/cq_decomposition.csv",
    "combined_rows": f"{X_ROOT}/combined/dataset_rows.csv",
    "ktrue3_laplace": f"{K9_ROOT}/laplace_by_k.csv",
    "ktrue3_cq": f"{K9_ROOT}/cq_decomposition.csv",
}
REPLICATES = tuple(f"rep{r:02d}" for r in range(1, 11))
N_OBS, D = 75, 12
LAPLACE_N = 75.0
RESIDUAL_TOL = 1e-10
COMPONENTS = ("delta34_Lap", "fit_gain_Lap_34", "volume_increment_34",
              "logdet_increment_34", "zhat_sq_increment_34",
              "param_increment_Lap", "margin_Lap", "delta34_Q",
              "fit_gain_Q_34", "P_Z_increment_34", "param_increment_Q",
              "margin_Q")


def d_K(k: int, d: int, n_gaussian_selected: int) -> int:
    """Same count as run_laplace_pilot.d_K / laplace_k_criterion
    .loading_parameter_count (K d - K(K-1)/2 + n_gaussian_selected);
    reimplemented here so no inference module is imported. A test checks
    equality with the original functions."""
    return k * d - k * (k - 1) // 2 + int(n_gaussian_selected)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sources(repo: Path = REPO_ROOT) -> dict[str, str]:
    """Bind every source to its bytes at the Phase 9X merge commit."""
    hashes = {}
    for name, rel in SOURCES.items():
        disk = (repo / rel).read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{PHASE9X_MERGE}:{rel}"], cwd=repo,
            capture_output=True, check=True).stdout
        if disk.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise SystemExit(f"{rel} differs from the Phase 9X merge: blocked")
        hashes[name] = _sha(committed)
    return hashes


def _read(repo: Path, rel: str) -> list[dict[str, str]]:
    with (repo / rel).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _d_mode(r: dict[str, str], k: int) -> float:
    return (-2 * float(r["phi_at_mode"]) - float(r["Z_hat_sq_norm"])
            - N_OBS * k * math.log(2 * math.pi))


def _vol(r: dict[str, str]) -> float:
    return float(r["Z_hat_sq_norm"]) + float(r["logdet_H"])


def decompose(lap_rows, cq_rows, k_hat: dict[str, tuple]) -> list[dict]:
    lap = {(r["replicate"], int(r["k"])): r for r in lap_rows}
    cq = {(r["replicate"], int(r["k"])): r for r in cq_rows}
    out = []
    for rep in REPLICATES:
        a, b = lap[(rep, 3)], lap[(rep, 4)]
        qa, qb = cq[(rep, 3)], cq[(rep, 4)]
        if not (a["laplace_status"] == b["laplace_status"] == "OK"):
            out.append({"replicate": rep, "available": False})
            continue
        direct_lap = float(b["C_Lap"]) - float(a["C_Lap"])
        fit_lap = _d_mode(a, 3) - _d_mode(b, 4)
        vol = _vol(b) - _vol(a)
        param_lap = float(b["parameter_term"]) - float(a["parameter_term"])
        param_lap_dk = (d_K(4, D, int(b["n_gaussian_selected"]))
                        - d_K(3, D, int(a["n_gaussian_selected"]))) \
            * math.log(LAPLACE_N)
        direct_q = float(b["C_Q"]) - float(a["C_Q"])
        fit_q = float(qa["D_K"]) - float(qb["D_K"])
        pz = float(qb["P_Z"]) - float(qa["P_Z"])
        param_q = float(qb["P_theta"]) - float(qa["P_theta"])
        param_q_dk = (int(qb["num_params"]) - int(qa["num_params"])) \
            * math.log(N_OBS)
        out.append({
            "replicate": rep, "available": True,
            "K_hat_Lap": k_hat[rep][0], "K_hat_Q": k_hat[rep][1],
            "delta34_Lap": direct_lap, "fit_gain_Lap_34": fit_lap,
            "volume_increment_34": vol,
            "logdet_increment_34": float(b["logdet_H"]) - float(a["logdet_H"]),
            "zhat_sq_increment_34": float(b["Z_hat_sq_norm"])
            - float(a["Z_hat_sq_norm"]),
            "param_increment_Lap": param_lap,
            "param_increment_Lap_from_dK": param_lap_dk,
            "d_K_3": d_K(3, D, int(a["n_gaussian_selected"])),
            "d_K_4": d_K(4, D, int(b["n_gaussian_selected"])),
            "margin_Lap": fit_lap - (vol + param_lap),
            "residual_Lap": abs((-fit_lap + vol + param_lap) - direct_lap),
            "delta34_Q": direct_q, "fit_gain_Q_34": fit_q,
            "P_Z_increment_34": pz, "param_increment_Q": param_q,
            "param_increment_Q_from_num_params": param_q_dk,
            "num_params_3": int(qa["num_params"]),
            "num_params_4": int(qb["num_params"]),
            "margin_Q": fit_q - (pz + param_q),
            "residual_Q": abs((-fit_q + pz + param_q) - direct_q),
            "group_Lap": ("Lap_K4_over_K3" if direct_lap < 0 else
                          "Lap_K3_over_K4" if direct_lap > 0 else "tie"),
            "group_Q": ("Q_K4_over_K3" if direct_q < 0 else
                        "Q_K3_over_K4" if direct_q > 0 else "tie"),
        })
    return out


def _dist(values) -> dict[str, Any]:
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "min": min(v), "median": statistics.median(v),
            "max": max(v), "negative": sum(x < 0 for x in v),
            "positive": sum(x > 0 for x in v), "zero": sum(x == 0 for x in v)}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["available"]]
    groups = {}
    for crit in ("Lap", "Q"):
        for label in sorted({r[f"group_{crit}"] for r in ok}):
            members = [r for r in ok if r[f"group_{crit}"] == label]
            groups[label] = {"n": len(members),
                             "replicates": [r["replicate"] for r in members],
                             "component_medians": {
                                 c: statistics.median(r[c] for r in members)
                                 for c in COMPONENTS}}
    return {"available": len(ok), "requested": len(rows),
            "components": {c: _dist([r[c] for r in ok]) for c in COMPONENTS},
            "max_residual_Lap": max((r["residual_Lap"] for r in ok),
                                    default=None),
            "max_residual_Q": max((r["residual_Q"] for r in ok),
                                  default=None),
            "descriptive_groups_exploratory": groups}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; never overwritten")
    hashes = verify_sources()
    combined = _read(REPO_ROOT, SOURCES["combined_rows"])
    k_hat = {int(kt): {r["replicate"]: (int(r["K_hat_Lap"]),
                                        int(r["K_hat_Q"]))
                       for r in combined if r["K_true"] == kt}
             for kt in ("3", "4")}
    k4 = decompose(_read(REPO_ROOT, SOURCES["ktrue4_laplace"]),
                   _read(REPO_ROOT, SOURCES["ktrue4_cq"]), k_hat[4])
    k3 = decompose(
        [r for r in _read(REPO_ROOT, SOURCES["ktrue3_laplace"])
         if r["replicate"] in REPLICATES],
        [r for r in _read(REPO_ROOT, SOURCES["ktrue3_cq"])
         if r["replicate"] in REPLICATES], k_hat[3])
    s4, s3 = summarize(k4), summarize(k3)
    checks = {}
    for name, rows in (("ktrue4", k4), ("ktrue3_context", k3)):
        ok = [r for r in rows if r["available"]]
        checks[name] = {
            "available": len(ok),
            "max_residual_Lap": max(r["residual_Lap"] for r in ok),
            "max_residual_Q": max(r["residual_Q"] for r in ok),
            "all_within_tol": all(r["residual_Lap"] <= RESIDUAL_TOL
                                  and r["residual_Q"] <= RESIDUAL_TOL
                                  for r in ok),
            "param_increment_Lap_equals_dK": all(
                abs(r["param_increment_Lap"]
                    - r["param_increment_Lap_from_dK"]) <= 1e-10 for r in ok),
            "param_increment_Q_equals_num_params": all(
                abs(r["param_increment_Q"]
                    - r["param_increment_Q_from_num_params"]) <= 1e-10
                for r in ok),
            "num_params_equals_d_K": all(
                r["num_params_3"] == r["d_K_3"] and r["num_params_4"] == r["d_K_4"]
                for r in ok)}
    checks["tolerance"] = RESIDUAL_TOL
    full = (s4["available"] == 10 and checks["ktrue4"]["all_within_tol"]
            and checks["ktrue4"]["param_increment_Lap_equals_dK"]
            and checks["ktrue4"]["param_increment_Q_equals_num_params"])
    decision = ("K34_BOUNDARY_DECOMPOSITION_CHARACTERIZED" if full else
                "K34_BOUNDARY_DECOMPOSITION_PARTIAL" if s4["available"]
                else "K34_BOUNDARY_DECOMPOSITION_BLOCKED")
    derived = {
        "P_Z_per_dimension": N_OBS * (1 + math.log(2 * math.pi)),
        "P_Z_formula": "P_Z = -2 Q_Z = nK ln(2 pi var_z) + mean_l "
                       "||Z_l||^2 / var_z; with var_z = 1 and the scale_Z "
                       "convention (mean_l ||Z_l||^2 = nK) this is "
                       "nK (1 + ln 2 pi)",
        "d_K_3_to_4_loading": (4 * D - 6) - (3 * D - 3),
        "param_increment_ln75": ((4 * D - 6) - (3 * D - 3)) * math.log(75),
    }
    from_git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                cwd=REPO_ROOT, capture_output=True,
                                text=True).stdout.strip())
    args.out.mkdir(parents=True)

    def write_csv(path, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def write_json(path, obj):
        path.write_text(json.dumps(obj, indent=2) + "\n", "utf-8")

    write_json(args.out / "source_provenance.json", {
        "phase9x_merge": PHASE9X_MERGE, "sources": SOURCES,
        "sha256_at_merge": hashes, "code_sha": from_git, "git_dirty": dirty,
        "em_executions": 0, "refits": 0, "candidate_b_re_evaluations": 0,
        "mode": "post-hoc exploratory; committed values only"})
    write_csv(args.out / "ktrue4_rows.csv", k4)
    write_csv(args.out / "ktrue3_context_rows.csv", k3)
    write_json(args.out / "ktrue4_summary.json",
               {**s4, "derived_prior_context": derived, "DECISION": decision})
    write_json(args.out / "ktrue3_context_summary.json", {
        **s3, "note": "context comparator only; same seed labels are not the "
                      "same datasets; no paired inference"})
    write_json(args.out / "reconstruction_checks.json", checks)
    print(json.dumps({"available": s4["available"], "checks": checks,
                      "DECISION": decision}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
