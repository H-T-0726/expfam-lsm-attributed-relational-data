"""Phase 9M (Issue #98): saved-theta stationarity and one-step local-ordering
diagnostic for the implemented Candidate-B objective.

Objective (implemented, not the exact marginal):
    ell_Lap(theta, K) = laplace_k_criterion.laplace_log_observed(...)
evaluated at the Phase 9K saved states. Nothing here runs EM, an E-step,
family selection or a refit, and no new data is drawn: X, Y are regenerated
from the frozen data seeds with the Phase 9K generator.

theta coordinates (var_z = 1 fixed, family assignment fixed):
  - F on the horizontal space: vec(F) = vec(F0) + B_H alpha, B_H the
    orthonormal complement of the rotation tangent {F A : Aᵀ = −A}
  - Gaussian-X columns: variance_l = variance_l0 · exp(beta_l)
  - w0, w
Every evaluation starts the joint-mode solver at the same Z_hat0 (the base
mode found from the saved Z_est); solver tolerances are the Phase 9H
defaults. A non-OK status makes that value unavailable; nothing is rescued.

Frozen numerics: central differences with h (F, log-variance: 1e-4; w0, w:
1e-4·max(1,|·|)) and h/2 (primary); directional curvature with
eps_dir = 1e-3; one step along u = g/‖g‖ with t* = ‖g‖/|curvature| tried at
factors 1, 1/2, …, 1/64, first finite increase accepted.

This is a one-dimensional local score-direction diagnostic, not a refit
and not a new criterion. Lineage E.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.linalg import null_space

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import laplace_k_criterion as lk                                   # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_lap_vs_cq_20 as pc                                      # noqa: E402
from model_dual_expfam_consistent import (                         # noqa: E402
    DualExpFamLSMPerColumnConsistent,
)

DIAGNOSTIC_VERSION = "theta-stationarity-v1"
K_SCOPE = (2, 3, 4)
H_BASE = 1e-4
EPS_DIR = 1e-3
STEP_FACTORS = (1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64)
RECONSTRUCTION_RTOL = 1e-9           # "to numerical precision"
SOURCE_DIR = Path("expfam/results/lap_vs_cq_20/phase9k_20260928")


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def rotation_tangent(F: np.ndarray) -> np.ndarray:
    """Columns vec(F A_q) for the skew generators A_q (row-major vec)."""

    d, k = F.shape
    cols = []
    for a in range(k):
        for b in range(a + 1, k):
            A = np.zeros((k, k))
            A[a, b], A[b, a] = 1.0, -1.0
            cols.append((F @ A).ravel())
    return np.array(cols).T if cols else np.zeros((d * k, 0))


def split_basis(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal vertical basis B_V and horizontal complement B_H."""

    V = rotation_tangent(F)
    if V.shape[1]:
        Q, _ = np.linalg.qr(V)
        rank = np.linalg.matrix_rank(V)
        B_V = Q[:, :rank]
    else:
        B_V = np.zeros((F.size, 0))
    B_H = null_space(B_V.T) if B_V.shape[1] else np.eye(F.size)
    return B_V, B_H


# --------------------------------------------------------------------------
# state reconstruction and the objective
# --------------------------------------------------------------------------

class SavedState:
    def __init__(self, entry: dict[str, Any], X: np.ndarray, Y: np.ndarray):
        self.entry = entry
        self.replicate, self.k = entry["replicate"], int(entry["k"])
        self.X, self.Y = X, Y
        self.families = list(entry["selected_assignment"])
        self.F0 = np.array(entry["F"], float)
        self.var0 = np.array(entry["sigma_diag"], float)
        self.w0_0, self.w_0 = float(entry["w0"]), float(entry["w"])
        self.var_z = float(entry["var_z"])
        self.Z_est = np.array(entry["Z_est"], float)
        self.gauss = [l for l, f in enumerate(self.families) if f == "gaussian"]
        self.B_V, self.B_H = split_basis(self.F0)
        self.model = DualExpFamLSMPerColumnConsistent(
            n=X.shape[0], d=X.shape[1], k=self.k, L=1,
            family_x_list=self.families, family_y="bernoulli")
        self.model.initialize_params(true_params=None, seed=0)
        self.Z_hat0: np.ndarray | None = None

    @property
    def n_horizontal(self) -> int:
        return self.B_H.shape[1]

    @property
    def n_coords(self) -> int:
        return self.n_horizontal + len(self.gauss) + 2

    def set_theta(self, theta: np.ndarray | None = None,
                  vertical: np.ndarray | None = None) -> None:
        theta = np.zeros(self.n_coords) if theta is None else theta
        nh, ng = self.n_horizontal, len(self.gauss)
        vec = self.F0.ravel() + self.B_H @ theta[:nh]
        if vertical is not None:
            vec = vec + self.B_V @ vertical
        var = self.var0.copy()
        var[self.gauss] = self.var0[self.gauss] * np.exp(theta[nh:nh + ng])
        self.model.params.update({
            "F": vec.reshape(self.F0.shape), "sigma": np.diag(var),
            "w0": self.w0_0 + theta[nh + ng],
            "w": self.w_0 + theta[nh + ng + 1], "var_z": self.var_z})

    def step_sizes(self) -> np.ndarray:
        nh, ng = self.n_horizontal, len(self.gauss)
        return np.concatenate([np.full(nh, H_BASE), np.full(ng, H_BASE),
                               [H_BASE * max(1.0, abs(self.w0_0))],
                               [H_BASE * max(1.0, abs(self.w_0))]])

    def ell(self, theta: np.ndarray | None = None,
            vertical: np.ndarray | None = None,
            Z0: np.ndarray | None = None) -> float:
        """Implemented ell_Lap at theta; NaN unless the status is OK."""

        self.set_theta(theta, vertical)
        start = self.Z_hat0 if Z0 is None else Z0
        try:
            res = lk.laplace_log_observed(self.model, self.X, self.Y, start)
        except (FloatingPointError, np.linalg.LinAlgError):
            return float("nan")
        return res.log_observed if res.ok else float("nan")

    def base(self) -> lk.LaplaceResult:
        """Candidate B from the saved Z_est; sets Z_hat0."""

        self.set_theta()
        res = lk.laplace_log_observed(self.model, self.X, self.Y, self.Z_est)
        self.Z_hat0 = res.Z_hat
        return res


# --------------------------------------------------------------------------
# derivatives and the one-step diagnostic
# --------------------------------------------------------------------------

def central_gradient(f: Callable[[np.ndarray], float], dim: int,
                     steps: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.full(dim, np.nan)
    unavailable = 0
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = steps[i]
        plus, minus = f(e), f(-e)
        if math.isfinite(plus) and math.isfinite(minus):
            g[i] = (plus - minus) / (2 * steps[i])
        else:
            unavailable += 1
    return g, unavailable


def one_step(f: Callable[[np.ndarray], float], g: np.ndarray,
             ell_base: float) -> dict[str, Any]:
    out: dict[str, Any] = {"directional_gradient": float("nan"),
                           "curvature": float("nan"), "t_star": float("nan"),
                           "accepted_factor": float("nan"),
                           "delta_ell_one_step": float("nan"),
                           "status": "", "reason": ""}
    if not np.all(np.isfinite(g)):
        out.update(status="UNAVAILABLE", reason="gradient component unavailable")
        return out
    norm = float(np.linalg.norm(g))
    out["directional_gradient"] = norm
    if norm == 0.0:
        out.update(status="UNAVAILABLE", reason="zero gradient")
        return out
    u = g / norm
    plus, minus = f(EPS_DIR * u), f(-EPS_DIR * u)
    if not (math.isfinite(plus) and math.isfinite(minus)):
        out.update(status="UNAVAILABLE", reason="curvature evaluation unavailable")
        return out
    curv = (plus - 2 * ell_base + minus) / EPS_DIR ** 2
    out["curvature"] = curv
    if curv >= 0:
        out.update(status="UNAVAILABLE", reason="non-negative curvature")
        return out
    t_star = -norm / curv
    out["t_star"] = t_star
    for factor in STEP_FACTORS:
        value = f(factor * t_star * u)
        if math.isfinite(value) and value > ell_base:
            out.update(status="ACCEPTED", accepted_factor=factor,
                       delta_ell_one_step=value - ell_base)
            return out
    out.update(status="NO_ACCEPTED_STEP",
               reason="no factor gave a finite increase")
    return out


def diagnose_state(state: SavedState, committed: dict[str, float]
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    base = state.base()
    ell_rec = base.log_observed if base.ok else float("nan")
    d_K = committed["d_K"]
    c_rec = -2.0 * ell_rec + d_K * math.log(committed["N"])
    rec_ok = (base.ok and math.isclose(ell_rec, committed["laplace"],
                                       rel_tol=RECONSTRUCTION_RTOL)
              and math.isclose(c_rec, committed["C_Lap"],
                               rel_tol=RECONSTRUCTION_RTOL))
    ell_base = state.ell()                      # from Z_hat0
    steps = state.step_sizes()
    f = state.ell
    g_h, un_h = central_gradient(f, state.n_coords, steps)
    g, un = central_gradient(f, state.n_coords, steps / 2)       # primary
    nh, ng = state.n_horizontal, len(state.gauss)
    fv = lambda v: state.ell(vertical=v)                       # noqa: E731
    g_v, un_v = central_gradient(fv, state.B_V.shape[1],
                                 np.full(state.B_V.shape[1], H_BASE / 2))
    diff = np.abs(g_h - g)
    gF = g[:nh]
    grad = {
        "replicate": state.replicate, "k": state.k,
        "reconstruction_ok": rec_ok,
        "ell_committed": committed["laplace"], "ell_reconstructed": ell_rec,
        "ell_base_from_Z_hat0": ell_base,
        "C_Lap_committed": committed["C_Lap"], "C_Lap_reconstructed": c_rec,
        "identifiable_coords": state.n_coords,
        "horizontal_F_coords": nh, "vertical_F_coords": state.B_V.shape[1],
        "log_variance_coords": ng,
        "grad_L2": float(np.linalg.norm(g)),
        "grad_inf": float(np.max(np.abs(g))),
        "gradF_L2": float(np.linalg.norm(gF)),
        "gradF_inf": float(np.max(np.abs(gF))),
        "grad_logvar_L2": float(np.linalg.norm(g[nh:nh + ng])),
        "grad_logvar_inf": float(np.max(np.abs(g[nh:nh + ng])))
        if ng else 0.0,
        "abs_grad_w0": float(abs(g[nh + ng])),
        "abs_grad_w": float(abs(g[nh + ng + 1])),
        "h_vs_h2_max_abs": float(np.max(diff)),
        "h_vs_h2_normalized": float(np.linalg.norm(g_h - g)
                                    / max(np.linalg.norm(g), 1e-300)),
        "unavailable_components": int(un + un_h),
        "vertical_score_L2": float(np.linalg.norm(g_v)),
        "horizontal_score_L2": float(np.linalg.norm(gF)),
        "vertical_over_total_F": float(
            np.linalg.norm(g_v) / max(math.hypot(np.linalg.norm(g_v),
                                                 np.linalg.norm(gF)),
                                      1e-300)),
        "vertical_unavailable": int(un_v),
    }
    step = one_step(f, g, ell_base)
    step.update({"replicate": state.replicate, "k": state.k,
                 "ell_base": ell_base, "C_Lap_committed": committed["C_Lap"],
                 "C_Lap_one_step_diag": committed["C_Lap"]
                 - 2 * step["delta_ell_one_step"]})
    return grad, step


# --------------------------------------------------------------------------
# pairwise summary
# --------------------------------------------------------------------------

def pairwise(paired_summary: dict[str, Any],
             steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {(s["replicate"], s["k"]): s for s in steps}
    out = []
    for d in paired_summary["datasets"]:
        rep = d["replicate"]
        curve = {int(k): float(v) for k, v in d["C_Lap"].items()}
        best, second, gap = d["lap_best_k"], d["lap_second_k"], d["lap_gap"]
        row: dict[str, Any] = {"replicate": rep, "best_k": best,
                               "second_k": second, "original_gap": gap,
                               "in_scope": best in K_SCOPE and second in K_SCOPE}
        sb, ss = by.get((rep, best)), by.get((rep, second))
        if row["in_scope"] and sb and ss and sb["status"] == "ACCEPTED" \
                and ss["status"] == "ACCEPTED":
            adj = (ss["C_Lap_one_step_diag"] - sb["C_Lap_one_step_diag"])
            row.update({"adjusted_gap": adj,
                        "ordering": "retained" if adj > 0 else
                        "flipped" if adj < 0 else "tie"})
        else:
            row.update({"adjusted_gap": float("nan"),
                        "ordering": "unavailable"})
        s2, s3 = by.get((rep, 2)), by.get((rep, 3))
        row["delta_Lap_23"] = curve[3] - curve[2]
        if s2 and s3 and s2["status"] == s3["status"] == "ACCEPTED":
            row["delta_Lap_23_one_step"] = row["delta_Lap_23"] - 2 * (
                s3["delta_ell_one_step"] - s2["delta_ell_one_step"])
        else:
            row["delta_Lap_23_one_step"] = float("nan")
        out.append(row)
    return out


def _stats(values):
    v = [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
    return ({"n": len(v), "min": min(v), "median": statistics.median(v),
             "max": max(v)} if v else {"n": 0})


def summarize(grads, steps, pairs) -> dict[str, Any]:
    by_k = lambda rows, key: {                                  # noqa: E731
        k: _stats([r[key] for r in rows if r["k"] == k]) for k in K_SCOPE}
    orderings = [p["ordering"] for p in pairs]
    flips = [p["replicate"] for p in pairs if p["ordering"] == "flipped"]
    if orderings.count("unavailable") or not all(p["in_scope"] for p in pairs):
        decision = "STATIONARITY_DIAGNOSTIC_INCONCLUSIVE"
    elif flips or orderings.count("tie"):
        decision = "ONE_STEP_LOCAL_ORDERING_CAN_CHANGE"
    else:
        decision = "ONE_STEP_LOCAL_ORDERING_STABLE"
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "states": len(grads),
        "reconstructed_ok": sum(g["reconstruction_ok"] for g in grads),
        "derivative_evaluable": sum(g["unavailable_components"] == 0
                                    for g in grads),
        "grad_L2_by_k": by_k(grads, "grad_L2"),
        "grad_inf_by_k": by_k(grads, "grad_inf"),
        "gradF_L2_by_k": by_k(grads, "gradF_L2"),
        "grad_logvar_L2_by_k": by_k(grads, "grad_logvar_L2"),
        "abs_grad_w0_by_k": by_k(grads, "abs_grad_w0"),
        "abs_grad_w_by_k": by_k(grads, "abs_grad_w"),
        "h_vs_h2_normalized": _stats([g["h_vs_h2_normalized"] for g in grads]),
        "h_vs_h2_max_abs": _stats([g["h_vs_h2_max_abs"] for g in grads]),
        "vertical_over_total_F": _stats([g["vertical_over_total_F"]
                                         for g in grads]),
        "one_step_status": {s: sum(x["status"] == s for x in steps)
                            for s in ("ACCEPTED", "NO_ACCEPTED_STEP",
                                      "UNAVAILABLE")},
        "delta_ell_one_step_by_k": by_k(steps, "delta_ell_one_step"),
        "ordering_counts": {o: orderings.count(o)
                            for o in ("retained", "flipped", "tie",
                                      "unavailable")},
        "flipped_replicates": flips,
        "DECISION": decision,
        "decision_scope": "one-step local diagnostic only; not a claim of "
                          "convergence to the MLE",
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a recorded run is never "
                         f"overwritten")
    states_json = json.loads((args.source / "fitted_states.json")
                             .read_text("utf-8"))
    lap_rows = {(r["replicate"], int(r["k"])): r
                for r in _read_csv(args.source / "laplace_by_k.csv")}
    paired = json.loads((args.source / "paired_summary.json")
                        .read_text("utf-8"))
    args.out.mkdir(parents=True)
    protocol = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "code_sha": pilot._git_sha(), "git_dirty": pilot._git_dirty(),
        "source_dir": str(args.source),
        "source_sha256": {n: _sha256(args.source / n) for n in
                          ("fitted_states.json", "laplace_by_k.csv",
                           "paired_summary.json")},
        "source_code_sha": states_json["code_sha"],
        "k_scope": list(K_SCOPE), "h_base": H_BASE, "primary_step": "h/2",
        "eps_dir": EPS_DIR, "step_factors": list(STEP_FACTORS),
        "reconstruction_rtol": RECONSTRUCTION_RTOL,
        "joint_mode": "Phase 9H defaults (grad_tol 1e-8, max_iter 200); "
                      "every evaluation starts at the base Z_hat0",
        "x_y": "regenerated from data_seed with run_family_selection_pilot."
               "build_dataset(run_lap_vs_cq_20.PROTOCOL, replicate)",
        "em_executions": 0,
    }
    pilot._write_json(args.out / "protocol.json", protocol)

    datasets = {r.label: pilot.build_dataset(pc.PROTOCOL, r)
                for r in pc.PROTOCOL.replicates}
    grads, steps = [], []
    for entry in states_json["entries"]:
        k = int(entry["k"])
        if k not in K_SCOPE:
            continue
        ds = datasets[entry["replicate"]]
        state = SavedState(entry, ds.X, ds.Y)
        row = lap_rows[(entry["replicate"], k)]
        committed = {"laplace": float(row["laplace_log_observed"]),
                     "C_Lap": float(row["C_Lap"]), "d_K": int(row["d_K"]),
                     "N": float(row["N"])}
        grad, step = diagnose_state(state, committed)
        if not grad["reconstruction_ok"]:
            pilot._write_csv(args.out / "per_state_gradient.csv",
                             grads + [grad])
            raise SystemExit(f"reconstruction mismatch at {entry['replicate']}"
                             f" K={k}: systemic blocker, stopped")
        grads.append(grad)
        steps.append(step)
        print(f"{entry['replicate']} K={k} |g|={grad['grad_L2']:.3e} "
              f"step={step['status']} dell={step['delta_ell_one_step']:.4g}",
              flush=True)
    pairs = pairwise(paired, steps)
    pilot._write_csv(args.out / "per_state_gradient.csv", grads)
    pilot._write_csv(args.out / "per_state_one_step.csv", steps)
    pilot._write_csv(args.out / "per_dataset_pairwise_summary.csv", pairs)
    summary = summarize(grads, steps, pairs)
    pilot._write_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("states", "reconstructed_ok", "one_step_status",
                       "ordering_counts", "DECISION")}))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
