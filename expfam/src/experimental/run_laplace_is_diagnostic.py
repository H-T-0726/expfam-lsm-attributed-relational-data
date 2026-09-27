"""Phase 9J (Issue #92): prospective Laplace approximation-error diagnostic.

Reuses the Phase 9D/9E joint runner, the Phase 9H evaluator and the Phase 9I
C_Lap convention. Only the frozen differences are set here:

- three new datasets rep01..rep03, seeds 991000+r / 992000+r / 993000+r;
  start_B only; K ∈ {2, 3} only; 12 planned EM executions
- after each fresh refit: Candidate B at the refit theta with Z0 = refit
  Z_est (grad_tol 1e-8, max_iter 200, no restart/alternate start/ridge);
  C_Lap = −2·laplace + d_K ln 75 as in Phase 9I
- if and only if the Laplace status is OK: the importance-sampling
  correction c_K = log E_q exp(r) with q = N(Z_hat, H⁻¹), 8 fixed batches of
  512, batch seed 995000 + 100·rep_index + 10·K + batch_index
- the minimal fitted state of every refit (F, Gaussian sigma diagonal, w0,
  w, var_z, Z_est, family assignment, seeds) and the X, Y arrays are saved
  in one NPZ for later post-hoc analysis

delta_IS_23 = delta_Lap_23 − 2(c_3 − c_2) is a diagnostic value, not an
adopted criterion. Lineage E (experimental prototype; not manuscript).

Usage (once, fresh directory)::

    python run_laplace_is_diagnostic.py --out <fresh directory>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import family_selection as fs                                      # noqa: E402
import laplace_importance as li                                    # noqa: E402
import laplace_k_criterion as lk                                   # noqa: E402
import run_family_selection_pilot as pilot                         # noqa: E402
import run_joint_family_k_selection as joint                       # noqa: E402
import run_laplace_pilot as lp                                     # noqa: E402

WRAPPER_VERSION = "laplace-is-diagnostic-v1"

PROTOCOL = dataclasses.replace(
    joint.PROTOCOL,
    stage="laplace_is_diagnostic",
    k_candidates=(2, 3),
    replicates=tuple(joint.Replicate(f"rep{r:02d}", 991000 + r, 992000 + r,
                                     993000 + r) for r in (1, 2, 3)),
    starts=(("start_B", "bernoulli"),),
    failure_scope="dataset",
)

AUTHORIZATION: dict[str, Any] = {
    "gate": "Phase 9J bounded Laplace approximation-error diagnostic",
    "authorized": True,
    "authorized_by": "Human",
    "authorized_in": "Issue #92 body (bounded authorization)",
}

N_BATCHES = 8
SAMPLES_PER_BATCH = 512
N_PENALTY = 75                                   # working convention


def batch_seed(rep_index: int, k: int, batch_index: int) -> int:
    return 995000 + 100 * rep_index + 10 * k + batch_index


def _rep_index(search_seed: int) -> tuple[str, int]:
    for index, rep in enumerate(PROTOCOL.replicates, start=1):
        if rep.search_seed == search_seed:
            return rep.label, index
    raise joint.RunnerStop(f"search seed {search_seed} is not frozen")


def evaluate_fit(result: dict[str, Any], X, Y, k: int, replicate: str,
                 rep_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Laplace + (if OK) importance correction on one refit."""

    refit = result["refit"]
    model = refit["model"]
    selected = result["selected_assignment"]
    n_gauss = sum(f == "gaussian" for f in selected)
    dk = lp.d_K(k, X.shape[1], n_gauss)
    record: dict[str, Any] = {
        "replicate": replicate, "k": k, "d_K": dk, "N": N_PENALTY,
        "selected_assignment": "|".join(selected), "C_Q": refit.get("bic")}
    res = lk.laplace_log_observed(model, X, Y,
                                  np.asarray(refit["Z_est"], float),
                                  grad_tol=1e-8, max_iter=200)
    out = lk.calc_C_Lap(res, d_K=dk, N=N_PENALTY)
    record.update({"laplace_status": res.status,
                   "laplace_log_observed": res.log_observed,
                   "C_Lap": out["C_Lap"], "logdet_H": res.logdet_H,
                   "min_eigenvalue_H": res.min_eigenvalue_H,
                   "grad_inf": res.grad_inf,
                   "joint_mode_iterations": res.iterations})
    if res.ok:
        H = lk.negative_hessian(model, X, Y, res.Z_hat)
        H = 0.5 * (H + H.T)
        seeds = [batch_seed(rep_index, k, b)
                 for b in range(1, N_BATCHES + 1)]
        record["importance"] = li.importance_correction(
            model, X, Y, res.Z_hat, H, seeds, SAMPLES_PER_BATCH)
    else:
        record["importance"] = None
    params = model.params
    state = {"replicate": replicate, "k": k,
             "F": np.asarray(params["F"], float),
             "sigma_diag": np.diag(np.asarray(params["sigma"], float)).copy(),
             "w0": float(params["w0"]), "w": float(params["w"]),
             "var_z": float(params["var_z"]),
             "Z_est": np.asarray(refit["Z_est"], float),
             "selected_assignment": list(selected),
             "X": np.asarray(X, float), "Y": np.asarray(Y, float)}
    return record, state


def make_driver(records: list, states: list,
                inner: Callable[..., dict[str, Any]] | None = None):
    def driver(X, Y, **kwargs):
        run = inner if inner is not None else fs.run_hybrid_family_selection
        result = run(X, Y, **kwargs)
        label, index = _rep_index(kwargs["search_seed"])
        try:
            record, state = evaluate_fit(result, X, Y, kwargs["k"], label,
                                         index)
        except lk.UnsupportedLaplacePath:
            raise                                        # systemic: stop
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            record = {"replicate": label, "k": kwargs["k"],
                      "laplace_status": "EVALUATION_ERROR",
                      "notes": f"{type(exc).__name__}: {exc}",
                      "importance": None}
            state = None
        records.append(record)
        if state is not None:
            states.append(state)
        return result

    return driver


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def replicate_summary(records: Sequence[dict[str, Any]], replicate: str
                      ) -> dict[str, Any]:
    mine = {r["k"]: r for r in records if r["replicate"] == replicate}
    out: dict[str, Any] = {"replicate": replicate}
    both = {2, 3} <= set(mine) and all(
        mine[k].get("laplace_status") == lk.STATUS_OK
        and mine[k].get("importance") for k in (2, 3))
    out["complete"] = bool(both)
    for k in (2, 3):
        r = mine.get(k, {})
        out[f"C_Lap_k{k}"] = r.get("C_Lap")
        imp = r.get("importance")
        out[f"c_pooled_k{k}"] = imp["pooled"]["correction"] if imp else None
    if not both:
        return out
    d_lap = mine[3]["C_Lap"] - mine[2]["C_Lap"]
    c2, c3 = mine[2]["importance"], mine[3]["importance"]
    d_c = c3["pooled"]["correction"] - c2["pooled"]["correction"]
    batch_d = [d_lap - 2.0 * (b3["correction"] - b2["correction"])
               for b2, b3 in zip(c2["batches"], c3["batches"])]
    out.update({
        "delta_Lap_23": d_lap, "delta_c_23": d_c,
        "delta_IS_23": d_lap - 2.0 * d_c,
        "batch_delta_IS_23": batch_d,
        "batch_delta_IS_23_summary": {
            "min": min(batch_d), "median": float(np.median(batch_d)),
            "max": max(batch_d),
            "negative": sum(v < 0 for v in batch_d),
            "positive": sum(v > 0 for v in batch_d),
            "zero": sum(v == 0.0 for v in batch_d)},
    })
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return _jsonable(value.item())
    return value


def save_states(path: Path, states: Sequence[dict[str, Any]],
                code_sha: str) -> None:
    arrays: dict[str, Any] = {}
    index = []
    for s in states:
        key = f"{s['replicate']}_K{s['k']}"
        for name in ("F", "sigma_diag", "Z_est", "X", "Y"):
            arrays[f"{key}__{name}"] = s[name]
        rep = next(r for r in PROTOCOL.replicates if r.label == s["replicate"])
        index.append({"key": key, "replicate": s["replicate"], "k": s["k"],
                      "data_seed": rep.data_seed,
                      "search_seed": rep.search_seed,
                      "refit_seed": rep.refit_seed,
                      "w0": s["w0"], "w": s["w"], "var_z": s["var_z"],
                      "selected_assignment": s["selected_assignment"]})
    meta = {"code_sha": code_sha, "evaluator_version": lk.EVALUATOR_VERSION,
            "is_version": li.IS_VERSION, "wrapper_version": WRAPPER_VERSION,
            "x_y_provenance": "X and Y arrays saved per entry; also "
                              "regenerable from data_seed with "
                              "run_family_selection_pilot.build_dataset",
            "entries": index}
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **arrays)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Issue #92 Phase 9J diagnostic once.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    records: list = []
    states: list = []
    code_sha = pilot._git_sha()
    try:
        joint.execute(args.out, protocol=PROTOCOL,
                      driver=make_driver(records, states),
                      authorization=AUTHORIZATION)
    finally:
        if args.out.is_dir():
            if states:
                save_states(args.out / "fitted_states.npz", states, code_sha)
            pilot._write_json(args.out / "is_records.json",
                              _jsonable(records))
    summary = {
        "wrapper_version": WRAPPER_VERSION,
        "protocol": {"k": [2, 3], "batches": N_BATCHES,
                     "samples_per_batch": SAMPLES_PER_BATCH,
                     "batch_seed": "995000 + 100*rep_index + 10*K + b",
                     "N": N_PENALTY},
        "replicates": [replicate_summary(records, r.label)
                       for r in PROTOCOL.replicates],
        "note": "delta_IS_23 is a diagnostic value, not an adopted "
                "criterion",
    }
    pilot._write_json(args.out / "is_summary.json", _jsonable(summary))
    for rep in summary["replicates"]:
        print(json.dumps(_jsonable({k: v for k, v in rep.items()
                                    if k != "batch_delta_IS_23"})))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
