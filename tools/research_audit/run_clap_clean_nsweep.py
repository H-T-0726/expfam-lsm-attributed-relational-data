"""Issue #128 experiment A1: add C_Lap to the canonical clean true-K n-sweep.

Same fits, more criteria.  The historical sweep
``expfam/results/k_selection/clean_true_k_asymptotics_20260904`` kept criterion
VALUES only, not fitted states, so C_Lap needs the states back:

- K_TRUE = 5: READ-ONLY from the audited external archive of the EM
  convergence experiment (checkpoint 8 == the clean-sweep fit, 448/448 exact,
  ``replication_vs_baseline.json``).  The state is used only after it
  reproduces the historical heldout / q_strict / S2 / S3 within the tolerance
  inherited from the audited ``audit_em_convergence_sensitivity.py``.
- K_TRUE = 1, 3: the fit is RE-RUN with the historical seed, arguments and
  code path and must reproduce heldout / q_strict / S2 / S3 / num_params
  EXACTLY (bit-for-bit).  ``failure_policy='fail_fast'`` only removes the
  legacy retry branch; with no non-finite E-step it is the same computation.

Any reproduction mismatch, exception or non-finite value STOPS the run
(failure.json).  There is no retry, replacement fit, seed change, tolerance
change or resume.  A C_Lap evaluation that is NOT_STATIONARY / HESSIAN_NOT_PD
is a recorded result (the dataset's C_Lap selection becomes INCOMPLETE); in
the pilot it fails the gate.

Criteria: C_Q (legacy Q-based complete-data / ICL-type criterion, NOT Schwarz
BIC) and C_Lap (Laplace-evidence candidate, NOT exact evidence) are primary;
S1 held-out is a prediction-oriented reference; S3 is a diagnostic only.

Lineage E (experimental prototype).  MANUSCRIPT_ADOPTION_STATUS:
NOT_DECIDED_BY_HUMAN.

Usage:
    python run_clap_clean_nsweep.py --preflight
    python run_clap_clean_nsweep.py --pilot --allow-em
    python run_clap_clean_nsweep.py --full --allow-em --confirm-a1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
for _path in (str(_EXPERIMENTAL), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_clean_true_k_sweep as hist                       # noqa: E402
import k_criterion_panel as panel                           # noqa: E402

# ===========================================================================
# FROZEN PROTOCOL (Issue #128, Human Gate 2026-10-06)
# ===========================================================================

EXPERIMENT_ID = "clap_clean_nsweep_20261006"
PROTOCOL_VERSION = "clap-clean-nsweep-protocol-v1"
OUT_ROOT = ROOT / "expfam" / "results" / "k_selection" / EXPERIMENT_ID
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / EXPERIMENT_ID

HIST_DIR = ROOT / "expfam" / "results" / "k_selection" / hist.ARTIFACT_DIRNAME
HIST_PROTOCOL_HASH = ("547880a16aef6530cfdf7903c4e32f16062397e0bacc0c109d5c77fb"
                      "9892ccc0")
HIST_FILES = ("fit_results.csv", "selection_matrix.csv", "manifest.csv",
              "generator_provenance.csv", "protocol.json", "runinfo.json",
              "summary.json", "audit_report.json", "gram_spectrum.csv")

EMC_NPZ = (Path(r"D:\tento\kennkyu_artifacts")
           / "em_convergence_sensitivity_20260905" / "checkpoint_states.npz")
EMC_NPZ_SHA256 = ("5e4390f1c61edba63acd3b5e055816331fd63758a46642b200b923e93d"
                  "40af84")
# The convergence experiment's identity table lives on an unmerged branch;
# it is read from the git object store at this pinned commit.
EMC_COMMIT = "dad67b1c7b90f1e0297e8b1e95afd394cf9b51db"
EMC_FIT_RESULTS = ("expfam/results/k_selection/em_convergence_sensitivity_"
                   "20260905/fit_results.csv")
EMC_CHECKPOINT = 8

REPLAY_K_TRUE = (1, 3)
NPZ_K_TRUE = (5,)
# Inherited from the audited audit_em_convergence_sensitivity.py.
NPZ_RTOL = 1e-9
NPZ_ATOL = 1e-6
RECOMPOSE_TOL = 1e-6          # C_Q / C_Lap decomposition closure (absolute)

PILOT_CELLS = ((3, 50, 1), (5, 50, 1))
CRITERIA = ("C_Q", "C_Lap", "S1", "S3")
LOWER_IS_BETTER = {"C_Q": True, "C_Lap": True, "S1": False, "S3": True}
CRITERION_ROLE = {"C_Q": "primary", "C_Lap": "primary",
                  "S1": "prediction_reference", "S3": "diagnostic"}


class A1Stop(RuntimeError):
    """Fail-closed stop.  Never caught to retry."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise A1Stop(message)


def frozen_protocol() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION,
        "issue": 128,
        "historical_experiment": hist.ARTIFACT_DIRNAME,
        "historical_protocol_hash": HIST_PROTOCOL_HASH,
        "family_x": hist.FAMILY_X, "family_y": hist.FAMILY_Y,
        "d": hist.D_FEATURES, "L": hist.L_SAMPLES, "num_iter": hist.NUM_ITER,
        "numerics_mode": hist.NUMERICS_MODE, "test_ratio": hist.TEST_RATIO,
        "k_true_grid": list(hist.K_TRUE_GRID), "n_grid": list(hist.N_GRID),
        "candidate_k": list(hist.CANDIDATE_K), "starts": list(hist.STARTS),
        "replicates_by_k_true": {str(k): v for k, v in
                                 hist.REPLICATES_BY_K_TRUE.items()},
        "seed_rule": "historical data/split/model seed functions, unchanged",
        "state_source": {"replay_exact": list(REPLAY_K_TRUE),
                         "npz_readonly": list(NPZ_K_TRUE)},
        "replay_failure_policy": "fail_fast (no retry, no Z_prev substitution)",
        "replay_match_rule": "exact equality of heldout, q_strict, S2, S3, "
                             "num_params; nan_occurred False",
        "npz_checkpoint": EMC_CHECKPOINT, "npz_sha256": EMC_NPZ_SHA256,
        "npz_identity_commit": EMC_COMMIT,
        "npz_match_rule": {"rtol": NPZ_RTOL, "atol": NPZ_ATOL},
        "laplace": {"grad_tol": panel.LAPLACE_GRAD_TOL,
                    "max_iter": panel.LAPLACE_MAX_ITER,
                    "Z0": "fit Z_est (last E-step sample)",
                    "d_K": "num_params of the fit (= K d - K(K-1)/2)",
                    "N": "n (working convention)",
                    "Y": "training pairs only (same mask as the fit)",
                    "ridge_jitter_restart": False},
        "criteria": {c: CRITERION_ROLE[c] for c in CRITERIA},
        "selector": "mean over the two starts per candidate K; best value; "
                    "ties within 1e-12 -> smallest K (historical rule)",
        "incomplete_rule": "a dataset whose C_Lap is non-OK at any (K, start) "
                           "has C_Lap selection INCOMPLETE; no imputation",
        "stop_rules": ["replay mismatch", "npz mismatch beyond tolerance",
                       "any exception", "non-finite C_Q/S1/S3",
                       "decomposition closure beyond tolerance"],
        "failure_policy": ["no_retry", "no_replacement_fit", "no_seed_rescue",
                           "no_tolerance_change", "no_resume"],
        "manuscript_adoption_status": "NOT_DECIDED_BY_HUMAN",
    }


def protocol_hash() -> str:
    body = json.dumps(frozen_protocol(), sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def git_head() -> str:
    return _git("rev-parse", "HEAD")


def working_tree_clean() -> bool:
    return _git("status", "--porcelain") == ""


def environment() -> dict[str, Any]:
    import scipy
    return {"python_version": platform.python_version(),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "platform": platform.platform()}


def historical_hashes() -> dict[str, str]:
    return {name: sha256_file(HIST_DIR / name) for name in HIST_FILES}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               default=float), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (repr(v) if isinstance(v, float) else v)
                        for c, v in r.items()})


# ---------------------------------------------------------------------------
# historical tables
# ---------------------------------------------------------------------------

def load_hist_fits() -> dict[tuple[int, int, int, int, int], dict[str, str]]:
    with open(HIST_DIR / "fit_results.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        key = (int(r["k_true"]), int(r["n"]), int(r["replicate"]),
               int(r["k_est"]), int(r["start"]))
        _require(key not in out, f"duplicate historical row {key}")
        out[key] = r
    _require(len(out) == 896, f"historical fit rows {len(out)} != 896")
    return out


def load_hist_selection() -> dict[tuple[str, int, int, int], int]:
    with open(HIST_DIR / "selection_matrix.csv", newline="",
              encoding="utf-8") as fh:
        return {(r["criterion"], int(r["K_TRUE"]), int(r["n"]),
                 int(r["replicate"])): int(r["selected_k"])
                for r in csv.DictReader(fh)}


def load_hist_provenance() -> dict[tuple[int, int, int], dict[str, str]]:
    with open(HIST_DIR / "generator_provenance.csv", newline="",
              encoding="utf-8") as fh:
        return {(int(r["K_TRUE"]), int(r["n"]), int(r["replicate"])): r
                for r in csv.DictReader(fh)}


def load_emc_identity() -> tuple[dict[tuple, dict[str, str]], str]:
    raw = subprocess.check_output(
        ["git", "show", f"{EMC_COMMIT}:{EMC_FIT_RESULTS}"], cwd=ROOT)
    digest = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    out = {}
    for r in rows:
        key = (int(r["k_true"]), int(r["n"]), int(r["replicate"]),
               int(r["k_est"]), int(r["start"]))
        _require(key not in out, f"duplicate EMC row {key}")
        _require(int(r["retry_count"]) == 0 and r["nan_detected"] == "False",
                 f"EMC fit {key} was not clean")
        out[key] = r
    _require(len(out) == 448, f"EMC identity rows {len(out)} != 448")
    return out, digest


# ---------------------------------------------------------------------------
# datasets and states
# ---------------------------------------------------------------------------

def make_cell(k_true: int, n: int, rep: int, prov_row: dict[str, str]):
    scale = hist.f_scale_for_row_norm(hist.ROW_NORM_SQ_TARGET,
                                      d=hist.D_FEATURES, k=k_true)
    w_true = hist.w_for_matched_y_signal(hist.W_REF, k=k_true, k_ref=hist.K_REF)
    ds = hist.generate_canonical_data(
        n=n, d=hist.D_FEATURES, k=k_true,
        seed=hist.data_seed(k_true, n, rep),
        family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y, f_scale=scale,
        w0=hist.W0_TRUE, w=w_true, poisson_lambda_max=hist.POISSON_LAMBDA_MAX)
    train, test = hist.build_masks(n, hist.split_seed(k_true, n, rep))
    _require(hist.stable_hash(train) == prov_row["train_mask_hash"],
             f"train mask hash differs at {(k_true, n, rep)}")
    _require(hist.stable_hash(test) == prov_row["test_mask_hash"],
             f"test mask hash differs at {(k_true, n, rep)}")
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    _require(float(ds.Y[upper].mean()) == float(prov_row["y_density"]),
             f"Y density differs at {(k_true, n, rep)}")
    _require(float(ds.X.mean()) == float(prov_row["x_mean"]),
             f"X mean differs at {(k_true, n, rep)}")
    return ds, train, test


def heldout_score(Z_est, w0, w, Y, test) -> float:
    rows, cols = np.where(np.triu(test, 1))
    eta = (w0 + w * (Z_est @ Z_est.T))[rows, cols]
    return hist.heldout_bernoulli_mean_log_score(Y[rows, cols], eta)


def state_from_replay(ds, train, k_est, seed) -> dict[str, Any]:
    import em_runner                                        # noqa: PLC0415
    t0 = time.perf_counter()
    res = em_runner.run_em_experimental(
        X=ds.X, Y=ds.Y, family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y,
        k=k_est, L=hist.L_SAMPLES, num_iter=hist.NUM_ITER, seed=seed,
        train_mask=train, numerics_mode=hist.NUMERICS_MODE,
        compute_strict_Q=True, failure_policy="fail_fast")
    return {"model": res["model"], "Z_samples": np.asarray(res["Z_samples"]),
            "Z_est": np.asarray(res["Z_est"], dtype=np.float64),
            "F": np.asarray(res["F"]), "sigma": np.asarray(res["sigma"]),
            "w0": float(res["w0"]), "w": float(res["w"]),
            "var_z": float(res["var_z"]),
            "runner_q_strict": float(res["Q_strict"]),
            "runner_s2": float(res["bic"]),
            "num_params": int(res["num_params"]),
            "nan_occurred": bool(res["nan_occurred"]),
            "retry_count": int(res.get("retry_count", 0)),
            "replacement_count": int(res.get("replacement_count", 0)),
            "seed_rescue_count": int(res.get("seed_rescue_count", 0)),
            "em_seconds": round(time.perf_counter() - t0, 3)}


def state_from_npz(npz, emc_row, n, k_est, train) -> dict[str, Any]:
    import em_runner                                        # noqa: PLC0415
    key = f"{int(emc_row['fit_index']):04d}_it{EMC_CHECKPOINT:03d}"
    Zs = np.asarray(npz[f"{key}_Zsamples"], dtype=np.float64)
    Z = np.asarray(npz[f"{key}_Z"], dtype=np.float64)
    F = np.asarray(npz[f"{key}_F"], dtype=np.float64)
    sigma = np.asarray(npz[f"{key}_sigma"], dtype=np.float64)
    w0, w, var_z = (float(v) for v in npz[f"{key}_scalars"])
    _require(Zs.shape == (n, k_est, hist.L_SAMPLES), f"Zsamples shape {key}")
    _require(np.array_equal(Z, Zs[:, :, -1]), f"Z != last sample at {key}")
    model = em_runner.build_model(n, hist.D_FEATURES, k_est, hist.L_SAMPLES,
                                  hist.FAMILY_X, hist.FAMILY_Y,
                                  train_mask=train,
                                  numerics_mode=hist.NUMERICS_MODE)
    model.initialize_params(true_params=None, seed=0)
    model.params.update({"Z": Z.copy(), "F": F, "sigma": sigma, "w0": w0,
                         "w": w, "var_z": var_z})
    num_params = k_est * hist.D_FEATURES - k_est * (k_est - 1) // 2
    return {"model": model, "Z_samples": Zs, "Z_est": Z, "F": F,
            "sigma": sigma, "w0": w0, "w": w, "var_z": var_z,
            "runner_q_strict": float("nan"), "runner_s2": float("nan"),
            "num_params": num_params, "nan_occurred": False,
            "retry_count": 0, "replacement_count": 0, "seed_rescue_count": 0,
            "em_seconds": 0.0, "npz_key": key}


# ---------------------------------------------------------------------------
# one fit -> one row
# ---------------------------------------------------------------------------

def evaluate_fit(cell, k_est, start, source, st, ds, train, test,
                 hist_row) -> tuple[dict[str, Any], np.ndarray]:
    k_true, n, rep = cell
    model = st["model"]
    s1 = heldout_score(st["Z_est"], st["w0"], st["w"], ds.Y, test)
    cq = panel.cq_terms(model, ds.X, ds.Y, st["Z_samples"],
                        num_params=st["num_params"], n=n)
    s3 = hist.s3_plugin_conditional(model, ds.X, ds.Y, st["Z_est"], st["F"],
                                    st["w0"], st["w"], st["num_params"], n)
    h = {k: float(hist_row[k]) for k in
         ("heldout_mean_log_score", "q_strict", "s2_q_based",
          "s3_plugin_conditional")}
    _require(int(hist_row["num_params"]) == st["num_params"],
             f"num_params differs at {cell, k_est, start}")
    got = {"heldout_mean_log_score": s1, "q_strict": cq["q_strict"],
           "s2_q_based": cq["C_Q"], "s3_plugin_conditional": s3}
    if source == "replay":
        got["q_strict"] = st["runner_q_strict"]
        got["s2_q_based"] = st["runner_s2"]
        _require(not st["nan_occurred"] and st["retry_count"] == 0
                 and st["replacement_count"] == 0
                 and st["seed_rescue_count"] == 0,
                 f"replay fit not clean at {cell, k_est, start}")
        # the panel's own Q must equal the runner's (same function, same args)
        _require(cq["q_strict"] == st["runner_q_strict"],
                 f"panel q_strict != runner q_strict at {cell, k_est, start}")
        for name, value in got.items():
            _require(value == h[name],
                     f"REPLAY MISMATCH {name} at {cell, k_est, start}: "
                     f"{value!r} != historical {h[name]!r}")
    else:
        for name, value in got.items():
            _require(math.isclose(value, h[name], rel_tol=NPZ_RTOL,
                                  abs_tol=NPZ_ATOL),
                     f"NPZ MISMATCH {name} at {cell, k_est, start}: "
                     f"{value!r} vs historical {h[name]!r}")
    for name, value in got.items():
        _require(math.isfinite(value), f"non-finite {name}")
    _require(abs(cq["C_Q_recomposed"] - cq["C_Q"]) <= RECOMPOSE_TOL,
             f"C_Q closure at {cell, k_est, start}")

    lap = panel.laplace_terms(model, ds.X, ds.Y, st["Z_est"],
                              num_params=st["num_params"], N=n)
    z_hat = lap.pop("Z_hat")
    if lap["lap_status"] == "OK":
        _require(math.isfinite(lap["C_Lap"]), "OK C_Lap is non-finite")
        _require(abs(lap["C_Lap_recomposed"] - lap["C_Lap"]) <= RECOMPOSE_TOL,
                 f"C_Lap closure at {cell, k_est, start}")

    row = {
        "k_true": k_true, "n": n, "replicate": rep, "k_est": k_est,
        "start": start, "state_source": source,
        "data_seed": hist.data_seed(k_true, n, rep),
        "split_seed": hist.split_seed(k_true, n, rep),
        "model_seed": hist.model_seed(k_true, n, rep, k_est, start),
        "npz_key": st.get("npz_key", ""),
        "num_params": st["num_params"],
        "S1": s1, "C_Q": got["s2_q_based"], "S3": s3,
        "q_strict": got["q_strict"],
        "hist_S1": h["heldout_mean_log_score"], "hist_C_Q": h["s2_q_based"],
        "hist_S3": h["s3_plugin_conditional"], "hist_q_strict": h["q_strict"],
        "absdiff_S1": abs(s1 - h["heldout_mean_log_score"]),
        "absdiff_C_Q": abs(got["s2_q_based"] - h["s2_q_based"]),
        "absdiff_S3": abs(s3 - h["s3_plugin_conditional"]),
        "absdiff_q_strict": abs(got["q_strict"] - h["q_strict"]),
        "Q_X": cq["Q_X"], "Q_Y": cq["Q_Y"], "Q_Z": cq["Q_Z"],
        "D_K": cq["D_K"], "P_Z": cq["P_Z"], "P_theta": cq["P_theta"],
        "C_Q_recomposed": cq["C_Q_recomposed"],
        "w0_hat": st["w0"], "w_hat": st["w"], "var_z": st["var_z"],
        **lap,
        "nan_occurred": st["nan_occurred"], "retry_count": st["retry_count"],
        "replacement_count": st["replacement_count"],
        "seed_rescue_count": st["seed_rescue_count"],
        "em_seconds": st["em_seconds"],
    }
    return row, z_hat


# ---------------------------------------------------------------------------
# selection (the independent auditor re-derives all of this from the CSV)
# ---------------------------------------------------------------------------

def _select(values: dict[int, float], lower: bool) -> tuple[int, list[int],
                                                            float, float]:
    sign = -1.0 if lower else 1.0
    score = {k: np.float64(sign * v) for k, v in values.items()}
    best = max(score.values())
    ties = sorted(k for k, v in score.items() if best - v <= hist.TIE_TOLERANCE)
    chosen = min(ties)
    others = sorted((v for k, v in score.items() if k != chosen), reverse=True)
    margin = float(best - others[0]) if others else float("nan")
    return chosen, ties, margin, float(sign * best)


def label(selected: int, k_true: int) -> str:
    return "exact" if selected == k_true else ("under" if selected < k_true
                                              else "over")


def analyse(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cells = sorted({(r["k_true"], r["n"], r["replicate"]) for r in rows})
    sel_rows, curve_rows, dec_rows = [], [], []
    for cell in cells:
        k_true, n, rep = cell
        sub = [r for r in rows if (r["k_true"], r["n"], r["replicate"]) == cell]
        ks = sorted({r["k_est"] for r in sub})
        _require(ks == list(hist.CANDIDATE_K), f"candidate K incomplete {cell}")
        by = {(r["k_est"], r["start"]): r for r in sub}
        _require(len(by) == len(ks) * len(hist.STARTS), f"fit count {cell}")
        for crit in CRITERIA:
            complete = all(by[(k, s)]["lap_status"] == "OK"
                           for k in ks for s in hist.STARTS) \
                if crit == "C_Lap" else True
            means = {k: float(np.mean([by[(k, s)][crit]
                                       for s in hist.STARTS])) for k in ks}
            for k in ks:
                curve_rows.append({
                    "criterion": crit, "role": CRITERION_ROLE[crit],
                    "k_true": k_true, "n": n, "replicate": rep, "k_est": k,
                    "mean_value": means[k],
                    **{f"start{s}_value": by[(k, s)][crit]
                       for s in hist.STARTS}})
            if not complete:
                sel_rows.append({"criterion": crit, "k_true": k_true, "n": n,
                                 "replicate": rep, "status": "INCOMPLETE",
                                 "selected_k": "", "label": "",
                                 "tie_candidates": "", "margin": "",
                                 "best_value": "", "selected_start1": "",
                                 "selected_start2": ""})
                continue
            chosen, ties, margin, best = _select(means, LOWER_IS_BETTER[crit])
            per_start = {s: _select({k: by[(k, s)][crit] for k in ks},
                                    LOWER_IS_BETTER[crit])[0]
                         for s in hist.STARTS}
            sel_rows.append({"criterion": crit, "k_true": k_true, "n": n,
                             "replicate": rep, "status": "COMPLETE",
                             "selected_k": chosen,
                             "label": label(chosen, k_true),
                             "tie_candidates": json.dumps(ties),
                             "margin": margin, "best_value": best,
                             "selected_start1": per_start[1],
                             "selected_start2": per_start[2]})
        # K -> K+1 increments, mean over starts (diagnostic only)
        def m(k, col):
            return float(np.mean([by[(k, s)][col] for s in hist.STARTS]))
        for k in ks[:-1]:
            d_fit_q = m(k + 1, "D_K") - m(k, "D_K")
            d_pz = m(k + 1, "P_Z") - m(k, "P_Z")
            d_pt = m(k + 1, "P_theta") - m(k, "P_theta")
            lap_ok = all(by[(kk, s)]["lap_status"] == "OK"
                         for kk in (k, k + 1) for s in hist.STARTS)
            if lap_ok:
                d_fit_l = m(k + 1, "lap_fit_mode") - m(k, "lap_fit_mode")
                d_vol = m(k + 1, "lap_volume") - m(k, "lap_volume")
                d_pen = m(k + 1, "lap_penalty") - m(k, "lap_penalty")
            else:
                d_fit_l = d_vol = d_pen = float("nan")
            dec_rows.append({
                "k_true": k_true, "n": n, "replicate": rep,
                "k_from": k, "k_to": k + 1,
                "CQ_fit_gain": -d_fit_q, "CQ_dPZ": d_pz, "CQ_dPtheta": d_pt,
                "CQ_delta": m(k + 1, "C_Q") - m(k, "C_Q"),
                "CQ_slack": -d_fit_q - (d_pz + d_pt),
                "Lap_complete": lap_ok,
                "Lap_fit_gain": -d_fit_l, "Lap_dvolume": d_vol,
                "Lap_dpenalty": d_pen,
                "Lap_delta": (m(k + 1, "C_Lap") - m(k, "C_Lap")
                              if lap_ok else float("nan")),
                "Lap_slack": (-d_fit_l - (d_vol + d_pen)
                              if lap_ok else float("nan"))})
    return {"selection": sel_rows, "curves": curve_rows,
            "decomposition": dec_rows}


def summarise(sel_rows) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for crit in CRITERIA:
        by_cell: dict[str, Any] = {}
        for k_true in hist.K_TRUE_GRID:
            for n in hist.N_GRID:
                sub = [r for r in sel_rows if r["criterion"] == crit
                       and r["k_true"] == k_true and r["n"] == n]
                if not sub:
                    continue
                comp = [r for r in sub if r["status"] == "COMPLETE"]
                by_cell[f"K{k_true}/n{n}"] = {
                    "datasets": len(sub), "complete": len(comp),
                    "incomplete": len(sub) - len(comp),
                    "selected_k": [r["selected_k"] if r["status"] == "COMPLETE"
                                   else None for r in
                                   sorted(sub, key=lambda r: r["replicate"])],
                    "exact": sum(r["label"] == "exact" for r in comp),
                    "under": sum(r["label"] == "under" for r in comp),
                    "over": sum(r["label"] == "over" for r in comp)}
        out[crit] = {"role": CRITERION_ROLE[crit], "by_cell": by_cell}
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(cells: list[tuple[int, int, int]], out_dir: Path, state_dir: Path,
        stage: str) -> int:
    _require(not out_dir.exists(), f"output directory exists: {out_dir}")
    _require(not state_dir.exists(), f"state directory exists: {state_dir}")
    _require(working_tree_clean(), "working tree must be clean")
    _require(sha256_file(EMC_NPZ) == EMC_NPZ_SHA256, "npz sha256 differs")
    before = historical_hashes()
    proto = json.loads((HIST_DIR / "protocol.json").read_text("utf-8"))
    _require(proto["protocol_hash"] == HIST_PROTOCOL_HASH,
             "historical protocol hash differs")
    hist_fits = load_hist_fits()
    hist_sel = load_hist_selection()
    hist_prov = load_hist_provenance()
    emc, emc_digest = load_emc_identity()
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    started = time.time()
    meta = {"experiment_id": EXPERIMENT_ID, "stage": stage,
            "protocol_hash": protocol_hash(), "git_head": git_head(),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_clean_before_execution": True,
            "environment": environment(), "started_at": started,
            "historical_hashes_before": before,
            "inputs": {"historical_dir": str(HIST_DIR.relative_to(ROOT)),
                       "npz": str(EMC_NPZ), "npz_sha256": EMC_NPZ_SHA256,
                       "emc_identity": f"{EMC_COMMIT}:{EMC_FIT_RESULTS}",
                       "emc_identity_sha256": emc_digest},
            "cells": [list(c) for c in cells]}
    write_json(out_dir / "protocol.json", {"protocol": frozen_protocol(),
                                           "protocol_hash": protocol_hash()})

    rows: list[dict[str, Any]] = []
    states: dict[str, np.ndarray] = {}
    npz = np.load(EMC_NPZ) if any(c[0] in NPZ_K_TRUE for c in cells) else None
    attempted = 0
    try:
        for cell in cells:
            k_true, n, rep = cell
            ds, train, test = make_cell(k_true, n, rep, hist_prov[cell])
            for k_est in hist.CANDIDATE_K:
                for start in hist.STARTS:
                    attempted += 1
                    key = (k_true, n, rep, k_est, start)
                    seed = hist.model_seed(k_true, n, rep, k_est, start)
                    _require(int(hist_fits[key]["model_seed"]) == seed,
                             f"historical model seed differs {key}")
                    if k_true in REPLAY_K_TRUE:
                        source = "replay"
                        st = state_from_replay(ds, train, k_est, seed)
                    else:
                        source = "npz_readonly"
                        er = emc[key]
                        _require(int(er["model_seed"]) == seed
                                 and int(er["data_seed"]) ==
                                 hist.data_seed(k_true, n, rep)
                                 and er["train_mask_hash"] ==
                                 hist.stable_hash(train),
                                 f"EMC identity differs {key}")
                        st = state_from_npz(npz, er, n, k_est, train)
                    row, z_hat = evaluate_fit(cell, k_est, start, source, st,
                                              ds, train, test, hist_fits[key])
                    rows.append(row)
                    tag = f"K{k_true}_n{n}_r{rep}_k{k_est}_s{start}"
                    states[f"{tag}_Zhat"] = z_hat
                    if source == "replay":
                        states[f"{tag}_Zsamples"] = st["Z_samples"]
                        states[f"{tag}_F"] = st["F"]
                        states[f"{tag}_scalars"] = np.array(
                            [st["w0"], st["w"], st["var_z"]])
                    print(f"[{stage}] {tag} src={source} "
                          f"lap={row['lap_status']} "
                          f"em={row['em_seconds']}s lap={row['lap_seconds']}s",
                          flush=True)
            # selection reproduction gate for the historical criteria
            for crit, hname in (("C_Q", "S2"), ("S1", "S1"), ("S3", "S3")):
                sub = [r for r in rows
                       if (r["k_true"], r["n"], r["replicate"]) == cell]
                res = analyse(sub)["selection"]
                got = [r for r in res if r["criterion"] == crit][0]
                _require(got["selected_k"] == hist_sel[(hname, *cell)],
                         f"selection of {crit} differs from historical at "
                         f"{cell}")
    except Exception as exc:                                # fail closed
        write_csv(out_dir / "fit_criteria_partial.csv", rows)
        write_json(out_dir / "failure.json", {
            **meta, "status": "STOPPED", "error_type": type(exc).__name__,
            "error": str(exc), "attempted_fits": attempted,
            "completed_rows": len(rows), "retry_count": 0,
            "replacement_fits_executed": 0, "seed_rescue_count": 0,
            "historical_hashes_after": historical_hashes()})
        print(f"STOPPED: {type(exc).__name__}: {exc}", flush=True)
        return 2

    analysis = analyse(rows)
    state_path = state_dir / "states.npz"
    np.savez_compressed(state_path, **states)
    write_csv(out_dir / "fit_criteria.csv", rows)
    write_csv(out_dir / "selection.csv", analysis["selection"])
    write_csv(out_dir / "criterion_curves.csv", analysis["curves"])
    write_csv(out_dir / "increment_decomposition.csv",
              analysis["decomposition"])
    write_json(out_dir / "summary.json", summarise(analysis["selection"]))
    after = historical_hashes()
    _require(after == before, "historical artifacts changed during the run")
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["lap_status"]] = statuses.get(r["lap_status"], 0) + 1
    completed = time.time()
    write_json(out_dir / "runinfo.json", {
        **meta, "status": "COMPLETED", "completed_at": completed,
        "wall_clock_seconds": round(completed - started, 1),
        "attempted_fits": attempted, "completed_rows": len(rows),
        "replayed_fits": sum(r["state_source"] == "replay" for r in rows),
        "npz_readonly_fits": sum(r["state_source"] == "npz_readonly"
                                 for r in rows),
        "replay_exact_matches": sum(r["state_source"] == "replay" and
                                    r["absdiff_q_strict"] == 0.0 and
                                    r["absdiff_C_Q"] == 0.0 and
                                    r["absdiff_S1"] == 0.0 and
                                    r["absdiff_S3"] == 0.0 for r in rows),
        "npz_max_absdiff": {c: max([r[f"absdiff_{c}"] for r in rows
                                    if r["state_source"] == "npz_readonly"],
                                   default=0.0)
                            for c in ("S1", "C_Q", "S3", "q_strict")},
        "laplace_status_counts": statuses,
        "retry_count": 0, "replacement_fits_executed": 0,
        "seed_rescue_count": 0, "tolerance_relaxations": 0, "resumed": False,
        "states_file": str(state_path),
        "states_sha256": sha256_file(state_path),
        "historical_hashes_after": after,
        "historical_unchanged": True})
    print(f"[{stage}] completed {len(rows)} fits; laplace {statuses}",
          flush=True)
    return 0


def all_cells() -> list[tuple[int, int, int]]:
    return [(c.k_true, c.n, c.replicate) for c in hist.build_cells()]


def preflight() -> dict[str, Any]:
    hist_fits = load_hist_fits()
    emc, digest = load_emc_identity()
    cells = all_cells()
    _require(len(cells) == 64, "cell count")
    for (k_true, n, rep, k, s), r in emc.items():
        h = hist_fits[(k_true, n, rep, k, s)]
        _require(r["model_seed"] == h["model_seed"], "EMC seed mismatch")
    return {"protocol_hash": protocol_hash(), "cells": len(cells),
            "fits": 896, "replay_fits": 448, "npz_fits": 448,
            "npz_sha256_ok": sha256_file(EMC_NPZ) == EMC_NPZ_SHA256,
            "emc_identity_sha256": digest,
            "historical_hashes": historical_hashes(),
            "working_tree_clean": working_tree_clean(),
            "git_head": git_head(), "environment": environment()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--allow-em", action="store_true")
    ap.add_argument("--confirm-a1", action="store_true")
    a = ap.parse_args(argv)
    if a.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    _require(a.allow_em, "--allow-em is required")
    if a.pilot:
        return run(list(PILOT_CELLS), OUT_ROOT / "pilot", STATE_ROOT / "pilot",
                   "pilot")
    _require(a.confirm_a1, "--confirm-a1 is required for the full run")
    return run(all_cells(), OUT_ROOT / "full", STATE_ROOT / "full", "full")


if __name__ == "__main__":
    sys.exit(main())
