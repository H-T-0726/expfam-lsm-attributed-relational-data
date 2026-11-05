"""Zero-fit re-analysis of EXISTING K-selection states (lineage E) and historical CSVs.

Part of research/k-selection-causal-analysis-20261106.  No MCEM fit is run here.

What it does
------------
1. Historical summary (0.5 lineage Scenario A/B/C Exp1, fixed_official Exp1 K1..9,
   paper-type #139 B) recomputed from the primary per-trial CSVs.
2. Canonical summary (A1 n-sweep, d-sweep, R2) from the primary selection.csv files.
3. X / Y / Z split of Q_MC for every saved canonical K_GEN=5 state:
     * d=15, n in {50,75,100,150}: EMC checkpoint-8 states (external npz)
     * d=30/60, n=75: d-sweep fit_states.npz
     * d=30, n in {50,100,150}: R2 fit_states npz
   Q_X, Q_Y, Q_Z come from k_criterion_panel.cq_terms on the stored Z samples and
   theta; C_Q is checked against the stored CSV value.  The per-sample spread of
   the L samples gives a Monte Carlo standard error of each Q component.
4. K -> K+1 increment decomposition (mean over the two starts, the selector's own
   averaging) with the stored Laplace columns (volume, ln|H| split).

Outputs (new directory, never overwrites):
  expfam/results/k_selection/historical_canonical_causal_20261106/existing_reanalysis/
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
_EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
for _p in (str(_EXPERIMENTAL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_clap_clean_nsweep as a1                    # noqa: E402
import run_k_selection_robustness as rob             # noqa: E402
import k_criterion_panel as panel                     # noqa: E402
import em_runner                                      # noqa: E402

hist = a1.hist
OUT = ROOT / "expfam" / "results" / "k_selection" / "historical_canonical_causal_20261106" / "existing_reanalysis"
RES = ROOT / "expfam" / "results"
KSEL = RES / "k_selection"
ART = Path(r"D:\tento\kennkyu_artifacts")
C2 = 1.0 + math.log(2.0 * math.pi)
K_GEN = 5


def need(c: bool, m: str) -> None:
    if not c:
        raise RuntimeError(m)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p: Path) -> list[dict[str, str]]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    need(not p.exists(), f"refuse to overwrite {p}")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({c: (repr(float(v)) if isinstance(v, (float, np.floating)) else v)
                        for c, v in r.items()})


def label(sel: int, k: int) -> str:
    return "exact" if sel == k else ("under" if sel < k else "over")


# ------------------------------------------------------------------ 1. historical
def historical_rows() -> list[dict[str, Any]]:
    rows = []
    fam = {"A": ("poisson", "bernoulli"), "B": ("gaussian", "poisson"), "C": ("bernoulli", "gaussian")}
    for sc in "ABC":
        p = RES / f"exp_scenario_{sc}_exp1_k.csv"
        data = read_csv(p)
        trials = sorted({int(r["trial"]) for r in data})
        for t in trials:
            bic = {int(r["k_est"]): float(r["BIC"]) for r in data if int(r["trial"]) == t}
            ks = sorted(bic)
            sel = min(ks, key=lambda k: bic[k])
            rows.append({"experiment_id": f"initial_dual_expfam_scenario_{sc}_exp1",
                         "generator_version": "data_generator_expfam.generate_dual_data (Z z-scored, ||f_l||^2=0.9 per row)",
                         "estimator_lineage": "old 0.5 (model_dual_expfam.py)",
                         "criterion": "C_Q (CSV column BIC)", "family_x": fam[sc][0], "family_y": fam[sc][1],
                         "k_gen": 3, "n": 150, "d": 15, "replicate": t, "candidate_k": f"{ks[0]}..{ks[-1]}",
                         "selected_k": sel, "label": label(sel, 3),
                         "slack_kgen_minus1": bic[2] - bic[3], "slack_kgen_plus1": bic[4] - bic[3],
                         "selection_rule": "per-trial argmin (single start)", "status": "COMPLETE",
                         "source_path": str(p.relative_to(ROOT)).replace("\\", "/")})
    p = RES / "fixed_official" / "exp1_k9" / "fixed_exp1_bic_k1to9_bestk_by_trial.csv"
    for r in read_csv(p):
        sc = r["scenario"]
        sel = int(r["best_k_by_trial_bic"])
        rows.append({"experiment_id": f"fixed_official_exp1_k1to9_scenario_{sc}",
                     "generator_version": "data_generator_expfam.generate_dual_data (Z z-scored, ||f_l||^2=0.9 per row)",
                     "estimator_lineage": "fixed (model_dual_expfam_fixed.py, no 0.5)",
                     "criterion": "C_Q (CSV column BIC)", "family_x": fam[sc][0], "family_y": fam[sc][1],
                     "k_gen": 3, "n": 150, "d": 15, "replicate": int(r["trial"]), "candidate_k": "1..9",
                     "selected_k": sel, "label": label(sel, 3), "slack_kgen_minus1": "", "slack_kgen_plus1": "",
                     "selection_rule": "per-trial argmin (single start)", "status": "COMPLETE",
                     "source_path": str(p.relative_to(ROOT)).replace("\\", "/")})
    p = KSEL / "paper_condition_k_criteria_20261006" / "full" / "selection.csv"
    for r in read_csv(p):
        if r["criterion"] not in ("C_Q", "C_Lap"):
            continue
        k = int(r["k_true"])
        done = r["status"] == "COMPLETE"
        rows.append({"experiment_id": "paper_condition_k_criteria_20261006 (#139 B)",
                     "generator_version": "reproduction/src/data_generator.py::set_true_params (paper-type; X z-scored)",
                     "estimator_lineage": "E (consistent)", "criterion": r["criterion"],
                     "family_x": "gaussian", "family_y": "bernoulli", "k_gen": k, "n": 150, "d": 15,
                     "replicate": int(r["replicate"]), "candidate_k": "1..10",
                     "selected_k": int(r["selected_k"]) if done else "",
                     "label": r["label"] if done else "", "slack_kgen_minus1": "", "slack_kgen_plus1": "",
                     "selection_rule": "mean over 2 starts, argmin", "status": r["status"],
                     "source_path": str(p.relative_to(ROOT)).replace("\\", "/")})
    return rows


# ------------------------------------------------------------------ 2. canonical
def canonical_rows() -> list[dict[str, Any]]:
    rows = []
    p = KSEL / "clap_clean_nsweep_20261006" / "full" / "selection.csv"
    for r in read_csv(p):
        if r["criterion"] not in ("C_Q", "C_Lap"):
            continue
        done = r["status"] == "COMPLETE"
        rows.append({"experiment_id": "clap_clean_nsweep_20261006 (#139 A1)", "criterion": r["criterion"],
                     "k_gen": int(r["k_true"]), "n": int(r["n"]), "d": 15, "L": 5, "em_iterations": 8,
                     "replicate": int(r["replicate"]), "status": r["status"],
                     "selected_k": int(r["selected_k"]) if done else "", "label": r["label"] if done else "",
                     "source_path": str(p.relative_to(ROOT)).replace("\\", "/")})
    p = KSEL / "dsweep_k_selection_20261105" / "full" / "selection.csv"
    for r in read_csv(p):
        done = r["status"] == "COMPLETE"
        rows.append({"experiment_id": "dsweep_k_selection_20261105", "criterion": r["criterion"],
                     "k_gen": 5, "n": int(r["n"]), "d": int(r["d"]), "L": 5, "em_iterations": 8,
                     "replicate": int(r["replicate"]), "status": r["status"],
                     "selected_k": int(r["selected_k"]) if done else "", "label": r["label"] if done else "",
                     "source_path": str(p.relative_to(ROOT)).replace("\\", "/")})
    for n in (50, 100, 150):
        p = KSEL / "k_selection_robustness_20261105" / f"r2_n{n}" / "selection.csv"
        for r in read_csv(p):
            if int(r["d"]) == 15 or int(r["checkpoint"]) != 8:
                continue      # d=15 rows re-evaluate the A1 states already listed above
            done = r["status"] == "COMPLETE"
            rows.append({"experiment_id": "k_selection_robustness_20261105 (R2)", "criterion": r["criterion"],
                         "k_gen": 5, "n": n, "d": int(r["d"]), "L": 5, "em_iterations": 8,
                         "replicate": int(r["replicate"]), "status": r["status"],
                         "selected_k": int(r["selected_k"]) if done else "",
                         "label": r["label"] if done else "",
                         "source_path": str(p.relative_to(ROOT)).replace("\\", "/")})
    return rows


def frequency(rows, keys) -> list[dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        o = out.setdefault(key, {**{k: r[k] for k in keys}, "datasets": 0, "complete": 0,
                                 "exact": 0, "under": 0, "over": 0, "incomplete": 0})
        o["datasets"] += 1
        if r["status"] == "COMPLETE":
            o["complete"] += 1
            o[r["label"]] += 1
        else:
            o["incomplete"] += 1
    return list(out.values())


# ------------------------------------------------------------------ 3. Q split
def q_split(model, X, Y, Zs, num_params: int, n: int) -> dict[str, float]:
    cq = panel.cq_terms(model, X, Y, Zs, num_params=num_params, n=n)
    t = panel.theta_of(model)
    L = Zs.shape[2]
    per = {"X": [], "Y": [], "Z": []}
    from eval_utils import calc_Q_dual_strict_exp
    for l in range(L):
        z = Zs[:, :, l:l + 1]
        tot = float(calc_Q_dual_strict_exp(X, Y, z, t["F"], t["sigma"], t["var_z"], t["w0"], t["w"], model))
        qz = panel.log_prior(Zs[:, :, l], t["var_z"])
        qy = float(model.calc_log_likelihood_Y(Y, z, t["w0"], t["w"]))
        per["Z"].append(qz); per["Y"].append(qy); per["X"].append(tot - qz - qy)
    se = {f"mcse_Q_{c}": float(np.std(v, ddof=1) / math.sqrt(L)) for c, v in per.items()}
    tot_l = np.array(per["X"]) + np.array(per["Y"]) + np.array(per["Z"])
    k = Zs.shape[1]
    var_z = t["var_z"]
    pz_formula = n * k * (math.log(2 * math.pi * var_z) + 1.0)
    return {**{c: cq[c] for c in ("q_strict", "Q_X", "Q_Y", "Q_Z", "D_K", "P_Z", "P_theta", "C_Q")},
            **se, "mcse_Q_total": float(np.std(tot_l, ddof=1) / math.sqrt(L)),
            "mean_z_sq": float(np.mean(Zs ** 2)), "var_z_hat": var_z,
            "P_Z_formula": pz_formula, "P_Z_minus_formula": cq["P_Z"] - pz_formula,
            "closure_mean_of_samples": float(np.mean(tot_l)) - cq["q_strict"]}


def model_for(n, d, k, train, F, sigma, w0, w, var_z, Z):
    m = em_runner.build_model(n, d, k, hist.L_SAMPLES, hist.FAMILY_X, hist.FAMILY_Y,
                              train_mask=train, numerics_mode=hist.NUMERICS_MODE)
    m.initialize_params(true_params=None, seed=0)
    m.params.update({"Z": Z.copy(), "F": F, "sigma": sigma, "w0": w0, "w": w, "var_z": var_z})
    return m


def load_lap_tables() -> dict[tuple, dict[str, str]]:
    """(n, d, rep, k, s) -> stored evaluation row (checkpoint 8, L=5)."""
    out = {}
    for r in read_csv(KSEL / "dsweep_k_selection_20261105" / "full" / "fit_criteria.csv"):
        out[(75, int(r["d"]), int(r["replicate"]), int(r["k_est"]), int(r["start"]))] = r
    for n in (50, 100, 150):
        for r in read_csv(KSEL / "k_selection_robustness_20261105" / f"r2_n{n}" / "fit_criteria.csv"):
            if int(r["L"]) == 5 and int(r["checkpoint"]) == 8:
                out[(n, int(r["d"]), int(r["replicate"]), int(r["k_est"]), int(r["start"]))] = r
    return out


def fit_level_split() -> list[dict[str, Any]]:
    prov = a1.load_hist_provenance()
    emc, _ = a1.load_emc_identity()
    lap = load_lap_tables()
    emc_npz = np.load(a1.EMC_NPZ)
    dsw = np.load(ART / "dsweep_k_selection_20261105" / "fit_states.npz")
    r2 = {n: np.load(ART / "k_selection_robustness_20261105" / f"r2_n{n}_fit_states.npz") for n in (50, 100, 150)}
    rows = []
    for n in (50, 75, 100, 150):
        for rep in range(1, 9):
            ds, train, test, dd, scale = rob.build_data(n, rep, prov)
            for d in ((15, 30, 60) if n == 75 else (15, 30)):
                X = dd[d]["X"]
                for k in hist.CANDIDATE_K:
                    for s in hist.STARTS:
                        if d == 15:
                            fi = int(emc[(K_GEN, n, rep, k, s)]["fit_index"])
                            key = f"{fi:04d}_it008"; src = emc_npz; pre = key
                            origin = f"emc_npz:{key}"
                        elif n == 75:
                            pre = f"d{d}_rep{rep}_k{k}_s{s}"; src = dsw; origin = f"dsweep_npz:{pre}"
                        else:
                            pre = f"n{n}_d{d}_L5_cp8_r{rep}_k{k}_s{s}"; src = r2[n]; origin = f"r2_npz:{pre}"
                        Zs = np.asarray(src[f"{pre}_Zsamples"], float)
                        Z = np.asarray(src[f"{pre}_Z"], float)
                        F = np.asarray(src[f"{pre}_F"], float)
                        sig = np.asarray(src[f"{pre}_sigma"], float)
                        w0, w, vz = (float(v) for v in src[f"{pre}_scalars"])
                        m = model_for(n, d, k, train, F, sig, w0, w, vz, Z)
                        npar = k * d - k * (k - 1) // 2
                        q = q_split(m, X, ds.Y, Zs, npar, n)
                        ref = lap[(n, d, rep, k, s)]
                        need(math.isclose(q["C_Q"], float(ref["C_Q"]), rel_tol=1e-10, abs_tol=1e-6),
                             f"C_Q differs from stored CSV {(n, d, rep, k, s)}")
                        row = {"n": n, "d": d, "replicate": rep, "k_est": k, "start": s,
                               "state_source": origin, "num_params": npar, **q,
                               "C_Q_stored": float(ref["C_Q"]), "lap_status": ref["lap_status"]}
                        for c in ("C_Lap", "lap_fit_mode", "lap_volume", "lap_penalty", "lap_logdet_H",
                                  "logdet_prior_X_hat", "logdet_Y_increment_hat", "termA", "R3",
                                  "gram_rel_change", "w_hat", "w0_hat"):
                            row[c] = float(ref[c]) if ref.get(c, "") not in ("", None) else ""
                        rows.append(row)
            print(f"n={n} rep={rep} split done", flush=True)
    return rows


def increments(rows) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(r["n"], r["d"], r["replicate"]) for r in rows})
    for n, d, rep in keys:
        by = {(r["k_est"], r["start"]): r for r in rows if (r["n"], r["d"], r["replicate"]) == (n, d, rep)}
        mean = lambda k, c: float(np.mean([float(by[(k, s)][c]) for s in hist.STARTS]))  # noqa: E731
        lap_ok = lambda k: all(by[(k, s)]["lap_status"] == "OK" for s in hist.STARTS)  # noqa: E731
        for k in range(1, 7):
            k1 = k + 1
            gx = 2 * (mean(k1, "Q_X") - mean(k, "Q_X"))
            gy = 2 * (mean(k1, "Q_Y") - mean(k, "Q_Y"))
            pc = -2 * (mean(k1, "Q_Z") - mean(k, "Q_Z"))
            pp = mean(k1, "P_theta") - mean(k, "P_theta")
            sd = mean(k, "C_Q") - mean(k1, "C_Q")
            se = math.sqrt(sum((2 * float(by[(kk, s)]["mcse_Q_total"]) / 2) ** 2
                               for kk in (k, k1) for s in hist.STARTS))
            row = {"experiment_id": "canonical_existing_states", "generator_version": "canonical-clean-v1 (+appended blocks for d>15)",
                   "estimator_lineage": "E (consistent)", "criterion": "C_Q", "family_x": "poisson",
                   "family_y": "bernoulli", "k_gen": K_GEN, "n": n, "d": d, "replicate": rep,
                   "k_from": k, "k_to": k1, "fit_gain_x": gx, "fit_gain_y": gy, "fit_gain_total": gx + gy,
                   "prior_cost": pc, "prior_cost_formula": n * C2, "parameter_penalty": pp,
                   "laplace_volume_term": "", "score_delta": sd, "selection_slack": sd,
                   "closure_error": sd - (gx + gy - pc - pp), "mc_se_score_delta_approx": se,
                   "status": "OK", "source_path": "existing_reanalysis/fit_level_q_split.csv"}
            out.append(row)
            if lap_ok(k) and lap_ok(k1):
                fm = mean(k, "lap_fit_mode") - mean(k1, "lap_fit_mode")
                vol = mean(k1, "lap_volume") - mean(k, "lap_volume")
                sl = mean(k, "C_Lap") - mean(k1, "C_Lap")
                out.append({**row, "criterion": "C_Lap", "fit_gain_x": "", "fit_gain_y": "",
                            "fit_gain_total": fm, "prior_cost": "", "prior_cost_formula": "",
                            "laplace_volume_term": vol, "score_delta": sl, "selection_slack": sl,
                            "closure_error": sl - (fm - vol - pp), "mc_se_score_delta_approx": "",
                            "dlogdet_H": mean(k1, "lap_logdet_H") - mean(k, "lap_logdet_H"),
                            "dlogdet_prior_X": mean(k1, "logdet_prior_X_hat") - mean(k, "logdet_prior_X_hat"),
                            "dlogdet_Y_increment": mean(k1, "logdet_Y_increment_hat") - mean(k, "logdet_Y_increment_hat"),
                            "status": "OK"})
            else:
                out.append({**{c: row[c] for c in ("experiment_id", "generator_version", "estimator_lineage",
                                                    "family_x", "family_y", "k_gen", "n", "d", "replicate",
                                                    "k_from", "k_to", "source_path")},
                            "criterion": "C_Lap", "status": "LAPLACE_NOT_OK (not imputed)"})
    return out


def paired_criteria(canon) -> list[dict[str, Any]]:
    by = {}
    for r in canon:
        if r["k_gen"] != 5:
            continue
        by.setdefault((r["experiment_id"], r["n"], r["d"], r["replicate"]), {})[r["criterion"]] = r
    out = []
    for (e, n, d, rep), v in sorted(by.items()):
        q, l = v.get("C_Q"), v.get("C_Lap")
        if not q or not l:
            continue
        both = q["status"] == "COMPLETE" and l["status"] == "COMPLETE"
        out.append({"experiment_id": e, "k_gen": 5, "n": n, "d": d, "replicate": rep,
                    "selected_k_C_Q": q["selected_k"], "selected_k_C_Lap": l["selected_k"] if both else "",
                    "C_Lap_status": l["status"],
                    "relation": ("" if not both else ("Q<Lap" if q["selected_k"] < l["selected_k"]
                                                       else "Q=Lap" if q["selected_k"] == l["selected_k"] else "Q>Lap")),
                    "source_path": q["source_path"]})
    return out


CONDITION_MATRIX = [
    # (series, field -> value) ; values transcribed from code (FACT) with file:line pointers in 'source'
    {"series": "initial Dual-ExpFam Scenario A/B/C (0.5)", "family_x": "A pois / B gauss / C bern",
     "family_y": "A bern / B pois / C gauss", "k_gen": 3, "candidate_k": "1..6", "n": 150, "d": 15,
     "z_generation": "N(0,I) then column z-score (normalize_zscore)", "z_normalisation": "yes (sample)",
     "f_generation": "N(0,5) rows, each row rescaled to ||f_l||^2 = 1-uniq = 0.9",
     "f_singular_values": "random (not controlled); F^T F approx 0.9 d/K I = 4.5 I on average",
     "loading_energy_per_direction": "approx 4.5 (= 0.9*15/3)", "w0": "-1 (bern) / 0 (pois, gauss)",
     "w": "1.5 (bern) / 0.5 (pois, gauss)", "w2_times_k": "6.75 (bern)", "x_post_standardisation": "Gaussian-X only (z-score)",
     "observation_mask": "none (all pairs)", "L": 5, "em_iterations": 8, "starts": "1 (single fit per trial)",
     "init": "random + informed w0/w + F scaling", "criterion": "C_Q (calc_bic_dual)",
     "selection": "argmin per trial", "lineage": "0.5 (model_dual_expfam.py)",
     "failure_handling": "up to 2 retries with halved Newton step (run_em_dual)",
     "source": "expfam/src/data_generator_expfam.py:223-330; expfam/src/exp_scenario_lib.py:39-44,66-90; expfam/src/utils_expfam.py:407-460"},
    {"series": "fixed_official Exp1 K1..9", "family_x": "as above", "family_y": "as above", "k_gen": 3,
     "candidate_k": "1..9", "n": 150, "d": 15, "z_generation": "as above", "z_normalisation": "yes",
     "f_generation": "as above", "f_singular_values": "as above", "loading_energy_per_direction": "approx 4.5",
     "w0": "as above", "w": "as above", "w2_times_k": "6.75 (bern)", "x_post_standardisation": "Gaussian-X only",
     "observation_mask": "none", "L": 5, "em_iterations": 8, "starts": "1", "init": "as above",
     "criterion": "C_Q", "selection": "argmin per trial", "lineage": "fixed (no 0.5)",
     "failure_handling": "retries (run_em_dual style)", "source": "expfam/src/run_fixed_official_exp1_bic_k9_extension.py"},
    {"series": "paper-type #139 B (lineage E)", "family_x": "gaussian", "family_y": "bernoulli",
     "k_gen": "1,3,5,7,9", "candidate_k": "1..10", "n": 150, "d": 15,
     "z_generation": "reproduction set_true_params", "z_normalisation": "see generator",
     "f_generation": "var_f=5, uniq=0.1, w_scale=3 (paper-type)", "f_singular_values": "random",
     "loading_energy_per_direction": "not controlled", "w0": "random per dataset", "w": "random (w_scale=3)",
     "w2_times_k": "random", "x_post_standardisation": "yes (X z-scored; K* undefined)",
     "observation_mask": "none (heldout=false)", "L": 5, "em_iterations": 8, "starts": 2, "init": "lineage E",
     "criterion": "C_Q and C_Lap (same fits)", "selection": "mean over 2 starts, argmin", "lineage": "E",
     "failure_handling": "fail fast; non-OK C_Lap -> INCOMPLETE",
     "source": "expfam/results/k_selection/paper_condition_k_criteria_20261006/full/protocol.json"},
    {"series": "canonical-clean-v1 A1 / d-sweep / R2 (lineage E)", "family_x": "poisson", "family_y": "bernoulli",
     "k_gen": "1,3,5 (A1); 5 (d-sweep, R2)", "candidate_k": "1..7", "n": "50,75,100,150", "d": "15 (A1); 15,30,60",
     "z_generation": "N(0,I), never normalised", "z_normalisation": "no",
     "f_generation": "reduced QR, all singular values f_scale; AVERAGE ||f_l||^2 = 0.5",
     "f_singular_values": "all equal: sqrt(0.5 d/K) (K=5,d=15: 1.2247)",
     "loading_energy_per_direction": "0.5 d/K (K=5,d=15: 1.5; K=3: 2.5)", "w0": -1.0,
     "w": "sqrt(3/K) (K=5: 0.7746)", "w2_times_k": 3.0, "x_post_standardisation": "none",
     "observation_mask": "train/test pair split, test_ratio 0.2", "L": 5, "em_iterations": 8, "starts": 2,
     "init": "lineage E em_runner", "criterion": "C_Q and C_Lap (same fits)",
     "selection": "mean over 2 starts, argmin", "lineage": "E",
     "failure_handling": "on_nan=fail; no retry; Laplace non-OK -> INCOMPLETE",
     "source": "tools/research_audit/run_clean_true_k_sweep.py:67-98; expfam/src/experimental/data_generator_canonical.py:127-201"},
]


def main() -> int:
    need(not OUT.exists(), f"output exists: {OUT}")
    OUT.mkdir(parents=True)
    hrows = historical_rows()
    crows = canonical_rows()
    write_csv(OUT / "historical_k_selection_summary.csv", hrows)
    write_csv(OUT / "historical_k_selection_frequency.csv",
              frequency(hrows, ("experiment_id", "criterion", "k_gen")))
    write_csv(OUT / "canonical_k_selection_summary.csv", crows)
    write_csv(OUT / "canonical_k_selection_frequency.csv",
              frequency(crows, ("experiment_id", "criterion", "k_gen", "n", "d")))
    write_csv(OUT / "historical_canonical_condition_matrix.csv", CONDITION_MATRIX)
    write_csv(OUT / "criterion_paired_comparison.csv", paired_criteria(crows))
    fl = fit_level_split()
    write_csv(OUT / "fit_level_q_split.csv", fl)
    inc = increments(fl)
    write_csv(OUT / "k_increment_decomposition.csv", inc)
    info = {"script": "tools/research_audit/analyze_k_causal_existing.py", "git_head": a1.git_head(),
            "mcem_fits": 0, "fit_states_evaluated": len(fl),
            "max_abs_P_Z_minus_formula": max(abs(r["P_Z_minus_formula"]) for r in fl),
            "max_abs_increment_closure": max(abs(r["closure_error"]) for r in inc if r.get("closure_error", "") != ""),
            "inputs_sha256": {str(p): sha256_file(p) for p in [
                a1.EMC_NPZ, ART / "dsweep_k_selection_20261105" / "fit_states.npz",
                *[ART / "k_selection_robustness_20261105" / f"r2_n{n}_fit_states.npz" for n in (50, 100, 150)]]},
            "outputs_sha256": {p.name: sha256_file(p) for p in sorted(OUT.glob("*.csv"))}}
    (OUT / "runinfo.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "outputs_sha256"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
