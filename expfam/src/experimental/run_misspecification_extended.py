"""Fixed, resumable extension of M1/E1. Run --prepare before --execute."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import wraps
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (HERE, HERE.parent, ROOT / "tools" / "research_audit"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_misspecification_m1 as m1
import run_z_point_estimator_e1 as e1
from diagnostics import validate_xy
from eval_utils import upper_pairs_of

OUT = ROOT / "expfam" / "results" / "misspecification_extended" / "phase_20261110"
RATES = (1.0, 0.5, 0.25, 0.1)
PRIMARY = ("correct", "bp6", "bg6", "pg3", "allg")
INTERMEDIATE = ("bp2", "bp4", "bg2", "bg4", "pg1", "pg2")
SEED_SERIES = {"A": m1.MODEL_SEED_BASE, "B": e1.PARTS["B"]}
FIELDNAMES = (
    "experiment_id", "condition_id", "replicate", "data_seed", "model_seed",
    "seed_series", "arm", "wrong_family", "wrong_columns", "n_wrong_columns",
    "y_rate", "n_train_pairs", "n_test_pairs", "family_x_true", "family_x_fit",
    "status", "source", "error", "rmse_mode", "rmse_last", "rmse_mean5",
    "gram_corr", "norm_ratio", "heldout_log_score", "heldout_auc", "heldout_ap",
    "w0_hat", "w_hat", "abs_w0_err", "abs_w_err", "mode_status",
    "mode_grad_inf", "mode_min_eig_H", "mode_iterations", "mode_phi",
    "mode_from_mean5_status", "modes_max_abs_diff", "n_warnings",
    "warning_messages", "nan_occurred", "retry_count", "replacement_count",
    "seed_rescue_count", "runtime_s",
)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, text=True,
                          capture_output=True, check=False).stdout.strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, columns=None):
    columns = list(columns or (list(rows[0]) if rows else []))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row["experiment_id"]


def column_order(r, family):
    columns = list(range(3, 9)) if family == "b" else list(range(9, 12))
    seed = 1610000 + 10 * r + (0 if family == "b" else 1)
    return list(np.random.default_rng(seed).permutation(columns).astype(int))


def assignment(r, arm):
    true = list(m1.FAMILY_X_TRUE)
    if arm == "correct":
        return true, [], ""
    if arm == "allg":
        cols = list(range(3, 12))
        target = "gaussian"
    else:
        source = arm[0]
        count = int(arm[2:])
        cols = column_order(r, source)[:count]
        target = "poisson" if arm[1] == "p" else "gaussian"
    fit = true.copy()
    for c in cols:
        fit[c] = target
    return fit, cols, target


def masks(r):
    old = m1.build_masks(r)
    pool, sparse = old["pool"], old["sparse"]
    rows, cols = upper_pairs_of(pool)
    extra = np.flatnonzero(~sparse[rows, cols])
    permutation = np.random.default_rng(1620000 + r).permutation(extra)
    selected = list(np.flatnonzero(sparse[rows, cols]))
    result = {1.0: old["dense"], 0.1: sparse, "test": old["test"], "pool": pool}
    for rate in (0.25, 0.5):
        count = int(round(len(rows) * rate))
        ids = np.asarray(selected + list(permutation[:count - len(selected)]), int)
        mask = np.zeros_like(pool)
        mask[rows[ids], cols[ids]] = True
        mask |= mask.T
        result[rate] = mask
    return result


def plan():
    rows = []
    for r, data_seed in enumerate(m1.DATA_SEEDS):
        mm = masks(r)
        for rate in RATES:
            arms = PRIMARY + (INTERMEDIATE if rate in (1.0, 0.1) else ())
            for series, seed_base in SEED_SERIES.items():
                for arm in arms:
                    fam, cols, target = assignment(r, arm)
                    experiment_id = f"MX2-r{r+1:02d}-{rate:g}-{series}-{arm}"
                    rows.append({
                        "experiment_id": experiment_id,
                        "condition_id": f"{arm}@{rate:g}", "replicate": f"rep{r+1:02d}",
                        "data_seed": data_seed, "model_seed": seed_base + r,
                        "seed_series": series, "arm": arm, "wrong_family": target,
                        "wrong_columns": "|".join(map(str, cols)),
                        "n_wrong_columns": len(cols), "y_rate": rate,
                        "n_train_pairs": m1.n_upper(mm[rate]),
                        "n_test_pairs": m1.n_upper(mm["test"]),
                        "family_x_true": "|".join(m1.FAMILY_X_TRUE),
                        "family_x_fit": "|".join(fam), "status": "PLANNED",
                    })
    assert len(rows) == 1280 and len({key(row) for row in rows}) == 1280
    return rows


def preflight(rows):
    assert len(rows) == 1280
    old = {x["replicate"]: x for x in read_csv(m1.OUT_DIR / "mask_provenance.csv")}
    records = []
    for r, seed in enumerate(m1.DATA_SEEDS):
        ds, mm = m1.generate(seed), masks(r)
        rep = f"rep{r+1:02d}"
        assert m1.sha16(ds.Z, ds.F, ds.X, ds.Y) == old[rep]["hash_dataset"]
        assert m1.sha16(mm["test"]) == old[rep]["hash_test"]
        assert m1.sha16(mm[1.0]) == old[rep]["hash_dense"]
        assert m1.sha16(mm[0.1]) == old[rep]["hash_sparse"]
        assert all(m1.n_upper(mm[rate]) == int(round(2497 * rate)) for rate in RATES)
        for small, large in ((0.1, 0.25), (0.25, 0.5), (0.5, 1.0)):
            assert np.all(mm[large][mm[small]])
        assert not np.any(mm["test"] & mm[1.0])
        for arm in PRIMARY + INTERMEDIATE:
            fam, cols, _ = assignment(r, arm)
            assert [i for i, (a, b) in enumerate(zip(m1.FAMILY_X_TRUE, fam))
                    if a != b] == sorted(cols) or set(
                        i for i, (a, b) in enumerate(zip(m1.FAMILY_X_TRUE, fam))
                        if a != b) == set(cols)
            validate_xy(ds.X, ds.Y, family_x_list=fam, family_y="bernoulli",
                        train_mask=mm[1.0], allow_support_mismatch=False)
        records.append({
            "replicate": rep, "data_seed": seed, "dataset_sha16": old[rep]["hash_dataset"],
            "test_sha16": old[rep]["hash_test"],
            **{f"train_{rate:g}_sha16": m1.sha16(mm[rate]) for rate in RATES},
            **{f"train_{rate:g}_pairs": m1.n_upper(mm[rate]) for rate in RATES},
            "test_pairs": m1.n_upper(mm["test"]),
            "bernoulli_order": "|".join(map(str, column_order(r, "b"))),
            "poisson_order": "|".join(map(str, column_order(r, "p"))),
        })
    return records


def protocol(rows, provenance):
    return {
        "experiment": "MX2 extended X-family misspecification", "frozen_before_new_fits": True,
        "source_head": git("rev-parse", "HEAD"),
        "lineage": "E-consistent per-column, experimental prototype; manuscript adoption requires human gate",
        "generator": "canonical-clean-mixed-v1", "n": 75, "d": 12,
        "K_true": 3, "K_fit": 3, "family_x_true": m1.FAMILY_X_TRUE,
        "family_y": "bernoulli", "sigma_x_var": 1.0,
        "f_scale": math.sqrt(2), "w0_true": -1.0, "w_true": 1.0,
        "data_seeds": list(m1.DATA_SEEDS), "model_seed_series": SEED_SERIES,
        "test_seed_rule": "1341000+r, exact M1 test",
        "sparse_seed_rule": "1342000+10r+1, exact M1 10% train",
        "extension_seed_rule": "1620000+r permutes pool excluding sparse; extend to rounded 25%,50%",
        "column_seed_rule": "1610000+10r+{0 Bernoulli,1 Poisson}; nested prefixes",
        "rates_of_train_candidate_pool": RATES, "test_ratio_of_all_pairs": 0.1,
        "arms_primary": PRIMARY, "arms_intermediate": INTERMEDIATE,
        "planned_unique_fits": len(rows), "planned_reused_e1": 160,
        "planned_new_fits": 1120, "pilot_new_fits": 10,
        "L": 5, "mcem_iterations": 8, "numerics_mode": "consistent",
        "failure_policy": "fail_fast", "validate_support": True,
        "allow_support_mismatch": False, "retry": False, "family_selection": False,
        "primary": ["mode Procrustes rotation-only absolute RMSE", "paired wrong-minus-correct RMSE"],
        "secondary": ["last and mean5 RMSE", "Gram correlation", "norm ratio",
                      "heldout Bernoulli log score/AUC/AP", "w and w0 error",
                      "mode gradient/curvature", "warnings", "runtime"],
        "mode": {"start": "last sample", "grad_tol": 1e-8, "max_iter": 200,
                 "secondary_start": "mean5", "valid_status": "OK"},
        "paired_validity": "both EM and mode status OK and all scored metrics finite",
        "failure_handling": "record failed cell once; no retry/reseed/replacement/imputation",
        "scientific_success": "no threshold or favorable-direction requirement",
        "reproduction_tolerances": {"exact": 1e-12, "investigate_fp": 1e-6},
        "bootstrap_unit": "data replicate; average two model seeds within replicate",
        "e1_source_sha256": {name: sha256(e1.OUT_DIR / "full" / name)
                             for name in ("fit_runs.csv", "fit_estimators.csv")},
        "mask_provenance_sha256": hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
    }


def import_e1(rows):
    runs = {(x["part"], x["replicate"], x["regime"], x["arm"]): x
            for x in read_csv(e1.OUT_DIR / "full" / "fit_runs.csv")}
    estimates = {(x["part"], x["replicate"], x["regime"], x["arm"], x["estimator"]): x
                 for x in read_csv(e1.OUT_DIR / "full" / "fit_estimators.csv")}
    out = []
    for cell in rows:
        if cell["arm"] not in ("correct", "bp6") or float(cell["y_rate"]) not in (1.0, 0.1):
            continue
        part, rep = cell["seed_series"], cell["replicate"]
        regime = "dense" if float(cell["y_rate"]) == 1.0 else "sparse"
        arm = "correct" if cell["arm"] == "correct" else "wrong"
        run = runs[(part, rep, regime, arm)]
        est = {name: estimates[(part, rep, regime, arm, name)]
               for name in ("mode", "last", "mean5")}
        assert int(run["model_seed"]) == int(cell["model_seed"])
        assert int(run["n_train_pairs"]) == int(cell["n_train_pairs"])
        assert run["status"] == "OK" and run["mode_status"] == "OK"
        out.append({
            **cell, "status": "OK", "source": "E1/full", "error": "",
            "rmse_mode": est["mode"]["rmse_Z"], "rmse_last": est["last"]["rmse_Z"],
            "rmse_mean5": est["mean5"]["rmse_Z"], "gram_corr": est["mode"]["gram_corr"],
            "norm_ratio": est["mode"]["z_est_norm_ratio"],
            "heldout_log_score": est["mode"]["heldout_mean_log_score"],
            "heldout_auc": est["mode"]["heldout_auc"], "heldout_ap": "",
            "w0_hat": run["w0_hat"], "w_hat": run["w_hat"],
            "abs_w0_err": abs(float(run["w0_hat"]) - m1.W0_TRUE),
            "abs_w_err": abs(float(run["w_hat"]) - m1.W_TRUE),
            "mode_status": run["mode_status"], "mode_grad_inf": run["mode_grad_inf"],
            "mode_min_eig_H": run["mode_min_eig_H"],
            "mode_iterations": run["mode_iterations"], "mode_phi": run["mode_phi"],
            "mode_from_mean5_status": run["mode_from_mean5_status"],
            "modes_max_abs_diff": run["modes_max_abs_diff"],
            "n_warnings": run["n_warnings"], "warning_messages": run["warning_messages"],
            "nan_occurred": run["nan_occurred"], "retry_count": run["retry_count"],
            "replacement_count": run["replacement_count"],
            "seed_rescue_count": run["seed_rescue_count"],
            "runtime_s": run["wall_runtime_s"],
        })
    assert len(out) == 160
    return out


def prepare():
    if OUT.exists():
        raise FileExistsError(f"output exists: {OUT}; use --execute to resume")
    rows = plan()
    provenance = preflight(rows)
    OUT.mkdir(parents=True)
    (OUT / "protocol.json").write_text(
        json.dumps(protocol(rows, provenance), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_csv(OUT / "mask_provenance.csv", provenance)
    reused = import_e1(rows)
    reused_keys = {key(x) for x in reused}
    write_csv(OUT / "manifest.csv",
              [{**row, "status": "REUSED_E1" if key(row) in reused_keys else "PLANNED"}
               for row in rows])
    write_csv(OUT / "fit_results.csv", reused, FIELDNAMES)
    (OUT / "runinfo.json").write_text(json.dumps({
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git("rev-parse", "HEAD"), "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(),
        "status": "PREPARED", "n_reused": len(reused)}, indent=2), encoding="utf-8")
    print(f"prepared {len(rows)} unique cells, {len(reused)} E1 reused, 1120 new")


def run_one(cell, ds, mm):
    from em_runner import run_em_experimental
    from sklearn.metrics import average_precision_score
    row = {**cell, "source": "MX2", "status": "FAILED", "error": ""}
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            fit = run_em_experimental(
                ds.X, ds.Y, family_x=None, family_y="bernoulli", k=3, L=5,
                num_iter=8, seed=int(cell["model_seed"]), train_mask=mm[float(cell["y_rate"])],
                family_x_list=cell["family_x_fit"].split("|"), compute_strict_Q=False,
                validate_support=True, allow_support_mismatch=False,
                numerics_mode="consistent", failure_policy="fail_fast")
            samples = np.asarray(fit["Z_samples"], float)
            assert np.array_equal(samples[:, :, -1], fit["Z_est"])
            mode = e1.joint_mode(fit["model"], ds.X, ds.Y, samples[:, :, -1])
            mode2 = e1.joint_mode(fit["model"], ds.X, ds.Y, samples.mean(axis=2))
            scores = {}
            for name, z in (("last", samples[:, :, -1]), ("mean5", samples.mean(axis=2))):
                scores[name] = e1.score_estimate(z, ds.Z, ds.Y, mm["test"],
                                                 fit["w0"], fit["w"])
            if mode["status"] == "OK":
                scores["mode"] = e1.score_estimate(mode["Z_hat"], ds.Z, ds.Y,
                                                   mm["test"], fit["w0"], fit["w"])
                rr, cc = upper_pairs_of(mm["test"])
                eta = fit["w0"] + fit["w"] * np.sum(
                    mode["Z_hat"][rr] * mode["Z_hat"][cc], axis=1)
                ap = float(average_precision_score(ds.Y[rr, cc],
                                                     1 / (1 + np.exp(-eta))))
            else:
                ap = float("nan")
            row.update({
                "status": "OK" if mode["status"] == "OK" else "MODE_INVALID",
                "rmse_mode": scores.get("mode", {}).get("rmse_Z", ""),
                "rmse_last": scores["last"]["rmse_Z"],
                "rmse_mean5": scores["mean5"]["rmse_Z"],
                "gram_corr": scores.get("mode", {}).get("gram_corr", ""),
                "norm_ratio": scores.get("mode", {}).get("z_est_norm_ratio", ""),
                "heldout_log_score": scores.get("mode", {}).get("heldout_mean_log_score", ""),
                "heldout_auc": scores.get("mode", {}).get("heldout_auc", ""),
                "heldout_ap": ap if np.isfinite(ap) else "",
                "w0_hat": fit["w0"], "w_hat": fit["w"],
                "abs_w0_err": abs(float(fit["w0"]) - m1.W0_TRUE),
                "abs_w_err": abs(float(fit["w"]) - m1.W_TRUE),
                "mode_status": mode["status"], "mode_grad_inf": mode["grad_inf"],
                "mode_min_eig_H": mode["min_eig_H"],
                "mode_iterations": mode["iterations"], "mode_phi": mode["phi"],
                "mode_from_mean5_status": mode2["status"],
                "modes_max_abs_diff": (float(np.max(np.abs(mode["Z_hat"] - mode2["Z_hat"])))
                                       if mode["Z_hat"] is not None and
                                       mode2["Z_hat"] is not None else ""),
                "nan_occurred": fit["nan_occurred"],
                "retry_count": fit["retry_count"], "replacement_count": fit["replacement_count"],
                "seed_rescue_count": fit["seed_rescue_count"],
            })
            vals = [row[k] for k in ("rmse_last", "rmse_mean5", "w0_hat", "w_hat")]
            if row["status"] == "OK":
                vals += [row[k] for k in ("rmse_mode", "gram_corr", "norm_ratio",
                                           "heldout_log_score", "heldout_auc")]
            if not all(np.isfinite(float(v)) for v in vals):
                row["status"] = "NONFINITE_METRIC"
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["n_warnings"] = len(recorded)
        row["warning_messages"] = " || ".join(
            sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in recorded}))[:1000]
    row["runtime_s"] = round(time.perf_counter() - start, 3)
    return row


def run_cell(cell):
    r = int(cell["replicate"][3:]) - 1
    return run_one(cell, m1.generate(int(cell["data_seed"])), masks(r))


def append_result(row):
    with (OUT / "fit_results.csv").open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDNAMES, extrasaction="ignore",
                                lineterminator="\n")
        writer.writerow(row)


def single_writer(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        # Windows kernel lock is released automatically if the owner exits.
        import msvcrt
        with (OUT / "execution.lock").open("a+b") as stream:
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"1")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another MX2 writer is active") from exc
            try:
                return function(*args, **kwargs)
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    return guarded


@single_writer
def execute(pilot=False, limit=None, workers=1):
    manifest = read_csv(OUT / "manifest.csv")
    done = {key(x) for x in read_csv(OUT / "fit_results.csv")}
    if len(done) != len(read_csv(OUT / "fit_results.csv")):
        raise RuntimeError("duplicate fit result key")
    if pilot:
        wanted = [x for x in manifest if x["replicate"] == "rep01" and
                  x["seed_series"] == "A" and
                  ((float(x["y_rate"]) == 0.25 and x["arm"] in PRIMARY) or
                   (float(x["y_rate"]) == 1.0 and x["arm"] in INTERMEDIATE[:-1]))]
    else:
        wanted = ([x for x in manifest if x["arm"] in PRIMARY] +
                  [x for x in manifest if x["arm"] in INTERMEDIATE])
    pending = [x for x in wanted if key(x) not in done]
    if limit is not None:
        pending = pending[:limit]
    if workers == 1:
        completed = ((cell, run_cell(cell)) for cell in pending)
        for index, (cell, row) in enumerate(completed, 1):
            append_result(row)
            if index % 10 == 0 or index == len(pending):
                print(f"{index}/{len(pending)} new; latest {key(cell)} {row['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_cell, cell): cell for cell in pending}
            for index, future in enumerate(as_completed(futures), 1):
                cell = futures[future]
                row = future.result()
                append_result(row)
                if index % 10 == 0 or index == len(pending):
                    print(f"{index}/{len(pending)} new; latest {key(cell)} {row['status']}", flush=True)
    update_manifest()


def update_manifest():
    rows = read_csv(OUT / "manifest.csv")
    results = {key(x): x for x in read_csv(OUT / "fit_results.csv")}
    assert len(results) == len(read_csv(OUT / "fit_results.csv"))
    for row in rows:
        if key(row) in results:
            row["status"] = "REUSED_E1" if results[key(row)]["source"] == "E1/full" else results[key(row)]["status"]
    write_csv(OUT / "manifest.csv", rows)


def repair_duplicate_incident():
    """Preserve raw incident CSV, then keep first recorded result per unique key."""
    source = OUT / "fit_results.csv"
    archive = OUT / "fit_results_duplicate_incident_raw.csv"
    if archive.exists():
        raise FileExistsError(archive)
    rows = read_csv(source)
    first, duplicates = {}, []
    for line, row in enumerate(rows, 2):
        if key(row) in first:
            original = first[key(row)]
            differences = [name for name in FIELDNAMES if row.get(name) != original.get(name)]
            duplicates.append({
                "experiment_id": key(row), "duplicate_csv_line": line,
                "different_fields": "|".join(differences),
                "first_status": original["status"], "duplicate_status": row["status"],
                "first_rmse_mode": original["rmse_mode"],
                "duplicate_rmse_mode": row["rmse_mode"],
                "first_runtime_s": original["runtime_s"],
                "duplicate_runtime_s": row["runtime_s"],
            })
        else:
            first[key(row)] = row
    if not duplicates:
        raise RuntimeError("no duplicate keys; repair not needed")
    shutil.copy2(source, archive)
    write_csv(OUT / "duplicate_incident.csv", duplicates)
    write_csv(source, list(first.values()), FIELDNAMES)
    update_manifest()
    print(f"archived {len(rows)} raw rows; preserved {len(first)} first results; "
          f"excluded {len(duplicates)} duplicate rows")


def analyze():
    manifest = read_csv(OUT / "manifest.csv")
    fits = read_csv(OUT / "fit_results.csv")
    assert len({key(x) for x in fits}) == len(fits)
    assert set(key(x) for x in fits) <= set(key(x) for x in manifest)
    valid = {(x["replicate"], x["seed_series"], x["y_rate"], x["arm"]): x
             for x in fits if x["status"] == "OK" and x["mode_status"] == "OK"}
    pairs = []
    for row in manifest:
        if row["arm"] == "correct":
            continue
        k = (row["replicate"], row["seed_series"], row["y_rate"])
        c, w = valid.get((*k, "correct")), valid.get((*k, row["arm"]))
        delta = float(w["rmse_mode"]) - float(c["rmse_mode"]) if c and w else ""
        pairs.append({
            "experiment_id": row["experiment_id"], "condition_id": row["condition_id"],
            "replicate": row["replicate"], "data_seed": row["data_seed"],
            "model_seed": row["model_seed"], "seed_series": row["seed_series"],
            "arm": row["arm"], "n_wrong_columns": row["n_wrong_columns"],
            "wrong_columns": row["wrong_columns"], "y_rate": row["y_rate"],
            "rmse_correct": c["rmse_mode"] if c else "",
            "rmse_wrong": w["rmse_mode"] if w else "", "delta_rmse": delta,
            "relative_change": delta / float(c["rmse_mode"]) if c and w else "",
            "valid": bool(c and w),
        })
    write_csv(OUT / "paired_rmse_z.csv", pairs)
    by_dataset = []
    for condition in sorted({x["condition_id"] for x in pairs}):
        for rep in (f"rep{r+1:02d}" for r in range(20)):
            rows = [x for x in pairs if x["condition_id"] == condition and
                    x["replicate"] == rep and x["valid"]]
            by_dataset.append({
                "condition_id": condition, "replicate": rep, "valid_seed_count": len(rows),
                "mean_delta_two_seeds": (float(np.mean([x["delta_rmse"] for x in rows]))
                                          if len(rows) == 2 else ""),
            })
    write_csv(OUT / "dataset_mean_delta.csv", by_dataset)
    pair_index = {(x["replicate"], x["seed_series"], x["arm"], x["y_rate"]): x
                  for x in pairs if x["valid"]}
    interaction = []
    for rep in (f"rep{r+1:02d}" for r in range(20)):
        for series in SEED_SERIES:
            for arm in PRIMARY[1:] + INTERMEDIATE:
                sparse = pair_index.get((rep, series, arm, "0.1"))
                dense = pair_index.get((rep, series, arm, "1.0"))
                if sparse and dense:
                    interaction.append({
                        "replicate": rep, "seed_series": series, "arm": arm,
                        "delta_sparse": sparse["delta_rmse"],
                        "delta_dense": dense["delta_rmse"],
                        "sparse_minus_dense": sparse["delta_rmse"] - dense["delta_rmse"],
                    })
    write_csv(OUT / "interaction_by_dataset.csv", interaction,
              ("replicate", "seed_series", "arm", "delta_sparse",
               "delta_dense", "sparse_minus_dense"))
    summary = []
    for condition in sorted({x["condition_id"] for x in manifest}):
        rows = [x for x in manifest if x["condition_id"] == condition]
        done = [x for x in fits if x["condition_id"] == condition]
        pp = [x for x in pairs if x["condition_id"] == condition and x["valid"]]
        deltas = np.array([float(x["delta_rmse"]) for x in pp])
        correct = np.array([float(x["rmse_correct"]) for x in pp])
        wrong = np.array([float(x["rmse_wrong"]) for x in pp])
        summary.append({
            "condition_id": condition, "planned_fits": len(rows),
            "attempted_fits": len(done), "successful_fits": sum(x["status"] == "OK" for x in done),
            "failed_fits": sum(x["status"] != "OK" for x in done),
            "valid_pairs": len(pp),
            "rmse_correct_mean": float(correct.mean()) if len(correct) else "",
            "rmse_correct_median": float(np.median(correct)) if len(correct) else "",
            "rmse_correct_sd": float(correct.std(ddof=1)) if len(correct) > 1 else "",
            "rmse_correct_q1": float(np.quantile(correct, .25)) if len(correct) else "",
            "rmse_correct_q3": float(np.quantile(correct, .75)) if len(correct) else "",
            "rmse_wrong_mean": float(wrong.mean()) if len(wrong) else "",
            "rmse_wrong_median": float(np.median(wrong)) if len(wrong) else "",
            "rmse_wrong_sd": float(wrong.std(ddof=1)) if len(wrong) > 1 else "",
            "rmse_wrong_q1": float(np.quantile(wrong, .25)) if len(wrong) else "",
            "rmse_wrong_q3": float(np.quantile(wrong, .75)) if len(wrong) else "",
            "delta_mean": float(deltas.mean()) if len(deltas) else "",
            "delta_median": float(np.median(deltas)) if len(deltas) else "",
            "delta_q1": float(np.quantile(deltas, .25)) if len(deltas) else "",
            "delta_q3": float(np.quantile(deltas, .75)) if len(deltas) else "",
            "delta_positive": int(sum(deltas > 0)), "delta_negative": int(sum(deltas < 0)),
            "warnings": sum(int(x["n_warnings"] or 0) for x in done),
        })
    write_csv(OUT / "summary_by_condition.csv", summary)
    seed_summary = []
    for condition in sorted({x["condition_id"] for x in pairs}):
        for series in SEED_SERIES:
            pp = [x for x in pairs if x["condition_id"] == condition and
                  x["seed_series"] == series and x["valid"]]
            delta = np.asarray([x["delta_rmse"] for x in pp], float)
            seed_summary.append({
                "condition_id": condition, "seed_series": series, "valid_pairs": len(pp),
                "mean_delta": float(delta.mean()) if len(delta) else "",
                "median_delta": float(np.median(delta)) if len(delta) else "",
                "positive": int(sum(delta > 0)), "negative": int(sum(delta < 0)),
            })
    write_csv(OUT / "summary_by_seed.csv", seed_summary)
    state = {
        "planned": len(manifest), "attempted": len(fits),
        "successful": sum(x["status"] == "OK" for x in fits),
        "failed": sum(x["status"] != "OK" for x in fits),
        "em_or_scoring_failed": sum(x["status"] in ("FAILED", "NONFINITE_METRIC") for x in fits),
        "mode_invalid": sum(x["status"] == "MODE_INVALID" for x in fits),
        "warning_count": sum(int(x["n_warnings"] or 0) for x in fits),
        "excluded_duplicate_completed_executions": (
            len(read_csv(OUT / "duplicate_incident.csv"))
            if (OUT / "duplicate_incident.csv").exists() else 0),
        "remaining": len(manifest) - len(fits),
        "valid_pairs": sum(x["valid"] for x in pairs),
        "completed": len(fits) == len(manifest),
    }
    (OUT / "summary.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    runinfo = json.loads((OUT / "runinfo.json").read_text(encoding="utf-8"))
    runinfo.update({
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "analysis_git_head": git("rev-parse", "HEAD"),
        "runner_sha256": sha256(Path(__file__)),
        "fit_path_sha256": hashlib.sha256("".join(
            inspect.getsource(func) for func in
            (column_order, assignment, masks, plan, run_one, run_cell)
        ).encode("utf-8")).hexdigest(),
        "status": "COMPLETE" if state["completed"] else "INCOMPLETE",
        **state,
    })
    (OUT / "runinfo.json").write_text(json.dumps(runinfo, indent=2), encoding="utf-8")
    print((OUT / "summary.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ("prepare", "pilot", "execute", "analyze", "repair"):
        group.add_argument("--" + action, action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.pilot:
        execute(pilot=True, workers=args.workers)
    elif args.execute:
        execute(limit=args.limit, workers=args.workers)
    else:
        repair_duplicate_incident() if args.repair else analyze()


if __name__ == "__main__":
    main()
