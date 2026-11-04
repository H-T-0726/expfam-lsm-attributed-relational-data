"""Read-only diagnostic: how much does the choice of the Z point estimate matter?

No EM, no refit, no sampling.  Every number comes from states that were saved
by already-completed, audited runs:

* A1 (Issue #128, clean n-sweep, Poisson-X / Bernoulli-Y)
  - K_TRUE=1,3: ``Zsamples`` / ``scalars`` / ``Zhat`` in the A1 state archive
  - K_TRUE=5:   ``Zsamples`` / ``scalars`` from the EM convergence-sensitivity
                checkpoint archive at iteration 8 (``npz_key`` in A1's
                ``fit_criteria.csv``), ``Zhat`` from the A1 state archive
* B  (Issue #128, paper-type generator, Gaussian-X / Bernoulli-Y)
* EM convergence-sensitivity CSVs (commit dad67b1, 448 fits, 8..40 iterations)

Three point estimates of Z are compared on the SAME fitted theta:

* ``last``  -- ``Z_samples[:, :, -1]``, the estimate every runner reports today
* ``mean5`` -- the mean of the L=5 samples of the final E-step
* ``mode``  -- ``Zhat``, the joint mode of ln p(X, Y_train | Z, theta) + ln p(Z)
              already computed for C_Lap with the fit's training mask

The true Z is regenerated from the frozen data seeds; the regenerated X / Y
must match the recorded provenance hashes or the script stops.

This is a post-hoc EXPLORATORY diagnostic.  It does not change any selection
rule, metric or historical artifact; it writes a new directory and refuses to
overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "research_audit"))
sys.path.insert(0, str(ROOT / "expfam" / "src"))
sys.path.insert(0, str(ROOT / "expfam" / "src" / "experimental"))

import run_clap_clean_nsweep as A1                                 # noqa: E402
import run_paper_condition_k_criteria as B                          # noqa: E402
from utils_expfam import calc_rmse, procrustes_rotation             # noqa: E402

A1_STATES = A1.STATE_ROOT / "full" / "states.npz"
B_STATES = B.STATE_ROOT / "full" / "states.npz"
A1_FIT = ROOT / "expfam/results/k_selection/clap_clean_nsweep_20261006/full/fit_criteria.csv"
A1_PROV = ROOT / "expfam/results/k_selection/clean_true_k_asymptotics_20260904/generator_provenance.csv"
B_PROV = ROOT / "expfam/results/k_selection/paper_condition_k_criteria_20261006/full/dataset_provenance.csv"
ESTIMATORS = ("last", "mean5", "mode")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rmse_z(z_est: np.ndarray, z_true: np.ndarray) -> float:
    rot, k = procrustes_rotation(z_est, z_true)
    return float(calc_rmse(z_true[:, :k], z_est[:, :k] @ rot))


def estimates(z_samples: np.ndarray, z_hat: np.ndarray) -> dict[str, np.ndarray]:
    return {"last": z_samples[:, :, -1], "mean5": z_samples.mean(axis=2),
            "mode": z_hat}


def a1_rows(emc_npz: Path):
    states = np.load(A1_STATES)
    emc = np.load(emc_npz)
    fit = pd.read_csv(A1_FIT)
    prov = pd.read_csv(A1_PROV, dtype=str)
    s1_rows, z_rows = [], []
    for _, pr in prov.iterrows():
        kt, n, rep = int(pr.K_TRUE), int(pr.n), int(pr.replicate)
        ds, _train, test = A1.make_cell(kt, n, rep, pr.to_dict())
        cell = fit[(fit.k_true == kt) & (fit.n == n) & (fit.replicate == rep)]
        for _, f in cell.iterrows():
            tag = f"K{kt}_n{n}_r{rep}_k{int(f.k_est)}_s{int(f.start)}"
            if f.state_source == "npz_readonly":
                zs, sc = emc[f"{f.npz_key}_Zsamples"], emc[f"{f.npz_key}_scalars"]
            else:
                zs, sc = states[f"{tag}_Zsamples"], states[f"{tag}_scalars"]
            w0, w = float(sc[0]), float(sc[1])
            est = estimates(zs, states[f"{tag}_Zhat"])
            row = {"k_true": kt, "n": n, "replicate": rep, "k_est": int(f.k_est),
                   "start": int(f.start), "lap_status": f.lap_status,
                   "S1_recorded": float(f.S1)}
            for name, z in est.items():
                row[f"S1_{name}"] = A1.heldout_score(z, w0, w, ds.Y, test)
            s1_rows.append(row)
            if int(f.k_est) == kt:
                z_rows.append({"experiment": "A1", "k_true": kt, "n": n,
                               "replicate": rep, "start": int(f.start),
                               "lap_status": f.lap_status,
                               **{f"rmse_{k}": rmse_z(z, ds.Z)
                                  for k, z in est.items()}})
    s1 = pd.DataFrame(s1_rows)
    gap = float((s1.S1_last - s1.S1_recorded).abs().max())
    if gap > 1e-12:
        raise SystemExit(f"S1 from saved states does not reproduce the record ({gap})")
    return s1, pd.DataFrame(z_rows), gap


def b_rows():
    states = np.load(B_STATES)
    prov = pd.read_csv(B_PROV)
    rows = []
    for kt in B.K_TRUE_GRID:
        for rep in B.REPLICATES:
            p = B.paper_gen.set_true_params(n=B.N, d=B.D, k=kt,
                                            seed=B.data_seed(kt, rep),
                                            **B.GENERATOR_KWARGS)
            x = np.asarray(p["X"], float)
            xs = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            if xs not in set(prov.x_sha256):
                raise SystemExit(f"B X hash mismatch at K_TRUE={kt} rep={rep}")
            z_true = np.asarray(p["Z"], float)
            for start in (1, 2):
                tag = f"K{kt}_r{rep}_k{kt}_s{start}"
                est = estimates(states[f"{tag}_Zsamples"], states[f"{tag}_Zhat"])
                rows.append({"experiment": "B", "k_true": kt, "n": B.N,
                             "replicate": rep, "start": start,
                             **{f"rmse_{k}": rmse_z(z, z_true)
                                for k, z in est.items()}})
    return pd.DataFrame(rows)


def s1_selection(s1: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name in ESTIMATORS:
        col = f"S1_{name}"
        mean = s1.groupby(["k_true", "n", "replicate", "k_est"])[col].mean().reset_index()
        sel = mean.loc[mean.groupby(["k_true", "n", "replicate"])[col].idxmax()]
        per_start = s1.loc[s1.groupby(["k_true", "n", "replicate", "start"])[col].idxmax()]
        ps = per_start.pivot_table(index=["k_true", "n", "replicate"],
                                   columns="start", values="k_est")
        wide = s1.pivot_table(index=["k_true", "n", "replicate", "k_est"],
                              columns="start", values=col)
        absdiff = (wide[1] - wide[2]).abs().groupby(level=["k_true", "n"]).median()
        for (kt, n), g in sel.groupby(["k_true", "n"]):
            dis = ps.loc[(kt, n)]
            out.append({"estimator": name, "k_true": kt, "n": n,
                        "datasets": len(g), "exact": int((g.k_est == kt).sum()),
                        "under": int((g.k_est < kt).sum()),
                        "over": int((g.k_est > kt).sum()),
                        "mean_selected_k": float(g.k_est.mean()),
                        "per_start_disagreement": int((dis[1] != dis[2]).sum()),
                        "median_abs_start_diff_S1": float(absdiff.loc[(kt, n)])})
    return pd.DataFrame(out)


def rmse_summary(z: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (exp, kt, n), g in z.groupby(["experiment", "k_true", "n"]):
        wide = {e: g.pivot_table(index="replicate", columns="start",
                                 values=f"rmse_{e}") for e in ESTIMATORS}
        out.append({"experiment": exp, "k_true": kt, "n": n, "fits": len(g),
                    **{f"median_rmse_{e}": float(g[f"rmse_{e}"].median())
                       for e in ESTIMATORS},
                    **{f"median_abs_start_diff_{e}":
                       float((wide[e][1] - wide[e][2]).abs().median())
                       for e in ESTIMATORS}})
    return pd.DataFrame(out)


def emc_noise(iter_csv: Path, cp_csv: Path):
    it = pd.read_csv(iter_csv)
    keys = ["gram_rel_change", "f_raw_change", "w_change", "w0_change",
            "eta_x_rel_change", "eta_y_rel_change"]
    floor = (it[it.iteration.isin([1, 2, 4, 8, 12, 16, 24, 32, 40])]
             .groupby("iteration")[keys].median().reset_index())
    cp = pd.read_csv(cp_csv)
    within = (cp.groupby(["n", "replicate", "k_est", "start"])
              [["s2_q_based", "heldout_mean_log_score"]].std().reset_index()
              .groupby("n")[["s2_q_based", "heldout_mean_log_score"]].median()
              .add_prefix("median_within_fit_sd_cp8_to_cp40_"))
    c8 = (cp[cp.checkpoint_iteration == 8]
          .groupby(["n", "replicate", "k_est"])
          [["s2_q_based", "heldout_mean_log_score"]].mean().reset_index())
    gaps = {}
    for col in ("s2_q_based", "heldout_mean_log_score"):
        wide = c8.pivot_table(index=["n", "replicate"], columns="k_est", values=col)
        gaps[f"median_abs_gap_K4_K5_{col}"] = (wide[4] - wide[5]).abs().groupby("n").median()
    noise = within.join(pd.DataFrame(gaps)).reset_index()
    return floor, noise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emc-npz", type=Path, required=True)
    ap.add_argument("--emc-iteration-csv", type=Path, required=True)
    ap.add_argument("--emc-checkpoint-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    s1, z_a1, gap = a1_rows(args.emc_npz)
    z_b = b_rows()
    z = pd.concat([z_a1, z_b], ignore_index=True)
    floor, noise = emc_noise(args.emc_iteration_csv, args.emc_checkpoint_csv)

    args.out.mkdir(parents=True)
    s1.to_csv(args.out / "a1_s1_by_fit_and_estimator.csv", index=False)
    s1_selection(s1).to_csv(args.out / "a1_s1_selection_by_estimator.csv", index=False)
    z.to_csv(args.out / "rmse_z_by_fit_and_estimator.csv", index=False)
    rmse_summary(z).to_csv(args.out / "rmse_z_summary.csv", index=False)
    floor.to_csv(args.out / "emc_per_iteration_change_median.csv", index=False)
    noise.to_csv(args.out / "emc_noise_vs_k_gap.csv", index=False)
    inputs = {label: sha256(path) for label, path in (
        ("A1_states", A1_STATES), ("B_states", B_STATES),
        ("A1_fit_criteria", A1_FIT), ("A1_generator_provenance", A1_PROV),
        ("B_dataset_provenance", B_PROV), ("emc_checkpoint_states", args.emc_npz),
        ("emc_iteration_diagnostics", args.emc_iteration_csv),
        ("emc_checkpoint_results", args.emc_checkpoint_csv))}
    (args.out / "inputs.json").write_text(json.dumps({
        "mode": "EXPLORATORY post-hoc diagnostic; no EM, no refit, no sampling",
        "emc_source_commit": "dad67b1c7b90f1e0297e8b1e95afd394cf9b51db",
        "input_paths": {"A1_states": str(A1_STATES), "B_states": str(B_STATES),
                        "emc_npz": str(args.emc_npz)},
        "sha256": inputs,
        "s1_reproduction_max_abs_diff": gap,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
