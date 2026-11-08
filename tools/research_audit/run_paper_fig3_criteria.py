"""Runner for paper_fig3_protocol.py.

Commands
  preflight            no fit: generate the 5 datasets, seed-collision scan, score the TRUE-parameter state of each
                       dataset with every criterion (technical check of generator / model / evaluators)
  fit   --kgen K       the 100 fits of one K_GEN (resume-safe, no retry); several K_GEN may run in parallel processes
  score --kgen K       score the fits of one K_GEN with the six criteria + decomposition terms
Nothing existing is overwritten.  BLAS threads pinned to 1.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import paper_fig3_protocol as pp  # noqa: E402
import run_criterion_impact as rci  # noqa: E402
from em_runner_instrumented import run_em_instrumented  # noqa: E402

rf = rci.rf
RES = ROOT / pp.RESULTS_REL
STATES = Path(pp.STATE_ROOT)
rci.SG["paper"] = 3          # seed-entropy code for this study (see pp.ELBO_SEED)


def need(c, m):
    if not c:
        raise SystemExit("STOP: " + m)


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def generate(kgen: int) -> dict:
    n, d = pp.N, pp.D
    seed = pp.data_seed(kgen)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, kgen))
    F = rng.normal(0.0, np.sqrt(10.0), size=(d, kgen))
    w0 = float(rng.standard_normal())
    w = float(3.0 * rng.standard_normal())
    X = Z @ F.T + np.sqrt(0.1) * rng.standard_normal((n, d))
    U = rng.random((n, n))
    P = 1.0 / (1.0 + np.exp(-(w0 + w * (Z @ Z.T))))
    Yu = np.triu((U < P).astype(float), 1)
    Y = Yu + Yu.T
    train = np.ones((n, n), dtype=bool)
    np.fill_diagonal(train, False)
    return dict(X=X, Y=Y, Z=Z, F=F, w0=w0, w=w, train=train, test=None, data_seed=seed, x_family="gaussian",
                sigma_true=np.full(d, 0.1))


def fit_list(kgen):
    return [dict(stage="paper", key=f"pf_K{kgen}_k{k}_t{t}", phase=3, xfam="gaussian", signal="paper", n=pp.N,
                 rep=kgen, kgen=kgen, k=k, s=t, trial=t, seed=pp.model_seed(kgen, k, t))
            for k in pp.K_CAND for t in pp.TRIALS]


def preflight():
    out = RES / "preflight.json"
    need(not out.exists(), "preflight exists")
    rec = {"protocol_hash": pp.protocol_hash(), "git_head": git_head(), "datasets": [], "true_state_scores": []}
    seeds = rci.prior_seeds()
    new_seeds = [pp.data_seed(k) for k in pp.K_GEN] + [f["seed"] for k in pp.K_GEN for f in fit_list(k)]
    need(len(set(new_seeds)) == len(new_seeds) == 505, "seeds not unique")
    coll = sorted(set(new_seeds) & seeds)
    rec["seed_check"] = {"n_new": len(new_seeds), "prior_values_scanned": len(seeds), "collisions": coll}
    need(not coll, f"seed collision {coll[:5]}")
    for kgen in pp.K_GEN:
        D = generate(kgen)
        up = np.triu(np.ones((pp.N, pp.N), bool), 1)
        eta = D["w0"] + D["w"] * (D["Z"] @ D["Z"].T)[up]
        p = 1 / (1 + np.exp(-eta))
        rec["datasets"].append({"K_GEN": kgen, "data_seed": D["data_seed"], "w0_true": D["w0"], "w_true": D["w"],
                                "y_density": float(D["Y"][up].mean()), "frac_p_outside_0.01_0.99": float(np.mean((p < .01) | (p > .99))),
                                "x_var_mean": float(D["X"].var(0).mean()), "FtF_eig_min": float(np.linalg.eigvalsh(D["F"].T @ D["F"]).min()),
                                "x_finite": bool(np.all(np.isfinite(D["X"]))), "x_sha256": rci.sha_arr(D["X"]), "y_sha256": rci.sha_arr(D["Y"])})
        st = {"Zsamples": np.repeat(D["Z"][:, :, None], pp.L, axis=2), "Z": D["Z"].copy(), "F": D["F"],
              "sigma": D["sigma_true"], "scalars": np.array([D["w0"], D["w"], 1.0])}
        f = dict(fit_list(kgen)[0], k=kgen, s=0, key=f"truth_K{kgen}")
        r = rci.score_state(f, D, st, 256)
        rec["true_state_scores"].append({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                                         for k, v in r.items() if k in ("key", "A_CQ", "B_CQ_minus_PZ", "C_cond_last", "D_cond_mode",
                                                                         "E_CLap", "F_CELBO", "lap_status", "elbo_status", "id1_resid", "id2_resid")})
    RES.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rec["datasets"], indent=1))
    print([(s["key"], s["lap_status"], s.get("id1_resid")) for s in rec["true_state_scores"]])


def fit(kgen):
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "expfam/src", "reproduction/src"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    need(not dirty, "code tree dirty")
    need((RES / "preflight.json").exists(), "preflight missing")
    head = git_head()
    D = generate(kgen)
    log = RES / f"run_log_K{kgen}.txt"
    with log.open("a", encoding="utf-8") as lg:
        lg.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start K_GEN={kgen} head={head} protocol={pp.protocol_hash()[:12]}\n")
        for f in fit_list(kgen):
            j = RES / "fits" / f"{f['key']}.json"
            if j.exists():
                row = json.loads(j.read_text(encoding="utf-8"))
                npz = STATES / f"{f['key']}.npz"
                if row["status"] == "FAILED" or (npz.exists() and rf.sha_file(npz) == row["npz_sha256"]):
                    continue
            while rf.avail_gb() < 1.5:
                time.sleep(30)
            holder, traj = {}, []

            def cb(rec):
                traj.append({"iteration": int(rec["iteration"]), "q_strict": float(rec["q_strict"]),
                             "F_fro": float(np.linalg.norm(rec["F"])), "w": float(rec["w"])})
                if int(rec["iteration"]) == pp.EM_ITERATIONS:
                    holder["rec"] = rec
            t0 = time.perf_counter()
            row = dict(f, protocol_hash=pp.protocol_hash(), git_head=head, data_seed=D["data_seed"],
                       x_sha256=rci.sha_arr(D["X"]), y_sha256=rci.sha_arr(D["Y"]))
            try:
                res = run_em_instrumented(D["X"], D["Y"], family_x="gaussian", family_y="bernoulli", k=f["k"], L=pp.L,
                                          max_iter=pp.EM_ITERATIONS, checkpoints=tuple(range(1, pp.EM_ITERATIONS + 1)),
                                          seed=f["seed"], train_mask=D["train"], numerics_mode="consistent",
                                          stability_window=4, on_nan="fail", on_checkpoint=cb)
                need(res["retry_count"] == 0 and not res["nan_detected"], "retry / NaN")
                rec = holder["rec"]
                st = {"Zsamples": np.array(rec["Z_samples"]), "Z": np.array(rec["Z_est"]), "F": np.array(rec["F"]),
                      "sigma": np.asarray(rec["sigma"], float), "scalars": np.array([rec["w0"], rec["w"], rec["var_z"]], float)}
                row["npz_sha256"] = rf.save_npz_atomic(STATES / f"{f['key']}.npz", st)
                row["status"], row["exception"] = "OK", ""
            except (FloatingPointError, SystemExit, Exception) as e:      # recorded, never retried
                row["status"], row["exception"], row["npz_sha256"] = "FAILED", f"{type(e).__name__}: {e}"[:300], ""
            row["runtime_s"] = round(time.perf_counter() - t0, 2)
            row["q_strict_trajectory"] = traj
            rf.atomic_write_bytes(j, json.dumps(row, indent=1, default=str).encode("utf-8"))
            lg.write(f"{time.strftime('%H:%M:%S')} {row['status']} {f['key']} {row['runtime_s']}s\n")
            lg.flush()
        lg.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} end K_GEN={kgen}\n")


def score(kgen):
    out = RES / f"score_by_state_K{kgen}.csv"
    need(not out.exists(), f"refusing to overwrite {out}")
    D = generate(kgen)
    rows = []
    for f in fit_list(kgen):
        j = RES / "fits" / f"{f['key']}.json"
        st_json = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {"status": "NOT_RUN"}
        base = {k: f[k] for k in ("key", "kgen", "k", "trial", "seed")}
        if st_json["status"] != "OK":
            rows.append(dict(base, fit_status=st_json["status"]))
            continue
        z = np.load(STATES / f"{f['key']}.npz")
        st = {k: z[k] for k in z.files}
        r = rci.score_state(f, D, st, pp.ELBO_M)
        r.update(base, fit_status="OK", git_head_scoring=git_head())
        rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print("rows", len(rows), "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "fit", "score"])
    ap.add_argument("--kgen", type=int)
    a = ap.parse_args()
    if a.cmd == "preflight":
        preflight()
    elif a.cmd == "fit":
        need(a.kgen in pp.K_GEN, "kgen")
        fit(a.kgen)
    else:
        need(a.kgen in pp.K_GEN, "kgen")
        score(a.kgen)
