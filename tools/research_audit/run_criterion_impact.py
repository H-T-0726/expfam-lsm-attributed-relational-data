"""Runner for the criterion impact validation (criterion_impact_protocol.py).

Commands
  preflight       no fit: builder equivalence on reps 1-4, seed collision check, protocol hash
  pilot           MC budget pilot on 6 fixed existing states (MCSE and time only; no selection)
  score-existing  Phase C: score the 448 PR #160 E2 states with criteria A-F (+ decomposition terms)
  fit-confirm     Phase E: 448 new MCEM fits on reps 11-14 (resume-safe, no retry)
  score-confirm   Phase E: score the new states with the same scorer

Every output is new; nothing existing is overwritten.  BLAS threads pinned to 1.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import criterion_impact_protocol as cp  # noqa: E402
import fair_common_evaluator as ev  # noqa: E402
import run_fair_comparison as rf  # noqa: E402
import evaluate_fair_comparison as ef  # noqa: E402
import run_k_signal_strength as rss  # noqa: E402
import laplace_k_criterion as lk  # noqa: E402
import elbo_k_criterion as ek  # noqa: E402
import em_runner  # noqa: E402

a1 = rf.a1
hist = rf.hist
RES = ROOT / cp.RESULTS_REL
STATES = Path(cp.STATE_ROOT)
BUDGET_FILE = RES / "elbo_mc_budget.json"
XF = {"gaussian": 1, "poisson": 2}
SG = {"weak": 1, "strong": 2}


def need(c, m):
    if not c:
        raise SystemExit("STOP: " + m)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- data (new replicates)
_W0C = None


def prov_row_for(n, rep):
    """Provenance row computed from the SAME generator call that a1.make_cell makes."""
    k = cp.K_GEN
    scale = hist.f_scale_for_row_norm(hist.ROW_NORM_SQ_TARGET, d=hist.D_FEATURES, k=k)
    w_true = hist.w_for_matched_y_signal(hist.W_REF, k=k, k_ref=hist.K_REF)
    ds = hist.generate_canonical_data(n=n, d=hist.D_FEATURES, k=k, seed=hist.data_seed(k, n, rep),
                                      family_x=hist.FAMILY_X, family_y=hist.FAMILY_Y, f_scale=scale,
                                      w0=hist.W0_TRUE, w=w_true, poisson_lambda_max=hist.POISSON_LAMBDA_MAX)
    train, test = hist.build_masks(n, hist.split_seed(k, n, rep))
    up = np.triu(np.ones((n, n), dtype=bool), k=1)
    return {"train_mask_hash": hist.stable_hash(train), "test_mask_hash": hist.stable_hash(test),
            "y_density": repr(float(ds.Y[up].mean())), "x_mean": repr(float(ds.X.mean()))}


def bundle(n, rep):
    global _W0C
    if _W0C is None:
        _W0C = rss.w0_c()
    return rss.build_all(n, rep, {(cp.K_GEN, n, rep): prov_row_for(n, rep)}, _W0C)


def data_new(n, rep, signal, xfam):
    """Mirror of run_fair_comparison.e2_data with the bundle built for any replicate."""
    b = bundle(n, rep)
    c = "A" if signal == "weak" else "D"
    cond = b["cond"][c]
    if xfam == "poisson":
        X, s2 = cond["X"], None
    else:
        X, s2 = rf.gaussian_x(b["ds"].Z, cond["F"], b["dseed"])
    return dict(X=X, Y=cond["Y"], Z=b["ds"].Z, F=cond["F"], w0=cond["w0"], w=cond["w"], train=b["train"],
                test=b["test"], data_seed=b["dseed"], x_family=xfam, gaussian_noise_var=s2,
                note=f"PR #155 condition {c} (new replicate)")


# ---------------------------------------------------------------- state lists
def existing_states():
    out = []
    for f in ef.all_fits():
        if f["kind"] != "E2":
            continue
        out.append(dict(f, phase=1, phase_label="EXPLORATORY_existing"))
    need(len(out) == 448, f"existing E2 states {len(out)} != 448")
    return out


def confirm_fits():
    out = []
    for xfam in ("gaussian", "poisson"):
        for n in (75, 150):
            for sig in ("weak", "strong"):
                for rep in cp.PHASE_E["replicates"]:
                    for k in cp.K_CAND:
                        for s in cp.STARTS:
                            out.append(dict(stage="confirm", key=f"cf_{xfam}_{sig}_n{n}_r{rep}_k{k}_s{s}", kind="E2",
                                            n=n, rep=rep, k=k, s=s, signal=sig, xfam=xfam, estimator="E",
                                            seed=hist.model_seed(5, n, rep, k, s), phase=2,
                                            phase_label="CONFIRMATORY_new"))
    need(len(out) == cp.PHASE_E["fit_cap"], "confirmation fit count")
    return out


def load_state(f):
    if f["phase"] == 1:
        st = ef.load_state(f)
        return st
    z = np.load(STATES / "confirm" / f"{f['key']}.npz")
    return {k: z[k] for k in z.files}


def data_for(f):
    if f["phase"] == 1:
        return rf.data_for(f) if f["stage"] != "reuse" else rf.e2_data(f["n"], f["rep"], f["signal"], "poisson")
    return data_new(f["n"], f["rep"], f["signal"], f["xfam"])


# ---------------------------------------------------------------- scorer
def build_model(X, train, st, th, xfam, k):
    n, d = X.shape
    m = em_runner.build_model(n, d, k, 5, xfam, "bernoulli", train_mask=train, numerics_mode="consistent")
    m.initialize_params(true_params=None, seed=0)
    m.params.update({"Z": np.asarray(st["Z"], float).copy(), "F": th["F"], "sigma": np.diag(th["sigma_diag"]),
                     "w0": th["w0"], "w": th["w"], "var_z": th["var_z"]})
    return m


def score_state(f, D, st, M):
    xfam = D["x_family"]
    th = ef.theta_of(st, xfam)
    X, Y, train = D["X"], D["Y"], D["train"]
    n, d = X.shape
    k = f["k"]
    Zs = np.asarray(st["Zsamples"], float)
    Zl = np.asarray(st["Z"], float)
    c = ev.criteria_abc(X, Y, Zs, Zl, th, xfam, train)
    r = {kk: f.get(kk) for kk in ("phase", "phase_label", "stage", "key", "xfam", "signal", "n", "rep", "k", "s", "seed")}
    r.update(x_sha256=sha_arr(X), y_sha256=sha_arr(Y), train_sha256=sha_arr(train), data_seed=D["data_seed"])
    r.update({kk: c[kk] for kk in ("P_theta", "Q_X", "Q_Y", "Q_Z", "P_Z", "A_CQ", "B_CQ_minus_PZ", "C_cond_last",
                                   "mcse_B", "sd_single_sample_C")})
    r.update(w0_hat=th["w0"], w_hat=th["w"], var_z=th["var_z"])
    m = build_model(X, train, st, th, xfam, k)
    t0 = time.perf_counter()
    lap = lk.laplace_log_observed(m, X, Y, Zl.copy(), grad_tol=1e-8, max_iter=200)
    r.update(lap_status=lap.status, lap_grad_inf=lap.grad_inf, lap_min_eig_H=lap.min_eigenvalue_H,
             lap_iterations=lap.iterations, lap_seconds=round(time.perf_counter() - t0, 3))
    nan = float("nan")
    if not lap.ok:
        r.update(E_CLap=nan, D_cond_mode=nan, F_CELBO=nan, F_CELBO_mcse=nan, elbo_status=lap.status)
        return r
    pen = c["P_theta"]
    c_lap = lk.calc_C_Lap(lap, d_K=ev.num_params(k, d, xfam), N=float(n))["C_Lap"]
    need(abs(lk.calc_C_Lap(lap, d_K=ev.num_params(k, d, xfam), N=float(n))["parameter_term"] - pen) < 1e-9, "penalty mismatch")
    ll_mode = ev.data_loglik(X, Y, lap.Z_hat, th, xfam, train)
    seed = [20261108, f["phase"], XF[xfam], SG[f["signal"]], n, f["rep"], k, f["s"]]
    el = ek.elbo_from_mode(m, X, Y, lap, n_samples=M, n_batches=cp.ELBO["n_batches"], seed_entropy=seed)
    nk = n * k
    zsq = float(np.sum(lap.Z_hat ** 2))
    r.update(E_CLap=c_lap, D_cond_mode=-2 * ll_mode + pen, ll_cond_mode=ll_mode, phi_hat=lap.phi_at_mode,
             prior_hat=-0.5 * nk * math.log(2 * math.pi) - 0.5 * zsq, zhat_sq=zsq, logdet_H=lap.logdet_H,
             lap_volume=zsq + lap.logdet_H, elbo_status=el.status, elbo_seed=json.dumps(seed), elbo_M=el.n_samples,
             elbo_seconds=round(el.seconds, 3))
    if not el.ok:
        r.update(F_CELBO=nan, F_CELBO_mcse=nan)
        return r
    q_mc = c["Q_X"] + c["Q_Y"] + c["Q_Z"]
    term_a = -2 * (q_mc - el.e_q_phi)
    c_elbo = -2 * el.elbo + pen
    r.update(H_q=el.entropy_q, two_H_q=2 * el.entropy_q, E_q_phi=el.e_q_phi, E_q_phi_mcse=el.e_q_phi_mcse,
             R3=el.R3, Q_MC=q_mc, termA=term_a, F_CELBO=c_elbo, F_CELBO_mcse=2 * el.e_q_phi_mcse,
             elbo_batch_sd=float(np.std(el.batch_means, ddof=1)) * 2,
             id1_resid=(c["A_CQ"] - c_lap) - (2 * el.entropy_q + term_a + 2 * el.R3),
             id2_resid=(c_elbo - c_lap) - 2 * el.R3,
             elbo_minus_lap=el.elbo - lap.log_observed)
    return r


def run_scoring(states, out_name, M, cross=None):
    out = RES / out_name
    need(not out.exists(), f"refusing to overwrite {out}")
    rows, cache = [], {}
    t0 = time.time()
    for i, f in enumerate(states):
        if not str(f.get("status", "OK")).startswith("OK"):
            rows.append({kk: f.get(kk) for kk in ("phase", "phase_label", "stage", "key", "xfam", "signal", "n", "rep",
                                                   "k", "s", "seed")} | {"fit_status": f.get("status")})
            continue
        dk = (f["phase"], f["xfam"], f["signal"], f["n"], f["rep"])
        if dk not in cache:
            cache.clear()
            cache[dk] = data_for(f)
        r = score_state(f, cache[dk], load_state(f), M)
        r["fit_status"] = f.get("status", "OK")
        rows.append(r)
        if i % 20 == 0:
            print(f"{i}/{len(states)} {f['key']} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if cross is not None:
        cross(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    print("rows", len(df), "->", out)


def cross_check_existing(df):
    old = pd.read_csv(ROOT / "expfam/results/k_selection/fair_model_estimator_criterion_comparison_20261106/criterion_scores.csv")
    old = old.set_index("key")
    rows = []
    for _, r in df.iterrows():
        o = old.loc[r["key"]]
        for a, b in (("A_CQ", "A_CQ"), ("B_CQ_minus_PZ", "B_CQ_minus_PZ"), ("C_cond_last", "C_cond_last"),
                     ("D_cond_mode", "D_cond_mode"), ("E_CLap", "E_CLap")):
            x, y = r[a], o[b]
            ok = (np.isnan(x) and np.isnan(y)) or abs(x - y) <= 1e-6
            rows.append({"key": r["key"], "quantity": a, "recomputed": x, "stored_pr160": y,
                         "abs_diff": abs(x - y) if not (np.isnan(x) or np.isnan(y)) else np.nan, "match": bool(ok)})
        need(str(o["lap_status"]) == str(r["lap_status"]), f"lap status differs {r['key']}")
    cc = pd.DataFrame(rows)
    cc.to_csv(RES / "crosscheck_vs_pr160.csv", index=False, lineterminator="\n")
    need(cc["match"].all(), f"cross-check mismatches: {int((~cc['match']).sum())}")
    print("cross-check vs PR #160: all", len(cc), "match; max diff", cc["abs_diff"].max())


# ---------------------------------------------------------------- preflight / pilot / fits
def prior_seeds():
    seeds = set()
    for p in glob.glob(str(ROOT / "expfam/results/**/*.csv"), recursive=True):
        try:
            with open(p, newline="", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                cols = [c for c in (rd.fieldnames or []) if c and "seed" in c.lower()]
                if not cols:
                    continue
                for row in rd:
                    for c in cols:
                        v = (row.get(c) or "").strip()
                        if v.lstrip("-").isdigit():
                            seeds.add(int(v))
        except (UnicodeDecodeError, csv.Error):
            continue
    return seeds


def preflight():
    out = RES / "preflight.json"
    need(not out.exists(), "preflight already recorded")
    rec = {"protocol_hash": cp.protocol_hash(), "git_head": git_head(), "builder_equivalence": [], "seed_check": {}}
    for xfam in ("gaussian", "poisson"):
        for n in (75, 150):
            for sig in ("weak", "strong"):
                for rep in (1, 2, 3, 4):
                    a = rf.e2_data(n, rep, sig, xfam)
                    b = data_new(n, rep, sig, xfam)
                    eq = all(np.array_equal(a[q], b[q]) for q in ("X", "Y", "Z", "F", "train", "test"))
                    rec["builder_equivalence"].append({"xfam": xfam, "n": n, "signal": sig, "rep": rep, "identical": eq})
                    need(eq, f"builder differs from e2_data at {(xfam, n, sig, rep)}")
    seeds = prior_seeds()
    new = {"data_seed": [hist.data_seed(5, n, r) for n in (75, 150) for r in cp.PHASE_E["replicates"]],
           "split_seed": [hist.split_seed(5, n, r) for n in (75, 150) for r in cp.PHASE_E["replicates"]],
           "model_seed": [f["seed"] for f in confirm_fits()]}
    used_reps = {(n, r) for n in (50, 75, 100, 150) for r in range(1, 9)}
    for kname, vals in new.items():
        coll = sorted(set(vals) & seeds)
        rec["seed_check"][kname] = {"n_new": len(set(vals)), "collisions_with_prior_csv_seeds": coll}
        need(not coll, f"{kname} collides with prior seeds: {coll[:5]}")
    # PR #160 E2 rule: model_seed depends on (n, rep, k, start) only and is SHARED across x-family and signal
    # (common random numbers), so 2 n x 4 reps x 7 K x 2 starts = 112 distinct values are expected.
    need(len(set(new["model_seed"])) == 112, "model seeds: expected 112 distinct (n, rep, k, start) values")
    rec["seed_check"]["model_seed_sharing"] = "shared across x-family and signal by the E2 rule; 112 distinct"
    rec["seed_check"]["n_prior_seed_values_scanned"] = len(seeds)
    rec["seed_check"]["replicates_used_before_for_this_generator"] = "1..8 (A1 n-sweep, PR #155-#162)"
    rec["seed_check"]["new_replicates_disjoint"] = all((n, r) not in used_reps for n in (75, 150) for r in cp.PHASE_E["replicates"])
    # new data generate without error (rate gate etc.)
    gen = []
    for xfam in ("gaussian", "poisson"):
        for n in (75, 150):
            for sig in ("weak", "strong"):
                for rep in cp.PHASE_E["replicates"]:
                    D = data_new(n, rep, sig, xfam)
                    up = np.triu(D["train"], 1)
                    gen.append({"xfam": xfam, "n": n, "signal": sig, "rep": rep, "data_seed": D["data_seed"],
                                "x_sha256": sha_arr(D["X"]), "y_sha256": sha_arr(D["Y"]), "train_sha256": sha_arr(D["train"]),
                                "y_train_density": float(D["Y"][up].mean()), "train_pairs": int(up.sum()),
                                "x_finite": bool(np.all(np.isfinite(D["X"])))})
    rec["new_data"] = gen
    RES.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print("preflight OK", cp.protocol_hash()[:12])


def pilot():
    need(not BUDGET_FILE.exists(), "budget already fixed")
    names = {("gaussian", "weak", 150, 1, 5, 1), ("gaussian", "weak", 150, 1, 7, 1), ("poisson", "weak", 150, 1, 5, 1),
             ("poisson", "weak", 150, 1, 7, 1), ("gaussian", "weak", 75, 1, 5, 1), ("poisson", "weak", 75, 1, 5, 1)}
    sts = [f for f in existing_states() if (f["xfam"], f["signal"], f["n"], f["rep"], f["k"], f["s"]) in names]
    need(len(sts) == 6, "pilot states")
    res = []
    for M in cp.PILOT["M_grid"]:
        for f in sts:
            D = data_for(f)
            st = load_state(f)
            th = ef.theta_of(st, D["x_family"])
            m = build_model(D["X"], D["train"], st, th, D["x_family"], f["k"])
            t0 = time.perf_counter()
            lap = lk.laplace_log_observed(m, D["X"], D["Y"], np.asarray(st["Z"], float).copy())
            el = ek.elbo_from_mode(m, D["X"], D["Y"], lap, n_samples=M, n_batches=cp.ELBO["n_batches"],
                                   seed_entropy=[20261108, 9, XF[f["xfam"]], SG[f["signal"]], f["n"], f["rep"], f["k"], f["s"]])
            res.append({"M": M, "key": f["key"], "status": el.status, "C_ELBO_mcse": 2 * el.e_q_phi_mcse,
                        "seconds": time.perf_counter() - t0})
    df = pd.DataFrame(res)
    summ = df.groupby("M").agg(median_mcse=("C_ELBO_mcse", "median"), max_seconds=("seconds", "max")).reset_index()
    ok = summ[(summ.median_mcse <= 1.0) & (summ.max_seconds <= 60)]
    M = int(ok.M.min()) if len(ok) else 2048
    rec = {"rule": cp.ELBO["budget_rule"], "pilot": df.to_dict("records"), "summary": summ.to_dict("records"),
           "chosen_M": M, "note": "pilot seeds use phase code 9 (not reused in scoring); no selection computed",
           "protocol_hash": cp.protocol_hash(), "git_head": git_head()}
    RES.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(summ.to_string(), "\nchosen M =", M)


def chosen_M():
    return int(json.loads(BUDGET_FILE.read_text(encoding="utf-8"))["chosen_M"])


def fit_confirm():
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools", "expfam/src", "reproduction/src"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    need(not dirty, "code tree dirty: commit before fitting")
    need((RES / "preflight.json").exists(), "preflight missing")
    head = git_head()
    log = RES / "confirm_run_log.txt"
    fits = confirm_fits()
    cache = {}
    with log.open("a", encoding="utf-8") as lg:
        lg.write(f"{time.strftime('%H:%M:%S')} start fits={len(fits)} head={head} protocol={cp.protocol_hash()[:12]}\n")
        for f in fits:
            j = RES / "confirm_fits" / f"{f['key']}.json"
            if j.exists():
                row = json.loads(j.read_text(encoding="utf-8"))
                npz = STATES / "confirm" / f"{f['key']}.npz"
                if row["status"] == "FAILED" or (npz.exists() and rf.sha_file(npz) == row["npz_sha256"]):
                    continue
            while rf.avail_gb() < 3.0:
                time.sleep(30)
            dk = (f["xfam"], f["signal"], f["n"], f["rep"])
            if dk not in cache:
                cache.clear()
                cache[dk] = data_new(f["n"], f["rep"], f["signal"], f["xfam"])
            D = cache[dk]
            t0 = time.perf_counter()
            row = dict(f, protocol_hash=cp.protocol_hash(), git_head=head, data_seed=D["data_seed"],
                       x_sha256=sha_arr(D["X"]), y_sha256=sha_arr(D["Y"]), train_sha256=sha_arr(D["train"]))
            try:
                st = rf.fit_E(D["X"], D["Y"], D["train"], f["k"], f["seed"], f["xfam"])
                row["npz_sha256"] = rf.save_npz_atomic(STATES / "confirm" / f"{f['key']}.npz", st)
                row["status"], row["exception"] = "OK", ""
            except (FloatingPointError, SystemExit, Exception) as e:  # recorded, never retried
                row["status"], row["exception"], row["npz_sha256"] = "FAILED", f"{type(e).__name__}: {e}"[:300], ""
            row["runtime_s"] = round(time.perf_counter() - t0, 2)
            rf.atomic_write_bytes(j, json.dumps(row, indent=1, default=str).encode("utf-8"))
            lg.write(f"{time.strftime('%H:%M:%S')} {row['status']} {f['key']} {row['runtime_s']}s\n")
            lg.flush()
        lg.write(f"{time.strftime('%H:%M:%S')} end\n")


def confirm_states():
    out = []
    for f in confirm_fits():
        j = RES / "confirm_fits" / f"{f['key']}.json"
        st = json.loads(j.read_text(encoding="utf-8"))["status"] if j.exists() else "NOT_RUN"
        out.append(dict(f, status=st))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "pilot", "score-existing", "fit-confirm", "score-confirm"])
    a = ap.parse_args()
    if a.cmd == "preflight":
        preflight()
    elif a.cmd == "pilot":
        pilot()
    elif a.cmd == "score-existing":
        run_scoring(existing_states(), "score_by_state_existing.csv", chosen_M(), cross=cross_check_existing)
    elif a.cmd == "fit-confirm":
        fit_confirm()
    else:
        run_scoring(confirm_states(), "score_by_state_confirm.csv", chosen_M())
