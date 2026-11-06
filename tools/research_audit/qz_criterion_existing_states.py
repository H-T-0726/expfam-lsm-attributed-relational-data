"""Q_Z versus the paper-type conditional criterion on EXISTING saved states.

No MCEM fit.  For every saved fitted state with all candidates K = 1..7
(canonical, lineage E) the script scores five criteria on the SAME state:

  A  C_Q        = -2 (Q_X + Q_Y + Q_Z) + p_K ln n          (Q2, MC mean)
  B  C_Q - P_Z  = -2 (Q_X + Q_Y)       + p_K ln n          (= C_cond_MC, DERIVED)
  C  C_cond_last= -2 l_data(Z_last)    + p_K ln n          (Q1 plug-in, last E-step sample)
  D  C_cond_mode= -2 l_data(Z_hat)     + p_K ln n          (Q1 plug-in, Laplace joint mode)
  E  C_Lap                                                 (Laplace approx. of Q3)

l_data(Z) = ln p(X, Y_train | Z, theta) with every constant (k_criterion_panel).
None of A-E is the paper's BIC; C/D are Eq.(16)-type at two explicit Z.
Selection rule (unchanged from the source experiments): mean over the two
starts per K, argmin, ties -> smallest K.

Lineage E prototype; NOT manuscript evidence.  C_Q is not Schwarz BIC; C_Lap
is not an exact marginal likelihood.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "expfam/src/experimental"))
sys.path.insert(0, str(ROOT / "tools/research_audit"))

import run_clap_clean_nsweep as a1  # noqa: E402
import run_k_selection_robustness as rob  # noqa: E402
import run_k_signal_strength as rss  # noqa: E402
import k_criterion_panel as panel  # noqa: E402
import em_runner  # noqa: E402

hist = a1.hist
ART = Path(r"D:\tento\kennkyu_artifacts")
KSEL = ROOT / "expfam/results/k_selection"
OUT = Path(os.environ.get("QZ_OUT", ROOT / "expfam/results/k_selection/qz_criterion_existing_states_20261106"))
C2 = 1.0 + math.log(2.0 * math.pi)
CRITS = ["A_CQ", "B_CQ_minus_PZ", "C_cond_last", "D_cond_mode", "E_CLap"]


def need(c, m):
    if not c:
        raise SystemExit("STOP: " + m)


def model_for(n, d, k, train, F, sigma, w0, w, var_z, Z):
    m = em_runner.build_model(n, d, k, hist.L_SAMPLES, hist.FAMILY_X, hist.FAMILY_Y,
                              train_mask=train, numerics_mode=hist.NUMERICS_MODE)
    m.initialize_params(true_params=None, seed=0)
    upd = {"Z": Z.copy(), "F": F, "w0": w0, "w": w, "var_z": var_z}
    if sigma is not None:
        upd["sigma"] = sigma
    m.params.update(upd)
    return m


def score_state(model, X, Y, Zs, Z_last, Z_hat, npar, n):
    """All criteria that need the state; returns per-fit dict."""
    cq = panel.cq_terms(model, X, Y, Zs, num_params=npar, n=n)
    t = panel.theta_of(model)
    L = Zs.shape[2]
    ll = np.array([panel.data_loglik(model, X, Y, Zs[:, :, l], t) for l in range(L)])
    ll_last = panel.data_loglik(model, X, Y, Z_last, t)
    ll_mode = panel.data_loglik(model, X, Y, Z_hat, t) if Z_hat is not None else float("nan")
    p = cq["P_theta"]
    k = Zs.shape[1]
    return {"num_params": npar, "P_theta": p, "Q_X": cq["Q_X"], "Q_Y": cq["Q_Y"], "Q_Z": cq["Q_Z"],
            "P_Z": cq["P_Z"], "P_Z_formula": n * k * C2 + n * k * math.log(t["var_z"]),
            "var_z": t["var_z"], "mean_z_sq": float(np.mean(Zs ** 2)),
            "A_CQ": cq["C_Q"], "B_CQ_minus_PZ": cq["C_Q"] - cq["P_Z"],
            "B_via_cond_MC": -2.0 * float(ll.mean()) + p,
            "C_cond_last": -2.0 * ll_last + p, "D_cond_mode": -2.0 * ll_mode + p,
            "mcse_B": 2.0 * float(ll.std(ddof=1)) / math.sqrt(L),
            "sd_single_sample_C": 2.0 * float(ll.std(ddof=1)),
            "last_is_sample_L": bool(np.array_equal(Z_last, Zs[:, :, -1]))}


# ---------------------------------------------------------------------------
# state sources (all K = 1..7, starts 1,2)
# ---------------------------------------------------------------------------

def src_nsweep(rows):
    """A1 clean n-sweep: K_GEN 1,3 (replayed states) and 5 (EMC checkpoint-8 npz)."""
    prov = a1.load_hist_provenance()
    emc, _ = a1.load_emc_identity()
    st = np.load(ART / "clap_clean_nsweep_20261006/full/states.npz")
    emc_npz = np.load(a1.EMC_NPZ)
    ref = pd.read_csv(KSEL / "clap_clean_nsweep_20261006/full/fit_criteria.csv")
    ref = ref.set_index(["k_true", "n", "replicate", "k_est", "start"])
    reps = {1: 4, 3: 4, 5: 8}
    for kt in (1, 3, 5):
        for n in (50, 75, 100, 150):
            for rep in range(1, reps[kt] + 1):
                ds, train, _ = a1.make_cell(kt, n, rep, prov[(kt, n, rep)])
                for k in hist.CANDIDATE_K:
                    for s in hist.STARTS:
                        tag = f"K{kt}_n{n}_r{rep}_k{k}_s{s}"
                        Zhat = np.asarray(st[f"{tag}_Zhat"], float)
                        if kt == 5:
                            fi = int(emc[(kt, n, rep, k, s)]["fit_index"])
                            key = f"{fi:04d}_it008"
                            Zs = np.asarray(emc_npz[f"{key}_Zsamples"], float)
                            Z = np.asarray(emc_npz[f"{key}_Z"], float)
                            F = np.asarray(emc_npz[f"{key}_F"], float)
                            sig = np.asarray(emc_npz[f"{key}_sigma"], float)
                            w0, w, vz = (float(v) for v in emc_npz[f"{key}_scalars"])
                            origin = f"emc_npz:{key}"
                        else:
                            Zs = np.asarray(st[f"{tag}_Zsamples"], float)
                            Z = Zs[:, :, -1]
                            F = np.asarray(st[f"{tag}_F"], float)
                            sig = None
                            w0, w, vz = (float(v) for v in st[f"{tag}_scalars"])
                            origin = f"nsweep_states:{tag}"
                        m = model_for(n, 15, k, train, F, sig, w0, w, vz, Z)
                        npar = k * 15 - k * (k - 1) // 2
                        r = score_state(m, ds.X, ds.Y, Zs, Z, Zhat, npar, n)
                        rr = ref.loc[(kt, n, rep, k, s)]
                        r.update(series="canonical A1 n-sweep", condition="A", k_gen=kt, n=n, d=15,
                                 replicate=rep, k_est=k, start=s, state_source=origin,
                                 E_CLap=float(rr.C_Lap), lap_status=rr.lap_status,
                                 stored_C_Q=float(rr.C_Q), stored_lap_fit_mode=float(rr.lap_fit_mode))
                        rows.append(r)
                print(f"nsweep K{kt} n{n} r{rep}", flush=True)


def src_signal(rows):
    """PR #155 B/C/D at n=75 and B at n=150 (A is the n-sweep)."""
    prov = a1.load_hist_provenance()
    w0C = rss.w0_c()
    ref = pd.read_csv(KSEL / "historical_canonical_causal_20261106/analysis/fit_results.csv")
    ref = ref.set_index(["condition", "n", "replicate", "k_est", "start"])
    for n, conds in ((75, "BCD"), (150, "B")):
        npz = np.load(ART / f"historical_canonical_causal_20261106/fit_states_n{n}.npz")
        for rep in range(1, 9):
            b = rss.build_all(n, rep, prov, w0C)
            for c in conds:
                X, Y = b["cond"][c]["X"], b["cond"][c]["Y"]
                for k in hist.CANDIDATE_K:
                    for s in hist.STARTS:
                        pre = f"n{n}_{c}_r{rep}_k{k}_s{s}"
                        Zs = np.asarray(npz[f"{pre}_Zsamples"], float)
                        Z = np.asarray(npz[f"{pre}_Z"], float)
                        F = np.asarray(npz[f"{pre}_F"], float)
                        sig = np.asarray(npz[f"{pre}_sigma"], float)
                        w0, w, vz = (float(v) for v in npz[f"{pre}_scalars"])
                        m = model_for(n, 15, k, b["train"], F, sig, w0, w, vz, Z)
                        npar = k * 15 - k * (k - 1) // 2
                        r = score_state(m, X, Y, Zs, Z, None, npar, n)
                        rr = ref.loc[(c, n, rep, k, s)]
                        r["D_cond_mode"] = float(rr.lap_fit_mode) + r["P_theta"]   # stored -2 l_data(Z_hat)
                        r.update(series="canonical signal-strength (PR #155)", condition=c, k_gen=5, n=n, d=15,
                                 replicate=rep, k_est=k, start=s, state_source=f"fit_states_n{n}:{pre}",
                                 E_CLap=float(rr.C_Lap), lap_status=rr.lap_status,
                                 stored_C_Q=float(rr.C_Q), stored_lap_fit_mode=float(rr.lap_fit_mode),
                                 D_source="stored lap_fit_mode (Z_hat not saved)")
                        rows.append(r)
            print(f"signal n{n} r{rep}", flush=True)


def src_dsweep(rows):
    """d=30/60 at n=75 (d-sweep) and d=30 at n=50/100/150 (R2)."""
    prov = a1.load_hist_provenance()
    ref = pd.read_csv(KSEL / "historical_canonical_causal_20261106/existing_reanalysis/fit_level_q_split.csv")
    ref = ref.set_index(["n", "d", "replicate", "k_est", "start"])
    dsw = np.load(ART / "dsweep_k_selection_20261105/fit_states.npz")
    r2 = {n: np.load(ART / "k_selection_robustness_20261105" / f"r2_n{n}_fit_states.npz") for n in (50, 100, 150)}
    for n in (50, 75, 100, 150):
        for rep in range(1, 9):
            ds, train, _, dd, _ = rob.build_data(n, rep, prov)
            for d in ((30, 60) if n == 75 else (30,)):
                X = dd[d]["X"]
                for k in hist.CANDIDATE_K:
                    for s in hist.STARTS:
                        if n == 75:
                            pre, src = f"d{d}_rep{rep}_k{k}_s{s}", dsw
                        else:
                            pre, src = f"n{n}_d{d}_L5_cp8_r{rep}_k{k}_s{s}", r2[n]
                        Zs = np.asarray(src[f"{pre}_Zsamples"], float)
                        Z = np.asarray(src[f"{pre}_Z"], float)
                        F = np.asarray(src[f"{pre}_F"], float)
                        sig = np.asarray(src[f"{pre}_sigma"], float)
                        w0, w, vz = (float(v) for v in src[f"{pre}_scalars"])
                        m = model_for(n, d, k, train, F, sig, w0, w, vz, Z)
                        npar = k * d - k * (k - 1) // 2
                        r = score_state(m, X, ds.Y, Zs, Z, None, npar, n)
                        rr = ref.loc[(n, d, rep, k, s)]
                        lfm = rr.lap_fit_mode
                        r["D_cond_mode"] = (float(lfm) + r["P_theta"]) if pd.notna(lfm) else float("nan")
                        r.update(series=f"canonical d-sweep / R2 (d={d})", condition="A", k_gen=5, n=n, d=d,
                                 replicate=rep, k_est=k, start=s, state_source=pre,
                                 E_CLap=float(rr.C_Lap) if pd.notna(rr.C_Lap) else float("nan"),
                                 lap_status=rr.lap_status, stored_C_Q=float(rr.C_Q),
                                 stored_lap_fit_mode=float(lfm) if pd.notna(lfm) else float("nan"),
                                 D_source="stored lap_fit_mode (Z_hat not saved)")
                        rows.append(r)
            print(f"dsweep n{n} r{rep}", flush=True)


def src_r1(rows):
    """Robustness R1: n=75, L=10, checkpoints 8 and 16, d = 15/30/60."""
    prov = a1.load_hist_provenance()
    ref = pd.read_csv(KSEL / "k_selection_robustness_20261105/r1/fit_criteria.csv")
    ref = ref.set_index(["d", "L", "checkpoint", "replicate", "k_est", "start"])
    npz = np.load(ART / "k_selection_robustness_20261105/r1_fit_states.npz")
    n = 75
    for rep in range(1, 9):
        ds, train, _, dd, _ = rob.build_data(n, rep, prov)
        for d in (15, 30, 60):
            X = dd[d]["X"] if d in dd else ds.X
            for cp in (8, 16):
                for k in hist.CANDIDATE_K:
                    for s in hist.STARTS:
                        pre = f"n75_d{d}_L10_cp{cp}_r{rep}_k{k}_s{s}"
                        Zs = np.asarray(npz[f"{pre}_Zsamples"], float)
                        Z = np.asarray(npz[f"{pre}_Z"], float)
                        F = np.asarray(npz[f"{pre}_F"], float)
                        sig = np.asarray(npz[f"{pre}_sigma"], float)
                        w0, w, vz = (float(v) for v in npz[f"{pre}_scalars"])
                        m = model_for(n, d, k, train, F, sig, w0, w, vz, Z)
                        npar = k * d - k * (k - 1) // 2
                        r = score_state(m, X, ds.Y, Zs, Z, None, npar, n)
                        rr = ref.loc[(d, 10, cp, rep, k, s)]
                        lfm = rr.lap_fit_mode
                        r["D_cond_mode"] = (float(lfm) + r["P_theta"]) if pd.notna(lfm) else float("nan")
                        r.update(series=f"canonical R1 (L=10, it={cp}, d={d})", condition="A", k_gen=5, n=n, d=d,
                                 replicate=rep, k_est=k, start=s, state_source=f"r1_fit_states:{pre}",
                                 E_CLap=float(rr.C_Lap) if pd.notna(rr.C_Lap) else float("nan"),
                                 lap_status=rr.lap_status, stored_C_Q=float(rr.C_Q),
                                 stored_lap_fit_mode=float(lfm) if pd.notna(lfm) else float("nan"),
                                 D_source="stored lap_fit_mode (Z_hat not saved)")
                        rows.append(r)
        print(f"r1 r{rep}", flush=True)


def src_paper_type(rows):
    """#139 paper-type (Gaussian-X / Bernoulli-Y, K 1..10): stored panel values (same panel code)."""
    f = pd.read_csv(KSEL / "paper_condition_k_criteria_20261006/full/fit_criteria.csv")
    for _, r in f.iterrows():
        rows.append(dict(series="paper-type #139 (Gaussian-X)", condition="paper-type", k_gen=int(r.k_true),
                         n=150, d=15, replicate=int(r.replicate), k_est=int(r.k_est), start=int(r.start),
                         state_source="paper_condition_k_criteria_20261006/full/fit_criteria.csv (stored)",
                         num_params=int(r.num_params), P_theta=r.P_theta, Q_X=r.Q_X, Q_Y=r.Q_Y, Q_Z=r.Q_Z,
                         P_Z=r.P_Z, P_Z_formula=150 * int(r.k_est) * C2 + 150 * int(r.k_est) * math.log(r.var_z),
                         var_z=r.var_z, A_CQ=r.C_Q, B_CQ_minus_PZ=r.C_Q - r.P_Z, B_via_cond_MC=r.C_cond_MC,
                         C_cond_last=r.C_cond_last, D_cond_mode=r.C_cond_mode, E_CLap=r.C_Lap,
                         lap_status=r.lap_status, stored_C_Q=r.C_Q, stored_lap_fit_mode=r.lap_fit_mode,
                         D_source="stored C_cond_mode"))


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def selection(scores: pd.DataFrame):
    keys = ["series", "condition", "k_gen", "n", "d", "replicate"]
    full, local = [], []
    for g_key, g in scores.groupby(keys):
        info = dict(zip(keys, g_key))
        kmax = int(g.k_est.max())
        for crit in CRITS:
            sub = g[["k_est", "start", crit] + (["lap_status"] if crit in ("E_CLap", "D_cond_mode") else [])]
            bad = sub[crit].isna().any() or (crit in ("E_CLap", "D_cond_mode") and (sub.lap_status != "OK").any())
            mean = sub.groupby("k_est")[crit].mean()
            complete = (not bad) and len(mean) == kmax and sorted(mean.index) == list(range(1, kmax + 1))
            if complete:
                kh = int(min(mean.index, key=lambda k: (mean[k], k)))
                lab = "exact" if kh == info["k_gen"] else ("under" if kh < info["k_gen"] else "over")
                cand = mean.sort_values()
                gap = float(cand.iloc[1] - cand.iloc[0])
            else:
                kh, lab, gap = np.nan, "INCOMPLETE", np.nan
            full.append({**info, "criterion": crit, "candidate_k": f"1..{kmax}", "k_hat": kh, "label": lab,
                         "best_minus_second": gap})
            kg = info["k_gen"]
            for a, b in sorted({(kg - 1, kg), (kg, kg + 1), (4, 5), (5, 6), (6, 7)}):
                if a < 1 or b > kmax or a not in mean.index or b not in mean.index:
                    continue
                ok = not (crit in ("E_CLap", "D_cond_mode") and
                          (g[g.k_est.isin([a, b])].lap_status != "OK").any())
                se = np.nan
                if crit in ("A_CQ", "B_CQ_minus_PZ") and "mcse_B" in g:
                    s_ = g[g.k_est.isin([a, b])].mcse_B
                    if s_.notna().all():
                        se = 0.5 * math.sqrt(float((s_ ** 2).sum()))
                local.append({**info, "criterion": crit, "k_from": a, "k_to": b,
                              "slack": float(mean[a] - mean[b]) if ok else np.nan,
                              "slack_per_node": float(mean[a] - mean[b]) / info["n"] if ok else np.nan,
                              "mc_se_approx": se, "status": "OK" if ok else "LAPLACE_NOT_OK"})
    return pd.DataFrame(full), pd.DataFrame(local)


def reselect():
    """Rebuild full / local selection from the stored criterion_scores.csv (no re-scoring)."""
    sc = pd.read_csv(OUT / "criterion_scores.csv")
    full, local = selection(sc)
    full.to_csv(OUT / "full_k_selection.csv", index=False, lineterminator="\n")
    local.to_csv(OUT / "local_selection_saved_states.csv", index=False, lineterminator="\n")


def main():
    if "--reselect" in sys.argv:
        need(not (OUT / "full_k_selection.csv").exists(), "remove the derived selection files first")
        reselect()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "criterion_scores.csv"
    need(not target.exists(), f"refusing to overwrite {target}")
    rows: list[dict] = []
    src_nsweep(rows)
    src_signal(rows)
    src_dsweep(rows)
    src_r1(rows)
    src_paper_type(rows)
    sc = pd.DataFrame(rows)
    # integrity checks (stored vs recomputed)
    canon = sc[sc.series != "paper-type #139 (Gaussian-X)"]
    checks = {
        "max_abs_CQ_recompute_minus_stored": float((canon.A_CQ - canon.stored_C_Q).abs().max()),
        "max_abs_B_identity_CQminusPZ_vs_condMC": float((sc.B_CQ_minus_PZ - sc.B_via_cond_MC).abs().max()),
        "max_abs_PZ_minus_nK(1+ln2pi)": float((sc.P_Z - sc.P_Z_formula).abs().max()),
        "max_abs_var_z_minus_1": float((sc.var_z - 1.0).abs().max()),
        "last_sample_identity_all": bool(canon.last_is_sample_L.all()),
    }
    ns = sc[sc.state_source.str.startswith(("emc_npz", "nsweep_states"))]
    ok = ns.lap_status == "OK"
    checks["max_abs_D_recompute_minus_stored_fitmode(nsweep)"] = float(
        ((ns.D_cond_mode - ns.P_theta) - ns.stored_lap_fit_mode)[ok].abs().max())
    for k_, v in checks.items():
        print(k_, v)
    need(checks["max_abs_CQ_recompute_minus_stored"] < 1e-6, "C_Q recompute differs")
    need(checks["max_abs_B_identity_CQminusPZ_vs_condMC"] < 1e-6, "B identity fails")
    sc.to_csv(target, index=False, lineterminator="\n")
    full, local = selection(sc)
    full.to_csv(OUT / "full_k_selection.csv", index=False, lineterminator="\n")
    local.to_csv(OUT / "local_selection_saved_states.csv", index=False, lineterminator="\n")
    (OUT / "integrity_checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
