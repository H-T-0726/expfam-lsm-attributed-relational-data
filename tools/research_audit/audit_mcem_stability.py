"""Independent audit of the MCEM-stability / signal-matched experiment.

Independence: Q_X / Q_Y / Q_Z and C_Q are recomputed with numpy / scipy from the stored
checkpoint states (no model class, no k_criterion_panel); slacks, paired medians, verdicts and
stability labels are recomputed from the CSVs without the analysis script's functions; the
replication gate is re-checked by comparing the stored iteration-8 states with the archived
EMC / PR #155 states directly.

Writes .../audit/audit_report.json and .../analysis/audit_summary.csv
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
BASE = ROOT / "expfam" / "results" / "k_selection" / "mcem_stability_signal_matched_20261106"
AN, FIG, OUT = BASE / "analysis", BASE / "figures", BASE / "audit"
BASE_BRANCH = "origin/research/k-selection-causal-analysis-20261106"
checks: list[dict] = []


def check(name, ok, detail=""):
    checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arr_sha(a):
    c = np.ascontiguousarray(a)
    return hashlib.sha256(c.tobytes() + str(c.shape).encode()).hexdigest()


def q_parts(X, Y, train, Zs, F, w0, w, vz):
    n, k, L = Zs.shape
    iu = np.triu_indices(n, 1)
    obs = train[iu].astype(bool)
    y = Y[iu][obs]
    qx = qy = qz = 0.0
    for l in range(L):
        Z = Zs[:, :, l]
        e = Z @ F.T
        qx += float(np.sum(X * e - np.exp(e) - gammaln(X + 1)))
        ey = (w0 + w * (Z @ Z.T))[iu][obs]
        qy += float(np.sum(y * ey - np.logaddexp(0.0, ey)))
        qz += -0.5 * n * k * math.log(2 * math.pi * vz) - 0.5 * float(np.sum(Z ** 2)) / vz
    return qx / L, qy / L, qz / L


def slack(df, crit, keys, k_lo, k_hi):
    g = df[df.criterion == crit].groupby(keys + ["k_fit"]).value.mean().unstack("k_fit")
    return (g[k_lo] - g[k_hi]).rename("S")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import run_mcem_stability as ms                    # generator, prereg and archived-state paths only
    pre = json.loads((BASE / "gates" / "prereg.json").read_text(encoding="utf-8"))
    gates = json.loads((BASE / "gates" / "gates.json").read_text(encoding="utf-8"))
    check("prereg file hash == code PREREG hash", pre["sha256"] == ms.prereg_hash())
    check("prereg body == code PREREG", json.dumps({k: v for k, v in pre.items() if k != "sha256"}, sort_keys=True)
          == json.dumps(ms.PREREG, sort_keys=True))
    check("gates PASSED", gates["status"] == "PASSED")
    runs = {}
    for st in ("rq1_AB", "rq1_CD", "rq2"):
        p = BASE / st / "runinfo.json"
        if p.exists():
            runs[st] = json.loads(p.read_text(encoding="utf-8"))
            check(f"{st}: prereg hash == gates", runs[st]["prereg_sha256"] == gates["prereg_sha256"] == pre["sha256"])
            check(f"{st}: states npz sha256", sha256(runs[st]["states"]) == runs[st]["states_sha256"])
            check(f"{st}: status COMPLETED", runs[st]["status"] == "COMPLETED", runs[st]["status"])
    pc = subprocess.run(["git", "log", "--format=%H %s", "--", "expfam/results/k_selection/mcem_stability_signal_matched_20261106/gates/prereg.json"],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip().splitlines()
    first_result = subprocess.run(["git", "log", "--format=%H", "--", *[f"expfam/results/k_selection/mcem_stability_signal_matched_20261106/{s}" for s in ("rq1_AB", "rq1_CD", "rq2")]],
                                  cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.split()
    if pc and first_result:
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", pc[-1].split()[0], first_result[-1]], cwd=ROOT).returncode == 0
        check("prereg committed before production results", anc and pc[-1].split()[0] != first_result[-1], pc[-1][:60])

    led = pd.read_csv(AN / "fit_execution_ledger.csv")
    prod = led[led.role == "production"]
    exp = {"RQ1": 128, "RQ2": 96}
    for rq, nfit in exp.items():
        stage_ok = (rq == "RQ1" and {"rq1_AB", "rq1_CD"} <= set(runs)) or (rq == "RQ2" and "rq2" in runs)
        if stage_ok:
            check(f"{rq}: ledger fits == plan ({nfit})", int((prod.rq == rq).sum()) == nfit, int((prod.rq == rq).sum()))
    fit_iters = int(prod[prod.status.isin(["FIT", "FAILED"])].fit_iterations.sum()) + 80
    check("fit-iterations within 12,000 cap", fit_iters <= 12000, fit_iters)
    sc = pd.read_csv(AN / "checkpoint_criterion_scores.csv")
    q = pd.read_csv(AN / "checkpoint_q_decomposition.csv")
    key = ["rq", "condition", "replicate", "k_fit", "start", "checkpoint"]
    check("no duplicate checkpoint keys", not q.duplicated(key).any())
    n1 = int((q.rq == "RQ1").sum()); n2 = int((q.rq == "RQ2").sum())
    check("checkpoint rows == fits x checkpoints (FIT fits)",
          n1 == 6 * int(((prod.rq == "RQ1") & (prod.status == "FIT")).sum()) and n2 == 4 * int(((prod.rq == "RQ2") & (prod.status == "FIT")).sum()),
          f"RQ1 {n1}, RQ2 {n2}")
    fails = pd.read_csv(AN / "failure_and_incomplete.csv")
    check("failed fits are listed, not dropped", int((prod.status == "FAILED").sum()) == int((fails.kind == "FIT_FAILED").sum()))
    check("non-OK Laplace rows listed", int((q.lap_status != "OK").sum()) == int(fails.kind.astype(str).str.startswith("LAPLACE_").sum()))

    # data regeneration
    prov = ms.a1.load_hist_provenance()
    w0C = ms.ss.w0_c()
    pv = pd.read_csv(AN / "dataset_provenance.csv")
    data1 = ms.rq1_data(prov, w0C)
    bad = 0
    for rep, b in data1.items():
        for c in "ABCD":
            r = pv[(pv.rq == "RQ1") & (pv.condition == c) & (pv.replicate == rep)].iloc[0]
            bad += int(arr_sha(b["cond"][c]["X"]) != r.hash_X or arr_sha(b["cond"][c]["Y"]) != r.hash_Y)
    check("RQ1 data == PR #155 data (hashes)", bad == 0, bad)
    data2 = {rep: ms.rq2_data(rep) for rep in range(1, 9)}
    bad = 0
    for rep, b in data2.items():
        for kg in (3, 5):
            r = pv[(pv.rq == "RQ2") & (pv.condition == f"K{kg}") & (pv.replicate == rep)].iloc[0]
            bad += int(arr_sha(b[kg]["X"]) != r.hash_X or arr_sha(b[kg]["Y"]) != r.hash_Y)
            ev = np.linalg.svd(b[kg]["F"], compute_uv=False) ** 2
            bad += int(not np.allclose(ev, 2.5))
        bad += int(not np.array_equal(b[3]["Z"], b[5]["Z"][:, :3]))
    check("RQ2 data hashes, eigenvalues 2.5, shared Z directions", bad == 0, bad)

    # replication gate re-checked independently
    emc = np.load(ms.a1.EMC_NPZ); prev = np.load(ms.PREV_STATES)
    emc_id, _ = ms.a1.load_emc_identity()
    states = {st: np.load(runs[st]["states"]) for st in runs}
    mism, nrep = 0, 0
    for st in ("rq1_AB", "rq1_CD"):
        if st not in states:
            continue
        S = states[st]
        for c in st[4:]:
            for rep in range(1, 9):
                for k in (4, 5):
                    for s in (1, 2):
                        mine = {nm: S[f"RQ1_{c}_r{rep}_k{k}_s{s}_it8_{nm}"] for nm in ("Zsamples", "F", "sigma", "scalars")}
                        if c == "A":
                            fi = int(emc_id[(5, 75, rep, k, s)]["fit_index"]); ref = {nm: emc[f"{fi:04d}_it008_{nm}"] for nm in mine}
                        else:
                            ref = {nm: prev[f"n75_{c}_r{rep}_k{k}_s{s}_{nm}"] for nm in mine}
                        nrep += 1
                        mism += int(not all(np.allclose(mine[nm], ref[nm], rtol=1e-9, atol=1e-6) for nm in mine))
    flagged = int(((q.rq == "RQ1") & (q.checkpoint == 8)).sum()) - int(((q.rq == "RQ1") & (q.checkpoint == 8)
                                                                        & pd.read_csv(AN / "checkpoint_metrics.csv").replication_status.eq("MATCH").values).sum())
    check("iteration-8 states == archived states (independent re-check)", mism == flagged, f"{nrep} fits, mismatches {mism}, flagged {flagged}")

    # independent Q recomputation at every checkpoint state
    worst = {"Q_X": 0.0, "Q_Y": 0.0, "Q_Z": 0.0, "C_Q": 0.0}
    cnt = 0
    for r in q.itertuples():
        st = ("rq1_AB" if r.condition in "AB" else "rq1_CD") if r.rq == "RQ1" else "rq2"
        if st not in states:
            continue
        S = states[st]; pre_ = f"{r.rq}_{r.condition}_r{r.replicate}_k{r.k_fit}_s{r.start}_it{r.checkpoint}"
        if r.rq == "RQ1":
            b = data1[r.replicate]; X, Y, train = b["cond"][r.condition]["X"], b["cond"][r.condition]["Y"], b["train"]
        else:
            b = data2[r.replicate]; kg = int(r.condition[1:]); X, Y, train = b[kg]["X"], b[kg]["Y"], b["train"]
        w0, w, vz = (float(x) for x in S[f"{pre_}_scalars"])
        qx, qy, qz = q_parts(X, Y, train, S[f"{pre_}_Zsamples"], S[f"{pre_}_F"], w0, w, vz)
        k = r.k_fit
        cq = -2 * (qx + qy + qz) + (k * 15 - k * (k - 1) // 2) * math.log(75)
        cq_csv = sc[(sc.rq == r.rq) & (sc.condition == r.condition) & (sc.replicate == r.replicate) & (sc.k_fit == k)
                    & (sc.start == r.start) & (sc.checkpoint == r.checkpoint) & (sc.criterion == "C_Q")].value.iloc[0]
        for nm, a, bb in (("Q_X", qx, r.Q_X), ("Q_Y", qy, r.Q_Y), ("Q_Z", qz, r.Q_Z), ("C_Q", cq, cq_csv)):
            worst[nm] = max(worst[nm], abs(a - bb) / (1 + abs(bb)))
        cnt += 1
    check(f"independent Q_X/Q_Y/Q_Z/C_Q at every checkpoint state ({cnt})", max(worst.values()) < 1e-8,
          {k: f"{v:.1e}" for k, v in worst.items()})
    okl = q[q.lap_status == "OK"]
    ide = okl.residual - (okl.termA + 2 * okl.R3)
    check("C_Q - C_Lap - 2H_Lap = termA + 2R3 (CSV)", float(np.max(np.abs(ide))) < 1e-6, f"{np.max(np.abs(ide)):.1e}")
    check("min eigenvalue of H > 0 for every OK Laplace", bool((okl.lap_min_eig_H > 0).all()))
    check("Laplace OK rows satisfy grad tolerance 1e-8", bool((okl.lap_grad_inf <= 1e-8).all()))

    # slacks, paired medians, verdicts recomputed
    loc = pd.read_csv(AN / "local_k4_k5_comparison.csv")
    r1 = sc[sc.rq == "RQ1"]
    Sq = slack(r1, "C_Q", ["condition", "replicate", "checkpoint"], 4, 5)
    m_ = loc.set_index(["condition", "replicate", "checkpoint"]).S_Q
    check("S_Q(4->5) recomputed from scores (start means)", float(np.max(np.abs(Sq.reindex(m_.index) - m_))) < 1e-8)
    ps = pd.read_csv(AN / "paired_slack_changes.csv")
    summ = ps[ps.row_type == "summary"]
    bad = 0
    for cp in (8, 16, 24, 32, 48, 64):
        for c in "BCD":
            d_ = np.array([Sq.get((c, rep, cp), np.nan) - Sq.get(("A", rep, cp), np.nan) for rep in range(1, 9)])
            d_ = d_[~np.isnan(d_)]
            v = "MAINTAINED" if np.median(d_) > 0 and (d_ > 0).sum() >= 6 else ("NOT_MAINTAINED" if np.median(d_) <= 0 else "INCONCLUSIVE")
            row = summ[(summ.criterion == "C_Q") & (summ.checkpoint == cp) & (summ.contrast == f"{c}-A")].iloc[0]
            bad += int(not math.isclose(row["median"], float(np.median(d_)), abs_tol=1e-8) or row.verdict_rule != v
                       or int(row.n_positive) != int((d_ > 0).sum()))
    check("C_Q paired medians / sign counts / verdicts recomputed", bad == 0, bad)
    met = pd.read_csv(AN / "checkpoint_metrics.csv")
    lab = ((met.max_gram_rel_change_last8 <= 0.01) & (met.max_eta_x_rel_change_last8 <= 0.01) & (met.max_eta_y_rel_change_last8 <= 0.01))
    check("stability flags recomputed with the pre-registered 0.01 rule", bool((lab == met.stable_at_checkpoint).all()))
    stb = pd.read_csv(AN / "stability_diagnostics.csv")
    fl = stb[stb.level == "fit"]
    m64 = met[(met.rq == "RQ1") & (met.checkpoint == 64)].set_index(["condition", "replicate", "k_fit", "start"])
    lab64 = ((m64.max_gram_rel_change_last8 <= 0.01) & (m64.max_eta_x_rel_change_last8 <= 0.01) & (m64.max_eta_y_rel_change_last8 <= 0.01))
    got = fl.set_index(["condition", "replicate", "k_fit", "start"]).label_at_64.eq("STABLE")
    check("fit labels at 64 == metrics", bool((got.reindex(lab64.index) == lab64).all()), f"STABLE {int(lab64.sum())}/{len(lab64)}")
    if (AN / "signal_matched_k3_k5.csv").exists() and "rq2" in runs:
        r2 = sc[sc.rq == "RQ2"]
        sm = pd.read_csv(AN / "signal_matched_k3_k5.csv")
        su = sm[(sm.row_type == "summary") & (sm.criterion == "C_Q") & (sm.checkpoint == 32)].iloc[0]
        s3 = slack(r2[r2.condition == "K3"], "C_Q", ["replicate", "checkpoint"], 2, 3)
        s5 = slack(r2[r2.condition == "K5"], "C_Q", ["replicate", "checkpoint"], 4, 5)
        d_ = np.array([s5[(rep, 32)] - s3[(rep, 32)] for rep in range(1, 9)])
        v = "K_HARDER" if np.median(d_) < 0 and (d_ < 0).sum() >= 6 else ("NOT_HARDER" if np.median(d_) >= 0 else "INCONCLUSIVE")
        check("RQ2 primary delta / verdict recomputed", math.isclose(su.median_delta, float(np.median(d_)), abs_tol=1e-8) and su.verdict_rule == v,
              f"median {np.median(d_):.2f} {v}")
    man = json.loads((AN / "analysis_manifest.json").read_text(encoding="utf-8"))
    check("analysis CSV hashes == manifest", all(sha256(AN / k) == v for k, v in man["outputs_sha256"].items()))
    check("figure hashes == manifest", all(sha256(FIG / k) == v for k, v in man["figures_sha256"].items()))
    out = subprocess.run(["git", "diff", "--name-status", f"{BASE_BRANCH}...HEAD"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    non_add = [l_ for l_ in out.splitlines() if l_ and not l_.startswith("A\t")]
    check("PR #155 artifacts untouched (only additions + append-only registry)",
          all(l_.endswith("EXPERIMENT_REGISTRY.md") for l_ in non_add), non_add[:4])
    if non_add:
        d = subprocess.run(["git", "diff", f"{BASE_BRANCH}...HEAD", "--", "EXPERIMENT_REGISTRY.md"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8").stdout
        check("EXPERIMENT_REGISTRY.md append-only", not [l_ for l_ in d.splitlines() if l_.startswith("-") and not l_.startswith("---")])
    rep = {"script": "tools/research_audit/audit_mcem_stability.py", "n_checks": len(checks),
           "n_pass": sum(c["result"] == "PASS" for c in checks), "n_fail": sum(c["result"] == "FAIL" for c in checks),
           "checks": checks}
    (OUT / "audit_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    pd.DataFrame(checks).assign(experiment_id="mcem_stability_signal_matched_20261106").to_csv(AN / "audit_summary.csv", index=False)
    print(f"{rep['n_pass']}/{rep['n_checks']} PASS")
    return 0 if rep["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
