"""Independent audit of the signal-strength experiment and its analysis.

Independence: Q_X / Q_Y / Q_Z and C_Q are recomputed here with numpy / scipy from
the stored states and regenerated data, WITHOUT the model classes, k_criterion_panel
or eval_utils.  Selection, frequencies and paired medians are recomputed from
fit_results.csv without the analysis script's functions.  Data regeneration uses
the runner's generator only to compare hashes against the stored provenance.

Writes  expfam/results/k_selection/historical_canonical_causal_20261106/audit/audit_report.json
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
BASE = ROOT / "expfam" / "results" / "k_selection" / "historical_canonical_causal_20261106"
SS, AN, FIG, EX = BASE / "signal_strength", BASE / "analysis", BASE / "figures", BASE / "existing_reanalysis"
ART = Path(r"D:\tento\kennkyu_artifacts") / "historical_canonical_causal_20261106"
OUT = BASE / "audit"
C2 = 1.0 + math.log(2.0 * math.pi)

checks: list[dict] = []


def check(name, ok, detail=""):
    checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arr_sha(a) -> str:
    c = np.ascontiguousarray(a)
    return hashlib.sha256(c.tobytes() + str(c.shape).encode()).hexdigest()


def indep_q(X, Y, train, Zs, F, w0, w, var_z):
    n, k, L = Zs.shape
    iu = np.triu_indices(n, 1)
    obs = train[iu].astype(bool)
    y = Y[iu][obs]
    qx = qy = qz = 0.0
    for l in range(L):
        Z = Zs[:, :, l]
        ex = Z @ F.T
        qx += float(np.sum(X * ex - np.exp(ex) - gammaln(X + 1)))
        ey = (w0 + w * (Z @ Z.T))[iu][obs]
        qy += float(np.sum(y * ey - np.logaddexp(0.0, ey)))
        qz += float(-0.5 * n * k * math.log(2 * math.pi * var_z) - 0.5 * np.sum(Z ** 2) / var_z)
    return qx / L, qy / L, qz / L


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import run_k_signal_strength as ss                 # generator + prereg only
    import run_clap_clean_nsweep as a1
    fr = pd.read_csv(AN / "fit_results.csv")
    sel = pd.read_csv(AN / "selection_by_replicate.csv")
    freq = pd.read_csv(AN / "selection_frequency.csv")
    summ = pd.read_csv(AN / "paired_comparison_summary.csv")
    prov = pd.read_csv(AN / "dataset_provenance.csv")
    man = json.loads((AN / "analysis_manifest.json").read_text(encoding="utf-8"))
    pre = json.loads((SS / "gates" / "prereg.json").read_text(encoding="utf-8"))
    gates = json.loads((SS / "gates" / "gates.json").read_text(encoding="utf-8"))

    # 1. prereg / code / run consistency
    check("prereg file hash == code PREREG hash", pre["sha256"] == ss.prereg_hash(), pre["sha256"][:12])
    body = {k: v for k, v in pre.items() if k != "sha256"}
    check("prereg file body == code PREREG", json.dumps(body, sort_keys=True) == json.dumps(ss.PREREG, sort_keys=True))
    runs = {}
    for n in (75, 150):
        p = SS / f"full_n{n}" / "runinfo.json"
        if p.exists():
            runs[n] = json.loads(p.read_text(encoding="utf-8"))
            check(f"n={n} runinfo prereg == gates prereg", runs[n]["prereg_sha256"] == gates["prereg_sha256"] == pre["sha256"])
            check(f"n={n} run COMPLETED", runs[n]["status"] == "COMPLETED")
            npz = Path(runs[n]["fit_states"])
            check(f"n={n} state npz sha256 matches runinfo", npz.exists() and sha256(npz) == runs[n]["fit_states_sha256"])
    check("gates PASSED", gates["status"] == "PASSED")

    # 2. counts, keys, ledger
    key = ["condition", "n", "replicate", "k_est", "start"]
    check("no duplicate fit keys", not fr.duplicated(key).any())
    exp = {(75, "A"): 112, (75, "B"): 112, (75, "C"): 112, (75, "D"): 112, (150, "A"): 112, (150, "B"): 112}
    cnt = fr.groupby(["n", "condition"]).size().to_dict()
    check("fit rows == preregistered plan", all(cnt.get(k, 0) == v for k, v in exp.items() if k[0] in runs) and
          set(cnt) <= set(exp), cnt)
    new = fr[fr.source == "new_fit"]
    check("new fits within cap 480 (incl. 5 gate fits)", len(new) + 5 <= 480, len(new) + 5)
    led = pd.read_csv(AN / "execution_ledger.csv")
    prod = led[led.role == "production"]
    check("ledger production rows == new fits", len(prod) == len(new) and set(prod.status) <= {"FIT", "FAILED"})
    check("no retries recorded", int(prod.retry.fillna(0).sum()) == 0)
    check("failed fits recorded, not dropped", int((fr.fit_status == "FAILED").sum()) == int((prod.status == "FAILED").sum()),
          int((fr.fit_status == "FAILED").sum()))

    # 3. data regeneration vs stored hashes
    prov_map = a1.load_hist_provenance()
    w0C = ss.w0_c()
    bad = 0
    bundles = {}
    for n in runs:
        for rep in range(1, 9):
            b = ss.build_all(n, rep, prov_map, w0C)
            bundles[(n, rep)] = b
            for c in "ABCD":
                r = prov[(prov.n == n) & (prov.replicate == rep) & (prov.condition == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                v = b["cond"][c]
                bad += int(arr_sha(v["X"]) != r.hash_X or arr_sha(v["Y"]) != r.hash_Y or arr_sha(b["ds"].Z) != r.hash_Z
                           or arr_sha(b["train"]) != r.hash_train)
    check("regenerated X/Y/Z/mask hashes == provenance", bad == 0, f"mismatches={bad}")
    sh = prov.groupby(["n", "replicate"]).apply(lambda g: (g.hash_Z.nunique() == 1) and (g.hash_train.nunique() == 1)
                                               and (g.set_index("condition").hash_Y.get("B") == g.set_index("condition").hash_Y.get("A"))
                                               and (g.set_index("condition").hash_X.get("C", g.set_index("condition").hash_X["A"]) == g.set_index("condition").hash_X["A"]))
    check("sharing rule (Z, mask shared; Y_B=Y_A; X_C=X_A)", bool(sh.all()))
    check("no seed rescue / dataset replacement flags", not prov.seed_rescue.any() and not prov.dataset_replaced.any())

    # 4. independent C_Q recomputation for every new fit and every reused A state
    emc = np.load(a1.EMC_NPZ)
    states = {n: np.load(runs[n]["fit_states"]) for n in runs}
    maxrel = {"Q_X": 0.0, "Q_Y": 0.0, "Q_Z": 0.0, "C_Q": 0.0}
    nchk = 0
    for r in fr[fr.fit_status != "FAILED"].itertuples():
        b = bundles[(r.n, r.replicate)]
        v = b["cond"][r.condition]
        if r.source == "new_fit":
            pre_ = f"n{r.n}_{r.condition}_r{r.replicate}_k{r.k_est}_s{r.start}"
            src = states[r.n]
        else:
            pre_ = r.source.split(":", 1)[1]
            src = emc
        Zs = src[f"{pre_}_Zsamples"]; F = src[f"{pre_}_F"]; w0, w, vz = (float(x) for x in src[f"{pre_}_scalars"])
        qx, qy, qz = indep_q(v["X"], v["Y"], b["train"], Zs, F, w0, w, vz)
        k = r.k_est
        cq = -2 * (qx + qy + qz) + (k * 15 - k * (k - 1) // 2) * math.log(r.n)
        for nm, a, bb in (("Q_X", qx, r.Q_X), ("Q_Y", qy, r.Q_Y), ("Q_Z", qz, r.Q_Z), ("C_Q", cq, r.C_Q)):
            maxrel[nm] = max(maxrel[nm], abs(a - bb) / (1 + abs(bb)))
        nchk += 1
    check(f"independent Q_X/Q_Y/Q_Z/C_Q recomputation ({nchk} states)", max(maxrel.values()) < 1e-8,
          {k: f"{v:.2e}" for k, v in maxrel.items()})
    check("P_Z == nK(1+ln 2pi) for every state (var_z=1)",
          float(np.nanmax(np.abs(fr.P_Z - fr.n * fr.k_est * C2))) < 1e-8 and bool((fr.var_z_hat.dropna() == 1.0).all()))
    ok = fr[fr.lap_status == "OK"]
    idres = (ok.C_Q - ok.C_Lap - (ok.n * ok.k_est * C2 - ok.lap_logdet_H)) - (ok.termA + 2 * ok.R3)
    check("identity C_Q - C_Lap - 2H_Lap = termA + 2R3 (recomputed)", float(np.max(np.abs(idres))) < 1e-6, f"{np.max(np.abs(idres)):.2e}")
    check("C_Lap = fit_mode + volume + penalty", float(np.max(np.abs(ok.C_Lap - ok.lap_fit_mode - ok.lap_volume - ok.lap_penalty))) < 1e-6)

    # 5. selection, frequency, paired medians recomputed
    mis = 0; recomputed = []
    for (c, n, rep), g in fr.groupby(["condition", "n", "replicate"]):
        failed = (g.fit_status == "FAILED").any()
        for crit in ("C_Q", "C_Lap"):
            if failed or (crit == "C_Lap" and (g.lap_status != "OK").any()):
                st, kk = "INCOMPLETE", None
            else:
                m = g.groupby("k_est")[crit].mean(); kk = int(m.idxmin()); st = "COMPLETE"
            s = sel[(sel.condition == c) & (sel.n == n) & (sel.replicate == rep) & (sel.criterion == crit)].iloc[0]
            mis += int(s.status != st or (st == "COMPLETE" and int(s.selected_k) != kk))
            recomputed.append((c, n, crit, st, kk))
    check("selection recomputed == selection_by_replicate", mis == 0, f"mismatches={mis}")
    rc = pd.DataFrame(recomputed, columns=["condition", "n", "criterion", "status", "k"])
    fm = 0
    for r in freq.itertuples():
        g = rc[(rc.condition == r.condition) & (rc.n == r.n) & (rc.criterion == r.criterion)]
        fm += int((g.k == 5).sum() != r.exact or (g.k < 5).sum() != r.under or (g.k > 5).sum() != r.over
                  or (g.status != "COMPLETE").sum() != r.incomplete)
    check("selection_frequency recomputed", fm == 0, f"mismatches={fm}")
    pm = 0
    for n in runs:
        sl = {}
        for (c, rep), g in fr[fr.n == n].groupby(["condition", "replicate"]):
            m = g.groupby("k_est").C_Q.mean(); sl[(c, rep)] = m[4] - m[5]
        for c in "BCD":
            diffs = [sl[(c, r)] - sl[("A", r)] for r in range(1, 9) if (c, r) in sl]
            if not diffs:
                continue
            row = summ[(summ.n == n) & (summ.criterion == "C_Q") & (summ.contrast == f"{c}-A") & (summ.metric == "selection_slack")]
            pm += int(row.empty or not math.isclose(float(row["median"].iloc[0]), float(np.median(diffs)), rel_tol=1e-9, abs_tol=1e-9)
                      or int(row.n_positive.iloc[0]) != sum(d > 0 for d in diffs))
    check("paired C_Q slack medians / sign counts recomputed", pm == 0, f"mismatches={pm}")

    # 6. analysis manifest hashes, figures present
    hm = sum(sha256(AN / k) != v for k, v in man["outputs_sha256"].items())
    check("analysis CSV hashes == manifest", hm == 0)
    figs = sorted(p.name for p in FIG.glob("*.png"))
    check("9 figure groups present (png+pdf)", len([f for f in figs if f.startswith(tuple(f"fig{i}_" for i in range(1, 10)))]) >= 9
          and all((FIG / f.replace(".png", ".pdf")).exists() for f in figs), len(figs))
    fh = sum(sha256(FIG / k) != v for k, v in man["figures_sha256"].items())
    check("figure hashes == manifest", fh == 0)

    # 7. historical artifacts untouched
    out = subprocess.run(["git", "diff", "--name-status", "origin/main...HEAD"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    non_add = [ln for ln in out.splitlines() if ln and not ln.startswith("A\t")]
    allowed = ("EXPERIMENT_REGISTRY.md",)
    bad_mod = [ln for ln in non_add if not ln.split("\t")[-1].endswith(allowed)]
    check("no existing file modified except append-only registry", not bad_mod, bad_mod[:5])
    if any(ln.endswith("EXPERIMENT_REGISTRY.md") for ln in non_add):
        d = subprocess.run(["git", "diff", "origin/main...HEAD", "--", "EXPERIMENT_REGISTRY.md"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8").stdout
        removed = [ln for ln in d.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        check("EXPERIMENT_REGISTRY.md append-only (no removed lines)", not removed, len(removed))

    rep = {"script": "tools/research_audit/audit_k_signal_strength.py",
           "independence": "own numpy/scipy log-likelihoods; selection/frequency/paired medians recomputed from fit_results",
           "n_checks": len(checks), "n_pass": sum(c["result"] == "PASS" for c in checks),
           "n_fail": sum(c["result"] == "FAIL" for c in checks), "checks": checks}
    (OUT / "audit_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"{rep['n_pass']}/{rep['n_checks']} PASS")
    return 0 if rep["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
