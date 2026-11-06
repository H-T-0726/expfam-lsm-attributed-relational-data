"""Independent audit of mcem_noise_separation_20261106.

Usage:  python tools/research_audit/audit_mcem_noise_separation.py
Writes  expfam/results/k_selection/mcem_noise_separation_20261106/audit/audit_report.json (refuses to overwrite)
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
from scipy.stats import binomtest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_mcem_noise_separation as ns  # noqa: E402
import run_mcem_noise_separation_v2 as v2  # noqa: E402

OUT = ns.OUT_ROOT
AN = OUT / "analysis"
AUD = OUT / "audit"
checks: list[dict] = []


def check(name, ok, detail=""):
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail != "" else ""), flush=True)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ns.rm.ROOT, encoding="utf-8").stdout


def main() -> int:
    if AUD.exists():
        print(f"refusing to overwrite {AUD}")
        return 1
    # ---- pre-registration ----------------------------------------------------
    pre = json.loads((OUT / "gates" / "prereg.json").read_text(encoding="utf-8"))
    check("prereg.json == code PREREG", pre == json.loads(json.dumps(ns.PREREG, ensure_ascii=False)))
    gates = json.loads((OUT / "gates" / "gates.json").read_text(encoding="utf-8"))
    check("gates PASSED with the same prereg hash", gates["status"] == "PASSED" and gates["prereg_sha256"] == ns.prereg_hash())
    log = git("log", "--format=%H %s", "--reverse", "--", "tools/research_audit/run_mcem_noise_separation.py",
              f"{ns.OUT_REL}/gates", f"{ns.OUT_REL}/exec_v2/phase3")
    first_prereg = git("log", "--format=%H", "--reverse", "--", "tools/research_audit/run_mcem_noise_separation.py").split()[0]
    first_res = git("log", "--format=%H", "--reverse", "--", f"{ns.OUT_REL}/exec_v2/phase3/fits").split()
    ok = bool(first_res) and git("merge-base", "--is-ancestor", first_prereg, first_res[0]) == "" and \
        subprocess.run(["git", "merge-base", "--is-ancestor", first_prereg, first_res[0]], cwd=ns.rm.ROOT).returncode == 0
    check("prereg committed before Phase 3 results", ok, log.strip().splitlines()[0] if log else "")
    add = git("log", "--format=%H", "--reverse", "--", f"{ns.OUT_REL}/operational_addendum_v2.json").split()
    ok = bool(add) and bool(first_res) and subprocess.run(["git", "merge-base", "--is-ancestor", add[0], first_res[0]],
                                                          cwd=ns.rm.ROOT).returncode == 0
    check("operational addendum v2 committed before exec_v2 results", ok)
    rt = json.loads((v2.RESULTS_ROOT / "resilience_tests" / "report.json").read_text(encoding="utf-8"))
    check("save/resume tests all PASS before production", rt["passed"] == rt["total"], f"{rt['passed']}/{rt['total']}")
    # ---- per-fit persistence: every expected fit re-validated (manifest, config hash, checksums) ----
    ctx = v2.Ctx(v2.RESULTS_ROOT, v2.ARTIFACT_ROOT)
    for stage, L, conds in (("phase3", None, ns.CONDS), ("phase4", 20, ns.P4_CONDS), ("phase4", 50, ns.P4_CONDS)):
        st = v2.stage_status(ctx, stage, L, conds)
        sname = v2.stage_name(stage, L)
        agg = v2.RESULTS_ROOT / sname / "aggregate"
        info = json.loads((agg / "runinfo.json").read_text(encoding="utf-8")) if agg.exists() else {"status": "NOT_AGGREGATED"}
        check(f"{sname}: per-fit validation (manifest + config hash + sha256 of every file incl. external npz)",
              st["counts"]["INCOMPLETE"] == 0, f"{st['counts']} aggregate={info['status']}")
        if agg.exists():
            led = pd.read_csv(agg / "ledger.csv")
            check(f"{sname}: aggregate ledger == validation, unique fit_id", (led.status.value_counts().to_dict() ==
                  {k: v for k, v in st["counts"].items() if v}) and not led.fit_id.duplicated().any())
    L3 = pd.read_csv(v2.RESULTS_ROOT / "phase3" / "aggregate" / "ledger.csv")
    check("Phase 3 ledger: 128 fits, no duplicates", len(L3) == 128 and not L3.duplicated(["condition", "replicate", "k_fit", "start"]).any(),
          f"{L3.status.value_counts().to_dict()}")
    agg3 = v2.RESULTS_ROOT / "phase3" / "aggregate"
    S = pd.read_csv(agg3 / "state_rows.csv")
    P = pd.read_csv(agg3 / "probe_rows.csv")
    R = pd.read_csv(agg3 / "replicate_rows.csv")
    nok = int((L3.status == "OK").sum())
    check("Phase 3 row counts match OK fits (6 states, 2 probes, 32 replicate rows per fit)",
          (len(S), len(P), len(R)) == (6 * nok, 2 * nok, 32 * nok), (len(S), len(P), len(R)))
    # ---- C_Lap recomputed at every archived state == PR #156 -----------------
    cr = pd.concat([pd.read_csv(ns.SRC_RESULTS / s / "checkpoint_rows.csv") for s in ("rq1_AB", "rq1_CD")])
    m = S.merge(cr[["condition", "replicate", "k_fit", "start", "checkpoint", "C_Lap", "lap_status"]],
                on=["condition", "replicate", "k_fit", "start", "checkpoint"], suffixes=("", "_pr156"))
    okm = m[(m.lap_status == "OK") & (m.lap_status_pr156 == "OK")]
    check("C_Lap at all OK archived states == PR #156 (deterministic recompute)", float((okm.C_Lap - okm.C_Lap_pr156).abs().max()) < 1e-6,
          f"{len(okm)} states, max diff {float((okm.C_Lap - okm.C_Lap_pr156).abs().max()):.1e}; "
          f"lap status mismatches {int((m.lap_status != m.lap_status_pr156).sum())}")
    # ---- evaluation identities -------------------------------------------------
    ev = R.groupby(["condition", "replicate", "k_fit", "start", "checkpoint"]).C_Q_L5.mean().rename("mean_L5").reset_index()
    mm = P.merge(ev, on=["condition", "replicate", "k_fit", "start", "checkpoint"])
    check("C_Q(L=80) == mean of the 16 C_Q(L=5) (pooled chains)", float((mm.C_Q_L80 - mm.mean_L5).abs().max()) < 1e-6,
          f"max diff {float((mm.C_Q_L80 - mm.mean_L5).abs().max()):.1e}")
    g20 = mm[[f"C_Q_L20_g{g}" for g in range(4)]].mean(axis=1)
    check("C_Q(L=80) == mean of the four C_Q(L=20)", float((g20 - mm.C_Q_L80).abs().max()) < 1e-6)
    zq = P.Q_Z_L80.round(6).groupby(P.k_fit).nunique().max()  # noqa
    check("Q_Z is constant for a given K (global scale_Z)", int(zq) == 1)
    # ---- posterior-mean eta change recomputed from archived arrays ------------
    data = ns.rq1_data()
    worst = 0.0
    for cond in ns.CONDS:
        src = ns.src_npz(cond)
        for rep in (1, 8):
            for k in ns.KS:
                arr = np.load(v2.ARTIFACT_ROOT / "phase3" / f"{cond}_r{rep}_k{k}_s1.npz")
                key = lambda cp: f"RQ1_{cond}_r{rep}_k{k}_s1_it{cp}"  # noqa: E731
                e = {}
                for cp in (32, 64):
                    F = np.asarray(src[f"{key(cp)}_F"], float)
                    e[cp] = np.asarray(arr[f"{key(cp)}_Zbar"]) @ F.T
                d = float(np.linalg.norm(e[64] - e[32]) / np.linalg.norm(e[32]))
                row = P[(P.condition == cond) & (P.replicate == rep) & (P.k_fit == k) & (P.start == 1) & (P.checkpoint == 64)].iloc[0]
                worst = max(worst, abs(d - row.D_eta_x_pm_32_64))
    check("posterior-mean eta_x change 32->64 recomputed from archived Zbar and PR #156 F", worst < 1e-9, f"max diff {worst:.1e}")
    # ---- re-execution of two pre-specified fits --------------------------------
    for cond, rep, k, s in (("A", 8, 4, 2), ("D", 3, 5, 1)):
        st, pr, rp, _ = ns.phase3_fit(data, ns.src_npz(cond), cond, rep, k, s)
        a = pd.DataFrame(pr).set_index("checkpoint").drop(columns=["probe_seconds"])
        b = P[(P.condition == cond) & (P.replicate == rep) & (P.k_fit == k) & (P.start == s)].set_index("checkpoint")[a.columns]
        num = a.select_dtypes("number").columns
        diff = float(np.nanmax(np.abs(a[num].values.astype(float) - b[num].values.astype(float))))
        check(f"re-execution of {cond} rep{rep} K{k} s{s} reproduces probe rows", diff < 1e-8, f"max diff {diff:.1e}")
    # ---- rules recomputed ------------------------------------------------------
    ft = pd.read_csv(AN / "fixed_theta_noise.csv")
    lab = ft.G_drift_share.map(lambda x: "ONE_STEP_NOISE_DOMINATED" if x <= 0.2 else ("ONE_STEP_SYSTEMATIC_DOMINATED" if x >= 0.5 else "MIXED"))
    check("R1 labels recomputed", (lab == ft.R1_G).all())
    worst = 0.0
    for _, row in ft[(ft.replicate == 2) & (ft.start == 2)].iterrows():
        arr = np.load(v2.ARTIFACT_ROOT / "phase3" / f"{row.condition}_r{row.replicate}_k{row.k_fit}_s{row.start}.npz")
        key = f"RQ1_{row.condition}_r{row.replicate}_k{row.k_fit}_s{row.start}_it{row.checkpoint}"
        Fc = arr[f"{key}_theta_F"]
        Gc = Fc @ Fc.T
        Gs = arr[f"{key}_G_onestep"]
        mbar = Gs.mean(axis=0)
        s2 = float(((Gs - mbar) ** 2).sum() / (len(Gs) - 1))
        drift2 = max(float(((mbar - Gc) ** 2).sum()) - s2 / len(Gs), 0.0)
        worst = max(worst, abs(drift2 / (drift2 + s2) - row.G_drift_share))
    check("R1 drift share recomputed from the persisted one-step G stacks (rep 2, start 2, all conditions/K/checkpoints)",
          worst < 1e-9, f"max diff {worst:.1e}")
    it = pd.concat([pd.read_csv(ns.SRC_RESULTS / s / "iteration_rows.csv") for s in ("rq1_AB", "rq1_CD")])
    w = it[(it.iteration >= 57) & (it.iteration <= 64)].groupby(["condition", "replicate", "k_fit", "start"]).gram_rel_change.median()
    f64 = ft[ft.checkpoint == 64].set_index(["condition", "replicate", "k_fit", "start"])
    ratio = w.reindex(f64.index) / f64.median_one_step_gram
    check("R2 gram ratio recomputed from PR #156 iteration rows", float((ratio - f64.R2_ratio_gram).abs().max()) < 1e-12)
    lr = pd.read_csv(AN / "long_run_movement.csv")
    ok = True
    for q in ("G", "w0", "w", "eta_x_mode", "eta_y_mode"):
        a = S[S.checkpoint == 16].set_index(["condition", "replicate", "k_fit", "start"])[f"D64_{q}"]
        b = S[S.checkpoint == 48].set_index(["condition", "replicate", "k_fit", "start"])[f"D64_{q}"]
        sel = a.notna() & b.notna()
        cnt, n = int((a[sel] > b[sel]).sum()), int(sel.sum())
        p = binomtest(cnt, n, 0.5).pvalue
        v = "SYSTEMATIC_MOVEMENT" if (p < 0.01 and cnt / n > 0.5 and float(np.median(a[sel] / b[sel])) > 1.1) else "NO_DETECTABLE_SYSTEMATIC_MOVEMENT"
        row = lr[(lr.quantity == q) & (lr.condition == "ALL")].iloc[0]
        ok &= (row.n_D16_gt_D48 == cnt and row.verdict == v and abs(row.binom_p_two_sided - p) < 1e-12)
    check("R5 counts, p-values and verdicts recomputed", ok)
    sl = pd.read_csv(AN / "slack_by_checkpoint.csv")
    pp = P.pivot_table(index=["condition", "replicate", "checkpoint"], columns=["k_fit", "start"], values="C_Q_L80")
    s80 = ((pp[(4, 1)] - pp[(5, 1)]) + (pp[(4, 2)] - pp[(5, 2)])) / 2
    m2 = sl.set_index(["condition", "replicate", "checkpoint"]).S_L80
    check("L=80 slack recomputed", float((s80.reindex(m2.index) - m2).abs().max()) < 1e-9)
    eff = pd.read_csv(AN / "paired_effects_L80.csv")
    ok = True
    for _, r in eff[eff.criterion == "S_L80"].iterrows():
        x = sl[sl.checkpoint == r.checkpoint].set_index(["condition", "replicate"]).S_L80
        d = (x.loc[r.contrast[0]] - x.loc["A"]).values
        med, pos = float(np.median(d)), int((d > 0).sum())
        v = "MAINTAINED" if (med > 0 and pos >= 6) else ("NOT_MAINTAINED" if med <= 0 else "INCONCLUSIVE")
        ok &= abs(med - r["median"]) < 1e-9 and pos == r.n_positive and v == r.verdict
    check("R6 paired medians, counts and verdicts recomputed", ok)
    # ---- manifest -------------------------------------------------------------
    man = json.loads((AN / "analysis_manifest.json").read_text(encoding="utf-8"))
    bad = [f for f, h in man["files"].items() if hashlib.sha256((OUT / f).read_bytes()).hexdigest() != h]
    check("analysis/figure hashes == manifest", not bad, bad[:3])
    # ---- git hygiene ------------------------------------------------------------
    base = "origin/research/k-selection-mcem-stability-20261106"
    diff = git("diff", "--name-status", f"{base}...HEAD").strip().splitlines()
    nonadd = [x for x in diff if not x.startswith("A\t")]
    check("only additions vs PR #156 branch (except append-only registry)", all(x == "M\tEXPERIMENT_REGISTRY.md" for x in nonadd), nonadd)
    reg = git("diff", f"{base}...HEAD", "--", "EXPERIMENT_REGISTRY.md")
    removed = [l for l in reg.splitlines() if l.startswith("-") and not l.startswith("---")]
    check("EXPERIMENT_REGISTRY.md append-only", not removed)
    AUD.mkdir(parents=True)
    n_pass = sum(c["status"] == "PASS" for c in checks)
    (AUD / "audit_report.json").write_text(json.dumps({"experiment_id": ns.EXPERIMENT_ID, "git_head": ns.git_head(),
                                                       "passed": n_pass, "total": len(checks), "checks": checks}, indent=1),
                                           encoding="utf-8")
    pd.DataFrame(checks).to_csv(AUD / "audit_summary.csv", index=False)
    print(f"{n_pass}/{len(checks)} PASS")
    return 0 if n_pass == len(checks) else 2


if __name__ == "__main__":
    sys.exit(main())
