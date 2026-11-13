"""AF6 step 10: independent re-verification of the AF6 results (after the run; no new MCEM, no new data).

Uses the Phase-A independent implementation (audit_af6_af5_independent: own log joint, gradient,
full negative Hessian incl. Y cross blocks, own Newton mode, eigen log-det, CMP2 closed-form normaliser),
NOT the AF5 code path, to recompute from the stored AF6 fit states:

  1. data independence / provenance: AF6 X/Y hashes vs the 20 AF2 datasets, regenerated hashes vs
     fit_manifest, npz sha256 vs fit_manifest;
  2. every OK candidate score S (Laplace at the stored f*), Delta, status (AF5 rule re-implemented),
     Laplace selection;
  3. the primary endpoint (sparse D_r = #correct A - #correct B per dataset, 40 datasets, percentile
     bootstrap 10000, seed 20261116) and the dense secondary, from the recomputed selections and the
     stored same-anchor Q selections, without using analyze_af6_laplace_confirmation.

Q itself (AF3 score_candidate) is not re-implemented here.

Writes to <AF6>/verification/ .
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_af6_af5_independent as aud        # noqa: E402
import run_af6_laplace_confirmation as a6      # noqa: E402

af2 = a6.af2
OUT = a6.OUT_ROOT / "verification"
FULL = a6.OUT_ROOT / "full"
FINITE = ("DECIDED", "DECIDED_LAPLACE_UNASSESSED", "UNDECIDED_LAPLACE", "UNDECIDED_NUMERIC")
BOOT_SEED = 20261116


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def independence(fm: pd.DataFrame):
    rows = []
    af2_x, af2_y = set(), set()
    for r in range(len(af2.DATA_SEEDS)):
        ds = af2.generate(r)
        af2_x.add(aud.sha16(ds.X))
        af2_y.add(aud.sha16(ds.Y))
    af6_x = []
    for r in range(a6.N_DATASETS):
        ds, masks = a6.dataset(r)
        xs, ys = aud.sha16(ds.X), aud.sha16(ds.Y)
        af6_x.append(xs)
        for g in a6.REGIMES:
            m = fm[(fm.replicate == a6.label(r)) & (fm.regime == g)].iloc[0]
            rows.append({"replicate": a6.label(r), "regime": g, "x_sha": xs, "y_sha": ys,
                         "x_in_af2": xs in af2_x, "y_in_af2": ys in af2_y,
                         "x_matches_manifest": xs == m.x_sha, "y_matches_manifest": ys == m.y_sha,
                         "train_matches_manifest": aud.sha16(masks[g]) == m.train_mask_sha,
                         "test_matches_manifest": aud.sha16(masks["test"]) == m.test_mask_sha,
                         "npz_sha256_matches_manifest": sha256_file(m.state_file) == m.state_sha256})
    d = pd.DataFrame(rows)
    d["af6_x_unique_within_af6"] = len(set(af6_x)) == a6.N_DATASETS
    return d


def recompute(cs: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    out, cache = [], {}
    t0 = time.time()
    ok = cs[cs.status == "OK"]
    for i, c in enumerate(ok.itertuples()):
        r = int(c.replicate[3:]) - 1
        key = (r, c.regime)
        if key not in cache:
            ds, masks = a6.dataset(r)
            m = fm[(fm.replicate == c.replicate) & (fm.regime == c.regime)].iloc[0]
            with np.load(m.state_file) as z:
                st = {k: np.asarray(z[k]) for k in z.files}
            cache[key] = (np.asarray(ds.X, float), np.asarray(ds.Y, float), masks[c.regime], st,
                          m.assignment.split("|"))
        X, Y, train, st, assign = cache[key]
        fams = list(assign)
        fams[int(c.column)] = c.candidate
        F = np.array(st["F"], float)
        F[int(c.column)] = aud.parse_f(c.f_star)
        w0, w, var_z = (float(v) for v in st["scalars"])
        J = aud.Joint(X, Y, train, fams, F, np.diag(st["sigma"]), w0, w, var_z)
        L = aud.laplace_S(J, np.asarray(st["Zmode"], float))
        out.append({"replicate": c.replicate, "regime": c.regime, "column": int(c.column), "candidate": c.candidate,
                    "S_af6": c.S, "S_indep": L["S"], "S_diff": L["S"] - c.S, "logdet_af6": c.logdet,
                    "logdet_indep": L["logdet"], "min_eig_indep": L["min_eig"], "grad_inf_indep": L["grad_inf"],
                    "start_disagreement": c.start_disagreement})
        if i % 100 == 0:
            print(f"  scores {i}/{len(ok)} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(out)


def boot(v):
    v = np.asarray(v, float)
    rng = np.random.default_rng(BOOT_SEED)
    m = v[rng.integers(0, v.size, size=(10000, v.size))].mean(axis=1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def endpoints(sel: pd.DataFrame, rec: pd.DataFrame):
    piv = rec.pivot_table(index=["replicate", "regime", "column"], columns="candidate", values="S_indep")
    s = sel.join((piv["cmp_nu2p0"] - piv["poisson"]).rename("delta_indep"), on=["replicate", "regime", "column"])
    sd = rec.groupby(["replicate", "regime", "column"]).start_disagreement.max().rename("start_dis")
    s = s.join(sd, on=["replicate", "regime", "column"])
    s["is_readable"] = s.is_readable.astype(str).str.lower() == "true"
    for col in ("is_delta_correction", "is_delta_correction_se", "delta"):
        s[col] = pd.to_numeric(s[col], errors="coerce")
    tmp = s.rename(columns={"delta": "delta_af6"}).assign(delta=s.delta_indep)
    s["status_indep"] = [aud.status_rule(r) for r in tmp.itertuples()]
    q = np.where(s.q_status == "SELECTED", s.q_selected, "")
    fin = s.status_indep.isin(FINITE) & np.isfinite(s.delta_indep)
    s["sel_A_indep"] = np.where(fin, np.where(s.delta_indep > 0, "cmp_nu2p0", "poisson"), q)
    s["ok_A_indep"] = s.sel_A_indep == s.true_model
    s["ok_B_indep"] = q == s.true_model
    res = {"columns": len(s), "status_mismatch": int((s.status_indep != s.status).sum()),
           "laplace_selection_mismatch": int((fin & (s.sel_A_indep != s.laplace_selected)).sum()),
           "delta_max_abs_diff": float((s.delta_indep - s.delta).abs().max())}
    reps = [a6.label(r) for r in range(a6.N_DATASETS)]
    for g in a6.REGIMES:
        t = s[s.regime == g]
        v = (t.ok_A_indep.astype(int) - t.ok_B_indep.astype(int)).groupby(t.replicate).sum().reindex(reps, fill_value=0)
        lo, hi = boot(v.to_numpy())
        res[f"{g}_acc_A"] = float(t.ok_A_indep.mean())
        res[f"{g}_acc_B"] = float(t.ok_B_indep.mean())
        res[f"{g}_D_mean"], res[f"{g}_D_ci"] = float(v.mean()), [lo, hi]
        res[f"{g}_D_pos_zero_neg"] = [int((v > 0).sum()), int((v == 0).sum()), int((v < 0).sum())]
    res["primary_success_indep"] = bool(res["sparse_D_mean"] > 0 and res["sparse_D_ci"][0] > 0)
    return res, s


def tables():
    """Descriptive tables quoted in README (status x true family; compute time). Post-hoc description."""
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(FULL / "analysis" / "column_annotated.csv")
    t = c.groupby(["regime", "status", "true_model", "ok_A"]).size().rename("columns").reset_index()
    t.to_csv(OUT / "posthoc_status_by_true_family.csv", index=False)
    fm = pd.read_csv(FULL / "fit_manifest.csv")
    cs = pd.read_csv(FULL / "column_selection.csv")
    rows = []
    for g in a6.REGIMES:
        f, s = fm[fm.regime == g], cs[cs.regime == g]
        rows.append({"regime": g, "fits": len(f), "fit_wall_s_mean": f.wall_runtime_s.mean(),
                     "fit_wall_h_total": f.wall_runtime_s.sum() / 3600, "select_columns": len(s),
                     "select_s_mean_per_column": s.seconds_total.mean(), "select_h_total": s.seconds_total.sum() / 3600})
    pd.DataFrame(rows).to_csv(OUT / "compute_time.csv", index=False)
    print(t.to_string())
    print(pd.DataFrame(rows).to_string())


def main():
    if "--tables" in sys.argv:
        tables()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fm = pd.read_csv(FULL / "fit_manifest.csv")
    cs = pd.read_csv(FULL / "candidate_scores.csv")
    sel = pd.read_csv(FULL / "column_selection.csv", keep_default_na=False, na_values=[""])
    ind = independence(fm)
    ind.to_csv(OUT / "data_independence.csv", index=False)
    rec = recompute(cs, fm)
    rec.to_csv(OUT / "indep_score_recomputation.csv", index=False)
    res, s = endpoints(sel, rec)
    s.to_csv(OUT / "indep_column_status.csv", index=False)
    stat = pd.read_csv(a6.OUT_ROOT / "statistical_analysis.csv")
    prim = stat[stat.endpoint == "PRIMARY"].iloc[0]
    res.update({
        "independence_all_x_new": bool(~ind.x_in_af2.any()), "independence_all_y_new": bool(~ind.y_in_af2.any()),
        "provenance_all_match": bool(ind[[c for c in ind.columns if c.endswith("matches_manifest")]].all().all()),
        "af6_x_unique": bool(ind.af6_x_unique_within_af6.all()),
        "scores_recomputed": len(rec), "S_max_abs_diff": float(rec.S_diff.abs().max()),
        "min_eig_indep_min": float(rec.min_eig_indep.min()), "grad_inf_indep_max": float(rec.grad_inf_indep.max()),
        "primary_reported": [float(prim.mean_D), float(prim.ci_low), float(prim.ci_high)],
        "primary_matches_reported": bool(np.isclose(prim.mean_D, res["sparse_D_mean"])
                                         and np.allclose([prim.ci_low, prim.ci_high], res["sparse_D_ci"])),
        "note": "Q (AF3 score_candidate) is not re-implemented; B uses the stored same-anchor Q selection.",
    })
    res["runinfo"] = a6.runinfo({"stage": "verification", "script": "expfam/src/experimental/verify_af6_results.py"})
    (OUT / "verification_summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "runinfo"}, indent=2, default=str))


if __name__ == "__main__":
    main()
