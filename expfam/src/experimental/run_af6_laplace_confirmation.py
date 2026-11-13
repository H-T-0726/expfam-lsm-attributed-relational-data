"""AF6: confirmation of AF5's local Laplace family selection on 40 NEW datasets.

CONFIRMATORY for one pre-registered primary endpoint (sparse-Y, Laplace vs the
Q-score on the SAME Auto anchor); everything else is secondary / diagnostic.

Design (fixed in protocol.json before any new data are generated):
  data       : AF2 generator and truth (n=75, K=3, Gaussian x3, Poisson x3,
               CMP nu=2 x3, tau=2, Y Bernoulli w0=-1, w=1), 40 NEW seeds,
               new test/thinning seeds (same algorithm as m1.build_masks)
  anchor     : one AF2-identical Auto (Poisson start) exploration + refit per
               dataset x regime (80 NEW MCEM fits; no Oracle/Wrong fits)
  methods    : A Laplace (AF5 run_column, unchanged code and settings, Auto
               anchor only)  B Q on the same anchor (AF3 Z7 route: existing
               selector on the refit Zsamples, zero start)  C AF2 Auto's own
               selection  D conservative: Laplace where AF5 status == DECIDED,
               otherwise B  E joint-MAP variant (diagnostic)
  approx.    : thermodynamic integration (TI) of Delta at the Laplace optima
               (independent implementation, audit_af6_af5_independent.py) on
               every sparse column and on dense columns with |Delta_L| < 2

Usage
    python run_af6_laplace_confirmation.py --write-protocol
    python run_af6_laplace_confirmation.py --smoke          # 1 extra seed, not one of the 40
    python run_af6_laplace_confirmation.py --fits [--shard i/n]
    python run_af6_laplace_confirmation.py --select [--shard i/n]
    python run_af6_laplace_confirmation.py --analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import audit_af6_af5_independent as aud                                 # noqa: E402
import run_af5_local_laplace_selection as af5                           # noqa: E402
from run_auto_family_z_recovery import sha16, sha256_file, write_csv   # noqa: E402

af4, af2 = af5.af4, af5.af2
m1 = af2.m1
EXPERIMENT = "AF6"
N, K, D = af5.N, af5.K, af5.D
TRUE_MODELS = list(af2.TRUE_MODELS)
COUNT_COLS = af5.COUNT_COLS
CANDS = af5.CANDS
REGIMES = ("dense", "sparse")
N_DATASETS = 40

# seeds: disjoint from AF1-AF5 (AF2 1661001-1661020 / 1662000 / 1663000;
# AF3-AF5 1664000-1667000; AF6 Phase A TI 1668000; m1 1341000-1343000)
DATA_SEED_BASE = 1681001          # data seed = base + r, r = 0..39
MODEL_SEED_BASE = 1682000
SEARCH_SEED_BASE = 1683000
SPLIT_SEED_BASE = 1684000         # test mask
THIN_SEED_BASE = 1685000          # dense/sparse thinning: base + 10 r + regime index
IS_SEED_BASE = 1686000
TI_SEED_BASE = 1687000
SMOKE_DATA_SEED = 1689001         # smoke only; never analysed
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20261116
PRIOR_SEED_SETS = {
    "AF2_data": list(range(1661001, 1661021)), "AF2_model": [1662000 + r for r in range(20)],
    "AF2_search": [1663000 + r for r in range(20)], "m1_split": [1341000 + r for r in range(40)],
    "m1_thin": [1342000 + r for r in range(400)], "AF5_IS": [1667000 + i for i in range(1200)],
    "AF6A_TI": [1668000 + i for i in range(400)],
}
TI_DENSE_ABS_DELTA = 2.0

OUT_ROOT = ROOT / "expfam" / "results" / "auto_family_z_recovery" / "af6_laplace_confirmation_20261114"
STATE_ROOT = Path(r"D:\tento\kennkyu_artifacts") / "af6_laplace_confirmation_20261114"


def label(r: int) -> str:
    return f"rep{r + 1:02d}"


def seeds(r: int) -> dict[str, int]:
    return {"data_seed": DATA_SEED_BASE + r, "model_seed": MODEL_SEED_BASE + r,
            "search_seed": SEARCH_SEED_BASE + r, "split_seed": SPLIT_SEED_BASE + r,
            "thin_seed_dense": THIN_SEED_BASE + 10 * r, "thin_seed_sparse": THIN_SEED_BASE + 10 * r + 1}


def generate(seed: int):
    from data_generator_multisupport import generate_multisupport_data
    reg, _ = af2.registry_and_priority()
    return generate_multisupport_data(n=N, k=K, seed=seed, column_specs=[reg[m] for m in TRUE_MODELS],
                                      tau=af2.TAU, w0=af2.W0, w=af2.W)


def build_masks(split_seed: int, thin_dense: int, thin_sparse: int) -> dict[str, np.ndarray]:
    """Same algorithm as m1.build_masks, new seeds."""
    pool, test = m1.make_pair_split(N, m1.TEST_RATIO, seed=split_seed)
    masks = {"test": test, "pool": pool}
    for (regime, rate), s in zip(m1.REGIMES.items(), (thin_dense, thin_sparse)):
        masks[regime] = m1.thin_pool_mask(pool, rate, seed=s)
    return masks


def dataset(r: int):
    s = seeds(r)
    return generate(s["data_seed"]), build_masks(s["split_seed"], s["thin_seed_dense"], s["thin_seed_sparse"])


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------

def protocol() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "study_type": "CONFIRMATORY for the single primary endpoint; all other endpoints secondary/diagnostic",
        "lineage": "E + PR #129 generic model (unmerged prototype; not manuscript evidence)",
        "phase_a": "independent audit of AF5 (phase_a/, independent_audit.md) completed before this protocol",
        "data": {"generator": "AF2 generate (data_generator_multisupport), truth " + "|".join(TRUE_MODELS),
                 "n": N, "K": K, "tau": af2.TAU, "w0": af2.W0, "w": af2.W, "nu_cmp": 2.0,
                 "datasets": N_DATASETS, "seeds": "seed_manifest.csv",
                 "masks": "m1 algorithm (test ratio 0.10; dense keep 1.0, sparse keep 0.1 of the pool) with new seeds",
                 "seed_disjointness": "seed values checked against AF1-AF5 seed sets before generation (seed_disjointness.json); X/Y hashes compared with the 20 AF2 datasets after generation (analysis)"},
        "mcem": {"fits": "auto_first only (AF2 run_fit: START_FIRST exploration + refit)",
                 "planned_fits": N_DATASETS * 2, "L": af2.L, "exploration_iters": af2.EXPLORATION_ITERS,
                 "refit_iters": af2.REFIT_ITERS, "newton_alpha": af2.NEWTON_ALPHA, "route": af2.ROUTE,
                 "states": str(STATE_ROOT / "fits"), "no_oracle_or_wrong_fits": True},
        "methods": {
            "A_laplace": "AF5 run_column unchanged (protocol sha b0601adb...; settings INNER/OUTER/IS as AF5), "
                         "Auto anchor; Delta = S(CMP2) - S(Poisson); selection by sign of Delta",
            "B_q_same_anchor": "family_selection_multisupport.score_candidate on the Auto refit Zsamples "
                               "(L=5), zero start, trust_exact, select_column (= AF3 arm Z7)",
            "C_af2_auto": "the Auto exploration's own selection (assignment of the fit)",
            "D_conservative": "Laplace selection where AF5 status == DECIDED (IS readable and passed); "
                              "otherwise B. No label information; rule fixed here",
            "E_jointmap": "sign of AF5 delta_jointmap_variant (diagnostic; never promoted)",
        },
        "laplace_status_rule": "AF5 rule unchanged: FAILED / UNDECIDED_NOT_CONVERGED / UNDECIDED_NUMERIC / "
                               "UNDECIDED_LAPLACE / DECIDED_LAPLACE_UNASSESSED / DECIDED",
        "primary_endpoint": {
            "regime": "sparse",
            "D_r": "number of count columns (6 per dataset) correctly selected by A minus by B, per dataset",
            "A_column_rule": "A uses the sign of Delta for every column with a finite Delta "
                             "(statuses DECIDED, DECIDED_LAPLACE_UNASSESSED, UNDECIDED_LAPLACE, UNDECIDED_NUMERIC); "
                             "columns with no finite Delta (FAILED / UNDECIDED_NOT_CONVERGED) use B's selection",
            "B_column_rule": "if B fails for a column, that column counts as incorrect for B",
            "fit_failure": "if the Auto fit of a dataset x regime fails, both A and B have no selection: D_r = 0 "
                           "(recorded in failure_registry.csv; no regeneration, no exclusion)",
            "statistic": "mean of D_r over the 40 datasets",
            "interval": "percentile bootstrap over datasets, 10000 resamples, seed 20261116, 95%",
            "success": "mean D_r > 0 AND lower 95% bound > 0",
            "unit": "dataset (40); dense and sparse are not independent replicates",
        },
        "secondary_endpoints": [
            "dense D_r (expected: no improvement; same statistic)",
            "A vs C (AF2 Auto) D_r, dense and sparse",
            "column accuracy by true family (CMP2 / Poisson) for A, B, C, D, E (denominator: all 240 columns per regime)",
            "corrections (B wrong, A right) and harms (B right, A wrong), also vs C; Poisson->CMP2 harms separately",
            "status rates (undecided, IS-unassessed)",
            "D (conservative) accuracy, D_r vs B, and the share of columns where D differs from B",
            "E (joint-MAP) D_r vs B and agreement with A",
            "computation time",
        ],
        "approximation_diagnostic": {
            "method": "thermodynamic integration of Delta at the Laplace optima (f_P*, f_C*): "
                      "log p_C - log p_P = int_0^1 E_t[l_C(x_j|Z f_C*) - l_P(x_j|Z f_P*)] dt, "
                      f"{aud.GL_NODES} Gauss-Legendre nodes, whitened HMC (eps {aud.HMC_EPS}, {aud.HMC_STEPS} steps, "
                      f"{aud.HMC_CHAINS} chains x {aud.HMC_WARM}+{aud.HMC_KEEP})",
            "subset": f"every sparse column; dense columns with |Delta_L| < {TI_DENSE_ABS_DELTA} (rule by Laplace, not by labels)",
            "ti_verified": "sign(Delta_TI) == sign(Delta_L) and |Delta_TI| > 2 SE_TI",
            "ti_selection": "sign(Delta_TI) where TI was run, else sign(Delta_L); the primary statistic is recomputed "
                            "with it (RQ4: improvement at the exact Z-marginal at f*, not the profile maximum)",
            "reporting": "all-column and TI-verified-column results are reported separately; columns are never "
                         "dropped from the primary endpoint",
            "limits": "TI targets the marginal at the Laplace optima f*; the exact profile maximum over f_j "
                      "differs by second-order terms; HMC Monte Carlo error is reported as SE_TI",
        },
        "multiplicity": "one primary test; secondary results are descriptive",
        "not_claimed": ["Z recovery after re-fitting a corrected assignment", "a final adopted algorithm",
                        "generality beyond this one condition"],
        "deviations_policy": "any deviation is recorded in README with reason; no setting is changed after "
                             "AF6 selection results are seen",
        "smoke": f"one extra dataset (data seed {SMOKE_DATA_SEED}, mask seeds +1..+3) ran the whole pipeline before this "
                 "protocol was frozen (scratch output, pipeline check only: 2 fits OK, 12 columns, 0 failures); never analysed",
        "compute_plan": "fits ~75 s per dataset (2 regimes); selection ~95 s (dense) + ~540 s (sparse incl. TI) per dataset; "
                        "40 datasets ~8 h serial, run as 5 shards (OMP threads 2); shards do not change any seed",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "power_note": "planning only, from AF5 (exploratory, likely inflated): sparse D_r SD 0.75; "
                      "normal-approx power at n=40 is 0.92 / 0.71 / 0.39 for true means 0.4 / 0.3 / 0.2",
        "budget": {"mcem_fits": N_DATASETS * 2, "laplace_columns": N_DATASETS * 2 * 6},
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def runinfo(extra: dict[str, Any]) -> dict[str, Any]:
    import scipy
    req = ROOT / "requirements.txt"
    return {"experiment": EXPERIMENT, "script": "expfam/src/experimental/run_af6_laplace_confirmation.py",
            "git_head": m1.git("rev-parse", "HEAD"), "git_branch": m1.git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_tracked": bool(m1.git("status", "--porcelain", "--untracked-files=no")),
            "git_untracked_experimental": m1.git("status", "--porcelain", "--", "expfam/src/experimental"),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "requirements_sha256": sha256_file(req) if req.exists() else "",
            "protocol_sha256": protocol_sha256(), "command": " ".join(sys.argv), **extra}


def seed_manifest() -> list[dict[str, Any]]:
    rows = []
    for r in range(N_DATASETS):
        rows.append({"replicate": label(r), "r": r, **seeds(r), "is_seed_base": IS_SEED_BASE,
                     "ti_seed_base": TI_SEED_BASE})
    return rows


def seed_disjointness() -> dict[str, Any]:
    new = set()
    for row in seed_manifest():
        new |= {v for k, v in row.items() if k.endswith("seed") or k.startswith("thin_seed") or k.startswith("split")}
    hits = {k: sorted(new & set(v)) for k, v in PRIOR_SEED_SETS.items()}
    return {"n_new_seeds": len(new), "overlaps": {k: v for k, v in hits.items() if v}}


# --------------------------------------------------------------------------
# fits
# --------------------------------------------------------------------------

def fit_one(r: int, regime: str, ds, masks) -> tuple[dict, list]:
    s = seeds(r)
    fit = {"r": r, "replicate": label(r), "data_seed": s["data_seed"], "regime": regime, "method": "auto_first",
           "model_seed": s["model_seed"], "search_seed": s["search_seed"], "start": "START_FIRST"}
    run, est_rows, col_rows, trace_rows, states = af2.run_fit(fit, ds, masks)
    if run["status"] == "OK":
        (STATE_ROOT / "fits").mkdir(parents=True, exist_ok=True)
        path = STATE_ROOT / "fits" / f"{label(r)}_{regime}_auto_first.npz"
        np.savez(path, **states)
        run["state_file"] = str(path)
        run["state_sha256"] = sha256_file(path)
    return run, col_rows


def shard_of(items, shard):
    if not shard:
        return items
    i, n = (int(v) for v in shard.split("/"))
    return [x for k, x in enumerate(items) if k % n == i]


def run_fits(out: Path, shard: str | None) -> None:
    prog = out / f"fits_progress{'_' + shard.replace('/', 'of') if shard else ''}.jsonl"
    done = set()
    if prog.exists():
        done = {(d["run"]["replicate"], d["run"]["regime"]) for d in map(json.loads, prog.read_text(encoding="utf-8").splitlines())}
    t0 = time.time()
    for r in shard_of(list(range(N_DATASETS)), shard):
        ds, masks = dataset(r)
        for regime in REGIMES:
            if (label(r), regime) in done:
                continue
            run, cols = fit_one(r, regime, ds, masks)
            with prog.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run": run, "cols": cols}, default=str) + "\n")
            print(f"fit {label(r)} {regime} {run['status']} {run.get('assignment', '')} {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# selection: A (Laplace), B (Q same anchor), TI
# --------------------------------------------------------------------------

def q_same_anchor(ds, st, j) -> dict[str, Any]:
    from family_selection_multisupport import score_candidate, select_column
    from family_specs import GROUP_COUNT
    reg, prio = af2.registry_and_priority()
    Zs = np.asarray(st["Zsamples"], float)
    x = np.asarray(ds.X[:, j], float)
    recs = [score_candidate(x, Zs, reg[c], np.zeros(K), j, route=af2.ROUTE) for c in CANDS]
    sel = select_column(recs, prio[GROUP_COUNT])
    sc = {r_.candidate_model: r_ for r_ in recs}
    return {"q_status": sel.status, "q_selected": sel.selected if sel.status == "SELECTED" else "",
            "q_score_poisson": sc["poisson"].score, "q_score_cmp_nu2p0": sc["cmp_nu2p0"].score,
            "q_margin_cmp_minus_pois": sc["cmp_nu2p0"].score - sc["poisson"].score,
            "q_failure": sel.failure_reason or ""}


def run_ti(r, regime, j, ds, masks, st, assignment, fP, fC) -> dict[str, Any]:
    fams = list(assignment)
    fams[j] = "poisson"
    F = np.array(st["F"], float)
    F[j] = fP
    w0, w, var_z = (float(v) for v in st["scalars"])
    JP = aud.Joint(np.asarray(ds.X, float), np.asarray(ds.Y, float), masks[regime], fams, F,
                   np.diag(st["sigma"]), w0, w, var_z)
    seed = TI_SEED_BASE + (r * 2 + REGIMES.index(regime)) * 10 + j
    return aud.ti_delta_joint(JP, j, fP, fC, np.asarray(st["Zmode"], float), seed)


def run_select(out: Path, shard: str | None, fits: dict, replicates=None, state_dir=None) -> None:
    af4.AF2_STATE = state_dir or (STATE_ROOT / "fits")            # anchors = AF6 Auto fits
    af5.IS_SEED_BASE = IS_SEED_BASE
    prog = out / f"select_progress{'_' + shard.replace('/', 'of') if shard else ''}.jsonl"
    done = set()
    if prog.exists():
        done = {(d["col"]["replicate"], d["col"]["regime"], d["col"]["column"])
                for d in map(json.loads, prog.read_text(encoding="utf-8").splitlines())}
    reps = replicates if replicates is not None else list(range(N_DATASETS))
    t0 = time.time()
    for r in shard_of(reps, shard):
        ds, masks = (smoke_dataset() if r == "smoke" else dataset(r))
        rr = 0 if r == "smoke" else r
        for regime in REGIMES:
            key = f"{label(rr)}_{regime}_auto_first"
            fr = fits.get(key)
            if fr is None or fr["status"] != "OK":
                continue                                  # recorded as fit failure in analysis
            st = af4.load_state(rr, regime, "auto_first")
            assignment = fr["assignment"].split("|")
            for j in COUNT_COLS:
                if (label(rr), regime, j) in done:
                    continue
                tc = time.time()
                with warnings.catch_warnings(record=True) as wl:
                    warnings.simplefilter("always")
                    cands, col = af5.run_column(rr, regime, "auto_first", j, ds, masks, {key: fr},
                                                case_index=(rr * 2 + REGIMES.index(regime)) * 10 + j)
                    try:
                        col.update(q_same_anchor(ds, st, j))
                    except Exception as exc:
                        col.update({"q_status": "FAILED", "q_selected": "", "q_failure": f"{type(exc).__name__}: {exc}"[:200]})
                    col["af2_auto_selected"] = assignment[j]
                    col["ti_run"] = False
                    if col.get("status") not in ("FAILED", None) and math.isfinite(col.get("delta", float("nan"))) \
                            and (regime == "sparse" or abs(col["delta"]) < TI_DENSE_ABS_DELTA):
                        cp = {c["candidate"]: c for c in cands}
                        try:
                            ti = run_ti(rr, regime, j, ds, masks, st, assignment,
                                        aud.parse_f(cp["poisson"]["f_star"]), aud.parse_f(cp["cmp_nu2p0"]["f_star"]))
                            col.update({"ti_run": True, **{f"ti_{k}" if not k.startswith("ti_") and not k.startswith("delta_ti")
                                                            else k: v for k, v in ti.items()}})
                        except Exception as exc:
                            col.update({"ti_run": True, "ti_error": f"{type(exc).__name__}: {exc}"[:200]})
                col["n_warnings"] = len(wl)
                col["seconds_total"] = round(time.time() - tc, 2)
                with prog.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"col": col, "cands": cands}, default=str) + "\n")
                print(f"sel {label(rr)} {regime} j={j} {col.get('status')} {time.time() - tc:.0f}s total {time.time() - t0:.0f}s", flush=True)


def smoke_dataset():
    return generate(SMOKE_DATA_SEED), build_masks(SMOKE_DATA_SEED + 1, SMOKE_DATA_SEED + 2, SMOKE_DATA_SEED + 3)


def load_fits(out: Path) -> dict:
    fits = {}
    for p in sorted(out.glob("fits_progress*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            fits[f"{d['run']['replicate']}_{d['run']['regime']}_auto_first"] = d["run"]
    return fits


def collect(out: Path) -> None:
    """Write fit_manifest.csv, af2_auto_columns.csv, candidate_scores.csv, column_selection.csv, failure_registry.csv."""
    runs, cols = [], []
    for p in sorted(out.glob("fits_progress*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            runs.append(d["run"])
            cols += d["cols"]
    sel, cands = [], []
    for p in sorted(out.glob("select_progress*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            sel.append(d["col"])
            cands += d["cands"]
    key = lambda x: (x["replicate"], x["regime"], int(x.get("column", 0)), x.get("candidate", ""))  # noqa: E731
    write_csv(out / "fit_manifest.csv", sorted(runs, key=lambda x: (x["replicate"], x["regime"])))
    write_csv(out / "af2_auto_columns.csv", sorted(cols, key=key))
    write_csv(out / "candidate_scores.csv", sorted(cands, key=key))
    write_csv(out / "column_selection.csv", sorted(sel, key=key))
    fail = [{"stage": "mcem_fit", "replicate": r_["replicate"], "regime": r_["regime"], "column": "",
             "detail": r_.get("error", "")} for r_ in runs if r_["status"] != "OK"]
    fail += [{"stage": "laplace", "replicate": c["replicate"], "regime": c["regime"], "column": c["column"],
              "detail": c.get("status")} for c in sel if c.get("status") in ("FAILED", "UNDECIDED_NOT_CONVERGED")]
    fail += [{"stage": "q_same_anchor", "replicate": c["replicate"], "regime": c["regime"], "column": c["column"],
              "detail": c.get("q_failure", "")} for c in sel if c.get("q_status") != "SELECTED"]
    fail += [{"stage": "ti", "replicate": c["replicate"], "regime": c["regime"], "column": c["column"],
              "detail": c.get("ti_error", "")} for c in sel if c.get("ti_error")]
    write_csv(out / "failure_registry.csv", fail or [{"stage": "", "replicate": "", "regime": "", "column": "", "detail": "none"}])
    print(f"collected {len(runs)} fits, {len(sel)} columns, {len(fail)} failure rows")


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AF6 Laplace confirmation")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-protocol", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--fits", action="store_true")
    g.add_argument("--select", action="store_true")
    g.add_argument("--collect", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    a = ap.parse_args(argv)
    out = Path(a.out_root)
    if a.write_protocol:
        p = out / "protocol.json"
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        dj = seed_disjointness()
        if dj["overlaps"]:
            raise SystemExit(f"seed overlap with earlier experiments: {dj['overlaps']}")
        out.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(protocol(), indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(out / "seed_manifest.csv", seed_manifest())
        (out / "seed_disjointness.json").write_text(json.dumps(dj, indent=2), encoding="utf-8")
        print(f"wrote {p} sha256(canonical)={protocol_sha256()}")
        return 0
    if a.analyze:
        import analyze_af6_laplace_confirmation as an
        an.main(out / "full")
        return 0
    frozen = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest() != protocol_sha256():
        raise SystemExit("protocol.json on disk differs from protocol(); refusing to run")
    if a.smoke:
        sm = out / "smoke"
        sm.mkdir(parents=True, exist_ok=True)
        global STATE_ROOT
        STATE_ROOT = STATE_ROOT / "smoke"
        ds, masks = smoke_dataset()
        fits = {}
        for regime in REGIMES:
            fit = {"r": 0, "replicate": "rep01", "data_seed": SMOKE_DATA_SEED, "regime": regime, "method": "auto_first",
                   "model_seed": SMOKE_DATA_SEED + 10, "search_seed": SMOKE_DATA_SEED + 20, "start": "START_FIRST"}
            run, _, cols, _, states = af2.run_fit(fit, ds, masks)
            (STATE_ROOT / "fits").mkdir(parents=True, exist_ok=True)
            np.savez(STATE_ROOT / "fits" / f"rep01_{regime}_auto_first.npz", **states)
            fits[f"rep01_{regime}_auto_first"] = run
            with (sm / "fits_progress.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run": run, "cols": cols}, default=str) + "\n")
            print("smoke fit", regime, run["status"], run.get("assignment"), run.get("wall_runtime_s"))
        run_select(sm, None, fits, replicates=["smoke"], state_dir=STATE_ROOT / "fits")
        collect(sm)
        (sm / "runinfo.json").write_text(json.dumps(runinfo({"stage": "smoke"}), indent=2), encoding="utf-8")
        return 0
    full = out / "full"
    full.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    if a.fits:
        run_fits(full, a.shard)
    elif a.select:
        run_select(full, a.shard, load_fits(full))
    elif a.collect:
        collect(full)
    tag = "fits" if a.fits else "select" if a.select else "collect"
    ri = full / f"runinfo_{tag}{'_' + a.shard.replace('/', 'of') if a.shard else ''}.json"
    prev = json.loads(ri.read_text(encoding="utf-8"))["sessions"] if ri.exists() else []
    ri.write_text(json.dumps(runinfo({"stage": tag, "shard": a.shard, "sessions": prev + [
        {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
         "runtime_s": round(time.time() - t0, 1), "git_head": m1.git("rev-parse", "HEAD")}],
        "state_dir": str(STATE_ROOT / "fits")}), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
