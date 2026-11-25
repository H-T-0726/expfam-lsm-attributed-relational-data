"""Independent audit of the K-selection Exp2-generalisation main run (8,000 fits).

Written after the run, for the audit requested on 2026-11-25. It does NOT import
the pre-specified analysis script and does not use the design module for any
count: constants come from the frozen protocol.json, the criterion definition is
re-implemented here, and selections are recomputed from per_fit.csv only. The
design module is used in one place only, the optional data-regeneration spot
check (--regen N), whose purpose is to test reproducibility of the generator.

Usage:
  python tools/research_audit/audit_k_exp2_independent.py \
      --base expfam/results/k_selection/k_exp2_generalization_20261125 \
      --out  <new dir> [--regen 8]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_git_blob(rev: str, path: str) -> str:
    """SHA-256 of the committed file, raw and as checked out with CRLF.

    The frozen hashes were taken from working-tree bytes. With core.autocrlf=true
    the blob is stored with LF and checked out with CRLF, so the blob is hashed
    after LF -> CRLF conversion (and also reported raw)."""
    data = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                          capture_output=True, check=True).stdout
    CR, LF = bytes([13]), bytes([10])
    crlf = data.replace(CR + LF, LF).replace(LF, CR + LF)
    return {"raw": hashlib.sha256(data).hexdigest(),
            "crlf": hashlib.sha256(crlf).hexdigest()}


def read_csv(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regen", type=int, default=0)
    a = ap.parse_args()
    base = (ROOT / a.base) if not Path(a.base).is_absolute() else Path(a.base)
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    if out.exists():
        print(f"STOP: {out} exists", file=sys.stderr)
        return 2
    out.mkdir(parents=True)

    checks: dict[str, dict] = {}

    def check(name, ok, **detail):
        checks[name] = {"ok": bool(ok), **detail}

    proto_p = base / "protocol_v2" / "protocol.json"
    proto = json.loads(proto_p.read_text(encoding="utf-8"))
    runinfo = json.loads((base / "main" / "runinfo.json").read_text(encoding="utf-8"))
    events = [json.loads(l) for l in (base / "main" / "run_events.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    man_p = base / "protocol_v2" / "manifest.csv"
    man = read_csv(man_p)
    rows = read_csv(base / "main" / "per_fit.csv")
    fails = read_csv(base / "main" / "failures.csv")
    hist = read_csv(base / "main" / "mstep_q_history.csv")

    n, d = int(proto["n"]), int(proto["d"])
    num_iter = int(proto["num_iter"])
    series = {s: tuple(v) for s, v in proto["series"].items()}
    k_trues, k_cands = proto["k_trues"], proto["k_candidates"]
    n_rep, n_init = int(proto["replicates"]), int(proto["inits"])

    # ── 1. provenance ──────────────────────────────────────────────
    psha = sha256_file(proto_p)
    check("protocol_sha256_matches_runinfo", psha == runinfo["protocol_sha256"],
          protocol_sha256=psha)
    # frozen definition (runner.manifest_hash): sha256 of json.dumps(rows,
    # sort_keys=True) with integer columns as int; re-implemented here
    int_cols = {"k_true", "rep", "k", "t", "data_seed", "init_seed", "mc_seed"}
    man_rows = [{c: (int(v) if c in int_cols else v) for c, v in m.items()} for m in man]
    mh = hashlib.sha256(json.dumps(man_rows, sort_keys=True).encode()).hexdigest()
    check("manifest_rows_hash_matches_protocol", mh == proto["manifest_sha256"],
          manifest_rows_sha256=mh, file_sha256=sha256_file(man_p))
    head = runinfo["git_head_at_start"]
    mism = {p: (h, sha256_git_blob(head, p)) for p, h in proto["code_sha256"].items()}
    bad = {p: v for p, v in mism.items() if v[0] not in v[1].values()}
    how = Counter(next(k for k, x in v[1].items() if x == v[0])
                  for p, v in mism.items() if p not in bad)
    check("code_at_exec_head_matches_frozen_hashes", not bad, head=head,
          matched_as=dict(how), mismatches=bad,
          note="frozen hashes are of working-tree bytes; blob compared raw "
               "and after LF->CRLF (core.autocrlf=true)")
    cur_bad = {p: h for p, h in proto["code_sha256"].items()
               if sha256_file(ROOT / p) != h}
    check("code_in_worktree_unchanged_since_freeze", not cur_bad,
          mismatches=list(cur_bad))
    check("runinfo_conditions", runinfo["workers"] == proto["workers"]
          and runinfo["blas_threads"] == proto["blas_threads"]
          and runinfo["n_fits_planned"] == proto["n_fits"],
          runinfo={k: runinfo[k] for k in ("workers", "blas_threads",
                                           "n_fits_planned", "python", "numpy")})
    ev = [e["event"] for e in events]
    check("run_events_single_start_finish_no_resume",
          ev == ["start", "finished_batch"]
          and events[0]["already_recorded"] == 0
          and events[-1]["recorded_total"] == proto["n_fits"]
          and events[0]["protocol_sha256"] == psha
          and events[0]["git_head"] == head, events=ev)

    # ── 2. manifest vs protocol (design grid, seeds) ───────────────
    sidx = {"O": 0, "A": 1, "B": 2, "C": 3}  # order of SERIES in protocol text
    exp_ids = set()
    seed_bad = 0
    for s in series:
        for r in ("a", "b"):
            for kt in k_trues:
                for rep in range(n_rep):
                    ds = 20261125000 + 100000 * sidx[s] + 1000 * kt + rep
                    for k in k_cands:
                        for t in range(n_init):
                            exp_ids.add((f"{s}-{r}-KT{kt}-r{rep}-k{k:02d}-t{t}",
                                         ds, 10 * ds + t, 10 * ds + 5 + t))
    man_ids = {(m["fit_id"], int(m["data_seed"]), int(m["init_seed"]),
                int(m["mc_seed"])) for m in man}
    check("manifest_equals_protocol_grid_and_seed_rule",
          man_ids == exp_ids and len(man) == proto["n_fits"],
          n_manifest=len(man), n_expected=len(exp_ids))
    by_id = {r["fit_id"]: r for r in rows}
    for m in man:
        r = by_id.get(m["fit_id"])
        if r is None or any(r[c] != m[c] for c in ("data_seed", "init_seed", "mc_seed")):
            seed_bad += 1
    check("per_fit_seeds_equal_manifest", seed_bad == 0, n_bad=seed_bad)

    # ── 3. completion ───────────────────────────────────────────────
    st = Counter(r["status"] for r in rows)
    check("all_8000_recorded_once", len(rows) == len(by_id) == proto["n_fits"]
          and set(by_id) == {m["fit_id"] for m in man}, n_rows=len(rows))
    check("all_status_OK", st == Counter({"OK": proto["n_fits"]}), status=dict(st))
    check("failures_csv_empty", len(fails) == 0, n=len(fails))
    integ = sum(int(r[c]) for r in rows for c in
                ("retry_count", "replacement_count", "seed_rescue_count", "nan_count"))
    check("no_retry_replacement_rescue_nan", integ == 0, total=integ)
    check("family_columns_match_series",
          all((r["family_x"], r["family_y"]) == series[r["series"]] for r in rows))

    # ── 4. iteration count and Q at the last iteration ─────────────
    it = defaultdict(list)
    q_last = {}
    for h in hist:
        it[h["fit_id"]].append(int(h["iteration"]))
        if int(h["iteration"]) == num_iter:
            q_last[h["fit_id"]] = float(h["q_after"])
    it_bad = [f for f in by_id if sorted(it.get(f, [])) != list(range(1, num_iter + 1))]
    check("each_fit_has_exactly_num_iter_iterations", not it_bad,
          num_iter=num_iter, n_bad=len(it_bad))
    q_eq = sum(float(by_id[f]["Q_strict"]) == q_last[f] for f in by_id)
    check("Q_strict_equals_mstep_q_after_at_last_iteration_bitwise",
          q_eq == len(by_id), n_equal=q_eq)
    dec = sum(h["decreased"] == "True" for h in hist)
    checks["mstep_q_decreases_total"] = {"ok": True, "count": dec,
                                         "note": "descriptive; not a gate"}

    # ── 5. generating parameters vs protocol rules ─────────────────
    cal = proto["calibration"]
    par_bad = []
    for r in rows:
        s, rg, kt = r["series"], r["regime"], int(r["k_true"])
        if s == "B":
            w = 0.12 if rg == "a" else 0.12 * math.sqrt(3 / kt)
            w0 = 0.5 * kt * math.log1p(-w * w)
        else:
            w = 1.0 if rg == "a" else math.sqrt(3 / kt)
            w0 = 0.0 if s == "C" else cal[f"{rg}-K{kt}"]["w0"]
        fs = math.sqrt(d / 6) if rg == "a" else math.sqrt(0.5 * d / kt)
        tw, tw0, tf = float(r["true_w"]), float(r["true_w0"]), float(r["true_f_scale"])
        if not (abs(tw - w) <= 1e-12 and abs(tw0 - w0) <= 1e-12 and abs(tf - fs) <= 1e-12):
            par_bad.append(r["fit_id"])
    check("generating_w_w0_fscale_match_protocol_rules", not par_bad,
          n_bad=len(par_bad), tol=1e-12)
    check("poisson_y_abs_w_below_quarter",
          max(float(r["true_w"]) for r in rows if r["series"] == "B") < 0.25)

    # ── 6. C_Q definition, re-implemented ──────────────────────────
    max_abs = 0.0
    n_exact = 0
    p_bad = 0
    cq_ind = {}
    for r in rows:
        k = int(r["k"])
        fx, fy = series[r["series"]]
        p = k * d - k * (k - 1) // 2 + (d if fx == "gaussian" else 0) \
            + (1 if fy == "gaussian" else 0)
        if int(r["num_params"]) != p:
            p_bad += 1
        c = -2.0 * float(r["Q_strict"]) + p * math.log(n)
        cq_ind[r["fit_id"]] = c
        diff = abs(c - float(r["C_Q"]))
        max_abs = max(max_abs, diff)
        n_exact += diff == 0.0
    check("num_params_matches_frozen_formula", p_bad == 0, n_bad=p_bad)
    check("C_Q_equals_minus2Q_plus_p_ln_n", max_abs <= 1e-9, max_abs_diff=max_abs,
          n_bitwise_equal=n_exact)

    # ── 7. data identity within a dataset ──────────────────────────
    dsh = defaultdict(set)
    for r in rows:
        dsh[r["dataset_id"]].add((r["x_hash"], r["y_hash"]))
    check("one_dataset_per_dataset_id",
          len(dsh) == 400 and all(len(v) == 1 for v in dsh.values()),
          n_datasets=len(dsh))

    # ── 8. selection (mean of 2 inits, ties <= 1e-9 -> smallest K) ─
    per = defaultdict(dict)
    for r in rows:
        per[r["dataset_id"]][(int(r["k"]), int(r["t"]))] = float(r["C_Q"])
    sel_rows = []
    for ds, dct in per.items():
        s, rg, kt, rep = ds.split("-")
        kt, rep = int(kt[2:]), int(rep[1:])
        assert len(dct) == len(k_cands) * n_init
        bar = {k: (dct[(k, 0)] + dct[(k, 1)]) / 2.0 for k in k_cands}
        mn = min(bar.values())
        sel = min(k for k in k_cands if bar[k] - mn <= 1e-9)
        ties = sum(1 for k in k_cands if bar[k] - mn <= 1e-9)
        per_init = [min(k_cands, key=lambda k: (dct[(k, t)], k)) for t in range(n_init)]
        srt = sorted(bar.values())
        sel_rows.append(dict(dataset_id=ds, series=s, regime=rg, k_true=kt, rep=rep,
                             selected_k=sel, outcome=("exact" if sel == kt else
                                                      "under" if sel < kt else "over"),
                             n_ties=ties, margin=srt[1] - srt[0],
                             init0_k=per_init[0], init1_k=per_init[1],
                             cqbar_ktrue_minus_min=bar[kt] - mn))
    ana = {r["dataset_id"]: r for r in read_csv(base / "analysis" / "selection_per_dataset.csv")}
    dis = [r["dataset_id"] for r in sel_rows
           if int(ana[r["dataset_id"]]["primary_k"]) != r["selected_k"]
           or int(ana[r["dataset_id"]]["init0_k"]) != r["init0_k"]
           or int(ana[r["dataset_id"]]["init1_k"]) != r["init1_k"]]
    check("selection_matches_prespecified_analysis", not dis, n_disagree=len(dis))
    check("no_ties_within_1e-9", all(r["n_ties"] == 1 for r in sel_rows))

    cnt = defaultdict(Counter)
    for r in sel_rows:
        cnt[r["regime"]][r["outcome"]] += 1
    check("regime_a_200_of_200", cnt["a"]["exact"] == 200, counts=dict(cnt["a"]))
    check("regime_b_176_of_200", cnt["b"]["exact"] == 176, counts=dict(cnt["b"]))
    under = sum(c["under"] for c in cnt.values())
    over = sum(c["over"] for c in cnt.values())
    check("under_24_over_0", under == 24 and over == 0, under=under, over=over)
    check("no_incomplete_dataset", len(per) == 400)

    # ── 9. K_TRUE = 3 duplication between regimes ──────────────────
    dup_bad = 0
    for s in series:
        for rep in range(n_rep):
            for k in k_cands:
                for t in range(n_init):
                    fa = by_id[f"{s}-a-KT3-r{rep}-k{k:02d}-t{t}"]
                    fb = by_id[f"{s}-b-KT3-r{rep}-k{k:02d}-t{t}"]
                    if (fa["C_Q"], fa["x_hash"], fa["y_hash"], fa["true_w"],
                            fa["true_f_scale"]) != (fb["C_Q"], fb["x_hash"],
                                                    fb["y_hash"], fb["true_w"],
                                                    fb["true_f_scale"]):
                        dup_bad += 1
    uniq_b = [r for r in sel_rows if r["regime"] == "b" and r["k_true"] != 3]
    check("K_TRUE3_regime_a_b_identical", dup_bad == 0, n_fit_pairs_differing=dup_bad,
          note="w=1=sqrt(3/3) and sqrt(d/6)=sqrt(0.5d/3): same condition, count once")
    checks["unique_evidence_counts"] = {
        "ok": True,
        "distinct_conditions": 400 - 40,
        "regime_a_exact": f"{cnt['a']['exact']}/200",
        "regime_b_excluding_KTRUE3_exact":
            f"{sum(r['outcome'] == 'exact' for r in uniq_b)}/{len(uniq_b)}",
    }

    # ── 10. state npz references ───────────────────────────────────
    miss = [r["fit_id"] for r in rows
            if not (base / "main" / "states" / r["state_file"]).exists()]
    check("state_npz_present_for_every_fit", not miss, n_missing=len(miss))

    # ── 11. optional regeneration spot check (uses design code) ────
    if a.regen:
        for p in ("expfam/src", "expfam/src/experimental"):
            sys.path.insert(0, str(ROOT / p))
        import k_exp2_design as kd  # noqa: E402
        bt = {(key.split("-")[0], int(c["k"])): c for key, c in cal.items()}
        picks = sorted(dsh)[:: max(1, len(dsh) // a.regen)][: a.regen]
        rg_bad = []
        for ds in picks:
            r0 = next(r for r in rows if r["dataset_id"] == ds)
            spec = kd.dataset_spec(r0["series"], r0["regime"], int(r0["k_true"]),
                                   int(r0["rep"]), bern_table=bt)
            assert spec.seed == int(r0["data_seed"])
            g = kd.generate(spec)
            if (kd.array_hash(g.X), kd.array_hash(g.Y)) != (r0["x_hash"], r0["y_hash"]):
                rg_bad.append(ds)
        check("regeneration_spot_check", not rg_bad, datasets=picks, mismatches=rg_bad)

    with open(out / "independent_selection.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0]))
        w.writeheader()
        for r in sorted(sel_rows, key=lambda r: (r["series"], r["regime"], r["k_true"], r["rep"])):
            w.writerow(r)
    cell = defaultdict(Counter)
    for r in sel_rows:
        cell[(r["series"], r["regime"], r["k_true"])][r["outcome"]] += 1
    with open(out / "independent_cell_counts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "regime", "k_true", "exact", "under", "over"])
        for c in sorted(cell):
            w.writerow([*c, cell[c]["exact"], cell[c]["under"], cell[c]["over"]])
    all_ok = all(v["ok"] for v in checks.values())
    (out / "audit_checks.json").write_text(json.dumps(
        {"all_ok": all_ok, "checks": checks}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    for k, v in checks.items():
        print(("OK  " if v["ok"] else "FAIL"), k,
              {x: y for x, y in v.items() if x != "ok"} or "")
    print("ALL_OK" if all_ok else "SOME_FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
