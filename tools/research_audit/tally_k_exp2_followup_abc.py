"""Independent tally for the follow-up experiments A, B (C_Lap numerical sensitivity) and
C (mixed-X high-K MCEM). Does NOT import any runner; standard library + numpy only.

Usage: python tools/research_audit/tally_k_exp2_followup_abc.py [--skip-c]
Writes <base>/followup_independent_tally/tally_checks.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "expfam/results/k_selection/k_exp2_generalization_20261125"
FU = BASE / "clap_followup_20261126"
MX = BASE / "mixedx_highk_20261126"
N, D = 150, 15
LN2PI = math.log(2 * math.pi)


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pick(bar):
    m = min(bar.values())
    return min(k for k, v in bar.items() if v - m <= 1e-9)


def select(f, col, stcol):
    if len(f) != 20 or any(f[(k, t)][stcol] != "OK" for k in range(1, 11) for t in (0, 1)):
        return None
    return pick({k: (float(f[(k, 0)][col]) + float(f[(k, 1)][col])) / 2 for k in range(1, 11)})


def tally_A(checks, notes):
    off = {(r["source"], r["fit_id"]): r for r in rd(BASE / "clap_comparison/run/evaluations.csv")}
    run = rd(FU / "expA/run/per_evaluation.csv")
    checks["A_1400_rows_unique"] = len(run) == 1400 and len({(r["source"], r["fit_id"])
                                                              for r in run}) == 1400
    g = defaultdict(dict)
    for r in run:
        o = off[(r["source"], r["fit_id"])]
        g[(r["source"], r["dataset_id"])][(int(r["k"]), int(r["t"]))] = {
            "sm": r["lap_status"], "cm": r["C_Lap"], "sl": o["lap_status"], "cl": o["C_Lap"],
            "kt": int(r["k_true"]), "cell": f'{r["series"]}-{r["regime"]}-{r["k_true"]}'}
    mine = {}
    for key, f in g.items():
        mine[key] = (select(f, "cl", "sl"), select(f, "cm", "sm"))
    ana = {(r["source"], r["dataset_id"]): r for r in rd(FU / "expA/analysis/selection_by_dataset.csv")}
    lab = lambda k: "INCOMPLETE" if k is None else str(k)            # noqa: E731
    checks["A_selections_equal_analysis"] = all(
        ana[k]["k_Lap_Zlast"] == lab(a) and ana[k]["k_Lap_Zmean"] == lab(b)
        for k, (a, b) in mine.items())
    notes["A_agree_when_both_complete"] = sum(1 for a, b in mine.values() if a and b and a == b)
    notes["A_both_complete"] = sum(1 for a, b in mine.values() if a and b)
    notes["A_status_pairs"] = dict(Counter(f'{v["sl"]}->{v["sm"]}' for f in g.values()
                                           for v in f.values()))


def tally_B(checks, notes):
    run = rd(FU / "expB/run/per_evaluation.csv")
    off = rd(BASE / "clap_comparison/run/evaluations.csv")
    targets = {(r["source"], r["fit_id"]) for r in off if r["lap_status"] != "OK"}
    checks["B_57_targets_exactly"] = {(r["source"], r["fit_id"]) for r in run} == targets
    checks["B_repro200_all_equal"] = all(r["repro200_equal"] == "True" for r in run)
    ns = [r for r in run if r["official_status"] == "NOT_STATIONARY"]
    notes["B_it400_status"] = dict(Counter(r["it400_lap_status"] for r in ns))
    notes["B_converged_k_gt_ktrue"] = sum(1 for r in ns if r["it400_lap_status"] == "OK"
                                          and int(r["k"]) > int(r["k_true"]))
    ok400 = [r for r in ns if r["it400_lap_status"] == "OK"]
    checks["B_converged_rows_stationary"] = all(float(r["it400_lap_grad_inf"]) <= 1e-8 and
                                                float(r["it400_lap_min_eig_H"]) > 0 for r in ok400)
    # B-rule INCOMPLETE count
    fixed = {(r["source"], r["fit_id"]) for r in ok400}
    inc = defaultdict(bool)
    for r in off:
        bad = r["lap_status"] != "OK" and (r["source"], r["fit_id"]) not in fixed
        inc[(r["source"], r["dataset_id"])] |= bad
    notes["B_rule_incomplete"] = dict(Counter(s for (s, _), v in inc.items() if v))
    ana = json.loads((FU / "expB/analysis/checks.json").read_text(encoding="utf-8"))
    checks["B_incomplete_equals_analysis"] = all(
        ana["totals"][s]["Brule"].get("INCOMPLETE", 0) == notes["B_rule_incomplete"].get(s, 0)
        for s in ("main10", "conv40"))


def tally_C(checks, notes):
    rows = rd(MX / "run/per_fit.csv")
    planned = {f"MX-KT{kt}-r{rep}-k{k:02d}-t{t}" for kt in (5, 7, 9) for rep in range(10)
               for k in range(1, 11) for t in (0, 1)}
    ids = [r["fit_id"] for r in rows]
    checks["C_each_recorded_fit_planned_once"] = set(ids) <= planned and len(ids) == len(set(ids))
    notes["C_recorded"] = len(ids)
    notes["C_missing"] = len(planned - set(ids))
    notes["C_fit_status"] = dict(Counter(r["status"] for r in rows))
    ok = [r for r in rows if r["status"] == "OK"]
    worst = {"cq": 0.0, "closure": 0.0, "llz": 0.0, "clap": 0.0}
    for r in ok:
        k = int(r["k"])
        p = k * D - k * (k - 1) // 2 + 5
        q = float(r["Q_strict"])
        worst["cq"] = max(worst["cq"], abs(float(r["C_Q"]) - (-2 * q + p * math.log(N))))
        tot = sum(float(str(r[c]).replace("np.float64(", "").rstrip(")"))
                  for c in ("llZ", "llX_gauss", "llX_bern", "llX_pois", "llY"))
        worst["closure"] = max(worst["closure"], abs(tot - q) / max(1, abs(q)))
        worst["llz"] = max(worst["llz"], abs(float(r["llZ"]) + N * k * (1 + LN2PI) / 2))
        if r["lap_status"] == "OK":
            dk = k * D - k * (k - 1) // 2
            c = float(r["C_Lap"])
            rec = (float(r["lap_fit_mode"]) + float(r["lap_zhat_sq"]) + float(r["lap_logdet_H"])
                   + dk * math.log(N))
            worst["clap"] = max(worst["clap"], abs(c - rec) / max(1, abs(c)))
    notes["C_worst"] = worst
    checks["C_CQ_formula_p_with_5_gaussian_dispersions"] = worst["cq"] <= 1e-6
    checks["C_family_terms_close_to_Q"] = worst["closure"] <= 1e-10
    checks["C_llZ_equals_minus_nK(1+ln2pi)/2"] = worst["llz"] <= 1e-6
    checks["C_CLap_four_term_closure"] = worst["clap"] <= 1e-10
    checks["C_npz_sha256"] = all(sha(MX / "run/states" / r["state_file"]) == r["state_sha256"]
                                 for r in ok)
    g = defaultdict(dict)
    for r in rows:
        g[r["dataset_id"]][(int(r["k"]), int(r["t"]))] = {**r, "fitok": r["status"],
                                                           "lapok": r["lap_status"]
                                                           if r["status"] == "OK" else "FAIL"}
    ana = {r["dataset_id"]: r for r in rd(MX / "analysis/selection_by_dataset.csv")}
    same = True
    cnt = Counter()
    for ds, f in g.items():
        kt = int(next(iter(f.values()))["k_true"])
        kq = select(f, "C_Q", "fitok")
        kl = select(f, "C_Lap", "lapok")
        same &= ana[ds]["k_Q"] == ("INCOMPLETE" if kq is None else str(kq))
        same &= ana[ds]["k_Lap"] == ("INCOMPLETE" if kl is None else str(kl))
        for crit, k in (("Q", kq), ("Lap", kl)):
            cnt[(kt, crit, "INCOMPLETE" if k is None else
                 ("exact" if k == kt else ("under" if k < kt else "over")))] += 1
    checks["C_selection_equals_analysis"] = same
    summ = rd(MX / "analysis/summary_by_cell.csv")
    checks["C_counts_equal_summary"] = all(
        int(s[f"{c}_{o}"]) == cnt[(int(s["k_true"]), c, o)]
        for s in summ for c in ("Q", "Lap") for o in ("exact", "under", "over")) and all(
        int(s[f"{c}_incomplete"]) == cnt[(int(s["k_true"]), c, "INCOMPLETE")]
        for s in summ for c in ("Q", "Lap"))
    notes["C_counts"] = {f"{a}-{b}-{c}": v for (a, b, c), v in sorted(cnt.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-c", action="store_true")
    a = ap.parse_args()
    out = FU / "followup_independent_tally"
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    checks, notes = {}, {}
    tally_A(checks, notes)
    tally_B(checks, notes)
    if not a.skip_c:
        tally_C(checks, notes)
    official = BASE / "clap_comparison"
    proto = json.loads((official / "protocol/protocol.json").read_text(encoding="utf-8"))
    checks["official_6600_run_unchanged"] = sha(official / "protocol/protocol.json") == \
        "ae62f0413ec0b2ad5196bdd945847808fe3defe2f687fcbcd0dc103050f53432" and all(
        sha(ROOT / p) == proto["input_sha256"][k] for k, p in {
            "main_per_fit": BASE / "main/per_fit.csv",
            "conv_per_fit": BASE / "convergence_sensitivity/run/per_fit.csv"}.items())
    fa = json.loads((FU / "expA/protocol/protocol.json").read_text(encoding="utf-8"))
    checks["official_evaluations_unchanged"] = sha(official / "run/evaluations.csv") == \
        fa["official_run"]["evaluations_sha256"]
    out.mkdir(parents=True)
    res = {"checks": checks, "all_ok": all(checks.values()), "notes": notes}
    (out / "tally_checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
