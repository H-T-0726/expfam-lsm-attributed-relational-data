"""Issue #128 derived tables for the report, from committed CSV artifacts only.

Read-only with respect to every run directory.  Writes only into a NEW output
directory (refuses to overwrite an existing file).  Descriptive counts of fixed
finite conditions; no rate, probability or general performance is implied.
Lineage E, MANUSCRIPT_ADOPTION_STATUS NOT_DECIDED_BY_HUMAN.

Tables:
  A_selection_by_cell.csv        A1: selected K per (criterion, K_TRUE, n)
  A_disagreements.csv            A1: datasets where C_Q and C_Lap differ
  A_lap_incomplete.csv           A1: the non-OK Laplace fits
  B_selection_by_k_true.csv      B: exact / under / over / incomplete counts
  B_selected_k_distribution.csv  B: histogram of selected K per K_TRUE
  B_disagreements.csv            B: datasets where C_Q and C_Lap differ
  B_lap_nonok.csv                B: the non-OK Laplace / FAILED fits
  B_boundary_terms.csv           B: C_Q / C_Lap increments at K_TRUE-1 -> K_TRUE
                                    and K_TRUE -> K_TRUE+1 (arithmetic split)
  B_start_agreement.csv          B: per-start selections vs mean-of-starts

Usage:
    python analyze_k_criterion_comparison.py --a-dir <A full> --b-dir <B full>
                                             --out <new dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(p: Path) -> list[dict[str, str]]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(out: Path, name: str, rows: list[dict]) -> None:
    path = out / name
    if path.exists():
        raise SystemExit(f"{path} exists; not overwriting")
    cols: list[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def a_tables(a_dir: Path, out: Path) -> None:
    sel = read(a_dir / "selection.csv")
    fits = read(a_dir / "fit_criteria.csv")
    by = defaultdict(list)
    for r in sel:
        by[(r["criterion"], int(r["k_true"]), int(r["n"]))].append(r)
    rows = []
    for (c, t, n), rs in sorted(by.items()):
        rs.sort(key=lambda r: int(r["replicate"]))
        rows.append({"criterion": c, "k_true": t, "n": n, "datasets": len(rs),
                     "exact": sum(r["label"] == "exact" for r in rs),
                     "under": sum(r["label"] == "under" for r in rs),
                     "over": sum(r["label"] == "over" for r in rs),
                     "incomplete": sum(r["status"] != "COMPLETE" for r in rs),
                     "selected_k": " ".join(r["selected_k"] or "inc"
                                            for r in rs)})
    write(out, "A_selection_by_cell.csv", rows)
    m = {(r["criterion"], int(r["k_true"]), int(r["n"]), int(r["replicate"])):
         r for r in sel}
    dis = []
    for (c, t, n, rep), r in sorted(m.items()):
        if c != "C_Q":
            continue
        lap = m[("C_Lap", t, n, rep)]
        if lap["selected_k"] != r["selected_k"]:
            dis.append({"k_true": t, "n": n, "replicate": rep,
                        "C_Q": r["selected_k"],
                        "C_Lap": lap["selected_k"] or "INCOMPLETE",
                        "S1_reference": m[("S1", t, n, rep)]["selected_k"]})
    write(out, "A_disagreements.csv", dis)
    write(out, "A_lap_incomplete.csv", [
        {k: x[k] for k in ("k_true", "n", "replicate", "k_est", "start",
                           "lap_status", "lap_grad_inf", "lap_iterations",
                           "lap_min_eig_H", "lap_notes")}
        for x in fits if x["lap_status"] != "OK"])


def b_tables(b_dir: Path, out: Path) -> None:
    sel = read(b_dir / "selection.csv")
    fits = read(b_dir / "fit_criteria.csv")
    dec = read(b_dir / "increment_decomposition.csv")
    crits = list(dict.fromkeys(r["criterion"] for r in sel))
    k_trues = sorted({int(r["k_true"]) for r in sel})
    rows, dist = [], []
    for c in crits:
        for t in k_trues:
            rs = sorted((r for r in sel if r["criterion"] == c
                         and int(r["k_true"]) == t),
                        key=lambda r: int(r["replicate"]))
            rows.append({"criterion": c, "k_true": t, "datasets": len(rs),
                         "exact": sum(r["label"] == "exact" for r in rs),
                         "under": sum(r["label"] == "under" for r in rs),
                         "over": sum(r["label"] == "over" for r in rs),
                         "incomplete": sum(r["status"] != "COMPLETE"
                                           for r in rs),
                         "selected_k": " ".join(r["selected_k"] or "inc"
                                                for r in rs)})
            cnt = Counter(r["selected_k"] or "inc" for r in rs)
            dist.append({"criterion": c, "k_true": t,
                         **{f"K{k}": cnt.get(str(k), 0) for k in range(1, 11)},
                         "inc": cnt.get("inc", 0)})
    write(out, "B_selection_by_k_true.csv", rows)
    write(out, "B_selected_k_distribution.csv", dist)
    m = {(r["criterion"], int(r["k_true"]), int(r["replicate"])): r
         for r in sel}
    dis = []
    for (c, t, rep), r in sorted(m.items()):
        if c != "C_Q":
            continue
        lap = m[("C_Lap", t, rep)]
        if lap["selected_k"] != r["selected_k"]:
            dis.append({"k_true": t, "replicate": rep, "C_Q": r["selected_k"],
                        "C_Lap": lap["selected_k"] or "INCOMPLETE",
                        **{cc: m[(cc, t, rep)]["selected_k"] or "INCOMPLETE"
                           for cc in crits if cc.startswith("C_cond")}})
    write(out, "B_disagreements.csv", dis)
    write(out, "B_lap_nonok.csv", [
        {k: x.get(k, "") for k in ("k_true", "replicate", "k_est", "start",
                                   "fit_status", "failure", "lap_status",
                                   "lap_grad_inf", "lap_iterations",
                                   "lap_min_eig_H", "lap_notes")}
        for x in fits if x["fit_status"] != "OK" or x["lap_status"] != "OK"])
    bnd = []
    for x in dec:
        t, kf = int(x["k_true"]), int(x["k_from"])
        if kf in (t - 1, t):
            bnd.append({"k_true": t, "replicate": int(x["replicate"]),
                        "step": f"{kf}->{kf + 1}",
                        "side": "below_true" if kf == t - 1 else "above_true",
                        **{k: x[k] for k in (
                            "CQ_fit_gain", "CQ_dPZ", "CQ_dPtheta", "CQ_slack",
                            "Lap_fit_gain", "Lap_dvolume", "Lap_dpenalty",
                            "Lap_slack", "Lap_complete")}})
    write(out, "B_boundary_terms.csv", bnd)
    agree = []
    for (c, t, rep), r in sorted(m.items()):
        if r["status"] != "COMPLETE":
            continue
        agree.append({"criterion": c, "k_true": t, "replicate": rep,
                      "selected_mean": r["selected_k"],
                      "selected_start1": r["selected_start1"],
                      "selected_start2": r["selected_start2"],
                      "starts_agree": r["selected_start1"]
                      == r["selected_start2"],
                      "margin": r["margin"]})
    write(out, "B_start_agreement.csv", agree)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", type=Path)
    ap.add_argument("--b-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {}
    if a.a_dir:
        a_tables(a.a_dir, a.out)
        meta["a_dir"] = a.a_dir.as_posix()
    if a.b_dir:
        b_tables(a.b_dir, a.out)
        meta["b_dir"] = a.b_dir.as_posix()
    p = a.out / "analysis_inputs.json"
    if p.exists():
        raise SystemExit(f"{p} exists; not overwriting")
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
