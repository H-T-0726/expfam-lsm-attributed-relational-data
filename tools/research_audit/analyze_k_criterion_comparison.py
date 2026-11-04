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


def b_figures(b_dir: Path, fig_dir: Path) -> None:
    """B_selected_vs_true.png and B_boundary_slack.png (descriptive)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                              # noqa: PLC0415

    fig_dir.mkdir(parents=True, exist_ok=True)
    for name in ("B_selected_vs_true.png", "B_boundary_slack.png"):
        if (fig_dir / name).exists():
            raise SystemExit(f"{fig_dir / name} exists; not overwriting")
    sel = read(b_dir / "selection.csv")
    dec = read(b_dir / "increment_decomposition.csv")
    crits = ["C_Q", "C_Lap", "C_cond_MC", "C_cond_last", "C_cond_mode"]
    color = {"C_Q": "#2a78d6", "C_Lap": "#eb6834", "C_cond_MC": "#1baf7a",
             "C_cond_last": "#eda100", "C_cond_mode": "#e87ba4"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    offs = {c: (i - 2) * 0.13 for i, c in enumerate(crits)}
    for c in crits:
        pts = defaultdict(int)
        inc = defaultdict(int)
        for r in sel:
            if r["criterion"] != c:
                continue
            t = int(r["k_true"])
            if r["status"] == "COMPLETE":
                pts[(t, int(r["selected_k"]))] += 1
            else:
                inc[t] += 1
        xs = [t + offs[c] for (t, _k) in pts]
        ys = [k for (_t, k) in pts]
        ss = [18 * v for v in pts.values()]
        ax.scatter(xs, ys, s=ss, color=color[c], alpha=0.8, label=c,
                   edgecolor="white", linewidth=0.5)
        for t, v in inc.items():
            ax.text(t + offs[c], 0.35, str(v), color=color[c], fontsize=7,
                    ha="center")
    ax.plot([0.5, 9.5], [0.5, 9.5], color="#6b6b6b", lw=0.8, ls="--")
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_yticks(range(1, 11))
    ax.set_ylim(0, 10.6)
    ax.set_xlabel("K_TRUE")
    ax.set_ylabel("selected K (marker area = # datasets of 5)")
    ax.set_title("B: selected K vs K_TRUE (numbers at bottom = INCOMPLETE)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "B_selected_vs_true.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=False)
    for ax, side, title in ((axes[0], "below", "K_TRUE-1 -> K_TRUE (>0 favours K_TRUE)"),
                            (axes[1], "above", "K_TRUE -> K_TRUE+1 (<0 favours K_TRUE)")):
        for c, col, off in (("CQ_slack", "#2a78d6", -0.15),
                            ("Lap_slack", "#eb6834", 0.15)):
            for x in dec:
                t, kf = int(x["k_true"]), int(x["k_from"])
                if (side == "below" and kf != t - 1) or (side == "above"
                                                         and kf != t):
                    continue
                if x[c] in ("", "nan"):
                    continue
                ax.scatter(t + off, float(x[c]), color=col, s=14, alpha=0.8)
        ax.axhline(0, color="#2b2b2b", lw=0.8)
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_xlabel("K_TRUE")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("slack = fit gain - Δpenalty terms")
    axes[0].scatter([], [], color="#2a78d6", label="C_Q")
    axes[0].scatter([], [], color="#eb6834", label="C_Lap")
    axes[0].legend(fontsize=7)
    fig.suptitle("B: arithmetic split at the K_TRUE boundary (mean of 2 starts;"
                 " not a causal decomposition)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "B_boundary_slack.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", type=Path)
    ap.add_argument("--b-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path)
    a = ap.parse_args(argv)
    if a.fig_dir and a.b_dir:
        b_figures(a.b_dir, a.fig_dir)
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
