"""Exact three-term split of the recorded C_Q (no MCEM, no new fits, no new selection).

Written 2026-11-26, AFTER the 8,000-fit main run and the 600-fit convergence-
sensitivity run (post-hoc arithmetic on recorded CSVs; exploratory/descriptive).

C_Q(K) = -2 Q_strict + p ln n and, for lineage E (em_runner.run_em_experimental):
  Q_strict = (1/L) sum_l [ln p(Z_l) + ln p(X|Z_l) + ln p(Y|Z_l)] + strict constants
  ln p(Z_l) = -(nK/2) ln(2 pi var_z) - ||Z_l||^2 / (2 var_z),  var_z = 1
  Z_samples are scale_Z'd so that mean(Z^2) over all (n, K, L) entries is 1,
  hence (1/L) sum_l ||Z_l||^2 = nK and the prior part is exactly
  -(nK/2)(1 + ln 2 pi)  (independent of the data).
Therefore, with P_Z(K) = nK(1 + ln 2 pi) and PEN(K) = p(K) ln n,
  D(K) := C_Q(K) - P_Z(K) - PEN(K) = -2 (1/L) sum_l [ln p(X|Z_l) + ln p(Y|Z_l)] - 2 const
is the MC-averaged observed-data term (X and Y together). The split of D into an
X part and a Y part needs the L Z-samples, which were NOT saved (NPZ holds only
Z_last and Z_mean), so it is not computed here. Up to floating-point rounding of
scale_Z the three-term split is exact.

The selection itself is taken from the existing pre-specified analyses; this
script only re-expresses the recorded C_Q. "C_Q without P_Z" is an arithmetic
counterfactual, NOT a criterion.

Usage: python tools/research_audit/decompose_cq_terms_k_exp2.py --out OUT_DIR
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "expfam/results/k_selection/k_exp2_generalization_20261125"
N, D_DIM = 150, 15
PZ_PER_DIM = N * (1.0 + math.log(2.0 * math.pi))
LN_N = math.log(N)


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect():
    """Yield (source, checkpoint, dataset_id, cell, k_true, k, t, C_Q, p)."""
    for r in rd(BASE / "main/per_fit.csv"):
        if r["status"] != "OK" or int(r["k_true"]) < 7:
            continue
        yield ("main", 10, r["dataset_id"], f'{r["series"]}-{r["regime"]}-{r["k_true"]}',
               int(r["k_true"]), int(r["k"]), int(r["t"]), float(r["C_Q"]),
               int(r["num_params"]))
    for r in rd(BASE / "convergence_sensitivity/run/checkpoint_scores.csv"):
        yield ("convsens", int(r["checkpoint"]), r["dataset_id"],
               f'{r["series"]}-{r["regime"]}-{r["k_true"]}', int(r["k_true"]), int(r["k"]),
               int(r["t"]), float(r["C_Q"]), int(r["num_params"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = ROOT / ap.parse_args().out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    fits = defaultdict(dict)   # (src, cp, ds) -> {(k, t): (C_Q, p)}
    meta = {}
    for src, cp, ds, cell, kt, k, t, cq, p in collect():
        fits[(src, cp, ds)][(k, t)] = (cq, p)
        meta[(src, cp, ds)] = (cell, kt)
    per_k, per_ds = [], []
    for key in sorted(fits):
        src, cp, ds = key
        cell, kt = meta[key]
        f = fits[key]
        if len(f) != 20:
            raise SystemExit(f"STOP: incomplete {key}")
        bar = {}
        for k in range(1, 11):
            cq = (f[(k, 0)][0] + f[(k, 1)][0]) / 2.0
            p = f[(k, 0)][1]
            pz, pen = PZ_PER_DIM * k, p * LN_N
            bar[k] = (cq, pz, pen, cq - pz - pen)
            per_k.append(dict(source=src, checkpoint=cp, dataset_id=ds, cell=cell,
                              k_true=kt, k=k, C_Q_mean=cq, P_Z=pz, PEN=pen, D=cq - pz - pen))
        m = min(v[0] for v in bar.values())
        ksel = min(k for k, v in bar.items() if v[0] - m <= 1e-9)
        a, b = bar[kt], bar[ksel]
        per_ds.append(dict(
            source=src, checkpoint=cp, dataset_id=ds, cell=cell, k_true=kt, selected_k=ksel,
            gap_CQ=a[0] - b[0], dP_Z=a[1] - b[1], dPEN=a[2] - b[2], dD=a[3] - b[3],
            # arithmetic counterfactual only (not a criterion)
            k_true_beats_selected_without_P_Z=(a[3] + a[2]) < (b[3] + b[2])))
    out.mkdir(parents=True)
    for name, rows in (("per_k_terms.csv", per_k), ("selected_vs_ktrue_terms.csv", per_ds)):
        with open(out / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # per-step medians: dD(K->K+1) vs the K-only cost P_Z step + PEN step
    steps = defaultdict(list)
    idx = {(r["source"], r["checkpoint"], r["dataset_id"], r["k"]): r for r in per_k}
    for r in per_k:
        if r["k"] == 10:
            continue
        nxt = idx[(r["source"], r["checkpoint"], r["dataset_id"], r["k"] + 1)]
        steps[(r["source"], r["checkpoint"], r["cell"], r["k"])].append(nxt["D"] - r["D"])
    with open(out / "step_medians.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "checkpoint", "cell", "k_from", "k_to", "median_dD",
                    "min_dD", "max_dD", "P_Z_step", "PEN_step", "K_only_cost"])
        for (src, cp, cell, k), v in sorted(steps.items()):
            pen_step = (D_DIM - k) * LN_N
            w.writerow([src, cp, cell, k, k + 1, st.median(v), min(v), max(v), PZ_PER_DIM,
                        pen_step, PZ_PER_DIM + pen_step])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
