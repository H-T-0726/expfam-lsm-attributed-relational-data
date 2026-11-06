"""Descriptive (no fit): information per node implied by the TRUE parameters, for the n=75 / n=150
conditions of this study and the PR #160 E2 Gaussian-X / E3 cells where C_Q under-selected.

Per dataset and node i (truth): X part tr(F' Sigma_X^-1 F), Y part w^2 sum_{j observed} s(1-s) ||z_j||^2,
X signal-to-noise mean_l ||f_l||^2 / sigma_l^2.  Medians over nodes.  This is a description of the
conditions, not a causal test (the conditions differ in several factors at once).
Writes condition_information.csv.  Refuses to overwrite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "reproduction/src", ROOT / "expfam/src/experimental", ROOT / "tools/research_audit"):
    sys.path.insert(0, str(p))

import n75_protocol as npc  # noqa: E402
import run_n75 as rn  # noqa: E402
import run_fair_comparison as rf  # noqa: E402


def info(D, sx2):
    Z, F, w0, w, M = D["Z"], D["F"], D["w0"], D["w"], D["train"]
    n = Z.shape[0]
    xi = float(np.trace(F.T @ np.diag(1 / sx2) @ F))
    eta = w0 + w * Z @ Z.T
    s = 1 / (1 + np.exp(-eta))
    q = s * (1 - s) * M
    np.fill_diagonal(q, 0)
    yi = w ** 2 * (q @ np.sum(Z ** 2, axis=1))
    return dict(n=n, x_info=xi, y_info_median=float(np.median(yi)), y_share_median=float(np.median(yi / (yi + xi + Z.shape[1]))),
                x_snr=float(np.mean(np.sum(F ** 2, axis=1) / sx2)), w_true=float(w), observed_pair_fraction=float(np.triu(M, 1).sum() / (n * (n - 1) / 2)))


def main():
    out = ROOT / npc.RESULTS_REL / "condition_information.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows = []
    for rep in (1, 2, 3, 4):
        D = rn.stage_data("B", rep)
        rows.append(dict(source="this study B", condition="G_canon n=75 all pairs (C_Q 4/4 both estimators)", rep=rep, **info(D, np.full(15, 0.1))))
        D = rn.stage_data("D", rep)
        rows.append(dict(source="this study D", condition="G_canon n=75 80% pairs (C_Q 4/4)", rep=rep, **info(D, np.full(15, 0.1))))
    for rep in (1, 2):
        D = rf.e1_data(1, 5, rep)
        rows.append(dict(source="PR #160 E1", condition="G_canon n=150 all pairs (C_Q 2/2)", rep=rep, **info(D, np.full(15, 0.1))))
    for n in (75, 150):
        for sig in ("weak", "strong"):
            for rep in (1, 2, 3, 4):
                D = rf.e2_data(n, rep, sig, "gaussian")
                rows.append(dict(source="PR #160 E2", condition=f"E2 Gaussian n={n} {sig} 80% pairs", rep=rep,
                                 **info(D, D["gaussian_noise_var"])))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(df.groupby("condition")[["x_snr", "x_info", "y_info_median", "y_share_median", "w_true", "observed_pair_fraction"]].median().round(3))


if __name__ == "__main__":
    main()
