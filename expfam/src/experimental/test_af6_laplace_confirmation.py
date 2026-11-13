"""Tests for AF6 (no MCEM)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze_af6_laplace_confirmation as an    # noqa: E402
import audit_af6_af5_independent as aud           # noqa: E402
import run_af6_laplace_confirmation as a6         # noqa: E402


def test_protocol_keys_and_seed_disjointness():
    p = a6.protocol()
    assert len(p) == len(set(p))
    assert p["primary_endpoint"]["regime"] == "sparse"
    assert not a6.seed_disjointness()["overlaps"]
    data = [s["data_seed"] for s in a6.seed_manifest()]
    assert len(set(data)) == 40 and not set(data) & set(a6.af2.DATA_SEEDS)
    assert a6.SMOKE_DATA_SEED not in data


def test_masks_same_algorithm_new_seeds():
    _, m = a6.dataset(0)
    old = a6.m1.build_masks(0)
    assert not np.array_equal(m["test"], old["test"])
    assert not np.any(np.triu(m["sparse"] & m["test"], 1))
    assert np.all(m["dense"][m["sparse"]])


def test_cmp2_closed_form_and_derivatives():
    from scipy.special import gammaln
    eta = np.linspace(-3, 4, 9)
    k = np.arange(400)
    brute = np.array([np.logaddexp.reduce(k * e - 2 * gammaln(k + 1)) for e in eta])
    assert np.max(np.abs(aud.cmp2_A(eta) - brute)) < 1e-10
    h = 1e-5
    assert np.max(np.abs((aud.cmp2_A(eta + h) - aud.cmp2_A(eta - h)) / (2 * h) - aud.cmp2_dA(eta))) < 1e-7
    assert np.max(np.abs((aud.cmp2_dA(eta + h) - aud.cmp2_dA(eta - h)) / (2 * h) - aud.cmp2_d2A(eta))) < 1e-7


def test_analysis_keeps_every_column_and_fit_failure_gives_zero(tmp_path):
    reps = [f"rep{r + 1:02d}" for r in range(an.N_DATASETS)]
    fits = pd.DataFrame([{"replicate": r, "regime": g, "status": "FAILED" if (r == "rep02" and g == "sparse") else "OK",
                          "assignment": "|".join(an.TRUE_MODELS)} for r in reps for g in an.REGIMES])
    rows = []
    for r in reps:
        for g in an.REGIMES:
            if r == "rep02" and g == "sparse":
                continue
            for j in an.COUNT_COLS:
                t = an.TRUE_MODELS[j]
                rows.append({"replicate": r, "regime": g, "column": j, "status": "DECIDED",
                             "delta": 1.0 if t == "cmp_nu2p0" else -1.0, "q_status": "SELECTED",
                             "q_selected": "poisson", "af2_auto_selected": "poisson",
                             "delta_jointmap_variant": 1.0, "is_readable": True})
    fits.to_csv(tmp_path / "fit_manifest.csv", index=False)
    pd.DataFrame(rows).to_csv(tmp_path / "column_selection.csv", index=False)
    c = an.frame(tmp_path)
    assert len(c) == 480
    res = an.dr(c, "A", "B", "sparse")
    assert res["datasets"] == 40 and res["n_zero"] == 1 and abs(res["mean_D"] - 39 * 3 / 40) < 1e-12
