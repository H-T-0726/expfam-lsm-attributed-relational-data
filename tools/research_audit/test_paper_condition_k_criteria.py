"""Zero-EM checks for Issue #128 experiment B (paper-type generator)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paper_condition_k_criteria as b                       # noqa: E402


def test_seed_rules_unique_and_rep1_is_historical():
    reps = b.REPLICATES + b.EXTENSION_REPLICATES
    ms = [b.model_seed(t, r, k, s) for t in b.K_TRUE_GRID for r in reps
          for k in b.CANDIDATE_K for s in b.STARTS]
    ds = [b.data_seed(t, r) for t in b.K_TRUE_GRID for r in reps]
    assert len(ms) == len(set(ms)) and len(ds) == len(set(ds))
    assert set(ms).isdisjoint(ds)
    for t in b.K_TRUE_GRID:
        assert b.data_seed(t, 1) == 1980 + 100 * t     # experiment_paper_2.py


def test_extension_replicates_are_fixed_but_not_in_this_run():
    assert b.REPLICATES == (1, 2, 3, 4, 5)
    assert set(b.REPLICATES).isdisjoint(b.EXTENSION_REPLICATES)


def test_parameter_count_is_the_papers():
    for k in b.CANDIDATE_K:
        assert b.num_params(k) == (k + 1) * b.D - k * (k - 1) // 2


def test_generator_is_deterministic_and_unchanged_defaults():
    x1 = b.make_dataset(3, 1)
    x2 = b.make_dataset(3, 1)
    assert x1["provenance"] == x2["provenance"]
    assert b.GENERATOR_KWARGS == {"var_f": 5.0, "uniq": 0.1, "w_scale": 3.0}
    assert x1["X"].shape == (b.N, b.D)


def test_protocol_marks_not_a_reproduction():
    p = b.frozen_protocol()
    assert p["is_paper_experiment2_reproduction"] is False
    assert p["manuscript_adoption_status"] == "NOT_DECIDED_BY_HUMAN"
    assert p["heldout"] is False


def _rows(fail=None, lap_bad=None):
    rows = []
    for k in b.CANDIDATE_K:
        for s in b.STARTS:
            v = float((k - 3) ** 2)
            r = {"k_true": 3, "replicate": 1, "k_est": k, "start": s,
                 "fit_status": "FAILED" if (k, s) == fail else "OK",
                 "lap_status": "NOT_STATIONARY" if (k, s) == lap_bad else "OK",
                 "C_Q": v, "C_Lap": math.nan if (k, s) == lap_bad else v + 1,
                 "C_cond_MC": -k, "C_cond_last": -k,
                 "C_cond_mode": math.nan if (k, s) == lap_bad else -k,
                 "D_K": -10.0 * k, "P_Z": 5.0 * k, "P_theta": 2.0 * k,
                 "lap_fit_mode": -9.0 * k, "lap_volume": 4.0 * k,
                 "lap_penalty": 2.0 * k}
            rows.append(r)
    return rows


def test_analyse_selection_and_incomplete_rules():
    sel = {r["criterion"]: r for r in b.analyse(_rows(), [(3, 1)])["selection"]}
    assert sel["C_Q"]["selected_k"] == 3 and sel["C_Lap"]["selected_k"] == 3
    assert sel["C_cond_MC"]["selected_k"] == 10
    sel = {r["criterion"]: r for r in
           b.analyse(_rows(lap_bad=(9, 1)), [(3, 1)])["selection"]}
    assert sel["C_Lap"]["status"] == "INCOMPLETE"
    assert sel["C_cond_mode"]["status"] == "INCOMPLETE"
    assert sel["C_Q"]["status"] == "COMPLETE"
    out = b.analyse(_rows(fail=(2, 2)), [(3, 1)])
    assert all(r["status"] == "INCOMPLETE" for r in out["selection"])
    assert out["decomposition"] == []


def test_analyse_rejects_missing_fits():
    with pytest.raises(b.BStop):
        b.analyse(_rows()[:-1], [(3, 1)])
