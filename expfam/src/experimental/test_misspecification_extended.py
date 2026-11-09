"""Contract tests for the frozen MX2 design; no MCEM fitting."""
import csv

import numpy as np

import run_misspecification_m1 as m1
import run_misspecification_extended as mx


def test_plan_counts_unique_and_shared_baseline():
    rows = mx.plan()
    assert len(rows) == 1280
    assert len({mx.key(x) for x in rows}) == 1280
    assert sum(x["arm"] == "correct" for x in rows) == 160
    assert sum(x["arm"] in ("correct", "bp6") and x["y_rate"] in (1.0, .1)
               for x in rows) == 160
    assert all(int(x["model_seed"]) in (
        1343000 + int(x["replicate"][3:]) - 1,
        1344000 + int(x["replicate"][3:]) - 1) for x in rows)


def test_nested_masks_reproduce_m1():
    for r in (0, 7, 19):
        mm, old = mx.masks(r), m1.build_masks(r)
        assert np.array_equal(mm[1.0], old["dense"])
        assert np.array_equal(mm[.1], old["sparse"])
        assert np.array_equal(mm["test"], old["test"])
        assert [m1.n_upper(mm[v]) for v in (.1, .25, .5, 1.0)] == [250, 624, 1248, 2497]
        assert all(np.all(mm[b][mm[a]]) for a, b in ((.1, .25), (.25, .5), (.5, 1.0)))
        assert not np.any(mm["test"] & mm[1.0])


def test_only_selected_columns_change_and_are_nested():
    for r in (0, 7, 19):
        for family, arms, counts in (
            ("b", ("bp2", "bp4", "bp6"), (2, 4, 6)),
            ("b", ("bg2", "bg4", "bg6"), (2, 4, 6)),
            ("p", ("pg1", "pg2", "pg3"), (1, 2, 3)),
        ):
            previous = set()
            for arm, count in zip(arms, counts):
                fitted, cols, _ = mx.assignment(r, arm)
                assert len(cols) == count and previous <= set(cols)
                assert {i for i, (a, b) in enumerate(zip(m1.FAMILY_X_TRUE, fitted))
                        if a != b} == set(cols)
                previous = set(cols)
        assert mx.assignment(r, "correct")[0] == list(m1.FAMILY_X_TRUE)
        assert mx.assignment(r, "allg")[0] == ["gaussian"] * 12


def test_preflight_source_hashes_and_support():
    provenance = mx.preflight(mx.plan())
    assert len(provenance) == 20
    assert all(row["train_0.1_pairs"] == 250 for row in provenance)


def test_prepared_csv_keys_and_reuse():
    rows = mx.read_csv(mx.OUT / "manifest.csv")
    fits = mx.read_csv(mx.OUT / "fit_results.csv")
    assert len(rows) == 1280 and len(fits) >= 160
    assert len({x["experiment_id"] for x in rows}) == len(rows)
    assert len({x["experiment_id"] for x in fits}) == len(fits)
    assert sum(x["source"] == "E1/full" for x in fits) == 160
    assert all(x["status"] == "OK" for x in fits if x["source"] == "E1/full")
