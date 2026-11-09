"""Independent artifact consistency checks for MX2 (no model imports or fits)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "expfam" / "results" / "misspecification_extended" / "phase_20261110"


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check():
    manifest, fits = read("manifest.csv"), read("fit_results.csv")
    pairs, summary = read("paired_rmse_z.csv"), read("summary_by_condition.csv")
    dataset_means = read("dataset_mean_delta.csv")
    by_key = {r["experiment_id"]: r for r in fits}
    assert len(manifest) == 1280
    assert len({r["experiment_id"] for r in manifest}) == 1280
    assert len(by_key) == len(fits)
    assert set(by_key) <= {r["experiment_id"] for r in manifest}
    assert len(pairs) == 1120
    assert len(dataset_means) == 560
    assert sum(r["source"] == "E1/full" for r in fits) == 160
    for row in fits:
        if row["status"] == "OK":
            for field in ("rmse_mode", "rmse_last", "rmse_mean5", "gram_corr",
                          "norm_ratio", "w0_hat", "w_hat"):
                assert math.isfinite(float(row[field])), (row["experiment_id"], field)
            assert row["mode_status"] == "OK"
            assert float(row["mode_grad_inf"]) <= 1e-8
            assert float(row["mode_min_eig_H"]) > 0
            assert row["nan_occurred"] in ("False", "0")
            assert int(row["retry_count"]) == 0
            assert int(row["replacement_count"]) == 0
            assert int(row["seed_rescue_count"]) == 0
    for pair in pairs:
        if pair["valid"] == "True":
            expected = float(pair["rmse_wrong"]) - float(pair["rmse_correct"])
            assert math.isclose(float(pair["delta_rmse"]), expected, abs_tol=1e-12)
            assert math.isclose(float(pair["relative_change"]),
                                expected / float(pair["rmse_correct"]), abs_tol=1e-12)
            wrong = by_key[pair["experiment_id"]]
            baseline_key = pair["experiment_id"].rsplit("-", 1)[0] + "-correct"
            baseline = by_key[baseline_key]
            assert math.isclose(float(pair["rmse_wrong"]), float(wrong["rmse_mode"]),
                                abs_tol=1e-12)
            assert math.isclose(float(pair["rmse_correct"]), float(baseline["rmse_mode"]),
                                abs_tol=1e-12)
    for dataset in dataset_means:
        pp = [r for r in pairs if r["condition_id"] == dataset["condition_id"] and
              r["replicate"] == dataset["replicate"] and r["valid"] == "True"]
        assert int(dataset["valid_seed_count"]) == len(pp)
        if len(pp) == 2:
            assert math.isclose(float(dataset["mean_delta_two_seeds"]),
                                np.mean([float(r["delta_rmse"]) for r in pp]),
                                abs_tol=1e-12)
    by_condition = {r["condition_id"]: r for r in summary}
    for condition, row in by_condition.items():
        ff = [r for r in fits if r["condition_id"] == condition]
        pp = [r for r in pairs if r["condition_id"] == condition and r["valid"] == "True"]
        assert int(row["attempted_fits"]) == len(ff)
        assert int(row["successful_fits"]) == sum(r["status"] == "OK" for r in ff)
        assert int(row["failed_fits"]) == sum(r["status"] != "OK" for r in ff)
        assert int(row["valid_pairs"]) == len(pp)
        if pp:
            values = np.asarray([float(r["delta_rmse"]) for r in pp])
            assert math.isclose(float(row["delta_mean"]), float(values.mean()),
                                abs_tol=1e-12)
            assert math.isclose(float(row["delta_median"]), float(np.median(values)),
                                abs_tol=1e-12)
    state = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert state["attempted"] == len(fits)
    assert state["planned"] == len(manifest)
    assert state["remaining"] == len(manifest) - len(fits)
    if state["completed"]:
        for index in range(1, 7):
            images = list((OUT / "figures").glob(f"fig{index}_*.png"))
            assert len(images) == 1 and images[0].stat().st_size > 10_000
            image = imread(images[0])
            assert image.shape[0] >= 900 and image.shape[1] >= 1200
        report = ROOT / "reports" / "misspecification_extended" / "results_20261110.md"
        assert report.exists() and report.stat().st_size > 2000
    result = {"verdict": "PASS", "planned": len(manifest), "attempted": len(fits),
              "valid_pairs": sum(r["valid"] == "True" for r in pairs),
              "failed": sum(r["status"] != "OK" for r in fits)}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    check()
