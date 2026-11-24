"""Technical timing smoke for the K-selection Exp2-generalisation budget.

Records ONLY wall time, peak memory proxy and whether the fit finished with
finite parameters.  No criterion value, Q value or selected K is computed or
written, so nothing here can inform a protocol choice by outcome.

Fits: candidate K in {1, 10} on one canonical-clean (matched) Series-O dataset
(K_TRUE = 9), L = 10, 10 EM iterations, for
  E  lineage E  (run_em_experimental, numerics_mode='consistent', fail_fast)
  C  fixed lineage (run_em_fixed_bic from run_fixed_official_exp1_bic_k9_extension,
     imported unchanged; its NaN-retry path is reported, not suppressed)
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("expfam/src", "expfam/src/experimental", "reproduction/src"):
    sys.path.insert(0, str(ROOT / p))

from data_generator_canonical import (  # noqa: E402
    generate_canonical_data, f_scale_for_row_norm, w_for_matched_y_signal)
from em_runner import run_em_experimental  # noqa: E402
import run_fixed_official_exp1_bic_k9_extension as fx  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
K_TRUE = 9
ds = generate_canonical_data(n=150, d=15, k=K_TRUE, seed=77_000_009,
                             family_x="gaussian", family_y="bernoulli",
                             f_scale=f_scale_for_row_norm(0.5, d=15, k=K_TRUE),
                             w0=-1.0, w=w_for_matched_y_signal(1.0, k=K_TRUE, k_ref=3),
                             sigma_x_var=1.0)
rows = []
for lineage in ("E", "C"):
    for k in (1, 10):
        tracemalloc.start(); t0 = time.perf_counter(); status = "OK"; finite = None
        try:
            if lineage == "E":
                r = run_em_experimental(ds.X, ds.Y, "gaussian", "bernoulli", k, L=10,
                                        num_iter=10, numerics_mode="consistent",
                                        failure_policy="fail_fast", init_seed=11, mc_seed=12)
                P = r["params"] if "params" in r else r
                finite = bool(all(np.all(np.isfinite(np.asarray(P[key], float)))
                                  for key in ("F", "w0", "w") if key in P))
            else:
                r = fx.run_em_fixed_bic(ds.X, ds.Y, {"Z": ds.Z, "w0": -1.0, "w": 1.0},
                                        "gaussian", "bernoulli", k, L=10, num_iter=10, seed=11)
                finite = bool(not r.get("nan_occurred", False))
        except Exception as e:  # recorded, not retried
            status = f"ERROR:{type(e).__name__}"
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 2**20; tracemalloc.stop()
        rows.append(dict(lineage=lineage, k=k, seconds=round(dt, 1),
                         peak_traced_mb=round(peak, 1), status=status, finished_finite=finite))
        print(rows[-1], flush=True)
(out / "timing_smoke.json").write_text(json.dumps(
    {"purpose": "runtime/memory only; no criterion computed", "k_true": K_TRUE,
     "L": 10, "num_iter": 10, "rows": rows}, indent=2))
