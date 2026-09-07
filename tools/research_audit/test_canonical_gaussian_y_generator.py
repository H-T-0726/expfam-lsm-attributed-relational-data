"""
Tests for the canonical Gaussian-Y generator (2026-09-07).

These are CODE-PATH tests, not distributional tests. In particular we never
assert that a sample mean is (or is not) close to zero -- that would be a
statistical statement. We assert that the generator's Z is bit-identical to the
raw `default_rng(seed).normal(...)` draw, which fails the moment any
standardization is inserted.

Run:
    python tools/research_audit/test_canonical_gaussian_y_generator.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))

from data_generator_canonical_gaussian_y_k import (  # noqa: E402
    FORBIDDEN_POSTPROCESSING,
    GENERATOR_ID,
    first_latent_draw,
    generate_canonical_gaussian_y,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    print("canonical Gaussian-Y generator tests")
    print("=" * 66)

    n, d, k, seed = 40, 15, 3, 1234

    # ---- T1: Z is EXACTLY the raw rng draw (no post-processing) ------------
    data = generate_canonical_gaussian_y(n=n, d=d, k=k, seed=seed)
    expected_Z = first_latent_draw(n, k, seed)
    check(
        "T1 Z is bit-identical to default_rng(seed).normal(0,1,(n,k))",
        np.array_equal(data["Z"], expected_Z),
        f"max abs diff = {np.max(np.abs(data['Z'] - expected_Z)):.3e}",
    )

    # ---- T2: source contains no forbidden post-processing call -------------
    src = inspect.getsource(generate_canonical_gaussian_y)
    bad = [tok for tok in FORBIDDEN_POSTPROCESSING if tok in src]
    check("T2 no forbidden standardization token in generator body", not bad, f"found {bad}")

    # ---- T3: the historical generator module is not imported --------------
    check(
        "T3 historical data_generator_expfam not imported by this generator",
        "data_generator_expfam" not in src
        and "generate_dual_data" not in src,
        "",
    )

    # ---- T4: shapes -------------------------------------------------------
    check("T4a X shape", data["X"].shape == (n, d), str(data["X"].shape))
    check("T4b Y shape", data["Y"].shape == (n, n), str(data["Y"].shape))
    check("T4c Z shape", data["Z"].shape == (n, k), str(data["Z"].shape))
    check("T4d F shape", data["F"].shape == (d, k), str(data["F"].shape))

    # ---- T5: Y symmetric, zero diagonal ------------------------------------
    check("T5a Y symmetric (exact)", np.array_equal(data["Y"], data["Y"].T))
    check("T5b Y diagonal exactly zero", np.all(np.diag(data["Y"]) == 0.0))

    # ---- T6: X is binary ---------------------------------------------------
    check("T6 X in {0,1}", np.all(np.isin(data["X"], (0.0, 1.0))))

    # ---- T7: F rows all have norm sqrt(1-uniq) -----------------------------
    row_norms = np.linalg.norm(data["F"], axis=1)
    target = np.sqrt(1.0 - data["uniq"])
    check(
        "T7 F row norms == sqrt(1-uniq)",
        np.allclose(row_norms, target, atol=1e-12),
        f"max dev = {np.max(np.abs(row_norms - target)):.3e}",
    )

    # ---- T8: rank not destroyed -------------------------------------------
    check("T8a rank(Z) == k", data["rank_Z"] == k, f"rank={data['rank_Z']}")
    check("T8b rank(F) == k", data["rank_F"] == k, f"rank={data['rank_F']}")

    # ---- T9: same-seed reproducibility -------------------------------------
    data2 = generate_canonical_gaussian_y(n=n, d=d, k=k, seed=seed)
    check(
        "T9 same seed reproduces identical dataset",
        data["dataset_hash_X"] == data2["dataset_hash_X"]
        and data["dataset_hash_Y"] == data2["dataset_hash_Y"],
    )

    # ---- T10: different seed gives different dataset -----------------------
    data3 = generate_canonical_gaussian_y(n=n, d=d, k=k, seed=seed + 1)
    check(
        "T10 different seed gives different dataset",
        data["dataset_hash_Y"] != data3["dataset_hash_Y"],
    )

    # ---- T11: dataset does not depend on candidate K (generator K fixed) ---
    # The runner generates once per (config, trial) and reuses across k_est.
    # Here we assert the generator is a pure function of its own arguments,
    # so the runner's single call is the only source of the dataset.
    check(
        "T11 generator signature has no candidate-K argument",
        "k_est" not in inspect.signature(generate_canonical_gaussian_y).parameters,
    )

    # ---- T12: Y noise scale is a STANDARD DEVIATION ------------------------
    # sigma_y_true=0 must give Y exactly equal to the deterministic mean part.
    d0 = generate_canonical_gaussian_y(n=n, d=d, k=k, seed=seed, sigma_y_true=0.0)
    Zq = d0["Z"]
    eta = d0["w0"] + d0["w"] * (Zq @ Zq.T)
    um = np.triu(np.ones((n, n), dtype=bool), k=1)
    check(
        "T12 sigma_y=0 gives Y == w0 + w z_i^T z_j exactly on upper triangle",
        np.allclose(d0["Y"][um], eta[um], atol=1e-12),
        f"max dev = {np.max(np.abs(d0['Y'][um] - eta[um])):.3e}",
    )

    # ---- T13: w=0 sanity (generator still runs; used only as a guard) ------
    dw0 = generate_canonical_gaussian_y(n=n, d=d, k=k, seed=seed, w_true=0.0)
    check("T13 w=0 generator runs and Y stays finite", np.all(np.isfinite(dw0["Y"])))

    # ---- T14: generator id ------------------------------------------------
    check("T14 generator_id recorded", data["generator_id"] == GENERATOR_ID, GENERATOR_ID)

    # ---- T15: X depends on Z (link is not constant) ------------------------
    # Structural check on the code path: eta_x varies across entries.
    eta_x = data["Z"] @ data["F"].T
    check("T15 eta_x is non-constant", float(eta_x.std()) > 0.0, f"sd={eta_x.std():.4f}")

    print("=" * 66)
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failing checks)")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
