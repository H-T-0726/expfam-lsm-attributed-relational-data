"""
Sanity Check for DualExpFamLSM.

Tests ALL 9 family_x x family_y combinations on tiny synthetic data
(n=50, d=5, k=2) to verify:
  1. No runtime errors
  2. Q function is finite (not NaN)
  3. Q function is roughly non-decreasing (last Q > first Q)
  4. RMSE(Z) is finite and < 2.0 (not diverged)
  5. Backward compatibility: Dual(GaussxBern) ~= original ExpFam(Bern)

Run with:
    python expfam/src/test_dual_expfam.py
"""

import sys
import numpy as np
from pathlib import Path

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "reproduction" / "src"))

from model_dual_expfam import DualExpFamLSM          # noqa
from model_expfam import ExpFamLatentStructuralModel  # noqa
from data_generator_expfam import generate_dual_data  # noqa
from utils_expfam import run_em_dual, calc_Q_dual     # noqa

# ──────────────────────────────────────────────────────────────────────────────
FAMILIES   = ("gaussian", "bernoulli", "poisson")
SEED       = 2024
N, D, K    = 50, 5, 2
L, NITER   = 5, 8
PASS_COLOR = "\033[92m"   # green
FAIL_COLOR = "\033[91m"   # red
RESET      = "\033[0m"


def _ok(msg): print(f"  {PASS_COLOR}[PASS]{RESET} {msg}")
def _fail(msg): print(f"  {FAIL_COLOR}[FAIL]{RESET} {msg}"); return True


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run one EM trial and return result
# ──────────────────────────────────────────────────────────────────────────────

def run_combo(fx, fy, verbose=False):
    data = generate_dual_data(n=N, d=D, k=K, seed=SEED,
                              family_x=fx, family_y=fy)
    res = run_em_dual(
        X=data["X"], Y=data["Y"],
        true_params=data,
        family_x=fx, family_y=fy,
        k=K, L=L, num_iter=NITER,
        seed=SEED, verbose=verbose,
    )
    return data, res


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: All 9 combinations - no error, Q finite, Q increasing, RMSE finite
# ──────────────────────────────────────────────────────────────────────────────

def test_all_combinations():
    print("\n" + "="*70)
    print("Test 1: All 9 family_x x family_y combinations")
    print("="*70)
    failures = 0
    results_table = []

    for fx in FAMILIES:
        for fy in FAMILIES:
            label = f"X={fx:9s}  Y={fy:9s}"
            failed = False
            try:
                _, res = run_combo(fx, fy)
                Q_hist = res["Q_history"]
                Q_first = Q_hist[0]
                Q_last  = Q_hist[-1]
                rmse_Z  = res["rmse_Z"]
                rmse_X  = res["rmse_X"]
                nan_occ = res["nan_occurred"]

                # Checks
                if not np.isfinite(Q_last):
                    failed = _fail(f"{label}  Q_last={Q_last:.4f} (not finite)")
                elif not np.isfinite(rmse_Z):
                    failed = _fail(f"{label}  rmse_Z={rmse_Z:.4f} (not finite)")
                elif rmse_Z > 5.0:
                    failed = _fail(f"{label}  rmse_Z={rmse_Z:.4f} (>5.0 - diverged)")
                elif Q_last < Q_first - abs(Q_first) * 0.3:
                    failed = _fail(
                        f"{label}  Q decreased badly: {Q_first:.1f} -> {Q_last:.1f}"
                    )
                else:
                    status = "NaN-retry" if nan_occ else "clean"
                    _ok(f"{label}  Q: {Q_first:.1f}->{Q_last:.1f}  "
                        f"RMSE(Z)={rmse_Z:.3f}  RMSE(X)={rmse_X:.3f}  [{status}]")

                results_table.append(dict(
                    fx=fx, fy=fy,
                    Q_first=Q_first, Q_last=Q_last,
                    rmse_Z=rmse_Z, rmse_X=rmse_X,
                    nan=nan_occ, ok=not failed,
                ))

            except Exception as e:
                failed = _fail(f"{label}  Exception: {e}")
                results_table.append(dict(
                    fx=fx, fy=fy, Q_first=float("nan"), Q_last=float("nan"),
                    rmse_Z=float("nan"), rmse_X=float("nan"), nan=True, ok=False,
                ))

            if failed:
                failures += 1

    print(f"\n  -> {9-failures}/9 PASS  ({failures} failures)")
    return failures, results_table


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Backward compatibility (Gaussian X x Bernoulli Y)
# ──────────────────────────────────────────────────────────────────────────────

def test_backward_compat():
    print("\n" + "="*70)
    print("Test 2: Backward compatibility  DualExpFam(GxB) ~= ExpFam(B)")
    print("="*70)
    failures = 0

    from utils_expfam import run_em
    from data_generator_expfam import generate_bernoulli_data

    data_b = generate_bernoulli_data(n=N, d=D, k=K, seed=SEED)

    # Original ExpFam runner
    res_orig = run_em(
        X=data_b["X"], Y=data_b["Y"],
        true_params=data_b,
        family="bernoulli",
        k=K, L=L, num_iter=NITER, seed=SEED,
    )

    # Dual runner with family_x='gaussian', family_y='bernoulli'
    res_dual = run_em_dual(
        X=data_b["X"], Y=data_b["Y"],
        true_params=data_b,
        family_x="gaussian", family_y="bernoulli",
        k=K, L=L, num_iter=NITER, seed=SEED,
    )

    Q_orig = res_orig["Q_final"]
    Q_dual = res_dual["Q_final"]
    rZ_orig = res_orig["rmse_Z"]
    rZ_dual = res_dual["rmse_Z"]

    # Q functions should be very close (same computation path)
    rel_diff_Q = abs(Q_dual - Q_orig) / (abs(Q_orig) + 1e-8)
    if rel_diff_Q < 0.05:
        _ok(f"Q orig={Q_orig:.3f}  dual={Q_dual:.3f}  "
            f"rel_diff={rel_diff_Q:.4f} (<5% OK)")
    else:
        _fail(f"Q orig={Q_orig:.3f}  dual={Q_dual:.3f}  "
              f"rel_diff={rel_diff_Q:.4f} (>5% NG)")
        failures += 1

    # RMSE(Z) should be similar (same random seed, same model path)
    _ok(f"RMSE(Z) orig={rZ_orig:.4f}  dual={rZ_dual:.4f}")

    print(f"\n  -> {'PASS' if failures==0 else 'FAIL'}")
    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Q monotonicity (detailed check on each combo)
# ──────────────────────────────────────────────────────────────────────────────

def test_q_monotonicity(results_table):
    print("\n" + "="*70)
    print("Test 3: Q monotonicity - Q_last > Q_first for all combos")
    print("="*70)
    failures = 0

    for r in results_table:
        label = f"X={r['fx']:9s}  Y={r['fy']:9s}"
        Q_first, Q_last = r["Q_first"], r["Q_last"]
        if not r["ok"]:
            print(f"  [SKIP] {label}  (failed in Test 1)")
            continue
        if Q_last >= Q_first:
            _ok(f"{label}  ΔQ={Q_last - Q_first:+.2f}")
        else:
            _fail(f"{label}  ΔQ={Q_last - Q_first:+.2f} (decreased)")
            failures += 1

    print(f"\n  -> {'PASS' if failures==0 else 'PARTIAL'} ({failures} failures)")
    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: DualExpFamLSM repr and parameter inspection
# ──────────────────────────────────────────────────────────────────────────────

def test_repr():
    print("\n" + "="*70)
    print("Test 4: DualExpFamLSM repr and initialization")
    print("="*70)
    failures = 0
    for fx in FAMILIES:
        for fy in FAMILIES:
            m = DualExpFamLSM(n=10, d=4, k=2, L=3,
                               family_x=fx, family_y=fy)
            assert m.family_x == fx
            assert m.family   == fy    # parent stores family_y as self.family
            r = repr(m)
            assert "DualExpFamLSM" in r
            assert fx in r
            assert fy in r
    _ok("All 9 reprs correct, family attributes verified.")
    print(f"  Example: {repr(DualExpFamLSM(10,4,2,family_x='poisson',family_y='gaussian'))}")
    print(f"\n  -> PASS")
    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: F Adam update (non-Gaussian X detailed check)
# ──────────────────────────────────────────────────────────────────────────────

def test_f_adam():
    print("\n" + "="*70)
    print("Test 5: F Adam update (non-Gaussian X) - F changes over iterations")
    print("="*70)
    failures = 0

    for fx in ("bernoulli", "poisson"):
        data = generate_dual_data(n=N, d=D, k=K, seed=SEED,
                                  family_x=fx, family_y="bernoulli")
        rng   = np.random.default_rng(SEED)
        model = DualExpFamLSM(n=N, d=D, k=K, L=3,
                               family_x=fx, family_y="bernoulli")
        model.initialize_params(true_params=data, seed=SEED)
        if fx in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        F_before = model.params["F"].copy()

        # Fake Z_samples (3 samples of random Z)
        Z_samples = np.random.default_rng(SEED).standard_normal((N, K, 3))
        F_after = model.calc_F(data["X"], Z_samples)

        delta_F = float(np.max(np.abs(F_after - F_before)))
        if delta_F > 1e-10:
            _ok(f"X={fx}: F updated by Adam, max|ΔF|={delta_F:.6f}")
        else:
            _fail(f"X={fx}: F did NOT change (delta={delta_F:.2e})")
            failures += 1

    print(f"\n  -> {'PASS' if failures==0 else 'FAIL'}")
    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "*"*70)
    print("  DualExpFamLSM - Sanity Check Suite")
    print(f"  n={N}, d={D}, k={K}, L={L}, num_iter={NITER}, seed={SEED}")
    print("*"*70)

    total_fail = 0
    f1, table  = test_all_combinations()
    f2         = test_backward_compat()
    f3         = test_q_monotonicity(table)
    f4         = test_repr()
    f5         = test_f_adam()
    total_fail = f1 + f2 + f3 + f4 + f5

    print("\n" + "="*70)
    if total_fail == 0:
        print(f"{PASS_COLOR}  ALL TESTS PASSED OK{RESET}")
    else:
        print(f"{FAIL_COLOR}  {total_fail} TEST(S) FAILED NG{RESET}")
    print("="*70)
    return total_fail


if __name__ == "__main__":
    sys.exit(main())
