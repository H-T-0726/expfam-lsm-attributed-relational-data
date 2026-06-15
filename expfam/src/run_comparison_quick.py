"""
Quick comparison: old (0.5) vs fixed (no 0.5) for key conditions.
Runs oracle + conv-like + worst per scenario at n=150, L=5, NITER=8.
5 trials each condition to save time; same seeds as existing exp4.
"""

import sys, time, warnings
import numpy as np, pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, '../../reproduction/src')
warnings.filterwarnings("ignore")

from data_generator_expfam import generate_dual_data
from run_mismatch_fixed import run_em_fixed   # fixed model
from utils_expfam import run_em_dual           # old model

SCENARIOS = [
    ("A", "poisson",   "bernoulli"),
    ("B", "gaussian",  "poisson"),
    ("C", "bernoulli", "gaussian"),
]
# Worst conditions per scenario (from existing CSV analysis)
WORST = {
    "A": ("bernoulli", "bernoulli"),
    "B": ("poisson",   "bernoulli"),
    "C": ("gaussian",  "poisson"),
}
N, D, K_TRUE = 150, 15, 3
N_TRIALS, L, NITER = 5, 5, 8
BASE_SEED = 6000

FAM_LABEL = {"gaussian": "Gaussian", "bernoulli": "Bernoulli", "poisson": "Poisson"}

def run_both(data, fx, fy, seed, fix_w=False, fix_x=False):
    """Run old and fixed model on the same data. Returns (old_rmse, fix_rmse)."""
    old = run_em_dual(
        X=data["X"], Y=data["Y"], true_params=data,
        family_x=fx, family_y=fy,
        k=K_TRUE, L=L, num_iter=NITER, seed=seed,
        fix_w=fix_w, fix_x=fix_x,
    )
    fix = run_em_fixed(
        X=data["X"], Y=data["Y"], true_params=data,
        family_x=fx, family_y=fy,
        k=K_TRUE, L=L, num_iter=NITER, seed=seed,
        fix_w=fix_w, fix_x=fix_x,
    )
    return old["rmse_Z"], fix["rmse_Z"]

records = []
t0 = time.perf_counter()

for scen_tag, true_fx, true_fy in SCENARIOS:
    worst_fx, worst_fy = WORST[scen_tag]
    key_conditions = [
        ("oracle",   true_fx,      true_fy,       False, False),
        ("conv",     "gaussian",   "bernoulli",   False, False),
        ("worst",    worst_fx,     worst_fy,      False, False),
    ]
    print(f"\n{'='*55}")
    print(f"Scenario {scen_tag}: X={FAM_LABEL[true_fx]}, Y={FAM_LABEL[true_fy]}")
    print(f"{'='*55}")

    for cond_name, fx, fy, fw, fxi in key_conditions:
        old_vals, fix_vals = [], []
        for trial in range(N_TRIALS):
            data_seed  = BASE_SEED + trial * 100
            model_seed = data_seed + 50
            data = generate_dual_data(n=N, d=D, k=K_TRUE, seed=data_seed,
                                      family_x=true_fx, family_y=true_fy)
            old_r, fix_r = run_both(data, fx, fy, model_seed, fw, fxi)
            old_vals.append(old_r)
            fix_vals.append(fix_r)
            elapsed = (time.perf_counter() - t0) / 60
            print(f"  [{cond_name:6s}] X={FAM_LABEL[fx][:4]} Y={FAM_LABEL[fy][:4]} "
                  f"t={trial}  old={old_r:.4f} fix={fix_r:.4f}  [{elapsed:.1f}min]")

        records.append({
            "scenario": scen_tag, "true_x": true_fx, "true_y": true_fy,
            "est_x": fx, "est_y": fy, "condition": cond_name,
            "old_mean": np.mean(old_vals), "old_std": np.std(old_vals),
            "fix_mean": np.mean(fix_vals), "fix_std": np.std(fix_vals),
            "diff":  np.mean(fix_vals) - np.mean(old_vals),
            "ratio_fix_old": np.mean(fix_vals) / np.mean(old_vals),
        })

df = pd.DataFrame(records)
out = "D:/tento/kennkyu/expfam/results/distribution_mismatch_fixed/comparison_quick.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")

print("\n" + "="*65)
print("COMPARISON SUMMARY: Old (0.5) vs Fixed (no 0.5)")
print("="*65)
print(f"{'Scenario':<8} {'Condition':<8} {'est_X':<10} {'est_Y':<10} "
      f"{'Old':>8} {'Fixed':>8} {'Diff':>8} {'Ratio':>6}")
print("-"*65)
for _, row in df.iterrows():
    print(f"{row['scenario']:<8} {row['condition']:<8} "
          f"{FAM_LABEL[row['est_x']]:<10} {FAM_LABEL[row['est_y']]:<10} "
          f"{row['old_mean']:8.4f} {row['fix_mean']:8.4f} "
          f"{row['diff']:+8.4f} {row['ratio_fix_old']:6.3f}x")
