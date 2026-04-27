"""Wine DualExpFam experiment — Gaussian X + Bernoulli Y, k=6."""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))
from utils_expfam import run_em_dual, calc_bic_dual
warnings.filterwarnings("ignore")

RES = _SRC.parent / "results"

wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
labels = wine.target
n, d = X.shape
Y = (labels[:,None]==labels[None,:]).astype(float)
np.fill_diagonal(Y, 0)

K, L, NITER, N_TRIALS = 6, 10, 20, 5
family_x, family_y = "gaussian", "bernoulli"
print(f"Wine: n={n}, d={d}, K={K}, L={L}, NITER={NITER}")

# Dummy true_params with correct shapes
dummy = {
    "Z": np.zeros((n, K)), "F": np.zeros((d, K)),
    "sigma": np.eye(d), "var_z": 1.0,
    "w0": 0.0, "w": 0.5,
    "X": X, "Y": Y,
    "family_x": family_x, "family_y": family_y,
}

records = []
for trial in range(N_TRIALS):
    seed = 2024 + trial * 100
    try:
        res = run_em_dual(
            X=X, Y=Y, true_params=dummy,
            family_x=family_x, family_y=family_y,
            k=K, L=L, num_iter=NITER, seed=seed,
            compute_strict_Q=True,
        )
        bic, npar = calc_bic_dual(
            Q_strict=res["Q_strict"], k=K, n=n, d=d,
            family_x=family_x, family_y=family_y,
        )
        rec = {"trial": trial, "rmse_X": res["rmse_X"], "rmse_Y": res["rmse_Y"],
               "w0": res.get("w0_hat", float("nan")),
               "w":  res.get("w_hat",  float("nan")),
               "BIC": bic}
        print(f"  Trial {trial}: RMSE_X={rec['rmse_X']:.4f} RMSE_Y={rec['rmse_Y']:.4f}"
              f" w0={rec['w0']:.4f} w={rec['w']:.4f} BIC={bic:.0f}", flush=True)
    except Exception as e:
        print(f"  Trial {trial}: ERROR {e}")
        rec = {"trial": trial, "rmse_X": float("nan"), "rmse_Y": float("nan"),
               "w0": float("nan"), "w": float("nan"), "BIC": float("nan")}
    records.append(rec)

df = pd.DataFrame(records)
out = RES / "wine_dual_results.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print("\n--- Mean ---"); print(df.mean().round(4).to_string())
print("\n--- Min  ---"); print(df.min().round(4).to_string())
