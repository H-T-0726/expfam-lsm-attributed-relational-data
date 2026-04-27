"""
Experiment 3: Robustness/scalability — all parameter RMSEs vs d.

Proves that estimation does not break down as attribute dimension d
(noise dimension) increases for all 3 families.

d in {5,10,15,20,25,30}, n=150, k=3, L=5, num_iter=8, 10 trials.
"""

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys, time

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_ROOT / "reproduction" / "src"))

from utils_expfam import run_em, calc_bic
from data_generator_expfam import (
    generate_bernoulli_data, generate_poisson_data, generate_gaussian_data,
)

OUT = _ROOT / "expfam" / "results"; OUT.mkdir(parents=True, exist_ok=True)

D_LIST=[5,10,15,20,25,30]; N=150; K=3; L=5; NI=8; NT=10; BS=9300
FAMILIES=["bernoulli","poisson","gaussian"]
GEN={"bernoulli":{"w0_true":-1.0,"w_true":1.5},
     "poisson":  {"w0_true": 0.0,"w_true":0.5},
     "gaussian": {"w0_true": 0.0,"w_true":0.5,"sigma_y_true":0.1}}
CLR={"bernoulli":"tab:blue","poisson":"tab:orange","gaussian":"tab:green"}
LBL={"bernoulli":"Bernoulli","poisson":"Poisson","gaussian":"Gaussian"}
METS=["rmse_Z","rmse_F","rmse_sigma","rmse_Y","rmse_X","w0_err","w_err"]

def gen(fam,d,seed):
    p=GEN[fam]
    if fam=="bernoulli": return generate_bernoulli_data(n=N,d=d,k=K,seed=seed,**p)
    if fam=="poisson":   return generate_poisson_data(n=N,d=d,k=K,seed=seed,**p)
    return generate_gaussian_data(n=N,d=d,k=K,seed=seed,**p)

def one_run(fam, d, dseed, mseed):
    data=gen(fam, d, dseed)
    si=GEN[fam].get("sigma_y_true",1.0)
    t0=time.time()
    res=run_em(data["X"],data["Y"],true_params=data,
               family=fam,k=K,L=L,num_iter=NI,
               seed=mseed,compute_strict_Q=True,sigma_y_init=si)
    elapsed=time.time()-t0
    bic,_=calc_bic(res["Q_strict"],k=K,n=N,d=d)
    row={m:res.get(m,float("nan")) for m in METS+["Q_strict","Q_final","nan_occurred"]}
    row["BIC"]=bic; row["elapsed_s"]=elapsed
    return row

def main():
    print("="*65)
    print("  Exp 3: d variation  n=%d k=%d L=%d iter=%d trials=%d"%(N,K,L,NI,NT))
    print("="*65)
    records=[]; t0=time.time()

    for fam in FAMILIES:
        print(f"\n--- {LBL[fam]} ---")
        for d in D_LIST:
            for tr in range(NT):
                ds=BS+tr*31; ms=BS+tr*31+13
                r=one_run(fam, d, ds, ms)
                records.append({"family":fam,"d":d,"n":N,"k":K,
                                 "trial":tr,"data_seed":ds,**r})
            dr=[x for x in records if x["family"]==fam and x["d"]==d]
            print(f"  d={d:2d}  RMSE(Z)={np.mean([x['rmse_Z'] for x in dr]):.4f}"
                  f"  RMSE(F)={np.mean([x['rmse_F'] for x in dr]):.4f}")

    df=pd.DataFrame(records)
    df.to_csv(OUT/"exp3_d_trials.csv", index=False)

    agg={m:["mean","std"] for m in METS+["BIC","elapsed_s"]}
    smry=df.groupby(["family","d"]).agg(agg)
    smry.columns=[f"{a}_{b}" for a,b in smry.columns]; smry=smry.reset_index()
    smry.to_csv(OUT/"exp3_d_summary.csv", index=False)

    print("\n  RMSE(Z) mean:")
    print("  "+"".join(f"  d={d:<3}" for d in D_LIST))
    for fam in FAMILIES:
        fd=smry[smry.family==fam]
        row=f"  {fam:<10}"
        for d in D_LIST:
            v=fd.loc[fd.d==d,"rmse_Z_mean"].values[0]
            row+=f"  {v:.4f}"
        print(row)

    _plots(df, smry)
    _report(smry, time.time()-t0)
    print(f"\n  Total: {time.time()-t0:.1f}s")
    print("[exp_synthetic_3_d DONE]")

def _plots(df, smry):
    plot_mets=[("rmse_Z","RMSE(Z)"),("rmse_F","RMSE(F)"),
               ("rmse_sigma","RMSE(Sigma)"),("rmse_X","RMSE(X)")]
    fig,axes=plt.subplots(2,2,figsize=(12,9),gridspec_kw={"hspace":0.4,"wspace":0.3})
    dv=np.array(D_LIST)
    for ax,(met,ylbl) in zip(axes.flat, plot_mets):
        for fam in FAMILIES:
            fd=smry[smry.family==fam]
            mn=fd[f"{met}_mean"].values; sd=fd[f"{met}_std"].values
            ax.errorbar(dv,mn,yerr=sd,fmt="o-",color=CLR[fam],capsize=4,
                       lw=2,ms=6,label=LBL[fam])
        ax.set_xlabel("d"); ax.set_ylabel(ylbl)
        ax.set_title(ylbl+" vs d"); ax.legend(fontsize=8)
        ax.grid(True,alpha=0.3); ax.set_xticks(dv)
    plt.suptitle(f"Exp 3: d variation (n={N},k={K},L={L},iter={NI},trials={NT})",
                 fontsize=11, fontweight="bold")
    p=OUT/"exp3_d_plot.png"; plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

def _report(smry, total):
    lines=["# Exp 3 Report: d Variation (Scalability)\n\n"]
    lines.append(f"n={N}, k={K}, L={L}, iter={NI}, trials={NT}\n\n")
    for met,mlbl in [("rmse_Z","RMSE(Z)"),("rmse_F","RMSE(F)"),
                     ("rmse_sigma","RMSE(Sigma)"),("rmse_Y","RMSE(Y)")]:
        lines.append(f"## {mlbl} mean ± std\n\n")
        lines.append("| Family |"+"".join(f" d={d} |" for d in D_LIST)+"\n")
        lines.append("|--------|"+"".join("--------|" for _ in D_LIST)+"\n")
        for fam in FAMILIES:
            fd=smry[smry.family==fam]; row=f"| {fam} |"
            for d in D_LIST:
                v=fd.loc[fd.d==d,f"{met}_mean"].values[0]
                s=fd.loc[fd.d==d,f"{met}_std"].values[0]
                row+=f" {v:.4f}±{s:.4f} |"
            lines.append(row+"\n")
        lines.append("\n")
    lines.append(f"実行時間: {total:.1f}s\n")
    p=OUT/"GEMINI_REPORT_EXP3.md"
    p.write_text("".join(lines), encoding="utf-8")
    print(f"  Saved: {p}")

if __name__=="__main__": main()
