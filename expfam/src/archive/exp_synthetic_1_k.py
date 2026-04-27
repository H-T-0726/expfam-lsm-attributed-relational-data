"""
Experiment 1: Latent dimension identification via BIC and RMSE(Z).

Proves that every family autonomously identifies k*=3 via BIC minimisation
and L-shaped RMSE(Z) curve when k is varied in {1,2,3,4,5,6}.

n=150, d=15, k_true=3, L=5, num_iter=8, 10 independent trials.
All 7 parameter RMSEs + BIC + timing recorded per trial.
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

# ── Config ──────────────────────────────────────────────────────────────
N=150; D=15; K_TRUE=3; K_LIST=[1,2,3,4,5,6]; L=5; NI=8; NT=10; BS=9100
FAMILIES = ["bernoulli","poisson","gaussian"]
GEN = {"bernoulli":{"w0_true":-1.0,"w_true":1.5},
       "poisson":  {"w0_true": 0.0,"w_true":0.5},
       "gaussian": {"w0_true": 0.0,"w_true":0.5,"sigma_y_true":0.1}}
CLR = {"bernoulli":"tab:blue","poisson":"tab:orange","gaussian":"tab:green"}
LBL = {"bernoulli":"Bernoulli","poisson":"Poisson","gaussian":"Gaussian"}
METS = ["rmse_Z","rmse_F","rmse_sigma","rmse_Y","rmse_X","w0_err","w_err"]

def gen(fam,n,d,k,seed):
    p=GEN[fam]
    if fam=="bernoulli": return generate_bernoulli_data(n=n,d=d,k=k,seed=seed,**p)
    if fam=="poisson":   return generate_poisson_data(n=n,d=d,k=k,seed=seed,**p)
    return generate_gaussian_data(n=n,d=d,k=k,seed=seed,**p)

def one_run(fam, k_est, dseed, mseed):
    data = gen(fam, N, D, K_TRUE, dseed)
    si = GEN[fam].get("sigma_y_true", 1.0)
    t0 = time.time()
    res = run_em(data["X"], data["Y"], true_params=data,
                 family=fam, k=k_est, L=L, num_iter=NI,
                 seed=mseed, compute_strict_Q=True, sigma_y_init=si)
    elapsed = time.time()-t0
    bic,_ = calc_bic(res["Q_strict"], k=k_est, n=N, d=D)
    row = {m: res.get(m, float("nan")) for m in METS+["Q_strict","Q_final","nan_occurred"]}
    row["BIC"] = bic; row["elapsed_s"] = elapsed
    return row

def main():
    print("="*65)
    print("  Exp 1: k variation  n=%d d=%d k*=%d L=%d iter=%d trials=%d"%(N,D,K_TRUE,L,NI,NT))
    print("="*65)
    records=[]; t0=time.time()

    for fam in FAMILIES:
        print(f"\n--- {LBL[fam]} ---")
        for k_est in K_LIST:
            for tr in range(NT):
                ds=BS+tr*23; ms=BS+tr*23+7
                r=one_run(fam, k_est, ds, ms)
                records.append({"family":fam,"k_est":k_est,"k_true":K_TRUE,
                                 "trial":tr,"data_seed":ds,**r})
            kr=[x for x in records if x["family"]==fam and x["k_est"]==k_est]
            mk="<-- k*" if k_est==K_TRUE else ""
            print(f"  k={k_est}  RMSE(Z)={np.mean([x['rmse_Z'] for x in kr]):.4f}"
                  f"  BIC={np.mean([x['BIC'] for x in kr]):.1f}  {mk}")

    df=pd.DataFrame(records)
    df.to_csv(OUT/"exp1_k_trials.csv", index=False)

    agg={m:["mean","std"] for m in METS+["BIC","elapsed_s"]}
    smry=df.groupby(["family","k_est"]).agg(agg)
    smry.columns=[f"{a}_{b}" for a,b in smry.columns]; smry=smry.reset_index()
    smry.to_csv(OUT/"exp1_k_summary.csv", index=False)

    # console matrix
    print("\n  BIC mean (rows=family, cols=k):")
    print("  "+"".join(f"{'':>3}k={k:<5}" for k in K_LIST))
    all_ok=True
    for fam in FAMILIES:
        fd=smry[smry.family==fam]
        best=int(fd.loc[fd.BIC_mean.idxmin(),"k_est"])
        if best!=K_TRUE: all_ok=False
        row=f"  {fam:<10}"
        for k in K_LIST:
            v=fd.loc[fd.k_est==k,"BIC_mean"].values[0]
            row+=f"  {'*' if k==best else ' '}{v:>9.0f}"
        print(row)
    print(f"\n  BIC selection: {'ALL PASS' if all_ok else 'SOME FAIL'}")

    _plots(df, smry)
    _report(smry, all_ok, time.time()-t0)
    print(f"\n  Total: {time.time()-t0:.1f}s")
    print("[exp_synthetic_1_k DONE]")

def _plots(df, smry):
    fig,axes=plt.subplots(2,3,figsize=(15,10),gridspec_kw={"hspace":0.42,"wspace":0.3})
    kv=np.array(K_LIST)
    for ci,fam in enumerate(FAMILIES):
        fd=smry[smry.family==fam]; ft=df[df.family==fam]; c=CLR[fam]
        for ri,(met,ylbl) in enumerate([("rmse_Z","RMSE(Z)"),("BIC","BIC")]):
            ax=axes[ri,ci]
            mn=fd[f"{met}_mean"].values; sd=fd[f"{met}_std"].values
            ax.errorbar(kv,mn,yerr=sd,fmt="o-",color=c,capsize=4,lw=2,ms=6,label="mean±std")
            for k in K_LIST:
                v=ft.loc[ft.k_est==k,met].values
                ax.scatter(np.full(len(v),k),v,color=c,s=12,alpha=0.3,zorder=4)
            ax.axvline(K_TRUE,color="gray",ls="--",lw=1.5,label=f"k*={K_TRUE}")
            bk=int(fd.loc[fd[f"{met}_mean"].idxmin(),"k_est"])
            ax.scatter([bk],[fd[f"{met}_mean"].min()],s=160,c="red",marker="*",zorder=6,label=f"min k={bk}")
            ax.set_xlabel("k"); ax.set_ylabel(ylbl)
            ax.set_title(f"{LBL[fam]}: {ylbl} vs k"); ax.legend(fontsize=7)
            ax.grid(True,alpha=0.3); ax.set_xticks(kv)
    plt.suptitle(f"Exp 1: k variation (n={N},d={D},k*={K_TRUE},L={L},iter={NI},trials={NT})",
                 fontsize=11, fontweight="bold")
    p=OUT/"exp1_k_plot.png"; plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

def _report(smry, all_ok, total):
    lines=["# Exp 1 Report: k Variation (BIC + All RMSE)\n\n"]
    lines.append(f"**Status:** {'ALL PASS' if all_ok else 'SOME FAIL'}  "
                 f"| n={N}, d={D}, k*={K_TRUE}, L={L}, iter={NI}, trials={NT}\n\n")
    lines.append("## BIC mean ± std matrix\n\n")
    lines.append("| Family |"+"".join(f" k={k} |" for k in K_LIST)+"\n")
    lines.append("|--------|"+"".join("-----------|" for _ in K_LIST)+"\n")
    for fam in FAMILIES:
        fd=smry[smry.family==fam]
        bk=int(fd.loc[fd.BIC_mean.idxmin(),"k_est"])
        row=f"| {fam} |"
        for k in K_LIST:
            v=fd.loc[fd.k_est==k,"BIC_mean"].values[0]
            s=fd.loc[fd.k_est==k,"BIC_std"].values[0]
            m="**" if k==bk else ""
            row+=f" {m}{v:.0f}±{s:.0f}{m} |"
        lines.append(row+"\n")
    lines.append("\n## RMSE(Z) mean ± std matrix\n\n")
    lines.append("| Family |"+"".join(f" k={k} |" for k in K_LIST)+"\n")
    lines.append("|--------|"+"".join("--------|" for _ in K_LIST)+"\n")
    for fam in FAMILIES:
        fd=smry[smry.family==fam]; row=f"| {fam} |"
        for k in K_LIST:
            v=fd.loc[fd.k_est==k,"rmse_Z_mean"].values[0]
            s=fd.loc[fd.k_est==k,"rmse_Z_std"].values[0]
            m="**" if k==K_TRUE else ""
            row+=f" {m}{v:.4f}±{s:.4f}{m} |"
        lines.append(row+"\n")
    lines.append(f"\n実行時間: {total:.1f}s\n")
    p=OUT/"GEMINI_REPORT_EXP1.md"
    p.write_text("".join(lines), encoding="utf-8")
    print(f"  Saved: {p}")

if __name__=="__main__": main()
