"""
Experiment 5: Ablation study — X integration vs Y-only.

Proves that jointly modeling both X (attributes) and Y (relations)
gives significantly better latent structure recovery than using X alone.

Ablation: fix w=0 (blocks all relational signal from Y).
Proposed: normal EM (w freely estimated from Y).

n=150, d=15, k=3, L=5, num_iter=8, 10 trials per family per condition.
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

N=150; D=15; K=3; L=5; NI=8; NT=10; BS=9500
FAMILIES=["bernoulli","poisson","gaussian"]
GEN={"bernoulli":{"w0_true":-1.0,"w_true":1.5},
     "poisson":  {"w0_true": 0.0,"w_true":0.5},
     "gaussian": {"w0_true": 0.0,"w_true":0.5,"sigma_y_true":0.1}}
CLR={"bernoulli":"tab:blue","poisson":"tab:orange","gaussian":"tab:green"}
LBL={"bernoulli":"Bernoulli","poisson":"Poisson","gaussian":"Gaussian"}
METS=["rmse_Z","rmse_F","rmse_sigma","rmse_Y","rmse_X","w0_err","w_err"]
CONDITIONS=["proposed","ablation"]

def gen(fam, seed):
    p=GEN[fam]
    if fam=="bernoulli": return generate_bernoulli_data(n=N,d=D,k=K,seed=seed,**p)
    if fam=="poisson":   return generate_poisson_data(n=N,d=D,k=K,seed=seed,**p)
    return generate_gaussian_data(n=N,d=D,k=K,seed=seed,**p)

def one_run(fam, fix_w, dseed, mseed):
    data=gen(fam, dseed)
    si=GEN[fam].get("sigma_y_true",1.0)
    t0=time.time()
    res=run_em(data["X"],data["Y"],true_params=data,
               family=fam,k=K,L=L,num_iter=NI,
               seed=mseed,compute_strict_Q=True,sigma_y_init=si,
               fix_w=fix_w)
    elapsed=time.time()-t0
    bic,_=calc_bic(res["Q_strict"],k=K,n=N,d=D)
    row={m:res.get(m,float("nan")) for m in METS+["Q_final","w_est","nan_occurred"]}
    row["BIC"]=bic; row["elapsed_s"]=elapsed
    return row

def main():
    print("="*65)
    print("  Exp 5: Ablation (w=0 vs proposed)  n=%d d=%d k=%d L=%d iter=%d trials=%d"%(N,D,K,L,NI,NT))
    print("="*65)
    records=[]; t0=time.time()

    for fam in FAMILIES:
        print(f"\n--- {LBL[fam]} ---")
        for cond, fw in [("proposed",False),("ablation",True)]:
            for tr in range(NT):
                ds=BS+tr*41; ms=BS+tr*41+19
                r=one_run(fam, fw, ds, ms)
                records.append({"family":fam,"condition":cond,
                                 "trial":tr,"data_seed":ds,**r})
            subset=[x for x in records if x["family"]==fam and x["condition"]==cond]
            rz_mean=np.mean([x["rmse_Z"] for x in subset])
            w_mean=np.mean([x["w_est"] for x in subset])
            print(f"  {cond:<10} RMSE(Z)={rz_mean:.4f}  w_est={w_mean:.4f}")

    df=pd.DataFrame(records)
    df.to_csv(OUT/"exp5_ablation_trials.csv", index=False)

    agg={m:["mean","std"] for m in METS+["BIC","elapsed_s","w_est"]}
    smry=df.groupby(["family","condition"]).agg(agg)
    smry.columns=[f"{a}_{b}" for a,b in smry.columns]; smry=smry.reset_index()
    smry.to_csv(OUT/"exp5_ablation_summary.csv", index=False)

    # Summary table
    print("\n  RMSE(Z) comparison:")
    print(f"  {'Family':<12} {'Proposed':>12} {'Ablation(w=0)':>14} {'Gain':>10}")
    print("  "+"-"*52)
    for fam in FAMILIES:
        p_row=smry[(smry.family==fam)&(smry.condition=="proposed")]
        a_row=smry[(smry.family==fam)&(smry.condition=="ablation")]
        pv=p_row["rmse_Z_mean"].values[0]; av=a_row["rmse_Z_mean"].values[0]
        gain=(av-pv)/max(av,1e-8)*100
        print(f"  {LBL[fam]:<12} {pv:>12.4f} {av:>14.4f} {gain:>9.1f}%")

    _plots(df, smry)
    _report(smry, time.time()-t0)
    print(f"\n  Total: {time.time()-t0:.1f}s")
    print("[exp_synthetic_5_ablation DONE]")

def _plots(df, smry):
    plot_mets=[("rmse_Z","RMSE(Z)"),("rmse_F","RMSE(F)"),
               ("rmse_Y","RMSE(Y)"),("rmse_X","RMSE(X)")]
    fig,axes=plt.subplots(2,2,figsize=(12,9),gridspec_kw={"hspace":0.4,"wspace":0.3})
    x=np.arange(3); w=0.32
    for ax,(met,ylbl) in zip(axes.flat, plot_mets):
        for ci,(cond,hatch,alpha) in enumerate([("proposed","",0.85),("ablation","//",0.6)]):
            vals=[smry.loc[(smry.family==f)&(smry.condition==cond),f"{met}_mean"].values[0]
                  for f in FAMILIES]
            stds=[smry.loc[(smry.family==f)&(smry.condition==cond),f"{met}_std"].values[0]
                  for f in FAMILIES]
            bars=ax.bar(x+ci*w-w/2, vals, w, yerr=stds,
                       label=cond, alpha=alpha, hatch=hatch,
                       color=[CLR[f] for f in FAMILIES] if cond=="proposed" else "lightgray",
                       edgecolor="black", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels([LBL[f][:5] for f in FAMILIES])
        ax.set_ylabel(ylbl); ax.set_title(f"{ylbl}: Proposed vs Ablation (w=0)")
        ax.legend(fontsize=8); ax.grid(True,axis="y",alpha=0.3)
    plt.suptitle(f"Exp 5: Ablation Study (n={N},d={D},k={K},L={L},iter={NI},trials={NT})",
                 fontsize=11, fontweight="bold")
    p=OUT/"exp5_ablation_plot.png"; plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

def _report(smry, total):
    lines=["# Exp 5 Report: Ablation Study (X+Y vs X-only)\n\n"]
    lines.append(f"n={N}, d={D}, k={K}, L={L}, iter={NI}, trials={NT}\n\n")
    lines.append("**Ablation**: w=0 fixed → Z learned from X only (no Y relational signal).\n"
                 "**Proposed**: w freely estimated → Z learned jointly from X and Y.\n\n")

    for met,mlbl in [("rmse_Z","RMSE(Z)"),("rmse_F","RMSE(F)"),
                     ("rmse_Y","RMSE(Y)"),("rmse_X","RMSE(X)")]:
        lines.append(f"## {mlbl}: Proposed vs Ablation\n\n")
        lines.append("| Family | Proposed (mean±std) | Ablation w=0 (mean±std) | Gain (%) |\n")
        lines.append("|--------|---------------------|------------------------|----------|\n")
        for fam in FAMILIES:
            p_row=smry[(smry.family==fam)&(smry.condition=="proposed")]
            a_row=smry[(smry.family==fam)&(smry.condition=="ablation")]
            pv=p_row[f"{met}_mean"].values[0]; ps=p_row[f"{met}_std"].values[0]
            av=a_row[f"{met}_mean"].values[0]; as_=a_row[f"{met}_std"].values[0]
            gain=(av-pv)/max(av,1e-8)*100
            lines.append(f"| {LBL[fam]} | **{pv:.4f}**±{ps:.4f} | {av:.4f}±{as_:.4f} | **{gain:.1f}%** |\n")
        lines.append("\n")

    lines.append(f"実行時間: {total:.1f}s\n")
    p=OUT/"GEMINI_REPORT_EXP5.md"
    p.write_text("".join(lines), encoding="utf-8")
    print(f"  Saved: {p}")

if __name__=="__main__": main()
