"""Follow-up checks requested by the independent review of mcem_stability_signal_matched_20261106.

Not pre-registered.  Everything here is either a technical check (T1, T2) or a descriptive,
exploratory re-tabulation of existing CSVs (E1-E4).  Nothing here changes a pre-registered verdict.

T1  C_Q at iteration 8 of every RQ1 fit == C_Q recorded by PR #155 (fit_criteria.csv), which the
    prereg asked for in addition to the state comparison.
T2  Evaluation isolation at iteration 64: rerun RQ1 A rep1 K5 s1 for 64 iterations WITHOUT the
    independent evaluation and compare its iteration-64 state with the stored state (64 extra
    fit-iterations; total 11,408 <= 12,000).
E1  Slack-change between checkpoints 48 and 64 measured with the independent evaluation itself,
    against 2 x sqrt(se48^2 + se64^2) of the evaluated slack (descriptive).
E2  S_Q - S_Q_eval (fit's own C_Q vs independent evaluation) by condition and checkpoint.
E3  RQ2 per-K data statistics (Bernoulli Fisher information mean p(1-p), X mean, row norm^2, w0).
E4  RQ2 per-replicate total gain (2dQ_X + 2dQ_Y) and its median, per side.

Usage:  python tools/research_audit/review_followup_mcem_stability.py
Writes  expfam/results/k_selection/mcem_stability_signal_matched_20261106/review_followup/ (refuses to overwrite).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import run_mcem_stability as rm  # noqa: E402

OUT = rm.OUT_ROOT / "review_followup"
AN = rm.OUT_ROOT / "analysis"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def t1() -> pd.DataFrame:
    prev = pd.read_csv(rm.PREV / "signal_strength" / "full_n75" / "fit_criteria.csv")
    prev = prev[(prev.checkpoint == 8) & (prev.k_est.isin([4, 5]))]
    mine = pd.concat([pd.read_csv(rm.OUT_ROOT / s / "checkpoint_rows.csv") for s in ("rq1_AB", "rq1_CD")])
    mine = mine[mine.checkpoint == 8]
    m = mine.merge(prev[["condition", "replicate", "k_est", "start", "C_Q"]].rename(columns={"k_est": "k_fit", "C_Q": "C_Q_pr155"}),
                   on=["condition", "replicate", "k_fit", "start"], how="left")
    m["abs_diff"] = (m.C_Q - m.C_Q_pr155).abs()
    return m[["condition", "replicate", "k_fit", "start", "C_Q", "C_Q_pr155", "abs_diff"]]


def t2() -> dict:
    prov = rm.a1.load_hist_provenance()
    data = rm.rq1_data(prov, rm.ss.w0_c())
    rep, k, s = 1, 5, 1
    b = data[rep]; v = b["cond"]["A"]
    seed = rm.hist.model_seed(5, rm.N, rep, k, s)
    t0 = time.time()
    _, snaps, _ = rm.run_fit(v["X"], v["Y"], b["train"], k, seed, 64, (64,), do_eval=False)
    mine = rm.state_arrays(snaps[64])
    npz = np.load(rm.STATE_ROOT / "rq1_AB_states.npz")
    key = f"RQ1_A_r{rep}_k{k}_s{s}_it64"
    diffs = {nm: float(np.max(np.abs(mine[nm] - np.asarray(npz[f"{key}_{nm}"], float)))) for nm in mine}
    return {"fit": key, "fit_iterations": 64, "runtime_s": round(time.time() - t0, 1), "max_abs_diff": diffs,
            "identical": all(d == 0.0 for d in diffs.values())}


def e1_e2() -> tuple[pd.DataFrame, pd.DataFrame]:
    loc = pd.read_csv(AN / "local_k4_k5_comparison.csv")
    p = loc.pivot_table(index=["condition", "replicate"], columns="checkpoint",
                        values=["S_Q", "S_Q_eval", "S_Q_eval_se"])
    e1 = pd.DataFrame({
        "abs_dS_Q_fit_64_48": (p[("S_Q", 64)] - p[("S_Q", 48)]).abs(),
        "abs_dS_Q_eval_64_48": (p[("S_Q_eval", 64)] - p[("S_Q_eval", 48)]).abs(),
        "two_se_diff_eval": 2 * np.sqrt(p[("S_Q_eval_se", 64)] ** 2 + p[("S_Q_eval_se", 48)] ** 2),
    }).reset_index()
    e1["eval_within_2se"] = e1.abs_dS_Q_eval_64_48 <= e1.two_se_diff_eval
    loc["S_Q_minus_eval"] = loc.S_Q - loc.S_Q_eval
    e2 = loc.groupby(["condition", "checkpoint"]).S_Q_minus_eval.agg(
        median="median", mean="mean", n_positive=lambda x: int((x > 0).sum()), n="count").reset_index()
    return e1, e2


def e3() -> pd.DataFrame:
    pv = pd.read_csv(rm.OUT_ROOT / "gates" / "rq2_dataset_provenance.csv")
    cols = ["w0", "bernoulli_fisher_mean_p1mp", "x_mean", "mean_row_norm_sq", "y_density_upper", "max_rate_truth"]
    g = pv.groupby("k_gen")[cols].agg(["min", "median", "max"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def e4() -> pd.DataFrame:
    d = pd.read_csv(AN / "signal_matched_k3_k5.csv")
    d = d[d.row_type == "local_slack"].copy()
    d["total_gain"] = d.gain_x + d.gain_y
    return d.groupby(["k_gen", "side", "checkpoint"]).agg(
        total_gain_median=("total_gain", "median"), gain_x_median=("gain_x", "median"),
        gain_y_median=("gain_y", "median"), penalty=("penalty", "first"), prior_cost=("prior_cost", "first"),
        S_Q_median=("S_Q", "median")).reset_index()


def main() -> int:
    if OUT.exists():
        print(f"refusing to overwrite {OUT}")
        return 1
    OUT.mkdir(parents=True)
    t1df = t1(); t1df.to_csv(OUT / "t1_cq_iter8_vs_pr155.csv", index=False)
    e1, e2 = e1_e2()
    e1.to_csv(OUT / "e1_eval_slack_change_48_64.csv", index=False)
    e2.to_csv(OUT / "e2_fit_vs_eval_slack.csv", index=False)
    e3().to_csv(OUT / "e3_rq2_data_statistics.csv", index=False)
    e4().to_csv(OUT / "e4_rq2_total_gain.csv", index=False)
    t2r = t2()
    summary = {
        "experiment_id": rm.EXPERIMENT_ID, "note": "not pre-registered; technical checks and descriptive tables",
        "T1": {"fits": int(len(t1df)), "missing": int(t1df.C_Q_pr155.isna().sum()),
               "max_abs_diff": float(t1df.abs_diff.max())},
        "T2": t2r,
        "E1": {"datasets": int(len(e1)), "eval_within_2se": int(e1.eval_within_2se.sum()),
               "median_abs_dS_fit": float(e1.abs_dS_Q_fit_64_48.median()),
               "median_abs_dS_eval": float(e1.abs_dS_Q_eval_64_48.median()),
               "median_two_se_diff_eval": float(e1.two_se_diff_eval.median())},
        "git_head": rm.subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=rm.ROOT).stdout.strip(),
    }
    summary["files"] = {p.name: sha_file(p) for p in sorted(OUT.glob("*.csv"))}
    (OUT / "review_followup_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("T1", "T2", "E1")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
