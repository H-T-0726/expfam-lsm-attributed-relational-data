"""Save / resume tests for run_mcem_noise_separation_v2 (must PASS before the production runs).

Runs real fits into a scratch results/artifact root, kills a running process, corrupts files, resumes,
and compares the saved rows / states with direct in-memory computation.

Usage:  python tools/research_audit/test_resume_mcem_noise_separation.py <scratch_dir>
Writes  expfam/results/k_selection/mcem_noise_separation_20261106/exec_v2/resilience_tests/report.json (refuses to overwrite)
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_mcem_noise_separation as ns  # noqa: E402
import run_mcem_noise_separation_v2 as v2  # noqa: E402

RUNNER = str(_HERE / "run_mcem_noise_separation_v2.py")
results: list[dict] = []


def rec(name, ok, detail=""):
    results.append({"test": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail != "" else ""), flush=True)


def cli(args, rr, ar, wait=True):
    cmd = [sys.executable, RUNNER, *args, "--results-root", str(rr), "--artifact-root", str(ar)]
    if wait:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def frame_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            if not np.array_equal(np.isnan(xv), np.isnan(yv)) or not np.array_equal(xv[~np.isnan(xv)], yv[~np.isnan(yv)]):
                return False
        elif not (x.astype(str) == y.astype(str)).all():
            return False
    return True


def main() -> int:
    scratch = Path(sys.argv[1]).resolve()
    out = v2.RESULTS_ROOT / "resilience_tests"
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 1
    rr, ar = scratch / "results", scratch / "artifacts"
    ns.need(not rr.exists() and not ar.exists(), "scratch not empty")
    ctx = v2.Ctx(rr, ar)
    specs = v2.expected_fits("phase3", None, ["A"])
    s1, s2, s3 = specs[0], specs[1], specs[2]

    # T1: one fit -> CSV + state file + valid manifest
    r = cli(["run", "--stage", "phase3", "--conds", "A", "--max-fits", "1"], rr, ar)
    st, why = v2.validate(ctx, s1)
    d1 = ctx.fit_dir(s1)
    rec("T1 one completed fit leaves CSVs, state npz and a valid manifest",
        r.returncode == 0 and st == "OK" and all((d1 / f"{n}.csv").exists() for n in v2.CSV_NAMES["phase3"]) and ctx.npz_path(s1).exists(),
        f"rc={r.returncode} state={st} {why}")

    # T2: kill while the third fit runs; completed fits survive
    p = cli(["run", "--stage", "phase3", "--conds", "A", "--max-fits", "5"], rr, ar, wait=False)
    t0 = time.time()
    while not (ctx.fit_dir(s3).exists()) and time.time() - t0 < 600:
        time.sleep(1)
    time.sleep(5)
    psutil.Process(p.pid).kill()
    p.wait()
    a, b, c = (v2.validate(ctx, s)[0] for s in (s1, s2, s3))
    rec("T2 forced kill: completed fits remain valid, the interrupted fit is INCOMPLETE (not complete)",
        (a, b, c) == ("OK", "OK", "INCOMPLETE"), (a, b, c))

    # T3: resume skips validated fits and reruns only the incomplete one
    r = cli(["run", "--stage", "phase3", "--conds", "A", "--max-fits", "1"], rr, ar)
    log = r.stdout
    aside = list(ctx.fit_dir(s3).parent.glob(f"{s3['fit_key']}.incomplete-*"))
    rec("T3 resume: validated fits skipped, incomplete fit set aside (renamed, not deleted) and rerun",
        f"SKIP {s1['fit_key']}" in log and f"SKIP {s2['fit_key']}" in log and f"SET_ASIDE {s3['fit_key']}" in log
        and v2.validate(ctx, s3)[0] == "OK" and len(aside) == 1, f"aside={len(aside)}")

    # T4: incomplete / corrupt files are never treated as complete
    stray = d1 / "probe_rows.csv.tmp-999"
    stray.write_text("partial", encoding="utf-8")
    ok_stray = v2.validate(ctx, s1)[0] == "OK"
    pr = ctx.fit_dir(s2) / "probe_rows.csv"
    orig = pr.read_bytes()
    pr.write_bytes(orig + b"x")
    bad = v2.validate(ctx, s2)
    pr.write_bytes(orig)
    s4 = specs[3]
    d4 = ctx.fit_dir(s4)
    d4.mkdir(parents=True)
    (d4 / "manifest.json").write_text('{"status": "OK", "files": {', encoding="utf-8")
    trunc = v2.validate(ctx, s4)
    (d4 / "manifest.json").write_text(json.dumps({"status": "OK", "config_hash": "0" * 64, "files": {}}), encoding="utf-8")
    stale = v2.validate(ctx, s4)
    rec("T4 stray .tmp ignored; checksum mismatch, truncated manifest and stale config are INCOMPLETE",
        ok_stray and bad[0] == "INCOMPLETE" and trunc[0] == "INCOMPLETE" and stale[0] == "INCOMPLETE",
        f"{bad[1]} | {trunc[1][:30]} | {stale[1]}")
    os.replace(d4, d4.with_name(f"{d4.name}.testfixture"))

    # T5: saved rows == direct in-memory computation (same seeds and conditions)
    S, P, R, arrays = ns.phase3_fit(ctx.data, ns.src_npz("A"), "A", s1["replicate"], s1["k_fit"], s1["start"])
    eq = all(frame_equal(pd.read_csv(d1 / f"{n}.csv"), pd.read_csv(io.StringIO(v2.csv_bytes(rows).decode())))
             for n, rows in (("state_rows", S), ("replicate_rows", R)))
    P2 = pd.DataFrame(P).drop(columns=["probe_seconds"])
    Psaved = pd.read_csv(d1 / "probe_rows.csv").drop(columns=["probe_seconds"])
    eq_np = frame_equal(Psaved, pd.read_csv(io.StringIO(v2.csv_bytes(P2.to_dict("records")).decode())))
    z = np.load(ctx.npz_path(s1))
    eq_arr = all(np.array_equal(z[k], arrays[k]) for k in arrays)
    rec("T5 rerun with the same seeds reproduces the saved rows and arrays exactly", eq and eq_np and eq_arr,
        f"state/replicate rows equal={eq}, probe rows equal (excluding wall-time probe_seconds)={eq_np}, arrays equal={eq_arr}")

    # T6: saving / snapshotting does not change the estimation path (Phase 4 code, L=20, 8 iterations)
    rep, k, L = 1, 5, 20
    _, C, _, _, _, arr = ns.phase4_fit(ctx.data, "A", rep, k, L, max_iter=8)
    b = ctx.data[rep]; v = b["cond"]["A"]
    plain = {}
    ns.run_em_instrumented(v["X"], v["Y"], family_x=ns.hist.FAMILY_X, family_y=ns.hist.FAMILY_Y, k=k, L=L, max_iter=8,
                           checkpoints=(8,), seed=ns.hist.model_seed(5, ns.N, rep, k, 1), train_mask=b["train"],
                           numerics_mode=ns.hist.NUMERICS_MODE, stability_window=ns.rm.STAB_WINDOW, on_nan="fail",
                           on_checkpoint=lambda r_: plain.update(Zs=np.array(r_["Z_samples"]), F=np.array(r_["F"])))
    key = f"P4_L{L}_A_r{rep}_k{k}_s1_it8"
    rec("T6 per-iteration snapshots + saving leave the MCEM path unchanged (bit-identical state at iteration 8)",
        np.array_equal(arr[f"{key}_Zsamples"], plain["Zs"]) and np.array_equal(arr[f"{key}_F"], plain["F"]))

    # T7: a full Phase 4 fit through the runner is saved and its C_Q can be recomputed from the saved state
    p4 = v2.expected_fits("phase4", 20, ["A"])[0]
    r = cli(["run", "--stage", "phase4", "--L", "20", "--conds", "A", "--max-fits", "1"], rr, ar)
    st4 = v2.validate(ctx, p4)
    z = np.load(ctx.npz_path(p4))
    kk = f"P4_L20_{p4['condition']}_r{p4['replicate']}_k{p4['k_fit']}_s1_it32"
    m = ns.new_model(p4["k_fit"], 20, ctx.data[p4["replicate"]]["train"])
    w0, w, var_z = (float(x) for x in z[f"{kk}_scalars"])
    th = {"F": z[f"{kk}_F"], "sigma": z[f"{kk}_sigma"], "w0": w0, "w": w, "var_z": var_z}
    ns.set_theta(m, th, z[f"{kk}_Z"])
    vv = ctx.data[p4["replicate"]]["cond"]["A"]
    cq = ns.panel.cq_terms(m, vv["X"], vv["Y"], z[f"{kk}_Zsamples"], num_params=ns.num_params(p4["k_fit"]), n=ns.N)
    crow = pd.read_csv(ctx.fit_dir(p4) / "checkpoint_rows.csv")
    saved = float(crow[crow.checkpoint == 32].C_Q_fit.iloc[0])
    man4 = json.loads((ctx.fit_dir(p4) / "manifest.json").read_text(encoding="utf-8"))
    rec("T7 Phase 4 fit saved via the runner; C_Q recomputed from the saved state equals the saved C_Q",
        r.returncode == 0 and st4[0] == "OK" and abs(cq["C_Q"] - saved) < 1e-6,
        f"diff {abs(cq['C_Q'] - saved):.1e}; L=20 fit {man4['seconds']} s, peak RSS {man4['peak_rss_mb']} MB")

    # T8: Phase 3 Q recomputation from the saved evaluation chains
    z = np.load(ctx.npz_path(s1))
    src = ns.src_npz("A")
    key64 = f"RQ1_A_r{s1['replicate']}_k{s1['k_fit']}_s{s1['start']}_it64"
    th, Zc, _ = ns.load_state(src, key64)
    m = ns.new_model(s1["k_fit"], 5, ctx.data[s1["replicate"]]["train"])
    ns.set_theta(m, th, Zc)
    vv = ctx.data[s1["replicate"]]["cond"]["A"]
    cq = ns.panel.cq_terms(m, vv["X"], vv["Y"], z[f"{key64}_eval_chains"], num_params=ns.num_params(s1["k_fit"]), n=ns.N)
    prow = pd.read_csv(d1 / "probe_rows.csv")
    saved = float(prow[prow.checkpoint == 64].C_Q_L80.iloc[0])
    rec("T8 Phase 3: C_Q(L=80) recomputed from the saved evaluation chains equals the saved value", abs(cq["C_Q"] - saved) < 1e-6,
        f"diff {abs(cq['C_Q'] - saved):.1e}")

    # T9: aggregation (partial) has no duplicates and lists missing fits as NOT_RUN
    r = cli(["aggregate", "--stage", "phase3", "--conds", "A", "--allow-partial"], rr, ar)
    led = pd.read_csv(rr / "phase3" / "aggregate" / "ledger.csv")
    pr = pd.read_csv(rr / "phase3" / "aggregate" / "probe_rows.csv")
    rec("T9 aggregation: no duplicate rows, exactly the validated fits included, the rest NOT_RUN",
        r.returncode == 0 and len(led) == 32 and int((led.status == "OK").sum()) == 3 and len(pr) == 6
        and not pr.duplicated(["fit_id", "checkpoint"]).any() and int((led.status == "NOT_RUN").sum()) == 29,
        led.status.value_counts().to_dict())

    out.mkdir(parents=True)
    n_pass = sum(x["status"] == "PASS" for x in results)
    (out / "report.json").write_text(json.dumps({"experiment_id": ns.EXPERIMENT_ID, "git_head": ns.git_head(), "scratch": str(scratch),
                                                 "passed": n_pass, "total": len(results), "tests": results}, indent=1),
                                     encoding="utf-8")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
