"""Generate the seminar summary tables (FACT only) from a run's CSVs.

Writes ``reports/distribution_selection/seminar_family_selection_summary_<date>.md``
(and a copy in the run's ``audit/``).  Every number comes from the run's
tables; interpretation is NOT generated here -- it lives in a separately
written report that cites these tables.

    python report_multisupport_family_selection.py --run-dir DIR [--date YYYYMMDD]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def rd(p):
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick(rows, **kw):
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]


def one(rows, **kw):
    g = pick(rows, **kw)
    return g[0] if g else None


def pct(r):
    if r is None or int(r["n_evaluated"]) == 0:
        return "–"
    return (f"{float(r['recovery_rate']):.3f} ({r['n_correct']}/{r['n_evaluated']}"
            f"{', failed ' + r['n_failed'] if r['n_failed'] != '0' else ''})")


def f3(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "–"


SG = {"COUNT": "GROUP_COUNT", "POSITIVE": "GROUP_POSITIVE_CONTINUOUS",
      "BOUNDED": "GROUP_BOUNDED_CONTINUOUS", "REAL": "GROUP_REAL"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    a = ap.parse_args(argv)
    run, t = a.run_dir, a.run_dir / "tables"
    info = json.loads((run / "runinfo.json").read_text(encoding="utf-8"))
    proto = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    audit = (json.loads((run / "audit_report.json").read_text(encoding="utf-8"))
             if (run / "audit_report.json").exists() else {"verdict": "NOT_RUN"})
    comp = rd(t / "family_compatibility.csv")
    reg = rd(t / "candidate_registry.csv")
    fam = rd(t / "family_recovery_summary.csv")
    cand = rd(t / "candidate_model_recovery_summary.csv")
    rmse = rd(t / "rmse_summary.csv")
    gaps = rd(t / "rmse_paired_gaps.csv")
    sa = rd(t / "start_agreement.csv")
    cal = rd(t / "difficulty_calibration.csv")
    conv = rd(t / "convergence_summary.csv")
    done = rd(run / "completed_cases.csv")
    L = []
    w = L.append
    w(f"# Multi-support family selection — seminar summary tables ({a.date})")
    w("")
    w("> **自動生成（FACT のみ）。** 数値はすべて run の CSV から生成した。解釈は別の報告書に分離している。")
    w("> 系列 **E（experimental prototype; 本文採用不可）**。controlled synthetic experiment（nuisance 固定・K=3 固定）。")
    w("")
    w(f"- run: `expfam/results/family_selection/{run.name}/`")
    w(f"- init git SHA: `{info['git_sha']}`; phase SHAs: "
      + ", ".join(f"{p['phase']}=`{p['git_sha'][:8]}`" for p in info.get("phase_runs", [])))
    w(f"- protocol sha256: `{info['protocol_sha256']}`")
    w(f"- audit verdict: **{audit.get('verdict')}** (BLOCKER {audit.get('n_blocker', '–')}, "
      f"WARNING {audit.get('n_warning', '–')})")
    by_phase = {}
    for r in done:
        by_phase.setdefault(r["phase"], [0, 0])
        by_phase[r["phase"]][0] += 1
        by_phase[r["phase"]][1] += r["status"] != "SUCCESS"
    w("- completed cases by phase (failed): " +
      ", ".join(f"{k} {v[0]} ({v[1]})" for k, v in sorted(by_phase.items())))
    w(f"- generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    w("")

    w("## 表1 実装・検証した family（数理 audit と test 結果）")
    w("")
    w("| family | support group | 判定 | T(x) | 固定 hyperparameter | 実装 | test |")
    w("|---|---|---|---|---|---|---|")
    for r in comp:
        w(f"| {r['family_name']} | {r['support_group']} | {r['compatibility_status']} | "
          f"{r['T_definition'] or '–'} | {r['fixed_hyperparameters'] or '–'} | "
          f"{r['implemented']} | {r['test_status']} |")
    w("")
    w("candidate model（priority 順。SUPPORT_ONLY は各群の先頭 admissible、START_FIRST/LAST は先頭/末尾）:")
    w("")
    for g, order in proto["priority"].items():
        w(f"- {g}: " + " → ".join(f"`{c}`" for c in order))
    w("")

    w("## 表2 support group 別 family recovery（main stage, EASY/HARD・n 込み）")
    w("")
    w("score-decided = admissible 候補が 2 つ以上あり score で決まった列のみ。all = gate で決まった列と失敗も含む全列"
      "（失敗は不正解として分母に残す）。値は rate (正解/評価数)。")
    w("")
    w("| support group | scope | A: selector only (true Z) | B: Auto START_FIRST | B: Auto START_LAST | C: mixed Auto START_FIRST | C: mixed Auto START_LAST |")
    w("|---|---|---|---|---|---|---|")
    for g in ("COUNT", "POSITIVE", "BOUNDED", "REAL"):
        for scope in ("score_decided_only", "all_columns"):
            cells = []
            for exp, meth in (("A", "SELECTOR_ONLY"), ("B", "AUTO_START_FIRST"),
                              ("B", "AUTO_START_LAST"), ("C", "AUTO_START_FIRST"),
                              ("C", "AUTO_START_LAST")):
                gd = "MIXED" if exp == "C" else g
                cells.append(pct(one(fam, scope=scope, grouping="pooled_difficulty_n",
                                     experiment=exp, stage="main", method=meth,
                                     group_design=gd, support_group=SG[g], true_family="ALL")))
            w(f"| {g} | {scope} | " + " | ".join(cells) + " |")
    w("")
    w("### 表2b true candidate model 別 recovery（A: selector only, main, score-decided, candidate-model level）")
    w("")
    w("| true candidate model | candidate-model recovery |")
    w("|---|---|")
    for r in pick(cand, scope="score_decided_only", grouping="pooled_difficulty_n",
                  experiment="A", stage="main", method="SELECTOR_ONLY"):
        if r["true_candidate_model"] == "ALL":
            continue
        w(f"| {r['true_candidate_model']} ({r['support_group']}) | {pct(r)} |")
    w("")
    w("| true family | family-level recovery (A, score-decided) |")
    w("|---|---|")
    for r in pick(fam, scope="score_decided_only", grouping="pooled_difficulty_n",
                  experiment="A", stage="main", method="SELECTOR_ONLY"):
        if r["true_family"] == "ALL" or r["support_group"] == "ALL":
            continue
        w(f"| {r['true_family']} ({r['support_group']}) | {pct(r)} |")
    w("")

    w("## 表3 RMSE(Z)（Procrustes 後, main stage）: Oracle / Auto / Support-only")
    w("")
    w("| design | difficulty | n | Oracle mean±sd (fits, failed) | Auto START_FIRST | Auto START_LAST | Support-only |")
    w("|---|---|---|---|---|---|---|")
    for g in ("COUNT", "POSITIVE", "BOUNDED", "MIXED"):
        for diff in ("EASY", "HARD"):
            for n in ("75", "150"):
                cells = []
                for m in ("ORACLE", "AUTO_START_FIRST", "AUTO_START_LAST", "SUPPORT_ONLY"):
                    r = one(rmse, stage="main", group_design=g, difficulty=diff, n=n, method=m)
                    cells.append("–" if r is None else
                                 f"{f3(r['rmse_mean'])}±{f3(r['rmse_sd'])} ({r['n_fits']}, {r['n_failed']})")
                if all(c == "–" for c in cells):
                    continue
                w(f"| {g} | {diff} | {n} | " + " | ".join(cells) + " |")
    w("")
    w("paired gap（同一 dataset 内の差、3 手法とも成功した dataset のみ・件数を併記）:")
    w("")
    w("| design | Auto | datasets | mean(Auto−Oracle) | mean(Auto−SupportOnly) | Auto < SupportOnly の dataset 数 |")
    w("|---|---|---|---|---|---|")
    for g in ("COUNT", "POSITIVE", "BOUNDED", "MIXED"):
        for auto in ("AUTO_START_FIRST", "AUTO_START_LAST"):
            rows = [r for r in gaps if r["stage"] == "main" and r["group_design"] == g
                    and r["auto_method"] == auto and r["complete_pair"] == "True"]
            allr = [r for r in gaps if r["stage"] == "main" and r["group_design"] == g
                    and r["auto_method"] == auto]
            if not allr:
                continue
            go = [float(r["gap_auto_minus_oracle"]) for r in rows]
            gs = [float(r["gap_auto_minus_support_only"]) for r in rows]
            w(f"| {g} | {auto} | {len(rows)}/{len(allr)} | "
              f"{(sum(go) / len(go)) if go else float('nan'):+.4f} | "
              f"{(sum(gs) / len(gs)) if gs else float('nan'):+.4f} | "
              f"{sum(1 for x in gs if x < 0)}/{len(gs)} |")
    w("")

    w("## 表4 EASY / HARD（difficulty は selector と独立に distribution distance で事前 freeze）")
    w("")
    w("| support group | τ EASY | τ HARD | D(τ_EASY) | D(τ_HARD) | A recovery EASY | A recovery HARD | B Auto(FIRST) EASY | B Auto(FIRST) HARD |")
    w("|---|---|---|---|---|---|---|---|---|")
    sel = proto["difficulty"]["selection"]
    for g in ("COUNT", "POSITIVE", "BOUNDED"):
        te, th = sel[SG[g]]["EASY"], sel[SG[g]]["HARD"]
        D = {float(r["tau"]): r["D_mean_min_kl"] for r in cal
             if r["row_type"] == "group_difficulty" and r["support_group"] == SG[g]}
        cells = []
        for exp, meth in (("A", "SELECTOR_ONLY"), ("B", "AUTO_START_FIRST")):
            for diff in ("EASY", "HARD"):
                cells.append(pct(one(fam, scope="score_decided_only", grouping="by_difficulty",
                                     experiment=exp, stage="main", method=meth, group_design=g,
                                     difficulty=diff, support_group=SG[g], true_family="ALL")))
        w(f"| {g} | {te} | {th} | {f3(D.get(float(te)))} | {f3(D.get(float(th)))} | " + " | ".join(cells) + " |")
    w("")
    w("D(τ) = 各 true candidate について最も近い同群 competitor への期待 pointwise-projection KL（nats/観測）の平均（下界）。")
    w("")

    w("## 補助表 n=75 vs n=150（score-decided, family-level）")
    w("")
    w("| support group | A n=75 | A n=150 | B Auto(FIRST) n=75 | B Auto(FIRST) n=150 |")
    w("|---|---|---|---|---|")
    for g in ("COUNT", "POSITIVE", "BOUNDED"):
        cells = []
        for exp, meth in (("A", "SELECTOR_ONLY"), ("B", "AUTO_START_FIRST")):
            for n in ("75", "150"):
                cells.append(pct(one(fam, scope="score_decided_only", grouping="by_n",
                                     experiment=exp, stage="main", method=meth, group_design=g,
                                     n=n, support_group=SG[g], true_family="ALL")))
        w(f"| {g} | " + " | ".join(cells) + " |")
    w("")
    w("## 補助表 start agreement（START_FIRST vs START_LAST, main）")
    w("")
    w("| design | datasets | both succeeded | identical final assignment | per-column agreement |")
    w("|---|---|---|---|---|")
    for g in ("COUNT", "POSITIVE", "BOUNDED", "MIXED"):
        rows = [r for r in sa if r["stage"] == "main" and r["group_design"] == g]
        if not rows:
            continue
        both = [r for r in rows if r["both_succeeded"] == "True"]
        nc = sum(int(r["n_columns"]) for r in both)
        w(f"| {g} | {len(rows)} | {len(both)} | "
          f"{sum(r['full_assignment_agree'] == 'True' for r in both)}/{len(both)} | "
          f"{(sum(int(r['n_columns_agree']) for r in both) / nc) if nc else float('nan'):.3f} |")
    w("")
    w("## 補助表 candidate optimiser diagnostics（main; grad_inf > 1e-8 は WARNING）")
    w("")
    w("| experiment | candidate model | evaluations | failed | grad_inf>1e-8 | median grad_inf | max grad_inf |")
    w("|---|---|---|---|---|---|---|")
    for r in conv:
        if r["stage"] != "main":
            continue
        w(f"| {r['experiment']} | {r['candidate_model']} | {r['n_candidate_evaluations']} | "
          f"{r['n_failed']} | {r['n_grad_inf_above_1e-8_WARNING']} | "
          f"{float(r['grad_inf_median']):.2e} | {float(r['grad_inf_max']):.2e} |")
    w("")
    text = "\n".join(L) + "\n"
    out = REPO / "reports" / "distribution_selection" / f"seminar_family_selection_summary_{a.date}.md"
    out.write_text(text, encoding="utf-8")
    (run / "audit" / out.name).write_text(text, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
