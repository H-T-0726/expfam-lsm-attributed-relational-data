"""No-fit diagnostic: per-iteration plug-in log likelihood saved in the OLD states (plugin_by_iter, 8 iterations),
weak n=75 all-pairs study and, for contrast, the PR #161 strong n=75 OLD states (Phase B).  Reports the change over
the last 3 iterations relative to the total change.  Lineage E states store only the final iteration, so no E row.
Writes convergence_diagnostic.csv.  Refuses to overwrite."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_audit"))
import weak_protocol as wp  # noqa: E402

out = ROOT / wp.RESULTS_REL / "convergence_diagnostic.csv"
if out.exists():
    raise SystemExit("refusing to overwrite")
rows = []
for study, root, pat, reps, ks in (("weak n=75 all pairs (this study)", Path(wp.STATE_ROOT) / "W", "W_r{r}_k{k}_s{s}_OLD", (1, 2, 3, 4), range(1, 8)),
                                   ("strong n=75 all pairs (PR #161 B)", Path(r"D:\tento\kennkyu_artifacts\n75_old_vs_expfam_controlled_20261106") / "B",
                                    "B_r{r}_k{k}_s{s}_OLD", (1, 2, 3, 4), range(1, 8))):
    for r in reps:
        for k in ks:
            for s in (1, 2):
                p = np.load(root / (pat.format(r=r, k=k, s=s) + ".npz"))["plugin_by_iter"]
                tot = p[-1] - p[0]
                rows.append(dict(study=study, rep=r, k=k, s=s, plugin_iter1=p[0], plugin_iter8=p[-1],
                                 change_last3=p[-1] - p[-4], change_last1=p[-1] - p[-2],
                                 frac_change_in_last3=(p[-1] - p[-4]) / tot if tot != 0 else np.nan))
df = pd.DataFrame(rows)
df.to_csv(out, index=False, lineterminator="\n")
print(df.groupby(["study", "k"])[["change_last1", "change_last3", "frac_change_in_last3"]].median().round(3).to_string())
