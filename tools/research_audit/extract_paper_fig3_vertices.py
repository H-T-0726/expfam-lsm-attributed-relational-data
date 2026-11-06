"""Extract the vector polylines of Fig. 3 (Mikawa et al. 2024, NOLTA 15(2), p.345).

Fig. 3 is stored as vector paths in the PDF, so the plotted BIC values can be
read from the path vertices instead of by eye.  This script needs PyMuPDF,
which is NOT in requirements*.txt; it was run once with a scratch install
(version recorded in the output) and its CSV is committed.  The main audit
script reads only the CSV.

Calibration: the y grid lines are located by scanning a raster of the plot
area for light-gray rows; their values (1000..5000) are the printed tick
labels, read from the rendered page.  x vertices are at k = 1..10 in order.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "paper" / "A_study_on_latent_structural_models_for_binary_rel.pdf"
OUT = ROOT / "reports/k_selection_theory/historical_code_lineage_audit_20261106/paper_fig3_vector_vertices.csv"

PAGE_INDEX = 10                      # printed page 345
FIG_CLIP = (200.0, 210.0, 410.0, 366.0)
PLOT_RECT = (221.3, 221.1, 393.3, 349.3)
GRID_VALUES = [5000.0, 4000.0, 3000.0, 2000.0, 1000.0]   # top to bottom
LEGEND = {(0.0, 0.0, 0.0): 1, (0.0, 0.0, 1.0): 3, (1.0, 0.0, 0.0): 5,
          (0.0, 0.5, 0.0): 7, (0.12, 0.47, 0.71): 9}


def gridlines(page) -> list[float]:
    z = 8
    clip = pymupdf.Rect(*PLOT_RECT)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=clip)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
    gray = (np.abs(a[:, :, 0] - a[:, :, 1]) < 8) & (a[:, :, 0] < 235) & (a[:, :, 0] > 120)
    rows = [i for i, v in enumerate(gray.mean(1)) if v > 0.5]
    groups: list[list[int]] = []
    for r in rows:
        if groups and r - groups[-1][-1] <= 2:
            groups[-1].append(r)
        else:
            groups.append([r])
    ys = [clip.y0 + float(np.mean(g)) / z for g in groups]
    inner = [y for y in ys if clip.y0 + 2 < y < clip.y1 - 2]   # drop the frame
    if len(inner) != 5:
        sys.exit(f"expected 5 interior grid lines, got {inner}")
    return inner


def main() -> None:
    doc = pymupdf.open(PDF)
    page = doc[PAGE_INDEX]
    grid = gridlines(page)
    slope, icpt = np.polyfit(grid, GRID_VALUES, 1)
    resid = max(abs(slope * y + icpt - v) for y, v in zip(grid, GRID_VALUES))
    fig = pymupdf.Rect(*FIG_CLIP)
    rows = []
    for p in page.get_drawings():
        if p.get("type") != "s" or not fig.intersects(p["rect"]):
            continue
        col = tuple(round(c, 2) for c in (p.get("color") or ()))
        if col not in LEGEND:
            continue
        pts: list[tuple[float, float]] = []
        for it in p["items"]:
            if it[0] != "l":
                continue
            for q in (it[1], it[2]):
                if not pts or abs(pts[-1][0] - q.x) > 1e-6 or abs(pts[-1][1] - q.y) > 1e-6:
                    pts.append((q.x, q.y))
        if len(pts) < 8:
            continue
        xs = sorted({round(x, 2) for x, _ in pts})
        step = 17.3711       # median spacing of the 10 x positions (pt)
        x1 = min(xs)
        for x, y in pts:
            k = 1 + round((x - x1) / step)
            rows.append({"k_star": LEGEND[col], "k": k, "x_pt": round(x, 3), "y_pt": round(y, 3),
                         "bic_read": round(float(slope * y + icpt), 2)})
    rows.sort(key=lambda r: (r["k_star"], r["k"]))
    meta = {"pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(), "pymupdf": pymupdf.__version__,
            "grid_y_pt": [round(g, 3) for g in grid], "calibration_max_residual": round(resid, 3),
            "bic_units_per_pt": round(-slope, 3)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        sys.exit(f"refusing to overwrite {OUT}")
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# " + repr(meta) + "\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(meta, len(rows))


if __name__ == "__main__":
    main()
