"""Checks every link in docs/research_results/*.md.

- GitHub permalinks .../blob/<sha>/<path> and .../tree/<sha>/<path> must resolve with `git cat-file -e <sha>:<path>`.
- Relative links must point to an existing file in this directory tree (anchors are not checked).
Usage (repository root): python docs/research_results/scripts/check_links.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GH = re.compile(r"https://github\.com/H-T-0726/expfam-lsm-attributed-relational-data/(?:blob|tree)/([0-9a-f]{7,40})/([^)\s#]+)")
REL = re.compile(r"\]\((?!https?://|#)([^)#\s]+)")
bad = 0
n = 0
for md in sorted(ROOT.glob("*.md")):
    text = md.read_text(encoding="utf-8")
    for sha, path in GH.findall(text):
        n += 1
        if subprocess.run(["git", "cat-file", "-e", f"{sha}:{path}"], capture_output=True).returncode:
            print(f"BROKEN {md.name}: {sha}:{path}")
            bad += 1
    for rel in REL.findall(text):
        n += 1
        if not (md.parent / rel).exists():
            print(f"BROKEN {md.name}: {rel}")
            bad += 1
print(f"checked {n} links, broken {bad}")
sys.exit(1 if bad else 0)
