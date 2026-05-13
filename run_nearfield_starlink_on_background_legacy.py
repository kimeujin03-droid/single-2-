#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import subprocess
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--selected", default="hera_catalog/final_pilot_baselines.csv")
p.add_argument("--pol", default="ee")
args = p.parse_args()
sel = pd.read_csv(args.selected)
Path("backgrounds").mkdir(exist_ok=True)
for b in sel["baseline"].astype(str):
    uvh5 = Path(f"hera_downloads/zen.LST.baseline.{b}.sum.uvh5")
    out = Path(f"backgrounds/bg_{b}_{args.pol}.npz")
    if not uvh5.exists():
        print("MISSING UVH5:", uvh5)
        continue
    if out.exists():
        print("exists:", out)
        continue
    print("exporting:", b)
    subprocess.run(["python", "scripts/export_uvh5_to_pathb_npz.py", "--uvh5", str(uvh5), "--out", str(out), "--pol", args.pol], check=True)
