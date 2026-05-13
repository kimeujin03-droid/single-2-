#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def mad(x):
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

rows = []
for p in sorted(Path("backgrounds").glob("*.npz")):
    d = np.load(p, allow_pickle=True)
    vis = d["vis_tf"]
    weights = d["weights_tf"] if "weights_tf" in d.files else np.ones(vis.shape)
    flags = d["flags_tf"] if "flags_tf" in d.files else weights <= 0
    bl = d["baseline_enu_m"]
    amp = np.abs(vis)
    valid = np.isfinite(amp) & (weights > 0)
    amp_valid = amp[valid]
    if amp_valid.size == 0:
        continue
    baseline = p.stem.replace("bg_", "").replace("_ee", "")
    L = float(np.linalg.norm(bl))
    row = {
        "file": str(p), "baseline": baseline,
        "baseline_length_m": L, "baseline_e_m": float(bl[0]), "baseline_n_m": float(bl[1]), "baseline_u_m": float(bl[2]),
        "orientation_deg": float(np.degrees(np.arctan2(bl[1], bl[0]))),
        "n_time": int(vis.shape[0]), "n_freq": int(vis.shape[1]),
        "flag_fraction": float(np.mean(flags)), "valid_fraction": float(np.mean(valid)), "nan_fraction": float(np.mean(~np.isfinite(amp))),
        "median_abs_vis": float(np.nanmedian(amp_valid)), "mad_abs_vis": float(mad(amp_valid)),
        "p95_abs_vis": float(np.nanpercentile(amp_valid, 95)),
        "p99_abs_vis": float(np.nanpercentile(amp_valid, 99)),
        "p999_abs_vis": float(np.nanpercentile(amp_valid, 99.9)),
        "length_group": "short" if L < 35 else ("mid" if L < 90 else "long"),
    }
    row["p99_over_median"] = row["p99_abs_vis"] / max(row["median_abs_vis"], 1e-30)
    row["p999_over_median"] = row["p999_abs_vis"] / max(row["median_abs_vis"], 1e-30)
    row["mad_over_median"] = row["mad_abs_vis"] / max(row["median_abs_vis"], 1e-30)
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No NPZ files found in backgrounds/ or no valid samples.")
score_cols = ["flag_fraction", "nan_fraction", "mad_over_median", "p99_over_median", "p999_over_median"]
terms = []
for col in score_cols:
    if df[col].nunique(dropna=False) > 1:
        rcol = col + "_rankpct"
        df[rcol] = df[col].rank(pct=True)
        terms.append(rcol)
df["qc_noise_score_ranked"] = df[terms].mean(axis=1) if terms else 0.5
if len(df) >= 3:
    df["noise_tier_ranked"] = pd.qcut(df["qc_noise_score_ranked"], q=3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
else:
    df["noise_tier_ranked"] = "mid"
Path("hera_catalog").mkdir(exist_ok=True)
df = df.sort_values(["baseline_length_m"])
df.to_csv("hera_catalog/background_qc_summary_ranked.csv", index=False)
print(df[["baseline", "length_group", "noise_tier_ranked", "baseline_length_m", "orientation_deg", "flag_fraction", "median_abs_vis", "p99_over_median", "qc_noise_score_ranked"]].to_string(index=False))
print("saved hera_catalog/background_qc_summary_ranked.csv")
