#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from pyuvdata import UVData

CATALOG_LONG = Path("hera_catalog/h6c_idr2_pspec_files_long.csv")
FRF_PKL = Path("hera_downloads/merged/baselines_merged.tavg.frf_losses.pkl")
REF_UVH5 = Path("hera_downloads/zen.LST.baseline.0_1.sum.uvh5")
OUT = Path("hera_catalog")
OUT.mkdir(exist_ok=True)

def get_antpos_enu_and_ants(uvd):
    if hasattr(uvd, "get_enu_antpos"):
        ret = uvd.get_enu_antpos()
    elif hasattr(uvd, "get_ENU_antpos"):
        ret = uvd.get_ENU_antpos()
    elif hasattr(uvd, "telescope") and hasattr(uvd.telescope, "get_enu_antpos"):
        ret = uvd.telescope.get_enu_antpos()
    else:
        raise AttributeError("No ENU antenna-position method found.")
    if isinstance(ret, tuple) and len(ret) >= 2:
        return np.asarray(ret[0]), np.asarray(ret[1])
    if isinstance(ret, np.ndarray):
        antpos_enu = np.asarray(ret)
        if hasattr(uvd, "telescope") and hasattr(uvd.telescope, "antenna_numbers"):
            ants = np.asarray(uvd.telescope.antenna_numbers)
        elif hasattr(uvd, "antenna_numbers"):
            ants = np.asarray(uvd.antenna_numbers)
        else:
            ants = np.array(sorted(set(map(int, uvd.ant_1_array)) | set(map(int, uvd.ant_2_array))))
        return antpos_enu, ants
    raise ValueError(f"Unexpected ENU return type: {type(ret)}")

def ew_distance_deg(theta):
    x = abs(float(theta)) % 180.0
    return min(x, abs(180.0 - x))

def length_group(L):
    return "short" if L < 35 else ("mid" if L < 90 else "long")

def load_catalog():
    df = pd.read_csv(CATALOG_LONG)
    df["baseline"] = df["baseline"].astype(str)
    piv = (df.assign(has=1)
           .pivot_table(index=["ant1", "ant2", "baseline"], columns="ftype", values="has", aggfunc="max", fill_value=0)
           .reset_index())
    for col in ["uvh5", "ipynb", "html", "pspec_h5", "tavg_pspec_h5", "tavg_window_pkl"]:
        if col not in piv.columns:
            piv[col] = 0
    for ftype in ["uvh5", "ipynb", "html", "tavg_pspec_h5"]:
        sub = df[df["ftype"] == ftype][["baseline", "filename", "url"]].copy()
        sub = sub.rename(columns={"filename": f"{ftype}_filename", "url": f"{ftype}_url"})
        piv = piv.merge(sub, on="baseline", how="left")
    return piv

def add_geometry(cat):
    uvd = UVData(); uvd.read(str(REF_UVH5), read_data=False)
    antpos_enu, ants = get_antpos_enu_and_ants(uvd)
    ant_to_pos = {int(a): antpos_enu[i] for i, a in enumerate(ants)}
    rows = []
    for _, r in cat.iterrows():
        a1, a2 = int(r["ant1"]), int(r["ant2"])
        if a1 not in ant_to_pos or a2 not in ant_to_pos:
            continue
        bl = ant_to_pos[a2] - ant_to_pos[a1]
        L = float(np.linalg.norm(bl))
        orient = float(np.degrees(np.arctan2(bl[1], bl[0])))
        row = r.to_dict()
        row.update({
            "baseline_e_m": float(bl[0]), "baseline_n_m": float(bl[1]), "baseline_u_m": float(bl[2]),
            "baseline_length_m": L, "orientation_deg": orient,
            "ew_angle_distance_deg": ew_distance_deg(orient), "length_group": length_group(L),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def load_frf():
    with open(FRF_PKL, "rb") as f:
        obj = pickle.load(f)
    rows = []
    for k, arr in obj.items():
        try:
            a1, a2 = k[0]
            baseline = f"{int(a1)}_{int(a2)}"
        except Exception:
            continue
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        rows.append({
            "baseline": baseline,
            "frf_loss_median": float(np.nanmedian(finite)),
            "frf_loss_p95": float(np.nanpercentile(finite, 95)),
            "frf_loss_p99": float(np.nanpercentile(finite, 99)),
            "frf_loss_max": float(np.nanmax(finite)),
            "frf_loss_min": float(np.nanmin(finite)),
            "frf_loss_has_negative": bool(np.nanmin(finite) < 0),
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["baseline"])

def assign_groupwise_tier(df):
    df = df.copy()
    df["frf_noise_tier_groupwise"] = "unknown"
    df["frf_qc_rank_within_group"] = np.nan
    for _, sub in df.groupby("length_group"):
        sub = sub.copy()
        sub["frf_qc_rank_within_group"] = sub["frf_qc_score_raw"].rank(pct=True)
        if len(sub) >= 3 and sub["frf_qc_score_raw"].nunique() >= 3:
            tiers = pd.qcut(sub["frf_qc_score_raw"], q=3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
        else:
            tiers = pd.Series(["mid"] * len(sub), index=sub.index)
        df.loc[sub.index, "frf_noise_tier_groupwise"] = tiers
        df.loc[sub.index, "frf_qc_rank_within_group"] = sub["frf_qc_rank_within_group"]
    return df

def main():
    cat = add_geometry(load_catalog())
    frf = load_frf()
    df = cat.merge(frf, on="baseline", how="left")
    df["has_frf"] = df["frf_loss_median"].notna()
    df["complete_score"] = df[["uvh5", "ipynb", "html", "pspec_h5", "tavg_pspec_h5"]].sum(axis=1)
    eligible = df[(df["uvh5"] == 1) & (df["has_frf"]) & (df["baseline_length_m"] >= 10)].copy()
    ew10 = eligible[eligible["ew_angle_distance_deg"] <= 10].copy()
    ew30 = eligible[eligible["ew_angle_distance_deg"] <= 30].copy()
    if len(ew10) >= 9:
        work, orientation_mode = ew10, "EW_like_le_10deg"
    elif len(ew30) >= 9:
        work, orientation_mode = ew30, "near_EW_le_30deg"
    else:
        work, orientation_mode = eligible, "all_orientations"
    work["frf_qc_score_raw"] = (work["frf_loss_median"].rank(pct=True) + work["frf_loss_p95"].rank(pct=True) + work["frf_loss_max"].rank(pct=True)) / 3.0
    work = assign_groupwise_tier(work)
    full_out = OUT / "catalog_frf_candidates_groupwise.csv"
    work.sort_values(["length_group", "baseline_length_m", "frf_qc_rank_within_group"]).to_csv(full_out, index=False)
    selected = []
    for lg in ["short", "mid", "long"]:
        sub_l = work[work["length_group"] == lg].copy()
        for tier in ["low", "mid", "high"]:
            sub = sub_l[sub_l["frf_noise_tier_groupwise"] == tier].copy()
            if sub.empty:
                continue
            tier_med = sub["frf_qc_rank_within_group"].median()
            sub["tier_center_distance"] = (sub["frf_qc_rank_within_group"] - tier_med).abs()
            sub["selection_score"] = -2.0*sub["complete_score"] + 0.03*sub["ew_angle_distance_deg"] + 0.001*sub["baseline_length_m"] + sub["tier_center_distance"]
            selected.append(sub.sort_values("selection_score").iloc[0])
    sel = pd.DataFrame(selected).drop_duplicates(subset=["baseline"])
    out = OUT / "selected_baselines_groupwise_stratified.csv"
    sel.to_csv(out, index=False)
    cols = ["baseline", "length_group", "frf_noise_tier_groupwise", "baseline_length_m", "orientation_deg", "ew_angle_distance_deg", "frf_loss_median", "frf_loss_p95", "frf_loss_max", "frf_qc_rank_within_group", "complete_score", "uvh5", "ipynb", "html", "pspec_h5", "tavg_pspec_h5"]
    print("orientation_mode:", orientation_mode)
    print("eligible:", len(eligible), "work:", len(work))
    print("saved", full_out)
    print("saved", out)
    print("\n=== counts by length × groupwise tier ===")
    print(pd.crosstab(work["length_group"], work["frf_noise_tier_groupwise"]).to_string())
    print("\n=== GROUPWISE STRATIFIED SELECTION ===")
    print(sel[cols].to_string(index=False))

if __name__ == "__main__":
    main()
