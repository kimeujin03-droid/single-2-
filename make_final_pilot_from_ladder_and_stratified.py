#!/usr/bin/env python3
from __future__ import annotations
import re
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://data.nrao.edu/hera/Notebooks/H6C_IDR2/pspec/redavg-smoothcal-inpaint-500ns-lstcal/"
OUT_DIR = Path("hera_catalog")
OUT_DIR.mkdir(exist_ok=True)


def classify_file(name: str) -> str:
    if name.endswith(".uvh5"):
        return "uvh5"
    if name.endswith(".tavg.pspec.h5"):
        return "tavg_pspec_h5"
    if name.endswith(".pspec.h5"):
        return "pspec_h5"
    if name.endswith(".tavg.pspec.window.pkl"):
        return "tavg_window_pkl"
    if name.endswith(".single_baseline_postprocessing_and_pspec.html"):
        return "html"
    if name.endswith(".single_baseline_postprocessing_and_pspec.ipynb"):
        return "ipynb"
    return "other"


def main() -> None:
    html = requests.get(BASE_URL, timeout=120).text
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for a in soup.find_all("a"):
        name = a.get("href", "")
        if not name or name.startswith("?") or name == "../":
            continue
        m = re.search(r"zen\.LST\.baseline\.(\d+)_(\d+)\.sum", name)
        if not m:
            continue
        ant1, ant2 = map(int, m.groups())
        rows.append({
            "ant1": ant1,
            "ant2": ant2,
            "baseline": f"{ant1}_{ant2}",
            "filename": name,
            "url": urljoin(BASE_URL, name),
            "ftype": classify_file(name),
        })
    df = pd.DataFrame(rows).sort_values(["ant1", "ant2", "ftype"])
    if df.empty:
        raise SystemExit("No files found. Check internet or BASE_URL.")
    df.to_csv(OUT_DIR / "h6c_idr2_pspec_files_long.csv", index=False)
    piv = (df.assign(has=1)
             .pivot_table(index=["ant1", "ant2", "baseline"], columns="ftype", values="has", aggfunc="max", fill_value=0)
             .reset_index())
    for col in ["uvh5", "ipynb", "html", "pspec_h5", "tavg_pspec_h5", "tavg_window_pkl"]:
        if col not in piv.columns:
            piv[col] = 0
    piv["complete_core"] = (piv["uvh5"].eq(1) & piv["ipynb"].eq(1) & piv["tavg_pspec_h5"].eq(1))
    piv["download_priority_score"] = 5*piv["uvh5"] + 3*piv["ipynb"] + 2*piv["html"] + 2*piv["tavg_pspec_h5"] + piv["pspec_h5"]
    piv = piv.sort_values(["download_priority_score", "ant1", "ant2"], ascending=[False, True, True])
    piv.to_csv(OUT_DIR / "h6c_idr2_baseline_catalog.csv", index=False)
    print("saved", OUT_DIR / "h6c_idr2_pspec_files_long.csv")
    print("saved", OUT_DIR / "h6c_idr2_baseline_catalog.csv")
    print("n_files", len(df), "n_baselines", len(piv))
    print(piv.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
