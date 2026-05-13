#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    if df.empty:
        raise SystemExit("empty results")

    # Figure 1: S_win vs flux, background variance per baseline group.
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, sub in df.groupby("baseline_group"):
        g = sub.groupby("S_ref_jy")["S_win_abs_delta_power_db"]
        x = np.array(sorted(g.groups.keys()), dtype=float)
        med = np.array([g.get_group(v).median() for v in x])
        q25 = np.array([g.get_group(v).quantile(0.25) for v in x])
        q75 = np.array([g.get_group(v).quantile(0.75) for v in x])
        ax.plot(x, med, marker="o", label=str(group))
        ax.fill_between(x, q25, q75, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel(r"$S_{ref}$ [Jy]")
    ax.set_ylabel(r"$S_{win}^{(|\Delta P|)}$ [dB]")
    ax.axvspan(10, 100, alpha=0.08, label="realistic-tier anchor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig01_swin_vs_flux.png", dpi=200)
    plt.close(fig)

    # Figure 2: observed minus null p95.
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, sub in df.groupby("baseline_group"):
        g = sub.groupby("S_ref_jy")["obs_minus_null_p95_db"]
        x = np.array(sorted(g.groups.keys()), dtype=float)
        med = np.array([g.get_group(v).median() for v in x])
        q25 = np.array([g.get_group(v).quantile(0.25) for v in x])
        q75 = np.array([g.get_group(v).quantile(0.75) for v in x])
        ax.plot(x, med, marker="o", label=str(group))
        ax.fill_between(x, q25, q75, alpha=0.18)
    ax.axhline(0.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$S_{ref}$ [Jy]")
    ax.set_ylabel(r"$S_{obs}-S_{null,p95}$ [dB]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig02_null_excess_vs_flux.png", dpi=200)
    plt.close(fig)

    # Figure 3: null distribution diagnostic for saved null arrays.
    nulls = list(Path(args.results).parent.glob("null_*.npy"))[:12]
    for p in nulls:
        vals = np.load(p)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=30)
        ax.axvline(np.percentile(vals, 50), linestyle="--", label="median")
        ax.axvline(np.percentile(vals, 95), linestyle="-", label="p95")
        ax.set_xlabel(r"null $S_{win}^{(|\Delta P|)}$ [dB]")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"hist_{p.stem}.png", dpi=160)
        plt.close(fig)

    print(f"figures saved under: {out}")


if __name__ == "__main__":
    main()
