#!/usr/bin/env python3
# make_pareto_overlays.py
"""
Build publication-friendly Pareto plots from rank-sweep CSVs.

Supports two input formats:

(A) "long" format (your latest):
    label,k,leak,dist
    (plus any extra columns are ignored)

(B) "wide" format (svd_diagnostics-style):
    k,bias_under,bias_over
    OR
    k,leak_median,dist_median,(optional *_low *_high)
    In this case you pass --label <name> and it will treat the file as one label.

Outputs (in outdir):
  1) pareto_front_overlay.png
     - scatter points for each label (k encoded by colormap)
     - Pareto front for each label highlighted (line)
     - optional: baseline_2 k=1 marked as outlier (or excluded)

  2) delta_injected_vs_base.png
     - Δleak(k) and Δdist(k): injected - baseline vs k (requires both labels)

  3) pareto_overlay_excluding_outlier.png (if you set --exclude-outlier)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------- utils -------------------------------------------
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_csv_any(path: str, default_label: str | None = None) -> np.ndarray:
    """
    Read CSV and normalize into a structured array with fields:
      label (str), k (int), leak (float), dist (float)

    If 'label' column is missing, uses default_label.
    Accepts either:
      - label,k,leak,dist
      - k,bias_under,bias_over  (maps to leak=bias_under, dist=bias_over)
      - k,leak_median,dist_median (maps to leak=leak_median, dist=dist_median)
    """
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    if raw.size == 0:
        raise ValueError(f"Empty CSV: {path}")

    names = set(raw.dtype.names or [])
    if "k" not in names:
        raise ValueError(f"CSV must contain column 'k': {path} (found: {sorted(names)})")

    # resolve leak/dist columns
    if "leak" in names and "dist" in names:
        leak_col, dist_col = "leak", "dist"
    elif "bias_under" in names and "bias_over" in names:
        leak_col, dist_col = "bias_under", "bias_over"
    elif "leak_median" in names and "dist_median" in names:
        leak_col, dist_col = "leak_median", "dist_median"
    else:
        raise ValueError(
            f"CSV must contain (leak,dist) or (bias_under,bias_over) or (leak_median,dist_median): {path}\n"
            f"Found: {sorted(names)}"
        )

    # resolve label
    has_label = "label" in names
    if not has_label and not default_label:
        raise ValueError(
            f"CSV has no 'label' column. Provide --label for single-file inputs: {path}"
        )

    # normalize into consistent dtype
    rows = []
    for i in range(raw.shape[0] if raw.ndim > 0 else 1):
        r = raw[i] if raw.ndim > 0 else raw
        label = r["label"] if has_label else default_label
        k = int(r["k"])
        leak = float(r[leak_col])
        dist = float(r[dist_col])
        if not np.isfinite(leak) or not np.isfinite(dist):
            continue
        rows.append((str(label), k, leak, dist))

    out = np.array(rows, dtype=[("label", "U128"), ("k", "i4"), ("leak", "f8"), ("dist", "f8")])
    if out.size == 0:
        raise ValueError(f"No valid rows after parsing: {path}")
    return out


def pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Compute the non-dominated Pareto front for minimization in both dimensions:
      minimize leak AND dist.

    Returns a boolean mask same length as points (True = on Pareto front).
    Complexity O(n log n) by sorting leak and scanning for best dist.
    """
    leak = points["leak"]
    dist = points["dist"]

    order = np.argsort(leak, kind="mergesort")
    best = np.inf
    front = np.zeros(points.shape[0], dtype=bool)

    # scan increasing leak; keep a point if it improves (strictly) the best dist so far
    for idx in order:
        d = dist[idx]
        if d < best:
            front[idx] = True
            best = d

    return front


def pareto_mask_minimize(x, y):
    """
    Pareto mask for 2D minimization: keep points not dominated by any other.
    Dominance: (xj<=xi and yj<=yi) and at least one strict.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        if np.any(dom):
            keep[i] = False
    return keep

def group_by_label(data: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, List[Tuple[str, int, float, float]]] = {}
    for row in data:
        out.setdefault(row["label"], []).append((row["label"], row["k"], row["leak"], row["dist"]))
    grouped = {}
    for lab, rows in out.items():
        arr = np.array(rows, dtype=data.dtype)
        # stable sort by k for consistent delta plots
        arr = arr[np.argsort(arr["k"], kind="mergesort")]
        grouped[lab] = arr
    return grouped


def ensure_outdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ---------------------------- plotting ----------------------------------------

def plot_pareto_overlay(
    grouped: Dict[str, np.ndarray],
    out_path: str,
    title: str,
    mark_outlier: Tuple[str, int] | None = None,
    exclude_outlier: bool = False,
) -> None:
    # Use the user's simpler Pareto mask & plotting style per-label
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 160,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10.6, 7.2))

    # global ranges for axis limits
    all_leak = np.concatenate([arr["leak"] for arr in grouped.values()])
    all_dist = np.concatenate([arr["dist"] for arr in grouped.values()])
    eps = 1e-12
    all_leak = np.clip(all_leak, eps, None)
    all_dist = np.clip(all_dist, eps, None)
    xmin, xmax = float(np.min(all_leak)), float(np.max(all_leak))
    ymin, ymax = float(np.min(all_dist)), float(np.max(all_dist))
    xmin *= 0.85; xmax *= 1.15
    ymin *= 0.85; ymax *= 1.15

    last_sc = None
    for lab, arr in grouped.items():
        leak = np.clip(arr["leak"], eps, None)
        dist = np.clip(arr["dist"], eps, None)
        k = arr["k"]

        # optionally exclude outlier point
        if exclude_outlier and mark_outlier and lab == mark_outlier[0]:
            mask_keep = ~(k == mark_outlier[1])
            leak_p = leak[mask_keep]
            dist_p = dist[mask_keep]
            k_p = k[mask_keep]
        else:
            leak_p, dist_p, k_p = leak, dist, k

        # faint scatter for all points
        sc = ax.scatter(leak_p, dist_p, c=k_p, cmap="viridis", s=28, alpha=0.35, linewidths=0, label=lab)
        last_sc = sc

        # compute Pareto front using the mask
        mask = pareto_mask_minimize(leak_p, dist_p)
        leak_pf = leak_p[mask]
        dist_pf = dist_p[mask]
        k_pf = k_p[mask]

        # sort front by leak for a tidy polyline
        order = np.argsort(leak_pf)
        leak_pf, dist_pf, k_pf = leak_pf[order], dist_pf[order], k_pf[order]

        # bold Pareto front line and markers
        ax.plot(leak_pf, dist_pf, linewidth=2.5, marker="o", markersize=6)

        # annotate each front point with its k value
        for kv, lx, dy in zip(k_pf, leak_pf, dist_pf):
            ax.annotate(f"k={int(kv)}", (max(lx, eps), max(dy, eps)), textcoords="offset points", xytext=(6, 6), fontsize=9, alpha=0.9)

    # mark outlier if requested
    if mark_outlier:
        lab0, k0 = mark_outlier
        if lab0 in grouped:
            arr = grouped[lab0]
            hit = np.where(arr["k"] == k0)[0]
            if hit.size > 0 and not exclude_outlier:
                i = int(hit[0])
                ax.scatter([max(arr["leak"][i], eps)], [max(arr["dist"][i], eps)], s=180, marker="*", edgecolor="k", linewidths=0.6, zorder=10)
                ax.annotate(f"outlier: {lab0}, k={k0}", (max(arr["leak"][i], eps), max(arr["dist"][i], eps)), textcoords="offset points", xytext=(8, 8), fontsize=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("RFI leakage proxy (lower is better; log scale)")
    ax.set_ylabel("Science distortion proxy (lower is better; log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.8, alpha=0.6)
    ax.legend(loc="best", frameon=True)

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=ax, pad=0.02)
        cbar.set_label("Rank k")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta_injected_vs_base(
    grouped: Dict[str, np.ndarray],
    base_label: str,
    injected_label: str,
    out_path: str,
) -> None:
    if base_label not in grouped or injected_label not in grouped:
        raise ValueError(f"Need both labels for delta plot: {base_label}, {injected_label}")

    base = grouped[base_label]
    inj = grouped[injected_label]

    # join by k
    kb = base["k"]
    ki = inj["k"]
    common = np.intersect1d(kb, ki)
    if common.size == 0:
        raise ValueError("No overlapping k values between baseline and injected.")

    # build maps
    b_map = {int(k): (float(le), float(di)) for k, le, di in zip(base["k"], base["leak"], base["dist"])}
    i_map = {int(k): (float(le), float(di)) for k, le, di in zip(inj["k"], inj["leak"], inj["dist"])}

    ks = common.astype(int)
    d_leak = np.array([i_map[int(k)][0] - b_map[int(k)][0] for k in ks], dtype=float)
    d_dist = np.array([i_map[int(k)][1] - b_map[int(k)][1] for k in ks], dtype=float)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 160,
    })

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=True)

    axes[0].plot(ks, d_leak, marker="o", lw=1.6)
    axes[0].axhline(0.0, lw=1.0, ls="--", alpha=0.6)
    axes[0].set_ylabel("Δleak (inj − base)")
    axes[0].grid(True, ls=":", alpha=0.6)

    axes[1].plot(ks, d_dist, marker="o", lw=1.6)
    axes[1].axhline(0.0, lw=1.0, ls="--", alpha=0.6)
    axes[1].set_ylabel("Δdist (inj − base)")
    axes[1].set_xlabel("Rank k")
    axes[1].grid(True, ls=":", alpha=0.6)

    fig.suptitle(f"Injected effect vs baseline: {injected_label} − {base_label}", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------- main --------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV path (long-format with label OR single-label wide-format)")
    ap.add_argument("--label", default=None, help="If CSV has no 'label' column, treat it as this single label")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--base-label", default="baseline_0", help="Baseline label for delta plot")
    ap.add_argument("--injected-label", default="baseline_0_injected_comb", help="Injected label for delta plot")
    ap.add_argument("--mark-outlier-label", default="baseline_2", help="Label that has an outlier point to mark/exclude")
    ap.add_argument("--mark-outlier-k", type=int, default=1, help="k value of the outlier point")
    ap.add_argument("--exclude-outlier", action="store_true", help="Exclude the outlier point in an additional overlay plot")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    data = read_csv_any(args.csv, default_label=args.label)
    grouped = group_by_label(data)

    out1 = os.path.join(args.outdir, "pareto_front_overlay.png")
    plot_pareto_overlay(
        grouped,
        out_path=out1,
        title="Pareto overlay (scatter by k + Pareto fronts)",
        mark_outlier=(args.mark_outlier_label, args.mark_outlier_k),
        exclude_outlier=False,
    )
    print(f"[OK] wrote: {out1}")

    # delta plot (only if both labels exist)
    if args.base_label in grouped and args.injected_label in grouped:
        out2 = os.path.join(args.outdir, "delta_injected_vs_base.png")
        plot_delta_injected_vs_base(
            grouped,
            base_label=args.base_label,
            injected_label=args.injected_label,
            out_path=out2,
        )
        print(f"[OK] wrote: {out2}")
    else:
        print(f"[SKIP] delta plot: missing labels ({args.base_label}, {args.injected_label}) in input.")

    if args.exclude_outlier:
        out3 = os.path.join(args.outdir, "pareto_overlay_excluding_outlier.png")
        plot_pareto_overlay(
            grouped,
            out_path=out3,
            title=f"Pareto overlay (excluding outlier: {args.mark_outlier_label}, k={args.mark_outlier_k})",
            mark_outlier=(args.mark_outlier_label, args.mark_outlier_k),
            exclude_outlier=True,
        )
        print(f"[OK] wrote: {out3}")


if __name__ == "__main__":
    main()
