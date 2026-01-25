#!/usr/bin/env python3
"""
Figure 4: Rank sweep showing RFI leakage vs science loss across k.
Reads a wide-format CSV with columns like:
  k, leak_median, dist_median
or
  k, bias_under, bias_over
If CSV has no label, pass --label to treat it as single label.

Saves: ../outputs/figure4_rank_sweep.png
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: str, default_label: str | None = None):
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if raw.size == 0:
        raise ValueError(f"Empty CSV: {path}")
    names = set(raw.dtype.names or [])
    if "k" not in names:
        raise ValueError("CSV must contain column 'k'")
    if "leak_median" in names and "dist_median" in names:
        leak_col, dist_col = "leak_median", "dist_median"
    elif "bias_under" in names and "bias_over" in names:
        leak_col, dist_col = "bias_under", "bias_over"
    else:
        # maybe long-format with label
        if "label" in names:
            # convert to structured arrays grouped by label
            data = {}
            for row in raw:
                lab = str(row['label'])
                k = int(row['k'])
                leak = float(row['leak']) if 'leak' in names else float(row.get('bias_under', np.nan))
                dist = float(row['dist']) if 'dist' in names else float(row.get('bias_over', np.nan))
                data.setdefault(lab, []).append((k, leak, dist))
            return {lab: np.array(rows, dtype=[('k','i4'),('leak','f8'),('dist','f8')]) for lab, rows in data.items()}
        raise ValueError("CSV must contain (leak_median,dist_median) or (bias_under,bias_over) or be long-format with 'label'.")

    # build simple array
    rows = []
    for i in range(raw.shape[0] if raw.ndim > 0 else 1):
        r = raw[i] if raw.ndim > 0 else raw
        k = int(r['k'])
        leak = float(r[leak_col])
        dist = float(r[dist_col])
        rows.append((k, leak, dist))
    arr = np.array(rows, dtype=[('k','i4'), ('leak','f8'), ('dist','f8')])
    label = default_label or os.path.basename(path)
    return {label: arr}


def plot_rank_sweep(grouped, outpath, title):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
    })
    fig, ax = plt.subplots(figsize=(8,5))

    for lab, arr in grouped.items():
        arr = arr[np.argsort(arr['k'])]
        k = arr['k']
        leak = arr['leak']
        dist = arr['dist']
        # plot both curves on same axis (log Y)
        ax.plot(k, leak, marker='o', lw=1.6, label=f'{lab} — RFI leakage')
        ax.plot(k, dist, marker='s', lw=1.6, label=f'{lab} — science loss')

    ax.set_yscale('log')
    ax.set_xlabel('Rank k')
    ax.set_ylabel('Proxy (lower is better; log scale)')
    ax.set_title(title)
    ax.grid(True, which='both', ls=':', alpha=0.6)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--label', default=None)
    ap.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    grouped = read_csv(args.csv, default_label=args.label)
    outpath = os.path.join(args.outdir, 'figure4_rank_sweep.png')
    plot_rank_sweep(grouped, outpath, 'Figure 4 — Rank sweep: RFI leakage vs science loss')
    print(f'WROTE: {outpath}')

if __name__ == '__main__':
    main()
