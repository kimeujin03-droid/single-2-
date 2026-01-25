#!/usr/bin/env python3
"""
Figure 7: HERA single-snapshot + top singular modes (reproduce synthetic layout)
 - left: time vs frequency snapshot (image)
 - right: stacked top frequency profiles (V^T rows) with explained variance

Saves: ../outputs/figure7_hera_modes.png
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import pandas as pd
    import hera_rank_sweep as hrs
except Exception:
    hrs = None

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)
OUTPATH = os.path.join(OUTDIR, 'figure7_hera_modes.png')


def load_first_snapshot(path):
    import pandas as pd
    payload = pd.read_pickle(path)
    # Try helpers from hera_rank_sweep if available
    if hrs is not None:
        try:
            # first try extracting slices
            slices = hrs.extract_slices_from_payload(payload, n_slices=1)
            if len(slices) > 0:
                return slices[0]
        except Exception:
            pass
        try:
            # try to extract a 2D matrix directly
            return hrs._extract_2d_matrix(payload)
        except Exception:
            pass

    # Fallback strategies: if payload is dict, search for first array-like value
    if isinstance(payload, dict):
        for v in payload.values():
            try:
                arr = np.asarray(v)
                if arr.ndim >= 2:
                    return np.squeeze(arr[0]) if arr.ndim >= 3 else np.squeeze(arr)
            except Exception:
                continue

    # If list-like, try the first array-like element
    if isinstance(payload, (list, tuple)) and len(payload) > 0:
        for el in payload:
            try:
                arr = np.asarray(el)
                if arr.ndim >= 2:
                    return np.squeeze(arr[0]) if arr.ndim >= 3 else np.squeeze(arr)
            except Exception:
                continue

    # Last resort: try to coerce to ndarray and pick the first 2D slice
    try:
        arr = np.asarray(payload)
        if arr.ndim >= 3:
            return np.squeeze(arr[0])
        elif arr.ndim == 2:
            return np.squeeze(arr)
    except Exception:
        pass

    raise RuntimeError('Unable to extract a 2D snapshot from the provided HERA pickle payload')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'HERA_04-03-2022_all.pkl'))
    ap.add_argument('--n_modes', type=int, default=5)
    ap.add_argument('--out', type=str, default=OUTPATH)
    args = ap.parse_args()

    X = load_first_snapshot(args.path)
    X = np.asarray(X, dtype=float)

    # SVD
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    explained = (s ** 2) / np.sum(s ** 2)
    modes = Vt[: args.n_modes]

    # normalize modes for stacked plotting
    modes_norm = modes.copy()
    for i in range(len(modes_norm)):
        m = modes_norm[i]
        if np.max(np.abs(m)) > 0:
            modes_norm[i] = m / np.max(np.abs(m))

    # plot layout: left image, right stacked modes
    fig = plt.figure(figsize=(10, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.2, 1.0), wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(X.T, aspect='auto', origin='lower', cmap='inferno')
    ax0.set_xlabel('time index')
    ax0.set_ylabel('frequency channel')
    ax0.set_title('HERA snapshot (time × frequency)')
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[0, 1])
    f = np.linspace(0, 1, X.shape[1])
    offset = 1.4
    colors = plt.cm.tab10(np.arange(args.n_modes))
    for i in range(args.n_modes):
        y = modes_norm[i] + i * offset
        ax1.plot(f, y, color=colors[i], lw=1.6)
        ax1.text(0.98, i * offset + 0.05, f'k={i+1} ({100*explained[i]:.1f}%)', ha='right', va='bottom', fontsize=9)

    ax1.set_yticks([])
    ax1.set_xlabel('frequency (normalized)')
    ax1.set_title('Top singular-frequency profiles (stacked)')
    ax1.axvspan(0.4, 0.48, color='cyan', alpha=0.12)

    fig.suptitle('Figure 7 — HERA snapshot and top singular modes', y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('WROTE:', args.out)


if __name__ == '__main__':
    main()
