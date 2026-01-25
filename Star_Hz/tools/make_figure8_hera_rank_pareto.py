#!/usr/bin/env python3
"""
Figure 8: HERA rank sweep + Pareto overlay (two-panel)
 - left: rank sweep (leakage vs k and distortion vs k) — similar to hera_rank_sweep figure
 - right: Pareto scatter (leak vs dist) with k labels

Saves: ../outputs/figure8_hera_rank_pareto.png
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

try:
    import pandas as pd
    import hera_rank_sweep as hrs
except Exception:
    hrs = None

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)
OUTPATH = os.path.join(OUTDIR, 'figure8_hera_rank_pareto.png')


def load_first_snapshot(path):
    import pandas as pd
    payload = pd.read_pickle(path)
    # Try helpers from hera_rank_sweep if available
    if hrs is not None:
        try:
            slices = hrs.extract_slices_from_payload(payload, n_slices=1)
            if len(slices) > 0:
                return slices[0]
        except Exception:
            pass
        try:
            return hrs._extract_2d_matrix(payload)
        except Exception:
            pass

    if isinstance(payload, dict):
        for v in payload.values():
            try:
                arr = np.asarray(v)
                if arr.ndim >= 2:
                    return np.squeeze(arr[0]) if arr.ndim >= 3 else np.squeeze(arr)
            except Exception:
                continue

    if isinstance(payload, (list, tuple)) and len(payload) > 0:
        for el in payload:
            try:
                arr = np.asarray(el)
                if arr.ndim >= 2:
                    return np.squeeze(arr[0]) if arr.ndim >= 3 else np.squeeze(arr)
            except Exception:
                continue

    try:
        arr = np.asarray(payload)
        if arr.ndim >= 3:
            return np.squeeze(arr[0])
        elif arr.ndim == 2:
            return np.squeeze(arr)
    except Exception:
        pass

    raise RuntimeError('Unable to extract a 2D snapshot from the provided HERA pickle payload')


def run_plot(path, K=50, out=OUTPATH):
    # load snapshot
    X = load_first_snapshot(path)
    X = np.asarray(X, dtype=float)

    # use existing run_rank_sweep_real for proxies if available
    if hrs is not None:
        ks, leak, dist = hrs.run_rank_sweep_real(X, max_k=K)
    else:
        # fallback: use same local proxy computation
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        max_k = min(K, min(X.shape))
        ks = np.arange(1, max_k + 1, dtype=int)
        leak = np.zeros_like(ks, dtype=float)
        dist = np.zeros_like(ks, dtype=float)
        norm_X = np.linalg.norm(X)
        for i, k in enumerate(ks):
            Lk = (U[:, :k] * s[:k][None, :]) @ Vt[:k, :]
            Ek = X - Lk
            sigma = float(np.std(Ek))
            thr = 5.0 * sigma
            leak[i] = float(np.mean(np.abs(Ek) > thr))
            preservation = np.linalg.norm(Ek) / (norm_X + 1e-12)
            preservation = float(min(1.0, max(0.0, preservation)))
            dist[i] = 1.0 - preservation

    # two-panel figure
    fig = plt.figure(figsize=(10.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.1, 1.0), wspace=0.28)

    # left: rank sweep (reuse style from hera_rank_sweep)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = ax1.twinx()
    ax1.plot(ks, leak, color='tab:blue', lw=1.8, marker='o', ms=3.0)
    ax2.plot(ks, dist, color='tab:orange', lw=1.8, marker='o', ms=3.0)
    ax1.set_xlabel('Rank k')
    ax1.set_ylabel('RFI leakage proxy', color='tab:blue')
    ax2.set_ylabel('Science distortion proxy', color='tab:orange')
    ax1.set_xlim(ks[0], ks[-1])
    ax1.set_xticks(ks[::2])
    fmt1 = ScalarFormatter(useMathText=True); fmt1.set_scientific(True); fmt1.set_powerlimits((0,0))
    ax1.yaxis.set_major_formatter(fmt1); ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fmt2 = ScalarFormatter(useMathText=True); fmt2.set_scientific(True); fmt2.set_powerlimits((0,0))
    ax2.yaxis.set_major_formatter(fmt2); ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.grid(True, which='both', ls=':', lw=0.7, alpha=0.6)
    ax1.set_title('(a) HERA rank sweep')

    def pareto_mask(x, y):
        # minimization pareto: True if no other point has both x<=xi and y<=yi with one strict
        x = np.asarray(x)
        y = np.asarray(y)
        n = x.size
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            # any j dominates i?
            dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
            if np.any(dominated):
                mask[i] = False
        return mask

    # right: Pareto scatter (color-coded by rank) with a rank colorbar
    axp = fig.add_subplot(gs[0, 1])
    sc = axp.scatter(dist, leak, c=ks, cmap='viridis', s=36, edgecolors='none', alpha=0.85)
    axp.set_xscale('log')
    axp.set_yscale('log')
    axp.set_xlabel('Science distortion proxy (log)')
    axp.set_ylabel('RFI leakage proxy (log)')
    axp.set_title('(b) HERA Pareto proxy space')
    # add colorbar for rank mapping
    try:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axp)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(sc, cax=cax)
    except Exception:
        cb = fig.colorbar(sc, ax=axp)
    cb.set_label('Rank k')
    try:
        cb.set_ticks([1, ks[-1]])
    except Exception:
        pass
    axp.set_xscale('log')
    axp.set_yscale('log')
    axp.set_xlabel('Science distortion proxy (log)')
    axp.set_ylabel('RFI leakage proxy (log)')
    axp.set_title('(b) HERA Pareto proxy space')

    fig.suptitle('Figure 8 — HERA rank sweep and Pareto (real data)', y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('WROTE:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'HERA_04-03-2022_all.pkl'))
    ap.add_argument('--K', type=int, default=50)
    ap.add_argument('--out', type=str, default=OUTPATH)
    args = ap.parse_args()
    run_plot(args.path, K=args.K, out=args.out)


if __name__ == '__main__':
    main()
