#!/usr/bin/env python3
"""
Figure 6: Overlay Pareto fronts for TSVD vs a simple FWSVD variant.
Shows that FWSVD moves the knee but does not remove the leakage floor.

Saves: ../outputs/figure6_fws_vs_tsvd.png
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)
OUTPNG = os.path.join(OUTDIR, 'figure6_fws_vs_tsvd.png')


def truncated_svd_Lk(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if k == 0:
        return np.zeros_like(X)
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    return (Uk * sk) @ Vtk


def fws_svd_Lk(X, k, freqs, alpha=3.0, centers=(0.43,)):
    # simple frequency-weighting: emphasize frequencies near centers when performing SVD
    # create diagonal weights w_j = 1 + alpha * sum_gaussians
    w = np.ones_like(freqs)
    for c in centers:
        w += alpha * np.exp(-0.5 * ((freqs - c) / 0.03) ** 2)
    W = np.diag(w)
    # weight columns
    Xw = X @ W
    # SVD on weighted data
    Uw, s, Vtw = np.linalg.svd(Xw, full_matrices=False)
    if k == 0:
        return np.zeros_like(X)
    Uwk = Uw[:, :k]
    sk = s[:k]
    Vtwk = Vtw[:k, :]
    Lw = (Uwk * sk) @ Vtwk
    # unweight columns
    Winv = np.diag(1.0 / w)
    return Lw @ Winv


def pareto_mask_minimize(x, y):
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


def run(seed=11, K=50, write_pdf=False):
    rng = np.random.default_rng(seed)
    T = 200
    F = 128
    t = np.linspace(0, 1, T)
    freqs = np.linspace(0, 1, F)

    core_center = 0.43
    core_width = 0.06
    band_center = 0.5
    science_core = np.exp(-0.5 * ((freqs - core_center) / core_width) ** 2)
    science_env = 1.0 / (1.0 + 5.0 * (freqs - band_center) ** 2)
    S_nu = 0.07 * science_env + 0.03 * science_core

    comb_freqs = np.array([0.12, 0.27, 0.43, 0.62, 0.80])
    comb_idx = np.array([int(np.argmin(np.abs(freqs - cf))) for cf in comb_freqs])
    comb = np.zeros((T, F))
    for i, idx in enumerate(comb_idx):
        duty = 0.12 + 0.05 * (i % 3)
        on = (rng.random(T) < duty).astype(float)
        amp = 0.6 if i % 2 == 0 else 0.35
        comb[:, idx] += on * amp
    comb += 0.03 * np.sin(2 * np.pi * 2.0 * np.outer(t, np.linspace(0, 1, F)))

    X = np.outer(np.ones(T), S_nu) + comb + 0.008 * rng.normal(size=(T, F))
    S_true = np.outer(np.ones(T), S_nu)
    R_true = comb

    ks = np.arange(1, K + 1)
    leak_t = []
    dist_t = []
    leak_fw = []
    dist_fw = []
    S_norm = np.linalg.norm(S_true, 'fro')
    R_norm = np.linalg.norm(R_true, 'fro')

    for k in ks:
        Lk_t = truncated_svd_Lk(X, k)
        Ek_t = X - Lk_t
        leak_val_t = np.linalg.norm(R_true - Lk_t, 'fro') / (R_norm + 1e-16)
        preservation_t = np.linalg.norm(Ek_t, 'fro') / (S_norm + 1e-16)
        preservation_t = min(preservation_t, 1.0)
        dist_val_t = 1.0 - preservation_t
        leak_t.append(leak_val_t)
        dist_t.append(dist_val_t)

        Lk_fw = fws_svd_Lk(X, k, freqs, alpha=3.0, centers=comb_freqs)
        Ek_fw = X - Lk_fw
        leak_val_fw = np.linalg.norm(R_true - Lk_fw, 'fro') / (R_norm + 1e-16)
        preservation_fw = np.linalg.norm(Ek_fw, 'fro') / (S_norm + 1e-16)
        preservation_fw = min(preservation_fw, 1.0)
        dist_val_fw = 1.0 - preservation_fw
        leak_fw.append(leak_val_fw)
        dist_fw.append(dist_val_fw)

    leak_t = np.array(leak_t)
    dist_t = np.array(dist_t)
    leak_fw = np.array(leak_fw)
    dist_fw = np.array(dist_fw)

    mask_t = pareto_mask_minimize(dist_t, leak_t)
    mask_fw = pareto_mask_minimize(dist_fw, leak_fw)
    front_idx_t = np.where(mask_t)[0]
    front_idx_fw = np.where(mask_fw)[0]

    front_x_t = dist_t[front_idx_t]
    front_y_t = leak_t[front_idx_t]
    front_k_t = ks[front_idx_t]
    order_t = np.argsort(front_x_t)
    front_x_t = front_x_t[order_t]
    front_y_t = front_y_t[order_t]
    front_k_t = front_k_t[order_t]

    front_x_fw = dist_fw[front_idx_fw]
    front_y_fw = leak_fw[front_idx_fw]
    front_k_fw = ks[front_idx_fw]
    order_fw = np.argsort(front_x_fw)
    front_x_fw = front_x_fw[order_fw]
    front_y_fw = front_y_fw[order_fw]
    front_k_fw = front_k_fw[order_fw]

    # knee detection (simple curvature)
    def detect_knee(front_x, front_y, front_k):
        if len(front_x) < 3:
            return int(front_k[len(front_k)//2])
        lx = np.log(front_x + 1e-16)
        ly = np.log(front_y + 1e-16)
        curv = np.zeros(len(front_x))
        for i in range(1, len(front_x)-1):
            v1 = np.array([lx[i]-lx[i-1], ly[i]-ly[i-1]])
            v2 = np.array([lx[i+1]-lx[i], ly[i+1]-ly[i]])
            nv1 = v1 / (np.linalg.norm(v1)+1e-16)
            nv2 = v2 / (np.linalg.norm(v2)+1e-16)
            cosang = np.clip(np.dot(nv1, nv2), -1.0, 1.0)
            curv[i] = 1.0 - cosang
        idx = int(np.argmax(curv))
        return int(front_k[idx])

    knee_t = detect_knee(front_x_t, front_y_t, front_k_t)
    knee_fw = detect_knee(front_x_fw, front_y_fw, front_k_fw)

    # floor approx (from highest ks)
    floor_t = np.min(leak_t[ks >= (K-5)])
    floor_fw = np.min(leak_fw[ks >= (K-5)])

    # Plot
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(6.8, 6))
    # TSVD scatter
    sc1 = ax.scatter(dist_t, leak_t, c=ks, cmap='viridis', s=40, marker='o', edgecolors='none')
    # FWSVD scatter (same color mapping for k)
    sc2 = ax.scatter(dist_fw, leak_fw, c=ks, cmap='viridis', s=40, marker='s', edgecolors='none', alpha=0.9)

    # very faint trajectory hints so the points read as a cloud rather than a continuous curve
    if len(front_x_t) > 1:
        ax.plot(front_x_t, front_y_t, linestyle=':', color='black', linewidth=0.6, alpha=0.25)
    if len(front_x_fw) > 1:
        ax.plot(front_x_fw, front_y_fw, linestyle='-.', color='black', linewidth=0.6, alpha=0.25)

    # annotate knee shift (text only)
    ax.text(0.52, 0.95, f'Example knee shift: TSVD k≈{knee_t} → FWSVD k≈{knee_fw}', transform=ax.transAxes, color='black', fontsize=10, va='top')

    # show floor line (single gray dashed line, same for both)
    floor = min(floor_t, floor_fw)
    ax.axhline(floor, color='gray', lw=1.2, linestyle='--')
    ax.text(0.55, floor * 1.02, 'bias floor (leakage)', color='gray', fontsize=9)

    # shading: mark an "unattainable region" near origin (low distortion & low leakage)
    try:
        all_dist = np.concatenate([dist_t, dist_fw])
        all_leak = np.concatenate([leak_t, leak_fw])
        xmin = np.min(all_dist)
        ymin = np.min(all_leak)
        x_shade = np.percentile(all_dist, 6)
        y_shade = np.percentile(all_leak, 6)
        ax.fill([xmin, x_shade, x_shade, xmin], [ymin, ymin, y_shade, y_shade], color='lightgray', alpha=0.35, zorder=0)
        ax.text(x_shade * 1.12, y_shade * 0.95, 'unreachable under\nsingle-epoch constraints', color='gray', fontsize=9, va='center', ha='left')
    except Exception:
        pass

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Bias⁻ — Science distortion (lower is better; log)')
    ax.set_ylabel('Bias⁺ — RFI leakage (lower is better; log)')
    ax.set_title('Figure 6 — Empirical Pareto trade-off overlay: TSVD vs FWSVD')
    cb = fig.colorbar(sc1, ax=ax, pad=0.02)
    cb.set_label('Rank k (within each method)')
    # subdued legend entries to avoid implying a definitive method-win
    ax.plot([], [], marker='o', color='k', linestyle='None', label='TSVD (example sweep)', markersize=6, alpha=0.6)
    ax.plot([], [], marker='s', color='k', linestyle='None', label='FWSVD (example sweep)', markersize=6, alpha=0.6)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUTPNG, dpi=220, bbox_inches='tight')
    if write_pdf:
        outpdf = OUTPNG.replace('.png', '.pdf')
        try:
            fig.savefig(outpdf, bbox_inches='tight')
            print('WROTE:', outpdf)
        except Exception as e:
            print('PDF export failed:', e)
    plt.close(fig)

    print('WROTE:', OUTPNG)
    print(f'knee_tsvd={knee_t}, knee_fws={knee_fw}, floor={floor:.4e}')
    return knee_t, knee_fw, floor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()
    run(seed=args.seed, K=args.K, write_pdf=args.pdf)


if __name__ == '__main__':
    main()
