#!/usr/bin/env python3
"""
Figure 5: Pareto front from synthetic rank sweep.
x = Bias^- (science distortion), y = Bias^+ (RFI leakage).
Each point = different rank k. Highlight Pareto front, knee, and bias floor.
Saves: ../outputs/figure5_pareto_front.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)
OUTPATH = os.path.join(OUTDIR, 'figure5_pareto_front.png')

# Recreate synthetic data and rank sweep (same as figure4 synthetic)
T = 200
F = 128
rng = np.random.default_rng(11)

t = np.linspace(0, 1, T)
f = np.linspace(0, 1, F)

core_center = 0.43
core_width = 0.06
band_center = 0.5
science_core = np.exp(-0.5 * ((f - core_center) / core_width) ** 2)
science_env = 1.0 / (1.0 + 5.0 * (f - band_center) ** 2)
S_nu = 0.07 * science_env + 0.03 * science_core

comb_freqs = np.array([0.12, 0.27, 0.43, 0.62, 0.80])
comb_idx = np.array([int(np.argmin(np.abs(f - cf))) for cf in comb_freqs])
comb = np.zeros((T, F))
for i, idx in enumerate(comb_idx):
    duty = 0.12 + 0.05 * (i % 3)
    on = (rng.random(T) < duty).astype(float)
    #!/usr/bin/env python3
    """
    Figure 5: Pareto front from synthetic rank sweep.
    x = Bias^- (science distortion), y = Bias^+ (RFI leakage).
    Each point = different rank k. Highlight Pareto front, knee, and bias floor.
    Saves: ../outputs/figure5_pareto_front.png
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(OUTDIR, exist_ok=True)
    OUTPATH = os.path.join(OUTDIR, 'figure5_pareto_front.png')

    # Recreate synthetic data and rank sweep (same as figure4 synthetic)
    T = 200
    F = 128
    rng = np.random.default_rng(11)

    t = np.linspace(0, 1, T)
    f = np.linspace(0, 1, F)

    core_center = 0.43
    core_width = 0.06
    band_center = 0.5
    science_core = np.exp(-0.5 * ((f - core_center) / core_width) ** 2)
    science_env = 1.0 / (1.0 + 5.0 * (f - band_center) ** 2)
    S_nu = 0.07 * science_env + 0.03 * science_core

    comb_freqs = np.array([0.12, 0.27, 0.43, 0.62, 0.80])
    comb_idx = np.array([int(np.argmin(np.abs(f - cf))) for cf in comb_freqs])
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

    # truncated SVD
    def truncated_svd_Lk(X, k):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        if k == 0:
            return np.zeros_like(X)
        Uk = U[:, :k]
        sk = s[:k]
        Vtk = Vt[:k, :]
        return (Uk * sk) @ Vtk

    # compute for a range of k
    K = 50
    ks = np.arange(1, K + 1)
    leak_vals = []
    dist_vals = []
    S_norm = np.linalg.norm(S_true, 'fro')
    R_norm = np.linalg.norm(R_true, 'fro')
    for k in ks:
        Lk = truncated_svd_Lk(X, k)
        Ek = X - Lk
        leak = np.linalg.norm(R_true - Lk, 'fro') / (R_norm + 1e-16)
        preservation = np.linalg.norm(Ek, 'fro') / (S_norm + 1e-16)
        if preservation > 1.0:
            preservation = 1.0
        distortion = 1.0 - preservation
        leak_vals.append(leak)
        dist_vals.append(distortion)
    leak = np.array(leak_vals)
    dist = np.array(dist_vals)

    # Pareto front: minimize both dist (x) and leak (y)
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

    mask = pareto_mask_minimize(dist, leak)
    # indices on front
    front_idx = np.where(mask)[0]
    front_k = ks[front_idx]
    front_x = dist[front_idx]
    front_y = leak[front_idx]
    # sort by x for line
    order = np.argsort(front_x)
    front_x = front_x[order]
    front_y = front_y[order]
    front_k = front_k[order]

    # knee detection: compute discrete curvature on log-log front
    lx = np.log(front_x + 1e-16)
    ly = np.log(front_y + 1e-16)
    # compute angle between consecutive segments
    curv = np.zeros(len(front_x))
    for i in range(1, len(front_x)-1):
        v1 = np.array([lx[i]-lx[i-1], ly[i]-ly[i-1]])
        v2 = np.array([lx[i+1]-lx[i], ly[i+1]-ly[i]])
        # normalized angle
        nv1 = v1 / (np.linalg.norm(v1)+1e-16)
        nv2 = v2 / (np.linalg.norm(v2)+1e-16)
        # curvature measure: 1 - cos(theta)
        cosang = np.clip(np.dot(nv1, nv2), -1.0, 1.0)
        curv[i] = 1.0 - cosang
    if np.any(curv>0):
        knee_idx_local = int(np.argmax(curv))
        knee_k = int(front_k[knee_idx_local])
        knee_x = float(front_x[knee_idx_local])
        knee_y = float(front_y[knee_idx_local])
    else:
        knee_k = int(front_k[len(front_k)//2])
        knee_x = float(front_x[len(front_x)//2])
        knee_y = float(front_y[len(front_y)//2])

    # bias floor: approximate floor as min y achieved for large k (last 5 ks)
    floor_y = np.min(leak[np.where(ks >= (K-5))])

    # Plot
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(6.5,6))
    # scatter: color encodes rank k; keep single marker style for all points
    sc = ax.scatter(dist, leak, c=ks, cmap='viridis', s=50, marker='o', edgecolors='none')
    # subtle dotted gray line connecting the Pareto front points (very low alpha) to hint the frontier
    if len(front_x) > 1:
        ax.plot(front_x, front_y, linestyle=':', color='gray', linewidth=0.8, alpha=0.35)

    # shading: mark an "unattainable region" near the origin (low distortion & low leakage)
    try:
        xmin = np.min(dist)
        ymin = np.min(leak)
        # use low-percentile thresholds to define shading extents
        x_shade = np.percentile(dist, 6)
        y_shade = np.percentile(leak, 6)
        # draw a light gray rectangle in data coords (works with log scales)
        ax.fill([xmin, x_shade, x_shade, xmin], [ymin, ymin, y_shade, y_shade],
                color='lightgray', alpha=0.35, zorder=0)
        ax.text(x_shade * 1.15, y_shade * 0.95, 'unreachable under\nsingle-epoch constraints',
                color='gray', fontsize=9, va='center', ha='left')
    except Exception:
        # defensive: don't fail plotting if percentiles can't be computed
        pass

    # knee: draw a thin dotted circle around the heuristic knee and add a plain text label (no arrow)
    from matplotlib.patches import Circle
    circle = Circle((knee_x, knee_y), radius=max(0.04 * knee_x, 0.02), facecolor='none', edgecolor='magenta', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.add_patch(circle)
    ax.text(knee_x * 1.05, knee_y * 0.98, f'Example knee (heuristic): k ≈ {knee_k}', color='magenta', fontsize=10, va='center')

    # bias floor horizontal line (interpretation only)
    ax.axhline(floor_y, color='gray', lw=1.2, linestyle='--')
    ax.text(np.max(dist) * 0.6, floor_y * 1.08, 'bias floor (leakage)', color='gray', fontsize=9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Bias⁻ — Science distortion (lower is better; log)')
    ax.set_ylabel('Bias⁺ — RFI leakage (lower is better; log)')
    ax.set_title('Figure 5 — Pareto front')
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('Rank k')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', ls=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUTPATH, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'WROTE: {OUTPATH}')
    print(f'knee k={knee_k}, knee_x={knee_x:.4e}, knee_y={knee_y:.4e}, floor_y={floor_y:.4e}')
