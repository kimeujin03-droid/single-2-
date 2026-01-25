#!/usr/bin/env python3
"""
Create a 3-panel figure illustrating why single-epoch low-rank SVD can be dangerous.
Panels:
 (a) original single-epoch time-frequency map
 (b) rank-1 reconstruction (top singular mode)
 (c) annotated panel highlighting science+RFI mixing (residual contours + text)

Saves: ../outputs/figure_single_epoch_lowrank_warning.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "figure_single_epoch_lowrank_warning.png")

# synth params
T = 256  # time samples
F = 128  # freq channels
rng = np.random.default_rng(1)

t = np.linspace(0, 1, T)
f = np.linspace(0, 1, F)

# (1) smooth foreground (low-rank) : outer product of a slow time curve and smooth bandpass
time_fg = 1.0 + 0.8 * np.cos(2 * np.pi * 0.5 * t)  # slow time variation
freq_fg = 1.0 / (1.0 + 3.0 * (f - 0.5) ** 2)  # smooth bandpass
FG = np.outer(time_fg, freq_fg)

# (2) comb-like narrowband RFI: narrow freq spikes present some times
comb = np.zeros((T, F))
comb_freq_idx = np.arange(5, F, 12)  # comb spacing
for idx in comb_freq_idx:
    # intermittent in time: bursts
    on = (rng.random(T) > 0.85).astype(float)
    comb[:, idx] += on * (0.6 + 0.4 * rng.random(T))
# add a broad ripple to one comb line to be realistic
comb += 0.04 * np.sin(2 * np.pi * 3.0 * np.outer(t, np.linspace(0, 1, F)))

# (3) faint science ripple: small-amplitude sinusoidal across freq and slowly varying in time
sci = 0.02 * np.sin(2 * np.pi * (5.0 * f[None, :] + 3.0 * t[:, None]))

# noise
noise = 0.005 * rng.normal(size=(T, F))

# final data
X = FG + comb + sci + noise

# do SVD and rank-1 reconstruction
U, s, Vt = np.linalg.svd(X, full_matrices=False)
rank1 = np.outer(U[:, 0] * s[0], Vt[0, :])
energy_frac = s[0] ** 2 / np.sum(s ** 2)

# residual highlighting (what rank-1 misses)
residual = X - rank1
res_abs = np.abs(residual)

# build figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
extent = [0, F, 0, T]

# panel (a) original
ax = axes[0]
im = ax.imshow(X, aspect='auto', origin='lower', cmap='viridis', extent=extent)
ax.set_title('(a) original single-epoch TF')
ax.set_xlabel('freq channel')
ax.set_ylabel('time sample')
plt.colorbar(im, ax=ax, pad=0.01)

# panel (b) rank-1
im2 = ax.imshow(rank1, aspect='auto', origin='lower', cmap='viridis', extent=extent)
ax = axes[1]
im2 = ax.imshow(rank1, aspect='auto', origin='lower', cmap='viridis', extent=extent)
# add a persuasive reviewer-facing subtitle emphasizing mixing
ax.set_title('(b) rank-1 reconstruction (top singular mode)\nDominated by mixed smooth + narrowband structures\nEnergy fraction: {0:.1f}%'.format(100*energy_frac))
ax.set_xlabel('freq channel')
plt.colorbar(im2, ax=ax, pad=0.01)

# panel (c) annotated: show original with residual contours (where science+RFI survive)
ax = axes[2]
im3 = ax.imshow(X, aspect='auto', origin='lower', cmap='viridis', extent=extent)
ax.set_title('(c) residuals (X - rank1) highlight science + RFI')
ax.set_xlabel('freq channel')
plt.colorbar(im3, ax=ax, pad=0.01)

# overlay a modest contour of residual amplitude (single representative level)
level = np.percentile(res_abs, 95)
cs = ax.contour(np.arange(F)+0.5, np.arange(T)+0.5, res_abs, levels=[level], colors=['yellow'], linewidths=1.2)
# don't add numeric contour labels — we only want a visual highlight

# annotate a few example comb frequencies (just 2-3 examples) and draw faint lines for others
for i, idx in enumerate(comb_freq_idx):
    ax.axvline(idx+0.5, color='white', lw=0.4, alpha=0.25)
    if i < 3:
        ax.text(idx+0.6, T*0.06, f'RFI (ex.)', color='white', fontsize=9, rotation=90, va='bottom', ha='left')

# single boxed region for the science core
x0, x1 = 30, 60
y0, y1 = 100, 150
rect = plt.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1.2, edgecolor='cyan', facecolor='none', alpha=0.95)
ax.add_patch(rect)
ax.text(x0+1, y1-6, 'science core region', color='cyan', fontsize=10)

fig.tight_layout()
fig.savefig(outpath, dpi=220, bbox_inches='tight')
plt.close(fig)
print(f"WROTE: {outpath}")
print(f"Top-singular energy fraction: {energy_frac:.4f}")
