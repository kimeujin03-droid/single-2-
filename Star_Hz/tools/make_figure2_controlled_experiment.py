#!/usr/bin/env python3
"""
Figure 2: Controlled experiment showing science alone, RFI comb alone, and composed D(t,nu).
- (a) Science S(nu) (1D) with shaded science core/band
- (b) RFI comb (time-frequency) with overlapping lines highlighted
- (c) Composite D(t,nu) = science (broadcast in time) + comb + noise

Saves: ../outputs/figure2_controlled_experiment.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)
OUTPATH = os.path.join(OUTDIR, "figure2_controlled_experiment.png")

# Parameters
T = 200
F = 128
rng = np.random.default_rng(2)

t = np.linspace(0, 1, T)
f = np.linspace(0, 1, F)

# Science: narrow-bandcore (Gaussian in freq) + broader band envelope
core_center = 0.43
core_width = 0.06
band_center = 0.5
band_width = 0.25

science_core = np.exp(-0.5 * ((f - core_center) / core_width) ** 2)
science_envelope = 1.0 / (1.0 + 5.0 * (f - band_center) ** 2)
S_nu = 0.08 * science_envelope + 0.04 * science_core

# RFI comb: choose a set of narrow frequencies, some overlap the science core and some don't
comb_freqs = np.array([0.10, 0.27, 0.43, 0.62, 0.80])  # one at 0.43 overlaps core
# Map to nearest freq indices
comb_idx = np.array([int(np.argmin(np.abs(f - cf))) for cf in comb_freqs])

# Build comb TF: intermittent in time
comb = np.zeros((T, F))
for i, idx in enumerate(comb_idx):
    # bursts with different duty cycles
    duty = 0.12 + 0.05 * (i % 3)
    on = (rng.random(T) < duty).astype(float)
    amplitude = 0.6 if i % 2 == 0 else 0.35
    comb[:, idx] += on * amplitude

# small broadband ripple to make it realistic
comb += 0.03 * np.sin(2 * np.pi * 2.0 * np.outer(t, np.linspace(0, 1, F)))

# composite D(t,nu): broadcast S_nu in time + comb + small noise
X = np.outer(np.ones(T), S_nu) + comb + 0.008 * rng.normal(size=(T, F))

# determine which comb lines overlap the science core (core defined by center +/- core_width*1.5)
core_lo = core_center - 1.5 * core_width
core_hi = core_center + 1.5 * core_width
overlap_mask = (comb_freqs >= core_lo) & (comb_freqs <= core_hi)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) science S(nu)
ax = axes[0]
ax.plot(f, S_nu, lw=2, color='navy')
ax.fill_between(f, 0, S_nu, color='navy', alpha=0.12)
# shade science core region
ax.axvspan(core_lo, core_hi, color='cyan', alpha=0.25)
ax.set_xlim(f[0], f[-1])
ax.set_xlabel('frequency (normalized)')
ax.set_ylabel('amplitude')
ax.set_title('(a) science S(\u03BD) (synthetic)')

# (b) RFI comb only
ax = axes[1]
im = ax.imshow(comb, origin='lower', aspect='auto', cmap='magma', extent=[0, F, 0, T])
ax.set_xlabel('freq channel')
ax.set_ylabel('time sample')
ax.set_title('(b) RFI comb (synthetic)')
plt.colorbar(im, ax=ax, pad=0.01)
# mark comb lines; highlight overlapping ones in red
for i, idx in enumerate(comb_idx):
    x = f[idx]
    color = 'red' if overlap_mask[i] else 'white'
    # make comb line markers subtle; mark overlaps in red
    ax.axvline(idx + 0.5, color=color, lw=0.8, alpha=0.8 if overlap_mask[i] else 0.35)

# (c) composite
ax = axes[2]
im2 = ax.imshow(X, origin='lower', aspect='auto', cmap='viridis', extent=[0, F, 0, T])
ax.set_xlabel('freq channel')
ax.set_title('(c) composite D(t,\u03BD) (synthetic)')
plt.colorbar(im2, ax=ax, pad=0.01)
# shade science band region in composite for visual link (use freq axis indices)
xlo = np.argmin(np.abs(f - core_lo))
xhi = np.argmin(np.abs(f - core_hi))
ax.axvspan(xlo, xhi, color='cyan', alpha=0.18)
ax.text(0.02, 0.9, 'science core (shaded)', transform=ax.transAxes, fontsize=9, color='cyan', va='top')

# minimal in-figure annotation only: shade science core for visual link
ax.text(0.02, 0.9, 'science core (shaded)', transform=ax.transAxes, fontsize=8, color='cyan', va='top')

fig.tight_layout()
fig.savefig(OUTPATH, dpi=220, bbox_inches='tight')
plt.close(fig)
print(f'WROTE: {OUTPATH}')
print(f'Comb freqs: {comb_freqs}')
print(f'Overlap mask: {overlap_mask}')
