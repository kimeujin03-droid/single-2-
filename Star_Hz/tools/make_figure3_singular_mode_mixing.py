#!/usr/bin/env python3
"""
Figure 3: Evidence of mixed singular modes (frequency profiles of top singular vectors).
Shows that mixing (science band + comb lines) appears in singular vectors before any rank selection.
Saves: ../outputs/figure3_singular_mode_mixing.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)
OUTPATH = os.path.join(OUTDIR, "figure3_singular_mode_mixing.png")

# synth params (similar to Fig2 but tuned for clear mode-mixing)
T = 300
F = 256
rng = np.random.default_rng(5)

t = np.linspace(0, 1, T)
f = np.linspace(0, 1, F)

# science: narrow core + broad envelope
core_center = 0.44
core_width = 0.04
band_center = 0.5
band_width = 0.22
science_core = np.exp(-0.5 * ((f - core_center) / core_width) ** 2)
science_env = 1.0 / (1.0 + 6.0 * (f - band_center) ** 2)
S_nu = 0.06 * science_env + 0.03 * science_core

# comb: multiple narrow spikes, include some inside the core to enforce mixing
comb_freqs = np.array([0.12, 0.28, 0.44, 0.60, 0.76])
comb_idx = np.array([int(np.argmin(np.abs(f - cf))) for cf in comb_freqs])

# time behaviour for comb: periodic bursts with a base periodicity so modes pick it up
comb = np.zeros((T, F))
for i, idx in enumerate(comb_idx):
    period = 8 + (i % 3) * 4
    phase = rng.integers(0, period)
    burst = ((np.arange(T) + phase) % period) < max(1, int(0.25 * period))
    amplitude = 0.5 if i % 2 == 0 else 0.3
    comb[burst, idx] += amplitude

# add a weak broadband ripple
comb += 0.02 * np.sin(2 * np.pi * 3.0 * np.outer(t, np.linspace(0, 1, F)))

# composite
X = np.outer(np.ones(T), S_nu) + comb + 0.006 * rng.normal(size=(T, F))

# SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)
# frequency profiles are rows of Vt (shape (F,) per row)
num_modes = 5
modes = Vt[:num_modes]

# explained variance
explained = (s ** 2) / np.sum(s ** 2)

# normalize modes for plotting (preserve sign convention to show features)
modes_norm = modes.copy()
for i in range(num_modes):
    # normalize to max abs 1
    m = modes_norm[i]
    modes_norm[i] = m / np.max(np.abs(m))

# plotting
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
colors = plt.cm.tab10(np.arange(num_modes))

# offset them vertically for clarity
offset = 1.6
for i in range(num_modes):
    y = modes_norm[i] + i * offset
    ax.plot(f, y, color=colors[i], lw=1.6, label=f'mode {i+1} ({100*explained[i]:.1f}%)')

# highlight science band/core
core_lo = core_center - 1.5 * core_width
core_hi = core_center + 1.5 * core_width
ax.axvspan(core_lo, core_hi, color='cyan', alpha=0.18)
ax.text(core_center, -0.5, 'science core', color='cyan', ha='center', fontsize=9)

# mark comb freqs with ticks and small labels; highlight overlapping ones
for i, cf in enumerate(comb_freqs):
    color = 'red' if (cf >= core_lo and cf <= core_hi) else 'gray'
    ax.axvline(cf, color=color, linestyle='--', lw=0.9, alpha=0.9)
    if i < 3:
        ax.text(cf, num_modes*offset - 0.1, f'comb {i+1}', color=color, fontsize=9, rotation=90, va='top', ha='center')

ax.set_xlabel('frequency (normalized)')
ax.set_yticks([])
ax.set_title('Figure 3 — Mixed singular modes: top frequency profiles')
ax.legend(loc='upper right')

# add small horizontal energy fraction bars and explicit percent text next to each mode
xr = f[-1] - f[0]
bar_start = f[-1] + 0.02 * xr
bar_end = f[-1] + 0.07 * xr
# extend xlim to make room for bars
xmax = f[-1] + 0.12 * xr
ax.set_xlim(f[0], xmax)
for i in range(num_modes):
    yloc = i * offset
    # draw a short thick line as a bar
    ax.hlines(yloc, bar_start, bar_end, color=colors[i], linewidth=8, alpha=0.9)
    # percent text to the right of the bar
    pct = 100.0 * explained[i]
    ax.text(bar_end + 0.01 * xr, yloc, f'mode {i+1}: {pct:.1f}%', va='center', fontsize=9)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUTPATH, dpi=220, bbox_inches='tight')
plt.close(fig)

print(f'WROTE: {OUTPATH}')
for i in range(num_modes):
    print(f'mode {i+1}: explained {100*explained[i]:.2f}%')
