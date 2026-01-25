#!/usr/bin/env python3
"""
Figure 4 (synthetic): rank sweep on the controlled experiment data (not HERA).
Computes RFI leakage and science distortion for k=1..K and plots both on one axes.
Saves: ../outputs/figure4_rank_sweep_synthetic.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)
OUTPATH = os.path.join(OUTDIR, 'figure4_rank_sweep_synthetic.png')

# build a controlled synthetic dataset (similar to Fig2)
T = 200
F = 128
rng = np.random.default_rng(11)

t = np.linspace(0, 1, T)
f = np.linspace(0, 1, F)

# science: narrow core + broader envelope
core_center = 0.43
core_width = 0.06
band_center = 0.5
science_core = np.exp(-0.5 * ((f - core_center) / core_width) ** 2)
science_env = 1.0 / (1.0 + 5.0 * (f - band_center) ** 2)
S_nu = 0.07 * science_env + 0.03 * science_core

# comb: narrow spikes, choose one inside core and others outside
comb_freqs = np.array([0.12, 0.27, 0.43, 0.62, 0.80])
comb_idx = np.array([int(np.argmin(np.abs(f - cf))) for cf in comb_freqs])
comb = np.zeros((T, F))
for i, idx in enumerate(comb_idx):
    duty = 0.12 + 0.05 * (i % 3)
    on = (rng.random(T) < duty).astype(float)
    amp = 0.6 if i % 2 == 0 else 0.35
    comb[:, idx] += on * amp
# small ripple
comb += 0.03 * np.sin(2 * np.pi * 2.0 * np.outer(t, np.linspace(0, 1, F)))

# composite
X = np.outer(np.ones(T), S_nu) + comb + 0.008 * rng.normal(size=(T, F))

# ground-truth components for metrics
S_true = np.outer(np.ones(T), S_nu)
R_true = comb

# helper: truncated SVD Lk
def truncated_svd_Lk(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if k == 0:
        return np.zeros_like(X)
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    return (Uk * sk) @ Vtk

# rank sweep
K = 30
ks = np.arange(1, K + 1)
leak_vals = []
dist_vals = []
S_norm = np.linalg.norm(S_true, 'fro')
R_norm = np.linalg.norm(R_true, 'fro')
for k in ks:
    Lk = truncated_svd_Lk(X, k)
    Ek = X - Lk
    # leakage: how well R_true is captured by Lk (we measure relative error of R_true vs Lk)
    leak = np.linalg.norm(R_true - Lk, 'fro') / (R_norm + 1e-16)
    # preservation: fraction of science remaining in residual Ek relative to true science
    preservation = np.linalg.norm(Ek, 'fro') / (S_norm + 1e-16)
    if preservation > 1.0:
        preservation = 1.0
    distortion = 1.0 - preservation
    leak_vals.append(leak)
    dist_vals.append(distortion)

leak_vals = np.array(leak_vals)
dist_vals = np.array(dist_vals)

# plot both curves
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 12})
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(ks, leak_vals, marker='o', lw=1.8, label='RFI leakage (lower better)')
ax.plot(ks, dist_vals, marker='s', lw=1.8, label='Science distortion (lower better)')
ax.set_yscale('log')
ax.set_xlabel('Rank k')
ax.set_ylabel('Proxy (log scale)')
ax.set_title('Figure 4 (synthetic) — Rank sweep: leakage vs distortion')
ax.grid(True, which='both', ls=':', alpha=0.6)
ax.legend(loc='best')

# shade non-identifiable band (example)
ax.axvspan(4.5, 12.5, color='gray', alpha=0.12)
ax.text(6.5, max(np.nanmax(leak_vals), np.nanmax(dist_vals)) * 0.6, 'non-identifiable band', color='k', fontsize=9, ha='center')

fig.tight_layout()
fig.savefig(OUTPATH, dpi=220, bbox_inches='tight')
plt.close(fig)
print(f'WROTE: {OUTPATH}')
