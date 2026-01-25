#!/usr/bin/env python
"""Inject a synthetic comb into HERA slices and run rank-sweep comparisons.

Produces:
 - hera_outputs/hera_injection_compare.png  (Pareto overlay: original vs variants)
 - hera_outputs/hera_injection_details.csv   (optional per-variant ks/leak/dist)

Usage:
  python tools\hera_inject_and_sweep.py <path_to_pickle> [outdir]

This script selects a few baseline indices from the first array in the pickle
and runs the same rank-sweep proxy used by `hera_rank_sweep.py`.
"""
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
# Ensure project root is on sys.path so we can import sibling modules
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from hera_rank_sweep import run_rank_sweep_real


def load_first_array(path: str) -> np.ndarray:
    payload = pd.read_pickle(path)
    if isinstance(payload, (list, tuple)) and len(payload) > 0:
        arr = np.asarray(payload[0])
    else:
        arr = np.asarray(payload)
    return arr


def extract_baseline_pol(arr: np.ndarray, baseline_idx: int = 0, pol_idx: int = 0) -> np.ndarray:
    """Reduce arr to a 2D (time, freq) matrix for a given baseline and pol index.

    Handles arrays with shapes like (T, F, B, P) or (T, F, B) or (T, F, P) etc.
    """
    X = np.asarray(arr)
    # If arr has >=4 dims (T,F,B,P), select baseline and pol
    if X.ndim >= 4:
        return X[:, :, baseline_idx, pol_idx].astype(np.float32, copy=False)
    # If arr has 3 dims, guess which axis is baseline/pol: prefer (T,F,B)
    if X.ndim == 3:
        # Heuristic: if third dim size is small (<8) treat as pol/baseline; else assume it's baseline
        if X.shape[2] <= 4:
            # treat as pol axis
            return X[:, :, pol_idx].astype(np.float32, copy=False)
        else:
            return X[:, :, baseline_idx].astype(np.float32, copy=False)
    # If already 2D
    if X.ndim == 2:
        return X.astype(np.float32, copy=False)
    # Otherwise, try to squeeze trailing dims
    Xs = np.squeeze(X)
    if Xs.ndim == 2:
        return Xs.astype(np.float32, copy=False)
    raise ValueError(f"Unable to reduce array to 2D, final shape={X.shape}")


def inject_comb(X: np.ndarray, peak_spacing: int = 20, amp_scale: float = 0.01, drift: float = 0.0) -> np.ndarray:
    """Return X with a synthetic comb injected.

    peak_spacing: spacing in frequency channels between comb teeth
    amp_scale: fraction of data std used as comb amplitude
    drift: channels per time-step to shift (can be 0)
    """
    Xc = X.copy().astype(float)
    T, F = Xc.shape
    std = np.std(Xc)
    amp = float(amp_scale * std)

    # choose peak centers
    peaks = list(range(0, F, peak_spacing))
    # create freq mask
    freq_mask = np.zeros(F, dtype=float)
    for p in peaks:
        # put a small Gaussian-shaped bump across a couple channels
        width = max(1, int(0.5 * peak_spacing))
        idx = np.arange(F)
        freq_mask += np.exp(-0.5 * ((idx - p) / float(width + 1e-6)) ** 2)
    # normalize mask
    freq_mask /= np.max(freq_mask) + 1e-12

    # time modulation (optionally drift the peaks)
    for t in range(T):
        shift = int(np.round(drift * t)) if drift != 0.0 else 0
        if shift == 0:
            Xc[t, :] += amp * freq_mask
        else:
            Xc[t, :] += amp * np.roll(freq_mask, shift)
    return Xc


def run_and_collect(X: np.ndarray, max_k: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks, leak, dist = run_rank_sweep_real(X, max_k=max_k)
    return ks, leak, dist


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools\\hera_inject_and_sweep.py <path_to_pickle> [outdir]")
        sys.exit(1)
    pkl = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "hera_outputs"
    os.makedirs(outdir, exist_ok=True)

    arr = load_first_array(pkl)
    print("Loaded array shape:", arr.shape)

    # choose a few baseline indices to test
    baselines = [0, 1, 2]
    pol = 0

    results = []  # tuples (label, ks, leak, dist)

    for b in baselines:
        try:
            X = extract_baseline_pol(arr, baseline_idx=b, pol_idx=pol)
        except Exception as e:
            print(f"Skipping baseline {b}: {e}")
            continue
        ks, leak, dist = run_and_collect(X, max_k=50)
        results.append((f"baseline_{b}", ks, leak, dist))

    # Inject comb into baseline 0 variant
    try:
        X0 = extract_baseline_pol(arr, baseline_idx=0, pol_idx=pol)
        X_inj = inject_comb(X0, peak_spacing=16, amp_scale=0.02, drift=0.0)
        ks_i, leak_i, dist_i = run_and_collect(X_inj, max_k=50)
        results.append(("baseline_0_injected_comb", ks_i, leak_i, dist_i))
    except Exception as e:
        print("Injection failed:", e)

    # Plot Pareto overlay (leak vs dist)
    plt.figure(figsize=(6.8, 5.0))
    for label, ks, leak, dist in results:
        plt.plot(leak, dist, marker="o", linestyle="-", lw=1.2, ms=3.0, label=label)
    plt.xlabel("RFI leakage proxy")
    plt.ylabel("Science distortion proxy")
    plt.xscale("log")
    plt.yscale("linear")
    plt.legend(loc="best", fontsize=8)
    plt.title("HERA: baseline variants + injected comb (Pareto overlay)")
    plt.tight_layout()
    figpath = os.path.join(outdir, "hera_injection_compare.png")
    plt.savefig(figpath, dpi=180)
    plt.close()
    print("Wrote:", figpath)

    # Save numeric results for later inspection
    csv_path = os.path.join(outdir, "hera_injection_details.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("label,k,leak,dist\n")
        for label, ks, leak, dist in results:
            for k, l, d in zip(ks, leak, dist):
                f.write(f"{label},{int(k)},{float(l):.8e},{float(d):.8e}\n")
    print("Wrote:", csv_path)


if __name__ == '__main__':
    main()
