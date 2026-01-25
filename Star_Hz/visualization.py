"""
Visualization utilities for radio interference subtraction.
"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from subtraction_core import SubtractMetrics


def plot_heatmaps(raw: np.ndarray, cln: np.ndarray, freqs: np.ndarray, 
                 title_prefix: str = "", save_path: str | None = None):
    """Plot raw and cleaned data heatmaps."""
    n_time = raw.shape[0]
    t = np.arange(n_time)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    extent = [t[0], t[-1], float(freqs[0]), float(freqs[-1])]

    im0 = axes[0].imshow(raw.T, origin="lower", aspect="auto", 
                        extent=extent, cmap="inferno")
    axes[0].set_title(f"{title_prefix}(a) RAW")
    axes[0].set_ylabel("Frequency (MHz)")
    plt.colorbar(im0, ax=axes[0], fraction=0.02)

    im1 = axes[1].imshow(cln.T, origin="lower", aspect="auto", 
                        extent=extent, cmap="inferno")
    axes[1].set_title(f"{title_prefix}(b) CLEANED")
    axes[1].set_ylabel("Frequency (MHz)")
    axes[1].set_xlabel("Time (s)")
    plt.colorbar(im1, ax=axes[1], fraction=0.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_mean_spectra(raw: np.ndarray, cln: np.ndarray, freqs: np.ndarray,
                     sat_window: Tuple[int,int], edges_margin: int,
                     target_freq: float, protect_bw: float, core_bw: float,
                     comb_peaks: List[float], save_path: str | None = None):
    """Plot mean spectra for OFF and SAT windows."""
    n_time = raw.shape[0]
    t0, t1 = sat_window
    em = edges_margin

    off_mask = np.zeros(n_time, dtype=bool)
    off_mask[:em] = True
    off_mask[-em:] = True

    sat_mask = np.zeros(n_time, dtype=bool)
    sat_mask[t0:t1] = True

    raw_off = np.mean(raw[off_mask], axis=0)
    raw_sat = np.mean(raw[sat_mask], axis=0)
    cln_off = np.mean(cln[off_mask], axis=0)
    cln_sat = np.mean(cln[sat_mask], axis=0)

    plt.figure(figsize=(10,4))
    plt.plot(freqs, raw_off, label="RAW OFF", alpha=0.7)
    plt.plot(freqs, raw_sat, label="RAW SAT", alpha=0.7)
    plt.plot(freqs, cln_off, label="CLN OFF", linewidth=2)
    plt.plot(freqs, cln_sat, label="CLN SAT", linewidth=2)
    plt.axvspan(target_freq-protect_bw, target_freq+protect_bw, 
               alpha=0.12, color='blue', label="protect")
    plt.axvspan(target_freq-core_bw, target_freq+core_bw, 
               alpha=0.20, color='red', label="science core")
    
    for pf in comb_peaks:
        plt.axvline(pf, ls="--", lw=1, color='gray', alpha=0.5)
        
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Amplitude (arb/Jy)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comb_resid(metrics: SubtractMetrics, save_path: str | None = None):
    """Plot comb peak residuals."""
    items = sorted(metrics.comb_peaks.items(), key=lambda kv: kv[0])
    if not items:
        print("No comb peaks to plot.")
        return
        
    xs = [str(k) for k,_ in items]
    ys = [v for _,v in items]
    
    plt.figure(figsize=(8,3))
    plt.bar(xs, ys, color='steelblue', alpha=0.7)
    plt.xlabel("Comb peak (MHz)")
    plt.ylabel("RMS residual")
    plt.title("Comb Peak Residuals")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
