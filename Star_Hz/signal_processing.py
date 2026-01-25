"""
Signal processing utilities: masking, regression, amplitude estimation.
"""
from typing import List
import numpy as np
from data_generator import gaussian

def make_band_mask(freqs: np.ndarray, center_mhz: float, half_bw_mhz: float) -> np.ndarray:
    """Create frequency band mask."""
    return (np.abs(freqs - center_mhz) <= half_bw_mhz)


def make_multi_band_mask(freqs: np.ndarray, centers_mhz: List[float], 
                         half_bw_mhz: float) -> np.ndarray:
    """Create mask for multiple frequency bands."""
    m = np.zeros_like(freqs, dtype=bool)
    for c in centers_mhz:
        m |= make_band_mask(freqs, c, half_bw_mhz)
    return m


def poly_design_matrix(x: np.ndarray, order: int) -> np.ndarray:
    """Polynomial design matrix [1, x, x^2, ..., x^order]."""
    X = [np.ones_like(x)]
    for k in range(1, order + 1):
        X.append(x**k)
    return np.vstack(X).T


def mad_sigma(r: np.ndarray) -> float:
    """Robust standard deviation via MAD."""
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    return 1.4826 * mad


def clipped_least_squares(A: np.ndarray, y: np.ndarray, 
                          clip_k: float = 2.5, n_iter: int = 3) -> np.ndarray:
    """Iteratively reweighted least squares with MAD-based clipping."""
    w = np.ones_like(y, dtype=float)
    coef = np.zeros(A.shape[1], dtype=float)
    
    for _ in range(n_iter):
        Aw = A * w[:, None]
        yw = y * w
        coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        r = y - A @ coef
        s = mad_sigma(r)
        
        if not np.isfinite(s) or s <= 0:
            break
            
        w = (np.abs(r) <= clip_k * s).astype(float)
        if w.sum() < max(10, 0.05 * len(w)):
            w[:] = 1.0
            break
            
    return coef


def estimate_target_amp_matched(
    spec: np.ndarray,
    freqs: np.ndarray,
    target_freq: float,
    target_sigma: float,
    core_bw: float,
    side_bw: float,
    poly_order: int = 1,
    clip_k: float = 2.5,
) -> float:
    """
    Estimate science target amplitude via matched filter.
    
    Steps:
    1. Fit polynomial baseline on sideband (side_bw region excluding core_bw)
    2. Subtract baseline from core region
    3. Apply matched filter with Gaussian template
    """
    core = make_band_mask(freqs, target_freq, core_bw)
    side = make_band_mask(freqs, target_freq, side_bw) & (~core)

    if core.sum() < 3:
        return 0.0
        
    if side.sum() < max(6, poly_order + 2):
        # Fallback: no baseline subtraction
        x_core = freqs[core].astype(float)
        y_core = spec[core].astype(float)
        g = gaussian(x_core, target_freq, target_sigma).astype(float)
        gg = float(np.dot(g, g) + 1e-12)
        return float(np.dot(g, y_core) / gg)

    # Fit baseline on sideband
    x_side = freqs[side].astype(float) - float(target_freq)
    y_side = spec[side].astype(float)
    P_side = poly_design_matrix(x_side, poly_order)
    b = clipped_least_squares(P_side, y_side, clip_k=clip_k, n_iter=3)

    # Subtract baseline from core
    x_core = freqs[core].astype(float) - float(target_freq)
    P_core = poly_design_matrix(x_core, poly_order)
    baseline_core = P_core @ b
    y_core = spec[core].astype(float) - baseline_core

    # Matched filter
    g = gaussian(freqs[core].astype(float), target_freq, target_sigma).astype(float)
    gg = float(np.dot(g, g) + 1e-12)
    amp = float(np.dot(g, y_core) / gg)
    
    return max(0.0, amp)
