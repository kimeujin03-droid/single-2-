"""
SVD/PCA-based UEMR (Starlink-like) subtraction demo with:
  1) SVD/PCA template (rank-r) instead of median template
  2) Protected science band + robust baseline + matched-filter amplitude
  3) 2-stage model (Stage1 broadband+comb, Stage2 comb-only)
  4) Smooth time gating using measured satellite strength (NO hard edges)

Single-file, hackable.
"""

from __future__ import annotations

import json
import math
import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 0) Synthetic data generator (paper-inspired toy model)
# =============================================================================

@dataclass
class SimParams:
    freq_min_mhz: float = 54.0
    freq_max_mhz: float = 66.0
    df_mhz: float = 0.05          # 50 kHz
    dt_s: float = 1.0             # 1 second
    duration_s: float = 60.0      # total duration

    flux_sat_peak_jy: float = 500.0
    flux_bg_mean_jy: float = 5.0
    flux_bg_sigma_jy: float = 1.0

    science_freq_mhz: float = 60.0
    science_flux_jy: float = 10.0
    science_sigma_mhz: float = 0.20

    comb_peak_freqs_mhz: Tuple[float, ...] = (56.0, 58.5, 61.0, 63.5)
    comb_peak_sigma_mhz: float = 0.30

    satellite_time_mu_s: float = 30.0
    satellite_time_sigma_s: float = 10.0

    broadband_floor: float = 0.2  # relative floor


def make_time_freq_axes(p: SimParams) -> Tuple[np.ndarray, np.ndarray]:
    n_time = int(round(p.duration_s / p.dt_s))
    n_freq = int(round((p.freq_max_mhz - p.freq_min_mhz) / p.df_mhz))
    freqs = np.linspace(p.freq_min_mhz, p.freq_max_mhz, n_freq, endpoint=False)
    times = np.arange(n_time, dtype=float) * p.dt_s
    return times, freqs


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12))**2)


def generate_synthetic_cube(p: SimParams, seed: int = 2025) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    times, freqs = make_time_freq_axes(p)
    n_time, n_freq = len(times), len(freqs)

    bg = rng.normal(p.flux_bg_mean_jy, p.flux_bg_sigma_jy, size=(n_time, n_freq)).astype(np.float32)

    sci_prof = gaussian(freqs, p.science_freq_mhz, p.science_sigma_mhz).astype(np.float32)
    sci = (p.science_flux_jy * sci_prof)[None, :].repeat(n_time, axis=0).astype(np.float32)

    t_env = gaussian(times, p.satellite_time_mu_s, p.satellite_time_sigma_s).astype(np.float32)
    broadband = (p.broadband_floor * np.ones_like(freqs, dtype=np.float32))
    peaks = np.zeros_like(freqs, dtype=np.float32)
    for pf in p.comb_peak_freqs_mhz:
        peaks += gaussian(freqs, pf, p.comb_peak_sigma_mhz).astype(np.float32)

    sat_f = broadband + peaks
    sat_f /= float(np.max(sat_f) + 1e-12)
    sat = (p.flux_sat_peak_jy * (t_env[:, None] * sat_f[None, :])).astype(np.float32)

    raw = (bg + sci + sat).astype(np.float32)

    return {
        "times": times.astype(np.float32),
        "freqs": freqs.astype(np.float32),
        "background": bg,
        "science": sci,
        "satellite": sat,
        "raw": raw,
    }


# =============================================================================
# 1) Utility: robust regression, masks, target amplitude estimation
# =============================================================================

def make_band_mask(freqs: np.ndarray, center_mhz: float, half_bw_mhz: float) -> np.ndarray:
    return (np.abs(freqs - center_mhz) <= half_bw_mhz)


def make_multi_band_mask(freqs: np.ndarray, centers_mhz: List[float], half_bw_mhz: float) -> np.ndarray:
    m = np.zeros_like(freqs, dtype=bool)
    for c in centers_mhz:
        m |= make_band_mask(freqs, c, half_bw_mhz)
    return m


def poly_design_matrix(x: np.ndarray, order: int) -> np.ndarray:
    X = [np.ones_like(x)]
    for k in range(1, order + 1):
        X.append(x**k)
    return np.vstack(X).T


def mad_sigma(r: np.ndarray) -> float:
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    return 1.4826 * mad


def clipped_least_squares(A: np.ndarray, y: np.ndarray, clip_k: float = 2.5, n_iter: int = 3) -> np.ndarray:
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
    Stable target amplitude estimate:
      - Fit polynomial baseline on sideband region (within side_bw but excluding core_bw)
      - Subtract baseline on core
      - Matched-filter with gaussian template on core

    This is far less fragile than "fit g + poly only inside protect".
    """
    core = make_band_mask(freqs, target_freq, core_bw)
    side = make_band_mask(freqs, target_freq, side_bw) & (~core)

    if core.sum() < 3:
        return 0.0
    if side.sum() < max(6, poly_order + 2):
        # fallback: no baseline, just matched filter on raw core
        x_core = freqs[core].astype(float)
        y_core = spec[core].astype(float)
        g = gaussian(x_core, target_freq, target_sigma).astype(float)
        gg = float(np.dot(g, g) + 1e-12)
        return float(np.dot(g, y_core) / gg)

    x_side = freqs[side].astype(float) - float(target_freq)
    y_side = spec[side].astype(float)
    P_side = poly_design_matrix(x_side, poly_order)
    b = clipped_least_squares(P_side, y_side, clip_k=clip_k, n_iter=3)

    x_core = freqs[core].astype(float) - float(target_freq)
    P_core = poly_design_matrix(x_core, poly_order)
    baseline_core = P_core @ b

    y_core = spec[core].astype(float) - baseline_core

    g = gaussian(freqs[core].astype(float), target_freq, target_sigma).astype(float)
    gg = float(np.dot(g, g) + 1e-12)
    amp = float(np.dot(g, y_core) / gg)
    return max(0.0, amp)


# =============================================================================
# 2) SVD/PCA subtraction core
# =============================================================================

@dataclass
class SubtractConfig:
    sat_window: Tuple[int, int] = (25, 35)
    edges_margin: int = 15
    shoulder: int = 4

    protect_bw: float = 0.4          # PBW (overall protect region)
    science_core_bw: float = 0.15    # CORE region inside protect to preserve science
    notch_bw: float = 0.10           # NBW for comb peaks

    # stage1
    r1: int = 2
    poly_order: int = 2
    clip_k: float = 2.5
    subtract_protect: bool = True

    # gating/safety
    safety_floor_q: float = 0.0      # default OFF (avoid science destruction)
    gate_taper_s: float = 6.0        # smooth in time (works with strength gate below)

    # stage2 comb-only
    r2: int = 1

    # strength gate
    gate_strength_k: float = 3.0     # threshold = median(off)+k*MAD(off)
    gate_strength_slope: float = 1.5 # logistic slope (bigger -> sharper)


@dataclass
class SubtractMetrics:
    reduction_db: float
    off_rms_ratio_all: float
    off_rms_ratio_protect: float
    off_rms_ratio_shoulder: float

    pres_off: float
    pres_sat: float

    comb_resid: float
    comb_peaks: Dict[float, float]
    trough_min: float

    clamp_active_ratio: float
    notch_override_energy: float
    s_ratio_01_stage1: float
    s_ratio_01_stage2: float

    # new diagnostics
    gate_mean_off: float
    gate_mean_sat: float


def compute_svd_basis(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, S, Vt


def learn_basis_from_window(
    data: np.ndarray,
    time_idx: np.ndarray,
    freq_mask: np.ndarray,
    center_mode: str = "median",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = data[time_idx][:, freq_mask].astype(float)

    if center_mode == "median":
        mu = np.median(X, axis=0)
    elif center_mode == "mean":
        mu = np.mean(X, axis=0)
    else:
        raise ValueError(f"Unknown center_mode: {center_mode}")

    Xc = X - mu[None, :]
    U, S, Vt = compute_svd_basis(Xc)
    return Xc, mu, U, S, Vt


def fit_rank_r_model(
    y_full: np.ndarray,
    freqs: np.ndarray,
    freq_mask_fit: np.ndarray,
    freq_mask_basis: np.ndarray,
    mu: np.ndarray,
    Vt: np.ndarray,
    r: int,
    poly_order: int,
    clip_k: float,
) -> Tuple[np.ndarray, float]:
    """
    Fit (rank-r SVD basis) + poly baseline on freq_mask_fit.
    Reconstruct basis part on freq_mask_basis.
    """
    idx_basis = np.where(freq_mask_basis)[0]
    idx_fit = np.where(freq_mask_fit)[0]
    if len(idx_fit) < max(10, r + poly_order + 2):
        # Not enough points -> return zero model
        model_full = np.full_like(y_full, np.nan, dtype=float)
        model_full[freq_mask_basis] = mu.copy()
        return model_full, 0.0

    # Vr in basis-space
    Vr = Vt[:r, :]  # (r, n_basis_features)

    # positions of fit indices inside basis indices
    pos = np.searchsorted(idx_basis, idx_fit)
    Vr_fit = Vr[:, pos]          # (r, n_fit)
    B = Vr_fit.T                 # (n_fit, r)

    x = freqs[idx_fit].astype(float)
    x0 = float(np.mean(x))
    P = poly_design_matrix(x - x0, poly_order)

    A = np.hstack([B, P])
    y_fit = y_full[idx_fit].astype(float)
    mu_fit = mu[pos].astype(float)

    coef = clipped_least_squares(A, y_fit - mu_fit, clip_k=clip_k, n_iter=3)
    c = coef[:r]

    model_masked = mu + (c @ Vr)
    model_full = np.full_like(y_full, np.nan, dtype=float)
    model_full[freq_mask_basis] = model_masked

    notch_energy = float(np.sum((B @ c)**2))
    return model_full, notch_energy


def apply_safety_floor(clean_row: np.ndarray, q: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Safer clamp: default OFF.
    If enabled, clamp only extreme negative tails based on robust stats of clean_row itself.
    """
    q = float(np.clip(q, 0.0, 1.0))
    if q <= 0.0:
        return clean_row, 0.0

    med = float(np.median(clean_row))
    mad = float(np.median(np.abs(clean_row - med)) + 1e-12)
    sigma = 1.4826 * mad

    k = float(np.clip(6.0 - 4.0 * q, 1.5, 6.0))
    floor = med - k * sigma

    below = clean_row < floor
    ratio = float(np.mean(below))
    out = clean_row.copy()
    out[below] = floor
    return out, ratio


def raised_cosine_taper(n_time: int, sat_window: Tuple[int, int], taper_s: float) -> np.ndarray:
    if taper_s <= 0:
        return np.ones(n_time, dtype=float)
    t0, t1 = sat_window
    L = int(round(taper_s))
    w = np.zeros(n_time, dtype=float)
    w[t0:t1] = 1.0

    a0 = max(0, t0 - L)
    a1 = t0
    if a1 > a0:
        u = np.linspace(0, 1, a1 - a0, endpoint=False)
        w[a0:a1] = 0.5 - 0.5 * np.cos(math.pi * u)

    b0 = t1
    b1 = min(n_time, t1 + L)
    if b1 > b0:
        u = np.linspace(0, 1, b1 - b0, endpoint=False)
        w[b0:b1] = 0.5 - 0.5 * np.cos(math.pi * (1 - u))
    return w


def satellite_strength_gate(
    raw: np.ndarray,
    freqs: np.ndarray,
    sat_window: Tuple[int, int],
    edges_margin: int,
    comb_peaks: List[float],
    notch_bw: float,
    protect: np.ndarray,
    k_mad: float,
    slope: float,
) -> np.ndarray:
    """
    Measure sat strength per time using comb-band RMS (excluding protect),
    then create smooth weights via logistic around OFF baseline.
    This removes the "black pillars" at sat_window boundaries.
    """
    n_time = raw.shape[0]
    comb_mask = make_multi_band_mask(freqs, comb_peaks, notch_bw) & (~protect)
    if comb_mask.sum() < 3:
        return np.ones(n_time, dtype=float)

    strength = np.std(raw[:, comb_mask].astype(float), axis=1)  # (time,)

    em = int(edges_margin)
    off = np.zeros(n_time, dtype=bool)
    off[:em] = True
    off[-em:] = True

    base = float(np.median(strength[off]))
    sig = float(mad_sigma(strength[off]) + 1e-12)
    thr = base + float(k_mad) * sig

    # z-score relative to threshold and scale by slope
    z = (strength - thr) / (sig + 1e-12)
    w = 1.0 / (1.0 + np.exp(-float(slope) * z))  # logistic in (0,1)

    # also apply a gentle raised-cosine envelope around the provided sat_window
    # (prevents weird far-out activations in noisy cases)
    w *= raised_cosine_taper(n_time, sat_window, taper_s=6.0)

    return w.astype(float)


def orthogonalize_core_against_science(
    model_full: np.ndarray,
    freqs: np.ndarray,
    target_freq: float,
    target_sigma: float,
    core_mask: np.ndarray,
) -> None:
    """
    In-place: remove component of model along the science template within core_mask.
    This prevents subtracting the science line even if baseline leaks into core.
    """
    if core_mask.sum() < 3:
        return
    g = gaussian(freqs[core_mask].astype(float), target_freq, target_sigma).astype(float)
    gg = float(np.dot(g, g) + 1e-12)
    m = model_full[core_mask].astype(float)
    proj = float(np.dot(g, m) / gg)
    model_full[core_mask] = m - proj * g


def run_two_stage_subtraction(
    raw: np.ndarray,
    freqs: np.ndarray,
    target_freq: float,
    target_sigma: float,
    comb_peaks: List[float],
    cfg: SubtractConfig,
    center_mode_stage1: str = "median",
) -> Tuple[np.ndarray, SubtractMetrics]:
    n_time, n_freq = raw.shape
    t0, t1 = cfg.sat_window

    em = int(cfg.edges_margin)
    off_mask_time = np.zeros(n_time, dtype=bool)
    off_mask_time[:em] = True
    off_mask_time[-em:] = True

    sat_mask_time = np.zeros(n_time, dtype=bool)
    sat_mask_time[t0:t1] = True

    shoulder = int(cfg.shoulder)
    sh_mask_time = np.zeros(n_time, dtype=bool)
    sh_mask_time[max(0, t0-shoulder):t0] = True
    sh_mask_time[t1:min(n_time, t1+shoulder)] = True

    protect = make_band_mask(freqs, target_freq, cfg.protect_bw)
    core = make_band_mask(freqs, target_freq, cfg.science_core_bw)
    comb_mask = make_multi_band_mask(freqs, comb_peaks, cfg.notch_bw)

    # Stage1 fit excludes protect (so basis isn't forced to explain science)
    basis1 = ~protect
    fit1 = ~protect

    Xc1, mu1, U1, S1, Vt1 = learn_basis_from_window(raw, np.arange(t0, t1), basis1, center_mode=center_mode_stage1)
    s_ratio_01_1 = float(S1[0] / (S1[1] + 1e-12)) if len(S1) > 1 else float("inf")

    # Smooth time weights: strength gate + optional taper
    w_gate = satellite_strength_gate(
        raw=raw,
        freqs=freqs,
        sat_window=cfg.sat_window,
        edges_margin=cfg.edges_margin,
        comb_peaks=comb_peaks,
        notch_bw=cfg.notch_bw,
        protect=protect,
        k_mad=cfg.gate_strength_k,
        slope=cfg.gate_strength_slope,
    )
    # user taper on top (optional)
    w_time = w_gate * raised_cosine_taper(n_time, cfg.sat_window, cfg.gate_taper_s)

    model1_all = np.zeros_like(raw, dtype=float)
    clamp_ratios = []
    notchE1 = 0.0

    for ti in range(n_time):
        model_masked, notch_energy = fit_rank_r_model(
            y_full=raw[ti],
            freqs=freqs,
            freq_mask_fit=fit1,
            freq_mask_basis=basis1,
            mu=mu1,
            Vt=Vt1,
            r=cfg.r1,
            poly_order=cfg.poly_order,
            clip_k=cfg.clip_k,
        )
        notchE1 += notch_energy

        model_full = np.zeros(n_freq, dtype=float)
        model_full[basis1] = model_masked[basis1]  # basis reconstruction on unprotected freqs

        if cfg.subtract_protect:
            # Baseline in protect from sideband poly fit (using fit1 region)
            x_fit = freqs[fit1].astype(float)
            x0 = float(np.mean(x_fit))
            y_fit = raw[ti, fit1].astype(float)
            P_fit = poly_design_matrix(x_fit - x0, cfg.poly_order)
            b = clipped_least_squares(P_fit, y_fit, clip_k=cfg.clip_k, n_iter=3)

            x_p = freqs[protect].astype(float)
            P_p = poly_design_matrix(x_p - x0, cfg.poly_order)
            model_full[protect] = P_p @ b

            # science core: orthogonalize against science template so we don't subtract it
            orthogonalize_core_against_science(model_full, freqs, target_freq, target_sigma, core)

        # Apply smooth time weight
        model_full *= float(w_time[ti])

        clean1 = raw[ti].astype(float) - model_full
        clean1, clamp_ratio = apply_safety_floor(clean1, q=cfg.safety_floor_q)
        clamp_ratios.append(clamp_ratio)

        model1_all[ti] = model_full

    cln1 = raw.astype(float) - model1_all
    cln1 = np.array([apply_safety_floor(cln1[i], q=cfg.safety_floor_q)[0] for i in range(n_time)])

    # Stage2: comb-only on cln1 (exclude protect)
    basis2 = comb_mask & (~protect)
    if basis2.sum() < 5 or cfg.r2 <= 0:
        cln2 = cln1.copy()
        s_ratio_01_2 = float("inf")
        notchE2 = 0.0
    else:
        Xc2, mu2, U2, S2, Vt2 = learn_basis_from_window(cln1, np.arange(t0, t1), basis2, center_mode="median")
        s_ratio_01_2 = float(S2[0] / (S2[1] + 1e-12)) if len(S2) > 1 else float("inf")

        model2_all = np.zeros_like(raw, dtype=float)
        notchE2 = 0.0
        for ti in range(n_time):
            model2_masked, notch_energy = fit_rank_r_model(
                y_full=cln1[ti],
                freqs=freqs,
                freq_mask_fit=basis2,
                freq_mask_basis=basis2,
                mu=mu2,
                Vt=Vt2,
                r=cfg.r2,
                poly_order=0,
                clip_k=cfg.clip_k,
            )
            notchE2 += notch_energy
            model2_full = np.zeros(n_freq, dtype=float)
            model2_full[basis2] = model2_masked[basis2]
            model2_full *= float(w_time[ti])
            model2_all[ti] = model2_full

        cln2 = cln1 - model2_all
        cln2 = np.array([apply_safety_floor(cln2[i], q=cfg.safety_floor_q)[0] for i in range(n_time)])

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    raw_sat = raw[sat_mask_time][:, ~protect]
    cln_sat = cln2[sat_mask_time][:, ~protect]
    rms_raw = np.std(raw_sat)
    rms_cln = np.std(cln_sat) + 1e-12
    reduction_db = float(20.0 * np.log10((rms_raw + 1e-12) / rms_cln))

    raw_off_all = raw[off_mask_time].astype(float)
    cln_off_all = cln2[off_mask_time].astype(float)
    off_rms_ratio_all = float(np.std(cln_off_all) / (np.std(raw_off_all) + 1e-12))

    raw_off_p = raw[off_mask_time][:, protect].astype(float)
    cln_off_p = cln2[off_mask_time][:, protect].astype(float)
    off_rms_ratio_protect = float(np.std(cln_off_p) / (np.std(raw_off_p) + 1e-12))

    if sh_mask_time.any():
        raw_sh = raw[sh_mask_time].astype(float)
        cln_sh = cln2[sh_mask_time].astype(float)
        off_rms_ratio_shoulder = float(np.std(cln_sh) / (np.std(raw_sh) + 1e-12))
    else:
        off_rms_ratio_shoulder = float("nan")

    # science preservation via matched estimate (OFF reference)
    ref_target_amp = float(np.mean([
        estimate_target_amp_matched(raw[i], freqs, target_freq, target_sigma,
                                    core_bw=cfg.science_core_bw, side_bw=max(cfg.protect_bw, 0.6),
                                    poly_order=1, clip_k=cfg.clip_k)
        for i in np.where(off_mask_time)[0]
    ]) + 1e-12)

    target_off = float(np.mean([
        estimate_target_amp_matched(cln2[i], freqs, target_freq, target_sigma,
                                    core_bw=cfg.science_core_bw, side_bw=max(cfg.protect_bw, 0.6),
                                    poly_order=1, clip_k=cfg.clip_k)
        for i in np.where(off_mask_time)[0]
    ]))
    target_sat = float(np.mean([
        estimate_target_amp_matched(cln2[i], freqs, target_freq, target_sigma,
                                    core_bw=cfg.science_core_bw, side_bw=max(cfg.protect_bw, 0.6),
                                    poly_order=1, clip_k=cfg.clip_k)
        for i in np.where(sat_mask_time)[0]
    ]))

    pres_off = float(target_off / ref_target_amp)
    pres_sat = float(target_sat / ref_target_amp)

    # Comb residual
    comb_peaks_res = {}
    comb_vals = []
    for pf in comb_peaks:
        m = make_band_mask(freqs, pf, cfg.notch_bw) & (~protect)
        if m.sum() < 2:
            continue
        v = cln2[sat_mask_time][:, m].astype(float)
        rr = float(np.std(v))
        comb_peaks_res[float(pf)] = rr
        comb_vals.append(rr)
    comb_resid = float(np.mean(comb_vals) if len(comb_vals) else 0.0)

    trough_min = float(np.min(cln2))
    clamp_active_ratio = float(np.mean(clamp_ratios)) if len(clamp_ratios) else 0.0
    notch_override_energy = float((notchE1 + notchE2))

    metrics = SubtractMetrics(
        reduction_db=reduction_db,
        off_rms_ratio_all=off_rms_ratio_all,
        off_rms_ratio_protect=off_rms_ratio_protect,
        off_rms_ratio_shoulder=off_rms_ratio_shoulder,
        pres_off=pres_off,
        pres_sat=pres_sat,
        comb_resid=comb_resid,
        comb_peaks=comb_peaks_res,
        trough_min=trough_min,
        clamp_active_ratio=clamp_active_ratio,
        notch_override_energy=notch_override_energy,
        s_ratio_01_stage1=s_ratio_01_1,
        s_ratio_01_stage2=s_ratio_01_2,
        gate_mean_off=float(np.mean(w_time[off_mask_time])),
        gate_mean_sat=float(np.mean(w_time[sat_mask_time])),
    )

    return cln2.astype(np.float32), metrics


# =============================================================================
# 3) Scoring + Grid Sweep
# =============================================================================

def score_config_balanced(
    m: SubtractMetrics,
    r1: int,
    r2: int,
    w_red: float = 10.0,
    w_off: float = 25.0,
    w_off_prot: float = 15.0,
    w_pres_off: float = 80.0,
    w_pres_sat: float = 60.0,
    w_comb: float = 4.0,
    w_r: float = 1.0,
    w_trough: float = 5.0,
    w_shoulder: float = 8.0,
    w_gate: float = 15.0,
) -> float:
    off_pen = abs(math.log(m.off_rms_ratio_all + 1e-12))
    offp_pen = abs(math.log(m.off_rms_ratio_protect + 1e-12))
    sh_pen = 0.0 if not np.isfinite(m.off_rms_ratio_shoulder) else abs(math.log(m.off_rms_ratio_shoulder + 1e-12))

    # gate sanity: off should be near 0, sat should be higher
    gate_pen = abs(m.gate_mean_off - 0.0) + abs(m.gate_mean_sat - 1.0)

    score = 0.0
    score += w_red * m.reduction_db
    score -= w_off * off_pen
    score -= w_off_prot * offp_pen
    score -= w_shoulder * sh_pen
    score -= w_pres_off * abs(m.pres_off - 1.0)
    score -= w_pres_sat * abs(m.pres_sat - 1.0)
    score -= w_comb * m.comb_resid
    score -= w_r * float(r1 + r2)
    score -= w_trough * max(0.0, -m.trough_min)
    score -= w_gate * gate_pen
    return float(score)


def sweep_grid(
    raw: np.ndarray,
    freqs: np.ndarray,
    target_freq: float,
    target_sigma: float,
    comb_peaks: List[float],
    sat_windows: List[Tuple[int, int]],
    edges_margins: List[int],
    protect_bws: List[float],
    core_bws: List[float],
    notch_bws: List[float],
    r1_list: List[int],
    r2_list: List[int],
    poly_orders: List[int],
    clip_ks: List[float],
    safety_qs: List[float],
    tapers: List[float],
    top_k: int = 10,
    verbose_every: int = 200,
) -> Tuple[Dict, List[Dict]]:
    results = []
    total = 0

    for (satw, em, pbw, cbw, nbw, r1, r2, poly, ck, q, taper) in itertools.product(
        sat_windows, edges_margins, protect_bws, core_bws, notch_bws,
        r1_list, r2_list, poly_orders, clip_ks, safety_qs, tapers
    ):
        total += 1
        cfg = SubtractConfig(
            sat_window=satw,
            edges_margin=em,
            protect_bw=pbw,
            science_core_bw=cbw,
            notch_bw=nbw,
            r1=r1,
            r2=r2,
            poly_order=poly,
            clip_k=ck,
            subtract_protect=True,
            safety_floor_q=q,
            gate_taper_s=taper,
        )

        cln, m = run_two_stage_subtraction(
            raw=raw,
            freqs=freqs,
            target_freq=target_freq,
            target_sigma=target_sigma,
            comb_peaks=comb_peaks,
            cfg=cfg,
            center_mode_stage1="median",
        )

        sc = score_config_balanced(m, r1=r1, r2=r2)

        rec = {"score": sc, "cfg": asdict(cfg), "metrics": asdict(m)}
        results.append(rec)

        if verbose_every and (total % verbose_every == 0):
            print(f"[{total}] score={sc:.2f} red={m.reduction_db:.2f}dB presOFF={m.pres_off*100:.1f}% presSAT={m.pres_sat*100:.1f}% comb={m.comb_resid:.2f}")

    results_sorted = sorted(results, key=lambda d: d["score"], reverse=True)
    top = results_sorted[:top_k]
    best = top[0] if top else {}
    return best, top


# =============================================================================
# 4) Plots + Save helpers
# =============================================================================

def plot_heatmaps(raw: np.ndarray, cln: np.ndarray, freqs: np.ndarray, title_prefix: str = ""):
    n_time = raw.shape[0]
    t = np.arange(n_time)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    extent = [t[0], t[-1], float(freqs[0]), float(freqs[-1])]

    im0 = axes[0].imshow(raw.T, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    axes[0].set_title(f"{title_prefix}(a) RAW")
    axes[0].set_ylabel("Frequency (MHz)")
    plt.colorbar(im0, ax=axes[0], fraction=0.02)

    im1 = axes[1].imshow(cln.T, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    axes[1].set_title(f"{title_prefix}(b) CLEANED")
    axes[1].set_ylabel("Frequency (MHz)")
    axes[1].set_xlabel("Time (s)")
    plt.colorbar(im1, ax=axes[1], fraction=0.02)

    plt.tight_layout()
    plt.show()


def plot_mean_spectra(raw: np.ndarray, cln: np.ndarray, freqs: np.ndarray,
                      sat_window: Tuple[int,int], edges_margin: int,
                      target_freq: float, protect_bw: float, core_bw: float,
                      comb_peaks: List[float]):
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
    plt.plot(freqs, raw_off, label="RAW OFF")
    plt.plot(freqs, raw_sat, label="RAW SAT")
    plt.plot(freqs, cln_off, label="CLN OFF")
    plt.plot(freqs, cln_sat, label="CLN SAT")
    plt.axvspan(target_freq-protect_bw, target_freq+protect_bw, alpha=0.12, label="protect")
    plt.axvspan(target_freq-core_bw, target_freq+core_bw, alpha=0.20, label="science core")
    for pf in comb_peaks:
        plt.axvline(pf, ls="--", lw=1)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Amplitude (arb/Jy)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def track_target_amp_over_time(data: np.ndarray, freqs: np.ndarray,
                               target_freq: float, target_sigma: float,
                               core_bw: float, side_bw: float, clip_k: float) -> np.ndarray:
    amps = []
    for i in range(data.shape[0]):
        a = estimate_target_amp_matched(
            data[i], freqs, target_freq, target_sigma,
            core_bw=core_bw, side_bw=side_bw, poly_order=1, clip_k=clip_k
        )
        amps.append(a)
    return np.array(amps, dtype=float)


def plot_target_amp_t(raw: np.ndarray, cln: np.ndarray, freqs: np.ndarray,
                      target_freq: float, target_sigma: float,
                      core_bw: float, side_bw: float, clip_k: float):
    a_raw = track_target_amp_over_time(raw, freqs, target_freq, target_sigma, core_bw, side_bw, clip_k)
    a_cln = track_target_amp_over_time(cln, freqs, target_freq, target_sigma, core_bw, side_bw, clip_k)

    plt.figure(figsize=(10,3))
    plt.plot(a_raw, label="RAW target_amp(t)")
    plt.plot(a_cln, label="CLN target_amp(t)")
    plt.xlabel("Time index")
    plt.ylabel("Estimated target amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comb_resid(metrics: SubtractMetrics):
    items = sorted(metrics.comb_peaks.items(), key=lambda kv: kv[0])
    if not items:
        print("No comb peaks to plot.")
        return
    xs = [str(k) for k,_ in items]
    ys = [v for _,v in items]
    plt.figure(figsize=(8,3))
    plt.bar(xs, ys)
    plt.xlabel("Comb peak (MHz)")
    plt.ylabel("RMS residual")
    plt.tight_layout()
    plt.show()


def save_best_json(best: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)


# =============================================================================
# 5) Demo main
# =============================================================================

def main():
    p = SimParams(df_mhz=0.05, dt_s=1.0)
    sim = generate_synthetic_cube(p, seed=2025)
    raw = sim["raw"]
    freqs = sim["freqs"]

    print(f"dt={p.dt_s:.2f}s | df={p.df_mhz*1000:.1f}kHz | raw shape={raw.shape}")

    # --- Single run (추천 기본값)
    cfg0 = SubtractConfig(
        sat_window=(15, 45),         # ★중요: sigma=10이면 25-35는 너무 좁음. 잔류/경계 문제 유발.
        edges_margin=15,
        shoulder=4,

        protect_bw=0.40,
        science_core_bw=0.15,        # science core는 좁게 (보존 우선)
        notch_bw=0.10,

        r1=2,
        r2=1,
        poly_order=2,
        clip_k=2.5,

        subtract_protect=True,       # ★핵심: protect도 baseline은 빼되 core는 직교화로 보존
        safety_floor_q=0.0,

        gate_taper_s=6.0,            # ★경계 부드럽게
        gate_strength_k=3.0,
        gate_strength_slope=1.5,
    )

    cln0, m0 = run_two_stage_subtraction(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        cfg=cfg0
    )

    print("\n[Single Run]")
    print("cfg:", cfg0)
    print("metrics:", m0)

    plot_heatmaps(raw, cln0, freqs, title_prefix="SingleRun ")
    plot_mean_spectra(raw, cln0, freqs, cfg0.sat_window, cfg0.edges_margin,
                      p.science_freq_mhz, cfg0.protect_bw, cfg0.science_core_bw,
                      list(p.comb_peak_freqs_mhz))
    plot_target_amp_t(raw, cln0, freqs, p.science_freq_mhz, p.science_sigma_mhz,
                      cfg0.science_core_bw, side_bw=max(cfg0.protect_bw, 0.6), clip_k=cfg0.clip_k)
    plot_comb_resid(m0)

    # --- Grid sweep (적당히)
    print("\n[Grid Sweep Setup]")
    print(f"- dt: {p.dt_s:.2f} s | df: {p.df_mhz*1000:.1f} kHz")
    print(f"- target: {p.science_freq_mhz} MHz | science_flux: {p.science_flux_jy} Jy")
    print("------------------------------------------------")

    best, top = sweep_grid(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        sat_windows=[(12,48),(15,45),(18,42)],  # ★sat_window 넓혀야 경계/잔류가 해결됨
        edges_margins=[15],
        protect_bws=[0.30,0.40,0.50],
        core_bws=[0.10,0.15,0.20],
        notch_bws=[0.08,0.10,0.15],
        r1_list=[1,2,3,4],
        r2_list=[0,1,2],
        poly_orders=[1,2,3],
        clip_ks=[2.0,2.5,3.0],
        safety_qs=[0.0],
        tapers=[4.0,6.0,8.0],
        top_k=10,
        verbose_every=0,
    )

    print("\n=== Top 10 configs ===")
    for rec in top:
        cfg = rec["cfg"]
        m = rec["metrics"]
        print(
            f"score={rec['score']:.2f} | r1={cfg['r1']},r2={cfg['r2']} | sat={tuple(cfg['sat_window'])} | "
            f"PBW={cfg['protect_bw']:.2f} CORE={cfg['science_core_bw']:.2f} NBW={cfg['notch_bw']:.2f} | "
            f"poly={cfg['poly_order']} clipK={cfg['clip_k']} taper={cfg['gate_taper_s']} | "
            f"red={m['reduction_db']:.2f}dB off={m['off_rms_ratio_all']:.2f} sh={m['off_rms_ratio_shoulder']:.2f} | "
            f"presOFF={m['pres_off']*100:.1f}% presSAT={m['pres_sat']*100:.1f}% comb={m['comb_resid']:.2f} | "
            f"gateOFF={m['gate_mean_off']:.2f} gateSAT={m['gate_mean_sat']:.2f}"
        )

    print("\n✅ Best cfg:", best["cfg"])
    print("✅ Best metrics:", best["metrics"])

    save_best_json(best, "best_config.json")
    print("Saved: best_config.json")

    best_cfg = SubtractConfig(**best["cfg"])
    cln_best, m_best = run_two_stage_subtraction(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        cfg=best_cfg
    )
    plot_heatmaps(raw, cln_best, freqs, title_prefix="BEST ")
    plot_mean_spectra(raw, cln_best, freqs, best_cfg.sat_window, best_cfg.edges_margin,
                      p.science_freq_mhz, best_cfg.protect_bw, best_cfg.science_core_bw,
                      list(p.comb_peak_freqs_mhz))
    plot_target_amp_t(raw, cln_best, freqs, p.science_freq_mhz, p.science_sigma_mhz,
                      best_cfg.science_core_bw, side_bw=max(best_cfg.protect_bw, 0.6), clip_k=best_cfg.clip_k)
    plot_comb_resid(m_best)


if __name__ == "__main__":
    main()
