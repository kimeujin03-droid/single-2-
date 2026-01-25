"""
Core PCA/SVD-based satellite interference subtraction algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import math
import numpy as np
from signal_processing import (
    make_band_mask, make_multi_band_mask, poly_design_matrix,
    clipped_least_squares, mad_sigma, estimate_target_amp_matched
)
from data_generator import gaussian


@dataclass
class SubtractConfig:
    """Configuration for two-stage subtraction algorithm."""
    sat_window: Tuple[int, int] = (25, 35)
    edges_margin: int = 15
    shoulder: int = 4

    protect_bw: float = 0.4
    science_core_bw: float = 0.15
    notch_bw: float = 0.10

    r1: int = 2
    r2: int = 1
    poly_order: int = 2
    clip_k: float = 2.5
    subtract_protect: bool = True

    safety_floor_q: float = 0.0
    gate_taper_s: float = 6.0

    gate_strength_k: float = 3.0
    gate_strength_slope: float = 1.5

    # science restoration guard (prevents catastrophic overshoot)
    restore_science: bool = True
    restore_max_frac: float = 0.5  # max |delta| as fraction of ref_target_amp


@dataclass
class SubtractMetrics:
    """Metrics for evaluating subtraction quality."""
    reduction_db: float
    off_rms_ratio_all: float
    off_rms_ratio_protect: float
    off_rms_ratio_shoulder: float

    pres_off: float
    pres_sat: float

    comb_resid: float
    comb_peaks: dict
    trough_min: float

    clamp_active_ratio: float
    notch_override_energy: float
    s_ratio_01_stage1: float
    s_ratio_01_stage2: float

    gate_mean_off: float
    gate_mean_sat: float


def compute_svd_basis(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD: X = U S V^T."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U, S, Vt


def learn_basis_from_window(
    data: np.ndarray,
    time_idx: np.ndarray,
    freq_mask: np.ndarray,
    center_mode: str = "median",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Learn SVD basis from specified time window and frequency mask."""
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
    """Fit rank-r SVD model + polynomial baseline."""
    idx_basis = np.where(freq_mask_basis)[0]
    idx_fit = np.where(freq_mask_fit)[0]
    
    if len(idx_fit) < max(10, r + poly_order + 2):
        model_full = np.full_like(y_full, np.nan, dtype=float)
        model_full[freq_mask_basis] = mu.copy()
        return model_full, 0.0

    Vr = Vt[:r, :]
    pos = np.searchsorted(idx_basis, idx_fit)
    Vr_fit = Vr[:, pos]
    B = Vr_fit.T

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


def raised_cosine_taper(n_time: int, sat_window: Tuple[int, int], 
                       taper_s: float) -> np.ndarray:
    """Create smooth raised-cosine time window."""
    if taper_s <= 0:
        return np.ones(n_time, dtype=float)
        
    t0, t1 = sat_window
    L = int(round(taper_s))
    w = np.zeros(n_time, dtype=float)
    w[t0:t1] = 1.0

    # Left taper
    a0 = max(0, t0 - L)
    a1 = t0
    if a1 > a0:
        u = np.linspace(0, 1, a1 - a0, endpoint=False)
        w[a0:a1] = 0.5 - 0.5 * np.cos(math.pi * u)

    # Right taper
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
    Create smooth time gate based on measured satellite strength.
    Uses logistic function for smooth transitions.
    """
    n_time = raw.shape[0]
    comb_mask = make_multi_band_mask(freqs, comb_peaks, notch_bw) & (~protect)
    
    if comb_mask.sum() < 3:
        return np.ones(n_time, dtype=float)

    strength = np.std(raw[:, comb_mask].astype(float), axis=1)

    em = int(edges_margin)
    off = np.zeros(n_time, dtype=bool)
    off[:em] = True
    off[-em:] = True

    base = float(np.median(strength[off]))
    sig = float(mad_sigma(strength[off]) + 1e-12)
    thr = base + float(k_mad) * sig

    z = (strength - thr) / (sig + 1e-12)
    w = 1.0 / (1.0 + np.exp(-float(slope) * z))

    w *= raised_cosine_taper(n_time, sat_window, taper_s=6.0)

    return w.astype(float)


def orthogonalize_core_against_science(
    model_full: np.ndarray,
    freqs: np.ndarray,
    target_freq: float,
    target_sigma: float,
    core_mask: np.ndarray,
) -> None:
    """Remove science template component from model (in-place)."""
    if core_mask.sum() < 3:
        return
        
    g = gaussian(freqs[core_mask].astype(float), target_freq, target_sigma).astype(float)
    gg = float(np.dot(g, g) + 1e-12)
    m = model_full[core_mask].astype(float)
    proj = float(np.dot(g, m) / gg)
    model_full[core_mask] = m - proj * g


def apply_safety_floor(clean_row: np.ndarray, q: float = 0.0) -> Tuple[np.ndarray, float]:
    """Apply conservative safety floor (default: OFF)."""
    q = float(np.clip(q, 0.0, 1.0))
    if q <= 0.0:
        return clean_row, 0.0

    med = float(np.median(clean_row))
    mad_val = float(np.median(np.abs(clean_row - med)) + 1e-12)
    sigma = 1.4826 * mad_val

    k = float(np.clip(6.0 - 4.0 * q, 1.5, 6.0))
    floor = med - k * sigma

    below = clean_row < floor
    ratio = float(np.mean(below))
    out = clean_row.copy()
    out[below] = floor
    
    return out, ratio


def _run_two_stage_subtraction(
    raw: np.ndarray,
    freqs_mhz: np.ndarray,
    cfg: SubtractConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, SubtractMetrics]:
    """
    Two-stage UEMR subtraction designed to:
      (1) remove satellite-window excess (UEMR) while
      (2) preserving OFF-window baseline and the protected science band.

    Key change vs earlier versions:
      - The subtraction model is learned and applied in *delta space*:
        delta(t,f) = raw(t,f) - mu_off(f), where mu_off is the OFF-time median spectrum.
        This prevents subtracting the baseline/science mean (which was causing the "black block").
      - By default we do NOT "inpaint/subtract" inside protect; we leave protect untouched and
        optionally restore the science core amplitude after subtraction.

    raw shape: (n_time, n_freq)
    freqs_mhz shape: (n_freq,)
    """
    raw = np.asarray(raw, dtype=float)
    freqs_mhz = np.asarray(freqs_mhz, dtype=float)
    n_time, n_freq = raw.shape
    if freqs_mhz.shape[0] != n_freq:
        raise ValueError(f"freqs_mhz length {freqs_mhz.shape[0]} != n_freq {n_freq}")

    # ----------------------------
    # Time windows
    # ----------------------------
    t0, t1 = cfg.sat_window
    target_f = float(getattr(cfg, 'target_f_mhz', 60.0))
    t0 = int(max(0, min(n_time - 1, t0)))
    t1 = int(max(0, min(n_time, t1)))
    if t1 <= t0 + 1:
        raise ValueError(f"sat_window too small: {cfg.sat_window}")

    sat_mask_t = np.zeros(n_time, dtype=bool)
    sat_mask_t[t0:t1] = True

    # OFF = outside sat window with a guard band (edges_margin interpreted in *time bins*)
    guard = int(max(0, cfg.edges_margin))
    off_mask_t = np.ones(n_time, dtype=bool)
    off_mask_t[max(0, t0 - guard):min(n_time, t1 + guard)] = False

    off_idx = np.where(off_mask_t)[0]
    if off_idx.size == 0:
        # Fallback: use everything outside the core sat window
        off_mask_t = ~sat_mask_t
        off_idx = np.where(off_mask_t)[0]
    sat_idx = np.where(sat_mask_t)[0]

    # ----------------------------
    # Frequency masks
    # ----------------------------
    df = float(np.median(np.diff(freqs_mhz))) if n_freq > 1 else 0.0
    protect_mask = make_band_mask(freqs_mhz, target_f, cfg.protect_bw)
    science_core_mask = make_band_mask(freqs_mhz, target_f, cfg.science_core_bw)

    # Comb peaks: either user-provided or detected (if cfg has detect comb)
    comb_peaks = cfg.comb_peaks_mhz if getattr(cfg, "comb_peaks_mhz", None) else None
    if comb_peaks is None:
        comb_peaks = [56.0, 58.5, 61.0, 63.5]  # fallback

    comb_mask = make_multi_band_mask(freqs_mhz, comb_peaks, cfg.notch_bw)

    # Fit masks (we never fit using protect bins)
    fit1 = ~protect_mask
    fit2 = comb_mask & (~protect_mask)

    # ----------------------------
    # Time taper (smooth gating)
    # ----------------------------
    def compute_energy_gate(
        raw: np.ndarray,
        sat_mask_t: np.ndarray,
        fit_mask: np.ndarray,
        k_mad: float = 3.0,
        slope: float = 1.5,
    ) -> np.ndarray:
        """
        Compute a smooth time gate based on total energy (std) in `fit_mask` freqs.
        Returns None if not enough data to form gate.
        """
        n_time_local = raw.shape[0]
        if fit_mask.sum() < 3:
            return None

        off_mask_t_local = ~sat_mask_t
        if off_mask_t_local.sum() < 3:
            return None

        strength = np.std(raw[:, fit_mask].astype(float), axis=1)
        base = float(np.median(strength[off_mask_t_local]))
        sig = float(mad_sigma(strength[off_mask_t_local]) + 1e-12)
        thr = base + float(k_mad) * sig

        z = (strength - thr) / (sig + 1e-12)
        w = 1.0 / (1.0 + np.exp(-float(slope) * z))
        return w.astype(float)

    w_time = raised_cosine_taper(n_time, (t0, t1), cfg.gate_taper_s)
    # Optional amplitude-based gate (kept, but safe)
    if getattr(cfg, "gate_strength_k", 0.0) and getattr(cfg, "gate_strength_slope", 0.0):
        w_gate = compute_energy_gate(raw, sat_mask_t, fit1, cfg.gate_strength_k, cfg.gate_strength_slope)
        if w_gate is not None:
            w_time = w_time * w_gate

    # ----------------------------
    # Reference science amplitude (OFF)
    # ----------------------------
    # Use matched filter around the science core width to define "amp"
    sigma_mhz = max(1.5 * df, 0.5 * float(cfg.science_core_bw), 1e-6)
    off_amps = []
    for ti in off_idx:
        off_amps.append(
            estimate_target_amp_matched(raw[ti], freqs_mhz, target_f, sigma_mhz, core_bw=cfg.science_core_bw, side_bw=cfg.protect_bw)
        )
    ref_target_amp = float(np.median(off_amps)) if len(off_amps) else 0.0

    # ----------------------------
    # Stage 1: broadband excess removal on fit1 (delta-space)
    # ----------------------------
    mu_off1 = np.median(raw[off_idx][:, fit1], axis=0)
    X1 = raw[sat_idx][:, fit1] - mu_off1[None, :]

    # Robust clipping (optional)
    if cfg.clip_k and cfg.clip_k > 0:
        med = np.median(X1, axis=0)
        mad = np.median(np.abs(X1 - med[None, :]), axis=0) + 1e-12
        z = (X1 - med[None, :]) / (1.4826 * mad[None, :])
        X1 = np.clip(X1, -cfg.clip_k, cfg.clip_k) * (1.4826 * mad[None, :]) + med[None, :]

    if cfg.r1 > 0:
        U1, S1, Vt1 = np.linalg.svd(X1, full_matrices=False)
        Vr1 = Vt1[: int(cfg.r1), :]
    else:
        Vr1 = None

    cln1 = raw.copy()
    if Vr1 is not None and Vr1.size:
        for ti in range(n_time):
            y_delta = raw[ti, fit1] - mu_off1
            c = y_delta @ Vr1.T
            recon = c @ Vr1
            model_full = np.zeros(n_freq, dtype=float)
            model_full[fit1] = recon
            # never subtract inside protect band
            model_full[protect_mask] = 0.0
            model_full *= float(w_time[ti])
            cln1[ti] = raw[ti] - model_full
    else:
        cln1 = raw.copy()

    # ----------------------------
    # Stage 2: comb-line refinement using per-time Gaussian-template linear regression
    # We model each spectrum as: b0 + b1*f + sum_k a_k * Gaussian(f; f_k, sigma_k)
    # Fit excluding protect band, estimate per-time amplitudes a_k, smooth them with EMA,
    # then subtract only the comb-model (keep baseline b0,b1 untouched).
    # ----------------------------
    cln2 = cln1.copy()
    # build comb template design matrix (F x (2+K))
    comb_peaks_list = comb_peaks if comb_peaks is not None else []
    K = len(comb_peaks_list)
    if K > 0:
        F = n_freq
        X = np.zeros((F, 2 + K), dtype=float)
        X[:, 0] = 1.0
        X[:, 1] = (freqs_mhz - float(np.mean(freqs_mhz))) / (float(np.std(freqs_mhz) + 1e-12))
        # use notch_bw as approximate line width (sigma). allow per-line sigma if desired
        sigma_k = max(1e-3, float(cfg.notch_bw))
        for k, fk in enumerate(comb_peaks_list):
            X[:, 2 + k] = gaussian(freqs_mhz, fk, sigma_k)

        # fit mask excludes protect band
        fit_mask = ~protect_mask
        if fit_mask.sum() >= (2 + K):
            Xf = X[fit_mask]
            # precompute XtX_inv for ridge regression stability
            ridge = 1e-3
            XtX = Xf.T @ Xf
            XtX += ridge * np.eye(XtX.shape[0])
            try:
                XtX_inv = np.linalg.inv(XtX)
            except Exception:
                XtX_inv = np.linalg.pinv(XtX)

            # collect per-time raw amplitudes
            amps_raw = np.zeros((n_time, K), dtype=float)
            for ti in range(n_time):
                y = cln1[ti]
                yf = y[fit_mask]
                Xty = Xf.T @ yf
                beta = XtX_inv @ Xty  # length 2+K
                if K > 0:
                    amps_raw[ti, :] = beta[2:2 + K]

            # smooth amplitudes in time (EMA) to remove vertical discontinuities
            def ema_vec(arr, alpha=0.2):
                y = np.zeros_like(arr)
                if arr.shape[0] == 0:
                    return y
                y[0] = arr[0]
                for i in range(1, arr.shape[0]):
                    y[i] = alpha * arr[i] + (1 - alpha) * y[i - 1]
                return y

            amps_sm = np.zeros_like(amps_raw)
            alpha = getattr(cfg, 'amp_ema_alpha', 0.2)
            for k in range(K):
                amps_sm[:, k] = ema_vec(amps_raw[:, k], alpha=alpha)

            # subtract comb model (only comb templates) per time, respecting protect and time gate
            for ti in range(n_time):
                comb_model = X[:, 2:] @ amps_sm[ti]
                comb_model = comb_model.astype(float)
                comb_model[protect_mask] = 0.0
                comb_model *= float(w_time[ti])
                cln2[ti] = cln1[ti] - comb_model
        else:
            # not enough freqs to fit; keep cln2 == cln1
            cln2 = cln1.copy()
    else:
        cln2 = cln1.copy()

    # ----------------------------
    # Optional: restore science core amplitude (protect band remains untouched)
    # ----------------------------
    if getattr(cfg, "restore_science", True) and ref_target_amp > 0:
        max_delta = float(cfg.restore_max_frac) * float(ref_target_amp)
        g = gaussian(freqs_mhz, target_f, sigma_mhz)
        g = g / (np.sum(g[science_core_mask]) + 1e-12)  # normalize over science core
        g = g * science_core_mask.astype(float)
        for ti in range(n_time):
            amp_clean = estimate_target_amp_matched(cln2[ti], freqs_mhz, target_f, sigma_mhz, core_bw=cfg.science_core_bw, side_bw=cfg.protect_bw)
            delta = float(ref_target_amp - amp_clean)
            delta = float(np.clip(delta, -max_delta, max_delta))
            cln2[ti] = cln2[ti] + delta * g

    # ----------------------------
    # Safety floor / clamping (optional)
    # ----------------------------
    clamp_active = np.zeros(n_time, dtype=bool)
    if cfg.safety_floor_q and cfg.safety_floor_q > 0:
        # Per-frequency floor based on OFF distribution
        floor = np.quantile(cln2[off_idx], float(cfg.safety_floor_q), axis=0)
        for ti in range(n_time):
            before = cln2[ti].copy()
            cln2[ti] = np.maximum(cln2[ti], floor)
            clamp_active[ti] = np.any(cln2[ti] != before)

    # ----------------------------
    # Enforce protect-band passthrough as final step (guarantee)
    # This ensures the protect band is identical to input in the final output.
    if protect_mask.any():
        cln2[:, protect_mask] = raw[:, protect_mask]
        # Verification assertion: fail early if other code modified protect band
        assert np.allclose(cln2[:, protect_mask], raw[:, protect_mask]), 'Protect-band passthrough violated'

    # ----------------------------
    # Metrics (delta-preserving)
    # ----------------------------
    off_raw = raw[off_idx]
    off_cln = cln2[off_idx]
    sat_raw = raw[sat_idx]
    sat_cln = cln2[sat_idx]

    off_mu_full = np.mean(off_raw, axis=0)
    raw_excess = sat_raw[:, fit1] - off_mu_full[fit1][None, :]
    cln_excess = sat_cln[:, fit1] - off_mu_full[fit1][None, :]

    rms_raw_excess = float(np.sqrt(np.mean(raw_excess ** 2)) + 1e-12)
    rms_cln_excess = float(np.sqrt(np.mean(cln_excess ** 2)) + 1e-12)
    reduction_db = 20.0 * float(np.log10(rms_raw_excess / rms_cln_excess))

    off_rms_ratio_all = float(np.sqrt(np.mean(off_cln[:, fit1] ** 2)) / (np.sqrt(np.mean(off_raw[:, fit1] ** 2)) + 1e-12))
    off_rms_ratio_protect = float(np.sqrt(np.mean(off_cln[:, protect_mask] ** 2)) / (np.sqrt(np.mean(off_raw[:, protect_mask] ** 2)) + 1e-12))
    off_rms_ratio_shoulder = off_rms_ratio_all  # kept for compatibility

    # Preservation (target amp vs OFF reference)
    # Use OFF-derived baseline matched filter to avoid per-time sideband fits that can be corrupted
    off_idxs_local = off_idx
    sat_idxs_local = sat_idx

    if off_idxs_local.size:
        off_mu_full_local = np.mean(raw[off_idxs_local], axis=0).astype(float)
    else:
        off_mu_full_local = np.mean(raw, axis=0).astype(float)

    core_mask_local = make_band_mask(freqs_mhz, target_f, cfg.science_core_bw)
    x_core_local = freqs_mhz[core_mask_local].astype(float)
    g_local = gaussian(x_core_local, target_f, sigma_mhz).astype(float)
    gg_local = float(np.dot(g_local, g_local) + 1e-12)

    def matched_amp_against_off_local(spec_row: np.ndarray) -> float:
        y = spec_row[core_mask_local].astype(float) - off_mu_full_local[core_mask_local]
        amp = float(np.dot(g_local, y) / gg_local)
        return float(max(0.0, amp))

    # Keep reference amplitude computed via per-time matched-filter (robust to local baseline)
    ref_amps_local = [
        estimate_target_amp_matched(raw[i], freqs_mhz, target_f, sigma_mhz,
                                    core_bw=cfg.science_core_bw, side_bw=max(cfg.protect_bw, 0.6),
                                    poly_order=cfg.poly_order, clip_k=cfg.clip_k)
        for i in off_idxs_local
    ] if off_idxs_local.size else [0.0]
    ref_target_amp = float(np.median(ref_amps_local) + 1e-12)

    target_off = float(np.mean([matched_amp_against_off_local(cln2[i]) for i in off_idxs_local])) if off_idxs_local.size else 0.0
    target_sat = float(np.mean([matched_amp_against_off_local(cln2[i]) for i in sat_idxs_local])) if sat_idxs_local.size else 0.0

    pres_off = float(target_off / ref_target_amp) if ref_target_amp > 0 else float('nan')
    pres_sat = float(target_sat / ref_target_amp) if ref_target_amp > 0 else float('nan')

    # Comb residual: median abs residual at comb peaks in sat window (relative to OFF mean)
    comb_resid = 0.0
    comb_peaks_dict = {}
    if comb_peaks:
        for f0 in comb_peaks:
            j = int(np.argmin(np.abs(freqs_mhz - f0)))
            res = sat_cln[:, j] - off_mu_full[j]
            comb_peaks_dict[float(f0)] = float(np.median(np.abs(res)))
        comb_resid = float(np.median(list(comb_peaks_dict.values()))) if comb_peaks_dict else 0.0

    trough_min = float(np.min(sat_cln - off_mu_full[None, :]))

    metrics = SubtractMetrics(
        reduction_db=reduction_db,
        off_rms_ratio_all=off_rms_ratio_all,
        off_rms_ratio_protect=off_rms_ratio_protect,
        off_rms_ratio_shoulder=off_rms_ratio_shoulder,
        pres_off=pres_off,
        pres_sat=pres_sat,
        comb_resid=comb_resid,
        comb_peaks=comb_peaks_dict,
        trough_min=trough_min,
        clamp_active_ratio=float(np.mean(clamp_active.astype(float))),
        notch_override_energy=0.0,
        s_ratio_01_stage1=float(rms_raw_excess / (rms_cln_excess + 1e-12)),
        s_ratio_01_stage2=1.0,
        gate_mean_off=float(np.mean(w_time[off_idx])) if off_idx.size else 0.0,
        gate_mean_sat=float(np.mean(w_time[sat_idx])) if sat_idx.size else 0.0,
    )

    return cln2, metrics


def run_two_stage_subtraction(*args, **kwargs):
    """Compatibility wrapper accepting legacy keyword names.

    Accepts either:
      - run_two_stage_subtraction(raw, freqs_mhz, cfg, verbose=False)
      - run_two_stage_subtraction(raw=..., freqs=..., cfg=..., target_freq=..., comb_peaks=...)
    This maps `freqs` -> freqs_mhz and any provided target/comb args onto `cfg` attributes before calling the implementation.
    """
    # Positional new-style: (raw, freqs_mhz, cfg[, verbose])
    if len(args) >= 3 and isinstance(args[2], SubtractConfig):
        raw = args[0]
        freqs = args[1]
        cfg = args[2]
        verbose = args[3] if len(args) > 3 else kwargs.get('verbose', False)
        return _run_two_stage_subtraction(raw, freqs, cfg, verbose=verbose)

    # Keyword/legacy handling
    raw = kwargs.get('raw', args[0] if len(args) > 0 else None)
    freqs = kwargs.get('freqs', kwargs.get('freqs_mhz', args[1] if len(args) > 1 else None))
    cfg = kwargs.get('cfg', None)
    verbose = kwargs.get('verbose', False)

    if cfg is None:
        raise TypeError('cfg (SubtractConfig) is required')

    # Map legacy scalar kwargs into cfg attributes if present
    if 'target_freq' in kwargs:
        setattr(cfg, 'target_f_mhz', kwargs['target_freq'])
    if 'target_sigma' in kwargs:
        setattr(cfg, 'target_sigma_mhz', kwargs['target_sigma'])
    if 'comb_peaks' in kwargs:
        setattr(cfg, 'comb_peaks_mhz', kwargs['comb_peaks'])

    return _run_two_stage_subtraction(raw, freqs, cfg, verbose=verbose)
def _compute_metrics(
    raw, cln2, freqs, cfg, sat_mask_time, off_mask_time, sh_mask_time,
    protect, core, comb_mask, comb_peaks, target_freq, target_sigma,
    w_time, clamp_ratios, notchE1, notchE2, s_ratio_01_1, s_ratio_01_2
) -> SubtractMetrics:
    """Compute all quality metrics."""
    # Reduction
    raw_sat = raw[sat_mask_time][:, ~protect]
    cln_sat = cln2[sat_mask_time][:, ~protect]
    rms_raw = np.std(raw_sat)
    rms_cln = np.std(cln_sat) + 1e-12
    reduction_db = float(20.0 * np.log10((rms_raw + 1e-12) / rms_cln))

    # Off-window stability
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

    # Science preservation: use OFF-derived baseline for matched-filter amplitudes
    off_idxs = np.where(off_mask_time)[0]
    sat_idxs = np.where(sat_mask_time)[0]

    if off_idxs.size:
        off_mu_full = np.mean(raw[off_idxs], axis=0).astype(float)
    else:
        off_mu_full = np.mean(raw, axis=0).astype(float)

    core_mask = make_band_mask(freqs, target_freq, cfg.science_core_bw)
    x_core = freqs[core_mask].astype(float)
    g = gaussian(x_core, target_freq, target_sigma).astype(float)
    gg = float(np.dot(g, g) + 1e-12)

    def matched_amp_against_off(spec_row: np.ndarray) -> float:
        y = spec_row[core_mask].astype(float) - off_mu_full[core_mask]
        amp = float(np.dot(g, y) / gg)
        return float(max(0.0, amp))

    # Reference amplitude from OFF (median)
    ref_amps = [matched_amp_against_off(raw[i]) for i in off_idxs] if off_idxs.size else [0.0]
    ref_target_amp = float(np.median(ref_amps) + 1e-12)

    target_off = float(np.mean([matched_amp_against_off(cln2[i]) for i in off_idxs])) if off_idxs.size else 0.0
    target_sat = float(np.mean([matched_amp_against_off(cln2[i]) for i in sat_idxs])) if sat_idxs.size else 0.0

    pres_off = float(target_off / ref_target_amp) if ref_target_amp > 0 else float('nan')
    pres_sat = float(target_sat / ref_target_amp) if ref_target_amp > 0 else float('nan')

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

    return SubtractMetrics(
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