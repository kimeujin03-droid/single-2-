"""
Synthetic radio astronomy data generator with satellite interference.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class SimParams:
    """Simulation parameters for synthetic data generation."""
    freq_min_mhz: float = 54.0
    freq_max_mhz: float = 66.0
    df_mhz: float = 0.05
    dt_s: float = 1.0
    duration_s: float = 60.0

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

    # --- New (v2) satellite realism controls ---
    # Number of independent satellite events (each has its own time-envelope).
    satellite_n_events: int = 1
    # Time separation between events (seconds). If >0 and satellite_n_events>1,
    # additional events are placed at mu + m*satellite_event_separation_s.
    satellite_event_separation_s: float = 0.0
    # Optional per-event peak scaling (multiplicative). If provided, length should be satellite_n_events.
    # (Kept out of dataclass for simplicity; see generate_synthetic_cube)
    #
    # Comb drift: peak centers shift linearly with time relative to the event center.
    # Units: MHz per second. 0.0 disables drift (recovering the original outer-product model).
    comb_drift_mhz_per_s: float = 0.0
    # Broadband smooth component can mimic bandpass/foreground-like structure, increasing subspace overlap.
    broadband_floor: float = 0.2
    broadband_slope: float = 0.0  # linear slope across band (dimensionless, applied to [0,1] normalized freq)
    ripple_amp: float = 0.0       # sinusoidal ripple amplitude (dimensionless)
    ripple_period_mhz: float = 2.0 # ripple period in MHz


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian profile."""
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12))**2)


def make_time_freq_axes(p: SimParams) -> Tuple[np.ndarray, np.ndarray]:
    """Create time and frequency axes."""
    n_time = int(round(p.duration_s / p.dt_s))
    n_freq = int(round((p.freq_max_mhz - p.freq_min_mhz) / p.df_mhz))
    freqs = np.linspace(p.freq_min_mhz, p.freq_max_mhz, n_freq, endpoint=False)
    times = np.arange(n_time, dtype=float) * p.dt_s
    return times, freqs


def generate_synthetic_cube(p: SimParams, seed: int = 2025) -> Dict[str, np.ndarray]:
    """Generate synthetic dynamic spectrum with background, science, and satellite."""
    rng = np.random.default_rng(seed)
    times, freqs = make_time_freq_axes(p)
    n_time, n_freq = len(times), len(freqs)

    # Background noise
    bg = rng.normal(p.flux_bg_mean_jy, p.flux_bg_sigma_jy, 
                    size=(n_time, n_freq)).astype(np.float32)

    # Science signal (narrow spectral line)
    sci_prof = gaussian(freqs, p.science_freq_mhz, p.science_sigma_mhz).astype(np.float32)
    sci = (p.science_flux_jy * sci_prof)[None, :].repeat(n_time, axis=0).astype(np.float32)

    # Satellite signal (v2): sum of one or more events, with optional comb drift and smooth bandpass-like structure.
    # This is designed to (optionally) break the pure outer-product form to create mixed singular directions.
    sat = np.zeros((n_time, n_freq), dtype=np.float32)

    # Normalized frequency coordinate in [0,1] for smooth shaping
    f01 = ((freqs - float(freqs.min())) / (float(freqs.max() - freqs.min()) + 1e-12)).astype(np.float32)

    # Smooth broadband component (floor + linear slope + optional ripple)
    broadband = (p.broadband_floor * np.ones_like(freqs, dtype=np.float32))
    if p.broadband_slope != 0.0:
        broadband = broadband * (1.0 + float(p.broadband_slope) * (f01 - 0.5))
    if p.ripple_amp != 0.0:
        broadband = broadband * (1.0 + float(p.ripple_amp) * np.sin(2.0 * np.pi * (freqs - freqs[0]) / float(p.ripple_period_mhz)))

    # Event centers
    n_events = max(int(p.satellite_n_events), 1)
    mu0 = float(p.satellite_time_mu_s)
    sep = float(p.satellite_event_separation_s)

    # Per-event amplitude scalings (optional, deterministic given seed)
    event_scales = 0.85 + 0.30 * rng.random(n_events)  # in [0.85,1.15]

    for m in range(n_events):
        mu_m = mu0 + m * sep
        t_env = gaussian(times, mu_m, p.satellite_time_sigma_s).astype(np.float32)  # (T,)

        # Comb peaks: optionally drifting with time around the event center.
        peaks_tf = np.zeros((n_time, n_freq), dtype=np.float32)
        drift = float(p.comb_drift_mhz_per_s)
        # Centers shift linearly with (t - mu_m)
        dt_rel = (times - mu_m).astype(np.float32)  # (T,)
        for pf0 in p.comb_peak_freqs_mhz:
            center_t = float(pf0) + drift * dt_rel  # (T,)
            # Gaussian over frequency for each time
            peaks_tf += np.exp(-0.5 * ((freqs[None, :] - center_t[:, None]) / (float(p.comb_peak_sigma_mhz) + 1e-12))**2).astype(np.float32)

        # Combine broadband + peaks and normalize per-event to keep peak scale interpretable
        sat_tf = (broadband[None, :] + peaks_tf)
        sat_tf /= float(np.max(sat_tf) + 1e-12)
        sat += (p.flux_sat_peak_jy * event_scales[m]) * (t_env[:, None] * sat_tf)

    sat = sat.astype(np.float32)

    raw = (bg + sci + sat).astype(np.float32)

    return {
        "times": times.astype(np.float32),
        "freqs": freqs.astype(np.float32),
        "background": bg,
        "science": sci,
        "satellite": sat,
        "raw": raw,
    }

