from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from skyfield.api import EarthSatellite, load, wgs84

from .io_background import BackgroundContext

C_M_PER_S = 299_792_458.0


@dataclass
class SatelliteRecord:
    sat: EarthSatellite
    norad_id: str
    epoch: str
    name: str


def _tle_field(line: str, start: int, stop: int) -> str:
    return line[start:stop].strip() if len(line) >= start else ""


def _norad(line1: str) -> str:
    return _tle_field(line1, 2, 7)


def _epoch(line1: str) -> str:
    return _tle_field(line1, 18, 32)


def load_tle_one_epoch_per_norad(path: str | Path, target_jd: Optional[float] = None, max_scan: Optional[int] = None) -> Tuple[List[SatelliteRecord], Dict[str, Any]]:
    """Load TLE records and keep one epoch per physical NORAD ID.

    This is treated as input provenance control, not as a scientific result.
    """
    ts = load.timescale()
    lines = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    recs: List[Dict[str, Any]] = []
    total = 0
    i = 0
    while i < len(lines):
        name = ""
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            line1, line2 = lines[i], lines[i + 1]
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue
        total += 1
        norad = _norad(line1)
        epoch = _epoch(line1)
        sat_name = name or f"SAT-{norad}-E{epoch}"
        sat = EarthSatellite(line1, line2, sat_name, ts)
        fallback_identity = f"NO_NORAD_{epoch}_{sat_name}_{line1[:68].strip()}_{line2[:68].strip()}"
        recs.append({"sat": sat, "norad_id": norad, "epoch": epoch, "name": sat_name, "epoch_jd": float(sat.epoch.tt), "fallback_identity": fallback_identity})
    grouped: Dict[str, Dict[str, Any]] = {}
    for rec in recs:
        key = rec["norad_id"] if rec["norad_id"] else rec["fallback_identity"]
        old = grouped.get(key)
        if old is None:
            grouped[key] = rec
        else:
            if target_jd is None:
                take = rec["epoch_jd"] > old["epoch_jd"]
            else:
                take = abs(rec["epoch_jd"] - target_jd) < abs(old["epoch_jd"] - target_jd)
            if take:
                grouped[key] = rec
    out = [SatelliteRecord(r["sat"], r["norad_id"], r["epoch"], r["name"]) for r in grouped.values()]
    out = sorted(out, key=lambda r: (r.norad_id, r.epoch))
    if max_scan:
        out = out[:int(max_scan)]
    meta = {"tle_records_total": total, "tle_unique_norad_or_fallback_identity": len(grouped), "tle_history_records_collapsed": max(len(recs) - len(grouped), 0), "tle_selection": "nearest_epoch_to_window" if target_jd is not None else "latest_epoch_per_norad"}
    return out, meta


def select_satellite(tle_path: str | Path, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[SatelliteRecord, Dict[str, Any]]:
    site = cfg["site"]
    scfg = cfg.get("starlink", {})
    target_jd = float(np.nanmedian(ctx.times_jd))
    sats, meta = load_tle_one_epoch_per_norad(tle_path, target_jd=target_jd, max_scan=scfg.get("max_scan_satellites", 2000))
    if not sats:
        raise ValueError(f"No TLE records in {tle_path}")
    name = scfg.get("satellite_name")
    if name:
        for rec in sats:
            if name.upper() in rec.name.upper() or name == rec.norad_id:
                return rec, {**meta, "selection": "requested_name_or_norad", "satellite_name": rec.name, "norad_id": rec.norad_id}
        raise ValueError(f"Requested satellite not found: {name}")

    ts = load.timescale()
    t = ts.tt_jd(ctx.times_jd)
    observer = wgs84.latlon(float(site["lat_deg"]), float(site["lon_deg"]), elevation_m=float(site.get("elev_m", 0.0)))
    amin = float(scfg.get("peak_alt_min_deg", 25.0))
    amax = float(scfg.get("peak_alt_max_deg", 85.0))
    rows = []
    for idx, rec in enumerate(sats):
        try:
            alt = (rec.sat - observer).at(t).altaz()[0].degrees
            peak = float(np.nanmax(alt))
            imax = int(np.nanargmax(alt))
            if amin <= peak <= amax:
                rows.append((abs(peak - 0.5 * (amin + amax)), idx, rec, peak, float(ctx.times_jd[imax])))
        except Exception:
            continue
    if not rows:
        rows = []
        for idx, rec in enumerate(sats):
            try:
                alt = (rec.sat - observer).at(t).altaz()[0].degrees
                rows.append((-float(np.nanmax(alt)), idx, rec, float(np.nanmax(alt)), float(ctx.times_jd[int(np.nanargmax(alt))])))
            except Exception:
                continue
        if not rows:
            raise ValueError("No usable satellite track in TLE set.")
    rows.sort(key=lambda x: x[0])
    _, idx, rec, peak, peak_jd = rows[0]
    return rec, {**meta, "selection": "peak_altitude_scan", "satellite_name": rec.name, "norad_id": rec.norad_id, "peak_alt_deg": peak, "peak_jd": peak_jd}


def altaz_to_enu_m(alt_deg: np.ndarray, az_deg: np.ndarray, distance_km: np.ndarray) -> np.ndarray:
    alt = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)
    r_m = distance_km * 1e3
    return np.column_stack([r_m * np.cos(alt) * np.sin(az), r_m * np.cos(alt) * np.cos(az), r_m * np.sin(alt)])


def compute_nearfield_track(sat: EarthSatellite, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    site = cfg["site"]
    ts = load.timescale()
    t = ts.tt_jd(ctx.times_jd)
    observer = wgs84.latlon(float(site["lat_deg"]), float(site["lon_deg"]), elevation_m=float(site.get("elev_m", 0.0)))
    app = (sat - observer).at(t)
    alt, az, dist = app.altaz()
    alt_deg = np.asarray(alt.degrees, dtype=float)
    az_deg = np.asarray(az.degrees, dtype=float)
    range_km = np.asarray(dist.km, dtype=float)
    sat_enu = altaz_to_enu_m(alt_deg, az_deg, range_km)
    r1 = np.linalg.norm(sat_enu - ctx.ant1_enu_m[None, :], axis=1)
    r2 = np.linalg.norm(sat_enu - ctx.ant2_enu_m[None, :], axis=1)
    tau_s = (r2 - r1) / C_M_PER_S
    time_sec = (ctx.times_jd - ctx.times_jd[0]) * 86400.0
    tau_dot = np.gradient(tau_s, time_sec) if len(time_sec) > 1 else np.zeros_like(tau_s)
    range_rate_m_s = np.gradient(range_km * 1e3, time_sec) if len(time_sec) > 1 else np.zeros_like(range_km)
    dnu = float(np.median(np.diff(ctx.freqs_hz))) if len(ctx.freqs_hz) > 1 else 1.0
    dt = float(np.median(np.diff(ctx.times_jd)) * 86400.0) if len(ctx.times_jd) > 1 else float(cfg.get("time_frequency", {}).get("dt_sec", 10.0))
    fringe_rate_hz = tau_dot[:, None] * ctx.freqs_hz[None, :]
    sinc_time = np.sinc(fringe_rate_hz * dt)
    sinc_freq = np.sinc(tau_s[:, None] * dnu)
    attenuation = np.abs(sinc_time * sinc_freq)
    track = pd.DataFrame({"jd": ctx.times_jd, "time_sec": time_sec, "alt_deg": alt_deg, "az_deg": az_deg, "range_km": range_km, "range_rate_m_s": range_rate_m_s, "tau_s": tau_s, "tau_dot_s_per_s": tau_dot})
    arrays = {"tau_s": tau_s, "fringe_rate_hz": fringe_rate_hz, "attenuation_tf": attenuation, "sat_enu_m": sat_enu}
    return track, arrays


def gaussian_beam(track: pd.DataFrame, freqs_hz: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    bcfg = cfg.get("beam", {})
    mode = str(bcfg.get("mode", "gaussian")).lower()
    if mode == "none":
        return np.ones((len(track), len(freqs_hz))), {"mode": "none"}
    fwhm_deg_ref = float(bcfg.get("fwhm_deg_ref", 10.0))
    freq_ref_hz = float(bcfg.get("freq_ref_hz", 150e6))
    za_deg = 90.0 - track["alt_deg"].to_numpy()[:, None]
    fwhm = fwhm_deg_ref * (freq_ref_hz / freqs_hz[None, :])
    sigma = fwhm / np.sqrt(8.0 * np.log(2.0))
    power = np.exp(-0.5 * (za_deg / sigma) ** 2)
    return np.clip(power, 0.0, 1.0), {"mode": "gaussian_power", "fwhm_deg_ref": fwhm_deg_ref, "freq_ref_hz": freq_ref_hz}


def spectral_template(freqs_hz: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    em = cfg.get("starlink", {}).get("emission_model", {})
    csv_path = em.get("spectral_template_csv")
    if csv_path:
        df = pd.read_csv(csv_path)
        f = df["freq_hz"].to_numpy(float) if "freq_hz" in df else df["freq_mhz"].to_numpy(float) * 1e6
        col = "relative_amplitude" if "relative_amplitude" in df else "relative_power" if "relative_power" in df else "flux_jy"
        y = np.clip(df[col].to_numpy(float), 0.0, None)
        spec = np.interp(freqs_hz, f[np.argsort(f)], y[np.argsort(f)], left=0.0, right=0.0)
        spec = spec / max(float(np.nanmax(spec)), 1e-30)
        return spec, {"mode": "csv", "path": str(csv_path), "column": col}
    # Literature-anchored toy morphology: broad windows + narrow/comb features.
    spec = np.zeros_like(freqs_hz, dtype=float)
    def top_hat(center_mhz, width_mhz, amp):
        edge = 0.4e6
        x = np.abs(freqs_hz - center_mhz * 1e6)
        return amp / (1.0 + np.exp((x - 0.5 * width_mhz * 1e6) / edge))
    spec += top_hat(120.0, 8.0, float(em.get("bassa_hba_window_flux_jy", 30.0)))
    spec += top_hat(161.0, 8.0, float(em.get("bassa_hba_window_flux_jy", 30.0)))
    for mhz in em.get("narrowband_lines_mhz", [125.0, 135.0, 143.05, 150.0, 175.0]):
        spec += float(em.get("narrowband_peak_flux_jy", 50.0)) * np.exp(-0.5 * ((freqs_hz - float(mhz) * 1e6) / max(float(em.get("line_render_width_hz", 6e4)), 1.0)) ** 2)
    if np.nanmax(spec) <= 0:
        spec += 1.0
    return spec / float(np.nanmax(spec)), {"mode": "literature_parameterized_order_of_magnitude_anchor", "not_proprietary_waveform": True}


def build_starlink_visibility(ctx: BackgroundContext, cfg: Dict[str, Any], s_ref_jy: Optional[float] = None) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    scfg = cfg.get("starlink", {})
    rec, sel_meta = select_satellite(scfg["tle_path"], ctx, cfg)
    track, geom = compute_nearfield_track(rec.sat, ctx, cfg)
    beam_tf, beam_meta = gaussian_beam(track, ctx.freqs_hz, cfg)
    spec, spec_meta = spectral_template(ctx.freqs_hz, cfg)
    s_ref = float(s_ref_jy if s_ref_jy is not None else scfg.get("reference_flux_jy", 100.0))
    r_ref = float(scfg.get("reference_range_km", 550.0))
    mode = scfg.get("range_attenuation_mode", "flux_density_1_over_r2")
    r = np.clip(track["range_km"].to_numpy(float), 1e-6, None)
    if mode == "none_observed_apparent_flux":
        range_att = np.ones_like(r)
    elif mode == "field_amplitude_1_over_r":
        range_att = r_ref / r
    else:
        range_att = (r_ref / r) ** 2
    phase = np.exp(-2j * np.pi * geom["tau_s"][:, None] * ctx.freqs_hz[None, :])
    amp = s_ref * range_att[:, None] * geom["attenuation_tf"] * beam_tf * spec[None, :]
    vis = amp * phase
    report = {"selected_satellite": sel_meta, "reference_flux_jy": s_ref, "reference_range_km": r_ref, "range_attenuation_mode": mode, "beam": beam_meta, "spectral_template": spec_meta, "peak_abs_jy": float(np.nanmax(np.abs(vis))), "mean_abs_jy": float(np.nanmean(np.abs(vis))), "tau_min_ns": float(np.nanmin(geom["tau_s"]) * 1e9), "tau_max_ns": float(np.nanmax(geom["tau_s"]) * 1e9)}
    return vis.astype(complex), track, report
