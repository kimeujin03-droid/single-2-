from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal.windows import blackmanharris, dpss

from .io_background import BackgroundContext

C_M_PER_S = 299_792_458.0


def make_taper(n: int, method: str = "blackman_harris", cfg: Dict[str, Any] | None = None) -> np.ndarray:
    cfg = cfg or {}
    method = method.lower()
    if method in {"blackman_harris", "blackmanharris", "bh"}:
        return blackmanharris(n, sym=False)
    if method == "hann":
        return np.hanning(n)
    if method in {"rect", "rectangular", "none"}:
        return np.ones(n)
    if method == "dpss":
        nw = float(cfg.get("nw", 2.5))
        return np.asarray(dpss(n, NW=nw, Kmax=1, sym=False)[0], dtype=float)
    raise ValueError(f"Unsupported taper method: {method}")


def delay_axis_s(freqs_hz: np.ndarray) -> np.ndarray:
    dnu = float(np.median(np.diff(freqs_hz))) if len(freqs_hz) > 1 else 1.0
    return np.fft.fftshift(np.fft.fftfreq(len(freqs_hz), d=dnu))


def weighted_delay_transform(vis_tf: np.ndarray, weights_tf: np.ndarray, taper: np.ndarray) -> np.ndarray:
    w = np.clip(weights_tf, 0.0, 1.0)
    x = np.where(np.isfinite(vis_tf), vis_tf, 0.0 + 0.0j) * w * taper[None, :]
    return np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)


def inverse_delay_transform(delay_tf: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.fft.ifftshift(delay_tf, axes=1), axis=1)


def apply_delay_highpass(vis_tf: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    dc = cfg.get("pipeline", {}).get("delay_filter", {})
    if not bool(dc.get("enabled", True)):
        return vis_tf.copy(), {"enabled": False}
    taper = make_taper(len(ctx.freqs_hz), dc.get("taper", "blackman_harris"), dc)
    dly = weighted_delay_transform(vis_tf, ctx.weights_tf, taper)
    delays = delay_axis_s(ctx.freqs_hz)
    horizon_ns = np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9
    buffer_ns = float(dc.get("buffer_ns", 100.0))
    keep = np.abs(delays) >= (horizon_ns + buffer_ns) * 1e-9
    out = inverse_delay_transform(dly * keep[None, :])
    return out, {"enabled": True, "taper": dc.get("taper", "blackman_harris"), "horizon_ns": float(horizon_ns), "buffer_ns": buffer_ns, "kept_delay_fraction": float(np.mean(keep))}


def apply_fringe_filter(vis_tf: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    pcfg = cfg.get("pipeline", {})
    nt = len(ctx.times_jd)
    if nt <= 1:
        return vis_tf.copy(), {"enabled": False, "reason": "single time sample"}
    dt = float(np.median(np.diff(ctx.times_jd)) * 86400.0)
    fr_hz = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    taper_t = np.hanning(nt) if nt > 2 else np.ones(nt)
    w_t = np.nanmean(np.clip(ctx.weights_tf, 0.0, 1.0), axis=1)
    fr_tf = np.fft.fftshift(np.fft.fft(vis_tf * (w_t * taper_t)[:, None], axis=0), axes=0)
    info: Dict[str, Any] = {}

    notch = pcfg.get("fr_zero_notch", {})
    if bool(notch.get("enabled", False)):
        width_mhz = float(notch.get("width_mhz", 0.03))
        keep = np.abs(fr_hz * 1e3) > width_mhz
        fr_tf = fr_tf * keep[:, None]
        info["fr_zero_notch"] = {"enabled": True, "width_mhz": width_mhz, "kept_fraction": float(np.mean(keep))}
    else:
        info["fr_zero_notch"] = {"enabled": False}

    main = pcfg.get("mainlobe_fr_filter", {})
    if bool(main.get("enabled", False)):
        width_mhz = float(main.get("width_mhz", 0.8))
        mode = main.get("mode", "remove_mainlobe")
        if mode == "remove_mainlobe":
            keep = np.abs(fr_hz * 1e3) > width_mhz
        elif mode == "keep_mainlobe":
            keep = np.abs(fr_hz * 1e3) <= width_mhz
        else:
            raise ValueError(f"Unsupported fringe-rate filter mode: {mode}")
        fr_tf = fr_tf * keep[:, None]
        info["mainlobe_fr_filter"] = {"enabled": True, "mode": mode, "width_mhz": width_mhz, "kept_fraction": float(np.mean(keep))}
    else:
        info["mainlobe_fr_filter"] = {"enabled": False}

    return np.fft.ifft(np.fft.ifftshift(fr_tf, axes=0), axis=0), info


def simple_spectral_inpainting(vis_tf: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simple per-time smooth-spectrum surrogate for flagged channels.

    This is intentionally modest. It is NOT a HERA production inpainting implementation.
    It fits weighted low-order polynomials to real/imag parts across frequency and fills
    only zero-weight samples before downstream delay/fringe operations.
    """
    icfg = cfg.get("pipeline", {}).get("inpainting_surrogate", {})
    if not bool(icfg.get("enabled", False)):
        return vis_tf.copy(), {"enabled": False}
    order = int(icfg.get("poly_order", 3))
    min_good = int(icfg.get("min_good_channels", max(12, order + 3)))
    x = (ctx.freqs_hz - np.nanmean(ctx.freqs_hz)) / max(np.nanptp(ctx.freqs_hz), 1.0)
    out = vis_tf.copy()
    filled = 0
    total_bad = int(np.sum(ctx.weights_tf <= 0))
    for it in range(vis_tf.shape[0]):
        good = ctx.weights_tf[it] > 0
        bad = ~good
        if np.sum(good) < min_good or not np.any(bad):
            continue
        wr = ctx.weights_tf[it, good]
        for part in [0, 1]:
            y = np.real(vis_tf[it, good]) if part == 0 else np.imag(vis_tf[it, good])
            coeff = np.polyfit(x[good], y, deg=min(order, np.sum(good) - 1), w=np.sqrt(wr))
            pred = np.polyval(coeff, x[bad])
            if part == 0:
                out[it, bad] = pred + 1j * np.imag(out[it, bad])
            else:
                out[it, bad] = np.real(out[it, bad]) + 1j * pred
        filled += int(np.sum(bad))
    return out, {"enabled": True, "method": "weighted_poly_fill_flagged_channels", "poly_order": order, "filled_samples": filled, "total_zero_weight_samples": total_bad}


def pipeline(vis_tf: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x, inp_info = simple_spectral_inpainting(vis_tf, ctx, cfg)
    x, dly_info = apply_delay_highpass(x, ctx, cfg)
    x, fr_info = apply_fringe_filter(x, ctx, cfg)
    return x, {
        "background_product": ctx.product,
        "background_processing_history": ctx.processing_history,
        "operator_scope": "post-processing operator applied on top of selected background product; not raw HERA production pipeline",
        "inpainting_surrogate": inp_info,
        "delay_filter": dly_info,
        "fringe_filter": fr_info,
    }
