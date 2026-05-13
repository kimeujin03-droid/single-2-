from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .io_background import BackgroundContext
from .pipeline import C_M_PER_S, delay_axis_s, make_taper, pipeline, weighted_delay_transform


def window_mask(ctx: BackgroundContext, cfg: Dict[str, Any]) -> np.ndarray:
    dc = cfg.get("metrics", {}).get("window", {})
    buffer_ns = float(dc.get("buffer_ns", cfg.get("pipeline", {}).get("delay_filter", {}).get("buffer_ns", 100.0)))
    delays = delay_axis_s(ctx.freqs_hz)
    horizon_ns = np.linalg.norm(ctx.baseline_enu_m) / C_M_PER_S * 1e9
    return np.abs(delays) >= (horizon_ns + buffer_ns) * 1e-9


def _metric_taper(ctx: BackgroundContext, cfg: Dict[str, Any]) -> np.ndarray:
    mc = cfg.get("metrics", {}).get("window", {})
    return make_taper(len(ctx.freqs_hz), mc.get("taper", cfg.get("pipeline", {}).get("delay_filter", {}).get("taper", "blackman_harris")), mc)


def compute_window_scores(proc_bg: np.ndarray, proc_dirty: np.ndarray, ctx: BackgroundContext, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Compute EoR-window residual-risk diagnostics.

    Main paper-facing metric:
      S_win_abs_delta_power_db = 10 log10( sum | |V_dirty~|^2 - |V_bg~|^2 | / sum |V_bg~|^2 )

    Secondary metric:
      R_win_int_power_db = 10 log10( sum |V_int~|^2 / sum |V_bg~|^2 )
    """
    taper = _metric_taper(ctx, cfg)
    mask = window_mask(ctx, cfg)
    bg_dly = weighted_delay_transform(proc_bg, ctx.weights_tf, taper)[:, mask]
    dirty_dly = weighted_delay_transform(proc_dirty, ctx.weights_tf, taper)[:, mask]
    int_dly = weighted_delay_transform(proc_dirty - proc_bg, ctx.weights_tf, taper)[:, mask]

    bg_power = np.abs(bg_dly) ** 2
    dirty_power = np.abs(dirty_dly) ** 2
    int_power = np.abs(int_dly) ** 2
    delta_power = dirty_power - bg_power
    eps = 1e-30

    bg_sum = float(np.nansum(bg_power))
    abs_delta_sum = float(np.nansum(np.abs(delta_power)))
    signed_delta_sum = float(np.nansum(delta_power))
    int_sum = float(np.nansum(int_power))

    return {
        "S_win_abs_delta_power_db": float(10.0 * np.log10(abs_delta_sum / max(bg_sum, eps))),
        "S_win_signed_abs_sum_db": float(10.0 * np.log10(abs(signed_delta_sum) / max(bg_sum, eps))),
        "R_win_int_power_db": float(10.0 * np.log10(int_sum / max(bg_sum, eps))),
        "window_bg_power_sum": bg_sum,
        "window_abs_delta_power_sum": abs_delta_sum,
        "window_signed_delta_power_sum": signed_delta_sum,
        "window_interaction_power_sum": int_sum,
        "window_delay_bins": int(np.sum(mask)),
    }


def run_observed_scores(ctx: BackgroundContext, cfg: Dict[str, Any], sat_vis: np.ndarray) -> Dict[str, Any]:
    proc_bg, pinfo_bg = pipeline(ctx.vis_tf, ctx, cfg)
    proc_dirty, pinfo_dirty = pipeline(ctx.vis_tf + sat_vis, ctx, cfg)
    scores = compute_window_scores(proc_bg, proc_dirty, ctx, cfg)
    return {**scores, "pipeline_bg": pinfo_bg, "pipeline_dirty": pinfo_dirty}
