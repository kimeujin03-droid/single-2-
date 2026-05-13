from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import skew

from .io_background import BackgroundContext
from .metrics import compute_window_scores
from .pipeline import pipeline


def phase_randomized_window_null(ctx: BackgroundContext, cfg: Dict[str, Any], sat_vis: np.ndarray, n_trials: int = 500, seed: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Null distribution for S_win_abs_delta_power_db.

    Preserves the injected amplitude envelope and time-frequency support. The default
    mode is per_time, which applies one random phase per integration while preserving
    the spectral coherence of the injected template. per_pixel is available only as an
    aggressive sensitivity test because it destroys both temporal and spectral coherence.
    This is distinct from a multi-emitter coherence-amplification null.
    """
    rng = np.random.default_rng(seed)
    null_cfg = cfg.get("metrics", {}).get("phase_randomized_null", {})
    mode = null_cfg.get("mode", "per_time")
    proc_bg, _ = pipeline(ctx.vis_tf, ctx, cfg)
    scores: List[float] = []
    for _ in range(int(n_trials)):
        if mode == "global":
            phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
        elif mode == "per_time":
            phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(sat_vis.shape[0], 1)))
        elif mode == "per_freq":
            phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(1, sat_vis.shape[1])))
        elif mode == "per_pixel":
            phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=sat_vis.shape))
        else:
            raise ValueError(f"Unsupported phase_randomized_null.mode: {mode}. Use global, per_time, per_freq, or per_pixel.")
        proc_dirty, _ = pipeline(ctx.vis_tf + sat_vis * phase, ctx, cfg)
        scores.append(compute_window_scores(proc_bg, proc_dirty, ctx, cfg)["S_win_abs_delta_power_db"])
    arr = np.asarray(scores, dtype=float)
    diag = {
        "n_trials": int(n_trials),
        "seed": int(seed),
        "mode": mode,
        "mode_interpretation": {
            "global": "one random phase for the whole injected visibility",
            "per_time": "one random phase per integration; main-claim default",
            "per_freq": "one random phase per channel",
            "per_pixel": "independent phase per time-frequency pixel; aggressive sensitivity test only",
        }.get(mode, "unknown"),
        "null_median": float(np.nanpercentile(arr, 50)),
        "null_p95": float(np.nanpercentile(arr, 95)),
        "null_p99": float(np.nanpercentile(arr, 99)),
        "null_std": float(np.nanstd(arr)),
        "null_iqr": float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)),
        "null_skewness": float(skew(arr, nan_policy="omit")) if len(arr) > 2 else float("nan"),
    }
    return arr, diag


def summarize_null(obs_score_db: float, null_scores_db: np.ndarray, diag: Dict[str, Any]) -> Dict[str, float]:
    p95 = float(np.nanpercentile(null_scores_db, 95))
    p99 = float(np.nanpercentile(null_scores_db, 99))
    return {
        **diag,
        "obs_S_win_abs_delta_power_db": float(obs_score_db),
        "obs_minus_null_p95_db": float(obs_score_db - p95),
        "obs_minus_null_p99_db": float(obs_score_db - p99),
        "obs_percentile_in_null": float(100.0 * np.mean(null_scores_db <= obs_score_db)),
        "null_exceeds_p95": bool(obs_score_db > p95),
    }
