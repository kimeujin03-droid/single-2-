from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BackgroundContext:
    vis_tf: np.ndarray
    freqs_hz: np.ndarray
    times_jd: np.ndarray
    baseline_enu_m: np.ndarray
    weights_tf: np.ndarray
    ant1_enu_m: np.ndarray
    ant2_enu_m: np.ndarray
    source_path: str
    bg_id: str
    product: str = "unknown"
    processing_history: str = "unspecified"
    flag_fraction: float = 0.0
    metadata: Dict[str, Any] | None = None
    baseline_override: bool = False
    baseline_caveat: str = "native background baseline"


def load_background_npz(path: str | Path, bg_id: Optional[str] = None, product: str = "unknown", processing_history: str = "unspecified") -> BackgroundContext:
    """Load a background visibility NPZ.

    Required NPZ fields:
      - vis_tf: complex, shape (Ntime, Nfreq)
      - freqs_hz: float, shape (Nfreq,)
      - times_jd: float, shape (Ntime,)
      - baseline_enu_m: float, shape (3,), ant2 - ant1 in ENU meters

    Optional fields:
      - weights_tf: float, same shape as vis_tf, 0..1
      - flags_tf: bool, same shape as vis_tf; converted to weights = ~flags
      - ant1_enu_m, ant2_enu_m: float, shape (3,)
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    required = ["vis_tf", "freqs_hz", "times_jd", "baseline_enu_m"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Background NPZ missing required fields: {missing}: {path}")

    vis_tf = np.asarray(data["vis_tf"], dtype=complex)
    freqs_hz = np.asarray(data["freqs_hz"], dtype=float)
    times_jd = np.asarray(data["times_jd"], dtype=float)
    baseline_enu_m = np.asarray(data["baseline_enu_m"], dtype=float).reshape(3)

    if vis_tf.shape != (len(times_jd), len(freqs_hz)):
        raise ValueError(f"vis_tf shape {vis_tf.shape} does not match times/freqs {(len(times_jd), len(freqs_hz))}")

    if "weights_tf" in data:
        weights_tf = np.asarray(data["weights_tf"], dtype=float)
    elif "flags_tf" in data:
        weights_tf = (~np.asarray(data["flags_tf"], dtype=bool)).astype(float)
    else:
        weights_tf = np.ones_like(np.abs(vis_tf), dtype=float)
    if weights_tf.shape != vis_tf.shape:
        raise ValueError("weights_tf/flags_tf shape does not match vis_tf")
    weights_tf = np.clip(weights_tf, 0.0, 1.0)

    # Treat nonfinite data as zero weight, but keep the array finite for FFT.
    finite = np.isfinite(vis_tf)
    weights_tf = weights_tf * finite.astype(float)
    vis_tf = np.where(finite, vis_tf, 0.0 + 0.0j)
    flag_fraction = float(np.mean(weights_tf <= 0.0))

    if "ant1_enu_m" in data and "ant2_enu_m" in data:
        ant1 = np.asarray(data["ant1_enu_m"], dtype=float).reshape(3)
        ant2 = np.asarray(data["ant2_enu_m"], dtype=float).reshape(3)
    else:
        ant1 = -0.5 * baseline_enu_m
        ant2 = +0.5 * baseline_enu_m

    md: Dict[str, Any] = {}
    for k in data.files:
        if k not in {"vis_tf", "freqs_hz", "times_jd", "baseline_enu_m", "weights_tf", "flags_tf", "ant1_enu_m", "ant2_enu_m"}:
            try:
                md[k] = data[k].tolist()
            except Exception:
                md[k] = str(data[k])

    return BackgroundContext(
        vis_tf=vis_tf,
        freqs_hz=freqs_hz,
        times_jd=times_jd,
        baseline_enu_m=baseline_enu_m,
        weights_tf=weights_tf,
        ant1_enu_m=ant1,
        ant2_enu_m=ant2,
        source_path=str(path),
        bg_id=bg_id or path.stem,
        product=product,
        processing_history=processing_history,
        flag_fraction=flag_fraction,
        metadata=md,
        baseline_override=False,
        baseline_caveat="native baseline from background NPZ",
    )


def with_baseline(ctx: BackgroundContext, baseline_enu_m: np.ndarray, group_id: str, baseline_id: str | None = None) -> BackgroundContext:
    """Return a shallow copy with an overridden baseline vector.

    WARNING: this is a synthetic geometry sweep. The visibility array remains the
    original background visibility from ctx.source_path, while baseline_enu_m is
    changed for satellite geometry, horizon/window definitions, and reporting. This
    is useful for controlled geometry stress tests, but it is NOT independent
    per-baseline HERA data and MUST NOT be described as such in the paper.
    """
    baseline_enu_m = np.asarray(baseline_enu_m, dtype=float).reshape(3)
    caveat = (
        "synthetic baseline-vector override: original background visibility is reused; "
        "only satellite geometry and delay-window/horizon calculations use the overridden baseline"
    )
    return BackgroundContext(
        vis_tf=ctx.vis_tf,
        freqs_hz=ctx.freqs_hz,
        times_jd=ctx.times_jd,
        baseline_enu_m=baseline_enu_m,
        weights_tf=ctx.weights_tf,
        ant1_enu_m=-0.5 * baseline_enu_m,
        ant2_enu_m=+0.5 * baseline_enu_m,
        source_path=ctx.source_path,
        bg_id=ctx.bg_id,
        product=ctx.product,
        processing_history=ctx.processing_history,
        flag_fraction=ctx.flag_fraction,
        metadata={**(ctx.metadata or {}), "baseline_group_id": group_id, "baseline_id": baseline_id or group_id, "baseline_override": True, "baseline_caveat": caveat, "native_baseline_enu_m": ctx.baseline_enu_m.tolist()},
        baseline_override=True,
        baseline_caveat=caveat,
    )
