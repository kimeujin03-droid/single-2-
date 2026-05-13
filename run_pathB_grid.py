from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import read_config
from .io_background import load_background_npz, with_baseline
from .metrics import run_observed_scores
from .nulls import phase_randomized_window_null, summarize_null
from .satellite import build_starlink_visibility


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _baseline_vector(group: Dict[str, Any], native_baseline: np.ndarray | None = None) -> np.ndarray:
    if bool(group.get("use_native_baseline", False)):
        if native_baseline is None:
            raise ValueError("use_native_baseline=true requires a native baseline")
        return np.asarray(native_baseline, dtype=float)
    if "baseline_enu_m" in group:
        return np.asarray(group["baseline_enu_m"], dtype=float)
    length = float(group.get("length_m", 14.6))
    orient = str(group.get("orientation", "EW")).upper()
    if orient == "EW":
        return np.array([length, 0.0, 0.0])
    if orient == "NS":
        return np.array([0.0, length, 0.0])
    raise ValueError(f"Unknown baseline orientation: {orient}")


def load_bg_from_cfg(bg: Dict[str, Any]):
    return load_background_npz(
        bg["path"],
        bg_id=bg.get("id"),
        product=bg.get("product", "unknown"),
        processing_history=bg.get("processing_history", "unspecified"),
    )


def run_case(ctx, cfg: Dict[str, Any], bg_id: str, baseline_group: Dict[str, Any], s_ref: float, n_null: int, seed: int, out_dir: Path, save_null: bool = True) -> Dict[str, Any]:
    group_id = baseline_group.get("id", baseline_group.get("name", "baseline"))
    use_native = bool(baseline_group.get("use_native_baseline", False))
    if use_native:
        ctx_bl = ctx
        # Attach reporting metadata without changing the actual visibility/baseline.
        ctx_bl.metadata = {**(ctx_bl.metadata or {}), "baseline_group_id": group_id, "baseline_id": baseline_group.get("baseline_id", group_id), "baseline_override": False}
    else:
        ctx_bl = with_baseline(ctx, _baseline_vector(baseline_group, ctx.baseline_enu_m), group_id=group_id, baseline_id=baseline_group.get("baseline_id"))
    sat_vis, track, sat_report = build_starlink_visibility(ctx_bl, cfg, s_ref_jy=s_ref)
    obs = run_observed_scores(ctx_bl, cfg, sat_vis)
    null_scores, null_diag = phase_randomized_window_null(ctx_bl, cfg, sat_vis, n_trials=n_null, seed=seed)
    summ = summarize_null(obs["S_win_abs_delta_power_db"], null_scores, null_diag)

    row = {
        "bg_id": bg_id,
        "bg_path": ctx.source_path,
        "product": ctx.product,
        "processing_history": ctx.processing_history,
        "flag_fraction": ctx.flag_fraction,
        "baseline_group": group_id,
        "baseline_id": (ctx_bl.metadata or {}).get("baseline_id", group_id),
        "baseline_override": bool(ctx_bl.baseline_override),
        "baseline_sweep_mode": "synthetic_override" if ctx_bl.baseline_override else "native_background_baseline",
        "baseline_caveat": ctx_bl.baseline_caveat,
        "native_baseline_length_m": float(np.linalg.norm(ctx.baseline_enu_m)),
        "baseline_length_m": float(np.linalg.norm(ctx_bl.baseline_enu_m)),
        "baseline_enu_e_m": float(ctx_bl.baseline_enu_m[0]),
        "baseline_enu_n_m": float(ctx_bl.baseline_enu_m[1]),
        "baseline_enu_u_m": float(ctx_bl.baseline_enu_m[2]),
        "S_ref_jy": float(s_ref),
        "S_win_abs_delta_power_db": obs["S_win_abs_delta_power_db"],
        "R_win_int_power_db": obs["R_win_int_power_db"],
        "S_win_signed_abs_sum_db": obs["S_win_signed_abs_sum_db"],
        "window_bg_power_sum": obs["window_bg_power_sum"],
        "window_abs_delta_power_sum": obs["window_abs_delta_power_sum"],
        "window_interaction_power_sum": obs["window_interaction_power_sum"],
        **summ,
        "sat_peak_abs_jy": sat_report["peak_abs_jy"],
        "sat_mean_abs_jy": sat_report["mean_abs_jy"],
        "sat_tau_min_ns": sat_report["tau_min_ns"],
        "sat_tau_max_ns": sat_report["tau_max_ns"],
    }
    case_id = f"{bg_id}__{group_id}__S{str(s_ref).replace('.','p')}Jy"
    if save_null:
        np.save(out_dir / f"null_{case_id}.npy", null_scores)
        track.to_csv(out_dir / f"track_{case_id}.csv", index=False)
        with open(out_dir / f"report_{case_id}.json", "w", encoding="utf-8") as f:
            json.dump({"row": row, "sat_report": sat_report, "pipeline_bg": obs.get("pipeline_bg"), "pipeline_dirty": obs.get("pipeline_dirty")}, f, ensure_ascii=False, indent=2, default=str)
    return row


def run_grid(config_path: str | Path, out_dir: str | Path, n_null: int | None = None, max_backgrounds: int | None = None, save_null: bool = True) -> Path:
    cfg = read_config(config_path)
    out_dir = ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = []
    n_null = int(n_null or cfg.get("metrics", {}).get("phase_randomized_null", {}).get("n_trials", 100))
    seed0 = int(cfg.get("metrics", {}).get("phase_randomized_null", {}).get("seed", 42))
    backgrounds = cfg["backgrounds"][:max_backgrounds] if max_backgrounds else cfg["backgrounds"]
    fluxes = cfg.get("experiment", {}).get("flux_grid_jy", cfg.get("experiment", {}).get("flux_values_jy", [10, 30, 100, 300, 1000]))
    baselines = cfg.get("baseline_groups", [])
    if not baselines:
        baselines = [{"id": "native", "use_native_baseline": True}]

    t0 = time.time()
    for ib, bg in enumerate(backgrounds):
        ctx = load_bg_from_cfg(bg)
        for ig, group in enumerate(baselines):
            for isf, s_ref in enumerate(fluxes):
                seed = seed0 + ib * 10000 + ig * 1000 + isf
                print(f"[run] bg={ctx.bg_id} group={group.get('id')} S={s_ref}Jy null={n_null}", flush=True)
                rows.append(run_case(ctx, cfg, ctx.bg_id, group, float(s_ref), n_null, seed, out_dir, save_null=save_null))
    df = pd.DataFrame(rows)
    csv_path = out_dir / "pathB_results.csv"
    df.to_csv(csv_path, index=False)
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config_path": str(config_path), "out_dir": str(out_dir), "n_rows": len(df), "elapsed_sec": time.time() - t0, "n_null": n_null}, f, ensure_ascii=False, indent=2)
    return csv_path


def benchmark(config_path: str | Path, out_dir: str | Path, n_null: int = 5) -> Dict[str, Any]:
    cfg = read_config(config_path)
    out_dir = ensure_dir(out_dir)
    bg = cfg["backgrounds"][0]
    group = cfg.get("baseline_groups", [{"id": "native", "use_native_baseline": True}])[0]
    s_ref = float(cfg.get("experiment", {}).get("flux_grid_jy", cfg.get("experiment", {}).get("flux_values_jy", [100]))[0])
    ctx = load_bg_from_cfg(bg)
    t0 = time.time()
    row = run_case(ctx, cfg, ctx.bg_id, group, s_ref, n_null, seed=123, out_dir=out_dir, save_null=False)
    elapsed = time.time() - t0
    per_null = elapsed / max(n_null, 1)
    est_75x500_hr = 75 * 500 * per_null / 3600.0
    info = {"elapsed_sec": elapsed, "n_null": n_null, "approx_sec_per_null_including_overhead": per_null, "estimated_75x500_hours": est_75x500_hr, "example_row": row}
    with open(Path(out_dir) / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    return info
