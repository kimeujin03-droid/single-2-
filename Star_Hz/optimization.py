"""
Grid search optimization for hyperparameter tuning.
"""
import itertools
from typing import List, Tuple, Dict
from dataclasses import asdict
import numpy as np
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from evaluation import score_config_balanced


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
    """
    Grid search over hyperparameter space.
    
    Returns:
        best: Dictionary with best configuration
        top: List of top_k configurations
    """
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
            print(f"[{total}] score={sc:.2f} red={m.reduction_db:.2f}dB "
                  f"presOFF={m.pres_off*100:.1f}% presSAT={m.pres_sat*100:.1f}% "
                  f"comb={m.comb_resid:.2f}")

    results_sorted = sorted(results, key=lambda d: d["score"], reverse=True)
    top = results_sorted[:top_k]
    best = top[0] if top else {}
    
    return best, top

