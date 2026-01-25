"""
Evaluation metrics and scoring functions.
"""
import math
import numpy as np
from subtraction_core import SubtractMetrics


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
    """
    Balanced scoring function for configuration quality.
    
    Higher score = better configuration
    """
    off_pen = abs(math.log(m.off_rms_ratio_all + 1e-12))
    offp_pen = abs(math.log(m.off_rms_ratio_protect + 1e-12))
    sh_pen = 0.0 if not np.isfinite(m.off_rms_ratio_shoulder) else abs(
        math.log(m.off_rms_ratio_shoulder + 1e-12)
    )

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