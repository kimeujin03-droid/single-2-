from data_generator import SimParams, generate_synthetic_cube, gaussian
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from signal_processing import estimate_target_amp_matched, make_band_mask
import numpy as np

p = SimParams(df_mhz=0.05, dt_s=1.0)
sim = generate_synthetic_cube(p, seed=2025)
raw = sim['raw']
freqs = sim['freqs']

cfg = SubtractConfig(
    sat_window=(25, 35), edges_margin=12, shoulder=4,
    protect_bw=0.36, science_core_bw=0.12, notch_bw=0.10,
    r1=1, r2=0, poly_order=1, clip_k=2.5,
    subtract_protect=False, safety_floor_q=0.0, gate_taper_s=2.0,
    gate_strength_k=2.0, gate_strength_slope=1.0,
    restore_science=True, restore_max_frac=0.3,
)

cln, m = run_two_stage_subtraction(raw=raw, freqs=freqs, cfg=cfg)

n_time = raw.shape[0]

df = float(np.median(np.diff(freqs))) if freqs.size>1 else 0.0
sigma_mhz = max(1.5 * df, 0.5 * float(cfg.science_core_bw), 1e-6)

mid = int((cfg.sat_window[0] + cfg.sat_window[1])//2)

core_mask = make_band_mask(freqs, p.science_freq_mhz, cfg.science_core_bw)

est_raw_mid = estimate_target_amp_matched(raw[mid], freqs, p.science_freq_mhz, sigma_mhz, core_bw=cfg.science_core_bw, side_bw=cfg.protect_bw, poly_order=cfg.poly_order, clip_k=cfg.clip_k)
est_cln_mid = estimate_target_amp_matched(cln[mid], freqs, p.science_freq_mhz, sigma_mhz, core_bw=cfg.science_core_bw, side_bw=cfg.protect_bw, poly_order=cfg.poly_order, clip_k=cfg.clip_k)

x = freqs[core_mask]
g = gaussian(x, p.science_freq_mhz, sigma_mhz)
y_raw = raw[mid, core_mask]
y_cln = cln[mid, core_mask]

print('mid', mid)
print('est_raw_mid', est_raw_mid)
print('est_cln_mid', est_cln_mid)
print('dot g raw', float(np.dot(g,y_raw)))
print('dot g cln', float(np.dot(g,y_cln)))
print('gg', float(np.dot(g,g)))
print('raw core', y_raw)
print('cln core', y_cln)
