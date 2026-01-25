from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
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
print('metrics:', m)
print('pres_off', m.pres_off, 'pres_sat', m.pres_sat)

# Extra diagnostics: show matched amps computed in _compute_metrics logic
from subtraction_core import make_band_mask
core_mask = make_band_mask(freqs, p.science_freq_mhz, cfg.science_core_bw)
time_idx = np.arange(raw.shape[0])
off_idxs = np.where(~((time_idx >= cfg.sat_window[0]) & (time_idx < cfg.sat_window[1])))[0]
sat_idxs = np.arange(cfg.sat_window[0], cfg.sat_window[1])
off_mu_full = np.mean(raw[off_idxs], axis=0) if off_idxs.size else np.mean(raw, axis=0)
df = float(np.median(np.diff(freqs))) if freqs.size>1 else 0.0
sigma_mhz = max(1.5 * df, 0.5 * float(cfg.science_core_bw), 1e-6)
from data_generator import gaussian as _gauss
g = _gauss(freqs[core_mask], p.science_freq_mhz, sigma_mhz)
gg = float(np.dot(g, g) + 1e-12)
amps_sat = []
for ti in sat_idxs:
    y = cln[ti, core_mask].astype(float) - off_mu_full[core_mask]
    amp = float(np.dot(g,y)/gg)
    amps_sat.append(amp)
print('sat amps (direct off-baseline matched):', amps_sat)
# compute ref and target means used in metrics
if off_idxs.size:
    arr = []
    for i in off_idxs:
        val = float(np.dot(g, (raw[i, core_mask].astype(float) - off_mu_full[core_mask])) / gg)
        arr.append(val)
    off_mean = float(np.mean(arr))
else:
    off_mean = 0.0
print('ref_target_amp (mean off matched against off-baseline):', off_mean)
print('mean sat amp (direct):', float(np.mean(amps_sat)))
