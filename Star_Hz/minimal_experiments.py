import os
import numpy as np
import matplotlib.pyplot as plt
from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from dataclasses import replace
from signal_processing import make_band_mask, gaussian, estimate_target_amp_matched

outdir = os.path.join(os.path.dirname(__file__), 'outputs', 'minimal')
os.makedirs(outdir, exist_ok=True)

p = SimParams(df_mhz=0.05, dt_s=1.0)
sim = generate_synthetic_cube(p, seed=2025)
raw = sim['raw']
freqs = sim['freqs']

# baseline config (same as your final cfg)
cfg_base = SubtractConfig(
    sat_window=(25, 35), edges_margin=15, shoulder=2.0,
    protect_bw=0.80, science_core_bw=0.15, notch_bw=0.20,
    r1=6, r2=2, poly_order=2, clip_k=5.0,
    subtract_protect=True, safety_floor_q=0.0, gate_taper_s=2.0,
)

labels = []
results = []
cleaned_list = []

def compute_off_indices(n_time, sat_window, edges_margin):
    t0, t1 = sat_window
    off_mask = np.ones(n_time, dtype=bool)
    off_mask[max(0, t0 - edges_margin):min(n_time, t1 + edges_margin)] = False
    if not off_mask.any():
        off_mask = ~(np.arange(n_time) >= t0) & ~(np.arange(n_time) < t1)
    return np.where(off_mask)[0], np.where((np.arange(n_time) >= t0) & (np.arange(n_time) < t1))[0]


def safe_preservation(raw, cln, freqs, cfg, label=None, outdir=None):
    # replicate OFF-derived matched amp logic from core but add eps+delta guards
    n_time = raw.shape[0]
    off_idx, sat_idx = compute_off_indices(n_time, cfg.sat_window, cfg.edges_margin)
    # OFF mean
    if off_idx.size:
        off_mu = np.mean(raw[off_idx], axis=0)
    else:
        off_mu = np.mean(raw, axis=0)

    # matched filter core
    df = float(np.median(np.diff(freqs))) if freqs.size>1 else 0.0
    sigma_mhz = max(1.5*df, 0.5*cfg.science_core_bw, 1e-6)
    core_mask = make_band_mask(freqs, getattr(cfg, 'target_f_mhz', p.science_freq_mhz), cfg.science_core_bw)
    x_core = freqs[core_mask]
    g = gaussian(x_core, getattr(cfg, 'target_f_mhz', p.science_freq_mhz), sigma_mhz)
    gg = float(np.dot(g, g) + 1e-12)

    def matched_against_off(spec_row):
        y = spec_row[core_mask].astype(float) - off_mu[core_mask].astype(float)
        amp = float(np.dot(g, y) / gg)
        return float(max(0.0, amp))

    # reference amplitude using per-time matched estimates over OFF (robust)
    ref_amps = []
    for ti in off_idx:
        a = estimate_target_amp_matched(raw[ti], freqs, getattr(cfg, 'target_f_mhz', p.science_freq_mhz), sigma_mhz,
                                        core_bw=cfg.science_core_bw, side_bw=max(cfg.protect_bw, 0.6), poly_order=cfg.poly_order, clip_k=cfg.clip_k)
        ref_amps.append(a)
    ref_target = float(np.median(ref_amps) + 1e-12)

    target_off = float(np.mean([matched_against_off(cln[i]) for i in off_idx])) if off_idx.size else 0.0
    target_sat = float(np.mean([matched_against_off(cln[i]) for i in sat_idx])) if sat_idx.size else 0.0

    # raw ratio (unclamped)
    eps = max(1e-12, 1e-6 * ref_target)
    pres_off_raw = float(target_off / (ref_target + eps)) if ref_target > 0 else float('nan')
    pres_sat_raw = float(target_sat / (ref_target + eps)) if ref_target > 0 else float('nan')

    # clipped ratio for display (but do not force a fixed cap)
    def clip_pres(x):
        if np.isnan(x):
            return x
        return float(max(0.0, min(100.0, x)))

    pres_off_clip = clip_pres(pres_off_raw)
    pres_sat_clip = clip_pres(pres_sat_raw)

    # diagnostics: raw/clean mean,std,L2 over OFF and SAT
    def stats(arr):
        a = np.asarray(arr, dtype=float)
        return dict(mean=float(np.mean(a)), std=float(np.std(a)), l2=float(np.linalg.norm(a)))

    raw_off_stats = stats(raw[off_idx]) if off_idx.size else {}
    cln_off_stats = stats(cln[off_idx]) if off_idx.size else {}
    raw_sat_stats = stats(raw[sat_idx]) if sat_idx.size else {}
    cln_sat_stats = stats(cln[sat_idx]) if sat_idx.size else {}

    # Save stamped cleaned array for this experiment if requested
    if label and outdir:
        try:
            stamp_path = os.path.join(outdir, f'clean_{label}.npy')
            np.save(stamp_path, cln)
        except Exception:
            stamp_path = None
    else:
        stamp_path = None

    # protect passthrough check (should be enforced by core; double-check here)
    # compute protect mask locally and check passthrough
    protect_mask = make_band_mask(freqs, getattr(cfg, 'target_f_mhz', p.science_freq_mhz), cfg.protect_bw)
    if protect_mask.any():
        ok = np.allclose(cln[:, protect_mask], raw[:, protect_mask])
        if not ok:
            raise AssertionError('Protect-band passthrough violated in safe_preservation')

    # print diagnostics
    print('SAFE-PRES DIAG:', label if label else '')
    print('  ref_target=', ref_target)
    print('  target_off=', target_off, ' target_sat=', target_sat)
    print('  pres_off_raw=', pres_off_raw, ' pres_off_clip=', pres_off_clip)
    print('  pres_sat_raw=', pres_sat_raw, ' pres_sat_clip=', pres_sat_clip)
    print('  raw_off_stats=', raw_off_stats)
    print('  cln_off_stats=', cln_off_stats)
    print('  raw_sat_stats=', raw_sat_stats)
    print('  cln_sat_stats=', cln_sat_stats)
    if stamp_path:
        print('  saved cleaned stamp ->', stamp_path)

    return dict(ref_target=ref_target, target_off=target_off, target_sat=target_sat,
                pres_off_raw=pres_off_raw, pres_sat_raw=pres_sat_raw,
                pres_off_clip=pres_off_clip, pres_sat_clip=pres_sat_clip,
                raw_off_stats=raw_off_stats, cln_off_stats=cln_off_stats,
                raw_sat_stats=raw_sat_stats, cln_sat_stats=cln_sat_stats, stamp_path=stamp_path)

# 1) Experiment: preservation with eps+delta (no change to processing)
label = 'exp1_pres_eps_delta'
print('\nRunning', label)
cln1, m1 = run_two_stage_subtraction(raw=raw, freqs=freqs, target_freq=p.science_freq_mhz, target_sigma=p.science_sigma_mhz, comb_peaks=list(p.comb_peak_freqs_mhz), cfg=cfg_base)
res1 = safe_preservation(raw, cln1, freqs, cfg_base, label=label, outdir=outdir)
print('core metrics:', m1.reduction_db, m1.pres_sat, m1.comb_resid)
print('safe metrics:', res1)
labels.append(label); results.append((m1, res1)); cleaned_list.append(cln1)

# 2) Experiment: poly_order=0 and protect band fully excluded
label = 'exp2_poly0_protect_excluded'
print('\nRunning', label)
# create a safe copy of cfg_base using dataclasses.replace
cfg2 = replace(cfg_base, poly_order=0, protect_bw=max(cfg_base.protect_bw, 2.0), subtract_protect=True)
cln2, m2 = run_two_stage_subtraction(raw=raw, freqs=freqs, target_freq=p.science_freq_mhz, target_sigma=p.science_sigma_mhz, comb_peaks=list(p.comb_peak_freqs_mhz), cfg=cfg2)
res2 = safe_preservation(raw, cln2, freqs, cfg2, label=label, outdir=outdir)
print('core metrics:', m2.reduction_db, m2.pres_sat, m2.comb_resid)
print('safe metrics:', res2)
labels.append(label); results.append((m2, res2)); cleaned_list.append(cln2)

# 3) Experiment: time-axis smoothing (EMA)
label = 'exp3_time_ema'
print('\nRunning', label)
alpha = 0.3
raw_smooth = raw.copy()
for t in range(1, raw.shape[0]):
    raw_smooth[t] = alpha * raw[t] + (1 - alpha) * raw_smooth[t-1]

cln3, m3 = run_two_stage_subtraction(raw=raw_smooth, freqs=freqs, target_freq=p.science_freq_mhz, target_sigma=p.science_sigma_mhz, comb_peaks=list(p.comb_peak_freqs_mhz), cfg=cfg_base)
res3 = safe_preservation(raw_smooth, cln3, freqs, cfg_base, label=label, outdir=outdir)
print('core metrics:', m3.reduction_db, m3.pres_sat, m3.comb_resid)
print('safe metrics:', res3)
labels.append(label); results.append((m3, res3)); cleaned_list.append(cln3)

# Combined spectra figure and save only spectra.png
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
for i, lab in enumerate(labels):
    m_core, safe = results[i]
    cln = cleaned_list[i]
    # compute mean spectra
    mean_raw = np.mean(raw, axis=0)
    off_idx, sat_idx = compute_off_indices(raw.shape[0], cfg_base.sat_window, cfg_base.edges_margin)
    off_mu = np.mean(raw[off_idx], axis=0)
    mean_cln = np.mean(cln, axis=0)
    axes[i].plot(freqs, mean_raw, label='raw mean', color='0.6')
    axes[i].plot(freqs, off_mu, label='off mu', color='C1')
    axes[i].plot(freqs, mean_cln, label='clean mean', color='C0')
    axes[i].set_title(f"{lab}: red={m_core.reduction_db:.2f} dB pres_sat_raw={safe.get('pres_sat_raw', float('nan')):.3f} pres_sat_clip={safe.get('pres_sat_clip', float('nan')):.3f}")
    axes[i].legend(loc='upper right')

plt.tight_layout()
out_png = os.path.join(outdir, 'spectra.png')
plt.savefig(out_png)
print('\nSaved combined spectra to', out_png)
plt.show()

# print final concise summary
print('\nSummary:')
for lab, (m_core, safe) in zip(labels, results):
    print(f"{lab}: reduction={m_core.reduction_db:.2f} dB, core_pres_sat={m_core.pres_sat:.3f}, safe_pres_sat_raw={safe.get('pres_sat_raw', float('nan')):.3f}, safe_pres_sat_clip={safe.get('pres_sat_clip', float('nan')):.3f}, comb_resid={m_core.comb_resid:.3f}")

print('\nDone')
