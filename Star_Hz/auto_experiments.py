import os
from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from visualization import plot_heatmaps, plot_mean_spectra, plot_comb_resid

outdir = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(outdir, exist_ok=True)

p = SimParams(df_mhz=0.05, dt_s=1.0)
sim = generate_synthetic_cube(p, seed=2025)
raw = sim['raw']
freqs = sim['freqs']
comb_peaks = list(p.comb_peak_freqs_mhz)

# Base conservative config used earlier
base = SubtractConfig(
    sat_window=(25, 35), edges_margin=12, shoulder=4,
    protect_bw=0.36, science_core_bw=0.12, notch_bw=0.10,
    r1=1, r2=0, poly_order=1, clip_k=2.5,
    subtract_protect=False, safety_floor_q=0.0, gate_taper_s=2.0,
    gate_strength_k=2.0, gate_strength_slope=1.0,
    restore_science=True, restore_max_frac=0.3,
)

experiments = []
# A) restore_science = False
cfgA = SubtractConfig(**vars(base))
cfgA.restore_science = False
experiments.append(('A_restore_off', cfgA))
# B) restore_max_frac small
cfgB = SubtractConfig(**vars(base))
cfgB.restore_max_frac = 0.05
experiments.append(('B_restore_max_0p05', cfgB))
# C) r1=2, r2=1
cfgC = SubtractConfig(**vars(base))
cfgC.r1 = 2
cfgC.r2 = 1
experiments.append(('C_r1_2_r2_1', cfgC))
# D) small grid sweep: restore_max_frac in [0.05,0.2], r1 in [1,2], r2 in [0,1]
labels = []
for rm in [0.05, 0.2]:
    for r1 in [1,2]:
        for r2 in [0,1]:
            cfg = SubtractConfig(**vars(base))
            cfg.restore_max_frac = rm
            cfg.r1 = r1
            cfg.r2 = r2
            label = f'D_grid_rm{int(rm*100)}_r1{r1}_r2{r2}'
            experiments.append((label, cfg))

results = []
for label, cfg in experiments:
    print('\n--- Running', label)
    cln, m = run_two_stage_subtraction(raw=raw, freqs=freqs, cfg=cfg)
    print(f"{label} -> Reduction={m.reduction_db:.2f} dB | pres_off={m.pres_off:.3f} | pres_sat={m.pres_sat:.3f} | comb_resid={m.comb_resid:.3f}")
    results.append((label, m))

    # Save plots
    hp = os.path.join(outdir, f"{label}_heatmap.png")
    sp = os.path.join(outdir, f"{label}_spectra.png")
    cp = os.path.join(outdir, f"{label}_comb.png")
    try:
        plot_heatmaps(raw, cln, freqs, title_prefix=label+' ', save_path=hp)
        plot_mean_spectra(raw, cln, freqs, cfg.sat_window, cfg.edges_margin, p.science_freq_mhz, cfg.protect_bw, cfg.science_core_bw, comb_peaks, save_path=sp)
        plot_comb_resid(m, save_path=cp)
    except Exception as e:
        print('Plotting failed for', label, e)

# Print summary table
print('\n=== Summary ===')
for label, m in results:
    print(f"{label}: reduction={m.reduction_db:.2f} dB, pres_off={m.pres_off:.3f}, pres_sat={m.pres_sat:.3f}, comb_resid={m.comb_resid:.3f}")

print('\nPlots saved to', outdir)
