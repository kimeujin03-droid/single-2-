import os
import csv
from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from visualization import plot_heatmaps, plot_mean_spectra, plot_comb_resid

outdir = os.path.join(os.path.dirname(__file__), 'outputs', 'aggressive')
os.makedirs(outdir, exist_ok=True)

p = SimParams(df_mhz=0.05, dt_s=1.0)
sim = generate_synthetic_cube(p, seed=2026)
raw = sim['raw']
freqs = sim['freqs']
comb_peaks = list(p.comb_peak_freqs_mhz)

base = SubtractConfig(
    sat_window=(25, 35), edges_margin=12, shoulder=4,
    protect_bw=0.36, science_core_bw=0.12, notch_bw=0.10,
    r1=1, r2=0, poly_order=1, clip_k=2.5,
    subtract_protect=False, safety_floor_q=0.0, gate_taper_s=2.0,
    gate_strength_k=2.0, gate_strength_slope=1.0,
    restore_science=True, restore_max_frac=0.3,
)

# aggressive params grid
restore_max_fracs = [0.01, 0.02]
gate_ks = [3.0, 4.0, 6.0]
gate_ts = [3.0, 5.0]

rows = []
count = 0
for rm in restore_max_fracs:
    for gk in gate_ks:
        for gt in gate_ts:
            cfg = SubtractConfig(**vars(base))
            cfg.restore_max_frac = rm
            cfg.gate_strength_k = gk
            cfg.gate_taper_s = gt
            label = f"agg_rm{int(rm*1000)}_gk{int(gk)}_gt{int(gt)}"
            print('Running', label)
            cln, m = run_two_stage_subtraction(raw=raw, freqs=freqs, cfg=cfg)
            print(f" -> Reduction={m.reduction_db:.2f} dB | pres_off={m.pres_off:.3f} | pres_sat={m.pres_sat:.3f} | comb_resid={m.comb_resid:.3f}")

            hp = os.path.join(outdir, f"{label}_heatmap.png")
            sp = os.path.join(outdir, f"{label}_spectra.png")
            cp = os.path.join(outdir, f"{label}_comb.png")
            try:
                plot_heatmaps(raw, cln, freqs, title_prefix=label+' ', save_path=hp)
                plot_mean_spectra(raw, cln, freqs, cfg.sat_window, cfg.edges_margin, p.science_freq_mhz, cfg.protect_bw, cfg.science_core_bw, comb_peaks, save_path=sp)
                plot_comb_resid(m, save_path=cp)
            except Exception as e:
                print('Plotting failed for', label, e)

            rows.append({'label': label, 'reduction_db': m.reduction_db, 'pres_off': m.pres_off, 'pres_sat': m.pres_sat, 'comb_resid': m.comb_resid, 'heatmap': hp, 'spectra': sp, 'comb': cp})
            count += 1

# write CSV summary
csv_path = os.path.join(outdir, 'aggressive_summary.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['label','reduction_db','pres_off','pres_sat','comb_resid','heatmap','spectra','comb'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('\nDone. Wrote', len(rows), 'runs to', outdir)
print('CSV summary at', csv_path)
