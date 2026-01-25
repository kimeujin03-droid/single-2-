import os
import csv
import math
from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from visualization import plot_heatmaps, plot_mean_spectra, plot_comb_resid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

outdir = os.path.join(os.path.dirname(__file__), 'outputs', 'expanded')
os.makedirs(outdir, exist_ok=True)

p = SimParams(df_mhz=0.05, dt_s=1.0)
sim = generate_synthetic_cube(p, seed=2027)
raw = sim['raw']
freqs = sim['freqs']
comb_peaks = list(p.comb_peak_freqs_mhz)

# Grid (moderate size)
r1_list = [2, 4, 6]
r2_list = [0, 1, 2]
restore_max_fracs = [0.01, 0.05]
gate_ks = [2.0, 4.0]

rows = []
run_count = 0
for r1 in r1_list:
    for r2 in r2_list:
        for rm in restore_max_fracs:
            for gk in gate_ks:
                cfg = SubtractConfig(
                    sat_window=(25, 35), edges_margin=12, shoulder=4,
                    protect_bw=0.36, science_core_bw=0.12, notch_bw=0.10,
                    r1=r1, r2=r2, poly_order=1, clip_k=2.5,
                    subtract_protect=True, safety_floor_q=0.0, gate_taper_s=2.0,
                    gate_strength_k=gk, gate_strength_slope=1.0,
                    restore_science=True, restore_max_frac=rm,
                )
                label = f"exp_r1{r1}_r2{r2}_rm{int(rm*1000)}_gk{int(gk)}"
                print('Running', label)
                cln, m = run_two_stage_subtraction(raw=raw, freqs=freqs, cfg=cfg)
                print(f" -> Reduction={m.reduction_db:.2f} dB | pres_off={m.pres_off:.3f} | pres_sat={m.pres_sat:.3f} | comb_resid={m.comb_resid:.3f}")

                # Save plots
                heat_path = os.path.join(outdir, f"{label}_heatmap.png")
                spec_path = os.path.join(outdir, f"{label}_spectra.png")
                comb_path = os.path.join(outdir, f"{label}_comb.png")
                try:
                    plot_heatmaps(raw, cln, freqs, title_prefix=label+' ', save_path=heat_path)
                    plot_mean_spectra(raw, cln, freqs, cfg.sat_window, cfg.edges_margin, p.science_freq_mhz, cfg.protect_bw, cfg.science_core_bw, comb_peaks, save_path=spec_path)
                    plot_comb_resid(m, save_path=comb_path)
                except Exception as e:
                    print('Plotting failed for', label, e)

                # Create thumbnails (small) by reading and re-saving smaller
                def make_thumb(src, dst, maxdim=240):
                    try:
                        img = mpimg.imread(src)
                        h, w = img.shape[0], img.shape[1]
                        scale = maxdim / max(h, w)
                        nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
                        plt.imsave(dst, img, format='png')
                        # Resize using PIL if available for better quality
                        try:
                            from PIL import Image
                            im = Image.open(dst)
                            im = im.resize((nw, nh), Image.LANCZOS)
                            im.save(dst)
                        except Exception:
                            pass
                    except Exception as e:
                        print('Thumb failed', src, e)

                heat_thumb = os.path.join(outdir, f"{label}_heat_thumb.png")
                spec_thumb = os.path.join(outdir, f"{label}_spec_thumb.png")
                comb_thumb = os.path.join(outdir, f"{label}_comb_thumb.png")
                make_thumb(heat_path, heat_thumb)
                make_thumb(spec_path, spec_thumb)
                make_thumb(comb_path, comb_thumb)

                rows.append({
                    'label': label,
                    'r1': r1, 'r2': r2, 'restore_max_frac': rm, 'gate_k': gk,
                    'reduction_db': m.reduction_db, 'pres_off': m.pres_off, 'pres_sat': m.pres_sat, 'comb_resid': m.comb_resid,
                    'heat': heat_path, 'spec': spec_path, 'comb': comb_path,
                    'heat_thumb': heat_thumb, 'spec_thumb': spec_thumb, 'comb_thumb': comb_thumb,
                })
                run_count += 1

# Save CSV
csv_path = os.path.join(outdir, 'expanded_summary.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['label','r1','r2','restore_max_frac','gate_k','reduction_db','pres_off','pres_sat','comb_resid','heat','spec','comb','heat_thumb','spec_thumb','comb_thumb'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# Create simple HTML index
html_path = os.path.join(outdir, 'index.html')
with open(html_path, 'w', encoding='utf8') as f:
    f.write('<html><head><meta charset="utf-8"><title>Expanded Sweep Results</title></head><body>')
    f.write('<h1>Expanded Sweep Results</h1>')
    f.write('<table border="1" cellpadding="6" cellspacing="0">')
    f.write('<tr><th>Label</th><th>Config</th><th>Metrics</th><th>Heat</th><th>Spec</th><th>Comb</th></tr>')
    for r in rows:
        f.write('<tr>')
        f.write(f"<td>{r['label']}</td>")
        f.write(f"<td>r1={r['r1']} r2={r['r2']} rm={r['restore_max_frac']} gk={r['gate_k']}</td>")
        f.write(f"<td>reduction={r['reduction_db']:.2f}<br>pres_off={r['pres_off']:.3f}<br>pres_sat={r['pres_sat']:.3f}<br>comb={r['comb_resid']:.3f}</td>")
        f.write(f"<td><a href='{os.path.basename(r['heat'])}'><img src='{os.path.basename(r['heat_thumb'])}'></a></td>")
        f.write(f"<td><a href='{os.path.basename(r['spec'])}'><img src='{os.path.basename(r['spec_thumb'])}'></a></td>")
        f.write(f"<td><a href='{os.path.basename(r['comb'])}'><img src='{os.path.basename(r['comb_thumb'])}'></a></td>")
        f.write('</tr>')
    f.write('</table></body></html>')

print('\nDone. Runs:', run_count)
print('CSV:', csv_path)
print('HTML index:', html_path)
