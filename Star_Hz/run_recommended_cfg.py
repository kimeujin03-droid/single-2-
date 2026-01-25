from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from visualization import plot_heatmaps, plot_mean_spectra, plot_comb_resid

if __name__ == '__main__':
    p = SimParams(df_mhz=0.05, dt_s=1.0)
    sim = generate_synthetic_cube(p, seed=2025)
    raw = sim['raw']
    freqs = sim['freqs']

    # Recommended conservative settings per your guidance
    cfg = SubtractConfig(
        sat_window=(25, 35),  # fix satellite window (priority 1)
        edges_margin=12,
        shoulder=4,

        protect_bw=0.36,           # protect wider than core (2-4x)
        science_core_bw=0.12,      # narrow core: what we must preserve
        notch_bw=0.10,

        r1=1,                      # conservative start
        r2=0,                      # avoid aggressive comb refinement initially
        poly_order=1,
        clip_k=2.5,

        subtract_protect=False,    # do NOT inpaint inside protect
        safety_floor_q=0.0,
        gate_taper_s=2.0,

        gate_strength_k=2.0,
        gate_strength_slope=1.0,

        restore_science=True,
        restore_max_frac=0.3,
    )

    cln, m = run_two_stage_subtraction(raw=raw, freqs=freqs, cfg=cfg)

    print("Recommended-config results:")
    print(f"  Reduction: {m.reduction_db:.2f} dB")
    print(f"  Preservation: OFF={m.pres_off*100:.1f}%, SAT={m.pres_sat*100:.1f}%")
    print(f"  Comb residual: {m.comb_resid:.3f}")

    # Show plots for visual inspection
    plot_heatmaps(raw, cln, freqs, title_prefix='RECOMMENDED ')
    plot_mean_spectra(raw, cln, freqs, cfg.sat_window, cfg.edges_margin,
                      p.science_freq_mhz, cfg.protect_bw, cfg.science_core_bw,
                      list(p.comb_peak_freqs_mhz))
    plot_comb_resid(m)
