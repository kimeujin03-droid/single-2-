from data_generator import SimParams, generate_synthetic_cube
from optimization import sweep_grid
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from visualization import plot_heatmaps, plot_mean_spectra, plot_comb_resid

if __name__ == '__main__':
    p = SimParams(df_mhz=0.05, dt_s=1.0)
    sim = generate_synthetic_cube(p, seed=2025)
    raw = sim['raw']
    freqs = sim['freqs']

    best, top = sweep_grid(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        sat_windows=[(15,45)],
        edges_margins=[15],
        protect_bws=[0.40],
        core_bws=[0.15],
        notch_bws=[0.10],
        r1_list=[1,2],
        r2_list=[0,1],
        poly_orders=[1,2],
        clip_ks=[2.5],
        safety_qs=[0.0],
        tapers=[6.0],
        top_k=5,
        verbose_every=0,
    )

    if not top:
        print('No results from grid.')
        raise SystemExit(1)

    print('Best config (top[0]):')
    print(top[0]['cfg'])
    cfg_dict = top[0]['cfg']
    cfg = SubtractConfig(**cfg_dict)

    cln_best, m_best = run_two_stage_subtraction(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        cfg=cfg
    )

    print(f"Reduction: {m_best.reduction_db:.2f} dB | pres_off={m_best.pres_off:.3f} pres_sat={m_best.pres_sat:.3f}")

    # Show plots
    plot_heatmaps(raw, cln_best, freqs, title_prefix='BEST ')
    plot_mean_spectra(raw, cln_best, freqs, cfg.sat_window, cfg.edges_margin,
                      p.science_freq_mhz, cfg.protect_bw, cfg.science_core_bw,
                      list(p.comb_peak_freqs_mhz))
    plot_comb_resid(m_best)
