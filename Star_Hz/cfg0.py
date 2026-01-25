from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from visualization import plot_heatmaps, plot_mean_spectra, plot_comb_resid

if __name__ == '__main__':
    p = SimParams(df_mhz=0.05, dt_s=1.0)
    sim = generate_synthetic_cube(p, seed=2025)
    raw = sim['raw']
    freqs = sim['freqs']

    cfg0 = SubtractConfig(
    sat_window=(25, 35),
    edges_margin=15,
    shoulder=4,

    protect_bw=0.40,
    science_core_bw=0.12,   # 0.15 → 0.12 정도로 시작 추천
    notch_bw=0.12,

    r1=1,                   # 핵심: 1로 시작
    poly_order=1,
    clip_k=4.0,

    subtract_protect=False, # 일단 False 유지 (restore 지옥 피하기)
    safety_floor_q=0.0,
    gate_taper_s=2.0,

    r2=1,
)

    cln0, m0 = run_two_stage_subtraction(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        cfg=cfg0
    )

    print(f"Reduction: {m0.reduction_db:.2f} dB")
    print(f"Preservation: OFF={m0.pres_off*100:.1f}%, SAT={m0.pres_sat*100:.1f}%")
    print(f"Comb residual: {m0.comb_resid:.2f}")

    # Show plots
    plot_heatmaps(raw, cln0, freqs, title_prefix='USER_CFG0 ')
    plot_mean_spectra(raw, cln0, freqs, cfg0.sat_window, cfg0.edges_margin,
                      p.science_freq_mhz, cfg0.protect_bw, cfg0.science_core_bw,
                      list(p.comb_peak_freqs_mhz))
    plot_comb_resid(m0)
