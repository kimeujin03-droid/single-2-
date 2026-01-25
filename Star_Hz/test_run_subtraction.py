from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction

if __name__ == '__main__':
    p = SimParams(df_mhz=0.05, dt_s=1.0)
    sim = generate_synthetic_cube(p, seed=2025)
    raw = sim["raw"]
    freqs = sim["freqs"]

    cfg = SubtractConfig(sat_window=(15,45), edges_margin=15, protect_bw=0.40, science_core_bw=0.15, notch_bw=0.10,
                         r1=2, r2=1, poly_order=2, clip_k=2.5, subtract_protect=True, gate_taper_s=6.0)

    cln0, m0 = run_two_stage_subtraction(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        cfg=cfg
    )

    print(f"Reduction: {m0.reduction_db:.2f} dB")
    print(f"Preservation: OFF={m0.pres_off*100:.1f}%, SAT={m0.pres_sat*100:.1f}%")
    print(f"Comb residual: {m0.comb_resid:.2f}")
