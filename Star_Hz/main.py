"""
Main pipeline for radio interference subtraction.
"""
import json
from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from optimization import sweep_grid
from visualization import plot_heatmaps, plot_mean_spectra, plot_comb_resid


def save_best_json(best: dict, path: str):
    """Save best configuration to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)


def main():
    """Main execution pipeline."""
    # Generate synthetic data
    p = SimParams(df_mhz=0.05, dt_s=1.0)
    sim = generate_synthetic_cube(p, seed=2025)
    raw = sim["raw"]
    freqs = sim["freqs"]

    print(f"Data: dt={p.dt_s:.2f}s | df={p.df_mhz*1000:.1f}kHz | shape={raw.shape}")

    # Single run with recommended defaults
    print("\n[Single Run with Defaults]")
    cfg0 = SubtractConfig(
        sat_window=(15, 45),
        edges_margin=15,
        protect_bw=0.40,
        science_core_bw=0.15,
        notch_bw=0.10,
        r1=2,
        r2=1,
        poly_order=2,
        clip_k=2.5,
        subtract_protect=True,
        gate_taper_s=6.0,
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

    plot_heatmaps(raw, cln0, freqs, title_prefix="Default ")
    plot_mean_spectra(raw, cln0, freqs, cfg0.sat_window, cfg0.edges_margin,
                     p.science_freq_mhz, cfg0.protect_bw, cfg0.science_core_bw,
                     list(p.comb_peak_freqs_mhz))
    plot_comb_resid(m0)

    # Grid search optimization
    print("\n[Grid Search Optimization]")
    best, top = sweep_grid(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        sat_windows=[(12,48),(15,45),(18,42)],
        edges_margins=[15],
        protect_bws=[0.30,0.40,0.50],
        core_bws=[0.10,0.15,0.20],
        notch_bws=[0.08,0.10,0.15],
        r1_list=[1,2,3,4],
        r2_list=[0,1,2],
        poly_orders=[1,2,3],
        clip_ks=[2.0,2.5,3.0],
        safety_qs=[0.0],
        tapers=[4.0,6.0,8.0],
        top_k=10,
        verbose_every=0,
    )

    print("\n=== Top 5 Configurations ===")
    for i, rec in enumerate(top[:5], 1):
        cfg = rec["cfg"]
        m = rec["metrics"]
        print(f"\n#{i} Score={rec['score']:.2f}")
        print(f"  r1={cfg['r1']}, r2={cfg['r2']} | sat_window={cfg['sat_window']}")
        print(f"  protect_bw={cfg['protect_bw']:.2f}, core_bw={cfg['science_core_bw']:.2f}")
        print(f"  Reduction: {m['reduction_db']:.2f} dB")
        print(f"  Preservation: OFF={m['pres_off']*100:.1f}%, SAT={m['pres_sat']*100:.1f}%")

    # Visualize best result
    best_cfg = SubtractConfig(**best["cfg"])
    cln_best, m_best = run_two_stage_subtraction(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        cfg=best_cfg
    )
    
    plot_heatmaps(raw, cln_best, freqs, title_prefix="BEST ")
    plot_mean_spectra(raw, cln_best, freqs, best_cfg.sat_window, 
                     best_cfg.edges_margin, p.science_freq_mhz, 
                     best_cfg.protect_bw, best_cfg.science_core_bw,
                     list(p.comb_peak_freqs_mhz))

    save_best_json(best, "best_config.json")
    print("\n✅ Best configuration saved to: best_config.json")


if __name__ == "__main__":
    main()