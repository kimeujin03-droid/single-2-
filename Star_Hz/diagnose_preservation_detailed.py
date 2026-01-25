from data_generator import SimParams, generate_synthetic_cube, gaussian
from subtraction_core import SubtractConfig, run_two_stage_subtraction
from signal_processing import estimate_target_amp_matched, make_band_mask
import numpy as np


def diagnose(cfg, label):
    p = SimParams(df_mhz=0.05, dt_s=1.0)
    sim = generate_synthetic_cube(p, seed=2025)
    raw = sim['raw']
    freqs = sim['freqs']

    cln, m = run_two_stage_subtraction(raw=raw, freqs=freqs, cfg=cfg)

    n_time = raw.shape[0]
    t0, t1 = cfg.sat_window
    sat_mask = np.zeros(n_time, dtype=bool)
    sat_mask[t0:t1] = True
    off_mask = ~sat_mask

    df = float(np.median(np.diff(freqs))) if freqs.size>1 else 0.0
    sigma_mhz = max(1.5 * df, 0.5 * float(cfg.science_core_bw), 1e-6)

    raw_amps = []
    cln_amps = []
    for ti in range(n_time):
        ra = estimate_target_amp_matched(raw[ti], freqs, getattr(cfg,'target_f_mhz', p.science_freq_mhz), sigma_mhz,
                                         core_bw=cfg.science_core_bw, side_bw=cfg.protect_bw,
                                         poly_order=getattr(cfg,'poly_order',1), clip_k=getattr(cfg,'clip_k',2.5))
        ca = estimate_target_amp_matched(cln[ti], freqs, getattr(cfg,'target_f_mhz', p.science_freq_mhz), sigma_mhz,
                                         core_bw=cfg.science_core_bw, side_bw=cfg.protect_bw,
                                         poly_order=getattr(cfg,'poly_order',1), clip_k=getattr(cfg,'clip_k',2.5))
        raw_amps.append(ra)
        cln_amps.append(ca)

    raw_amps = np.array(raw_amps)
    cln_amps = np.array(cln_amps)

    def stats(x):
        return f"median={np.median(x):.4g}, mean={np.mean(x):.4g}, min={np.min(x):.4g}, max={np.max(x):.4g}, zeros={int((x==0).sum())}"

    print(f"--- DIAG ({label}) ---")
    print(f"ref_target_amp (median off raw amps): {np.median(raw_amps[off_mask]):.6g}")
    print("Raw amps (OFF):", stats(raw_amps[off_mask]))
    print("Raw amps (SAT):", stats(raw_amps[sat_mask]))
    print("Cln amps (OFF):", stats(cln_amps[off_mask]))
    print("Cln amps (SAT):", stats(cln_amps[sat_mask]))

    # show individual SAT times with raw vs cln and delta
    print("--- SAT per-time samples (raw -> cln) ---")
    for ti in np.where(sat_mask)[0]:
        print(f"t={ti}: raw={raw_amps[ti]:.6g} -> cln={cln_amps[ti]:.6g} delta={cln_amps[ti]-raw_amps[ti]:.6g}")

    # Inspect core region for a central SAT time
    mid = int((t0+t1)//2)
    core_mask = make_band_mask(freqs, getattr(cfg,'target_f_mhz', p.science_freq_mhz), cfg.science_core_bw)
    x = freqs[core_mask]
    g = gaussian(x, getattr(cfg,'target_f_mhz', p.science_freq_mhz), sigma_mhz)
    y_raw = raw[mid, core_mask]
    y_cln = cln[mid, core_mask]
    print(f"\nCentral SAT time t={mid} core samples count={core_mask.sum()}")
    print(f"gaussian template sum={g.sum():.6g} g_max={g.max():.6g}")
    print(f"raw core (first 8): {y_raw[:8]}")
    print(f"cln core (first 8): {y_cln[:8]}")
    print(f"dot(g, raw_core)={float(np.dot(g, y_raw)):.6g}, dot(g, cln_core)={float(np.dot(g,y_cln)):.6g}")


if __name__ == '__main__':
    # Use the recommended conservative cfg used previously
    cfg_rec = SubtractConfig(
        sat_window=(25, 35), edges_margin=12, shoulder=4,
        protect_bw=0.36, science_core_bw=0.12, notch_bw=0.10,
        r1=1, r2=0, poly_order=1, clip_k=2.5,
        subtract_protect=False, safety_floor_q=0.0, gate_taper_s=2.0,
        gate_strength_k=2.0, gate_strength_slope=1.0,
        restore_science=True, restore_max_frac=0.3,
    )

    diagnose(cfg_rec, 'recommended')

    # Also diagnose the user's earlier cfg0 (from run_user_cfg0)
    cfg0 = SubtractConfig(
        sat_window=(25, 35), edges_margin=15, shoulder=4,
        protect_bw=0.40, science_core_bw=0.15, notch_bw=0.10,
        r1=2, poly_order=1, clip_k=2.5,
        subtract_protect=False, safety_floor_q=0.0, gate_taper_s=2.0,
        r2=1,
    )

    diagnose(cfg0, 'user_cfg0')
