from data_generator import SimParams, generate_synthetic_cube
from optimization import sweep_grid

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

    print('=== Top Results ===')
    for i, rec in enumerate(top, 1):
        print(i, rec['score'], rec['cfg']['r1'], rec['cfg']['r2'], rec['metrics']['reduction_db'])
