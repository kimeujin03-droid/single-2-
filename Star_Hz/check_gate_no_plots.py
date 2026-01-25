from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction

p = SimParams(df_mhz=0.05, dt_s=1.0)
sim = generate_synthetic_cube(p, seed=2025)
raw = sim['raw']
freqs = sim['freqs']

cfg0 = SubtractConfig(
    sat_window=(25, 35), edges_margin=15, shoulder=4,
    protect_bw=0.40, science_core_bw=0.15, notch_bw=0.10,
    r1=2, poly_order=1, clip_k=2.5,
    subtract_protect=False, safety_floor_q=0.0, gate_taper_s=2.0,
    r2=1,
)

cln, m = run_two_stage_subtraction(raw, freqs, cfg0)
print(f"reduction_db={m.reduction_db:.2f} pres_off={m.pres_off:.3f} pres_sat={m.pres_sat:.3f} comb_resid={m.comb_resid:.3f}")
