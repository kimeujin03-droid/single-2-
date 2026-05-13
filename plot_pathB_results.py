# Path B configuration template.
# Replace paths with your own HERA-like NPZ backgrounds and TLE file.

site:
  # HERA approximate site coordinates, edit if needed.
  lat_deg: -30.7215
  lon_deg: 21.4283
  elev_m: 1073.0

backgrounds:
  - id: bg_01_clean
    path: examples/smoke_background.npz
    product: smoke_test_npz_not_science
    processing_history: synthetic background for code smoke test only
  # Real example structure:
  # - id: bg_LST_01_clean
  #   path: /mnt/d/HERA/backgrounds/bg_LST_01.npz
  #   product: HERA_IDR_xxx_exported_NPZ
  #   processing_history: "calibrated/LST-binned/flagged; inpainting status: specify here"
  # - id: bg_LST_04_flagged
  #   path: /mnt/d/HERA/backgrounds/bg_LST_04.npz
  #   product: HERA_IDR_xxx_exported_NPZ
  #   processing_history: "include flag fraction and preprocessing notes"

baseline_groups:
  # By default these are synthetic baseline-vector overrides: the same background
  # visibility product is reused while only satellite geometry and window/horizon
  # calculations use the listed baseline vector. For real per-baseline runs, export
  # separate NPZ backgrounds for each baseline and use `use_native_baseline: true`.
  - id: short_EW_14p6m
    length_m: 14.6
    orientation: EW
  - id: mid_EW_29p2m
    length_m: 29.2
    orientation: EW
  - id: long_EW_73m
    length_m: 73.0
    orientation: EW
  # Example for a real per-baseline NPZ background, when the loaded background
  # already has the desired baseline_enu_m:
  # - id: native_from_npz
  #   use_native_baseline: true

starlink:
  # For smoke test this points to an old sample TLE. For science, replace with current Starlink GP/TLE catalog.
  tle_path: examples/sample_for_smoke_test_only.tle
  max_scan_satellites: 500
  peak_alt_min_deg: 15.0
  peak_alt_max_deg: 90.0
  reference_flux_jy: 100.0
  reference_range_km: 550.0
  range_attenuation_mode: flux_density_1_over_r2
  emission_model:
    # Optional: set a measured/literature-derived template CSV with freq_hz/freq_mhz and relative_amplitude/relative_power/flux_jy.
    # spectral_template_csv: /mnt/d/HERA/templates/starlink_uemr_template.csv
    bassa_hba_window_flux_jy: 30.0
    narrowband_peak_flux_jy: 50.0
    narrowband_lines_mhz: [125.0, 135.0, 143.05, 150.0, 175.0]
    line_render_width_hz: 60000.0

beam:
  mode: gaussian
  fwhm_deg_ref: 10.0
  freq_ref_hz: 150000000.0

pipeline:
  inpainting_surrogate:
    enabled: false
    poly_order: 3
    min_good_channels: 12
  delay_filter:
    enabled: true
    taper: blackman_harris
    buffer_ns: 100.0
  fr_zero_notch:
    enabled: false
    width_mhz: 0.03
  mainlobe_fr_filter:
    enabled: false
    mode: remove_mainlobe
    width_mhz: 0.8

metrics:
  window:
    taper: blackman_harris
    buffer_ns: 100.0
  phase_randomized_null:
    n_trials: 100
    seed: 42
    # Main-claim default: per_time preserves the spectral template coherence while
    # randomizing the phase independently per integration.
    # Alternatives: global, per_freq, per_pixel.
    # Use per_pixel only as an aggressive sensitivity test because it destroys
    # both temporal and spectral coherence and may make p95 too easy to exceed.
    mode: per_time

experiment:
  flux_grid_jy: [10, 30, 100, 300, 1000]
