"""
svd_diagnostics.py

Single-epoch truncated-SVD diagnostics for the Star_Hz synthetic generator.

This script is intentionally minimal and deterministic:
- Generates one synthetic snapshot (raw = background + science + satellite)
- Runs a rank sweep for truncated-SVD removal (cleaned product = residual Ek = X - Lk)
- Computes:
  * Bias_under(k): relative RFI leakage, using R_hat = Lk vs R_true = satellite
  * Bias_over(k): template-based science loss proxy, using overlap between removed subspace Lk and science template S_true
  * Pareto curve in (Bias_under, Bias_over)

Optionally, it can run a lightweight frequency-weighted SVD (FWSVD) using MAD-based channel downweighting.

Outputs:
- outputs/svd_rank_sweep.csv
- outputs/svd_rank_sweep.png
- outputs/svd_pareto.png
- outputs/svd_mixed_modes.png
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib

# Force a non-interactive backend for reliability (prevents Tk/Tcl issues on some Windows setups)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import SimParams, generate_synthetic_cube


def _fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def fro_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius-normalized correlation: <A,B> / (||A||_F ||B||_F)."""
    num = float(np.sum(A * B))
    den = float(np.linalg.norm(A, ord="fro") * np.linalg.norm(B, ord="fro") + 1e-12)
    return num / den


def snr_peak_energy(S_true: np.ndarray, R_true: np.ndarray, N_true: np.ndarray) -> Dict[str, float]:
    """Peak-based & energy-based SNR for RFI vs noise."""
    # Peak SNR (RFI vs noise)
    peak_R = float(np.max(np.abs(R_true)))
    sigma_N = float(np.std(N_true))
    snr_peak = peak_R / (sigma_N + 1e-12)

    # Energy SNR (Frobenius)
    energy_R = _fro_norm(R_true)
    energy_N = _fro_norm(N_true)
    snr_energy = energy_R / (energy_N + 1e-12)

    return {"snr_peak": snr_peak, "snr_energy": snr_energy}


def truncated_svd_Lk(X: np.ndarray, k: int) -> np.ndarray:
    """Rank-k truncated SVD reconstruction Lk."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = int(max(0, min(k, s.shape[0])))
    if k == 0:
        return np.zeros_like(X)
    return (U[:, :k] * s[:k][None, :]) @ Vt[:k, :]


def fwsdv_Lk(X: np.ndarray, k: int, tau: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frequency-weighted SVD (FWSVD) contaminant estimate.

    We compute per-channel MAD across time, identify "outlier" channels by comparing to the
    median MAD, downweight them, and compute SVD on XW.

    Returns
    -------
    (Lk, w)
      Lk : rank-k contaminant estimate in the original X space
      w  : per-channel weights (shape (F,))
    """
    # MAD per channel
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med[None, :]), axis=0) + 1e-12
    mad_med = float(np.median(mad) + 1e-12)

    # outlier rule: MAD > tau * median(MAD)
    out = mad > (float(tau) * mad_med)
    w = np.ones(X.shape[1], dtype=np.float32)
    w[out] = np.clip(1.0 - float(alpha), 0.05, 1.0)

    Xw = X * w[None, :]
    U, s, Vt = np.linalg.svd(Xw, full_matrices=False)
    k = int(max(0, min(k, s.shape[0])))
    if k == 0:
        return np.zeros_like(X), w

    Lw = (U[:, :k] * s[:k][None, :]) @ Vt[:k, :]  # approx to XW
    # Map back to original coordinates: X ≈ Lw W^{-1}
    Lk = Lw / (w[None, :] + 1e-12)
    return Lk.astype(np.float32), w


def bias_under(R_true: np.ndarray, R_hat: np.ndarray) -> float:
    """Relative RFI leakage: ||R_true - R_hat||_F / ||R_true||_F."""
    denom = _fro_norm(R_true) + 1e-12
    return _fro_norm(R_true - R_hat) / denom


def bias_over(S_true: np.ndarray, S_hat: np.ndarray) -> float:
    """Science distortion metric derived from a preservation proxy.

    We measure a simple preservation proxy as the fraction of the science
    Frobenius norm that remains in the cleaned product S_hat:

        preservation = ||S_hat||_F / ||S_true||_F

    and define distortion = 1 - preservation so that distortion -> 0 when
    science is fully preserved. The returned value is clamped to [0, +inf)
    in practice (preservation may slightly exceed 1 due to numerical/overlap
    effects; we clip the preservation to at most 1 to keep distortion >= 0).
    """
    denom = _fro_norm(S_true) + 1e-12
    preservation = _fro_norm(S_hat) / denom
    # Clamp preservation to at most 1 to avoid negative distortion from noise
    preservation = min(1.0, float(preservation))
    distortion = 1.0 - preservation
    return float(distortion)


def make_plots(
    outdir: str,
    freqs: np.ndarray,
    times: np.ndarray,
    raw: np.ndarray,
    sci: np.ndarray,
    sat: np.ndarray,
    ks: np.ndarray,
    under: np.ndarray,
    over: np.ndarray,
    method_label: str,
    extra_tag: str = ""
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Rank sweep plot
    plt.figure()
    plt.plot(ks, under, label="Bias_under (RFI leakage)")
    plt.plot(ks, over, label="Bias_over (science loss proxy)")
    plt.xlabel("Rank cutoff k")
    plt.ylabel("Relative bias (log scale; lower is better)")
    plt.yscale("log")
    plt.axvline(8, ls="--", lw=1, alpha=0.5)
    plt.title(f"Rank sweep ({method_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"svd_rank_sweep{extra_tag}.png"), dpi=180)
    plt.close()

    # Pareto plot
    plt.figure()
    plt.plot(under, over, marker="o", linestyle="-")
    for i, k in enumerate(ks):
        if i % max(1, len(ks)//10) == 0:
            plt.text(float(under[i]), float(over[i]), str(int(k)), fontsize=8)
    plt.xlabel("Bias_under (RFI leakage)")
    plt.ylabel("Bias_over (science loss proxy)")
    plt.title(f"Pareto curve ({method_label})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"svd_pareto{extra_tag}.png"), dpi=180)
    plt.close()

    # Mixed-mode visualization: show top singular vectors and a representative rank-1 mode
    U, s, Vt = np.linalg.svd(raw, full_matrices=False)
    u1 = U[:, 0]
    v1 = Vt[0, :]
    L1 = (U[:, :1] * s[:1][None, :]) @ Vt[:1, :]

    # Mixing diagnostic: how aligned is the leading mode with science vs RFI?
    rho_L1_S = fro_corr(L1, sci)
    rho_L1_R = fro_corr(L1, sat)
    print(f"[MIXING] corr(L1, Science) = {rho_L1_S:.3f}, corr(L1, RFI) = {rho_L1_R:.3f}")

    plt.figure()
    plt.plot(times, u1)
    plt.xlabel("Time (s)")
    plt.ylabel("u1 (unitless)")
    plt.title("Leading left singular vector u1(t)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"svd_u1{extra_tag}.png"), dpi=180)
    plt.close()

    plt.figure()
    plt.plot(freqs, v1)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("v1 (unitless)")
    plt.title("Leading right singular vector v1(ν)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"svd_v1{extra_tag}.png"), dpi=180)
    plt.close()

    # Heatmap comparison: raw vs satellite vs science vs L1
    def _imshow(mat, title, ax):
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       extent=[float(freqs.min()), float(freqs.max()), float(times.min()), float(times.max())])
        ax.set_title(title)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Time (s)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    _imshow(raw, "Raw X", axs[0, 0])
    _imshow(sat, "True RFI R", axs[0, 1])
    _imshow(sci, "True Science S", axs[1, 0])
    _imshow(L1, "Rank-1 removed mode L1", axs[1, 1])
    fig.suptitle("Mixed-mode inspection (representative)")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"svd_mixed_modes{extra_tag}.png"), dpi=180)
    plt.close(fig)


def make_pareto_compare_plot(
    outdir: str,
    under_tsvd: np.ndarray,
    over_tsvd: np.ndarray,
    under_fwsdv: np.ndarray,
    over_fwsdv: np.ndarray,
    extra_tag: str = "",
) -> None:
    """Overlay TSVD vs FWSVD Pareto curves (log-log)."""
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(under_tsvd, over_tsvd, marker="o", linestyle="-", label="TSVD")
    plt.plot(under_fwsdv, over_fwsdv, marker="o", linestyle="-", label="FWSVD")
    plt.xscale("log")
    plt.yscale("log")

    # Annotate a few TSVD points: k=1, elbow-ish k=8, and max k
    n = int(len(under_tsvd))
    for idx in [0, 7, -1]:
        j = idx if idx >= 0 else (n - 1)
        k_val = j + 1
        plt.text(
            float(under_tsvd[idx]),
            float(over_tsvd[idx]),
            f"k={k_val}",
            fontsize=8,
            color="blue",
        )
    plt.xlabel("Bias_under (RFI leakage)")
    plt.ylabel("Bias_over (science loss proxy)")
    plt.title("Pareto curve comparison (TSVD vs FWSVD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"svd_pareto_compare{extra_tag}.png"), dpi=180)
    plt.close()


def run_rank_sweep(
    sim: Dict[str, np.ndarray],
    max_k: int,
    use_fwsdv: bool = False,
    tau: float = 3.5,
    alpha: float = 0.2
) -> Dict[str, np.ndarray]:
    # 1) Ground truth: never overwrite these.
    S_true = sim["science"].astype(np.float32)
    R_true = sim["satellite"].astype(np.float32)

    # 2) Data matrix X: take as generated (includes background+noise etc).
    # Do not regenerate or modify it during the sweep.
    X = sim["raw"].astype(np.float32)

    max_k = int(max(1, min(max_k, min(X.shape))))
    ks = np.arange(1, max_k + 1, dtype=int)

    under = np.zeros_like(ks, dtype=np.float64)
    over = np.zeros_like(ks, dtype=np.float64)

    if use_fwsdv:
        # FWSVD path keeps its own weighted SVD per k.
        for i, k in enumerate(ks):
            L_k, _w = fwsdv_Lk(X, k, tau=tau, alpha=alpha)
            E_k = X - L_k

            S_hat = E_k
            R_hat = L_k

            under[i] = bias_under(R_true, R_hat)
            over[i] = bias_over(S_true, S_hat)
    else:
        # 3) Single SVD on X, used for all k.
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        for i, k in enumerate(ks):
            # rank-k low-rank approximation (removed subspace estimate)
            L_k = (U[:, :k] * s[:k][None, :]) @ Vt[:k, :]
            E_k = X - L_k

            S_hat = E_k
            R_hat = L_k

            under[i] = bias_under(R_true, R_hat)
            over[i] = bias_over(S_true, S_hat)

    return {"k": ks, "bias_under": under, "bias_over": over}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--max_k", type=int, default=25)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--run_tag", type=str, default="", help="Optional suffix tag for output filenames (e.g., simple_comb)")

    # Amplitude knobs (useful for making the bias curves readable)
    ap.add_argument("--sat_scale", type=float, default=1.0, help="Multiply p.flux_sat_peak_jy by this factor")
    ap.add_argument("--sci_scale", type=float, default=1.0, help="Multiply p.science_flux_jy by this factor")

    # Realism toggles for satellite mixing (exposes the new v2 parameters)
    ap.add_argument("--sat_n_events", type=int, default=1)
    ap.add_argument("--sat_event_sep", type=float, default=0.0)
    ap.add_argument("--comb_drift", type=float, default=0.0, help="MHz per second")
    ap.add_argument("--broadband_slope", type=float, default=0.0)
    ap.add_argument("--ripple_amp", type=float, default=0.0)
    ap.add_argument("--ripple_period_mhz", type=float, default=2.0)

    # Optional complexity knob: duplicate comb peaks to increase effective rank.
    # This perturbs the default peak list by adding small, deterministic offsets.
    ap.add_argument("--comb_peaks_mul", type=int, default=1, help="Duplicate comb peaks with small offsets (>=1)")

    # Optional FWSVD
    ap.add_argument("--use_fwsdv", action="store_true")
    ap.add_argument("--compare_fwsdv", action="store_true", help="Also run FWSVD and overlay Pareto vs TSVD")
    ap.add_argument("--tau", type=float, default=3.5)
    ap.add_argument("--alpha", type=float, default=0.2)

    args = ap.parse_args()

    p = SimParams(df_mhz=0.05, dt_s=1.0)

    # Apply amplitude scaling (do this before generating the cube)
    p.flux_sat_peak_jy = float(p.flux_sat_peak_jy) * float(args.sat_scale)
    p.science_flux_jy = float(p.science_flux_jy) * float(args.sci_scale)

    # Apply realism toggles
    p.satellite_n_events = int(args.sat_n_events)
    p.satellite_event_separation_s = float(args.sat_event_sep)
    p.comb_drift_mhz_per_s = float(args.comb_drift)
    p.broadband_slope = float(args.broadband_slope)
    p.ripple_amp = float(args.ripple_amp)
    p.ripple_period_mhz = float(args.ripple_period_mhz)

    # Optionally increase comb peak count (more structure => bias_under tends to decay more slowly with k)
    mul = max(int(args.comb_peaks_mul), 1)
    if mul > 1:
        base = list(p.comb_peak_freqs_mhz)
        offsets = np.linspace(-0.25, 0.25, mul, dtype=float)  # MHz offsets
        peaks = []
        for pf in base:
            for off in offsets:
                peaks.append(float(pf) + float(off))
        p.comb_peak_freqs_mhz = tuple(peaks)

    sim = generate_synthetic_cube(p, seed=args.seed)

    # Optional SNR summary (RFI vs noise). If the generator doesn't expose noise explicitly,
    # approximate it as (raw - science - satellite).
    if "noise" in sim:
        noise = sim["noise"]
    else:
        noise = sim["raw"] - sim["science"] - sim["satellite"]
    snr = snr_peak_energy(sim["science"], sim["satellite"], noise)
    print(f"[SNR] peak={snr['snr_peak']:.2f}, energy={snr['snr_energy']:.2f}")

    # Rank sweep
    sweep = run_rank_sweep(sim, max_k=args.max_k, use_fwsdv=bool(args.use_fwsdv), tau=args.tau, alpha=args.alpha)
    ks = sweep["k"]
    under = sweep["bias_under"]
    over = sweep["bias_over"]

    method = "FWSVD" if args.use_fwsdv else "TSVD"
    tag = "_fwsdv" if args.use_fwsdv else ""
    if str(args.run_tag).strip():
        safe = "".join([c if (c.isalnum() or c in "-_") else "_" for c in str(args.run_tag).strip()])
        tag = f"{tag}_{safe}"

    # Save CSV
    outdir = os.path.join(os.path.dirname(__file__), args.outdir)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"svd_rank_sweep{tag}.csv")
    header = "k,bias_under,bias_over\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        for k, bu, bo in zip(ks, under, over):
            f.write(f"{int(k)},{float(bu):.8f},{float(bo):.8f}\n")

    # Make plots
    make_plots(
        outdir=outdir,
        freqs=sim["freqs"].astype(np.float32),
        times=sim["times"].astype(np.float32),
        raw=sim["raw"].astype(np.float32),
        sci=sim["science"].astype(np.float32),
        sat=sim["satellite"].astype(np.float32),
        ks=ks,
        under=under,
        over=over,
        method_label=method,
        extra_tag=tag
    )

    # Print a compact summary
    i_best = int(np.argmin(under + over))
    print(f"[{method}] done. Outputs written to: {outdir}")
    print(f"Best (min bias_under+bias_over) at k={int(ks[i_best])}: "
          f"Bias_under={float(under[i_best]):.4f}, Bias_over={float(over[i_best]):.4f}")

    # Optional comparison plot: overlay TSVD vs FWSVD Pareto curves
    if bool(args.compare_fwsdv):
        sweep_t = run_rank_sweep(sim, max_k=args.max_k, use_fwsdv=False, tau=args.tau, alpha=args.alpha)
        sweep_f = run_rank_sweep(sim, max_k=args.max_k, use_fwsdv=True, tau=args.tau, alpha=args.alpha)
        make_pareto_compare_plot(
            outdir=outdir,
            under_tsvd=sweep_t["bias_under"],
            over_tsvd=sweep_t["bias_over"],
            under_fwsdv=sweep_f["bias_under"],
            over_fwsdv=sweep_f["bias_over"],
            extra_tag=tag,
        )
        it = int(np.argmin(sweep_t["bias_under"] + sweep_t["bias_over"]))
        jf = int(np.argmin(sweep_f["bias_under"] + sweep_f["bias_over"]))
        print(
            f"[COMPARE] TSVD best k={int(ks[it])}: "
            f"Bias_under={float(sweep_t['bias_under'][it]):.4f}, Bias_over={float(sweep_t['bias_over'][it]):.4f}"
        )
        print(
            f"[COMPARE] FWSVD best k={int(ks[jf])}: "
            f"Bias_under={float(sweep_f['bias_under'][jf]):.4f}, Bias_over={float(sweep_f['bias_over'][jf]):.4f}"
        )
    print(f"[COMPARE] Overlay written to: {os.path.join(outdir, f'svd_pareto_compare{tag}.png')}")
    print("SimParams (key toggles):", {
        "sat_n_events": p.satellite_n_events,
        "sat_event_sep": p.satellite_event_separation_s,
        "comb_drift": p.comb_drift_mhz_per_s,
        "broadband_slope": p.broadband_slope,
        "ripple_amp": p.ripple_amp,
        "ripple_period_mhz": p.ripple_period_mhz,
        "sat_scale": float(args.sat_scale),
        "sci_scale": float(args.sci_scale),
        "comb_peaks_mul": int(args.comb_peaks_mul),
    })


if __name__ == "__main__":
    main()
