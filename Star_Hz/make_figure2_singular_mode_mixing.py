"""make_figure2_singular_mode_mixing.py

Figure 2. Singular-mode mixing diagnostics.

Panels:
 (a) L1 (rank-1 removed mode) heatmap – Simple comb
 (b) L1 (rank-1 removed mode) heatmap – Comb drift
 (c) corr(L1, Science) vs corr(L1, RFI) point plot for the two scenarios

We intentionally recompute L1 and correlations here (using the same generator + fro_corr
as svd_diagnostics.py) so the figure matches the logged numbers.

Output:
  outputs/figure2_singular_mode_mixing.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import SimParams, generate_synthetic_cube
from svd_diagnostics import fro_corr


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    seed: int
    comb_drift: float
    sat_n_events: int
    sat_event_sep: float
    comb_peaks_mul: int = 1
    ripple_amp: float = 0.0
    ripple_period_mhz: float = 2.0


def _apply_comb_peaks_mul(p: SimParams, mul: int) -> None:
    mul = max(int(mul), 1)
    if mul <= 1:
        return
    base = list(p.comb_peak_freqs_mhz)
    offsets = np.linspace(-0.25, 0.25, mul, dtype=float)
    peaks = []
    for pf in base:
        for off in offsets:
            peaks.append(float(pf) + float(off))
    p.comb_peak_freqs_mhz = tuple(peaks)


def compute_L1_and_corr(sc: Scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    p = SimParams(df_mhz=0.05, dt_s=1.0)

    p.comb_drift_mhz_per_s = float(sc.comb_drift)
    p.satellite_n_events = int(sc.sat_n_events)
    p.satellite_event_separation_s = float(sc.sat_event_sep)
    p.ripple_amp = float(sc.ripple_amp)
    p.ripple_period_mhz = float(sc.ripple_period_mhz)
    _apply_comb_peaks_mul(p, sc.comb_peaks_mul)

    sim = generate_synthetic_cube(p, seed=sc.seed)

    X = sim["raw"].astype(np.float32)
    S = sim["science"].astype(np.float32)
    R = sim["satellite"].astype(np.float32)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    L1 = (U[:, :1] * s[:1][None, :]) @ Vt[:1, :]

    rho_S = float(fro_corr(L1, S))
    rho_R = float(fro_corr(L1, R))

    freqs = sim["freqs"].astype(np.float32)
    times = sim["times"].astype(np.float32)

    return freqs, times, L1, sim["raw"].astype(np.float32), rho_S, rho_R


def _imshow(
    ax,
    mat: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> None:
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=[float(freqs.min()), float(freqs.max()), float(times.min()), float(times.max())],
    )
    ax.set_title(title)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Time (s)")
    return im


def main() -> None:
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)

    # Match Figure 1 tags (same failure-mode definitions). Seed fixed for reproducibility.
    sc_simple = Scenario(
        key="simple",
        title="(a) L1 mode heatmap (Simple comb)",
        seed=2025,
        comb_drift=0.0,
        sat_n_events=1,
        sat_event_sep=0.0,
    )

    sc_drift = Scenario(
        key="drift",
        title="(b) L1 mode heatmap (Comb drift)",
        seed=2025,
        comb_drift=0.08,
        sat_n_events=3,
        sat_event_sep=6.0,
    )

    f1, t1, L1_simple, _Xsimple, rhoS_simple, rhoR_simple = compute_L1_and_corr(sc_simple)
    f2, t2, L1_drift, _Xdrift, rhoS_drift, rhoR_drift = compute_L1_and_corr(sc_drift)

    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.75], wspace=0.25, hspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # Shared color scale across (a)(b) for fair visual comparison
    all_vals = np.concatenate([L1_simple.ravel(), L1_drift.ravel()])
    # Robust range to avoid a single hot pixel dominating the colormap
    v = float(np.quantile(np.abs(all_vals), 0.995))
    vmin, vmax = -v, v

    im_a = _imshow(ax_a, L1_simple, f1, t1, sc_simple.title, vmin=vmin, vmax=vmax, cmap="coolwarm")
    im_b = _imshow(ax_b, L1_drift, f2, t2, sc_drift.title, vmin=vmin, vmax=vmax, cmap="coolwarm")

    # One shared colorbar with explicit units label (arb. units is acceptable in PASP/AAS style)
    cbar = fig.colorbar(im_b, ax=[ax_a, ax_b], fraction=0.046, pad=0.03)
    cbar.set_label("L1 amplitude (arb. units)")

    # (c) correlation summary
    ax_c.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax_c.axvline(0.0, color="k", lw=0.8, alpha=0.4)

    ax_c.scatter([rhoS_simple], [rhoR_simple], s=120, label="Simple", marker="o")
    ax_c.scatter([rhoS_drift], [rhoR_drift], s=120, label="Drift", marker="^")

    ax_c.set_xlabel("corr(L1, Science)")
    ax_c.set_ylabel("corr(L1, RFI)")
    ax_c.set_title("(c) Singular-mode mixing (correlation)")
    ax_c.grid(True, ls=":", lw=0.7, alpha=0.6)
    ax_c.legend(loc="best")

    # Bounds for readability: RFI-correlation is the key story; zoom y-range.
    ax_c.set_xlim(-0.25, 1.05)
    ax_c.set_ylim(0.7, 1.02)

    fig.suptitle("Figure 2. Singular-mode mixing diagnostics", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(outdir, "figure2_singular_mode_mixing.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("[FIG2] corr(L1,Science), corr(L1,RFI):")
    print(f"  Simple comb: ({rhoS_simple:.4f}, {rhoR_simple:.4f})")
    print(f"  Comb drift : ({rhoS_drift:.4f}, {rhoR_drift:.4f})")
    print(f"[FIG2] wrote: {out_path}")


if __name__ == "__main__":
    main()
