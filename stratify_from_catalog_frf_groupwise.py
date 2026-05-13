#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/smoke_background.npz")
    ap.add_argument("--nt", type=int, default=48)
    ap.add_argument("--nf", type=int, default=192)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    freqs_hz = np.linspace(110e6, 190e6, args.nf)
    # Fake JD grid near the sample TLE epoch. Science runs must use real times.
    times_jd = 2460310.5 + np.arange(args.nt) * 10.0 / 86400.0
    baseline_enu_m = np.array([73.0, 0.0, 0.0])
    # Smooth-ish complex foreground-like background + thermal component.
    f = (freqs_hz - freqs_hz.mean()) / np.ptp(freqs_hz)
    t = np.linspace(0, 1, args.nt)[:, None]
    smooth = (10.0 * np.exp(-2.0 * f[None, :] ** 2)) * np.exp(2j * np.pi * (0.2 * t + 0.5 * f[None, :]))
    noise = 0.05 * (rng.normal(size=(args.nt, args.nf)) + 1j * rng.normal(size=(args.nt, args.nf)))
    vis_tf = smooth + noise
    weights_tf = np.ones((args.nt, args.nf), dtype=float)
    # Add a few invalid/flagged stripes to test weighting.
    weights_tf[10:12, 80:100] = 0.0
    weights_tf[30, 120:145] = 0.0
    np.savez(out, vis_tf=vis_tf, freqs_hz=freqs_hz, times_jd=times_jd, baseline_enu_m=baseline_enu_m, weights_tf=weights_tf)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
