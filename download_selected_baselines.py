#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import re
import numpy as np
from pyuvdata import UVData


def get_antpos_enu_and_ants(uvd):
    if hasattr(uvd, "get_enu_antpos"):
        ret = uvd.get_enu_antpos()
    elif hasattr(uvd, "get_ENU_antpos"):
        ret = uvd.get_ENU_antpos()
    elif hasattr(uvd, "telescope") and hasattr(uvd.telescope, "get_enu_antpos"):
        ret = uvd.telescope.get_enu_antpos()
    else:
        raise AttributeError("No ENU antenna-position method found in this pyuvdata version.")
    if isinstance(ret, tuple) and len(ret) >= 2:
        return np.asarray(ret[0]), np.asarray(ret[1])
    if isinstance(ret, np.ndarray):
        antpos_enu = np.asarray(ret)
        if hasattr(uvd, "telescope") and hasattr(uvd.telescope, "antenna_numbers"):
            ants = np.asarray(uvd.telescope.antenna_numbers)
        elif hasattr(uvd, "antenna_numbers"):
            ants = np.asarray(uvd.antenna_numbers)
        else:
            ants = np.array(sorted(set(map(int, uvd.ant_1_array)) | set(map(int, uvd.ant_2_array))))
        return antpos_enu, ants
    raise ValueError(f"Unexpected get_enu_antpos return type: {type(ret)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--uvh5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pol", default=None)
    args = p.parse_args()
    path = Path(args.uvh5)
    m = re.search(r"baseline\.(\d+)_(\d+)\.sum", path.name)
    if not m:
        raise ValueError(f"Cannot parse baseline pair from filename: {path.name}")
    ant1, ant2 = map(int, m.groups())
    print("reading:", path)
    uvd = UVData()
    uvd.read(str(path))
    pols = list(uvd.get_pols())
    pol = args.pol or ("ee" if "ee" in pols else pols[0])
    print("internal antpairs:", uvd.get_antpairs())
    print("available pols:", pols)
    print("using antpair/pol:", ant1, ant2, pol)
    data = uvd.get_data((ant1, ant2, pol))
    flags = uvd.get_flags((ant1, ant2, pol))
    nsamples = uvd.get_nsamples((ant1, ant2, pol))
    freqs_hz = np.ravel(uvd.freq_array).astype(float)
    blt_inds = np.where((uvd.ant_1_array == ant1) & (uvd.ant_2_array == ant2))[0]
    times_jd = uvd.time_array[blt_inds].astype(float)
    antpos_enu, ants = get_antpos_enu_and_ants(uvd)
    ant_to_pos = {int(a): antpos_enu[i] for i, a in enumerate(ants)}
    if ant1 not in ant_to_pos or ant2 not in ant_to_pos:
        raise ValueError(f"Could not find ENU positions for antennas {ant1}, {ant2}")
    ant1_enu = ant_to_pos[ant1]
    ant2_enu = ant_to_pos[ant2]
    baseline_enu_m = ant2_enu - ant1_enu
    weights_tf = (~flags).astype(float) * np.clip(nsamples.astype(float), 0.0, None)
    if np.nanmax(weights_tf) > 0:
        weights_tf = weights_tf / np.nanmax(weights_tf)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vis_tf=data.astype(np.complex64),
        freqs_hz=freqs_hz.astype(np.float64),
        times_jd=times_jd.astype(np.float64),
        baseline_enu_m=baseline_enu_m.astype(np.float64),
        ant1_enu_m=ant1_enu.astype(np.float64),
        ant2_enu_m=ant2_enu.astype(np.float64),
        flags_tf=flags.astype(bool),
        weights_tf=weights_tf.astype(np.float32),
        source_uvh5=str(path),
        ant1=int(ant1), ant2=int(ant2), pol=str(pol),
        processing_history=(
            "HERA H6C_IDR2 pspec/redavg-smoothcal-inpaint-500ns-lstcal "
            "single-baseline LST visibility product. This is a processed public "
            "background product, not raw correlator visibility. PathB operator P "
            "is applied on top of this product."
        ),
    )
    print("saved:", out)
    print("shape:", data.shape)
    print("freq range MHz:", freqs_hz.min()/1e6, freqs_hz.max()/1e6)
    print("Ntimes:", len(times_jd))
    print("baseline_enu_m:", baseline_enu_m)
    print("baseline_length_m:", float(np.linalg.norm(baseline_enu_m)))
    print("flag_fraction:", float(np.mean(flags)))
    print("weight_nonzero_fraction:", float(np.mean(weights_tf > 0)))

if __name__ == "__main__":
    main()
