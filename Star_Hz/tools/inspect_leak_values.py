#!/usr/bin/env python
"""Inspect per-snapshot leakage/distortion values for specific k indices.

Usage:
  python tools\inspect_leak_values.py <pickle_path> [n_slices]

Prints dtype, sample unique values, min/max/mean and counts >0 for selected ks.
"""
import sys
import numpy as np
import pandas as pd


def _get_2d_from_array(arr: np.ndarray) -> np.ndarray:
    X = np.asarray(arr)
    X = np.squeeze(X)
    while X.ndim > 2:
        X = X[..., 0]
        X = np.squeeze(X)
    return X.astype(np.float32, copy=False)


def extract_slices_from_payload(payload, n_slices: int | None = None):
    if n_slices is None or n_slices <= 0:
        n_slices = None
    if isinstance(payload, np.ndarray):
        if payload.ndim >= 3:
            T = payload.shape[0]
            use = T if n_slices is None else min(T, n_slices)
            return [ _get_2d_from_array(payload[t]) for t in range(use) ]
        return [_get_2d_from_array(payload)]
    if isinstance(payload, (list, tuple)) and len(payload) > 0:
        first = payload[0]
        if isinstance(first, np.ndarray) and first.ndim >= 3:
            T = first.shape[0]
            use = T if n_slices is None else min(T, n_slices)
            return [ _get_2d_from_array(first[t]) for t in range(use) ]
        try:
            return [_get_2d_from_array(first)]
        except Exception:
            pass
    return [_get_2d_from_array(np.asarray(payload))]


def run_rank_sweep_real(X: np.ndarray, max_k: int = 25):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    max_k = int(max(1, min(max_k, min(X.shape))))
    ks = np.arange(1, max_k + 1, dtype=int)

    leakage_proxy = np.zeros_like(ks, dtype=np.float64)
    distortion_proxy = np.zeros_like(ks, dtype=np.float64)

    norm_X = float(np.linalg.norm(X, ord='fro'))

    for i, k in enumerate(ks):
        Lk = (U[:, :k] * s[:k][None, :]) @ Vt[:k, :]
        Ek = X - Lk

        sigma = float(np.std(Ek))
        thr = 5.0 * sigma
        frac_out = float(np.mean(np.abs(Ek) > thr))

        preservation = np.linalg.norm(Ek, ord='fro') / (norm_X + 1e-12)
        preservation = float(min(1.0, max(0.0, preservation)))
        distortion = 1.0 - preservation

        leakage_proxy[i] = frac_out
        distortion_proxy[i] = distortion

    return ks, leakage_proxy, distortion_proxy


def summarize(arr, name):
    arr = np.asarray(arr, dtype=float)
    print(f"--- {name} ---")
    print("dtype:", arr.dtype)
    print("count:", arr.size)
    if arr.size == 0:
        return
    print("unique(first10):", np.unique(arr[:10]))
    print("min, max, mean:", np.min(arr), np.max(arr), np.mean(arr))
    print("count>0:", int(np.sum(arr > 0)))


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools\\inspect_leak_values.py <pickle_path> [n_slices]")
        sys.exit(1)
    path = sys.argv[1]
    n_slices = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    payload = pd.read_pickle(path)
    slices = extract_slices_from_payload(payload, n_slices=n_slices)
    print(f"Found {len(slices)} slices (using n_slices={n_slices})")

    K = 50
    ks_vals = {1: [], 2: [], 3: [], 50: []}

    for X in slices:
        ks, leak, dist = run_rank_sweep_real(X, max_k=K)
        for k_idx in [1,2,3,50]:
            if k_idx <= len(ks):
                ks_vals[k_idx].append(leak[k_idx-1])

    for k_idx in [1,2,3,50]:
        summarize(ks_vals[k_idx], f"leakage k={k_idx}")


if __name__ == '__main__':
    main()
