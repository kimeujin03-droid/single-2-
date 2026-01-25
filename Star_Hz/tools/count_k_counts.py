#!/usr/bin/env python
"""Count how many snapshot values contributed to each k in the aggregated run.

Usage:
  python tools\count_k_counts.py <path_to_pickle> [n_slices]

Prints lines like: k=1: n=420

This is a small standalone script that uses local copies of the
helper functions from `hera_rank_sweep.py` to avoid import path issues.
"""
import sys
import pandas as pd
import numpy as np


def _get_2d_from_array(arr: np.ndarray) -> np.ndarray:
    X = np.asarray(arr)
    X = np.squeeze(X)
    if X.ndim < 2:
        raise ValueError(f"Expected at least 2D array after squeeze, got shape={X.shape}")
    while X.ndim > 2:
        X = X[..., 0]
        X = np.squeeze(X)
    if X.ndim != 2:
        raise ValueError(f"Could not reduce payload to 2D matrix; final shape={X.shape}")
    return X.astype(np.float32, copy=False)


def extract_slices_from_payload(payload, n_slices: int | None = None):
    if n_slices is None or n_slices <= 0:
        n_slices = None

    if isinstance(payload, np.ndarray):
        if payload.ndim >= 3:
            T = payload.shape[0]
            use = T if n_slices is None else min(T, n_slices)
            slices = [ _get_2d_from_array(payload[t]) for t in range(use) ]
            return slices
        else:
            return [_get_2d_from_array(payload)]

    if isinstance(payload, (list, tuple)) and len(payload) > 0:
        first = payload[0]
        if isinstance(first, np.ndarray) and first.ndim >= 3:
            T = first.shape[0]
            use = T if n_slices is None else min(T, n_slices)
            slices = [ _get_2d_from_array(first[t]) for t in range(use) ]
            return slices
        try:
            return [_get_2d_from_array(first)]
        except Exception:
            pass

    try:
        return [_get_2d_from_array(np.asarray(payload))]
    except Exception as e:
        raise ValueError(f"Unable to extract slices from payload: {e}")


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools\\count_k_counts.py <path_to_pickle> [n_slices]")
        sys.exit(1)
    path = sys.argv[1]
    n_slices = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    payload = pd.read_pickle(path)
    slices = extract_slices_from_payload(payload, n_slices=n_slices)
    if len(slices) == 0:
        print("No slices extracted.")
        return

    min_shape = min(min(s.shape) for s in slices)
    K = min(50, min_shape)

    counts = [0] * K
    for X in slices:
        ks, leak, dist = run_rank_sweep_real(X, max_k=K)
        L = len(ks)
        for j in range(L):
            counts[j] += 1

    for idx, c in enumerate(counts):
        print(f"k={idx+1}: n={c}")


if __name__ == '__main__':
    main()
