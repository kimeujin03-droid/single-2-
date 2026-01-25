# hera_rank_sweep.py
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

# Headless backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

from svd_diagnostics import _fro_norm  # 이미 있으니까 재사용

# ----------------------------------------------------------------------
# Global style (논문용으로 약간만 다듬음)
# ----------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


def _extract_2d_matrix(obj) -> np.ndarray:
    """Convert common HERA pickle payloads into a 2D (time,freq) float32 matrix.

    Supported payload shapes:
    - pandas DataFrame -> df.values
    - numpy ndarray with ndim>=2 -> squeeze to 2D by taking the first index along extra dims
    - list/tuple of arrays -> uses the first element

    This keeps the script usable across slightly different dump formats.
    """
    # DataFrame
    if hasattr(obj, "values") and not isinstance(obj, np.ndarray):
        X = np.asarray(obj.values)
    # list/tuple
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError("Pickle contained an empty list/tuple; cannot extract matrix")
        X = np.asarray(obj[0])
    else:
        X = np.asarray(obj)

    # Reduce to 2D
    X = np.squeeze(X)
    if X.ndim < 2:
        raise ValueError(f"Expected at least 2D array after squeeze, got shape={X.shape}")

    # If it's more than 2D (e.g., (time, freq, freq, pol)), take the first slice along extra dims.
    while X.ndim > 2:
        X = X[..., 0]
        X = np.squeeze(X)

    if X.ndim != 2:
        raise ValueError(f"Could not reduce payload to 2D matrix; final shape={X.shape}")

    return X.astype(np.float32, copy=False)


def _get_2d_from_array(arr: np.ndarray) -> np.ndarray:
    """Reduce an ndarray to a 2D matrix by squeezing extra dims and
    taking the first index along trailing dimensions when needed.

    This is like _extract_2d_matrix but operates on an already-loaded
    ndarray representing one snapshot (or one frame).
    """
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
    """Return a list of 2D matrices extracted from the pickle payload.

    If the payload contains a time axis (ndim>=3), treat the first axis as
    snapshots and return up to n_slices frames. If payload is a list/tuple,
    try to use the first array in it similarly. If payload is already 2D,
    return it as a single-element list.
    """
    if n_slices is None or n_slices <= 0:
        n_slices = None

    # Prefer raw ndarray payloads
    if isinstance(payload, np.ndarray):
        if payload.ndim >= 3:
            T = payload.shape[0]
            use = T if n_slices is None else min(T, n_slices)
            slices = [ _get_2d_from_array(payload[t]) for t in range(use) ]
            return slices
        else:
            return [_get_2d_from_array(payload)]

    # If a list/tuple, inspect first element
    if isinstance(payload, (list, tuple)) and len(payload) > 0:
        first = payload[0]
        if isinstance(first, np.ndarray) and first.ndim >= 3:
            T = first.shape[0]
            use = T if n_slices is None else min(T, n_slices)
            slices = [ _get_2d_from_array(first[t]) for t in range(use) ]
            return slices
        # Otherwise try to reduce first element to 2D
        try:
            return [_get_2d_from_array(first)]
        except Exception:
            pass

    # Fallback: try to convert payload to a single 2D array
    try:
        return [_get_2d_from_array(np.asarray(payload))]
    except Exception as e:
        raise ValueError(f"Unable to extract slices from payload: {e}")


def load_hera_matrix(path: str) -> np.ndarray:
    """
    HERA_04-03-2022_all.pkl에서 하나의 time–frequency 스냅샷 X(T,F) 뽑아오기.
    실제 구조는 데이터 열 이름에 따라 조정 필요.
    """
    payload = pd.read_pickle(path)
    return _extract_2d_matrix(payload)


def run_rank_sweep_real(X: np.ndarray, max_k: int = 25):
    """실제 HERA 스냅샷에 대해 rank sweep 수행하고 proxy 두 개 계산."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    max_k = int(max(1, min(max_k, min(X.shape))))
    ks = np.arange(1, max_k + 1, dtype=int)

    # proxy들 – ground truth가 없으니 대리 지표만 사용
    leakage_proxy = np.zeros_like(ks, dtype=np.float64)
    distortion_proxy = np.zeros_like(ks, dtype=np.float64)

    norm_X = _fro_norm(X)

    for i, k in enumerate(ks):
        Lk = (U[:, :k] * s[:k][None, :]) @ Vt[:k, :]
        Ek = X - Lk

        # 예시 1: narrowband outlier fraction을 leakage proxy로 사용
        sigma = float(np.std(Ek))
        thr = 5.0 * sigma
        frac_out = float(np.mean(np.abs(Ek) > thr))
        # Preservation proxy: fraction of total Frobenius norm remaining in the
        # cleaned/residual Ek. We then define distortion = 1 - preservation so
        # that distortion -> 0 indicates good preservation of the science/sky
        # structure (smaller is better).
        preservation = _fro_norm(Ek) / (norm_X + 1e-12)
        # Clamp preservation to [0, 1] to avoid small numerical overshoots
        preservation = float(min(1.0, max(0.0, preservation)))
        distortion = 1.0 - preservation

        leakage_proxy[i] = frac_out
        distortion_proxy[i] = distortion

    return ks, leakage_proxy, distortion_proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True)
    ap.add_argument("--max_k", type=int, default=25)
    ap.add_argument("--aggregate", action="store_true", help="Aggregate proxies across multiple snapshots (compute median+16-84% bands)")
    ap.add_argument("--n_slices", type=int, default=50, help="Max number of snapshots to aggregate (if available)")
    ap.add_argument("--p_low", type=float, default=16.0, help="Lower percentile for band (default 16)")
    ap.add_argument("--p_high", type=float, default=84.0, help="Upper percentile for band (default 84)")
    ap.add_argument("--outdir", type=str, default="hera_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    payload = pd.read_pickle(args.path)

    if bool(args.aggregate):
        slices = extract_slices_from_payload(payload, n_slices=int(args.n_slices))
        if len(slices) == 0:
            raise RuntimeError("No usable slices found for aggregation")

        # Determine global max_k that fits all slices (cap by min dimension across slices)
        min_shape = min(min(s.shape) for s in slices)
        global_max_k = int(max(1, min(int(args.max_k), int(min_shape))))
        ks = np.arange(1, global_max_k + 1, dtype=int)

        all_leaks = []
        all_dists = []
        for Xs in slices:
            ks_s, leak_s, dist_s = run_rank_sweep_real(Xs, max_k=global_max_k)
            # run_rank_sweep_real returns arrays of appropriate length
            all_leaks.append(leak_s)
            all_dists.append(dist_s)

        # Build per-k lists of snapshot values and compute robust stats per-k.
        # This avoids surprises if some slices have differing lengths or if
        # there are empty entries. It mirrors the pattern you suggested:
        # vals = np.array(leak_list_for_k); handle empty -> nan
        p_low = float(args.p_low)
        p_high = float(args.p_high)

        K = len(ks)
        leak_lists = [[] for _ in range(K)]
        dist_lists = [[] for _ in range(K)]
        for row in all_leaks:
            # row is expected to be length K; skip rows with wrong shape
            if len(row) != K:
                continue
            for j in range(K):
                # force float to avoid accidental integer casting later
                try:
                    leak_lists[j].append(float(row[j]))
                except Exception:
                    leak_lists[j].append(float(np.nan))
        for row in all_dists:
            if len(row) != K:
                continue
            for j in range(K):
                try:
                    dist_lists[j].append(float(row[j]))
                except Exception:
                    dist_lists[j].append(float(np.nan))

        leak_median = np.zeros(K, dtype=float)
        leak_low = np.zeros(K, dtype=float)
        leak_high = np.zeros(K, dtype=float)
        dist_median = np.zeros(K, dtype=float)
        dist_low = np.zeros(K, dtype=float)
        dist_high = np.zeros(K, dtype=float)

        # Diagnostic prints for the first few k to check dtypes and sample values
        for j in range(min(3, K)):
            if len(leak_lists[j]) > 0:
                v0 = leak_lists[j][0]
                arr = np.asarray(leak_lists[j])
                print(f"[DEBUG] k={j+1}: type(vals[0])={type(v0)}, np.asarray(vals).dtype={arr.dtype}, unique(first10)={np.unique(arr[:10])}")
            else:
                print(f"[DEBUG] k={j+1}: n=0 (no values)")

        for j in range(K):
            vals = np.array(leak_lists[j], dtype=float)
            if vals.size == 0:
                leak_median[j] = np.nan
                leak_low[j] = np.nan
                leak_high[j] = np.nan
            else:
                leak_median[j] = float(np.nanmedian(vals))
                leak_low[j] = float(np.nanpercentile(vals, p_low))
                leak_high[j] = float(np.nanpercentile(vals, p_high))

            v2 = np.array(dist_lists[j], dtype=float)
            if v2.size == 0:
                dist_median[j] = np.nan
                dist_low[j] = np.nan
                dist_high[j] = np.nan
            else:
                dist_median[j] = float(np.nanmedian(v2))
                dist_low[j] = float(np.nanpercentile(v2, p_low))
                dist_high[j] = float(np.nanpercentile(v2, p_high))

        # Save aggregated CSV
        csv_path = os.path.join(args.outdir, "hera_rank_sweep_aggregated.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("k,leak_median,leak_low,leak_high,dist_median,dist_low,dist_high\n")
            for k, lm, ll, lh, dm, dl, dh in zip(ks, leak_median, leak_low, leak_high, dist_median, dist_low, dist_high):
                f.write(f"{int(k)},{float(lm):.8e},{float(ll):.8e},{float(lh):.8e},{float(dm):.8e},{float(dl):.8e},{float(dh):.8e}\n")

        # Use median curves for plotting below
        leak = leak_median
        dist = dist_median
        agg_bands = {
            "leak_low": leak_low,
            "leak_high": leak_high,
            "dist_low": dist_low,
            "dist_high": dist_high,
        }

    else:
        X = _extract_2d_matrix(payload)
        ks, leak, dist = run_rank_sweep_real(X, max_k=args.max_k)
        agg_bands = None

    # ------------------------------------------------------------------
    # CSV 저장
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.outdir, "hera_rank_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("k,leakage_proxy,distortion_proxy\n")
        for k, a, b in zip(ks, leak, dist):
            f.write(f"{int(k)},{float(a):.8e},{float(b):.8e}\n")

    # ------------------------------------------------------------------
    # Figure 3: rank sweep plot with dual y-axes
    # (left: leakage, right: distortion)
    # ------------------------------------------------------------------
    # Use constrained_layout for journal-friendly spacing (esp. with legends outside axes)
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    ax2 = ax1.twinx()

    # 비식별(non-identifiable) rank 범위를 강조 (예: k=5~20)
    non_id_low, non_id_high = 5, 20
    ax1.axvspan(non_id_low, non_id_high, color="0.85", alpha=0.35, zorder=0)

    # If aggregation produced percentile bands, plot them as filled regions
    if agg_bands is not None:
        ax1.fill_between(ks, agg_bands["leak_low"], agg_bands["leak_high"], color="tab:blue", alpha=0.18)
        ax2.fill_between(ks, agg_bands["dist_low"], agg_bands["dist_high"], color="tab:orange", alpha=0.18)

    # 곡선들
    l1 = ax1.plot(
        ks,
        leak,
        color="tab:blue",
        lw=1.8,
        marker="o",
        ms=3.0,
        label=("RFI leakage proxy (median)" if agg_bands is not None else "RFI leakage proxy"),
    )
    l2 = ax2.plot(
        ks,
        dist,
        color="tab:orange",
        lw=1.8,
        marker="o",
        ms=3.0,
        label=("Science distortion proxy (median)" if agg_bands is not None else "Science distortion proxy"),
    )

    # 축 라벨/범위
    ax1.set_xlabel("Rank k")
    ax1.set_ylabel("RFI leakage proxy", color="tab:blue")
    ax2.set_ylabel("Science distortion proxy", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # x축은 정수 tick (rank)
    ax1.set_xlim(ks[0], ks[-1])
    # Reduce tick clutter: show every other k (still clearly conveys "no knee")
    ax1.set_xticks(ks[::2])

    # 과학적 표기 (한 줄짜리 ×10^n offset)
    fmt1 = ScalarFormatter(useMathText=True)
    fmt1.set_scientific(True)
    fmt1.set_powerlimits((0, 0))  # 항상 ×10^n 사용
    ax1.yaxis.set_major_formatter(fmt1)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fmt2 = ScalarFormatter(useMathText=True)
    fmt2.set_scientific(True)
    fmt2.set_powerlimits((0, 0))
    ax2.yaxis.set_major_formatter(fmt2)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax1.grid(True, which="both", ls=":", lw=0.7, alpha=0.6)

    # 비식별 구간 라벨 (데이터 플롯 후 ylim 기준으로 배치)
    y1_min, y1_max = ax1.get_ylim()
    # Place the annotation centered in the shaded band (interpretive comment, not competing with data)
    y_text = y1_min + 0.88 * (y1_max - y1_min)
    x_center = 0.5 * (non_id_low + non_id_high)
    ax1.text(
        x_center,
        y_text,
        "Non-identifiable rank range (k≈5–20)",
        ha="center",
        va="top",
        fontsize=9,
        color="0.45",
        alpha=0.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 2.0},
    )

    # Single combined legend
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    dummy = plt.Line2D(
        [], [], color="none", label="Proxy definitions are given in Section 5.4"
    )
    # Put legend outside the axes (journal-friendly; avoids occluding data)
    ax1.legend(
        lines + [dummy],
        labels + [dummy.get_label()],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.suptitle("Figure 3. HERA rank sweep", y=1.02)

    # Nudge scientific-offset text so it doesn't visually compete with the outside legend/title.
    # (These are the "×10^n" annotations produced by ScalarFormatter.)
    try:
        ax1.yaxis.get_offset_text().set_x(-0.08)
        ax1.yaxis.get_offset_text().set_y(1.02)
        ax1.yaxis.get_offset_text().set_fontsize(8.5)

        ax2.yaxis.get_offset_text().set_x(1.08)
        ax2.yaxis.get_offset_text().set_y(1.02)
        ax2.yaxis.get_offset_text().set_fontsize(8.5)
    except Exception:
        # If a backend/formatter doesn't create offset text, just skip.
        pass
    fig_path = os.path.join(args.outdir, "figure3_hera_rank_sweep.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Supplementary Pareto plot (HERA Pareto proxy space)
    # ------------------------------------------------------------------
    plt.figure(figsize=(4.2, 3.6))
    plt.plot(leak, dist, marker="o", lw=1.2)
    for i, k in enumerate(ks):
        # 너무 촘촘하지 않게 일부 rank만 라벨
        if i % max(1, len(ks) // 10) == 0:
            plt.text(float(leak[i]), float(dist[i]), str(int(k)), fontsize=7)

    plt.xlabel("RFI leakage proxy")
    plt.ylabel("Science distortion proxy")
    plt.tight_layout()
    pareto_path = os.path.join(args.outdir, "hera_pareto.png")
    plt.savefig(pareto_path, dpi=180)
    plt.close()

    print(f"[HERA] done. CSV/plots written to: {args.outdir}")
    print(f"  - Rank sweep figure: {fig_path}")
    print(f"  - Pareto figure:     {pareto_path}")


if __name__ == "__main__":
    main()
