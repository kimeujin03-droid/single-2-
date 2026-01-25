# make_figure1_synthetic_failure_modes.py (patched)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 150,
    }
)

EPS = 1e-12  # for log safety


@dataclass(frozen=True)
class Scenario:
    title: str
    tag: str


def read_sweep_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a CSV produced by svd_diagnostics.py rank sweep.

    Supported column sets:
      A) k, bias_under, bias_over
      B) k, leak_median, dist_median  (use medians as the curve)
    """
    data = np.genfromtxt(path, delimiter=",", names=True)

    k = data["k"].astype(int)

    names = set(data.dtype.names or [])
    if {"bias_under", "bias_over"} <= names:
        under = data["bias_under"].astype(float)
        over = data["bias_over"].astype(float)
    elif {"leak_median", "dist_median"} <= names:
        under = data["leak_median"].astype(float)
        over = data["dist_median"].astype(float)
    else:
        raise ValueError(f"Unrecognized columns in {path}: {data.dtype.names}")

    # log-safe clamp
    under = np.where(under > 0, under, EPS)
    over = np.where(over > 0, over, EPS)
    return k, under, over


def pareto_front_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return indices of the Pareto-efficient set for minimization in (x,y).
    A point i is efficient if no other point has x<=xi and y<=yi with at least one strict.
    """
    idx = np.argsort(x)  # sort by x ascending
    x_s = x[idx]
    y_s = y[idx]

    best_y = np.inf
    keep = []
    for j in range(len(idx)):
        if y_s[j] < best_y:  # strict improvement in y as x increases
            keep.append(idx[j])
            best_y = y_s[j]

    keep = np.array(keep, dtype=int)
    # sort kept points by x for plotting
    keep = keep[np.argsort(x[keep])]
    return keep


def best_k(k: np.ndarray, under: np.ndarray, over: np.ndarray) -> int:
    """
    Select k* by minimizing a balanced scalar score in log-space:
      score = log10(under) + log10(over)
    This avoids one metric dominating by raw scale.
    """
    score = np.log10(under) + np.log10(over)
    return int(k[int(np.argmin(score))])


def main() -> None:
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")

    scenarios = [
        Scenario("(a) Simple comb", "fig1_simple_comb"),
        Scenario("(b) Complex comb + ripple", "fig1_complex_comb_ripple"),
        Scenario("(c) Comb drift (core failure)", "fig1_comb_drift"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))

    loaded = []
    for sc in scenarios:
        csv_path = os.path.join(outdir, f"svd_rank_sweep_{sc.tag}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV for scenario {sc.tag}: {csv_path}")
        k, under, over = read_sweep_csv(csv_path)
        loaded.append((sc, k, under, over))

    all_under = np.concatenate([u for _sc, _k, u, _o in loaded])
    all_over = np.concatenate([o for _sc, _k, _u, o in loaded])

    # robust axis ranges (avoid single outlier blowing up)
    xmin = float(np.percentile(all_under, 1)) * 0.85
    xmax = float(np.percentile(all_under, 99)) * 1.15
    ymin = float(np.percentile(all_over, 1)) * 0.85
    ymax = float(np.percentile(all_over, 99)) * 1.15

    styles = {
        "fig1_simple_comb": {"color": "0.35", "marker": "o"},
        "fig1_complex_comb_ripple": {"color": "teal", "marker": "s"},
        "fig1_comb_drift": {"color": "tab:blue", "marker": "o"},
    }

    for ax, (sc, k, under, over) in zip(axes, loaded):
        st = styles.get(sc.tag, {"color": "tab:blue", "marker": "o"})

        # plot Pareto-front only (cleaner & matches claim "Pareto geometry")
        pidx = pareto_front_indices(under, over)
        ax.plot(
            under[pidx],
            over[pidx],
            marker=st["marker"],
            ms=4.0,
            lw=1.6,
            color=st["color"],
            alpha=0.95,
        )

        # Optional: faintly show all k-trajectory for context
        ax.plot(
            under,
            over,
            lw=0.6,
            color=st["color"],
            alpha=0.25,
        )

        if sc.tag == "fig1_comb_drift":
            k_star = best_k(k, under, over)
            i_star = int(np.where(k == k_star)[0][0])

            ax.scatter(
                [under[i_star]],
                [over[i_star]],
                s=120,
                marker="*",
                color="tab:blue",
                edgecolor="k",
                linewidths=0.5,
                zorder=6,
            )
            ax.annotate(
                f"k* = {k_star}",
                (under[i_star], over[i_star]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )

            ax.text(
                0.58,
                0.78,
                "Comb-drift–dominated\n(non-identifiable regime)",
                transform=ax.transAxes,
                fontsize=9,
                color="tab:blue",
                alpha=0.9,
                ha="right",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
            )

        ax.set_title(sc.title)
        ax.grid(True, which="both", ls=":", lw=0.7, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(max(xmin, EPS), max(xmax, EPS))
        ax.set_ylim(max(ymin, EPS), max(ymax, EPS))

    axes[0].set_ylabel("Over-subtraction bias\n(science loss metric, log scale)")
    for ax in axes:
        ax.set_xlabel("Under-subtraction bias (RFI leakage, log scale)")

    fig.suptitle("Figure 1. Synthetic failure-mode overview (Pareto trade-off)", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(outdir, "figure1_synthetic_failure_modes.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIG1] wrote: {out_path}")


if __name__ == "__main__":
    main()
