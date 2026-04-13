"""
Plot hysteresis curves and max-hysteresis stability tracking.

Reproduces from the notebooks:
  1. Hysteresis curves (I_fwd - I_bwd vs Vsd) for each experiment set
  2. Max positive/negative hysteresis vs experiment index (endurance)

Uses both raw_hysteresis and poly_corrected_hysteresis methods.

Usage:
    python scripts/plot_hysteresis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.load_data import load_all_raw, load_calibrations, filter_experiments
from scripts.plot_style import apply_default_style
from src.derived.algorithms.hysteresis import (
    compute_raw_hysteresis,
    extract_raw_hysteresis_maxima,
)

apply_default_style()

OUTDIR = Path("figures")


def plot_hysteresis_curves(
    props,
    data,
    *,
    cmap_name: str = "magma",
    cal_fn: interp1d | None = None,
    area_factor: float = 1.0,
    legend_title: str = "",
    outpath: Path | None = None,
    show: bool = False,
):
    """Plot overlaid hysteresis difference curves (I_fwd - I_bwd)."""
    n = props.height
    if n == 0:
        return

    colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, n))

    fig, ax = plt.subplots()
    for i in range(n):
        row = props.row(i, named=True)
        key = row["data_key"]
        if key not in data:
            continue

        df = data[key]
        step = row.get("VSD step", 0.1)
        dv_threshold = step / 2

        hyst = compute_raw_hysteresis(df, dv_threshold=dv_threshold)
        if hyst.is_empty():
            continue

        v = hyst["Vsd (V)"].to_numpy()
        i_diff = hyst["I (A)"].to_numpy()

        laser_v = row.get("Laser voltage", 0.0)
        if cal_fn is not None and isinstance(laser_v, (int, float)):
            power_uW = float(cal_fn(laser_v)) * area_factor * 1e6
            label = f"{power_uW:.1f} \u00b5W"
        else:
            label = f"{laser_v:.1f} V"

        ax.plot(v, i_diff, color=colors[i], label=label)

    ax.set_xlabel(r"$V_{DS}$ (V)")
    ax.set_ylabel("Hysteresis [A]")

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 25:
        ax.legend(
            title=legend_title or None,
            frameon=True,
            fancybox=True,
            edgecolor="black",
            framealpha=0.9,
        )
    fig.tight_layout()

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300)
        print(f"  Saved -> {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_hysteresis_stability(
    props,
    data,
    *,
    outpath_pos: Path | None = None,
    outpath_neg: Path | None = None,
    show: bool = False,
):
    """
    Plot max positive and negative hysteresis vs experiment index.

    Reproduces the stability/endurance plots from Mayo.ipynb.
    """
    n = props.height
    if n == 0:
        return

    max_pos_list = []
    max_neg_list = []
    indices = []

    for i in range(n):
        row = props.row(i, named=True)
        key = row["data_key"]
        if key not in data:
            continue

        df = data[key]
        step = row.get("VSD step", 0.1)
        hyst = compute_raw_hysteresis(df, dv_threshold=step / 2)
        maxima = extract_raw_hysteresis_maxima(hyst)

        max_pos_list.append(maxima["max_positive_hysteresis"])
        max_neg_list.append(maxima["max_negative_hysteresis"])
        indices.append(i)

    indices = np.array(indices)
    max_pos = np.array(max_pos_list)
    max_neg = np.array(max_neg_list)

    # Positive hysteresis
    fig_pos, ax_pos = plt.subplots()
    ax_pos.plot(indices, max_pos, "o-")
    ax_pos.set_xlabel("Index")
    ax_pos.set_ylabel("Max Positive Hysteresis [A]")
    fig_pos.tight_layout()

    if outpath_pos:
        outpath_pos.parent.mkdir(parents=True, exist_ok=True)
        fig_pos.savefig(outpath_pos, dpi=300)
        print(f"  Saved -> {outpath_pos}")

    # Negative hysteresis
    fig_neg, ax_neg = plt.subplots()
    ax_neg.plot(indices, max_neg, "o-")
    ax_neg.set_xlabel("Index")
    ax_neg.set_ylabel("Max Negative Hysteresis [A]")
    fig_neg.tight_layout()

    if outpath_neg:
        outpath_neg.parent.mkdir(parents=True, exist_ok=True)
        fig_neg.savefig(outpath_neg, dpi=300)
        print(f"  Saved -> {outpath_neg}")

    if show:
        plt.show()
    else:
        plt.close(fig_pos)
        plt.close(fig_neg)


# ── Experiment definitions ──

STABILITY_EXPERIMENTS = [
    {
        "folder_prefix": "2025-05-06",
        "information": "SnS - Stability",
        "wavelength": 0.0,
        "label": "stability_no_light",
    },
    {
        "folder_prefix": "2025-05-06",
        "information": "SnS - Stability - white light",
        "wavelength": 0.0,
        "label": "stability_white_light",
    },
    {
        "folder_prefix": "2025-04-16",
        "information": "SnS - 10 um - No Light - 10 times",
        "wavelength": 0.0,
        "label": "10um_no_light_10x",
    },
]

HYSTERESIS_CURVE_EXPERIMENTS = [
    {
        "folder_prefix": "2025-05-15",
        "information": "SnS - 850 nm - up sequence",
        "wavelength": 850.0,
        "vsd_end": 10.0,
        "label": "850nm_up",
    },
    {
        "folder_prefix": "2025-05-15",
        "information": "SnS - 680 nm - UP sequence 2",
        "wavelength": 680.0,
        "vsd_end": 10.0,
        "label": "680nm_up",
    },
    {
        "folder_prefix": "2025-05-16",
        "information": "SnS - 590 nm - up sequence",
        "wavelength": 590.0,
        "vsd_end": 10.0,
        "label": "590nm_up",
    },
    {
        "folder_prefix": "505_up_down",
        "information": "SnS - 505 nm - up sequence",
        "wavelength": 505.0,
        "vsd_end": 10.0,
        "label": "505nm_up",
    },
    {
        "folder_prefix": "2025-05-29",
        "information": "SnS - 455 nm - power up 2",
        "wavelength": 455.0,
        "vsd_end": 10.0,
        "label": "455nm_up",
    },
    {
        "folder_prefix": "2025-05-29",
        "information": "SnS - 385 nm - power up",
        "wavelength": 385.0,
        "vsd_end": 10.0,
        "label": "385nm_up",
    },
]


SPOT_PROBE = np.pi * 179**2
DEVICE_PROBE = 121 * 100
FACTOR_PROBE = DEVICE_PROBE / SPOT_PROBE


def _build_cal_fn(cal_props, cal_data, date_prefix, wavelength):
    """Build a single calibration interp1d for a given date + wavelength."""
    filtered = cal_props.filter(
        (cal_props["Laser wavelength"] == wavelength)
        & (cal_props["data_key"].str.starts_with(date_prefix))
    )
    if filtered.is_empty():
        return None
    key = filtered["data_key"][0]
    df = cal_data[key]
    return interp1d(
        df["VL (V)"].to_numpy(),
        df["Power (W)"].to_numpy(),
        kind="linear",
        fill_value="extrapolate",
    )


def main():
    print("Loading all raw data...")
    props, data = load_all_raw()
    print(f"Loaded {props.height} experiments")

    print("Loading calibration data...")
    cal_props, cal_data = load_calibrations()
    print(f"Loaded {cal_props.height} calibration files\n")

    # 1. Hysteresis curves for wavelength-dependent experiments
    print("=== Hysteresis Curves ===")
    for exp in HYSTERESIS_CURVE_EXPERIMENTS:
        label = exp["label"]
        wl = exp.get("wavelength", 0.0)
        folder_prefix = exp.get("folder_prefix", "misc")

        filtered = filter_experiments(
            props,
            folder_prefix=folder_prefix,
            information=exp.get("information"),
            wavelength=wl,
            vsd_end=exp.get("vsd_end"),
        )
        print(f"[{label}] {filtered.height} sweeps")

        cal_fn = None
        legend_title = ""
        if wl > 0:
            cal_fn = _build_cal_fn(cal_props, cal_data, folder_prefix, wl)
            legend_title = f"{wl:.0f} nm"
            if cal_fn is None:
                print(f"  [warn] no calibration for {wl} nm on {folder_prefix}")

        plot_hysteresis_curves(
            filtered, data,
            cal_fn=cal_fn,
            area_factor=FACTOR_PROBE,
            legend_title=legend_title,
            outpath=OUTDIR / folder_prefix / f"hysteresis_{label}.png",
        )

    # 2. Stability tracking
    print("\n=== Hysteresis Stability ===")
    for exp in STABILITY_EXPERIMENTS:
        label = exp["label"]
        filtered = filter_experiments(
            props,
            folder_prefix=exp.get("folder_prefix"),
            information=exp.get("information"),
            wavelength=exp.get("wavelength"),
        )
        print(f"[{label}] {filtered.height} sweeps")

        folder = exp.get("folder_prefix", "misc")
        plot_hysteresis_stability(
            filtered, data,
            outpath_pos=OUTDIR / folder / f"max_pos_hyst_{label}.png",
            outpath_neg=OUTDIR / folder / f"max_neg_hyst_{label}.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
