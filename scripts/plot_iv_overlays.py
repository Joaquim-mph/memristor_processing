"""
Plot overlaid IV curves colored by laser voltage or run index.

Reproduces the raw IV plots from Mayo.ipynb and wl_depen.ipynb:
  - Multiple IV sweeps on one axis
  - Colored by laser voltage (viridis/magma colormap)
  - Current in uA

Usage:
    python scripts/plot_iv_overlays.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.load_data import load_all_raw, load_calibrations, filter_experiments
from scripts.plot_style import apply_default_style

apply_default_style()

OUTDIR = Path("figures")


def plot_iv_overlay(
    props,
    data,
    *,
    color_by: str = "Laser voltage",
    cmap_name: str = "magma",
    y_scale: float = 1e6,
    y_label: str = r"$I_{DS}$ ($\mu$A)",
    x_label: str = r"$V_{DS}$ (V)",
    title: str = "",
    cal_fn: interp1d | None = None,
    area_factor: float = 1.0,
    legend_title: str = "",
    outpath: Path | None = None,
    show: bool = False,
):
    """Plot overlaid IV curves from filtered props/data.

    Parameters
    ----------
    cal_fn : interp1d, optional
        Calibration function mapping laser voltage to optical power.
        When provided, legend labels show power instead of voltage.
    area_factor : float
        Device-area / spot-area ratio applied to calibrated power.
    legend_title : str
        Title shown above the legend entries (e.g. wavelength).
    """
    n = props.height
    if n == 0:
        print(f"  [skip] No experiments for: {title}")
        return

    colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, n))

    fig, ax = plt.subplots()
    for i in range(n):
        row = props.row(i, named=True)
        key = row["data_key"]
        if key not in data:
            continue

        df = data[key]
        v = df["Vsd (V)"].to_numpy()
        current = df["I (A)"].to_numpy() * y_scale

        laser_v = row.get(color_by, i)
        if cal_fn is not None and isinstance(laser_v, (int, float)):
            power_uW = float(cal_fn(laser_v)) * area_factor * 1e6
            label = f"{power_uW:.1f} \u00b5W"
        elif isinstance(laser_v, float):
            label = f"{laser_v:.1f} V"
        else:
            label = str(laser_v)

        ax.plot(v, current, color=colors[i], label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

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


# ── Experiment definitions (matching the notebooks) ──

EXPERIMENTS = [
    # Mayo.ipynb stability experiments
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
    # Wavelength-dependent power-up sequences (wl_depen.ipynb)
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
    # No-light repeatability (no_light_statistic.ipynb)
    {
        "folder_prefix": "2025-06-26",
        "information": "SnS - no light - PAIN",
        "wavelength": 385.0,
        "vsd_end": 8.0,
        "label": "PAIN_no_light",
    },
]


SPOT_PROBE = np.pi * 179**2
DEVICE_PROBE = 121 * 100
FACTOR_PROBE = DEVICE_PROBE / SPOT_PROBE

SPOT_WIREBOND = np.pi * 190**2
DEVICE_WIREBOND = 61 * 114
FACTOR_WIREBOND = DEVICE_WIREBOND / SPOT_WIREBOND


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

    for exp in EXPERIMENTS:
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

        # Build calibration if this is a light experiment
        cal_fn = None
        area_factor = 1.0
        legend_title = ""
        if wl > 0:
            cal_fn = _build_cal_fn(cal_props, cal_data, folder_prefix, wl)
            # Pick area factor based on date
            if folder_prefix >= "2025-06-20":
                area_factor = FACTOR_WIREBOND
            else:
                area_factor = FACTOR_PROBE
            legend_title = f"{wl:.0f} nm"
            if cal_fn is None:
                print(f"  [warn] no calibration for {wl} nm on {folder_prefix}")

        outpath = OUTDIR / folder_prefix / f"IV_{label}.png"

        plot_iv_overlay(
            filtered,
            data,
            title="",
            cal_fn=cal_fn,
            area_factor=area_factor,
            legend_title=legend_title,
            outpath=outpath,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
