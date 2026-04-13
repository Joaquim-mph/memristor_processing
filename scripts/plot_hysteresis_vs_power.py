"""
Plot max hysteresis vs LED optical power for wavelength-dependent experiments.

Reproduces the overlay_hyst_vs_power_multi plots from wl_depen.ipynb:
  - Build per-wavelength calibration functions (laser voltage -> optical power)
  - Apply area factor correction (device area / LED spot area)
  - Compute raw hysteresis max for each IV sweep
  - Plot max positive hysteresis vs effective power, one curve per wavelength

Two sample geometries:
  - Probe-contacted: spot = pi*179^2, device = 121*100  -> factor ~ 0.120
  - Wire-bonded:     spot = pi*190^2, device = 61*114   -> factor ~ 0.061

Usage:
    python scripts/plot_hysteresis_vs_power.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.load_data import (
    load_all_raw,
    load_calibrations,
    filter_experiments,
)
from scripts.plot_style import apply_default_style
from src.derived.algorithms.hysteresis import (
    compute_raw_hysteresis,
    extract_raw_hysteresis_maxima,
)

apply_default_style()

OUTDIR = Path("figures")

# ── Area factors ──

SPOT_PROBE = np.pi * 179**2
DEVICE_PROBE = 121 * 100
FACTOR_PROBE = DEVICE_PROBE / SPOT_PROBE  # ~0.120

SPOT_WIREBOND = np.pi * 190**2
DEVICE_WIREBOND = 61 * 114
FACTOR_WIREBOND = DEVICE_WIREBOND / SPOT_WIREBOND  # ~0.061


# ── Calibration helpers ──


def build_calibration_functions(
    cal_props,
    cal_data,
    date_prefix: str,
    wavelengths: list[float],
) -> dict[float, interp1d]:
    """
    Build linear interpolation functions mapping laser voltage to optical power.

    Filters calibration files by date_prefix and wavelength, returns one
    interp1d per wavelength.
    """
    calibration: dict[float, interp1d] = {}

    for wl in wavelengths:
        filtered = cal_props.filter(
            (cal_props["Laser wavelength"] == wl)
            & (cal_props["data_key"].str.starts_with(date_prefix))
        )
        if filtered.is_empty():
            continue

        key = filtered["data_key"][0]
        df = cal_data[key]
        vl = df["VL (V)"].to_numpy()
        pw = df["Power (W)"].to_numpy()

        f = interp1d(vl, pw, kind="linear", fill_value="extrapolate")
        calibration[wl] = f

    return calibration


# ── Hysteresis-vs-power plotting ──


def overlay_hyst_vs_power(
    experiments: list[dict],
    props,
    data,
    cal_props,
    cal_data,
    *,
    area_factor: float,
    hyst_key: str = "max_positive_hysteresis",
    baseline: bool = False,
    outpath: Path | None = None,
    show: bool = False,
):
    """
    Overlay max hysteresis vs LED power for multiple wavelength experiments.

    For each experiment:
      1. Build calibration function for that wavelength + date
      2. Filter IV sweeps, compute raw hysteresis maxima for each
      3. Convert laser voltage to effective power (calibration * area_factor)
      4. Plot hysteresis vs power
    """
    fig, ax = plt.subplots()

    for exp in experiments:
        dp = exp["folder_prefix"]
        wl = exp["wavelength"]
        info = exp["information"]
        vsd_end = exp.get("vsd_end")

        # Build calibrator
        cals = build_calibration_functions(
            cal_props, cal_data,
            date_prefix=dp,
            wavelengths=[wl],
        )
        if wl not in cals:
            print(f"  [warn] no calibration for {wl} nm on {dp}, skipping")
            continue

        cal_fn = cals[wl]

        # Filter IV sweeps
        filtered = filter_experiments(
            props,
            folder_prefix=dp,
            information=info,
            wavelength=wl,
            vsd_end=vsd_end,
        )
        if filtered.height == 0:
            print(f"  [warn] no IV data for {info}, skipping")
            continue

        # Compute hysteresis max for each sweep + map voltage to power
        powers = []
        hyst_vals = []

        for i in range(filtered.height):
            row = filtered.row(i, named=True)
            key = row["data_key"]
            if key not in data:
                continue

            df = data[key]
            step = row.get("VSD step", 0.1)
            hyst = compute_raw_hysteresis(df, dv_threshold=step / 2)
            maxima = extract_raw_hysteresis_maxima(hyst)

            laser_v = row["Laser voltage"]
            power = float(cal_fn(laser_v)) * area_factor
            powers.append(power)
            hyst_vals.append(maxima[hyst_key])

        if not powers:
            continue

        x = np.array(powers)
        y = np.array(hyst_vals)

        if baseline and len(y) > 0:
            y = y - y[0]

        ax.plot(x, y, "o-", label=f"{wl:.0f} nm")

    ax.set_xlabel("Effective Power [W]")
    ylabel = (
        r"$\Delta$ Max Positive Hysteresis [A]"
        if baseline
        else "Max Positive Hysteresis [A]"
    )
    ax.set_ylabel(ylabel)
    ax.legend(title="Wavelength")
    fig.tight_layout()

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300)
        print(f"  Saved -> {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Experiment definitions (from wl_depen.ipynb) ──

# Probe-contacted sample (original device, factor ~ 0.120)
EXPTS_UP = [
    dict(folder_prefix="2025-05-15", wavelength=850.0, information="SnS - 850 nm - up sequence", vsd_end=10.0),
    dict(folder_prefix="2025-05-15", wavelength=680.0, information="SnS - 680 nm - UP sequence 2", vsd_end=10.0),
    dict(folder_prefix="2025-05-16", wavelength=590.0, information="SnS - 590 nm - up sequence", vsd_end=10.0),
    dict(folder_prefix="505_up_down", wavelength=505.0, information="SnS - 505 nm - up sequence", vsd_end=10.0),
    dict(folder_prefix="2025-05-29", wavelength=455.0, information="SnS - 455 nm - power up 2", vsd_end=10.0),
    dict(folder_prefix="2025-05-29", wavelength=385.0, information="SnS - 385 nm - power up", vsd_end=10.0),
]

EXPTS_DOWN = [
    dict(folder_prefix="2025-05-15", wavelength=850.0, information="SnS - 850 nm - down sequence True 2", vsd_end=10.0),
    dict(folder_prefix="2025-05-15", wavelength=680.0, information="SnS - 680 nm - down sequence", vsd_end=10.0),
    dict(folder_prefix="2025-05-16", wavelength=590.0, information="SnS - 590 nm - down sequence", vsd_end=10.0),
    dict(folder_prefix="505_up_down", wavelength=505.0, information="SnS - 505 nm - down sequence", vsd_end=10.0),
    dict(folder_prefix="455_down_2", wavelength=455.0, information="SnS - 455 nm - power down", vsd_end=10.0),
    dict(folder_prefix="2025-05-29", wavelength=385.0, information="SnS - 385 nm - power down", vsd_end=10.0),
]

# Wire-bonded sample (new device, factor ~ 0.061)
EXPTS_UP_WB = [
    dict(folder_prefix="2025-06-23", wavelength=850.0, information="SnS - 850 nm - power up", vsd_end=5.0),
    dict(folder_prefix="2025-06-24", wavelength=625.0, information="SnS - 625 nm - power up", vsd_end=5.0),
    dict(folder_prefix="2025-06-24", wavelength=565.0, information="SnS - 565 nm - power up", vsd_end=5.0),
    dict(folder_prefix="2025-06-25", wavelength=505.0, information="SnS - 505 nm - power up", vsd_end=5.0),
    dict(folder_prefix="2025-06-25", wavelength=455.0, information="SnS - 455 nm - power up", vsd_end=5.0),
    dict(folder_prefix="2025-06-25", wavelength=385.0, information="SnS - 385 nm - power up", vsd_end=5.0),
]

EXPTS_DOWN_WB = [
    dict(folder_prefix="2025-06-23", wavelength=850.0, information="SnS - 850 nm - power down", vsd_end=5.0),
    dict(folder_prefix="2025-06-24", wavelength=625.0, information="SnS - 625 nm - power down 2", vsd_end=5.0),
    dict(folder_prefix="2025-06-24", wavelength=565.0, information="SnS - 565 nm - power down", vsd_end=5.0),
    dict(folder_prefix="2025-06-25", wavelength=505.0, information="SnS - 505 nm - power down", vsd_end=5.0),
    dict(folder_prefix="2025-06-25", wavelength=455.0, information="SnS - 455 nm - power down", vsd_end=5.0),
    dict(folder_prefix="2025-06-25", wavelength=385.0, information="SnS - 385 nm - power down 2", vsd_end=5.0),
]


def main():
    print("Loading all raw data...")
    props, data = load_all_raw()
    print(f"Loaded {props.height} IV experiments")

    print("Loading calibration data...")
    cal_props, cal_data = load_calibrations()
    print(f"Loaded {cal_props.height} calibration files\n")

    # === Probe-contacted sample ===
    print("=== Probe-contacted sample (factor={:.4f}) ===".format(FACTOR_PROBE))

    print("[power up]")
    overlay_hyst_vs_power(
        EXPTS_UP, props, data, cal_props, cal_data,
        area_factor=FACTOR_PROBE,
        outpath=OUTDIR / "hyst_vs_power_up.png",
    )

    print("[power up, baseline]")
    overlay_hyst_vs_power(
        EXPTS_UP, props, data, cal_props, cal_data,
        area_factor=FACTOR_PROBE,
        baseline=True,
        outpath=OUTDIR / "hyst_vs_power_up_baseline.png",
    )

    print("[power down]")
    overlay_hyst_vs_power(
        EXPTS_DOWN, props, data, cal_props, cal_data,
        area_factor=FACTOR_PROBE,
        outpath=OUTDIR / "hyst_vs_power_down.png",
    )

    print("[power down, baseline]")
    overlay_hyst_vs_power(
        EXPTS_DOWN, props, data, cal_props, cal_data,
        area_factor=FACTOR_PROBE,
        baseline=True,
        outpath=OUTDIR / "hyst_vs_power_down_baseline.png",
    )

    # === Wire-bonded sample ===
    print("\n=== Wire-bonded sample (factor={:.4f}) ===".format(FACTOR_WIREBOND))

    print("[power up]")
    overlay_hyst_vs_power(
        EXPTS_UP_WB, props, data, cal_props, cal_data,
        area_factor=FACTOR_WIREBOND,
        outpath=OUTDIR / "hyst_vs_power_up_wirebonded.png",
    )

    print("[power down]")
    overlay_hyst_vs_power(
        EXPTS_DOWN_WB, props, data, cal_props, cal_data,
        area_factor=FACTOR_WIREBOND,
        outpath=OUTDIR / "hyst_vs_power_down_wirebonded.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
