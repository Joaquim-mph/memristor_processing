"""
Plot memristor figures of merit vs LED optical power for wavelength-dependent experiments.

Three FOMs:
  1. Hysteresis loop area (trapezoidal integral of |I_fwd - I_bwd| dV)
  2. On/off current ratio at a fixed read voltage
  3. Coercive voltages (V_SET and V_RESET)

Usage:
    python scripts/plot_fom_vs_power.py
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
    compute_hysteresis_area,
    compute_on_off_ratio,
    compute_coercive_voltages,
)

apply_default_style()

OUTDIR = Path("figures")

# ── Area factors ──

SPOT_PROBE = np.pi * 179**2
DEVICE_PROBE = 121 * 100
FACTOR_PROBE = DEVICE_PROBE / SPOT_PROBE

SPOT_WIREBOND = np.pi * 190**2
DEVICE_WIREBOND = 61 * 114
FACTOR_WIREBOND = DEVICE_WIREBOND / SPOT_WIREBOND


# ── Calibration helpers ──

def build_calibration_functions(cal_props, cal_data, date_prefix, wavelengths):
    calibration = {}
    for wl in wavelengths:
        filtered = cal_props.filter(
            (cal_props["Laser wavelength"] == wl)
            & (cal_props["data_key"].str.starts_with(date_prefix))
        )
        if filtered.is_empty():
            continue
        key = filtered["data_key"][0]
        df = cal_data[key]
        f = interp1d(
            df["VL (V)"].to_numpy(),
            df["Power (W)"].to_numpy(),
            kind="linear",
            fill_value="extrapolate",
        )
        calibration[wl] = f
    return calibration


# ── Per-sweep FOM extraction ──

def _extract_sweep_foms(df, step, read_voltage=2.0, smooth_window=0):
    """Extract all three FOMs from a single IV sweep."""
    dv = step / 2

    area_result = compute_hysteresis_area(df, dv_threshold=dv, smooth_window=smooth_window)
    ratio_result = compute_on_off_ratio(df, read_voltage=read_voltage, dv_threshold=dv, smooth_window=smooth_window)
    coercive_result = compute_coercive_voltages(df, step_size=step)

    return {
        "hysteresis_area": area_result["total_area"],
        "on_off_ratio": ratio_result["on_off_ratio"],
        "v_peak_pos": coercive_result["v_peak_pos"],
        "v_onset_pos": coercive_result["v_onset_pos"],
        "v_peak_neg": coercive_result["v_peak_neg"],
        "v_onset_neg": coercive_result["v_onset_neg"],
    }


# ── Collect power vs FOM data for a set of experiments ──

def _collect_fom_vs_power(
    experiments, props, data, cal_props, cal_data,
    *, area_factor, read_voltage=2.0, smooth_window=0,
):
    """
    For each wavelength experiment, build calibration, iterate sweeps,
    and return {wavelength: {"powers": [...], "foms": {key: [...]}}}
    """
    results = {}

    for exp in experiments:
        dp = exp["folder_prefix"]
        wl = exp["wavelength"]
        info = exp["information"]
        vsd_end = exp.get("vsd_end")

        cals = build_calibration_functions(
            cal_props, cal_data, date_prefix=dp, wavelengths=[wl],
        )
        if wl not in cals:
            print(f"  [warn] no calibration for {wl} nm on {dp}, skipping")
            continue
        cal_fn = cals[wl]

        filtered = filter_experiments(
            props, folder_prefix=dp, information=info,
            wavelength=wl, vsd_end=vsd_end,
        )
        if filtered.height == 0:
            print(f"  [warn] no IV data for {info}, skipping")
            continue

        powers = []
        fom_keys = ["hysteresis_area", "on_off_ratio", "v_peak_pos", "v_onset_pos", "v_peak_neg", "v_onset_neg"]
        foms = {k: [] for k in fom_keys}

        for i in range(filtered.height):
            row = filtered.row(i, named=True)
            key = row["data_key"]
            if key not in data:
                continue

            df = data[key]
            step = row.get("VSD step", 0.1)
            sweep_foms = _extract_sweep_foms(df, step, read_voltage=read_voltage, smooth_window=smooth_window)

            laser_v = row["Laser voltage"]
            power = float(cal_fn(laser_v)) * area_factor
            powers.append(power)
            for k in foms:
                foms[k].append(sweep_foms[k])

        if powers:
            results[wl] = {"powers": np.array(powers), "foms": {k: np.array(v) for k, v in foms.items()}}
            print(f"  [{wl:.0f} nm] {len(powers)} sweeps")

    return results


# ── Plotting ──

def plot_fom_vs_power(
    collected,
    *,
    fom_key,
    ylabel,
    baseline=False,
    outpath=None,
    show=False,
):
    fig, ax = plt.subplots()

    for wl in sorted(collected.keys()):
        entry = collected[wl]
        x = entry["powers"]
        y = entry["foms"][fom_key]

        mask = ~np.isnan(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue

        if baseline and len(y) > 0:
            y = y - y[0]

        ax.plot(x, y, "o-", label=f"{wl:.0f} nm")

    ax.set_xlabel("Effective Power [W]")
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


def plot_coercive_vs_power(
    collected,
    *,
    baseline=False,
    outpath=None,
    show=False,
):
    panels = [
        ("v_peak_pos", r"$V_{peak}$ (V > 0)"),
        ("v_onset_pos", r"$V_{onset}$ (V > 0)"),
        ("v_peak_neg", r"$V_{peak}$ (V < 0)"),
        ("v_onset_neg", r"$V_{onset}$ (V < 0)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    for ax, (key, title) in zip(axes.flat, panels):
        for wl in sorted(collected.keys()):
            entry = collected[wl]
            x = entry["powers"]
            y = entry["foms"][key]
            mask = ~np.isnan(y)
            xm, ym = x[mask], y[mask]
            if len(xm) == 0:
                continue
            if baseline and len(ym) > 0:
                ym = ym - ym[0]
            ax.plot(xm, ym, "o-", label=f"{wl:.0f} nm")

        ax.set_title(title)
        ax.set_xlabel("Effective Power [W]")
        ax.set_ylabel("Voltage [V]")
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


# ── Experiment definitions (same as plot_hysteresis_vs_power.py) ──

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


def _average_up_down(collected_up, collected_down, n_points=50):
    """
    Average up and down sequences onto a common power grid per wavelength.

    For each wavelength present in both dicts, builds a uniform power grid
    spanning the overlapping range, interpolates both directions, and returns
    the mean (and std as error bars).
    """
    common_wls = set(collected_up.keys()) & set(collected_down.keys())
    averaged = {}

    for wl in sorted(common_wls):
        up = collected_up[wl]
        dn = collected_down[wl]

        p_lo = max(up["powers"].min(), dn["powers"].min())
        p_hi = min(up["powers"].max(), dn["powers"].max())
        if p_lo >= p_hi:
            continue

        p_grid = np.linspace(p_lo, p_hi, n_points)
        fom_mean = {}
        fom_std = {}

        for key in up["foms"]:
            y_up = up["foms"][key]
            y_dn = dn["foms"][key]

            mask_up = ~np.isnan(y_up)
            mask_dn = ~np.isnan(y_dn)
            if mask_up.sum() < 2 or mask_dn.sum() < 2:
                fom_mean[key] = np.full(n_points, np.nan)
                fom_std[key] = np.full(n_points, np.nan)
                continue

            f_up = interp1d(up["powers"][mask_up], y_up[mask_up], kind="linear", bounds_error=False, fill_value=np.nan)
            f_dn = interp1d(dn["powers"][mask_dn], y_dn[mask_dn], kind="linear", bounds_error=False, fill_value=np.nan)

            y_up_interp = f_up(p_grid)
            y_dn_interp = f_dn(p_grid)

            stack = np.vstack([y_up_interp, y_dn_interp])
            fom_mean[key] = np.nanmean(stack, axis=0)
            fom_std[key] = np.nanstd(stack, axis=0)

        averaged[wl] = {"powers": p_grid, "foms": fom_mean, "fom_std": fom_std}

    return averaged


def plot_fom_vs_power_mean(
    averaged,
    *,
    fom_key,
    ylabel,
    baseline=False,
    outpath=None,
    show=False,
):
    """Plot mean of up/down with shaded std band."""
    fig, ax = plt.subplots()

    for wl in sorted(averaged.keys()):
        entry = averaged[wl]
        x = entry["powers"]
        y = entry["foms"][fom_key]
        yerr = entry["fom_std"][fom_key]

        mask = ~np.isnan(y)
        x, y, yerr = x[mask], y[mask], yerr[mask]
        if len(x) == 0:
            continue

        if baseline and len(y) > 0:
            y = y - y[0]

        ax.plot(x, y, "-", label=f"{wl:.0f} nm")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    ax.set_xlabel("Effective Power [W]")
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


def plot_coercive_vs_power_mean(
    averaged,
    *,
    baseline=False,
    outpath=None,
    show=False,
):
    """Plot mean coercive voltages with shaded std band (2x2 panel)."""
    panels = [
        ("v_peak_pos", r"$V_{peak}$ (V > 0)"),
        ("v_onset_pos", r"$V_{onset}$ (V > 0)"),
        ("v_peak_neg", r"$V_{peak}$ (V < 0)"),
        ("v_onset_neg", r"$V_{onset}$ (V < 0)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    for ax, (key, title) in zip(axes.flat, panels):
        for wl in sorted(averaged.keys()):
            entry = averaged[wl]
            x = entry["powers"]
            y = entry["foms"][key]
            yerr = entry["fom_std"][key]
            mask = ~np.isnan(y)
            xm, ym, ye = x[mask], y[mask], yerr[mask]
            if len(xm) == 0:
                continue
            if baseline and len(ym) > 0:
                ym = ym - ym[0]
            ax.plot(xm, ym, "-", label=f"{wl:.0f} nm")
            ax.fill_between(xm, ym - ye, ym + ye, alpha=0.2)

        ax.set_title(title)
        ax.set_xlabel("Effective Power [W]")
        ax.set_ylabel("Voltage [V]")
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


def _run_sample(label, experiments, props, data, cal_props, cal_data, area_factor, read_voltage, smooth_window=0):
    print(f"\n=== {label} (factor={area_factor:.4f}, smooth={smooth_window}) ===")

    all_collected = {}
    for direction, expts in experiments:
        print(f"\n--- {direction} ---")
        collected = _collect_fom_vs_power(
            expts, props, data, cal_props, cal_data,
            area_factor=area_factor, read_voltage=read_voltage,
            smooth_window=smooth_window,
        )
        all_collected[direction] = collected

        suffix = f"_{direction.lower().replace(' ', '_')}"

        plot_fom_vs_power(
            collected,
            fom_key="hysteresis_area",
            ylabel="Hysteresis Area [V·A]",
            outpath=OUTDIR / f"area_vs_power{suffix}.png",
        )

        plot_fom_vs_power(
            collected,
            fom_key="hysteresis_area",
            ylabel=r"$\Delta$ Hysteresis Area [V·A]",
            baseline=True,
            outpath=OUTDIR / f"area_vs_power{suffix}_baseline.png",
        )

        plot_fom_vs_power(
            collected,
            fom_key="on_off_ratio",
            ylabel=f"On/Off Ratio (V_read = {read_voltage} V)",
            outpath=OUTDIR / f"on_off_vs_power{suffix}.png",
        )

        plot_coercive_vs_power(
            collected,
            outpath=OUTDIR / f"coercive_vs_power{suffix}.png",
        )

    # ── Mean of up/down ──
    directions = list(all_collected.keys())
    if len(directions) == 2:
        print(f"\n--- mean ({directions[0]} + {directions[1]}) ---")
        averaged = _average_up_down(all_collected[directions[0]], all_collected[directions[1]])
        d0 = directions[0].lower().replace(" ", "_")
        # strip "up"/"down" to get shared prefix (e.g. "power" or "power_wb")
        base = d0.replace("_up", "").replace("_down", "")
        suffix = f"_{base}_mean"

        plot_fom_vs_power_mean(
            averaged,
            fom_key="hysteresis_area",
            ylabel="Hysteresis Area [V·A]",
            outpath=OUTDIR / f"area_vs_power{suffix}.png",
        )

        plot_fom_vs_power_mean(
            averaged,
            fom_key="hysteresis_area",
            ylabel=r"$\Delta$ Hysteresis Area [V·A]",
            baseline=True,
            outpath=OUTDIR / f"area_vs_power{suffix}_baseline.png",
        )

        plot_fom_vs_power_mean(
            averaged,
            fom_key="on_off_ratio",
            ylabel=f"On/Off Ratio (V_read = {read_voltage} V)",
            outpath=OUTDIR / f"on_off_vs_power{suffix}.png",
        )

        plot_coercive_vs_power_mean(
            averaged,
            outpath=OUTDIR / f"coercive_vs_power{suffix}.png",
        )


def main():
    print("Loading all raw data...")
    props, data = load_all_raw()
    print(f"Loaded {props.height} IV experiments")

    print("Loading calibration data...")
    cal_props, cal_data = load_calibrations()
    print(f"Loaded {cal_props.height} calibration files")

    smooth = 15

    _run_sample(
        "Probe-contacted",
        [("power up", EXPTS_UP), ("power down", EXPTS_DOWN)],
        props, data, cal_props, cal_data,
        area_factor=FACTOR_PROBE,
        read_voltage=5.0,
        smooth_window=smooth,
    )

    _run_sample(
        "Wire-bonded",
        [("power up WB", EXPTS_UP_WB), ("power down WB", EXPTS_DOWN_WB)],
        props, data, cal_props, cal_data,
        area_factor=FACTOR_WIREBOND,
        read_voltage=2.5,
        smooth_window=smooth,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
