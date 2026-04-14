"""
Hysteresis analysis algorithms for IV sweep data.

Pure functions (Polars + numpy for polyfit) that compute hysteresis curves
and extract maxima. Used by both extractors (scalar metrics) and notebooks
(full curves).

Two methods:
- raw_hysteresis: I_forward - I_backward at each voltage
- poly_corrected_hysteresis: Polynomial background subtraction then peak extraction
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Tuple

from scipy.signal import savgol_filter


def _smooth_branch(current: np.ndarray, window: int, poly_order: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing, interpolating over NaNs first."""
    if window < poly_order + 2:
        window = poly_order + 2
    if window % 2 == 0:
        window += 1
    if len(current) < window:
        return current

    arr = current.copy()
    nans = np.isnan(arr)
    if nans.any():
        good = ~nans
        if good.sum() < window:
            return current
        arr[nans] = np.interp(np.where(nans)[0], np.where(good)[0], arr[good])

    return savgol_filter(arr, window, poly_order)


def compute_raw_hysteresis(
    df: pl.DataFrame,
    dv_threshold: float = 0.05,
    smooth_window: int = 0,
) -> pl.DataFrame:
    """
    Compute raw hysteresis: I_forward - I_backward at each voltage.

    Polars port of forward_minus_backward() from Kedro preprocessing nodes.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns "Vsd (V)" and "I (A)" in time order.
    dv_threshold : float, optional
        Minimum voltage step to classify as forward/backward sweep (default 0.05 V).
    smooth_window : int, optional
        Savitzky-Golay window length applied to each branch before taking
        the difference.  0 (default) disables smoothing.  A good starting
        value is 11-21 points depending on step size.

    Returns
    -------
    pl.DataFrame
        Columns: ["Vsd (V)", "I (A)"] where I (A) = I_fwd - I_bwd
    """
    df = df.clone()
    df = df.with_columns(pl.col("Vsd (V)").round(2))

    # Classify direction by sign of dV
    dv = df["Vsd (V)"].diff()
    directions = pl.when(dv > dv_threshold).then(pl.lit("forward")) \
                   .when(dv < -dv_threshold).then(pl.lit("backward")) \
                   .otherwise(None)
    df = df.with_columns(directions.alias("direction"))

    # Forward-fill then backward-fill to propagate direction labels
    df = df.with_columns(pl.col("direction").forward_fill().backward_fill())

    # Split into forward and backward segments
    forward = df.filter(pl.col("direction") == "forward").drop("direction")
    backward = df.filter(pl.col("direction") == "backward").drop("direction")

    forward = forward.sort("Vsd (V)")
    backward = backward.sort("Vsd (V)")

    # Patch curves to ensure overlap at edges
    if not backward.is_empty():
        forward = pl.concat([backward.head(1), forward], how="vertical")
    if not forward.is_empty():
        backward = pl.concat([backward, forward.tail(1)], how="vertical")

    # Group and average
    forward_grouped = forward.group_by("Vsd (V)", maintain_order=True).agg(
        pl.col("I (A)").mean()
    ).sort("Vsd (V)")

    backward_grouped = backward.group_by("Vsd (V)", maintain_order=True).agg(
        pl.col("I (A)").mean()
    ).sort("Vsd (V)")

    # Smooth each branch before differencing
    if smooth_window > 0:
        forward_grouped = forward_grouped.with_columns(
            pl.Series("I (A)", _smooth_branch(forward_grouped["I (A)"].to_numpy(), smooth_window))
        )
        backward_grouped = backward_grouped.with_columns(
            pl.Series("I (A)", _smooth_branch(backward_grouped["I (A)"].to_numpy(), smooth_window))
        )

    # Join on Vsd and compute difference
    merged = forward_grouped.join(
        backward_grouped,
        on="Vsd (V)",
        how="inner",
        suffix="_bwd"
    )

    result = merged.with_columns(
        (pl.col("I (A)") - pl.col("I (A)_bwd")).alias("I (A)")
    ).select(["Vsd (V)", "I (A)"])

    return result


def extract_raw_hysteresis_maxima(hysteresis_df: pl.DataFrame) -> dict:
    """
    Extract maximum hysteresis values for negative and positive voltage branches.

    Parameters
    ----------
    hysteresis_df : pl.DataFrame
        Output from compute_raw_hysteresis() with columns ["Vsd (V)", "I (A)"]

    Returns
    -------
    dict
        {"max_negative_hysteresis": float, "max_positive_hysteresis": float}
    """
    negative = hysteresis_df.filter(pl.col("Vsd (V)") < 0)
    positive = hysteresis_df.filter(pl.col("Vsd (V)") > 0)

    max_neg = negative["I (A)"].max() if not negative.is_empty() else float("nan")
    max_pos = positive["I (A)"].max() if not positive.is_empty() else float("nan")

    return {
        "max_negative_hysteresis": max_neg,
        "max_positive_hysteresis": max_pos,
    }


def compute_hysteresis_area(
    df: pl.DataFrame,
    dv_threshold: float = 0.05,
    smooth_window: int = 0,
) -> dict:
    """
    Compute hysteresis loop area via trapezoidal integration of |I_fwd - I_bwd| dV.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns "Vsd (V)" and "I (A)" in time order.
    dv_threshold : float, optional
        Minimum voltage step to classify as forward/backward sweep (default 0.05 V).
    smooth_window : int, optional
        Savitzky-Golay window length for branch smoothing (0 = off).

    Returns
    -------
    dict
        {"total_area": float, "positive_area": float, "negative_area": float}
        Areas in units of V*A (watts). positive/negative refer to the V > 0 and V < 0
        branches of the hysteresis loop.
    """
    hyst = compute_raw_hysteresis(df, dv_threshold=dv_threshold, smooth_window=smooth_window)

    if hyst.is_empty() or hyst.height < 2:
        return {
            "total_area": float("nan"),
            "positive_area": float("nan"),
            "negative_area": float("nan"),
        }

    v = hyst["Vsd (V)"].to_numpy()
    i_diff = hyst["I (A)"].to_numpy()

    total_area = float(np.trapz(np.abs(i_diff), v))

    pos_mask = v > 0
    neg_mask = v < 0

    positive_area = float("nan")
    if np.sum(pos_mask) >= 2:
        positive_area = float(np.trapz(np.abs(i_diff[pos_mask]), v[pos_mask]))

    negative_area = float("nan")
    if np.sum(neg_mask) >= 2:
        negative_area = float(np.trapz(np.abs(i_diff[neg_mask]), v[neg_mask]))

    return {
        "total_area": total_area,
        "positive_area": positive_area,
        "negative_area": negative_area,
    }


def compute_on_off_ratio(
    df: pl.DataFrame,
    read_voltage: float = 2.0,
    dv_threshold: float = 0.05,
    voltage_tolerance: float = 0.15,
    smooth_window: int = 0,
) -> dict:
    """
    Compute on/off current ratio at a fixed read voltage.

    The ratio is |I_forward / I_backward| at the voltage closest to read_voltage.
    For memristors, the forward (SET) branch carries more current than the
    backward (RESET) branch, so the ratio should be > 1.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns "Vsd (V)" and "I (A)" in time order.
    read_voltage : float
        Voltage at which to evaluate the on/off ratio (default 2.0 V).
    dv_threshold : float
        Minimum voltage step to classify sweep direction (default 0.05 V).
    voltage_tolerance : float
        Maximum distance from read_voltage to accept a data point (default 0.15 V).
    smooth_window : int, optional
        Savitzky-Golay window length for branch smoothing (0 = off).

    Returns
    -------
    dict
        {"on_off_ratio": float, "i_on": float, "i_off": float, "v_read_actual": float}
    """
    nan_result = {
        "on_off_ratio": float("nan"),
        "i_on": float("nan"),
        "i_off": float("nan"),
        "v_read_actual": float("nan"),
    }

    df = df.clone().with_columns(pl.col("Vsd (V)").round(2))
    dv = df["Vsd (V)"].diff()
    directions = (
        pl.when(dv > dv_threshold).then(pl.lit("forward"))
        .when(dv < -dv_threshold).then(pl.lit("backward"))
        .otherwise(None)
    )
    df = df.with_columns(directions.alias("direction"))
    df = df.with_columns(pl.col("direction").forward_fill().backward_fill())

    forward = df.filter(pl.col("direction") == "forward")
    backward = df.filter(pl.col("direction") == "backward")

    if forward.is_empty() or backward.is_empty():
        return nan_result

    fwd_grouped = (
        forward.group_by("Vsd (V)", maintain_order=True)
        .agg(pl.col("I (A)").mean())
        .sort("Vsd (V)")
    )
    bwd_grouped = (
        backward.group_by("Vsd (V)", maintain_order=True)
        .agg(pl.col("I (A)").mean())
        .sort("Vsd (V)")
    )

    if smooth_window > 0:
        fwd_grouped = fwd_grouped.with_columns(
            pl.Series("I (A)", _smooth_branch(fwd_grouped["I (A)"].to_numpy(), smooth_window))
        )
        bwd_grouped = bwd_grouped.with_columns(
            pl.Series("I (A)", _smooth_branch(bwd_grouped["I (A)"].to_numpy(), smooth_window))
        )

    fwd_v = fwd_grouped["Vsd (V)"].to_numpy()
    bwd_v = bwd_grouped["Vsd (V)"].to_numpy()

    fwd_idx = int(np.argmin(np.abs(fwd_v - read_voltage)))
    bwd_idx = int(np.argmin(np.abs(bwd_v - read_voltage)))

    if (abs(fwd_v[fwd_idx] - read_voltage) > voltage_tolerance
            or abs(bwd_v[bwd_idx] - read_voltage) > voltage_tolerance):
        return nan_result

    i_fwd = float(fwd_grouped["I (A)"][fwd_idx])
    i_bwd = float(bwd_grouped["I (A)"][bwd_idx])

    if abs(i_bwd) < 1e-15:
        return nan_result

    v_actual = float(fwd_v[fwd_idx])
    i_on = max(abs(i_fwd), abs(i_bwd))
    i_off = min(abs(i_fwd), abs(i_bwd))

    return {
        "on_off_ratio": i_on / i_off,
        "i_on": i_on,
        "i_off": i_off,
        "v_read_actual": v_actual,
    }


def compute_coercive_voltages(
    df: pl.DataFrame,
    step_size: float = 0.1,
    poly_order: int = 7,
    onset_fraction: float = 0.1,
    v_min_abs: float = 0.3,
    smooth_window: int = 15,
) -> dict:
    """
    Extract characteristic voltages from the poly-corrected IV hysteresis.

    Uses polynomial background subtraction to remove baseline drift, then
    finds the peak and onset voltages of the corrected forward-branch
    current on each polarity side.

    * **Peak voltage** -- voltage at which the corrected hysteresis is
      maximum.  This is where the memory window is widest.
    * **Onset voltage** -- voltage at which the corrected hysteresis first
      exceeds ``onset_fraction`` of the peak value.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns "Vsd (V)" and "I (A)" in time order.
    step_size : float
        Voltage step for the poly-correction grid snap (default 0.1 V).
    poly_order : int
        Polynomial order for background fitting (default 7).
    onset_fraction : float
        Fraction of peak hysteresis used to define onset (default 0.1 = 10 %).
    v_min_abs : float
        Ignore voltages closer to zero than this to avoid noise (default 0.3 V).
    smooth_window : int, optional
        Savitzky-Golay window applied to the corrected forward branch
        before peak finding (default 15).  0 disables smoothing.

    Returns
    -------
    dict
        {"v_peak_pos": float, "v_onset_pos": float,
         "v_peak_neg": float, "v_onset_neg": float}
        Positive-branch values are > 0, negative-branch values are < 0.
        NaN when a branch has no measurable hysteresis.
    """
    nan_result = {
        "v_peak_pos": float("nan"),
        "v_onset_pos": float("nan"),
        "v_peak_neg": float("nan"),
        "v_onset_neg": float("nan"),
    }

    corrected = compute_poly_corrected_hysteresis(
        df, step_size=step_size, poly_order=poly_order,
    )

    if corrected.is_empty():
        return nan_result

    fwd = corrected.filter(pl.col("direction") == "forward").sort("Vsd (V)")
    if fwd.is_empty():
        return nan_result

    v_all = fwd["Vsd (V)"].to_numpy()
    i_all = fwd["I (A)"].to_numpy()

    if smooth_window > 0 and len(i_all) >= smooth_window:
        i_all = _smooth_branch(i_all, smooth_window)

    result = dict(nan_result)

    for sign, v_filter, key_peak, key_onset in [
        ("pos", v_all > v_min_abs, "v_peak_pos", "v_onset_pos"),
        ("neg", v_all < -v_min_abs, "v_peak_neg", "v_onset_neg"),
    ]:
        if np.sum(v_filter) < 2:
            continue

        v_branch = v_all[v_filter]
        h_branch = i_all[v_filter]

        if sign == "pos":
            peak_idx = int(np.argmax(h_branch))
            result[key_peak] = float(v_branch[peak_idx])
            threshold = onset_fraction * h_branch[peak_idx]
            above = np.where(h_branch > threshold)[0]
            if len(above) > 0:
                result[key_onset] = float(v_branch[above[0]])
        else:
            # Negative branch: hysteresis dips below zero, use argmin
            peak_idx = int(np.argmin(h_branch))
            result[key_peak] = float(v_branch[peak_idx])
            threshold = onset_fraction * h_branch[peak_idx]
            below = np.where(h_branch < threshold)[0]
            if len(below) > 0:
                result[key_onset] = float(v_branch[below[-1]])

    return result


def compute_poly_corrected_hysteresis(
    df: pl.DataFrame,
    step_size: float = 0.1,
    poly_order: int = 7,
) -> pl.DataFrame:
    """
    Background-correct an I-V sweep using polynomial fitting and return cleaned curve.

    Polars port of normalize_background_current() from Kedro SnS_preprocessing nodes.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain "Vsd (V)" and "I (A)" columns in time order.
    step_size : float, default 0.1
        Programmed voltage increment of the instrument (used to snap jitter to grid).
    poly_order : int, default 7
        Order of the polynomial fitted on each backward leg.

    Returns
    -------
    pl.DataFrame
        Columns: ["Vsd (V)", "I (A)", "direction"]
        I (A) is the background-corrected current.
    """
    tol = 0.8 * step_size

    df = df.clone()

    # Snap every voltage to the grid
    df = df.with_columns(
        ((pl.col("Vsd (V)") / step_size).round().cast(pl.Float64) * step_size).alias("Vsd (V)")
    )

    # Identify zero crossings
    is_zero = df["Vsd (V)"].abs() <= tol
    is_zero_arr = is_zero.to_numpy()
    zero_idx = np.flatnonzero(is_zero_arr & ~np.roll(is_zero_arr, 1))
    # Fix first element if it was incorrectly detected
    if len(zero_idx) > 0 and zero_idx[0] == 0 and not is_zero_arr[0]:
        zero_idx = zero_idx[1:]

    # Not enough zero crossings
    if len(zero_idx) < 3:
        empty_df = pl.DataFrame({
            "Vsd (V)": pl.Series([], dtype=pl.Float64),
            "I (A)": pl.Series([], dtype=pl.Float64),
            "direction": pl.Series([], dtype=pl.String),
        })
        return empty_df

    # Work with numpy arrays for segment-by-segment processing
    v_sd = df["Vsd (V)"].to_numpy().copy()
    i_raw = df["I (A)"].to_numpy().copy()
    directions = np.full(len(df), None, dtype=object)
    i_corr = np.full(len(df), np.nan, dtype=np.float64)

    n_loops = (len(zero_idx) - 1) // 2

    for loop in range(n_loops):
        i0, i1, i2 = int(zero_idx[2 * loop]), int(zero_idx[2 * loop + 1]), int(zero_idx[2 * loop + 2])

        for lo, hi in [(i0 + 1, i1), (i1 + 1, i2)]:
            if hi - lo < poly_order + 1:
                continue

            mid = lo + (hi - lo) // 2

            # Set direction labels
            directions[lo:mid + 1] = "forward"
            directions[mid + 1:hi + 1] = "backward"

            # Fit polynomial to backward leg
            x_back = v_sd[mid + 1:hi + 1]
            y_back = i_raw[mid + 1:hi + 1]

            if len(x_back) < poly_order + 1:
                continue

            coeffs = np.polyfit(x_back, y_back, poly_order)
            p = np.poly1d(coeffs)

            # Subtract polynomial from full segment
            x_seg = v_sd[lo:hi + 1]
            y_seg = i_raw[lo:hi + 1]
            i_corr[lo:hi + 1] = y_seg - p(x_seg)

    # Build result DataFrame
    # Convert direction array to strings (None -> null)
    direction_str = [d if d is not None else None for d in directions]
    
    result_df = pl.DataFrame({
        "Vsd (V)": pl.Series(v_sd),
        "I (A)": pl.Series(i_corr),
        "direction": pl.Series(direction_str, dtype=pl.String),
    })

    # Keep voltages present in both directions
    direction_counts = result_df.group_by("Vsd (V)", maintain_order=True).agg(
        pl.col("direction").drop_nulls().n_unique().alias("n_directions")
    )
    shared_voltages = direction_counts.filter(pl.col("n_directions") == 2).select("Vsd (V)")

    if shared_voltages.is_empty():
        empty_df = pl.DataFrame({
            "Vsd (V)": pl.Series([], dtype=pl.Float64),
            "I (A)": pl.Series([], dtype=pl.Float64),
            "direction": pl.Series([], dtype=pl.String),
        })
        return empty_df

    df_shared = result_df.join(shared_voltages, on="Vsd (V)", how="inner")

    # Forward-fill and backward-fill corrected current and direction
    df_shared = df_shared.with_columns([
        pl.col("I (A)").forward_fill().backward_fill(),
        pl.col("direction").forward_fill().backward_fill(),
    ])

    # Return clean curve sorted by direction and voltage
    result = df_shared.sort(["direction", "Vsd (V)"])

    return result


def extract_poly_corrected_maxima(corrected_df: pl.DataFrame) -> dict:
    """
    Locate peak hysteresis current on negative and positive voltage branches.

    Parameters
    ----------
    corrected_df : pl.DataFrame
        Output from compute_poly_corrected_hysteresis() with columns
        ["Vsd (V)", "I (A)", "direction"]

    Returns
    -------
    dict
        {"max_neg_I": float, "max_neg_V": float, "max_pos_I": float, "max_pos_V": float}
    """
    if corrected_df.is_empty():
        return {
            "max_neg_I": float("nan"),
            "max_neg_V": float("nan"),
            "max_pos_I": float("nan"),
            "max_pos_V": float("nan"),
        }

    # Split by sign of Vsd
    neg = corrected_df.filter(pl.col("Vsd (V)") < 0)
    pos = corrected_df.filter(pl.col("Vsd (V)") > 0)

    # Negative branch
    if neg.is_empty():
        max_neg_I = float("nan")
        max_neg_V = float("nan")
    else:
        max_neg_row = neg.row(index=neg["I (A)"].arg_max(), named=True)
        max_neg_I = max_neg_row["I (A)"]
        max_neg_V = max_neg_row["Vsd (V)"]

    # Positive branch
    if pos.is_empty():
        max_pos_I = float("nan")
        max_pos_V = float("nan")
    else:
        max_pos_row = pos.row(index=pos["I (A)"].arg_max(), named=True)
        max_pos_I = max_pos_row["I (A)"]
        max_pos_V = max_pos_row["Vsd (V)"]

    return {
        "max_neg_I": max_neg_I,
        "max_neg_V": max_neg_V,
        "max_pos_I": max_pos_I,
        "max_pos_V": max_pos_V,
    }
