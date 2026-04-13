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


def compute_raw_hysteresis(
    df: pl.DataFrame,
    dv_threshold: float = 0.05
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
