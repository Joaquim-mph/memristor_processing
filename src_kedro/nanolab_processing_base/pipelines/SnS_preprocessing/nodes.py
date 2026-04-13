"""
This is a boilerplate pipeline 'SnS_preprocessing'
generated using Kedro 0.19.11
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Callable



def normalize_background_current(
    df: pd.DataFrame,
    *,
    step_size: float = 0.1,             
    poly_order: int = 7,
) -> pd.DataFrame:
    """
    Background-correct an I-V sweep and return the cleaned forward/backward-
    averaged curve on the instrument’s voltage grid.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain "Vsd (V)" and "I (A)" columns in time order.
    step_size : float, default 0.1
        Programmed voltage increment of the instrument (used to snap
        ±0.0005 V jitter to the exact grid).
    tol : float, optional
        Zero-crossing tolerance.  If None → 0.8·step_size.
    poly_order : int, default 7
        Order of the polynomial fitted on each backward leg.

    Returns
    -------
    pd.DataFrame
        Two columns:
          • "Vsd (V)"  – grid voltages
          • "I (A)"    – (raw − poly-fit) current, averaged fwd/bwd
    """
    tol = 0.8 * step_size         

    df = df.reset_index(drop=True).copy()

    # SNAP EVERY VOLTAGE TO THE GRID 
    df["Vsd (V)"] = (
        (df["Vsd (V)"] / step_size).round().astype(float) * step_size
    )

    # IDENTIFY ZERO CROSSINGS
    is_zero  = df["Vsd (V)"].abs().le(tol)
    zero_idx = np.flatnonzero(is_zero & ~is_zero.shift(fill_value=False))
    # if len(zero_idx) < 3 or (len(zero_idx) - 1) % 2:
    #     raise ValueError("Sweep must contain full 0→−V→0→+V→0 cycles.")

    df["direction"] = np.nan
    df["I_corr"]    = np.nan

    n_loops = (len(zero_idx) - 1) // 2
    for loop in range(n_loops):
        i0, i1, i2 = zero_idx[2*loop : 2*loop + 3]
        for lo, hi in [(i0 + 1, i1), (i1 + 1, i2)]:
            if hi - lo < poly_order + 1:
                continue
            mid = lo + (hi - lo) // 2
            df.loc[lo      : mid, "direction"] = "forward"
            df.loc[mid + 1 : hi , "direction"] = "backward"

            back_idx = range(mid + 1, hi + 1)
            x_back   = df.loc[back_idx, "Vsd (V)"]
            y_back   = df.loc[back_idx, "I (A)"]
            coeffs   = np.polyfit(x_back, y_back, poly_order)
            p        = np.poly1d(coeffs)

            seg_idx = range(lo, hi + 1)
            df.loc[seg_idx, "I_corr"] = (
                df.loc[seg_idx, "I (A)"] - p(df.loc[seg_idx, "Vsd (V)"])
            )

    # KEEP VOLTAGES PRESENT IN BOTH DIRECTIONS 
    shared_mask = df.groupby("Vsd (V)")["direction"].nunique().eq(2)
    df_shared   = df[df["Vsd (V)"].isin(shared_mask[shared_mask].index)]

    df_shared["I_corr"]    = df_shared["I_corr"].ffill().bfill()
    df_shared["direction"] = df_shared["direction"].ffill().bfill()

    # RETURN CLEAN I-V CURVE
    return (
        df_shared
        .drop(columns="I (A)") 
        .rename(columns={"I_corr": "I (A)"})
        .sort_values(["direction", "Vsd (V)"])
        .reset_index(drop=True)
    )
    
    
def process_partitioned_bg_corrected(
    props: pd.DataFrame,
    data: Dict[str, Callable],
) -> Dict[str, pd.DataFrame]:
    """
    Apply `normalize_background_current` to every IV partition listed in *props*.

    Parameters
    ----------
    props : pd.DataFrame
        Must contain at least the columns
            • "Procedure type"  (filter == "IV")
            • "data_key"        (partition name)
            • "VSD step"        (programmed step size in volts)
            • optional "poly_order" (override per run)
    data  : Dict[str, Callable]
        The Kedro partition: {data_key: lambda -> raw DataFrame}
    default_poly_order : int, default 7
        Used when *props* has no "poly_order" column.

    Returns
    -------
    Dict[str, pd.DataFrame]
        {data_key: baseline-corrected DataFrame with forward/backward rows}
    """
    processed = {}

    for _, row in props.iterrows():
        if row.get("Procedure type") != "IV":
            continue

        key        = row["data_key"]
        step_size  = row["VSD step"]

        raw_df = data[key]()           # load partition

        cleaned = normalize_background_current(
            raw_df,
            step_size=step_size,
            poly_order=7,
        )
        processed[key] = cleaned

    return processed



def get_maximum(
    df: pd.DataFrame,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Locate the peak hysteresis current on the negative- and positive-
    voltage sides and return both the current *and* the voltage where that
    peak occurs.

    Returns
    -------
    Tuple[
        (I_neg_max, V_at_I_neg_max),
        (I_pos_max, V_at_I_pos_max)
    ]
    NaNs are returned when the corresponding sweep segment is empty.
    """
    if df.empty:
        nan2 = (float("nan"), float("nan"))
        return nan2, nan2

    # split by sign of Vsd
    pos = df[df["Vsd (V)"] > 0]
    neg = df[df["Vsd (V)"] < 0]

    # ---- negative branch ---------------------------------------------
    if neg.empty:
        I_neg, V_neg = float("nan"), float("nan")
    else:
        idx_neg = neg["I (A)"].idxmax()          # row label of the max
        I_neg   = neg.at[idx_neg, "I (A)"]
        V_neg   = neg.at[idx_neg, "Vsd (V)"]

    # ---- positive branch ---------------------------------------------
    if pos.empty:
        I_pos, V_pos = float("nan"), float("nan")
    else:
        idx_pos = pos["I (A)"].idxmax()
        I_pos   = pos.at[idx_pos, "I (A)"]
        V_pos   = pos.at[idx_pos, "Vsd (V)"]

    return (I_neg, V_neg), (I_pos, V_pos)


def get_maximums_bg_partitioned(
    props: pd.DataFrame,
    data: Dict[str, Callable],
) -> pd.DataFrame:
    """
    Attach peak-current information (value *and* voltage) to every IV entry
    in a partitioned Kedro dataset.

    Parameters
    ----------
    props : pd.DataFrame
        Must contain at least "data_key" and "Procedure type".
    data  : Dict[str, Callable]
        {data_key: lambda → raw or cleaned DataFrame}.

    Returns
    -------
    pd.DataFrame
        `props` with four new columns:
            • max_neg_I , max_neg_V
            • max_pos_I , max_pos_V
    """
    results = []
    for _, row in props.iterrows():
        if row["Procedure type"] != "IV":
            continue

        key = row["data_key"]
        df  = data[key]()                    # load partition

        (I_neg, V_neg), (I_pos, V_pos) = get_maximum(df)

        results.append(
            {
                "data_key":   key,
                "max_neg_I":  I_neg,
                "max_neg_V":  V_neg,
                "max_pos_I":  I_pos,
                "max_pos_V":  V_pos,
            }
        )

    peaks_df = pd.DataFrame(results)
    return props.merge(peaks_df, on="data_key", how="left")