"""
Preprocessing pipeline for hysteresis and I-V characterization data.

This module provides nodes used in the 'preprocessing' pipeline of the Kedro project.
It includes functionality for:
- Separating forward and backward voltage sweeps from I-V measurements,
- Computing hysteresis loops by subtracting forward and backward curves,
- Partitioning and processing hysteresis datasets across multiple experiments,
- Extracting key metrics such as maximum positive and negative hysteresis.

These transformations standardize the raw measurement data for further analysis,
visualization, and model input preparation.

Developed for use with SnS-based memristive device characterization experiments.

Generated and adapted using Kedro 0.19.11
"""

import pandas as pd
import numpy as np
from typing import  Tuple, Dict, List, Callable


## Tomás:

def forward_minus_backward(df: pd.DataFrame, dv_threshold: float = 0.05) -> pd.DataFrame:
    """
    Computes the difference between forward and backward current sweeps in a hysteresis loop.

    This function identifies forward and backward sweeps based on the change in source-drain voltage (`Vsd (V)`).
    It averages the current values at each unique `Vsd` point in both sweeps, aligns the forward and backward
    curves, and returns the resulting differential current as a new DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing columns "Vsd (V)" and "I (A)", representing source-drain voltage and measured current.
        The data should include both forward and backward voltage sweeps in sequence.
    
    dv_threshold : float, optional
        The minimum voltage step required to classify a change in voltage as part of a forward or backward sweep.
        Default is 0.05 V.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - "Vsd (V)": Rounded voltage values shared between forward and backward sweeps.
        - "I (A)": The difference in current between forward and backward sweeps at each `Vsd`, i.e.,
                   I_forward - I_backward.

    Notes
    -----
    - The function attempts to patch both curves to ensure overlap at the sweep edges by adding a point from
      the opposite direction's boundary.
    - The function handles grouping and averaging automatically for repeated `Vsd` values.
    """
    
    df = df.copy()
    df["Vsd (V)"] = df["Vsd (V)"].round(2)
    df["direction"] = np.nan

    dv = df["Vsd (V)"].diff()
    df.loc[dv > dv_threshold, "direction"] = "forward"
    df.loc[dv < -dv_threshold, "direction"] = "backward"

    # Fill first values
    df["direction"] = df["direction"].ffill().bfill()

    forward = df[df["direction"] == "forward"].drop(columns="direction").copy()
    backward = df[df["direction"] == "backward"].drop(columns="direction").copy()

    forward.sort_values("Vsd (V)", inplace=True)
    backward.sort_values("Vsd (V)", inplace=True)

    # Patch one point at ends if needed
    if not backward.empty:
        forward = pd.concat([backward.iloc[[0]], forward], ignore_index=True)
    if not forward.empty:
        backward = pd.concat([backward, forward.iloc[[-1]]], ignore_index=True)

    # Group and average
    forward_grouped = forward.groupby("Vsd (V)", as_index=False).mean()
    backward_grouped = backward.groupby("Vsd (V)", as_index=False).mean()

    # Merge on Vsd (V) to ensure alignment
    merged = pd.merge(forward_grouped, backward_grouped, on="Vsd (V)", suffixes=("_fwd", "_bwd"))

    merged["I (A)"] = merged["I (A)_fwd"] - merged["I (A)_bwd"]
    return merged[["Vsd (V)", "I (A)"]]



def process_partitioned_hysteresis(props: pd.DataFrame, data: Dict[str, Callable]) -> Dict[str, pd.DataFrame]:
    """
    Process the partitioned hysteresis data.

    Parameters
    ----------
    props : pd.DataFrame
        A DataFrame containing the properties of the hysteresis data.
    data : Dict[str, Callable]
        A dictionary containing the partitioned hysteresis data.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary containing the processed hysteresis data.
    """
    
    processed_data = {}
    for _, row in props.iterrows():
        if row["Procedure type"] == "IV":
            key = row["data_key"]
            df = data[key]()
            dv_threshold = row["VSD step"] / 2
            processed_data[key] = forward_minus_backward(df, dv_threshold)
    return processed_data


def get_maximum(df: pd.DataFrame) -> Tuple[float]:
    """
    This function gets the maximum value of the hysteresis in the positive side and the negative side.
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the hysteresis data.

    Returns
    -------
    Tuple[float]
        A tuple containing the maximum value of the hysteresis in the negative side and the positive side, respectively.
    """
    
    positive = df.loc[df["Vsd (V)"] > 0, "I (A)"]
    negative = df.loc[df["Vsd (V)"] < 0, "I (A)"]
    return negative.max(), positive.max()



def get_maximums_hysteresis_partitioned(props: pd.DataFrame, data: Dict[str, Callable]) -> pd.DataFrame:
    props = props.copy()
    mask_iv = props["Procedure type"] == "IV"
    keys = props.loc[mask_iv, "data_key"]

    max_neg = []
    max_pos = []

    for key in keys:
        df = data[key]()
        neg, pos = get_maximum(df)
        max_neg.append(neg)
        max_pos.append(pos)

    props.loc[mask_iv, "max_negative_hysteresis"] = max_neg
    props.loc[mask_iv, "max_positive_hysteresis"] = max_pos

    return props



