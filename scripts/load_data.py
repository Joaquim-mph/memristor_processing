"""
Lightweight data loader for raw IV measurement CSVs.

Replaces the Kedro catalog loading (properties_project_joaco / data_project_joaco)
with direct CSV parsing.  Returns a metadata DataFrame and a data dictionary
keyed by relative path, matching the notebook convention.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl


# ── Header parsing (mirrors src/core/stage_raw_measurements.parse_header) ──

_PROC_RE = re.compile(r"^#\s*Procedure:\s*<(.+)>")
_PARAMS_RE = re.compile(r"^#\s*Parameters:\s*$")
_META_RE = re.compile(r"^#\s*Metadata:\s*$")
_DATA_RE = re.compile(r"^#\s*Data:\s*$")
_KV_RE = re.compile(r"^#\s*\t(.+?):\s*(.+)$")


def _parse_header(path: Path) -> Tuple[Dict[str, str], int]:
    """Return (params_dict, data_start_line) from a measurement CSV."""
    params: Dict[str, str] = {}
    data_line = 0
    mode: Optional[str] = None

    with path.open("r", errors="ignore") as f:
        for i, raw in enumerate(f):
            s = raw.rstrip("\n").strip()
            if _DATA_RE.match(s):
                data_line = i + 1
                break
            if _PARAMS_RE.match(s):
                mode = "params"
                continue
            if _META_RE.match(s):
                mode = "meta"
                continue

            m = _KV_RE.match(raw.rstrip("\n"))
            if m and mode in ("params", "meta"):
                key, val = m.group(1).strip(), m.group(2).strip()
                params[key] = val

    return params, data_line


def _safe_float(s: str) -> Optional[float]:
    """Extract leading number from strings like '0.001 A' or '8 V'."""
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", s)
    return float(m.group(1)) if m else None


# ── Public API ──


def load_experiment_folder(
    folder: Path,
    *,
    recursive: bool = True,
) -> Tuple[pl.DataFrame, Dict[str, pl.DataFrame]]:
    """
    Load all IV CSVs from *folder* and return metadata + data.

    Parameters
    ----------
    folder : Path
        Directory containing ``IV*.csv`` files (searched recursively).
    recursive : bool
        If True, search subdirectories too.

    Returns
    -------
    props : pl.DataFrame
        One row per CSV with columns: data_key, Information, Laser wavelength,
        Laser voltage, VSD start, VSD end, VSD step, Start time, Irange, ...
    data : Dict[str, pl.DataFrame]
        ``{data_key: DataFrame}`` where each DataFrame has at least
        ``Vsd (V)`` and ``I (A)`` columns.
    """
    pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = sorted(folder.glob(pattern))

    rows = []
    data_dict: Dict[str, pl.DataFrame] = {}

    for csv_path in csv_files:
        if not csv_path.name.startswith("IV"):
            continue

        params, skip_rows = _parse_header(csv_path)
        data_key = str(csv_path.relative_to(folder))

        # Read the data portion
        try:
            df = pl.read_csv(csv_path, skip_rows=skip_rows)
        except Exception:
            continue

        if "Vsd (V)" not in df.columns or "I (A)" not in df.columns:
            continue

        data_dict[data_key] = df

        # Build metadata row
        row = {
            "data_key": data_key,
            "source_path": str(csv_path),
            "Information": params.get("Information", ""),
            "Laser wavelength": _safe_float(params.get("Laser wavelength", "0")) or 0.0,
            "Laser voltage": _safe_float(params.get("Laser voltage", "0")) or 0.0,
            "VSD start": _safe_float(params.get("VSD start", "0")) or 0.0,
            "VSD end": _safe_float(params.get("VSD end", "0")) or 0.0,
            "VSD step": _safe_float(params.get("VSD step", "0.1")) or 0.1,
            "Start time": _safe_float(params.get("Start time", "0")) or 0.0,
            "Irange": _safe_float(params.get("Irange", "0")) or 0.0,
            "NPLC": _safe_float(params.get("NPLC", "1")) or 1.0,
            "Burn-in time": _safe_float(params.get("Burn-in time", "0")) or 0.0,
            "Step time": _safe_float(params.get("Step time", "0")) or 0.0,
            "Procedure type": "IV",
        }
        rows.append(row)

    if not rows:
        props = pl.DataFrame(
            schema={
                "data_key": pl.Utf8,
                "source_path": pl.Utf8,
                "Information": pl.Utf8,
                "Laser wavelength": pl.Float64,
                "Laser voltage": pl.Float64,
                "VSD start": pl.Float64,
                "VSD end": pl.Float64,
                "VSD step": pl.Float64,
                "Start time": pl.Float64,
                "Irange": pl.Float64,
                "NPLC": pl.Float64,
                "Burn-in time": pl.Float64,
                "Step time": pl.Float64,
                "Procedure type": pl.Utf8,
            }
        )
    else:
        props = pl.DataFrame(rows)

    # Sort by start time
    props = props.sort("Start time")

    return props, data_dict


def load_all_raw(
    raw_root: Path = Path("data/01_raw"),
) -> Tuple[pl.DataFrame, Dict[str, pl.DataFrame]]:
    """
    Load every IV CSV under ``data/01_raw/`` into a single props + data pair.

    The ``data_key`` is ``<folder_name>/<filename>`` so it can be filtered by
    folder prefix just like the notebooks do with ``date_prefix``.
    """
    all_rows = []
    all_data: Dict[str, pl.DataFrame] = {}

    for subfolder in sorted(raw_root.iterdir()):
        if not subfolder.is_dir() or subfolder.name.startswith("."):
            continue

        props, data = load_experiment_folder(subfolder, recursive=False)

        # Prefix data_key with folder name
        for row in props.iter_rows(named=True):
            old_key = row["data_key"]
            new_key = f"{subfolder.name}/{old_key}"
            new_row = dict(row)
            new_row["data_key"] = new_key
            all_rows.append(new_row)

            if old_key in data:
                all_data[new_key] = data[old_key]

    if not all_rows:
        return pl.DataFrame(), {}

    all_props = pl.DataFrame(all_rows).sort("Start time")
    return all_props, all_data


def load_calibrations(
    raw_root: Path = Path("data/01_raw"),
) -> Tuple[pl.DataFrame, Dict[str, pl.DataFrame]]:
    """
    Load all LaserCalibration CSVs under *raw_root*.

    Returns
    -------
    cal_props : pl.DataFrame
        One row per calibration file with columns: data_key, source_path,
        Laser wavelength, Optical fiber, Start time, ...
    cal_data : Dict[str, pl.DataFrame]
        {data_key: DataFrame} where each has columns "VL (V)" and "Power (W)".
    """
    all_rows: list[dict] = []
    all_data: Dict[str, pl.DataFrame] = {}

    for subfolder in sorted(raw_root.iterdir()):
        if not subfolder.is_dir() or subfolder.name.startswith("."):
            continue

        for csv_path in sorted(subfolder.glob("LaserCalibration*.csv")):
            params, skip_rows = _parse_header(csv_path)

            try:
                df = pl.read_csv(csv_path, skip_rows=skip_rows)
            except Exception:
                continue

            if "VL (V)" not in df.columns or "Power (W)" not in df.columns:
                continue

            data_key = f"{subfolder.name}/{csv_path.name}"
            all_data[data_key] = df

            row = {
                "data_key": data_key,
                "source_path": str(csv_path),
                "Laser wavelength": _safe_float(params.get("Laser wavelength", "0")) or 0.0,
                "Optical fiber": params.get("Optical fiber", ""),
                "Start time": _safe_float(params.get("Start time", "0")) or 0.0,
            }
            all_rows.append(row)

    if not all_rows:
        return pl.DataFrame(), {}

    cal_props = pl.DataFrame(all_rows).sort("Start time")
    return cal_props, all_data


def filter_experiments(
    props: pl.DataFrame,
    *,
    folder_prefix: Optional[str] = None,
    information: Optional[str] = None,
    wavelength: Optional[float] = None,
    vsd_end: Optional[float] = None,
) -> pl.DataFrame:
    """
    Filter metadata table by common criteria.

    Mirrors the notebook ``filter_experiments()`` function.
    """
    df = props.clone()

    if wavelength is not None:
        df = df.filter(pl.col("Laser wavelength") == wavelength)
    if vsd_end is not None:
        df = df.filter(pl.col("VSD end") == vsd_end)
    if information is not None:
        df = df.filter(pl.col("Information") == information)
    if folder_prefix is not None:
        df = df.filter(pl.col("data_key").str.starts_with(folder_prefix))

    return df.sort("Laser voltage")
