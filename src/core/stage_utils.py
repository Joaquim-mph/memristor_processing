from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import polars as pl
import yaml
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

# ----------------------------- Config -----------------------------
DEFAULT_LOCAL_TZ = "America/Santiago"
DEFAULT_WORKERS = 6
DEFAULT_POLARS_THREADS = 1

PROC_LINE_RE = re.compile(r"^#\s*Procedure\s*:\s*<([^>]+)>\s*$", re.I)
PARAMS_LINE_RE = re.compile(r"^#\s*Parameters\s*:\s*$", re.I)
META_LINE_RE = re.compile(r"^#\s*Metadata\s*:\s*$", re.I)
DATA_LINE_RE = re.compile(r"^#\s*Data\s*:\s*$", re.I)
KV_PAT = re.compile(r"^#\s*([^:]+):\s*(.*)\s*$")


def warn(msg: str) -> None:
    """
    Output a warning message to standard error.
    
    Args:
        msg: The warning message to display
        
    Example:
        >>> warn("Configuration file not found")
        [warn] Configuration file not found
    """
    print(f"[warn] {msg}", file=sys.stderr)


def sha1_short(s: str, n: int = 16) -> str:
    """
    Generate a short SHA-1 hash of a string.
    
    Creates a hexadecimal hash identifier useful for generating unique IDs,
    caching keys, or content fingerprints.
    
    Args:
        s: Input string to hash
        n: Number of characters to return from the hash (default: 16)
        
    Returns:
        The first n characters of the SHA-1 hash in hexadecimal format
        
    Example:
        >>> sha1_short("my_procedure")
        'a1b2c3d4e5f6g7h8'
        >>> sha1_short("test", n=8)
        'a94a8fe5'
    """
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def to_bool(s: Any) -> bool:
    """
    Convert various input types to boolean values.
    
    Recognizes common truthy values: "1", "true", "yes", "on", "y"
    (case-insensitive). All other values are considered False.
    
    Args:
        s: Value to convert (can be string, int, or any type)
        
    Returns:
        True if the normalized value is in the truthy set, False otherwise
        
    Example:
        >>> to_bool("YES")
        True
        >>> to_bool("1")
        True
        >>> to_bool("false")
        False
        >>> to_bool(0)
        False
    """
    return str(s).strip().lower() in {"1", "true", "yes", "on", "y"}


def parse_number_unit(s: Any) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value and unit from a string.
    
    Parses strings containing a number followed by an optional unit,
    supporting scientific notation. Useful for parsing measurements,
    durations, or data sizes.
    
    Args:
        s: Input value (string, number, or None)
        
    Returns:
        Tuple of (numeric_value, unit_string). Returns (None, None) if
        parsing fails. If input is a pure number, unit will be None.
        
    Example:
        >>> parse_number_unit("100ms")
        (100.0, 'ms')
        >>> parse_number_unit("5.2GB")
        (5.2, 'GB')
        >>> parse_number_unit(42)
        (42.0, None)
        >>> parse_number_unit("1.5e-3kg")
        (0.0015, 'kg')
        >>> parse_number_unit("invalid")
        (None, None)
    """
    if s is None:
        return None, None
    if isinstance(s, (int, float)):
        return float(s), None
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([^\s]+)?\s*$", str(s))
    if not m:
        return None, None
    return float(m.group(1)), m.group(2)


def parse_datetime_any(x: Any) -> Optional[dt.datetime]:
    """
    Parse datetime from multiple input formats.
    
    Handles Unix timestamps (as int/float/string) and ISO 8601 formatted
    strings. All outputs are normalized to UTC timezone.
    
    Args:
        x: Input value (Unix timestamp, ISO string, or None)
        
    Returns:
        datetime object in UTC timezone, or None if parsing fails
        
    Example:
        >>> parse_datetime_any(1609459200)
        datetime.datetime(2021, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        >>> parse_datetime_any("2021-01-01T00:00:00Z")
        datetime.datetime(2021, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        >>> parse_datetime_any("1609459200")
        datetime.datetime(2021, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        >>> parse_datetime_any("invalid")
        None
        
    Note:
        - If input has no timezone, UTC is assumed
        - If input has timezone, it's converted to UTC
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc)
        except Exception:
            return None
    s = str(x).strip()
    try:
        return dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc)
    except Exception:
        pass
    try:
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        return None


def local_date_for_partition(ts_utc: dt.datetime, tz_name: str) -> str:
    """
    Convert UTC datetime to local date string for data partitioning.
    
    Useful for organizing data by business date rather than UTC date,
    ensuring events are grouped by the date they occurred in local time.
    
    Args:
        ts_utc: UTC datetime to convert
        tz_name: IANA timezone name (e.g., "America/New_York")
        
    Returns:
        ISO format date string (YYYY-MM-DD) in the local timezone
        
    Example:
        >>> from datetime import datetime, timezone
        >>> utc_time = datetime(2025, 10, 4, 3, 0, 0, tzinfo=timezone.utc)
        >>> local_date_for_partition(utc_time, "America/New_York")
        '2025-10-03'
        >>> local_date_for_partition(utc_time, "Asia/Tokyo")
        '2025-10-04'
        
    Note:
        Falls back to UTC date if timezone conversion fails or
        ZoneInfo is not available.
    """
    if tz_name and ZoneInfo is not None:
        try:
            return ts_utc.astimezone(ZoneInfo(tz_name)).date().isoformat()
        except Exception:
            pass
    return ts_utc.date().isoformat()


def ensure_dir(p: Path) -> None:
    """
    Create a directory and all parent directories if they don't exist.
    
    This is an idempotent operation - safe to call multiple times.
    Does nothing if the directory already exists.
    
    Args:
        p: Path object representing the directory to create
        
    Example:
        >>> ensure_dir(Path("/data/2025/10/04"))
        # Creates /data/, /data/2025/, /data/2025/10/, and /data/2025/10/04/
        
    Note:
        Uses exist_ok=True to prevent errors if directory exists.
    """
    p.mkdir(parents=True, exist_ok=True)


def extract_date_from_path(p: Path) -> Optional[str]:
    """
    Extract date string from file path or filename.
    
    Searches for dates in two formats:
    1. Delimited: YYYY-MM-DD or YYYY_MM_DD
    2. Compact: YYYYMMDD
    
    Validates that extracted components form a valid date before returning.
    
    Args:
        p: Path object to search for date patterns
        
    Returns:
        ISO format date string (YYYY-MM-DD) if found and valid, None otherwise
        
    Example:
        >>> extract_date_from_path(Path("/data/2025-10-04/file.csv"))
        '2025-10-04'
        >>> extract_date_from_path(Path("/data/20251004_export.csv"))
        '2025-10-04'
        >>> extract_date_from_path(Path("/data/results_2025_10_04.parquet"))
        '2025-10-04'
        >>> extract_date_from_path(Path("/data/2025-13-99/file.csv"))
        None
        >>> extract_date_from_path(Path("/data/file.csv"))
        None
        
    Note:
        Only returns the first valid date found in the path.
    """
    s = str(p)
    m = re.search(r"\b(\d{4})[-_](\d{2})[-_](\d{2})\b", s)
    if m:
        y, mo, d = m.groups()
        try:
            dt.date(int(y), int(mo), int(d))
            return f"{y}-{mo}-{d}"
        except Exception:
            pass
    m = re.search(r"\b(\d{8})\b", s)
    if m:
        raw = m.group(1)
        y, mo, d = raw[:4], raw[4:6], raw[6:8]
        try:
            dt.date(int(y), int(mo), int(d))
            return f"{y}-{mo}-{d}"
        except Exception:
            pass
    return None

def extract_date_from_path(p: Path) -> Optional[str]:
    s = str(p)
    m = re.search(r"\b(\d{4})[-_](\d{2})[-_](\d{2})\b", s)
    if m:
        y, mo, d = m.groups()
        try:
            dt.date(int(y), int(mo), int(d))
            return f"{y}-{mo}-{d}"
        except Exception:
            pass
    m = re.search(r"\b(\d{8})\b", s)
    if m:
        raw = m.group(1)
        y, mo, d = raw[:4], raw[4:6], raw[6:8]
        try:
            dt.date(int(y), int(mo), int(d))
            return f"{y}-{mo}-{d}"
        except Exception:
            pass
    return None