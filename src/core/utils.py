from __future__ import annotations
import re
from pathlib import Path
from typing import Dict
import polars as pl

# -------------------------------
# Small helpers
# -------------------------------
def _file_index(p: str) -> int:
    """Extract the trailing _NN.csv number as an int (for ordering)."""
    m = re.search(r"_(\d+)\.csv$", p)
    return int(m.group(1)) if m else -1

def _proc_from_path(p: str) -> str:
    """Infer procedure from path."""
    name = p.lower()
    if "/ivg" in name:
        return "IVg"
    if "/it" in name:
        return "ITS"
    if "/iv" in name:
        return "IV"
    return "OTHER"

def _find_data_start(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return 0

    data_pat = re.compile(r"^\s*#?\s*Data\s*:\s*$", re.IGNORECASE)
    for i, line in enumerate(lines):
        if data_pat.match(line):
            return i + 1  # header is the next non-empty, non-comment line

    # Fallback: first CSV-ish line that looks like a real header
    for i, line in enumerate(lines):
        s = line.strip()
        if "," in s and any(t in s.lower() for t in ("vg", "vsd", "vds", "i", "t (", "t,")):
            return i

    return 0

def _std_rename(cols: list[str]) -> Dict[str, str]:
    """Standardize typical column names (units/spacing/case-insensitive)."""
    mapping = {}
    for c in cols:
        s = c.strip()
        s = re.sub(r"\(.*?\)", "", s)   # drop units like (V), (A)
        s = s.replace("degC", "")
        s = re.sub(r"\s+", " ", s).strip()
        s_low = s.lower()

        if s_low in {"vg", "gate", "gate v", "gate voltage"}:
            mapping[c] = "VG"
        elif s_low in {"vsd", "vds", "drain-source", "drain source", "v"}:
            mapping[c] = "VSD"
        elif s_low in {"i", "id", "current"}:
            mapping[c] = "I"
        elif s_low in {"t", "time", "t s"}:
            mapping[c] = "t"
        elif s_low in {"vl", "laser", "laser v"}:
            mapping[c] = "VL"
        else:
            mapping[c] = c  # keep as-is
    return mapping

def _read_measurement(path: Path) -> pl.DataFrame:
    import io

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return pl.DataFrame()

    start = _find_data_start(path)

    # find first non-empty, non-comment line after start -> header
    j = start
    while j < len(lines) and (lines[j].strip() == "" or lines[j].lstrip().startswith("#")):
        j += 1
    if j >= len(lines):
        return pl.DataFrame()

    header_line = lines[j].lstrip("\ufeff").strip()
    if "," not in header_line:
        # fallback: scan down to first CSV-like line
        k = j + 1
        while k < len(lines) and ("," not in lines[k]):
            k += 1
        if k >= len(lines):
            return pl.DataFrame()
        header_line = lines[k].strip()
        j = k

    header_cols = [h.strip() for h in header_line.split(",")]
    body_lines = lines[j + 1 :]

    # truncate/pad each row to the header length, skip comment/blank rows
    cleaned = []
    n = len(header_cols)
    for line in body_lines:
        row = line.rstrip("\n\r")
        if not row or row.lstrip().startswith("#"):
            continue
        parts = row.split(",")
        if len(parts) < n:
            parts += [""] * (n - len(parts))
        elif len(parts) > n:
            parts = parts[:n]
        cleaned.append(",".join(parts))

    if not cleaned:
        return pl.DataFrame()

    buf = io.StringIO(",".join(header_cols) + "\n" + "\n".join(cleaned))
    df = pl.read_csv(
        buf,
        infer_schema_length=5000,
        null_values=["", "nan", "NaN"],
        ignore_errors=True,
    )

    # Standardize names and coerce numerics
    df = df.rename(_std_rename(df.columns))
    for col in ("VG", "VSD", "I", "t", "VL"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # Drop all-null columns
    df = df.select([c for c in df.columns if not df[c].is_null().all()])
    return df



# -------------------------------
# Make timeline + sessions
# -------------------------------
def read_measurement_parquet(path: Path) -> pl.DataFrame:
    """
    Read measurement data from staged Parquet file.

    This is the modern replacement for _read_measurement() that reads from
    schema-validated Parquet files in the staging area instead of raw CSVs.

    Parameters
    ----------
    path : Path
        Path to staged Parquet file (e.g., data/02_stage/raw_measurements/proc=It/date=2025-10-18/run_id=abc123/part-000.parquet)

    Returns
    -------
    pl.DataFrame
        Measurement data with standardized column names

    Notes
    -----
    Parquet files from the staging system already have:
    - Schema validation
    - Correct types
    - Standardized column names
    - No header parsing needed

    Example
    -------
    >>> path = Path("data/02_stage/raw_measurements/proc=It/date=2025-10-18/run_id=a1b2c3d4/part-000.parquet")
    >>> df = read_measurement_parquet(path)
    >>> print(df.columns)
    ['t (s)', 'I (A)', 'VL (V)', ...]
    """

    try:
        df = pl.read_parquet(path)
        
        # --- NEW: Join metadata from manifest ---
        # The parquet file now only contains RunID and data. 
        # We must look up the metadata (wavelength, chip, etc.) from the manifest.
        if "run_id" in df.columns:
            # 1. Infer manifest path (go up from .../proc=X/date=Y/run_id=Z/part-000.parquet)
            # data/02_stage/raw_measurements/_manifest/manifest.parquet
            try:
                # Path(path) is like: roots/proc=X/date=Y/run_id=Z/part-000.parquet
                # Parent x3 is 'roots' (raw_measurements)
                stage_root = Path(path).parent.parent.parent.parent
                manifest_path = stage_root / "_manifest" / "manifest.parquet"
                
                if manifest_path.exists():
                    # Read only the row for this run_id for efficiency
                    # But Polars scan_parquet doesn't support fast point lookup well without reading metadata
                    # For now, let's cache the manifest or use scan
                    # To keep it simple and robust:
                    
                    # We only need the metadata columns, not everything
                    # And we filter by the run_id found in the data (should be single value)
                    rid = df["run_id"][0]
                    
                    manifest_df = pl.read_parquet(manifest_path).filter(pl.col("run_id") == rid)
                    
                    if len(manifest_df) > 0:
                        meta_row = manifest_df.row(0, named=True)
                        
                        # Columns to exclude from join (already in data or not needed)
                        exclude = {"run_id", "proc", "path", "source_file", "rows", "status", "ingested_at_utc", "validation_errors", "validation_warnings", "validation_messages"}
                        
                        # Add metadata columns to df
                        new_cols = []
                        for k, v in meta_row.items():
                            if k not in df.columns and k not in exclude and v is not None:
                                new_cols.append(pl.lit(v).alias(k))
                        
                        if new_cols:
                            df = df.with_columns(new_cols)
            except Exception as e:
                # If manifest lookup fails, just return what we have (graceful degradation)
                # print(f"Warning: Failed to join manifest metadata: {e}")
                pass

        return df
    except FileNotFoundError:
        return pl.DataFrame()
    except Exception as e:
        print(f"Warning: Failed to read Parquet file {path}: {e}")
        return pl.DataFrame()


def load_and_prepare_metadata(meta_csv: str, chip: float) -> pl.DataFrame:
    df = pl.read_csv(meta_csv, infer_schema_length=1000)
    # Normalize column names we will use often
    df = df.rename({"Chip number": "Chip number",
                    "Laser voltage": "Laser voltage",
                    "Laser toggle": "Laser toggle",
                    "source_file": "source_file"})

    # Filter chip
    df = df.filter(pl.col("Chip number") == chip)

    # Infer procedure and index
    df = df.with_columns([
        pl.col("source_file").map_elements(_proc_from_path).alias("proc"),
        pl.col("source_file").map_elements(_file_index).alias("file_idx"),
        pl.when(pl.col("Laser toggle").cast(pl.Utf8).str.to_lowercase() == "true")
          .then(pl.lit(True))
          .otherwise(pl.lit(False))
          .alias("with_light"),
        pl.col("Laser voltage").cast(pl.Float64).alias("VL_meta"),
        pl.col("VG").cast(pl.Float64).alias("VG_meta").fill_null(strategy="zero")
    ]).sort("file_idx")

    # Build sessions = [IVg → ITS… → IVg] blocks
    # We'll assign the *closing* IVg to the same session as the preceding ITS.
    session_id = 0
    seen_any_ivg = False
    seen_its_since_ivg = False
    ids = []
    roles = []

    for proc in df["proc"].to_list():
        if proc == "IVg":
            if not seen_any_ivg:
                # first IVg starts session
                session_id += 1
                seen_any_ivg = True
                seen_its_since_ivg = False
                roles.append("pre_ivg")
                ids.append(session_id)
            else:
                if seen_its_since_ivg:
                    # this IVg closes the existing session
                    roles.append("post_ivg")
                    ids.append(session_id)
                    # next block will start on the next IVg or ITS as needed
                    seen_its_since_ivg = False
                    seen_any_ivg = False  # force new session on next IVg/ITS
                else:
                    # back-to-back IVg → treat as a new session pre_ivg
                    session_id += 1
                    roles.append("pre_ivg")
                    ids.append(session_id)
                    seen_its_since_ivg = False
        elif proc == "ITS":
            if not seen_any_ivg:
                # ITS without a prior IVg — start a new session
                session_id += 1
                seen_any_ivg = True
            roles.append("its")
            ids.append(session_id)
            seen_its_since_ivg = True
        else:
            roles.append("other")
            ids.append(session_id)

    df = df.with_columns([
        pl.Series("session", ids),
        pl.Series("role", roles),
    ])
    return df
