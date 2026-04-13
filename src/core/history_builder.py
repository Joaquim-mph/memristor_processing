"""
Build chip histories from staged manifest data.

This module creates chip history Parquet files from the manifest.parquet generated
during the staging process. Histories are saved to data/02_stage/chip_histories/ and include
sequential experiment numbers, dates, procedures, summaries, and paths to staged
Parquet measurement files.
"""

from pathlib import Path
from typing import Optional
import polars as pl
from datetime import datetime


def compute_parquet_path(stage_root: Path, proc: str, date_local: str, run_id: str) -> str:
    """
    Compute path to staged Parquet file from manifest columns.

    The staging system uses Hive-style partitioning:
    stage_root/proc={proc}/date={date_local}/run_id={run_id}/part-000.parquet

    Parameters
    ----------
    stage_root : Path
        Stage root directory (e.g., data/02_stage/raw_measurements)
    proc : str
        Procedure type (e.g., "IVg", "It")
    date_local : str
        Local date string (YYYY-MM-DD)
    run_id : str
        Run ID (16-char hash)

    Returns
    -------
    str
        Relative path to Parquet file from project root
    """
    return str(stage_root / f"proc={proc}" / f"date={date_local}" / f"run_id={run_id}" / "part-000.parquet")


def build_chip_history_from_manifest(
    manifest_path: Path,
    stage_root: Optional[Path] = None,
    chip_number: Optional[int] = None,
    chip_group: Optional[str] = None,
    information: Optional[str] = None,
    proc_filter: Optional[str] = None,
) -> pl.DataFrame:
    """
    Build experiment history for a specific chip from manifest.parquet.

    Filters manifest by chip identifier and returns chronologically ordered
    history with sequential experiment numbers and paths to staged Parquet files.

    Parameters
    ----------
    manifest_path : Path
        Path to manifest.parquet file
    stage_root : Path, optional
        Stage root directory for computing parquet paths
        (e.g., data/02_stage/raw_measurements). If None, tries to infer from manifest_path.
    chip_number : int, optional
        Chip numeric ID (e.g., 67 for Alisson67)
    chip_group : str, optional
        Chip group prefix (e.g., "Alisson")
    information : str, optional
        Information field to filter by (used when chip_number/group are missing)
    proc_filter : str, optional
        Filter by procedure type (e.g., "It", "IVg")

    Returns
    -------
    pl.DataFrame
        Chip history with columns: seq, date, time_hms, proc, summary, has_light, parquet_path, etc.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Infer stage_root from manifest_path if not provided
    # manifest_path is typically: data/02_stage/raw_measurements/_manifest/manifest.parquet
    if stage_root is None:
        stage_root = manifest_path.parent.parent  # Go up from _manifest/ to raw_measurements/

    # Load manifest rows
    # Both "ok" (freshly staged) and "skipped" (already existed) are valid
    df = (
        pl.read_parquet(manifest_path)
        .filter(pl.col("status").is_in(["ok", "skipped"]))
    )

    # Filter by chip identifier
    if chip_number is not None:
        df = df.filter(pl.col("chip_number") == chip_number)
        if chip_group:
            df = df.filter(pl.col("chip_group") == chip_group)
    elif information:
        # Use Information column for filtering
        if "information" in df.columns:
            df = df.filter(pl.col("information") == information)
        else:
            raise ValueError("Information column not found in manifest")
    else:
        raise ValueError("Must provide either (chip_number, chip_group) or information")

    # Apply procedure filter if specified
    if proc_filter:
        df = df.filter(pl.col("proc") == proc_filter)

    # Parse start_time_utc as datetime if it's a string
    if "start_time_utc" in df.columns and df["start_time_utc"].dtype == pl.Utf8:
        df = df.with_columns([
            pl.col("start_time_utc").str.to_datetime(
                format="%Y-%m-%d %H:%M:%S%.f%z",
                time_zone="UTC"
            ).alias("start_time_utc")
        ])

    # Sort by time
    df = df.sort("start_time_utc")

    # Add sequential experiment numbers
    df = df.with_row_count("seq", offset=1)

    # Extract date/time and derived timeline fields
    date_exprs = [
        pl.col("start_time_utc").dt.date().cast(pl.Utf8).alias("date"),
        pl.col("start_time_utc").dt.strftime("%H:%M:%S").alias("time_hms"),
        # Combined datetime label in local timezone (human-readable)
        pl.col("start_time_utc").dt.convert_time_zone("America/Santiago").dt.strftime("%Y-%m-%d %H:%M:%S").alias("datetime_local"),
        (pl.col("start_time_utc").dt.epoch(time_unit="us").cast(pl.Float64) / 1_000_000).alias("start_time"),
        pl.col("source_file").map_elements(
            lambda s: s.replace("\\", "/").split("/")[2] if len(s.replace("\\", "/").split("/")) > 2 else None,
            return_dtype=pl.Utf8
        ).alias("day_folder"),
    ]

    if "laser_voltage_V" in df.columns:
        date_exprs.append(pl.col("laser_voltage_V").cast(pl.Float64).alias("laser_voltage_v"))
    elif "laser_voltage_v" in df.columns:
        date_exprs.append(pl.col("laser_voltage_v").cast(pl.Float64).alias("laser_voltage_v"))

    if "wavelength_nm" in df.columns:
        date_exprs.append(pl.col("wavelength_nm").cast(pl.Float64).alias("wavelength_nm"))

    if "vds_v" in df.columns:
        date_exprs.append(pl.col("vds_v").cast(pl.Float64).alias("vds_v"))

    if "ids_a" in df.columns:
        date_exprs.append(pl.col("ids_a").cast(pl.Float64).alias("ids_a"))

    if "vg_fixed_v" in df.columns:
        date_exprs.append(pl.col("vg_fixed_v").cast(pl.Float64).alias("vg_fixed_v"))

    if "vg_start_v" in df.columns:
        date_exprs.append(pl.col("vg_start_v").cast(pl.Float64).alias("vg_start_v"))

    if "vg_end_v" in df.columns:
        date_exprs.append(pl.col("vg_end_v").cast(pl.Float64).alias("vg_end_v"))

    if "vg_step_v" in df.columns:
        date_exprs.append(pl.col("vg_step_v").cast(pl.Float64).alias("vg_step_v"))

    if "laser_period_s" in df.columns:
        date_exprs.append(pl.col("laser_period_s").cast(pl.Float64).alias("laser_period_s"))

    df = df.with_columns(date_exprs)

    # Generate chip_name if it doesn't exist
    if "chip_name" not in df.columns:
        if chip_number is not None and chip_group:
            df = df.with_columns([
                pl.lit(f"{chip_group}{chip_number}").alias("chip_name")
            ])
        elif information:
            import re
            cleaned = re.sub(r'[^\w\s-]', '', information)
            cleaned = re.sub(r'[-\s]+', '_', cleaned)
            df = df.with_columns([
                pl.lit(cleaned or "UnknownChip").alias("chip_name")
            ])

    # Extract file_idx from source_file if it doesn't exist
    if "file_idx" not in df.columns:
        df = df.with_columns([
            pl.col("source_file").str.extract(r'_(\d+)\.csv$', 1).cast(pl.Int64, strict=False).alias("file_idx")
        ])

    # Generate rich summary text
    chip_display = pl.when(
        pl.col("chip_group").is_not_null() & pl.col("chip_number").is_not_null()
    ).then(
        pl.col("chip_group") + pl.col("chip_number").cast(pl.Utf8)
    ).otherwise(
        pl.col("chip_name").fill_null("")
    )

    df = df.with_columns([
        pl.when(pl.col("has_light") == True)
        .then(pl.lit("💡"))
        .when(pl.col("has_light") == False)
        .then(pl.lit("🌙"))
        .otherwise(pl.lit("❔"))
        .alias("light_glyph"),
        chip_display.alias("chip_display"),
    ])

    snippet_exprs = [
        pl.when(pl.col("chip_display") != "")
        .then(pl.lit(" ") + pl.col("chip_display"))
        .otherwise(pl.lit(""))
        .alias("chip_snippet"),
    ]

    if "vds_v" in df.columns:
        snippet_exprs.append(
            pl.when(pl.col("vds_v").is_not_null())
            .then(pl.lit(" VDS=") + pl.col("vds_v").round(3).cast(pl.Utf8) + pl.lit(" V"))
            .otherwise(pl.lit(""))
            .alias("vds_snippet")
        )
    else:
        snippet_exprs.append(pl.lit("").alias("vds_snippet"))

    if "vg_start_v" in df.columns and "vg_end_v" in df.columns:
        if "vg_step_v" in df.columns:
            vg_step_suffix = (
                pl.when(pl.col("vg_step_v").is_not_null())
                .then(pl.lit(" (step ") + pl.col("vg_step_v").round(3).cast(pl.Utf8) + pl.lit(")"))
                .otherwise(pl.lit(""))
            )
        else:
            vg_step_suffix = pl.lit("")

        snippet_exprs.append(
            pl.when(pl.col("vg_start_v").is_not_null() & pl.col("vg_end_v").is_not_null())
            .then(
                pl.lit(" VG=")
                + pl.col("vg_start_v").round(2).cast(pl.Utf8)
                + pl.lit("→")
                + pl.col("vg_end_v").round(2).cast(pl.Utf8)
                + vg_step_suffix
            )
            .otherwise(pl.lit(""))
            .alias("vg_range_snippet")
        )
    else:
        snippet_exprs.append(pl.lit("").alias("vg_range_snippet"))

    if "vg_fixed_v" in df.columns:
        snippet_exprs.append(
            pl.when(pl.col("vg_fixed_v").is_not_null())
            .then(pl.lit(" VG=") + pl.col("vg_fixed_v").round(3).cast(pl.Utf8) + pl.lit(" V"))
            .otherwise(pl.lit(""))
            .alias("vg_fixed_snippet")
        )
    else:
        snippet_exprs.append(pl.lit("").alias("vg_fixed_snippet"))

    if "laser_voltage_v" in df.columns:
        snippet_exprs.append(
            pl.when(pl.col("laser_voltage_v").is_not_null())
            .then(pl.lit(" VL=") + pl.col("laser_voltage_v").round(3).cast(pl.Utf8) + pl.lit(" V"))
            .otherwise(pl.lit(""))
            .alias("laser_snippet")
        )
    else:
        snippet_exprs.append(pl.lit("").alias("laser_snippet"))

    if "wavelength_nm" in df.columns:
        snippet_exprs.append(
            pl.when(pl.col("wavelength_nm").is_not_null())
            .then(pl.lit(" λ=") + pl.col("wavelength_nm").round(1).cast(pl.Utf8) + pl.lit(" nm"))
            .otherwise(pl.lit(""))
            .alias("wavelength_snippet")
        )
    else:
        snippet_exprs.append(pl.lit("").alias("wavelength_snippet"))

    snippet_exprs.append(
        pl.when(pl.col("file_idx").is_not_null())
        .then(pl.lit(" #") + pl.col("file_idx").cast(pl.Utf8))
        .otherwise(pl.lit(""))
        .alias("file_idx_snippet")
    )

    df = df.with_columns(snippet_exprs)

    df = df.with_columns([
        pl.concat_str(
            [
                pl.col("light_glyph"),
                pl.lit(" "),
                pl.col("proc"),
                pl.col("chip_snippet"),
                pl.col("vds_snippet"),
                pl.col("vg_range_snippet"),
                pl.col("vg_fixed_snippet"),
                pl.col("laser_snippet"),
                pl.col("wavelength_snippet"),
                pl.col("file_idx_snippet"),
            ],
            separator=""
        ).alias("summary")
    ])

    # Add parquet_path column pointing to staged measurement data
    # Use Polars format string to construct the path
    if all(col in df.columns for col in ["proc", "date_local", "run_id"]):
        df = df.with_columns([
            pl.format(
                "{}/proc={}/date={}/run_id={}/part-000.parquet",
                pl.lit(str(stage_root)),
                pl.col("proc"),
                pl.col("date_local"),
                pl.col("run_id"),
            ).alias("parquet_path")
        ])

    # Select relevant columns for history
    history_cols = [
        "seq",
        "date",
        "time_hms",
        "datetime_local",  # Combined date+time in local timezone
        "proc",
        "summary",
        "has_light",
        "laser_voltage_v",
        "wavelength_nm",
        "vds_v",
        "ids_a",  # Drain-source current for VVg/Vt procedures
        "vg_fixed_v",
        "vg_start_v",
        "vg_end_v",
        "vg_step_v",
        "laser_period_s",
        "parquet_path",  # Path to staged Parquet measurement file
        "source_file",
        "chip_number",
        "chip_group",
        "chip_name",
        "run_id",
        "rows",
        "file_idx",
        "date_local",
        "day_folder",
        "start_time",
        "start_time_utc",
        "ingested_at_utc",
        "date_origin",
    ]

    # Only select columns that exist in the dataframe
    available_cols = [col for col in history_cols if col in df.columns]
    history = df.select(available_cols)

    # Drop helper columns that were only used for summary construction
    history = history.drop(
        [
            col
            for col in [
                "light_glyph",
                "chip_display",
                "chip_snippet",
                "vds_snippet",
                "vg_range_snippet",
                "vg_fixed_snippet",
                "laser_snippet",
                "wavelength_snippet",
                "file_idx_snippet",
            ]
            if col in history.columns
        ]
    )

    return history


def generate_chip_name(
    chip_number: Optional[int],
    chip_group: Optional[str],
    information: Optional[str],
) -> str:
    """
    Generate chip name for history file.

    Parameters
    ----------
    chip_number : int, optional
        Chip numeric ID
    chip_group : str, optional
        Chip group prefix
    information : str, optional
        Information field value

    Returns
    -------
    str
        Chip name for filename (e.g., "Alisson67" or "CustomChipName")
    """
    if chip_number is not None and chip_group:
        return f"{chip_group}{chip_number}"
    elif information:
        # Clean information string for filename
        import re
        cleaned = re.sub(r'[^\w\s-]', '', information)  # Remove special chars
        cleaned = re.sub(r'[-\s]+', '_', cleaned)  # Replace spaces/hyphens with underscore
        return cleaned or "UnknownChip"
    else:
        return "UnknownChip"


def save_chip_history(
    history: pl.DataFrame,
    output_dir: Path,
    chip_name: str,
) -> Path:
    """
    Save chip history to Parquet file.

    Parameters
    ----------
    history : pl.DataFrame
        Chip history dataframe
    output_dir : Path
        Output directory (e.g., data/02_stage/chip_histories)
    chip_name : str
        Chip name for filename

    Returns
    -------
    Path
        Path to saved history file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{chip_name}_history.parquet"
    history.write_parquet(output_path)
    return output_path


def generate_all_chip_histories(
    manifest_path: Path,
    output_dir: Path,
    stage_root: Optional[Path] = None,
    min_experiments: int = 5,
    chip_group: Optional[str] = None,
) -> dict[str, Path]:
    """
    Generate history files for all chips found in manifest.

    Automatically discovers all unique chips and creates individual
    history Parquet files in the output directory.

    Parameters
    ----------
    manifest_path : Path
        Path to manifest.parquet file
    output_dir : Path
        Output directory for history files (e.g., data/02_stage/chip_histories)
    stage_root : Path, optional
        Stage root directory for computing parquet paths.
        If None, tries to infer from manifest_path.
    min_experiments : int
        Minimum number of experiments required to generate history
    chip_group : str, optional
        Filter by specific chip group

    Returns
    -------
    dict[str, Path]
        Mapping of chip_name -> history_file_path
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Infer stage_root if not provided
    if stage_root is None:
        stage_root = manifest_path.parent.parent

    # Load manifest
    df = pl.read_parquet(manifest_path)

    # Filter by chip group if specified
    if chip_group:
        df = df.filter(pl.col("chip_group") == chip_group)

    # Discover unique chips
    chip_identifiers = []

    # Group by chip_number and chip_group (if available)
    if "chip_number" in df.columns and "chip_group" in df.columns:
        chip_groups = (
            df.filter(pl.col("chip_number").is_not_null())
            .group_by(["chip_number", "chip_group"])
            .agg(pl.count().alias("count"))
            .filter(pl.col("count") >= min_experiments)
        )

        for row in chip_groups.iter_rows(named=True):
            chip_identifiers.append({
                "chip_number": row["chip_number"],
                "chip_group": row["chip_group"],
                "information": None,
                "count": row["count"],
            })

    # Also find chips identified by Information field
    if "information" in df.columns:
        info_groups = (
            df.filter(
                (pl.col("chip_number").is_null()) | (pl.col("chip_group").is_null())
            )
            .filter(pl.col("information").is_not_null())
            .group_by("information")
            .agg(pl.count().alias("count"))
            .filter(pl.col("count") >= min_experiments)
        )

        for row in info_groups.iter_rows(named=True):
            chip_identifiers.append({
                "chip_number": None,
                "chip_group": None,
                "information": row["information"],
                "count": row["count"],
            })

    # Generate history for each chip
    histories = {}

    for chip_id in chip_identifiers:
        try:
            # Build history
            history = build_chip_history_from_manifest(
                manifest_path,
                stage_root=stage_root,
                chip_number=chip_id["chip_number"],
                chip_group=chip_id["chip_group"],
                information=chip_id["information"],
            )

            # Generate chip name
            chip_name = generate_chip_name(
                chip_id["chip_number"],
                chip_id["chip_group"],
                chip_id["information"],
            )

            # Save history
            output_path = save_chip_history(history, output_dir, chip_name)
            histories[chip_name] = output_path

        except Exception as e:
            # Log error but continue with other chips
            print(f"Warning: Failed to generate history for {chip_id}: {e}")
            continue

    return histories


def print_chip_history(
    manifest_path: Path,
    chip_number: Optional[int] = None,
    chip_group: Optional[str] = None,
    information: Optional[str] = None,
    proc_filter: Optional[str] = None,
    max_rows: int = 50,
) -> None:
    """
    Print chip history to console (for debugging/quick viewing).

    Parameters
    ----------
    manifest_path : Path
        Path to manifest.parquet file
    chip_number : int, optional
        Chip numeric ID
    chip_group : str, optional
        Chip group prefix
    information : str, optional
        Information field value
    proc_filter : str, optional
        Filter by procedure type
    max_rows : int
        Maximum rows to display
    """
    history = build_chip_history_from_manifest(
        manifest_path,
        chip_number=chip_number,
        chip_group=chip_group,
        information=information,
        proc_filter=proc_filter,
    )

    chip_name = generate_chip_name(chip_number, chip_group, information)

    print(f"\n{'='*80}")
    print(f"Chip History: {chip_name}")
    print(f"Total Experiments: {len(history)}")
    print(f"{'='*80}\n")

    # Display with polars
    if len(history) > max_rows:
        print(f"Showing first {max_rows} of {len(history)} experiments:\n")
        print(history.head(max_rows))
    else:
        print(history)

    print(f"\n{'='*80}\n")


def get_chip_summary_stats(manifest_path: Path, chip_name: str) -> dict:
    """
    Get summary statistics for a chip.

    Parameters
    ----------
    manifest_path : Path
        Path to manifest.parquet file
    chip_name : str
        Chip name (e.g., "Alisson67")

    Returns
    -------
    dict
        Summary statistics including total experiments, date range, procedures
    """
    # Parse chip name to extract number and group
    import re
    match = re.match(r"^([A-Za-z]+)(\d+)$", chip_name)

    if match:
        chip_group = match.group(1)
        chip_number = int(match.group(2))
        history = build_chip_history_from_manifest(
            manifest_path,
            chip_number=chip_number,
            chip_group=chip_group,
        )
    else:
        # Try as information field
        history = build_chip_history_from_manifest(
            manifest_path,
            information=chip_name,
        )

    if len(history) == 0:
        return {
            "total_experiments": 0,
            "date_range": None,
            "procedures": {},
        }

    # Calculate statistics
    stats = {
        "total_experiments": len(history),
        "date_range": (
            history["date"].min(),
            history["date"].max(),
        ) if "date" in history.columns else None,
        "procedures": {},
    }

    # Count by procedure
    if "proc" in history.columns:
        proc_counts = history.group_by("proc").agg(pl.count().alias("count"))
        stats["procedures"] = {
            row["proc"]: row["count"]
            for row in proc_counts.iter_rows(named=True)
        }

    return stats
