"""
CLI plotting helpers for terminal-based plot generation.

This module provides helper functions for the CLI plotting commands
(plot-its, plot-ivg, plot-transconductance) in process_and_analyze.py.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import hashlib
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Global console (use get_verbose_console() for config-aware console)
console = Console()


def get_verbose_console() -> Console:
    """
    Get console with verbosity settings from config.

    Returns
    -------
    Console
        Rich Console instance with quiet mode based on config.verbose

    Examples
    --------
    >>> console = get_verbose_console()
    >>> console.print("This respects config.verbose setting")
    """
    from src.cli.main import get_config

    config = get_config()
    return Console(quiet=not config.verbose)


def parse_seq_list(seq_str: str) -> list[int]:
    """
    Parse comma-separated seq numbers from string, with range support.

    Parameters
    ----------
    seq_str : str
        Comma-separated seq numbers (e.g., "52,57,58" or "52, 57, 58")
        Supports ranges with dash notation (e.g., "89-117" or "89-92,95,100-105")

    Returns
    -------
    list[int]
        List of seq numbers (sorted, duplicates removed)

    Raises
    ------
    ValueError
        If any seq number is not a valid integer or range format is invalid

    Examples
    --------
    >>> parse_seq_list("52,57,58")
    [52, 57, 58]
    >>> parse_seq_list("1, 2, 3")
    [1, 2, 3]
    >>> parse_seq_list("89-92")
    [89, 90, 91, 92]
    >>> parse_seq_list("89-92,95,100-102")
    [89, 90, 91, 92, 95, 100, 101, 102]
    """
    result = []
    try:
        # Split by comma first
        parts = seq_str.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                # Range notation (e.g., "89-117")
                range_parts = part.split("-")
                if len(range_parts) != 2:
                    raise ValueError(f"Invalid range format: '{part}'. Expected format: 'start-end'")
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
                if start > end:
                    raise ValueError(f"Invalid range: {start}-{end}. Start must be <= end.")
                result.extend(range(start, end + 1))
            else:
                # Single number
                result.append(int(part))

        # Remove duplicates and sort
        return sorted(set(result))

    except ValueError as e:
        if "Invalid range" in str(e) or "Invalid range format" in str(e):
            raise  # Re-raise our custom error messages
        raise ValueError(
            f"Invalid seq number format: '{seq_str}'. "
            f"Expected comma-separated integers or ranges (e.g., '52,57,58' or '89-117' or '89-92,95,100-105')."
        ) from e


def generate_timestamp_tag() -> str:
    """
    Generate timestamp tag for default output filenames.

    Returns
    -------
    str
        Timestamp string in format YYYYMMDD_HHMMSS

    Examples
    --------
    >>> generate_timestamp_tag()
    '20251020_143052'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_plot_tag(seq_numbers: list[int], custom_tag: str | None = None) -> str:
    """
    Generate unique plot tag based on seq numbers to prevent overwrites.

    Creates a short hash of the seq numbers so that plots with different
    experiments never overwrite each other, but the same experiments
    (in any order) will produce the same filename.

    Parameters
    ----------
    seq_numbers : list[int]
        List of seq numbers being plotted
    custom_tag : str, optional
        Custom tag to append to the hash

    Returns
    -------
    str
        Unique tag string (e.g., "seq_52_57_58" or "seq_52_57_58_custom")

    Examples
    --------
    >>> generate_plot_tag([52, 57, 58])
    'seq_52_57_58'
    >>> generate_plot_tag([52, 57, 58], "test")
    'seq_52_57_58_test'
    >>> generate_plot_tag([58, 52, 57])  # Same hash as [52, 57, 58]
    'seq_52_57_58'
    """
    # Sort seq numbers to ensure consistent ordering
    sorted_seqs = sorted(seq_numbers)

    # Create readable seq string for short lists
    if len(sorted_seqs) <= 5:
        seq_str = "_".join(str(s) for s in sorted_seqs)
        tag = f"seq_{seq_str}"
    else:
        # For long lists, use first 3 + count + hash
        first_three = "_".join(str(s) for s in sorted_seqs[:3])
        # Create short hash of all seq numbers
        seq_hash = hashlib.md5("_".join(str(s) for s in sorted_seqs).encode()).hexdigest()[:6]
        tag = f"seq_{first_three}_plus{len(sorted_seqs)-3}_{seq_hash}"

    if custom_tag:
        tag = f"{tag}_{custom_tag}"

    return tag


def setup_output_dir(chip: int, chip_group: str, output_dir: Path | None = None) -> Path:
    """
    Create and return chip-specific output directory using config.

    Creates: output_dir/{chip_group}{chip}/

    Parameters
    ----------
    chip : int
        Chip number
    chip_group : str
        Chip group name (e.g., "Alisson")
    output_dir : Path, optional
        Override output directory. If None, uses config.output_dir

    Returns
    -------
    Path
        Chip-specific output directory path (created if needed)

    Examples
    --------
    >>> setup_output_dir(67, "Alisson")
    PosixPath('figs/Alisson67')
    >>> setup_output_dir(67, "Alisson", Path("/tmp/plots"))
    PosixPath('/tmp/plots/Alisson67')
    """
    from src.cli.main import get_config

    if output_dir is None:
        config = get_config()
        output_dir = config.output_dir

        if config.verbose:
            console.print(f"[dim]Using output directory: {output_dir}[/dim]")

    chip_subdir = output_dir / f"{chip_group}{chip}"
    chip_subdir.mkdir(parents=True, exist_ok=True)
    return chip_subdir


def auto_select_experiments(
    chip: int,
    proc: str,
    chip_group: str,
    history_dir: Path | None = None,
    filters: dict | None = None
) -> list[int]:
    """
    Auto-select experiments based on procedure type and optional filters using config.

    Parameters
    ----------
    chip : int
        Chip number
    proc : str
        Procedure type to filter ("ITS", "IVg", "IV")
    chip_group : str
        Chip group name prefix
    history_dir : Path, optional
        Directory containing chip history files. If None, uses config.history_dir
    filters : dict, optional
        Additional filters:
        - "vg": Filter by gate voltage (float)
        - "vds": Filter by VDS voltage (float)
        - "wavelength": Filter by wavelength (float)
        - "date": Filter by date (str, "YYYY-MM-DD")

    Returns
    -------
    list[int]
        List of seq numbers matching criteria

    Raises
    ------
    FileNotFoundError
        If chip history file doesn't exist
    ValueError
        If no experiments match the filters

    Examples
    --------
    >>> auto_select_experiments(67, "ITS", "Alisson")
    [52, 57, 58, 61, 63]

    >>> auto_select_experiments(67, "ITS", "Alisson", filters={"vg": -0.4})
    [52, 57, 58]
    """
    from src.cli.main import get_config

    if history_dir is None:
        config = get_config()
        history_dir = config.history_dir

        if config.verbose:
            console.print(f"[dim]Using history directory: {history_dir}[/dim]")

    chip_name = f"{chip_group}{chip}"
    history_file = history_dir / f"{chip_name}_history.parquet"

    if not history_file.exists():
        raise FileNotFoundError(
            f"Chip history file not found: {history_file}\n"
            f"Run 'build-all-histories' command first to generate history files."
        )

    # Load history
    history = pl.read_parquet(history_file)

    # Filter by procedure
    filtered = history.filter(pl.col("proc") == proc)

    if filtered.height == 0:
        raise ValueError(
            f"No {proc} experiments found for {chip_name}.\n"
            f"Available procedures: {history['proc'].unique().to_list()}"
        )

    # Apply additional filters if provided
    if filters:
        if "vg" in filters and "VG" in filtered.columns:
            vg_target = float(filters["vg"])
            filtered = filtered.filter((pl.col("VG") - vg_target).abs() <= 1e-6)

        if "vds" in filters and "VDS" in filtered.columns:
            vds_target = float(filters["vds"])
            filtered = filtered.filter((pl.col("VDS") - vds_target).abs() <= 1e-6)

        if "wavelength" in filters and "Laser wavelength" in filtered.columns:
            wl_target = float(filters["wavelength"])
            filtered = filtered.filter((pl.col("Laser wavelength") - wl_target).abs() <= 1e-6)

        if "date" in filters and "date" in filtered.columns:
            date_str = filters["date"]
            filtered = filtered.filter(pl.col("date") == date_str)

    if filtered.height == 0:
        filter_desc = ", ".join(f"{k}={v}" for k, v in (filters or {}).items())
        raise ValueError(
            f"No {proc} experiments found for {chip_name} with filters: {filter_desc}"
        )

    # Extract seq numbers
    seq_numbers = filtered["seq"].to_list()
    return seq_numbers


def validate_experiments_exist(
    seq_numbers: list[int],
    chip: int,
    chip_group: str,
    history_dir: Path | None = None
) -> tuple[bool, list[str]]:
    """
    Check that seq numbers exist in chip history and return details using config.

    Parameters
    ----------
    seq_numbers : list[int]
        Seq numbers to validate
    chip : int
        Chip number
    chip_group : str
        Chip group name prefix
    history_dir : Path, optional
        Directory containing chip history files. If None, uses config.history_dir

    Returns
    -------
    tuple[bool, list[str]]
        (all_exist, error_messages)
        - all_exist: True if all seq numbers are valid
        - error_messages: List of error messages for invalid seq numbers

    Examples
    --------
    >>> validate_experiments_exist([52, 57, 999], 67, "Alisson")
    (False, ['Seq number 999 not found in Alisson67 history'])
    """
    from src.cli.main import get_config

    if history_dir is None:
        config = get_config()
        history_dir = config.history_dir

    chip_name = f"{chip_group}{chip}"
    history_file = history_dir / f"{chip_name}_history.parquet"

    if not history_file.exists():
        return False, [f"Chip history file not found: {history_file}"]

    # Load history
    history = pl.read_parquet(history_file)
    valid_seqs = set(history["seq"].to_list())

    # Check each seq number
    errors = []
    for seq in seq_numbers:
        if seq not in valid_seqs:
            errors.append(f"Seq number {seq} not found in {chip_name} history")

    return len(errors) == 0, errors


def load_history_for_plotting(
    seq_numbers: list[int],
    chip: int,
    chip_group: str,
    history_dir: Path | None = None
) -> pl.DataFrame:
    """
    Load chip history data for plotting, filtered by seq numbers using config.

    This function loads the chip history Parquet file and returns rows
    for the specified seq numbers. The returned DataFrame includes the
    `parquet_path` column pointing to staged measurement data.

    Parameters
    ----------
    seq_numbers : list[int]
        Seq numbers to load
    chip : int
        Chip number
    chip_group : str
        Chip group name prefix
    history_dir : Path, optional
        Directory containing chip history Parquet files. If None, uses config.history_dir

    Returns
    -------
    pl.DataFrame
        History data for specified experiments with columns:
        seq, date, time_hms, proc, summary, has_light, parquet_path, ...

    Raises
    ------
    FileNotFoundError
        If chip history file doesn't exist
    ValueError
        If seq numbers not found in history

    Examples
    --------
    >>> history = load_history_for_plotting([52, 57, 58], 67, "Alisson")
    >>> print(history.columns)
    ['seq', 'date', 'proc', 'parquet_path', ...]
    """
    from src.cli.main import get_config

    if history_dir is None:
        config = get_config()
        history_dir = config.history_dir

        if config.verbose:
            console.print(f"[dim]Loading history from: {history_dir}[/dim]")

    chip_name = f"{chip_group}{chip}"
    history_file = history_dir / f"{chip_name}_history.parquet"

    if not history_file.exists():
        raise FileNotFoundError(
            f"Chip history file not found: {history_file}\n"
            f"Run 'build-all-histories' command first to generate history files."
        )

    # Load full history
    history = pl.read_parquet(history_file)

    # Filter by seq numbers
    filtered = history.filter(pl.col("seq").is_in(seq_numbers))

    if filtered.height == 0:
        raise ValueError(
            f"No experiments found for seq numbers: {seq_numbers}\n"
            f"Available seq range: {history['seq'].min()} to {history['seq'].max()}"
        )

    # Verify all seq numbers were found
    found_seqs = set(filtered["seq"].to_list())
    missing_seqs = set(seq_numbers) - found_seqs
    if missing_seqs:
        raise ValueError(
            f"Seq numbers not found in {chip_name} history: {sorted(missing_seqs)}"
        )

    # Sort by seq to maintain order
    filtered = filtered.sort("seq")

    return filtered


def apply_metadata_filters(
    meta: pl.DataFrame,
    vg: float | None = None,
    vds: float | None = None,
    wavelength: float | None = None,
    date: str | None = None
) -> pl.DataFrame:
    """
    Apply user-specified filters to metadata DataFrame.

    Parameters
    ----------
    meta : pl.DataFrame
        Metadata DataFrame to filter
    vg : float, optional
        Gate voltage filter
    vds : float, optional
        VDS voltage filter
    wavelength : float, optional
        Wavelength filter (nm)
    date : str, optional
        Date filter (YYYY-MM-DD)

    Returns
    -------
    pl.DataFrame
        Filtered metadata

    Examples
    --------
    >>> filtered = apply_metadata_filters(meta, vg=-0.4, wavelength=365.0)
    """
    filtered = meta

    if vg is not None and "VG_meta" in filtered.columns:
        filtered = filtered.filter((pl.col("VG_meta") - vg).abs() <= 1e-6)

    if vds is not None and "VDS" in filtered.columns:
        filtered = filtered.filter((pl.col("VDS") - vds).abs() <= 1e-6)

    if wavelength is not None and "Laser wavelength" in filtered.columns:
        filtered = filtered.filter((pl.col("Laser wavelength") - wavelength).abs() <= 1e-6)

    if date is not None and "date" in filtered.columns:
        filtered = filtered.filter(pl.col("date") == date)

    return filtered


def display_experiment_list(experiments: pl.DataFrame, title: str = "Selected Experiments"):
    """
    Pretty-print selected experiments using Rich.

    Parameters
    ----------
    experiments : pl.DataFrame
        Metadata DataFrame to display
    title : str
        Title for the display panel

    Examples
    --------
    >>> display_experiment_list(meta, "ITS Experiments for Plotting")
    """
    if experiments.height == 0:
        console.print("[yellow]No experiments to display[/yellow]")
        return

    # Create table
    table = Table(title=title, show_header=True)

    # Add columns based on what's available
    if "file_idx" in experiments.columns:
        table.add_column("File #", style="cyan")
    if "proc" in experiments.columns:
        table.add_column("Proc", style="green")
    if "VG_meta" in experiments.columns:
        table.add_column("VG (V)", style="magenta")
    if "Laser wavelength" in experiments.columns:
        table.add_column("λ (nm)", style="yellow")
    if "Laser voltage" in experiments.columns:
        table.add_column("LED V", style="blue")
    if "source_file" in experiments.columns:
        table.add_column("File", style="dim")

    # Add rows
    for row in experiments.iter_rows(named=True):
        row_data = []
        if "file_idx" in row:
            row_data.append(str(row["file_idx"]))
        if "proc" in row:
            row_data.append(str(row["proc"]))
        if "VG_meta" in row:
            row_data.append(f"{row['VG_meta']:.2f}" if row['VG_meta'] is not None else "N/A")
        if "Laser wavelength" in row:
            wl = row["Laser wavelength"]
            row_data.append(f"{wl:.0f}" if wl is not None else "N/A")
        if "Laser voltage" in row:
            lv = row["Laser voltage"]
            row_data.append(f"{lv:.2f}" if lv is not None else "N/A")
        if "source_file" in row:
            # Shorten path for display
            path_str = str(row["source_file"])
            row_data.append(path_str[-40:] if len(path_str) > 40 else path_str)

        table.add_row(*row_data)

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {experiments.height} experiment(s)")


def display_plot_settings(settings: dict):
    """
    Pretty-print plot settings using Rich.

    Parameters
    ----------
    settings : dict
        Dictionary of setting names and values

    Examples
    --------
    >>> display_plot_settings({
    ...     "Legend by": "led_voltage",
    ...     "Padding": "0.05",
    ...     "Output dir": "figs/"
    ... })
    """
    lines = []
    for key, value in settings.items():
        lines.append(f"[cyan]{key}:[/cyan] {value}")

    panel = Panel(
        "\n".join(lines),
        title="[bold]Plot Settings[/bold]",
        border_style="blue"
    )
    console.print(panel)


def display_plot_success(output_file: Path):
    """
    Show success message with output file path.

    Parameters
    ----------
    output_file : Path
        Path to the generated plot file

    Examples
    --------
    >>> display_plot_success(Path("figs/Alisson67_its_overlay_cli_20251020.png"))
    """
    console.print(f"\n[bold green]✓ Plot generated successfully![/bold green]")
    console.print(f"[cyan]Output:[/cyan] {output_file}")
