"""
Output formatters for CLI commands.

This module provides decoupled output formatting for data commands, enabling
multiple output formats (Rich tables, JSON, CSV) for better scripting integration
and automation.

Usage:
    >>> from src.cli.formatters import get_formatter
    >>> formatter = get_formatter("json")
    >>> output = formatter.format_dataframe(df, title="My Data")
    >>> print(output)

Available Formats:
    - table: Rich terminal tables (default)
    - json: Machine-readable JSON
    - csv: Spreadsheet-compatible CSV
"""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from rich import box


# ============================================================================
# Abstract Base Class
# ============================================================================

class OutputFormatter(ABC):
    """
    Abstract base class for output formatters.

    Output formatters decouple data presentation from command logic,
    enabling the same data to be rendered in multiple formats (table, JSON, CSV).
    """

    @abstractmethod
    def format_dataframe(
        self,
        df: pl.DataFrame,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format a Polars DataFrame for output.

        Parameters
        ----------
        df : pl.DataFrame
            Data to format
        title : str, optional
            Title or header for the output
        metadata : dict, optional
            Additional context (chip name, filters applied, etc.)

        Returns
        -------
        str
            Formatted string ready for output to stdout
        """
        pass

    @abstractmethod
    def format_summary(self, data: Dict[str, Any]) -> str:
        """
        Format a summary or statistics dictionary.

        Parameters
        ----------
        data : dict
            Summary data (counts, statistics, etc.)

        Returns
        -------
        str
            Formatted summary string
        """
        pass


# ============================================================================
# Rich Table Formatter (Default - Current Behavior)
# ============================================================================

class RichTableFormatter(OutputFormatter):
    """
    Rich table formatter for beautiful terminal output.

    This is the default formatter and preserves the current behavior
    of displaying data as styled Rich tables in the terminal.

    Features:
        - Color-coded columns
        - Box borders and styling
        - Pagination support
        - Light status emojis (💡 🌙 ❗)
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize Rich table formatter.

        Parameters
        ----------
        console : Console, optional
            Rich console instance. If None, creates a new one.
        """
        self.console = console or Console()

    def format_dataframe(
        self,
        df: pl.DataFrame,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format DataFrame as Rich table.

        Parameters
        ----------
        df : pl.DataFrame
            Data to display
        title : str, optional
            Table title
        metadata : dict, optional
            Additional context for display

        Returns
        -------
        str
            Formatted Rich table (captured from console)
        """
        # Create table with title
        table = Table(
            title=title or None,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        # Add columns
        for col in df.columns:
            # Style based on column name
            if col in ["seq", "file_idx"]:
                table.add_column(col, style="bold blue", justify="right")
            elif col in ["procedure", "proc"]:
                table.add_column(col, style="magenta")
            elif "date" in col.lower() or "time" in col.lower():
                table.add_column(col, style="cyan")
            elif "light" in col.lower():
                table.add_column(col, style="yellow")
            elif col in ["cnp_voltage", "delta_current", "delta_voltage"]:
                table.add_column(col, style="green", justify="right")
            else:
                table.add_column(col, justify="left")

        # Add rows
        for row in df.iter_rows(named=True):
            row_values = []
            for col in df.columns:
                value = row[col]

                # Format value based on type
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    formatted = "[dim]—[/dim]"
                elif isinstance(value, float):
                    # Format floats with reasonable precision
                    formatted = f"{value:.4g}"
                elif isinstance(value, bool):
                    formatted = "✓" if value else "✗"
                elif col == "light_status":
                    # Add emoji for light status
                    if value == "light":
                        formatted = "💡 Light"
                    elif value == "dark":
                        formatted = "🌙 Dark"
                    elif value == "unknown":
                        formatted = "❗ Unknown"
                    else:
                        formatted = str(value)
                else:
                    formatted = str(value)

                row_values.append(formatted)

            table.add_row(*row_values)

        # Capture table output to string
        with self.console.capture() as capture:
            self.console.print(table)

        return capture.get()

    def format_summary(self, data: Dict[str, Any]) -> str:
        """
        Format summary as Rich text.

        Parameters
        ----------
        data : dict
            Summary statistics

        Returns
        -------
        str
            Formatted summary
        """
        lines = []
        for key, value in data.items():
            lines.append(f"[cyan]{key}:[/cyan] {value}")

        with self.console.capture() as capture:
            for line in lines:
                self.console.print(line)

        return capture.get()


# ============================================================================
# JSON Formatter (Machine-Readable)
# ============================================================================

class JSONFormatter(OutputFormatter):
    """
    JSON formatter for machine-readable output.

    Converts DataFrames to JSON with proper type handling for:
    - Datetime → ISO 8601 strings
    - NaN/Inf → null
    - Float precision control
    - UTF-8 encoding

    Output Structure:
        {
            "metadata": {...},
            "data": [{...}, {...}, ...]
        }
    """

    def __init__(self, indent: int = 2, ensure_ascii: bool = False):
        """
        Initialize JSON formatter.

        Parameters
        ----------
        indent : int, optional
            Indentation spaces (default: 2)
        ensure_ascii : bool, optional
            If True, escape non-ASCII characters (default: False)
        """
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def format_dataframe(
        self,
        df: pl.DataFrame,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format DataFrame as JSON.

        Parameters
        ----------
        df : pl.DataFrame
            Data to serialize
        title : str, optional
            Title (added to metadata)
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        str
            JSON string
        """
        # Convert to list of dicts
        data_dicts = df.to_dicts()

        # Post-process for JSON compatibility
        for row in data_dicts:
            for key, value in list(row.items()):
                row[key] = self._serialize_value(value)

        # Build output structure
        output = {
            "metadata": metadata or {},
            "data": data_dicts,
        }

        # Add title to metadata if provided
        if title:
            output["metadata"]["title"] = title

        # Add row count
        output["metadata"]["row_count"] = len(data_dicts)

        return json.dumps(output, indent=self.indent, ensure_ascii=self.ensure_ascii)

    def format_summary(self, data: Dict[str, Any]) -> str:
        """
        Format summary as JSON.

        Parameters
        ----------
        data : dict
            Summary data

        Returns
        -------
        str
            JSON string
        """
        # Serialize all values
        serialized = {k: self._serialize_value(v) for k, v in data.items()}
        return json.dumps(serialized, indent=self.indent, ensure_ascii=self.ensure_ascii)

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize a value for JSON compatibility.

        Handles:
        - NaN/Inf → null
        - datetime → ISO 8601 string
        - Float precision
        - numpy types → Python types

        Parameters
        ----------
        value : Any
            Value to serialize

        Returns
        -------
        Any
            JSON-compatible value
        """
        # Handle None
        if value is None:
            return None

        # Handle datetime
        if isinstance(value, datetime):
            return value.isoformat()

        # Handle numpy/Polars datetime
        if hasattr(value, 'isoformat'):
            return value.isoformat()

        # Handle float (including numpy floats)
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return None  # NaN/Inf → null
            # Round to avoid precision artifacts
            return round(float(value), 10)

        # Handle numpy integers
        if isinstance(value, (np.integer,)):
            return int(value)

        # Handle numpy bool
        if isinstance(value, (np.bool_,)):
            return bool(value)

        # Handle lists/arrays
        if isinstance(value, (list, np.ndarray)):
            return [self._serialize_value(v) for v in value]

        # Handle dicts
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        # Default: return as-is
        return value


# ============================================================================
# CSV Formatter (Spreadsheet Export)
# ============================================================================

class CSVFormatter(OutputFormatter):
    """
    CSV formatter for spreadsheet export.

    Generates standard CSV files compatible with Excel, Google Sheets, etc.

    Features:
        - UTF-8 encoding
        - Proper escaping of commas/quotes
        - Null handling (empty strings)
        - Header row included
    """

    def __init__(self, null_value: str = ""):
        """
        Initialize CSV formatter.

        Parameters
        ----------
        null_value : str, optional
            String to use for null values (default: "" empty string)
        """
        self.null_value = null_value

    def format_dataframe(
        self,
        df: pl.DataFrame,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format DataFrame as CSV.

        Parameters
        ----------
        df : pl.DataFrame
            Data to export
        title : str, optional
            Title (ignored for CSV - would break parsing)
        metadata : dict, optional
            Metadata (ignored for CSV)

        Returns
        -------
        str
            CSV string with header row
        """
        # Handle nested columns (List, Struct) by converting to strings
        # CSV cannot represent nested data, so we stringify it
        df_flat = df.clone()
        for col in df.columns:
            dtype = df[col].dtype
            # Check if column is nested (List or Struct)
            if dtype in [pl.List, pl.Struct] or str(dtype).startswith("List") or str(dtype).startswith("Struct"):
                # Convert nested column to string representation
                df_flat = df_flat.with_columns(
                    pl.col(col).cast(pl.String).alias(col)
                )

        # Use Polars built-in CSV writer
        buffer = io.StringIO()
        df_flat.write_csv(buffer, null_value=self.null_value)
        return buffer.getvalue()

    def format_summary(self, data: Dict[str, Any]) -> str:
        """
        Format summary as CSV.

        Parameters
        ----------
        data : dict
            Summary data

        Returns
        -------
        str
            CSV with "key,value" format
        """
        # Convert dict to DataFrame
        df = pl.DataFrame({
            "key": list(data.keys()),
            "value": [str(v) for v in data.values()],
        })

        buffer = io.StringIO()
        df.write_csv(buffer, null_value=self.null_value)
        return buffer.getvalue()


# ============================================================================
# Formatter Registry and Factory
# ============================================================================

# Registry of available formatters
FORMATTERS: Dict[str, Type[OutputFormatter]] = {
    "table": RichTableFormatter,
    "json": JSONFormatter,
    "csv": CSVFormatter,
}

# Aliases for convenience
FORMATTER_ALIASES: Dict[str, str] = {
    "rich": "table",
    "terminal": "table",
    "text": "table",
    "jsonl": "json",  # For now, same as JSON (could add streaming later)
    "tsv": "csv",  # Could customize CSVFormatter with tab separator
}


def get_formatter(format_name: str) -> OutputFormatter:
    """
    Get formatter instance by name.

    Parameters
    ----------
    format_name : str
        Format name: "table", "json", "csv", or alias

    Returns
    -------
    OutputFormatter
        Formatter instance

    Raises
    ------
    ValueError
        If format name is unknown

    Examples
    --------
    >>> formatter = get_formatter("json")
    >>> output = formatter.format_dataframe(df)

    >>> formatter = get_formatter("table")
    >>> output = formatter.format_dataframe(df, title="My Data")
    """
    # Normalize format name
    format_name = format_name.strip().lower()

    # Check aliases first
    if format_name in FORMATTER_ALIASES:
        format_name = FORMATTER_ALIASES[format_name]

    # Get formatter class
    if format_name not in FORMATTERS:
        valid_formats = list(FORMATTERS.keys()) + list(FORMATTER_ALIASES.keys())
        raise ValueError(
            f"Unknown format: '{format_name}'. "
            f"Valid formats: {', '.join(sorted(set(valid_formats)))}"
        )

    formatter_class = FORMATTERS[format_name]
    return formatter_class()


def list_formatters() -> List[str]:
    """
    List all available formatter names.

    Returns
    -------
    list of str
        Available format names (canonical names only)

    Examples
    --------
    >>> list_formatters()
    ['csv', 'json', 'table']
    """
    return sorted(FORMATTERS.keys())


def register_formatter(name: str, formatter_class: Type[OutputFormatter]) -> None:
    """
    Register a custom formatter.

    Parameters
    ----------
    name : str
        Format name (used with --format flag)
    formatter_class : Type[OutputFormatter]
        Formatter class (must inherit from OutputFormatter)

    Raises
    ------
    TypeError
        If formatter_class doesn't inherit from OutputFormatter
    ValueError
        If name is already registered

    Examples
    --------
    >>> class MyFormatter(OutputFormatter):
    ...     def format_dataframe(self, df, title="", metadata=None):
    ...         return "custom output"
    ...     def format_summary(self, data):
    ...         return "custom summary"
    >>> register_formatter("custom", MyFormatter)
    >>> formatter = get_formatter("custom")
    """
    if not issubclass(formatter_class, OutputFormatter):
        raise TypeError(
            f"Formatter class must inherit from OutputFormatter, "
            f"got {formatter_class.__name__}"
        )

    name = name.strip().lower()
    if name in FORMATTERS:
        raise ValueError(f"Formatter '{name}' is already registered")

    FORMATTERS[name] = formatter_class
