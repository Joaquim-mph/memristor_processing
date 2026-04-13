"""
Shared helpers for chip history inspection.

Provides reusable filtering and summarization logic for both the CLI
(`show-history`) and the TUI history browser to ensure consistent behaviour.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict

import polars as pl


class HistoryFilterError(Exception):
    """
    Exception raised when history filtering fails.

    Attributes
    ----------
    exit_code : int
        Suggested exit code for CLI commands (0 for graceful exits).
    """

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


_LIGHT_NORMALIZATION = {
    "light": "light",
    "l": "light",
    "💡": "light",
    "dark": "dark",
    "d": "dark",
    "🌙": "dark",
    "unknown": "unknown",
    "u": "unknown",
    "?": "unknown",
    "❗": "unknown",
}

_LIGHT_DESCRIPTIONS = {
    "light": "light experiments",
    "dark": "dark experiments",
    "unknown": "unknown light status",
}


def filter_history(
    df: pl.DataFrame,
    *,
    proc_filter: Optional[str] = None,
    light_filter: Optional[str] = None,
    limit: Optional[int] = None,
    strict: bool = True,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Apply procedure/light/limit filters to a history DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Source history dataframe.
    proc_filter : str, optional
        Procedure name to filter by (e.g., "IVg", "It").
    light_filter : str, optional
        Light filter string (accepts aliases: light/l/💡, dark/d/🌙, unknown/u/?/❗).
    limit : int, optional
        Keep only the last N experiments (tail).
    strict : bool
        If True, raise HistoryFilterError when filters yield no rows.
        Invalid filters always raise, regardless of strict.

    Returns
    -------
    Tuple[pl.DataFrame, List[str]]
        Filtered dataframe and a list of textual filter descriptors.
    """
    filtered = df
    applied_filters: List[str] = []

    if proc_filter:
        filtered = filtered.filter(pl.col("proc") == proc_filter)
        applied_filters.append(f"proc={proc_filter}")
        if filtered.height == 0 and strict:
            raise HistoryFilterError(
                f"No experiments found with procedure '{proc_filter}'",
                exit_code=0,
            )

    if light_filter:
        normalized = _normalize_light_filter(light_filter)
        applied_filters.append(f"light={normalized}")

        if normalized == "light":
            filtered = filtered.filter(pl.col("has_light") == True)  # noqa: E712
        elif normalized == "dark":
            filtered = filtered.filter(pl.col("has_light") == False)  # noqa: E712
        else:  # unknown
            filtered = filtered.filter(pl.col("has_light").is_null())

        if filtered.height == 0 and strict:
            raise HistoryFilterError(
                f"No {_LIGHT_DESCRIPTIONS[normalized]} found",
                exit_code=0,
            )

    if limit is not None:
        try:
            limit_int = int(limit)
        except (TypeError, ValueError):
            raise HistoryFilterError(
                f"Invalid limit value '{limit}'. Provide a positive integer.",
                exit_code=1,
            ) from None

        if limit_int < 0:
            raise HistoryFilterError(
                "Limit must be non-negative",
                exit_code=1,
            )

        if limit_int and filtered.height > limit_int:
            filtered = filtered.tail(limit_int)
        elif limit_int == 0:
            filtered = filtered.head(0)

        applied_filters.append(f"limit={limit_int}")

    return filtered, applied_filters


def summarize_history(df: pl.DataFrame) -> Dict[str, object]:
    """
    Produce summary statistics for a chip history.

    Returns a dictionary with keys:
    - total (int)
    - date_range (str)
    - num_days (int)
    - proc_counts (List[Tuple[str, int]])
    - light_counts (Dict[str, int] | None)
    """
    summary: Dict[str, object] = {
        "total": df.height,
        "date_range": "unknown",
        "num_days": 0,
        "proc_counts": [],
        "light_counts": None,
    }

    if "date" in df.columns:
        dates = [
            d for d in df["date"].drop_nulls().to_list()
            if isinstance(d, str) and d.lower() != "unknown"
        ]
        if dates:
            summary["date_range"] = f"{min(dates)} to {max(dates)}"
            summary["num_days"] = len(set(dates))

    if "proc" in df.columns:
        proc_counts = (
            df.group_by("proc")
              .agg(pl.len().alias("count"))
              .sort("proc")
        )
        summary["proc_counts"] = [
            (row["proc"], int(row["count"]))
            for row in proc_counts.iter_rows(named=True)
        ]

    if "has_light" in df.columns:
        summary["light_counts"] = {
            "light": df.filter(pl.col("has_light") == True).height,   # noqa: E712
            "dark": df.filter(pl.col("has_light") == False).height,   # noqa: E712
            "unknown": df.filter(pl.col("has_light").is_null()).height,
        }

    return summary


def _normalize_light_filter(value: str) -> str:
    """Normalize light filter aliases."""
    normalized = _LIGHT_NORMALIZATION.get(value.lower())
    if normalized is None:
        raise HistoryFilterError(
            f"Invalid light filter '{value}'. Use: light, dark, or unknown",
            exit_code=1,
        )
    return normalized
