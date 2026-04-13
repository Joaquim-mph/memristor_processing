"""
Base class for metric extractors.

All metric extractors must inherit from MetricExtractor and implement
the abstract methods to define which procedures they apply to and how
to extract metrics from measurement data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import polars as pl

from src.models.derived_metrics import DerivedMetric


class MetricExtractor(ABC):
    """
    Abstract base class for all metric extractors.

    Extractors compute derived metrics from staged measurement data.
    Each extractor specifies which procedures it applies to and implements
    the extraction logic.

    Subclasses must implement:
    - applicable_procedures: List of procedure types this extractor handles
    - metric_name: Unique identifier for this metric
    - metric_category: Category (electrical, photoresponse, etc.)
    - extract: Core extraction logic
    - validate: Quality checks for extracted metric

    Example
    -------
    >>> class CNPExtractor(MetricExtractor):
    ...     @property
    ...     def applicable_procedures(self) -> List[str]:
    ...         return ["IVg", "VVg"]
    ...
    ...     @property
    ...     def metric_name(self) -> str:
    ...         return "cnp_voltage"
    ...
    ...     @property
    ...     def metric_category(self) -> str:
    ...         return "electrical"
    ...
    ...     def extract(self, measurement: pl.DataFrame, metadata: dict) -> Optional[DerivedMetric]:
    ...         # Extract charge neutrality point from IVg/VVg data
    ...         # ... implementation ...
    ...         return metric
    ...
    ...     def validate(self, result: DerivedMetric) -> bool:
    ...         # Check if CNP is in reasonable range
    ...         return -10.0 <= result.value_float <= 10.0
    """

    # ═══════════════════════════════════════════════════════════════════
    # Abstract Properties (Must be implemented by subclasses)
    # ═══════════════════════════════════════════════════════════════════

    @property
    @abstractmethod
    def applicable_procedures(self) -> List[str]:
        """
        List of procedure types this extractor applies to.

        Returns
        -------
        List[str]
            Procedure names (e.g., ["IVg", "VVg"] or ["It", "Vt"])

        Examples
        --------
        For CNP extraction:
            return ["IVg", "VVg"]

        For photoresponse:
            return ["It", "Vt"]
        """
        pass

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """
        Unique identifier for this metric.

        Should be lowercase with underscores (e.g., 'cnp_voltage', 'delta_ids').

        Returns
        -------
        str
            Metric identifier

        Examples
        --------
        >>> extractor.metric_name
        'cnp_voltage'
        """
        pass

    @property
    @abstractmethod
    def metric_category(self) -> str:
        """
        Metric category for organization.

        Returns
        -------
        str
            Category: 'electrical', 'photoresponse', 'thermal', 'optical', 'structural'

        Examples
        --------
        >>> extractor.metric_category
        'electrical'
        """
        pass

    # ═══════════════════════════════════════════════════════════════════
    # Abstract Methods (Must be implemented by subclasses)
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    def extract(
        self,
        measurement: pl.DataFrame,
        metadata: Dict[str, Any]
    ) -> Optional[DerivedMetric]:
        """
        Extract metric from a single measurement.

        This is the core extraction logic. Subclasses should:
        1. Validate that measurement has required columns
        2. Perform analytical computation
        3. Compute confidence score based on data quality
        4. Set flags for any warnings
        5. Return DerivedMetric or None if extraction failed

        Parameters
        ----------
        measurement : pl.DataFrame
            Measurement data from staged Parquet file
            Contains columns like 'Vg (V)', 'Ids (A)', 'time (s)', etc.

        metadata : Dict[str, Any]
            Metadata from manifest.parquet row, including:
            - run_id: str - Unique measurement identifier
            - chip_number: int - Chip ID
            - chip_group: str - Chip group name
            - procedure: str - Procedure type (IVg, It, etc.)
            - seq_num: Optional[int] - Sequence number from history
            - extraction_version: str - Code version
            - vds_v: Optional[float] - Fixed drain voltage (if applicable)
            - vg_fixed_v: Optional[float] - Fixed gate voltage (if applicable)
            - ... other procedure-specific parameters

        Returns
        -------
        Optional[DerivedMetric]
            Extracted metric, or None if extraction failed

        Examples
        --------
        >>> def extract(self, measurement, metadata):
        ...     # Get required data
        ...     vg = measurement["Vg (V)"].to_numpy()
        ...     ids = measurement["Ids (A)"].to_numpy()
        ...
        ...     # Compute metric
        ...     cnp_voltage = find_max_resistance_point(vg, ids)
        ...
        ...     # Return DerivedMetric
        ...     return DerivedMetric(
        ...         run_id=metadata["run_id"],
        ...         chip_number=metadata["chip_number"],
        ...         chip_group=metadata["chip_group"],
        ...         procedure=metadata["procedure"],
        ...         metric_name=self.metric_name,
        ...         metric_category=self.metric_category,
        ...         value_float=cnp_voltage,
        ...         unit="V",
        ...         extraction_method="max_resistance",
        ...         extraction_version=metadata["extraction_version"],
        ...         confidence=1.0
        ...     )
        """
        pass

    @abstractmethod
    def validate(self, result: DerivedMetric) -> bool:
        """
        Validate extracted metric passes quality checks.

        This is an optional QA step that checks if the metric value is
        physically reasonable. Failed validation doesn't prevent saving
        the metric, but can be used to flag suspicious results.

        Parameters
        ----------
        result : DerivedMetric
            Extracted metric to validate

        Returns
        -------
        bool
            True if metric passes quality checks, False otherwise

        Examples
        --------
        >>> def validate(self, result):
        ...     # Check CNP is in reasonable voltage range
        ...     if result.value_float is None:
        ...         return False
        ...     return -10.0 <= result.value_float <= 10.0
        """
        pass

    # ═══════════════════════════════════════════════════════════════════
    # Optional Helper Methods (Can be overridden by subclasses)
    # ═══════════════════════════════════════════════════════════════════

    def can_extract(self, procedure: str) -> bool:
        """
        Check if this extractor applies to a given procedure.

        Parameters
        ----------
        procedure : str
            Procedure type (e.g., 'IVg', 'It')

        Returns
        -------
        bool
            True if extractor applies to this procedure

        Examples
        --------
        >>> extractor.can_extract("IVg")
        True
        >>> extractor.can_extract("LaserCalibration")
        False
        """
        return procedure in self.applicable_procedures

    def __repr__(self) -> str:
        """String representation of extractor."""
        return (
            f"{self.__class__.__name__}("
            f"metric='{self.metric_name}', "
            f"category='{self.metric_category}', "
            f"procedures={self.applicable_procedures})"
        )


# ══════════════════════════════════════════════════════════════════════
# Helper Functions for Extractors
# ══════════════════════════════════════════════════════════════════════

def safe_get_column(
    df: pl.DataFrame,
    col_name: str,
    default: Optional[Any] = None
) -> Optional[Any]:
    """
    Safely get a column from DataFrame, returning default if not present.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to query
    col_name : str
        Column name
    default : Optional[Any]
        Default value if column not present

    Returns
    -------
    Optional[Any]
        Column data or default value

    Examples
    --------
    >>> vl = safe_get_column(measurement, "VL (V)", default=None)
    """
    if col_name in df.columns:
        return df[col_name]
    return default


def compute_confidence(
    checks: Dict[str, bool],
    penalties: Dict[str, float]
) -> float:
    """
    Compute confidence score from quality checks.

    Starts at 1.0 and multiplies by penalty for each failed check.

    Parameters
    ----------
    checks : Dict[str, bool]
        Dictionary of check name -> passed (True/False)
    penalties : Dict[str, float]
        Dictionary of check name -> penalty multiplier (e.g., 0.7 = 30% penalty)

    Returns
    -------
    float
        Confidence score (0.0-1.0)

    Examples
    --------
    >>> checks = {"at_edge": False, "noisy": True}
    >>> penalties = {"at_edge": 0.5, "noisy": 0.8}
    >>> compute_confidence(checks, penalties)
    0.5  # Failed at_edge check, passed noisy check
    """
    confidence = 1.0
    for check_name, passed in checks.items():
        if not passed and check_name in penalties:
            confidence *= penalties[check_name]
    return max(0.0, min(1.0, confidence))


def build_flags(checks: Dict[str, bool]) -> Optional[str]:
    """
    Build comma-separated flag string from failed checks.

    Parameters
    ----------
    checks : Dict[str, bool]
        Dictionary of check name -> passed (True/False)

    Returns
    -------
    Optional[str]
        Comma-separated flag names, or None if all checks passed

    Examples
    --------
    >>> checks = {"CNP_AT_EDGE": False, "NOISY_DATA": True, "GOOD_SNR": True}
    >>> build_flags(checks)
    'CNP_AT_EDGE'
    """
    failed = [name for name, passed in checks.items() if not passed]
    return ",".join(failed) if failed else None
