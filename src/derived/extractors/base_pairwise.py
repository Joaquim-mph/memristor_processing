"""Base class for pairwise metric extractors.

Pairwise extractors work on consecutive pairs of measurements to compute
differences, correlations, or other comparative metrics.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
import polars as pl

from src.models.derived_metrics import DerivedMetric


class PairwiseMetricExtractor(ABC):
    """
    Base class for extractors that analyze pairs of consecutive measurements.

    Pairwise extractors enable comparative analysis between consecutive experiments,
    such as tracking device evolution, computing differences, or identifying trends.

    Example Use Cases
    -----------------
    - Consecutive IVg sweep differences (track CNP evolution)
    - Photoresponse decay (compare It measurements before/after treatment)
    - Hysteresis analysis (compare forward/backward sweeps)

    Subclass Requirements
    ---------------------
    Subclasses must implement:
    - applicable_procedures: List of procedure types to process
    - metric_name: Unique identifier for the pairwise metric
    - metric_category: Category classification
    - extract_pairwise: Core extraction logic for measurement pairs
    - validate: Quality validation for extracted metrics
    """

    @property
    @abstractmethod
    def applicable_procedures(self) -> List[str]:
        """
        List of procedure types this extractor handles.

        Returns
        -------
        List[str]
            Procedure names (e.g., ['IVg', 'VVg'])

        Notes
        -----
        Only measurements with matching procedure types will be paired.
        To pair IVg with IVg separately from VVg with VVg, list both:
        ['IVg', 'VVg']
        """
        pass

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """
        Unique identifier for this pairwise metric.

        Returns
        -------
        str
            Metric name (e.g., 'consecutive_sweep_difference')
        """
        pass

    @property
    @abstractmethod
    def metric_category(self) -> str:
        """
        Category of the metric.

        Returns
        -------
        str
            One of: 'electrical', 'photoresponse', 'thermal', 'optical', 'structural'
        """
        pass

    @property
    def pairing_strategy(self) -> str:
        """
        Strategy for pairing measurements.

        Returns
        -------
        str
            Pairing strategy identifier. Default: 'consecutive_same_proc'

        Strategies
        ----------
        - 'consecutive_same_proc': Pair consecutive seq_num with same procedure
        - 'consecutive_any': Pair any consecutive measurements (not recommended)
        - 'time_window': Pair measurements within time window (not yet implemented)
        """
        return "consecutive_same_proc"

    @abstractmethod
    def extract_pairwise(
        self,
        measurement_1: pl.DataFrame,
        metadata_1: Dict[str, Any],
        measurement_2: pl.DataFrame,
        metadata_2: Dict[str, Any]
    ) -> Optional[List[DerivedMetric]]:
        """
        Extract metrics from a pair of consecutive measurements.

        Parameters
        ----------
        measurement_1 : pl.DataFrame
            Earlier measurement data (loaded from Parquet)
            Columns depend on procedure (e.g., 'Vg (V)', 'I (A)' for IVg)
        metadata_1 : Dict[str, Any]
            Earlier measurement metadata from manifest.parquet
            Includes: run_id, chip_number, chip_group, proc, seq_num, etc.
        measurement_2 : pl.DataFrame
            Later measurement data (loaded from Parquet)
            Must be same procedure as measurement_1
        metadata_2 : Dict[str, Any]
            Later measurement metadata from manifest.parquet

        Returns
        -------
        Optional[List[DerivedMetric]]
            List of extracted metrics (can return multiple per pair)
            Returns None if extraction fails or pair is invalid

        Notes
        -----
        - Metrics should be linked to the SECOND (later) measurement via run_id
        - Store reference to first measurement in value_json
        - Use metadata_2['run_id'] for the metric's run_id field
        - Include metadata_1['run_id'] in the JSON payload
        """
        pass

    @abstractmethod
    def validate(self, result: DerivedMetric) -> bool:
        """
        Validate extracted metric quality.

        Parameters
        ----------
        result : DerivedMetric
            Extracted metric to validate

        Returns
        -------
        bool
            True if metric passes validation, False otherwise
        """
        pass

    def should_pair(
        self,
        metadata_1: Dict[str, Any],
        metadata_2: Dict[str, Any]
    ) -> bool:
        """
        Determine if two measurements should be paired.

        Default implementation enforces:
        1. Same chip_number
        2. Same procedure type
        3. Consecutive seq_num (seq_2 = seq_1 + 1)

        Parameters
        ----------
        metadata_1 : Dict[str, Any]
            Earlier measurement metadata
        metadata_2 : Dict[str, Any]
            Later measurement metadata

        Returns
        -------
        bool
            True if measurements should be paired, False otherwise

        Notes
        -----
        Override this method to implement custom pairing logic
        (e.g., time-window based pairing, gap tolerance, etc.)
        """
        # Check same chip
        if metadata_1.get("chip_number") != metadata_2.get("chip_number"):
            return False

        # Check same procedure
        if metadata_1.get("proc") != metadata_2.get("proc"):
            return False

        # Check procedure is applicable
        if metadata_1.get("proc") not in self.applicable_procedures:
            return False

        # Check consecutive seq_num
        seq_1 = metadata_1.get("seq_num")
        seq_2 = metadata_2.get("seq_num")

        if seq_1 is None or seq_2 is None:
            return False

        # Must be consecutive (no gaps)
        return seq_2 == seq_1 + 1

    def __repr__(self) -> str:
        """String representation of extractor."""
        return (
            f"{self.__class__.__name__}("
            f"metric_name='{self.metric_name}', "
            f"procedures={self.applicable_procedures}, "
            f"strategy='{self.pairing_strategy}')"
        )
