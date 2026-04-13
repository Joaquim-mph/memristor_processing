"""
Polynomial-corrected hysteresis extractor.

Computes background-corrected hysteresis using polynomial fitting, then extracts
peak current and voltage for both negative and positive branches.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging

import polars as pl

from src.models.derived_metrics import DerivedMetric
from src.derived.extractors.base import MetricExtractor
from src.derived.algorithms.hysteresis import (
    compute_poly_corrected_hysteresis,
    extract_poly_corrected_maxima,
)

logger = logging.getLogger(__name__)


class PolyCorrectedHysteresisExtractor(MetricExtractor):
    """
    Extract peak corrected hysteresis metrics from IV sweeps.

    Fits a polynomial background to backward legs and subtracts it,
    then finds peak current and voltage for negative/positive branches.

    Parameters
    ----------
    target_metric : str
        One of: "max_neg_I", "max_neg_V", "max_pos_I", "max_pos_V"
    step_size : float, optional
        Voltage step size for snapping grid (default 0.1 V).
    poly_order : int, optional
        Polynomial order for background fitting (default 7).
    """

    VALID_METRICS = {
        "poly_corrected_max_neg_i",
        "poly_corrected_max_neg_v",
        "poly_corrected_max_pos_i",
        "poly_corrected_max_pos_v",
    }
    METRIC_UNITS = {
        "poly_corrected_max_neg_i": "A",
        "poly_corrected_max_neg_v": "V",
        "poly_corrected_max_pos_i": "A",
        "poly_corrected_max_pos_v": "V",
    }
    # Map from our metric names to the keys returned by extract_poly_corrected_maxima()
    _METRIC_TO_KEY = {
        "poly_corrected_max_neg_i": "max_neg_I",
        "poly_corrected_max_neg_v": "max_neg_V",
        "poly_corrected_max_pos_i": "max_pos_I",
        "poly_corrected_max_pos_v": "max_pos_V",
    }

    def __init__(
        self,
        target_metric: str,
        step_size: float = 0.1,
        poly_order: int = 7,
    ):
        if target_metric not in self.VALID_METRICS:
            raise ValueError(
                f"target_metric must be one of {self.VALID_METRICS}, got '{target_metric}'"
            )
        self._target_metric = target_metric
        self.step_size = step_size
        self.poly_order = poly_order

        # Cache maxima per run_id to avoid recomputation
        self._cache: Dict[str, dict] = {}

    @property
    def applicable_procedures(self) -> list[str]:
        return ["IV"]

    @property
    def metric_name(self) -> str:
        return self._target_metric

    @property
    def metric_category(self) -> str:
        return "electrical"

    @property
    def extraction_method(self) -> str:
        return "poly_corrected_hysteresis"

    @property
    def unit(self) -> str:
        return self.METRIC_UNITS[self._target_metric]

    def extract(
        self,
        measurement: pl.DataFrame,
        metadata: Dict[str, Any],
    ) -> Optional[DerivedMetric]:
        """
        Extract polynomial-corrected hysteresis peak from IV measurement.

        Parameters
        ----------
        measurement : pl.DataFrame
            Must contain "Vsd (V)" and "I (A)" columns.
        metadata : Dict[str, Any]
            Measurement metadata including run_id, chip_number, etc.

        Returns
        -------
        Optional[DerivedMetric]
            Extracted metric or None if extraction fails.
        """
        run_id = metadata.get("run_id")
        if run_id is None:
            logger.error("No run_id in metadata")
            return None

        # Check cache first
        if run_id not in self._cache:
            try:
                corrected_df = compute_poly_corrected_hysteresis(
                    measurement,
                    step_size=self.step_size,
                    poly_order=self.poly_order,
                )
                maxima = extract_poly_corrected_maxima(corrected_df)
                self._cache[run_id] = maxima
            except Exception as e:
                logger.error(
                    f"Failed to compute poly-corrected hysteresis for {run_id}: {e}",
                    exc_info=True,
                )
                return None

        maxima = self._cache[run_id]
        algo_key = self._METRIC_TO_KEY[self._target_metric]
        value = maxima.get(algo_key)

        if value is None or (isinstance(value, float) and value != value):  # NaN check
            logger.warning(f"Poly-corrected hysteresis extraction returned NaN for {run_id}")
            return None

        return DerivedMetric(
            run_id=run_id,
            chip_number=metadata.get("chip_number"),
            chip_group=metadata.get("chip_group"),
            procedure=metadata.get("proc", "IV"),
            seq_num=metadata.get("seq_num"),
            metric_name=self.metric_name,
            metric_category=self.metric_category,
            value_float=value,
            unit=self.unit,
            extraction_method=self.extraction_method,
            extraction_version=metadata.get("extraction_version", "unknown"),
            extraction_timestamp=datetime.now(timezone.utc),
            confidence=1.0,
            flags=None,
        )

    def validate(self, result: DerivedMetric) -> bool:
        """Validate that peak values are physically reasonable."""
        if result.value_float is None:
            return False

        # Current should be in reasonable range (nanoamps to milliamps typical)
        if self._target_metric in ("poly_corrected_max_neg_i", "poly_corrected_max_pos_i"):
            return 1e-12 < abs(result.value_float) < 1.0

        # Voltage should be in reasonable range (typically < 10V for these devices)
        if self._target_metric in ("poly_corrected_max_neg_v", "poly_corrected_max_pos_v"):
            return abs(result.value_float) < 20.0

        return True
