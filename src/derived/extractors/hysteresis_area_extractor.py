"""
Hysteresis loop area extractor.

Computes the integrated area between forward and backward IV branches,
a standard memristor figure of merit quantifying the memory window.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging

import polars as pl

from src.models.derived_metrics import DerivedMetric
from src.derived.extractors.base import MetricExtractor
from src.derived.algorithms.hysteresis import compute_hysteresis_area

logger = logging.getLogger(__name__)


class HysteresisAreaExtractor(MetricExtractor):
    """
    Extract hysteresis loop area from IV sweeps.

    Integrates |I_fwd - I_bwd| over voltage using the trapezoidal rule.

    Parameters
    ----------
    target_metric : str
        One of: "hysteresis_total_area", "hysteresis_positive_area",
        "hysteresis_negative_area"
    dv_threshold : float, optional
        Minimum voltage step to classify sweep direction (default 0.05 V).
    """

    VALID_METRICS = {
        "hysteresis_total_area",
        "hysteresis_positive_area",
        "hysteresis_negative_area",
    }
    _METRIC_TO_KEY = {
        "hysteresis_total_area": "total_area",
        "hysteresis_positive_area": "positive_area",
        "hysteresis_negative_area": "negative_area",
    }

    def __init__(
        self,
        target_metric: str,
        dv_threshold: float = 0.05,
    ):
        if target_metric not in self.VALID_METRICS:
            raise ValueError(
                f"target_metric must be one of {self.VALID_METRICS}, got '{target_metric}'"
            )
        self._target_metric = target_metric
        self.dv_threshold = dv_threshold
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
        return "hysteresis_area"

    @property
    def unit(self) -> str:
        return "V*A"

    def extract(
        self,
        measurement: pl.DataFrame,
        metadata: Dict[str, Any],
    ) -> Optional[DerivedMetric]:
        run_id = metadata.get("run_id")
        if run_id is None:
            logger.error("No run_id in metadata")
            return None

        if run_id not in self._cache:
            try:
                areas = compute_hysteresis_area(
                    measurement, dv_threshold=self.dv_threshold
                )
                self._cache[run_id] = areas
            except Exception as e:
                logger.error(f"Failed to compute hysteresis area for {run_id}: {e}", exc_info=True)
                return None

        areas = self._cache[run_id]
        key = self._METRIC_TO_KEY[self._target_metric]
        value = areas.get(key)

        if value is None or (isinstance(value, float) and value != value):
            logger.warning(f"Hysteresis area extraction returned NaN for {run_id}")
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
        if result.value_float is None:
            return False
        return result.value_float >= 0.0
