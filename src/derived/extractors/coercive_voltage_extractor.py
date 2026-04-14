"""
Characteristic voltage extractor.

Extracts peak and onset voltages from the hysteresis difference curve
for both positive and negative voltage branches.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging

import polars as pl

from src.models.derived_metrics import DerivedMetric
from src.derived.extractors.base import MetricExtractor
from src.derived.algorithms.hysteresis import compute_coercive_voltages

logger = logging.getLogger(__name__)


class CoerciveVoltageExtractor(MetricExtractor):
    """
    Extract characteristic voltages from IV hysteresis.

    Parameters
    ----------
    target_metric : str
        One of: "v_peak_pos", "v_onset_pos", "v_peak_neg", "v_onset_neg".
    dv_threshold : float, optional
        Minimum voltage step to classify sweep direction (default 0.05 V).
    """

    VALID_METRICS = {"v_peak_pos", "v_onset_pos", "v_peak_neg", "v_onset_neg"}

    def __init__(
        self,
        target_metric: str,
        step_size: float = 0.1,
        poly_order: int = 7,
        smooth_window: int = 15,
    ):
        if target_metric not in self.VALID_METRICS:
            raise ValueError(
                f"target_metric must be one of {self.VALID_METRICS}, got '{target_metric}'"
            )
        self._target_metric = target_metric
        self.step_size = step_size
        self.poly_order = poly_order
        self.smooth_window = smooth_window
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
        return "coercive_voltage"

    @property
    def unit(self) -> str:
        return "V"

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
                voltages = compute_coercive_voltages(
                    measurement,
                    step_size=self.step_size,
                    poly_order=self.poly_order,
                    smooth_window=self.smooth_window,
                )
                self._cache[run_id] = voltages
            except Exception as e:
                logger.error(f"Failed to compute coercive voltages for {run_id}: {e}", exc_info=True)
                return None

        voltages = self._cache[run_id]
        value = voltages.get(self._target_metric)

        if value is None or (isinstance(value, float) and value != value):
            logger.warning(f"Coercive voltage extraction returned NaN for {run_id}")
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
        return abs(result.value_float) < 20.0
