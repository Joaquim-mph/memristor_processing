"""
On/off current ratio extractor.

Computes |I_on / I_off| at a fixed read voltage from IV sweep data.
Key metric for memory applications.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging

import polars as pl

from src.models.derived_metrics import DerivedMetric
from src.derived.extractors.base import MetricExtractor
from src.derived.algorithms.hysteresis import compute_on_off_ratio

logger = logging.getLogger(__name__)


class OnOffRatioExtractor(MetricExtractor):
    """
    Extract on/off current ratio at a fixed read voltage from IV sweeps.

    Parameters
    ----------
    read_voltage : float
        Voltage at which to evaluate the ratio (default 2.0 V).
    dv_threshold : float
        Minimum voltage step to classify sweep direction (default 0.05 V).
    voltage_tolerance : float
        Maximum distance from read_voltage to accept a data point (default 0.15 V).
    """

    def __init__(
        self,
        read_voltage: float = 2.0,
        dv_threshold: float = 0.05,
        voltage_tolerance: float = 0.15,
    ):
        self.read_voltage = read_voltage
        self.dv_threshold = dv_threshold
        self.voltage_tolerance = voltage_tolerance
        self._cache: Dict[str, dict] = {}

    @property
    def applicable_procedures(self) -> list[str]:
        return ["IV"]

    @property
    def metric_name(self) -> str:
        return "on_off_ratio"

    @property
    def metric_category(self) -> str:
        return "electrical"

    @property
    def extraction_method(self) -> str:
        return "on_off_ratio"

    @property
    def unit(self) -> str:
        return "dimensionless"

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
                result = compute_on_off_ratio(
                    measurement,
                    read_voltage=self.read_voltage,
                    dv_threshold=self.dv_threshold,
                    voltage_tolerance=self.voltage_tolerance,
                )
                self._cache[run_id] = result
            except Exception as e:
                logger.error(f"Failed to compute on/off ratio for {run_id}: {e}", exc_info=True)
                return None

        result = self._cache[run_id]
        value = result.get("on_off_ratio")

        if value is None or (isinstance(value, float) and value != value):
            logger.warning(f"On/off ratio extraction returned NaN for {run_id}")
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
        return result.value_float >= 1.0
