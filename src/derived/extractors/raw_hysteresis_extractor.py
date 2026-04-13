"""
Raw hysteresis extractor.

Computes max_negative_hysteresis and max_positive_hysteresis metrics
from IV sweep data using I_forward - I_backward at each voltage.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging

import polars as pl

from src.models.derived_metrics import DerivedMetric
from src.derived.extractors.base import MetricExtractor
from src.derived.algorithms.hysteresis import (
    compute_raw_hysteresis,
    extract_raw_hysteresis_maxima,
)

logger = logging.getLogger(__name__)


class RawHysteresisExtractor(MetricExtractor):
    """
    Extract maximum raw hysteresis from IV sweeps.

    Computes I_forward - I_backward at each voltage, then returns
    the maximum value for negative and positive voltage branches.

    Parameters
    ----------
    target_metric : str
        Either "max_negative_hysteresis" or "max_positive_hysteresis"
    dv_threshold : float, optional
        Minimum voltage step to classify sweep direction (default 0.05 V).
    """

    VALID_METRICS = {"max_negative_hysteresis", "max_positive_hysteresis"}

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
        return "raw_hysteresis"

    @property
    def unit(self) -> str:
        return "A"

    def extract(
        self,
        measurement: pl.DataFrame,
        metadata: Dict[str, Any],
    ) -> Optional[DerivedMetric]:
        """
        Extract raw hysteresis maximum from IV measurement.

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
                hysteresis_df = compute_raw_hysteresis(
                    measurement, dv_threshold=self.dv_threshold
                )
                maxima = extract_raw_hysteresis_maxima(hysteresis_df)
                self._cache[run_id] = maxima
            except Exception as e:
                logger.error(f"Failed to compute raw hysteresis for {run_id}: {e}", exc_info=True)
                return None

        maxima = self._cache[run_id]
        value = maxima.get(self._target_metric)

        if value is None or (isinstance(value, float) and value != value):  # NaN check
            logger.warning(f"Raw hysteresis extraction returned NaN for {run_id}")
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
        """Validate that hysteresis value is physically reasonable."""
        if result.value_float is None:
            return False
        # Hysteresis should be non-negative (I_fwd - I_bwd >= 0 for typical devices)
        # Allow small negative values due to noise
        return result.value_float > -1e-3
