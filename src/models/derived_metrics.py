"""
Pydantic models for derived metrics schema.

Derived metrics are analytical results extracted from staged measurements,
such as charge neutrality point (CNP), photoresponse (ΔI, ΔV), mobility, etc.

Each metric has provenance tracking (extraction method, version, confidence)
and links back to the source measurement via run_id.

Schema version: 1
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ══════════════════════════════════════════════════════════════════════
# Metric Category Enum
# ══════════════════════════════════════════════════════════════════════

MetricCategory = Literal["electrical", "photoresponse", "thermal", "optical", "structural", "stability"]
"""
Metric categories for organizing derived results.

- electrical: Electrical properties (CNP, mobility, on/off ratio, etc.)
- photoresponse: Light-induced changes (ΔI, ΔV, responsivity, etc.)
- thermal: Temperature-dependent properties
- optical: Optical properties (absorption, reflection, etc.)
- stability: Drift, noise, baseline stability metrics
- structural: Physical/structural properties (thickness, roughness, etc.)
"""


# ══════════════════════════════════════════════════════════════════════
# Derived Metric Schema
# ══════════════════════════════════════════════════════════════════════

class DerivedMetric(BaseModel):
    """
    Single derived metric extracted from a measurement.

    Each metric represents an analytical result computed from raw measurement
    data, such as:
    - Charge neutrality point (CNP) from IVg/VVg
    - Photoresponse (ΔI_ds, ΔV_ds) from It/Vt
    - Mobility (μ) from IVg
    - Hysteresis voltage from IVg

    Linkage to Source Measurement
    ------------------------------
    - run_id: Foreign key to manifest.parquet (identifies source measurement)
    - chip_number: Chip numeric ID (denormalized for filtering)
    - chip_group: Chip group name (denormalized for filtering)
    - procedure: Procedure type (IVg, It, etc.)
    - seq_num: Optional sequence number from chip history

    Metric Identity
    ---------------
    - metric_name: Unique identifier (e.g., 'cnp_voltage', 'delta_ids', 'mobility')
    - metric_category: Category for organization (electrical, photoresponse, etc.)

    Metric Value (Polymorphic)
    ---------------------------
    - value_float: Numeric value (most common)
    - value_str: String value (e.g., categorical results)
    - value_json: Complex value as JSON string (e.g., multiple related values)
    - unit: Physical unit (V, A, Ω, cm²/V·s, etc.)

    Provenance & Quality
    --------------------
    - extraction_method: Algorithm/function name (e.g., 'max_resistance', 'peak_transconductance')
    - extraction_version: Code version that computed this (e.g., 'v0.1.0+g1a2b3c')
    - extraction_timestamp: When this metric was computed (UTC)
    - confidence: Quality score 0-1 (1.0 = high confidence, < 0.5 = suspect)
    - flags: Comma-separated warnings (e.g., 'CNP_AT_EDGE,NOISY_DATA')

    Example
    -------
    >>> from datetime import datetime, timezone
    >>> metric = DerivedMetric(
    ...     run_id="a1b2c3d4e5f67890",
    ...     chip_number=67,
    ...     chip_group="Alisson",
    ...     procedure="IVg",
    ...     seq_num=15,
    ...     metric_name="cnp_voltage",
    ...     metric_category="electrical",
    ...     value_float=-0.35,
    ...     unit="V",
    ...     extraction_method="max_resistance",
    ...     extraction_version="v0.1.0",
    ...     extraction_timestamp=datetime.now(timezone.utc),
    ...     confidence=1.0,
    ...     flags=None
    ... )
    """

    model_config = ConfigDict(
        extra="forbid",            # Fail on unknown fields
        validate_assignment=True,  # Validate on field updates
        arbitrary_types_allowed=False
    )

    # ═══════════════════════════════════════════════════════════════════
    # Linkage to Source Measurement (Required)
    # ═══════════════════════════════════════════════════════════════════

    run_id: str = Field(
        ...,
        min_length=16,
        max_length=64,
        description="Foreign key to manifest.parquet - identifies source measurement"
    )

    chip_number: Optional[int] = Field(
        default=None,
        ge=0,
        description="Chip numeric ID (e.g., 67) - denormalized from manifest"
    )

    chip_group: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Chip group name (e.g., 'Alisson') - denormalized from manifest"
    )

    procedure: str = Field(
        ...,
        min_length=2,
        description="Procedure type (IVg, It, etc.) - denormalized from manifest"
    )

    seq_num: Optional[int] = Field(
        default=None,
        ge=0,
        description="Sequence number from chip history (if available)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Metric Identity (Required)
    # ═══════════════════════════════════════════════════════════════════

    metric_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Unique metric identifier (e.g., 'cnp_voltage', 'delta_ids', 'mobility')"
    )

    metric_category: MetricCategory = Field(
        ...,
        description="Metric category: electrical, photoresponse, thermal, optical, structural"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Metric Value (Polymorphic - at least one required)
    # ═══════════════════════════════════════════════════════════════════

    value_float: Optional[float] = Field(
        default=None,
        description="Numeric value (most common case)"
    )

    value_str: Optional[str] = Field(
        default=None,
        max_length=500,
        description="String value for categorical results"
    )

    value_json: Optional[str] = Field(
        default=None,
        description="JSON-encoded complex value (e.g., {'peak_v': 0.5, 'peak_i': 1e-6})"
    )

    unit: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Physical unit (V, A, Ω, cm²/V·s, etc.) or 'dimensionless'"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Provenance & Quality (Required)
    # ═══════════════════════════════════════════════════════════════════

    extraction_method: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Algorithm/function name (e.g., 'max_resistance', 'mean_difference')"
    )

    extraction_version: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Code version that computed this metric (e.g., 'v0.1.0+g1a2b3c')"
    )

    extraction_timestamp: datetime = Field(
        ...,
        description="When this metric was computed (UTC, timezone-aware)"
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Quality score: 1.0 = high confidence, < 0.5 = suspect, 0.0 = failed"
    )

    flags: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Comma-separated warnings (e.g., 'CNP_AT_EDGE,NOISY_DATA')"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Validators
    # ═══════════════════════════════════════════════════════════════════

    @field_validator("run_id")
    @classmethod
    def _lowercase_runid(cls, v: str) -> str:
        """Normalize run_id to lowercase for consistency."""
        return v.strip().lower()

    @field_validator("chip_group")
    @classmethod
    def _titlecase_group(cls, v: Optional[str]) -> Optional[str]:
        """Normalize chip group to title case (e.g., 'alisson' → 'Alisson')."""
        if v is None:
            return None
        return v.strip().title()

    @field_validator("extraction_timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC)."""
        if v.tzinfo is None:
            raise ValueError(f"extraction_timestamp must be timezone-aware (UTC): {v}")
        return v

    @field_validator("metric_name")
    @classmethod
    def _lowercase_metric_name(cls, v: str) -> str:
        """Normalize metric name to lowercase with underscores."""
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    def model_post_init(self, __context) -> None:
        """Validate that at least one value field is set."""
        if self.value_float is None and self.value_str is None and self.value_json is None:
            raise ValueError("At least one of value_float, value_str, or value_json must be set")


# ══════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════

def metric_display_name(metric_name: str) -> str:
    """
    Get human-readable display name for metric.

    Parameters
    ----------
    metric_name : str
        Metric identifier (e.g., 'cnp_voltage', 'delta_ids')

    Returns
    -------
    str
        Human-readable display name

    Examples
    --------
    >>> metric_display_name("cnp_voltage")
    'Charge Neutrality Point Voltage'
    >>> metric_display_name("delta_ids")
    'Photocurrent (ΔI_ds)'
    >>> metric_display_name("mobility")
    'Field-Effect Mobility'
    """
    names = {
        "cnp_voltage": "Charge Neutrality Point Voltage",
        "cnp_resistance": "Resistance at CNP",
        "tau_dark": "Dark Relaxation Time (τ)",
        "sweep_delta_current": "Sweep ΔCurrent",
        "sweep_delta_voltage": "Sweep ΔVoltage",
        "delta_ids": "Photocurrent (ΔI_ds)",
        "delta_vds": "Photovoltage (ΔV_ds)",
        "photoresponse_ratio": "Photoresponse Ratio (ΔI/I_dark)",
        "mobility": "Field-Effect Mobility",
        "on_off_ratio": "On/Off Current Ratio",
        "hysteresis_voltage": "Hysteresis Voltage",
        "threshold_voltage": "Threshold Voltage",
        "subthreshold_swing": "Subthreshold Swing",
        "max_negative_hysteresis": "Max Negative Hysteresis (Raw)",
        "max_positive_hysteresis": "Max Positive Hysteresis (Raw)",
        "poly_corrected_max_neg_i": "Max Negative Current (Poly-Corrected)",
        "poly_corrected_max_neg_v": "Voltage at Max Negative Current (Poly-Corrected)",
        "poly_corrected_max_pos_i": "Max Positive Current (Poly-Corrected)",
        "poly_corrected_max_pos_v": "Voltage at Max Positive Current (Poly-Corrected)",
    }
    return names.get(metric_name, metric_name.replace("_", " ").title())


def format_metric_value(metric: DerivedMetric) -> str:
    """
    Format metric value with appropriate unit and precision.

    Parameters
    ----------
    metric : DerivedMetric
        Metric to format

    Returns
    -------
    str
        Formatted value string (e.g., '-0.35 V', '1.23e-6 A')

    Examples
    --------
    >>> m = DerivedMetric(
    ...     run_id="test", chip_number=67, chip_group="Alisson",
    ...     procedure="IVg", metric_name="cnp_voltage", metric_category="electrical",
    ...     value_float=-0.35, unit="V", extraction_method="test",
    ...     extraction_version="v1", extraction_timestamp=datetime.now(timezone.utc)
    ... )
    >>> format_metric_value(m)
    '-0.35 V'
    """
    if metric.value_float is not None:
        # Format based on magnitude
        val = metric.value_float
        if abs(val) >= 1000:
            formatted = f"{val:.2e}"
        elif abs(val) >= 1:
            formatted = f"{val:.2f}"
        elif abs(val) >= 0.01:
            formatted = f"{val:.3f}"
        else:
            formatted = f"{val:.2e}"

        if metric.unit:
            return f"{formatted} {metric.unit}"
        return formatted

    elif metric.value_str is not None:
        return metric.value_str

    elif metric.value_json is not None:
        return metric.value_json

    return "N/A"
