"""
Pydantic models for manifest schema validation.

The manifest.parquet file is the authoritative source of truth for all measurement
metadata. Each row represents one measurement run with complete metadata extracted
from CSV headers, filenames, and data analysis.

Schema version: 1
"""

from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ══════════════════════════════════════════════════════════════════════
# Procedure Type Enum
# ══════════════════════════════════════════════════════════════════════

Proc = Literal[
    "IVg", "IV", "IVgT", "It", "It2", "ITt", "ItVg", "ItWl",
    "LaserCalibration", "Pt", "Pwl", "Tt", "Vt", "VVg",
]
"""
Procedure types from procedures.yml.

- IVg: Gate voltage sweep (Vg vs I)
- IV: Drain voltage sweep (Vsd vs I)
- IVgT: Gate voltage sweep with temperature
- It: Current vs time (photoresponse, time series)
- It2: Current vs time, 3-phase (OFF/ON/OFF)
- ITt: Current vs time with temperature
- ItVg: Current vs time with gate sweep
- ItWl: Current vs time with wavelength sweep
- LaserCalibration: Laser power calibration
- Pt: Power vs time
- Pwl: Power vs wavelength
- Tt: Temperature vs time
- Vt: Voltage vs time
- VVg: Voltage vs gate voltage sweep
"""


# ══════════════════════════════════════════════════════════════════════
# Manifest Row Schema
# ══════════════════════════════════════════════════════════════════════

class ManifestRow(BaseModel):
    """
    Single row in the authoritative manifest.parquet table.

    Each row represents one measurement run with complete metadata extracted
    from CSV headers, filenames, and measurement data.

    Identity & Partitioning
    -----------------------
    - run_id: Deterministic SHA-1 hash (path|timestamp_utc), 16 chars
    - source_file: Relative path from raw_root (e.g., "Alisson_15_sept/Alisson67_001.csv")
    - proc: Procedure type from procedures.yml
    - date_local: Local calendar date (America/Santiago) for Hive partitioning
    - start_time_utc: Measurement start timestamp (UTC, timezone-aware)

    Chip Identification
    -------------------
    - chip_group: Extracted from filename (e.g., "Alisson" from Alisson67_015.csv)
    - chip_number: Numeric chip ID (e.g., 67)
    - chip_name: Full chip name (e.g., "Alisson67")
    - file_idx: File number from filename (e.g., 15 from Alisson67_015.csv)

    Experiment Descriptors
    ----------------------
    Light status:
    - has_light: True (light), False (dark), None (unknown)
    - laser_voltage_V: Laser/LED voltage (V < 0.1 = dark, V >= 0.1 = light)
    - wavelength_nm: Laser wavelength (nm)
    - ids_a: Drain-source current (A) from metadata
    - laser_period_s: Laser ON+OFF period (s) for It/ITt procedures

    Voltage parameters (procedure-specific):
    - vg_fixed_v: Fixed gate voltage (It, IV procedures)
    - vg_start_v, vg_end_v, vg_step_v: IVg sweep parameters
    - vds_v: Drain-source voltage (It, IVg procedures)
    - vsd_start_v, vsd_end_v, vsd_step_v: IV sweep parameters

    Measurement parameters:
    - duration_s: Measurement duration calculated from data
    - sampling_time_s: Sampling time (excluding Keithley settling)

    Instrument settings:
    - irange: Current measurement range
    - nplc: Number of power line cycles (integration time)
    - n_avg: Number of averages
    - burn_in_time_s: Burn-in time before measurement starts (IVg, IV)
    - step_time_s: Time per voltage step (IVg, IV)

    Temperature (IVgT, It, ITt, Tt procedures):
    - initial_temp_c: Initial temperature (degC)
    - target_temp_c: Target temperature (degC)
    - temp_start_c, temp_end_c, temp_step_c: Temperature sweep parameters (Tt)
    - temp_step_start_time_s: Time when temperature step starts (It)

    UX & Governance
    ---------------
    - summary: Human-readable description (e.g., "It (Vg=-3V, λ=455nm, 120s)")
    - schema_version: Manifest schema version (bump for breaking changes)
    - extraction_version: Parser version (e.g., "v0.4.2+g1a2b3c" from git describe)
    - ingested_at_utc: Staging timestamp (when this row was created)

    Example
    -------
    >>> from datetime import datetime, timezone
    >>> row = ManifestRow(
    ...     run_id="a1b2c3d4e5f67890",
    ...     source_file=Path("Alisson_15_sept/Alisson67_015.csv"),
    ...     proc="It",
    ...     date_local=date(2025, 10, 18),
    ...     start_time_utc=datetime(2025, 10, 18, 17, 30, 0, tzinfo=timezone.utc),
    ...     chip_group="Alisson",
    ...     chip_number=67,
    ...     chip_name="Alisson67",
    ...     file_idx=15,
    ...     has_light=True,
    ...     laser_voltage_V=3.5,
    ...     wavelength_nm=455.0,
    ...     laser_period_s=120.0,
    ...     vg_fixed_v=-3.0,
    ...     vds_v=0.1,
    ...     duration_s=3600.0,
    ...     summary="It (Vg=-3V, λ=455nm, 120s)",
    ...     schema_version=1,
    ...     extraction_version="v0.4.2+g1a2b3c",
    ...     ingested_at_utc=datetime.now(timezone.utc)
    ... )
    """

    model_config = ConfigDict(
        extra="forbid",            # Fail on unknown fields (prevent drift)
        validate_assignment=True,  # Validate on field updates
        arbitrary_types_allowed=True  # Allow Path objects
    )

    # ═══════════════════════════════════════════════════════════════════
    # Identity & Partitioning (Required)
    # ═══════════════════════════════════════════════════════════════════

    run_id: str = Field(
        ...,
        min_length=16,
        max_length=64,
        description="SHA-1 hash of (normalized_path|timestamp_utc), truncated to 16 chars, lowercase"
    )

    source_file: Path = Field(
        ...,
        description="Relative path from raw_root (e.g., 'Alisson_15_sept/Alisson67_001.csv')"
    )

    proc: Proc = Field(
        ...,
        description="Procedure type: IVg, IV, IVgT, It, It2, ITt, ItVg, ItWl, LaserCalibration, Pt, Pwl, Tt, Vt, VVg"
    )

    date_local: date = Field(
        ...,
        description="Local calendar date (America/Santiago) for Hive partitioning"
    )

    start_time_utc: datetime = Field(
        ...,
        description="Measurement start timestamp in UTC (timezone-aware)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Chip Identification (Optional - extracted from filename + metadata)
    # ═══════════════════════════════════════════════════════════════════

    chip_group: Optional[str] = Field(
        default=None,
        description="Chip group prefix (e.g., 'Alisson' from Alisson67_015.csv)"
    )

    chip_number: Optional[int] = Field(
        default=None,
        ge=0,
        description="Chip numeric ID (e.g., 67 from Alisson67_015.csv)"
    )

    chip_name: Optional[str] = Field(
        default=None,
        description="Full chip name (e.g., 'Alisson67')"
    )

    file_idx: Optional[int] = Field(
        default=None,
        ge=0,
        description="File number from filename (e.g., 15 from Alisson67_015.csv)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Light Status & Laser Parameters
    # ═══════════════════════════════════════════════════════════════════

    has_light: Optional[bool] = Field(
        default=None,
        description="Light illumination status: True (light), False (dark), None (unknown)"
    )

    laser_voltage_V: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Laser/LED voltage (V) - V < 0.1 = dark, V >= 0.1 = light"
    )

    wavelength_nm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Laser wavelength (nm) - typical: 455, 530, 625, etc. 0 = dark/no laser"
    )

    ids_a: Optional[float] = Field(
        default=None,
        description="Drain-source current (A) from metadata"
    )

    laser_period_s: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Laser ON+OFF period (s) - It, ITt procedures only"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Voltage Parameters (Procedure-Specific)
    # ═══════════════════════════════════════════════════════════════════

    # Gate voltage (fixed for It, IV; swept for IVg)
    vg_fixed_v: Optional[float] = Field(
        default=None,
        description="Fixed gate voltage (V) - It, IV, ITt procedures"
    )

    vg_start_v: Optional[float] = Field(
        default=None,
        description="Gate voltage sweep start (V) - IVg, IVgT procedures"
    )

    vg_end_v: Optional[float] = Field(
        default=None,
        description="Gate voltage sweep end (V) - IVg, IVgT procedures"
    )

    vg_step_v: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Gate voltage step size (V) - IVg, IVgT procedures"
    )

    # Drain-source voltage (fixed for It, IVg; swept for IV)
    vds_v: Optional[float] = Field(
        default=None,
        description="Drain-source voltage (V) - It, IVg, ITt, IVgT procedures"
    )

    vsd_start_v: Optional[float] = Field(
        default=None,
        description="Source-drain voltage sweep start (V) - IV procedure"
    )

    vsd_end_v: Optional[float] = Field(
        default=None,
        description="Source-drain voltage sweep end (V) - IV procedure"
    )

    vsd_step_v: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Source-drain voltage step size (V) - IV procedure"
    )

    # VVg/Vt instrument setting
    vrange: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Voltage measurement range (V) - VVg, Vt procedures"
    )

    # It2 3-phase durations
    phase1_duration_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Phase 1 duration, laser OFF (s) - It2 procedure"
    )

    phase2_duration_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Phase 2 duration, laser ON (s) - It2 procedure"
    )

    phase3_duration_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Phase 3 duration, laser OFF (s) - It2 procedure"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Measurement Parameters
    # ═══════════════════════════════════════════════════════════════════

    duration_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Measurement duration (s) calculated from data (max time)"
    )

    sampling_time_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Sampling time excluding Keithley settling (s) - It, ITt procedures"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Instrument Settings
    # ═══════════════════════════════════════════════════════════════════

    irange: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Current measurement range (A) - Keithley setting"
    )

    nplc: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of power line cycles (integration time) - Keithley setting"
    )

    n_avg: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of averages per measurement point"
    )

    burn_in_time_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Burn-in time before measurement starts (s) - IVg, IV, IVgT procedures"
    )

    step_time_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time per voltage step (s) - IVg, IV, IVgT procedures"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Temperature Parameters (IVgT, It, ITt, Tt procedures)
    # ═══════════════════════════════════════════════════════════════════

    initial_temp_c: Optional[float] = Field(
        default=None,
        description="Initial (current) temperature (degC) - It, ITt, Tt procedures"
    )

    target_temp_c: Optional[float] = Field(
        default=None,
        description="Target temperature (degC) - It, ITt, Tt procedures"
    )

    temp_start_c: Optional[float] = Field(
        default=None,
        description="Temperature sweep start (degC) - Tt procedure"
    )

    temp_end_c: Optional[float] = Field(
        default=None,
        description="Temperature sweep end (degC) - Tt procedure"
    )

    temp_step_c: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Temperature step size (degC) - Tt procedure"
    )

    temp_step_start_time_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time when temperature step starts (s) - It procedure"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Laser Calibration Parameters (LaserCalibration procedure)
    # ═══════════════════════════════════════════════════════════════════

    optical_fiber: Optional[str] = Field(
        default=None,
        description="Optical fiber identifier - LaserCalibration procedure"
    )

    laser_voltage_start_v: Optional[float] = Field(
        default=None,
        description="Laser voltage sweep start (V) - LaserCalibration procedure"
    )

    laser_voltage_end_v: Optional[float] = Field(
        default=None,
        description="Laser voltage sweep end (V) - LaserCalibration procedure"
    )

    laser_voltage_step_v: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Laser voltage step size (V) - LaserCalibration procedure"
    )

    sensor_model: Optional[str] = Field(
        default=None,
        description="Power sensor model - LaserCalibration procedure"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Metadata & Additional Info
    # ═══════════════════════════════════════════════════════════════════

    sample: Optional[str] = Field(
        default=None,
        description="Sample identifier from CSV metadata"
    )

    information: Optional[str] = Field(
        default=None,
        description="Additional information field from CSV metadata"
    )

    # ═══════════════════════════════════════════════════════════════════
    # UX & Governance (Required)
    # ═══════════════════════════════════════════════════════════════════

    summary: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Human-readable summary (e.g., 'It (Vg=-3V, λ=455nm, 120s)')"
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        description="Manifest schema version - bump for breaking changes"
    )

    extraction_version: Optional[str] = Field(
        default=None,
        description="Parser version (e.g., 'v0.4.2+g1a2b3c' from git describe)"
    )

    ingested_at_utc: datetime = Field(
        ...,
        description="Staging timestamp in UTC (when this row was created)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Staging Internals
    # ═══════════════════════════════════════════════════════════════════

    status: Optional[str] = Field(
        default=None,
        description="Staging result status: 'ok', 'skipped', or 'reject'"
    )

    path: Optional[str] = Field(
        default=None,
        description="Output Parquet file path (relative to stage root)"
    )

    rows: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of data rows in the staged measurement"
    )

    date_origin: Optional[str] = Field(
        default=None,
        description="Source of partition date: 'meta', 'path', or 'mtime'"
    )

    validation_errors: Optional[int] = Field(
        default=None,
        ge=0,
        description="Count of validation errors from staging"
    )

    validation_warnings: Optional[int] = Field(
        default=None,
        ge=0,
        description="Count of validation warnings from staging"
    )

    validation_messages: Optional[list[str]] = Field(
        default=None,
        description="Validation messages from staging (warnings, errors, info)"
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

    @field_validator("start_time_utc", "ingested_at_utc")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC)."""
        if v.tzinfo is None:
            raise ValueError(f"Datetime must be timezone-aware (UTC): {v}")
        return v

    @field_validator("source_file", mode="before")
    @classmethod
    def _coerce_source_file(cls, v) -> Path:
        """Coerce source_file to Path if string."""
        if isinstance(v, str):
            return Path(v)
        return v


# ══════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════

def proc_display_name(proc: Proc) -> str:
    """
    Get display name for procedure type.

    Parameters
    ----------
    proc : Proc
        Procedure type

    Returns
    -------
    str
        Human-readable display name

    Examples
    --------
    >>> proc_display_name("It")
    'Current vs Time'
    >>> proc_display_name("IVg")
    'Gate Voltage Sweep'
    """
    names = {
        "IVg": "Gate Voltage Sweep",
        "IV": "Drain Voltage Sweep",
        "IVgT": "Gate Voltage Sweep (Temperature)",
        "It": "Current vs Time",
        "It2": "Current vs Time (3-Phase)",
        "ITt": "Current vs Time (Temperature)",
        "ItVg": "Current vs Time (Gate Sweep)",
        "ItWl": "Current vs Time (Wavelength Sweep)",
        "LaserCalibration": "Laser Power Calibration",
        "Pt": "Power vs Time",
        "Pwl": "Power vs Wavelength",
        "Tt": "Temperature vs Time",
        "Vt": "Voltage vs Time",
        "VVg": "Voltage vs Gate Voltage",
    }
    return names.get(proc, proc)


def proc_short_name(proc: Proc) -> str:
    """
    Get short abbreviation for procedure type.

    Parameters
    ----------
    proc : Proc
        Procedure type

    Returns
    -------
    str
        Short abbreviation (2-4 chars)

    Examples
    --------
    >>> proc_short_name("It")
    'ITS'
    >>> proc_short_name("IVg")
    'IVg'
    """
    # Map procedures.yml names to standard abbreviations
    names = {
        "IVg": "IVg",
        "IV": "IV",
        "IVgT": "IVgT",
        "It": "ITS",      # Current vs Time → ITS (I-T Series)
        "It2": "ITS2",    # Current vs Time 3-Phase
        "ITt": "ITS",     # Temperature variant also ITS
        "ItVg": "ItVg",
        "ItWl": "ItWl",
        "LaserCalibration": "Cal",
        "Pt": "Pt",
        "Pwl": "Pwl",
        "Tt": "Tt",
        "Vt": "Vt",
        "VVg": "VVg",
    }
    return names.get(proc, proc)
