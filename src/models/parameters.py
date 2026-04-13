"""
Pydantic models for configuration and validation throughout the pipeline.

These models provide:
- Type validation and coercion
- Field-level constraints (min/max values, required fields)
- Clear documentation of expected parameters
- Default values for common configurations
- JSON serialization/deserialization support
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class StagingParameters(BaseModel):
    """
    Parameters for CSV-to-Parquet staging pipeline.

    Controls parallel processing, data validation, and output configuration
    for the staging layer that transforms raw CSVs into schema-validated Parquet files.

    Example:
        >>> params = StagingParameters(
        ...     raw_root=Path("data/01_raw"),
        ...     stage_root=Path("data/02_stage/raw_measurements"),
        ...     procedures_yaml=Path("config/procedures.yml"),
        ...     workers=8,
        ...     force=True
        ... )
    """

    # Required paths
    raw_root: Path = Field(
        ...,
        description="Root directory containing raw CSV files"
    )
    stage_root: Path = Field(
        ...,
        description="Output root for staged Parquet files"
    )
    procedures_yaml: Path = Field(
        ...,
        description="Path to YAML schema defining procedure types and columns"
    )

    # Optional paths (with defaults)
    rejects_dir: Optional[Path] = Field(
        None,
        description="Directory for reject records (default: {stage_root}/../_rejects)"
    )
    events_dir: Optional[Path] = Field(
        None,
        description="Directory for per-run event JSONs (default: {stage_root}/_manifest/events)"
    )
    manifest: Optional[Path] = Field(
        None,
        description="Path to consolidated manifest Parquet file (default: {stage_root}/_manifest/manifest.parquet)"
    )

    # Parallel processing
    workers: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Number of parallel worker processes"
    )
    polars_threads: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Polars threads per worker process"
    )

    # Timezone and behavior
    local_tz: str = Field(
        default="America/Santiago",
        description="IANA timezone name for date partitioning"
    )
    force: bool = Field(
        default=False,
        description="Overwrite existing Parquet files if True"
    )
    only_yaml_data: bool = Field(
        default=False,
        description="Drop columns not defined in YAML schema if True"
    )
    strict: bool = Field(
        default=False,
        description="Strict validation mode - fail on schema validation errors (missing required columns, etc.)"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"  # Reject unknown parameters
    )

    @field_validator("raw_root", "procedures_yaml")
    @classmethod
    def path_must_exist(cls, v: Path) -> Path:
        """Validate that required input paths exist."""
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    @field_validator("procedures_yaml")
    @classmethod
    def yaml_must_be_file(cls, v: Path) -> Path:
        """Validate that YAML path is a file."""
        if not v.is_file():
            raise ValueError(f"Not a file: {v}")
        return v

    @model_validator(mode="after")
    def set_default_paths(self) -> StagingParameters:
        """Set default paths based on stage_root if not provided."""
        if self.rejects_dir is None:
            self.rejects_dir = self.stage_root.parent / "_rejects"
        if self.events_dir is None:
            self.events_dir = self.stage_root / "_manifest" / "events"
        if self.manifest is None:
            self.manifest = self.stage_root / "_manifest" / "manifest.parquet"
        return self


class IntermediateParameters(BaseModel):
    """
    Parameters for intermediate preprocessing layer.

    Controls procedure-specific preprocessing between staging and analysis.
    For IV procedures, this includes segment detection and classification.

    Example:
        >>> params = IntermediateParameters(
        ...     stage_root=Path("data/02_stage/raw_measurements"),
        ...     output_root=Path("data/03_intermediate"),
        ...     procedure="IV",
        ...     voltage_col="Vsd (V)",
        ...     workers=8
        ... )
    """

    # Required paths
    stage_root: Path = Field(
        ...,
        description="Root directory of staged Parquet data"
    )
    output_root: Path = Field(
        ...,
        description="Root directory for intermediate processed data"
    )

    # Processing configuration
    procedure: str = Field(
        default="IV",
        pattern=r"^(IV|IVg|IVgT)$",
        description="Procedure name (IV, IVg, IVgT)"
    )
    voltage_col: str = Field(
        default="Vsd (V)",
        description="Voltage column name for segment detection"
    )

    # Segment detection parameters
    dv_threshold: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Minimum voltage change to consider valid (filters noise)"
    )
    min_segment_points: int = Field(
        default=5,
        ge=2,
        le=1000,
        description="Minimum points required for valid segment"
    )

    # Parallel processing
    workers: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Number of parallel worker processes"
    )
    polars_threads: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Polars threads per worker process"
    )

    # Behavior
    force: bool = Field(
        default=False,
        description="Overwrite existing intermediate files if True"
    )

    # Optional paths (with defaults)
    events_dir: Optional[Path] = Field(
        None,
        description="Directory for event JSONs (default: {output_root}/_manifest/events)"
    )
    manifest: Optional[Path] = Field(
        None,
        description="Path to processing manifest (default: {output_root}/_manifest/manifest.parquet)"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )

    @field_validator("stage_root")
    @classmethod
    def stage_root_must_exist(cls, v: Path) -> Path:
        """Validate that stage root exists."""
        if not v.exists():
            raise ValueError(f"Stage root does not exist: {v}")
        return v

    @model_validator(mode="after")
    def set_default_paths(self) -> IntermediateParameters:
        """Set default paths based on output_root if not provided."""
        if self.events_dir is None:
            self.events_dir = self.output_root / "_manifest" / "events"
        if self.manifest is None:
            self.manifest = self.output_root / "_manifest" / "manifest.parquet"
        return self

    def get_output_dir(self) -> Path:
        """Get output directory for this procedure's intermediate data."""
        # Map procedure to subdirectory name
        proc_subdir = {
            "IV": "iv_segments",
            "IVg": "ivg_segments",
            "IVgT": "ivgt_segments",
        }.get(self.procedure, f"{self.procedure.lower()}_segments")

        return self.output_root / proc_subdir


class IVAnalysisParameters(BaseModel):
    """
    Parameters for IV curve analysis pipeline.

    Controls statistical aggregation, polynomial fitting, hysteresis calculation,
    and peak detection for IV measurement analysis.

    Example:
        >>> params = IVAnalysisParameters(
        ...     stage_root=Path("data/02_stage/raw_measurements"),
        ...     date="2025-09-11",
        ...     output_base_dir=Path("data/04_analysis"),
        ...     poly_orders=[1, 3, 5, 7],
        ...     fit_backward=True
        ... )
    """

    # Required inputs
    stage_root: Path = Field(
        ...,
        description="Root directory of staged Parquet data"
    )
    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date to analyze in YYYY-MM-DD format"
    )
    output_base_dir: Path = Field(
        ...,
        description="Base directory for analysis outputs"
    )

    # Optional intermediate layer
    intermediate_root: Optional[Path] = Field(
        None,
        description="Root directory for intermediate data (if using 4-layer architecture). If provided, analysis reads from here instead of stage_root."
    )
    use_segments: bool = Field(
        default=False,
        description="If True, read from segmented intermediate data instead of raw staged data"
    )

    # Filtering
    procedure: str = Field(
        default="IV",
        description="Procedure name (IV, IVg, etc.)"
    )
    chip_number: Optional[str] = Field(
        None,
        description="Filter by chip number (optional)"
    )
    v_max: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Filter by specific V_max value (optional)"
    )

    # Polynomial fitting
    poly_orders: List[int] = Field(
        default=[1, 3, 5, 7],
        description="Polynomial orders to fit (e.g., [1, 3, 5, 7])"
    )
    fit_backward: bool = Field(
        default=True,
        description="Fit polynomial to backward (return) trace"
    )

    # Hysteresis computation
    compute_hysteresis: bool = Field(
        default=True,
        description="Compute hysteresis (forward - backward)"
    )
    voltage_rounding_decimals: int = Field(
        default=2,
        ge=0,
        le=6,
        description="Decimal places for voltage rounding during alignment"
    )

    # Peak detection
    analyze_peaks: bool = Field(
        default=False,
        description="Analyze hysteresis peak locations"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )

    @field_validator("poly_orders")
    @classmethod
    def validate_poly_orders(cls, v: List[int]) -> List[int]:
        """Validate polynomial orders are positive and reasonable."""
        if not v:
            raise ValueError("Must specify at least one polynomial order")
        for order in v:
            if order < 1:
                raise ValueError(f"Polynomial order must be >= 1, got {order}")
            if order > 15:
                raise ValueError(f"Polynomial order {order} too high (max 15)")
            if order % 2 == 0:
                raise ValueError(f"Polynomial order {order} should be odd for symmetric fitting")
        return sorted(set(v))  # Remove duplicates and sort

    @field_validator("stage_root")
    @classmethod
    def stage_root_must_exist(cls, v: Path) -> Path:
        """Validate that stage root exists."""
        if not v.exists():
            raise ValueError(f"Stage root does not exist: {v}")
        return v

    def get_stats_dir(self) -> Path:
        """Get output directory for IV statistics."""
        return self.output_base_dir / "iv_stats" / f"{self.date}_{self.procedure}"

    def get_hysteresis_dir(self) -> Path:
        """Get output directory for hysteresis data."""
        return self.output_base_dir / "hysteresis" / f"{self.date}_{self.procedure}"

    def get_peaks_dir(self) -> Path:
        """Get output directory for peak analysis."""
        return self.output_base_dir / "hysteresis_peaks" / f"{self.date}_{self.procedure}"


class PlottingParameters(BaseModel):
    """
    Parameters for visualization and plotting.

    Controls figure quality, formatting, and output specifications for
    publication-ready plots.

    Example:
        >>> params = PlottingParameters(
        ...     output_dir=Path("plots/analysis_2025"),
        ...     dpi=300,
        ...     format="png",
        ...     style="publication"
        ... )
    """

    # Output configuration
    output_dir: Path = Field(
        ...,
        description="Directory for plot outputs"
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=1200,
        description="DPI resolution for raster outputs"
    )
    format: str = Field(
        default="png",
        pattern=r"^(png|pdf|svg|jpg)$",
        description="Output format: png, pdf, svg, or jpg"
    )

    # Figure dimensions
    figure_width: float = Field(
        default=12.0,
        ge=4.0,
        le=30.0,
        description="Figure width in inches"
    )
    figure_height: float = Field(
        default=8.0,
        ge=3.0,
        le=20.0,
        description="Figure height in inches"
    )

    # Style and appearance
    style: str = Field(
        default="publication",
        pattern=r"^(publication|presentation|notebook)$",
        description="Plot style preset: publication, presentation, or notebook"
    )
    font_size: int = Field(
        default=10,
        ge=6,
        le=24,
        description="Base font size for labels and text"
    )
    line_width: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Default line width"
    )

    # Advanced options
    show_error_bars: bool = Field(
        default=True,
        description="Display error bars where applicable"
    )
    show_grid: bool = Field(
        default=True,
        description="Display grid lines"
    )
    grid_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Grid transparency (0=invisible, 1=opaque)"
    )

    # Comparison plots
    compact_layout: bool = Field(
        default=False,
        description="Use compact subplot layout for multi-panel figures"
    )
    show_residuals: bool = Field(
        default=False,
        description="Include residuals plots for polynomial fits"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )

    def get_figsize(self) -> tuple[float, float]:
        """Get figure size as (width, height) tuple."""
        return (self.figure_width, self.figure_height)

    def get_style_params(self) -> dict:
        """Get matplotlib rcParams based on style preset."""
        base_params = {
            "font.size": self.font_size,
            "lines.linewidth": self.line_width,
            "grid.alpha": self.grid_alpha,
        }

        if self.style == "publication":
            base_params.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "DejaVu Sans"],
                "axes.linewidth": 1.2,
                "xtick.major.width": 1.0,
                "ytick.major.width": 1.0,
            })
        elif self.style == "presentation":
            base_params.update({
                "font.size": self.font_size + 2,
                "axes.linewidth": 1.5,
                "lines.linewidth": self.line_width + 0.5,
            })

        return base_params


class PipelineParameters(BaseModel):
    """
    Combined parameters for complete pipeline execution.

    Supports both 3-layer and 4-layer architectures:
    - 3-layer: Stage → Analysis → Plotting
    - 4-layer: Stage → Intermediate → Analysis → Plotting

    Aggregates all sub-parameter models for end-to-end pipeline runs.

    Example (3-layer):
        >>> params = PipelineParameters(
        ...     staging=StagingParameters(...),
        ...     analysis=IVAnalysisParameters(...),
        ...     plotting=PlottingParameters(...)
        ... )

    Example (4-layer):
        >>> params = PipelineParameters(
        ...     staging=StagingParameters(...),
        ...     intermediate=IntermediateParameters(...),
        ...     analysis=IVAnalysisParameters(..., use_segments=True),
        ...     plotting=PlottingParameters(...),
        ...     run_intermediate=True
        ... )
    """

    staging: StagingParameters = Field(
        ...,
        description="Staging pipeline parameters"
    )
    intermediate: Optional[IntermediateParameters] = Field(
        None,
        description="Intermediate preprocessing parameters (optional for 4-layer architecture)"
    )
    analysis: IVAnalysisParameters = Field(
        ...,
        description="IV analysis parameters"
    )
    plotting: PlottingParameters = Field(
        ...,
        description="Plotting parameters"
    )

    # Pipeline control
    run_staging: bool = Field(
        default=True,
        description="Execute staging step"
    )
    run_intermediate: bool = Field(
        default=False,
        description="Execute intermediate preprocessing step (4-layer architecture)"
    )
    run_analysis: bool = Field(
        default=True,
        description="Execute analysis step"
    )
    run_plotting: bool = Field(
        default=True,
        description="Execute plotting step"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> PipelineParameters:
        """Validate cross-parameter consistency for 3-layer or 4-layer architecture."""
        # Ensure analysis stage_root matches staging output
        if self.analysis.stage_root != self.staging.stage_root:
            raise ValueError(
                f"Analysis stage_root ({self.analysis.stage_root}) must match "
                f"staging stage_root ({self.staging.stage_root})"
            )

        # If using intermediate layer, validate consistency
        if self.run_intermediate:
            if self.intermediate is None:
                raise ValueError(
                    "run_intermediate=True but no intermediate parameters provided. "
                    "Either set run_intermediate=False or provide intermediate parameters."
                )

            # Intermediate stage_root must match staging output
            if self.intermediate.stage_root != self.staging.stage_root:
                raise ValueError(
                    f"Intermediate stage_root ({self.intermediate.stage_root}) must match "
                    f"staging stage_root ({self.staging.stage_root})"
                )

            # Analysis should read from intermediate
            if not self.analysis.use_segments:
                raise ValueError(
                    "run_intermediate=True but analysis.use_segments=False. "
                    "Set use_segments=True to read from intermediate layer."
                )

            # Set intermediate_root in analysis if not already set
            if self.analysis.intermediate_root is None:
                self.analysis.intermediate_root = self.intermediate.get_output_dir()

        # Ensure analysis runs before plotting if both enabled
        if self.run_plotting and not self.run_analysis:
            # Check if analysis outputs exist
            stats_dir = self.analysis.get_stats_dir()
            if not stats_dir.exists():
                raise ValueError(
                    "Cannot run plotting without analysis: "
                    f"analysis output directory {stats_dir} does not exist. "
                    "Either enable run_analysis or run analysis separately first."
                )

        return self

    @classmethod
    def from_json(cls, path: Path | str) -> PipelineParameters:
        """
        Load parameters from JSON configuration file.

        Args:
            path: Path to JSON config file

        Returns:
            Validated PipelineParameters instance

        Example JSON structure:
            {
                "staging": {
                    "raw_root": "data/01_raw",
                    "stage_root": "data/02_stage/raw_measurements",
                    "procedures_yaml": "config/procedures.yml",
                    "workers": 8
                },
                "analysis": {
                    "stage_root": "data/02_stage/raw_measurements",
                    "date": "2025-09-11",
                    "output_base_dir": "data/04_analysis",
                    "poly_orders": [1, 3, 5, 7]
                },
                "plotting": {
                    "output_dir": "plots/2025-09-11",
                    "dpi": 300
                }
            }
        """
        import json

        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        return cls(**data)

    def to_json(self, path: Path | str, indent: int = 2) -> None:
        """
        Save parameters to JSON configuration file.

        Args:
            path: Path to output JSON file
            indent: JSON indentation level
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(
                self.model_dump(mode="json"),
                f,
                indent=indent,
                default=str
            )

    def validate_all_paths(self) -> None:
        """
        Validate all input paths exist and create output directories.

        Raises:
            ValueError: If required input paths don't exist
        """
        # Input validations already handled by field validators

        # Create output directories
        self.staging.stage_root.mkdir(parents=True, exist_ok=True)
        if self.staging.rejects_dir:
            self.staging.rejects_dir.mkdir(parents=True, exist_ok=True)
        if self.staging.events_dir:
            self.staging.events_dir.mkdir(parents=True, exist_ok=True)

        self.analysis.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.plotting.output_dir.mkdir(parents=True, exist_ok=True)
