"""
Pydantic models for staging pipeline configuration.

These models define the configuration parameters for the staging layer that
transforms raw CSV files into schema-validated Parquet files with a centralized
manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class StagingConfig(BaseModel):
    """
    Configuration for CSV-to-Parquet staging pipeline.

    Controls parallel processing, data validation, and output configuration
    for the staging layer that transforms raw CSVs into schema-validated Parquet files.

    Required Paths
    --------------
    - raw_root: Root directory containing raw CSV files (must exist)
    - stage_root: Output root for staged Parquet files
    - procedures_yaml: Path to YAML schema defining procedure types and columns (must exist)

    Optional Paths (auto-filled from stage_root if not provided)
    --------------------------------------------------------------
    - rejects_dir: Directory for reject records (default: {stage_root}/../_rejects)
    - events_dir: Directory for per-run event JSONs (default: {stage_root}/_manifest/events)
    - manifest_path: Path to manifest Parquet file (default: {stage_root}/_manifest/manifest.parquet)

    Performance Settings
    --------------------
    - workers: Number of parallel worker processes (1-32, default: 6)
    - polars_threads: Polars threads per worker process (1-16, default: 1)

    Behavior
    --------
    - local_tz: IANA timezone name for date partitioning (default: "America/Santiago")
    - force: Overwrite existing Parquet files if True (default: False)
    - only_yaml_data: Drop columns not in YAML schema if True (default: False)
    - extraction_version: Parser version (auto-detected from git if None)

    Example
    -------
    >>> from pathlib import Path
    >>> cfg = StagingConfig(
    ...     raw_root=Path("data/01_raw"),
    ...     stage_root=Path("data/02_stage/raw_measurements"),
    ...     procedures_yaml=Path("config/procedures.yml"),
    ...     workers=8,
    ...     force=True
    ... )
    >>> print(cfg.manifest_path)
    data/02_stage/_manifest/manifest.parquet
    >>> print(cfg.extraction_version)
    v0.4.2+g1a2b3c
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"  # Reject unknown parameters (prevent drift)
    )

    # ═══════════════════════════════════════════════════════════════════
    # Required Paths
    # ═══════════════════════════════════════════════════════════════════

    raw_root: Path = Field(
        ...,
        description="Root directory containing raw CSV files (must exist)"
    )

    stage_root: Path = Field(
        ...,
        description="Output root for staged Parquet files (will be created)"
    )

    procedures_yaml: Path = Field(
        ...,
        description="Path to YAML schema defining procedure types and columns (must exist)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Optional Paths (auto-filled from stage_root)
    # ═══════════════════════════════════════════════════════════════════

    rejects_dir: Optional[Path] = Field(
        default=None,
        description="Directory for reject records (default: {stage_root}/../_rejects)"
    )

    events_dir: Optional[Path] = Field(
        default=None,
        description="Directory for per-run event JSONs (default: {stage_root}/_manifest/events)"
    )

    manifest_path: Optional[Path] = Field(
        default=None,
        description="Path to consolidated manifest Parquet file (default: {stage_root}/_manifest/manifest.parquet)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Performance Settings
    # ═══════════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════════
    # Behavior & Localization
    # ═══════════════════════════════════════════════════════════════════

    local_tz: str = Field(
        default="America/Santiago",
        description="IANA timezone name for date partitioning (e.g., 'America/Santiago')"
    )

    force: bool = Field(
        default=False,
        description="Overwrite existing Parquet files if True (idempotent re-staging)"
    )

    only_yaml_data: bool = Field(
        default=False,
        description="Drop columns not defined in YAML schema if True (strict mode)"
    )

    extraction_version: Optional[str] = Field(
        default=None,
        description="Parser version (e.g., 'v0.4.2+g1a2b3c' from git describe). Auto-detected if None."
    )

    # ═══════════════════════════════════════════════════════════════════
    # Validators
    # ═══════════════════════════════════════════════════════════════════

    @field_validator("raw_root", "procedures_yaml")
    @classmethod
    def _path_must_exist(cls, v: Path, info) -> Path:
        """Validate that required input paths exist."""
        if not v.exists():
            raise ValueError(f"{info.field_name} does not exist: {v}")
        return v.resolve()  # Return absolute path

    @field_validator("procedures_yaml")
    @classmethod
    def _yaml_must_be_file(cls, v: Path) -> Path:
        """Validate that YAML path is a file."""
        if not v.is_file():
            raise ValueError(f"procedures_yaml must be a file, got directory: {v}")
        return v

    @model_validator(mode="after")
    def _set_default_paths(self) -> StagingConfig:
        """
        Set default paths based on stage_root if not provided.

        Auto-fills:
        - rejects_dir: {stage_root}/../_rejects
        - events_dir: {stage_root}/_manifest/events
        - manifest_path: {stage_root}/_manifest/manifest.parquet
        """
        if self.rejects_dir is None:
            # Place rejects at same level as stage_root (data/02_stage/raw_measurements → data/02_stage/_rejects)
            self.rejects_dir = self.stage_root.parent / "_rejects"

        if self.events_dir is None:
            self.events_dir = self.stage_root / "_manifest" / "events"

        if self.manifest_path is None:
            self.manifest_path = self.stage_root / "_manifest" / "manifest.parquet"

        return self

    @model_validator(mode="after")
    def _auto_detect_version(self) -> StagingConfig:
        """
        Auto-detect extraction version from git if not provided.

        Runs: git describe --tags --always --dirty
        Example output: v0.4.2-3-g1a2b3c-dirty

        Falls back to "unknown" if:
        - Not in a git repository
        - git command fails
        - git not installed
        """
        if self.extraction_version is None:
            import subprocess

            try:
                result = subprocess.run(
                    ["git", "describe", "--tags", "--always", "--dirty"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2.0  # Don't hang
                )
                self.extraction_version = result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # Not in git repo, git not installed, or command failed
                self.extraction_version = "unknown"

        return self

    # ═══════════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════════

    def create_directories(self) -> None:
        """
        Create output directories if they don't exist.

        Creates:
        - stage_root
        - rejects_dir
        - events_dir
        - manifest_path parent directory
        """
        self.stage_root.mkdir(parents=True, exist_ok=True)

        if self.rejects_dir:
            self.rejects_dir.mkdir(parents=True, exist_ok=True)

        if self.events_dir:
            self.events_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def get_partition_path(self, proc: str, date_local: str, run_id: str) -> Path:
        """
        Get Hive-style partition path for a measurement run.

        Parameters
        ----------
        proc : str
            Procedure type (e.g., "IVg", "It")
        date_local : str
            Local date in ISO format (YYYY-MM-DD)
        run_id : str
            Run identifier (16-char hash)

        Returns
        -------
        Path
            Full path: {stage_root}/proc={proc}/date={date}/run_id={run_id}

        Example
        -------
        >>> cfg = StagingConfig(raw_root="data/01_raw", stage_root="data/02_stage/raw_measurements", procedures_yaml="config/procedures.yml")
        >>> cfg.get_partition_path("It", "2025-10-18", "a1b2c3d4e5f67890")
        PosixPath('data/02_stage/raw_measurements/proc=It/date=2025-10-18/run_id=a1b2c3d4e5f67890')
        """
        return (
            self.stage_root
            / f"proc={proc}"
            / f"date={date_local}"
            / f"run_id={run_id}"
        )

    def validate_timezone(self) -> bool:
        """
        Validate that local_tz is a valid IANA timezone.

        Returns
        -------
        bool
            True if valid, raises ValueError otherwise

        Raises
        ------
        ValueError
            If timezone is invalid
        """
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(self.local_tz)
            return True
        except ZoneInfoNotFoundError:
            raise ValueError(
                f"Invalid timezone: {self.local_tz}. "
                f"Must be a valid IANA timezone (e.g., 'America/Santiago', 'UTC')."
            )
