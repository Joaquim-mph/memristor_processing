#!/usr/bin/env python3
"""
Configuration Management Layer for CLI Module

Provides centralized configuration with support for:
- Environment variables (CLI_* prefix)
- Config files (~/.optothermal_cli_config.json or project-specific)
- Command-line overrides
- Validated defaults

Configuration priority (highest to lowest):
1. Command-line overrides
2. Config file
3. Environment variables
4. Hardcoded defaults
"""

import json
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class CLIConfig(BaseModel):
    """
    Central configuration for the optothermal processing pipeline CLI.

    All paths are resolved to absolute paths and directories are auto-created
    during validation.
    """

    # Directory paths
    raw_data_dir: Path = Field(
        default=Path("data/01_raw"),
        description="Directory containing raw CSV measurement files"
    )
    stage_dir: Path = Field(
        default=Path("data/02_stage"),
        description="Directory for staged Parquet files and manifest"
    )
    history_dir: Path = Field(
        default=Path("data/02_stage/chip_histories"),
        description="Directory for per-chip history Parquet files"
    )
    output_dir: Path = Field(
        default=Path("figs"),
        description="Directory for generated plots and outputs"
    )

    # Behavior settings
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging output"
    )
    dry_run: bool = Field(
        default=False,
        description="Show what would happen without executing"
    )

    # Processing settings
    parallel_workers: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Number of parallel workers for data processing"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching for faster repeated operations"
    )
    cache_ttl: int = Field(
        default=300,
        ge=0,
        description="Cache time-to-live in seconds"
    )
    cache_max_items: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of items in cache"
    )
    cache_max_size_mb: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Maximum cache size in megabytes"
    )

    # Plot defaults
    default_plot_format: Literal["png", "pdf", "svg", "jpg"] = Field(
        default="png",
        description="Default output format for plots"
    )
    plot_dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for plot output"
    )
    plot_theme: Literal["prism_rain", "paper", "presentation", "minimal"] = Field(
        default="prism_rain",
        description=(
            "Default matplotlib theme for plots:\n"
            "  - prism_rain: Lab notebooks (large fonts, colorful, current default)\n"
            "  - paper: Publications (small serif fonts, high DPI, Nature/IEEE)\n"
            "  - presentation: Slides/posters (extra large fonts, vivid colors)\n"
            "  - minimal: Web dashboards (clean sans-serif, understated colors)"
        )
    )

    # Config metadata (not user-configurable)
    config_version: str = Field(
        default="1.0.0",
        description="Configuration schema version"
    )

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }

    @field_validator("raw_data_dir", "stage_dir", "history_dir", "output_dir", mode="before")
    @classmethod
    def resolve_path(cls, v) -> Path:
        """
        Resolve paths to absolute paths.
        Relative paths are resolved relative to current working directory.
        """
        if v is None:
            return v
        path = Path(v)
        # Resolve to absolute path
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @model_validator(mode="after")
    def create_directories(self):
        """
        Create directories if they don't exist.
        Only creates parent directories for safety.
        """
        for field_name in ["raw_data_dir", "stage_dir", "history_dir", "output_dir"]:
            path = getattr(self, field_name)
            if path:
                # Create directory and parents
                path.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def from_env(cls, prefix: str = "CLI_") -> "CLIConfig":
        """
        Load configuration from environment variables.

        Environment variable format: {prefix}{FIELD_NAME}
        Example: CLI_VERBOSE=true, CLI_OUTPUT_DIR=/tmp/plots

        Args:
            prefix: Prefix for environment variables (default: "CLI_")

        Returns:
            CLIConfig instance with values from environment
        """
        config_dict = {}

        for field_name in cls.model_fields.keys():
            env_var = f"{prefix}{field_name.upper()}"
            env_value = os.getenv(env_var)

            if env_value is not None:
                # Type coercion based on field type
                field_info = cls.model_fields[field_name]
                field_type = field_info.annotation

                # Handle different types
                if field_type == bool or (hasattr(field_type, "__origin__") and field_type.__origin__ == bool):
                    config_dict[field_name] = env_value.lower() in ("true", "1", "yes", "on")
                elif field_type == int or (hasattr(field_type, "__origin__") and field_type.__origin__ == int):
                    config_dict[field_name] = int(env_value)
                elif field_type == Path or (hasattr(field_type, "__origin__") and field_type.__origin__ == Path):
                    config_dict[field_name] = Path(env_value)
                else:
                    config_dict[field_name] = env_value

        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_file: Path) -> "CLIConfig":
        """
        Load configuration from JSON config file.

        Args:
            config_file: Path to JSON configuration file

        Returns:
            CLIConfig instance with values from file

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        config_file = Path(config_file)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        # Remove any comments or metadata fields that aren't part of the model
        config_dict = {k: v for k, v in config_dict.items() if k in cls.model_fields}

        return cls(**config_dict)

    def save(self, config_file: Path, pretty: bool = True) -> None:
        """
        Save current configuration to JSON file.

        Args:
            config_file: Path where to save configuration
            pretty: If True, format JSON with indentation

        Raises:
            PermissionError: If can't write to config file location
        """
        config_file = Path(config_file)

        # Ensure parent directory exists
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path serialization
        config_dict = self.model_dump(mode="json")

        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        # Write to file
        with open(config_file, "w") as f:
            if pretty:
                json.dump(config_dict, f, indent=2, sort_keys=False)
                f.write("\n")  # Add trailing newline
            else:
                json.dump(config_dict, f)

    def merge_with(self, **overrides) -> "CLIConfig":
        """
        Create a new config with specified overrides.

        Args:
            **overrides: Field values to override

        Returns:
            New CLIConfig instance with overrides applied
        """
        config_dict = self.model_dump()
        config_dict.update(overrides)
        return CLIConfig(**config_dict)

    def get_field_source(self, field_name: str, check_env: bool = True, check_default: bool = True) -> str:
        """
        Determine the source of a configuration field value.

        Args:
            field_name: Name of the field to check
            check_env: Whether to check environment variables
            check_default: Whether to check if value matches default

        Returns:
            String indicating source: "env", "file", "override", or "default"
        """
        if not check_env and not check_default:
            return "unknown"

        current_value = getattr(self, field_name)

        # Check if value matches default
        if check_default:
            default_value = self.model_fields[field_name].default
            if current_value == default_value:
                return "default"

        # Check if environment variable is set
        if check_env:
            env_var = f"CLI_{field_name.upper()}"
            if os.getenv(env_var) is not None:
                return "env"

        return "override"


class ConfigProfile:
    """
    Predefined configuration profiles for common use cases.
    """

    @staticmethod
    def development() -> CLIConfig:
        """
        Development profile with verbose output and dry-run mode.
        Useful for testing and debugging.
        """
        return CLIConfig(
            verbose=True,
            dry_run=True,
            parallel_workers=2,
            cache_enabled=False,
        )

    @staticmethod
    def production() -> CLIConfig:
        """
        Production profile optimized for batch processing.
        Maximum parallelism with caching enabled.
        """
        return CLIConfig(
            verbose=False,
            dry_run=False,
            parallel_workers=8,
            cache_enabled=True,
            cache_ttl=600,
        )

    @staticmethod
    def testing() -> CLIConfig:
        """
        Testing profile with temporary directories.
        Useful for automated tests.
        """
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="optothermal_test_"))
        return CLIConfig(
            raw_data_dir=tmp / "raw",
            stage_dir=tmp / "stage",
            history_dir=tmp / "stage" / "chip_histories",
            output_dir=tmp / "output",
            verbose=True,
            dry_run=False,
            parallel_workers=1,
            cache_enabled=False,
        )

    @staticmethod
    def high_quality() -> CLIConfig:
        """
        Profile for high-quality publication figures.
        High DPI, PDF output.
        """
        return CLIConfig(
            default_plot_format="pdf",
            plot_dpi=600,
        )


def load_config_with_precedence(
    config_file: Optional[Path] = None,
    check_env: bool = True,
    check_user_config: bool = True,
    check_project_config: bool = True,
    **overrides
) -> CLIConfig:
    """
    Load configuration with proper precedence handling.

    Precedence (highest to lowest):
    1. Explicit overrides (**overrides)
    2. Specified config file (config_file parameter)
    3. Project-local config (./.optothermal_cli_config.json)
    4. User config (~/.optothermal_cli_config.json)
    5. Environment variables (CLI_*)
    6. Defaults

    Args:
        config_file: Explicit config file path (highest priority after overrides)
        check_env: Whether to load from environment variables
        check_user_config: Whether to check user home directory for config
        check_project_config: Whether to check current directory for config
        **overrides: Direct field overrides (highest priority)

    Returns:
        CLIConfig instance with merged configuration
    """
    # Start with defaults
    config = CLIConfig()

    # Layer 1: Environment variables (if enabled)
    if check_env:
        try:
            env_config = CLIConfig.from_env()
            # Only update fields that differ from defaults
            for field_name in CLIConfig.model_fields.keys():
                env_value = getattr(env_config, field_name)
                default_value = CLIConfig.model_fields[field_name].default
                if env_value != default_value:
                    setattr(config, field_name, env_value)
        except Exception:
            # Silently ignore env loading errors
            pass

    # Layer 2: User config file (if enabled)
    if check_user_config:
        user_config_path = Path.home() / ".optothermal_cli_config.json"
        if user_config_path.exists():
            try:
                file_config = CLIConfig.from_file(user_config_path)
                # Update all fields from file
                for field_name in CLIConfig.model_fields.keys():
                    setattr(config, field_name, getattr(file_config, field_name))
            except Exception:
                # Silently ignore file loading errors
                pass

    # Layer 3: Project config file (if enabled)
    if check_project_config:
        project_config_path = Path.cwd() / ".optothermal_cli_config.json"
        if project_config_path.exists() and project_config_path != (Path.home() / ".optothermal_cli_config.json"):
            try:
                file_config = CLIConfig.from_file(project_config_path)
                # Update all fields from file
                for field_name in CLIConfig.model_fields.keys():
                    setattr(config, field_name, getattr(file_config, field_name))
            except Exception:
                # Silently ignore file loading errors
                pass

    # Layer 4: Explicit config file (if provided)
    if config_file is not None:
        config_file = Path(config_file)
        if config_file.exists():
            config = CLIConfig.from_file(config_file)

    # Layer 5: Direct overrides (highest priority)
    if overrides:
        config = config.merge_with(**overrides)

    return config
