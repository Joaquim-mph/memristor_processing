#!/usr/bin/env python3
"""
Main CLI application entry point with plugin system.

Commands are auto-discovered from the commands/ directory using
the @cli_command decorator. No manual registration required.
"""

import typer
from pathlib import Path
from typing import Optional

from src.cli.plugin_system import discover_commands
from src.cli.config import CLIConfig, load_config_with_precedence
from src.plotting.config import PlotConfig


# Global configuration singletons
_config: Optional[CLIConfig] = None
_plot_config: Optional[PlotConfig] = None


def get_config() -> CLIConfig:
    """
    Get or create global config instance.

    Returns the cached config if available, otherwise loads with precedence:
    1. Project config (./.optothermal_cli_config.json)
    2. User config (~/.optothermal_cli_config.json)
    3. Environment variables (CLI_*)
    4. Defaults

    Returns:
        CLIConfig: Global configuration instance
    """
    global _config
    if _config is None:
        _config = load_config_with_precedence()
    return _config


def set_config(config: CLIConfig) -> None:
    """
    Set global config instance.

    Useful for testing and command-line overrides.

    Args:
        config: CLIConfig instance to use globally
    """
    global _config
    _config = config


def get_plot_config() -> PlotConfig:
    """
    Get or create global PlotConfig instance.

    Creates PlotConfig from the current CLIConfig, inheriting:
    - output_dir
    - plot_dpi
    - plot_theme
    - default_plot_format

    Returns:
        PlotConfig: Global plotting configuration instance

    Example:
        >>> config = get_plot_config()
        >>> print(config.theme, config.dpi)
        prism_rain 300
    """
    global _plot_config
    if _plot_config is None:
        cli_config = get_config()
        _plot_config = PlotConfig.from_cli_config(cli_config)
    return _plot_config


def set_plot_config(config: PlotConfig) -> None:
    """
    Set global PlotConfig instance.

    Useful for command-specific overrides (e.g., --theme, --dpi flags).

    Args:
        config: PlotConfig instance to use globally

    Example:
        >>> from src.plotting.config import PlotConfig
        >>> config = PlotConfig(theme="paper", dpi=600)
        >>> set_plot_config(config)
    """
    global _plot_config
    _plot_config = config


# Create the main Typer app
app = typer.Typer(
    name="process_and_analyze",
    help="Complete data processing and analysis pipeline for semiconductor device characterization",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def global_options(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output for all commands"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Override output directory for plots"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Use specific config file"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would happen without executing"
    ),
    # Plotting options (global overrides)
    plot_theme: Optional[str] = typer.Option(
        None,
        "--plot-theme",
        help="Override plot theme (prism_rain, paper, presentation, minimal)"
    ),
    plot_dpi: Optional[int] = typer.Option(
        None,
        "--plot-dpi",
        help="Override plot DPI (72-1200)"
    ),
    plot_format: Optional[str] = typer.Option(
        None,
        "--plot-format",
        help="Override plot format (png, pdf, svg, jpg)"
    ),
):
    """
    Global options applied to all commands.

    These options override configuration from files and environment variables.

    Plotting Options:
        --plot-theme: Change visual style (prism_rain=lab, paper=publication,
                     presentation=slides, minimal=web)
        --plot-dpi: Resolution for output (300=standard, 600=publication, 150=web)
        --plot-format: Output format (png=raster, pdf=vector, svg=editable vector)
    """
    # Load base config with specified file (if any)
    config = load_config_with_precedence(config_file=config_file)

    # Apply overrides from command-line options
    overrides = {}
    if verbose:
        overrides["verbose"] = verbose
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if dry_run:
        overrides["dry_run"] = dry_run
    if plot_theme is not None:
        overrides["plot_theme"] = plot_theme
    if plot_dpi is not None:
        overrides["plot_dpi"] = plot_dpi
    if plot_format is not None:
        overrides["default_plot_format"] = plot_format

    # Create merged config and set as global
    if overrides:
        config = config.merge_with(**overrides)

    set_config(config)

    # Reset plot config to pick up new CLI config changes
    # (forces get_plot_config() to recreate from updated CLI config)
    global _plot_config
    _plot_config = None


# Auto-discover and register all command plugins
discover_commands(
    app,
    commands_dir=Path("src/cli/commands"),
    config_path=Path("config/cli_plugins.yaml"),
    verbose=False  # Set to True for debugging
)


def main():
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
