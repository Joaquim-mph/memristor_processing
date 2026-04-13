"""
CLI Plugin System - Auto-discovery and registration of commands.

Provides decorator-based command registration and automatic discovery
from the commands directory.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass, field

import typer
import yaml


@dataclass
class CommandMetadata:
    """Metadata for a CLI command plugin."""

    name: str
    """Command name (kebab-case, e.g., 'full-pipeline')"""

    function: Callable
    """The actual command function"""

    group: str = "general"
    """Command group for organization (e.g., 'pipeline', 'plotting')"""

    description: str = ""
    """Short description for the command"""

    aliases: List[str] = field(default_factory=list)
    """Alternative names for the command"""

    enabled: bool = True
    """Whether the command is enabled"""

    priority: int = 0
    """Registration priority (higher = earlier)"""

    hidden: bool = False
    """Whether to hide the command from help output"""


# Global registry of discovered commands
_COMMAND_REGISTRY: Dict[str, CommandMetadata] = {}


def cli_command(
    name: str,
    group: str = "general",
    description: str = "",
    aliases: Optional[List[str]] = None,
    priority: int = 0,
    hidden: bool = False,
):
    """
    Decorator to register a function as a CLI command plugin.

    Parameters
    ----------
    name : str
        Command name (kebab-case, e.g., 'show-history')
    group : str
        Command group for organization (e.g., 'history', 'plotting')
    description : str
        Short description (overrides docstring first line)
    aliases : list[str], optional
        Alternative command names
    priority : int
        Registration priority (higher = registered earlier)
    hidden : bool
        Whether to hide the command from help output (default: False)

    Examples
    --------
    >>> @cli_command(name="show-history", group="history")
    ... def show_history_command(chip_number: int, ...):
    ...     '''Display chip experiment history'''
    ...     pass

    Notes
    -----
    The decorated function is registered in the global command registry
    and will be discovered by discover_commands().
    """
    def decorator(func: Callable) -> Callable:
        # Extract description from docstring if not provided
        if not description and func.__doc__:
            desc = func.__doc__.strip().split('\n')[0]
        else:
            desc = description

        # Create metadata
        metadata = CommandMetadata(
            name=name,
            function=func,
            group=group,
            description=desc,
            aliases=aliases or [],
            priority=priority,
            hidden=hidden,
        )

        # Register command
        _COMMAND_REGISTRY[name] = metadata

        # Return original function (decorator doesn't wrap)
        return func

    return decorator


def load_plugin_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load plugin configuration from YAML file.

    Parameters
    ----------
    config_path : Path, optional
        Path to plugin config YAML. Defaults to config/cli_plugins.yaml

    Returns
    -------
    dict
        Plugin configuration with keys:
        - enabled_groups: List of enabled command groups
        - disabled_commands: List of disabled command names
        - settings: Additional plugin settings
    """
    if config_path is None:
        config_path = Path("config/cli_plugins.yaml")

    # Default configuration
    default_config = {
        "enabled_groups": ["all"],  # "all" means enable everything
        "disabled_commands": [],
        "settings": {},
    }

    if not config_path.exists():
        return default_config

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Merge with defaults
        return {**default_config, **(config or {})}

    except Exception as e:
        # If config fails to load, use defaults
        print(f"Warning: Failed to load plugin config: {e}", file=sys.stderr)
        return default_config


def discover_commands(
    app: typer.Typer,
    commands_dir: str | Path = "src/cli/commands",
    config_path: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """
    Auto-discover and register all command plugins.

    Scans the commands directory for Python modules, imports them to
    trigger @cli_command decorators, then registers discovered commands
    with the Typer app.

    Parameters
    ----------
    app : typer.Typer
        The Typer application to register commands with
    commands_dir : str | Path
        Directory containing command modules
    config_path : Path, optional
        Path to plugin configuration YAML
    verbose : bool
        Print discovered commands (for debugging)

    Examples
    --------
    >>> app = typer.Typer()
    >>> discover_commands(app, "src/cli/commands")
    >>> # All decorated commands are now registered

    Notes
    -----
    Commands are registered in priority order (highest first), then
    alphabetically by name.
    """
    commands_path = Path(commands_dir)

    if not commands_path.exists():
        raise FileNotFoundError(f"Commands directory not found: {commands_path}")

    # Load plugin configuration
    config = load_plugin_config(config_path)
    enabled_groups = config["enabled_groups"]
    disabled_commands = config["disabled_commands"]

    # Import all command modules (triggers @cli_command decorators)
    for module_path in commands_path.glob("*.py"):
        if module_path.stem == "__init__":
            continue

        # Convert path to module name: src/cli/commands/history.py -> src.cli.commands.history
        module_name = str(module_path.with_suffix("")).replace("/", ".")

        try:
            importlib.import_module(module_name)
            if verbose:
                print(f"Loaded plugin module: {module_name}")
        except Exception as e:
            print(f"Warning: Failed to load plugin {module_name}: {e}", file=sys.stderr)

    # Register discovered commands
    # Sort by priority (descending) then name (ascending)
    sorted_commands = sorted(
        _COMMAND_REGISTRY.values(),
        key=lambda c: (-c.priority, c.name)
    )

    registered_count = 0
    for metadata in sorted_commands:
        # Check if command group is enabled
        if "all" not in enabled_groups and metadata.group not in enabled_groups:
            if verbose:
                print(f"Skipped (group disabled): {metadata.name} [{metadata.group}]")
            continue

        # Check if specific command is disabled
        if metadata.name in disabled_commands:
            if verbose:
                print(f"Skipped (explicitly disabled): {metadata.name}")
            continue

        # Map group to Rich help panel name
        group_panels = {
            "pipeline": "Pipeline Commands",
            "history": "History Commands",
            "plotting": "Plotting Commands",
            "staging": "Staging Commands",
            "utilities": "Utilities",
            "general": "General Commands",
        }
        rich_help_panel = group_panels.get(metadata.group, f"{metadata.group.title()} Commands")

        # Register command with Typer (with Rich help panel for grouping)
        app.command(name=metadata.name, rich_help_panel=rich_help_panel, hidden=metadata.hidden)(metadata.function)
        registered_count += 1

        if verbose:
            hidden_label = " [hidden]" if metadata.hidden else ""
            print(f"Registered: {metadata.name} [{metadata.group}] -> panel: {rich_help_panel}{hidden_label}")

        # Register aliases (hidden from help output)
        for alias in metadata.aliases:
            app.command(name=alias, hidden=True, rich_help_panel=rich_help_panel)(metadata.function)
            if verbose:
                print(f"  Alias: {alias} -> {metadata.name} [hidden]")

    if verbose:
        print(f"\nTotal commands registered: {registered_count}/{len(_COMMAND_REGISTRY)}")


def list_available_commands(group: Optional[str] = None) -> List[CommandMetadata]:
    """
    Get list of all available command plugins.

    Parameters
    ----------
    group : str, optional
        Filter by command group

    Returns
    -------
    list[CommandMetadata]
        List of command metadata objects
    """
    commands = list(_COMMAND_REGISTRY.values())

    if group:
        commands = [c for c in commands if c.group == group]

    return sorted(commands, key=lambda c: c.name)


def get_command_groups() -> List[str]:
    """Get list of all command groups."""
    return sorted(set(c.group for c in _COMMAND_REGISTRY.values()))


def clear_registry() -> None:
    """Clear the command registry (useful for testing)."""
    _COMMAND_REGISTRY.clear()
