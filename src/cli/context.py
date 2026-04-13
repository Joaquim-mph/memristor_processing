"""
Command context object for CLI commands.

Provides unified access to shared resources (config, console, cache)
and convenience methods for common operations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Confirm

from src.cli.config import CLIConfig
from src.cli.cache import DataCache, get_cache


@dataclass
class CommandContext:
    """Shared context for all CLI commands"""

    console: Console
    config: CLIConfig
    cache: DataCache

    # Convenience properties from config
    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @property
    def dry_run(self) -> bool:
        return self.config.dry_run

    @property
    def output_dir(self) -> Path:
        return self.config.output_dir

    @property
    def history_dir(self) -> Path:
        return self.config.history_dir

    @property
    def stage_dir(self) -> Path:
        return self.config.stage_dir

    @property
    def raw_data_dir(self) -> Path:
        return self.config.raw_data_dir

    # Convenience methods
    def print(self, *args, **kwargs):
        """Print to console"""
        self.console.print(*args, **kwargs)

    def print_verbose(self, *args, **kwargs):
        """Print only if verbose mode enabled"""
        if self.verbose:
            self.console.print(*args, **kwargs)

    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"[red]Error:[/red] {message}")

    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"[green]✓[/green] {message}")

    def print_warning(self, message: str):
        """Print warning message"""
        self.console.print(f"[yellow]⚠[/yellow] {message}")

    def confirm_action(self, prompt: str) -> bool:
        """Prompt user for confirmation (returns True in dry-run)"""
        if self.dry_run:
            self.print(f"[dim][DRY RUN] Would ask: {prompt}[/dim]")
            return True

        return Confirm.ask(prompt)


# Global context
_context: Optional[CommandContext] = None


def get_context() -> CommandContext:
    """
    Get or create global command context.

    Note: Context is process-local and will be recreated in child processes
    after fork (e.g., in multiprocessing workers). This is intentional to
    avoid sharing Console objects and other fork-unsafe resources.
    """
    global _context
    if _context is None:
        from src.cli.main import get_config
        _context = CommandContext(
            console=Console(),
            config=get_config(),
            cache=get_cache()
        )
    return _context


def set_context(ctx: CommandContext):
    """Set global context (for testing)"""
    global _context
    _context = ctx


def reset_context():
    """
    Reset global context instance.

    Useful for multiprocessing scenarios where child processes need
    to reinitialize the context after fork.
    """
    global _context
    _context = None


def create_context(
    config: Optional[CLIConfig] = None,
    console: Optional[Console] = None,
    cache: Optional[DataCache] = None
) -> CommandContext:
    """Create a new context (useful for testing)"""
    from src.cli.main import get_config

    return CommandContext(
        console=console or Console(),
        config=config or get_config(),
        cache=cache or get_cache()
    )
