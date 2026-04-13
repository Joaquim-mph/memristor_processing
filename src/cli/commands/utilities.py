"""Utility commands: list-plugins."""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.cli.plugin_system import cli_command, list_available_commands, get_command_groups

console = Console()


@cli_command(
    name="list-plugins",
    group="utilities",
    description="List all available command plugins",
    aliases=["plugins", "list-commands"]
)
def list_plugins_command(
    group: Optional[str] = typer.Option(
        None,
        "--group",
        "-g",
        help="Filter by command group (pipeline, history, staging, plotting, utilities)"
    ),
    show_aliases: bool = typer.Option(
        False,
        "--show-aliases",
        "-a",
        help="Show command aliases"
    ),
):
    """
    List all available command plugins with their metadata.

    Displays commands organized by group with descriptions. Useful for
    discovering available functionality and exploring command groups.

    Examples:
        # List all commands
        python process_and_analyze.py list-plugins

        # List only plotting commands
        python process_and_analyze.py list-plugins --group plotting

        # Show aliases
        python process_and_analyze.py list-plugins --show-aliases

        # Use alias
        python process_and_analyze.py plugins
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Available Command Plugins[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Get commands
    commands = list_available_commands(group)

    if not commands:
        if group:
            console.print(f"[yellow]No commands found in group '{group}'[/yellow]")
            console.print(f"\n[cyan]Available groups:[/cyan] {', '.join(get_command_groups())}")
        else:
            console.print("[yellow]No commands available[/yellow]")
        console.print()
        return

    # Create table
    table = Table(
        title=f"Command Plugins{f' (group: {group})' if group else ''}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    table.add_column("Command", style="yellow", width=25)
    table.add_column("Group", style="magenta", width=12)
    table.add_column("Description", style="white", width=50)

    if show_aliases:
        table.add_column("Aliases", style="dim", width=20)

    # Add rows
    for cmd in commands:
        desc = cmd.description
        if len(desc) > 47:
            desc = desc[:47] + "..."

        if show_aliases:
            aliases_str = ", ".join(cmd.aliases) if cmd.aliases else "—"
            table.add_row(cmd.name, cmd.group, desc, aliases_str)
        else:
            table.add_row(cmd.name, cmd.group, desc)

    console.print(table)
    console.print()

    # Summary
    groups = get_command_groups()
    console.print(f"[cyan]Total commands:[/cyan] {len(commands)}")

    if not group:
        console.print(f"[cyan]Command groups:[/cyan] {', '.join(groups)}")
        console.print()
        console.print("[dim]Tip: Use --group <name> to filter by group[/dim]")

    console.print()
