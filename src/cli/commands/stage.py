"""Staging pipeline commands: stage-all, stage-incremental, validate-manifest, inspect-manifest."""

import typer
from src.cli.plugin_system import cli_command
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box
import time

console = Console()


@cli_command(
    name="stage-all",
    group="staging",
    description="Stage all raw CSV files to Parquet format"
)
def stage_all_command(
    raw_root: Optional[Path] = typer.Option(
        None,
        "--raw-root",
        "-r",
        help="Root directory containing raw CSV files (default: from config)"
    ),
    stage_root: Optional[Path] = typer.Option(
        None,
        "--stage-root",
        "-s",
        help="Output directory for staged Parquet files (default: from config)"
    ),
    procedures_yaml: Path = typer.Option(
        Path("config/procedures.yml"),
        "--procedures-yaml",
        "-p",
        help="YAML schema file defining procedures and column types"
    ),
    rejects_dir: Optional[Path] = typer.Option(
        None,
        "--rejects-dir",
        help="Directory for reject records (auto: {stage_root}/../_rejects)"
    ),
    events_dir: Optional[Path] = typer.Option(
        None,
        "--events-dir",
        help="Directory for event JSONs (auto: {stage_root}/_manifest/events)"
    ),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help="Manifest Parquet file path (auto: {stage_root}/_manifest/manifest.parquet)"
    ),
    local_tz: str = typer.Option(
        "America/Santiago",
        "--local-tz",
        "-tz",
        help="Timezone for date partitioning"
    ),
    workers: int = typer.Option(
        8,
        "--workers",
        "-w",
        help="Number of parallel worker processes"
    ),
    polars_threads: int = typer.Option(
        2,
        "--polars-threads",
        help="Polars threads per worker"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing Parquet files"
    ),
    only_yaml_data: bool = typer.Option(
        False,
        "--only-yaml-data",
        help="Drop columns not defined in YAML schema"
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Strict validation mode - fail on schema errors (missing required columns, etc.)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed file-by-file progress"
    ),
):
    """
    Stage all raw CSV files to Parquet format with manifest tracking.

    This command discovers all CSV files in the raw data directory,
    parses headers and data, validates against the procedures schema,
    and writes partitioned Parquet files with a centralized manifest.

    By default, shows a clean spinner during processing. Use --verbose
    to see detailed file-by-file progress.

    Examples:

        # Basic staging with defaults (clean output)
        process_and_analyze stage-all

        # Show detailed file-by-file progress
        process_and_analyze stage-all --verbose

        # Custom paths and parallel processing
        process_and_analyze stage-all -r data/01_raw -s data/02_stage/raw_measurements -w 8 -f

        # Strict schema mode
        process_and_analyze stage-all --only-yaml-data
    """
    # Load config for defaults
    from src.cli.main import get_config
    config = get_config()

    if raw_root is None:
        raw_root = config.raw_data_dir
        if config.verbose or verbose:
            console.print(f"[dim]Using raw data directory from config: {raw_root}[/dim]")

    if stage_root is None:
        stage_root = config.stage_dir / "raw_measurements"
        if config.verbose or verbose:
            console.print(f"[dim]Using stage directory from config: {stage_root}[/dim]")

    from src.models.parameters import StagingParameters
    from src.core import run_staging_pipeline, discover_csvs

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Staging Pipeline[/bold cyan]\n"
        "CSV → Schema Validation → Parquet + Manifest",
        border_style="cyan"
    ))
    console.print()

    # Validate input paths
    if not raw_root.exists():
        console.print(f"[bold red]Error:[/bold red] Raw root does not exist: {raw_root}")
        raise typer.Exit(1)

    if not procedures_yaml.exists():
        console.print(f"[bold red]Error:[/bold red] Procedures YAML not found: {procedures_yaml}")
        raise typer.Exit(1)

    # Display configuration
    config_table = Table(title="Configuration", show_header=False, box=box.SIMPLE)
    config_table.add_column("Parameter", style="cyan", width=20)
    config_table.add_column("Value", style="white")

    config_table.add_row("Raw Root", str(raw_root))
    config_table.add_row("Stage Root", str(stage_root))
    config_table.add_row("Procedures YAML", str(procedures_yaml))
    config_table.add_row("Workers", str(workers))
    config_table.add_row("Polars Threads", str(polars_threads))
    config_table.add_row("Timezone", local_tz)
    config_table.add_row("Force Overwrite", "✓ Yes" if force else "✗ No")
    config_table.add_row("Only YAML Columns", "✓ Yes" if only_yaml_data else "✗ No")
    config_table.add_row("Strict Validation", "✓ Yes" if strict else "✗ No")

    console.print(config_table)
    console.print()

    try:
        # Create staging parameters
        console.print("[cyan]Creating staging configuration...[/cyan]")
        params = StagingParameters(
            raw_root=raw_root,
            stage_root=stage_root,
            procedures_yaml=procedures_yaml,
            rejects_dir=rejects_dir,
            events_dir=events_dir,
            manifest=manifest,
            local_tz=local_tz,
            workers=workers,
            polars_threads=polars_threads,
            force=force,
            only_yaml_data=only_yaml_data,
            strict=strict,
        )

        # Discover files
        csvs = discover_csvs(raw_root)
        console.print(f"[green]✓[/green] Discovered {len(csvs)} CSV files")
        console.print()

        if not csvs:
            console.print("[yellow]No CSV files found. Nothing to do.[/yellow]")
            console.print()
            return

        # Show preview of files
        if len(csvs) <= 10:
            tree = Tree(f"[bold]{raw_root}[/bold]")
            for csv in csvs:
                tree.add(f"[dim]{csv.relative_to(raw_root)}[/dim]")
            console.print(tree)
            console.print()
        else:
            console.print(f"[dim]First 5 files:[/dim]")
            for csv in csvs[:5]:
                console.print(f"  [dim]• {csv.relative_to(raw_root)}[/dim]")
            console.print(f"  [dim]... and {len(csvs) - 5} more[/dim]")
            console.print()

        # Run staging pipeline
        console.print("[bold cyan]Running staging pipeline...[/bold cyan]")
        console.print()

        start_time = time.time()

        # Suppress verbose output unless --verbose flag is set
        if not verbose:
            import sys
            import io
            from contextlib import redirect_stdout

            # Show spinner with live counter while processing
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("•"),
                TextColumn("[cyan]{task.fields[status]}"),
                console=console,
                transient=False,
                refresh_per_second=10  # Update 10 times per second
            ) as progress:
                task = progress.add_task(
                    "Staging files",
                    total=None,  # Indeterminate progress (spinner mode)
                    status=f"0/{len(csvs)} files processed"
                )

                # Progress callback function
                def update_progress(current, total, proc, status_str):
                    progress.update(
                        task,
                        status=f"{current}/{total} files • {proc} ({status_str})"
                    )

                # Run staging with progress callback (print statements disabled in core)
                run_staging_pipeline(params, progress_callback=update_progress)
                progress.update(task, status=f"✓ {len(csvs)}/{len(csvs)} files processed")
        else:
            console.print("[dim]Detailed file-by-file progress:[/dim]")
            console.print()
            run_staging_pipeline(params)

        elapsed = time.time() - start_time

        # Read manifest to show summary
        summary_text = f"[bold green]✓ Staging Complete[/bold green]\n\nTime: {elapsed:.1f}s\n"

        try:
            import polars as pl
            if params.manifest.exists():
                manifest_df = pl.read_parquet(params.manifest)

                # Count by procedure
                if "proc" in manifest_df.columns:
                    proc_counts = manifest_df.group_by("proc").agg(
                        pl.count().alias("count")
                    ).sort("count", descending=True)

                    summary_text += f"\n[cyan]Procedures Found:[/cyan]\n"
                    for row in proc_counts.iter_rows(named=True):
                        summary_text += f"  • {row['proc']}: {row['count']:,} files\n"

                    summary_text += f"\n[dim]Total: {len(manifest_df):,} measurements[/dim]"
        except Exception:
            # If we can't read manifest, just show basic info
            pass

        summary_text += f"\n\nOutput: {stage_root}\nManifest: {params.manifest}"

        console.print()
        console.print(Panel.fit(
            summary_text,
            border_style="green"
        ))
        console.print()

        # Helpful next steps
        console.print("[dim]Next steps:[/dim]")
        console.print("  [cyan]• build-all-histories[/cyan] - Generate chip history files")
        console.print("  [cyan]• validate-manifest[/cyan] - Validate staged data")
        console.print("  [cyan]• inspect-manifest[/cyan] - Browse manifest contents")
        console.print()

    except Exception as e:
        console.print()
        console.print(Panel.fit(
            f"[bold red]✗ Staging Failed[/bold red]\n"
            f"Error: {str(e)}",
            border_style="red"
        ))
        console.print()
        raise typer.Exit(1)


@cli_command(
    name="validate-manifest",
    group="staging",
    description="Validate manifest schema and data quality"
)
def validate_manifest_command(
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help="Path to manifest Parquet file"
    ),
    show_details: bool = typer.Option(
        False,
        "--details",
        "-d",
        help="Show detailed field statistics"
    ),
):
    """
    Validate manifest schema and check for data quality issues.

    Performs comprehensive checks:
    - Schema validation against Pydantic model
    - Duplicate run_id detection
    - Missing required fields
    - Data completeness statistics
    - Summary by procedure type

    Examples:

        # Validate default manifest
        process_and_analyze validate-manifest

        # Validate with detailed statistics
        process_and_analyze validate-manifest --details

        # Validate custom manifest
        process_and_analyze validate-manifest -m path/to/manifest.parquet
    """
    # Load config for defaults
    from src.cli.main import get_config
    config = get_config()

    if manifest is None:
        manifest = config.stage_dir / "raw_measurements" / "_manifest" / "manifest.parquet"
        if config.verbose:
            console.print(f"[dim]Using manifest path from config: {manifest}[/dim]")

    import polars as pl
    from pydantic import TypeAdapter
    from src.models.manifest import ManifestRow

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Manifest Validation[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    try:
        if not manifest.exists():
            console.print(f"[bold red]Error:[/bold red] Manifest not found: {manifest}")
            console.print()
            raise typer.Exit(1)

        # Load manifest
        console.print(f"[cyan]Loading manifest:[/cyan] {manifest}")
        df = pl.read_parquet(manifest)
        console.print(f"[green]✓[/green] Loaded {len(df):,} rows")
        console.print()

        # Track validation status
        issues_found = False

        # Check for duplicates
        console.print("[cyan]Checking for duplicate run_ids...[/cyan]")
        duplicates = df.group_by("run_id").agg(pl.count().alias("count")).filter(pl.col("count") > 1)
        if len(duplicates) > 0:
            console.print(f"[bold red]✗[/bold red] Found {len(duplicates)} duplicate run_ids")
            issues_found = True
            if show_details:
                console.print(duplicates.head(10))
        else:
            console.print("[bold green]✓[/bold green] No duplicate run_ids")
        console.print()

        # Schema validation
        console.print("[cyan]Validating schema against Pydantic model...[/cyan]")
        try:
            ta = TypeAdapter(list[ManifestRow])
            rows = df.to_dicts()
            ta.validate_python(rows)
            console.print("[bold green]✓[/bold green] Schema validation passed")
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Schema validation failed:")
            console.print(f"  [red]{str(e)[:200]}...[/red]" if len(str(e)) > 200 else f"  [red]{str(e)}[/red]")
            issues_found = True
        console.print()

        # Completeness checks
        console.print("[cyan]Checking field completeness...[/cyan]")
        completeness_table = Table(
            title="Required Fields",
            show_header=True,
            box=box.ROUNDED,
            header_style="bold cyan"
        )
        completeness_table.add_column("Field", style="cyan")
        completeness_table.add_column("Complete", justify="right")
        completeness_table.add_column("Missing", justify="right")
        completeness_table.add_column("% Complete", justify="right")

        required_fields = ["run_id", "source_file", "proc", "chip_number", "start_time_utc"]
        for field in required_fields:
            if field not in df.columns:
                completeness_table.add_row(field, "0", str(len(df)), "0.0%")
                issues_found = True
                continue

            null_count = df[field].null_count()
            complete_count = len(df) - null_count
            pct = (complete_count / len(df)) * 100 if len(df) > 0 else 0

            status_style = "green" if null_count == 0 else "yellow"
            completeness_table.add_row(
                field,
                f"[{status_style}]{complete_count:,}[/{status_style}]",
                f"[{status_style}]{null_count:,}[/{status_style}]",
                f"[{status_style}]{pct:.1f}%[/{status_style}]"
            )

            if null_count > 0:
                issues_found = True

        console.print(completeness_table)
        console.print()

        # Summary statistics
        console.print("[cyan]Summary Statistics:[/cyan]")
        summary_table = Table(show_header=False, box=box.SIMPLE)
        summary_table.add_column("Metric", style="cyan", width=30)
        summary_table.add_column("Value", style="white")

        summary_table.add_row("Total Measurements", f"{len(df):,}")

        if "proc" in df.columns:
            proc_counts = df.group_by("proc").agg(pl.count().alias("count")).sort("count", descending=True)
            for row in proc_counts.iter_rows(named=True):
                summary_table.add_row(f"  └─ {row['proc']}", f"{row['count']:,}")

        if "chip_number" in df.columns:
            unique_chips = df["chip_number"].n_unique()
            summary_table.add_row("Unique Chips", f"{unique_chips:,}")

        if "date_local" in df.columns:
            date_range = df["date_local"].min(), df["date_local"].max()
            summary_table.add_row("Date Range", f"{date_range[0]} to {date_range[1]}")

        console.print(summary_table)
        console.print()

        # Detailed statistics
        if show_details and "proc" in df.columns:
            console.print("[cyan]Detailed Statistics by Procedure:[/cyan]")
            detail_table = Table(show_header=True, box=box.ROUNDED, header_style="bold cyan")
            detail_table.add_column("Procedure")
            detail_table.add_column("Count", justify="right")
            detail_table.add_column("Chips", justify="right")
            detail_table.add_column("Dates", justify="right")

            for row in proc_counts.iter_rows(named=True):
                proc_df = df.filter(pl.col("proc") == row['proc'])
                chips = proc_df["chip_number"].n_unique() if "chip_number" in proc_df.columns else 0
                dates = proc_df["date_local"].n_unique() if "date_local" in proc_df.columns else 0

                detail_table.add_row(
                    row['proc'],
                    f"{row['count']:,}",
                    f"{chips:,}" if chips > 0 else "—",
                    f"{dates:,}" if dates > 0 else "—"
                )

            console.print(detail_table)
            console.print()

        # Final status
        if issues_found:
            console.print(Panel.fit(
                "[bold yellow]⚠ Validation Complete with Issues[/bold yellow]\n"
                "See details above for warnings",
                border_style="yellow"
            ))
        else:
            console.print(Panel.fit(
                "[bold green]✓ Validation Complete[/bold green]\n"
                "No issues found",
                border_style="green"
            ))
        console.print()

    except Exception as e:
        console.print()
        console.print(Panel.fit(
            f"[bold red]✗ Validation Failed[/bold red]\n"
            f"Error: {str(e)}",
            border_style="red"
        ))
        console.print()
        raise typer.Exit(1)


@cli_command(
    name="inspect-manifest",
    group="staging",
    description="Inspect manifest contents with filtering"
)
def inspect_manifest_command(
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help="Path to manifest Parquet file"
    ),
    proc: Optional[str] = typer.Option(
        None,
        "--proc",
        "-p",
        help="Filter by procedure type (IVg, It, IV, etc.)"
    ),
    chip: Optional[int] = typer.Option(
        None,
        "--chip",
        "-c",
        help="Filter by chip number"
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of rows to display"
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table (default), json, csv"
    ),
):
    """
    Inspect manifest contents with optional filtering.

    Browse the manifest data with filtering by procedure type, chip number,
    or other criteria. Useful for quick exploration of staged data.

    Output formats:
        - table: Rich terminal table with colors and styling (default)
        - json: Machine-readable JSON for scripting/automation
        - csv: Spreadsheet-compatible CSV for data analysis

    Examples:

        # Show first 20 rows
        process_and_analyze inspect-manifest

        # Show IVg measurements only
        process_and_analyze inspect-manifest --proc IVg -n 50

        # Show measurements for chip 67
        process_and_analyze inspect-manifest --chip 67

        # Combine filters
        process_and_analyze inspect-manifest -p It -c 67 -n 10

        # Export as JSON
        process_and_analyze inspect-manifest --format json > manifest.json

        # Export as CSV
        process_and_analyze inspect-manifest --chip 67 --format csv > chip67_manifest.csv
    """
    # Load config for defaults
    from src.cli.main import get_config
    config = get_config()

    if manifest is None:
        manifest = config.stage_dir / "raw_measurements" / "_manifest" / "manifest.parquet"
        if config.verbose:
            console.print(f"[dim]Using manifest path from config: {manifest}[/dim]")

    import polars as pl

    try:
        if not manifest.exists():
            console.print(f"[bold red]Error:[/bold red] Manifest not found: {manifest}")
            console.print()
            raise typer.Exit(1)

        # Load manifest
        df = pl.read_parquet(manifest)
        total_rows = len(df)

        # Apply filters
        filters = []
        if proc:
            df = df.filter(pl.col("proc") == proc)
            filters.append(f"proc={proc}")

        if chip is not None:
            df = df.filter(pl.col("chip_number") == chip)
            filters.append(f"chip={chip}")

        filtered_rows = len(df)

        if filtered_rows == 0:
            if format == "table":
                console.print("[yellow]No rows match the filter criteria[/yellow]")
                console.print()
            return

        # If format is not table, use formatters for JSON/CSV output
        if format != "table":
            from src.cli.formatters import get_formatter

            # Validate format
            try:
                formatter = get_formatter(format)
            except ValueError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

            # Limit output
            output_df = df.head(limit)

            # Build metadata
            metadata = {
                "manifest_path": str(manifest),
                "total_rows": total_rows,
                "filtered_rows": filtered_rows,
                "showing_rows": len(output_df),
                "filters": filters,
            }

            # Format and output
            output = formatter.format_dataframe(
                output_df,
                title="Manifest Data",
                metadata=metadata,
            )

            # Print to stdout (pipeable)
            print(output)
            return

        # Display table (table format only from here on)
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Manifest Inspector[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        console.print(f"[cyan]Loading:[/cyan] {manifest}")
        console.print(f"[green]✓[/green] Loaded {total_rows:,} rows")
        console.print()

        # Display filter info
        if filters:
            console.print(f"[cyan]Filters:[/cyan] {', '.join(filters)}")
            console.print(f"[cyan]Showing:[/cyan] {filtered_rows:,} of {total_rows:,} rows")
        else:
            console.print(f"[cyan]Showing:[/cyan] first {min(limit, filtered_rows):,} of {total_rows:,} rows")
        console.print()

        display_df = df.head(limit)

        # Select key columns for display
        display_cols = []
        for col in ["run_id", "proc", "chip_number", "date_local", "source_file", "summary"]:
            if col in display_df.columns:
                display_cols.append(col)

        table = Table(show_header=True, box=box.ROUNDED, header_style="bold cyan")
        for col in display_cols:
            table.add_column(col, overflow="fold")

        for row in display_df.select(display_cols).iter_rows(named=True):
            table.add_row(*[str(row[col])[:50] if row[col] else "—" for col in display_cols])

        console.print(table)
        console.print()

        if filtered_rows > limit:
            console.print(f"[dim]Showing {limit} of {filtered_rows:,} filtered rows. Use -n to show more.[/dim]")
            console.print()

    except Exception as e:
        console.print()
        console.print(Panel.fit(
            f"[bold red]✗ Inspection Failed[/bold red]\n"
            f"Error: {str(e)}",
            border_style="red"
        ))
        console.print()
        raise typer.Exit(1)


@cli_command(
    name="staging-stats",
    group="staging",
    description="Show staging statistics and disk usage"
)
def staging_stats_command(
    stage_root: Path = typer.Option(
        Path("data/02_stage"),
        "--stage-root",
        "-s",
        help="Staging root directory"
    ),
):
    """
    Show staging statistics and disk usage.

    Displays information about the staged data including:
    - Directory sizes
    - File counts
    - Partition distribution
    - Reject statistics

    Examples:

        # Show statistics for default staging directory
        process_and_analyze staging-stats

        # Show statistics for custom directory
        process_and_analyze staging-stats -s path/to/staging
    """
    import subprocess

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Staging Statistics[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    if not stage_root.exists():
        console.print(f"[bold red]Error:[/bold red] Staging root not found: {stage_root}")
        console.print()
        raise typer.Exit(1)

    # Build directory tree
    tree = Tree(f"[bold]{stage_root}[/bold]")

    for subdir in ["raw_measurements", "_manifest", "_rejects"]:
        path = stage_root / subdir
        if path.exists():
            # Get size
            try:
                result = subprocess.run(
                    ["du", "-sh", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                size = result.stdout.split()[0] if result.returncode == 0 else "?"
            except:
                size = "?"

            # Count files
            file_count = sum(1 for _ in path.rglob("*") if _.is_file())

            branch = tree.add(f"[cyan]{subdir}[/cyan] [dim]({size}, {file_count} files)[/dim]")

            # Show partitions for raw_measurements
            if subdir == "raw_measurements":
                proc_dirs = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("proc=")])
                for proc_dir in proc_dirs[:10]:  # Limit to first 10
                    proc_name = proc_dir.name.replace("proc=", "")
                    date_count = sum(1 for _ in proc_dir.iterdir() if _.is_dir())
                    branch.add(f"[yellow]{proc_name}[/yellow] [dim]({date_count} dates)[/dim]")
                if len(proc_dirs) > 10:
                    branch.add(f"[dim]... and {len(proc_dirs) - 10} more procedures[/dim]")

    console.print(tree)
    console.print()

    # Check manifest
    manifest_path = stage_root / "_manifest" / "manifest.parquet"
    if manifest_path.exists():
        try:
            import polars as pl
            df = pl.read_parquet(manifest_path)

            stats_table = Table(show_header=False, box=box.SIMPLE)
            stats_table.add_column("Metric", style="cyan", width=30)
            stats_table.add_column("Value", style="white")

            stats_table.add_row("Total Measurements", f"{len(df):,}")

            if "proc" in df.columns:
                proc_counts = df.group_by("proc").agg(pl.count().alias("count")).sort("count", descending=True)
                for row in proc_counts.iter_rows(named=True):
                    stats_table.add_row(f"  └─ {row['proc']}", f"{row['count']:,}")

            console.print(stats_table)
            console.print()
        except Exception as e:
            console.print(f"[yellow]⚠ Could not read manifest: {e}[/yellow]")
            console.print()

    console.print("[dim]Tip: Use [cyan]validate-manifest[/cyan] for detailed validation[/dim]")
    console.print()
