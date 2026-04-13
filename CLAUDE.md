# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python data processing and visualization pipeline for 2-terminal memristor IV characterization. Processes raw measurement CSVs from lab equipment (Keithley sourcemeter), stages them into Parquet format, builds experiment histories, averages repeated sweeps, and extracts memristor figures of merit (hysteresis area, coercive voltage, on/off ratio, etc.).

**Key Technologies:** Python 3.11+, Polars (NOT pandas), Pydantic v2+, Typer + Rich CLI, Matplotlib + scienceplots

## Environment Setup

```bash
source .venv/bin/activate
pip install -e .              # Editable install, registers `memristor` command
pip install -e ".[dev]"       # Include pytest
```

## Entry Point

| Command | Entry point | Description |
|---|---|---|
| `memristor` | `src.cli.main:main` | Typer CLI (data processing, plotting, validation) |

## Essential Commands

```bash
# Staging and history
memristor stage-all              # CSV -> Parquet
memristor build-all-histories    # Manifest -> device histories

# Validation
memristor validate-manifest
```

### Testing

```bash
python3 -m pytest tests/ -v
```

## Architecture

### Data Flow

```
data/01_raw/          Raw CSVs from PyMeasure
    |
    v
[stage-all]           CSV -> Parquet + manifest
    |
    v
data/02_stage/
  raw_measurements/   Parquet files by procedure type
  _manifest/          manifest.parquet (one row per measurement)
  device_histories/   Per-device Parquet (grouped by `information` field)
    |
    v
data/03_derived/
  averaged/           Averaged I-V curves with uncertainty
  _metrics/           Extracted memristor figures of merit
```

### Module Structure

- **`src/core/`** - Staging pipeline, schema validation, history builder, utilities
- **`src/models/`** - Pydantic schemas (manifest rows, staging params)
- **`src/derived/`** - Metric extraction pipeline with auto-discovered extractors
- **`src/averaging/`** - IV sweep averaging (grouping, interpolation, mean/std)
- **`src/plotting/`** - Plot modules for memristor visualization
- **`src/cli/`** - Typer CLI with plugin auto-discovery (`@cli_command` decorator)

### Configuration

- **`config/procedures.yml`** - Schema definitions for measurement procedures
- **`config/cli_plugins.yaml`** - Enable/disable CLI command groups

## Critical Rules

### Always Use Polars, Never pandas
Different API: `.filter()` not `.query()`, `.select()` not `[]`.

### Parquet is Source of Truth
Never read CSV files directly in new code. Use `read_measurement_parquet()` from `src/core/utils.py`.

### Device Identification
This project uses the `information` field from CSV metadata as the primary device identifier. There is NO chip_group/chip_number/chip_name convention. History files are grouped by the `information` field value.

### Plotting Style
- **NO GRIDS** -- never call `plt.grid(True)` or `ax.grid(True)`
- Use data procedure names (`procedure="IV"`)

## Development Patterns

### Adding a CLI Command

Create file in `src/cli/commands/` with `@cli_command` decorator -- auto-discovered:

```python
from src.cli.plugin_system import cli_command
import typer

@cli_command(name="my-command", group="plotting", description="Brief description")
def my_command(args: ...):
    pass
```

### Adding a Derived Metric Extractor

1. Create `src/derived/extractors/my_metric.py` inheriting `MetricExtractor`
2. Implement `applicable_procedures`, `metric_name`, `extract()`, `validate()`
3. Import in `extractors/__init__.py`

### Adding a New Procedure Type

1. Add schema to `config/procedures.yml` (Parameters, Metadata, Data sections)
2. No Python changes needed -- staging auto-validates

## Key Concepts

- **Run IDs**: Deterministic `SHA1(normalized_path|timestamp_utc)[:16]` for idempotent staging
- **Timezone**: Raw timestamps are Unix epoch, localized to `America/Santiago`, stored as UTC
- **Device grouping**: Based on `information` metadata field, not filenames
