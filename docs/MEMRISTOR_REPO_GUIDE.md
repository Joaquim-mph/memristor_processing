# Memristor Repo: Migration Guide from Biotite

This document captures the findings and plan for creating a standalone repository for 2-terminal memristor IV characterization, reusing the staging infrastructure from biotite.

## Context

- **Source repo**: biotite (optothermal semiconductor pipeline for 3-terminal FETs)
- **Target**: New repo for 2-terminal memristor analysis
- **Same lab software**: CSV format is identical (PyMeasure with `# Procedure:` / `# Parameters:` / `# Metadata:` / `# Data:` comment blocks)
- **Procedures**: Primarily `IV` (voltage sweep), plus a future custom ramp procedure
- **Key new requirement**: Experiment averaging (multiple IV sweeps under same conditions -> averaged curve)
- **Device identification**: No `chip_group`/`chip_number`/`chip_name` -- relies on the `information` field in the manifest

---

## What to Copy (Verbatim)

These modules work as-is with no modifications:

| File / Directory | Purpose |
|---|---|
| `src/core/stage_raw_measurements.py` | CSV parsing, header extraction, type casting, Parquet writer. Fully config-driven via `procedures.yml` |
| `src/core/schema_validator.py` | Validates staged data against YAML procedure specs. Severity levels (ERROR/WARN/INFO) |
| `src/core/utils.py` | `read_measurement_parquet()` -- single function all downstream code uses to load staged data |
| `src/core/pipeline.py` | Formal pipeline builder with error handling, rollback, checkpointing |
| `src/cli/main.py` | Typer app skeleton, config loading, `get_config()`/`set_config()` singleton |
| `src/cli/plugin_system.py` | `@cli_command` decorator + `discover_commands()` auto-discovery. Drop a file in `commands/`, it registers itself |
| `src/cli/commands/stage.py` | `stage-all` CLI command (discovers CSVs, creates `StagingParameters`, runs pipeline) |
| `src/models/parameters.py` | `StagingParameters` Pydantic model (paths, workers, force flag, etc.) |
| `src/derived/extractors/base.py` | `MetricExtractor` ABC -- reuse the pattern for memristor extractors |
| `src/derived/metric_pipeline.py` | `MetricPipeline` orchestrator -- swap out the `_default_extractors()` list |
| `src/models/derived_metrics.py` | `DerivedMetric` Pydantic model (run_id, metric_name, value, confidence, flags) |

---

## What to Adapt

### 1. `config/procedures.yml` -- Strip and Simplify

Keep the `ManifestColumnMap` section but remove transistor-specific entries (gate voltage fields, laser fields). Keep only what IV memristor measurements use.

**Keep from current IV procedure:**
```yaml
IV:
  Parameters:
    Irange: float
    NPLC: int
    N_avg: int
    Burn-in time: float
    Information: str          # <-- Primary device identifier
    Sample: str
    Step time: float
    VSD end: float
    VSD start: float
    VSD step: float
  Metadata:
    Start time: datetime
  Data:
    Vsd (V): float
    I (A): float
    t (s):
      type: float
      required: false
```

**Remove**: `Chip group name`, `Chip number`, `Laser toggle`, `Laser voltage`, `Laser wavelength`, `VG`, `Show more`, `Procedure version`, `Chained execution` -- unless your PyMeasure procedure still emits them. If the CSV contains them but you don't need them, leaving them in the YAML is harmless (they get parsed and stored but ignored downstream).

**Future ramp procedure**: Add a new entry when ready (e.g., `IVRamp:` with different voltage step parameters). No Python changes needed -- staging auto-discovers it from the YAML.

### 2. `src/models/manifest.py` -- Simplify for 2-Terminal

**Remove these fields entirely** (transistor/optoelectronic-specific):
- `chip_group`, `chip_number`, `chip_name`, `file_idx` (no chip naming convention)
- `has_light`, `laser_voltage_V`, `wavelength_nm`, `laser_period_s` (no laser)
- `vg_fixed_v`, `vg_start_v`, `vg_end_v`, `vg_step_v` (no gate)
- `ids_a`, `vrange` (not applicable)
- `phase1_duration_s`, `phase2_duration_s`, `phase3_duration_s` (no 3-phase)
- `initial_temp_c`, `target_temp_c`, `temp_*` fields (unless you measure at different temperatures)
- `optical_fiber`, `laser_voltage_start_v`, etc. (no laser calibration)

**Keep and promote:**
- `information: str` -- **this becomes your primary device/sample identifier** (currently Optional, make it required or at least the primary grouping key)
- `sample: Optional[str]` -- secondary identifier if needed
- `run_id`, `source_file`, `proc`, `date_local`, `start_time_utc` (identity)
- `vsd_start_v`, `vsd_end_v`, `vsd_step_v` (IV sweep parameters)
- `irange`, `nplc`, `n_avg`, `burn_in_time_s`, `step_time_s` (instrument settings)
- `duration_s`, `summary`, `schema_version`, `extraction_version`, `ingested_at_utc`
- All staging internals: `status`, `path`, `rows`, `date_origin`, `validation_*`

**Simplify the `Proc` literal:**
```python
Proc = Literal["IV"]  # Add "IVRamp" later when the new procedure is ready
```

**Update `proc_display_name()` and `proc_short_name()`** to only include memristor-relevant procedures.

### 3. `src/core/history_builder.py` -- Already Supports `information`

The current history builder has a dual identification system:
- **Primary**: `chip_number` + `chip_group` (what biotite uses)
- **Fallback**: `information` field (what you need)

Key code paths that already work for you (lines ~97-104, ~504-524):
- When `chip_number` is null, `generate_all_chip_histories()` groups by `information` column instead
- Filenames are generated from cleaned `information` strings
- `build_chip_history_from_manifest()` accepts `information=` parameter for filtering

**What to change**: Make `information` the primary path, not the fallback. Simplify by removing the `chip_number`/`chip_group` branches entirely. Consider renaming the function parameter from `information` to `device_id` or `sample_name` for clarity.

### 4. `config/chip_params.yaml` -- Replace or Remove

This file contains transistor-specific chip metadata. Either:
- **Replace** with a `device_params.yaml` containing memristor device metadata (material, electrode stack, area, etc.)
- **Remove** if all relevant info lives in the `information` field and CSV parameters

### 5. `config/cli_plugins.yaml` -- Keep As-Is

The plugin system is group-based. Your new commands will register under groups like `"plotting"`, `"pipeline"`, etc. No changes needed unless you want different group names.

### 6. `pyproject.toml` -- New Package Identity

- Rename package (e.g., `memristor-pipeline` or your preferred name)
- Update entry points (e.g., `memristor` CLI command)
- Keep the same dependency stack: polars, pydantic, typer, rich, matplotlib, scienceplots
- Add numba if you plan numerical fitting (hysteresis area integration, etc.)

---

## What to Build New

### 1. Averaging Layer (New Module: `src/averaging/`)

This sits between staging and derived metrics. Purpose: group multiple IV sweeps by matching conditions and produce averaged I-V curves.

**Suggested approach:**
```
src/averaging/
    __init__.py
    grouping.py       # Group experiments by conditions (information, vsd range, etc.)
    interpolation.py  # Interpolate curves onto common voltage grid
    averager.py       # Compute mean + std across grouped sweeps
```

**Pipeline position:**
```
stage-all -> build-histories -> average-sweeps -> derive-metrics -> plot
                                 ^^^^^^^^^^^
                                 NEW STEP
```

**Grouping logic**: Use `information` + voltage sweep parameters (`vsd_start_v`, `vsd_end_v`, `vsd_step_v`) + instrument settings to identify "same condition" experiments.

**Output**: Averaged Parquet files in `data/03_derived/averaged/` (or similar), with columns:
- `Vsd (V)`: Common voltage grid
- `I_mean (A)`: Averaged current
- `I_std (A)`: Standard deviation
- `n_sweeps`: Number of sweeps in average
- Metadata columns linking back to source run_ids

**CLI command**: `average-sweeps` with `@cli_command(group="pipeline")`

### 2. Memristor Extractors (New: `src/derived/extractors/`)

Create new extractor classes inheriting from `MetricExtractor`. Each declares `applicable_procedures = ["IV"]`.

| Extractor | Metric | Description |
|---|---|---|
| `HysteresisAreaExtractor` | `hysteresis_area` | Enclosed area of the I-V loop (forward - backward). Use `np.trapz()` or Polars equivalent on interpolated curves |
| `CoerciveVoltageExtractor` | `coercive_voltage_pos`, `coercive_voltage_neg` | Voltage where I crosses zero on forward and backward sweeps |
| `DissipatedPowerExtractor` | `dissipated_power` | Integral of V*I over a full cycle |
| `OnOffRatioExtractor` | `on_off_ratio` | Ratio of max to min current (or at specific read voltages) |
| `SetResetVoltageExtractor` | `set_voltage`, `reset_voltage` | Voltages where resistance switching occurs (dI/dV peaks) |
| `ResistanceStateExtractor` | `r_hrs`, `r_lrs` | High/low resistance state values at a defined read voltage |
| `EnduranceExtractor` | `endurance_cycles` | Track on/off ratio degradation over repeated sweeps (uses averaging groups) |
| `RetentionExtractor` | `retention_time` | From time-series data if you add an It procedure later |

### 3. Memristor Plot Commands (New: `src/cli/commands/plot_*.py`)

| Command | Purpose |
|---|---|
| `plot-iv-loop` | Single or overlaid I-V hysteresis loops with forward/backward branches colored |
| `plot-iv-avg` | Averaged I-V with standard deviation shading |
| `plot-endurance` | On/off ratio vs cycle number |
| `plot-resistance-states` | HRS/LRS values over time or cycle number |
| `plot-coercive-voltage` | Coercive voltage evolution over cycles |
| `plot-power` | Dissipated power per cycle |
| `plot-retention` | Resistance state vs time (if It data available) |

---

## Data Flow (New Repo)

```
data/01_raw/                  Raw CSVs from PyMeasure (same format as biotite)
    |
    v
[stage-all]                   CSV -> Parquet + manifest (reused from biotite)
    |
    v
data/02_stage/
  raw_measurements/proc=IV/   Hive-partitioned Parquet files
  _manifest/manifest.parquet  One row per sweep, `information` as device identifier
  device_histories/           Per-device Parquet (grouped by `information`)
    |
    v
[average-sweeps]              NEW: Group by conditions, interpolate, average
    |
    v
data/03_derived/
  averaged/                   Averaged I-V curves with uncertainty
  _metrics/metrics.parquet    Extracted memristor figures of merit
  device_histories_enriched/  Histories + derived metric columns
    |
    v
[plot commands]               Publication-quality memristor figures
    |
    v
figs/
```

---

## Key Differences from Biotite (Summary)

| Aspect | Biotite (FET) | New Repo (Memristor) |
|---|---|---|
| Device ID | `chip_group` + `chip_number` from filename | `information` field from CSV parameters |
| Terminal count | 3-terminal (gate, drain, source) | 2-terminal (top, bottom electrode) |
| Key procedures | IVg, It, VVg, Vt, LaserCalibration | IV (+ future IVRamp) |
| Derived metrics | CNP, photoresponse, relaxation times | Hysteresis area, coercive voltage, on/off ratio, set/reset voltage |
| Light/laser | Core feature (dark vs illuminated) | Not applicable |
| Temperature | Optional (IVgT, ITt) | Possibly relevant -- keep if needed |
| Averaging | Not needed | Core requirement (multiple sweeps -> averaged curve) |
| History grouping | By chip number | By `information` field |

---

## Migration Checklist

1. [ ] Create new repo, copy modules listed in "What to Copy"
2. [ ] Strip `procedures.yml` to IV-only (+ ManifestColumnMap cleanup)
3. [ ] Simplify `ManifestRow` (remove FET/laser fields, promote `information`)
4. [ ] Simplify `Proc` literal to `Literal["IV"]`
5. [ ] Refactor `history_builder.py` to use `information` as primary key
6. [ ] Update `pyproject.toml` with new package name and entry points
7. [ ] Build averaging module (`src/averaging/`)
8. [ ] Implement memristor extractors (hysteresis area, coercive voltage, etc.)
9. [ ] Build IV loop plot command with forward/backward branch visualization
10. [ ] Build averaged IV plot command with uncertainty bands
11. [ ] Test staging with real memristor CSV files
12. [ ] Add remaining plot commands as needed

---

## Files to NOT Copy

These are biotite-specific and have no value in the memristor repo:

- `src/plotting/its.py`, `ivg.py`, `vvg.py`, `vt.py`, `transconductance.py`, `cnp_time.py`, `photoresponse.py`, `laser_calibration.py` -- all FET-specific
- `src/derived/extractors/cnp_extractor.py`, `photoresponse_extractor.py`, `its_relaxation_extractor.py`, `its_three_phase_fit_extractor.py`, `calibration_matcher.py` -- all FET-specific
- `src/cli/commands/plot_its.py`, `plot_ivg.py`, `plot_vvg.py`, `plot_vt.py`, etc. -- all FET-specific
- `src/models/parameters.py` `IntermediateParameters`, `IVAnalysisParameters` -- FET analysis models
- `src/tui/` -- Rebuild if needed, but the wizard is biotite-specific
- `config/chip_params.yaml` -- FET chip metadata
- `config/batch_plots/` -- biotite-specific plot configs
