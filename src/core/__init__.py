"""
Staging Layer - CSV to Parquet Pipeline
========================================

This module provides the staging layer that transforms raw measurement CSV files
into schema-validated, partitioned Parquet files with a centralized manifest.

Key Functions
-------------
- run_staging_pipeline: Main pipeline orchestrator
- discover_csvs: Find all CSV files in a directory tree
- merge_events_to_manifest: Consolidate staging events into manifest

Usage
-----
Basic staging from Python:
    >>> from pathlib import Path
    >>> from src.models.parameters import StagingParameters
    >>> from src.core import run_staging_pipeline
    >>>
    >>> params = StagingParameters(
    ...     raw_root=Path("data/01_raw"),
    ...     stage_root=Path("data/02_stage/raw_measurements"),
    ...     procedures_yaml=Path("config/procedures.yml"),
    ...     workers=8,
    ...     force=True
    ... )
    >>> run_staging_pipeline(params)

Command-line usage:
    $ python -m src.core.stage_raw_measurements \
        --raw-root data/01_raw \
        --stage-root data/02_stage/raw_measurements \
        --procedures-yaml config/procedures.yml \
        --workers 8 --force

Architecture
------------
Raw CSV -> Header Parser -> Type Validation -> Parquet Writer -> Manifest
                |                                     |
         procedures.yml                    Hive partitioning:
         (schema spec)                     proc=X/date=Y/run_id=Z/

Features
--------
- Parallel processing with ProcessPoolExecutor
- Atomic writes (temp file + rename)
- Idempotent (deterministic run_id from SHA-1)
- Type validation via YAML schema
- Timezone-aware timestamp handling
- Reject/event logging for observability

See Also
--------
- stage_raw_measurements.py: Main pipeline implementation
- stage_utils.py: Utility functions (parsing, hashing, etc.)
- README.md: Detailed documentation
"""

from .stage_raw_measurements import (
    run_staging_pipeline,
    run_staging_pipeline_tui,
    discover_csvs,
    merge_events_to_manifest,
    load_procedures_yaml,
    get_procs_cached,
    ingest_file_task,
)

__all__ = [
    "run_staging_pipeline",
    "run_staging_pipeline_tui",
    "discover_csvs",
    "merge_events_to_manifest",
    "load_procedures_yaml",
    "get_procs_cached",
    "ingest_file_task",
]
