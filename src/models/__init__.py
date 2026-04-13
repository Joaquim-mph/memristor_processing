"""
Data models and configuration schemas for the nanolab pipeline.

Staging Layer
-------------
- StagingConfig: Configuration for CSV-to-Parquet staging pipeline
- ManifestRow: Schema for manifest.parquet metadata table
- Proc: Procedure type enum (IVg, IV, It, etc.)

Analysis & Plotting
-------------------
- StagingParameters: Legacy staging config (use StagingConfig instead)
- IntermediateParameters: Intermediate preprocessing layer config
- IVAnalysisParameters: IV curve analysis parameters
- PlottingParameters: Visualization and plotting config
- PipelineParameters: Combined parameters for full pipeline
"""

# Staging layer models (Phase 1 - new staging architecture)
from .config import StagingConfig
from .manifest import ManifestRow, Proc, proc_display_name, proc_short_name

# Pipeline parameter models (existing)
from .parameters import (
    StagingParameters,
    IntermediateParameters,
    IVAnalysisParameters,
    PlottingParameters,
    PipelineParameters,
)

__all__ = [
    # Staging layer (new)
    "StagingConfig",
    "ManifestRow",
    "Proc",
    "proc_display_name",
    "proc_short_name",
    # Pipeline parameters (existing)
    "StagingParameters",
    "IntermediateParameters",
    "IVAnalysisParameters",
    "PlottingParameters",
    "PipelineParameters",
]
