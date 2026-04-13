"""
Algorithms for derived metric extraction.

Pure computation functions that can be used by both extractors (scalar metrics)
and notebooks (full curves).
"""

from .hysteresis import (
    compute_raw_hysteresis,
    extract_raw_hysteresis_maxima,
    compute_poly_corrected_hysteresis,
    extract_poly_corrected_maxima,
)

__all__ = [
    "compute_raw_hysteresis",
    "extract_raw_hysteresis_maxima",
    "compute_poly_corrected_hysteresis",
    "extract_poly_corrected_maxima",
]
