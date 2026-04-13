"""
Schema validation for measurement CSV files against procedures.yml specification.

Supports schema evolution:
- Required vs optional columns (backward compatible with old data)
- Graceful degradation (missing optional columns → null values)
- Validation warnings for data quality issues
- Strict mode for production validation

Usage:
    from src.core.schema_validator import validate_measurement_schema

    result = validate_measurement_schema(
        proc="IV",
        spec=proc_spec,
        parsed_params=params_dict,
        parsed_meta=meta_dict,
        df_columns=["Vsd (V)", "I (A)"],
        strict=False
    )

    if result.has_errors and strict:
        raise ValidationError(result.errors)

    for warning in result.warnings:
        print(f"[warn] {warning}")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Union
from enum import Enum


class Severity(Enum):
    """Validation message severity levels."""
    ERROR = "ERROR"    # Critical - fails in strict mode
    WARN = "WARN"      # Important but non-blocking
    INFO = "INFO"      # Informational


@dataclass
class ColumnSpec:
    """
    Specification for a single column (Parameter, Metadata, or Data).

    Supports two YAML formats:
    1. Simple: "Laser wavelength: float"
    2. Extended: "Laser wavelength: {type: float, required: false}"

    Attributes:
        name: Column name (e.g., "Laser wavelength")
        type: Data type (e.g., "float", "int", "datetime")
        required: Whether column must be present (default: varies by section)
    """
    name: str
    type: str
    required: bool = True

    @classmethod
    def from_yaml_value(cls, name: str, value: Union[str, Dict[str, Any]], default_required: bool = True) -> ColumnSpec:
        """
        Parse column specification from YAML value.

        Supports both simple string format and extended dict format:
        - Simple: value = "float" → ColumnSpec(name, "float", default_required)
        - Extended: value = {"type": "float", "required": false} → ColumnSpec(name, "float", False)

        Args:
            name: Column name from YAML key
            value: YAML value (string or dict)
            default_required: Default required status if not specified

        Returns:
            ColumnSpec instance

        Example:
            >>> ColumnSpec.from_yaml_value("VDS", "float")
            ColumnSpec(name='VDS', type='float', required=True)

            >>> ColumnSpec.from_yaml_value("t (s)", {"type": "float", "required": False})
            ColumnSpec(name='t (s)', type='float', required=False)
        """
        if isinstance(value, str):
            return cls(name=name, type=value, required=default_required)
        elif isinstance(value, dict):
            return cls(
                name=name,
                type=value.get("type", "str"),
                required=value.get("required", default_required)
            )
        else:
            return cls(name=name, type="str", required=default_required)


@dataclass
class ValidationMessage:
    """Single validation message with severity and context."""
    severity: Severity
    section: str  # "Parameters", "Metadata", or "Data"
    message: str
    column: Optional[str] = None
    suggestion: Optional[str] = None

    def format(self) -> str:
        """Format message for console output."""
        prefix = f"[{self.severity.value.lower()}]"
        if self.column:
            msg = f"{prefix} {self.section}: column '{self.column}' - {self.message}"
        else:
            msg = f"{prefix} {self.section}: {self.message}"

        if self.suggestion:
            msg += f" (suggestion: {self.suggestion})"

        return msg


@dataclass
class ValidationResult:
    """
    Complete validation result for a measurement file.

    Contains all validation messages categorized by severity,
    plus helpers for decision making.
    """
    proc: str
    messages: List[ValidationMessage] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationMessage]:
        """Critical errors that should fail in strict mode."""
        return [m for m in self.messages if m.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        """Important warnings that should be logged."""
        return [m for m in self.messages if m.severity == Severity.WARN]

    @property
    def info(self) -> List[ValidationMessage]:
        """Informational messages."""
        return [m for m in self.messages if m.severity == Severity.INFO]

    @property
    def has_errors(self) -> bool:
        """True if any ERROR-level messages exist."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """True if any WARN-level messages exist."""
        return len(self.warnings) > 0

    def add_error(self, section: str, message: str, column: Optional[str] = None, suggestion: Optional[str] = None):
        """Add an ERROR-level message."""
        self.messages.append(ValidationMessage(Severity.ERROR, section, message, column, suggestion))

    def add_warning(self, section: str, message: str, column: Optional[str] = None, suggestion: Optional[str] = None):
        """Add a WARN-level message."""
        self.messages.append(ValidationMessage(Severity.WARN, section, message, column, suggestion))

    def add_info(self, section: str, message: str, column: Optional[str] = None):
        """Add an INFO-level message."""
        self.messages.append(ValidationMessage(Severity.INFO, section, message, column))

    def format_all(self) -> str:
        """Format all messages for console output."""
        return "\n".join(m.format() for m in self.messages)


def parse_column_specs(yaml_dict: Dict[str, Any], default_required: bool = True) -> Dict[str, ColumnSpec]:
    """
    Parse column specifications from YAML dictionary.

    Args:
        yaml_dict: Dictionary from YAML (Parameters, Metadata, or Data section)
        default_required: Default required status for columns without explicit setting

    Returns:
        Dictionary mapping column names to ColumnSpec objects

    Example:
        >>> yaml_data = {
        ...     "VDS": "float",
        ...     "t (s)": {"type": "float", "required": False}
        ... }
        >>> specs = parse_column_specs(yaml_data)
        >>> specs["VDS"].required
        True
        >>> specs["t (s)"].required
        False
    """
    specs = {}
    for name, value in yaml_dict.items():
        specs[name] = ColumnSpec.from_yaml_value(name, value, default_required)
    return specs


def find_similar_column(target: str, available: List[str], threshold: float = 0.6) -> Optional[str]:
    """
    Find similar column name using simple similarity heuristic.

    Uses normalized string matching (removes spaces, punctuation, case).

    Args:
        target: Column name to find match for
        available: List of available column names
        threshold: Similarity threshold (0.0 to 1.0)

    Returns:
        Most similar column name if similarity >= threshold, else None

    Example:
        >>> find_similar_column("time_s", ["t (s)", "I (A)", "Vsd (V)"])
        't (s)'
        >>> find_similar_column("gate_voltage", ["Vg (V)", "Vsd (V)"])
        'Vg (V)'
    """
    from difflib import SequenceMatcher

    def normalize(s: str) -> str:
        """Normalize for comparison: lowercase, no spaces/punctuation."""
        import re
        s = s.lower()
        s = re.sub(r"[^\w]", "", s)
        return s

    target_norm = normalize(target)
    best_match = None
    best_score = 0.0

    for candidate in available:
        candidate_norm = normalize(candidate)
        score = SequenceMatcher(None, target_norm, candidate_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match
    return None


def validate_parameters(
    proc: str,
    param_specs: Dict[str, ColumnSpec],
    parsed_params: Dict[str, Any],
    result: ValidationResult,
    proc_config: Dict[str, Any] = None
) -> None:
    """
    Validate Parameters section.

    Checks:
    - Required parameters are present
    - Unknown parameters (potential typos)
    - Critical parameters (chip_number, chip_group) exist (unless requires_chip=false)

    Args:
        proc: Procedure name
        param_specs: Expected parameter specifications from YAML
        parsed_params: Parsed parameters from CSV header
        result: ValidationResult to append messages to
        proc_config: Optional procedure configuration from YAML Config section
    """
    # Check if procedure requires chip information
    requires_chip = True
    if proc_config and "requires_chip" in proc_config:
        requires_chip = proc_config.get("requires_chip", True)

    # Check critical parameters (chip info) - only if required
    if requires_chip:
        critical_params = {"Chip number", "Chip group name"}
        for param in critical_params:
            if param not in parsed_params or parsed_params[param] is None:
                result.add_error(
                    "Parameters",
                    f"critical parameter missing: '{param}' (required for all procedures)",
                    column=param
                )

    # Check required parameters from schema
    spec_names = set(spec.name for spec in param_specs.values())
    for spec in param_specs.values():
        if spec.required and spec.name not in parsed_params:
            # Check if similar name exists (potential typo)
            similar = find_similar_column(spec.name, list(parsed_params.keys()))
            if similar:
                result.add_warning(
                    "Parameters",
                    f"required parameter '{spec.name}' not found",
                    column=spec.name,
                    suggestion=f"found similar '{similar}'"
                )
            else:
                result.add_warning(
                    "Parameters",
                    f"required parameter '{spec.name}' not found (will use null/default)",
                    column=spec.name
                )

    # Check for unknown parameters (not in schema)
    for param_name in parsed_params.keys():
        if param_name not in spec_names:
            result.add_info(
                "Parameters",
                f"parameter '{param_name}' not in schema (may be procedure-specific or deprecated)",
                column=param_name
            )


def validate_metadata(
    proc: str,
    meta_specs: Dict[str, ColumnSpec],
    parsed_meta: Dict[str, Any],
    result: ValidationResult
) -> None:
    """
    Validate Metadata section.

    Checks:
    - Required metadata fields are present (especially "Start time")
    - Unknown metadata fields

    Args:
        proc: Procedure name
        meta_specs: Expected metadata specifications from YAML
        parsed_meta: Parsed metadata from CSV header
        result: ValidationResult to append messages to
    """
    # Start time is critical for timestamp resolution
    if "Start time" not in parsed_meta or parsed_meta["Start time"] is None:
        result.add_warning(
            "Metadata",
            "'Start time' not found - will use file path or mtime fallback",
            column="Start time"
        )

    # Check required metadata from schema
    spec_names = set(spec.name for spec in meta_specs.values())
    for spec in meta_specs.values():
        if spec.required and spec.name not in parsed_meta:
            result.add_warning(
                "Metadata",
                f"required field '{spec.name}' not found",
                column=spec.name
            )

    # Check for unknown metadata
    for meta_name in parsed_meta.keys():
        if meta_name not in spec_names:
            result.add_info(
                "Metadata",
                f"field '{meta_name}' not in schema",
                column=meta_name
            )


def validate_data_columns(
    proc: str,
    data_specs: Dict[str, ColumnSpec],
    df_columns: List[str],
    rename_map: Dict[str, str],
    result: ValidationResult
) -> Dict[str, ColumnSpec]:
    """
    Validate Data columns section.

    Checks:
    - Required data columns are present (mapped or renamed)
    - Optional data columns missing (will be added as null)
    - Unmapped CSV columns (potential typos or undocumented columns)

    Args:
        proc: Procedure name
        data_specs: Expected data column specifications from YAML
        df_columns: Actual column names from CSV DataFrame
        rename_map: Mapping from CSV columns to YAML canonical names (from build_yaml_rename_map)
        result: ValidationResult to append messages to

    Returns:
        Dictionary of missing optional columns that need to be added as null

    Example:
        >>> missing = validate_data_columns(
        ...     proc="IV",
        ...     data_specs={"Vsd (V)": ColumnSpec("Vsd (V)", "float", True),
        ...                 "t (s)": ColumnSpec("t (s)", "float", False)},
        ...     df_columns=["VDS", "I"],
        ...     rename_map={"VDS": "Vsd (V)"},
        ...     result=result
        ... )
        >>> # "Vsd (V)" is mapped, "t (s)" is missing optional
        >>> missing
        {"t (s)": ColumnSpec("t (s)", "float", False)}
    """
    # Track which YAML columns are covered by CSV
    mapped_yaml_cols = set(rename_map.values())
    unmapped_csv_cols = set(df_columns) - set(rename_map.keys())

    missing_optional = {}

    # Check each expected column
    for spec in data_specs.values():
        if spec.name not in mapped_yaml_cols:
            # Column is missing from CSV
            if spec.required:
                # Required column missing - ERROR in strict mode
                similar = find_similar_column(spec.name, df_columns)
                if similar:
                    result.add_error(
                        "Data",
                        f"required column '{spec.name}' not found",
                        column=spec.name,
                        suggestion=f"found unmapped column '{similar}' - check synonym rules"
                    )
                else:
                    result.add_error(
                        "Data",
                        f"required column '{spec.name}' not found in CSV",
                        column=spec.name
                    )
            else:
                # Optional column missing - WARN and track for null filling
                result.add_warning(
                    "Data",
                    f"optional column '{spec.name}' not found (will add as null for schema consistency)",
                    column=spec.name
                )
                missing_optional[spec.name] = spec

    # Check for unmapped CSV columns (potential typos or undocumented)
    for csv_col in unmapped_csv_cols:
        # Check if it's similar to any YAML column
        similar = find_similar_column(csv_col, [spec.name for spec in data_specs.values()])
        if similar:
            result.add_warning(
                "Data",
                f"unmapped CSV column '{csv_col}'",
                column=csv_col,
                suggestion=f"similar to YAML column '{similar}' - potential typo or missing synonym?"
            )
        else:
            result.add_info(
                "Data",
                f"unmapped CSV column '{csv_col}' (not in YAML schema - will be kept if not using --only-yaml-data)",
                column=csv_col
            )

    return missing_optional


def validate_measurement_schema(
    proc: str,
    param_specs: Dict[str, Any],
    meta_specs: Dict[str, Any],
    data_specs: Dict[str, Any],
    parsed_params: Dict[str, Any],
    parsed_meta: Dict[str, Any],
    df_columns: List[str],
    rename_map: Dict[str, str],
    strict: bool = False,
    proc_config: Dict[str, Any] = None
) -> ValidationResult:
    """
    Validate complete measurement schema.

    This is the main entry point for schema validation. Checks Parameters,
    Metadata, and Data sections against YAML specification.

    Args:
        proc: Procedure name (e.g., "IV", "IVg", "ITt")
        param_specs: Parameter specifications from YAML (raw dict)
        meta_specs: Metadata specifications from YAML (raw dict)
        data_specs: Data column specifications from YAML (raw dict)
        parsed_params: Parsed parameters from CSV header
        parsed_meta: Parsed metadata from CSV header
        df_columns: Column names from CSV DataFrame
        rename_map: Mapping from CSV columns to YAML canonical names
        strict: If True, treat validation errors as critical
        proc_config: Optional procedure configuration from YAML Config section

    Returns:
        ValidationResult with all messages and missing column info

    Example:
        >>> result = validate_measurement_schema(
        ...     proc="IV",
        ...     param_specs={"VDS": "float", "Chip number": "int"},
        ...     meta_specs={"Start time": "datetime"},
        ...     data_specs={"Vsd (V)": "float", "I (A)": "float", "t (s)": {"type": "float", "required": False}},
        ...     parsed_params={"VDS": 0.1, "Chip number": 67},
        ...     parsed_meta={"Start time": datetime.now()},
        ...     df_columns=["vsd_v", "i_a"],
        ...     rename_map={"vsd_v": "Vsd (V)", "i_a": "I (A)"},
        ...     strict=False,
        ...     proc_config={"requires_chip": True}
        ... )
        >>> result.has_errors
        False
        >>> len(result.warnings)
        1  # Missing optional "t (s)" column
    """
    result = ValidationResult(proc=proc)

    # Parse column specs (convert YAML dict to ColumnSpec objects)
    # Default: Parameters are required, Data columns are optional (for backward compat)
    param_col_specs = parse_column_specs(param_specs, default_required=False)
    meta_col_specs = parse_column_specs(meta_specs, default_required=False)
    data_col_specs = parse_column_specs(data_specs, default_required=False)  # Changed to False for backward compat

    # Validate each section - pass proc_config to validate_parameters
    validate_parameters(proc, param_col_specs, parsed_params, result, proc_config)
    validate_metadata(proc, meta_col_specs, parsed_meta, result)
    missing_optional = validate_data_columns(proc, data_col_specs, df_columns, rename_map, result)

    # Store missing optional columns in result for downstream use
    result.missing_optional_columns = missing_optional

    return result
