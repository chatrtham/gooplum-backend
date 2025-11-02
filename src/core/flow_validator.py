"""Flow validator for input validation against flow signatures."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.flow_discovery import FlowMetadata, FlowParameter


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """Represents a validation error or warning."""

    field: str
    message: str
    severity: ValidationSeverity
    expected_type: Optional[str] = None
    actual_value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    sanitized_parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class FlowValidator:
    """Validates input parameters against flow signatures."""

    def __init__(self):
        self.type_validators = {
            "str": self._validate_string,
            "int": self._validate_integer,
            "float": self._validate_float,
            "bool": self._validate_boolean,
            "list": self._validate_list,
            "dict": self._validate_dict,
            "any": self._validate_any,
        }

    def validate_parameters(
        self,
        flow_metadata: FlowMetadata,
        parameters: Dict[str, Any],
        strict_mode: bool = False,
    ) -> ValidationResult:
        """
        Validate parameters against flow signature.

        Args:
            flow_metadata: Metadata of the flow to validate against
            parameters: Input parameters to validate
            strict_mode: If True, reject unknown parameters

        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        sanitized_params = parameters.copy()

        # Create parameter lookup
        param_lookup = {p.name: p for p in flow_metadata.parameters}

        # Check for missing required parameters
        self._check_required_parameters(param_lookup, parameters, errors)

        # Check for unknown parameters
        if strict_mode:
            self._check_unknown_parameters(
                param_lookup, parameters, errors, ValidationSeverity.ERROR
            )
        else:
            self._check_unknown_parameters(
                param_lookup, parameters, warnings, ValidationSeverity.WARNING
            )

        # Validate each provided parameter
        for param_name, param_value in parameters.items():
            if param_name in param_lookup:
                param_schema = param_lookup[param_name]
                self._validate_single_parameter(
                    param_schema, param_value, errors, warnings
                )

                # Sanitize parameter if needed
                sanitized_value = self._sanitize_parameter(param_schema, param_value)
                if sanitized_value is not None:
                    sanitized_params[param_name] = sanitized_value

        # Fill in default values for missing optional parameters
        self._fill_default_values(param_lookup, sanitized_params, warnings)

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sanitized_parameters=sanitized_params if is_valid else None,
            metadata={
                "flow_name": flow_metadata.name,
                "parameter_count": len(parameters),
                "required_count": len(
                    [p for p in flow_metadata.parameters if p.required]
                ),
                "validation_mode": "strict" if strict_mode else "lenient",
            },
        )

    def _check_required_parameters(
        self,
        param_lookup: Dict[str, FlowParameter],
        parameters: Dict[str, Any],
        errors: List[ValidationError],
    ):
        """Check for missing required parameters."""
        required_params = {
            name: param for name, param in param_lookup.items() if param.required
        }

        for param_name, param_schema in required_params.items():
            if param_name not in parameters:
                errors.append(
                    ValidationError(
                        field=param_name,
                        message=f"Missing required parameter '{param_name}'",
                        severity=ValidationSeverity.ERROR,
                        expected_type=param_schema.type,
                    )
                )

    def _check_unknown_parameters(
        self,
        param_lookup: Dict[str, FlowParameter],
        parameters: Dict[str, Any],
        issues: List[ValidationError],
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ):
        """Check for unknown parameters."""
        known_params = set(param_lookup.keys())
        provided_params = set(parameters.keys())

        unknown_params = provided_params - known_params
        for param_name in unknown_params:
            issues.append(
                ValidationError(
                    field=param_name,
                    message=f"Unknown parameter '{param_name}'",
                    severity=severity,
                    actual_value=parameters[param_name],
                )
            )

    def _validate_single_parameter(
        self,
        param_schema: FlowParameter,
        value: Any,
        errors: List[ValidationError],
        warnings: List[ValidationError],
    ):
        """Validate a single parameter against its schema."""
        # Extract base type (handle generics like List[str])
        base_type = self._extract_base_type(param_schema.type)

        # Get the appropriate validator
        validator = self.type_validators.get(base_type, self._validate_any)

        # Perform validation
        validation_errors = validator(value, param_schema)

        # Categorize validation issues
        for error in validation_errors:
            if error.severity == ValidationSeverity.ERROR:
                errors.append(
                    ValidationError(
                        field=param_schema.name,
                        message=error.message,
                        severity=error.severity,
                        expected_type=param_schema.type,
                        actual_value=value,
                    )
                )
            else:
                warnings.append(
                    ValidationError(
                        field=param_schema.name,
                        message=error.message,
                        severity=error.severity,
                        expected_type=param_schema.type,
                        actual_value=value,
                    )
                )

    def _extract_base_type(self, type_str: str) -> str:
        """Extract base type from type string (e.g., 'List[str]' -> 'list')."""
        if "[" in type_str:
            return type_str.split("[")[0].lower()
        return type_str.lower()

    def _validate_string(
        self, value: Any, param_schema: FlowParameter
    ) -> List[ValidationError]:
        """Validate string parameter."""
        errors = []

        if not isinstance(value, str):
            errors.append(
                ValidationError(
                    field="",
                    message=f"Expected string, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return errors

        # Check string constraints
        if hasattr(param_schema, "min_length") and len(value) < param_schema.min_length:
            errors.append(
                ValidationError(
                    field="",
                    message=f"String too short (minimum {param_schema.min_length} characters)",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if hasattr(param_schema, "max_length") and len(value) > param_schema.max_length:
            errors.append(
                ValidationError(
                    field="",
                    message=f"String too long (maximum {param_schema.max_length} characters)",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Check for empty strings
        if param_schema.required and not value.strip():
            errors.append(
                ValidationError(
                    field="",
                    message="String cannot be empty",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return errors

    def _validate_integer(
        self, value: Any, param_schema: FlowParameter
    ) -> List[ValidationError]:
        """Validate integer parameter."""
        errors = []

        if not isinstance(value, int) or isinstance(
            value, bool
        ):  # bool is subclass of int
            errors.append(
                ValidationError(
                    field="",
                    message=f"Expected integer, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return errors

        # Check numeric constraints
        if hasattr(param_schema, "min_value") and value < param_schema.min_value:
            errors.append(
                ValidationError(
                    field="",
                    message=f"Value too small (minimum {param_schema.min_value})",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if hasattr(param_schema, "max_value") and value > param_schema.max_value:
            errors.append(
                ValidationError(
                    field="",
                    message=f"Value too large (maximum {param_schema.max_value})",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return errors

    def _validate_float(
        self, value: Any, param_schema: FlowParameter
    ) -> List[ValidationError]:
        """Validate float parameter."""
        errors = []

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(
                ValidationError(
                    field="",
                    message=f"Expected number, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return errors

        # Check numeric constraints
        if hasattr(param_schema, "min_value") and value < param_schema.min_value:
            errors.append(
                ValidationError(
                    field="",
                    message=f"Value too small (minimum {param_schema.min_value})",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if hasattr(param_schema, "max_value") and value > param_schema.max_value:
            errors.append(
                ValidationError(
                    field="",
                    message=f"Value too large (maximum {param_schema.max_value})",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return errors

    def _validate_boolean(
        self, value: Any, param_schema: FlowParameter
    ) -> List[ValidationError]:
        """Validate boolean parameter."""
        errors = []

        if not isinstance(value, bool):
            errors.append(
                ValidationError(
                    field="",
                    message=f"Expected boolean, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return errors

    def _validate_list(
        self, value: Any, param_schema: FlowParameter
    ) -> List[ValidationError]:
        """Validate list parameter."""
        errors = []

        if not isinstance(value, list):
            errors.append(
                ValidationError(
                    field="",
                    message=f"Expected list, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return errors

        # Check list constraints
        if hasattr(param_schema, "min_items") and len(value) < param_schema.min_items:
            errors.append(
                ValidationError(
                    field="",
                    message=f"List too short (minimum {param_schema.min_items} items)",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if hasattr(param_schema, "max_items") and len(value) > param_schema.max_items:
            errors.append(
                ValidationError(
                    field="",
                    message=f"List too long (maximum {param_schema.max_items} items)",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Validate list item type if specified
        if "[" in param_schema.type and "]" in param_schema.type:
            item_type = param_schema.type.split("[")[1].rstrip("]")
            for i, item in enumerate(value):
                item_errors = self._validate_list_item(item, item_type, i)
                errors.extend(item_errors)

        return errors

    def _validate_list_item(
        self, item: Any, expected_type: str, index: int
    ) -> List[ValidationError]:
        """Validate individual list item."""
        errors = []
        base_type = self._extract_base_type(expected_type)

        if base_type == "str" and not isinstance(item, str):
            errors.append(
                ValidationError(
                    field=f"item[{index}]",
                    message=f"Expected string, got {type(item).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
        elif base_type == "int" and (
            not isinstance(item, int) or isinstance(item, bool)
        ):
            errors.append(
                ValidationError(
                    field=f"item[{index}]",
                    message=f"Expected integer, got {type(item).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return errors

    def _validate_dict(
        self, value: Any, param_schema: FlowParameter
    ) -> List[ValidationError]:
        """Validate dictionary parameter."""
        errors = []

        if not isinstance(value, dict):
            errors.append(
                ValidationError(
                    field="",
                    message=f"Expected dictionary, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return errors

        # Basic validation - could be extended with schema validation
        if not value and param_schema.required:
            errors.append(
                ValidationError(
                    field="",
                    message="Dictionary cannot be empty when required",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return errors

    def _validate_any(
        self, value: Any, param_schema: FlowParameter
    ) -> List[ValidationError]:
        """Validate any type (basic validation only)."""
        errors = []

        # Check for None values in required parameters
        if param_schema.required and value is None:
            errors.append(
                ValidationError(
                    field="",
                    message="Required parameter cannot be None",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return errors

    def _sanitize_parameter(self, param_schema: FlowParameter, value: Any) -> Any:
        """
        Sanitize parameter value if needed.

        Args:
            param_schema: Parameter schema
            value: Original value

        Returns:
            Sanitized value or None if no sanitization needed
        """
        # Strip whitespace from strings
        if isinstance(value, str):
            return value.strip()

        # Convert numeric strings to numbers for int/float types
        if param_schema.type in ["int", "float"] and isinstance(value, str):
            try:
                if param_schema.type == "int":
                    return int(value)
                else:
                    return float(value)
            except ValueError:
                return value  # Return original if conversion fails

        return None  # No sanitization needed

    def _fill_default_values(
        self,
        param_lookup: Dict[str, FlowParameter],
        parameters: Dict[str, Any],
        warnings: List[ValidationError],
    ):
        """Fill in default values for missing optional parameters."""
        for param_name, param_schema in param_lookup.items():
            if param_name not in parameters and param_schema.default is not None:
                parameters[param_name] = param_schema.default
                warnings.append(
                    ValidationError(
                        field=param_name,
                        message=f"Using default value for optional parameter '{param_name}'",
                        severity=ValidationSeverity.INFO,
                        actual_value=param_schema.default,
                    )
                )

    def create_json_schema(self, flow_metadata: FlowMetadata) -> Dict[str, Any]:
        """
        Create JSON schema for the flow parameters.

        Args:
            flow_metadata: Flow metadata

        Returns:
            JSON schema dictionary
        """
        properties = {}
        required = []

        for param in flow_metadata.parameters:
            param_schema = self._create_parameter_schema(param)
            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _create_parameter_schema(self, param: FlowParameter) -> Dict[str, Any]:
        """Create JSON schema for a single parameter."""
        base_type = self._extract_base_type(param.type)

        # Map to JSON schema types
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "any": "object",
        }

        schema = {
            "type": type_mapping.get(base_type, "object"),
            "description": param.description or f"Parameter {param.name}",
        }

        # Add default value if present
        if param.default is not None:
            schema["default"] = param.default

        # Handle array item types
        if base_type == "list" and "[" in param.type:
            item_type = param.type.split("[")[1].rstrip("]")
            item_schema_type = type_mapping.get(
                self._extract_base_type(item_type), "object"
            )
            schema["items"] = {"type": item_schema_type}

        return schema
