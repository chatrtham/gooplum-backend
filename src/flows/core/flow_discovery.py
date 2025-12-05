"""Flow discovery system for parsing generated code and extracting flow signatures."""

import ast
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class FlowParameter:
    """Represents a parameter in a flow function."""

    name: str
    type: str
    default: Optional[str] = None
    required: bool = True
    description: Optional[str] = None


@dataclass
class FlowMetadata:
    """Metadata for a discovered flow."""

    id: str
    name: str
    description: str
    parameters: List[FlowParameter]
    return_type: str
    docstring: Optional[str] = None
    source_code: Optional[str] = None
    explanation: Optional[str] = None
    created_at: Optional[datetime] = None


class FlowDiscovery:
    """Discovers and parses flow functions from generated code."""

    def __init__(self):
        self.flows: Dict[str, FlowMetadata] = {}

    def discover_flows(self, code: str) -> Dict[str, FlowMetadata]:
        """
        Discover all async flow functions in the given code.

        Args:
            code: Python source code to parse

        Returns:
            Dictionary mapping flow names to their metadata
        """
        self.flows = {}

        # Parse the AST
        tree = ast.parse(code)

        # Find all async function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                flow_metadata = self._extract_flow_metadata(node, code)
                if flow_metadata:
                    self.flows[flow_metadata.name] = flow_metadata

        return self.flows

    def _extract_flow_metadata(
        self, node: ast.AsyncFunctionDef, source_code: str
    ) -> Optional[FlowMetadata]:
        """Extract metadata from an async function definition."""
        try:
            # Extract function name
            name = node.name

            # Skip private functions (starting with _)
            if name.startswith("_"):
                return None

            # Extract docstring
            docstring = ast.get_docstring(node) or ""

            # Extract description from docstring (first line)
            description = (
                docstring.split("\n")[0].strip() if docstring else f"Flow: {name}"
            )

            # Extract parameters
            parameters = self._extract_parameters(node)

            # Extract return type
            return_type = self._get_return_type(node)

            # Extract source code
            source_code = self._extract_source_code(node, source_code)

            return FlowMetadata(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                source_code=source_code,
            )

        except Exception as e:
            print(f"Error extracting metadata for function {node.name}: {e}")
            return None

    def _extract_parameters(self, node: ast.AsyncFunctionDef) -> List[FlowParameter]:
        """Extract parameter information from function definition."""
        parameters = []

        # Skip self parameter for methods
        skip_first = node.args.args and node.args.args[0].arg == "self"

        args_to_process = node.args.args[1:] if skip_first else node.args.args

        for i, arg in enumerate(args_to_process):
            param_name = arg.arg

            # Get type annotation
            param_type = "any"
            if arg.annotation:
                param_type = self._get_type_string(arg.annotation)

            # Check if parameter has a default value
            default_idx = i - (len(node.args.args) - len(node.args.defaults))
            has_default = default_idx >= 0
            default_value = None

            if has_default and node.args.defaults:
                default_node = node.args.defaults[default_idx]
                default_value = self._get_default_value(default_node)

            # Check if parameter is required
            required = not has_default

            # Extract parameter description from docstring
            param_description = self._extract_param_description(node, param_name)

            parameters.append(
                FlowParameter(
                    name=param_name,
                    type=param_type,
                    default=default_value,
                    required=required,
                    description=param_description,
                )
            )

        return parameters

    def _get_type_string(self, type_node) -> str:
        """Convert AST type annotation to string."""
        try:
            if isinstance(type_node, ast.Name):
                return type_node.id
            elif isinstance(type_node, ast.Attribute):
                return f"{type_node.value.id}.{type_node.attr}"
            elif isinstance(type_node, ast.Subscript):
                base = self._get_type_string(type_node.value)
                subscript = self._get_type_string(type_node.slice)
                return f"{base}[{subscript}]"
            elif isinstance(type_node, ast.Constant):
                return str(type_node.value)
            else:
                return "any"
        except:
            return "any"

    def _get_default_value(self, default_node) -> Optional[str]:
        """Extract default value from AST node."""
        try:
            if isinstance(default_node, ast.Constant):
                return repr(default_node.value)
            elif isinstance(default_node, ast.Str):  # Python < 3.8
                return repr(default_node.s)
            elif isinstance(default_node, ast.Num):  # Python < 3.8
                return repr(default_node.n)
            elif isinstance(default_node, ast.NameConstant):  # Python < 3.8
                return repr(default_node.value)
            elif isinstance(default_node, ast.List):
                elements = [self._get_default_value(elt) for elt in default_node.elts]
                return f"[{', '.join(filter(None, elements))}]"
            elif isinstance(default_node, ast.Dict):
                items = []
                for k, v in zip(default_node.keys, default_node.values):
                    key = self._get_default_value(k)
                    value = self._get_default_value(v)
                    if key and value:
                        items.append(f"{key}: {value}")
                return f"{{{', '.join(items)}}}"
            else:
                return None
        except:
            return None

    def _extract_param_description(
        self, node: ast.AsyncFunctionDef, param_name: str
    ) -> Optional[str]:
        """Extract parameter description from docstring."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return None

        # Look for "Args:" section
        args_match = re.search(r"Args:\s*\n((?:\s*.*:\s*.*\n?)*)", docstring)
        if not args_match:
            return None

        args_section = args_match.group(1)

        # Look for specific parameter
        param_pattern = rf"{param_name}:\s*(.*?)(?=\n\s*\w+:|$)"
        param_match = re.search(param_pattern, args_section, re.DOTALL)

        if param_match:
            return param_match.group(1).strip()

        return None

    def _get_return_type(self, node: ast.AsyncFunctionDef) -> str:
        """Extract return type from function definition."""
        if node.returns:
            return self._get_type_string(node.returns)
        return "any"

    def _extract_source_code(self, node: ast.AsyncFunctionDef, full_source: str) -> str:
        """Extract the source code for a specific function."""
        try:
            lines = full_source.split("\n")
            start_line = node.lineno - 1  # AST is 1-indexed, we need 0-indexed

            # Find the end line by looking for dedentation or end of function
            end_line = len(lines)

            # Simple heuristic: find the next line at the same or lower indentation level
            if hasattr(node, "end_lineno") and node.end_lineno:
                end_line = node.end_lineno
            else:
                # Fallback: look for dedentation
                current_indent = len(lines[start_line]) - len(
                    lines[start_line].lstrip()
                )
                for i in range(start_line + 1, len(lines)):
                    line = lines[i]
                    if (
                        line.strip()
                        and len(line) - len(line.lstrip()) <= current_indent
                    ):
                        end_line = i
                        break

            # Extract the function source
            function_lines = lines[start_line:end_line]
            return "\n".join(function_lines)

        except Exception as e:
            print(f"Error extracting source code: {e}")
            return f"# Could not extract source for {node.name}"

    def get_flow_schema(self, flow_name: str) -> Optional[Dict[str, Any]]:
        """
        Get JSON schema for a specific flow.

        Args:
            flow_name: Name of the flow

        Returns:
            JSON schema for the flow parameters
        """
        if flow_name not in self.flows:
            return None

        flow = self.flows[flow_name]

        properties = {}
        required = []

        for param in flow.parameters:
            param_schema = {
                "type": self._map_type_to_json_schema(param.type),
                "description": param.description or f"Parameter {param.name}",
            }

            if param.default is not None:
                param_schema["default"] = param.default

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "name": flow.name,
            "description": flow.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "return_type": flow.return_type,
        }

    def _map_type_to_json_schema(self, python_type: str) -> str:
        """Map Python type to JSON schema type."""
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "any": "object",
        }

        # Handle generic types like List[str], Dict[str, int]
        if "[" in python_type:
            base_type = python_type.split("[")[0]
            return type_mapping.get(base_type, "object")

        return type_mapping.get(python_type, "object")

    def list_flows(self) -> List[Dict[str, Any]]:
        """Get a list of all discovered flows with basic info."""
        flows_list = []

        for flow_name, flow_metadata in self.flows.items():
            flows_list.append(
                {
                    "name": flow_name,
                    "description": flow_metadata.description,
                    "parameter_count": len(flow_metadata.parameters),
                    "required_parameters": len(
                        [p for p in flow_metadata.parameters if p.required]
                    ),
                    "return_type": flow_metadata.return_type,
                }
            )

        return flows_list
