"""Flow executor for running async flows with parameters in E2B sandbox."""

import json
import asyncio
import ast
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import traceback

from src.core.sandbox import run_python_code
from src.core.flow_discovery import FlowDiscovery, FlowMetadata


@dataclass
class ExecutionResult:
    """Result of flow execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class FlowExecutor:
    """Executes flows in E2B sandbox with parameter injection."""

    def __init__(self):
        self.discovery = FlowDiscovery()
        self._compiled_flows: Dict[str, str] = {}  # Cache of compiled flow code

    def _extract_production_code(self, full_code: str) -> str:
        """
        Extract only the production code (above `if __name__ == "__main__":` block).

        Args:
            full_code: Complete Python source code

        Returns:
            Code only from above the __main__ block
        """
        try:
            tree = ast.parse(full_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # Check if this is a __name__ == '__main__' block
                    if (
                        isinstance(node.test, ast.Compare)
                        and len(node.test.ops) == 1
                        and isinstance(node.test.ops[0], ast.Eq)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__"
                        and len(node.test.comparators) > 0
                        and isinstance(node.test.comparators[0], ast.Constant)
                        and node.test.comparators[0].value == "__main__"
                    ):

                        # Found the __main__ block, extract code before it
                        lines = full_code.split("\n")
                        production_lines = lines[: node.lineno - 1]  # AST is 1-indexed

                        # Remove any trailing empty lines
                        while production_lines and not production_lines[-1].strip():
                            production_lines.pop()

                        return "\n".join(production_lines)

            # No __main__ block found, return full code
            return full_code

        except SyntaxError:
            # If parsing fails, return full code as fallback
            return full_code

    async def compile_flow(
        self, code: str, flow_name: Optional[str] = None
    ) -> Dict[str, FlowMetadata]:
        """
        Compile and discover flows from code.

        Args:
            code: Python source code containing flow functions
            flow_name: Optional specific flow to compile

        Returns:
            Dictionary of discovered flow metadata
        """
        # Extract only production code (above __main__ block)
        production_code = self._extract_production_code(code)

        # Discover flows in the production code
        flows = self.discovery.discover_flows(production_code)

        if not flows:
            raise ValueError("No async flow functions found in the provided code")

        # Store the production code for execution
        self._compiled_flows.update({name: production_code for name in flows.keys()})

        # If specific flow requested, validate it exists
        if flow_name and flow_name not in flows:
            available_flows = list(flows.keys())
            raise ValueError(
                f"Flow '{flow_name}' not found. Available flows: {available_flows}"
            )

        return flows

    async def execute_flow(
        self, flow_name: str, parameters: Dict[str, Any], timeout: int = 300
    ) -> ExecutionResult:
        """
        Execute a specific flow with provided parameters.

        Args:
            flow_name: Name of the flow to execute
            parameters: Parameters to pass to the flow
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with the flow output
        """
        if flow_name not in self._compiled_flows:
            raise ValueError(
                f"Flow '{flow_name}' not compiled. Available flows: {list(self._compiled_flows.keys())}"
            )

        # Get the flow code
        flow_code = self._compiled_flows[flow_name]

        # Create execution script
        execution_script = self._create_execution_script(flow_name, parameters)

        try:
            start_time = asyncio.get_event_loop().time()

            # Create combined code with flow and execution script
            combined_code = f"""
{flow_code}

{execution_script}
"""

            # Execute using the existing sandbox function
            result = await run_python_code(combined_code)

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # Parse the result
            try:
                # Try to parse as JSON first
                if isinstance(result, str):
                    result = result.strip()
                    if result.startswith("{") and result.endswith("}"):
                        output_data = json.loads(result)
                        return ExecutionResult(
                            success=True,
                            data=output_data,
                            execution_time=execution_time,
                            metadata={"flow_name": flow_name, "parameters": parameters},
                        )

                # If not JSON, return as string
                return ExecutionResult(
                    success=True,
                    data=result,
                    execution_time=execution_time,
                    metadata={"flow_name": flow_name, "parameters": parameters},
                )
            except Exception as parse_error:
                return ExecutionResult(
                    success=False,
                    error=f"Failed to parse execution result: {str(parse_error)}",
                    execution_time=execution_time,
                    metadata={
                        "flow_name": flow_name,
                        "parameters": parameters,
                        "raw_result": str(result)[
                            :500
                        ],  # First 500 chars for debugging
                    },
                )

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error=f"Flow execution timed out after {timeout} seconds",
                metadata={
                    "flow_name": flow_name,
                    "parameters": parameters,
                    "timeout": timeout,
                },
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Unexpected error during flow execution: {str(e)}",
                metadata={
                    "flow_name": flow_name,
                    "parameters": parameters,
                    "error_type": type(e).__name__,
                },
            )

    def _create_execution_script(
        self, flow_name: str, parameters: Dict[str, Any]
    ) -> str:
        """
        Create the execution script for running the flow in the sandbox.

        Args:
            flow_name: Name of the flow to execute
            parameters: Parameters to pass to the flow

        Returns:
            Python script as string
        """
        # Import standard libraries
        imports = """
import json
import asyncio
import traceback
from datetime import datetime
"""

        # Create async execution function
        execution_function = f"""
async def execute_flow():
    # Parameters
    parameters = {json.dumps(parameters, indent=2)}

    try:
        # Call the flow function
        result = await {flow_name}(**parameters)

        # Prepare output
        output = {{
            "success": True,
            "data": result,
            "timestamp": datetime.now(datetime.UTC).isoformat(),
            "flow_name": "{flow_name}",
            "parameters": parameters
        }}

    except Exception as e:
        # Handle errors
        output = {{
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now(datetime.UTC).isoformat(),
            "flow_name": "{flow_name}",
            "parameters": parameters
        }}

    # Print result as JSON
    print(json.dumps(output, indent=2))

# Run the execution directly
await execute_flow()
"""

        return imports + execution_function

    async def validate_flow_execution(
        self, flow_name: str, parameters: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Validate parameters against flow signature without executing the flow.

        Args:
            flow_name: Name of the flow
            parameters: Parameters to validate

        Returns:
            ExecutionResult indicating validation success/failure
        """
        if flow_name not in self._compiled_flows:
            return ExecutionResult(
                success=False, error=f"Flow '{flow_name}' not compiled"
            )

        # Get flow metadata
        flows = self.discovery.discover_flows(self._compiled_flows[flow_name])
        if flow_name not in flows:
            return ExecutionResult(
                success=False, error=f"Flow '{flow_name}' not found in compiled code"
            )

        flow_metadata = flows[flow_name]

        # Validate parameters
        validation_errors = []

        # Check for missing required parameters
        required_params = {p.name for p in flow_metadata.parameters if p.required}
        provided_params = set(parameters.keys())

        missing_params = required_params - provided_params
        if missing_params:
            validation_errors.append(f"Missing required parameters: {missing_params}")

        # Check for unknown parameters
        known_params = {p.name for p in flow_metadata.parameters}
        unknown_params = provided_params - known_params
        if unknown_params:
            validation_errors.append(f"Unknown parameters: {unknown_params}")

        # Validate parameter types (basic validation)
        for param in flow_metadata.parameters:
            if param.name in parameters:
                value = parameters[param.name]
                type_error = self._validate_parameter_type(
                    param.name, param.type, value
                )
                if type_error:
                    validation_errors.append(type_error)

        if validation_errors:
            return ExecutionResult(
                success=False,
                error=f"Parameter validation failed: {'; '.join(validation_errors)}",
                metadata={
                    "flow_name": flow_name,
                    "validation_errors": validation_errors,
                },
            )

        return ExecutionResult(
            success=True,
            data={"valid": True},
            metadata={"flow_name": flow_name, "validated_parameters": parameters},
        )

    def _validate_parameter_type(
        self, param_name: str, expected_type: str, value: Any
    ) -> Optional[str]:
        """
        Validate a single parameter type.

        Args:
            param_name: Name of the parameter
            expected_type: Expected type string
            value: Actual value

        Returns:
            Error message if validation fails, None if valid
        """
        # Basic type validation
        type_mapping = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        # Handle generic types
        if "[" in expected_type:
            base_type = expected_type.split("[")[0]
            expected_python_type = type_mapping.get(base_type)
        else:
            expected_python_type = type_mapping.get(expected_type)

        if expected_python_type and not isinstance(value, expected_python_type):
            return f"Parameter '{param_name}' should be of type {expected_type}, got {type(value).__name__}"

        return None

    async def get_available_flows(self) -> List[Dict[str, Any]]:
        """Get list of all available compiled flows."""
        flows_list = []

        for flow_name in self._compiled_flows.keys():
            flows = self.discovery.discover_flows(self._compiled_flows[flow_name])
            if flow_name in flows:
                flow_metadata = flows[flow_name]
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

    async def get_flow_schema(self, flow_name: str) -> Optional[Dict[str, Any]]:
        """Get JSON schema for a specific flow."""
        if flow_name not in self._compiled_flows:
            return None

        # Rediscover flows to get latest metadata
        flows = self.discovery.discover_flows(self._compiled_flows[flow_name])
        if flow_name not in flows:
            return None

        return self.discovery.get_flow_schema(flow_name)

    def clear_compiled_flows(self):
        """Clear all compiled flows."""
        self._compiled_flows.clear()

    def remove_flow(self, flow_name: str):
        """Remove a specific compiled flow."""
        if flow_name in self._compiled_flows:
            del self._compiled_flows[flow_name]
