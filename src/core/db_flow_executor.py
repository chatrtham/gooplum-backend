"""Simplified database-backed Flow executor without cache."""

import json
import asyncio
import ast
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from src.core.sandbox import run_python_code, run_python_code_with_streaming
from src.core.flow_discovery import FlowDiscovery, FlowMetadata
from src.core.flow_explainer import FlowExplainer
from src.db.supabase_client import get_flow_db, FlowRecord


@dataclass
class StreamResult:
    """Single streamed result from flow execution."""

    input: Any
    status: str  # "success" or "failed"
    message: str
    timestamp: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of flow execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    streams: Optional[List[StreamResult]] = None


class DBFlowExecutor:
    """Simplified database-backed flow executor that always reads from database."""

    def __init__(self):
        """Initialize the database-backed flow executor."""
        self.discovery = FlowDiscovery()
        self.explainer = FlowExplainer()
        self.db = get_flow_db()

    async def _get_flow_by_id(self, flow_id: str) -> Optional[FlowRecord]:
        """Get flow record by ID from database."""
        try:
            return await self.db.get_flow(UUID(flow_id))
        except Exception:
            return None

    async def _get_flow_by_name(self, flow_name: str) -> Optional[FlowRecord]:
        """Get flow record by name from database."""
        try:
            return await self.db.get_flow_by_name(flow_name)
        except Exception:
            return None

    async def _get_flow_metadata_by_id(self, flow_id: str) -> Optional[FlowMetadata]:
        """Get flow metadata by ID from database."""
        flow_record = await self._get_flow_by_id(flow_id)
        if not flow_record:
            return None

        # Get parameters and create metadata
        parameters = await self.db.get_flow_parameters(flow_record.id)
        return self.db.flow_record_to_metadata(flow_record, parameters)

    def _extract_production_code(self, full_code: str) -> str:
        """Extract only the production code (above `if __name__ == "__main__":` block)."""
        try:
            tree = ast.parse(full_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
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
                        lines = full_code.split("\n")
                        production_lines = lines[: node.lineno - 1]
                        while production_lines and not production_lines[-1].strip():
                            production_lines.pop()
                        return "\n".join(production_lines)
            return full_code
        except SyntaxError:
            return full_code

    async def compile_flow(
        self, code: str, flow_name: Optional[str] = None
    ) -> Dict[str, FlowMetadata]:
        """Compile and discover flows from code, then store in database."""
        production_code = self._extract_production_code(code)
        flows = self.discovery.discover_flows(production_code)

        if not flows:
            raise ValueError("No async flow functions found in the provided code")

        if flow_name and flow_name not in flows:
            available_flows = list(flows.keys())
            raise ValueError(
                f"Flow '{flow_name}' not found. Available flows: {available_flows}"
            )

        # Generate explanations and add timestamps
        current_time = datetime.now(timezone.utc)
        created_flows = {}

        for name, flow_metadata in flows.items():
            flow_metadata.created_at = current_time

            # Generate explanation
            try:
                explanation = await self.explainer.generate_explanation(flow_metadata)
                flow_metadata.explanation = explanation
            except Exception as e:
                print(f"Warning: Failed to generate explanation for flow '{name}': {e}")
                flow_metadata.explanation = None

            # Store in database
            try:
                await self.db.create_flow(flow_metadata, production_code)
                created_flows[name] = flow_metadata
            except Exception as e:
                raise Exception(f"Failed to store flow '{name}' in database: {e}")

        return created_flows

    async def execute_flow(
        self, flow_name: str, parameters: Dict[str, Any], timeout: int = 300
    ) -> ExecutionResult:
        """Execute a specific flow with provided parameters."""
        # Get flow from database
        flow_record = await self._get_flow_by_name(flow_name)
        if not flow_record:
            return ExecutionResult(
                success=False,
                error=f"Flow '{flow_name}' not found in database",
                metadata={},
            )

        flow_code = flow_record.source_code

        # Convert parameter types
        converted_parameters = await self._convert_parameter_types(
            flow_name, parameters
        )

        # Create execution script
        execution_script = self._create_execution_script(
            flow_name, converted_parameters
        )

        # 1. Create initial execution record with pending status
        from datetime import datetime

        execution_record = None
        try:
            execution_record = await self.db.create_execution_record(
                flow_id=flow_record.id,
                parameters=parameters,
                success=False,  # Will be updated later
                status="pending",
            )
        except Exception as e:
            # Continue execution even if record creation fails
            print(f"Warning: Failed to create execution record: {e}")

        try:
            start_time = asyncio.get_event_loop().time()

            # 2. Update status to running when execution begins
            if execution_record:
                await self.db.update_execution_status(
                    execution_record.id, "running", started_at=datetime.now()
                )

            combined_code = f"{flow_code}\n\n{execution_script}"
            result = await run_python_code(combined_code)
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # Parse result (similar to original executor)
            try:
                if (
                    hasattr(result, "results")
                    and result.results
                    and len(result.results) > 0
                ):
                    first_result = result.results[0]
                    if hasattr(first_result, "__dict__"):
                        result_dict = vars(first_result)
                        if "json" in result_dict and result_dict["json"]:
                            output_data = result_dict["json"]
                        else:
                            execution_result = ExecutionResult(
                                success=False,
                                error="No JSON data found in E2B result",
                                execution_time=execution_time,
                                metadata={"available_keys": list(result_dict.keys())},
                            )
                    else:
                        output_data = first_result

                    execution_result = ExecutionResult(
                        success=output_data.get("success", False),
                        data=output_data.get("data"),
                        error=output_data.get("error"),
                        execution_time=execution_time,
                        metadata=output_data.get("metadata", {}),
                    )
                else:
                    execution_result = ExecutionResult(
                        success=False,
                        error="No results returned from flow execution",
                        execution_time=execution_time,
                        metadata={"flow_name": flow_name, "parameters": parameters},
                    )
            except Exception as parse_error:
                execution_result = ExecutionResult(
                    success=False,
                    error=f"Failed to parse execution result: {str(parse_error)}",
                    execution_time=execution_time,
                    metadata={"flow_name": flow_name, "parameters": parameters},
                )

            # 3. Update execution record with final results
            if execution_record:
                final_status = "completed" if execution_result.success else "failed"
                await self.db.update_execution_record(
                    execution_record.id,
                    {
                        "status": final_status,
                        "success": execution_result.success,
                        "result": (
                            execution_result.data if execution_result.success else None
                        ),
                        "error": (
                            execution_result.error
                            if not execution_result.success
                            else None
                        ),
                        "completed_at": datetime.now(),
                        "execution_time_ms": (
                            int(execution_time * 1000) if execution_time else None
                        ),
                        "metadata": execution_result.metadata,
                    },
                )
            else:
                # Fallback: create execution record after completion
                await self.db.create_execution_record(
                    flow_id=flow_record.id,
                    parameters=parameters,
                    success=execution_result.success,
                    result=execution_result.data if execution_result.success else None,
                    error=(
                        execution_result.error if not execution_result.success else None
                    ),
                    execution_time_ms=(
                        int(execution_time * 1000) if execution_time else None
                    ),
                    metadata=execution_result.metadata,
                    status="completed" if execution_result.success else "failed",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )

            return execution_result

        except asyncio.TimeoutError:
            error_result = ExecutionResult(
                success=False,
                error=f"Flow execution timed out after {timeout} seconds",
                metadata={
                    "flow_name": flow_name,
                    "parameters": parameters,
                    "timeout": timeout,
                },
            )
            # Update execution record with timeout status
            if execution_record:
                await self.db.update_execution_record(
                    execution_record.id,
                    {
                        "status": "failed",
                        "success": False,
                        "error": error_result.error,
                        "completed_at": datetime.now(),
                        "execution_time_ms": timeout * 1000,
                        "metadata": error_result.metadata,
                    },
                )
            else:
                # Fallback: create execution record after timeout
                await self.db.create_execution_record(
                    flow_id=flow_record.id,
                    parameters=parameters,
                    success=False,
                    error=error_result.error,
                    execution_time_ms=timeout * 1000,
                    metadata=error_result.metadata,
                    status="failed",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )
            return error_result

        except Exception as e:
            error_result = ExecutionResult(
                success=False,
                error=f"Unexpected error during flow execution: {str(e)}",
                metadata={"flow_name": flow_name, "parameters": parameters},
            )
            # Update execution record with exception status
            if execution_record:
                await self.db.update_execution_record(
                    execution_record.id,
                    {
                        "status": "failed",
                        "success": False,
                        "error": error_result.error,
                        "completed_at": datetime.now(),
                        "metadata": error_result.metadata,
                    },
                )
            else:
                # Fallback: create execution record after exception
                await self.db.create_execution_record(
                    flow_id=flow_record.id,
                    parameters=parameters,
                    success=False,
                    error=error_result.error,
                    metadata=error_result.metadata,
                    status="failed",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )
            return error_result

    async def execute_flow_by_id(
        self, flow_id: str, parameters: Dict[str, Any], timeout: int = 300
    ) -> ExecutionResult:
        """Execute a flow by ID."""
        flow_record = await self._get_flow_by_id(flow_id)
        if not flow_record:
            return ExecutionResult(
                success=False,
                error=f"Flow with ID '{flow_id}' not found",
                metadata={},
            )

        return await self.execute_flow(flow_record.name, parameters, timeout)

    async def get_available_flows(self) -> List[Dict[str, Any]]:
        """Get list of all available flows from database."""
        flows = await self.db.list_flows()
        flows_list = []

        for flow_record in flows:
            parameters = await self.db.get_flow_parameters(flow_record.id)
            flow_metadata = self.db.flow_record_to_metadata(flow_record, parameters)

            flows_list.append(
                {
                    "id": flow_metadata.id,
                    "name": flow_record.name,
                    "description": flow_metadata.description,
                    "parameter_count": len(flow_metadata.parameters),
                    "required_parameters": len(
                        [p for p in flow_metadata.parameters if p.required]
                    ),
                    "return_type": flow_metadata.return_type,
                    "created_at": (
                        flow_metadata.created_at.isoformat()
                        if flow_metadata.created_at
                        else None
                    ),
                }
            )

        return flows_list

    async def get_flow_schema(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get JSON schema for a specific flow by ID."""
        flow_metadata = await self._get_flow_metadata_by_id(flow_id)
        if not flow_metadata:
            return None

        # Build schema directly from flow metadata instead of using discovery
        schema = {
            "name": flow_metadata.name,
            "description": flow_metadata.description,
            "parameters": {},
            "return_type": flow_metadata.return_type,
            "docstring": flow_metadata.docstring,
            "explanation": flow_metadata.explanation,
        }

        # Add parameters
        for param in flow_metadata.parameters:
            schema["parameters"][param.name] = {
                "type": param.type,
                "description": param.description,
                "required": param.required,
                "default": param.default,
            }

        # Add database fields
        schema["id"] = flow_metadata.id
        schema["created_at"] = (
            flow_metadata.created_at.isoformat() if flow_metadata.created_at else None
        )

        return schema

    async def get_flow_schema_by_id(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get JSON schema for a specific flow by ID (alias for get_flow_schema)."""
        return await self.get_flow_schema(flow_id)

    async def validate_flow_execution(
        self, flow_name: str, parameters: Dict[str, Any]
    ) -> ExecutionResult:
        """Validate parameters against flow schema without executing."""
        flow_record = await self._get_flow_by_name(flow_name)
        if not flow_record:
            return ExecutionResult(success=False, error=f"Flow '{flow_name}' not found")

        # Get metadata and validate
        parameters_list = await self.db.get_flow_parameters(flow_record.id)
        flow_metadata = self.db.flow_record_to_metadata(flow_record, parameters_list)

        required_params = {p.name for p in flow_metadata.parameters if p.required}
        provided_params = set(parameters.keys())

        missing_params = required_params - provided_params
        if missing_params:
            return ExecutionResult(
                success=False,
                error=f"Missing required parameters: {missing_params}",
            )

        return ExecutionResult(success=True, data={"valid": True})

    async def validate_flow_execution_by_id(
        self, flow_id: str, parameters: Dict[str, Any]
    ) -> ExecutionResult:
        """Validate parameters against flow schema without executing."""
        flow_metadata = await self._get_flow_metadata_by_id(flow_id)
        if not flow_metadata:
            return ExecutionResult(
                success=False, error=f"Flow with ID '{flow_id}' not found"
            )

        # Basic validation
        required_params = {p.name for p in flow_metadata.parameters if p.required}
        provided_params = set(parameters.keys())

        missing_params = required_params - provided_params
        if missing_params:
            return ExecutionResult(
                success=False,
                error=f"Missing required parameters: {missing_params}",
            )

        return ExecutionResult(success=True, data={"valid": True})

    async def remove_flow_by_id(self, flow_id: str) -> bool:
        """Remove a flow by ID."""
        try:
            return await self.db.delete_flow(UUID(flow_id))
        except Exception:
            return False

    async def remove_flow(self, flow_name: str) -> bool:
        """Remove a flow by name."""
        flow_record = await self._get_flow_by_name(flow_name)
        if not flow_record:
            raise ValueError(f"Flow '{flow_name}' not found")

        return await self.db.delete_flow(flow_record.id)

    async def execute_flow_with_streaming(
        self,
        flow_name: str,
        parameters: Dict[str, Any],
        timeout: int = 300,
        on_stream: Optional[callable] = None,
    ) -> ExecutionResult:
        """Execute a flow with streaming (simplified version - just calls execute_flow)."""
        # For now, just use regular execution
        # TODO: Implement proper streaming if needed
        return await self.execute_flow(flow_name, parameters, timeout)

    async def execute_flow_by_id_with_streaming(
        self,
        flow_id: str,
        parameters: Dict[str, Any],
        timeout: int = 300,
        on_stream: Optional[callable] = None,
    ) -> ExecutionResult:
        """Execute a flow by ID with streaming (simplified version)."""
        flow_record = await self._get_flow_by_id(flow_id)
        if not flow_record:
            return ExecutionResult(
                success=False,
                error=f"Flow with ID '{flow_id}' not found",
                streams=[],
                metadata={},
            )

        return await self.execute_flow_with_streaming(
            flow_record.name, parameters, timeout, on_stream
        )

    def _get_flow_name_by_id(self, flow_id: str) -> Optional[str]:
        """Get flow name by ID (synchronous version for API compatibility)."""
        # This is a sync version for API compatibility - the API should use the async version
        # but for now we'll make it work
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use asyncio.run
                # For now, return None and let the async version handle it
                return None
            else:
                return asyncio.run(self._get_flow_name_by_id_async(flow_id))
        except:
            return None

    async def _get_flow_name_by_id_async(self, flow_id: str) -> Optional[str]:
        """Async version of get flow name by ID."""
        flow_record = await self._get_flow_by_id(flow_id)
        return flow_record.name if flow_record else None

    def get_enriched_flow_metadata(self, flow_name: str) -> Optional[FlowMetadata]:
        """Get enriched flow metadata (sync version for API compatibility)."""
        # This is a sync wrapper around async operations
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, this won't work
                # For now, return None - the API should be updated to use async
                return None
            else:
                return asyncio.run(self._get_flow_metadata_by_name_async(flow_name))
        except:
            return None

    async def _get_flow_metadata_by_name_async(
        self, flow_name: str
    ) -> Optional[FlowMetadata]:
        """Async version to get flow metadata by name."""
        flow_record = await self._get_flow_by_name(flow_name)
        if not flow_record:
            return None

        parameters = await self.db.get_flow_parameters(flow_record.id)
        return self.db.flow_record_to_metadata(flow_record, parameters)

    async def get_enriched_flow_metadata_by_id(
        self, flow_id: str
    ) -> Optional[FlowMetadata]:
        """Get enriched flow metadata by ID."""
        return await self._get_flow_metadata_by_id(flow_id)

    async def clear_all_flows(self) -> int:
        """Clear all flows from database."""
        flows = await self.db.list_flows()
        count = len(flows)

        for flow in flows:
            await self.db.delete_flow(flow.id)

        return count

    def _create_execution_script(
        self, flow_name: str, parameters: Dict[str, Any]
    ) -> str:
        """Create the execution script for running the flow in the sandbox."""
        imports = """
import json
import traceback
import sys
from datetime import datetime, timezone
"""
        execution_function = f"""
async def execute_flow():
    parameters = {json.dumps(parameters, indent=2)}

    import io
    original_stdout = sys.stdout
    captured_output = io.StringIO()

    try:
        sys.stdout = captured_output
        result = await {flow_name}(**parameters)
        captured_logs = str(captured_output.getvalue())

        output = {{
            "success": True,
            "data": result,
            "logs": {{"stdout": captured_logs, "stderr": ""}},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flow_name": "{flow_name}",
            "parameters": parameters
        }}
    except Exception as e:
        captured_logs = str(captured_output.getvalue())
        output = {{
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "logs": {{"stdout": captured_logs, "stderr": ""}},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flow_name": "{flow_name}",
            "parameters": parameters
        }}
    finally:
        sys.stdout = original_stdout

    return output

await execute_flow()
"""
        return imports + execution_function

    async def _convert_parameter_types(
        self, flow_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert parameters to their expected types."""
        # For now, just return parameters as-is
        # Type conversion could be added here if needed
        return parameters
