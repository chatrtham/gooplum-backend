"""Simplified database-backed Flow executor without cache."""

import json
import asyncio
import ast
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from .sandbox import run_python_code, run_python_code_with_streaming
from .flow_discovery import FlowDiscovery, FlowMetadata
from .flow_explainer import FlowExplainer
from .supabase_client import get_flow_db, FlowRecord


@dataclass
class StreamResult:
    """Single streamed result from flow execution."""

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

    async def compile_flow(
        self, code: str, flow_name: Optional[str] = None
    ) -> Dict[str, FlowMetadata]:
        """Compile and discover flows from code, then store in database."""
        production_code = self._extract_production_code(code)
        flows = self.discovery.discover_flows(production_code)

        if not flows:
            raise ValueError("No async flow functions found in the provided code")

        if len(flows) > 1:
            raise ValueError(
                f"Multiple flows found: {list(flows.keys())}. Only one flow per file is allowed."
            )

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

    async def _execute_flow_impl(
        self,
        flow_record: FlowRecord,
        parameters: Dict[str, Any],
        timeout: int = 300,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[ExecutionResult, UUID]:
        """Internal implementation for executing a flow.

        Args:
            flow_record: The flow record to execute
            parameters: Parameters to pass to the flow
            timeout: Execution timeout in seconds
            metadata: Optional metadata to store with the run record

        Returns:
            Tuple of (ExecutionResult, run_id)
        """
        flow_code = flow_record.source_code

        # Convert parameter types
        converted_parameters = await self._convert_parameter_types(
            flow_record.name, parameters
        )

        # Create execution script
        execution_script = self._create_execution_script(
            flow_record.name, converted_parameters
        )

        # Merge default metadata with provided metadata
        run_metadata = {"flow_name": flow_record.name}
        if metadata:
            run_metadata.update(metadata)

        try:
            # Create run record
            flow_run = await self.db.create_flow_run(
                flow_id=flow_record.id,
                parameters=parameters,
                metadata=run_metadata,
            )
            flow_run_id = flow_run.id

            start_time = asyncio.get_event_loop().time()
            combined_code = f"{flow_code}\n\n{execution_script}"
            result = await run_python_code(combined_code)
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # Parse result
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
                        metadata={
                            "flow_name": flow_record.name,
                            "parameters": parameters,
                        },
                    )
            except Exception as parse_error:
                execution_result = ExecutionResult(
                    success=False,
                    error=f"Failed to parse execution result: {str(parse_error)}",
                    execution_time=execution_time,
                    metadata={"flow_name": flow_record.name, "parameters": parameters},
                )

            # Update execution record (COMPLETED/FAILED)
            await self.db.update_flow_run(
                run_id=flow_run_id,
                status="COMPLETED" if execution_result.success else "FAILED",
                result=execution_result.data if execution_result.success else None,
                error=execution_result.error if not execution_result.success else None,
                execution_time_ms=(
                    int(execution_time * 1000) if execution_time else None
                ),
            )

            return execution_result, flow_run_id

        except asyncio.TimeoutError:
            error_result = ExecutionResult(
                success=False,
                error=f"Flow execution timed out after {timeout} seconds",
                metadata={
                    "flow_name": flow_record.name,
                    "parameters": parameters,
                    "timeout": timeout,
                },
            )
            # Update execution record (FAILED)
            if "flow_run_id" in locals():
                await self.db.update_flow_run(
                    run_id=flow_run_id,
                    status="FAILED",
                    error=error_result.error,
                    execution_time_ms=timeout * 1000,
                )
            return error_result, flow_run_id if "flow_run_id" in locals() else None

        except Exception as e:
            error_result = ExecutionResult(
                success=False,
                error=f"Unexpected error during flow execution: {str(e)}",
                metadata={"flow_name": flow_record.name, "parameters": parameters},
            )
            # Update execution record (FAILED)
            if "flow_run_id" in locals():
                await self.db.update_flow_run(
                    run_id=flow_run_id,
                    status="FAILED",
                    error=error_result.error,
                )
            return error_result, flow_run_id if "flow_run_id" in locals() else None

    async def execute_flow_by_id(
        self, flow_id: str, parameters: Dict[str, Any], timeout: int = 300
    ) -> tuple[ExecutionResult, UUID]:
        """Execute a flow by ID.

        Returns:
            Tuple of (ExecutionResult, run_id)
        """
        flow_record = await self._get_flow_by_id(flow_id)
        if not flow_record:
            return (
                ExecutionResult(
                    success=False,
                    error=f"Flow with ID '{flow_id}' not found",
                    metadata={},
                ),
                None,
            )

        # Check if flow is ready
        if flow_record.status != "ready":
            return (
                ExecutionResult(
                    success=False,
                    error=f"Flow '{flow_record.name}' is not activated. Current status: {flow_record.status}",
                    metadata={},
                ),
                None,
            )

        return await self._execute_flow_impl(flow_record, parameters, timeout)

    async def get_flow_run_status(self, run_id: UUID) -> Optional[Dict[str, Any]]:
        """Get the status and result of a flow run.

        Returns:
            Dict with run status, result, error, etc. or None if not found
        """
        run = await self.db.get_flow_run(run_id)
        if not run:
            return None

        return {
            "run_id": str(run.id),
            "status": run.status,
            "result": run.result,
            "error": run.error,
            "execution_time_ms": run.execution_time_ms,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }

    async def get_available_flows(
        self, limit: int = 12, offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get paginated list of available flows from database.

        Args:
            limit: Maximum number of flows to return
            offset: Number of flows to skip

        Returns:
            Tuple of (flows_list, total_count)
        """
        flows, total = await self.db.list_flows(limit=limit, offset=offset)
        flows_list = [
            {
                "id": str(flow_record.id),
                "name": flow_record.name,
                "description": flow_record.description or "",
                "created_at": (
                    flow_record.created_at.isoformat()
                    if flow_record.created_at
                    else None
                ),
            }
            for flow_record in flows
        ]
        return flows_list, total

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

    async def _execute_flow_with_streaming_impl(
        self,
        flow_record: FlowRecord,
        parameters: Dict[str, Any],
        timeout: int = 1800,
        on_stream: Optional[callable] = None,
        run_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Internal implementation for executing a flow with streaming.

        Args:
            flow_record: The flow record to execute
            parameters: Parameters to pass to the flow
            timeout: Execution timeout in seconds
            on_stream: Callback for stream events
            run_id: Optional pre-generated run ID to use instead of creating new one
        """

        flow_code = flow_record.source_code

        # Create execution record or use provided run_id
        try:
            if run_id:
                flow_run = await self.db.create_flow_run(
                    flow_id=flow_record.id,
                    parameters=parameters,
                    metadata={"flow_name": flow_record.name},
                    run_id=UUID(run_id),
                )
            else:
                flow_run = await self.db.create_flow_run(
                    flow_id=flow_record.id,
                    parameters=parameters,
                    metadata={"flow_name": flow_record.name},
                )
        except Exception as e:
            print(f"Failed to create flow run: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"Failed to create flow run: {str(e)}",
                metadata={"flow_name": flow_record.name},
            )

        # Convert parameter types
        converted_parameters = await self._convert_parameter_types(
            flow_record.name, parameters
        )

        # Create execution script
        execution_script = self._create_execution_script(
            flow_record.name, converted_parameters
        )

        collected_streams = []

        async def handle_stream_output(output_line: Any):
            """Handle streaming output from sandbox."""
            # Convert OutputMessage to string if needed
            if hasattr(output_line, "line"):
                output_line = output_line.line

            if not isinstance(output_line, str):
                return

            if output_line.startswith("STREAM_RESULT:"):
                try:
                    json_str = output_line.replace("STREAM_RESULT:", "", 1).strip()
                    stream_data = json.loads(json_str)

                    # Create stream result object
                    stream_result = StreamResult(
                        status=stream_data.get("status", "unknown"),
                        message=stream_data.get("message", ""),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                    collected_streams.append(stream_result)

                    # Call callback if provided
                    if on_stream:
                        if asyncio.iscoroutinefunction(on_stream):
                            await on_stream(stream_result)
                        else:
                            on_stream(stream_result)

                    # Add to database
                    await self.db.add_stream_event(
                        run_id=flow_run.id,
                        event_type=stream_data.get("type", "item"),
                        payload=stream_data,
                    )

                except Exception as e:
                    print(f"Error parsing stream result: {e}")

        try:
            start_time = asyncio.get_event_loop().time()
            combined_code = f"{flow_code}\n\n{execution_script}"

            result = await run_python_code_with_streaming(
                combined_code, on_stdout=handle_stream_output, timeout=timeout
            )

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            # Parse result
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
                                streams=collected_streams,
                            )
                    else:
                        output_data = first_result

                    execution_result = ExecutionResult(
                        success=output_data.get("success", False),
                        data=output_data.get("data"),
                        error=output_data.get("error"),
                        execution_time=execution_time,
                        metadata=output_data.get("metadata", {}),
                        streams=collected_streams,
                    )
                else:
                    execution_result = ExecutionResult(
                        success=False,
                        error="No results returned from flow execution",
                        execution_time=execution_time,
                        metadata={
                            "flow_name": flow_record.name,
                            "parameters": parameters,
                        },
                        streams=collected_streams,
                    )
            except Exception as parse_error:
                execution_result = ExecutionResult(
                    success=False,
                    error=f"Failed to parse execution result: {str(parse_error)}",
                    execution_time=execution_time,
                    metadata={"flow_name": flow_record.name, "parameters": parameters},
                    streams=collected_streams,
                )

            # Update execution record (COMPLETED/FAILED)
            try:
                await self.db.update_flow_run(
                    run_id=flow_run.id,
                    status="COMPLETED" if execution_result.success else "FAILED",
                    result=execution_result.data if execution_result.success else None,
                    error=(
                        execution_result.error if not execution_result.success else None
                    ),
                    execution_time_ms=(
                        int(execution_time * 1000) if execution_time else None
                    ),
                )
            except Exception as db_error:
                print(
                    f"Failed to update flow run {flow_run.id} after execution: {str(db_error)}"
                )

            return execution_result

        except asyncio.TimeoutError:
            error_result = ExecutionResult(
                success=False,
                error=f"Flow execution timed out after {timeout} seconds",
                metadata={
                    "flow_name": flow_record.name,
                    "parameters": parameters,
                    "timeout": timeout,
                },
                streams=collected_streams,
            )
            # Update execution record (FAILED)
            try:
                await self.db.update_flow_run(
                    run_id=flow_run.id,
                    status="FAILED",
                    error=error_result.error,
                    execution_time_ms=timeout * 1000,
                )
            except Exception as db_error:
                print(
                    f"Failed to update flow run {flow_run.id} after timeout: {str(db_error)}"
                )
            return error_result

        except Exception as e:
            error_result = ExecutionResult(
                success=False,
                error=f"Unexpected error during flow execution: {str(e)}",
                metadata={"flow_name": flow_record.name, "parameters": parameters},
                streams=collected_streams,
            )
            # Update execution record (FAILED)
            try:
                await self.db.update_flow_run(
                    run_id=flow_run.id,
                    status="FAILED",
                    error=error_result.error,
                )
            except Exception as db_error:
                print(
                    f"Failed to update flow run {flow_run.id} after exception: {str(db_error)}"
                )
            return error_result

    async def execute_flow_by_id_with_streaming(
        self,
        flow_id: str,
        parameters: Dict[str, Any],
        timeout: int = 300,
        on_stream: Optional[callable] = None,
        run_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a flow by ID with streaming.

        Args:
            flow_id: ID of the flow to execute
            parameters: Parameters to pass to the flow
            timeout: Execution timeout in seconds
            on_stream: Callback for stream events
            run_id: Optional pre-generated run ID to use instead of creating new one
        """
        flow_record = await self._get_flow_by_id(flow_id)
        if not flow_record:
            return ExecutionResult(
                success=False,
                error=f"Flow with ID '{flow_id}' not found",
                streams=[],
                metadata={},
            )

        # Check if flow is ready
        if flow_record.status != "ready":
            return ExecutionResult(
                success=False,
                error=f"Flow '{flow_record.name}' is not activated. Current status: {flow_record.status}",
                streams=[],
                metadata={},
            )

        return await self._execute_flow_with_streaming_impl(
            flow_record, parameters, timeout, on_stream, run_id
        )

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
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            if hasattr(f, 'flush'):
                f.flush()
    def flush(self):
        for f in self.files:
            if hasattr(f, 'flush'):
                f.flush()

async def execute_flow():
    parameters = {json.dumps(parameters, indent=2)}

    import io
    original_stdout = sys.stdout
    captured_output = io.StringIO()
    
    # Use Tee to write to both original stdout (for streaming) and capture buffer (for logs)
    tee_stdout = Tee(original_stdout, captured_output)

    try:
        sys.stdout = tee_stdout
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
