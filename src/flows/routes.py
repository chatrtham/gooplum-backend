"""FastAPI endpoints for flow management and execution."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
import asyncio
from uuid import UUID

from src.flows.core.flow_discovery import FlowDiscovery
from src.flows.core.flow_validator import FlowValidator
from src.flows.core.db_flow_executor import DBFlowExecutor
from src.flows.core.flow_explainer import FlowExplainer
from src.flows.core.supabase_client import get_flow_db


# Pydantic models for API
class FlowCodeRequest(BaseModel):
    code: str = Field(..., description="Python code containing async flow functions")
    flow_name: Optional[str] = Field(None, description="Specific flow name to compile")


class FlowExecutionRequest(BaseModel):
    parameters: Dict[str, Any] = Field(
        ..., description="Parameters to pass to the flow"
    )
    timeout: Optional[int] = Field(300, description="Execution timeout in seconds")


class FlowInfo(BaseModel):
    id: str
    name: str
    description: str
    parameter_count: int
    required_parameters: int
    return_type: str
    created_at: Optional[str] = None


class FlowSchema(BaseModel):
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str
    created_at: Optional[str] = None


class ExecutionResponse(BaseModel):
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    sanitized_parameters: Optional[Dict[str, Any]] = None


class CompilationResponse(BaseModel):
    success: bool
    flows: List[FlowInfo] = []
    errors: List[str] = []
    compiled_count: int = 0


class ExplanationResponse(BaseModel):
    flow_name: str
    explanation: str
    created_at: Optional[datetime] = None


class FlowRunInfo(BaseModel):
    id: str
    flow_id: str
    status: str
    execution_time_ms: Optional[int] = None
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class FlowRunDetail(FlowRunInfo):
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    stream_events: List[Dict[str, Any]] = []


class PaginatedFlowRuns(BaseModel):
    runs: List[FlowRunInfo]
    total: int
    page: int
    limit: int


# Create router
router = APIRouter(prefix="/flows", tags=["flows"])

# Global instances (in production, these would be properly managed)
flow_executor = DBFlowExecutor()
flow_discovery = FlowDiscovery()
flow_validator = FlowValidator()
flow_explainer = FlowExplainer()
flow_db = get_flow_db()


@router.post("/compile", response_model=CompilationResponse)
async def compile_flows(request: FlowCodeRequest):
    """
    Compile and discover flows from provided code.

    Args:
        request: Flow code to compile

    Returns:
        Compilation result with discovered flows
    """
    try:
        # Compile the flows
        flows = await flow_executor.compile_flow(request.code, request.flow_name)

        # Convert to response format
        flow_infos = []
        for flow_name, flow_metadata in flows.items():
            flow_infos.append(
                FlowInfo(
                    id=flow_metadata.id,
                    name=flow_name,
                    description=flow_metadata.description,
                    parameter_count=len(flow_metadata.parameters),
                    required_parameters=len(
                        [p for p in flow_metadata.parameters if p.required]
                    ),
                    return_type=flow_metadata.return_type,
                    created_at=(
                        flow_metadata.created_at.isoformat()
                        if flow_metadata.created_at
                        else None
                    ),
                )
            )

        return CompilationResponse(
            success=True, flows=flow_infos, compiled_count=len(flows)
        )

    except ValueError as e:
        return CompilationResponse(success=False, errors=[str(e)], compiled_count=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compilation error: {str(e)}")


@router.post("/{flow_id}/activate")
async def activate_flow(flow_id: str):
    """
    Activate a draft flow, making it callable and visible in the flow list.

    Args:
        flow_id: ID of the flow to activate

    Returns:
        Activated flow info
    """
    try:
        flow_record = await flow_db.activate_flow(UUID(flow_id))
        return {
            "success": True,
            "flow_id": str(flow_record.id),
            "name": flow_record.name,
            "status": flow_record.status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activating flow: {str(e)}")


@router.get("/", response_model=List[FlowInfo])
async def list_flows():
    """
    List all available ready flows.

    Returns:
        List of available flows
    """
    try:
        flows = await flow_executor.get_available_flows()
        return [FlowInfo(**flow) for flow in flows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing flows: {str(e)}")


@router.get("/{flow_id}/schema", response_model=FlowSchema)
async def get_flow_schema(flow_id: str):
    """
    Get detailed schema for a specific flow.

    Args:
        flow_id: ID of the flow

    Returns:
        Flow schema with parameter details
    """
    try:
        schema = await flow_executor.get_flow_schema_by_id(flow_id)
        if not schema:
            raise HTTPException(
                status_code=404, detail=f"Flow with ID '{flow_id}' not found"
            )

        return FlowSchema(**schema)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting flow schema: {str(e)}"
        )


@router.post("/{flow_id}/validate", response_model=ValidationResponse)
async def validate_flow_parameters(flow_id: str, parameters: Dict[str, Any]):
    """
    Validate parameters against flow schema without executing.

    Args:
        flow_id: ID of the flow
        parameters: Parameters to validate

    Returns:
        Validation result
    """
    try:
        # Use the executor's built-in validation
        validation_result = await flow_executor.validate_flow_execution_by_id(
            flow_id, parameters
        )

        return ValidationResponse(
            is_valid=validation_result.success,
            errors=[validation_result.error] if validation_result.error else [],
            warnings=[],
            sanitized_parameters=parameters if validation_result.success else None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.post("/{flow_id}/execute", response_model=ExecutionResponse)
async def execute_flow(flow_id: str, request: FlowExecutionRequest):
    """
    Execute a flow with provided parameters.

    Args:
        flow_id: ID of the flow to execute
        request: Execution request with parameters

    Returns:
        Execution result
    """
    try:
        # Validate parameters first
        validation_result = await validate_flow_parameters(flow_id, request.parameters)
        if not validation_result.is_valid:
            return ExecutionResponse(
                success=False,
                error=f"Parameter validation failed: {'; '.join(validation_result.errors)}",
                metadata={"validation_errors": validation_result.errors},
            )

        # Use validated/sanitized parameters if available
        parameters = validation_result.sanitized_parameters or request.parameters

        # Execute the flow by ID
        result, run_id = await flow_executor.execute_flow_by_id(
            flow_id=flow_id, parameters=parameters, timeout=request.timeout
        )

        # Include run_id in metadata
        metadata = result.metadata or {}
        if run_id:
            metadata["run_id"] = str(run_id)

        return ExecutionResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            execution_time=result.execution_time,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")


@router.post("/{flow_id}/execute-stream")
async def execute_flow_stream(flow_id: str, request: FlowExecutionRequest):
    """
    Execute a flow with real-time streaming using Server-Sent Events.

    This endpoint streams intermediate results as they complete, allowing
    clients to see progress in real-time. Each input is processed in isolation
    so failures don't affect other inputs.

    Args:
        flow_id: ID of the flow to execute
        request: Execution request with parameters

    Returns:
        Server-Sent Events stream with real-time results
    """

    async def generate_stream():
        """Generate SSE stream for flow execution."""
        try:
            # Validate parameters first
            validation_result = await validate_flow_parameters(
                flow_id, request.parameters
            )
            if not validation_result.is_valid:
                error_data = {
                    "type": "error",
                    "message": f"Parameter validation failed: {'; '.join(validation_result.errors)}",
                    "validation_errors": validation_result.errors,
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            # Use validated/sanitized parameters if available
            parameters = validation_result.sanitized_parameters or request.parameters

            # Get flow name for logging
            flow_name = flow_executor._get_flow_name_by_id(flow_id)

            # Send start event
            start_data = {
                "type": "start",
                "flow_id": flow_id,
                "flow_name": flow_name,
                "parameters": parameters,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            yield f"data: {json.dumps(start_data)}\n\n"

            # Set up a queue for real-time streaming
            stream_queue = asyncio.Queue()
            execution_complete = asyncio.Event()
            final_result = None
            execution_error = None

            def on_stream(stream_result):
                """Handle each streamed result from the flow - send to queue immediately."""
                stream_data = {
                    "type": "stream",
                    "status": stream_result.status,
                    "message": stream_result.message,
                    "timestamp": stream_result.timestamp,
                }
                # Put directly in queue for immediate streaming
                asyncio.create_task(stream_queue.put(("stream", stream_data)))

            # Execute flow in background
            async def execute_flow_background():
                nonlocal final_result, execution_error
                try:
                    result = await flow_executor.execute_flow_by_id_with_streaming(
                        flow_id=flow_id,
                        parameters=parameters,
                        timeout=request.timeout,
                        on_stream=on_stream,
                    )
                    final_result = result
                except Exception as e:
                    execution_error = e
                finally:
                    execution_complete.set()

            # Start execution in background
            execution_task = asyncio.create_task(execute_flow_background())

            # Stream results as they come in
            while not execution_complete.is_set() or not stream_queue.empty():
                try:
                    # Wait for a stream result with timeout
                    event_type, event_data = await asyncio.wait_for(
                        stream_queue.get(), timeout=1.0
                    )
                    yield f"data: {json.dumps(event_data)}\n\n"
                except asyncio.TimeoutError:
                    # No stream result yet, continue waiting
                    continue

            # Wait for execution to complete if not already done
            await execution_complete.wait()

            # Send final result
            if execution_error:
                final_data = {
                    "type": "complete",
                    "success": False,
                    "error": str(execution_error),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                final_data = {
                    "type": "complete",
                    "success": final_result.success if final_result else False,
                    "data": final_result.data if final_result else None,
                    "error": final_result.error if final_result else None,
                    "execution_time": (
                        final_result.execution_time if final_result else None
                    ),
                    "total_streams": (
                        len(final_result.streams)
                        if final_result and final_result.streams
                        else 0
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": final_result.metadata if final_result else {},
                }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            # Send error event
            error_data = {
                "type": "error",
                "message": f"Execution error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@router.delete("/{flow_id}")
async def delete_flow(flow_id: str):
    """
    Remove a compiled flow.

    Args:
        flow_id: ID of the flow to remove

    Returns:
        Deletion confirmation
    """
    try:
        success = await flow_executor.remove_flow_by_id(flow_id)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Flow with ID '{flow_id}' not found"
            )

        return {
            "success": True,
            "message": f"Flow with ID '{flow_id}' removed successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting flow: {str(e)}")


@router.delete("/")
async def clear_all_flows():
    """
    Clear all compiled flows.

    Returns:
        Clear confirmation
    """
    try:
        flow_count = await flow_executor.clear_all_flows()

        return {"success": True, "message": f"Cleared {flow_count} flows successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing flows: {str(e)}")


@router.get("/{flow_id}/code")
async def get_flow_code(flow_id: str):
    """
    Get the source code for a specific flow.

    Args:
        flow_id: ID of the flow

    Returns:
        Flow source code
    """
    try:
        # Get flow from database
        flow_record = await flow_db.get_flow(UUID(flow_id))
        if not flow_record:
            raise HTTPException(
                status_code=404, detail=f"Flow with ID '{flow_id}' not found"
            )

        flow_name = flow_executor._get_flow_name_by_id(flow_id)
        if not flow_name:
            flow_name = flow_record.name

        return {
            "flow_id": flow_id,
            "flow_name": flow_name,
            "source_code": flow_record.source_code,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting flow code: {str(e)}"
        )


@router.get("/{flow_id}/explanation", response_model=ExplanationResponse)
async def get_flow_explanation(flow_id: str):
    """
    Get the generated explanation for a specific flow.

    Args:
        flow_id: ID of the flow

    Returns:
        Flow explanation in markdown format
    """
    try:
        # Get the enriched flow metadata with explanations
        flow_metadata = await flow_executor.get_enriched_flow_metadata_by_id(flow_id)

        if not flow_metadata:
            raise HTTPException(
                status_code=404, detail=f"Flow with ID '{flow_id}' not found"
            )

        if not flow_metadata.explanation:
            raise HTTPException(
                status_code=404,
                detail=f"No explanation available for flow with ID '{flow_id}'. The flow might have been compiled before explanation generation was implemented.",
            )

        return ExplanationResponse(
            flow_name=flow_metadata.name,
            explanation=flow_metadata.explanation,
            created_at=flow_metadata.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting flow explanation: {str(e)}"
        )


@router.post("/{flow_id}/regenerate-explanation", response_model=ExplanationResponse)
async def regenerate_flow_explanation(flow_id: str):
    """
    Regenerate the explanation for a specific flow.

    Args:
        flow_id: ID of the flow

    Returns:
        Newly generated flow explanation
    """
    try:
        # Get the current flow metadata
        flow_metadata = await flow_executor.get_enriched_flow_metadata_by_id(flow_id)

        if not flow_metadata:
            raise HTTPException(
                status_code=404, detail=f"Flow with ID '{flow_id}' not found"
            )

        # Generate new explanation
        try:
            new_explanation = await flow_explainer.generate_explanation(flow_metadata)

            # Update the flow in database with new explanation
            await flow_db.update_flow(UUID(flow_id), {"explanation": new_explanation})

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to generate explanation: {str(e)}"
            )

        return ExplanationResponse(
            flow_name=flow_metadata.name,
            explanation=new_explanation,
            created_at=datetime.now(timezone.utc),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error regenerating flow explanation: {str(e)}"
        )


@router.get("/{flow_id}/runs", response_model=PaginatedFlowRuns)
async def list_flow_runs(flow_id: str, page: int = 1, limit: int = 10):
    """
    List recent runs for a specific flow.

    Args:
        flow_id: ID of the flow
        page: Page number (1-based)
        limit: Maximum number of runs to return

    Returns:
        Paginated list of flow runs
    """
    try:
        offset = (page - 1) * limit
        runs, total = await flow_db.get_flow_runs(UUID(flow_id), limit, offset)

        run_infos = [
            FlowRunInfo(
                id=str(run.id),
                flow_id=str(run.flow_id),
                status=run.status,
                execution_time_ms=run.execution_time_ms,
                created_at=run.created_at.isoformat(),
                completed_at=run.completed_at.isoformat() if run.completed_at else None,
                result=run.result,
                error=run.error,
            )
            for run in runs
        ]

        return PaginatedFlowRuns(runs=run_infos, total=total, page=page, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing flow runs: {str(e)}"
        )


@router.get("/runs/{run_id}", response_model=FlowRunDetail)
async def get_flow_run_details(run_id: str):
    """
    Get detailed information about a specific flow run, including stream events.

    Args:
        run_id: ID of the run

    Returns:
        Flow run details
    """
    try:
        run = await flow_db.get_flow_run(UUID(run_id))
        if not run:
            raise HTTPException(
                status_code=404, detail=f"Run with ID '{run_id}' not found"
            )

        # Get stream events
        events = await flow_db.get_run_events(UUID(run_id))

        return FlowRunDetail(
            id=str(run.id),
            flow_id=str(run.flow_id),
            status=run.status,
            execution_time_ms=run.execution_time_ms,
            created_at=run.created_at.isoformat(),
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
            parameters=run.parameters,
            result=run.result,
            error=run.error,
            metadata=run.metadata,
            stream_events=events,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting run details: {str(e)}"
        )
