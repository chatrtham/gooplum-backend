"""FastAPI endpoints for flow management and execution."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import asyncio

from src.core.flow_discovery import FlowDiscovery
from src.core.flow_validator import FlowValidator
from src.core.shared_flow_executor import get_shared_flow_executor


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
    name: str
    description: str
    parameter_count: int
    required_parameters: int
    return_type: str


class FlowSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str


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


# Create router
router = APIRouter(prefix="/flows", tags=["flows"])

# Global instances (in production, these would be properly managed)
flow_executor = get_shared_flow_executor()
flow_discovery = FlowDiscovery()
flow_validator = FlowValidator()


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
                    name=flow_name,
                    description=flow_metadata.description,
                    parameter_count=len(flow_metadata.parameters),
                    required_parameters=len(
                        [p for p in flow_metadata.parameters if p.required]
                    ),
                    return_type=flow_metadata.return_type,
                )
            )

        return CompilationResponse(
            success=True, flows=flow_infos, compiled_count=len(flows)
        )

    except ValueError as e:
        return CompilationResponse(success=False, errors=[str(e)], compiled_count=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compilation error: {str(e)}")


@router.get("/", response_model=List[FlowInfo])
async def list_flows():
    """
    List all available compiled flows.

    Returns:
        List of available flows
    """
    try:
        flows = await flow_executor.get_available_flows()
        return [FlowInfo(**flow) for flow in flows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing flows: {str(e)}")


@router.get("/{flow_name}", response_model=FlowSchema)
async def get_flow_schema(flow_name: str):
    """
    Get detailed schema for a specific flow.

    Args:
        flow_name: Name of the flow

    Returns:
        Flow schema with parameter details
    """
    try:
        schema = await flow_executor.get_flow_schema(flow_name)
        if not schema:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        return FlowSchema(**schema)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting flow schema: {str(e)}"
        )


@router.post("/{flow_name}/validate", response_model=ValidationResponse)
async def validate_flow_parameters(flow_name: str, parameters: Dict[str, Any]):
    """
    Validate parameters against flow schema without executing.

    Args:
        flow_name: Name of the flow
        parameters: Parameters to validate

    Returns:
        Validation result
    """
    try:
        # Get flow metadata
        schema = await flow_executor.get_flow_schema(flow_name)
        if not schema:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        # Recreate flow metadata for validation
        flows = flow_discovery.discover_flows(
            flow_executor._compiled_flows.get(flow_name, "")
        )
        if flow_name not in flows:
            raise HTTPException(status_code=404, detail="Flow metadata not found")

        flow_metadata = flows[flow_name]

        # Validate parameters
        validation_result = flow_validator.validate_parameters(
            flow_metadata, parameters
        )

        return ValidationResponse(
            is_valid=validation_result.is_valid,
            errors=[error.message for error in validation_result.errors],
            warnings=[warning.message for warning in validation_result.warnings],
            sanitized_parameters=validation_result.sanitized_parameters,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.post("/{flow_name}/execute", response_model=ExecutionResponse)
async def execute_flow(flow_name: str, request: FlowExecutionRequest):
    """
    Execute a flow with provided parameters.

    Args:
        flow_name: Name of the flow to execute
        request: Execution request with parameters

    Returns:
        Execution result
    """
    try:
        # Validate parameters first
        validation_result = await validate_flow_parameters(
            flow_name, request.parameters
        )
        if not validation_result.is_valid:
            return ExecutionResponse(
                success=False,
                error=f"Parameter validation failed: {'; '.join(validation_result.errors)}",
                metadata={"validation_errors": validation_result.errors},
            )

        # Use validated/sanitized parameters if available
        parameters = validation_result.sanitized_parameters or request.parameters

        # Execute the flow
        result = await flow_executor.execute_flow(
            flow_name=flow_name, parameters=parameters, timeout=request.timeout
        )

        return ExecutionResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            execution_time=result.execution_time,
            metadata=result.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")


@router.post("/{flow_name}/execute-stream")
async def execute_flow_stream(flow_name: str, request: FlowExecutionRequest):
    """
    Execute a flow with real-time streaming using Server-Sent Events.

    This endpoint streams intermediate results as they complete, allowing
    clients to see progress in real-time. Each input is processed in isolation
    so failures don't affect other inputs.

    Args:
        flow_name: Name of the flow to execute
        request: Execution request with parameters

    Returns:
        Server-Sent Events stream with real-time results
    """

    async def generate_stream():
        """Generate SSE stream for flow execution."""
        try:
            # Validate parameters first
            validation_result = await validate_flow_parameters(
                flow_name, request.parameters
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

            # Send start event
            start_data = {
                "type": "start",
                "flow_name": flow_name,
                "parameters": parameters,
                "timestamp": datetime.utcnow().isoformat(),
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
                    "input": stream_result.input_data,
                    "success": stream_result.success,
                    "data": stream_result.data,
                    "error": stream_result.error,
                    "completed_at": stream_result.completed_at,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                # Put directly in queue for immediate streaming
                asyncio.create_task(stream_queue.put(("stream", stream_data)))

            # Execute flow in background
            async def execute_flow_background():
                nonlocal final_result, execution_error
                try:
                    result = await flow_executor.execute_flow_with_streaming(
                        flow_name=flow_name,
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
                    "timestamp": datetime.utcnow().isoformat(),
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
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": final_result.metadata if final_result else {},
                }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            # Send error event
            error_data = {
                "type": "error",
                "message": f"Execution error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
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


@router.delete("/{flow_name}")
async def delete_flow(flow_name: str):
    """
    Remove a compiled flow.

    Args:
        flow_name: Name of the flow to remove

    Returns:
        Deletion confirmation
    """
    try:
        if flow_name not in flow_executor._compiled_flows:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        flow_executor.remove_flow(flow_name)

        return {"success": True, "message": f"Flow '{flow_name}' removed successfully"}

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
        flow_count = len(flow_executor._compiled_flows)
        flow_executor.clear_compiled_flows()

        return {"success": True, "message": f"Cleared {flow_count} flows successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing flows: {str(e)}")


@router.get("/{flow_name}/code")
async def get_flow_code(flow_name: str):
    """
    Get the source code for a specific flow.

    Args:
        flow_name: Name of the flow

    Returns:
        Flow source code
    """
    try:
        if flow_name not in flow_executor._compiled_flows:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        # Get the flow code and extract specific function
        full_code = flow_executor._compiled_flows[flow_name]
        flows = flow_discovery.discover_flows(full_code)

        if flow_name in flows and flows[flow_name].source_code:
            return {"flow_name": flow_name, "source_code": flows[flow_name].source_code}
        else:
            return {"flow_name": flow_name, "source_code": full_code}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting flow code: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the flow service.

    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "compiled_flows": len(flow_executor._compiled_flows),
        "service": "flow-executor",
    }
