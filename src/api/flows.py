"""FastAPI endpoints for flow management and execution."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.flow_executor import FlowExecutor
from src.core.flow_discovery import FlowDiscovery
from src.core.flow_validator import FlowValidator


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
flow_executor = FlowExecutor()
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
