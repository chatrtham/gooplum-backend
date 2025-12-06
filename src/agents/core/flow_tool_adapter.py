"""Convert flows to LangChain tools for use in agents."""

import json
from typing import Any
from uuid import UUID

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from src.flows.core.db_flow_executor import DBFlowExecutor
from src.flows.core.supabase_client import get_flow_db, FlowRecord, FlowParameterRecord


async def get_flow_with_params(
    flow_id: UUID,
) -> tuple[FlowRecord, list[FlowParameterRecord]] | None:
    """Get flow record and its parameters."""
    db = get_flow_db()
    flow = await db.get_flow(flow_id)
    if not flow:
        return None
    params = await db.get_flow_parameters(flow_id)
    return flow, params


def create_flow_tool(
    flow: FlowRecord, parameters: list[FlowParameterRecord]
) -> BaseTool:
    """Create a LangChain tool from a flow record.

    The tool will execute the flow asynchronously and return a run_id.
    Users can poll for results using the flow runs API.
    """
    # Build dynamic Pydantic model for tool arguments
    field_definitions = {}
    for param in parameters:
        # Map flow parameter types to Python types
        python_type = _map_type(param.type)
        default = ... if param.required else param.default_value
        field_definitions[param.name] = (
            python_type,
            Field(default=default, description=param.description or ""),
        )

    # Create args schema - always create a model, even if empty
    ArgsSchema = create_model(
        f"{flow.name}Args",
        __doc__=flow.description or f"Arguments for {flow.name}",
        **field_definitions,
    )

    # Capture flow in closure
    _flow = flow

    async def _execute_flow(**kwargs: Any) -> str:
        """Start flow execution in background and return run_id immediately."""
        import asyncio

        executor = DBFlowExecutor()

        # Create the run record first (status: RUNNING)
        run_id = await executor.create_flow_run_record(
            _flow.name,
            kwargs,
            metadata={"source": "agent_tool", "flow_name": _flow.name},
        )

        if not run_id:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Flow '{_flow.name}' not found or not ready",
                }
            )

        # Start execution in background (fire-and-forget)
        asyncio.create_task(
            executor.execute_flow_with_run_id(
                _flow.name,
                kwargs,
                run_id=run_id,
            )
        )

        # Return immediately with run_id
        return json.dumps(
            {
                "status": "started",
                "run_id": str(run_id),
                "message": f"Flow '{_flow.name}' started.",
            }
        )

    return StructuredTool(
        name=flow.name,
        description=flow.description or f"Execute the {flow.name} flow",
        coroutine=_execute_flow,
        args_schema=ArgsSchema,
    )


async def create_flow_tools(flow_ids: list[UUID]) -> list[BaseTool]:
    """Create LangChain tools from a list of flow IDs.

    Only includes flows that exist and are in 'ready' status.
    """
    tools = []

    for flow_id in flow_ids:
        result = await get_flow_with_params(flow_id)
        if result is None:
            continue

        flow, params = result

        # Only include ready flows
        if flow.status != "ready":
            continue

        tool = create_flow_tool(flow, params)
        tools.append(tool)

    return tools


def create_flow_status_tool() -> BaseTool:
    """Create a tool for checking flow run status.

    This tool is always included when flow tools are used,
    allowing the agent to poll for long-running flow results.
    """

    class FlowStatusArgs(BaseModel):
        """Arguments for checking flow run status."""

        run_id: str = Field(description="The run_id returned when the flow was started")

    async def _check_status(run_id: str) -> str:
        """Check the status of a flow run."""
        from uuid import UUID as UUIDType

        executor = DBFlowExecutor()

        try:
            uuid_id = UUIDType(run_id)
        except ValueError:
            return json.dumps({"error": f"Invalid run_id format: {run_id}"})

        status = await executor.get_flow_run_status(uuid_id)

        if not status:
            return json.dumps({"error": f"Flow run not found: {run_id}"})

        return json.dumps(status)

    return StructuredTool(
        name="get_flow_run_status",
        description="Check the status and result of a previously started flow. Use this to poll for results of long-running flows.",
        coroutine=_check_status,
        args_schema=FlowStatusArgs,
    )


def _map_type(type_str: str) -> type:
    """Map flow parameter type strings to Python types."""
    type_mapping = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "bool": bool,
        "boolean": bool,
        "list": list,
        "dict": dict,
        "List": list,
        "Dict": dict,
        "List[str]": list,
        "List[int]": list,
        "Dict[str, Any]": dict,
        "Optional[str]": str | None,
        "Optional[int]": int | None,
    }
    return type_mapping.get(type_str, str)
