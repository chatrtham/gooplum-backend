"""Convert flows to LangChain tools for use in agents."""

import json
from typing import Any
from uuid import UUID

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model

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
        """Execute the flow asynchronously and return run_id."""
        executor = DBFlowExecutor()

        # Create a run record (RUNNING status)
        db = get_flow_db()
        flow_run = await db.create_flow_run(
            flow_id=_flow.id,
            parameters=kwargs,
            metadata={"source": "agent_tool", "flow_name": _flow.name},
        )

        # Start execution in background (fire and forget pattern)
        import asyncio

        async def execute_in_background():
            try:
                result = await executor.execute_flow(_flow.name, kwargs)
                await db.update_flow_run(
                    run_id=flow_run.id,
                    status="COMPLETED" if result.success else "FAILED",
                    result=result.data if result.success else None,
                    error=result.error if not result.success else None,
                )
            except Exception as e:
                await db.update_flow_run(
                    run_id=flow_run.id,
                    status="FAILED",
                    error=str(e),
                )

        asyncio.create_task(execute_in_background())

        return json.dumps(
            {
                "status": "started",
                "run_id": str(flow_run.id),
                "message": f"Flow '{_flow.name}' execution started. Poll /flows/runs/{flow_run.id} for results.",
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
