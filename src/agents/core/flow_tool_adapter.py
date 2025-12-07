"""Convert flows to LangChain tools for use in agents."""

import json
from typing import Annotated, Any
from uuid import UUID, uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from src.flows.core.supabase_client import FlowParameterRecord, FlowRecord, get_flow_db


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

    The tool pre-generates a run_id and returns flow info in artifact.
    Frontend calls the execute-stream endpoint with the run_id.
    Agent can later check status via get_flow_run_status tool.
    """
    # Build dynamic Pydantic model for tool arguments
    field_definitions = {}
    for param in parameters:
        python_type = _map_type(param.type)
        default = ... if param.required else param.default_value
        field_definitions[param.name] = (
            python_type,
            Field(default=default, description=param.description or ""),
        )

    # Add injected tool_call_id for artifact support
    field_definitions["tool_call_id"] = (
        Annotated[str, InjectedToolCallId],
        Field(default=...),
    )

    ArgsSchema = create_model(
        f"{flow.name}Args",
        __doc__=flow.description or f"Arguments for {flow.name}",
        **field_definitions,
    )

    # Capture flow in closure
    _flow = flow

    async def _request_flow_execution(tool_call_id: str, **kwargs: Any) -> Command:
        """Return flow execution request with pre-generated run_id."""
        # Pre-generate run_id so agent can track it later
        run_id = str(uuid4())

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Flow '{_flow.name}' will be executed. Run ID: {run_id}. Results will be shown in the UI. You can check the status later using get_flow_run_status.",
                        tool_call_id=tool_call_id,
                        artifact={
                            "type": "flow_execution_request",
                            "flow_id": str(_flow.id),
                            "flow_name": _flow.name,
                            "run_id": run_id,
                            "parameters": kwargs,
                        },
                    )
                ]
            }
        )

    return StructuredTool(
        name=_flow.name,
        description=_flow.description or f"Execute the {_flow.name} flow",
        coroutine=_request_flow_execution,
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

    This allows the agent to check the final result of a flow
    that was triggered via artifact (executed by frontend).
    """

    class FlowStatusArgs(BaseModel):
        """Arguments for checking flow run status."""

        run_id: str = Field(
            description="The run_id from the flow execution request artifact"
        )

    async def _check_status(run_id: str) -> str:
        """Check the status of a flow run."""
        from src.flows.core.db_flow_executor import DBFlowExecutor

        executor = DBFlowExecutor()

        try:
            uuid_id = UUID(run_id)
        except ValueError:
            return json.dumps({"error": f"Invalid run_id format: {run_id}"})

        status = await executor.get_flow_run_status(uuid_id)

        if not status:
            return json.dumps(
                {
                    "status": "pending",
                    "message": "Flow run not found yet. The frontend may not have started execution, or it's still initializing.",
                }
            )

        return json.dumps(status)

    return StructuredTool(
        name="get_flow_run_status",
        description="Check the status and result of a flow that was triggered. Use the run_id from the flow execution request. Returns status (RUNNING, COMPLETED, FAILED) and result/error if finished.",
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
