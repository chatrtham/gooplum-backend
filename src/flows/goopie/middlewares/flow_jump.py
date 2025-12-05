"""Middleware for jumping to end after flow compilation."""

from langchain.agents.middleware import (
    before_model,
    hook_config,
    AgentState,
)
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime


@before_model
@hook_config(can_jump_to=["end"])
async def jump_to_end(state: AgentState, runtime: Runtime) -> dict | None:
    """Jump to end if the last message is a ToolMessage and has flow_id in artifact.

    Args:
        state: The current agent state

    Returns:
        Dict with jump_to instruction if last message is ToolMessage and has flow_id in artifact, None otherwise
    """
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            artifact = getattr(last_message, "artifact", None)
            if isinstance(artifact, dict) and "flow_id" in artifact:
                return {"jump_to": "end"}
    return None
