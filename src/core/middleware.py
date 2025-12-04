"""Custom middleware for loading gumcp documentation."""

from langchain.agents.middleware import (
    before_agent,
    before_model,
    hook_config,
    AgentState,
)
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from src.core.document_loader import load_gumcp_files_async


@before_agent
async def add_gumcp_docs(state: AgentState, runtime: Runtime) -> dict:
    """Add gumcp documentation to the state before agent execution.

    This decorator-based middleware loads gumcp documentation files and adds them
    to the state before the agent starts execution, similar to the previous
    add_gumcp_docs node. The documentation is only loaded if no files already
    exist in the state.

    Args:
        state: The current agent state

    Returns:
        Dict with files update if gumcp docs need to be loaded, empty dict otherwise
    """
    # Only add gumcp docs if no files exist
    if not state.get("files", {}):
        gumcp_files = await load_gumcp_files_async()
        return {"files": gumcp_files}

    return {}


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
