"""Custom middleware for loading gumcp documentation."""

from langchain.agents.middleware import before_agent
from src.core.document_loader import load_gumcp_files_async


@before_agent
async def add_gumcp_docs(state, runtime) -> dict:
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
