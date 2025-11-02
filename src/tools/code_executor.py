"""Code execution tool for running Python code in sandbox."""

from src.core.sandbox import run_python_code
from deepagents import DeepAgentState
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing import Annotated


@tool
async def python_code_executor(
    file_name: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Execute Python code in a secure sandbox environment.

    Args:
        file_name (str): The name of the file containing the code to execute e.g., "flows.py".
    Returns:
        result (str): The output from executing the code.
    """
    code = state.get("files", {}).get(file_name)
    try:
        result = await run_python_code(code)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"{result}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    except Exception as e:
        return f"Error executing code: {e}"


# Export the tool function
__all__ = ["python_code_executor"]
