"""Code execution tool for running Python code in sandbox."""

from src.flows.core.sandbox import run_python_code
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing import Annotated


@tool(description="Execute Python code in a secure sandbox environment.")
async def python_code_executor(
    file_path: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Execute Python code in a secure sandbox environment.

    Args:
        file_path (str): The path of the file containing the code to execute e.g., "flows.py".
    Returns:
        result (str): The output from executing the code.
    """
    # Ensure file_name starts with / for compatibility with deepagents filesystem
    file_data = state.get("files", {}).get(file_path)

    if not file_data or not file_data.get("content"):
        return f"Error: File '{file_path}' not found or is empty. Available files: {list(state.get('files', {}).keys())}"

    # Join lines back into code string
    code = "\n".join(file_data["content"])
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


__all__ = ["python_code_executor"]
