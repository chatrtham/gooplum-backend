"""Flow compiler tool for compiling Python flows from files."""

from src.flows.core.db_flow_executor import DBFlowExecutor
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing import Annotated
import traceback


@tool(description="Compile a Python flow from a file path.")
async def flow_compiler(
    file_path: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Compile a Python flow from a file path.

    Args:
        file_path (str): The path to the file containing flow code to compile.
    Returns:
        result (str): Compilation status and discovered flow information.
    """
    # Get the code from the agent state files
    file_data = state.get("files", {}).get(file_path)

    if not file_data or not file_data.get("content"):
        error_msg = f"File '{file_path}' not found or is empty. Available files: {list(state.get('files', {}).keys())}"
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: {error_msg}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Join lines back into code string
    code = "\n".join(file_data["content"])

    # Get the database flow executor
    executor = DBFlowExecutor()

    try:
        # Compile the flow
        flows = await executor.compile_flow(code)

        # Get the single flow
        flow_name = list(flows.keys())[0]
        flow_metadata = flows[flow_name]

        success_msg = f"Successfully compiled flow '{flow_name}' from '{file_path}'"

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=success_msg,
                        tool_call_id=tool_call_id,
                        artifact={
                            "flow_id": flow_metadata.id,
                            "flow_name": flow_name,
                            "flow_description": flow_metadata.description,
                            "flow_explanation": flow_metadata.explanation,
                        },
                    )
                ]
            },
        )

    except ValueError as ve:
        # Handle flow discovery/validation errors
        error_msg = f"Flow compilation failed: {str(ve)}"

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        error_msg,
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    except SyntaxError as se:
        # Handle syntax errors in the flow code
        error_msg = f"Syntax Error in flow code:\n{se.msg}\nLine {se.lineno}, Column {se.offset}\n\n{se.text}"

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        error_msg,
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error during compilation: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        error_msg,
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


__all__ = ["flow_compiler"]
