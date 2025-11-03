"""Flow compiler tool for compiling Python flows from files."""

from src.core.shared_flow_executor import get_shared_flow_executor
from deepagents import DeepAgentState
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing import Annotated
import traceback


@tool
async def flow_compiler(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Compile a Python flow from a file path.

    This tool compiles flow code from a specified file, discovering async functions
    and caching them for execution. The compilation process extracts production code
    (above __main__ blocks) and validates flow structure.

    Args:
        file_path (str): The path to the file containing flow code to compile.
    Returns:
        result (str): Compilation status and discovered flow information.
    """
    # Get the code from the agent state files
    code = state.get("files", {}).get(file_path)

    if not code:
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

    # Get the shared flow executor
    executor = get_shared_flow_executor()

    try:
        # Compile the flow
        flows = await executor.compile_flow(code)

        # Prepare compilation result
        flow_count = len(flows)
        flow_names = list(flows.keys())
        flow_details = []

        for flow_name, flow_metadata in flows.items():
            details = {
                "name": flow_name,
                "description": flow_metadata.description,
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.type,
                        "required": param.required,
                        "description": param.description,
                    }
                    for param in flow_metadata.parameters
                ],
                "return_type": flow_metadata.return_type,
            }
            flow_details.append(details)

        success_msg = f"""
✅ Successfully compiled {flow_count} flow(s) from '{file_path}':

Discovered flows:
{chr(10).join([f"- {name}: {details['description']}" for name, details in zip(flow_names, flow_details)])}

Flow details:
"""

        for i, (flow_name, details) in enumerate(zip(flow_names, flow_details)):
            success_msg += f"""
{i+1}. {flow_name}
   Description: {details['description']}
   Parameters: {len(details['parameters'])} total ({len([p for p in details['parameters'] if p['required']])} required)
   Return type: {details['return_type']}
"""

        success_msg += f"""
Next steps:
- Use the flow execution API or tools to run these flows
- Validate parameters before execution if needed
- Flow names are now available: {flow_names}
"""

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        success_msg,
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    except ValueError as ve:
        # Handle flow discovery/validation errors
        error_msg = f"❌ Flow compilation failed: {str(ve)}"

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
        error_msg = f"❌ Unexpected error during compilation: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

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


# Export the tool function
__all__ = ["flow_compiler"]
