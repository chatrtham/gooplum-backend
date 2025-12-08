"""Agent factory for dynamically building agents from LangGraph Assistant configuration.

This module builds agents using configuration from LangGraph Assistants.
The assistant's context (model_preset, system_prompt, flow_tool_ids, gumcp_services)
is passed via config.configurable when invoking the agent.

Frontend uses the LangGraph Assistants API directly:
- POST /assistants - Create assistant with context
- GET /assistants - List assistants
- PATCH /assistants/{id} - Update assistant
- DELETE /assistants/{id} - Delete assistant
"""

from typing import Optional
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.runnables import RunnableConfig

from src.agents.core.llms import get_model_from_preset
from src.agents.core.flow_tool_adapter import create_flow_tools, create_flow_status_tool
from src.agents.core.gumcp_tool_loader import load_gumcp_tools


async def build_agent_from_context(
    model_preset: str,
    system_prompt: Optional[str] = "",
    flow_tool_ids: Optional[list[str]] = None,
    gumcp_services: Optional[list[str]] = None,
):
    """Build a LangGraph agent from context values.

    Args:
        model_preset: Model preset name (e.g., "claude-sonnet")
        system_prompt: System prompt for the agent
        flow_tool_ids: List of flow UUIDs to use as tools
        gumcp_services: List of guMCP service names

    Returns:
        A compiled LangGraph agent
    """
    # 1. Build the model from preset
    model = get_model_from_preset(model_preset)

    # 2. Load flow tools (from compiled flows)
    flow_ids = [UUID(fid) for fid in (flow_tool_ids or [])]
    flow_tools = await create_flow_tools(flow_ids)

    # 3. Load guMCP tools (from external services)
    gumcp_tools = await load_gumcp_tools(gumcp_services or [])

    # 4. Combine all tools
    all_tools = flow_tools + gumcp_tools

    # 5. If there are flow tools, add the status checker tool
    if flow_tools:
        all_tools.append(create_flow_status_tool())

    # 6. Create the agent
    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=all_tools,
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", 17000),
                keep=("messages", 6),
                trim_tokens_to_summarize=None,
            ),
        ],
    )

    return agent


# --- LangGraph entrypoint for langgraph.json ---


async def make_agent(config: RunnableConfig):
    """Factory function for langgraph.json that rebuilds graph per-run.

    This function is called by LangGraph server for each new run.
    It reads assistant configuration from config.configurable and builds
    the agent with the correct model, tools, and system prompt.

    The assistant's context fields are passed via configurable:
    - model_preset: Model preset name
    - system_prompt: System prompt for the agent
    - flow_tool_ids: List of flow UUIDs to use as tools
    - gumcp_services: List of guMCP service names

    Usage in langgraph.json:
        "custom_agent": "./src/agents/core/agent_factory.py:make_agent"

    Invocation (via LangGraph API):
        1. Create an assistant:
           POST /assistants
           {
               "graph_id": "custom_agent",
               "name": "my-assistant",
               "config": {
                   "configurable": {
                       "model_preset": "claude-sonnet",
                       "system_prompt": "You are a helpful assistant...",
                       "flow_tool_ids": ["uuid-1", "uuid-2"],
                       "gumcp_services": ["gmail", "gsheets"]
                   }
               }
           }

        2. Invoke using assistant_id:
           POST /threads/{thread_id}/runs
           {
               "assistant_id": "assistant-uuid-from-step-1",
               "input": {"messages": [{"role": "user", "content": "..."}]}
           }

    Args:
        config: RunnableConfig containing configurable with assistant context

    Returns:
        A compiled LangGraph agent
    """
    configurable = config.get("configurable", {})

    # Required fields
    model_preset = configurable.get("model_preset")

    if not model_preset:
        raise ValueError(
            "model_preset must be provided in assistant context. "
            "Create an assistant with config.configurable.model_preset"
        )

    # Optional fields
    system_prompt = configurable.get("system_prompt", "")
    flow_tool_ids = configurable.get("flow_tool_ids", [])
    gumcp_services = configurable.get("gumcp_services", [])

    return await build_agent_from_context(
        model_preset=model_preset,
        system_prompt=system_prompt,
        flow_tool_ids=flow_tool_ids,
        gumcp_services=gumcp_services,
    )
