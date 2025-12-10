"""
Agent factory for dynamically building agents from LangGraph Assistant configuration.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.runnables import RunnableConfig

from src.agents.core.llms import get_model_from_preset
from src.agents.core.flow_tool_adapter import create_flow_tools, create_flow_status_tool
from src.agents.core.gumcp_tool_loader import load_gumcp_tools
from src.agents.core.self_improvement_middleware import SelfImprovementMiddleware


def build_system_prompt(
    instructions: str,
    has_flow_tools: bool,
    has_gumcp_tools: bool,
) -> str:
    """Build the system prompt by combining user instructions with GoopLum context.

    Args:
        instructions: Custom instructions provided by the user
        has_flow_tools: Whether the agent has flow tools available
        has_gumcp_tools: Whether the agent has guMCP tools available

    Returns:
        Complete system prompt with GoopLum context and user instructions
    """
    sections = []

    # 1. User instructions (only if provided)
    if instructions and instructions.strip():
        sections.append(f"# Instructions:\n\n{instructions.strip()}\n\n---")

    # 2. GoopLum platform context (only if user has tools)
    has_any_tools = has_flow_tools or has_gumcp_tools
    if has_any_tools:
        sections.append("You are an AI assistant created on the GoopLum platform.")

    # 3. Available tools section (only if tools exist)
    if has_any_tools:
        available_tools = []
        if has_flow_tools:
            available_tools.append(
                "- GoopLum workflow tools: run saved automations (workflows) to process data and perform tasks."
            )
            available_tools.append(
                "- get_flow_run_status: retrieve the status and outputs of a pipeline run using its run_id."
            )
        if has_gumcp_tools:
            available_tools.append(
                "- guMCP servers: MCP servers with curated capabilities."
            )

        tools_section = "Available Tools:\n\n" + "\n".join(available_tools)
        sections.append(tools_section)

    # 4. Tool usage guidelines (only if tools exist)
    if has_any_tools:
        guidelines = ["Tool Usage Guidelines:"]

        if has_flow_tools:
            guidelines.append(
                "1. When calling a GoopLum workflow tool, provide inputs if needed"
            )
            guidelines.append(
                "2. To check a run's progress, call get_flow_run_status with the run_id, but DO NOT call this unless asked."
            )

        if has_gumcp_tools:
            guideline_num = 3 if has_flow_tools else 1
            guidelines.append(
                f"{guideline_num}. For guMCP tools, specify the action and required parameters as indicated by their schemas."
            )

        general_guideline_num = len(guidelines)
        guidelines.append(
            f"{general_guideline_num}. Before using a tool, briefly state what you're doing and why. Keep explanations concise and useful."
        )

        sections.append("\n".join(guidelines))

    # 5. Interaction etiquette (always include)
    sections.append("Interaction Etiquette: Be helpful, accurate, and efficient.")

    # 6. Current datetime (always include)
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
    sections.append(f"Current datetime: {current_time}")

    # 7. Function calling guidelines (only if tools exist)
    if has_any_tools:
        sections.append(
            "Function Calling Guidelines: When making function calls using tools that accept array or object parameters, ensure those are structured using JSON. For example, array/object parameters should be properly formatted. DO NOT make up values for optional parameters; only include them if provided by the user."
        )

    # Join all sections with double newlines
    return "\n\n".join(sections)


async def build_agent_from_context(
    model_preset: str,
    instructions: Optional[str] = "",
    flow_tool_ids: Optional[list[str]] = None,
    gumcp_services: Optional[list[str]] = None,
    can_suggest_improvements: bool = False,
    assistant_id: Optional[str] = None,
):
    """Build a LangGraph agent from context values.

    Args:
        model_preset: Model preset name (e.g., "claude-sonnet")
        instructions: Custom instructions for the agent's behavior
        flow_tool_ids: List of flow UUIDs to use as tools
        gumcp_services: List of guMCP service names
        can_suggest_improvements: Whether to enable self-improvement from conversations
        assistant_id: The assistant ID (required if can_suggest_improvements is True)

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

    # 6. Build system prompt from template
    system_prompt = build_system_prompt(
        instructions=instructions or "",
        has_flow_tools=len(flow_tools) > 0,
        has_gumcp_tools=len(gumcp_tools) > 0,
    )

    # 7. Build middleware list
    middleware_list = [
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 17000),
            keep=("messages", 6),
            trim_tokens_to_summarize=None,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
    ]

    # 8. Add self-improvement middleware if enabled
    if can_suggest_improvements and assistant_id:
        middleware_list.append(
            SelfImprovementMiddleware(
                assistant_id=assistant_id,
                current_instructions=instructions or "",
            )
        )

    # 9. Create the agent
    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=all_tools,
        middleware=middleware_list,
    )

    return agent


# --- LangGraph entrypoint for langgraph.json ---


async def make_agent(config: RunnableConfig):
    """Factory function for langgraph.json that rebuilds graph per-run.

    This function is called by LangGraph server for each new run.
    It reads assistant configuration from config.configurable and builds
    the agent with the correct model, tools, and system prompt template.

    The assistant's context fields are passed via configurable:
    - model_preset: Model preset name (required)
    - instructions: Custom instructions for the agent (optional)
    - flow_tool_ids: List of flow UUIDs to use as tools (optional)
    - gumcp_services: List of guMCP service names (optional)

    The system prompt is built dynamically from a template that includes:
    - User instructions (if provided)
    - GoopLum platform context (if tools are available)
    - Available tools description (if tools are available)
    - Tool usage guidelines (if tools are available)
    - Current datetime (always included)

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
                       "instructions": "You are an email automation specialist. Focus on efficient batch processing.",
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
    instructions = configurable.get("instructions", "")
    flow_tool_ids = configurable.get("flow_tool_ids", [])
    gumcp_services = configurable.get("gumcp_services", [])
    can_suggest_improvements = configurable.get("can_suggest_improvements", False)
    assistant_id = configurable.get("assistant_id")  # Set by LangGraph automatically

    return await build_agent_from_context(
        model_preset=model_preset,
        instructions=instructions,
        flow_tool_ids=flow_tool_ids,
        gumcp_services=gumcp_services,
        can_suggest_improvements=can_suggest_improvements,
        assistant_id=assistant_id,
    )
