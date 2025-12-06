"""Agent factory for dynamically building agents from database configuration."""

from uuid import UUID

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from src.agents.core.supabase_client import get_agent, AgentRecord
from src.agents.core.model_config import get_model_from_preset
from src.agents.core.flow_tool_adapter import create_flow_tools, create_flow_status_tool
from src.agents.core.gumcp_tool_loader import load_gumcp_tools


async def build_agent_from_record(agent_record: AgentRecord):
    """Build a LangGraph agent from an AgentRecord.

    Args:
        agent_record: The agent configuration from the database

    Returns:
        A compiled LangGraph agent
    """
    # 1. Build the model from preset
    model = get_model_from_preset(agent_record.model_preset)

    # 2. Load flow tools (from compiled flows)
    flow_tools = await create_flow_tools(agent_record.flow_tool_ids)

    # 3. Load guMCP tools (from external services)
    gumcp_tools = await load_gumcp_tools(agent_record.gumcp_services)

    # 4. Combine all tools
    all_tools = flow_tools + gumcp_tools

    # 5. If there are flow tools, add the status checker tool
    #    This allows the agent to poll for long-running flow results
    if flow_tools:
        all_tools.append(create_flow_status_tool())

    # 5. Create the agent
    agent = create_agent(
        model=model,
        system_prompt=agent_record.system_prompt,
        tools=all_tools,
    )

    return agent


async def build_agent(agent_id: UUID):
    """Build a LangGraph agent from an agent ID.

    Args:
        agent_id: UUID of the agent to build

    Returns:
        A compiled LangGraph agent

    Raises:
        ValueError: If agent not found
    """
    agent_record = await get_agent(agent_id)

    if not agent_record:
        raise ValueError(f"Agent with ID '{agent_id}' not found")

    return await build_agent_from_record(agent_record)


async def build_agent_by_name(name: str):
    """Build a LangGraph agent by name.

    Args:
        name: Name of the agent to build

    Returns:
        A compiled LangGraph agent

    Raises:
        ValueError: If agent not found
    """
    from src.agents.core.supabase_client import get_agent_by_name

    agent_record = await get_agent_by_name(name)

    if not agent_record:
        raise ValueError(f"Agent '{name}' not found")

    return await build_agent_from_record(agent_record)


# --- LangGraph entrypoint for langgraph.json ---


async def make_agent(config: RunnableConfig):
    """Factory function for langgraph.json that rebuilds graph per-run.

    This function is called by LangGraph server for each new run.
    It reads agent_id from config.configurable and builds the appropriate
    agent with the correct model, tools, and system prompt.

    Usage in langgraph.json:
        "custom_agent": "./src/agents/core/agent_factory.py:make_agent"

    Invocation (via LangGraph API):
        POST /runs
        {
            "assistant_id": "custom_agent",
            "input": {"messages": [{"role": "user", "content": "..."}]},
            "config": {"configurable": {"agent_id": "uuid-here"}}
        }

    Args:
        config: RunnableConfig containing configurable.agent_id

    Returns:
        A compiled LangGraph agent
    """
    agent_id_str = config.get("configurable", {}).get("agent_id")

    if not agent_id_str:
        raise ValueError(
            "agent_id must be provided in config.configurable. "
            "Example: config={'configurable': {'agent_id': 'your-uuid'}}"
        )

    agent_id = UUID(agent_id_str)
    agent_record = await get_agent(agent_id)

    if not agent_record:
        raise ValueError(f"Agent with ID '{agent_id}' not found")

    return await build_agent_from_record(agent_record)
