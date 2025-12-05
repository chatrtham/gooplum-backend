"""Supabase client operations for agents."""

from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field
from src.db.client import get_supabase_client


class AgentRecord(BaseModel):
    """Database record for an agent."""

    id: UUID
    name: str
    description: Optional[str] = None
    system_prompt: str
    model_preset: str  # e.g., "claude-sonnet", "gpt-4o"
    flow_tool_ids: list[UUID] = Field(default_factory=list)
    gumcp_services: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class AgentCreate(BaseModel):
    """Request model for creating an agent."""

    name: str
    description: Optional[str] = None
    system_prompt: str
    model_preset: str  # e.g., "claude-sonnet", "gpt-4o"
    flow_tool_ids: list[UUID] = Field(default_factory=list)
    gumcp_services: list[str] = Field(default_factory=list)


class AgentUpdate(BaseModel):
    """Request model for updating an agent."""

    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    model_preset: Optional[str] = None
    flow_tool_ids: Optional[list[UUID]] = None
    gumcp_services: Optional[list[str]] = None


async def create_agent(agent: AgentCreate) -> AgentRecord:
    """Create a new agent in the database."""
    client = await get_supabase_client()

    data = {
        "name": agent.name,
        "description": agent.description,
        "system_prompt": agent.system_prompt,
        "model_preset": agent.model_preset,
        "flow_tool_ids": [str(fid) for fid in agent.flow_tool_ids],
        "gumcp_services": agent.gumcp_services,
    }

    result = await client.table("agents").insert(data).execute()

    if not result.data:
        raise ValueError("Failed to create agent")

    return _parse_agent_record(result.data[0])


async def get_agent(agent_id: UUID) -> Optional[AgentRecord]:
    """Get an agent by ID."""
    client = await get_supabase_client()

    result = await client.table("agents").select("*").eq("id", str(agent_id)).execute()

    if not result.data:
        return None

    return _parse_agent_record(result.data[0])


async def get_agent_by_name(name: str) -> Optional[AgentRecord]:
    """Get an agent by name."""
    client = await get_supabase_client()

    result = await client.table("agents").select("*").eq("name", name).execute()

    if not result.data:
        return None

    return _parse_agent_record(result.data[0])


async def list_agents() -> list[AgentRecord]:
    """List all agents."""
    client = await get_supabase_client()

    query = client.table("agents").select("*").order("created_at", desc=True)
    result = await query.execute()

    return [_parse_agent_record(row) for row in result.data]


async def update_agent(agent_id: UUID, update: AgentUpdate) -> Optional[AgentRecord]:
    """Update an agent."""
    client = await get_supabase_client()

    # Build update data, excluding None values
    data = {}
    if update.name is not None:
        data["name"] = update.name
    if update.description is not None:
        data["description"] = update.description
    if update.system_prompt is not None:
        data["system_prompt"] = update.system_prompt
    if update.model_preset is not None:
        data["model_preset"] = update.model_preset
    if update.flow_tool_ids is not None:
        data["flow_tool_ids"] = [str(fid) for fid in update.flow_tool_ids]
    if update.gumcp_services is not None:
        data["gumcp_services"] = update.gumcp_services

    if not data:
        return await get_agent(agent_id)

    result = await client.table("agents").update(data).eq("id", str(agent_id)).execute()

    if not result.data:
        return None

    return _parse_agent_record(result.data[0])


async def delete_agent(agent_id: UUID) -> bool:
    """Delete an agent."""
    client = await get_supabase_client()

    result = await client.table("agents").delete().eq("id", str(agent_id)).execute()

    return len(result.data) > 0


def _parse_agent_record(row: dict) -> AgentRecord:
    """Parse a database row into an AgentRecord."""
    return AgentRecord(
        id=UUID(row["id"]),
        name=row["name"],
        description=row.get("description"),
        system_prompt=row["system_prompt"],
        model_preset=row["model_preset"],
        flow_tool_ids=[UUID(fid) for fid in (row.get("flow_tool_ids") or [])],
        gumcp_services=row.get("gumcp_services") or [],
        created_at=row.get("created_at"),
    )
