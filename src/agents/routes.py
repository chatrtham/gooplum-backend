"""FastAPI routes for GoopLum Agents CRUD and discovery."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

from src.agents.core.supabase_client import (
    AgentCreate,
    AgentUpdate,
    AgentRecord,
    create_agent,
    get_agent,
    get_agent_by_name,
    list_agents,
    update_agent,
    delete_agent,
)
from src.agents.core.model_config import get_available_presets
from src.agents.core.gumcp_tool_loader import get_available_gumcp_services


router = APIRouter(prefix="/agents", tags=["agents"])


# --- Response Models ---


class AgentResponse(BaseModel):
    """Response model for agent endpoints."""

    id: str
    name: str
    description: Optional[str] = None
    system_prompt: str
    model_preset: str
    flow_tool_ids: list[str] = Field(default_factory=list)
    gumcp_services: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class AgentListResponse(BaseModel):
    """Response model for listing agents."""

    agents: list[AgentResponse]
    total: int


class ModelPresetResponse(BaseModel):
    """Response model for model presets."""

    name: str
    provider: str
    model: str
    description: str


class GumcpServiceResponse(BaseModel):
    """Response model for guMCP services."""

    services: list[str]


# --- Helper Functions ---


def agent_record_to_response(record: AgentRecord) -> AgentResponse:
    """Convert AgentRecord to AgentResponse."""
    return AgentResponse(
        id=str(record.id),
        name=record.name,
        description=record.description,
        system_prompt=record.system_prompt,
        model_preset=record.model_preset,
        flow_tool_ids=[str(fid) for fid in record.flow_tool_ids],
        gumcp_services=record.gumcp_services,
        created_at=record.created_at,
    )


# --- Discovery Endpoints ---


@router.get("/presets", response_model=list[ModelPresetResponse])
async def list_model_presets():
    """List available model presets."""
    presets = get_available_presets()
    return presets


@router.get("/gumcp-services", response_model=GumcpServiceResponse)
async def list_gumcp_services():
    """List available guMCP services."""
    services = get_available_gumcp_services()
    return GumcpServiceResponse(services=services)


# --- CRUD Endpoints ---


@router.post("", response_model=AgentResponse, status_code=201)
async def create_agent_endpoint(agent_data: AgentCreate):
    """Create a new agent."""
    try:
        record = await create_agent(agent_data)
        return agent_record_to_response(record)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=AgentListResponse)
async def list_agents_endpoint():
    """List all agents."""
    records = await list_agents()
    return AgentListResponse(
        agents=[agent_record_to_response(r) for r in records],
        total=len(records),
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent_endpoint(agent_id: UUID):
    """Get an agent by ID."""
    record = await get_agent(agent_id)
    if not record:
        raise HTTPException(
            status_code=404, detail=f"Agent with ID '{agent_id}' not found"
        )
    return agent_record_to_response(record)


@router.get("/name/{name}", response_model=AgentResponse)
async def get_agent_by_name_endpoint(name: str):
    """Get an agent by name."""
    record = await get_agent_by_name(name)
    if not record:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return agent_record_to_response(record)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent_endpoint(agent_id: UUID, update_data: AgentUpdate):
    """Update an agent."""
    record = await update_agent(agent_id, update_data)
    if not record:
        raise HTTPException(
            status_code=404, detail=f"Agent with ID '{agent_id}' not found"
        )
    return agent_record_to_response(record)


@router.delete("/{agent_id}", status_code=204)
async def delete_agent_endpoint(agent_id: UUID):
    """Delete an agent."""
    success = await delete_agent(agent_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Agent with ID '{agent_id}' not found"
        )
    return None
