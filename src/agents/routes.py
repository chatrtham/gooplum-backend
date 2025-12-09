"""FastAPI routes for GoopLum Agents discovery.

Agent CRUD is handled by LangGraph Assistants API (/assistants).
These routes only provide discovery for available presets and services.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.agents.core.llms import get_available_presets
from src.agents.core.gumcp_tool_loader import get_available_gumcp_services


router = APIRouter(prefix="/agents", tags=["agents"])


# --- Response Models ---


class ModelPresetResponse(BaseModel):
    """Response model for model presets."""

    name: str
    provider: str
    model: str
    description: str


class GumcpServiceResponse(BaseModel):
    """Response model for guMCP services."""

    services: list[str]


# --- Discovery Endpoints ---


@router.get("/presets", response_model=list[ModelPresetResponse])
async def list_model_presets():
    """List available model presets.

    Returns the available model presets that can be used when creating
    an assistant via the LangGraph Assistants API.

    Example assistant creation with preset:
        POST /assistants
        {
            "graph_id": "custom_agent",
            "name": "my-assistant",
            "config": {
                "configurable": {
                    "model_preset": "claude-sonnet",  # Use a preset name from this endpoint
                    ...
                }
            }
        }
    """
    presets = get_available_presets()
    return presets


@router.get("/gumcp-services", response_model=GumcpServiceResponse)
async def list_gumcp_services():
    """List available guMCP services.

    Returns the available guMCP service names that can be used when creating
    an assistant via the LangGraph Assistants API.

    Example assistant creation with guMCP services:
        POST /assistants
        {
            "graph_id": "custom_agent",
            "name": "my-assistant",
            "config": {
                "configurable": {
                    "gumcp_services": ["gmail", "gsheets"],  # Use service names from this endpoint
                    ...
                }
            }
        }
    """
    services = get_available_gumcp_services()
    return GumcpServiceResponse(services=services)
