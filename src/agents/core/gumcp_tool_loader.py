"""Load guMCP tools dynamically for specified services.

Uses MultiServerMCPClient to connect to guMCP services and convert them to LangChain tools.
"""

import os
from typing import Optional
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


# Base URL pattern for guMCP services
GUMCP_BASE_URL = "https://mcp.gumloop.com"


def get_available_gumcp_services() -> list[str]:
    """Get list of available guMCP services by scanning gumcp_docs directory."""
    gumcp_docs_dir = Path("resources/gumcp_docs")

    if not gumcp_docs_dir.exists():
        return []

    services = []
    for item in gumcp_docs_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            services.append(item.name)

    return sorted(services)


def build_mcp_client_config(service_names: list[str]) -> dict:
    """Build MultiServerMCPClient configuration for specified services."""
    credentials = os.getenv("GUMCP_CREDENTIALS")
    if not credentials:
        raise ValueError("GUMCP_CREDENTIALS environment variable is not set")

    config = {}
    for service_name in service_names:
        config[service_name] = {
            "transport": "streamable_http",
            "url": f"{GUMCP_BASE_URL}/{service_name}/{credentials}/mcp",
        }

    return config


async def load_gumcp_tools(service_names: list[str]) -> list[BaseTool]:
    """Load guMCP tools for specified services.

    Args:
        service_names: List of service names (e.g., ["gmail", "gsheets"])

    Returns:
        List of LangChain tools from all specified services
    """
    if not service_names:
        return []

    # Filter to only available services
    available = set(get_available_gumcp_services())
    valid_services = [s for s in service_names if s in available]

    if not valid_services:
        return []

    # Build client config
    config = build_mcp_client_config(valid_services)

    # Create client and get tools
    client = MultiServerMCPClient(config)
    tools = await client.get_tools()

    return tools


class GumcpToolLoader:
    """Helper class for managing guMCP tool loading with caching."""

    _client: Optional[MultiServerMCPClient] = None
    _loaded_services: set[str] = set()
    _tools: list[BaseTool] = []

    def __init__(self, service_names: list[str]):
        """Initialize with list of service names to load."""
        self.service_names = service_names

    async def get_tools(self) -> list[BaseTool]:
        """Get tools, loading them if necessary."""
        if not self.service_names:
            return []

        # Check if we need to reload
        current_services = set(self.service_names)
        if current_services != self._loaded_services:
            self._tools = await load_gumcp_tools(self.service_names)
            self._loaded_services = current_services

        return self._tools

    @classmethod
    def reset(cls):
        """Reset the cached client and tools."""
        cls._client = None
        cls._loaded_services = set()
        cls._tools = []
