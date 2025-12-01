import asyncio
import os
import json
import sys
from datetime import datetime
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Gumloop user ID from environment
GUMCP_CREDENTIALS = os.getenv("GUMCP_CREDENTIALS")

if not GUMCP_CREDENTIALS:
    print("ERROR: GUMCP_CREDENTIALS environment variable not set.")
    print("Please set it to use guMCP integrations.")
    exit(1)


def get_integrations_list():
    """Read available integrations from gumcp_integrations.txt"""
    try:
        with open("scripts/gumcp_integrations.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()

        integrations = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip comments and empty lines
                integrations.append(line)

        return integrations
    except FileNotFoundError:
        print("ERROR: gumcp_integrations.txt not found in scripts directory.")
        return []
    except Exception as e:
        print(f"Error reading gumcp_integrations.txt: {e}")
        return []


def generate_tool_index(tools: list, integration_name: str) -> str:
    """Generate index file with tool descriptions only"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    doc = f"# {integration_name.title()} Tools Index\n\n"
    doc += f"*Generated on: {current_date}*\n\n"
    doc += f"## Available Tools ({len(tools)} total)\n\n"

    for i, tool in enumerate(tools, 1):
        doc += f"{i}. **{tool.name}** - {tool.description}\n"

    return doc


def generate_individual_tool_file(tool) -> str:
    """Generate documentation for a single tool with full schema"""
    doc = f"Parameters Schema:\n"
    doc += f"```json\n"
    doc += f"{json.dumps(tool.args, indent=2)}\n"
    doc += f"```\n"

    return doc


async def discover_and_document_tools(integration_name: str, user_id: str):
    """Discover tools and generate hybrid documentation structure"""
    try:
        print(f"Discovering {integration_name} guMCP tools for user: {user_id}")

        # Initialize guMCP client
        client = MultiServerMCPClient(
            {
                integration_name: {
                    "transport": "streamable_http",
                    "url": f"https://mcp.gumloop.com/{integration_name}/{user_id}/mcp",
                }
            }
        )

        # Get all available tools
        tools = await client.get_tools()

        print(f"\n=== Found {len(tools)} {integration_name} guMCP Tools ===\n")

        # Create directory for this integration
        integration_dir = f"resources/gumcp_docs/{integration_name}"
        os.makedirs(integration_dir, exist_ok=True)

        # Generate index file
        index_doc = generate_tool_index(tools, integration_name)
        index_file = f"{integration_dir}/_index.txt"
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(index_doc)

        print(f"Generated index file: {index_file}")

        # Generate individual tool files
        for tool in tools:
            tool_doc = generate_individual_tool_file(tool)
            tool_file = f"{integration_dir}/{tool.name}.txt"
            with open(tool_file, "w", encoding="utf-8") as f:
                f.write(tool_doc)

            print(f"Generated tool file: {tool_file}")

        print(
            f"\n=== Hybrid documentation structure created for {integration_name} ==="
        )
        print(f"Index: {index_file}")
        print(f"Individual tools: {len(tools)} files")

        # Also display tool info in console
        for i, tool in enumerate(tools, 1):
            print(f"{i}. Tool Name: {tool.name}")
            print(f"   Description: {tool.description}")
            print("-" * 80)

        return tools

    except Exception as e:
        print(f"Error discovering tools for {integration_name}: {e}")
        return []


async def main():
    """Main function to discover all available integrations"""
    integrations = get_integrations_list()

    if not integrations:
        print("No integrations found in gumcp_integrations.txt")
        return

    print(
        f"Found {len(integrations)} integrations to document: {', '.join(integrations)}"
    )

    # Handle command line arguments
    if len(sys.argv) > 1:
        selected_integration = sys.argv[1]
        if selected_integration in integrations:
            print(f"Selected integration: {selected_integration}")
            await discover_and_document_tools(selected_integration, GUMCP_CREDENTIALS)
        else:
            print(
                f"Integration '{selected_integration}' not found. Available: {', '.join(integrations)}"
            )
    else:
        # Generate documentation for all integrations by default
        print(f"Generating documentation for all integrations...")
        for integration in integrations:
            await discover_and_document_tools(integration, GUMCP_CREDENTIALS)


# Run the discovery and documentation
if __name__ == "__main__":
    asyncio.run(main())
