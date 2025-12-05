"""Middleware and utilities for loading guMCP documentation."""

import asyncio
import aiofiles
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langchain.agents.middleware import (
    before_agent,
    AgentState,
)
from langgraph.runtime import Runtime


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: list[str]
    """Lines of the file."""

    created_at: str
    """ISO 8601 timestamp of file creation."""

    modified_at: str
    """ISO 8601 timestamp of last modification."""


async def load_gumcp_files_async() -> dict:
    """Load gumcp documentation files using non-blocking async operations."""
    gumcp_files = {}
    gumcp_docs_dir = Path("resources/gumcp_docs")

    # Use asyncio.to_thread for the directory existence check
    dir_exists = await asyncio.to_thread(gumcp_docs_dir.exists)
    if not dir_exists:
        print(f"Warning: {gumcp_docs_dir} directory not found")
        return gumcp_files

    # Use asyncio.to_thread for the glob operation (blocking)
    # Load both index files and individual tool files from hybrid structure
    file_paths = await asyncio.to_thread(list, gumcp_docs_dir.rglob("*.txt"))

    # Process files asynchronously
    for file_path in file_paths:
        if await asyncio.to_thread(file_path.is_file):
            filename = file_path.name
            try:
                # Use aiofiles for non-blocking file reading
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()

                # Get file stats for timestamps
                stat = await asyncio.to_thread(file_path.stat)
                timestamp = datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat()

                # Split content into lines and create FileData object
                lines = content.splitlines()
                # Store with relative path preserving directory structure for ls tool compatibility
                relative_path = file_path.relative_to(gumcp_docs_dir)
                # Convert to forward slashes for Linux environment
                relative_path_str = str(relative_path).replace("\\", "/")
                gumcp_path = f"/gumcp_docs/{relative_path_str}"
                gumcp_files[gumcp_path] = FileData(
                    content=lines, created_at=timestamp, modified_at=timestamp
                )
            except Exception as e:
                print(f"Warning: Could not read file {filename}: {e}")

    print(
        f"Loaded {len(gumcp_files)} guMCP documentation files: {list(gumcp_files.keys())}"
    )
    return gumcp_files


@before_agent
async def add_gumcp_docs(state: AgentState, runtime: Runtime) -> dict:
    """Add gumcp documentation to the state before agent execution.

    This decorator-based middleware loads gumcp documentation files and adds them
    to the state before the agent starts execution, similar to the previous
    add_gumcp_docs node. The documentation is only loaded if no files already
    exist in the state.

    Args:
        state: The current agent state

    Returns:
        Dict with files update if gumcp docs need to be loaded, empty dict otherwise
    """
    # Only add gumcp docs if no files exist
    if not state.get("files", {}):
        gumcp_files = await load_gumcp_files_async()
        return {"files": gumcp_files}

    return {}
