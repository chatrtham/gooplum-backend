"""Document loading utilities for guMCP documentation."""

import asyncio
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: list[str]
    """Lines of the file."""

    created_at: str
    """ISO 8601 timestamp of file creation."""

    modified_at: str
    """ISO 8601 timestamp of last modification."""


async def add_gumcp_docs_to_state(state) -> dict:
    """Automatically add gumcp documentation files to the state if they don't already exist."""
    existing_files = state.get("files", {})

    # Only add gumcp docs if no files exist
    if not existing_files:
        gumcp_files = await load_gumcp_files_async()
        return {"files": gumcp_files}

    # Files already exist, don't modify
    return {}


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
    file_paths = await asyncio.to_thread(list, gumcp_docs_dir.glob("gumcp*.txt"))

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
                timestamp = datetime.fromtimestamp(stat.st_mtime).isoformat()

                # Split content into lines and create FileData object
                lines = content.splitlines()
                # Store with absolute path prefix for ls tool compatibility
                file_path = f"/{filename}"
                gumcp_files[file_path] = FileData(
                    content=lines, created_at=timestamp, modified_at=timestamp
                )
            except Exception as e:
                print(f"Warning: Could not read file {filename}: {e}")

    print(
        f"Loaded {len(gumcp_files)} guMCP documentation files: {list(gumcp_files.keys())}"
    )
    return gumcp_files
