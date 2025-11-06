"""Document loading utilities for guMCP documentation."""

import os
import glob
import aiofiles
import asyncio
from deepagents import DeepAgentState


async def add_gumcp_docs_to_state(state: DeepAgentState) -> DeepAgentState:
    """Automatically add gumcp documentation files to the state if they don't already exist."""
    existing_files = state.get("files", {})

    # Only add gumcp docs if no files exist
    if not existing_files:
        gumcp_files = await load_gumcp_files()
        return {"files": gumcp_files}

    # Files already exist, don't modify
    return {}


async def load_gumcp_files() -> dict:
    """Dynamically load gumcp documentation files from the gumcp_docs directory."""
    gumcp_files = {}
    gumcp_docs_dir = "resources/gumcp_docs"  # Updated path

    # Use asyncio.to_thread to run the blocking os.path.exists in a thread
    dir_exists = await asyncio.to_thread(os.path.exists, gumcp_docs_dir)
    if not dir_exists:
        print(f"Warning: {gumcp_docs_dir} directory not found")
        return gumcp_files

    # Use asyncio.to_thread for the blocking glob operation
    file_paths = await asyncio.to_thread(
        glob.glob, os.path.join(gumcp_docs_dir, "gumcp*.txt")
    )

    for file_path in file_paths:
        # Check if file exists using asyncio.to_thread
        is_file = await asyncio.to_thread(os.path.isfile, file_path)
        if is_file:
            filename = os.path.basename(file_path)
            try:
                # Use aiofiles for async file reading
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                gumcp_files[filename] = content
            except Exception as e:
                print(f"Warning: Could not read file {filename}: {e}")

    print(
        f"Loaded {len(gumcp_files)} guMCP documentation files: {list(gumcp_files.keys())}"
    )
    return gumcp_files
