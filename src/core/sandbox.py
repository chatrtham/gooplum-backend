"""Sandbox management for code execution."""

from contextlib import asynccontextmanager
from e2b_code_interpreter import AsyncSandbox
import os


@asynccontextmanager
async def get_sandbox():
    """Context manager for sandbox lifecycle."""
    sandbox = None
    try:
        sandbox = await AsyncSandbox.create(
            envs={
                "GUMCP_CREDENTIALS": os.getenv("GUMCP_CREDENTIALS", ""),
                "ZAI_API_KEY": os.getenv("ZAI_API_KEY", ""),
            },
            timeout=3600,
        )  # 1 hour timeout
        yield sandbox
    finally:
        if sandbox:
            await sandbox.kill()


async def run_python_code(code: str):
    """Run python code in a sandbox and return the output."""
    async with get_sandbox() as sandbox:
        # Install required packages
        packages = [
            "langchain-mcp-adapters>=0.1.11",
            "langchain-openai>=0.3.35",
            "python-dotenv>=1.1.1",
        ]

        await sandbox.commands.run(f"pip install {' '.join(packages)}")

        result = await sandbox.run_code(code)
        return result
