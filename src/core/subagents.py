from langchain.agents import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from deepagents import CompiledSubAgent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

from dotenv import load_dotenv

from src.core.model_config import load_discovery_prompt, get_discovery_model
from src.tools.code_executor import python_code_executor

load_dotenv()


def create_discovery_subagent():
    """Create the discovery subagent for exploring guMCP services"""
    discovery_instructions = load_discovery_prompt()

    discovery_subagent = CompiledSubAgent(
        name="gumcp-discovery-agent",
        description="Specialize guMCP subagent exploring external services (guMCP), understanding data structures, and testing tool capabilities for you.",
        runnable=create_agent(
            model=get_discovery_model(),
            tools=[python_code_executor],
            system_prompt=discovery_instructions,
            middleware=[
                FilesystemMiddleware(),
                SummarizationMiddleware(
                    model=get_discovery_model(),
                    trigger=("tokens", 17000),
                    keep=("messages", 6),
                    trim_tokens_to_summarize=None,
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
        ),
    )

    return discovery_subagent


def get_subagents():
    """Get all available subagents"""
    subagents = [
        create_discovery_subagent(),
    ]
    return subagents
