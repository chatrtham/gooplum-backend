from langchain.agents import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

from dotenv import load_dotenv

from src.core.model_config import get_model, load_system_prompt
from src.core.middleware import add_gumcp_docs, jump_to_end
from src.core.subagents import get_subagents
from src.tools.ask_user import ask_user
from src.tools.code_executor import python_code_executor
from src.tools.flow_compiler import flow_compiler

load_dotenv()

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

# Get configured model and instructions
model = get_model()
instructions = load_system_prompt()

# Get subagents
subagents = get_subagents()

# Create goopie agent
agent = create_agent(
    model=model,
    system_prompt=(
        instructions + "\n\n" + BASE_AGENT_PROMPT if instructions else BASE_AGENT_PROMPT
    ),
    tools=[ask_user, python_code_executor, flow_compiler],
    middleware=[
        FilesystemMiddleware(),
        SubAgentMiddleware(
            default_model=model,
            subagents=subagents,
            general_purpose_agent=False,
        ),
        jump_to_end,
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 17000),
            keep=("messages", 6),
            trim_tokens_to_summarize=None,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
        add_gumcp_docs,
    ],
)
