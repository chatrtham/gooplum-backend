from deepagents import create_deep_agent
from dotenv import load_dotenv

# Import extracted modules
from src.core.model_config import get_model, load_system_prompt
from src.core.middleware import add_gumcp_docs
from src.core.subagents import get_subagents
from src.tools.code_executor import python_code_executor
from src.tools.flow_compiler import flow_compiler

load_dotenv()

# Get configured model and instructions
model = get_model()
instructions = load_system_prompt()

# Get subagents
subagents = get_subagents()

# Create the deep agent
agent = create_deep_agent(
    tools=[python_code_executor, flow_compiler],
    system_prompt=instructions,
    model=model,
    middleware=[add_gumcp_docs],
    subagents=subagents,
)
