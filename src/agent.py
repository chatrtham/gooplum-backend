from deepagents import create_deep_agent

# Import extracted modules
from src.core.model_config import get_model, load_system_prompt
from src.core.middleware import add_gumcp_docs
from src.tools.code_executor import python_code_executor
from src.tools.flow_compiler import flow_compiler

# Get configured model and instructions
model = get_model()
instructions = load_system_prompt()

# Create the deep agent with decorator-based middleware for gumcp docs
agent = create_deep_agent(
    tools=[python_code_executor, flow_compiler],
    system_prompt=instructions,
    model=model,
    middleware=[add_gumcp_docs],
)
