from deepagents import async_create_deep_agent, DeepAgentState
from langgraph.graph import StateGraph, START, END

# Import extracted modules
from src.core.model_config import get_model, load_system_prompt
from src.core.document_loader import add_gumcp_docs_to_state
from src.tools.code_executor import python_code_executor
from src.tools.flow_compiler import flow_compiler

# Get configured model and instructions
model = get_model()
instructions = load_system_prompt()


# Create the deepagent subgraph
deepagent_subgraph = async_create_deep_agent(
    tools=[python_code_executor, flow_compiler],
    instructions=instructions,
    model=model,
)

# Create parent graph that automatically adds gumcp docs
parent_builder = StateGraph(DeepAgentState)
parent_builder.add_node("add_gumcp_docs", add_gumcp_docs_to_state)
parent_builder.add_node("deepagent", deepagent_subgraph)

# Wire up the flow: START -> add_gumcp_docs -> deepagent -> END
parent_builder.add_edge(START, "add_gumcp_docs")
parent_builder.add_edge("add_gumcp_docs", "deepagent")
parent_builder.add_edge("deepagent", END)

# Compile the parent graph
agent = parent_builder.compile()
