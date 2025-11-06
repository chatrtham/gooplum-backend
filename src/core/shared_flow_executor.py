"""Shared FlowExecutor instance for API and tool usage."""

from src.core.flow_executor import FlowExecutor

# Global shared instance
_shared_flow_executor = None


def get_shared_flow_executor() -> FlowExecutor:
    """
    Get the shared FlowExecutor instance.

    Returns:
        FlowExecutor: The shared flow executor instance
    """
    global _shared_flow_executor
    if _shared_flow_executor is None:
        _shared_flow_executor = FlowExecutor()
    return _shared_flow_executor


def reset_shared_flow_executor():
    """Reset the shared FlowExecutor instance (useful for testing)."""
    global _shared_flow_executor
    _shared_flow_executor = None
