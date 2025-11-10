"""Flow explanation engine for generating detailed explanations using LLM."""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from .flow_discovery import FlowMetadata
import traceback

load_dotenv()


class FlowExplainer:
    """Handles generating detailed explanations for compiled flows using GLM-4.6."""

    def __init__(self):
        """Initialize the flow explainer with GLM-4.6 model."""
        # Always use GLM-4.6 for explanation generation, regardless of main model configuration
        self.model = ChatOpenAI(
            temperature=0,
            model="glm-4.6",
            openai_api_key=os.getenv("ZAI_API_KEY"),
            openai_api_base="https://api.z.ai/api/coding/paas/v4/",
        )

    async def generate_explanation(self, flow_metadata: FlowMetadata) -> str:
        """
        Generate a comprehensive explanation for a flow using LLM.

        Args:
            flow_metadata (FlowMetadata): The metadata of the compiled flow

        Returns:
            str: Generated markdown explanation

        Raises:
            Exception: If explanation generation fails
        """
        try:
            # Prepare the flow information for the LLM
            flow_info = self._prepare_flow_info(flow_metadata)

            # Simple one-line prompt as requested
            prompt = f"Generate a comprehensive explanation of what this Python flow does, including its parameters, return values, and usage examples in markdown format.\n\n{flow_info}"

            # Generate explanation using the model
            messages = [HumanMessage(content=prompt)]

            response = await self.model.ainvoke(messages)
            explanation = response.content.strip()

            return explanation

        except Exception as e:
            error_msg = f"Failed to generate explanation: {str(e)}"
            print(f"Error in FlowExplainer: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            raise Exception(error_msg)

    def _prepare_flow_info(self, flow_metadata: FlowMetadata) -> str:
        """
        Prepare flow information as a formatted string for the LLM.

        Args:
            flow_metadata (FlowMetadata): The flow metadata

        Returns:
            str: Formatted flow information
        """
        info_parts = []

        # Basic flow information
        info_parts.append(f"Flow Name: {flow_metadata.name}")

        if flow_metadata.description:
            info_parts.append(f"Description: {flow_metadata.description}")

        # Parameters information
        if flow_metadata.parameters:
            info_parts.append("\nParameters:")
            for param in flow_metadata.parameters:
                param_info = f"- {param.name} ({param.type})"
                if param.required:
                    param_info += " (required)"
                if param.description:
                    param_info += f": {param.description}"
                if param.default is not None:
                    param_info += f" (default: {param.default})"
                info_parts.append(param_info)

        # Return type information
        if flow_metadata.return_type:
            info_parts.append(f"\nReturn Type: {flow_metadata.return_type}")

        # Source code
        if flow_metadata.source_code:
            info_parts.append(
                f"\nSource Code:\n```python\n{flow_metadata.source_code}\n```"
            )

        # Docstring if available
        if flow_metadata.docstring:
            info_parts.append(f"\nDocstring:\n{flow_metadata.docstring}")

        return "\n".join(info_parts)
