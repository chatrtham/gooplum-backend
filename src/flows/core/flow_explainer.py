"""Flow explanation engine for generating detailed explanations using LLM."""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from .flow_discovery import FlowMetadata
import traceback

load_dotenv()


class FlowExplainer:
    """Handles generating detailed explanations for compiled flows using LLM."""

    def __init__(self):
        """Initialize the flow explainer with LLM."""
        self.model = ChatOpenAI(
            temperature=0.7,
            model="gemini-2.5-flash",
            openai_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            reasoning_effort=None,
            name="flow-explainer-llm",
            tags=["flow-explainer"],
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
            # Get only the source code from the flow metadata
            if not flow_metadata.source_code:
                raise Exception("No source code available in flow metadata")

            prompt = f"""
            Generate a super concise explanation of what this function does \
            step-by-step to non-technical users. 
            
            **Important**:
            - RETURN MARKDOWN ONLY. DO NOT INCLUDE ANY OTHER TEXT AND THE ```mardown ``` tags.
            - Use emojis where appropriate to make it easier to read.
            - Use examples or tables if helpful.
            - DO NOT lose details on important steps. By important steps, I mean steps that you want to know when you are using the flow.
            - DO NOT call it "function." Call it "flow."
            - DO NOT include any code.
            - DO NOT talk about reporting like streaming and final return values in the code because they won't see it when they run the flow.
            
            ```python
            {flow_metadata.source_code}
            ```
            """
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
