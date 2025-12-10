"""
Self-improvement middleware for agents to learn from conversations.

This middleware runs a background Learning LLM after each agent turn to detect
user feedback and automatically update the agent's instructions.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from langgraph_sdk import get_client
from pydantic import BaseModel, Field

load_dotenv()


class LearningAnalysis(BaseModel):
    """Structured output from the Learning LLM."""

    should_update: bool = Field(
        description="Whether the instructions should be updated based on this conversation"
    )
    updated_instructions: str = Field(
        default="",
        description="The full rewritten instructions with the learning applied. Empty if should_update is false.",
    )


class SelfImprovementMiddleware(AgentMiddleware):
    """
    Middleware that analyzes conversations and updates agent instructions.

    After each agent turn, this middleware:
    1. Passes the conversation + current instructions to a Learning LLM
    2. If the LLM detects actionable feedback, it rewrites the instructions
    3. Updates the assistant via LangGraph API (no user confirmation)

    The Learning LLM uses structured output to ensure consistent responses.
    """

    def __init__(self, assistant_id: str, current_instructions: str):
        """
        Initialize the self-improvement middleware.

        Args:
            assistant_id: The LangGraph assistant ID to update
            current_instructions: The current instructions for context
        """
        super().__init__()
        self.assistant_id = assistant_id
        self.current_instructions = current_instructions

        # Learning LLM - fast and cheap for background analysis
        self.learning_model = ChatOpenAI(
            temperature=0.7,
            model="gemini-2.5-flash",
            openai_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            reasoning_effort=None,
            name="learning-llm",
            tags=["self-improvement"],
        ).with_structured_output(LearningAnalysis)

    def _build_learning_prompt(self, messages: list) -> str:
        """Build the prompt for the Learning LLM."""
        # Format recent messages for analysis (last 10 messages max)
        recent_messages = messages[-10:] if len(messages) > 10 else messages

        # Exclude the last message (latest AI response) to avoid confusion
        messages_for_analysis = (
            recent_messages[:-1] if len(recent_messages) > 1 else recent_messages
        )
        formatted_messages = []

        for msg in messages_for_analysis:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            if content:  # Skip empty messages
                formatted_messages.append(f"{role}: {content}")

        conversation_text = "\n".join(formatted_messages)

        return f"""Analyze if the user gave NEW feedback that changes the assistant's instructions.

CURRENT INSTRUCTIONS:
{self.current_instructions or ""}

CONVERSATION:
{conversation_text}

Only set should_update=true if:
- User requests NEW behavior not already in instructions
- User corrects/contradicts existing instructions
- User shows frustration with current behavior

Set should_update=false if:
- User feedback already matches current instructions
- No behavioral feedback given
- Just casual conversation

If updating: rewrite instructions fully, replace conflicts with new feedback.
Use empty string ("") for cleaning up the instructions entirely if needed.
"""

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """
        Analyze conversation after agent completes and update instructions if needed.

        This runs after the agent has finished responding to the user.
        """
        try:
            messages = state.get("messages", [])

            # Skip if conversation is too short
            if len(messages) < 2:
                return None

            # Build prompt and analyze
            prompt = self._build_learning_prompt(messages)
            analysis: LearningAnalysis = await self.learning_model.ainvoke(prompt)

            # If no update needed, return early
            if not analysis.should_update:
                return None

            # Update the assistant's instructions via LangGraph API
            client = get_client()
            current_assistant = await client.assistants.get(self.assistant_id)

            # Merge new instructions into existing config
            current_config = current_assistant.get("config", {})
            current_configurable = current_config.get("configurable", {})

            updated_configurable = {
                **current_configurable,
                "instructions": analysis.updated_instructions.strip(),
            }

            await client.assistants.update(
                self.assistant_id,
                config={"configurable": updated_configurable},
            )

            # Update our local copy for future analysis in same session
            self.current_instructions = analysis.updated_instructions.strip()

            print(
                f"[SelfImprovement] Updated instructions for assistant {self.assistant_id}"
            )

        except Exception as e:
            # Don't fail the agent if learning fails - just log and continue
            print(f"[SelfImprovement] Error analyzing conversation: {e}")

        return None
