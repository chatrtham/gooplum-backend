"""Tool for asking questions to the user during flow execution."""

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionItem(BaseModel):
    """A single question with optional suggested answers."""

    question: str = Field(description="A single question to ask the user.")
    suggested_answers: Optional[List[str]] = Field(
        default=None,
        description="Suggested answers (users can always type freely). Only include concrete, actionable options - no vague placeholders like 'Other', 'I'll specify', etc.",
    )


class AskUserInput(BaseModel):
    """Input for asking user questions."""

    questions: List[QuestionItem] = Field(
        description="List of questions, each with a question and optional suggested answers"
    )


@tool(
    args_schema=AskUserInput,
    description="Ask one or more questions to the user, optionally with suggested answers. DO NOT call this in parallel with other tools.",
)
async def ask_user(
    questions: List[QuestionItem],
) -> str:
    """
    Ask one or more questions to the user, optionally with suggested answers.

    Args:
        questions (List[Dict[str, Any]]): List of question objects, each containing:
            - question (str): The question to ask the user
            - suggested_answers (Optional[List[str]]): List of suggested answers

    Returns:
        Command: Updates the state with the questions asked.
    """
    answers_str = interrupt({"questions": questions})

    if not answers_str:
        return "No answers provided."

    return answers_str


__all__ = ["ask_user"]
