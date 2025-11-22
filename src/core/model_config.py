"""Model configuration and system prompt loading."""

import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def get_model():
    """Get the configured language model."""
    model = init_chat_model("anthropic:claude-sonnet-4-5-20250929", temperature=0)

    # model = ChatOpenAI(
    #     temperature=0,
    #     model="glm-4.6",
    #     openai_api_key=os.getenv("ZAI_API_KEY"),
    #     openai_api_base="https://api.z.ai/api/coding/paas/v4/",
    # )

    # model = ChatGoogleGenerativeAI(
    #     temperature=0,
    #     model="gemini-3-pro-preview",
    #     thinking_budget=1024,
    #     include_thoughts=True,
    # )
    
    return model


def load_system_prompt() -> str:
    """Load system prompt from the markdown file."""
    try:
        with open("resources/system_prompt.md", "r", encoding="utf-8") as f:
            content = f.read() + "\n\n" + "---"
        return content
    except Exception as e:
        print(f"Warning: Could not read resources/system_prompt.md: {e}")
        return ""  # Return empty string if file can't be read
