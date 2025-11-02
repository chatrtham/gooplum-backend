"""Configuration settings and environment management."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    def __init__(self):
        # API Configuration
        self.zai_api_key: Optional[str] = os.getenv("ZAI_API_KEY")
        self.gumcp_credentials: Optional[str] = os.getenv("GUMCP_CREDENTIALS")

        # Model Configuration
        self.model_name: str = "glm-4.6"
        self.model_temperature: float = 0.6
        self.model_api_base: str = "https://api.z.ai/api/coding/paas/v4/"

        # Path Configuration
        self.resources_dir: str = "resources"
        self.gumcp_docs_dir: str = os.path.join(self.resources_dir, "gumcp_docs")
        self.system_prompt_path: str = os.path.join(
            self.resources_dir, "system_prompt.md"
        )

    def validate(self) -> bool:
        """Validate that required settings are present."""
        if not self.zai_api_key:
            print("Warning: ZAI_API_KEY not set")
            return False

        if not self.gumcp_credentials:
            print("Warning: GUMCP_CREDENTIALS not set")

        return True


# Global settings instance
settings = Settings()
