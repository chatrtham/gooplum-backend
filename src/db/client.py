"""Shared Supabase client for database operations."""

import os
from typing import Optional

from dotenv import load_dotenv
from supabase import acreate_client, AsyncClient

# Load environment variables
load_dotenv()


class SupabaseClient:
    """Base Supabase client with connection management."""

    _instance: Optional["SupabaseClient"] = None
    _client: Optional[AsyncClient] = None

    def __init__(self):
        """Initialize Supabase client configuration."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY environment variables must be set"
            )

    async def get_client(self) -> AsyncClient:
        """Get or create the async Supabase client."""
        if self._client is None:
            self._client = await acreate_client(self.supabase_url, self.supabase_key)
        return self._client

    @classmethod
    def get_instance(cls) -> "SupabaseClient":
        """Get the singleton instance of SupabaseClient."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_supabase() -> SupabaseClient:
    """Get the global Supabase client instance."""
    return SupabaseClient.get_instance()


async def get_supabase_client() -> AsyncClient:
    """Get the initialized async Supabase client."""
    return await get_supabase().get_client()
