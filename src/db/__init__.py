"""Shared database utilities."""

from .client import SupabaseClient, get_supabase, get_supabase_client

__all__ = ["SupabaseClient", "get_supabase", "get_supabase_client"]
