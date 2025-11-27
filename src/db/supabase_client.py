"""Supabase database client for flow persistence."""

import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from dotenv import load_dotenv
from supabase import acreate_client, AsyncClient
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from src.core.flow_discovery import FlowMetadata, FlowParameter


class FlowRecord(BaseModel):
    """Database model for flow records."""

    id: UUID
    name: str
    description: str
    source_code: str
    return_type: str
    docstring: Optional[str] = None
    explanation: Optional[str] = None
    created_at: datetime


class FlowParameterRecord(BaseModel):
    """Database model for flow parameter records."""

    id: UUID
    flow_id: UUID
    name: str
    type: str
    default_value: Optional[str] = None
    required: bool = True
    description: Optional[str] = None


class FlowRunRecord(BaseModel):
    """Database model for flow run records."""

    id: UUID
    flow_id: UUID
    parameters: Dict[str, Any]
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class SupabaseFlowDB:
    """Supabase database operations for flow management."""

    def __init__(self):
        """Initialize Supabase client configuration."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY environment variables must be set"
            )

        self.client: AsyncClient = None

    async def initialize(self):
        """Initialize the async Supabase client."""
        self.client = await acreate_client(self.supabase_url, self.supabase_key)

    async def _ensure_client(self):
        """Ensure the Supabase client is initialized."""
        if self.client is None:
            await self.initialize()

    async def create_flow(
        self, flow_metadata: FlowMetadata, source_code: str
    ) -> FlowRecord:
        """Create a new flow in the database."""
        await self._ensure_client()

        flow_data = {
            "id": flow_metadata.id,
            "name": flow_metadata.name,
            "description": flow_metadata.description,
            "source_code": source_code,
            "return_type": flow_metadata.return_type,
            "docstring": flow_metadata.docstring,
            "explanation": flow_metadata.explanation,
            "created_at": (
                flow_metadata.created_at or datetime.now(timezone.utc)
            ).isoformat(),
        }

        # Insert flow
        result = await self.client.table("flows").insert(flow_data).execute()

        if result.data:
            flow_record = FlowRecord(**result.data[0])

            # Insert parameters
            for param in flow_metadata.parameters:
                param_data = {
                    "flow_id": flow_metadata.id,
                    "name": param.name,
                    "type": param.type,
                    "default_value": param.default,
                    "required": param.required,
                    "description": param.description,
                }
                await self.client.table("flow_parameters").insert(param_data).execute()

            return flow_record
        else:
            raise Exception(f"Failed to create flow: {result}")

    async def get_flow(self, flow_id: UUID) -> Optional[FlowRecord]:
        """Get a flow by ID."""
        await self._ensure_client()
        result = (
            await self.client.table("flows")
            .select("*")
            .eq("id", str(flow_id))
            .execute()
        )

        if result.data:
            return FlowRecord(**result.data[0])
        return None

    async def get_flow_by_name(self, name: str) -> Optional[FlowRecord]:
        """Get a flow by name."""
        await self._ensure_client()
        result = await self.client.table("flows").select("*").eq("name", name).execute()

        if result.data:
            return FlowRecord(**result.data[0])
        return None

    async def list_flows(self) -> List[FlowRecord]:
        """List all flows."""
        await self._ensure_client()
        result = (
            await self.client.table("flows")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return [FlowRecord(**flow) for flow in result.data]

    async def update_flow(self, flow_id: UUID, updates: Dict[str, Any]) -> FlowRecord:
        """Update a flow."""
        await self._ensure_client()
        result = (
            await self.client.table("flows")
            .update(updates)
            .eq("id", str(flow_id))
            .execute()
        )

        if result.data:
            return FlowRecord(**result.data[0])
        else:
            raise Exception(f"Failed to update flow: {result}")

    async def delete_flow(self, flow_id: UUID) -> bool:
        """Delete a flow and its parameters."""
        await self._ensure_client()
        # Parameters will be deleted automatically due to CASCADE constraint
        result = (
            await self.client.table("flows").delete().eq("id", str(flow_id)).execute()
        )
        return len(result.data) > 0

    async def get_flow_parameters(self, flow_id: UUID) -> List[FlowParameterRecord]:
        """Get all parameters for a flow."""
        await self._ensure_client()
        result = (
            await self.client.table("flow_parameters")
            .select("*")
            .eq("flow_id", str(flow_id))
            .execute()
        )

        return [FlowParameterRecord(**param) for param in result.data]

    async def create_flow_run(
        self,
        flow_id: UUID,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FlowRunRecord:
        """Create a new flow run record with RUNNING status."""
        await self._ensure_client()

        run_data = {
            "flow_id": str(flow_id),
            "parameters": parameters,
            "status": "RUNNING",
            "metadata": metadata,
        }

        result = await self.client.table("flow_runs").insert(run_data).execute()

        if result.data:
            return FlowRunRecord(**result.data[0])
        else:
            raise Exception(f"Failed to create flow run: {result}")

    async def update_flow_run(
        self,
        run_id: UUID,
        status: str,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
    ) -> FlowRunRecord:
        """Update a flow run status and results."""
        await self._ensure_client()

        # Convert result to dict format for database storage
        formatted_result = None
        if result is not None:
            if isinstance(result, dict):
                formatted_result = result
            else:
                formatted_result = {"value": result, "type": type(result).__name__}

        updates = {
            "status": status,
            "result": formatted_result,
            "error": error,
            "execution_time_ms": execution_time_ms,
        }

        if status in ["COMPLETED", "FAILED", "CANCELLED"]:
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Remove None values to avoid overwriting with null
        updates = {k: v for k, v in updates.items() if v is not None}

        result_data = (
            await self.client.table("flow_runs")
            .update(updates)
            .eq("id", str(run_id))
            .execute()
        )

        if result_data.data:
            return FlowRunRecord(**result_data.data[0])
        else:
            raise Exception(f"Failed to update flow run: {result_data}")

    async def add_stream_event(
        self,
        run_id: UUID,
        event_type: str,
        payload: Dict[str, Any],
    ):
        """Add a stream event to the flow run."""
        await self._ensure_client()

        event_data = {
            "run_id": str(run_id),
            "event_type": event_type,
            "payload": payload,
        }

        await self.client.table("flow_stream_events").insert(event_data).execute()

    async def get_flow_runs(
        self, flow_id: UUID, limit: int = 10, offset: int = 0
    ) -> Tuple[List[FlowRunRecord], int]:
        """Get recent runs for a flow."""
        await self._ensure_client()
        result = (
            await self.client.table("flow_runs")
            .select("*", count="exact")
            .eq("flow_id", str(flow_id))
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return [FlowRunRecord(**run) for run in result.data], result.count

    async def get_flow_run(self, run_id: UUID) -> Optional[FlowRunRecord]:
        """Get a specific flow run."""
        await self._ensure_client()
        result = (
            await self.client.table("flow_runs")
            .select("*")
            .eq("id", str(run_id))
            .execute()
        )

        if result.data:
            return FlowRunRecord(**result.data[0])
        return None

    async def get_run_events(self, run_id: UUID) -> List[Dict[str, Any]]:
        """Get all stream events for a run."""
        await self._ensure_client()
        result = (
            await self.client.table("flow_stream_events")
            .select("*")
            .eq("run_id", str(run_id))
            .order("sequence_order", desc=False)
            .execute()
        )
        return result.data

    def flow_record_to_metadata(
        self, flow_record: FlowRecord, parameters: List[FlowParameterRecord]
    ) -> FlowMetadata:
        """Convert FlowRecord and parameters to FlowMetadata."""
        flow_params = [
            FlowParameter(
                name=param.name,
                type=param.type,
                default=param.default_value,
                required=param.required,
                description=param.description,
            )
            for param in parameters
        ]

        return FlowMetadata(
            id=str(flow_record.id),
            name=flow_record.name,
            description=flow_record.description,
            parameters=flow_params,
            return_type=flow_record.return_type,
            docstring=flow_record.docstring,
            source_code=flow_record.source_code,
            explanation=flow_record.explanation,
            created_at=flow_record.created_at,
        )


# Global database instance
_db_instance: Optional[SupabaseFlowDB] = None


def get_flow_db() -> SupabaseFlowDB:
    """Get the global flow database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SupabaseFlowDB()
    return _db_instance


async def get_initialized_flow_db() -> SupabaseFlowDB:
    """Get the global flow database instance and ensure it's initialized."""
    db = get_flow_db()
    await db.initialize()
    return db
