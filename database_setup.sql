-- GraphLoop Backend Database Setup
-- Run these commands in your Supabase SQL editor to set up the database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Main flows table (removed redundant execution tracking columns)
CREATE TABLE IF NOT EXISTS flows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    source_code TEXT NOT NULL,
    return_type VARCHAR(100) NOT NULL,
    docstring TEXT,
    explanation TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Flow parameters table
CREATE TABLE IF NOT EXISTS flow_parameters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flow_id UUID NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    type VARCHAR(100) NOT NULL,
    default_value TEXT,
    required BOOLEAN DEFAULT true,
    description TEXT,

    -- Ensure each parameter name is unique within a flow
    UNIQUE(flow_id, name)
);

-- Flow execution history table (enhanced with status lifecycle tracking)
CREATE TABLE IF NOT EXISTS flow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flow_id UUID NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
    parameters JSONB NOT NULL,
    success BOOLEAN NOT NULL,

    -- Execution lifecycle status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Results and timing
    result JSONB,
    error TEXT,
    execution_time_ms INTEGER,

    -- Optional streaming data (for individual item tracking within executions)
    streams JSONB,

    -- Additional metadata and audit
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_flows_name ON flows(name);
CREATE INDEX IF NOT EXISTS idx_flows_created_at ON flows(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flow_parameters_flow_id ON flow_parameters(flow_id);

-- Enhanced indexes for flow_executions with new status fields
CREATE INDEX IF NOT EXISTS idx_flow_executions_flow_id ON flow_executions(flow_id);
CREATE INDEX IF NOT EXISTS idx_flow_executions_created_at ON flow_executions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flow_executions_status ON flow_executions(status);
CREATE INDEX IF NOT EXISTS idx_flow_executions_flow_status ON flow_executions(flow_id, status);

-- Optional: Add comments for documentation
COMMENT ON TABLE flows IS 'Stores compiled flow definitions and metadata';
COMMENT ON TABLE flow_parameters IS 'Stores parameters for each flow';
COMMENT ON TABLE flow_executions IS 'Stores execution history and results for flows';