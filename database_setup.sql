-- GoopLum Backend Database Setup
-- Run these commands in your Supabase SQL editor to set up the database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- FLOWS TABLES
-- ============================================

-- Main flows table
CREATE TABLE IF NOT EXISTS flows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    source_code TEXT NOT NULL,
    return_type VARCHAR(100) NOT NULL,
    docstring TEXT,
    explanation TEXT,
    status VARCHAR(50) DEFAULT 'draft', -- 'draft' or 'ready'
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

-- Flow runs table (formerly flow_executions)
CREATE TABLE IF NOT EXISTS flow_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flow_id UUID NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
    parameters JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'RUNNING', -- 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    result JSONB,
    error TEXT,
    execution_time_ms INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- Flow stream events table
CREATE TABLE IF NOT EXISTS flow_stream_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES flow_runs(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL, -- 'item', 'step', 'log', 'error'
    payload JSONB NOT NULL,          -- The actual data
    sequence_order SERIAL,           -- To keep them in order
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_flows_name ON flows(name);
CREATE INDEX IF NOT EXISTS idx_flows_created_at ON flows(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flows_status ON flows(status);
CREATE INDEX IF NOT EXISTS idx_flow_parameters_flow_id ON flow_parameters(flow_id);
CREATE INDEX IF NOT EXISTS idx_flow_runs_flow_id ON flow_runs(flow_id);
CREATE INDEX IF NOT EXISTS idx_flow_runs_created_at ON flow_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flow_runs_status ON flow_runs(status);
CREATE INDEX IF NOT EXISTS idx_stream_events_run_id ON flow_stream_events(run_id);


-- Optional: Add comments for documentation
COMMENT ON TABLE flows IS 'Stores compiled flow definitions and metadata';
COMMENT ON TABLE flow_parameters IS 'Stores parameters for each flow';
COMMENT ON TABLE flow_runs IS 'Stores execution history and results for flows';
COMMENT ON TABLE flow_stream_events IS 'Stores individual stream events for a flow run';