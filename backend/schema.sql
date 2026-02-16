-- PostureAnalysis Database Schema
-- PostgreSQL

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(512) NOT NULL,
    media_type VARCHAR(10) NOT NULL CHECK (media_type IN ('image', 'video')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    total_frames INTEGER DEFAULT 0,
    processed_frames INTEGER DEFAULT 0,
    file_path TEXT,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_created_at ON sessions(created_at);

-- Students table (per session, tracked by BoT-SORT)
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    tracker_id INTEGER NOT NULL,
    first_seen_frame INTEGER DEFAULT 0,
    last_seen_frame INTEGER DEFAULT 0,
    total_posture_changes INTEGER DEFAULT 0,
    UNIQUE(session_id, tracker_id)
);

CREATE INDEX idx_students_session_id ON students(session_id);
CREATE INDEX idx_students_session_tracker ON students(session_id, tracker_id);

-- Posture records (one per student per frame)
CREATE TABLE posture_records (
    id SERIAL PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL,
    posture VARCHAR(30) NOT NULL CHECK (posture IN (
        'Listening', 'Looking Screen', 'Hand Raising',
        'Sleeping / Head Down', 'Turning Back', 'Standing',
        'Writing', 'Reading'
    )),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_posture_student_frame ON posture_records(student_id, frame_id);
CREATE INDEX idx_posture_student_id ON posture_records(student_id);
CREATE INDEX idx_posture_posture ON posture_records(posture);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
