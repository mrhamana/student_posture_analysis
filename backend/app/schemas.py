from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime
from uuid import UUID
from app.models import PostureClass, MediaType, SessionStatus


# --- Inference API Schemas ---


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedStudent(BaseModel):
    id: int
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    posture: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class InferenceFrameResult(BaseModel):
    frame_id: int
    students: List[DetectedStudent]


# --- Session Schemas ---


class SessionCreate(BaseModel):
    filename: str
    media_type: MediaType


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    media_type: MediaType
    status: SessionStatus
    total_frames: int
    processed_frames: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int


# --- Student Schemas ---


class StudentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tracker_id: int
    first_seen_frame: int
    last_seen_frame: int
    total_posture_changes: int


class StudentListResponse(BaseModel):
    students: List[StudentResponse]
    total: int


# --- Posture Record Schemas ---


class PostureRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    frame_id: int
    posture: PostureClass
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    timestamp: datetime


class PostureRecordListResponse(BaseModel):
    records: List[PostureRecordResponse]
    total: int


# --- Annotated Image Schemas ---


class AnnotatedDetection(BaseModel):
    tracker_id: int
    posture: str
    confidence: float
    bbox: List[float] = Field(..., min_length=4, max_length=4)


class AnnotatedImageMetadataResponse(BaseModel):
    frame_id: int
    detections: List[AnnotatedDetection]


# --- Posture Summary Schemas ---


class PostureDistribution(BaseModel):
    posture: str
    count: int
    percentage: float


class PostureSummaryResponse(BaseModel):
    session_id: UUID
    total_students: int
    total_frames: int
    posture_distribution: List[PostureDistribution]
    most_common_posture: Optional[str] = None
    average_confidence: float
    students_summary: List[Dict]


# --- Upload Response ---


class UploadResponse(BaseModel):
    session_id: UUID
    message: str
    status: SessionStatus


# --- Error Schemas ---


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


# --- Progress Schema ---


class ProcessingProgress(BaseModel):
    session_id: UUID
    status: SessionStatus
    total_frames: int
    processed_frames: int
    progress_percent: float
