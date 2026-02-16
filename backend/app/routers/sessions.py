import logging
import os
from uuid import UUID
from typing import Optional, Dict, Tuple

import cv2
from fastapi.responses import Response

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import SessionStatus, MediaType
from app.schemas import (
    SessionResponse,
    SessionListResponse,
    StudentListResponse,
    StudentResponse,
    PostureSummaryResponse,
    ProcessingProgress,
    PostureRecordListResponse,
    PostureRecordResponse,
    AnnotatedImageMetadataResponse,
    AnnotatedDetection,
    ErrorResponse,
)
from app.services import (
    SessionService,
    StudentService,
    PostureRecordService,
    AnalysisService,
)

logger = logging.getLogger(__name__)
router = APIRouter()

POSTURE_COLOR_HEX: Dict[str, str] = {
    "Listening": "#22c55e",
    "Looking Screen": "#3b82f6",
    "Hand Raising": "#f59e0b",
    "Sleeping / Head Down": "#ef4444",
    "Turning Back": "#a855f7",
    "Standing": "#ec4899",
    "Writing": "#06b6d4",
    "Reading": "#84cc16",
}


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


POSTURE_COLORS: Dict[str, Tuple[int, int, int]] = {
    name: _hex_to_bgr(color) for name, color in POSTURE_COLOR_HEX.items()
}


def _normalize_bbox(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    nx1 = max(0, min(int(x1), width - 1))
    ny1 = max(0, min(int(y1), height - 1))
    nx2 = max(0, min(int(x2), width - 1))
    ny2 = max(0, min(int(y2), height - 1))
    if nx2 <= nx1:
        nx2 = min(width - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(height - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _load_frame(session_file_path: str, media_type: MediaType, frame_id: int):
    if not os.path.exists(session_file_path):
        raise HTTPException(status_code=404, detail="Session media file not found")

    if media_type == MediaType.IMAGE:
        frame = cv2.imread(session_file_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to read image")
        return frame

    cap = cv2.VideoCapture(session_file_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to read video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise HTTPException(status_code=400, detail="Failed to extract video frame")
    return frame


def _encode_jpeg(frame) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return buffer.tobytes()


async def _get_annotation_payload(
    db: AsyncSession, session
) -> Tuple[int, Dict[int, int], list]:
    if session.media_type == MediaType.IMAGE:
        frame_id = 0
    else:
        frame_id = await PostureRecordService.get_latest_frame_id_by_session(db, session.id)
        if frame_id is None:
            raise HTTPException(status_code=404, detail="No posture records found")

    records = await PostureRecordService.get_records_by_session_and_frame(
        db, session.id, frame_id
    )
    if not records:
        raise HTTPException(status_code=404, detail="No posture records found")

    students = await StudentService.get_students_by_session(db, session.id)
    student_map = {s.id: s.tracker_id for s in students}
    return frame_id, student_map, records


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List all analysis sessions",
)
async def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    sessions, total = await SessionService.list_sessions(db, skip, limit)
    return SessionListResponse(
        sessions=[SessionResponse.model_validate(s) for s in sessions],
        total=total,
    )


@router.get(
    "/session/{session_id}",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get session details",
)
async def get_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    session = await SessionService.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse.model_validate(session)


@router.get(
    "/session/{session_id}/progress",
    response_model=ProcessingProgress,
    responses={404: {"model": ErrorResponse}},
    summary="Get session processing progress",
)
async def get_session_progress(session_id: UUID, db: AsyncSession = Depends(get_db)):
    session = await SessionService.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    total = session.total_frames or 0
    processed = session.processed_frames or 0
    progress = (processed / total * 100) if total > 0 else 0.0

    return ProcessingProgress(
        session_id=session.id,
        status=session.status,
        total_frames=total,
        processed_frames=processed,
        progress_percent=round(progress, 2),
    )


@router.get(
    "/session/{session_id}/students",
    response_model=StudentListResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get all students detected in a session",
)
async def get_session_students(session_id: UUID, db: AsyncSession = Depends(get_db)):
    session = await SessionService.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    students = await StudentService.get_students_by_session(db, session_id)
    return StudentListResponse(
        students=[StudentResponse.model_validate(s) for s in students],
        total=len(students),
    )


@router.get(
    "/session/{session_id}/posture-summary",
    response_model=PostureSummaryResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get posture analysis summary for a session",
)
async def get_posture_summary(session_id: UUID, db: AsyncSession = Depends(get_db)):
    try:
        summary = await AnalysisService.get_posture_summary(db, session_id)
        return summary
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/session/{session_id}/records",
    response_model=PostureRecordListResponse,
    summary="Get posture records for a session (latest per student)",
)
async def get_session_records(session_id: UUID, db: AsyncSession = Depends(get_db)):
    session = await SessionService.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    records = await PostureRecordService.get_latest_records_by_session(db, session_id)
    return PostureRecordListResponse(
        records=[PostureRecordResponse.model_validate(r) for r in records],
        total=len(records),
    )


@router.get(
    "/session/{session_id}/annotated-image",
    responses={404: {"model": ErrorResponse}},
    summary="Get annotated image for a session",
)
async def get_annotated_image(session_id: UUID, db: AsyncSession = Depends(get_db)):
    session = await SessionService.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Session is not completed")

    frame_id, student_map, records = await _get_annotation_payload(db, session)

    frame = _load_frame(session.file_path or "", session.media_type, frame_id)
    height, width = frame.shape[:2]

    for record in records:
        tracker_id = student_map.get(record.student_id)
        posture = record.posture.value if hasattr(record.posture, "value") else str(record.posture)
        color = POSTURE_COLORS.get(posture, (107, 114, 128))

        x1, y1, x2, y2 = _normalize_bbox(
            record.bbox_x1,
            record.bbox_y1,
            record.bbox_x2,
            record.bbox_y2,
            width,
            height,
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tracker_id} {posture} {record.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x1
        text_y = max(y1 - 6, text_h + 6)
        cv2.rectangle(
            frame,
            (text_x, text_y - text_h - 6),
            (text_x + text_w + 6, text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (text_x + 3, text_y - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    image_bytes = _encode_jpeg(frame)
    return Response(content=image_bytes, media_type="image/jpeg")


@router.get(
    "/session/{session_id}/annotated-metadata",
    response_model=AnnotatedImageMetadataResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get metadata for annotated image overlays",
)
async def get_annotated_metadata(session_id: UUID, db: AsyncSession = Depends(get_db)):
    session = await SessionService.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Session is not completed")

    frame_id, student_map, records = await _get_annotation_payload(db, session)

    detections = []
    for record in records:
        tracker_id = student_map.get(record.student_id)
        if tracker_id is None:
            continue
        posture = record.posture.value if hasattr(record.posture, "value") else str(record.posture)
        detections.append(
            AnnotatedDetection(
                tracker_id=tracker_id,
                posture=posture,
                confidence=float(record.confidence),
                bbox=[record.bbox_x1, record.bbox_y1, record.bbox_x2, record.bbox_y2],
            )
        )

    return AnnotatedImageMetadataResponse(frame_id=frame_id, detections=detections)


@router.get(
    "/session/{session_id}/students/{tracker_id}/crop",
    responses={404: {"model": ErrorResponse}},
    summary="Get cropped student image by tracker ID",
)
async def get_student_crop(
    session_id: UUID,
    tracker_id: int,
    db: AsyncSession = Depends(get_db),
):
    session = await SessionService.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Session is not completed")

    student = await StudentService.get_student_by_tracker_id(db, session_id, tracker_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")

    record = await PostureRecordService.get_latest_record_by_student(db, student.id)
    if record is None:
        raise HTTPException(status_code=404, detail="No posture record found for student")

    frame_id = 0 if session.media_type == MediaType.IMAGE else record.frame_id
    frame = _load_frame(session.file_path or "", session.media_type, frame_id)
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = _normalize_bbox(
        record.bbox_x1,
        record.bbox_y1,
        record.bbox_x2,
        record.bbox_y2,
        width,
        height,
    )
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise HTTPException(status_code=400, detail="Invalid crop region")

    image_bytes = _encode_jpeg(crop)
    return Response(content=image_bytes, media_type="image/jpeg")


@router.delete(
    "/session/{session_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Delete a session and all its data",
)
async def delete_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    deleted = await SessionService.delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}
