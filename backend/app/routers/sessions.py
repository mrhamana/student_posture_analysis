import logging
from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import SessionStatus
from app.schemas import (
    SessionResponse,
    SessionListResponse,
    StudentListResponse,
    StudentResponse,
    PostureSummaryResponse,
    ProcessingProgress,
    PostureRecordListResponse,
    PostureRecordResponse,
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
