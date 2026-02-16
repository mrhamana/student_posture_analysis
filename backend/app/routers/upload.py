import os
import logging
import asyncio
from uuid import UUID
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import MediaType, SessionStatus
from app.schemas import UploadResponse, ErrorResponse
from app.services import SessionService, AnalysisService
from app.inference import inference_client

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _determine_media_type(content_type: str) -> MediaType:
    if content_type in settings.allowed_image_types_list:
        return MediaType.IMAGE
    elif content_type in settings.allowed_video_types_list:
        return MediaType.VIDEO
    raise ValueError(f"Unsupported content type: {content_type}")


async def _process_image(session_id: UUID, file_content: bytes, filename: str):
    """Background task: process a single image."""
    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            await SessionService.update_status(
                db, session_id, SessionStatus.PROCESSING, total_frames=1
            )

            result = await inference_client.analyze_image(file_content, filename)

            student_last_posture = {}
            await AnalysisService.process_frame_result(
                db, session_id, result, student_last_posture
            )
            await db.commit()

            await SessionService.update_status(
                db,
                session_id,
                SessionStatus.COMPLETED,
                processed_frames=1,
                total_frames=1,
            )
        except Exception as e:
            logger.exception(f"Image processing failed for session {session_id}")
            await SessionService.update_status(
                db, session_id, SessionStatus.FAILED, error_message=str(e)
            )


async def _process_video(session_id: UUID, file_content: bytes, filename: str):
    """Background task: process video with streaming inference."""
    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            await SessionService.update_status(db, session_id, SessionStatus.PROCESSING)

            student_last_posture = {}
            frame_count = 0
            batch_size = 10
            batch_count = 0

            async for frame_result in inference_client.analyze_video_stream(
                file_content, filename
            ):
                student_last_posture = await AnalysisService.process_frame_result(
                    db, session_id, frame_result, student_last_posture
                )
                frame_count += 1
                batch_count += 1

                # Batch commit every N frames to reduce DB overhead
                if batch_count >= batch_size:
                    await db.commit()
                    await SessionService.update_status(
                        db,
                        session_id,
                        SessionStatus.PROCESSING,
                        processed_frames=frame_count,
                    )
                    batch_count = 0

            # Final commit
            await db.commit()
            await SessionService.update_status(
                db,
                session_id,
                SessionStatus.COMPLETED,
                processed_frames=frame_count,
                total_frames=frame_count,
            )
        except Exception as e:
            logger.exception(f"Video processing failed for session {session_id}")
            await SessionService.update_status(
                db, session_id, SessionStatus.FAILED, error_message=str(e)
            )


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    summary="Upload image or video for posture analysis",
)
async def upload_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    # Validate content type
    content_type = file.content_type or ""
    if content_type not in settings.allowed_types_list:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: {settings.allowed_types_list}",
        )

    # Read and validate file size
    file_content = await file.read()
    if len(file_content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    media_type = _determine_media_type(content_type)
    filename = file.filename or "unknown"

    # Save file to disk
    file_path = os.path.join(UPLOAD_DIR, f"{os.urandom(8).hex()}_{filename}")
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Create session
    session = await SessionService.create_session(db, filename, media_type, file_path)

    # Process in background
    if media_type == MediaType.IMAGE:
        background_tasks.add_task(_process_image, session.id, file_content, filename)
    else:
        background_tasks.add_task(_process_video, session.id, file_content, filename)

    return UploadResponse(
        session_id=session.id,
        message=f"{'Image' if media_type == MediaType.IMAGE else 'Video'} uploaded successfully. Processing started.",
        status=SessionStatus.PENDING,
    )
