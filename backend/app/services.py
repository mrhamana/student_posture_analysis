import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Session,
    Student,
    PostureRecord,
    SessionStatus,
    PostureClass,
    MediaType,
)
from app.schemas import (
    InferenceFrameResult,
    SessionResponse,
    StudentResponse,
    PostureSummaryResponse,
    PostureDistribution,
    PostureRecordResponse,
)

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing analysis sessions."""

    @staticmethod
    async def create_session(
        db: AsyncSession, filename: str, media_type: MediaType, file_path: str
    ) -> Session:
        session = Session(
            filename=filename,
            media_type=media_type,
            status=SessionStatus.PENDING,
            file_path=file_path,
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session

    @staticmethod
    async def get_session(db: AsyncSession, session_id: UUID) -> Optional[Session]:
        result = await db.execute(select(Session).where(Session.id == session_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_sessions(
        db: AsyncSession, skip: int = 0, limit: int = 50
    ) -> Tuple[List[Session], int]:
        count_result = await db.execute(select(func.count(Session.id)))
        total = count_result.scalar_one()

        result = await db.execute(
            select(Session)
            .order_by(Session.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        sessions = list(result.scalars().all())
        return sessions, total

    @staticmethod
    async def update_status(
        db: AsyncSession,
        session_id: UUID,
        status: SessionStatus,
        error_message: Optional[str] = None,
        processed_frames: Optional[int] = None,
        total_frames: Optional[int] = None,
    ):
        values: Dict = {"status": status, "updated_at": datetime.utcnow()}
        if error_message is not None:
            values["error_message"] = error_message
        if processed_frames is not None:
            values["processed_frames"] = processed_frames
        if total_frames is not None:
            values["total_frames"] = total_frames

        await db.execute(
            update(Session).where(Session.id == session_id).values(**values)
        )
        await db.commit()

    @staticmethod
    async def delete_session(db: AsyncSession, session_id: UUID) -> bool:
        session = await SessionService.get_session(db, session_id)
        if session is None:
            return False
        await db.delete(session)
        await db.commit()
        return True


class StudentService:
    """Service for managing tracked students."""

    @staticmethod
    async def get_or_create_student(
        db: AsyncSession, session_id: UUID, tracker_id: int, frame_id: int
    ) -> Student:
        result = await db.execute(
            select(Student).where(
                Student.session_id == session_id,
                Student.tracker_id == tracker_id,
            )
        )
        student = result.scalar_one_or_none()

        if student is None:
            student = Student(
                session_id=session_id,
                tracker_id=tracker_id,
                first_seen_frame=frame_id,
                last_seen_frame=frame_id,
                total_posture_changes=0,
            )
            db.add(student)
            await db.flush()
        else:
            student.last_seen_frame = max(student.last_seen_frame, frame_id)

        return student

    @staticmethod
    async def get_students_by_session(
        db: AsyncSession, session_id: UUID
    ) -> List[Student]:
        result = await db.execute(
            select(Student)
            .where(Student.session_id == session_id)
            .order_by(Student.tracker_id)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_posture_changes(
        db: AsyncSession,
        student_id: int,
        previous_posture: Optional[str],
        new_posture: str,
    ):
        if previous_posture is not None and previous_posture != new_posture:
            await db.execute(
                update(Student)
                .where(Student.id == student_id)
                .values(total_posture_changes=Student.total_posture_changes + 1)
            )


class PostureRecordService:
    """Service for managing posture records."""

    @staticmethod
    async def batch_create_records(
        db: AsyncSession,
        records: List[Dict],
    ):
        """Batch insert posture records for efficiency."""
        if not records:
            return
        db_records = [PostureRecord(**r) for r in records]
        db.add_all(db_records)

    @staticmethod
    async def get_records_by_student(
        db: AsyncSession, student_id: int, limit: int = 100
    ) -> List[PostureRecord]:
        result = await db.execute(
            select(PostureRecord)
            .where(PostureRecord.student_id == student_id)
            .order_by(PostureRecord.frame_id.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_latest_records_by_session(
        db: AsyncSession, session_id: UUID
    ) -> List[PostureRecord]:
        """Get the latest posture record for each student in a session."""
        # Subquery to get the max frame_id per student
        subq = (
            select(
                PostureRecord.student_id,
                func.max(PostureRecord.frame_id).label("max_frame"),
            )
            .join(Student)
            .where(Student.session_id == session_id)
            .group_by(PostureRecord.student_id)
            .subquery()
        )

        result = await db.execute(
            select(PostureRecord).join(
                subq,
                (PostureRecord.student_id == subq.c.student_id)
                & (PostureRecord.frame_id == subq.c.max_frame),
            )
        )
        return list(result.scalars().all())


class AnalysisService:
    """Orchestrates the analysis pipeline: processes inference results and persists them."""

    @staticmethod
    async def process_frame_result(
        db: AsyncSession,
        session_id: UUID,
        frame_result: InferenceFrameResult,
        student_last_posture: Dict[int, str],
    ) -> Dict[int, str]:
        """
        Process a single frame's inference results.
        Returns updated student_last_posture map.
        """
        records_to_insert = []

        for det in frame_result.students:
            # Validate and normalize posture
            posture = det.posture
            try:
                PostureClass(posture)
            except ValueError:
                logger.warning(
                    f"Unknown posture class '{posture}' for student {det.id}, skipping"
                )
                continue

            # Skip low confidence (< 0.1)
            if det.confidence < 0.1:
                logger.debug(
                    f"Low confidence {det.confidence} for student {det.id}, skipping"
                )
                continue

            student = await StudentService.get_or_create_student(
                db, session_id, det.id, frame_result.frame_id
            )

            # Track posture changes
            prev_posture = student_last_posture.get(det.id)
            await StudentService.update_posture_changes(
                db, student.id, prev_posture, posture
            )
            student_last_posture[det.id] = posture

            records_to_insert.append(
                {
                    "student_id": student.id,
                    "frame_id": frame_result.frame_id,
                    "posture": posture,
                    "confidence": det.confidence,
                    "bbox_x1": det.bbox[0],
                    "bbox_y1": det.bbox[1],
                    "bbox_x2": det.bbox[2],
                    "bbox_y2": det.bbox[3],
                    "timestamp": datetime.utcnow(),
                }
            )

        # Batch insert posture records
        await PostureRecordService.batch_create_records(db, records_to_insert)

        return student_last_posture

    @staticmethod
    async def get_posture_summary(
        db: AsyncSession, session_id: UUID
    ) -> PostureSummaryResponse:
        """Generate posture summary for a session."""
        session = await SessionService.get_session(db, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        students = await StudentService.get_students_by_session(db, session_id)

        # Get posture distribution
        result = await db.execute(
            select(PostureRecord.posture, func.count(PostureRecord.id))
            .join(Student)
            .where(Student.session_id == session_id)
            .group_by(PostureRecord.posture)
        )
        posture_counts = result.all()

        total_records = sum(c for _, c in posture_counts)
        distribution = [
            PostureDistribution(
                posture=p.value if hasattr(p, "value") else str(p),
                count=c,
                percentage=round(
                    (c / total_records * 100) if total_records > 0 else 0, 2
                ),
            )
            for p, c in posture_counts
        ]
        distribution.sort(key=lambda x: x.count, reverse=True)

        most_common = distribution[0].posture if distribution else None

        # Average confidence
        avg_conf_result = await db.execute(
            select(func.avg(PostureRecord.confidence))
            .join(Student)
            .where(Student.session_id == session_id)
        )
        avg_confidence = avg_conf_result.scalar_one() or 0.0

        # Per-student summary
        students_summary = []
        for student in students:
            latest_records = await PostureRecordService.get_records_by_student(
                db, student.id, limit=1
            )
            current_posture = None
            current_confidence = 0.0
            if latest_records:
                val = latest_records[0].posture
                current_posture = val.value if hasattr(val, "value") else str(val)
                current_confidence = latest_records[0].confidence

            students_summary.append(
                {
                    "tracker_id": student.tracker_id,
                    "current_posture": current_posture,
                    "current_confidence": round(current_confidence, 4),
                    "total_posture_changes": student.total_posture_changes,
                    "first_seen_frame": student.first_seen_frame,
                    "last_seen_frame": student.last_seen_frame,
                }
            )

        return PostureSummaryResponse(
            session_id=session_id,
            total_students=len(students),
            total_frames=session.total_frames,
            posture_distribution=distribution,
            most_common_posture=most_common,
            average_confidence=round(float(avg_confidence), 4),
            students_summary=students_summary,
        )
