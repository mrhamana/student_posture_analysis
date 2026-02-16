import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Enum as SAEnum,
    Text,
    Index,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


class MediaType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"


class SessionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PostureClass(str, enum.Enum):
    LISTENING = "Listening"
    LOOKING_SCREEN = "Looking Screen"
    HAND_RAISING = "Hand Raising"
    SLEEPING = "Sleeping / Head Down"
    TURNING_BACK = "Turning Back"
    STANDING = "Standing"
    WRITING = "Writing"
    READING = "Reading"


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    media_type = Column(SAEnum(MediaType), nullable=False)
    status = Column(
        SAEnum(SessionStatus), default=SessionStatus.PENDING, nullable=False
    )
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    file_path = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    students = relationship(
        "Student", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_sessions_status", "status"),
        Index("idx_sessions_created_at", "created_at"),
    )


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        Uuid,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    tracker_id = Column(Integer, nullable=False)  # BoT-SORT assigned ID
    first_seen_frame = Column(Integer, default=0)
    last_seen_frame = Column(Integer, default=0)
    total_posture_changes = Column(Integer, default=0)

    session = relationship("Session", back_populates="students")
    posture_records = relationship(
        "PostureRecord", back_populates="student", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_students_session_tracker", "session_id", "tracker_id", unique=True),
        Index("idx_students_session_id", "session_id"),
    )


class PostureRecord(Base):
    __tablename__ = "posture_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(
        Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    frame_id = Column(Integer, nullable=False)
    posture = Column(SAEnum(PostureClass), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    student = relationship("Student", back_populates="posture_records")

    __table_args__ = (
        Index("idx_posture_student_frame", "student_id", "frame_id"),
        Index("idx_posture_student_id", "student_id"),
        Index("idx_posture_posture", "posture"),
    )
