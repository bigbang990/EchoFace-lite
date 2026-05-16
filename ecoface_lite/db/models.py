"""ORM models — SQLite today, PostgreSQL later.

Embeddings are stored as float32 bytes (BLOB); PostgreSQL can later use pgvector
without changing service APIs if embeddings stay as bytes at the DB boundary.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ecoface_lite.db.base import Base


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_image_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # SHA-256 hex of uploaded bytes; used for dedupe (partial unique index on SQLite via migration).
    source_image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    embeddings: Mapped[list[FaceEmbedding]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
    )
    detection_events: Mapped[list[DetectionEvent]] = relationship(back_populates="person")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    # Same as Person.source_image_hash when produced from that upload (supports dedupe lookup).
    ingest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    person: Mapped[Person] = relationship(back_populates="embeddings")


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_label: Mapped[str | None] = mapped_column(String(512), nullable=True)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    extra_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    person: Mapped[Person | None] = relationship(back_populates="detection_events")


class ProcessingStatus(Base):
    """Tracks long-running video jobs for UI polling (e.g. Streamlit progress bar)."""

    __tablename__ = "processing_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    video_label: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    alerts_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_faces_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_faces_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blur_rejections: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_suppressions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
