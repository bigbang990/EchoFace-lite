"""ORM models — SQLite today, PostgreSQL later.

Embeddings are stored as float32 bytes (BLOB); PostgreSQL can later use pgvector
without changing service APIs if embeddings stay as bytes at the DB boundary.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Table, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ecoface_lite.db.base import Base


incident_persons = Table(
    "incident_persons",
    Base.metadata,
    Column("incident_id", Integer, ForeignKey("incidents.id", ondelete="CASCADE"), primary_key=True),
    Column("person_id", Integer, ForeignKey("persons.id", ondelete="CASCADE"), primary_key=True),
)


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_image_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # SHA-256 hex of uploaded bytes; used for dedupe (partial unique index on SQLite via migration).
    source_image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # JSON-encoded list of additional enrolled photo paths (from POST /persons/{id}/photos)
    extra_photo_paths: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    person: Mapped[Person | None] = relationship(back_populates="detection_events")
    camera: Mapped["Camera | None"] = relationship()


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
    camera_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class Site(Base):
    """Physical deployment site (e.g. 'Main Campus', 'Airport Terminal 1').

    Sits at the top of the location hierarchy: Site → Zone → Camera.
    Country/State/District are reserved for VSL Phase 5 NVR integration;
    dissertation demo only requires Site and Zone.
    """
    __tablename__ = "sites"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    zones: Mapped[list["Zone"]] = relationship(back_populates="site", cascade="all, delete-orphan")


class Zone(Base):
    """Named area within a site (e.g. 'Main Entrance', 'Parking Lot A').

    Zone-aware alert clustering is introduced in Phase 10; VSL Phase 2 only
    establishes the schema so every camera carries zone context from day one.
    """
    __tablename__ = "zones"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    site: Mapped[Site] = relationship(back_populates="zones")
    cameras: Mapped[list["Camera"]] = relationship(back_populates="zone")


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    stream_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    location: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    # VSL Phase 1: source abstraction fields
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="file", server_default="file")
    zone: Mapped[str | None] = mapped_column(String(255), nullable=True)  # free-text fallback; use zone_id when possible
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown", server_default="unknown")
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # VSL Phase 2: normalized zone FK (nullable — existing cameras migrated gradually)
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    sightings: Mapped[list["Sighting"]] = relationship(back_populates="camera")
    zone_obj: Mapped["Zone | None"] = relationship("Zone", back_populates="cameras", foreign_keys=[zone_id])


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    operator_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_paused: Mapped[bool] = mapped_column(default=False, nullable=False)
    # Closure fields — populated atomically by POST /incidents/{id}/close
    closing_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    closing_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    closed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    evidence_paths: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list of uploaded file paths
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    sightings: Mapped[list["Sighting"]] = relationship(back_populates="incident", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="incident", cascade="all, delete-orphan")
    persons: Mapped[list["Person"]] = relationship(
        "Person",
        secondary=incident_persons,
        backref="incidents",
    )


class Sighting(Base):
    __tablename__ = "sightings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)
    detection_id: Mapped[int | None] = mapped_column(ForeignKey("detection_events.id", ondelete="SET NULL"), nullable=True)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    # Phase 8: alert session fields
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("alerts.id", ondelete="SET NULL"), nullable=True, index=True)
    person_id: Mapped[int | None] = mapped_column(ForeignKey("persons.id", ondelete="SET NULL"), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, server_default="live", default="live")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="pending", default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    incident: Mapped[Incident] = relationship(back_populates="sightings")
    camera: Mapped[Camera | None] = relationship(back_populates="sightings")
    detection: Mapped["DetectionEvent | None"] = relationship()
    alert: Mapped["Alert | None"] = relationship(back_populates="sightings")
    person: Mapped["Person | None"] = relationship()


class Alert(Base):
    """One Alert per continuous presence session (incident × person × camera).

    Phase 10 will cluster across cameras using zone_id.
    Phase 11 will promote level: sighting → candidate → verified → critical.
    VSL Phase 4 sets source="historical" for footage-search alerts.
    """
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String(128), nullable=True)  # Phase 10 cross-camera clustering
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    level: Mapped[str] = mapped_column(String(32), nullable=False, default="sighting")  # Phase 11 promotion
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="live")  # VSL Phase 4
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sighting_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    operator_notes: Mapped[str | None] = mapped_column(Text, nullable=True)  # append-only timestamped lines
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    incident: Mapped["Incident"] = relationship(back_populates="alerts")
    person: Mapped["Person"] = relationship()
    camera: Mapped["Camera | None"] = relationship()
    sightings: Mapped[list["Sighting"]] = relationship(back_populates="alert")
