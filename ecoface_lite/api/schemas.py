"""Pydantic schemas — API boundary types decoupled from ORM models."""

from __future__ import annotations

from datetime import datetime

import json as _json

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PersonCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=255)
    notes: str | None = Field(default=None, max_length=4000)


class PersonOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    display_name: str
    notes: str | None
    source_image_path: str | None
    source_image_hash: str | None = None
    extra_photo_paths: list[str] = []
    created_at: datetime

    @field_validator("extra_photo_paths", mode="before")
    @classmethod
    def _decode_extra(cls, v: object) -> list[str]:
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = _json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []


class PersonEnrollOut(BaseModel):
    """POST /persons — existing row returned with 200 when upload bytes match a prior enrollment."""

    person: PersonOut
    deduplicated: bool = False


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    person_id: int | None
    person_name: str | None = None
    confidence: float
    threshold_used: float
    source_type: str
    source_label: str | None
    frame_index: int | None
    snapshot_path: str | None
    created_at: datetime


class VideoProcessRequest(BaseModel):
    """Path relative to configured `VIDEOS_DIR` or absolute path on server."""

    video_relative_path: str = Field(min_length=1, max_length=1024)


class ProcessingStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    video_label: str | None
    total_frames: int
    processed_frames: int
    alerts_created: int = 0
    avg_fps: float | None = None
    avg_confidence: float | None = None
    total_faces_detected: int = 0
    total_faces_rejected: int = 0
    blur_rejections: int = 0
    duplicate_suppressions: int = 0
    processing_duration_seconds: float | None = None
    status: str
    error_message: str | None
    created_at: datetime


class AsyncVideoJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    status_url: str


class LiveTestMatchResponse(BaseModel):
    matched: bool
    person_id: int | None = None
    person_name: str | None = None
    similarity_score: float | None = None
    threshold: float
    detail: str
    snapshot_path: str | None = None


class PersonEnrollMultiOut(BaseModel):
    """POST /persons/{person_id}/photos — add extra photos to existing person."""
    person: PersonOut
    photos_accepted: int
    photos_rejected: int
    rejection_reasons: list[str] = []


class PersonPhotoAddRequest(BaseModel):
    person_id: int


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str
    stream_url: str | None
    location: str | None
    is_active: bool
    created_at: datetime


class CameraCreate(BaseModel):
    label: str = Field(min_length=1, max_length=255)
    stream_url: str | None = Field(default=None, max_length=1024)
    location: str | None = Field(default=None, max_length=512)


class IncidentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ref: str = ''
    title: str
    description: str | None
    status: str
    operator_id: str | None
    is_paused: bool = False
    created_at: datetime
    updated_at: datetime
    person_count: int = 0
    alert_count: int = 0
    pending_alert_count: int = 0
    # Closure fields — null on open incidents
    closing_reason: str | None = None
    closing_summary: str | None = None
    closed_by: str | None = None
    closed_at: datetime | None = None
    evidence_paths: list[str] = []

    @field_validator("evidence_paths", mode="before")
    @classmethod
    def _decode_evidence(cls, v: object) -> list[str]:
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = _json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []


class IncidentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    description: str | None = Field(default=None, max_length=4000)
    operator_id: str | None = Field(default=None, max_length=128)


class IncidentStatusUpdate(BaseModel):
    status: str = Field(pattern="^(open|active|closed)$")


class IncidentPauseUpdate(BaseModel):
    is_paused: bool


class IncidentCloseRequest(BaseModel):
    reason: str = Field(pattern="^(person_found_safe|person_found_deceased|case_withdrawn|duplicate_case|other)$")
    summary: str = Field(min_length=1, max_length=4000)
    closed_by: str | None = Field(default=None, max_length=128)


class SightingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    incident_id: int
    detection_id: int | None
    camera_id: int | None
    notes: str | None
    status: str = 'pending'
    created_at: datetime
    # enriched fields populated by the endpoint via detection_event + person joins
    person_id: int | None = None
    person_name: str | None = None
    confidence: float | None = None
    source_name: str | None = None
    frame_index: int | None = None
    snapshot_path: str | None = None


class SightingStatusUpdate(BaseModel):
    status: str = Field(pattern="^(pending|confirmed|rejected)$")


class SightingCreate(BaseModel):
    incident_id: int
    detection_id: int | None = None
    camera_id: int | None = None
    notes: str | None = Field(default=None, max_length=4000)


class IncidentPersonOut(BaseModel):
    incident_id: int
    person_id: int
    person_name: str


class AlertOut(BaseModel):
    """Operator-facing alert session — one per continuous presence.

    level follows Phase 11 promotion ladder: sighting → candidate → verified → critical.
    source is "live" or "historical" (VSL Phase 4).
    zone_id is populated by VSL Phase 2 for cross-camera clustering (Phase 10).
    sightings list is only populated when fetching a single alert (detail view).
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    incident_id: int
    person_id: int
    camera_id: int | None
    zone_id: str | None
    status: str
    level: str
    source: str
    first_seen_at: datetime
    last_seen_at: datetime
    sighting_count: int
    operator_notes: str | None = None
    created_at: datetime
    updated_at: datetime
    # enriched fields (populated by endpoint)
    person_name: str | None = None
    camera_label: str | None = None
    incident_status: str | None = None
    sightings: list[SightingOut] = []


class AlertStatusUpdate(BaseModel):
    status: str = Field(pattern="^(open|closed|confirmed|rejected)$")


class AlertLevelUpdate(BaseModel):
    level: str = Field(pattern="^(sighting|candidate|verified|critical)$")


class AlertNoteCreate(BaseModel):
    note: str = Field(min_length=1, max_length=4000)
