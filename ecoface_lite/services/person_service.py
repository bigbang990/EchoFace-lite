"""Business logic for persons and embeddings."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ecoface_lite.core.config import Settings
from ecoface_lite.core.logging import get_logger
from ecoface_lite.db.models import FaceEmbedding, Person

if TYPE_CHECKING:
    import numpy as np

    from ecoface_lite.ai_engine.pipeline import RecognitionPipeline

logger = get_logger(__name__)


@dataclass
class EnrollmentConflictError(Exception):
    """Raised when a new enrollment closely matches a person already in an open incident."""
    person_id: int
    person_name: str
    incident_id: int
    incident_ref: str
    incident_title: str
    incident_status: str
    incident_opened_at: datetime
    similarity: float


def _validate_enrollment_image(pipeline: "RecognitionPipeline", image: "np.ndarray") -> None:
    """Raise ValueError with a specific message if the image is unsuitable for enrollment.

    Checks are ordered cheapest-first:
      0 faces  → reject (no signal)
      >1 faces → reject (ambiguous identity — operator must crop to one face)
    Quality gate happens inside enroll_reference_embedding after this passes.
    """
    n = pipeline.count_enrollment_faces(image)
    if n == 0:
        raise ValueError("No face detected in reference photo")
    if n > 1:
        raise ValueError(f"Multiple faces detected ({n}) — crop to one face per photo")


def sha256_hex(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


async def _find_person_by_ingest_hash(session: AsyncSession, digest: str) -> Person | None:
    """Match prior enrollment by stored person hash or legacy embedding ingest hash."""
    r1 = await session.execute(select(Person).where(Person.source_image_hash == digest).limit(1))
    found = r1.scalar_one_or_none()
    if found is not None:
        return found

    r2 = await session.execute(
        select(Person)
        .join(FaceEmbedding, FaceEmbedding.person_id == Person.id)
        .where(FaceEmbedding.ingest_sha256 == digest)
        .limit(1)
    )
    return r2.scalar_one_or_none()


async def _check_identity_conflict(
    session: AsyncSession,
    new_embedding: "np.ndarray",
    threshold: float,
) -> None:
    """Raise EnrollmentConflictError if the embedding matches a person in an open incident.

    ArcFace embeddings are L2-normalised so cosine similarity == dot product.
    Only checks persons currently linked to at least one open, non-paused incident.
    """
    import numpy as np

    from ecoface_lite.db.models import Incident, incident_persons

    rows = (await session.execute(
        select(
            FaceEmbedding.person_id,
            FaceEmbedding.embedding,
            Person.display_name,
            Incident.id.label("incident_id"),
            Incident.title.label("incident_title"),
            Incident.status.label("incident_status"),
            Incident.created_at.label("incident_opened_at"),
        )
        .join(Person, Person.id == FaceEmbedding.person_id)
        .join(incident_persons, incident_persons.c.person_id == Person.id)
        .join(Incident, Incident.id == incident_persons.c.incident_id)
        .where(Incident.status == "open")
        .where(Incident.is_paused == False)
    )).all()

    best_sim = 0.0
    best_row = None
    for row in rows:
        vec = np.frombuffer(row.embedding, dtype=np.float32)
        sim = float(np.dot(new_embedding, vec))
        if sim > best_sim:
            best_sim = sim
            best_row = row

    if best_sim >= threshold and best_row is not None:
        inc_id = int(best_row.incident_id)
        raise EnrollmentConflictError(
            person_id=int(best_row.person_id),
            person_name=str(best_row.display_name),
            incident_id=inc_id,
            incident_ref=f"INC-{inc_id:03d}",
            incident_title=str(best_row.incident_title),
            incident_status=str(best_row.incident_status),
            incident_opened_at=best_row.incident_opened_at,
            similarity=best_sim,
        )


async def list_persons(session: AsyncSession) -> list[Person]:
    result = await session.execute(select(Person).order_by(Person.id.desc()))
    return list(result.scalars().all())


async def create_person_from_image(
    session: AsyncSession,
    pipeline: RecognitionPipeline,
    settings: Settings,
    *,
    file_bytes: bytes,
    original_filename: str,
    display_name: str,
    notes: str | None,
    skip_conflict_check: bool = False,
    force_enroll: bool = False,
) -> tuple[Person, bool]:
    """Create a person + embedding, or return an existing person when bytes match a prior upload.

    Returns (person, deduplicated).
    """
    import cv2
    import numpy as np

    digest = sha256_hex(file_bytes)
    existing = await _find_person_by_ingest_hash(session, digest)
    if existing is not None:
        logger.info("Enrollment dedupe hit hash=%s person_id=%s", digest[:12], existing.id)
        return existing, True

    uploads = settings.resolved_uploads_dir()
    uploads.mkdir(parents=True, exist_ok=True)
    ext = Path(original_filename).suffix.lower() or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = uploads / stored_name
    stored_path.write_bytes(file_bytes)

    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or corrupted image file")

    _validate_enrollment_image(pipeline, image)
    try:
        embedding = pipeline.enroll_reference_embedding(image, enrollment_mode=force_enroll)
    except ValueError:
        raise  # quality rejection — already has a specific message

    if not skip_conflict_check:
        await _check_identity_conflict(session, embedding, settings.enrollment_conflict_threshold)

    rel_upload = str(Path("data/uploads") / stored_name)

    person = Person(
        display_name=display_name,
        notes=notes,
        source_image_path=rel_upload,
        source_image_hash=digest,
    )
    session.add(person)
    await session.flush()

    face = FaceEmbedding(
        person_id=person.id,
        ingest_sha256=digest,
        embedding=embedding.astype(np.float32).tobytes(),
        embedding_dim=int(embedding.shape[0]),
        model_name=settings.insightface_model_name,
    )
    session.add(face)
    await session.flush()
    await session.refresh(person)
    logger.info("Enrolled person id=%s name=%s", person.id, display_name)
    return person, False


async def add_photos_to_person(
    db: AsyncSession,
    pipeline: RecognitionPipeline,
    settings: Settings,
    person_id: int,
    files: list[bytes],
    filenames: list[str],
    force_enroll: bool = False,
) -> tuple[int, int, list[str]]:
    """Enroll additional reference photos for an existing person.

    Returns (accepted_count, rejected_count, rejection_reasons).
    Raises HTTPException 400 if more than 5 photos are submitted.
    """
    import json
    import cv2
    import numpy as np
    from fastapi import HTTPException

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Max 5 photos per call")

    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    accepted = 0
    rejected = 0
    reasons: list[str] = []
    new_paths: list[str] = []

    uploads = settings.resolved_uploads_dir()
    uploads.mkdir(parents=True, exist_ok=True)

    for file_bytes, filename in zip(files, filenames):
        digest = sha256_hex(file_bytes)

        dup = await db.execute(
            select(FaceEmbedding)
            .where(FaceEmbedding.person_id == person_id, FaceEmbedding.ingest_sha256 == digest)
            .limit(1)
        )
        if dup.scalar_one_or_none() is not None:
            rejected += 1
            reasons.append(f"{filename}: duplicate (already enrolled for this person)")
            continue

        buf = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            rejected += 1
            reasons.append(f"{filename}: invalid or corrupted image file")
            continue

        try:
            _validate_enrollment_image(pipeline, image)
            embedding = pipeline.enroll_reference_embedding(image, enrollment_mode=force_enroll)
        except ValueError as exc:
            rejected += 1
            reasons.append(f"{filename}: {exc}")
            continue

        # Save image file to uploads dir so it can be shown in the gallery
        ext = Path(filename).suffix.lower() or ".jpg"
        stored_name = f"{uuid.uuid4().hex}{ext}"
        (uploads / stored_name).write_bytes(file_bytes)
        rel_path = str(Path("data/uploads") / stored_name)
        new_paths.append(rel_path)

        face = FaceEmbedding(
            person_id=person_id,
            ingest_sha256=digest,
            embedding=embedding.astype(np.float32).tobytes(),
            embedding_dim=int(embedding.shape[0]),
            model_name=settings.insightface_model_name,
        )
        db.add(face)
        await db.flush()
        accepted += 1
        logger.info("Added photo for person id=%s hash=%s path=%s", person_id, digest[:12], rel_path)

    if new_paths:
        existing = json.loads(person.extra_photo_paths) if person.extra_photo_paths else []
        person.extra_photo_paths = json.dumps(existing + new_paths)
        await db.flush()

    return accepted, rejected, reasons


# ── Multi-face enrollment with operator face picker ───────────────────────────

@dataclass
class DetectedEnrollmentFace:
    face_id: str
    bbox: tuple[float, float, float, float]
    det_score: float
    pose_bucket: str
    quality_score: float | None
    thumbnail_b64: str
    embedding: "np.ndarray"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_jpeg_b64(image: "np.ndarray", max_width: int = 200) -> str | None:
    """Base64-encoded JPEG of `image`, downscaled so the wider side is <= max_width."""
    try:
        import base64
        import cv2
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(image, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode()
    except Exception:
        return None


def _crop_face_thumbnail(image_bgr: "np.ndarray", bbox: tuple[float, float, float, float], pad_ratio: float = 0.20) -> "np.ndarray | None":
    """Crop a face region from the original-resolution image with padding for context."""
    import numpy as np
    try:
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_ratio, bh * pad_ratio
        cx1 = max(0, int(x1 - px))
        cy1 = max(0, int(y1 - py))
        cx2 = min(w, int(x2 + px))
        cy2 = min(h, int(y2 + py))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return image_bgr[cy1:cy2, cx1:cx2].copy()
    except Exception:
        return None


def _compute_face_id(image_hash: str, bbox: tuple[float, float, float, float]) -> str:
    """Content-addressable ID — same image + same bbox always yields same id."""
    x1, y1, x2, y2 = bbox
    payload = f"{image_hash}:{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def detect_faces_in_batch(
    pipeline: "RecognitionPipeline",
    files: list[bytes],
    filenames: list[str],
) -> list[dict]:
    """Detect all faces in each uploaded photo. No DB writes, no face selection.

    Returns a list of dicts (one per input file) with shape matching PhotoDetectionResult.
    The operator picks ONE face per photo from these candidates in the frontend.
    """
    import cv2
    import numpy as np
    from ecoface_lite.ai_engine.geometry import compute_face_geometry
    from ecoface_lite.ai_engine.pose_estimator import classify_pose_bucket

    results: list[dict] = []

    for idx, (file_bytes, filename) in enumerate(zip(files, filenames)):
        image_hash = _hash_bytes(file_bytes)
        buf = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            results.append({
                "photo_index": idx, "filename": filename, "image_hash": image_hash,
                "status": "invalid", "reason": "could not decode image",
                "photo_thumbnail_b64": None, "faces": [],
            })
            continue

        photo_thumb = _make_jpeg_b64(image, max_width=200)

        try:
            detected = pipeline.detect_enrollment_faces(image, min_det_score=0.50)
        except Exception as exc:
            results.append({
                "photo_index": idx, "filename": filename, "image_hash": image_hash,
                "status": "invalid", "reason": f"detection error: {exc}",
                "photo_thumbnail_b64": photo_thumb, "faces": [],
            })
            continue

        if not detected:
            results.append({
                "photo_index": idx, "filename": filename, "image_hash": image_hash,
                "status": "no_face", "reason": "no face detected in this photo",
                "photo_thumbnail_b64": photo_thumb, "faces": [],
            })
            continue

        # Sort by face area descending → the largest face is usually the subject in a portrait
        # and in groups it lets the operator see the prominent faces first.
        faces_with_area = []
        for face in detected:
            bb = face.bbox
            area = max(0.0, (bb.x2 - bb.x1) * (bb.y2 - bb.y1))
            faces_with_area.append((area, face))
        faces_with_area.sort(key=lambda x: x[0], reverse=True)

        face_out_list = []
        for rank, (_, face) in enumerate(faces_with_area):
            bb = face.bbox
            bbox_tuple = (float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2))
            face_id = _compute_face_id(image_hash, bbox_tuple)

            # Pose classification
            try:
                pose = classify_pose_bucket(face.landmarks, face.bbox) if face.landmarks is not None else None
                pose_value = pose.value if pose is not None else "unknown"
            except Exception:
                pose_value = "unknown"

            # Quality score (geometric proxy — width/height/det_score blend, no hard gate)
            try:
                geom = compute_face_geometry(face, image.shape)
                w_ratio = min(1.0, geom.width / 200.0)
                h_ratio = min(1.0, geom.height / 200.0)
                quality = float(min(1.0, 0.4 * face.det_score + 0.3 * w_ratio + 0.3 * h_ratio))
            except Exception:
                quality = float(face.det_score)

            # Thumbnail crop of just the face region
            face_crop = _crop_face_thumbnail(image, bbox_tuple)
            face_thumb = _make_jpeg_b64(face_crop, max_width=140) if face_crop is not None else _make_jpeg_b64(image, max_width=140)

            face_out_list.append({
                "face_id": face_id,
                "bbox": list(bbox_tuple),
                "det_score": float(face.det_score),
                "pose_bucket": pose_value,
                "quality_score": quality,
                "thumbnail_b64": face_thumb or "",
                "is_recommended": rank == 0,  # largest face flagged as suggested
            })

        results.append({
            "photo_index": idx, "filename": filename, "image_hash": image_hash,
            "status": "ok", "reason": None,
            "photo_thumbnail_b64": photo_thumb, "faces": face_out_list,
        })

    return results


async def confirm_face_selections(
    pipeline: "RecognitionPipeline",
    files: list[bytes],
    filenames: list[str],
    selections: list[dict],
    outlier_threshold: float = 0.30,
) -> dict:
    """Re-detect faces in each photo, match operator selection by face_id, extract embeddings.

    Then run cross-selection outlier detection — if any selected face is dissimilar to the
    others (mean cosine sim < threshold), flag it for operator review.

    Returns a dict matching BatchConfirmOut. Embeddings are returned in `selections[*]["_embedding"]`
    so the caller can persist them (the response model strips that field).
    """
    import cv2
    import numpy as np

    sel_by_photo: dict[int, dict] = {s["photo_index"]: s for s in selections}
    enriched: list[dict] = []

    for idx, (file_bytes, filename) in enumerate(zip(files, filenames)):
        if idx not in sel_by_photo:
            continue
        sel = sel_by_photo[idx]
        image_hash = _hash_bytes(file_bytes)
        if image_hash != sel.get("image_hash"):
            enriched.append({
                "photo_index": idx, "face_id": sel["face_id"],
                "pose_bucket": "unknown", "quality_score": None,
                "is_outlier": False, "mean_similarity": None,
                "_error": "image_hash_mismatch — re-upload this photo",
                "_embedding": None,
            })
            continue

        buf = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            enriched.append({
                "photo_index": idx, "face_id": sel["face_id"],
                "pose_bucket": "unknown", "quality_score": None,
                "is_outlier": False, "mean_similarity": None,
                "_error": "invalid image bytes", "_embedding": None,
            })
            continue

        # Re-detect (stateless — we don't trust client-side bbox)
        detected = pipeline.detect_enrollment_faces(image, min_det_score=0.50)
        target_face = None
        for face in detected:
            bb = face.bbox
            fid = _compute_face_id(image_hash, (float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)))
            if fid == sel["face_id"]:
                target_face = face
                break

        if target_face is None:
            enriched.append({
                "photo_index": idx, "face_id": sel["face_id"],
                "pose_bucket": "unknown", "quality_score": None,
                "is_outlier": False, "mean_similarity": None,
                "_error": "selected face no longer detectable — re-pick", "_embedding": None,
            })
            continue

        try:
            embedding = pipeline.embed_enrollment_face(image, target_face)
            embedding = embedding.astype(np.float32)
            norm = float(np.linalg.norm(embedding))
            if norm > 1e-6:
                embedding = embedding / norm
        except Exception as exc:
            enriched.append({
                "photo_index": idx, "face_id": sel["face_id"],
                "pose_bucket": "unknown", "quality_score": None,
                "is_outlier": False, "mean_similarity": None,
                "_error": f"embedding extraction failed: {exc}", "_embedding": None,
            })
            continue

        # Pose + quality
        try:
            from ecoface_lite.ai_engine.geometry import compute_face_geometry
            from ecoface_lite.ai_engine.pose_estimator import classify_pose_bucket
            pose = classify_pose_bucket(target_face.landmarks, target_face.bbox) if target_face.landmarks is not None else None
            pose_value = pose.value if pose is not None else "unknown"
            geom = compute_face_geometry(target_face, image.shape)
            w_ratio = min(1.0, geom.width / 200.0)
            h_ratio = min(1.0, geom.height / 200.0)
            quality = float(min(1.0, 0.4 * target_face.det_score + 0.3 * w_ratio + 0.3 * h_ratio))
        except Exception:
            pose_value = "unknown"
            quality = float(target_face.det_score)

        enriched.append({
            "photo_index": idx, "face_id": sel["face_id"],
            "pose_bucket": pose_value, "quality_score": quality,
            "is_outlier": False, "mean_similarity": None,
            "_embedding": embedding, "_error": None,
        })

    # Outlier check — only when >=3 successful selections
    successful = [e for e in enriched if e["_embedding"] is not None]
    outlier_indices: list[int] = []
    if len(successful) >= 3:
        embs = np.stack([e["_embedding"] for e in successful])
        sim_matrix = embs @ embs.T
        n = len(successful)
        for i, entry in enumerate(successful):
            others = [sim_matrix[i, j] for j in range(n) if j != i]
            mean_sim = float(np.mean(others)) if others else 1.0
            entry["mean_similarity"] = mean_sim
            if mean_sim < outlier_threshold:
                entry["is_outlier"] = True
                outlier_indices.append(entry["photo_index"])

    return {
        "selections": enriched,
        "outlier_indices": outlier_indices,
        "all_similar": len(outlier_indices) == 0,
    }


async def create_person_from_selections(
    session: AsyncSession,
    settings: "Settings",
    *,
    display_name: str,
    notes: str | None,
    files: list[bytes],
    filenames: list[str],
    confirmed: dict,
    skip_conflict_check: bool = False,
) -> tuple["Person", int]:
    """Create a Person and persist N FaceEmbedding rows (one per confirmed selection).

    Returns (person, embeddings_written).
    Stores the FIRST file as `source_image_path` for the gallery display.
    """
    import json
    import numpy as np
    import cv2  # noqa: F401  — needed by called helpers via Settings

    # Filter successful selections
    successful = [s for s in confirmed["selections"] if s.get("_embedding") is not None]
    if not successful:
        raise ValueError("no valid face selections to enroll")

    # Conflict check: use the highest-quality embedding as the conflict probe
    successful.sort(key=lambda s: s.get("quality_score") or 0.0, reverse=True)
    primary = successful[0]
    primary_embedding = primary["_embedding"]

    if not skip_conflict_check:
        await _check_identity_conflict(session, primary_embedding, settings.enrollment_conflict_threshold)

    # Save source image (first file) to uploads
    uploads = settings.resolved_uploads_dir()
    uploads.mkdir(parents=True, exist_ok=True)
    first_filename = filenames[0]
    first_bytes = files[0]
    digest = sha256_hex(first_bytes)
    ext = Path(first_filename).suffix.lower() or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = uploads / stored_name
    stored_path.write_bytes(first_bytes)
    rel_upload = str(Path("data/uploads") / stored_name)

    person = Person(
        display_name=display_name,
        notes=notes,
        source_image_path=rel_upload,
        source_image_hash=digest,
    )
    session.add(person)
    await session.flush()

    # Persist all extra photos too so the gallery can show them
    extra_paths: list[str] = []
    for i, (b, fn) in enumerate(zip(files, filenames)):
        if i == 0:
            continue
        ext_i = Path(fn).suffix.lower() or ".jpg"
        name_i = f"{uuid.uuid4().hex}{ext_i}"
        (uploads / name_i).write_bytes(b)
        extra_paths.append(str(Path("data/uploads") / name_i))
    if extra_paths:
        person.extra_photo_paths = json.dumps(extra_paths)

    # Write embeddings
    written = 0
    for sel in successful:
        emb_bytes = sel["_embedding"].astype(np.float32).tobytes()
        face_row = FaceEmbedding(
            person_id=person.id,
            ingest_sha256=digest if sel["photo_index"] == 0 else None,
            embedding=emb_bytes,
            embedding_dim=int(sel["_embedding"].shape[0]),
            model_name=settings.insightface_model_name,
            pose_bucket=sel.get("pose_bucket") or "unknown",
            quality_score=sel.get("quality_score"),
        )
        session.add(face_row)
        written += 1

    await session.flush()
    await session.refresh(person)
    logger.info("Multi-face enrolled person id=%s name=%s embeddings=%d", person.id, display_name, written)
    return person, written
