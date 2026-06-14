"""NVR/DVR management endpoints (VSL Phase 5).

POST /cameras/{id}/nvr/test-onvif
  → connects ONVIF, returns device info (model, firmware, serial)
  → requires onvif-zeep; returns 501 if not installed

PATCH /cameras/{id}/nvr/credentials
  → update ONVIF host/port/username/password without full camera edit
  → password stored as base64 in onvif_password_enc column

GET /cameras/discover-onvif
  → WS-Discovery probe, returns candidate list (opt-in, never auto-registers)
  → requires onvif-zeep; returns 501 if not installed
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ecoface_lite.api.deps import DbSession
from ecoface_lite.api.schemas import NVRCredentialsUpdate, ONVIFDeviceInfo, ONVIFDiscoveryResult
from ecoface_lite.db.models import Camera
from ecoface_lite.input_sources.nvr_source import NVRSource, _ONVIF_MISSING_MSG

router = APIRouter(tags=["nvr"])


async def _get_camera_or_404(db: DbSession, camera_id: int) -> Camera:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


def _build_nvr_source(camera: Camera) -> NVRSource:
    if not camera.onvif_host:
        raise HTTPException(
            status_code=422,
            detail="Camera has no onvif_host — set it via PATCH /cameras/{id}/nvr/credentials first",
        )
    if not camera.stream_url:
        raise HTTPException(
            status_code=422,
            detail="Camera has no stream_url (live RTSP URL required for NVR source)",
        )
    return NVRSource(
        source_id=str(camera.id),
        name=camera.label,
        stream_url=camera.stream_url,
        onvif_host=camera.onvif_host,
        onvif_port=camera.onvif_port or 80,
        onvif_username=camera.onvif_username or "admin",
        onvif_password_enc=camera.onvif_password_enc,
    )


@router.post("/cameras/{camera_id}/nvr/test-onvif", response_model=ONVIFDeviceInfo)
async def test_onvif_connection(camera_id: int, db: DbSession) -> ONVIFDeviceInfo:
    """Connect to the camera's ONVIF service and return device information.

    Returns 501 if onvif-zeep is not installed.
    Returns 502 if the ONVIF connection fails (wrong host/port/credentials).
    """
    camera = await _get_camera_or_404(db, camera_id)
    source = _build_nvr_source(camera)

    try:
        info = source.get_device_info()
    except ImportError:
        raise HTTPException(status_code=501, detail=_ONVIF_MISSING_MSG)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"ONVIF connection failed: {exc}",
        )

    return ONVIFDeviceInfo(**info)


@router.patch("/cameras/{camera_id}/nvr/credentials", response_model=dict)
async def update_nvr_credentials(
    camera_id: int,
    body: NVRCredentialsUpdate,
    db: DbSession,
) -> dict:
    """Update ONVIF credentials for an NVR/DVR camera.

    Password is base64-encoded before storage — never stored in plaintext.
    Only fields present in the request body are updated (PATCH semantics).
    """
    camera = await _get_camera_or_404(db, camera_id)

    if body.onvif_host is not None:
        camera.onvif_host = body.onvif_host
    if body.onvif_port is not None:
        camera.onvif_port = body.onvif_port
    if body.onvif_username is not None:
        camera.onvif_username = body.onvif_username
    if body.onvif_password is not None:
        camera.onvif_password_enc = base64.b64encode(body.onvif_password.encode()).decode()

    db.add(camera)
    await db.commit()
    return {"updated": True, "camera_id": camera_id}


@router.get("/cameras/discover-onvif", response_model=list[ONVIFDiscoveryResult])
async def discover_onvif_devices(timeout: float = 5.0) -> list[ONVIFDiscoveryResult]:
    """WS-Discovery probe for ONVIF devices on the local network.

    This is opt-in and operator-triggered — never called automatically.
    Returns candidates; does NOT register them as cameras.
    Returns 501 if onvif-zeep is not installed.

    Query param:
      timeout  float  seconds to wait for responses (default 5.0, max 30.0)
    """
    timeout = min(max(timeout, 1.0), 30.0)
    try:
        results = NVRSource.discover(timeout_seconds=timeout)
    except ImportError:
        raise HTTPException(status_code=501, detail=_ONVIF_MISSING_MSG)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Discovery failed: {exc}")

    return [ONVIFDiscoveryResult(**r) for r in results]
