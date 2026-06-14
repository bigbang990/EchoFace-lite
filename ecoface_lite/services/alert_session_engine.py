"""Alert Session Engine — Phase 8.

Converts per-frame recognition matches into operator-facing Alert sessions.
One continuous presence (incident × person × camera) = one open Alert.
Sightings accumulate silently beneath it.

Registry key: (incident_id, person_id, camera_id)
  — per-camera sessions so each feed has independent presence tracking.
  — Phase 10 will cluster across cameras using zone_id.
  — Phase 11 will promote level: sighting → candidate → verified → critical.
  — VSL Phase 4 sets source="historical" for footage-search paths.

Gap threshold: now - last_seen_at > gap_seconds closes the current session.
Zone boundary: zone_id change closes the session (Phase 10 gate — always None for now).

On restart: rebuild_from_db() restores open sessions from the last N minutes
so a server restart doesn't re-open duplicate alerts for ongoing presences.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from ecoface_lite.core.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ecoface_lite.db.models import Alert, Sighting

logger = get_logger(__name__)

_RegistryKey = tuple[int, int, int | None]  # (incident_id, person_id, camera_id)


@dataclass
class _ActiveSession:
    alert_id: int
    last_seen_at: datetime
    sighting_count: int
    zone_id: str | None = None


class AlertSessionEngine:
    """In-memory session registry backed by SQLite.

    One instance lives for the process lifetime (see get_alert_session_engine).
    All mutations go through _lock so async video jobs and live-stream paths
    can share the same engine safely.
    """

    def __init__(self, gap_seconds: int = 60, min_confidence_floor: float = 0.65) -> None:
        self._gap = timedelta(seconds=gap_seconds)
        self._min_confidence_floor = min_confidence_floor
        self._registry: dict[_RegistryKey, _ActiveSession] = {}
        self._lock = asyncio.Lock()

    async def rebuild_from_db(self, session: AsyncSession, lookback_minutes: int = 10) -> None:
        """Restore registry from open Alerts written in the last N minutes.

        Call once from the app lifespan after init_db() so the engine picks up
        any sessions that were live when the server was last stopped.
        """
        from sqlalchemy import select

        from ecoface_lite.db.models import Alert

        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=lookback_minutes)
        rows = (await session.execute(
            select(Alert)
            .where(Alert.status == "open")
            .where(Alert.last_seen_at >= cutoff)
        )).scalars().all()

        async with self._lock:
            for alert in rows:
                key: _RegistryKey = (alert.incident_id, alert.person_id, alert.camera_id)
                self._registry[key] = _ActiveSession(
                    alert_id=alert.id,
                    last_seen_at=alert.last_seen_at,
                    sighting_count=alert.sighting_count,
                    zone_id=alert.zone_id,
                )

        if rows:
            logger.info("AlertSessionEngine: rebuilt %d sessions from DB", len(rows))

    async def record_match(
        self,
        session: AsyncSession,
        *,
        incident_id: int,
        person_id: int,
        camera_id: int | None,
        confidence: float,
        detection_id: int | None = None,
        frame_index: int | None = None,
        snapshot_path: str | None = None,
        detected_at: datetime | None = None,
        source: str = "live",
        zone_id: str | None = None,
    ) -> tuple[Alert | None, Sighting]:
        """Core method: find or open an alert session, always persist a sighting.

        Returns (alert_or_None, sighting). Alert is None when confidence is below
        the configured floor — the sighting is still written as an audit record
        but no session is opened or extended.

        A new Alert is opened when:
          - No session exists for this (incident_id, person_id, camera_id) key
          - Gap since last_seen_at exceeds the configured threshold
          - zone_id differs from the current session (Phase 10 boundary crossing)
        """
        from ecoface_lite.db.models import Alert, Sighting

        now = detected_at or datetime.now(tz=timezone.utc)
        key: _RegistryKey = (incident_id, person_id, camera_id)

        async with self._lock:
            # Below-floor match: write audit sighting, no alert session
            if confidence < self._min_confidence_floor:
                sighting = Sighting(
                    incident_id=incident_id,
                    person_id=person_id,
                    camera_id=camera_id,
                    detection_id=detection_id,
                    confidence=confidence,
                    frame_index=frame_index,
                    snapshot_path=snapshot_path,
                    source=source,
                    status="pending",
                )
                session.add(sighting)
                return None, sighting
            existing = self._registry.get(key)
            gap_expired = existing is not None and (now - existing.last_seen_at) > self._gap
            zone_changed = existing is not None and existing.zone_id != zone_id

            new_session_needed = existing is None or gap_expired or zone_changed

            if new_session_needed:
                # Close the prior Alert row if there is one
                if existing is not None:
                    prior = await session.get(Alert, existing.alert_id)
                    if prior is not None and prior.status == "open":
                        prior.status = "closed"
                        prior.updated_at = now
                        logger.info(
                            "Alert closed alert_id=%d gap_expired=%s zone_changed=%s gap_s=%.1f",
                            prior.id, gap_expired, zone_changed,
                            (now - existing.last_seen_at).total_seconds() if existing else 0.0,
                        )

                alert = Alert(
                    incident_id=incident_id,
                    person_id=person_id,
                    camera_id=camera_id,
                    zone_id=zone_id,
                    status="open",
                    level="sighting",
                    source=source,
                    first_seen_at=now,
                    last_seen_at=now,
                    sighting_count=1,
                )
                session.add(alert)
                await session.flush()  # assigns alert.id

                self._registry[key] = _ActiveSession(
                    alert_id=alert.id,
                    last_seen_at=now,
                    sighting_count=1,
                    zone_id=zone_id,
                )
                logger.info(
                    "Alert opened alert_id=%d incident_id=%d person_id=%d camera_id=%s source=%s",
                    alert.id, incident_id, person_id, camera_id, source,
                )
            else:
                # Extend existing session
                existing.last_seen_at = now
                existing.sighting_count += 1

                alert = await session.get(Alert, existing.alert_id)
                alert.last_seen_at = now
                alert.sighting_count = existing.sighting_count
                alert.updated_at = now

            sighting = Sighting(
                alert_id=alert.id,
                incident_id=incident_id,
                person_id=person_id,
                camera_id=camera_id,
                detection_id=detection_id,
                confidence=confidence,
                frame_index=frame_index,
                snapshot_path=snapshot_path,
                source=source,
                status="pending",
            )
            session.add(sighting)

        return alert, sighting

    async def evict_incident(self, incident_id: int) -> int:
        """Remove all in-memory sessions for a closing incident.

        The caller is responsible for updating Alert rows in the DB before calling this.
        Returns the number of sessions evicted.
        """
        async with self._lock:
            keys = [k for k in self._registry if k[0] == incident_id]
            for k in keys:
                del self._registry[k]
        if keys:
            logger.info("AlertSessionEngine: evicted %d sessions for incident_id=%d", len(keys), incident_id)
        return len(keys)

    async def close_all(self, session: AsyncSession) -> None:
        """Close every open session in the registry (call on graceful shutdown)."""
        from ecoface_lite.db.models import Alert

        now = datetime.now(tz=timezone.utc)
        async with self._lock:
            for active in self._registry.values():
                alert = await session.get(Alert, active.alert_id)
                if alert is not None and alert.status == "open":
                    alert.status = "closed"
                    alert.updated_at = now
            self._registry.clear()


# Process-lifetime singleton — survives across request sessions and background jobs.
_engine_instance: AlertSessionEngine | None = None


def get_alert_session_engine() -> AlertSessionEngine:
    global _engine_instance
    if _engine_instance is None:
        from ecoface_lite.core.config import get_settings
        settings = get_settings()
        _engine_instance = AlertSessionEngine(
            gap_seconds=settings.alert_session_gap_seconds,
            min_confidence_floor=settings.alert_min_confidence_floor,
        )
    return _engine_instance
