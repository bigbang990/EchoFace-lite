# Checkpoint — 2026-06-14 — VSL Phase 2 complete: Location Intelligence + Health Monitor

## Phase
VSL Phase 2 — Location Intelligence + Health Monitoring (complete)
Branch: `vsl-phase2-location-health`

## Regression baseline metrics (14/14 pass)
identity_switch_rate: 0.000
stable_matches: green
confirmation_rate: green
validator_rejection_rate: green
bbox_jitter: green

## Changes this session (VSL Phase 2)

### DB — `ecoface_lite/db/models.py`
- `Site` model: id, name (unique), description, created_at
  → has `zones` relationship (cascade delete-orphan)
- `Zone` model: id, site_id (FK → sites.id ON DELETE CASCADE), name, description, created_at
  → has `site` and `cameras` relationships
- `Camera`: added `zone_id` FK (→ zones.id ON DELETE SET NULL, nullable)
  → `zone` (free-text, Phase 1 fallback) kept for backward compat
  → `zone_obj` relationship to Zone

### DB migrations — `ecoface_lite/db/session.py`
- `CREATE TABLE IF NOT EXISTS sites`
- `CREATE TABLE IF NOT EXISTS zones` (with site_id FK)
- `CREATE INDEX IF NOT EXISTS ix_zones_site_id ON zones(site_id)`
- `ALTER TABLE cameras ADD COLUMN zone_id INTEGER REFERENCES zones(id) ON DELETE SET NULL`
- `CREATE INDEX IF NOT EXISTS ix_cameras_zone_id ON cameras(zone_id)`

### Config — `ecoface_lite/core/config.py`
- `health_monitor_enabled: bool = True` (HEALTH_MONITOR_ENABLED)
- `health_monitor_interval_seconds: int = 60` (HEALTH_MONITOR_INTERVAL_SECONDS)

### Health monitor — `ecoface_lite/services/health_monitor.py` (NEW)
- `_poll_all_cameras(session_factory, settings)`: queries active cameras,
  `build_source()` → `connect()` → `health_check()` → `disconnect()` per camera,
  writes `status` + `last_seen` directly to DB (no HTTP overhead)
- `_health_monitor_loop(session_factory, settings)`: runs forever,
  `asyncio.sleep(interval)` between passes; exceptions logged, never crash
- `start_health_monitor(session_factory, settings) → asyncio.Task | None`:
  creates named task `"health_monitor"` if enabled; returns None if disabled
- **Isolation verified**: runs as standalone asyncio.Task, NOT in frame loop;
  camera offline cannot block frame acquisition

### API — `ecoface_lite/api/routers/sites.py` (NEW)
- `POST /sites` (201)
- `GET /sites`
- `GET /sites/{id}`
- `GET /sites/{id}/zones` — convenience sub-resource
- `DELETE /sites/{id}` (cascade deletes zones)

### API — `ecoface_lite/api/routers/zones.py` (NEW)
- `POST /zones` (validates site_id, 404 if site missing)
- `GET /zones`
- `GET /zones/{id}`
- `DELETE /zones/{id}`

### API — `ecoface_lite/api/routers/cameras.py` (MODIFIED)
- `POST /cameras`: validates `zone_id` FK (404 if zone missing), stores it
- `GET /cameras/health-summary` (NEW): `{total, online, offline, reconnecting, unknown}`

### Schemas — `ecoface_lite/api/schemas.py`
- `SiteCreate`: name, description
- `SiteOut`: id, name, description, created_at
- `ZoneCreate`: site_id, name, description
- `ZoneOut`: id, site_id, name, description, created_at
- `CameraOut`: added `zone_id: int | None`
- `CameraCreate`: added `zone_id: int | None`

### App lifespan — `ecoface_lite/api/main.py`
- Imports `sites`, `zones` routers; both registered at `/api/v1`
- `start_health_monitor()` called after `init_db()` + alert engine rebuild
- Task cancelled with `task.cancel() + await task` on shutdown (CancelledError swallowed)

## Architecture state

```
Location hierarchy:
  sites    → id, name
  zones    → id, site_id FK, name
  cameras  → id, zone_id FK (nullable), source_type, status, last_seen

Health monitor:
  services/health_monitor.py → standalone asyncio.Task "health_monitor"
  polls every HEALTH_MONITOR_INTERVAL_SECONDS (default 60s)
  writes directly to cameras.status / cameras.last_seen
  NOT in frame loop (hard boundary per CLAUDE.md VSL hard stops)

API surface added:
  GET  /api/v1/sites
  POST /api/v1/sites
  GET  /api/v1/sites/{id}
  GET  /api/v1/sites/{id}/zones
  DELETE /api/v1/sites/{id}
  GET  /api/v1/zones
  POST /api/v1/zones
  GET  /api/v1/zones/{id}
  DELETE /api/v1/zones/{id}
  GET  /api/v1/cameras/health-summary
```

## VSL Phase 3 prerequisites (next)
- `AndroidCameraSource` — connects via IP Webcam or RTSP app (same RTSPSource interface)
- Multi-source frame scheduler — round-robin or priority-based frame pull
- Per-source frame rate tracking
- Source isolation — one source failing doesn't crash others
- When multi-source scheduler lands: wire `get_frame()` into the pipeline
  (replacing `frames()` iterator — the planned Phase 1 production path switch)

## Prior phases preserved
VSL Phase 1 changes intact on `vsl-phase1-source-abstraction`.
Phase 8 + 8.5–8.7 intact on `phase8-lifecycle-enrollment`.
