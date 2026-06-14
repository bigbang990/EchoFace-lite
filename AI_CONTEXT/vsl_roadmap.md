# EchoFace — Video Source Layer (VSL) Roadmap

## Architectural contract (never changes)
The AI Engine, Tracking Engine, Alert Engine, and Incident Platform call
`get_frame()` and `get_historical_stream()` on a `BaseVideoSource`.
They never know if it's a file, a phone, an RTSP camera, or an NVR.
Established in VSL Phase 1, never violated.

## Interface every source must implement

```python
connect() -> bool
disconnect() -> None
get_frame() -> Frame | None
get_metadata() -> CameraMetadata
health_check() -> HealthStatus
get_historical_stream(start_time, end_time)  # stubbed NotImplemented until Phase 4
```

---

## VSL Phase 1 — Source Abstraction Foundation
*The system stops knowing where video comes from.*

**Goals**
- `BaseVideoSource` abstract class with standard interface
- `VideoFileSource` — wraps existing OpenCV video file reading
- `RTSPSource` — connects to RTSP streams (Hikvision, Dahua, Android IP Webcam)
- `SourceRegistry` — SQLite-backed registry of registered sources
- Camera object fields: `camera_id`, `name`, `source_type`, `stream_url`,
  `zone`, `location`, `status`, `last_seen`
- Manual source registration — add by name, URL, location

**Possible issues**
| Issue | Handling |
|---|---|
| RTSP stream drops mid-session | `health_check()` in background thread; auto-reconnect with backoff |
| OpenCV RTSP on Colab latency | Frame buffer with drop policy — always serve latest frame, never queue stale |
| Source registry lost on restart | SQLite-backed from day one — survives restarts |
| Android IP Webcam URL format varies | Document tested URL patterns in CLAUDE.md; treat as RTSP source |

**What doesn't change**
Existing video file path handling in the pipeline continues working —
`VideoFileSource` wraps it transparently.

**Regression gate**
- All Phase 7 tests pass with video routed through `VideoFileSource`
- RTSP connect/disconnect tested against real stream or mock

---

## VSL Phase 2 — Location Intelligence + Health Monitoring
*Cameras become part of a place, not just a URL.*

**Goals**
- Location hierarchy in SQLite: `Country → State → District → Site → Zone → Camera`
- Every camera assigned to a zone at registration
- Health monitor background task — polls each source every N seconds,
  updates `status` and `last_seen`
- Dashboard panel: Connected / Online / Offline / Warning counts
- Zone-aware alert routing — alerts carry zone context, not just camera ID

**Location schema**
```
site_id, site_name
zone_id, zone_name, site_id
camera_id, zone_id, ...
```

**Possible issues**
| Issue | Handling |
|---|---|
| Health monitor thread blocks main pipeline | Runs as async background task completely separate from frame pipeline |
| Zone hierarchy is overkill for single-site dissertation demo | Schema supports full hierarchy; UI only requires Zone + Camera for demo |
| Camera goes offline mid-incident | Alert stays open, `last_seen_at` freezes, status shows `OFFLINE` |
| Multiple cameras in same zone | All trigger sightings; Intelligence Layer (Phase 10) handles deduplication |

**Regression gate**
- Health status updates without blocking frame processing
- Zone field populated on all existing test cameras

---

## VSL Phase 3 — Android Camera Source + Multi-Source Pipeline
*Real deployment becomes possible without professional hardware.*

**Goals**
- `AndroidCameraSource` — connects via IP Webcam or RTSP app
- Multi-source frame scheduler — round-robin or priority-based frame pull
- Per-source frame rate tracking — each source reports actual FPS independently
- Source isolation — one source failing doesn't crash others
- Colab demo: video file + Android camera running simultaneously

**Possible issues**
| Issue | Handling |
|---|---|
| Multi-source threading on Colab causes instability | Each source in its own daemon thread with exception isolation |
| Android FPS much lower than video file | Frame scheduler normalizes pull rate per source |
| Network drop on Android camera during demo | Auto-reconnect with 5s backoff; shows `RECONNECTING` in dashboard |
| CPU can't process two sources at real-time | Detection interval scales per source — lower priority sources get larger intervals |

**Regression gate**
- Single-source behavior identical to Phase 1
- One source failure doesn't affect other active sources

---

## VSL Phase 4 — Historical Footage Access
*The system can investigate the past, not just monitor the present.*

**Goals**
- `get_historical_stream(start_time, end_time)` implemented on `VideoFileSource`
- Historical Search Job — triggered from incident, pulls footage, runs recognition,
  creates sightings
- Sightings from historical search tagged as `source: historical` vs `source: live`
- Results surface in Case Management UI: "Found in footage from 10:00–10:15, Gate Cam A"

**Historical Search Pipeline**
```
Incident Created
      ↓
Historical Search Job
      ↓
VSL: get_historical_stream()
      ↓
Recognition Engine
      ↓
Sighting Creation (tagged historical)
      ↓
Alert Correlation Engine
```

**Possible issues**
| Issue | Handling |
|---|---|
| Historical search on large video file is slow | Job runs async; operator sees progress indicator |
| Video file has no timestamp metadata | Frame index as proxy; creation time offset if available |
| Historical sightings trigger live alert rules | `source` tag separates pipelines — historical goes to case history |
| NVR/DVR historical access is a different API | Stubbed `NotImplemented` on RTSP/NVR sources — Phase 5 VSL concern |

**Regression gate**
- Live pipeline unaffected when historical job is running
- Historical sightings don't appear in live alert feed

---

## VSL Phase 5 — NVR / DVR Integration (Post-Dissertation)
*Enterprise deployment becomes real.*

**Goals**
- `NVRSource` — ONVIF protocol for live stream and historical playback
- `DVRSource` — legacy support via RTSP live + manual footage export for historical
- `get_historical_stream()` fully implemented for both
- Camera auto-discovery via ONVIF device search on local network

**Possible issues**
| Issue | Handling |
|---|---|
| ONVIF implementation is complex and vendor-specific | Use `onvif-zeep` Python library; test against Hikvision and Dahua first |
| DVR historical API is proprietary per vendor | DVR historical treated as video file export — operator pulls clip, VSL processes as `VideoFileSource` |
| Auto-discovery exposes cameras on network scan | Discovery is opt-in, not automatic |
| Historical stream from NVR needs authentication | Credentials stored encrypted in SQLite, never in config files or env |

**Regression gate**
- All existing source types unaffected
- NVR live stream passes through existing pipeline without modification

---

## Summary

```
VSL Phase 1  →  abstraction foundation, file + RTSP sources
VSL Phase 2  →  location hierarchy, health monitoring
VSL Phase 3  →  android camera, multi-source pipeline
VSL Phase 4  →  historical footage, search pipeline
VSL Phase 5  →  NVR/DVR, enterprise integration (post-dissertation)
```
