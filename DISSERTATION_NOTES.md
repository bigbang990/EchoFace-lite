# EchoFace Lite — Dissertation Notes

## System Architecture

EchoFace Lite is an occlusion-resilient missing person re-identification
system built for surveillance footage analysis. It implements a multi-stage
AI pipeline on Google Colab T4 GPU.

### Stack
| Component | Technology |
|---|---|
| Detector (GPU) | YOLOv8-face (derronqi), PyTorch, 5-point landmarks |
| Detector (CPU) | SCRFD via InsightFace buffalo_l |
| Embedder | ArcFace (InsightFace buffalo_l, ONNX CPU) |
| Tracker | SORT (custom track_manager.py) |
| Backend | FastAPI + uvicorn + ngrok |
| Frontend | Streamlit dashboard |
| Database | SQLite via aiosqlite + SQLAlchemy async |
| Runtime | Google Colab T4 / CUDA 12.8 / PyTorch 2.11.0 / Python 3.12.13 |

### Detector Abstraction
Two detectors behind a common BaseDetector interface.
DETECTOR_PROVIDER env var selects at runtime — no code change required.
SCRFD for CPU dev, YOLOv8 for GPU production.

### Key Design Decisions
- No onnxruntime-gpu (no stable CUDA 12.8 wheel — confirmed across 5 attempts)
- Lazy face_app load — only initialises on SCRFD path
- All settings injected via SERVER_ENV dict to subprocess — .env unreliable on Colab
- Landmarks required: validator hard-rejects landmarks=None
- No affine alignment — landmark source does not affect embedding quality

## Benchmark Results (v0.6.9 Pre-Release Stress Evaluation)

### System Score: 8.4 / 10 — Grade: A-

| Scenario | Frames | Faces Detected | Avg FPS | Key Behaviour |
|---|---|---|---|---|
| A — High-density crowd (~50p) | 358 | 5,426 | 15.3 | Load shed, confidence → 0.40 |
| B — Extreme saturation (100p+) | 508 | 9,707 | 13.2 | Queue pressure 11.15, 150 rescales |
| C — Degraded low-light | 858 | 10,387 | 18.8 | 167 emergency recalls, 1,634 decays |
| D — 60 FPS 1080p marathon | 1,633 | 10,565 | 19.7 | 136 tracking-only cycles |

### Core Strengths
- Identity Switch Rate: 0.0000 across all scenarios (40,000+ raw detections)
- Detector latency: 15.8ms – 19.4ms (deterministic across all load profiles)
- Temporal safety nets: emergency_recall_recoveries fired 757 times in marathon
- Adaptive resolution: clamped down 150 times under saturation — prevented VRAM crash

### Regression Gate Thresholds (enforced per phase)
| Metric | Threshold |
|---|---|
| identity_switch_rate | = 0 |
| stable_matches | ≥ 35 per video |
| detector_runtime_ms | ≤ 100ms GPU |
| average_processing_fps | ≥ 10 GPU |
| alerts_per_video | ≥ 1 |

### Known Limitations
1. State-machine throughput degrades past 15 concurrent tracks — 13–23 FPS under crowd load
2. Confidence floor drops to 0.40 on overload — counterproductive in dense scenes (fix: invert to 0.55)
3. Tracking queue was uncapped — spiked to 113 concurrent unconfirmed tracks (fix: hard ceil at 32)
4. Metrics accumulated across runs — fixed in v0.7.0 via per-session UUID and metrics.reset()
5. Single-camera only — multi-CCTV is Phase 9

## Demo Script (enroll → detect → alert)

### Setup
1. Launch Colab cell — ngrok tunnel starts, copy public URL
2. Open Streamlit dashboard at tunnel URL

### Step 1 — Enroll a person
POST /api/v1/persons
- display_name: "Test Subject"
- image: clear frontal photo (JPEG/PNG, ≤ max_image_mb)
Expected: 200, person_id returned, embedding stored

### Step 2 — Add reference photos (multi-angle)
POST /api/v1/persons/{person_id}/photos
- images: 2–3 additional photos (profile, partial occlusion)
Expected: PersonEnrollMultiOut with photos_accepted count

### Step 3 — Process surveillance video
POST /api/v1/processing
- video_relative_path: "test_clip.mp4"
Expected: job_id returned immediately (async)

### Step 4 — Poll job status
GET /api/v1/processing/{job_id}
Watch: processed_frames climbing, avg_fps, alerts_created

### Step 5 — View alerts
GET /api/v1/detections
Expected: detection records with confidence scores, snapshot paths, frame indices

### Step 6 — Confirm in dashboard
Streamlit shows bounding boxes, track IDs, match confidence, alert feed

## Future Roadmap
| Phase | What | Timeline |
|---|---|---|
| 8 | Full DB schema impl, async detector, CPU quantisation | 1–2 months post-submission |
| 9 | Forensic dashboard, multi-CCTV, case management | 3–4 months |
| 10 | Age/gender, cross-camera re-ID, liveness detection | 6 months |
| 11 | Commercial API (Face++ model) — Surveillance SDK + Incident Case API | 12 months |
