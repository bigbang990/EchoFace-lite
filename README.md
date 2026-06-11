# EchoFace Lite

Real-time missing person re-identification from surveillance footage.
Multi-face tracking · ArcFace embeddings · YOLOv8 detection · FastAPI backend

**Version:** v0.7.0 | **Runtime:** Google Colab T4 | **Status:** Dissertation complete

---

## Quick Start (Colab)

1. Open `notebooks/colab_setup.ipynb` or paste the setup cell
2. Set branch: `phase6-detector-abstraction` (or `main` for v0.7.0)
3. Run setup cell — installs deps, downloads YOLOv8 weights, starts server
4. Copy ngrok URL from output

### Environment (SERVER_ENV)

```python
SERVER_ENV = {
    "DETECTOR_PROVIDER":               "yolo",
    "DETECTOR_INPUT_WIDTH":            "640",
    "DETECTOR_INPUT_HEIGHT":           "640",
    "DETECTOR_MAX_INPUT_PIXELS":       "409600",
    "DETECTOR_RESOLUTION_CAP_ENABLED": "0",
    "GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE": "15",
    "INSIGHTFACE_CTX_ID":              "0",
}
```

---

## API Endpoints

### Persons
| Method | Path | Description |
|---|---|---|
| GET | /api/v1/persons | List all enrolled persons |
| POST | /api/v1/persons | Enroll person (single photo) |
| POST | /api/v1/persons/{id}/photos | Add reference photos (max 5) |

### Processing
| Method | Path | Description |
|---|---|---|
| POST | /api/v1/processing | Submit video job (async) |
| GET | /api/v1/processing/{job_id} | Poll job status + metrics |

### Detections
| Method | Path | Description |
|---|---|---|
| GET | /api/v1/detections | List all detection alerts |

### Observability
| Method | Path | Description |
|---|---|---|
| GET | /api/v1/observability/metrics | Live telemetry + session_id |
| GET | /api/v1/health | Health check |

---

## Architecture

```
Video frame → Detector (YOLOv8 GPU / SCRFD CPU) → Validator (landmarks · pose · quality) → Tracker (SORT) → Embedder (ArcFace ONNX) → Pipeline (governance · matching · alerts) → FastAPI → SQLite
```

Detector selected via `DETECTOR_PROVIDER` env var — no code change required.

---

## Regression Gate

Run before every merge to main:
```bash
python -m pytest tests/ -v --tb=short
```
Expected: 30 passed.

---

## Known Constraints
- onnxruntime-gpu: no stable CUDA 12.8 wheel — ONNX runs CPU only
- .env files unreliable on Colab — use SERVER_ENV dict injection
- Multi-CCTV: Phase 9 (not in current release)

---

## Roadmap
See [DISSERTATION_NOTES.md](DISSERTATION_NOTES.md) for full roadmap and benchmark results.
