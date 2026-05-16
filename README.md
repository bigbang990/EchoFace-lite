# EcoFace Lite (Engineering scaffold)

Real-time–style **missing-person alert** platform: prerecorded video first, same code paths extend later to webcam/RTSP. AI is **integrated** (InsightFace + ONNXRuntime), not custom-trained.

This repository is structured for a **solo, weekend cadence**: clear layers, small files, configuration over code, and an API-first core with Streamlit as an optional UI shell.

---

## 1. Recommended folder structure (what is in this repo)

```text
ecoface_lite/                 # Installable Python package (importable from anywhere)
  core/                       # Cross-cutting: settings + logging
  ai_engine/                  # Detection, embeddings, matching, pipeline orchestration
  api/                        # FastAPI routers + app factory
  db/                         # SQLAlchemy models + async engine/session
  services/                   # Use-cases orchestrating DB + AI + IO (keep routers thin)
  input_sources/              # Video file today; webcam/RTSP implement same contracts later
dashboard/                    # Streamlit UI (HTTP client to API — not a second business core)
data/                         # Local data dirs (uploads, snapshots, videos, sqlite file)
logs/                         # Rotatable file logs + stdout
tests/                        # Pytest smoke tests (expand per phase)
Dockerfile
docker-compose.yml
requirements.txt
.env.example
```

**Why package layout (`ecoface_lite/` as a package, not a monolithic `main.py`):** imports stay explicit, testability improves, and Docker/CI can install the same tree you use locally.

---

## 2. Modular architecture blueprint

| Layer | Responsibility | Depends on |
|------|----------------|------------|
| **AI Engine** | Detect faces, produce embeddings, score similarity | NumPy, OpenCV, InsightFace/ONNX only inside this layer |
| **Services** | Enrollment, gallery loading, video job orchestration, persistence rules | AI engine interfaces, DB session, filesystem paths from settings |
| **API** | HTTP, validation (Pydantic), auth hooks later, status codes | Services |
| **DB** | Schema, migrations later (Alembic), async session | SQLAlchemy |
| **Input sources** | Iterate frames from file/webcam/RTSP | OpenCV capture (isolated) |
| **Dashboard** | UX only; calls API over HTTP | `requests` |

**Separation principles used here**

- **FastAPI routers stay thin** so you can add a CLI worker later without duplicating rules.
- **Heavy imports are lazy** (`deps.get_recognition_pipeline`, CV2 inside service methods) so health checks and tests do not require InsightFace on disk.
- **Embeddings are bytes in SQLite** today; swap to `pgvector` later without changing service method signatures if you keep “embedding = `np.ndarray` at the boundary” inside the AI layer.

---

## 3. Configuration system

- `ecoface_lite/core/config.py` — `pydantic-settings` loads `.env` + environment variables.
- `.env.example` lists keys; **copy to `.env`** for local overrides.

**Why:** twelve-factor style configuration is what makes EC2/ECS migration mostly “set env vars + mount volumes”, not rewrites.

---

## 4. Logging system

- `ecoface_lite/core/logging.py` — `setup_logging()` configures stdout + `logs/ecoface_lite.log`.

**Why:** centralized setup means you can later add JSON logs, rotation, or shipping to CloudWatch without touching business modules.

---

## 5. Database schema (SQLite now, PostgreSQL-ready)

Tables (`ecoface_lite/db/models.py`):

- **`persons`** — who is missing (display metadata + stored upload path).
- **`face_embeddings`** — float32 embedding bytes + `embedding_dim` + `model_name` (supports multiple embeddings per person later).
- **`detection_events`** — alert rows with confidence, threshold snapshot, source metadata, optional `snapshot_path`.

**Why SQLAlchemy async:** identical service code can target `postgresql+asyncpg://...` by changing `DATABASE_URL`.

---

## 6. Suggested API endpoints (implemented baseline)

Base prefix: `/api/v1`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness for Docker/ALB |
| GET | `/persons` | List enrolled persons |
| POST | `/persons` | Multipart enroll (`display_name`, `notes`, `image`) |
| GET | `/detections` | Recent detection events (`limit` query) |
| POST | `/videos/process` | Legacy synchronous JSON processing for short clips only |
| POST | `/videos/process/async` | Create a background job for a server-side video path |
| POST | `/videos/upload-and-process` | Upload a video, persist it locally, create a background job, return immediately |
| GET | `/videos/processing-status/{job_id}` | Poll job status, progress, errors, and alert count |

**Video processing workflow:** Dashboard → FastAPI upload/path endpoint → queued `processing_status` row → single local background worker → `VideoFileSource` frame iterator → frame skipping + resize → InsightFace pipeline → event dedupe + snapshot persistence → dashboard polling.

**Why this fixes request timeouts:** the HTTP request only saves the file and creates a job row, then returns `{ job_id, status_url }`. CPU-heavy OpenCV/InsightFace work happens outside the API request lifecycle.

---

## 7. Environment setup (Python 3.10.11)

```powershell
cd f:\Joydeb-Data\EchoFace_Eng1.0.1
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

**InsightFace models** download on first use (ensure disk + network available once).

---

## 8. Local development workflow

**API**

```powershell
uvicorn ecoface_lite.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` for Swagger.

**Dashboard (separate terminal)**

```powershell
$env:ECOFACE_API_BASE = "http://127.0.0.1:8000/api/v1"
streamlit run dashboard/app.py
```

**Typical demo flow**

1. Upload a clip from Streamlit or put a prerecorded clip under `data/videos/`.
2. Enroll a missing person via Swagger `POST /persons` or Streamlit tab.
3. Call `POST /videos/upload-and-process` or `POST /videos/process/async`.
4. Poll `GET /videos/processing-status/{job_id}` until `completed` or `failed`.
5. Inspect `GET /detections` and snapshot files under `data/snapshots/`.

**CPU-friendly video settings**

- `VIDEO_FRAME_SKIP` — process every Nth frame.
- `VIDEO_INFERENCE_WIDTH` — downscale frames before AI inference.
- `VIDEO_PROGRESS_INTERVAL` — reduce DB writes during long jobs.
- `VIDEO_EVENT_DEDUPE_FRAMES` — suppress duplicate alerts for the same person across nearby frames.

**Tests**

```powershell
pytest -q
```

---

## 9. Docker preparation strategy

- **`Dockerfile`**: single image with runtime deps for OpenCV headless + API default CMD.
- **`docker-compose.yml`**: `api` on `8000`, `dashboard` on `8501` with `ECOFACE_API_BASE` pointing at the internal service name.

**Next hardening steps (when you enter Phase 5/6):** non-root user, pinned base image digest, healthcheck on `/api/v1/health`, volume mounts for `data/` and `logs/`, `.env` managed via SSM/Secrets Manager on EC2.

---

## 10. Future scalability (intentionally incremental)

- **External job queue for long videos/cloud:** replace the in-process worker with Celery/RQ/Arq/SQS while reusing the same `video_service.run_async_video_job` entry point.
- **PostgreSQL + Alembic:** add migrations; consider `pgvector` for similarity search at scale.
- **AuthN/Z:** FastAPI dependencies for `get_current_user` + route groups; DB tables for org/camera RBAC.
- **Multi-camera / RTSP:** implement `VideoSource` subclasses; reuse the same `RecognitionPipeline.process_frame` contract.
- **Anti-spoofing:** add a `LivenessChecker` interface in `ai_engine/` called from the pipeline behind a feature flag.

---

## 11. Architectural decisions (short “why” list)

- **API-first + Streamlit as a client:** avoids duplicating business rules in the UI while staying within your “no SPA yet” constraint.
- **Service layer:** keeps FastAPI thin and gives you a stable place to add transactions, idempotency keys, and job orchestration later.
- **SQLite first:** fastest solo velocity; SQLAlchemy keeps the door open to Postgres.
- **Lazy imports for CV/AI:** keeps developer feedback loops fast and CI cheap until you add optional “full stack” jobs.

---

## 12. License / academic use

Add your university-required license/academic notice here when you publish the report or public repo.
