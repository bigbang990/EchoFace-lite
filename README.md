# EchoFace Lite

Real-time missing person re-identification from surveillance footage.  
Multi-face tracking · ArcFace embeddings · SCRFD detection · FastAPI backend · React investigation frontend

**Version:** v1.0.1 | **Runtime:** Google Colab T4 / local CPU | **Branch:** phase8-pipeline-decompose

---

## What it does

Operators create a **case** for a missing person, enroll reference photos, feed in CCTV footage,
and receive **alerts** when the AI detects a match. The full lifecycle:

```
Create Case → Enroll Person → Activate Tracking →
Monitor Sources → Receive Alert → Investigate → Resolve → Close
```

The frontend is a dark, professional investigation platform. The backend is a FastAPI service
with a SQLite store, running the SORT tracker and ArcFace embedder.

---

## Quick Start

### Backend (Google Colab or local)

1. Open `notebooks/colab_setup.ipynb` or paste the setup cell
2. Run setup — installs deps, starts FastAPI on port 8000, opens ngrok tunnel
3. Copy the ngrok URL

```python
SERVER_ENV = {
    "DETECTOR_PROVIDER":                  "insightface",
    "INSIGHTFACE_CTX_ID":                 "-1",
    "GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE":"15",
}
```

### Frontend (local)

Requires Node.js 18+ — install from [nodejs.org](https://nodejs.org) if not present.

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Enter `DEMO` or `ADMIN` at the access gate.

> Vite proxies `/api/*` to `http://localhost:8000` — set the backend URL in
> `frontend/vite.config.ts` if your backend runs elsewhere.

---

## Frontend — Screens

| Screen | Route | Access |
|---|---|---|
| Overview | `/` | DEMO + ADMIN |
| Create Case | `/cases/new` | DEMO + ADMIN |
| Operations | `/operations` | DEMO + ADMIN |
| Cases | `/cases` | DEMO + ADMIN |
| Case Workspace | `/cases/:id` | DEMO + ADMIN |
| System Health | `/system-health` | ADMIN only |

**Access codes** — entered at the gate on first load:
- `DEMO` — clean investigation view; ideal for presentations
- `ADMIN` — same plus live telemetry tiles, System Health screen, greyed-out Administration stub (v2.0)

**Frontend stack:** Vite 5 · React 18 · TypeScript · Tailwind CSS 3 · Framer Motion 11 ·
Zustand 4 · React Router 6 · Recharts 2 · Lucide React

All data is currently mocked in `frontend/src/mock/data.ts`. Shapes match the real API
responses — wiring is a data-source swap, no component changes needed.

---

## API Endpoints

### Incidents (Cases)
| Method | Path | Description |
|---|---|---|
| GET | /api/v1/incidents | List all incidents |
| POST | /api/v1/incidents | Create incident |
| GET | /api/v1/incidents/{id} | Get incident |
| PATCH | /api/v1/incidents/{id}/status | Update status |
| GET | /api/v1/incidents/{id}/persons | List enrolled persons |
| POST | /api/v1/incidents/{id}/persons/{person_id} | Link person to incident |
| DELETE | /api/v1/incidents/{id}/persons/{person_id} | Unlink person |
| GET | /api/v1/incidents/{id}/sightings | List sightings / alerts |

### Persons
| Method | Path | Description |
|---|---|---|
| GET | /api/v1/persons | List all enrolled persons |
| POST | /api/v1/persons | Enroll person (single photo) |
| POST | /api/v1/persons/{id}/photos | Add reference photos (max 5) |

### Cameras
| Method | Path | Description |
|---|---|---|
| GET | /api/v1/cameras | List all camera sources |
| POST | /api/v1/cameras | Register camera source |
| GET | /api/v1/cameras/{id} | Get camera |

### Processing
| Method | Path | Description |
|---|---|---|
| POST | /api/v1/processing | Submit video job (async) |
| GET | /api/v1/processing/{job_id} | Poll job status + metrics |

### Observability
| Method | Path | Description |
|---|---|---|
| GET | /api/v1/observability/metrics | Live telemetry — fps, latency, tracks, identity_switch_rate |
| GET | /api/v1/health | Health check |

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │  React Frontend  (frontend/)             │
                        │  AccessGate → Layout → 5 screens         │
                        │  Mock data → real API swap in next phase │
                        └─────────────┬───────────────────────────┘
                                      │ /api/v1/*  (Vite proxy)
                        ┌─────────────▼───────────────────────────┐
                        │  FastAPI  (ecoface_lite/)                │
                        │  routers: incidents · persons · cameras  │
                        │           processing · observability     │
                        └─────────────┬───────────────────────────┘
                                      │
          ┌───────────────────────────▼────────────────────────────┐
          │                   Pipeline                             │
          │  Video → Detector (SCRFD/YOLO) → Validator → Tracker   │
          │       → Embedder (ArcFace) → Matcher → Sighting store  │
          └───────────────────────────┬────────────────────────────┘
                                      │
                              SQLite (aiosqlite)
```

**Detector selection:** set `DETECTOR_PROVIDER=insightface` (SCRFD) or `yolo` (YOLOv8).  
Platform bootstrap auto-selects CPU/GPU config at startup — no manual tuning.

---

## GPU / CPU Baselines

| Metric | GPU (T4) | CPU |
|---|---|---|
| Average FPS | 81 | ~12 |
| Detector latency | 13.2 ms | ~220 ms |
| Identity switch rate | 0 | 0 |
| Stable matches | 50/50 | 50/50 |
| det_size | (640, 640) | (320, 320) |

---

## Regression Gate

Run before every merge to main:

```bash
python -m pytest tests/ -v --tb=short
python -m identity_stress_suite
```

Expected: all tests pass · identity_switch_rate = 0 · stable_matches ≥ baseline.
If any metric regresses: **do not merge**.

---

## Known Constraints

- `onnxruntime-gpu` has no stable CUDA 12.8 wheel — ONNX runs CPU-only on Colab T4
- `.env` files unreliable on Colab — use `SERVER_ENV` dict injection
- Multi-CCTV grid view: Phase 9 (not in current release)
- Real auth / RBAC: Phase 10 (frontend access gate is a mode-switch placeholder)

---

## Roadmap

See [DISSERTATION_NOTES.md](DISSERTATION_NOTES.md) and [AI_CONTEXT/roadmap.md](AI_CONTEXT/roadmap.md).

**Next phase:** Wire frontend to live FastAPI backend — replace `frontend/src/mock/data.ts`
exports with fetch/SWR calls. No component changes required.
