# Key architectural decisions

## Decision: detector abstraction layer
Date: June 2026
Context: onnxruntime-gpu cannot use GPU on CUDA 12.8.
         5 wheel versions tested, all failed.
Decision: Abstract detector behind FaceDetector interface (ABC in detector.py).
          SCRFD and YOLOv8-face are both implementations.
          DETECTOR_PROVIDER env var will select at runtime (not yet wired).
          Hardware does not decide the detector. Config does.
Reverts cleanly: one env var change, no code migration.

## Decision: no open-world enrollment
The pipeline does not create new person_ids during video processing.
DB gallery is static per job.
Matching is against enrolled persons only.
Fix for now: one target person enrolled per test session.
Future: Incident/Case system scopes gallery by open cases.

## Decision: Incident/Case system (future)
Scope: data layer + API/dashboard only.
Detection/tracking/embedding untouched.
Core model: Incident (id, status OPEN/CLOSED, persons[])
Pipeline receives gallery filtered to OPEN incidents only.
Events stamped with incident_id.
Not implemented until core model is stable.

## Decision: landmarks required, not optional
face_candidate_validator hard-rejects landmarks=None.
Any detector swap must provide 5-point landmarks.
Landmark source does not affect embedding quality
(no affine alignment in codebase — verified June 2026).

## Decision: single shared FaceAnalysis instance
bootstrap.py constructs one FaceAnalysis and injects it into both
InsightFaceDetector and InsightFaceEmbedder.
Reason: InsightFace model weights load once per worker (~seconds).
Do not construct separate instances for detector and embedder.

## Decision: onnxruntime CPU-only on CUDA 12.8
onnxruntime-gpu has no CUDA 12.8 wheel.
CPUExecutionProvider is the only viable provider.
Do not attempt CUDAExecutionProvider — it will fail at runtime.
