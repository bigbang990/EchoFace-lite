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

## Decision: YOLOv8 feasibility — Gate A failure (2026-06-08) — RESOLVED
weights/yolov8n-face.pt downloaded (6.2 MB via HuggingFace arnabdhar mirror).
scripts/download_yolov8_face.py created. Gate A now passes.

## Decision: YOLOv8 feasibility — Gate B environment constraint (2026-06-08)
Gate B failed locally: torch and ultralytics not installed on the local Windows
machine. Project is Colab-first; no local ML environment exists.
Alternative: run gates B–E inside Google Colab after mounting the repo.
URL attempted for weights: https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt — succeeded.

## Decision: Phase 6 merge approved
Date: June 2026
Evidence: stable_matches=183, alerts=2, detector_runtime=17ms,
          identity_switch_rate=0, FPS=56 on T4
Known debt: ghost_survival 26s (starvation override),
            resolution cap 480px (detection_optimizer.py:112)
Neither blocks production use.

## Decision: YOLOv8-face weights source (June 2026)
Confirmed source: derronqi/yolov8-face (Google Drive)
Drive ID: 1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb
Gate results: A PASS(6.4MB)  B PASS(cuda)  C PASS(5-kpt [N,5,2])
              D PASS(8.5ms avg, 117.9fps T4)
Rejected: arnabdhar/YOLOv8-Face-Detection — detection-only, no keypoint head
  (6.2MB HuggingFace mirror; Gate C would fail — r.keypoints is None)
Rejected: akanametov/yolo-face — detection-only fork, no keypoint head
Keypoint order confirmed: [left_eye, right_eye, nose, left_mouth, right_mouth]
  — matches FaceLandmarks convention exactly.
