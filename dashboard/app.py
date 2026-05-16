"""Streamlit dashboard — thin client over the FastAPI backend.

Why Streamlit here:
- Matches your roadmap (no SPA framework yet) while staying decoupled from the API.
- You can swap this layer later for React/Vue without touching the AI engine or DB schema.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import requests
import streamlit as st

API_BASE = os.environ.get("ECOFACE_API_BASE", "http://127.0.0.1:8000/api/v1")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _release_live_camera() -> None:
    cap = st.session_state.get("live_camera_capture")
    if cap is not None:
        cap.release()
    st.session_state.pop("live_camera_capture", None)


st.set_page_config(page_title="EcoFace Lite", layout="wide")
st.title("EcoFace Lite — Dashboard")
st.caption(f"API: `{API_BASE}`")

health_tab, enroll_tab, live_test_tab, video_tab, detections_tab, observability_tab = st.tabs(
    [
        "Health",
        "Enroll missing person",
        "Live Recognition Test",
        "Video processing",
        "Detections",
        "Observability",
    ]
)

with health_tab:
    if st.button("Ping API"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=10)
            st.json(r.json())
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

with enroll_tab:
    name = st.text_input("Display name")
    notes = st.text_area("Notes (optional)", height=80)
    image = st.file_uploader("Missing person image", type=["jpg", "jpeg", "png"])
    if st.button("Upload & enroll"):
        if not name or not image:
            st.warning("Provide a name and an image.")
        else:
            files = {"image": (image.name, image.getvalue(), image.type or "application/octet-stream")}
            data = {"display_name": name, "notes": notes or ""}
            try:
                r = requests.post(f"{API_BASE}/persons", files=files, data=data, timeout=120)
                if r.ok:
                    st.success("Person enrolled")
                    payload = r.json()
                    if payload.get("deduplicated"):
                        st.info("Same image bytes as an existing enrollment — returning existing person (200 OK).")
                    st.json(payload.get("person", payload))
                else:
                    st.error(r.text)
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

with live_test_tab:
    st.markdown("Click **Start live camera** to enable webcam matching. Click **Stop live camera** to turn it off.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start live camera", key="btn_start_live_cam"):
            _release_live_camera()
            st.session_state.live_camera_running = True
            st.session_state.pop("live_matched_frame", None)
            st.session_state.pop("live_match_result", None)
            st.rerun()
    with c2:
        if st.button("Stop live camera", key="btn_stop_live_cam"):
            st.session_state.live_camera_running = False
            _release_live_camera()
            st.rerun()

    live_status = st.empty()
    live_frame_box = st.empty()
    live_result_box = st.empty()
    matched_frame = st.session_state.get("live_matched_frame")
    matched_result = st.session_state.get("live_match_result")
    if matched_frame is not None and matched_result:
        live_frame_box.image(matched_frame, caption="Matched frame")
        score = matched_result.get("similarity_score")
        threshold = matched_result.get("threshold")
        live_result_box.success(
            f"Matched **{matched_result.get('person_name')}** "
            f"(score: `{score:.3f}`, threshold: `{threshold:.3f}`)"
        )

    if st.session_state.get("live_camera_running"):
        live_status.info("Live camera is running.")
        cap = st.session_state.get("live_camera_capture")
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
            st.session_state.live_camera_capture = cap
        ok, frame = cap.read()
        if not ok:
            live_status.error("Could not read from laptop camera.")
            st.session_state.live_camera_running = False
            _release_live_camera()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_frame_box.image(frame_rgb, caption="Live camera frame")
            encoded_ok, encoded = cv2.imencode(".jpg", frame)
            if encoded_ok:
                files = {"image": ("live-camera.jpg", encoded.tobytes(), "image/jpeg")}
                try:
                    r = requests.post(f"{API_BASE}/test-match", files=files, timeout=120)
                    if not r.ok:
                        live_result_box.error(r.text)
                    else:
                        result = r.json()
                        score = result.get("similarity_score")
                        threshold = result.get("threshold")
                        if result.get("matched"):
                            st.session_state.live_camera_running = False
                            st.session_state.live_matched_frame = frame_rgb
                            st.session_state.live_match_result = result
                            _release_live_camera()
                            live_result_box.success(
                                f"Live match: **{result.get('person_name')}** "
                                f"(score: `{score:.3f}`, threshold: `{threshold:.3f}`)"
                            )
                            st.rerun()
                        elif score is not None:
                            live_result_box.warning(
                                f"No live match. Top score: `{score:.3f}` / threshold `{threshold:.3f}`."
                            )
                        else:
                            live_result_box.info(result.get("detail") or "No face detected.")
                except requests.RequestException as e:
                    live_result_box.error(f"Request failed: {e}")
            time.sleep(0.2)
            st.rerun()
    else:
        _release_live_camera()
        live_status.info("Live camera is stopped.")

with video_tab:
    st.markdown(
        "Upload or select a video, start a **background job**, then the page polls status/results without "
        "blocking the API request."
    )
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"])
    rel = st.text_input("Or use video path already under `data/videos`", value="", key="video_rel_path")
    c1, c2, c3 = st.columns(3)
    with c1:
        upload_async = st.button("Upload & start job", key="btn_upload_video_async")
    with c2:
        start_async = st.button("Start path job", key="btn_start_video_async")
    with c3:
        if st.button("Stop polling", key="btn_stop_video_poll"):
            st.session_state.pop("video_job_id", None)
            st.session_state.pop("video_poll_count", None)
            st.rerun()

    if upload_async:
        if not uploaded_video:
            st.warning("Choose a video file first.")
        else:
            files = {
                "video": (
                    uploaded_video.name,
                    uploaded_video.getvalue(),
                    uploaded_video.type or "application/octet-stream",
                )
            }
            try:
                r = requests.post(f"{API_BASE}/videos/upload-and-process", files=files, timeout=120)
                if not r.ok:
                    st.error(r.text)
                else:
                    st.session_state.video_job_id = r.json()["job_id"]
                    st.session_state.video_poll_count = 0
                    st.success(f"Job queued: `{st.session_state.video_job_id}`")
                    st.rerun()
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    if start_async and rel.strip():
        try:
            r = requests.post(
                f"{API_BASE}/videos/process/async",
                json={"video_relative_path": rel.strip()},
                timeout=60,
            )
            if not r.ok:
                st.error(r.text)
            else:
                st.session_state.video_job_id = r.json()["job_id"]
                st.session_state.video_poll_count = 0
                st.success(f"Job queued: `{st.session_state.video_job_id}`")
                st.rerun()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

    job_id = st.session_state.get("video_job_id")
    if job_id:
        st.session_state.video_poll_count = int(st.session_state.get("video_poll_count") or 0) + 1
        if st.session_state.video_poll_count > 1200:
            st.warning("Stopped polling (timeout). Clear and restart if needed.")
            st.session_state.pop("video_job_id", None)
            st.session_state.pop("video_poll_count", None)
        else:
            try:
                r = requests.get(f"{API_BASE}/videos/processing-status/{job_id}", timeout=10)
                if not r.ok:
                    st.error(r.text)
                else:
                    data = r.json()
                    total = max(int(data.get("total_frames") or 0), 1)
                    processed = int(data.get("processed_frames") or 0)
                    alerts = int(data.get("alerts_created") or 0)
                    st.progress(min(1.0, processed / total))
                    st.caption(
                        f"**{data.get('status')}** — frames **{processed}** / **{total}** — alerts **{alerts}**"
                    )
                    st.json(
                        {
                            "avg_fps": data.get("avg_fps"),
                            "avg_confidence": data.get("avg_confidence"),
                            "total_faces_detected": data.get("total_faces_detected"),
                            "total_faces_rejected": data.get("total_faces_rejected"),
                            "blur_rejections": data.get("blur_rejections"),
                            "duplicate_suppressions": data.get("duplicate_suppressions"),
                            "processing_duration_seconds": data.get("processing_duration_seconds"),
                        }
                    )
                    preview = requests.get(f"{API_BASE}/videos/processing-preview/{job_id}", timeout=5)
                    if preview.ok and preview.json().get("preview_url"):
                        preview_url = preview.json()["preview_url"]
                        st.image(
                            f"{API_BASE.rsplit('/api/v1', 1)[0]}{preview_url}?t={time.time()}",
                            caption="Latest annotated processing preview",
                        )
                        st.caption("Legend: 🟨 rejected before/at quality stage · 🟥 embedded but not validated · 🟩 confirmed alert")
                    rejected = requests.get(f"{API_BASE}/videos/processing-rejected-faces/{job_id}", params={"limit": 12}, timeout=5)
                    if rejected.ok and rejected.json().get("items"):
                        st.write("Rejected face debug crops")
                        base_url = API_BASE.rsplit("/api/v1", 1)[0]
                        cols = st.columns(4)
                        for idx, item in enumerate(rejected.json()["items"]):
                            meta = item.get("metadata", {})
                            cols[idx % 4].image(
                                f"{base_url}{item['image_url']}",
                                caption=f"{meta.get('reason')} | {meta.get('face_width')}x{meta.get('face_height')} | {meta.get('detector_confidence')}",
                            )
                    if data.get("status") == "failed":
                        st.error(data.get("error_message") or "Job failed")
                        st.session_state.pop("video_job_id", None)
                        st.session_state.pop("video_poll_count", None)
                    elif data.get("status") == "completed":
                        st.success("Processing complete.")
                        st.session_state.pop("video_job_id", None)
                        st.session_state.pop("video_poll_count", None)
                    else:
                        time.sleep(0.4)
                        st.rerun()
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

with detections_tab:
    c1, c2 = st.columns(2)
    with c1:
        refresh_detections = st.button("Refresh detections")
    with c2:
        auto_refresh = st.checkbox("Auto refresh every 3 seconds", value=False)

    if refresh_detections or auto_refresh:
        try:
            r = requests.get(f"{API_BASE}/detections", params={"limit": 100}, timeout=30)
            if not r.ok:
                st.error(r.text)
            else:
                rows = r.json()
                display_rows = [
                    {
                        "timestamp": row.get("created_at"),
                        "name": row.get("person_name") or f"person_id={row.get('person_id')}",
                        "score": row.get("confidence"),
                        "threshold": row.get("threshold_used"),
                        "source": row.get("source_type"),
                        "frame": row.get("frame_index"),
                        "snapshot": row.get("snapshot_path"),
                    }
                    for row in rows
                ]
                st.dataframe(display_rows, use_container_width=True)
                for row in rows[:20]:
                    snap = row.get("snapshot_path")
                    if snap:
                        p = _project_root() / snap
                        if p.is_file():
                            st.image(
                                str(p),
                                caption=(
                                    f"{row.get('created_at')} | {row.get('person_name') or row.get('person_id')} | "
                                    f"{row.get('source_type')} | score={row.get('confidence'):.3f}"
                                ),
                            )
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
        if auto_refresh:
            time.sleep(3)
            st.rerun()

with observability_tab:
    st.subheader("System Metrics")
    c1, c2 = st.columns(2)
    with c1:
        refresh_metrics = st.button("Refresh observability")
    with c2:
        auto_metrics = st.checkbox("Auto refresh every 5 seconds", value=False, key="auto_observability_refresh")

    if refresh_metrics or auto_metrics:
        try:
            metrics_resp = requests.get(f"{API_BASE}/observability/metrics", timeout=10)
            diagnostics_resp = requests.get(f"{API_BASE}/observability/diagnostics", timeout=10)
            logs_resp = requests.get(f"{API_BASE}/observability/logs/recent", params={"limit": 80}, timeout=10)
            if not metrics_resp.ok:
                st.error(metrics_resp.text)
            elif not diagnostics_resp.ok:
                st.error(diagnostics_resp.text)
            elif not logs_resp.ok:
                st.error(logs_resp.text)
            else:
                metrics_data = metrics_resp.json()
                diagnostics_data = diagnostics_resp.json()
                counters = metrics_data.get("counters", {})
                averages = metrics_data.get("averages", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Frames processed", counters.get("total_frames_processed", 0))
                m2.metric("Faces detected", counters.get("total_faces_detected", 0))
                m3.metric("Events created", counters.get("detection_events_created", 0))
                m4.metric("Avg FPS", f"{averages.get('average_processing_fps', 0):.2f}")
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Detection cycles", counters.get("detection_cycles", 0))
                t2.metric("Tracking cycles", counters.get("tracking_cycles", 0))
                t3.metric("Early rejections", counters.get("detector_early_rejections", 0))
                t4.metric("Overload warnings", counters.get("detector_overload_warnings", 0))
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Yellow boxes", counters.get("yellow_box_count", 0))
                b2.metric("Red boxes", counters.get("red_box_count", 0))
                b3.metric("Green boxes", counters.get("green_box_count", 0))
                b4.metric("Missed-face estimate", counters.get("detector_missed_face_estimate", 0))

                stats_tab, diag_tab, fps_tab, confidence_tab, logs_tab = st.tabs(
                    [
                        "Processing Statistics",
                        "Pipeline Diagnostics",
                        "FPS Monitoring",
                        "Confidence Trends",
                        "Recent Errors/Warnings",
                    ]
                )
                with stats_tab:
                    st.json({"counters": counters, "averages": averages})
                    st.write("Face-size distributions")
                    size_cols = st.columns(2)
                    size_cols[0].line_chart(metrics_data.get("recent_values", {}).get("avg_detected_face_area", []))
                    size_cols[1].line_chart(metrics_data.get("recent_values", {}).get("avg_rejected_face_area", []))
                    st.write("Tracking and visibility")
                    st.json(
                        {
                            "tracker_refresh_count": counters.get("tracker_refresh_count", 0),
                            "stale_track_replacements": counters.get("stale_track_replacements", 0),
                            "face_visibility_ratio": averages.get("face_visibility_ratio", 0),
                            "overlay_debug_frame_count": counters.get("overlay_debug_frame_count", 0),
                            "rejected_face_snapshots_saved": counters.get("rejected_face_snapshots_saved", 0),
                        }
                    )
                with diag_tab:
                    st.write("Rejection and event reasons")
                    st.json(
                        {
                            "reason_counts": diagnostics_data.get("reason_counts", {}),
                            "category_counts": diagnostics_data.get("category_counts", {}),
                        }
                    )
                    st.dataframe(diagnostics_data.get("recent_events", []), use_container_width=True)
                with fps_tab:
                    recent_fps = metrics_data.get("recent_values", {}).get("average_processing_fps", [])
                    recent_latency = metrics_data.get("recent_values", {}).get("total_frame_processing_duration", [])
                    st.line_chart(recent_fps)
                    st.write("Frame processing latency")
                    st.line_chart(recent_latency)
                with confidence_tab:
                    confidence_values = diagnostics_data.get("confidence_values", [])
                    st.metric("Average confidence", f"{diagnostics_data.get('average_confidence', 0):.3f}")
                    st.line_chart(confidence_values)
                with logs_tab:
                    lines = logs_resp.json().get("lines", [])
                    warning_lines = [
                        line for line in lines if "WARNING" in line or "ERROR" in line or "rejected" in line.lower()
                    ]
                    st.text_area("Recent warnings/errors/rejections", "\n".join(warning_lines[-80:]), height=400)
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
        if auto_metrics:
            time.sleep(5)
            st.rerun()
