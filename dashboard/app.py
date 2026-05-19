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
import streamlit as st

from dashboard.api_client import APIClient
from dashboard.backend_registry import BACKENDS, DEFAULT_BACKEND


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _release_live_camera() -> None:
    cap = st.session_state.get("live_camera_capture")
    if cap is not None:
        cap.release()
    st.session_state.pop("live_camera_capture", None)


def _cleanup_api_client() -> None:
    client = st.session_state.get("api_client")
    if client is not None:
        client.close()
        st.session_state.pop("api_client", None)


def _get_api_client() -> APIClient:
    if "api_client" not in st.session_state:
        backend_name = st.session_state.get("active_backend", DEFAULT_BACKEND)
        backend_config = BACKENDS.get(backend_name, BACKENDS[DEFAULT_BACKEND])
        st.session_state.api_client = APIClient(backend_config["url"])
    return st.session_state.api_client


def _render_connection_status() -> None:
    client = _get_api_client()
    backend_name = st.session_state.get("active_backend", DEFAULT_BACKEND)
    backend_config = BACKENDS.get(backend_name, BACKENDS[DEFAULT_BACKEND])
    
    health = client.health()
    latency = client.get_latency_ms()
    
    status_color = "🟢" if health.get("status") == "ok" else "🔴"
    
    with st.sidebar:
        st.divider()
        st.subheader("Connection Status")
        col1, col2 = st.columns(2)
        with col1:
            st.text("Backend:")
            st.text("Status:")
            st.text("Latency:")
            st.text("Type:")
        with col2:
            st.text(backend_name)
            st.text(f"{status_color} {'Online' if health.get('status') == 'ok' else 'Offline'}")
            st.text(f"{latency:.0f} ms")
            st.text(backend_config["type"])
        
        if health.get("status") == "ok":
            st.caption(f"API: v{health.get('version', 'unknown')}")
            if health.get("gpu"):
                st.caption(f"GPU: {health.get('device', 'unknown')}")
            else:
                st.caption(f"Device: {health.get('device', 'CPU')}")
        
        version_ok, version_msg = client.validate_version()
        if not version_ok:
            st.warning(version_msg)


st.set_page_config(page_title="EcoFace Lite", layout="wide")
st.title("EcoFace Lite — Dashboard")

if "active_backend" not in st.session_state:
    st.session_state.active_backend = DEFAULT_BACKEND

with st.sidebar:
    st.subheader("Backend Selection")
    backend_name = st.selectbox(
        "Select backend",
        options=list(BACKENDS.keys()),
        index=list(BACKENDS.keys()).index(st.session_state.active_backend),
        key="backend_selector",
    )
    
    if backend_name != st.session_state.active_backend:
        _release_live_camera()
        _cleanup_api_client()
        st.session_state.active_backend = backend_name
        st.session_state.video_job_id = None
        st.session_state.video_poll_count = None
        st.rerun()

_render_connection_status()

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
        client = _get_api_client()
        result = client.health(use_cache=False)
        if "error" in result:
            st.error(result["error"])
        else:
            st.json(result)

with enroll_tab:
    name = st.text_input("Display name")
    notes = st.text_area("Notes (optional)", height=80)
    image = st.file_uploader("Missing person image", type=["jpg", "jpeg", "png"])
    if st.button("Upload & enroll"):
        if not name or not image:
            st.warning("Provide a name and an image.")
        else:
            client = _get_api_client()
            image_file = (image.name, image.getvalue(), image.type or "application/octet-stream")
            result = client.enroll_person(name, image_file, notes or "")
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Person enrolled")
                if result.get("deduplicated"):
                    st.info("Same image bytes as an existing enrollment — returning existing person (200 OK).")
                st.json(result.get("person", result))

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
                client = _get_api_client()
                image_file = ("live-camera.jpg", encoded.tobytes(), "image/jpeg")
                result = client.test_match(image_file)
                if "error" in result:
                    live_result_box.error(result["error"])
                else:
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
            client = _get_api_client()
            video_file = (
                uploaded_video.name,
                uploaded_video.getvalue(),
                uploaded_video.type or "application/octet-stream",
            )
            result = client.upload_video(video_file)
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.video_job_id = result["job_id"]
                st.session_state.video_poll_count = 0
                st.success(f"Job queued: `{st.session_state.video_job_id}`")
                st.rerun()

    if start_async and rel.strip():
        client = _get_api_client()
        result = client.process_video(rel.strip())
        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state.video_job_id = result["job_id"]
            st.session_state.video_poll_count = 0
            st.success(f"Job queued: `{st.session_state.video_job_id}`")
            st.rerun()

    job_id = st.session_state.get("video_job_id")
    if job_id:
        st.session_state.video_poll_count = int(st.session_state.get("video_poll_count") or 0) + 1
        if st.session_state.video_poll_count > 1200:
            st.warning("Stopped polling (timeout). Clear and restart if needed.")
            st.session_state.pop("video_job_id", None)
            st.session_state.pop("video_poll_count", None)
        else:
            client = _get_api_client()
            data = client.get_video_status(job_id)
            if "error" in data:
                st.error(data["error"])
            else:
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
                preview = client.get_preview(job_id)
                if preview.get("preview_url"):
                    preview_url = preview["preview_url"]
                    base_url = client.base_url.rsplit("/api/v1", 1)[0]
                    st.image(
                        f"{base_url}{preview_url}?t={time.time()}",
                        caption="Latest annotated processing preview",
                    )
                    st.caption("Legend: 🟨 rejected before/at quality stage · 🟥 embedded but not validated · 🟩 confirmed alert")
                rejected = client.get_rejected_faces(job_id, limit=12)
                if rejected.get("items"):
                    st.write("Rejected face debug crops")
                    base_url = client.base_url.rsplit("/api/v1", 1)[0]
                    cols = st.columns(4)
                    for idx, item in enumerate(rejected["items"]):
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

with detections_tab:
    c1, c2 = st.columns(2)
    with c1:
        refresh_detections = st.button("Refresh detections")
    with c2:
        auto_refresh = st.checkbox("Auto refresh every 3 seconds", value=False)

    if refresh_detections or auto_refresh:
        client = _get_api_client()
        result = client.get_detections(limit=100)
        if "error" in result:
            st.error(result["error"])
        else:
            rows = result
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
        client = _get_api_client()
        metrics_data = client.get_metrics()
        diagnostics_data = client.get_diagnostics()
        logs_data = client.get_logs(limit=80)
        
        if "error" in metrics_data:
            st.error(metrics_data["error"])
        elif "error" in diagnostics_data:
            st.error(diagnostics_data["error"])
        elif "error" in logs_data:
            st.error(logs_data["error"])
        else:
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
                st.subheader("Tracking Health")
                th1, th2, th3, th4 = st.columns(4)
                th1.metric("Active tracks", f"{averages.get('active_tracks', 0):.0f}")
                th2.metric("Recovered tracks", counters.get("recovered_tracks", 0))
                th3.metric("Tracker reuse rate", f"{averages.get('tracker_reuse_rate', 0):.1%}")
                th4.metric("Stale replacements", counters.get("stale_track_replacements", 0))
                st.subheader("Recognition Stability")
                rs1, rs2, rs3 = st.columns(3)
                rs1.metric("Identity switches", counters.get("identity_switches", 0))
                rs2.metric("Stable matches", counters.get("stable_matches", 0))
                rs3.metric("Embedding reuse", f"{averages.get('embedding_reuse_rate', 0):.1%}")
                st.subheader("Performance")
                pf1, pf2, pf3 = st.columns(3)
                pf1.metric("Detection cycles", counters.get("detection_cycles", 0))
                pf2.metric("Tracking cycles", counters.get("tracking_cycles", 0))
                pf3.metric("Embedding cache hits", counters.get("embedding_cache_hits", 0))
                st.write("Tracking and visibility")
                st.json(
                    {
                        "tracker_refresh_count": counters.get("tracker_refresh_count", 0),
                        "stale_track_replacements": counters.get("stale_track_replacements", 0),
                        "face_visibility_ratio": averages.get("face_visibility_ratio", 0),
                        "avg_track_lifetime": averages.get("avg_track_lifetime", 0),
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
                lines = logs_data.get("lines", [])
                warning_lines = [
                    line for line in lines if "WARNING" in line or "ERROR" in line or "rejected" in line.lower()
                ]
                st.text_area("Recent warnings/errors/rejections", "\n".join(warning_lines[-80:]), height=400)
        if auto_metrics:
            time.sleep(5)
            st.rerun()
