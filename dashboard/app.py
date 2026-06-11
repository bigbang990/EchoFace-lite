"""Streamlit dashboard — thin client over the FastAPI backend.

Why Streamlit here:
- Matches your roadmap (no SPA framework yet) while staying decoupled from the API.
- You can swap this layer later for React/Vue without touching the AI engine or DB schema.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import streamlit as st

# Add project root to sys.path for absolute imports
_project_root_path = Path(__file__).resolve().parents[1]
if str(_project_root_path) not in sys.path:
    sys.path.insert(0, str(_project_root_path))

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

health_tab, enroll_tab, live_test_tab, video_tab, detections_tab, cameras_tab, incidents_tab, observability_tab, experimental_tab = st.tabs(
    [
        "Health",
        "Enroll missing person",
        "Live Recognition Test",
        "Video processing",
        "Detections",
        "Cameras",
        "Incidents",
        "Observability",
        "Experimental Console",
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

with cameras_tab:
    st.subheader("Camera Registry")
    
    # Add camera form
    with st.form(key="add_camera_form"):
        label = st.text_input("Label", placeholder="e.g., Front Entrance")
        stream_url = st.text_input("Stream URL (optional)", placeholder="e.g., rtsp://camera1.local:554/stream")
        location = st.text_input("Location (optional)", placeholder="e.g., Building A, Floor 2")
        submitted = st.form_submit_button("Add Camera")
        
        if submitted:
            if not label:
                st.warning("Label is required.")
            else:
                client = _get_api_client()
                payload = {
                    "label": label,
                    "stream_url": stream_url or None,
                    "location": location or None,
                }
                try:
                    response = client._session.post(
                        f"{client.base_url}/cameras",
                        json=payload,
                        timeout=10,
                    )
                    response.raise_for_status()
                    st.success("Camera added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add camera: {str(e)}")
    
    st.divider()
    
    # List cameras
    refresh_cameras = st.button("Refresh Cameras")
    if refresh_cameras or "cameras_refreshed" not in st.session_state:
        st.session_state.cameras_refreshed = True
        client = _get_api_client()
        try:
            response = client._session.get(
                f"{client.base_url}/cameras",
                timeout=10,
            )
            response.raise_for_status()
            cameras = response.json()
            st.session_state.cameras = cameras
        except Exception as e:
            st.error(f"Failed to load cameras: {str(e)}")
            cameras = []
    
    if "cameras" in st.session_state and st.session_state.cameras:
        for camera in st.session_state.cameras:
            with st.expander(f"Camera {camera['id']}: {camera['label']}"):
                st.write(f"**Location:** {camera.get('location') or 'N/A'}")
                st.write(f"**Stream URL:** {camera.get('stream_url') or 'N/A'}")
                st.write(f"**Active:** {'Yes' if camera.get('is_active') else 'No'}")
                st.write(f"**Created At:** {camera.get('created_at')}")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button(f"Toggle Active", key=f"toggle_camera_{camera['id']}"):
                        client = _get_api_client()
                        try:
                            response = client._session.patch(
                                f"{client.base_url}/cameras/{camera['id']}",
                                json={"is_active": not camera.get('is_active')},
                                timeout=10,
                            )
                            response.raise_for_status()
                            st.success("Camera status updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update camera: {str(e)}")
                with c2:
                    if st.button(f"Delete Camera", key=f"delete_camera_{camera['id']}"):
                        client = _get_api_client()
                        try:
                            response = client._session.delete(
                                f"{client.base_url}/cameras/{camera['id']}",
                                timeout=10,
                            )
                            response.raise_for_status()
                            st.success("Camera deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete camera: {str(e)}")
    else:
        st.info("No cameras registered yet.")

with incidents_tab:
    st.subheader("Incident Management")
    
    # Create incident form
    with st.form(key="create_incident_form"):
        title = st.text_input("Title", placeholder="e.g., Unauthorized Person Detected")
        description = st.text_area("Description (optional)", placeholder="Additional details about the incident...")
        operator_id = st.text_input("Operator ID (optional)", placeholder="e.g., operator_123")
        submitted = st.form_submit_button("Create Incident")
        
        if submitted:
            if not title:
                st.warning("Title is required.")
            else:
                client = _get_api_client()
                payload = {
                    "title": title,
                    "description": description or None,
                    "operator_id": operator_id or None,
                }
                try:
                    response = client._session.post(
                        f"{client.base_url}/incidents",
                        json=payload,
                        timeout=10,
                    )
                    response.raise_for_status()
                    st.success("Incident created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create incident: {str(e)}")
    
    st.divider()
    
    # Filter by status
    status_filter = st.selectbox("Filter by Status", ["all", "open", "active", "closed"], index=0)
    
    # List incidents
    refresh_incidents = st.button("Refresh Incidents")
    if refresh_incidents or "incidents_refreshed" not in st.session_state:
        st.session_state.incidents_refreshed = True
        client = _get_api_client()
        try:
            response = client._session.get(
                f"{client.base_url}/incidents",
                timeout=10,
            )
            response.raise_for_status()
            incidents = response.json()
            st.session_state.incidents = incidents
        except Exception as e:
            st.error(f"Failed to load incidents: {str(e)}")
            incidents = []
    
    if "incidents" in st.session_state and st.session_state.incidents:
        # Filter incidents
        filtered_incidents = [
            inc for inc in st.session_state.incidents 
            if status_filter == "all" or inc.get("status") == status_filter
        ]
        
        for incident in filtered_incidents:
            # Status badge color
            status_color = "blue" if incident.get("status") == "open" else "orange" if incident.get("status") == "active" else "gray"
            
            with st.expander(f"Incident {incident['id']}: {incident['title']} ({incident['status']})"):
                st.write(f"**Description:** {incident.get('description') or 'N/A'}")
                st.write(f"**Status:** {incident.get('status')}")
                st.write(f"**Operator ID:** {incident.get('operator_id') or 'N/A'}")
                st.write(f"**Created At:** {incident.get('created_at')}")
                st.write(f"**Updated At:** {incident.get('updated_at') or 'N/A'}")
                
                # Update status buttons
                st.write("**Update Status:**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Open", key=f"open_incident_{incident['id']}"):
                        client = _get_api_client()
                        try:
                            response = client._session.patch(
                                f"{client.base_url}/incidents/{incident['id']}/status",
                                json={"status": "open"},
                                timeout=10,
                            )
                            response.raise_for_status()
                            st.success("Incident status updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update status: {str(e)}")
                with c2:
                    if st.button("Active", key=f"active_incident_{incident['id']}"):
                        client = _get_api_client()
                        try:
                            response = client._session.patch(
                                f"{client.base_url}/incidents/{incident['id']}/status",
                                json={"status": "active"},
                                timeout=10,
                            )
                            response.raise_for_status()
                            st.success("Incident status updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update status: {str(e)}")
                with c3:
                    if st.button("Closed", key=f"closed_incident_{incident['id']}"):
                        client = _get_api_client()
                        try:
                            response = client._session.patch(
                                f"{client.base_url}/incidents/{incident['id']}/status",
                                json={"status": "closed"},
                                timeout=10,
                            )
                            response.raise_for_status()
                            st.success("Incident status updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update status: {str(e)}")
                
                st.divider()
                
                # Add sighting form
                st.write("**Add Sighting:**")
                with st.form(key=f"add_sighting_{incident['id']}"):
                    detection_id = st.number_input("Detection ID (optional)", min_value=1, step=1, value=None)
                    camera_id = st.number_input("Camera ID (optional)", min_value=1, step=1, value=None)
                    notes = st.text_input("Notes (optional)", placeholder="Additional details about this sighting...")
                    add_sighting_submitted = st.form_submit_button("Add Sighting")
                    
                    if add_sighting_submitted:
                        client = _get_api_client()
                        payload = {
                            "detection_id": int(detection_id) if detection_id else None,
                            "camera_id": int(camera_id) if camera_id else None,
                            "notes": notes or None,
                        }
                        try:
                            response = client._session.post(
                                f"{client.base_url}/incidents/{incident['id']}/sightings",
                                json=payload,
                                timeout=10,
                            )
                            response.raise_for_status()
                            st.success("Sighting added successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to add sighting: {str(e)}")
                
                st.divider()
                
                # View sightings
                if st.button("View Sightings", key=f"view_sightings_{incident['id']}"):
                    client = _get_api_client()
                    try:
                        response = client._session.get(
                            f"{client.base_url}/incidents/{incident['id']}/sightings",
                            timeout=10,
                        )
                        response.raise_for_status()
                        sightings = response.json()
                        st.session_state[f"sightings_{incident['id']}"] = sightings
                    except Exception as e:
                        st.error(f"Failed to load sightings: {str(e)}")
                        sightings = []
                
                if f"sightings_{incident['id']}" in st.session_state:
                    st.write("**Sightings:**")
                    sightings = st.session_state[f"sightings_{incident['id']}"]
                    if sightings:
                        for sighting in sightings:
                            st.write(f"- Sighting {sighting['id']}:")
                            st.write(f"  - Detection ID: {sighting.get('detection_id') or 'N/A'}")
                            st.write(f"  - Camera ID: {sighting.get('camera_id') or 'N/A'}")
                            st.write(f"  - Notes: {sighting.get('notes') or 'N/A'}")
                            st.write(f"  - Created At: {sighting.get('created_at')}")
                    else:
                        st.info("No sightings for this incident.")
    else:
        st.info("No incidents created yet.")

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

with experimental_tab:
    st.subheader("Experimental Configuration Console")
    
    # Get current backend type for conditional UI
    backend_name = st.session_state.get("active_backend", DEFAULT_BACKEND)
    backend_config = BACKENDS.get(backend_name, BACKENDS[DEFAULT_BACKEND])
    is_local_cpu = backend_name == "Local CPU"
    
    # Load current overrides from backend
    client = _get_api_client()
    current_overrides = client.get_current_overrides()
    current_config = current_overrides.get("overrides", {})
    session_id = current_overrides.get("experiment_session_id", "N/A")
    
    st.info(f"Experiment Session ID: `{session_id}`")
    
    # Authoritative Form Control Loop - NO LIVE SLIDER RUNS
    with st.form(key="experimental_tuning_form"):
        st.subheader("Runtime Tuning Parameters")
        
        # Pipeline Mode
        pipeline_mode = st.selectbox(
            "Pipeline Mode",
            options=["LEGACY_ONLY", "HYBRID", "UNIFIED_ONLY"],
            index=["LEGACY_ONLY", "HYBRID", "UNIFIED_ONLY"].index(current_config.get("pipeline_mode", "HYBRID")),
            help="LEGACY_ONLY: Legacy validators ON, Unified OFF | HYBRID: Both ON | UNIFIED_ONLY: Unified ON, Legacy OFF"
        )
        
        # Core validator parameters
        col1, col2 = st.columns(2)
        with col1:
            validator_quality_cutoff = st.slider(
                "Validator Quality Cutoff",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get("validator_quality_cutoff", 0.35),
                step=0.01,
                help="Minimum quality score for validator acceptance (0.0-1.0)"
            )
            validator_strict_cutoff = st.slider(
                "Validator Strict Cutoff",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get("validator_strict_cutoff", 0.70),
                step=0.01,
                help="Strict threshold for high-confidence validation (0.0-1.0)"
            )
        with col2:
            validator_min_detector_confidence = st.slider(
                "Min Detector Confidence",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get("validator_min_detector_confidence", 0.45),
                step=0.01,
                help="Minimum detection confidence for validator input (0.0-1.0)"
            )
            validator_min_blur_var = st.slider(
                "Min Blur Variance",
                min_value=0.0,
                max_value=255.0,
                value=current_config.get("validator_min_blur_var", 45.0),
                step=1.0,
                help="Minimum blur variance threshold (0.0-255.0)"
            )
        
        # Advanced Experimental Tuning (expander)
        with st.expander("Advanced Experimental Tuning"):
            col3, col4 = st.columns(2)
            with col3:
                validator_max_faces_per_frame = st.slider(
                    "Max Faces Per Frame",
                    min_value=1,
                    max_value=12,
                    value=current_config.get("validator_max_faces_per_frame", 8),
                    step=1,
                    help="Maximum faces to process per frame (1-12)"
                )
                validator_min_quality_for_embedding = st.slider(
                    "Min Quality for Embedding",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_config.get("validator_min_quality_for_embedding", 0.55),
                    step=0.01,
                    help="Minimum quality for embedding generation (0.0-1.0)"
                )
            with col4:
                tracker_detection_interval = st.slider(
                    "Tracker Detection Interval",
                    min_value=1,
                    max_value=18,
                    value=current_config.get("tracker_detection_interval", 8),
                    step=1,
                    help="Frames between detection runs (1-18)"
                )
                track_confirmation_frames = st.slider(
                    "Track Confirmation Frames",
                    min_value=1,
                    max_value=10,
                    value=current_config.get("track_confirmation_frames", 2),
                    step=1,
                    help="Frames required for track confirmation (1-10)"
                )
            
            track_lost_buffer = st.slider(
                "Track Lost Buffer",
                min_value=1,
                max_value=30,
                value=current_config.get("track_lost_buffer", 18),
                step=1,
                help="Frames to track lost face before deletion (1-30)"
            )
            
            identity_match_threshold = st.slider(
                "Identity Match Threshold",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get("identity_match_threshold", 0.38),
                step=0.01,
                help="Cosine similarity threshold for identity matching (0.0-1.0)"
            )
        
        # CPU Safeguards Configuration (conditional UI)
        st.subheader("CPU Protection Safeguards")
        if is_local_cpu:
            cpu_protection = client.get_cpu_protection_state()
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric(
                    "Protection Status",
                    "ACTIVE" if cpu_protection.get("protection_active") else "INACTIVE",
                    delta=f"Events: {cpu_protection.get('overload_event_count', 0)}"
                )
                st.metric(
                    "Current Detection Interval",
                    cpu_protection.get("current_detection_interval", 8),
                    delta="Elevated" if cpu_protection.get("current_detection_interval", 8) > 8 else "Normal"
                )
            with col6:
                st.metric(
                    "Embedding Suppression",
                    "ACTIVE" if cpu_protection.get("embedding_suppression_active") else "INACTIVE"
                )
                st.metric(
                    "Debug Truncation",
                    "ACTIVE" if cpu_protection.get("debug_truncation_active") else "INACTIVE"
                )
            
            st.caption("CPU safeguards are active on LOCAL_CPU backend")
        else:
            st.warning("CPU safeguards inactive on GPU backends")
            st.caption("CPU protection only applies to LOCAL_CPU backend. Current backend: " + backend_name)
        
        # Submit button
        submitted = st.form_submit_button("Apply Changes", type="primary")
        
        if submitted:
            # Build config payload
            config_payload = {
                "pipeline_mode": pipeline_mode,
                "validator_quality_cutoff": validator_quality_cutoff,
                "validator_strict_cutoff": validator_strict_cutoff,
                "validator_min_detector_confidence": validator_min_detector_confidence,
                "validator_min_blur_var": validator_min_blur_var,
                "validator_max_faces_per_frame": validator_max_faces_per_frame,
                "validator_min_quality_for_embedding": validator_min_quality_for_embedding,
                "tracker_detection_interval": tracker_detection_interval,
                "track_confirmation_frames": track_confirmation_frames,
                "track_lost_buffer": track_lost_buffer,
                "identity_match_threshold": identity_match_threshold,
            }
            
            # POST to backend
            result = client.apply_overrides(config_payload)
            
            if "error" in result:
                st.error(f"Failed to apply changes: {result['error']}")
            else:
                st.success(f"Changes applied successfully!")
                applied = result.get("applied", {})
                if applied:
                    st.json(applied)
                # Force rerun to show clamped values
                st.rerun()
    
    # Reset button (outside form)
    col7, col8 = st.columns(2)
    with col7:
        if st.button("Reset to Defaults", key="btn_reset_experimental"):
            result = client.reset_experimental_settings()
            if "error" in result:
                st.error(f"Failed to reset: {result['error']}")
            else:
                st.success(f"Reset complete. New session: `{result.get('new_session_id')}`")
                st.rerun()
    
    with col8:
        if st.button("Refresh Status", key="btn_refresh_experimental"):
            st.rerun()
    
    st.divider()
    
    # Experiment Comparison Panel
    st.subheader("Experiment Comparison")
    comparison_subtabs = st.tabs(["History", "Action Log"])
    
    with comparison_subtabs[0]:
        # Load experiment snapshots
        snapshots_result = client.get_experiment_snapshots(limit=20)
        
        if "error" in snapshots_result:
            st.error(f"Failed to load experiment history: {snapshots_result['error']}")
        else:
            snapshots = snapshots_result.get("snapshots", [])
            
            if not snapshots:
                st.info("No experiment history yet. Process videos with experimental configuration to populate history.")
            else:
                # Build comparison table
                comparison_rows = []
                for snap in snapshots:
                    metrics = snap.get("raw_metrics", {})
                    comparison_rows.append({
                        "Run ID": snap.get("snapshot_id", "N/A")[:8],
                        "Mode": snap.get("pipeline_mode", "N/A"),
                        "FPS": f"{metrics.get('average_processing_fps', 0):.2f}",
                        "Stable Matches": metrics.get("stable_matches", 0),
                        "Fragmentation": metrics.get("identity_switches", 0),
                        "Embedding Skips": metrics.get("embedding_skips", 0),
                        "False Positives": metrics.get("false_positives", 0),
                        "Avg Detection Confidence": f"{metrics.get('avg_detection_confidence', 0):.3f}",
                        "Validator Rejection Rate": f"{metrics.get('validator_rejection_rate', 0):.1%}",
                        "Timestamp": snap.get("timestamp", "N/A"),
                    })
                
                st.dataframe(comparison_rows, use_container_width=True)
    
    with comparison_subtabs[1]:
        show_action_log = st.checkbox("Show Action Log", value=False)
        
        if show_action_log:
            # Load action log entries
            actions_result = client.get_action_log(limit=100)
            
            if "error" in actions_result:
                st.error(f"Failed to load action log: {actions_result['error']}")
            else:
                actions = actions_result.get("actions", [])
                
                if not actions:
                    st.info("No action log entries yet. Actions will be logged when configuration changes occur.")
                else:
                    # Build action log table
                    action_rows = []
                    for action in actions:
                        action_rows.append({
                            "Timestamp": action.get("timestamp", "N/A"),
                            "Actor": action.get("actor", "N/A"),
                            "Event Type": action.get("event_type", "N/A"),
                            "Metadata": str(action.get("metadata", {}))[:100],
                        })
                    
                    st.dataframe(action_rows, use_container_width=True)
        else:
            st.caption("Enable checkbox to view action log.")
