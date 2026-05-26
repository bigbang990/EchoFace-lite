from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np

from ecoface_lite.ai_engine.tracking.track_manager import FaceTrackManager
from ecoface_lite.ai_engine.detector import DetectedFace, BoundingBox
from identity_stress_suite.metrics.engine import ContinuityMetricsEngine
from identity_stress_suite.replay.system import DeterministicReplay, ScenarioGenerator, RealVideoReplay
from identity_stress_suite.reports.reporter import BenchmarkReporter
from identity_stress_suite.overlays.engine import VisualOverlayEngine
from identity_stress_suite.logs.logger import FrameEventLogger

class StressSuiteRunner:
    """
    Runner for the Identity Stress Suite with Visual Continuity Overlay.
    Exposes truth via visual inspection and automated trend analysis.
    """
    
    def __init__(self, settings: Any = None):
        from ecoface_lite.core.config import get_settings
        self.settings = settings or get_settings()
        self.metrics_dir = Path("identity_stress_suite/metrics")
        self.reports_dir = Path("identity_stress_suite/reports")
        self.export_dir = Path("identity_stress_suite/overlay_exports")
        self.log_dir = Path("identity_stress_suite/logs")
        self.video_dir = Path("data/validation_videos")
        
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.reporter = BenchmarkReporter(self.reports_dir)
        self.visualizer = VisualOverlayEngine()
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy load and optimize production pipeline for CPU validation."""
        if self._pipeline is None:
            from ecoface_lite.ai_engine.bootstrap import build_recognition_pipeline
            # --- PHASE 2C.3B PERFORMANCE OVERRIDE ---
            # Drastically reduce resolution for CPU-only validation speed
            res = 160
            self.settings.detector_input_width = res
            self.settings.detector_input_height = res
            self.settings.detector_medium_width = res
            self.settings.detector_medium_height = res
            self.settings.detector_large_width = res
            self.settings.detector_large_height = res
            
            # Force resolution cap
            self.settings.detector_resolution_cap_enabled = True
            self.settings.detector_max_input_pixels = res * res
            
            # Skip detections more often - we are validating CONTINUITY/TRACKING
            self.settings.detector_interval_frames = 15
            
            # Aggressive candidate expiration to prevent "Confirmation Pending Saturation"
            self.settings.validator_track_only_max_age_frames = 10
            self.settings.validator_weak_pass_max_age_frames = 20
            
            # Increase search window for low-FPS CPU processing
            self.settings.motion_max_frame_displacement_px = 500.0 
            
            # Disable heavy redundant legacy checks
            self.settings.enable_legacy_face_validation = False
            self.settings.enable_legacy_quality_checks = False
            
            self._pipeline = build_recognition_pipeline(self.settings)
        return self._pipeline

    def _to_detected_face(self, d: Dict[str, Any]) -> DetectedFace:
        bbox = BoundingBox(x1=d["bbox"][0], y1=d["bbox"][1], x2=d["bbox"][2], y2=d["bbox"][3])
        return DetectedFace(bbox=bbox, det_score=d["score"])

    def run_real_video(self, scenario_name: str, video_path: Path, export_video: bool = True) -> Dict[str, Any]:
        """Execute a real video scenario through the production pipeline."""
        pipeline = self._get_pipeline()
        
        # Reset tracker state for new video
        pipeline._track_manager._tracks = {}
        pipeline._track_manager._pending = []
        pipeline._track_manager._next_track_id = 1
        
        print(f"--- Processing Real Video: {scenario_name} ---")
        engine = ContinuityMetricsEngine()
        logger = FrameEventLogger(self.log_dir, scenario_name)
        
        replay = RealVideoReplay(scenario_name, video_path)
        frames_to_export = []
        
        print(f"Starting optimized analysis pass for {scenario_name} (sampling 1/21 frames)...")
        
        # Aggressive sampling: only process every 21st frame (approx 1 sample per 0.7s at 30fps)
        for frame_idx, timestamp_ms, frame_bgr in replay.frame_iterator(skip_frames=20):
            # Process through production pipeline
            _ = pipeline.process_frame(frame_bgr, frame_idx, gallery=[])
            
            active_tracks = []
            # Look at ALL tracks in the manager for deeper drift analysis
            for track_id, track in pipeline._track_manager._tracks.items():
                # Only log tracks that are currently visible or recently lost
                if not track.is_active and track.state != "LOST": 
                    continue
                
                x1, y1, x2, y2 = track.bbox
                w, h = x2 - x1, y2 - y1
                vx, vy = track.metadata.get("velocity", (0.0, 0.0))
                
                # Log canonical schema
                event = {
                    "frame": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "track_id": track.numeric_track_id,
                    "state": track.state,
                    "confidence": float(track.confidence),
                    "detector_confidence": float(track.metadata.get("detector_confidence", track.confidence)),
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "bbox_delta": float(track.metadata.get("last_bbox_delta", 0.0)),
                    "velocity": [float(vx), float(vy)],
                    "detector_refresh": True, # Every processed frame is effectively a refresh when skipping
                    "occlusion_duration": int(track.lost_frames),
                    "match_similarity": float(track.identity_confidence),
                    "stable_match": bool(track.is_stable),
                    "identity_locked": bool(track.identity is not None)
                }
                logger.log_event(event)
                
                track_data = {
                    "track_id": track_id,
                    "bbox": track.bbox,
                    "identity": track.identity,
                    "confidence": track.confidence,
                    "state": track.state,
                    "visibility_age": track.visibility_age,
                    "lost_frames": track.lost_frames,
                    "recovery_count": track.recovery_count,
                    "stable_match_count": track.stable_match_count
                }
                active_tracks.append(track_data)
            
            engine.record_frame(frame_idx, active_tracks)
            
            if export_video: 
                annotated = self.visualizer.draw_overlay(frame_bgr, active_tracks)
                frames_to_export.append(annotated)
                
        if export_video and frames_to_export:
            output_name = f"{scenario_name}_real_overlay.mp4"
            self.visualizer.export_video(frames_to_export, str(self.export_dir / output_name))

        metrics = engine.calculate()
        return {
            "name": scenario_name,
            "metrics": metrics.to_dict(),
            "type": "real_video"
        }

    def run_scenario(self, replay: DeterministicReplay, export_video: bool = True) -> Dict[str, Any]:
        """Execute a single scenario and return its results."""
        manager = FaceTrackManager(self.settings)
        engine = ContinuityMetricsEngine()
        
        frames_to_export = []
        
        for frame_data in replay.frames:
            frame_idx = frame_data["frame_index"]
            raw_detections = frame_data["detections"]
            
            # Convert to DetectedFace objects
            faces = [self._to_detected_face(d) for d in raw_detections]
            
            # Update tracker
            matched = manager.update_from_detections(faces, frame_idx)
            
            # Record state for metrics and visualization
            active_tracks = []
            for _, track in matched:
                if track:
                    active_tracks.append({
                        "track_id": track.track_id,
                        "bbox": track.bbox,
                        "identity": track.identity,
                        "confidence": track.confidence,
                        "state": track.state,
                        "visibility_age": track.visibility_age,
                        "lost_frames": track.lost_frames,
                        "recovery_count": track.recovery_count,
                        "stable_match_count": track.stable_match_count
                    })
            
            engine.record_frame(frame_idx, active_tracks)
            
            if export_video:
                # Create a synthetic frame (black background) for overlay
                # 720p canvas
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                annotated = self.visualizer.draw_overlay(canvas, active_tracks)
                frames_to_export.append(annotated)
            
        if export_video and frames_to_export:
            output_name = f"{replay.scenario_name}_overlay.mp4"
            self.visualizer.export_video(frames_to_export, str(self.export_dir / output_name))

        metrics = engine.calculate()
        return {
            "name": replay.scenario_name,
            "metrics": metrics.to_dict()
        }

    def run_all(self, baseline_filename: str = "Phase_2C2_Baseline.json"):
        """Run all standard stress scenarios and real video validation."""
        # 1. Synthetic Scenarios
        scenarios = [
            ScenarioGenerator.create_crossing_scenario(),
            ScenarioGenerator.create_occlusion_scenario(),
            ScenarioGenerator.create_reentry_scenario(),
            ScenarioGenerator.create_crowd_scenario(),
            ScenarioGenerator.create_small_faces_scenario(),
            ScenarioGenerator.create_profile_rotation_scenario(),
            ScenarioGenerator.create_partial_visibility_scenario(),
            ScenarioGenerator.create_same_clothes_scenario()
        ]
        
        results = []
        for scenario in scenarios:
            print(f"Running synthetic scenario: {scenario.scenario_name}...")
            results.append(self.run_scenario(scenario))

        # 2. Real Video Scenarios (Phase 2C.3B)
        real_video_files = [
            ("crowd_crossing", "crowd_crossing.mp4"),
            ("same_clothes", "same_clothes_people.mp4"),
            ("occlusion", "partial_occlusion.mp4"),
            ("re_entry", "reentry_scene.mp4"),
            ("profile_rotation", "profile_rotation.mp4"),
            ("small_faces", "small_faces_distance.mp4"),
            ("low_light", "low_light_walk.mp4"),
            ("camera_motion", "camera_motion.mp4")
        ]

        print("\n--- Starting Real Video Validation Pass ---")
        coverage_report = []
        
        for name, filename in real_video_files:
            video_path = self.video_dir / filename
            if not video_path.exists():
                print(f"Skipping missing video: {filename}")
                coverage_report.append({"name": name, "status": "MISSING"})
                continue
            
            # Rule: DO NOT TRUST THE FIRST VIDEO RESULTS
            # We run it twice to ensure deterministic continuity and detect synchronization bugs
            print(f"\nRunning Validation Pass 1 for {name}...")
            res1 = self.run_real_video(f"{name}_pass1", video_path, export_video=True)
            
            print(f"Running Validation Pass 2 for {name} (Consistency Check)...")
            res2 = self.run_real_video(f"{name}_pass2", video_path, export_video=True)
            
            # Use Pass 2 for the final report
            results.append(res2)
            coverage_report.append({"name": name, "status": "VALIDATED"})

        # Load Baseline
        baseline = None
        baseline_path = self.reports_dir / baseline_filename
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)
        
        # Load Previous (most recent report in dir)
        previous = None
        all_reports = sorted(list(self.reports_dir.glob("report_*.json")))
        if all_reports:
            with open(all_reports[-1], 'r') as f:
                previous = json.load(f)
                
        report_file = self.reporter.generate_report(results, baseline, previous)
        print(f"\nBenchmark complete. Report generated: {report_file}")
        print(f"Video Coverage: {sum(1 for c in coverage_report if c['status'] == 'VALIDATED')}/{len(real_video_files)}")
        
        # Load the generated report to show recommendations
        with open(self.reports_dir / report_file, 'r') as f:
            final_report = json.load(f)
            
        recs = self.reporter.recommend_actions(final_report)
        if recs:
            print("\nRecommended Actions:")
            for r in recs:
                print(f" - {r}")
        
        return results

if __name__ == "__main__":
    runner = StressSuiteRunner()
    runner.run_all()
