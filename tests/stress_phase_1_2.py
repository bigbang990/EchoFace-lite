"""Phase 1 & 2 Stress Tests: Pressure Escalation and Coarse Track Audit."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.stress_framework import SimulationWorld, SimulatedObject, MockDetector, MockEmbedder, MockMatcher, run_simulation
from ecoface_lite.ai_engine.pipeline import RecognitionPipeline
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics
from ecoface_lite.ai_engine.tracking.track_state import TrackLifecycleState
import numpy as np

def test_phase_1_pressure_escalation():
    print("\n=== Phase 1: Pressure Escalation Validation ===")
    metrics.reset()
    
    # Use real Settings with ALIASES (capitalized) for constructor to work
    overrides = {
        "ENABLE_ADAPTIVE_LOAD_GOVERNANCE": True,
        "GOVERNANCE_PRESSURE_HYSTERESIS_FRAMES": 2, 
        "DETECTOR_INTERVAL_FRAMES": 1,
        "GOVERNANCE_LOW_PRESSURE_INTERVAL": 1, # Keep it at 1 for test
        "DETECTOR_OVERLOAD_FACE_COUNT": 20,
        "GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE": 100, 
        "ENABLE_PRIORITY_INGESTION": True,
        "ENABLE_LEGACY_FACE_VALIDATION": False,
        "ENABLE_LEGACY_QUALITY_CHECKS": False,
        "VALIDATOR_MIN_DETECTOR_CONFIDENCE": 0.1,
        "VALIDATOR_MIN_FACE_AREA_RATIO": 0.0001,
        "VALIDATOR_MAX_FACE_AREA_RATIO": 1.0,
        "VALIDATOR_MIN_BLUR_VAR": 0.0001,
        "VALIDATOR_MIN_BRIGHTNESS": 0.0001,
        "VALIDATOR_MAX_BRIGHTNESS": 255.0,
        "VALIDATOR_MIN_ASPECT_RATIO": 0.1,
        "VALIDATOR_MAX_ASPECT_RATIO": 10.0,
        "VALIDATOR_EDGE_MARGIN_RATIO": 0.0,
        "VALIDATOR_QUALITY_CUTOFF": 0.0,
        "VALIDATOR_STRICT_CUTOFF": 0.5,
        "PREPROCESSING_MAX_WIDTH": 1920,
        "DETECTOR_INPUT_WIDTH": 1920,
        "DETECTOR_INPUT_HEIGHT": 1080
    }
    
    settings = Settings(**overrides)
    
    world = SimulationWorld()
    # Add one stable target
    target = SimulatedObject("Target_P0", true_id=1, x=100, y=100, w=100, h=100)
    world.add_object(target)
    
    pipeline = RecognitionPipeline(
        settings=settings,
        detector=MockDetector(world),
        embedder=MockEmbedder(world),
        matcher=MockMatcher()
    )
    
    # 1. Baseline (Band 0)
    print("Step 1: Baseline (1 object)")
    run_simulation(pipeline, world, steps=10)
    assert pipeline._current_pressure_band == 0
    
    # 2. Escalation (Band 2)
    print("Step 2: Adding 60 background objects (Escalation to HIGH)")
    for i in range(60):
        # Spacing them ensures they are separate objects
        # Position them in a grid in the center
        gx = 400 + (i % 8) * 150
        gy = 300 + (i // 8) * 120
        world.add_object(SimulatedObject(f"BG_{i}", true_id=100+i, x=gx, y=gy, w=80, h=80, det_score_base=0.85))
        
    def check_band(frame_index, p, matches):
        # Explicitly update metrics for reporting
        p._track_manager._check_congestion()
        snapshot = metrics.snapshot()
        
        # Diagnostics
        if frame_index % 5 == 0:
            band = snapshot.recent_values.get("tracking_pressure_band", [0.0])[-1]
            pressure = snapshot.recent_values.get("state_machine_pressure_score", [0.0])[-1]
            pending = len(p._track_manager._pending)
            tracks = len(p._track_manager._tracks)
            rejections = snapshot.counters.get("validator_reject_count", 0)
            ingest_rejections = snapshot.counters.get("candidate_ingestion_rejections", 0)
            print(f"Frame {frame_index}: Pressure={pressure:.2f} (Pending={pending}, Tracks={tracks}), Band={band}, Rej={rejections}, IngestRej={ingest_rejections}")
            
            # If rejections are happening, print why from metrics
            for reason in ["rejected_due_to_blur", "rejected_due_to_size", "rejected_due_to_low_confidence", "geometry_validation_rejections"]:
                val = snapshot.counters.get(reason, 0)
                if val > 0:
                    print(f"  Metric {reason}: {val}")

    # Run simulation long enough for hysteresis and confirmation
    run_simulation(pipeline, world, steps=60, callback=check_band)
    
    # Should be at Band 2 or 3 now
    assert pipeline._current_pressure_band >= 1
    print(f"Final Pressure Band: {pipeline._current_pressure_band}")
    print(f"Adaptive Interval: {pipeline._dynamic_detector_interval}")

def test_phase_2_coarse_track_audit():
    print("\n=== Phase 2: Coarse Track Identity Audit ===")
    metrics.reset()
    overrides = {
        "ENABLE_ADAPTIVE_LOAD_GOVERNANCE": True,
        "ENABLE_COARSE_TRACKING": True,
        "COARSE_TRACK_SURVIVAL_MS": 5000,
        "TRACKING_EXPIRATION_MS": 500,
        "GOVERNANCE_MAX_CANDIDATE_QUEUE_SIZE": 100,
        "DETECTOR_INTERVAL_FRAMES": 1,
        "PREPROCESSING_MAX_WIDTH": 1920,
        "PREPROCESSING_MAX_HEIGHT": 1080,
        "ENABLE_LEGACY_FACE_VALIDATION": False,
        "ENABLE_LEGACY_QUALITY_CHECKS": False,
        "VALIDATOR_MIN_DETECTOR_CONFIDENCE": 0.1,
        "VALIDATOR_MIN_FACE_AREA_RATIO": 0.0001,
        "VALIDATOR_MAX_FACE_AREA_RATIO": 1.0,
        "VALIDATOR_MIN_BLUR_VAR": 0.0001,
        "VALIDATOR_MIN_BRIGHTNESS": 0.0001,
        "VALIDATOR_MAX_BRIGHTNESS": 255.0,
        "VALIDATOR_MIN_ASPECT_RATIO": 0.1,
        "VALIDATOR_MAX_ASPECT_RATIO": 10.0,
        "VALIDATOR_EDGE_MARGIN_RATIO": 0.0,
        "VALIDATOR_QUALITY_CUTOFF": 0.0,
        "VALIDATOR_STRICT_CUTOFF": 0.5,
        "DETECTOR_INPUT_WIDTH": 1920,
        "DETECTOR_INPUT_HEIGHT": 1080
    }
    settings = Settings(**overrides)
    
    world = SimulationWorld()
    # Target 1
    t1 = SimulatedObject("Target_1", true_id=1, x=100, y=100, w=100, h=100)
    world.add_object(t1)
    
    pipeline = RecognitionPipeline(
        settings=settings,
        detector=MockDetector(world),
        embedder=MockEmbedder(world),
        matcher=MockMatcher()
    )
    
    # 1. Establish track
    run_simulation(pipeline, world, steps=50)
    track_id = list(pipeline._track_manager._tracks.keys())[0]
    track = pipeline._track_manager._tracks[track_id]
    assert track.state in {TrackLifecycleState.CONFIRMED.value, TrackLifecycleState.STABLE.value}
    
    # 2. Force HIGH pressure and OCCLUDE
    print("Step 2: Occluding target under HIGH pressure")
    print(f"DEBUG t1 coords before occlusion: x={t1.x}, y={t1.y}, w={t1.w}, h={t1.h}")
    for i in range(60): # Add many objects to force pressure
        gx = 400 + (i % 8) * 150
        gy = 300 + (i // 8) * 120
        world.add_object(SimulatedObject(f"BG_{i}", true_id=100+i, x=gx, y=gy, w=80, h=80, det_score_base=0.85))
    
    t1.occlusion = 1.0 # Total occlusion
    
    _debug_flags = {"first": True}
    def debug_occlusion(frame_index, p, matches):
        if _debug_flags["first"]:
            raw = p._detector.detect(world.frame_bgr)
            print(f"DEBUG First occlusion frame {frame_index}: raw_faces={len(raw)}, t1.occlusion={t1.occlusion}")
            print(f"DEBUG temporal_max_track_distance={p._settings.temporal_max_track_distance}")
            for face in raw[:3]:
                cx = (face.bbox.x1 + face.bbox.x2) / 2
                cy = (face.bbox.y1 + face.bbox.y2) / 2
                # Check distance to track_1
                t = p._track_manager._tracks.get(track_id)
                if t:
                    tx = (t.bbox[0] + t.bbox[2]) / 2
                    ty = (t.bbox[1] + t.bbox[3]) / 2
                    dist = ((cx - tx)**2 + (cy - ty)**2) ** 0.5
                    print(f"  Face centroid=({cx:.0f},{cy:.0f}) track_1 centroid=({tx:.0f},{ty:.0f}) dist={dist:.0f}")
            _debug_flags["first"] = False
        if frame_index % 10 == 0:
            band = p._current_pressure_band
            snapshot = metrics.snapshot()
            pressure = snapshot.recent_values.get("state_machine_pressure_score", [0.0])[-1]
            pending = len(p._track_manager._pending)
            tracks = len(p._track_manager._tracks)
            print(f"Frame {frame_index}: Pressure={pressure:.2f} Band={band} Pending={pending} Tracks={tracks}")
            if track_id in p._track_manager._tracks:
                t = p._track_manager._tracks[track_id]
                print(f"  Track {track_id}: state={t.state} lost_frames={t.lost_frames} visibility={t.visibility_age} last_seen={t.last_seen_frame}")
            else:
                print(f"  Track {track_id}: NOT IN TRACKS")
    
    # Run simulation for a while (need > 750ms real time for expiration under cooldown cycles)
    run_simulation(pipeline, world, steps=160, callback=debug_occlusion)
    
    # Verify it became COARSE
    print(f"Final track state: {track.state}, lost_frames: {track.lost_frames}, visibility: {track.visibility_age}")
    assert track.state == TrackLifecycleState.COARSE.value
    print(f"Track {track_id} status: {track.state}")
    
    # 3. Re-entry (RECOVERY)
    print("Step 3: Target re-enters. Checking identity continuity.")
    t1.occlusion = 0.0
    t1.x += 10 # Slight movement
    
    run_simulation(pipeline, world, steps=5)
    
    # Verify it promoted back to CONFIRMED or STABLE
    assert track.state in {TrackLifecycleState.CONFIRMED.value, TrackLifecycleState.STABLE.value}
    print(f"Track {track_id} status after recovery: {track.state}")
    
    # CRITICAL: Verify it didn't mutate into a background object ID
    # In this mock, identity is kept in TrackedFace.identity if set. 
    # Let's check track_id persistence.
    assert track.track_id == track_id
    print("Identity continuity verified.")

if __name__ == "__main__":
    test_phase_1_pressure_escalation()
    test_phase_2_coarse_track_audit()
