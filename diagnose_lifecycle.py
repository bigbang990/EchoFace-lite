
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
import time

def diagnose_lifecycle():
    print("\n=== DIAGNOSIS: Track Lifecycle Timing relationship ===")
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
    t1 = SimulatedObject("Target_1", true_id=1, x=100, y=100, w=100, h=100)
    world.add_object(t1)
    
    pipeline = RecognitionPipeline(
        settings=settings,
        detector=MockDetector(world),
        embedder=MockEmbedder(world),
        matcher=MockMatcher()
    )
    
    state_history = []
    
    def tracker_audit(frame_index, p, matches):
        tm = p._track_manager
        tracks = tm._tracks
        # Only log every 10 frames to keep output clean, plus transitions
        if not tracks:
            state_history.append((frame_index, "NO_TRACKS", 0, 0, 0, p._dynamic_detector_interval, p._current_pressure_band))
            return
            
        track = tracks.get("track_1")
        if track:
            state = track.state
            last_state = state_history[-1][1] if state_history else None
            if frame_index % 10 == 0 or state != last_state:
                state_history.append((
                    frame_index, 
                    state, 
                    track.confirmation_hits, 
                    track.lost_frames, 
                    track.recovery_grace_frames,
                    p._dynamic_detector_interval,
                    p._current_pressure_band
                ))
        else:
            state_history.append((frame_index, "TRACK_1_GONE", 0, 0, 0, p._dynamic_detector_interval, p._current_pressure_band))

    print("--- Step 1: Establishment ---")
    run_simulation(pipeline, world, steps=50, callback=tracker_audit)
    
    print("--- Step 2: Occlusion & Pressure ---")
    for i in range(60):
        gx = 400 + (i % 8) * 150
        gy = 300 + (i // 8) * 120
        world.add_object(SimulatedObject(f"BG_{i}", true_id=100+i, x=gx, y=gy, w=80, h=80, det_score_base=0.85))
    
    t1.occlusion = 1.0
    run_simulation(pipeline, world, steps=160, callback=tracker_audit)
    
    print("--- Step 3: Re-entry ---")
    t1.occlusion = 0.0
    t1.x += 10
    # Run for 50 steps to see if it eventually promotes
    run_simulation(pipeline, world, steps=50, callback=tracker_audit)
    
    print("\nState Evolution:")
    print("Frame | State | Hits | Lost | Grace | DetInterval | PressureBand")
    print("-" * 70)
    for entry in state_history:
        print(f"{entry[0]:5} | {entry[1]:10} | {entry[2]:4.1f} | {entry[3]:4} | {entry[4]:5} | {entry[5]:11} | {entry[6]:12}")

    # Determine association updates
    tm = pipeline._track_manager
    track = tm._tracks.get("track_1")
    if track:
        print(f"\nFinal Track 1 Stats:")
        print(f"State: {track.state}")
        print(f"Confirmation Hits: {track.confirmation_hits}")
        print(f"Visibility Age: {track.visibility_age}")
        print(f"Lifetime MS: {track.lifetime_ms:.1f}")
        print(f"Time Since Last Seen MS: {track.time_since_last_seen_ms:.1f}")
        print(f"Coarse Survival MS: {settings.coarse_track_survival_ms}")
        print(f"Is skip-matched: {track.time_since_last_seen_ms > settings.coarse_track_survival_ms}")
        print(f"Recovery Count: {track.recovery_count}")

    snapshot = metrics.snapshot()
    print(f"\nGlobal Metrics:")
    print(f"Detection Cycles: {snapshot.counters.get('detection_cycles', 0)}")
    print(f"Recovered Tracks: {snapshot.counters.get('recovered_tracks', 0)}")
    print(f"New Tracks Created: {snapshot.counters.get('new_tracks_created', 0)}")
    print(f"Total Faces Detected: {snapshot.counters.get('total_faces_detected', 0)}")

if __name__ == "__main__":
    diagnose_lifecycle()
